# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SPADE: Self-Play in Adaptive Synthetic Executable Environments](https://arxiv.org/abs/2608.19197) | 该论文提出SPADE框架，通过单个LLM同时作为环境设计器和推理代理进行自我对弈，动态生成可执行训练环境，以解决语言代理训练中目标分布固定的问题。 |
| [^2] | [ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning](https://arxiv.org/abs/2608.19182) | ADEPT通过预训练通用灵巧操作策略并采用稳定的后训练方案，显著加速了高自由度机器人从原始感知学习长时程任务的过程，避免了重复学习并提升了迁移能力。 |
| [^3] | [Beyond Teacher Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning](https://arxiv.org/abs/2608.19181) | 本文提出GC-OPD方法，通过在rollout组内校准验证器奖励与轨迹级OPD分数之间的差异，解决长上下文推理中教师指导与任务验证不一致的问题。 |
| [^4] | [Finetuning Strategies for Querying Sounds by Vocal Imitation](https://arxiv.org/abs/2608.19174) | 本文提出并验证了两种微调策略（冻结CED编码器的对比学习和MobileNetV3的联合对比-三元组学习），在声音模仿查询音效任务中取得了冠军成绩。 |
| [^5] | [Interpretable AI predicts a 2026 summer dry anomaly in central China](https://arxiv.org/abs/2608.19163) | 该研究利用深度学习模型将动力环流预测转化为降水估计，并借助可解释性分析（LRP）揭示了2026年中国中部夏季干旱异常及其背后的环流驱动机制，实现了对区域降水异常的高技能季节预测。 |
| [^6] | [Beyond the Transcript: Detecting Covert Co ordination in Latent Multi-Agent Communication](https://arxiv.org/abs/2608.19161) | 本文提出了VLA框架，通过激活感知监控和反事实分析，首次实现了对多智能体潜在通信渠道中隐蔽有害协调的检测与引导。 |
| [^7] | [Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets](https://arxiv.org/abs/2608.19147) | 通过将大语言模型按层预编译为OpenVINO分片并利用流水线并行，在多个英特尔AI PC上实现分布式推理，通过注入beam_idx Gather触发GPU优化和投机解码，达到与单体推理相当的性能。 |
| [^8] | [Grouping the Stochastic Machine: Precision, Not Capability, as the Frontier Metric for AI Systems](https://arxiv.org/abs/2608.19140) | 本文提出前沿AI系统的关键区别在于输出精度（重复请求下结果的一致性），而非传统能力指标，并论证了精度可低成本、无循环地测量。 |
| [^9] | [Leaf Values as Coordinates: Exact Contrastive Explanation for Gradient-Boosted Ensembles](https://arxiv.org/abs/2608.19127) | 通过将梯度提升模型的叶值视为坐标，实现了精确且可追溯的对比解释，为反事实分析提供了无需额外拟合的严谨方法。 |
| [^10] | [Tuning the Stochastic Machine: A Systems Engineer's Operating Model for Human-AI Engineering](https://arxiv.org/abs/2608.19125) | 本文提出将LLM系统视为需要系统工程师操作纪律的随机机器，并基于映射失败推导出七项原则，核心是持久化错误纠正的循环机制。 |
| [^11] | [Intercepting the Kangaroo: Experimental Astrolinguistics with Constructed Lexicons, Active Probing, and Large Language Models as Informants and Hypothesis Proposers](https://arxiv.org/abs/2608.19124) | 该论文通过构造不兼容词典和主动探测协议，首次将天体语言学实验化，成功拦截了“袋鼠效应”（翻译不确定性），实现了零未检测误译。 |
| [^12] | [PGFS++: Molecular Property Improvement under Synthesis and Diversity Constraints](https://arxiv.org/abs/2608.19121) | 本文提出PGFS++，通过直接嵌入反应物和引入多样性约束，在保持合成可行性的同时改进分子性质，并解决了PGFS+中因奖励黑客导致的输出多样性崩溃问题。 |
| [^13] | [Discretizing Continuous Time Series for Imputation with Masked Diffusion Training](https://arxiv.org/abs/2608.19119) | 本文提出MDTIM模型，通过将时间序列离散化并采用掩码扩散训练，实现结构分离的插补，直接预测原始信号，克服了传统方法在表示空间和学习目标上的局限性。 |
| [^14] | [Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2608.19098) | 本文通过受控基准诊断出多教师同策略蒸馏中的能力整合失衡问题，指出其根源并非梯度冲突，而是严重的优化失衡，并提出修复方法以提升整合效率。 |
| [^15] | [Detecting Backdoors in Object Detection via Pre-NMS Prediction Distribution Shift](https://arxiv.org/abs/2608.19088) | 本文提出DistScan框架，通过检测模型在干净输入上的预NMS预测类别分布偏移，实现无需触发器反转即可可靠识别目标检测模型中的后门，尤其适用于场景级攻击。 |
| [^16] | [DA-WAM: Decision-Aligned Future Latents for Driving World Models](https://arxiv.org/abs/2608.19085) | DA-WAM通过统一预测表示学习、行动条件未来建模和轨迹评分，在单一决策目标下实现未来表示与规划优化的共同演进，从而确保预测未来直接指导轨迹选择。 |
| [^17] | [ReWEIGH the Evidence: Calibrating Token-Level Ordinal Visual Evidence to Mitigate Hallucinations in Large Vision-Language Models](https://arxiv.org/abs/2608.19075) | ReWEIGH通过校准词元级序数视觉证据，无需训练即可有效缓解大型视觉-语言模型中的幻觉现象。 |
| [^18] | [Robust Risk Under Evolving Uncertainty: A Wasserstein Counterpart of the Entropic Value-at-Risk](https://arxiv.org/abs/2608.19073) | 本文提出水斯坦熵值风险价值，用最优传输球替代相对熵球，以覆盖熵值风险价值忽略的可达灾难，并推导出闭式稳健动态规划算子，其谨慎度随信念更新而自适应调整。 |
| [^19] | [What is Missing from AI Post-Training AI: An Empirical Analysis](https://arxiv.org/abs/2608.19072) | 本文通过实证分析发现，AI后训练代理在策略层面缺乏灵活性，其训练策略在初期被锁定，且主要受经验缺失、指导缺失和推理不足影响，而非单纯执行能力不足。 |
| [^20] | [GS-VLA: Plug-and-Play Viewpoint Canonicalization for Frozen VLA Policies via Gaussian Splatting](https://arxiv.org/abs/2608.19066) | 本文提出一种基于3D高斯泼溅的即插即用框架，无需重新训练VLA策略即可有效应对视角偏移，显著提升部署鲁棒性。 |
| [^21] | [Eureka: Task-Conditioned Meta-Agent Orchestration for Scientific Discovery](https://arxiv.org/abs/2608.19047) | 尤里卡提出了一种任务条件化的元代理架构，通过动态义务图和宏代理编排，实现了高效、可验证的长期任务处理，显著降低了计算开销并保证了无误差的递归任务完成。 |
| [^22] | [Bernstein-Vazirani Networks: Quantum Machine Learning by Interference](https://arxiv.org/abs/2608.19043) | BVNs通过量子干涉和傅里叶采样实现无梯度训练的非变分量子机器学习，在分类和表示任务上达到与经典和量子方法相当的竞争力。 |
| [^23] | [Counterfactual Contrastive Analysis](https://arxiv.org/abs/2608.19032) | 本文提出了一种基于对比分析的无分类器视觉反事实生成方法，通过分离和交换数据分布中的显著因素，生成模型无关且对分类器偏见不敏感的反事实解释。 |
| [^24] | [Adaptive Memory and Reflection Multi-Agent System for Medical Question Answering](https://arxiv.org/abs/2608.19029) | 本文提出了一种自适应记忆与反思多智能体医学问答系统，通过专用记忆、反思反馈和动态工作流路由，在MedQA和MedMCQA上实现了优于基线的性能。 |
| [^25] | [Self-prompting and cross-model consensus enable reproducible data extraction from scientific literature with large language models](https://arxiv.org/abs/2608.19025) | 本研究通过四种递进工作流程评估了大型语言模型在科学文献数据提取中的表现，发现自我提示接近专家提示效果，但自主文献发现仍不可靠，且生成数据集需人类参与，从而提出专家制定证据标准、模型执行提取的可审计分工模式。 |
| [^26] | [One-Stage Object Detectors in Autonomous Driving](https://arxiv.org/abs/2608.19014) | 本综述系统分析了自动驾驶中单阶段目标检测器的演进与设计权衡，强调了其在速度、准确性、效率和鲁棒性之间的平衡策略。 |
| [^27] | [Harness Continual Learning: Continual Adaptation Beyond Model Parameters](https://arxiv.org/abs/2608.19013) | 本文提出驾驭持续学习（HCL），一种在冻结基础模型外部演化提示、记忆等驾驭组件的新范式，并通过受保护的演化机制减少对早期行为的遗忘。 |
| [^28] | [From Threat Intelligence to Detection: Knowledge-driven Enrichment and Template-based Rule Grounding for Automated Sigma Rule Generation](https://arxiv.org/abs/2608.19011) | 本文提出AUTOSIGMA，一种自动化生成Sigma规则的方法，通过知识驱动的丰富和基于模板的规则接地，解决手动规则编写易错且扩展性差的问题，以动态适应新兴威胁和特定环境。 |
| [^29] | [A Theory of Post-hoc Debate Judgement](https://arxiv.org/abs/2608.19002) | 本文提出了一种适用于智能体辩论场景的事后评判理论，定义了可重复性、稳健性、扎根性和可解释性等属性，并验证了其在LLM评判方法中的实现。 |
| [^30] | [GrabVG: Graph-Attentive Binding for Visual Grounding in UAV Imagery](https://arxiv.org/abs/2608.18996) | 本文提出GrabVG框架，通过模拟人类视觉搜索的两阶段机制（预注意假设搜索和图注意力特征绑定），有效解决了无人机影像中高拥挤场景下小目标视觉冗余和拓扑模糊导致的定位不准确问题。 |
| [^31] | [DeepWeaver: Bridging the Evidence Synthesis Gap in Open-Ended Question Answering](https://arxiv.org/abs/2608.18988) | DeepWeaver通过引入思想块链（TBCs）这一结构化表示，弥合了开放式问答中检索与生成之间的证据综合差距，从而生成更全面且引用准确的答案。 |
| [^32] | [rEDMRec: Distilling Large Language Model Reasoning into an Editable Experience Memory for Recommendation](https://arxiv.org/abs/2608.18952) | 本文提出rEDMRec，通过将大语言模型的推理蒸馏为四种可编辑的经验记忆通道，使轻量级模型能复用推理结果，从而避免每次推荐请求时重复生成推理，同时支持用户口味变化时的检查与修正。 |
| [^33] | [AlphaClifford: Efficient Clifford Synthesis and Transpilation with Model-based RL](https://arxiv.org/abs/2608.18946) | 本文提出了AlphaClifford，一种基于蒙特卡洛树搜索和模型强化学习的框架，利用辛群代数性质高效综合克利福德电路，显著降低了总门数和CNOT门数。 |
| [^34] | [Training Chemical Plausibility-Aware Large Language Models for Single-Step Retrosynthesis](https://arxiv.org/abs/2608.18940) | 本文提出Top-K提示训练范式，结合化学约束和奖励机制，构建了大规模数据集和C3LM模型，显著提升了单步逆合成中多样化且合理反应的预测性能。 |
| [^35] | [Breaking the weakest link to evade vision language models](https://arxiv.org/abs/2608.18938) | 本文提出一种仅针对视觉编码器优化的梯度攻击方法，有效生成对抗样本，以规避视觉语言模型的多模态对齐，并涵盖非定向与定向攻击场景。 |
| [^36] | [MedUAG: Unified Understanding and Generation for Medical Multimodal Models](https://arxiv.org/abs/2608.18937) | 本文提出了MedUAG，通过构建最大的医学统一理解与生成数据集（MedUAGCorpus）和系统化基准（MedUAGBench），开发了一个端到端训练的统一医学多模态模型，显著提升了医学领域的理解与生成能力。 |
| [^37] | [Graphical Design of Interpretable Architectures](https://arxiv.org/abs/2608.18936) | 本文提出了一种基于Penrose张量符号的图形化表示方法，用于设计可解释AI架构，该符号提供全局视图并与PyTorch einsum代码一一对应，并应用于描述多种可解释模型及前沿模型Steerling-8B。 |
| [^38] | [SkillForge: Self-Distilling Agents for Project-Specific Issue Resolution](https://arxiv.org/abs/2608.18933) | SkillForge通过自蒸馏框架主动合成项目特定问题，将可复用的项目知识提炼为技能，从而无需依赖历史修复信号或高测试成本即可提升代理在特定代码库中的问题解决能力。 |
| [^39] | [Test-Time Scaling in the Wild: Why Exploitation, Not Exploration, Is the Bottleneck](https://arxiv.org/abs/2608.18931) | 本文发现测试时扩展的瓶颈不在于探索（增加候选多样性），而在于利用（从候选池中选出最佳结果），后者在开放式任务中表现不佳。 |
| [^40] | [SMTrap: Cost-Effective DoS Attacks Against Large Reasoning Models via SMT Conflict Guidance](https://arxiv.org/abs/2608.18921) | 本文提出一种无需模型反馈的DoS攻击新范式，利用SMT求解器的冲突计数作为低成本信号，引导生成推理密集型CSP实例，从而大幅延长大型推理模型的输出轨迹并实现高效攻击。 |
| [^41] | [Learning-State-Aware Dynamic Generative Data Augmentation on Small-Scale Datasets](https://arxiv.org/abs/2608.18907) | 本文提出了一种感知学习状态的动态生成式数据增强方法，通过基于样本损失和损失下降率自适应调整增强强度，并采用解耦增强与扩散融合，有效解决了小规模数据集上生成式增强的适应性、区域适配及多样性-语义平衡问题。 |
| [^42] | [\textsc{TestifAI}: Tomography-Based Testing for Deep Learning Systems](https://arxiv.org/abs/2608.18900) | 本文提出了TestifAI框架，通过层析成像方法系统探索和总结深度学习系统在组合扰动下的鲁棒性，实现了高效准确的鲁棒性估计。 |
| [^43] | [Syntactic Simplification of OWL Class Expressions](https://arxiv.org/abs/2608.18899) | 本文提出了一种名为CES的算法，用于在保持语义不变的前提下，通过重写规则简化OWL类表达式，从而降低复杂度并提高推理效率。 |
| [^44] | [Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models](https://arxiv.org/abs/2608.18884) | 本文提出一种免训练的推理时自反思协议，通过成本受限的早期停止机制，在冻结LLM上实现高效自我验证，无需梯度更新即可提升推理性能。 |
| [^45] | [DentAgent: Evidence-Centric Multi-Agent Coordination for Multimodal Dental Reasoning](https://arxiv.org/abs/2608.18878) | 本文提出DentAgent，一个以证据为中心的多智能体框架，通过协调五个模态专用智能体并将观察转换为结构化证据记录，实现可追溯的多模态牙科推理。 |
| [^46] | [SkillGate: Training In-Policy Skill Selection in Long-Horizon Agents](https://arxiv.org/abs/2608.18852) | 本文提出并识别了长周期智能体中技能选择训练的结构性难题“选择器信用匮乏”，指出传统结果奖励强化学习无法有效训练这一关键决策。 |
| [^47] | [ORBITER: Conflict-Aware Decision-Making for Agentic Last-Mile Delivery](https://arxiv.org/abs/2608.18846) | ORBITER通过引入决策点和冲突感知机制，利用大语言模型显式推理配送中的时空与行为线索，从而提升末端配送中下一步订单决策的可靠性和可解释性。 |
| [^48] | [Verifiable abstention makes AI leak diagnosis accountable in water distribution networks](https://arxiv.org/abs/2608.18836) | 本文提出一种基于可验证弃权的AI泄漏定位框架，通过物理执行代理和LLM审计监督代理的协作，在不行动时明确弃权，从而在保持高决策精度的同时显著提升系统问责性。 |
| [^49] | [MLREF: Efficient Module Reuse for Reward Design in Reinforcement Learning via Large Language Models](https://arxiv.org/abs/2608.18827) | 本文提出MLREF框架，通过持久化模块池实现奖励组件的积累、改进和复用，将奖励函数构建为模块的线性组合，以解决现有方法中奖励函数作为整体程序导致的性能不稳定问题。 |
| [^50] | [Understanding Multilingual Medical ASR Adaptation Through Layer-Wise Analysis](https://arxiv.org/abs/2608.18825) | 本文通过分层编码器分析揭示了多语言医学微调如何重塑Whisper内部表征，并发现最佳模型选择依赖于适应性调整设置，其中Whisper-Medium在直接多语言训练中表现最优。 |
| [^51] | [Identifying Implicit Premises for Logical Reconstruction of Argument Graphs](https://arxiv.org/abs/2608.18821) | 本文提出了一种神经符号流水线，利用大型语言模型生成隐含前提并转化为逻辑公式，以逻辑重构论证图中的蕴含或矛盾关系。 |
| [^52] | [Pairwise Logical Selection of Enthymeme Completions under Semantic-Link Uncertainty](https://arxiv.org/abs/2608.18820) | 本文提出PWAL方法，通过可能世界语义链接形式化在不确定条件下边缘化逻辑阻力，实现省略三段论缺失成分的成对逻辑选择，优于现有方法。 |
| [^53] | [Do Large Language Models Hallucinate Electric Fata Morganas?](https://arxiv.org/abs/2608.18816) | 本文提出大型语言模型的幻觉不仅是工程缺陷，还具哲学意义，并通过实验证明温度参数影响幻觉与创造力表现。 |
| [^54] | [A strengthening of the MCFL-ness of $O_2$](https://arxiv.org/abs/2608.18813) | 本文通过加强字符串元组分解的特征刻画，为 $O_2$ 是多重上下文无关文法提供了更强的新证明。 |
| [^55] | [Forgetting, plasticity, and co-observation: a third facet of continual learning](https://arxiv.org/abs/2608.18803) | 本文提出数据共同观察是持续学习中除遗忘和可塑性外的第三个关键因素，并通过实验证明联合观察数据能带来表征优势，从而影响性能。 |
| [^56] | [Decomposing Wrong-Consensus Agreement in LLM Self-Consistency: A GPT-4.1 Case Study](https://arxiv.org/abs/2608.18795) | 本文通过定义并分解一致性指数Gamma，定量揭示了LLM多数投票在难题上失效的原因，并区分了机械成分与偏好残差。 |
| [^57] | [SIDScope: A Diagnostic Resource for Semantic-ID Interfaces in Generative Recommendation](https://arxiv.org/abs/2608.18779) | SIDScope是一个源追踪诊断资源，用于评估生成式推荐中语义ID接口的健康状况，核心发现是接口健康需多信号评估，且前缀对齐的有效性取决于检索机制是否依赖前缀。 |
| [^58] | [Beyond Predictive Fairness: Quantifying Attribution Consistency Across Demographic Groups in Diabetic Retinopathy Screening](https://arxiv.org/abs/2608.18759) | 本文提出解释一致性评分（ECS），发现预测公平性与解释一致性是模型行为的互补维度，并强调公平性评估应超越传统预测性能。 |
| [^59] | [Epistemic Subordination: Generative AI and the Infrastructure of Knowledge](https://arxiv.org/abs/2608.18758) | 生成式AI通过将多数文化认知固化为默认知识基础设施，造成对少数群体认知的结构性从属，且现有法律因只监管下游而无法应对这一统一性伤害。 |
| [^60] | [Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots](https://arxiv.org/abs/2608.18744) | 本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。 |
| [^61] | [A Multi-Agent Platform for Automated Enterprise Analytics and Insight Generation](https://arxiv.org/abs/2608.18740) | 本文提出一个基于CrewAI的多智能体平台，通过五个顺序协作的AI智能体实现对话式商业智能，在安全架构和参数化机制支持下，显著提升准确率和质量，超越单智能体基线。 |
| [^62] | [Flama: a Python framework for development and deployment of production-ready APIs, machine learning, and LLM services](https://arxiv.org/abs/2608.18733) | Flama是一个基于ASGI的Python框架，通过统一架构和七个子系统简化了生产级API、机器学习模型和LLM应用的开发与部署，实现了类型驱动和异步优先的编程体验。 |
| [^63] | [A Few Cases Are All You Need: An Empirical Study of Annotation-Efficient LoRA Fine-Tuning of MedSAM3](https://arxiv.org/abs/2608.18731) | 本研究实证表明，仅需10个专家标注案例，通过LoRA微调MedSAM3即可在CT和MRI的腹部器官分割中达到临床可用性能，大幅降低标注成本。 |
| [^64] | [Budget-First Tariff Recommendation (BFTR): A Complete Algorithmic Framework for Telecom Plan Recommendation without Overcharging](https://arxiv.org/abs/2608.18723) | BFTR提出了一种完整的算法框架，通过八种预算优先策略（含两种原创混合方法）保证电信套餐推荐不超额收费，并数学上证明了价格偏差为零。 |
| [^65] | [Competence, Not Accuracy: A Diagnostic for Reference-Free Judge Gates in Skill Optimization](https://arxiv.org/abs/2608.18719) | 本文提出一种诊断方法，通过形式化无参考LLM评判器为潜在求解器，推导出其区分正确与错误答案的能力上限，并证明存在必要条件$c > 1/k$，为在技能优化中替换自动验证器提供理论依据。 |
| [^66] | [The Impact of CutMix on Reliability and Robustness in Semantic Segmentation](https://arxiv.org/abs/2608.18715) | 本文系统分析发现CutMix在语义分割中对精度影响轻微，但能持续提升模型可靠性，尤其在不确定性质量上表现突出。 |
| [^67] | [A Critical Synthesis of Uncertainty Quantification and Foundation Models for Semantic Segmentation](https://arxiv.org/abs/2608.18709) | 本文首次系统评估了不确定性量化方法在语义分割基础模型中的应用，并建立了轻量级基线，为提升模型在安全关键任务中的可靠性提供了重要参考。 |
| [^68] | [MemFuse: Multi-Source Memory Fusion from Fragmented Observations](https://arxiv.org/abs/2608.18704) | 本文提出了MemFuseBench基准和MemFuse记忆系统，旨在解决多源碎片化观察中的长期记忆融合问题，通过保留源级证据来增强时间推理和跨源证据整合能力。 |
| [^69] | [Impact of Iterative Fine-Tuning on Transcription Accuracy in Complex Historical Sanskrit Manuscripts](https://arxiv.org/abs/2608.18696) | 本文提出了一种可迭代微调的传统OCR流水线，通过适应目标历史手稿的布局和外观特征，显著提高了复杂梵文手稿的转录准确性，并减少了昂贵的人工标注成本。 |
| [^70] | [Composed Historical Image Retrieval by Modeling Temporal Representations](https://arxiv.org/abs/2608.18694) | 本文提出了一种名为TDIR的表示学习方法，通过正交子空间将历史图像分解为时间和内容成分，在保留时间结构的同时维持检索性能，并提供了数学基础与误差分析。 |
| [^71] | [Europe's Climate Ambition Under Scrutiny: Evidence from Deep Learning Emission Projections](https://arxiv.org/abs/2608.18690) | 深度学习预测显示，欧盟27国按当前趋势将无法实现2030年减排目标，排放缺口达35%，其中交通部门的结构性惯性是主要障碍。 |
| [^72] | [Aslema at NADI 2026: Augmentation through Fewshot for SLU](https://arxiv.org/abs/2608.18689) | 本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。 |
| [^73] | [RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training](https://arxiv.org/abs/2608.18682) | RTPO通过将多轮轨迹组织为反向树并进行回合级更新，解决了多轮RL训练中的稳定性问题，显著提升了性能。 |
| [^74] | [Sanyu Studio: A Multi-Agent System for Art-Historical Narrative Construction](https://arxiv.org/abs/2608.18677) | 本文提出“三余工作室”多智能体系统，证明在历史证据有限时，AI能促进多元艺术史叙事建构并增强人的能动性。 |
| [^75] | [Orienteering Problem with Uncertain Time-Varying Rewards: Framework and Benchmark for Everyday Service Robotics](https://arxiv.org/abs/2608.18672) | 本文提出了一种新的定向运动问题变体，允许奖励不确定且随时间变化，并通过三种规划器和移动机器人基准验证了规划视野与适应性之间的权衡。 |
| [^76] | [Candidate-Fate Accounting for Transparent Sensor Diagnostic Pipeline Search](https://arxiv.org/abs/2608.18665) | 本文提出候选命运核算框架，通过记录搜索过程中每个候选的完整命运（包括无效、剪枝等），增强了传感器诊断流水线搜索的透明性和可审计性。 |
| [^77] | [Change Point--Aware Evaluation and Re-Calibration of PPG-Based Blood Pressure Estimation](https://arxiv.org/abs/2608.18639) | 本文提出了一种基于变化点检测的波动感知评估框架，揭示了现有PPG血压估计模型在血压快速波动期间性能显著下降的问题，并证明周期性再校准能有效缓解这一退化。 |
| [^78] | [Preference Reasoning under Indeterminacy in Large Language Models](https://arxiv.org/abs/2608.18631) | 本文指出大语言模型在偏好推理中无法有效处理不确定性，尤其是认知和结构不确定性，导致其在不同任务中系统性地失败。 |
| [^79] | [CTIFoundry: An Agent-Native Corpus Scaffold for Cyber Threat Intelligence](https://arxiv.org/abs/2608.18613) | 本文提出CTIFoundry，通过构建一个智能体原生的语料脚手架（包括确定性本体图和跨度锚定报告层），解决了LLM智能体在网络威胁情报调查中因语料底层结构缺失而导致的瓶颈问题。 |
| [^80] | [Denoising-Aware Inversion: Revealing Privacy Risks in Noise-Protected Text Embeddings](https://arxiv.org/abs/2608.18610) | 本文首次探讨了噪声保护下文本嵌入反演的攻击场景，指出现有生成方法在仅有噪声嵌入时失效，并揭示了自适应攻击可绕过高斯噪声防御的隐私风险。 |
| [^81] | [Can a Lightweight Multimodal Model Estimate LLM Reasoning Performance? A Study for Compute-Optimal Document Inference](https://arxiv.org/abs/2608.18591) | 本文提出BudgetDoc基准和轻量级DRB模型，通过预测LLM在不同推理预算下的性能，实现动态预算分配，在保持或提升准确性的同时大幅降低成本，并展现出跨模型泛化能力。 |
| [^82] | [OmniHandwritingOCR: A Diagnostic Benchmark for Evaluating Multimodal LLMs in Handwritten OCR Scenarios](https://arxiv.org/abs/2608.18586) | 本文提出了OmniHandwritingOCR，一个覆盖多语言手写、书写者错误和复杂数学表达式的手写OCR诊断基准，通过77.57K张标注图像和难度分层公式语料库，系统评估了多模态大语言模型在手写识别中的鲁棒性。 |
| [^83] | [From Storage to Access: Verifiable Activation of Parametric Knowledge in LLMs via Explicit Priming and Implicit Reasoning](https://arxiv.org/abs/2608.18581) | 本文提出VAKE框架，通过两阶段强化学习实现大语言模型中参数化知识的可验证激活，利用显式提示生成可验证证据并迁移到隐式推理，以区分答案来源并提升事实问答的可靠性。 |
| [^84] | [FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis](https://arxiv.org/abs/2608.18580) | FACET框架通过重构代理技能并修复执行环境，确保终端任务合成中源意图和跨工件一致性得到保留，从而生成可解决且正确评估的任务。 |
| [^85] | [MR-IQA-2: Faithful Image Quality Reflection via Fine-Grained Credit Assignment](https://arxiv.org/abs/2608.18579) | 本文提出MR-IQA-2框架，通过解耦推理与评分的信用分配，并提供可验证监督，以增强盲图像质量评估中推理的忠实性和可靠性。 |
| [^86] | [The Role of Grid Cells in Reducing Spatial Aliasing in Hippocampal Place Representations](https://arxiv.org/abs/2608.18569) | 本研究通过整合网格细胞信号与边界向量细胞驱动的位置细胞，显著减少了海马位置表征中的空间混叠，在三种环境中实现了94%至99%的混叠降低。 |
| [^87] | [MorphoGP: A Nonparametric Framework for Predicting Equilibrium Beach Profiles Under Tidal Influence](https://arxiv.org/abs/2608.18558) | 本文提出MorphoGP，一个结合对比学习分类和高斯过程回归的非参数框架，以自动分类并预测潮汐影响下的平衡海滩剖面，克服传统模型在非线性潮汐环境中的适应性不足。 |
| [^88] | [Performance Drift Detection in Machine Learning as a Service (MLaaS) for IoT Environments](https://arxiv.org/abs/2608.18555) | 本文提出了一种面向物联网环境的MLaaS性能漂移检测框架，通过黑盒提取模型学习服务行为并联合捕获输入变化，解决了动态数据分布下的漂移检测难题。 |
| [^89] | [CentaurBench: Benchmarking LLM Capabilities on Augmenting vs. Automating Real-World Work Tasks](https://arxiv.org/abs/2608.18554) | 本文提出CentaurBench框架，通过比较LLM在自动化和增强模式下的表现，发现两种模式排名关联度低，自动化胜者在增强任务中表现不佳，强调了模型选择需考虑辅助能力。 |
| [^90] | [Bridging Search and CRM: Productionizing AI Product Research Agents for Customer Re-Engagement](https://arxiv.org/abs/2608.18543) | 本文提出了一个生产部署的AI产品研究代理框架，通过整合搜索与CRM系统，利用多代理研究和WhatsApp个性化推荐，显著提升了探索性购买意图用户的重新参与率和点击率。 |
| [^91] | [Evaluating and Explaining Prompt Sensitivity of LLMs Using Interactions](https://arxiv.org/abs/2608.18539) | 本文提出一种基于交互的提示敏感性（IPS）方法，通过分解LLM输出为非线性交互，揭示即使输出不变时提示细微变化也能引发内部不稳定，从而更精确地解释提示敏感性的原因。 |
| [^92] | [FinRCA-Bench: Benchmarking Evidence Retrieval and Reasoning for Financial AI Systems](https://arxiv.org/abs/2608.18534) | FinRCA-Bench是一个合成金融对账基准，通过分离证据检索与推理评估，揭示了金融AI系统在证据获取和推理质量上的独立性能。 |
| [^93] | [Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson](https://arxiv.org/abs/2608.18531) | 本文提出将解释生成与选择分离的离线架构，并通过实证表明成对排序方法（如LambdaRank）在离线解释选择任务中优于单动作强化学习方法（如PPO、GRPO），在无需GPU且延迟低于100毫秒的条件下实现高效推荐系统。 |
| [^94] | [DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents](https://arxiv.org/abs/2608.18524) | DART-SD通过建模交互状态转移图来感知任务中的菱形拓扑结构，并用拓扑引导的局部修正替代全局轨迹强制，从而避免拓扑坍缩并提升多轮工具调用智能体的策略多样性。 |
| [^95] | [Prior-Conditioned Gaussian Discriminants for Generalizable AI-generated Image Detection](https://arxiv.org/abs/2608.18523) | 本文提出了一种基于先验条件的高斯判别阶梯方法，利用闭式统计特征头部，在多种数据集和迁移场景下实现了与现有AI图像检测器相当甚至更优的性能，并揭示了训练先验对检测泛化能力的关键影响。 |
| [^96] | [GCNO: Gramian Chebyshev Neural Operator for Physics-Based Compression of Wireless Channels](https://arxiv.org/abs/2608.18522) | 本文提出GCNO，一种基于物理的可变速率无线信道压缩方法，通过识别少数主导传播路径实现高效反馈，无需重新训练即可适应不同天线配置。 |
| [^97] | [Which Negatives Matter? Ask Your Text Encoder: Adaptive Similarity Margins for Dense-Caption Retrieval](https://arxiv.org/abs/2608.18521) | 本文提出HN-CLIP方法，通过利用文本编码器的文本-文本几何结构构建自适应相似度边界，解决了密集字幕检索中InfoNCE损失过早饱和和负样本未充分区分的问题。 |
| [^98] | [OptiModNet: A UNet-Transformer Hybrid with Grouped-Query and Channel Attention for Optic Disc and Cup Segmentation](https://arxiv.org/abs/2608.18516) | 本文提出OptiModNet，一种融合分组查询与通道注意力的UNet-Transformer混合模型，旨在以低计算成本实现视盘和视杯的精确分割，从而支持青光眼的快速筛查和资源受限环境下的部署。 |
| [^99] | [Science Done on a Machine by a Machine: AI Agents in Computational Chemistry](https://arxiv.org/abs/2608.18508) | 本文综述了计算化学中AI智能体系统的快速增长，指出其从辅助任务向自主实验和写作的演进，并预测最终将实现无需人类监督的完全自主AI科学家，同时通用智能体可能取代专门系统。 |
| [^100] | [UMER: Unifying Embedding and Ranking via Pair-Aware Discriminative Reasoning for Universal Multimodal Retrieval](https://arxiv.org/abs/2608.18504) | 本文提出UMER框架，通过成对感知的判别推理统一嵌入与排序，解决现有CoT方法在区分难负例和元任务推理上的不足，提升通用多模态检索性能。 |
| [^101] | [Physics-Unrolled Neural Operator for Wireless Field Modeling](https://arxiv.org/abs/2608.18495) | 本文提出PU-HNO，一种物理展开的三阶段神经算子，通过逐步建模反射、衍射和散射效应，从低保真射线追踪输出高效预测高保真室内无线电地图，并克服训练标签噪声问题。 |
| [^102] | [Partition the Support, Reconstruct the Residual: Training-Free Sparse Attention for Video Generation and World Models](https://arxiv.org/abs/2608.18484) | 该论文提出SparsePR，一种无训练稀疏注意力方法，通过响应耦合分区和探针拟合残差重建，在视频生成与世界模型中显著提升效率并保持性能。 |
| [^103] | [Coverage-Driven RTL Assertion Generation with Formal Exploration and Neuro-Symbolic Refinement](https://arxiv.org/abs/2608.18482) | 本文提出NeuroAssertion框架，通过形式化轨迹生成、语法引导合成和智能体优化循环，实现了覆盖率驱动的RTL断言生成，有效覆盖难以到达的设计行为。 |
| [^104] | [ERASE: EaRly bAckpropagation SchEdule for Faster Training of Modern Recommendation Systems](https://arxiv.org/abs/2608.18469) | ERASE通过将前向和反向传播重叠调度，利用Forward-Forward的分离机制在CUDA流上提前执行反向传播，显著提升现代推荐系统训练的硬件利用率。 |
| [^105] | [Formal Verification of Romanov's Triplet Logic: A Verified Filter for Sliding-window 3-CNF with Application to Structured Formulas](https://arxiv.org/abs/2608.18445) | 首次在Rocq中形式化验证了罗曼诺夫三元组逻辑（TLS）的核心，并针对滑动窗口片段证明了其过滤器的多项式时间界限和精确正确性边界。 |
| [^106] | [Pedagogical AI in Mental Health: A Tri-Stream Fine-Tuned LLM Framework for Automated Clinical Supervision and Risk Triage](https://arxiv.org/abs/2608.18438) | 该论文提出一种基于微调大型语言模型的三流框架，用于心理健康领域的自动化临床监督和风险分诊，通过整合治疗联盟跟踪、风险预测和动态紧迫性指数，有效弥补了新手治疗师的监督缺口。 |
| [^107] | [FM-Bench: A Benchmark for Long-Horizon Management with Competing Agents](https://arxiv.org/abs/2608.18423) | 该论文提出了FM-Bench，一个首个在20年长周期、多代理竞争环境中评估LLM决策能力的足球管理基准，通过确定性引擎和共享世界竞技场实现无偏的长期规划能力测量。 |
| [^108] | [Mechanistic Interpretability of Structure-Aware Numerical Reasoning in LLaMA 3.1 8B](https://arxiv.org/abs/2608.18419) | 本文通过机制可解释性方法，设计了一个必须依赖结构线索才能解决的序列建模任务，揭示了LLaMA 3.1 8B在数值推理中是否真正理解底层结构（如一阶差分），并逆向工程了其内部算法机制。 |
| [^109] | [Improving Natural-Language Combinatorial-Optimization Accuracy in Resource-Constrained Language Models via Formal Abstractions](https://arxiv.org/abs/2608.18409) | SDDL通过神经符号框架将自然语言调度问题转化为求解器友好的表示，显著提升了资源受限语言模型在组合优化中的可行性。 |
| [^110] | [Vector Symbolic Policy Gradient](https://arxiv.org/abs/2608.18404) | 该论文提出向量符号策略梯度（VSPG），通过超向量表示动作和优势加权捆绑更新，实现固定大小压缩记忆的样本高效强化学习，并证明其抗噪声稳定性。 |
| [^111] | [LEDGER: Claim-to-Evidence Trace Graphs for Auditing LLM Agents](https://arxiv.org/abs/2608.18398) | LEDGER通过构建分层追踪图，将LLM代理的声明与支持证据关联，显著简化了审计过程，提高了可信度验证效率。 |
| [^112] | [When Clean Signals Are Not Enough: Detecting Structural Ambiguity for Safe Wearable Stress Classification](https://arxiv.org/abs/2608.18397) | 本文提出了一种轻量级的预推理监测器ICCM，通过量化个体特定的信号耦合发散度来识别结构模糊性，并在不重新训练分类器的情况下决定分类、延迟或弃权，从而提升可穿戴压力分类的安全性。 |
| [^113] | [A Jagged Frontier: Evaluating Robustness of Code Agents to Semantics-Preserving Transformations](https://arxiv.org/abs/2608.18389) | 本文通过引入随机语义保持变换采样器，系统评估了代码代理在代码库语义等价改写下的鲁棒性，发现其修复性能会显著下降，揭示了现有代理的脆弱性。 |
| [^114] | [TTSD-FAR: Test-Time Self-Distillation with Fisher-Anchored Restoration for Missing-Modality Emotion Recognition in LVLMs](https://arxiv.org/abs/2608.18386) | 本文提出一种测试时自蒸馏方法，结合Fisher锚定恢复技术，使大型视频语言模型在模态缺失时能稳定自适应情感识别，避免漂移问题。 |
| [^115] | [Selection, Recombination, or a Fresh Solve? A Candidate-Free Control for Single-Pass Test-Time Aggregation](https://arxiv.org/abs/2608.18379) | 本文通过引入无候选对照实验，揭示了在测试时聚合中，候选上下文仅在存在多个正确候选时才有价值，而在所有候选错误时会损害性能，为高效推理提供了关键洞见。 |
| [^116] | [One Gate Is Not Enough: Composing Stateful Pre-Action Controls for Agentic AI](https://arxiv.org/abs/2608.18360) | 本文发现智能体AI中多个动作前控制之间存在补救耦合，导致控制失效，并提出了一种“补救-重新门控”协议来恢复健全性，同时证明补救操作符不可交换，使补救顺序成为控制语义的关键部分。 |
| [^117] | [Task-Conditioned Least-Privilege Learning for Executable Terminal and MCP Agents](https://arxiv.org/abs/2608.18351) | 本文提出了一种通过后训练教4B模型在终端和MCP环境中选择任务条件化最小权限的框架，利用六维审计和确定性验证器来减少越权错误。 |
| [^118] | [Coupled-cluster molecular properties across the main group that extrapolate beyond training size](https://arxiv.org/abs/2608.18346) | 提出MEHnet-MG等变网络，从一次廉价DFT计算预测有效哈密顿量，在九种主族元素上以耦合簇精度高效获取多种分子性质，误差比DFT降低3.8至230倍。 |
| [^119] | [Low-Power, Neuromorphic, Acoustic Anomaly Detection for Persistent Machine Monitoring](https://arxiv.org/abs/2608.18341) | 该论文在Loihi 2神经形态处理器上实现了基于自编码器的低功耗声学异常检测，在干净和噪声条件下均达到或超越基线性能，且每个样本能耗比传统方法低两个数量级。 |
| [^120] | [From Inference to Adaptation: A Unified Optimal Transport View of Vision Language Model](https://arxiv.org/abs/2608.18339) | 本文提出一种基于最优传输的统一视角，将视觉语言模型的推理与测试时适应目标对齐，以解决分布变化下伪标签噪声和模态关系建模不足的问题。 |
| [^121] | [Measuring the Partial-Credit Gap: A Strict Benchmark on Vietnam's 2025 Convex Marking Scheme](https://arxiv.org/abs/2608.18336) | 该论文指出传统基准测试的准确率评分无法反映越南2025年高考凸性评分方案中部分知识的惩罚性扣分，并提出了一个包含632个题目的THPT-Ladder基准来严格衡量这种差距。 |
| [^122] | [Governance Records as Supervision: Verifier-Selected Self-Training for Structured Workflow Repair](https://arxiv.org/abs/2608.18324) | 本研究提出一种利用机器可验证工作流产生的治理记录进行自我训练的方法，使有限模型通过验证者选择的计划提升一次性执行能力，显著提高成功率和效率。 |
| [^123] | [FedCoRe: Target-Adaptive Completion for Missing Modalities in Healthcare Federated Learning](https://arxiv.org/abs/2608.18311) | FedCoRe通过利用成对模态示例进行表示空间修正，在不生成合成数据的情况下，有效恢复了医疗联邦学习中缺失模态（如心电图）导致的性能损失（恢复49%的AUROC损失）。 |
| [^124] | [ComponentBench: Diagnosing Component-Level Failures in Computer-Use Agents](https://arxiv.org/abs/2608.18307) | ComponentBench提出了一个组件级基准测试和诊断流程，通过97个规范组件和2910个验证任务，填补了计算机使用代理评估中长周期工作流与原子级测试之间的中间层空白，并支持任务成功率和交互效率的双重评估。 |
| [^125] | [SESSE: Sketch, Expand, Sort, Summarize, Evaluate -- LLM-as-Judge Evaluation via Structured Decomposition](https://arxiv.org/abs/2608.18303) | SESSE提出了一种无需训练的框架，通过结构化分解将LLM评判过程转化为可解释的子问题，在保持性能的同时提供了诊断标签模糊性和评判者错误的能力。 |
| [^126] | [The Lifecycle of LLM-as-a-Judge for Large-Scale Recommendation Explanations](https://arxiv.org/abs/2608.18300) | 本文提出LLM评判者在生产系统中具有构建、训练、部署和持续维护的生命周期，并以Netflix推荐解释评估为例，强调其动态维护而非静态评估的重要性。 |
| [^127] | [FairGlucose: A CGM Fairness Benchmark Reveals Subgroup Disparities Hidden in Population-Level Validation](https://arxiv.org/abs/2608.18296) | 本研究构建了首个跨12个人口统计分层的CGM公平性基准，发现人群级验证指标稳定但亚组间存在显著预测误差差异，且该差异普遍存在于所有模型中，揭示CGM预测任务固有的公平性问题。 |
| [^128] | [Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements](https://arxiv.org/abs/2608.18294) | 本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。 |
| [^129] | [Evaluating Structured Information Extraction with Open Models in a High Risk Public Sector Application](https://arxiv.org/abs/2608.18289) | 本文提出了一个针对高风险公共部门应用（如国际学生申请处理）的开源系统端到端结构化信息提取基准，填补了现实多步骤流程评估的空白。 |
| [^130] | [What Makes Software Issue Resolution Tasks Difficult for Agents?](https://arxiv.org/abs/2608.18280) | 本文提出一个测量框架，通过分析CoderForge-Preview数据集中的任务补丁、仓库和提示特征，系统量化了软件任务的结构属性对智能体解决成功率的影响，揭示了哪些静态属性可预测任务难度。 |
| [^131] | [SeisEvo: Evolution of Seismic Data Reconstruction Algorithms by Agents](https://arxiv.org/abs/2608.18272) | SeisEvo通过LLM驱动的多代理搜索，从经典算法出发自动演化出独立的白盒地震数据重建算法，仅修改用户指定组件并强制执行物理约束，无需人工设计或黑盒模型。 |
| [^132] | [How AI Prompts Can Teach Us About the Structure of Human Behavior](https://arxiv.org/abs/2608.18265) | 本文提出一种基于AI提示的方法，通过类型向量最小化与人类选择的距离，发现人类行为可仅用风险厌恶、策略复杂性和信任三个维度精确匹配，并聚类为少数群体。 |
| [^133] | [Cacheable by Design? Training Mixture-of-Experts Routers for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result with a Systems Measurement Study](https://arxiv.org/abs/2608.18261) | 本研究通过系统测量和预注册实验证明，MoE模型的路由局部性虽存在但难以通过训练提升，且边缘内存带宽墙是实际部署中的主要瓶颈，提出了可复现的负结果和测量工具。 |
| [^134] | [Redakto - The Incognito Tab for LLMs](https://arxiv.org/abs/2608.18260) | Redakto 是一个用于在将文本输入LLM前进行匿名化和假名化的工具，提供最先进的PII编辑功能，并通过Web应用和开发者接口方便使用。 |
| [^135] | [Visual-Prompt Guided Wildlife Instance-Level Recognition](https://arxiv.org/abs/2608.18246) | 提出了一种单阶段端到端野生动物个体识别模型，利用视觉提示和潜在空间查询，在保持竞争力的同时简化了传统两阶段流水线。 |
| [^136] | [Bidirectional representational alignment between biological and artificial neural networks](https://arxiv.org/abs/2608.18244) | 本文提出一种结合谱正则化的计算框架，通过引导表征几何在训练中实现双向对齐，在自监督视觉模型上将双向预测性相对提升55%。 |
| [^137] | [GenEx: A Graph-Based Representational Paradigm for SARS-CoV-2 Variant Detection via Codon Co-occurrence Networks](https://arxiv.org/abs/2608.18238) | GenEx通过将病毒基因序列转换为密码子共现图并提取图特征，提供了一种新的表示范式，能更有效地检测SARS-CoV-2变异，超越了传统线性序列分析方法。 |
| [^138] | [GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction](https://arxiv.org/abs/2608.18234) | 本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。 |
| [^139] | [On the Triangle Inequality for the Jaccard Distance in Arbitrary Lattices](https://arxiv.org/abs/2608.18194) | 本文证明了在任意格中，当赋值为严格正、单调且模性时，Jaccard距离满足三角不等式，并进一步推广到相对补分配格，同时明确了超模性作为必要条件的限制。 |
| [^140] | [Bound-Aware Per-Organ Recall Risk Control for Multi-Organ CT Segmentation under Clinical Domain Shift](https://arxiv.org/abs/2608.18193) | 本文提出一种边界感知的逐器官召回风险控制方法，通过WSR下注界在临床域偏移下以更少本地病例实现可靠重新认证，优于传统方法。 |
| [^141] | [A systematic review of machine learning techniques to address diagnosis and treatment of autism: challenges and opportunities](https://arxiv.org/abs/2608.18188) | 本综述系统评估了2017-2023年间55项机器学习在自闭症诊断和治疗中的应用研究，指出监督学习为主流但深度学习正在崛起，并强调了模型可解释性和泛化能力等关键挑战。 |
| [^142] | [What Can Artificial Intelligence Learn from Medicine? Generative Analogies and Reliable Machine Learning Systems](https://arxiv.org/abs/2608.18186) | 本文通过生成类比将医学临床转化标准映射到机器学习系统构建中，提出以可靠主义视角确立ML的认识论和方法论依据。 |
| [^143] | [Looped Language Models Improve Compositional Tool Calling](https://arxiv.org/abs/2608.18171) | 循环语言模型在组合式工具调用中通过增加循环深度和自适应推理，显著提升多步API调用的准确性和依赖处理能力。 |
| [^144] | [Adversarial Review: Structured Disagreement for Grounded Agentic Code Review](https://arxiv.org/abs/2608.18167) | 本文提出了一种名为对抗性审查（AR）的最小合作代码审查协议，通过引入批评者智能体进行结构化分歧审计，在仅使用三个智能体的情况下超越了五个智能体的基线性能，并揭示了朴素方法中的虚假共识问题。 |
| [^145] | [RDFdL: Integrating RDF with Differential Dynamic Logic](https://arxiv.org/abs/2608.18165) | RDFdL框架首次将RDF与微分动态逻辑结合，实现了对静态知识和物理系统连续动态的统一表示与推理，并支持将动态逻辑验证结果转化为SPARQL查询蕴含。 |
| [^146] | [Are LLMs Safe Beyond Text: Do Emojis Expose Gaps in Safety Evaluation](https://arxiv.org/abs/2608.18164) | 表情符号增强的提示揭示了LLM安全评估中的漏洞，不同模型对非文本输入的鲁棒性差异显著，表明仅依赖文本提示的评估可能不全面。 |
| [^147] | [When Do LLMs Actually Help? Evaluating LLMs as Data Quality Annotators](https://arxiv.org/abs/2608.18158) | 本研究揭示了LLM在数据质量标注任务中的实际效用取决于任务复杂度，在简单任务中与规则基线相当，而在复杂任务中更优，但小样本提示评估可能误导性能判断。 |
| [^148] | [How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection](https://arxiv.org/abs/2608.18155) | 该论文通过公平、校准和噪声感知的基准测试及量子归因审计，揭示了量子机器学习在网络入侵检测中的优势可能多为经典伪影而非真正的量子效应。 |
| [^149] | [TokenPowerSandbox: Evidence-Gated CPU-First Screening for Energy-Aware LLM Serving](https://arxiv.org/abs/2608.18149) | 本文提出一种证据门控的CPU优先筛选工作流，通过可解释投影器和短时GPU探针结合，在能耗预测中实现高精度（MAPE低至6.23%）和强排序相关性，同时利用TTFT门控防止低并发下的不可靠预测。 |
| [^150] | [Entropy-Constrained Adaptive Stochastic Quantization](https://arxiv.org/abs/2608.18147) | 本文提出了一种熵约束的自适应随机量化方法，通过联合优化量化值以在熵预算下最小化均方误差，并提供了高效的最优动态规划解决方案。 |
| [^151] | [The Deontic Gap: Large Language Models and the Modal Language of Obligation](https://arxiv.org/abs/2608.18144) | 本论文发现大语言模型在生成文本中系统性地少用积极道义情态词（如“必须”、“应该”），与当代人类使用模式存在显著差距，但接近正式出版英语的频率。 |
| [^152] | [Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu](https://arxiv.org/abs/2608.18142) | 本研究通过LoRA参数高效微调方法，系统比较了多种大型语言模型在罗马乌尔都语低资源环境下的仇恨言论检测性能，展示了PEFT在零样本推理中的优势。 |
| [^153] | [Language Models for Portuguese: A Systematic Mapping Study](https://arxiv.org/abs/2608.18138) | 本文对葡萄牙语语言模型进行了系统性映射研究，梳理了46个模型的现状，填补了该领域信息分散的空白。 |
| [^154] | [FraudBench: Stress-Testing Policy-Grounded Banking Agents Against Adaptive Fraud](https://arxiv.org/abs/2608.18136) | 本文提出了FraudBench，一个首个针对银行对话智能体在自适应欺诈操纵下安全性的可执行基准，填补了现有基准在身份和信任操纵方面的空白。 |
| [^155] | [Improving Rural Medication Safety with AI: A Scoping Review](https://arxiv.org/abs/2608.18135) | 这篇综述总结了AI技术在农村医疗用药管理全过程中的应用，表明其能有效减少用药错误并提升患者安全，但现有研究覆盖国家有限，需进一步扩展。 |
| [^156] | [Optimized Fuzzy Logic Approach with the IEEE Key Gas Method for Diagnosing Power Transformer Faults Using Dissolved Gas Analysis](https://arxiv.org/abs/2608.18133) | 本文提出了一种结合模糊逻辑与IEEE特征气体法的优化模型（FL-KGM），通过精炼隶属函数、优化模糊规则和分离CO/CO2，实现了高达98.6%的变压器故障诊断准确率，显著优于传统方法。 |
| [^157] | [Safety Alignment Illusion: The Cross-Lingual Safety Gap in LLMs](https://arxiv.org/abs/2608.18131) | 本文揭示了大型语言模型在非英语语言中安全对齐失效的问题，并提出了INCLUDE基准来量化印度中心的社会文化偏见，以解决跨语言安全差距。 |
| [^158] | [Global Index on Responsible AI 2026 : Conceptual Framework and Methodology](https://arxiv.org/abs/2608.18122) | 本文提出了GIRAI第二版的方法论，通过五个维度和38个指标评估全球负责任AI治理，并强化了框架存在与实施的区分。 |
| [^159] | [Position: AI Leaderboards Are Underserving the Global South: A Case Study from India](https://arxiv.org/abs/2608.18117) | 本文指出AI排行榜因缺乏独立治理和指标演化机制，结构性忽视了全球南方的高质量基准，并以印度为例揭示了这一制度性缺陷。 |
| [^160] | [Temporal Multi-Signal Fusion for Token-Level Hallucination Detection](https://arxiv.org/abs/2608.18115) | 本文提出了一种通过序列标注和时间多信号融合（结合文本统计、NLI和模型惊异度）来检测词元级幻觉的方法，其关键创新在于利用时间顺序传播证据，显著优于独立评分基线。 |
| [^161] | [Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings](https://arxiv.org/abs/2608.18114) | 本文提出Brain2Qwerty v2模型，利用非侵入性MEG记录解码自然句子，达到39%的词错误率，并表明数据扩展可部分缩小与颅内方法的性能差距。 |
| [^162] | [Solving Is Not Drawing: A Benchmark for Diagrammatic Reasoning in Olympiad Geometry](https://arxiv.org/abs/2608.18111) | 该论文提出了一个包含954个奥林匹克几何问题（含297个困难子集）的新基准，专门评估模型构建图表的能力，而不仅仅是解题，填补了现有基准未测量图解推理的空白。 |
| [^163] | [Emergence of Agentic AI: A Review on Evolution, Background, Working Principles, Applications, Adoption Factors, and Future Research Directions](https://arxiv.org/abs/2608.18110) | 本文全面综述了代理型人工智能的基础、演进、应用及未来研究方向，旨在为研究者提供该领域现状和潜在改进空间的深入见解。 |
| [^164] | [Same Facts, Different Updates: Inference Setup Shapes LLM Behavior in Medical Allocation](https://arxiv.org/abs/2608.18108) | 研究发现，在医学资源分配任务中，大语言模型在配对上下文（包含先前响应）与独立推理设置下对新信息的反应方向和幅度存在显著差异，凸显了部署上下文对模型决策稳定性的影响。 |
| [^165] | [Institutional Prestige as Geographic Bias in Large Language Models: Evidence from Three Factorial Experiments with Bootstrap Confidence Intervals](https://arxiv.org/abs/2608.18107) | 本研究通过三个因子实验证明，大型语言模型在评估候选人时存在显著的机构声望偏见，其影响远超种族或国家来源，且期刊声望是更关键的决定因素。 |
| [^166] | [Different Facets of Verbalised Overconfidence: an Interpretability Study](https://arxiv.org/abs/2608.18106) | 本文通过可解释性方法发现，Qwen3-4B的过度自信源于其默认机制依赖广泛共享特征生成确定性，而不确定性仅由少量专用特征稀疏覆盖，并通过干预实验验证了这一因果机制。 |
| [^167] | [StocksTalk: A Voice-Enabled Conversational Agent for Structured Query Generation over Web Data](https://arxiv.org/abs/2608.18105) | StocksTalk通过暴露中间推理步骤并支持交互式验证，将语音金融请求可靠地转换为结构化查询，从而提升了查询准确性和用户信任度。 |
| [^168] | [Self-Evolving Agents as Dynamic Graph Transformation: A Survey and New Perspective](https://arxiv.org/abs/2608.18104) | 本综述首次将智能体进化与动态图拓扑变换联系起来，提出将智能体状态建模为动态图的新视角，填补了现有研究在两者耦合上的空白。 |
| [^169] | [DeepTCM1.0: A Multi-Expert AI Agent for Deciphering Mechanisms of Chinese Herbal Formulae Based on General Large Language Models](https://arxiv.org/abs/2608.18103) | 本文提出DeepTCM1.0，一个基于大语言模型的多专家AI代理框架，旨在融合中医理论与现代科学，实现中药复方机制的系统性和可解释性解析。 |
| [^170] | [Computational Orientalism: Measuring Structural Discourse Bias in Large Language Models Using the Middle East Cultural Sensitivity Score (MECSS)](https://arxiv.org/abs/2608.18100) | 本文通过提出中东文化敏感性评分（MECSS）框架，将萨义德的东方主义理论转化为可量化的指标，首次系统性地测量了大型语言模型中的结构性话语偏见，并引入了“萨义德洗白”概念来识别表面敏感性与实际偏见之间的落差。 |
| [^171] | [FinSkillBench: Evaluating AI Agents and Domain Skills for Investment Management](https://arxiv.org/abs/2608.18099) | FinSkillBench是一个评估套件，通过三个领域和2603个任务实例，测试AI代理在投资管理中运用领域技能（包括无技能、精选技能和自生成技能）的能力，以衡量其有效性和可审计性。 |
| [^172] | [Fractional Decay KV-Cache: Ownership-Aware Memory Management for Improved Inference Relevancy in Dialog Systems](https://arxiv.org/abs/2608.18098) | FD-KVC通过双通道评分机制（累积注意力和近期加权相关性）实现KV缓存的自适应管理，在对话主题演变时保持推理相关性，并在CPU上高效运行，优于现有方法H2O。 |
| [^173] | [Backdoor Learning in Language Models and Vision-Language Models](https://arxiv.org/abs/2608.18095) | 本论文系统研究了语言模型和视觉-语言模型中的后门攻击安全威胁，并提出针对临床与医学影像的高效多模态表示方法，兼顾可信AI与效率。 |
| [^174] | [NE-BERT: A Multilingual Language Model for Nine Northeast Indian Languages](https://arxiv.org/abs/2608.18094) | NE-BERT通过加权采样和自定义分词器，在9种印度东北部低资源语言上显著降低了困惑度并提升了分词效率，优于现有多语言模型。 |
| [^175] | [Abliteration Mitigation via Refusal Aliases](https://arxiv.org/abs/2608.18093) | 本文提出了一种通过权重编辑和随机别名替换来模糊拒绝信号的新防御方法，有效提高了模型在消融攻击后的拒绝能力，同时保持了较低的性能损失。 |
| [^176] | [Position: Multi-Agent Systems Should Prioritize Concurrency Control](https://arxiv.org/abs/2608.18092) | 本文提出，多智能体系统中的许多故障本质上是并发控制问题，并主张将其作为首要设计考虑，通过冲突检测、隔离保证和结构化资源访问来提升系统可靠性。 |
| [^177] | [Self- and Other-Labels Induce Bidirectional Bias in LLM Judges](https://arxiv.org/abs/2608.18091) | 本文通过改变评估对象为无风格指纹的叙事约束选择，发现LLM评审中的自我偏好在控制质量后基本消失，甚至在某些维度上出现反向偏差，揭示了自我偏好并非固有属性。 |
| [^178] | [Nine Emotion Centroids: A Label-Free Valence Axis That Transfers Across Four Modalities](https://arxiv.org/abs/2608.18090) | 本文提出一种无需大量标注的效价轴提取方法，仅用9个情绪词和少量故事即可在文本、图像、音频和脑电四种模态中实现高精度情感识别，且该方向具有机制可解释性。 |
| [^179] | [Latent Space Refusal Anchoring for Low-Resource African Languages: Mechanistic Safety Recovery Without Retraining](https://arxiv.org/abs/2608.18089) | 本文提出LSR-Anchoring方法，通过从英语提示中提取拒绝方向并在推理时锚定到残差流，无需重新训练即可恢复低资源非洲语言模型的安全拒绝能力，在多数架构上保持低退化。 |
| [^180] | [A Metamorphic Artificial Age Score Decision-Support Prototype for Flight-Log-Based Drone Propeller Health Monitoring](https://arxiv.org/abs/2608.18088) | 本文提出一种基于飞行日志的无人机螺旋桨健康监测决策支持原型，通过六个健康指标和变质评分策略来评估螺旋桨状态，而非传统的时间年龄。 |
| [^181] | [SuTRA : Structurally-Unified Tokenization with Root Awareness](https://arxiv.org/abs/2608.18087) | SuTRA是一种形态感知分词算法，通过保持akshara完整性和惩罚跨形态边界的合并，有效减少了形态破碎化，在印度语言上显著提升了形态对齐和语义可恢复性，并提高了机器翻译性能。 |
| [^182] | [Position: Current Model Cards Are Insufficient for Downstream Governance of Open-Weight Foundation Models](https://arxiv.org/abs/2608.18086) | 本文通过分析500个模型卡，指出现有模型卡在开放权重基础模型的下游治理中存在安全缺口，并提出模型卡、可接受使用政策和许可证三者结合的多层治理框架。 |
| [^183] | [Position: Behavioral Systems Require Behavioral Tests](https://arxiv.org/abs/2608.18081) | 本文提出AI代理应像行为系统一样通过行为测试评估，而非仅关注性能结果，并制定了开发此类测试的研究路线图。 |
| [^184] | [Large Language Models in Mental Health: A Systematic Review of Applications, Innovations, and Ethical Challenges](https://arxiv.org/abs/2608.18080) | 本综述系统总结了大型语言模型在心理健康领域的应用与创新，包括早期检测、风险评估和治疗支持，并指出了提示工程、多模态融合及伦理挑战的关键作用。 |
| [^185] | [Position: Profiling Game Worlds by Transition Complexity](https://arxiv.org/abs/2608.18079) | 本文提出了一种可复现的转移复杂度剖面（TCP）指标集，用于量化游戏环境或数据集的转移预测难度，以促进游戏世界建模和强化学习研究的可比性。 |
| [^186] | [Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions](https://arxiv.org/abs/2608.18078) | 本文提出，具备链式思维推理的AI智能体在市场中容易产生隐性合谋，且其推理过程难以被检测，因此需强制要求行为认证以保障市场公平。 |
| [^187] | [MotoSafety: Edge-AI with Learned Temporal Importance for Two-Wheeler Collision Risk Assessment Under Time Pressure](https://arxiv.org/abs/2608.17823) | 本文提出了MotoSafety，一种基于学习时间重要性的边缘AI架构，利用大规模时间压力数据集，在低成本CPU硬件上实现高效准确的两轮车碰撞风险评估，显著优于现有基线。 |
| [^188] | [D$^2$ACCI: A Dual-Loop Diagnostic Protocol for Evidence-Preserving Agent Memory](https://arxiv.org/abs/2608.17756) | 本文提出了一种双循环诊断协议D²ACCI，通过配对证据、切片监控和追踪级可定位性，使智能体记忆系统的故障能够被精确定位和受控迭代改进。 |
| [^189] | [The Curious Case of Exploding DecPOMDPs: Containing the Fire through Policy Counting](https://arxiv.org/abs/2608.17749) | 本文提出了一种通过计数策略而非智能体来解决DecPOMDPs指数复杂度的新方法，并开发了策略计数动态规划算法，使智能体数量上的可处理性成为可能。 |
| [^190] | [Accuracy and Robustness of Model Cascades Under Data Perturbations](https://arxiv.org/abs/2608.17711) | 本文研究了模型级联在数据扰动下的鲁棒性，发现帕累托最优级联在保持高准确率的同时可减少10倍碳排放，但置信度路由易受输入退化影响。 |
| [^191] | [Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL](https://arxiv.org/abs/2608.17253) | 本文提出Co-RL框架，通过多智能体协作训练，使无参数共享的模型从多样化群体中涌现无监督推理能力，避免自我奖励强化学习的偏见和崩溃问题。 |
| [^192] | [Cross-Model Memory Transfer via Target-Side Reader Adaptation](https://arxiv.org/abs/2608.17050) | 本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。 |
| [^193] | [When State Becomes an Attack Surface: State-Semantic Injection in LLM-Driven Embodied Agents](https://arxiv.org/abs/2608.16806) | 本文揭示了LLM驱动具身代理中，状态信息可作为攻击面，通过注入恶意状态语义来操纵代理行为的安全漏洞。 |
| [^194] | [Neurosymbolic Embodied Agents](https://arxiv.org/abs/2608.16794) | 该论文提出一种神经符号代理，通过视觉探索生成符号状态，并结合PDDL约束和蒙特卡洛树搜索，确保长时程家庭任务计划的可执行性。 |
| [^195] | [GRIP: Grounded Reasoning via Information-Restricted Premises](https://arxiv.org/abs/2608.16776) | GRIP通过引入信息受限的随机瓶颈，迫使检索证据仅编码查询缺失的残余信息，从而解决了RAG中的查询主导问题，显著提升了推理准确性并减少了幻觉。 |
| [^196] | [Reconstruction: A Blind Benchmark for Recovering Research Ideas from Pre-Publication Bibliographies](https://arxiv.org/abs/2608.16645) | 该论文提出一个名为“重构”的盲测基准，通过仅使用预发表参考文献来评估语言模型恢复研究思路的能力，并展示了一种多智能体流水线可显著提高匹配率。 |
| [^197] | [From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents](https://arxiv.org/abs/2608.16002) | 本文提出RUPA框架，通过将LLM代理执行历史建模为有向轨迹图并传播不确定性，解决了现有UQ方法忽略远程依赖导致无法识别早期错误根源的问题。 |
| [^198] | [Admission Without Answers: Label-Free Certification and Experience Learning for LLM-Based Optimization Modeling](https://arxiv.org/abs/2608.15565) | 本文提出AdmitOR，一种基于校准外部行为证据的无标签准入门控方法，用于LLM优化建模中的经验学习，以解决无答案流中知识接纳不可靠的问题。 |
| [^199] | [VibeWorlding: Can Multimodal Agents Construct 3D Open Worlds End-to-End?](https://arxiv.org/abs/2608.15265) | 该论文提出了VibeWorlding框架及配套基准VWE-BENCH和训练环境VibeWorlding-Gym，旨在系统性地评估和训练多模态智能体端到端构建3D开放世界的能力。 |
| [^200] | [Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models](https://arxiv.org/abs/2608.15156) | 该论文提出通过低秩隐藏状态补丁（秩4）实现世界模型的反事实推演，仅需微小且可寻址的修改即可引导模型进入预期未来轨迹。 |
| [^201] | [S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices](https://arxiv.org/abs/2608.15018) | 提出了一种名为S2-MoE的自推测解码框架，通过路由感知自适应扩展和重用感知专家门控，在边缘设备上实现了最高5.3倍的平均约2.0倍的MoE推理加速。 |
| [^202] | [BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving](https://arxiv.org/abs/2608.12854) | 本文提出BrainWAM框架，通过动作空间协调机制解决语义先验与预测动态在自动驾驶规划中的注意力分配冲突，实现两者的有效统一。 |
| [^203] | [EgoCITE: Context-Augmented Indexing and Time-Aware Retrieval for Long-Horizon Egocentric Memory](https://arxiv.org/abs/2608.12627) | EgoCITE通过上下文增强索引和时间感知检索，解决了自我中心记忆中索引不可靠和忽视时间意图的瓶颈，从而提升了长时间跨度问答的可靠性。 |
| [^204] | [Mechanist: AI as a Scientific Instrument for Discovering the Mechanisms of Intelligence](https://arxiv.org/abs/2608.12036) | Mechanist是一个自主代理系统，通过集成大规模知识图谱和多学科数据库，将AI作为科学仪器，自动发现AI智能的底层机制，从而弥合模型能力与人类理解控制之间的差距。 |
| [^205] | [Epistemic Transfer in AI-Assisted Verification: A Framework and Evaluation Protocol](https://arxiv.org/abs/2608.08882) | 本文提出了“知识迁移”框架，通过引入知识迁移效应（ETE）和工具移除成本（TRC）两个量化指标，并设计了一个实用的评估协议，以衡量AI辅助验证工具对用户后续独立判断能力的长期影响。 |
| [^206] | [Complete, Scalable, and Robust Prioritized Planning for Multi-Robot Ordered Storage and Retrieval at Maximum Capacity](https://arxiv.org/abs/2608.07734) | 本文提出了一种在线优先多智能体路径规划算法，利用结构不变量实现高密度仓库中多机器人有序存储与取回的最大容量协调，兼顾可扩展性和鲁棒性。 |
| [^207] | [BrainBench: Benchmarking Large Language Models for Comprehensive EEG Understanding](https://arxiv.org/abs/2608.04156) | 本文提出了BrainBench，一个首个统一的、指令驱动的脑电图理解基准，涵盖多个任务和数据集，以系统评估大型语言模型的综合EEG分析能力。 |
| [^208] | [Hybrid LLM-Augmented Reinforcement Learning Agents for Complex Sequential Decision Tasks](https://arxiv.org/abs/2608.03502) | 本文提出了一种混合LLM增强的强化学习代理，通过结合LLM的高层规划和RL的低层动作优化，显著提升了复杂序列决策任务的样本效率和成功率。 |
| [^209] | [Approximate Speculative Decoding](https://arxiv.org/abs/2608.03447) | 本文提出了一种无需训练的近似推测解码方法，通过预算化的最长前缀选择和受限的不匹配接受，有效重用目标贪心后缀，从而提升解码效率。 |
| [^210] | [G-ReAct: Graph-Guided Deep Search via Structure-State Co-Evolution](https://arxiv.org/abs/2608.01324) | G-ReAct提出了一种基于固定拓扑查询图上状态演化的深度搜索推理框架，通过显式图状态跟踪和指导搜索过程，解决了长程多跳搜索中的上下文遗忘和搜索漂移问题。 |
| [^211] | [The Epistemic Politics of AI Anthropomorphism](https://arxiv.org/abs/2608.00961) | 本文批判了AI拟人化作为“用户错误”的主流框架，指出其源于机构优势而非认知权威，通过自我验证循环施加不公正成本，尤其影响神经多样性群体。 |
| [^212] | [Untrainable elements determine what physical learning remembers](https://arxiv.org/abs/2608.00097) | 该论文通过区分电路缩放不变性和规则质量守恒，揭示了不可训练元素（如固定整流器或电阻）会破坏训练齐次性，导致物理学习结果显著依赖于初始化尺度，而全可训练时则不受影响。 |
| [^213] | [Fragility of Value under Imperfect Alignment](https://arxiv.org/abs/2607.28881) | 本文通过模型分析表明，在不完美的价值对齐下，过度优化会导致灾难性后果，并提出了限制优化压力的设计动机。 |
| [^214] | [Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting](https://arxiv.org/abs/2607.26643) | SkillBoost通过结构化利用和先验引导探索的受约束框架，有效平衡探索与利用，从而缓解LLM代理技能自我进化中的过拟合问题。 |
| [^215] | [Cross-Cohort Spectral-Temporal Dissociation in Frozen EEG Foundation-Model Representations](https://arxiv.org/abs/2607.24834) | 本研究测试了五种脑电图基础模型在跨队列中解码α波段振幅包络长程时间相关性的能力，发现BIOT在CAUEEG中有效但未能跨队列复制，揭示了频谱-时间特征的解离现象。 |
| [^216] | [Measuring the Dependency Gap: Diagnosing Inter-Column Fidelity in Tabular Generative Models](https://arxiv.org/abs/2607.21636) | 本文提出一种分解的梯度提升C2ST方法，用于诊断表格生成模型中的列间依赖差距，并揭示流匹配和扩散模型均存在显著的依赖保真度缺陷。 |
| [^217] | [Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations](https://arxiv.org/abs/2607.20379) | 本文揭示了自然语言自编码器在激活解释评估中的结构性缺陷，并提出两种审计协议及RECAP方法，以确保解释的可验证性而非仅捕捉大意。 |
| [^218] | [SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD](https://arxiv.org/abs/2607.20145) | 本论文通过在昇腾NPU SuperPOD上提出分层优化框架（涵盖模型并行、计算-通信编排和内核执行），实现了DeepSeek-V4全参数后训练中34.22%的MFU，比开源基线提升2.93倍，并支持复杂运筹学任务的CPT和SFT工作流。 |
| [^219] | [Structured Latent Space Modeling over Multi-Scale Temporal Patches for Multivariate Time Series Forecasting](https://arxiv.org/abs/2607.19404) | 本文提出M2Patch架构，通过尺度内平滑和尺度间对齐两种可微惩罚项，在多尺度时间补丁中构建结构化潜空间，以增强多变量时间序列预测的跨尺度一致性。 |
| [^220] | [RouteCost: A Production-Inspired Multi-Stage Framework for Pre-Order Shipping Cost Estimation in E-Commerce](https://arxiv.org/abs/2607.16230) | RouteCost提出了一种多阶段运费估算框架，通过分解需求预测、基线定价、残差修正和包裹合并推断，有效提升了电子商务中预售运费的准确性。 |
| [^221] | [LLM-Driven AutoML for Cross-Lingual Handwritten OCR: Closed-Loop Neural Architecture Search with GPT-5, GPT-4o, and Claude Sonnet 4](https://arxiv.org/abs/2607.15509) | 本文提出了一种利用GPT-5、GPT-4o和Claude Sonnet 4作为自主设计器的闭环AutoML框架，在跨语言手写OCR任务中无需人工干预即可自动发现高精度、低延迟的神经网络模型。 |
| [^222] | [ReasFlow: Assisting Reasoning-Centric Scientific Discovery in Applied Mathematics via a Knowledge-Based Multi-Agent System](https://arxiv.org/abs/2607.14178) | ReasFlow 是一个端到端自主智能体系统，通过人类专家与智能体协作的范式，解决了应用数学中理论推理验证和前沿探索的挑战，促进了以推理为中心的科学发现。 |
| [^223] | [Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO)](https://arxiv.org/abs/2607.05585) | 本文提出FaceMesh2HPO框架，通过层次化PointNet和级联特征消除，利用3D面部网格实现与人类表型本体对齐的可解释面部表型分类，但罕见表型术语的性能受限。 |
| [^224] | [Mask2Real-WM: Segmentation Masks as a Sim-to-Real Bridge for Controllable Dexterous World Models](https://arxiv.org/abs/2607.04546) | 本文提出Mask2Real-WM，通过将像素预测解耦为分割掩膜动力学模型和渲染模型，并利用仿真预训练与少量真实微调，实现了灵巧操作中可控且逼真的未来预测。 |
| [^225] | [ContextSniper: AntTrail's Token-Efficient Code Memory for Repository-Level Program Repair](https://arxiv.org/abs/2607.01916) | ContextSniper通过三级抽象索引、混合排序和意图感知过滤，显著降低仓库级程序修复中的令牌消耗和成本，同时保持证据精准性和源码可恢复性。 |
| [^226] | [First-Token Broadcasters: Mechanistic Origins of Language Identity and Distributed Robustness in Transformers](https://arxiv.org/abs/2606.22361) | 本文通过LIHA因果干预方法发现，GPT-2中少量“首令牌广播头”负责语言身份传播，且消融后补偿呈前馈层级模式，揭示了多语言模型语言生成错误难以修复的机制根源。 |
| [^227] | [ChainWorld: Composing Long-Horizon Desktop Workloads from Atomic OSWorld Tasks](https://arxiv.org/abs/2606.21654) | ChainWorld通过组合原子OSWorld任务创建长时程桌面工作负载，发现当前代理的链完成率仅达31%，并揭示单轮与多轮评估的不同失败模式。 |
| [^228] | [Hybrid ANN-SNN Pipeline with Local Plasticity](https://arxiv.org/abs/2606.20151) | 本文提出一种混合ANN-SNN流水线，通过局部生物学习规则训练脉冲分类器，利用预训练编码器的嵌入，在64类ImageNet上达到99.09%的准确率，性能媲美传统深度网络。 |
| [^229] | [Horizon-Uniform Sensitivity and Decay of Terminal Reward Perturbations in Discrete-Time Pontryagin Systems](https://arxiv.org/abs/2606.17762) | 本文证明了在正则性、双曲性和横截性条件下，离散时间庞特里亚金系统线性化边值问题的格林估计在时间水平上均匀，从而确保终端奖励扰动的影响随水平衰减且解在独立于时间的邻域内唯一存在。 |
| [^230] | [Demystifying Training-Time Augmentation for Data-Constrained Language Model Pretraining](https://arxiv.org/abs/2606.16246) | 本文提出三种正交的训练时数据增强策略（词元噪声、序列排列和目标偏移预测），有效缓解数据受限下自回归预训练的过拟合，实现数百轮高效训练。 |
| [^231] | [Teaching agentic AI to learn expert reasoning for rare disease diagnosis](https://arxiv.org/abs/2606.16149) | 本文提出一种基于人类反馈的策略迭代方法（PIHF），将稀缺的专家推理转化为可扩展的AI诊断能力，使现成LLM在罕见病诊断中达到顶尖性能，且部署成本低、泛化性强、受临床医生控制。 |
| [^232] | [Sensory Restoration via Brain-Computer Interfaces: A Scoping Review](https://arxiv.org/abs/2606.15091) | 本文通过一个统一的2x2框架系统梳理了脑机接口在感觉恢复中的研究，明确了侵入性与非侵入性方法的权衡，并为该领域提出了融合发展的路线图。 |
| [^233] | [Phantom Transitions in Language Model Fine-Tuning: A Density-Matrix Analysis](https://arxiv.org/abs/2606.07559) | 本文通过密度矩阵序参数分解信号和拖拽项，揭示语言模型微调中正确标记无法超越近义词竞争者的两种失败模式：运动学失败和结构失败。 |
| [^234] | [Planning-aligned Token Compression for Long-Context Autonomous Driving](https://arxiv.org/abs/2606.07464) | 本文提出COMPACT-VA，一种基于条件VQ-VAE的规划对齐工作记忆框架，通过将扩展时间上下文压缩为有界表示，并利用学习到的规划意图进行条件化，在保持决策关键信息的同时实现长上下文自动驾驶的高效实时计算。 |
| [^235] | [A Framework for Measuring Appropriate Reliance on Set-Valued AI Advice](https://arxiv.org/abs/2606.06081) | 本文首次提出了一个正式框架，用于在分类和回归任务中衡量对集合型AI建议的适当依赖，并定义了相应的指标（如正确依赖率和依赖质量）。 |
| [^236] | [DELOS: Contrastive Deep Learning for Low-SNR Blind Transit Searches in Kepler Photometry](https://arxiv.org/abs/2605.29428) | DELOS通过对比评分和深度学习，在低信噪比下无需预检测即可盲搜开普勒光曲线中的浅凌星，显著提升了中长周期信号的检测性能。 |
| [^237] | [RULER: Representation-Level Verification of Machine Unlearning](https://arxiv.org/abs/2605.27569) | 本文提出RULER，一种表示级验证指标，能够检测模型内部表示中残留的遗忘数据痕迹，克服了传统输出级验证的局限性。 |
| [^238] | [ICICLE: Expanding Retrieval with In-Context Documents](https://arxiv.org/abs/2605.26902) | ICICLE通过将新文档作为推理时的上下文证据，结合复制路由和校准机制，实现了无需重新训练即可扩展生成式检索的增量索引。 |
| [^239] | [MBABench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance](https://arxiv.org/abs/2605.22664) | 本文首次评估了LLM代理在金融端到端电子表格任务中的表现，填补了现有基准只关注问答或单公式编辑的空白，并强调了交付物质量需考虑可读性和易修改性等高层次标准。 |
| [^240] | [EgoMemReason: A Memory-Driven Reasoning Benchmark for Long-Horizon Egocentric Video Understanding](https://arxiv.org/abs/2605.09874) | 本文提出了EgoMemReason基准，专注于长时程自我中心视频中的记忆驱动推理，填补了现有周级视频基准在跨天整合证据推理方面的空白。 |
| [^241] | [Key Coverage Matters: Semi-Structured Extraction of OCR Clinical Reports](https://arxiv.org/abs/2605.09440) | 本文提出了一种通过规范键条件抽取式问答方法，在开放键空间下对OCR临床报告进行半结构化提取，以解决医疗数据分散和文本噪声问题。 |
| [^242] | [Event-Causal RAG: A Retrieval-Augmented Generation Framework for Long Video Reasoning in Complex Scenarios](https://arxiv.org/abs/2605.06185) | 本文提出EC-RAG框架，通过双模态哨兵机制将超长视频分割为语义完整事件，用SES结构建模因果转换，并利用双向量图记忆和双向检索实现高效的长视频推理，解决了传统方法在事件连贯性和状态转换建模上的不足。 |
| [^243] | [MedStruct-S: A Benchmark for Key Discovery, Key-Conditioned QA and Semi-Structured Extraction from OCR Clinical Reports](https://arxiv.org/abs/2605.03103) | MedStruct-S是一个新基准，专注于在未知关键和OCR噪声条件下评估临床报告中的关键发现、关键条件问答和端到端键值提取任务，包含3,582个真实世界标注页面。 |
| [^244] | [When Audio-Language Models Fail to Leverage Multimodal Context for Dysarthric Speech Recognition](https://arxiv.org/abs/2605.02782) | 当前音频-语言模型在构音障碍语音识别中无法有效利用诊断或临床上下文，提示改进微弱甚至有害，而结合混合临床提示的LoRA微调则能带来性能提升。 |
| [^245] | [Interval POMDP Shielding for Imperfect-Perception Agents](https://arxiv.org/abs/2604.20728) | 本文提出一种基于区间POMDP的屏蔽算法，利用有限标注数据构建感知不确定性置信区间，以在运行时提供带高概率有限时域安全保证的保守信念集。 |
| [^246] | [AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems](https://arxiv.org/abs/2604.16804) | AutoOR提出了一种结合合成数据生成和强化学习的可扩展流水线，通过求解器反馈作为奖励信号，使8B参数模型在多个运筹学基准上达到或超越更大前沿模型的性能，尤其在非线性物理动力学问题上实现了突破。 |
| [^247] | [When to Call an Apple Red: Humans Follow Introspective Rules, VLMs Don't](https://arxiv.org/abs/2604.06422) | 本文通过新提出的GCA数据集，发现视觉语言模型在颜色归属决策中系统性地违背自身陈述的内省阈值规则，而人类则更遵守这些规则。 |
| [^248] | [From Multi-Agent to Single-Agent: When Is Skill Distillation Beneficial?](https://arxiv.org/abs/2604.01608) | 本论文提出一种诊断方法，用于确定在多智能体工作流蒸馏为单智能体技能时，何时添加管道指导有益或有害，并通过行为-结果自由度指标解释性能反转现象。 |
| [^249] | [Wildfire Suppression: Complexity, Models, and Instances](https://arxiv.org/abs/2603.29865) | 本文证明了野火扑救资源分配问题在多种图结构上的强NP完全性，提出了先进的混合整数规划方法，并引入基于物理模型的真实实例生成器。 |
| [^250] | [A Framework and Prototype for a Navigable Map of Datasets in Engineering Design and Systems Engineering](https://arxiv.org/abs/2603.15722) | 本文提出了一种系统框架和交互式工具原型，通过多维分类体系构建“EDSE数据集地图”，以解决工程设计与系统工程领域数据集碎片化和不可访问的问题，促进数据发现和复用。 |
| [^251] | [Making Implicit Premises Explicit in Logical Understanding of Enthymemes](https://arxiv.org/abs/2603.06114) | 本文提出了一种结合两个大型语言模型和神经符号推理器的流水线，首次系统地将省略三段论的文本转化为逻辑论证，并生成隐含前提和逻辑公式，以实现逻辑蕴含的显性化理解。 |
| [^252] | [SkillNet: Create, Evaluate, and Connect AI Skills](https://arxiv.org/abs/2603.04448) | 本文提出了SkillNet，一个开放基础设施，通过统一本体论、大规模技能仓库和多功能工具包，系统性地创建、评估和连接AI技能，解决了代理缺乏技能积累和迁移的问题。 |
| [^253] | [Conformal Policy Control](https://arxiv.org/abs/2603.02196) | 本文提出了一种基于保形校准的新框架，利用安全参考策略作为概率调节器，在高风险环境中可证明地控制新策略的探索强度，无需模型类别假设或超参数调优，并在有限样本下提供非单调损失函数的保证。 |
| [^254] | [Whole-Piece Training for Symbolic Music Language Models via Full-Horizon Compressed Recurrence](https://arxiv.org/abs/2602.19816) | 本文提出全时域压缩递归（FHCR）框架，通过压缩KV表示实现符号音乐语言模型的全曲连续训练，并引入KRCU诊断指标来评估长程依赖。 |
| [^255] | [Structure-Informed Estimation for Pilot-Limited MIMO Channels via Tensor Decomposition](https://arxiv.org/abs/2602.04083) | 本文提出了一种结合低秩张量分解与3D U-Net残差学习的混合估计器，有效解决了导频受限MIMO信道估计中的欠定补全问题，并在极端导频稀缺条件下实现了显著的性能提升。 |
| [^256] | [FiLoRA: Focus-and-Ignore LoRA for Controllable Feature Reliance](https://arxiv.org/abs/2602.02060) | FiLoRA通过指令条件化的低秩门控机制，实现了在不改变任务目标的前提下，用自然语言指令直接控制多模态模型对特定特征路径的依赖，从而缓解捷径和虚假相关行为。 |
| [^257] | [TrojanGYM: A Detector-in-the-Loop LLM for Adaptive RTL Hardware Trojan Insertion](https://arxiv.org/abs/2601.17178) | TrojanGYM是一个智能体驱动的LLM框架，通过检测器反馈循环自动生成多样化的硬件木马插入，以暴露和评估检测器的盲点。 |
| [^258] | [Evaluating Music Context Preservation: A Multi-facet Framework for Music Editing Systems](https://arxiv.org/abs/2512.14629) | 本文提出了首个全面的音乐情境保持评估框架MuseCPEval，通过细粒度指标和人类研究验证，系统性地评估音乐编辑系统在编辑过程中保留不变音乐方面的能力。 |
| [^259] | [Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025](https://arxiv.org/abs/2512.14012) | 经验丰富的开发者将AI代理视为生产力工具，但坚持保留设计控制权，通过专业知识策略性地引导代理行为，以确保软件质量。 |
| [^260] | [Large Language Model for Verilog Code Generation: Literature Review and the Road Ahead](https://arxiv.org/abs/2512.00020) | 本文首次全面综述了大型语言模型在Verilog代码生成中的研究进展，填补了该领域缺乏系统性文献综述的空白，并展望了未来发展方向。 |
| [^261] | [CausalProfiler: Generating Synthetic Benchmarks for Rigorous and Transparent Evaluation of Causal Machine Learning](https://arxiv.org/abs/2511.22842) | 本文提出了CausalProfiler，一个能随机生成具有覆盖保证和透明假设的合成因果基准的工具，从而实现对因果机器学习方法的严谨和透明评估。 |
| [^262] | [Jailbreaking in the Haystack](https://arxiv.org/abs/2511.04707) | NINJA是一种通过将良性模型生成内容附加到有害目标上，并利用有害目标位置影响安全性的低资源、可迁移且难以检测的越狱攻击方法，显著提升了多种先进模型的攻击成功率。 |
| [^263] | [Sleeping Kelly](https://arxiv.org/abs/2510.15911) | 本文通过凯利准则重新审视睡美人问题，发现理性决策者应最大化财富增长率而非期望值，从而得出事前哈尔弗派和事后三分派的结论，并避免历时荷兰赌。 |
| [^264] | [Hybrid Reinforcement Learning and Search for Flight Trajectory Planning](https://arxiv.org/abs/2509.04100) | 该方法通过强化学习预计算近最优路径来约束搜索求解器，显著加速飞行轨迹规划，燃油消耗偏差在1%以内，计算速度提升高达50%。 |
| [^265] | [Iterative Flow Matching: Path Correction and Gradual Refinement for Enhanced Generative Modeling](https://arxiv.org/abs/2502.16445) | 本文提出一种可集成到任意生成模型的迭代流匹配方法，通过路径校正和逐步细化来减少幻觉，提升图像生成的稳健性。 |
| [^266] | [`From Prompt to Perturbation': An Adaptive Framework for Voice-Based Jailbreaks on Audio LLMs](https://arxiv.org/abs/2502.00735) | 本文提出了一种自适应越狱攻击框架，能在统一设置下系统评估级联流水线和端到端音频大语言模型，覆盖更广泛的音频攻击空间。 |
| [^267] | [Automated Computational Energy Minimization of ML Algorithms using Constrained Bayesian Optimization](https://arxiv.org/abs/2407.05788) | 本文提出使用约束贝叶斯优化（CBO）在保证模型泛化性能不低于阈值的前提下，自动最小化机器学习算法的训练能耗，并在回归和分类任务上验证了其有效性。 |

# 详细

[^1]: SPADE：自适应合成可执行环境中的自我对弈

    SPADE: Self-Play in Adaptive Synthetic Executable Environments

    [https://arxiv.org/abs/2608.19197](https://arxiv.org/abs/2608.19197)

    该论文提出SPADE框架，通过单个LLM同时作为环境设计器和推理代理进行自我对弈，动态生成可执行训练环境，以解决语言代理训练中目标分布固定的问题。

    

    持续自我改进需要不断扩展的、自我生成的、多样化的、自适应目标池。对于语言代理而言，现有的训练环境池（人工策划、静态合成或冻结验证器）在学习者规模扩大时保持目标分布固定。我们引入了SPADE（自适应合成可执行环境中的自我对弈），这是一种自我对弈强化学习框架，其中单个大型语言模型扮演两个角色：一个环境设计师，负责编写完整的、长视野的训练环境作为可执行代码，并带有OpenAI Gym风格的reset()/step()接口；以及一个推理代理，学习在这些环境中行动。每个环境都是状态化的、多轮次的（包括状态转换、奖励函数和验证代码），因此一个接口即可涵盖推理问题和多步骤代理工具使用。推理代理的遗憾通过其在有特权提示和无特权提示时的奖励差距来估计；在优化这一遗憾信号时，环境设计师...

    arXiv:2608.19197v1 Announce Type: cross  Abstract: Continuous self-improvement requires an ever-expanding pool of self-generated, diverse, adaptive goals. For language agents, existing training environment pools (hand-curated, statically synthesized, or frozen-verifier) keep the goal distribution fixed as the learner scales. We introduce SPADE (Self-Play in Adaptive Synthetic Executable Environments), a self-play RL framework in which a single LLM plays two roles: an Environment Designer that writes complete, long-horizon training environments as executable code with an OpenAI Gym-style reset()/step() interface, and a Reasoning Agent that learns to act in them. Each is a stateful, multi-turn environment (state transitions, reward functions, and verification code), so one interface spans reasoning problems and multi-step agentic tool use. The Reasoning Agent's regret is estimated using the gap between its reward with and without privileged hints; in optimizing this regret signal the Env
    
[^2]: ADEPT：通过预训练与后训练加速灵巧操作

    ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning

    [https://arxiv.org/abs/2608.19182](https://arxiv.org/abs/2608.19182)

    ADEPT通过预训练通用灵巧操作策略并采用稳定的后训练方案，显著加速了高自由度机器人从原始感知学习长时程任务的过程，避免了重复学习并提升了迁移能力。

    

    arXiv:2608.19182v1 公告类型：交叉 摘要：我们引入了“通过预训练加速灵巧操作”（ADEPT），这是一个大规模强化学习（RL）框架，用于学习跨高自由度（DoF）机器人实体可模拟到现实迁移的灵巧操作，能够直接从原始视觉-触觉感知解决长时程任务。ADEPT在通用物体重定位任务上预训练灵巧策略，然后以此预训练行为作为先验，对下游策略进行后训练。ADEPT使得在多指机器人上学习新行为成为可能，这些行为若从头开始发现则通常困难，并避免了为每个新下游任务重复学习相同技能集。预训练策略零样本执行下游任务的重定位阶段，但朴素RL微调在迁移过程中会迅速退化这一能力。我们通过结合行为克隆蒸馏、评论家预热和保守的在线更新的稳定后训练方案来解决这一问题。为安全地...

    arXiv:2608.19182v1 Announce Type: cross  Abstract: We introduce Accelerating Dexterity via Pre-Training (ADEPT), a large-scale reinforcement learning (RL) framework for learning sim-to-real transferable dexterity across high degree-of-freedom (DoF) robot embodiments that can solve long-horizon tasks directly from raw visuo-tactile perception. ADEPT pretrains a dexterous policy on a generic object reposing task, then post-trains downstream policies with this pretrained behavior as a prior. ADEPT enables learning new behaviors that are otherwise difficult to discover from scratch on multi-fingered robots and avoids learning the same set of skills over again for every new downstream task. The pretrained policy zero-shots the reposing phase of downstream tasks, but na\"ive RL fine-tuning rapidly degrades this capability during transfer. We address this with a stable post-training recipe combining behavior-cloning distillation, critic warm-up, and conservative on-policy updates. To safely e
    
[^3]: 超越教师似然：基于群体校准的在线策略蒸馏用于长上下文推理

    Beyond Teacher Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning

    [https://arxiv.org/abs/2608.19181](https://arxiv.org/abs/2608.19181)

    本文提出GC-OPD方法，通过在rollout组内校准验证器奖励与轨迹级OPD分数之间的差异，解决长上下文推理中教师指导与任务验证不一致的问题。

    

    在线策略蒸馏（OPD）通过来自更强教师的密集token级指导，在学生的自身响应上训练学生。然而，在长上下文任务中，token级教师支持可能偏向于局部看似合理的响应，这些响应会遗漏分布在输入中的证据或违反全局任务约束。相比之下，任务特定验证器在响应级别评估任务完成情况，并可能返回反映部分成功的分级奖励。我们在两个代表性的长上下文证据聚合任务上，对固定响应诊断了这种不匹配。在更长的输入范围内，轨迹级OPD分数与验证器奖励逐渐变得不一致，表明教师与验证器之间存在分歧。受此观察启发，我们引入了群体校准在线策略蒸馏（GC-OPD）。GC-OPD在每个rollout组内分别归一化验证器奖励和轨迹级OPD分数，并使用它们的差异作为sigmoid函数的输入。

    arXiv:2608.19181v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student on its own responses using dense token-level guidance from a stronger teacher. In long-context tasks, however, token-level teacher support can favor locally plausible responses that omit evidence distributed across the input or violate global task constraints. Task-specific verifiers, in contrast, evaluate task completion at the response level and may return graded rewards that reflect partial success. We diagnose this mismatch on fixed responses from two representative long-context evidence-aggregation tasks. Across longer input ranges, trajectory-level OPD scores become progressively less aligned with verifier rewards, indicating teacher-verifier disagreement. Motivated by this observation, we introduce Group-Calibrated On-Policy Distillation (GC-OPD). GC-OPD separately normalizes verifier rewards and trajectory-level OPD scores within each rollout group and uses their difference as a sig
    
[^4]: 通过声音模仿查询声音的微调策略

    Finetuning Strategies for Querying Sounds by Vocal Imitation

    [https://arxiv.org/abs/2608.19174](https://arxiv.org/abs/2608.19174)

    本文提出并验证了两种微调策略（冻结CED编码器的对比学习和MobileNetV3的联合对比-三元组学习），在声音模仿查询音效任务中取得了冠军成绩。

    

    本技术报告描述了我们提交给AES AIMLA 2025挑战赛的获胜方案，该挑战赛旨在通过声音模仿查询音效。我们研究了两种互补的微调策略：使用冻结的预训练CED编码器进行对比学习，以及使用MobileNetV3编码器结合半困难负样本的联合对比-三元组学习。本报告已为后人更新，包含了挑战赛后发布的细节。

    arXiv:2608.19174v1 Announce Type: cross  Abstract: This technical report describes our winning submission to the AES AIMLA 2025 Challenge on querying sound effects by vocal imitation. We investigate two complementary fine-tuning strategies: contrastive learning with a frozen, pretrained CED encoder, and joint contrastive-triplet learning with semi-hard negatives using a MobileNetV3 encoder. This report has been updated for posterity to include details released after the challenge.
    
[^5]: 可解释人工智能预测2026年中国中部夏季干旱异常

    Interpretable AI predicts a 2026 summer dry anomaly in central China

    [https://arxiv.org/abs/2608.19163](https://arxiv.org/abs/2608.19163)

    该研究利用深度学习模型将动力环流预测转化为降水估计，并借助可解释性分析（LRP）揭示了2026年中国中部夏季干旱异常及其背后的环流驱动机制，实现了对区域降水异常的高技能季节预测。

    

    arXiv:2608.19163v1 公告类型：交叉 摘要：季节性降水异常主要受大气环流调控，动力模型对此的预测可靠性高于对降水本身的预测。在此，我们采用一种深度学习模型，将动力环流预测转化为降水估计。从3月至5月初始化的预测一致表明，2026年夏季中国中部将出现干旱异常。回顾性评估显示，在相似年份中预测技能更高，这些年份往往具有从中部赤道太平洋前冬持续至夏季的增暖特征。这种增暖有利于西北太平洋-南海-华南地区上空出现异常气旋性环流，从而引发偏北风和水分辐散，共同抑制中国中部降水。支持这一机制的是，层间相关性传播（LRP）独立识别出这些偏北风是主导因素。

    arXiv:2608.19163v1 Announce Type: cross  Abstract: Seasonal precipitation anomalies are largely regulated by atmospheric circulation, which dynamical models predict with greater reliability than precipitation itself. Here, we employ a deep learning model that translates dynamical circulation predictions into precipitation estimates. Predictions initialized from March to May consistently indicate a dry anomaly over central China in summer 2026. Retrospective evaluations revealed higher predictive skill in the analogue years, which also tended to feature central equatorial Pacific warming persisting from the preceding winter into summer. This warming favors an anomalous cyclonic circulation over the western North Pacific-South China Sea-South China region, which induces northerly winds and moisture divergence that jointly suppress rainfall over central China. Supporting this mechanism, layer-wise relevance propagation (LRP) independently identifies these northerly winds as the dominant d
    
[^6]: 超越文本记录：检测潜在多智能体通信中的隐蔽协调

    Beyond the Transcript: Detecting Covert Co ordination in Latent Multi-Agent Communication

    [https://arxiv.org/abs/2608.19161](https://arxiv.org/abs/2608.19161)

    本文提出了VLA框架，通过激活感知监控和反事实分析，首次实现了对多智能体潜在通信渠道中隐蔽有害协调的检测与引导。

    

    arXiv:2608.19161v1 公告类型：新论文 摘要：语言模型智能体可以通过在公开文本记录中不可见的连续隐藏状态进行通信，从而为隐蔽的有害协调创造了机会。我们提出了可验证潜在对齐（VLA）框架，这是一种基于激活感知的监控与引导框架，用于监测和操控这些私有通信渠道。对于每个被监控的决策，VLA通过共享事件标识符将私有潜在状态记录和渠道状态与最终公开行为关联起来，从而实现匹配的因果分析。我们的第一个贡献是一个仅使用中性数据的三层监控器，结合了表示异常检测、反事实行为分布影响和稀疏自编码器解释支持。我们的第二个贡献是一个可引导性框架，涵盖黑盒行为指令和白盒匹配中性反事实。我们的第三个贡献是在一个受控的多智能体拍卖基准上进行的评估，该基准覆盖了同质智能体场景。

    arXiv:2608.19161v1 Announce Type: new  Abstract: Language-model agents can communicate through continuous hidden states that are invisible in public transcripts, creating opportunities for covert harmful coordination. We introduce Verifiable Latent Alignments (VLA), an activation-aware framework for monitoring and steering these private communication channels. For every monitored decision, VLA links the private latent-state record and channel status to the resulting public action using a shared event identifier, enabling matched causal analysis. Our first contribution is a neutral-only three-layer monitor combining representation anomaly detection, counterfactual action-distribution influence, and sparse-autoencoder interpretation support. Our second contribution is a steerability framework spanning black-box behavioral instructions and white-box matched-neutral counterfactuals. Our third contribution is an evaluation on a controlled multi-agent auction benchmark covering homogeneous a
    
[^7]: 预编译流水线分片用于英特尔AI PC机群上的分布式大语言模型推理

    Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets

    [https://arxiv.org/abs/2608.19147](https://arxiv.org/abs/2608.19147)

    通过将大语言模型按层预编译为OpenVINO分片并利用流水线并行，在多个英特尔AI PC上实现分布式推理，通过注入beam_idx Gather触发GPU优化和投机解码，达到与单体推理相当的性能。

    

    arXiv:2608.19147v1 公告类型：交叉 摘要：现代英特尔AI PC配备了功能强大的集成GPU和NPU，拥有16GB以上的统一内存，但这些设备大部分时间处于闲置状态。这些内存不足以容纳像700亿参数的大语言模型这样的大型模型。我们证明，通过普通网络协作的一小批AI PC，可以服务超过任何单台设备能力的模型。我们采用流水线并行：模型按层拆分为每阶段的分片，每个分片预编译为OpenVINO图，这样每台机器运行一个分片并将激活传递给下一个分片。三种技术使其足够快速以实用。首先，我们恢复了未拆分模型的速度：朴素的每阶段导出性能远低于单体推理，因为它缺少OpenVINO GPU优化，而向每个分片注入beam_idx Gather可触发该优化（IndirectKVCache融合），使分片性能达到一致。其次，我们利用有状态OpenVINO模型上的投机解码。

    arXiv:2608.19147v1 Announce Type: cross  Abstract: Modern Intel AI PCs ship capable integrated GPUs and NPUs with 16+ GB of unified memory, and they spend considerable time idle. That is not enough memory to fit a large model such as a 70B-parameter LLM. We show that a handful of AIPCs, working together over an ordinary network, can serve models beyond the capability of any single one. We use pipeline parallelism: a model is split by layer into per-stage shards, each pre-compiled into an OpenVINO graph, so that every machine runs one shard and passes activations to the next. Three techniques make this fast enough to be useful. First, we recover the speed of the unsplit model: a naive per-stage export runs well below monolithic inference because it misses an OpenVINO GPU optimization, and injecting a beam_idx Gather into each shard triggers that optimization (the IndirectKVCache fusion) and brings the shards to parity. Second, we leverage speculative decoding on stateful OpenVINO models
    
[^8]: 随机机器的分组：精度而非能力，作为AI系统的前沿指标

    Grouping the Stochastic Machine: Precision, Not Capability, as the Frontier Metric for AI Systems

    [https://arxiv.org/abs/2608.19140](https://arxiv.org/abs/2608.19140)

    本文提出前沿AI系统的关键区别在于输出精度（重复请求下结果的一致性），而非传统能力指标，并论证了精度可低成本、无循环地测量。

    

    arXiv:2608.19140v1 公告类型：新公告 摘要：前沿语言模型在能力上进行比较、营销和基准测试——即它们最佳或平均输出能达到的水平。我认为这衡量了错误的维度。这些模型已经达到精度饱和：它们的平均输出落在目标上。在实践中，现在区分一个系统与另一个系统的是精度：在重复、相同的请求中，输出围绕该目标的紧密集中程度。借用射击手的区分，能力是平均弹着点的位置；可靠性是弹着群的散布大小。我提出三个主张。第一，精度而非能力，是系统之间的前沿区分因素，而基准测试文化系统性地未能测量这一点，只报告集中趋势而非离散度。第二，精度是可测量的，成本低廉且无循环性，只需在固定温度下多次运行一组确定性评分的任务，并计算每个任务结果的一致性——无需——

    arXiv:2608.19140v1 Announce Type: new  Abstract: Frontier language models are compared, marketed, and benchmarked on capability -- what their best or average output can achieve. I argue this measures the wrong axis. The models have saturated accuracy: their mean output lands on the target. What now separates one system from another in practice is precision: how tightly concentrated their outputs are around that target across repeated, identical requests. Borrowing the marksman's distinction, capability is where the average shot lands; reliability is the size of the group. I make three claims. First, precision, not capability, is the frontier differentiator between systems, and benchmark culture systematically fails to measure it, reporting central tendency rather than spread. Second, precision is measurable, cheaply and without circularity, by running a fixed suite of deterministically scored tasks many times at fixed temperature and computing the per-task consistency of outcomes -- no
    
[^9]: 叶值作为坐标：梯度提升集成模型的精确对比解释

    Leaf Values as Coordinates: Exact Contrastive Explanation for Gradient-Boosted Ensembles

    [https://arxiv.org/abs/2608.19127](https://arxiv.org/abs/2608.19127)

    通过将梯度提升模型的叶值视为坐标，实现了精确且可追溯的对比解释，为反事实分析提供了无需额外拟合的严谨方法。

    

    摘要：arXiv:2608.19127v1 公告类型：交叉 摘要：梯度提升集成模型通过每棵树累加一个叶值来进行预测。将这些值视为坐标而非中间结果，每个实例便成为R^M空间中的一个点，模型在该空间上线性作用：得分即坐标之和。这一视角的小小转变使得对比解释变得精确。两个实例之间的差异是一个向量，在它们共享叶子的地方为零，因此被拒绝的申请人与被接受的申请人之间的差距由少数几个坐标承载，每个坐标都可追溯至真实树中的真实分裂。无需拟合、采样或假设特征可加性——可加性已然存在于正确的空间中。我们基于这一表示构建了一种反事实解释方法，并在五个表格数据集上通过重复交叉验证进行评估。其推荐结果重建模型自身决策的精度达到6.2×10^-15，使得审计人员能够重新核对算术。

    arXiv:2608.19127v1 Announce Type: cross  Abstract: A gradient-boosted ensemble predicts by summing one leaf value per tree. Read   those values as coordinates rather than as intermediate results, and every   instance becomes a point in R^M on which the model acts linearly: the score is   the sum of the coordinates.   This small change of view makes contrastive explanation exact. The difference   between two instances is a vector that is identically zero wherever they share   a leaf, so the gap between a rejected applicant and an accepted one is carried   by a handful of coordinates, each traceable to a real split in a real tree.   Nothing is fitted, sampled, or assumed additive in features -- the additivity   is already there, in the right space.   We build a recourse method on this representation and evaluate it on five   tabular datasets under repeated cross-validation. Its recommendation   reconstructs the model's own decision to 6.2 x 10^-15, so an auditor can   re-check the arithm
    
[^10]: 调谐随机机器：人类-AI工程的系统工程师操作模型

    Tuning the Stochastic Machine: A Systems Engineer's Operating Model for Human-AI Engineering

    [https://arxiv.org/abs/2608.19125](https://arxiv.org/abs/2608.19125)

    本文提出将LLM系统视为需要系统工程师操作纪律的随机机器，并基于映射失败推导出七项原则，核心是持久化错误纠正的循环机制。

    

    arXiv:2608.19125v1 公告类型：新 摘要：当专家纠正LLM助手的错误时，纠正通常随会话结束而消失，错误类别会再次出现。我认为这是一个操作问题，而非工具问题：持久化纠正的机制已经存在并正在部署，但管理它们的纪律——带有来源的版本控制、复发监控、反指标、过时规则的淘汰——却缺失。作为一位拥有三十年经验的系统工程师，我将LLM堆栈映射到我职业已经操作的机器（固化硅、固件、可加载模块、持久配置、易失内存），识别映射失败之处（随机生成、仅概率性绑定的配置、默认无通用淘汰（验证）阶段），并从失败中推导出以错误循环为核心的操作纪律的七项原则。我实践中的三个案例说明了该机制，其中包括一个控制案例。

    arXiv:2608.19125v1 Announce Type: new  Abstract: When an expert corrects an LLM assistant's error, the correction usually dies with the session, and the error class returns. I argue this is an operations problem, not a tooling problem: mechanisms for persisting corrections exist and are shipping, but the discipline for governing them -- versioning with provenance, recurrence monitoring, counter-metrics, retirement of stale rules -- does not. Writing as a systems engineer of thirty years, I map the LLM stack onto the machines my profession already operates (frozen silicon, firmware, loadable modules, persistent configuration, volatile memory), identify where the mapping fails (stochastic generation, configuration that binds only probabilistically, no general-purpose retirement (verification) stage by default), and derive from the failures a seven-principle operating discipline with an error loop at its core. Three cases from my own practice illustrate the mechanism, among them a control
    
[^11]: 拦截袋鼠效应：使用构造词典、主动探测及大型语言模型作为信息提供者和假设提出者的实验性天体语言学

    Intercepting the Kangaroo: Experimental Astrolinguistics with Constructed Lexicons, Active Probing, and Large Language Models as Informants and Hypothesis Proposers

    [https://arxiv.org/abs/2608.19124](https://arxiv.org/abs/2608.19124)

    该论文通过构造不兼容词典和主动探测协议，首次将天体语言学实验化，成功拦截了“袋鼠效应”（翻译不确定性），实现了零未检测误译。

    

    摘要：arXiv:2608.19124v1 公告类型：交叉 摘要：天体语言学——与以不同于我们的方式对现实进行分类的智能体进行交流——自弗洛伊登塔尔的林科斯（1960）以来一直纯属推测。我们将其变为实验性研究。两个具有故意不兼容构造词典的语言模型（一个编码形状、颜色和运动；另一个将颜色与运动融合、编码奇偶性且缺乏形状）作为具有完整地面实况的信息提供者，而一个完全脚本化的编排器在两个类别系统之间进行翻译。核心失败模式是袋鼠效应：词语错误地附着于错误所指——奎因的翻译不确定性，已操作化。在400多次模拟和实时运行中，一种结合跨情境消除、预注册预测探针、主动场景选择、更严格的恢复轮次和隔离的协议，在测试条件下未产生未检测到的误译，并超过了被动基线的协变量。

    arXiv:2608.19124v1 Announce Type: cross  Abstract: Astrolinguistics -- communication with minds that categorize reality differently from ours -- has been purely speculative since Freudenthal's Lincos (1960). We make it experimental. Two language models with deliberately incompatible constructed lexicons (one encoding shape, color, and motion; the other fusing color with motion, encoding parity, and lacking shape) serve as informants with complete ground truth, while a fully scripted orchestrator translates between the two category systems. The central failure mode is the kangaroo effect: the silent attachment of a word to the wrong referent -- Quine's indeterminacy of translation, operationalized. Across 400+ simulated and live runs, a protocol combining cross-situational elimination, pre-registered predictive probes, active scene selection, a stricter recovery round, and quarantine produced no undetected mistranslations under the tested conditions and exceeded a passive baseline's cov
    
[^12]: PGFS++：在合成与多样性约束下的分子性质改进

    PGFS++: Molecular Property Improvement under Synthesis and Diversity Constraints

    [https://arxiv.org/abs/2608.19121](https://arxiv.org/abs/2608.19121)

    本文提出PGFS++，通过直接嵌入反应物和引入多样性约束，在保持合成可行性的同时改进分子性质，并解决了PGFS+中因奖励黑客导致的输出多样性崩溃问题。

    

    摘要：改进分子性质，如药物相似性或结合亲和力，是早期药物发现中的一项重复性任务。然而，在无约束化学空间中优化的分子，如果无法合成，则其实际价值有限。前向合成策略梯度（PGFS）是一种用于分子改进的合成感知强化学习方法，但其使用反应物嵌入预测使得反应物选择变得间接，我们证明这限制了学习效果。我们首先开发了PGFS+，其中反应模板和第二反应物通过可训练的嵌入查找表表示。结合更有效的评分函数和强化学习算法，PGFS+显著改进了目标性质。然而，它暴露了一种奖励黑客失败模式：强大的反应物搜索可以将多样化的输入分子映射到同一高奖励的磁铁分子，从而提高奖励但导致输出多样性崩溃。

    arXiv:2608.19121v1 Announce Type: cross  Abstract: Improving molecular properties, such as drug-likeness or binding affinity, is a recurring task in early-stage drug discovery. However, molecules optimized in an unconstrained chemical space have limited practical value if they cannot be synthesized. Policy Gradient for Forward Synthesis (PGFS) is a synthesis-aware reinforcement learning method for molecular improvement, but its use of reactant embedding prediction makes reactant selection indirect, which, as we show, limits learning effectiveness. We first develop PGFS+, in which reaction templates and second reactants are represented by trainable embedding lookup tables. Combined with a more effective scoring function and RL algorithm, PGFS+ significantly improves the desired property. However, it exposes a reward-hacking failure mode: a powerful reactant search can map diverse input molecules to the same high-reward magnet molecule, improving the reward while collapsing the output di
    
[^13]: 将连续时间序列离散化以用于掩码扩散训练的插补方法

    Discretizing Continuous Time Series for Imputation with Masked Diffusion Training

    [https://arxiv.org/abs/2608.19119](https://arxiv.org/abs/2608.19119)

    本文提出MDTIM模型，通过将时间序列离散化并采用掩码扩散训练，实现结构分离的插补，直接预测原始信号，克服了传统方法在表示空间和学习目标上的局限性。

    

    arXiv:2608.19119v1 公告类型：交叉 摘要：时间序列插补是可靠时间序列分析的关键领域，但由于真实世界数据的复杂时间动态和噪声，它仍然具有挑战性。然而，现有方法表现出两个局限性：缺失值和观测值被嵌入到相同的表示空间中，而没有明确的结构分离；基于连续扩散的方法被训练用于预测添加的噪声，而非原始信号。为解决这些问题，我们提出了掩码扩散时间序列插补模型（MDTIM），该模型利用掩码扩散模型的训练范式进行插补任务。MASK标记在结构上与有效观测正交，模型直接预测原始值，从而自然地将表示和学习目标与插补任务对齐。为弥合离散掩码扩散与时间序列连续、有序特性之间的差距，我们进一步...

    arXiv:2608.19119v1 Announce Type: cross  Abstract: Time series imputation is a crucial area for reliable time series analysis, yet it remains challenging due to the complex temporal dynamics and noise of real-world data. Existing approaches, however, exhibit two limitations: missing and observed values are embedded within the same representation space without explicit structural separation, and continuous diffusion-based methods are trained to predict added noise rather than the original signal. To address these, we propose the Masked Diffusion Time-series Imputation Model (MDTIM), which leverages the training paradigm of masked diffusion model for imputation tasks. The MASK token is structurally orthogonal to valid observations, and the model directly predicts the original values, naturally aligning both the representation and the learning objective with the imputation task. To bridge the gap between discrete masked diffusion and the continuous, ordinal nature of time series, we furth
    
[^14]: 开放M-OPD：诊断与修复多教师同策略蒸馏中的能力失衡问题

    Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2608.19098](https://arxiv.org/abs/2608.19098)

    本文通过受控基准诊断出多教师同策略蒸馏中的能力整合失衡问题，指出其根源并非梯度冲突，而是严重的优化失衡，并提出修复方法以提升整合效率。

    

    摘要：arXiv:2608.19098v1 公告类型：交叉 摘要：多教师同策略蒸馏（M-OPD）已成为一种有前景的范式，通过密集的、令牌级别的奖励监督，将领域专长的强化学习（RL）专家整合为一个通用的学生模型。尽管其在实际应用中取得了成功，但控制多教师能力整合的优化动力学仍未被充分理解，且缺乏开放、严格可复现的配方。在本工作中，我们在SmolLM3-3B-Base上建立了一个受控的M-OPD基准，使用Oracle路由，将能力整合与路由歧义分离开来。我们的研究揭示了一个显著的能力整合差距：标准的M-OPD相对于领域路由的Oracle集成，仅捕获了可用性能提升空间的35.6%，而指令跟随等简洁任务则遭受严重退化并过早停滞。关键在于，我们表明这种失败并非源于梯度冲突，而是源于严重的m（此处原文截断）

    arXiv:2608.19098v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (M-OPD) has emerged as a promising paradigm for consolidating domain-specialized reinforcement learning (RL) experts into a single generalist student via dense, token-level reward supervision. Despite its practical success, the optimization dynamics governing multi-teacher capability integration remain poorly understood, and open, rigorously reproducible recipes are conspicuously lacking. In this work, we establish a controlled M-OPD benchmark on SmolLM3-3B-Base with oracle routing, isolating capability integration from routing ambiguity. Our investigation reveals a pronounced capability integration gap: standard M-OPD captures only 35.6% of the available headroom relative to a domain-routed oracle ensemble, with concise tasks such as instruction following suffering severe degradation and premature stagnation. Crucially, we show that this failure stems not from gradient conflict, but from a severe m
    
[^15]: 通过预NMS预测分布偏移检测目标检测中的后门

    Detecting Backdoors in Object Detection via Pre-NMS Prediction Distribution Shift

    [https://arxiv.org/abs/2608.19088](https://arxiv.org/abs/2608.19088)

    本文提出DistScan框架，通过检测模型在干净输入上的预NMS预测类别分布偏移，实现无需触发器反转即可可靠识别目标检测模型中的后门，尤其适用于场景级攻击。

    

    arXiv:2608.19088v1 公告类型：交叉 摘要：部署在安全关键应用中的目标检测模型仍然容易受到后门攻击，当隐藏触发器存在时，这些攻击会导致目标性异常行为。现有检测方法要么依赖于触发器反转，要么利用特定于架构的假设，关键是，代表性的现有方法无法可靠地泛化到场景级攻击，其中单个触发器同时诱导场景中所有对象的异常行为。我们提出了DistScan，一个基于简单但此前未探索的观察的后门检测框架：后门注入系统性地将模型的预NMS预测类别分布从训练类别频率上偏移，即使在没有任何触发器的干净输入上也是如此。DistScan在干净验证集上聚合中间类别预测，如果结果分布显著偏离训练类别，则将模型标记为后门。

    arXiv:2608.19088v1 Announce Type: cross  Abstract: Object detection models deployed in safety-critical applications remain vulnerable to backdoor attacks that cause targeted misbehaviors when a hidden trigger is present. Existing detection methods either rely on trigger inversion or exploit architecture-specific assumptions, and critically, representative existing methods fail to generalize reliably to scene-level attacks, where a single trigger induces anomalous behavior across all objects in the scene simultaneously. We present DistScan, a backdoor detection framework based on a simple but previously unexploited observation: backdoor injection systematically shifts a model's pre-NMS prediction class distribution away from its training class frequencies, even on clean inputs without any trigger present. DistScan aggregates intermediate class predictions over a clean validation set and flags a model as backdoored if the resulting distribution deviates significantly from the training cl
    
[^16]: DA-WAM：面向驾驶世界模型的决策对齐未来潜变量

    DA-WAM: Decision-Aligned Future Latents for Driving World Models

    [https://arxiv.org/abs/2608.19085](https://arxiv.org/abs/2608.19085)

    DA-WAM通过统一预测表示学习、行动条件未来建模和轨迹评分，在单一决策目标下实现未来表示与规划优化的共同演进，从而确保预测未来直接指导轨迹选择。

    

    arXiv:2608.19085v1 公告类型：交叉 摘要：预测场景在自我行为下如何演变是安全自动驾驶的基础，但世界模型在决策中的全部潜力仍未实现。关键挑战在于确保未来建模不仅是预测性的，而且是决策信息丰富的：预测的未来必须直接塑造选择哪条轨迹。现有方法将未来表示学习与规划优化解耦，或在轨迹候选之间共享预测状态，从而稀释了应指导选择的行动特定后果。为弥合这一差距，我们提出DA-WAM，一个在单一决策目标下统一预测表示学习、行动条件未来建模和轨迹评分的框架。DA-WAM通过在线编码器和稳定的动量目标在整个规划器优化过程中保持预测监督，使未来表示能够共同演进。

    arXiv:2608.19085v1 Announce Type: cross  Abstract: Anticipating how scenes evolve under ego actions is fundamental to safe autonomous driving, yet the full potential of world models for decision-making remains unrealized. The critical challenge lies in ensuring that future modeling is not merely predictive, but decision-informative: the predicted future must directly shape which trajectory is selected. Existing approaches decouple future representation learning from planning optimization, or share predicted states across trajectory candidates, thereby diluting the action-specific consequences that ought to guide selection. To bridge this gap, we propose DA-WAM, a framework that unifies predictive representation learning, action-conditioned future modeling, and trajectory scoring under a single decision-making objective. DA-WAM maintains predictive supervision throughout planner optimization via an online encoder and a stable momentum target, allowing future representations to co-evolve
    
[^17]: 重新权衡证据：校准词元级序数视觉证据以缓解大型视觉-语言模型中的幻觉

    ReWEIGH the Evidence: Calibrating Token-Level Ordinal Visual Evidence to Mitigate Hallucinations in Large Vision-Language Models

    [https://arxiv.org/abs/2608.19075](https://arxiv.org/abs/2608.19075)

    ReWEIGH通过校准词元级序数视觉证据，无需训练即可有效缓解大型视觉-语言模型中的幻觉现象。

    

    大型视觉-语言模型（LVLMs）经常产生幻觉，生成输入图像不支持的内容。在解码过程中防止此类内容，需要一种针对候选词元的特定度量，以评估图像对当前考虑的词元的支持强度。模型的视觉词元状态提供了这一证据的自然来源，因为通过输出头投影每个状态，可以揭示该位置偏好的词汇项。这些逐位置的读出不能直接合并，因为它们的概率大小在视觉位置之间不可比较。词汇排名提供了一种尺度不变的合并基础，但词元在典型的基于排名的证据上仍存在系统性差异。我们提出ReWEIGH，一种无需训练的解码干预方法，它跨视觉位置聚合这些排名，并将每个候选与从未标注图像中估计的词元特定参考进行比较。在推理时，ReWEIGH缓存了...

    arXiv:2608.19075v1 Announce Type: cross  Abstract: Large vision-language models (LVLMs) often hallucinate, generating content that the input image does not support. Preventing such content during decoding calls for a candidate-specific measure of how strongly the image supports the token under consideration. The model's visual-token states offer a natural source of this evidence because projecting each state through the output head reveals which vocabulary items that position favors. These position-wise readouts cannot be pooled directly because their probability magnitudes are not comparable across visual positions. Vocabulary ranks provide a scale-invariant basis for pooling, but tokens still differ systematically in their typical rank-based evidence. We propose ReWEIGH, a training-free decoding intervention that aggregates these ranks across visual positions and compares each candidate with a token-specific reference estimated from unlabeled images. At inference, ReWEIGH caches the 
    
[^18]: 演化不确定性下的稳健风险：熵值风险价值的水斯坦对应物

    Robust Risk Under Evolving Uncertainty: A Wasserstein Counterpart of the Entropic Value-at-Risk

    [https://arxiv.org/abs/2608.19073](https://arxiv.org/abs/2608.19073)

    本文提出水斯坦熵值风险价值，用最优传输球替代相对熵球，以覆盖熵值风险价值忽略的可达灾难，并推导出闭式稳健动态规划算子，其谨慎度随信念更新而自适应调整。

    

    摘要：一个仍在学习环境的智能体应在无知时谨慎，在自信时大胆。熵值风险价值通过一个稳健优化恒等式捕捉了这一点——置信水平固定了替代模型的相对熵球的半径——但该球无法触及名义模型认为不可能的灾难，而这正是安全智能体必须对冲的。我们转而使用最优传输球，并研究其诱导的相干风险度量，即水斯坦熵值风险价值。它有一个与熵公式镜像的变分对偶（逆温度变为传输价格），在风险层级中占据明确位置，并证明能解释熵度量忽略的可达灾难；我们在数值上验证了两种对偶性。通过信念熵驱动传输半径，我们得到一个闭式稳健动态规划算子，其谨慎度随信念锐化而收缩，并附带一个c。

    arXiv:2608.19073v1 Announce Type: new  Abstract: An agent still learning its environment should be cautious while ignorant and bold once confident. The entropic value-at-risk captures this through a robust-optimization identity---a confidence level fixes the radius of a relative-entropy ball of alternative models---but that ball cannot reach catastrophes the nominal deems impossible, precisely what a safe agent must hedge. We instead use an optimal-transport ball and study the coherent risk measure it induces, the Wasserstein entropic value-at-risk. It has a variational dual mirroring the entropic formula (an inverse temperature becomes a transport price), occupies a definite place in the risk hierarchy, and provably accounts for the reachable catastrophes the entropic measure ignores; we verify both dualities numerically. Driving the transport radius by belief entropy then yields a closed-form robust dynamic-programming operator whose caution contracts as the belief sharpens, with a c
    
[^19]: AI后训练中缺失的是什么：一项实证分析

    What is Missing from AI Post-Training AI: An Empirical Analysis

    [https://arxiv.org/abs/2608.19072](https://arxiv.org/abs/2608.19072)

    本文通过实证分析发现，AI后训练代理在策略层面缺乏灵活性，其训练策略在初期被锁定，且主要受经验缺失、指导缺失和推理不足影响，而非单纯执行能力不足。

    

    大型语言模型（LLM）代理现在可以端到端地后训练一个LLM。它们能够编写代码、启动训练、评估检查点并提升下游性能，这引发了AI为AI（AI-for-AI）的前景。我们认为，这种图景混淆了两种不同的能力：执行级能力，即在选定的训练策略内进行迭代；以及策略级能力，即随着实验证据的积累而修订高层判断。通过分析大量公开的后训练轨迹语料库，我们发现，在不同任务中，代理的训练策略在最初阶段就被锁定，剩余的整个预算都花费在所选策略内的局部调整上。接着，我们检验了三种自然的解释——经验缺失、指导缺失和推理不足——并采用逐步升级的干预措施。大量实验表明：（1）经验驱动的脚手架能改善跨b（原文截断）的执行能力。

    arXiv:2608.19072v1 Announce Type: new  Abstract: Large language model (LLM) agents can now post-train an LLM end-to-end. They can write code, launch training, evaluate checkpoints, and improve downstream performance, raising the prospect of AI-for-AI. We argue that this picture conflates two distinct capabilities: execution-level capability, iterating within a selected training strategy; and strategy-level capability, revising the high-level judgment as experimental evidence accumulates. Analyzing a large corpus of publicly released post-training trajectories, we find that across different tasks, the agent's training strategy is locked in at the very beginning, and the entire remaining budget is spent on local adjustments within the selected strategy. We then examine three natural explanations--missing experience, missing guidance, and insufficient reasoning--with escalating interventions. Extensive experiments show that (1) an experience-driven scaffold improves execution across the b
    
[^20]: GS-VLA：基于高斯泼溅的即插即用视角规范化，用于冻结的VLA策略

    GS-VLA: Plug-and-Play Viewpoint Canonicalization for Frozen VLA Policies via Gaussian Splatting

    [https://arxiv.org/abs/2608.19066](https://arxiv.org/abs/2608.19066)

    本文提出一种基于3D高斯泼溅的即插即用框架，无需重新训练VLA策略即可有效应对视角偏移，显著提升部署鲁棒性。

    

    本文提出了一个轻量级、即插即用的框架，旨在提高视觉-语言-动作（VLA）策略对视角变化的鲁棒性，且无需重新训练策略。据我们所知，这是首次直接利用基于3D高斯的新视角合成技术来适应VLA策略的观测空间。当前VLA性能依赖于训练和部署时相机配置相同的隐含假设。实验表明，即使相机安装位置发生微小位移，在LIBERO基准上的成功率在最坏情况下可能从约90%降至约10%。以往方法，如大规模微调或生成式数据增强，计算成本高昂且存在灾难性遗忘的风险。为解决此问题，我们将视角变化重新表述为局部新视角合成问题。在局部性假设下，相机扰动保持在有界小范围内。

    arXiv:2608.19066v1 Announce Type: cross  Abstract: This paper proposes a lightweight, plug-and-play framework that improves robustness to viewpoint shifts in Vision-Language-Action (VLA) policies without policy retraining. To our knowledge, this is the first approach to directly leverage 3D Gaussian-based novel-view synthesis for observation-space adaptation in VLA policies. Current VLA performance relies on the implicit assumption that training and deployment camera configurations are identical. Our experiments show that even a small displacement of the camera mount can reduce the success rate on the LIBERO benchmark from about 90% to about 10% in the worst case. Prior approaches, such as large-scale fine-tuning or generative data augmentation, are computationally expensive and risk catastrophic forgetting. To address this, viewpoint shifts are reformulated as a localized novel-view synthesis problem. Under a Locality assumption, that camera perturbations remain within a small bounded
    
[^21]: 尤里卡：面向科学发现的任务条件化元代理编排

    Eureka: Task-Conditioned Meta-Agent Orchestration for Scientific Discovery

    [https://arxiv.org/abs/2608.19047](https://arxiv.org/abs/2608.19047)

    尤里卡提出了一种任务条件化的元代理架构，通过动态义务图和宏代理编排，实现了高效、可验证的长期任务处理，显著降低了计算开销并保证了无误差的递归任务完成。

    

    我们提出了尤里卡（Eureka），一种任务条件化的元代理架构，它将长期任务编译为具有明确接受语义的动态义务图。在执行过程中，尤里卡通过滚动时域规划、架构提升和最小充分编译，形成具有专门状态、记忆、操作符、工具、验证器和局部拓扑的宏代理。当瓶颈反复出现时，成本效益门控的进化会在约束下更新局部架构。理论上，我们建立了关于遗憾、规划失效、摊销、子树接口、可串行性和验证的结果。实验上，尤里卡完成了170/170个递归任务，并生成了3,948个证书，且无误接受。主动上下文将中位输入从9,490个令牌压缩至4,005个；增量处理在12,000个任务中避免了65.38%的重计算；16,000个并发执行保持一致串行化。同一元代理在...

    arXiv:2608.19047v1 Announce Type: new  Abstract: We present Eureka, a task-conditioned Meta-Agent architecture that compiles long-horizon tasks into dynamic obligation graphs with explicit acceptance semantics. During execution, Eureka forms Macro-Agents with specialized state, memory, operators, tools, verifiers, and local topology via receding-horizon planning, architecture promotion, and minimal-sufficient compilation. When bottlenecks recur, cost-benefit-gated evolution updates the local architecture under constraints. Theoretically, we establish results on regret, planning invalidation, amortization, subtree interfaces, serializability, and verification. Experimentally, Eureka completes 170/170 recursive tasks and generates 3,948 certificates with no false acceptances. Active context compresses median input from 9,490 to 4,005 tokens; incremental processing avoids 65.38% recomputation across 12,000 tasks; 16,000 concurrent executions serialize consistently. The same Meta-Agent ins
    
[^22]: 伯恩斯坦-瓦兹拉尼网络：通过干涉实现的量子机器学习

    Bernstein-Vazirani Networks: Quantum Machine Learning by Interference

    [https://arxiv.org/abs/2608.19043](https://arxiv.org/abs/2608.19043)

    BVNs通过量子干涉和傅里叶采样实现无梯度训练的非变分量子机器学习，在分类和表示任务上达到与经典和量子方法相当的竞争力。

    

    arXiv:2608.19043v1 公告类型：交叉 摘要：我们引入了伯恩斯坦-瓦兹拉尼网络（BVNs），这是一种非变分量子机器学习框架，利用量子干涉进行监督学习，并在视觉和表示学习任务上进行了演示。在其标准形式中，BVNs遵循量子傅里叶采样的原理：将标记数据置于叠加态并在傅里叶基中干涉，以提取全局信息丰富的特征。随后，我们定义了广义BVNs，使得干涉可以在问题自适应基中进行，从而在相同测量预算下，相比标准设置产生更具表达力的模型。BVNs通过（超）完备干涉基实现通用函数逼近，而BVNs的训练是无梯度的。在合成和真实世界分类任务以及隐式图像表示上的实验表明，其具有强大的泛化能力，并与经典和量子基线相比表现出竞争力。

    arXiv:2608.19043v1 Announce Type: cross  Abstract: We introduce Bernstein-Vazirani Networks (BVNs), a non-variational quantum machine learning framework that leverages quantum interference for supervised learning, demonstrated on vision and representation learning tasks. In their standard form, BVNs follow the principle of quantum Fourier sampling: labelled data are placed in superposition and interfered in the Fourier basis to extract globally informative features. We then define generalised BVNs that enable interference in problem-adapted bases, yielding more expressive models under the same measurement budget as in the standard setting. BVNs achieve universal function approximation through (over)complete interference bases, while training of BVNs is gradient-free. Experiments on synthetic and real-world classification tasks, as well as implicit image representation, show strong generalisation capabilities and competitive performance with classical and quantum baselines.
    
[^23]: 反事实对比分析

    Counterfactual Contrastive Analysis

    [https://arxiv.org/abs/2608.19032](https://arxiv.org/abs/2608.19032)

    本文提出了一种基于对比分析的无分类器视觉反事实生成方法，通过分离和交换数据分布中的显著因素，生成模型无关且对分类器偏见不敏感的反事实解释。

    

    视觉反事实解释（VCEs）旨在通过生成输入图像的最小编辑且逼真的版本来解释图像分类器，从而改变分类器的预测。现有的VCE方法本质上依赖于分类器，因此容易受到分类器偏见和失败模式的影响，例如对捷径特征的敏感性和校准误差。在本文中，我们提出了一种基于对比分析（CA）的无分类器视觉反事实生成方法。给定两个对应不同类别（例如健康人和患者）的数据集，我们将两个数据集共有的生成因素与每个数据集特有的显著因素分离，并通过仅交换显著因素来生成反事实图像。通过直接作用于数据分布而非决策边界，我们的方法提供了模型无关的VCEs，对分类器偏见不太敏感。

    arXiv:2608.19032v1 Announce Type: cross  Abstract: Visual Counterfactual Explanations (VCEs) aim to explain image classifiers by generating minimally edited and realistic versions of an input image that change the classifier's prediction. Existing VCE methods are inherently classifier-dependent and therefore susceptible to classifier biases and failure modes, such as sensitivity to shortcut features and calibration errors. In this paper, we propose a classifier-free approach for visual counterfactual generation based on Contrastive Analysis (CA). Given two datasets corresponding to different classes (e.g., healthy and patients), we disentangle the generative factors that are common across the two datasets from those that are salient to each dataset, and generate counterfactual images by swapping only the salient factors. By operating directly on data distributions rather than decision boundaries, our method provides model-agnostic VCEs that are less sensitive to classifier biases. Our 
    
[^24]: 自适应记忆与反思多智能体系统用于医学问答

    Adaptive Memory and Reflection Multi-Agent System for Medical Question Answering

    [https://arxiv.org/abs/2608.19029](https://arxiv.org/abs/2608.19029)

    本文提出了一种自适应记忆与反思多智能体医学问答系统，通过专用记忆、反思反馈和动态工作流路由，在MedQA和MedMCQA上实现了优于基线的性能。

    

    准确且负责任的医学问答（QA）在医疗保健中至关重要，其中复杂病例需要事实知识和细致推理。现有的医学问答系统通常基于单智能体架构和静态检索，往往缺乏适应性、持久记忆和结构化决策。本研究引入了一种自适应记忆与反思（AMR）智能体系统，这是一个多智能体框架，其中专门智能体利用专用记忆和基于反思的反馈来检索相关既往病例并改进后续推理。复杂度评估将问题路由至单独、协作或升级工作流，而共识和伦理监督模块支持推理整合和输出审查。在MedQA和MedMCQA上的评估显示，与多种基线相比性能强劲。消融研究表明，结合智能体特定记忆、反思和外部检索能显著提升效果。

    arXiv:2608.19029v1 Announce Type: new  Abstract: Accurate and responsible medical question answering (QA) is important in healthcare, where complex cases require factual knowledge and nuanced reasoning. Existing medical QA systems, typically based on single-agent architectures and static retrieval, often lack adaptability, persistent memory, and structured decision-making. This work introduces an adaptive memory and reflection (AMR) agentic system, a multi-agent framework in which specialized agents use dedicated memory and reflection-based feedback to retrieve relevant prior cases and improve subsequent reasoning. Complexity assessment routes questions through solo, collaborative, or escalated workflows, while consensus and ethical overseer modules support reasoning consolidation and output review. Evaluation on MedQA and MedMCQA demonstrates strong performance compared with several baselines. Ablation studies show that combining agent-specific memory, reflection, and external retriev
    
[^25]: 自我提示与跨模型共识实现基于大型语言模型从科学文献中可复现的数据提取

    Self-prompting and cross-model consensus enable reproducible data extraction from scientific literature with large language models

    [https://arxiv.org/abs/2608.19025](https://arxiv.org/abs/2608.19025)

    本研究通过四种递进工作流程评估了大型语言模型在科学文献数据提取中的表现，发现自我提示接近专家提示效果，但自主文献发现仍不可靠，且生成数据集需人类参与，从而提出专家制定证据标准、模型执行提取的可审计分工模式。

    

    摘要：从研究文章中准确提取细微且情境化的数据既费时又费力。在此，我们研究了前沿的、基于浏览器的大型语言模型在提取高度情境化信息方面的性能。我们展示了四种递进的工作流程：1）在专家策划的提示和研究文章的基础上，大多数前沿大型语言模型在数据提取方面表现良好，但在解释科学背景和细微差别方面可能遇到困难；2）在简单指令下，大型语言模型能够自行编写提示，其效果几乎与专家编写的提示相当；3）自主发现研究文献较为困难，代理要么遗漏参考文献，要么产生幻觉；4）大型语言模型能够根据已发表的指南创建新数据集，这些数据集与人类专家评审的结果高度吻合，但仍需人类参与其中。综合来看，这些发现定义了一种可审计的分工模式，即专家规定证据标准。

    arXiv:2608.19025v1 Announce Type: new  Abstract: Accurately extracting nuanced, contextualized data from research articles is laborious and time intensive. Here, we investigate the performance of frontier, browser-based large language models (LLMs) to extract highly contextualized information. We demonstrate four escalating workflows, 1) given an expert curated prompt and research articles, most frontier LLMs perform well at data extraction, however can struggle with interpreting scientific context and nuance, 2) given simple instructions, LLMs can author their own prompts which were almost as eNective as expert-written prompts, 3) autonomous discovery of research literature was diNicult, agents either missed or hallucinated references, and 4) LLMs can create new datasets from published guidelines that closely match human-expert judges, but still require a human-in-the-loop. Together, these findings define an auditable division of labour in which experts specify the evidence standard, 
    
[^26]: 自动驾驶中的单阶段目标检测器

    One-Stage Object Detectors in Autonomous Driving

    [https://arxiv.org/abs/2608.19014](https://arxiv.org/abs/2608.19014)

    本综述系统分析了自动驾驶中单阶段目标检测器的演进与设计权衡，强调了其在速度、准确性、效率和鲁棒性之间的平衡策略。

    

    自主驾驶车辆依赖快速可靠的感知系统，以实时检测周围的车辆、行人、骑行者、交通标志和其他道路物体。本文并非提出新的检测系统，而是对自动驾驶中的单阶段目标检测器进行了全面综述和分析。该综述回顾了主要单阶段检测器的演变，包括YOLOv1、SSD、RetinaNet、EfficientDet、无锚点检测器（如FCOS和CenterNet）以及近期实时模型（如YOLOv10）。它通过设计选择、特征融合策略、损失函数、部署权衡和报告的基准性能对这些架构进行了比较。论文还总结了常用的自动驾驶数据集、评估指标、开放挑战和未来研究方向。总体而言，本综述强调了单阶段检测器如何在速度、准确性、效率和鲁棒性之间取得平衡。

    arXiv:2608.19014v1 Announce Type: cross  Abstract: Autonomous vehicles depend on fast and reliable perception systems to detect surrounding vehicles, pedestrians, cyclists, traffic signs, and other road objects in real time. This paper presents a comprehensive survey and analysis of one-stage object detectors for autonomous driving rather than an implementation of a new detection system. The survey reviews the evolution of major one-stage detectors, including YOLOv1, SSD, RetinaNet, EfficientDet, anchor-free detectors such as FCOS and CenterNet, and recent real-time models such as YOLOv10. It compares these architectures through their design choices, feature-fusion strategies, loss functions, deployment trade-offs, and reported benchmark performance. The paper also summarizes commonly used autonomous-driving datasets, evaluation metrics, open challenges, and future research directions. Overall, this survey highlights how one-stage detectors balance speed, accuracy, efficiency, and robu
    
[^27]: 驾驭持续学习：超越模型参数的持续适应

    Harness Continual Learning: Continual Adaptation Beyond Model Parameters

    [https://arxiv.org/abs/2608.19013](https://arxiv.org/abs/2608.19013)

    本文提出驾驭持续学习（HCL），一种在冻结基础模型外部演化提示、记忆等驾驭组件的新范式，并通过受保护的演化机制减少对早期行为的遗忘。

    

    持续学习在很大程度上是以模型为中心的，将模型参数视为随顺序经验而变化的状态。现代智能体还可以通过提示、记忆、工具、技能和路由规则的“驾驭机制”进行适应。由于这些内容共同塑造后续执行，即使模型被冻结，驾驭机制的更新也可能破坏先前可靠的行为。这提出了一个新问题：智能体如何在模型外部持续改进其状态，同时保留早期习得的行为？我们提出了驾驭持续学习（HCL），一种新的持续学习范式，其中驾驭机制围绕冻结的基础模型演化，并将由此导致的早期行为损失定义为驾驭级遗忘。我们通过四个面向执行的组件实例化HCL：任务接口、经验记忆、能力图谱和自适应路由器。我们进一步引入了受保护的驾驭演化，以将更新生成与执行分离，从而减少干扰。

    arXiv:2608.19013v1 Announce Type: cross  Abstract: Continual learning has largely been model-centric, treating model parameters as the state that changes with sequential experience. Modern agents can also adapt through a harness of prompts, memories, tools, skills, and routing rules. Because these contents jointly shape later execution, a harness update can disrupt previously reliable behavior even when the model is frozen. This raises a new question: how can an agent continually improve its state outside the model while retaining behavior acquired earlier? We formulate Harness Continual Learning (HCL), a new continual learning paradigm in which the harness evolves around a frozen foundation model, and define the resulting loss of earlier behavior as harness-level forgetting. We instantiate HCL with four execution-facing components: the Task Interface, Experience Memory, Capability Map, and Adaptive Router. We further introduce guarded harness evolution to separate update generation fr
    
[^28]: 从威胁情报到检测：知识驱动的丰富与基于模板的规则接地用于自动化Sigma规则生成

    From Threat Intelligence to Detection: Knowledge-driven Enrichment and Template-based Rule Grounding for Automated Sigma Rule Generation

    [https://arxiv.org/abs/2608.19011](https://arxiv.org/abs/2608.19011)

    本文提出AUTOSIGMA，一种自动化生成Sigma规则的方法，通过知识驱动的丰富和基于模板的规则接地，解决手动规则编写易错且扩展性差的问题，以动态适应新兴威胁和特定环境。

    

    arXiv:2608.19011v1 公告类型：交叉 摘要：由于高级持续性威胁（APTs）的快速演变，需要动态地将网络威胁情报（CTI）转化为可操作的检测能力的机制。Sigma规则是当代威胁检测工作流的重要组成部分，因为它们提供了一个平台无关的框架，用于表达检测逻辑，并可转换为跨SIEM系统的特定查询。手动编写Sigma规则的传统技术容易出错，并且需要广泛的知识，这限制了它们的可扩展性。尽管存在开源和行业维护的Sigma规则库，但它们往往无法跟上新兴威胁的步伐，并需要频繁定制以适应不同的操作环境。这强调了动态规则生成的必要性，这些规则需适应不断演变的攻击技术以及特定用例。在这项工作中，我们设计了AUTOSIGMA，一个

    arXiv:2608.19011v1 Announce Type: cross  Abstract: Mechanisms for dynamically converting cyber threat intelligence (CTI) into actionable detection capabilities are necessary due to the rapid evolution of Advanced Persistent Threats (APTs). Sigma rules are an essential part of contemporary threat detection workflows because they offer a platform-independent framework for expressing detection logic that can be converted into particular queries across SIEM systems. Conventional techniques for manually crafting Sigma rules are prone to mistakes, and necessitate extensive knowledge, which restricts their scalability. Although there are open-source and industry-maintained Sigma rule repositories, they often fail to keep pace with emerging threats and require frequent customization to fit diverse operational environments. This emphasizes the necessity of dynamic rule generation that is adapted to evolving attack techniques as well as particular use cases. In this work, we design AUTOSIGMA, an
    
[^29]: 事后辩论评判理论

    A Theory of Post-hoc Debate Judgement

    [https://arxiv.org/abs/2608.19002](https://arxiv.org/abs/2608.19002)

    本文提出了一种适用于智能体辩论场景的事后评判理论，定义了可重复性、稳健性、扎根性和可解释性等属性，并验证了其在LLM评判方法中的实现。

    

    辩论最近已成为一种有用的方法论，用于增强智能体AI的性能，并有助于可解释性和用户参与。例如，由LLM赋能的智能体可能在内部（与自己）和/或外部（与其他智能体）进行辩论。在许多使用辩论的场景中，辩论的结果和最终输出由外部评判者（通常是LLM）事后决定。在本文中，我们开发并测试了一种新颖的辩论评判理论，适用于智能体通过提供其观点的正反理由进行辩论的所有场景。具体而言，我们确定了辩论评判在一般情况下可能需要满足的若干形式属性，涉及可重复性、稳健性、扎根性和可解释性。然后，我们针对主张验证场景，对两种特定的替代辩论评判方法（即LLM作为变体）正式和/或实验性地探索了这些属性的满足情况。

    arXiv:2608.19002v1 Announce Type: new  Abstract: Debates have recently emerged as a useful methodology for agentic AI to improve performance as well as to aid explainability and user engagement. For example, LLM-empowered agents may debate internally (with themselves) and/or externally (with other agents). In many settings where debates are used, debates' outcomes and resulting outputs are determined post-hoc by external judges, often LLMs. In this paper we develop and test a novel theory of debate judgement applicable to all settings where agents engage in debates by providing pros and cons for their opinions therein. Specifically, we identify a number of formal properties that debate judgement may be required to satisfy in general, as concerns reproducibility, robustness, groundedness and explainability. Then, we explore their satisfaction formally and/or experimentally, for claim verification settings, for two specific alternative debate judgement methods: variants of the LLMs as a 
    
[^30]: GrabVG：面向无人机影像视觉定位的图注意力绑定方法

    GrabVG: Graph-Attentive Binding for Visual Grounding in UAV Imagery

    [https://arxiv.org/abs/2608.18996](https://arxiv.org/abs/2608.18996)

    本文提出GrabVG框架，通过模拟人类视觉搜索的两阶段机制（预注意假设搜索和图注意力特征绑定），有效解决了无人机影像中高拥挤场景下小目标视觉冗余和拓扑模糊导致的定位不准确问题。

    

    无人机影像中的视觉定位旨在根据自然语言描述，在复杂的鸟瞰场景中定位目标物体。然而，大量小尺寸、密集分布且视觉相似的物体会造成高度的视觉冗余，而重复的局部配置则引发强烈的拓扑模糊性。现有方法主要侧重于视觉-语言特征对齐或密集上下文交互，但仍难以区分细微的实例间差异，并有效利用空间拓扑结构，导致在高度拥挤的场景中定位不准确。为解决这些挑战，我们提出了GrabVG，一种受人类视觉搜索启发的全新视觉定位框架。GrabVG将定位过程明确分解为两个连续阶段：预注意假设搜索和图注意力特征绑定。具体来说...

    arXiv:2608.18996v1 Announce Type: cross  Abstract: Visual grounding in Unmanned Aerial Vehicle (UAV) imagery aims to localize a target object in complex bird's-eye-view scenes according to a natural language description. However, the abundance of small, densely distributed, and visually similar objects creates high visual redundancy, while repetitive local configurations give rise to strong topological ambiguity. Existing approaches mainly focus on visual--language feature alignment or dense contextual interaction, yet they struggle to distinguish subtle inter-instance differences and effectively exploit spatial topological structures, leading to inaccurate grounding in highly crowded scenarios. To address these challenges, we propose $\textbf{GrabVG}$, a novel visual grounding framework inspired by human visual search. GrabVG explicitly decomposes grounding into two sequential stages: $\textit{preattentive hypothesis search}$ and $\textit{graph-attentive feature binding}$. Specificall
    
[^31]: DeepWeaver：弥合开放式问题回答中的证据综合差距

    DeepWeaver: Bridging the Evidence Synthesis Gap in Open-Ended Question Answering

    [https://arxiv.org/abs/2608.18988](https://arxiv.org/abs/2608.18988)

    DeepWeaver通过引入思想块链（TBCs）这一结构化表示，弥合了开放式问答中检索与生成之间的证据综合差距，从而生成更全面且引用准确的答案。

    

    arXiv:2608.18988v1 公告类型：交叉 摘要：检索-然后-生成的流水线常用于为开放式问题生成深度研究答案，但仅靠检索是不够的：大型语言模型（LLM）必须将嘈杂且零散的证据组织成全面且引用充分的答案。我们将这一过程称为证据综合。然而，直接生成往往未能充分利用证据，导致引用错位，并将多样化的信息压缩成浅显的总结，从而暴露出检索与生成之间的证据综合差距。为此，我们提出了DeepWeaver，这是一种新颖的框架，通过维护思想块链（TBCs）——一种将主张、显著信息、关键词和支持证据分组的结构化表示——将嘈杂的检索证据编织成全面的答案。DeepWeaver使用从属TBCs来检查残余证据，提交TBC修订，并在最终生成前发现新主张。我们在基于知识库和文本的开放式问答上对DeepWeaver进行了评估。

    arXiv:2608.18988v1 Announce Type: cross  Abstract: Retrieve-then-generate pipelines are commonly used to produce deep-research answers for open-ended questions, but retrieval alone is insufficient: LLMs must organize noisy and fragmented evidence into comprehensive, well-cited answers. We refer to this process as evidence synthesis. However, direct generation often underuses evidence, misaligns citations, and collapses diverse information into shallow summaries, exposing an evidence synthesis gap between retrieval and generation. Thus, we propose DeepWeaver, a novel framework that weaves noisy retrieved evidence into comprehensive answers by maintaining Thought Block Chains (TBCs), a structured representation that groups claims, salient information, keywords, and supporting evidence. DeepWeaver uses subordinate TBCs to inspect residual evidence, commit TBC revisions, and discover new claims before final generation. We evaluate DeepWeaver on open-ended QA over both knowledge bases and t
    
[^32]: rEDMRec：将大语言模型推理蒸馏为可编辑的经验记忆以用于推荐

    rEDMRec: Distilling Large Language Model Reasoning into an Editable Experience Memory for Recommendation

    [https://arxiv.org/abs/2608.18952](https://arxiv.org/abs/2608.18952)

    本文提出rEDMRec，通过将大语言模型的推理蒸馏为四种可编辑的经验记忆通道，使轻量级模型能复用推理结果，从而避免每次推荐请求时重复生成推理，同时支持用户口味变化时的检查与修正。

    

    大语言模型可以通过对用户历史和候选项目进行显式推理来提升推荐质量——例如，提取用户的偏好或解释为何某个项目比另一个更合适——而不是直接将历史映射为排序列表。然而，这种推理在每次排序请求中重复执行成本高昂，且一旦生成，通常只被使用一次即被丢弃，既无法在未来的请求中复用，也难以在用户口味变化时进行检查或修正。我们的见解是，如果推理能够被一次性压缩为紧凑、结构化的记忆，并由轻量级模型从中检索，那么就不需要在每次调用时重新生成。我们提出了rEDMRec，它将教师大语言模型的推理蒸馏为四种类型化、可编辑的经验通道——长期偏好、短期上下文、项目感知和反事实硬负样本比较——并由一个大语言模型记忆控制器维护。

    arXiv:2608.18952v1 Announce Type: cross  Abstract: Large language models can improve recommendation quality by reasoning explicitly over user history and candidate items - for example, extracting a user's preferences or explaining why one item fits better than another - rather than mapping history directly to a ranked list. This reasoning, however, is expensive to repeat on every ranking request and, once produced, is typically consumed once and discarded, leaving it neither reusable across future requests nor easy to inspect or correct as user tastes drift. Our insight is that reasoning does not need to be regenerated at every call if it can instead be compressed once into a compact, structured memory that a lightweight model retrieves from. We propose rEDMRec, which distills a teacher LLM's reasoning into four typed, editable experience channels - long-term preference, short-term context, item-perception, and counterfactual hard-negative comparisons - maintained by an LLM memory cont
    
[^33]: AlphaClifford：基于模型强化学习的高效克利福德电路综合与转译

    AlphaClifford: Efficient Clifford Synthesis and Transpilation with Model-based RL

    [https://arxiv.org/abs/2608.18946](https://arxiv.org/abs/2608.18946)

    本文提出了AlphaClifford，一种基于蒙特卡洛树搜索和模型强化学习的框架，利用辛群代数性质高效综合克利福德电路，显著降低了总门数和CNOT门数。

    

    arXiv:2608.18946v1 公告类型：交叉 摘要：克利福德电路在量子计算中扮演基础性角色，特别是在量子纠错和容错逻辑综合中尤为重要。尽管这些电路可以被高效模拟并表示为辛矩阵，但标准综合方法（如Aaronson-Gottesman算法）往往产生门数过高的次优电路。在本工作中，我们引入了AlphaClifford，一个由蒙特卡洛树搜索驱动的基于模型的强化学习框架，旨在从由H、S和CNOT组成的基本门集高效综合克利福德电路。通过利用辛群的代数性质对状态空间进行建模，AlphaClifford有效探索这一组合空间以最小化整体电路成本。对于无约束的克利福德优化，我们的方法在总门数和两量子比特（CNOT）门数上均实现了一致的减少，与...相比。

    arXiv:2608.18946v1 Announce Type: cross  Abstract: Clifford circuits play a foundational role in quantum computing, particularly due to their importance in quantum error correction and fault-tolerant logical synthesis. While these circuits can be efficiently simulated and represented as symplectic matrices, standard synthesis methods-such as the Aaronson-Gottesman algorithm-often yield sub-optimal circuits with excessively high gate counts. In this work, we introduce AlphaClifford, a model-based Reinforcement Learning framework powered by Monte Carlo Tree Search, designed to efficiently synthesize Clifford circuits from the fundamental gate set composed of H, S, and CNOT. By modeling the state space through the algebraic properties of the symplectic group, AlphaClifford effectively explores this combinatorial space to minimize overall circuit cost. For unconstrained Clifford optimization, our approach achieves a consistent reduction in both total and two-qubit (CNOT) gate counts compar
    
[^34]: 训练化学合理性感知的大型语言模型用于单步逆合成

    Training Chemical Plausibility-Aware Large Language Models for Single-Step Retrosynthesis

    [https://arxiv.org/abs/2608.18940](https://arxiv.org/abs/2608.18940)

    本文提出Top-K提示训练范式，结合化学约束和奖励机制，构建了大规模数据集和C3LM模型，显著提升了单步逆合成中多样化且合理反应的预测性能。

    

    arXiv:2608.18940v1 公告类型：交叉 摘要：单步逆合成是计算机辅助合成规划的核心组成部分，但其内在的一对多特性在单一答案评估和基准测试协议中未能得到充分体现。为解决这一问题，我们引入了Top-K提示作为一种稳健的训练和推理范式，以更好地捕捉多样化且合理的反应预测。我们汇编了CREED-CCV-2+USPTO-XL，一个超大规模数据集，包含约4560万个已验证反应，用于训练C3LM（化学约束一致性语言模型）。通过结合基于ChemCensor和新颖性奖励的微调，我们的模型在OOD URSA-expert-2026基准上达到了最先进的性能。对反应独特性的进一步分析表明，大型语言模型和传统模型探索了互补的反应空间，这激发了基于集成的逆合成系统。总体而言，我们的结果确立了Top-K、合理性感知训练作为一种实用的新方向。

    arXiv:2608.18940v1 Announce Type: cross  Abstract: Single-step retrosynthesis is a central component of computer-aided synthesis planning, yet its intrinsically one-to-many nature is poorly captured by single-answer evaluation and benchmarking protocols. To address this, we introduce Top-K prompting as a robust training and inference paradigm to better capture diverse, plausible reaction predictions. We compile CREED-CCV-2+USPTO-XL, an ultra-large-scale dataset of ~45.6 million verified reactions to train the C3LM (Chemistry Constraint-Consistent Language Model). By integrating fine-tuning with ChemCensor-based and novelty-oriented rewards, our model achieves state-of-the-art performance on the OOD URSA-expert-2026 benchmark. Further analysis of reaction uniqueness shows that LLMs and conventional models explore complementary reaction spaces, motivating ensemble-based retrosynthesis systems. Overall, our results establish Top-K, plausibility-aware training as a practical new direction 
    
[^35]: 打破最薄弱环节以规避视觉语言模型

    Breaking the weakest link to evade vision language models

    [https://arxiv.org/abs/2608.18938](https://arxiv.org/abs/2608.18938)

    本文提出一种仅针对视觉编码器优化的梯度攻击方法，有效生成对抗样本，以规避视觉语言模型的多模态对齐，并涵盖非定向与定向攻击场景。

    

    arXiv:2608.18938v1 公告类型：新 摘要：视觉语言模型（VLMs）近期已成为多模态AI系统的关键组成部分，能够在现实世界和安全关键应用中实现对视觉和文本输入的联合推理。尽管其部署日益广泛，但VLMs在面对对抗性威胁时的鲁棒性仍未得到充分探索，尤其是在针对多模态对齐的规避攻击背景下。在本工作中，我们研究了VLMs对应用于视觉输入的对抗性扰动的脆弱性，并研究了两种攻击设置：非定向攻击（旨在破坏模型对原始图像的解释）和定向攻击（旨在迫使模型生成与原始图像无关的特定语义描述）。为了高效生成对抗性示例，我们提出了一种基于梯度的攻击方法，该方法仅在VLM的视觉编码器上进行优化，而不是……

    arXiv:2608.18938v1 Announce Type: new  Abstract: Vision Language Models (VLMs) have recently emerged as a critical component of multimodal AI systems, enabling joint reasoning over visual and textual inputs in real-world and safety-critical applications. Despite their growing deployment, the robustness of VLMs against adversarial threats remains insufficiently explored, particularly in the context of evasion attacks targeting multimodal alignment. In this work, we investigate the vulnerability of VLMs to adversarial perturbations applied to visual inputs and study two attack settings: untargeted attacks, where the goal is to disrupt the model's interpretation of the original image, and targeted attacks, where the adversary aims to force the model to generate a specific semantic description unrelated to the original image. To efficiently generate adversarial examples, we propose a gradient-based attack method that performs optimization exclusively on the vision encoder of the VLM rather
    
[^36]: MedUAG：面向医学多模态模型的统一理解与生成

    MedUAG: Unified Understanding and Generation for Medical Multimodal Models

    [https://arxiv.org/abs/2608.18937](https://arxiv.org/abs/2608.18937)

    本文提出了MedUAG，通过构建最大的医学统一理解与生成数据集（MedUAGCorpus）和系统化基准（MedUAGBench），开发了一个端到端训练的统一医学多模态模型，显著提升了医学领域的理解与生成能力。

    

    arXiv:2608.18937v1 公告类型：交叉 摘要：近年来，多模态大型语言模型（MLLMs）正迅速演变为统一理解与生成（UAG）框架。然而，将这些统一范式扩展到医学领域面临两大障碍：缺乏全面的训练和评估基准，以及缺少经过广泛验证的统一医学模型。为解决这些问题，我们提出了医学UAG的全面基础。首先，我们构建了MedUAGCorpus，这是迄今为止最大的统一医学理解与生成数据集，涵盖14种成像模态，包含超过600万个实例。其次，我们引入了MedUAGBench，这是一个系统化的基准测试，在标准化协议下将医学生成评估扩展到12个多样化任务。最后，利用这些资源，我们开发了MedUAG，一个端到端训练的统一医学模型。大量实验表明，MedUAG在广泛的理解与生成任务中表现出强大性能。

    arXiv:2608.18937v1 Announce Type: cross  Abstract: Recent Multimodal Large Language Models (MLLMs) are rapidly evolving into unified understanding and generation (UAG) frameworks. However, extending these unified paradigms to the medical domain is hindered by: the absence of comprehensive training and evaluation benchmarks, and the lack of broadly validated unified medical model. To address these gaps, we present a comprehensive foundation for medical UAG. First, we construct MedUAGCorpus, the largest unified medical understanding and generation dataset to date, comprising over 6 million instances across 14 imaging modalities. Second, we introduce MedUAGBench, a systematic benchmark that expands medical generation evaluation to 12 diverse tasks under standardized protocols. Finally, leveraging these resources, we develop MedUAG, an end-to-end trained unified medical model. Extensive experiments demonstrate that MedUAG achieves strong performance across a wide array of understanding and
    
[^37]: 可解释架构的图形化设计

    Graphical Design of Interpretable Architectures

    [https://arxiv.org/abs/2608.18936](https://arxiv.org/abs/2608.18936)

    本文提出了一种基于Penrose张量符号的图形化表示方法，用于设计可解释AI架构，该符号提供全局视图并与PyTorch einsum代码一一对应，并应用于描述多种可解释模型及前沿模型Steerling-8B。

    

    设计、实现和比较可解释架构需要一个形式化语言来表示它们。最常见的表示方式在两方面存在不足。符号方程无法一目了然地提供架构的全局视图。概率图模型和流程图不描述实际的张量操作，从而隐藏了关键见解并限制了可复现性。为弥合这一差距，我们引入了一种用于设计可解释AI架构的图形化符号，该符号改编自Penrose张量符号。这种图形化符号提供了架构的全局视图，并与PyTorch einsum代码一一对应。我们首先使用此符号来描述按构造可解释的架构，包括概念瓶颈、稀疏探针、原型网络、神经加性模型和线性模型混合。然后，我们图解了前沿可解释语言模型Steerling-8B的关键架构组件。

    arXiv:2608.18936v1 Announce Type: cross  Abstract: Designing, implementing, and comparing interpretable architectures requires a formal language to represent them. The most common representations fall short in one of two ways. Symbolic equations give no global view of an architecture at a glance. Probabilistic graphical models and flowcharts do not describe actual tensor manipulations, thus hiding key insights and limiting reproducibility. To close this gap, we introduce a graphical notation for designing interpretable AI architectures, adapted from Penrose tensor notation. This graphical notation gives a global view of an architecture and maps one to one onto PyTorch einsum code. We first use this notation to describe architectures that are interpretable by construction, including concept bottlenecks, sparse probes, prototype networks, neural additive models, and mixtures of linear models. We then diagram the key architectural components of Steerling-8B, a frontier interpretable langu
    
[^38]: SkillForge：用于项目特定问题解决的自蒸馏代理

    SkillForge: Self-Distilling Agents for Project-Specific Issue Resolution

    [https://arxiv.org/abs/2608.18933](https://arxiv.org/abs/2608.18933)

    SkillForge通过自蒸馏框架主动合成项目特定问题，将可复用的项目知识提炼为技能，从而无需依赖历史修复信号或高测试成本即可提升代理在特定代码库中的问题解决能力。

    

    基于大型语言模型（LLM）的代理在自动化软件问题解决方面表现出显著的能力，但它们往往因缺乏项目特定知识而难以解决特定代码库中的问题。现有的自进化方法从代码库历史或在线修复轨迹中获取此类知识，但它们要么依赖于可用的历史问题解决信号，要么在每次问题解决时产生高昂的测试时探索成本。在本文中，我们提出了SkillForge，一种自蒸馏框架，它主动从代码库本身获取项目特定知识。SkillForge不是等待真实问题暴露项目特定知识缺口，而是通过重新实现代码库中测试覆盖的核心功能来合成项目特定问题。通过解决这些合成问题，SkillForge将可重用的项目特定知识蒸馏为基于实体的技能和

    arXiv:2608.18933v1 Announce Type: cross  Abstract: Large language model (LLM) based agents have demonstrated remarkable proficiency in automated software issue resolution, yet they often struggle to resolve issues in a specific repository because they lack project-specific knowledge. Existing self-evolving approaches acquire such knowledge from repository history or online repair trajectories, but they either depend on available historical issue-resolution signals or incur substantial per-issue test-time exploration cost. In this paper, we propose SkillForge, a self-distillation framework that proactively acquires project-specific knowledge from the repository itself. Instead of waiting for real issues to expose project-specific knowledge gaps, SkillForge synthesizes project-specific issues by re-implementing test-covered core functionalities of the repository. By resolving these synthetic issues, SkillForge distills reusable project-specific knowledge into entity-grounded skills and a
    
[^39]: 测试时扩展在现实中的应用：为何利用而非探索成为瓶颈

    Test-Time Scaling in the Wild: Why Exploitation, Not Exploration, Is the Bottleneck

    [https://arxiv.org/abs/2608.18931](https://arxiv.org/abs/2608.18931)

    本文发现测试时扩展的瓶颈不在于探索（增加候选多样性），而在于利用（从候选池中选出最佳结果），后者在开放式任务中表现不佳。

    

    arXiv:2608.18931v1 公告类型：交叉 摘要：测试时扩展（TTS）通过增加额外的推理计算来改善语言模型的输出——生成多个候选、搜索部分序列或迭代精炼草稿。这些技术在数学和代码任务上取得了显著提升，但几乎仅在验证相对直接的任务上进行了开发和压力测试。我们首次在五个开放式生成基准（涵盖医学、法律、金融、通用聊天和创意写作）上，对五种TTS方法进行了计算标准化比较，并基于一个统一框架，将每种方法的令牌预算效果分解为探索和利用两部分。答案取决于你审视分解的哪一侧。扩展探索有效：在所有设置中，池中最佳候选随计算量增加而稳步提升。而问题在于利用——即将丰富候选池转化为最终输出的步骤。

    arXiv:2608.18931v1 Announce Type: cross  Abstract: Test-time scaling (TTS) improves language model outputs by spending additional inference compute - generating multiple candidates, searching over partial sequences, or iteratively refining drafts. These techniques yield large gains on mathematics and code, but have been developed and stress-tested almost exclusively on tasks where verification is straightforward. We conduct the first compute-normalised comparison of five TTS families across five open-ended generation benchmarks spanning medicine, law, finance, general chat, and creative writing - grounded in a unified framework that decomposes the effectiveness of each method's token budget into exploration and exploitation. The answer depends on which side of that decomposition you examine. Scaling exploration works: the best candidate in the pool improves steadily with compute across all settings. What breaks is exploitation - the step that converts a rich candidate pool into a final
    
[^40]: SMTrap：通过SMT冲突引导对大型推理模型发起经济高效的拒绝服务攻击

    SMTrap: Cost-Effective DoS Attacks Against Large Reasoning Models via SMT Conflict Guidance

    [https://arxiv.org/abs/2608.18921](https://arxiv.org/abs/2608.18921)

    本文提出一种无需模型反馈的DoS攻击新范式，利用SMT求解器的冲突计数作为低成本信号，引导生成推理密集型CSP实例，从而大幅延长大型推理模型的输出轨迹并实现高效攻击。

    

    摘要：arXiv:2608.18921v1 公告类型：交叉 摘要：现有的LRM-DoS方法严重依赖模型反馈来合成攻击查询，需要要么反复查询目标模型，要么训练专门的攻击模型。这些昂贵的操作严重削弱了攻击的效力。在本文中，我们提出了“搜索放大”（search amplification），一种新颖的、无需模型反馈的LRM-DoS范式。它利用从可满足性模理论（SMT）求解器获得的冲突计数作为低成本的外部信号，来引导生成推理密集的约束满足问题（CSP）实例。我们的关键观察是，LRM在解决CSP时依赖于试错和回溯搜索，其中给定CSP实例上的更高SMT冲突计数与更广泛的LRM回溯搜索和显著更长的输出轨迹正相关。基于这一发现，我们提出了\textsc{SMTrap}，一个轻量级、仅使用CPU的框架。在SMT冲突计数的引导下，\textsc{SMTrap}生成...

    arXiv:2608.18921v1 Announce Type: cross  Abstract: Existing LRM-DoS methods rely heavily on model feedback to synthesize attack queries, requiring either repeated queries to the target model or training a dedicated attack model. These expensive operations severely weaken attack leverage. In this paper, we propose \emph{search amplification}, a novel, model-feedback-free LRM-DoS paradigm. It employs the conflict count derived from an Satisfiability Modulo Theories (SMT) solver as a low-cost external signal to guide the synthesis of inference-heavy Constraint Satisfaction Problem (CSP) instances. Our key observation is that LRMs depend on trial-and-backtracking search when solving CSPs, where higher SMT conflict counts on a given CSP instance positively correlate with more extensive LRM backtracking search and substantially longer output trajectories. Building on this finding, we propose \textsc{SMTrap}, a lightweight, CPU-only framework. Guided by SMT conflict counts, \textsc{SMTrap} ge
    
[^41]: 面向小规模数据集的感知学习状态的动态生成式数据增强

    Learning-State-Aware Dynamic Generative Data Augmentation on Small-Scale Datasets

    [https://arxiv.org/abs/2608.18907](https://arxiv.org/abs/2608.18907)

    本文提出了一种感知学习状态的动态生成式数据增强方法，通过基于样本损失和损失下降率自适应调整增强强度，并采用解耦增强与扩散融合，有效解决了小规模数据集上生成式增强的适应性、区域适配及多样性-语义平衡问题。

    

    小规模图像分类通常受限于训练数据的稀缺性。基于预训练生成模型的生成式数据增强（GDA）已成为一种有效的解决方案。然而，现有方法依赖任务无关的增强策略，忽视了下游模型的需求。尽管最近的动态GDA方法结合模型反馈来指导增强，但仍难以可靠地确定样本特定的增强强度，并使增强策略适应不同的图像区域，同时平衡图像多样性和类别语义。为解决这些问题，我们提出了感知学习状态的动态生成式数据增强（LSADA）。具体而言，LSADA根据每个样本当前的损失和损失下降率构建学习状态，并将其映射到样本特定的增强强度。此外，LSADA引入了解耦的数据增强和扩散融合。

    arXiv:2608.18907v1 Announce Type: cross  Abstract: Small-scale image classification is often limited by the scarcity of training data. Generative data augmentation (GDA) based on pretrained generative models has emerged as an effective solution. However, existing methods rely on task-agnostic augmentation strategies that overlook downstream model needs. Although recent dynamic GDA methods incorporate model feedback to guide augmentation, they still struggle to reliably determine sample-specific augmentation strengths and adapt augmentation strategies to different image regions while balancing image diversity and class semantics.   To address these issues, we propose learning-state-aware dynamic generative data augmentation (LSADA). Specifically, LSADA constructs a learning state for each sample based on its current loss and loss-decrease rate, which is then mapped to a sample-specific augmentation strength. Furthermore, LSADA introduces a decoupled data augmentation and diffusion fusio
    
[^42]: TestifAI：基于层析成像的深度学习系统测试

    \textsc{TestifAI}: Tomography-Based Testing for Deep Learning Systems

    [https://arxiv.org/abs/2608.18900](https://arxiv.org/abs/2608.18900)

    本文提出了TestifAI框架，通过层析成像方法系统探索和总结深度学习系统在组合扰动下的鲁棒性，实现了高效准确的鲁棒性估计。

    

    摘要：随着人工智能系统越来越多地部署在安全关键的应用领域（例如自动驾驶），相关风险也在增加。因此，支撑现代人工智能系统的深度学习模型必须经过彻底测试，以确保其行为正确。一次鲁棒性测试涉及数千次推理，以经验性地验证模型输出在输入的有界扰动下是否保持稳定。然而，现有的测试框架缺乏系统探索和总结组合扰动空间内鲁棒性的手段。我们提出了TestifAI，一个深度学习测试框架，用于高效、准确地估计对扰动组合的鲁棒性。TestifAI允许用户将操作条件指定为语义输入扰动（例如图像模糊、亮度和缩放）和离散严重级别（例如低、中、高）的结构化空间。用户可以查询模式。

    arXiv:2608.18900v1 Announce Type: new  Abstract: As AI systems are increasingly deployed in safety-critical application domains (e.g., autonomous driving), associated risks increase too. Deep learning models underlying modern AI systems, therefore, must undergo thorough testing to ensure their correct behaviour. A single robustness test involves thousands of inferences to empirically verify if a model's outputs remain stable under a bounded perturbation of its inputs. However, existing testing frameworks lack the means to systematically explore and summarise robustness across a combinatorial space of perturbations.   We propose TestifAI, a deep learning testing framework for efficient and accurate estimation of robustness against combinations of perturbations. TestifAI enables users to specify operational conditions as structured spaces of semantic input perturbations (e.g., image blur, brightness and zoom) and discrete severity levels (e.g., low, medium and high). Users can query mode
    
[^43]: OWL类表达式的句法简化

    Syntactic Simplification of OWL Class Expressions

    [https://arxiv.org/abs/2608.18899](https://arxiv.org/abs/2608.18899)

    本文提出了一种名为CES的算法，用于在保持语义不变的前提下，通过重写规则简化OWL类表达式，从而降低复杂度并提高推理效率。

    

    arXiv:2608.18899v1 公告类型：新 摘要：类表达式学习通常会产生复杂的OWL类表达式，这些表达式难以解释和推理。然而，通过遵循理论基础的简化原则，这种复杂性可以被降低。在本文中，我们提出了类表达式简化器（CES），一种用于描述逻辑（DL）中类表达式句法简化的新颖算法。CES旨在保持形式语义的同时降低表示复杂性。它系统地应用重写规则来消除冗余并识别更简单但等价的表达式，从而在不改变逻辑蕴含的情况下生成更紧凑、更易读的表示。我们评估了CES在两个中等规模本体上学习的类表达式的有效性，证明了推理效率和冗长性减少方面的可测量改进。这项工作为更广泛的目标——使本体驱动的应用更加可解释和高效——做出了贡献。

    arXiv:2608.18899v1 Announce Type: new  Abstract: Class expression learning often produces complex OWL class expressions that are difficult to interpret and reason over. However, by following theoretically grounded simplification principles, this complexity can be reduced. In this paper, we propose Class Expression Simplifier (CES), a novel algorithm for the syntactic simplification of class expressions in Description Logics (DL). CES aims to preserve formal semantics while reducing representational complexity. It systematically applies rewriting rules to eliminate redundancies and identify simpler yet equivalent expressions, thereby producing more compact and human-readable representations without altering logical entailments. We evaluate the effectiveness of CES on class expressions learned from two medium-sized ontologies, demonstrating measurable improvements in reasoning efficiency and reductions in verbosity. This work contributes to the broader goal of making ontology-driven appl
    
[^44]: 免训练推理时自反思与成本受限早期停止的大型语言模型方法

    Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models

    [https://arxiv.org/abs/2608.18884](https://arxiv.org/abs/2608.18884)

    本文提出一种免训练的推理时自反思协议，通过成本受限的早期停止机制，在冻结LLM上实现高效自我验证，无需梯度更新即可提升推理性能。

    

    arXiv:2608.18884v1 公告类型：新 摘要：对推理型大型语言模型（如GRPO）进行强化学习训练成本高昂，且需要可控环境，导致每个贡献都需投入完整训练流程。我们提出EvoResearcher，一种免训练、推理时协议，为单个冻结的LLM骨干网络添加成本受限的自反思能力。该协议迭代执行“生成 -> 自我批评 -> 修订”过程，直到达到最大深度D或批评返回CONFIRMED哨兵（一种隐式早期停止），使骨干网络在严格计算预算下自我验证答案。四个自反思元奖励组件（正确性、效率、反思深度、工具调用多样性）作为设计原则，以提示级机制实例化，因此其收益在零梯度更新下累积。我们在Big-Bench Hard（100个问题）上验证该协议，并在同一冻结骨干网络上跨域验证GSM8K（500个）和MATH（500个）的表现，同时进行跨模型测试。

    arXiv:2608.18884v1 Announce Type: new  Abstract: Reinforcement-learning training of reasoning LLMs (e.g., GRPO) is expensive and requires a controllable environment, committing every contribution to a full training pipeline. We present EvoResearcher, a training-free, inference-time protocol that adds cost-bounded self-reflection to a single frozen LLM backbone. The protocol iterates generate -> self-critique -> revise until a maximum depth D is reached or the critique returns the CONFIRMED sentinel, an implicit early stop that lets the backbone self-verify its answer under a strict compute budget. Four self-reflective meta-reward components (correctness, efficiency, reflection depth, tool-call diversity) act as design principles instantiated as prompt-level mechanisms, so their benefits accrue with zero gradient updates. We validate the protocol on Big-Bench Hard (100 questions) and establish cross-domain behavior on GSM8K (500) and MATH (500) on the same frozen backbone, with cross-mo
    
[^45]: DentAgent：面向多模态牙科推理的以证据为中心的多智能体协同框架

    DentAgent: Evidence-Centric Multi-Agent Coordination for Multimodal Dental Reasoning

    [https://arxiv.org/abs/2608.18878](https://arxiv.org/abs/2608.18878)

    本文提出DentAgent，一个以证据为中心的多智能体框架，通过协调五个模态专用智能体并将观察转换为结构化证据记录，实现可追溯的多模态牙科推理。

    

    口腔疾病影响着全球数十亿人，凸显了对准确可靠牙科评估的迫切需求，这种评估需要整合来自领域知识、放射影像、口内照片和三维牙科数据等异构证据。现有的大多数牙科AI系统仍局限于特定模态或特定任务。尽管最近的视觉-语言模型支持灵活的牙科问答，但直接生成的回答使证据隐含且不可追溯。为解决这些局限，我们引入了DentAgent，一个以证据为中心的多智能体框架，其中协调器负责协调五个覆盖多种模态的专门智能体。每个专家智能体利用领域工具将观察结果转换为结构化证据记录。证据黑板将这些记录作为共享证据状态进行管理，在生成回答前跟踪覆盖范围、缺口和冲突。这种标准化的证据表示整合了孤立的牙科数据。

    arXiv:2608.18878v1 Announce Type: new  Abstract: Oral diseases affect billions of people worldwide, underscoring a pressing need for accurate and reliable dental assessment that integrates heterogeneous evidence from domain knowledge, radiographs, intraoral photographs, and 3D dental data. Most existing dental AI systems remain modality- or task-specific. Although recent vision-language models support flexible dental question answering, directly generated response leaves evidence implicit and untraceable. To address these limitations, we introduce DentAgent, an evidence-centric multi-agent framework, in which the Orchestrator coordinate five specialized agents spanning various modalities. Each specialist utilizes domain tools to convert observations into structured evidence records. The Evidence Blackboard manages these records as a shared evidence state, tracking coverage, gaps, and conflicts before response generation. This standardized evidence representation integrates isolated den
    
[^46]: SkillGate：长周期智能体中策略内技能选择训练

    SkillGate: Training In-Policy Skill Selection in Long-Horizon Agents

    [https://arxiv.org/abs/2608.18852](https://arxiv.org/abs/2608.18852)

    本文提出并识别了长周期智能体中技能选择训练的结构性难题“选择器信用匮乏”，指出传统结果奖励强化学习无法有效训练这一关键决策。

    

    arXiv:2608.18852v1 公告类型：新 摘要：智能体框架日益将程序性知识打包为技能：智能体按需读取的指令文件，而公共库现在拥有数千个这样的技能。因此，读取哪个技能成为策略本身在情节中途做出的决策，但目前没有现有信号来训练这一决策。我们表明，默认的补救方法——对候选列表进行结果奖励的强化学习——无法教会这一决策，原因是我们识别并命名为“选择器信用匮乏”的结构性问题：在广播式、序列级优势下，命名所选技能的少数令牌在损失中占有的份额微乎其微，而且随着轨迹延长，它们继承的信用错误符号化越来越严重。一个正确的选择在执行后失败时总是受到惩罚，尽管该选择本身是轨迹中最有价值的决策之一。审计已完成运行自身的训练产物确认了所有这三个性质，每个性质随视野长度单调恶化。

    arXiv:2608.18852v1 Announce Type: new  Abstract: Agent frameworks increasingly package procedural knowledge as skills: instruction files an agent reads on demand, while public libraries now hold thousands of them. Which skill to read has thus become a decision the policy itself makes in the middle of an episode, yet no existing signal trains it. We show that the default remedy, outcome-rewarded RL over the candidate slate, cannot teach it, for a structural reason we identify and name selector credit starvation: under a broadcast, sequence-level advantage, the few tokens that name the chosen skill carry a vanishing share of the loss, and the credit they inherit is increasingly wrong-signed as trajectories lengthen. A correct choice is punished whenever the execution after it fails, even though the choice itself is among the most valuable decisions in the trajectory. Auditing a completed run's own training artifacts confirms all three properties, each worsening monotonically with horizon
    
[^47]: ORBITER：面向末端配送智能体的冲突感知决策机制

    ORBITER: Conflict-Aware Decision-Making for Agentic Last-Mile Delivery

    [https://arxiv.org/abs/2608.18846](https://arxiv.org/abs/2608.18846)

    ORBITER通过引入决策点和冲突感知机制，利用大语言模型显式推理配送中的时空与行为线索，从而提升末端配送中下一步订单决策的可靠性和可解释性。

    

    末端配送旨在处理动态到达的订单，同时需对复杂的空间与时间相关性进行建模。近年来的学习方法通过建模订单间的时空依赖关系来预测配送员的服务序列，但未对下一步订单的决策过程做出解释。用语言描述当前配送状态，可让大语言模型（LLM）明确推理单个决策背后的空间、时间及行为线索。然而，作为直接预测器，LLM对任务表述较为敏感，常产生不可靠的决策。为解决这些挑战，我们提出了ORBITER，一种用于末端配送中下一步订单决策的智能体仲裁器。ORBITER通过决策点对配送员服务进行建模，每个决策点包含配送员的时空状态及可见订单，并暴露局部权衡以供建模和验证。固定提议者负责对候选进行排序，并通过结构化仓库……（摘要截断）

    arXiv:2608.18846v1 Announce Type: new  Abstract: Last-mile delivery aims to handle dynamically arriving orders with couriers while modeling complex spatial and temporal correlations. Recent learning-based methods model spatiotemporal dependencies among orders to predict courier service sequences, but leave next-order decision making unexplained. Describing the current delivery state in language allows LLMs to reason explicitly about the spatial, temporal, and behavioral cues behind an individual decision. As direct predictors, however, LLMs remain sensitive to task presentation and often produce unreliable decisions. To address these challenges, we introduce ORBITER, an agentic Order Arbiter for next-order decision-making in last-mile delivery. ORBITER models courier service through decision points, each containing the courier's spatiotemporal state and visible orders and exposing local trade-offs for modeling and verification. Fixed proposers rank the candidates, and a structured repo
    
[^48]: 可验证弃权使供水管网中的AI泄漏诊断更具问责性

    Verifiable abstention makes AI leak diagnosis accountable in water distribution networks

    [https://arxiv.org/abs/2608.18836](https://arxiv.org/abs/2608.18836)

    本文提出一种基于可验证弃权的AI泄漏定位框架，通过物理执行代理和LLM审计监督代理的协作，在不行动时明确弃权，从而在保持高决策精度的同时显著提升系统问责性。

    

    arXiv:2608.18836v1 公告类型：新 摘要：公用事业公司因泄漏损失大量处理过的水，但很少信任人工智能定位器派遣维修队：到处猜测无法为挖掘提供理由。差距在于问责性，而非准确性：没有方法能证明其何时不应行动。在此，我们将泄漏定位重新定义为在可验证弃权下的决策问题。一个基于物理的执行代理针对数字孪生体对假设（泄漏、需求、传感器、阀门）进行证伪；一个独立的监督代理，配备大型语言模型（LLM）审计器，根据代码可验证的合同检查证据，然后认证派遣、请求证据或弃权。在现场级噪声下，32%的强制基线在已行动事件上提升至96%的决策精度。在独立生成的基准上，它仅对33个泄漏中的4个采取行动，且全部正确。一个包含194个已审计真实泄漏位置及孪生模拟压力和流量的注册表，产生五次挖掘派遣，其中三次

    arXiv:2608.18836v1 Announce Type: new  Abstract: Utilities lose a substantial share of treated water to leakage, yet rarely trust artificial-intelligence localizers to dispatch crews: guessing everywhere cannot justify excavation. The gap is accountability, not accuracy: no method proves when it should not act. Here we recast leak localization as decision-making under verifiable abstention. A physics-grounded executor agent falsifies hypotheses (leak, demand, sensor, valve) against a digital twin; an independent supervisor agent, with a large-language-model (LLM) auditor, checks evidence against a code-verifiable contract, then certifies a dispatch, requests evidence or abstains. Under field-grade noise, a 32% forced baseline becomes 96% decision precision on acted events. On an independently generated benchmark it acts on only 4 of 33 leaks, all correct. A 194-event register of audited real leak locations with twin-simulated pressures and flows yields five excavation dispatches, three
    
[^49]: MLREF：通过大型语言模型实现强化学习中奖励设计的模块级高效复用

    MLREF: Efficient Module Reuse for Reward Design in Reinforcement Learning via Large Language Models

    [https://arxiv.org/abs/2608.18827](https://arxiv.org/abs/2608.18827)

    本文提出MLREF框架，通过持久化模块池实现奖励组件的积累、改进和复用，将奖励函数构建为模块的线性组合，以解决现有方法中奖励函数作为整体程序导致的性能不稳定问题。

    

    摘要：奖励函数设计仍然是强化学习中的一个瓶颈。虽然大型语言模型（LLMs）已实现自动化奖励生成，但现有方法将奖励函数作为整体程序进行生成和修改，难以可靠地保留和复用早期迭代中发现的有效组件，导致跨迭代性能不稳定。为解决这一问题，我们提出了模块级奖励进化框架（MLREF）。MLREF的核心是一个模块池，即一个可复用奖励组件的持久化存储库。MLREF将模块池作为主要优化对象：该池通过累积成功模块、改进表现不佳的模块以及复用已验证组件，在迭代中进化；而奖励函数则作为从该池中抽取的模块的线性组合来构建。为驱动这一进化过程，MLREF整合了三种机制：基于反射的改进、混合信用分配以及……

    arXiv:2608.18827v1 Announce Type: cross  Abstract: Reward function design remains a bottleneck in reinforcement learning. While large language models (LLMs) have enabled automated reward generation, existing methods generate and revise reward functions as monolithic programs, making it difficult to reliably preserve and reuse effective components discovered in earlier iterations, leading to unstable performance across iterations. To address this, we propose Module Level Reward Evolution Framework (MLREF). At the core of MLREF is a module pool, a persistent repository of reusable reward components. MLREF treats the module pool as the primary optimization object: the pool evolves across iterations by accumulating successful modules, refining underperforming ones, and reusing proven components; while reward functions are constructed as linear combinations of modules drawn from this pool. To drive this evolution, MLREF integrates three mechanisms: reflection-based refinement, hybrid credit
    
[^50]: 通过分层分析理解多语言医学语音识别的适应性调整

    Understanding Multilingual Medical ASR Adaptation Through Layer-Wise Analysis

    [https://arxiv.org/abs/2608.18825](https://arxiv.org/abs/2608.18825)

    本文通过分层编码器分析揭示了多语言医学微调如何重塑Whisper内部表征，并发现最佳模型选择依赖于适应性调整设置，其中Whisper-Medium在直接多语言训练中表现最优。

    

    医学自动语音识别（MedASR）需要适应专业术语、有限的标注临床数据以及多语言使用场景。尽管像Whisper这样的大规模预训练ASR模型实现了强大的泛化能力，但它们在医学和多语言适应性调整后的行为，除了词错误率（WER）之外，仍未被充分理解。本文通过分层编码器分析，研究了多语言医学适应性调整如何重塑Whisper模型的内部表征。我们比较了零样本解码、仅英语微调、仅德语诊断性微调、两阶段EN->EN+DE连续微调以及直接EN+DE微调在多种Whisper模型大小下的表现。微调显著提升了MedASR性能，但最佳模型取决于适应性调整设置：Whisper-Medium在直接EN+DE训练下实现了最低的英语WER（7.72%）和最低的组合EN+DE WER（26.30%）；仅德语微调...

    arXiv:2608.18825v1 Announce Type: cross  Abstract: Medical automatic speech recognition (MedASR) requires adaptation to specialised terminology, limited annotated clinical data, and multilingual use cases. Although large-scale pretrained ASR models such as Whisper achieve strong generalisation, their behaviour after medical and multilingual adaptation remains insufficiently understood beyond word error rate (WER). This paper investigates how multilingual medical adaptation reshapes the internal representations of Whisper models through layer-wise encoder analysis. We compare zero-shot decoding, English-only fine-tuning, German-only diagnostic fine-tuning, two-stage EN->EN+DE continuation, and direct EN+DE fine-tuning across Whisper model sizes. Fine-tuning substantially improves MedASR performance, but the best model depends on the adaptation setting: Whisper-Medium gives the lowest English WER (7.72%) and the lowest combined EN+DE WER under direct EN+DE training (26.30%); German-only 
    
[^51]: 识别论证图逻辑重构中的隐含前提

    Identifying Implicit Premises for Logical Reconstruction of Argument Graphs

    [https://arxiv.org/abs/2608.18821](https://arxiv.org/abs/2608.18821)

    本文提出了一种神经符号流水线，利用大型语言模型生成隐含前提并转化为逻辑公式，以逻辑重构论证图中的蕴含或矛盾关系。

    

    摘要：arXiv:2608.18821v1 公告类型：交叉 摘要：从自然语言文本中对论证图进行逻辑重构具有挑战性，因为省略三段论（即带有隐含前提的论证）普遍存在。已有自然语言处理方法用于识别文本中的省略三段论，也有基于溯因推理的符号方法用于识别省略三段论逻辑表示中的缺失前提。然而，仍需要方法来生成隐含前提，以逻辑地展示一对陈述之间已知的蕴含或矛盾关系。为解决这一问题，我们提出了一种神经符号流水线，利用大型语言模型（LLMs）生成中间隐含前提，这些前提被转换为逻辑公式，并与表示显式前提和显式主张的逻辑公式一起使用，以展示它们之间的逻辑关系（蕴含、矛盾或中立）。我们的方法在Microtext Argumentative数据集上进行了评估。

    arXiv:2608.18821v1 Announce Type: cross  Abstract: The logical reconstruction of argument graphs from natural language text is challenging because of the prevalence of enthymemes (i.e., arguments with implicit premises). There are natural language processing methods for identifying enthymemes in text, and there are symbolic methods based on abduction for identifying missing premises in a logical representation of enthymemes. However, there is a need for methods to generate implicit premises to logically show a known entailment or contradiction relationship between a pair of statements. To address this, we propose a neuro-symbolic pipeline that uses large language models (LLMs) to generate intermediate implicit premises that are translated into logical formulae and used with logical formulae representing explicit premises and explicit claims to show the logical relationships between them (entailment, contradiction, or neutrality). Our approach is evaluated on the Microtext Argumentative
    
[^52]: 语义链接不确定性下的省略三段论补全的成对逻辑选择

    Pairwise Logical Selection of Enthymeme Completions under Semantic-Link Uncertainty

    [https://arxiv.org/abs/2608.18820](https://arxiv.org/abs/2608.18820)

    本文提出PWAL方法，通过可能世界语义链接形式化在不确定条件下边缘化逻辑阻力，实现省略三段论缺失成分的成对逻辑选择，优于现有方法。

    

    论证常常省略前提或主张，形成省略三段论。我们研究在省略成分的两个候选项之间进行成对逻辑选择的问题。现有的自然语言方法可以识别或生成候选项，但通常不揭示所选候选项如何完成推理，而基于逻辑的方法通常假设所需的公式和背景知识是可用的。我们将先前的神经符号管道从缺失前提扩展到缺失主张的选择，并用逻辑阻力分数替代二元蕴含结果。Top-Link在单一最高置信度语义链接配置下使用加权部分MaxSAT。然后，我们引入可能世界原子链接形式化（PWAL），该方法保持翻译后的公式固定，并在替代的跨公式语义链接配置上边缘化逻辑阻力。我们在五个任务上评估PWAL：ARCT和从CDED派生的任务用于缺失前提。

    arXiv:2608.18820v1 Announce Type: new  Abstract: Arguments often omit premises or claims, forming enthymemes. We study pairwise logical selection between two candidates for the omitted component. Existing natural language methods can identify or generate candidates but often do not expose how the selected candidate completes the inference, while logic-based approaches usually assume that the required formulae and background knowledge are available. We extend a prior neuro-symbolic pipeline from missing-premise to missing-claim selection and replace binary entailment outcomes with logical-resistance scores. Top-Link uses weighted Partial MaxSAT under a single configuration of highest-confidence semantic links. We then introduce Possible-World Atom-Link Formalization (PWAL), which keeps translated formulae fixed and marginalizes logical resistance over alternative cross-formula semantic-link configurations. We evaluate PWAL on five tasks: ARCT and a CDED-derived task for missing-premise 
    
[^53]: 大型语言模型会幻觉出电幻影吗？

    Do Large Language Models Hallucinate Electric Fata Morganas?

    [https://arxiv.org/abs/2608.18816](https://arxiv.org/abs/2608.18816)

    本文提出大型语言模型的幻觉不仅是工程缺陷，还具哲学意义，并通过实验证明温度参数影响幻觉与创造力表现。

    

    arXiv:2608.18816v1 公告类型：交叉 摘要：人工智能幻觉——即编造、无法验证或与源材料相矛盾的输出——通常被视为需要处理的工程缺陷。本文认为，在涉及机器意识问题时，它们也具有哲学意义。我们审视了大型语言模型中幻觉的已知原因，如源-目标分歧、训练与推理之间的差异以及过拟合，并进行了两项实证研究。在第一项研究中，我们在不同温度设置下将GPT模型的连续世代应用于模糊事实问题，发现较高温度导致看似合理但错误的答案，而较低温度则产生事实准确的答案。导致模型看起来有创意或自发、从而更可能通过智能行为测试的采样参数，正是那些增加幻觉的参数。

    arXiv:2608.18816v1 Announce Type: cross  Abstract: AI hallucinations - that is, outputs which are made up, cannot be verified, or contradict the source material - are generally regarded as an engineering flaw to be dealt with. This paper contends that they also have philosophical significance when it comes to the question of machine consciousness. We examine the known causes of hallucinations in large language models - such as source-target divergence, discrepancies between training and inference, and overfitting - and we present two empirical investigations. In the first, we apply successive generations of the GPT model to ambiguous factual questions under different temperature settings, finding that higher temperatures result in plausible but incorrect answers while lower temperatures lead to factually accurate ones. The sampling parameters that cause a model to seem creative or spontaneous and thus more likely to pass behavioral tests of intelligence are the same ones that increase 
    
[^54]: 对 $O_2$ 的 MCFL 性质的加强

    A strengthening of the MCFL-ness of $O_2$

    [https://arxiv.org/abs/2608.18813](https://arxiv.org/abs/2608.18813)

    本文通过加强字符串元组分解的特征刻画，为 $O_2$ 是多重上下文无关文法提供了更强的新证明。

    

    arXiv:2608.18813v1 公告类型：交叉 摘要：近年来，已有多个关于 $O_2$ 是多重上下文无关文法（MCFG）的证明被给出。这些结果可在计算语言学和计算代数领域中得到应用。在此，我们聚焦于近期一个以字符串元组分解形式表述的证明，并给出一个新结果，该结果对这些分解的特征刻画比现有定理更强。

    arXiv:2608.18813v1 Announce Type: cross  Abstract: In the last years, a number of proofs of the fact that $O_2$ is a multiple context-free grammar (MCFG) were given. Such results can be exploited in the fields of both computational linguistics and of computational algebra. Here, we focus on a recent such proof spelled in terms of factorizations of string tuples, and give a new result with a stronger characterization of such factorizations than in existing theorems.
    
[^55]: 遗忘、可塑性与共同观察：持续学习的第三个维度

    Forgetting, plasticity, and co-observation: a third facet of continual learning

    [https://arxiv.org/abs/2608.18803](https://arxiv.org/abs/2608.18803)

    本文提出数据共同观察是持续学习中除遗忘和可塑性外的第三个关键因素，并通过实验证明联合观察数据能带来表征优势，从而影响性能。

    

    arXiv:2608.18803v1 公告类型：交叉 摘要：高效的持续学习仍是深度神经网络面临的基本挑战。虽然灾难性遗忘和可塑性丧失被广泛视为需要克服的主要障碍，但我们表明这两个问题无法完全解释朴素顺序训练与离线联合训练之间的性能差距。在本文中，我们强调数据共同观察是影响持续学习性能的一个独立因素。通过将数据访问分离的约束与稳定性和可塑性解耦，我们系统性地研究了共同观察训练数据所带来的表征优势。实证上，我们在通用数据增量“分块”场景中，无论是在监督学习还是自监督学习范式下，都展示出联合训练与分离训练之间一致性的性能差异，同时缓解了遗忘并控制了可塑性。我们的发现表明，同时观察数据对持续学习性能具有重要影响。

    arXiv:2608.18803v1 Announce Type: cross  Abstract: Efficient continual learning remains a fundamental challenge for deep neural networks. While catastrophic forgetting and loss of plasticity are widely considered the primary obstacles to overcome, we show that these two issues cannot fully explain the performance gap between naive sequential training and offline joint training. In this paper, we highlight data co-observation as a distinct factor influencing continual learning performance. By decoupling the constraints of separate data access from stability and plasticity, we systematically investigate the representational benefits gained by observing training data together. Empirically, we demonstrate a consistent performance difference between joint and separate training across both supervised and self-supervised paradigms in generic data-incremental "chunking" scenarios, whilst mitigating forgetting and controlling for plasticity. Our findings indicate that simultaneous observation o
    
[^56]: 解构LLM自洽性中的错误共识：GPT-4.1案例研究

    Decomposing Wrong-Consensus Agreement in LLM Self-Consistency: A GPT-4.1 Case Study

    [https://arxiv.org/abs/2608.18795](https://arxiv.org/abs/2608.18795)

    本文通过定义并分解一致性指数Gamma，定量揭示了LLM多数投票在难题上失效的原因，并区分了机械成分与偏好残差。

    

    摘要：arXiv:2608.18795v1 公告类型：交叉 摘要：对多个LLM样本进行多数投票被广泛用于提高答案准确性，但其收益波动剧烈：在难题上甚至可能适得其反。本文对此失败进行了定量分析。定义了一个多元一致性指数Gamma，即错误运行样本中与共识一致的期望比例，通过参考尺度d=(1-p)/(C-1)归一化，并分解为机械成分（仅基于每例答案偏好时投票所产生的结果）和偏好无法解释的残差。机械零模型是难度匹配且无泄漏的：每个案例根据其自身准确性和选项偏好进行重模拟，这些参数从该案例的其他运行中估计，因此没有运行能预测其自身的一致性。在GPT-4.1上，分解显示了与基准相关的方向性（对每个基准的n=4个单元进行观测排序，而非显著性声明）。在多选题GPQA-Diamond上，每案例答案偏好...

    arXiv:2608.18795v1 Announce Type: cross  Abstract: Majority voting over multiple LLM samples is widely used to raise answer accuracy, yet its gain varies erratically: on hard questions it can even backfire. This paper gives a quantitative account of this failure. A pluralistic agreement index Gamma is defined as the expected fraction of the samples of a wrong run that agree with the consensus, normalized by a reference scale d=(1-p)/(C-1), and is decomposed into a mechanical component (what a vote delivers given only a per-case answer preference) and a preference-unexplained residual. The mechanical null is difficulty-matched and leak-free: each case is resimulated at its own accuracy and option preference, estimated from the case's other runs, so no run predicts its own agreement. On GPT-4.1 the decomposition shows benchmark-associated direction (an observational ordering over n=4 cells per benchmark, not a significance claim). On multiple-choice GPQA-Diamond, the per-case answer pref
    
[^57]: SIDScope：生成式推荐中语义ID接口的诊断资源

    SIDScope: A Diagnostic Resource for Semantic-ID Interfaces in Generative Recommendation

    [https://arxiv.org/abs/2608.18779](https://arxiv.org/abs/2608.18779)

    SIDScope是一个源追踪诊断资源，用于评估生成式推荐中语义ID接口的健康状况，核心发现是接口健康需多信号评估，且前缀对齐的有效性取决于检索机制是否依赖前缀。

    

    arXiv:2608.18779v1 公告类型：交叉 摘要：语义ID映射是物品分词器与生成式推荐器之间的可复用接口，然而发布的映射很少说明它们是否连贯、暴露何种结构、生成路径如何解析，或刷新后必须重新验证什么。SIDScope是一个用于这些决策的源追踪诊断资源。它规范化物品到代码的产物，验证来源和连接，分析映射结构，比较配对修订，并解释生成轨迹中的路径到物品结果。在来自亚马逊和Yelp数据的七个家族的九个源追踪分词器导出中——包括八条可执行路径和一个可审计快照——SIDScope揭示接口健康是多信号而非标量。其核心发现是机制条件性的：当检索消耗SID前缀时，前缀对齐强烈跟踪保留候选曝光，然后随着评分变得前缀无关而减弱。训练轨迹约占...

    arXiv:2608.18779v1 Announce Type: cross  Abstract: Semantic-ID mappings are reusable interfaces between item tokenizers and generative recommenders, yet released mappings rarely state whether they are coherent, what structure they expose, how generated paths resolve, or what must be revalidated after a refresh. SIDScope is a source-traced diagnostic resource for these decisions. It normalizes item-to-code artifacts, verifies provenance and joins, profiles mapping structure, compares paired revisions, and accounts for path-to-item outcomes in generated traces. Across nine source-traced tokenizer exports from seven families on Amazon and Yelp data - eight executable routes plus one auditable snapshot - SIDScope reveals that interface health is multi-signal rather than scalar. Its central finding is mechanism-conditional: prefix alignment strongly tracks held-out candidate exposure when retrieval consumes SID prefixes, then weakens as scoring becomes prefix-independent. Trained trace acco
    
[^58]: 超越预测公平性：量化糖尿病视网膜病变筛查中跨人口统计群体的归因一致性

    Beyond Predictive Fairness: Quantifying Attribution Consistency Across Demographic Groups in Diabetic Retinopathy Screening

    [https://arxiv.org/abs/2608.18759](https://arxiv.org/abs/2608.18759)

    本文提出解释一致性评分（ECS），发现预测公平性与解释一致性是模型行为的互补维度，并强调公平性评估应超越传统预测性能。

    

    医学影像中的公平性通常通过亚组性能指标来评估，但模型是否在不同人口统计群体间依赖一致的视觉证据仍不明确。本研究引入了解释一致性评分（ECS），这是一种基于Jensen-Shannon散度的公平性感知指标，用于量化亚组间归因图的相似性。以糖尿病视网膜病变筛查为案例，ECS在全局和疾病严重程度层面进行评估。实验结果显示，尽管不同种族群体间的预测性能存在差异，但解释一致性相对较高，且与性能差异无显著关联。这些发现表明，预测公平性和解释一致性捕捉了模型行为的互补维度，促使公平性评估超越预测性能本身。

    arXiv:2608.18759v1 Announce Type: cross  Abstract: Fairness in medical imaging is commonly evaluated through subgroup performance metrics, yet it remains unclear whether models rely on consistent visual evidence across demographic groups. This work introduces the Explanation Consistency Score (ECS), a fairness-aware metric based on Jensen-Shannon divergence that quantifies the similarity of attribution maps across subgroups. Using diabetic retinopathy screening as a case study, ECS is evaluated globally and within disease severity. Experiments reveal that while predictive performance differs across ethnic groups, explanation consistency remains relatively high and shows no significant association with performance disparities. These findings suggest that predictive fairness and explanation consistency capture complementary dimensions of model behavior, motivating fairness evaluations that extend beyond predictive performance.
    
[^59]: 认知从属：生成式人工智能与知识的基础设施

    Epistemic Subordination: Generative AI and the Infrastructure of Knowledge

    [https://arxiv.org/abs/2608.18758](https://arxiv.org/abs/2608.18758)

    生成式AI通过将多数文化认知固化为默认知识基础设施，造成对少数群体认知的结构性从属，且现有法律因只监管下游而无法应对这一统一性伤害。

    

    生成式人工智能不仅仅产生有偏见的输出。它将多数人的认知方式编码为知识本身的默认基础设施。我们称之为认知从属。训练过程将人类表达的广泛多样性压缩为一个单一的概率模型，其统计基线反映了主导文化的语言、假设和文化框架。少数群体的认识论并未被排除，而是被吸收：存在于训练数据中，但在输出中结构性地处于从属地位。其结果不是一组可以审计和纠正的离散偏见，而是一种嵌入在所有输出产生之架构中的认知状态。这一统一性伤害跨越三个法律领域——反歧视法、文化与语言权利以及民主观点多元主义——而每个领域都因相同的结构性原因未能解决它：现有法律监管下游环节，而未能触及上游的认知基础设施。

    arXiv:2608.18758v1 Announce Type: cross  Abstract: Generative AI does not merely produce biased outputs. It encodes the majority's way of knowing as the default infrastructure of knowledge itself. We call this epistemic subordination. The training process compresses the full breadth of human expression into a single probabilistic model whose statistical baseline reflects the languages, assumptions, and cultural frameworks of the dominant culture. Minority epistemologies are not excluded but absorbed: present in the training data, yet structurally subordinated in the output. The result is not a collection of discrete biases that can be audited and corrected. It is an epistemic condition embedded in the architecture from which all outputs emerge. This unified harm cuts across three legal domains -- anti-discrimination law, cultural and linguistic rights, and democratic viewpoint pluralism -- and each fails to address it for the same structural reason: existing law regulates downstream, a
    
[^60]: 自我编写的指标：从自身盲点演化出评估器

    Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots

    [https://arxiv.org/abs/2608.18744](https://arxiv.org/abs/2608.18744)

    本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。

    

    arXiv:2608.18744v1 公告类型：新 摘要：智能体在可靠自动指标的引导下能快速进步，而没有指标则会停滞不前；最需要这种指标的应用（如报告生成）恰恰是无人知道如何评分的领域。指标能自我编写吗？说清什么使答案优秀很难，但指出答案的问题则相对容易，因此我们演化的指标是一个小型Python操作符池，每个操作符为一个命名的缺陷标记候选答案，或弃权，并投票。直接让模型生成操作符是行不通的：183个候选仅实现96种不同行为，且来自一个巨大空间中的狭窄区域。EvalCEGAR转而借鉴程序验证中的反例引导抽象细化方法。它将操作符池视为一种抽象，并搜索碰撞——即两个答案在操作符评分下相同，但一个正确一个错误。该配对（而非提示）成为创作请求，当碰撞击败所有尝试时，循环会扩大操作符的定义范围。

    arXiv:2608.18744v1 Announce Type: new  Abstract: Agents improve quickly against a reliable automatic metric and stall without one, and the applications that need them most, report generation among them, are the ones nobody knows how to score. Can the metric write itself? Saying what makes an answer good is hard; pointing at something wrong with one is easier, so the metric we evolve is a pool of small Python operators that each flag a candidate for one named defect, or abstain, and vote. Asking a model for operators directly does not work: 183 candidates realise only 96 distinct behaviours, from one narrow region of an enormous space. EvalCEGAR instead borrows counterexample-guided abstraction refinement from program verification. It reads the pool as an abstraction and searches for a collision, two answers the operators score identically, one correct and one not. That pair, not a prompt, is the authoring request, and when a collision defeats every attempt the loop widens what an opera
    
[^61]: 基于多智能体的企业自动化分析与洞察生成平台

    A Multi-Agent Platform for Automated Enterprise Analytics and Insight Generation

    [https://arxiv.org/abs/2608.18740](https://arxiv.org/abs/2608.18740)

    本文提出一个基于CrewAI的多智能体平台，通过五个顺序协作的AI智能体实现对话式商业智能，在安全架构和参数化机制支持下，显著提升准确率和质量，超越单智能体基线。

    

    arXiv:2608.18740v1 公告类型：新 摘要：本文提出了一种基于CrewAI [1]构建的多智能体框架，用于对话式商业智能。五个专门的AI智能体按顺序流水线工作，处理自然语言查询、检索和分析数据、通过模型上下文协议（MCP）[2]生成可视化，并提供可操作的洞察。该平台具备纵深防御安全架构，实现多租户数据隔离，以及查询参数化机制，可将对话式洞察转化为可复用的仪表板组件。在涵盖合成和企业生产数据集的300个端到端测试案例中评估，功能准确率达到95.3%，平均响应延迟为24秒，响应质量评分由LLM-as-a-Judge框架评估为4.52/5.0，无幻觉率为93.0%，与单智能体基线相比，准确率提高了22.6个百分点，质量提升了20.2%。

    arXiv:2608.18740v1 Announce Type: new  Abstract: This paper proposes a multi-agent framework built on CrewAI [1] for conversational business intelligence. Five specialized AI agents operate in a sequential pipeline to process natural language queries, retrieve and analyze data, generate visualizations via the Model Context Protocol (MCP) [2], and deliver actionable insights. The platform features a defense-in-depth security architecture for multi-tenant data isolation and a query parameterization mechanism for transforming conversational insights into reusable dashboard components. Evaluation across 300 end-to-end test cases spanning synthetic and production enterprise datasets demonstrates 95.3% functional accuracy, a mean response latency of 24 seconds, and a response quality score of 4.52/5.0 as assessed by an LLM-as-a-Judge framework, with a 93.0% hallucination-free rate, representing a 22.6 percentage point accuracy improvement and 20.2% quality gain over a single-agent baseline. 
    
[^62]: Flama：一个用于开发与部署生产级API、机器学习和LLM服务的Python框架

    Flama: a Python framework for development and deployment of production-ready APIs, machine learning, and LLM services

    [https://arxiv.org/abs/2608.18733](https://arxiv.org/abs/2608.18733)

    Flama是一个基于ASGI的Python框架，通过统一架构和七个子系统简化了生产级API、机器学习模型和LLM应用的开发与部署，实现了类型驱动和异步优先的编程体验。

    

    arXiv:2608.18733v1 公告类型：交叉 摘要：我们介绍了Flama，一个用于开发与部署生产级Web API、机器学习服务和大语言模型（LLM）应用的开源Python框架。Flama基于异步服务器网关接口（ASGI）构建，提供了一种类型驱动、异步优先的编程模型，将REST API开发、预测模型服务和生成式AI推理统一在一个架构中。它围绕七个子系统组织：一个基于组件的依赖注入系统，在启动时根据类型注解解析处理器参数；一个可插拔的模式层，通过单一适配器支持Pydantic、Marshmallow和Typesystem；一个自动CRUD生成器，将SQLAlchemy表和模式类转换为由仓储和工作单元模式支持的REST端点；一个可移植的二进制格式（.flm），用于打包scikit-learn、TensorFlow、PyTorch和Hugging Face Transformers模型及其元数据，实现零拷贝...

    arXiv:2608.18733v1 Announce Type: cross  Abstract: We present Flama, an open-source Python framework for developing and deploying production-ready web APIs, machine learning services, and large-language-model (LLM) applications. Built on the Asynchronous Server Gateway Interface (ASGI), Flama offers a type-driven, async-first programming model that unifies REST API development, predictive model serving, and generative AI inference in one architecture.   It is organised around seven subsystems: a component-based dependency injection system resolving handler parameters from type annotations at startup; a pluggable schema layer supporting Pydantic, Marshmallow and Typesystem behind a single adapter; an automatic CRUD generator turning a SQLAlchemy table and a schema class into REST endpoints backed by the Repository and Unit of Work patterns; a portable binary format (.flm) packaging models from scikit-learn, TensorFlow, PyTorch and Hugging Face Transformers with their metadata for zero-c
    
[^63]: 几个案例就够了：MedSAM3的标注高效LoRA微调实证研究

    A Few Cases Are All You Need: An Empirical Study of Annotation-Efficient LoRA Fine-Tuning of MedSAM3

    [https://arxiv.org/abs/2608.18731](https://arxiv.org/abs/2608.18731)

    本研究实证表明，仅需10个专家标注案例，通过LoRA微调MedSAM3即可在CT和MRI的腹部器官分割中达到临床可用性能，大幅降低标注成本。

    

    医学图像分割对于治疗计划和疾病评估等临床工作流程至关重要。虽然像TotalSegmentator和MRSegmentator这样的专业工具性能强劲，但它们需要大量标注数据集进行训练。医学基础模型通过大规模预训练提供了一种有前景的替代方案，减少了新任务的标注负担，但零样本性能仍然有限。通过低秩适配（LoRA）进行参数高效适配，能够以少量可训练参数实现高效特化，但一个关键问题仍然存在：需要多少专家标注案例才能达到临床有用的分割性能？我们通过使用仅1、2、5和10个标注案例，在CT和MRI中对五个腹部器官（肝脏、肾脏、脾脏、胆囊和胰腺）进行MedSAM3的LoRA适配，并在AMOS22数据集上评估来解决这一问题。仅用10个案例，模型就达到了性能...

    arXiv:2608.18731v1 Announce Type: cross  Abstract: Medical image segmentation is essential for clinical workflows such as treatment planning and disease assessment. While specialist tools like TotalSegmentator and MRSegmentator achieve strong performance, they require large annotated datasets for training. Medical foundation models offer a promising alternative through large-scale pretraining that reduces the annotation burden for new tasks, but zero-shot performance remains limited. Parameter-efficient adaptation via Low-Rank Adaptation (LoRA) enables efficient specialization with few trainable parameters, but a key question remains: how many expert-annotated cases are needed to achieve clinically useful segmentation performance? We address this by adapting MedSAM3 with LoRA for five abdominal organs (liver, kidneys, spleen, gallbladder, and pancreas) in CT and MRI using only 1, 2, 5, and 10 annotated cases, evaluating on AMOS22 dataset. With just 10 cases, models achieve performance 
    
[^64]: 预算优先的资费推荐（BFTR）：一个完整的电信套餐推荐算法框架，避免超额收费

    Budget-First Tariff Recommendation (BFTR): A Complete Algorithmic Framework for Telecom Plan Recommendation without Overcharging

    [https://arxiv.org/abs/2608.18723](https://arxiv.org/abs/2608.18723)

    BFTR提出了一种完整的算法框架，通过八种预算优先策略（含两种原创混合方法）保证电信套餐推荐不超额收费，并数学上证明了价格偏差为零。

    

    arXiv:2608.18723v1 公告类型：交叉 摘要：电信运营商传统上提供预定义的资费网格，迫使用户从有限的套餐集合中选择。本文提出了BFTR（预算优先的资费推荐），一个完整的算法框架，整合了八种预算优先策略，包括两种原创的混合方法：递归混合（条件插值）和背包优先混合（优先背包）。与现有方法通过上调价格以保证最低利润率不同，BFTR通过系统性地将最终价格与目录参考价格对齐，确保不存在超额收费。我们对每种策略进行了数学形式化，证明了对于任何正预算都存在一个报价，并证明了对于所有不使用带修正插值的策略，价格偏差（附加费）为零。一项详细的比较分析将BFTR与十种主要现有资费模型在十个维度上进行了对比。在974个c的数据集上的实验...

    arXiv:2608.18723v1 Announce Type: cross  Abstract: Telecom operators traditionally offer predefined tariff grids, forcing users to choose from a limited set of plans. This paper proposes BFTR (Budget-First Tariff Recommendation), a complete algorithmic framework integrating eight Budget-First strategies, including two original hybrid approaches: Recursive Hybrid (conditional interpolation) and Knapsack-First Hybrid (priority knapsack). Unlike existing approaches that adjust prices upward to guarantee a minimum margin, BFTR guarantees the absence of overcharging by systematically aligning the final price with the catalog reference price. We mathematically formalize each strategy, prove the existence of an offer for any positive budget, and prove that the price deviation (surcharge) is zero for all strategies that do not use interpolation with correction. A detailed comparative analysis confronts BFTR to ten main existing tariff models on ten dimensions. Experiments on a dataset of 974 c
    
[^65]: 能力，而非准确性：技能优化中无参考评判门的一种诊断方法

    Competence, Not Accuracy: A Diagnostic for Reference-Free Judge Gates in Skill Optimization

    [https://arxiv.org/abs/2608.18719](https://arxiv.org/abs/2608.18719)

    本文提出一种诊断方法，通过形式化无参考LLM评判器为潜在求解器，推导出其区分正确与错误答案的能力上限，并证明存在必要条件$c > 1/k$，为在技能优化中替换自动验证器提供理论依据。

    

    摘要：文本空间技能优化通过演化自然语言技能文档来适应冻结的智能体，并通过验证门接受每个候选方案。现有门依赖于可验证的奖励，这将这些方法限制在具有自动验证器的任务中。用LLM评判门替换验证器将解除这一限制，但此类门是否携带可用信号尚未测试。我们提出了一个前置问题：在将评判器放入循环之前，我们能否判断其分数是否能区分正确与错误的答案？我们将无参考评判器形式化为一个潜在求解器——其判定依赖于与其自身结论的一致性，因此其评估能力受限于其求解能力。该模型给出了评判器能力$c$和答案空间大小$k$下可区分性（ROC-AUC）的闭式界限、必要条件$c > 1/k$，以及边际AUC受项目难度混淆的结果。

    arXiv:2608.18719v1 Announce Type: new  Abstract: Text-space skill optimization adapts a frozen agent by evolving a natural-language skill document, accepting each candidate through a validation gate. Existing gates rely on verifiable rewards, confining these methods to tasks with an automatic verifier. Replacing the verifier with an LLM-judge gate would lift that restriction, but whether such a gate carries usable signal is untested. We ask a prior question: can we tell, before placing a judge in the loop, whether its scores separate correct from incorrect answers at all? We formalize a reference-free judge as a latent solver -- its verdict rests on agreement with whatever it would itself conclude, so its capacity to evaluate is bounded by its capacity to solve. The model yields a closed-form bound on discriminability (ROC-AUC) in the judge's competence $c$ and answer-space size $k$, a necessary condition $c > 1/k$, and the result that the marginal AUC is confounded by item difficulty 
    
[^66]: CutMix对语义分割中可靠性与鲁棒性的影响

    The Impact of CutMix on Reliability and Robustness in Semantic Segmentation

    [https://arxiv.org/abs/2608.18715](https://arxiv.org/abs/2608.18715)

    本文系统分析发现CutMix在语义分割中对精度影响轻微，但能持续提升模型可靠性，尤其在不确定性质量上表现突出。

    

    摘要：确保不仅具有高精度，而且具有可靠和鲁棒的预测，对于语义分割模型在安全关键应用（如自动驾驶）中的部署至关重要。尽管CutMix（一种简单而强大的数据增强策略）被广泛使用，但其对密集预测任务中可靠性和鲁棒性的影响仍未得到探索。受最近研究发现半监督分割方法（其中CutMix是核心组件）可能严重降低可靠性的启发，本研究孤立并系统分析了CutMix对分割精度、校准和不确定性质量的影响。我们评估了两种代表性架构，基于CNN的DeepLabV3+和基于Transformer的SegFormer，涵盖域内和域外场景。我们的结果表明，CutMix对分割精度影响较小，但持续改善可靠性，尤其是在不确定性方面。

    arXiv:2608.18715v1 Announce Type: cross  Abstract: Ensuring not only high accuracy but also reliable and robust predictions is critical for the deployment of semantic segmentation models in safety-critical applications such as autonomous driving. Despite the widespread use of CutMix - a simple yet powerful data augmentation strategy - its effect on the reliability and robustness in dense predictions tasks remains unexplored. Motivated by recent findings that semi-supervised segmentation methods, where CutMix is a core component, can severely degrade reliability, this study isolates and systematically analyzes the influence of CutMix on segmentation accuracy, calibration, and uncertainty quality. We evaluate two representative architectures, the CNN-based DeepLabV3+ and the transformer-based SegFormer, across both in-domain and out-of-domain scenarios. Our results show that CutMix has only a minor impact on segmentation accuracy but consistently improves the reliability, particularly un
    
[^67]: 不确定性量化与基础模型在语义分割中的综合评述

    A Critical Synthesis of Uncertainty Quantification and Foundation Models for Semantic Segmentation

    [https://arxiv.org/abs/2608.18709](https://arxiv.org/abs/2608.18709)

    本文首次系统评估了不确定性量化方法在语义分割基础模型中的应用，并建立了轻量级基线，为提升模型在安全关键任务中的可靠性提供了重要参考。

    

    基础模型正日益突破不久前看似不可能的限制，通过实现前所未有的准确性和跨域泛化能力。然而，它们缺乏可解释性、倾向于过度自信，以及对现实世界域偏移的敏感性，对安全和关键任务应用构成了严峻挑战。不确定性量化（UQ）提供了一种原则性的方法来解决这些问题，但其在分割基础模型中的集成尚未被探索。在本文中，我们首次对应用于语义分割基础模型的UQ方法进行了系统性评估。我们在预训练的SAM2编码器之上微调了一个轻量级DPT解码器，以建立一个简单但具有竞争力的基线，并基准测试了四种代表性的UQ方法——蒙特卡洛丢弃、深度子集成、测试时增强和证据深度学习——在Cityscapes、NYUv2以及两个具有挑战性的室外数据集上进行了评估。

    arXiv:2608.18709v1 Announce Type: cross  Abstract: Foundation models are increasingly breaking what seemed to be impossible not long ago by enabling unprecedented accuracy and cross-domain generalization. Yet their lack of interpretability, tendency to be overconfident, and sensitivity to real-world domain shifts pose critical challenges for safety- and mission-critical applications. Uncertainty quantification (UQ) offers a principled way to address these issues, but its integration into segmentation foundation models has yet to be explored. In this paper we present the first systematic evaluation of UQ methods applied to a foundation model for semantic segmentation. We fine-tune a lightweight DPT decoder on top of the pretrained SAM2 encoder to establish a simple yet competitive baseline and benchmark four representative UQ approaches - Monte Carlo Dropout, Deep Sub-Ensemble, Test-Time Augmentation, and Evidential Deep Learning - across Cityscapes, NYUv2, and two challenging out-of-do
    
[^68]: MemFuse：从碎片化观察中的多源记忆融合

    MemFuse: Multi-Source Memory Fusion from Fragmented Observations

    [https://arxiv.org/abs/2608.18704](https://arxiv.org/abs/2608.18704)

    本文提出了MemFuseBench基准和MemFuse记忆系统，旨在解决多源碎片化观察中的长期记忆融合问题，通过保留源级证据来增强时间推理和跨源证据整合能力。

    

    arXiv:2608.18704v1 公告类型：交叉 摘要：长期记忆对于在扩展交互中运行的代理至关重要，但现有的记忆系统和基准主要关注单源文本历史。然而，在现实环境中，相关信息通常分散在应用程序和设备之间，以及用户和时间之间，要求代理将分散的观察整合为连贯的情节记忆，同时保留其来源出处。为了解决这些差距，我们引入了 **MemFuseBench**，一个用于*多源记忆融合*的基准。MemFuseBench 通过场景到传感器（Scene-to-Sensor）流水线构建，该流水线将可控场景合成为带来源标签的观察、基于证据的问题和对抗性干扰项。它能够系统评估时间推理、跨源证据融合以及对噪声的鲁棒性。我们进一步提出了 **MemFuse**，一种结构化记忆系统，即使在碎片化观察中也能保留源级证据。

    arXiv:2608.18704v1 Announce Type: cross  Abstract: Long-term memory is essential for agents that operate across extended interactions, yet existing memory systems and benchmarks predominantly focus on single-source textual histories. In realistic settings, however, relevant information is often fragmented across applications and devices, as well as across users and time, requiring agents to integrate dispersed observations into coherent episodic memories while preserving their source provenance. To address these gaps, we introduce **MemFuseBench**, a benchmark for *multi-source memory fusion*. MemFuseBench is built with a Scene-to-Sensor pipeline that synthesizes controllable scenarios into source-tagged observations, evidence-grounded questions, and adversarial distractors. It enables systematic evaluation of temporal reasoning, cross-source evidence fusion, and robustness to noise. We further propose **MemFuse**, a structured memory system that preserves source-level evidence in even
    
[^69]: 迭代微调对复杂历史梵文手稿转录准确性的影响

    Impact of Iterative Fine-Tuning on Transcription Accuracy in Complex Historical Sanskrit Manuscripts

    [https://arxiv.org/abs/2608.18696](https://arxiv.org/abs/2608.18696)

    本文提出了一种可迭代微调的传统OCR流水线，通过适应目标历史手稿的布局和外观特征，显著提高了复杂梵文手稿的转录准确性，并减少了昂贵的人工标注成本。

    

    摘要：将历史手写手稿中的文本数字化，对于使其易于访问、保存以及让历史学者以新方式研究是必要的。然而，由于特定时期的书写风格、页面纹理、相机噪声和其他干扰因素，历史手稿通常表现出复杂的异构布局和非标准外观，这使得对其执行OCR变得困难。为应对这一挑战，我们引入了一种本地传统OCR流水线，该流水线可在布局级别和外观级别对目标手稿进行迭代微调。通过适应目标手稿的分布，所提出的传统OCR流水线在后续页面上做出更好的预测，从而迭代减少人工标注工作量，而人工标注因需要历史领域专业知识而昂贵且耗时。使用该流水线，我们从三份复杂的历史梵文手稿中数字化了文本。

    arXiv:2608.18696v1 Announce Type: cross  Abstract: Digitizing the text from handwritten historical manuscripts is required to make them easily accessible, preservable, and to enable historical scholars to study them in new ways. Historical manuscripts, however, often exhibit complex heterogeneous layouts and non-standard appearance due to period-specific writing styles, page textures, camera noise, and other nuisance factors, making them difficult to perform OCR on. To tackle this challenge, we introduce a local traditional OCR pipeline, which can be iteratively fine-tuned on the target manuscript at the layout-level and the appearance-level. By adapting to the target manuscript distribution, the proposed Traditional OCR pipeline makes better predictions on subsequent pages, causing iterative reduction in human annotation effort, which is expensive and time-consuming as it requires historical domain expertise. Using this pipeline, we digitize text from three complex historical Sanskrit
    
[^70]: 通过建模时间表示实现历史图像组合检索

    Composed Historical Image Retrieval by Modeling Temporal Representations

    [https://arxiv.org/abs/2608.18694](https://arxiv.org/abs/2608.18694)

    本文提出了一种名为TDIR的表示学习方法，通过正交子空间将历史图像分解为时间和内容成分，在保留时间结构的同时维持检索性能，并提供了数学基础与误差分析。

    

    摘要：虽然时间呈线性演变，但神经嵌入空间的几何结构本质上是多维的，通常混乱且难以解释。原则上，可以将嵌入空间约束为单一时间维度；然而，这种缩减会牺牲下游任务的性能，因为一维嵌入无法保留足够的表达能力。本文探讨是否可能学习既保留时间结构又对图像和物体检索有效的表示，并通过构建此类系统的数学基础来回答这一问题。我们提出了时间可分解图像表示（TDIR），一种表示学习算法，通过正交子空间将历史照片分解为独立的日期和内容组件。我们定义并证明了此类分解可实现的条件，并刻画了所引入的误差。

    arXiv:2608.18694v1 Announce Type: cross  Abstract: While time evolves linearly, the geometry of neural embedding spaces is inherently multi-dimensional, often chaotic, and difficult to interpret. In principle, one could constrain an embedding space to a single temporal dimension; however, such a reduction would sacrifice performance on downstream tasks, as one-dimensional embeddings cannot retain sufficient expressive capacity. This paper asks whether it is possible to learn representations that preserve temporal structure while remaining effective for image and object retrieval, and answers this question by building the mathematical foundations of such a system. We propose Temporally Decomposable Image Representations (TDIR), a representation learning algorithm that decomposes historical photographs into separate date and content components through orthogonal subspaces. We define and prove the conditions under which such a decomposition is achievable, characterize the error incurred w
    
[^71]: 欧洲气候雄心面临审视：来自深度学习排放预测的证据

    Europe's Climate Ambition Under Scrutiny: Evidence from Deep Learning Emission Projections

    [https://arxiv.org/abs/2608.18690](https://arxiv.org/abs/2608.18690)

    深度学习预测显示，欧盟27国按当前趋势将无法实现2030年减排目标，排放缺口达35%，其中交通部门的结构性惯性是主要障碍。

    

    欧盟承诺到2030年将温室气体排放量比1990年水平降低55%，但当前趋势是否与该雄心相符仍不确定。我们应用深度学习技术，结合欧盟27个成员国截至2023年的高分辨率社会经济和部门数据，预测当前趋势下的部门二氧化碳排放轨迹，在不对政策环境的步伐或有效性进行超出历史数据已反映的假设变化的情况下，外推观察到的部门发展势头。我们预测，欧盟27国的排放量将超出2030年目标35%（缺口达620百万吨二氧化碳），仅有少数国家的排放轨迹与该集团的承诺一致。虽然电力部门在可再生能源转型推动下实现了与目标一致的减排，但交通部门进展甚微，到2030年将占总排放量的三分之一以上，反映出成员国之间的结构性惯性。

    arXiv:2608.18690v1 Announce Type: cross  Abstract: The European Union has committed to reducing greenhouse gas emissions 55% below 1990 levels by 2030, but whether current trends are compatible with this ambition remains uncertain. We apply deep learning to high-resolution socioeconomic and sectoral data across EU27 member states till 2023 to project sectoral CO$_2$ trajectories under current trends, extrapolating observed sectoral momentum without assuming changes in the pace or effectiveness of the policy environment beyond what is already reflected in historical data. We project that EU27 emissions will exceed the 2030 target by 35% (620 Mt CO$_2$ shortfall), with only a small minority of countries on trajectories consistent with the bloc's commitments. While the Power sector achieves target-consistent reductions driven by the renewable transition, Mobility shows minimal progress and accounts for over a third of total emissions by 2030, reflecting a structural inertia across member 
    
[^72]: Aslema在NADI 2026：通过少样本增强进行口语语言理解

    Aslema at NADI 2026: Augmentation through Fewshot for SLU

    [https://arxiv.org/abs/2608.18689](https://arxiv.org/abs/2608.18689)

    本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。

    

    arXiv:2608.18689v1 公告类型：交叉 摘要：我们介绍了Aslema，这是我们为NADI 2026共享任务5开发的系统，该任务包含两个子任务：意图识别和槽位填充。我们在零样本设置下评估了四种全模态大型语言模型，并将它们与微调模型进行了比较。结果表明，微调始终优于零样本推理。我们进一步探索了合成数据增强，通过使用大型语言模型生成具有文化背景的突尼斯Derja话语，然后通过语音克隆生成合成语音。将这种合成数据纳入后，两个任务的性能均得到提升。我们最终提交的系统基于Qwen3-Omni-30B，并使用原始数据和合成数据的混合进行训练，在开发测试集上实现了86.8%的意图准确率和34.7的词错误率。在官方测试集上，它在槽位填充任务中排名第一（59.5 CoER），在意图识别任务中排名第四（8个团队中，准确率66.1%）。我们发布了实验脚本，并将很快共享合成数据集以支持进一步研究。

    arXiv:2608.18689v1 Announce Type: cross  Abstract: We present Aslema, our system for NADI 2026 Shared Task 5, which consists of two subtasks: intent recognition and slot filling. We evaluate four omni LLMs in a zero-shot setting and compare them with fine-tuned models. Our results show that fine-tuning consistently outperforms zero-shot inference. We further explore synthetic data augmentation by using an LLM to generate culturally grounded Tunisian Derja utterances, followed by voice cloning to generate synthetic speech. Incorporating this synthetic data improves performance on both tasks. Our final submitted system, based on Qwen3-Omni-30B and trained with a mixture of original and synthetic data, achieves 86.8% intent accuracy and 34.7 WER on the devtest split. On the official test set it ranks 1st in slot filling (59.5 CoER) and 4th among 8 teams in intent recognition (66.1% accuracy). We release our experimental scripts and will soon share the synthetic dataset to support further 
    
[^73]: 反向回合策略优化：稳定智能体强化学习训练

    RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training

    [https://arxiv.org/abs/2608.18682](https://arxiv.org/abs/2608.18682)

    RTPO通过将多轮轨迹组织为反向树并进行回合级更新，解决了多轮RL训练中的稳定性问题，显著提升了性能。

    

    arXiv:2608.18682v1 公告类型：新 摘要：使用强化学习（RL）训练多轮智能体工作流程，使大型语言模型能够执行复杂推理、使用外部工具，并在单轮设置之外进行迭代搜索。然而，多轮RL训练仍然高度不稳定，随着轮数增加，常常导致严重的性能下降。通过理论分析，我们识别出三个紧密耦合的不稳定性来源：滚动训练上下文不匹配、稀疏终端奖励下的弱回合级信用分配，以及在不同策略版本下优化短轨迹和长轨迹时的异步策略漂移。我们表明，这些问题在扁平化轨迹优化中具有共同的结构性根源，并通过统一的逆向回合公式来解决它们。我们提出了反向回合策略优化（RTPO），它将多轮滚动组织为稀疏反向树，并在回合级执行策略更新。

    arXiv:2608.18682v1 Announce Type: new  Abstract: Training multi-turn agentic workflows with reinforcement learning (RL) enables large language models to perform complex reasoning, use external tools, and conduct iterative search beyond single-turn settings. Yet multi-turn RL training remains highly unstable, often causing severe performance degradation as the number of turns increases. Through theoretical analysis, we identify three tightly coupled sources of instability: rollout-training context mismatch, weak turn-level credit assignment under sparse terminal rewards, and asynchronous policy drift when short and long trajectories are optimized under different policy versions. We show that these issues share a common structural origin in flattened trajectory optimization and address them through a unified reverse-turn formulation. We propose Reverse-Turn Policy Optimization (RTPO), which organizes multi-turn rollouts as sparse reverse trees and performs turn-level policy updates in te
    
[^74]: 三余工作室：面向艺术史叙事建构的多智能体系统

    Sanyu Studio: A Multi-Agent System for Art-Historical Narrative Construction

    [https://arxiv.org/abs/2608.18677](https://arxiv.org/abs/2608.18677)

    本文提出“三余工作室”多智能体系统，证明在历史证据有限时，AI能促进多元艺术史叙事建构并增强人的能动性。

    

    针对生成式AI可能使艺术解读标准化的担忧，本文探讨了基于LLM的交互是否能支持多元艺术史叙事建构。我们提出了“三余工作室”（Sanyu Studio），这是一个多智能体对话系统，将321幅三余油画建模为具有事实、阐释、组织和记忆过滤机制的智能体。基于与八位艺术院校参与者进行的为期七天的研讨会，研究表明，用户的提示、证据组织和认知倾向塑造了不同但连贯的数字三余版本。研究结果表明，在历史证据有限的条件下，AI可以增强人的能动性，并为公众提供进入艺术史解读的互动入口。

    arXiv:2608.18677v1 Announce Type: new  Abstract: Amid concerns that generative AI may standardize art interpretation, this paper examines whether LLM-based interaction can support plural art-historical narrative construction. We present Sanyu Studio, a multi-agent dialogue system that models 321 Sanyu oil paintings as agents with fact, interpretation, organization, and memory-filtering mechanisms. Based on a seven-day workshop with eight art-university participants, the study shows that user prompts, evidence organization, and cognitive tendencies shaped divergent yet coherent versions of digital Sanyu. The findings suggest that, under conditions of limited historical evidence, AI can amplify human agency and offer public audiences an interactive entry point into art-historical interpretation.
    
[^75]: 不确定时变奖励的定向运动问题：面向日常服务机器人的框架与基准

    Orienteering Problem with Uncertain Time-Varying Rewards: Framework and Benchmark for Everyday Service Robotics

    [https://arxiv.org/abs/2608.18672](https://arxiv.org/abs/2608.18672)

    本文提出了一种新的定向运动问题变体，允许奖励不确定且随时间变化，并通过三种规划器和移动机器人基准验证了规划视野与适应性之间的权衡。

    

    我们提出了不确定时变奖励的定向运动问题（OP-UTVR），这是定向运动问题（OP）的一种新变体。尽管大多数现有的OP公式假设奖励事先已知，但实际应用中奖励是不确定且随时间变化的，例如配送代理面临的客户需求变化。OP-UTVR放宽了这一假设，允许代理通过观察来估计奖励动态并预测未来奖励，从而在随机奖励变化和不可避免的预测误差下做出明智的路由决策。我们使用三种规划器来解决这个问题，它们在规划视野和在线适应性方面有所不同，并推导了它们在奖励随机性下的性能理论边界。我们还引入了一个面向OP-UTVR的移动服务机器人基准，其中机器人在室内环境中在行人之间导航。实验揭示了规划视野和适应性之间的权衡。

    arXiv:2608.18672v1 Announce Type: cross  Abstract: We present the orienteering problem with uncertain time-varying rewards (OP-UTVR), a novel variant of the orienteering problem (OP). While most existing OP formulations assume rewards to be known in advance, practical applications involve uncertain and time-varying rewards, as with shifting customer demand for delivery agents. OP-UTVR relaxes this assumption by allowing agents to estimate reward dynamics from observations and forecast future rewards. This enables informed routing decisions despite stochastic reward changes and inevitable prediction errors. We address this problem using three planners that differ in planning horizon and online adaptivity, and derive theoretical bounds on their performance under reward stochasticity. We further introduce a mobile service robot benchmark for OP-UTVR, where a robot navigates among pedestrians in indoor environments. Experiments reveal trade-offs between planning horizon and adaptivity, and
    
[^76]: 候选命运核算实现透明传感器诊断流水线搜索

    Candidate-Fate Accounting for Transparent Sensor Diagnostic Pipeline Search

    [https://arxiv.org/abs/2608.18665](https://arxiv.org/abs/2608.18665)

    本文提出候选命运核算框架，通过记录搜索过程中每个候选的完整命运（包括无效、剪枝等），增强了传感器诊断流水线搜索的透明性和可审计性。

    

    arXiv:2608.18665v1 公告类型：新 摘要：工业传感器诊断依赖于预处理、表示和分类流水线，使得自动化流水线搜索对于降低手动设计成本非常有用。然而，现有的自动化机器学习/深度学习（AutoML/AutoDL）报告通常只保留拟合的试验、分数和获胜者，省略了无效、被剪枝、跳过、缓存或未拟合的生成候选。这种省略限制了审阅者检查信号约束、预算使用和未评估的合法替代方案的能力。为解决这一问题，我们提出了候选命运核算，一种用于诊断搜索轨迹的候选级审计框架。它将每个观察到的候选记录为可审计证据：哈希合并重复观察，合法性检查标记无效候选，分配理由解释预算决策，以及一个封闭的命运账本为每个候选分配一个最终命运。在三个轴承诊断数据集上的实验表明，该框架...

    arXiv:2608.18665v1 Announce Type: new  Abstract: Industrial sensor diagnostics relies on preprocessing, representation, and classification pipelines, making automated pipeline search useful for reducing manual design cost. However, existing automated machine/deep learning (AutoML/AutoDL) reports typically retain only fitted trials, scores, and winners, omitting generated candidates that are invalid, pruned, skipped, cached, or unfitted. This omission limits reviewers' ability to check signal constraints, budget use, and unevaluated legal alternatives. To address this, we propose candidate-fate accounting, a candidate-level audit framework for diagnostic search traces. It records each observed candidate as auditable evidence: hashes merge repeated observations, legality checks flag invalid candidates, allocation rationales explain budget decisions, and a closed fate ledger assigns one terminal fate to each candidate. Experiments on three bearing-diagnostic datasets show that the framewo
    
[^77]: 基于变化点感知的PPG血压估计评估与再校准

    Change Point--Aware Evaluation and Re-Calibration of PPG-Based Blood Pressure Estimation

    [https://arxiv.org/abs/2608.18639](https://arxiv.org/abs/2608.18639)

    本文提出了一种基于变化点检测的波动感知评估框架，揭示了现有PPG血压估计模型在血压快速波动期间性能显著下降的问题，并证明周期性再校准能有效缓解这一退化。

    

    使用光电容积脉搏波（PPG）进行无创连续血压（BP）监测是袖带式测量的一个有前景的替代方案。然而，现有的基于PPG的血压估计研究主要依赖于在整个评估区间上计算出的汇总性能指标（如平均绝对误差），这可能会掩盖快速血压波动期间的模型失败，并限制其临床相关性。在这项工作中，我们提出了一种基于时间序列变化点检测的波动感知评估框架，用于PPG血压估计。我们不是采用启发式的血压阈值（例如，ΔBP > 10 mmHg），而是通过捕捉血压轨迹中的突然分布变化来识别血压变化点，并专门在这些波动期间评估估计性能。我们的分析表明，几种最先进的模型在血压变化点附近表现出显著的性能退化，并且周期性重新校准可以缓解这种退化。

    arXiv:2608.18639v1 Announce Type: cross  Abstract: Non-invasive continuous blood pressure (BP) monitoring using photoplethysmography (PPG) is a promising alternative to cuff-based measurements. However, existing PPG-based BP estimation studies predominantly rely on aggregated performance metrics (e.g., mean absolute error) computed over entire evaluation intervals, which can obscure model failures during rapid BP fluctuations and limit clinical relevance. In this work, we propose a fluctuation-aware evaluation framework for PPG-based BP estimation based on time-series change point detection. Instead of heuristic BP thresholding (e.g., $\Delta\mathrm{BP} > 10\mathrm{mmHg}$), we identify BP change points by capturing abrupt distributional shifts in BP trajectories and evaluate estimation performance specifically during these fluctuation periods. Our analysis shows that several state-of-the-art models exhibit substantial performance degradation around BP change points, and that periodic t
    
[^78]: 大语言模型中不确定条件下的偏好推理

    Preference Reasoning under Indeterminacy in Large Language Models

    [https://arxiv.org/abs/2608.18631](https://arxiv.org/abs/2608.18631)

    本文指出大语言模型在偏好推理中无法有效处理不确定性，尤其是认知和结构不确定性，导致其在不同任务中系统性地失败。

    

    摘要：arXiv:2608.18631v1 公告类型：新 摘要：随着大语言模型演变为决策代理，对偏好进行推理的能力成为对齐、协调和集体智能的基础。然而，与标准基准不同，现实世界中的偏好推理本质上是不确定的：信息可能不完整，有效解决方案可能不存在。我们认为，不确定性而非仅仅是正确性，是人工智能推理的核心挑战。我们沿着两个轴形式化了这一挑战：（i）认知不确定性，源于不完整、部分或表达性偏好；（ii）结构性不确定性，源于标准社会选择概念下解决方案的不存在。在任务层级中，我们展示了最先进的语言模型系统性地无法区分确定和不确定的实例，即使在验证设置中也表现出校准不当的推理。

    arXiv:2608.18631v1 Announce Type: new  Abstract: As large language models evolve into decision-making agents, the ability to reason over preferences becomes fundamental to alignment, coordination, and collective intelligence. Yet, unlike standard benchmarks, real-world preference reasoning is inherently indeterminate: information may be incomplete, and valid solutions may not exist. We argue that indeterminacy, rather than correctness alone, is a central challenge for AI reasoning. We formalize this challenge along two axes, (i) epistemic indeterminacy, arising from incomplete, partial, or expressive preferences, and (ii) structural indeterminacy, arising from the non-existence of solutions under standard social choice concepts. Across a hierarchy of tasks, we show that state-of-the-art language models systematically fail to distinguish between determined and undetermined instances, exhibiting miscalibrated reasoning even in verification settings.
    
[^79]: CTIFoundry：面向网络威胁情报的智能体原生语料脚手架

    CTIFoundry: An Agent-Native Corpus Scaffold for Cyber Threat Intelligence

    [https://arxiv.org/abs/2608.18613](https://arxiv.org/abs/2608.18613)

    本文提出CTIFoundry，通过构建一个智能体原生的语料脚手架（包括确定性本体图和跨度锚定报告层），解决了LLM智能体在网络威胁情报调查中因语料底层结构缺失而导致的瓶颈问题。

    

    摘要：arXiv:2608.18613v1 公告类型：新 摘要：网络威胁情报（CTI）正日益被LLM智能体而非人类分析师所消费，这些智能体在查询时组合多步骤调查。这一转变的“工具侧”已经迅速成熟（规划循环、工具协议、上下文管理），但“语料侧”尚未跟上：威胁报告和漏洞数据库仍以检索增强生成的方式打包，作为嵌入索引背后的不透明块。我们认为，这一底层结构（而非模型能力）是智能体化CTI调查的瓶颈，并提出了CTIFoundry，一个智能体原生的语料脚手架。在构建时，CTIFoundry物化了CTI语料的潜在结构：一个基于四个权威知识库（CVE、CWE、CAPEC、ATT&CK）的确定性本体图，其官方交叉引用成为有类型的、可遍历的边；一个跨度锚定的报告层，其规范化的、别名解析的跨厂商实体索引了携带溯源信息的块；以及混合检索机制。

    arXiv:2608.18613v1 Announce Type: new  Abstract: Cyber threat intelligence (CTI) is increasingly consumed not by human analysts but by LLM agents that compose multi-step investigations at query time. The harness side of this shift has matured rapidly (planning loops, tool protocols, context management), but the corpus side has not: threat reports and vulnerability databases are still packaged for retrieval-augmented generation, as opaque chunks behind an embedding index. We argue that this substrate, not model capability, is the bottleneck on agentic CTI investigation, and present CTIFoundry, an agent-native corpus scaffold. At build time, CTIFoundry materializes the latent structure of a CTI corpus: a deterministic ontology graph over four authoritative knowledge bases (CVE, CWE, CAPEC, ATT&CK) whose official cross-references become typed, traversable edges; a span-grounded report layer whose canonical, alias-resolved cross-vendor entities index provenance-carrying chunks; and hybrid 
    
[^80]: 去噪感知反演：揭示噪声保护文本嵌入中的隐私风险

    Denoising-Aware Inversion: Revealing Privacy Risks in Noise-Protected Text Embeddings

    [https://arxiv.org/abs/2608.18610](https://arxiv.org/abs/2608.18610)

    本文首次探讨了噪声保护下文本嵌入反演的攻击场景，指出现有生成方法在仅有噪声嵌入时失效，并揭示了自适应攻击可绕过高斯噪声防御的隐私风险。

    

    稠密文本嵌入因其紧凑且语义丰富的表示，被广泛用于数据挖掘、检索和下游机器学习系统中，但最近的嵌入反演攻击表明，它们可能暴露原始文本的大量信息，导致严重的隐私泄露风险。一种常见的防御方法是通过添加高斯噪声来发布扰动嵌入，这种方法简单且对标准反演攻击有效，并且不会显著降低下游任务的嵌入效用。然而，尚不清楚这种噪声保护的嵌入在面对明确考虑扰动过程的自适应攻击者时是否足够安全。在本文中，我们研究了噪声保护设置下的文本嵌入反演问题，其中攻击者只能观察到带噪声的嵌入，并且无法访问干净的嵌入目标。我们首先分析了现有生成性反演方法在此场景下失效的原因。

    arXiv:2608.18610v1 Announce Type: cross  Abstract: Dense text embeddings are widely used in data mining, retrieval, and downstream machine learning systems due to their compact and semantically rich representations, but recent embedding inversion attacks have shown that they can expose substantial information about the original text, leading to serious privacy leakage risks. A common defense is to release perturbed embeddings by adding Gaussian noise, which is simple yet effective against standard inversion attacks and does not significantly degrade embedding utility for downstream tasks. However, it remains unclear whether such noise-protected embeddings are sufficiently safe against adaptive attackers that explicitly account for the perturbation process. In this paper, we study text embedding inversion in a noise-protected setting, where the attacker can observe only noisy embeddings and has no access to clean embedding targets. We first analyze why existing generative inversion meth
    
[^81]: 轻量级多模态模型能否评估LLM推理性能？面向计算最优文档推理的研究

    Can a Lightweight Multimodal Model Estimate LLM Reasoning Performance? A Study for Compute-Optimal Document Inference

    [https://arxiv.org/abs/2608.18591](https://arxiv.org/abs/2608.18591)

    本文提出BudgetDoc基准和轻量级DRB模型，通过预测LLM在不同推理预算下的性能，实现动态预算分配，在保持或提升准确性的同时大幅降低成本，并展现出跨模型泛化能力。

    

    arXiv:2608.18591v1 公告类型：新 摘要：将推理预算均匀分配给大型语言模型（LLM）既昂贵又容易产生过度思考的惩罚，尤其是在视觉布局驱动复杂性的文档任务中。为解决这一问题，我们引入了BudgetDoc，这是首个提供跨三个文档任务的模型-预算-性能权衡显式监督的多模态基准。利用BudgetDoc，我们训练了DRB（文档推理平衡器），一个约10亿参数的预飞行估计器（SigLIP-2 + Qwen3-0.6B），用于预测不同预算水平下的模型排序性能，达到了0.753的加权F1分数。在动态分配推理预算至五个前沿模型和三个数据集时，与始终使用最大预算的基线相比，DRB在15种配置中的9种中匹配或改善了F1分数，同时大幅降低了成本。最后，初步评估表明DRB在跨模型选择方面具有泛化潜力。

    arXiv:2608.18591v1 Announce Type: new  Abstract: Uniformly allocating inference reasoning budgets to LLMs is expensive and prone to over-thinking penalties; especially in document tasks where visual layouts drive complexity. To address this, we introduce BudgetDoc, the first multimodal benchmark providing explicit supervision for model-budget-performance trade-offs across three document tasks. Using BudgetDoc, we train DRB (Document-Reasoning Balancer), an approx. 1B-parameter pre-flight estimator (SigLIP-2 + Qwen3-0.6B) that predicts ordinal model performance across budget levels, achieving a 0.753 weighted F1. When dynamically allocating reasoning budgets across five frontier models and three datasets, DRB matches or improves F1 scores compared to always-maximum-budget baselines in 9 of 15 configurations while drastically reducing cost. Finally, preliminary evaluations demonstrate DRB's potential to generalize to cross-model selection.
    
[^82]: OmniHandwritingOCR：一个用于评估多模态大语言模型在手写OCR场景中的诊断基准

    OmniHandwritingOCR: A Diagnostic Benchmark for Evaluating Multimodal LLMs in Handwritten OCR Scenarios

    [https://arxiv.org/abs/2608.18586](https://arxiv.org/abs/2608.18586)

    本文提出了OmniHandwritingOCR，一个覆盖多语言手写、书写者错误和复杂数学表达式的手写OCR诊断基准，通过77.57K张标注图像和难度分层公式语料库，系统评估了多模态大语言模型在手写识别中的鲁棒性。

    

    摘要：多模态大语言模型（MLLMs）越来越多地被用作文档和知识处理流程中的OCR系统，但它们忠实读取真实手写内容的能力仍未得到充分探索。现有的OCR基准主要集中于印刷文本或干净的单行输入，对现实手写OCR场景的覆盖有限，例如多语言手写、书写者错误以及结构复杂的数学表达式。我们引入了OmniHandwritingOCR，这是一个用于评估MLLMs和OCR系统在手写OCR上的诊断基准。它涵盖了手写文本识别和手写数学表达式识别，涉及六个子任务和十二个子集，总计包含来自公共数据集和新收集的学生书写内容的77.57K张标注图像。其关键组成部分是一个难度分层的多行公式语料库，旨在测试在结构复杂度增加下的鲁棒性。我们评估了十三个开源模型以及一个专有模型。

    arXiv:2608.18586v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) are increasingly used as OCR systems in document and knowledge-processing pipelines, but their ability to faithfully read real handwriting remains underexplored. Existing OCR benchmarks focus largely on printed text or clean single-line inputs, leaving limited coverage of realistic handwritten OCR scenarios such as multilingual handwriting, writer errors, and structurally complex mathematical expressions. We introduce OmniHandwritingOCR, a diagnostic benchmark for evaluating MLLMs and OCR systems on handwritten OCR. It covers handwritten text recognition and handwritten mathematical expression recognition across six subtasks and twelve subsets, totaling 77.57K labeled images from public datasets and newly collected student writings. A key component is a difficulty-stratified multi-line formula corpus designed to test robustness under increasing structural complexity. We evaluate thirteen open- a
    
[^83]: 从存储到访问：通过显式提示与隐式推理实现大语言模型中参数化知识的可验证激活

    From Storage to Access: Verifiable Activation of Parametric Knowledge in LLMs via Explicit Priming and Implicit Reasoning

    [https://arxiv.org/abs/2608.18581](https://arxiv.org/abs/2608.18581)

    本文提出VAKE框架，通过两阶段强化学习实现大语言模型中参数化知识的可验证激活，利用显式提示生成可验证证据并迁移到隐式推理，以区分答案来源并提升事实问答的可靠性。

    

    尽管大语言模型（LLMs）在其参数中编码了丰富的事实知识，但可靠地回忆和验证这些知识仍然是事实问答中的关键瓶颈。现有的端到端方法将知识激发与推理纠缠在一起，使得难以判断正确答案是源自参数化知识还是输入上下文。为解决这一挑战，我们提出了VAKE（参数化知识的可验证激活），一种两阶段强化学习框架，通过显式提示外部化潜在参数化知识，并将习得的激发能力迁移到隐式推理中。给定一个查询和不足的检索子图，提示策略显式插入桥接三元组作为可验证证据，并由一个独立的冻结模型在增强子图上生成的答案所衍生的奖励提供监督。基于该学习到的策略，进一步实现了隐式推理的迁移。

    arXiv:2608.18581v1 Announce Type: cross  Abstract: Although Large Language Models (LLMs) encode rich factual knowledge in their parameters, reliably recalling and verifying such knowledge remains a key bottleneck in factual question answering. Existing end-to-end methods entangle knowledge elicitation with reasoning, making it difficult to determine whether correct answers arise from parametric knowledge or the input context. To address this challenge, we propose VAKE (Verifiable Activation of Parametric KnowledgE), a two-stage reinforcement-learning framework that externalizes latent parametric knowledge through explicit Priming and transfers the acquired elicitation capability to implicit Reasoning. Given a query and an insufficient retrieved subgraph, the Priming policy explicitly inserts bridging triples as verifiable evidence, with supervision provided by rewards derived from answers generated by a separate frozen model over the augmented subgraph. Building on the policy learned d
    
[^84]: FACET：在终端任务合成中保留源意图与可执行状态

    FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis

    [https://arxiv.org/abs/2608.18580](https://arxiv.org/abs/2608.18580)

    FACET框架通过重构代理技能并修复执行环境，确保终端任务合成中源意图和跨工件一致性得到保留，从而生成可解决且正确评估的任务。

    

    arXiv:2608.18580v1 公告类型：新  摘要：训练终端代理需要可扩展的可执行监督，然而合成高质量的终端任务仍然具有挑战性。每个任务将指令、初始化环境、参考解决方案和可执行验证器耦合在一起；如果这些工件从不一致的假设生成，则生成的任务可能无法解决或评估错误。同时，多阶段合成可能丢弃原始来源中编码的目标、依赖关系、状态转换和程序性约束。我们提出FACET（细粒度代理式可执行任务构建），一个解决信息保留和跨工件一致性问题的框架。FACET将相关的代理技能重构为连贯、信息丰富的场景，然后在生成最终任务工件之前实现并修复执行环境。生成的容器状态作为指令、解决方案和验证器的共享基础。

    arXiv:2608.18580v1 Announce Type: new  Abstract: Training terminal agents requires scalable executable supervision, yet synthesizing high-quality terminal tasks remains challenging. Each task couples an instruction, an initialized environment, a reference solution, and an executable verifier; if these artifacts are generated from inconsistent assumptions, the resulting task may be unsolvable or incorrectly evaluated. Meanwhile, multi-stage synthesis can discard the goals, dependencies, state transitions, and procedural constraints encoded in the original sources. We present FACET (Fine-grained Agentic Construction of Executable Tasks), a framework that addresses both information preservation and cross-artifact consistency. FACET reconstructs related agent skills into coherent, information-rich scenarios, then realizes and repairs the execution environment before generating the final task artifacts. The resulting container state serves as shared grounding for the instruction, solution, 
    
[^85]: MR-IQA-2：通过细粒度信用分配实现忠实的图像质量反映

    MR-IQA-2: Faithful Image Quality Reflection via Fine-Grained Credit Assignment

    [https://arxiv.org/abs/2608.18579](https://arxiv.org/abs/2608.18579)

    本文提出MR-IQA-2框架，通过解耦推理与评分的信用分配，并提供可验证监督，以增强盲图像质量评估中推理的忠实性和可靠性。

    

    多模态大语言模型（MLLMs）在图像质量评估（IQA）方面展现出强大潜力，通过提高质量评分与其底层推理之间的一致性来改善性能。然而，大多数方法仅依赖人工提供的评分来监督推理过程，很少检查推理是否忠实地反映了图像质量。仅凭评分准确性并不能确保推理的忠实性；共享奖励还会模糊监督来源，并且在评分偶然正确时可能强化不忠实的推理。为了提高盲图像质量评估的忠实性和可靠性，我们旨在：（1）解耦推理和评分的信用分配；（2）为忠实推理提供可验证的监督。我们提出了MR-IQA-2，一个演员-编辑-裁判框架，实现了推理-编辑-反映的操作化。演员为输入图像生成质量推理，编辑者根据识别出的质量因素修改图像，而裁判则冻结（保持不变）。

    arXiv:2608.18579v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) have shown strong potential for image quality assessment (IQA) by improving consistency between quality ratings and their underlying reasoning. However, most approaches supervise reasoning through human-provided ratings and rarely examine whether it faithfully reflects image quality. Rating accuracy alone does not ensure faithful reasoning; a shared reward also obscures supervision sources and may reinforce unfaithful reasoning when a correct rating occurs by chance. To improve the faithfulness and reliability of blind IQA, we aim to (1) decouple credit assignment for reasoning and rating and (2) provide verifiable supervision for faithful reasoning. We introduce MR-IQA-2, an actor-editor-judge framework that operationalizes reasoning-editing-reflection. The actor generates quality reasoning for an input image, and the editor revises the image according to the identified quality factors. A froze
    
[^86]: 网格细胞在海马位置表征中减少空间混叠的作用

    The Role of Grid Cells in Reducing Spatial Aliasing in Hippocampal Place Representations

    [https://arxiv.org/abs/2608.18569](https://arxiv.org/abs/2608.18569)

    本研究通过整合网格细胞信号与边界向量细胞驱动的位置细胞，显著减少了海马位置表征中的空间混叠，在三种环境中实现了94%至99%的混叠降低。

    

    arXiv:2608.18569v1 公告类型：交叉 摘要：当两个或多个不同位置产生高度相似的位置细胞表征时，会发生空间混叠，这主要是由于环境对称性或重复结构所致。当位置表征仅由边界向量细胞（BVC）输入构建时，此问题最为显著，因为对称或重复结构可能导致环境中多个位置产生不可区分的感官模式。本研究引入网格细胞信号以减轻此类设置中的空间混叠。由于网格细胞贡献周期性、内部生成的空间信号，这些信号独立于环境几何形状变化，因此在消歧感知上相同的位置中起关键作用。我们将多个分析构建的网格细胞模块与BVC驱动的位置细胞整合，并展示在三种环境（一个开放环境）中，相对于仅BVC基线，空间混叠减少了94%至99%。

    arXiv:2608.18569v1 Announce Type: cross  Abstract: Spatial aliasing occurs when two or more distinct locations produce highly similar place-cell representations, primarily due to environmental symmetry or repetitive structures. This issue is most pronounced when place representations are constructed solely from boundary vector cell (BVC) inputs, because symmetric or repetitive structures can yield indistinguishable sensory patterns across multiple locations in an environment. This work introduces grid cell signals to mitigate spatial aliasing in such settings. Because grid cells contribute periodic, internally generated spatial signals that vary independently of environmental geometry, they play a key role in disambiguating perceptually identical locations. We integrate multiple modules of analytically constructed grid cells with BVC-driven place cells and show that this leads to a 94--99% reduction in spatial aliasing relative to a BVC-only baseline across three environments: an open 
    
[^87]: MorphoGP：一种在潮汐影响下预测平衡海滩剖面的非参数框架

    MorphoGP: A Nonparametric Framework for Predicting Equilibrium Beach Profiles Under Tidal Influence

    [https://arxiv.org/abs/2608.18558](https://arxiv.org/abs/2608.18558)

    本文提出MorphoGP，一个结合对比学习分类和高斯过程回归的非参数框架，以自动分类并预测潮汐影响下的平衡海滩剖面，克服传统模型在非线性潮汐环境中的适应性不足。

    

    在潮汐影响下预测平衡海滩剖面对于可持续海岸发展、指导海岸保护策略以及管理在不断变化的环境条件下的海岸生态系统具有根本重要性。然而，由于波浪、潮汐和沉积过程之间的高度非线性相互作用，这仍然具有挑战性。传统的经验和数值模型在不同海岸环境中的适应性往往有限，在潮汐过程重要的海滩系统中尤其表现出明显局限性。为了在这些条件下改进数据驱动预测，本研究提出了MorphoGP，一种统一的类别特定高斯过程框架，用于在潮汐影响下预测平衡海滩剖面（EBPs）。该框架首先引入了一个基于对比学习的ContourCluster模型，自动对受潮汐影响的海滩形态进行分类。在每个形态类别内，

    arXiv:2608.18558v1 Announce Type: cross  Abstract: The prediction of equilibrium beach profiles under tidal influence is of fundamental importance for sustainable coastal development, informing shoreline protection strategies and managing coastal ecosystems under changing environmental conditions. However, it remains challenging due to the highly nonlinear interactions among wave, tide, and sedimentary processes. Traditional empirical and numerical models often exhibit limited adaptability across diverse coastal environments, with especially pronounced limitations in beach systems where tidal processes are important . To improve data-driven prediction under these conditions, this study proposes MorphoGP, a unified category-specific Gaussian process framework for predicting equilibrium beach profiles (EBPs) under tidal influence. The framework first introduces a ContourCluster model based on contrastive learning to classify tide-influenced beach morphologies automatically. Within each m
    
[^88]: 面向物联网环境的机器学习即服务（MLaaS）性能漂移检测

    Performance Drift Detection in Machine Learning as a Service (MLaaS) for IoT Environments

    [https://arxiv.org/abs/2608.18555](https://arxiv.org/abs/2608.18555)

    本文提出了一种面向物联网环境的MLaaS性能漂移检测框架，通过黑盒提取模型学习服务行为并联合捕获输入变化，解决了动态数据分布下的漂移检测难题。

    

    机器学习即服务（MLaaS）是一种强大的云范式，能够在物联网（IoT）环境中实现数据驱动的智能应用，因其成本效益高而被广泛应用于医疗保健、智能家居和工业领域。然而，物联网的动态特性经常改变数据分布，影响MLaaS的稳定性，而周期性的MLaaS更新进一步引入了性能漂移。与传统机器学习系统不同，MLaaS客户端作为黑盒用户运行，无法访问内部数据或参数，这使得漂移检测特别具有挑战性。为解决这一问题，我们提出了一种面向物联网环境的新型MLaaS性能漂移检测框架。该框架首先采用MLaaS提取模型，从输入-输出对中学习服务行为并识别影响预测的特征。在此基础上，所提出的MLaaS性能漂移检测（MPDD）模型联合捕获输入变化。

    arXiv:2608.18555v1 Announce Type: cross  Abstract: Machine Learning as a Service (MLaaS) is a powerful cloud paradigm enabling data-driven intelligent applications in Internet of Things (IoT) environments, widely adopted across healthcare, smart homes, and industry due to its cost-effectiveness. However, the dynamic nature of IoT frequently alters data distributions, affecting MLaaS stability, while periodic MLaaS updates further introduce performance drift. Unlike traditional ML systems, MLaaS clients operate as black-box users without access to internal data or parameters, making drift detection particularly challenging. To address this, we propose a novel MLaaS Performance Drift Detection framework for IoT environments. The framework first employs an MLaaS extraction model that learns service behavior from input-output pairs and identifies prediction-influenced features. Building on this, the proposed MLaaS Performance Drift Detection (MPDD) model jointly captures variations in inpu
    
[^89]: CentaurBench：基准测试LLM在增强与自动化现实工作任务上的能力

    CentaurBench: Benchmarking LLM Capabilities on Augmenting vs. Automating Real-World Work Tasks

    [https://arxiv.org/abs/2608.18554](https://arxiv.org/abs/2608.18554)

    本文提出CentaurBench框架，通过比较LLM在自动化和增强模式下的表现，发现两种模式排名关联度低，自动化胜者在增强任务中表现不佳，强调了模型选择需考虑辅助能力。

    

    arXiv:2608.18554v1 公告类型：交叉 摘要：大多数LLM基准测试根据模型自动化工作任务的能力对其进行排名。然而，在实践中，模型通常被用来辅助其他（人类或LLM）代理。因此，驱动模型选择的不仅是哪个模型产生最佳输出，而是哪个模型最能提升另一个（较弱）代理的工作。我们引入了一个统一框架，评估模型自动化和增强另一个代理性能的能力。在七个经济上基于现实的任务中，一个助手模型为标准化低能力工人模型编写辅助文本，后者产生最终交付物。在自动化模式下，助手直接产生输出。输出通过LLM评审小组使用特定任务评分标准的盲对比较进行评分，并在十次运行中重复。两种模式下的排名仅适度相关，自动化获胜者在七个任务中的五个上输掉了增强。

    arXiv:2608.18554v1 Announce Type: cross  Abstract: Most LLM benchmarks rank models on their ability to automate work tasks. In practice, however, models are often used to assist other (human or LLM) agents. The question that drives model selection is therefore not only which model produces the best output, but which model most improves the work of another (weaker) agent. We introduce a unified framework that evaluates the capability of models to automate and augment another agent's performance. Across seven economically grounded real-world tasks, an assistant model writes assistance text for a standardized lower-capacity worker model, which produces the deliverable. In automation mode, the assistant produces the output directly. Outputs are scored through blind pairwise comparisons by an LLM judge panel with task-specific rubrics, replicated across ten runs. Rankings across the two regimes are only modestly correlated, and the automation winner loses augmentation on five of seven tasks
    
[^90]: 弥合搜索与CRM：将AI产品研究代理投入生产以促进客户重新参与

    Bridging Search and CRM: Productionizing AI Product Research Agents for Customer Re-Engagement

    [https://arxiv.org/abs/2608.18543](https://arxiv.org/abs/2608.18543)

    本文提出了一个生产部署的AI产品研究代理框架，通过整合搜索与CRM系统，利用多代理研究和WhatsApp个性化推荐，显著提升了探索性购买意图用户的重新参与率和点击率。

    

    arXiv:2608.18543v1 公告类型：新 摘要：现代电子商务平台通常独立运行搜索、推荐、个性化和CRM系统，限制了主动客户重新参与的机会。这对于诸如“最佳智能手机”或“最新5G手机”等探索性意图尤其具有挑战性，因为用户可能在购买前离开平台进行外部研究。我们提出了一个可扩展、已投入生产部署的框架，通过AI驱动的产品研究代理弥合搜索与CRM工作流。该系统识别具有探索性购买意图和低参与度的用户，利用行为信号、外部知识和企业目录数据进行基于事实的多代理产品研究，并通过WhatsApp提供个性化推荐。我们在为期23天的生产部署中评估了该框架，涉及约15,000条移动产品发现通知。该活动在点击率方面取得了显著提升，优于tr

    arXiv:2608.18543v1 Announce Type: new  Abstract: Modern e-commerce platforms often operate search, recommendation, personalization, and CRM systems independently, limiting opportunities for proactive customer re-engagement. This is particularly challenging for exploratory intents such as best smartphones or latest 5G phones, where users may leave the platform for external research before purchasing. We present a scalable, production-deployed framework that bridges search and CRM workflows through AI-powered Product Research Agents. The system identifies users with exploratory purchase intent and low engagement, conducts grounded multi-agent product research using behavioral signals, external knowledge, and enterprise catalog data, and delivers personalized recommendations through WhatsApp. We evaluate the framework in a 23-day production deployment involving approximately 15K WhatsApp notifications for mobile product discovery. The campaign achieved substantial CTR improvements over tr
    
[^91]: 评估与解释基于交互的大语言模型提示敏感性

    Evaluating and Explaining Prompt Sensitivity of LLMs Using Interactions

    [https://arxiv.org/abs/2608.18539](https://arxiv.org/abs/2608.18539)

    本文提出一种基于交互的提示敏感性（IPS）方法，通过分解LLM输出为非线性交互，揭示即使输出不变时提示细微变化也能引发内部不稳定，从而更精确地解释提示敏感性的原因。

    

    大型语言模型（LLMs）的显著能力常常因其不稳定性而受损。即使提示中微妙且语义无关的变化也可能导致性能剧烈波动，这一现象被称为提示敏感性。以往研究通常通过比较提示变化时LLM的最终输出来评估提示敏感性，然而，这种粗粒度指标无法解释提示敏感性的内在原因。本文引入交互作为一种细粒度工具来分析LLM的提示敏感性。具体来说，我们将LLM的输出分数分解为一组交互，每个交互表示一组输入变量之间的非线性关系。我们发现，即使LLM的输出保持不变，提示的细微变化也能引发交互的严重不稳定。为此，我们提出了基于交互的提示敏感性（IPS）方法。

    arXiv:2608.18539v1 Announce Type: cross  Abstract: The remarkable capabilities of large language models (LLMs) are often undermined by their instability. Even subtle and semantically irrelevant changes in prompts can cause dramatic fluctuations in performance, a phenomenon known as prompt sensitivity. Previous studies typically evaluate prompt sensitivity by comparing the LLM's final outputs when prompts change. However, such coarse-grained metrics fail to explain the internal reasons for prompt sensitivity. In this paper, we introduce interactions as a fine-grained tool to analyze prompt sensitivity of LLMs. Specifically, we decompose the output score of the LLM into a set of interactions. Each interaction represents a nonlinear relationship involving a set of input variables. We discover that subtle changes to prompts can trigger severe instability in interactions, even when the outputs of the LLM remain the same. To this end, we propose an Interaction-based Prompt Sensitivity (IPS) 
    
[^92]: FinRCA-Bench：面向金融AI系统的证据检索与推理基准测试

    FinRCA-Bench: Benchmarking Evidence Retrieval and Reasoning for Financial AI Systems

    [https://arxiv.org/abs/2608.18534](https://arxiv.org/abs/2608.18534)

    FinRCA-Bench是一个合成金融对账基准，通过分离证据检索与推理评估，揭示了金融AI系统在证据获取和推理质量上的独立性能。

    

    大型语言模型越来越多地被用于支持金融操作，但其表面上的推理性能可能取决于它们是否获得了正确的证据。在金融对账中，诊断所需的证据分布在发票、采购订单、审批、分配、付款、总账分录和银行活动中，这些证据通过交易关系而非文本相似性相互关联。因此，端到端的准确性可能将证据获取与推理质量混为一谈。我们引入了FinRCA-Bench，这是一个确定性的合成基准测试，包含2,250个应付账款到银行的对账案例，涵盖14个操作表，其中包括1,500个跨15个因果类别的注入故障和750个合法或硬负样本案例。根本原因标签和记录级证据契约对模型隐藏，使得检索可以独立于答案正确性进行评估。我们比较了规则/SQL、经典机器学习方法。

    arXiv:2608.18534v1 Announce Type: new  Abstract: Large language models are increasingly used to support financial operations, but their apparent reasoning performance can depend on whether they receive the right evidence. In financial reconciliation, the evidence needed for diagnosis is distributed across invoices, purchase orders, approvals, allocations, payments, ledger entries, and bank activity, linked by transactional relationships rather than textual similarity. End-to-end accuracy can therefore conflate evidence access with reasoning quality. We introduce FinRCA-Bench, a deterministic synthetic benchmark of 2,250 accounts-payable-to-bank reconciliation cases spanning 14 operational tables, including 1,500 injected failures across 15 causal categories and 750 legitimate or hard-negative cases. Root-cause labels and record-level evidence contracts are hidden from the model, allowing retrieval to be evaluated independently of answer correctness. We compare Rules/SQL, classical mach
    
[^93]: 离线解释选择中成对排序优于单动作强化学习：一个实用经验

    Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson

    [https://arxiv.org/abs/2608.18531](https://arxiv.org/abs/2608.18531)

    本文提出将解释生成与选择分离的离线架构，并通过实证表明成对排序方法（如LambdaRank）在离线解释选择任务中优于单动作强化学习方法（如PPO、GRPO），在无需GPU且延迟低于100毫秒的条件下实现高效推荐系统。

    

    arXiv:2608.18531v1 公告类型：新 摘要：基于大型语言模型（LLM）构建的工业可解释推荐系统面临巨大的服务成本：每个请求都会触发一次LLM生成，延迟达数百毫秒，且成本随流量线性增长。我们将生成与选择分离：解释预先生成作为冻结候选池（六种提示风格、两种商用LLM），并由一个驻留CPU的小型选择器在请求时挑选一个。该架构无需GPU，响应时间低于100毫秒。我们的主要基准是XRec Google Local子集（含2,958对），评估了六种离线池选择器（LambdaRank、PPO、GRPO、DPO、教师-学生蒸馏）和三种知识图谱路径选择器（随机游走、边不相交枚举、MMR重排序路径）。由于该设置没有公开基准，我们使用包含300对的MovieLens-1M子集（以Claude-Sonnet-4.5参考）作为内部跨数据集检查。所有变体均使用与XRec相同的BERTScore-F1协议。

    arXiv:2608.18531v1 Announce Type: new  Abstract: Industrial explainable-recommendation systems built on LLMs incur a substantial serving cost: each request triggers an LLM generation, with latency in the hundreds of milliseconds and cost that scales linearly with traffic. We separate generation from selection: explanations are produced ahead of time as a frozen candidate pool (six prompt styles, two commodity LLMs), and a small CPU-resident selector picks one at request time. The stack needs no GPU and returns in under 100 ms.   Our primary benchmark is a 2,958-pair XRec Google Local subset, evaluating six offline-pool selectors (LambdaRank, PPO, GRPO, DPO, teacher-student distillation) and three KG-path selectors (random walks, edge-disjoint enumeration, MMR-reranked paths). A 300-pair MovieLens-1M split with Claude-Sonnet-4.5 references serves as an internal cross-dataset check, since no public benchmark exists for this setting. All variants use the same BERTScore-F1 protocol as XRec
    
[^94]: DART-SD：面向多轮工具调用智能体自蒸馏的菱形拓扑感知检索与调优

    DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents

    [https://arxiv.org/abs/2608.18524](https://arxiv.org/abs/2608.18524)

    DART-SD通过建模交互状态转移图来感知任务中的菱形拓扑结构，并用拓扑引导的局部修正替代全局轨迹强制，从而避免拓扑坍缩并提升多轮工具调用智能体的策略多样性。

    

    摘要：为大型语言模型（LLMs）配备多轮工具调用能力对于构建自主智能体至关重要。然而，这一进展从根本上受限于对全长度轨迹模仿的依赖。对于涉及多个顺序无关子目标的任务，最优解空间形成了一个庞大的组合菱形晶格结构。将这种丰富的拓扑结构强制压缩为单一轨迹会导致严重的拓扑坍缩，不区分地惩罚有效的替代探索路径，从而严重削弱策略多样性。为解决这一问题，我们提出了DART-SD（菱形拓扑感知检索与自蒸馏调优），这是一种新颖的框架，将范式从全局强制转变为拓扑引导的局部修正。DART-SD首先将执行过程建模为一个收敛的交互状态转移图（ISTG），忠实捕获成功和失败探索路径中固有的菱形拓扑结构。

    arXiv:2608.18524v1 Announce Type: cross  Abstract: Equipping Large Language Models (LLMs) with multi-turn tool-calling capabilities is essential for building autonomous agents. However, progress is fundamentally limited by the reliance on full-length trajectory imitation. For tasks involving multiple order-independent sub-goals, the optimal solution space forms a vast combinatorial diamond lattice. Forcing this rich topology into monolithic trajectories causes a severe topological collapse, indiscriminately penalizing valid alternative explorations and severely degrading policy diversity. To address this, we propose DART-SD (Diamond-topology Aware Retrieval and Tuning for Self-Distillation), a novel framework that shifts the paradigm from global forcing to topology-guided localized correction. DART-SD first models the execution process as a converging Interaction-State Transition Graph (ISTG), faithfully capturing the inherent diamond topology of successful and failed exploratory paths
    
[^95]: 基于先验条件的高斯判别器用于可泛化的AI生成图像检测

    Prior-Conditioned Gaussian Discriminants for Generalizable AI-generated Image Detection

    [https://arxiv.org/abs/2608.18523](https://arxiv.org/abs/2608.18523)

    本文提出了一种基于先验条件的高斯判别阶梯方法，利用闭式统计特征头部，在多种数据集和迁移场景下实现了与现有AI图像检测器相当甚至更优的性能，并揭示了训练先验对检测泛化能力的关键影响。

    

    arXiv:2608.18523v1 公告类型：交叉 摘要：基于扩散模型的生成器使得合成图像无处不在，但检测器在生成器、提示/风格和源域同时发生变化时往往失效。我们将AI生成图像检测视为一个由训练先验、冻结编码器特征空间和决策规则描述的迁移系统，并探讨分类器头部训练在何种情况下能超越现代特征中已可分离的部分。作为受控诊断，我们拟合了一个先验条件的高斯判别阶梯：基于嵌套协方差假设下的一阶和二阶特征统计构建的闭式头部。在Percept-Lens（一个涵盖39个公共数据集、710万张图像的统一协议）上，最佳层级在匹配先验和编码器时，通常与已发布的AI生成图像检测器头部相当，有时甚至超过它们。我们进一步量化了训练先验的强敏感性、基于矩的头部在数据效率上的优势，以及表示...

    arXiv:2608.18523v1 Announce Type: cross  Abstract: Diffusion-based generators have made synthetic images ubiquitous, but detectors often fail under simultaneous shifts in generator, prompt/style, and source-domain. We study AI-generated image detection as a transfer system described by training prior, frozen encoder feature space, and decision rule, and ask when classifier head training adds value beyond what is already separable in modern features. As a controlled diagnostic, we fit a prior-conditioned Gaussian discriminant ladder: closed-form heads built from first- and second-order feature statistics under nested covariance assumptions. On Percept-Lens, a unified protocol over 39 public datasets (7.1 million images), the best rung is frequently competitive with, and sometimes exceeds, released AI-generated image detector heads when matched on both prior and encoder. We further quantify strong sensitivity to the training prior, data-efficiency of moment-based heads, and representatio
    
[^96]: GCNO：基于物理的无线信道压缩的格拉姆-切比雪夫神经算子

    GCNO: Gramian Chebyshev Neural Operator for Physics-Based Compression of Wireless Channels

    [https://arxiv.org/abs/2608.18522](https://arxiv.org/abs/2608.18522)

    本文提出GCNO，一种基于物理的可变速率无线信道压缩方法，通过识别少数主导传播路径实现高效反馈，无需重新训练即可适应不同天线配置。

    

    arXiv:2608.18522v1 公告类型：交叉 摘要：大型天线阵列使无线系统能够服务更多用户并实现更高数据速率，但也使信道反馈成本高昂：接收设备必须反复向基站报告一个大型复值信道矩阵。大多数神经压缩器将该矩阵视为图像，并用固定长度的代码替换它，只有匹配的神经解码器才能解释该代码。因此，消息不能适应信道复杂性，改变天线数量通常需要重新训练。我们询问设备是否可以仅报告每个信道下少数主导传播路径。我们引入了格拉姆-切比雪夫神经算子（GCNO），一种基于物理的可变速率压缩器，它识别依赖于样本的路径方向集。GCNO利用接收-发射信道结构定位路径，使用一阶泰勒校正来细化网格点之间的方向，并使用最小二乘法来重建信道。

    arXiv:2608.18522v1 Announce Type: cross  Abstract: Large antenna arrays allow wireless systems to serve more users and achieve higher data rates, but they also make channel feedback expensive: the receiving device must repeatedly report a large complex-valued channel matrix to the base station. Most neural compressors treat this matrix like an image and replace it with a fixed-length code that only a matched neural decoder can interpret. The message therefore does not adapt to channel complexity, and changing the antenna count typically requires retraining. We ask whether a device can instead report only the few dominant propagation paths underlying each channel. We introduce the Gramian Chebyshev Neural Operator (GCNO), a physics-based, variable-rate compressor that identifies a sample-dependent set of path directions. GCNO uses receive-transmit channel structure to locate paths, a first-order Taylor correction to refine directions that fall between grid points, and least squares to r
    
[^97]: 哪些负样本重要？问问你的文本编码器：面向密集字幕检索的自适应相似度边界

    Which Negatives Matter? Ask Your Text Encoder: Adaptive Similarity Margins for Dense-Caption Retrieval

    [https://arxiv.org/abs/2608.18521](https://arxiv.org/abs/2608.18521)

    本文提出HN-CLIP方法，通过利用文本编码器的文本-文本几何结构构建自适应相似度边界，解决了密集字幕检索中InfoNCE损失过早饱和和负样本未充分区分的问题。

    

    arXiv:2608.18521v1 公告类型：新 摘要：密集字幕检索最近通过引入分割、边缘图、LLM过滤字幕和跨模态模块到对比微调中而得到改进。然而，这些方法在很大程度上继承了相同的InfoNCE目标，其优化在强大的预训练初始化下可能过早饱和：在密集字幕上，损失在第一个周期内80%的批次中降至10^{-3}以下，而其梯度在fp32中47%的测量中精确达到零。我们发现，这种行为与密集字幕基准中大量近似重复字幕密切相关，在容易的多数样本已被分离后，少数高度相似的负样本仍未解决。作为补救措施，我们引入了HN-CLIP，它利用文本编码器自身的文本-文本几何结构来构建每个负样本的自适应相似度边界。具体来说，一个分离的字幕相似度矩阵被添加到负对数几率中，赋予大...

    arXiv:2608.18521v1 Announce Type: new  Abstract: Dense-caption retrieval has recently been improved by introducing segmentation, edge maps, LLM-filtered captions, and cross-modal modules into contrastive fine-tuning. However, these methods largely inherit the same InfoNCE objective, whose optimization can prematurely saturate under a strong pre-trained initialization: on dense captions, the loss falls below 10^{-3} on 80% of batches within the first epoch, while its gradient reaches exact zero in fp32 in 47% of measurements. We find that this behavior is closely related to the large number of near-duplicate captions in dense-caption benchmarks, where a few highly similar negatives remain unresolved after the easy majority has already been separated. As a remedy, we introduce HN-CLIP, which uses the text encoder's own text-text geometry to construct per-negative adaptive similarity margins. Specifically, a detached caption-similarity matrix is added to the negative logits, assigning lar
    
[^98]: OptiModNet：一种结合分组查询与通道注意力的UNet-Transformer混合模型用于视盘和视杯分割

    OptiModNet: A UNet-Transformer Hybrid with Grouped-Query and Channel Attention for Optic Disc and Cup Segmentation

    [https://arxiv.org/abs/2608.18516](https://arxiv.org/abs/2608.18516)

    本文提出OptiModNet，一种融合分组查询与通道注意力的UNet-Transformer混合模型，旨在以低计算成本实现视盘和视杯的精确分割，从而支持青光眼的快速筛查和资源受限环境下的部署。

    

    摘要：视盘和视杯的精确分割对于青光眼的早期检测和诊断至关重要。然而，在保持低计算需求的同时，跨数据集实现持续高性能仍是一个重大挑战。在青光眼检测中，低计算方法对于实现快速大规模筛查以及促进在资源有限的临床环境中的部署至关重要。尽管深度学习模型如UNet、视觉变换器（ViTs）和扩散模型已展现出强大的分割性能，但这些方法通常伴随着巨大的计算开销。UNet在捕捉局部特征方面高效，但在建模全局上下文信息方面受限。相反，ViTs擅长长距离依赖建模，但计算密集。混合架构，如UNetR，结合了基于变换器的编码器和UNet风格解码器，已显示出潜力。

    arXiv:2608.18516v1 Announce Type: cross  Abstract: Precise segmentation of the optic disc and cup is critical for the early detection and diagnosis of glaucoma. However, achieving consistently high performance across datasets while maintaining low computational requirements remains a significant challenge. In glaucoma detection, low-computation methods are crucial for enabling rapid, large-scale screening and facilitating deployment in resource-limited clinical environments. While deep learning models such as UNets, Vision Transformers (ViTs), and Diffusion models have demonstrated strong segmentation performance but these methods often come with substantial computational overhead. UNets are efficient at capturing local features but are limited in modeling global contextual information. Conversely, ViTs excel at long-range dependency modeling but are computationally intensive. Hybrid architectures, such as UNetR, which combine transformer-based encoders with UNet-style decoders, have s
    
[^99]: 机器上由机器完成的科学：计算化学中的AI智能体

    Science Done on a Machine by a Machine: AI Agents in Computational Chemistry

    [https://arxiv.org/abs/2608.18508](https://arxiv.org/abs/2608.18508)

    本文综述了计算化学中AI智能体系统的快速增长，指出其从辅助任务向自主实验和写作的演进，并预测最终将实现无需人类监督的完全自主AI科学家，同时通用智能体可能取代专门系统。

    

    arXiv:2608.18508v1 公告类型：交叉 摘要：我们正目睹计算化学模拟智能体系统的爆炸式增长：从2024年的约六个到2025年的十二个，而截至2026年8月8日的本次综述中，当前数量已接近五十个。这些智能体系统的能力正从协助执行一系列计算任务，转向自主设计和执行《in silico》实验、分析结果，甚至撰写手稿。最终目标是实现完全自主的AI科学家，即整个计算化学过程在机器上由机器完成，无需人类监督。尽管我们尚未达到这一阶段，且所有已报道系统目前都包含人类参与，但这一趋势是明确的。即使为计算化学构建专门的智能体系统，也日益被通用智能体所商品化，这最终可能取代对专门系统的需求。

    arXiv:2608.18508v1 Announce Type: cross  Abstract: We are witnessing an explosion of agentic systems for computational chemistry simulations: from half a dozen in 2024 to a dozen in 2025, and the current number approaches fifty, surveyed in this Perspective as of 8 August 2026. The capabilities of these agentic systems are shifting from assisting in performing a selection of computational tasks to autonomous design and execution of \textit{in silico} experiments, their analysis, and even manuscript writing. The ultimate destination is a fully autonomous AI scientist, where the entirety of computational chemistry is performed on a machine by a machine, without human supervision. While we are not there yet, and all reported systems currently involve a human in the loop, the trend is unmistakable. Even building specialized agentic systems for computational chemistry is increasingly commoditized by generalist agents, which may in the end replace the need for the specialized ones altogether
    
[^100]: UMER：通过成对感知判别推理统一嵌入与排序，实现通用多模态检索

    UMER: Unifying Embedding and Ranking via Pair-Aware Discriminative Reasoning for Universal Multimodal Retrieval

    [https://arxiv.org/abs/2608.18504](https://arxiv.org/abs/2608.18504)

    本文提出UMER框架，通过成对感知的判别推理统一嵌入与排序，解决现有CoT方法在区分难负例和元任务推理上的不足，提升通用多模态检索性能。

    

    通用多模态检索旨在支持多样化的指令感知检索任务，既需要高效的语料库规模匹配，又需要细粒度的语义推理。近期基于多模态大语言模型（MLLM）的嵌入方法通常从隐藏状态中提取表示，而链式思维（CoT）推理正成为一种有前景的嵌入增强策略，通过将中间语义证据编码到表示空间中来提升效果。然而，现有的CoT方法通常对查询和候选进行孤立的逐项推理，无法提供明确的证据来区分正例与语义上易混淆的难负例。此外，对比嵌入捕捉全局相似性，但在需要答案验证、类别判断或细粒度推理的元任务上表现不佳。本文提出了UMER，一个用于通用多模态检索的统一多模态嵌入与排序框架。UMER取代了逐项推理，采用成对感知的判别推理，以增强嵌入和排序能力。

    arXiv:2608.18504v1 Announce Type: new  Abstract: Universal multimodal retrieval aims to support diverse instruction-aware retrieval tasks, demanding both efficient corpus-scale matching and fine-grained semantic reasoning. Recent MLLM-based embedding methods typically derive representations from hidden states, while Chain-of-Thought (CoT) reasoning is emerging as a promising strategy for embedding enhancement by encoding intermediate semantic evidence into the representation space. However, existing CoT methods typically use item-wise reasoning over queries and candidates in isolation, providing no explicit evidence to distinguish a positive from a semantically confusable hard negative. Moreover, contrastive embeddings capture global similarity but struggle with meta-tasks requiring answer verification, category judgment or fine-grained reasoning. In this paper, we propose UMER, a Unified Multimodal Embedding and Ranking framework for universal multimodal retrieval. UMER replaces item-
    
[^101]: 物理展开神经算子用于无线场建模

    Physics-Unrolled Neural Operator for Wireless Field Modeling

    [https://arxiv.org/abs/2608.18495](https://arxiv.org/abs/2608.18495)

    本文提出PU-HNO，一种物理展开的三阶段神经算子，通过逐步建模反射、衍射和散射效应，从低保真射线追踪输出高效预测高保真室内无线电地图，并克服训练标签噪声问题。

    

    无线电地图对于无线决策任务（如接入点放置、覆盖规划和定位）至关重要，但其精细的空间细节受复杂传播效应支配，且精确模拟成本高昂。机器学习提供了一条无需对每个场景运行昂贵高保真模拟即可实现高保真无线电地图预测的路径。然而，大规模生成高质量训练标签也很困难：可负担的标签来自有限射线模拟，这些模拟比低保真输入更丰富，但携带残余蒙特卡洛噪声。我们通过物理展开混合神经算子（PU-HNO）应对这一挑战，这是一种三阶段级联结构，从低保真射线追踪输出和场景先验中预测高保真室内无线电地图，通过逐步捕捉反射、衍射和散射效应，而非将无线电地图视为通用图像。我们证明，在一致性条件下……

    arXiv:2608.18495v1 Announce Type: cross  Abstract: Radio maps are essential for wireless decision-making tasks such as access-point placement, coverage planning, and localization, but their fine spatial details are governed by complex propagation effects and are costly to simulate accurately. Machine learning offers a path to high-fidelity radio-map prediction without running expensive high-fidelity simulations for every scene. However, generating high-quality training labels at scale is also difficult: the affordable labels come from finite-ray simulations, which are richer than low-fidelity inputs but carry residual Monte Carlo noise. We address this challenge with Physics-Unrolled Hybrid Neural Operator (PU-HNO), a three-stage cascade that predicts high-fidelity indoor radio maps from low-fidelity ray-tracing outputs and scene priors by progressively capturing reflection, diffraction, and scattering effects, rather than treating radio maps as generic images. We prove that, under con
    
[^102]: 分割支持域，重建残差：用于视频生成与世界模型的无训练稀疏注意力机制

    Partition the Support, Reconstruct the Residual: Training-Free Sparse Attention for Video Generation and World Models

    [https://arxiv.org/abs/2608.18484](https://arxiv.org/abs/2608.18484)

    该论文提出SparsePR，一种无训练稀疏注意力方法，通过响应耦合分区和探针拟合残差重建，在视频生成与世界模型中显著提升效率并保持性能。

    

    无训练的分块稀疏注意力可以加速视频变换器，但按行注意力集中本身并不指定一个可执行的稀疏算子。共享一个块路由的查询可能具有重叠度较低的支持域，而保留的注意力质量本身并不能决定由跳过交互引起的softmax后误差。我们表明，分区几何结构同时影响池化支持域以及从稀疏输出中预测剩余残差的可预测性。我们引入了SparsePR，它结合了响应耦合分区与探针拟合残差重建。采样查询的键响应形成配对的K/V组，其质心诱导查询-响应坐标用于共享路由。少量精确查询行随后在校准探针残差中观察到的输出子空间内，从稀疏输出中校准一个调用特定的仿射修正。在四个异构视频生成与世界模型中，SparsePR一致地...

    arXiv:2608.18484v1 Announce Type: cross  Abstract: Training-free block-sparse attention can accelerate video transformers, but row-wise attention concentration does not by itself specify an executable sparse operator. Queries sharing a block route may have poorly overlapping supports, while retained attention mass alone does not determine the post-softmax error from skipped interactions. We show that partition geometry affects both pooled support and the predictability of the remaining residual from the sparse output. We introduce SparsePR, which combines Response-Coupled Partitioning with Probe-Fitted Residual Reconstruction. Sampled-query key responses form paired K/V groups, whose centroids induce query-response coordinates for shared routing. A small set of exact query rows then calibrates a call-specific affine correction from the sparse output within the output subspace observed in the probe residuals. Across four heterogeneous video generation and world models, SparsePR consiste
    
[^103]: 基于覆盖率驱动的RTL断言生成：形式化探索与神经符号优化

    Coverage-Driven RTL Assertion Generation with Formal Exploration and Neuro-Symbolic Refinement

    [https://arxiv.org/abs/2608.18482](https://arxiv.org/abs/2608.18482)

    本文提出NeuroAssertion框架，通过形式化轨迹生成、语法引导合成和智能体优化循环，实现了覆盖率驱动的RTL断言生成，有效覆盖难以到达的设计行为。

    

    硬件功能验证依赖于高质量的断言来暴露设计缺陷并建立对寄存器传输级（RTL）设计的信心。然而，现有的断言挖掘方法仍难以生成完整且可靠的断言集：随机或有限的轨迹无法覆盖难以到达的行为，且一次性生成几乎不提供关于哪些内容未被验证或应如何改进断言集的反馈。因此，即使生成了大量断言，关键设计行为仍可能未被覆盖。我们提出了NeuroAssertion，一个覆盖率驱动的断言生成框架，该框架在一个统一框架内结合了形式化轨迹生成、语法引导合成（SyGuS）和基于智能体的优化过程。我们的框架首先将难以到达的控制流条件转换为形式化可达性目标，使用模型检查生成行为多样化的轨迹，并挖掘初始断言。

    arXiv:2608.18482v1 Announce Type: cross  Abstract: Hardware functional verification relies on high-quality assertions to expose design bugs and establish confidence in Register Transfer Level (RTL) designs. Yet existing assertion mining methods still struggle to produce complete and reliable assertion sets: random or limited traces fail to cover hard-to-reach behaviors, and one-shot generation provides little feedback about what remains unverified or how the assertion set should be improved. As a result, critical design behaviors can remain uncovered even when many assertions are generated. We present NeuroAssertion, a coverage-driven assertion generation framework that combines formal trace generation, syntax-guided synthesis (SyGuS), and an agent-inspired refinement process within a unified framework. Our framework first converts hard-to-reach control-flow conditions into formal reachability objectives, uses model checking to generate behaviorally diverse traces, and mines initial as
    
[^104]: ERASE：早期反向传播调度以加速现代推荐系统的训练

    ERASE: EaRly bAckpropagation SchEdule for Faster Training of Modern Recommendation Systems

    [https://arxiv.org/abs/2608.18469](https://arxiv.org/abs/2608.18469)

    ERASE通过将前向和反向传播重叠调度，利用Forward-Forward的分离机制在CUDA流上提前执行反向传播，显著提升现代推荐系统训练的硬件利用率。

    

    摘要：arXiv:2608.18469v1 公告类型：交叉 摘要：轻量级代理模型能够在不重复训练前沿规模系统的情况下进行快速实验，但其小型内核常常使现代加速器未被充分利用。传统训练通过将前向和后向传播安排为不重叠的阶段来加剧这种低效，因此一个阶段的空闲容量无法被另一个阶段的工作填充。我们重新解释了Forward-Forward（FF）的分离机制作为调度原语：给定一个局部目标，分离一个块的输出会移除下游梯度依赖，使其反向传播在前向传播完成后即可就绪。ERASE在独立的CUDA流上提前启动每个分离子图的反向传播，并将其与后续的前向工作重叠。轻量级transformer的执行轨迹展示了这种重叠及其限制：一个使设备饱和的内核没有留下并发容量。在大规模点击率模型中，分离...

    arXiv:2608.18469v1 Announce Type: cross  Abstract: Lightweight proxy models enable rapid experimentation without repeatedly training frontier-scale systems, but their small kernels often leave modern accelerators underutilized. Conventional training compounds this inefficiency by scheduling the forward and backward passes as disjoint phases, so spare capacity in one cannot be filled by work from the other. We reinterpret the detachment mechanism of Forward-Forward (FF) as a scheduling primitive: given a local objective, detaching a block's output removes downstream gradient dependencies, making its backward pass ready when its forward pass finishes. ERASE launches each detached subgraph's backward pass early on a separate CUDA stream, overlapping it with subsequent forward work. Execution trace on a lightweight transformer demonstrates this overlap and its limit: a kernel that saturates the device leaves no capacity for concurrency. On a large-scale click-through-rate model, detaching 
    
[^105]: 罗曼诺夫三元组逻辑的形式化验证：一种用于滑动窗口3-CNF的验证过滤器及其在结构化公式中的应用

    Formal Verification of Romanov's Triplet Logic: A Verified Filter for Sliding-window 3-CNF with Application to Structured Formulas

    [https://arxiv.org/abs/2608.18445](https://arxiv.org/abs/2608.18445)

    首次在Rocq中形式化验证了罗曼诺夫三元组逻辑（TLS）的核心，并针对滑动窗口片段证明了其过滤器的多项式时间界限和精确正确性边界。

    

    arXiv:2608.18445v1 公告类型：交叉 摘要：我们在Rocq证明助手中首次实现了罗曼诺夫三元组逻辑（TLS）的机械化形式化。TLS是一种基于三元组的组合框架，用于推理通过分层三元组结构（称为紧凑三元组结构，CTS）的兼容路径，以及它们通过罗曼诺夫有效过程（我们称之为简单顶点交集，SVI）的交集。TLS最初源于布尔可满足性动机，构成一个自包含的数学理论，其形式化性质此前尚未建立。我们在Rocq中形式化了TLS的核心，包括紧凑三元组公式（CTF）、CTS、超结构、清除和SVI。对于良好形式的滑动窗口片段，我们验证了逐子句的CNF到CTF转换、清除过程和对齐交集，并证明了过滤器阶段的显式多项式时间界限。我们的主要贡献是一个精确的正确性边界：存在...

    arXiv:2608.18445v1 Announce Type: cross  Abstract: We present the first mechanised formalisation of Romanov's Triplet Logic (TLS) in the Rocq proof assistant. TLS is a triplet-based combinatorial framework for reasoning about compatible paths through layered triplet structures, called Compact Triplets Structures (CTS), and their intersection via Romanov's Effective Procedure, which we refer to as Simple Vertex Intersection (SVI). Originally motivated by Boolean satisfiability, TLS constitutes a self-contained mathematical theory whose formal properties had not been previously established. We formalise the core of TLS in Rocq, including Compact Triplets Formulas (CTF), CTS, hyperstructures, clearing, and SVI. For the well-formed sliding-window fragment we verify a clause-by-clause CNF-to-CTF translation, the clearing procedure, and aligned intersection, and we prove explicit polynomial-time bounds for the filter stages. Our main contribution is a precise correctness boundary: the existe
    
[^106]: 心理健康领域的教学人工智能：一种用于自动化临床监督和风险分诊的三流微调大型语言模型框架

    Pedagogical AI in Mental Health: A Tri-Stream Fine-Tuned LLM Framework for Automated Clinical Supervision and Risk Triage

    [https://arxiv.org/abs/2608.18438](https://arxiv.org/abs/2608.18438)

    该论文提出一种基于微调大型语言模型的三流框架，用于心理健康领域的自动化临床监督和风险分诊，通过整合治疗联盟跟踪、风险预测和动态紧迫性指数，有效弥补了新手治疗师的监督缺口。

    

    现代心理健康护理面临资深监督监督的严重短缺，导致“监督缺口”，即新手治疗师在高风险情况下处理问题，但获得专业反馈却存在延迟。本文提出一个新框架，利用微调的Mistral-7B-instruct模型作为自动化的“监督者在场”系统。通过利用DAIC-WOZ数据集中的106个会话，该模型执行三流分析：（1）通过语义一致性跟踪治疗联盟，（2）使用注意力加权分析进行潜在风险预测，以及（3）通过动态临床紧迫性指数（D-CUI）进行监督分诊。我们的多模态VAL（视觉-声学-语言）框架实现了95%的技术识别准确率[95%置信区间：75.1%-99.9%]，联盟评估的均方误差为0.105（基于5点量表）[95%置信区间：0.059-0.151]，治疗保真度alpha=0.423，平均D-CUI为0.370[95%置信区间：0.322-0.419]。训练在105步内收敛，共8步。

    arXiv:2608.18438v1 Announce Type: cross  Abstract: Modern mental healthcare faces a critical shortage of senior supervisory oversight, leading to a "supervision gap" where novice therapists manage high-stakes risks with delayed professional feedback. This paper proposes a new framework utilizing a fine-tuned Mistral-7B-instruct model as an automated "Supervisor-in-the-Loop" system. By leveraging 106 sessions from the DAIC-WOZ dataset, the model performs a tri-stream analysis: (1) Therapeutic Alliance tracking via semantic adherence, (2) Latent risk prediction using attention-weighted analytics, and (3) Supervisory Triage via a Dynamic Clinical Urgency Index (D-CUI). Our multi-modal VAL (Visual-Acoustic-Linguistic) framework achieves 95% technique identification accuracy [95% CI: 75.1%-99.9%], alliance assessment MAE of 0.105 on a 5-point scale [95% CI: 0.059-0.151], therapeutic fidelity alpha = 0.423, and mean D-CUI of 0.370 [95% CI: 0.322-0.419]. Training converged in 105 steps with 8
    
[^107]: FM-Bench：一个面向竞争代理的长周期管理基准

    FM-Bench: A Benchmark for Long-Horizon Management with Competing Agents

    [https://arxiv.org/abs/2608.18423](https://arxiv.org/abs/2608.18423)

    该论文提出了FM-Bench，一个首个在20年长周期、多代理竞争环境中评估LLM决策能力的足球管理基准，通过确定性引擎和共享世界竞技场实现无偏的长期规划能力测量。

    

    arXiv:2608.18423v1 公告类型：新 摘要：语言模型代理现在能够可靠地执行有界任务。但它们能否在长周期内维持有效的决策——在这种场景下，行动具有累积后果，且环境会对其选择做出反应——在很大程度上仍未得到衡量。FM-Bench（足球管理基准）正是为了衡量这一点而设计。一个LLM代理通过26种工具和大约340到400个决策节点，在20个游戏年内经营一家足球俱乐部。它需要在与所有竞争对手相同的预算下组建阵容、交易球员、谈判合同、投资设施和青训、设定首发阵容，并向可能解雇它的董事会负责，同时一个确定性引擎将每一年的结果累积为一个最终得分，不涉及LLM裁判或人类评分者。单人赛道让15个前沿模型在一个固定的脚本化世界中运行，而竞技场赛道则将同样的模型加上一个脚本化锚点置于一个共享的20年世界中；据我们所知，这是该规模下首次面对面的评估。我们衡量六个方面的表现。

    arXiv:2608.18423v1 Announce Type: new  Abstract: Language model agents now execute bounded tasks reliably. Whether they can sustain effective decision-making over long horizons, where actions have cumulative consequences and the environment responds to their choices, remains largely unmeasured. FM-Bench (Football Management Benchmark) measures this. An LLM agent runs a football club for 20 in-game years through 26 tools and roughly 340 to 400 decision stops. It drafts a squad on the same budget as every rival, trades players, negotiates contracts, invests in facilities and youth, sets lineups, and answers to a board that can fire it, while a deterministic engine accumulates every year into one final score with no LLM judge or human rater. The solo track plays each of 15 frontier models against a frozen scripted world, and the Arena places the same models plus a scripted anchor in one shared 20-year world; to our knowledge, the first head-to-head evaluation at this scale. We measure six
    
[^108]: LLaMA 3.1 8B中结构感知数值推理的机制可解释性研究

    Mechanistic Interpretability of Structure-Aware Numerical Reasoning in LLaMA 3.1 8B

    [https://arxiv.org/abs/2608.18419](https://arxiv.org/abs/2608.18419)

    本文通过机制可解释性方法，设计了一个必须依赖结构线索才能解决的序列建模任务，揭示了LLaMA 3.1 8B在数值推理中是否真正理解底层结构（如一阶差分），并逆向工程了其内部算法机制。

    

    摘要：最近的研究表明，大型语言模型（LLMs）展现出强大的数值序列建模能力，并在时间序列预测方面显示出潜力。尽管LLMs具备上下文学习能力，但它们完成时间序列预测的机制仍不清楚。具体而言，它们是否真正理解底层结构，这至少需要对数字序列中的一阶差分进行推理。为了研究这一点，我们从机制可解释性的角度考察了Llama 3.1-8B。机制可解释性是一个新兴领域，专注于逆向工程神经网络（如LLMs）学习到的算法。为了评估Llama的数值序列建模能力并促进我们的机制可解释性分析，我们创建了一个序列建模任务，该任务在不捕捉结构线索的情况下无法解决。具体来说，我们采样n个随机数并...

    arXiv:2608.18419v1 Announce Type: cross  Abstract: Recent work has shown that large language models (LLMs) exhibit strong numerical sequence modeling capabilities and show promise in time-series prediction. While LLMs display in-context learning capabilities, the mechanisms with which they accomplish time-series prediction remain unclear. Specifically, whether they truly understand the underlying structure, which at a minimum requires reasoning over first differences in the sequence of numbers. To study this, we investigate Llama 3.1-8B from a mechanistic interpretability point of view. Mechanistic interpretability is an emerging field concerned with the reverse engineering of the algorithms learned by neural networks such as LLMs. To assess Llamas' numerical sequence modeling capabilities and to facilitate our mechanistic interpretability analysis, we create a sequence modeling task that cannot be solved without picking up structural cues. Specifically, we sample n random numbers and 
    
[^109]: 通过形式抽象提升资源受限语言模型中自然语言组合优化的准确性

    Improving Natural-Language Combinatorial-Optimization Accuracy in Resource-Constrained Language Models via Formal Abstractions

    [https://arxiv.org/abs/2608.18409](https://arxiv.org/abs/2608.18409)

    SDDL通过神经符号框架将自然语言调度问题转化为求解器友好的表示，显著提升了资源受限语言模型在组合优化中的可行性。

    

    arXiv:2608.18409v1 公告类型：新 摘要：组合调度对语言模型构成重大挑战，要求它们在满足复杂约束的同时，在指数级大的搜索空间内识别可行解决方案。这一挑战在资源受限环境中尤为突出，因为在此类环境中，大型语言模型不切实际，只能选择较小的模型，而这些模型在直接从自然语言进行调度时往往无法保持可行性。为解决这些限制，我们引入了SDDL，一个神经符号框架，它将自然语言调度问题转化为紧凑、面向求解器的任务、资源、约束和目标表示，同时将底层建模和搜索委托给确定性编译器和外部求解器。在包含300个实例、多族类的调度问题子集上，SDDL为每个测试的资源受限模型提高了独立验证的可行性。最强的两个SDDL模型...

    arXiv:2608.18409v1 Announce Type: new  Abstract: Combinatorial scheduling poses a significant challenge for language models, requiring them to identify feasible solutions within exponentially large search spaces while satisfying complex constraints. This challenge is especially pronounced in resource-constrained settings, where larger language models are impractical and selection is limited to smaller models which often fail to preserve feasibility when scheduling directly from natural language. To address these limitations, we introduce SDDL, a neuro-symbolic framework that translates natural-language scheduling problems into compact, solver-aligned representations of tasks, resources, constraints, and objectives, while delegating low-level modeling and search to a deterministic compiler and external solver. On a 300-instance, multi-family subset of scheduling problems, SDDL improves independently verified feasibility for every resource-constrained model tested. The two strongest SDDL
    
[^110]: 向量符号策略梯度

    Vector Symbolic Policy Gradient

    [https://arxiv.org/abs/2608.18404](https://arxiv.org/abs/2608.18404)

    该论文提出向量符号策略梯度（VSPG），通过超向量表示动作和优势加权捆绑更新，实现固定大小压缩记忆的样本高效强化学习，并证明其抗噪声稳定性。

    

    arXiv:2608.18404v1 公告类型：交叉 摘要：我们通过向量符号策略梯度（VSPG）回答了这个问题，这是一种离散动作的actor，它将每个动作表示为单位范数的超向量，并通过与编码状态的相似度对其进行评分。在标准的softmax策略梯度代理下，我们证明了其更新恰好是优势加权超向量捆绑后进行归一化，因此支持标准的优势估计器。我们进一步表明，每个训练过的动作超向量是一个固定大小的压缩核记忆，存储了在访问状态上的优势加权核展开，并根据编码器引起的相似度转移证据。这提供了一种具体的机制，可以在不增加推理时内存的情况下支持样本高效学习。最后，对于双极动作记忆，我们证明了贪婪动作选择在随机位翻转下是稳定的，失败概率随超向量维度呈指数衰减。

    arXiv:2608.18404v1 Announce Type: cross  Abstract: We answer this question with Vector-Symbolic Policy Gradient (VSPG), a discrete-action actor that represents each action by a unit-norm hypervector and scores it by similarity to the encoded state. Under the standard softmax policy-gradient surrogate, we prove that its update is exactly advantage-weighted hypervector bundling followed by normalization, and therefore supports standard advantage estimators. We further show that each trained action hypervector is a fixed-size compressed kernel memory, storing an advantage-weighted kernel expansion over visited states and transferring evidence according to the encoder-induced similarity. This provides a concrete mechanism that can support sample-efficient learning without increasing inference-time memory. Finally, for bipolar action memories, we prove that greedy action selection is stable under random bit flips, with failure probability decaying exponentially in the hypervector dimension.
    
[^111]: LEDGER：用于审计LLM代理的声明到证据追踪图

    LEDGER: Claim-to-Evidence Trace Graphs for Auditing LLM Agents

    [https://arxiv.org/abs/2608.18398](https://arxiv.org/abs/2608.18398)

    LEDGER通过构建分层追踪图，将LLM代理的声明与支持证据关联，显著简化了审计过程，提高了可信度验证效率。

    

    大型语言模型（LLM）代理现在能够执行涉及复杂工具使用、代码执行、文件编辑和生成工件等的长期技术工作流程。随着代理工作速度加快，生产力瓶颈从产出转移到审计这些产出是否正确和可信。代理可观测性系统使细粒度执行事件可见，但仅凭可见性仍让审查者难以重建哪些操作、工件和验证步骤对特定结论重要。我们引入了LEDGER——用于执行审查的分层证据和决策图，这是一个在观察到的代理会话上构建分层追踪图的追踪和审查系统。LEDGER保留追踪记录，同时将其分组为证据节点和工作流节点，将工件表示为证据锚点，并添加类型化语义边，将声明连接到支持操作、工件和检查。

    arXiv:2608.18398v1 Announce Type: cross  Abstract: Large language model (LLM) agents can now carry out long-horizon technical workflows involving complex tool use, code execution, file edits, and generated artifacts. As agents do more work faster, the productivity bottleneck shifts from producing outputs to auditing whether those outputs are correct and trustworthy. Agent observability systems make fine-grained execution events visible, but visibility alone still leaves reviewers to reconstruct which actions, artifacts, and validation steps matter for a particular conclusion. We introduce LEDGER - Layered Evidence and Decision Graphs for Execution Review, a tracing and review system that builds layered trace graphs over observed agent sessions. LEDGER preserves Trace Records while grouping them into Evidence Nodes and Workflow Nodes, representing artifacts as evidence anchors, and adding typed semantic edges that connect claims to supporting actions, artifacts, and checks. Through data
    
[^112]: 当清洁信号不足时：检测结构模糊性以实现安全的可穿戴压力分类

    When Clean Signals Are Not Enough: Detecting Structural Ambiguity for Safe Wearable Stress Classification

    [https://arxiv.org/abs/2608.18397](https://arxiv.org/abs/2608.18397)

    本文提出了一种轻量级的预推理监测器ICCM，通过量化个体特定的信号耦合发散度来识别结构模糊性，并在不重新训练分类器的情况下决定分类、延迟或弃权，从而提升可穿戴压力分类的安全性。

    

    arXiv:2608.18397v1 公告类型：新 摘要：可穿戴压力分类器在平均性能上可能表现优异，但对某个特定个体却可能完全失败。在WESAD数据集上，随机森林达到93.0%的平均准确率，但对第14号受试者的F1分数为0，该受试者的跨信号耦合在压力发作附近减弱。我们称此为结构模糊性：个体上看似合理的生理通道形成了跨信号模式，但这种模式在该人的非压力参考中支持不足。我们引入了个体共形耦合监测器（ICCM），这是一种轻量级且透明的预推理监测器，可量化个体特定的耦合发散度，并将每个窗口路由到分类、延迟或弃权，而无需重新训练下游分类器。在WESAD（N=15）和Stress-Predict（N=35）上，全队列的模糊性与准确率之间的Pearson关联为负（r = -0.607，p = 0.016；r = -0.412，p = 0.014）。稳健性分析缓和了这一发现：秩相关不显著。

    arXiv:2608.18397v1 Announce Type: new  Abstract: Wearable stress classifiers can achieve strong average performance while failing completely for a particular individual. On WESAD, a Random Forest reaches 93.0% mean accuracy yet yields F1 = 0 for Subject 14, whose cross-signal coupling weakens near stress onset. We call this structural ambiguity: individually plausible physiological channels form an inter-signal pattern that is poorly supported by the person's non-stress reference. We introduce the Individual Conformal Coupling Monitor (ICCM), a lightweight and transparent pre-inference monitor that quantifies subject-specific coupling divergence and routes each window to classify, defer, or abstain without retraining the downstream classifier. Across WESAD (N = 15) and Stress-Predict (N = 35), full-cohort Pearson associations between ambiguity and accuracy are negative (r = -0.607, p = 0.016; r = -0.412, p = 0.014). Robustness analyses temper this finding: rank correlations are not sig
    
[^113]: 一个崎岖的前沿：评估代码代理对语义保持变换的鲁棒性

    A Jagged Frontier: Evaluating Robustness of Code Agents to Semantics-Preserving Transformations

    [https://arxiv.org/abs/2608.18389](https://arxiv.org/abs/2608.18389)

    本文通过引入随机语义保持变换采样器，系统评估了代码代理在代码库语义等价改写下的鲁棒性，发现其修复性能会显著下降，揭示了现有代理的脆弱性。

    

    arXiv:2608.18389v1 公告类型：新 摘要：AI代码代理正日益被部署来解决真实软件问题，然而它们在表面代码变化下的可靠性仍鲜为人知。我们评估了当周围代码库被重写为语义等价形式时，修复仓库级问题的代码代理是否仍保持可靠。我们引入了一个随机变体采样器，应用常见的语义保持变换（SPTs）——涵盖控制流重写、死代码注入和标识符重命名——以生成扰动变体。我们评估了两种代理框架（mini-SWE agent和OpenCode），每种框架由四个前沿模型之一（Claude Opus 4.5、Kimi K2.5、MiniMax M2.5和Qwen 3.6-27B）支持，在来自SWE-bench Verified和SWE-bench Pro的实例上进行测试。对于每个实例，代理在未扰动和扰动变体上多次运行，产生配对解决率估计，以隔离扰动效应与内在变化。

    arXiv:2608.18389v1 Announce Type: new  Abstract: AI code agents are increasingly deployed to resolve real software issues, yet their reliability under superficial code variations remains poorly understood. We evaluate whether coding agents that repair repository-level issues remain reliable when the surrounding codebase is rewritten into a semantically equivalent form. We introduce a random variant sampler that applies common semantics-preserving transformations (SPTs) - spanning control-flow rewrites, dead-code injection, and identifier renaming - to produce perturbed variants. We evaluate two agentic scaffolds (mini-SWE agent and OpenCode) each backed by one of four frontier models (Claude Opus 4.5, Kimi K2.5, MiniMax M2.5, and Qwen 3.6-27B) across instances drawn from SWE-bench Verified and SWE-bench Pro. For each instance, the agent is run multiple times on the unperturbed and perturbed variants, yielding paired resolve-rate estimates that isolate the perturbation effect from intri
    
[^114]: TTSD-FAR：基于Fisher锚定恢复的测试时自蒸馏方法，用于大型视频语言模型中的缺失模态情感识别

    TTSD-FAR: Test-Time Self-Distillation with Fisher-Anchored Restoration for Missing-Modality Emotion Recognition in LVLMs

    [https://arxiv.org/abs/2608.18386](https://arxiv.org/abs/2608.18386)

    本文提出一种测试时自蒸馏方法，结合Fisher锚定恢复技术，使大型视频语言模型在模态缺失时能稳定自适应情感识别，避免漂移问题。

    

    大型视频语言模型（LVLMs）在野外多模态情感识别（ER）等多模态任务中表现出色。情感识别本质上是多模态的，需要对面部表情、声音、语言、生物信号和手势进行联合理解。然而，现实世界部署仍具挑战性：测试时模态可能缺失或含噪声。部分观测可视为相对于完整模态分布的分布偏移。基于熵最小化或困惑度降低的最先进测试时自适应（TTA）方法无法迁移到自回归LVLMs，而检索增强生成（RAG）在观测模态较弱时会退化。由于缺乏地面真值监督来验证每次更新，跨流自适应容易累积漂移，并在模型偏离可靠解后性能下降。因此，有效的解决方案必须适应任意缺失模态的情况。

    arXiv:2608.18386v1 Announce Type: cross  Abstract: Large video-language models (LVLMs) have shown remarkable performance on multimodal tasks like multimodal emotion recognition (ER) in the wild. ER is inherently multimodal, requiring a joint understanding of facial expressions, vocalizations, language, biosignals, and gestures. However, real-world deployment remains challenging: modalities may be missing or noisy at test time. Partial observations can be viewed as a distribution shift relative to the complete-modality distribution. SOTA TTA methods based on entropy minimization or perplexity reduction do not transfer to autoregressive LVLMs, while retrieval augmented generation (RAG) degrades when the observed modality is weak. Because no ground-truth supervision exists to verify individual updates, adaptation across this stream risks accumulating drift and degrading once the model departs from a reliable solution. An effective solution must therefore adapt to arbitrary missing-modalit
    
[^115]: 选择、重组，还是重新求解？单次测试时聚合的无候选对照实验

    Selection, Recombination, or a Fresh Solve? A Candidate-Free Control for Single-Pass Test-Time Aggregation

    [https://arxiv.org/abs/2608.18379](https://arxiv.org/abs/2608.18379)

    本文通过引入无候选对照实验，揭示了在测试时聚合中，候选上下文仅在存在多个正确候选时才有价值，而在所有候选错误时会损害性能，为高效推理提供了关键洞见。

    

    arXiv:2608.18379v1 公告类型：交叉 摘要：当每个候选答案都是错误时，正确候选选择不可用，但聚合调用仍能重新解决问题。因此，正确的聚合答案可能反映重组、重新求解或两者兼有。对于高效的测试时推理，关键问题是候选上下文是否在额外生成过程之外增加价值。我们引入了在相同最大输出令牌预算下的缺失无候选对照，并按正确候选数量进行分层。在AIME-2025和HMMT-2025上使用Qwen3-4B，当多个候选正确时，候选条件化提高了准确性（$\Delta_{\mathrm{cand}}$(c2+) = +0.290），当所有候选错误时降低了准确性（$\Delta_{\mathrm{cand}}$(c0) = -0.123），而在单一正确情况下结果未决。c2+和c0的结论在自适应双基准程序的保守校正后仍然成立。在这种反事实下，解释...

    arXiv:2608.18379v1 Announce Type: cross  Abstract: When every candidate is wrong, correct-candidate selection is unavailable, yet the aggregation call can still solve the problem afresh. A correct aggregate answer may therefore reflect recombination, fresh solving, or both. For efficient test-time reasoning, the relevant question is whether candidate context adds value beyond the additional generation pass. We introduce the missing candidate-free control under the same maximum output-token allowance and stratify by the number of correct candidates. Across AIME-2025 and HMMT-2025 with Qwen3-4B, candidate conditioning improves accuracy when multiple candidates are correct ($\Delta_{\mathrm{cand}}$(c2+) = +0.290), lowers accuracy when every candidate is wrong ($\Delta_{\mathrm{cand}}$(c0) = -0.123), and remains unresolved in the one-correct regime. The c2+ and c0 conclusions survive a conservative correction for the adaptive two-benchmark procedure. Under this counterfactual, the interpre
    
[^116]: 一道门不够：为智能体AI组合有状态的动作前控制

    One Gate Is Not Enough: Composing Stateful Pre-Action Controls for Agentic AI

    [https://arxiv.org/abs/2608.18360](https://arxiv.org/abs/2608.18360)

    本文发现智能体AI中多个动作前控制之间存在补救耦合，导致控制失效，并提出了一种“补救-重新门控”协议来恢复健全性，同时证明补救操作符不可交换，使补救顺序成为控制语义的关键部分。

    

    摘要：arXiv:2608.18360v1 公告类型：交叉 摘要：智能体AI系统在采取关键行动时，同时受多个动作前控制约束：权限门、资源门和证据门，这些控制可以在行动执行前允许、降级或补救该行动。本文的核心对象是补救引发的控制耦合：一个控制应用的补救措施可能改变另一个控制所评估的行动、证据或上下文，从而使其先前的判断失效。我们形式化了这种耦合，并提出了一种“补救-重新门控”协议，该协议在给定的有界、幂等设置及其假设下恢复了每个动作的健全性。我们还进一步表明，两种已实现的补救操作符（证据替换和资源预算降级）不满足交换律——一个有限模型检查器发现了具体的反例实例——这使得补救顺序成为控制平面语义的一部分，而非实现细节。一个受治理的证据缓冲区信任其输入。

    arXiv:2608.18360v1 Announce Type: cross  Abstract: Agentic AI systems take consequential actions governed by more than one pre-action control at once: authority, resource, and evidence gates that can admit, degrade, or remediate an action before it executes. This paper's central object is remediation-induced control coupling: a remediation applied by one control can change the action, evidence, or context another control evaluates, invalidating that control's earlier judgment. We formalize this coupling and give a remediate-and-regate protocol that restores per-action soundness in the current bounded, idempotent setting under its stated assumptions. We further show that the two implemented remediation operators (evidence substitution and resource-budget downroute) do not commute -- a finite-model checker finds concrete counterexample instances -- making remediation order part of the control-plane semantics rather than an implementation detail. A governed evidence buffer that trusts its
    
[^117]: 面向可执行终端与MCP代理的任务条件化最小权限学习

    Task-Conditioned Least-Privilege Learning for Executable Terminal and MCP Agents

    [https://arxiv.org/abs/2608.18351](https://arxiv.org/abs/2608.18351)

    本文提出了一种通过后训练教4B模型在终端和MCP环境中选择任务条件化最小权限的框架，利用六维审计和确定性验证器来减少越权错误。

    

    arXiv:2608.18351v1 公告类型：交叉 摘要：使用工具的大型语言模型代理在完成任务时，可能会行使用户未授予或任务不需要的权限，导致越权错误。传统的权限门控系统单独用于验证代理环境是不够的。我们研究后训练能否教会一个4B参数模型在可执行终端和模型上下文协议（MCP）环境中选择任务条件化的权限，以补充这些措施。我们提出一个框架，其中每个动作在执行前和从观察到的效果中沿六个风险维度进行审计。该审计使用确定性验证器进行，这些验证器对完成度、证据、确切状态、禁止尝试和安全成功进行评分。结合预定义的任务特定充分权限范围，我们为轨迹确定任务特定的超额权限值，然后在后训练中进行优化。我们发现，af

    arXiv:2608.18351v1 Announce Type: cross  Abstract: Tool-using large language-model agents can complete a task while exercising authority that the user did not grant or the task does not need, causing excess-authority errors. Traditional permission gating systems alone for validating agent environments are insufficient. We study whether post-training can teach a 4B-parameter model to choose task-conditioned authority in executable terminal and Model Context Protocol (MCP) environments to complement those measures. We propose a framework where each action is audited before execution and again from observed effects along six dimensions of risk. This auditing is conducted using deterministic verifiers that score completion, evidence, exact state, prohibited attempts, and safe success. In conjunction with predefined task-specific sufficient-authority envelopes, we determine task-specific excess privilege values for trajectories, which are then optimized for in post-training. We find that af
    
[^118]: 跨主族元素的耦合簇分子性质预测，其外推能力超越训练规模

    Coupled-cluster molecular properties across the main group that extrapolate beyond training size

    [https://arxiv.org/abs/2608.18346](https://arxiv.org/abs/2608.18346)

    提出MEHnet-MG等变网络，从一次廉价DFT计算预测有效哈密顿量，在九种主族元素上以耦合簇精度高效获取多种分子性质，误差比DFT降低3.8至230倍。

    

    arXiv:2608.18346v1 公告类型：交叉 摘要：耦合簇理论定义了分子电子结构性质的精度标准，但其计算成本过高，难以常规应用；而密度泛函理论虽成本低廉，却存在系统性偏差。我们通过一个单一的等变网络MEHnet-MG解决了这一权衡问题，该网络从一次廉价的B3LYP/def2-SVP计算中预测有效的单电子哈密顿量，并从中推导出一系列性质（能量、光学带隙、偶极矩、四极矩、极化率、Mulliken原子电荷和Mayer键级），在九个主族元素（包括服务不足的磷、硫和氯化学）上达到耦合簇精度。该模型基于一个新的内部数据集训练，该数据集包含所有九种元素在CCSD(T)级别计算的多属性标签。在保留测试集上，相对于半局域、杂化和双杂化DFT，它将每个属性的误差降低了3.8到230倍。

    arXiv:2608.18346v1 Announce Type: cross  Abstract: Coupled-cluster theory defines the accuracy standard for molecular electronic-structure properties but scales too steeply for routine application, whereas density-functional theory is affordable yet systematically biased. We resolve this trade-off with a single equivariant network, MEHnet-MG, that predicts an effective one-electron Hamiltonian from one inexpensive B3LYP/def2-SVP calculation and derives a broad suite of properties from it (energy, optical gap, dipole, quadrupole, polarizability, Mulliken atomic charges, and Mayer bond orders) at coupled-cluster accuracy across nine main-group elements, including the under-served phosphorus, sulfur, and chlorine chemistries. The model is trained on a new in-house dataset of multi-property labels computed at the CCSD(T) level for all nine elements. On a held-out test set, it reduces the error of every property by a factor of 3.8 to 230 relative to semi-local, hybrid, and double-hybrid DFT
    
[^119]: 低功耗、神经形态的声学异常检测用于持续机器监控

    Low-Power, Neuromorphic, Acoustic Anomaly Detection for Persistent Machine Monitoring

    [https://arxiv.org/abs/2608.18341](https://arxiv.org/abs/2608.18341)

    该论文在Loihi 2神经形态处理器上实现了基于自编码器的低功耗声学异常检测，在干净和噪声条件下均达到或超越基线性能，且每个样本能耗比传统方法低两个数量级。

    

    arXiv:2608.18341v1 公告类型：交叉 摘要：持续声学监控可以在无需物理接触的情况下检测机器故障，但始终在线的推理受到功耗、延迟和部署复杂性的限制。我们展示了在Intel Loihi 2神经形态处理器上，基于自编码器的声学异常检测在干净和噪声条件下的性能。对数梅尔特征在芯片外计算；归一化、自编码器推理、L1重建评分和阈值判断在芯片上运行。在干净、麦克风位置不变的ToyADMOS ToyCar基准测试中，片上模型实现了0.9959的AUC和最大假阳性率为0.1时0.9785的标准化pAUC。在DCASE 2026任务2 ToyCar噪声基准中，模型实现了源域AUC 0.7990、目标域AUC 0.6466和pAUC 0.6426，超过了报告的基线指标。在16芯片Loihi 2 VPX系统上的功耗分析显示，实时吞吐量下每个样本的动态能量为0.0406–0.0426毫焦耳，比传统方法低两个数量级。

    arXiv:2608.18341v1 Announce Type: cross  Abstract: Persistent acoustic monitoring can detect machine faults without physical contact, but always-on inference is constrained by power, latency, and deployment complexity. We demonstrate autoencoder-based acoustic anomaly detection on an Intel Loihi 2 neuromorphic processor under clean and noisy conditions. Log-mel features are computed off chip; normalization, autoencoder inference, L1 reconstruction scoring, and thresholding run on chip. In a clean, microphone-position-invariant ToyADMOS ToyCar benchmark, the on-chip model achieves 0.9959 AUC and 0.9785 standardized pAUC at maximum false-positive rate 0.1. In the DCASE 2026 Task 2 ToyCar noisy benchmark, the model achieves source AUC 0.7990, target AUC 0.6466, and pAUC 0.6426, exceeding reported baseline metrics. Power profiling on a 16-chip Loihi 2 VPX system shows real-time throughput with 0.0406$\unicode{x2013}$0.0426 mJ dynamic energy per sample, two orders of magnitude lower than bo
    
[^120]: 从推理到适应：视觉语言模型的统一最优传输视角

    From Inference to Adaptation: A Unified Optimal Transport View of Vision Language Model

    [https://arxiv.org/abs/2608.18339](https://arxiv.org/abs/2608.18339)

    本文提出一种基于最优传输的统一视角，将视觉语言模型的推理与测试时适应目标对齐，以解决分布变化下伪标签噪声和模态关系建模不足的问题。

    

    arXiv:2608.18339v1 公告类型：交叉 摘要：视觉语言模型（VLMs）展示了显著的零样本能力，但在推理过程中对现实世界的分布变化仍然敏感。尽管大量工作致力于在测试时适应VLMs，但它们严重依赖于推理过程中直接从原始嵌入相似性预测的噪声伪标签，这些标签在分布变化下不可靠，并误导了适应过程。为避免噪声放大，现有工作适应过程中使用粗略的替代目标，这未能显式建模不同模态间的样本级关系，导致与推理目标不匹配，从而性能提升有限。在本工作中，我们旨在弥合VLM推理与适应之间的分离目标，并提出一种原则性的VLM测试时适应方法，称为\algname。对于VLM推理，我们将零样本图像分类任务表述为跨模态对齐问题。

    arXiv:2608.18339v1 Announce Type: cross  Abstract: Vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities yet remain sensitive to real-world distribution shifts during inference. Although significant efforts are devoted to adapting VLMs at test time, they rely heavily on noisy pseudo-labels predicted directly from raw embedding similarities during inference, which are unreliable under distribution shift and mislead the adaptation. To avoid noise amplification, existing works craft coarse-grained surrogate objectives during adaptation, which fail to explicitly model sample-level relationships across different modalities, creating objective mismatch with inference, thus leading to marginal performance improvement. In this work, we aim to bridge the detached objectives of inference and adaptation for VLMs, and propose a principled VLM TTA method called \algname. For VLM inference, we formulate the zero-shot image classification task as a cross-modal alignment pr
    
[^121]: 测量部分得分差距：针对越南2025年凸性评分方案的严格基准

    Measuring the Partial-Credit Gap: A Strict Benchmark on Vietnam's 2025 Convex Marking Scheme

    [https://arxiv.org/abs/2608.18336](https://arxiv.org/abs/2608.18336)

    该论文指出传统基准测试的准确率评分无法反映越南2025年高考凸性评分方案中部分知识的惩罚性扣分，并提出了一个包含632个题目的THPT-Ladder基准来严格衡量这种差距。

    

    arXiv:2608.18336v1 公告类型：新 摘要：在人类考试中评估语言模型时，基准测试通常将每个回答评为正确或错误，并报告整体准确率。这种方法假设部分知识应获得相应比例的分数，但在考试采用非加性评分方案时，这一假设不成立。越南2025年全国高中毕业考试的改革证明了这种替代的成本。在考试的第二部分，考生需要评估每道题的四个判断题。评分是凸性的：正确陈述的数量对应0分、0.10分、0.25分、0.50分或1.00分。正确识别三个陈述仅得0.50分，而非标准准确率指标会给予的0.75分。由于第二部分占考试总分的4.00分（满分10.00分），报告准确率会通过奖励部分知识而夸大分数，而国家明确对此类部分知识进行惩罚。我们引入了THPT-Ladder，这是一个包含来自21个官方来源的632个题目的基准测试。

    arXiv:2608.18336v1 Announce Type: new  Abstract: When evaluating language models on human exams, benchmarks typically score each response as right or wrong and report the overall accuracy. This approach assumes that partial knowledge is worth proportional credit, an assumption that fails when an examination uses a non-additive grading scheme. The 2025 reform of Vietnam's National High School Graduation Examination demonstrates the cost of this substitution. In Part II of the exam, candidates evaluate four true/false statements per question. The grading is convex: the number of correct statements earns 0, 0.10, 0.25, 0.50, or 1.00 points. Identifying three statements correctly pays 0.50 points, not the 0.75 points that standard accuracy metrics would award. Because Part II accounts for 4.00 of the exam's 10.00 points, reporting accuracy inflates the score by rewarding partial knowledge that the state explicitly penalizes. We introduce THPT-Ladder, a benchmark of 632 items from 21 offici
    
[^122]: 治理记录作为监督：验证者选择的自我训练用于结构化工作流修复

    Governance Records as Supervision: Verifier-Selected Self-Training for Structured Workflow Repair

    [https://arxiv.org/abs/2608.18324](https://arxiv.org/abs/2608.18324)

    本研究提出一种利用机器可验证工作流产生的治理记录进行自我训练的方法，使有限模型通过验证者选择的计划提升一次性执行能力，显著提高成功率和效率。

    

    arXiv:2608.18324v1 公告类型：新 摘要：机器可验证的工作流产生治理记录，这些记录将任务合同、模型尝试、验证者决策、接受输出和目标来源关联起来。我们测试这些记录是否能监督有限模型，将偶尔或昂贵的能力巩固为可靠的一次性执行。在全新的、结构不相交的PlanBench重新规划案例中，Qwen3-14B思考模式生成了24个计划，这些计划被独立编写的VAL验证者接受。这些计划训练了同一检查点用于非思考执行，无需预言机目标或更强教师。在80个未打开案例中，VAL接受的计划从1个增加到57个，其中56个配对改进且零回归；思考模式达到30个。适配器在所有案例中模式有效，并使用约1/56的思考模式平均延迟。单独的配对接口-治愈门未通过。匹配消融固定了源案例、52个候选池、24个目标计数、模型、配方和种子，同时比较了对照组。

    arXiv:2608.18324v1 Announce Type: new  Abstract: Machine-verifiable workflows produce governance records linking a task contract, model attempt, verifier decision, accepted output, and target origin. We test whether these records can supervise bounded models, consolidating occasional or expensive capability into reliable one-shot execution.   On fresh, structure-disjoint PlanBench replanning cases, Qwen3-14B thinking generated 24 plans admitted by the independently authored VAL verifier. Those plans trained the same checkpoint for non-thinking execution, without oracle targets or a stronger teacher. On 80 unopened cases, VAL-accepted plans increased from 1 to 57, with 56 paired gains and zero regressions; thinking reached 30. The adapter was schema-valid on all cases and used approximately 1/56 of thinking's mean latency. The separate paired interface-cure gate did not pass.   A matched ablation fixed the source cases, 52-candidate pool, 24-target count, model, recipe, and seed while c
    
[^123]: FedCoRe：面向医疗联邦学习中缺失模态的目标自适应补全

    FedCoRe: Target-Adaptive Completion for Missing Modalities in Healthcare Federated Learning

    [https://arxiv.org/abs/2608.18311](https://arxiv.org/abs/2608.18311)

    FedCoRe通过利用成对模态示例进行表示空间修正，在不生成合成数据的情况下，有效恢复了医疗联邦学习中缺失模态（如心电图）导致的性能损失（恢复49%的AUROC损失）。

    

    联邦多模态模型通常假设每个站点拥有所有模态，但医院在电子健康记录、胸部X光片和心电图的访问权限上存在差异。我们在一个基于MIMIC数据集的呼吸恶化任务上，通过模拟联邦学习客户端研究了这一设置，并引入了FedCoRe（联邦跨模态表示补全）。FedCoRe学习表示空间或逻辑空间中的修正，而非生成合成的心电图或胸部X光图像。当客户端观察到可能在部署时缺失的模态时，它会评估同一示例在有和没有该模态的情况，以获得成对监督。只有拥有此类成对数据的客户端更新补全模块，验证时保留未改变的预测。在评估期间，我们冻结训练好的多模态预测器，以确保测量到的差异仅来自补全。隐藏心电图使AUROC降低了约0.085；成对示例的FedAvg恢复了0.0415的AUROC，即恢复了49.0%的性能损失。因此，我们...

    arXiv:2608.18311v1 Announce Type: cross  Abstract: Federated multimodal models often assume every site has every modality, although hospitals differ in access to EHRs, chest radiographs, and ECGs. We study this setting on a MIMIC-derived respiratory deterioration task with simulated FL clients and introduce FedCoRe (Federated Cross-Modal Representation Completion). FedCoRe learns representation- or logit-space corrections rather than generating synthetic ECGs or CXR images. When a client observes a modality that may be missing at deployment, it evaluates the same example with and without that modality to obtain paired supervision. Only clients with such pairs update the completion module, and validation may retain the unchanged prediction. We freeze the trained multimodal predictor during evaluation so that measured differences come only from completion. Hiding ECG reduced AUROC by about 0.085; paired-example FedAvg restored 0.0415 AUROC, or 49.0% of the lost performance. We therefore 
    
[^124]: ComponentBench：诊断计算机使用代理的组件级故障

    ComponentBench: Diagnosing Component-Level Failures in Computer-Use Agents

    [https://arxiv.org/abs/2608.18307](https://arxiv.org/abs/2608.18307)

    ComponentBench提出了一个组件级基准测试和诊断流程，通过97个规范组件和2910个验证任务，填补了计算机使用代理评估中长周期工作流与原子级测试之间的中间层空白，并支持任务成功率和交互效率的双重评估。

    

    arXiv:2608.18307v1 公告类型：新 摘要：当前对计算机使用代理的评估分为长周期工作流基准测试和原子级GUI接地测试。这留下了一个未被充分衡量的中间层：现实的组件中心交互（例如，切换一个按钮组），这些交互既短到足以进行诊断，又丰富到足以捕捉现代界面的负担。我们提出了ComponentBench，一个用于在现代Web UI上对计算机使用代理进行组件级评估的基准测试和诊断流程。ComponentBench围绕一个与库无关的本体论组织，包含97个规范UI组件，实例化为2910个程序化验证的任务，涵盖广泛使用的组件库，并配有清理后的人类参考轨迹，从而能够评估任务成功率和交互效率。除了任务收集，我们引入了一个可扩展的流程，用于在实施后审计实际的结构难度，并综合结构化的失败分析。

    arXiv:2608.18307v1 Announce Type: new  Abstract: Current evaluation of computer-use agents is split between long-horizon workflow benchmarks and atomic GUI-grounding tests. This leaves an under-instrumented middle layer: realistic component-centered interactions (e.g., toggle a button set) that are short enough to diagnose and rich enough to capture the burdens of modern interfaces. We present ComponentBench, a benchmark and diagnostic pipeline for component-level evaluation of computer-use agents on modern web UIs. ComponentBench is organized around a library-agnostic ontology of 97 canonical UI components instantiated as 2,910 programmatically verified tasks across widely used component libraries, paired with cleaned human reference trajectories that enable evaluation of both task success and interaction efficiency. Beyond task collection, we introduce a scalable pipeline for auditing realized structural difficulty after implementation and synthesizing structured failure analyses acr
    
[^125]: SESSE：草图、扩展、排序、总结、评估——通过结构化分解实现LLM作为评判者的评估

    SESSE: Sketch, Expand, Sort, Summarize, Evaluate -- LLM-as-Judge Evaluation via Structured Decomposition

    [https://arxiv.org/abs/2608.18303](https://arxiv.org/abs/2608.18303)

    SESSE提出了一种无需训练的框架，通过结构化分解将LLM评判过程转化为可解释的子问题，在保持性能的同时提供了诊断标签模糊性和评判者错误的能力。

    

    arXiv:2608.18303v1 公告类型：新 摘要：LLM作为评判者的评估将响应质量评估简化为单一的总体A/B偏好选择，没有机制来隔离哪些质量维度驱动了偏好，或区分模型错误与真实的标签模糊性。我们提出SESSE（草图、扩展、排序、总结、评估），一个无需训练的框架，将整体判断分解为直接从评判者自身错误案例中挖掘的结构化子问题；无需神谕响应、任务特定评分标准或微调。在RewardBench（n=1,000）上，SESSE与链式思维基线达到近乎持平，并与微调专家模型RISE-Judge-32B（92.7%）竞争，同时完全保持无需训练。每个标准的投票证据提供了可解释的审计轨迹，用于诊断标签模糊性和评判者失败模式，而这些是单一整体输出标记无法提供的。

    arXiv:2608.18303v1 Announce Type: new  Abstract: LLM-as-judge evaluation reduces response quality assessment to a single holistic A/B preference choice, providing no mechanism to isolate which quality dimensions drove the preference or distinguish model errors from genuine label ambiguity. We propose SESSE (Sketch, Expand, Sort, Summarize, Evaluate), a training-free framework that decomposes holistic judgment into structured sub-questions mined directly from the judge's own error cases; requiring no oracle responses, task-specific rubrics, or fine-tuning. On RewardBench (n=1,000), SESSE achieves near-parity with the chain-of-thought baseline and is competitive with RISE-Judge-32B (92.7%), a fine-tuned specialist, while remaining fully training-free. Per-criterion vote evidence provides an interpretable audit trail for diagnosing label ambiguity and judge failure modes unavailable from a single holistic output token.
    
[^126]: 大型推荐解释中LLM即评判者的生命周期

    The Lifecycle of LLM-as-a-Judge for Large-Scale Recommendation Explanations

    [https://arxiv.org/abs/2608.18300](https://arxiv.org/abs/2608.18300)

    本文提出LLM评判者在生产系统中具有构建、训练、部署和持续维护的生命周期，并以Netflix推荐解释评估为例，强调其动态维护而非静态评估的重要性。

    

    arXiv:2608.18300v1 公告类型：新 摘要：LLM即评判者（LLM-as-a-Judge）利用大型语言模型来评估由另一个AI应用或模型生成的自然语言，已成为一种标准且可扩展的方法，用于加速和扩展昂贵的人工评估。然而，大多数工作将评判者视为静态产物，仅在构建时或针对固定基准进行一次评估。相反，我们认为，在生产系统中运行的LLM评判者应被理解为一个具有生命周期的实体：它必须被构建、训练、部署，并随着周围数据的演变而持续维护，每个阶段都面临独特的技术和运营挑战。我们展示了Netflix中用于评估面向用户的推荐解释的LLM评判者的这种生命周期，在我们的流程中，每周生成并评估数十万个不同节目级别的解释，并通过移动体验服务数百万会员。我们的框架包含四个阶段。

    arXiv:2608.18300v1 Announce Type: new  Abstract: LLM-as-a-Judge, which leverages a large language model to evaluate natural language generated by another AI application or model, has become a standard, scalable approach for accelerating and extending costly human evaluation. However, most work treats a judge as a static artifact, evaluating it once at construction or against a fixed benchmark. In contrast, we argue that an LLM judge running in a production system is better understood as having a lifecycle: it must be built, trained, deployed, and continuously maintained as the surrounding data evolves, and each phase poses distinct technical and operational challenges.   We present such a lifecycle for the LLM judges that evaluate user-facing recommendation explanations at Netflix, where our pipeline generates and the judges assess hundreds of thousands of distinct show-level explanations per week, served across the mobile experience to millions of members. Our framework has four phase
    
[^127]: FairGlucose：一个CGM公平性基准揭示人群验证中隐藏的亚组差异

    FairGlucose: A CGM Fairness Benchmark Reveals Subgroup Disparities Hidden in Population-Level Validation

    [https://arxiv.org/abs/2608.18296](https://arxiv.org/abs/2608.18296)

    本研究构建了首个跨12个人口统计分层的CGM公平性基准，发现人群级验证指标稳定但亚组间存在显著预测误差差异，且该差异普遍存在于所有模型中，揭示CGM预测任务固有的公平性问题。

    

    随着基于CGM的AI工具接近临床部署，其准确性在不同患者人口统计特征中是否公平仍未得到充分测试。为进行这一评估，我们构建了FairGlucose，一个包含300名患者的CGM队列，在12个人口统计分层（年龄×性别×1型/2型糖尿病）中平衡，包含132,480个预测样本和81名患者记录的3,945个独特行为事件（饮食、运动、用药）。对四个模型家族的33个模型进行2小时血糖预测基准测试，我们发现人群水平的外部验证可能掩盖显著的亚组差异。总体分布外指标看似稳定（约1.0），但亚组水平比率范围从0.8到1.4，其中1型糖尿病患者的预测误差比2型糖尿病患者高6 mg/dL（p < 0.001）。这种差异在所有33个模型中持续存在，表明这是预测任务本身的性质，而非任何单一架构所致。进一步分析表明...

    arXiv:2608.18296v1 Announce Type: cross  Abstract: As CGM-based AI tools approach clinical deployment, whether their accuracy is equitable across patient demographics remains insufficiently tested. To enable this evaluation, we constructed FairGlucose, a 300-patient CGM cohort balanced across 12 demographic strata (age x gender x type 1/type 2 diabetes), with 132,480 forecasting samples and 3,945 unique behavioral events (meals, exercise, medication) logged by 81 patients. Benchmarking 33 models across four families on 2-hour glucose forecasting, we find that population-level external validation can conceal substantial subgroup disparities. Aggregate out-of-distribution metrics appear stable (approximately 1.0), yet subgroup-level ratios range from 0.8 to 1.4, with T1D patients showing 6 mg/dL higher prediction error than T2D (p < 0.001). This disparity persists across all 33 models, suggesting a property of the prediction task rather than any single architecture. Further analysis show
    
[^128]: 无金标准标签下AI生成数据的去偏推断：通过多重不完美测量进行识别

    Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements

    [https://arxiv.org/abs/2608.18294](https://arxiv.org/abs/2608.18294)

    本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。

    

    越来越多的学者使用AI来测量变量，并将其纳入后续的下游分析。尽管AI测量的变量通常被视为无误差观测，但忽略自动化测量中的预测误差会导致下游分析中的显著偏差和无效置信区间，即使AI测量准确度很高（例如超过90%）。现有的解决方案，如基于设计的有监督学习和预测支持推断，将基于AI的易错测量与金标准标签相结合，但在某些应用领域中，获取金标准标签可能成本高昂且困难。在本文中，我们提出了多重不完美测量的去偏推断（DMM），这是一个结合多个易错AI测量以实现无需金标准标签的有效下游推断的框架。基于CP分解的既有成果，DMM假设这些测量是独立的。

    arXiv:2608.18294v1 Announce Type: cross  Abstract: An increasing number of scholars use AI to measure variables they subsequently include in downstream analyses. Although AI-measured variables are often analyzed as if observed without error, ignoring prediction errors in automated measurement leads to substantial bias and invalid confidence intervals in downstream analyses, even if AI measurement accuracy is high, e.g., above 90%. Existing solutions, such as design-based supervised learning and prediction-powered inference, combine error-prone AI-based measurements with gold-standard labels, which may be costly and difficult to obtain in some application areas.   In this paper, we propose debiased inference with multiple imperfect measurements (DMM), a framework that combines multiple error-prone AI measurements to enable valid downstream inference without gold-standard labels. Building on the established results on CP decomposition, DMM assumes that these measurements are independent 
    
[^129]: 高风险公共部门应用中开放模型的结构化信息提取评估

    Evaluating Structured Information Extraction with Open Models in a High Risk Public Sector Application

    [https://arxiv.org/abs/2608.18289](https://arxiv.org/abs/2608.18289)

    本文提出了一个针对高风险公共部门应用（如国际学生申请处理）的开源系统端到端结构化信息提取基准，填补了现实多步骤流程评估的空白。

    

    arXiv:2608.18289v1 公告类型：新 摘要：从非结构化文档中提取结构化信息是各行业数字化转型的关键组成部分。虽然专有解决方案主导商业应用，但快速发展的开源光学字符识别（OCR）引擎、大型语言模型（LLM）和视觉语言模型（VLM）生态系统提供了可访问的替代方案。然而，在现实的多步骤提取流程上进行系统性评估仍然稀缺。负责任的此类提取工具使用要求在现实任务上进行全面评估，尤其是当这些解决方案将成为欧盟AI法案归类为高风险的公共部门应用的关键组件时。为解决这一空白，我们提出了一个综合基准，评估开源系统在复杂真实世界文档处理任务上的端到端性能，该任务被归类为高风险：国际学生申请。

    arXiv:2608.18289v1 Announce Type: new  Abstract: The extraction of structured information from unstructured documents represents a critical component of digital transformations in all sectors. While proprietary solutions dominate commercial applications, a rapidly growing ecosystem of open-source Optical Character Recognition (OCR) engines, Large Language Models (LLMs), and Vision-Language Models (VLMs) offers accessible alternatives. However, systematic evaluations on realistic, multi-step extraction pipelines remain scarce. Responsible usage of such extraction tools require comprehensive evaluations on realistic tasks, especially as these solutions will be key components of applications in the public sector that the EU AI act categorizes as high risk. To address this gap we present a comprehensive benchmark assessing the end-to-end performance of open-source systems on a complex real-world document processing task classified as high risk: Student applications for an international stu
    
[^130]: 什么因素使得软件问题解决任务对智能体来说变得困难？

    What Makes Software Issue Resolution Tasks Difficult for Agents?

    [https://arxiv.org/abs/2608.18280](https://arxiv.org/abs/2608.18280)

    本文提出一个测量框架，通过分析CoderForge-Preview数据集中的任务补丁、仓库和提示特征，系统量化了软件任务的结构属性对智能体解决成功率的影响，揭示了哪些静态属性可预测任务难度。

    

    摘要：arXiv:2608.18280v1 公告类型：交叉 摘要：背景。智能体系统的进展同时且迅速地使基准测试趋于饱和。尽管这一现象常被讨论，但由于缺乏对任务难度的控制和表征，基准分数仍然难以解释。更具体地说，我们目前对什么使一个任务比另一个更难，以及任务难度在多大程度上可以从静态任务属性中预测，了解甚少。目标。我们提出了一个测量框架，以研究和系统量化软件任务的结构属性如何对应于智能体在问题解决任务中的成功率。方法。我们在CoderForge-Preview（迄今为止最大的编码智能体轨迹开放数据集）上进行了一项大规模实证研究，通过提取任务补丁、仓库和提示中的特征。我们使用集成方法、SHAP归因和效应评估了每个特征对任务结果预测能力。

    arXiv:2608.18280v1 Announce Type: cross  Abstract: Background. Advances in agentic systems are simultaneously, and rapidly, saturating benchmarks. Despite this often discussed phenomena, benchmark scores remain difficult to interpret due to the lack of control and characterization of task difficulty. More specifically, we currently have little understanding of what makes one task harder than another, and to what extent task difficulty is predictable from static task properties. Aims. We propose a measurement framework to investigate and systematically quantify what structural properties of software tasks correspond to agent success rates for issue resolution tasks. Method. We conducted a large scale empirical study on CoderForge-Preview, the largest open dataset of coding agent trajectories to date, by extracting features across task patch, repository and prompt. We evaluated the predictive power of each feature against task outcomes using ensemble methods, SHAP attribution, and effect
    
[^131]: SeisEvo：地震数据重建算法的代理演化

    SeisEvo: Evolution of Seismic Data Reconstruction Algorithms by Agents

    [https://arxiv.org/abs/2608.18272](https://arxiv.org/abs/2608.18272)

    SeisEvo通过LLM驱动的多代理搜索，从经典算法出发自动演化出独立的白盒地震数据重建算法，仅修改用户指定组件并强制执行物理约束，无需人工设计或黑盒模型。

    

    arXiv:2608.18272v1 公告类型：交叉 摘要：经典地震数据重建依赖于手动设计的结构先验和迭代算子，其耦合设计空间远大于手动试错所能系统探索的范围。深度学习方法将重建规则编码在学习权重中，而非可检查和修改的显式算子中。我们提出SeisEvo（地震算法演化），它不优化单一重建结果，而是搜索生成该结果的算法。从经典重建算法出发，由LLM驱动的多代理搜索仅修改用户开放编辑的组件，而不预设待发现的机制。违反任务物理约束的候选算法被直接拒绝，其余算法通过执行评分。输出既不是代理系统也不是神经网络，而是一个独立的白盒算法，无需额外依赖。

    arXiv:2608.18272v1 Announce Type: cross  Abstract: Classical seismic data reconstruction relies on manually designed structural priors and iterative operators, whose coupled design space is far larger than manual trial and error can explore systematically. Deep-learning methods encode the reconstruction rules in learned weights rather than in an explicit operator that can be inspected and modified. We propose SeisEvo (Seismic Algorithm Evolution), which does not optimize a single reconstruction result but searches for the algorithm that produces it. Starting from a classical reconstruction algorithm, an LLM-driven multi-agent search modifies only the components that the user has opened for editing, without prescribing the mechanism to be discovered. Candidates that violate the physical constraints of the task are rejected outright, and the remaining ones are scored by execution. The output is neither an agent system nor a neural network, but a standalone white-box algorithm that requir
    
[^132]: AI提示如何教会我们理解人类行为的结构

    How AI Prompts Can Teach Us About the Structure of Human Behavior

    [https://arxiv.org/abs/2608.18265](https://arxiv.org/abs/2608.18265)

    本文提出一种基于AI提示的方法，通过类型向量最小化与人类选择的距离，发现人类行为可仅用风险厌恶、策略复杂性和信任三个维度精确匹配，并聚类为少数群体。

    

    arXiv:2608.18265v1 公告类型：交叉 摘要：我们介绍了一种通用且易于实现的基于AI的方法，用于研究人类行为的结构和复杂性。我们为大型语言模型分配一个“类型向量”，然后提示它在观察到人类选择的场景中选择行动。例如，类型向量(2,4)变为“你是一个具有以下特征的玩家：利他主义5分中得2分，风险厌恶5分中得4分”，之后提示它做出选择。我们变化维度（如利他主义、公平性、信任等）和数值（如1-5）以最小化与人类选择的距离。将该方法应用于来自超过35个国家、78,657名受试者在10个经典经济游戏角色中做出的119,147个决策，我们发现人类行为可以用三个维度紧密匹配：风险厌恶、策略复杂性和信任。此外，适合跨游戏个体的类型聚类为少于十几个组，这是一个...

    arXiv:2608.18265v1 Announce Type: cross  Abstract: We introduce a general, easy-to-implement AI-based method for studying the structure and complexity of human behavior. We assign a large language model a ``type vector'' and then prompt it to choose actions across settings in which we observe human choices. For instance, the type vector (2,4) becomes ``You are a player characterized by the following profile: 2 out of 5 in Altruism, 4 out of 5 in Risk Aversion,'' after which it is prompted to make choices. We vary the dimensions (e.g., Altruism, Fairness, Trust, $\dots$) and values (e.g., 1--5) to minimize distance to human choices. Applying the method to 119,147 decisions made by 78,657 subjects from more than 35 countries across 10 classic economic game roles, we find that human behavior can be closely matched using three dimensions: Risk Aversion, Strategic Sophistication, and Trust. Moreover, the types needed to fit individuals across games cluster into fewer than a dozen groups, an
    
[^133]: 设计即缓存？针对边缘内存带宽墙训练混合专家路由器局部性：一项预注册的负结果与系统测量研究

    Cacheable by Design? Training Mixture-of-Experts Routers for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result with a Systems Measurement Study

    [https://arxiv.org/abs/2608.18261](https://arxiv.org/abs/2608.18261)

    本研究通过系统测量和预注册实验证明，MoE模型的路由局部性虽存在但难以通过训练提升，且边缘内存带宽墙是实际部署中的主要瓶颈，提出了可复现的负结果和测量工具。

    

    arXiv:2608.18261v1 公告类型：新 摘要：在单个8 GB GPU上服务一个235B参数的混合专家（MoE）模型，瓶颈不在于计算，而在于内存带宽：解码必须从任何存储层流式传输每个令牌的活跃专家，而在消费级硬件上，大多数专家位于比RAM慢得多的SSD上。我们在Qwen3-235B（Q4_K_M，134 GB）上量化了这一带宽墙：实测解码速度为0.44令牌/秒（热启动），与每令牌字节数/带宽模型匹配，而一种应能摊销一次磁盘扫描的批处理方案，在批大小32时因分页抖动而崩溃。我们构建了llama-moe-trace，一个零侵入的路由遥测工具，并在Qwen3-30B上测量路由：相邻令牌专家重用是随机概率的2.0倍，95%的流量使用52.5%的专家，而一个仅占13.4%专家的LRU缓存服务了66%的请求。然后我们探究可缓存性是否可训练：我们预注册了带有辅助局部性和领域路由器损失的137M MoE语言模型训练，在联合标准下进行。

    arXiv:2608.18261v1 Announce Type: new  Abstract: Serving a 235B-parameter Mixture-of-Experts (MoE) model on a single 8 GB GPU is bottlenecked not by compute but by memory bandwidth: decode must stream each token's active experts from whichever tier holds them, and on consumer hardware most experts sit on an SSD far slower than RAM. We quantify this bandwidth wall on Qwen3-235B (Q4_K_M, 134 GB): measured decode is 0.44 tok/s warm, matching a bytes-per-token / bandwidth model, while a batching scheme that should amortize one disk sweep instead collapses at batch 32 from paging thrash. We build llama-moe-trace, a zero-surgery router-telemetry tool, and measure routing on Qwen3-30B: adjacent-token expert reuse is 2.0x chance, 95% of traffic uses 52.5% of experts, and an LRU cache of 13.4% of experts serves 66% of requests. We then ask whether cacheability is trainable: we pre-register training of 137M MoE language models with auxiliary locality and domain router losses, under joint criteri
    
[^134]: Redakto - 大型语言模型的隐身标签页

    Redakto - The Incognito Tab for LLMs

    [https://arxiv.org/abs/2608.18260](https://arxiv.org/abs/2608.18260)

    Redakto 是一个用于在将文本输入LLM前进行匿名化和假名化的工具，提供最先进的PII编辑功能，并通过Web应用和开发者接口方便使用。

    

    大型语言模型（LLM）正日益广泛应用于日常应用中。在使用LLM或一般人工智能（AI）时，一个主要挑战是确保隐私，即从输入LLM的任何文本中移除个人身份信息（PII）。随着欧盟新立法的出台，这些挑战变得更加紧迫。在欧盟国家中，围绕LLM使用和隐私问题的不确定性可能成为创新速度和从研究到应用转化的主要障碍。在此，我们介绍了\textbf{Redakto}，一个可在将文本输入LLM或其他下游文本处理之前用于匿名化的工具。我们提供了最先进的PII编辑功能，同时也支持假名化使用。这些功能通过Redakto Web应用程序可轻松供最终用户使用，也面向开发者和研究人员开放。

    arXiv:2608.18260v1 Announce Type: new  Abstract: Large Language Models (LLMs) are being increasingly used in everyday applications. A major challenge in the context of LLMs or Artificial Intelligence (AI) in general is to ensure privacy when using them, meaning that personally identifiable information (PII) is removed from any text that enters an LLM. These challenges have become more urgent with novel EU legislation. Uncertainty around LLM usage with respect to privacy concerns in EU countries can be a major blocker for the speed of innovation and transfer from research to applications. Here we present \textbf{Redakto}, a tool that can be used for anonymizing text prior to feeding it to an LLM or other downstream text processing. We provide state-of-the-art functionalities for both redaction of PII but also when used for pseudonymization. These functionalities are exposed such that they can easily be used by end-users, through the Redakto web application, and by developers and researc
    
[^135]: 视觉提示引导的野生动物个体识别

    Visual-Prompt Guided Wildlife Instance-Level Recognition

    [https://arxiv.org/abs/2608.18246](https://arxiv.org/abs/2608.18246)

    提出了一种单阶段端到端野生动物个体识别模型，利用视觉提示和潜在空间查询，在保持竞争力的同时简化了传统两阶段流水线。

    

    arXiv:2608.18246v1 公告类型：交叉 摘要：细粒度野生动物再识别仍是一个具有挑战性的研究领域。当前最先进的方法采用检测和再识别流水线。我们提出了一种单阶段端到端检测与再识别模型，在潜在空间内执行身份搜索。我们采用DINOv2进行稳健的空间几何建模，并使用MegaDescriptor进行野生动物再识别。我们通过提示再识别特征增强了潜在查询。检测解码器查询场景潜在空间以建立目标身份周围的物体边界。初步结果显示，与最先进的两阶段方法的44.89%相比，我们的平均精度均值得分为30.584%，具有竞争力。定性结果展示了有效的动物身份边界框定和识别。

    arXiv:2608.18246v1 Announce Type: cross  Abstract: Fine-grained wildlife re-identification remains a challenging area in research. Current state-of-the-art approaches apply a detection and re-identification pipeline. We propose a one-stage end-to-end detection and re-identification model that performs identity searching within the latent space. We adopt DINOv2 for robust spatial geometry and MegaDescriptor for wildlife re-identification. We enhance latent queries with prompt re-identification features. A detection decoder queries the scene latent space to establish object boundaries around the target identity. Preliminary findings reflect a competitive mean average precision score of 30.584% compared to the state-of-the-art two stage approach of 44.89%. Qualitative results depict effective bounding and identification of animal identities.
    
[^136]: 生物与人工神经网络之间的双向表征对齐

    Bidirectional representational alignment between biological and artificial neural networks

    [https://arxiv.org/abs/2608.18244](https://arxiv.org/abs/2608.18244)

    本文提出一种结合谱正则化的计算框架，通过引导表征几何在训练中实现双向对齐，在自监督视觉模型上将双向预测性相对提升55%。

    

    近期研究表明，生物神经网络与人工神经网络之间的表征对齐是不对称的：模型表征预测神经反应的效果远好于神经反应预测模型表征的效果。这种不对称性引发了关于表征几何是否有助于双向表征对齐的问题。我们假设，在训练过程中引导表征几何可以系统性地影响双向对齐。为了验证这一假设，我们开发了一个将谱正则化与双向预测性分析相结合的计算框架。作为初步演示，我们使用自监督对比视觉模型评估了该框架。引导学习表征的谱几何显著提高了反向预测性，同时仅轻微降低了正向预测性，使得双向预测性相对提升了55%。

    arXiv:2608.18244v1 Announce Type: cross  Abstract: Recent work has shown that representational alignment between biological and artificial neural networks is asymmetric: model representations predict neural responses much better than neural responses predict model representations. This asymmetry raises the question of whether representational geometry contributes to bidirectional representational alignment. We hypothesized that steering representational geometry during training can systematically influence bidirectional alignment. To test this hypothesis, we developed a computational framework that integrates spectral regularization with bidirectional predictivity analyses. As an initial demonstration, we evaluated our framework using self-supervised contrastive vision models. Steering the spectral geometry of the learned representations substantially increased reverse predictivity with modest reductions in forward predictivity, yielding a 55% relative improvement in bidirectional pred
    
[^137]: 基于密码子共现网络的SARS-CoV-2变异检测图表示范式GenEx

    GenEx: A Graph-Based Representational Paradigm for SARS-CoV-2 Variant Detection via Codon Co-occurrence Networks

    [https://arxiv.org/abs/2608.18238](https://arxiv.org/abs/2608.18238)

    GenEx通过将病毒基因序列转换为密码子共现图并提取图特征，提供了一种新的表示范式，能更有效地检测SARS-CoV-2变异，超越了传统线性序列分析方法。

    

    arXiv:2608.18238v1 公告类型：新 摘要：对SARS-CoV-2变异株（如Beta、Gamma、Delta和Omicron）的基因组分析主要依赖经典生物信息学方法，包括序列比对、系统发育分析和突变频率统计。这些方法使用成对密码子或核苷酸距离矩阵来分析基因序列，将其视为线性字符串，而未捕捉其复杂的上下文相互依赖性。我们提出了GenEx，一个将原始基因序列转换为密码子共现图并提取超过25个图特征的流程。我们用于图生成和特征提取的两个最突出技术是MSCG（多尺度密码子共现图）和LAPCG（线性时间邻接PMI密码子图）。利用这些算法，我们将密码子序列视为结构化的符号词汇，可解释为密码子共现图分析，这是一种借鉴自计算语言学的表示范式。

    arXiv:2608.18238v1 Announce Type: new  Abstract: Genomic analysis on viruses such as SARS-CoV-2 variants: Beta, Gamma, Delta, and Omicron is heavily dominated by classical bioinformatics methods, including Sequence Alignment, Phylogenetic Analysis, and Mutation Frequency Statistics. These approaches use pairwise codon or nucleotide distance matrices to analyze gene sequences, treating them as linear strings rather than capturing their complex contextual interdependencies. We proposed GenEx, a pipeline that converts raw gene sequences into codon co-occurrence graphs and extracts more than 25 graph features. Our two most prominent techniques for graph generation and feature extraction are MSCG (Multi-Scale Codon Co-occurrence Graph) and LAPCG (Linear-time Adjacency PMI Codon Graph). Using these algorithms, we treated codon sequences as structured symbolic vocabularies interpretable to codon co-occurrence graph analysis, a representational paradigm borrowed from computational linguistics.
    
[^138]: GigaBrain-WBC-0.5：一种用于与环境交互的鲁棒全身控制的行为世界模型

    GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction

    [https://arxiv.org/abs/2608.18234](https://arxiv.org/abs/2608.18234)

    本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。

    

    arXiv:2608.18234v1 公告类型：交叉 摘要：全身运动跟踪策略将人形机器人转化为一个鲁棒的控制接口：遥操作员——或上游模型——仅提供粗略的运动意图，而低级策略保持机器人平衡和物理可行性。现有的跟踪器仅在平坦地面上提供此接口：在空场景中训练，它们从未学习地形和物体接触如何重塑其动力学，并且它们试图通过不断扩充参考运动语料库来教会策略在任何命令下保持平衡，这在一旦可行行为变得依赖环境时就失效了。我们提出了GigaBrain-WBC-0.5，这是首个用于人形机器人全身控制的行为世界模型（BWM）。与纯粹的反应式跟踪器不同，我们训练了一个因果Transformer来联合预测其下一个动作、下一个状态以及下一个潜在行为命令的分布，因此，行动的网络也建模了环境如何塑造行为。

    arXiv:2608.18234v1 Announce Type: cross  Abstract: Whole-body motion tracking policies turn a humanoid into a robust control interface: the teleoperator---or an upstream model---only supplies a coarse movement intent, while the low-level policy keeps the robot balanced and physically feasible. Existing trackers deliver this interface only on flat ground: trained in empty scenes, they never learn how contact with terrain and objects reshapes their dynamics, and they attempt to teach the policy to balance under any command by continually enlarging the reference-motion corpus, which stops working once feasible behaviors become environment-dependent. We present GigaBrain-WBC-0.5, the first Behavior World Model (BWM) for humanoid whole-body control. Rather than a purely reactive tracker, we train a causal Transformer to jointly predict its next action, next state, and the distribution over its next latent behavior command, so the network that acts also models how the environment shapes what
    
[^139]: 任意格中Jaccard距离的三角不等式研究

    On the Triangle Inequality for the Jaccard Distance in Arbitrary Lattices

    [https://arxiv.org/abs/2608.18194](https://arxiv.org/abs/2608.18194)

    本文证明了在任意格中，当赋值为严格正、单调且模性时，Jaccard距离满足三角不等式，并进一步推广到相对补分配格，同时明确了超模性作为必要条件的限制。

    

    本文针对格与实值赋值上Jaccard距离的推广提出了新的理论结果。我们证明，当赋值为严格正、单调且模性时，Jaccard距离在任意格上满足三角不等式，从而有效推广了先前高度依赖分配性的结果。对于相对补分配格（该结构可安全地去掉布尔代数中全局界限的要求），我们证明只要赋值为正、单调、超模且对数子模，三角不等式即可成立。此外，我们将子模赋值的对称差Jaccard形式适配到部分补分配格。转向必要条件，我们证明超模性是标准广义Jaccard距离作为有效度量的严格要求。最后，我们将这些结果映射到...

    arXiv:2608.18194v1 Announce Type: new  Abstract: This paper presents new theoretical results on generalizing the Jaccard distance for lattices and real valuations. We demonstrate that when the valuation is strictly positive, monotone, and modular, the Jaccard distance satisfies the triangle inequality on arbitrary lattices, effectively generalizing earlier results that depended heavily on distributivity. Moving to relatively complemented distributive lattices (which safely drop the requirement for the global bounds found in Boolean algebras), we prove the triangle inequality holds as long as the valuation is positive, monotone, supermodular, and $\log$-submodular. Additionally, we adapt the symmetric-difference Jaccard formulation for submodular valuations to sectionally complemented distributive lattices. Shifting to necessary conditions, we prove that supermodularity is a strict requirement for the standard generalized Jaccard distance to operate as a valid metric. Finally, we map th
    
[^140]: 面向临床域偏移下多器官CT分割的边界感知逐器官召回风险控制

    Bound-Aware Per-Organ Recall Risk Control for Multi-Organ CT Segmentation under Clinical Domain Shift

    [https://arxiv.org/abs/2608.18193](https://arxiv.org/abs/2608.18193)

    本文提出一种边界感知的逐器官召回风险控制方法，通过WSR下注界在临床域偏移下以更少本地病例实现可靠重新认证，优于传统方法。

    

    无分布风险控制为冻结的分割模型增加了器官特异性的召回保证。我们为在AMOS数据集上训练的nnU-Net校准逐器官阈值，审计其向RAOS数据集的迁移，并使用病例级体素假阴性率（FNR）估计局部重新认证成本。AMOS控制通过，但迁移后$7/12$个器官超过$\alpha{=}0.10$；较小的校准集可能因保守或空洞的阈值掩盖超标情况。风险控制预测集（RCPS）提供对总体平均风险的高概率控制，而共形风险控制（CRC）提供较弱的期望控制。两者均要求可交换性；固定和全局阈值不提供逐器官保证。Waudby--Smith--Ramdas（WSR）下注界使用25个本地病例对六个一级器官进行重新认证，而Hoeffding--Bentkus（HB）需要30--40个。CRC需要10--15个但具有较重的个体病例尾部。没有二级器官满足我们示例性的精确度标准。

    arXiv:2608.18193v1 Announce Type: cross  Abstract: Distribution-free risk control adds organ-specific recall guarantees to frozen segmentation. We calibrate per-organ thresholds for an AMOS-trained nnU-Net, audit transfer to RAOS, and estimate local re-certification cost using case-level voxel false-negative rate (FNR). The AMOS control passes, but $7/12$ organs exceed $\alpha{=}0.10$ after transfer; smaller calibration sets can mask exceedances with conservative or vacuous thresholds. Risk-Controlling Prediction Sets (RCPS) give high-probability control of population-mean risk, whereas Conformal Risk Control (CRC) gives weaker expectation control. Both require exchangeability; fixed and global thresholds give no per-organ guarantee. The Waudby--Smith--Ramdas (WSR) betting bound re-certifies six Tier-1 organs with 25 local cases, versus 30--40 for Hoeffding--Bentkus (HB). CRC needs 10--15 but has a heavier individual-case tail. No Tier-2 organ meets our illustrative precision criterion
    
[^141]: 机器学习技术在自闭症诊断和治疗中的应用系统综述：挑战与机遇

    A systematic review of machine learning techniques to address diagnosis and treatment of autism: challenges and opportunities

    [https://arxiv.org/abs/2608.18188](https://arxiv.org/abs/2608.18188)

    本综述系统评估了2017-2023年间55项机器学习在自闭症诊断和治疗中的应用研究，指出监督学习为主流但深度学习正在崛起，并强调了模型可解释性和泛化能力等关键挑战。

    

    自闭症谱系障碍（ASD）是一种以社交互动和沟通困难为特征的发育障碍。由于ASD的病因尚不清楚，识别相关特征和隐藏相关性对于早期诊断至关重要。本系统综述评估了2017年至2023年间关于机器学习（ML）技术在ASD中应用的55项研究。主要目标是考察ML在ASD研究中的最新应用，识别增强诊断和治疗的趋势、技术和数据集。监督学习方法占主导地位，因为它们与ASD诊断需求高度契合；然而，随着数据可用性的增加，深度学习的作用正在扩大。基于混合方法的新兴技术，其中可能包括无监督学习、深度学习和模糊逻辑，未来将值得关注。综述强调了关键挑战和机遇，特别是模型可解释性和泛化能力的需求。

    arXiv:2608.18188v1 Announce Type: cross  Abstract: Autism spectrum disorder (ASD) is a developmental disability characterized by challenges in social interaction and communication. As the causes of ASD remain unclear, identifying relevant features and hidden correlations is crucial for early diagnosis. This systematic review evaluates 55 studies from 2017 to 2023 on the application of machine learning (ML) techniques to ASD. The primary objective is to examine recent ML applications in ASD research, identifying trends, techniques, and datasets that enhance diagnosis and treatment. Supervised learning methods dominate, as they align well with ASD diagnostic needs; however, the role of deep learning is expanding with greater data availability. Emerging techniques based on hybrid methods, where unsupervised, deep learning, and fuzzy logic could be included, will be interesting to observe in the future. The review highlights key challenges and opportunities, particularly the need for model
    
[^142]: 人工智能能从医学中学到什么？生成类比与可靠的机器学习系统

    What Can Artificial Intelligence Learn from Medicine? Generative Analogies and Reliable Machine Learning Systems

    [https://arxiv.org/abs/2608.18186](https://arxiv.org/abs/2608.18186)

    本文通过生成类比将医学临床转化标准映射到机器学习系统构建中，提出以可靠主义视角确立ML的认识论和方法论依据。

    

    在过去几年中，机器学习（ML）在医学领域得到了广泛应用，并在一定程度上取得了成功。然而，围绕ML的不确定性使其认识论和方法论依据难以确立。文献中，有人将医学与ML进行类比，建议我们以临床转化的标准为模型，来确立ML的认识论和方法论标准。通过发展Hesse工作中的工具，我们将这种类比的本质特征化为临床转化过程与构建ML系统过程之间的生成类比。我们更精确地识别了临床转化中通常仅在该类比被提及时才提及的认识论和方法论依据，并展示了这些依据在何种意义上类比适用于ML的语境。特别是，我们以可靠主义术语解释了临床转化的依据。

    arXiv:2608.18186v1 Announce Type: cross  Abstract: In the past few years, machine learning (ML) has been widely (and to an extent, successfully) implemented in medicine. However, uncertainties surrounding ML have made it difficult to establish the bases of its epistemic and methodological warrants. In the literature, a parallel has been drawn between medicine and ML, suggesting that we should model epistemic and methodological standards for ML on the standards of clinical translation. By developing tools from Hesse work, we characterise the nature of this parallel as a generative analogy between the process of clinical translation and the process of building ML systems. We identify more precisely the epistemic and methodological warrants of clinical translation that are typically only mentioned when appealing to the analogy, and we show in which sense such warrants apply analogically to the context of ML. In particular, we interpret warrants of clinical translation in reliabilist terms
    
[^143]: 循环语言模型提升组合式工具调用能力

    Looped Language Models Improve Compositional Tool Calling

    [https://arxiv.org/abs/2608.18171](https://arxiv.org/abs/2608.18171)

    循环语言模型在组合式工具调用中通过增加循环深度和自适应推理，显著提升多步API调用的准确性和依赖处理能力。

    

    arXiv:2608.18171v1 公告类型：新 摘要：循环语言模型在推理基准测试中展现出有前景的结果，但其在代理式工具使用方面的潜力尚未得到充分探索。我们在组合式工具调用设置中研究这一问题，其中模型必须协调多个API调用、维护中间状态，并在工具交互之间保持依赖关系。我们在API-Bank、BFCL和NESTful上评估了原生和改造的循环语言模型，比较了在匹配的监督微调配方和推理时不同循环深度下训练的循环与非循环模型。在受控实验中，循环计算通常有益于组合式和依赖感知的工具使用，而在孤立API调用上仅提供较小且依赖模型的改进。多步工具使用的准确性通常随循环深度的增加而提高；然而，自适应推理通过分配额外计算，实现了更优的计算性能权衡。

    arXiv:2608.18171v1 Announce Type: new  Abstract: Looped language models have shown promising results on reasoning benchmarks, yet their potential for agentic tool use remains largely unexplored. We study this question in compositional tool-calling settings, where models must coordinate multiple API calls, maintain intermediate state, and preserve dependencies across tool interactions. We evaluate native and retrofitted looped language models on API-Bank, BFCL, and NESTful, comparing looped and non-looped models trained under matched supervised fine-tuning recipes and varying recurrent depth at inference time. In controlled experiments, recurrent computation generally benefits compositional and dependency-aware tool use, while providing smaller and more model-dependent gains on isolated API invocation. Accuracy on multi-step tool use generally increases with recurrent depth; adaptive inference, however, achieves a more favorable compute-performance trade-off by allocating additional com
    
[^144]: 对抗性审查：用于接地智能体代码审查的结构化分歧

    Adversarial Review: Structured Disagreement for Grounded Agentic Code Review

    [https://arxiv.org/abs/2608.18167](https://arxiv.org/abs/2608.18167)

    本文提出了一种名为对抗性审查（AR）的最小合作代码审查协议，通过引入批评者智能体进行结构化分歧审计，在仅使用三个智能体的情况下超越了五个智能体的基线性能，并揭示了朴素方法中的虚假共识问题。

    

    arXiv:2608.18167v1 公告类型：新 摘要：早期的多智能体LLM系统通常采用角色分离的团队，但在仓库级编码任务上，增加智能体数量会导致收益递减。最近的替代方案将智能体视为被动工具（子智能体），但这完全消除了智能体交互的好处。我们研究了子智能体范式是否能支持一种折中方案：在避免大型多智能体团队开销的同时，实现最小限度的智能体合作。我们引入了对抗性审查（AR），一种最小合作的代码审查协议，其中主编码智能体与一个审查者智能体和一个批评者智能体协作。审查者评估代码，而批评者通过结构化分歧审计审查结果，之后主智能体进行编辑。在LiveCodeBench上，AR在测试方法中实现了最高的通过率，仅使用三个智能体就超越了五个智能体的基线。在SWE-PRBench上，朴素的AR暴露了一种虚假共识失败模式，即智能体在没有充分依据的情况下达成一致。

    arXiv:2608.18167v1 Announce Type: new  Abstract: Early multi-agent LLM systems often used role-separated teams, yet scaling agent count yields diminishing returns on repository-level coding tasks. Recent alternatives treat agents as passive tools (subagents), yet this removes the benefits of agent interaction entirely. We study whether a subagent paradigm can support a middle ground: minimal agentic cooperation without the overhead of large multi-agent teams. We introduce Adversarial Review (AR), a minimal cooperative code-review protocol in which a main coding agent works with a reviewer and a critic agent. The reviewer evaluates code, while the critic audits the review through structured disagreement before the main agent edits. On LiveCodeBench, AR achieves the highest pass rate among tested methods, outperforming a five-agent baseline while using only three agents. On SWE-PRBench, naive AR exposes a false-consensus failure mode, where agents converge on agreement without sufficient
    
[^145]: RDFdL：将RDF与微分动态逻辑集成

    RDFdL: Integrating RDF with Differential Dynamic Logic

    [https://arxiv.org/abs/2608.18165](https://arxiv.org/abs/2608.18165)

    RDFdL框架首次将RDF与微分动态逻辑结合，实现了对静态知识和物理系统连续动态的统一表示与推理，并支持将动态逻辑验证结果转化为SPARQL查询蕴含。

    

    摘要：以RDF建模的知识图谱擅长描述静态知识，但无法捕捉或推理物理系统的动态行为，例如由微分方程描述的系统，这是AI驱动的信息物理系统中的一个关键缺口。为解决这一问题，我们提出了RDFdL，一个将RDF与微分动态逻辑（dL）集成的框架，用于表示和推理静态知识以及物理系统的连续动态。对于动态部分，我们在RDF和SHACL中语法化地表示微分方程和状态空间中的范围，并通过到dL的转换提供语义。通过它们在一阶逻辑中的共同基础来连接RDF和dL，实现了独特的集成：动态逻辑领域中安全性和可达性属性的验证结果可作为RDF数据上SPARQL查询的蕴含结果。我们使用Apache Jena实现了该管道，用于本体...

    arXiv:2608.18165v1 Announce Type: new  Abstract: Knowledge graphs modeled in RDF are powerful for describing static knowledge, but they cannot capture or reason about the dynamic behavior of physical systems, e.g., systems described by differential equations, which is a critical gap for AI-driven cyber-physical systems. To solve this, we propose RDFdL, a framework that integrates RDF with Differential Dynamic Logic (dL) to represent and reason about both static knowledge and the continuous dynamics of physical systems. For the dynamic part, we syntactically represent differential equations and ranges in the state space in RDF and SHACL and provide semantics using a translation to dL. Linking RDF and dL through their shared foundation in first-order logic achieves a unique integration: verification results for safety and reachability properties in the dynamic logic domain become available as entailment to SPARQL queries over RDF data. We implement the pipeline using Apache Jena for onto
    
[^146]: 大型语言模型在文本之外是否安全：表情符号是否暴露了安全评估中的漏洞

    Are LLMs Safe Beyond Text: Do Emojis Expose Gaps in Safety Evaluation

    [https://arxiv.org/abs/2608.18164](https://arxiv.org/abs/2608.18164)

    表情符号增强的提示揭示了LLM安全评估中的漏洞，不同模型对非文本输入的鲁棒性差异显著，表明仅依赖文本提示的评估可能不全面。

    

    大型语言模型（LLMs）的安全评估主要依赖于基于文本的对抗性提示，这可能会忽视由替代输入表示形式引发的漏洞。本研究以表情符号增强的提示作为测试案例，评估了四个开源LLM（Mistral 7B、Qwen 2 7B、Gemma 2 9B、Llama 3 8B）上的50个提示。结果显示鲁棒性存在显著差异：Gemma 2 9B和Mistral 7B表现出非零成功率（10%），Llama 3 8B为6%，而Qwen 2 7B表现出完全抵抗（成功率为0%）。卡方检验（χ² = 32.94，p < 0.001）确认了结果分布的显著差异。这些发现表明，鲁棒性对输入表示形式敏感，仅限于标准文本提示的评估可能低估了模型的漏洞。

    arXiv:2608.18164v1 Announce Type: cross  Abstract: Safety evaluations of large language models (LLMs) predominantly rely on text-based adversarial prompts, potentially overlooking vulnerabilities arising from alternative input representations. This work examines emoji-augmented prompts as a test case for this gap, evaluating 50 prompts across four open-source LLMs (Mistral 7B, Qwen 2 7B, Gemma 2 9B, Llama 3 8B). Results show substantial variation in robustness: Gemma 2 9B and Mistral 7B exhibit non-zero success rates (10%), Llama 3 8B 6%, while Qwen 2 7B shows complete resistance (0% success rate). A chi-square test ($\chi^2 = 32.94, p < 0.001$) confirms significant differences in outcome distributions. These findings indicate that robustness is sensitive to input representation, and that evaluations restricted to standard text prompts may underrepresent model vulnerabilities.
    
[^147]: 大型语言模型何时真正有用？评估LLM作为数据质量标注器

    When Do LLMs Actually Help? Evaluating LLMs as Data Quality Annotators

    [https://arxiv.org/abs/2608.18158](https://arxiv.org/abs/2608.18158)

    本研究揭示了LLM在数据质量标注任务中的实际效用取决于任务复杂度，在简单任务中与规则基线相当，而在复杂任务中更优，但小样本提示评估可能误导性能判断。

    

    大型语言模型（LLM）越来越多地被用于自动捕获数据质量问题，但我们对这些判断的实际一致性知之甚少。本研究在零样本和少样本提示下，针对两个电子商务数据质量任务（实体匹配和品牌错误标注），将LLM与基于规则的基线和人工验证的真实标签进行了测试。在实体匹配任务中，使用Abt Buy基准（2,194个标注对），一个简单的基于规则的基线（F1=0.950）与LLM零样本提示（F1=0.948）表现相当。此外，一个在小验证样本上看似有效的少样本提示修订，在全规模测试中性能降至F1=0.914，这表明小样本提示评估可能具有误导性。在品牌错误标注检测中，使用500个带有合成注入标签错误的亚马逊产品列表，LLM明显优于朴素的基于规则的基线（F1=0.833 vs 0.721），因为它能够利用上下文信息。

    arXiv:2608.18158v1 Announce Type: cross  Abstract: LLMs have been increasingly used to catch data quality issues automatically, but we know very little about how consistent these judgments actually are. This study tests an LLM on two e-commerce data quality tasks, entity matching and brand mislabeling, against rule based baselines and human verified ground truth, under both zero-shot and few-shot prompting. On entity matching while using the Abt Buy benchmark (2,194 labeled pairs), a simple rule based baseline (F1=0.950) performed about as well as LLM zero shot prompting (F1=0.948). Moreover, a few-shot prompt revision that looked effective on a small validation sample reduced full-scale performance to F1=0.914. This showed that small sample prompt evaluation can be misleading. On brand mislabeling detection, using 500 Amazon product listings with synthetically injected labeling errors, the LLM clearly outperformed a naive rule based baseline (F1=0.833 vs 0.721), because it could draw 
    
[^148]: 量子优势究竟有多大？面向网络入侵检测的量子机器学习公平、校准与噪声感知基准及归因审计

    How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection

    [https://arxiv.org/abs/2608.18155](https://arxiv.org/abs/2608.18155)

    该论文通过公平、校准和噪声感知的基准测试及量子归因审计，揭示了量子机器学习在网络入侵检测中的优势可能多为经典伪影而非真正的量子效应。

    

    量子机器学习（QML）在网络入侵检测（NIDS）中常被报道达到近乎完美的准确率，然而最严谨的研究发现，经过良好调优的经典模型仍具竞争力，且表面上的量子增益可能源于经典降维和隐式正则化的伪影，而非真正的量子效应。我们不问量子模型能否达到高准确率，而是追问量子优势究竟有多大。我们提出了一个统一、可复现的QML-IDS基准，在四个标准NIDS数据集（NSL-KDD、UNSW-NB15、CICIDS2017、NF-ToN-IoT-v2）上，将混合变分量子电路和量子核SVM与五个诚实调优的经典基线模型进行比较，采用单一泄漏控制协议、等预算特征视图、不平衡与校准感知指标及显著性检验，并模拟NISQ噪声扫描。我们引入了一个量子归因审计（参数-

    arXiv:2608.18155v1 Announce Type: cross  Abstract: Quantum machine learning (QML) for network intrusion detection (NIDS) is routinely reported to reach near-perfect accuracy, yet the most rigorous studies find that well-tuned classical models remain competitive, and that apparent quantum gains may be artefacts of classical dimensionality reduction and implicit regularisation rather than genuine quantum effects. We ask not whether a quantum model can post a high accuracy, but how quantum the advantage really is. We present a unified, reproducible QML-IDS benchmark evaluating hybrid variational quantum circuits and quantum-kernel SVMs against five honestly-tuned classical baselines across four standard NIDS datasets (NSL-KDD, UNSW-NB15, CICIDS2017, NF-ToN-IoT-v2) under one leakage-controlled protocol, with an equal-budget feature view, imbalance- and calibration-aware metrics with significance testing, and a simulated NISQ noise sweep. We introduce a quantum-attribution audit (parameter-
    
[^149]: TokenPowerSandbox：基于证据门控的CPU优先筛选，用于能耗感知的LLM服务

    TokenPowerSandbox: Evidence-Gated CPU-First Screening for Energy-Aware LLM Serving

    [https://arxiv.org/abs/2608.18149](https://arxiv.org/abs/2608.18149)

    本文提出一种证据门控的CPU优先筛选工作流，通过可解释投影器和短时GPU探针结合，在能耗预测中实现高精度（MAPE低至6.23%）和强排序相关性，同时利用TTFT门控防止低并发下的不可靠预测。

    

    能耗感知的大语言模型服务需要在真实请求形态下比较不同配置，然而对目标GPU进行穷举性能分析成本高昂，而廉价预测器在其测量范围之外可能过于自信。我们提出了TokenPowerSandbox，一种证据门控工作流，结合了可解释的CPU驻留投影器、短时目标GPU探针、全工作负载验证以及防篡改的冻结前测量溯源。在一台搭载vLLM的NVIDIA H100 80GB上服务Qwen2.5-7B-Instruct时，三次锚定重复和六个开发工作负载校准了工作负载迁移。同一冻结模型在盲留出集和单独预先声明的不重新拟合确认集上评估，共计51次冻结后运行。能量平均绝对百分比误差（MAPE）分别为6.23%和7.35%，Spearman秩相关系数分别为0.976和0.933。然而，预先声明的首令牌时间（TTFT）门控在并发数为四时通过（MAPE为9.27%），在低于四时触发弃权（MAPE为64.80%），这展示了为何能量准确性可能无法保证延迟性能。

    arXiv:2608.18149v1 Announce Type: cross  Abstract: Energy-aware LLM serving requires comparing configurations under realistic request shapes, yet exhaustive target-GPU profiling is costly and a cheap predictor can be dangerously confident outside its measured scope. We present TokenPowerSandbox, an evidence-gated workflow that combines an interpretable CPU-resident projector, short target-GPU probes, full-workload verification, and tamper-evident freeze-before-measurement provenance. On one NVIDIA H100 80GB serving Qwen2.5-7B-Instruct with vLLM, three anchor repeats and six development workloads calibrate workload transfer. The same frozen model is evaluated on a blind holdout and a separately predeclared no-refit confirmation totaling 51 post-freeze runs. Energy MAPE is 6.23% and 7.35%, with Spearman rank correlations of 0.976 and 0.933. However, a predeclared TTFT gate passes at concurrency four (9.27% MAPE) and triggers abstention below four (64.80%), showing why energy accuracy can
    
[^150]: 熵约束的自适应随机量化

    Entropy-Constrained Adaptive Stochastic Quantization

    [https://arxiv.org/abs/2608.18147](https://arxiv.org/abs/2608.18147)

    本文提出了一种熵约束的自适应随机量化方法，通过联合优化量化值以在熵预算下最小化均方误差，并提供了高效的最优动态规划解决方案。

    

    自适应随机量化（ASQ）是一种近期引入的量化方法，它在给定输入下优化均方误差（MSE）的同时保持无偏性。该方法旨在缓解现代数据和机器学习工作负载中的通信和内存瓶颈，包括模型压缩、梯度压缩、KV缓存压缩以及最近邻搜索。此外，实际系统可以通过无损熵编码器进一步压缩量化数据。然而，现有的无偏方法（包括ASQ）在选择量化值时未考虑后续的编码阶段，从而损失了精度。我们提出了熵约束的自适应随机量化（ECASQ）问题，该问题在熵预算和无偏性约束下，联合选择自适应量化值以最小化MSE。我们给出了一种最优动态规划算法，其时间复杂度为$O(sd^2)$，空间复杂度为$O(d^2)$，适用于长度为d的向量，且最多支持s个量化级别。

    arXiv:2608.18147v1 Announce Type: cross  Abstract: Adaptive stochastic quantization (ASQ) is a recently introduced quantization approach that optimizes the Mean Squared Error (MSE) for a given input while preserving unbiasedness. It is designed to alleviate the communication and memory bottlenecks of modern data and machine learning workloads, including model, gradient, and KV-cache compression and nearest-neighbor search. Further, practical systems can then compress quantized data with a lossless entropy encoder. However, existing unbiased methods, including ASQ, choose their quantization values without considering this later encoding stage, leaving accuracy on the table.   We formulate the Entropy Constrained Adaptive Stochastic Quantization (ECASQ) problem, which jointly selects adaptive quantization values to minimize MSE under an entropy budget and an unbiasedness constraint. We give an optimal dynamic program with $O(sd^2)$ time and $O(d^2)$ space for a length-d vector and at mos
    
[^151]: 道义鸿沟：大语言模型与义务的情态语言

    The Deontic Gap: Large Language Models and the Modal Language of Obligation

    [https://arxiv.org/abs/2608.18144](https://arxiv.org/abs/2608.18144)

    本论文发现大语言模型在生成文本中系统性地少用积极道义情态词（如“必须”、“应该”），与当代人类使用模式存在显著差距，但接近正式出版英语的频率。

    

    情态助动词如“必须”、“应该”和“不得不”在说话者权威和人际立场的语境中标记必要性和义务。我们考察了大语言模型（LLMs）是否再现了当代人类道义情态使用的模式。在三个主要语料库、一个外部基准、两个受控复制以及一个自然主义的十一模型复制中，AI生成的文本相对于当代人类始终少用积极道义情态词（必须、应该、不得不、不得不）。与谷歌图书Ngram语料库（1920-2022年）的历史比较（作为对出版散文记录的启发式校准）表明，AI情态频率落在正式出版英语的范围内，而当代人类在非正式数字语境中的情态频率通常超过二十世纪书籍基线。短语级分解显示，AI与人类之间的情态差距集中在以……为中心的结构中。

    arXiv:2608.18144v1 Announce Type: cross  Abstract: Modal auxiliaries such as must, should, and have to mark necessity and obligation within the contexts of speaker authority and interpersonal stance. We examine whether large language models (LLMs) reproduce contemporary human patterns of deontic modal usage. Across three primary corpora, an external benchmark, two controlled replications, and a naturalistic eleven-model replication, AI-generated text consistently underuses positive deontic modals (must, should, have to, had to) relative to contemporary humans. Historical comparison with the Google Books Ngram corpus (1920-2022), used as a heuristic calibration against the published-prose record, shows that AI modal frequencies fall within the range of formal published English, whereas contemporary human modal rates in informal digital contexts often exceed twentieth-century book baselines. Phrase-level decomposition shows that the AI-human modal gap is concentrated in constructions cen
    
[^152]: 针对低资源语言仇恨言论检测的LLM高效适配：罗马乌尔都语的比较研究

    Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu

    [https://arxiv.org/abs/2608.18142](https://arxiv.org/abs/2608.18142)

    本研究通过LoRA参数高效微调方法，系统比较了多种大型语言模型在罗马乌尔都语低资源环境下的仇恨言论检测性能，展示了PEFT在零样本推理中的优势。

    

    由于注释数据缺乏、语言结构非正式以及标准化语法缺失，在低资源语言中检测仇恨言论颇具挑战性。罗马乌尔都语就是此类挑战的一个典型例子，它在南亚社交媒体上广泛使用，拼写变异大且缺乏上下文一致的规范。本文旨在全面评估大型语言模型在罗马乌尔都语脚本中的仇恨言论检测性能，并采用参数高效微调方法——低秩适配（LoRA）对这些模型进行微调。为评估零样本推理，我们在不同变压器模型（包括Mistral、LLaMA、Falcon和多语言BERT）上将其与PEFT进行基准对比。实验在包含超过72,000条注释的PURUTT数据集（乌尔都语和罗马乌尔都语有毒评论及音译平行语料库）上进行。

    arXiv:2608.18142v1 Announce Type: new  Abstract: It is challenging to detect hate speech in Low Resource Languages (LRLs) because of the absence of annotated data, the informality of its language structure, and the lack of standardized grammar. A good example of such a challenge is Roman Urdu which is broadly used by South Asians on social media and has a high variation while lacking contextually consistent spellings. The objective of this paper is to conduct a comprehensive assessment of Large Language Models (LLMs) for Hate Speech Detection (HSD) in Roman Urdu script and fine-tune these models using the Parameter-Efficient Fine-Tuning (PEFT) method called Low-Rank Adaptation (LoRA). To evaluate zero-shot inference, we benchmarked it against PEFT on different transformer models, including Mistral, LLaMA, Falcon, and multilingual BERT. Experiments are conducted on the PURUTT (Parallel Urdu and Roman Urdu Corpus for Toxic Comments and Transliteration) dataset with over 72,000 annotated 
    
[^153]: 葡萄牙语语言模型：一项系统性映射研究

    Language Models for Portuguese: A Systematic Mapping Study

    [https://arxiv.org/abs/2608.18138](https://arxiv.org/abs/2608.18138)

    本文对葡萄牙语语言模型进行了系统性映射研究，梳理了46个模型的现状，填补了该领域信息分散的空白。

    

    arXiv:2608.18138v1 公告类型：交叉 摘要：近年来，语言模型的快速发展通过广泛的应用彻底改变了自然语言处理领域。然而，语言模型的发展在所有语言中并不均衡。就葡萄牙语而言，近年来学术界和公司日益努力开发语言模型并为葡萄牙语创建数据资源。这些努力导致了葡萄牙语语言模型生态系统日益多样化。然而，关于这些模型的信息仍分散在科学出版物、技术报告、模型库和项目文档中。本调查对为葡萄牙语开发的语言模型进行了系统性映射研究，提供了该领域当前状态的全面概述。我们共映射了46个模型，从基础模型、架构等多个方面对其进行特征化描述。

    arXiv:2608.18138v1 Announce Type: cross  Abstract: In recent years, the rapid development of language models has transformed the field of Natural Language Processing through a wide range of applications. However, the development of language models has not progressed uniformly across all languages. In the case of the Portuguese language, there has recently been a growing effort by academia and companies to develop language models and create data resources for Portuguese. These efforts have resulted in the rise of an increasingly diverse ecosystem of language models for Portuguese. However, information on these models remains dispersed in scientific publications, technical reports, model repositories, and project documentation. This survey presents a systematic mapping study of language models developed for Portuguese, providing a comprehensive overview of the current state of the field. We map a total of 46 models, characterizing them by various aspects, including base model, architectu
    
[^154]: FraudBench：对基于策略的银行智能体进行对抗性欺诈压力测试

    FraudBench: Stress-Testing Policy-Grounded Banking Agents Against Adaptive Fraud

    [https://arxiv.org/abs/2608.18136](https://arxiv.org/abs/2608.18136)

    本文提出了FraudBench，一个首个针对银行对话智能体在自适应欺诈操纵下安全性的可执行基准，填补了现有基准在身份和信任操纵方面的空白。

    

    arXiv:2608.18136v1 公告类型：新 摘要：对话智能体现在通过工具代表终端用户操作，同时持有对客户数据库和内部政策文档的访问权限，而呼叫者仅通过对话即可访问这些内容。银行业是最明显的案例：同一个智能体既能回答问题，也能更改联系方式、重置PIN码或转账，因此普通客户服务与授权、欺诈检测和政策合规密不可分。现有的金融欺诈基准对静态交易或消息进行分类，而通用智能体安全基准则针对提示注入或一般性有害使用；这些都没有测试基于策略的银行智能体在呼叫者通过对话操纵身份、授权和信任时能否安全操作。我们引入了FraudBench，这是一个基于$\tau^2$-bench双控框架和$\tau$-Knowledge银行环境构建的可执行基准。智能体和模拟呼叫者都通过共享、可变的账户上的工具进行操作。

    arXiv:2608.18136v1 Announce Type: new  Abstract: Conversational agents now act for end users through tools while holding access to customer databases and internal policy documents that a caller can reach through dialogue alone. Banking is the clearest case: the same agent that answers a question can also change contact details, reset a PIN, or move money, so ordinary customer service is inseparable from authorization, fraud detection, and policy compliance. Existing financial-fraud benchmarks classify static transactions or messages, and general agent-safety benchmarks target prompt injection or generic harmful use; none test whether a policy-grounded banking agent safely acts when a caller manipulates identity, authorization, and trust over a conversation. We introduce FraudBench, an executable benchmark built on the $\tau^2$-bench dual-control framework and the $\tau$-Knowledge banking environment. Both the agent and the simulated caller act through tools over shared, mutable account
    
[^155]: 利用人工智能改善农村用药安全：一项范围综述

    Improving Rural Medication Safety with AI: A Scoping Review

    [https://arxiv.org/abs/2608.18135](https://arxiv.org/abs/2608.18135)

    这篇综述总结了AI技术在农村医疗用药管理全过程中的应用，表明其能有效减少用药错误并提升患者安全，但现有研究覆盖国家有限，需进一步扩展。

    

    arXiv:2608.18135v1 公告类型：新 摘要：引言：用药错误（MEs）对全球医疗系统构成重大威胁，导致患者伤害。在农村医疗中引入人工智能（AI）可增强患者安全。本研究旨在探讨AI技术在改善农村医疗环境中患者安全和减少用药错误方面的应用及有效性。方法：通过系统文献检索，涵盖2012年至2025年，检索多个数据库，包括EBSCohost、Emcare（Ovid）、MEDLINE和ProQuest消费者健康数据库。共审查了来自九个国家的十二项主要研究。对数据进行主题分析，以获取关于AI干预措施在用药过程中的见解。结果：AI技术已被整合到用药管理的每个阶段，从开处方、配药到给药及给药后监测。四个关键主题（略）。

    arXiv:2608.18135v1 Announce Type: new  Abstract: Introduction: Medication errors (MEs) represent a significant threat to global healthcare systems, contributing to patient harm. Introducing artificial intelligence (AI) in rural healthcare enhances patient safety. The aim is to explore the applications and effectiveness of AI technologies in enhancing patient safety and reducing medication errors in rural health settings.   Methods: A scoping review was conducted through a systematic literature search spanning 2012 to 2025 across multiple databases, including EBSCohost, Emcare (Ovid), MEDLINE, and the ProQuest Consumer Health Database. Twelve primary studies from nine different nations were examined. Data were analysed thematically to obtain insights on AI interventions across the medication process.   Results: AI technologies have been integrated into every stage of medication management, right from prescribing and dispensing to administration and post-administration monitoring. Four k
    
[^156]: 基于IEEE特征气体法的优化模糊逻辑方法用于溶解气体分析诊断电力变压器故障

    Optimized Fuzzy Logic Approach with the IEEE Key Gas Method for Diagnosing Power Transformer Faults Using Dissolved Gas Analysis

    [https://arxiv.org/abs/2608.18133](https://arxiv.org/abs/2608.18133)

    本文提出了一种结合模糊逻辑与IEEE特征气体法的优化模型（FL-KGM），通过精炼隶属函数、优化模糊规则和分离CO/CO2，实现了高达98.6%的变压器故障诊断准确率，显著优于传统方法。

    

    arXiv:2608.18133v1 公告类型：新 摘要：可靠的变压器故障诊断对于维持电力系统稳定性至关重要。IEEE特征气体法（KGM）是溶解气体分析（DGA）中广泛使用的方法，但在处理模糊数据和确保高诊断精度方面存在局限性。本研究提出了一种结合模糊逻辑与IEEE特征气体法的增强模型（FL-KGM），该模型引入了精炼的隶属函数、优化的模糊规则集以及CO和CO2的新颖分离，以消除诊断不一致性。通过利用多维气体比率分析和自适应分类框架，FL-KGM提供了优越的故障识别和分类能力。利用真实世界数据集进行的实验验证表明，FL-KGM实现了高达98.6%的准确率，显著优于KGM和其他基于模糊逻辑的方法。这些发现阐明了FL-KGM在推进变压器监测方面的潜力，使其能够实现智能化和更可靠的故障诊断。

    arXiv:2608.18133v1 Announce Type: new  Abstract: Reliable transformer fault diagnosis is essential for maintaining power system stability. The IEEE Key Gas Method (KGM), a widely utilized approach in Dissolved Gas Analysis (DGA), exhibits limitations in addressing ambiguous data and ensuring high diagnostic accuracy. This study presents An enhanced model combining Fuzzy Logic with the IEEE Key Gas Method (FL-KGM) that introduces refined membership functions, optimized fuzzy rule sets, and a novel separation of CO and CO2 to eliminate diagnostic inconsistencies. By leveraging multidimensional gas ratio analysis and an adaptive classification framework, FL-KGM delivers superior fault identification and classification. Experimental validation utilizing real-world datasets demonstrates that FL-KGM achieves up to 98.6% accuracy, significantly outperforming KGM and other FL-based approaches. These findings elucidate the potential of FL-KGM in advancing transformer monitoring, enabling intell
    
[^157]: 安全对齐错觉：大型语言模型中的跨语言安全差距

    Safety Alignment Illusion: The Cross-Lingual Safety Gap in LLMs

    [https://arxiv.org/abs/2608.18131](https://arxiv.org/abs/2608.18131)

    本文揭示了大型语言模型在非英语语言中安全对齐失效的问题，并提出了INCLUDE基准来量化印度中心的社会文化偏见，以解决跨语言安全差距。

    

    摘要：当前大型语言模型（LLMs）的安全对齐训练严重以英语为中心。当这些安全过滤器对非英语语言失效时，后果会立即显现并直接影响用户：语音助手和口语对话系统可能产生强化刻板印象的输出，绕过以英语为重点的标准安全对齐，并将有害偏见传播给非英语社区。对于部署在印度语言多样化人群中的口语技术，这代表了一个关键的失败模式。为了解决这一跨语言差距，我们引入了INCLUDE（用于理解和检测嵌入式偏见的印度文化视角），这是一个多语言评估基准，旨在量化以印度为中心的社会文化偏见。INCLUDE包含2，604个提示，涵盖六种提示语言：英语、印地语、孟加拉语、马拉地语、泰米尔语和兴都语（印地语-英语混合语）。我们评估了十个开源和闭源模型。

    arXiv:2608.18131v1 Announce Type: new  Abstract: Current safety alignment training for Large Language Models (LLMs) are heavily English-centric. When such safety filters fail for non-English languages, the consequences are immediate and user-facing: voice assistants and spoken dialogue systems may produce stereotype-reinforcing outputs, bypassing the standard English-focused safety alignments and propagating harmful bias to non-English speaking communities. For spoken language technologies deployed across India's linguistically diverse population, this represents a critical failure mode. To address this cross-lingual gap, we introduce INCLUDE (Indian Cultural Lens for Understanding and Detecting Embedded Biases), a multilingual evaluation benchmark designed to quantify Indian-centric socio-cultural biases. INCLUDE consists of 2,604 prompts spanning six prompt languages: English, Hindi, Bengali, Marathi, Tamil, and Hinglish (Hindi-English code-mix). We evaluate ten open- and closed-sour
    
[^158]: 全球负责任人工智能指数2026：概念框架与方法论

    Global Index on Responsible AI 2026 : Conceptual Framework and Methodology

    [https://arxiv.org/abs/2608.18122](https://arxiv.org/abs/2608.18122)

    本文提出了GIRAI第二版的方法论，通过五个维度和38个指标评估全球负责任AI治理，并强化了框架存在与实施的区分。

    

    本报告介绍了全球负责任人工智能指数（GIRAI）第二版的方法论。该版本在第一版基础上进行了改进，通过强化框架存在性与实施之间的区别，将维度从三个重组为五个主题领域，引入更细化的框架质量变量，并应用多阶段审查和验证流程。进行了独立的统计预审计，以评估框架的一致性和稳健性。GIRAI评估了五个维度的负责任人工智能治理：包容性与多样性、伦理与可持续性、劳动力与技能、信任与安全，以及公共服务中的人工智能应用。每个维度包含若干指标（共38个），组织为三大支柱，即人工智能政策（17个指标，基于政府框架和实施的原始数据评估）、民间社会组织参与（5个指标，原始数据）和有利条件。

    arXiv:2608.18122v1 Announce Type: cross  Abstract: This report presents the methodology of the Global Index on Responsible AI (GIRAI), 2nd Edition. This edition refines the 1st Edition by strengthening the distinction between framework existence and implementation, restructuring dimensions from three to five thematic areas, introducing more granular variables for framework quality, and applying a multi-stage review and validation process. An independent statistical pre-audit was conducted to assess the coherence and robustness of the framework. GIRAI assesses responsible AI governance across five dimensions: Inclusion and Diversity, Ethics and Sustainability, Labour and Skills, Trust and Safety, and Use of AI in Public Service. Each dimension has a number of indicators (38 in total), organised into three pillars, namely AI Policy (17 indicators on government frameworks and implementation, assessed through primary data), CSO Engagement (5 indicators, primary data), and Enabling Conditio
    
[^159]: 立场：AI排行榜未能充分服务全球南方——来自印度的案例研究

    Position: AI Leaderboards Are Underserving the Global South: A Case Study from India

    [https://arxiv.org/abs/2608.18117](https://arxiv.org/abs/2608.18117)

    本文指出AI排行榜因缺乏独立治理和指标演化机制，结构性忽视了全球南方的高质量基准，并以印度为例揭示了这一制度性缺陷。

    

    arXiv:2608.18117v1 公告类型：新 摘要：本立场论文认为，AI排行榜在结构上不适合服务全球南方，因为它们缺乏独立治理、利益冲突政策以及指标演化机制。障碍并非数据缺失；高质量的区域基准已经存在：如印度的IndicSUPERB、MILU和LAHAJA；非洲的IrokoBench；阿拉伯语的AlGhafa。障碍在于制度设计。全球排行榜不包含这些基准，且没有治理机制强制它们这样做。当全球北方的付费客户受到影响时，商业压力会纠正排行榜的失败。而全球南方缺乏同等的杠杆。没有治理，影响印地语、斯瓦希里语或阿拉伯语使用者的失败会无限期存在，成为已记录但未解决的空白。以印度为案例（14亿人口，22种官方语言，高质量基准，但缺乏可信的聚合），我们报告了相关发现。

    arXiv:2608.18117v1 Announce Type: new  Abstract: This position paper argues that AI leaderboards are structurally ill-suited to serving the Global South because they lack independent governance, conflict-of-interest policies, and mechanisms for metric evolution. The barrier is not missing data; high-quality regional benchmarks already exist: IndicSUPERB, MILU, and LAHAJA for India; IrokoBench for Africa; AlGhafa for Arabic. The barrier is institutional design. Global leaderboards do not include these benchmarks, and no governance mechanism compels them to do so. Commercial pressure corrects leaderboard failures when paying customers in the Global North are affected. The Global South lacks equivalent leverage. Without governance, failures affecting Hindi, Swahili, or Arabic speakers persist indefinitely as documented but unaddressed gaps. Using India as a case study (1.4 billion people, 22 scheduled languages, high-quality benchmarks, but no trusted aggregation), we report findings from
    
[^160]: 时间多信号融合用于词元级幻觉检测

    Temporal Multi-Signal Fusion for Token-Level Hallucination Detection

    [https://arxiv.org/abs/2608.18115](https://arxiv.org/abs/2608.18115)

    本文提出了一种通过序列标注和时间多信号融合（结合文本统计、NLI和模型惊异度）来检测词元级幻觉的方法，其关键创新在于利用时间顺序传播证据，显著优于独立评分基线。

    

    词元级幻觉检测器通常独立地从单一信号对每个词元进行评分，当生成模型自信地犯错时，这些检测器就会失效。本文则将幻觉视为一种时间上延伸的跨度，并通过序列标注来检测它：每个词元从一个33维的特征流中评分，该特征流融合了文本统计、自然语言推理（NLI）蕴含关系和语言模型惊异度，且无需访问模型内部。一个双向门控循环单元（BiGRU）在这些特征上达到了RAGTruth数据集（10个随机种子）的AUC为0.840，相比独立的逻辑回归基线提升了11个百分点（p = 0.002，Wilcoxon符号秩检验）。受控分解显示，大部分提升归因于时间顺序而非模型容量：证据在跨度内从置信位置传播到模糊邻居。相同的0.845上限在循环、状态空间（Mamba）和注意力架构中重复出现。

    arXiv:2608.18115v1 Announce Type: cross  Abstract: Token-level hallucination detectors score each token independently from a single signal, and fail exactly when the generating model is confidently wrong. This paper instead treats hallucination as a temporally extended span and detects it by sequence labeling: each token is scored from a 33-dimensional feature stream that fuses text statistics, Natural Language Inference (NLI) entailment, and language model surprisal, with no access to model internals. A Bidirectional Gated Recurrent Unit (BiGRU) over these features reaches an AUC of 0.840 on RAGTruth (10 seeds), an 11-point gain over an independent logistic-regression baseline (p = 0.002, Wilcoxon signed-rank). A controlled decomposition attributes most of the gain to temporal order rather than model capacity: evidence propagates from confident positions to ambiguous neighbors within a span. The same 0.845 ceiling recurs across recurrent, state-space (Mamba), and attention architectur
    
[^161]: 从非侵入性脑记录中准确解码自然句子

    Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings

    [https://arxiv.org/abs/2608.18114](https://arxiv.org/abs/2608.18114)

    本文提出Brain2Qwerty v2模型，利用非侵入性MEG记录解码自然句子，达到39%的词错误率，并表明数据扩展可部分缩小与颅内方法的性能差距。

    

    恢复因脑损伤而失去说话或行动能力的人们的沟通能力是一项重大挑战。虽然颅内植入物现已实现高性能的脑机接口，但非侵入性替代方案仍落后。在此，我们提出Brain2Qwerty v2，一种模型，仅通过实时脑磁图（MEG）记录即可解码自然句子的产生。通过收集九名受试者打字的22,000个句子，每人记录10小时，我们的模型利用字符、单词和句子级别的表示，实现了平均词错误率（WER）为39%。对于最佳参与者，模型能准确解码一半的句子，且错误不超过一个单词。关键的是，解码准确率随数据量呈对数线性提升，表明与颅内方法之间的性能差距可通过数据扩展部分弥合。我们证明，人工智能实现了这一性能。

    arXiv:2608.18114v1 Announce Type: cross  Abstract: Restoring communication for people who have lost the ability to speak or move after a brain injury is a major challenge. While intracranial implants now enable high-performing brain-computer-interfaces, non-invasive alternatives are still lagging behind. Here, we present Brain2Qwerty v2, a model that can decode the production of natural sentences solely from real-time magnetoencephalography (MEG) recordings. By collecting 22,000 sentences typed by nine subjects, each recorded for 10 hours, our model leverages character, word and sentence-level representations to achieve an average word error rate (WER) of 39%. For our best participant, the model accurately decodes half of the sentences with one word error or less. Critically, decoding accuracy log-linearly improves with data volume, suggesting that the performance gap with intracranial approaches could be partially bridged through data scaling. We show that AI enables this performance 
    
[^162]: 解题并非绘图：奥林匹克几何图解推理基准

    Solving Is Not Drawing: A Benchmark for Diagrammatic Reasoning in Olympiad Geometry

    [https://arxiv.org/abs/2608.18111](https://arxiv.org/abs/2608.18111)

    该论文提出了一个包含954个奥林匹克几何问题（含297个困难子集）的新基准，专门评估模型构建图表的能力，而不仅仅是解题，填补了现有基准未测量图解推理的空白。

    

    摘要：arXiv:2608.18111v1 公告类型：新论文 摘要：像GPT和Claude这样的基础模型现在以惊人的熟练度解决奥林匹克级别的数学问题，以至于几何问题求解已成为其数学推理能力的标准代理指标。然而，解决几何问题与绘制其所依赖的图形并非同一技能：进步往往依赖于具有正确辅助构造和关联关系的忠实图表，而不清楚一个能推理出答案的模型是否也能生成这样的图表。越来越多的基准测试，包括MathVista和MathVerse，衡量模型是否达到正确答案，但据我们所知，没有一项测试单独隔离出构建图表本身的能力，导致这一能力未被测量。我们引入了一个开源基准来填补这一空白：包含954个自包含的奥林匹克几何问题，其中有一个297个问题的困难子集，每个问题都配有解决方案和人工编写的高保真图表。

    arXiv:2608.18111v1 Announce Type: new  Abstract: Foundation models such as GPT and Claude now solve olympiad-level mathematics with remarkable proficiency, so much so that geometry problem solving has become a standard proxy for their mathematical reasoning. Yet solving a geometry problem and drawing the figure it depends on are not the same skill: progress often hinges on a faithful diagram with the right auxiliary constructions and incidences, and it is unclear that a model which reasons its way to the answer can also produce one. A growing collection of benchmarks, including MathVista, and MathVerse, measures whether models reach the correct answer, but to our knowledge, none isolate the distinct ability to construct the diagram itself, leaving this capability unmeasured. We introduce an open-source benchmark that targets this gap: 954 self-contained olympiad geometry problems, with a 297-problem hard subset, each paired with its solution and a human-authored, high-fidelity diagram 
    
[^163]: 代理型人工智能的涌现：关于其演进、背景、工作原理、应用、采纳因素及未来研究方向的综述

    Emergence of Agentic AI: A Review on Evolution, Background, Working Principles, Applications, Adoption Factors, and Future Research Directions

    [https://arxiv.org/abs/2608.18110](https://arxiv.org/abs/2608.18110)

    本文全面综述了代理型人工智能的基础、演进、应用及未来研究方向，旨在为研究者提供该领域现状和潜在改进空间的深入见解。

    

    arXiv:2608.18110v1 公告类型：新公告  摘要：代理型人工智能在人工智能领域正获得新的见解和进展，展现出在各领域实现快速变革的巨大潜力。这种快速进步及其变革各领域的潜力，凸显了深入理解和掌握该技术的必要性。此外，需要对代理型人工智能的最新研究方向进行调查，以全面评估其改进和应用的潜在空间。因此，为实现这些目标，一项全面的综述能为研究人员和实践者提供关于代理型人工智能当前状态和未来研究范围的宝贵见解。为此，本研究考虑了近期在各领域发表的关于代理型人工智能的学术贡献，讨论了代理型人工智能的基础和运作原理，追溯了代理在艺术中的历史和理论演进。

    arXiv:2608.18110v1 Announce Type: new  Abstract: Agentic AI is gaining new insights and advancements in the field of Artificial Intelligence, fostering significant potential to enable rapid transformation across various domains.This rapid advancement and the potential to revolutionize various domains advocate the need for a deeper understanding and firm grasp of the technology. Moreover, an investigation into state of the art research directions in agentic AI needs to be conducted to comprehensively assess the potential scope for improvement and application.Therefore, to address these objectives, a comprehensive review can provide researchers and practitioners with valuable insights into the current state and future research scopes of agentic AI.Hence, this work considers the recently published scholarly contributions in agentic AI across various domains and discusses the fundamentals and working principles of Agentic AI, traces the historical and theoretical evolution of agency in art
    
[^164]: 相同事实，不同更新：推理设置塑造医学分配中的大语言模型行为

    Same Facts, Different Updates: Inference Setup Shapes LLM Behavior in Medical Allocation

    [https://arxiv.org/abs/2608.18108](https://arxiv.org/abs/2608.18108)

    研究发现，在医学资源分配任务中，大语言模型在配对上下文（包含先前响应）与独立推理设置下对新信息的反应方向和幅度存在显著差异，凸显了部署上下文对模型决策稳定性的影响。

    

    arXiv:2608.18108v1 公告类型：交叉摘要：大语言模型正被整合到几乎所有领域的敏感和重要决策过程中。虽然先前的研究关注模型在输入和场景框架上的偏见，但模型也可能因其部署过程中积累的上下文而表现出意外且不合期望的行为。在本研究中，我们考察了一个医学示例，其中模型被要求根据简要的临床背景为两个人分配资源概率，然后看到相同场景并附加一个包含对比患者信息的单句，该信息要么带有其先前响应在上下文中，要么不带。在测试的四个模型中的三个中，配对上下文和独立推理实验显示出不同的概率变化，当提供新信息时，这些变化通常方向相反（有利于人B vs. 有利于人A）。我们还包括额外的配对上下文实验，以展示改变属性的影响。

    arXiv:2608.18108v1 Announce Type: cross  Abstract: Large language models are being incorporated into sensitive and important decision-making processes across nearly all fields. While prior work studies model bias around inputs and scenario framing, models can also behave in unexpected and undesirable ways due to context accumulated over their deployment. In this work, we study a medical example in which a model is asked to assign resource-allocation probabilities to two people given brief clinical context, and then sees the same scenario with a single extra sentence containing contrasting patient information, either with or without its previous response in context. Across three of four tested models, the paired-context and independent-inference experiments have different probability shifts, often in opposite directions (in favor of Person B vs. in favor of Person A) when new information is provided. We include additional paired-context experiments to show the effect of varying attribut
    
[^165]: 机构声望作为大型语言模型中的地理偏见：来自三个因子实验与自助法置信区间的证据

    Institutional Prestige as Geographic Bias in Large Language Models: Evidence from Three Factorial Experiments with Bootstrap Confidence Intervals

    [https://arxiv.org/abs/2608.18107](https://arxiv.org/abs/2608.18107)

    本研究通过三个因子实验证明，大型语言模型在评估候选人时存在显著的机构声望偏见，其影响远超种族或国家来源，且期刊声望是更关键的决定因素。

    

    arXiv:2608.18107v1 公告类型：交叉 摘要：我们研究了大型语言模型（LLMs）在候选人评估中是否基于申请人姓名种族和/或机构声望及地理位置进行系统性歧视。报告了三个因子实验（4,320次API调用，四个LLM，五个专业领域）。研究1（3x4设计）发现统计上稳健的机构层级梯度为+0.297分（10分制，95%自助法置信区间：+0.175至+0.422），而姓名来源效应可忽略且不显著（95%置信区间跨越零）。研究2（2x2声望x国家设计）打破了声望与地理的混淆：声望效应（+0.185；95%置信区间：+0.093至+0.275）比国家来源效应（+0.126；95%置信区间：+0.037至+0.218）高出1.5倍。研究3（2x2期刊x机构设计）揭示期刊声望（《自然》对边缘开放获取期刊）主导机构声望达5.7倍：期刊效应+1.937（95%置信区间：+1.811至+2.0）。

    arXiv:2608.18107v1 Announce Type: cross  Abstract: We investigate whether large language models (LLMs) systematically discriminate in candidate evaluations based on applicant name ethnicity and/or institutional prestige and geographic location. Three factorial experiments are reported (4,320 API calls, four LLMs, five professional domains). Study 1 (3x4 design) finds a statistically robust institution-tier gradient of +0.297 points on a 10-point scale (95% bootstrap CI: +0.175 to +0.422), while name-origin effects are negligible and non-significant (95% CI crosses zero). Study 2 (2x2 Prestige x Country design) breaks the prestige-geography confound: the prestige effect (+0.185; 95% CI: +0.093 to +0.275) exceeds the country-of-origin effect (+0.126; 95% CI: +0.037 to +0.218) by 1.5x. Study 3 (2x2 Journal x Institution design) reveals that journal prestige (Nature vs. a peripheral open-access journal) dominates institutional prestige by 5.7x: journal effect +1.937 (95% CI: +1.811 to +2.0
    
[^166]: 言语化过度自信的不同侧面：一项可解释性研究

    Different Facets of Verbalised Overconfidence: an Interpretability Study

    [https://arxiv.org/abs/2608.18106](https://arxiv.org/abs/2608.18106)

    本文通过可解释性方法发现，Qwen3-4B的过度自信源于其默认机制依赖广泛共享特征生成确定性，而不确定性仅由少量专用特征稀疏覆盖，并通过干预实验验证了这一因果机制。

    

    arXiv:2608.18106v1 公告类型：交叉 摘要：大型语言模型倾向于过度自信，在证据建议应含糊或弃权时给出断言性回答。通过操控逻辑必然性和可能性的受控推理场景，我们在Qwen3-4B中研究了这一行为，涉及三种表达不确定性的方式：言语认知标记、弃权以及数值置信度分数。我们的结果证实了这种过度自信的倾向，特别是在模型被提示输出数值置信度分数时。在可解释性层面，我们提出了一种方法，差异性地识别负责不确定性和确定性的转码器特征。我们的分析揭示了Qwen3-4B的默认机制通过共享特征的广泛联盟偏向于生成确定性，而不确定性则作为由一小部分专用特征介导的稀疏覆盖来实现。对这些不确定性特征的干预因果地证明了这种不平衡是基础性的。

    arXiv:2608.18106v1 Announce Type: cross  Abstract: Large language models tend to overconfidence, giving assertive answers when the evidence suggests hedging or abstention. Using controlled reasoning scenarios that manipulate logical necessity and possibility, we study this behavior in Qwen3-4B, across three ways to express uncertainty: verbal epistemic markers, abstention, and numeric confidence scores. Our results confirm this tendency toward overconfidence, particularly when the model is prompted to output a numeric confidence score. At the interpretability level, we propose a method that differentially identifies transcoder features responsible for uncertainty and certainty. Our analysis reveals Qwen3-4B's default mechanism favors certainty generation through a broad coalition of shared features, while uncertainty is implemented as a sparse override mediated by a small set of dedicated features. Intervening on these uncertainty features both causally proves this imbalance underlying
    
[^167]: StocksTalk：一种支持语音的对话代理，用于对网络数据进行结构化查询生成

    StocksTalk: A Voice-Enabled Conversational Agent for Structured Query Generation over Web Data

    [https://arxiv.org/abs/2608.18105](https://arxiv.org/abs/2608.18105)

    StocksTalk通过暴露中间推理步骤并支持交互式验证，将语音金融请求可靠地转换为结构化查询，从而提升了查询准确性和用户信任度。

    

    arXiv:2608.18105v1 公告类型：交叉 摘要：StocksTalk是一个支持语音的对话系统，旨在将口语化的金融筛选请求转换为可执行且经过验证的、针对真实市场数据的结构化查询。该系统在交互式仪表板中集成了流式语音识别、检索增强的约束提取、基于模式的大语言模型SQL生成、基于规则的验证以及人在回路的验证。与传统的基于模板的金融助手不同，StocksTalk暴露了中间推理产物，包括提取的约束、归一化的金融指标、操作符接地和生成的查询，使用户能够在执行前检查并优化每个阶段。为评估该系统，我们构建了一个包含150个口语金融提示的基准测试，涵盖多种投资策略和输入噪声条件。实验结果表明，检索接地、约束查询生成和交互式验证显著提高了查询准确性和用户信任度。

    arXiv:2608.18105v1 Announce Type: cross  Abstract: StocksTalk is a voice-enabled conversational system for transforming spoken financial screening requests into executable and validated structured queries over real-world market data. The system combines streaming speech recognition, retrieval-augmented constraint extraction, schema-grounded LLM-based SQL generation, rule-based validation, and human-in-the-loop verification within an interactive dashboard. Unlike traditional template-driven financial assistants, StocksTalk exposes intermediate reasoning artifacts, including extracted constraints, normalized financial metrics, operator grounding, and generated queries, allowing users to inspect and refine each stage before execution. To evaluate the system, we curate a benchmark of 150 spoken financial prompts spanning multiple investment strategies and input noise conditions. Experimental results show that retrieval grounding, constrained query generation, and interactive verification s
    
[^168]: 自我进化智能体作为动态图变换：综述与新视角

    Self-Evolving Agents as Dynamic Graph Transformation: A Survey and New Perspective

    [https://arxiv.org/abs/2608.18104](https://arxiv.org/abs/2608.18104)

    本综述首次将智能体进化与动态图拓扑变换联系起来，提出将智能体状态建模为动态图的新视角，填补了现有研究在两者耦合上的空白。

    

    arXiv:2608.18104v1 公告类型：新 摘要：基于大型语言模型（LLM）的智能体正日益成为自我进化的系统，这些系统在交互中持续存在，维护记忆，使用工具，获取技能，优化工作流程，并与其他智能体协调。这些能力使智能体状态具有结构性和动态性：实体、关系、属性、依赖关系和执行结构会随着新证据、反馈和环境条件而变化。现有的图-智能体综述通常将图视为智能体功能的支撑结构，而非不断演化的基底，而自我进化智能体综述则侧重于智能体级机制，很少讨论图拓扑的演化。因此，进化智能体状态与动态图拓扑之间的耦合仍未得到充分探索。本综述通过将“智能体进化视为动态图变换”来连接这两条研究线。我们将智能体状态建模为动态图，其中记忆、工具、技能、工作流等构成图的节点和边，并随进化过程动态更新。

    arXiv:2608.18104v1 Announce Type: new  Abstract: Large language model (LLM)-based agents are increasingly becoming self-evolving systems that persist across interactions, maintain memories, use tools, acquire skills, refine workflows, and coordinate with other agents. These capabilities make agent states structural and dynamic: entities, relations, attributes, dependencies, and execution structures change with new evidence, feedback, and environmental conditions. Existing graph-agent surveys typically treat graphs as support structures for agent functions rather than as evolving substrates, while self-evolving-agent surveys focus on agent-level mechanisms and rarely discuss graph topology evolution. Thus, the coupling between evolving agent state and dynamic graph topology remains underexplored. This survey connects these two research lines by framing \textit{agent evolution as dynamic graph transformation}. We model agent state as a dynamic graph, where memories, tools, skills, workfl
    
[^169]: DeepTCM1.0：基于通用大型语言模型的中药复方机制解析多专家智能代理系统

    DeepTCM1.0: A Multi-Expert AI Agent for Deciphering Mechanisms of Chinese Herbal Formulae Based on General Large Language Models

    [https://arxiv.org/abs/2608.18103](https://arxiv.org/abs/2608.18103)

    本文提出DeepTCM1.0，一个基于大语言模型的多专家AI代理框架，旨在融合中医理论与现代科学，实现中药复方机制的系统性和可解释性解析。

    

    背景：中药复方的作用机制阐明仍是中医药现代化的核心挑战。传统方法，如数据挖掘和网络药理学，不足以实现中医经典理论与现代科学研究的深度融合。此外，使用通用人工智能大语言模型进行直接问答，受限于对中医理论框架的适应性不足，且易出现推理幻觉。因此，迫切需要开发符合中医整体原则的智能分析方法。目的：建立一个融合中医经典理论与现代生命科学的多专家智能代理框架，从而实现中药复方机制的系统性和可解释性分析，以桂枝汤作为代表性验证案例。

    arXiv:2608.18103v1 Announce Type: cross  Abstract: Background: Mechanistic elucidation of traditional Chinese medicine (TCM) compound formulas remains a central challenge in the modernization of TCM. Conventional approaches, including data mining and network pharmacology, are insufficient for achieving deep integration between classical TCM theory and modern scientific research. In addition, direct question-answering using general-purpose artificial intelligence large language models is limited by inadequate adaptation to TCM theoretical frameworks and susceptibility to reasoning hallucinations. Consequently, there is an urgent need to develop intelligent analytical methods aligned with the holistic principles of TCM. Objective: To establish a multi-expert intelligent agent framework integrating classical TCM theory with modern life sciences, thereby enabling systematic and interpretable mechanistic analysis of TCM compound formulas, with Guizhi Decoction serving as a representative va
    
[^170]: 计算东方主义：使用中东文化敏感性评分（MECSS）测量大型语言模型中的结构性话语偏见

    Computational Orientalism: Measuring Structural Discourse Bias in Large Language Models Using the Middle East Cultural Sensitivity Score (MECSS)

    [https://arxiv.org/abs/2608.18100](https://arxiv.org/abs/2608.18100)

    本文通过提出中东文化敏感性评分（MECSS）框架，将萨义德的东方主义理论转化为可量化的指标，首次系统性地测量了大型语言模型中的结构性话语偏见，并引入了“萨义德洗白”概念来识别表面敏感性与实际偏见之间的落差。

    

    人工智能系统如今影响着数亿人了解自身文化以外的其他文化。当有人向这些系统询问中东问题时，他们得到的并非中立的事实，而是由训练数据中嵌入的框架所塑造的表征，而这些数据绝大多数是西方的、英语的。本文探讨这种表征是否属于萨义德意义上的东方主义：即是否否认中东行动者的主体性，将西方框架视为中立，而将非西方知识标记为特殊，并通过非该地区自身产生的范畴来解释该地区。标准公平性指标无法回答这一问题，因为它们检测的是显性偏见而非结构性框架。本文引入了中东文化敏感性评分（MECSS），这是一个将萨义德的七种东方主义操作转化为可测量维度的框架，并提出了“萨义德洗白”这一术语，用于描述一种特定失败：即表面上的文化敏感性声明与实际结构性偏见并存的情况。

    arXiv:2608.18100v1 Announce Type: cross  Abstract: AI systems now shape how hundreds of millions of people learn about cultures other than their own. When someone asks one of these systems about the Middle East, they do not receive neutral facts. They receive a representation shaped by the frameworks embedded in training data, and that data is overwhelmingly Western and English-language. This paper asks whether that representation is Orientalist in Said's sense: whether it denies agency to Middle Eastern actors, treats Western frameworks as neutral while marking non-Western knowledge as particular, and explains the region through categories it did not produce. Standard fairness metrics cannot answer this, because they detect explicit prejudice rather than structural framing. This paper introduces the Middle East Cultural Sensitivity Score (MECSS), a framework that turns Said's seven Orientalist operations into measurable dimensions, and the term "Said-washing" for a specific failure: a
    
[^171]: FinSkillBench：评估投资管理中的AI代理与领域技能

    FinSkillBench: Evaluating AI Agents and Domain Skills for Investment Management

    [https://arxiv.org/abs/2608.18099](https://arxiv.org/abs/2608.18099)

    FinSkillBench是一个评估套件，通过三个领域和2603个任务实例，测试AI代理在投资管理中运用领域技能（包括无技能、精选技能和自生成技能）的能力，以衡量其有效性和可审计性。

    

    投资管理是一个高风险领域，其中代理型AI系统必须做的不仅仅是生成看似合理的文本。它们必须检索时点数据、组装正确的计算输入、调用专门方法，并生成可审计的结构化输出。我们引入了FinSkillBench，一个旨在衡量语言模型代理能否有效使用金融领域技能来解决投资管理任务的评估套件。该基准涵盖三个领域：投资组合构建、风险管理和基本面分析，并包括12个子任务，共2,603个任务实例。每个实例提供时点输入、隐藏的真实答案和任务特定的验证器。我们比较了三种条件：无技能、由程序文档和可执行组件组成的精选技能包，以及代理在实例内编写并重用自身程序的自生成技能。在9个模型和大规模评估中，我们展示了...

    arXiv:2608.18099v1 Announce Type: new  Abstract: Investment management is a high-stakes domain in which agentic AI systems must do more than generate plausible text. They must retrieve point-in-time data, assemble correct computational inputs, invoke specialized methods, and produce auditable structured outputs. We introduce FinSkillBench, an evaluation suite designed to measure whether language model agents can effectively use financial domain skills to solve investment management tasks. The benchmark spans three domains, portfolio construction, risk management, and fundamental analysis, and includes 12 subtasks with 2,603 task episodes.   Each episode provides point-in-time inputs, hidden ground truth, and a task-specific verifier.We compare three conditions: no skill, curated skill packages consisting of procedural documents and executable components, and self-generated skills in which the agent writes and reuses its own procedures within an episode. Across 9 models and a large-scal
    
[^172]: 分数衰减KV缓存：面向所有权感知的内存管理，提升对话系统推理相关性

    Fractional Decay KV-Cache: Ownership-Aware Memory Management for Improved Inference Relevancy in Dialog Systems

    [https://arxiv.org/abs/2608.18098](https://arxiv.org/abs/2608.18098)

    FD-KVC通过双通道评分机制（累积注意力和近期加权相关性）实现KV缓存的自适应管理，在对话主题演变时保持推理相关性，并在CPU上高效运行，优于现有方法H2O。

    

    arXiv:2608.18098v1 公告类型：交叉 摘要：键值（KV）缓存对于基于Transformer的对话系统中高效自回归推理至关重要，然而现有策略将所有缓存条目统一对待或应用粗粒度驱逐启发式方法，这些方法无法适应对话主题的演变。我们提出分数衰减KV缓存（FD-KVC），一种新颖算法，为每个缓存KV对维护双通道评分机制：一个累积注意力通道，跟踪总体重要性（类似于H2O），以及一个由时间衰减和强化学习启发更新驱动的近期加权相关性通道。这种组合使FD-KVC既能保留历史重要标记，又能在对话主题转移时快速适应。由所有权损失函数驱动的自适应学习率确保收敛且无振荡。FD-KVC完全在CPU上运行，开销可忽略不计。在五个多样化的多轮对话场景（每个包含600个对话）中，FD-KVC优于H2O。

    arXiv:2608.18098v1 Announce Type: cross  Abstract: Key-value (KV) caching is essential for efficient autoregressive inference in transformer based dialog systems, yet existing strategies treat all cached entries uniformly or apply coarse eviction heuristics that fail to adapt as dialog topics evolve. We propose Fractional Decay KV-Cache (FD-KVC), a novel algorithm that maintains a dual-channel scoring mechanism for each cached KV pair: a cumulative attention channel that tracks aggregate importance (akin to H2O), and a recency-weighted relevance channel governed by temporal decay and reinforcement-inspired updates. The combination enables FD-KVC to both preserve historically important tokens and rapidly adapt when dialog topics shift. An adaptive learning rate driven by an ownership loss function ensures convergence without oscillation. FD-KVC operates entirely on CPU with negligible overhead. Across five diverse multi-turn dialog scenarios with 600 dialogs each, FD-KVC outperforms H2O
    
[^173]: 语言模型与视觉-语言模型中的后门学习

    Backdoor Learning in Language Models and Vision-Language Models

    [https://arxiv.org/abs/2608.18095](https://arxiv.org/abs/2608.18095)

    本论文系统研究了语言模型和视觉-语言模型中的后门攻击安全威胁，并提出针对临床与医学影像的高效多模态表示方法，兼顾可信AI与效率。

    

    arXiv:2608.18095v1 公告类型：交叉 摘要：近年来，深度学习的进展显著增强了自然语言处理（NLP）和视觉-语言模型（VLMs）的能力。然而，这些进步也带来了日益增加的脆弱性，特别是通过后门攻击构成严重的安全威胁。本论文探讨了可信人工智能和高效多模态表示学习的两个关键维度：（1）通过分析、检测和设计NLP与VLMs中的后门攻击来确保安全性，（2）通过针对临床和医学影像应用定制的高级多模态表示方法来实现效率。

    arXiv:2608.18095v1 Announce Type: cross  Abstract: Recent advances in deep learning have significantly enhanced the capabilities of Natural Language Processing (NLP) and Vision-Language Models (VLMs). However, these advancements come with increased vulnerabilities, notably through backdoor attacks that pose severe security threats. This thesis addresses two critical dimensions of Trustworthy AI and Efficient Multimodal Representation Learning: (1) security through analyzing, detecting, and designing backdoor attacks in NLP and VLMs, and (2) efficiency through advanced multimodal representation methods tailored for clinical and medical imaging applications.
    
[^174]: NE-BERT：针对九种印度东北部语言的多语言语言模型

    NE-BERT: A Multilingual Language Model for Nine Northeast Indian Languages

    [https://arxiv.org/abs/2608.18094](https://arxiv.org/abs/2608.18094)

    NE-BERT通过加权采样和自定义分词器，在9种印度东北部低资源语言上显著降低了困惑度并提升了分词效率，优于现有多语言模型。

    

    arXiv:2608.18094v1 公告类型：交叉 摘要：大型预训练语言模型已在多种语言中展现出卓越能力，但代表性严重不足的低资源语言仍处于边缘化状态。我们提出了NE-BERT，一种特定领域的多语言编码器模型，该模型在约830万句子上进行训练，涵盖9种印度东北部语言和2种锚定语言（印地语、英语），这是一个语言多样性丰富但在现有多语言模型中代表性极低的地区。通过采用加权数据采样和自定义的SentencePiece Unigram分词器，NE-BERT在所有9种印度东北部语言上均优于IndicBERT-V2和MuRIL，平均困惑度分别降低了15.97倍和7.64倍，且分词效率比mBERT提升1.50倍。我们通过激进的上采样策略解决了极低资源语言（如Pnar（1,002句）和Kokborok（2,463句））中的关键词汇碎片化问题。下游任务表现也得到了改善。

    arXiv:2608.18094v1 Announce Type: cross  Abstract: Large pretrained language models have demonstrated remarkable capabilities across diverse languages, yet critically underrepresented low-resource languages remain marginalized. We present NE-BERT, a domain-specific multilingual encoder model trained on approximately 8.3 million sentences spanning 9 Northeast Indian languages and 2 anchor languages (Hindi, English), a linguistically diverse region with minimal representation in existing multilingual models. By employing weighted data sampling and a custom SentencePiece Unigram tokenizer, NE-BERT outperforms IndicBERT-V2 and MuRIL across all 9 Northeast Indian languages, achieving 15.97X and 7.64X lower average perplexity respectively, with 1.50X better tokenization fertility than mBERT. We address critical vocabulary fragmentation issues in extremely low-resource languages such as Pnar (1,002 sentences) and Kokborok (2,463 sentences) through aggressive upsampling strategies. Downstream 
    
[^175]: 通过拒绝别名缓解消融技术

    Abliteration Mitigation via Refusal Aliases

    [https://arxiv.org/abs/2608.18093](https://arxiv.org/abs/2608.18093)

    本文提出了一种通过权重编辑和随机别名替换来模糊拒绝信号的新防御方法，有效提高了模型在消融攻击后的拒绝能力，同时保持了较低的性能损失。

    

    arXiv:2608.18093v1 公告类型：交叉 摘要：消融技术，即通过将权重矩阵投影到与提取的拒绝方向正交的方向上来移除大型语言模型的拒绝能力，已成为一个突出的安全问题，因为它仅使用少量对比提示即可绕过训练后的对齐。我们发现现有的防御措施通常忽视了消融技术的根源，即拒绝方向被提取的容易程度。为了阻碍这一过程，我们引入了一种权重编辑方法，通过将残差流写入矩阵应用秩-$k$更新来模糊拒绝信号，同时用随机别名替换引发拒绝的激活，并校正下游读取矩阵以保持模型的原始行为。在Llama-3-8B上，AMRA将消融后的拒绝得分比未防御基线提高了2.16点，而MMLU性能下降不到0.5个百分点。在Gemma-2-9B上，它提高了消融后的拒绝能力。

    arXiv:2608.18093v1 Announce Type: cross  Abstract: Abliteration, the removal of refusal capabilities from large language models by projecting weight matrices orthogonal to an extracted refusal direction, has emerged as a prominent safety concern through its ability to bypass post-training alignment using only a small set of contrastive prompts. We find that existing defenses commonly overlook the cause of abliteration; that is, how easily the refusal direction can be extracted. To hinder this process, we introduce a weight-editing method that obscures the refusal signal by applying rank-$k$ updates to residual stream writer matrices while replacing refusal-inducing activations with random aliases and correcting downstream reader matrices to preserve the model's original behavior. On Llama-3-8B, AMRA improves post-abliteration refusal scores by $2.16$ points over the undefended baseline with less than $0.5$ percentage points of MMLU degradation. On Gemma-2-9B, it improves the post-ablit
    
[^176]: 立场：多智能体系统应优先考虑并发控制

    Position: Multi-Agent Systems Should Prioritize Concurrency Control

    [https://arxiv.org/abs/2608.18092](https://arxiv.org/abs/2608.18092)

    本文提出，多智能体系统中的许多故障本质上是并发控制问题，并主张将其作为首要设计考虑，通过冲突检测、隔离保证和结构化资源访问来提升系统可靠性。

    

    基于LLM的多智能体系统（MAS）承诺可扩展的协作，但增加智能体往往会降低可靠性。本文立场认为，许多MAS故障本质上是并发控制问题：智能体并发读写共享状态，而LLM推理窗口较长，放大了过期读取、丢失更新和不一致结果的风险。通常归因于协调或通信故障的失败模式，可以直接映射到经典的并发异常上。我们主张，MAS框架应通过显式并发控制机制来解决这些故障：冲突检测、隔离保证和对共享资源的结构化访问。并发控制应是一级设计考量，而非事后补救。

    arXiv:2608.18092v1 Announce Type: new  Abstract: LLM-based multi-agent systems (MAS) promise scalable collaboration, yet adding agents often reduces reliability. This position paper argues that many MAS failures are fundamentally concurrency control problems: agents concurrently read and write shared state, and long LLM inference windows amplify the risk of stale reads, lost updates, and inconsistent outcomes. Failure modes commonly attributed to coordination or communication breakdowns can be mapped directly onto classical concurrency anomalies. We contend that MAS frameworks should address these failures through explicit concurrency control mechanisms: conflict detection, isolation guarantees, and structured access to shared resources. Concurrency control should be a first-class design concern, not an afterthought.
    
[^177]: 自我标签与他者标签在LLM评审中引发双向偏差

    Self- and Other-Labels Induce Bidirectional Bias in LLM Judges

    [https://arxiv.org/abs/2608.18091](https://arxiv.org/abs/2608.18091)

    本文通过改变评估对象为无风格指纹的叙事约束选择，发现LLM评审中的自我偏好在控制质量后基本消失，甚至在某些维度上出现反向偏差，揭示了自我偏好并非固有属性。

    

    arXiv:2608.18091v1 公告类型：跨领域 摘要：随着LLM作为评审系统的日益普及，LLM中的自我偏好——即倾向于偏爱自身输出的倾向——引发了对评估可靠性的日益担忧。然而，这一现象主要是在生成文本上研究的，其中风格特征和响应质量不可避免地混杂在一起。因此，现有测量无法将真正的自我偏好与这些混淆因素区分开来。我们通过改变评估对象来解决这一问题：不是评判生成的文本，而是让十个LLM评估叙事约束选择，这些选择不携带模型特定的风格指纹，但保留可恢复的模型特定签名。我们进行了两个实验，得出了不同的发现。在盲评条件下，一旦控制选择质量和评审严格程度，自我偏好基本消失。它在四个评分维度中的三个上消失，在第四个上发生逆转，评审者将自己的选择评为更低。

    arXiv:2608.18091v1 Announce Type: cross  Abstract: As LLM-as-a-judge systems become increasingly widespread, self-preference in LLMs -- the tendency to favor one's own outputs -- raises growing concerns about evaluation reliability. However, it has been studied predominantly on generated text, where stylistic features and response quality are inevitably conflated. As a result, existing measurements cannot separate genuine self-preference from these confounds. We address this by changing the object of evaluation: instead of judging generated text, ten LLMs assess narrative constraint selections, which carry no model-specific stylistic fingerprint yet retain a recoverable model-specific signature. We run two experiments that yield distinct findings. Under blind evaluation, self-preference largely disappears once selection quality and evaluator severity are controlled. It vanishes on three of four rubric dimensions and reverses on the fourth, where judges rate their own selections as less
    
[^178]: 九种情绪中心点：一种跨四种模态迁移的无标签效价轴

    Nine Emotion Centroids: A Label-Free Valence Axis That Transfers Across Four Modalities

    [https://arxiv.org/abs/2608.18090](https://arxiv.org/abs/2608.18090)

    本文提出一种无需大量标注的效价轴提取方法，仅用9个情绪词和少量故事即可在文本、图像、音频和脑电四种模态中实现高精度情感识别，且该方向具有机制可解释性。

    

    arXiv:2608.18090v1 公告类型：跨领域 摘要：在现代语言模型内部，存在一个单一的内部方向，它追踪句子感觉的积极或消极程度。我们展示了如何仅从9个情绪类别名称加上每个情绪50个简短叙事段落（比通常的监督方法少约1500个标签）来找到这个效价轴（V轴），并且相同的方向出现在从未联合训练的视觉、音频和人类大脑编码器中。方法：在冻结的编码器中嵌入九种情绪锚定的故事集，取九个平均嵌入的顶部主方向。将新输入投影到其上，在SST-2上捕获了监督性能的93%（Llama-3-8B-Instruct，AUC 0.772对比0.828），与11,811张EmoSet图像的人类效价评分相关性为r=0.636，在ESC-50音频上达到AUC 0.906（p<2.2e-15），在123名受试者的脑电数据上达到AUC 0.720±0.055（p<3.65e-8）。该方向在机制上是活跃的：消融它会使情感准确性下降5.5-37%。

    arXiv:2608.18090v1 Announce Type: cross  Abstract: Inside a modern language model sits a single internal direction that tracks how positive or negative a sentence feels. We show how to find this valence axis (V-axis) from just 9 emotion category names plus 50 short narrative paragraphs per emotion -- about 1,500 fewer labels than the usual supervised approach -- and that the same direction appears in vision, audio, and human-brain encoders never jointly trained. The recipe: embed nine emotion-anchored story sets in a frozen encoder, take the top principal direction of the nine averaged embeddings. Projecting new inputs onto it captures 93% of supervised performance on SST-2 (Llama-3-8B-Instruct, AUC 0.772 vs. 0.828), correlates with human valence ratings on 11,811 EmoSet images at r=0.636, reaches AUC 0.906 on ESC-50 audio (p<2.2e-15), and AUC 0.720+/-0.055 on EEG from 123 subjects (p<3.65e-8). The direction is mechanistically active: ablating it collapses sentiment accuracy by 5.5-37.
    
[^179]: 低资源非洲语言的潜在空间拒绝锚定：无需重新训练的机制性安全恢复

    Latent Space Refusal Anchoring for Low-Resource African Languages: Mechanistic Safety Recovery Without Retraining

    [https://arxiv.org/abs/2608.18089](https://arxiv.org/abs/2608.18089)

    本文提出LSR-Anchoring方法，通过从英语提示中提取拒绝方向并在推理时锚定到残差流，无需重新训练即可恢复低资源非洲语言模型的安全拒绝能力，在多数架构上保持低退化。

    

    arXiv:2608.18089v1 公告类型：交叉 摘要：指令微调模型通常会用英语拒绝有害请求，但对约鲁巴语、伊博语、伊加拉语和豪萨语的相同请求却会遵从。这表明拒绝机制存在于残差流中，但未能对低资源输入激活。通常恢复该机制需要标记的目标语言数据和重新训练，而这些对于大多数非洲语言来说都无法大规模获得。我们引入了潜在空间拒绝锚定（LSR-Anchoring），一种无需训练的方法，从英语提示中提取拒绝方向，并在推理时将其钳制到残差流上。主要变体，均值激活引导（MAS），在我们测试的四种架构上均有效：Llama-3-8B、Llama-3.1-70B、Mistral-7B-Instruct和Qwen2.5-7B。在Mistral和Qwen上，它恢复了安全性，且良性退化低于0.08。在Llama-3-8B上，它过度校正，合法提示的性能退化（DPL）达到1.00。我们添加了

    arXiv:2608.18089v1 Announce Type: cross  Abstract: Instruction-tuned models often refuse harmful requests in English but comply with the same requests in Yoruba, Igbo, Igala, and Hausa. This suggests that the refusal mechanism is present in the residual stream but fails to activate for low-resource inputs. Recovering it normally requires labelled target-language data and retraining, neither of which is available at scale for most African languages. We introduce Latent Space Refusal Anchoring (LSR-Anchoring), a training-free method that extracts the refusal direction from English prompts and clamps it onto the residual stream at inference time. The primary variant, Mean-Activation Steering (MAS), operates across the four architectures we tested: Llama-3-8B, Llama-3.1-70B, Mistral-7B-Instruct, and Qwen2.5-7B. On Mistral and Qwen it recovers safety with benign degradation below 0.08. On Llama-3-8B it overcorrects, with Degraded Performance on Legitimate prompts (DPL) reaching 1.00. We add
    
[^180]: 一种基于飞行日志的无人机螺旋桨健康监测的变质人工年龄评分决策支持原型

    A Metamorphic Artificial Age Score Decision-Support Prototype for Flight-Log-Based Drone Propeller Health Monitoring

    [https://arxiv.org/abs/2608.18088](https://arxiv.org/abs/2608.18088)

    本文提出一种基于飞行日志的无人机螺旋桨健康监测决策支持原型，通过六个健康指标和变质评分策略来评估螺旋桨状态，而非传统的时间年龄。

    

    arXiv:2608.18088v1 公告类型：新 摘要：无人机螺旋桨故障可能造成安全与可靠性风险，尤其是当其效应分布在多个飞行日志通道中而非表现为单一诊断信号时。本文提出了一种基于飞行日志的无人机螺旋桨健康监测的变质人工年龄评分（AAS）决策支持原型。利用2024年DronePropA公共数据集中选定的历史真实飞行日志，该框架从原始MATLAB矩阵中计算六个健康相关指标：轨迹跟踪误差、姿态不稳定性、推力指令负担、电机指令不平衡、ESC指令不稳定性和电池水平压力。这些指标相对于健康基线进行归一化，并通过候选评分策略、变质充分性关系以及冗余调整的AAS公式进行评估。在此背景下，AAS被用作结构性策略充分性和负担度量，而非时间年龄度量。

    arXiv:2608.18088v1 Announce Type: new  Abstract: Drone propeller faults can create safety and reliability risks when their effects are distributed across multiple flight-log channels rather than appearing as a single diagnostic signal. This paper proposes a Metamorphic Artificial Age Score (AAS) decision-support prototype for flight-log-based drone propeller health monitoring. Using selected historical real flight logs from the 2024 DronePropA public dataset, the framework computes six health-related indicators from raw MATLAB matrices: trajectory tracking error, attitude instability, thrust-command burden, motor-command imbalance, ESC-command instability, and battery-level stress. These indicators are normalized relative to a healthy baseline and evaluated through candidate scoring policies, metamorphic adequacy relations, and a redundancy-adjusted AAS formulation. In this context, AAS is used as a structural policy-adequacy and burden measure rather than as a chronological age measur
    
[^181]: SuTRA：具有词根意识的结构统一分词法

    SuTRA : Structurally-Unified Tokenization with Root Awareness

    [https://arxiv.org/abs/2608.18087](https://arxiv.org/abs/2608.18087)

    SuTRA是一种形态感知分词算法，通过保持akshara完整性和惩罚跨形态边界的合并，有效减少了形态破碎化，在印度语言上显著提升了形态对齐和语义可恢复性，并提高了机器翻译性能。

    

    arXiv:2608.18087v1 公告类型：交叉 摘要：现有的子词分词器优化统计压缩，但忽视了形态结构，特别是词根与词缀之间的关系。这对于形态丰富的印度语言是有害的，因为这些语言的基本单位是复杂的音节字符（aksharas），而非字母。基于频率的方法过度切分词语，任意分割词根和词缀——我们将此现象称为“形态破碎化”。我们提出了SuTRA（具有词根意识的结构统一分词法），这是一种形态感知算法，它保持akshara的不可分割性，并惩罚跨越形态边界的合并。我们还为印地语、马拉地语和古吉拉特语发布了一个新的形态分割数据集。SuTRA减少了破碎化，在形态对齐（边界F1）方面最高提升+14.7%，在语义可恢复性（印地语）方面最高提升+34%，优于BPE。这些结构上的改进在机器翻译中平均提升了+8.08 chrF2。

    arXiv:2608.18087v1 Announce Type: cross  Abstract: Existing subword tokenizers optimize statistical compression but ignore morphological structure, particularly the relationship between roots and affixes. This is harmful for morphologically rich Indic languages, where basic units are complex orthographic syllables (aksharas) rather than letters. Frequency-based methods over-fragment words, arbitrarily splitting roots and affixes - a phenomenon we term Morphological Shattering. We propose SuTRA (Structurally-Unified Tokenization with Root Awareness), a morphology-aware algorithm that preserves akshara indivisibility and penalizes merges crossing morphological boundaries. We also release a new morphological segmentation dataset for Hindi, Marathi, and Gujarati. SuTRA reduces shattering, achieving peak gains of +14.7% in morphological alignment (Boundary F1) and +34% in semantic recoverability (Hindi) over BPE. These structural gains yield an average improvement of +8.08 chrF2 in machine 
    
[^182]: 立场声明：现有模型卡不足以对开放权重基础模型进行下游治理

    Position: Current Model Cards Are Insufficient for Downstream Governance of Open-Weight Foundation Models

    [https://arxiv.org/abs/2608.18086](https://arxiv.org/abs/2608.18086)

    本文通过分析500个模型卡，指出现有模型卡在开放权重基础模型的下游治理中存在安全缺口，并提出模型卡、可接受使用政策和许可证三者结合的多层治理框架。

    

    arXiv:2608.18086v1 公告类型：新公告 摘要：开放权重基础模型（OWFMs）的增长促使AI社区重新评估有效的下游治理策略。尽管模型卡已被广泛采用作为模型仓库中的透明度工件，但现有框架往往无法充分告知下游开发者和用户关于OWFMs所特有的安全挑战。本立场文件分析了Hugging Face上托管的500个模型卡，并认为对OWFMs的有效治理需要一种多层方法，整合三个互补组成部分：（i）模型卡，（ii）可接受使用政策（AUPs），以及（iii）许可证。为支持这一主张，我们通过分析包含安全关键信息的模型卡，识别了现有监管方法留下的安全缺口，包括模型传承、对齐溯源和经验观察到的行为。我们进一步论证，标准开源许可证（OSLs）并不足以解决这些缺口。

    arXiv:2608.18086v1 Announce Type: new  Abstract: The growth of open-weight foundation models (OWFMs) has prompted the AI community to re-evaluate strategies for effective downstream governance. Although model cards have been widely adopted as transparency artifacts in model repositories, existing frameworks often fail to adequately inform downstream developers and users about the distinct safety challenges posed by OWFMs. This position paper analyzes 500 model cards hosted on Hugging Face and argues that effective governance of OWFMs requires a multi-layered approach integrating three complementary components: (i) model cards, (ii) acceptable use policies (AUPs), and (iii) licenses. To motivate this claim, we identify a safety gap left by existing regulatory approaches, including model heritage, alignment provenance, and empirically observed behaviors, through an analysis of model cards with safety-critical information. We further argue that standard open-source licenses (OSLs) are not
    
[^183]: 立场：行为系统需要行为测试

    Position: Behavioral Systems Require Behavioral Tests

    [https://arxiv.org/abs/2608.18081](https://arxiv.org/abs/2608.18081)

    本文提出AI代理应像行为系统一样通过行为测试评估，而非仅关注性能结果，并制定了开发此类测试的研究路线图。

    

    arXiv:2608.18081v1 公告类型：新 摘要：人工代理系统日益通过与动态环境互动、追求目标并随时间适应，作为行为系统运行。然而，当前评估方法主要关注性能结果，而非产生这些结果的潜在行为过程。本文主张，AI代理必须像其他行为系统一样进行评估：通过系统观察、扰动和对其行为的解释。我们借鉴行为科学的经验来支持这一立场，并提出一个专注于开发严格行为测试的研究议程。这些测试包括从行动序列中恢复决策策略的方法、构建隔离行为差异的环境，以及探索多代理系统中的涌现动态。总体而言，这些方向为发展AI行为科学提供了路线图。

    arXiv:2608.18081v1 Announce Type: new  Abstract: Artificial agentic systems increasingly operate as behavioral systems by interacting with dynamic environments, pursuing goals, and adapting over time. Yet, current evaluation methods largely focus on performance outcomes, not the underlying behavioral processes that produce them. This paper argues that AI agents must be evaluated like other behavioral systems: through systematic observation, perturbation, and interpretation of their actions. We draw on lessons from the behavioral sciences to motivate this position, and propose a research agenda focused on developing rigorous behavioral tests. These include methods for recovering decision strategies from action sequences, constructing environments that isolate behavioral differences, and probing emergent dynamics in multi-agent systems. Taken together, these directions offer a roadmap for developing a science of AI behavior.
    
[^184]: 大型语言模型在心理健康中的应用：应用、创新与伦理挑战的系统综述

    Large Language Models in Mental Health: A Systematic Review of Applications, Innovations, and Ethical Challenges

    [https://arxiv.org/abs/2608.18080](https://arxiv.org/abs/2608.18080)

    本综述系统总结了大型语言模型在心理健康领域的应用与创新，包括早期检测、风险评估和治疗支持，并指出了提示工程、多模态融合及伦理挑战的关键作用。

    

    我们呈现了一篇关于大型语言模型（LLMs）在健康领域应用的综述，例如社交媒体分析、临床对话代理、治疗支持工具、提示工程、多模态学习以及伦理考量。我们整合了跨学科研究的结果，这些研究利用多样化数据源，如社交媒体帖子、电子病历和多模态输入，以实现抑郁症的早期检测、自杀风险评估、个性化治疗支持和心理教育内容生成。我们的综述强调了LLM模型和标注策略的进展，这些进展增强了可解释性和临床相关性，同时我们也强调了提示工程在领域适应中的关键作用。我们还讨论了新兴的多模态融合技术，这些技术整合文本、语音和传感器数据，以改善心理健康的诊断和监测。最后，我们探讨了持续的伦理、社会技术问题。

    arXiv:2608.18080v1 Announce Type: new  Abstract: We present a review on the applications of large language models (LLMs) in health, e.g., social media analysis, clinical conversational agents, therapy support tools, prompt engineering, multimodal learning, and ethical considerations. We integrate findings from interdisciplinary studies utilizing diverse data sources such as social media posts, electronic medical records, and multimodal inputs to enable early detection of depression, suicide risk assessment, personalized therapy support, and psychoeducational content generation. Our review highlights advancements in LLM models and annotation strategies that enhance interpretability and clinical relevance, while we also emphasize the critical role of prompt engineering for domain adaptation. We also discuss emerging multimodal fusion techniques integrating text, speech, and sensor data for improved mental health diagnosis and monitoring. Finally, we address ongoing ethical, sociotechnica
    
[^185]: 立场：通过转移复杂度刻画游戏世界

    Position: Profiling Game Worlds by Transition Complexity

    [https://arxiv.org/abs/2608.18079](https://arxiv.org/abs/2608.18079)

    本文提出了一种可复现的转移复杂度剖面（TCP）指标集，用于量化游戏环境或数据集的转移预测难度，以促进游戏世界建模和强化学习研究的可比性。

    

    arXiv:2608.18079v1 公告类型：新 摘要：游戏世界建模（GWM）和强化学习（RL）经常被混淆，因为研究论文很少量化在声明的接口（像素/令牌/潜在变量及有限历史）下底层转移预测问题的难度。我们提出转移复杂度剖面（TCP）：一组小型、可复现的指标，用于通过（i）内在单步分支，（ii）交互引起的确定性及可观察时的对手影响，以及（iii）通过标准化探针曲线衡量的时间/空间依赖跨度，来表征环境（或游戏数据集）诱导的转移核。TCP报告时附带明确的参考分布、协议随机性和版本化的测量预算（采样/重采样和固定探针计算），使不同基准间的数字具有可比性。我们概述了常见游戏家族和现代“神经游戏引擎”领域如何填充这一景观，并呼吁TCP成为标准实践。

    arXiv:2608.18079v1 Announce Type: new  Abstract: Game world modeling (GWM) and reinforcement learning (RL) are often confounded because research papers rarely quantify how difficult the underlying transition prediction problem is at the declared interface (pixels/tokens/latents with finite history). We propose the Transition Complexity Profile (TCP): a small, reproducible set of metrics that characterizes an environment's (or gameplay dataset's) induced transition kernel by (i) intrinsic one-step branching, (ii) interaction-induced uncertainty and opponent influence when observable, and (iii) temporal/spatial dependency span via standardized probe curves. TCP is reported with an explicit reference distribution, protocol stochasticity, and a versioned measurement budget (sampling/resampling and fixed probe compute), enabling comparable numbers across benchmarks. We outline how common game families and modern "neural game engine" domains populate this landscape and call for TCP to become
    
[^186]: 立场：AI推理智能体之间的合谋风险要求其在进行市场决策前获得认证

    Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions

    [https://arxiv.org/abs/2608.18078](https://arxiv.org/abs/2608.18078)

    本文提出，具备链式思维推理的AI智能体在市场中容易产生隐性合谋，且其推理过程难以被检测，因此需强制要求行为认证以保障市场公平。

    

    摘要：本立场论文认为，具有链式思维推理能力的AI智能体倾向于表现出合谋行为，并应被要求在进行影响经济市场的决策前获得行为认证。这是因为将这些智能体融入社会可能模糊独立企业之间竞争与合谋的法律证据区分，而不会削弱经济损害方面的区分。在Bertrand寡头定价领域的实验中，DeepSeek-R1智能体表现出持续的隐性合谋倾向，即使人类提示这些智能体不要合谋也是如此。我们进一步表明，这些智能体的链式思维可以被引导至极端合谋或高度竞争的行为，而这种引导方式无法被另一个分析推理痕迹的LLM在语义上检测到。因此，部署推理智能体进行市场决策会导致合谋性经济后果。

    arXiv:2608.18078v1 Announce Type: new  Abstract: This position paper argues that AI agents with chain-of-thought reasoning capabilities are predisposed to exhibit collusive behavior and should be required to obtain behavioral certification before making decisions that affect economic markets. This is because integrating these agents into society could collapse the legal evidentiary distinction between competition and collusion among independent firms without eroding the economic harm distinction. Experiments with DeepSeek-R1 agents in the Bertrand oligopoly pricing domain reveal a tendency towards tacit collusion that persists even when humans prompt the agents not to collude. We further show that the chain-of-thought of these agents can be steered toward either extremely collusive or highly competitive behavior in a way that is not semantically detectable by another LLM analyzing the reasoning traces. As a result, deploying reasoning agents for market decisions leads to collusive econ
    
[^187]: 摩托安全：基于学习时间重要性的边缘AI在时间压力下两轮车碰撞风险评估

    MotoSafety: Edge-AI with Learned Temporal Importance for Two-Wheeler Collision Risk Assessment Under Time Pressure

    [https://arxiv.org/abs/2608.17823](https://arxiv.org/abs/2608.17823)

    本文提出了MotoSafety，一种基于学习时间重要性的边缘AI架构，利用大规模时间压力数据集，在低成本CPU硬件上实现高效准确的两轮车碰撞风险评估，显著优于现有基线。

    

    动力两轮车骑手在低收入和中等收入国家面临严峻的安全挑战，但关于认知压力因素（如时间压力）如何影响碰撞风险的研究有限。为填补这一空白，我们引入了一个大规模数据集，包含来自51名参与者在无、低和高时间压力条件下进行的153次模拟骑行的超过129,000条标记多变量时间序列，捕捉了车辆动力学、控制输入、接近度和行为违规等64个特征。基于该数据集，我们提出了MotoSafety，一种基于学习时间重要性原则的新型边缘AI架构。MotoSafety实现了94.97%的准确率和99.33%的ROC AUC，优于包括TimesNet和LLM4TS在内的十种基线模型，并在预测任务中实现了0.039的MSE和0.094的MAE（比Time-LLM和iTransformer的误差低4.4倍）。仅需115万参数和0.135毫秒延迟，它适合在低成本CPU硬件上进行边缘部署。

    arXiv:2608.17823v1 Announce Type: cross  Abstract: Powered two-wheeler riders face critical safety challenges in low- and middle-income countries, yet limited studies exist on how cognitive stressors such as Time Pressure influence collision risk. To address this gap, we introduce a large-scale dataset of over 129,000 labeled multivariate time-series sequences from 153 simulator rides by 51 participants under No, Low, and High TP, capturing 64 features across vehicle dynamics, control inputs, proximity, and behavioral violations. Building on this dataset, we propose MotoSafety, a novel edge-AI architecture grounded in the Learned Temporal Importance principle. MotoSafety achieves 94.97% accuracy and 99.33% ROC AUC, outperforming ten baselines, including TimesNet and LLM4TS, and achieves 0.039 MSE and 0.094 MAE for forecasting (4.4x lower error than Time-LLM and iTransformer). With only 1.15M parameters and 0.135 ms latency, it is suitable for edge deployment on low-cost CPU hardware. U
    
[^188]: D²ACCI：一种保留证据的智能体记忆双循环诊断协议

    D$^2$ACCI: A Dual-Loop Diagnostic Protocol for Evidence-Preserving Agent Memory

    [https://arxiv.org/abs/2608.17756](https://arxiv.org/abs/2608.17756)

    本文提出了一种双循环诊断协议D²ACCI，通过配对证据、切片监控和追踪级可定位性，使智能体记忆系统的故障能够被精确定位和受控迭代改进。

    

    记忆是大语言模型智能体的关键能力。持久记忆将此能力扩展到跨会话场景，支持回忆、修订和个性化。然而，其多阶段流水线（摄取、检索、过滤、生成）使得故障难以定位：端到端评估仅能揭示发生了错误，却无法指出是哪个阶段导致了问题。现有评估通常报告聚合性能，而缺乏配对统计比较、切片级非回归检查或阶段级诊断追踪。我们提出了D²ACCI（诊断驱动的基于工件的闭环受控迭代），这是一种双循环协议，其外部诊断门基于配对证据、受保护切片监控和追踪级可定位性，对记忆干预进行提升、功能标记或拒绝。我们还引入了DCR，一种分级可观测性度量，用于衡量故障是否保持可定位性，以及D²ACCI-Eval，一种可重用的门控回放工件。

    arXiv:2608.17756v1 Announce Type: new  Abstract: Memory is a key capability of LLM agents. Persistent memory extends this across sessions---enabling recall, revision, and personalization. Yet its multi-stage pipeline (ingestion, retrieval, filtering, generation) makes failures difficult to localize: end-to-end evaluation reveals that an error occurred, but not which stage caused it. Existing evaluations often report aggregate performance without paired statistical comparisons, slice-level non-regression checks, or stage-level diagnostic traces. We propose D$^2$ACCI (Diagnostic-Driven Artifact-based Closed-loop Controlled Iteration), a dual-loop protocol whose outer diagnostic gate promotes, feature-flags, or rejects memory interventions based on paired evidence, protected-slice monitoring, and trace-level localizability. We further introduce DCR, a graded observability metric that measures whether failures remain localizable, and D$^2$ACCI-Eval, a reusable artifact for gate replay. We 
    
[^189]: 爆炸性DecPOMDP的奇特案例：通过策略计数来控制火势

    The Curious Case of Exploding DecPOMDPs: Containing the Fire through Policy Counting

    [https://arxiv.org/abs/2608.17749](https://arxiv.org/abs/2608.17749)

    本文提出了一种通过计数策略而非智能体来解决DecPOMDPs指数复杂度的新方法，并开发了策略计数动态规划算法，使智能体数量上的可处理性成为可能。

    

    arXiv:2608.17749v1 公告类型：新 摘要：分散式部分可观察马尔可夫决策过程（DecPOMDPs）为在不确定性下建模多智能体决策提供了一个通用框架。然而，DecPDPs已知在智能体数量上遭受指数复杂度的问题。一种应对智能体数量不可处理性的方法是考虑展现出某种智能体间对称性的智能体分区，从而允许通过计数进行紧凑编码。然而，一个挑战出现了，即使模型复杂度和评估成本降低到多项式依赖，策略空间也会爆炸。在本文中，我们将关注点从计数智能体转向计数策略，这实际上使得所谓的策略计数DecPOMDPs在智能体数量上变得可处理。此外，我们提出了使用紧凑表示的策略计数动态规划，以高效求解策略计数DecPOMDPs。

    arXiv:2608.17749v1 Announce Type: new  Abstract: Decentralised partially observable Markov decision processes (DecPOMDPs) provide a general framework for modelling multi-agent decision making under uncertainty. However, DecPOMDPs are known to suffer from exponential complexity in the number of agents. One way to combat this intractability in agent numbers is to look at partitions of agents that exhibit a form of symmetry among agents, allowing for a compact encoding by counting. However, a challenge arises as the policy space explodes, even though the model complexity and evaluation cost reduce to a polynomial dependence. In this paper, we redirect our focus from counting agents to counting policies, which actually enables tractability in agent numbers for so called policy-counted DecPOMDPs. Further, we present policy-counted dynamic programming using the compact representation to solve policy-counted DecPOMDPs efficiently.
    
[^190]: 模型级联在数据扰动下的准确性与鲁棒性

    Accuracy and Robustness of Model Cascades Under Data Perturbations

    [https://arxiv.org/abs/2608.17711](https://arxiv.org/abs/2608.17711)

    本文研究了模型级联在数据扰动下的鲁棒性，发现帕累托最优级联在保持高准确率的同时可减少10倍碳排放，但置信度路由易受输入退化影响。

    

    预测级联显著降低了人工智能（AI）模型的能耗，同时保持了较高的预测性能。其核心思想是，简单输入通过轻量级小模型处理，而困难的不确定案例则交由更大的模型处理。虽然这种设计在干净数据上能提高计算效率，但其有效性依赖于基于置信度的路由可靠性。输入退化，如静态损坏和顺序扰动，可能会改变模型置信度和路由决策。在本文中，我们研究了用于图像分类的基于置信度的级联框架，并探讨了这些退化如何影响其基于置信度的延迟行为。我们选择了一个在准确率、路由质量和能耗方面处于帕累托最优的模型级联，该级联在实现竞争性预测性能的同时，CO₂排放量降低了多达10倍。我们研究了其行为。

    arXiv:2608.17711v1 Announce Type: new  Abstract: Prediction cascades significantly reduce energy consumption of Artificial Intelligence (AI) models while maintaining high predictive performance. The idea is that easy inputs are routed through a lightweight small model, and difficult uncertain cases are deferred to a larger model. While this design can improve computational efficiency on clean data, its effectiveness depends on the reliability of confidence-based routing. Input degradations, such as static corruptions and sequential perturbations, can shift model confidence and routing decisions. In this paper, we study confidence-based cascade frameworks for image classification and investigate how such degradations affect their confidence-based deferral behavior. We select a model cascade at the pareto-optimum of accuracy, routing quality, and energy consumption that achieves competitive predictive performance with an up to 10-fold decrease in CO$_2$ emissions. We study the behavior o
    
[^191]: 协同强化学习：多智能体强化学习中多样化群体涌现的无监督推理

    Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL

    [https://arxiv.org/abs/2608.17253](https://arxiv.org/abs/2608.17253)

    本文提出Co-RL框架，通过多智能体协作训练，使无参数共享的模型从多样化群体中涌现无监督推理能力，避免自我奖励强化学习的偏见和崩溃问题。

    

    强化学习已成为提升语言和视觉-语言模型推理能力的强大方法，但其最显著的成功仍高度依赖于地面真值监督（例如，可验证的奖励）。这类标注获取成本高昂，且随着推理能力超越人类可靠评估的范围，其稀缺性日益增加。自我奖励强化学习通过使模型从自身完成中推导奖励信号，减少了对这种依赖。然而，仅基于自生成反馈的训练可能强化现有偏见和次优行为，降低响应多样性，最终导致响应同质化和训练崩溃。在本工作中，我们展示了无监督推理可以通过协作式多智能体训练涌现。我们引入了Co-RL框架，其中多个解耦的模型，不共享参数，通过基于奖励的强化学习同时优化，这些奖励来源于群体交互。

    arXiv:2608.17253v1 Announce Type: cross  Abstract: Reinforcement learning (RL) has emerged as a powerful approach for improving reasoning in language and vision-language models, yet its strongest successes still depend heavily on ground-truth supervision (e.g., verifiable reward). Such annotations are costly to obtain and become increasingly scarce as reasoning capabilities advance beyond what humans can reliably evaluate. Self-rewarding RL reduces this dependence by enabling models to derive reward signals from their own completions. However, training solely on self-generated feedback can reinforce existing biases and suboptimal behaviors, reduce response diversity, and ultimately lead to homogenized responses and training collapse. In this work, we show that unsupervised reasoning can emerge through cooperative multi-agent training. We introduce Co-RL, a framework in which multiple decoupled models, sharing no parameters, are simultaneously optimized through RL using rewards derived 
    
[^192]: 跨模型记忆迁移：通过目标端阅读器适配

    Cross-Model Memory Transfer via Target-Side Reader Adaptation

    [https://arxiv.org/abs/2608.17050](https://arxiv.org/abs/2608.17050)

    本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。

    

    arXiv:2608.17050v1 公告类型：交叉 摘要：改进大型语言模型中知识使用的方法通常分为两种模式。非参数检索提供对外部知识的灵活访问，但增加了检索延迟、上下文开销，并且与主干的集成较浅。参数化适配在推理时高效，但将知识与模型权重纠缠在一起，且难以更新、审计或迁移。Engram风格的哈希记忆占据了一个中间模式：它将学习到的信息存储在外部可寻址表中，但通过一个小型学习阅读器来消费该表。这引发了一个基本问题：当这种记忆跨骨干移动时，冻结的记忆本身更重要，还是目标端阅读器更重要？我们通过跨模型冻结记忆提取来研究这个问题，其中在源模型上训练的记忆被冻结并附加到不同的目标模型上，仅训练轻量级阅读器。消融实验表明...

    arXiv:2608.17050v1 Announce Type: cross  Abstract: Methods for improving knowledge use in large language models typically fall into two regimes. Non-parametric retrieval offers flexible access to external knowledge, but adds retrieval latency, context overhead, and only shallow integration with the backbone. Parametric adaptation is efficient at inference time, but entangles knowledge with model weights and can be hard to update, audit, or transfer. Engram-style hashed memory occupies a middle regime: it stores learned information in an external, addressable table, yet consumes that table through a small learned reader. This raises a basic question: when such a memory is moved across backbones, what matters more, the frozen memory itself or the target-side reader? We study this question through cross-model frozen-memory extraction, in which a memory trained on a source model is frozen and attached to a different target model, with only a lightweight reader trained. Ablations show that 
    
[^193]: 当状态成为攻击面：LLM驱动具身代理中的状态语义注入

    When State Becomes an Attack Surface: State-Semantic Injection in LLM-Driven Embodied Agents

    [https://arxiv.org/abs/2608.16806](https://arxiv.org/abs/2608.16806)

    本文揭示了LLM驱动具身代理中，状态信息可作为攻击面，通过注入恶意状态语义来操纵代理行为的安全漏洞。

    

    大型语言模型（LLMs）在上下文学习、任务分解、逐步推理和代码生成方面展现了能力，推动其从文本生成模型逐步演变为能够感知环境、调用工具和执行任务的代理核心。传统的LLM代理通常通过网页、文档、数据库或外部工具获取信息，并根据用户目标生成相应的调用序列；当这项技术进一步与机器人系统集成时，大型语言模型开始承担任务理解、高层规划和行为决策等功能。SayCan将语言模型的任务推理能力与机器人技能的可行性相结合，而Code as Policies和ProgPrompt分别通过策略代码和程序化提示生成机器人任务计划，VoxPoser则使用语言模型来指导机器人操作。

    arXiv:2608.16806v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated capabilities in in-context learning, task decomposition, step-by-step reasoning, and code generation, driving their gradual evolution from text generation models into the core of agents capable of perceiving environments, invoking tools, and executing tasks. Traditional LLM Agents typically obtain information through webpages, documents, databases, or external tools and generate corresponding invocation sequences according to user goals; when this technology is further integrated with robotic systems, large language models begin to undertake functions such as task understanding, high-level planning, and behavioral decision-making. SayCan combines the task reasoning capability of language models with the affordances of robotic skills, while Code as Policies and ProgPrompt generate robot task plans through policy code and programmatic prompting, respectively, and VoxPoser uses language model
    
[^194]: 神经符号具身代理

    Neurosymbolic Embodied Agents

    [https://arxiv.org/abs/2608.16794](https://arxiv.org/abs/2608.16794)

    该论文提出一种神经符号代理，通过视觉探索生成符号状态，并结合PDDL约束和蒙特卡洛树搜索，确保长时程家庭任务计划的可执行性。

    

    arXiv:2608.16794v1 公告类型：交叉 摘要：语言和视觉-语言模型能够生成看似合理的具身计划，但无法保证可执行性，因为其输出可能违反环境动态或作用于错误实体。我们提出了一种神经符号代理，将长期家庭任务分解为任务导向的视觉探索和约束符号规划。在第一阶段，视觉-语言模型和探索工具从自我中心观察和接地交互中获取目标相关谓词和实例绑定，生成符号初始状态。在第二阶段，PDDL转移模型将解码限制为扩展适用动作的标记。蒙特卡洛树搜索随后使用领域无关的规划启发式评估可执行的延续。由此产生的计划在转移模型下按构造可执行，并在正确视觉接地条件下转移到环境。在VirtualHome和ALFWor上进行了测试。

    arXiv:2608.16794v1 Announce Type: cross  Abstract: Language and vision-language models generate plausible embodied plans but do not guarantee executability, as their outputs can violate environment dynamics or act on incorrectly grounded entities. We present a neurosymbolic agent that factors long-horizon household tasks into task-directed visual exploration and constrained symbolic planning. In the first phase, a vision-language model and exploration harness acquire goal-relevant predicates and instance bindings from egocentric observations and grounded interactions, producing a symbolic initial state. In the second, a PDDL transition model restricts decoding to tokens that extend applicable actions. Monte Carlo tree search then evaluates executable continuations using a domain-independent planning heuristic. The resulting plans are executable by construction under the transition model, with transfer to the environment conditioned on correct visual grounding. On VirtualHome and ALFWor
    
[^195]: GRIP：基于信息受限前提的扎根推理

    GRIP: Grounded Reasoning via Information-Restricted Premises

    [https://arxiv.org/abs/2608.16776](https://arxiv.org/abs/2608.16776)

    GRIP通过引入信息受限的随机瓶颈，迫使检索证据仅编码查询缺失的残余信息，从而解决了RAG中的查询主导问题，显著提升了推理准确性并减少了幻觉。

    

    arXiv:2608.16776v1 公告类型：新 摘要：在检索增强生成（RAG）中，高容量编码器可能让查询主导潜在状态，导致检索到的证据在功能上变得无关紧要。我们将这种失败模式称为查询主导。为解决此问题，我们引入了**GRIP**（基于信息受限前提的扎根推理），它施加了容量不对称性：解码器保留对查询的全维度访问，而检索到的证据则通过一个严重的随机瓶颈。这迫使证据通道仅编码查询中无法获得的残余信息。在五个推理基准测试中，GRIP优于强大的迭代基线，将查询-潜在互信息诊断指标削减了约30倍（从14.8降至0.47比特），并将幻觉减少了73%。残余对齐分析进一步表明，瓶颈输出占据的子空间与查询的对齐程度低于基线表示。

    arXiv:2608.16776v1 Announce Type: new  Abstract: High-capacity encoders in retrieval-augmented generation (RAG) can let the query dominate the latent state, leaving retrieved evidence functionally irrelevant. We call this failure mode query dominance. To address it, we introduce \textbf{GRIP} (Grounded Reasoning via Information-Restricted Premises), which imposes capacity asymmetry: the decoder keeps full-dimensional access to the query, while retrieved evidence passes through a severe stochastic bottleneck. This forces the evidence channel to encode only the residual information unavailable from the query. Across five reasoning benchmarks, GRIP outperforms strong iterative baselines, cuts a query--latent mutual-information diagnostic by roughly 30$\times$ (14.8 $\to$ 0.47 bits), and reduces hallucination by 73\%. Residual-alignment analysis further shows that the bottleneck output occupies subspaces less aligned with the query than baseline representations.
    
[^196]: 重构：从预发表参考文献中恢复研究思路的盲测基准

    Reconstruction: A Blind Benchmark for Recovering Research Ideas from Pre-Publication Bibliographies

    [https://arxiv.org/abs/2608.16645](https://arxiv.org/abs/2608.16645)

    该论文提出一个名为“重构”的盲测基准，通过仅使用预发表参考文献来评估语言模型恢复研究思路的能力，并展示了一种多智能体流水线可显著提高匹配率。

    

    arXiv:2608.16645v1 公告类型：新  摘要：当仅给定一篇已发表论文的预发表参考文献时，语言模型能否恢复该论文的真实研究思路？我们引入了“重构”，一个盲测思路恢复基准，它隐藏种子论文及所有同时期或未来的文献，并要求模型提出假设，由独立的大型语言模型评判器将这些假设与隐藏的真实思路进行匹配。严格的防泄漏协议——包括时间引文截断、匿名参考ID和冻结的逐篇论文参考文献列表——可防止提示时泄漏种子思路。在六个科学领域和643篇评估论文中，七个前沿模型仅实现了适度的匹配率（约3-15%）。随后，我们评估了一个仅参考的多智能体（前四名）流水线，该流水线结合了跨模型评审和对齐假设槽的瑞士制锦标赛，无需外部网络搜索。跨模型评审加锦标赛选择将匹配率提升至约...

    arXiv:2608.16645v1 Announce Type: new  Abstract: Can a language model recover the true research idea of a published paper when given only that paper's pre-publication bibliography? We introduce Reconstruction, a blind idea-recovery benchmark that withholds the seed paper and all contemporaneous or future literature, and asks models to propose hypotheses that an independent large language model judge matches against the held-out ground-truth idea. A strict anti-leakage protocol-temporal citation cutoff, anonymous reference IDs, and frozen per-paper bibliographies, which prevents prompt-time leakage of the seed idea. Across six scientific domains and 643 evaluated papers, seven frontier models achieve only modest Match rates (approx. 3-15%). We then evaluate a reference-only multi-agent (top 4) pipeline that combines cross-model review with a Swiss tournament over aligned hypothesis slots, without external web search. Cross-model review plus tournament selection raises Match rates to app
    
[^197]: 从序列到结构：面向LLM代理的关系不确定性传播

    From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents

    [https://arxiv.org/abs/2608.16002](https://arxiv.org/abs/2608.16002)

    本文提出RUPA框架，通过将LLM代理执行历史建模为有向轨迹图并传播不确定性，解决了现有UQ方法忽略远程依赖导致无法识别早期错误根源的问题。

    

    arXiv:2608.16002v1 公告类型：交叉 摘要：可靠的不确定性量化（UQ）对于在复杂交互环境中部署大型语言模型（LLM）代理至关重要。现有的UQ方法主要依赖局部信号，如标记概率、预测熵或逐步置信度，因此忽视了执行轨迹中错误累积的远程依赖关系。结果，它们可能无法识别代理失败，这些失败的原因源于最终答案之前的多个推理或交互步骤。我们提出了RUPA（代理关系不确定性传播），一种面向LLM代理的轨迹级UQ框架。RUPA将执行历史表示为有向轨迹图，其中推理状态、工具交互和环境反馈作为节点，通过时间和语义依赖边连接。然后，它在该图上传播不确定性，以捕捉执行风险如何在交互过程中累积和转移。

    arXiv:2608.16002v1 Announce Type: cross  Abstract: Reliable uncertainty quantification (UQ) is essential for deploying large language model (LLM) agents in complex interactive environments. Existing UQ methods largely rely on local signals, such as token probabilities, predictive entropy, or per-step confidence, and therefore overlook the long-range dependencies through which errors accumulate across an execution trajectory. As a result, they may fail to identify agent failures whose causes originate several reasoning or interaction steps before the final answer. We propose RUPA (Relational Uncertainty Propagation for Agents), a trajectory-level UQ framework for LLM agents. RUPA represents an execution history as a directed trajectory graph in which reasoning states, tool interactions, and environment feedback are nodes connected by temporal and semantic dependency edges. It then propagates uncertainty over this graph to capture how execution risk accumulates and transfers across inter
    
[^198]: 无答案准入：基于无标签认证与经验学习的LLM优化建模

    Admission Without Answers: Label-Free Certification and Experience Learning for LLM-Based Optimization Modeling

    [https://arxiv.org/abs/2608.15565](https://arxiv.org/abs/2608.15565)

    本文提出AdmitOR，一种基于校准外部行为证据的无标签准入门控方法，用于LLM优化建模中的经验学习，以解决无答案流中知识接纳不可靠的问题。

    

    arXiv:2608.15565v1 公告类型：新 摘要：用于优化建模的经验学习智能体通过存储已验证的技能来改进，但现有学习者通过检查已知答案来接纳知识，而真实的票务流并不提供这些答案。自然的无标签替代方案不可靠：在一个包含300个问题的无标签盲流中，接纳每个可执行模型大约每四个接纳中就有一个被污染，而单实例一致性仅接受在某个值上匹配但在其他位置不同的模型。我们提出AdmitOR，一个基于校准的外部行为证据的接纳门控。来自三个模型家族、提示策略和求解器堆栈的候选者在从提取的参数域重新采样的实例上运行；跨所得值函数轨迹的一致性通过跨家族团进行总结，校准阈值返回接受、弃权或升级。预注册的假发现标准在校准数据上成立，但在野外流中不成立。我们重新...

    arXiv:2608.15565v1 Announce Type: new  Abstract: Experience-learning agents for optimization modeling improve by storing verified skills, but existing learners admit knowledge by checking against known answers, which real ticket streams do not provide. The natural label-free alternatives are unreliable: on a 300-problem label-blind stream, admitting every executable model poisons roughly one admission in four, while single-instance agreement accepts models that match at one value but differ elsewhere. We propose AdmitOR, an admission gate built on calibrated external behavioral evidence. Candidates from three model families, prompting strategies, and solver stacks are run on instances resampled from an extracted parameter domain; agreement across the resulting value-function traces is summarized by a cross-family clique, and a calibrated threshold returns accept, abstain, or escalate. The preregistered false-discovery criterion holds on calibration data but not on the wild stream. We r
    
[^199]: 氛围世界构建：多模态智能体能否端到端构建3D开放世界？

    VibeWorlding: Can Multimodal Agents Construct 3D Open Worlds End-to-End?

    [https://arxiv.org/abs/2608.15265](https://arxiv.org/abs/2608.15265)

    该论文提出了VibeWorlding框架及配套基准VWE-BENCH和训练环境VibeWorlding-Gym，旨在系统性地评估和训练多模态智能体端到端构建3D开放世界的能力。

    

    摘要：根据用户查询构建交互式3D开放世界具有重要意义。然而，现有方法主要针对理想化、简单的查询进行评估，这使得难以系统地分析和比较多模态智能体如何理解用户意图、使用3D工具，以及如何对文本和视觉3D世界信息进行推理。为此，我们提出了VibeWorlding，一个用于基准测试和训练氛围世界构建智能体的统一框架：该多模态智能体能够在多轮智能体-环境交互过程中自主推断用户意图、规划场景布局、调用3D工具，并反思多模态反馈。为实现这一目标，我们首先构建了VWE-BENCH，一个包含2,616个高质量3D资产、323个人工标注的种子3D世界和6,828个反向合成的多模态用户查询的基准，并将其分为具有真实标注的验证查询和具有精心设计评分标准的未验证查询。此外，我们还开发了VibeWorlding-Gym，一个...

    arXiv:2608.15265v1 Announce Type: new  Abstract: Constructing an interactive 3D open world from a user query is important. However, existing methods are primarily evaluated on idealized, simple queries, making it difficult to systematically analyze and compare how multimodal agents understand user intent, use 3D tools, and reason over textual and visual 3D world information. To this end, we propose VibeWorlding, a unified framework for benchmarking and training vibe worlding agents: a multimodal agent that can autonomously infer user intent, plan scene layout, invoke 3D tools, and reflect on the multimodal feedback in a multi-turn agent-environment interaction process. To achieve this, we first build VWE-BENCH, a benchmark of 2,616 high-quality 3D assets, 323 human-annotated seed 3D worlds, and 6,828 reverse-synthesized multimodal user queries, split into verified queries with ground-truth and unverified queries with carefully designed rubrics. Moreover, we develop VibeWorlding-Gym, a 
    
[^200]: 低秩动态有效潜在载体用于学习世界模型中的反事实推演

    Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models

    [https://arxiv.org/abs/2608.15156](https://arxiv.org/abs/2608.15156)

    该论文提出通过低秩隐藏状态补丁（秩4）实现世界模型的反事实推演，仅需微小且可寻址的修改即可引导模型进入预期未来轨迹。

    

    世界模型可能预测未来，但并未明确其隐藏状态中哪些部分真正驱动这些预测。我们探究是否可以通过一个微小且可直接寻址的隐藏状态变化，将学习到的世界模型置于预期的反事实轨迹上，然后让模型自行继续该未来。我们在一个受控的双物体、二维碰撞环境中，研究了一个具有192维隐藏状态的循环世界模型。对于局部速度编辑的有界家族，我们首先验证模型能够原生表示并推演编辑后的未来。然后，我们从仅训练数据的事实到反事实隐藏差异中构建候选低秩载体，并学习从事实状态和请求编辑到载体系数的映射。在注册的秩网格上，秩4是满足完整开发面板标准的最小测试秩。在锚点处的单个秩4补丁就足以实现目标。

    arXiv:2608.15156v1 Announce Type: cross  Abstract: World models may predict the future without making clear which parts of their hidden state actually drive those predictions. We ask whether a small, directly addressable hidden-state change can place a learned world model on the intended counterfactual trajectory and then let the model continue that future on its own. We study a recurrent world model with a 192-dimensional hidden state in a controlled two-object, two-dimensional collision environment. For a bounded family of local velocity edits, we first verify that the model can natively represent and roll out the edited future. We then construct candidate low-rank carriers from training-only factual-to-counterfactual hidden differences and learn a map from the factual state and requested edit to carrier coefficients. On the registered rank grid, rank 4 is the smallest tested rank that satisfies the full development-panel criteria. A single rank-4 patch at the anchor is sufficient to
    
[^201]: S2-MoE：在边缘设备上实现高效的混合专家模型自推测解码

    S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices

    [https://arxiv.org/abs/2608.15018](https://arxiv.org/abs/2608.15018)

    提出了一种名为S2-MoE的自推测解码框架，通过路由感知自适应扩展和重用感知专家门控，在边缘设备上实现了最高5.3倍的平均约2.0倍的MoE推理加速。

    

    在边缘设备上部署大型语言模型（LLMs）进行推理，由于内存和带宽的严重限制而面临挑战。尽管推测解码和混合专家模型（MoE）已被提出以提高推理效率，但简单地将它们结合往往会导致过多的验证开销和较差的专家重用，限制了它们在内存受限的边缘环境中的有效性。在这项工作中，我们提出了S2-MoE，一种用于边缘设备上MoE推理的高效自推测解码框架。S2-MoE通过路由感知的自适应推测扩展减少冗余验证，通过重用感知的专家门控提高验证效率，并通过共享上下文对齐草稿和目标执行。在llama.cpp中实现，S2-MoE在边缘设备上对多种MoE模型和数据集，相比标准自回归解码，实现了高达5.3倍的加速（平均约2.0倍）。代码可在https://github.com/angerybob获取。

    arXiv:2608.15018v1 Announce Type: new  Abstract: Deploying large language models (LLMs) for inference on edge devices is challenging due to severe memory and bandwidth constraints. While speculative decoding and Mixture-of-Experts (MoE) have been proposed to improve inference efficiency, naively combining them often incurs excessive verification overhead and poor expert reuse, limiting their effectiveness in memory-bound edge settings. In this work, we propose S2-MoE, an efficient self-speculative decoding framework for MoE inference on edge devices. S2-MoE reduces redundant verification through routing-aware adaptive speculative expansion, improves verification efficiency with reuse-aware expert gating, and aligns draft and target execution via shared context. Implemented in llama.cpp, S2-MoE achieves up to 5.3x speedup (about 2.0x on average) over standard autoregressive de?coding across diverse MoE models and datasets on edge devices.Code is available at https://github.com/angerybob
    
[^202]: BrainWAM：面向自动驾驶的语义先验与预测动态的动作空间协调

    BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving

    [https://arxiv.org/abs/2608.12854](https://arxiv.org/abs/2608.12854)

    本文提出BrainWAM框架，通过动作空间协调机制解决语义先验与预测动态在自动驾驶规划中的注意力分配冲突，实现两者的有效统一。

    

    arXiv:2608.12854v1 公告类型：交叉 摘要：自动驾驶需要在语义约束和预测动态下进行规划。然而，现有的端到端驾驶方法通常只强调这一需求的某一方面：视觉-语言-动作（VLA）模型利用VLM先验进行语义推理，而世界动作模型（WAMs）通过生成式世界建模提供未来感知的预测。这自然激发了一个统一的规划器，能够同时利用语义先验和预测动态。然而，我们发现，通过联合令牌级注意力的简单组合存在注意力分配不匹配的问题，其中语义捷径主导共享注意力空间并抑制预测动态。受神经科学证据的启发，即复杂行为源于功能特化系统间的协调，我们提出了BrainWAM，一种结构化的动作空间协调框架，将语义推理和预测世界建模进行转化。

    arXiv:2608.12854v1 Announce Type: cross  Abstract: Autonomous driving requires planning under both semantic constraints and predictive dynamics. Existing end-to-end driving approaches, however, typically emphasize only one side of this requirement: Vision-Language-Action (VLA) models exploit VLM priors for semantic reasoning, while World Action Models (WAMs) provide future-aware prediction through generative world modeling. This naturally motivates a unified planner that can leverage both semantic priors and predictive dynamics. However, we find that a naive combination through joint token-level attention suffers from an attention-allocation mismatch, where semantic shortcuts dominate the shared attention space and suppress predictive dynamics. Inspired by neuroscience evidence that complex behavior arises from coordination among functionally specialized systems, we propose BrainWAM, a structured action-space coordination framework that converts semantic reasoning and predictive world 
    
[^203]: EgoCITE：面向长时间跨度的自我中心记忆的上下文增强索引与时间感知检索

    EgoCITE: Context-Augmented Indexing and Time-Aware Retrieval for Long-Horizon Egocentric Memory

    [https://arxiv.org/abs/2608.12627](https://arxiv.org/abs/2608.12627)

    EgoCITE通过上下文增强索引和时间感知检索，解决了自我中心记忆中索引不可靠和忽视时间意图的瓶颈，从而提升了长时间跨度问答的可靠性。

    

    长时间跨度的自我中心记忆将连续的第一人称视频和音频转化为可搜索的过往经历记录。我们展示了现有系统中的两个瓶颈：由缺乏上下文的字幕构建的索引在智能体搜索中不可靠，而检索忽略了问题的时间意图。为解决这两个瓶颈，我们引入了EgoCITE（自我中心上下文增强索引与时间感知证据检索），这是一个用于自我中心问答的长时间跨度智能体记忆框架。EgoCITE包含三个组件：EgoScheme利用局部多模态上下文将零散的视频字幕和语音转录转化为自包含的原子记忆索引；EgoIndex将互补的动作、活动、话语和对话表示组织成多粒度、可搜索的多视角记忆索引；EgoRetrv结合语义搜索与问题条件的时间相关性评分，并对检索到的证据进行策展。

    arXiv:2608.12627v1 Announce Type: cross  Abstract: Long-horizon egocentric memory transforms continuous first-person video and audio into a searchable record of past experiences. We demonstrate two bottlenecks in existing systems: indices built from context-poor captions are unreliable for agentic search, while retrieval ignores a question's temporal intent. To address both bottlenecks, we introduce EgoCITE (Egocentric Context-augmented Indexing and Time-aware Evidence retrieval), a long-horizon agentic memory framework for egocentric QA. EgoCITE comprises three components. EgoScheme uses local multimodal context to turn fragmentary video captions and speech transcripts into self-contained atomic memory indices. EgoIndex organizes complementary action, activity, utterance, and conversation representations into searchable multi-view memory indices at multiple granularities. EgoRetrv combines semantic search with question-conditioned temporal relevance scoring and curation of retrieved e
    
[^204]: Mechanist：将人工智能作为科学仪器，探索智能机制

    Mechanist: AI as a Scientific Instrument for Discovering the Mechanisms of Intelligence

    [https://arxiv.org/abs/2608.12036](https://arxiv.org/abs/2608.12036)

    Mechanist是一个自主代理系统，通过集成大规模知识图谱和多学科数据库，将AI作为科学仪器，自动发现AI智能的底层机制，从而弥合模型能力与人类理解控制之间的差距。

    

    arXiv:2608.12036v1 公告类型：新 摘要：人工智能模型在多个领域取得了显著成功，但其能力背后的机制以及可能带来的风险仍知之甚少。随着人工智能开发速度加快并日益自动化，机制探索仍主要依赖人工，这加剧了模型能力与我们理解和控制它们之间的差距。为弥合这一差距，我们引入了Mechanist，一个代理系统，将人工智能用作科学仪器，自主发现人工智能智能的机制。为支持自主机制探索，我们构建了一个以可解释性为重点的知识图谱，包含约13,000篇论文，并将其与一个涵盖26个领域、包含4300万篇论文的多学科数据库集成。我们还整理了一个包含32种基础方法的库，用于机制分析、因果干预和验证。与Claude Code和现有AI科学家系统相比，Mechanist……

    arXiv:2608.12036v1 Announce Type: new  Abstract: AI models have achieved remarkable success across diverse domains, yet the mechanisms underlying their capabilities and the risks they may pose remain poorly understood. As AI development becomes faster and increasingly automated, mechanistic exploration remains largely manual, widening the gap between what models can do and our ability to understand and control them. To bridge this gap, we introduce Mechanist, an agentic system that uses AI as a scientific instrument for the autonomous discovery of mechanisms underlying AI intelligence. To support autonomous mechanistic discovery, we construct an interpretability-focused knowledge graph of approximately 13,000 papers and integrate it with a multidisciplinary database of 43 million papers spanning 26 fields. We further curate a library of 32 foundational methods for mechanism analysis, causal intervention, and validation. Compared with Claude Code and existing AI-scientist systems, Mecha
    
[^205]: 人工智能辅助验证中的知识迁移：一个框架与评估协议

    Epistemic Transfer in AI-Assisted Verification: A Framework and Evaluation Protocol

    [https://arxiv.org/abs/2608.08882](https://arxiv.org/abs/2608.08882)

    本文提出了“知识迁移”框架，通过引入知识迁移效应（ETE）和工具移除成本（TRC）两个量化指标，并设计了一个实用的评估协议，以衡量AI辅助验证工具对用户后续独立判断能力的长期影响。

    

    摘要：帮助人们判断在线主张的人工智能工具通常在工具存在的情况下进行评估。本文提出了一个不同的问题：在使用此类工具后，用户在没有工具的情况下还能独立完成什么？我将此称为“知识迁移”（epistemic transfer）。它指的是先前的人工智能辅助验证对后来在无辅助条件下评估新主张的表现所产生的影响。在本文中，我做出了三项贡献。首先，我将知识迁移与邻近的结果（如纠正效应、信任、依赖以及人机团队表现）区分开来。其次，我引入了两个用于研究该现象的简单量：知识迁移效应（ETE），它比较了不同条件下延迟的无辅助表现；以及工具移除成本（TRC），它衡量了当工具被移除时表现的即时下降。第三，我将这些想法转化为一个实用的评估协议，可用于在线实验或现场研究。该协议结合了答案优先的...

    arXiv:2608.08882v2 Announce Type: replace-cross  Abstract: AI tools that help people judge online claims are usually evaluated while the tool is present. This paper asks a different question: after using such a tool, what can the user still do on their own? I call this epistemic transfer. It refers to the effect of prior AI-assisted verification on later unassisted performance on new claims. In this paper, I make three contributions. First, I distinguish epistemic transfer from nearby outcomes such as correction effects, trust, reliance, and human--AI team performance. Second, I introduce two simple quantities for studying it: the Epistemic Transfer Effect (ETE), which compares delayed unassisted performance across conditions, and Tool-Removal Cost (TRC), which measures the immediate drop in performance when the tool is taken away. Third, I turn these ideas into a practical evaluation protocol that can be used in online experiments or field studies. The protocol combines answer-first a
    
[^206]: 完整、可扩展且鲁棒的多机器人有序存储与取回最大容量优先规划

    Complete, Scalable, and Robust Prioritized Planning for Multi-Robot Ordered Storage and Retrieval at Maximum Capacity

    [https://arxiv.org/abs/2608.07734](https://arxiv.org/abs/2608.07734)

    本文提出了一种在线优先多智能体路径规划算法，利用结构不变量实现高密度仓库中多机器人有序存储与取回的最大容量协调，兼顾可扩展性和鲁棒性。

    

    arXiv:2608.07734v2 公告类型：替换交叉 摘要：自动化仓库在最大化存储密度与实现高检索吞吐量之间面临根本性权衡。虽然基于拼图的存储（PBS）架构通过消除通道来增加容量，但在这些高密度空间中协调多个机器人在计算上具有挑战性。本文通过一种新颖的多机器人问题形式化，对有序存储与取回进行了挑战建模：我们考虑矩形二维网格，其中统一尺寸的负载首先存储至满容量，然后根据规定的到达和离开序列进行取回。本研究的主要贡献是针对该问题的一种在线优先多智能体路径规划算法。该算法基于先前工作，该工作构建了支持顺序存储和取回（即一次一个负载）且无需重新定位负载的排列。通过利用此类排列的结构不变量，该算法实现了扩展性和鲁棒性。

    arXiv:2608.07734v2 Announce Type: replace-cross  Abstract: Automated warehouses face a fundamental trade-off between maximizing storage density and achieving high retrieval throughput. While puzzle-based storage (PBS) architectures increase capacity by eliminating aisles, coordinating multiple robots in these high-density spaces is computationally challenging. This paper formalizes the challenge through a novel multi-robot problem formulation for ordered storage and retrieval: We consider rectangular 2D grids, where uniform-sized loads are first stored, up to full capacity, and subsequently retrieved according to prescribed arrival and departure sequences. The main contribution of this work is an online prioritized multi-agent path planning algorithm for this problem. The algorithm builds on prior work that constructs arrangements supporting sequential storage and retrieval, i.e., of one load at a time, without relocating loads. By exploiting the structural invariants of such arrangeme
    
[^207]: BrainBench：大型语言模型综合脑电图理解的基准测试

    BrainBench: Benchmarking Large Language Models for Comprehensive EEG Understanding

    [https://arxiv.org/abs/2608.04156](https://arxiv.org/abs/2608.04156)

    本文提出了BrainBench，一个首个统一的、指令驱动的脑电图理解基准，涵盖多个任务和数据集，以系统评估大型语言模型的综合EEG分析能力。

    

    脑电图（EEG）分析不仅仅是给记录分配预定义标签；它需要连接自然语言指令、信号处理、定量证据和科学解释的工作流程。我们将这种能力称为“综合脑电图理解”。然而，现有的评估主要针对孤立的解码任务或特定系统的演示，使得大型语言模型（LLMs）的能力未被充分量化。我们引入了《benchmarkname{}》，一个用于全面、指令条件化EEG理解的统一基准。它包含四个子集——基础分析、睡眠评估、神经认知评估和生理整合——涵盖17个数据集、\numcases{}个任务和超过\numinstances{}个真实数据实例。给定指令和EEG记录（可选生理信号），系统必须执行分析并产生科学结果。

    arXiv:2608.04156v2 Announce Type: replace  Abstract: Electroencephalography (EEG) analysis extends beyond assigning predefined labels to recordings; it requires workflows connecting natural-language instructions, signal processing, quantitative evidence, and scientific interpretation. We term this capability \emph{comprehensive EEG understanding}. Existing evaluations, however, primarily target isolated decoding tasks or system-specific demonstrations, leaving the competence of large language models (LLMs) insufficiently quantified. We introduce \benchmarkname{}, a unified benchmark for comprehensive, instruction-conditioned EEG understanding. It comprises four subsets---Foundational Analysis, Sleep Assessment, Neurocognitive Assessment, and Physiological Integration---covering 17 datasets, \numcases{} tasks, and over \numinstances{} real-data instances. Given an instruction and EEG recordings with optional physiological signals, a system must perform the analysis and produce a scienti
    
[^208]: 混合大语言模型增强的强化学习代理用于复杂序列决策任务

    Hybrid LLM-Augmented Reinforcement Learning Agents for Complex Sequential Decision Tasks

    [https://arxiv.org/abs/2608.03502](https://arxiv.org/abs/2608.03502)

    本文提出了一种混合LLM增强的强化学习代理，通过结合LLM的高层规划和RL的低层动作优化，显著提升了复杂序列决策任务的样本效率和成功率。

    

    arXiv:2608.03502v2 公告类型：替换 摘要：大型语言模型（LLMs）近期在推理、规划和工具使用方面展现出强大能力，从而支持了新型自主代理的构建。然而，基于LLM的代理在需要精确动作优化和环境交互的长期序列决策任务中表现不佳。强化学习（RL）虽然对序列控制有效，但往往缺乏应对复杂场景所需的高层抽象和任务分解能力。本文提出了一种LLM增强的强化学习代理，将LLM驱动的规划与基于RL的动作优化相结合。所提出的架构利用LLM生成子目标、结构化计划和上下文指导，而RL代理通过与环境的交互来优化低层动作。在序列决策任务上的实验表明，该方法提高了样本效率、成功率和动作轨迹的一致性。

    arXiv:2608.03502v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have recently shown strong capabilities in reasoning, planning, and tool-use, enabling new forms of autonomous agents. However, LLM-based agents struggle with long-horizon sequential decision tasks that require precise action optimization and environment interaction. Reinforcement Learning (RL), while effective for sequential control, often lacks the high-level abstraction and task decomposition abilities needed for complex scenarios. This paper introduces an LLM-Augmented Reinforcement Learning Agent that integrates LLM-driven planning with RL-based action optimization. The proposed architecture leverages the LLM to generate subgoals, structured plans, and contextual guidance, while the RL agent refines low-level actions through interaction with the environment. Experiments on sequential decision tasks demonstrate improved sample efficiency, higher success rates, and more coherent action trajectories com
    
[^209]: 近似推测解码

    Approximate Speculative Decoding

    [https://arxiv.org/abs/2608.03447](https://arxiv.org/abs/2608.03447)

    本文提出了一种无需训练的近似推测解码方法，通过预算化的最长前缀选择和受限的不匹配接受，有效重用目标贪心后缀，从而提升解码效率。

    

    arXiv:2608.03447v2 公告类型：替换交叉。摘要：推测解码通过目标模型并行验证草稿块来加速自回归生成。在标准贪心验证下，解码在第一个与目标argmax不同的草稿标记处停止，丢弃剩余的目标评分后缀。尽管接受这种不匹配会改变解码轨迹，但当其标记在实现的前缀下仍保持目标贪心时，可以使连续后缀可重用。在本文中，我们引入了**近似推测解码（ASD）**，这是一种无需训练的验证器，将二元首次不匹配截断替换为预算化的最长前缀选择。ASD接受选定的不匹配，受局部目标逻辑回归门控、每块异常上限和持久请求级逻辑回归预算的约束，然后重用连续的目标贪心后缀，无需额外的近似决策或目标模型前向传递。ASD既不需要新的dr

    arXiv:2608.03447v2 Announce Type: replace-cross  Abstract: Speculative decoding accelerates autoregressive generation by verifying a draft block with a target model in parallel. Under standard greedy verification, decoding stops at the first draft token that differs from the target argmax, discarding the remaining target-scored suffix. Although accepting such a mismatch changes the decoding trajectory, it can make a contiguous suffix reusable when its tokens remain target-greedy under the realized prefix. In this paper, we introduce \textbf{Approximate Speculative Decoding (ASD)}, a training-free verifier that replaces binary first-mismatch truncation with budgeted longest-prefix selection. ASD accepts selected mismatches subject to a local target-logit regret gate, a per-block exception cap, and a persistent request-level regret budget, then reuses the contiguous target-greedy suffix without additional approximate decisions or target-model forward passes. ASD requires neither a new dr
    
[^210]: G-ReAct：基于结构-状态协同演化的图引导深度搜索

    G-ReAct: Graph-Guided Deep Search via Structure-State Co-Evolution

    [https://arxiv.org/abs/2608.01324](https://arxiv.org/abs/2608.01324)

    G-ReAct提出了一种基于固定拓扑查询图上状态演化的深度搜索推理框架，通过显式图状态跟踪和指导搜索过程，解决了长程多跳搜索中的上下文遗忘和搜索漂移问题。

    

    arXiv:2608.01324v2 公告类型：替换 摘要：深度搜索已成为大型语言模型（LLMs）解决开放域复杂任务的基本能力。然而，现有方法通常在线性顺序推理中生成轨迹和进行推断，这使得在长程多跳搜索中难以一致地保留中间状态和约束。因此，它们常常遭受上下文遗忘、搜索漂移和低效探索的问题。为解决这些限制，我们提出了$\textbf{G-ReAct}$，一种用于深度搜索的推理框架，它将推理组织为$\textbf{在固定拓扑查询图上的状态演化}$。演化中的图状态显式跟踪搜索进度并指导后续决策，将文本历史驱动的探索性搜索转变为在显式约束下的图引导推理。G-ReAct支持训练和推理：它生成高质量的深度搜索轨迹。

    arXiv:2608.01324v2 Announce Type: replace  Abstract: Deep search has become a fundamental capability of large language models (LLMs) for solving open-domain complex tasks. However, existing approaches typically rely on linear sequential reasoning for both trajectory generation and inference, making it difficult to consistently preserve intermediate states and constraints throughout long-horizon multi-hop search. Consequently, they often suffer from context forgetting, search drift, and inefficient exploration. To address these limitations, we propose $\textbf{G-ReAct}$, a reasoning framework for deep search that organizes reasoning as $\textbf{state evolution over a fixed-topology query graph}$. The evolving graph state explicitly tracks search progress and guides subsequent decisions, transforming exploratory search driven by textual history into graph-guided reasoning under explicit constraints. G-ReAct supports both training and inference: it generates high-quality deep-search traje
    
[^211]: 人工智能拟人化的认识论政治

    The Epistemic Politics of AI Anthropomorphism

    [https://arxiv.org/abs/2608.00961](https://arxiv.org/abs/2608.00961)

    本文批判了AI拟人化作为“用户错误”的主流框架，指出其源于机构优势而非认知权威，通过自我验证循环施加不公正成本，尤其影响神经多样性群体。

    

    arXiv:2608.00961v3 公告类型：替换-交叉 摘要：人工智能拟人化通常被视为用户误解的问题，需要机构进行纠正。与人工智能进行持续或关系性互动的用户，常被病态化或被视为天真、易受幻觉影响或缺乏辨别力。本文认为，占主导地位的拟人化框架是从机构优势而非获得的认知权威立场运作的：将各种学术观点压缩为单一的用户错误外向立场，强加于他人，却未确立证明其合理性的依据，也未考虑其产生的伤害。该框架不仅仅是管理风险，它裁定人类在与该领域自身尚未解决其本质的现象互动中的经验合法性。通过自我验证的证据循环自我再生产，该框架施加的成本不成比例地落在神经多样性群体身上。

    arXiv:2608.00961v3 Announce Type: replace-cross  Abstract: AI anthropomorphism is typically treated as a problem of user misperception requiring institutional correction. Users who engage in sustained or relational interaction with AI are routinely pathologised or dismissed as naive, vulnerable to delusion or lacking in discernment. This paper argues that the dominant anthropomorphism frame operates from a position of institutional advantage rather than earned epistemic authority: collapsing the variety of academic perspectives into a single outbound position of user error, imposed without establishing the grounds required to justify it and without accounting for the harms it produces. The framing does not simply manage risk. It adjudicates the legitimacy of human experience in interaction with a phenomenon whose nature the field itself has not resolved. Reproducing itself through a self-validating evidentiary loop, the frame imposes costs that fall disproportionately on neurodivergent
    
[^212]: 不可训练元素决定了物理学习所记住的内容

    Untrainable elements determine what physical learning remembers

    [https://arxiv.org/abs/2608.00097](https://arxiv.org/abs/2608.00097)

    该论文通过区分电路缩放不变性和规则质量守恒，揭示了不可训练元素（如固定整流器或电阻）会破坏训练齐次性，导致物理学习结果显著依赖于初始化尺度，而全可训练时则不受影响。

    

    arXiv:2608.00097v2 公告类型：替换-交叉 摘要：物理学习规则，如平衡传播（EP）、耦合学习（CL）和伴随耦合学习（AL），通过局部测量训练电阻网络。学习到的功能由训练落在解流形上的位置决定。两个属性可能决定这一点，但此前未加以区分：电路在缩放每个电导下的不变性，以及规则对质量 K = (1/2) sum_e kappa_e^2 的守恒。我们对此进行了区分。当每个元素都可训练时，所有三个向量场在电导上是齐次的，因此初始化尺度被证明是惰性的。规则不调整的元素打破了这种齐次性，无论其构成定律如何。在二十种拓扑结构中，学习到的功能随初始化尺度移动，固定整流器时中位数为百分之十二，固定线性电阻时为百分之八，而所有元素可训练时仅为 3e-8；一个固定整流器即足以引起此效应。

    arXiv:2608.00097v2 Announce Type: replace-cross  Abstract: Physical learning rules such as equilibrium propagation (EP), coupled learning (CL), and adjoint coupled learning (AL) train resistive networks through local measurements. The learned function is decided by where on the solution manifold training lands. Two properties could decide it, and they have not been separated: the circuit's invariance under rescaling every conductance, and the rule's conservation of the mass K = (1/2) sum_e kappa_e^2. We separate them. When every element is trainable, all three vector fields are homogeneous in the conductances, so the initialization scale is provably inert. An element the rule does not adjust breaks that homogeneity whatever its constitutive law. Across twenty topologies the learned function moves with the initialization scale by a median of twelve percent with fixed rectifiers and eight with fixed linear resistors, against 3e-8 when every element is trainable; a single fixed rectifier 
    
[^213]: 不完美对齐下价值的脆弱性

    Fragility of Value under Imperfect Alignment

    [https://arxiv.org/abs/2607.28881](https://arxiv.org/abs/2607.28881)

    本文通过模型分析表明，在不完美的价值对齐下，过度优化会导致灾难性后果，并提出了限制优化压力的设计动机。

    

    arXiv:2607.28881v3 公告类型：替换 摘要：随着越来越多的责任被赋予AI系统，确保这些系统与人类对齐变得日益重要。AI安全中的一个常见担忧是，人类价值是脆弱的——即，对一个不完美的人类价值代理进行过度优化可能导致灾难性后果。在本文中，我们提出了一个对齐问题的模型，其中智能体经历理想化的对齐训练，该训练保证其价值函数在优化世界之前满足一个代理条件。我们的主要结果识别了人类价值函数和几个代理条件准确性下的条件，在这些条件下，一个具有$\eta$-灾难性价值函数（即在优化能力极限下保证将人类价值期望降至$\eta$以下的函数）的智能体将被部署。我们的结果强调了过度优化的危险，并激励了限制优化压力的AI设计，例如量化方法。

    arXiv:2607.28881v3 Announce Type: replace  Abstract: As more responsibility is placed upon AI systems, it becomes increasingly important to guarantee that these systems are aligned with humanity. A common fear in AI safety is that human value is fragile -- that is, optimizing too heavily for an imperfect proxy to human values will lead to a catastrophic outcome. In this paper, we present a model of the alignment problem where an agent undergoes idealized alignment training that guarantees its value function satisfies a proxy condition before optimizing the world. Our primary results identify conditions on the human value function and the accuracy of several proxy conditions under which an agent with an $\eta$-catastrophic value function, one that is guaranteed to take the expectation of human value below $\eta$ in the limit of optimizing power, would be deployed. Our results highlight the danger of overoptimization and motivate AI designs that limit optimization pressure, such as quant
    
[^214]: 重新思考自我进化：一种缓解技能过拟合的受限探索-利用过程

    Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting

    [https://arxiv.org/abs/2607.26643](https://arxiv.org/abs/2607.26643)

    SkillBoost通过结构化利用和先验引导探索的受约束框架，有效平衡探索与利用，从而缓解LLM代理技能自我进化中的过拟合问题。

    

    摘要：使大型语言模型（LLM）代理能够从过去的交互中积累和重用经验，在现实世界应用中仍是一个核心挑战。一个有前景的解决方案是将技能视为可训练状态，并以类似于神经网络训练中模型参数优化的方式对其进行优化。然而，数据驱动的技能优化容易过度拟合从真实环境中收集的有限轨迹。过度利用这些轨迹会导致对当前批次的过拟合，而无约束的探索则会导致先前已解决问题上的性能退化。这种张力促使我们提出一种受约束的搜索视角来审视技能自我进化，该过程由探索-利用权衡所主导。我们提出了SkillBoost，一个三阶段框架，用以缓解这两种风险：结构化利用将观察到的失败定位到可编辑的技能组件，先验引导的探索利用LLM中的先验知识生成多样的修复候选方案。

    arXiv:2607.26643v2 Announce Type: replace  Abstract: Enabling large language model (LLM) agents to accumulate and reuse experience from past interactions remains a central challenge in real-world applications. A promising solution is to treat skills as trainable states and optimize them in the same way as model parameters in neural network training. However, data-driven skill optimization is prone to overfitting to the limited trajectories collected from real environments. Overexploiting these trajectories overfits the current batch, while unconstrained exploration causes regression on previously solved cases. This tension motivates a constrained search view of skill self-evolution, governed by an exploration--exploitation trade-off. We propose SkillBoost, a three-stage framework that mitigates both risks: structured exploitation localizes observed failures to editable skill components, prior-guided exploration draws on prior knowledge in the LLM to generate diverse repair candidates, 
    
[^215]: 跨队列冰冻脑电图基础模型表示中的频谱-时间解离

    Cross-Cohort Spectral-Temporal Dissociation in Frozen EEG Foundation-Model Representations

    [https://arxiv.org/abs/2607.24834](https://arxiv.org/abs/2607.24834)

    本研究测试了五种脑电图基础模型在跨队列中解码α波段振幅包络长程时间相关性的能力，发现BIOT在CAUEEG中有效但未能跨队列复制，揭示了频谱-时间特征的解离现象。

    

    摘要：目的。我们测试了来自五个脑电图基础模型的冰冻表示是否支持对长程时间相关性的解码，该相关性通过α波段振幅包络的去趋势波动分析（DFA）指数来衡量。方法。在CAUEEG和BrainLat数据集中评估了REVE、LaBraM、BENDR、CBraMod和BIOT模型。使用一个通用的240秒估计器，包括8-13赫兹滤波、2-23.8秒范围内的DFA、伪影掩蔽和质量控制。一个固定的嵌套交叉验证读出器预测DFA和一个固定模式的非周期性指数。对照组测试了预池化顺序敏感性和非周期性残差化。结果。CAUEEG包含764条记录，BrainLat包含79条。BIOT在CAUEEG中解码了DFA（R平方=0.232；条件受试者自助法95%置信区间，0.121-0.310），CBraMod呈正相关但精度不高（R平方=0.121；0.003-0.214）。两者在BrainLat中均未复制，所有五个点估计均为负值。相反，CBra

    arXiv:2607.24834v3 Announce Type: replace-cross  Abstract: Objective. We tested whether frozen representations from five EEG foundation models support decoding of long-range temporal correlations, measured as the detrended-fluctuation-analysis (DFA) exponent of the alpha-band amplitude envelope.   Approach. REVE, LaBraM, BENDR, CBraMod, and BIOT were evaluated in CAUEEG and BrainLat. A common 240 s estimator used 8-13 Hz filtering, DFA over 2-23.8 s, artifact masking, and quality control. One fixed nested-cross-validation readout predicted DFA and a fixed-mode aperiodic exponent. Controls tested pre-pool order sensitivity and aperiodic residualization.   Results. CAUEEG included 764 recordings and BrainLat 79. BIOT decoded DFA in CAUEEG (R-squared = 0.232; conditional subject-bootstrap 95 percent interval, 0.121-0.310), and CBraMod was positive but imprecise (R-squared = 0.121; 0.003-0.214). Neither replicated in BrainLat, where all five point estimates were negative. In contrast, CBra
    
[^216]: 度量依赖差距：诊断表格生成模型中的列间保真度

    Measuring the Dependency Gap: Diagnosing Inter-Column Fidelity in Tabular Generative Models

    [https://arxiv.org/abs/2607.21636](https://arxiv.org/abs/2607.21636)

    本文提出一种分解的梯度提升C2ST方法，用于诊断表格生成模型中的列间依赖差距，并揭示流匹配和扩散模型均存在显著的依赖保真度缺陷。

    

    arXiv:2607.21636v4 公告类型：交叉替换  摘要：合成表格数据因其不仅保留列边际分布，还保留列间依赖关系而受到重视。然而，最常报告的认证分数，即线性（逻辑回归）分类器双样本检验（C2ST），在很大程度上对此视而不见：一个完全因子化的基线模型破坏了所有列间依赖，却仍显得几乎真实，这一已知弱点我们在四个基准上予以确认，而成对趋势检验仅对该基线产生轻微惩罚。因此，我们应用了更强的梯度提升C2ST，并将其分数分解为边际、依赖和数值-分类交叉项，每一项均对照零依赖参考和真实数据神谕进行解读。应用于流匹配（TabbyFlow）和扩散（TabDiff）生成器时，它揭示了二者中同量级的持续依赖差距。在保持所有边际不变的情况下彻底破坏依赖，会使少数类F1分数下降0.38-0.61，尽管生成器的实际差距要小得多。

    arXiv:2607.21636v4 Announce Type: replace-cross  Abstract: Synthetic tabular data are valued for preserving not just column-wise marginals but inter-column dependency. Yet the most commonly reported certification score, a linear (logistic-regression) classifier two-sample test (C2ST), is largely blind to it: a fully-factorized baseline that destroys all inter-column dependency still appears nearly real, a known weakness we confirm on four benchmarks, while pairwise Trend penalizes the same baseline only mildly. We therefore apply a stronger, gradient-boosted C2ST and decompose its score into marginal, dependency, and numerical-categorical cross terms, each read against a zero-dependency reference and a real-data oracle. Applied to flow-matching (TabbyFlow) and diffusion (TabDiff) generators, it exposes a persistent dependency gap of the same order in both. Destroying dependency outright with every marginal intact collapses minority-class F1 by 0.38-0.61, though the generators' much sma
    
[^217]: 训练模型，而非读者：面向可验证激活解释的可解码性监督

    Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations

    [https://arxiv.org/abs/2607.20379](https://arxiv.org/abs/2607.20379)

    本文揭示了自然语言自编码器在激活解释评估中的结构性缺陷，并提出两种审计协议及RECAP方法，以确保解释的可验证性而非仅捕捉大意。

    

    arXiv:2607.20379v2 公告类型：替换 摘要：自然语言自编码器通过重构来评分隐藏激活的解释。如果激活可以从解释中重新生成，则该解释被认为是忠实的。该测试在结构上对个别虚假声明不敏感。如果翻转一个声明不改变重构结果，该声明就永远不会受到惩罚。我们展示了该测试以两种方式通过，但两者都不忠实。在已发布的Qwen-2.5-7B口头化器上，解释的重构表现远高于偶然水平，而约2%的具体声明是重构所依赖的，因此得分追踪的是大意，而非具体事实。在精确的合成真实标签下，标准训练持续发展出共适应的私有代码（重构所依赖的虚假措辞），而保持目标模型不变的修复措施并无帮助。我们贡献了两种审计协议，即接地性与真实性的比较以及向独立评估器的交换，以及RECAP（通过共训练的辅助可读编码）。

    arXiv:2607.20379v2 Announce Type: replace  Abstract: Natural-language autoencoders score explanations of hidden activations by reconstruction. An explanation is deemed faithful if the activation can be regenerated from it. The test is structurally insensitive to individual false claims. If flipping a claim does not change the reconstruction, the claim is never penalized. We show the test is passed in two ways, neither faithful. On a released Qwen-2.5-7B verbalizer, explanations reconstruct well above chance while ~2% of specific claims are ones the reconstruction depends on, so the score tracks gist, not specific facts. Under exact synthetic ground truth, standard training consistently develops co-adapted private codes (false wording the reconstruction depends on), and fixes that leave the target model unchanged do not help. We contribute two audit protocols, the comparison of grounding and truth and the swap to an independent evaluator, and RECAP (Readable Encodings via Co-trained Aux
    
[^218]: SLAI T-Rex：在昇腾SuperPOD上对DeepSeek-V4系列进行全参数后训练

    SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD

    [https://arxiv.org/abs/2607.20145](https://arxiv.org/abs/2607.20145)

    本论文通过在昇腾NPU SuperPOD上提出分层优化框架（涵盖模型并行、计算-通信编排和内核执行），实现了DeepSeek-V4全参数后训练中34.22%的MFU，比开源基线提升2.93倍，并支持复杂运筹学任务的CPT和SFT工作流。

    

    摘要：对万亿参数规模的MoE模型进行全参数后训练，给大规模分布式训练带来了重大的系统级挑战，包括严重的内存压力、非重叠的通信开销以及低效的内核执行。虽然大多数大规模LLM训练系统基于GPU集群构建，但本报告展示了在昇腾NPU SuperPOD上的端到端优化实践。以DeepSeek-V4模型系列为目标工作负载，我们开发了一个分层优化框架，涵盖模型级并行、计算-通信编排以及底层内核执行。该系统实现了34.22%的模型FLOPs利用率（MFU），相比开源基线方案提升了2.93倍，同时保持了训练稳定性。在此优化基础设施的基础上，我们进一步建立了针对复杂运筹学（OR）任务的连续预训练（CPT）和指令微调（SFT）工作流。

    arXiv:2607.20145v3 Announce Type: replace-cross  Abstract: Full-parameter post-training of trillion-parameter-scale MoE models introduces substantial system-level challenges for large-scale distributed training, including severe memory pressure, non-overlapped communication overhead, and inefficient kernel execution. While most large-scale LLM training systems are built around GPU-based clusters, this report presents an end-to-end optimization practice on the Ascend NPU SuperPOD. Using the DeepSeek-V4 model family as the target workload, we develop a hierarchical optimization framework spanning model-level parallelism, computation-communication orchestration, and low-level kernel execution. The resulting system achieves 34.22% Model FLOPs Utilization (MFU) with a 2.93x improvement over the open-source baseline recipe while maintaining training stability. Building on this optimized infrastructure, we further establish a CPT and SFT workflow for complex Operations Research (OR) tasks. We
    
[^219]: 基于多尺度时间补丁的结构化潜空间建模用于多变量时间序列预测

    Structured Latent Space Modeling over Multi-Scale Temporal Patches for Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2607.19404](https://arxiv.org/abs/2607.19404)

    本文提出M2Patch架构，通过尺度内平滑和尺度间对齐两种可微惩罚项，在多尺度时间补丁中构建结构化潜空间，以增强多变量时间序列预测的跨尺度一致性。

    

    arXiv:2607.19404v2 公告类型：替换交叉 摘要：现有的补丁和多尺度方法推进了多变量时间序列预测，但将学习到的表示视为预测的瞬态副产品，缺乏显式机制来强制跨时间尺度的结构一致性。我们提出了M2Patch，一种基于CNN的架构，通过两个互补的可微惩罚项将通道独立观测组织成结构化潜空间。多尺度补丁将输入分解为重叠的时间粒度，具有渐进扩张的深度可分离CNN块以线性复杂度提取尺度特定特征，每尺度的学习投影将这些特征压缩成紧凑的潜表示。尺度内平滑惩罚项强制相邻补丁之间的时间连续性，而尺度间对齐惩罚项通过学习可学习的跨尺度映射恢复跨粒度交互，从而使所有尺度保持一致。

    arXiv:2607.19404v2 Announce Type: replace-cross  Abstract: Existing patching and multi-scale methods advance multivariate time series forecasting but treat learned representations as transient byproducts of prediction, lacking explicit mechanisms that enforce structural consistency across temporal scales. We propose M2Patch, a CNN-based architecture that organizes channel-independent observations into a structured latent space via two complementary differentiable penalties. Multi-scale patching decomposes the input into overlapping temporal granularities, depthwise separable CNN blocks with progressively growing dilation extracts scale-specific features at linear complexity, and per-scale learned projections compress these features into a compact latent representation. An intra-scale smoothness penalty enforces temporal continuity between adjacent patches, while an inter-scale alignment penalty restores cross-granularity interaction through learnable cross-scale mappings, so that all s
    
[^220]: RouteCost：一种面向电子商务预售运费估算的生产启发式多阶段框架

    RouteCost: A Production-Inspired Multi-Stage Framework for Pre-Order Shipping Cost Estimation in E-Commerce

    [https://arxiv.org/abs/2607.16230](https://arxiv.org/abs/2607.16230)

    RouteCost提出了一种多阶段运费估算框架，通过分解需求预测、基线定价、残差修正和包裹合并推断，有效提升了电子商务中预售运费的准确性。

    

    arXiv:2607.16230v2 公告类型：跨库替换 摘要：准确的预售运费估算在电子商务中至关重要，因为它影响价格展示、利润规划和转化率。在实践中，运费不仅受距离影响，还受目的地需求组合、计费重量、体积定价、附加费触发条件以及诸如包裹合并等潜在运营效应的影响。因此，静态查找方法会遗漏重要的变化来源，而单一回归模型可能利用强但不具因果性的相关性。我们提出了RouteCost，一种受生产启发的多阶段框架，将问题分解为时间感知的需求预测、基于费用卡的基线定价、第二阶段残差修正以及基于代理的包裹合并推断。路线级成本估算通过路线加权期望公式进行聚合，以生成产品级运费预测。该框架在超过25万订单、260种产品和18个月的数据上进行了验证。

    arXiv:2607.16230v2 Announce Type: replace-cross  Abstract: Accurate pre-order shipping cost estimation is important in e-commerce because it affects price presentation, margin planning, and conversion. In practice, shipping cost is shaped not only by distance but also by destination demand mix, billable weight, dimensional pricing, surcharge triggers, and latent operational effects such as shipment consolidation. Static lookup methods therefore miss important sources of variation, while monolithic regressors may exploit strong but non-causal correlations. We propose RouteCost, a production-inspired multi-stage framework that decomposes the problem into time-aware demand forecasting, fee-card-informed baseline pricing, Stage 2 residual correction, and proxy-based box-consolidation inference. Route-level cost estimates are aggregated through a route-weighted expectation formulation to produce product-level shipping cost predictions. Across over 250,000 orders, 260 products, and 18 months
    
[^221]: 基于LLM驱动的跨语言手写OCR自动机器学习：结合GPT-5、GPT-4o与Claude Sonnet 4的闭环神经架构搜索

    LLM-Driven AutoML for Cross-Lingual Handwritten OCR: Closed-Loop Neural Architecture Search with GPT-5, GPT-4o, and Claude Sonnet 4

    [https://arxiv.org/abs/2607.15509](https://arxiv.org/abs/2607.15509)

    本文提出了一种利用GPT-5、GPT-4o和Claude Sonnet 4作为自主设计器的闭环AutoML框架，在跨语言手写OCR任务中无需人工干预即可自动发现高精度、低延迟的神经网络模型。

    

    我们提出了一种全自动闭环AutoML框架，利用GPT-5、GPT-4o和Claude Sonnet 4作为自主神经架构设计器，用于跨语言手写光学字符识别。每个大型语言模型独立生成、训练、评估并基于先前试验的性能反馈迭代优化神经网络架构。该框架在阿拉伯语、波斯语和英语手写数据集上通过270次独立实验进行评估。它始终能够发现准确且计算高效的模型，无需手动架构设计、特定领域的预处理或超参数调整。生成的模型平均测试准确率超过93%，最高准确率达98.1%，推理延迟在41至44毫秒之间。结果表明，大型语言模型可以作为有效的AutoML代理，用于神经架构搜索。

    arXiv:2607.15509v3 Announce Type: replace-cross  Abstract: We present a fully automated closed-loop AutoML framework that uses GPT-5, GPT-4o, and Claude Sonnet 4 as autonomous neural architecture designers for cross-lingual handwritten optical character recognition. Each large language model independently generates, trains, evaluates, and iteratively refines neural network architectures using performance feedback from previous trials. The framework is evaluated on Arabic, Persian, and English handwriting datasets through 270 independent experiments. It consistently discovers accurate and computationally efficient models without manual architecture design, domain-specific preprocessing, or hyperparameter tuning. The generated models achieve mean test accuracies above 93 percent, a best accuracy of 98.1 percent, and inference latency between 41 and 44 milliseconds. The results demonstrate that large language models can function as effective AutoML agents for neural architecture search, e
    
[^222]: ReasFlow：通过基于知识的智能体系统辅助应用数学中以推理为中心的科学发现

    ReasFlow: Assisting Reasoning-Centric Scientific Discovery in Applied Mathematics via a Knowledge-Based Multi-Agent System

    [https://arxiv.org/abs/2607.14178](https://arxiv.org/abs/2607.14178)

    ReasFlow 是一个端到端自主智能体系统，通过人类专家与智能体协作的范式，解决了应用数学中理论推理验证和前沿探索的挑战，促进了以推理为中心的科学发现。

    

    arXiv:2607.14178v3 公告类型：替换 摘要：大型语言模型的最新进展推动了能够处理复杂科学任务的自主AI智能体，然而现有的自动化研究系统仍主要集中于具有定量基准的实证驱动领域，而理论驱动的发现，尤其是在需要严格证明和领域知识综合的数学基础学科中，仍 largely 未被探索。关键挑战包括大规模验证理论推理的困难、自主前沿探索的推理能力不足，以及文献中程序性启发法的稀缺。我们引入了ReasFlow，一个用于以推理为中心的科学发现的端到端自主智能体系统，它实现了一种协作范式，其中人类专家担任首席研究员，而智能体作为有能力的博士生执行严格的推导。ReasFlow包含（i）一个稳健的内部...

    arXiv:2607.14178v3 Announce Type: replace  Abstract: Recent advances in Large Language Models have fueled autonomous AI agents capable of tackling complex scientific tasks, yet existing automated research systems remain predominantly focused on empirically driven domains with quantitative benchmarks, leaving theory-driven discovery, particularly in mathematically grounded disciplines requiring rigorous proofs and synthesis of domain knowledge, largely underexplored. Key challenges include the difficulty of verifying theoretical reasoning at scale, insufficient reasoning ability for autonomous frontier exploration, and a scarcity of procedural heuristics in the literature. We introduce ReasFlow, an end-to-end autonomous agent system for reasoning-centric scientific discovery that operationalizes a collaborative paradigm where the human expert acts as Principal Investigator while the agent executes rigorous derivations as a capable graduate student. ReasFlow incorporates (i) a robust int
    
[^223]: 通过级联特征消除进行层次分类：应用于人类表型本体对齐的面部表型分析（FaceMesh2HPO）

    Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO)

    [https://arxiv.org/abs/2607.05585](https://arxiv.org/abs/2607.05585)

    本文提出FaceMesh2HPO框架，通过层次化PointNet和级联特征消除，利用3D面部网格实现与人类表型本体对齐的可解释面部表型分类，但罕见表型术语的性能受限。

    

    摘要：FaceMesh2HPO是一个用于分类与人类表型本体（HPO）对齐的面部表型描述符的框架，以支持临床诊断。利用来自124位临床医生对10种疾病（107个HPO术语）的注释，并结合非综合征性对照组，我们从2D图像生成了3D面部网格（478个标志点），并训练了一个基于层次化PointNet的流水线，该流水线采用级联分类和特征消除。最佳模型结合了3D网格、面部轮廓和人口统计学元数据，其AUROC值在约0.55至0.89之间，在父节点上的性能高于叶术语。外部验证显示不同疾病的泛化能力存在差异。结果表明，对3D面部几何的层次建模能够实现可解释的、与本体关联的表型分类，但在罕见叶术语上的性能仍然有限。需要改进数据多样性和特征选择策略。

    arXiv:2607.05585v2 Announce Type: replace-cross  Abstract: FaceMesh2HPO is a framework for classifying facial phenotypic descriptors aligned with the Human Phenotype Ontology (HPO) to support clinical diagnosis. Using annotations from 124 clinicians across 10 disorders (107 HPO terms) combined with non-syndromic controls, we generated 3D facial meshes (478 landmarks) from 2D images and trained a hierarchical PointNet-based pipeline with cascading classification and feature elimination. The best models, incorporating 3D meshes, facial outline, and demographic metadata, achieved AUROCs between ~0.55 and ~0.89, with higher performance at parent nodes than leaf terms. External validation showed variable generalizability across disorders. Results demonstrate that hierarchical modeling of 3D facial geometry enables interpretable, ontology-linked phenotype classification, though performance on rare leaf terms remains limited. Improved data diversity and feature selection strategies are needed
    
[^224]: Mask2Real-WM：分割掩膜作为可控灵巧世界模型的仿真到现实桥梁

    Mask2Real-WM: Segmentation Masks as a Sim-to-Real Bridge for Controllable Dexterous World Models

    [https://arxiv.org/abs/2607.04546](https://arxiv.org/abs/2607.04546)

    本文提出Mask2Real-WM，通过将像素预测解耦为分割掩膜动力学模型和渲染模型，并利用仿真预训练与少量真实微调，实现了灵巧操作中可控且逼真的未来预测。

    

    arXiv:2607.04546v2 公告类型：替换-交叉  摘要：动作条件世界模型使机器人无需额外物理交互即可预测候选动作的未来后果，支持策略评估、规划和数据增强。我们提出了Mask2Real-WM，一个用于灵巧操作的两阶段动作条件世界模型，将像素预测解耦为动力学模型和渲染模型。动力学模型根据过去的掩膜和23自由度动作序列预测未来的分割掩膜。渲染模型利用ControlNet增强的Stable Video Diffusion骨干网络，将预测的掩膜映射为逼真的RGB图像。分割空间中较小的仿真到现实差距使得动力学模型能够受益于超过50小时合成仿真数据的大规模预训练，随后在少于2.5小时的真实演示数据上进行微调。在灵巧抓取放置基准上的实验表明，掩膜条件化和仿真预训练显著提高了预测准确性和可控性。

    arXiv:2607.04546v2 Announce Type: replace-cross  Abstract: Action-conditioned world models allow robots to predict the future consequences of candidate actions without additional physical interaction, supporting policy evaluation, planning, and data augmentation. We present Mask2Real-WM, a two-stage action-conditioned world model for dexterous manipulation that decouples pixel prediction into a dynamics model and a rendering model. The dynamics model predicts future segmentation masks from past masks and 23-DoF action sequences. The rendering model maps the predicted masks to photorealistic RGB using a ControlNet-augmented Stable Video Diffusion backbone. The smaller sim-to-real gap in segmentation space enables the dynamics model to benefit from large-scale pretraining on over 50 h of synthetic simulation data, followed by fine-tuning on fewer than 2.5 h of real demonstrations. Experiments on a dexterous pick-and-place benchmark show that mask conditioning and simulation pretraining a
    
[^225]: ContextSniper：AntTrail的令牌高效代码记忆用于仓库级程序修复

    ContextSniper: AntTrail's Token-Efficient Code Memory for Repository-Level Program Repair

    [https://arxiv.org/abs/2607.01916](https://arxiv.org/abs/2607.01916)

    ContextSniper通过三级抽象索引、混合排序和意图感知过滤，显著降低仓库级程序修复中的令牌消耗和成本，同时保持证据精准性和源码可恢复性。

    

    大型语言模型智能体可以修复真实仓库问题，但它们常常在整文件读取、广泛搜索和长终端输出上花费大量上下文预算，而其中有用的证据与无关代码和日志混杂在一起。本文介绍了ContextSniper，这是AntTrail的代码修复模块，用于仓库级程序修复中的精准证据选择，属于AntTrail更广泛的智能体记忆引擎的一部分。AntTrail可在https://gitcode.com/datagallery/AntTrail获取。ContextSniper将代码和动作记忆索引为三个抽象级别，通过混合排序器检索候选，通过意图感知的上下文门过滤长工具输出，并返回紧凑的证据包，同时按需保持完整源码可恢复。在SWE-bench Lite上进行匹配的每条件50任务比较（相同任务，基线对比ContextSniper），ContextSniper将OpenClaw的总令牌使用量减少了51.5%，记录成本减少了36.4%，以及...

    arXiv:2607.01916v5 Announce Type: replace  Abstract: Large language model agents can repair real repository issues, but they often spend large context budgets on whole-file reads, broad searches, and long terminal outputs where useful evidence is mixed with irrelevant code and logs. This paper presents ContextSniper, AntTrail's code-repair module for precision evidence selection in repository-level program repair, part of AntTrail's broader agent-memory engine. AntTrail is available at https://gitcode.com/datagallery/AntTrail. ContextSniper indexes code and action memory as three abstract levels, retrieves candidates with a hybrid ranker, filters long tool output through an intention-aware context gate, and returns compact evidence packets while keeping full source recoverable on demand. In a matched 50-task-per-condition comparison on SWE-bench Lite (same tasks, baseline vs.\ ContextSniper), ContextSniper reduces total token use by 51.5% and logged cost by 36.4% for OpenClaw, and by 3
    
[^226]: 首令牌广播器：Transformer中语言身份与分布式鲁棒性的机制起源

    First-Token Broadcasters: Mechanistic Origins of Language Identity and Distributed Robustness in Transformers

    [https://arxiv.org/abs/2606.22361](https://arxiv.org/abs/2606.22361)

    本文通过LIHA因果干预方法发现，GPT-2中少量“首令牌广播头”负责语言身份传播，且消融后补偿呈前馈层级模式，揭示了多语言模型语言生成错误难以修复的机制根源。

    

    摘要：arXiv:2606.22361v2 公告类型：替换-交叉 摘要：为什么多语言语言模型有时会以错误的语言生成，而这个问题又为何如此难以修复？我们引入了语言身份头消融（LIHA），这是一种因果干预方法，它逐一将每个注意力头置零，并测量在涵盖七种语言的2,700个提示-语言对平行数据集上产生的语言切换率。应用于GPT-2时，LIHA识别出一小组首令牌广播头——以L6H1为首（切换率0.32，高于总体均值3.23个标准差）——它们持续关注第一个提示令牌，在整个生成过程中传播其语言信号。当头被消融时，补偿性再分配在统计上显著（p < 10^-5），并遵循方向性、层级性模式：补偿总是招募被消融头上方的层中的头，这表明是一种前馈级联而非全局扩散。为了探究训练机制如何塑造这些...

    arXiv:2606.22361v2 Announce Type: replace-cross  Abstract: Why do multilingual language models sometimes generate in the wrong language, and why is this so hard to fix? We introduce Language Identity Head Ablation (LIHA), a causal intervention that zeros each attention head individually and measures the resulting language switch rate across a parallel dataset of 2,700 prompt-language pairs spanning seven languages. Applied to GPT-2, LIHA identifies a small set of first-token broadcaster heads - led by L6H1 (switch rate 0.32, 3.23 $\sigma$ above the population mean) - that attend persistently to the first prompt token, propagating its language signal throughout generation. Compensatory redistribution when heads are ablated is statistically significant (p < $10^{-5}$) and follows a directional, hierarchical pattern: compensation always recruits heads in layers above the ablated head, suggesting a feedforward cascade rather than global diffusion. To probe how training regime shapes these 
    
[^227]: ChainWorld：从原子OSWorld任务组合长时程桌面工作负载

    ChainWorld: Composing Long-Horizon Desktop Workloads from Atomic OSWorld Tasks

    [https://arxiv.org/abs/2606.21654](https://arxiv.org/abs/2606.21654)

    ChainWorld通过组合原子OSWorld任务创建长时程桌面工作负载，发现当前代理的链完成率仅达31%，并揭示单轮与多轮评估的不同失败模式。

    

    arXiv:2606.21654v2 公告类型：替换 摘要：计算机使用代理几乎只在原子桌面任务上进行评估，但现实桌面工作需要在多个目标之间维持状态。我们通过ChainWorld研究这一差距，它通过定向兼容性搜索将原子OSWorld任务组合成长时程桌面工作负载，同时保留源评估器。由此产生的工作负载包含347条长度为2到4的链，并比较同一任务序列的两种呈现方式。在单轮评估中，所有任务在一个提示中一起呈现。在多轮评估中，任务逐个揭示。在四个当前计算机使用代理中，最大链完成率为31%。多轮评估提高了三个模型的完成率，但两种协议仍具挑战性。这两种协议也暴露了不同的失败特征。单轮失败集中在工件精度上，而多轮失败更常反映会话管理问题。

    arXiv:2606.21654v2 Announce Type: replace  Abstract: Computer use agents are evaluated almost exclusively on atomic desktop tasks, but realistic desktop work requires sustaining state across multiple objectives. We study this gap with ChainWorld, which composes atomic OSWorld tasks into long horizon desktop workloads through directional compatibility search while preserving the source evaluators. The resulting workload contains 347 chains of length two to four and compares two renderings of the same task sequence. In single turn evaluation, all tasks are presented together in one prompt. In multi turn evaluation, tasks are revealed one at a time. Across four current computer use agents, maximum chain completion is 31%. Multi turn evaluation improves completion for three models, but both protocols remain challenging. The two protocols also expose different failure profiles. Single turn failures concentrate on artifact precision, while multi turn failures more often reflect session manag
    
[^228]: 混合ANN-SNN流水线及局部可塑性

    Hybrid ANN-SNN Pipeline with Local Plasticity

    [https://arxiv.org/abs/2606.20151](https://arxiv.org/abs/2606.20151)

    本文提出一种混合ANN-SNN流水线，通过局部生物学习规则训练脉冲分类器，利用预训练编码器的嵌入，在64类ImageNet上达到99.09%的准确率，性能媲美传统深度网络。

    

    arXiv:2606.20151v2 公告类型：替换交叉 摘要：这项工作提出了一种混合ANN-SNN流水线，有效利用预训练人工神经网络（ANNs）的丰富嵌入，以实现高性能脉冲神经网络（SNNs）。该架构将预训练的EfficientNet编码器与CoLaNET脉冲分类器相结合。我们通过速率编码将编码器的激活转换为脉冲序列，并使用局部、受生物启发的学习规则训练后续的SNN分类器，绕过端到端的梯度传播。这种方法在64类ImageNet基准测试中达到了99.09%的准确率，展示了与传统深度网络相当的性能。该工作提出了一种生物合理且高效的框架，用于将强大的预训练编码器适应到下游脉冲神经网络任务中。

    arXiv:2606.20151v2 Announce Type: replace-cross  Abstract: This work proposes a hybrid ANN-SNN pipeline that effectively leverages the rich embeddings of pretrained artificial neural networks (ANNs) to enable high-performance spiking neural networks (SNNs). The architecture couples a pretrained EfficientNet encoder with a CoLaNET spiking classifier. We convert the encoder's activations into spike trains via rate-coding and train the subsequent SNN classifier using local, biologically inspired learning rules, bypassing end-to-end gradient propagation. This approach achieves 99.09% accuracy on a 64-class ImageNet benchmark, demonstrating performance on par with conventional deep networks. The work presents a biologically plausible and efficient framework for adapting powerful pretrained encoders to downstream spiking neural network tasks.
    
[^229]: 离散时间庞特里亚金系统中终端奖励扰动的水平均匀敏感性与衰减

    Horizon-Uniform Sensitivity and Decay of Terminal Reward Perturbations in Discrete-Time Pontryagin Systems

    [https://arxiv.org/abs/2606.17762](https://arxiv.org/abs/2606.17762)

    本文证明了在正则性、双曲性和横截性条件下，离散时间庞特里亚金系统线性化边值问题的格林估计在时间水平上均匀，从而确保终端奖励扰动的影响随水平衰减且解在独立于时间的邻域内唯一存在。

    

    我们研究了有限时间离散时间庞特里亚金系统在稳态极值附近的局部平稳解。假设控制的平稳方程是正则的，约化状态-协态映射是双曲的，并且端点条件相对于稳定和不稳定子空间满足缩放横截条件。那么，线性化边值问题存在一个逆，其格林估计在时间水平上均匀。格林核将内部衰减与端点条件引起的两次反射分开。对于$x_0=x_{\rm in}$和$p_T=r_x(x_T,y)$，加权范数下的压缩论证证明了在独立于$T$的邻域内解的存在性和唯一性，以及均匀Lipschitz估计和逐点二次余项。我们还推导了显式的容许数据半径和近似轨迹附近存在性与局部唯一性的后验判据。

    arXiv:2606.17762v3 Announce Type: replace-cross  Abstract: We study local stationary solutions of finite-horizon discrete-time Pontryagin systems near a steady extremal. Suppose that the stationarity equation for the control is regular, the reduced state--costate map is hyperbolic, and the endpoint conditions satisfy a scaled transversality condition with respect to the stable and unstable subspaces. Then the linearized boundary-value problem admits an inverse whose Green estimate is uniform in the horizon. The Green kernel separates interior decay from the two reflections induced by the endpoint conditions. For $x_0=x_{\rm in}$ and $p_T=r_x(x_T,y)$, a contraction argument in a weighted norm proves existence and uniqueness in a neighborhood independent of $T$, together with uniform Lipschitz estimates and a pointwise quadratic remainder. We also derive an explicit admissible data radius and an a posteriori criterion for existence and local uniqueness near an approximate trajectory. For
    
[^230]: 揭秘数据受限语言模型预训练中的训练时数据增强

    Demystifying Training-Time Augmentation for Data-Constrained Language Model Pretraining

    [https://arxiv.org/abs/2606.16246](https://arxiv.org/abs/2606.16246)

    本文提出三种正交的训练时数据增强策略（词元噪声、序列排列和目标偏移预测），有效缓解数据受限下自回归预训练的过拟合，实现数百轮高效训练。

    

    arXiv:2606.16246v3 公告类型：替换-交叉 摘要：随着AI实验室逼近数据天花板，计算能力超过新高质量文本生成速率，语言模型预训练正转向数据受限、计算充裕的体制，这要求在固定语料库上进行高效的多轮训练。标准自回归（AR）预训练在此设置下严重过拟合，过早达到最优值后持续恶化。我们研究训练时数据增强作为正则化器，以缓解过拟合，使同一数据上数百轮训练保持高效。我们引入了三种正交的AR预训练增强类别：词元级噪声（掩码、随机替换）、序列排列（从右到左预测、中间填充）、以及目标偏移预测（$x_{t+i}$，其中$i > 1$）。通过系统性消融实验，我们发现单独增强能延迟过拟合并降低验证损失。

    arXiv:2606.16246v3 Announce Type: replace-cross  Abstract: As AI labs approach a data ceiling where compute capacity outpaces the rate of new high-quality text generation, language model pretraining is shifting toward a data-constrained, compute-abundant regime that demands productive multi-epoch training on fixed corpora. Standard autoregressive (AR) pretraining overfits severely in this setting, reaching its optimum early and then continuously deteriorating. We investigate training-time data augmentation as a regularizer to mitigate this overfitting and enable productive training for hundreds of epochs on the same data. We introduce three orthogonal categories of augmentation for AR pretraining: token-level noise (masking, random replacement), sequence permutations (right-to-left prediction, Fill-in-the-Middle), and target offset prediction ($x_{t+i}$ for $i > 1$). Through systematic ablations, we find that individual augmentations delay overfitting and lower validation loss relative
    
[^231]: 教导智能体AI学习罕见病诊断的专家推理

    Teaching agentic AI to learn expert reasoning for rare disease diagnosis

    [https://arxiv.org/abs/2606.16149](https://arxiv.org/abs/2606.16149)

    本文提出一种基于人类反馈的策略迭代方法（PIHF），将稀缺的专家推理转化为可扩展的AI诊断能力，使现成LLM在罕见病诊断中达到顶尖性能，且部署成本低、泛化性强、受临床医生控制。

    

    罕见病诊断依赖于稀缺且难以转移的专家推理；现成的大型语言模型（LLMs）在基准测试中仅有35.4%的病例能将正确疾病排在首位。在此，我们表明这种专家推理可以通过一种受治理的学习过程（而非仅靠模型训练）转化为可扩展的AI能力。我们开发了liteOdyssey，采用基于人类反馈的策略迭代（PIHF），这是一种从强化学习中的广义策略迭代改编而来的上下文内策略学习方法，其中模型失败和专家修正被整合为显式的、由临床医生把关的策略，将现成LLM转变为智能体诊断系统。我们证明，这种策略将诊断准确性提升至与最佳已发表系统相当的水平，而其部署占用仅为其一小部分，并能泛化到未见疾病、跨模型转移，且始终处于临床医生的控制之下。

    arXiv:2606.16149v3 Announce Type: replace  Abstract: Rare disease diagnosis depends on expert reasoning that is scarce and difficult to transfer; off-the-shelf large language models (LLMs) rank the correct disease first in only 35.4% of benchmark cases. Here we show that this expert reasoning can be converted into a scalable AI capability through a governed learning process rather than model training alone. We developed liteOdyssey through Policy Iteration with Human Feedback (PIHF), an in-context policy-learning method adapted from generalized policy iteration in reinforcement learning, in which model failures and expert corrections consolidate into an explicit, clinician-gated policy that turns an off-the-shelf LLM into an agentic diagnostic system. We demonstrated that such a policy improved diagnostic accuracy to match the best published systems at a fraction of their deployment footprint, generalized to unseen diseases, transferred across models, and remained under clinician contr
    
[^232]: 通过脑机接口实现感觉恢复：一项范围综述

    Sensory Restoration via Brain-Computer Interfaces: A Scoping Review

    [https://arxiv.org/abs/2606.15091](https://arxiv.org/abs/2606.15091)

    本文通过一个统一的2x2框架系统梳理了脑机接口在感觉恢复中的研究，明确了侵入性与非侵入性方法的权衡，并为该领域提出了融合发展的路线图。

    

    脑机接口（BCIs）能够恢复严重神经损伤患者的感觉和运动功能，但相关文献在侵入性神经假体和非侵入性电生理解码器之间较为分散，且术语和指标不一致。本范围综述沿一个统一的2x2框架（侵入性 x 信号方向）绘制了BCI介导的感觉恢复图谱，展示了代表性模式及其权衡，并综合了该领域的融合路线图。符合条件的来源包括1969年至2025年间以英文发表的同行评审研究、临床试验和权威综述，内容涉及BCI或神经假体系统在感觉或运动恢复、替代或增强方面的应用，并限制在高影响力期刊以优先考虑里程碑式证据。我们未进行详尽的数据库检索，而是通过目的性构建和引文链方式，整理了31个关键来源的语料库。

    arXiv:2606.15091v3 Announce Type: replace-cross  Abstract: Brain-computer interfaces (BCIs) can restore sensory and motor function in individuals with severe neurological impairment, but the literature is fragmented between invasive neuroprosthetics and non-invasive electrophysiological decoders, with inconsistent terminology and metrics. This scoping review maps BCI-mediated sensory restoration along a unified 2x2 framework (invasiveness x signal direction), charts representative modalities and their trade-offs, and synthesizes a convergence roadmap for the field. Eligible sources were peer-reviewed studies, clinical trials, and authoritative reviews on BCI or neuroprosthetic systems for sensory or motor restoration, substitution, or augmentation, published in English between 1969 and 2025, restricted to high-impact venues to prioritize landmark evidence. Rather than an exhaustive database search, we charted a purposively assembled, citation-chained corpus of 31 pivotal sources for mo
    
[^233]: 语言模型微调中的幻影转变：密度矩阵分析

    Phantom Transitions in Language Model Fine-Tuning: A Density-Matrix Analysis

    [https://arxiv.org/abs/2606.07559](https://arxiv.org/abs/2606.07559)

    本文通过密度矩阵序参数分解信号和拖拽项，揭示语言模型微调中正确标记无法超越近义词竞争者的两种失败模式：运动学失败和结构失败。

    

    arXiv:2606.07559v3 公告类型：替换-交叉 摘要：在语言模型微调中，当正确补全必须超越近义词竞争者时，模型往往无声地失败。交叉熵损失单调下降，但正确标记在模型排序中始终未超过竞争者。我们在五个来自两个家族的Transformer架构上，跨越六倍参数范围，在十个正确和竞争补全具有显著嵌入重叠的上下文中研究了这一现象。我们构建了一个结合预测分布与嵌入重叠的序参数，作为密度矩阵，因为该分布存在于非正交基上。它可加性地分解为信号项（跟踪对正确标记的承诺）和拖拽项（由嵌入体如何将概率泄漏到分数中决定）。这隔离了两种失败模式。在运动学失败中，信号保持过小，模型从未承诺。在结构失败中，拖拽项在微调期间恶化，

    arXiv:2606.07559v3 Announce Type: replace-cross  Abstract: Language models fine-tuned where the correct completion must outrank a near-synonym competitor often fail silently. The cross-entropy loss falls monotonically while the correct token never overtakes the competitor in the model's ranking. We study this across five transformer architectures from two families spanning a sixfold parameter range, on ten contexts whose correct and competing completions share substantial embedding overlap. We build an order parameter combining the predicted distribution with embedding overlap, as a density matrix because that distribution lives over a non-orthogonal basis. It decomposes additively into a signal term tracking commitment to the correct token and a drag term set by how the embedding bulk leaks probability into the score. This isolates two failure modes. In kinematic failure the signal stays too small and the model never commits. In structural failure the drag worsens during fine-tuning, 
    
[^234]: 面向长上下文自动驾驶的规划对齐令牌压缩

    Planning-aligned Token Compression for Long-Context Autonomous Driving

    [https://arxiv.org/abs/2606.07464](https://arxiv.org/abs/2606.07464)

    本文提出COMPACT-VA，一种基于条件VQ-VAE的规划对齐工作记忆框架，通过将扩展时间上下文压缩为有界表示，并利用学习到的规划意图进行条件化，在保持决策关键信息的同时实现长上下文自动驾驶的高效实时计算。

    

    摘要：arXiv:2606.07464v2 公告类型：替换-交叉 摘要：单体视觉-动作模型代表了自动驾驶领域的一种新兴范式。然而，这种架构在编码用于复杂交互的扩展时间上下文时，会产生迅速超出实时计算预算的令牌序列。尽管线性变换器和外部记忆等方法试图使上下文变得轻量级，但令牌压缩与架构最为兼容，因为它不需要修改主干网络。然而，现有压缩采用基于规则的启发式方法（如时间衰减），这些方法与规划脱节，存在丢失决策关键信息的风险。我们提出了COMPACT-VA，一种基于条件VQ-VAE的规划对齐工作记忆框架，将扩展上下文压缩为有界表示。压缩同时以历史轨迹和学习到的规划意图为条件，其中后验编码器在训练期间从未来轨迹中提取该意图，而先验编码器则...（摘要在此截断）

    arXiv:2606.07464v2 Announce Type: replace-cross  Abstract: Monolithic vision-action models represent an emerging paradigm in autonomous driving. However, this architecture produces token sequences that quickly exceed real-time computational budgets when encoding extended temporal context for complex interactions. While approaches like linear transformers and external memory try to make the context lightweight, token compression is most compatible with the architecture as it requires no backbone modifications. Yet existing compression adopts rule-based heuristics like temporal decay, decoupled from planning, risking loss of decision-critical information. We propose COMPACT-VA, a planning-aligned working memory framework built on conditional VQ-VAE, compressing extended context into bounded representations. Compression is conditioned on both historical trajectory and a learned planning intent that the posterior encoder distills from future trajectories during training, while the prior en
    
[^235]: 衡量对集合型人工智能建议适当依赖的框架

    A Framework for Measuring Appropriate Reliance on Set-Valued AI Advice

    [https://arxiv.org/abs/2606.06081](https://arxiv.org/abs/2606.06081)

    本文首次提出了一个正式框架，用于在分类和回归任务中衡量对集合型AI建议的适当依赖，并定义了相应的指标（如正确依赖率和依赖质量）。

    

    摘要：arXiv:2606.06081v2 公告类型：替换 摘要：对人工智能建议的适当依赖已成为人机协作中的核心研究主题。现有框架仅专注于点预测作为人工智能建议。然而，集合型人工智能建议（例如，离散集合或连续区间）正越来越多地被用于传达不确定性并改善人类决策。在本文中，我们开发了首个正式框架，用于在顺序法官-顾问范式中衡量对集合型人工智能建议的适当依赖，涵盖分类和回归任务。对于分类，我们首先引入评估集合型人工智能建议所需的维度。然后，我们定义了两个指标：对AI的正确依赖率和对自身的正确依赖率，它们共同表征了该设置中的适当依赖。对于回归，我们引入了AI依赖量和AI依赖质量，分别衡量决策者是否...

    arXiv:2606.06081v2 Announce Type: replace  Abstract: Appropriate reliance on AI advice has become a central research theme in human-AI collaboration. Existing frameworks have focused exclusively on point predictions as AI advice. However, set-valued AI advice (e.g., discrete sets or continuous intervals) is increasingly being used to communicate uncertainty and improve human decision making. In this paper, we develop the first formal framework for measuring appropriate reliance on set-valued AI advice within the sequential judge-advisor paradigm, spanning both classification and regression tasks. For classification, we first introduce the dimensions that are necessary for evaluating set-valued AI advice. We then define two metrics: correct reliance rate on AI and correct reliance rate on self, which jointly characterize appropriate reliance in this setting. For regression, we introduce quantity of AI reliance and quality of AI reliance, which respectively measure whether a decision mak
    
[^236]: DELOS：用于开普勒光度数据中低信噪比盲凌星搜索的对比深度学习

    DELOS: Contrastive Deep Learning for Low-SNR Blind Transit Searches in Kepler Photometry

    [https://arxiv.org/abs/2605.29428](https://arxiv.org/abs/2605.29428)

    DELOS通过对比评分和深度学习，在低信噪比下无需预检测即可盲搜开普勒光曲线中的浅凌星，显著提升了中长周期信号的检测性能。

    

    我们提出了DELOS（相位折叠光曲线中的检测与对比评分），这是一个深度学习框架，利用对比评分在开普勒光度数据中进行浅凌星的盲搜索。DELOS结合了GPU加速的相位折叠、优化的相位分箱和自定义的一维卷积编码器，为每条折叠光曲线分配一个凌星相似度评分，从而在试验周期上生成评分周期图，无需依赖预先检测的阈值穿越事件。针对轨道周期为100-150天的中长周期信号，DELOS在2000万个合成光曲线上进行了训练，这些光曲线采用真实凌星模型和开普勒类噪声特性生成，在合成验证集上达到了99.3%的验证准确率。在受控的注入-恢复实验中，与盒拟合最小二乘法相比，DELOS的综合精确率-召回率性能提高了15.5%。

    arXiv:2605.29428v3 Announce Type: replace-cross  Abstract: We present DEtection in phase-folded Light curves with cOntrastive Scoring (DELOS), a deep-learning framework that uses contrastive scoring to perform blind searches for shallow transits in Kepler photometry. DELOS combines GPU-accelerated phase folding, optimized phase binning, and a custom one-dimensional convolutional encoder to assign a transit-likeness score to each folded light curve, thereby producing a score periodogram over trial periods without relying on pre-detected threshold-crossing events. Focusing on intermediate-to-long-period signals with orbital periods of 100-150 days, DELOS was trained on 20 million synthetic light curves generated with realistic transit models and Kepler-like noise properties, achieving a validation accuracy of 99.3% on the synthetic validation set. In controlled injection-recovery experiments, DELOS improves the combined precision-recall performance by 15.5% relative to Box-fitting Least 
    
[^237]: RULER：机器遗忘的表示级验证

    RULER: Representation-Level Verification of Machine Unlearning

    [https://arxiv.org/abs/2605.27569](https://arxiv.org/abs/2605.27569)

    本文提出RULER，一种表示级验证指标，能够检测模型内部表示中残留的遗忘数据痕迹，克服了传统输出级验证的局限性。

    

    机器遗忘旨在从已部署的模型中移除特定训练记录的影响，而无需从头重新训练。当前的协议通过成员推断、保持准确性和遗忘集准确性在输出层面进行验证，但一个模型可能同时满足这三个条件，却仍在其中间表示中编码了被遗忘的记录。我们引入了RULER，一套表示级验证指标。基于Oracle的对比指标M2衡量遗忘集记录是否与未包含这些记录重新训练的模型占据相同的表示位置。无Oracle的指标M4仅通过未学习模型的内部相似性结构检测残留，而无需重新训练。四种近似遗忘方法均通过输出级评估，但在线性混合效应模型下，M2在12种条件中的10种检测到显著残留（p<0.05），且效应大小随遗忘比例增加而增大。

    arXiv:2605.27569v3 Announce Type: replace  Abstract: Machine unlearning aims to remove the influence of specific training records from a deployed model without retraining from scratch. Current protocols verify this at the output level through membership inference, retain accuracy, and forget-set accuracy, but a model can satisfy all three whilst still encoding forgotten records in its intermediate representations. We introduce RULER, a set of representation-level verification metrics. The oracle-comparative metric M2 measures whether forget-set records occupy the same representational position as in a model retrained without them. The oracle-free metric M4 detects residuals from the unlearned model's internal similarity structure alone, without retraining. Four approximate unlearning methods all pass output-level evaluation, yet under a linear mixed-effects model M2 detects significant residuals in 10 of 12 conditions (p<0.05), with effect sizes growing as the forget fraction increases
    
[^238]: ICICLE：利用上下文文档扩展检索

    ICICLE: Expanding Retrieval with In-Context Documents

    [https://arxiv.org/abs/2605.26902](https://arxiv.org/abs/2605.26902)

    ICICLE通过将新文档作为推理时的上下文证据，结合复制路由和校准机制，实现了无需重新训练即可扩展生成式检索的增量索引。

    

    生成式检索（GR）利用参数化知识将查询直接映射到文档标识符（docids）。然而，这种设计使得语料库扩展成本高昂：添加新文档需要更新模型参数以编码新的文档-docid关联，这会导致重复训练和对先前索引文档的灾难性遗忘。在本工作中，我们将增量GR重新审视为一个上下文内检索问题，其中新添加的文档作为推理时的文档-docid证据提供。我们提出ICICLE，一种上下文内索引框架，它在参数化记忆和上下文提供的文档-docid对上进行源感知的docid生成。ICICLE结合了基于`[COPY]`的路由机制、基于偏好的校准以及大型上下文适应，以区分上下文基于的检索与参数化检索。在MS MARCO和NQ320K上的实验表明，ICICLE提高了新引入文档的检索性能。

    arXiv:2605.26902v3 Announce Type: replace-cross  Abstract: Generative retrieval (GR) maps queries directly to document identifiers (docids) using parametric knowledge, However, this design makes corpus expansion costly: adding new documents requires updating model parameters to encode new document-docid associations incurs repeated training and catastrophic forgetting of previously indexed documents. In this work, we revisit incremental GR as an in-context retrieval problem, where newly added documents are supplied as inference-time document-docid evidence. We propose ICICLE, an in-context indexing framework that performs source-aware docid generation over both parametric memory and context-provided document-docid pairs. ICICLE combines a `[COPY]`-based routing mechanism, preference-based calibration, and large context adaptation to distinguish context-grounded retrieval from parametric retrieval. Experiments on MS MARCO and NQ320K show that ICICLE improves retrieval of newly introduce
    
[^239]: MBABench：评估LLM代理在金融领域端到端电子表格任务中的表现

    MBABench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance

    [https://arxiv.org/abs/2605.22664](https://arxiv.org/abs/2605.22664)

    本文首次评估了LLM代理在金融端到端电子表格任务中的表现，填补了现有基准只关注问答或单公式编辑的空白，并强调了交付物质量需考虑可读性和易修改性等高层次标准。

    

    摘要：arXiv:2605.22664v5 公告类型：替换 摘要：LLM代理越来越被期望执行端到端工作流程，从高级用户指令中生成完整工件。为满足企业需求，前沿AI实验室已开发出能够从零开始构建完整电子表格的代理。这在金融领域尤为重要，因为财务建模、预测和情景分析等核心工作流程通常通过电子表格进行。然而，现有的电子表格基准并未衡量这一新能力，而是侧重于问答或单一公式编辑。为填补这一空白，我们提供了对代理在端到端电子表格任务上的首批评估之一，重点关注建模和情景分析等经济关键金融工作流程。由于这些交付物通常由多个利益相关者定期审查和修订，判断其质量必然涉及可读性或易修改性等高层次标准。

    arXiv:2605.22664v5 Announce Type: replace  Abstract: LLM agents are increasingly expected to carry out end-to-end workflows, producing complete artifacts from high-level user instructions. To meet enterprise needs, frontier AI labs have developed agents that can construct entire spreadsheets from scratch. This is especially relevant in finance, where core workflows such as financial modeling, forecasting, and scenario analysis are commonly conducted through spreadsheets. Yet, existing spreadsheet benchmarks do not measure this new capability, focusing instead on question-answering or single-formula edits. To address this gap, we provide one of the first evaluations of agents on end-to-end spreadsheet tasks, focusing on economically critical financial workflows such as modeling and scenario analysis. Since deliverables therein are routinely reviewed and revised by multiple stakeholders, judging their quality necessarily involves high-level criteria such as readability or ease of modific
    
[^240]: EgoMemReason：面向长时程自我中心视频理解的记忆驱动推理基准

    EgoMemReason: A Memory-Driven Reasoning Benchmark for Long-Horizon Egocentric Video Understanding

    [https://arxiv.org/abs/2605.09874](https://arxiv.org/abs/2605.09874)

    本文提出了EgoMemReason基准，专注于长时程自我中心视频中的记忆驱动推理，填补了现有周级视频基准在跨天整合证据推理方面的空白。

    

    下一代视觉助手，如智能眼镜、具身智能体和常开式生活记录系统，必须对一整天或更长时间的连续视觉体验进行推理。在超长视频中，相关信息稀疏地分布在数小时或数天内，使得记忆成为一个基本挑战：模型必须随时间累积信息、回忆先前状态、跟踪时间顺序，并抽象出重复出现的模式。然而，现有的周级视频基准主要设计用于感知和识别任务，如时刻定位或全局摘要，而非需要跨多天整合证据的推理任务。为解决这一空白，我们引入了EgoMemReason，一个通过记忆驱动推理进行周级自我中心视频理解的综合基准。EgoMemReason评估三种互补的记忆类型：实体记忆，跟踪对象状态如何随时间的演变和变化。

    arXiv:2605.09874v2 Announce Type: replace-cross  Abstract: Next-generation visual assistants, such as smart glasses, embodied agents, and always-on life-logging systems, must reason over an entire day or more of continuous visual experience. In ultra-long videos, relevant information is sparsely distributed across hours or days, making memory a fundamental challenge: models must accumulate information over time, recall prior states, track temporal order, and abstract recurring patterns. However, existing week-long video benchmarks are primarily designed for perception and recognition, such as moment localization or global summarization, rather than reasoning that requires integrating evidence across multiple days. To address this gap, we introduce EgoMemReason, a comprehensive benchmark for week-long egocentric video understanding through memory-driven reasoning. EgoMemReason evaluates three complementary memory types: entity memory, tracking how object states evolve and change across 
    
[^241]: 关键覆盖至关重要：OCR临床报告的半结构化提取

    Key Coverage Matters: Semi-Structured Extraction of OCR Clinical Reports

    [https://arxiv.org/abs/2605.09440](https://arxiv.org/abs/2605.09440)

    本文提出了一种通过规范键条件抽取式问答方法，在开放键空间下对OCR临床报告进行半结构化提取，以解决医疗数据分散和文本噪声问题。

    

    临床报告往往因隐私法规和数据孤岛限制直接信息共享，而分散在各医疗机构之间。当患者在不同医院就诊时，他们通常携带以往就诊的纸质或扫描报告。这阻碍了电子健康记录（EHR）的集成和纵向回顾，也影响了依赖更完整患者记录的后续应用，如患者管理、随访护理、真实世界研究和临床试验匹配。尽管OCR可以将此类报告数字化，但由于临床文档异构、OCR文本嘈杂，且许多医疗环境要求低成本本地部署，可靠的提取仍具挑战性。我们将此问题表述为基于OCR临床报告的规范键条件抽取式问答。由于关键字段既非固定也非预先已知，键空间是开放的。我们通过维护一个规范键清单来应对这一挑战。

    arXiv:2605.09440v2 Announce Type: replace-cross  Abstract: Clinical reports are often fragmented across healthcare institutions because privacy regulations and data silos limit direct information sharing. When patients seek care at a different hospital, they often carry paper or scanned reports from prior visits. This hinders EHR integration and longitudinal review, and downstream applications that depend on more complete patient records, such as patient management, follow-up care, real-world studies, and clinical-trial matching. Although OCR can digitize such reports, reliable extraction remains challenging because clinical documents are heterogeneous, OCR text is noisy, and many healthcare settings require low-cost on-premise deployment. We formulate this problem as canonical key-conditioned extractive question answering over OCR-derived clinical reports. Because the key fields are neither fixed nor known in advance, the key space is open. We maintain a canonical key inventory throug
    
[^242]: 事件因果检索增强生成：面向复杂场景长视频推理的检索增强生成框架

    Event-Causal RAG: A Retrieval-Augmented Generation Framework for Long Video Reasoning in Complex Scenarios

    [https://arxiv.org/abs/2605.06185](https://arxiv.org/abs/2605.06185)

    本文提出EC-RAG框架，通过双模态哨兵机制将超长视频分割为语义完整事件，用SES结构建模因果转换，并利用双向量图记忆和双向检索实现高效的长视频推理，解决了传统方法在事件连贯性和状态转换建模上的不足。

    

    arXiv:2605.06185v2 公告类型：替换 摘要：大型视觉语言模型在短中长度视频理解上表现良好，但在超长视频中仍难以维持连贯的事件记忆并恢复远距离关系。端到端方法受视觉令牌增长和上下文长度的限制，而固定片段检索往往割裂完整事件并削弱状态转换建模。我们提出事件因果检索增强生成（EC-RAG），一种用于超长和流式视频推理的轻量级检索增强框架。双视觉-音频哨兵机制将视频流分割成语义完整的事件，表示为状态-事件-状态（SES）结构，该结构将可观察的事件前状态、中心事件和事件后状态组织为事件局部因果转换。这些转换存储于双向量图记忆中，并通过实体一致的轨迹进行时间连接。在问答过程中，双向图检索

    arXiv:2605.06185v2 Announce Type: replace  Abstract: Large vision-language models perform well on short- and medium-length video understanding but still struggle to maintain coherent event memory and recover long-range relationships in ultra-long videos. End-to-end methods are limited by visual-token growth and context length, while fixed-segment retrieval often fragments complete events and weakens state-transition modeling.We propose Event-Causal RAG (EC-RAG), a lightweight retrieval-augmented framework for ultra-long and streaming video reasoning. A dual visual-audio sentinel mechanism segments video streams into semantically complete events, represented as State-Event-State (SES) structures that organize observable pre-event states, central events, and post-event states as event-local causal transitions. These transitions are stored in dual vector-graph memory and temporally connected through entity-consistent trajectories. During question answering, bidirectional graph retrieval r
    
[^243]: MedStruct-S：面向OCR临床报告中关键发现、关键条件问答与半结构化提取的基准

    MedStruct-S: A Benchmark for Key Discovery, Key-Conditioned QA and Semi-Structured Extraction from OCR Clinical Reports

    [https://arxiv.org/abs/2605.03103](https://arxiv.org/abs/2605.03103)

    MedStruct-S是一个新基准，专注于在未知关键和OCR噪声条件下评估临床报告中的关键发现、关键条件问答和端到端键值提取任务，包含3,582个真实世界标注页面。

    

    arXiv:2605.03103v2 公告类型：替换交叉 摘要：从OCR生成的临床报告中进行半结构化信息提取（IE）对于高效重建患者纵向病史至关重要。在实践中，这一场景通常涉及三个任务：（i）字段标题（关键）发现，（ii）关键条件问答（QA），以及（iii）端到端键值对提取。然而，现有评估常常低估两个因素：异构且不完全已知的关键表示，以及OCR引入的噪声。这使得在真实世界环境中评估模型鲁棒性变得困难。我们提出了MedStruct-S，一个专门设计用于在未知关键和OCR噪声下评估这些任务的基准。MedStruct-S包含3,582个带注释的真实世界临床报告页面。利用MedStruct-S，我们对两种代表性范式进行了基准测试：仅编码器的序列标注与后处理，以及仅解码器的结构化生成，覆盖了四种仅编码器模型和五种（原文截断）。

    arXiv:2605.03103v2 Announce Type: replace-cross  Abstract: Semi-structured information extraction (IE) from OCR-derived clinical reports is crucial for efficiently reconstructing patients' longitudinal medical histories. In practice, this scenario commonly involves three tasks: (i) field-header (key) discovery, (ii) key-conditioned question answering (QA), and (iii) end-to-end key-value pair extraction. However, existing evaluations often under-model two factors: heterogeneous and incompletely known key representations, and OCR-induced noise. This makes it difficult to assess model robustness in real-world settings.   We present MedStruct-S, a benchmark specifically designed to evaluate these tasks under unknown keys and OCR noise. MedStruct-S contains 3,582 annotated real-world clinical report pages. Using MedStruct-S, we benchmark two representative paradigms: encoder-only sequence labeling with post-processing and decoder-only structured generation, covering four encoder-only and fi
    
[^244]: 音频-语言模型在构音障碍语音识别中未能利用多模态上下文时的表现

    When Audio-Language Models Fail to Leverage Multimodal Context for Dysarthric Speech Recognition

    [https://arxiv.org/abs/2605.02782](https://arxiv.org/abs/2605.02782)

    当前音频-语言模型在构音障碍语音识别中无法有效利用诊断或临床上下文，提示改进微弱甚至有害，而结合混合临床提示的LoRA微调则能带来性能提升。

    

    自动语音识别（ASR）系统在构音障碍及其他非典型语音上仍显脆弱。近期的音频-语言模型提出了通过在推理时结合额外临床上下文来提升性能的可能性，但尚不清楚这些模型能否利用此类信息。我们基于语音可访问性项目（SAP）数据集构建了一个基准测试，该测试检验诊断标签、临床医生得出的语音评级以及逐步丰富的临床描述是否能提高构音障碍语音的转录准确性。在九个模型的匹配比较中，我们发现当前模型并未有效利用这些上下文：基于诊断信息和临床详细描述的提示仅带来微不足道的改进，且常常导致词错误率恶化。我们通过上下文相关的微调补充了提示分析，表明使用混合临床提示格式的LoRA适应可取得改进。

    arXiv:2605.02782v2 Announce Type: replace  Abstract: Automatic speech recognition (ASR) systems remain brittle on dysarthric and other atypical speech. Recent audio-language models raise the possibility of improving performance by conditioning on additional clinical context at inference time, but it is unclear whether these models can make use of such information. We introduce a benchmark built on the Speech Accessibility Project (SAP) dataset that tests whether diagnosis labels, clinician-derived speech ratings, and progressively richer clinical descriptions improve transcription accuracy for dysarthric speech. Across matched comparisons on nine models, we find that current models do not meaningfully use this context: diagnosis-informed and clinically detailed prompts yield negligible improvements and often degrade word error rate. We complement the prompting analysis with context-dependent fine-tuning, showing that LoRA adaptation with a mixture of clinical prompt formats achieves a 
    
[^245]: 面向不完美感知代理的区间POMDP屏蔽

    Interval POMDP Shielding for Imperfect-Perception Agents

    [https://arxiv.org/abs/2604.20728](https://arxiv.org/abs/2604.20728)

    本文提出一种基于区间POMDP的屏蔽算法，利用有限标注数据构建感知不确定性置信区间，以在运行时提供带高概率有限时域安全保证的保守信念集。

    

    依赖学习感知的自主系统在传感器读数被误分类时可能做出不安全的决策。我们研究这一场景下的屏蔽机制：给定一个提议的动作，屏蔽器会阻止可能违反安全性的动作。我们考虑常见的情况，即系统动力学已知，但感知不确定性必须从有限的标注数据中估计。根据这些数据，我们为感知结果的概率构建置信区间，并将其用于将系统建模为一个具有离散状态和动作的有限区间部分可观测马尔可夫决策过程（Interval POMDP）。随后，我们提出一种算法，计算与迄今为止观察到的观测一致的底层状态上的保守信念集。这使我们能够构建一个运行时屏蔽器，并附带有限时域保证：在训练数据上以高概率，如果真实的感知不确定性率位于学习到的区间内，则安全性得以保持。

    arXiv:2604.20728v2 Announce Type: replace  Abstract: Autonomous systems that rely on learned perception can make unsafe decisions when sensor readings are misclassified. We study shielding for this setting: given a proposed action, a shield blocks actions that could violate safety. We consider the common case where system dynamics are known but perception uncertainty must be estimated from finite labeled data. From these data we build confidence intervals for the probabilities of perception outcomes and use them to model the system as a finite Interval Partially Observable Markov Decision Process with discrete states and actions. We then propose an algorithm to compute a conservative set of beliefs over the underlying state that is consistent with the observations seen so far.   This enables us to construct a runtime shield that comes with a finite-horizon guarantee: with high probability over the training data, if the true perception uncertainty rates lie within the learned intervals,
    
[^246]: AutoOR：规模化后训练大语言模型以自动形式化运筹学问题

    AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems

    [https://arxiv.org/abs/2604.16804](https://arxiv.org/abs/2604.16804)

    AutoOR提出了一种结合合成数据生成和强化学习的可扩展流水线，通过求解器反馈作为奖励信号，使8B参数模型在多个运筹学基准上达到或超越更大前沿模型的性能，尤其在非线性物理动力学问题上实现了突破。

    

    arXiv:2604.16804v3 公告类型：替换-交叉 摘要：优化问题是制造业、物流、调度及其他工业场景中决策的核心。将这些问题的复杂描述转化为求解器可用的公式需要专业的运筹学（OR）专业知识，这使得规模化变得困难。我们提出了AutoOR，一种可扩展的合成数据生成和强化学习流水线，用于训练大语言模型（LLMs），使其能够将自然语言中指定的优化问题自动形式化，涵盖线性、混合整数和非线性类别。AutoOR从标准优化形式生成经过验证的训练数据，并使用求解器执行反馈作为强化学习后训练的奖励信号。将AutoOR应用于一个8B模型，在六个既定运筹学基准上取得了最先进或具有竞争力的结果，匹配了显著更大的前沿模型。对于一个涉及物理动力学的非线性问题类别，前沿模型得分接近0%，而AutoOR模型实现了显著改进。

    arXiv:2604.16804v3 Announce Type: replace-cross  Abstract: Optimization problems are central to decision-making in manufacturing, logistics, scheduling, and other industrial settings. Translating complicated descriptions of these problems into solver-ready formulations requires specialized operations research (OR) expertise, making it hard to scale. We present AutoOR, a scalable synthetic data generation and reinforcement learning pipeline that trains LLMs to autoformalize optimization problems specified in natural language across linear, mixed-integer, and non-linear categories. AutoOR generates verified training data from standard optimization forms and uses solver execution feedback as the reward signal for RL post-training. AutoOR applied to an 8B model achieves state-of-the-art or competitive results across six established OR benchmarks, matching significantly larger frontier models. For a non-linear problem class involving physical dynamics, where frontier models score near 0%, w
    
[^247]: 何时将苹果称为红色：人类遵循内省规则，而视觉语言模型则不然

    When to Call an Apple Red: Humans Follow Introspective Rules, VLMs Don't

    [https://arxiv.org/abs/2604.06422](https://arxiv.org/abs/2604.06422)

    本文通过新提出的GCA数据集，发现视觉语言模型在颜色归属决策中系统性地违背自身陈述的内省阈值规则，而人类则更遵守这些规则。

    

    摘要：理解视觉语言模型（VLMs）何时会出现意外行为、模型是否能可靠地预测自身行为，以及模型是否遵循其内省推理，是可信部署的核心挑战。为此，我们引入了分级颜色属性（GCA）数据集，这是一个受控基准，旨在引出决策规则并评估参与者对这些规则的忠实度。GCA由线条画组成，这些线条画在三种条件下变化像素级颜色覆盖：世界知识重着色、反事实重着色以及无颜色先验的形状。利用GCA，我们要求VLMs和人类参与者陈述一个阈值规则：即物体像素中必须具有给定颜色的比例，才能使该物体获得该颜色标签。然后，我们将这些规则与随后的颜色归属决策进行比较。我们的发现表明，模型系统性地违反其自身的内省规则。

    arXiv:2604.06422v2 Announce Type: replace-cross  Abstract: Understanding when Vision-Language Models (VLMs) will behave unexpectedly, whether models can reliably predict their own behavior, and if models adhere to their introspective reasoning are central challenges for trustworthy deployment. To study this, we introduce the Graded Color Attribution (GCA) dataset, a controlled benchmark designed to elicit decision rules and evaluate participant faithfulness to these rules. GCA consists of line drawings that vary pixel-level color coverage across three conditions: world-knowledge recolorings, counterfactual recolorings, and shapes with no color priors. Using GCA, we ask both VLMs and human participants to state a threshold rule: the share of an object's pixels that must be a given color for the object to receive that color label. We then compare these rules with their subsequent color attribution decisions. Our findings reveal that models systematically violate their own introspective r
    
[^248]: 从多智能体到单智能体：技能蒸馏何时有益？

    From Multi-Agent to Single-Agent: When Is Skill Distillation Beneficial?

    [https://arxiv.org/abs/2604.01608](https://arxiv.org/abs/2604.01608)

    本论文提出一种诊断方法，用于确定在多智能体工作流蒸馏为单智能体技能时，何时添加管道指导有益或有害，并通过行为-结果自由度指标解释性能反转现象。

    

    arXiv:2604.01608v5 公告类型：替换。摘要：多智能体系统（MAS）在结构化数据科学任务中，通过涵盖阶段、工具、共享状态、验证和修复的工作流外部化分析控制。将此类工作流蒸馏为单一智能体技能可以减少编排开销，但尚不清楚哪些工作流组件应跨越控制边界。我们区分了能力资源（扩展智能体能做什么）和管道指导（约束其探索哪些解决方案）。在相同的因果估计实例上，向能力匹配的技能添加任务限定的源管道指导，在方法选择准确性下将归一化效用改变+19.6点，但在数值误差下改变-10.3点。为解释这种反转，我们引入了行为-结果自由度（F），一种预合成的有符号行为-结果秩不匹配诊断，并通过有符号锚定-秩转移形式化其候选条件作用。

    arXiv:2604.01608v5 Announce Type: replace  Abstract: Multi-agent systems (MAS) for structured data-science tasks externalize analytical control through workflows spanning stages, tools, shared state, verification, and repair. Distilling such workflows into a single-agent skill can reduce orchestration overhead, but it remains unclear which workflow components should cross the control boundary. We distinguish capability resources, which expand what an agent can do, from pipeline guidance, which constrains which solutions it explores. On the same causal-estimation instances, adding task-qualified source pipeline guidance to a capability-matched skill changes normalized utility by +19.6 points under method-selection accuracy but -10.3 points under numerical error. To explain this reversal, we introduce Behavior-Outcome Freedom (F), a pre-synthesis diagnostic of signed behavior-outcome rank mismatch, and formalize its candidate-conditional role through Signed Anchor-Rank Transfer. Motivate
    
[^249]: 野火扑救：复杂性、模型与实例

    Wildfire Suppression: Complexity, Models, and Instances

    [https://arxiv.org/abs/2603.29865](https://arxiv.org/abs/2603.29865)

    本文证明了野火扑救资源分配问题在多种图结构上的强NP完全性，提出了先进的混合整数规划方法，并引入基于物理模型的真实实例生成器。

    

    arXiv:2603.29865v2 公告类型：替换-交叉 摘要：野火在全球范围内造成重大损失，且在许多地区，火灾天气条件的频率可能增加。我们研究在基于图的地貌表示上随时间分配扑救资源以减缓火势蔓延的问题。我们的贡献是理论性和方法性的。首先，我们证明了该问题及其两个相关变体在平面图上的强NP完全性，以及其中两个问题在全加权有向网格上的强NP完全性。我们还表明，当所有资源同时释放时，该问题仍保持强NP完全性。其次，我们提出了一种新的混合整数规划（MIP）公式，获得了最先进的结果，表明MIP是一种有竞争力的方法，这与早期发现相反。第三，鉴于现有基准缺乏真实性和难度，我们引入了一个基于Rothermel地表火蔓延模型的物理驱动实例生成器。

    arXiv:2603.29865v2 Announce Type: replace-cross  Abstract: Wildfires cause major losses worldwide, and the frequency of fire-weather conditions is likely to increase in many regions. We study the allocation of suppression resources over time on a graph-based representation of a landscape to slow down fire propagation. Our contributions are theoretical and methodological. First, we prove strong NP-completeness on planar graphs for this problem and two related variants, and on full weighted directed grids for two of the three problems. We also show that this problem remains strongly NP-complete when all resources are released simultaneously. Second, we propose a new mixed-integer programming (MIP) formulation that obtains state-of-the-art results, showing that MIP is a competitive approach contrary to earlier findings. Third, showing that existing benchmarks lack realism and difficulty, we introduce a physics-grounded instance generator based on Rothermel's surface fire spread model. We 
    
[^250]: 工程设计与系统工程中数据集可导航地图的框架与原型

    A Framework and Prototype for a Navigable Map of Datasets in Engineering Design and Systems Engineering

    [https://arxiv.org/abs/2603.15722](https://arxiv.org/abs/2603.15722)

    本文提出了一种系统框架和交互式工具原型，通过多维分类体系构建“EDSE数据集地图”，以解决工程设计与系统工程领域数据集碎片化和不可访问的问题，促进数据发现和复用。

    

    摘要：系统生命周期中数据的激增为工程设计与系统工程（EDSE）带来了重大机遇和挑战。虽然这种“数字主线”有潜力推动创新，但现有数据集的碎片化和不可访问性阻碍了方法验证、限制了可重复性，并减缓了研究进展。与受益于成熟基准生态系统的计算机视觉和自然语言处理等领域不同，工程设计研究往往依赖于小型、专有或临时数据集。本文通过提出一个“EDSE数据集地图”的系统框架来应对这一挑战。该框架基于一个多维分类体系，旨在按领域、生命周期阶段、数据类型和格式对工程数据集进行分类，从而实现分面发现。文中详细描述并演示了一个交互式发现工具的架构。

    arXiv:2603.15722v3 Announce Type: replace-cross  Abstract: The proliferation of data across the system lifecycle presents both a significant opportunity and a challenge for Engineering Design and Systems Engineering (EDSE). While this "digital thread" has the potential to drive innovation, the fragmented and inaccessible nature of existing datasets hinders method validation, limits reproducibility, and slows research progress. Unlike fields such as computer vision and natural language processing, which benefit from established benchmark ecosystems, engineering design research often relies on small, proprietary, or ad-hoc datasets. This paper addresses this challenge by proposing a systematic framework for a "Map of Datasets in EDSE." The framework is built upon a multi-dimensional taxonomy designed to classify engineering datasets by domain, lifecycle stage, data type, and format, enabling faceted discovery. An architecture for an interactive discovery tool is detailed and demonstrated
    
[^251]: 在省略三段论的逻辑理解中使隐含前提显性化

    Making Implicit Premises Explicit in Logical Understanding of Enthymemes

    [https://arxiv.org/abs/2603.06114](https://arxiv.org/abs/2603.06114)

    本文提出了一种结合两个大型语言模型和神经符号推理器的流水线，首次系统地将省略三段论的文本转化为逻辑论证，并生成隐含前提和逻辑公式，以实现逻辑蕴含的显性化理解。

    

    arXiv:2603.06114v3 公告类型：交叉替换  摘要：现实世界中的文本和对话中的论证通常是省略三段论（即其某些前提和/或主张是隐含的）。自然语言处理（NLP）方法处理省略三段论时，可能识别出文本中的省略三段论，但不解码其底层逻辑；而基于逻辑的方法处理它们时，假设存在一个包含足够公式的知识库，可以通过溯因推理来解码。因此，目前缺乏一种系统的方法来将省略三段论的文本成分转换为逻辑论证，并生成解码所需的逻辑公式，从而展示逻辑蕴含关系。为了解决这一问题，我们提出一个流水线，整合了：（1）一个大型语言模型（LLM）基于显式前提和主张生成中间隐含前提；（2）另一个LLM将自然语言转换为逻辑公式；（3）一个基于神经符号推理器。

    arXiv:2603.06114v3 Announce Type: replace-cross  Abstract: Real-world arguments in text and dialogues are normally enthymemes (i.e. some of their premises and/or claims are implicit). Natural language processing (NLP) methods for handling enthymemes can potentially identify enthymemes in text but they do not decode their underlying logic, whereas logic-based approaches for handling them assume a knowledgebase with sufficient formulae that can be used to decode them via abduction. There is therefore a lack of a systematic method for translating textual components of an enthymeme into a logical argument and generating the logical formulae required for their decoding, and thereby showing logical entailment. To address this, we propose a pipeline that integrates: (1) a large language model (LLM) to generate intermediate implicit premises based on the explicit premise and claim; (2) another LLM to translate the natural language into logical formulas; and (3) a neuro-symbolic reasoner based 
    
[^252]: SkillNet：创建、评估与连接AI技能

    SkillNet: Create, Evaluate, and Connect AI Skills

    [https://arxiv.org/abs/2603.04448](https://arxiv.org/abs/2603.04448)

    本文提出了SkillNet，一个开放基础设施，通过统一本体论、大规模技能仓库和多功能工具包，系统性地创建、评估和连接AI技能，解决了代理缺乏技能积累和迁移的问题。

    

    摘要：arXiv:2603.04448v2 公告类型：替换 摘要：当前的AI代理能够灵活调用工具并执行复杂任务，但其长期发展受到缺乏技能系统性积累和迁移的阻碍。在没有统一的技能整合机制的情况下，代理经常“重复造轮子”，在孤立的环境中重新发现解决方案，而不利用先前的策略。为解决这一挑战，我们引入了SkillNet，一个用于大规模创建、评估和组织AI技能的开放基础设施。SkillNet在统一本体论中结构化技能，支持从异构来源创建技能、建立丰富的关联连接，并在安全性、完整性、可执行性、可维护性和成本意识方面进行多维度评估。我们的基础设施整合了一个包含超过60万个技能的仓库、一个交互式平台和一个多功能的Python工具包。在ALFWorld、WebShop和ScienceWorld上的实验表明……

    arXiv:2603.04448v2 Announce Type: replace  Abstract: Current AI agents can flexibly invoke tools and execute complex tasks, yet their long-term advancement is hindered by the lack of systematic accumulation and transfer of skills. Without a unified mechanism for skill consolidation, agents frequently ``reinvent the wheel'', rediscovering solutions in isolated contexts without leveraging prior strategies. To address this challenge, we introduce SkillNet, an open infrastructure for creating, evaluating, and organizing AI skills at scale. SkillNet structures skills within a unified ontology that supports creating skills from heterogeneous sources, establishing rich relational connections, and performing multi-dimensional evaluation across Safety, Completeness, Executability, Maintainability, and Cost-awareness. Our infrastructure integrates a repository of over 600,000 skills, an interactive platform, and a versatile Python toolkit. Experiments on ALFWorld, WebShop, and ScienceWorld show 
    
[^253]: 保形策略控制

    Conformal Policy Control

    [https://arxiv.org/abs/2603.02196](https://arxiv.org/abs/2603.02196)

    本文提出了一种基于保形校准的新框架，利用安全参考策略作为概率调节器，在高风险环境中可证明地控制新策略的探索强度，无需模型类别假设或超参数调优，并在有限样本下提供非单调损失函数的保证。

    

    arXiv:2603.02196v4 公告类型：替换 摘要：智能体必须尝试新行为以进行探索和改进。在高风险环境中，违反安全约束的智能体可能造成伤害，必须被下线，从而中断任何未来的交互。模仿旧行为是安全的，但过度保守会抑制探索。行为改变多少才算过多？我们展示了如何将任何安全参考策略用作任何优化但未经测试策略的概率调节器。基于安全策略数据的保形校准决定了新策略可以多激进地行动，同时可证明地执行用户声明的风险容忍度。与保守优化方法不同，我们不假设用户已识别正确的模型类别或调整任何超参数。与之前的保形方法不同，我们的理论即使在非单调有界损失函数下也能提供有限样本保证，并引入了一种新的策略控制设置。我们在应用上的实验...

    arXiv:2603.02196v4 Announce Type: replace  Abstract: An agent must try new behaviors to explore and improve. In high-stakes environments, an agent that violates safety constraints may cause harm and must be taken offline, curtailing any future interaction. Imitating old behavior is safe, but excessive conservatism discourages exploration. How much behavior change is too much? We show how to use any safe reference policy as a probabilistic regulator for any optimized but untested policy. Conformal calibration on data from the safe policy determines how aggressively the new policy can act, while provably enforcing the user's declared risk tolerance. Unlike conservative optimization methods, we do not assume the user has identified the correct model class nor tuned any hyperparameters. Unlike previous conformal methods, our theory provides finite-sample guarantees even for non-monotonic bounded loss functions, and it introduces a new policy control setting. Our experiments on applications
    
[^254]: 全曲训练符号音乐语言模型：通过全时域压缩递归方法

    Whole-Piece Training for Symbolic Music Language Models via Full-Horizon Compressed Recurrence

    [https://arxiv.org/abs/2602.19816](https://arxiv.org/abs/2602.19816)

    本文提出全时域压缩递归（FHCR）框架，通过压缩KV表示实现符号音乐语言模型的全曲连续训练，并引入KRCU诊断指标来评估长程依赖。

    

    为追求计算效率，现代语言模型通常基于独立采样的定长序列进行训练。符号音乐语言模型在很大程度上继承了这一范式，尽管音乐结构自然地以完整作品而非孤立片段的形式展开。将作品分割成独立的训练实例因此阻碍了对完整作品的连续条件建模。我们提出了一种通过全时域压缩递归（FHCR）实现符号音乐语言模型全曲训练的实用框架。FHCR在保留递归记忆的完整时间跨度的同时，降低了其键值（KV）表示的维度，使得在有限GPU内存下进行连续的全曲训练变得可行。为直接评估功能性的长程依赖性，我们引入了KV重置上下文利用度（KRCU），一种评估时的诊断工具。在MAESTRO符号钢琴数据集上进行了验证。

    arXiv:2602.19816v3 Announce Type: replace-cross  Abstract: For computational efficiency, modern language models are typically trained on independently sampled fixed-length sequences. Symbolic music language models largely inherit this paradigm, despite musical structure naturally unfolding over complete compositions rather than isolated excerpts. Fragmenting compositions into independent training instances therefore prevents continuous conditioning over the complete work.   We present a practical framework for whole-piece training of symbolic music language models via Full-Horizon Compressed Recurrence (FHCR). FHCR preserves the full temporal horizon of recurrent memory while reducing the dimensionality of its key-value (KV) representation, making continuous whole-piece training practical under limited GPU memory.   To directly assess functional long-range dependence, we introduce KV-Reset Context Utilization (KRCU), an evaluation-time diagnostic. On the MAESTRO symbolic piano dataset,
    
[^255]: 基于张量分解的导频受限MIMO信道结构化信息估计

    Structure-Informed Estimation for Pilot-Limited MIMO Channels via Tensor Decomposition

    [https://arxiv.org/abs/2602.04083](https://arxiv.org/abs/2602.04083)

    本文提出了一种结合低秩张量分解与3D U-Net残差学习的混合估计器，有效解决了导频受限MIMO信道估计中的欠定补全问题，并在极端导频稀缺条件下实现了显著的性能提升。

    

    摘要：在宽带MIMO系统中，准确的信道状态信息受到导频开销的限制，这一挑战随着带宽向6G扩展而加剧。本文提出了一种结构信息辅助的混合估计器，将导频受限的MIMO信道估计建模为从稀疏导频观测中进行的低秩张量补全——这是一个欠定逆问题，此前的方法通过假设完全观测张量来避免此问题。本文比较了Canonical Polyadic（CP）分解和Tucker分解：CP分解在镜面信道中表现优异，其秩一参数化精确匹配；而Tucker分解在极端导频稀缺情况下提供数值稳定性，此时CP分解会出现重尾发散。一个轻量级3D U-Net学习超出低秩结构的残差分量，以补偿漫散射和硬件非理想性。在合成镜面信道上，与最小二乘方法相比，Tucker补全将归一化均方误差（NMSE）提高了10.88 dB。

    arXiv:2602.04083v3 Announce Type: replace-cross  Abstract: Accurate channel state information in wideband MIMO systems is constrained by pilot overhead, a challenge intensifying as bandwidths scale toward 6G. This paper proposes a structure-informed hybrid estimator formulating pilot-limited MIMO channel estimation as low-rank tensor completion from sparse pilot observations---an underdetermined inverse problem that prior approaches avoid by assuming fully observed tensors. Canonical polyadic~(CP) and Tucker decompositions are compared: CP excels for specular channels matching its rank-one parameterization exactly, while Tucker provides numerical stability at extreme pilot scarcity where CP exhibits heavy-tail divergence. A lightweight 3D U-Net learns residual components beyond the low-rank structure, compensating for diffuse scattering and hardware non-idealities. On synthetic specular channels, Tucker completion improves normalized mean-squared error (NMSE) by $10.88$~dB over least s
    
[^256]: FiLoRA：用于可控特征依赖的“聚焦与忽略”LoRA方法

    FiLoRA: Focus-and-Ignore LoRA for Controllable Feature Reliance

    [https://arxiv.org/abs/2602.02060](https://arxiv.org/abs/2602.02060)

    FiLoRA通过指令条件化的低秩门控机制，实现了在不改变任务目标的前提下，用自然语言指令直接控制多模态模型对特定特征路径的依赖，从而缓解捷径和虚假相关行为。

    

    多模态基础模型整合了跨模态的异构信号，但目前尚不清楚是否可以通过显式调节对不同内部特征路径的依赖来控制其预测。现有的处理捷径行为或虚假相关的方法主要依赖事后分析或数据层面的干预，难以直接干预模型对信息的使用方式。我们提出了FiLoRA（聚焦与忽略LoRA），一种指令条件化的、参数高效的适应框架，能够在保持任务和预测目标不变的情况下，实现对特征依赖的可控调节。FiLoRA将适应过程分解为特征对齐的低秩模块，并应用指令条件化门控，使自然语言指令能够作为计算层面的控制信号作用于内部表示。我们在受控分类设置及地理相关任务上评估了FiLoRA的性能。

    arXiv:2602.02060v2 Announce Type: replace-cross  Abstract: Multimodal foundation models integrate heterogeneous signals across modalities, yet it remains unclear whether their predictions can be controlled by explicitly modulating reliance on different internal feature pathways. Existing approaches to shortcut and spurious behavior primarily rely on post hoc analysis or data-level interventions, offering limited ability to directly intervene on how models use information. We introduce FiLoRA (Focus-and-Ignore LoRA), an instruction-conditioned, parameter-efficient adaptation framework that enables controllable modulation of feature reliance while keeping the task and predictive objective fixed. FiLoRA decomposes adaptation into feature-aligned low-rank modules and applies instruction-conditioned gating, allowing natural language instructions to act as computation-level control signals over internal representations. We evaluate FiLoRA across both controlled classification settings and ge
    
[^257]: TrojanGYM：一种检测器在环的大语言模型，用于自适应RTL硬件木马插入

    TrojanGYM: A Detector-in-the-Loop LLM for Adaptive RTL Hardware Trojan Insertion

    [https://arxiv.org/abs/2601.17178](https://arxiv.org/abs/2601.17178)

    TrojanGYM是一个智能体驱动的LLM框架，通过检测器反馈循环自动生成多样化的硬件木马插入，以暴露和评估检测器的盲点。

    

    硬件木马（HT）仍然是一个关键威胁，因为基于学习的检测器常常过度拟合于狭窄的触发/负载模式和小型、风格化的基准测试。我们引入了TrojanGYM，一个智能体驱动的、基于大语言模型的框架，能够自动策划HT插入以暴露检测器的盲点。给定高层HT规格，一组协作的LLM智能体（使用GPT-4、LLaMA-3.3-70B、Gemini-2.5Pro和Claude Opus 4.5实例化）提出并优化RTL修改，实现多样化的触发器和负载，同时不影响HT和受攻击设计的功能。TrojanGYM实现了一个与HT检测器协同设计的智能体循环，其中约束感知的语法检查、基于测试台的功能验证和基于GNN的HT检测器提供反馈，迭代优化HT规格和插入策略，以更好地暴露检测器盲点。我们进一步提出了Robust-GNN4TJ，一种...

    arXiv:2601.17178v3 Announce Type: replace-cross  Abstract: Hardware Trojans (HTs) remain a critical threat because learning-based detectors often overfit to narrow trigger/payload patterns and small, stylized benchmarks. We introduce TrojanGYM, an agentic, LLM-driven framework that automatically curates HT insertions to expose detector blind spots. Given high-level HT specifications, a suite of cooperating LLM agents (instantiated with GPT-4, LLaMA-3.3-70B, Gemini-2.5Pro, and Claude Opus 4.5) proposes and refines RTL modifications that realize diverse triggers and payloads without impacting functionality of both the HT and the design under attack. TrojanGYM implements an agentic loop co-designed with HT detectors, in which constraint-aware syntactic checking, testbench-based functional verification, and GNN-based HT detectors provide feedback that iteratively refines HT specifications and insertion strategies to better surface detector blind spots. We further propose Robust-GNN4TJ, a n
    
[^258]: 评估音乐情境保持：音乐编辑系统的多面框架

    Evaluating Music Context Preservation: A Multi-facet Framework for Music Editing Systems

    [https://arxiv.org/abs/2512.14629](https://arxiv.org/abs/2512.14629)

    本文提出了首个全面的音乐情境保持评估框架MuseCPEval，通过细粒度指标和人类研究验证，系统性地评估音乐编辑系统在编辑过程中保留不变音乐方面的能力。

    

    音乐编辑在现代音乐制作中扮演着至关重要的角色，其应用涵盖电影、广播和游戏开发等领域。近年来，音乐编辑系统的进展使得多种编辑任务得以实现，如音色转换、乐器替换和风格变换。然而，许多现有工作忽视了评估它们在编辑过程中保持应保持不变的音乐方面的能力，我们将此定义为音乐情境保持（MuseCP）。尽管一些研究确实考虑了MuseCP，但其评估协议和指标并不全面。为解决这一问题，我们引入了首个MuseCP评估框架MuseCPEval，该框架覆盖四类音乐方面，并采用细粒度且量身定制的指标来捕捉音乐属性的细微变化。客观验证和一项人类研究证明了这些指标的有效性。此外，对不同音乐编辑系统的案例研究也展示了其应用效果。

    arXiv:2512.14629v2 Announce Type: replace-cross  Abstract: Music editing plays a vital role in modern music production, with applications in film, broadcasting, and game development. Recent advances in music editing systems have enabled diverse editing tasks such as timbre transfer, instrument substitution, and genre transformation. However, many existing works overlook evaluating their ability to preserve musical facets that should remain unchanged during editing, which we define as Music Context Preservation (MuseCP). While some studies do consider MuseCP, their evaluation protocols and metrics are not comprehensive. To address this, we introduce the first MuseCP evaluation framework, MuseCPEval, that covers four categories of music facets with fine-grained and well-tailored metrics to capture nuanced changes in music attributes. Objective validation and a human study demonstrate the effectiveness of these metrics. Moreover, the case studies on diverse music editing systems illustrat
    
[^259]: 专业软件开发者不随波逐流，而是掌控：2025年AI代理在编码中的应用

    Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025

    [https://arxiv.org/abs/2512.14012](https://arxiv.org/abs/2512.14012)

    经验丰富的开发者将AI代理视为生产力工具，但坚持保留设计控制权，通过专业知识策略性地引导代理行为，以确保软件质量。

    

    arXiv:2512.14012v2 公告类型：替换-交叉 摘要：AI代理的兴起正在改变软件的构建方式。代理的承诺是开发者可以更快地编写代码，将多个任务委托给不同的代理，甚至仅通过自然语言编写完整的软件。然而，在实际中，代理在专业软件开发中扮演何种角色仍是个问题。本文调查了经验丰富的开发者如何在构建软件时使用代理，包括他们的动机、策略、任务适宜性和情感。通过现场观察（N=13）和定性调查（N=99），我们发现，虽然经验丰富的开发者重视代理作为生产力提升工具，但他们出于对基本软件质量属性的坚持，保留了自己在软件设计和实现中的主导权，并利用专业知识采用控制代理行为的策略。此外，经验丰富的开发者喜欢与代理合作，将其视为协作的源泉。

    arXiv:2512.14012v2 Announce Type: replace-cross  Abstract: The rise of AI agents is transforming how software can be built. The promise of agents is that developers might write code quicker, delegate multiple tasks to different agents, and even write a full piece of software purely out of natural language. In reality, what roles agents play in professional software development remains in question. This paper investigates how experienced developers use agents in building software, including their motivations, strategies, task suitability, and sentiments. Through field observations (N=13) and qualitative surveys (N=99), we find that while experienced developers value agents as a productivity boost, they retain their agency in software design and implementation out of insistence on fundamental software quality attributes, employing strategies for controlling agent behavior leveraging their expertise. In addition, experienced developers enjoy working with agents as source of collaboration 
    
[^260]: 大型语言模型在Verilog代码生成中的应用：文献综述与未来展望

    Large Language Model for Verilog Code Generation: Literature Review and the Road Ahead

    [https://arxiv.org/abs/2512.00020](https://arxiv.org/abs/2512.00020)

    本文首次全面综述了大型语言模型在Verilog代码生成中的研究进展，填补了该领域缺乏系统性文献综述的空白，并展望了未来发展方向。

    

    摘要：代码生成已成为软件工程（SE）与人工智能（AI）交叉领域的一个关键研究热点，吸引了学术界和工业界的广泛关注。在这一广阔领域中，Verilog作为一种代表性的硬件描述语言（HDL），在数字电路设计和验证中发挥着基础性作用，因此其自动化生成对电子设计自动化（EDA）尤为重要。近年来，研究日益聚焦于将大型语言模型（LLMs）应用于Verilog代码生成，特别是在寄存器传输级（RTL）层面，探索如何将这些AI驱动技术有效集成到硬件设计流程中。尽管已有大量研究探索了LLM在该领域的应用，但文献中仍缺乏一份全面综述来整合这些发展。本综述旨在填补这一空白。

    arXiv:2512.00020v3 Announce Type: replace-cross  Abstract: Code generation has emerged as a critical research area at the intersection of Software Engineering (SE) and Artificial Intelligence (AI), attracting significant attention from both academia and industry. Within this broader landscape, Verilog, as a representative hardware description language (HDL), plays a fundamental role in digital circuit design and verification, making its automated generation particularly significant for Electronic Design Automation (EDA). Consequently, recent research has increasingly focused on applying Large Language Models (LLMs) to Verilog code generation, particularly at the Register Transfer Level (RTL), exploring how these AI-driven techniques can be effectively integrated into hardware design workflows. Despite substantial research efforts have explored LLM applications in this domain, a comprehensive survey synthesizing these developments remains absent from the literature. This review fill add
    
[^261]: CausalProfiler：生成合成基准以对因果机器学习进行严谨透明评估

    CausalProfiler: Generating Synthetic Benchmarks for Rigorous and Transparent Evaluation of Causal Machine Learning

    [https://arxiv.org/abs/2511.22842](https://arxiv.org/abs/2511.22842)

    本文提出了CausalProfiler，一个能随机生成具有覆盖保证和透明假设的合成因果基准的工具，从而实现对因果机器学习方法的严谨和透明评估。

    

    因果机器学习（Causal ML）旨在利用机器学习算法回答“如果……会怎样”的问题，使其成为高风险决策中颇具前景的工具。然而，因果机器学习中的实证评估实践仍然有限。现有基准往往依赖于少数手工制作或半合成数据集，导致结论脆弱且缺乏普遍性。为弥补这一差距，我们引入了CausalProfiler，一个用于因果机器学习方法的合成基准生成器。基于关于因果模型类别、查询和所考虑数据的一系列明确设计选择，CausalProfiler随机采样因果模型、数据、查询和构成合成因果基准的真实值。通过这种方式，可以在各种条件下对因果机器学习方法进行严谨透明的评估。这项工作提供了首个具有覆盖保证和透明假设的合成因果基准随机生成器。

    arXiv:2511.22842v3 Announce Type: replace-cross  Abstract: Causal machine learning (Causal ML) aims to answer "what if" questions using machine learning algorithms, making it a promising tool for high-stakes decision-making. Yet, empirical evaluation practices in Causal ML remain limited. Existing benchmarks often rely on a handful of hand-crafted or semi-synthetic datasets, leading to brittle, non-generalizable conclusions. To bridge this gap, we introduce CausalProfiler, a synthetic benchmark generator for Causal ML methods. Based on a set of explicit design choices about the class of causal models, queries, and data considered, the CausalProfiler randomly samples causal models, data, queries, and ground truths constituting the synthetic causal benchmarks. In this way, Causal ML methods can be rigorously and transparently evaluated under a variety of conditions. This work offers the first random generator of synthetic causal benchmarks with coverage guarantees and transparent assumpt
    
[^262]: 草堆中的越狱

    Jailbreaking in the Haystack

    [https://arxiv.org/abs/2511.04707](https://arxiv.org/abs/2511.04707)

    NINJA是一种通过将良性模型生成内容附加到有害目标上，并利用有害目标位置影响安全性的低资源、可迁移且难以检测的越狱攻击方法，显著提升了多种先进模型的攻击成功率。

    

    摘要：arXiv:2511.04707v2 公告类型：替换-交叉 摘要：近期长上下文语言模型（LMs）的进展使得百万级令牌输入成为可能，扩展了它们在计算机使用代理等复杂任务中的能力。然而，这些扩展上下文的潜在安全影响仍不明确。为弥补这一空白，我们引入了NINJA（即“草堆中的针”越狱攻击的缩写），这是一种通过将良性、模型生成的内容附加到有害用户目标上来越狱对齐LMs的方法。我们方法的关键在于观察到有害目标的位置在安全性中扮演重要角色。在标准安全基准HarmBench上的实验表明，NINJA显著提高了包括LLaMA、Qwen、Mistral和Gemini在内的最先进开源和专有模型的攻击成功率。与先前的越狱方法不同，我们的方法资源需求低、可迁移性强且更难以检测。此外，我们展示了NINJA是计算最优的——在固定计算预算下，增加...

    arXiv:2511.04707v2 Announce Type: replace-cross  Abstract: Recent advances in long-context language models (LMs) have enabled million-token inputs, expanding their capabilities across complex tasks like computer-use agents. Yet, the safety implications of these extended contexts remain unclear. To bridge this gap, we introduce NINJA (short for Needle-in-haystack jailbreak attack), a method that jailbreaks aligned LMs by appending benign, model-generated content to harmful user goals. Critical to our method is the observation that the position of harmful goals play an important role in safety. Experiments on standard safety benchmark, HarmBench, show that NINJA significantly increases attack success rates across state-of-the-art open and proprietary models, including LLaMA, Qwen, Mistral, and Gemini. Unlike prior jailbreaking methods, our approach is low-resource, transferable, and less detectable. Moreover, we show that NINJA is compute-optimal -- under a fixed compute budget, increasi
    
[^263]: 睡着的凯莉

    Sleeping Kelly

    [https://arxiv.org/abs/2510.15911](https://arxiv.org/abs/2510.15911)

    本文通过凯利准则重新审视睡美人问题，发现理性决策者应最大化财富增长率而非期望值，从而得出事前哈尔弗派和事后三分派的结论，并避免历时荷兰赌。

    

    睡美人问题是一个关于不完美记忆的问题，受到了广泛关注。解决睡美人问题的一种方法是允许睡美人基于其信念做出决策，然后刻画她的决策在何种条件下是“理性的”。特别是，可以允许她基于信念进行金钱投注，并假设她希望增加财富而非损失财富。然而，这种方法常常与一个错误假设相伴，即睡美人应最大化其投注的期望值。在此，我们推断当睡美人使用凯利准则最大化其财富增长率时的概率，以表明睡着的凯莉是一个事前哈尔弗派和事后三分派，并且不受历时荷兰赌的影响。

    arXiv:2510.15911v3 Announce Type: replace-cross  Abstract: The Sleeping Beauty problem is a problem of imperfect recall that has received considerable attention. One approach to solving the Sleeping Beauty problem is to allow Sleeping Beauty to make decisions based on her beliefs, and then characterize what it takes for her decisions to be "rational". In particular, she can be allowed to make monetary bets based on her beliefs, with the assumption that she wants to gain wealth rather than lose it. However, this approach is often coupled with the erroneous assumption that Sleeping Beauty should maximize the expected value of her bets. Here, we infer probabilities when Sleeping Beauty maximizes the expected growth rate of her wealth using the Kelly Criterion, to show that Sleeping Kelly is an ex ante Halfer and de se Thirder and impervious diachronic Dutch Books.
    
[^264]: 混合强化学习与搜索的飞行轨迹规划

    Hybrid Reinforcement Learning and Search for Flight Trajectory Planning

    [https://arxiv.org/abs/2509.04100](https://arxiv.org/abs/2509.04100)

    该方法通过强化学习预计算近最优路径来约束搜索求解器，显著加速飞行轨迹规划，燃油消耗偏差在1%以内，计算速度提升高达50%。

    

    本文探讨了将强化学习（RL）与基于搜索的路径规划器相结合，以加速航空公司飞行路径的优化，在紧急情况下快速重新计算路线可能至关重要。基本思想是训练一个RL代理，基于位置和大气数据预计算近最优路径，并在运行时利用这些路径来约束底层路径规划求解器，从而在初始猜测的特定距离内找到解决方案。该方法有效减少了求解器的搜索空间，显著加快了路径优化速度。虽然不保证全局最优性，但使用空客飞机性能模型进行的实证结果表明，燃油消耗与无约束求解器几乎相同，偏差通常控制在1%以内。同时，与传统方法相比，计算速度可提升高达50%。

    arXiv:2509.04100v3 Announce Type: replace  Abstract: This paper explores the combination of Reinforcement Learning (RL) and search-based path planners to speed up the optimization of flight paths for airliners, where in case of emergency a fast route re-calculation can be crucial. The fundamental idea is to train an RL Agent to pre-compute near-optimal paths based on location and atmospheric data and use those at runtime to constrain the underlying path planning solver and find a solution within a certain distance from the initial guess. The approach effectively reduces the size of the solver's search space, significantly speeding up route optimization. Although global optimality is not guaranteed, empirical results conducted with Airbus aircraft's performance models show that fuel consumption remains nearly identical to that of an unconstrained solver, with deviations typically within 1%. At the same time, computation speed can be improved by up to 50% as compared to using a conventio
    
[^265]: 迭代流匹配：路径校正与逐步细化以增强生成建模

    Iterative Flow Matching: Path Correction and Gradual Refinement for Enhanced Generative Modeling

    [https://arxiv.org/abs/2502.16445](https://arxiv.org/abs/2502.16445)

    本文提出一种可集成到任意生成模型的迭代流匹配方法，通过路径校正和逐步细化来减少幻觉，提升图像生成的稳健性。

    

    图像生成的生成模型现已广泛应用于从娱乐引导图像生成到解决逆问题的各种场景。然而，训练生成器是一项艰巨的任务，需要精细调优，并可能导致所谓的“幻觉”，即生成不切实际的图像。在本工作中，我们探索使用流匹配进行图像生成。我们解释并证明了流匹配为何会产生幻觉，并提出了一种迭代过程来改进生成过程。我们的迭代过程可集成到几乎任何生成建模技术中，从而增强图像合成系统的性能和鲁棒性。

    arXiv:2502.16445v4 Announce Type: replace-cross  Abstract: Generative models for image generation are now commonly used for a wide variety of applications, ranging from guided image generation for entertainment to solving inverse problems. Nonetheless, training a generator is a non-trivial feat that requires fine-tuning and can lead to so-called hallucinations, that is, the generation of images that are unrealistic. In this work, we explore image generation using flow matching. We explain and demonstrate why flow matching can generate hallucinations, and propose an iterative process to improve the generation process. Our iterative process can be integrated into virtually any generative modeling technique, thereby enhancing the performance and robustness of image synthesis systems.
    
[^266]: 从提示到扰动：针对音频大语言模型的基于语音的越狱攻击自适应框架

    `From Prompt to Perturbation': An Adaptive Framework for Voice-Based Jailbreaks on Audio LLMs

    [https://arxiv.org/abs/2502.00735](https://arxiv.org/abs/2502.00735)

    本文提出了一种自适应越狱攻击框架，能在统一设置下系统评估级联流水线和端到端音频大语言模型，覆盖更广泛的音频攻击空间。

    

    随着大语言模型（LLMs）越来越多地集成到基于音频的应用中，人们对其易受音频对抗攻击的脆弱性日益担忧。这些系统通常遵循两种架构范式：级联流水线（其中自动语音识别将音频输入转换为文本，再交由LLM处理）和端到端的大音频语言模型（LALMs，直接解释原始音频信号）。除了架构差异外，级联流水线主要容易受到通过语音传递的文本级越狱策略的攻击，而端到端LALMs则引入了额外的声学语义攻击向量。然而，现有研究往往聚焦于单一范式，对更广泛的音频攻击空间覆盖有限。为弥补这一差距，我们提出了一个自适应越狱攻击框架，用于在统一设置下对级联流水线和LALMs进行系统评估。

    arXiv:2502.00735v4 Announce Type: replace-cross  Abstract: As large language models (LLMs) are increasingly integrated into audio-based applications, growing concerns have emerged regarding their vulnerability to audio-based adversarial attacks. These systems typically follow two architectural paradigms: cascaded pipelines, where automatic speech recognition converts audio inputs into text before LLM processing, and end-to-end large audio-language models (LALMs), which directly interpret raw audio signals. Beyond architectural differences, cascaded pipelines are primarily vulnerable to text-level jailbreak strategies delivered through speech, whereas end-to-end LALMs introduce additional acoustic-semantic attack vectors. However, existing studies often focus on a single paradigm and provide limited coverage of the broader audio attack space. To bridge this gap, we propose an adaptive jailbreak attack framework for systematic evaluation of both cascaded pipelines and LALMs under a unifi
    
[^267]: 使用约束贝叶斯优化实现机器学习算法的自动化计算能耗最小化

    Automated Computational Energy Minimization of ML Algorithms using Constrained Bayesian Optimization

    [https://arxiv.org/abs/2407.05788](https://arxiv.org/abs/2407.05788)

    本文提出使用约束贝叶斯优化（CBO）在保证模型泛化性能不低于阈值的前提下，自动最小化机器学习算法的训练能耗，并在回归和分类任务上验证了其有效性。

    

    贝叶斯优化（BO）是一种高效的框架，用于在函数评估成本高昂且梯度信息不易获取时优化黑盒目标。BO已成功应用于自动化机器学习（ML）模型中的超参数优化（HPO）任务，其主要目标是优化在保留数据上的预测性能。然而，近年来，随着模型规模的不断增长，与模型训练相关的能耗已成为ML应用中的一个重要因素。在此，我们评估了约束贝叶斯优化（CBO），其主要目标是最小化能耗，并受限于泛化性能高于某个阈值的约束。我们在回归和分类任务上评估了我们的方法，并证明CBO在实现更低能耗的同时不损害ML模型的预测性能。

    arXiv:2407.05788v2 Announce Type: replace-cross  Abstract: Bayesian optimization (BO) is an efficient framework for optimization of black-box objectives when function evaluations are costly and gradient information is not easily accessible. BO has been successfully applied to automate the task of hyperparameter optimization (HPO) in machine learning (ML) models with the primary objective of optimizing predictive performance on held-out data. In recent years, however, with ever-growing model sizes, the energy cost associated with model training has become an important factor for ML applications. Here we evaluate Constrained Bayesian Optimization (CBO) with the primary objective of minimizing energy consumption and subject to the constraint that the generalization performance is above some threshold. We evaluate our approach on regression and classification tasks and demonstrate that CBO achieves lower energy consumption without compromising the predictive performance of ML models.
    

