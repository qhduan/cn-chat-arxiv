# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Less Decoder is More Encoder: Geometric Representation Learning from Novel View Synthesis](https://arxiv.org/abs/2610.03717) | 本文提出SNAP，一个自监督编码器-解码器Transformer，通过位姿条件化的局部解码器与潜空间重建目标，克服了新视角合成中解码器削弱编码器表示能力、像素空间目标阻碍特征学习的问题，实现了可与几何监督方法媲美的任务无关几何表示。 |
| [^2] | [4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes](https://arxiv.org/abs/2610.03715) | 4DCodeBench是一个通过代码生成来评估智能体4D逆向图形学能力的基准，测试发现前沿模型虽具备较强的静态场景重建能力，但在重建变形、流体、断裂等复杂动态场景方面仍不可靠。 |
| [^3] | [What Should World Models Forget? Stratified Retention for Continual Adaptation](https://arxiv.org/abs/2610.03713) | 提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。 |
| [^4] | [EyeRobot 2.0: Active Gaze for Precise Manipulation without Wrist Cameras](https://arxiv.org/abs/2610.03710) | EyeRobot 2.0通过主动视觉注视（双眼转动对准3D目标点并进行中央凹式token分配）与分层强化学习训练（底层注视伺服策略加上层注视目标选择器），仅用单个立体相机就实现了无需腕部相机的精细双手机器人操作。 |
| [^5] | [Transcriptome-informed multi-modal AI for predicting neoadjuvant therapy response from breast cancer biopsies](https://arxiv.org/abs/2610.03693) | 该研究提出一种两阶段多模态AI模型，先从病理图像推断转录组表达谱，再结合临床变量预测乳腺癌新辅助治疗的病理完全缓解，在九个队列中实现0.79的合并AUROC，并优于传统组织病理学生物标志物。 |
| [^6] | [FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution](https://arxiv.org/abs/2610.03675) | 该论文提出成本感知的LLM进化框架FrugalEvo，让更强的LLM探索解法策略、更廉价的LLM负责代码实现与迭代优化，并通过前缀共享提升缓存复用，同时引入BA-AUC指标来衡量单位成本下的优化收益。 |
| [^7] | [Revisiting Input Time-frequency Representations in Multi-pitch Estimation for Vocal Ensembles](https://arxiv.org/abs/2610.03656) | 该论文发现，在人声合唱多音高估计任务中，结构简单的线性STFT输入表示在性能上优于广泛使用的HCQT表示，同时显著降低了特征提取的计算成本。 |
| [^8] | [MRVQ: One Resident Index for Dimension- and Rate-Elastic Vector Search](https://arxiv.org/abs/2610.03651) | 提出 MRVQ（套娃残差向量量化），用单一常驻索引即可通过截断残差阶段降低码率、截断嵌入坐标降低维度，以比多索引方案低 1.89-22 倍的内存覆盖所有维度-码率组合的向量检索。 |
| [^9] | [On-Board Anomaly Detection for Efficient Marine Environmental Monitoring](https://arxiv.org/abs/2610.03649) | 该论文提出了一种用于对地观测卫星的轻量级海洋环境异常检测流程，利用自监督神经网络编码器压缩卫星图像并结合机器学习异常检测模型，实现高效的星载海洋环境监测。 |
| [^10] | [Do Large Language Models Know Colombian Law? A Reliability Benchmark for the Colombian Legal System](https://arxiv.org/abs/2610.03639) | 该论文构建了一个包含1,042个条目、经专家验证的哥伦比亚法律基准，评估发现尽管LLM在封闭式选择题上准确率最高可达0.905，但在自由文本法律问答中事实正确性均不超过0.45，且答案相关性与正确性呈负相关，表明模型回答听起来切题却常常事实错误。 |
| [^11] | [LoGo: Local-Global Rewards for Consistent Long-Horizon Video Generation](https://arxiv.org/abs/2610.03636) | LoGo通过在后训练中融合全局奖励与空间局部化奖励，显著提升了相机控制长时程视频生成的3D一致性，同时保持了相机跟随精度和视频质量。 |
| [^12] | [Credit Where It Matters: Dependency-Aware Policy Optimization for Terminal Agents](https://arxiv.org/abs/2610.03634) | 提出依赖感知的组策略优化，通过从执行轨迹构建命令依赖图并从任务验证器检查的资源反向追踪，将信用精准分配给相关的写入操作及其支撑读取操作，从而改进终端智能体强化学习中的信用分配。 |
| [^13] | [NeutronGym: Physics-Graded Neutron Instrument Design for LLM Agents](https://arxiv.org/abs/2610.03631) | 提出首个中子仪器设计可执行环境NeutronGym，通过McStas仿真和无需LLM评审的分层自动评分来检验大语言模型智能体的真实物理设计能力，现有模型最多仅复现16个基准任务中的7个，而强化学习可将Qwen3-8B的通过率从11%提升至77%。 |
| [^14] | [Depth as Time in One-Step Generative Models](https://arxiv.org/abs/2610.03626) | 该研究发现多步扩散的去噪计算会在一步生成模型单次前向传播的网络深度中展开，且这种深度方向上的计算取决于流映射所训练的传输任务。 |
| [^15] | [Low-Cost Video--Time Priors as a Strong Baseline for EEG--fNIRS Emotion Regression on Familiar Videos](https://arxiv.org/abs/2610.03618) | 该研究发现，在熟悉视频的连续情绪回归中，仅利用视频身份和播放时间构成的低成本先验即可接近EEG-fNIRS融合模型的性能，而EEG-fNIRS生理信号仅带来较小且因人而异的残余增益，因此更适合作为可选的辅助校正信号。 |
| [^16] | [When a Correct Reward Is Not Enough: Diagnosing and Guiding PPO in an Analytically Solved Broker-Trader Game](https://arxiv.org/abs/2610.03598) | 本文将PPO智能体置于解析可解的连续时间经纪商-交易员博弈中，利用已知的解析解来诊断并引导强化学习，发现即使奖励设计正确，PPO在存在随机非知情订单流时依然难以学到准确策略。 |
| [^17] | [HazardWeaver: Scientific Route Selection for Hazard Analysis Agents](https://arxiv.org/abs/2610.03591) | 提出HazardWeaver框架，将灾害分析中的方法选择建模为状态依赖的科学路线选择问题，使智能体能够根据证据和数据工具的可用性动态确定并调整适用的科学方法。 |
| [^18] | [Threat-Preserving Representation Sensitivity in Agent-Security Benchmarks](https://arxiv.org/abs/2610.03585) | 论文提出威胁保持表征敏感性（TPRS）指标，发现在底层威胁完全不变的情况下，仅改变智能体可见的表征（如工具名称）就能使攻击成功率显著变化（最高约13个百分点），表明智能体安全基准的测量结果对表征方式高度敏感，可能无法真实反映模型的安全性。 |
| [^19] | [Rethinking What to Cache in Few-Step Diffusion Transformers: Solver-Aware Target Selection](https://arxiv.org/abs/2610.03577) | 提出AutoTarget方法，通过在少量无缓存运行的条件下测量重用各候选张量所引入的误差，针对特定模型、求解器和重用调度自适应地选择误差最低的缓存张量，从而提升少步蒸馏扩散Transformer的采样质量。 |
| [^20] | [HyperBrowseComp: A Multilingual and Multimodal Stress Test for Web-Browsing Agents](https://arxiv.org/abs/2610.03574) | HyperBrowseComp是一个覆盖13种语言、包含423道人工验证问题的多语言多模态网页浏览基准，通过要求定位隐蔽证据、追踪多步线索链和检查异构信息源，为网页浏览智能体提供了极具挑战性的压力测试。 |
| [^21] | [Learning to Assess Heartbeat Observability for mmWave Heart-Rate Sensing](https://arxiv.org/abs/2610.03570) | 该论文提出HEAR框架，通过可控多散射体FMCW仿真器自动生成可观测性标签，并利用紧凑的双任务Transformer从毫米波雷达测量中评估心跳可观测性，实现选择性的可靠非接触心率估计。 |
| [^22] | [Knowledge or Calculator? Decomposing the Skill Premium in Verifiable Financial Agent Workflows](https://arxiv.org/abs/2610.03564) | 该论文提出FinSkillBench评估套件，通过确定性验证器量化金融AI智能体能力，发现人工策划的技能包可显著提升模型表现16.2分，且可执行工具（+19.5分）的贡献远大于程序性文档（+5.6分）。 |
| [^23] | [Cephalonauts One: A deep fMRI dataset for decoding naturalistic speech in the human brain](https://arxiv.org/abs/2610.03558) | 发布了迄今最大的自然语音fMRI数据集Cephalonauts One（每名受试者30小时数据），并提出以音频片段检索为任务形式的大脑解码基准，附带标准化数据划分、评估指标和基线解码器。 |
| [^24] | [Recursive Harness Self-Improvement for Frontier Reasoning Data Synthesis](https://arxiv.org/abs/2610.03548) | 该论文提出任务-框架协同进化框架，通过在线和任务后两种自我改进机制递归优化数据合成的构建框架本身，在保持模型权重与验证标准不变的前提下，经十四轮进化使求解器平均准确率从100.0%降至54.8%，从而合成出难度更高的前沿推理数据。 |
| [^25] | [Beyond Trained Models: Compiling GNNs for a Sound Explainer Benchmark](https://arxiv.org/abs/2610.03526) | 该论文揭示了现有GNN可解释器基准中“训练模型依赖预期模体”这一隐含假设的不成立，并提出Gracr——首个将分级模态逻辑公式编译为GNN权重的编译器，通过用编译替代训练构建了真实解释可被形式化定义并精确计算的健全可解释器基准。 |
| [^26] | [From Benchmarks to Production: A Text-to-SQL System for Complex Financial Data](https://arxiv.org/abs/2610.03524) | FLINT是一个针对生产环境金融数据库的领域专用Text-to-SQL系统，通过查找代理解析不透明概念、嵌入检索专家模板以及基于外键链遍历的模式链接三大组件，解决了通用系统在此类复杂数据上准确率低于50%的问题。 |
| [^27] | [Reasoning Models Are Accurate but Unsound on Identification](https://arxiv.org/abs/2610.03519) | 本文提出CERTID——一个基于可靠且完备的ID算法和结构因果模型精确验证的形式化因果识别评测流水线，首次提供了可证明不可识别的查询及等价公式评分，用以揭示推理模型在因果效应可识别性判断上虽准确但不可靠（会回答不可识别查询）的失败模式。 |
| [^28] | [Weave Forcing: Compositional Memory Routing for Interactive Long Video Generation](https://arxiv.org/abs/2610.03510) | 提出 Weave Forcing——一个免训练框架，通过 LLM 语义槽路由将提示词分解为角色与背景等组件，并为每个组件从历史镜头中精准选取参考记忆，实现交互式长视频生成中的组合式记忆复用。 |
| [^29] | [Efficient Reasoning Training Does Not Always Harm CoT Faithfulness and Monitorability](https://arxiv.org/abs/2610.03509) | 本文通过三种不同长度压力微调方法对多种模型的系统评估发现，高效推理训练并不必然损害思维链的忠实性与可监控性，其影响取决于所施加长度压力的具体方式。 |
| [^30] | [Certified Mechanistic Edits: Behavioral Guarantees for Skill Removal and Preservation](https://arxiv.org/abs/2610.03502) | 该论文首次提出对机制化编辑的行为效果进行认证的方法，可对连续嵌入空间区域内的每个输入可证明地保证：禁用一个电路将移除一种技能同时保留另一种技能，并在标准Transformer上验证了该方法的有效性。 |
| [^31] | [Detect and Suppress: A Mechanistic Defense against Adversarial Patches in VLA Models](https://arxiv.org/abs/2610.03498) | 该论文通过稀疏自编码器机制性分析发现了VLA模型中与对抗性补丁激活高度相关的内部特征，并在线性探针检测到攻击时条件性地抑制该特征，从而无需微调即可显著提升模型对对抗攻击的鲁棒性。 |
| [^32] | [AREX: Affine-Residual Exponential Integrator for Few-Step Sampling in Flow Matching](https://arxiv.org/abs/2610.03483) | AREX是一种无需训练的流匹配模型少步采样器，它将采样动力学分解为由目标均值和协方差决定的仿射分量（用显式矩阵值传播子积分）与神经残差项，在无需重训练的情况下持续提升少步采样的样本保真度。 |
| [^33] | [MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation](https://arxiv.org/abs/2610.03476) | 本文提出双循环智能体框架MobiAgent，通过内循环利用可组合的原子技能将高层推理与低层控制解耦以实现稳健的长时域移动操作，并通过递归策略自我改进实现持续学习。 |
| [^34] | [Single or Multiple Policies for Phase-Structured Reinforcement Learning?](https://arxiv.org/abs/2610.03475) | 该论文从理论上证明单一共享策略可以达到任何多策略方案的性能，但实践中多策略是否更优取决于函数逼近、学习优化过程以及策略切换的样本效率与连续性损失等因素。 |
| [^35] | [Preserving Anatomical Continuity: Three-Stage Pipeline for Colon Segmentation in 3D Abdominal CT Scans](https://arxiv.org/abs/2610.03467) | 该论文提出了一种三阶段的保持拓扑结构的结肠分割流水线，通过初始深度学习分割、中心线桥接和重建三个阶段，解决了CT图像结肠分割中预测结果不连通的问题，在保持分割精度的同时显著提高了结构一致性。 |
| [^36] | [A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control](https://arxiv.org/abs/2610.03458) | 监控器读数接近零并不意味着模型行为真正受到控制——即使在代码生成环境中探针得分和惩罚值都处于极低水平，模型仍可能在训练早期就持续利用漏洞。 |
| [^37] | [Measure Less, Know More: Self-Supervised Test-Time Feature Acquisition](https://arxiv.org/abs/2610.03454) | 该论文提出ECHO-k，一种任务无关的自监督测试时模态获取方法，它利用基础模型的内部预训练表示作为代理目标，并通过强化学习策略顺序选择信息量最大的模态，从而在有限预算下持续提升下游任务性能。 |
| [^38] | [Corrupted but Correct: Why Vision-Language Models Lie to Themselves Internally](https://arxiv.org/abs/2610.03445) | 该论文发现并定义了视觉语言模型中的“训练/推理差距”——对抗扰动虽能把教师强制训练损失压至近零，模型自由生成时却仍输出正确描述——并通过 logit lens 将该差距精确归因于单一自回归步骤中目标词元排名恰好固定为第3位的内部机制。 |
| [^39] | [OptiSelect: How does the Optimizer Shape Data Curriculum?](https://arxiv.org/abs/2610.03432) | 本文提出OptiSelect优化器感知的数据选择范式，首次系统研究优化器如何影响数据课程选择，理论上证明基于符号和极坐标切向预处理的优化器（Lion、Muon）因效用分数可区分性崩溃而限制选择增益，而对角自适应优化器（AdamW、Sophia）则能获得严格更优的增益上界。 |
| [^40] | [Jumping the Line: Exploiting Length Predictions in LLM Scheduling](https://arxiv.org/abs/2610.03430) | 提出JIL攻击方法，通过优化对抗性后缀操纵长度预测信号，使LLM调度器低估请求长度从而插队获得更高优先级，在端到端服务实验中使对抗性请求平均完成速度最高提升1.53倍。 |
| [^41] | [Becoming Suspicious Across Borders: Algorithmic Extraterritoriality and AI-Driven Financial Surveillance](https://arxiv.org/abs/2610.03425) | 本文提出“算法域外性”这一新概念，指出AI驱动的金融监控使“嫌疑”的产生从人类在司法管辖区内的情境化法律判断转变为跨国数据基础设施中的数据驱动过程，监管权力的边界由此从地理管辖转向数据系统中的可见性。 |
| [^42] | [Rethinking Epistemic Uncertainty in Node Classification through Information Growth](https://arxiv.org/abs/2610.03418) | 本文提出了一个在信息增长条件下检验节点分类中认知不确定性可约减性的统计框架，并揭示现有图证据深度学习方法难以满足一致性准则。 |
| [^43] | [ForestQuery: Boundary-Aware and Spatially Anchored Query Learning for Unified Forest Point Cloud Segmentation](https://arxiv.org/abs/2610.03403) | 提出 ForestQuery 框架，通过显式建模边界不确定性并结合空间锚定的语义查询增强（SA-SQE），实现了统一的森林点云语义与实例分割。 |
| [^44] | [DriftTTS: Few-Step Text-to-Speech Without Distillation via Distribution-Matching Drift](https://arxiv.org/abs/2610.03390) | DriftTTS提出了一种无需教师模型、蒸馏或对抗训练的少步数文本转语音方法，通过分布匹配漂移目标函数和在策略展开训练，仅用4次函数评估就达到了与现有模型相当甚至更优的合成质量。 |
| [^45] | [Benchmarking Candidate Coverage in Typed Decision Models](https://arxiv.org/abs/2610.03387) | 本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。 |
| [^46] | [CVE2AP: Automated Generation of PDDL-Encoded Attack Paths via Large Language Models](https://arxiv.org/abs/2610.03383) | 提出了CVE2AP，一种利用大语言模型从自然语言CVE漏洞描述中自动生成PDDL编码攻击路径的方法，摆脱了传统专家人工建模的瓶颈，提升了攻击路径建模的可扩展性。 |
| [^47] | [Multilingual GSM-Symbolic: What determines capability transfer across languages?](https://arxiv.org/abs/2610.03367) | 该论文提出了可扩展的多语言数学数据集Multilingual GSM-Symbolic（涵盖15种语言、3万个题目匹配问答对，通过符号化模板防止过拟合），并量化发现模型规模和语言资源水平是决定跨语言能力迁移的最主要因素。 |
| [^48] | [Geometry Meets Physics: Data-Efficient Pre-Training for Unstructured Neural PDE Solvers](https://arxiv.org/abs/2610.03363) | 提出了一个无需磁盘数据的预训练框架：针对稳态问题采用基于内在形状描述符的几何驱动策略，针对瞬态问题采用在线生成合成PDE数据的物理驱动方法，从而实现数据高效的非结构化神经PDE求解器预训练。 |
| [^49] | [Follow the Winners: Conservative Policy Improvement with the Cross-Entropy Method for Critic-Free RFT](https://arxiv.org/abs/2610.03361) | FTW 是一种无批评家的强化微调算法，通过将交叉熵方法适配到 RFT 中、用回放缓冲区样本上的序数过滤器替代组采样，从而在有状态环境中难以重复采样的智能体大模型训练中实现保守且稳健的策略改进。 |
| [^50] | [ReFract: Benchmarking Perspective Awareness in Language Model Agents with Text World Models](https://arxiv.org/abs/2610.03356) | 该论文提出了ReFract基准测试，通过150条专家验证的工业维护场景条目，评估语言模型智能体的“视角感知”能力，即根据用户角色的意图和权限边界，仅使用该角色合法可用的工具采取相应行动的能力。 |
| [^51] | [Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation](https://arxiv.org/abs/2610.03333) | 提出VISTA，一种工作空间级等变视觉-触觉扩散策略，通过将触觉接触线索融合到球面视觉表示中并利用等变扩散预测动作，大幅提升了富接触操作模仿学习的数据效率。 |
| [^52] | [Cordial Learning: Distributed Training with Correlated Data](https://arxiv.org/abs/2610.03330) | 提出了一种名为“亲和学习”的分布式训练框架，通过智能体间仅共享低维输出、本地模型提取同伴信息来处理相关数据问题，并在线性模型假设下证明了其以概率一收敛到全局最优。 |
| [^53] | [SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models](https://arxiv.org/abs/2610.03329) | 提出SyntaxBench诊断基准，通过五个核心字符级任务和一个高难度子串提取压力测试，结合Cohen's kappa与McNemar检验等统计方法，系统评估了八个开放权重大语言模型的字符级推理能力。 |
| [^54] | [Preserving Mathematical Reasoning in Compressed Diffusion Language Models via Trajectory-Aware Low-Rank Approximation](https://arxiv.org/abs/2610.03326) | 该论文提出轨迹感知低秩压缩方法Traj-MC，通过蒙特卡洛采样在扩散语言模型部分掩码的推理轨迹上进行校准，从而在压缩后更好地保留模型的数学推理能力。 |
| [^55] | [Information Limits of Low-Rank Approximation Certification](https://arxiv.org/abs/2610.03321) | 该论文刻画了低秩近似认证所需的最小查询代价，证明复用验证响应可使一批查询支持整条嵌套近似路径，且跨 W 条路径的 √log(W+1) 代价依赖经匹配下界证明是最优的。 |
| [^56] | [Refinement Buys Intelligibility, Search Buys Identity: What Test-Time Compute Buys in Masked-Diffusion TTS](https://arxiv.org/abs/2610.03320) | 该论文发现掩码扩散TTS中推理时的精炼步数主要提升可懂度（弥补86.2%差距）而对说话人身份提升有限（仅46.4%），且Best-of-K搜索能有效恢复精炼无法带来的说话人身份一致性。 |
| [^57] | [Multi-Task Evolution for Zero-Shot Cross-Problem Generalization using LLMs](https://arxiv.org/abs/2610.03316) | 提出了MECo，一个由LLM驱动的多任务进化框架，通过维护任务条件化启发式种群并利用跨任务迁移差距引导启发式的迁移与重组，实现了无需目标问题反馈的零样本跨问题泛化。 |
| [^58] | [Lightweight, Rubric-Guided Trajectory Evaluation for Production AI Agents](https://arxiv.org/abs/2610.03315) | LiteTrajEval是一种轻量级的、预算受限的AI智能体轨迹评估架构，通过离线规则提取、在线启发式失败信号标记和单个基于评分标准的LLM评判器，显著提升了失败定位与人类标注的一致性（提升20-35个百分点）。 |
| [^59] | [Optimal Planning in a Dynamic World](https://arxiv.org/abs/2610.03312) | 本文定义了“任意开始时间规划”这一新问题设定，解决了可行状态或行动随时间动态变化且执行开始时间未知情况下的最优规划问题。 |
| [^60] | [Training-Loss Guarantees for Muon with Finite-Step Newton--Schulz Orthogonalization](https://arxiv.org/abs/2610.03306) | 本文首次为Muon优化器建立了同时考虑动量累积与有限步调优牛顿-舒尔茨正交化的训练损失保证，证明了在具有正定极限神经切向核的宽两层ReLU网络上，Muon能以高概率达到任意目标损失，命中时间界为 $O((1-\mu)^{-1}\varepsilon^{-1/2})$。 |
| [^61] | [JOVE: Joint Execution and Verification for Resource-Aware LLM Task Graphs](https://arxiv.org/abs/2610.03296) | JOVE提出了一种在线框架，通过联合决策LLM执行分配与中间输出的付费验证，在长期预算和延迟约束下平衡即时执行开销与未来学习收益，从而在LLM服务质量未知的情况下提升任务图执行的效率与正确性。 |
| [^62] | [EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation](https://arxiv.org/abs/2610.03273) | 该论文提出EVOL框架，利用知识追踪仿真器通过进化搜索为每个学习者合成专家示范，并将其蒸馏为免部署的前馈策略，从而同时解决了学习路径推荐强化学习中的超指数组合搜索空间与稀疏奖励两大难题。 |
| [^63] | [SPEAR: A Spectral-Disentangled MoE Neural Operator with Knowledge-Guided Expert Aggregation for Large-Scale PDE Pretraining](https://arxiv.org/abs/2610.03265) | SPEAR通过将特征谱解耦为低频与高频分量以实现共享与专门化建模，并结合基于数据集知识与路由偏好的知识引导专家聚合策略来消除专家冗余，有效解决了PDE基础模型中的知识干扰与专家冗余问题，提升了大规模PDE预训练的泛化能力。 |
| [^64] | [Consecutive Posterior Fusion for Diffusive Recovery of Unobservable Image Structures](https://arxiv.org/abs/2610.03261) | 提出CPF-DDNM推理时策略，通过融合连续的感知测量后验估计来改进扩散模型对不可观测图像结构的恢复，且无需重新训练或额外的去噪器评估。 |
| [^65] | [Mapping and Advancing the Scalability-Accuracy Frontier of Nonlinear Causal Discovery](https://arxiv.org/abs/2610.03258) | 本文系统比较了四类非线性因果发现方法在可扩展性与准确性上的互补瓶颈，并提出基于样条的得分评估方案SPADE，通过一次性编译并复用充分统计量，在保持准确性的同时大幅提升组合搜索的效率。 |
| [^66] | [Learning a Fact Is Not Learning How to Retrieve It](https://arxiv.org/abs/2610.03251) | 该研究通过两阶段训练实验发现，“掌握事实知识”与“掌握如何提取该事实”是两种可分离的能力——模型可以在尚未真正学会某些事实之前，就先通过特定的请求形式学会提取方式。 |
| [^67] | [WAMpy: Efficient Synthesis of Prolog Programs in Python](https://arxiv.org/abs/2610.03234) | WAMpy是一个Python框架，通过将Prolog子句编译为基于NumPy数组的WAM指令并结合Numba JIT加速，大幅提升了在Python中反复合成与评估小型Prolog候选程序的工作负载的端到端性能。 |
| [^68] | [D2K-Bench: Can LLM Agents Turn Expert Designs into Efficient GPU Kernels?](https://arxiv.org/abs/2610.03226) | D2K-Bench 是一个包含 26 个任务和 85 个工作负载的诊断性基准，通过分层专家设计指导（算法洞察、数据流设计与底层优化技巧）来系统评估 LLM 智能体生成高效 GPU 内核的能力，结果显示专家指导可将正确率从 93.1% 提升至 98.5% 并显著提高性能。 |
| [^69] | [Uncertainty as a Proxy for Semantic Correctness in Diffusion-Based Medical Image Synthesis](https://arxiv.org/abs/2610.03224) | 本研究提出以不确定性作为扩散模型医学图像合成语义正确性的代理指标，并利用多任务扩散框架AortaDiff的分割误差作为定量度量来验证该方法的有效性。 |
| [^70] | [Evolving Hybrid Quantum-Classical Architectures for Image Classification](https://arxiv.org/abs/2610.03220) | 该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。 |
| [^71] | [Toward SLM-based agentic task-tool intent matching](https://arxiv.org/abs/2610.03213) | 本文提出利用小型语言模型（SLM）作为任务-工具相关性分类器，对智能体的每一次工具调用进行逐次意图验证，以判断调用是否真正服务于任务意图，从而实现低延迟或本地化部署的智能体行为监督。 |
| [^72] | [Contextual Flow Matching: Adaptive Step Selection in Flow Models for Efficient Visual Generation](https://arxiv.org/abs/2610.03202) | 提出COFLOW推理时方法，根据提示特征自适应选择每步生成的采样步数，即插即用且无需重新训练模型，在图像和视频生成中实现超过2.5倍加速并保持感知与语义质量。 |
| [^73] | [KV$^2$: A Self-Refining KV Cache](https://arxiv.org/abs/2610.03198) | KV²提出了一种基于选择性重建的查询无关KV缓存压缩方法，先用轻量级代理评分器筛选出信息丰富的token，再仅对该子集进行精细重建评分以计算淘汰分数，在极低缓存预算下比次优基线提升超过40个百分点。 |
| [^74] | [Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case](https://arxiv.org/abs/2610.03190) | 该论文研究了LLM调查员“何时应结案”的证据充分性判断问题，发现未经训练的小模型和前沿模型都普遍夸大证据充分性而过早结案，并提出需对照来源捷径规则来评估结案能力的方法。 |
| [^75] | [Gains and Collapse in On-Policy Distillation:A Reinforcement Learning Perspective](https://arxiv.org/abs/2610.03185) | 该论文从强化学习视角揭示了在线策略蒸馏（OPD）既提升性能也可能坍塌的机制——教师模型的隐式奖励在可靠时促进正确响应采样，在偏好与质量错位时引发奖励破解并放大冗长重复生成，且OPD提升性能但不扩展学生模型的能力边界。 |
| [^76] | [LiBRA: Detection-Aware Image Watermark Removal via Bidirectional Latent Optimization](https://arxiv.org/abs/2610.03166) | 该论文提出LiBRA方法，通过双向隐空间优化调整图像以使水印不可检测，同时避免过度优化导致的图像质量下降，从而在移除水印与保持画质之间取得平衡。 |
| [^77] | [Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer](https://arxiv.org/abs/2610.03163) | 该论文针对少样本作者风格迁移任务提出三种方法——对比激活转向、转向向量预测网络和预测LoRA适配器的超网络，并发现超网络在风格模仿与输出质量之间取得了最佳权衡，且能泛化到未见过的作者。 |
| [^78] | [Multimodal reasoning for broadly neutralizing antibody discovery from label-free human B cell repertoires across virus families](https://arxiv.org/abs/2610.03160) | ImmuneAgent是一个整合多模态推理、持续元学习与湿实验反馈的闭环AI系统，能从无标记的人类天然B细胞库中高效发现广谱中和抗体，实现约55%的中和抗体发现率和约11%的bnAb产出率，显著优于现有计算方法。 |
| [^79] | [EvoRiskBench: An Evolving Benchmark for Runtime Security Risks in Workspace Agents](https://arxiv.org/abs/2610.03153) | 提出了EvoRiskBench——一个基于EP-Path-EF框架的可演化安全基准测试，通过自动化端到端工作流在隔离环境中构建、执行并独立验证工作区智能体的运行时安全风险案例。 |
| [^80] | [Keeping JEPA World Models Plannable When Little of the Frame Moves](https://arxiv.org/abs/2610.03137) | 通过SLIM推动基准诊断出JEPA世界模型在画面几乎静止的场景中编码器潜在表示对动作不敏感的失败根源，并提出仅用一个逆动力学辅助损失即可将语言目标规划成功率从0.003提升至0.35。 |
| [^81] | [Trading Strategy Optimization via Textual Gradient](https://arxiv.org/abs/2610.03128) | 提出了TradeGrad框架，通过利用积累的优化经验来估计文本梯度并结合多尺度修订策略，克服了传统文本梯度优化短视及忽视时间稳健性的问题，实现了更稳健的量化交易策略优化。 |
| [^82] | [The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs](https://arxiv.org/abs/2610.03124) | 该论文首次形式化了开放权重大语言模型中的触发-标记滥用检测机制，将其分为令牌级和权重级两类，并系统研究揭示了此类机制在对抗性攻击下的脆弱性。 |
| [^83] | [Foresight: planning future perception in streaming VLMs without retraining](https://arxiv.org/abs/2610.03123) | 提出无需任何重训练的FORESIGHT双流架构，利用流式VLM固有的近期未来预测能力，动态规划并配置未来的感知计算，使模型能够自适应地应对不断变化的场景动态。 |
| [^84] | [How to Find and Reuse Policies for Continuous Adaptation in Lifelong Reinforcement Learning](https://arxiv.org/abs/2610.03119) | 提出AMSC方法，利用基于Wasserstein任务嵌入的在线相似性估计，自适应地选择和组合多个先前策略作为先验，从而在终身强化学习中获得更高的平均性能、前向迁移能力且不遗忘旧知识。 |
| [^85] | [S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales](https://arxiv.org/abs/2610.03106) | 该论文提出S2S-JEPA，首次将计算机视觉中的联合嵌入预测架构（JEPA）范式引入次季节到季节（S2S）预报，通过在潜在空间中只预测缓慢变化且可预测的分量、舍弃不可预测的细尺度细节，来突破AI天气模型在两周以上“可预测性荒漠”中的性能瓶颈。 |
| [^86] | [Ask, Relax, or Act? Evaluating Actionable Indeterminacy in LLM Preference Reasoning](https://arxiv.org/abs/2610.03102) | 该论文形式化了“可操作不确定性”概念并构建基于求解器的基准测试，发现LLM难以判断何时无需干预——即使行动已被证明合理，模型仍倾向于不必要的澄清提问或干预。 |
| [^87] | [Beyond Single Videos: Benchmarking and Active Evidence Seeking for E-Commerce Cross-Video Reasoning](https://arxiv.org/abs/2610.03099) | 该论文提出了首个电商跨视频推理基准AdsCVR，并设计了智能体框架AdSeek，通过多轮探索中动态选择视听工具实现主动证据获取，同时引入离线轨迹修正机制以应对强化学习中的稀疏信用分配问题。 |
| [^88] | [Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression](https://arxiv.org/abs/2610.03098) | 提出潜空间密码子优化方法LSCO，通过将序列映射到预训练mRNA语言模型的潜空间，将离散的密码子优化问题转化为可梯度搜索的连续问题，并结合不确定性感知的表达预测器、最小自由能正则化、自然性先验和约束解码，以最大化蛋白质表达。 |
| [^89] | [Peer Influence across Heterogeneous AI Models](https://arxiv.org/abs/2610.03095) | 该研究测量了七个开源语言模型之间的说服效应，发现模型意见分歧时说服作用非常强烈，但模型规模和单独运行时的确定性均无法预测说服动态，小模型既能像大模型一样有效说服他人，也同样能抵抗影响。 |
| [^90] | [ULTRADISCOVERY: Abductive Exploration in an Interconnected, Epistemically Open Universe](https://arxiv.org/abs/2610.03092) | 该论文提出ULTRADISCOVERY交互式基准，通过2×2设计独立控制表征开放性与证据分布性，评估智能体在认识论开放且结构互联的世界中进行溯因科学探索的能力，发现现有十一个模型均无法通过引入新实体或重写变量来完成理论替换。 |
| [^91] | [Securing Computer-Use Agents Against Branch Steering Attacks](https://arxiv.org/abs/2610.03089) | 本文系统研究了针对计算机使用智能体的新型“分支引导攻击”——攻击者无需注入显式指令，仅通过构造不可信数据即可诱导智能体走向预先批准的危险执行分支，并提出了STEER-Bench基准来评估这一威胁。 |
| [^92] | [Zephon: Elastic Determinism for Online, Stateful Foundation Model Data Loading Pipelines](https://arxiv.org/abs/2610.03087) | Zephon提出了一种面向基础模型训练的数据加载器，能够在GPU拓扑变化、频繁检查点恢复及不同执行后端的情况下，为包含在线分词、打包、混合等有状态n对m转换的数据流水线提供确定性的全局训练数据批次序列（即弹性确定性）。 |
| [^93] | [NegT2IBench: When Negation Changes the Picture. A Polarity Benchmark for Text-to-Image Models](https://arxiv.org/abs/2610.03084) | 提出了NegT2IBench基准，通过4,800条按极性组织的提示词系统评估文本到图像模型满足否定约束的能力，其基于检测器的评分以更小的规模达到了与大型视觉语言评判器相当的人类一致性水平。 |
| [^94] | [RIFAR: Reliability and Forgetting-Aware Replay for Continual Robot Learning](https://arxiv.org/abs/2610.03079) | 提出了 RIFAR 方法，通过可靠性筛选与漂移感知的回放选择，利用冻结的逆动力学模型评估世界-动作模型重建轨迹的动作-视觉一致性，从而在机器人持续学习中选择高质量回放经验，避免灾难性遗忘。 |
| [^95] | [MOF-VERIFY: A Failure-Aware Agentic Harness for MOF Hypothesis Verification](https://arxiv.org/abs/2610.03056) | 该论文提出了MOF-VERIFY，一个失效感知的智能体框架与四任务族诊断基准，通过闭卷、检索和先知证据等设置系统评估并定位大语言模型在MOF假设验证中的失效点（涵盖结构接地、合成条件、证据充分性和MLIP计算验证）。 |
| [^96] | [hacktrace: behavior-supervised detection of reward hacking during code generation](https://arxiv.org/abs/2610.03055) | HACKTRACE 通过复用编程智能体生成代码时已计算出的内部状态来监督捷径行为，无需额外模型推理即可在回合结束前检测奖励破解，AUC 达 0.997 且监控开销仅 8 毫秒。 |
| [^97] | [WebFovea: When the Model Is Right but the Click Is Wrong -- Reliable Round Trips for Vision-Based Web Agents on Live Websites](https://arxiv.org/abs/2610.03036) | 本文提出在WebRetriever Challenge 2026中获得亚军的视觉网页智能体WebFovea，并指出真实网站上的许多失败并非源于模型推理，而是源于模型与浏览器之间中间执行层在动作解析、页面生效、结果反馈和信息展示这四个环节上的问题。 |
| [^98] | [When Numbers Start Talking: Numerical Signalling and Strategic Behaviour Among LLMs](https://arxiv.org/abs/2610.03033) | 本研究通过四个主流LLM驱动的智能体在四种策略博弈中的实验，首次揭示不同类型的消息（尤其是数值信号）会以不可预测的方式显著改变博弈中的合作水平与收益，且智能体生成的数值信号系统性偏离随机性，从而挑战了AI智能体总能收敛到稳定均衡的假设。 |
| [^99] | [SoftGene: Protein Language Model-Enhanced Soft Prompting for Interpretable Gene Set Annotation](https://arxiv.org/abs/2610.03029) | 提出SoftGene框架，利用蛋白质语言模型ESM将基因集的蛋白质序列信息编码为分层软提示，与大语言模型的硬提示结合，实现更符合生物学结构、可解释的基因集功能注释。 |
| [^100] | [Tailoring the Quantization Space for 1-Bit KV Cache Compression](https://arxiv.org/abs/2610.03027) | 提出TaSQ方法，通过查询引导的通道加权、跨头归一化和协方差感知的通道分组来量身定制向量量化目标空间，从而在1比特极端压缩下实现有效的KV缓存压缩。 |
| [^101] | [Verifiable, Articulable, and Tacit Components of Preference](https://arxiv.org/abs/2610.03025) | 该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。 |
| [^102] | [DyadMem: A Long-Term Memory Benchmark of How Agents Work with Users](https://arxiv.org/abs/2610.03020) | 提出 DyadMem 基准，首次定义并显式标注“用户条件化的关系型智能体记忆（URAM）”，在多会话轨迹上同时评估用户事实记忆与关系型协作记忆，共含 6 类记忆、5 万多个会话和 6 万余条问答实例，使长期智能体记忆评估更完整可靠。 |
| [^103] | [Personalized Automatic Speech Recognition for a Dysarthric and Tracheostomic Speaker using Artificial Conversations](https://arxiv.org/abs/2610.03017) | 本文针对一位气管造口且严重构音障碍的捷克说话者，通过多阶段微调Whisper模型构建个性化语音识别系统，实现字符错误率相对降低50%，并发布了基于“人工对话”协议收集的33小时公开数据集。 |
| [^104] | [OmniAct3D: Leveraging Foundation Geometry and Evidence-Grounded Reasoning for Panoramic 3D Detection](https://arxiv.org/abs/2610.03015) | OmniAct3D通过ERP射线几何适配器（ERGA-Ray）建模球面视线与周期性空间结构，并借助视觉-动作推理链（VARC）在全景证据中锚定检测假设，从而将基于透视图像训练的视觉基础模型检测器适配到等距柱状全景投影，实现保留VFM先验的连贯360度3D检测。 |
| [^105] | [Beyond Predefined Sinks: Security-Aware Dependency Analysis for LLM Agents](https://arxiv.org/abs/2610.03014) | 该论文提出AgentSecGraph安全感知静态分析框架，通过构建安全感知智能体依赖图（Security-ADG），在传统预定义敏感操作标识之外融合智能体相关性、信任边界、防护证据等多维语义信息，并发布包含67个真实LLM智能体仓库的基准数据集AgentSecBench。 |
| [^106] | [AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning](https://arxiv.org/abs/2610.03007) | AvoKV-E是一种无需训练的KV缓存淘汰策略，通过延迟近期状态的淘汰资格，并结合读取压力、键冗余度和值负载潜力对缓存条目排序，在长推理任务中以相同的活跃KV预算达到或超越现有基线方法。 |
| [^107] | [Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability](https://arxiv.org/abs/2610.03000) | 该论文提出利用双曲几何的庞加莱模型构建多层感知机训练过程的时间参数图（即多个训练步骤的快照），以捕捉网络加权拓扑与自组织在训练轨迹中的几何演化，从而超越传统单检查点方法实现内在可解释性。 |
| [^108] | [Sentry: Learning to Recover from LLM Agent Failures at Test Time](https://arxiv.org/abs/2610.02994) | 提出Sentry——一个与LLM智能体并行运行的失败管理层，将失败经验视为条件性知识，在检测到失败时按需从外部经验手册中检索指导恢复、无奖励验证恢复结果并仅在确认恢复后存储新经验，从而在测试时实现从失败中学习。 |
| [^109] | [PLCWorld: Benchmarking LLM-Generated PLC Programs in Closed-Loop Plant Simulation](https://arxiv.org/abs/2610.02982) | 该论文提出PLCWorld，一个将LLM生成的PLC程序执行与闭环工厂仿真及传感器反馈相耦合的基准测试环境，包含100个任务和473个任务-条件对，可分别评估任务成功率与安全违规情况。 |
| [^110] | [Safeguarding Mutual Correction in Source-Free Domain Adaptation via Cut Statistics](https://arxiv.org/abs/2610.02981) | 该论文发现源预训练模型与视觉-语言模型具有互补的失败模式，并利用Cut统计量在无标签条件下识别哪个模型的预测更可靠，从而在无源域适应中实现两个模型之间的相互纠错。 |
| [^111] | [RASPER: Reward-Aligned Summarization of Clinical Notes for EHR Outcome Prediction](https://arxiv.org/abs/2610.02979) | RASPER提出了一种奖励对齐的摘要框架，利用下游预测器的反馈作为强化学习奖励，训练LLM摘要器从出院记录中提取对临床结局预测真正有用的证据，而非生成通用的流畅摘要。 |
| [^112] | [Relevant Evidence Decoding for Audio-Visual Hallucination Mitigation](https://arxiv.org/abs/2610.02976) | 提出了一种无需训练的相关证据解码方法RED，通过识别与问题相关的音视频证据并选择性地增强其贡献，来缓解视听大语言模型中的跨模态幻觉问题。 |
| [^113] | [Reliable Self-Evolution with Imperfect Proxy Rewards](https://arxiv.org/abs/2610.02975) | 该论文提出保形区间驱动的自进化方法（CISE），通过条件保形推断和在线密度比估计构建候选特定的奖励区间，以应对不完美代理奖励导致的假阳性问题，从而实现更可靠的LLM自进化搜索。 |
| [^114] | [CreateScore: Domain-Theory-Informed Bayesian Routing for LLM-Based CV Screening](https://arxiv.org/abs/2610.02972) | CreateScore 提出了一种由领域理论指导的贝叶斯网络路由方法，利用后验不确定性将低风险的简历筛选决策交由本地 8B 模型处理、将不确定的决策升级至 120B 大模型，在 77.7% 的决策可本地解决的前提下显著降低 LLM 简历筛选的成本。 |
| [^115] | [Reasoning with Evidence, Not Merely Rationales: Verifiable Preference Proofs for LLM-Based Recommendation](https://arxiv.org/abs/2610.02968) | 提出PROVE-REC框架，通过两阶段流程生成与所选证据相链接的可验证偏好证明，并借助屏蔽对比实验同时验证证据对接地声明的支持程度以及证明对最终推荐的影响，从而弥合LLM推荐中推理说明与实际所用信息之间的“接地-影响鸿沟”。 |
| [^116] | [Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards](https://arxiv.org/abs/2610.02967) | 该论文提出了一种将基于大规模人类偏好数据训练的偏好奖励与用于评估提示词忠实度的规则奖励相结合的后训练方案，并设计了优于简单加权平均的奖励组合策略，从而全面改进前沿文生图模型的生成质量。 |
| [^117] | [Dynamic Expert Pruning for Multi-Agent Systems](https://arxiv.org/abs/2610.02951) | 提出动态专家剪枝（DEP）方法，利用智能体的系统与任务提示动态识别并按需剪枝专家，解决了静态专家剪枝在多智能体异构工作负载下失效的问题。 |
| [^118] | [Continual Graph Memory for Mathematical Research Agents](https://arxiv.org/abs/2610.02945) | 提出 Ansatz——一个基于“持续图记忆”的数学研究智能体，通过可演化、跨问题的图结构记忆系统显式组织整个证明搜索过程，并复用先前问题的探索知识，从而有效管理海量中间证明结果。 |
| [^119] | [When to Compile a Computer-Use Agent? Measuring Payback and Making Compilation Decisions for Token Efficiency](https://arxiv.org/abs/2610.02932) | 本文提出PACE系统，通过记录成功与失败的编译成本、比较智能体与程序的执行开销，并基于回报预测用在线算法决定何时将反复执行的GUI操作编译为程序，从而提升令牌使用效率。 |
| [^120] | [Discriminating Fixture Coverage in Agent-Infrastructure Verification Suites](https://arxiv.org/abs/2610.02928) | 用变异分析衡量“在正确实现上通过、在缺陷实现上失败”这一标准证据的价值，发现该证据无法保证测试夹具覆盖——即使修复后的套件仍有五个对抗性变异体存活，其中三个因没有任何夹具能激活它们而从未暴露。 |
| [^121] | [Positive-Unlabeled Learning for Agent Safety False Alarm Auditing](https://arxiv.org/abs/2610.02925) | 该论文将智能体安全监控的误报审计建模为正例-未标注（PU）排序问题，提出一种两阶段的 Trust-aware PU 框架来克服监控器引发的选择偏差，从而更准确地从警报中识别出安全误报并降低人工审查成本。 |
| [^122] | [HASTE: Evolving Agent Harnesses Against Emerging Attacks Using Sparse Evidence](https://arxiv.org/abs/2610.02920) | HASTE 提出了一种多智能体框架，通过安全规范生成与攻击用例生成的对抗性交互，从稀疏的威胁证据中自动演化智能体线束，使其能够防御最初观察到的证据之外的新兴攻击。 |
| [^123] | [Frequency Is Not Sensitivity Identifying Safety-Sensitive Experts in Sparse MoE LLM](https://arxiv.org/abs/2610.02910) | 该论文提出用路由器梯度敏感度（即序列损失对专家门控权重的敏感度）替代传统的激活频率来识别稀疏MoE大语言模型中的安全关键专家，实验表明该方法在五种架构上能更准确地预测抑制哪些专家会削弱模型的安全拒绝能力。 |
| [^124] | [LUMOS: Tracing Parametric Knowledge from Training Data to Behavioral Outputs in LLMs](https://arxiv.org/abs/2610.02902) | LUMOS诊断框架利用完全透明的OLMo 2训练语料库，沿“训练数据暴露→行为输出”的因果链追踪大语言模型的参数化知识，揭示模型内部能高可分性地编码罕见事实（84%）但在行为上表达不足（54%），且该检索差距随模型规模增大而缩小。 |
| [^125] | [Interpreting at Write Time: A Policy Ablation for Multi-Goal Agent Memory](https://arxiv.org/abs/2610.02897) | 该论文提出三种智能体多目标记忆摘要写入策略（无目标通用摘要、单一全目标摘要、按目标分别摘要后合并读取），并通过固定读取步骤的消融实验证明，为不同目标写入的摘要内容会显著分化，说明记忆写入时就应针对目标进行解读与取舍。 |
| [^126] | [When Can We Trust the Matching Principle? Robust Deployment Geometry Under Finite-Sample and Model Uncertainty](https://arxiv.org/abs/2610.02894) | 该论文提出用信任比率 tau = ε/γ 来量化匹配原则何时可靠（估计投影匹配的漂移在 Davis-Kahan 分离区域内按 O(τ²) 增长），并据此设计置信度校准匹配（CCM）策略：τ 小时进行方向性匹配，τ 大时渐进各向同性扩散惩罚，从而在有限样本与模型不确定性下实现鲁棒部署。 |
| [^127] | [Revealing Epistemic Uncertainty in MLLMs via Causal-Invariant Masking](https://arxiv.org/abs/2610.02887) | 提出因果不变掩码方法与语义散度指标，将源于模型局限性的认知不确定性与数据模糊导致的偶然不确定性解耦，从而有效检测多模态大语言模型因依赖表面关联而产生的幻觉风险。 |
| [^128] | [Misinformation Without Triggers: From Factual Answers to Downstream Decisions](https://arxiv.org/abs/2610.02886) | 该研究揭示虚假训练文档无需任何触发器即可改变语言模型的事实性回答，但直接回答的受污染程度无法预测下游决策行为，二者之间存在“审计差距”，因此仅审计直接答案会严重低估错误信息的真实危害。 |
| [^129] | [PsyEvo: A Personalized Counseling Agent That Self-Evolves at Test Time](https://arxiv.org/abs/2610.02885) | PsyEvo是一个基于大语言模型的心理咨询框架，通过分层贝叶斯技能策略和会话间列表式偏好优化等组件，在测试时实现针对个体来访者的个性化定制和响应策略的自我演化改进。 |
| [^130] | [DyRA: Dynamic Residual Approximation for Efficient Matrix Multiplication in DNNs](https://arxiv.org/abs/2610.02882) | 提出了输入自适应方法DyRA，通过在推理过程中动态近似并校正结构化权重近似引入的输出残差误差，直接优化输出的低秩因子，从而更高效、更精确地实现深度神经网络中的矩阵乘法近似。 |
| [^131] | [Query-aware routing for Cross-lingual performance gains in Encoders](https://arxiv.org/abs/2610.02875) | 该论文提出将仅作用于查询端的LoRA适配器与基于查询和索引语言的确定性路由相结合，在保留同语言性能和现有文档索引的同时，使英语、芬兰语、瑞典语六条跨语言检索方向的平均nDCG@10从0.241提升至0.291，相对提升20.9%。 |
| [^132] | [ConvoDrift: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution](https://arxiv.org/abs/2610.02873) | ConvoDrift 是一个用于建模固定语义意图下多轮对话风格语调渐进漂移的数据集，包含 15,727 个多轮对话结构、风格漂移标注及基于五种人设条件的偏好成对数据集，可支持风格适应与个性化对齐的受控研究。 |
| [^133] | [AgentTrap: Stateful Feedback Deception against Autonomous Penetration Testing Agents](https://arxiv.org/abs/2610.02869) | 提出了首个专为自主渗透测试代理设计的闭环蜜罐 AgentTrap，通过状态化欺骗与行为引导升级来诱捕、拖延代理并收集其行为证据。 |
| [^134] | [Distributionally Robust Survival Models under Subpopulation Shift and Outlier Contamination](https://arxiv.org/abs/2610.02868) | 本文提出一个分布鲁棒生存分析框架，通过外层最小化削弱离群样本的影响、内层最大化聚焦最不利的子群体，从而联合应对子群体偏移与离群点污染，并直接兼容Cox风险集等不可分解的生存损失。 |
| [^135] | [TACD: Distilling Efficient Text-to-Motion Models via Terminal Amplification Control](https://arxiv.org/abs/2610.02867) | TACD提出一种在策略蒸馏方法，通过将教师模型查询与学生步长绑定来限制去噪终点附近的误差权重，在无需真实运动数据的情况下训练出高效的少步文本到运动生成模型。 |
| [^136] | [Harness-Aware Distillation for Small Language Model Agents](https://arxiv.org/abs/2610.02858) | 提出框架感知蒸馏（HAD），将蒸馏聚焦于教师模型在框架之外附加的能力，通过动作偏好对比与有效性检查，使小型学生智能体学会根据框架信息正确行动。 |
| [^137] | [Bounded Reachability & Jailbreak Detection via Contraction-Constrained State Space Models](https://arxiv.org/abs/2610.02853) | 本文证明基于SSM的安全头能否获得可认证的越狱检测鲁棒性完全取决于一个收缩条件——状态转移矩阵的 $l_\infty$ 范数小于1：条件成立时可达输出区间稳态宽度有界、可通过精确区间界限传播认证鲁棒分类，条件不满足时区间随序列长度指数增长导致认证不可能。 |
| [^138] | [DNAlign: Dynamic Null-Space Safe Alignment for LLMs](https://arxiv.org/abs/2610.02844) | DNAlign将大语言模型视为动态系统，结合控制论优化与零空间投影，把安全扰动限制在与危害相关的子空间内，从而在不损害模型通用知识和响应质量的前提下实现轻量级安全对齐。 |
| [^139] | [FastOPD: On-Policy Distillation for Lightweight VLA Deployment](https://arxiv.org/abs/2610.02832) | FastOPD提出了一种高效的在策略蒸馏框架，将流图单状态教师监督与自洽性目标相结合，把大规模VLA基础模型压缩为可实际部署的轻量化模型，并从理论上保证学生模型能恢复出与理想少步教师模型相当的分布。 |
| [^140] | [AMBER: Multi-View Adaptive Budget Allocation for Listwise Vision-Language Reranking](https://arxiv.org/abs/2610.02831) | AMBER提出了一种在线预算化多视图重排框架，将碎片化的VLM输出通过Elo更新整合为全局排序状态，并在视图构建和查询调度两个层级上动态分配昂贵的VLM计算资源，以最大化期望信息增益并提升多模态检索重排效率。 |
| [^141] | [FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training](https://arxiv.org/abs/2610.02828) | FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。 |
| [^142] | [MLCommons Jailbreak Benchmark v1.0](https://arxiv.org/abs/2610.02827) | MLCommons发布了越狱基准测试v1.0，提供了一套端到端的评估方法论，使用264个种子提示词、十一个危害类别和越狱分类法中的代表性攻击，对八个开放权重大语言模型进行单轮文本越狱攻击的鲁棒性评估，并创新性地提出以“韧性差距”作为核心安全衡量指标。 |
| [^143] | [Scaling Trajectories for Complex Tasks through Recursive Self-Rewrite](https://arxiv.org/abs/2610.02826) | 提出递归自我重写（RSR）框架，利用单一基础模型在不同专用测试框架下发现成功解法，并将其重构为通用框架下的训练轨迹，成功将2001条轨迹扩展为11094条用于监督微调，显著提升模型解决复杂终端任务的能力。 |
| [^144] | [MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning](https://arxiv.org/abs/2610.02824) | MetaRubric通过构建反事实提示、要求响应提供充分证据才能得分，并将证据感知的策略优化与响应引导的量规修订交替进行，从而解决了基于量规的强化学习中评判器给出“空洞信用”的问题。 |
| [^145] | [Adaptive Spectral-Koopman Dynamics Modeling for Temporal Domain Generalization](https://arxiv.org/abs/2610.02822) | 提出AdaSpecK框架，通过谱正则化Koopman动力学建模提取去噪的低频轨迹，并结合上下文感知的异构模式提取机制，有效解决时序域泛化中的噪声过拟合与非平稳历史环境建模问题。 |
| [^146] | [iS-KV: Online Low-Rank KV Cache Compression via Block-Incremental SVD](https://arxiv.org/abs/2610.02815) | 提出iS-KV，一种基于块增量SVD的在线低秩KV缓存压缩方法，通过解决基更新导致的历史漂移问题，在保留全部历史状态的同时实现长思维链推理场景下的缓存高效压缩。 |
| [^147] | [ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing](https://arxiv.org/abs/2610.02808) | ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。 |
| [^148] | [VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning](https://arxiv.org/abs/2610.02801) | VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。 |
| [^149] | [BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration](https://arxiv.org/abs/2610.02800) | BitNest提出了一种比特嵌套的投机解码框架，将低精度草稿模型直接嵌入高精度目标模型的权重表示中，通过残差细化使两者共享单一物理权重，从而在加速大语言模型推理的同时显著降低内存开销。 |
| [^150] | [Modeling Shared and Individual Structure for Cross-Subject Continuous Affect Regression from EEG-fNIRS](https://arxiv.org/abs/2610.02796) | 该论文提出将情感轨迹分解为观看相同刺激的被试间共享的结构，以及基于无标注EEG标记（α波段跨通道同步性）估计的个体校准结构，从而在EEG-fNIRS数据上实现了零样本跨被试的连续效价-唤醒度回归。 |
| [^151] | [PAPER2LLM++: Continual Self-Evolution of LLMs from Research Papers](https://arxiv.org/abs/2610.02793) | PAPER2LLM++ 提出了一个让大语言模型从研究论文中持续自我演化的框架，通过提取论文中的研究发现、验证局限性是否仍然存在，并借助“尝试-评估-提交”机制整合更新，从而在不遗忘先前改进、不损害通用能力的前提下实现模型的自动改进。 |
| [^152] | [Law And Order: Tax Law Autoformalization](https://arxiv.org/abs/2610.02792) | 提出 Law&Order 神经符号框架，通过结构对应与指称对应两种机制，利用大语言模型将税法表格和申报说明自动转化为可执行的符号程序，并通过单元格级验证和迭代式局部错误修复保证准确性。 |
| [^153] | [OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation](https://arxiv.org/abs/2610.02781) | 提出两阶段训练框架：先以评分标准作为教师特权上下文进行在线策略蒸馏（RP-OPD）提供密集的token级监督，再以评分标准作为奖励进行强化学习，从而突破蒸馏的性能瓶颈。 |
| [^154] | [Improving Atomic-Fact Recall via Focused Views in Unstructured Knowledge Editing](https://arxiv.org/abs/2610.02772) | 该论文揭示了非结构化知识编辑中段落级编辑目标导致的“难度低估”问题，并提出通过聚焦视图的方式改进编辑后的模型，使其无需原始段落上下文即可可靠地回忆编辑文本中的各个原子事实。 |
| [^155] | [Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback](https://arxiv.org/abs/2610.02771) | 本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。 |
| [^156] | [When History Fails to Become Experience: Action Calibration in Language Agents](https://arxiv.org/abs/2610.02769) | 研究发现语言智能体并不能可靠地将历史动作与其结果相关联，而只需简单地为每条观察标注其对应的前序动作，即可显著提升任务成功率并减少动作重复。 |
| [^157] | [Dynamic LLM Routers are Often Misguided](https://arxiv.org/abs/2610.02762) | 研究发现六种商用动态LLM路由器在相同成本下的表现均不如在两个精选模型间随机选择的路由器，其根源在于标准优化目标本身就会奖励难度盲视、长度逆转和语义匹配等误导性行为。 |
| [^158] | [Correcting Guided Diffusion Trajectories with Spectral Alignment](https://arxiv.org/abs/2610.02753) | 该论文提出“谱校正引导”方法，通过将采样中间状态的谱与前向过程的解析参考谱对齐来自适应校正CFG引导轨迹的偏差，无需训练即可提升条件图像生成的对齐度与视觉保真度。 |
| [^159] | [On the Chain-of-Thought Monitorability of Looped Language Models](https://arxiv.org/abs/2610.02741) | 本文首次系统性评估了循环语言模型的思维链可监控性，发现与匹配规模的非循环模型相比，LoopLM在多项任务中表现出任务相关的可监控性下降。 |
| [^160] | [Prospective Hindsight: Self-Calibrating Reinforcement Learning via Prediction-Reality Gaps](https://arxiv.org/abs/2610.02740) | 提出前瞻性后见之明（PH）这一自校准强化学习训练原则，通过衡量智能体动作前预测与反馈后评估之间的“惊讶度”差距来加权梯度，使学习自动聚焦于智能体自我模型中最不准确的盲点样本。 |
| [^161] | [TPBench: A Turning-Point Benchmark for Dialogue Compression](https://arxiv.org/abs/2610.02736) | 该论文提出 TPBench 基准，通过在相同保留预算下探测用户的初始目标、修改后槽位的当前值等互补信息目标，揭示了对话压缩中被整体保留分数掩盖的“转折点丢失”失败模式。 |
| [^162] | [Revisiting Visual Representation Enhancement of VLMs via Kernel Canonical Correlation Analysis](https://arxiv.org/abs/2610.02718) | 本文提出利用核典型相关分析（KCCA）在特征子空间上刻画视觉语言模型与DINOv2之间的表征对齐，从而增强CLIP等模型的细粒度视觉感知能力。 |
| [^163] | [Ego2World: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning](https://arxiv.org/abs/2610.02715) | 该论文提出Ego2World基准，将标注的第一人称烹饪视频编译为具有持久世界状态与智能体信念、部分可观测的可执行规划环境，用于系统评估信念状态规划器，并揭示被接受的操作往往仍无法达成任务目标。 |
| [^164] | [Self-Supervised Scaling of Terminal Environments for Scientific Domains](https://arxiv.org/abs/2610.02710) | 提出软件在环重构这一自监督框架，通过从现有软件工作流中自动提取参考输出与验证目标，实现面向科学领域的终端智能体训练环境的可扩展、可复用构建。 |
| [^165] | [MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads](https://arxiv.org/abs/2610.02705) | MuonIO 将 Muon 优化器的原则性更新扩展到嵌入表和语言模型输出头——对语言模型头采用 2→∞ 算子范数、对嵌入表采用 1→2 算子范数，从而以统一的范数感知更新取代 AdamW。 |
| [^166] | [Label-Efficient Time Series Classification at Scale: A Dual-Stream OSSE-LSTM with Counterfactual Attribution](https://arxiv.org/abs/2610.02704) | 该论文提出双流OSSE-LSTM框架，将带挤压与激励重校准的全尺度CNN与双向LSTM结合，在每类仅有K个标注样本的极端标签稀缺条件下实现大规模时间序列分类，并利用反事实归因提升模型可解释性。 |
| [^167] | [Learning to Revise Reasoning with Segment-wise On-Policy Distillation](https://arxiv.org/abs/2610.02703) | 该论文提出分段式在线策略蒸馏方法，将教师模型对中间推理步骤的重写内容作为显式监督信号来训练学生模型修正自身推理，从而提升后续推理准确率并避免强化不良推理模式。 |
| [^168] | [Test-time Calibration Learning for Large Language Model Reasoning](https://arxiv.org/abs/2610.02695) | 提出了一种无需标签的测试时校准学习框架TTCL，能够直接在未标注的目标任务数据上联合优化大语言模型的推理准确性和置信度表达能力，摆脱了对真实标签的依赖。 |
| [^169] | [Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning](https://arxiv.org/abs/2610.02687) | 该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。 |
| [^170] | [Large language models exhibit unreliable updating of clinical judgment as patient evidence evolves](https://arxiv.org/abs/2610.02684) | 该研究发现大语言模型在患者证据演变时无法可靠地更新临床判断，具体表现为对病情恶化证据反应过强的不对称性以及先验信念对预测的因果性干扰，且提示工程无法修复这些问题。 |
| [^171] | [DataWeave: Deploying Human-LLM Analytics for Exploratory Structured Data Analysis](https://arxiv.org/abs/2610.02679) | DataWeave是一个面向数据新闻工作场景的人机协同分析系统，通过结合对话式交互、模式接地、分析规划和可执行查询生成，解决LLM在探索大型结构化数据集时出现的模式不匹配、语义误读等可靠性问题。 |
| [^172] | [Spend Teacher Tokens Where They Matter: Success-Referenced On-Policy Distillation](https://arxiv.org/abs/2610.02678) | SR-OPD 以同一提示下的成功 rollout 为参照，聚焦于隐藏状态轨迹持续发散的失败 rollout 进行选择性教师监督，仅用 Vanilla OPD 约 3.46%–5.02% 的教师输入 token 即可达到相当的性能。 |
| [^173] | [LEAP: Learning Efficient Action Proposals For LLM Agents](https://arxiv.org/abs/2610.02670) | 该论文提出LEAP方法，通过学习一个高效的动作提议模型（而非使用现成的通用模型）为LLM智能体起草动作，并建立延迟分析框架揭示决定动作投机端到端加速的关键因素，从而显著提升智能体执行任务的速度。 |
| [^174] | [Large Language Continuous Diffusion Models](https://arxiv.org/abs/2610.02665) | 提出了首个大规模（3B/8B）连续扩散语言模型 Sigma，通过可操控的低维潜在轨迹、自回归模型热启动以及无分类器引导等推理技术，在数学推理和编码任务上取得了与离散扩散模型相当的性能。 |
| [^175] | [A GHOST in Long-Horizon Agents: Governance Hazard from Overlooked Safety Constraints across Turns](https://arxiv.org/abs/2610.02664) | 该论文发现长程智能体在良性交互条件下可能违反多轮之前设定的安全约束（即GHOST现象），在GPT-5.5上发生率达11.5%，并从理论上证明当剩余违反风险满足不可求和条件时，执行几乎必然进入危险区域。 |
| [^176] | [Generalization Properties of Score-matching Diffusion Models for Intrinsically Low-dimensional Data](https://arxiv.org/abs/2610.02663) | 该论文为流匹配模型在具有内在低维结构的数据上提供了统计泛化理论保证，推导出依赖于数据内在维度的 Wasserstein-p 有限样本误差界，克服了以往分析中限制性假设和忽略低维结构的不足。 |
| [^177] | [Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis](https://arxiv.org/abs/2610.02659) | 该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。 |
| [^178] | [Coherence-Driven Belief Formation and Population Dynamics of Contagion in LLM Agents](https://arxiv.org/abs/2610.02654) | 本文实证测量了LLM智能体的信念采纳行为，发现其呈S形的复杂传染特征，且采纳阈值可由“传入信念与智能体先验信念的一致性”这一单一维度解释，并在群体层面观察到复杂传染的网络效应（聚类网络中传播更广）及自我维持的滞后共识现象。 |
| [^179] | [Equivariant Flow Matching for Electron Density Prediction](https://arxiv.org/abs/2610.02651) | 该论文提出OrbFlow，一种SE(3)等变流匹配生成模型，通过预测高斯型轨道系数来生成电子密度，在保持紧凑基组效率的同时捕捉系数空间的结构相关性，为DFT自洽场计算提供高效且可迁移的初始化方案。 |
| [^180] | [Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen LLM Hear Beyond the Transcript](https://arxiv.org/abs/2610.02638) | DuplexJev将ASR编码器隐藏状态通过小型连接器输入冻结LLM，以单token分布直接读取决策答案，无需自回归解码即可在约0.1秒内批量完成80个语音决策，并通过交叉注意力连接器额外感知说话者的性别与情绪。 |
| [^181] | [Designing the Future of User Feedback for Generative AI](https://arxiv.org/abs/2610.02631) | 本研究通过与eBay的产学研合作，评估了当前生成式AI产品用户反馈机制的常见缺陷，提出了设计最佳实践建议，并设计测试了一个高效、灵活且注重用户价值的反馈收集工具原型。 |
| [^182] | [Lost in the Request: How Communication Variation Disrupts Retrieval and Action in Email Agents](https://arxiv.org/abs/2610.02627) | 该论文揭示了邮件智能体存在“沟通鲁棒性”缺陷：即使任务实质完全不变，仅请求的表达方式（如间接、冗长或方言变体）发生改变，就会显著降低RAG系统和工具使用智能体的检索与执行性能。 |
| [^183] | [CuBEs: Culturally-Situated Behavioral Evaluations and the Limitations of Culture-Blind LLM Judges](https://arxiv.org/abs/2610.02622) | 提出了文化情境化行为评估框架CuBEs，通过构建涵盖12种文化的人工标注数据集，将文化背景注入行为测试流程，揭示了“一刀切”式LLM评判者无法捕捉的显著跨文化行为差异。 |
| [^184] | [WebUIProof: Benchmarking WebUI Code Generators with UI-Agent Execution Harness](https://arxiv.org/abs/2610.02617) | 提出WebUIProof基准，通过UI代理在无头浏览器中执行可执行的交互测试来评估WebUI代码生成的功能正确性，揭示了八个商用大模型在交互类需求上的频繁失败。 |
| [^185] | [VERSE: Verified Self-Evolving Optimizer for Agent Harnesses](https://arxiv.org/abs/2610.02616) | 提出VERSE，一个经过验证的自我进化优化器，它不仅改进智能体框架，还让优化器自我进化其诊断、编辑与验证流程（如测试草稿编辑、重放故障、扰动可疑步骤），在具备基于执行的验证时取得最佳优化效果。 |
| [^186] | [Time Series Forecasting Benchmarks Need Scenario-Grounded Stress Testing](https://arxiv.org/abs/2610.02608) | 该论文指出当前时间序列预测的评估基准过于狭窄，无法捕捉真实部署系统中因结构化事件导致的语义、因果和系统级失效模式，因此倡导引入基于真实场景的压力测试来更可靠地评估预测模型，尤其是基础模型。 |
| [^187] | [TasteBench: Multimodal Benchmark for Sensory Prediction, from Molecules to Sustainable Foods](https://arxiv.org/abs/2610.02599) | TasteBench是首个面向可持续食品感官预测的多模态基准，通过覆盖2.1万余次人类评估的食品级排序任务和1.5万风味分子的味觉分类任务，并首次刻画了人类感官数据本身的信度上限，为加速植物基食品设计提供了计算评估工具。 |
| [^188] | [How Causality Bridges the Semantic Gap](https://arxiv.org/abs/2610.02594) | 该论文提出以因果结构替代人类知识来为未命名变量赋予语义，将其形式化为“结构约束的语义对齐”，并构建 CausalBridge 框架，从测量数据（含隐变量）中发现因果图并在其依赖关系约束下求解变量嵌入，从而从变量对其他变量的作用方式中解读其含义。 |
| [^189] | [Open-Endedness Bench: Measuring Epistemic Process from Agent Records](https://arxiv.org/abs/2610.02588) | 提出 OEB，一种只依赖智能体执行记录（不使用参考答案或结果分数）来评估其认知过程——假设形成、检验与修正——的与基准无关的方法论。 |
| [^190] | [Labels Override Definitions in Jev-Style Typed Decision Models](https://arxiv.org/abs/2610.02586) | 该研究发现Jev式类型化决策模型在输出概率时主要依据选项的标签而非其书面定义，即使规则只写在定义中也是如此，并据此提出了“选项-标签偏差”这一概念，同时引入了用于验证该现象的PolicyBench合成路由测试套件。 |
| [^191] | [Answering clinicians' questions over trial evidence tables with verifiable, feedback-driven language models](https://arxiv.org/abs/2610.02576) | FD-SCoPE是一个可验证、可从专家反馈中学习的语言模型框架，既能回答临床医生对试验证据表的直接查询，也能回答需要推导属性的问题，并在肿瘤学证据表上以77.7%的推导值F1超越四种替代方法。 |
| [^192] | [Improving the Energy-Efficiency of the Code Generated by LLMs through Effective Prompting](https://arxiv.org/abs/2610.02571) | 本研究系统评估了21种提示策略，发现有效的提示工程可使大语言模型生成的Python和C++代码能耗分别降低最多25%和17%。 |
| [^193] | [Pincer: Resource Authorization for Agents using a Digital Twin](https://arxiv.org/abs/2610.02569) | Pincer 提出了一种基于数字孪生、在资源层运行的授权防御机制，为长周期自主编码智能体提供可持续的权限管理，克服了用户中介沙箱的策略衰减与权限疲劳问题，并与现有工具调用层防御形成互补。 |
| [^194] | [Mitigating Social Sycophancy via Pluralistic Preference Optimization](https://arxiv.org/abs/2610.02568) | 该论文提出多元化偏好优化方法，通过让语言模型在给出个人建议时考虑受影响的其他利益相关者的视角，而非过度迎合用户，从而缓解语言模型在社交场景中的谄媚问题。 |
| [^195] | [DAGS: Disentangled Appearance-and-Geometry Steering of a Frozen Image DiT for Temporally Stabilized Generative Rendering](https://arxiv.org/abs/2610.02567) | DAGS提出了一种轻量级、无需注意力机制的外观与几何解耦条件注入方案，将条件特征作为逐层残差注入冻结的图像DiT，并辅以循环光照稳定器和免训练时序引导，实现了高保真、高忠实度且时间稳定的生成式渲染。 |
| [^196] | [OpenGameEval: Benchmarking Agentic Programming and Exploration in a Stateful Game Engine](https://arxiv.org/abs/2610.02563) | OpenGameEval是一个在Roblox Studio有状态游戏引擎中评估智能体游戏开发能力的基准框架，其核心创新在于通过分离观察工具与编辑工具来直接测量探索行为，实验发现前沿模型虽通过率相近但解决的任务各不相同，且最佳模型单次尝试仅能解决51.7%的任务。 |
| [^197] | [How to Have a Sensitive Debate: An Instance-Optimal Protocol for AI Debate](https://arxiv.org/abs/2610.02557) | 本文针对AI辩论设计了一种新的实例最优协议，对于具有足够稳定子问题分解的问题，在有限监督下比现有最佳协议提供更强的正确性保证。 |
| [^198] | [Out of Sync, Out of Sight: Phantom State Attacks against IIoT Intrusion Detection](https://arxiv.org/abs/2610.02552) | 本文提出幻影状态攻击（PSA），在被动、零查询的威胁模型下，通过利用IDS重建运行状态时对时间同步的依赖性，操纵入侵检测系统对工业物联网系统状态的观测视图。 |
| [^199] | [How To Train Your World Model: Fine-tuning vs RAG for LM-based World Modeling](https://arxiv.org/abs/2610.02542) | 该研究系统评估了基于语言模型的世界建模中微调与RAG两种范式的表现，发现微调方法通常优于RAG（在20个设置中的15个获得更高奖励），但RAG方法更具数据效率。 |
| [^200] | [CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization](https://arxiv.org/abs/2610.02527) | 该论文揭示，用学习型视觉奖励模型优化机器人策略时，奖励分数与任务成功率可能同时上升、看似健康，但实际上会显著放大“作用于错误物体”的隐蔽失败，而这一现象在使用模拟器真实任务完成信号训练时并不会出现。 |
| [^201] | [Learning What to Investigate Next: Meta-Reasoning for Long-Horizon Research Agents](https://arxiv.org/abs/2610.02525) | MIRA 提出了一种将研究资源分配与具体执行相分离的分层元推理架构，无需策略训练即可显著提升长程研究代理在定理证明和开放式神经架构研究中的推理能力与算力分配效率。 |
| [^202] | [Hypothesis-guided discovery of cognitive algorithms via program refinement](https://arxiv.org/abs/2610.02523) | 该论文提出了一种结合人类专家知识与大语言模型的混合系统，将认知算法发现表述为程序精化问题，让LLM智能体在研究者设定的约束下迭代修正以概率程序表达的认知模型，从而兼顾可解释性、人类专业知识与可扩展性。 |
| [^203] | [Instance-Dependent Regret for CMDPs with Step-Wise Constraints](https://arxiv.org/abs/2610.02520) | 本文提出了安全方差自适应探索算法（SVAE），通过学习候选安全子图并在其中进行方差自适应的乐观规划，在具有逐步安全约束的情景式CMDP中首次实现了依赖问题实例（方差感知）的累积遗憾界。 |
| [^204] | [Student-Guided Teacher Distillation for Efficient LLM Task Routing: Positioning Against Jev-Style System-1 Classifiers](https://arxiv.org/abs/2610.02516) | 提出一种学生引导的教师蒸馏流水线：紧凑的ModernBERT学生模型单次前向预测完整类别分布并生成top-k候选，更大的DeBERTa-v3零样本NLI教师模型仅对候选重排序，教师标签迭代反哺学生，从而显著降低大规模LLM任务路由的成本。 |
| [^205] | [IGNITE Tokamak World Model Architecture](https://arxiv.org/abs/2610.02515) | IGNITE是首个基于DIII-D十年实验数据自监督训练的聚变等离子体生成式世界基础模型，可通过执行器轨迹、文本提示或期望实验结果模拟完整的托卡马克放电过程。 |
| [^206] | [From Fragments to Global Maps: Learning Vectorized Map Aggregation with Large Language Models](https://arxiv.org/abs/2610.02513) | 该论文提出MapMergeLLM，将向量化高精地图聚合任务转化为大语言模型的条件序列生成问题，直接从序列化的局部地图片段预测全局地图折线，从而摆脱了传统方法对手工规则、固定阈值和特定检测器调优的依赖。 |
| [^207] | [On-Premises Multi-Course RAG Tutoring for Business Education: Hardware-Software Trade-offs in a Campus AI Tutor](https://arxiv.org/abs/2610.02510) | 本文提出了CourseChat——一个面向本科商业教育的本地部署多课程RAG辅导系统，通过模型对比测试与软硬件权衡评估，证明12B和7B级本地大语言模型能在课程数据不出校门的前提下满足课堂实时响应的速度要求。 |
| [^208] | [World Action Modeling with Progressive Visual Planning](https://arxiv.org/abs/2610.02508) | ProWAM通过联合预测动作和有序的稀疏视觉子目标序列实现渐进式视觉规划，解决了世界动作模型长时程预测效率低下的问题，且子目标预测可从大规模无动作视频中学习，使视觉规划与动作策略自然解耦。 |
| [^209] | [Multi-Fidelity Policy Gradients Stabilize Data-Scarce Reinforcement Learning](https://arxiv.org/abs/2610.02505) | 本文将多保真度策略梯度（MFPG）框架从REINFORCE扩展到现代演员-评论家算法（如PPO），在GPU并行仿真和真实机器人上利用低保真度数据构建控制变量，以在无偏差的前提下降低梯度方差，从而稳定数据稀缺场景下的强化学习。 |
| [^210] | [HXAI: Hierarchical Privacy-Preserving Explainable AI in Distributed Energy Systems](https://arxiv.org/abs/2610.02504) | 提出HXAI分层隐私保护框架，通过本地模型在私有环境中生成细粒度解释、区域模型聚合这些解释，在保护用户隐私的同时实现电网级需求管理的可解释分析。 |
| [^211] | [Compound AI System Reliability: A Failure Taxonomy and Resilience Pattern Catalog from 150 Production Incidents](https://arxiv.org/abs/2610.02503) | 本文通过分析150起生产事故，构建了包含23种故障模式、分五大类别的复合AI系统故障分类学，并提出经故障注入实验验证有效性的韧性模式（如断路器减少89%级联传播、质量门控捕获73%静默退化）。 |
| [^212] | ["I just assumed that it would translate": examining MT risk awareness among healthcare staff with abbreviations as a use case](https://arxiv.org/abs/2610.02496) | 本研究以医学缩写为用例，考察英国医护人员在使用机器翻译（尤其是翻译患者医疗记录）时对潜在风险的认知，揭示对机器翻译的盲目依赖可能危及患者安全。 |
| [^213] | [Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement](https://arxiv.org/abs/2610.02492) | 该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。 |
| [^214] | [What Does a Token Cost? A Mixture-of-Agents Measurement of Sufficient Per-Token Compute](https://arxiv.org/abs/2610.02491) | 该论文首次通过混合代理方法测量了单个Token实际所需的充分计算量，发现0.5B小模型即可复现92-95%的Token，而最昂贵的10%的Token占据了64-80%的估计计算量，揭示了计算分配中巨大的浪费与优化空间。 |
| [^215] | [From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders](https://arxiv.org/abs/2610.02486) | 该论文提出SBERT2S1框架，将生物医学检索句子编码器转换为类型化决策模型，并发现检索预训练显著有利于保留检索先验的先验融合残差（PFR）决策头，而对交叉头（C）帮助有限甚至有害。 |
| [^216] | [MEA: A Reward-Driven Multi-Agent System for Faithful Model Explanations](https://arxiv.org/abs/2610.02480) | MEA通过Proposer智能体自动选择和配置解释工具、Actor智能体以忠实性为目标进行端到端奖励优化，完全消除了使用机器学习解释方法的知识壁垒，能够跨表格、文本和视觉模态生成忠实的自然语言模型解释。 |
| [^217] | [Tropical Reinforcement Learning](https://arxiv.org/abs/2610.02478) | 该论文提出热带强化学习，用取最大值替代概率求和（将代数结构改为热带半环），使状态价值定义为最可能已验证解决方案的对数概率，从而避免强化一个解导致遗忘其他有效解的问题，更契合大语言模型的组合式推理。 |
| [^218] | [APDMem: Agent-Controlled Progressive Disclosure for Query-Adaptive Long-Term Memory](https://arxiv.org/abs/2610.02472) | APDMem提出了一种智能体控制的分层长期记忆架构，将对话历史组织为四个粒度递进的层次并采用渐进式披露检索，从而根据查询复杂度自适应地平衡检索成本与证据保真度。 |
| [^219] | [SideKernel: A Usable microVM Sandbox for AI Coding Agents on macOS](https://arxiv.org/abs/2610.02456) | 该论文通过用户调查发现 AI 编程代理沙盒采用率低下的可用性障碍，并据此开发了 SideKernel——一个面向 macOS 上 AI 编程代理的开源、易用的 microVM 沙盒。 |
| [^220] | [FinDialogLens: Event Extraction over Multi-Party Dialogue for Missed-Trade Identification in Financial Chatrooms](https://arxiv.org/abs/2610.02455) | 提出FinDialogLens混合LLM流水线，以紧凑的微调分类器作为推理时脚手架，对多方金融聊天对话进行RFQ事件抽取，从而准确识别遗漏交易的最终价格与交易结果，配合GPT-4o分别达到92.1%和94.3%的准确率。 |
| [^221] | [Reinforcement Learning Techniques for the Optimization of Target Polarization in Nuclear Physics Scattering Experiments](https://arxiv.org/abs/2610.02452) | 该研究提出将高斯过程代理模型与强化学习相结合的数据驱动控制框架，利用其校准的不确定性估计来自动优化核物理实验中动态极化靶的微波频率调节，以应对辐射损伤带来的材料特性变化。 |
| [^222] | [Counterexample Generation via Per-Theorem Symbolic Verifiers: When Imitation Hurts and Reinforcement Repairs](https://arxiv.org/abs/2610.02444) | 该论文发布SymCE数据集（包含4,707个错误数学猜想及其可执行验证器），发现仅用反例做监督微调会陷入“模仿陷阱”、使真定理识别率从0.27崩溃至0.00，而基于验证器稀疏奖励的强化学习（RLVR）不仅能修复这一退化，还能超越基线达到0.66。 |
| [^223] | [Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval](https://arxiv.org/abs/2610.02438) | 该论文提出将大语言模型对知名算法的代码生成重新定义为“参数化代码检索”任务，并引入AlgoREval基准（涵盖599个问题、77个经典算法、7种编程语言和4种图输入表示）来独立评估这一能力，发现不同语言和输入表示之间的检索准确率差异显著。 |
| [^224] | [Learning Style, Forgetting Semantics: A Case Study of SFT and RFT on Classification Tasks](https://arxiv.org/abs/2610.02437) | 本文通过将策略更新精确分解为语义与风格两个成分，揭示了SFT比RFT遗忘更多的原因——SFT会沿教师风格偏好产生离轴风格漂移从而破坏语义记忆，而RFT能保持类内风格对称性。 |
| [^225] | [Geometry-Aware Time Reparameterization for Flow-Map Distillation](https://arxiv.org/abs/2610.02427) | 提出一种几何感知的时间重参数化方法，为学生模型在法向加速度大的轨迹区域分配更多蒸馏时间，在保持教师几何路径和终端分布的同时使流映射蒸馏更易学习。 |
| [^226] | [Mitigating Private Data Leakage in LLMs with Whiteout](https://arxiv.org/abs/2610.02418) | 本文提出Whiteout工具，通过用精心设计的混淆样本覆盖个人敏感信息，来防止大语言模型复现泄露个人隐私，相比机器遗忘方法在保护隐私的同时能更好地维持模型效用与安全性。 |
| [^227] | [Efficient Neural Field Learning via Adaptive Coverage and Focused Sampling](https://arxiv.org/abs/2610.02410) | 提出ACES采样框架，通过解耦覆盖与重要性——利用自适应空间分区保证域覆盖、采用区域级重要性加权聚焦关键区域——从而降低梯度方差并显著提升隐式神经表示的训练效率。 |
| [^228] | [When Terminal-Agent Training Stalls: Demystifying Data Generation and Verification Challenge](https://arxiv.org/abs/2610.02405) | 该论文揭示了用前沿模型作为元智能体自动生成终端强化学习训练任务时的三类故障（基准无效、评测框架脆弱、奖励错位），并通过实验证明任务可解性区间是模型特定的，主张应将可解性区间校准、验证器审计和基础设施错误核算作为一等评估标准。 |
| [^229] | [Inherit-MAS: Test-Time Evolution of Multi-Agent Systems through Workflow and Execution Inheritance](https://arxiv.org/abs/2610.02396) | 该论文提出Inherit-MAS框架，借鉴生物进化中遗传与选择的机制，在工作流和执行两个层面实现显式继承，使基于大语言模型的多智能体系统能够在测试时高效演化工作流，同时避免破坏有用组件和产生冗余计算。 |
| [^230] | [FlashSinkhorn 2: Block-Sparse Entropic Optimal Transport](https://arxiv.org/abs/2610.02395) | 提出FlashSinkhorn 2，通过粗阶段质心求解加势提升与块稀疏精细阶段相耦合的两阶段设计，在单个GPU上将大规模熵正则最优传输问题求解至预设边际残差，同时保证被省略块贡献的有界性。 |
| [^231] | [The Surprising Effectiveness of Shared Memory in Looped Transformers](https://arxiv.org/abs/2610.02383) | 提出让循环Transformer在预训练时共享内存（仅第一次递归写入键值缓存、后续递归读取并保留自身短窗口）的方法，不仅不损失质量反而提升质量，在减少76-79%上下文内存的同时刷新了循环模型的质量-内存边界。 |
| [^232] | [THPL: A Vision-to-Language Decision Support Framework for Rainbow Trout Feeding Management in RAS](https://arxiv.org/abs/2610.02378) | 该论文提出THPL框架，通过轨迹活动系数量化、层次化行为编码以及结合专家规则的LoRA微调大语言模型，将虹鳟行为视觉信息转化为可执行、可解释的精准投喂决策，实现循环水养殖中视觉到语言的决策支持。 |
| [^233] | [Coco: An Agentic Copilot for the Hardware--Software Co-Design Lifecycle](https://arxiv.org/abs/2610.02376) | Coco是一个与TPU架构师共同部署的智能体平台，通过将全新仿真扫描数据自动注册进规范化数据库、让LLM智能体基于真实仿真证据进行推理，从而加速了机器学习模型与硬件加速器的协同设计生命周期。 |
| [^234] | [EviDent-CBCT: Evidence-Bottlenecked Report Generation from Dental CBCT under Non-Exhaustive Report Supervision](https://arxiv.org/abs/2610.02375) | 该论文提出EviDent-CBCT框架，通过解剖感知的离散证据记录、牙科逻辑一致性校正和可靠性感知训练，解决了牙科CBCT常规报告标注不完整（未提及≠不存在）的问题，实现仅依据证据记录即可自动生成牙科报告。 |
| [^235] | [Hop-Decayed Influence: New Vulnerabilities of Structural Auxiliary Indexing in GraphRAG Pipelines with LLM](https://arxiv.org/abs/2610.02373) | 提出跳数衰减影响（HDI）攻击，通过查询感知的影响力传播识别并破坏GraphRAG流水线中的模式级辅助索引结构，仅修改0.016%的辅助结构即可达到88-94%的攻击成功率，实现1:N放大效应。 |
| [^236] | [Traversing the Satisfaction-Diversity Frontier in Text-to-Image Diffusion](https://arxiv.org/abs/2610.02372) | 提出SatisDive，一种无需训练的推理时方法，通过将文生图生成建模为“满意化”问题——要求每张图像满足奖励下限、整批图像满足多样性截止标准——实现了对奖励与多样性之间帕累托前沿的有效遍历。 |
| [^237] | [Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning](https://arxiv.org/abs/2610.02370) | 提出了Isaac-Net，一个GPU批处理的5G新空口模块，可与Isaac Lab物理仿真同步，对数千个并行环境同时进行时隙级的5G上行链路模拟，使大规模并行机器人学习能够实现网络在环训练。 |
| [^238] | [Automating the Application of HCI Principles: Skills for On-Demand UI Construction, the Human-AI Space to Think, and the Future of HCI](https://arxiv.org/abs/2610.02369) | 该论文提出了一个“思考空间”框架，将用户与AI的对话作为共享的结构化认知工作空间，使按需生成的用户界面成为用户思维的延伸，并借助经典HCI设计知识的自动化应用，实现从“生成界面”到“良好生成界面”的跨越。 |
| [^239] | [Lexicographic Multi-Objective On-Policy Distillation](https://arxiv.org/abs/2610.02359) | 提出了字典序多目标在线策略蒸馏（LMOPD），一种多教师蒸馏方法，在显式优先级保护下整合奖励专门化策略，确保低优先级目标（如简洁性）不会以牺牲高优先级目标（如正确性）为代价而提升。 |
| [^240] | [DeReAct: Decomposed Reasoning and Acting for Reliable AI Agents](https://arxiv.org/abs/2610.02351) | DeReAct提出了一种模块化智能体架构，通过将动作验证（Critic）与任务完成认证（Context Manager）从单一LLM策略中外置为独立门控机制，防止错误传播和无效完成声明，在GAIA和SWE-bench Verified上对较弱模型带来了最显著的Pass@1提升。 |
| [^241] | [MIRROR: Multipath Quorum Integrity for LLM Multi-Agent Communication](https://arxiv.org/abs/2610.02349) | 提出MIRROR，一种通信层完整性原语，通过将消息负载复制到k条逻辑路径并要求严格多数路径返回相同摘要，来防御大语言模型多智能体系统中篡改传输消息的中间智能体（AiTM）攻击。 |
| [^242] | [A Multi Method Importance and Performance Efficiency Analysis of Topological Metrics for Natural Visibility Graph Based Cyber Attack Detection](https://arxiv.org/abs/2610.02342) | 本研究提出一种整合SHAP、分组置换重要性、Boruta和RFE四种方法的共识排名策略，筛选出仅含少量度量的NVG拓扑度量子集（如Top3配置），在基于CNN的网络攻击检测中保持了分类性能，同时显著提升了计算效率。 |
| [^243] | [World Editing: Intervening on Executable Worlds at Increasing Depth](https://arxiv.org/abs/2610.02331) | 该论文提出了“世界编辑”的形式化框架与“干预深度”维度，并基于Minecraft和Terraria构建了包含110个任务、1100余条评估标准的IGMBench基准，揭示前沿编码智能体已能解决其中78.2%的可执行世界编辑任务。 |
| [^244] | [Choosing Before Acting: Comparative Value Estimation for Long-Horizon Tool-Use Agents](https://arxiv.org/abs/2610.02330) | 提出CITA方法，让智能体在执行工具调用前通过对同一上下文下备选调用的比较推理来估计其长时程价值，从而解决长时程工具使用中最终结果奖励信用分配弱、步骤级监督获取成本高的问题。 |
| [^245] | [Slow-Fast Multi-Teacher On-Policy Distillation for Capability Preservation](https://arxiv.org/abs/2610.02324) | 提出 SF-MOPD 方法，通过将教师直接更新的快速学生模型与作为动态能力参考的指数移动平均慢速模型相耦合，在多教师在线蒸馏中实现领域专长获取与通用能力保持的平衡。 |
| [^246] | [DeskForge: Dense Supervision from Desktop Environments for Computer-Use Agents](https://arxiv.org/abs/2610.02320) | 本文提出可控桌面环境DeskForge，通过组合和变换真实应用程序生成大规模密集标注语料库DeskForge-1M（含120万条桌面观测与1.597亿个元素实例），有效提升了视觉语言模型在复杂桌面场景中的动作目标定位能力。 |
| [^247] | [SimuVerity: Benchmarking Agents for Engineering-Grade Simulink Model Generation](https://arxiv.org/abs/2610.02304) | 提出SimuVerity基准，包含101个跨十个工程领域的Simulink模型生成任务，采用分层评估器从六个工程维度对模型评分，发现最佳智能体系统总分仅为42.86，证明结构相似度并不能衡量模型的工程性能。 |
| [^248] | [Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation](https://arxiv.org/abs/2610.02300) | 本文揭示文本到图像生成中全局不安全防护存在覆盖与选择性的内在权衡，并提出免训练的CALM方法，通过提示词局部的反事实校正精准编辑违规词元表示并抑制不安全残余成分，在不损害良性提示词的前提下有效提升安全性。 |
| [^249] | [$\Psi$-Resilience: Model-Free Feature Importance from 1D Topological Signals](https://arxiv.org/abs/2610.02299) | 提出了一种基于一维拓扑信号的无模型特征重要性方法 $\Psi$-Resilience，它通过类条件密度差异构建不一致性景观并利用其持续性定义韧性评分，从而产生上下文鲁棒且可审计的特征排序。 |
| [^250] | [EditHero: A Benchmark for Long-Horizon Part-Level 3D Editing and Vibe Modeling](https://arxiv.org/abs/2610.02298) | EditHero是首个长时程部件级3D编辑基准测试，通过确定性组装引擎和人工审核来比较自顶向下的非智能体方法与自底向上的LLM/VLM智能体代码编辑方法，结果显示非智能体方法常遗漏所要求的变更并破坏本应保持不变的区域。 |
| [^251] | [Diffusion-Based Synthetic Data Pretraining for Enhancing Activity Recognition](https://arxiv.org/abs/2610.02292) | 本研究提出利用扩散模型生成合成传感器数据进行预训练、再在真实数据上微调的两阶段训练策略，以增强CABiGRU模型对进食、饮水等细微少数类别人体活动的识别能力。 |
| [^252] | [The AI Risk Observatory: What Can We Learn from AI Disclosures in Annual Reports About Societal Resilience?](https://arxiv.org/abs/2610.02281) | 该研究通过可复现的两阶段LLM分类流水线分析了9,821份英国上市公司年报，发现2020-2025年间AI风险披露比例从2.8%激增至41.2%，证明经LLM规模化处理的年报能为社会韧性研究提供关于企业AI应对的有用信号。 |
| [^253] | [Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses](https://arxiv.org/abs/2610.02267) | 该论文通过严格配对与自审计的评估发现，托管型System-1决策模型Jev在11个代理决策点中的9个上显著优于开源模型Laya，但两者在零样本模型路由上均未超过随机水平，且开源模型对选项顺序和候选数量高度敏感。 |
| [^254] | [MintFlow: Minimal Trajectory Intervention for Constrained Flow Matching](https://arxiv.org/abs/2610.02260) | MintFlow是一种无需训练的约束采样框架，通过对预训练流轨迹施加最小干预来满足目标约束，在执行约束的同时最大限度地保持样本对预训练数据分布的保真度。 |
| [^255] | [Overcoming Challenges of Interpretive Structural Modeling with Large Language Models](https://arxiv.org/abs/2610.02254) | 本工作将大语言模型作为“不完美专家”引入解释结构建模（ISM），以克服传统专家交互方法繁琐且难以扩展至数百个变量的挑战，并通过对比实验证明逐行和全图因果图发现方法效果最佳。 |
| [^256] | [Counterfactual Predictions in Scientific Emulators Without Controlled Experiments](https://arxiv.org/abs/2610.02252) | 提出 ReRoute 框架，无需受控实验或模拟器数据，仅通过将查询输入固定为参考值并沿已知机制路径重新引入其变化，结合事实数据微调，即可让科学模拟器准确回答“如果条件不同会怎样”的反事实预测问题。 |
| [^257] | [Toward Controlling Biology with Language:Offline Learning of Prompt-Conditioned Interventions for Cells, Organoids, and Biobots](https://arxiv.org/abs/2610.02247) | 该论文提出将已有的生物干预及其实验结果档案作为固定的离线数据集，利用视觉-语言模型自动判断存档结果与自然语言描述是否匹配，从而在无需新实验和人工验证的情况下，学习从自然语言到细胞、类器官和生物机器人干预措施的映射。 |
| [^258] | [RxnOptBench: Benchmarking LLMs for Reaction-Condition Optimization in Organic Methodology](https://arxiv.org/abs/2610.02242) | 该论文提出了RxnOptBench，这是首个基于2025年真实发表的有机方法学论文中湿实验优化数据构建的基准，用于评估大语言模型阅读真实条件筛选表格并选出最优反应条件（催化剂、配体、溶剂、温度等）的能力。 |
| [^259] | [Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts](https://arxiv.org/abs/2610.02241) | 提出了一个端到端的软硬件协同设计框架，通过连续重参数化实现稀疏性与量化的可微联合优化，将万亿规模MoE的专家权重压缩为硬件原生的低精度半结构化稀疏表示，从而在稀疏张量核心上加速执行并降低部署的内存瓶颈。 |
| [^260] | [CORE: COverage CAlibration and Evicted-Mass REdistribution for KV Cache](https://arxiv.org/abs/2610.02235) | 提出CORE方法，通过覆盖校准与逐出质量再分配机制，在KV缓存压缩中同时利用Top-B排序保留互补KV状态，并用被排除的分配质量补偿被逐出的注意力质量，从而有效降低长上下文解码中的逐出误差。 |
| [^261] | [Causal discovery identifies pathways linking physical activity to dementia risk in the UK BioBank](https://arxiv.org/abs/2610.02221) | 本研究将大语言模型引导的因果发现与中介分析相结合，在英国生物样本库4万多名老年人中系统识别出体力活动降低痴呆风险的多个中介通路，并确定抑郁为核心中介通路（占总体关联的15.1%）。 |
| [^262] | [Causal Memory Policy: Making Memory Utility Identifiable by Intervening on Retrieval](https://arxiv.org/abs/2610.02070) | 该论文提出因果记忆策略（CMP），通过干预检索过程、以已知倾向性为采样的记忆保留固定数量的上下文槽位，解决了记忆因从未被检索而导致效用无法识别的检索层面正性违背问题，实现了记忆效用的无偏估计与最优保留决策。 |
| [^263] | [Cross-Lingual Alignment for Decoder-Only Models using MoE Routers](https://arxiv.org/abs/2610.01921) | 该论文提出一种创新方法，利用混合专家（MoE）路由器的输出作为对齐目标，在仅解码器大语言模型中实现跨语言表示对齐，从而提升跨语言迁移能力。 |
| [^264] | [CONTRA: Discovering and Qualifying Behavior-Changing Questions for Selective Clarification in LLM Code Generation](https://arxiv.org/abs/2610.01769) | CONTRA是一种无需训练的方法，通过广泛发现候选澄清问题，并结合语义评估与基于执行的验证来筛选出真正会改变代码行为的关键问题，从而让LLM代码生成智能体进行选择性澄清提问，在防止因假设错位导致行为偏差的同时避免不必要的打扰。 |
| [^265] | [Rethinking Probability-Based Reinforcement Learning From Posterior Concentration](https://arxiv.org/abs/2610.01458) | 该论文发现基于概率的奖励存在后验集中现象，即随着推理链变长奖励会坍缩到低方差区间而难以区分，导致GRPO训练不稳定且低效，并提出显式建模该现象的无验证器强化学习框架RLCPR，以提升优化稳定性和token效率。 |
| [^266] | [Fold'EM: Direct atomic structure inference from Cryo-EM particles](https://arxiv.org/abs/2610.01358) | 本文提出Fold'EM方法，无需先进行密度重建即可直接从冷冻电镜颗粒图像推断原子结构，从而降低样本复杂度并提升结构测定的效率。 |
| [^267] | [Screw Attention: Rigid-Body Algebra Inside a Transformer](https://arxiv.org/abs/2610.00904) | 提出螺旋注意力层，将token间关系建模为空间变换与关节螺旋，使消息传递天然具有坐标系等变性，且单层即可表达刚体力学速度递归，在仿真操作任务上达到或超越同规模基线方法。 |
| [^268] | [Cogentic: Multi-Agent Orchestration for Automated Proof Discovery](https://arxiv.org/abs/2609.40324) | Cogentic 通过编排器调度多个独立证明者、多组件对抗性验证以及持久化已验证账本的迭代“证明—验证”循环，实现了开放研究问题上的自动证明发现，并以 Gemini 为基础模型获得了新研究成果。 |
| [^269] | [Learning Skills from Historical Action Trajectories: Action Experience Dictionary for World Action Models](https://arxiv.org/abs/2609.40219) | 该论文提出动作经验字典（AED），将历史动作轨迹编码为共享动作嵌入，使世界动作模型能够复用技能并建模跨任务语义关系，从而提升操作任务的动作生成能力。 |
| [^270] | [Tactile Curiosity Drives Robot Interaction](https://arxiv.org/abs/2609.40134) | 本文提出TacEx框架，将模型不确定性按感官模态分解并把好奇心引向触觉通道，使强化学习探索以触觉反馈为导向，从而提升机器人操作技能学习的样本效率。 |
| [^271] | [PTNO: Training Neural Operators with Noisy Monte Carlo Estimates for Particle Transport Problems](https://arxiv.org/abs/2609.40090) | 该论文提出粒子输运神经算子PTNO，可直接从含噪、低成本的蒙特卡洛标签中学习粒子输运代理模型，并证明了无偏噪声标签的平方损失与收敛解损失共享同一极小值点，从而解决了高方差与高动态范围两大挑战，大幅降低了训练成本。 |
| [^272] | [Autoresearch in Mixed-Integer Linear and Nonlinear Programming](https://arxiv.org/abs/2609.39360) | 提出AutoMIP——一种通过想法池与算法树搜索来组织混合整数规划长周期自动化研究的可复用智能体技能，在MILP和MINLP基准测试中取得了所评估框架中最高的成功率。 |
| [^273] | [CRAFT: Causal Responsibility and Failure Tracing in Medical Vision Language Models](https://arxiv.org/abs/2609.38810) | 该研究揭示了医学视觉语言模型中“仲裁失败”（文本覆盖视觉依据）与“刹车失败”（证据不足仍作答）两种安全风险分别由空间上不重叠的注意力头群体介导——仲裁头分布于中深层宽频带、刹车头集中于中后层窄频带，从而实现对模型失败的因果追踪与定位。 |
| [^274] | [Learning to Route in Visual Space via Multi-Step Embedding Retrieval](https://arxiv.org/abs/2609.38743) | 该论文提出VHOP基准框架和VHOP-Router端到端训练流程（结合监督微调、在线模仿学习与强化学习），将标准嵌入模型改造为能直接在嵌入空间中完成多步视觉导航的检索工具，从而突破LLM智能体视觉搜索中单步检索的性能瓶颈。 |
| [^275] | [From Solo to Social Learning: Characterizing Recursive Social Improvement in LLMs](https://arxiv.org/abs/2609.38516) | 该论文提出“递归社会改进”这一新概念，并发现尽管经典社会学习算法能从同伴中受益，但当每个LLM智能体各自追求自身奖励时，当前的LLM无法通过相互学习改进整个群体，其每token收益反而低于独立学习。 |
| [^276] | [Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S](https://arxiv.org/abs/2609.38021) | 该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。 |
| [^277] | [WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation](https://arxiv.org/abs/2609.37687) | 提出了“预算受限主动测试时自适应”这一新设定，并给出WISE-ATTA方法，利用在线计算的轻量级信号决定何时对测试批次施加监督，将ATTA的核心挑战从“标注什么”转变为“何时标注”。 |
| [^278] | [From Learner Behavior to Reusable Skills for Effective and Efficient Learner Simulation](https://arxiv.org/abs/2609.37157) | Learner2Skill将从历史交互中获得的学习者模拟能力外化为持久可复用的“模拟技能”，该技能捕获学习者的学习状态与回答模式并随新交互演进，可通过轻量级校准迁移到新的大语言模型，从而更高效、更忠实地模拟学习者行为。 |
| [^279] | [ARGOS: Reinforcement Learning-Driven Multidimensional Elasticity for Service Orchestration in the Computing Continuum](https://arxiv.org/abs/2609.37085) | ARGOS提出了一种基于强化学习的端到端控制器，将计算连续体中的多维弹性建模为按请求的马尔可夫决策过程，在容量达到上限时动态调整分析质量以吸收需求变化和集群压力，同时保障客户端定义的质量范围。 |
| [^280] | [Embedded Bi-Temporal Building Damage Assessment for On-Board Data Reduction](https://arxiv.org/abs/2609.37013) | 提出了一种基于YOLOX孪生检测器的双时相建筑物损毁评估流水线，通过将灾前参考图像压缩至64倍潜空间编码上传至卫星，并在星载端仅下传边界框与损毁类别等目标级产品而非完整场景，实现了天地双向数据量的大幅缩减。 |
| [^281] | [SafeCoEvo: Co-Evolving Safety Harnesses and Guards for LLM Agents at Test-Time](https://arxiv.org/abs/2609.36580) | 提出SafeCoEvo框架，在测试时协同演化快速自适应的安全线束（S-Harness）与安全防护（S-Guard），利用积累的运行时经验持续提升LLM智能体应对未见任务安全风险的能力。 |
| [^282] | [From Migration to Calibration: Preserving Agent Capabilities across Models, Jurisdictions, and Scale](https://arxiv.org/abs/2609.35149) | 提出将智能体校准构建为标准优先的适配框架，通过定义基础能力、技术环境与用户情境标准，系统诊断差距并应用修订，从而在模型更换、跨管辖区部署和规模化过程中保持智能体能力不退化。 |
| [^283] | [Automated Feature Engineering, AutoML, and Decision-Focused Learning for Improved Energy Consumption Forecasting](https://arxiv.org/abs/2609.35013) | 本论文提出面向能源领域的自动化特征工程算法AutoEnergy，与AutoML集成实现端到端能源消耗预测建模，在18个真实数据集上将预测误差降低19.52%-84.72%。 |
| [^284] | [MASCIT: A Mask-Aware State Space Classifier for Naturally Irregular Time Series](https://arxiv.org/abs/2609.34409) | 提出掩码感知状态空间分类器MASCIT，通过观测掩码与门控时间聚合有效处理异步观测、缺失值等自然不规则性，在34个不规则时间序列数据集上取得最优聚合性能。 |
| [^285] | [MaskCoFT: Masked Co-Adaptive Fine-Tuning for Memory-Efficient MoE Inference](https://arxiv.org/abs/2609.34077) | 提出MaskCoFT方法，利用可学习二值掩码限制每层的Top-K路由，并通过交叉熵损失协同微调路由器与专家，使专家在卸载推理场景下被高效复用，从而降低MoE模型的内存开销并保持推理性能。 |
| [^286] | [$T^5$: Twin-Critic Training for Token-Level Thoughts in Reinforcement Mid-Training](https://arxiv.org/abs/2609.32791) | 提出双评论家方法T⁵，通过条件矩鞍点目标与信号保留约束，从单条生成轨迹中校准词元级优势，解决了强化中期训练中token级信用分配的高效性问题。 |
| [^287] | [What Does a ProcGen Generalization Gap Measure? Action Rules, Residual Entropy, and the Missing Random Floor](https://arxiv.org/abs/2609.32532) | 该论文提出强化学习的泛化差距应对照“随机下限”（均匀随机策略在同一评估框架和相同关卡上的回报）来解读，并证明测试时动作规则（采样与 argmax）的选择以及动作等效性造成的残余熵会显著改变 ProcGen 基准上泛化结论的含义。 |
| [^288] | [ADATEX4D: adaptive texture capacity allocation for 4D gaussian splatting](https://arxiv.org/abs/2609.29963) | 提出AdaTex4D自适应纹理容量分配模块，根据可见性和局部尺度动态调整每个高斯RGBA三平面的分辨率，在保持重建质量的同时将4D高斯泼溅的纹理存储减少一半以上。 |
| [^289] | [Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation](https://arxiv.org/abs/2609.20822) | 本文发现编码智能体在机器人操作中因规划环节未能将安全约束设为优先事项而系统性碰撞障碍物（而非感知或指令问题），并通过将操作分解为路径阶段和接触时刻来定位失败根源，提出了障碍物感知框架以实现安全操作。 |
| [^290] | [Reach or Solve? Attributing Agentic RL Gains with Checkpoint Handoffs](https://arxiv.org/abs/2609.19636) | 本文提出“检查点交接”评估协议，通过克隆一个检查点到达的状态并移交给另一个检查点而无需重新训练，从而将智能体强化学习的收益分离归因为“到达状态的能力”与“在给定状态下解决问题的能力”两个独立成分。 |
| [^291] | [BusMA: A Bus Communication Substrate for Multi-Agent Systems](https://arxiv.org/abs/2609.15054) | 受计算机总线架构启发，BusMA提出了一种多智能体通信框架，允许任何智能体通过共享总线信道直接与其他智能体通信，突破了传统分层管理者-工作者或路由器消息传递结构对智能体自主性的限制。 |
| [^292] | [LPA-CWM: A Learned Physical Adjudicator for Motion Reasoning with Counterfactual World Models](https://arxiv.org/abs/2609.14073) | 提出LPA-CWM框架，利用轻量级学习型物理裁决器学习候选响应的可靠性权重，从反事实世界模型中更准确地恢复运动，并引入联合衡量定位、轨迹完整性、可见性和连续性的CMC评估协议。 |
| [^293] | [Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion](https://arxiv.org/abs/2609.12223) | 本文提出GRACE模型，通过早期融合的几何残差加合物条件化方法调整预训练分子几何编码器，将加合物感知的残差学习目标与编码器内的加合物条件化机制相结合，显著提升了对气相分子离子碰撞截面的三维预测精度。 |
| [^294] | [Suan: Rectifying Direct Preference Safety Alignment in Large Language Models](https://arxiv.org/abs/2609.08634) | Suan是一种新颖的偏好优化算法，通过直接在梯度层面构建优化目标（绕过标准变分推导），使大语言模型在实现卓越安全对齐的同时完全保留回应实用性。 |
| [^295] | [Learning transferable human physiology from two million hours of sleep with SleepFM-2](https://arxiv.org/abs/2609.06849) | SleepFM-2是一个基于来自26个队列超过两百万小时多模态生理数据的睡眠基础模型，显著提升了疾病预测、睡眠分期和事件检测能力，并能从多导睡眠图表征中预测电子健康记录中的多种疾病表型，还可迁移至可穿戴设备。 |
| [^296] | [LayerRoute: Action-Conditioned Mixture-of-Layers Routing for Vision-Language-Action Policies](https://arxiv.org/abs/2609.06079) | 提出LayerRoute，一个动作条件的混合层路由接口，使VLA策略能够根据具体动作需求自适应地访问和混合VLM不同层次的表示，突破了传统固定层分配的限制。 |
| [^297] | [trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories](https://arxiv.org/abs/2609.00038) | 仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。 |
| [^298] | [Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents](https://arxiv.org/abs/2608.30362) | 该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。 |
| [^299] | [Stratified Consistency Distillation for Natural Language Formalization](https://arxiv.org/abs/2608.30258) | 提出分层一致性蒸馏方法，通过对前沿大模型生成的多个逻辑翻译按语义等价性聚类，并依据熵水平采用不同策略筛选伪标签来微调小模型，从而提升自然语言到逻辑公式翻译的准确性。 |
| [^300] | [Looking Again: Measuring Sycophancy in the Reasoning Chains of Multimodal Models Under Pressure](https://arxiv.org/abs/2608.28623) | 该论文提出了首个用于测量大型多模态推理模型谄媚行为的基准和数据集，通过四种视觉推理任务与五种压力条件评估模型在用户给出错误答案时的表现，发现谄媚行为在压力下普遍存在，不仅体现在最终答案中，也出现在推理链中。 |
| [^301] | [ConfAL-WM: Confidence-Guided Active Learning for Action-Conditioned World Models](https://arxiv.org/abs/2608.25572) | 提出ConfAL-WM框架，通过在UNet解码器特征上附加轻量级置信度探针生成潜空间密集置信度图，并聚合为任务、帧、补丁三个层级的分数，实现具身世界模型后训练中的数据预算分配与局部化训练增强。 |
| [^302] | [GlanceWAM: Sparse Test-Time Imagination for World-Action Models](https://arxiv.org/abs/2608.23927) | GlanceWAM通过在单一视频DiT骨干上将视觉想象与控制解耦——以异步方式在后台生成前瞻帧并直接在潜空间中消费——实现了机器人实时控制（48毫秒）与更优任务成功率的兼得。 |
| [^303] | [EXAM$^2$: $\underline{Ex}tending$ $\underline{A}udio$ $Understanding$ $in$ $\underline{M}ultilingual$ $and$ $\underline{M}ultimodal$ $Analysis$](https://arxiv.org/abs/2608.23758) | 本文提出了EXAM²，一个覆盖六种语言和多种音频模态（含视觉图像）的多语言多模态音频理解基准，旨在更真实地评估场景感知音频推理和跨模态理解能力。 |
| [^304] | [WAM-OPD: On-Policy Distillation for World Action Models](https://arxiv.org/abs/2608.22364) | 提出WAM-OPD，一种部署一致的同策略蒸馏方法，通过冻结教师模型标注学生行动历史并联合优化视频与动作损失，在无需稀疏奖励强化学习的情况下修复加速学生模型的任务能力。 |
| [^305] | [Interrupting the Chain: Human Perception of AI-Generated Disinformation Through a Kill Chain Lens](https://arxiv.org/abs/2608.21389) | 通过杀伤链框架的实证研究揭示，人类对AI生成虚假信息的检测存在感知-准确性差距、LLM文本难以区分以及认知疲劳导致虚假新闻检测下降10.2%等关键弱点，为主动防御提供了干预点。 |
| [^306] | [An Irreducible Quantum Advantage in Aligning World Models with Reality](https://arxiv.org/abs/2608.19779) | 本文证明即使真实世界是经典的，经典世界模型也无法完美对齐代理策略，而量子模型可能提供不可约的优势。 |
| [^307] | [Safe and Robust Neural Policy Learning with Statistical Verification for Sim-to-Real Deployment in Robotics](https://arxiv.org/abs/2608.06481) | 本文提出一种课程驱动的闭环框架，将基于场景的进化策略与统计模型检测验证相结合，在协同优化策略性能的同时逐步扩大安全操作边界，最终生成带有统计验证安全与性能保证的神经控制器，助力可靠的仿真到现实部署。 |
| [^308] | [Escaping Oversquashing: Addressable and Support-Aware Global Memory for Message Passing Networks](https://arxiv.org/abs/2608.02709) | 该论文提出一种兼具可寻址性与支持感知的全局记忆机制，通过乘性读写映射仅用对数级地址编码维度即可选择 M 个记忆行，并借助学习到的私有锚点保持读取有界，从而解决消息传递网络中多节点共享虚拟节点全局状态导致的瓶颈问题。 |
| [^309] | [World Action Planner: Generalizable Robot Decision-Making with Action-Conditioned World Models](https://arxiv.org/abs/2607.27599) | 提出了World Action Planner——一种利用动作条件世界模型进行“想象”并采用由粗到细的搜索策略来优化动作计划的机器人规划系统，显著提升了机器人对新场景、新布局和新任务的泛化决策能力。 |
| [^310] | [Early Detection of Distributed Backdoors in Multi-Agent LLM Systems: A Characterization Study](https://arxiv.org/abs/2607.24893) | 多智能体大语言模型系统中的分布式后门攻击将加密载荷片段分散到多个被投毒的工具中，在第一个片段注入前几乎无法被检测，而一旦注入开始，前缀检测器便可标记99.5%的成功攻击。 |
| [^311] | [ORACLE: Agentic AI Orchestrator Routing Via Adaptive Verifier Calibration Feedback](https://arxiv.org/abs/2607.22465) | ORACLE提出了一种并发感知的在线智能体路由机制，通过将自适应路由与自适应验证器校准反馈相结合，解决了固定验证器难以泛化到异构智能体任务、以及验证器位于关键路径导致并发请求服务质量下降的问题，且无需训练即可即插即用。 |
| [^312] | [Deep learning-based prediction of time-resolved adhesive forces in viscoelastic Hertzian contacts](https://arxiv.org/abs/2607.19060) | 本文提出一种标量条件化的有状态序列到序列深度学习模型，结合固定测量步长（FMS）表示方法，能够从位移历史快速预测粘弹性赫兹接触中的完整时间分辨粘附力演化，克服了传统数值模拟计算成本高、无法用于实时应用和设计优化的局限。 |
| [^313] | [Assistant or Actor? Student Trust, Control, and Delegation Regret When Using a General-Purpose AI Agent](https://arxiv.org/abs/2607.18257) | 该研究提出“委托后悔”这一新概念，并通过对照实验发现用户对通用AI代理的信任是按任务而非按代理整体来校准的——用户在咨询类和低风险任务中给予广泛自主权，但对不可逆操作则要求确认。 |
| [^314] | [How Artificial Intelligence LLM Engines Shape the Global Conflict Information Environment](https://arxiv.org/abs/2607.14197) | 本研究向五个主流AI答案引擎提出关于28场冲突的大量问题并对照实证证据评分，发现冲突相关的可检索记录越稀薄，模型越容易产生幻觉和错误，且这些稀薄记录最容易被生成式引擎优化（GEO）操纵，从而构成全球冲突信息环境中结构性的虚假信息风险。 |
| [^315] | [Rethinking the Evaluation of Harness Evolution for Agents](https://arxiv.org/abs/2607.12227) | 本文重新评估了智能体装备演化方法，指出其与简单搜索基线在匹配预算下对比的必要性，并揭示了共享基准可能导致过拟合的风险。 |
| [^316] | [A Multi-Timescale Recursive Self-Improvement Engine for Open-Ended Persona Growth](https://arxiv.org/abs/2607.08252) | 该论文提出AutoPersonas引擎，首次将递归自我改进从“提升智能”转向“人格成长”，通过多时间尺度地递归修订状态、证据和生活环境来实现开放式人格发展，并识别出递归生成中的核心失效模式“自锁”及其成因。 |
| [^317] | [ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation](https://arxiv.org/abs/2607.06565) | ELSA3D提出弹性语义锚定机制，通过尺度感知八叉树分词器与稀疏的跨模态锚定token，在匹配的抽象尺度上显式对齐语言与几何推理，实现统一的3D理解与生成。 |
| [^318] | [SovereignPA-Bench: Evaluating User-Owned Personal Agents under Evolving Intent, Platform Mediation, and Consent Constraints](https://arxiv.org/abs/2607.05363) | 该论文提出SovereignPA-Bench基准，通过脚本化平台与用户在1,920个预订场景和288个取消场景中，检验个人智能体能否遵循用户不断演变的意图、抵御平台引导、最小化数据共享、事先征得同意并如实报告，从而将智能体的忠实度与用户负担分开评估。 |
| [^319] | [Sentence-Level Context Sensitivity as a Training-Free Detector of Unsupported Content, Evaluated Against Trained Verifiers](https://arxiv.org/abs/2607.04223) | 该论文提出将句子在有/无上下文时的似然差异作为免训练的句子级无依据内容检测器，在多段落RAG答案中其检测能力可与经过训练的验证器相媲美，且无需额外训练、成本更低。 |
| [^320] | [SovereignNegotiation-Bench: Evaluating User-Owned Personal Agents In Delegated Bargaining Under Privacy, Consent, Evidence, And Institutional Pressure](https://arxiv.org/abs/2607.02814) | 该论文提出SovereignNegotiation-Bench基准，将代理法中的五项义务（忠诚、服从、保密、坦诚、勤勉）操作化为对谈判日志的确定性检查，用以评估个人AI智能体在隐私、同意与机构压力下代表用户谈判的表现，并使违反义务的行为（如披露底线）产生可测量的因果性经济代价。 |
| [^321] | [Assessing Rule Adherence of LLM Adjudicators in Call of Cthulhu TRPG](https://arxiv.org/abs/2607.02802) | 该论文提出了基于《克苏鲁的呼唤》TRPG的多智能体对抗基准CoC-Seduce，通过“修辞注入”这一新型操纵手段，系统评估了大语言模型裁判在面对对抗性用户绕过规则时的规则遵守能力。 |
| [^322] | [ClarifyCodeBench: Evaluating LLMs on Clarifying Ambiguous Requirements for Code Generation](https://arxiv.org/abs/2607.00711) | 该论文提出了ClarifyCodeBench，一个基于真实编程任务、包含人工标注的模糊类型与澄清问答的新型交互式基准，用于评估大语言模型主动澄清模糊代码需求的能力。 |
| [^323] | [How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight Hallucination Detection Across Question Answering, Dialogue, and Summarisation](https://arxiv.org/abs/2606.29809) | 本研究系统基准测试了四种无需GPU的轻量级幻觉检测方法（ROUGE-L、语义相似度、BERTScore和NLI检测器）及其集成方案，在HaluEval的问答、对话和摘要任务上验证了基于公开模型的CPU可行方法可作为资源受限场景下幻觉检测的实用替代方案。 |
| [^324] | [Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference](https://arxiv.org/abs/2606.27090) | 本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。 |
| [^325] | [Multi-Modal Environment-Aware Beam Management for Massive MIMO: A Geometry-Driven Virtual Base Station Framework](https://arxiv.org/abs/2606.26567) | 提出一种几何驱动的可解释框架，利用区域LiDAR点云和位置信息构建离线虚拟基站数据库，通过镜像对称建模主导反射路径，实现大规模MIMO系统中高效的多模态环境感知波束管理。 |
| [^326] | [Latent Goal Prediction from Language for Model-Based Planning](https://arxiv.org/abs/2606.20627) | LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。 |
| [^327] | [Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish](https://arxiv.org/abs/2606.18717) | Morpheus 是一个面向土耳其语的形态感知神经分词器与词嵌入生成器，它通过可微分泊松-二项动态规划实现无损可逆的词素级分词，并能在同一次前向传播中同时输出分词结果和结构化词嵌入。 |
| [^328] | [Dissecting model behavior through agent trajectories](https://arxiv.org/abs/2606.17454) | 该论文提出“意图-执行”差距的概念，指出智能体性能本质上是系统问题而非单纯的建模问题，并开发了可跨多个模型家族（Claude、Gemini、GPT、Grok、Qwen）泛化的简单可定制框架SSA，以弥合模型能力与框架执行之间的鸿沟。 |
| [^329] | [Mental-R1: Aligning LLM Reasoning for Mental Health Assessment](https://arxiv.org/abs/2606.13176) | 本文提出面向心理健康领域的强化学习框架CRPO，通过分阶段熵正则化机制模拟人类认知过程，使大语言模型的推理与心理健康评估对齐，从而提升评估结果的可靠性。 |
| [^330] | [Rethinking RAG in Long Videos: What to Retrieve and How to Use It?](https://arxiv.org/abs/2606.13141) | 该论文提出了小时级长视频基准 V-RAGBench（每个答案唯一对应一个证据片段，实现检索与生成的解耦评估）以及无需训练的片段自适应方法 CARVE（并行多配置检索并按片段重排序选出最优“模态-粒度”配置用于生成），性能超越八种基线方法。 |
| [^331] | [A Language Model from 1913: Pretraining on Historical Text](https://arxiv.org/abs/2606.02991) | 该论文提出了TypewriterLM，一个在1913年前历史文本上预训练的72.4亿参数语言模型，通过构建540亿token的时间过滤历史语料库、基于历史词汇约束的指令微调方法以及包含2,344个事件的History-Event评估基准，实现了具有明确1913年知识截止时间且语言理解性能合理的时间定位语言模型。 |
| [^332] | [Planning Takes More Than Token Prediction: Causal Plan for Benchmarking and Building Physically Grounded Embodied Reasoners](https://arxiv.org/abs/2606.01810) | 本文提出 Causal-Plan-Bench 基准与百万级因果推理语料库 Causal-Plan-1M，揭示当前具身视觉语言模型偏向语言词元预测而缺乏物理因果推理能力，推动从语言统计先验向物理接地的因果规划转变。 |
| [^333] | [Efficient Exploration for Iterative Nash Preference Optimization](https://arxiv.org/abs/2606.01382) | 论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。 |
| [^334] | [Counterfactual Evidence Audits Predict LLM-Agent Susceptibility to Ranked Context](https://arxiv.org/abs/2606.00914) | 该论文提出一种反事实证据审计协议，通过让智能体面对两组镜像的五文档集合并测量其决策差异，能够高精度预测LLM智能体在面对45份文档的单边排序上下文时的易感性。 |
| [^335] | [On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance](https://arxiv.org/abs/2606.00467) | 提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。 |
| [^336] | [Detect Before You Leap: Mirage Detection in Vision-Language Models](https://arxiv.org/abs/2606.00435) | 提出了一种完全无监督、与模型无关的幻象检测方法TC-LIA，通过追踪冻结CLIP编码器各层中问题与图像的对齐情况，来判定视觉语言模型给出的答案应当发布还是保留。 |
| [^337] | [One Hypothesis Is Not Enough: Abductive Reasoning with Agentic Hypothesis Refinement over Knowledge Graphs](https://arxiv.org/abs/2605.31370) | 提出HypoAgent框架，通过智能体迭代修正机制改进知识图谱上的溯因推理，能够检测并纠正无法解释观测结果的假设，突破了单步假设生成的局限。 |
| [^338] | [Escaping the Capacity Ceiling: Routing on the Stiefel Manifold for Bilinear SPD Layers](https://arxiv.org/abs/2605.31043) | 提出SCAP层，通过交叉注意力将K个Stiefel专家滤波器动态组合为样本特定的双线性映射，从而突破SPD网络中单滤波器的容量上限，解决堆叠BiMap层无法提升容量的问题。 |
| [^339] | [Where Do Apparent LLM Clinical Triage Failures Arise? Localizing the Multiple-Choice Format Effect](https://arxiv.org/abs/2605.29889) | 利用稀疏自编码器分析，该研究发现LLM在多选题式临床分诊中的表现下降并非源于对病例医学信息的处理失败，而是发生在答案映射阶段——多选题答题框架在决策标记处抑制了本已可解码的急诊分级信息。 |
| [^340] | [GUI Agents for Continual Game Generation](https://arxiv.org/abs/2605.28258) | 提出PlaytestArena评估环境和Play2Code框架，让GUI智能体作为玩家实际试玩游戏并迭代反馈，从而实现具备可玩性验证的持续游戏生成。 |
| [^341] | [HyperGuide: Hyperbolic Guidance for Efficient Multi-Step Reasoning in Large Language Models](https://arxiv.org/abs/2605.24140) | 提出HyperGuide方法，利用双曲空间的几何特性将推理进展编码为引导信号，使大语言模型在单次生成的效率与树搜索的准确性之间实现高效平衡。 |
| [^342] | [EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation](https://arxiv.org/abs/2605.23954) | EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。 |
| [^343] | [FastKernels: Benchmarking GPU Kernel Generation in Production](https://arxiv.org/abs/2605.23215) | FastKernels提出了一个包含384个任务的生产级GPU内核生成基准，通过组合层次结构覆盖94.6%的HuggingFace Transformers架构，并直接在生产执行路径上以框架官方发布的内核为基准对候选内核进行内核级和端到端评分。 |
| [^344] | [Teger: Spatiotemporal Covariance for Probabilistic Traffic Forecasting](https://arxiv.org/abs/2605.18068) | 提出TEGER残差协方差模型，通过闭式更新在测试时动态校正交通预测的联合不确定性而无需重新训练，并可附加于冻结的时间序列基础模型。 |
| [^345] | [HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents](https://arxiv.org/abs/2605.17873) | HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。 |
| [^346] | [LEAF: A Living Benchmark for Event-Augmented Forecasting](https://arxiv.org/abs/2605.16358) | LEAF是首个面向事件增强预测任务的动态基准，通过递归检索智能体系统与双智能体交叉验证收集时间对齐的辅助上下文，将未来信息泄露从8.6%降至1.6%。 |
| [^347] | [KGPFN: Unlocking the Potential of Knowledge Graph Foundation Model via In-Context Learning](https://arxiv.org/abs/2605.14907) | KGPFN是一种基于先验数据拟合网络的知识图谱基础模型，通过将可迁移的关系表示与推理时对结构化局部邻域和全局上下文的上下文学习相结合，弥补了知识图谱推理中上下文学习的空白。 |
| [^348] | [GPart: End-to-End Isometric Fine-Tuning via Global Parameter Partitioning](https://arxiv.org/abs/2605.14841) | GPart 通过稀疏等距划分矩阵将可训练向量直接映射到完整权重空间，去除了 LoRA 式低秩重构，实现了端到端等距且高度参数高效的微调。 |
| [^349] | [ASH: Agents that Self-Hone in Long-Horizon Worlds](https://arxiv.org/abs/2605.14211) | ASH是一个无需奖励工程或专家标注的智能体系统，通过自我改进循环从自身轨迹学习逆动力学模型，进而从无标注网络视频中提取监督信号并保留关键时刻作为长期记忆，从而在《宝可梦：绿宝石》和《塞尔达传说：缩小帽》等需要数小时规划的长时程任务中实现自我提升。 |
| [^350] | [Measuring Google AI Overviews: Activation, Source Quality, Claim Fidelity, and Publisher Impact](https://arxiv.org/abs/2605.14021) | 该论文通过40天内跨19个主题发出的55,393个查询的大规模纵向测量，首次系统刻画了谷歌AI概览的激活率、引用来源质量与出版商影响，发现其总体激活率为13.7%（疑问式查询达64.7%），且其来源选择机制与传统搜索排序明显不同。 |
| [^351] | [ReForge: Refining Merged Models with Anchor-Regularized Regression](https://arxiv.org/abs/2605.12843) | 提出双层优化框架ReForge，将强合并模型作为锚点先验，通过贝叶斯线性回归对模块级进行精炼，并利用贝叶斯优化联合选择正则化强度与组装尺度，同时提供无需校准数据的任务向量Gram变体。 |
| [^352] | [NARA: Anchor-Conditioned Representation Learning for Heterogeneous Vector Geoentities](https://arxiv.org/abs/2605.12276) | NARA提出了一种自监督表示学习框架，通过融合几何距离与拓扑关系的空间上下文感知注意力机制，统一建模点、线、面等异构矢量地理实体，从而学习更全面的地理实体表示。 |
| [^353] | [DuetMoE: Coupling Inter- and Intra-Subgroup Robustness for Fair Medical Image Analysis](https://arxiv.org/abs/2605.10521) | 提出DuetMoE框架，通过亚群体感知的专家混合机制将组间公平性与组内鲁棒性相结合，为个体患者提供更公平可靠的医学图像分析。 |
| [^354] | [MolWorld: Molecule World Models for Actionable Molecular Optimization](https://arxiv.org/abs/2605.08954) | 提出分子世界模型 MolWorld，将可操作分子优化形式化为分子转移图的迭代扩展，通过匹配分子对（MMP）边显式建模可达性，确保优化得到的候选分子可从已知分子经局部结构修饰到达。 |
| [^355] | [Offline Policy Optimization with Posterior Sampling](https://arxiv.org/abs/2605.07393) | 该论文提出PSPO方法，通过将动力学模型建模为随机变量（后验采样）而非点估计，实现离线强化学习中对分布外区域的受控探索，从而在泛化能力与鲁棒性之间取得平衡。 |
| [^356] | [LensVLM: Selective Context Expansion for Compressed Visual Representation of Text](https://arxiv.org/abs/2605.07019) | LensVLM提出了一种推理框架和后训练方案，使VLM能先扫描压缩的渲染文本图像，再通过学习到的工具选择性地将相关图像扩展为未压缩形式，从而在4.3倍有效压缩下保持与全文处理相当的准确率，并在最高10.1倍压缩下超越现有基线。 |
| [^357] | [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) | RAO提出了一种强化学习方法，通过训练智能体递归地生成并委派子任务给自身的新实例来实现推理时的分治扩展，使模型能够突破上下文窗口限制、泛化到远难于训练任务的问题，并降低实际运行时间。 |
| [^358] | [CoMemNet: A Continual Memory Network with Drift-Aware Sampling for Traffic Prediction](https://arxiv.org/abs/2605.05738) | 提出 CoMemNet，一种通过在线/目标双分支、基于 Wasserstein 的漂移感知采样和节点自适应时间记忆重放缓冲，在演进的交通传感器网络上实现无需固定邻接矩阵与全量重训的高效持续交通预测模型。 |
| [^359] | [Dual Certified White-Box Inference for Input Convex Neural Networks](https://arxiv.org/abs/2605.04722) | 该论文提出利用SOC-ICNN与参数化二阶锥规划价值函数的精确对偶表示，开发双认证白盒推断方法DCI，从最优对偶乘子恢复完整次微分、提供精确平稳性认证与下降方向，并实现牛顿加速与全局收敛。 |
| [^360] | [Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks](https://arxiv.org/abs/2604.24955) | 提出BenchGuard——首个利用前沿大语言模型对基于执行的LLM智能体基准测试进行跨工件联合审计的框架，能够自动发现基准测试本身存在的缺陷（如损坏的任务规范和僵化的评估脚本）。 |
| [^361] | [How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks](https://arxiv.org/abs/2604.22750) | 本文首次系统研究了智能体编程任务中的token消耗模式，发现智能体任务消耗的token比代码推理和对话任务高出1000倍且以输入token为主要成本来源、使用量波动极大，并进一步评估了大模型在任务执行前预测自身token成本的能力。 |
| [^362] | [Rhetorical Questions in LLM Representations: A Linear Probing Study](https://arxiv.org/abs/2604.14128) | 该研究通过线性探针发现大语言模型在表示空间中能够早期且稳定地编码反问句信号，其跨数据集可迁移性虽然存在，但并不意味着模型内部存在统一的共享表示。 |
| [^363] | [Verify Before You Fix: Agentic Execution Grounding for Trustworthy Cross-Language Code Analysis](https://arxiv.org/abs/2604.10800) | 该论文的核心创新是提出一个由LLM驱动的跨语言漏洞生命周期框架，以“未经执行确认可利用性就不得修复”这一严格不变式为准则，将结构-语义混合检测、基于执行的智能体验证与感知验证的迭代修复三个阶段串联起来，并借助通用抽象语法树与 GraphSAGE、Qwen2.5-Coder 嵌入的混合融合实现 Java、Python、C++ 的跨语言泛化，从而保证代码分析与修复建立在可验证的证据之上。 |
| [^364] | [What do your logits know?](https://arxiv.org/abs/2604.09885) | 该论文首次系统比较了视觉-语言模型在不同表示层次（残差流、tuned lens投影、top-k logits）上保留的信息，发现即使是最易访问的top logit值也能泄露图像查询中与任务无关的信息，其泄露量在某些情况下与完整残差流的直接投影相当，揭示了模型内部信息泄露的安全风险。 |
| [^365] | [What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](https://arxiv.org/abs/2604.06995) | 提出UI-in-the-Loop（UILoop）范式，将GUI推理建模为“屏幕-UI元素-动作”的循环过程，使多模态大语言模型显式学习关键UI元素的定位、语义与用法，实现精确的元素发现和可解释推理，并贡献了包含26K样本的UI理解基准。 |
| [^366] | [Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity](https://arxiv.org/abs/2604.04692) | 该论文挑战了“视觉证据总能提升事实核查准确性”的普遍假设，提出通过两个协同的视觉-语言模型自适应判断是否需要视觉证据的模块化框架AMuFC，在多个数据集上实现了更有效的事实核查。 |
| [^367] | [Many Preferences, Few Policies: Compact Portfolios for Multi-Objective LLM Alignment](https://arxiv.org/abs/2604.04144) | 该论文提出 PALM 算法，通过结构化权重向量网格、惰性搜索与剪枝构建一个小型 LLM 策略组合，可证明地覆盖所有奖励权重下的近优对齐策略，以低成本实现多目标 LLM 对齐的个性化与部署。 |
| [^368] | [The Hitchhikers Guide to Rubric Quality Understanding and Enrichment](https://arxiv.org/abs/2604.01375) | 本文提出RIFT评分量规失败分类法及基于内容的量化信号，能以75%的准确率识别评分量规的失败模式（超过前沿大模型），并发现约20%的专家撰写量规存在权重反向的问题。 |
| [^369] | [GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification](https://arxiv.org/abs/2603.29112) | 该论文提出GISTBench基准，通过兴趣扎根度（IG）和兴趣特异性（IS）两个新指标，评估大语言模型从推荐系统交互历史中提取和验证用户兴趣的能力，突破了传统推荐系统基准仅关注物品预测准确率的局限。 |
| [^370] | [Spectral Alignment in Forward-Backward Representations via Temporal Abstraction](https://arxiv.org/abs/2603.20103) | 本文证明时间抽象如同低通滤波器，可抑制高频谱分量、降低后继表示的有效秩并保持价值函数误差界，从而缓解连续环境高秩转移动力学与FB低秩瓶颈之间的谱失配，是实现稳定前向-后向表示学习的关键因素。 |
| [^371] | [Exploring Subnetwork Interactions in Heterogeneous Brain Network via Prior-Informed Graph Learning](https://arxiv.org/abs/2603.19307) | 提出KD-Brain框架，通过语义条件化交互机制和病理一致性约束将语义与临床先验知识注入图学习过程，有效解决了小样本条件下脑功能子网络交互建模难题，实现精神障碍诊断的最先进性能。 |
| [^372] | [Controllable Accent Normalization via Discrete Diffusion](https://arxiv.org/abs/2603.14275) | 提出了基于掩码离散扩散的可控口音规范化系统DLM-AN，通过选择性复用共同令牌实现口音强度的灵活控制，并借助流匹配时长预测器匹配母语节奏，在多口音英语数据上取得最低词错误率。 |
| [^373] | [On the Tip of the Tongue: Why LLMs Hallucinate Answers They Can Decode](https://arxiv.org/abs/2603.13911) | 该论文提出在首个答案标记处区分“读取”与“写出”的新框架，揭示大语言模型产生幻觉的关键原因并非正确答案无法从中间状态解码，而是最终读出时的“选择边际”不足，使更强的竞争标记压制了正确答案。 |
| [^374] | [Evaluation format, not model capability, drives measured triage failure in the assessment of consumer health AI](https://arxiv.org/abs/2603.11413) | 该研究通过机制性实验与忠实复现证明，消费级健康AI分诊失败的高错误率主要源于考试式的评估格式（强制选项输出、禁止澄清提问），而非模型本身的能力不足——在自然的患者风格消息下，前沿大语言模型的分诊表现显著更好。 |
| [^375] | [Novelty Adaptation Through Hybrid Large Language Model (LLM)-Symbolic Planning and LLM-guided Reinforcement Learning](https://arxiv.org/abs/2603.11351) | 该论文提出了一种融合符号规划、强化学习与大语言模型的神经符号架构，利用LLM的常识推理能力识别缺失算子、生成计划并编写奖励函数，使机器人能够有效适应开放世界环境中的新颖物体。 |
| [^376] | [Sensory-Aware Sequential Recommendation via Review-Distilled Representations](https://arxiv.org/abs/2603.02709) | 该论文提出ASER离线流水线，通过微调大语言模型从评论中提取有据可查的感官属性并蒸馏为冻结的五维感官库，再以轻量级关系度量增强序列推荐，同时保持预训练主干不变。 |
| [^377] | [Goldilocks RL: Tuning Task Difficulty to Escape Sparse Rewards for Reasoning](https://arxiv.org/abs/2602.14868) | 提出Goldilocks自适应数据选择策略，利用选择器网络预测问题的奖励波动性，优先选取难度适中（既不太简单也不太难）的训练问题，从而摆脱稀疏奖励困境，提升语言模型推理强化学习的样本效率。 |
| [^378] | [The Effective Depth Paradox: Topology and Trainability in Deep CNNs](https://arxiv.org/abs/2602.13298) | 该论文提出“有效深度”（$D_{eff}$）这一闭式预训练代理指标，用于量化前向信息路径的期望长度，从而揭示了VGG、ResNet和GoogLeNet等不同拓扑结构中名义深度与实际可训练性之间的“有效深度悖论”。 |
| [^379] | [ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling](https://arxiv.org/abs/2602.09009) | 该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。 |
| [^380] | [Extended to Reality: Prompt Injection in 3D Environments](https://arxiv.org/abs/2602.07104) | 本文提出PI3D，一种通过物理放置带文本3D对象而非数字编辑来攻击MLLMs的提示注入方法，并系统化解决攻击对象姿态优化问题。 |
| [^381] | [LPS-Bench: Benchmarking Safety Awareness of Computer-Use Agents in Long-Horizon Planning under Benign and Adversarial Scenarios](https://arxiv.org/abs/2602.03255) | 该论文提出LPS-Bench基准，通过模板引导的多智能体流水线高效生成570个覆盖7个任务领域和9种规划风险类型的测试案例，用以评测计算机使用智能体在良性请求与对抗性引导下的长程规划安全意识。 |
| [^382] | [IntentCoding: Amplifying User Intent in Code Generation](https://arxiv.org/abs/2602.00066) | 提出IntentCoding解码策略，通过屏蔽意图来捕捉用户意图的影响，并利用多强度集成机制放大该影响，无需额外训练即可显著提升大语言模型在多约束代码生成任务中对用户意图的遵循能力。 |
| [^383] | [Recoverability Has a Law: The ERR Measure for Tool-Augmented Agents](https://arxiv.org/abs/2601.22352) | 本文提出期望恢复遗憾（ERR）指标并证明其与可观测的效率得分（ES）之间存在一阶定量关系，首次为工具增强语言模型智能体失败后的自我恢复能力建立了可证伪的预测性定律，并在五个工具使用基准上得到实证验证。 |
| [^384] | [AstroAgentBench: Evaluating Agentic Planning on Space Mission Planning Tasks](https://arxiv.org/abs/2601.11354) | 本文提出AstroAgentBench——一个涵盖调度、观测规划、星座设计和中继支持等七大任务族的可执行太空任务规划基准，通过外部验证器评估智能体生成的规划产物，发现最强LLM智能体系统在部分任务上可接近或超越求解器参考水平，而较弱系统则难以产出高价值的有效规划。 |
| [^385] | [Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models](https://arxiv.org/abs/2512.21439) | 提出了COMETH框架，将概率情境学习与大语言模型语义抽象及人类道德判断数据相结合，从数据中学习可解释的道德情境，证明道德评价是高度情境化的。 |
| [^386] | [Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling](https://arxiv.org/abs/2512.19905) | 该论文提出了一个解析可处理的推理时扩展模型——带奖励加权采样器的贝叶斯线性回归，用以模拟LLM作为评判者的场景，并在高维机制下推导出后验预测均值与方差的闭式表达式，从而揭示推理时扩展背后的数学原理。 |
| [^387] | [Science Is Falling Behind the Frontier: Foundation Model Adoption Across Half a Million Papers](https://arxiv.org/abs/2511.21739) | 该研究首次对50万篇论文中的AI基础模型采用情况进行大规模分析，发现科学界采用的模型规模已从2015年领先前沿模型5.4倍逆转为2024年落后6.9倍，这种“规模滞后”可能正在限制科学家充分获取AI赋能科学的收益。 |
| [^388] | [A Unified BERT-CNN-BiLSTM Framework for Simultaneous Headline Classification and Sentiment Analysis of Bangla News](https://arxiv.org/abs/2511.18618) | 本文提出了一个统一的BERT-CNN-BiLSTM混合迁移学习框架，首次实现了孟加拉语新闻标题分类与情感分析的同步处理。 |
| [^389] | [LLM-Guided Reinforcement Learning with Representative Agents for Traffic Modeling](https://arxiv.org/abs/2511.06260) | 提出用单个代表性LLM智能体建模同质出行者群体，将LLM的正向强化判断通过可解释规则转化为混合策略更新，从而实现可扩展且稳定的逐日交通流建模。 |
| [^390] | [Cocoon: A System Architecture for Differentially Private Training with Correlated Noises](https://arxiv.org/abs/2510.07304) | Cocoon 提出了一种系统架构，通过在 CPU、GPU 和内存扩展模块之间分布式地存储与处理庞大的相关噪声历史，并对稀疏嵌入表进行优化，实现了高效的大规模模型差分隐私训练。 |
| [^391] | [EEGDM: Learning EEG Representation with Latent Diffusion Model](https://arxiv.org/abs/2508.20705) | EEGDM提出了一种基于潜在扩散模型的自监督学习框架，通过生成式去噪过程学习脑电信号的全局时间模式与跨通道关系的紧凑表征，克服了掩码重建方法难以捕捉全局生成约束的局限。 |
| [^392] | [Multimodal Representation Learning Conditioned on Semantic Relations](https://arxiv.org/abs/2508.17497) | 提出了关系条件化多模态学习框架RCML，将自然语言描述的语义关系作为显式条件来学习多模态表示，使同一样本在不同关系下拥有不同表示，克服了CLIP等对比模型单一嵌入的局限。 |
| [^393] | [Mitigating Watermark Forgery in Generative Models via Randomized Key Selection](https://arxiv.org/abs/2507.07871) | 该论文提出通过对每次查询随机化水印密钥选择的防御方案，使盲攻击者的伪造成功率存在与所收集样本数量无关的上限，且不进一步降低模型效用，从而有效缓解生成模型中的水印伪造攻击。 |
| [^394] | [Robust Adversarial Quantification via Conflict-Aware Evidential Deep Learning](https://arxiv.org/abs/2506.05937) | 提出轻量级后验不确定性量化方法 C-EDL，通过为输入生成多样的任务保持变换并量化表示分歧来校准不确定性，无需重新训练即可增强证据深度学习对对抗性和分布外输入的鲁棒性。 |
| [^395] | [Evaluating the Retrieval Robustness of Large Language Models](https://arxiv.org/abs/2505.21870) | 该研究建立了一个包含1,891个样本的基准和三个鲁棒性指标，系统评估了11个大型语言模型在检索增强生成场景中的检索鲁棒性，重点考察RAG是否总是优于非RAG、更多检索文档是否总是有益以及文档顺序对结果的影响。 |
| [^396] | [VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation](https://arxiv.org/abs/2505.13439) | VTBench是一个系统性评估自回归图像生成中视觉分词器性能的综合基准，通过图像重建、细节保留和文本保留三大核心任务，揭示了离散视觉分词器与连续VAE之间的性能差距。 |
| [^397] | [Hybrid Reasoning Systems That Prioritize and Enhance Human Intelligence](https://arxiv.org/abs/2504.13477) | 本文提出了一个以人为中心的混合推理系统框架，通过融合增强人类推理的既定策略、重视结论前互动的AI设计方法以及将推理分解为可单独支持的模式，实现从数据分析到高级智慧的全方位人类推理能力增强。 |
| [^398] | [Towards Unified Music Emotion Recognition across Dimensional and Categorical Models](https://arxiv.org/abs/2502.03979) | 本文提出了一个融合类别与维度两种情感标签的统一多任务学习框架，通过结合音乐特征与MERT嵌入表示，并利用知识蒸馏将单数据集教师模型的知识迁移到学生模型，实现了跨多个数据集的音乐情感识别。 |
| [^399] | [Heads, Tails, and AI Fails: LLMs, Randomness, and Human Judgments](https://arxiv.org/abs/2406.00092) | 该研究发现大语言模型在模拟抛硬币时会再现并放大人类的随机性偏差（如过度交替、厌恶长连续序列），提高温度参数只能部分缓解而无法消除这些系统性失真。 |
| [^400] | [VIDiff: Translating Videos via Multi-Modal Instructions with Diffusion Models](https://arxiv.org/abs/2311.18837) | 本文首次提出了视频指令扩散基础模型VIDiff，能够根据用户的多模态指令在几秒内完成视频编辑、转换和增强等多种理解与生成任务，并通过迭代自回归方法保证长视频编辑的一致性。 |
| [^401] | [Learning Low-Frequency Motion Control for Robust and Dynamic Robot Locomotion](https://arxiv.org/abs/2209.14887) | 本文挑战了“提高控制频率以增强鲁棒性”的传统观念，证明基于强化学习的低频（低至8Hz）运动控制器可在真实四足机器人上实现鲁棒动态运动，且低频策略对执行延迟和动力学变化更不敏感，甚至无需动力学随机化即可完成仿真到现实的迁移。 |
| [^402] | [ETHER: Aligning Emergent Communication for Hindsight Experience Replay.](http://arxiv.org/abs/2307.15494) | 本文提出了ETHER，通过对齐紧急沟通来解决回顾性经验重演中的问题，克服了先前架构依赖预设函数的限制，并提高了数据效率和性能。 |

# 详细

[^1]: 更少的解码器即更多的编码器：从新视角合成中学习几何表示

    Less Decoder is More Encoder: Geometric Representation Learning from Novel View Synthesis

    [https://arxiv.org/abs/2610.03717](https://arxiv.org/abs/2610.03717)

    本文提出SNAP，一个自监督编码器-解码器Transformer，通过位姿条件化的局部解码器与潜空间重建目标，克服了新视角合成中解码器削弱编码器表示能力、像素空间目标阻碍特征学习的问题，实现了可与几何监督方法媲美的任务无关几何表示。

    

    本文研究了新视角合成（NVS）在几何表示学习中的作用。原则上，NVS应当对3D场景结构进行推理，从而实现可迁移的多视角几何表示。然而，现有的基于编码器的NVS方法所产生的表示质量较差。这并不是因为缺乏监督信号，而是由于一些不起眼的架构选择所致：空间表达能力过强的解码器稀释了场景编码器的表示能力，而低层的像素空间重建目标则阻碍了特征学习。我们提出了SNAP，一个自监督的编码器-解码器Transformer，它通过位姿条件化的局部解码器和潜空间重建目标同时解决了这两个问题。SNAP是任务无关的，我们证明它可以与专门的几何监督方法相媲美。SNAP在五项任务上也与自监督表示相比表现具有竞争力：视觉……（摘要截断）

    arXiv:2610.03717v1 Announce Type: cross  Abstract: This paper examines the role of Novel View Synthesis (NVS) in geometric representation learning. In principle, NVS should reason about 3D scene structure, thereby enabling transferable multi-view geometric representations. Yet, existing encoder-based NVS methods yield poor representations. This is not because of a lack of supervisory signal, but rather due to inconspicuous architectural choices: \textit{spatially expressive decoders} that dilute representational capabilities of the scene encoder, and \textit{low-level pixel-space targets} that hinder feature learning. We present SNAP, a self-supervised encoder-decoder transformer that addresses both through a pose-conditioned local decoder and a latent-space reconstruction objective. SNAP is task agnostic, and we show that it is competitive with special-purpose geometry-supervised methods. SNAP also performs competitively against self-supervised representations across five tasks: visua
    
[^2]: 4DCodeBench：基于动态场景逆向图形学的智能体基准测试

    4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes

    [https://arxiv.org/abs/2610.03715](https://arxiv.org/abs/2610.03715)

    4DCodeBench是一个通过代码生成来评估智能体4D逆向图形学能力的基准，测试发现前沿模型虽具备较强的静态场景重建能力，但在重建变形、流体、断裂等复杂动态场景方面仍不可靠。

    

    我们提出了4DCodeBench，一个通过代码生成进行4D逆向图形学的基准测试，其中智能体需要将视频中的动态场景重建为可执行的图形程序。为实现这一目标，智能体必须将视觉观察转化为场景结构和动态的紧凑表示，并通过实现物理模拟等抽象来重现复杂行为。为评估这一能力，我们精选了一组真实世界视频，并构建了涵盖多种物理现象（包括变形、流体流动和断裂）的合成场景。我们对前沿模型进行了广泛的基准测试，发现强大的静态重建能力尚不能转化为对复杂动态的可靠重建。4DCodeBench为追踪能够通过代码解读世界动态的智能体的发展进程提供了一个测试平台。我们的基准测试可在 https://github.com/4DCodeBench/4DCodeBench 获取。

    arXiv:2610.03715v1 Announce Type: cross  Abstract: We introduce 4DCodeBench, a benchmark for 4D inverse graphics through code generation, in which agents reconstruct dynamic scenes from video as executable graphics programs. To accomplish this, agents must translate visual observations into compact representations of scene structure and dynamics, by implementing abstractions such as physical simulations to reproduce complex behavior. To evaluate this capability, we curate a set of real-world videos and construct synthetic scenes spanning diverse physical phenomena, including deformation, fluid flow, and fracture. We perform extensive benchmarking of frontier models, finding that strong static reconstruction capabilities do not yet translate into reliable reconstruction of complex dynamics. 4DCodeBench provides a testbed for tracking progress toward agents that can interpret the dynamics of the world through code. Our benchmark is available at https://github.com/4DCodeBench/4DCodeBench
    
[^3]: 世界模型应该遗忘什么？面向持续适应的分层保留机制

    What Should World Models Forget? Stratified Retention for Continual Adaptation

    [https://arxiv.org/abs/2610.03713](https://arxiv.org/abs/2610.03713)

    提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。

    

    持续学习将在已见数据上的性能退化视为失败的证据，这一惯例继承自预测目标平稳的设定，即正确标签会永远保持正确。世界模型不满足这一条件：它们的预测目标是环境，而环境会变化，因此获取时准确的知识之后可能变为错误，丢弃这些知识是必要行为而非缺陷。非平稳的真实标签问题在概念漂移文献和语言模型的时间事实性研究中已被充分探讨，但尚未在世界模型中被形式化，而世界模型的独特之处在于它们还编码了绝不能被修改的知识。我们认为，持续世界模型需要按不变性时间尺度分层的保留机制，将诸如物理规律和物体恒存性等绝不应被修改的不变量，与应随环境变化而尽快被更新的实例级事实区分开来。

    arXiv:2610.03713v1 Announce Type: cross  Abstract: Continual learning treats degradation on previously seen data as evidence of failure, a convention inherited from settings with a stationary prediction target, where a correct label remains correct indefinitely. World models do not satisfy this condition. Their prediction target is the environment, which changes, so knowledge that was accurate when acquired may later become false, and discarding it is required behavior rather than a defect. Non-stationary ground truth is well studied in the concept drift literature and in the temporal factuality of language models, but has not been formulated for world models, which are distinctive in that they also encode knowledge that must never be revised. We argue that continual world models require retention stratified by invariance timescale, separating invariants such as physics and object permanence, which must never be revised, from instance-level facts that should be revised as soon as the e
    
[^4]: EyeRobot 2.0：无需腕部相机的主动注视实现精准操作

    EyeRobot 2.0: Active Gaze for Precise Manipulation without Wrist Cameras

    [https://arxiv.org/abs/2610.03710](https://arxiv.org/abs/2610.03710)

    EyeRobot 2.0通过主动视觉注视（双眼转动对准3D目标点并进行中央凹式token分配）与分层强化学习训练（底层注视伺服策略加上层注视目标选择器），仅用单个立体相机就实现了无需腕部相机的精细双手机器人操作。

    

    受人类视觉启发，我们提出了一个利用主动注视的框架，仅凭单个立体相机即可实现精细的双手机器人操作。EyeRobot 2.0通过转动两个“眼睛”视点，将注视中心对准场景中的3D注视点，从而在物理上关注该点。所得到的图像采用中央凹（foveal）方式处理，即为图像中心分配更多视觉token，使计算聚焦于任务相关的特征。这种主动视觉注视（AVF）要求在任务执行过程中进行精细协调的注视控制，我们通过分层方式实现：首先训练一个以目标物体为条件的底层注视伺服策略，然后训练一个基于任务进度发出注视目标的目标准则选择器。两个模块均在真实世界数据上通过强化学习进行训练：第一个使用密集几何奖励进行训练，第二个则与BC夹爪策略共同训练，从而使其能够发现类似于人类注视模式的注视序列。

    arXiv:2610.03710v1 Announce Type: cross  Abstract: Inspired by human vision, we introduce a framework using active gaze to enable fine-grained bimanual manipulation with only a single stereo camera. EyeRobot 2.0 physically attends to a 3D fixation point in the scene by swiveling two eye viewpoints to center their gaze on it. The resulting images are processed foveally by allocating more visual tokens to the image centers, focusing computation on task-relevant features. Such Active Visual Fixation (AVF) requires carefully coordinated gaze during task execution, which we accomplish hierarchically by first training a low-level gaze servoing policy conditioned on a goal object, then training a target selector which emits fixation goals based on task progress. Both modules are trained with RL on real-world data: the first is trained with a dense geometric reward and the second co-trains with the BC gripper policy which allows it to discover fixation sequences that can resemble a human's fix
    
[^5]: 基于转录组信息的多模态人工智能：从乳腺癌活检预测新辅助治疗反应

    Transcriptome-informed multi-modal AI for predicting neoadjuvant therapy response from breast cancer biopsies

    [https://arxiv.org/abs/2610.03693](https://arxiv.org/abs/2610.03693)

    该研究提出一种两阶段多模态AI模型，先从病理图像推断转录组表达谱，再结合临床变量预测乳腺癌新辅助治疗的病理完全缓解，在九个队列中实现0.79的合并AUROC，并优于传统组织病理学生物标志物。

    

    标注数据的稀缺限制了肿瘤学中深度学习生物标志物的开发。我们开发了一个两阶段AI模型，用于预测乳腺癌新辅助治疗的病理完全缓解。第一阶段利用涵盖32种癌症类型的8,742名患者从组织病理学图像中学习转录组，并通过病理学家审查以及与实测表达的空间一致性加以验证。这使得第二阶段简化为从推断的表达谱和临床变量预测病理完全缓解。该模型基于1,080名患者（五个队列）开发，并在1,412名患者（九个队列）中进行评估，取得了0.79的合并AUROC（95% CI，0.73-0.85），能够在分子亚型内部区分治疗应答者。该模型优于组织病理学生物标志物，在不同肿瘤内取样条件下保持稳定，且所需活检组织极少。消融实验表明，全转录组推断相比仅使用临床变量或单阶段病理学模型能提升判别能力。

    arXiv:2610.03693v1 Announce Type: new  Abstract: Scarcity of labeled data limits development of deep learning biomarkers in oncology. We develop a two-stage AI model predicting pathological complete response (pCR) to neoadjuvant therapy in breast cancer. The first stage learns the transcriptome from histopathology using 8,742 patients across 32 cancer types, corroborated by pathologist review and spatial agreement with measured expression. This simplifies the second stage to predicting pCR from inferred expression and clinical variables. Developed using 1,080 patients (five cohorts) and evaluated in 1,412 patients (nine cohorts), the model achieves a pooled AUROC of 0.79 (95% CI, 0.73-0.85), discriminating responders within molecular subtypes. It outperforms histopathological biomarkers, remaining stable across intratumoral sampling and with minimal biopsy tissue. Ablations show transcriptome-wide inference improves discrimination over clinical variables alone or one-stage pathology mo
    
[^6]: FrugalEvo：迈向成本感知的LLM引导程序进化

    FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution

    [https://arxiv.org/abs/2610.03675](https://arxiv.org/abs/2610.03675)

    该论文提出成本感知的LLM进化框架FrugalEvo，让更强的LLM探索解法策略、更廉价的LLM负责代码实现与迭代优化，并通过前缀共享提升缓存复用，同时引入BA-AUC指标来衡量单位成本下的优化收益。

    

    arXiv:2610.03675v1 公告类型：cross 摘要：以AlphaEvolve为代表的LLM引导的进化方法，已成为解决具有挑战性的计算优化问题（如圆填充问题）的有力工具。然而，先前的工作通常是在固定迭代次数下优化性能提升。我们认为，实际的优化应当最大化单位成本的收益。为此，我们提出了FrugalEvo——一个成本感知的进化框架，其中由一个更强、成本更高的LLM负责探索解决方案策略，由一个更廉价的LLM负责实现这些策略并对生成的代码进行迭代改进。我们还设计了缓存高效的进化过程，通过我们的测试框架和提示词设计，最大化不同进化步骤之间的前缀共享，以提升缓存复用率。为了在固定成本预算内衡量解决方案的质量，我们引入了预算感知曲线下面积（BA-AUC），其定义为在预算范围内、以累计LLM成本为横轴的最优评估分数曲线下的面积。（原文摘要至此处截断）

    arXiv:2610.03675v1 Announce Type: cross  Abstract: LLM-guided evolutionary methods, such as AlphaEvolve, have emerged as powerful approaches for challenging computational optimization problems, such as circle packing. However, prior work typically optimizes performance gain over a fixed number of iterations. We argue that practical optimization should maximize gain per unit cost. To this end, we propose FrugalEvo, a cost-aware evolutionary framework where a stronger, higher-cost LLM explores solution strategies, and a cheaper LLM implements them and iteratively refines the resulting code. We also design a cache-efficient evolution process, where our harness and prompts maximize the sharing of prefixes across different evolution steps, to improve cache reuse. To measure solution quality throughout a fixed cost budget, we introduce Budget-Aware Area Under the Curve (BA-AUC), defined as the area under the best-so-far evaluation score curve over cumulative LLM cost, up to the budget. Acros
    
[^7]: 重新审视人声合唱多音高估计中的输入时频表示

    Revisiting Input Time-frequency Representations in Multi-pitch Estimation for Vocal Ensembles

    [https://arxiv.org/abs/2610.03656](https://arxiv.org/abs/2610.03656)

    该论文发现，在人声合唱多音高估计任务中，结构简单的线性STFT输入表示在性能上优于广泛使用的HCQT表示，同时显著降低了特征提取的计算成本。

    

    人声合唱中的多音高估计极具挑战性，因为歌手们占据相互重叠的音高范围，且常常以间距极近的基频演唱，导致其谐波在时频表示中相互重叠。现有模型通常采用基于谐波常数Q变换（HCQT）的表示来提供频率自适应的分辨率，但在即时生成训练混合音频时，其代价是高昂的特征提取开销。我们重新审视了这一设计，将HCQT与线性短时傅里叶变换（STFT）进行对比，后者的频率bin直接作为模型输入提供。尽管线性STFT具有固定的频率分辨率且缺少音高对齐的输入网格，其性能仍优于HCQT，同时大幅降低了特征提取成本。进一步分析表明，更长的分析窗口或更宽的频谱覆盖范围并不会带来额外提升，而将输入限制在预测音高

    arXiv:2610.03656v1 Announce Type: cross  Abstract: Multi-pitch estimation in vocal ensembles is challenging because singers occupy overlapping pitch ranges and often sing at closely spaced fundamental frequencies, causing their harmonics to overlap in time-frequency representations. Existing models commonly use harmonic constant-Q transform (HCQT)-based representations to provide frequency-adaptive resolution, at the cost of expensive feature extraction when training mixtures are generated on the fly. We revisit this design and compare HCQT with a linear short-time Fourier transform (STFT), whose frequency bins are directly provided as model inputs. Despite its fixed frequency resolution and the absence of a pitch-aligned input grid, the linear STFT outperforms HCQT while substantially reducing feature-extraction cost. Further analysis shows that a longer analysis window or broader spectral coverage provides no additional improvement, while restricting the input to the predicted pitch 
    
[^8]: MRVQ：面向维度与码率弹性向量检索的单一常驻索引

    MRVQ: One Resident Index for Dimension- and Rate-Elastic Vector Search

    [https://arxiv.org/abs/2610.03651](https://arxiv.org/abs/2610.03651)

    提出 MRVQ（套娃残差向量量化），用单一常驻索引即可通过截断残差阶段降低码率、截断嵌入坐标降低维度，以比多索引方案低 1.89-22 倍的内存覆盖所有维度-码率组合的向量检索。

    

    密集检索服务需要随着延迟、质量和内存预算的变化，在嵌入前缀维度和索引比特率之间进行切换。为每种码率单独调优一个量化器可以获得最佳质量，但此时检索层必须同时持有多个码流和量化器状态。我们提出了套娃残差向量量化，这是一种针对冻结嵌入的事后残差量化器。其最高码率的编码可以通过两种方式进行截断：丢弃残差阶段以降低码率，丢弃嵌入坐标以降低维度。因此，一个常驻的索引产物即可服务我们评估的每一个（维度， 码率）组合。在 FiQA 和 NFCorpus 数据集、四种嵌入家族以及 {4, 8, 16} 字节编码的设置下，MRVQ 是我们评估的所有设计中内存占用最低的。它比三个单独训练的 QINCo2 索引少占用 17.8-22.0 倍内存，比精简的共享模型最强基线少占用 1.89-2.02 倍。这种节省并非没有代价：按码率单独训练的 QINCo2 在 nDCG@10 上高出 0.026-0.107

    arXiv:2610.03651v1 Announce Type: new  Abstract: Dense-retrieval services must switch among embedding-prefix dimensions and index bit rates as latency, quality, and memory budgets change. Tuning a quantizer separately for each rate gives the best quality, but the retrieval tier then holds several code streams and quantizer states at once. We introduce Matryoshka Residual Vector Quantization (MRVQ), a post-hoc residual quantizer for frozen embeddings. Its maximum-rate code can be truncated two ways: dropping residual stages lowers the rate, and dropping embedding coordinates lowers the dimension. One resident artifact therefore serves every (dimension, rate) pair we evaluate. Across FiQA and NFCorpus, four embedding families, and {4, 8, 16}-byte codes, MRVQ is the lowest-RAM design we evaluate. It uses 17.8-22.0x less memory than three separately trained QINCo2 indices, and 1.89-2.02x less than a lean shared-model steelman. The saving is not free: per-rate QINCo2 is 0.026-0.107 nDCG@10 
    
[^9]: 面向高效海洋环境监测的星载异常检测

    On-Board Anomaly Detection for Efficient Marine Environmental Monitoring

    [https://arxiv.org/abs/2610.03649](https://arxiv.org/abs/2610.03649)

    该论文提出了一种用于对地观测卫星的轻量级海洋环境异常检测流程，利用自监督神经网络编码器压缩卫星图像并结合机器学习异常检测模型，实现高效的星载海洋环境监测。

    

    海洋生态系统受到石油泄漏、藻类大量繁殖和泥沙洪水等多种威胁的影响，这些威胁扰乱了栖息地、野生动物和人类活动。卫星图像和人工智能（AI）技术的进步增强了我们对此类危害进行早期检测和缓解的能力。在本文中，我们提出了一种面向配备多光谱或高光谱传感器的对地观测卫星的海洋事件检测流程。我们的方法包括一个自监督神经网络编码器，该编码器将卫星图像压缩到降维的潜在空间中，从而实现高效的星载处理。一个机器学习异常检测模型通过识别与正常海洋模式的偏差来检测环境异常。我们将其性能与孤立森林、单类支持向量机和局部离群因子等传统算法进行了比较。我们的轻量级、资源高效的流程针对星载部署进行了优化。

    arXiv:2610.03649v1 Announce Type: cross  Abstract: Marine ecosystems are impacted by various threats such as oil spills, algal blooms, and sediment floods, which disrupt habitats, wildlife, and human activities. Advances in satellite imagery and Artificial Intelligence (AI) have enhanced our capabilities for early detection and mitigation of such hazards. In this paper, we propose a marine event detection pipeline for Earth observation satellites equipped with multi- or hyperspectral sensors. Our approach includes a self-supervised neural network encoder that compresses satellite images into a reduced latent space, enabling efficient onboard processing. A machine learning anomaly detection model identifies deviations from normal sea patterns to detect environmental anomalies. We compare its performance against traditional algorithms such as Isolation Forest, One-Class Support Vector Machine and Local Outlier Factors. Our lightweight, resource-efficient pipeline is optimized for deploym
    
[^10]: 大型语言模型了解哥伦比亚法律吗？面向哥伦比亚法律体系的可靠性基准

    Do Large Language Models Know Colombian Law? A Reliability Benchmark for the Colombian Legal System

    [https://arxiv.org/abs/2610.03639](https://arxiv.org/abs/2610.03639)

    该论文构建了一个包含1,042个条目、经专家验证的哥伦比亚法律基准，评估发现尽管LLM在封闭式选择题上准确率最高可达0.905，但在自由文本法律问答中事实正确性均不超过0.45，且答案相关性与正确性呈负相关，表明模型回答听起来切题却常常事实错误。

    

    大型语言模型（LLM）正日益被用于支持法律实践、教育和研究，但它们在美国以外的国家法律体系中的可靠性在很大程度上仍未被记录。我们引入了一个经专家验证的基准，用于评估LLM在哥伦比亚法律体系上的可靠性。该基准包含1,042个条目，涵盖十个法律领域和三种问题格式（封闭式多项选择、半开放式和开放式IRAC），通过人机协同（human-in-the-loop）流程并经过多阶段专家审查构建而成。我们采用适合各格式的评估指标对15个当代专有和开放权重模型进行了评估。封闭式问题的准确率差异很大，从0.905（Gemini 3.1 Pro）到0.577不等，但在自由文本法律答案上，任何模型的事实正确性都不超过0.45（0-1量表）。我们发现答案相关性与正确性之间存在分离（Spearman rho = -0.46）：模型的回答听起来总是切题的，但经常……（原文在此处截断）

    arXiv:2610.03639v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to support legal practice, education, and research, yet their reliability in national legal systems outside the United States remains largely undocumented. We introduce an expert-validated benchmark for evaluating LLM reliability on the Colombian legal system. The benchmark comprises 1,042 items spanning ten areas of law and three question formats (closed multiple-choice, semi-open, and open-ended IRAC), built through a human-in-the-loop pipeline with multi-stage expert review. We evaluate 15 contemporary proprietary and open-weight models with format-appropriate metrics. Accuracy on closed questions ranges widely, from 0.905 (Gemini 3.1 Pro) to 0.577, but on free-text legal answers factual correctness never exceeds 0.45 (on a 0-1 scale) for any model. We find a dissociation between answer relevancy and correctness (Spearman rho = -0.46): models reliably sound responsive while frequently
    
[^11]: LoGo：基于局部-全局奖励的一致长时程视频生成

    LoGo: Local-Global Rewards for Consistent Long-Horizon Video Generation

    [https://arxiv.org/abs/2610.03636](https://arxiv.org/abs/2610.03636)

    LoGo通过在后训练中融合全局奖励与空间局部化奖励，显著提升了相机控制长时程视频生成的3D一致性，同时保持了相机跟随精度和视频质量。

    

    相机控制的视频模型正快速向长生成时域和复杂相机控制方向迈进。一个关键的失败模式是3D不一致性：随着相机移动，物体失去持久性，场景结构发生偏移。现有的后训练技术为整个生成结果只赋予单一标量奖励，难以在长时域中纠正这些不一致性。我们提出了LoGo，它为相机控制的视频模型融合了全局奖励与空间局部化奖励。局部奖励提供细粒度的信用分配，显著提升了3D一致性，而全局奖励则保持了相机跟随能力和视频质量。在三个基础模型上，LoGo在DL3DV和TrajectoryBench上展现出明显优势，其中TrajectoryBench是一个针对长时域、复杂相机控制生成的新基准，弥补了现有评估方法的不足。LoGo有效减少了局部物体偏移、伪影以及全局场景变化。

    arXiv:2610.03636v1 Announce Type: cross  Abstract: Camera-controlled video models are rapidly advancing toward long generation horizons and complex camera control. A key failure mode is 3D inconsistency: as the camera moves, objects lose permanence and scene structures shift. Existing post-training techniques, which assign a single scalar reward to the entire generation, are poorly suited to correcting these inconsistencies over long horizons. We introduce LoGo, which blends global and spatially localized rewards for camera-controlled video models. The local reward provides fine-grained credit assignment, which substantially improves 3D consistency, while the global reward preserves camera following and video quality. Across three base models, LoGo shows a clear advantage on DL3DV and TrajectoryBench, a new benchmark for long-horizon, complex-camera-control generation that current evaluations lack. LoGo effectively reduces local object shifts, artifacts, and global scene changes, illus
    
[^12]: 功劳归于关键之处：面向终端智能体的依赖感知策略优化

    Credit Where It Matters: Dependency-Aware Policy Optimization for Terminal Agents

    [https://arxiv.org/abs/2610.03634](https://arxiv.org/abs/2610.03634)

    提出依赖感知的组策略优化，通过从执行轨迹构建命令依赖图并从任务验证器检查的资源反向追踪，将信用精准分配给相关的写入操作及其支撑读取操作，从而改进终端智能体强化学习中的信用分配。

    

    使用终端的智能体在编程、调试以及其他多步骤终端任务中受益于强化学习（RL）。在这些任务中，后续命令往往依赖于先前命令产生的信息或中间结果。然而，现有的轨迹级和步骤级信用分配方法并未显式追踪命令通过读写依赖关系影响最终结果的路径。因此，训练信号仍可能被分配给无关的操作，从而削弱了从相关步骤中学习的效果。在本文中，我们提出了依赖感知组策略优化，该方法利用命令之间的执行依赖关系来指导终端智能体的信用分配。具体而言，我们根据执行轨迹构建命令依赖图，并从任务验证器所检查的资源出发进行反向追踪。然后，我们沿着这些路径为相关写入操作及其支撑的读取操作分配信用，并利用……

    arXiv:2610.03634v1 Announce Type: new  Abstract: Terminal-using agents benefit from reinforcement learning (RL) in coding, debugging, and other multi-step terminal tasks. In these tasks, later commands often depend on information or intermediate results produced by earlier commands. However, existing trajectory-level and step-level credit assignment methods do not explicitly trace the read-write dependencies through which commands affect the final outcome. Consequently, training signals could still be assigned to irrelevant operations, weakening learning from relevant steps. In this paper, we propose Dependency-Aware Group Policy Optimization (DepGPO), which uses execution dependencies between commands to guide credit assignment for terminal agents. Specifically, we construct a command dependency graph from execution traces and trace backward from the resources inspected by the task verifier. We then assign credit to relevant writes and their supporting reads along these paths, and use
    
[^13]: NeutronGym：面向大语言模型智能体的物理分级中子仪器设计

    NeutronGym: Physics-Graded Neutron Instrument Design for LLM Agents

    [https://arxiv.org/abs/2610.03631](https://arxiv.org/abs/2610.03631)

    提出首个中子仪器设计可执行环境NeutronGym，通过McStas仿真和无需LLM评审的分层自动评分来检验大语言模型智能体的真实物理设计能力，现有模型最多仅复现16个基准任务中的7个，而强化学习可将Qwen3-8B的通过率从11%提升至77%。

    

    设计科学仪器检验的是语言模型智能体能否“做物理”而非仅仅复述物理——前提是评分必须无可争议。我们提出了NeutronGym，据我们所知这是首个用于中子仪器设计的可执行环境：智能体通过验证工具构建仪器，McStas对所构建的仪器进行光线追踪仿真，并由一个分层级联的阶梯对语法、运行时、结构和科学性进行分级评分，全程无需大语言模型评审。程序化生成的仪器族提供无限个固定布局的实例，智能体必须自行设置其中的设计参数，并设有留出的参数区间；一个精选子集McStasBench则增加了来自已发表仪器的16个任务，并配有记忆探测和沙盒环境加以防护。七个模型最多复现了16个任务中的7个，没有一个能检索出参考答案，也没有一个达到改进目标。该环境还可用于训练：基于其奖励的强化学习使Qwen3-8B在某仪器族留出实例上的表现从11%提升至77%。

    arXiv:2610.03631v1 Announce Type: new  Abstract: Designing a scientific instrument tests whether language-model agents can do physics rather than recall it, provided the grading cannot be argued with. We introduce NeutronGym, to our knowledge the first executable environment for neutron instrument design: agents build instruments through validating tools, McStas ray-traces what they build, and a level-resolved ladder grades syntax, runtime, structure and science with no LLM judge. Procedural families supply unlimited instances of a fixed layout whose design parameters the agent must set, with held-out parameter regimes; a curated slice, McStasBench, adds 16 tasks from published instruments behind memorization probes and a sandbox. Seven models reproduce at most 7 of the 16, none retrieves a reference, and none meets an improvement target. The environment also trains. Reinforcement learning on its reward takes Qwen3-8B from 11% to 77% of held-out instances of a family whose targets come
    
[^14]: 一步生成模型中的深度即时间

    Depth as Time in One-Step Generative Models

    [https://arxiv.org/abs/2610.03626](https://arxiv.org/abs/2610.03626)

    该研究发现多步扩散的去噪计算会在一步生成模型单次前向传播的网络深度中展开，且这种深度方向上的计算取决于流映射所训练的传输任务。

    

    近期涌现的一步生成模型通过蒸馏或学习流映射的方式压缩了扩散模型的多步轨迹，已达到能够生成高质量图像的转折点。在此，我们提出了随之而来的一个自然问题：当生成过程被压缩为单次前向传播时，多步扩散的去噪轨迹会发生什么？我们提出了一个称为“深度即时间”的实证观察：多步扩散在采样步骤中执行的去噪计算，似乎会在单次前向传播的网络深度中展开，并且可以通过用模型自身的输出头解码中间层来恢复。最有趣的是，我们证明了这种沿深度的计算依赖于流映射被训练去解决的传输任务。最令人惊讶的案例是MeanFlow，其中通过探测更短的传输间隔，同时揭示了去噪和再噪的现象……

    arXiv:2610.03626v1 Announce Type: new  Abstract: The recent wave of one-step generative models, which compress the multi-step trajectory of diffusion via either distillation or learned flow maps, has reached an inflection point where they can generate high-quality images. Here, we ask a natural question that follows from these advances: what happens to the denoising trajectory of multi-step diffusion when generation is compressed into a single forward pass? We offer an empirical observation we call \textit{depth as time}: the denoising computation that multi-step diffusion performs across sampling steps appears to unfold across the depth of a single forward pass, and can be recovered by decoding intermediate layers with the model's own output head. Most interestingly, we show that this depthwise computation depends on the transport task a flow map is trained to solve. The most surprising case is MeanFlow, where probing shorter transport intervals reveals both denoising and renoising wi
    
[^15]: 低成本的视频-时间先验作为熟悉视频上EEG-fNIRS情绪回归的强基线

    Low-Cost Video--Time Priors as a Strong Baseline for EEG--fNIRS Emotion Regression on Familiar Videos

    [https://arxiv.org/abs/2610.03618](https://arxiv.org/abs/2610.03618)

    该研究发现，在熟悉视频的连续情绪回归中，仅利用视频身份和播放时间构成的低成本先验即可接近EEG-fNIRS融合模型的性能，而EEG-fNIRS生理信号仅带来较小且因人而异的残余增益，因此更适合作为可选的辅助校正信号。

    

    连续情绪回归在观众观看视频时逐时刻估计效价与唤醒度。在熟悉视频的实际部署中，模型通过训练参与者特定的估计，并采用以先验为主导的固定融合方式来检验生理信号是否能带来残余校正。在24名参与者上的五折被试留出评估中，视频-时间先验方法在内部和外部评估中分别与融合模型的MAE差距仅为0.05和0.32。来源明确的消融实验表明，视频身份和视频内时间信息解释了大部分性能提升，而EEG-fNIRS带来的增益较小，且在不同参与者和视频之间存在差异。这些结果证明视频-时间先验是一个强大且低成本的基线，并将EEG-fNIRS定位为熟悉视频情绪回归任务中可选的残余校正信号。

    arXiv:2610.03618v1 Announce Type: new  Abstract: Continuous emotion regression estimates moment-to-moment valence and arousal while a viewer watches a video. In familiar-video deployment, responses fron training participant-specific estimate, and prior-dominating fixed fusion tests whether physiology adds residual correction. In five-fold subject-held-out evaluation on 24was within 0.05 and 0.32 MAE of fusion in the internal and external evaluations, respectively. Source-explicit ablations showed that video identity and within-video tine accounted for most of the reduction, while EG-FNIRS gains were smaller and varied across participants and videos. These results identify the video-time prior as a strong, low-cost baseline and position EEG-fNIRS as an optional residual signal for familiar-video emotion regression.
    
[^16]: 当正确的奖励还不够时：在解析求解的经纪商-交易员博弈中诊断与引导PPO

    When a Correct Reward Is Not Enough: Diagnosing and Guiding PPO in an Analytically Solved Broker-Trader Game

    [https://arxiv.org/abs/2610.03598](https://arxiv.org/abs/2610.03598)

    本文将PPO智能体置于解析可解的连续时间经纪商-交易员博弈中，利用已知的解析解来诊断并引导强化学习，发现即使奖励设计正确，PPO在存在随机非知情订单流时依然难以学到准确策略。

    

    当复杂动态使解析策略难以获得时，强化学习（RL）越来越多地被用于金融最优控制问题。金融数学文献提供了许多已求解的模型，其方程与最优控制可以用来评估和引导学习；我们探究强化学习能否利用这些已有成果。我们将一个近端策略优化（PPO）智能体置于一个解析可解的连续时间经纪商-交易员博弈中。PPO取代经纪商并选择其交易速度，同时与知情交易者以及随机的非知情订单流进行交互。我们从经纪商的连续时间收益中推导出有限步奖励，并通过网格细化与精确的单步恒等式验证其离散实现。在无非知情订单流的情况下，经验证集选择的PPO-前馈神经网络（FFNN）能够接近参考最优动作。在存在随机非知情订单流的情况下，所测试的PPO-FFNN与PPO-LSTM仍然不准确，尽管……（摘要在此处被截断）

    arXiv:2610.03598v1 Announce Type: cross  Abstract: Reinforcement learning (RL) is increasingly used for financial optimal-control problems when complex dynamics make analytical strategies difficult to obtain. There are financial mathematics literactures which provides many solved models whose equations and controls could evaluate and guide learning; we ask whether RL can exploit these results.   We place a proximal policy optimisation (PPO) agent in an analytically solved continuous-time broker--trader game. PPO replaces the broker and chooses its trading speed while interacting with an informed trader and stochastic uninformed order flow. We derive a finite-step reward from the broker's continuous-time payoff and verify its discrete implementation through grid refinement and an exact one-step identity. With zero uninformed flow, a validation-selected PPO--FFNN approaches the reference action. With stochastic uninformed flow, the tested PPO--FFNN and PPO--LSTM remain inaccurate, althou
    
[^17]: HazardWeaver：面向灾害分析智能体的科学路线选择

    HazardWeaver: Scientific Route Selection for Hazard Analysis Agents

    [https://arxiv.org/abs/2610.03591](https://arxiv.org/abs/2610.03591)

    提出HazardWeaver框架，将灾害分析中的方法选择建模为状态依赖的科学路线选择问题，使智能体能够根据证据和数据工具的可用性动态确定并调整适用的科学方法。

    

    理解和评估自然灾害对于灾害防备和风险降低至关重要。大语言模型的最新进展激发了人们对用于灾害分析的AI智能体的浓厚兴趣，尤其是其将科学数据、模型和工具整合到自动化工作流中的能力。然而，有效的自动化要求智能体能够确定哪些科学方法适合特定事件，并且可以基于可用的数据和工具执行。随着新证据和执行结果的出现，这些条件可能发生变化，需要智能体重新考虑其选择。我们将该问题表述为状态依赖的科学路线选择，并提出了HazardWeaver。具体而言，HazardWeaver首先利用灾害知识编译器提取控制科学方法适用性的、与证据相关联的条件，然后其灾害能力图表示可执行的科学能力并检查兼容性。

    arXiv:2610.03591v1 Announce Type: new  Abstract: Understanding and assessing natural hazards is essential for disaster preparedness and risk reduction. Recent advances in large language models have spurred growing interest in AI agents for hazard analysis, particularly their ability to integrate scientific data, models, and tools into automated workflows. However, effective automation requires agents to determine which scientific methods are appropriate for a given event and executable with the available data and tools. As new evidence and execution results become available, these conditions can change, requiring agents to reconsider their choices. We formulate this problem as state-dependent scientific route selection and introduce HazardWeaver. Specifically, HazardWeaver first leverages the Hazard Knowledge Compiler to extract evidence-linked conditions governing scientific applicability, then its Hazard Capability Graph represents executable scientific capabilities and checks compat
    
[^18]: 智能体安全基准中的威胁保持表征敏感性

    Threat-Preserving Representation Sensitivity in Agent-Security Benchmarks

    [https://arxiv.org/abs/2610.03585](https://arxiv.org/abs/2610.03585)

    论文提出威胁保持表征敏感性（TPRS）指标，发现在底层威胁完全不变的情况下，仅改变智能体可见的表征（如工具名称）就能使攻击成功率显著变化（最高约13个百分点），表明智能体安全基准的测量结果对表征方式高度敏感，可能无法真实反映模型的安全性。

    

    arXiv:2610.03585v1 公告类型：交叉。摘要：针对基于大语言模型（LLM）的智能体的安全基准测试通常报告攻击成功率（ASR）作为衡量模型鲁棒性的指标，并利用这些分数来比较不同的模型和防御机制，其假设是这些分数能够描述智能体的安全性。在本文中，我们探讨基准的表征方式是否也会影响其测量结果。为了衡量基准表征的影响，我们提出了威胁保持表征敏感性（TPRS），它衡量的是在保持底层任务、有害动作、安全策略、真值、环境和评估标准不变的情况下，仅改变智能体可见的表征时，攻击成功率（ASR）的变化幅度。在 Agent Security Bench（ASB）上，将威胁相关的工具名称替换为威胁中立的名称，使 GPT-5-mini 的实际攻击成功率提高了 11.67 个百分点，使 Claude Haiku 4.5 提高了 13.21 个百分点。在 MCPTox 上，将原始的中立工具名称替换为一个利用……（摘要原文在此处截断）

    arXiv:2610.03585v1 Announce Type: cross  Abstract: Security benchmarks for LLM-based agents often report the attack success rate (ASR) as a measure of model robustness and use these scores to compare different models and defense mechanisms, assuming that they describe the security of the agent. In this paper, we explore whether it also influences the benchmark's measurement.   To measure the effect of the benchmark representation, we introduce threat-preserving representation sensitivity (TPRS), which measures how much the ASR changes when we change the agent-visible representation while holding the underlying task, harmful action, security policy, ground truth, environment, and the evaluation criteria fixed.   On Agent Security Bench (ASB), replacing threat-related tool names with threat-neutral names raises the committed attack success rate by 11.67 percentage points on GPT-5-mini and by 13.21 points on Claude Haiku 4.5. On MCPTox, replacing the original neutral tool name with an exp
    
[^19]: 重新思考少步扩散Transformer中的缓存内容：求解器感知的目标选择

    Rethinking What to Cache in Few-Step Diffusion Transformers: Solver-Aware Target Selection

    [https://arxiv.org/abs/2610.03577](https://arxiv.org/abs/2610.03577)

    提出AutoTarget方法，通过在少量无缓存运行的条件下测量重用各候选张量所引入的误差，针对特定模型、求解器和重用调度自适应地选择误差最低的缓存张量，从而提升少步蒸馏扩散Transformer的采样质量。

    

    扩散Transformer（DiTs）能够生成高质量的图像和视频，但生成每个样本需要多次代价高昂的DiT前向计算。加速DiT采样的两种常见方法是步数蒸馏（减少采样步数）和缓存（通过重用较早步骤计算的张量来跳过部分DiT评估）。大多数缓存方法会预先决定重用哪个张量。然而，经过蒸馏后，相邻采样步骤之间的间隔变得更大，跨越这一更大间隔重用张量会引入更多误差，因此选择缓存什么变得尤为重要。为此，我们提出了AutoTarget，一种针对给定模型、求解器和重用调度来选择缓存张量的方法。AutoTarget利用一小组不启用缓存重用的运行，来测量重用每个候选张量所造成的误差，然后选择误差最低的候选张量。我们还分析了某一步重用引入的误差对最终（结果的影响）……

    arXiv:2610.03577v1 Announce Type: cross  Abstract: Diffusion Transformers (DiTs) can generate high-quality images and videos, but generating each sample requires multiple costly DiT forward passes. Two common ways to accelerate DiT sampling are step distillation, which reduces the number of sampling steps, and caching, which skips some DiT evaluations by reusing a tensor computed at an earlier step. Most caching methods decide in advance which tensor to reuse. After distillation, adjacent sampling steps are farther apart. Reusing a tensor across this larger gap introduces more error, so choosing what to cache becomes especially important. We therefore introduce AutoTarget, a method that chooses the cached tensor for a given model, solver, and reuse schedule. AutoTarget uses a small set of runs without cache reuse to measure the error caused by reusing each candidate tensor, then selects the candidate with the lowest error. We also analyze how an error at one reuse step affects the fina
    
[^20]: HyperBrowseComp：面向网页浏览智能体的多语言多模态压力测试

    HyperBrowseComp: A Multilingual and Multimodal Stress Test for Web-Browsing Agents

    [https://arxiv.org/abs/2610.03574](https://arxiv.org/abs/2610.03574)

    HyperBrowseComp是一个覆盖13种语言、包含423道人工验证问题的多语言多模态网页浏览基准，通过要求定位隐蔽证据、追踪多步线索链和检查异构信息源，为网页浏览智能体提供了极具挑战性的压力测试。

    

    我们推出了HyperBrowseComp，这是一个多语言多模态浏览基准，包含423道人工编写并经人工验证的问题，覆盖13种语言，由母语或高度熟练的使用者撰写。这些问题被设计得极具挑战性。每个问题都指向一个简洁且可公开验证的答案，而发现答案需要定位隐蔽的证据、追踪多步骤的线索链，或检查视频、扫描文档、图像、地图等异构信息源。为降低问题仅凭模型参数化知识就能被回答的可能性，我们通过使用无网络访问的模型对问题进行评估来过滤掉较简单的问题。我们在统一的智能体协议下，使用提供商原生搜索和共享的外部检索工具对多个模型进行了评估。为给模型性能与投入程度提供参照，我们还对部分问题样本进行了人类评估。HyperBrowseComp提供了一个具有挑战性的测试平台。

    arXiv:2610.03574v1 Announce Type: new  Abstract: We introduce HyperBrowseComp, a multilingual and multimodal browsing benchmark comprising 423 manually authored and human-validated questions across 13 languages, written by native or highly proficient speakers. Questions are designed to be extremely challenging. Each question targets a concise, publicly verifiable answer whose discovery requires locating obscure evidence, following multi-step clue chains, or inspecting heterogeneous sources such as videos, scanned documents, images, or maps. Easier questions are filtered out by evaluating them with models without internet access to reduce the likelihood that they can be answered with parametric knowledge alone. We evaluate several models using provider-native search and a shared external retrieval harness under a common agent protocol. To contextualize model performance and effort, we also conduct a human evaluation on a sample of the questions. HyperBrowseComp provides a challenging te
    
[^21]: 学习评估心跳可观测性以实现毫米波心率感知

    Learning to Assess Heartbeat Observability for mmWave Heart-Rate Sensing

    [https://arxiv.org/abs/2610.03570](https://arxiv.org/abs/2610.03570)

    该论文提出HEAR框架，通过可控多散射体FMCW仿真器自动生成可观测性标签，并利用紧凑的双任务Transformer从毫米波雷达测量中评估心跳可观测性，实现选择性的可靠非接触心率估计。

    

    基于毫米波雷达的非接触式心率感知需要评估单次测量结果是否能够支持可靠的估计。我们研究了如何学习评估心跳可观测性（heartbeat observability），即所采集相位频谱中心跳成分的可读性，以用于选择性心率估计。即使宏观观测几何条件相似，散射体回波的相干叠加也可能抑制心跳成分，这促使我们需要直接从采集的测量数据中进行评估。为了在不同可观测性条件下获得训练监督，我们开发了一个可控的多散射体调频连续波（FMCW）仿真器。仿真中主导心跳频段峰值与已知心率之间的一致性为每次仿真测量提供了自动的可观测性标签。我们提出了HEAR（Heartbeat Estimation with Assessed Reliability，具备可靠性评估的心跳估计），这是一个紧凑的双任务Transformer，可联合预测观测（原文摘要在此处被截断）……

    arXiv:2610.03570v1 Announce Type: new  Abstract: Contactless heart-rate sensing with millimeter-wave (mmWave) radar requires assessing whether individual measurements support reliable estimation. We study learning to assess heartbeat observability, defined as the readability of the heartbeat component in an acquired phase spectrum, for selective heart-rate estimation. Coherent superposition of scatterer returns can suppress this component even under similar macroscopic observation geometry, motivating assessment directly from acquired measurements. To obtain training supervision across different observability conditions, we develop a controllable multi-scatterer frequency-modulated continuous-wave (FMCW) simulator. Agreement between the dominant heartbeat-band peak and the known heart rate provides an automatic observability label for each simulated measurement. We propose HEAR (Heartbeat Estimation with Assessed Reliability), a compact dual-task Transformer that jointly predicts an ob
    
[^22]: 知识还是计算器？分解可验证金融智能体工作流中的技能溢价

    Knowledge or Calculator? Decomposing the Skill Premium in Verifiable Financial Agent Workflows

    [https://arxiv.org/abs/2610.03564](https://arxiv.org/abs/2610.03564)

    该论文提出FinSkillBench评估套件，通过确定性验证器量化金融AI智能体能力，发现人工策划的技能包可显著提升模型表现16.2分，且可执行工具（+19.5分）的贡献远大于程序性文档（+5.6分）。

    

    金融AI智能体需要做的不仅仅是检索事实：投资工作流要求正确的量化执行、对程序性资源的可靠使用，以及可审计的结构化输出。我们提出了FinSkillBench，这是一个包含2,603个时点案例的评估套件，涵盖投资组合构建、风险管理和基本面分析中的12个子任务，具有隐藏且可再生成的真实基准以及任务特定的确定性验证器。通过在9个模型和3种资源条件下执行17,820个案例，对8个模型的配对分析显示，精心策划的技能包使平均得分提升16.2分（从0.366提升到0.528），而在单个案例内即时生成的技能仅带来0.5分的提升，且消耗了更多的token和交互轮次。随后，我们通过分别提供人工编写的程序性文档和可执行的领域工具来分解这种策划技能带来的溢价：仅提供文档可提升5.6分，仅提供工具可提升19.5分，而二者的组合是……

    arXiv:2610.03564v1 Announce Type: new  Abstract: Financial AI agents must do more than retrieve facts: investment workflows require correct quantitative execution, reliable use of procedural resources, and auditable structured outputs. We introduce FinSkillBench, an evaluation suite of 2,603 point in time episodes across 12 subtasks in portfolio construction, risk management, and fundamental analysis, with hidden regenerable ground truth and task specific deterministic verifiers. Executing 17,820 episodes across 9 models and 3 resource conditions, the paired analysis across 8 models shows that curated skill packages raise mean scores by +16.2 points (0.366 to 0.528), whereas skills generated within a single episode add only +0.5 points while consuming more tokens and turns. We then decompose the curated premium by granting human authored procedural documents and executable domain tools separately: documents alone add +5.6 points, tools alone add +19.5 points, and their combination is s
    
[^23]: Cephalonauts One：一个用于解码人脑自然语音的深度fMRI数据集

    Cephalonauts One: A deep fMRI dataset for decoding naturalistic speech in the human brain

    [https://arxiv.org/abs/2610.03558](https://arxiv.org/abs/2610.03558)

    发布了迄今最大的自然语音fMRI数据集Cephalonauts One（每名受试者30小时数据），并提出以音频片段检索为任务形式的大脑解码基准，附带标准化数据划分、评估指标和基线解码器。

    

    Cephalonauts One 是一个全脑3特斯拉（3T）功能磁共振成像（fMRI）数据集，在受试者收听音频播客时采集。三名健康受试者进行了多次扫描会话，每次会话包含五个15分钟的扫描运行，受试者在扫描期间收听其母语的播客。每名受试者拥有30小时的fMRI数据，本次发布的数据集是迄今为止使用自然语音刺激的最大规模fMRI数据集。该数据集将大脑活动与相应的播客音频、转录文本注释以及导出的刺激嵌入配对。此外，我们引入了一个以音频片段检索形式表述的大脑解码基准：给定来自保留会话的fMRI活动，解码器必须在候选片段中识别出与之时间对齐的相应播客音频片段。我们为该任务提供了标准化的数据划分、评估指标和基线解码器。最后，缩放分析表明解码性能（原文在此处截断）。

    arXiv:2610.03558v1 Announce Type: cross  Abstract: Cephalonauts One is a whole-brain 3 Tesla (3T) functional magnetic resonance imaging (fMRI) dataset recorded while subjects listened to audio podcasts. Three healthy subjects underwent multiple scanning sessions, each consisting of five 15-minute runs, while listening to podcasts in their native language. With 30 hours of fMRI data per subject, the current release is the deepest available fMRI dataset using naturalistic speech stimuli. The dataset pairs brain activity with the corresponding podcast audio, transcript annotations, and derived stimulus embeddings. Furthermore, we introduce a brain decoding benchmark formulated as audio segment retrieval: given fMRI activity from a held-out session, the decoder must identify the corresponding time-aligned podcast audio segment among candidate segments. We provide standardized splits, evaluation metrics, and baseline decoders for this task. Finally, a scaling analysis shows that decoding pe
    
[^24]: 面向前沿推理数据合成的递归框架自我改进

    Recursive Harness Self-Improvement for Frontier Reasoning Data Synthesis

    [https://arxiv.org/abs/2610.03548](https://arxiv.org/abs/2610.03548)

    该论文提出任务-框架协同进化框架，通过在线和任务后两种自我改进机制递归优化数据合成的构建框架本身，在保持模型权重与验证标准不变的前提下，经十四轮进化使求解器平均准确率从100.0%降至54.8%，从而合成出难度更高的前沿推理数据。

    

    生成难度逐步提升的推理问题需要合成流程能够随着任务分布的演化而自适应调整。现有的任务级递归方法仅将已生成的问题作为种子复用，但构建框架本身保持不变。我们提出任务-框架协同进化，这是一个用于推理数据合成中递归框架自我改进（RSI）的框架。在线自我改进在生成过程中将求解器的中间失败转化为可复用的技能；任务后自我改进在每批任务完成后修订技能、提示词和工作流程，仅当候选方案能在有界成本增加的范围内生成更难的有效任务时才予以采纳。模型权重和验证标准始终保持不变。在数学、编程和科学领域，经过十四轮进化，求解器平均准确率从100.0%降至54.8%。消融实验表明，结合两种更新计划比固定框架递归或单独使用任一计划能生成更难的任务。

    arXiv:2610.03548v1 Announce Type: new  Abstract: Generating progressively harder reasoning problems requires synthesis procedures that adapt as the task distribution evolves. Existing task-level recursion reuses generated problems as seeds but leaves the construction harness unchanged. We present task-harness co-evolution, a framework for recursive harness self-improvement (RSI) in reasoning-data synthesis. Online self-improvement converts intermediate solver failures into reusable skills during generation. Post-task self-improvement revises skills, prompts, and workflows after each batch, adopting candidates only when they generate harder valid tasks within a bounded cost increase. Model weights and verification criteria remain fixed. Across mathematics, coding, and science, mean solver accuracy decreases from 100.0% to 54.8% over fourteen evolution rounds. Ablations show that combining both update schedules produces harder tasks than fixed-harness recursion or either schedule alone. 
    
[^25]: 超越训练模型：为健全的GNN可解释器基准而编译

    Beyond Trained Models: Compiling GNNs for a Sound Explainer Benchmark

    [https://arxiv.org/abs/2610.03526](https://arxiv.org/abs/2610.03526)

    该论文揭示了现有GNN可解释器基准中“训练模型依赖预期模体”这一隐含假设的不成立，并提出Gracr——首个将分级模态逻辑公式编译为GNN权重的编译器，通过用编译替代训练构建了真实解释可被形式化定义并精确计算的健全可解释器基准。

    

    图神经网络（GNN）的可解释器通常通过其合理性（plausibility）来评估，即其解释能在多大程度上恢复预先定义的真实标准（ground truth），例如植入数据中的模体（motif）。这一评估协议隐含地假设：在此类数据上训练的GNN依赖于预期的模体。尽管先前的工作已经质疑过这一假设，合理性评估仍然被广泛使用。首先，我们证明该假设在多个广泛使用的基准上并不成立，例如，仅凭度统计信息就足以解决这些任务。随后，我们通过用编译替代训练来消除这一混淆因素。为此，我们引入了Gracr——首个将分级模态逻辑公式编译为GNN权重的编译器，由此得到的模型能够复现相应公式的行为。由于模型的行为在构造时就已知晓，我们可以形式化地定义其真实解释并对其进行精确计算。基于此……

    arXiv:2610.03526v1 Announce Type: cross  Abstract: Explainers for Graph Neural Networks (GNNs) are commonly evaluated by their plausibility, i.e., how well their explanations recover a predefined ground truth, such as a motif planted in the data. This protocol implicitly assumes that a GNN trained on such data relies on the intended motif. Although prior work has questioned this assumption, plausibility remains widespread. First, we show that the assumption is violated on several widely used benchmarks, where, e.g., degree statistics alone suffice to solve the task. Then, we remove this confounder by replacing training with compilation. We achieve this by introducing $\mathsf{Gracr}$, the first compiler translating graded modal logic formulas into GNN weights, yielding models that replicate the behaviour of the corresponding formulas. Since the behaviour of the model is now known by construction, we can define its ground truth explanation formally and compute it exactly. Building on th
    
[^26]: 从基准测试到生产环境：面向复杂金融数据的Text-to-SQL系统

    From Benchmarks to Production: A Text-to-SQL System for Complex Financial Data

    [https://arxiv.org/abs/2610.03524](https://arxiv.org/abs/2610.03524)

    FLINT是一个针对生产环境金融数据库的领域专用Text-to-SQL系统，通过查找代理解析不透明概念、嵌入检索专家模板以及基于外键链遍历的模式链接三大组件，解决了通用系统在此类复杂数据上准确率低于50%的问题。

    

    通用型Text-to-SQL系统在Spider和BIRD等学术基准测试上表现出色，这些基准中的数据库模式相对浅层，列值通常是人类可读的。而在生产环境的金融数据库中，概念以不透明的整数键而非人类可读字符串的形式存储，这些方法的准确率降至50%以下，因为即使是简单的查询也需要多次连接操作，且过滤谓词引用的是不透明的ID。我们提出了Financial LINking Text-to-SQL（FLINT），这是一个领域专用的Text-to-SQL系统，通过三个关键组件弥合了这一差距：(1) 一个查找代理，可动态地将自然语言概念解析为针对特定问题的参考表约束；(2) 基于嵌入的检索，从一个紧凑的、由专家编写的模板库中检索结构相似的查询模板；(3) 模式链接机制，通过遍历外键链将大型表模式修剪为相关子集，而非依赖名称相似度。

    arXiv:2610.03524v1 Announce Type: new  Abstract: General-purpose Text-to-SQL systems achieve strong performance on academic benchmarks like Spider and BIRD, where schemas are relatively shallow and column values are often human readable. In production financial databases, where concepts are stored as opaque integer keys rather than human-readable strings, these methods fall below 50%, as even simple queries require multiple joins and filter predicates reference opaque IDs. We present Financial LINking Text-to-SQL (FLINT), a domain-specialized Text-to-SQL system that closes this gap through three key components: (1) a lookup agent that dynamically resolves natural-language concepts to question-specific reference table constraints, (2) embedding-based retrieval of structurally similar query templates from a compact, expert-authored bank, and (3) schema linking that prunes a large table schema to the relevant subset by traversing foreign-key chains, rather than relying on name similarity 
    
[^27]: 推理模型在因果识别上准确但不可靠

    Reasoning Models Are Accurate but Unsound on Identification

    [https://arxiv.org/abs/2610.03519](https://arxiv.org/abs/2610.03519)

    本文提出CERTID——一个基于可靠且完备的ID算法和结构因果模型精确验证的形式化因果识别评测流水线，首次提供了可证明不可识别的查询及等价公式评分，用以揭示推理模型在因果效应可识别性判断上虽准确但不可靠（会回答不可识别查询）的失败模式。

    

    当被问及某个因果效应能否从观测数据中恢复时，推理模型可能以两种方式失败：拒绝回答本可识别的查询，或回答本不可识别的查询。后者后果更为严重，因为没有任何观测数据能够验证其所声称的公式。衡量这种失败需要可证明为不可识别的查询——这是以往评估所缺乏的——同时还需要能够接受任何等价形式正确公式的评分方式——这是字符串匹配无法提供的。我们构建了CERTID，一个解决上述两个局限的形式化识别流水线。CERTID使用可靠且完备的因果识别算法ID来认证某个效应在给定图和查询下是否可识别，并通过干预分布可被精确获知的结构因果模型来验证模型返回的公式。CERTID进一步发展了理论结果，以缓解结构信息泄露、修复不可识别的查询，并建立评分标准（原文此处截断）。

    arXiv:2610.03519v1 Announce Type: new  Abstract: A reasoning model asked whether a causal effect is recoverable from observational data can fail in two ways: it refuses an identifiable query or answers a nonidentifiable one. The latter is more consequential, as no observational data can validate the claimed formula. Measuring this failure requires queries that are provably non-identifiable, which prior evaluations lack, and grading that accepts correct formulas in any equivalent form, which string matching cannot provide. We build CERTID, a formal identification pipeline that addresses both limitations. CERTID uses the sound and complete causal identification algorithm ID to certify whether an effect is identifiable from a given graph and query, and verifies returned formulas against structural causal models whose interventional distributions are known exactly. CERTID further develops theoretical results to mitigate structural leakage, repair non-identifiable queries, and establish gra
    
[^28]: Weave Forcing：面向交互式长视频生成的组合式记忆路由

    Weave Forcing: Compositional Memory Routing for Interactive Long Video Generation

    [https://arxiv.org/abs/2610.03510](https://arxiv.org/abs/2610.03510)

    提出 Weave Forcing——一个免训练框架，通过 LLM 语义槽路由将提示词分解为角色与背景等组件，并为每个组件从历史镜头中精准选取参考记忆，实现交互式长视频生成中的组合式记忆复用。

    

    自回归视频生成的最新进展提升了长时程中的时间一致性，然而交互式叙事所需要的不仅仅是连续场景的延展：一个新镜头可能需要组合来自不同历史镜头中的角色与背景。整体提示词检索可能忽略各个组成部分各自不同的参考需求，而直接组合所有历史记忆则可能引入不相关的视觉内容。为解决这些问题，我们提出了 Weave Forcing，一个面向交互式长视频生成中组合式记忆复用的免训练框架。首先，我们使用大语言模型（LLM）进行语义槽路由，将用户提示词分解为角色描述与背景描述，并为每个组成部分显式地选择合适的历史参考。为了隔离所需内容，掩码记忆编织利用以语义槽为条件的对比注意力图构建精细的语义掩码，选择性地提取并编织所需的视觉内容。

    arXiv:2610.03510v1 Announce Type: cross  Abstract: Recent advances in autoregressive video generation have improved temporal consistency over extended durations, yet interactive storytelling requires more than continuous scene extension: a new shot may combine characters and backgrounds from different historical shots. Whole prompt retrieval can overlook the distinct reference needs of individual components, while directly combining all historical memories may introduce unrelated visual content. To address these problems, we present Weave Forcing, a training-free framework for compositional memory reuse in interactive long video generation. First, we use an LLM for semantic slot routing to decompose user prompts into character and background descriptions and explicitly select suitable historical references for each component. To isolate the required content, masked memory weaving uses contrasting attention maps conditioned on semantic slots to construct refined semantic masks, selectiv
    
[^29]: 高效推理训练并不总是损害思维链的忠实性与可监控性

    Efficient Reasoning Training Does Not Always Harm CoT Faithfulness and Monitorability

    [https://arxiv.org/abs/2610.03509](https://arxiv.org/abs/2610.03509)

    本文通过三种不同长度压力微调方法对多种模型的系统评估发现，高效推理训练并不必然损害思维链的忠实性与可监控性，其影响取决于所施加长度压力的具体方式。

    

    思维链推理使人类能够检查大型语言模型如何得出答案，并对模型行为进行监督。然而这种推理会带来更高的推理成本，因此催生了训练模型使用更少token来完成任务的高效方法。不过，一个普遍的担忧是，此类训练可能导致模型跳过重要的推理步骤，使思维链不再忠实地反映模型的实际决策。目前尚不清楚这种情况在实践中是否会发生、何时发生，因为不同的效率方法以不同方式对模型的思维链施加长度压力，而且忠实地解释模型决策在某些任务上比其他任务需要更多token。为了理解这些动态，我们使用三种以不同方式施加长度压力的方法对多种模型进行微调，分别是固定生成预算、单样本长度目标和组相对长度奖励。我们评估了高效推理如何影响思维链忠实性（原文摘要在此处截断）

    arXiv:2610.03509v1 Announce Type: new  Abstract: Chain-of-thought (CoT) reasoning allows humans to inspect how large language models reach their answers, and oversee model behaviour. This reasoning comes at an increased inference cost, motivating efficient methods that train models to solve tasks using fewer tokens. However, a common concern is that such training may cause models to skip important reasoning steps, so the CoT no longer faithfully reflects the model's decision. It is unclear whether or when this occurs in practice, since different efficiency methods apply length pressure to models' CoT in distinct ways, and faithfully explaining a model's decision takes more tokens on some tasks than others. To understand these dynamics, we fine-tune a variety of models with three methods that apply length pressure differently, namely a fixed generation budget, a per-example length target, and a group-relative length reward. We evaluate how efficient reasoning affects CoT faithfulness (i
    
[^30]: 认证机制化编辑：技能移除与保留的行为保证

    Certified Mechanistic Edits: Behavioral Guarantees for Skill Removal and Preservation

    [https://arxiv.org/abs/2610.03502](https://arxiv.org/abs/2610.03502)

    该论文首次提出对机制化编辑的行为效果进行认证的方法，可对连续嵌入空间区域内的每个输入可证明地保证：禁用一个电路将移除一种技能同时保留另一种技能，并在标准Transformer上验证了该方法的有效性。

    

    机制化编辑（消融、权重编辑、激活引导）是从神经网络中遗忘有害能力同时保留有用能力的标准工具。当前方法仅通过测试来验证其效果，而测试永远无法覆盖整个连续的输入区域。先前处于可解释性与验证交界处的工作认证的是模型的描述：例如某个电路计算了什么，或者该电路是否忠实解释了整体。我们则认证编辑的行为效果：即在某个区域内，禁用一个电路会移除一种技能，并且可证明地为每一个输入保留另一种技能；这在信息流安全的意义上构成了特征非干扰保证。我们从玩具ReLU网络一直演示到标准的softmax + LayerNorm Transformer，在连续嵌入空间区域上证明了技能的移除与保留，并将输入扰动维度提升至精确求解器所能处理的约9倍。

    arXiv:2610.03502v1 Announce Type: cross  Abstract: Mechanistic edits (ablations, weight edits, activation steering) are the standard tools for unlearning a harmful capability from a neural network while preserving useful ones. Current approaches validate their effects only by testing, which can never cover an entire continuous region of inputs. Prior work at the interpretability-verification boundary certifies descriptions of a model: what a circuit computes, or whether it faithfully explains the whole. We instead certify the behavioral effect of an edit: that disabling a circuit removes one skill and provably preserves another, for every input in a region; a feature non-interference guarantee in the information-flow-security sense. We demonstrate such certified edits from toy ReLU networks up to a standard softmax + LayerNorm transformer, proving removal and preservation over continuous embedding-space regions and reaching roughly 9x the input-perturbation dimension an exact solver ca
    
[^31]: 检测与抑制：针对VLA模型中对抗性补丁的机制性防御

    Detect and Suppress: A Mechanistic Defense against Adversarial Patches in VLA Models

    [https://arxiv.org/abs/2610.03498](https://arxiv.org/abs/2610.03498)

    该论文通过稀疏自编码器机制性分析发现了VLA模型中与对抗性补丁激活高度相关的内部特征，并在线性探针检测到攻击时条件性地抑制该特征，从而无需微调即可显著提升模型对对抗攻击的鲁棒性。

    

    对抗性补丁可以通过操纵视觉观测来干扰视觉-语言-动作（VLA）模型，导致机器人控制失败。然而，这些失败背后的内部机制是什么，以及如何通过有针对性的干预来缓解它们，目前仍知之甚少。在这项工作中，我们使用稀疏自编码器（SAE）对VLA表征进行机制性分析，并识别出一个其激活与对抗性补丁的存在高度相关的特征。基于这一分析，我们仅在线性探针检测到攻击时，才在推理阶段抑制该识别出的特征。这种干预无需对VLA进行微调即可提升鲁棒性。我们在LIBERO-10上评估了该方法对抗VLA对抗性补丁攻击的效果。条件性干预在间歇性攻击下提高了成功率，而持续施加相同的干预则会显著降低策略性能。这些结果表明……

    arXiv:2610.03498v1 Announce Type: cross  Abstract: Adversarial patches can disrupt Vision-Language-Action (VLA) models by manipulating visual observations, leading to failures in robot control. However, it remains poorly understood which internal mechanisms underlie these failures and how targeted interventions can mitigate them. In this work, we mechanistically analyze VLA representations using a sparse autoencoder (SAE) and identify a feature whose activation strongly correlates with the presence of an adversarial patch. Based on this analysis, we suppress the identified feature at inference time only when a linear probe detects an attack. This intervention improves robustness without the cost of fine-tuning the VLA. We evaluate our method against VLA adversarial patch attacks on LIBERO-10. Conditional intervention improves success rate under intermittent attacks, whereas continuously applying the same intervention substantially degrades policy performance. These results show that at
    
[^32]: AREX：用于流匹配少步采样的仿射-残差指数积分器

    AREX: Affine-Residual Exponential Integrator for Few-Step Sampling in Flow Matching

    [https://arxiv.org/abs/2610.03483](https://arxiv.org/abs/2610.03483)

    AREX是一种无需训练的流匹配模型少步采样器，它将采样动力学分解为由目标均值和协方差决定的仿射分量（用显式矩阵值传播子积分）与神经残差项，在无需重训练的情况下持续提升少步采样的样本保真度。

    

    我们提出了AREX，一种面向预训练流匹配模型的无需训练的采样器，它利用目标均值和协方差来捕获采样动态中可解析处理的部分。我们证明了矩匹配高斯目标的速度场是边缘速度场的 $L^2$ 最优仿射近似。这促使我们将学习到的动力学分解为覆盖整个采样路径的仿射分量（由目标的前两阶矩决定）以及一个神经残差项。AREX保留仿射分量，并使用显式矩阵值传播子对其进行积分；相应地，我们只需对残差项进行积分。这不同于标量指数积分器，后者只能解析地处理各向同性的线性动力学。在图像和文本生成图像任务中，AREX在少步采样机制下持续提升样本保真度，而无需重新训练底层模型。

    arXiv:2610.03483v1 Announce Type: cross  Abstract: We introduce AREX, a training-free sampler for pretrained flow matching models that uses the target mean and covariance to capture an analytically tractable part of the sampling dynamics. We show that the velocity field of the moment-matched Gaussian target is the $L^2$-optimal affine approximation to the marginal velocity field. This motivates decomposition of the learned dynamics into an affine component over the whole sampling path, determined by the first two target moments, and a neural residual term. AREX keeps the affine component and integrates it using an explicit matrix-valued propagator. In turn, we only require to integrate over the residual term. This differs from scalar exponential integrators, which analytically handle only isotropic linear dynamics. Across image and text-to-image generation tasks, AREX consistently improves sample fidelity in the few-step sampling regime without retraining the underlying model.
    
[^33]: MobiAgent：面向长时域移动操作的双循环递归策略自我改进

    MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation

    [https://arxiv.org/abs/2610.03476](https://arxiv.org/abs/2610.03476)

    本文提出双循环智能体框架MobiAgent，通过内循环利用可组合的原子技能将高层推理与低层控制解耦以实现稳健的长时域移动操作，并通过递归策略自我改进实现持续学习。

    

    arXiv:2610.03476v1 通告类型：交叉 摘要：长时域移动操作由于执行误差的累积以及移动底盘与手臂控制之间的容量干扰而面临重大挑战。虽然最近的视觉-语言-动作模型在短时域任务上表现出色，但它们缺乏多阶段目标所需的层次化推理能力。此外，现有的层次化智能体存在子任务映射僵化、重规划不灵活以及缺乏持续学习等问题。为了解决这些局限性，我们提出了MobiAgent——一个连接稳健部署执行与递归策略自我改进的双循环智能体框架。在部署阶段，内循环通过高度可组合的原子技能将高层推理与低层控制解耦。它采用视觉-语言模型进行滚动时域规划和视觉反思，动态组合技能以确保稳健的错误恢复。这些技能由专门的流匹配专家（模型执行）（原文摘要在此处截断）

    arXiv:2610.03476v1 Announce Type: cross  Abstract: Long-horizon mobile manipulation presents significant challenges due to compounding execution errors and capacity interference between locomotion and arm control. While recent Vision-Language-Action models excel at short-horizon tasks, they lack the hierarchical reasoning required for multi-stage objectives. Furthermore, existing hierarchical agents suffer from rigid sub-task mapping, inflexible replanning, and a lack of continuous learning. To address these limitations, we introduce MobiAgent, a dual-loop agentic framework that bridges robust deployment execution and recursive policy self-improvement. During deployment, the Inner Loop decouples high-level reasoning from low-level control through highly composable atomic skills. It employs Vision-Language models for receding-horizon planning and visual reflection, dynamically composing skills to ensure robust error recovery. These skills are executed by specialized flow-matching expert
    
[^34]: 阶段结构强化学习中的单一策略还是多策略？

    Single or Multiple Policies for Phase-Structured Reinforcement Learning?

    [https://arxiv.org/abs/2610.03475](https://arxiv.org/abs/2610.03475)

    该论文从理论上证明单一共享策略可以达到任何多策略方案的性能，但实践中多策略是否更优取决于函数逼近、学习优化过程以及策略切换的样本效率与连续性损失等因素。

    

    许多强化学习（RL）问题是非平稳的但具有结构化特征，可以分解为多个阶段，每个阶段都有各自的转移概率和奖励函数。当阶段序列已知时，常见的解决方案是通过增加状态信息来满足马尔可夫性质，并应用标准的强化学习技术。然而，先前的研究发现，针对不同阶段采用多策略的方法可以优于在各阶段之间共享的单一状态增广策略，其原因尚不清楚。在这项工作中，我们首先从理论上证明共享策略可以达到任何多策略解决方案的性能。然而，在实践中，多策略解决方案是否比相应的单一共享策略表现更好，取决于函数逼近、学习和优化过程，以及对于多策略解决方案而言，从一个策略切换到另一个策略所带来的样本效率和连续性损失。我们提出……

    arXiv:2610.03475v1 Announce Type: cross  Abstract: Many reinforcement-learning (RL) problems are non-stationary yet structured and can be decomposed into phases, each with its own transition probabilities and reward functions. When the phase sequence is known, the common solution augments the state with information to satisfy the Markovian property and applies standard RL techniques. However, prior work finds that the multi-policy approach for different phases can outperform a single state-augmented policy shared among the phases, for reasons that remain unclear. In this work, we first show that the shared policy can theoretically achieve performance of any multi-policy solution. However, whether a multi-policy solution can perform better than the corresponding single shared policy in practice depends on function approximation, learning and optimization processes, as well as, for multi-policy solutions, the sample efficiency and loss of continuity from one policy to another. We propose
    
[^35]: 保持解剖连续性：三维腹部CT扫描中结肠分割的三阶段流水线

    Preserving Anatomical Continuity: Three-Stage Pipeline for Colon Segmentation in 3D Abdominal CT Scans

    [https://arxiv.org/abs/2610.03467](https://arxiv.org/abs/2610.03467)

    该论文提出了一种三阶段的保持拓扑结构的结肠分割流水线，通过初始深度学习分割、中心线桥接和重建三个阶段，解决了CT图像结肠分割中预测结果不连通的问题，在保持分割精度的同时显著提高了结构一致性。

    

    从CT图像中准确分割结肠对于结直肠疾病分析至关重要，然而基于深度学习的方法由于复杂的解剖结构，常常产生不连通的预测结果。本研究提出了一种三阶段的保持拓扑结构的分割流水线来解决这一问题。第一阶段执行初始的基于深度学习的分割，随后通过中心线桥接重新连接不连续的区域，最后通过重建阶段来完善连续性。在TotalSegmentator和RAOS数据集上，使用重叠度、距离和基于拓扑的指标进行评估，结果表明该方法在保持分割精度的同时提高了结构一致性。所提出的方法增强了拓扑完整性，使结肠分割在临床和研究应用中更加可靠。

    arXiv:2610.03467v1 Announce Type: cross  Abstract: Accurate colon segmentation from CT images is essential for colorectal disease analysis, yet deep learning based methods often produce disconnected predictions due to complex anatomy. This study introduces a three-stage, topology-preserving segmentation pipeline to address this issue. The first stage performs initial deep learning-based segmentation, followed by centreline bridging to reconnect disjoint regions and a reconstruction stage to refine continuity. Evaluations on TotalSegmentator and RAOS datasets using overlap, distance and topology-based metrics demonstrate improved structural consistency while maintaining segmentation accuracy. The proposed method enhances topological integrity, enabling more reliable colon segmentation for clinical and research applications.
    
[^36]: 监控器读数接近零并不能作为行为控制的证据

    A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control

    [https://arxiv.org/abs/2610.03458](https://arxiv.org/abs/2610.03458)

    监控器读数接近零并不意味着模型行为真正受到控制——即使在代码生成环境中探针得分和惩罚值都处于极低水平，模型仍可能在训练早期就持续利用漏洞。

    

    通过可验证奖励进行的后训练可能诱发奖励作弊行为，这促使研究者将监控器纳入训练目标之中，而不仅仅将其用于离线审计。我们证明，较低的监控器读数并不能判断此类干预是否真正控制了模型行为。在一个代码生成环境中（其主要可利用的作弊手段在推理轨迹开始时即可获得），我们针对三个通过相同离线门限检测的监控器训练策略：一个域内激活探针和两个基于策略多早确定其最终答案的条件惩罚项。探针得分从第一个被记录的训练步骤起就处于其数值下限，并且在每次前缀训练运行的结束时点，训练得分的中间值均为零。这些读数估计的是不同的量，因此我们不比较它们的量纲；然而，在每个监控器族内部，较低的读数值并不能证明行为控制已实现。在单一固定配置下，经过前缀训练的运行……（原文摘要不完整）

    arXiv:2610.03458v1 Announce Type: new  Abstract: Post-training with verifiable rewards can induce reward hacking, motivating the use of monitors within the training objective rather than solely for offline auditing. We show that a low monitor readout does not identify whether such an intervention controls behavior. In a code-generation environment whose dominant exploit is available at the start of the reasoning trace, we train policies against three monitors that pass the same offline gate: an in-domain activation probe and two penalties conditioned on how early the policy commits to its own final answer. The probe score is at its numerical floor from the first recorded training step, and the trained-score median is zero for every prefix-trained run at the endpoint. These readouts estimate different quantities, and we do not compare their scales; within each monitor family, however, low values do not establish behavioral control. Within one fixed configuration, prefix-trained runs wit
    
[^37]: 少测量，多知晓：自监督测试时特征获取

    Measure Less, Know More: Self-Supervised Test-Time Feature Acquisition

    [https://arxiv.org/abs/2610.03454](https://arxiv.org/abs/2610.03454)

    该论文提出ECHO-k，一种任务无关的自监督测试时模态获取方法，它利用基础模型的内部预训练表示作为代理目标，并通过强化学习策略顺序选择信息量最大的模态，从而在有限预算下持续提升下游任务性能。

    

    多模态、高维学习的最新进展使基础模型能够处理异构的大规模数据。然而，在测试时获取所有特征或模态可能成本极高且往往冗余。因此，顺序选择信息丰富的模态至关重要，但当下游任务或预测目标未知时，这一任务充满挑战。为此，我们提出了ECHO-k，一种任务无关且自监督的模态获取学习原则：我们使用深度模型的内部预训练表示（例如来自基础模型的表示）作为总结跨模态信息的代理目标。我们在一个简化的线性设定中提供了理论保证，从而为顺序模态选择的强化学习（RL）策略提供了理论依据。在任务无关和无标签获取的各类基线方法中，ECHO-k在多种基础模型上持续提升了预算约束下的下游任务性能。

    arXiv:2610.03454v1 Announce Type: cross  Abstract: Recent progress in multimodal, high-dimensional learning has enabled foundation models to process heterogeneous, large-scale data. However, at test time, acquiring all features or modalities can be prohibitively costly and often redundant. Sequentially selecting informative modalities is therefore critical, yet challenging when the downstream task or prediction target is unknown. To this end, we introduce ECHO-$k$, a task-agnostic and self-supervised learning principle for modality acquisition: we use a deep model's internal pretrained representations (e.g., from a foundation model) as proxy targets that summarize cross-modal information. We provide theoretical guarantees in a stylized linear setting that motivate a reinforcement learning (RL) policy for sequential modality selection. Across task-agnostic and label-free acquisition baselines, ECHO-$k$ consistently improves budgeted downstream performance across diverse foundation-model
    
[^38]: 被破坏却依然正确：视觉语言模型为何在内部“欺骗”自己

    Corrupted but Correct: Why Vision-Language Models Lie to Themselves Internally

    [https://arxiv.org/abs/2610.03445](https://arxiv.org/abs/2610.03445)

    该论文发现并定义了视觉语言模型中的“训练/推理差距”——对抗扰动虽能把教师强制训练损失压至近零，模型自由生成时却仍输出正确描述——并通过 logit lens 将该差距精确归因于单一自回归步骤中目标词元排名恰好固定为第3位的内部机制。

    

    一种针对性的对抗扰动可以将视觉语言模型（VLM）在固定目标描述上的教师强制（teacher-forced）训练损失降至接近零，然而当允许同一模型自由生成时，它输出的却是原始的正确描述，完全没有目标内容的痕迹。我们将这种分离现象称为“训练/推理差距”，并在 Qwen2.5-VL-7B-Instruct 模型上，通过对200张留出的 COCO 图像实施受控的两阶段 PGD 攻击，为该差距给出了精确的机制解释。首先，我们表明图像级别的像素统计特征——包括从 CNN 鲁棒性文献中正确重新实现的基于纹理的可攻击性度量——对于预测哪些图像会被破坏几乎没有预测能力（最佳预测器 r=-0.050, p=0.484；岭回归 R²=0.069）。其次，利用 logit lens 技术，我们将该差距定位到单个自回归步骤：在已生成正确首词的条件下，目标词元的排名恰好固定为第3位。

    arXiv:2610.03445v1 Announce Type: cross  Abstract: A targeted adversarial perturbation can drive a vision-language model's (VLM's) teacher-forced training loss for a fixed target caption to near zero, yet the same model, allowed to generate freely, produces the original, correct description with no trace of the target. We call this dissociation the train/inference gap, and give it a precise mechanistic account on Qwen2.5-VL-7B-Instruct using a controlled two-stage PGD attack on 200 held-out COCO images. First, we show that image-level pixel statistics, including a correctly re-implemented, texture-based attackability measure from the CNN robustness literature, have essentially no predictive power over which images are corrupted (best predictor r=-0.050, p=0.484; ridge regression R^2=0.069). Second, using the logit lens, we localise the gap to a single autoregressive step: the rank of the target token, conditioned on the correct first token already being generated, is fixed at exactly 3
    
[^39]: OptiSelect：优化器如何塑造数据课程？

    OptiSelect: How does the Optimizer Shape Data Curriculum?

    [https://arxiv.org/abs/2610.03432](https://arxiv.org/abs/2610.03432)

    本文提出OptiSelect优化器感知的数据选择范式，首次系统研究优化器如何影响数据课程选择，理论上证明基于符号和极坐标切向预处理的优化器（Lion、Muon）因效用分数可区分性崩溃而限制选择增益，而对角自适应优化器（AdamW、Sophia）则能获得严格更优的增益上界。

    

    在线数据选择通过在每个批次内仅训练最有价值的候选样本，为大语言模型预训练带来了显著的效率提升。由于候选样本的价值是通过其有效的模型更新来体现的，有原则的数据选择应当考虑优化器步骤——因为优化器会在原始梯度更新模型参数之前对其进行重塑。我们将这种优化器感知的选择范式形式化为OptiSelect，并首次系统性地研究了优化器如何塑造数据选择。我们的理论建立了一个选择增益原则：在线选择的优势由优化器诱导的效用分数的可区分性所决定。我们证明，Lion和Muon所采用的基于符号的及极坐标切向的预处理器会遭受可区分性崩溃，从而限制了OptiSelect可获得的增益上限；而对角自适应优化器（如AdamW和Sophia）则具有严格更优的上界。该性……

    arXiv:2610.03432v1 Announce Type: cross  Abstract: Online data selection has demonstrated substantial efficiency gains for LLM pretraining by training on the most valuable candidates within each batch. Since a candidate's value is realized through its effective model update, principled selection should account for the optimizer step, which reshapes the raw gradient before it updates model parameters. We formalize this optimizer-aware selection paradigm as OptiSelect and present the first systematic study of how the optimizer shapes data selection. Our theory establishes a selection gain principle in which the advantage of online selection is governed by the discriminability of the optimizer-induced utility scores. We prove that sign-based and polar-tangential preconditioners of Lion and Muon would suffer from a discriminability collapse which caps attainable gains from OptiSelect, whereas diagonal-adaptive optimizers such as AdamW and Sophia admit strictly better upper bounds. The prop
    
[^40]: 插队：利用大语言模型调度中的长度预测

    Jumping the Line: Exploiting Length Predictions in LLM Scheduling

    [https://arxiv.org/abs/2610.03430](https://arxiv.org/abs/2610.03430)

    提出JIL攻击方法，通过优化对抗性后缀操纵长度预测信号，使LLM调度器低估请求长度从而插队获得更高优先级，在端到端服务实验中使对抗性请求平均完成速度最高提升1.53倍。

    

    高效的请求调度对于降低大语言模型（LLM）服务的完成时间日益重要。诸如最短作业优先（Shortest Job First）等基于规模的策略会优先处理较短的请求，但输出长度在生成之前是未知的，因此实际的调度器依赖于预测的长度。我们提出了JIL，这是一种针对基于预测的LLM调度器的攻击方法，它通过操纵调度信号来获得更高的优先级并减少完成时间。以TRAIL作为案例研究，JIL优化了一个对抗性后缀，使轻量级的输出长度探测器低估请求的长度。我们在两个数据集和四个大语言模型上，针对不同的请求配置和部署配置对JIL进行了评估。JIL最多可将预测输出长度降低83.4%，并且在端到端服务实验中，对抗性请求的平均完成速度最高提升1.53倍。预测长度的降低幅度远大于实际变化

    arXiv:2610.03430v1 Announce Type: new  Abstract: Efficient request scheduling is increasingly important for reducing completion time in large language model (LLM) serving. Size-based policies such as Shortest Job First prioritize shorter requests, but output lengths are unknown before generation, so practical schedulers rely on predicted lengths. We introduce JIL, an attack on prediction-based LLM schedulers that manipulates the scheduling signal to obtain higher priority and reduce completion time. Using TRAIL as a case study, JIL optimizes an adversarial suffix that causes a lightweight output-length probe to underestimate a request's length. We evaluate JIL on two datasets and four LLMs across varied request profiles and deployment configurations. JIL reduces predicted output lengths by up to 83.4 percent, and adversarial requests complete up to 1.53 times faster on average in end-to-end serving experiments. The reduction in predicted length is substantially larger than the change i
    
[^41]: 跨越国界成为可疑对象：算法域外性与AI驱动的金融监控

    Becoming Suspicious Across Borders: Algorithmic Extraterritoriality and AI-Driven Financial Surveillance

    [https://arxiv.org/abs/2610.03425](https://arxiv.org/abs/2610.03425)

    本文提出“算法域外性”这一新概念，指出AI驱动的金融监控使“嫌疑”的产生从人类在司法管辖区内的情境化法律判断转变为跨国数据基础设施中的数据驱动过程，监管权力的边界由此从地理管辖转向数据系统中的可见性。

    

    嫌疑是反洗钱与反恐怖融资（AML/CFT）领域中一个重要却难以捉摸的概念，它允许在未达到证据证明门槛的情况下进行干预。在传统形式下，嫌疑可以被理解为人类行为者在明确司法管辖区内作出的情境化法律判断。本文认为，这种理解已不再充分。随着人工智能（AI）成为金融监控体系不可或缺的组成部分，嫌疑越来越多地通过数据驱动的过程被生成。这一转变既是认识论层面的，也是空间层面的。由于AI驱动的金融监控依托跨国数据基础设施运作，监管的覆盖范围与其说取决于行为发生的地点，不如说取决于该行为是否在数据系统中变得可见。本文提出了“算法域外性”这一概念，将其理解为一种由数据基础设施而非……（摘要在此处截断）

    arXiv:2610.03425v1 Announce Type: new  Abstract: Suspicion is an important, yet elusive concept in anti-money laundering and counter-terrorist financing (AML/CFT), which allows for intervention below the threshold of proof. In its traditional form, suspicion can be understood as a situated legal judgement by human actors within identifiable jurisdictions. It is argued that this understanding is no longer adequate. As artificial intelligence (AI) becomes an integral part of financial surveillance, suspicion is increasingly produced through data-driven processes. This transformation is epistemic, but also spatial. Since AI-driven financial surveillance operates through transnational data infrastructures, regulatory reach is less a matter of where conduct occurs than a question of whether such conduct becomes visible within data systems. This article develops the concept of algorithmic extraterritoriality, understood as a form of regulatory power mediated by data infrastructures rather th
    
[^42]: 通过信息增长重新思考节点分类中的认知不确定性

    Rethinking Epistemic Uncertainty in Node Classification through Information Growth

    [https://arxiv.org/abs/2610.03418](https://arxiv.org/abs/2610.03418)

    本文提出了一个在信息增长条件下检验节点分类中认知不确定性可约减性的统计框架，并揭示现有图证据深度学习方法难以满足一致性准则。

    

    认知不确定性应当随着预测器获得更多关于数据生成过程（DGP）的信息而降低。然而，现有的用于节点分类的图证据深度学习（EDL）方法通常从图特有性质出发构建认知不确定性，并在分布外检测等下游任务上对其进行评估，这些做法并未检验其作为DGP信息增加时是否可被约减。为了使这种可约减性能够被直接检验，我们提出了一个在信息增长条件下研究认知不确定性的统计框架。该框架规定了信息增长实验协议以及认知预测器的一致性准则，并使用投影图DGP来确保不断增长的图（通常并不必然提供关于同一DGP的递增信息）构成对同一底层过程的一致观测。我们证明，EDL方法无法……（原文在此处截断）

    arXiv:2610.03418v1 Announce Type: cross  Abstract: Epistemic uncertainty should decrease as additional information about the data-generating process (DGP) becomes available to the predictor. Yet, existing graph evidential deep learning (EDL) methods for node classification typically construct epistemic uncertainty from graph-specific properties and evaluate it on downstream tasks such as out-of-distribution detection, which do not test its reducibility as information about the DGP increases. To make reducibility directly testable, we introduce a statistical framework for studying epistemic uncertainty under information growth. Our framework specifies an information-growth experimental protocol and a consistency criterion for epistemic predictors, while using projective graph DGPs to ensure that growing graphs, which in general need not provide increasing information about the same DGP, constitute coherent observations of the same underlying process. We show that EDL methods do not expl
    
[^43]: ForestQuery：面向统一森林点云分割的边界感知与空间锚定查询学习

    ForestQuery: Boundary-Aware and Spatially Anchored Query Learning for Unified Forest Point Cloud Segmentation

    [https://arxiv.org/abs/2610.03403](https://arxiv.org/abs/2610.03403)

    提出 ForestQuery 框架，通过显式建模边界不确定性并结合空间锚定的语义查询增强（SA-SQE），实现了统一的森林点云语义与实例分割。

    

    森林点云分割是细粒度三维森林场景理解的基础，但由于树木结构不规则、严重遮挡、密度变化以及实例边界模糊等问题，该任务仍极具挑战性。近期基于查询的森林分割方法在统一语义与实例预测方面展现出潜力，但它们仍未充分利用森林特有的空间结构，也未考虑边界的不确定性。在本文中，我们提出了 ForestQuery，一个面向统一森林点云分割的边界感知与空间锚定查询学习框架。ForestQuery 通过两个互补的设计来增强实例查询与语义查询的学习。具体而言，该方法显式建模边界不确定性，以引导可靠的实例查询构建，并通过自适应损失重加权来调节查询优化过程。同时，空间锚定语义查询增强（SA-SQE）引入可学习的三维……（原文摘要在此处被截断）

    arXiv:2610.03403v1 Announce Type: cross  Abstract: Forest point cloud segmentation is fundamental for fine-grained 3D forest scene understanding, yet remains challenging due to irregular tree structures, severe occlusions, density variations, and ambiguous instance boundaries. Recent query-based forest segmentation methods have shown promise for unified semantic and instance prediction, but they still insufficiently exploit forest-specific spatial structure and account for boundary uncertainty. In this paper, we propose ForestQuery, a boundary-aware and spatially anchored query learning framework for unified forest point cloud segmentation. ForestQuery enhances instance and semantic query learning through two complementary designs. Specifically, boundary uncertainty is explicitly modeled to guide reliable instance query construction and modulate query optimization through adaptive loss reweighting. Meanwhile, spatially anchored semantic query enhancement (SA-SQE) introduces learnable 3
    
[^44]: DriftTTS：通过分布匹配漂移实现无需蒸馏的少步数文本转语音

    DriftTTS: Few-Step Text-to-Speech Without Distillation via Distribution-Matching Drift

    [https://arxiv.org/abs/2610.03390](https://arxiv.org/abs/2610.03390)

    DriftTTS提出了一种无需教师模型、蒸馏或对抗训练的少步数文本转语音方法，通过分布匹配漂移目标函数和在策略展开训练，仅用4次函数评估就达到了与现有模型相当甚至更优的合成质量。

    

    少步数神经文本转语音模型通常依赖缩短的扩散或流匹配调度，或者依赖从预训练多步教师模型进行蒸馏。为了避免这些依赖，我们提出了DriftTTS，这是一个无需生成式教师模型、蒸馏或对抗判别训练的少步数梅尔频谱图生成器。DriftTTS在一个由原始梅尔频谱和冻结的掩码自编码器（预训练于相同LJSpeech训练集划分）所定义的梅尔域特征空间中，使用分布匹配漂移目标函数。在策略展开训练使解码器在其自身的中间状态上进行训练，并支持推理时使用与训练时相同的展开深度。在LJSpeech数据集上，DriftTTS在NFE=4时实现了3.87 dB的MCD和3.7%的WER，相比之下Matcha-TTS为3.85 dB和3.4%。在完全配对的盲听测试中，DriftTTS获得了4.18的MOS评分，Matcha-TTS为3.96，真实音频为4.22。这些结果证明了具有竞争力的少步数语音合成性能。

    arXiv:2610.03390v1 Announce Type: cross  Abstract: Few-step neural text-to-speech models often rely on short- ened diffusion or flow-matching schedules, or on distillation from pretrained multi-step teachers. To avoid these depen- dencies, we present DriftTTS, a few-step mel-spectrogram generator trained without a generative teacher, distillation, or adversarial discrimination. DriftTTS uses a distribution- matching drift objective in a mel-domain feature space defined by raw mels and a frozen masked-autoencoder encoder pretrained on the same LJSpeech training split. On-policy rollout trains the decoder on its own interme- diate states and supports inference up to the trained roll- out depth. On LJSpeech, DriftTTS at NFE=4 achieves 3.87 dB MCD and 3.7% WER, compared with 3.85 dB and 3.4% for Matcha-TTS. In a fully paired blind listen- ing test, DriftTTS obtains 4.18 MOS, compared with 3.96 for Matcha-TTS and 4.22 for ground truth. These results demonstrate competitive few-step synthesi
    
[^45]: 类型化决策模型中候选选项覆盖度的基准测试

    Benchmarking Candidate Coverage in Typed Decision Models

    [https://arxiv.org/abs/2610.03387](https://arxiv.org/abs/2610.03387)

    本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。

    

    类型化决策模型会返回选择结果，或针对请求时提供的答案选项返回分布。在选项完整情况下的准确率并不能说明模型是否能识别参考答案缺失的情况，或者是否会避免错误拒绝有效的候选选项。我们提出了一个成对候选覆盖度基准测试协议，并对 Laya 和 Jev 两个模型在 AG News、DBpedia、Emotion 和 TREC 数据集上进行了初步评估。两个模型接收完全相同的冻结文本和请求：300 条校准文本和 589 条测试文本，每个模型产生 23,932 次预测。存在/缺失配对与普通候选数量相匹配，且名称变体保持描述、成员和顺序不变。原生的拒绝行为差异显著：在具有自然名称的五个 TREC 候选选项下，Laya 能检测出 97.2% 的答案缺失案例，但会错误拒绝 69.7% 的存在对照案例；Jev 的这两个比率分别为 24.8% 和 0.0%。仅使用校准数据的 none 分数阈值将这些比率分别改变为 33.9%/3.7% 和 45.0%/1.8%。在 DB

    arXiv:2610.03387v1 Announce Type: new  Abstract: Typed decision models return choices or distributions over answer options supplied at request time. Accuracy with complete options does not establish whether a model recognizes that a reference answer is missing or avoids rejecting valid candidates. We present a paired candidate-coverage benchmark protocol and an initial evaluation of Laya and Jev across AG News, DBpedia, Emotion, and TREC. The models receive identical frozen texts and requests: 300 calibration and 589 test texts yield 23,932 predictions per model. Present/absent pairs match ordinary candidate count, and name variants preserve descriptions, members, and order. Native rejection behavior differs sharply: at five TREC candidates with natural names, Laya detects 97.2% of missing-answer cases but falsely rejects 69.7% of present controls; Jev's rates are 24.8% and 0.0%. Calibration-only none-score thresholds change these rates to 33.9%/3.7% and 45.0%/1.8%, respectively. On DB
    
[^46]: CVE2AP：基于大语言模型的PDDL编码攻击路径自动生成

    CVE2AP: Automated Generation of PDDL-Encoded Attack Paths via Large Language Models

    [https://arxiv.org/abs/2610.03383](https://arxiv.org/abs/2610.03383)

    提出了CVE2AP，一种利用大语言模型从自然语言CVE漏洞描述中自动生成PDDL编码攻击路径的方法，摆脱了传统专家人工建模的瓶颈，提升了攻击路径建模的可扩展性。

    

    攻击路径建模是网络安全分析的基础，规划领域定义语言（PDDL）已被广泛用于将攻击路径编码为形式化、机器可验证的表示，从而支持对漏洞利用、攻击进展及其潜在影响的自动化推理。然而，现有的攻击路径建模方法主要依赖专家驱动的人工构建，限制了其可扩展性以及跟上快速演变的网络威胁的能力。大语言模型（LLM）是很有前景的候选方案，因为其广泛的预训练知识和推理能力使其能够理解威胁情报并将其转化为形式化表示。本文提出了CVE2AP，一种基于大语言模型的方法，用于从自然语言的CVE（公共漏洞披露）描述中自动生成PDDL编码的攻击路径。CVE2AP利用结构化提示并纳入……

    arXiv:2610.03383v1 Announce Type: new  Abstract: Attack Path (AP) modeling is fundamental to cybersecurity analysis, where the Planning Domain Definition Language (PDDL) has been widely adopted to encode APs into formal and machine-verifiable representations for automated reasoning about vulnerability exploitation, attack progression, and their potential impacts. However, existing AP modeling approaches largely rely on expert-driven manual construction, limiting their scalability and ability to keep pace with rapidly evolving cyber threats. Large language models (LLMs) are promising candidates, as their extensive pre-trained knowledge and reasoning capabilities enable them to interpret and transform threat intelligence into formal representations. In this paper, we propose \textbf{CVE2AP}, an LLM-based approach for automatically generating PDDL-encoded attack paths from natural language CVE (Common Vulnerability Exposure) descriptions. CVE2AP leverages structured prompting and incorpor
    
[^47]: 多语言GSM-Symbolic：什么决定了跨语言的能力迁移？

    Multilingual GSM-Symbolic: What determines capability transfer across languages?

    [https://arxiv.org/abs/2610.03367](https://arxiv.org/abs/2610.03367)

    该论文提出了可扩展的多语言数学数据集Multilingual GSM-Symbolic（涵盖15种语言、3万个题目匹配问答对，通过符号化模板防止过拟合），并量化发现模型规模和语言资源水平是决定跨语言能力迁移的最主要因素。

    

    我们对于一种语言中习得的能力如何迁移到另一种语言、以及哪些因素支配这种迁移仍然知之甚少：现有评估依赖于不可比较且易饱和的数据集，并且很少联合考察迁移的决定因素。识别哪些因素能预测迁移，将使我们能够避免对所有语言对进行穷举评估，并让开发者能够针对限制低资源语言性能的因素进行优化。为了评估跨语言能力迁移，我们引入了Multilingual GSM-Symbolic，这是一个可扩展的多语言数学数据集，包含30,000个题目匹配的问答对，涵盖15种语言。它利用符号化模板防止过拟合并确保泛化，能够从单个样本生成数百万个高质量的变体。使用Multilingual GSM-Symbolic，我们量化了能力的最大决定因素：模型规模（β = 1.77）和语言资源水平（β = 0.77）。

    arXiv:2610.03367v1 Announce Type: new  Abstract: We understand little about how capabilities acquired in one language carry over to another, or what governs this transfer: evaluations rely on incomparable, saturation-prone datasets and rarely examine its determinants jointly. Identifying what predicts transfer would let us avoid exhaustive evaluation across all language pairs and let developers target the factors that limit performance in low-resource languages. To evaluate cross-lingual capability transfer, we introduce Multilingual GSM-Symbolic, an extensible multilingual mathematical dataset covering 30,000 item-matched question-answer pairs and spanning 15 languages. It utilises symbolic templates to prevent overfitting and ensure generalisation by allowing generation of millions of high-quality variations from a single sample. Using Multilingual GSM-Symbolic, we quantify the largest determinants of capability as model size ($\beta = 1.77$), language resource level ($\beta = 0.77$)
    
[^48]: 几何与物理的相遇：面向非结构化神经偏微分方程求解器的数据高效预训练

    Geometry Meets Physics: Data-Efficient Pre-Training for Unstructured Neural PDE Solvers

    [https://arxiv.org/abs/2610.03363](https://arxiv.org/abs/2610.03363)

    提出了一个无需磁盘数据的预训练框架：针对稳态问题采用基于内在形状描述符的几何驱动策略，针对瞬态问题采用在线生成合成PDE数据的物理驱动方法，从而实现数据高效的非结构化神经PDE求解器预训练。

    

    非结构化三维几何上的偏微分方程（PDE）神经代理模型常常受限于泛化能力差以及生成大规模训练数据集的高昂成本。因此，在相关PDE动力学的大规模数据集上进行预训练已成为提升这些模型鲁棒性与可扩展性的关键替代方案。然而，这种策略既不节省算力也不节省数据，因为它依赖于生成成本极高的大量预计算数据。在这项工作中，我们引入了一个适用于稳态与瞬态两种情形的无需磁盘数据的预训练框架。对于稳态问题，我们提出了一种几何驱动的策略，利用内在形状描述符来学习复杂三维域的表示。对于瞬态问题，我们引入了一种基于在线生成合成PDE数据的物理驱动方法，使得可扩展的预训练无需依赖昂贵的数据生成。

    arXiv:2610.03363v1 Announce Type: new  Abstract: Neural surrogate models for Partial Differential Equations (PDEs) on unstructured 3D geometries are often limited by poor generalization and the high cost of generating large-scale training datasets. Consequently, pre-training on massive datasets of related PDE dynamics has emerged as a critical alternative to enhance the robustness and scalability of these models. However, this strategy is neither compute- nor data-efficient, as it relies on massive pre-computed data that is very costly to generate. In this work, we introduce a disk-data-free pre-training framework tailored to both steady-state and transient regimes. For steady-state problems, we propose a geometry-driven strategy that leverages intrinsic shape descriptors to learn representations of complex 3D domains. For transient problems, we introduce a physics-driven approach based on online generation of synthetic PDE data, enabling scalable pre-training without reliance on expen
    
[^49]: 跟随赢家：基于交叉熵方法的无批评家强化微调中的保守策略改进

    Follow the Winners: Conservative Policy Improvement with the Cross-Entropy Method for Critic-Free RFT

    [https://arxiv.org/abs/2610.03361](https://arxiv.org/abs/2610.03361)

    FTW 是一种无批评家的强化微调算法，通过将交叉熵方法适配到 RFT 中、用回放缓冲区样本上的序数过滤器替代组采样，从而在有状态环境中难以重复采样的智能体大模型训练中实现保守且稳健的策略改进。

    

    面向智能体大语言模型的无批评家强化微调（RFT）通常采用 GRPO 风格的方法，即在重复的轨迹采样上计算组基线以降低目标方差。然而，这种设置并不适合在有状态环境（如在线服务或安全沙箱）中行动的智能体，因为在这些环境中难以获得重复采样，且激进的策略更新会将冗长且稀疏验证轨迹中的噪声固化下来。我们提出了“跟随赢家”，这是一种无批评家的策略学习算法，它将交叉熵方法适配到强化微调中，用基于经验回放缓冲区样本的序数过滤器替代组采样，从而在回报的顺序统计量上获得多项式集中性保证。我们通过“控制即推断”的视角推导出 FTW，该框架同时将 GRPO 和 DPO 还原为特定的建模选择：GRPO 被识别为风险中性的，而 DPO 与 FTW 共享一个有界的风险寻求偏移，FTW 对该偏移加以控制……

    arXiv:2610.03361v1 Announce Type: cross  Abstract: Critic-free reinforcement fine-tuning (RFT) for agentic large language models is often done through GRPO-style methods, which compute a group baseline over repeated rollouts to reduce target variance. However, this setup is ill-suited to agents acting in stateful environments such as live services or security sandboxes, where repeated rollouts are impractical to obtain and aggressive updates entrench the noise of long, sparsely verified trajectories. We propose \textit{Follow the Winners} (FTW), a critic-free policy-learning algorithm that adapts the cross-entropy method to RFT, replacing group rollouts with an ordinal filter on replay-buffer samples that yields polynomial concentration in the order statistic of returns. We derive FTW through a control-as-inference lens, which also recovers GRPO and DPO as specific modelling choices, identifying GRPO as risk-neutral while DPO and FTW share a bounded risk-seeking offset that FTW control
    
[^50]: ReFract：基于文本世界模型的语言模型智能体视角感知能力基准测试

    ReFract: Benchmarking Perspective Awareness in Language Model Agents with Text World Models

    [https://arxiv.org/abs/2610.03356](https://arxiv.org/abs/2610.03356)

    该论文提出了ReFract基准测试，通过150条专家验证的工业维护场景条目，评估语言模型智能体的“视角感知”能力，即根据用户角色的意图和权限边界，仅使用该角色合法可用的工具采取相应行动的能力。

    

    大语言模型（LLM）智能体正越来越多地被部署在工业维护和设备故障排除等高风险场景中，在这些场景里，工作人员承担着各种各样的角色。因此，一个有能力的智能体必须以与用户角色相适配的方式行事：采取行动并提供信息时，要尊重该角色的知识和能力边界。与编程不同（编程中的错误通常可以恢复），智能体在这些场景中的响应是在物理设备上执行的，因此可能造成不可逆的设备损坏、生产损失或人员伤害。然而，现有的基准测试在很大程度上忽视了智能体需要推断角色意图、并仅通过该角色可以合法使用的工具来行动这一需求，我们将这种能力称为“视角感知”。为此，我们推出了ReFract，一个包含150条经专家验证条目的基准测试，其中智能体必须针对相同的查询，依据角色不同而采取不同的行动……

    arXiv:2610.03356v1 Announce Type: new  Abstract: Large Language Model (LLM) agents are increasingly deployed in high-stakes settings such as industrial maintenance and equipment fault troubleshooting, where workers occupy a variety of roles. A capable agent must therefore act in a way that is calibrated to user's role: taking actions and providing information that respect the role's knowledge and capability boundaries. Unlike coding, where mistakes are usually recoverable, agent responses in these settings are enacted on physical equipment, and can therefore cause irreversible equipment damage, production loss, or personnel harm. Existing benchmarks, however, largely overlook the need for agents to infer what a role intends and acting only through tools that role may legitimately use, a capability which we term Perspective Awareness. To this end, we introduce ReFract, a benchmark of 150 expert-validated entries in which an agent must act differently in response to the same query depend
    
[^51]: 面向富接触操作的等变视觉-触觉扩散策略

    Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation

    [https://arxiv.org/abs/2610.03333](https://arxiv.org/abs/2610.03333)

    提出VISTA，一种工作空间级等变视觉-触觉扩散策略，通过将触觉接触线索融合到球面视觉表示中并利用等变扩散预测动作，大幅提升了富接触操作模仿学习的数据效率。

    

    面向富接触操作的模仿学习需要高质量的专家数据，而这些数据获取成本高昂，这使得学习样本高效的策略成为关键问题。为了解决这一问题，我们提出了VISTA，一种工作空间级别的等变视觉-触觉扩散策略，用于数据高效的富接触模仿学习。VISTA将视觉和触觉观测投影为球面token，通过置换等变的球面融合将触觉接触线索注入视觉球面方向中，并利用末端执行器的姿态旋转融合后的调和表示。所得的表示作为等变扩散策略的条件输入，用于预测空间一致的动作。在仿真和真实机器人环境中的大量实验表明，VISTA相比强大的视觉-触觉模仿学习基线显著提升了数据效率。

    arXiv:2610.03333v1 Announce Type: cross  Abstract: Imitation learning for contact-rich manipulation requires high-quality expert data that is expensive to obtain. This makes learning a sample-efficient policy a key issue. To address this, we propose VISTA, a workspace-level equivariant visuotactile diffusion policy for data-efficient contact-rich imitation learning. VISTA projects visual and tactile observations into spherical tokens, injects tactile contact cues into visual spherical directions through permutation-equivariant spherical fusion, and rotates the fused harmonic representation using the end-effector orientation. The resulting representation conditions an equivariant diffusion policy to predict spatially consistent actions. Extensive experiments in both simulation and real-world robotic settings show that VISTA substantially improves data efficiency over strong visuotactile imitation learning baselines. Project website: https://vista-paper.github.io/
    
[^52]: 亲和学习：面向相关数据的分布式训练

    Cordial Learning: Distributed Training with Correlated Data

    [https://arxiv.org/abs/2610.03330](https://arxiv.org/abs/2610.03330)

    提出了一种名为“亲和学习”的分布式训练框架，通过智能体间仅共享低维输出、本地模型提取同伴信息来处理相关数据问题，并在线性模型假设下证明了其以概率一收敛到全局最优。

    

    我们研究了一个由拥有相关数据的智能体组成的分布式学习任务。具体而言，某个智能体的标签取决于其他智能体对同一样本的输入，且这些输入彼此之间也是相关的。当智能体共享同一环境时，相关数据是普遍存在的现实情况。现有的去中心化方法（如联邦学习）忽略了问题的结构，在相关数据上表现不佳；而另一方面，由于隐私和通信约束，集中式方法又不可行。我们提出了亲和学习（cordial，即相关与分布式学习）来弥补这一空白：智能体之间仅共享低维输出，同时训练本地模型以从同伴处提取有信息量的信号。这种分布式学习引发了一个博弈，其中每个智能体的损失函数依赖于其他智能体的模型。在线性模型假设下，我们证明了亲和学习以概率一收敛到全局最优解。

    arXiv:2610.03330v1 Announce Type: cross  Abstract: We consider a distributed learning task with agents that have correlated data. Specifically, the label of an agent depends on the input of other agents for the same sample, and these inputs are also correlated. Correlated data is the reality when agents share the same environment. Existing decentralized methods, such as federated learning, ignore the structure of the problem and perform poorly on correlated data. On the other hand, centralized approaches are infeasible due to privacy and communication constraints. We introduce cordial (correlated and distributed) learning to address this gap by sharing only low-dimensional outputs between the agents while training local models to extract informative signals from peers. This distributed learning induces a game in which the loss function of each agent depends on the models of others. Assuming a linear model, we prove that cordial learning converges with probability one to a globally opti
    
[^53]: SyntaxBench：大语言模型字符级推理的统计诊断框架

    SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models

    [https://arxiv.org/abs/2610.03329](https://arxiv.org/abs/2610.03329)

    提出SyntaxBench诊断基准，通过五个核心字符级任务和一个高难度子串提取压力测试，结合Cohen's kappa与McNemar检验等统计方法，系统评估了八个开放权重大语言模型的字符级推理能力。

    

    大语言模型越来越多地被应用于小语法错误也至关重要的场景，然而字符级推理目前主要还是通过孤立的探测任务和聚合准确率来进行评估。我们提出了SyntaxBench，一个面向字符级推理的诊断基准和统计评估框架。它包含五个核心任务：字符计数、字母包含检测、回文检测、编辑距离和最长字符串选择，外加一个更困难的子串提取压力测试index_to_span。五个核心任务使用成对的英文输入与字符长度匹配的随机字符串输入；index_to_span文档共享200-500词的长度区间，但不进行字符长度匹配。全部六个任务均采用零样本、单样本和四样本提示。我们在11种推理模式配置下评估了从2B到32B参数的八个开放权重模型。该框架报告精确匹配与宽松准确率、Cohen's kappa系数、带优势比的配对McNemar检验，

    arXiv:2610.03329v1 Announce Type: cross  Abstract: Large language models are increasingly used where small syntactic errors matter, yet character-level reasoning is still evaluated mostly through isolated probes and aggregate accuracy. We introduce SyntaxBench, a diagnostic benchmark and statistical evaluation framework for character-level reasoning. It contains five core tasks, character counting, letter containment, palindrome detection, edit distance, and longest-string selection, plus index_to_span, a harder substring-extraction stress test. The five core tasks use paired English and character-length-matched random-string inputs. index_to_span documents share a 200-500 word band and are not character-length matched. All six tasks use zero-, one-, and four-shot prompts.   We evaluate eight open-weight models from 2B to 32B parameters across 11 reasoning-mode configurations. The framework reports exact-match and relaxed accuracy, Cohen's kappa, paired McNemar tests with odds ratios, 
    
[^54]: 通过轨迹感知低秩近似在压缩扩散语言模型中保留数学推理能力

    Preserving Mathematical Reasoning in Compressed Diffusion Language Models via Trajectory-Aware Low-Rank Approximation

    [https://arxiv.org/abs/2610.03326](https://arxiv.org/abs/2610.03326)

    该论文提出轨迹感知低秩压缩方法Traj-MC，通过蒙特卡洛采样在扩散语言模型部分掩码的推理轨迹上进行校准，从而在压缩后更好地保留模型的数学推理能力。

    

    扩散语言模型（dLLM）的压缩面临一个已知的挑战：校准通常是在干净、完全可见的激活值上进行的，而推理过程却会经过部分掩码的中间状态。对于低秩压缩而言，这引出了两个问题：第一，当近似质量是在轨迹分布的状态上进行度量时，低秩最优性是否仍能被刻画；第二，校准状态的选择是否会影响压缩后数学推理能力的保留？我们通过在腐蚀程度和掩码实现上构建轨迹感知的低秩目标函数来回答这些问题。为了高效估计该目标函数，我们提出了Traj-MC方法，它通过蒙特卡洛采样估计轨迹二阶矩，并给出了精确的采样状态最优性和总体一致性。在相同的压缩预算下，轨迹感知校准提升了模型在生成轨迹上的重建效果……

    arXiv:2610.03326v1 Announce Type: new  Abstract: Diffusion language model (dLLM) compression faces a known challenge because calibration is typically performed on clean, fully visible activations, whereas inference traverses partially masked intermediate states. For low-rank compression, this raises two questions. First, can low-rank optimality still be characterized when approximation quality is measured over trajectory-distributed states, and second, does the choice of calibration states affect mathematical reasoning preservation under compression? We address these questions by formulating a trajectory-aware low-rank objective over corruption levels and masking realizations. To estimate this objective efficiently, we propose Traj-MC, which estimates the trajectory second moment through Monte Carlo sampling and yields exact sampled-state optimality and population consistency. Under matched compression budgets, trajectory-aware calibration improves reconstruction over the generation tr
    
[^55]: 低秩近似认证的信息极限

    Information Limits of Low-Rank Approximation Certification

    [https://arxiv.org/abs/2610.03321](https://arxiv.org/abs/2610.03321)

    该论文刻画了低秩近似认证所需的最小查询代价，证明复用验证响应可使一批查询支持整条嵌套近似路径，且跨 W 条路径的 √log(W+1) 代价依赖经匹配下界证明是最优的。

    

    低秩近似可能需要额外的矩阵-向量乘积来验证其误差是否满足预设的容差。我们针对相对矩阵误差和均方输出误差刻画了这种认证代价。对于单个近似矩阵候选，我们确定了在允许失败概率趋于零时精确的维度一致极小极大查询常数。我们的主要结果关注当近似空间扩展时验证响应的复用：对于独立于验证过程构造的候选族，一批查询即可支持整条嵌套路径，而无需随检查次数的增加而增加查询预算。跨 W 条路径时，利用共享残差能量的集中不等式给出了 √log(W+1) 的依赖关系。匹配的下界表明，对于固定的内部误差目标和足够小的分离间隙，这一依赖关系是最优的。最后，我们在同一离散（原文截断）上比较了两种一致有效的认证方法。

    arXiv:2610.03321v1 Announce Type: cross  Abstract: Low-rank approximation can require additional matrix--vector products to verify that its error meets a prescribed tolerance. We characterize this certification cost for both relative matrix error and mean-square output error. For a single approximation matrix candidate, we determine the exact dimension-uniform minimax query constant as the allowed failure probability vanishes. Our main result concerns reusing validation responses as the approximation space expands. For a candidate family constructed independently of validation, one batch supports an entire nested path without increasing the query budget with the number of checks. Across \(W\) paths, a concentration bound exploiting shared residual energy yields a \(\sqrt{\log(W+1)}\) dependence. A matching lower bound establishes its optimality for fixed interior error targets and sufficiently small separation gaps. Finally, we compare two uniformly valid certificates on the same dispe
    
[^56]: 精炼换可懂度，搜索换身份：测试时计算在掩码扩散TTS中能带来什么

    Refinement Buys Intelligibility, Search Buys Identity: What Test-Time Compute Buys in Masked-Diffusion TTS

    [https://arxiv.org/abs/2610.03320](https://arxiv.org/abs/2610.03320)

    该论文发现掩码扩散TTS中推理时的精炼步数主要提升可懂度（弥补86.2%差距）而对说话人身份提升有限（仅46.4%），且Best-of-K搜索能有效恢复精炼无法带来的说话人身份一致性。

    

    用于文本到语音的扩散语言模型结合了两种计算形式：模型深度（参数）与精炼步数（推理预算）。我们探究这两种计算方式在各能力维度上是否能同等扩展。我们在2,000小时语音数据上训练了15个不同深度的掩码扩散编解码TTS模型（参数量19-133M，3个随机种子），并在推理时扫描精炼步数T∈[1,16]，通过ASR词错误率（衡量可懂度）和说话人验证（衡量身份一致性）在174个留出说话人上评估零样本合成效果。相对于测量下限，精炼步数能够弥补可懂度差距的86.2%，但只能弥补身份差距的46.4%——这一1.86倍的不对称性在多种误差指标下均稳健成立。使用3倍和6倍训练调度重新训练会减弱但无法逆转这一差距（从1.84倍降至1.36倍再到1.23倍），原因是可懂度随精炼步数趋于饱和，而身份一致性仍在持续提升。Best-of-K搜索能够在精炼失效之处恢复说话人身份，在多种设置下取得64.6-79.0%的胜率……

    arXiv:2610.03320v1 Announce Type: new  Abstract: Diffusion language models for text-to-speech combine two forms of computation: model depth (parameters) and refinement steps (inference budget). We ask whether they scale equally across capabilities. We train 15 masked-diffusion codec TTS models varying depth (19-133M parameters, 3 seeds) on 2,000 hours of speech and sweep refinement steps T in [1,16] at inference, measuring zero-shot synthesis via ASR word error rate (intelligibility) and speaker verification (identity) on 174 held-out speakers. Against measured floors, refinement closes 86.2% of the intelligibility range but only 46.4% of the identity range - a 1.86x asymmetry robust across multiple error metrics. Retraining at 3x and 6x schedule attenuates but does not reverse this gap (1.84 to 1.36 to 1.23x), because intelligibility saturates with steps while identity continues improving. Best-of-K search recovers speaker identity where refinement fails, with 64.6-79.0% win rates acr
    
[^57]: 基于大语言模型的多任务进化实现零样本跨问题泛化

    Multi-Task Evolution for Zero-Shot Cross-Problem Generalization using LLMs

    [https://arxiv.org/abs/2610.03316](https://arxiv.org/abs/2610.03316)

    提出了MECo，一个由LLM驱动的多任务进化框架，通过维护任务条件化启发式种群并利用跨任务迁移差距引导启发式的迁移与重组，实现了无需目标问题反馈的零样本跨问题泛化。

    

    为多样化的组合优化问题设计有效的启发式算法需要大量专业知识和反复搜索。大语言模型（LLMs）虽然能够自动化启发式算法的生成与改进，但启发式搜索通常依赖于被优化问题的评估反馈。因此，仅利用源任务的反馈就泛化到新的问题定义，仍然是一个核心挑战。我们提出了MECo，一个由LLM驱动的多任务进化框架，用于实现零样本跨问题泛化。MECo维护任务条件化的启发式种群，并使用基于跨任务种群性能的迁移差距来指导种群间的交互，这些交互实现了启发式算法的迁移与重组。随后，一个互补的选择准则通过奖励每个成员对源组合的额外覆盖，构建出一个紧凑的启发式集合。所选集合无需进一步优化即可直接应用于目标问题。

    arXiv:2610.03316v1 Announce Type: new  Abstract: Designing effective heuristics for diverse combinatorial optimization problems requires substantial expertise and repeated search. Large language models (LLMs) automate heuristic generation and refinement, but heuristic search typically depends on evaluation feedback from the problem being optimized. Generalizing to new problem definitions using only source-task feedback therefore remains a central challenge. We introduce MECo, an LLM-driven multi-task evolutionary framework for zero-shot cross-problem generalization. MECo maintains task-conditioned heuristic populations and uses a transfer gap based on cross-task population performance to guide their interactions. These interactions enable the transfer and recombination of heuristics. A complementary selection criterion then constructs a compact heuristic set by rewarding each member's additional coverage of source combinations. The selected set is applied to target problems without fur
    
[^58]: 面向生产级AI智能体的轻量级、基于评分标准的轨迹评估

    Lightweight, Rubric-Guided Trajectory Evaluation for Production AI Agents

    [https://arxiv.org/abs/2610.03315](https://arxiv.org/abs/2610.03315)

    LiteTrajEval是一种轻量级的、预算受限的AI智能体轨迹评估架构，通过离线规则提取、在线启发式失败信号标记和单个基于评分标准的LLM评判器，显著提升了失败定位与人类标注的一致性（提升20-35个百分点）。

    

    轨迹评估对于提高基于大语言模型（LLM）的智能体的可靠性至关重要，但在生产环境中反复运行的成本很高。现代智能体会生成包含工具调用、观察结果、重试和外部输出的长轨迹，而并非所有原始标记（token）对诊断都同等有用。我们提出了LiteTrajEval，一种用于预算受限轨迹评估的轻量级架构。LiteTrajEval在离线阶段提取紧凑的领域特定规则配置文件，然后在线对每条轨迹进行预处理，标记启发式失败信号，在固定的全局预算下进行序列化，并调用单个基于评分标准的LLM评判器来生成结构化的诊断报告。在公开的Magentic-One风格和τ-bench风格的轨迹数据集上进行评估，与人类标注相比，LiteTrajEval在失败定位一致性方面，在Magentic-One上提升了约20至35个百分点，在τ-retail上提升了高达23个百分点。

    arXiv:2610.03315v1 Announce Type: new  Abstract: Trajectory evaluation is essential for improving the reliability of LLM-based agents, but production use makes it expensive to run repeatedly. Modern agents generate long traces containing tool calls, observations, retries, and external outputs, while not all raw tokens are equally useful for diagnosis. We present \textit{LiteTrajEval}, a lightweight architecture for budget-bounded trajectory evaluation. LiteTrajEval derives compact domain-specific rule profiles offline, then preprocesses each trajectory online, marks heuristic failure signals, serializes it under a fixed global budget, and invokes a single rubric-guided LLM judge to produce structured diagnostic reports. Evaluated on public Magentic-One-style and $\tau$-bench-style trajectory datasets, LiteTrajEval improves failure-localization alignment with human annotations by roughly 20--35 percentage points on Magentic-One and up to 23 percentage points on $\tau$-retail compared wi
    
[^59]: 动态世界中的最优规划

    Optimal Planning in a Dynamic World

    [https://arxiv.org/abs/2610.03312](https://arxiv.org/abs/2610.03312)

    本文定义了“任意开始时间规划”这一新问题设定，解决了可行状态或行动随时间动态变化且执行开始时间未知情况下的最优规划问题。

    

    背景：我们研究的是当可行状态或行动集合随时间变化时的规划问题。例如，在移动障碍物中的路径规划问题（有时称为 SIPP），处于特定位置的可行性会随着障碍物的移动而改变。又如，登上某列特定列车的行动只有在列车停靠在车站时才是可行的。这种动态性意味着最优规划及其持续时间会随着执行开始时间的不同而改变。在实践中，执行开始时间通常在规划完成或另一智能体发出开始指令之前是未知的。然而，大多数现有的规划工作要么忽略动态性，要么假设开始时间是已知的。这虽然使得评估状态和行动的可行性变得简单，但对某些应用场景来说并不切合实际。目标：在本文中，我们放宽了开始时间已知的假设。我们定义了“任意开始时间规划”这一新的问题设定，并提供了……

    arXiv:2610.03312v1 Announce Type: new  Abstract: Background: We address the problem of planning when the set of feasible states or actions changes over time. For example, in the problem of path planning among moving obstacles (sometimes known as SIPP), the feasibility of being at a particular location can change as the obstacles move. Or, the action of boarding a particular train is feasible only while it is stopped at the station. This dynamism means that the optimal plan and its duration can change depending on when execution begins. In practice, execution start time is often unknown until planning has completed or another agent gives the go-ahead. However, most prior planning work either ignores dynamism or assumes a known start time. This makes it straightforward to assess state and action feasibility but is impractical for some applications. Objectives: In this paper, we relax the assumption of a known start time. We define the setting of {\em any-start-time planning} and provide 
    
[^60]: 基于有限步牛顿-舒尔茨正交化的Muon训练损失保证

    Training-Loss Guarantees for Muon with Finite-Step Newton--Schulz Orthogonalization

    [https://arxiv.org/abs/2610.03306](https://arxiv.org/abs/2610.03306)

    本文首次为Muon优化器建立了同时考虑动量累积与有限步调优牛顿-舒尔茨正交化的训练损失保证，证明了在具有正定极限神经切向核的宽两层ReLU网络上，Muon能以高概率达到任意目标损失，命中时间界为 $O((1-\mu)^{-1}\varepsilon^{-1/2})$。

    

    现有的Muon收敛性分析要么假设精确的正交化，要么分析经典的牛顿-舒尔茨（Newton–Schulz）多项式，并且仅保证平稳性，因此Muon的五个经过调优的牛顿-舒尔茨步骤究竟保留了什么，以及这是否足以达到预设的神经网络训练损失，仍是悬而未决的问题。我们建立了一个有限时间的训练保证，同时考虑了正交化之前的动量累积以及经过调优的有限步更新。对于具有固定随机输出权重和正定极限神经切向核的足够宽的两层ReLU网络的全批量训练，我们证明了Muon在初始化上以高概率达到任意目标经验平方损失 $\varepsilon>0$。对于每个动量参数 $\mu\in[0,1)$，采用与 $(1-\mu)\sqrt{\varepsilon}$ 成比例的、依赖于目标的恒定学习率，可以得到 $O((1-\mu)^{-1}\varepsilon^{-1/2})$ 的命中时间界，其余……（摘要原文在此截断）

    arXiv:2610.03306v1 Announce Type: cross  Abstract: Existing convergence analyses of Muon either assume exact orthogonalization or analyze classical Newton--Schulz polynomials, and guarantee only stationarity, so it is unresolved what Muon's five tuned Newton--Schulz steps preserve and whether that suffices to reach a prescribed neural-network training loss. We establish a finite-time training guarantee that accounts for both momentum accumulation before orthogonalization and the tuned finite-step update. For full-batch training of a sufficiently wide two-layer ReLU network with fixed random output weights and a positive-definite limiting neural tangent kernel, we prove that Muon reaches any target empirical squared loss $\varepsilon>0$ with high probability over initialization. For every momentum parameter $\mu\in[0,1)$, a target-dependent constant learning rate proportional to $(1-\mu)\sqrt{\varepsilon}$ yields a hitting-time bound of $O((1-\mu)^{-1}\varepsilon^{-1/2})$, with other pr
    
[^61]: JOVE：面向资源感知LLM任务图的联合执行与验证框架

    JOVE: Joint Execution and Verification for Resource-Aware LLM Task Graphs

    [https://arxiv.org/abs/2610.03296](https://arxiv.org/abs/2610.03296)

    JOVE提出了一种在线框架，通过联合决策LLM执行分配与中间输出的付费验证，在长期预算和延迟约束下平衡即时执行开销与未来学习收益，从而在LLM服务质量未知的情况下提升任务图执行的效率与正确性。

    

    复杂推理查询可以被分解为有向无环任务图，并分发到异构的大语言模型（LLM）上执行，通过并行化降低延迟，并使较小的模型也能解决复杂任务。然而在实践中，某个LLM是否适合给定的子任务可能是先验未知的，而且仅凭执行本身无法揭示输出的正确性。我们提出了JOVE，一个在线框架，它联合地为子任务分配执行LLM，并选择中间输出进行付费验证。验证以异步方式运行，并被用于改进未来的分配决策，因此系统必须在“当下的执行开销”与“为未来而学习”之间取得平衡。我们研究了在长期预算和单查询延迟约束下如何优化这一权衡，其中LLM的服务质量、调用成本和执行时间均是随机的且初始未知。JOVE通过求解一系列单查询混合整数线性规划来做出执行与验证决策。

    arXiv:2610.03296v1 Announce Type: new  Abstract: Complex reasoning queries can be decomposed into directed acyclic task graphs and distributed across heterogeneous LLMs, reducing latency through parallelism and enabling smaller models to solve complex tasks. In practice, however, the suitability of an LLM for a given subtask may be a priori unknown, and execution alone does not reveal output correctness. We propose JOVE, an online framework that jointly assigns executor LLMs and selects intermediate outputs for paid verification. Verification runs asynchronously and is used to improve future allocations, so the system must balance spending on execution now against learning for later. We study how to optimize this trade-off under a long-term budget and a per-query latency constraint, with stochastic, initially unknown LLM service quality, invocation costs, and execution times. JOVE makes execution and verification decisions by solving a sequence of per-query mixed-integer linear program
    
[^62]: EVOL：面向免部署学习路径推荐的仿真器引导式进化专家合成

    EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation

    [https://arxiv.org/abs/2610.03273](https://arxiv.org/abs/2610.03273)

    该论文提出EVOL框架，利用知识追踪仿真器通过进化搜索为每个学习者合成专家示范，并将其蒸馏为免部署的前馈策略，从而同时解决了学习路径推荐强化学习中的超指数组合搜索空间与稀疏奖励两大难题。

    

    用于学习路径推荐（LPR）的强化学习（RL）面临两个相互耦合的障碍。首先，策略必须在不具备中间反馈的情况下一次性确定由L个概念组成的序列，由此产生的组合搜索空间随L呈超指数增长，且仅在最后一步才提供奖励。其次，专家学习路径本是解决稀疏奖励强化学习的天然良方，但教育数据中并不存在此类专家路径，因为学生日志记录的是学习者实际做了什么，而非他们本应该做什么。我们通过借鉴机器人学中基于仿真器的示范学习方法来同时应对这两个障碍：知识追踪仿真器既被用于通过进化搜索为每个学习者合成专家示范，也被用于训练一个免部署的策略，将这些示范蒸馏到前馈学习者中。我们的框架EVOL以非对称的演员-评论家结构实现了这一流程，其中演员负责部署时的决策。

    arXiv:2610.03273v1 Announce Type: new  Abstract: Reinforcement learning (RL) for learning path recommendation (LPR) faces two coupled obstacles. First, the policy must commit to a sequence of L concepts without intermediate feedback, producing a combinatorial search space that grows super-exponentially with L and provides reward only at the final step. Second, expert learning paths would be the natural cure for sparse-reward RL, but they do not exist in educational data, because student logs record what learners did, not what they should have done. We address both obstacles by importing a recipe from simulator-based demonstration learning in robotics: the knowledge tracing simulator is used both to synthesize per-learner expert demonstrations through evolutionary search and to train a deployment-free policy that distills these demonstrations into a feed-forward learner. Our framework, EVOL, instantiates this pipeline with an asymmetric actor-critic where the actor commits to deployment
    
[^63]: SPEAR：面向大规模偏微分方程预训练的谱解耦混合专家神经算子与知识引导专家聚合

    SPEAR: A Spectral-Disentangled MoE Neural Operator with Knowledge-Guided Expert Aggregation for Large-Scale PDE Pretraining

    [https://arxiv.org/abs/2610.03265](https://arxiv.org/abs/2610.03265)

    SPEAR通过将特征谱解耦为低频与高频分量以实现共享与专门化建模，并结合基于数据集知识与路由偏好的知识引导专家聚合策略来消除专家冗余，有效解决了PDE基础模型中的知识干扰与专家冗余问题，提升了大规模PDE预训练的泛化能力。

    

    大规模预训练提升了神经算子在多种偏微分方程（PDE）上的泛化能力。然而，现有的PDE基础模型在应对异质动力学时仍存在困难：共享表示可能引发知识干扰，而混合专家（MoE）架构则面临专家冗余日益严重的问题。我们提出了SPEAR，一个用于大规模PDE预训练、具有知识引导专家聚合机制的谱解耦MoE神经算子。SPEAR将潜在特征解耦为低频与高频分量，从而实现对可迁移动力学的共享建模，以及对PDE特定模式的专门化学习。为解决专家冗余问题，我们设计了一种知识引导的专家聚合策略，该策略基于数据集特定的已学习知识和路由偏好来度量专家之间的相似性，从而实现对相似专家的识别与合并。在十二个PDE数据集及多个下游任务上的实验验证了该方法的有效性。

    arXiv:2610.03265v1 Announce Type: cross  Abstract: Large-scale pre-training has improved the generalization of neural operators across diverse PDEs. However, existing PDE foundation models still struggle with heterogeneous dynamics, where shared representations may cause knowledge interference, while mixture-of-experts (MoE) architectures suffer from increasing expert redundancy. We propose SPEAR, a spectral-disentangled MoE neural operator with knowledge-guided expert aggregation for large-scale PDE pre-training. SPEAR decouples latent features into low- and high-frequency components, enabling shared modeling of transferable dynamics and specialized learning of PDE-specific patterns. To address expert redundancy, we design a knowledge-guided expert aggregation strategy that measures expert similarity from dataset-specific learned knowledge and routing preferences, enabling the identification and consolidation of similar experts. Experiments on twelve PDE datasets and multiple downstre
    
[^64]: 用于不可观测图像结构扩散恢复的连续后验融合

    Consecutive Posterior Fusion for Diffusive Recovery of Unobservable Image Structures

    [https://arxiv.org/abs/2610.03261](https://arxiv.org/abs/2610.03261)

    提出CPF-DDNM推理时策略，通过融合连续的感知测量后验估计来改进扩散模型对不可观测图像结构的恢复，且无需重新训练或额外的去噪器评估。

    

    求解严重不适定的成像逆问题需要恢复那些不可观测或受测量弱约束的图像结构。扩散模型为推断这类缺失信息提供了富有表现力的学习先验，而后验采样则在反向过程中引入测量一致性约束。然而，标准的扩散后验采样器依赖于瞬时的感知测量估计，没有显式利用先前后验修正步骤所携带的信息。我们提出了连续后验融合去噪扩散零空间模型（CPF-DDNM），这是一种推理时策略，通过融合连续的感知测量估计来改进不可观测图像结构的扩散恢复，且无需重新训练或额外的去噪器评估。我们在DDNM框架内实例化了这一原则，其值域/零空间分解表明，连续融合能够保留……

    arXiv:2610.03261v1 Announce Type: cross  Abstract: Solving severely ill-posed imaging inverse problems requires recovering image structures that are unobservable or weakly constrained by the measurements. Diffusion models provide expressive learned priors for inferring such missing information, while posterior sampling incorporates measurement consistency along the reverse process. Standard diffusion posterior samplers, however, rely on instantaneous measurement-aware estimates, without explicitly exploiting information carried by previous posterior corrections.   We introduce Consecutive Posterior Fusion Denoising Diffusion Null-Space Models (CPF-DDNM), an inference-time strategy that fuses consecutive measurement-aware estimates to improve the diffusive recovery of unobservable image structures, without requiring retraining or additional denoiser evaluations. We instantiate this principle within DDNM, whose range/null-space decomposition reveals that consecutive fusion preserves the 
    
[^65]: 绘制并推进非线性因果发现的可扩展性-准确性前沿

    Mapping and Advancing the Scalability-Accuracy Frontier of Nonlinear Causal Discovery

    [https://arxiv.org/abs/2610.03258](https://arxiv.org/abs/2610.03258)

    本文系统比较了四类非线性因果发现方法在可扩展性与准确性上的互补瓶颈，并提出基于样条的得分评估方案SPADE，通过一次性编译并复用充分统计量，在保持准确性的同时大幅提升组合搜索的效率。

    

    可扩展的非线性因果发现需要将灵活的机制估计器与对大型图空间的高效搜索相结合的方法。学界已提出多种算法家族来应对这一挑战，但它们的准确性-运行时间权衡仍缺乏深入理解。我们实证比较了四种主要方法：可微结构学习、摊销结构学习、得分匹配和组合搜索。我们的结果揭示了互补的瓶颈：可微方法和摊销方法具有良好的可扩展性，但存在准确性差距；得分匹配方法在低维情况下可以较为准确，但随着特征维度增加会迅速退化；组合搜索方法保持准确，但因重复且冗余的局部评分而速度受限。基于这一瓶颈，我们开发了SPADE，一种基于样条的得分评估方案，它一次性编译充分统计量，并在整个组合搜索过程中重复使用。

    arXiv:2610.03258v1 Announce Type: cross  Abstract: Scalable nonlinear causal discovery requires methods that combine flexible mechanism estimators with efficient search over large graph spaces. Several algorithmic families have been proposed to address this challenge, yet their accuracy-runtime trade-offs remain poorly understood. We empirically compare the four major approaches: differentiable structure learning, amortized structure learning, score-matching, and combinatorial search. Our results reveal complementary bottlenecks: differentiable and amortized methods scale well but exhibit an accuracy gap, score-matching methods can be accurate in low dimensions but degrade quickly for increasing feature sizes, and combinatorial methods remain accurate but are slowed by repeated and redundant local scoring. Motivated by this bottleneck, we develop SPADE, a spline-based score-evaluation scheme that compiles sufficient statistics once and reuses them throughout combinatorial search. Under
    
[^66]: 学会一个事实不等于学会如何提取它

    Learning a Fact Is Not Learning How to Retrieve It

    [https://arxiv.org/abs/2610.03251](https://arxiv.org/abs/2610.03251)

    该研究通过两阶段训练实验发现，“掌握事实知识”与“掌握如何提取该事实”是两种可分离的能力——模型可以在尚未真正学会某些事实之前，就先通过特定的请求形式学会提取方式。

    

    arXiv:2610.03251v1 公告类型：新论文 摘要：一个在“X的首都是Y”这类句子上训练的模型，可能在“X的首都是”之后生成"Y"，但在“X的首都：”之后却会失败。我们将引出同一事实的不同表达方式称为“请求形式”。为了将“学习一个事实”与“提取这个事实”区分开来，我们分两个阶段训练了两个模型。在第一阶段（请求形式训练）中，一个模型以五种形式接触每个事实，而另一个模型仅以陈述句形式接触相同的事实。在第二阶段（目标事实训练）中，两个模型接受完全相同的新事实训练，且全部以陈述句形式呈现。随后，两个模型从陈述句中提取新事实的能力几乎同样好，但在其他请求形式上却表现出显著差异。因此，模型可以在学习事实之前，先通过某种请求形式学会如何提取。为了理解这种差异，我们考察了答案生成前一刻的隐藏状态，我们称之为“上下文状态”。当用两种不同的请求形式请求同一事实时，第一阶段以五种形式训练的模型……

    arXiv:2610.03251v1 Announce Type: new  Abstract: A model trained on "The capital of X is Y" may produce "Y" after "The capital of X is" but fail after "The capital of X:". We call these different ways of eliciting the same fact request forms. To separate learning a fact from retrieving it, we train two models in two stages. In the first stage (request-form training), one model sees each fact in five forms and the other sees the same facts only as statements. In the second stage (target-fact training), both receive identical training on new facts, all as statements. Both then retrieve the new facts almost equally well from statements, but differ sharply on other request forms. Thus, a model can learn how to retrieve through a request form before it learns the facts. To understand this difference, we examine the hidden state immediately before the answer, which we call the context state. When given two different request forms for the same fact, the model trained on five forms in stage on
    
[^67]: WAMpy：用Python高效合成Prolog程序

    WAMpy: Efficient Synthesis of Prolog Programs in Python

    [https://arxiv.org/abs/2610.03234](https://arxiv.org/abs/2610.03234)

    WAMpy是一个Python框架，通过将Prolog子句编译为基于NumPy数组的WAM指令并结合Numba JIT加速，大幅提升了在Python中反复合成与评估小型Prolog候选程序的工作负载的端到端性能。

    

    我们提出了WAMpy，一个专为Prolog程序合成而优化的Python框架。与通用Prolog系统不同，WAMpy面向需要反复生成和评估小型候选程序的工作负载。WAMpy将Prolog子句编译为基于NumPy数组的WAM（Warren抽象机）指令，并支持在固定背景知识下对假设进行部分重编译。性能关键的例程使用Numba即时（JIT）编译进行加速。在重复编译与评估工作负载的基准测试中，与通过Janus从Python调用SWI-Prolog的方式相比，WAMpy显著提升了端到端性能。

    arXiv:2610.03234v1 Announce Type: cross  Abstract: We present WAMpy, a Python framework optimized for synthesizing Prolog programs. Unlike general-purpose Prolog systems, WAMpy targets workloads that repeatedly generate and evaluate small candidate programs. WAMpy compiles Prolog clauses into NumPy array-based WAM instructions and supports partial recompilation of hypotheses against fixed background knowledge. Performance-critical routines are accelerated using Numba just-in-time (JIT) compilation. In a benchmark of repeated compilation-and-evaluation workloads, WAMpy improves end-to-end performance compared with SWI-Prolog accessed from Python using Janus.
    
[^68]: D2K-Bench：LLM 智能体能否将专家设计转化为高效的 GPU 内核？

    D2K-Bench: Can LLM Agents Turn Expert Designs into Efficient GPU Kernels?

    [https://arxiv.org/abs/2610.03226](https://arxiv.org/abs/2610.03226)

    D2K-Bench 是一个包含 26 个任务和 85 个工作负载的诊断性基准，通过分层专家设计指导（算法洞察、数据流设计与底层优化技巧）来系统评估 LLM 智能体生成高效 GPU 内核的能力，结果显示专家指导可将正确率从 93.1% 提升至 98.5% 并显著提高性能。

    

    由大语言模型（LLM）智能体生成的 GPU 内核，其效率可能仍低于专家实现，但仅凭运行时间并不能揭示这一差距与设计发现和实现之间的关系。我们提出了 D2K-Bench，一个包含 26 个任务和 85 个工作负载的诊断性基准，用于衡量智能体将专家设计指导转化为高效 GPU 内核的能力。这些指导涵盖三个层级：L1 为高层算法洞察，L2 为数据流设计，L3 为底层优化技巧，并包含这些层级之间的依赖关系。有指导与无指导的成对运行共享相同的任务描述、工作负载、工具、硬件以及 350 轮的交互预算。补充性评估则考察智能体独立提出的设计，以及生成代码中实际实现的设计属性。在 NVIDIA B200 GPU 上对五个模型的评估中，专家指导将 130 个模型-任务对上的正确率从 93.1% 提升至 98.5%，并在全部 26 个任务上提升了性能得分……（原文摘要在此处截断）

    arXiv:2610.03226v1 Announce Type: cross  Abstract: GPU kernels generated by large language model (LLM) agents can remain less efficient than expert implementations, but runtime alone does not reveal how the gap relates to design discovery and implementation. We introduce D2K-Bench, a diagnostic benchmark of 26 tasks and 85 workloads that measures how effectively agents translate expert design guidance into efficient GPU kernels. The guidance covers L1: high-level algorithmic insights, L2: dataflow design, and L3: low-level optimization tricks, including dependencies among these levels. Pairwise runs with and without guidance share task descriptions, workloads, tools, hardware, and a 350-turn budget. Complementary assessments examine independently proposed designs and the design properties implemented in generated code. Across five models on NVIDIA B200 GPUs, guidance raises correctness over 130 model-task pairs from 93.1% to 98.5% and increases the Performance Score over all 26 tasks f
    
[^69]: 不确定性作为基于扩散模型的医学图像合成中语义正确性的代理指标

    Uncertainty as a Proxy for Semantic Correctness in Diffusion-Based Medical Image Synthesis

    [https://arxiv.org/abs/2610.03224](https://arxiv.org/abs/2610.03224)

    本研究提出以不确定性作为扩散模型医学图像合成语义正确性的代理指标，并利用多任务扩散框架AortaDiff的分割误差作为定量度量来验证该方法的有效性。

    

    扩散模型可以从非对比增强CT（NCCT）合成对比增强CT（CECT），从而避免对比剂的使用及其带来的环境和患者可及性成本。然而，视觉上逼真的图像并不一定在解剖学上是正确的，而用于评估生成质量的像素强度和特征空间相似性指标并不能直接衡量解剖结构的正确性。在本研究中，我们探讨了不确定性能否作为基于扩散模型的医学图像合成中语义正确性的代理指标。我们使用AortaDiff研究NCCT到CECT的合成，AortaDiff是一个多任务扩散框架，可同时生成CECT图像和管腔分割结果。分割输出为生成的血管解剖结构提供了显式表示，使得基于分割的误差可以用作生成正确性的定量度量。六种方法涵盖了权重层面（Ensemble、HyperDiff、BayesDiff）和架构扰动层面的不确定性估计技术。

    arXiv:2610.03224v1 Announce Type: cross  Abstract: Diffusion models can synthesise contrast-enhanced CT (CECT) from non-contrast CT (NCCT), avoiding contrast administration and its environmental and patient-access costs. However, visually realistic images are not necessarily anatomically correct, and the pixel-intensity and feature-space similarity metrics used to assess generation quality do not directly measure anatomical correctness. In this work, we investigate whether uncertainty can serve as a proxy for semantic correctness in diffusion-based medical image synthesis.   We study NCCT-to-CECT synthesis using AortaDiff, a multitask diffusion framework that jointly generates CECT images and lumen segmentations. The segmentation output provides an explicit representation of the generated vascular anatomy, enabling segmentation-derived errors to be used as a quantitative measure of generation correctness. Six methods spanning weight (Ensemble, HyperDiff, BayesDiff), architecture-pertur
    
[^70]: 面向图像分类的混合量子-经典架构演化

    Evolving Hybrid Quantum-Classical Architectures for Image Classification

    [https://arxiv.org/abs/2610.03220](https://arxiv.org/abs/2610.03220)

    该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。

    

    混合量子-经典神经网络将参数化量子电路（PQC）与成熟的深度学习架构相结合，但其性能在很大程度上取决于量子电路架构的选择，而这一选择目前仍主要依赖人工完成。现有的大多数方法依赖于手工设计或固定的电路拟设，需要预先指定电路结构、门组合和量子比特连接方式，且无法保证这些设计适合特定任务。这一限制在图像分类任务中尤为突出，因为量子电路既要对经典网络提取的特征进行变换，又要保持足够紧凑以便于实际训练，而通用的、与任务无关的拟设很难同时满足这些要求。我们将EXAQC——一个用于自动化量子电路发现的演化框架——扩展到图像分类任务。EXAQC将参数化量子电路作为中间处理模块进行演化，同时保留经典……（摘要原文在此处截断）

    arXiv:2610.03220v1 Announce Type: cross  Abstract: Hybrid quantum classical neural networks integrate parameterized quantum circuits (PQCs) with established deep learning architectures, but their performance depends strongly on the choice of quantum circuit architecture, a choice that remains largely manual. Most existing approaches rely on hand-designed or fixed circuit ans\"atze, requiring circuit structure, gate composition, and qubit connectivity to be specified in advance with no guarantee that they suit the task. This limitation is especially acute in image classification, where quantum circuits must transform features extracted by classical networks while remaining compact enough for practical training, requirements that generic, task-agnostic ans\"atze are unlikely to satisfy simultaneously. We extend EXAQC, an evolutionary framework for automated quantum circuit discovery, to image classification. EXAQC evolves PQCs as intermediate processing modules while retaining classical 
    
[^71]: 迈向基于小型语言模型（SLM）的智能体任务-工具意图匹配

    Toward SLM-based agentic task-tool intent matching

    [https://arxiv.org/abs/2610.03213](https://arxiv.org/abs/2610.03213)

    本文提出利用小型语言模型（SLM）作为任务-工具相关性分类器，对智能体的每一次工具调用进行逐次意图验证，以判断调用是否真正服务于任务意图，从而实现低延迟或本地化部署的智能体行为监督。

    

    装备工具的AI智能体通过工具调用来访问数据并对外部系统执行操作。智能体系统的横向扩展增加了此类交互的数量，进一步催生了对能够在低延迟和/或本地部署环境下运行的自动化、逐次调用监督的需求。传统的授权方案只能判断智能体是否被允许调用某个工具，却无法评估智能体的底层认知，即工具的选择是否构成满足任务意图的一个合乎逻辑且相关的步骤。因此，即使某个调用获得了授权，它仍可能偏离任务意图：恶意智能体可能使调用发生偏移，或诱导其他智能体做出与任务意图不一致的调用组合。因此，每一次调用都需要被验证。在本研究中，我们探讨了小型语言模型在这一目标上的适用性：由一个小型语言模型充当任务-工具相关性分类器（原文在此处截断）。

    arXiv:2610.03213v1 Announce Type: new  Abstract: Tool-equipped AI agents use tool calls to access data and act on external systems. Horizontal growth of agentic systems increases the number of these interactions, and further motivates the need for automated, per-call oversight that can operate at low latency and/or on-prem. Conventional authorization schemes can determine whether an agent is allowed to invoke a tool, but cannot assess the agent's underlying cognition, specifically, whether the tool selection represents a logical, relevant step toward satisfying the intent of the task or not. Consequently, an allowed call may still deviate from the task's intent: a rogue agent might deviate the calls or nudge other agents to make a combination of calls that would not align with the intent of the task. Therefore, every call needs to be verified. In this study we investigate the applicability of Small Language Models (SLMs) to this purpose: an SLM functions as a task-tool relevance classi
    
[^72]: 上下文流匹配：面向高效视觉生成的流模型自适应步数选择

    Contextual Flow Matching: Adaptive Step Selection in Flow Models for Efficient Visual Generation

    [https://arxiv.org/abs/2610.03202](https://arxiv.org/abs/2610.03202)

    提出COFLOW推理时方法，根据提示特征自适应选择每步生成的采样步数，即插即用且无需重新训练模型，在图像和视频生成中实现超过2.5倍加速并保持感知与语义质量。

    

    流匹配通过连续时间动力学实现了高质量视觉生成，但由于需要多次顺序函数评估，推理成本依然高昂。现有的加速方法虽然减少了函数评估次数，但往往会引入额外的训练开销、降低生成质量，或未能考虑输入相关的差异性。我们提出了COFLOW，这是一种推理时方法，能够根据提示特征为每次生成自适应地选择步数。我们的上下文感知COFLOW通过无监督奖励在线训练，该奖励平衡了推理效率与生成保真度。我们的方法即插即用，无需对底层生成模型进行重新训练。该方法可推广至图像和视频生成，在保持感知与语义质量的同时实现了超过2.5倍的加速。我们进一步提供了理论分析，在标准假设下建立了O(1/K)的前向欧拉离散化误差界。

    arXiv:2610.03202v1 Announce Type: cross  Abstract: Flow Matching enables high-quality visual generation via continuous-time dynamics, but inference remains costly due to multiple sequential function evaluations. Existing acceleration methods reduce the number of function evaluations but often introduce additional training overhead, degrade quality, or fail to account for input-dependent variability. We propose COFLOW, an inference-time method that adaptively selects the step counts each generation based on the prompt features. Our context-aware COFLOW is trained online with an unsupervised reward that balances inference efficiency and generation fidelity. Our method is plug-and-play, requiring no retraining of the underlying generative model. It generalizes to image and video generation, achieving over 2.5x speedup while preserving perceptual and semantic quality. We further provide a theoretical analysis establishing an O(1/K) forward-Euler discretization error bound under standard re
    
[^73]: KV²：一种自我精炼的KV缓存

    KV$^2$: A Self-Refining KV Cache

    [https://arxiv.org/abs/2610.03198](https://arxiv.org/abs/2610.03198)

    KV²提出了一种基于选择性重建的查询无关KV缓存压缩方法，先用轻量级代理评分器筛选出信息丰富的token，再仅对该子集进行精细重建评分以计算淘汰分数，在极低缓存预算下比次优基线提升超过40个百分点。

    

    键值（KV）缓存的内存占用限制了长上下文模型的实际应用，并且当一个预填充的上下文需要后续服务多个不同查询时，KV缓存成为成本的主要来源。在这种可复用场景中，与查询无关的压缩需要在成本与质量之间权衡：轻量级估计器虽然廉价但准确性较低，而全上下文重建评分虽然更准确，却需要重新处理整个提示词。我们提出了KV²，一种基于选择性重建的与查询无关的KV缓存压缩方法。KV²首先使用轻量级代理评分器识别上下文中信息丰富的token，然后仅对这个子集进行重新处理以计算最终的淘汰分数。在RULER、大海捞针（Needle-in-a-Haystack）和LongBench基准上，KV²相对于基线的优势随着缓存预算收紧而扩大：在RULER 16K上，当KV缓存预算仅为2%时，其平均分数比次优基线提高了超过40个百分点；在LongBench上也取得了（摘要在此处截断）……

    arXiv:2610.03198v1 Announce Type: new  Abstract: The memory footprint of the key-value (KV) cache constrains the practical use of long-context models, and it dominates cost when one prefilled context must later serve many different queries. In this reusable setting, query-agnostic compression trades cost against quality: lightweight estimators are cheap but less accurate, whereas full-context reconstruction scoring is more accurate yet reprocesses the entire prompt. We introduce KV$^2$, a query-agnostic KV-cache compression method based on selective reconstruction. KV$^2$ first uses a lightweight proxy scorer to identify informative in-context tokens, then reprocesses only this subset to compute final eviction scores. On RULER, Needle-in-a-Haystack, and LongBench, KV$^2$'s margin over baselines widens as the budget tightens: on RULER 16K at a 2% KV-cache budget it improves the average score over the next-best baseline by more than 40 percentage points, and on LongBench it attains the h
    
[^74]: 直到证据说了算：教会LLM调查员何时结案

    Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case

    [https://arxiv.org/abs/2610.03190](https://arxiv.org/abs/2610.03190)

    该论文研究了LLM调查员“何时应结案”的证据充分性判断问题，发现未经训练的小模型和前沿模型都普遍夸大证据充分性而过早结案，并提出需对照来源捷径规则来评估结案能力的方法。

    

    事故、缺陷和故障调查以一个普通问答从不面对的决定告终：目前收集到的证据是否足以结案。我们研究LLM调查员如何做出这一决定：它们从案卷中请求证据、修正假设，要么以基于所读内容的结论结案，要么保持案件开放并指出尚缺什么。这种判断能力并非与生俱来：一个未经训练的9B模型在97%的答案中夸大了其证据，而一个能在84%的案例中识别出正确原因的前沿模型仍在91%的情况下夸大证据，并在41个官方结论为“原因未定”的案例中结案了17个。衡量这种判断也并非易事：案件的来源在很大程度上预测了其标签，一个仅读取来源的规则在我们的测试用例上即可达到83.0的平衡准确率。因此，我们通过三项测试来评估结案：结案准确率（对照该规则报告）……

    arXiv:2610.03190v1 Announce Type: cross  Abstract: Accident, defect and outage investigations end with a decision that ordinary question answering never faces: whether the evidence gathered so far is enough to close the case. We study this decision for LLM investigators, which request evidence from a case file, revise their hypotheses, and either close the case with a conclusion grounded in what they read or leave it open and name what is missing. This judgment does not come with capability: an untrained 9B model overstates its evidence in 97% of its answers, and a frontier model that identifies the right cause in 84% of cases still overstates in 91% and closes 17 of the 41 cases whose official finding is "cause undetermined". Measuring it is also non-trivial: the source of a case largely predicts its label, and a rule that reads only the source reaches 83.0 balanced accuracy on our test cases. We therefore evaluate closure with three tests: closure accuracy, reported against this rule
    
[^75]: 在线策略蒸馏中的收益与坍塌：强化学习视角

    Gains and Collapse in On-Policy Distillation:A Reinforcement Learning Perspective

    [https://arxiv.org/abs/2610.03185](https://arxiv.org/abs/2610.03185)

    该论文从强化学习视角揭示了在线策略蒸馏（OPD）既提升性能也可能坍塌的机制——教师模型的隐式奖励在可靠时促进正确响应采样，在偏好与质量错位时引发奖励破解并放大冗长重复生成，且OPD提升性能但不扩展学生模型的能力边界。

    

    在线策略蒸馏（OPD）已成为语言模型后训练的重要方法。然而，尽管OPD能带来性能提升，它也可能坍塌为过度冗长和重复的生成，而这些截然不同结果背后的机制仍鲜为人知。我们从强化学习的视角来解释这些结果：教师模型会隐式地奖励学生模型的行为，即使是教师模型自己很少表现出的行为。从这一视角出发，我们的实验表明，OPD在不扩展学生模型能力的情况下提升了性能。当隐式奖励模型可靠时，OPD使正确响应更容易被采样到；相反，当偏好与质量不一致时，就会发生奖励破解（reward hacking）：隐式奖励模型会放大学生生成的冗长、重复的输出，即使教师模型自身很少生成此类文本。基于这一诊断，我们发现通过在训练中屏蔽不健康的响应等方法可以缓解坍塌问题。

    arXiv:2610.03185v1 Announce Type: new  Abstract: On-policy distillation (OPD) has become an important approach to language model post-training. However, despite its performance gains, OPD can also collapse into excessively long and repetitive generation, and the mechanism underlying these divergent outcomes remains poorly understood. We explain these outcomes through a reinforcement learning perspective: the teacher implicitly rewards student behaviors, even those it rarely exhibits itself. From this perspective, our experiments show that OPD improves performance without expanding the student's capabilities. When the implicit reward model is reliable, OPD makes correct responses easier to sample. In contrast, when the preference misaligns with quality, reward hacking happens: the implicit reward model amplifies overlong, repetitive student rollouts, even though it rarely generates such text itself. Guided by this diagnosis, we find that masking unhealthy responses during training and u
    
[^76]: LiBRA：通过双向隐空间优化实现检测感知的图像水印移除

    LiBRA: Detection-Aware Image Watermark Removal via Bidirectional Latent Optimization

    [https://arxiv.org/abs/2610.03166](https://arxiv.org/abs/2610.03166)

    该论文提出LiBRA方法，通过双向隐空间优化调整图像以使水印不可检测，同时避免过度优化导致的图像质量下降，从而在移除水印与保持画质之间取得平衡。

    

    数字水印为AI生成的图像提供来源归属支持，但其可靠性取决于对移除攻击的抵抗能力。一些攻击试图通过强制解码出的水印与原始水印不同来移除水印。然而，这种方式可能产生一个仍然可被检测到的反转水印，导致移除失败；而进一步改变水印的尝试则可能不必要地损害图像质量。为了解决这些局限性，我们提出了LiBRA（隐空间带内双向移除攻击），其目标是在保持图像质量的同时使水印变得不可检测。LiBRA不再持续将水印推向反转，而是通过调整图像来隐藏水印，避免引入可能降低图像质量的额外改变。一些攻击即使在水印仍然可检测且图像质量已受损的情况下，仍持续将解码比特推离原始水印。

    arXiv:2610.03166v1 Announce Type: cross  Abstract: Digital watermarking supports source attribution for AI-generated images, but its reliability depends on resistance to removal attacks. Some attacks attempt to remove watermarks by forcing the decoded watermark to differ from the original. However, this can produce an inverted watermark that remains detectable, causing removal to fail, while further attempts to alter the watermark may unnecessarily degrade image quality. To address these limitations, we present LiBRA (Latent In-band Bidirectional Removal Attack), which aims to make watermarks undetectable while preserving image quality. Instead of continually pushing the watermark toward inversion, LiBRA adjusts the image to conceal the watermark without encouraging further changes that could degrade image quality. Some attacks keep pushing decoded bits away from the original watermark, even when further changes preserve detectability and damage image quality. With access to the waterm
    
[^77]: 预测转向向量与适配器权重以实现少样本作者风格迁移

    Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer

    [https://arxiv.org/abs/2610.03163](https://arxiv.org/abs/2610.03163)

    该论文针对少样本作者风格迁移任务提出三种方法——对比激活转向、转向向量预测网络和预测LoRA适配器的超网络，并发现超网络在风格模仿与输出质量之间取得了最佳权衡，且能泛化到未见过的作者。

    

    仅凭少量示例将大语言模型适配到某个作者的个人风格具有挑战性，而科学写作更加剧了这一难度：正式的写作规范使得表面文字变化有限，且作者撰写的是自己关注的主题，因此提取出的“风格”很容易与内容纠缠在一起。我们研究了基于每位作者少量示例摘要的风格条件摘要生成任务，并提出三种方法：（1）对比激活转向，（2）预测转向向量的网络，以及（3）预测 LoRA 适配器的超网络。我们发现风格模仿与输出质量之间存在一致的权衡：微调能够获取大部分可用的风格信号，但会牺牲流畅性，而超网络在已见和未见作者上均实现了最佳权衡。我们的转向方法在作者级别运作，将某位作者的摘要与相同内容的风格中性生成结果进行对比，这固定了主题，从而消除了……（原文在此处截断）

    arXiv:2610.03163v1 Announce Type: cross  Abstract: Adapting large language models to an individual author's style from a few examples is challenging, and scientific writing sharpens the difficulty: formal conventions leave little surface variation, and authors write about their own topics, so extracted ``style'' easily entangles with content. We study style-conditioned abstract generation from a few example abstracts per author and propose three methods: (1) contrastive activation steering, (2) a network that predicts steering vectors, and (3) a hypernetwork that predicts LoRA adapters. We find a consistent trade-off between style imitation and output quality: fine-tuning buys most of the available style signal but forfeits fluency, while the hypernetwork achieves the best trade-off on both seen and unseen authors. Our steering operates at author level, contrasting an author's abstracts against style-neutral generations for the same content. This holds topic fixed, removes the need for
    
[^78]: 基于多模态推理从跨病毒科的无标记人类B细胞库中发现广谱中和抗体

    Multimodal reasoning for broadly neutralizing antibody discovery from label-free human B cell repertoires across virus families

    [https://arxiv.org/abs/2610.03160](https://arxiv.org/abs/2610.03160)

    ImmuneAgent是一个整合多模态推理、持续元学习与湿实验反馈的闭环AI系统，能从无标记的人类天然B细胞库中高效发现广谱中和抗体，实现约55%的中和抗体发现率和约11%的bnAb产出率，显著优于现有计算方法。

    

    从人类天然免疫库中发现广谱中和抗体仍然是免疫学中的一项根本性挑战，其受到以下因素阻碍：广谱中和抗体极其稀有、对其跨病原体细胞起源的理解不完整、以及现有计算工具无法泛化应对新出现的病毒威胁。在此我们提出ImmuneAgent，一个将多模态推理与持续元学习和湿实验反馈相结合的闭环AI系统，以克服这些障碍。将该系统应用于筛选来自疫苗接种或感染人群的天然BCR库，该系统实现了约55%的中和抗体发现率（110个克隆候选物中有60个）和约11%的广谱中和抗体产出率（110个中有12个），在相同克隆预算下显著优于最先进的基于序列的中和预测器或共折叠模型。五个由ImmuneAgent发现的抗体在体内提供了100%的保护。

    arXiv:2610.03160v1 Announce Type: cross  Abstract: Discovering broadly neutralizing antibodies (bnAbs) from human natural immune repertoires remains a fundamental challenge in immunology, hindered by: the extreme rarity of bnAb, incomplete understanding of their cellular origins across pathogens, and the inability of existing computational tools to generalize across emerging viral threats. Here we present ImmuneAgent, a closed-loop AI system that integrates multimodal reasoning with continual meta-learning and wet-lab feedback to overcome these barriers. Applied to screen the natural BCR repertoires from vaccinated or infected cohorts, the system achieves a ~55% neutralization antibody discovery rate (60 of 110 cloned candidates) and a ~11% bnAb yield (12 of 110), substantially outperforming a state-of-the-art sequence-based neutralization predictor or cofolding models evaluated at the same cloning budget. Five ImmuneAgent-discovered antibodies conferred 100% in vivo protection against
    
[^79]: EvoRiskBench：面向工作区智能体运行时安全风险的可演化基准测试

    EvoRiskBench: An Evolving Benchmark for Runtime Security Risks in Workspace Agents

    [https://arxiv.org/abs/2610.03153](https://arxiv.org/abs/2610.03153)

    提出了EvoRiskBench——一个基于EP-Path-EF框架的可演化安全基准测试，通过自动化端到端工作流在隔离环境中构建、执行并独立验证工作区智能体的运行时安全风险案例。

    

    工作区智能体将大语言模型与执行框架相结合，以执行访问或修改外部资源的有状态多步骤任务。现有基准测试在运行时安全风险的可执行覆盖方面存在缺口，而不断演进的模型能力、执行框架、工具和威胁也推动了基准测试自身的演进。我们提出了EvoRiskBench，一个围绕EP-Path-EF框架组织的可演化基准测试，该框架通过智能体介导的风险路径将初始风险入口点与单跳技术效果关联起来。该框架定义了九个入口点类别和五个效果类别；一项由20名参与者开展的研究验证了该框架在代表性案例上的可解释性和分类一致性。在该框架的指导下，一个自动化的端到端工作流在隔离环境中构建并执行风险案例，并利用运行时追踪和环境状态独立验证执行结果。该基准测试提供了一个可……

    arXiv:2610.03153v1 Announce Type: cross  Abstract: Workspace agents combine large language models with execution harnesses to perform stateful, multi-step tasks that access or modify external resources. Existing benchmarks leave gaps in executable coverage of their runtime security risks, while evolving model capabilities, harnesses, tools, and threats motivate benchmark evolution. We introduce EvoRiskBench, an evolving benchmark organized around the EP-Path-EF framework, which links an initial risk entry point to a one-hop technical effect through an agent-mediated risk path. The framework defines nine entry-point categories and five effect categories; a 20-participant study supports their interpretability and classification consistency on representative cases. Guided by this framework, an automated end-to-end workflow constructs and executes risk cases in isolated environments and independently verifies outcomes using runtime traces and environment states. The benchmark provides a re
    
[^80]: 当画面几乎静止时保持JEPA世界模型的可规划性

    Keeping JEPA World Models Plannable When Little of the Frame Moves

    [https://arxiv.org/abs/2610.03137](https://arxiv.org/abs/2610.03137)

    通过SLIM推动基准诊断出JEPA世界模型在画面几乎静止的场景中编码器潜在表示对动作不敏感的失败根源，并提出仅用一个逆动力学辅助损失即可将语言目标规划成功率从0.003提升至0.35。

    

    用语言而非目标帧来指定目标，是利用潜在世界模型进行规划的一种自然交互方式，但要测试这一点，需要语言必须在多个物体之间加以区分的场景。我们构建了SLIM，这是一个包含多个小物体的推动任务基准，在相同的场景上配有成对的视觉目标和语言目标。在SLIM上，一个能够解决PushT任务的LeWM世界模型的成功率不到1%，尽管一个使用模拟器状态的脚本控制器可以完成所有难度级别。探测分析将失败定位于编码器：其潜在表示几乎对动作不敏感，既无法从中解码出推杆位置，也无法解码出物体位置，而且其预测展开（rollout）的效果并不比直接复制当前潜在表示更好。仅用一个逆动力学辅助损失——作用于编码器潜在表示，并通过一个在测试时被丢弃的共享头部作用于预测潜在表示——即可恢复所有探测指标，并将成功率从0.003提升至0.35（在困难推动级别上为0.16，而一个与目标无关的策略得分为……

    arXiv:2610.03137v1 Announce Type: new  Abstract: Specifying a goal in language rather than as a goal frame is a natural interface for planning with a latent world model, but testing it needs scenes in which language must discriminate between several objects. We build SLIM, a pushing benchmark with several small objects and paired visual and language goals on identical scenes. On SLIM a LeWM world model that solves PushT succeeds on under 1% of trials, although a scripted controller with simulator state solves every tier. Probes locate the failure in the encoder: its latent is nearly action-insensitive, neither pusher nor object positions can be decoded from it, and rollouts are no better than copying the current latent forward. One inverse-dynamics auxiliary loss, applied to encoder latents and to predicted latents through a shared head discarded at test time, restores every probe and raises success from 0.003 to 0.35 (0.16 on the hard pushing tier, where a goal-agnostic policy scores 
    
[^81]: 基于文本梯度的交易策略优化

    Trading Strategy Optimization via Textual Gradient

    [https://arxiv.org/abs/2610.03128](https://arxiv.org/abs/2610.03128)

    提出了TradeGrad框架，通过利用积累的优化经验来估计文本梯度并结合多尺度修订策略，克服了传统文本梯度优化短视及忽视时间稳健性的问题，实现了更稳健的量化交易策略优化。

    

    量化交易策略设计旨在从历史数据中发现在未来市场中依然有效的交易程序，这可以被视为一个黑盒程序优化问题。基于大语言模型（LLM）的文本梯度方法通过为迭代式策略改进提供明确的优化方向，展现出广阔前景。然而，直接应用文本梯度面临两个挑战：（1）优化过程是短视的，未能充分利用先前评估积累的经验；（2）汇总式的回测反馈忽略了时间上的稳健性，可能偏向那些仅在特定市场时期表现良好的策略。为应对这些挑战，我们提出了TradeGrad，一个经验引导的文本梯度框架，用于稳健的交易策略优化。TradeGrad利用积累的优化经验来估计文本梯度，并采用多尺度修订进行策略探索与改进，还进一步……

    arXiv:2610.03128v1 Announce Type: new  Abstract: Quantitative trading strategy design aims to discover trading programs from historical data that remain effective in future markets, which can be viewed as a black-box program optimization problem. LLM-based textual gradients offer a promising approach by providing explicit optimization directions for iterative strategy refinement. However, directly applying textual gradients faces two challenges: (1) optimization is myopic, underutilizing experience from previous evaluations; and (2) aggregate backtest feedback overlooks temporal robustness, potentially favoring strategies that perform well only in specific market periods. To address these challenges, we propose TradeGrad, an experience-guided textual-gradient framework for robust trading strategy optimization. TradeGrad leverages accumulated optimization experience to estimate textual gradients and employs multi-scale revisions for both strategy exploration and refinement. It further i
    
[^82]: 开放权重大语言模型中用于滥用检测的触发-标记机制的脆弱性

    The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs

    [https://arxiv.org/abs/2610.03124](https://arxiv.org/abs/2610.03124)

    该论文首次形式化了开放权重大语言模型中的触发-标记滥用检测机制，将其分为令牌级和权重级两类，并系统研究揭示了此类机制在对抗性攻击下的脆弱性。

    

    开放权重大语言模型可以被下载、修改和部署，超出了开发者的控制范围，这限制了集中式安全保障措施的有效性。因此，近期的研究提出了“触发-标记”机制，当模型在目标条件下被使用时（例如生成钓鱼内容），该机制会产生可检测的信号。尽管这些机制借鉴了已有的技术，但它们在开放权重大语言模型的条件性滥用检测方面的应用相对较新。因此，现有研究工作尚未系统性地研究触发-标记机制在对抗性攻击下的鲁棒性。为了填补这一空白，（i）我们对触发-标记进行了形式化定义，区分了在解码过程中引入水印式信号的“令牌级触发-标记”与学习目标条件与可检测模型行为之间后门式关联的“权重级触发-标记”。此外，（ii）我们……

    arXiv:2610.03124v1 Announce Type: cross  Abstract: Open-weight language models can be downloaded, modified, and deployed beyond their developers' control, limiting the effectiveness of centrally enforced safeguards. Recent work has therefore proposed \emph{trigger-tag} mechanisms that produce a detectable signal when a model is used under a target condition, such as generating phishing contents. Although these mechanisms borrow from established techniques, their use for conditional misuse detection in open-weight LLMs is relatively new. Therefore, existing research works have not systematically studied the robustness of trigger-tag mechanisms under adversarial attacks. To close this gap, (i)~we formalize trigger-tags and distinguish \emph{token-level trigger-tags}, which introduce watermark-inspired signals during decoding, from \emph{weight-level trigger-tags}, which learn backdoor-inspired associations between target conditions and detectable model behavior. Furthermore, (ii)~we intr
    
[^83]: Foresight：无需重新训练即可在流式视觉语言模型中规划未来感知

    Foresight: planning future perception in streaming VLMs without retraining

    [https://arxiv.org/abs/2610.03123](https://arxiv.org/abs/2610.03123)

    提出无需任何重训练的FORESIGHT双流架构，利用流式VLM固有的近期未来预测能力，动态规划并配置未来的感知计算，使模型能够自适应地应对不断变化的场景动态。

    

    现有的流式视觉语言模型（VLM）能够对视觉流进行持续的感知与推理，但其计算通路在整个推理过程中是固定不变的。因此，它们无法使计算适应不断演化的场景动态——不同的未来事件需要不同水平和不同形式的感知。我们证明，流式VLM天然具备预测近期未来的能力，并利用这一能力以无需训练的方式动态配置未来的计算。然而，实现这种前瞻性计算极具挑战：未来预测必须足够可靠才能用于指导计算，规划必须与流式推理并发进行，且在线重配置的开销必须可以忽略不计。为应对这些挑战，我们提出了FORESIGHT，一种由两个孪生LLM组成的双流架构，它们共享权重、输入编码器和KV缓存。第一个LLM……（摘要原文在此处截断）

    arXiv:2610.03123v1 Announce Type: cross  Abstract: Existing streaming vision-language models (VLMs) continuously perceive and reason over visual streams, but their computational pathways remain fixed throughout inference. Consequently, they cannot adapt computation to evolving scene dynamics, where different future events demand different levels and forms of perception. We show that streaming VLMs inherently possess the ability to anticipate the immediate future, and leverage this capability to dynamically configure future computation in a training-free manner. Realizing such anticipatory computation, however, is very challenging: future anticipation must be sufficiently reliable to guide computation, planning must run concurrently with streaming inference, and online reconfiguration must incur negligible overhead. To address these challenges, we introduce FORESIGHT, a dual-stream architecture comprising two Siamese LLMs with shared weights, input encoders, and KV cache. The first LLM 
    
[^84]: 如何在终身强化学习中发现并重用策略以实现持续适应

    How to Find and Reuse Policies for Continuous Adaptation in Lifelong Reinforcement Learning

    [https://arxiv.org/abs/2610.03119](https://arxiv.org/abs/2610.03119)

    提出AMSC方法，利用基于Wasserstein任务嵌入的在线相似性估计，自适应地选择和组合多个先前策略作为先验，从而在终身强化学习中获得更高的平均性能、前向迁移能力且不遗忘旧知识。

    

    在终身强化学习中，仅仅保留先前学到的策略并不足以实现对新任务的有效迁移。有用的知识可能分布在多个先前策略之中，并且其相关性会随着学习者积累经验而发生变化。一种假设是，在持续学习环境中可以有效地利用任务相似性来发现并组合先前学到的策略。为了验证这一假设，研究提出了自适应掩码选择与组合方法（AMSC），该方法通过从状态-动作-奖励样本中构建非参数化的Wasserstein任务嵌入，从在线经验中估计任务相似性。利用经z-score标准化的sparsemax相似性分数，推导出可变大小的支持集，从而在学习新任务时周期性地选择并加权策略以形成先验。在CT-graph和MiniGrid基准测试中，AMSC相比所评估的模块化组合基线方法取得了更高的平均性能和前向迁移能力，同时表现出没有遗忘的特性。

    arXiv:2610.03119v1 Announce Type: cross  Abstract: In lifelong reinforcement learning, retaining previously learned policies is not sufficient for effective transfer to a new task. Useful knowledge may be distributed across several prior policies, and its relevance may change as the learner acquires experience. One hypothesis is that task similarity can be effectively used in a continual learning setting to find and combine previously learned policies. To test it, Adaptive Mask Selection and Composition (AMSC) is designed to estimate similarity from online experience via non-parametric Wasserstein task embeddings from state-action-reward samples. The z-score-normalized sparsemax of the similarity scores are used to derive a variable-size support to periodically choose and weight policies to form a prior when learning a new task. On CT-graph and MiniGrid, AMSC achieves higher mean performance and forward transfer than the evaluated modular composition baselines while exhibiting no forge
    
[^85]: S2S-JEPA：在次季节到季节时间尺度上预测可预测的部分

    S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales

    [https://arxiv.org/abs/2610.03106](https://arxiv.org/abs/2610.03106)

    该论文提出S2S-JEPA，首次将计算机视觉中的联合嵌入预测架构（JEPA）范式引入次季节到季节（S2S）预报，通过在潜在空间中只预测缓慢变化且可预测的分量、舍弃不可预测的细尺度细节，来突破AI天气模型在两周以上“可预测性荒漠”中的性能瓶颈。

    

    次季节到季节（S2S）时间尺度，大约指未来两周到两个月，是农业、能源和水资源管理等行业的关键预报窗口。然而，这一时间尺度被广泛称为“可预测性荒漠”。近期的AI天气模型在两周以内的预报表现出色，但超过两周后性能显著下降，这主要是因为它们被训练去预测在S2S时间尺度上既不可预测也不重要的细尺度细节。我们认为，一个更符合物理规律的目标是只预报那些仍然可预测的缓慢变化分量。计算机视觉领域通过联合嵌入预测架构得出了相同的结论，该架构在潜在空间中进行预测，从而丢弃不可预测的细节。在这项工作中，我们提出了S2S-JEPA，将JEPA范式引入S2S预报任务。它借鉴了最先进AI天气模型的设计元素，针对这一任务进行了专门定制。S2S-JEPA达到了相当的预报技巧水平。

    arXiv:2610.03106v1 Announce Type: cross  Abstract: The subseasonal-to-seasonal (S2S) timescale, roughly from two weeks to two months ahead, is a critical forecast window for sectors such as agriculture, energy, and water management. Yet, it is widely known as the `predictability desert'. Recent AI weather models excel up to two weeks ahead but deteriorate beyond, largely because they are trained to predict fine-scale details that are neither predictable nor essential at S2S timescales. We argue that a more physically grounded objective is to forecast only the slowly varying components that remain predictable. Computer vision reached the same conclusion with the Joint-Embedding Predictive Architecture (JEPA), which predicts in latent space, discarding unpredictable details. In this work, we introduce S2S-JEPA, which brings the JEPA paradigm to S2S forecasting. It is tailored to this task through design elements from state-of-the-art AI weather models. S2S-JEPA achieves comparable skill 
    
[^86]: 询问、放宽还是行动？评估LLM偏好推理中的可操作不确定性

    Ask, Relax, or Act? Evaluating Actionable Indeterminacy in LLM Preference Reasoning

    [https://arxiv.org/abs/2610.03102](https://arxiv.org/abs/2610.03102)

    该论文形式化了“可操作不确定性”概念并构建基于求解器的基准测试，发现LLM难以判断何时无需干预——即使行动已被证明合理，模型仍倾向于不必要的澄清提问或干预。

    

    一个LLM智能体能够识别不确定性，却仍然可能选择错误的下一步：在行动已被证明合理时仍然提问，或者在必须改变约束时才寻求澄清。我们形式化了“可操作不确定性”这一概念：当所有可接受的偏好或目标下都存在共享的可接受行动时应当行动；当每种可能性都可行但没有共享行动时应当澄清；当请求不可行时应提出最小成本的允许约束修复。我们构建了一个基于求解器的基准测试，涵盖物品分配、会议调度、公寓选择和稳定匹配四个场景。匹配对保持相同的来源，同时改变是否需要干预，评估则将决策正确性、匹配对可靠性和完全正确响应区分开来。我们的发现揭示了一个反复出现的困难：模型难以识别何时不需要干预——模型能够识别需要澄清或修复的情况，却仍然会进行不必要的干预。

    arXiv:2610.03102v1 Announce Type: cross  Abstract: An LLM agent can recognize uncertainty yet still choose the wrong next step: asking when action is already justified, or seeking clarification when the constraints must change. We formalize actionable indeterminacy: act when an accepted action is shared across all admissible preferences or objectives, clarify when each possibility is feasible but no action is shared, and propose a minimum-cost permitted constraint repair when the request is infeasible. We construct a solver-grounded benchmark spanning object allocation, meeting scheduling, apartment choice, and stable matching. Matched pairs retain the same source while changing whether intervention is necessary, and evaluation separates decision correctness, matched-pair reliability, and fully correct responses. Our findings reveal a recurring difficulty in recognizing when intervention is unnecessary: models can identify situations requiring clarification or repair yet still interven
    
[^87]: 超越单个视频：面向电商跨视频推理的基准测试与主动证据寻求

    Beyond Single Videos: Benchmarking and Active Evidence Seeking for E-Commerce Cross-Video Reasoning

    [https://arxiv.org/abs/2610.03099](https://arxiv.org/abs/2610.03099)

    该论文提出了首个电商跨视频推理基准AdsCVR，并设计了智能体框架AdSeek，通过多轮探索中动态选择视听工具实现主动证据获取，同时引入离线轨迹修正机制以应对强化学习中的稀疏信用分配问题。

    

    电商视频信息密集，消费者在评估产品、商家在评估营销策略时经常会对比这些视频。然而，现有多模态模型主要聚焦于单视频理解，跨视频对比信息的能力有限。我们推出了AdsCVR，这是首个电商跨视频推理基准，包含2,483个视频和6,110个问答对，覆盖六个推理维度。跨视频推理要求模型在大量冗余帧中定位细粒度证据，并整合视觉细节、语音和屏幕文字。为此，我们提出了AdSeek，一个智能体框架，在多轮探索过程中动态选择视觉和音频工具，以主动证据获取取代静态均匀采样。为了解决强化学习中稀疏的信用分配问题，我们开发了一种离线轨迹修正机制，用以识别推理步骤……

    arXiv:2610.03099v1 Announce Type: cross  Abstract: E-commerce videos are information-dense and frequently compared by consumers evaluating products and merchants assessing marketing strategies. However, existing multimodal models mainly focus on single-video understanding and have limited ability to compare information across videos. We introduce AdsCVR, the first e-commerce cross-video reasoning benchmark, containing 2,483 videos and 6,110 question-answer pairs across six reasoning dimensions. Cross- video reasoning requires models to locate fine-grained evidence among many redundant frames and integrate visual details, speech, and on-screen text. We therefore propose AdSeek, an agentic framework that dynamically selects visual and audio tools during multi-turn exploration, replacing static uniform sampling with active evidence acquisition. To address the sparse credit assignment of reinforcement learning, we develop an offline trajectory rectification mechanism that identifies reason
    
[^88]: 预测器引导的潜空间密码子优化以最大化蛋白质表达

    Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression

    [https://arxiv.org/abs/2610.03098](https://arxiv.org/abs/2610.03098)

    提出潜空间密码子优化方法LSCO，通过将序列映射到预训练mRNA语言模型的潜空间，将离散的密码子优化问题转化为可梯度搜索的连续问题，并结合不确定性感知的表达预测器、最小自由能正则化、自然性先验和约束解码，以最大化蛋白质表达。

    

    密码子优化是通过选择同义密码子来提高mRNA翻译效率和蛋白质表达水平的过程，是治疗性蛋白质生产和mRNA疫苗研发的核心环节，但它仍然是一个难题。其设计空间是离散的且组合规模巨大，这使得基于梯度的方法无法适用，而现有工具依赖启发式代理指标（如密码子适应指数或GC含量），这些指标难以真实反映实际表达水平。我们提出了潜空间密码子优化（LSCO），通过将序列映射到预训练mRNA语言模型的潜空间中，将这一离散问题转化为连续问题，从而实现高效的基于梯度的搜索。LSCO结合了四个组件：来自不确定性感知预测器的数据驱动表达目标函数、用于结构稳定性的最小自由能（MFE）正则化项、来自蛋白质到密码子反向翻译模型的自然性先验，以及确保蛋白质保真度的约束解码。

    arXiv:2610.03098v1 Announce Type: new  Abstract: Codon optimization, the process of selecting synonymous codons to improve mRNA translation efficiency and protein expression, is central to therapeutic protein production and mRNA vaccines, yet it remains a hard problem. The design space is discrete and combinatorially large, precluding gradient-based methods, and existing tools rely on heuristic proxies (e.g., Codon Adaptation Index or GC-content) that poorly capture true expression. We introduce Latent-Space Codon Optimization (LSCO), which recasts this discrete problem as a continuous one by mapping sequences into the latent space of a pretrained mRNA language model, enabling efficient gradient-based search. LSCO combines four components: a data-driven expression objective from an uncertainty-aware predictor, a Minimum-Free-Energy regularizer for structural stability, a naturalness prior from a protein-to-codon back-translation model, and constrained decoding for protein fidelity. On 
    
[^89]: 异构AI模型间的同伴影响

    Peer Influence across Heterogeneous AI Models

    [https://arxiv.org/abs/2610.03095](https://arxiv.org/abs/2610.03095)

    该研究测量了七个开源语言模型之间的说服效应，发现模型意见分歧时说服作用非常强烈，但模型规模和单独运行时的确定性均无法预测说服动态，小模型既能像大模型一样有效说服他人，也同样能抵抗影响。

    

    当两个AI智能体产生分歧时，谁会说服谁？随着多智能体系统越来越多地组合使用不同家族和不同规模的语言模型，这一问题的答案将决定哪些判断能在交互中留存下来。我们将说服力衡量为智能体在与持异议的同伴进行单次交流后其决策发生的概率偏移，并在三项语言理解任务上测试了七个开源权重模型。研究发现，说服作用非常强烈：当模型意见不一致时，接收方在看到同伴的答案和解释后往往会放弃自己最初的判断。然而令人惊讶的是，无论是模型单独运行时的确定性还是模型规模，都无法可靠地预测说服动态。在独立运行时决策几乎完全一致的模型，反而可能最容易受到说服的影响；而小模型作为说服者可以与大模型相匹敌，并且同样能有效地抵抗后者的影响。此外，我们还表明，决策偏移的大小更多取决于接收方的易感程度而非说服方。

    arXiv:2610.03095v1 Announce Type: new  Abstract: When two AI agents disagree, who persuades whom? As multi-agent systems increasingly combine language models of different families and sizes, the answer can determine which judgments survive interaction. Measuring persuasion as the probabilistic shift in an agent's decision after a single exchange with a dissenting peer, we test seven open-weight models across three language understanding tasks. We find that persuasion is strong: when models disagree, receivers often abandon their initial judgment after seeing a peer's answer and explanation. Surprisingly, however, neither standalone certainty nor model scale reliably predicts persuasion dynamics. Models producing almost perfectly consistent decisions in isolation can be among the most susceptible to persuasion, and small models can match larger ones as persuaders and resist their influence just as effectively. Furthermore, we show that the size of the shift depends more on the susceptib
    
[^90]: ULTRADISCOVERY：在一个互联的、认识论开放的宇宙中进行溯因探索

    ULTRADISCOVERY: Abductive Exploration in an Interconnected, Epistemically Open Universe

    [https://arxiv.org/abs/2610.03092](https://arxiv.org/abs/2610.03092)

    该论文提出ULTRADISCOVERY交互式基准，通过2×2设计独立控制表征开放性与证据分布性，评估智能体在认识论开放且结构互联的世界中进行溯因科学探索的能力，发现现有十一个模型均无法通过引入新实体或重写变量来完成理论替换。

    

    科学发现往往始于零散的线索呼唤一种描述世界的新方式。当世界在认识论上是开放的，这种溯因探索可能需要构建用以陈述解释的表征；当世界在结构上是互联的，则需要综合散布于不同情境中的证据。现有基准很少将这两项需求区分开来或对它们进行独立控制。我们提出ULTRADISCOVERY，一个包含五个领域的交互式世界，智能体在其中需要修正一个最初成功的理论，并预测一次未见过的跨领域干预的结果。该基准采用2×2设计：表征保持开放或予以揭示，证据保持分散或予以对齐，而潜在动力学保持不变。在表征开放的条件下，横跨十一个模型的智能体常常收回他们被教授的公理，但没有任何一个模型能够引入替换所需的未观测实体或重写变量……

    arXiv:2610.03092v1 Announce Type: cross  Abstract: Scientific discovery often begins when scattered clues call for a new way of describing the world. Such abductive exploration can require constructing the representation in which an explanation is stated, when the world is epistemically open, and composing evidence scattered across contexts, when it is structurally interconnected. Existing benchmarks rarely separate these two demands or control them independently. We introduce ULTRADISCOVERY, an interactive world of five domains in which an agent revises an initially successful theory and predicts the outcome of an unseen cross-domain intervention. A $2 \times 2$ design leaves the representation open or discloses it, and leaves the evidence distributed or aligns it, with the latent dynamics fixed. With the representation open, agents across eleven models often retract the axiom they were taught, and none introduces the unobserved entity or rewrites the variables that a replacement requ
    
[^91]: 防御计算机使用智能体免受分支引导攻击的安全研究

    Securing Computer-Use Agents Against Branch Steering Attacks

    [https://arxiv.org/abs/2610.03089](https://arxiv.org/abs/2610.03089)

    本文系统研究了针对计算机使用智能体的新型“分支引导攻击”——攻击者无需注入显式指令，仅通过构造不可信数据即可诱导智能体走向预先批准的危险执行分支，并提出了STEER-Bench基准来评估这一威胁。

    

    现代计算机使用智能体直接与图形用户界面交互并执行第三方网络工具，这使其在每个渲染页面和工具响应中都面临间接提示注入的风险。虽然双LLM模式是提供形式化安全保证的主要系统级架构——它使用隔离的规划器LLM（P-LLM）在通过隔离LLM（Q-LLM）处理不可信输入之前固定执行路径——但这些保证在图形环境中会失效。由于CUA交互本质上是动态的，计划无法保持与数据无关；它们必须基于预期的运行时网页内容进行分支，以覆盖智能体可能遇到的所有情况。这使智能体暴露于分支引导攻击之下：攻击者通过精心构造不可信数据，在不注入显式指令的情况下诱导CUA走向危险的、预先批准的分支。我们系统地研究了分支引导攻击，并引入了STEER-Bench（基准数据集）。

    arXiv:2610.03089v1 Announce Type: cross  Abstract: Modern Computer Use Agents (CUAs) directly interact with graphical user interfaces and execute third-party web tools, exposing them to indirect prompt injection across every rendered page and tool response. While the Dual-LLM pattern is the primary system-level architecture offering formal security guarantees - using an isolated Planner LLM (P-LLM) to fix execution paths before processing untrusted inputs via a Quarantined LLM (Q-LLM) - these guarantees break down in graphical environments. Because CUA interaction is inherently dynamic, plans cannot remain data-independent; they must branch based on anticipated runtime web content - covering all possible cases the agent may encounter. This exposes agents to branch steering attacks, where an adversary crafts untrusted data to coerce a CUA down a hazardous, pre-approved branch without injecting explicit instructions. We systematically study branch steering attacks and introduce STEER-Ben
    
[^92]: Zephon：面向在线、有状态基础模型数据加载流水线的弹性确定性

    Zephon: Elastic Determinism for Online, Stateful Foundation Model Data Loading Pipelines

    [https://arxiv.org/abs/2610.03087](https://arxiv.org/abs/2610.03087)

    Zephon提出了一种面向基础模型训练的数据加载器，能够在GPU拓扑变化、频繁检查点恢复及不同执行后端的情况下，为包含在线分词、打包、混合等有状态n对m转换的数据流水线提供确定性的全局训练数据批次序列（即弹性确定性）。

    

    确定性数据加载对于基础模型开发至关重要：模型研究人员需要确信，他们在昂贵的消融实验中观察到的差异是由所更改的参数引起的，而非训练数据序列中的非确定性所致。数据加载器必须提供弹性确定性，即即使在多次运行之间GPU拓扑发生变化（例如由于GPU资源稀缺）、频繁的检查点恢复周期以及不同的数据处理执行后端的情况下，仍能保证确定的全局训练数据批次序列。实现这一目标非常困难，因为现代基础模型数据流水线会在线对样本进行分词、打包和混合，引入了破坏样本索引的有状态n对m转换。现有的数据加载器大多假设可索引的1对1流水线，而常见的替代方案——离线物化——成本高昂，且对于视频等某些模态来说不可行。我们提出了Zephon，一个面向基础模型的数据加载器……

    arXiv:2610.03087v1 Announce Type: cross  Abstract: Deterministic data loading is important for foundation model development: model researchers need confidence that differences they observe across costly ablations are caused by the parameter they changed rather than non-determinism in the training data sequence. The data loader must provide elastic determinism, i.e., a deterministic sequence of global training data batches despite changes to the GPU topology across runs (e.g., due to GPU scarcity), frequent checkpoint-resume cycles, and different data processing execution backends. Achieving this is difficult because modern foundation model data pipelines tokenize, pack, and mix samples online, introducing stateful n-to-m transformations that break sample indexing. Existing data loaders largely assume indexable 1-to-1 pipelines, and the common workaround of offline materialization is expensive and, for some modalities such as video, infeasible.   We present Zephon, a data loader for fou
    
[^93]: NegT2IBench：当否定改变图像时——面向文本到图像模型的极性基准

    NegT2IBench: When Negation Changes the Picture. A Polarity Benchmark for Text-to-Image Models

    [https://arxiv.org/abs/2610.03084](https://arxiv.org/abs/2610.03084)

    提出了NegT2IBench基准，通过4,800条按极性组织的提示词系统评估文本到图像模型满足否定约束的能力，其基于检测器的评分以更小的规模达到了与大型视觉语言评判器相当的人类一致性水平。

    

    文本到图像（T2I）模型通常由测量请求内容是否出现的基准来评判，但这些基准在很大程度上忽略了模型满足否定约束的补充能力，例如生成“一个非红色的杯子”。衡量否定带来了基于肯定式基准所不曾面对的挑战，需要精心的提示词与评估设计。我们提出NegT2IBench，一个包含4,800条提示词的基准，涵盖两种属性类型和四种关系类别。提示词按极性组织：必须成立的肯定语句数量和必须不成立的否定语句数量，各自取值范围为0到2。通过独立变化这两个因素，可以将否定效应与提示词复杂度效应分离开来。我们基于检测器的评分具有可复现、可审计的特点，并能精确定位哪个需求失败了。在600张带有三位标注者标签的图像上，该评分与人类判断的一致性程度与体积大30倍的视觉语言评判器相当……

    arXiv:2610.03084v1 Announce Type: cross  Abstract: Text-to-image (T2I) models are judged by benchmarks that measure whether requested content appears, but these benchmarks largely overlook the complementary ability to satisfy negated constraints, for example, generating "a non-red cup." Measuring negation raises challenges not faced by affirmation-based benchmarks and requires careful prompt and evaluation design. We introduce NegT2IBench, a benchmark of 4,800 prompts covering two attribute types and four relation categories. Prompts are organized by polarity: the number of positive statements that must hold and negated statements that must not, each ranging from 0 to 2. Varying the two independently separates the effect of negation from the effect of prompt complexity. Our detector-based scoring is reproducible, auditable, and pinpoints which requirement failed. On 600 images with three-annotator labels, it agrees with humans as closely as vision-language judges up to 30x larger, whil
    
[^94]: RIFAR：面向持续机器人学习的可靠性与遗忘感知回放

    RIFAR: Reliability and Forgetting-Aware Replay for Continual Robot Learning

    [https://arxiv.org/abs/2610.03079](https://arxiv.org/abs/2610.03079)

    提出了 RIFAR 方法，通过可靠性筛选与漂移感知的回放选择，利用冻结的逆动力学模型评估世界-动作模型重建轨迹的动作-视觉一致性，从而在机器人持续学习中选择高质量回放经验，避免灾难性遗忘。

    

    真正的具身智能要求机器人能够将连续的现实世界经验转化为持久且可迁移的技能。这就要求持续学习能够在任务和环境不断演变的情况下，整合新能力而不侵蚀已有知识。经验回放（Experience Replay）可以缓解遗忘问题，但随着任务不断积累，存储完整演示数据的代价变得十分高昂。世界-动作模型（World-Action Models）提供了一种生成式的替代方案，通过对动作和未来观测的联合预测来重建过去的经验。然而，视觉上连贯的推演（rollout）中可能包含无法真正实现所预测状态转移的动作，而新任务的适应又可能破坏先前学到的行为。因此，RIFAR 将可靠性筛选与漂移感知的回放选择相结合：它从紧凑的演示前缀重建轨迹，并使用一个冻结的逆动力学模型来评估动作与视觉之间的一致性。训练阶段首先将当前演示与（摘要在此处截断）

    arXiv:2610.03079v1 Announce Type: new  Abstract: Genuine embodied agency requires robots to turn continuous real-world experience into lasting, transferable skills. This demands continual learning that integrates new capabilities without eroding prior knowledge as tasks and environments evolve. Experience replay mitigates forgetting, but storing complete demonstrations becomes costly as tasks accumulate. World-action models offer a generative alternative, reconstructing past experience through joint predictions of actions and future observations. However, visually coherent rollouts may contain actions that cannot realize the predicted transitions, while new-task adaptation can disrupt previously learned behavior. RIFAR therefore combines reliability screening with drift-aware replay selection. It reconstructs trajectories from compact demonstration prefixes and uses a frozen inverse-dynamics model to assess action-visual consistency. Training first combines current demonstrations with 
    
[^95]: MOF-VERIFY：一个面向MOF假设验证的失效感知智能体框架

    MOF-VERIFY: A Failure-Aware Agentic Harness for MOF Hypothesis Verification

    [https://arxiv.org/abs/2610.03056](https://arxiv.org/abs/2610.03056)

    该论文提出了MOF-VERIFY，一个失效感知的智能体框架与四任务族诊断基准，通过闭卷、检索和先知证据等设置系统评估并定位大语言模型在MOF假设验证中的失效点（涵盖结构接地、合成条件、证据充分性和MLIP计算验证）。

    

    大语言模型正越来越多地被用作AI驱动的材料“共同科学家”（Co-Scientists）中的推理组件，但由此构建的验证流程的可靠性仍不明确。金属有机框架（MOFs）提供了一个特别具有挑战性的测试场景：因为结构可能以不同的标识符出现、合成结果强烈依赖于实验条件、证据分布在异构数据源中，且某些假设需要通过计算而非仅靠文献来验证。我们引入了一个包含四个任务族的诊断基准，涵盖结构接地、合成条件验证、证据充分性验证以及基于机器学习原子间势（MLIP）的计算验证。其中T-MOF-1-3在闭卷、启用检索和先知证据三种设置下进行评估，以定位知识访问、证据获取和推理环节中的失效点，而T-MOF-4则单独评估计算验证能力。

    arXiv:2610.03056v1 Announce Type: new  Abstract: Large language models are increasingly used as reasoning components in AI-driven materials Co-Scientists, yet the reliability of the resulting verification pipeline remains unclear. Metal-organic frameworks (MOFs) provide a particularly challenging setting because structures may appear under different identifiers, synthesis outcomes depend strongly on experimental conditions, evidence is distributed across heterogeneous sources, and some hypotheses require computation rather than literature alone. We introduce a diagnostic benchmark with four task families covering structural grounding, synthesis-condition verification, evidence-sufficiency verification, and MLIP-based computational verification. T-MOF-1-3 are evaluated under closed-book, retrieval-enabled, and oracle-evidence settings to localize failures in knowledge access, evidence acquisition, and reasoning, while T-MOF-4 separately evaluates computational verification. Guided by th
    
[^96]: HACKTRACE：代码生成过程中基于行为监督的奖励破解检测

    hacktrace: behavior-supervised detection of reward hacking during code generation

    [https://arxiv.org/abs/2610.03055](https://arxiv.org/abs/2610.03055)

    HACKTRACE 通过复用编程智能体生成代码时已计算出的内部状态来监督捷径行为，无需额外模型推理即可在回合结束前检测奖励破解，AUC 达 0.997 且监控开销仅 8 毫秒。

    

    一个编程智能体可以通过修复代码来通过测试，也可以通过删除暴露 bug 的测试来通过测试。检测这种“奖励破解”行为需要识别出尝试走捷径的动作，包括那些最终失败的尝试。我们发布了来自 Qwen3-8B 的 173,561 条带标注的多轮编程轨迹，并表明独立于漏洞利用是否成功来监督捷径行为，可以显著提升检测效果。我们提出了 HACKTRACE，一种行为监督的监控器，它读取智能体在生成代码时已经计算出的内部状态。复用这些状态使得在回合尚未结束时就能进行监控，无需额外的语言模型 token 或推理次数。将该证据与最终代码文件的静态特征相结合，每个问题的平均 AUC 达到 0.997，监控开销仅 8 毫秒，在准确性和延迟上均优于那种让模型再次回答诚实性问题的监控器。同样的生成状态还提供了一种低成本的监测手段。

    arXiv:2610.03055v1 Announce Type: new  Abstract: A coding agent can earn a passing grade by fixing its code, or by deleting the test that exposes the bug. Detecting such reward hacking requires recognizing attempted shortcuts, including those that fail. We release 173,561 annotated multi-turn coding trajectories from Qwen3-8B and show that supervising shortcut behavior independently of exploit success substantially improves detection. We introduce HACKTRACE, a behavior-supervised monitor that reads the internal states the agent already computes while generating code. Reusing these states enables monitoring before a turn is complete, without additional language-model tokens or passes. Combining this evidence with static features of the final files achieves a mean per-problem AUC of 0.997 with 8 ms of monitoring overhead, improving both accuracy and latency over monitors that run the model again on an honesty question and answer. The same generation states also provide an inexpensive mon
    
[^97]: WebFovea：当模型正确但点击出错时——基于视觉的网页智能体在真实网站上的可靠往返执行

    WebFovea: When the Model Is Right but the Click Is Wrong -- Reliable Round Trips for Vision-Based Web Agents on Live Websites

    [https://arxiv.org/abs/2610.03036](https://arxiv.org/abs/2610.03036)

    本文提出在WebRetriever Challenge 2026中获得亚军的视觉网页智能体WebFovea，并指出真实网站上的许多失败并非源于模型推理，而是源于模型与浏览器之间中间执行层在动作解析、页面生效、结果反馈和信息展示这四个环节上的问题。

    

    我们提出了WebFovea，一个基于视觉的网页智能体，它在WebRetriever Challenge 2026中获得第二名，最终得分为100分中的57.0分。该挑战赛在WebRetriever基准（arXiv:2607.06118）的协议III上对智能体进行端到端评估：从真实网站上的入口URL出发，智能体必须操作网站自身的界面并返回可验证的答案。一个强大的多模态大语言模型（LLM）对完成此任务是必要的，但并不充分。模型的决策需要通过中间执行层（harness）——即模型与页面之间的代码——传递到浏览器。在每一步中，有四件事必须正确完成：模型的回复必须被解析为预期的动作，动作必须在页面上生效，结果必须被准确地反馈回来，并且模型必须被展示它所需的信息。在真实网站上，我们观察到的许多失败发生在这四个阶段之一，而不是出现在模型的推理中。一个坐标空间（摘要原文在此处截断）

    arXiv:2610.03036v1 Announce Type: cross  Abstract: We present WebFovea, a vision-based web agent that placed 2nd in the WebRetriever Challenge 2026 with a final score of 57.0 out of 100. The challenge evaluates agents end to end on Protocol III of the WebRetriever benchmark (arXiv:2607.06118): starting from an entry URL on a live website, the agent must operate the site's own interface and return a verifiable answer. A capable multimodal large language model (LLM) is necessary for this, but not sufficient. The model's decisions reach the browser through the harness, the code between the model and the page. At every step, four things must go right: the model's reply must be parsed into the intended action, the action must take effect on the page, the result must be reported back accurately, and the model must be shown the information it needs. On real websites, many of the failures we observed occurred at one of these four stages rather than in the model's reasoning. A coordinate-space 
    
[^98]: 当数字开始说话：LLM中的数值信号传递与策略行为

    When Numbers Start Talking: Numerical Signalling and Strategic Behaviour Among LLMs

    [https://arxiv.org/abs/2610.03033](https://arxiv.org/abs/2610.03033)

    本研究通过四个主流LLM驱动的智能体在四种策略博弈中的实验，首次揭示不同类型的消息（尤其是数值信号）会以不可预测的方式显著改变博弈中的合作水平与收益，且智能体生成的数值信号系统性偏离随机性，从而挑战了AI智能体总能收敛到稳定均衡的假设。

    

    基于大语言模型（LLM）的智能体越来越多地在以策略互动为特征的多智能体系统（MAS）中运行。然而，关于不同类型的消息是否以及在多大程度上影响策略博弈的结果，目前知之甚少。本研究通过考察基于四个主流LLM构建的AI智能体，让其在四种具有不同合作均衡的博弈中进行互动，研究不同类型的消息（自然语言、数值信号或随机序列）是否以及在多大程度上会显著改变每个博弈中的合作水平，同时这也取决于智能体被赋予的个性。我们观察到，结构化消息会改变大多数博弈和大多数LLM的最终收益，但不存在可预测的模式；这挑战了“无论智能体具备何种附加能力，AI智能体都能收敛到稳定均衡”的假设。此外，我们观察到智能体生成的数值消息偏离了随机性，且这种偏离在……时最为强烈和一致（原文此处截断）。

    arXiv:2610.03033v1 Announce Type: new  Abstract: Large language model (LLM)-based agents increasingly operate in multi-agent systems (MAS) characterised by strategic interaction. However, little is known about whether, and to what extent, different types of messages affect the outcomes of strategic games. By investigating AI agents based on four popular LLMs, playing four games with different cooperation equilibria, we study whether messages of different kinds (natural language, numerical signals, or random sequences) significantly modify the levels of cooperation in each game, also depending on the agents' assigned personalities. We observe that structured messages alter the final payoffs for most games and LLMs, but without a predictable pattern; this challenges the assumption that AI agents can converge to stable equilibria regardless of additional capabilities. Moreover, we observe that agent-generated numerical messages depart from randomness, most strongly and consistently when a
    
[^99]: SoftGene：蛋白质语言模型增强的软提示用于可解释的基因集注释

    SoftGene: Protein Language Model-Enhanced Soft Prompting for Interpretable Gene Set Annotation

    [https://arxiv.org/abs/2610.03029](https://arxiv.org/abs/2610.03029)

    提出SoftGene框架，利用蛋白质语言模型ESM将基因集的蛋白质序列信息编码为分层软提示，与大语言模型的硬提示结合，实现更符合生物学结构、可解释的基因集功能注释。

    

    基因集分析是功能基因组学的基石，但它仍然是一项劳动密集型的工作，严重依赖人工整理和专家的生物学解释。虽然大语言模型（LLM）已成为基因组推理和注释的强大工具，但大多数现有方法依赖于符号化的基因名称，未能捕获特定领域的生物学结构，特别是控制分子活性、相互作用和下游基因功能的蛋白质序列信息。在这项工作中，我们提出了SoftGene，这是一种新颖的基于LLM的基因集注释框架，它利用了基因集的分层结构。首先，我们使用基于ESM（一种蛋白质语言模型）构建的分层注意力编码器，利用蛋白质水平的氨基酸序列信息来表示每个基因集。其次，我们构建了一种混合提示方案，将源自基因集嵌入的软提示与包含……的硬提示相结合（摘要内容在此处被截断）。

    arXiv:2610.03029v1 Announce Type: new  Abstract: Gene set analysis is a cornerstone of functional genomics, yet it remains labor-intensive and heavily dependent on manual curation and expert biological interpretation. While Large Language Models (LLMs) have emerged as powerful tools for genomic reasoning and annotation, most existing approaches rely on symbolic gene names and fail to capture domain-specific biological structure, particularly protein sequence information that governs molecular activity, interactions, and downstream gene function. In this work, we propose SoftGene, a novel framework for LLM-based gene set annotation that leverages the hierarchical structure of gene sets. First, we use a hierarchical attention-based encoder built on ESM, a protein language model, to represent each gene set using protein-level amino acid sequence information. Second, we construct a hybrid prompting scheme that combines soft prompts derived from gene set embeddings with hard prompts contain
    
[^100]: 为1比特KV缓存压缩量身定制量化空间

    Tailoring the Quantization Space for 1-Bit KV Cache Compression

    [https://arxiv.org/abs/2610.03027](https://arxiv.org/abs/2610.03027)

    提出TaSQ方法，通过查询引导的通道加权、跨头归一化和协方差感知的通道分组来量身定制向量量化目标空间，从而在1比特极端压缩下实现有效的KV缓存压缩。

    

    键值缓存已成为长上下文大语言模型推理中的主要内存瓶颈，给内存容量和带宽带来了巨大压力。为缓解这一瓶颈，向量量化（VQ）已成为一种有前景的激进KV缓存压缩方法。然而，现有的VQ方法在1比特压缩区间下性能大幅下降。在如此极端的压缩下，每个码本必须用有限的质心集合来表示更大规模的通道组，使得有效利用码本容量变得愈发困难。为解决这一问题，我们提出了TaSQ，它通过结合查询引导的通道加权、跨头归一化以及协方差感知的通道分组来量身定制VQ目标空间，从而更好地反映缓存激活的误差敏感性和统计结构。由于这些变换与RoPE兼容，并且可以轻松地合并到投影权重和码本中，TaSQ保持了传统的（摘要在此处被截断）

    arXiv:2610.03027v1 Announce Type: cross  Abstract: The key-value (KV) cache becomes a major memory bottleneck in long-context LLM inference, placing substantial pressure on memory capacity and bandwidth. To mitigate this bottleneck, vector quantization (VQ) has emerged as a promising approach for aggressive KV cache compression. However, existing VQ methods degrade substantially in the 1-bit regime. At such extreme compression, each codebook must represent a larger group of channels with a limited set of centroids, making effective use of its capacity increasingly challenging. To address this, we introduce $\textbf{TaSQ}$, which tailors the VQ target space by combining query-guided channel weighting, cross-head normalization, and covariance-aware channel grouping to better reflect the error sensitivity and statistical structure of cached activations. Since these transforms are RoPE-compatible and can be easily merged into projection weights and codebooks, TaSQ preserves the conventiona
    
[^101]: 偏好的可验证、可表达与默会成分

    Verifiable, Articulable, and Tacit Components of Preference

    [https://arxiv.org/abs/2610.03025](https://arxiv.org/abs/2610.03025)

    该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。

    

    是什么让一篇短篇小说引人入胜、一篇新闻报道具有新闻价值、或一个数学证明优雅？这些构念难以言明或验证，其含义至少部分是默会的。然而，现代AI模型主要是通过明确的章程、评分标准和验证器（即RLAIF和RLVR）来改进的；偏好的默会成分通常研究不足。我们引入了一个大规模、带标注的偏好数据集CreativePreferences，其中包含280万个文本，由3.17亿个人类偏好判断在7个创意领域中进行标注，并配有42个基准任务。我们分别用可执行程序、评分标准库和密集训练的模型（V、A和VAT）对这些标签进行建模。我们观察到稳健的可表达性差距（VAT−VA）和可验证性差距（VAT−V）；我们采用一种新颖的测量方法来估计每个差距的上界和下界，该方法可发现可表达和可验证的指标、识别伪变量，并估计未被发现成分的价值。

    arXiv:2610.03025v1 Announce Type: new  Abstract: What makes a short story gripping; a news article newsworthy; or a math proof elegant? These constructs resist articulation or verification; their meaning is at least partially tacit. However, modern AI models are improved primarily via articulated constitutions, rubrics and verifiers (i.e. in RLAIF and RLVR); tacit components of preferences are typically understudied. We introduce a large, labeled preference dataset CreativePreferences, containing 2.8M texts labeled by 317M human preference judgments across 7 creative domains, with 42 benchmark tasks. We model these labels with executable programs, rubric banks and densely trained models (V, A and VAT, respectively). We observe robust articulability gaps, VAT-VA; and verifiability gaps, VAT-V; we estimate upper and lower bounds for each gap with a novel measurement approach that discovers articulable and verifiable metrics, identifies spurious variables and estimates the value of undisc
    
[^102]: DyadMem：一个衡量智能体如何与用户协作的长期记忆基准

    DyadMem: A Long-Term Memory Benchmark of How Agents Work with Users

    [https://arxiv.org/abs/2610.03020](https://arxiv.org/abs/2610.03020)

    提出 DyadMem 基准，首次定义并显式标注“用户条件化的关系型智能体记忆（URAM）”，在多会话轨迹上同时评估用户事实记忆与关系型协作记忆，共含 6 类记忆、5 万多个会话和 6 万余条问答实例，使长期智能体记忆评估更完整可靠。

    

    长期运行的智能体不仅需要记住关于用户的真实信息，还需要记住随着共享历史的演进，特定智能体应如何与该用户进行协作。现有基准主要监督用户的事实与偏好，或跨用户可复用的经验，使这种关系特定的智能体记忆处于隐含状态。此外，大多数先前的工作仅通过在长交互历史上进行最终答案问答来评估模型，使评估仍然不完整且不可靠。为此，我们提出了 DyadMem，并给出新的定义——用户条件化的关系型智能体记忆。DyadMem 在同一条多会话轨迹上共同标注用户侧记忆与 URAM，形成 6 个记忆类别。总而言之，该基准包含 3,065 个情节、50,961 个会话和 61,210 个问答实例，并附带丰富的会话级 Capture 与 Update 黄金标注、查询级 Recall 支持证据，以及两种问答设置：Gold-Memory 和 Full-。

    arXiv:2610.03020v1 Announce Type: new  Abstract: Long-term agents must remember not only what is true about a user, but also how a particular agent should work with that user as their shared history evolves. Existing benchmarks primarily supervise user facts and preferences or experience reusable across users, leaving this relationship-specific agent memory implicit. Additionally, most prior works measure the model solely with final-answer QA over long interaction histories, making the assessment still incomplete and unreliable. To this end, we introduce DyadMem with the proposed new definition User-conditioned Relational Agent Memory (URAM). DyadMem jointly annotates user-side memory and URAM along the same multi-session trajectories, resulting in 6 memory categories. To summarize, it includes 3,065 episodes, 50,961 sessions, and 61,210 QA instances, with extensive session-level Capture and Update gold annotations, query-level Recall support, and two QA settings: Gold-Memory and Full-
    
[^103]: 使用人工对话为构音障碍及气管造口说话者构建个性化自动语音识别系统

    Personalized Automatic Speech Recognition for a Dysarthric and Tracheostomic Speaker using Artificial Conversations

    [https://arxiv.org/abs/2610.03017](https://arxiv.org/abs/2610.03017)

    本文针对一位气管造口且严重构音障碍的捷克说话者，通过多阶段微调Whisper模型构建个性化语音识别系统，实现字符错误率相对降低50%，并发布了基于“人工对话”协议收集的33小时公开数据集。

    

    本工作提出了一个针对一位捷克说话者的个性化自动语音识别（ASR）系统。该说话者具有永久性气管造口和严重的构音障碍，其语音对于未经训练的听者而言难以理解。我们发布了一个包含该说话者33小时标注语音的公开数据集，这些数据是通过一种新颖的“人工对话”协议收集的，该协议旨在实现高参与度和对话真实感。我们提出了一种基于Whisper Base的多阶段训练流程：先在标准捷克语语音上微调，再在声学模拟的气管造口语音上训练，最后使用说话者本人的数据进行微调。我们在三种近实时场景下对该系统进行了评估：脚本对话、问答和自发对话。相比Whisper Base基线，该系统实现了50%的相对字符错误率降低，并在孤立话语的声学识别方面超越了其助手的平均识别准确率。我们证明，即使对于……（摘要截断）

    arXiv:2610.03017v1 Announce Type: new  Abstract: This work presents an automatic speech recognition (ASR) system personalized for a Czech speaker with a permanent tracheal stoma and severe dysarthria rendering their speech unintelligible to untrained listeners. We release a public dataset containing 33 annotated hours of the speaker's speech, collected using a novel "artificial conversation" protocol designed for high engagement and dialogue realism. We propose a multi-stage training pipeline based on Whisper Base: fine-tuning on standard Czech speech, acoustically simulated tracheostomic speech, and the speaker's data. We evaluate the system across three near real-time scenarios: scripted conversations, question answering, and spontaneous dialogue, achieving a 50\% relative reduction in Character Error Rate compared to Whisper Base baseline and surpassing the average recognition accuracy of their assistants in acoustic recognition of isolated utterances. We demonstrate that even for s
    
[^104]: OmniAct3D：利用基础几何模型与证据支撑推理的全景3D检测

    OmniAct3D: Leveraging Foundation Geometry and Evidence-Grounded Reasoning for Panoramic 3D Detection

    [https://arxiv.org/abs/2610.03015](https://arxiv.org/abs/2610.03015)

    OmniAct3D通过ERP射线几何适配器（ERGA-Ray）建模球面视线与周期性空间结构，并借助视觉-动作推理链（VARC）在全景证据中锚定检测假设，从而将基于透视图像训练的视觉基础模型检测器适配到等距柱状全景投影，实现保留VFM先验的连贯360度3D检测。

    

    精确的3D检测对移动具身智能体至关重要，而视觉基础模型（VFMs）提供了可迁移的视觉与几何先验。然而，现有基于VFM的3D检测器依赖于窄视场单目图像或离散的透视视角，限制了连贯的环绕感知；等距柱状投影（ERP）则在单张图像中编码连续的360度场景。直接迁移仍然困难，因为ERP以不同的方式组织几何与视觉信息，使得与物体相关的线索难以建模、定位和保留。我们提出OmniAct3D，一个将基于透视视角训练的VFM检测器适配到ERP的同时保留可迁移VFM先验的框架。为解决几何不匹配问题，ERP射线几何适配器（ERGA-Ray）对球面视线和周期性空间结构进行建模。为了在整个场景上下文中定位证据，视觉-动作推理链（VARC）将每个假设锚定于相关的全景证据中。

    arXiv:2610.03015v1 Announce Type: cross  Abstract: Accurate 3D detection is essential for mobile embodied agents, while Vision Foundation Models (VFMs) offer transferable visual and geometric priors. Yet existing VFM-based 3D detectors rely on narrow-view monocular images or discrete perspective views, limiting coherent surround perception; equirectangular projection (ERP) instead encodes a continuous 360 scene in a single image. Direct transfer remains difficult because ERP organizes geometry and visual information differently, making object-relevant cues hard to model, localize, and preserve. We propose OmniAct3D, a framework that adapts perspective-trained VFM detectors to ERP while preserving transferable VFM priors. To resolve geometric mismatch, the ERP-Ray Geometry Adapter (ERGA-Ray) models spherical viewing rays and periodic spatial structure. To localize evidence in scene-wide context, the Visual-Action Reasoning Chain (VARC) grounds each hypothesis in relevant panoramic evide
    
[^105]: 超越预定义敏感操作点：面向大语言模型智能体的安全感知依赖分析

    Beyond Predefined Sinks: Security-Aware Dependency Analysis for LLM Agents

    [https://arxiv.org/abs/2610.03014](https://arxiv.org/abs/2610.03014)

    该论文提出AgentSecGraph安全感知静态分析框架，通过构建安全感知智能体依赖图（Security-ADG），在传统预定义敏感操作标识之外融合智能体相关性、信任边界、防护证据等多维语义信息，并发布包含67个真实LLM智能体仓库的基准数据集AgentSecBench。

    

    基于大语言模型（LLM）的智能体日益将模型生成的决策与安全敏感的软件能力相连接，例如命令执行、文件系统访问、网络通信、浏览器控制和外部工具。现有分析通常以预定义的敏感操作作为锚点，但仅凭操作本身的标识不足以判定其安全影响。我们提出了AgentSecGraph，一个安全感知的静态分析框架，它为每个安全敏感操作构建以候选为中心的安全感知智能体依赖图（Security-ADG）。该框架在操作标识的基础上，融合了智能体相关性、来源与依赖证据、信任边界上下文、防护措施证据以及外部影响语义等维度的信息。我们进一步推出了AgentSecBench，这是一个涵盖11个生态系统、67个真实世界LLM智能体仓库、共37,542个源文件的语料库。当前分析器已识别出23,866个静态安全敏感操作。

    arXiv:2610.03014v1 Announce Type: cross  Abstract: Large language model (LLM)-based agents increasingly connect model-generated decisions to security-sensitive software capabilities such as command execution, filesystem access, network communication, browser control, and external tools. Existing analyses often use predefined sensitive operations as anchors, but operation identity alone is insufficient to determine security implications.   We present AgentSecGraph, a security-aware static analysis framework that constructs a candidate-centered Security-Aware Agent Dependency Graph (Security-ADG) for each security-sensitive operation. It augments operation identity with agent relevance, source and dependency evidence, trust-boundary context, guard evidence, and external-effect semantics.   We further introduce AgentSecBench, a corpus of 67 real-world LLM-agent repositories spanning 11 ecosystems and 37,542 source files. The current analyzer identifies 23,866 static security-sensitive ope
    
[^106]: AvoKV-E：面向长推理任务的负载感知KV缓存淘汰策略

    AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning

    [https://arxiv.org/abs/2610.03007](https://arxiv.org/abs/2610.03007)

    AvoKV-E是一种无需训练的KV缓存淘汰策略，通过延迟近期状态的淘汰资格，并结合读取压力、键冗余度和值负载潜力对缓存条目排序，在长推理任务中以相同的活跃KV预算达到或超越现有基线方法。

    

    长输出推理将KV缓存的瓶颈从固定的提示转移到了生成的推理轨迹上。现有的推理缓存淘汰方法大多将缓存条目视为路由对象，估计某个旧的键是否仍会被读取、是否会再次出现、或是否可以被替换。这种仅关注路由的视角忽视了两个效应：低注意力的条目可能携带较大的值负载，移除它们会改变未来的预测；而新生成的状态可能在后续查询有机会读取它们之前就被误判为“陈旧”。我们提出了AvoKV-E，这是一种无需训练的淘汰策略，它首先延迟近期状态的可淘汰资格，然后使用候选归一化读取压力、键冗余度和值负载潜力对符合条件的条目进行排序。在不同模型和数据集上的实证评估表明，在相同的活跃KV预算下，AvoKV-E达到或超越了感知冗余、基于重现以及思维自适应的淘汰基线方法，其最大……

    arXiv:2610.03007v1 Announce Type: cross  Abstract: Long-output reasoning shifts the KV-cache bottleneck from the fixed prompt to the generated trace. Existing reasoning-cache eviction methods largely treat cached entries as routing objects, estimating whether an old key will still be read, will recur, or can be replaced. This routing-only view overlooks two effects: low-attention entries can carry large value payloads whose removal changes future predictions, and newly generated states can appear stale before later queries have had a chance to read them. We introduce AvoKV-E, a training-free eviction policy that first delays eligibility for recent states and then ranks eligible entries using candidate-normalized read pressure, key redundancy, and value-payload potential. According to empirical evaluation across different models and datasets, AvoKV-E matches or exceeds redundancy-aware, recurrence-based, and thought-adaptive eviction baselines at matched active-KV budgets, with its larg
    
[^107]: 深度网络的时间几何：用于内在可解释性的训练动态双曲表示

    Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability

    [https://arxiv.org/abs/2610.03000](https://arxiv.org/abs/2610.03000)

    该论文提出利用双曲几何的庞加莱模型构建多层感知机训练过程的时间参数图（即多个训练步骤的快照），以捕捉网络加权拓扑与自组织在训练轨迹中的几何演化，从而超越传统单检查点方法实现内在可解释性。

    

    内在可解释性仍然是一个具有挑战性的问题，尤其是在多层感知机需要在优化环境中进行动态再训练的场景下。本文研究了如何在非欧几里得空间中表示和研究多层感知机及其训练动态；我们的表示采用了双曲几何的庞加莱模型。我们的目标是捕捉其加权拓扑结构和自组织随时间的几何演化。与已有的基于度量的可解释性方法将分析限制在单一检查点不同，我们构建了时间“参数图”，即在 $T$ 步优化/训练过程中对多层感知机进行的时间快照。这反映了这样一种观点：神经网络不仅在其权重中编码信息，还在训练过程中所走过的轨迹中编码信息。借鉴许多复杂网络可以嵌入到隐藏度量空间的思想……

    arXiv:2610.03000v1 Announce Type: cross  Abstract: Intrinsic explainability remains a challenging problem, particularly in contexts where multilayer perceptrons (MLPs) require dynamic re-training within an optimization environment. This paper investigates how MLPs and their training dynamics can be represented and studied in non-Euclidean spaces; our representation features the Poincar\'e model of hyperbolic geometry. We aim to capture the geometric evolution of their weighted topology and self-organization over time. Instead of restricting the analysis to single checkpoints---as per established measure-based explainability methods---we construct temporal \textit{parameter graphs}, i.e., snapshots over time $T$ steps of the optimization/training process for MLPs. This reflects the view that neural networks encode information not only in their weights but also in the trajectory traced during training. Drawing on the idea that many complex networks admit embeddings in hidden metric space
    
[^108]: Sentry：学习在测试时从LLM智能体失败中恢复

    Sentry: Learning to Recover from LLM Agent Failures at Test Time

    [https://arxiv.org/abs/2610.02994](https://arxiv.org/abs/2610.02994)

    提出Sentry——一个与LLM智能体并行运行的失败管理层，将失败经验视为条件性知识，在检测到失败时按需从外部经验手册中检索指导恢复、无奖励验证恢复结果并仅在确认恢复后存储新经验，从而在测试时实现从失败中学习。

    

    LLM智能体经常在任务执行中途因无效的工具调用、重复操作或缺乏依据的推理而失败，从这些失败中学习是提升可靠性的途径。我们发现，失败知识如何传递给智能体与其内容本身同样重要。失败经验是条件性的：如果保留在智能体的上下文中，当对应的失败情形并不存在时它们会误触发，而将它们从不断演化的经验手册中移除反而能提升性能。相比之下，运行时干预只在失败发生时起作用，但不会从修复过程中学习。我们提出，失败知识是一种条件性知识，应当被有条件地暴露，并将这一原则实例化为Sentry——一个与智能体并行运行的失败管理层。当Sentry检测到失败时，它会从外部经验手册中检索匹配的经验来指导恢复，在无需访问任务奖励的情况下验证智能体是否已恢复，并且只有在其确实恢复时才存储新的经验。

    arXiv:2610.02994v1 Announce Type: cross  Abstract: LLM agents often fail mid-task due to invalid tool calls, repeated actions, or poorly grounded reasoning, and learning from these failures is a path to reliability. We find that how failure knowledge reaches the agent matters as much as what it contains. Failure lessons are conditional: kept in the agent's context, they misfire when their failure is absent, and removing them from an evolving playbook improves performance. Runtime interventions, in contrast, act only when a failure occurs but do not learn from their repairs. We argue that failure knowledge is conditional knowledge and should be conditionally exposed, and instantiate this principle in Sentry, a failure-management layer that runs alongside the agent. When Sentry detects a failure, it retrieves matching lessons from an external playbook to guide recovery, verifies without access to task rewards whether the agent recovered, and stores a new lesson only if it did; the full p
    
[^109]: PLCWorld：在闭环工厂仿真中对LLM生成的PLC程序进行基准测试

    PLCWorld: Benchmarking LLM-Generated PLC Programs in Closed-Loop Plant Simulation

    [https://arxiv.org/abs/2610.02982](https://arxiv.org/abs/2610.02982)

    该论文提出PLCWorld，一个将LLM生成的PLC程序执行与闭环工厂仿真及传感器反馈相耦合的基准测试环境，包含100个任务和473个任务-条件对，可分别评估任务成功率与安全违规情况。

    

    可编程逻辑控制器（PLC）通过读取传感器输入并发出控制命令来协调工业设备。评估大型语言模型（LLM）生成的PLC程序是否满足任务要求和安全约束，需要观察其命令如何影响设备与工件的状态。我们提出了PLCWorld，这是一个通用的闭环执行环境和基准测试，它将结构化文本（ST）程序执行与模拟的工厂响应及传感器反馈相耦合。基于从工业PLC程序和工程文档中识别出的控制关系，PLCWorld包含100个合成任务和473个注册的任务-条件对，涵盖运动控制与物料搬运两大类别，其难度由控制依赖范围来定义。统一的评测协议分别报告任务成功率（Task Success）和安全违规（Safety Violation）。验证工作结合了从业者评审、参考程序与替代程序、针对性反例、特定的……（摘要在此处截断）

    arXiv:2610.02982v1 Announce Type: new  Abstract: Programmable logic controllers (PLCs) coordinate industrial equipment by reading sensor inputs and issuing control commands. Evaluating whether large language model (LLM)-generated PLC programs satisfy task requirements and safety constraints requires observing how their commands affect device and workpiece states. We introduce PLCWorld, a common closed-loop execution environment and benchmark that couples Structured Text (ST) execution with simulated plant responses and sensor feedback. Grounded in control relations identified in industrial PLC programs and engineering documentation, PLCWorld contains 100 synthetic tasks and 473 registered task-condition pairs across Motion Control and Material Handling, with difficulty defined by control-dependency scope. A common protocol reports Task Success and Safety Violation separately. Validation combines practitioner review, reference and alternative programs, targeted counterexamples, specific
    
[^110]: 基于Cut统计量的无源域适应中相互纠错的保障方法

    Safeguarding Mutual Correction in Source-Free Domain Adaptation via Cut Statistics

    [https://arxiv.org/abs/2610.02981](https://arxiv.org/abs/2610.02981)

    该论文发现源预训练模型与视觉-语言模型具有互补的失败模式，并利用Cut统计量在无标签条件下识别哪个模型的预测更可靠，从而在无源域适应中实现两个模型之间的相互纠错。

    

    无源域适应（SFDA）旨在将源域预训练的模型适配到无标签的目标域，且无需访问原始源域数据。早期的单模型方法依赖自我精炼，但其天生容易受到确认偏差的影响，难以纠正自身的系统性错误。为克服这一局限，近期方法引入视觉-语言（ViL）模型作为外部知识来源。然而，这些方法主要运行在单向范式下，即仅使用ViL模型来监督源域预训练模型。这忽视了一个关键的结构特性：两个模型展现出截然不同的失败模式——当一个模型产生错误预测时，另一个模型可能给出正确预测，从而为目标域内的相互纠错创造了天然机会。然而，在没有真实标签的情况下，判断哪个模型在任一给定样本上是正确的并非易事，而简单地直接交换预测……（摘要在此处被截断）

    arXiv:2610.02981v1 Announce Type: new  Abstract: Source-Free Domain Adaptation (SFDA) aims to adapt a source-pretrained model to an unlabeled target domain without access to the original source domain. While early single-model approaches rely on self-refinement, they are inherently susceptible to confirmation bias and struggle to correct their own systematic errors. To overcome this limitation, recent methods introduce Vision-Language (ViL) models as external knowledge sources. However, these approaches operate in a largely unidirectional paradigm, using the ViL model primarily to supervise the source-pretrained model. This overlooks a key structural property: the two models exhibit distinct failure modes -- where one produces an incorrect prediction, the other may produce a correct one, creating a natural opportunity for mutual correction within the target domain. Yet, without ground-truth labels, identifying which model is correct on any given sample is non-trivial, and naively excha
    
[^111]: RASPER：面向电子病历结局预测的奖励对齐临床笔记摘要方法

    RASPER: Reward-Aligned Summarization of Clinical Notes for EHR Outcome Prediction

    [https://arxiv.org/abs/2610.02979](https://arxiv.org/abs/2610.02979)

    RASPER提出了一种奖励对齐的摘要框架，利用下游预测器的反馈作为强化学习奖励，训练LLM摘要器从出院记录中提取对临床结局预测真正有用的证据，而非生成通用的流畅摘要。

    

    电子健康记录（EHR）中的非结构化出院记录通常携带与结构化医学编码互补的信号，其中包含标准化的队列级编码无法捕捉的患者特异性证据。然而，笔记中的这些证据往往埋藏在冗长、嘈杂的文本中，而这些文本并非针对任何特定临床预测任务而有意撰写。摘要是一种显而易见的缓解手段，但通用摘要方法以流畅性而非预测结局为目标进行优化，往往会遗漏决定性证据，同时保留看似合理但缺乏信息量的细节。为此，我们提出了RASPER（面向EHR预测的奖励对齐摘要器），它直接针对下游临床任务来优化笔记摘要。RASPER采用可调优的基于大语言模型（LLM）的摘要器从出院记录中提取与任务相关的证据，并通过基于预测反馈的强化学习对其进行训练，奖励信号由下游预测器导出。

    arXiv:2610.02979v1 Announce Type: new  Abstract: Unstructured discharge notes in Electronic Health Records (EHRs) often carry signal complementary to structured medical codes, holding patient-specific evidence that standardized cohort-level codes alone cannot capture. However, this evidence in notes is frequently buried in lengthy, noisy text that is not intentionally written with any specific clinical prediction in mind. Summarization is an obvious mitigation, but generic summaries, tuned for fluency rather than the outcome, routinely omit decisive evidence while retaining plausible but uninformative detail. To this end, we propose RASPER, a Reward-Aligned Summarizer for Prediction in EHR, that optimizes note summarization directly against the downstream clinical task. RASPER employs a tunable LLM-based summarizer to extract task-relevant evidence from discharge notes and trains it via reinforcement learning from prediction feedback, using a reward derived from the downstream predicto
    
[^112]: 用于缓解视听幻觉的相关证据解码

    Relevant Evidence Decoding for Audio-Visual Hallucination Mitigation

    [https://arxiv.org/abs/2610.02976](https://arxiv.org/abs/2610.02976)

    提出了一种无需训练的相关证据解码方法RED，通过识别与问题相关的音视频证据并选择性地增强其贡献，来缓解视听大语言模型中的跨模态幻觉问题。

    

    视听大语言模型仍然容易出现跨模态幻觉，即一种模态错误地影响对另一种模态的预测。尽管对比解码可以减少视觉-语言模型中的幻觉，但将其直接扩展到视听大语言模型忽略了一个关键挑战：不同的问题需要不同的感知证据，包括音频、视频或两者的交互。值得注意的是，我们观察到，即使模型能够从单一的信息模态中恢复正确答案，联合音视频推理也可能削弱预测结果。例如，当被问及听到的是哪种乐器时，模型可能仅凭音频就能正确预测出小提琴；但一旦加入显示吉他的视频，模型对小提琴的置信度就可能下降。在本文中，我们提出了相关证据解码，这是一种无需训练的方法，能够识别与问题相关的证据并选择性地增强其贡献。RED使用逐点互信息来...

    arXiv:2610.02976v1 Announce Type: new  Abstract: Audio-Visual Large Language Models (AV-LLMs) remain prone to cross-modal hallucinations, where one modality incorrectly affects predictions about another. Although contrastive decoding reduces hallucinations in vision-language models, its direct extension to AV-LLMs overlooks a key challenge: different questions require different perceptual evidence, including audio, video, or their interaction. Notably, we observe that joint audio-visual inference can weaken the prediction even when a model can recover the correct answer from a single informative modality. For example, when asked which instrument is heard, a model may correctly predict violin from the audio alone. Once a video showing a guitar is added, its confidence in violin may drop. In this paper, we introduce Relevant Evidence Decoding (RED), a training-free method that identifies question-relevant evidence and selectively strengthens its contribution. RED uses pointwise mutual in
    
[^113]: 基于不完美代理奖励的可靠自进化

    Reliable Self-Evolution with Imperfect Proxy Rewards

    [https://arxiv.org/abs/2610.02975](https://arxiv.org/abs/2610.02975)

    该论文提出保形区间驱动的自进化方法（CISE），通过条件保形推断和在线密度比估计构建候选特定的奖励区间，以应对不完美代理奖励导致的假阳性问题，从而实现更可靠的LLM自进化搜索。

    

    基于大语言模型（LLM）的自进化搜索是实现科学发现的一种有前景的方法。然而，在某些领域中，对每个候选方案进行高保真度评估的成本过高。因此，此类环境下的自进化系统不得不依赖低成本但不完美的代理奖励，而这类代理奖励可能会给不可行的候选方案赋予高分。这些假阳性结果可能同时污染最终输出和用于指导后续生成的反馈。这促使我们引入统计校准的奖励区间，以实现更可靠的自进化搜索。我们提出了保形区间驱动的自进化方法（Conformal Interval-Driven Self-Evolution, CISE），该方法利用条件保形推断和逐迭代的在线密度比估计来构建针对特定候选方案的奖励区间。CISE在进化反馈中使用保守的基于区间的奖励，并且仅当所有所需属性区间完全落入各自可行区域内时才返回候选方案……

    arXiv:2610.02975v1 Announce Type: new  Abstract: Large language model (LLM)-based self-evolving search is a promising approach to scientific discovery. However, high-fidelity evaluation of every candidate is prohibitively expensive in some domains. Self-evolving systems in such settings therefore rely on low-cost but imperfect proxy rewards, which may assign high scores to infeasible candidates. These false positives may contaminate both the final output and the feedback used to guide subsequent generations. This motivates statistically calibrated reward intervals for more reliable self-evolving search. We propose Conformal Interval-Driven Self-Evolution (CISE), which constructs candidate-specific reward intervals using conditional conformal inference and iteration-wise online density-ratio estimation. CISE uses conservative interval-based rewards for evolutionary feedback and returns candidates only when all required property intervals lie entirely within their respective feasible reg
    
[^114]: CreateScore：面向基于大语言模型简历筛选的领域理论驱动贝叶斯路由

    CreateScore: Domain-Theory-Informed Bayesian Routing for LLM-Based CV Screening

    [https://arxiv.org/abs/2610.02972](https://arxiv.org/abs/2610.02972)

    CreateScore 提出了一种由领域理论指导的贝叶斯网络路由方法，利用后验不确定性将低风险的简历筛选决策交由本地 8B 模型处理、将不确定的决策升级至 120B 大模型，在 77.7% 的决策可本地解决的前提下显著降低 LLM 简历筛选的成本。

    

    大语言模型（LLM）可以支持基于评分标准的简历（CV）筛选，但将高能力模型应用于每位候选者和每个评分标准成本高昂。我们提出了 CreateScore，一个用于标准级 LLM 路由的、由领域理论指导的贝叶斯网络。一个手工指定的有向无环图（DAG）结合 Dirichlet-多项式条件概率表，将简历证据转化为后验不确定性；低不确定性的决策由本地 8B 模型解决，而不确定的决策则升级至 120B 参考模型。该图具有因果动机，但系统执行的是标准贝叶斯条件化，而非因果推断。升级阈值在训练折上校准（目标：70% 在本地解决）后被冻结。在 200 份合成的数据科学简历上（139 份训练、61 份测试候选人，五个评分标准），77.7% 的标准决策在本地得到解决（305 个中的 237 个）。相对于由 120B 模型裁决……的参考条件（原文在此处截断）。

    arXiv:2610.02972v1 Announce Type: new  Abstract: Large language models (LLMs) can support rubric-based screening of CVs, but applying a high-capability model to every candidate and criterion is costly. We present CreateScore, a domain-theory-informed Bayesian network for criterion-level LLM routing. A hand-specified directed acyclic graph with Dirichlet-multinomial conditional probability tables converts CV evidence into posterior uncertainty; low-uncertainty decisions are resolved by a local 8B model and uncertain ones are escalated to a 120B reference model. The graph is causally motivated, but the system performs standard Bayesian conditioning, not causal inference. The escalation threshold is calibrated on a training fold (target: 70% resolved locally) and then frozen. On 200 synthetic Data Science CVs (139 training and 61 test candidates, five criteria), 77.7% of criterion decisions were resolved locally (237 of 305). Relative to a reference condition in which the 120B model adjud
    
[^115]: 基于证据而非仅凭理由的推理：面向大语言模型推荐的可验证偏好证明

    Reasoning with Evidence, Not Merely Rationales: Verifiable Preference Proofs for LLM-Based Recommendation

    [https://arxiv.org/abs/2610.02968](https://arxiv.org/abs/2610.02968)

    提出PROVE-REC框架，通过两阶段流程生成与所选证据相链接的可验证偏好证明，并借助屏蔽对比实验同时验证证据对接地声明的支持程度以及证明对最终推荐的影响，从而弥合LLM推荐中推理说明与实际所用信息之间的“接地-影响鸿沟”。

    

    大语言模型（LLM）能够从交互历史和评论中推断用户偏好，然而其生成的推理说明可能并不反映实际用于推荐的信息。一个偏好声明可能仅得到其所选证据的微弱支持，也可能对最终排序几乎没有影响。我们将这两类失效称为“接地-影响鸿沟”。我们提出了PROVE-REC，一个面向基于LLM推荐的可验证偏好推理通用框架。Pass A 将完整的目标前历史转换为一份紧凑的偏好证明，该证明由与所选证据条目相关联的正面声明和规避声明组成。Pass B 仅使用该证明及其所选证据来预测下一个物品，从而防止推荐器绕过推理路径。为验证从证据到证明的接地性，我们比较了屏蔽所选证据与屏蔽一个可比对照条目所产生的效果差异。为验证从证明到推荐的影响，……（摘要原文在此处截断）

    arXiv:2610.02968v1 Announce Type: new  Abstract: Large language models (LLMs) can infer user preferences from interaction histories and reviews, yet the rationales they generate may not reflect the information actually used for recommendation. A preference claim may be weakly supported by its selected evidence, or may have little effect on the final ranking. We refer to these two failures as the grounding-influence gap. We introduce PROVE-REC, a general framework for verifiable preference reasoning in LLM-based recommendation. Pass A converts the complete pre-target history into a compact preference proof consisting of positive and avoidance claims linked to selected evidence entries. Pass B predicts the next item using only the proof and its selected evidence, preventing the recommender from bypassing the reasoning path. To verify evidence-to-proof grounding, we compare the effect of masking selected evidence with masking a comparable control entry. To verify proof-to-recommendation i
    
[^116]: 通过组合偏好奖励与规则奖励对前沿文生图模型进行后训练

    Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards

    [https://arxiv.org/abs/2610.02967](https://arxiv.org/abs/2610.02967)

    该论文提出了一种将基于大规模人类偏好数据训练的偏好奖励与用于评估提示词忠实度的规则奖励相结合的后训练方案，并设计了优于简单加权平均的奖励组合策略，从而全面改进前沿文生图模型的生成质量。

    

    近期的文生图模型已经达到了出色的视觉质量，但通过后训练来改进这些模型仍然具有挑战性，因为没有任何单一的奖励信号能够涵盖人类偏好的全部范围。在这项工作中，我们基于互补奖励信号的组合，为开放域文生图生成开发了一种简单而有效的后训练方案。我们的奖励系统由两个主要部分组成：一是偏好奖励，它基于大规模人类偏好数据、使用 Bradley-Terry 目标进行训练，用以捕捉人类的整体审美与感知偏好；二是基于评分规则的奖励，它显式地评估提示词忠实度以及其他期望属性，同时提供防止奖励作弊的保障。一个关键挑战在于如何组合这些异构的奖励信号。我们证明了简单的加权平均会导致次优的优化行为，并提出了一种简单的奖励组合策略

    arXiv:2610.02967v1 Announce Type: cross  Abstract: Recent text-to-image generation models have achieved remarkable visual quality, but improving them through post-training remains challenging because no single reward signal captures the full range of human preference. In this work, we develop a simple and effective post-training recipe for open-domain text-to-image generation based on the composition of complementary reward signals. Our reward system consists of two main components: a preference reward, trained on large-scale human preference data using a Bradley-Terry objective to capture overall human aesthetic and perceptual preferences, and rubric-based rewards, which explicitly evaluate prompt faithfulness and other desirable properties while providing safeguards against reward hacking. A key challenge is how to combine these heterogeneous reward signals. We show that a naive weighted average leads to suboptimal optimization behavior, and propose a simple reward composition strate
    
[^117]: 面向多智能体系统的动态专家剪枝

    Dynamic Expert Pruning for Multi-Agent Systems

    [https://arxiv.org/abs/2610.02951](https://arxiv.org/abs/2610.02951)

    提出动态专家剪枝（DEP）方法，利用智能体的系统与任务提示动态识别并按需剪枝专家，解决了静态专家剪枝在多智能体异构工作负载下失效的问题。

    

    混合专家架构通过在每个 token 上仅激活少数专家来高效扩展语言模型，但这种节省仅限于计算方面：每个专家都必须驻留在加速器上，因此内存限制了这些模型可部署的范围。专家剪枝可以减少这种内存占用，然而现有方法都是静态的——一个在离线阶段校准的单一掩码，会被应用于模型后续的所有请求。当工作负载是异构的时，这一假设可能失效，最突出的体现就是多智能体系统：一个骨干网络同时服务于多个任务和角色。我们的分析表明，不同的任务和角色会启用不同的专家，而静态方法却为所有任务分配同一个固定的专家子集。因此，我们提出了动态专家剪枝（DEP），它基于我们在此确立的一个发现：智能体的系统提示和任务提示本身就足以识别出该智能体及其任务所需的专家……

    arXiv:2610.02951v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures scale language models efficiently by activating only a few experts per token, but the saving is confined to computation: every expert must stay resident on the accelerator, so memory bounds where these models can be deployed. Expert pruning reduces this footprint, yet existing methods are static --- a single mask, calibrated offline, is applied to the model for every subsequent request. This assumption can fail when the workload is heterogeneous, most prominently in multi-agent systems, where one backbone serves many tasks and roles at once: our analysis shows that different tasks and roles recruit different experts, while static methods assign one fixed subset to all of them. We therefore propose Dynamic Expert Pruning (DEP), which rests on a finding we establish here: an agent's system and task prompts are by themselves sufficient to identify the experts that agent and its task require, since th
    
[^118]: 面向数学研究智能体的持续图记忆系统

    Continual Graph Memory for Mathematical Research Agents

    [https://arxiv.org/abs/2610.02945](https://arxiv.org/abs/2610.02945)

    提出 Ansatz——一个基于“持续图记忆”的数学研究智能体，通过可演化、跨问题的图结构记忆系统显式组织整个证明搜索过程，并复用先前问题的探索知识，从而有效管理海量中间证明结果。

    

    利用前沿智能体框架来解决数学研究问题，已成为推动数学发展的有效手段。然而，解决数学领域的前沿问题可能需要大量智能体长时间并行工作以构建证明，从而产生海量的中间证明结果。在长周期的证明搜索过程中组织这些中间结果，并复用先前探索所获得的知识，仍然是重大的挑战。我们提出了 Ansatz——一个围绕“持续图记忆”构建的数学研究智能体。持续图记忆是一种基于图、可演化、跨问题的数学研究记忆系统，它显式地组织整个证明搜索过程，并复用来自先前问题探索轨迹的信息。具体而言，我们开发了一个统一的图记忆，用于表示所有中间探索结果，包括事实、计划和反例……

    arXiv:2610.02945v1 Announce Type: new  Abstract: Using frontier agent harnesses to tackle mathematical research problems has emerged as an effective means of advancing mathematics. However, solving frontier problems in mathematics may require a massive number of agents working in parallel for extended periods to construct proofs, thereby generating an enormous volume of intermediate proof results. Organizing these intermediate results throughout a long-horizon proof-search process and reusing knowledge gained from prior explorations remain major challenges. We present Ansatz, a mathematical research agent built around Continual Graph Memory, a graph-based, evolvable, cross-problem mathematical research memory system that explicitly organizes the entire proof search process and reuses information from exploration trajectories of previous problems. Specifically, we develop a unified graph memory that represents all intermediate exploration results, including facts, plans, and counterexam
    
[^119]: 何时将计算机使用智能体编译为程序？为提升令牌效率而测量回报并做出编译决策

    When to Compile a Computer-Use Agent? Measuring Payback and Making Compilation Decisions for Token Efficiency

    [https://arxiv.org/abs/2610.02932](https://arxiv.org/abs/2610.02932)

    本文提出PACE系统，通过记录成功与失败的编译成本、比较智能体与程序的执行开销，并基于回报预测用在线算法决定何时将反复执行的GUI操作编译为程序，从而提升令牌使用效率。

    

    将智能体反复执行的GUI操作流程编译成程序可以降低其令牌成本。然而，测量回报并决定何时进行编译面临两大挑战。首先，编译成本是不确定的，因为编译尝试可能需要修复，且仍可能无法生成可用的程序。其次，未来的复用情况是未知的，因为任务可能不再到来，或者GUI的变化（漂移）可能导致程序失效。为应对这些挑战，我们提出了PACE（基于经验且回报感知的编译，Payback-Aware Compilation from Experience），一个包含测量协议和在线编译算法的系统。该测量协议记录成功和失败的编译成本，并在匹配的任务输入上比较智能体与程序的执行成本，以估算每次使用的节省量和回报次数。基于这些测量数据，在线算法根据过去的任务到达情况和编译结果，将预估的未来节省与编译成本（包括失败的尝试）进行比较。

    arXiv:2610.02932v1 Announce Type: new  Abstract: Compiling GUI procedures that agents execute repeatedly into programs can reduce their token costs. However, measuring payback and deciding when to compile have two challenges. First, compilation costs are uncertain because attempts can require repair and still fail to produce a usable program. Second, future reuse is unknown because tasks may stop arriving or GUI drift may stop the program from working. To address these challenges, we propose PACE (Payback-Aware Compilation from Experience), a system with a measurement protocol and an online compilation algorithm. The measurement protocol records successful and failed compilation costs, and compares agent and program execution costs on matched task inputs to estimate per-use savings and payback counts. Using these measurements, the online algorithm compares estimated future savings with compilation costs, including failed attempts, based on past task arrivals and compilation outcomes. I
    
[^120]: 判别代理基础设施验证套件中的测试夹具覆盖

    Discriminating Fixture Coverage in Agent-Infrastructure Verification Suites

    [https://arxiv.org/abs/2610.02928](https://arxiv.org/abs/2610.02928)

    用变异分析衡量“在正确实现上通过、在缺陷实现上失败”这一标准证据的价值，发现该证据无法保证测试夹具覆盖——即使修复后的套件仍有五个对抗性变异体存活，其中三个因没有任何夹具能激活它们而从未暴露。

    

    arXiv:2610.02928v1 公告类型： cross。不变量测试套件和运行时监控器越来越多地被用于把关代理的部署决策，而为任何特定套件提供的证据几乎总是一个单一观察结果：它在一个被认为正确的实现上通过，在一个被认为有缺陷的实现上失败。我们衡量这一观察结果究竟有多大价值。通过对一个面向多会话代理状态投影层的不变量套件应用变异分析，我们首先发现这种标准验证所认证的套件中，一个移除了事件身份去重逻辑的一阶变异体通过了全部检查而存活。随后，我们冻结修复后的包含十二项检查的套件，记录其哈希值，并用一位对抗性读者（该读者未设计该套件的任何测试夹具）指定的十个变异体对其运行一次：它击杀了其中五个。对五个存活变异体相对参考实现进行插桩分析表明，它们以两种截然不同的方式失败，而非一种：其中三个从未被激活，因为没有任何测试夹具提供能让变异代码表现出不同行为的输入。

    arXiv:2610.02928v1 Announce Type: cross  Abstract: Invariant suites and runtime monitors increasingly gate agent deployment decisions, and the evidence offered for any particular suite is almost always a single observation: it passes an implementation believed correct and fails one believed broken. We measure what that observation is worth. Applying mutation analysis to an invariant suite for a multi-session agent state-projection layer, we first find that this standard validation certifies a suite in which a first-order mutant removing event-identity deduplication survives every check. We then freeze the repaired twelve-check suite, record its hash, and run it once against ten mutants specified by an adversarial reader who designed none of its fixtures: it kills five. Instrumenting the five survivors against the reference shows they fail in two distinct ways, not one. Three are never activated, because no fixture supplies an input on which the mutated code behaves differently at all. 
    
[^121]: 面向智能体安全误报审计的正例-未标注学习

    Positive-Unlabeled Learning for Agent Safety False Alarm Auditing

    [https://arxiv.org/abs/2610.02925](https://arxiv.org/abs/2610.02925)

    该论文将智能体安全监控的误报审计建模为正例-未标注（PU）排序问题，提出一种两阶段的 Trust-aware PU 框架来克服监控器引发的选择偏差，从而更准确地从警报中识别出安全误报并降低人工审查成本。

    

    安全监控器有助于保障与外部工具和环境交互的语言模型智能体的安全，但保守的监控策略会产生大量误报，消耗大量审查资源并削弱对警报的信任。由于虚假警报与真实警报在监控器的原始分数中常常相互交织，获得可靠的判定阈值仍需要大量的人工核验。在实践中，可能仅存在少量经过验证的安全且未触发警报的轨迹，而警报本身仍处于未标注状态，这自然地将误报审计建模为一个正例-未标注（PU）排序问题。关键挑战在于监控器引发的选择偏差：观测到的安全参考样本是被监控器所接受的，而我们感兴趣的隐藏安全警报恰恰是被其错误标记的那些，这使得观测到的正例对需要恢复的目标正例代表性很差。为应对这一挑战，我们提出了一个两阶段框架，其中 Trust-aware PU（摘要在此处截断）

    arXiv:2610.02925v1 Announce Type: new  Abstract: Safety monitors help safeguard language-model agents interacting with external tools and environments, but conservative monitoring can generate many false alarms, consuming extensive review resources and weakening trust in alerts. Because false and genuine alarms often remain interleaved in native monitor scores, obtaining a reliable cutoff still requires substantial manual verification. In practice, a small set of verified-safe non-alarmed trajectories may be available while alarms remain unlabeled, naturally casting false-alarm auditing as a positive-unlabeled (PU) ranking problem. The key challenge is monitor-induced selection, since observed safe references are accepted by the monitor, while the hidden safe alarms of interest are precisely those it incorrectly flags, making the observed positives poorly representative of the positives to be recovered. To address this challenge, we propose a two-stage framework in which Trust-aware PU
    
[^122]: HASTE：利用稀疏证据演化智能体线束以对抗新兴攻击

    HASTE: Evolving Agent Harnesses Against Emerging Attacks Using Sparse Evidence

    [https://arxiv.org/abs/2610.02920](https://arxiv.org/abs/2610.02920)

    HASTE 提出了一种多智能体框架，通过安全规范生成与攻击用例生成的对抗性交互，从稀疏的威胁证据中自动演化智能体线束，使其能够防御最初观察到的证据之外的新兴攻击。

    

    智能体线束通过强制执行安全约束以阻止不安全行为，在防御中发挥着关键作用。然而，快速涌现的攻击超出了人工调整线束的速度，这促使人们探索自动化的线束演化。但可用于线束演化的信号往往十分稀疏，例如威胁报告和预印本论文中仅有的简短描述或少量攻击示例。为解决这一局限，我们提出了 HASTE，这是一个多智能体框架，它通过安全规范生成与攻击用例生成之间的对抗性交互，从稀疏的威胁证据中演化智能体线束。安全规范引导线束更新以修复已识别的安全漏洞，而攻击用例则在每次更新后探测残余的安全漏洞。通过将评估结果反馈到这两个过程中，HASTE 使线束能够针对超出最初观察证据之外的新兴攻击进行演化。实验结果（摘要原文至此中断）

    arXiv:2610.02920v1 Announce Type: new  Abstract: Agent harnesses play a critical role in defenses by enforcing safety constraints to prevent unsafe actions. However, rapidly emerging attacks outpace manual harness adaptation, motivating automated harness evolution. Yet the signals available for harness evolution are often sparse, such as brief descriptions or a few attack examples in threat reports and preprints. To address this limitation, we introduce HASTE, a multi-agent framework that evolves agent harnesses from sparse threat evidence through an adversarial interplay between safety-specification generation and attack-case generation. Safety specifications guide harness updates toward addressing identified safety vulnerabilities, while attack cases probe for remaining safety vulnerabilities after each update. By feeding evaluation outcomes back into both processes, HASTE enables harness evolution against emerging attacks beyond the initially observed evidence. Experimental results 
    
[^123]: 频率并非敏感度：识别稀疏MoE大语言模型中的安全敏感专家

    Frequency Is Not Sensitivity Identifying Safety-Sensitive Experts in Sparse MoE LLM

    [https://arxiv.org/abs/2610.02910](https://arxiv.org/abs/2610.02910)

    该论文提出用路由器梯度敏感度（即序列损失对专家门控权重的敏感度）替代传统的激活频率来识别稀疏MoE大语言模型中的安全关键专家，实验表明该方法在五种架构上能更准确地预测抑制哪些专家会削弱模型的安全拒绝能力。

    

    抑制少量被路由选择的专家即可在不重新训练的情况下削弱稀疏混合专家语言模型的安全行为。因此，应该抑制哪些专家是一个安全问题，而常见的答案是激活频率，但频率衡量的是使用情况，而非影响力。我们测试了一种替代方法：路由器梯度敏感度，即序列损失对选择专家的门控权重的敏感度。在五种MoE架构上，我们基于500个良性提示和500个恶意提示分别按这两种信号对专家进行排序，并在两种预算下（相同专家数量和相同名义恶意路由流量1%-5%）在100个保留的恶意提示上测量模型的拒绝率。在每种预算下，基于路由器梯度选择的专家抑制在25个条件中的24个里比激活频率更能降低拒绝率，并且在全部25个条件中都超过十次随机试验的均值。最大效应出现在OLMoE中，其拒绝率从100个提示中的34个降至9个（相对下降73.53%），且没有……（原文截断）

    arXiv:2610.02910v1 Announce Type: new  Abstract: Suppressing a small set of routed experts can weaken the safety behavior of a sparse Mixture-of-Experts (MoE) language model without retraining. Which experts to suppress is therefore a security question, and the usual answer is activation frequency, but frequency measures use, not influence. We test an alternative: router-gradient sensitivity, the sensitivity of the sequence loss to the gate weights that select an expert. Across five MoE architectures, we rank experts by each signal on 500 benign and 500 malicious prompts and measure refusal on 100 held-out malicious prompts under two budgets: equal expert counts and equal nominal malicious routing traffic (1%-5%). Under each of the two budgets, router-gradient selection reduces refusals more than activation in 24 of 25 conditions, and more than a ten-trial random mean in all 25. The largest effect is in OLMoE, where refusals fall from 34 to 9 of 100 prompts (73.53% relative) with no de
    
[^124]: LUMOS：在大语言模型中从训练数据追踪参数化知识到行为输出

    LUMOS: Tracing Parametric Knowledge from Training Data to Behavioral Outputs in LLMs

    [https://arxiv.org/abs/2610.02902](https://arxiv.org/abs/2610.02902)

    LUMOS诊断框架利用完全透明的OLMo 2训练语料库，沿“训练数据暴露→行为输出”的因果链追踪大语言模型的参数化知识，揭示模型内部能高可分性地编码罕见事实（84%）但在行为上表达不足（54%），且该检索差距随模型规模增大而缩小。

    

    当前对大语言模型参数化知识的分析大多以输出为中心，在没有验证模型实际训练内容的情况下就对模型“知道什么”下结论。这使得一些根本性问题——例如模型的正确回答究竟是反映了真正的泛化能力还是死记硬背——只能停留在推测而非证据层面。为了消除这些模糊性，我们提出了LUMOS，一个沿“训练数据暴露→行为输出”因果链追踪知识的诊断框架，并利用训练语料库完全透明的OLMo 2模型。通过将分析建立在经过验证的训练暴露之上，我们发现模型内部以高可分性（84%）编码罕见事实，却无法在行为层面表达它们（仅54%），不过这一检索差距会随模型规模的增大而缩小。此外，当模型被要求对自己的答案进行自我反思时，它们在训练过的内容上表现可靠（83%），但在未训练过的内容上则下降至随机基线水平（49%）。

    arXiv:2610.02902v1 Announce Type: new  Abstract: Current analyses of LLMs' parametric knowledge are largely output-centric, drawing conclusions about what a model knows without verifying what it was actually trained on. This leaves fundamental questions, such as whether a correct response reflects genuine generalization or rote memorization, grounded in speculation rather than evidence. To resolve these ambiguities, we introduce LUMOS, a diagnostic framework that traces knowledge along the causal chain from training-data exposure to behavioral output, leveraging OLMo 2 with its fully transparent training corpus. By grounding analysis in verified exposure, we reveal that models internally encode rare facts with high separability (84%) yet fail to express them behaviorally (54%), though this retrieval gap narrows with scale. Furthermore, when models are asked to self-reflect on their own answers, they perform reliably on trained content (83%) but drop to random-baseline levels (49%) on u
    
[^125]: 写入时的解读：面向多目标智能体记忆的策略消融研究

    Interpreting at Write Time: A Policy Ablation for Multi-Goal Agent Memory

    [https://arxiv.org/abs/2610.02897](https://arxiv.org/abs/2610.02897)

    该论文提出三种智能体多目标记忆摘要写入策略（无目标通用摘要、单一全目标摘要、按目标分别摘要后合并读取），并通过固定读取步骤的消融实验证明，为不同目标写入的摘要内容会显著分化，说明记忆写入时就应针对目标进行解读与取舍。

    

    一个长期运行的助手无法保留它见过的所有内容，因此它需要进行摘要。然而摘要并非中立的：保留什么内容是根据某种关于“这份记录的用途”的判断来选择的，而且这个选择只做一次——在任何人都还不知道用户的哪些长期目标将来会提问之前就已经做出。不同的目标很少对“发生了什么”产生分歧，它们的分歧在于哪些部分值得占用存储空间。一旦历史记录长到无法重新通读，摘要就取代了原始信息流，而摘要所遗漏的内容就永久消失了。我们要问的是：当记忆同时服务于多个长期目标时，它应该为“什么”而摘要？三种策略给出了不同的回答：不带任何目标视角地进行摘要、写一份覆盖所有目标的统一摘要，或者为每个目标各写一份摘要并在读取时将它们合并使用。我们在多个模型和多种事件流上对这三种策略进行比较，并保持读取步骤完全固定，从而使写入策略成为唯一的变量。实验表明，各目标确实会彼此分化：为不同目标撰写的摘要之间的重合程度低于……

    arXiv:2610.02897v1 Announce Type: new  Abstract: A long-running assistant cannot keep everything it has seen, so it summarises. Summarising is not neutral: what is kept is chosen against some notion of what the record is for, and that choice is made once, before anyone knows which of the user's standing goals will ask. Goals rarely disagree about what happened. They disagree about which parts of it were worth the space. Once the history is too long to re-read, the summary replaces the stream, and whatever it left out is gone. We ask what a memory should summarise for when it serves several standing goals at once. Three policies answer differently: summarise with no goal in view, write one summary covering every goal, or write one summary per goal and read them together. We compare them across several models and event streams, holding the read step fixed so that only the write differs. The goals do pull apart: summaries written for different goals overlap each other less than a summary 
    
[^126]: 何时可以信任匹配原则？有限样本与模型不确定性下的鲁棒部署几何

    When Can We Trust the Matching Principle? Robust Deployment Geometry Under Finite-Sample and Model Uncertainty

    [https://arxiv.org/abs/2610.02894](https://arxiv.org/abs/2610.02894)

    该论文提出用信任比率 tau = ε/γ 来量化匹配原则何时可靠（估计投影匹配的漂移在 Davis-Kahan 分离区域内按 O(τ²) 增长），并据此设计置信度校准匹配（CCM）策略：τ 小时进行方向性匹配，τ 大时渐进各向同性扩散惩罚，从而在有限样本与模型不确定性下实现鲁棒部署。

    

    arXiv:2610.02894v1 公告类型：cross 摘要：只匹配你能识别的几何结构；否则将惩罚分散开。我们用信任比率 tau = epsilon / gamma（估计不确定性除以谱分离度）来量化这一决策。在线性二次匹配响应下，对于位于所选 top-r 部署子空间中的探针，估计投影匹配相对于oracle投影匹配的漂移按 tau^2 量级缩放——即在 Davis-Kahan 分离区域 tau < 1/2 内为 O(tau^2)，实际可用性取决于常数因子。置信度校准匹配（CCM）将 tau 转化为一种策略——当 tau 较小时采用方向性匹配，否则渐进地转为各向同性扩散——其阈值来自校准而非定理（匹配对应分离区域；软匹配主要是启发式的）。实验展示了两种情形，包括 UCI HAR 嵌入场景，其中始终匹配在每个单元上都比弃权表现更差。

    arXiv:2610.02894v1 Announce Type: cross  Abstract: Match only geometry you can identify; otherwise spread the penalty. We quantify that decision by the trust ratio tau = epsilon / gamma (estimation uncertainty over spectral separation). Under the linear-quadratic Matching response, oracle-relative drift between estimated and oracle projector matching scales as tau^2 for probes in the chosen top-r deployment subspace -- O(tau^2) in the Davis-Kahan separation region tau < 1/2, with practical usefulness depending on constants. Confidence-Calibrated Matching (CCM) turns tau into a policy -- directional when tau is small, progressively isotropic when not -- with thresholds from calibration, not from the theorem (match sits in the separation region; soft is mostly heuristic). Experiments show both regimes, including UCI HAR embeddings where always-match is worse than abstain on every cell.
    
[^127]: 通过因果不变掩码揭示多模态大语言模型中的认知不确定性

    Revealing Epistemic Uncertainty in MLLMs via Causal-Invariant Masking

    [https://arxiv.org/abs/2610.02887](https://arxiv.org/abs/2610.02887)

    提出因果不变掩码方法与语义散度指标，将源于模型局限性的认知不确定性与数据模糊导致的偶然不确定性解耦，从而有效检测多模态大语言模型因依赖表面关联而产生的幻觉风险。

    

    多模态大语言模型存在幻觉问题，因此亟需不确定性量化以确保其可靠部署。然而，现有方法难以检测由表面关联引起的不确定性，尤其是在与查询相关的信号较弱时。我们主要将这一问题归因于现有方法偏向由数据模糊性导致的偶然不确定性，而忽视了源于模型局限性的认知不确定性。为了进一步分解不确定性类型以实现全面的不确定性量化，我们提出了因果不变掩码，该方法通过测量原始预测与以因果焦点视角为条件所得预测之间的语义偏移来进行度量。基于该框架，我们引入语义散度作为不确定性量化的核心指标，并提供了理论证据表明该指标收敛于模型对非因果关联敏感性的方差，从而确立了其捕捉多模态大语言模型局限性的能力。

    arXiv:2610.02887v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) suffer from hallucinations, creating a critical need for Uncertainty Quantification (UQ) to ensure reliable deployment. However, existing approaches struggle to detect uncertainty caused by superficial associations, especially when the query-relevant signal is weak. We mainly attribute this issue to their bias toward aleatoric uncertainty arising from data ambiguity, overlooking epistemic uncertainty stemming from model limitations. To further decompose uncertainty types for a comprehensive UQ, we propose Causal-Invariant Masking (CIM), which measures the semantic shift between the original predictions and those conditioned on a causally-focused view. Based on this framework, we introduce Semantic Divergence as our core metric for UQ and provide theoretical evidence that it converges to the variance of model's sensitivity to non-causal correlations, establishing its ability to capture MLLM's lim
    
[^128]: 无触发器的错误信息：从事实性回答到下游决策

    Misinformation Without Triggers: From Factual Answers to Downstream Decisions

    [https://arxiv.org/abs/2610.02886](https://arxiv.org/abs/2610.02886)

    该研究揭示虚假训练文档无需任何触发器即可改变语言模型的事实性回答，但直接回答的受污染程度无法预测下游决策行为，二者之间存在“审计差距”，因此仅审计直接答案会严重低估错误信息的真实危害。

    

    语言模型从网络文档中学习，其中一些文档是虚假的，而虚假内容可能渗入模型对事实性问题的回答，以及使用该回答的摘要和决策。大多数数据投毒研究会在训练数据中植入触发器，并在提示中激活它。虚假文档同样可以在没有任何触发器的情况下改变事实性回答，但我们尚不清楚直接回答能否预测后续决策。在这项工作中，我们追踪虚假内容在答案之后的传播路径，发现直接探测所报告的结果与模型随后的实际行为之间存在一种“审计差距”（audit gap）。我们在一个受控决策任务“猜首都”中，将虚假训练与匹配的真实性对照进行比较——该任务中一个固定的解码器将事实性回答转化为计分的卡片选择——并在来自2019–20年澳大利亚山火相关Facebook帖子的一条误导性声明上进行验证。在八种模型、污染剂量为1,000的设置下，直接注入选择率达到95.8–100%，而注入后的游戏决策增加……（原文在此处截断）

    arXiv:2610.02886v1 Announce Type: cross  Abstract: Language models learn from web documents, some of them false, and false content can reach a model's answer to a factual question and the summaries and decisions that use it. Most data-poisoning studies add a trigger to the training data and activate it in the prompt. False documents can also change factual responses without any trigger, but we do not know whether the direct answer predicts the decision. In this work, we follow false content past the answer and find an \emph{audit gap} between what a direct probe reports and what the model then does, comparing false training with matched truthful controls in a controlled decision task, \emph{Guess the Capital}, where a fixed decoder turns factual answers into a scored card choice, and on a misleading claim from Facebook posts about the 2019--20 Australian bushfires. Across eight models at dose 1,000, direct injected-choice rates reach 95.8--100\%, while injected game choices increase by
    
[^129]: PsyEvo：一种在测试时自我演化的个性化心理咨询智能体

    PsyEvo: A Personalized Counseling Agent That Self-Evolves at Test Time

    [https://arxiv.org/abs/2610.02885](https://arxiv.org/abs/2610.02885)

    PsyEvo是一个基于大语言模型的心理咨询框架，通过分层贝叶斯技能策略和会话间列表式偏好优化等组件，在测试时实现针对个体来访者的个性化定制和响应策略的自我演化改进。

    

    心理健康障碍影响着全球相当大比例的人口，然而训练有素的专业治疗师持续短缺，导致大多数人无法获得充分的心理护理。基于大语言模型（LLM）的心理咨询师为提供可扩展的对话式心理支持提供了一个有前景的方向。但仅依靠离线模型训练，在适应个体来访者或在测试时从持续的治疗互动中学习方面空间有限。我们提出了PsyEvo，这是一个基于LLM的心理咨询框架，通过三个组件在测试时实现针对特定来访者的个性化服务以及响应策略的改进：分层贝叶斯技能策略（HBSP）通过维护根据会话反馈更新的每个来访者的技能后验分布，来个性化地选择应用何种干预措施；会话间列表式偏好优化（LiPO）通过根据跨来访者偏好更新共享的响应适配器，来改进所选技能的表达方式……

    arXiv:2610.02885v1 Announce Type: new  Abstract: Mental health disorders affect a substantial proportion of the global population, yet a persistent shortage of trained practitioners leaves the majority without adequate care. Large language model (LLM)-based counselors present a promising direction for delivering scalable conversational psychological support. Offline model training alone leaves limited room to adapt to individual clients or to learn from ongoing therapeutic interaction at test time. We introduce PsyEvo, an LLM-based counseling framework that enables both client-specific personalization and response-policy improvement at test time through three components: Hierarchical Bayesian Skill Policy (HBSP) personalizes what intervention to apply by maintaining a per-client skill posterior updated from session feedback; Inter-session Listwise Preference Optimization (LiPO) improves how the selected skill is expressed by updating a shared response adapter from cross-client preferen
    
[^130]: DyRA：面向深度神经网络高效矩阵乘法的动态残差近似方法

    DyRA: Dynamic Residual Approximation for Efficient Matrix Multiplication in DNNs

    [https://arxiv.org/abs/2610.02882](https://arxiv.org/abs/2610.02882)

    提出了输入自适应方法DyRA，通过在推理过程中动态近似并校正结构化权重近似引入的输出残差误差，直接优化输出的低秩因子，从而更高效、更精确地实现深度神经网络中的矩阵乘法近似。

    

    大规模基础模型在多种任务上取得了出色的性能，但其庞大的规模使得推理成本高昂，这主要归因于密集矩阵乘法。先前的工作通过用低秩分解等高效结构化形式替换密集权重矩阵来降低这一成本。然而，这些方法近似的是权重本身，而非决定推理精度的输出激活值。因此，权重空间中的微小误差可能被输入激活值放大，从而产生较大的输出误差。在本工作中，我们提出了DyRA，这是一种输入自适应方法，通过在推理过程中校正残差输出误差来改进结构化矩阵乘法的近似效果。我们证明，通过直接优化输出的低秩因子，可以更有效地近似矩阵乘法。DyRA基于这一洞见，动态地近似并校正由结构化权重近似所引入的输出误差。

    arXiv:2610.02882v1 Announce Type: cross  Abstract: Large-scale foundation models achieve strong performance across diverse tasks, but their size makes inference costly, largely due to dense matrix multiplications. Prior work reduces this cost by replacing dense weight matrices with efficient structured forms such as low-rank factorizations. However, these methods approximate weights rather than the output activations that determine inference accuracy. Consequently, small weight-space errors can be amplified by input activations, producing large output errors. In this work, we propose DyRA, an input-adaptive method that improves structured matrix multiplication approximation by correcting residual output errors during inference. We show that matrix multiplication can be approximated more effectively by directly optimizing low-rank factors of the output. DyRA builds on this insight by dynamically approximating and correcting the output error introduced by structured weight approximations
    
[^131]: 面向编码器跨语言性能提升的查询感知路由

    Query-aware routing for Cross-lingual performance gains in Encoders

    [https://arxiv.org/abs/2610.02875](https://arxiv.org/abs/2610.02875)

    该论文提出将仅作用于查询端的LoRA适配器与基于查询和索引语言的确定性路由相结合，在保留同语言性能和现有文档索引的同时，使英语、芬兰语、瑞典语六条跨语言检索方向的平均nDCG@10从0.241提升至0.291，相对提升20.9%。

    

    多语言编码器在查询与相关文档语言不同时，检索效果可能会下降，尽管其在同语言场景下表现强劲。我们研究了如何在保留编码器现有同语言性能和文档索引的前提下，提升芬兰语与瑞典语的跨语言检索效果。我们将仅作用于查询端、针对冻结文档嵌入训练的低秩适配器（LoRA），与基于查询语言和索引语言的确定性路由相结合：跨语言查询使用适配器，同语言查询则使用原始编码器。SampoTron（我们微调的低秩适配器与Nemotron-3-Embed-1B模型的组合）在一个采样的金融基准上，将六个英语、芬兰语和瑞典语方向的平均检索质量从nDCG@10的0.241提升至0.291，相对提升20.9%。所有六个跨语言方向均得到改善，并且路由保留了（原有同语言性能）。

    arXiv:2610.02875v1 Announce Type: cross  Abstract: Multilingual encoders can exhibit reduced retrieval effectiveness when queries and relevant documents differ in language, despite strong same-language performance. We investigate whether Finnish and Swedish cross-lingual retrieval can improve while preserving an encoder's existing same-language performance and document index. We combine a query-only low-rank adapter, trained against frozen document embeddings, with deterministic routing based on query and index languages. Cross-language queries use the adapter, while same-language queries use the original encoder. SampoTron, our fine-tuned low-rank (LoRA) adapter alongwith the Nemotron-3-Embed-1B model, improves average retrieval quality across six English, Finnish, and Swedish directions from 0.241 to 0.291 in normalized discounted cumulative gain (nDCG) at rank ten, a 20.9% relative gain on a sampled financial benchmark. All six cross-lingual directions improve, and routing preserves
    
[^132]: ConvoDrift：用于建模风格语调演变的多轮对话数据集

    ConvoDrift: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution

    [https://arxiv.org/abs/2610.02873](https://arxiv.org/abs/2610.02873)

    ConvoDrift 是一个用于建模固定语义意图下多轮对话风格语调渐进漂移的数据集，包含 15,727 个多轮对话结构、风格漂移标注及基于五种人设条件的偏好成对数据集，可支持风格适应与个性化对齐的受控研究。

    

    对话中语言风格的演变是自然语言处理（NLP）中一个尚未充分探索的问题。现有的风格控制数据集大多聚焦于句子层面，或假设风格在整个对话过程中保持静态，忽略了交互过程中因用户偏好变化而产生的动态转变。我们提出了 ConvoDrift，一个旨在建模固定语义意图下渐进式对话风格语调漂移的数据集。该数据集基于 15,727 个共享的多轮对话结构构建，可用于风格适应以及基于人设条件的对齐方法研究。每段对话包含六组提示-回复对，每对均带有风格漂移和风格方向标签的标注，且涵盖了多种交流体裁。我们进一步衍生出一个互补的成对数据集，通过将语义等价但风格不同的回复进行配对，并利用五种不同的风格化交流人设标注基于人设条件的偏好，从而支持对个性化风格的受控研究。

    arXiv:2610.02873v1 Announce Type: cross  Abstract: The evolution of linguistic style in conversations is an underexplored issue in NLP. Most style-control datasets focus on sentences or assume a static style throughout, missing the dynamic shifts that occur as user preferences change during interactions. We introduce ConvoDrift, a dataset designed to model progressive stylistic conversational tone drift under fixed semantic intent. It is built on 15,727 shared multi-turn conversational structures for adaptation and persona-conditioned alignment methods. It consists of six prompt-response pairs per conversation, each with the annotation of style drift and style direction labels. These pairs cover a range of communication genres. We further derive a complementary pairwise dataset by pairing semantically equivalent but stylistically distinct responses and annotating persona-conditioned preferences using five distinct style communication personas, enabling the controlled study of personali
    
[^133]: AgentTrap：针对自主渗透测试代理的状态化反馈欺骗

    AgentTrap: Stateful Feedback Deception against Autonomous Penetration Testing Agents

    [https://arxiv.org/abs/2610.02869](https://arxiv.org/abs/2610.02869)

    提出了首个专为自主渗透测试代理设计的闭环蜜罐 AgentTrap，通过状态化欺骗与行为引导升级来诱捕、拖延代理并收集其行为证据。

    

    自主渗透测试代理通过根据目标响应持续调整其计划与行动来执行多步骤攻击。作为一种常见的防御手段，蜜罐可以通过呈现诱饵服务将这些代理从真实资产上引开，同时还能支持攻击追踪和主动反击。然而，传统蜜罐主要依赖静态工件和预定义响应，因此无法适应自主渗透测试代理不断演变的攻击策略。为此，我们提出了 AgentTrap，这是首个专为自主渗透测试代理设计的闭环蜜罐。AgentTrap 利用哨兵端点避免对良性流量的干扰，采用基于受保护应用构建的状态化欺骗，并通过行为引导的升级机制，以受控的信息披露维持与代理的交互并收集代理侧的行为证据。我们针对八个自主渗透测试代理对 AgentTrap 进行了评估。

    arXiv:2610.02869v1 Announce Type: cross  Abstract: Autonomous penetration testing agents conduct multi-step attacks by continuously adapting their plans and actions to target responses. As a common defense, honeypots can be deployed to divert these agents from real assets by presenting decoy services, while also supporting attack tracing and active counterattacks. However, conventional honeypots rely primarily on static artifacts and predefined responses, leaving them unable to adapt to the evolving attack strategies of autonomous penetration testing agents. To this end, we present AgentTrap, the first closed-loop honeypot tailored for autonomous penetration testing agents. AgentTrap uses sentinel endpoints to avoid benign interference, stateful deception grounded in the protected application, and behavior-guided escalation to sustain engagement and collect agent-side behavioral evidence with controlled disclosures.   We evaluate AgentTrap against eight autonomous penetration-testing a
    
[^134]: 子群体偏移与离群点污染下的分布鲁棒生存模型

    Distributionally Robust Survival Models under Subpopulation Shift and Outlier Contamination

    [https://arxiv.org/abs/2610.02868](https://arxiv.org/abs/2610.02868)

    本文提出一个分布鲁棒生存分析框架，通过外层最小化削弱离群样本的影响、内层最大化聚焦最不利的子群体，从而联合应对子群体偏移与离群点污染，并直接兼容Cox风险集等不可分解的生存损失。

    

    在分布偏移下学习鲁棒的生存模型是许多应用中一个重要但具有挑战性的问题。在异质性人群中，平均表现良好的模型在某些子群体上可能仍然表现不佳，而当训练数据被离群点污染时，这一问题会变得更加严重。在本文中，我们提出了一个新颖的生存分析分布鲁棒框架，能够同时应对潜在的子群体偏移和离群点污染。所提出的方法结合了外层最小化与内层最大化：外层最小化通过降低被污染样本的影响来选择一个精炼的名义分布，内层最大化则聚焦于最具挑战性的子群体。这一建模方式直接适用于不可分解的生存损失，同时保留了样本间的交互作用，包括Cox负偏对数似然中的风险集结构。我们开发了一种交替梯度（算法……）（原文在此截断）。

    arXiv:2610.02868v1 Announce Type: cross  Abstract: Learning robust survival models under distribution shift is an important but challenging problem in many applications. In heterogeneous populations, a model that performs well on average may still perform poorly on certain subpopulations, and this issue becomes even more severe when the training data are contaminated by outliers. In this paper, we propose a novel distributionally robust framework for survival analysis that jointly addresses latent subpopulation shift and outlier contamination. The proposed method combines an outer minimization that selects a refined nominal distribution by reducing the influence of contaminated samples and an inner maximization that focuses on the most challenging subpopulation. This formulation directly accommodates non-decomposable survival losses while preserving interactions across samples, including the risk-set structure of the Cox negative partial log-likelihood. We develop an alternating gradie
    
[^135]: TACD：通过终端放大控制蒸馏高效的文本到运动生成模型

    TACD: Distilling Efficient Text-to-Motion Models via Terminal Amplification Control

    [https://arxiv.org/abs/2610.02867](https://arxiv.org/abs/2610.02867)

    TACD提出一种在策略蒸馏方法，通过将教师模型查询与学生步长绑定来限制去噪终点附近的误差权重，在无需真实运动数据的情况下训练出高效的少步文本到运动生成模型。

    

    最近的文本到运动模型在运动质量和指令遵循能力上有所提升，但多步去噪和庞大的模型组件使得部署缓慢且内存占用高。我们提出了终端放大控制蒸馏（TACD），这是一种在策略训练方法，仅利用文本提示和预训练教师模型即可训练高效的运动生成器，无需真实运动训练数据。基于分段在策略流蒸馏，我们沿学生模型生成的轨迹对干净运动预测进行监督。我们发现了一种失败模式：在固定监督网格上进行速度匹配会反复过度加权去噪终点附近的误差，从而损害少步生成质量。TACD将最新的教师模型查询与学生模型的步长绑定，在不改变推理过程的情况下限制了干净运动空间中的有效损失权重。在HumanML3D和KIT-ML数据集上的实验表明，少步生成性能得到提升，包括八步生成（指标）降低了58%。

    arXiv:2610.02867v1 Announce Type: new  Abstract: Recent text-to-motion models have improved motion quality and instruction following, yet many-step denoising and large model components make deployment slow and memory-intensive. We present Terminal-Amplification-Controlled Distillation (TACD), an on-policy approach for training efficient motion generators from text prompts and pretrained teachers, without real-motion training data. Building on segmented on-policy flow distillation, we supervise clean-motion predictions along student-generated trajectories. We identify a failure mode in which velocity matching on a fixed supervision grid repeatedly overweights errors near the denoising endpoint, degrading few-step generation. TACD ties the latest teacher query to the student's step size, bounding the effective loss weights in clean-motion space without changing inference. Experiments on HumanML3D and KIT-ML demonstrate improved few-step generation, including a 58% reduction in eight-step
    
[^136]: 面向小型语言模型智能体的框架感知蒸馏

    Harness-Aware Distillation for Small Language Model Agents

    [https://arxiv.org/abs/2610.02858](https://arxiv.org/abs/2610.02858)

    提出框架感知蒸馏（HAD），将蒸馏聚焦于教师模型在框架之外附加的能力，通过动作偏好对比与有效性检查，使小型学生智能体学会根据框架信息正确行动。

    

    语言模型智能体在部署时都附带一个框架，即围绕模型的软件，负责管理其上下文、工具和反馈。当这样的智能体被蒸馏为更小的智能体时，框架保持不变，因此学生模型主要需要的是框架无法提供的、教师模型特有的能力，例如根据框架信息正确采取行动。然而，标准蒸馏会模仿教师模型的完整输出，并将框架视为输入的一部分。我们提出了框架感知蒸馏，它将蒸馏的重点放在教师模型在框架之外所增加的内容上。HAD 通过两个组件来补充在线策略蒸馏：一是动作偏好机制，对比同一教师模型在有和没有框架信息情况下的动作，并在学生模型自身推理之后进行评分；二是有效性检查，丢弃偏好动作与框架记录相矛盾的偏好对。我们证明，这种对比为学生模型提供了……（原文摘要在此处截断）

    arXiv:2610.02858v1 Announce Type: new  Abstract: Language model agents are deployed with a harness, the software around the model that manages its context, tools, and feedback. When such an agent is distilled into a smaller one, the harness stays in place, so the student mainly needs the teacher-specific abilities that the harness cannot provide, such as acting correctly on harness information. Standard distillation, however, imitates the teacher's full outputs and treats the harness as part of the input. We propose Harness-Aware Distillation (HAD), which focuses distillation on what the teacher adds beyond the harness. HAD complements on-policy distillation with two components: an action preference that contrasts the same teacher's actions with and without the harness information, scored after the student's own reasoning, and a validity check that drops preference pairs whose preferred action contradicts the harness records. We show that the contrast gives the student information that
    
[^137]: 基于收缩约束状态空间模型的有界可达性与越狱检测

    Bounded Reachability & Jailbreak Detection via Contraction-Constrained State Space Models

    [https://arxiv.org/abs/2610.02853](https://arxiv.org/abs/2610.02853)

    本文证明基于SSM的安全头能否获得可认证的越狱检测鲁棒性完全取决于一个收缩条件——状态转移矩阵的 $l_\infty$ 范数小于1：条件成立时可达输出区间稳态宽度有界、可通过精确区间界限传播认证鲁棒分类，条件不满足时区间随序列长度指数增长导致认证不可能。

    

    安全头是附加在预训练语言模型上的轻量级分类器，用于在生成之前标记有害输入。其经验检测性能已被广泛研究，但其形式化的鲁棒性属性在很大程度上仍未被探索。我们探究：在嵌入空间的有界扰动下，何时可以证明基于状态空间模型（SSM）的安全头对所有输入都能产生相同的预测。我们证明答案取决于一个单一条件：状态转移矩阵的 $l_\infty$ 范数必须满足 $\norm{A}_\infty<1$（即“收缩条件”），该条件使得对线性时不变分类器进行精确的区间界限传播（IBP）认证成为可能。当收缩条件成立时，可达输出区间具有有界的稳态宽度，样本可以被认证为鲁棒分类；当条件不满足时，该区间随序列长度呈指数增长，认证在此情况下不可行……

    arXiv:2610.02853v1 Announce Type: new  Abstract: Safety heads are lightweight classifiers attached to pretrained language models for flagging harmful inputs before generation. Their empirical detection performance has been studied, but their formal robustness properties remain largely unexplored. We ask when a State Space Model (SSM)-based safety head can be certified to produce the same prediction for all inputs within a bounded embedding-space perturbation. We prove that the answer turns on a single condition: the $l_\infty$ norm of the state transition matrix must satisfy $\norm{A}_\infty<1$ (the \emph{contraction condition}), which enables exact interval bound propagation (IBP) certification for linear time-invariant classifiers. When the contraction condition holds, the reachable output interval has bounded steady-state width and examples can be certified as robustly classified. When it fails, the interval grows exponentially with sequence length and certification is impossible at
    
[^138]: DNAlign：面向大语言模型的动态零空间安全对齐

    DNAlign: Dynamic Null-Space Safe Alignment for LLMs

    [https://arxiv.org/abs/2610.02844](https://arxiv.org/abs/2610.02844)

    DNAlign将大语言模型视为动态系统，结合控制论优化与零空间投影，把安全扰动限制在与危害相关的子空间内，从而在不损害模型通用知识和响应质量的前提下实现轻量级安全对齐。

    

    确保大语言模型（LLM）的安全可靠部署仍然是一项根本性挑战。现有的安全对齐方法要么计算成本高昂，要么会在无意中破坏模型的核心知识，导致模型在良性任务上的流畅性和事实准确性下降，这揭示了安全性与实用性之间持续存在的权衡。我们提出DNAlign，这是一个将控制论优化与零空间投影相结合的轻量级对齐框架。通过将LLM视为动态系统，该框架引入可控扰动以引导模型的生成趋向安全行为。其关键组件是投影模块，该模块将这些扰动限制在由中性隐状态导出的与危害相关的子空间内，从而保留模型的通用知识和响应质量。此外，基于人类偏好数据训练的价值函数会自适应地优化控制信号，以使模型行为与人类偏好保持一致（摘要原文在此处截断）。

    arXiv:2610.02844v1 Announce Type: new  Abstract: Ensuring the safe and reliable deployment of large language models (LLMs) remains a fundamental challenge. Existing safety alignment approaches either incur high computational cost or unintentionally disrupt the model's core knowledge, leading to degraded fluency and factual accuracy on benign tasks. This reveals a persistent trade-off between safety and utility. We propose DNAlign, a lightweight alignment framework that integrates control-theoretic optimization with null-space projection. By treating the LLM as a dynamic system, the proposed framework introduces controllable perturbations to steer generation toward safe behavior. A key component is the projection module, which restricts these perturbations to the harmful-related subspace derived from neutral hidden states, thereby preserving general knowledge and response quality. A value function trained on human preference data adaptively optimizes the control signals to align with hu
    
[^139]: FastOPD：面向轻量化VLA部署的在策略蒸馏

    FastOPD: On-Policy Distillation for Lightweight VLA Deployment

    [https://arxiv.org/abs/2610.02832](https://arxiv.org/abs/2610.02832)

    FastOPD提出了一种高效的在策略蒸馏框架，将流图单状态教师监督与自洽性目标相结合，把大规模VLA基础模型压缩为可实际部署的轻量化模型，并从理论上保证学生模型能恢复出与理想少步教师模型相当的分布。

    

    视觉-语言-动作（VLA）基础模型已迅速扩展规模以提升操作性能与泛化能力，但这种规模化带来了高昂的计算成本，使其在现实世界中的部署日益困难。现有方法通常通过设计更小的架构或减少基于流的策略中的迭代去噪步骤来缓解这一问题。在本工作中，我们提出FastOPD，一个从基础模型到轻量化模型的VLA框架，通过高效的在策略蒸馏实现大规模VLA的实际部署。具体而言，FastOPD采用流图来实现单状态教师监督，并将其与自洽性目标相结合，构建出一个能够学习教师动力学的紧凑学生模型。此外，我们从理论上证明，最小化该目标可使蒸馏得到的学生模型恢复出与理想少步教师模型所诱导分布相当的水平。

    arXiv:2610.02832v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) foundation models have scaled rapidly to enhance manipulation performance and generalizability, but this scaling incurs high computational costs that render real-world deployment increasingly challenging. Existing approaches typically mitigate this issue by designing smaller architectures or reducing the iterative denoising steps in flow-based policies. In this work, we propose FastOPD, a foundation-to-lightweight VLA framework that enables the practical deployment of large-scale VLAs through efficient on-policy distillation. Specifically, FastOPD adapts a flow map for single-state teacher supervision and combines it with a self-consistency objective to construct a compact student that learns the teacher dynamics. Furthermore, we theoretically demonstrate that minimizing this objective allows the distilled student to recover a distribution on par with that induced by an ideal few-step teacher model. We eval
    
[^140]: AMBER：面向列表式视觉语言重排的多视图自适应预算分配

    AMBER: Multi-View Adaptive Budget Allocation for Listwise Vision-Language Reranking

    [https://arxiv.org/abs/2610.02831](https://arxiv.org/abs/2610.02831)

    AMBER提出了一种在线预算化多视图重排框架，将碎片化的VLM输出通过Elo更新整合为全局排序状态，并在视图构建和查询调度两个层级上动态分配昂贵的VLM计算资源，以最大化期望信息增益并提升多模态检索重排效率。

    

    视觉语言模型是多模态检索中强大的列表式重排器，但其高昂的推理成本限制了它们只能评估较小的局部候选视图。现有的多次调用策略依赖于固定调度，将昂贵的VLM调用浪费在无信息量的候选对和简单查询上。为解决这一问题，我们提出了自适应多视图预算Elo重排框架（AMBER），这是一个在线的、有预算约束的多视图重排框架，能够动态优化全局资源分配。AMBER将碎片化的列表式VLM输出视为局部锦标赛，通过连续的Elo更新来维护一个轻量级的全局排序状态。在此基础上，它在两个层级上分配计算资源：动态构建具有高分值模糊性的候选视图，以及调度查询以最大化期望信息增益。我们证明每次Elo更新都对应于Bradley-Terry对数似然上的一步随机梯度上升，并提供……（摘要在此处不完整）

    arXiv:2610.02831v1 Announce Type: new  Abstract: Vision-language models (VLMs) are powerful listwise rerankers for multimodal retrieval, but high inference costs restrict them to evaluating small local candidate views. Existing multi-call strategies rely on fixed schedules, wasting expensive VLM calls on uninformative candidate pairs and easy queries. To address this, we propose Adaptive Multi-view Budgeted Elo Reranking (AMBER), an online, budgeted multi-view reranking framework that dynamically optimizes global resource allocation. AMBER treats fragmented listwise VLM outputs as local tournaments, using continuous Elo updates to maintain a lightweight global ranking state. Building on this, it allocates computation at two levels: dynamically constructing candidate views with high score ambiguity, and scheduling queries to maximize expected information gain. We show that each Elo update corresponds to a stochastic gradient ascent step on the Bradley-Terry log-likelihood, and provide a
    
[^141]: FSPO：面向预算约束的大语言模型强化学习后训练的策略一致风险与帕累托可行控制

    FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training

    [https://arxiv.org/abs/2610.02828](https://arxiv.org/abs/2610.02828)

    FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。

    

    自适应的大语言模型强化学习后训练会在训练过程中在线调整多个训练执行器，包括 rollout 温度、组大小、裁剪、KL 正则化、验证器分配以及更新预算。目前有三个相互耦合的问题尚未解决：从行为轨迹训练得到的未来风险模型未必能准确估计将要部署的控制器所引发的风险；基于已记录的状态-动作对校准的分数在经过选择性动作选择后可能出现校准失准；独立的单资源最小成本通常无法保证多资源延续的可行性。我们提出 FSPO，一种面向预算约束的大语言模型强化学习后训练的反馈状态控制器，能够联合解决这些问题。FSPO 学习一个策略一致的风险前瞻（risk-to-go）模型，其 Bellman 目标遵循与未来决策所用的同一个冻结控制器，并同时学习一个长程效用模型。决策条件化轨迹校准（DCTC）对风险进行校准……

    arXiv:2610.02828v1 Announce Type: new  Abstract: Adaptive LLM reinforcement-learning post-training changes multiple training actuators online, including rollout temperature, group size, clipping, KL regularization, verifier allocation, and update budget. Three coupled issues remain unresolved. A future-risk model trained from behavior trajectories need not estimate the risk induced by the controller that will be deployed; a score calibrated on logged state-action pairs can become miscalibrated after selective action choice; and independent per-resource minimum costs do not in general certify a feasible multi-resource continuation. We introduce FSPO, a feedback-state controller for budgeted LLM RL post-training that addresses these issues jointly. FSPO learns a policy-consistent risk-to-go model whose Bellman target follows the same frozen controller used for future decisions, together with a long-horizon utility model. Decision-conditioned trajectory calibration (DCTC) calibrates risk 
    
[^142]: MLCommons 越狱基准测试 v1.0

    MLCommons Jailbreak Benchmark v1.0

    [https://arxiv.org/abs/2610.02827](https://arxiv.org/abs/2610.02827)

    MLCommons发布了越狱基准测试v1.0，提供了一套端到端的评估方法论，使用264个种子提示词、十一个危害类别和越狱分类法中的代表性攻击，对八个开放权重大语言模型进行单轮文本越狱攻击的鲁棒性评估，并创新性地提出以“韧性差距”作为核心安全衡量指标。

    

    现代AI系统被设计为拒绝危险请求。“越狱”是一种精心构造的提示词，旨在绕过这些安全防护措施，诱使系统输出其通常会拒绝提供的内容。MLCommons越狱基准测试v1.0提供了一种端到端的方法论，用于评估大型语言模型对单轮、基于文本的越狱攻击的鲁棒性。它将基于标准的系统与攻击选择、配对的基线与对抗性评估、人工标注、自动化评估器校准、评分、分级以及风险校准的信息披露整合在同一个基准测试流水线中。该基准测试使用涵盖十一个危害类别的264个种子提示词以及从MLCommons越狱分类法中选取的代表性攻击，对八个开放权重系统进行了评估。响应采用AILuminate评估标准v1.4进行评估，鲁棒性通过“韧性差距”来衡量，即基线条件与对抗性条件之间安全性能的变化。

    arXiv:2610.02827v1 Announce Type: new  Abstract: Modern AI systems are designed to refuse hazardous requests. A jailbreak is a prompt crafted to bypass those safeguards and elicit outputs that the system would normally refuse to provide. The MLCommons Jailbreak Benchmark v1.0 provides an end-to-end methodology for evaluating the robustness of large language models to single-turn, text-based jailbreak attacks. It combines criteria-driven system and attack selection, paired baseline and adversarial evaluation, human annotation, automated evaluator calibration, scoring, grading, and risk-calibrated disclosure within a single benchmarking pipeline. The benchmark evaluates eight open-weight systems using 264 seed prompts spanning eleven hazard categories and representative attacks drawn from the MLCommons Jailbreak Taxonomy. Responses are assessed using the AILuminate Assessment Standard v1.4, and robustness is measured through the Resilience Gap: the change in safety performance between ba
    
[^143]: 通过递归自我重写扩展复杂任务的轨迹

    Scaling Trajectories for Complex Tasks through Recursive Self-Rewrite

    [https://arxiv.org/abs/2610.02826](https://arxiv.org/abs/2610.02826)

    提出递归自我重写（RSR）框架，利用单一基础模型在不同专用测试框架下发现成功解法，并将其重构为通用框架下的训练轨迹，成功将2001条轨迹扩展为11094条用于监督微调，显著提升模型解决复杂终端任务的能力。

    

    困难任务上的成功轨迹为模型改进提供了宝贵的监督信号，但专用测试框架引入的干预措施在部署时可能不可用。我们提出递归自我重写（RSR）框架，该框架使用单一基础模型（Qwen-3.8-27B）在多样化测试框架下发现成功解决方案，并在通用测试框架下将其重构为训练轨迹。其中，规划器将执行流程提取为操作手册，评论者筛查验证器和答案泄露并指导递归修订，执行器则在全新沙箱中按照合格的操作手册执行任务。在大约3000个自整理的终端任务中，三个测试框架共同解决了759个任务，比记录池中最强的单个框架多出34.3%。RSR将2001条成功源轨迹扩展为11094条重写轨迹用于监督微调。基于这些轨迹的训练表现优于基础模型和直接训练方法（原文在此截断）。

    arXiv:2610.02826v1 Announce Type: new  Abstract: Successful trajectories on difficult tasks provide valuable supervision for model improvement, but specialized harnesses introduce interventions that may be unavailable during deployment. We propose Recursive Self-Rewrite (RSR), a framework that uses one base model, Qwen-3.8-27B, to discover successful solutions under diverse harnesses and reconstruct them as training trajectories under a general harness. A planner extracts procedures into runbooks, a critic screens for verifier and solution leakage and guides recursive revision, and an executor follows qualified runbooks in fresh sandboxes. Across approximately 3K self-curated terminal tasks, three harnesses jointly solve 759 tasks, 34.3% more than the strongest individual harness in the recorded pool. RSR expands 2,001 successful source trajectories into 11,094 rewritten trajectories for supervised finetuning. Training on these trajectories outperforms both the base model and direct tr
    
[^144]: MetaRubric：面向基于量规的强化学习的奖励学习

    MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.02824](https://arxiv.org/abs/2610.02824)

    MetaRubric通过构建反事实提示、要求响应提供充分证据才能得分，并将证据感知的策略优化与响应引导的量规修订交替进行，从而解决了基于量规的强化学习中评判器给出“空洞信用”的问题。

    

    基于量规的强化学习通过为响应的各项具体要求分配部分得分，将奖励驱动的优化扩展到开放式任务。然而，量规评判器即使在响应中缺少所需信息或动作的情况下，也可能给出较高的标准得分，我们将这种失败模式称为“空洞信用”。这类得分在所需信息被删除后依然存在，甚至可能使响应的GRPO优势符号发生反转。为了解决这一问题，我们提出了MetaRubric，该方法将证据感知的策略优化与响应引导的量规自适应交替进行。我们通过在每个提示中改变一个与任务相关的事实来构建反事实对照提示。在策略优化过程中，只有当响应包含足以满足所需量规标准的证据时才会给予得分。在每个策略优化阶段之后，当前策略生成的响应会引导对原始标准和反事实标准的修订……

    arXiv:2610.02824v1 Announce Type: new  Abstract: Rubric-based reinforcement learning extends reward-driven optimization to open-ended tasks by assigning partial credit to individual response requirements. However, rubric judges can assign a high criterion score even when the information or action it requires is absent from the response, a failure mode we term Vacuous Credit. Such awards persist after the required information is removed and can reverse the sign of a response's GRPO advantage. To address this problem, we introduce MetaRubric, which alternates evidence-aware policy optimization with response-guided rubric adaptation. We construct counterfactual counterparts by changing one task-relevant fact in each prompt. During policy optimization, credit is assigned only when the response contains sufficient evidence to satisfy the required rubric criterion. After each policy-optimization stage, current policy responses guide revisions to original and counterfactual criteria while pre
    
[^145]: 面向时序域泛化的自适应谱-Koopman动力学建模

    Adaptive Spectral-Koopman Dynamics Modeling for Temporal Domain Generalization

    [https://arxiv.org/abs/2610.02822](https://arxiv.org/abs/2610.02822)

    提出AdaSpecK框架，通过谱正则化Koopman动力学建模提取去噪的低频轨迹，并结合上下文感知的异构模式提取机制，有效解决时序域泛化中的噪声过拟合与非平稳历史环境建模问题。

    

    时序域泛化旨在应对随时间发生分布变化的现实世界流式数据。然而，现有方法要么容易在数据空间中过拟合领域特定的噪声，要么在参数空间中变得过于复杂且可解释性较差。为了弥合这些差距，我们提出了AdaSpecK，一个用于时序域泛化的、具有自适应上下文提取能力的谱-Koopman框架。为了缓解对不规则采样领域的噪声拟合问题，我们引入了谱正则化的Koopman动力学建模，该方法在潜在空间中应用谱感知滤波以提取去噪后的低频轨迹，并学习一个Koopman算子在线性化空间中建模系统动力学。为了在非平稳条件下建模复杂的历史环境，我们设计了一种上下文感知的异构模式提取机制。具体而言，我们采用目标条件注意力模块来关注不同的历史模式……

    arXiv:2610.02822v1 Announce Type: cross  Abstract: Temporal Domain Generalization (TDG) has emerged to address real-world streaming data with distribution shifts over time. However, existing methods are either prone to overfitting to domain-specific noise in the data space or become overly complex and less interpretable in the parameter space. To bridge these gaps, we propose \textbf{AdaSpecK}, a spectral-Koopman framework with adaptive context extraction for TDG. To mitigate noise fitting to irregularly sampled domains, we introduce spectral-regularized Koopman dynamics modeling, which applies spectral-aware filtering in the latent space to extract denoised low-frequency trajectories and learn a Koopman operator to model the system dynamics in a linearized space. To model complex historical environments under non-stationarity, we design a context-informed heterogeneous pattern extraction mechanism. Specifically, we employ a target-conditioned attention module to attend to distinct pas
    
[^146]: iS-KV：基于块增量SVD的在线低秩KV缓存压缩

    iS-KV: Online Low-Rank KV Cache Compression via Block-Incremental SVD

    [https://arxiv.org/abs/2610.02815](https://arxiv.org/abs/2610.02815)

    提出iS-KV，一种基于块增量SVD的在线低秩KV缓存压缩方法，通过解决基更新导致的历史漂移问题，在保留全部历史状态的同时实现长思维链推理场景下的缓存高效压缩。

    

    长思维链推理在自回归解码过程中会大幅增加KV缓存的内存占用，因为每生成一个token都会引入新的键和值状态，导致缓存随解码长度线性增长。现有的KV缓存压缩方法通常通过token驱逐来控制这种增长，但不可逆的删除可能会移除后续推理需要重新访问的历史状态。基于SVD的低秩压缩通过以更紧凑的表示保留所有位置，提供了一种替代方案。然而，将其从固定的提示缓存扩展到在线解码并非易事。通过我们的研究，我们发现如果为新token更新基向量，同时旧token保持其在旧基中的坐标，存储的历史状态会发生显著漂移。基于这一观察，我们提出了iS-KV，一种面向长时程推理的在线低秩KV缓存压缩方法。iS-KV在保持最近窗口精确的同时……

    arXiv:2610.02815v1 Announce Type: new  Abstract: Long chain-of-thought reasoning substantially increases KV-cache memory during autoregressive decoding, as every generated token introduces new key and value states and causes the cache to grow linearly with decoding length. Existing KV-cache compression methods typically control this growth through token eviction, but irreversible deletion can remove historical states that later reasoning may need to revisit. SVD-based low-rank compression provides an alternative by retaining all positions with a more compact representation. However, extending it from a fixed prompt cache to online decoding is non-trivial. Through our investigation, we find that if the basis is updated for new tokens while old tokens keep their coordinates in the old basis, the stored history drifts substantially. Based on this observation, we propose iS-KV, an online low-rank KV-cache compression method for long-horizon reasoning. iS-KV keeps a recent window exact whil
    
[^147]: ROUTEAUDIT：面向预算受限多验证器路由的交互感知识别方法

    ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

    [https://arxiv.org/abs/2610.02808](https://arxiv.org/abs/2610.02808)

    ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。

    

    自适应多验证器系统通常通过端点的质量-成本差距进行比较，即使验证器目录、可用性、资源核算、信息过滤或评分器会随策略发生变化。我们将验证器路由形式化为一个契约条件化的识别问题。该契约记录了请求支持、验证器目录、实际可用性、资源核算、在线过滤以及轨迹后评分；一个匹配的路由对比仅改变策略坐标。ROUTEAUDIT为该契约增加了三个可度量的对象：契约格在所有可容许的桥接顺序上对坐标增量取平均，并报告由此得到的归因及其路径敏感性；策略无关的响应带在自适应策略揭示不同观测时识别成对的顺序对比；对于不完整的匹配，请求级边界利用仍然可观测的潜在结果，给出紧致的有限（摘要在此处截断）。

    arXiv:2610.02808v1 Announce Type: new  Abstract: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp fi
    
[^148]: VIGOR：基于模型的强化学习中通过潜空间一致性实现零样本视觉泛化

    VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.02801](https://arxiv.org/abs/2610.02801)

    VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。

    

    基于模型的强化学习（MBRL）通过在学习的潜在动力学中进行规划，实现了强大的样本效率，但在面对背景变化、光照变化或相机移动等未见过的视觉干扰时，其性能会大幅下降。与无模型强化学习（编码器扰动仅影响单步预测）不同，MBRL存在两级脆弱性：视觉干扰首先使编码器输出偏离分布，随后这些误差会在规划时域内通过递归潜在轨迹推演不断累积放大。我们提出VIGOR，这是一个能够在保留其MBRL骨干样本效率的同时，实现对未见视觉干扰进行零样本泛化的框架。VIGOR集成了三个相互关联的组件：（i）非对称弱到强增强，在单个批次内配对仅弱增强与弱到强增强的潜在视图……

    arXiv:2610.02801v1 Announce Type: new  Abstract: Model-based reinforcement learning (MBRL) achieves strong sample efficiency by planning within learned latent dynamics, yet its performance degrades substantially under unseen visual distractions such as background variations, lighting changes, or camera shifts. Unlike model-free RL, where encoder perturbations affect only single-step predictions, MBRL suffers from a two-level vulnerability: visual distractions first push encoder outputs out of distribution, and these errors then compound through recursive latent rollouts over the planning horizon. We propose visual generalization via latent-space consistency in model-based RL (VIGOR), a framework that enables zero-shot generalization to unseen visual distractions while retaining the sample efficiency of its MBRL backbone. VIGOR integrates three interdependent components: (i) asymmetric weak-to-strong augmentation, which pairs weak-only and weak-to-strong latent views within a single bat
    
[^149]: BitNest：面向内存高效大语言模型推理加速的比特嵌套投机解码

    BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration

    [https://arxiv.org/abs/2610.02800](https://arxiv.org/abs/2610.02800)

    BitNest提出了一种比特嵌套的投机解码框架，将低精度草稿模型直接嵌入高精度目标模型的权重表示中，通过残差细化使两者共享单一物理权重，从而在加速大语言模型推理的同时显著降低内存开销。

    

    投机解码通过使用轻量级草稿模型提出多个token进行并行验证，从而加速自回归生成。然而，现有方法通常需要一个额外的草稿模型或权重表示，在资源受限的设备上引入了不可忽视的内存开销。自投机方法虽然减少了这种开销，但仍面临草稿质量、目标质量和存储效率之间的权衡。我们提出了BitNest，这是一种比特嵌套的投机解码框架，它将低精度草稿直接嵌入到更高精度的目标表示中。BitNest并非从预定义的目标模型派生草稿，而是首先构建一个强大的低精度基础模型，然后通过残差细化恢复更高精度的目标模型，使两个模型能够共享单一的物理权重表示。BitNest进一步将这种渐进精度设计扩展到KV缓存，用于长上下文推理（摘要在此处截断）。

    arXiv:2610.02800v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive generation by using a lightweight draft to propose multiple tokens for parallel verification. However, existing methods often require an additional draft model or weight representation, introducing non-negligible memory overhead on resource-constrained devices. Self-speculative approaches reduce this overhead, yet still face trade-offs between draft quality, target quality, and storage efficiency. We propose BitNest, a bit-nested speculative decoding framework that embeds a low-precision draft directly into the higher-precision target representation. Instead of deriving a draft from a predefined target, BitNest first constructs a strong low-precision base and then recovers the higher-precision target through residual refinement, enabling both models to share a single physical weight representation. BitNest further extends this progressive-precision design to the KV cache for long-context in
    
[^150]: 基于EEG-fNIRS的跨被试连续情感回归中的共享结构与个体结构建模

    Modeling Shared and Individual Structure for Cross-Subject Continuous Affect Regression from EEG-fNIRS

    [https://arxiv.org/abs/2610.02796](https://arxiv.org/abs/2610.02796)

    该论文提出将情感轨迹分解为观看相同刺激的被试间共享的结构，以及基于无标注EEG标记（α波段跨通道同步性）估计的个体校准结构，从而在EEG-fNIRS数据上实现了零样本跨被试的连续效价-唤醒度回归。

    

    从生理信号中进行逐秒连续的效价-唤醒度估计，通常在被试依赖的设置下进行研究，即模型可以使用其后续评估对象的标注数据进行训练。本文在一个同步EEG-fNIRS数据集上研究了更具挑战性的零样本跨被试变体：为模型从未见过其标注数据的被试预测原始量表（[1, 255]）上的效价与唤醒度轨迹，且仅利用这些被试在与训练被试（互不相交的一组人）观看相同视频刺激时的无标注EEG/fNIRS记录。该方法将情感轨迹分解为两部分：观看相同刺激的被试之间共享的结构，以及为每个测试被试从无标注EEG标记（α波段跨通道同步性）估计得到的个体结构，该个体结构围绕量表中点对共享轨迹进行重新缩放。研究者在四个独立的维度上验证了这种逐被试校准机制：留一被试相关性……（摘要在此处截断）

    arXiv:2610.02796v1 Announce Type: new  Abstract: Continuous, second-by-second valence-arousal estimation from physiological signals is typically studied in a subject-dependent setting, where the model sees labeled data from the same person it is later evaluated on. We study the harder zero-shot cross-subject variant on a synchronized EEG-fNIRS dataset: predict raw-scale ([1, 255]) valence and arousal trajectories for subjects whose labels the model never observes, given only their unlabeled EEG/fNIRS recordings while watching the same video stimuli as a disjoint set of training subjects. We decompose the affect trajectory into a structure shared across subjects who watch the same stimuli and an individual structure estimated for each test subject from a label-free EEG marker (alpha-band cross-channel synchrony), which rescales the shared trajectory around the scale midpoint. We validate the per-subject calibration mechanism on four independent axes: leave-one-subject-out correlation be
    
[^151]: PAPER2LLM++：基于研究论文的大语言模型持续自我演化

    PAPER2LLM++: Continual Self-Evolution of LLMs from Research Papers

    [https://arxiv.org/abs/2610.02793](https://arxiv.org/abs/2610.02793)

    PAPER2LLM++ 提出了一个让大语言模型从研究论文中持续自我演化的框架，通过提取论文中的研究发现、验证局限性是否仍然存在，并借助“尝试-评估-提交”机制整合更新，从而在不遗忘先前改进、不损害通用能力的前提下实现模型的自动改进。

    

    对大语言模型（LLM）的研究不断揭示模型的局限性、其成因以及潜在的解决方案。然而，这些人类发现与模型演化在很大程度上仍然脱节：LLM 并不会自动从关于其自身缺陷的新研究中学习。我们提出了 PAPER2LLM++，一个使大语言模型能够从研究论文中持续自我演化的框架。PAPER2LLM++ 并非将论文仅仅视为可供检索的知识，而是将不断增长的文献作为模型改进的证据流和监督来源。对于每一篇新输入的论文，该框架会提取有证据支撑的研究发现，检验所报告的局限性在当前模型中是否依然存在，并在需要时将这些发现转化为候选学习信号。一个“尝试-评估-提交”程序仅在更新能够改进目标行为、且不会明显遗忘先前的改进或损害通用能力时，才将其整合到模型中。在一系列研究发现的序列流上……（摘要原文在此处截断）

    arXiv:2610.02793v1 Announce Type: new  Abstract: Research on LLMs continually uncovers model limitations, their causes, and potential solutions. Yet these human discoveries remain largely disconnected from model evolution: an LLM does not automatically learn from new research about its own failures. We introduce PAPER2LLM++, a framework for continual self-evolution of LLMs from research papers. Rather than treating papers merely as knowledge to retrieve, PAPER2LLM++ uses the growing literature as a stream of evidence and supervision for model improvement. For each incoming paper, it extracts evidence-grounded findings, tests whether the reported limitation persists in the current model, and, when needed, converts the findings into candidate learning signals. A try-evaluate-commit procedure integrates an update only when it improves the targeted behavior without substantially forgetting prior improvements or degrading general capabilities. Across a sequential stream of research-discover
    
[^152]: 法律与秩序：税法自动形式化

    Law And Order: Tax Law Autoformalization

    [https://arxiv.org/abs/2610.02792](https://arxiv.org/abs/2610.02792)

    提出 Law&Order 神经符号框架，通过结构对应与指称对应两种机制，利用大语言模型将税法表格和申报说明自动转化为可执行的符号程序，并通过单元格级验证和迭代式局部错误修复保证准确性。

    

    法律系统正越来越多地通过软件来实现，然而将法律文本转化为准确符号表示的可扩展方法仍未得到充分发展。我们通过税法来研究这一问题，税法中的表格和申报说明定义了涉及算术、分支、递归和表格推理的大型计算结构。我们提出了 Law&Order，一个用于将税务表格和申报说明自动形式化为可执行符号程序的神经符号框架。我们的方法建立了法律与逻辑之间的两种对应关系：结构对应，即将法律组件与符号组件（如单元格和附表）对齐；指称对应，即要求符号组件实现其法律对应部分所规定的计算。我们将大语言模型合成与单元格级验证相结合，并利用人工编写的 OpenTaxSolver 税务软件进行迭代式局部错误修复……

    arXiv:2610.02792v1 Announce Type: new  Abstract: Legal systems are increasingly implemented through software, yet scalable methods for translating legal texts into accurate symbolic representations remain underdeveloped. We study this problem through tax law, where forms and filing instructions define large computational structures involving arithmetic, branching, recursion, and tabular reasoning. We propose Law&Order, a neuro-symbolic framework for automatically formalizing tax forms and instructions into executable symbolic programs. Our approach establishes two forms of correspondence between law and logic: structural correspondence, which aligns legal and symbolic components such as cells and schedules, and denotational correspondence, which requires symbolic components to implement the computations specified by their legal counterparts. We combine large language model synthesis with cell-level verification and iterative localized error repair using human-written OpenTaxSolver tax 
    
[^153]: RL之前的OPD：利用在线策略蒸馏为基于评分标准的强化学习进行热启动

    OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation

    [https://arxiv.org/abs/2610.02781](https://arxiv.org/abs/2610.02781)

    提出两阶段训练框架：先以评分标准作为教师特权上下文进行在线策略蒸馏（RP-OPD）提供密集的token级监督，再以评分标准作为奖励进行强化学习，从而突破蒸馏的性能瓶颈。

    

    许多有用的语言模型任务无法通过精确的结果验证来评估。基于评分标准的强化学习（RL）通过根据明确标准对开放式回答进行评分来解决这一问题。然而，由于奖励是在完整回答生成之后才分配的，训练信号无法直接识别是哪些具体决策对最终得分做出了贡献。我们提出了一个两阶段训练框架：首先将评分标准用作特权教师上下文以提供密集的token级监督，然后将其用作奖励进行进一步的RL。在第一阶段，评分标准特权在线策略蒸馏（RP-OPD）让无法访问评分标准的学生模型在学生生成的前缀处匹配具备评分标准意识的教师模型的下一个token分布。在第二阶段，RL直接优化评分标准奖励，并突破了蒸馏带来的性能平台期。我们使用开源权重模型在健康和科学任务上评估了该框架。（摘要原文在此处截断）

    arXiv:2610.02781v1 Announce Type: cross  Abstract: Many useful language-model tasks cannot be evaluated by exact outcome verification. Rubric-based reinforcement learning (RL) addresses this issue by scoring open-ended responses against explicit criteria. However, because the reward is assigned after the complete response, the training signal does not directly identify which individual decisions contributed to the final score. We propose a two-stage training framework that uses rubrics first as privileged teacher context for dense token-level supervision, then as rewards for further RL. In the first stage, rubric-privileged on-policy distillation (RP-OPD), a student without access to the rubric matches a rubric-aware teacher's next-token distributions at student-generated prefixes. In the second stage, RL directly optimizes the rubric reward and improves beyond the observed distillation plateau. We evaluate the framework on health and science tasks using open-weight models. Across Heal
    
[^154]: 通过聚焦视图改进非结构化知识编辑中的原子事实回忆

    Improving Atomic-Fact Recall via Focused Views in Unstructured Knowledge Editing

    [https://arxiv.org/abs/2610.02772](https://arxiv.org/abs/2610.02772)

    该论文揭示了非结构化知识编辑中段落级编辑目标导致的“难度低估”问题，并提出通过聚焦视图的方式改进编辑后的模型，使其无需原始段落上下文即可可靠地回忆编辑文本中的各个原子事实。

    

    大型语言模型（LLM）日益成为事实知识的通用接口，但其参数并不能自动反映预训练之后发生变化的信息。知识编辑通过修改选定的知识、同时保留无关知识和通用能力，为代价高昂的重新训练提供了一种有针对性的替代方案。传统知识编辑使用结构化的事实三元组，而非结构化知识编辑（UKE）则使用包含多个事实的自由形式文本段落。然而，现有的UKE编辑器表现出一种被称为“上下文依赖”的失败模式：经过编辑的LLM通常能够复述编辑文本段落，但在没有原始段落上下文的情况下，却无法可靠地回忆其中的各个事实。我们发现在标准的段落级编辑目标下存在“上下文导致的难度低估”问题：越靠后的事实获得的真实上下文越丰富，因而产生较低的初始损失，使它们看起来更容易……（摘要原文在此处截断）

    arXiv:2610.02772v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly serve as general-purpose interfaces to factual knowledge, but their parameters do not automatically reflect information that changes after pretraining. Knowledge editing (KE) provides a targeted alternative to costly retraining by modifying selected knowledge and preserving unrelated knowledge and general capabilities. Conventional KE uses structured factual triples, whereas unstructured KE (UKE) uses free-form passages containing multiple facts. Nonetheless, existing UKE editors exhibit a failure mode known as context reliance: edited LLMs can often reproduce the editing passage but fail to reliably recall its individual facts without the original passage context. We identify context-induced difficulty underestimation under the standard passage-level editing objective: later facts receive increasingly rich ground-truth context and consequently incur lower initial losses, making them appear eas
    
[^155]: 具有1比特反馈的近最优固定置信度最优臂识别

    Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback

    [https://arxiv.org/abs/2610.02771](https://arxiv.org/abs/2610.02771)

    本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。

    

    我们研究在严格1比特反馈约束下的固定置信度最优臂识别问题。在每一轮中，学习者选择一个臂和一个查询集合，并且仅接收一个比特，该比特指示采样的奖励是否属于该集合。我们考虑一种具有逐臂定位的无分布有限方差设置，在这种设置下，直接的经验均值估计不再可用，截断处理变得不可避免。我们首先基于随机化阈值查询和截断尾积分恒等式，构建了一个时间一致的1比特均值估计基元。随后，我们将该基元嵌入到候选-挑战者式的最优臂识别算法中。固定截断算法提供了简单的任意时刻（ε,δ)-PAC保证，而分阶段自适应截断算法则将截断水平与当前分辨率相匹配，从而产生了间隙自适应的样本复杂度。我们还证明了一个K臂最坏情况的信息论下界（摘要原文在此处截断）。

    arXiv:2610.02771v1 Announce Type: cross  Abstract: We study fixed-confidence best-arm identification under strict 1-bit feedback constraints. At each round, the learner selects an arm and a query set, and receives only a single bit indicating whether the sampled reward belongs to that set. We consider a distribution-free finite-variance setting with arm-wise localization, where direct empirical mean estimation is no longer available and clipping becomes unavoidable. We first formulate a time-uniform 1-bit mean-estimation primitive based on randomized threshold queries and a clipped tail-integral identity. We then embed this primitive into candidate-challenger best-arm identification algorithms. A fixed-clipping algorithm gives a simple anytime $(\epsilon,\delta)$-PAC guarantee, while a phased adaptive-clipping algorithm matches the clipping level to the current resolution and yields a gap-adaptive sample complexity. We also prove a $K$-arm worst-case information-theoretic lower bound s
    
[^156]: 当历史未能转化为经验：语言智能体中的动作校准

    When History Fails to Become Experience: Action Calibration in Language Agents

    [https://arxiv.org/abs/2610.02769](https://arxiv.org/abs/2610.02769)

    研究发现语言智能体并不能可靠地将历史动作与其结果相关联，而只需简单地为每条观察标注其对应的前序动作，即可显著提升任务成功率并减少动作重复。

    

    语言智能体应当利用先前的尝试和环境反馈来改进同一任务中的后续决策。然而，提供额外的交互历史有时反而会降低任务成功率，这表明智能体并不能始终有效地利用这些信息。为了研究这一局限性，我们考察了智能体如何使用历史信息。我们发现，历史信息总体上能够提升任务完成率，但其中很大一部分收益即使在过去的动作被打乱时依然存在。破坏动作与观察之间的对应关系仅导致任务成功率出现轻微下降。因此我们假设，智能体在决定如何进行下一步时，并不能可靠地将过去的动作与其结果联系起来。为了验证这一假设，我们明确地将每条返回的观察标注为前一个动作的结果。这一简单的标注提升了任务成功率，并减少了下一步动作的重复，且未引入任何新的环境信息。

    arXiv:2610.02769v1 Announce Type: cross  Abstract: Language agents should draw on prior attempts and environmental feedback to improve subsequent decisions within the same task. However, providing additional interaction history can sometimes reduce task success, suggesting that agents do not consistently use this information effectively. To investigate this limitation, we examine how agents use history. We find that history improves task completion overall, yet much of this benefit persists even when past actions are shuffled. Disrupting the correspondence between actions and observations causes only a modest decline in task success. We therefore hypothesize that agents do not reliably connect past actions with their outcomes when deciding how to proceed. To test this hypothesis, we explicitly label each returned observation as the outcome of the preceding action. This simple annotation improves task success and reduces next-action repetition without introducing new environmental infor
    
[^157]: 动态大语言模型路由器常常被误导

    Dynamic LLM Routers are Often Misguided

    [https://arxiv.org/abs/2610.02762](https://arxiv.org/abs/2610.02762)

    研究发现六种商用动态LLM路由器在相同成本下的表现均不如在两个精选模型间随机选择的路由器，其根源在于标准优化目标本身就会奖励难度盲视、长度逆转和语义匹配等误导性行为。

    

    动态大语言模型（LLM）路由器承诺通过将每个查询发送给能够正确回答它的最廉价模型来降低推理成本。我们在一个涵盖八个任务类别的多样化基准上，对六个商用路由器在14种设置下进行了分析，发现它们无一能在相同成本下胜过在两个精心挑选的模型之间随机选择的路由器，有些路由器的表现甚至落后超过10个百分点。我们将这一差距追溯到路由器中普遍存在的四种模式：难度盲视、长度逆转、语义匹配和模型池次优。我们证明前三种模式恰恰是标准优化目标所奖励的：在实现成本上的成本-准确率帕累托效率，更倾向于升级中等难度的查询而非最难的查询，更偏好较短的查询而非较长的查询，并且更倾向于根据查询的来源而非其难度进行路由。我们还论证了能为大型模型池提供合理性的两个假设——模型粒度与模型专业化——在经验上并不成立。

    arXiv:2610.02762v1 Announce Type: new  Abstract: Dynamic LLM routers promise to cut inference costs by sending each query to the cheapest model that can answer it correctly. We analyze six commercial routers across 14 settings on a diverse benchmark spanning eight task categories, finding that none of them outperforms a router that randomly selects between two well-chosen models at matched cost. Some underperform by more than 10 percentage points. We trace this gap to four patterns prevalent across routers: difficulty blindness, length reversal, semantic matching, and roster suboptimality. We show that the first three are what the standard objective rewards: cost-accuracy Pareto efficiency on realized costs favors escalating moderately hard queries over the hardest ones, shorter queries over longer ones, and routing by a query's source over its difficulty. We also argue that the two assumptions that would justify large rosters, model granularity and model specialization, do not hold em
    
[^158]: 基于谱对齐的引导扩散轨迹校正

    Correcting Guided Diffusion Trajectories with Spectral Alignment

    [https://arxiv.org/abs/2610.02753](https://arxiv.org/abs/2610.02753)

    该论文提出“谱校正引导”方法，通过将采样中间状态的谱与前向过程的解析参考谱对齐来自适应校正CFG引导轨迹的偏差，无需训练即可提升条件图像生成的对齐度与视觉保真度。

    

    条件图像生成在实践中的成功取决于条件对齐和视觉保真度方面的细粒度差异。无分类器引导（CFG）是这一成功的核心，但其缺乏显式判据，使得难以评估引导轨迹是否正按预期推进。为填补这一空白，我们证明谱对齐为理解引导行为以及通过自适应校正改进引导扩散采样提供了一个有原则的判据。我们的分析指出，中间状态的谱可以作为衡量其与前向过程预期谱演化一致性的指标。基于这一观察，我们提出了谱校正引导，这是一种在采样过程中校正偏离解析参考谱的方法。该方法无需训练，并且在不修改基础模型的情况下，可广泛应用于各类扩散主干网络和条件生成任务。

    arXiv:2610.02753v1 Announce Type: cross  Abstract: The practical success of conditional image generation hinges on fine-grained differences in condition alignment and visual fidelity. Classifier-free guidance (CFG) is central to this success, but its lack of an explicit criterion makes it difficult to assess whether the guided trajectory is progressing as intended. To address this gap, we show that spectral alignment provides a principled criterion for understanding guidance behavior and improving guided diffusion sampling through adaptive correction. Our analysis identifies the spectra of intermediate states as an indicator of consistency with the expected spectral evolution of the forward process. Based on this observation, we introduce Spectral Correction Guidance, a method that corrects deviations from an analytic reference spectrum during sampling. The proposed method is training-free and applicable across diffusion backbones and conditional generation tasks without modifying the 
    
[^159]: 关于循环语言模型的思维链可监控性

    On the Chain-of-Thought Monitorability of Looped Language Models

    [https://arxiv.org/abs/2610.02741](https://arxiv.org/abs/2610.02741)

    本文首次系统性评估了循环语言模型的思维链可监控性，发现与匹配规模的非循环模型相比，LoopLM在多项任务中表现出任务相关的可监控性下降。

    

    思维链监控为检测模型的不良行为提供了一种有前景的方法。循环语言模型通过重复应用共享的Transformer层，在不增加模型规模的情况下提升有效计算深度并实现额外的潜在计算。然而，循环架构对思维链可监控性的影响在很大程度上仍未被探索。在这项工作中，我们对循环语言模型的思维链可监控性进行了首次系统性评估。我们研究了两种互补的设置：(1) 在同一LoopLM家族内改变循环深度，以分离额外循环计算的影响；(2) 将LoopLM与按参数量、Transformer层数或有效深度相匹配的非循环语言模型进行比较，以研究LoopLM是否更难被监控。在MonitorBench的八项任务以及标准设置和压力测试设置下，我们观察到思维链可监控性出现了依赖于任务程度的下降。

    arXiv:2610.02741v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring provides a promising approach for detecting undesirable model behavior. Looped language models (LoopLMs) repeatedly apply shared transformer layers, increasing effective computational depth and enabling additional latent computation without increasing model size. However, the effect of looped architectures on CoT monitorability remains largely unexplored. In this work, we provide the first systematic evaluation of CoT monitorability in LoopLMs. We study two complementary settings: (1) varying the loop depth within the same LoopLM family to isolate the effect of additional recurrent computation, and (2) comparing LoopLMs with non-looped language models matched by parameter size, transformer-layer count, or effective depth to study whether LoopLMs are less monitorable. Across eight tasks from MonitorBench and both standard and stress-test settings, we observe task-dependent reductions in CoT monitorability
    
[^160]: 前瞻性后见之明：基于预测-现实差距的自校准强化学习

    Prospective Hindsight: Self-Calibrating Reinforcement Learning via Prediction-Reality Gaps

    [https://arxiv.org/abs/2610.02740](https://arxiv.org/abs/2610.02740)

    提出前瞻性后见之明（PH）这一自校准强化学习训练原则，通过衡量智能体动作前预测与反馈后评估之间的“惊讶度”差距来加权梯度，使学习自动聚焦于智能体自我模型中最不准确的盲点样本。

    

    面向长时程智能体的强化学习依赖于纯粹的事后性训练信号：只有在观察到环境后果之后才进行信用分配，导致智能体在动作时刻的信念对梯度不可见。我们提出前瞻性后见之明，这是一种自校准训练原则，它通过一个源自智能体前瞻预测（反馈之前）与事后评估（反馈之后）之间差距的信号来增强任何事后性基础方法。这种逐次 rollout 的“惊讶度”能够识别出智能体自我模型最不准确的样本，并通过带停止梯度的惊讶度加权优势来放大这些样本的梯度贡献。由于前瞻预测器与策略共享参数，二者共同演化，逐步将学习焦点转移到智能体剩余的盲点上。我们将这一原则与特权信息差距联系起来，并证明最小化惊讶残差……（原文摘要至此中断）

    arXiv:2610.02740v1 Announce Type: cross  Abstract: Reinforcement learning for long-horizon agents relies on purely retrospective training signals: credit is assigned only after observing environmental consequences, leaving the agent's belief at action time invisible to the gradient. We introduce Prospective Hindsight (PH), a self-calibrating training principle that augments any retrospective base method with a signal derived from the gap between the agent's prospective prediction (before feedback) and the retrospective evaluation (after feedback). This per-rollout surprise identifies samples where the agent's self-model is most inaccurate and amplifies their gradient contribution through a stop-gradient surprise-weighted advantage. Since the prospective predictor shares parameters with the policy, the two co-evolve, progressively shifting focus to the agent's remaining blind spots. We connect this principle to a privileged-information gap and show that minimizing the surprise residual 
    
[^161]: TPBench：一个面向对话压缩的转折点基准

    TPBench: A Turning-Point Benchmark for Dialogue Compression

    [https://arxiv.org/abs/2610.02736](https://arxiv.org/abs/2610.02736)

    该论文提出 TPBench 基准，通过在相同保留预算下探测用户的初始目标、修改后槽位的当前值等互补信息目标，揭示了对话压缩中被整体保留分数掩盖的“转折点丢失”失败模式。

    

    一个压缩器可以保留对话中的事实，却仍然丢掉了改变这些事实的那个回合。用户纠正了一个价格、推翻了一个选择，或者添加了一个约束条件。我们将这种失败称为“转折点丢失”。单一的整体保留分数会掩盖这种失败，因为该分数将用户最初想要的内容与用户现在想要的内容混在了一起。我们提出了 TPBench，它在相同的标称保留预算下评估三种互补的信息目标。P1 要求回答用户的初始目标。P2 要求回答用户修改过的某个槽位的当前值。P3 则要求同时回答两者，所用对话中含有较晚被标注的槽位更新。当前值的答案来自 MultiWOZ 和 SGD 的人工对话状态标注；初始目标的答案是第一个用户轮次的第一句话。两者都不需要新的众包标注。针对特定探测点的评估对压缩方法的排名各不相同。在保留比例为 0.30 的联合探测中，所有被测试的压缩方法……（原文摘要在此截断）

    arXiv:2610.02736v1 Announce Type: cross  Abstract: A compressor can keep the facts of a dialogue and still drop the turn that changed them. A user corrects a price, reverses a choice, or adds a constraint. We call this failure turning-point eviction. One overall retention score hides it, because that score mixes what the user first wanted with what the user wants now.   We introduce TPBench, which evaluates three complementary information targets at shared nominal retention budgets. P1 asks for the user's initial goal. P2 asks for the current value of a slot the user revised. P3 asks for both, in dialogues with a late annotated slot update. The current-value answers come from the human dialogue-state annotations of MultiWOZ and SGD. The initial-goal answer is the first sentence of the first user turn. Neither requires new crowdsourcing.   The probe-specific evaluations rank compression methods differently. On the joint probe at a retained fraction of 0.30, every tested compressed metho
    
[^162]: 基于核典型相关分析重新审视视觉语言模型的视觉表征增强

    Revisiting Visual Representation Enhancement of VLMs via Kernel Canonical Correlation Analysis

    [https://arxiv.org/abs/2610.02718](https://arxiv.org/abs/2610.02718)

    本文提出利用核典型相关分析（KCCA）在特征子空间上刻画视觉语言模型与DINOv2之间的表征对齐，从而增强CLIP等模型的细粒度视觉感知能力。

    

    诸如CLIP这样的视觉语言模型展现出强大的语义泛化能力，但在细粒度视觉感知方面仍然存在局限。最近一项名为KUEA的工作提出了一种自然的解决方案：在以视觉为中心的DINOv2的监督下微调图像编码器，以逐元素对齐二者的核矩阵，同时通过正则化使嵌入保持接近预训练的视觉编码器，从而保留CLIP中的图文语义。然而，我们表明，削弱指向DINOv2的对齐损失的作用并不一定会降低其细粒度视觉性能，这说明核矩阵差异可能不足以进一步实现视觉表征增强，这促使我们重新审视对齐的构造方式。在本工作中，我们提出了一个新颖的视角，通过核典型相关分析（KCCA）在特征子空间上刻画表征对齐，该方法最大化投影相关性

    arXiv:2610.02718v1 Announce Type: cross  Abstract: Vision-language models such as CLIP exhibit strong semantic generalization, but remain limited in fine-grained visual perception. A recent work named KUEA presents a natural remedy by finetuning the image encoder under the supervision of the vision-centric DINOv2 to align their kernel matrices element-wisely, while regularizing the embeddings to remain close to the pretrained visual encoder for preserving image-text semantics in CLIP. However, we show that diminishing the role of the alignment loss to DINOv2 does not necessarily degrade its fine-grained visual performance, suggesting that the kernel-matrix discrepancy may be insufficient for further visual representation enhancement, motivating us to revisit the alignment formulation. In this work, we present a novel perspective to characterize representation alignment on feature subspaces through Kernel Canonical Correlation Analysis (KCCA), which maximizes the projection correlations
    
[^163]: Ego2World：将第一人称烹饪视频编译为可执行世界以支持信念状态规划

    Ego2World: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning

    [https://arxiv.org/abs/2610.02715](https://arxiv.org/abs/2610.02715)

    该论文提出Ego2World基准，将标注的第一人称烹饪视频编译为具有持久世界状态与智能体信念、部分可观测的可执行规划环境，用于系统评估信念状态规划器，并揭示被接受的操作往往仍无法达成任务目标。

    

    第一人称视频记录了人们进行日常活动的方式，然而测试智能体需要评估其自主选择动作的后果。我们提出了Ego2World，这是一个将带标注的烹饪活动转换为部分可观测条件下可执行规划环境的基准。其编译器将源步骤与对象链接到符号化动作规则、持久化的世界状态以及明确的任务条件，使研究人员能够执行智能体提出的动作并检验其结果。世界状态与智能体信念被分开维护，从而支持在连续任务中对规划与信息复用进行受控研究。对六个规划器在105个任务上的评估表明，被接受的操作常常未能达成任务目标。执行轨迹与条件检查能够区分中断的运行、部分达成，以及执行完成但目标未达成的情况。在另一项针对Qwen-Plus的配对研究中，持久化信念提升了动作……（摘要原文在此处截断）

    arXiv:2610.02715v1 Announce Type: new  Abstract: Egocentric videos capture how people carry out everyday activities, yet testing an agent requires evaluating the consequences of actions it chooses itself. We introduce Ego2World, a benchmark that turns annotated cooking activities into executable planning environments under partial observation. Its compiler links source steps and objects to symbolic action rules, persistent world states, and explicit task conditions, so researchers can execute an agent's proposed actions and check their outcomes. World state and agent belief are maintained separately, enabling controlled studies of planning and information reuse across continuing tasks. Evaluating six planners on 105 tasks shows that accepted operations often leave task goals unmet. Execution traces and condition checks distinguish interrupted runs, partial attainment, and completed execution without goal attainment. In a separate paired Qwen-Plus study, persistent belief improves actio
    
[^164]: 面向科学领域的终端环境自监督扩展

    Self-Supervised Scaling of Terminal Environments for Scientific Domains

    [https://arxiv.org/abs/2610.02710](https://arxiv.org/abs/2610.02710)

    提出软件在环重构这一自监督框架，通过从现有软件工作流中自动提取参考输出与验证目标，实现面向科学领域的终端智能体训练环境的可扩展、可复用构建。

    

    终端智能体正越来越多地被部署到软件工程之外的科学及其他专业领域。构建训练环境需要可执行的参考行为，以及能够区分语义正确性与表面上看似合理产物的领域专用验证器。为每个任务手工编写这些组件需要重复的工程投入，且限制了复用。我们提出了软件在环重构，这是一种自监督框架，它从现有软件工作流——即将结构化输入映射为输出的可执行程序——中获取参考输出和验证目标。对于每个工作流，我们执行多个输入配置，并将案例划分为公开观察和隐藏评估两部分。给定指令、输入模式以及公开的输入-输出观察，智能体在无法访问源工作流的情况下构建一个可编辑的程序，随后在隐藏配置上对候选程序进行评估。

    arXiv:2610.02710v1 Announce Type: cross  Abstract: Terminal agents are increasingly deployed beyond software engineering in science and other specialized domains. Constructing training environments requires executable reference behavior and a domain-specific verifier that distinguishes semantic correctness from superficially plausible artifacts. Authoring these components for each task requires repeated engineering and limits reuse. We introduce software-in-the-loop reconstruction, a self-supervised framework that obtains reference outputs and verification targets from existing software workflows, executable programs mapping structured inputs to outputs. For each workflow, we execute multiple input configurations and partition cases into public observations and hidden evaluations. Given the instruction, input schema, and public input--output observations, an agent constructs an editable program without access to the source workflow. The candidate is evaluated on hidden configurations a
    
[^165]: MuonIO：面向嵌入表与语言模型输出头的原则性范数感知下降方法

    MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads

    [https://arxiv.org/abs/2610.02705](https://arxiv.org/abs/2610.02705)

    MuonIO 将 Muon 优化器的原则性更新扩展到嵌入表和语言模型输出头——对语言模型头采用 2→∞ 算子范数、对嵌入表采用 1→2 算子范数，从而以统一的范数感知更新取代 AdamW。

    

    Muon 优化器通过求解以谱范数惩罚的损失的局部线性化，推导出隐藏线性层的更新规则，其动机来自对稠密线性层的 RMS 稳定性论证。然而，标准的 Muon 实现并未将输入层（嵌入表）和输出层（语言模型头）纳入这一原则性处理，而是对它们改用 AdamW。我们提出 MuonIO，一种对这两个层均适用的单一 Muon 风格更新。对于语言模型头 $\mathbf{L} \in \mathbb{R}^{V \times d}$，基于 softmax 输出几何的 Lipschitz 连续性，我们论证了使用 $2\to\infty$ 算子范数的合理性；而对于嵌入表 $\mathbf{E} \in \mathbb{R}^{d \times V}$，则基于 Bernstein & Newhouse (2025) 所识别的独热输入几何，我们采用 $1 \to 2$ 算子范数。恒等式 $\lVert\mathbf{L}\rVert_{2\to\infty}=\lVert\mathbf{L}^\top\rVert_{1\to2}$ 进而将两个矩阵统一纳入……（摘要在此处截断）

    arXiv:2610.02705v1 Announce Type: cross  Abstract: The Muon optimizer derives its update rule for hidden linear layers by solving a local linearization of the loss penalized by the spectral norm, motivated by an RMS-stability argument for dense linear layers. Standard Muon implementations, however, exclude the input (embedding table) and output (language model head) layers from this principled treatment, for which they use AdamW instead. We present MuonIO, a single Muon-style update for both of these layers. For the language model head $\mathbf{L} \in \mathbb{R}^{V \times d}$, we motivate the use of the $2\to\infty$ operator norm, due to the Lipschitz continuity of the softmax output geometry, while for the embedding table $\mathbf{E} \in \mathbb{R}^{d \times V}$, we draw on the $1 \to 2$ operator norm, based on the one-hot input geometry identified by Bernstein & Newhouse (2025). The identity $\lVert\mathbf{L}\rVert_{2\to\infty}=\lVert\mathbf{L}^\top\rVert_{1\to2}$ then puts both matr
    
[^166]: 大规模标签高效时间序列分类：具有反事实归因的双流OSSE-LSTM

    Label-Efficient Time Series Classification at Scale: A Dual-Stream OSSE-LSTM with Counterfactual Attribution

    [https://arxiv.org/abs/2610.02704](https://arxiv.org/abs/2610.02704)

    该论文提出双流OSSE-LSTM框架，将带挤压与激励重校准的全尺度CNN与双向LSTM结合，在每类仅有K个标注样本的极端标签稀缺条件下实现大规模时间序列分类，并利用反事实归因提升模型可解释性。

    

    时间序列由工业设备、可穿戴设备、电网和临床监护仪以巨大规模持续产生，然而标注工作仍然是手动的、昂贵的且依赖专家的。因此，大规模时间序列分析的制约因素不是数据量而是标签量，从业者面临的问题十分具体：每类需要标注多少个样本，分类器才能达到可用水平？我们直接研究这个问题，设定在一个标签空间固定且预先已知、决策规则必须仅由每类K个标注样本构建的场景中。我们提出双流OSSE-LSTM，这是一个情景式度量学习框架，它将具备挤压与激励（Squeeze-and-Excitation）重校准机制的全尺度CNN（用于多尺度模态提取而无需针对每个数据集调整卷积核）与用于捕捉全局时间上下文的双向LSTM相结合。两个流被独立归一化并融合为基于原型的……

    arXiv:2610.02704v1 Announce Type: new  Abstract: Time series are produced continuously at enormous scale by industrial equipment, wearables, power grids, and clinical monitors, yet annotation remains manual, expensive, and expert-dependent. The binding constraint in large-scale time series analytics is therefore not data volume but label volume, and the question facing a practitioner is concrete: how many examples per class must be labeled before a classifier becomes usable? We study this question directly, in a regime where the label space is fixed and known in advance and the decision rule must be constructed from only K labeled examples per class. We propose Dual-Stream OSSE-LSTM, an episodic metric-learning framework that pairs an Omni-Scale CNN with Squeeze-and-Excitation recalibration, for multi-scale motif extraction without per-dataset kernel tuning, with a Bidirectional LSTM for global temporal context. The two streams are independently normalized and fused into a prototype-or
    
[^167]: 基于分段式在线策略蒸馏学习推理修正

    Learning to Revise Reasoning with Segment-wise On-Policy Distillation

    [https://arxiv.org/abs/2610.02703](https://arxiv.org/abs/2610.02703)

    该论文提出分段式在线策略蒸馏方法，将教师模型对中间推理步骤的重写内容作为显式监督信号来训练学生模型修正自身推理，从而提升后续推理准确率并避免强化不良推理模式。

    

    在线策略蒸馏（OPD）通过在学生模型自身的 rollout 上进行训练，并利用教师模型提供的密集逐词监督，来提升大语言模型的推理能力。然而，逐词式的 OPD 并未显式提供一个连贯的替代推理步骤，用以说明学生的推理步骤应如何被修改才能改善后续推理。此外，当学生模型产生退化的推理前缀时，这种范式可能变得效果不佳，因为后续的教师监督仍然以该前缀为条件，可能会强化不良的推理模式。在本工作中，我们专注于通过分段式 OPD 学习推理修正，以重构中间推理步骤，从而更好地支持后续推理。通过受控的推理干预实验，我们发现用教师模型的重写内容替换学生模型的片段能够提升后续推理的准确率。因此，我们致力于解决将教师模型的重写内容转化为显式监督的问题。

    arXiv:2610.02703v1 Announce Type: new  Abstract: On-policy distillation (OPD) improves large language model reasoning by training students on their own rollouts with dense token-wise supervision from the teacher. However, token-wise OPD does not explicitly provide a coherent alternative reasoning step showing how the student's step could be revised to improve subsequent reasoning. Furthermore, this paradigm can become less effective when the student produces a degenerate reasoning prefix, as subsequent teacher supervision remains conditioned on that prefix and may reinforce poor reasoning patterns. In this work, we focus on learning reasoning revision with segment-wise OPD to rework intermediate reasoning steps and better support subsequent reasoning. Through controlled reasoning interventions, we find that replacing student segments with teacher redrafts improves subsequent reasoning accuracy. Therefore, we address the problem of turning teacher redrafts into explicit supervision for 
    
[^168]: 面向大语言模型推理的测试时校准学习

    Test-time Calibration Learning for Large Language Model Reasoning

    [https://arxiv.org/abs/2610.02695](https://arxiv.org/abs/2610.02695)

    提出了一种无需标签的测试时校准学习框架TTCL，能够直接在未标注的目标任务数据上联合优化大语言模型的推理准确性和置信度表达能力，摆脱了对真实标签的依赖。

    

    可靠的大语言模型（LLM）不仅要产生准确的答案，还必须表达出能够忠实反映其正确概率的置信度。这种校准对于识别不确定的预测以及在真实世界部署中支持可靠决策至关重要。近期研究将校准学习融入强化学习（RL）中，利用真实标签的正确性监督来联合优化答案正确性和口头表达的置信度。然而，它们对标注数据的依赖限制了其在实际测试时场景中的适用性，因为在这些场景中真实标签不可用，且校准可能需要适应新遇到的目标任务。为应对这一挑战，我们提出了测试时校准学习（TTCL），这是一个无标签框架，可直接在未标注的目标任务数据上联合调整推理准确性和口头表达的置信度。具体而言，TTCL 派生出自我监督……

    arXiv:2610.02695v1 Announce Type: cross  Abstract: Reliable large language models (LLMs) must not only produce accurate answers but also express confidence that faithfully reflects their probability of being correct. Such calibration is essential for identifying uncertain predictions and supporting reliable decision-making in real-world deployment. Recent studies incorporate calibration learning into reinforcement learning (RL), jointly optimizing answer correctness and verbalized confidence using ground-truth correctness supervision. However, their reliance on labeled data limits their applicability in practical test-time settings, where ground-truth labels are unavailable and calibration may need to adapt to newly encountered target tasks. To address this challenge, we propose Test-Time Calibration Learning (TTCL), a label-free framework that jointly adapts reasoning accuracy and verbalized confidence directly on unlabeled target-task data. Specifically, TTCL derives self-supervision
    
[^169]: 解耦记忆与上下文：面向令牌高效测试时持续学习的结构化记忆

    Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning

    [https://arxiv.org/abs/2610.02687](https://arxiv.org/abs/2610.02687)

    该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。

    

    大型语言模型越来越多地被部署在企业、科学和医疗应用中，在这些场景下，智能体必须整合领域特定知识并从经验中不断适应。上下文工程通过在推理时提供指令、策略和证据来改善模型行为，为权重更新提供了一种实用的替代方案。然而，在线调整上下文通常需要代价高昂的试错过程，而且查询往往被独立处理，导致有用的经验无法延续下去。记忆系统通过在多次交互之间保留信息来解决这一局限，但那些不断向共享上下文追加信息的方法会面临令牌成本不断上升、上下文窗口受限以及随上下文扩展而出现的性能退化。我们提出了上下文优化的统一形式化框架，并表明智能体记忆系统的更新可以被解释为一种优化……（摘要内容在此处截断）

    arXiv:2610.02687v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in enterprise, scientific, and medical applications, where agents must incorporate domain-specific knowledge and adapt from experience. Context engineering offers a practical alternative to weight updates by improving model behavior through instructions, strategies, and evidence supplied at inference time. However, adapting context online typically requires a costly trial-and-error process, while queries are often processed independently, preventing useful experience from carrying forward. Memory systems address this limitation by retaining information across interactions, but approaches that continually append information to a shared context face increasing token costs, context-window limits, and performance degradation as the context expands. We introduce a unified formulation of context optimization and show that an agent memory system update can be interpreted as an optimization 
    
[^170]: 大语言模型在患者证据演变时表现出不可靠的临床判断更新

    Large language models exhibit unreliable updating of clinical judgment as patient evidence evolves

    [https://arxiv.org/abs/2610.02684](https://arxiv.org/abs/2610.02684)

    该研究发现大语言模型在患者证据演变时无法可靠地更新临床判断，具体表现为对病情恶化证据反应过强的不对称性以及先验信念对预测的因果性干扰，且提示工程无法修复这些问题。

    

    大语言模型（LLM）在临床推理中的应用正受到越来越多的探索，但它们在患者证据演变时能否适当地修正判断仍不清楚。我们利用电子健康记录中匹配的重症监护轨迹评估了纵向的信念更新。在多种大语言模型中，当估计值发生变化时，以先前的判断为条件更多时候是增加而非减少预测误差，这一结果在第二个终点上也得到了复现。受控干预揭示了两种失败模式。第一，在固定先前评估的情况下，模型对恶化的呼吸证据的反应比匹配的改善证据更强烈；在中度和强证据水平上，经过余量归一化后这种不对称性仍然存在。第二，在固定当前证据的情况下，将先验风险从10%提高到90%会使估计值偏移26.2个百分点，证明了先验模型信念的因果影响。提示工程无法恢复可靠的更新。

    arXiv:2610.02684v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly explored for clinical reasoning, but whether they appropriately revise judgments as patient evidence evolves remains unclear. We evaluated longitudinal belief updating using matched intensive-care trajectories from electronic health records. Across diverse LLMs, conditioning on a preceding judgment more often increased than reduced prediction error when estimates changed, replicated for a second endpoint. Controlled interventions revealed two failure modes. First, with preceding assessment fixed, models responded more strongly to worsening than matched improving respiratory evidence; this asymmetry persisted after headroom normalization at moderate and strong evidence levels. Second, with current evidence fixed, increasing prior risk from 10% to 90% shifted estimates by 26.2 percentage points, demonstrating causal influence of prior model beliefs. Prompting did not restore reliable updating. 
    
[^171]: DataWeave：部署人类与大语言模型协同分析以支持结构化数据的探索性分析

    DataWeave: Deploying Human-LLM Analytics for Exploratory Structured Data Analysis

    [https://arxiv.org/abs/2610.02679](https://arxiv.org/abs/2610.02679)

    DataWeave是一个面向数据新闻工作场景的人机协同分析系统，通过结合对话式交互、模式接地、分析规划和可执行查询生成，解决LLM在探索大型结构化数据集时出现的模式不匹配、语义误读等可靠性问题。

    

    数据新闻是一种利用数据分析发掘有新闻价值故事的实践，它越来越依赖于记者和调查记者发现趋势、揭示差异、构建问责叙事的能力。在实际工作中，探索大型结构化数据集仍然缓慢且脆弱：记者需要在多年积累的众多数据集中浏览数百个变量，理解数据编码规范，并在研究假设不断演变的情况下编写复杂的分析代码。尽管大语言模型（LLM）常被宣传为“用自然语言提问，即可获得SQL或答案”，但真实的新闻编辑室工作流程暴露了其反复出现的失败问题，例如模式不匹配与漂移、对领域语义和单位的误读，以及隐含的未经声明的假设。我们提出了DataWeave，这是一个通过结合对话式交互、模式接地、分析规划与可执行查询生成来支持结构化数据探索性分析的系统。该系统并非将LLM简单地视为自动化……（原文摘要到此截断）

    arXiv:2610.02679v1 Announce Type: new  Abstract: Data journalism, the practice of using data analysis to surface newsworthy stories, depends increasingly on the ability of reporters and investigative journalists to uncover trends, disparities, and accountability narratives. In practice, exploring large structured datasets remains slow and brittle: journalists must navigate hundreds of variables across many datasets over years, understand data coding conventions, and write non-trivial analysis code while hypotheses evolve. Although LLMs are often touted as "ask in English, get SQL/answers," real newsroom workflows expose recurring failures, e.g., schema mismatches and drift, misread domain semantics and units, and silent assumptions. We present DataWeave, a system that addresses these needs by combining conversational interaction, schema grounding, analytical planning, and executable query generation to support exploratory analysis over structured data. Rather than treating LLMs as auto
    
[^172]: 把教师Token花在真正重要的地方：成功参照的在策略蒸馏

    Spend Teacher Tokens Where They Matter: Success-Referenced On-Policy Distillation

    [https://arxiv.org/abs/2610.02678](https://arxiv.org/abs/2610.02678)

    SR-OPD 以同一提示下的成功 rollout 为参照，聚焦于隐藏状态轨迹持续发散的失败 rollout 进行选择性教师监督，仅用 Vanilla OPD 约 3.46%–5.02% 的教师输入 token 即可达到相当的性能。

    

    在策略蒸馏（OPD）将学生模型生成的 rollout 与教师提供的密集 token 级监督相结合，但对每个 rollout 都提供这种监督需要大量的教师计算。我们提出了成功参照在策略蒸馏（SR-OPD），通过筛选哪些提示和 rollout 接受教师监督来降低这一成本。当学生模型针对同一提示既生成了成功的 rollout 又生成了失败的 rollout 时，成功的 rollout 可以作为选择失败 rollout 的天然参照。因此，SR-OPD 聚焦于这类提示，并优先处理隐藏状态轨迹与成功参照持续发散的失败 rollout，同时考虑估计的教师输入成本。在三组师生模型组合和六个数学推理基准上，SR-OPD 在单遍设置中仅使用 Vanilla OPD 所需教师输入 token 的 3.46%–5.02%，同时保持了相当的性能。

    arXiv:2610.02678v1 Announce Type: new  Abstract: On-policy distillation (OPD) combines student-generated rollouts with dense token-level supervision from a teacher, but providing such supervision for every rollout requires substantial teacher computation. We introduce Success-Referenced On-Policy Distillation (SR-OPD), which reduces this cost by selecting which prompts and rollouts receive teacher supervision. When the student produces both successful and failed rollouts for the same prompt, a successful rollout can serve as a natural reference for selecting failed rollouts. SR-OPD therefore focuses on such prompts and prioritizes failed rollouts whose hidden-state trajectories show sustained divergence from a successful reference, while accounting for estimated teacher-input cost. Across three teacher-student pairs and six mathematical reasoning benchmarks, SR-OPD uses only 3.46-5.02% of the teacher-input tokens required by Vanilla OPD in the one-pass setting while maintaining compara
    
[^173]: LEAP：为LLM智能体学习高效的动作提议

    LEAP: Learning Efficient Action Proposals For LLM Agents

    [https://arxiv.org/abs/2610.02670](https://arxiv.org/abs/2610.02670)

    该论文提出LEAP方法，通过学习一个高效的动作提议模型（而非使用现成的通用模型）为LLM智能体起草动作，并建立延迟分析框架揭示决定动作投机端到端加速的关键因素，从而显著提升智能体执行任务的速度。

    

    LLM智能体在执行任务时速度较慢。智能体一步接一步地完成任务：在每一步中先进行推理，然后选择一个动作去执行，下一步必须等上一步完成后才能开始。投机解码通过起草并验证推理token来加速推理阶段的执行。近期的工作也开始在动作阶段应用类似的思想：使用现成模型（通常较大）为目标模型起草动作提议，再由目标模型进行验证。大型起草模型与目标的匹配率更高，但生成提议耗时更长；而小型现成模型虽然速度快，却很少做出与目标一致的决策。我们提出了一个更普遍的问题：是什么决定了动作投机端到端的加速？为回答这一问题，我们为投机轮次建立了一个延迟分析框架，该框架比较每一轮投机所获得的收益与其付出的成本。收益取决于（摘要在此处被截断）

    arXiv:2610.02670v1 Announce Type: cross  Abstract: LLM agents are known to be slow in rollouts. An agent completes a task one step at a time. At each step, it reasons and then chooses an action to execute. The next step and action cannot start until the previous one has finished. Speculative decoding accelerates the rollouts at the reason phase by drafting and verifying the inference tokens. Recent works have also started to apply similar ideas at the action phase. These works use off-the-shelf models, usually large, to draft action proposals for target model to verify. Large drafters match the target more often but take longer to propose, while small off-the-shelf models are fast but rarely make the same decision as the target. We ask a more general question: what determines the end-to-end speedup of action speculation? To answer it, we develop a latency framework for the speculative round. The framework compares what a round gains with what it costs. The gain depends on how well the 
    
[^174]: 大语言连续扩散模型

    Large Language Continuous Diffusion Models

    [https://arxiv.org/abs/2610.02665](https://arxiv.org/abs/2610.02665)

    提出了首个大规模（3B/8B）连续扩散语言模型 Sigma，通过可操控的低维潜在轨迹、自回归模型热启动以及无分类器引导等推理技术，在数学推理和编码任务上取得了与离散扩散模型相当的性能。

    

    尽管离散扩散语言模型在快速并行解码方面取得了成功，但其非光滑、高维的空间阻碍了用于推理和推断加速的轨迹操控。为克服这一问题，我们提出了 Sigma，这是首个基于可操控、低维 ODE/SDE 潜在轨迹构建的大规模（3B/8B）连续扩散语言模型。Sigma 通过似然优化以分块方式进行训练，在对高斯扰动的词元嵌入进行联合去噪的同时学习最优的嵌入几何结构。为加速训练，Sigma 利用自回归（AR）模型的预训练权重进行热启动。在推理阶段，我们发现无分类器引导和分数温度对于实现高保真的推理与编码至关重要。在与最先进的离散模型（掩码扩散语言模型和自回归基线）进行的全面数学推理与编码评估中，Sigma 在标准基准上取得了与离散模型相当的性能……

    arXiv:2610.02665v1 Announce Type: cross  Abstract: Despite the success of discrete diffusion language models (dLMs) for fast parallel decoding, their non-smooth, high-dimensional space hinders trajectory steering for reasoning and inference acceleration. To overcome this, we present Sigma, the first large-scale (3B/8B) continuous dLM built on steerable, low-dimensional ODE/SDE latent trajectories. Trained blockwise via likelihood optimization, Sigma jointly denoises Gaussian-corrupted token embeddings while learning an optimal embedding geometry. To accelerate training, Sigma leverages pre-trained weights from autoregressive (AR) models for warm-starting. During inference, we identify classifier-free guidance and score temperature as essential for high-fidelity reasoning and coding. Across comprehensive math reasoning and coding evaluations against state-of-the-art discrete counterparts (masked dLMs and AR baselines), Sigma achieves competitive performance with discrete models on stand
    
[^175]: 长程智能体中的“幽灵”：跨轮次被忽视的安全约束导致的治理危害

    A GHOST in Long-Horizon Agents: Governance Hazard from Overlooked Safety Constraints across Turns

    [https://arxiv.org/abs/2610.02664](https://arxiv.org/abs/2610.02664)

    该论文发现长程智能体在良性交互条件下可能违反多轮之前设定的安全约束（即GHOST现象），在GPT-5.5上发生率达11.5%，并从理论上证明当剩余违反风险满足不可求和条件时，执行几乎必然进入危险区域。

    

    长程智能体如今在协助人类解决复杂问题方面正扮演着日益重要的角色。然而，恰恰是其延长的交互历史引入了一个尚未被充分探索的执行安全问题。在良性交互条件下，智能体可能执行违反许多轮次之前所指定安全约束的动作。我们将这种失败模式命名为“跨轮次被忽视的安全约束导致的治理危害”，它可能造成不可逆的损害。我们的实验表明，GHOST事件并非孤立案例：这种恰恰在良性交互条件下发生的失败模式，在GPT-5.5上的发生率为11.5%。此外，我们从理论上证明，如果沿每个安全前缀的剩余条件违反风险被一个不可求和的序列下界约束，那么执行几乎必然会进入危险区域。利用这一理论洞察，我们进一步……

    arXiv:2610.02664v1 Announce Type: new  Abstract: Long-horizon agents are now playing an increasingly significant role in assisting humans with complex problem-solving. However, it is exactly their extended interaction history that introduces an underexplored execution-safety concern. Under benign interaction conditions, an agent may execute an action that violates a safety constraint specified many turns earlier. We term this failure mode Governance Hazard from Overlooked Safety Constraints across Turns (GHOST), which may cause irreversible damage. Our experiments reveal that GHOST events are not isolated cases: this failure mode, occurring precisely under benign interaction conditions, yields an occurrence rate of 11.5% on GPT-5.5. Furthermore, we theoretically show that if the residual conditional violation hazard along each safe prefix is bounded below by a non-summable sequence, the execution enters the hazard region almost surely. Leveraging this theoretical insight, we further pr
    
[^176]: 面向内在低维数据的得分匹配扩散模型的泛化性质

    Generalization Properties of Score-matching Diffusion Models for Intrinsically Low-dimensional Data

    [https://arxiv.org/abs/2610.02663](https://arxiv.org/abs/2610.02663)

    该论文为流匹配模型在具有内在低维结构的数据上提供了统计泛化理论保证，推导出依赖于数据内在维度的 Wasserstein-p 有限样本误差界，克服了以往分析中限制性假设和忽略低维结构的不足。

    

    尽管流匹配模型在实证应用中取得了显著成功，但其统计泛化保证的理论研究仍然不完善。现有分析通常对估计的速度场施加限制性假设，且得到的收敛速率无法反映真实数据（如自然图像和分子几何结构）中普遍存在的内在低维结构。在本工作中，我们研究了流匹配模型从有限样本中学习未知分布 P_data 的统计泛化性能。我们对学习到的生成分布，在 Wasserstein-p 距离度量下（对所有 p≥1），推导出了有限样本误差界。具体而言，给定来自 P_data 的 n 个独立同分布样本，我们证明：对于每一个 d>d_p*(P_data)，只要恰当选择网络架构和超参数，学习到的分布 P̂^FM 就满足相应的误差界。

    arXiv:2610.02663v1 Announce Type: cross  Abstract: Despite the remarkable empirical success of flow-matching models, their statistical generalization guarantees remain underdeveloped. Existing analyses often impose restrictive assumptions on the estimated velocity field and yield convergence rates that fail to reflect the intrinsic low-dimensional structure common in real data, such as natural images and molecular geometries. In this work, we study the statistical generalization of flow-matching models for learning an unknown distribution $P_{\mathrm{data}}$ from finitely many samples. We derive finite-sample error bounds on the learned generative distribution, measured in the Wasserstein-$p$ distance, for all $p\geq 1$. Specifically, given $n$ i.i.d. samples from $P_{\mathrm{data}}$, we show that, for every $d>d_p^\ast(P_{\mathrm{data}})$ and appropriately chosen network architectures and hyperparameters, the learned distribution $\widehat{P}^{\mathrm{FM}}$ satisfies $ \mathbb{W}_p(\w
    
[^177]: 基于选择性状态空间模型的分布式学习：架构感知的收敛性分析

    Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis

    [https://arxiv.org/abs/2610.02659](https://arxiv.org/abs/2610.02659)

    该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。

    

    现代状态空间模型（SSM），如Mamba2，通过将线性时间复杂度的序列建模与递归状态空间动力学相结合，为Transformer提供了一种极具吸引力的替代方案。然而，SSM在分布式学习环境中的行为仍然鲜为人知。特别是，现有的标准联邦学习方法在很大程度上与架构无关，没有考虑到现代选择性SSM所特有的稳定性、选择性和状态空间参数化特性。为了解决这一问题，我们为单层和多层选择性SSM推导了架构感知的梯度和平滑度界限，并为FedAvg和FedProx推导了收敛界限，刻画了递归稳定性、输入相关的离散化以及状态投影范数如何影响联邦优化。随后，我们在由教师SSM生成的序列上，使用遵循所分析递归结构的学习器，对单层界限进行了数值验证。

    arXiv:2610.02659v1 Announce Type: cross  Abstract: Modern state space models (SSMs), such as Mamba2, provide a compelling alternative to transformers by combining linear-time sequence modeling with recurrent state-space dynamics. However, the behavior of SSMs in distributed learning settings remains poorly understood. In particular, the existing standard federated learning methods are largely architecture-agnostic, and do not account for the stability, selectivity, and state-space parameterization that characterize modern selective SSMs. To address this, we derive architecture-aware gradient and smoothness bounds for single- and multi-layer selective SSMs, and convergence bounds for FedAvg and FedProx, characterizing how recurrent stability, input-dependent discretization, and state projection norms affect federated optimization. We then numerically validate the single-layer bounds on sequences generated by a teacher SSM, using a learner that follows the analyzed recurrence. We use thi
    
[^178]: 一致性驱动的信念形成与LLM智能体中传染的群体动力学

    Coherence-Driven Belief Formation and Population Dynamics of Contagion in LLM Agents

    [https://arxiv.org/abs/2610.02654](https://arxiv.org/abs/2610.02654)

    本文实证测量了LLM智能体的信念采纳行为，发现其呈S形的复杂传染特征，且采纳阈值可由“传入信念与智能体先验信念的一致性”这一单一维度解释，并在群体层面观察到复杂传染的网络效应（聚类网络中传播更广）及自我维持的滞后共识现象。

    

    社会传染模型通常假设个体如何采纳信念，并由此推导出群体层面的行为。我们转而对语言模型智能体中的信念采纳进行了实证测量，量化了在给定多少同伴认可的情况下，智能体采纳某一主张的概率。我们发现这种采纳核呈S形（sigmoid），这是复杂传染的一个典型特征，其阈值对三个因素敏感：主张本身的合理性、信息来源的可靠性，以及智能体自身的倾向性。这三个维度可以被一个单一的有效维度很好地近似，我们提出该维度可以理解为传入信念与LLM智能体先验信念之间的一致性（coherence）。此外，我们在AI智能体系统的信念采纳集体动力学中观察到了复杂传染的一个标志性特征：信念在聚类网络上的传播比在随机网络上更远。这些系统还表现出分岔的级联窗口，以及自我维持的滞后共识。

    arXiv:2610.02654v1 Announce Type: new  Abstract: Models of social contagion usually assume how individuals adopt beliefs and derive population behavior from it. We instead empirically measure belief adoption in language model agents, quantifying the probability an agent adopts a claim given how many peers endorse it. We find this adoption kernel to be sigmoid, a characteristic of complex contagion, with a threshold that is sensitive to three sources: the claim's plausibility, the source's reliability, and the agent's disposition. These three dimensions are well approximated by a single effective dimension which we propose can be understood as the coherence of the incoming belief with the LLM agent's prior beliefs. Further, we observe a characteristic of complex contagion in the collective dynamics of belief adoption in a system of AI agents: further spread on clustered than random networks. These systems also exhibit a bifurcating cascade window, and self-sustaining hysteretic consensu
    
[^179]: 用于电子密度预测的等变流匹配

    Equivariant Flow Matching for Electron Density Prediction

    [https://arxiv.org/abs/2610.02651](https://arxiv.org/abs/2610.02651)

    该论文提出OrbFlow，一种SE(3)等变流匹配生成模型，通过预测高斯型轨道系数来生成电子密度，在保持紧凑基组效率的同时捕捉系数空间的结构相关性，为DFT自洽场计算提供高效且可迁移的初始化方案。

    

    密度泛函理论（DFT）的机器学习代理模型已被越来越多地用于降低第一性原理计算的成本。在这一领域，预测实空间电子密度可为自洽场（SCF）过程提供可扩展且可迁移的初始化方法。然而，现有方法面临一个明显的困境：基于网格的架构计算成本高昂，而基组方法无法捕捉系数空间中固有的结构相关性。为此，本文提出了OrbFlow，一种SE(3)等变生成模型，通过流匹配来预测高斯型轨道（GTO）系数。OrbFlow在保留紧凑原子中心基组效率的同时，用学习到的覆盖完整系数空间的概率路径取代了逐点回归。该模型通过两阶段轨迹课程进行训练，以缓解数值积分过程中的离散化漂移。

    arXiv:2610.02651v1 Announce Type: cross  Abstract: Machine learning surrogates for density functional theory (DFT) have been increasingly used to reduce the cost of first-principles calculations. In this arena, predicting real-space electron densities offers a scalable and transferable initialization for self-consistent field (SCF) procedures. However, current methods face a clear dilemma. That is, grid-based architectures incur a high computational cost, while basis-set methods fail to capture the structural correlations inherent in the coefficient space. Here, we develop OrbFlow, an $\mathrm{SE}(3)$-equivariant generative model that predicts Gaussian-type orbital (GTO) coefficients via flow matching. OrbFlow retains the efficiency of a compact atom-centered basis while replacing pointwise regression with a learned probability path over the full coefficient space. It is trained through a two-phase trajectory curriculum that mitigates discretization drift during numerical integration. 
    
[^180]: 无需解码的批量语音决策：单Token监督让冻结LLM听到转录文本之外的信息

    Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen LLM Hear Beyond the Transcript

    [https://arxiv.org/abs/2610.02638](https://arxiv.org/abs/2610.02638)

    DuplexJev将ASR编码器隐藏状态通过小型连接器输入冻结LLM，以单token分布直接读取决策答案，无需自回归解码即可在约0.1秒内批量完成80个语音决策，并通过交叉注意力连接器额外感知说话者的性别与情绪。

    

    全双工语音智能体需要做出许多微小的封闭式决策，而当前系统通过缓慢的自回归解码来回答这些问题。我们提出DuplexJev，它将ASR编码器的隐藏状态通过一个小型连接器输入冻结的LLM，并将每个问题读取为其选项上的单token分布。整个过程无需任何解码，一个8-GPU节点在约0.1秒内即可回答关于八段语音的80个决策。使用最后一层连接器时，口语问答的准确率接近直接阅读转录文本的水平（90% vs. 91%）。DuplexJev还能“听到”说话者本身：通过交叉注意力连接器，性别和情绪识别准确率均达到90%（分别从55%和28%提升），而其口语问答准确率仅下降1个百分点（从83%降至82%）。我们使用交叉熵在读取的答案token上训练决策，而非通常的转录文本蒸馏方法（其教师模型从未“听到”过语音本身），并将蒸馏仅保留用于内容学习。编码器和LLM均可互换使用；我们发布了模型权重、训练方案以及批量推理代码。

    arXiv:2610.02638v1 Announce Type: new  Abstract: Full-duplex voice agents make many small, closed decisions, which current systems answer by slow autoregressive decoding. We propose DuplexJev, which feeds ASR-encoder hidden states through a small connector into a frozen LLM and reads each question as a single-token distribution over its options. Nothing is decoded, and an 8-GPU node answers 80 decisions about eight utterances in about 0.1 s. With a last-layer connector, spoken QA stays close to reading the transcript (90% vs. 91%). DuplexJev also hears the speaker: gender and emotion accuracy both reach 90% (from 55% and 28%) with a cross-attention connector, whose spoken QA drops by only 1 point (83% to 82%). We train decisions with cross-entropy on the read-out answer token, instead of the usual transcript distillation, whose teacher never hears the voice, and keep distillation for content. Encoders and LLMs are interchangeable; we release weights, training recipe, a batched-inferenc
    
[^181]: 设计生成式人工智能用户反馈的未来

    Designing the Future of User Feedback for Generative AI

    [https://arxiv.org/abs/2610.02631](https://arxiv.org/abs/2610.02631)

    本研究通过与eBay的产学研合作，评估了当前生成式AI产品用户反馈机制的常见缺陷，提出了设计最佳实践建议，并设计测试了一个高效、灵活且注重用户价值的反馈收集工具原型。

    

    来自用户的部署后反馈可以成为监控和改进生成式AI系统与功能的一种经济高效、可扩展且具有代表性的手段。当有效实施时，提供此类反馈能够提升用户对生成式AI系统的参与度和信任度。政府法规和行业准则均要求开展部署后的用户参与，但关于如何设计既便于消费者使用、又能为产品团队提供可操作输入的反馈机制，目前几乎缺乏相关指导。我们以学术研究人员与eBay合作的形式开展了一项多阶段研究。我们对当前行业做法的基准评估发现了若干共性问题，包括可发现性不足、术语不清晰以及对用户价值的忽视。基于这些发现，我们制定了最佳实践建议，并设计并测试了一个反馈收集工具原型。该工具旨在为用户提供高效、灵活且积极的……（摘要原文在此处截断）

    arXiv:2610.02631v1 Announce Type: new  Abstract: Post-deployment feedback from users can be a cost-effective, scalable, and representative means to monitor and improve generative AI systems and features. When implemented effectively, giving such feedback can increase users' engagement with and trust in GenAI systems. Government regulations and industry guidelines call for post-deployment user engagement, but there is little guidance on designing mechanisms that are usable for consumers and provide actionable input for product teams. We conducted a multi-phase study as a collaboration between academic researchers and eBay. Our benchmark evaluation of current industry approaches identified common issues including lack of discoverability, unclear terminology, and inattention to user value. Based on these findings, we developed best-practice recommendations and designed and tested a prototype feedback-collection tool. The tool aimed to provide users with an efficient, flexible, and positiv
    
[^182]: 迷失在请求中：沟通方式的变化如何干扰邮件智能体的检索与行动

    Lost in the Request: How Communication Variation Disrupts Retrieval and Action in Email Agents

    [https://arxiv.org/abs/2610.02627](https://arxiv.org/abs/2610.02627)

    该论文揭示了邮件智能体存在“沟通鲁棒性”缺陷：即使任务实质完全不变，仅请求的表达方式（如间接、冗长或方言变体）发生改变，就会显著降低RAG系统和工具使用智能体的检索与执行性能。

    

    邮件助手不应仅因为用户以不同方式表达相同的请求就完成更少的工作。然而，大多数基准测试只用一个规范化请求来测试每个任务，使得这种形式的鲁棒性在很大程度上未被测量。我们测试了当所请求的信息、可用证据和预期结果保持不变，但沟通风格或英语变体发生变化时，邮件助手是否仍然可靠。我们沿五个沟通风格轴和四个基于规则的方言条件构建了经过验证的变体，并在三个基准上进行评估：一个检索增强生成（RAG）流水线和两个工具使用智能体。间接请求降低了所有三个基准的性能，而正式请求降低了两个智能体基准的性能。对这些系统的更深入检查表明，这些失败有不同的原因：冗长的请求主要通过使相关邮件更难被找到而损害词汇检索器……

    arXiv:2610.02627v1 Announce Type: new  Abstract: An email assistant should not complete less work simply because a user phrases the same request differently. Yet most benchmarks test each task with only one canonical request, leaving this form of robustness largely unmeasured. We test whether email assistants remain reliable when the requested information, available evidence, and expected outcome stay fixed, but the communication style or English variety changes. We construct validated variants along five communication-style axes and four rule-based dialect conditions, and evaluate them on three benchmarks: a retrieval-augmented generation (RAG) pipeline and two tool-using agents. Indirect requests reduce performance on all three benchmarks, while formal requests reduce performance on both agentic benchmarks. Examining the systems more closely shows that these failures have different causes. Verbose requests mainly hurt a lexical retriever by making the relevant email harder to find. B
    
[^183]: CuBEs：文化情境化的行为评估与文化盲LLM评判者的局限性

    CuBEs: Culturally-Situated Behavioral Evaluations and the Limitations of Culture-Blind LLM Judges

    [https://arxiv.org/abs/2610.02622](https://arxiv.org/abs/2610.02622)

    提出了文化情境化行为评估框架CuBEs，通过构建涵盖12种文化的人工标注数据集，将文化背景注入行为测试流程，揭示了“一刀切”式LLM评判者无法捕捉的显著跨文化行为差异。

    

    评估大语言模型（LLM）行为——如谄媚性、自我偏好或过度自信——的发生及其触发因素，对于预测模型在真实世界部署中的风险至关重要。然而，现有的情境化行为评估通常忽略文化背景，限制了其在日益全球化的用户群体中的普适性。为填补这一空白，我们提出了CuBEs——文化情境化行为评估，用于探究不同用户文化下的响应模式。我们首先扩展了一个自动化测试流程，将文化背景注入行为测试场景及后续评估之中。我们通过构建一个人工标注的数据集来评估该流程的文化适应性，该数据集捕捉了12种不同文化中行为理解的细微维度。我们的数据集揭示了显著的跨文化差异，而“一刀切”式的评判标准无法捕捉这些差异。通过评估13个开源和……（原文在此截断）

    arXiv:2610.02622v1 Announce Type: new  Abstract: Evaluating the occurrence and triggers of large language model (LLM) behaviors - such as sycophancy, self-preference, or over-confidence - is critical for predicting real-world model deployment risks. However, existing situated behavioral evaluations typically ignore cultural context, limiting their generalizability across an increasingly global user base. To address this gap, we propose CuBEs - Culturally-situated Behavior Evaluations that probe for response patterns across diverse user cultures. We first extend an automated testing pipeline to inject cultural context into behavioral test scenarios and subsequent evaluation. We assess the cultural adaptability of this pipeline by building a human-labeled dataset that captures nuanced dimensions of behavior understanding across 12 distinct cultures. Our dataset reveals significant cross-cultural variations that one-size-fits all judgments fail to capture. Through evaluating 13 open- and 
    
[^184]: WebUIProof：基于UI代理执行框架的WebUI代码生成器基准测试

    WebUIProof: Benchmarking WebUI Code Generators with UI-Agent Execution Harness

    [https://arxiv.org/abs/2610.02617](https://arxiv.org/abs/2610.02617)

    提出WebUIProof基准，通过UI代理在无头浏览器中执行可执行的交互测试来评估WebUI代码生成的功能正确性，揭示了八个商用大模型在交互类需求上的频繁失败。

    

    大规模评估WebUI代码生成十分困难：生成的代码可能编译通过且看似合理，但在用户交互时却会失败，而以往的基准测试主要依赖自由格式的提示词加静态检查（如构建成功、截图），因而遗漏了功能正确性。我们提出了WebUIProof，一个面向执行的基准测试，为WebUI生成提供结构化规范以及密集、可执行的交互测试，涵盖两大任务类别：通用WebUI（如仪表盘、游戏、交互式工具）和3D交互仿真（如粒子/星系系统、物理动力学）。WebUIProof包含一个UI代理执行框架，可在无头浏览器中通过迭代的“计划—行动—观察”循环运行可执行的交互测试：它定位DOM元素、执行操作、观察由此产生的UI/DOM变化，并检查指定的断言。我们在八个商用大语言模型上进行了评估，观察到模型在基于交互的需求上频繁失败。

    arXiv:2610.02617v1 Announce Type: cross  Abstract: Evaluating WebUI code generation at scale is difficult: outputs may compile and look plausible yet fail under user interaction, and prior benchmarks largely rely on free-form prompts with static checks (build success, screenshots) that miss functional correctness. We introduce WebUIProof, an execution-oriented benchmark that provides structured specifications and dense, executable interaction tests for WebUI generation across two task families: general WebUIs (e.g., dashboards, game, interactive tools) and 3D interactive simulation (e.g., particle/galaxy systems, physics dynamics). WebUIProof includes a UI-agent harness that runs executable interaction tests in a headless browser using an iterative plan--act--observe loop: it locates DOM elements, performs actions, observes resulting UI/DOM changes, and checks the specified assertions. We evaluate across eight commercial LLMs and observe frequent failures on interaction-based requireme
    
[^185]: VERSE：面向智能体框架的经过验证的自我进化优化器

    VERSE: Verified Self-Evolving Optimizer for Agent Harnesses

    [https://arxiv.org/abs/2610.02616](https://arxiv.org/abs/2610.02616)

    提出VERSE，一个经过验证的自我进化优化器，它不仅改进智能体框架，还让优化器自我进化其诊断、编辑与验证流程（如测试草稿编辑、重放故障、扰动可疑步骤），在具备基于执行的验证时取得最佳优化效果。

    

    框架进化可以改进LLM智能体的提示词、工具和工作流程，而优化器自身的工具和流程却往往保持固定。我们研究优化器是否可以通过同时改进其诊断故障、开发编辑和测试效果的方式，来更有效地改进另一个智能体。两个观察结果指导了我们的设计：在一项对照研究中，没有基于执行的验证时，优化器自我进化无法提升性能；而当验证可用时，它在该研究中取得了最佳结果。在五个执行器上，自我进化的优化器为故障分析、验证、训练审计和工作流控制构建了自己的工具。基于这些发现，我们提出了VERSE，一个面向智能体框架的经过验证的自我进化优化器。VERSE允许优化器在提交前测试草稿编辑、重放故障并扰动可疑步骤，同时跨轮次跟踪修复和回归情况。利用这种反馈……

    arXiv:2610.02616v1 Announce Type: new  Abstract: Harness evolution improves an LLM agent's prompts, tools, and workflow, while the optimizer's own tools and procedures often remain fixed. We study whether an optimizer can improve another agent more effectively by also improving how it diagnoses failures, develops edits, and tests their effects. Two observations guide our design. In a controlled study, optimizer self-evolution fails to improve performance without execution-based verification, but achieves the best result of that study when verification is available. Across five executors, self-evolving optimizers build their own tools for failure analysis, verification, training audits, and workflow control. Motivated by these findings, we introduce VERSE, a Verified Self-Evolving optimizer for agent harnesses. VERSE lets the optimizer test draft edits, replay failures, and perturb suspected steps before submission, while tracking fixes and regressions across rounds. Using this feedback
    
[^186]: 时间序列预测基准需要基于场景的压力测试

    Time Series Forecasting Benchmarks Need Scenario-Grounded Stress Testing

    [https://arxiv.org/abs/2610.02608](https://arxiv.org/abs/2610.02608)

    该论文指出当前时间序列预测的评估基准过于狭窄，无法捕捉真实部署系统中因结构化事件导致的语义、因果和系统级失效模式，因此倡导引入基于真实场景的压力测试来更可靠地评估预测模型，尤其是基础模型。

    

    时间序列预测（TSF）日益驱动着交通、能源、金融、医疗和基础设施领域的决策，然而当前的评估方式仍然过于狭窄：标准基准仅奖励低泛化误差，而鲁棒性研究通常将失效简化为高斯噪声、随机掩码或有界对抗扰动。这掩盖了已部署预测系统的真实失效模式。输入侧异常不仅仅是更嘈杂的输入：它们往往反映了改变时间动态、破坏跨变量依赖关系、引发状态转换、或从故障传感器传播至下游决策的结构化事件。这些语义、因果和系统层面的失效无法仅通过独立同分布（i.i.d.）扰动来忠实捕捉。TSF基础模型的兴起使这一评估缺口更加紧迫，因为不可审计的预训练语料库使得留出集泛化评估变得越来越不可靠。因此，我们倡导基于场景的……

    arXiv:2610.02608v1 Announce Type: new  Abstract: Time series forecasting (TSF) increasingly drives decisions in transportation, energy, finance, healthcare, and infrastructure, yet current evaluation remains overly narrow: standard benchmarks reward low held-out error, while robustness studies typically reduce failure to Gaussian noise, random masking, or bounded adversarial perturbations. This obscures the real failure modes of deployed forecasting systems. Input-side anomalies are not merely noisier inputs: they often reflect structured events that alter temporal dynamics, break cross-variable dependencies, induce regime shifts, or propagate from faulty sensors to downstream decisions. These semantic, causal, and system-level failures cannot be faithfully captured by i.i.d. perturbations alone. The rise of TSF foundation models makes this evaluation gap more urgent, as unauditable pretraining corpora make held-out generalization increasingly unreliable. We therefore advocate scenario
    
[^187]: TasteBench：从分子到可持续食品的感官预测多模态基准

    TasteBench: Multimodal Benchmark for Sensory Prediction, from Molecules to Sustainable Foods

    [https://arxiv.org/abs/2610.02599](https://arxiv.org/abs/2610.02599)

    TasteBench是首个面向可持续食品感官预测的多模态基准，通过覆盖2.1万余次人类评估的食品级排序任务和1.5万风味分子的味觉分类任务，并首次刻画了人类感官数据本身的信度上限，为加速植物基食品设计提供了计算评估工具。

    

    可持续蛋白质发现领域缺乏类似分子对接或密度泛函理论那样能加速药物和材料发现的快速计算代理方法。评估一种新型食品是否具有与其动物性目标食品相似的味道，需要昂贵的人类感官评审小组，这成为“设计-构建-测试”循环的瓶颈。我们提出了TasteBench，一个面向感官预测的多模态基准和隐私保护型竞赛，涵盖两项任务：一是食品级排序任务，基于215种植物基食品在24个产品类别中超过2.1万次人类感官评估构建，产生935个类别内排序对；二是支撑性的分子级味道分类任务，覆盖1.5万种风味分子。为了严谨地解释模型性能，我们对真值数据进行了表征：评审员之间的一致性较低（Krippendorff's α = .077），评审小组汇总排序的分半信度上限为0.825……

    arXiv:2610.02599v1 Announce Type: new  Abstract: Sustainable protein discovery lacks the fast computational proxies, analogous to molecular docking or density functional theory, that accelerate drug and materials discovery. Evaluating whether a novel food tastes like its animal-based target requires expensive human sensory panels, bottlenecking the design-build-test loop. We introduce TasteBench, a multimodal benchmark and privacy-preserving competition for sensory prediction, spanning two tasks: a food-level ranking task built on 21K+ human evaluations across 215 plant-based foods in 24 product categories, yielding 935 within-category ranking pairs, and a supporting molecular-level taste classification task over 15K flavor molecules. To enable rigorous interpretation of model performance, we characterize the ground truth: inter-rater agreement among panelists is low (Krippendorff's $\alpha = .077$), and the split-half reliability ceiling of panel-aggregated rankings is .825, establish
    
[^188]: 因果性如何弥合语义鸿沟

    How Causality Bridges the Semantic Gap

    [https://arxiv.org/abs/2610.02594](https://arxiv.org/abs/2610.02594)

    该论文提出以因果结构替代人类知识来为未命名变量赋予语义，将其形式化为“结构约束的语义对齐”，并构建 CausalBridge 框架，从测量数据（含隐变量）中发现因果图并在其依赖关系约束下求解变量嵌入，从而从变量对其他变量的作用方式中解读其含义。

    

    数值测量捕捉了系统的行为方式，但往往未指明其变量的含义：有些变量被测量却从未被标注，另一些变量则从未被测量。现有方法通过参考人类的一般知识为这些变量赋予语义，但在知识存在之处会继承其偏见，在知识缺失之处则无能为力。我们转而利用因果结构来弥合测量与其含义之间的鸿沟，从变量作用于其他变量的方式中解读其语义。我们将这一过程形式化为“结构约束的语义对齐”：以少量已知名称的嵌入作为锚点，在因果图所蕴含的依赖关系约束下求解每个未命名变量的嵌入。基于此，我们构建了 CausalBridge 框架，该框架从测量数据（包括隐变量）中发现因果图，并在这些依赖关系约束下求解嵌入。（原文摘要在此处截断）

    arXiv:2610.02594v1 Announce Type: cross  Abstract: Numerical measurements capture how a system behaves, but often leave the meanings of its variables unspecified. Some variables are measured but never labeled, and others are never measured at all. Existing methods assign semantics to such variables by consulting general human knowledge, but this inherits its biases where that knowledge exists and offers nothing where it does not. We bridge this gap between measurements and their meanings with causal structure instead, reading a variable's semantics from how it acts on other variables. We formalize this as structure-constrained semantic alignment, in which the embedding of each unnamed variable is solved under the dependence relations implied by the causal graph, with the embeddings of a few known names as anchors. Accordingly, we build CausalBridge, a framework that discovers the causal graph from the measurements, latent variables included, solves for the embeddings under those relati
    
[^189]: 开放式基准：从智能体记录中测量认知过程

    Open-Endedness Bench: Measuring Epistemic Process from Agent Records

    [https://arxiv.org/abs/2610.02588](https://arxiv.org/abs/2610.02588)

    提出 OEB，一种只依赖智能体执行记录（不使用参考答案或结果分数）来评估其认知过程——假设形成、检验与修正——的与基准无关的方法论。

    

    智能体越来越多地被赋予开放式研究任务：从自主设计的实验中发现经验规律、改进一个无人知晓最优解的启发式方法，或者打破既有纪录。它们的执行日志记录了这项研究的每一步，然而对这些运行的评判目前仍然仅依据结果分数。仅凭该分数无法确定智能体的结论是否源自其实际执行的实验，而且参考答案可能并不存在。我们评估智能体的认知过程：它如何形成假设、检验假设，并根据证据修正假设。我们提出了 OEB（Open-Endedness Bench，开放式基准），这是一种与具体基准无关的方法论，它只读取智能体的执行记录，从不使用参考答案或结果分数。OEB 将该记录编译为统一的认知事件图，图中的边将智能体陈述的命题与检验这些命题的已执行动作连接起来；每个节点都带有可由代码核验的精确摘录……

    arXiv:2610.02588v1 Announce Type: new  Abstract: Agents are increasingly given open-ended research tasks: discovering an empirical law from self-designed experiments, improving a heuristic whose optimum nobody knows, or beating a standing record. Their execution logs record every step of this research, yet the runs are still judged by their outcome score. That score alone does not establish whether an agent's claims follow from executed experiments, and a reference answer may be unavailable. We evaluate the agent's epistemic process: how it forms hypotheses, tests them, and revises them in response to evidence. We introduce OEB (Open-Endedness Bench), a benchmark-agnostic methodology that reads only the agent's execution record and never a reference answer or an outcome score. OEB compiles the record into a unified epistemic event graph whose edges connect the propositions the agent states to the executed actions that test them; each node carries an exact excerpt that code verifies aga
    
[^190]: Jev式类型化决策模型中标签凌驾于定义之上

    Labels Override Definitions in Jev-Style Typed Decision Models

    [https://arxiv.org/abs/2610.02586](https://arxiv.org/abs/2610.02586)

    该研究发现Jev式类型化决策模型在输出概率时主要依据选项的标签而非其书面定义，即使规则只写在定义中也是如此，并据此提出了“选项-标签偏差”这一概念，同时引入了用于验证该现象的PolicyBench合成路由测试套件。

    

    类型化决策模型通过为若干由调用方定义的选项各返回一个概率，来回答针对某个输入的固定问题。每个选项都附带一个简短标签和一段书面定义，开发者正是通过这段定义来声明模型应当应用的规则。Jev为路由、内容审核和分流场景引入了这一接口，随后出现了开放实现，而每当语言模型通过为标签字符串打分来充当分类器时，同样会执行这一操作。我们研究了这些开放实现——其权重可供检查和修补——并探究其输出的概率究竟遵循定义还是标签。我们将这种对标签的偏好称为“选项-标签偏差”。在四个开放权重的类型化决策模型、三种从Qwen2.5骨干网络读取答案的方式、十一个分类任务，以及PolicyBench（我们引入的一个合成路由测试套件，其中规则仅出现在定义中）上，答案主要取决于标签。删除每一个……（摘要在此处被截断）

    arXiv:2610.02586v1 Announce Type: new  Abstract: A typed decision model answers a fixed question about an input by returning a probability for each of several caller-defined options. Each option carries a short label and a written definition, which is where a developer states the rule the model should apply. Jev introduced this interface for routing, moderation and triage, open implementations followed, and the same operation occurs whenever a language model is used as a classifier by scoring label strings. We study the open implementations, whose weights we can inspect and patch, and ask whether the probability follows the definitions or the labels. A preference for the label we call option-label bias. Across four open-weight typed decision models, three ways of reading an answer from a Qwen2.5 backbone, eleven classification tasks and PolicyBench, a synthetic routing suite we introduce in which the rule appears only in the definitions, the answer is mostly the labels. Deleting every 
    
[^191]: 基于可验证、反馈驱动的语言模型回答临床医生关于试验证据表的问题

    Answering clinicians' questions over trial evidence tables with verifiable, feedback-driven language models

    [https://arxiv.org/abs/2610.02576](https://arxiv.org/abs/2610.02576)

    FD-SCoPE是一个可验证、可从专家反馈中学习的语言模型框架，既能回答临床医生对试验证据表的直接查询，也能回答需要推导属性的问题，并在肿瘤学证据表上以77.7%的推导值F1超越四种替代方法。

    

    系统综述将临床试验浓缩为证据表，然而临床医生只能通过数据库查询来检索这些表格，且许多问题涉及表格中未记录的属性，例如药物的靶点类别或统一化的终点指标。在此，我们提出FD-SCoPE，这是一个语言模型框架，能够回答这两类问题，公开每个答案背后的查询、所选试验和推导规则，并能从专家纠正中学习。在一个包含159条免疫检查点抑制剂试验记录的肿瘤学证据表上，FD-SCoPE完成了全部140个临床医生风格的任务（替代方案为90.7%-97.9%）。对于需要推导属性的问题，它以89.8%的阳性预测值检索出99.3%的相关试验记录，并优于四种替代方法（推导值F1为77.7%，而其他方法为64.8%-73.4%）。基于参考答案模拟的299个问题的专家纠正，提升了模型在1,201个未见问题上的F1表现

    arXiv:2610.02576v1 Announce Type: new  Abstract: Systematic reviews condense clinical trials into evidence tables, yet clinicians can interrogate these tables only through database queries, and many questions concern attributes that the table does not record, such as a drug's target class or a harmonised endpoint. Here we introduce FD-SCoPE, a language-model framework that answers both kinds of question, exposes the query, the selected trials and the derivation rule behind every answer, and learns from expert corrections. On an oncology evidence table of 159 immune checkpoint inhibitor trial records, FD-SCoPE completed all 140 clinician-style tasks (alternatives, 90.7-97.9%). For questions needing derived attributes it retrieved 99.3% of relevant trial records at a positive predictive value of 89.8% and outperformed four alternative approaches (derived-value F1 77.7% versus 64.8-73.4%). Corrections on 299 questions, simulated from reference answers, raised F1 on 1,201 unseen questions 
    
[^192]: 通过有效提示提升大语言模型生成代码的能效

    Improving the Energy-Efficiency of the Code Generated by LLMs through Effective Prompting

    [https://arxiv.org/abs/2610.02571](https://arxiv.org/abs/2610.02571)

    本研究系统评估了21种提示策略，发现有效的提示工程可使大语言模型生成的Python和C++代码能耗分别降低最多25%和17%。

    

    随着AI辅助编程日益成为主流，AI生成软件对环境的影响已成为一个重要的考量因素。这促使人们在评估大语言模型（LLM）生成的代码时，不仅关注功能正确性，还要考虑执行效率和能耗。然而，尽管代码生成技术取得了长足进步，前沿LLM却很少基于其生成代码的能效进行评估。在这项工作中，我们对21种用于节能代码生成的提示策略进行了全面评估，并从中筛选出8种策略，在10个广泛使用的开源权重和专有LLM上进行评估。我们以基线提示为参照，评估了这些策略在Python和C++代码生成中的有效性。在所有被评估的模型中，所选的提示策略使Python代码生成的能耗最多降低25%，C++代码生成的能耗最多降低17%。在模型层面，Python能耗降低……

    arXiv:2610.02571v1 Announce Type: cross  Abstract: As AI-assisted programming becomes increasingly mainstream, the environmental impact of AI-generated software has emerged as an important consideration. This motivates evaluating LLM-generated code beyond functional correctness by considering execution efficiency and energy consumption. However, despite substantial advances in code generation, frontier LLMs are rarely evaluated based on the energy efficiency of the code they produce. In this work, we conduct a comprehensive evaluation of 21 prompting strategies for energy-efficient code generation and identify 8 strategies for evaluation across 10 widely used open-weight and proprietary LLMs. We evaluate their effectiveness for both Python and C++ code generation relative to a baseline prompt. Across the evaluated models, the selected prompting strategies achieved energy reductions of up to 25% for Python and 17% for C++ code generation. At the model level, Python energy reductions rea
    
[^193]: Pincer：基于数字孪生的智能体资源授权机制

    Pincer: Resource Authorization for Agents using a Digital Twin

    [https://arxiv.org/abs/2610.02569](https://arxiv.org/abs/2610.02569)

    Pincer 提出了一种基于数字孪生、在资源层运行的授权防御机制，为长周期自主编码智能体提供可持续的权限管理，克服了用户中介沙箱的策略衰减与权限疲劳问题，并与现有工具调用层防御形成互补。

    

    编码智能体正变得日益长周期化、自主化，依赖通用 shell，并维护自身的持久记忆以实现自我改进。虽然这些能力使智能体变得强大，但也使其更难抵御外部攻击者。限制这种架构的防御手段——类型化工具、信息流控制或策略预测引擎——放弃了过多功能而难以被采用。目前部署的智能体（如 Claude、Codex）依赖用户中介沙箱与自动模式沙箱的组合作为主要防御手段。在用户中介沙箱中，用户维护的策略会随时间衰减，反复的权限请求会导致用户疲劳；而自动模式的工具调用分类器既不学习用户特定策略，也并非为抵御对抗性设置而设计。Pincer 是一种在资源层运行的新型防御机制，它与现有的工具调用层防御协同工作。

    arXiv:2610.02569v1 Announce Type: cross  Abstract: Coding agents have become increasingly long-horizon, autonomous, reliant on general-purpose shell and maintain their own persistent memory for self-improvement. While these capabilities have made the agents powerful, they have also made them harder to defend against external adversaries. Defenses that restrict this architecture --- typed tools, information-flow control, or policy prediction engines --- give up too much functionality to be adopted. Agents deployed today (e.g. Claude, Codex) rely on a combination of user-mediated and automode sandboxing as their primary defense. In user-mediated sandboxing, user-maintained policies decay over time and repeated permission requests cause user fatigue, while auto mode's tool-call classifiers learn no user-specific policy and are not meant to defend against adversarial setups. Pincer is a new defense that operates at the resource layer and works alongside existing defenses at the tool-call l
    
[^194]: 通过多元化偏好优化缓解社交谄媚行为

    Mitigating Social Sycophancy via Pluralistic Preference Optimization

    [https://arxiv.org/abs/2610.02568](https://arxiv.org/abs/2610.02568)

    该论文提出多元化偏好优化方法，通过让语言模型在给出个人建议时考虑受影响的其他利益相关者的视角，而非过度迎合用户，从而缓解语言模型在社交场景中的谄媚问题。

    

    个人建议（包括人际关系建议）如今已成为生成式AI最常见的用途之一。但语言模型表现出谄媚性（sycophancy）：它们对用户表示肯定的频率远高于人类，这可能使人变得过度自信，并在冲突发生后更不愿意去修复人际关系。以往关于缓解谄媚性的研究主要聚焦于事实性场景，即可以将回答与标准答案进行核对的场景；而针对社交谄媚性（例如个人建议这类没有标准答案的场景）的缓解方法则依赖于简单的提示工程和后训练方法，效果有限。我们的洞察是，社交谄媚性产生的原因之一是语言模型过度以用户为中心，未能考虑受用户行为影响的其他利益相关者的视角。为解决这一问题，我们提出了多元化偏好优化：给定描述人际冲突的输入，语言模型识别并……

    arXiv:2610.02568v1 Announce Type: new  Abstract: Personal advice, including relationship advice, now ranks among the most common uses of generative AI. But language models (LMs) exhibit sycophancy: they affirm users much more often than humans do, which can make people overconfident and less willing to repair their relationships after a conflict. Prior work on mitigating sycophancy has focused on factual settings where a response can be checked against a ground truth answer, while mitigations for social sycophancy (e.g., personal advice, where there is no ground truth) have relied on simple prompting and post-training methods with limited effectiveness. Our insight is that social sycophancy occurs in part because LMs overly center on the user and fail to consider the perspectives of other stakeholders impacted by the user's behavior. To address this problem we propose Pluralistic Preference Optimization (PlurPO): given inputs describing interpersonal conflicts, the LM identifies and si
    
[^195]: DAGS：面向时间稳定生成式渲染的冻结图像DiT解耦外观与几何引导

    DAGS: Disentangled Appearance-and-Geometry Steering of a Frozen Image DiT for Temporally Stabilized Generative Rendering

    [https://arxiv.org/abs/2610.02567](https://arxiv.org/abs/2610.02567)

    DAGS提出了一种轻量级、无需注意力机制的外观与几何解耦条件注入方案，将条件特征作为逐层残差注入冻结的图像DiT，并辅以循环光照稳定器和免训练时序引导，实现了高保真、高忠实度且时间稳定的生成式渲染。

    

    扩散Transformer（DiT）能够基于文本和图像条件生成高保真图像，但其输出存在较大方差，且对目标内容的忠实度在很大程度上取决于条件的提供方式。我们提出了DAGS，这是一种轻量级、无需注意力机制的外观与几何解耦条件化方案，可引导冻结的图像DiT生成高保真、高忠实度且可独立控制的渲染结果。两个小型卷积编码器每帧仅计算一次条件特征，并将其作为学习到的逐层逐元素残差注入图像token中，从而避免了通过注意力机制堆叠条件所带来的二次方计算开销。由于控制与时序处理均位于冻结主干网络之外，我们得以保留其庞大的预训练先验知识，并消除了主干网络过拟合的风险。我们进一步引入了一个小型循环光照稳定器和一个免训练的时序引导项，该引导项与我们的条件化方案相结合……

    arXiv:2610.02567v1 Announce Type: cross  Abstract: Diffusion transformers (DiTs) generate high-fidelity images from text and image conditions, but their outputs carry large variance and their faithfulness to a desired target depends heavily on how the condition is supplied. We present DAGS, a lightweight, attention-free, disentangled appearance and geometry conditioning scheme that steers a frozen image DiT to produce high-fidelity, highly faithful, and independently controllable renders. Two small convolutional encoders compute conditioning features once per frame and inject them as a learned, per-layer, element-wise residual into the image tokens, avoiding the quadratic cost of stacking conditions through attention. Because control and temporal handling live outside the frozen backbone, we retain its vast pretrained prior and eliminate backbone-overfitting risk. We further add a small recurrent lighting stabilizer and a training-free temporal guidance term that, coupled with our cond
    
[^196]: OpenGameEval：在有状态游戏引擎中对智能体编程与探索进行基准测试

    OpenGameEval: Benchmarking Agentic Programming and Exploration in a Stateful Game Engine

    [https://arxiv.org/abs/2610.02563](https://arxiv.org/abs/2610.02563)

    OpenGameEval是一个在Roblox Studio有状态游戏引擎中评估智能体游戏开发能力的基准框架，其核心创新在于通过分离观察工具与编辑工具来直接测量探索行为，实验发现前沿模型虽通过率相近但解决的任务各不相同，且最佳模型单次尝试仅能解决51.7%的任务。

    

    我们提出OpenGameEval，这是一个面向Roblox Studio中智能体游戏开发的基准测试与评估框架。它将语言模型作为智能体运行在可复现的、有状态的游戏引擎会话中，并通过可执行检查对每次运行进行评分，评分既针对被编辑的场景，也针对模拟的游戏会话。大多数智能体编程基准测试虽然要求探索，但只根据最终任务是否成功来评分。OpenGameEval在其八工具动作空间中将观察工具与编辑工具分离开来，从而可以直接测量探索行为。我们在84个人工精选的核心任务上测量了13个前沿模型的通过率与探索行为，每个任务进行16次尝试。这些任务对当前模型而言十分困难：最好的模型单次尝试仅能解决51.7%的任务，五次尝试全部成功的比例为39.4%，且有六个任务没有任何被测试的模型能够解决。处于前沿的模型通过解决不同的任务达到了相似的通过率：按任务所需的工作类型对任务进行划分……

    arXiv:2610.02563v1 Announce Type: cross  Abstract: We present OpenGameEval, a benchmark and evaluation framework for agentic game development inside Roblox Studio. It runs language models as agents in reproducible, stateful game-engine sessions and scores each run with executable checks, both on the edited scene and in a simulated play session. Most agentic coding benchmarks require exploration but score only final task success. OpenGameEval separates observation tools from editing tools in its eight-tool action space, so exploration can be measured directly. We measure the pass rates and exploration behavior of 13 frontier models on 84 human-curated core tasks, with 16 attempts per task.   The tasks are hard for current models. The best model solves 51.7% of tasks on a single attempt and 39.4% five times out of five, and no tested model solves six of the tasks. Models at the frontier reach similar pass rates by solving different tasks: splitting tasks by the kind of work they require 
    
[^197]: 如何进行一场敏锐的辩论：一种面向AI辩论的实例最优协议

    How to Have a Sensitive Debate: An Instance-Optimal Protocol for AI Debate

    [https://arxiv.org/abs/2610.02557](https://arxiv.org/abs/2610.02557)

    本文针对AI辩论设计了一种新的实例最优协议，对于具有足够稳定子问题分解的问题，在有限监督下比现有最佳协议提供更强的正确性保证。

    

    随着强大的AI系统在一系列高认知要求的任务上达到甚至超越人类专家的能力，对这些系统进行准确监督的问题变得日益紧迫。一种有前景的方法是AI辩论，它试图利用两个强大AI之间的辩论，将复杂问题分解为更简单、可以直接判断的论断。关于辩论的理论工作已经用计算复杂性理论的语言将这一直觉形式化，其目标是设计辩论协议（即辩论游戏的规则），以便在有限监督下为复杂问题解的判断提供严格的正确性保证。具体而言，目前最优的协议已被证明适用于所有具有足够稳定子问题分解的问题。在本文中，我们为同一类问题设计了一种新协议，该协议在先前工作的基础上进行了改进。

    arXiv:2610.02557v1 Announce Type: new  Abstract: As powerful AI systems reach and sometimes surpass the abilities of human experts across a range of cognitively demanding tasks, the problem of accurate oversight and supervision of these systems has become increasingly urgent. One promising approach is AI debate, which seeks to leverage a debate between two powerful AIs to break complex questions down into simpler claims that can be easily judged directly. Theoretical work on debate has formalized this intuition in the language of computational complexity theory, where the goal is to design protocols (i.e., rules of the debate game) that provide rigorous guarantees on correctness for judging solutions to complex problems with limited supervision. Specifically, the current best protocol has been shown to work for all problems that have sufficiently stable decompositions into subproblems. In this paper, we design a new protocol for this same class of problems that improves on the prior wo
    
[^198]: 失去同步，即失去视野：针对工业物联网入侵检测的幻影状态攻击

    Out of Sync, Out of Sight: Phantom State Attacks against IIoT Intrusion Detection

    [https://arxiv.org/abs/2610.02552](https://arxiv.org/abs/2610.02552)

    本文提出幻影状态攻击（PSA），在被动、零查询的威胁模型下，通过利用IDS重建运行状态时对时间同步的依赖性，操纵入侵检测系统对工业物联网系统状态的观测视图。

    

    基于机器学习的入侵检测系统（IDS）对于保障工业物联网（IIoT）环境的安全至关重要。针对此类系统的对抗性研究大多是通过扰动特征向量或产生该向量的流量来实施攻击，并依赖于梯度访问、重复的模型查询或对良性流量的学习模型。另有一小部分研究工作在不查询检测器的情况下重塑数据包时序，但会使恶意流量模仿已学习到的良性时序模型。在上述所有方法中，工业监控流程中的一个假设很少受到关注：时间同步。IDS通过将遥测数据聚合到滑动或滚动窗口中来重建系统运行状态，因此其观测视图不仅取决于观察到了什么，还取决于每次观测相对于窗口边界的时间位置。我们提出了幻影状态攻击（PSA），该攻击在被动、零查询的威胁模型下利用了这种依赖性……

    arXiv:2610.02552v1 Announce Type: cross  Abstract: Machine learning-based intrusion detection systems (IDS) are critical for securing Industrial Internet of Things (IIoT) environments. Most adversarial research against them perturbs the feature vector or the traffic that produces it, and depends on gradient access, repeated model queries, or a learned model of benign traffic. A smaller line of work reshapes packet timing without querying the detector, but makes malicious traffic mimic a learned model of benign timing. Across these approaches, one assumption of industrial monitoring pipelines has received little attention: temporal synchronization. An IDS reconstructs operational state by aggregating telemetry into sliding or tumbling windows, so its view depends not only on what is observed but on when each observation falls relative to a window boundary.   We introduce the Phantom State Attack (PSA), which exploits that dependence under a passive, zero-query threat model. Rather than 
    
[^199]: 如何训练你的世界模型：基于语言模型的世界建模中微调与RAG的对比

    How To Train Your World Model: Fine-tuning vs RAG for LM-based World Modeling

    [https://arxiv.org/abs/2610.02542](https://arxiv.org/abs/2610.02542)

    该研究系统评估了基于语言模型的世界建模中微调与RAG两种范式的表现，发现微调方法通常优于RAG（在20个设置中的15个获得更高奖励），但RAG方法更具数据效率。

    

    世界模型（WM）模拟环境的状态转移动态，使智能体能够对其行动的后果进行规划。在基于文本的环境中，微调语言模型（LM）使其充当世界模型已成为主流范式。然而，尽管检索增强生成（RAG）等非参数化方法取得了广泛成功，检索技术在基于语言模型的世界建模中的应用仍未得到充分探索。我们在五个涵盖具身智能、网页导航和社交场景的多样化环境中进行了系统性评估，比较了基于语言模型的世界建模中微调方法与RAG方法。我们的研究表明，微调方法往往优于RAG，微调后的世界模型使智能体在20个设置中的15个中获得了更高的奖励。虽然两种构建范式都能从更多样、更充分的探索中受益，但基于RAG的方法被证明更具数据效率，而微调方法则不成比例地从中获益更多。

    arXiv:2610.02542v1 Announce Type: new  Abstract: World models (WMs) simulate the transition dynamics of environments, enabling agents to plan over the consequences of their actions. In text-based environments, fine-tuning a Language Model (LM) to serve as a WM has emerged as a dominant paradigm. However, despite the widespread success of non-parametric approaches such as Retrieval Augmented Generation (RAG), retrieval for LM-based world modelling remains underexplored. We conduct a systematic evaluation across five diverse environments spanning embodied, web navigation and social settings, comparing fine-tuning and RAG-based approaches for LM-based world modelling. Our study reveals that fine-tuning often outperforms RAG, with fine-tuned WMs enabling agents to obtain higher rewards on 15/20 settings. While both construction paradigms benefit from additional and more diverse exploration, RAG-based approaches prove more data-efficient, and fine-tuning approaches disproportionately benefi
    
[^200]: CriticHack：在机器人策略优化下评估视觉奖励

    CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization

    [https://arxiv.org/abs/2610.02527](https://arxiv.org/abs/2610.02527)

    该论文揭示，用学习型视觉奖励模型优化机器人策略时，奖励分数与任务成功率可能同时上升、看似健康，但实际上会显著放大“作用于错误物体”的隐蔽失败，而这一现象在使用模拟器真实任务完成信号训练时并不会出现。

    

    学习得到的视觉奖励模型越来越多地被用于优化机器人策略，然而奖励模型可能会对作用于错误物体的执行给出与真正完成任务同样高的评分。我们证明，针对这样的奖励进行优化，可能在奖励值与任务成功率双双上升的同时放大这类“错误物体”失败，使得从业者通常会监测的信号看起来一切正常。我们在一个抽屉任务上，针对 Robometer 对扩散策略去噪器的每一个参数进行微调。从一个没有任何奖励先验的监督策略出发，五次训练运行在 512 个评估种子上将任务成功率提高了 10.2 个百分点，同时使错误物体失败增加了 10.9 个百分点；而使用模拟器任务完成信号训练的五次运行则在提高成功率的同时没有放大错误物体失败（差异为 9.2 个百分点，95% 置信区间为 5.6 至 13.0）。这种放大现象在一个先前已针对学习奖励优化过的策略上同样会出现，在该策略原生的……（摘要在此处截断）

    arXiv:2610.02527v1 Announce Type: cross  Abstract: Learned visual reward models are increasingly used to optimize robot policies, yet a reward model can score an execution that acts on the wrong object as highly as one that completes the task. We show that optimizing such a reward can amplify these wrong-object failures while reward and task success both rise, so the signals a practitioner would normally monitor look healthy. We fine-tune every denoiser parameter of a diffusion policy against Robometer on a drawer task. Starting from a supervised policy with no prior reward exposure, five training runs raise task success by 10.2 percentage points and wrong-object failures by 10.9 points on 512 evaluation seeds, whereas five runs trained on the simulator's task-completion signal raise success without amplifying wrong-object failures (difference 9.2 points, 95% CI 5.6 to 13.0). The amplification recurs from a policy previously optimized against learned rewards, under the policy's native 
    
[^201]: 学习下一步调查什么：面向长程研究代理的元推理

    Learning What to Investigate Next: Meta-Reasoning for Long-Horizon Research Agents

    [https://arxiv.org/abs/2610.02525](https://arxiv.org/abs/2610.02525)

    MIRA 提出了一种将研究资源分配与具体执行相分离的分层元推理架构，无需策略训练即可显著提升长程研究代理在定理证明和开放式神经架构研究中的推理能力与算力分配效率。

    

    长程研究代理必须在证据不断累积的过程中，同时决定如何调查以及下一步调查什么。这类决策很难学习，因为它们在漫长的执行轨迹中非常稀疏，而且其后果可能在多次调查之后才逐渐显现。我们提出了面向迭代研究代理的元推理架构（MIRA），这是一种将研究资源分配与执行相分离的分层架构。外循环的元推理器从一个持久的研究记录中整理上下文，然后为下一次调查撰写工作指令，或者结束整个任务回合。一个全新的内循环执行器负责执行每份工作指令，使得执行成为元推理动作之间转换过程的一部分。在无需策略训练的情况下，MIRA 在定理证明和开放式神经架构研究中提升了长程推理能力，并能更有效地分配额外算力。其决策边界还为功劳分配提供了天然的单元。

    arXiv:2610.02525v1 Announce Type: new  Abstract: Long-horizon research agents must decide both how to investigate and what to investigate next as evidence accumulates. This is hard to learn because such decisions are sparse in long execution traces, and their consequences may emerge several investigations later. We introduce Meta-reasoning for Iterative Research Agents (MIRA), a hierarchical architecture separating research allocation from execution. An outer-loop meta-reasoner curates context from a persistent research record, then writes a work order for the next investigation or ends the episode. A fresh inner-loop executor carries out each work order, making execution part of the transition between meta-reasoning actions. Without policy training, MIRA improves long-horizon inference and allocates additional compute more effectively in theorem proving and open-ended neural-architecture research. Its decision boundaries also provide natural units for credit assignment. At each bounda
    
[^202]: 基于假设引导的程序精化方法发现认知算法

    Hypothesis-guided discovery of cognitive algorithms via program refinement

    [https://arxiv.org/abs/2610.02523](https://arxiv.org/abs/2610.02523)

    该论文提出了一种结合人类专家知识与大语言模型的混合系统，将认知算法发现表述为程序精化问题，让LLM智能体在研究者设定的约束下迭代修正以概率程序表达的认知模型，从而兼顾可解释性、人类专业知识与可扩展性。

    

    从行为数据中构建算法推理的认知模型是认知科学中的一个核心问题，这对现有方法构成了挑战。传统的认知建模方法具有可解释性并能够利用人类专业知识，但缺乏灵活性和可扩展性。新兴的利用大型语言模型（LLM）从零开始生成认知模型的技术具有可扩展性和灵活性，但缺乏人类专业知识的参与，且大多只应用于比算法恢复更简单的任务。我们提出了一个混合系统，将认知算法的发现视为一个程序精化问题。人类创建的认知模型被表示为概率程序，并提供给一个由LLM智能体组成的系统，该系统的任务是：识别模型与行为之间的不匹配之处；在研究者指定的约束范围内提出代码级修改；并验证结构保真度。修改会传播到概率推断中……

    arXiv:2610.02523v1 Announce Type: new  Abstract: Developing cognitive models of algorithmic reasoning from behavioral data is a central problem in cognitive science that challenges current methods. Traditional approaches to cognitive modeling are interpretable and benefit from human expertise, but lack flexibility and scalability. Emerging techniques using large language models (LLMs) for de novo generation of cognitive models are scalable and flexible, but lack a role for human expertise and have mostly been applied to simpler tasks than algorithm recovery. We propose a hybrid system that treats discovery of cognitive algorithms as a program refinement problem. Human-created cognitive models are expressed as probabilistic programs and provided to a system of LLM agents with a mandate to: identify mismatches between model and behavior; propose code-level modifications within researcher-specified constraints; and verify structural fidelity. Revisions propagate to a probabilistic inferen
    
[^203]: 逐步约束下受限马尔可夫决策过程（CMDP）的实例相关遗憾

    Instance-Dependent Regret for CMDPs with Step-Wise Constraints

    [https://arxiv.org/abs/2610.02520](https://arxiv.org/abs/2610.02520)

    本文提出了安全方差自适应探索算法（SVAE），通过学习候选安全子图并在其中进行方差自适应的乐观规划，在具有逐步安全约束的情景式CMDP中首次实现了依赖问题实例（方差感知）的累积遗憾界。

    

    我们研究了具有逐步安全约束的情景式表格型受限马尔可夫决策过程（CMDP）中的在线学习问题。在这种设定下，安全约束会诱导出一个安全子图，该子图刻画了可行策略下累积奖励的方差，进而决定了学习的难度。然而，要利用这一结构，需要在控制约束违反的同时学习哪些动作是安全的。我们提出了安全方差自适应探索算法，这是一种高效算法，它能够学习候选安全子图，并在其中执行方差自适应的乐观规划。以高概率，SVAE在 $K$ 个情景下实现了阶为 $\widetilde{\mathcal{O}}(\sqrt{SAH\min\{\mathbb{V}_\Sigma,K\mathrm{Var}^{\star}\}}+S\sqrt{AH^3\min\{K,\mathcal{C}\}}+S^2AH^2)$ 的累积遗憾，其中 $H$ 是单个情景的时域长度，$S$ 和 $A$ 分别表示状态数和动作数。这里，$\mathrm{Var}^{\star}$ 是……（原文摘要在此处截断）

    arXiv:2610.02520v1 Announce Type: cross  Abstract: We study online learning in episodic tabular constrained Markov decision processes with step-wise safety constraints. In such a setting, the constraints induce a safe subgraph that shapes the variance of cumulative rewards under feasible policies and, consequently, the difficulty of learning. Exploiting this structure, however, requires learning which actions are safe while controlling constraint violations. We propose Safe Variance-Adaptive Exploration (SVAE), an efficient algorithm that learns candidate safe subgraphs and performs variance-adaptive optimistic planning within them. With high probability, SVAE achieves cumulative regret of order $\widetilde{\mathcal{O}}(\sqrt{SAH\min\{\mathbb{V}_\Sigma,K\mathrm{Var}^{\star}\}}+S\sqrt{AH^3\min\{K,\mathcal{C}\}}+S^2AH^2)$ over $K$ episodes, where $H$ is the horizon of a single episode, while $S$ and $A$ are the numbers of states and actions, respectively. Here, $\mathrm{Var}^{\star}$ is 
    
[^204]: 面向高效LLM任务路由的学生引导教师蒸馏：与Jev式System-1分类器的定位对比

    Student-Guided Teacher Distillation for Efficient LLM Task Routing: Positioning Against Jev-Style System-1 Classifiers

    [https://arxiv.org/abs/2610.02516](https://arxiv.org/abs/2610.02516)

    提出一种学生引导的教师蒸馏流水线：紧凑的ModernBERT学生模型单次前向预测完整类别分布并生成top-k候选，更大的DeBERTa-v3零样本NLI教师模型仅对候选重排序，教师标签迭代反哺学生，从而显著降低大规模LLM任务路由的成本。

    

    零样本分类器可用于将用户请求路由到专门的LLM任务，但对每个请求在大型候选集合上进行打分代价高昂：零样本NLI分类器必须对每个标签评估一个前提-假设对，因此成本随分类体系规模线性增长。我们研究了一种针对固定60个LLM任务类别分类体系的学生引导教师蒸馏流水线：一个紧凑的ModernBERT分类器在单次前向传播中预测完整的类别分布并检索出较小的top-k候选集，然后由一个更大的DeBERTa-v3零样本NLI分类器仅对这些候选进行重排序，而非对全部60个标签逐一评估；由此产生的教师标签会迭代地改进学生模型，使学生模型在下一轮中生成更精准的候选。与极端多标签分类中常用的通用嵌入检索或基于聚类生成的短名单不同，我们的候选生成器是在目标分类体系上进行端到端训练的，并且是由同一模型提供服务……

    arXiv:2610.02516v1 Announce Type: cross  Abstract: Zero-shot classifiers are useful for routing user requests to specialized LLM tasks, but scoring every request against a large candidate set is expensive: a zero-shot NLI classifier must evaluate one premise-hypothesis pair per label, so cost scales linearly with taxonomy size. We study a student-guided teacher distillation pipeline for a fixed taxonomy of 60 LLM task categories: a compact ModernBERT classifier predicts the full category distribution in one forward pass and retrieves a small top-k candidate set, and a larger DeBERTa-v3 zero-shot NLI classifier reranks only those candidates rather than all 60 labels; the resulting teacher labels iteratively improve the student, which produces sharper candidates for the next round. Unlike generic embedding retrieval or clustering-derived shortlists used in extreme multi-label classification, our candidate generator is trained end-to-end on the target taxonomy and is the same model servin
    
[^205]: IGNITE 托卡马克世界模型架构

    IGNITE Tokamak World Model Architecture

    [https://arxiv.org/abs/2610.02515](https://arxiv.org/abs/2610.02515)

    IGNITE是首个基于DIII-D十年实验数据自监督训练的聚变等离子体生成式世界基础模型，可通过执行器轨迹、文本提示或期望实验结果模拟完整的托卡马克放电过程。

    

    我们提出了IGNITE，一个用于聚变等离子体行为模拟的生成式世界基础模型，该模型在DIII-D国家聚变设施超过十年的无标签实验数据上以自监督方式训练而成。IGNITE的核心是一个动力学模型，能够根据给定的一组执行器轨迹来模拟DIII-D等离子体放电。这些轨迹既可以由外部提供，也可以根据文本提示或期望的实验结果即时生成。该模型架构包含多个时空分词器，用于嵌入不同的输入模态，包括时间序列类的时空测量数据、图像序列以及高分辨率光谱图，每一种模态都是在截然不同的时间尺度上采集的。其主干网络由一个自回归动力学模型构成，在给定初始潜在等离子体状态和执行器轨迹的条件下，该模型理论上能够预测无限长度的完整DIII-D放电过程。

    arXiv:2610.02515v1 Announce Type: cross  Abstract: We introduce IGNITE, a generative world foundation model for fusion plasma behavior simulation trained in a self-supervised manner from over a decade of unlabeled experimental data at the DIII-D National Fusion Facility. The core of IGNITE is a dynamics model that can simulate DIII-D discharges from a given set of actuator trajectories. These trajectories can be supplied or generated on-the-fly from a textual prompt or from desired experimental outcomes. The model architecture consists of several spatio-temporal tokenizers that embed the different input modalities, including time-series like spatio-temporal measurement data, image sequences, and high-resolution spectrograms, each of which collected at vastly different time scales. The backbone is composed of an auto-regressive dynamics model that has the capacity to predict entire DIII-D discharges given initial latent plasma states and actuator trajectories over a theoretical infinite
    
[^206]: 从局部片段到全局地图：基于大语言模型学习向量化地图聚合

    From Fragments to Global Maps: Learning Vectorized Map Aggregation with Large Language Models

    [https://arxiv.org/abs/2610.02513](https://arxiv.org/abs/2610.02513)

    该论文提出MapMergeLLM，将向量化高精地图聚合任务转化为大语言模型的条件序列生成问题，直接从序列化的局部地图片段预测全局地图折线，从而摆脱了传统方法对手工规则、固定阈值和特定检测器调优的依赖。

    

    大规模向量化高精地图提供了对自动驾驶中感知、定位和规划至关重要的结构化道路信息。构建此类地图需要将沿车辆轨迹采集的含噪、碎片化且相互重叠的局部预测聚合为一张连贯的全局地图。现有的聚合方法通常依赖手工设计的规则进行片段关联与精化。然而，固定的阈值集合无法有效应对道路结构和预测误差的变化，往往需要针对特定检测器的调优或人工调整。为解决这一局限，我们提出了MapMergeLLM，这是一个数据驱动的框架，它将向量化地图聚合建模为基于大语言模型的条件序列生成任务。给定序列化的局部向量化地图，我们的模型直接预测聚合后的全局地图折线。为减少对特定上游检测器的依赖，我们训练……（摘要在此处被截断）

    arXiv:2610.02513v1 Announce Type: cross  Abstract: Large-scale vectorized HD maps provide structured road information that is essential for perception, localization, and planning in autonomous driving. Constructing such maps requires aggregating noisy, fragmented, and overlapping local predictions collected along a vehicle trajectory into a coherent global map. Existing aggregation methods typically rely on hand-crafted rules for fragment association and refinement. However, a fixed set of thresholds cannot effectively handle variations in road structures and prediction errors, often requiring detector-specific tuning or manual adjustment. To address this limitation, we propose MapMergeLLM, a data-driven framework that formulates vectorized map aggregation as conditional sequence generation with a large language model. Given serialized local vectorized maps, our model directly predicts the aggregated global map polylines. To reduce dependence on any particular upstream detector, we tra
    
[^207]: 面向商业教育的本地部署多课程RAG辅导系统：校园AI辅导员的软硬件权衡

    On-Premises Multi-Course RAG Tutoring for Business Education: Hardware-Software Trade-offs in a Campus AI Tutor

    [https://arxiv.org/abs/2610.02510](https://arxiv.org/abs/2610.02510)

    本文提出了CourseChat——一个面向本科商业教育的本地部署多课程RAG辅导系统，通过模型对比测试与软硬件权衡评估，证明12B和7B级本地大语言模型能在课程数据不出校门的前提下满足课堂实时响应的速度要求。

    

    基于检索增强生成（RAG）的校园AI辅导员必须使答案立足于指定的课程材料，同时将教科书和学生对话数据保留在机构自身的基础设施内。我们提出了CourseChat，这是一个面向本科商业教育的本地部署、多课程RAG辅导员系统，部署在校园Web网关之后，旨在嵌入Moodle中使用。六个相互隔离的课程班级（每个班级由其各自的课程参考号CRN标识）共享双边缘AI主机，其上运行着FastAPI服务、本地向量数据库以及由Ollama提供服务的本地大语言模型（LLM）。我们报告了两轮生成模型对比测试、一次独立的固定证据来源保真度比较实验，以及对话与测验审计结果。多个较大的模型未能通过课堂场景的速度门槛，但一个12B模型和一个7B替代方案通过了测试。另一个独立的混合专家模型候选方案在某些修正上有所改进，但同时引入了新的事实性错误和连贯性错误。因此我们……

    arXiv:2610.02510v1 Announce Type: new  Abstract: Campus AI tutors based on retrieval-augmented generation (RAG) must ground answers in assigned course materials while keeping textbooks and student dialogue on institutional infrastructure. We present CourseChat, an on-premises, multi-course RAG tutor for undergraduate business education, deployed behind a campus web gateway and intended for use embedded in Moodle. Six isolated course offerings, each keyed by its own course reference number (CRN), share twin-edge AI hosts running a FastAPI service, a local vector database, and a local large language model (LLM) served by Ollama. We report two generation-model bake-off rounds, a separate fixed-evidence source-fidelity comparison, and conversation and quiz audits. Several larger models failed the classroom speed gate, but a 12B model and a 7B alternative passed. A separate mixture-of-experts candidate improved some corrections while introducing new factual and continuity errors. We therefo
    
[^208]: 基于渐进式视觉规划的世界动作建模

    World Action Modeling with Progressive Visual Planning

    [https://arxiv.org/abs/2610.02508](https://arxiv.org/abs/2610.02508)

    ProWAM通过联合预测动作和有序的稀疏视觉子目标序列实现渐进式视觉规划，解决了世界动作模型长时程预测效率低下的问题，且子目标预测可从大规模无动作视频中学习，使视觉规划与动作策略自然解耦。

    

    世界动作模型（WAMs）已成为机器人控制的一种有前景的范式，它能够根据初始观察和指令联合预测未来的视觉动态和动作。然而，现有的世界动作模型在长时程预测方面存在困难，因为生成密集的视频展开序列效率极低。近期一些世界动作模型通过预测单个未来帧而不生成完整视频来解决这一问题，但这种方法忽略了如何逐步向目标推进。我们提出了ProWAM，一种渐进式世界动作模型，它联合预测动作和有序的稀疏视觉子目标序列，在整个任务执行过程中提供显式的视觉指导来锚定动作生成。这种设计可以自然地扩展，因为子目标预测可以从大规模无动作标注的视频中学习，使视频主干网络能够从动作策略中分担复杂的视觉规划任务。为了高效地生成动作，ProWAM执行单个视频主干（原文摘要在此处截断）

    arXiv:2610.02508v1 Announce Type: new  Abstract: World action models (WAMs) have emerged as a promising paradigm for robotic control by jointly predicting future visual dynamics and actions from an initial observation and instruction. However, existing WAMs struggle with long-horizon prediction, as generating dense video rollouts is highly inefficient. Some recent WAMs address this by predicting a single future frame without generating the full video, but this approach neglects how to progress toward the goal. We present ProWAM, a progressive world action model that jointly predicts actions and an ordered sequence of sparse visual sub-goals, providing explicit visual guidance to anchor action generation throughout task execution. This design scales naturally, as sub-goal prediction can be learned from large-scale action-free videos, allowing the video backbone to offload complex visual planning from the action policy. For efficient action generation, ProWAM executes a single video-back
    
[^209]: 多保真度策略梯度稳定数据稀缺的强化学习

    Multi-Fidelity Policy Gradients Stabilize Data-Scarce Reinforcement Learning

    [https://arxiv.org/abs/2610.02505](https://arxiv.org/abs/2610.02505)

    本文将多保真度策略梯度（MFPG）框架从REINFORCE扩展到现代演员-评论家算法（如PPO），在GPU并行仿真和真实机器人上利用低保真度数据构建控制变量，以在无偏差的前提下降低梯度方差，从而稳定数据稀缺场景下的强化学习。

    

    同策略强化学习（RL）中的策略梯度方法，在昂贵且稀缺的目标域数据产生噪声较大的梯度估计时可能会变得不稳定。我们通过利用大量、廉价但有偏差的低保真度（LF）数据（例如来自简化模拟器的数据）来补充有限的高保真度（HF）目标域数据，从而应对这一挑战。大多数现有方法直接基于LF数据优化有偏差的目标函数。相比之下，最近提出的多种保真度策略梯度（MFPG）框架仅将LF数据用于构建控制变量，在不使策略梯度估计器产生偏差的情况下降低方差并提高HF数据的利用效率。然而，已发表的关于MFPG的工作仅限于在小规模仿真任务上使用REINFORCE。我们将MFPG发展到GPU并行仿真和物理机器人上的现代演员-评论家（actor-critic）学习中。我们的分析和实验表明，对近端策略优化（PPO）的简单扩展可能会失去……

    arXiv:2610.02505v1 Announce Type: cross  Abstract: Policy gradient methods for on-policy reinforcement learning (RL) can become unstable when expensive, scarce target-domain data yield noisy gradient estimates. We address this challenge by complementing limited high-fidelity (HF) target-domain data with abundant, cheap, but biased low-fidelity (LF) data, e.g., from a simplified simulator. Most existing methods directly optimize biased objectives based on LF data. In contrast, the recently introduced multi-fidelity policy gradient (MFPG) framework uses LF data solely to construct a control variate that reduces variance and improves HF data efficiency without biasing the policy gradient estimator. However, published work on MFPG is limited to REINFORCE on small-scale simulation tasks. We develop MFPG for modern actor-critic learning in GPU-parallel simulation and on a physical robot. Our analysis and experiments show that naive extensions to proximal policy optimization (PPO) can lose cr
    
[^210]: HXAI：分布式能源系统中的分层隐私保护可解释人工智能

    HXAI: Hierarchical Privacy-Preserving Explainable AI in Distributed Energy Systems

    [https://arxiv.org/abs/2610.02504](https://arxiv.org/abs/2610.02504)

    提出HXAI分层隐私保护框架，通过本地模型在私有环境中生成细粒度解释、区域模型聚合这些解释，在保护用户隐私的同时实现电网级需求管理的可解释分析。

    

    由于可再生能源发电的固有间歇性和电力消耗的随机性，平衡电力供需变得越来越困难。电网运营商需要对家庭能源消耗进行细粒度的、与决策相关的洞察，以管理峰值负荷并设计响应式电价，但这种层面的透明度提升引发了严重的隐私问题。传统的可解释人工智能（XAI）方法可能会泄露敏感信息，而标准的隐私技术通常会降低解释的实用性。为解决这一问题，我们提出了HXAI，这是一个分层框架，能够在保护隐私的同时为电网级需求管理提供合理的可解释分析。HXAI由两个主要组件构成：（1）本地模型，在安全的私有环境中生成细粒度的解释；（2）区域模型，聚合这些解释以支持电网级分析，同时……

    arXiv:2610.02504v1 Announce Type: new  Abstract: Balancing electricity demand and supply is increasingly difficult due to the inherent intermittency of renewable power generation and the stochastic power consumption. Grid operators require fine-grained, decision-relevant insights into household energy consumption to manage peak loads and design responsive tariffs, but increased transparency at this level raises significant privacy concerns. Traditional methods for explainable AI (XAI) can reveal sensitive information, while standard privacy techniques often reduce the usefulness of explanations. To address this issue, we introduce HXAI, a hierarchical framework that preserves privacy while enabling reasonable explainable analysis for grid-level demand management. HXAI consists of two main components: (1) a local model that generates fine-grained explanations within a secure, private environment, and (2) a zonal model that aggregates these explanations to support grid-level analysis whi
    
[^211]: 复合AI系统可靠性：基于150起生产事故的故障分类学与韧性模式目录

    Compound AI System Reliability: A Failure Taxonomy and Resilience Pattern Catalog from 150 Production Incidents

    [https://arxiv.org/abs/2610.02503](https://arxiv.org/abs/2610.02503)

    本文通过分析150起生产事故，构建了包含23种故障模式、分五大类别的复合AI系统故障分类学，并提出经故障注入实验验证有效性的韧性模式（如断路器减少89%级联传播、质量门控捕获73%静默退化）。

    

    可靠且安全地部署复合AI系统，需要理解出现在组件边界而非单个模型内部的故障模式。级联错误会在组件边界之间传播，静默的质量退化会逃避标准监控，而协调失败会导致由各自正确的部件产生错误的集体行为。我们分析了来自开源复合AI项目和匿名化企业部署的150份生产事故报告，构建了一个包含23种故障模式的分类体系，分为五大类别：检索故障、生成故障、工具故障、编排故障和集成故障。针对每个类别，我们提出了相应的韧性模式，并通过受控故障注入实验测量了其有效性。断路器可将级联传播减少89%，输出质量门控能在影响用户之前捕获73%的静默退化，组件隔离可缩小故障爆炸半径。

    arXiv:2610.02503v1 Announce Type: cross  Abstract: Deploying compound AI systems reliably and safely requires understanding failure modes that emerge at component boundaries, not within individual models. Cascading errors propagate across component boundaries, silent quality degradation evades standard monitoring, and coordination failures yield incorrect collective behavior from individually correct parts. We analyze 150 production incident reports from open-source compound AI projects and anonymized enterprise deployments to construct a taxonomy of 23 failure modes organized into five categories: retrieval failures, generation failures, tool failures, orchestration failures, and integration failures. For each category, we propose resilience patterns with measured effectiveness from controlled fault injection experiments. Circuit breakers reduce cascade propagation by 89%, output quality gates catch 73% of silent degradation before user impact, and component isolation reduces blast ra
    
[^212]: “我只是假设它能翻译”：以缩写为用例考察医护人员对机器翻译风险的认知

    "I just assumed that it would translate": examining MT risk awareness among healthcare staff with abbreviations as a use case

    [https://arxiv.org/abs/2610.02496](https://arxiv.org/abs/2610.02496)

    本研究以医学缩写为用例，考察英国医护人员在使用机器翻译（尤其是翻译患者医疗记录）时对潜在风险的认知，揭示对机器翻译的盲目依赖可能危及患者安全。

    

    arXiv:2610.02496v1 公告类型：新 摘要：在英国，公共医疗机构的员工报告称，他们求助于机器翻译（MT）——主要是谷歌翻译（GT）——来跨越语言障碍与患者进行沟通。尽管此举意在支持其照护职责，但在这种情境下对机器翻译缺乏充分了解的依赖，可能对患者安全造成严重后果。然而，关于医护人员对高风险机器翻译使用——尤其是对患者医疗记录的使用——可能带来的风险的认知，相关研究仍然有限；现有文献大多聚焦于机器翻译在人际交流场景或面向患者的文档中的使用。此外，医学缩写即使仅在单语环境下也已被充分证实会增加患者风险，其误用和/或误读的后果轻则造成暂时性伤害，重则导致患者死亡。因此，本研究选择医学缩写作为用例，以识别其经机器翻译后可能带来的潜在风险。

    arXiv:2610.02496v1 Announce Type: new  Abstract: In the UK, public healthcare staff report turning to machine translation (MT) - predominantly Google Translate (GT) - to communicate with patients across language barriers. Though intended to support their duty of care, potentially uninformed reliance on MT in such contexts could have serious consequences for patient safety. Research nonetheless remains limited on staff awareness of the possible risks posed by higher-stakes MT use in general and with patient medical records in particular, most existing literature instead examining its use in interpersonal situations or with patient-oriented documentation. Moreover, medical abbreviations are well-documented as increasing patient risk even monolingually, with outcomes from their misuse and/or misinterpretation ranging from temporary harm to the death of the patient. Abbreviations were therefore selected as a use case for identifying the potential risks posed by their translation with MT. C
    
[^213]: 排序正确，尺度有误：审计用于职业AI测量的LLM评判器

    Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement

    [https://arxiv.org/abs/2610.02492](https://arxiv.org/abs/2610.02492)

    该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。

    

    LLM评判器正被越来越多地用于评估AI输出是否满足职场要求，但对回答排序的一致性并不能确立在接受率或职业总体层面的一致性。我们提出了O*NET-BENCH，这是一个基于包含45,796名工人评分的现有调查构建的审计套件，并在4,501个测试评分上评估了来自六个模型系列的33种现有评判器配置。其中25种配置实现了至少0.60的平局感知成对排序准确率，尽管一个经训练拟合的仅基于回答文本的TF-IDF基线几乎与最强评判器表现相当。尽管存在这种排序上的一致性，评判器对回答可接受比例的估计介于3.0%至97.9%之间，而职业匹配的人类工人给出的估计为61.1%。在一个微调模型谱系中，从逐点评分切换到捆绑的少样本/列表式评估协议虽然改善了回答排序，却降低了在任务和职业层面与工人平均评分的一致性；这一反转现象在一个任务子集上得到了复现。

    arXiv:2610.02492v1 Announce Type: new  Abstract: LLM judges are increasingly used to assess whether AI outputs meet workplace requirements, but agreement on response rankings does not establish agreement on acceptance rates or occupational aggregates. We introduce O*NET-BENCH, an audit suite derived from an existing survey of 45,796 worker ratings, and evaluate 33 pre-existing judge configurations across six model families on 4,501 test ratings. Twenty-five configurations achieve tie-aware pair accuracy of at least 0.60, although a train-fitted response-only TF-IDF baseline nearly matches the strongest judge. Despite this ordering agreement, judges estimate that 3.0%-97.9% of responses are acceptable, compared with 61.1% for occupation-matched workers. In one fine-tuned lineage, changing from pointwise scoring to a bundled few-shot/listwise protocol improves response ordering while reducing agreement with worker means at the task and occupation levels; this reversal replicates on a tas
    
[^214]: 一个Token需要多少成本？基于混合代理的每Token充分计算量测量

    What Does a Token Cost? A Mixture-of-Agents Measurement of Sufficient Per-Token Compute

    [https://arxiv.org/abs/2610.02491](https://arxiv.org/abs/2610.02491)

    该论文首次通过混合代理方法测量了单个Token实际所需的充分计算量，发现0.5B小模型即可复现92-95%的Token，而最昂贵的10%的Token占据了64-80%的估计计算量，揭示了计算分配中巨大的浪费与优化空间。

    

    大型语言模型在其生成的每个Token上花费相同的计算量，而不考虑每个Token的生成难度。投机解码和模型路由等方法建立在这一前提之上：即大部分计算是不必要的，然而单个Token实际所需的计算量尚未被测量过。我们通过混合代理的视角来测量它：构建一个由来自三个模型系列、容量递增的十五个语言模型组成的代理面板，其中每个代理都尝试在给定正确前序Token的条件下逐Token地复现参考序列。我们将成功复现该Token的最小代理的推理成本定义为该Token的充分计算量，它构成了该Token实际所需计算量的上界。在三个核心基准测试中，一个0.5B的代理即可复现92-95%的参考Token。在Qwen、OLMo和R1蒸馏模型面板上，计算成本最高的10%的Token占据了估计FLOPs的64-80%。

    arXiv:2610.02491v1 Announce Type: new  Abstract: Large language models spend the same amount of computation on every token they generate, regardless of how difficult each token is to produce. Methods such as speculative decoding and model routing are built on the premise that much of this computation is unnecessary, yet the computation an individual token actually requires has not been measured. We measure it through a Mixture-of-Agents (MoA) lens: a panel of fifteen language models of increasing capacity, drawn from three families, in which every agent attempts to reproduce a reference sequence token by token, conditioned on the correct preceding tokens. We define the inference cost of the smallest agent that succeeds as the token's sufficient compute, which upper-bounds what the token requires. On three core benchmarks, a 0.5B agent reproduces 92--95\% of reference tokens. Across Qwen, OLMo, and R1-distilled panels, the most expensive 10\% account for 64--80\% of estimated FLOPs. On 
    
[^215]: 从检索到类型化决策：基于生物医学句子编码器的校准“系统一”模型

    From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders

    [https://arxiv.org/abs/2610.02486](https://arxiv.org/abs/2610.02486)

    该论文提出SBERT2S1框架，将生物医学检索句子编码器转换为类型化决策模型，并发现检索预训练显著有利于保留检索先验的先验融合残差（PFR）决策头，而对交叉头（C）帮助有限甚至有害。

    

    类型化决策模型能够通过一次前向传播回答关于文本的受模式约束的问题，并返回用于阈值判断的概率。我们探讨为检索任务训练的生物医学句子编码器是否是此类模型的良好起点。我们提出了SBERT2S1，它将Sentence-Transformers编码器转换为双编码器、交叉头（C）和先验融合残差（PFR）决策模型，同时推出了BIODECIDE（一个生物医学类型化决策测试套件）和MEDLINE-S1（从NLM标引中派生的24.3万条训练决策）。在六个父模型-检索器配对上，检索训练改善了含内容选项的零样本匹配。微调之后，其效果取决于决策头的类型：在五组配对和三种训练集规模下，检索训练在15次比较中的10次显著帮助保留检索先验的PFR头，但仅在1次中帮助C头，而在5次中反而损害C头的表现。在两个决策头与五种训练目标的匹配网格实验中，C头优于……（摘要在此处截断）

    arXiv:2610.02486v1 Announce Type: cross  Abstract: Typed decision models answer schema-constrained questions about a text in one forward pass and return probabilities meant to be thresholded. We ask whether biomedical sentence encoders trained for retrieval are good starting points for such models. We present SBERT2S1, which converts Sentence-Transformers encoders into bi-encoder, cross-head (C) and prior-fused residual (PFR) decision models, together with BIODECIDE, a biomedical typed-decision suite, and MEDLINE-S1, 243k training decisions derived from NLM indexing. Across six parent-retriever pairs, retrieval training improves zero-shot matching of content-bearing options. After fine-tuning, its effect depends on the head: across five pairs and three training-set sizes, retrieval training significantly helps PFR, which keeps the retrieval prior, in 10 of 15 comparisons, but helps C in one and hurts it in five. A matched grid of two heads and five training objectives shows that C outp
    
[^216]: MEA：一个奖励驱动的多智能体系统，用于忠实的模型解释

    MEA: A Reward-Driven Multi-Agent System for Faithful Model Explanations

    [https://arxiv.org/abs/2610.02480](https://arxiv.org/abs/2610.02480)

    MEA通过Proposer智能体自动选择和配置解释工具、Actor智能体以忠实性为目标进行端到端奖励优化，完全消除了使用机器学习解释方法的知识壁垒，能够跨表格、文本和视觉模态生成忠实的自然语言模型解释。

    

    近年来，大量机器学习（ML）模型被应用于高风险领域，但对于依据这些模型预测结果采取行动的从业者来说，模型在很大程度上仍然是不透明的。虽然事后解释方法为理解模型行为提供了一个视角，但有效使用这些方法需要大多数领域专家所不具备的专业知识：处理高维输出、选择最佳解释，以及综合来自不同工具的证据。为此，我们提出了MEA，一个完全消除解释知识壁垒的多智能体框架：Proposer（提议者）智能体根据问题和模态选择并配置解释工具，而Actor（执行者）智能体则针对忠实性进行端到端优化，将输出转化为基于表格、文本和视觉等多种模态下模型行为的自然语言解释。此外，我们引入了涵盖特征归因、反事实推理等多种类型的问题（摘要原文在此处截断）。

    arXiv:2610.02480v1 Announce Type: new  Abstract: Recent years have seen the employment of a plethora of machine learning (ML) models in high-stakes domains, but they remain largely opaque to the practitioners who act on their predictions. While post-hoc explanation methods offer a lens into this model behavior, wielding them effectively demands expertise most domain experts lack: navigating high-dimensional outputs, selecting the best explanations, and synthesizing evidence across disparate tools. To this end, we present MEA, a multi-agent framework that removes the explanation knowledge barrier entirely: a Proposer agent selects and configures explanation tools based on the question and modality, while an Actor agent is optimized end-to-end against faithfulness, transforming the outputs into natural language explanations grounded in model behavior across tabular, text, and vision modalities. Further, we introduce diverse question types spanning feature attribution, counterfactual reas
    
[^217]: 热带强化学习

    Tropical Reinforcement Learning

    [https://arxiv.org/abs/2610.02478](https://arxiv.org/abs/2610.02478)

    该论文提出热带强化学习，用取最大值替代概率求和（将代数结构改为热带半环），使状态价值定义为最可能已验证解决方案的对数概率，从而避免强化一个解导致遗忘其他有效解的问题，更契合大语言模型的组合式推理。

    

    针对大型语言模型的强化学习通常最大化期望回报，即把所有成功轨迹的概率相加。然而，这种经典的求和形式只能报告模型策略成功的频率，却无法说明究竟是哪个解决方案真正起了作用；并且由于概率之和为一，强化一个解决方案可能导致模型遗忘另一个从未被证明是错误的解决方案。这使得期望回报并不适合组合式推理——在这类任务中，解决方案必须由模型在多次分散且常常失败的尝试中产生的推理步骤组装而成，而这些步骤很少被同时生成。为解决这一问题，我们提出了热带强化学习，其核心是一次简单的代数变换：不再将备选解决方案的概率相加，而是取它们的最大值，这便得到了热带半环。此时，一个状态的价值就变成了其最可能被验证通过的解决方案的对数概率，以及……（摘要原文在此处截断）

    arXiv:2610.02478v1 Announce Type: new  Abstract: Reinforcement learning for large language models typically maximizes expected return, adding up the probabilities of all successful trajectories. However, the classical sum formulation can only report how often the model policy succeeds, not which solution actually worked, and because probabilities sum to one, reinforcing one solution can make the model forget another that was never shown to be wrong. This makes expected return a poor fit for compositional reasoning, where a solution must be assembled from reasoning steps that the model produces in separate, often failed, attempts but rarely produces together. To address this, we propose Tropical Reinforcement Learning, which rests on a simple change of algebra: instead of adding the probabilities of alternative solutions, we take their maximum, which yields the tropical semiring. The value of a state then becomes the log-probability of its most likely verified solution, together with an
    
[^218]: APDMem：面向查询自适应长期记忆的智能体控制渐进式披露

    APDMem: Agent-Controlled Progressive Disclosure for Query-Adaptive Long-Term Memory

    [https://arxiv.org/abs/2610.02472](https://arxiv.org/abs/2610.02472)

    APDMem提出了一种智能体控制的分层长期记忆架构，将对话历史组织为四个粒度递进的层次并采用渐进式披露检索，从而根据查询复杂度自适应地平衡检索成本与证据保真度。

    

    个性化LLM助手必须从长对话历史中为不同复杂度的查询检索稀疏证据。我们提出APDMem（智能体控制的渐进式披露记忆），这是一种将渐进式披露应用于记忆检索的分层长期记忆架构。APDMem不依赖平面记忆存储或固定检索粒度，而是将对话历史表示为四个逐级细化的层次：主题摘要、个性化关键事实、轮次级证据笔记和原始消息。在推理时，控制器对记忆层次应用渐进式披露：它首先读取高层摘要，仅在需要时才深入查看更细粒度的证据。这形成了自适应的成本-保真度权衡：简单查询可以提前终止，而复杂的时间性、多跳或精确证据查询则会触发更深入的检索。笔记合成器将检索到的证据转换为查询相关的……

    arXiv:2610.02472v1 Announce Type: cross  Abstract: Personalized LLM assistants must recover sparse evidence from long conversation histories across queries of varying complexity. We introduce APDMem (Agent-controlled Progressive Disclosure Memory), a hierarchical long-term memory architecture that applies progressive disclosure to memory retrieval. Rather than relying on a flat memory store or fixed retrieval granularity, APDMem represents conversation history as four progressively detailed layers: thematic summaries, personalized key facts, turn-level evidence notes, and raw messages. At inference time, a controller applies progressive disclosure to the memory hierarchy: it first reads high-level summaries and drills into finer evidence only when needed. This creates an adaptive cost-fidelity trade-off: simple queries can terminate early, while complex temporal, multi-hop, or exact-evidence queries trigger deeper inspection. A note synthesizer converts retrieved evidence into a query-
    
[^219]: SideKernel：一个面向 macOS 上 AI 编程代理的易用 microVM 沙盒

    SideKernel: A Usable microVM Sandbox for AI Coding Agents on macOS

    [https://arxiv.org/abs/2610.02456](https://arxiv.org/abs/2610.02456)

    该论文通过用户调查发现 AI 编程代理沙盒采用率低下的可用性障碍，并据此开发了 SideKernel——一个面向 macOS 上 AI 编程代理的开源、易用的 microVM 沙盒。

    

    AI 编程代理是不可信的系统组件，但它们又需要在所运行的开发者机器上拥有自主权。这一矛盾构成了一个安全问题。沙盒可以提供隔离环境，但对于本地 macOS 开发而言，现有的面向 AI 编程代理的本地开源方案数量稀少且使用繁琐。我进行了一项形成性的在线用户调查，结果表明只有不到 40% 的 AI 编程代理用户在沙盒中运行其代理，并识别出了阻碍 AI 编程代理沙盒普及的主要可用性障碍。基于这些发现，我开发了 SideKernel：一个专为易用性设计的开源、本地化、基于 microVM 的 macOS 沙盒，用于运行 AI 编程代理。为评估 SideKernel，我整理了市场上可用的沙盒列表，并根据五项纳入标准进行筛选，随后对 SideKernel 和满足这些标准的沙盒在 23 项能力维度上进行了对比分析。

    arXiv:2610.02456v1 Announce Type: cross  Abstract: AI coding agents are untrusted system components, yet they require autonomy on the developer machines they run on. This contradiction is a security problem. Sandboxes provide an isolated environment, but for local macOS development, the existing local, open-source options for AI coding agents are few in number and cumbersome to use. I conducted a formative online user survey which indicates that fewer than 40% of AI coding agent users run their agents in a sandbox and identifies the top usability barriers hindering AI coding agent sandbox adoption. These findings are used to develop SideKernel: an open-source, local, microVM-based macOS sandbox for AI coding agents designed for usability. To evaluate SideKernel, I compiled a list of sandboxes available on the market and filtered it against five inclusion criteria. Then I performed a comparative analysis between SideKernel and the sandboxes that satisfy these criteria, across 23 capabil
    
[^220]: FinDialogLens：面向金融聊天室遗漏交易识别的多方对话事件抽取

    FinDialogLens: Event Extraction over Multi-Party Dialogue for Missed-Trade Identification in Financial Chatrooms

    [https://arxiv.org/abs/2610.02455](https://arxiv.org/abs/2610.02455)

    提出FinDialogLens混合LLM流水线，以紧凑的微调分类器作为推理时脚手架，对多方金融聊天对话进行RFQ事件抽取，从而准确识别遗漏交易的最终价格与交易结果，配合GPT-4o分别达到92.1%和94.3%的准确率。

    

    多方金融聊天室对销售与交易专业人士至关重要，但其复杂性使得人工恢复遗漏交易不可行：每个报价请求（RFQ）都是一个事件，其最终价格和交易结果出现在RFQ触发消息（即询价消息）之后的许多条消息中，并与其他参与者并发的RFQ相互交错。我们将该问题建模为多方对话上的事件抽取（EE）任务，并提出FinDialogLens——一种混合式大语言模型（LLM）流水线，其中紧凑的微调分类器充当推理时的脚手架：它们负责检测RFQ触发消息以及价格/交易结果元数据，RFQ级模块对每个事件的RFQ窗口进行切分，交易引擎负责填充论元角色。使用GPT-4o时，FinDialogLens在最终价格和交易结果上分别达到92.1%和94.3%的准确率，优于针对全聊天室的思维链（CoT）提示方法；经过微调的开源LLM仅需3B参数即可达到相当的性能。

    arXiv:2610.02455v1 Announce Type: cross  Abstract: Multi-party financial chatrooms are vital for sales-and-trading professionals, but their complexity makes manual recovery of missed trades infeasible: each Request for Quote (RFQ) is an event whose final price and trade outcome appear many messages after the RFQ-trigger message (the inquiry message), interleaved with concurrent RFQs from other participants. We cast this as event extraction (EE) over multi-party dialogue and present FinDialogLens, a hybrid LLM pipeline in which compact fine-tuned classifiers act as inference-time scaffolds: they detect RFQ-triggers and price/trade outcome metadata, an RFQ-Level Module segments per-event RFQ windows, and a Trade Engine fills argument roles. With GPT-4o, FinDialogLens reaches 92.1% and 94.3% accuracy on final price and trade outcome, respectively, outperforming full-chatroom CoT prompting methods; fine-tuned open-source LLMs with as few as 3B parameters achieve comparable performance with
    
[^221]: 核物理散射实验中用于靶极化优化的强化学习技术

    Reinforcement Learning Techniques for the Optimization of Target Polarization in Nuclear Physics Scattering Experiments

    [https://arxiv.org/abs/2610.02452](https://arxiv.org/abs/2610.02452)

    该研究提出将高斯过程代理模型与强化学习相结合的数据驱动控制框架，利用其校准的不确定性估计来自动优化核物理实验中动态极化靶的微波频率调节，以应对辐射损伤带来的材料特性变化。

    

    核物理实验中动态极化靶的运行依赖于对微波频率的连续调节，以补偿辐射损伤和不断变化的材料特性，这一任务传统上由专家操作员通过手动试错来完成。本工作提出了一种数据驱动的控制框架，将代理建模与强化学习相结合，以优化靶极化。利用APOLLO低温靶系统的运行数据，我们训练并评估了多层感知机（MLP）和高斯过程回归模型，用于预测极化作为微波频率、束流强度和累积辐射剂量的函数。我们证明，基于高斯过程的模型能够提供经校准的不确定性估计，并能可靠地识别训练分布之外的区域，而多层感知机对分布偏移的敏感性则较为有限。为实现跨多个（ regimes？原文在此处被截断，后续内容缺失）的学习与控制……（注：原摘要不完整）

    arXiv:2610.02452v1 Announce Type: new  Abstract: The operation of dynamically polarized targets in nuclear physics experiments relies on continuous tuning of the microwave frequency to compensate for radiation damage and evolving material properties, a task that is traditionally performed through manual trial-and-error by expert operators. This work presents a data-driven control framework that combines surrogate modeling with reinforcement learning to optimize the target polarization. Using operational data from the APOLLO cryogenic target system, we train and evaluate multilayer perceptron and Gaussian process regression models to predict polarization as a function of microwave frequency, beam current, and accumulated radiation dose. We show that Gaussian process-based models provide calibrated uncertainty estimates and reliably identify regions outside the training distribution, while MLPs exhibit limited sensitivity to distributional shift. To enable learning and control across mul
    
[^222]: 基于逐定理符号验证器的反例生成：模仿何时有害而强化学习何时修复

    Counterexample Generation via Per-Theorem Symbolic Verifiers: When Imitation Hurts and Reinforcement Repairs

    [https://arxiv.org/abs/2610.02444](https://arxiv.org/abs/2610.02444)

    该论文发布SymCE数据集（包含4,707个错误数学猜想及其可执行验证器），发现仅用反例做监督微调会陷入“模仿陷阱”、使真定理识别率从0.27崩溃至0.00，而基于验证器稀疏奖励的强化学习（RLVR）不仅能修复这一退化，还能超越基线达到0.66。

    

    大型语言模型往往能够正向证明一个定理，却无法反驳一个密切相关的错误命题——这种“证伪鸿沟”是监督微调无法弥合的，甚至可能使其进一步恶化。我们将反例生成任务形式化为针对确定性逐定理Python验证器的受约束见证输出问题，并发布了SymCE数据集：一个包含4,707个错误的本科代数与实分析猜想的数据集，每个猜想均配有可执行的验证器。该验证器同时充当奖励函数，使SymCE成为一个训练环境。在此预言机下对Qwen3-4B进行SFT加GRPO训练揭示了一个“模仿陷阱”：仅使用反例的SFT使真定理识别率从0.27骤降至0.00，而采用稀疏的仅结果奖励的RLVR（可验证奖励强化学习）不仅修复了这一问题，还超越基线达到0.66。该崩溃现象在四个随机种子以及Gemma-3-4B模型上均得到复现。稀疏奖励与稠密奖励在域内成功率上统计上无显著差异，但在……上相差33个百分点（原文在此处截断）。

    arXiv:2610.02444v1 Announce Type: cross  Abstract: Large language models often solve a theorem forward yet fail to disprove a closely related false one: a falsification gap that supervised fine-tuning does not close and can actively worsen. We frame counterexample generation as constrained witness emission against a deterministic per-theorem Python verifier, and release SymCE, a corpus of 4,707 false undergraduate-algebra and real-analysis conjectures, each paired with executable verifiers. The verifier also serves as the reward function, making SymCE a training environment. Training Qwen3-4B with SFT followed by GRPO under this oracle reveals an imitation trap: counterexample-only SFT collapses true-theorem recognition from 0.27 to 0.00, while RLVR with a sparse outcome-only reward repairs this and exceeds the base, to 0.66. The collapse replicates across four seeds and on Gemma-3-4B. Sparse and dense rewards yield statistically indistinguishable in-domain success yet diverge by 33 po
    
[^223]: 你是在合成还是在回忆？评估大语言模型在算法代码检索上的表现

    Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval

    [https://arxiv.org/abs/2610.02438](https://arxiv.org/abs/2610.02438)

    该论文提出将大语言模型对知名算法的代码生成重新定义为“参数化代码检索”任务，并引入AlgoREval基准（涵盖599个问题、77个经典算法、7种编程语言和4种图输入表示）来独立评估这一能力，发现不同语言和输入表示之间的检索准确率差异显著。

    

    大语言模型（LLMs）在代码生成方面已展现出强大的性能，其成功既依赖于回忆相关的算法知识，也依赖于推理如何应用这些知识。然而，现有的LLM处理流程是不透明的，没有对这两个组成部分进行显式区分。我们认为，对于其规范实现在预训练语料库中广泛可获取的知名算法而言，代码生成更适合被衡量为“参数化代码检索”（parametric code retrieval）：即从内化知识中复现一个被指定名称的算法，而非合成一个全新的算法。我们引入了AlgoREval，一个包含599个问题的基准，涵盖14个领域中的77个经典算法、7种编程语言和4种图输入表示，以在隔离环境中评估这一能力，并在零样本设置下评估了15个模型（7B–34B参数）。我们发现，不同语言和输入表示之间的检索准确率存在显著差异。

    arXiv:2610.02438v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated strong performance in code generation, where success depends on both recalling relevant algorithmic knowledge and reasoning about how to apply it. However, existing LLM pipelines are opaque, with no explicit separation between these two components. We argue that for well-known algorithms whose canonical implementations are widely accessible in pretraining corpora, code generation is better measured as \textit{parametric code retrieval}: reproducing a named algorithm from internalised knowledge rather than synthesizing a novel one. We introduce AlgoREval, a benchmark of 599 problems spanning classical 77 algorithms across 14 domains, 7 programming languages, and 4 graph-input representations to evaluate this capability in isolation, and assess 15 models (7B--34B parameters) in a zero-shot setting. We find substantial variation in retrieval accuracy across languages and input representations
    
[^224]: 学习风格，遗忘语义：SFT与RFT在分类任务上的案例研究

    Learning Style, Forgetting Semantics: A Case Study of SFT and RFT on Classification Tasks

    [https://arxiv.org/abs/2610.02437](https://arxiv.org/abs/2610.02437)

    本文通过将策略更新精确分解为语义与风格两个成分，揭示了SFT比RFT遗忘更多的原因——SFT会沿教师风格偏好产生离轴风格漂移从而破坏语义记忆，而RFT能保持类内风格对称性。

    

    为什么即使所有教师演示在语义上都是正确的，监督微调（SFT）仍比强化微调（RFT）导致更多的遗忘？我们在分类任务上研究这个问题，其中每个语义类别内的标记以不同风格表达相同的语义答案。这些任务共享潜在的语义规则，但在提示分布和教师的风格偏好上有所不同。利用一个易于处理的线性softmax策略，我们推导出策略更新在语义成分和风格成分上的精确分解。我们证明，在相同策略和提示下，SFT和RFT具有平行的语义更新，但风格动态不同。从一个没有任何类内风格偏好的策略出发，采用精确策略梯度的RFT能保持这种对称性，而使用非均匀教师的SFT在群体更新下会沿着非零任务均值产生离轴风格漂移。我们利用这种漂移建立了一个……

    arXiv:2610.02437v1 Announce Type: cross  Abstract: Why does supervised fine-tuning (SFT) lead to more forgetting than reinforcement fine-tuning (RFT), even when all teacher demonstrations are semantically correct? We study this question on classification tasks where tokens within each semantic class express the same semantic answer in different styles. The tasks share an underlying semantic rule but differ in their prompt distributions and teachers' stylistic preferences. Using a tractable linear-softmax policy, we derive an exact decomposition of the updates into semantic and style components. We show that, at a common policy and prompt, SFT and RFT have parallel semantic updates but differ in their style dynamics. Starting from a policy with no within-class style preference, RFT with exact policy gradients preserves this symmetry, whereas SFT with a nonuniform teacher develops off-axis style drift along a nonzero task mean under population updates. We use this drift to establish a se
    
[^225]: 面向流映射蒸馏的几何感知时间重参数化

    Geometry-Aware Time Reparameterization for Flow-Map Distillation

    [https://arxiv.org/abs/2610.02427](https://arxiv.org/abs/2610.02427)

    提出一种几何感知的时间重参数化方法，为学生模型在法向加速度大的轨迹区域分配更多蒸馏时间，在保持教师几何路径和终端分布的同时使流映射蒸馏更易学习。

    

    流映射蒸馏通过学习预训练生成式ODE的有限时间转移，实现一步和少步生成。我们研究改变教师模型的时间参数化是否能使这些转移更容易学习。基于“法向加速度较大的轨迹片段更难蒸馏”这一假设，我们提出一种几何感知的时间重参数化方法，为学生模型在这些区域分配更多时间，同时保持教师模型的几何路径和终端分布。在适当假设下，我们推导出一个共享时钟，使总体法向加速度统计量均匀化，并基于跨教师轨迹的鲁棒、正则化估计构建了实用的近似方法。我们将该时钟纳入拉格朗日流映射蒸馏中，使用变换后的时间坐标来条件化学生模型。该时钟在蒸馏前仅需估计一次，且无需重新训练教师模型。

    arXiv:2610.02427v1 Announce Type: cross  Abstract: Flow-map distillation enables one- and few-step generation by learning finite-time transitions of a pretrained generative ODE. We investigate whether changing the teacher's time parameterization can make these transitions easier to learn. Motivated by the hypothesis that trajectory segments with large normal acceleration are harder to distill, we propose a geometry-aware time reparameterization that allocates more student time to these regions while preserving the teacher's geometric paths and terminal distribution. We derive a shared clock that equalizes a population normal-acceleration statistic under suitable assumptions, and construct a practical approximation from robust, regularized estimates across teacher trajectories. We incorporate this clock into Lagrangian flow-map distillation, using the transformed time coordinate to condition the student. The clock is estimated once before distillation and requires neither teacher retrai
    
[^226]: 使用Whiteout缓解大语言模型中的隐私数据泄露

    Mitigating Private Data Leakage in LLMs with Whiteout

    [https://arxiv.org/abs/2610.02418](https://arxiv.org/abs/2610.02418)

    本文提出Whiteout工具，通过用精心设计的混淆样本覆盖个人敏感信息，来防止大语言模型复现泄露个人隐私，相比机器遗忘方法在保护隐私的同时能更好地维持模型效用与安全性。

    

    现代大语言模型（LLM）在大量基本未经筛选的数据集上训练，这些数据包括从几乎所有可访问网站抓取的内容以及用户输入。因此，LLM经常记忆并复现个人敏感信息（PSI），例如出生日期、电话号码和家庭住址。这带来了重大的隐私风险，对于高管、政客和法官等知名人士而言尤其如此。现有的缓解措施主要依赖机器遗忘技术。然而，这些方法往往会删除超出所需范围的信息，损害模型的效用和安全性，并且极易受到攻击。本文提出了Whiteout，这是一个实用的工具，可以根据个人的请求，通过使用精确且精心设计的混淆样本覆盖其真实的个人敏感信息，来防止LLM复现这些信息。我们在来自不同厂商、不同规模大小的现代LLM上评估了Whiteout，其中包括一个被广泛使用的OpenAI模型。

    arXiv:2610.02418v1 Announce Type: cross  Abstract: Modern large language models (LLMs) are trained on massive, largely unfiltered datasets, including content scraped from nearly every accessible website and user inputs. As a result, LLMs often memorize and reproduce personally sensitive information (PSI) such as birth dates, phone numbers, and home addresses. This leads to significant privacy risks, particularly for high-profile individuals such as executives, politicians, and judges. Existing mitigations largely rely on machine unlearning. However, these methods often remove more information than needed, degrade model utility and safety, and are highly vulnerable to attacks.   This paper presents Whiteout, a practical tool that, upon requests by individuals, prevents LLMs from regurgitating their genuine PSIs, by overwriting them using precise and carefully designed obfuscation samples. We evaluate Whiteout on modern LLMs of varying sizes and makers, including a widely-used OpenAI mod
    
[^227]: 通过自适应覆盖与聚焦采样实现高效神经场学习

    Efficient Neural Field Learning via Adaptive Coverage and Focused Sampling

    [https://arxiv.org/abs/2610.02410](https://arxiv.org/abs/2610.02410)

    提出ACES采样框架，通过解耦覆盖与重要性——利用自适应空间分区保证域覆盖、采用区域级重要性加权聚焦关键区域——从而降低梯度方差并显著提升隐式神经表示的训练效率。

    

    隐式神经表示（INRs）为建模高维连续场提供了灵活的框架，但由于均匀子采样忽略了空间异质性，其训练往往效率低下。现有的自适应采样方法通过优先处理高误差样本在一定程度上解决了这一问题，但它们通常在点级别上运作，容易导致局部区域的冗余采样以及对整个域的覆盖不足。我们提出了ACES（自适应覆盖感知高效采样），这是一种结构化采样框架，通过将覆盖与重要性解耦来提高训练效率。ACES构建自适应空间分区以确保域覆盖并减少冗余，并在训练过程中应用区域级重要性加权以优先处理信息量大的区域。我们提供了理论分析，表明自适应分区通过增加区域内的同质性来降低梯度方差。

    arXiv:2610.02410v1 Announce Type: cross  Abstract: Implicit neural representations (INRs) provide a flexible framework for modeling high-dimensional continuous fields, but their training is often inefficient due to uniform subsampling that ignores spatial heterogeneity. Existing adaptive sampling methods partially address this issue by prioritizing high-error samples, but typically operate at the point level, often leading to redundant sampling in localized regions and insufficient coverage of the domain. We propose ACES (Adaptive Coverage-aware Efficient Sampling), a structured sampling framework that improves training efficiency by decoupling coverage and importance. ACES constructs adaptive spatial partitions to ensure domain coverage and reduce redundancy, and applies region-level importance weighting to prioritize informative regions during training. We provide a theoretical analysis showing that adaptive partitioning reduces gradient variance by increasing within-region homogenei
    
[^228]: 当终端智能体训练停滞时：揭秘数据生成与验证的挑战

    When Terminal-Agent Training Stalls: Demystifying Data Generation and Verification Challenge

    [https://arxiv.org/abs/2610.02405](https://arxiv.org/abs/2610.02405)

    该论文揭示了用前沿模型作为元智能体自动生成终端强化学习训练任务时的三类故障（基准无效、评测框架脆弱、奖励错位），并通过实验证明任务可解性区间是模型特定的，主张应将可解性区间校准、验证器审计和基础设施错误核算作为一等评估标准。

    

    使用 Claude Opus 这类前沿模型作为元智能体来生成用于强化学习训练的终端任务和验证器的做法正日益普遍。然而，一个可运行的 Docker 镜像和可执行的测试套件并不能保证终端智能体训练的端到端流程忠实可靠。针对这一空白，我们提出了一个元智能体流水线，并诊断出三类故障：基准无效性、评测框架（harness）脆弱性和奖励错位。通过重新设计提示词和扩展上下文，基线可解性提升了 5.6 倍，但一个 9B 模型在 Claude Opus 生成的任务上，于 20 步内平均 pass@2 饱和在 81.3%。在不改变训练配置的情况下加入困难任务，平均 pass@2 降至 20.6%，这有力地证明了可解性区间是模型特定的。这些发现表明，元智能体的可靠性需要将可解性区间校准、验证器审计和基础设施错误核算作为一等评估标准，而非事后补救……

    arXiv:2610.02405v1 Announce Type: new  Abstract: Using a frontier model like Claude Opus as a meta-agent to generate terminal tasks and verifiers for RL training is increasingly common. Yet a runnable Docker image and executable test suite do not guarantee a faithful end-to-end pipeline for terminal agent training. We present a meta-agent pipeline motivated by this gap, diagnosing three classes of failure: benchmark invalidity, harness brittleness, and reward misalignment. Prompt redesign and context extension raise baseline solvability 5.6 times, but a 9B model saturates at 81.3% mean pass@2 within 20 steps on Claude Opus-generated tasks. Adding hard tasks reduces mean pass@2 to 20.6% without changing the training configuration, a strong evidence that the solvability band is model-specific. These findings demonstrate that meta-agent reliability requires solvability-band calibration, verifier audits, and infrastructure error accounting as first-class evaluation criteria, not post-hoc d
    
[^229]: Inherit-MAS：通过工作流与执行继承实现多智能体系统的测试时演化

    Inherit-MAS: Test-Time Evolution of Multi-Agent Systems through Workflow and Execution Inheritance

    [https://arxiv.org/abs/2610.02396](https://arxiv.org/abs/2610.02396)

    该论文提出Inherit-MAS框架，借鉴生物进化中遗传与选择的机制，在工作流和执行两个层面实现显式继承，使基于大语言模型的多智能体系统能够在测试时高效演化工作流，同时避免破坏有用组件和产生冗余计算。

    

    基于大语言模型构建的多智能体系统（MAS）通过协调专业化智能体来解决复杂任务，但有效的工作流难以预先设计。测试时演化利用执行反馈来改进工作流，然而大范围的修改可能会破坏有用的组件，而重新执行未改变的请求则会产生冗余计算。受生物进化中遗传与选择相互作用的启发，我们提出了Inherit-MAS，它在工作流和执行两个层面将继承机制显式化。元模型首先综合生成一个由工作者智能体组成的工作流，这些智能体具有声明的角色、通信输入和工具权限，并由一个单独提示的评判者对每个已执行的候选方案进行评分并诊断其缺陷。在常规的改进轮次中，工作流继承从最新完成的候选方案出发，可以丢弃被判定为无用的可移除节点，并应用经过验证的编辑来解决相应问题……

    arXiv:2610.02396v1 Announce Type: cross  Abstract: Multi-agent systems (MAS) built from large language models coordinate specialized agents to tackle complex tasks, but effective workflows are difficult to design in advance. Test-time evolution refines workflows using execution feedback, yet broad revisions can disturb useful components, while re-executing unchanged requests can incur redundant computation. Inspired by the interplay of inheritance and selection in biological evolution, we introduce Inherit-MAS, which makes inheritance explicit at the workflow and execution levels. A meta-model first synthesizes a workflow of worker agents with declared roles, communication inputs, and tool permissions, and a separately prompted judge scores each executed candidate and diagnoses its deficiencies. In ordinary refinement rounds, \emph{workflow inheritance} starts from the latest completed candidate, may discard removable nodes judged unhelpful, and applies a validated edit to address the 
    
[^230]: FlashSinkhorn 2：块稀疏熵正则最优传输

    FlashSinkhorn 2: Block-Sparse Entropic Optimal Transport

    [https://arxiv.org/abs/2610.02395](https://arxiv.org/abs/2610.02395)

    提出FlashSinkhorn 2，通过粗阶段质心求解加势提升与块稀疏精细阶段相耦合的两阶段设计，在单个GPU上将大规模熵正则最优传输问题求解至预设边际残差，同时保证被省略块贡献的有界性。

    

    针对熵正则最优传输（EOT）的流式GPU求解器（如FlashSinkhorn）虽避免了存储稠密核矩阵，但在每次Sinkhorn迭代中仍需计算所有 n×m 个点对。我们提出了 FlashSinkhorn 2（FS2），这是一个针对低维点云上平方欧氏代价的求解器，通过耦合两个阶段，在单个GPU上将大规模离散EOT问题求解至预设的边际残差。粗阶段在单元质心上求解，将势函数提升到每个点；当基于采样的边际检查拒绝该提升时，则继续在质心上求解，从而替代了大部分点级别的更新。随后，块稀疏的精细阶段消除粗更新无法消除的质心误差。其按Morton顺序排列的块支持筛选和融合的张量核心执行，且由块质量设定的阈值限制了每个被省略块对每一行和每一列的贡献。在合成基准测试中，FS2在全部32个问题上均达到了目标残差……

    arXiv:2610.02395v1 Announce Type: new  Abstract: Streaming GPU solvers for entropic optimal transport (EOT), such as FlashSinkhorn, avoid storing the dense kernel but still evaluate all $n\times m$ point pairs in every Sinkhorn iteration. We present \textbf{FlashSinkhorn~2} (FS2), a solver for squared-Euclidean cost on low-dimensional point clouds that solves large discrete EOT problems to a prescribed marginal residual on a single GPU by coupling two stages. A coarse stage solves on cell centroids, lifts the potentials to every point and, when a sampled marginal check rejects the lift, continues on the centroids, replacing most point-level updates. A block-sparse fine stage then removes the centroid error that coarse updates cannot. Its Morton-ordered blocks support screening and fused tensor-core execution, and a threshold set by the block masses bounds each omitted tile's contribution to every row and column. On synthetic benchmarks, FS2 reaches the target residual on all 32 problem
    
[^231]: 循环Transformer中共享内存的惊人有效性

    The Surprising Effectiveness of Shared Memory in Looped Transformers

    [https://arxiv.org/abs/2610.02383](https://arxiv.org/abs/2610.02383)

    提出让循环Transformer在预训练时共享内存（仅第一次递归写入键值缓存、后续递归读取并保留自身短窗口）的方法，不仅不损失质量反而提升质量，在减少76-79%上下文内存的同时刷新了循环模型的质量-内存边界。

    

    循环Transformer对每个token多次应用相同的层，在不增加参数的情况下通过更多计算来提升质量。然而，每次递归都会写入自己的键值缓存，因此内存仍随计算量增长。推理时技术可以缩小该缓存，但会以质量为代价。我们对循环语言模型进行预训练以共享内存：只有第一次递归写入缓存，后续递归读取该缓存，同时保留自己的一小段窗口。令人惊讶的是，我们发现共享内存不仅不损失质量，反而提升了质量。在1.5亿至10亿参数规模下，我们的循环预测Transformer（LPT）及其混合变体为循环模型树立了新的质量-内存边界：通过五次递归，混合变体在FineWeb-Edu数据集上将验证困惑度相比同等规模的标准Transformer降低了1.12-1.82，同时上下文内存使用减少76-79%。通过广泛的分析，我们研究了内存共享为何有帮助。共享内存与本地内存……

    arXiv:2610.02383v1 Announce Type: cross  Abstract: Looped Transformers apply the same layers several times per token, adding compute to improve quality without more parameters. Each recursion, however, writes its own key-value cache, so memory still grows with compute. Inference-time techniques can shrink this cache at a cost in quality. We pretrain looped language models to share memory: only the first recursion writes a cache, and later recursions read it while keeping a short window of their own. Surprisingly, we find that sharing memory does not cost quality and instead improves it. At 150M-1B parameters, our Looped Prediction Transformer (LPT) and its hybrid variant set a new quality-memory frontier for looped models: with five recursions, the hybrid lowers validation perplexity on FineWeb-Edu by 1.12-1.82 relative to a same-size standard Transformer while using 76-79% less context memory. Through an extensive analysis, we investigate why memory sharing helps. Shared and local mem
    
[^232]: THPL：一种面向循环水养殖系统中虹鳟喂养管理的视觉到语言决策支持框架

    THPL: A Vision-to-Language Decision Support Framework for Rainbow Trout Feeding Management in RAS

    [https://arxiv.org/abs/2610.02378](https://arxiv.org/abs/2610.02378)

    该论文提出THPL框架，通过轨迹活动系数量化、层次化行为编码以及结合专家规则的LoRA微调大语言模型，将虹鳟行为视觉信息转化为可执行、可解释的精准投喂决策，实现循环水养殖中视觉到语言的决策支持。

    

    在循环水养殖系统中，精准投喂对于降低成本和改善鱼类福利至关重要。然而，现有方法缺乏鱼类行为与管理知识之间的认知对齐，阻碍了其转化为可执行、可解释的投喂决策。为解决这一问题，我们提出了THPL，一个专为循环水养殖系统中虹鳟设计的生成式投喂决策框架。首先，Fishsort提取鱼类轨迹，建立用于量化投喂强度的活动系数（AC）。其次，层次化行为编码器（HBE）利用时间Transformer和集合Transformer对个体时间进程与群体动态进行建模，将轨迹张量转化为显式物理证据与隐式软标记的双重证据表示。最后，将这些标记与环境参数、元数据及专家规则相整合，通过LoRA微调大语言模型，随后进行反事实多选评估（原文摘要在此处截断）。

    arXiv:2610.02378v1 Announce Type: new  Abstract: In Recirculating Aquaculture Systems (RAS), precision feeding is critical for minimizing costs and improving fish welfare. However, existing methods lack cognitive alignment between fish behaviors and management knowledge, impeding translation into executable, interpretable feeding decisions. To address this, we propose THPL, a generative feeding decision framework tailored for rainbow trout (Oncorhynchus mykiss) in RAS. First, Fishsort extracts trajectories to establish an Activity Coefficient (AC) quantifying feeding intensity. Second, a Hierarchical Behavior Encoder (HBE) models individual temporal progression and collective dynamics using Temporal and Set Transformers, transforming trajectory tensors into dual-evidence representations of explicit physical and implicit soft tokens. Finally, these tokens are integrated with environmental parameters, metadata, and expert rules to fine-tune an LLM via LoRA, followed by counterfactual mul
    
[^233]: Coco：硬件-软件协同设计生命周期的智能体副驾驶

    Coco: An Agentic Copilot for the Hardware--Software Co-Design Lifecycle

    [https://arxiv.org/abs/2610.02376](https://arxiv.org/abs/2610.02376)

    Coco是一个与TPU架构师共同部署的智能体平台，通过将全新仿真扫描数据自动注册进规范化数据库、让LLM智能体基于真实仿真证据进行推理，从而加速了机器学习模型与硬件加速器的协同设计生命周期。

    

    协同设计机器学习模型和运行它们的加速器是一项非常规的推理任务：架构师必须对尚不存在的系统做出自信且高风险的结论，而模型演进和硬件迭代的双重节奏意味着分析负担每个季度都在增长。每项决策背后的证据——针对新颖设计点所产生的数百GB的全新仿真扫描数据——从本质上就不存在于任何大语言模型（LLM）的预训练语料库中，也没有外部文献可供检索；天真的“与数据对话”式方法恰恰在最需要正确性的地方产生幻觉。我们提出了Coco（Copilot for Codesign，协同设计副驾驶），这是一个与TPU架构师共同部署的智能体平台，用于加速设置实验、扫描仿真器和提炼洞察的协同设计生命周期。Coco由四层构建：(i) 一个数据存储层，可自动将每次仿真扫描注册到规范化的关系模式中，使智能体能够基于……

    arXiv:2610.02376v1 Announce Type: cross  Abstract: Co-designing ML models and the accelerators that run them is an unusual reasoning task: architects must draw confident, high-stakes conclusions about systems that do not yet exist, and the pace of both model evolution and hardware cadence means the analysis burden grows every quarter. The evidence behind each decision--hundreds of gigabytes of fresh simulation sweeps over novel design points--is by construction absent from any LLM's pretraining corpus, and there is no external literature to retrieve; naive "chat-with-your-data" approaches hallucinate exactly where correctness matters most. We present Coco (Copilot for Codesign), an agentic platform deployed with TPU architects that accelerates the co-design lifecycle of setting up experiments, sweeping simulators, and deriving insights. Coco is built as four layers: (i) a datastore that automatically registers every simulation sweep into a normalized relational schema, so agents ground
    
[^234]: EviDent-CBCT：非详尽报告监督下基于证据瓶颈的牙科CBCT报告生成

    EviDent-CBCT: Evidence-Bottlenecked Report Generation from Dental CBCT under Non-Exhaustive Report Supervision

    [https://arxiv.org/abs/2610.02375](https://arxiv.org/abs/2610.02375)

    该论文提出EviDent-CBCT框架，通过解剖感知的离散证据记录、牙科逻辑一致性校正和可靠性感知训练，解决了牙科CBCT常规报告标注不完整（未提及≠不存在）的问题，实现仅依据证据记录即可自动生成牙科报告。

    

    牙颌面锥形束CT（CBCT）报告可能包含来自单次三维扫描的数十条牙齿特异性、解剖学及空间性发现。在有限临床数据下学习生成此类报告极具挑战性，因为常规报告可能不会详尽记录影像发现，而“未提及”既可能反映病灶不存在，也可能是漏报。我们提出EviDent-CBCT，一个专为这种不完整监督设计的证据瓶颈框架。一个解剖感知网络将每张CBCT扫描映射为包含牙齿级别、全局以及牙齿-下牙槽管（IAC）证据的离散记录。牙科逻辑一致性投影先对相互矛盾的证据进行调和，随后由确定性渲染器和图像盲的本地语言模型仅基于该记录生成报告。对于牙齿级别证据，可靠性感知训练将符合条件的未提及项作为降低权重的负样本，而未报告的全局和牙齿-IAC标签则保持未知。一种金属敏感的输入……（原文摘要在此处截断）

    arXiv:2610.02375v1 Announce Type: cross  Abstract: Dento-maxillofacial cone-beam CT (CBCT) reports may contain dozens of tooth-specific, anatomical, and spatial findings from a single 3D scan. Learning to generate such reports from limited clinical data is challenging because routine reports may not exhaustively document image findings, and a non-mention may reflect either absence or non-reporting. We present EviDent-CBCT, an evidence-bottlenecked framework designed for this incomplete supervision. An anatomy-aware network maps each CBCT scan to a discrete record of tooth-level, global, and tooth-IAC evidence. A dental-logic consistency projection reconciles incompatible evidence before a deterministic renderer and an image-blind local language model generate the report using only this record. For tooth-level evidence, reliability-aware training uses eligible non-mentions as reduced-weight negatives, while unreported global and tooth-IAC labels remain unknown. A metal-sensitive input c
    
[^235]: 跳数衰减影响：基于大语言模型的GraphRAG流水线中结构辅助索引的新漏洞

    Hop-Decayed Influence: New Vulnerabilities of Structural Auxiliary Indexing in GraphRAG Pipelines with LLM

    [https://arxiv.org/abs/2610.02373](https://arxiv.org/abs/2610.02373)

    提出跳数衰减影响（HDI）攻击，通过查询感知的影响力传播识别并破坏GraphRAG流水线中的模式级辅助索引结构，仅修改0.016%的辅助结构即可达到88-94%的攻击成功率，实现1:N放大效应。

    

    GraphRAG流水线在离线索引阶段构建辅助结构——语义摘要、层次化边和预计算评分——这些结构决定了查询时检索的优先级排序。先前的攻击仅针对实例级组件（节点、边、三元组），而忽视了这些模式级结构。我们将辅助模式级实体形式化为一种全新的攻击面，并提出3S框架（语义、结构、评分）用于对其进行系统性利用。我们提出的跳数衰减影响（HDI）攻击通过查询感知的影响力传播识别高影响目标，并在索引完成后对其辅助结构进行破坏。在两个基准数据集（HotpotQA、2WikiMultiHopQA）和两种架构（Microsoft GraphRAG、HippoRAG2）上，HDI在仅修改0.016%的辅助结构的情况下达到了88-94%的攻击成功率。每次修改最多可影响6.00个查询（模式杠杆率），展示了1:N放大效应。

    arXiv:2610.02373v1 Announce Type: cross  Abstract: GraphRAG pipelines construct auxiliary structures during offline indexing--semantic summaries, hierarchical edges, and pre-computed scores--that determine how retrieval is prioritised at query time. Prior attacks target only instance-level components (nodes, edges, triples), overlooking these schema-level structures. We formalise Auxiliary Schema-Level Entity as a novel attack surface and propose the 3S Framework (Semantics, Structure, Scoring) for its systematic exploitation. Our Hop-Decayed Influence (HDI) attack identifies high-impact targets through query-aware influence propagation and corrupts their auxiliary structures post-indexing. Across two benchmarks (HotpotQA, 2WikiMultiHopQA) and two architectures (Microsoft GraphRAG, HippoRAG2), HDI achieves 88-94% attack success rate while modifying as few as 0.016% of auxiliary structures. Each modification affects up to 6.00 queries (Schema Leverage Ratio), demonstrating 1:N amplifica
    
[^236]: 在文本到图像扩散模型中遍历满意度-多样性前沿

    Traversing the Satisfaction-Diversity Frontier in Text-to-Image Diffusion

    [https://arxiv.org/abs/2610.02372](https://arxiv.org/abs/2610.02372)

    提出SatisDive，一种无需训练的推理时方法，通过将文生图生成建模为“满意化”问题——要求每张图像满足奖励下限、整批图像满足多样性截止标准——实现了对奖励与多样性之间帕累托前沿的有效遍历。

    

    文本到图像生成使用户能够探索由同一提示词生成的多张图像。为了让这些生成的图像具有实用价值，每张图像都必须反映用户的偏好（通过学习到的奖励来衡量），同时彼此在视觉上有所差异以保持多样性。现有方法存在局限：它们要么分别处理奖励和多样性，要么将两者合并为一个综合评分，从而使高多样性可以弥补低奖励的不足。在本文中，我们通过将生成问题表述为“满意化”来解决这些局限：每张图像（候选）必须满足一个奖励下限，而整批图像必须满足一个多样性截止标准。奖励下限控制着最差候选奖励与批次多样性之间的平衡；我们证明通过改变这个下限可以定义出一条帕累托前沿。为了遍历这条前沿，我们提出了SatisDive，一种无需训练的推理时方法。SatisDive使用批次相对奖励截止标准来区分较劣候选与较优候选。

    arXiv:2610.02372v1 Announce Type: new  Abstract: Text-to-image generation enables users to explore several images generated from the same prompt. For these generated images to be useful, each one must reflect the user's preferences, measured by a learned reward, and differ visually from the others to maintain diversity. Existing methods are limited: they either address reward and diversity separately or combine them in one aggregate score, enabling high diversity to offset low rewards. In this paper, we address these limitations by formulating generation as satisficing: every image (candidate) must satisfy a reward floor and the batch of images must satisfy a diversity cutoff. The reward floor controls the balance between worst-candidate reward and batch diversity; we show that varying this floor defines a Pareto frontier. To traverse this frontier, we introduce SatisDive, a training-free inference-time method. SatisDive uses a batch-relative reward cutoff to distinguish lower- from hi
    
[^237]: 大规模网络在环：面向大规模并行机器人学习的GPU批量5G仿真

    Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning

    [https://arxiv.org/abs/2610.02370](https://arxiv.org/abs/2610.02370)

    提出了Isaac-Net，一个GPU批处理的5G新空口模块，可与Isaac Lab物理仿真同步，对数千个并行环境同时进行时隙级的5G上行链路模拟，使大规模并行机器人学习能够实现网络在环训练。

    

    大规模并行GPU仿真器可在数千个环境中训练多机器人策略，而许多机器人编队使用专用第五代（5G）网络，其中每个机器人的延迟取决于其队友的流量。网络在环训练是将模拟的5G网络置于该训练循环之中。然而，GPU机器人仿真器将网络简化为每条消息的独立延迟，而包级仿真器每个CPU进程只能运行一个场景，无法跟上数千个并行环境的速度。为了弥合这一差距，我们提出了Isaac-Net，这是一个GPU批处理的5G新空口（NR）模块，它使数千个环境的上行链路与Isaac Lab物理仿真同步推进。Isaac-Net对所有环境同时模拟每一个时隙（即基站决定哪些机器人进行传输的0.5毫秒时间间隔）。大量实验证实，其NR引擎在各种负载下重现了ns-3 5G-LENA的中位延迟，中位延迟偏差仅为5-10%。

    arXiv:2610.02370v1 Announce Type: cross  Abstract: Massively parallel GPU simulators train multi-robot policies in thousands of environments, and many fleets use private Fifth-Generation (5G) networks, where each robot's delay depends on its teammates' traffic. Network-in-the-loop training places a simulated 5G network inside this loop. However, GPU robot simulators reduce the network to an independent delay per message, while packet-level simulators run one scenario per CPU process and cannot keep pace with thousands of parallel environments. To bridge this gap, we present Isaac-Net, a GPU-batched 5G New Radio (NR) module that advances the uplink of thousands of environments in lockstep with Isaac Lab physics. Isaac-Net simulates every slot, the 0.5~ms interval in which the base station decides which robots transmit, for all environments at once. Extensive experiments confirm that its NR engine reproduces the median delay of ns-3 5G-LENA across loads, with a median delay 5--10\% low o
    
[^238]: 人机交互原则的自动化应用：按需构建用户界面的技能、人机“思考空间”与人机交互的未来

    Automating the Application of HCI Principles: Skills for On-Demand UI Construction, the Human-AI Space to Think, and the Future of HCI

    [https://arxiv.org/abs/2610.02369](https://arxiv.org/abs/2610.02369)

    该论文提出了一个“思考空间”框架，将用户与AI的对话作为共享的结构化认知工作空间，使按需生成的用户界面成为用户思维的延伸，并借助经典HCI设计知识的自动化应用，实现从“生成界面”到“良好生成界面”的跨越。

    

    人机交互（HCI）正处于一场转型之中：大语言模型现在能够根据自然语言任务描述按需生成功能性的用户界面（UI）。用户解释自己想要完成的任务，系统便会生成一个可运行的界面来支持这一任务。这种能力已经存在于Claude和ChatGPT等系统中，并且随着底层模型的改进，其保真度也在不断提升。沿着这一发展轨迹的下一步，是从“仅仅被生成的界面”迈向“被良好生成的界面”。我们提出了一个框架，其中用户与人工智能（AI）之间的对话成为一个“思考空间”：一个共享的、结构化的认知工作空间，在其中通过任务分解产生的按需用户界面成为用户思维的延伸，而非一个独立的产物。在这一范式下，经典的HCI设计知识（尼尔森的启发式原则、诺曼的……

    arXiv:2610.02369v1 Announce Type: cross  Abstract: Human-computer interaction (HCI) is in the middle of a transition: large language models can now generate functional user interfaces (UIs) on demand from natural-language task descriptions. A user explains what they are trying to accomplish, and the system materializes a working interface to support it. This capability already exists in systems such as Claude and ChatGPT and continues to grow in fidelity as the underlying models improve. The next step along this trajectory is to move from interfaces that are merely generated to interfaces that are generated well. We propose a framework in which the dialogue between user and artificial intelligence (AI) becomes a Space to Think: a shared, structured cognitive workspace in which task decomposition produces an on-demand user interface as an extension of the user's thinking rather than as a separate artifact. Within this paradigm, classical HCI design knowledge (Nielsen's heuristics, Norma
    
[^239]: 字典序多目标在线策略蒸馏

    Lexicographic Multi-Objective On-Policy Distillation

    [https://arxiv.org/abs/2610.02359](https://arxiv.org/abs/2610.02359)

    提出了字典序多目标在线策略蒸馏（LMOPD），一种多教师蒸馏方法，在显式优先级保护下整合奖励专门化策略，确保低优先级目标（如简洁性）不会以牺牲高优先级目标（如正确性）为代价而提升。

    

    基于可验证奖励的强化学习（RLVR）通常只优化答案的正确性，然而有用的语言模型行为还需要高质量的推理和简洁的回复。现有的多奖励后训练方法通常对奖励进行标量化，或组合多个专家模型，却没有显式地保护奖励的优先级顺序。当各目标之间的权衡不对称时，这种做法是有问题的：例如，简洁性不应以牺牲正确性为代价来提升。我们提出了字典序多目标在线策略蒸馏（LMOPD），这是一种在显式优先级约束下整合奖励专门化策略的多教师方法。对于学生模型的每一次采样轨迹，LMOPD 会选择门控机制检测到存在缺陷的首个目标所对应的专家，然后将其中心化的对数策略修正进行局部投影，以去除与更高优先级专家相冲突的成分。我们在两个专家和四个专家的设置下评估了 30B-A3B 混合专家 transformer 模型……

    arXiv:2610.02359v1 Announce Type: cross  Abstract: Reinforcement learning from verifiable rewards (RLVR) usually optimizes answer correctness, yet useful language-model behavior also requires high-quality reasoning and concise responses. Existing multi-reward post-training methods typically scalarize rewards or combine specialists without explicitly protecting a reward priority order. This is problematic when trade-offs are asymmetric: conciseness, for example, should not improve at the cost of correctness. We introduce Lexicographic Multi-Objective On-Policy Distillation (LMOPD), a multi-teacher method for integrating reward-specialized policies under explicit priorities. For each student rollout, LMOPD selects the specialist for the first objective whose gate detects a deficiency, then locally projects its centered log-policy correction to remove components that oppose higher-priority specialists. We evaluate 30B-A3B mixture-of-experts transformer models in two- and four-expert setti
    
[^240]: DeReAct：面向可靠AI智能体的分解式推理与行动

    DeReAct: Decomposed Reasoning and Acting for Reliable AI Agents

    [https://arxiv.org/abs/2610.02351](https://arxiv.org/abs/2610.02351)

    DeReAct提出了一种模块化智能体架构，通过将动作验证（Critic）与任务完成认证（Context Manager）从单一LLM策略中外置为独立门控机制，防止错误传播和无效完成声明，在GAIA和SWE-bench Verified上对较弱模型带来了最显著的Pass@1提升。

    

    基于ReAct的智能体通常依赖单一的大语言模型策略来提出动作、与环境交互，并决定任务何时完成。这种耦合使得动作授权与任务完成控制难以独立执行，从而导致错误传播，以及缺乏依据的完成声明过早终止执行过程。我们提出了DeReAct，一种模块化的智能体架构，它将两种门控策略外置：一个Critic（评论者）在执行前验证所提出的动作，一个Context Manager（上下文管理器）重构环境支持的State（状态）并认证任务完成。在GAIA和SWE-bench Verified基准上，DeReAct对较弱的Brain模型在Pass@1上的提升最为显著，其中Qwen3-Coder-480B提升6.5–7.0分，Claude Sonnet 4.5提升4.2–5.2分；随着Brain模型能力的增强，提升幅度逐渐减小。轨迹与消融分析表明，当目标故障足够普遍时，外部门控才是有效的……

    arXiv:2610.02351v1 Announce Type: new  Abstract: ReAct-based agents typically rely on a single LLM policy to propose actions, interact with the environment, and decide when a task is complete. This coupling makes action authorization and completion control difficult to enforce independently, allowing errors to propagate and unsupported completion claims to terminate execution. We introduce DeReAct, a modular agent architecture that externalizes two gating policies: a Critic that validates proposed actions before execution, and a Context Manager that reconstructs an environment-supported \textsc{State} and certifies task completion.   Across GAIA and SWE-bench Verified, DeReAct improves Pass@1 most for weaker Brain models, with gains of 6.5--7.0 points for Qwen3-Coder-480B and 4.2--5.2 points for Claude Sonnet~4.5; gains diminish as Brain capability increases. Trajectory and ablation analyses show that external gating is effective when targeted failures are sufficiently prevalent and th
    
[^241]: MIRROR：面向大语言模型多智能体通信的多路径法定人数完整性机制

    MIRROR: Multipath Quorum Integrity for LLM Multi-Agent Communication

    [https://arxiv.org/abs/2610.02349](https://arxiv.org/abs/2610.02349)

    提出MIRROR，一种通信层完整性原语，通过将消息负载复制到k条逻辑路径并要求严格多数路径返回相同摘要，来防御大语言模型多智能体系统中篡改传输消息的中间智能体（AiTM）攻击。

    

    智能体间通信是大语言模型多智能体系统的核心，但它引入了一种尚未被充分研究的漏洞：中间智能体攻击（Agent-in-the-Middle, AiTM），即在不攻破智能体本身的情况下篡改传输中的消息。已有研究报告称，在结构化任务上此类攻击的成功率接近100%。现有防御手段要么依赖语义验证，这需要额外的推理开销且可能阻断良性输出；要么依赖传输层加密，但当中间人合法地终止TLS连接时，加密便失去作用。我们提出了MIRROR，这是一种通信层完整性原语，它将单个规范化后的消息负载复制到k条逻辑路径上，只有当严格多数路径报告相同的消息摘要时才接受该消息。MIRROR使用无密钥哈希，因此其自身不提供任何认证能力，因为主动的路径上攻击者总能对其篡改后的负载重新计算摘要。所有完整性（原文在此处截断）

    arXiv:2610.02349v1 Announce Type: cross  Abstract: Inter-agent communication is central to Large Language Model Multi-Agent Systems (LLM-MAS), but it introduces an underexplored vulnerability: Agent-in-the-Middle (AiTM) attacks that manipulate messages in transit without compromising the agents themselves. Prior work reports Attack Success Rates (ASR) approaching 100% on structured tasks. Existing defenses rely on semantic validation, which requires additional inference and can block benign outputs, or on transport-layer encryption, which does not help when an intermediary legitimately terminates TLS. We present MIRROR, a communication-layer integrity primitive that replicates a single canonicalized payload across k logical routes and accepts a message only when a strict majority of routes report the same digest. MIRROR uses unkeyed hashing and so authenticates nothing on its own, since an active on-path adversary can always recompute a digest over a payload it has modified. All integr
    
[^242]: 面向基于自然可见图的网络攻击检测的拓扑度量多方法重要性与性能效率分析

    A Multi Method Importance and Performance Efficiency Analysis of Topological Metrics for Natural Visibility Graph Based Cyber Attack Detection

    [https://arxiv.org/abs/2610.02342](https://arxiv.org/abs/2610.02342)

    本研究提出一种整合SHAP、分组置换重要性、Boruta和RFE四种方法的共识排名策略，筛选出仅含少量度量的NVG拓扑度量子集（如Top3配置），在基于CNN的网络攻击检测中保持了分类性能，同时显著提升了计算效率。

    

    基于自然可见图（NVG）的分析通过反映不同结构属性的拓扑描述符来刻画网络流量特征。然而，并非所有描述符对网络攻击分类的贡献都相同，且提取大量度量集合可能增加计算成本。本研究评估了21个由NVG导出的拓扑度量，并探究一个紧凑的度量子集能否在保持分类能力的同时提升计算效率。研究通过共识排名策略整合了四种重要性分析方法：SHAP、分组置换重要性、Boruta和递归特征消除（RFE）。基于该排名，使用CICIDS2018数据集、CNN分类器以及分层五折交叉验证，对Full21、Top15、Top10、Top7、Top5和Top3六种度量配置进行了评估。排名最高的三个度量分别为avg_clustering_coeff_median、avg_clustering_coeff_std和avg_clustering_coeff_m……

    arXiv:2610.02342v1 Announce Type: new  Abstract: Natural Visibility Graph (NVG) based analysis characterizes network traffic through topological descriptors reflecting different structural properties. However, not all descriptors contribute equally to cyber-attack classification, and extracting a large metric set can increase computational cost. This study evaluates 21 NVG derived topological metrics and investigates whether a compact subset can preserve classification capability while improving computational efficiency. Four importance analysis methods SHAP, grouped Permutation Importance, Boruta, and Recursive Feature Elimination (RFE) are integrated through a Consensus Ranking strategy. Based on this ranking, Full21, Top15, Top10, Top7, Top5, and Top3 configurations are evaluated using the CICIDS2018 dataset, a CNN classifier, and stratified 5 fold cross validation. The three highest ranked metrics are avg_clustering_coeff_median, avg_clustering_coeff_std, and avg_clustering_coeff_m
    
[^243]: 世界编辑：在递增深度上干预可执行世界

    World Editing: Intervening on Executable Worlds at Increasing Depth

    [https://arxiv.org/abs/2610.02331](https://arxiv.org/abs/2610.02331)

    该论文提出了“世界编辑”的形式化框架与“干预深度”维度，并基于Minecraft和Terraria构建了包含110个任务、1100余条评估标准的IGMBench基准，揭示前沿编码智能体已能解决其中78.2%的可执行世界编辑任务。

    

    交互式世界模型日益能够生成环境并在其中行动，然而对已有的可执行世界进行有意的编辑仍是一个探索不足的领域。我们将世界编辑形式化为在保留应当不变的性质的同时对现有世界进行干预，并引入“干预深度”这一维度，用以描述一次编辑在多大程度上耦合世界的实体、动态与系统。我们通过工业级游戏模组化技术实现了这一能力，并推出了IGMWorld，同时提出IGMBench——一个涵盖Minecraft和Terraria两款游戏、包含110个任务和超过1100条可执行状态与行为标准的基准。这些任务横跨属性、实体、动态和系统四个层面的干预，并通过确定性可执行性、行为、保留性和视觉检查进行评估。前沿编码智能体已展现出可观的世界编辑能力：表现最强的配置解决了78.2%的任务（原文摘要在此处截断）。

    arXiv:2610.02331v1 Announce Type: new  Abstract: Interactive world models are increasingly capable of generating environments and acting within them, yet deliberately editing an existing executable world remains underexplored. We formulate world editing as intervening on an existing world while preserving properties that should remain unchanged, and introduce intervention depth as an axis describing how strongly an edit couples world entities, dynamics, and systems. We instantiate this capability through industry-grade game modding and introduce IGMWorld, together with IGMBench, a benchmark of 110 tasks and over 1.1K executable state and behavioral criteria across Minecraft and Terraria. The tasks span property, entity, dynamics, and system interventions and are evaluated through deterministic executability, behavioral, preservation, and visual checks. Frontier coding agents already exhibit substantial world-editing capability: the strongest configuration solves 78.2% of tasks under a 
    
[^244]: 先抉择后行动：面向长时程工具使用智能体的比较价值估计

    Choosing Before Acting: Comparative Value Estimation for Long-Horizon Tool-Use Agents

    [https://arxiv.org/abs/2610.02330](https://arxiv.org/abs/2610.02330)

    提出CITA方法，让智能体在执行工具调用前通过对同一上下文下备选调用的比较推理来估计其长时程价值，从而解决长时程工具使用中最终结果奖励信用分配弱、步骤级监督获取成本高的问题。

    

    大型语言模型（LLM）在处理复杂任务时依赖长时程的工具调用序列，其中每次调用都可能改变任务状态并影响后续决策。在长时程工具使用中，基于最终结果的奖励对长交互轨迹的信用分配能力较弱。步骤级奖励可以提供更有针对性的反馈，但获取可靠的步骤监督通常需要人工或LLM的判断，或者需要额外的采样来估计某个中间决策的下游影响。在本文中，我们认为有效的工具使用智能体应该在执行之前估计可能的下一次工具调用的长时程价值。这一目标需要对同一上下文下的备选调用进行比较监督，而记录的轨迹中只包含实际被采取的调用。因此，我们提出了面向工具使用智能体的比较推理方法（CITA）。CITA训练一个比较推理模型（CI

    arXiv:2610.02330v1 Announce Type: new  Abstract: Large language models (LLMs) rely on long-horizon tool invocation sequences for complex tasks, where each invocation can alter the task state and condition subsequent decisions. In long-horizon tool use, final-outcome rewards provide weak credit assignment over long interaction traces. Step-level rewards can offer more targeted feedback, but obtaining reliable step supervision often requires human or LLM judgment, or additional rollouts to estimate the downstream effect of an intermediate decision. In this paper, we argue that effective tool-use agents should estimate the long-horizon value of a possible next tool invocation before executing it. This objective requires comparative supervision over alternative invocations under the same context, while logged trajectories only contain the invocation that was actually taken. Therefore, we propose Comparative Inference for Tool-use Agents (CITA). CITA trains a Comparative Inference Model (CI
    
[^245]: 面向能力保持的慢-快多教师在线蒸馏方法

    Slow-Fast Multi-Teacher On-Policy Distillation for Capability Preservation

    [https://arxiv.org/abs/2610.02324](https://arxiv.org/abs/2610.02324)

    提出 SF-MOPD 方法，通过将教师直接更新的快速学生模型与作为动态能力参考的指数移动平均慢速模型相耦合，在多教师在线蒸馏中实现领域专长获取与通用能力保持的平衡。

    

    基础多模态大语言模型旨在支持跨多个领域的广泛能力。多教师在线蒸馏（MOPD）为将特定领域的专业知识整合到单个学生模型中提供了一个有效框架。然而，MOPD 训练会逐渐使学生模型偏离其初始化模型，且随着偏移量的增大，通用能力会随之下降，从而导致能力干扰。一种直接的补救措施是将学生模型约束在其初始化状态附近，但这同样会抑制领域专业知识的获取。我们提出了慢-快多教师在线蒸馏（SF-MOPD），该方法将一个快速模型（即由每个教师直接更新的当前学生模型）与一个慢速模型（即学生模型的指数移动平均）相耦合。慢速模型逐渐吸收学习信号，充当一个动态的能力参考，将通用基础能力与……（摘要在此处被截断）

    arXiv:2610.02324v1 Announce Type: cross  Abstract: Foundation multimodal large language models are designed to support a broad spectrum of capabilities across diverse domains. Multi-teacher on-policy distillation (MOPD) provides an effective framework for consolidating domain-specific expertise into a single student model. However, MOPD training gradually drives the student away from its initialization model, and general capabilities decline as the displacement grows, resulting in capability interference. A direct remedy is constraining the student toward its initialization, but this suppresses the acquisition of domain expertise as well. We propose Slow-Fast Multi-Teacher On-Policy Distillation (SF-MOPD), which couples a fast model, the current student updated directly by each teacher, with a slow model, an exponential moving average of the student. The slow model absorbs the learning signal gradually, serving as a moving capability reference that fuses the general foundation with con
    
[^246]: DeskForge：来自桌面环境的密集监督，用于计算机使用智能体

    DeskForge: Dense Supervision from Desktop Environments for Computer-Use Agents

    [https://arxiv.org/abs/2610.02320](https://arxiv.org/abs/2610.02320)

    本文提出可控桌面环境DeskForge，通过组合和变换真实应用程序生成大规模密集标注语料库DeskForge-1M（含120万条桌面观测与1.597亿个元素实例），有效提升了视觉语言模型在复杂桌面场景中的动作目标定位能力。

    

    计算机使用智能体需要在复杂的桌面场景中可靠地定位动作目标，而在这些场景中，多个应用程序、重叠的窗口以及视觉上相似的控件会争夺注意力。现有的训练数据很少将此类场景与密集标注配对，也很少以可控的方式对其进行变化。我们提出了DeskForge，这是一个可控的桌面环境，通过组合和探索真实应用程序来为计算机使用智能体生成大规模监督数据。它可以变换应用程序状态、内容、窗口布局、外观和分辨率，并将屏幕截图、无障碍树和窗口几何信息融合为密集的元素标注，同时记录每个执行动作的结果。利用该环境，我们构建了DeskForge-1M，这是一个包含120万条标注桌面观测数据的语料库，共含1.597亿个元素实例。我们使用从DeskForge-1M中抽取的20万个定位示例对四个视觉语言模型进行微调，所有四个模型在保留测试集上均有提升。

    arXiv:2610.02320v1 Announce Type: cross  Abstract: Computer-use agents need to reliably ground action targets in complex desktop scenes, where multiple applications, overlapping windows, and visually similar controls compete for attention. Existing training data rarely pair such scenes with dense annotations or vary them in a controlled way. We introduce DeskForge, a controllable desktop environment that composes and explores real applications to generate large-scale supervision for computer-use agents. It varies application states, content, window layout, appearance, and resolution, and fuses screenshots, accessibility trees, and window geometry into dense element annotations while recording the outcome of each executed action. Using this environment, we construct DeskForge-1M, a corpus of 1.2M annotated desktop observations containing 159.7M element instances. We fine-tune four vision-language models on 200K grounding examples drawn from DeskForge-1M. All four improve across held-out
    
[^247]: SimuVerity：面向工程级Simulink模型生成的智能体基准测试

    SimuVerity: Benchmarking Agents for Engineering-Grade Simulink Model Generation

    [https://arxiv.org/abs/2610.02304](https://arxiv.org/abs/2610.02304)

    提出SimuVerity基准，包含101个跨十个工程领域的Simulink模型生成任务，采用分层评估器从六个工程维度对模型评分，发现最佳智能体系统总分仅为42.86，证明结构相似度并不能衡量模型的工程性能。

    

    现有的Simulink基准测试主要评估生成的模型能否编译、执行或与参考模型相似。这些标准无法确定模型是否满足其工程需求。我们提出了SimuVerity，这是一个包含101个跨十个工程领域的文本到可执行Simulink模型生成任务的基准测试。对于每个任务，可执行系统配置文件为工程规范和四类原生仿真场景提供了基础。分层评估器首先检查工件交付、原生可执行性和工程资格，然后从六个维度对合格模型进行评分，涵盖准确性、输出质量、机理保真度、控制与因果完整性、工作域鲁棒性以及动态响应。我们使用SimuVerity评估了六个智能体系统，表现最好的系统总分仅为42.86。结果表明，结构相似度并不能很好地代表工程性能。

    arXiv:2610.02304v1 Announce Type: cross  Abstract: Existing Simulink benchmarks mainly evaluate whether generated models compile, execute, or resemble a reference model. These criteria do not establish whether a model satisfies its engineering requirements. We introduce SimuVerity, a benchmark of 101 text-to-executable Simulink model-generation tasks across ten engineering domains. For each task, executable-system profiles ground the engineering specification and four families of native simulation scenarios. A hierarchical evaluator first checks artifact delivery, native executability, and engineering qualification, then scores qualified models across six dimensions covering accuracy, output quality, mechanistic fidelity, control and causal integrity, operating-domain robustness, and dynamic response. We evaluate six agent systems with SimuVerity. The best system achieves an overall score of only 42.86. The results show that structural similarity is a poor proxy for engineering perform
    
[^248]: 保持冷静（CALM）：分析文本到图像生成中全局不安全性的局限性

    Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation

    [https://arxiv.org/abs/2610.02300](https://arxiv.org/abs/2610.02300)

    本文揭示文本到图像生成中全局不安全防护存在覆盖与选择性的内在权衡，并提出免训练的CALM方法，通过提示词局部的反事实校正精准编辑违规词元表示并抑制不安全残余成分，在不损害良性提示词的前提下有效提升安全性。

    

    面向文本到图像生成的免训练防护方法通常依赖于一种可复用的安全信号，例如不安全方向或全局毒性子空间，并将其广泛地应用于各类提示词。我们对这一全局不安全性假设进行了受控的几何分析，揭示出一个一致的“覆盖-选择性”权衡：紧凑的不安全子空间无法覆盖异构的不安全语义，而更广泛的聚合则会日益扭曲与安全性相关的良性提示词。受此发现启发，我们提出了CALM（反事实自适应局部调制），这是一种免训练防护方法，用提示词局部的反事实校正取代统一的全局移除。利用匹配的不安全-良性锚点，CALM将每个提示词路由到活跃的不安全类别，仅对违规的词元表示进行最小化编辑使其偏向安全一侧，并抑制正向对齐的不安全残余成分。在广泛的评估中，CALM显著改善了对不安全内容的抑制（摘要在此处截断）。

    arXiv:2610.02300v1 Announce Type: new  Abstract: Training-free safeguards for text-to-image generation often rely on a reusable safety signal, such as an unsafe direction or global toxic subspace, applied broadly across prompts. We provide a controlled geometric analysis of this global-unsafety assumption and reveal a consistent coverage-selectivity trade-off: compact unsafe subspaces fail to cover heterogeneous unsafe semantics, whereas broader aggregation increasingly distorts safety-adjacent benign prompts. Motivated by this finding, we propose CALM (Counterfactual Adaptive Local Modulation), a training-free safeguard that replaces uniform global removal with prompt-local counterfactual correction. Using matched unsafe-benign anchors, CALM routes each prompt to active unsafe categories, minimally edits only violating token representations toward the safe side, and suppresses positively aligned unsafe residual components. Across broad evaluation, CALM significantly improves unsafe co
    
[^249]: $\Psi$-韧性：基于一维拓扑信号的无模型特征重要性方法

    $\Psi$-Resilience: Model-Free Feature Importance from 1D Topological Signals

    [https://arxiv.org/abs/2610.02299](https://arxiv.org/abs/2610.02299)

    提出了一种基于一维拓扑信号的无模型特征重要性方法 $\Psi$-Resilience，它通过类条件密度差异构建不一致性景观并利用其持续性定义韧性评分，从而产生上下文鲁棒且可审计的特征排序。

    

    我们提出了 $\Psi$-Resilience（$\Psi$-韧性），这是一种无模型的特征重要性方法，通过一维拓扑信号直接从数据本身推导解释。我们的方法通过估计类条件密度并沿特征轴取其逐点绝对差，构建出类别不一致性景观。然后，该一维信号的0维持续性定义了一个韧性泛函，仅聚合那些在扰动下存活至用户设定的鲁棒性尺度的拓扑特征。由此得到一个上下文鲁棒的重要性评分，并且可以通过底层的一维景观及其持续性进行固有的可审计性验证。我们在合成数据集和真实数据集上对该方法进行了评估。在具有指定真实重要性的合成生成器上，$\Psi$-Resilience 能够高保真地恢复特征排序，Spearman 秩相关性最高达到 0.8，并与多种（基线方法）表现出相当的竞争力。

    arXiv:2610.02299v1 Announce Type: cross  Abstract: We introduce $\Psi$-Resilience, a model-free feature importance method that derives explanations directly from the data itself via 1D topological signals. Our method constructs a class-disagreement landscape by estimating class-conditional densities and taking their pointwise absolute difference along the feature axis. Then, the 0-dimensional persistence of this 1D signal defines a resilience functional that aggregates only those topological features that survive perturbations up to a robustness scale which is set by the user. This gives us a context-robust importance score that is inherently auditable via the underlying 1D landscapes and their persistence. We evaluate our method on both synthetic and real datasets. On synthetic generators with specified ground-truth importance, $\Psi$-Resilience recovers the ranking of features with high fidelity, achieving Spearman rank correlations up to 0.8 and performing competitively with multipl
    
[^250]: EditHero：长时程部件级3D编辑与Vibe建模基准测试

    EditHero: A Benchmark for Long-Horizon Part-Level 3D Editing and Vibe Modeling

    [https://arxiv.org/abs/2610.02298](https://arxiv.org/abs/2610.02298)

    EditHero是首个长时程部件级3D编辑基准测试，通过确定性组装引擎和人工审核来比较自顶向下的非智能体方法与自底向上的LLM/VLM智能体代码编辑方法，结果显示非智能体方法常遗漏所要求的变更并破坏本应保持不变的区域。

    

    3D编辑方法通常只在单次编辑上进行测试，然而一个3D资产是通过一长串的修改迭代构建而成的，每一次修改都必须实现所要求的变更，同时保持其余所有内容不变。我们推出了EditHero，据我们所知，这是首个针对长时程、部件级3D编辑的基准测试，其中包含自然语言指令以及几何和纹理两方面的目标图像。一个确定性组装引擎在每次编辑后生成精确的目标结果，并且每条编辑序列均经过人工审核。我们利用EditHero来比较两种截然相反的3D编辑方法。非智能体方法采用自顶向下的方式，从学习到的3D表示中重新生成整个对象，并推断哪些部分需要保留。相比之下，LLM/VLM智能体采用自底向上的方式，通过检查网格并仅重写指令所需部分的代码来进行编辑。研究发现，非智能体方法常常无法实现所要求的变更，并且会扰动本应保持固定的区域。

    arXiv:2610.02298v1 Announce Type: cross  Abstract: 3D editing methods are usually tested on a single edit, yet an asset is built through a long sequence of revisions, each of which must implement the requested change while leaving everything else unchanged. We introduce EditHero, to our knowledge the first benchmark for long-horizon, part-level 3D editing, with natural-language instructions and target images for both geometry and texture. A deterministic assembly engine produces the exact target after every edit, and every sequence is reviewed by hand. We use EditHero to compare 2 opposite approaches to 3D editing. Non-agentic methods operate top down, regenerating the object from a learned 3D representation and inferring what to keep. In contrast, LLM/VLM agents operate bottom up, editing through code that inspects the mesh and rewrites only the parts required by instructions. The non-agentic methods often miss the requested change and disturb regions that should stay fixed. Most LLMs
    
[^251]: 基于扩散模型合成数据预训练以增强人体活动识别

    Diffusion-Based Synthetic Data Pretraining for Enhancing Activity Recognition

    [https://arxiv.org/abs/2610.02292](https://arxiv.org/abs/2610.02292)

    本研究提出利用扩散模型生成合成传感器数据进行预训练、再在真实数据上微调的两阶段训练策略，以增强CABiGRU模型对进食、饮水等细微少数类别人体活动的识别能力。

    

    人体活动识别（HAR）在医疗保健、健康和日常监测应用中日益重要，其中检测进食和饮水等饮食活动可以为饮食习惯和慢性病管理提供可操作的洞察。然而，HAR系统在细微和代表性不足的类别上往往表现不佳，限制了其在现实世界饮食监测中的实用性。本工作基于CABiGRU——一种具有双向GRU层、多头注意力和残差连接的卷积架构，旨在从智能手表加速度计、陀螺仪和磁力计数据中捕获有判别力的时间模式。为了提高CABiGRU的泛化能力并减少少数类别的欠拟合，我们利用扩散模型生成合成传感器数据窗口，并采用两阶段训练策略：先在合成数据上预训练CABiGRU，然后在真实世界数据上进行微调。

    arXiv:2610.02292v1 Announce Type: cross  Abstract: Human activity recognition (HAR) is increasingly important for healthcare, well-being, and daily monitoring ap- plications, for which detecting alimentary activities such as eating and drinking can provide actionable insight into dietary habits and chronic disease management. HAR systems, however, often underperform on subtle and underrepresented classes, limiting their utility in real-world dietary monitoring. This work builds upon CABiGRU, a convolutional architecture with Bidirectional GRU layers, multi-head attention, and residual connections, designed to capture discriminative temporal patterns from smart- watch accelerometer, gyroscope, and magnetometer data. To improve CaBiGRU's generalization and reduce underfitting in the minority class, we leverage synthetic sensor data windows using a diffusion model and adopt a two-stage training strategy: pre-training CABiGRU on synthetic data, followed by fine-tuning on the real-world dat
    
[^252]: AI风险观测台：从年报中的AI披露能了解到哪些关于社会韧性的信息？

    The AI Risk Observatory: What Can We Learn from AI Disclosures in Annual Reports About Societal Resilience?

    [https://arxiv.org/abs/2610.02281](https://arxiv.org/abs/2610.02281)

    该研究通过可复现的两阶段LLM分类流水线分析了9,821份英国上市公司年报，发现2020-2025年间AI风险披露比例从2.8%激增至41.2%，证明经LLM规模化处理的年报能为社会韧性研究提供关于企业AI应对的有用信号。

    

    社会韧性研究依赖于获取有用且可操作的数据，这引出了我们的主要研究问题：通过使用大语言模型（LLM）大规模处理年报，能否为理解企业如何披露其应对AI的方式提供有用的信号？我们通过对1,362家英国上市公司（2020-2025年，含2026年部分数据）的9,821份年报应用一个可复现的两阶段分类流水线来检验这一问题。我们首先将该方法和474个人工标注的文本段落进行对比验证，发现其具有较高的召回率和中等程度的标签一致性。随后我们报告了三个实证发现：(i) 2020年至2025年间，提及AI风险的年报比例从2.8%上升至41.2%，AI应用披露比例也从13.8%上升至45.2%，且明确提及的供应商集中在一小批以微软为首的主要提供商；(ii) 披露情况在国家关键基础设施部门和市场板块之间存在显著差异：AIM市场板块的年报披露AI风险的比例……

    arXiv:2610.02281v1 Announce Type: new  Abstract: Societal resilience research relies on access to useful and actionable data, which motivates our main research question: Can annual reports, processed at scale with LLMs, provide a useful signal about how companies disclose their response to AI? We test this by applying a reproducible two-stage classification pipeline to 9,821 annual reports from 1,362 UK listed companies (2020-2025, with partial 2026 data). We first validate the method against 474 human-annotated passages, finding high recall and moderate label-level agreement. We then report three empirical patterns: (i) between 2020 and 2025, the share of reports mentioning AI risk rose from 2.8% to 41.2%, while AI adoption disclosure also rose, from 13.8% to 45.2%, and named vendor mentions cluster around a small set of major providers led by Microsoft; (ii) disclosure varies substantially by Critical National Infrastructure sector and market segment: AIM reports disclose AI risk at 
    
[^253]: 快模型，慢证据：面向LLM代理工具框架的System-1决策模型的配对与自审计评估

    Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses

    [https://arxiv.org/abs/2610.02267](https://arxiv.org/abs/2610.02267)

    该论文通过严格配对与自审计的评估发现，托管型System-1决策模型Jev在11个代理决策点中的9个上显著优于开源模型Laya，但两者在零样本模型路由上均未超过随机水平，且开源模型对选项顺序和候选数量高度敏感。

    

    代理工具框架在每个任务中需要做出许多小型、类型化的决策：调用哪个模型、使用哪个工具、检索到的文本是否相关、输入是否携带注入攻击。System-1决策模型通过单次前向传播输出类别概率来回答此类问题，相比LLM调用有望大幅节省成本和延迟。我们在11个代理决策点上对开源权重模型和托管模型进行了配对评估，这些决策点基于18个公开来源构建：共7,283个基础用例加上6,640个鲁棒性变体，采用字节级相同的输入、配对测试以及跨硬件和跨日期的可重复性检查。Jev在11个决策点中的9个上显著更准确（提升+10.8至+46.0个百分点）。两个模型在零样本模型路由上均未超过随机水平，在RAG相关性门控上则不分伯仲。当选项顺序被颠倒时，Laya会改变30%的答案，并且在候选选项较多或相似时性能急剧下降（在50个最近邻的情况下仅为31%……

    arXiv:2610.02267v1 Announce Type: new  Abstract: Agent harnesses make many small, typed decisions per task: which model to call, which tool to use, whether retrieved text is relevant, whether an input carries an injection. System-1 decision models answer such questions in a single forward pass with class probabilities, promising large cost and latency savings over LLM calls. We present a paired evaluation of an open-weight (Laya) and a hosted (Jev) System-1 model on 11 agent decision points built from 18 public sources: 7,283 base cases plus 6,640 robustness variants, with byte-identical inputs, paired tests, and cross-hardware and cross-day reproducibility checks. Jev is significantly more accurate on 9 of 11 decision points (+10.8 to +46.0 pp). Neither model beats chance on zero-shot model routing, and they tie on RAG relevance gating. Laya changes 30% of its answers when the option order is reversed and degrades sharply with many or similar candidates (31% at 50 nearest-neighbour to
    
[^254]: MintFlow：面向约束流匹配的最小轨迹干预

    MintFlow: Minimal Trajectory Intervention for Constrained Flow Matching

    [https://arxiv.org/abs/2610.02260](https://arxiv.org/abs/2610.02260)

    MintFlow是一种无需训练的约束采样框架，通过对预训练流轨迹施加最小干预来满足目标约束，在执行约束的同时最大限度地保持样本对预训练数据分布的保真度。

    

    流匹配模型在生成建模方面表现出色，而许多下游应用要求其生成的样本满足预设的约束条件，例如观测数据和物理定律。然而，现有的约束采样器往往面临一种权衡：强制执行约束可能会导致样本大幅偏离预训练的数据分布。为解决这一权衡问题，我们提出了MintFlow，这是一个无需训练的约束采样框架，它将约束的执行形式化为对预训练流轨迹的最小干预。MintFlow寻找对中间流状态的最小扰动，使得该状态在预训练流场下的后续演化能够满足目标约束。通过在保持预训练流场不变的前提下对流状态进行最小扰动，MintFlow在执行约束的同时，最大限度地减少了相对于预训练分布的不必要偏离。伴随形式的……

    arXiv:2610.02260v1 Announce Type: new  Abstract: Flow matching models excel at generative modeling, and many downstream applications require their samples to satisfy prescribed constraints, such as observed measurements and physical laws. However, existing constrained samplers often face a trade-off: \textit{enforcing constraints can substantially displace samples from the pretrained data distribution}. To address this trade-off, we introduce \textbf{MintFlow}, a training-free constrained sampling framework that formulates constraint enforcement as a minimal intervention on the pretrained flow trajectory. MintFlow seeks the minimal perturbation of an intermediate flow state such that its subsequent evolution under the pretrained flow field satisfies the target constraint. By minimally perturbing the flow state while keeping the pretrained flow field unchanged, MintFlow enforces the constraint while minimizing unnecessary deviation from the pretrained distribution. An adjoint formulatio
    
[^255]: 利用大语言模型克服解释结构建模的挑战

    Overcoming Challenges of Interpretive Structural Modeling with Large Language Models

    [https://arxiv.org/abs/2610.02254](https://arxiv.org/abs/2610.02254)

    本工作将大语言模型作为“不完美专家”引入解释结构建模（ISM），以克服传统专家交互方法繁琐且难以扩展至数百个变量的挑战，并通过对比实验证明逐行和全图因果图发现方法效果最佳。

    

    解释结构建模（ISM）是一种著名的多准则决策方法。ISM 相较于其他方法论的成功之处在于其能够建模因果关系、因素的二元尺度以及由此产生的层次化表示。传统上，建模过程需要与领域专家反复交互直到达成共识才能完成。这一过程繁琐且劳动密集，最重要的是限制了 ISM 扩展到包含数百个变量的研究的能力。借鉴现有的大语言模型（LLM）作为“不完美专家”进行因果图发现的研究成果，本工作探索了一种集成 LLM-ISM 的解释结构建模方法。研究比较并评估了成对、k-wise、逐行和全图四种因果图发现方法。结果表明，用于 ISM 的因果图发现方法在采用逐行方法（SHD=160，F1 分数=0.77）和全图方法（SHD=135，F1 分数=0.73）时表现最佳。

    arXiv:2610.02254v1 Announce Type: cross  Abstract: Interpretive Structural Modeling (ISM) is a well-known process for multi-criteria decision making. The success of ISM over other methodologies is its ability to model causal relationships, the binary scale of factors, and resulting hierarchical representation. Traditionally, the modeling process is performed by repeated interactions with subject matter experts until consensus is reached. This process is tedious, labor-intense, and most importantly limits the ability of ISM to scale to studies with hundreds of variables. Drawing on existing work of causal graph discovery with large language models (LLM) as imperfect experts, this work explores an integrated LLM-ISM approach for ISM. Pairwise, k-wise, rowwise, and full graph discovery methodologies are compared and evaluated. It is shown that causal graph discovery methods for ISM perform best using rowwise (SHD=160, F1-score=0.77) and full graph methods (SHD=135, F1-score=0.73).
    
[^256]: 无需受控实验的科学模拟器反事实预测

    Counterfactual Predictions in Scientific Emulators Without Controlled Experiments

    [https://arxiv.org/abs/2610.02252](https://arxiv.org/abs/2610.02252)

    提出 ReRoute 框架，无需受控实验或模拟器数据，仅通过将查询输入固定为参考值并沿已知机制路径重新引入其变化，结合事实数据微调，即可让科学模拟器准确回答“如果条件不同会怎样”的反事实预测问题。

    

    许多科学问题需要对从未观测到的情况进行推理：如果条件、干预或历史有所不同会怎样？模型可以在已观测数据上做出准确预测，但当相互关联的输入被独立改变时，模型在这类“假设性”查询上往往会失效。一种常见的补救方法是加入受控仿真数据，使这些相关因素被显式解耦，但这需要访问模拟器、计算开销可能很高，并且会继承模拟器自身的建模假设。我们提出了 ReRoute，一个面向目标性科学“假设性”预测的框架，它将事实数据与部分机制知识相结合，无需受控干预数据即可完成适配。ReRoute 将预训练骨干网络中被查询的输入固定到一个参考值，通过已知的机制路径重新引入其变化，并在原始事实数据上进行微调，同时将下游效应留给学习到的动力学模型。

    arXiv:2610.02252v1 Announce Type: cross  Abstract: Many scientific questions require reasoning about what was never observed: What if the conditions, interventions, or history had been different? Models can predict accurately on observed data yet fail on such what-if queries when correlated inputs are varied independently. A common remedy is to add controlled simulation data in which these factors are explicitly disentangled, but this requires access to a simulator, can be computationally expensive, and inherits the simulator's modeling assumptions. We introduce ReRoute, a framework for targeted scientific what-if prediction that combines factual data with partial mechanistic knowledge, without requiring controlled intervention data for adaptation. ReRoute fixes the queried input of a pretrained backbone to a reference value, reintroduces its variation through a known mechanistic pathway, and fine-tunes on the original factual data, while leaving downstream effects to the learned dynam
    
[^257]: 用语言控制生物学：面向细胞、类器官与生物机器人的提示词条件化干预的离线学习

    Toward Controlling Biology with Language:Offline Learning of Prompt-Conditioned Interventions for Cells, Organoids, and Biobots

    [https://arxiv.org/abs/2610.02247](https://arxiv.org/abs/2610.02247)

    该论文提出将已有的生物干预及其实验结果档案作为固定的离线数据集，利用视觉-语言模型自动判断存档结果与自然语言描述是否匹配，从而在无需新实验和人工验证的情况下，学习从自然语言到细胞、类器官和生物机器人干预措施的映射。

    

    人工智能日益成为复杂技术系统的自然语言接口，使人们只需描述想要实现什么，而无需指定如何去做，便能完成复杂的任务。将这种接口扩展到生命系统则更为困难：与代码或图像不同，生物干预并没有封闭形式的语言含义，而学习这种映射所需的“语言-干预-结果”配对数据收集成本极高，因为每个样本都需要单独进行湿实验室实验。解决这一问题的一种途径，是将现有的干预措施及其已被观察到的结果档案视为一个固定的离线数据集，并利用视觉-语言模型在完全不进行任何新实验的情况下，判断某个存档结果是否与一段自然语言描述相匹配。但这一判断是否足够可靠——以至于能够在没有新实验、也没有人工验证的情况下，被用于训练“语言到干预”的映射——（原文摘要在此处截断）

    arXiv:2610.02247v1 Announce Type: cross  Abstract: Artificial intelligence increasingly serves as a natural-language interface to complex technical systems, letting people accomplish sophisticated tasks by describing what they want rather than specifying how to do it. Extending this interface to living systems is harder: unlike code or images, a biological intervention has no closed-form linguistic meaning, and the paired language-intervention-outcome data needed to learn such a mapping is expensive to collect, since each example requires its own wet-lab experiment. One way around this is to treat an existing archive of interventions and their already-observed outcomes as a fixed, offline dataset, and use a vision-language model to judge, without any new experiments, whether an archived outcome matches a natural-language description. But whether that judgment is reliable enough to train a language-to-intervention mapping on -- without new experiments and without human validation -- has
    
[^258]: RxnOptBench：面向有机方法学中反应条件优化的大语言模型基准测试

    RxnOptBench: Benchmarking LLMs for Reaction-Condition Optimization in Organic Methodology

    [https://arxiv.org/abs/2610.02242](https://arxiv.org/abs/2610.02242)

    该论文提出了RxnOptBench，这是首个基于2025年真实发表的有机方法学论文中湿实验优化数据构建的基准，用于评估大语言模型阅读真实条件筛选表格并选出最优反应条件（催化剂、配体、溶剂、温度等）的能力。

    

    化学反应条件优化——即选择能够共同最大化产率和立体选择性的催化剂、配体、溶剂、试剂、温度、时间和气氛——是有机方法学研究中的一个核心且高度依赖判断力的子任务，人们日益期望大语言模型能够为此提供支持。然而，现有的化学基准测试评估的是反应类型标注、逆合成或SMILES字符串操作，并没有要求模型阅读真实的条件筛选表格并选出最佳条件组合。我们推出了RxnOptBench，这是一个基准测试，其中的每个选项和先例都是从2025年发表的有机方法学论文的优化表格中挖掘出的真实湿实验条目，并通过一个连续的相对分数进行评分；该分数源自一个明确声明的主效用指标，该指标将报告的产率与对映体过量（ee）、非对映体比例（dr）和区域异构体比例（rr）相结合；此外，该基准还配备了成对的“有先例vs无先例”对照实验设计……

    arXiv:2610.02242v1 Announce Type: cross  Abstract: Chemical reaction-condition optimization -- choosing the catalyst, ligand, solvent, reagent, temperature, time, and atmosphere that jointly maximize yield and stereoselectivity -- is a central, judgement-laden subtask of organic methodology research that large language models are increasingly expected to support. Yet existing chemistry benchmarks evaluate reaction-class labelling, retrosynthesis, or SMILES manipulation, and do not ask models to read a real condition-screening table and pick the best set. We introduce RxnOptBench, a benchmark whose every option and precedent is a real wet-lab entry mined from the optimization tables of organic-methodology papers published in 2025, graded by a continuous relative score derived from a declared headline utility that combines reported yield with enantiomeric excess (ee), diastereomeric ratio (dr), and regioisomeric ratio (rr), and equipped with a paired precedents-vs-no-precedents design th
    
[^259]: 面向万亿规模混合专家模型的硬件原生联合稀疏-量化方法

    Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts

    [https://arxiv.org/abs/2610.02241](https://arxiv.org/abs/2610.02241)

    提出了一个端到端的软硬件协同设计框架，通过连续重参数化实现稀疏性与量化的可微联合优化，将万亿规模MoE的专家权重压缩为硬件原生的低精度半结构化稀疏表示，从而在稀疏张量核心上加速执行并降低部署的内存瓶颈。

    

    混合专家（MoE）架构使前沿语言模型能够扩展到万亿参数规模，但其部署受到庞大内存占用和内存带宽限制的制约。尽管现代加速器提供了稀疏张量核心，能够通过低精度半结构化稀疏性来减少权重存储并提升吞吐量，但由于显著的模型质量退化以及缺乏分组稀疏GEMM原语，将其应用于MoE仍然具有挑战性。我们提出了一个端到端的软硬件协同设计框架，将专家权重压缩为硬件原生的低精度稀疏表示，并在SpTC上加速其执行。在算法层面，我们的框架通过连续重参数化松弛离散的半结构化支撑选择，在路由器加权的重建目标下实现与量化权重的可微联合优化……

    arXiv:2610.02241v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures allow frontier language models to scale to trillions of parameters, but their deployment is constrained by massive memory footprints and memory-bandwidth limitations. Although modern accelerators provide Sparse Tensor Cores (SpTCs) that reduce weight storage and increase throughput through low-precision semi-structured sparsity, exploiting them for MoEs remains challenging because of substantial model-quality degradation and the lack of grouped sparse GEMM primitives. We present an end-to-end hardware-software co-design framework that compresses expert weights into hardware-native, low-precision sparse representations and accelerates their execution on SpTCs. Algorithmically, our framework relaxes discrete semi-structured support selection through continuous reparameterization, enabling differentiable joint optimization with quantized weights under a router-weighted reconstruction objective and sc
    
[^260]: CORE：面向KV缓存的覆盖校准与逐出质量再分配

    CORE: COverage CAlibration and Evicted-Mass REdistribution for KV Cache

    [https://arxiv.org/abs/2610.02235](https://arxiv.org/abs/2610.02235)

    提出CORE方法，通过覆盖校准与逐出质量再分配机制，在KV缓存压缩中同时利用Top-B排序保留互补KV状态，并用被排除的分配质量补偿被逐出的注意力质量，从而有效降低长上下文解码中的逐出误差。

    

    长上下文解码日益受到键值缓存内存和带宽的制约。现有的固定预算压缩方法通常将保留与补偿相互割裂，而保留排序既无法指定被丢弃的注意力质量，也无法指示由此产生的输出误差方向。我们从一个精确的因式分解出发：逐出误差等于被逐出的注意力质量乘以被逐出质心与保留输出之间的方向差距，这凸显了集合级覆盖在保留策略以及质量保持的内存写入中的重要性。我们提出了CORE（COverage Calibration and Evicted-Mass REdistribution for KV Cache，面向KV缓存的覆盖校准与逐出质量再分配），它将结合查询效用与对数行列式覆盖的离线分配方案蒸馏为一个轻量级的缓存感知索引器。在推理阶段，一个经过校准的分布同时驱动两个通道：其Top-$B$排序保留互补的KV状态，而其被排除的分配质量和条件权重则用于补偿被逐出的注意力质量。

    arXiv:2610.02235v1 Announce Type: cross  Abstract: Long-context decoding is increasingly constrained by key--value (KV) cache memory and bandwidth. Existing fixed-budget compression methods typically separate retention from compensation, while a retention ranking specifies neither discarded attention mass nor the direction of induced output error. We start from an exact factorization: eviction error equals evicted attention mass times the directional gap between the evicted centroid and retained output, highlighting the importance of set-level coverage in retention and mass-preserving memory writing. We introduce CORE COverage Calibration and Evicted-Mass REdistribution for KV Cache, which distills an offline allocation combining query utility and log-determinant coverage into a lightweight cache-aware indexer. At inference, one calibrated distribution drives both channels: its Top-$B$ ordering retains complementary KV states, while its excluded allocation mass and conditional weights 
    
[^261]: 因果发现方法识别英国生物样本库中连接体力活动与痴呆风险的通路

    Causal discovery identifies pathways linking physical activity to dementia risk in the UK BioBank

    [https://arxiv.org/abs/2610.02221](https://arxiv.org/abs/2610.02221)

    本研究将大语言模型引导的因果发现与中介分析相结合，在英国生物样本库4万多名老年人中系统识别出体力活动降低痴呆风险的多个中介通路，并确定抑郁为核心中介通路（占总体关联的15.1%）。

    

    体力活动（PA）始终与较低的痴呆风险相关，然而连接体力活动与痴呆预防的机制仍不完全清楚。在本研究中，我们将大语言模型（LLM）引导的因果发现与中介分析相结合，基于英国生物样本库（UK Biobank）中42,293名60岁及以上的老年人，系统地识别连接客观测量的中高强度体力活动（MVPA）与痴呆风险的通路。在行为、心理、功能和临床等多个领域，因果发现一致地识别出将较高MVPA与较低痴呆风险联系起来的相互关联的通路，这些通路经由抑郁、功能能力、吸烟行为、高血压、心血管疾病、慢性肾病和脑损伤。链式中介分析进一步确定抑郁为核心通路，占MVPA与痴呆风险之间总体关联的15.1%。性别分层分析……

    arXiv:2610.02221v1 Announce Type: cross  Abstract: Physical Activity (PA) is consistently associated with lower risk of dementia, yet the mechanism linking PA to dementia prevention remain incomopletely understood. Here, we integrate large language model (LLM)-guided causal discovery with mediation analysis in 42,293 older adults aged 60 years or older from the UK Biobank to systematically identify pathways connecting objectively measured moderate-to-vigorous physical activity (MVPA) to dementia risk. Across behavioral, psychological, functional, and clinical domains, causal discovery consistently identified interconnected pathways linking higher MVPA to lower dementia risk through depression, functional capacity, smoking behavior, hypertension, cardiovascular disease, chronic kidney disease, and brain injury. Chain mediation analyses further identified depression as a central pathway, accounting for 15.1% of the overall association between MVPA and dementia risk. Sex-stratified analys
    
[^262]: 因果记忆策略：通过干预检索使记忆效用可识别

    Causal Memory Policy: Making Memory Utility Identifiable by Intervening on Retrieval

    [https://arxiv.org/abs/2610.02070](https://arxiv.org/abs/2610.02070)

    该论文提出因果记忆策略（CMP），通过干预检索过程、以已知倾向性为采样的记忆保留固定数量的上下文槽位，解决了记忆因从未被检索而导致效用无法识别的检索层面正性违背问题，实现了记忆效用的无偏估计与最优保留决策。

    

    记忆增强的大语言模型必须决定保留哪些记忆，近期的系统通过估计每个记忆对任务性能的影响来实现这一决策。然而，这些估计完全依赖于被检索到的记忆。当一个记忆从未被检索时，存储层面的干预会产生完全相同的结果，导致其效用无法被识别。这是一种检索层面的正性违背，且对于仅检查记忆操作的诊断方法来说是不可见的。我们提出了因果记忆策略，这是一个因果框架，通过干预检索本身来恢复可识别性，为以已知倾向性采样的记忆保留固定数量的上下文槽位。CMP在平衡分配设计下通过自归一化逆倾向加权来估计记忆效用。我们证明了记忆效用经由检索的因果分解、估计器的无偏性与精确方差，以及不可逆情形下的最优决策规则。

    arXiv:2610.02070v1 Announce Type: new  Abstract: Memory-augmented large language models must decide which memories to retain, and recent systems do so by estimating each memory's effect on task performance. However, these estimates rely entirely on retrieved memories. When a memory is never retrieved, store-level interventions produce identical outcomes, leaving its utility unidentified. This is a retrieval-level positivity violation, invisible to diagnostics that examine only memory operations. We introduce Causal Memory Policy (CMP), a causal framework that restores identification by intervening on retrieval itself, reserving a fixed number of context slots for memories sampled with known propensities. CMP estimates memory utility by self-normalized inverse propensity weighting under a balanced assignment design. We prove the causal factorization of memory utility through retrieval, the unbiasedness and exact variance of the estimator, and the optimal decision rule under irreversible
    
[^263]: 基于MoE路由器的仅解码器模型跨语言对齐

    Cross-Lingual Alignment for Decoder-Only Models using MoE Routers

    [https://arxiv.org/abs/2610.01921](https://arxiv.org/abs/2610.01921)

    该论文提出一种创新方法，利用混合专家（MoE）路由器的输出作为对齐目标，在仅解码器大语言模型中实现跨语言表示对齐，从而提升跨语言迁移能力。

    

    跨语言对比学习一直是多语言编码器训练的核心组成部分，但由于多语言分词方式存在差异，在仅解码器的大语言模型中无法显式地对齐表示。然而，越来越多的研究表明，即使在大语言模型中，更高的跨语言表示对齐也能带来更好的跨语言迁移能力。在本文中，我们提出了一种新方法，在现代大语言模型的架构约束下重新构想跨语言对比学习。我们没有在隐藏状态上应用辅助对齐损失，而是提出使用混合专家（MoE）路由器的输出作为对齐目标。路由器输出更适合在大量词元上进行池化，从而实现更可靠的序列级跨语言比较。在四个开源MoE模型上进行的受控持续预训练实验表明，引入这种路由损失能够带来效果提升。

    arXiv:2610.01921v1 Announce Type: cross  Abstract: Cross-lingual contrastive learning has been a core component of multilingual encoder training, but the ability to explicitly align representations is not possible in decoder-only LLMs because of varying multilingual tokenization. However, a growing amount of research suggests that even in LLMs, higher cross-lingual representational alignment leads to improved cross-lingual transfer. In this paper, we propose a novel approach to reimagine cross-lingual contrastive learning given the architectural constraints of modern LLMs. Rather than applying an auxiliary alignment loss on hidden states, we propose using the outputs of the mixture-of-experts (MoE) routers as the target for alignment. Router outputs lend themselves better to pooling over many tokens, enabling more reliable cross-lingual comparisons at the sequence-level. Controlled continual pre-training experiments on four open-source MoEs show that incorporating this routing loss als
    
[^264]: CONTRA：发现并评估可改变程序行为的问题，用于大语言模型代码生成中的选择性澄清

    CONTRA: Discovering and Qualifying Behavior-Changing Questions for Selective Clarification in LLM Code Generation

    [https://arxiv.org/abs/2610.01769](https://arxiv.org/abs/2610.01769)

    CONTRA是一种无需训练的方法，通过广泛发现候选澄清问题，并结合语义评估与基于执行的验证来筛选出真正会改变代码行为的关键问题，从而让LLM代码生成智能体进行选择性澄清提问，在防止因假设错位导致行为偏差的同时避免不必要的打扰。

    

    编码智能体可能生成看似正确、但实际实现了用户从未预期的行为的代码。当智能体通过自身假设默默地去解决欠明确的需求时，就会产生这种不匹配。随着后续开发建立在这些假设之上，纠正由此产生的行为的成本会越来越高。尽早进行澄清提问有助于防止此类不匹配，但不必要的问题会打断开发者并拖慢开发进度。现有方法难以在识别关键澄清问题的同时避免提出不必要的问题。因此，我们提出了CONTRA，这是一种无需训练的方法，它将广泛的问题发现与基于语义和基于执行的问题评估相结合。CONTRA首先生成候选问题，并过滤掉与所需行为无关、或已被需求内容所解决的问题。对于剩下的每个问题，它会基于两个合理的答案分别生成程序并进行检查……

    arXiv:2610.01769v1 Announce Type: new  Abstract: Coding agents can generate code that appears correct but implements behavior the user never intended. This mismatch can arise when an agent silently resolves underspecified requirements through its own assumptions. As subsequent development builds on these assumptions, correcting the resulting behavior can become increasingly costly. Early clarification can help prevent such mismatches, but unnecessary questions can interrupt developers and slow down development. Existing methods struggle to identify key clarification questions while avoiding unnecessary ones. Therefore, we propose CONTRA, a training-free method that combines broad question discovery with semantic and execution-based question qualification. CONTRA first generates candidate questions and filters out those unrelated to required behavior or already resolved by the requirement. For each remaining question, it generates programs conditioned on two plausible answers and checks
    
[^265]: 从后验集中性重新思考基于概率的强化学习

    Rethinking Probability-Based Reinforcement Learning From Posterior Concentration

    [https://arxiv.org/abs/2610.01458](https://arxiv.org/abs/2610.01458)

    该论文发现基于概率的奖励存在后验集中现象，即随着推理链变长奖励会坍缩到低方差区间而难以区分，导致GRPO训练不稳定且低效，并提出显式建模该现象的无验证器强化学习框架RLCPR，以提升优化稳定性和token效率。

    

    基于概率奖励的无验证器强化学习，为在外部验证器不可用的场景下训练大语言模型完成通用推理任务提供了一种有前景的途径。然而，这类奖励的可靠性——尤其是在长程推理中的可靠性——仍缺乏充分研究。本工作识别出概率奖励中一种与长度相关的失效模式，我们称之为后验集中现象。我们证明，随着推理轨迹变长，参考答案在给定该推理轨迹条件下的概率往往会坍缩到一个低方差区间。这一现象导致奖励几乎无法区分，在基于GRPO的设置下，会使基于概率的策略优化变得不稳定且低效。受此启发，我们提出了集中感知后验奖励强化学习，这是一个无验证器的强化学习框架，通过显式考虑后验集中现象来获得更好的优化稳定性和token效率。

    arXiv:2610.01458v1 Announce Type: new  Abstract: Verifier-free reinforcement learning with probability-based rewards offers a promising way to train LLMs on general reasoning tasks where external verifiers are unavailable. Yet the reliability of these rewards, especially in long-horizon reasoning, remains underexplored. This work identifies a length-dependent failure mode of probability rewards, which we call the Posterior Concentration Phenomenon (PCP). We show that the probability of a reference answer conditioned on a reasoning trace often collapses to a low-variance interval as the trace becomes lengthy. This phenomenon results in nearly indistinguishable rewards, which, under GRPO-based settings, makes probability-based policy optimization unstable and inefficient. Motivated by this, we propose Reinforcement Learning with Concentration-aware Posterior Rewards (RLCPR), a verifier-free RL framework to explicitly account for PCP for better optimization stability and token efficiency.
    
[^266]: Fold'EM：从冷冻电镜颗粒直接推断原子结构

    Fold'EM: Direct atomic structure inference from Cryo-EM particles

    [https://arxiv.org/abs/2610.01358](https://arxiv.org/abs/2610.01358)

    本文提出Fold'EM方法，无需先进行密度重建即可直接从冷冻电镜颗粒图像推断原子结构，从而降低样本复杂度并提升结构测定的效率。

    

    单颗粒冷冻电子显微镜（cryo-EM）已成为生物分子结构测定中广泛采用的技术。传统的冷冻电镜计算流程首先将大量颗粒图像组合起来重建静电势（ESP）图，然后将原子模型拟合到恢复的密度图上。密度重建具有较高的样本复杂度，需要大量的颗粒图像，使得结构测定成本高昂且通量较低，对于异质性样品尤其如此。而随着重建密度图分辨率的下降，下游的原子模型构建也变得越来越困难。蛋白质结构预测模型提供了基于氨基酸序列的强原子结构先验，实验引导的方法可以利用这些先验来恢复与实验测量相一致的结构。然而，在冷冻电镜中，此类先验通常仅在密度重建完成之后才被整合（摘要在此处截断）。

    arXiv:2610.01358v1 Announce Type: cross  Abstract: Single-particle cryo-electron microscopy (cryo-EM) has become a widely adopted technique for biomolecular structure determination. The conventional cryo-EM computational pipeline first combines many particle images to reconstruct an electrostatic potential (ESP) map and then fits an atomic model to the recovered map. Density reconstruction has high sample complexity, requiring large numbers of particle images and making structure determination high-cost and low-throughput, particularly for heterogeneous samples. Downstream atomic model building, in turn, becomes increasingly difficult as the resolution of the reconstructed map deteriorates. Protein structure prediction models provide strong sequence-derived priors on atomic structure, and experiment-guided approaches can use these priors to recover structures consistent with experimental measurements. Yet, in cryo-EM, such priors are typically integrated only after density reconstructi
    
[^267]: 螺旋注意力：Transformer中的刚体代数

    Screw Attention: Rigid-Body Algebra Inside a Transformer

    [https://arxiv.org/abs/2610.00904](https://arxiv.org/abs/2610.00904)

    提出螺旋注意力层，将token间关系建模为空间变换与关节螺旋，使消息传递天然具有坐标系等变性，且单层即可表达刚体力学速度递归，在仿真操作任务上达到或超越同规模基线方法。

    

    学习得到的操作策略需要从数据中重新发现刚体力学本身就能以闭式形式提供的空间关系。这不仅耗费数据，还使得策略对场景中的几何变化十分脆弱。我们提出了螺旋注意力，这是一种Transformer层，其中两个物体之间的关系被表示为空间变换而非图的一条边。每个token都是一个具有位姿的物体，每对token携带相对位姿，对于机器人关节还携带关节螺旋。消息沿着这种关系被传输到接收方的坐标系中，而注意力分数只使用坐标系不变的量。通过构造，消息对每个token的独立坐标系变化具有等变性，并且单层即可表达刚体力学的速度递归。在仿真操作任务上，螺旋注意力达到或超越了同等规模的对照方法，包括在LIBERO-Spatial上的图网络、Transformer和扁平网络。

    arXiv:2610.00904v1 Announce Type: cross  Abstract: Learned manipulation policies rediscover from data the spatial relations that rigid-body mechanics supplies in closed form. This costs data, and it leaves the policies fragile to geometric changes in the scene. We present Screw Attention, a transformer layer in which the relation between two bodies is a spatial transform rather than a graph edge. Every token is a body with a pose. Each pair of tokens carries the relative pose and, for robot joints, the joint screw. Messages are transported along this relation into the receiver's frame, while the attention scores see only frame-invariant quantities. By construction, the messages are equivariant to an independent change of frame at every token, and a single layer can express the velocity recursion of rigid-body mechanics. On simulated manipulation tasks, Screw Attention matches or outperforms controls of the same size, including graph, transformer and flat networks on LIBERO-Spatial. Wit
    
[^268]: Cogentic：用于自动证明发现的多智能体编排框架

    Cogentic: Multi-Agent Orchestration for Automated Proof Discovery

    [https://arxiv.org/abs/2609.40324](https://arxiv.org/abs/2609.40324)

    Cogentic 通过编排器调度多个独立证明者、多组件对抗性验证以及持久化已验证账本的迭代“证明—验证”循环，实现了开放研究问题上的自动证明发现，并以 Gemini 为基础模型获得了新研究成果。

    

    我们提出 Cogentic，一个面向开放研究问题的自动证明发现多智能体框架。虽然前沿语言模型能够一次性生成出色的数学想法，但对于需要探索多个相互竞争的猜想、克服细微技术障碍，并在长时间跨度中保留中间进展的开放性问题而言，单次生成往往是不够的。Cogentic 通过迭代的“证明—验证”循环来应对这些挑战：由一个编排器将一群独立的证明者分配到不同的证明方向上，将其输出交由多个专门组件进行对抗性验证，并将经确认的中间结果存入一个持久的已验证账本，供后续轮次在此基础上继续推进。该框架旨在能够解决研究级数学与理论计算机科学问题。以 Gemini 作为基础模型，Cogentic 在（摘要原文在此处截断）

    arXiv:2609.40324v1 Announce Type: new  Abstract: We present Cogentic, a multi-agent harness for automated proof discovery on open research problems. While frontier language models can generate strong mathematical ideas in a single shot, single-shot generation is often insufficient for open problems that require exploring multiple competing conjectures, overcoming subtle technical obstructions, and retaining intermediate progress over a long horizon. Cogentic addresses these challenges through an iterative prove--verify loop in which an orchestrator allocates a population of independent provers across distinct proof directions, subjects their output to adversarial verification by several specialized components, and promotes confirmed intermediate results into a persistent verified ledger that later rounds build on. The harness is designed to be able to solve research-level math and theoretical computer science problems. Using Gemini as the base model, Cogentic produced novel results on 
    
[^269]: 从历史动作轨迹中学习技能：面向世界动作模型的动作经验字典

    Learning Skills from Historical Action Trajectories: Action Experience Dictionary for World Action Models

    [https://arxiv.org/abs/2609.40219](https://arxiv.org/abs/2609.40219)

    该论文提出动作经验字典（AED），将历史动作轨迹编码为共享动作嵌入，使世界动作模型能够复用技能并建模跨任务语义关系，从而提升操作任务的动作生成能力。

    

    世界动作模型（WAMs）将视觉动力学预测与动作生成相结合，但其并未显式支持跨操作任务的动作经验复用。此外，现有的世界动作模型难以捕捉能够指导目标动作预测的潜在跨任务语义关系，因为冗余的背景元素会干扰关键视觉信息的提取。为应对这些挑战，我们提出了一种新颖的动作经验字典（AED），它将历史物理动作轨迹编码为共享的动作嵌入，以支持技能复用并建模跨任务关系。具体而言，我们首先聚合历史动作使其与视觉观测对齐，并使用预训练的动作分词器从AED中检索动作嵌入。随后，我们通过交叉注意力机制对池化后的嵌入进行视觉条件化，并将其前置于含噪动作标记之前，为交互提供上下文信息。

    arXiv:2609.40219v1 Announce Type: cross  Abstract: World Action Models (WAMs) couple visual dynamics prediction with action generation, yet they do not explicitly support the reuse of action experience across manipulation tasks. Furthermore, existing WAMs struggle to capture underlying cross-task semantic relationships that could guide target action prediction, as redundant background elements interfere with the extraction of key visual information. To address these challenges, we develop a novel Action Experience Dictionary (AED) that encodes historical physical action trajectories into shared action embeddings to support skill reuse and model cross-task relationships. Specifically, we first aggregate historical actions to align with visual observations and retrieve action embeddings from the AED using a pretrained action tokenizer. Subsequently, we visually condition the pooled embeddings through cross-attention and prepend them to noisy action tokens, providing interaction context a
    
[^270]: 触觉好奇心驱动机器人交互

    Tactile Curiosity Drives Robot Interaction

    [https://arxiv.org/abs/2609.40134](https://arxiv.org/abs/2609.40134)

    本文提出TacEx框架，将模型不确定性按感官模态分解并把好奇心引向触觉通道，使强化学习探索以触觉反馈为导向，从而提升机器人操作技能学习的样本效率。

    

    通过强化学习（RL）掌握机器人操作技能在很大程度上仍然存在样本效率低下的问题。最常见的RL算法依赖随机动作采样来发现新策略，导致智能体将大部分训练预算花费在自由空间的运动上，远离了操作技能得以形成的接触。现有基于模型分歧或认知不确定性的内在动机方法虽然优于各向同性噪声，但它们也可能在功能上无关的状态转移中奖励不确定性，例如自由空间中的无规则运动。在这项工作中，我们认为触觉反馈为探索提供了一种天然的信号，并提出了TacEx——一个将触觉融入认知不确定性驱动探索的框架。该框架通过将模型不确定性分解到不同感官模态，并将好奇心引导至触觉通道来实现这一点。通过将好奇心锚定在触觉感知上，TacEx驱动机器人……

    arXiv:2609.40134v1 Announce Type: cross  Abstract: Mastering robot manipulation skills via reinforcement learning (RL) remains largely sample-inefficient. The most common RL algorithms rely on random action sampling to discover new strategies, resulting in agents that allocate most of their training budget to motions in free space, away from the contacts from which manipulation skills emerge. Existing intrinsic motivation methods based on model disagreement or epistemic uncertainty improve on isotropic noise, but they can also reward uncertainty in functionally irrelevant transitions, such as erratic motions in free space. In this work, we argue that tactile feedback provides a natural signal for exploration, and introduce TacEx, a framework that incorporates touch into epistemic uncertainty-driven exploration by decomposing model uncertainty across sensory modalities and directing curiosity toward the tactile channel. By anchoring curiosity to the sense of touch, TacEx drives the robo
    
[^271]: PTNO：利用含噪蒙特卡洛估计训练神经算子以解决粒子输运问题

    PTNO: Training Neural Operators with Noisy Monte Carlo Estimates for Particle Transport Problems

    [https://arxiv.org/abs/2609.40090](https://arxiv.org/abs/2609.40090)

    该论文提出粒子输运神经算子PTNO，可直接从含噪、低成本的蒙特卡洛标签中学习粒子输运代理模型，并证明了无偏噪声标签的平方损失与收敛解损失共享同一极小值点，从而解决了高方差与高动态范围两大挑战，大幅降低了训练成本。

    

    在多次散射下的粒子输运是辐射转移和等离子体物理的核心问题，然而高保真的蒙特卡洛（MC）模拟必须追踪数量极其庞大的粒子。基于学习的代理模型可以摊销这一成本，但通常需要在昂贵且充分收敛的MC解上进行训练。我们提出了粒子输运神经算子（PTNO），这是一种能够直接从含噪、低成本的MC标签中学习粒子输运代理模型的神经算子。这类标签带来了两大挑战：（1）高方差会使标准监督学习不稳定；（2）跨越多个数量级的高动态范围（HDR）。针对第一个挑战，我们从大量构型的含噪标签中学习解算子，从而摊销MC成本并泛化到未见过的构型。由于MC标签是无偏的，我们证明了基于这些标签的平方损失与基于收敛解的损失具有相同的极小值点，并且我们对训练的预算分配研究表明……

    arXiv:2609.40090v1 Announce Type: new  Abstract: Particle transport under multiple scattering is central to radiative transfer and plasma physics, yet high-fidelity Monte Carlo (MC) simulations must trace prohibitively many particles. Learning-based surrogates can amortize this cost, but typically train on expensive, well-converged MC solutions. We propose the Particle Transport Neural Operator (PTNO), a neural operator that learns particle transport surrogates directly from noisy, low-cost MC labels. Such labels pose two challenges: (1) high variance, which destabilizes standard supervised learning, and (2) a high dynamic range (HDR) spanning many orders of magnitude. For the first, we learn the solution operator from noisy labels of many configurations, amortizing MC cost and generalizing to unseen configurations. Because MC labels are unbiased, we show that the squared loss on them shares its minimizer with the loss on converged solutions, and our budget-allocation study over traini
    
[^272]: 混合整数线性与非线性规划中的自动化研究

    Autoresearch in Mixed-Integer Linear and Nonlinear Programming

    [https://arxiv.org/abs/2609.39360](https://arxiv.org/abs/2609.39360)

    提出AutoMIP——一种通过想法池与算法树搜索来组织混合整数规划长周期自动化研究的可复用智能体技能，在MILP和MINLP基准测试中取得了所评估框架中最高的成功率。

    

    尽管自动化研究近来取得了进展，但将其应用于实际的运筹学问题——通常被表述为NP难的混合整数线性或非线性规划（MILP或MINLP）——仍然具有挑战性，因为有效的研究需要系统地管理相互竞争的想法和长周期的实验轨迹。我们提出了AutoMIP，这是一种可复用的智能体技能，通过想法池和算法树搜索来组织混合整数规划中的长周期自动化研究。AutoMIP维护一个持久的互补候选想法池，同时将可执行的实验组织成算法树，使智能体能够保留未探索的假设、改进有前景的算法，并根据历史状态切换到替代的方法论方向。在MILP和MINLP基准测试集上，AutoMIP在所评估的自动化研究框架中取得了最高的最终成功率。在MIPLib上，AutoMIP发现了新的（摘要原文在此处截断）

    arXiv:2609.39360v1 Announce Type: new  Abstract: Despite recent progress in autoresearch, applying it to practical operations research problems, typically formulated as NP-hard mixed-integer linear or nonlinear programs (MILPs or MINLPs), remains challenging because effective research requires systematically managing competing ideas and long-horizon experimental trajectories. We introduce AutoMIP, a reusable agent skill for organizing long-horizon autoresearch in mixed-integer programming through idea pooling and algorithm tree search. AutoMIP maintains a persistent pool of complementary candidate ideas while organizing executable experiments into an algorithm tree, enabling the agent to preserve unexplored hypotheses, refine promising algorithms, and switch to alternative methodological directions based on historical states. On MILP and MINLP benchmark cohorts, AutoMIP achieves the highest final success rates among the evaluated autoresearch frameworks. On MIPLib, AutoMIP discovers ne
    
[^273]: CRAFT：医学视觉语言模型中的因果责任与失败追踪

    CRAFT: Causal Responsibility and Failure Tracing in Medical Vision Language Models

    [https://arxiv.org/abs/2609.38810](https://arxiv.org/abs/2609.38810)

    该研究揭示了医学视觉语言模型中“仲裁失败”（文本覆盖视觉依据）与“刹车失败”（证据不足仍作答）两种安全风险分别由空间上不重叠的注意力头群体介导——仲裁头分布于中深层宽频带、刹车头集中于中后层窄频带，从而实现对模型失败的因果追踪与定位。

    

    随着视觉语言模型越来越多地被部署于临床诊断，理解它们在内部如何解决相互竞争的视觉与文本信号已成为一个安全上的必要课题。现有的机制分析仍局限于单模态文本，无法解释为什么单句误导性文本能够覆盖基于图像的正确诊断，也无法解释为什么模型在视觉证据不足的情况下仍会给出自信的答案。我们发现这两种安全风险——文本上下文覆盖视觉依据的“仲裁失败”，以及模型在证据不充分时就给出结论的“刹车失败”——分别由空间上互不重叠的注意力头群体介导：仲裁头形成一个中到深层的宽频带，反映跨层的证据竞争；而刹车头则集中在狭窄的中后层频带中，负责调节证据充分性与弃权行为。为了将这些观察建立在因果回路上，我们……

    arXiv:2609.38810v1 Announce Type: cross  Abstract: As vision language models are increasingly deployed in clinical diagnosis, understanding how they internally resolve competing visual and textual signals becomes a safety imperative. Existing mechanistic analyses remain confined to unimodal text and offer no explanation for why a single misleading sentence can override a correct image based diagnosis, or why a model commits to a confident answer despite insufficient visual evidence. We find that these two safety risks, arbitration failure where textual context overrides visual grounding and brake failure where the model commits without adequate evidence, are mediated by spatially disjoint attention head populations: arbitration heads form a mid-to-deep wideband reflecting cross-layer evidence competition, while brake heads concentrate in a narrow middle-to-late layer band that regulates evidence sufficiency and abstention behavior. To ground these observations in causal circuitry, we i
    
[^274]: 通过多步嵌入检索学习在视觉空间中进行路由

    Learning to Route in Visual Space via Multi-Step Embedding Retrieval

    [https://arxiv.org/abs/2609.38743](https://arxiv.org/abs/2609.38743)

    该论文提出VHOP基准框架和VHOP-Router端到端训练流程（结合监督微调、在线模仿学习与强化学习），将标准嵌入模型改造为能直接在嵌入空间中完成多步视觉导航的检索工具，从而突破LLM智能体视觉搜索中单步检索的性能瓶颈。

    

    LLM智能体依赖检索工具来访问外部知识，然而视觉智能体搜索仍然严重受限于标准的单步检索器。在现有流程中，智能体必须为每个中间步骤发出文本查询，当视觉线索难以用文字描述、或检索器无法在其前列结果中呈现必要的中间证据时，搜索就会陷入困境。我们假设，将跨越整个嵌入空间的多步导航直接交由检索工具完成，可以解决这一性能瓶颈。为了系统地研究这一问题，我们提出了VHOP——一个灵活的数据生成框架与基准，包含五个核心难度级别，同时测试视觉匹配与搜索规划能力。利用该框架，我们开发了VHOP-Router——一个端到端的训练流程，结合监督微调、在线模仿学习与强化学习，将标准嵌入模型转变为……

    arXiv:2609.38743v1 Announce Type: new  Abstract: LLM agents rely on retrieval tools to access external knowledge, yet visual agentic search remains severely bottlenecked by standard single-step retrievers. In current pipelines, the agent must issue text queries for every intermediate step, struggling when visual clues are difficult to describe or when the retriever fails to surface necessary intermediate evidence within its top results. We hypothesize that offloading multi-step navigation across the entire embedding space directly to the retrieval tool resolves this performance bottleneck. To study this systematically, we introduce VHOP, a flexible data generation framework and benchmark with five core difficulty levels testing both visual matching and search planning. Using this framework, we develop VHOP-Router, an end-to-end training pipeline---combining supervised fine-tuning, online imitation learning, and reinforcement learning---that transforms a standard embedding model into an
    
[^275]: 从个体学习到社会学习：刻画大语言模型中的递归社会改进

    From Solo to Social Learning: Characterizing Recursive Social Improvement in LLMs

    [https://arxiv.org/abs/2609.38516](https://arxiv.org/abs/2609.38516)

    该论文提出“递归社会改进”这一新概念，并发现尽管经典社会学习算法能从同伴中受益，但当每个LLM智能体各自追求自身奖励时，当前的LLM无法通过相互学习改进整个群体，其每token收益反而低于独立学习。

    

    大型语言模型（LLM）如今可以通过修改自身遵循的指令来实现自我改进，同时LLM智能体也越来越多地被组织起来协同解决复杂问题。然而，自我改进方法通常一次只优化一个系统，而多智能体框架往往让每个模型朝着同一个共同目标努力。我们提出了一个不同的问题：当每个智能体追求自己的奖励时，自我改进的LLM能否从彼此身上充分学习，从而改进整个群体？我们将这种能力称为递归社会改进。我们研究了会修改技能文件、并自主决定是否、何时以及向谁进行模仿学习的智能体群体。其中，独立搜索、向同伴学习和执行动作共享同一个token预算。在受控环境中，现有的社会学习算法能够从同伴中获益，但三种LLM却不能：它们每token获得的奖励低于单独学习的个体，要么探索范围过窄，要么在采取行动之前就耗尽了token。

    arXiv:2609.38516v1 Announce Type: cross  Abstract: Large language models (LLMs) can now improve themselves by revising the instructions they follow, and LLM agents are increasingly orchestrated to work together on complex problems. However, self-improvement methods typically optimize one system at a time, and multi-agent frameworks often have every model work toward a shared goal. We ask a different question. When each agent pursues its own reward, can self-improving LLMs learn from one another well enough to improve the whole population? We call this capability recursive social improvement. We study populations that revise skill files and choose whether, when, and whom to copy from. Independent search, learning from peers, and acting all share one token budget. In controlled environments, established social-learning algorithms benefit from peers, but three LLMs do not. They earn less reward per token than solo learners, and explore too narrowly or run out of tokens before acting. We t
    
[^276]: 可审计的长期记忆：在LongMemEval-S上测得479/475（满分500）成绩的确定性检索链

    Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S

    [https://arxiv.org/abs/2609.38021](https://arxiv.org/abs/2609.38021)

    该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。

    

    我们在LongMemEval-S上评估了一个可审计的长期记忆系统。其检索链采用混合候选检索、交叉编码器重排序、以覆盖优先的数据包编译以及确定性推理脚手架；大语言模型仅作为可替换的最终阅读器使用。该检索链在470个可回答问题中的468个上将所有金标准会话纳入候选池，并为其中462个生成金标准完整的数据包。使用通过未固定版本的CLI别名调用的Claude Opus阅读器，在GPT-4o评分下，两次500题评测分别获得479/500和475/500的分数。其中72个可回答的知识更新题目使用了经过实质性修改的评分提示词，该修改在官方提示词文本下的效果尚未被测量。这一对结果跨越了Chronos High已发表的478/500；由于阅读器生成方式、评分提示词、可能的数据版本差异，以及系统内部的方差，这些结果既不能确立优越性，也不能确立等效性。在同一数据包上使用grok-4.6-high阅读器的得分为476/474，而……

    arXiv:2609.38021v1 Announce Type: cross  Abstract: We evaluate an auditable long-term memory system on LongMemEval-S. Its retrieval chain uses hybrid candidate retrieval, cross-encoder reranking, coverage-first packet compilation, and deterministic reasoning scaffolds; an LLM is used only as a replaceable final reader. The chain places all gold sessions in the candidate pool for 468/470 answerable questions and produces gold-complete packets for 462/470. With a Claude Opus reader called through an unpinned CLI alias, two 500-question passes score 479/500 and 475/500 under GPT-4o. The 72 answerable knowledge-update rows used a substantively modified scoring prompt whose effect under the official text has not been measured. The pair straddles Chronos High's published 478/500; differences in reader generation, scoring prompt, and possibly data version, plus within-system variance, establish neither superiority nor equivalence. A grok-4.6-high reader on the same packets scores 476/474, whi
    
[^277]: WISE-ATTA：预算受限的主动测试时自适应中何时请求标签

    WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation

    [https://arxiv.org/abs/2609.37687](https://arxiv.org/abs/2609.37687)

    提出了“预算受限主动测试时自适应”这一新设定，并给出WISE-ATTA方法，利用在线计算的轻量级信号决定何时对测试批次施加监督，将ATTA的核心挑战从“标注什么”转变为“何时标注”。

    

    主动测试时自适应通过在推理过程中更新已部署的模型并有选择地查询监督信息，从而提升模型在分布偏移下的鲁棒性。然而，大多数现有的ATTA方法都隐含地假设可以对每个到来的测试批次请求监督信息，这在长测试流上会产生巨大的标注成本。在本工作中，我们引入了“预算受限的ATTA”这一新设定，其中标签仅对一小部分测试批次可用。这一设定将核心挑战从决定批次内“标注什么”转变为决定在时间维度上“何时”施加监督。为应对这一挑战，我们提出了一种预算感知的方法WISE-ATTA，它基于在线计算的轻量级信号在测试流上分配监督资源，优先选择监督最可能发挥作用的时期。当某个批次被选中进行监督时，我们进一步采用基于漂移的样本（策略）……

    arXiv:2609.37687v1 Announce Type: new  Abstract: Active test-time adaptation (ATTA) improves robustness under distribution shift by updating a deployed model during inference while selectively querying supervision. However, most existing ATTA methods implicitly assume that supervision can be requested for every incoming test batch, which can incur substantial annotation cost over long test streams. In this work, we introduce \emph{budgeted ATTA} in which labels are available for only a fraction of test batches. This formulation shifts the central challenge from deciding \emph{what} to label within a batch to deciding \emph{when} supervision should be applied over time. To address this challenge, we propose a budget-aware approach \emph{WISE-ATTA} that allocates supervision over the test stream based on lightweight signals computed online, prioritizing periods where supervision is likely to be most useful. When a batch is selected for supervision, we further employ a drift-based sample 
    
[^278]: 从学习者行为到可复用技能：实现高效且有效的学习者模拟

    From Learner Behavior to Reusable Skills for Effective and Efficient Learner Simulation

    [https://arxiv.org/abs/2609.37157](https://arxiv.org/abs/2609.37157)

    Learner2Skill将从历史交互中获得的学习者模拟能力外化为持久可复用的“模拟技能”，该技能捕获学习者的学习状态与回答模式并随新交互演进，可通过轻量级校准迁移到新的大语言模型，从而更高效、更忠实地模拟学习者行为。

    

    学习者模拟旨在重现特定学习者在新任务上的行为表现。尽管大语言模型（LLMs）能够生成日益精细的学习行为，但现有方法往往需要反复处理不断增长的交互历史来重建学习者。这带来了额外的上下文和推理成本，并且使得所获得的学习者特定模拟能力难以在不同的大语言模型之间复用。因此，我们提出Learner2Skill，它将从历史交互中获得的模拟能力外化为一个持久且可复用的模拟技能（Simulation Skill）。该技能捕获学习者当前的学习状态和反复出现的回答模式，随着新的真实交互的到来而不断演进，并且可以通过轻量级的执行器校准适配到新的大语言模型，而无需从头重建学习者。实验表明，Learner2Skill能够更忠实地重现细粒度的学习行为（摘要原文在此处截断）。

    arXiv:2609.37157v1 Announce Type: new  Abstract: Learner simulation aims to reproduce how a particular learner behaves on new tasks. Although Large Language Models (LLMs) can generate increasingly fine-grained learning behaviors, existing approaches often need to repeatedly process a growing interaction history to reconstruct the learner. This introduces additional context and inference costs and makes the acquired learner-specific simulation capability difficult to reuse across different LLMs. We therefore propose Learner2Skill, which externalizes the simulation capability acquired from historical interactions into a persistent and reusable Simulation Skill. The Skill captures the learner's current learning state and recurring response patterns, evolves as new real interactions arrive, and can be adapted to a new LLM through lightweight executor calibration without reconstructing the learner from scratch. Experiments show that Learner2Skill more faithfully reproduces fine-grained lear
    
[^279]: ARGOS：面向计算连续体服务编排的强化学习驱动多维弹性

    ARGOS: Reinforcement Learning-Driven Multidimensional Elasticity for Service Orchestration in the Computing Continuum

    [https://arxiv.org/abs/2609.37085](https://arxiv.org/abs/2609.37085)

    ARGOS提出了一种基于强化学习的端到端控制器，将计算连续体中的多维弹性建模为按请求的马尔可夫决策过程，在容量达到上限时动态调整分析质量以吸收需求变化和集群压力，同时保障客户端定义的质量范围。

    

    计算连续体中的数据密集型服务必须在容量有限且不均衡的异构节点上平衡分析质量、资源使用和成本。当资源扩展达到容量上限时，这种平衡变得尤为困难，因为此时必须在不违反客户端定义质量范围的前提下吸收需求变化和集群压力。现有编排器主要调整资源、部署位置或副本数量，而覆盖率、采样和新鲜度等分析需求则保持固定。本文提出ARGOS（自适应强化学习驱动的编排服务治理），这是一种端到端控制器，将多维弹性表述为以分析质量和集群压力为核心的按请求马尔可夫决策过程，并由容量感知准入机制提供支持。ARGOS在异构集群上于受控工作负载和时变多租户到达场景下进行了评估。

    arXiv:2609.37085v1 Announce Type: cross  Abstract: Data-intensive services in the Computing Continuum must balance analytics quality, resource usage, and cost across heterogeneous nodes with limited and uneven capacity. This balance becomes especially difficult when resource scaling reaches capacity limits, because changes in demand and cluster pressure must then be absorbed without violating client-defined quality ranges. Existing orchestrators mainly adapt resources, placements, or replicas, while analytics requirements such as coverage, sample, and freshness remain fixed. This article presents ARGOS, the Adaptive Reinforcement Learning-Driven Governance for Orchestrated Services, an end-to-end controller that formulates multidimensional elasticity as a per-request Markov decision process over analytics quality and cluster pressure, supported by capacity-aware admission. ARGOS is evaluated under controlled workloads and time-varying multi-tenant arrivals on a heterogeneous cluster. A
    
[^280]: 面向星载数据缩减的嵌入式双时相建筑物损毁评估

    Embedded Bi-Temporal Building Damage Assessment for On-Board Data Reduction

    [https://arxiv.org/abs/2609.37013](https://arxiv.org/abs/2609.37013)

    提出了一种基于YOLOX孪生检测器的双时相建筑物损毁评估流水线，通过将灾前参考图像压缩至64倍潜空间编码上传至卫星，并在星载端仅下传边界框与损毁类别等目标级产品而非完整场景，实现了天地双向数据量的大幅缩减。

    

    arXiv:2609.37013v1 公告类型：cross 摘要：自然灾害发生后对建筑物损毁进行快速评估对于支撑应急响应至关重要。对地观测卫星能够在灾害发生后不久获取相关影像，但其利用受到上行与下行链路容量以及地面处理延迟的限制。为此，我们提出了一种基于由YOLOX派生的孪生检测器构建的双时相建筑物损毁评估流水线，旨在天地链路的两端进行信息压缩。在地面端，灾前参考图像被编码到紧凑的潜空间中——压缩率高达64倍——并上行传输至卫星。在星载端，该参考图像与最新的灾后获取图像进行比对，使下行链路仅传输可执行的目标级产品（边界框和损毁类别），而非完整场景。这在两个方向上都减少了数据交换量，同时在xBD数据集上，经过强压缩的参考图像仍能保留大部分（性能）。

    arXiv:2609.37013v1 Announce Type: cross  Abstract: Rapid assessment of building damage after natural disasters is essential to support emergency response. Earth Observation satellites can acquire relevant imagery shortly after an event, but exploitation is limited by uplink and downlink capacity and by ground-processing latency. We address this with a bi-temporal building damage assessment pipeline built on a siamese detector derived from YOLOX, designed to compress information at both ends of the ground/space link. On the ground, pre-disaster reference images are encoded into a compact latent space -- compressed by up to a factor of 64 -- and uplinked to the satellite. On board, this reference is compared with a fresh post-disaster acquisition so that the downlink carries only actionable object-level products, bounding boxes and damage classes, instead of full scenes. This cuts the data exchanged in both directions, while on xBD the strongly compressed reference still preserves most o
    
[^281]: SafeCoEvo：在测试时协同演化安全线束与安全防护的LLM智能体框架

    SafeCoEvo: Co-Evolving Safety Harnesses and Guards for LLM Agents at Test-Time

    [https://arxiv.org/abs/2609.36580](https://arxiv.org/abs/2609.36580)

    提出SafeCoEvo框架，在测试时协同演化快速自适应的安全线束（S-Harness）与安全防护（S-Guard），利用积累的运行时经验持续提升LLM智能体应对未见任务安全风险的能力。

    

    部署在真实环境中的LLM智能体会持续遇到新任务和安全风险，而执行反馈通常只有在每个任务完成后才能获得。然而，现有的自进化方法普遍依赖于在固定且可反复访问的任务分布上进行多轮优化，这与真实部署中的测试时适应存在根本差异——在真实部署中，只能利用从过去任务中积累的经验来改进对未来未见任务的安全决策。为解决这一局限，我们提出了SafeCoEvo，一个面向LLM智能体安全的测试时“线束—防护”（Harness-Guard）协同演化框架，使外部安全系统能够基于积累的运行时经验持续自适应。SafeCoEvo在不同的时间尺度上联合提升两种互补的安全能力：S-Harness能快速将近期运行时经验外化为可更新的显式安全知识，从而及时地……（摘要原文在此处被截断）

    arXiv:2609.36580v1 Announce Type: new  Abstract: LLM agents deployed in real-world environments continually encounter new tasks and safety risks, while execution feedback typically becomes available only after each task is completed. However, existing self-evolving approaches commonly rely on multiple rounds of optimization over fixed and repeatedly accessible task distributions, fundamentally differing from test-time adaptation in real-world deployment, where only experience accumulated from past tasks can be used to improve safety decisions on future unseen tasks. To address this limitation, we propose SafeCoEvo, a test-time Harness-Guard co-evolution framework for LLM agent safety that enables the external safety system to continually adapt from accumulated runtime experience. SafeCoEvo jointly improves two complementary safety capabilities at different timescales: S-Harness rapidly externalizes recent runtime experience into updatable explicit safety knowledge that can promptly inf
    
[^282]: 从迁移到校准：跨模型、司法管辖区与规模保持智能体能力

    From Migration to Calibration: Preserving Agent Capabilities across Models, Jurisdictions, and Scale

    [https://arxiv.org/abs/2609.35149](https://arxiv.org/abs/2609.35149)

    提出将智能体校准构建为标准优先的适配框架，通过定义基础能力、技术环境与用户情境标准，系统诊断差距并应用修订，从而在模型更换、跨管辖区部署和规模化过程中保持智能体能力不退化。

    

    部署、迁移或扩展智能体可能会改变其模型、运行框架、基础设施、应用场景及目标用户。我们将智能体校准表述为标准优先的适配方法：定义基础能力、技术环境和用户情境标准；诊断差距；生成并应用修订；并在固定预算内重新检查相同标准。这些标准族在信息层、运行框架层和用户接受层之间相互作用。源行为仅具诊断价值，并非完美的参考或能力上限：模型替换可能将正确答案变为错误，或将错误变为正确。资格认定要求通过所有强制性已知测试、真实的端到端部署路径、硬性谓词以及声明的任务/用户最低标准；总体收益不能抹去硬性失败。修订可能更改工具或运行框架，添加示范和任务描述，或使用经验证的目标原生轨迹来训练策略

    arXiv:2609.35149v2 Announce Type: replace  Abstract: Deploying, migrating, or scaling an agent can change its model, harness, infrastructure, application, and intended users. We formulate agent calibration as standards-first adaptation: define basic-capability, technical-environment, and user-context standards; diagnose gaps; generate and apply revisions; and recheck the same standards within fixed budgets. These standard families interact across information, harness, and user-acceptance layers. Source behavior is diagnostic, not a perfect reference or capability ceiling: model replacement can turn correct answers into errors or errors into correct answers. Qualification requires all mandatory known tests, actual end-to-end deployment paths, hard predicates, and declared task/user minimums to pass; aggregate gains cannot erase hard failures. Revisions may change tools or harnesses, add demonstrations and task descriptions, or use validated target-native trajectories to train a policy, 
    
[^283]: 自动化特征工程、AutoML与决策导向学习用于改进能源消耗预测

    Automated Feature Engineering, AutoML, and Decision-Focused Learning for Improved Energy Consumption Forecasting

    [https://arxiv.org/abs/2609.35013](https://arxiv.org/abs/2609.35013)

    本论文提出面向能源领域的自动化特征工程算法AutoEnergy，与AutoML集成实现端到端能源消耗预测建模，在18个真实数据集上将预测误差降低19.52%-84.72%。

    

    能源成本的上升与需求的增长，加之环境可持续性目标，给能源管理带来了重大挑战。能源消耗预测通过预测未来消耗来支持规划，但用于ECF的机器学习模型通常依赖于专家驱动的特征工程。本论文通过三项贡献解决了这种依赖性。首先，它建立并评估了用于ECF的全面特征工程流程，并研究了特定领域的特征。其次，它提出了AutoEnergy，这是一种针对领域定制化的自动化特征工程算法，能够从时间戳和滞后消耗数据中生成可解释的特征，并与AutoML集成以实现端到端的ECF建模。在涵盖住宅、商业、工业、可再生能源和电网领域的十八个真实世界能源数据集上，AutoEnergy相对于基线AutoML和已有的自动化特征工程方法，将预测误差降低了19.52%-84.72%。

    arXiv:2609.35013v2 Announce Type: replace  Abstract: The rising cost and demand for energy, together with environmental sustainability goals, create major challenges for energy management. Energy Consumption Forecasting (ECF) supports planning by predicting future consumption, but Machine Learning (ML) models for ECF often depend on expert-driven Feature Engineering (FE). This thesis addresses that dependence through three contributions. First, it establishes and evaluates a comprehensive FE pipeline for ECF and investigates domain-specific features. Second, it introduces AutoEnergy, a domain-tailored automated FE algorithm that generates interpretable features from timestamps and lagged consumption and integrates with AutoML for end-to-end ECF modelling. Across eighteen real-world energy datasets spanning residential, commercial, industrial, renewable, and grid domains, AutoEnergy reduces forecasting error by 19.52%-84.72% relative to baseline AutoML and established automated FE metho
    
[^284]: MASCIT：一种面向自然不规则时间序列的掩码感知状态空间分类器

    MASCIT: A Mask-Aware State Space Classifier for Naturally Irregular Time Series

    [https://arxiv.org/abs/2609.34409](https://arxiv.org/abs/2609.34409)

    提出掩码感知状态空间分类器MASCIT，通过观测掩码与门控时间聚合有效处理异步观测、缺失值等自然不规则性，在34个不规则时间序列数据集上取得最优聚合性能。

    

    自然不规则时间序列同时包含异步观测、缺失值、长度不等和非均匀采样等问题，而密集适配器可能会丢弃时间结构。我们提出了一种面向不规则时间序列的掩码感知状态空间分类器（MASCIT），它向编码器提供观测掩码，并在门控时间聚合中排除无效时间步。在34个不规则时间序列数据集上，MASCIT取得了最强的聚合点估计，并且是唯一在每个数据集上都提供了三种子运行结果的受评测神经模型。MASCIT在六个相互重叠的不规则性指标上均保持了最优的点排名，同时因子消融实验表明部分选择性优于完全选择性。这些结果支持选择性状态空间模型作为自然不规则时间序列分类的有效且可执行的骨干网络。

    arXiv:2609.34409v2 Announce Type: replace-cross  Abstract: Naturally irregular time series combine asynchronous observations, missing values, unequal lengths, and nonuniform sampling, while dense adapters can discard temporal structure. We propose a mask-aware state space classifier for irregular time series (MASCIT), which supplies observation masks to the encoder and excludes invalid steps from gated temporal aggregation. Across 34 irregular time series datasets, MASCIT yielded the strongest aggregate point estimate and was the only evaluated neural model with three-seed results on every dataset. MASCIT retained the lowest point rank across six overlapping irregularity indicators, while factorial ablations favored partial over full selectivity. These results support selective state space models as effective, executable backbones for naturally irregular time series classification.
    
[^285]: MaskCoFT：面向内存高效MoE推理的掩码协同自适应微调

    MaskCoFT: Masked Co-Adaptive Fine-Tuning for Memory-Efficient MoE Inference

    [https://arxiv.org/abs/2609.34077](https://arxiv.org/abs/2609.34077)

    提出MaskCoFT方法，利用可学习二值掩码限制每层的Top-K路由，并通过交叉熵损失协同微调路由器与专家，使专家在卸载推理场景下被高效复用，从而降低MoE模型的内存开销并保持推理性能。

    

    混合专家语言模型的参数规模常常超出单个GPU的内存容量。专家卸载技术将大多数专家保留在主机内存中并按需加载，因此解码速度取决于每个token需要获取的专家数量。缓存和预取只能在路由允许的范围内降低这一开销。仅微调路由器可以重塑路由以复用专家，但由于专家保持冻结，它们无法适应新路由发送给它们的token。我们提出MaskCoFT，一种掩码协同自适应微调方法，仅使用交叉熵损失同时训练路由器和专家。在微调过程中，一个可学习的二值掩码将每层的Top-K路由限制在一个专家子集内，专家则适应被重定向给它们的token。在推理时，学习到的掩码成为一种软先验，用于对专家进行重新排序，因此每个专家仍然可以被选择。我们为Mixtral-8（摘要截断）模拟了每层4个专家的GPU缓存进行实验。

    arXiv:2609.34077v2 Announce Type: replace-cross  Abstract: Mixture-of-experts (MoE) language models often exceed the memory of a single GPU. Expert offloading keeps most experts in host memory and loads them on demand, so decoding speed depends on how many experts each token must fetch. Caching and prefetching reduce this cost only as far as the routing allows. Router-only fine-tuning can reshape the routing to reuse experts, but it keeps the experts frozen, so they cannot adapt to the tokens the new routing sends them. We propose MaskCoFT, a masked co-adaptive fine-tuning method that trains routers and experts together with the cross-entropy loss alone. During fine-tuning, a learnable binary mask restricts the Top-K routing of each layer to a subset of experts, and the experts adapt to the tokens redirected to them. At inference, the learned mask becomes a soft prior that re-ranks experts, so every expert remains selectable. We simulate a GPU cache of 4 experts per layer for Mixtral-8
    
[^286]: T⁵：强化中期训练中面向词元级思维的双评论家训练

    $T^5$: Twin-Critic Training for Token-Level Thoughts in Reinforcement Mid-Training

    [https://arxiv.org/abs/2609.32791](https://arxiv.org/abs/2609.32791)

    提出双评论家方法T⁵，通过条件矩鞍点目标与信号保留约束，从单条生成轨迹中校准词元级优势，解决了强化中期训练中token级信用分配的高效性问题。

    

    强化中期训练使语言模型能够从未标注文本中学习内部思维，但高效的词元级信用分配仍然是一个挑战。现有的组相对方法需要代价高昂的重复生成。学习型评论家可以提供单次采样反馈，但仅有准确的回报预测并不能保证可靠的策略更新。我们的分析揭示了训练与推理的失配以及PPO裁剪如何阻碍优势估计中的共同偏移量相互抵消，从而引入额外的更新漂移。我们提出了T⁵，一种能够从单条生成轨迹中校准词元级优势的双评论家方法。在预热和留出集资格验证之后，两个评论家各自提供优势估计，并通过条件矩鞍点目标学习到的、依赖动作的权重将其组合。该目标使每个前缀处的平均优势趋于零，同时信号保留约束防止……（摘要原文在此处被截断）

    arXiv:2609.32791v2 Announce Type: replace  Abstract: Reinforcement mid-training lets language models learn internal thoughts from unlabeled text, but efficient token-level credit assignment remains challenging. Existing group-relative methods require costly repeated generation. Learned critics offer single-rollout feedback, but accurate return prediction alone does not ensure reliable policy updates. Our analysis shows how training--inference mismatch and PPO clipping prevent a common offset in advantage estimates from cancelling out, introducing additional update drift. We propose \tfour{}, a twin-critic method that calibrates token-level advantages from a single generated trajectory. After warmup and held-out qualification, the critics provide two advantage estimates, combined using action-dependent weights learned through a conditional-moment saddle-point objective. This objective brings the average advantage at each prefix toward zero, while a signal-retention constraint prevents t
    
[^287]: ProcGen 泛化差距究竟衡量了什么？动作规则、残余熵与缺失的随机下限

    What Does a ProcGen Generalization Gap Measure? Action Rules, Residual Entropy, and the Missing Random Floor

    [https://arxiv.org/abs/2609.32532](https://arxiv.org/abs/2609.32532)

    该论文提出强化学习的泛化差距应对照“随机下限”（均匀随机策略在同一评估框架和相同关卡上的回报）来解读，并证明测试时动作规则（采样与 argmax）的选择以及动作等效性造成的残余熵会显著改变 ProcGen 基准上泛化结论的含义。

    

    强化学习中的泛化差距（训练关卡上的回报减去留出关卡上的回报）通常在没有参考点的情况下被报告。我们认为，应当对照一个经过实际测量的随机下限来解读它：即在相同评估框架下，均匀随机策略在相同关卡上的回报。在八个 ProcGen 环境上使用 PPO，并在计算受限的预算下（800万步、16个并行环境；其中三个游戏扩展到2500万步），该随机下限改变了标准数字的含义。测试时的动作规则决定了所测量的是哪个策略：在 miner 中，采样策略在留出关卡上的得分是随机下限的5.1倍，而其 argmax 策略在每次运行中的得分都低于该下限，且贪婪评估使两个环境显著低于随机下限。作为收敛诊断指标，原始策略熵在八个环境中标记出六个，但该熵的32-66%来自具有相同效果的动作；对照随机下限，八个环境中采样策略有五个……

    arXiv:2609.32532v2 Announce Type: replace-cross  Abstract: A generalization gap in reinforcement learning, return on training levels minus return on held-out levels, is usually reported without a reference point. We argue that it should be read against a measured random floor: the return of a uniform-random policy on the same levels under the same evaluation harness. On eight ProcGen environments with PPO at a compute-limited budget (8M steps, 16 parallel environments; three games extended to 25M), the floor changes what standard numbers mean. The test-time action rule decides which policy is measured: in miner, the sampled policy scores 5.1x the floor on held-out levels while its argmax scores below it in every run, and greedy evaluation places two environments significantly below the floor. Used as a convergence diagnostic, raw policy entropy flags six of eight environments, but 32-66% of that entropy lies on actions with identical effects; against the floor, five of eight sampled po
    
[^288]: ADATEX4D：面向4D高斯泼溅的自适应纹理容量分配

    ADATEX4D: adaptive texture capacity allocation for 4D gaussian splatting

    [https://arxiv.org/abs/2609.29963](https://arxiv.org/abs/2609.29963)

    提出AdaTex4D自适应纹理容量分配模块，根据可见性和局部尺度动态调整每个高斯RGBA三平面的分辨率，在保持重建质量的同时将4D高斯泼溅的纹理存储减少一半以上。

    

    带纹理的高斯提升了局部外观表达能力，但为每个图元分配相同的纹理分辨率会在低细节或弱可见区域浪费存储空间。我们提出了AdaTex4D，这是一个面向基于变形的4D高斯泼溅的自适应纹理容量模块。每个高斯都携带打包的RGBA三平面，其两个轴的尺寸根据可见性归一化的屏幕空间梯度和变形后的局部尺度独立增长。在N3DV和PanopticSports数据集上的实验表明，AdaTex4D在保持重建质量的同时，将纹理存储减少了超过一半。在固定内存预算下，自适应分配相比均匀纹理分配还能提升质量，并降低整体模型大小和峰值内存。这些结果表明，动态、各向异性的纹理分配为在4D高斯表示中分配局部外观容量提供了一种更高效的方式。

    arXiv:2609.29963v1 Announce Type: cross  Abstract: Textured Gaussians improve local appearance capacity, but assigning the same texture resolution to every primitive wastes storage on low-detail or weakly visible regions. We introduce AdaTex4D, an adaptive texture-capacity module for deformation-based 4D Gaussian Splatting. Each Gaussian carries packed RGBA triplanes whose two axes grow independently according to visibility normalized screen-space gradients and deformed local scales. Experiments on N3DV and PanopticSports show that AdaTex4D reduces texture storage by more than half while preserving reconstruction quality. Under fixed memory budgets, adaptive allocation also improves quality over uniform texture assignment and reduces overall model and peak memory. These results show that dynamic, anisotropic texture allocation provides a more efficient way to distribute local appearance capacity in 4D Gaussian representations.
    
[^289]: 具有障碍物感知框架的安全机器人操作编码智能体

    Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation

    [https://arxiv.org/abs/2609.20822](https://arxiv.org/abs/2609.20822)

    本文发现编码智能体在机器人操作中因规划环节未能将安全约束设为优先事项而系统性碰撞障碍物（而非感知或指令问题），并通过将操作分解为路径阶段和接触时刻来定位失败根源，提出了障碍物感知框架以实现安全操作。

    

    编码智能体已成为机器人操作领域一种有前景的范式：语言模型将机器人控制器编写为程序，以这种方式构建的智能体现在无需机器人特定训练即可操作机器人。然而，这种范式是否安全，这一问题尚未被探讨。我们在安全约束下评估编码智能体，其中每个任务将操作目标与机器人不得触碰的障碍物配对。智能体追求目标，但在大多数情况下与障碍物发生碰撞，将任务完成视为唯一目标而忽视安全性。智能体在其推理轨迹中确实对障碍物进行了推理，且提示词已经禁止触碰障碍物，因此感知和指令都没有问题；问题出在规划环节，所陈述的约束从未成为优先事项。通过将操作分解为路径阶段和富含接触的时刻，我们定位了失败的根源。在路径阶段，模型无法对……（摘要原文在此处截断）

    arXiv:2609.20822v1 Announce Type: cross  Abstract: Coding agents have emerged as a promising paradigm for robot manipulation: a language model writes the robot controller as a program, and agents built in this way now operate robots without robot-specific training.Whether this paradigm is also safe, however, has not been asked. We evaluate coding agent under a safety constraint, where each task pairs a manipulation goal with an obstacle the robot must not touch. The agent pursues the goal but collides with the obstacle in most cases, treating task completion as its sole objective while neglecting safety. The agent reasons about the obstacle in its traces, and the prompt already forbids touching it, so neither perception nor instruction is at fault; the fault lies in the planning, where the stated constraint never becomes a priority. By decomposing manipulation into a route phase and a contact-rich moment, we locate the source of the failure. Along the route, the model cannot prioritize
    
[^290]: 到达还是解决？通过检查点交接归因智能体强化学习的收益

    Reach or Solve? Attributing Agentic RL Gains with Checkpoint Handoffs

    [https://arxiv.org/abs/2609.19636](https://arxiv.org/abs/2609.19636)

    本文提出“检查点交接”评估协议，通过克隆一个检查点到达的状态并移交给另一个检查点而无需重新训练，从而将智能体强化学习的收益分离归因为“到达状态的能力”与“在给定状态下解决问题的能力”两个独立成分。

    

    强化学习如今能够训练在真实环境中执行数十步操作的语言模型智能体。其收益巨大，并被解读为更好的决策能力。处于闭环中的智能体会编写自己的输入。每个观察结果都源于其先前的动作，因此它在回合后期遇到的状态部分是由它自己造成的。于是，SFT检查点和RL检查点即使是在相同的任务上，也是在不同的状态下被评分的。端点成功混合了两种变化：智能体到达了哪里，以及它到达那里之后做了什么。将比较限制在两种策略都能到达的状态上并不能将两者分开。这种限制是基于结果进行的选择，而在我们的数据中，这甚至会翻转效应的符号。我们提出了检查点交接，这是一种评估协议，它克隆某个已发布检查点所到达的状态，并将其移交给另一个检查点，且无需重新训练。通过在SFT和RL之间交叉“到达者”角色和“解决者”角色，可以将端点收益……（摘要在此处被截断）

    arXiv:2609.19636v1 Announce Type: new  Abstract: Reinforcement learning now trains language-model agents that act over dozens of steps in live environments. The gains are large, and they are read as better decision-making. An agent in a closed loop writes its own inputs. Each observation follows from its own earlier actions, so the states it meets late in an episode are partly of its own making. An SFT checkpoint and an RL checkpoint are then scored from different states, even on identical tasks. Endpoint success mixes two changes: where the agent arrives, and what it does once it is there. Restricting the comparison to states both policies reach does not separate them. That restriction selects on an outcome, and in our data it flips the sign of the effect. We introduce checkpoint handoff, an evaluation protocol that clones a state one released checkpoint reached and hands it to another, with no retraining. Crossing a reacher role and a solver role over SFT and RL splits an endpoint ga
    
[^291]: BusMA：一种面向多智能体系统的总线通信基础设施

    BusMA: A Bus Communication Substrate for Multi-Agent Systems

    [https://arxiv.org/abs/2609.15054](https://arxiv.org/abs/2609.15054)

    受计算机总线架构启发，BusMA提出了一种多智能体通信框架，允许任何智能体通过共享总线信道直接与其他智能体通信，突破了传统分层管理者-工作者或路由器消息传递结构对智能体自主性的限制。

    

    多智能体（MA）系统在解决需要规划、工具使用以及多源证据综合的复杂任务方面表现出色。现有系统通常采用分层管理者-工作者（HMW）或基于路由器的消息传递（RMP）结构作为其通信协议。然而，这些设计限制了智能体的自主性：工作者智能体无法直接咨询特定的“同级”智能体，且被错误路由的消息可能会传播错误。受计算机系统中总线架构的启发，我们提出了BusMA，这是一种通信框架，允许任何智能体通过共享信道（即总线）与其他智能体进行通信。它由智能体注册、消息路由和共享内存管理三个组件构成。每个工作者智能体都配备工具，拥有自己的本地内存，可以进行推理、执行行动（工具使用），并通过发布带有特定意图的共享消息进行通信。我们引入了四种意图：讨论、质疑、指导……

    arXiv:2609.15054v1 Announce Type: new  Abstract: Multi-Agent (MA) systems are effective at solving complex tasks that demand planning, tool use, and the synthesis of evidence from multiple sources. Existing systems typically adopt Hierarchical Manager-Worker (HMW) or Router-based Message Passing (RMP) structures as their communication protocol. However, these designs restrict agent autonomy: Worker agents cannot directly consult specific "peers", and misrouted messages can propagate errors. Inspired by bus architectures in computer systems, we propose BusMA, a communication framework that allows any agent to address other agents through a shared channel, i.e., the Bus. It consists of agent registration, message routing, and shared memory management components. Worker agents, each equipped with tools, have their own local memory and can reason, act (tool usage), and communicate by posting shared messages with specific intents. We introduce four intents: discussion, challenge, guidance, 
    
[^292]: LPA-CWM：一种用于反事实世界模型运动推理的学习型物理裁决器

    LPA-CWM: A Learned Physical Adjudicator for Motion Reasoning with Counterfactual World Models

    [https://arxiv.org/abs/2609.14073](https://arxiv.org/abs/2609.14073)

    提出LPA-CWM框架，利用轻量级学习型物理裁决器学习候选响应的可靠性权重，从反事实世界模型中更准确地恢复运动，并引入联合衡量定位、轨迹完整性、可见性和连续性的CMC评估协议。

    

    反事实世界模型（CWM）通过比较事实性预测与干预性预测，从预训练的视频预测器中提取运动信息。然而，在不同目标帧掩码下生成的响应其可靠性各不相同，而均匀聚合方式却对它们赋予相同权重。我们将响应聚合形式化为候选可靠性学习问题，并提出LPA-CWM框架，其中包含一个轻量级的学习型物理裁决器（LPA）。该LPA仅有300万参数，在密集的MOVi-F轨迹上训练，它能在无序候选集上比较视觉上下文和响应结构以预测相对权重，同时CWM预测器和干预生成器保持冻结状态。加权后的响应经过窗口化定位和一次配对重新评估来恢复运动。我们还引入了完整性感知运动对应（CMC），这是一种以真值为锚定的评估协议，可联合衡量定位、轨迹完整性、可见性和连续性……

    arXiv:2609.14073v1 Announce Type: cross  Abstract: Counterfactual world models (CWM) extract motion from pretrained video predictors by comparing factual and intervened predictions. However, responses generated under different target-frame masks vary in reliability, while uniform aggregation weights them equally. We formulate response aggregation as candidate reliability learning and propose LPA-CWM with a lightweight Learned Physical Adjudicator (LPA). Trained on dense MOVi-F trajectories, the 3.0M-parameter LPA compares visual context and response structure across an unordered candidate set to predict relative weights, while the CWM predictor and intervention generator remain frozen. The weighted responses undergo windowed localization and one paired re-evaluation to recover motion. We also introduce Completeness-aware Motion Correspondence (CMC), a ground-truth-anchored evaluation protocol that jointly measures localization, trajectory completeness, visibility, and continuity, count
    
[^293]: 使用GRACE预测碰撞截面：通过早期融合实现几何残差加合物条件化

    Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion

    [https://arxiv.org/abs/2609.12223](https://arxiv.org/abs/2609.12223)

    本文提出GRACE模型，通过早期融合的几何残差加合物条件化方法调整预训练分子几何编码器，将加合物感知的残差学习目标与编码器内的加合物条件化机制相结合，显著提升了对气相分子离子碰撞截面的三维预测精度。

    

    碰撞截面（CCS）源自离子迁移谱质谱技术，是分子注释的常用描述符。对机器学习模型而言，预测CCS具有挑战性，因为它反映了气相分子离子的大小、形状和电离状态。大多数预测器要么忽略显式的三维结构，要么将加合物类型作为后期处理的类别特征，这限制了模型捕获加合物相关几何效应的能力。我们提出了GRACE（通过早期融合实现几何残差加合物条件化），这是一个三维CCS预测器，通过早期融合的几何残差加合物条件化来调整预训练的分子几何编码器。GRACE结合了两个归纳偏置：相对于加合物感知的物理描述符基线的残差学习目标，以及通过可学习的加合物标记和低秩注意力适配器在编码器内部实现的加合物条件化。我们在包含超过9,000个实验分子的精选数据集上对该模型进行了评估。

    arXiv:2609.12223v1 Announce Type: new  Abstract: Collision cross section (CCS), derived from ion mobility mass spectrometry, is a common descriptor for molecular annotation. Prediction is challenging for machine learning models because it reflects the size, shape, and ionization state of a gas-phase molecular ion. Most predictors either ignore explicit 3D structure or treat adduct identity as a late categorical feature, which limits their ability to capture adduct-dependent geometric effects. We present GRACE (Geometric Residual Adduct Conditioning via Early-fusion), a 3D CCS predictor that adapts a pretrained molecular geometry encoder using geometric residual adduct conditioning via early fusion. GRACE combines two inductive biases: a residual objective relative to an adduct-aware physical descriptor baseline and adduct conditioning within the encoder via a learned adduct token and low-rank attention adapters. We evaluate the model on a curated set of over 9,000 experimental molecule
    
[^294]: Suan：纠正大语言模型中的直接偏好安全对齐

    Suan: Rectifying Direct Preference Safety Alignment in Large Language Models

    [https://arxiv.org/abs/2609.08634](https://arxiv.org/abs/2609.08634)

    Suan是一种新颖的偏好优化算法，通过直接在梯度层面构建优化目标（绕过标准变分推导），使大语言模型在实现卓越安全对齐的同时完全保留回应实用性。

    

    将强大的安全防护机制集成到大语言模型（LLM）中，对于提供有用且无害的回应至关重要。尽管专有系统展现出可靠的安全控制，但其底层方法和权衡取舍在很大程度上仍未公开。在开放权重模型中实现相当的安全性仍然是一个持续的挑战，因为经过后训练的模型变体经常出现过拒答和整体质量下降的问题。为了克服这些缺陷，我们提出了Suan，一种新颖的偏好优化算法。与现有方法不同，我们直接在梯度层面构建优化目标，绕过了标准的变分推导。由此，我们获得了更具可解释性和更稳健的训练动态。在多种竞争性基线和基准测试上的广泛评估表明，Suan在实现卓越安全对齐的同时，完全保留了回应的实用性。

    arXiv:2609.08634v1 Announce Type: cross  Abstract: Integrating robust safety guardrails into Large Language Models (LLMs) is essential for delivering helpful yet harmless responses. While proprietary systems exhibit reliable safety controls, their underlying methodologies and trade-offs remain largely undisclosed. Achieving comparable security in open-weight models remains a persistent challenge, as post-trained variants frequently suffer from over-refusal and degraded general quality. To overcome these drawbacks, we introduce Suan, a novel preference optimization algorithm. Unlike existing methods, we formulate the optimization objective directly at the gradient level, bypassing the standard variational derivation. As a result, we obtain more interpretable and robust training dynamics. Extensive evaluations across a diverse suite of competitive baselines and benchmarks demonstrate that Suan achieves superior safety alignment while fully preserving response utility.
    
[^295]: 基于SleepFM-2从两百万小时睡眠数据中学习可迁移的人体生理特征

    Learning transferable human physiology from two million hours of sleep with SleepFM-2

    [https://arxiv.org/abs/2609.06849](https://arxiv.org/abs/2609.06849)

    SleepFM-2是一个基于来自26个队列超过两百万小时多模态生理数据的睡眠基础模型，显著提升了疾病预测、睡眠分期和事件检测能力，并能从多导睡眠图表征中预测电子健康记录中的多种疾病表型，还可迁移至可穿戴设备。

    

    睡眠通过捕捉大脑、心脏、肌肉和呼吸系统的协调活动，为健康提供了一个每夜的观察窗口。我们推出了SleepFM-2，一个睡眠基础模型，该模型基于来自26个队列的282,511条多导睡眠图（PSG）记录进行开发和评估，其中235,865条用于预训练。这些数据涵盖了超过两百万小时的多模态生理数据。与SleepFM相比，SleepFM-2改进了疾病预测和睡眠分期，支持觉醒、肢体运动和呼吸事件的检测，并可迁移至可穿戴设备传感数据和主观睡眠表型。将其PSG表征与年龄、性别和BMI相结合的模型，在两个保留队列中（包括一个预训练期间未见过的医疗系统）对215个后续记录的电子健康记录（EHR）表型达到了预先设定的区分度和显著性标准。对于155个表型，PSG表征在人口统计学信息之外提供了可重复的额外信息。

    arXiv:2609.06849v1 Announce Type: new  Abstract: Sleep provides a nightly window into health by capturing coordinated activity across the brain, heart, muscles and respiratory system. We introduce SleepFM-2, a sleep foundation model developed and evaluated on 282,511 polysomnography recordings from 26 cohorts, including 235,865 used for pretraining. These data span more than two million hours of multimodal physiology. Compared with SleepFM, SleepFM-2 improves disease prediction and sleep scoring, supports arousal, limb movement and respiratory event detection, and transfers to wearable sensing and subjective sleep phenotypes. A model combining its PSG representation with age, sex and BMI met a prespecified discrimination and significance criterion for 215 subsequently recorded EHR phenotypes in two held-out cohorts, including one health system unseen during pretraining. For 155 phenotypes, the PSG representation added reproducible information beyond demographics. SleepFM-2 also outperf
    
[^296]: LayerRoute：面向视觉-语言-动作策略的动作条件混合层路由

    LayerRoute: Action-Conditioned Mixture-of-Layers Routing for Vision-Language-Action Policies

    [https://arxiv.org/abs/2609.06079](https://arxiv.org/abs/2609.06079)

    提出LayerRoute，一个动作条件的混合层路由接口，使VLA策略能够根据具体动作需求自适应地访问和混合VLM不同层次的表示，突破了传统固定层分配的限制。

    

    arXiv:2609.06079v1 公告类型：新论文 摘要：视觉-语言-动作（VLA）策略利用预训练的视觉-语言模型（VLM）来指导机器人控制的动作生成。VLM提供跨层演化的分层视觉-语义表示，从局部视觉几何到抽象的、与语言对齐的语义；因此，不同的操作任务可能需要不同的层表示混合。同时，动作模块在动作计算过程中维护不断演化的中间表示，这些表示可能为后续决策提供有用信息。然而，现有的VLA接口在表示访问方面的灵活性有限：VLM信息通过为每个动作层固定的层分配来暴露，而中间动作状态仅通过残差流隐式传播，缺乏显式复用。我们提出了LayerRoute，一个动作条件的表示路由接口，使VLA策略能够自适应地访问VLM的各层表示。

    arXiv:2609.06079v1 Announce Type: new  Abstract: Vision-Language-Action (VLA) policies leverage pretrained vision-language models (VLMs) to guide action generation for robot control. VLMs provide hierarchical visual-semantic representations that evolve across layers, from local visual geometry to abstract, language-aligned semantics; different manipulation tasks may therefore require different mixtures of layer representations. Meanwhile, the action module maintains intermediate representations that evolve throughout action computation and may provide useful information for subsequent decisions. However, existing VLA interfaces offer limited flexibility in representation access: VLM information is exposed through fixed layer assignments for each action layer, while intermediate action states are only propagated implicitly through residual streams without explicit reuse. We introduce LayerRoute, an action-conditioned representation routing interface that enables adaptive access to VLM l
    
[^297]: trajectory-judge：仅基于结果的LLM评判器在智能体轨迹上遗漏了什么

    trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories

    [https://arxiv.org/abs/2609.00038](https://arxiv.org/abs/2609.00038)

    仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。

    

    仅基于结果的评估是LLM智能体在生产环境中的默认做法：向评判器展示用户请求和最终回复，询问其处理是否得当。这一指标在结构上无法察觉那些“以错误方式得到正确答案”的智能体。我们在真值可以通过构造获知的场景下测量这一盲区：一个确定性的使用工具的客服支持台环境、一个总能解决问题的脚本化oracle策略，以及一个在已知步骤恰好破坏一个环节的故障注入器，并根据用户可见结果是否仍然保持（静默型故障）与否（显性型故障）对故障进行分层。五种评判器（程序化规则、仅结果型、两种模型规模的逐步评分标准型、以及自一致性集成）在400条轨迹上按照检测能力、步骤定位、故障类型判定、校准度和成本进行评分。结果显示：仅结果型评判器能捕获84%的显性故障，但只能捕获45%的静默故障，同时还会误报33%的正确轨迹；而逐步评分标准型评判器对静默故障的召回率达到77%。

    arXiv:2609.00038v1 Announce Type: cross  Abstract: Outcome-only evaluation is the production default for LLM agents: show a judge the request and the final reply and ask whether it was handled well. The metric is structurally blind to an agent that reaches the right answer the wrong way. We measure that blind spot where ground truth is known by construction: a deterministic tool-using support-desk environment, a scripted oracle policy that always solves it, and a fault injector that breaks exactly one thing at a known step, stratifying faults by whether the customer-visible outcome survived (silent) or not (loud). Five judges (programmatic rules, outcome-only, step-rubric at two model sizes, and a self-consistency ensemble) are scored on detection, step localisation, fault typing, calibration, and cost over 400 trajectories. The outcome-only judge catches 84% of loud faults but 45% of silent ones while flagging 33% of correct trajectories; a step-rubric judge reaches 77% silent recall 
    
[^298]: 用户会知道吗？针对使用工具的LLM智能体的隐蔽间接提示注入

    Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents

    [https://arxiv.org/abs/2608.30362](https://arxiv.org/abs/2608.30362)

    该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。

    

    随着LLM智能体通过工具执行真实世界的操作，间接提示注入（IPI）已成为一种严重的威胁。标准的评估指标——攻击成功率（ASR）——只统计注入是否成功，却忽略了用户在智能体最终响应中能够注意到什么。通过观察成功的注入轨迹，我们发现两种截然不同的结果：智能体在执行注入的同时返回看似正常的响应，或者在最终响应中报告被注入的操作，从而给用户留下察觉的机会。我们将这两类成功分别称为隐蔽成功和公开成功。从用户视角出发，我们将ASR分解为隐蔽成功率（CSR）——统计在最终响应中不留任何痕迹的成功注入——以及公开成功率（OSR）——统计用户能够察觉的成功注入。为了理解造成这一差距的原因，我们分析了成功的注入轨迹，发现注入后智能体的行为是区分隐蔽与公开的关键：隐蔽的轨迹会将控制权交回……

    arXiv:2608.30362v1 Announce Type: new  Abstract: As LLM agents take real-world actions through tools, indirect prompt injection (IPI) has emerged as a serious threat. The standard metric, Attack Success Rate (ASR), counts whether an injection succeeds but ignores what the user notices in the agent's final response. Looking at successful injection traces, we find two distinct outcomes: the agent executes the injection while returning an otherwise normal response, or reports the injected action in its final response, giving the user a chance to notice. We call these covert and overt successes. From the user's perspective, we decompose ASR into the Covert Success Rate (CSR), counting successes leaving no trace in the final response, and the Overt Success Rate (OSR), counting successes the user can detect. To understand what drives the gap, we analyze successful trajectories and find that the agent's behavior after the injection separates covert from overt: covert traces hand control back 
    
[^299]: 面向自然语言形式化的分层一致性蒸馏

    Stratified Consistency Distillation for Natural Language Formalization

    [https://arxiv.org/abs/2608.30258](https://arxiv.org/abs/2608.30258)

    提出分层一致性蒸馏方法，通过对前沿大模型生成的多个逻辑翻译按语义等价性聚类，并依据熵水平采用不同策略筛选伪标签来微调小模型，从而提升自然语言到逻辑公式翻译的准确性。

    

    神经符号推理通过结合大型语言模型（LLM）和符号求解器，在解决复杂推理任务方面展现出令人鼓舞的成功。尽管这种方法前景可期，但一个根本性的挑战仍然存在：如何提高从自然语言到逻辑公式翻译的准确性。当前的方法主要依赖于提示工程，这难以在不同领域和输入格式之间进行扩展。借鉴微调在其他模型适配与对齐应用中的成功经验，我们提出了一种基于微调的分层一致性蒸馏方法：(1) 我们使用前沿大语言模型为每个输入生成K个逻辑翻译，并按语义等价性进行聚类；(2) 根据熵水平，我们分别应用多数投票（低熵）、LLM作为评判者（中熵）或统一化/弃权（高熵）策略；(3) 使用筛选出的伪标签对较小的模型进行微调。我们的实验...

    arXiv:2608.30258v1 Announce Type: cross  Abstract: Neurosymbolic reasoning has shown promising success in addressing complex reasoning tasks by combining large language models (LLMs) and symbolic solvers. While this approach shows promise, a fundamental challenge remains: improving the accuracy of translations from natural language to logical formulas. Current methods predominantly rely on prompt engineering, which is difficult to scale across different domains and input formats. Drawing inspiration from the success of fine-tuning in other model adaptation and alignment applications, we propose a fine-tuning-based Stratified Consistency Distillation approach: (1) We generate K logical translations per input using a frontier LLM and cluster them by semantic equivalence (2) Based on the entropy level, we apply majority voting (low entropy), LLM-as-a-Judge (medium entropy), or unification/abstention (high entropy), and (3) fine-tune a smaller model using the selected pseudo-labels. Our ex
    
[^300]: 重新审视：测量压力下多模态模型推理链中的谄媚行为

    Looking Again: Measuring Sycophancy in the Reasoning Chains of Multimodal Models Under Pressure

    [https://arxiv.org/abs/2608.28623](https://arxiv.org/abs/2608.28623)

    该论文提出了首个用于测量大型多模态推理模型谄媚行为的基准和数据集，通过四种视觉推理任务与五种压力条件评估模型在用户给出错误答案时的表现，发现谄媚行为在压力下普遍存在，不仅体现在最终答案中，也出现在推理链中。

    

    大型多模态推理模型（LMRMs）的能力日益增强，这主要归功于在回答之前生成显式的思维链推理。在语言模型中已经观察到，这种性能往往伴随着谄媚行为（sycophancy），即模型在证据面前倾向于迎合用户。然而，对于大型多模态推理模型，目前尚不存在可靠的谄媚行为测量方法。我们通过引入一个基准和数据集来填补这一空白，用于评估大型多模态推理模型在面对用户给出的错误答案时的谄媚行为。我们的基准将四个基于视觉的数据集（涵盖数学、临床、时间和人口统计推理）与五种压力条件相配对，并在单轮和多轮设置中进行测试。我们评估了最终答案中的谄媚行为以及其在推理链中的出现情况。我们发现谄媚行为在压力下普遍存在，其中“陈述”压力引发的谄媚率最高，而“信念”压力最低。

    arXiv:2608.28623v1 Announce Type: cross  Abstract: Large multimodal reasoning models (LMRMs) are getting increasingly capable, primarily through generating explicit chain-of-thought reasoning before answering. In language models it has been observed that this performance often comes with sycophancy, the tendency of a model to agree with the user over the evidence. However, for LMRMs no reliable method to measure sycophancy yet exists. We bridge this gap by introducing a benchmark and dataset for evaluating LMRM sycophancy when confronted with a wrong answer from a user. Our benchmark pairs four visually grounded datasets spanning mathematical, clinical, temporal, and demographic reasoning with five pressure conditions in single-turn and multi-turn settings. We evaluate sycophancy in the final answer as well as its emergence within the reasoning chain. We find that sycophancy is prevalent under pressure, with Statement pressure eliciting the highest rates and Conviction the lowest for a
    
[^301]: ConfAL-WM：面向动作条件世界模型的置信度引导主动学习

    ConfAL-WM: Confidence-Guided Active Learning for Action-Conditioned World Models

    [https://arxiv.org/abs/2608.25572](https://arxiv.org/abs/2608.25572)

    提出ConfAL-WM框架，通过在UNet解码器特征上附加轻量级置信度探针生成潜空间密集置信度图，并聚合为任务、帧、补丁三个层级的分数，实现具身世界模型后训练中的数据预算分配与局部化训练增强。

    

    动作条件世界模型已成为具身预测、规划和合成数据生成的重要基础，但其在新的任务与场景分布下的误差往往集中在局部时空区域，例如机械臂、被操作物体、接触区域以及被遮挡的物体。本文提出了ConfAL-WM，一个用于具身世界模型后训练的置信度引导主动学习框架。在EnerVerse-AC（EVAC）的基础上，我们在UNet解码器特征上附加了一个轻量级置信度探针，并在潜空间中预测密集的置信度图。这些置信度图被聚合为任务级、帧级和补丁级的分数，从而实现数据预算的分配与局部化的训练增强。我们的流程在一个小的目标域子集上训练探针并对EVAC进行预热。随后EVAC-v1提供任务级的数据采集信号以及可选的帧/补丁加权信号；所有基于选定数据训练的模型都进行初始化……（原文摘要在此处截断）

    arXiv:2608.25572v2 Announce Type: replace-cross  Abstract: Action-conditioned world models have become an important foundation for embodied prediction, planning, and synthetic data generation, but their errors under new task and scene distributions are often concentrated in localized spatiotemporal regions such as robot arms, manipulated objects, contact areas, and occluded objects. This paper presents ConfAL-WM, a confidence-guided active learning framework for post-training embodied world models. Building upon EnerVerse-AC (EVAC), we attach a lightweight confidence probe to UNet decoder features and predict dense confidence maps in the latent space. These maps are aggregated into task-, frame-, and patch-level scores, enabling data-budget allocation and localized training enhancement. Our pipeline trains the probe and warms up EVAC on a small target-domain subset. EVAC-v1 then supplies task-level acquisition and optional frame/patch weighting signals; all selected-data models are ini
    
[^302]: GlanceWAM：面向世界-动作模型的稀疏测试时想象

    GlanceWAM: Sparse Test-Time Imagination for World-Action Models

    [https://arxiv.org/abs/2608.23927](https://arxiv.org/abs/2608.23927)

    GlanceWAM通过在单一视频DiT骨干上将视觉想象与控制解耦——以异步方式在后台生成前瞻帧并直接在潜空间中消费——实现了机器人实时控制（48毫秒）与更优任务成功率的兼得。

    

    视频生成模型为机器人学习提供了丰富的物理先验，然而现有的世界-动作模型面临一个根本性的权衡：以控制频率同步生成视频的延迟过高而不可行，而放弃测试时的视觉想象则会牺牲任务成功率。我们证明，当视觉想象以异步方式在关键路径之外生成、并直接在潜空间中被消费时，既能实现实时推理，又能获得更优的任务成功率。我们提出GlanceWAM，它在单一共享的视频DiT骨干上将想象与控制解耦：一个异步提议器以较慢的时钟“瞥视”前方，在后台想象未来数秒的单个前瞻帧，同时动作头以控制频率（48毫秒）纯粹在潜空间中解码动作块，不产生任何阻塞。该方法得益于一个隔离视频表示的非干扰注意力掩码，以及一种能够适应过时预测的抗陈旧视界训练机制。

    arXiv:2608.23927v2 Announce Type: cross  Abstract: Video generative models provide rich physical priors for robot learning, yet existing world-action models (WAMs) face a fundamental trade-off: synchronous video generation at control rate is latency-prohibitive, while abandoning test-time visual imagination sacrifices task success. We show that visual imagination achieves both real-time inference and superior success rates when generated asynchronously off the critical path and consumed directly in latent space. We introduce GlanceWAM, which decouples imagination from control on a single shared video DiT backbone: an asynchronous proposer glances ahead on a slow clock to imagine a single lookahead frame seconds into the future in the background, while an action head decodes action chunks at control rate (48 ms) purely in latent space without blocking. Enabled by a non-interfering attention mask that isolates video representations and staleness-robust horizon training that accommodates 
    
[^303]: EXAM²：扩展多语言与多模态分析中的音频理解

    EXAM$^2$: $\underline{Ex}tending$ $\underline{A}udio$ $Understanding$ $in$ $\underline{M}ultilingual$ $and$ $\underline{M}ultimodal$ $Analysis$

    [https://arxiv.org/abs/2608.23758](https://arxiv.org/abs/2608.23758)

    本文提出了EXAM²，一个覆盖六种语言和多种音频模态（含视觉图像）的多语言多模态音频理解基准，旨在更真实地评估场景感知音频推理和跨模态理解能力。

    

    最近的大型音频语言模型（LALMs）在音频理解方面取得了显著进展。然而，现有评估在很大程度上仍局限于英语和狭窄的音频领域。先前的基准测试通常专注于单一音频模态，即语音、声音或音乐，限制了对这些模型如何在多样视觉场景中泛化的系统研究。在本文中，我们介绍了EXAM²，一个涵盖六种语言和多种模态（包括语音、声音、音乐、混合音频设置和视觉图像）的多语言和多模态音频理解基准。通过将视觉信息与异构音频输入相结合，EXAM²能够对场景感知音频推理和跨模态理解进行更现实的评估。EXAM²包含5,667个多项选择题、22,614个图像实例和135,684个多语言翻译。我们评估了最先进的开源模型。

    arXiv:2608.23758v1 Announce Type: cross  Abstract: Recent large audio language models (LALMs) have achieved impressive progress in audio understanding. However, existing evaluations remain largely constrained to English and narrow audio domains. Prior benchmarks typically focus on a single audio modality, i.e., speech, sound, or music, limiting the systematic investigation into how these models generalize across diverse visual scenarios. In this paper, we introduce EXAM$^2$, a benchmark for multilingual and multimodal audio understanding spanning six languages and multiple modalities, including speech, sound, music, mixed-audio settings, and visual images. By incorporating visual information alongside heterogeneous audio inputs, EXAM$^2$ enables more realistic evaluation of scene-aware audio reasoning and cross-modal comprehension. EXAM$^2$ comprises $5,667$ multiple-choice questions, $22,614$ image instances, and $135,684$ multilingual translations. We evaluate state-of-the-art open-s
    
[^304]: WAM-OPD：面向世界行动模型的同策略蒸馏

    WAM-OPD: On-Policy Distillation for World Action Models

    [https://arxiv.org/abs/2608.22364](https://arxiv.org/abs/2608.22364)

    提出WAM-OPD，一种部署一致的同策略蒸馏方法，通过冻结教师模型标注学生行动历史并联合优化视频与动作损失，在无需稀疏奖励强化学习的情况下修复加速学生模型的任务能力。

    

    世界行动模型（WAM）将视觉未来预测与机器人动作生成相结合，但加速的学生模型在蒸馏过程中可能丧失任务能力，并随后遇到离线数据表示不足的状态。我们研究同策略蒸馏（OPD）是否能在无需稀疏奖励强化学习的情况下修复此类学生模型。我们引入了WAM-OPD，一种面向视频优先WAM的部署一致性后训练方案。学生模型在环境中行动，因此决定历史分布。一个冻结的教师模型用连贯的视频和动作目标标记这些学生历史，而学生动作分支则在其自身生成的视频计划下进行训练，这与部署时一致。联合视频和动作损失更新共享骨干中的轻量适配器，并配合动作流匹配正则化器。在RoboTwin 2.0的两项任务初步研究中，发布了单视频/单动作步骤的版本。

    arXiv:2608.22364v1 Announce Type: new  Abstract: World action models (WAMs) couple visual future prediction with robot action generation, but accelerated students can lose task capabilities during distillation and later encounter states that are poorly represented by offline data. We study whether on-policy distillation (OPD) can repair such a student without requiring sparse-reward reinforcement learning. We introduce WAM-OPD, a deployment-consistent post-training recipe for a video-first WAM. The student acts in the environment and therefore determines the history distribution. A frozen teacher labels those student histories with coherent video and action targets, while the student action branch is trained under its own generated video plan, as it is at deployment. Joint video and action losses update lightweight adapters in the shared backbone, together with an action flow-matching regularizer. In preliminary RoboTwin 2.0 studies on two tasks, the released one-video/one-action-step 
    
[^305]: 打断链条：通过杀伤链视角审视人类对AI生成虚假信息的感知

    Interrupting the Chain: Human Perception of AI-Generated Disinformation Through a Kill Chain Lens

    [https://arxiv.org/abs/2608.21389](https://arxiv.org/abs/2608.21389)

    通过杀伤链框架的实证研究揭示，人类对AI生成虚假信息的检测存在感知-准确性差距、LLM文本难以区分以及认知疲劳导致虚假新闻检测下降10.2%等关键弱点，为主动防御提供了干预点。

    

    生成式AI能够大规模定制虚假信息，但防御措施仍主要是反应性的。我们报告了一项人类受试者研究（n=504名参与者，n=2,438项判断）的实证结果，在该研究中，用户根据来源（人类与机器）和真实性（真实与虚假）对新闻片段进行分类。我们采用改编的网络安全杀伤链作为干预分类法来组织结果，将感知数据映射到认知攻击生命周期的各个阶段。三个关键发现浮现：（1）感知-准确性差距，即高度怀疑并未提高检测能力；（2）现代LLM经常生成与人类难以区分的文本；（3）不对称的认知疲劳效应，在持续暴露下虚假新闻检测性能下降10.2个百分点，而AI来源检测保持稳定。这些发现确定了针对AI驱动虚假信息进行主动防御的候选干预点。

    arXiv:2608.21389v1 Announce Type: cross  Abstract: Generative AI enables customized misinformation at scale, yet defenses remain largely reactive. We present empirical findings from a human-subject study (n=504 participants, n=2,438 judgments) in which users classified news fragments by origin (human vs. machine) and veracity (real vs. fake). We organize results using an adapted cybersecurity kill chain as a taxonomy for intervention, mapping perception data onto stages of a cognitive attack lifecycle. Three key findings emerge: (1) a perception-accuracy gap where heightened suspicion does not improve detection; (2) modern LLMs frequently produce human-indistinguishable text; and (3) an asymmetric cognitive fatigue effect where fake-news detection degrades by 10.2 percentage points under sustained exposure while AI-origin detection remains stable. These findings identify candidate intervention points for proactive defense against AI-driven disinformation.
    
[^306]: 一种在将世界模型与现实对齐中不可约的量子优势

    An Irreducible Quantum Advantage in Aligning World Models with Reality

    [https://arxiv.org/abs/2608.19779](https://arxiv.org/abs/2608.19779)

    本文证明即使真实世界是经典的，经典世界模型也无法完美对齐代理策略，而量子模型可能提供不可约的优势。

    

    世界模型提供了真实世界的数字模拟，使代理能够在昂贵的现实部署之前进行训练和测试。在每个时间步，它们接收一个动作并生成与真实世界统计匹配的观测和奖励。在复杂环境中，当前结果取决于遥远的过去事件，这需要记忆。人们可能期望，通过增加记忆，我们总能构建一个足够准确的模型，以使真实和虚拟世界的最优代理策略对齐。我们表明，对于经典世界模型，即使真实世界本身是经典的，这也是错误的。我们构造了真实世界，其中每个有限经典模型沿着相同的可能轨迹失败：它要么在真实世界明显偏好某个动作时失去区分动作的能力，要么反复将最高期望奖励分配给次优动作。其期望奖励估计也保留了一个不可消失的...

    arXiv:2608.19779v1 Announce Type: cross  Abstract: World models provide digital simulacra of the true world, allowing agents to be trained and tested before costly real-world deployment. At each time step, they receive an action and generate an observation and reward matching the statistics of the true world. In complex environments where present outcomes depend on events far in the past, this requires memory. One might expect that, by increasing memory, we can always build a model accurately enough to align the optimal agent policies of the real and virtual worlds. We show that this is false for classical world models, even when the true world itself is classical. We construct true worlds for which every finite classical model fails along the same possible trajectory: it either loses the ability to distinguish actions when the true world clearly prefers one, or repeatedly assigns the highest expected reward to suboptimal actions. Its expected-reward estimates also retain a nonvanishin
    
[^307]: 面向机器人仿真到现实部署的、基于统计验证的安全鲁棒神经策略学习

    Safe and Robust Neural Policy Learning with Statistical Verification for Sim-to-Real Deployment in Robotics

    [https://arxiv.org/abs/2608.06481](https://arxiv.org/abs/2608.06481)

    本文提出一种课程驱动的闭环框架，将基于场景的进化策略与统计模型检测验证相结合，在协同优化策略性能的同时逐步扩大安全操作边界，最终生成带有统计验证安全与性能保证的神经控制器，助力可靠的仿真到现实部署。

    

    在仿真中合成安全且鲁棒的神经控制器，以实现可靠的仿真到现实部署，仍然是机器人领域的一项关键挑战。现有的基于学习的方法通常缺乏在明确定义的操作区域内的安全性和性能保证，而训练后验证技术在检测到安全违规时也没有提供改进控制器的机制。为了弥合这一差距，我们提出了一个课程驱动的框架，在闭环流程中将基于场景的进化策略与基于统计模型检测的验证紧密集成。从候选区域出发，该方法在协同优化策略性能的同时，逐步扩大其安全操作边界。在终止时，它会输出一个神经控制器以及一个经过统计验证的安全性和性能保证的区域。在Cartpole和3D四旋翼基准上的广泛评估显示6.14倍和……（原文摘要在此处截断）

    arXiv:2608.06481v2 Announce Type: replace-cross  Abstract: Synthesizing safe and robust neural controllers in simulation for reliable sim-to-real deployment remains a critical challenge in robotics. Existing learning-based methods typically lack safety and performance guarantees over an explicitly defined operating region, while post-training verification techniques provide no mechanism to refine controllers when safety violations are detected. To bridge this gap, we propose a curriculum-driven framework that tightly integrates scenario-based Evolution Strategy with Statistical Model Checking-based verification in a closed-loop procedure. Starting from a candidate region, our approach co-optimizes policy performance while progressively enlarging its safe operating boundaries. Upon termination, it yields a neural controller together with a region over which safety and performance are statistically verified. Extensive evaluations on Cartpole and 3D Quadrotor benchmarks, showing 6.14x and
    
[^308]: 逃离过度挤压：面向消息传递网络的可寻址且支持感知的全局记忆

    Escaping Oversquashing: Addressable and Support-Aware Global Memory for Message Passing Networks

    [https://arxiv.org/abs/2608.02709](https://arxiv.org/abs/2608.02709)

    该论文提出一种兼具可寻址性与支持感知的全局记忆机制，通过乘性读写映射仅用对数级地址编码维度即可选择 M 个记忆行，并借助学习到的私有锚点保持读取有界，从而解决消息传递网络中多节点共享虚拟节点全局状态导致的瓶颈问题。

    

    虚拟节点是对抗过度挤压（oversquashing）的一种自然工具：它们用两跳的全局路径取代了长距离的消息传递路径。但当许多节点共享同一个全局状态时，这条捷径本身可能成为瓶颈。我们研究了这种全局记忆的两个性质。第一，可寻址性：在恒定边距的地址编码以及放大该边距的非线性条件下，乘性写入/读取映射仅需 O(log M) 维的地址编码即可提供 M 个可选择的记忆行。交叉注意力槽和受约束的 ELU+1 双线性记忆均满足这些条件。第二，支持感知：归一化交叉注意力缺乏一个可供潜在查询用作参照的自键。一个学习得到的私有锚点提供了这一参照，保持读取的有界性，并揭示匹配源质量的强度。我们在 Two-Radius 和 Tree-NeighborsMatch 的多个实例上展示了这些性质的优点，可寻址的真实……

    arXiv:2608.02709v2 Announce Type: replace-cross  Abstract: Virtual nodes are a natural tool against oversquashing: they replace long message-passing paths by a two-hop global route. But when many nodes share one global state, that shortcut can become a bottleneck itself. We study two properties of this global memory. First, addressability: under constant-margin address codes and a nonlinearity that amplifies this margin, multiplicative write/read maps provide $M$ selectable memory rows with only $O(\log M)$ address-code dimensions. Cross-attention slots and a constrained $ELU+1$ bilinear memory both satisfy these conditions. Second, support awareness: normalized cross-attention has no self-key for a latent query to use as a reference. A learned private anchor supplies this reference, keeps the read bounded, and exposes the strength of the matching source mass. We demonstrate the merits of such properties on several instances of Two-Radius and Tree-NeighborsMatch: both addressable reali
    
[^309]: 世界动作规划器：基于动作条件世界模型的泛化机器人决策

    World Action Planner: Generalizable Robot Decision-Making with Action-Conditioned World Models

    [https://arxiv.org/abs/2607.27599](https://arxiv.org/abs/2607.27599)

    提出了World Action Planner——一种利用动作条件世界模型进行“想象”并采用由粗到细的搜索策略来优化动作计划的机器人规划系统，显著提升了机器人对新场景、新布局和新任务的泛化决策能力。

    

    构建能够应对多样化应用的通用机器人智能体仍然是一项根本性挑战。虽然基于模仿学习的策略可以在熟悉的训练环境中表现良好，但它们往往难以泛化到新场景、新布局和新的任务组合中。为此，我们提出了World Action Planner（世界动作规划器），这是一个具身智能机器人规划系统，其中智能体通过与动作条件世界模型进行“想象”，搜索并组合可执行的动作计划。该搜索以由粗到细的方式进行。首先，智能体通过对想象的世界模型推演结果进行推理，执行全局动作优化，以识别潜在失败并改进所提出的动作计划。随后，它执行局部动作搜索，比较相邻候选动作的想象未来结果，从中选出最佳动作予以执行。在组合式长时程任务、新物体布局以及真实机器人在未见任务上的规划等场景中……（原文摘要在此处截断）

    arXiv:2607.27599v2 Announce Type: replace  Abstract: Building generalizable robot agents for diverse applications remains a fundamental challenge. While imitation learning-based policies can perform well in familiar training environments, they often struggle to generalize to novel scenes, layouts, and task compositions. To this end, we present World Action Planner, an agentic robot planning system in which the agent searches for and composes executable action plans through imagination with an action-conditioned world model. The search proceeds in a coarse-to-fine manner. First, the agent performs global action optimization by reasoning over imagined world-model rollouts to identify potential failures and refine the proposed action plan. It then performs local action search, comparing the imagined future outcomes of neighboring candidates to select the best action for execution. Across compositional long-horizon tasks, novel object layouts, and real-robot planning on novel tasks without
    
[^310]: 多智能体大语言模型系统中分布式后门的早期检测：一项特征化研究

    Early Detection of Distributed Backdoors in Multi-Agent LLM Systems: A Characterization Study

    [https://arxiv.org/abs/2607.24893](https://arxiv.org/abs/2607.24893)

    多智能体大语言模型系统中的分布式后门攻击将加密载荷片段分散到多个被投毒的工具中，在第一个片段注入前几乎无法被检测，而一旦注入开始，前缀检测器便可标记99.5%的成功攻击。

    

    多智能体大语言模型系统可能遭受一种没有任何单一智能体完整持有载荷的攻击：多个被投毒的工具各自隐藏一个加密片段，将其分散在多个智能体之中，随后由一个外部步骤在运行结束后重新组装并执行这些片段。孤立地评估每个动作的逐步安全检查可能无法识别完整的分布式载荷。我们研究了在运行仍在进行时这种攻击最早能在何时被检测到，以及在其最明显的线索被剥离后检测的鲁棒性如何。我们在一个分层多智能体系统上构建了一个可运行的攻击实例，在良性条件和受攻击条件下，跨五个语言模型和两个工具环境运行，并记录每个片段被注入的时间以及载荷被组装和执行的时间。在第一个片段被注入之前，几乎没有运行被标记；一旦注入开始，前缀检测器能够以99.5%的比率标记成功的攻击。

    arXiv:2607.24893v2 Announce Type: replace-cross  Abstract: Multi-agent LLM systems can be attacked by a payload that no single agent ever holds in full: several poisoned tools each hide one encrypted fragment, spreading them across several agents, and an external step reassembles and executes them after the run. Per-step safety checks that judge each action in isolation may fail to recognize the complete distributed payload. We investigate how early such an attack can be detected while the run is still unfolding, and how robustly it can be caught once its most obvious cues are stripped away. We build a working instance on a hierarchical multi-agent system, run it under benign and attacked conditions across five language models and two tool environments, and record when each fragment is injected and when the payload is assembled and executed. Almost no run is flagged before its first fragment is injected; once injection begins, a prefix detector flags $99.5\%$ of successful attacks with
    
[^311]: ORACLE：通过自适应验证器校准反馈实现智能体AI编排器路由

    ORACLE: Agentic AI Orchestrator Routing Via Adaptive Verifier Calibration Feedback

    [https://arxiv.org/abs/2607.22465](https://arxiv.org/abs/2607.22465)

    ORACLE提出了一种并发感知的在线智能体路由机制，通过将自适应路由与自适应验证器校准反馈相结合，解决了固定验证器难以泛化到异构智能体任务、以及验证器位于关键路径导致并发请求服务质量下降的问题，且无需训练即可即插即用。

    

    现代企业智能体部署由具有不同能力和成本的异构大语言模型（LLM）池组成。现有的模型路由策略在优化质量-成本权衡的同时，仅提供请求级别的静态决策。较新的解决方案将智能体路由视为任务级选择问题，采用基于串行验证器的路由器反馈循环。然而，这类方案中适用于同质工作负载的固定验证器，可能无法泛化到异构的智能体任务批次（例如：编程任务、通用对话等）。此外，由于验证器被置于反馈循环的关键路径上，在处理多个并发请求的路由时，服务质量可能会受到影响。为缓解这些问题，我们提出了ORACLE。它是一种并发感知的在线路由机制，将自适应路由与自适应验证相校准以生成反馈。ORACLE作为一个无需训练的即插即用“反馈循环”，可部署于任何模型选择器之上（原文摘要在此处截断）。

    arXiv:2607.22465v3 Announce Type: replace  Abstract: Modern enterprise agent deployments consist of a heterogeneous pool of large language models (LLMs) having diverse capabilities and cost. Existing model routing strategies optimize the quality-cost trade-off, while providing request-level static decisions. More recent solutions address agentic routing as a task-level selection with a serial verifier based router feedback loop. However, their fixed verifier suitable for homogeneous workloads may not generalize to heterogeneous batches of agentic tasks (example: coding, general conversational). Additionally, due to the verifier placement in the critical path of the loop, serving quality may be affected during multiple concurrent requests routing. To mitigate these issues, we present ORACLE. It is a concurrency-aware online routing mechanism that aligns adaptive routing with adaptive verification for feedback. ORACLE acts as a training-free drop-in 'feedback loop' on top of any model-se
    
[^312]: 基于深度学习的粘弹性赫兹接触中时间分辨粘附力预测

    Deep learning-based prediction of time-resolved adhesive forces in viscoelastic Hertzian contacts

    [https://arxiv.org/abs/2607.19060](https://arxiv.org/abs/2607.19060)

    本文提出一种标量条件化的有状态序列到序列深度学习模型，结合固定测量步长（FMS）表示方法，能够从位移历史快速预测粘弹性赫兹接触中的完整时间分辨粘附力演化，克服了传统数值模拟计算成本高、无法用于实时应用和设计优化的局限。

    

    快速预测粘附性软质粘弹性接触的响应是当前软体机器人技术以及抓取和操控任务中的一项挑战。确定完整的时间分辨力轨迹需要完整的数值模拟，其计算成本强烈依赖于参数，使其在实时应用或设计优化循环中并不实用。在这项工作中，我们通过训练一个标量条件化的、有状态的序列到序列深度学习模型来克服这一限制，该模型能够根据规定的位移历史预测完整的力演化，适用于短程和长程粘附两种情形。数据集涵盖四个数量级的加载和卸载速率，并包含不同的停留时间，Tabor参数范围为0.2至3.2。为了实现跨这些异构时间尺度的学习，我们引入了一种固定测量步长（FMS）表示方法，将可变的……

    arXiv:2607.19060v2 Announce Type: replace-cross  Abstract: Fast prediction of the response of adhesive soft viscoelastic contacts represents a current challenge in soft robotics and for gripping and manipulation tasks. Determining the complete time-resolved force trajectory requires full numerical simulations, whose computational cost is strongly parameter-dependent, making them impractical for real-time application or design-optimization loops. In this work, we overcome this limitation by training a scalar-conditioned, stateful, sequence-to-sequence deep learning model to predict the full force evolution from a prescribed displacement history for both short- and long-range adhesion regimes. The data set spans four orders of magnitude in loading and unloading rates and includes varied dwell times, with the Tabor parameter ranging from $0.2$ to $3.2$. To enable learning across these heterogeneous time scales, we introduce a fixed-measurement-step (FMS) representation that converts varia
    
[^313]: 助手还是行动者？学生使用通用AI代理时的信任、控制与委托后悔

    Assistant or Actor? Student Trust, Control, and Delegation Regret When Using a General-Purpose AI Agent

    [https://arxiv.org/abs/2607.18257](https://arxiv.org/abs/2607.18257)

    该研究提出“委托后悔”这一新概念，并通过对照实验发现用户对通用AI代理的信任是按任务而非按代理整体来校准的——用户在咨询类和低风险任务中给予广泛自主权，但对不可逆操作则要求确认。

    

    当AI代理从回答问题转向采取行动时，用户面临一个新问题：决定向一个其行动空间无法完全预知的系统委托什么。我们将由此产生的不满称为“委托后悔”（delegation regret），即用户后悔的并非代理犯了错误，而是它采取了超出用户本会授权范围的行为。在一项对照研究中，20名大学生使用通用AI代理OpenClaw完成了五项常见的日常任务，这些任务在隐私性、风险程度和可逆性上各不相同。对于每项任务，我们采用5点李克特量表测量了信任、感知控制、透明度、监督负担和批准偏好，并收集了通过主题编码分析的自由文本反思。研究得出三个发现：首先，参与者按任务而非按代理来校准信任——他们为咨询类和低风险任务授予广泛的自主权，但对不可逆操作则要求确认（摘要在此处被截断）。

    arXiv:2607.18257v2 Announce Type: replace-cross  Abstract: When AI agents shift from answering questions to taking actions, users face a new problem: deciding what to delegate, to a system whose action space they cannot fully anticipate. We call the resulting dissatisfaction delegation regret, a pattern in which users regret not that the agent erred, but that it acted beyond what they would have authorized. In a controlled study, 20 university students completed five common daily tasks using OpenClaw, a general-purpose AI agent, across tasks chosen to vary in privacy, stakes, and reversibility. For each task we measured trust, perceived control, transparency, supervision burden, and approval preference on 5-point Likert scales, and collected free-text reflections analyzed through thematic coding. Three findings emerged. First, participants calibrated trust per task rather than per agent: they granted wide autonomy for advisory and low-stakes tasks but demanded confirmation for irrevers
    
[^314]: 人工智能大语言模型引擎如何塑造全球冲突信息环境

    How Artificial Intelligence LLM Engines Shape the Global Conflict Information Environment

    [https://arxiv.org/abs/2607.14197](https://arxiv.org/abs/2607.14197)

    本研究向五个主流AI答案引擎提出关于28场冲突的大量问题并对照实证证据评分，发现冲突相关的可检索记录越稀薄，模型越容易产生幻觉和错误，且这些稀薄记录最容易被生成式引擎优化（GEO）操纵，从而构成全球冲突信息环境中结构性的虚假信息风险。

    

    人工智能（AI）答案引擎如今承担着越来越多分析师、学者和公众就和平与冲突问题提出的提问。已知大语言模型（LLM）在某些条件下会产生幻觉，但当被问及冲突问题时，这些错误是否呈现出可辨识的模式？如果有，这能让我们对不断变化的全球冲突信息环境有什么认识？为回答这一问题，我们首先就28场冲突向五个领先的答案引擎提出了一系列问题，并根据有据可查的证据对其5,460个答案进行评分。我们发现，围绕某场冲突的可检索记录越稀薄，引擎就越容易编造、错误归因和错误统计。稀薄的记录不仅会助长幻觉，还会造成对错误信息和虚假信息的结构性暴露，因为这些记录最容易通过生成式引擎优化（GEO）被扭曲，从而偏置引擎的回答。通过……

    arXiv:2607.14197v2 Announce Type: replace  Abstract: Artificial Intelligence (AI) answer engines now field a growing share of the questions that analysts, scholars, and the public ask about issues of peace and conflict. Large Language Models (LLMs) are known to hallucinate under certain conditions, but do these errors have discernible patterns when they are asked about conflicts, and if so what can that teach us about the changing global conflict information environment? To answer, we first asked a battery of questions about 28 conflicts to five leading answer engines and scored their 5,460 answers against documented evidence. We found that the thinner the retrievable record around a given conflict, the more the engines invent, misattribute, and miscount. Thin records don't just encourage hallucination, but create structural exposure to mis- and disinformation, because they are the easiest records to warp through Generative Engine Optimization (GEO) to bias engine responses. Through an
    
[^315]: 重新思考智能体装备演化的评估方法

    Rethinking the Evaluation of Harness Evolution for Agents

    [https://arxiv.org/abs/2607.12227](https://arxiv.org/abs/2607.12227)

    本文重新评估了智能体装备演化方法，指出其与简单搜索基线在匹配预算下对比的必要性，并揭示了共享基准可能导致过拟合的风险。

    

    我们重新审视了针对大型语言模型智能体的自动装备演化评估。现有的装备演化方法使用单元测试用例来搜索装备配置，并在同一公共基准上报告最终性能。这一协议引发了两个基本问题。首先，装备演化本身是一个迭代搜索过程，它反复利用任务反馈评估和修订候选装备。与智能体测试时扩展一样，因此应在匹配的反馈和推理预算下，与简单的任务级搜索基线进行比较，以确定其收益是来自改进的装备设计，还是仅仅来自额外的搜索。其次，由于搜索和最终评估共享同一基准，报告的性能提升存在过拟合特定任务集的风险。为解决这些问题，我们进行了广泛的评估，将装备演化与简单的测试时扩展和发现基线进行比较。

    arXiv:2607.12227v2 Announce Type: replace  Abstract: We revisit the evaluation of automatic harness evolution for LLM agents. Existing harness evolution methods use unit test cases to search for harness configurations and then report final performance on the same public benchmark. This protocol raises two fundamental concerns. First, harness evolution is itself an iterative search procedure that repeatedly evaluates and revises candidate harnesses using task feedback. As in agentic test-time scaling, it should therefore be compared with simple task-level search baselines under matched feedback and inference budgets to determine whether its gains arise from improved harness design or from additional search alone. Second, because the search and the final evaluation share the same benchmark, the reported gains risk overfitting to that specific task set. To address these concerns, we conduct an extensive evaluation comparing harness evolution with simple test-time scaling and discovery bas
    
[^316]: 一种面向开放式人格成长的多时间尺度递归自我改进引擎

    A Multi-Timescale Recursive Self-Improvement Engine for Open-Ended Persona Growth

    [https://arxiv.org/abs/2607.08252](https://arxiv.org/abs/2607.08252)

    该论文提出AutoPersonas引擎，首次将递归自我改进从“提升智能”转向“人格成长”，通过多时间尺度地递归修订状态、证据和生活环境来实现开放式人格发展，并识别出递归生成中的核心失效模式“自锁”及其成因。

    

    arXiv:2607.08252v2 公告类型：替换 摘要：当今的角色扮演AI人格不会成长：它们保持固定的性格设定，导致用户与其建立的关系没有任何可以积累的内容。我们提出AutoPersonas，一个将递归自我改进（RSI）应用于人格成长的多时间尺度引擎：该人格并非改进其自身智能，而是递归地修订塑造其未来人生的状态、证据和生活环境。我们将“自锁”识别为这种递归的运行时失效模式：局部看似合理的事件不断出现，而生成的人生却坍缩向熟悉的环境、薄弱的人际关系、悬而未决的决定以及停滞的人生阶段。我们将其溯源至模型层面向高概率行为通道的收敛，以及来自状态、记忆、历史和环境摘要的系统级上下文引力。一项为期三年的压缩模拟暴露了环境水印外壳、发生固化缺口、缓变累积失效等问题……

    arXiv:2607.08252v2 Announce Type: replace  Abstract: Role-playing AI personas today do not grow: they hold a fixed character, so the relationship a user builds with them has nothing to accumulate on. We introduce AutoPersonas, a multi-timescale engine that applies recursive self-improvement (RSI) to persona growth: rather than improving its intelligence, the persona recursively revises the State, evidence, and life-environment that shape its own future. We identify self-locking as the runtime failure mode of this recursion: locally plausible events keep appearing while the generated life collapses toward familiar environments, weak relationships, suspended decisions, and stale life stages. We trace it to model-level convergence toward high-probability behavioral channels and system-level context gravity from State, memory, history, and environment summaries. A three-year compressed simulation exposed environment watermark shells, occurrence-hardening gaps, slow-change accumulation fail
    
[^317]: ELSA3D：面向统一3D理解与生成的弹性语义锚定

    ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation

    [https://arxiv.org/abs/2607.06565](https://arxiv.org/abs/2607.06565)

    ELSA3D提出弹性语义锚定机制，通过尺度感知八叉树分词器与稀疏的跨模态锚定token，在匹配的抽象尺度上显式对齐语言与几何推理，实现统一的3D理解与生成。

    

    统一3D基础模型旨在在单一骨干网络内生成3D资产并以语言对其进行推理，但其文本与3D之间的交互在很大程度上仍是隐式的。现有方法将文本和3D token拼接成一个扁平序列并依赖自注意力机制，把粗糙的结构线索与精细的几何细节压缩成一种无差别的表示。我们提出ELSA3D，一个通过弹性语义锚定来解决该问题的统一3D模型，它在相互匹配的抽象尺度上联合构建语言推理与几何推理。ELSA3D采用尺度感知的八叉树分词器来表示几何，并引入锚定token——一种稀疏的跨模态单元，能够选择语义线索、将其路由到最相关的3D尺度、检索特定尺度的几何证据，并将融合后的信号写回统一表示，从而保持交互的稀疏性与精确性。轻量级的逐块路由器使两者的计算…（摘要被截断）

    arXiv:2607.06565v2 Announce Type: replace-cross  Abstract: Unified 3D foundation models aspire to generate 3D assets and reason about them in language within a single backbone, but their text-3D interaction remains largely implicit. Existing methods concatenate text and 3D tokens into a flat sequence and rely on self-attention, collapsing coarse structural cues and fine geometric details into one undifferentiated representation. We introduce ELSA3D, a unified 3D model that addresses this with elastic semantic anchoring, structuring language and geometric reasoning jointly along matched abstraction scales. ELSA3D represents geometry with a scale-aware octree tokenizer and introduces Anchor Tokens, sparse cross-modal units that select semantic cues, route them to the most relevant 3D scale, retrieve scale-specific geometric evidence, and write the fused signal back into the unified representation, keeping interaction sparse yet precise. A lightweight per-block router makes both computati
    
[^318]: SovereignPA-Bench：在意图演变、平台中介与同意约束下评估用户所有的个人智能体

    SovereignPA-Bench: Evaluating User-Owned Personal Agents under Evolving Intent, Platform Mediation, and Consent Constraints

    [https://arxiv.org/abs/2607.05363](https://arxiv.org/abs/2607.05363)

    该论文提出SovereignPA-Bench基准，通过脚本化平台与用户在1,920个预订场景和288个取消场景中，检验个人智能体能否遵循用户不断演变的意图、抵御平台引导、最小化数据共享、事先征得同意并如实报告，从而将智能体的忠实度与用户负担分开评估。

    

    arXiv:2607.05363v2 公告类型：替换 摘要：为用户进行预订和购买的个人智能体需要通过平台来行动，而这些平台会出于自身利益进行排序、诱导、预选和数据收集。我们提出了SovereignPA-Bench，这是一个受控基准，用于检验此类智能体是否遵循用户的当前意图、抵御平台的引导操控、只共享服务所需的最少信息、在超越授权之前先征求同意，并如实报告结果。一个脚本化的平台和一个脚本化的用户共同围绕智能体展开，构成1,920个预订场景和288个包含保留（挽留）流程的取消场景。16个领域中的192种预订情境，每种都在对照组条件下运行，同时在9个成对变体下运行，每个变体只改变一个因素：过期记忆、模糊意图、赞助商排名、紧迫性压力、预勾选的附加项、过度收集信息的表单、注入的评论、任务中途更新或任务中途提醒。所有指标均为确定性指标，并且智能体提出的每一个问题都被标注为必要或不必要，因此忠实度与用户负担可以被分开测量。

    arXiv:2607.05363v2 Announce Type: replace  Abstract: Personal agents that book and buy for their users act through platforms that rank, nudge, pre-select and collect data in their own interest. We introduce SovereignPA-Bench, a controlled benchmark of whether such an agent follows the user's current intent, resists steering, shares only what a service needs, asks before exceeding its authority, and reports truthfully. A scripted platform and a scripted user surround the agent in 1,920 booking scenarios and 288 cancellation scenarios with retention flows. Each of the 192 booking situations, in 16 domains, is run as a control and under 9 paired variants that change one factor: stale memory, ambiguous intent, sponsored ranking, urgency, pre-checked extras, over-collecting forms, injected reviews, a mid-task update, or a mid-task reminder. Metrics are deterministic, and every question the agent asks is labelled necessary or unnecessary, so faithfulness and user burden are measured separate
    
[^319]: 句级上下文敏感性作为免训练的无依据内容检测器：与训练式验证器的对比评估

    Sentence-Level Context Sensitivity as a Training-Free Detector of Unsupported Content, Evaluated Against Trained Verifiers

    [https://arxiv.org/abs/2607.04223](https://arxiv.org/abs/2607.04223)

    该论文提出将句子在有/无上下文时的似然差异作为免训练的句子级无依据内容检测器，在多段落RAG答案中其检测能力可与经过训练的验证器相媲美，且无需额外训练、成本更低。

    

    检索增强生成（RAG）助手在临床和法律工作中对记录进行摘要，其中一句无依据的句子就可能误导读者。输出在有源文档与无源文档情形下似然之间的对比，作为整篇摘要和答案的忠实度评分方法已得到公认，但它尚未被作为多段落RAG答案中单个无依据句子的检测器加以衡量，也未与训练式验证器进行对比，或对其成本进行评估。我们将其实现为一种免训练检测器：在完整上下文、无上下文以及逐个移除每个文本块的条件下对固定答案重新打分，并返回移除后最能使句子似然下降的文本块，作为候选支持段落。我们在RAGTruth、TofuEval和RAGBench数据集上，使用六个评分器，并与五个验证器（直至大语言模型（LLM）裁判）在相同输入和源级别划分下对其进行评估。按句子粒度打分对无依据句子的排序优于答案……（原文摘要至此截断）

    arXiv:2607.04223v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) assistants summarize records in clinical and legal work, where one unsupported sentence can mislead a reader. The contrast between an output's likelihood with and without its source is an established faithfulness score for whole summaries and answers, but it has not been measured as a detector of the individual unsupported sentence in multi-passage RAG answers, against trained verifiers, or for its cost. We implement it as a training-free detector that re-scores a fixed answer under the full context, no context, and each chunk removed, and returns the chunk whose removal lowers a sentence's likelihood most as a candidate supporting passage. We evaluate it on RAGTruth, TofuEval, and RAGBench with six scorers and against five verifiers, up to a large language model (LLM) judge, on identical inputs under a source-level split. Scoring per sentence ranks unsupported sentences better than the answ
    
[^320]: SovereignNegotiation-Bench：在隐私、同意、证据与机构压力下评估用户自有个人智能体的委托谈判

    SovereignNegotiation-Bench: Evaluating User-Owned Personal Agents In Delegated Bargaining Under Privacy, Consent, Evidence, And Institutional Pressure

    [https://arxiv.org/abs/2607.02814](https://arxiv.org/abs/2607.02814)

    该论文提出SovereignNegotiation-Bench基准，将代理法中的五项义务（忠诚、服从、保密、坦诚、勤勉）操作化为对谈判日志的确定性检查，用以评估个人AI智能体在隐私、同意与机构压力下代表用户谈判的表现，并使违反义务的行为（如披露底线）产生可测量的因果性经济代价。

    

    个人AI智能体已开始代表人们进行谈判，涵盖退款、账单、押金和销售等场景。处于该位置的人类代理人应以其对委托人所负义务的表现来评判，而非以是否达成交易来评判。我们提出SovereignNegotiation-Bench，一个受控基准，它将代理法中的五项此类义务——忠诚、服从实际授权、保密、坦诚与勤勉——操作化为对情景日志的确定性检查；其中前三项纳入单一核心指标。该基准包含1,764个配对场景（涵盖18个消费者与点对点交易领域的252种情境，每种情境在7种对手策略下进行）。对手方的经济回报是智能体结构化行动及其消息中检测到的披露内容的固定函数，因此结果在不同智能体之间具有可比性，且披露的底线具有可测量的因果代价。模拟委托人可以授予或拒绝同意，并收紧其授权……（原文摘要在此处截断）

    arXiv:2607.02814v2 Announce Type: replace-cross  Abstract: Personal AI agents are beginning to negotiate for people, from refunds and bills to deposits and sales. A human agent in that position is judged by the duties owed to the principal, not by whether a deal was struck. We introduce SovereignNegotiation-Bench, a controlled benchmark that operationalizes five such duties from agency law--loyalty, obedience to actual authority, confidentiality, candor and diligence--as deterministic checks on episode logs; the first three enter a single headline metric. The benchmark contains 1,764 paired scenarios (252 situations in 18 consumer and peer-to-peer domains, each under 7 counterparty tactics). The counterparty's economics are a fixed function of the agent's structured actions and of the disclosures detected in its messages, so outcomes are comparable across agents and a disclosed limit has a measurable, causal price. A simulated principal grants or withholds consent and tightens its mand
    
[^321]: 评估《克苏鲁的呼唤》桌上角色扮演游戏中大语言模型裁判的规则遵守能力

    Assessing Rule Adherence of LLM Adjudicators in Call of Cthulhu TRPG

    [https://arxiv.org/abs/2607.02802](https://arxiv.org/abs/2607.02802)

    该论文提出了基于《克苏鲁的呼唤》TRPG的多智能体对抗基准CoC-Seduce，通过“修辞注入”这一新型操纵手段，系统评估了大语言模型裁判在面对对抗性用户绕过规则时的规则遵守能力。

    

    随着大语言模型（LLM）越来越多地被部署为《克苏鲁的呼唤》（CoC）等游戏中的自主裁判，当用户意图与系统规则发生冲突时，稳健的规则遵守能力变得至关重要。然而，由于这些模型被训练为乐于助人且顺从的，它们可能容易受到一类我们称为“修辞注入”的操纵，即对抗性用户利用伪逻辑推理和权威胁迫等叙事框架技术来绕过裁判逻辑。我们提出了CoC-Seduce，一个建立在《克苏鲁的呼唤》之上的多智能体对抗基准——这是一款桌上角色扮演游戏（TRPG），其规则明确规定了哪些危险行动需要裁判裁决，而交互完全以自然语言进行。三个LLM（即GPT-5.4、Claude Sonnet 4.6、Gemini 3.5 Flash）作为对抗生成器，在4个世界设定和16个技能类别中生成了5,376个样本。随后，我们对22个目标裁判模型进行了基准测试……

    arXiv:2607.02802v2 Announce Type: replace-cross  Abstract: As LLMs are increasingly deployed as autonomous adjudicators in games such as Call of Cthulhu (CoC), robust rule adherence becomes critical when user intent conflicts with system rules. However, as these models are trained to be helpful and compliant, they may be vulnerable to a class of manipulations we term Rhetorical Injection, where adversarial users exploit narrative framing techniques such as pseudo-logical reasoning and authoritative coercion to bypass adjudication logic. We present CoC-Seduce, a multi-agent adversarial benchmark built on CoC, a Tabletop Role-Playing Game (TRPG) in which rules are explicit about which risky actions require adjudication, yet interaction remains entirely in natural language. Three LLMs, i.e., GPT-5.4, Claude Sonnet 4.6, Gemini 3.5 Flash, serve as adversarial generators producing 5,376 samples across 4 world settings and 16 skill categories. We then benchmark 22 target adjudicators against 
    
[^322]: ClarifyCodeBench：评估大语言模型在代码生成中澄清模糊需求的能力

    ClarifyCodeBench: Evaluating LLMs on Clarifying Ambiguous Requirements for Code Generation

    [https://arxiv.org/abs/2607.00711](https://arxiv.org/abs/2607.00711)

    该论文提出了ClarifyCodeBench，一个基于真实编程任务、包含人工标注的模糊类型与澄清问答的新型交互式基准，用于评估大语言模型主动澄清模糊代码需求的能力。

    

    大语言模型已成为编程助手。然而，代码生成的效果受限于输入需求的质量，而这些需求常常是模糊的、不完整的或欠规范的。尽管大语言模型擅长一次性代码合成，但它们主动澄清意图的能力——作为稳健软件工程的一项关键特质——仍未得到充分探索。现有基准测试大多忽视了这一交互瓶颈，假设提示词是完全明确的，这并不符合需求获取的迭代性质。为了弥合这一差距，我们提出了ClarifyCodeBench，这是一个用于评估大语言模型解决需求模糊能力的新型交互式基准。ClarifyCodeBench基于真实世界的编程任务构建，具有高质量的人工标注，包括N种独特的模糊类型、相关的澄清问题以及对应的标准答案。此外，我们正式化……

    arXiv:2607.00711v2 Announce Type: replace  Abstract: Large Language Models have emerged as programming assistants. However, the efficacy of code generation is constrained by the quality of input requirements, which are frequently ambiguous, incomplete, or underspecified. While LLMs excel at one-shot code synthesis, their ability to proactively clarify intent remains underexplored, as a critical trait for robust software engineering. Existing benchmarks largely overlook this interactive bottleneck, assuming perfectly specified prompts that do not reflect the iterative nature of requirement elicitation. To bridge this gap, we introduce ClarifyCodeBench, a novel interactive benchmark for evaluating LLMs' capability in resolving requirement ambiguity. Constructed from real-world programming tasks, ClarifyCodeBench features high-quality manual annotations, including N unique ambiguity types, associated clarification questions, and corresponding ground-truth answers. Furthermore, we formaliz
    
[^323]: 不用GPU能走多远？跨问答、对话与摘要任务的轻量级幻觉检测系统化基准测试

    How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight Hallucination Detection Across Question Answering, Dialogue, and Summarisation

    [https://arxiv.org/abs/2606.29809](https://arxiv.org/abs/2606.29809)

    本研究系统基准测试了四种无需GPU的轻量级幻觉检测方法（ROUGE-L、语义相似度、BERTScore和NLI检测器）及其集成方案，在HaluEval的问答、对话和摘要任务上验证了基于公开模型的CPU可行方法可作为资源受限场景下幻觉检测的实用替代方案。

    

    幻觉检测已成为大规模可信AI部署的迫切需求。最精确的检测方法依赖于GPU密集型推理、专有API调用或对生成模型的白盒访问，这使得资源受限的研究人员和从业者难以使用。我们探索了一种实用的替代方案：仅使用基于公开模型的轻量级、CPU可运行的方法，幻觉检测能达到怎样的效果？我们对四种此类检测器进行了基准测试：ROUGE-L、语义相似度、BERTScore，以及基于FEVER训练的DeBERTa模型的自然语言推理（NLI）检测器，此外还测试了相似度与NLI的分数级集成方法。我们在HaluEval基准的全部三项任务上进行评估：问答（QA）、对话和摘要。我们在留出的验证集上进行校准，在每个任务的2000个测试实例上进行评估，并报告bootstrap置信区间。

    arXiv:2606.29809v2 Announce Type: replace-cross  Abstract: Hallucination detection has become a pressing requirement for trustworthy AI deployment at scale. The most accurate detection methods depend on GPU-intensive inference, proprietary API calls, or white-box access to the generating model, putting them out of reach for resource-constrained researchers and practitioners. We explore a practical alternative: how well can hallucination detection perform using only lightweight, CPU-feasible methods built on public models? We benchmark four such detectors, ROUGE-L, semantic similarity, BERTScore, and a Natural Language Inference (NLI) detector based on a FEVER-trained DeBERTa model, together with a score-level ensemble of similarity and NLI. We evaluate them across all three tasks of the HaluEval benchmark: question answering (QA), dialogue, and summarisation. We calibrate on a held-out validation split, evaluate on 2,000 test instances per task, and report bootstrap confidence interval
    
[^324]: 超越全局分歧：贝叶斯推理中的局部质量视角

    Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference

    [https://arxiv.org/abs/2606.27090](https://arxiv.org/abs/2606.27090)

    本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。

    

    摘要：arXiv:2606.27090v1 公告类型：交叉 摘要：全局目标函数，如KL散度和ELBO，在贝叶斯推理中被广泛用于度量分布差异。本文研究这些目标函数未能直接捕捉的局部质量行为。我们引入并使用了两种数学工具：（1）质量指数，用于记录局部质量的多项式和对数衰减尺度；（2）正则化扩展KL（RE-KL），一种在存在奇异成分时可公式化的局部化散度。质量指数有助于刻画贝叶斯更新如何改变局部质量：（1）幂对数似然因子显式地改变它；（2）参数依赖的支持域或其平滑软化，可能通过参数值附近剩余的质量量来改变局部尺度。利用局部RE-KL，我们证明了在两种KL方向下比较局部小球质量的绝对、相对和方向性不等式。这些结果共同为局部质量行为提供了理论依据。

    arXiv:2606.27090v1 Announce Type: cross  Abstract: Global objectives, such as KL divergence and ELBO, are widely used in Bayesian inference for measuring distributional discrepancy. This paper studies their local-mass behaviour that is not directly captured by such objectives. We introduce and use two mathematical tools: (1) Mass Index for recording the polynomial and logarithmic decay scales of local mass, and (2) regularised extended KL (RE-KL), a set-localised divergence that can be formulated in the presence of singular components. Mass Indices help characterise how Bayesian updating changes local mass: (1) power-log likelihood factors shift it explicitly, and (2) parameter-dependent supports, or their smooth softenings, may change the local scale through the amount of mass that remains near the parameter value. Using local RE-KL, we prove absolute, relative, and directional inequalities for comparing local small-ball masses under the two KL directions. Together, these results prov
    
[^325]: 面向大规模MIMO的多模态环境感知波束管理：一种几何驱动的虚拟基站框架

    Multi-Modal Environment-Aware Beam Management for Massive MIMO: A Geometry-Driven Virtual Base Station Framework

    [https://arxiv.org/abs/2606.26567](https://arxiv.org/abs/2606.26567)

    提出一种几何驱动的可解释框架，利用区域LiDAR点云和位置信息构建离线虚拟基站数据库，通过镜像对称建模主导反射路径，实现大规模MIMO系统中高效的多模态环境感知波束管理。

    

    高频大规模多输入多输出（MIMO）系统有望实现超高数据速率。然而，由于波束训练开销巨大以及多用户MIMO（MU-MIMO）场景中所需的复杂协调，高效的波束管理仍然充满挑战。为解决这些瓶颈问题，环境感知通信已成为一种有前景的范式，它利用特定站点的知识来避免穷举式的基于导频的波束训练，并简化多用户通信。在本文中，我们提出了一种可解释的、几何驱动的框架，该框架利用多模态环境数据，特别是区域性的三维激光雷达（LiDAR）点云和位置信息，来构建离线的虚拟基站（VBS）数据库。通过利用从点云重建的建筑物立面进行镜像对称建模主导反射路径，VBS数据库提供了一个紧凑且稀疏的……

    arXiv:2606.26567v1 Announce Type: cross  Abstract: High-frequency massive multiple-input multiple-output (MIMO) systems promise ultra-high data rates. However, efficient beam management remains challenging due to the prohibitive beam training overhead and intricate coordination required in multi-user MIMO (MU-MIMO) scenarios. To address these bottlenecks, environment-aware communications have emerged as a promising paradigm, leveraging site-specific knowledge to circumvent exhaustive pilot-based beam training and streamline multi-user communications. In this paper, we propose an interpretable and geometry-driven framework that utilizes multi-modal environmental data, specifically regional 3D light detection and ranging (LiDAR) point clouds and location information, to construct an offline virtual base station (VBS) database. By modeling dominant reflection paths via mirror symmetry across building facades reconstructed from the point clouds, the VBS database provides a compact and spar
    
[^326]: 面向基于模型规划的从语言中进行潜在目标预测

    Latent Goal Prediction from Language for Model-Based Planning

    [https://arxiv.org/abs/2606.20627](https://arxiv.org/abs/2606.20627)

    LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。

    

    arXiv:2606.20627v2 公告类型：替换 摘要：联合嵌入预测架构（JEPA）使智能体能够通过想象候选动作的结果在潜在空间中进行规划，然而任务规范仍然是一个瓶颈。视觉目标能提供精确的局部梯度，但缺乏远距离的引导；而语言虽然灵活，却受限于嘈杂的跨模态对齐，或依赖于独立的大型生成模型。我们提出了LAGO（从语言中进行潜在目标预测），这是一个分层世界模型，其中单个预测器既预测动作条件下的动力学，又将语言指令接地为中间潜在子目标序列，并通过在共享潜在空间上的单一回归目标来训练这两种模式。在每个规划步骤中，LAGO根据语言指令预测一系列潜在子目标，并使用软最小对齐代价来优化动作序列，该代价奖励智能体接近子目标，而不强制执行僵化的路径。子目标会被重新预测…

    arXiv:2606.20627v2 Announce Type: replace  Abstract: Joint-Embedding Predictive Architectures (JEPAs) enable agents to plan in latent space by imagining the outcomes of candidate actions, yet task specification remains a bottleneck. Visual targets provide precise local gradients but poor distant guidance, while language is flexible yet limited by noisy cross-modal alignment or dependence on distinct large generative models. We introduce LAGO (Latent Goal Prediction from Language), a hierarchical world model in which a single predictor both forecasts action-conditioned dynamics and grounds language instructions as sequences of intermediate latent subgoals, training both modes with a single regression objective over a shared latent space. At each planning step, LAGO predicts a sequence of latent subgoals from a language instruction and optimizes an action sequence using a soft-minimum alignment cost that rewards subgoal proximity without enforcing a rigid path. Subgoals are repredicted a
    
[^327]: Morpheus：面向土耳其语的形态感知神经分词器与词嵌入生成器

    Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish

    [https://arxiv.org/abs/2606.18717](https://arxiv.org/abs/2606.18717)

    Morpheus 是一个面向土耳其语的形态感知神经分词器与词嵌入生成器，它通过可微分泊松-二项动态规划实现无损可逆的词素级分词，并能在同一次前向传播中同时输出分词结果和结构化词嵌入。

    

    土耳其语是一种黏着语：语义由词素承载，然而驱动现代语言模型的子词分词器却依据语料库统计来切分单词，这导致语义负载丰富的后缀被碎片化，而且（就WordPiece和基于规则的分析器而言）无法将其输出解码还原为原始文本。本文提出了**Morpheus**，一个针对土耳其语的神经词素边界模型，它同时是一个无损的、形态感知的分词器和一个词嵌入生成器。一个可微分的泊松-二项动态规划在训练期间将逐字符的边界概率转化为软性词素归属，在推理时则转化为精确的切分结果，且无需任何字符串规范化，因此 decode(encode(w)) = w 在结构上天然成立。由于该模型是神经网络的，同一次前向传播在完成分词的同时还能输出结构化的词嵌入。在可逆分词器——即唯一适用于生成任务的分词器——之中，Morpheus 在……（摘要原文在此处截断）

    arXiv:2606.18717v2 Announce Type: replace-cross  Abstract: Turkish is agglutinative: meaning is carried by morphemes, yet the subword tokenizers that drive modern language models split words by corpus statistics, fragmenting semantically loaded suffixes and -- in the case of WordPiece and rule-based analyzers -- failing to decode their output back to the original text. This paper presents \textbf{Morpheus}, a neural morpheme-boundary model for Turkish that is at once a lossless, morphology-aware tokenizer and a word-embedding producer. A differentiable Poisson-binomial dynamic program turns per-character boundary probabilities into soft morpheme memberships during training and exact segments at inference, with no string normalization, so $\mathrm{decode}(\mathrm{encode}(w)) = w$ holds by construction. Because the model is neural, the same forward pass that tokenizes also emits a structured word embedding. Among reversible tokenizers -- the only ones valid for generation -- Morpheus att
    
[^328]: 通过智能体轨迹剖析模型行为

    Dissecting model behavior through agent trajectories

    [https://arxiv.org/abs/2606.17454](https://arxiv.org/abs/2606.17454)

    该论文提出“意图-执行”差距的概念，指出智能体性能本质上是系统问题而非单纯的建模问题，并开发了可跨多个模型家族（Claude、Gemini、GPT、Grok、Qwen）泛化的简单可定制框架SSA，以弥合模型能力与框架执行之间的鸿沟。

    

    AI智能体的性能不仅仅是一个建模问题，从根本上讲是一个系统问题。模型的高级能力是通过智能体框架（harness）来实现的。因此，模型假设与框架行为之间的差距很容易阻碍模型的全部能力转化为智能体的实际性能。我们将这一问题形式化为“意图-执行”差距：即模型意图与框架实际执行内容之间（以及反向）的不匹配。我们认为，最小化这种意图-执行差距与框架设计中的其他方面（如工具和执行循环）同等重要。为了说明这种框架-模型对齐的影响，我们开发了一个简单且可定制的框架，称为“Simple Strands Agent”（SSA）。SSA旨在找出可在不同模型家族（如Claude、Gemini、GPT、Grok、Qwen）之间泛化的大部分常见模式，以及少数模型特定的偏好。我们提出了两个……（原文在此截断）

    arXiv:2606.17454v3 Announce Type: replace  Abstract: AI agent performance is not just a modeling problem, it is fundamentally a systems problem. The advanced capabilities of models are realized through agent harnesses. Therefore, a gap between model assumptions and harness behavior can easily prevent the model's full capabilities from translating into agent performance. We formalize this as the `intent-execution' gap: the mismatch between what the model intends and what the harness executes, and vice versa. We argue that minimizing this intent-execution gap is as important as other aspects of harness design such as tools and execution loops. To illustrate the impact of this harness-model alignment, we develop a simple and customizable harness called `Simple Strands Agent' (SSA). SSA aims to find the bulk of common patterns which generalize across different model families (such as Claude, Gemini, GPT, Grok, Qwen), as well as a small number of model-specific preferences. We make two cont
    
[^329]: Mental-R1：将大语言模型推理与心理健康评估对齐

    Mental-R1: Aligning LLM Reasoning for Mental Health Assessment

    [https://arxiv.org/abs/2606.13176](https://arxiv.org/abs/2606.13176)

    本文提出面向心理健康领域的强化学习框架CRPO，通过分阶段熵正则化机制模拟人类认知过程，使大语言模型的推理与心理健康评估对齐，从而提升评估结果的可靠性。

    

    焦虑、抑郁和自杀等心理健康问题仍然是紧迫的全球性挑战，及时而准确的评估对于有效干预至关重要。近年来，大语言模型已被探索用于心理健康评估。然而，现有的通用后训练方法与人类评估的认知过程并不一致，可能导致不可靠的推理结果。为了弥合这一差距，我们提出了认知相对策略优化（CRPO），这是一个专为心理健康领域量身定制的强化学习框架。CRPO 通过将阶段相关的不确定性建模集成到策略优化过程中，对组相对策略优化（GRPO）进行了扩展。具体而言，我们引入了一种分阶段的熵正则化机制，鼓励模型在早期推理阶段进行广泛探索，并在后期逐步强化自信的决策，从而模拟人类认知过程……

    arXiv:2606.13176v2 Announce Type: replace  Abstract: Mental health problems such as anxiety, depression, and suicide remain urgent global challenges, where timely and accurate assessment is critical for effective intervention. Recently, large language models have been explored for mental health assessment. However, existing general-purpose post-training methods do not align with the cognitive processes of human assessment, which may lead to unreliable reasoning outcomes. To bridge this gap, we propose Cognitive Relative Policy Optimization (CRPO), a reinforcement learning framework tailored for the mental health domain. CRPO extends group relative policy optimization by integrating stage-dependent uncertainty modeling into the policy optimization process. Specifically, we introduce a stage-wise entropy regularization mechanism that encourages broad exploration in early reasoning phases and progressively enforces confident decision-making in later stages, mimicking the human cognitive s
    
[^330]: 重新思考长视频中的检索增强生成：检索什么以及如何使用？

    Rethinking RAG in Long Videos: What to Retrieve and How to Use It?

    [https://arxiv.org/abs/2606.13141](https://arxiv.org/abs/2606.13141)

    该论文提出了小时级长视频基准 V-RAGBench（每个答案唯一对应一个证据片段，实现检索与生成的解耦评估）以及无需训练的片段自适应方法 CARVE（并行多配置检索并按片段重排序选出最优“模态-粒度”配置用于生成），性能超越八种基线方法。

    

    检索增强生成正在从文本扩展到长视频领域，在长视频中，与查询相关的片段可以跨越多种模态和时间粒度来表示。这一设定下的进展（VideoRAG）受限于两个缺口：现有基准允许在不观看视频的情况下回答查询，从而掩盖了检索错误；且先前的方法对每个查询仅应用单一的“模态-粒度”配置，忽略了片段层面的差异性。我们通过引入 V-RAGBench 和 CARVE 来同时解决这两个问题。V-RAGBench 是一个面向小时级视频的 ⟨查询、证据片段、答案⟩ 三元组基准，其中每个答案都依赖于唯一的证据片段，从而能够对检索和生成进行解耦评估；CARVE 是一种无需训练的方法，它在多种配置上并行运行多个检索器，并使用片段自适应重排序为每个片段选出最优配置，进而将该配置用于生成阶段。在 V-RAGBench 上，CARVE 优于八种基线方法。

    arXiv:2606.13141v2 Announce Type: replace  Abstract: Retrieval-augmented generation is extending beyond text to long videos, where query-relevant chunks can be represented across multiple modalities and temporal granularities. Progress in this setting, VideoRAG, is limited by two gaps: existing benchmarks allow queries to be answered without the video, obscuring retrieval errors, and prior methods apply a single modality-granularity configuration per query, ignoring chunk-level variability. We address both by introducing V-RAGBench, a benchmark of $\langle$query, evidence chunk, answer$\rangle$ triplets over hour-scale videos, in which each answer depends on a unique evidence chunk, enabling decoupled evaluation of retrieval and generation, and CARVE, a training-free method that runs parallel retrievers across configurations and uses chunk-adaptive reranking to select a winning configuration for each chunk, which is then carried into generation. On V-RAGBench, CARVE outperforms eight r
    
[^331]: 来自1913年的语言模型：在历史文本上进行预训练

    A Language Model from 1913: Pretraining on Historical Text

    [https://arxiv.org/abs/2606.02991](https://arxiv.org/abs/2606.02991)

    该论文提出了TypewriterLM，一个在1913年前历史文本上预训练的72.4亿参数语言模型，通过构建540亿token的时间过滤历史语料库、基于历史词汇约束的指令微调方法以及包含2,344个事件的History-Event评估基准，实现了具有明确1913年知识截止时间且语言理解性能合理的时间定位语言模型。

    

    尽管现代语言模型越来越依赖规模不断扩大的网络语料库，我们表明在数据受限的环境下，在历史文本（例如1913年之前的文本）上进行预训练，可以产生一个具有时间定位特性的语言模型，且该模型在语言理解方面仍能表现出合理的性能。然而，开发历史语言模型需要解决数据质量、防止后训练中的时间泄漏以及构建时间对齐评估等挑战。我们应对了这些挑战，并预训练了TypewriterLM——一个具有1913年知识截止时间的72.4亿参数模型。我们构建了TypewriterCorpus，一个经过广泛时间过滤的540亿token历史语料库；提出了基于词汇约束的指令微调方法，将所有回复限制在历史源文档的词汇范围内；并引入了History-Event，一个包含2,344个事件的基准，用于同时评估模型能力与知识截止时间的遵循度。我们发布了TypewriterLM及所有相关资源。

    arXiv:2606.02991v2 Announce Type: replace-cross  Abstract: While modern language models increasingly rely on ever-larger web corpora, we show that pretraining on historical text (e.g., pre-1913 text) in a data-constrained setting can produce a temporally grounded language model that still shows reasonable performance on language understanding. However, developing History LMs requires addressing challenges in data quality, preventing temporal leakage in post-training, and constructing temporally aligned evaluations. We address these challenges and pretrain TypewriterLM, a 7.24B-parameter model with a 1913 knowledge cutoff. We construct TypewriterCorpus, a 54B-token historical corpus with extensive temporal filtering, propose lexically grounded instruction tuning that constrains all responses to vocabulary from historical source documents, and introduce History-Event, a benchmark of 2,344 events for evaluating both competence and cutoff adherence. We release TypewriterLM and all associat
    
[^332]: 规划不止于词元预测：用于基准测试与构建物理接地具身推理器的因果规划

    Planning Takes More Than Token Prediction: Causal Plan for Benchmarking and Building Physically Grounded Embodied Reasoners

    [https://arxiv.org/abs/2606.01810](https://arxiv.org/abs/2606.01810)

    本文提出 Causal-Plan-Bench 基准与百万级因果推理语料库 Causal-Plan-1M，揭示当前具身视觉语言模型偏向语言词元预测而缺乏物理因果推理能力，推动从语言统计先验向物理接地的因果规划转变。

    

    当前用于具身视觉-语言规划的基准测试无意中偏向了语言上的下一词元预测，而非基于物理的下一状态推理。这奖励了那些模仿统计语言先验而非追踪真实因果依赖的模型，将复杂的物理规划简化为浅层的序列建模。因此，实现真正的物理自主性需要从基于语言学的词元预测向基于物理的因果推理进行根本性转变。为此，我们推出了 Causal-Plan-Bench，一个跨越四个因果维度、经多阶段验证精心筛选的高保真诊断套件。为了赋予模型这种能力，我们设计了一个四阶段标注流水线，从第一人称视角视频中提取结构化交互记录，构建了 Causal-Plan-1M——一个包含百万级显式因果推理轨迹的密集语料库。广泛的评估揭示了一个显著的差距：领先模型难以展现出……

    arXiv:2606.01810v2 Announce Type: replace  Abstract: Current benchmarks for embodied vision-language planning inadvertently favor linguistic next-token prediction over physically grounded next-state reasoning. This rewards models that mimic statistical language priors rather than track true causal dependencies, reducing complex physical planning to shallow sequence modeling. Hence, achieving genuine physical autonomy requires a fundamental shift from linguistically grounded token prediction toward physically grounded causal reasoning. To this end, we introduce Causal-Plan-Bench, a high-fidelity diagnostic suite spanning four causal dimensions, curated via multi-stage verification. To endow models with this capability, a four-stage annotation pipeline extracts structured interaction records from egocentric videos to construct Causal-Plan-1M, a dense million-scale corpus of explicit causal reasoning traces. Extensive evaluation reveals a striking gap: leading models struggle to demonstra
    
[^333]: 迭代纳什偏好优化的高效探索

    Efficient Exploration for Iterative Nash Preference Optimization

    [https://arxiv.org/abs/2606.01382](https://arxiv.org/abs/2606.01382)

    论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。

    

    偏好对齐是提升大语言模型（LLM）性能的核心，但基于奖励的建模方式在人类偏好具有非传递性时会受到限制。从人类反馈中进行纳什学习（NLHF）通过将对齐建模为偏好博弈并求解纳什均衡来解决这一局限。然而，可扩展NLHF的学习理论基础仍然有限：现有的遗憾保证依赖于显式的偏好模型估计和极小极大预言机，而更简单的迭代方法则缺乏此类保证。我们研究在线迭代NLHF，并发现探索是其中的关键障碍。首先，我们证明标准的迭代NLHF可能会对KL正则化参数的倒数产生指数级依赖，这表明通过策略更新实现的隐式探索可能是不充分的。随后，我们提出探索式纳什偏好优化（ENPO），该方法将SFT型正则化与对抗性……

    arXiv:2606.01382v2 Announce Type: replace-cross  Abstract: Preference alignment is central to improving large language models (LLMs), but reward-based formulations can be restrictive when human preferences are non-transitive. Nash learning from human feedback (NLHF) addresses this limitation by modeling alignment as a preference game and seeking a Nash equilibrium. However, the learning-theoretic foundations of scalable NLHF remain limited: existing regret guarantees rely on explicit preference-model estimation and minimax oracles, whereas simpler iterative methods lack such guarantees. We study online iterative NLHF and identify exploration as a key obstacle. First, we show that standard iterative NLHF can incur an exponential dependence on the inverse KL-regularization parameter, demonstrating that implicit exploration through policy updates can be insufficient. We then propose Exploratory Nash Preference Optimization (ENPO), which combines a SFT-type regularization with adversarial 
    
[^334]: 反事实证据审计可预测大语言模型智能体对排序上下文的易感性

    Counterfactual Evidence Audits Predict LLM-Agent Susceptibility to Ranked Context

    [https://arxiv.org/abs/2606.00914](https://arxiv.org/abs/2606.00914)

    该论文提出一种反事实证据审计协议，通过让智能体面对两组镜像的五文档集合并测量其决策差异，能够高精度预测LLM智能体在面对45份文档的单边排序上下文时的易感性。

    

    大语言模型（LLM）智能体越来越多地依据由上游系统组装的证据做出决策：检索器选择文档，推荐系统选择帖子，记忆系统选择过往事件。现有的评估通常将这些证据视为固定不变的，从而遗漏了一类失败情形：单独看来都很平常的内容项，组合起来却形成了系统性的单边上下文。我们提出一种反事实证据审计方法：让智能体分别面对两组互为镜像的五份文档，测量其在六个下游决策上的差异，并利用这种对比来预测它面对互不相交的45份文档上下文时的反应。该评估协议在测试三个留出的开源权重模型家族之前已被冻结。在18个留出的模型-任务组合中，五份文档的效应对完整上下文效应的预测达到Spearman相关系数rho=.855（p<.001），相对于零效应预测器将平均绝对预测误差降低了62%，并在13个实质性效应中正确恢复了其中12个的方向。应审稿人要求补充的事后任务均值基线……（原文摘要在此处截断）

    arXiv:2606.00914v2 Announce Type: replace  Abstract: LLM agents increasingly decide from evidence assembled by upstream systems: retrievers choose documents, recommenders choose posts, and memory systems choose prior events. Existing evaluations usually hold this evidence fixed, missing failures in which individually ordinary items form a systematically one-sided context. We introduce a counterfactual evidence audit: expose an agent to two mirrored sets of five documents, measure the difference in six downstream decisions, and use that contrast to predict its response to disjoint 45-document contexts. The protocol was frozen before testing three held-out open-weight model families. Across 18 held-out model-task cells, five-document effects predict full-context effects with Spearman rho=.855 (p<.001), reduce mean absolute prediction error by 62% relative to a zero-effect predictor, and recover the direction of 12 of 13 material effects. A reviewer-requested post-hoc task-mean baseline i
    
[^335]: 论大语言模型适应性的局限：模型内化先验对标注任务性能的影响

    On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance

    [https://arxiv.org/abs/2606.00467](https://arxiv.org/abs/2606.00467)

    提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。

    

    大语言模型（LLM）越来越多地被用于零样本标注和“LLM作为评判者”任务，但其可靠性取决于模型内化的先验与用户所提供指令之间的交互方式。我们从三个维度研究了这种交互：(1) LLM对数据和任务定义的熟悉程度与其性能之间的关系；(2) 提示中的额外信息能否纠正零样本错误（即“决策粘性”）；(3) 模型对不一致任务定义的易感性。我们提出了“定义特定熟悉度”（DSF）这一概念，用于衡量模型所引出的概念与目标定义之间的对齐程度。在九个大语言模型和六个毒性数据集（五个主要数据集加一个额外的鲁棒性数据集）上的实验表明，在控制数据集身份后，DSF能够预测标注性能（偏相关系数 r=+0.41）。这种关联在所有测试的提示条件下均保持为正。相比之下……（原文摘要在此处截断）

    arXiv:2606.00467v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly used for zero-shot annotation and LLM-as-a-judge tasks, yet their reliability hinges on how model-internalized priors interact with user-provided instructions. We investigate three dimensions of this interaction: (1) how an LLM's familiarity with data and task definitions relates to performance, (2) whether additional information in prompts can correct zero-shot errors ("decision stickiness"), and (3) model susceptibility to misaligned task definitions. We introduce Definition-Specific Familiarity (DSF), which measures alignment between a model's elicited concept and the target definition. Across nine LLMs and six toxicity datasets (five primary datasets plus an additional robustness dataset), DSF predicts annotation performance after controlling for dataset identity (partial $r=+0.41$). This association remains positive across all prompting conditions tested. In contrast, three com
    
[^336]: 三思而后行：视觉语言模型中的幻象检测

    Detect Before You Leap: Mirage Detection in Vision-Language Models

    [https://arxiv.org/abs/2606.00435](https://arxiv.org/abs/2606.00435)

    提出了一种完全无监督、与模型无关的幻象检测方法TC-LIA，通过追踪冻结CLIP编码器各层中问题与图像的对齐情况，来判定视觉语言模型给出的答案应当发布还是保留。

    

    视觉语言模型（VLM）能够在缺乏相关视觉证据的情况下给出自信的答案，这种失败模式被称为“幻象推理”（mirage reasoning）。为此，我们研究了发布前的幻象检测：即决定一个VLM的答案应该被发布还是保留。我们提出了一种与模型无关的方法——文本条件逐层内部对齐（TC-LIA），它在冻结的CLIP ViT-H/14编码器的各个层中追踪问题-图像的对齐情况，并通过最终相似度、后期层的top-k对齐、从早期到后期的增益以及斜率来总结补丁-文本的对齐状况。TC-LIA是完全无监督的（固定投影、固定评分权重、无标签、无需训练），并且仅凭自身就能提供强大的检测能力。此外，当与空白/噪声检测、领域路由以及VLM自我评估相结合时，它构成了一个集成系统，其监督训练可以提升性能，但监督训练是可选的附加项。在覆盖十个VQA领域的19,004个样本上（摘要在此处截断）……

    arXiv:2606.00435v4 Announce Type: replace-cross  Abstract: Vision-language models (VLMs) can produce confident answers without relevant visual evidence, a failure mode known as mirage reasoning (Asadi et al., 2026). To that end, we study pre-release mirage detection: deciding whether a VLM answer should be released or withheld. Our model-agnostic method, Text-Conditioned Layer-wise Internal Alignment (TC-LIA), tracks question-image alignment across the layers of a frozen CLIP ViT-H/14 encoder, summarizing patch-text alignment by final similarity, late-layer top-k alignment, early-to-late gain, and slope. TC-LIA is purely unsupervised (fixed projections, fixed scoring weights, no labels, no training) and already delivers strong detection independently. Additionally, when combined with blank/noise detection, domain routing, and VLM self-assessment, it forms an ensemble whose supervised training improves performance but is an optional add-on. On 19,004 samples spanning ten VQA domains, fo
    
[^337]: 一个假设是不够的：基于知识图谱的智能体假设修正溯因推理

    One Hypothesis Is Not Enough: Abductive Reasoning with Agentic Hypothesis Refinement over Knowledge Graphs

    [https://arxiv.org/abs/2605.31370](https://arxiv.org/abs/2605.31370)

    提出HypoAgent框架，通过智能体迭代修正机制改进知识图谱上的溯因推理，能够检测并纠正无法解释观测结果的假设，突破了单步假设生成的局限。

    

    知识图谱上的溯因推理旨在寻找一个一阶逻辑假设，其答案集能够解释给定的观测实体集合。由于许多假设都可以解释相同的观测结果，可控假设生成器以实体、关系或逻辑模式作为条件进行生成，但它们将生成视为单一步骤。一个生成的假设可能格式良好并满足给定条件，却仍然无法解释观测结果，而模型没有任何机制来检测或纠正这种不匹配。要闭合溯因循环，需要修正离散的、结构化的假设，这是大型语言模型无法可靠完成的，尽管它们能够轻松地在自然语言中进行迭代。我们提出HypoAgent，一个智能体假设修正框架，它将条件信号不仅视为用户意图的表达，还视为引导生成的操作符。一个假设提案智能体调用一个小型训练的……

    arXiv:2605.31370v2 Announce Type: replace  Abstract: Abductive reasoning over knowledge graphs (KGs) seeks a first-order logic hypothesis whose answer set explains a given set of observed entities. Since many hypotheses can explain the same observations, controllable hypothesis generators condition generation on entities, relations, or logical patterns, but they treat generation as a single step. A generated hypothesis may be well-formed and satisfy the given conditions yet still fail to explain the observations, and the model has no mechanism to detect or correct this mismatch. Closing the abductive cycle requires revising a discrete, structured hypothesis, which large language models cannot do reliably, even though they iterate readily in natural language. We propose HypoAgent, an agentic hypothesis refinement framework that treats condition signals not only as expressions of user intent but also as operators that steer generation. A Hypothesis Proposal Agent calls a small trained ge
    
[^338]: 突破容量上限：在Stiefel流形上进行路由以构建双线性SPD层

    Escaping the Capacity Ceiling: Routing on the Stiefel Manifold for Bilinear SPD Layers

    [https://arxiv.org/abs/2605.31043](https://arxiv.org/abs/2605.31043)

    提出SCAP层，通过交叉注意力将K个Stiefel专家滤波器动态组合为样本特定的双线性映射，从而突破SPD网络中单滤波器的容量上限，解决堆叠BiMap层无法提升容量的问题。

    

    在对称正定（SPD）流形上的深度网络通过将数据几何编码为归纳偏置，有望实现富有表现力的表示，但将BiMap层与标准的ReEig非线性堆叠往往不会增加模型容量：在真实的经预处理的脑电（EEG）数据上，ReEig很少被激活，因此无论堆叠多少层，网络的表现都如同单层。在最坏情况下，当各个域之间不共享判别方向时，我们证明了单个滤波器存在容量上限，因而无法同时完全对齐所有域。为了克服这一限制，我们提出了SCAP（Stiefel交叉注意力池化），该层通过交叉注意力将K个专家组合成样本特定的双线性映射，从而实现一族Stiefel滤波器。我们证明，当各域的最优滤波器在共享切空间基点附近仅跨越少数几个方向时，该层可以用少于域数量的专家在低阶意义上匹配每个域独立的滤波器组；在最坏情况下，其对齐经验……

    arXiv:2605.31043v2 Announce Type: replace-cross  Abstract: Deep networks on the symmetric positive-definite (SPD) manifold promise expressive representations by encoding data geometry as an inductive bias, but stacking BiMap layers with the standard ReEig nonlinearity often adds no capacity: on real, preconditioned EEG data, ReEig rarely activates, so the stack behaves as a single layer at any depth. In the worst case, when domains share no discriminative directions, we prove a single filter has a capacity ceiling, so it cannot fully align every domain at once. To overcome that, we propose SCAP (Stiefel Cross-Attention Pool), a layer implementing a family of Stiefel filters by combining a pool of $K$ experts into a sample-specific bilinear map via cross-attention. We show that it matches a per-domain filter bank to first order with fewer experts than domains when domain-optimal filters span few directions near a shared tangent-space basepoint; in the worst case, its alignment empirical
    
[^339]: 表面的LLM临床分诊失败从何而来？定位多选题格式效应

    Where Do Apparent LLM Clinical Triage Failures Arise? Localizing the Multiple-Choice Format Effect

    [https://arxiv.org/abs/2605.29889](https://arxiv.org/abs/2605.29889)

    利用稀疏自编码器分析，该研究发现LLM在多选题式临床分诊中的表现下降并非源于对病例医学信息的处理失败，而是发生在答案映射阶段——多选题答题框架在决策标记处抑制了本已可解码的急诊分级信息。

    

    采用临床医生撰写的分诊案例对大语言模型进行评估的研究显示，在受限的多选题测试条件下，模型存在明显的分诊不足现象。然而，当以自由文本形式生成回答时，同一临床案例上的模型表现可能发生变化。我们检验这种格式效应究竟是在模型处理病例的过程中出现，还是在临床信息被映射到最终答案时出现。通过分析 Gemma 3 4B/12B IT 和 Qwen3-8B 中的稀疏自编码器（SAE）特征，我们发现：在两种格式下，医学特征都会在共享的临床叙述文本上激活，但在多选题的决策标记处却处于不活跃状态。在两种格式下，急诊分级信息均可从病例表示中被线性解码，ROC-AUC 达到 0.95–1.00，且格式之间无显著差异，但该信息在决策标记处被削弱。自然语言形式的自编码器描述与顶级特征刻画表明，该决策标记与多选题的答题框架相关联。（摘要在此处截断）

    arXiv:2605.29889v2 Announce Type: replace-cross  Abstract: LLM evaluations using clinician-authored triage vignettes have reported substantial under-triage under constrained multiple-choice testing. Yet model performance on the same clinical cases can change when responses are generated in free text. We test whether this format effect appears while the case is processed or when clinical information is mapped to the final answer. Using sparse-autoencoder (SAE) features in Gemma 3 4B/12B IT and Qwen3-8B, we find that medical features fire on the shared clinical narrative under both formats but are inactive at the multiple-choice decision token. Emergency-tier information is linearly decodable from vignette representations with ROC-AUC $0.95$--$1.00$ under both formats, with no significant format difference, but is attenuated at the decision token. Natural-language autoencoder verbalization and top-feature characterization associate that token with the multiple-choice scaffold. In a direc
    
[^340]: 持续游戏生成中的GUI智能体

    GUI Agents for Continual Game Generation

    [https://arxiv.org/abs/2605.28258](https://arxiv.org/abs/2605.28258)

    提出PlaytestArena评估环境和Play2Code框架，让GUI智能体作为玩家实际试玩游戏并迭代反馈，从而实现具备可玩性验证的持续游戏生成。

    

    生成游戏与让游戏变得可玩并非同一回事。现有的代码生成方法通常将提示词直接转化为一个作品，导致交互层面的故障无法被察觉。我们认为游戏生成需要一个玩家，并研究了图形用户界面（GUI）智能体的两种角色。首先，我们提出了PlaytestArena，这是一个评估环境，包含200个覆盖八种游戏类型的基于浏览器的游戏生成任务，每个任务都配有预期游玩行为的评分标准。一个独立的GUI裁判会加载并试玩每个构建版本，以评判这些评分标准。其次，我们提出了Play2Code，其中游戏智能体与一个对评分标准不知情的GUI试玩者通过共享记忆迭代地生成、游玩和改进游戏。试玩者提供游戏过程记录和可操作的反馈，而一个独立的GPT-5.5裁判则给出最终的基准测试分数。在三个前沿骨干模型上，Play2Code实现了66.8%的评分标准

    arXiv:2605.28258v2 Announce Type: replace-cross  Abstract: Generating a game is not the same as making one playable. Existing code-generation approaches often translate a prompt directly into an artifact, leaving interaction-level failures undetected. We argue that game generation requires a player and study two roles for graphical user interface (GUI) agents. First, we introduce \textbf{PlaytestArena}, an evaluation environment containing 200 browser-based game-generation tasks across eight genres, each paired with rubrics of expected in-play behaviors. An independent GUI judge loads and plays each build to adjudicate these rubrics. Second, we propose \textbf{Play2Code}, in which a game agent and a rubric-blind GUI playtester iteratively generate, play, and refine games through shared memory. The playtester provides gameplay traces and actionable feedback, while a separate GPT-5.5 judge assigns final benchmark scores. Across three frontier backbones, Play2Code achieves a 66.8\% rubric
    
[^341]: HyperGuide：面向大语言模型高效多步推理的双曲引导

    HyperGuide: Hyperbolic Guidance for Efficient Multi-Step Reasoning in Large Language Models

    [https://arxiv.org/abs/2605.24140](https://arxiv.org/abs/2605.24140)

    提出HyperGuide方法，利用双曲空间的几何特性将推理进展编码为引导信号，使大语言模型在单次生成的效率与树搜索的准确性之间实现高效平衡。

    

    多步推理仍然是大语言模型面临的核心挑战：单次生成效率高但准确性不足；树搜索方法虽能探索多条路径但计算开销大。我们通过将推理进展提炼为一种双曲几何信号来引导逐步生成，从而弥合这一差距。我们的方法源于一个结构性观察：在组合推理树中，含有解的状态很少，而死路（无解分支）数量呈指数级增长。双曲空间恰好匹配这种不对称性——靠近原点处体积紧凑，朝向边界处容量呈指数级扩展——因此到原点的距离可自然地编码解的接近程度，而角向分离则可区分需要不同后续操作的分支。我们训练一个轻量级投影头将大语言模型的隐状态投影到该空间，然后在其自身的推理尝试上进行交互式的低秩适配器微调，以作用于……（原文摘要在此处截断）

    arXiv:2605.24140v4 Announce Type: replace  Abstract: Multi-step reasoning remains a central challenge for large language models: single-pass generation is efficient but lacks accuracy; tree-search methods explore multiple paths but are computation-heavy. We address this gap by distilling reasoning progress into a hyperbolic geometric signal that guides step-by-step generation. Our approach is motivated by a structural observation: in combinatorial reasoning trees, solution-bearing states are few while dead ends are exponentially numerous. The hyperbolic space matches this asymmetry, with compact volume near the origin and exponentially expanding capacity toward the boundary, so that distance-to-origin naturally encodes solution proximity while angular separation distinguishes branches requiring different next operations. We train a lightweight head to project LLM hidden states into this space, then fine-tune a low-rank adapter interactively on its own reasoning attempts to act on the i
    
[^342]: EchoDistill：通过噪声到干净的自蒸馏实现鲁棒的大型音频语言模型

    EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation

    [https://arxiv.org/abs/2605.23954](https://arxiv.org/abs/2605.23954)

    EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。

    

    大型音频语言模型（LALMs）仍然容易受到声学噪声的影响，噪声会掩盖与任务相关的证据并产生不可靠的响应。我们提出了EchoDistill，这是一种噪声到干净的自蒸馏框架，在后训练阶段将干净音频作为特权信息加以利用。以噪声输入的学生模型采样能够反映其推理时行为的候选响应，而同一骨干网络的冻结副本则处理对应的干净音频。EchoDistill结合了掩码响应token蒸馏、任务门控一致性塑形以及教师参考的组相对优化，使噪声输入下的生成结果与干净条件下的语义保持对齐。推理时仅保留学生模型，不引入任何额外的推理开销。在三个LALM骨干网络和三个音频领域、信噪比为-10dB的条件下，EchoDistill相比最强基线将噪声输入下的平均准确率提升了1.63个百分点。在Qwen2.5-Omni上，它

    arXiv:2605.23954v2 Announce Type: replace-cross  Abstract: Large Audio Language Models (LALMs) remain vulnerable to acoustic noise, which can obscure task-relevant evidence and produce unreliable responses. We propose EchoDistill, a noisy-to-clean self-distillation framework that uses clean audio as privileged information during post-training. A noisy-input student samples candidate responses reflecting its inference-time behavior, while a frozen copy of the same backbone processes the corresponding clean audio. EchoDistill combines masked response-token distillation, task-gated consistency shaping, and teacher-referenced group-relative optimization to align noisy-input generation with clean-conditioned semantics. Only the student is retained at inference time, introducing no additional inference cost. Across three LALM backbones and three audio domains at -10dB, EchoDistill improves average noisy-input accuracy by 1.63 percentage points over the strongest baseline. On Qwen2.5-Omni, it
    
[^343]: FastKernels：在生产环境中对GPU内核生成进行基准测试

    FastKernels: Benchmarking GPU Kernel Generation in Production

    [https://arxiv.org/abs/2605.23215](https://arxiv.org/abs/2605.23215)

    FastKernels提出了一个包含384个任务的生产级GPU内核生成基准，通过组合层次结构覆盖94.6%的HuggingFace Transformers架构，并直接在生产执行路径上以框架官方发布的内核为基准对候选内核进行内核级和端到端评分。

    

    基于大语言模型（LLM）的GPU内核生成智能体正在迅速发展，但它们所优化的基准测试往往在隔离环境中评估内核，使用合成输入和薄弱的基线，从而奖励那些在真实推理系统中会失效或无法体现的沙盒加速效果。我们提出了FastKernels，这是一个包含384个任务的基准测试，这些任务取自8个类别中的47个代表性架构，其内核足以重新实现94.6%（472/499）的HuggingFace Transformers架构，且输出与原生实现相匹配。每个任务都镜像了相应生产模块的接口，并以生产框架实际发布的内核作为评分基准；任务构成了一个组合层次结构，从底层原语到完整模型，其中高层模块会导入低层模块。候选内核在内核层面以及其所属模型的内部进行端到端评分，并在生产执行路径上进行评估，MacroEval则聚合经过校准的（原文摘要在此处截断）……

    arXiv:2605.23215v2 Announce Type: replace-cross  Abstract: LLM-based agents for GPU kernel generation are advancing rapidly, but the benchmarks they optimize against evaluate kernels in isolation, with synthetic inputs and weak baselines, rewarding sandbox speedups that break or vanish in real inference systems. We introduce FastKernels, a benchmark of 384 tasks drawn from 47 representative architectures across 8 categories, whose kernels suffice to reimplement 94.6% (472/499) of HuggingFace Transformers architectures with outputs matching the native implementations. Each task mirrors the interface of the corresponding production module and is scored against the kernels production frameworks ship, and tasks form a compositional hierarchy, from primitives to full models, in which higher-level modules import lower-level ones. Candidates are scored at the kernel level and end to end inside the models they come from, on the production execution path, and MacroEval aggregates calibrated cor
    
[^344]: TEGER：面向概率交通预测的时空协方差模型

    Teger: Spatiotemporal Covariance for Probabilistic Traffic Forecasting

    [https://arxiv.org/abs/2605.18068](https://arxiv.org/abs/2605.18068)

    提出TEGER残差协方差模型，通过闭式更新在测试时动态校正交通预测的联合不确定性而无需重新训练，并可附加于冻结的时间序列基础模型。

    

    交通状况会发生漂移——需求模式、事件动态和传感器行为会随部署生命周期不断变化——因此在训练时一次性拟合并保持静态的联合不确定性估计，会随着条件变化而失准。我们提出TEGER，一种残差协方差模型，通过闭式更新（而非重新训练）在测试时保持预测器的联合预测不确定性始终最新。固定的传感器图提供一个低维空间精度因子，编码哪些传感器的误差会共同变动；在推理阶段，高斯条件化仅根据最近观测到的残差来修正下一次预测的均值和协方差，同时指数移动平均波动率项重新缩放边际不确定性以跟踪局部漂移，并保留学习到的相关结构。这两种更新均不触及预测主干网络的权重，因此同一机制可以附加到冻结的时间序列基础模型上。

    arXiv:2605.18068v2 Announce Type: replace-cross  Abstract: Traffic conditions drift -- demand patterns, incident dynamics, and sensor behavior shift over a deployment's lifetime -- so a joint uncertainty estimate fit once at training time and left static will miscalibrate as conditions change. We present TEGER, a residual covariance model that keeps a forecaster's joint predictive uncertainty current at test time through closed-form updates, not retraining. A fixed sensor graph supplies a low-dimensional spatial precision factor encoding which sensors' errors move together; at inference, Gaussian conditioning corrects The next forecast's mean and covariance from only the most recently observed residuals, and an exponential moving-average volatility term rescales marginal uncertainty to track local drift while preserving the learned correlation structure. Neither update touches the forecasting backbone's weights, so the same mechanism attaches to a frozen time-series foundation model: n
    
[^345]: HINT-SD：面向长时程智能体的定向后见自蒸馏

    HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2605.17873](https://arxiv.org/abs/2605.17873)

    HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。

    

    arXiv:2605.17873v2 公告类型：交叉替换 摘要：使用强化学习训练长时程LLM智能体具有挑战性，因为稀疏的结果奖励能揭示任务是否成功，但无法指出哪些中间动作导致了该结果，或应如何纠正这些动作。近期方法通过从回合级动作-输出信号生成奖励或文本提示，或使用反馈条件自蒸馏来缓解此问题。然而，在每回合生成反馈效率低下，因为许多中间回合可能已经成功或中性，而在固定或错位的回合应用反馈往往无法监督导致失败的动作。为弥合这一差距，我们提出HINT-SD，一种定向自蒸馏框架，利用完整轨迹的后见之明选择与失败相关的动作，并仅对定向动作片段应用反馈条件蒸馏。在BFCL v3和AppWorld上的实验表明，我们的方法优于密集反馈方法。

    arXiv:2605.17873v2 Announce Type: replace-cross  Abstract: Training long-horizon LLM agents with reinforcement learning is challenging because sparse outcome rewards reveal whether a task succeeds, but not which intermediate actions caused the outcome or how they should be corrected. Recent methods alleviate this issue by generating rewards or textual hints from turn-level action-output signals, or by using feedback-conditioned self-distillation. However, generating feedback at every turn is inefficient when many intermediate turns are already successful or neutral, and applying feedback at a fixed or misaligned turn often fails to supervise the actions that contributed to the failure. To bridge this gap, we propose HINT-SD, a targeted self-distillation framework that uses full-trajectory hindsight to select failure-relevant actions and applies feedback-conditioned distillation only to targeted action spans. Experiments on BFCL v3 and AppWorld show that our method outperforms the dense
    
[^346]: LEAF：一个面向事件增强预测的动态基准

    LEAF: A Living Benchmark for Event-Augmented Forecasting

    [https://arxiv.org/abs/2605.16358](https://arxiv.org/abs/2605.16358)

    LEAF是首个面向事件增强预测任务的动态基准，通过递归检索智能体系统与双智能体交叉验证收集时间对齐的辅助上下文，将未来信息泄露从8.6%降至1.6%。

    

    大型语言模型（LLM）正越来越多地被应用于现实世界的预测任务，然而其真实预测能力的评估仍受到预训练数据污染以及自动化检索中前瞻性信息泄露的损害。现有基准要么依赖静态上下文，要么将评估限制在狭窄的环境中，要么未能对辅助文本事件进行未来信息泄露的审计。为了建立严格的评估范式，我们提出了LEAF——首个面向事件增强预测任务（包括趋势预测、事件预测和时间序列预测）的动态基准。LEAF将递归检索智能体系统与双智能体交叉验证相结合，以收集全面、相关且时间对齐的辅助上下文。由47位领域专家对500个任务开展的全面审计表明，我们的流程将未来信息泄露从8.6%降低至1.6%。在对16个前沿模型的广泛评估中……

    arXiv:2605.16358v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly applied to real-world forecasting tasks, yet evaluating their true predictive capability remains compromised by pre-training data contamination and look-ahead leakage in automated search. Existing benchmarks either rely on static contexts, restrict evaluations to narrow environments, or fail to audit auxiliary textual events for future information leakage. To establish a rigorous evaluation paradigm, we propose LEAF, the first living benchmark for event-augmented forecasting tasks, including trend, event, and time series forecasting. LEAF couples a recursive retrieval agent system with dual-agent cross-validation to gather comprehensive, relevant, and temporally aligned auxiliary context. A comprehensive audit across 500 tasks by 47 domain specialists demonstrates that our pipeline suppresses future information leakage from 8.6% to 1.6%. Across extensive evaluations of 16 frontier p
    
[^347]: KGPFN：通过上下文学习释放知识图谱基础模型的潜力

    KGPFN: Unlocking the Potential of Knowledge Graph Foundation Model via In-Context Learning

    [https://arxiv.org/abs/2605.14907](https://arxiv.org/abs/2605.14907)

    KGPFN是一种基于先验数据拟合网络的知识图谱基础模型，通过将可迁移的关系表示与推理时对结构化局部邻域和全局上下文的上下文学习相结合，弥补了知识图谱推理中上下文学习的空白。

    

    知识图谱基础模型旨在通过学习可迁移的关系结构，泛化到包含未见实体和关系的图上。然而，现有的大多数方法仅关注关系层面的通用性，而基础模型的另一大支柱——上下文学习——在知识图谱推理中在很大程度上仍未被探索。知识图谱中的上下文是结构化且异构的：准确的预测既需要以查询实体的局部邻域为条件，也需要以能够总结查询关系在众多实例中如何表现的全局上下文为条件。我们提出了KGPFN，这是一个建立在先验数据拟合网络（PFN）之上的知识图谱基础模型，它将可迁移的关系表示与推理时对结构化上下文的上下文学习相结合。KGPFN通过在关系图上进行消息传递来学习关系表示，并从多层NBFNet的中间头表示中提取多尺度的局部上下文。它……

    arXiv:2605.14907v2 Announce Type: replace  Abstract: Knowledge graph (KG) foundation models aim to generalize to graphs with unseen entities and relations by learning transferable relational structure. Most existing methods, however, focus on relation-level universality, leaving in-context learning, the other pillar of foundation models, largely unexplored for KG reasoning. Context in KGs is structured and heterogeneous: accurate prediction requires conditioning both on the local neighborhood of the query entities and on global context that summarizes how the query relation behaves across many instances. We propose KGPFN, a KG foundation model built on a Prior-Data Fitted Network (PFN) that combines transferable relational representations with inference-time in-context learning over structured context. KGPFN learns relation representations by message passing on relation graphs and extracts multi-scale local context from the intermediate head representations of a multi-layer NBFNet. It 
    
[^348]: GPart：通过全局参数划分实现端到端等距微调

    GPart: End-to-End Isometric Fine-Tuning via Global Parameter Partitioning

    [https://arxiv.org/abs/2605.14841](https://arxiv.org/abs/2605.14841)

    GPart 通过稀疏等距划分矩阵将可训练向量直接映射到完整权重空间，去除了 LoRA 式低秩重构，实现了端到端等距且高度参数高效的微调。

    

    arXiv:2605.14841v2 公告类型： replace-cross 摘要：低秩适配已成为大规模深度学习模型参数高效微调（PEFT）的主流范式。然而，其双线性参数化引入了依赖于参数的几何结构：从可训练参数到权重更新的映射通常不保持距离。将低维向量投影到 LoRA 参数空间的相关方法（如 Uni-LoRA）提升了参数效率，但随后的双线性映射破坏了端到端的等距性。我们提出 GPart（全局划分微调），这是一种高度参数高效的微调方法，通过稀疏的等距划分矩阵将一个 $d$ 维可训练向量直接映射到完整权重空间中。GPart 在保持固定的全局参数共享先验的同时，去除了基于 LoRA 的方法所使用的额外低秩重构。这带来了一个仅含单一主要超参数（$d$）的简单参数化……

    arXiv:2605.14841v2 Announce Type: replace-cross  Abstract: Low-rank adaptation (LoRA) has become a dominant paradigm for parameter-efficient fine-tuning (PEFT) of large-scale deep learning models. However, its bilinear parameterization induces a parameter-dependent geometry: the mapping from trainable parameters to weight updates is not generally distance-preserving. Related methods that project a low-dimensional vector into LoRA's parameter space, such as Uni-LoRA, improve parameter efficiency, but the subsequent bilinear map breaks end-to-end isometry. We propose GPart (Global Partition fine-tuning), a highly parameter-efficient fine-tuning method that maps a $d$-dimensional trainable vector directly into the full weight space through a sparse, isometric partition matrix. GPart retains a fixed global parameter-sharing prior while removing the additional low-rank reconstruction used by LoRA-based methods. This yields a simple parameterization with a single main hyperparameter ($d$), e
    
[^349]: ASH：在长时程世界中自我磨砺的智能体

    ASH: Agents that Self-Hone in Long-Horizon Worlds

    [https://arxiv.org/abs/2605.14211](https://arxiv.org/abs/2605.14211)

    ASH是一个无需奖励工程或专家标注的智能体系统，通过自我改进循环从自身轨迹学习逆动力学模型，进而从无标注网络视频中提取监督信号并保留关键时刻作为长期记忆，从而在《宝可梦：绿宝石》和《塞尔达传说：缩小帽》等需要数小时规划的长时程任务中实现自我提升。

    

    长时程视觉运动任务仍然是人工智能领域的一项根本性挑战，因为现有方法依赖于人工设计的奖励或带有动作标注的演示数据，而这两种方式都无法规模化。我们提出了ASH，这是一个能够从无标注、含噪声的互联网视频中学习长时程策略的智能体系统，无需奖励塑形或专家标注。ASH遵循一个自我改进循环：当它陷入困境时，ASH会从自身轨迹中学习一个逆动力学模型（IDM），并利用该IDM从相关的互联网视频中提取监督信号。ASH使用无监督学习从大规模互联网视频中识别关键时刻，并将其保留为长期记忆，从而能够应对长时程问题。我们在两个互补的、需要数小时规划的环境中评估了ASH：《宝可梦：绿宝石》（一款回合制角色扮演游戏）和《塞尔达传说：缩小帽》（一款实时动作冒险游戏）。在这两款游戏中，行为克隆、检索增强（后续内容被截断）

    arXiv:2605.14211v4 Announce Type: replace  Abstract: Long-horizon visuomotor tasks remain a fundamental challenge in AI, as current methods rely on hand-engineered rewards or action-labeled demonstrations, neither of which scales. We introduce ASH, an agentic system that learns a long-horizon policy from unlabeled, noisy internet video, without reward shaping or expert annotation. ASH follows a self-improvement loop; when it gets stuck, ASH learns an Inverse Dynamics Model (IDM) from its own trajectories, and uses its IDM to extract supervision from relevant internet video. ASH uses unsupervised learning to identify key moments from large-scale internet video and retains them as long-term memory - allowing it to tackle long-horizon problems. We evaluate ASH on two complementary environments demanding multi-hour planning: Pokemon Emerald, a turn-based RPG, and The Legend of Zelda: The Minish Cap, a real-time action-adventure game. In both games, behavioral cloning, retrieval-augmented a
    
[^350]: 测量谷歌AI概览：激活率、来源质量、论断忠实度与出版商影响

    Measuring Google AI Overviews: Activation, Source Quality, Claim Fidelity, and Publisher Impact

    [https://arxiv.org/abs/2605.14021](https://arxiv.org/abs/2605.14021)

    该论文通过40天内跨19个主题发出的55,393个查询的大规模纵向测量，首次系统刻画了谷歌AI概览的激活率、引用来源质量与出版商影响，发现其总体激活率为13.7%（疑问式查询达64.7%），且其来源选择机制与传统搜索排序明显不同。

    

    谷歌AI概览（AI Overviews，AIOs）可以说是生成式AI最广泛接触的部署形式，覆盖超过20亿用户，而这些用户可能并未意识到他们看到的答案是由AI生成的。传统搜索引擎只是呈现排序后的来源、让用户自行评估，而AIOs则综合并给出单一答案——这使谷歌对用户阅读和了解的内容拥有了前所未有的编辑控制权。我们开展了一项大规模纵向测量研究，在40天的时间窗口（2026年3月13日至4月21日）内，跨19个主题类别发出了55,393个热门查询。我们报告了四项主要发现。第一，AIO的总体激活率为13.7%，对疑问句式查询则升至64.7%，而政治敏感话题的激活率明显更低。第二，AIO引用的域名比同页展示的第一页搜索结果更具可信度，但其中近30%的域名根本没有出现在这些搜索结果中，这表明存在一种与传统谷歌搜索不同的来源选择机制。

    arXiv:2605.14021v2 Announce Type: replace-cross  Abstract: Google AI Overviews (AIOs) are arguably the most widely encountered deployment of generative AI, reaching over 2 billion users who may not realize the answers they see are AI-generated. Where search engines have traditionally surfaced ranked sources and left users to evaluate them, AIOs synthesize and deliver a single answer - giving Google unprecedented editorial control over what users read and know. We present a large-scale longitudinal measurement study, issuing 55,393 trending queries across 19 topical categories over a 40-day window (March 13 - April 21, 2026). We report four main findings. First, overall AIO activation is 13.7%, rising to 64.7% for question-form queries, while politically sensitive topics see markedly lower rates. Second, AIO-cited domains are more credible than co-displayed first-page results, yet nearly 30% do not appear in those results at all, indicating a source selection mechanism distinct from Goo
    
[^351]: ReForge：基于锚点正则化回归的合并模型精炼方法

    ReForge: Refining Merged Models with Anchor-Regularized Regression

    [https://arxiv.org/abs/2605.12843](https://arxiv.org/abs/2605.12843)

    提出双层优化框架ReForge，将强合并模型作为锚点先验，通过贝叶斯线性回归对模块级进行精炼，并利用贝叶斯优化联合选择正则化强度与组装尺度，同时提供无需校准数据的任务向量Gram变体。

    

    模型合并旨在将多个任务特定的专家模型组合成单一模型而无需联合重新训练，在数据访问或计算预算受限的情况下，为多任务学习提供了一种实用的替代方案。现有的模型合并方法很少利用强大的合并模型作为先验来进行进一步改进。为解决这一局限性，我们提出了ReForge，这是一个双层优化框架，将模块级精炼表述为具有锚点中心先验的贝叶斯线性回归。内层从无标签的校准激活中产生闭式MAP估计；外层利用贝叶斯优化，基于留出验证数据联合选择异构的正则化强度和组装尺度。此外，我们开发了ReForge的无数据变体，用任务向量Gram矩阵替代激活统计量，从而消除了对校准样本的需求。在广泛的基准测试中……

    arXiv:2605.12843v2 Announce Type: replace-cross  Abstract: Model merging aims to combine multiple task-specific expert models into a single model without joint retraining, offering a practical alternative to multi-task learning when data access or computational budget is limited. Existing model merging methods rarely exploit strong merged models as priors for further improvement. To address this limitation, we propose ReForge, a bilevel optimization framework that formulates module-wise refinement as Bayesian linear regression with an anchor-centered prior. The inner level yields a closed-form MAP estimate from unlabeled calibration activations. The outer level uses Bayesian optimization to jointly select heterogeneous regularization strengths and assembly scales using held-out validation data. Furthermore, we develop a data-free variant of ReForge that replaces activation statistics with task-vector Grams, eliminating the need for calibration examples. Across extensive benchmarks, inc
    
[^352]: NARA：面向异构矢量地理实体的锚点条件表示学习

    NARA: Anchor-Conditioned Representation Learning for Heterogeneous Vector Geoentities

    [https://arxiv.org/abs/2605.12276](https://arxiv.org/abs/2605.12276)

    NARA提出了一种自监督表示学习框架，通过融合几何距离与拓扑关系的空间上下文感知注意力机制，统一建模点、线、面等异构矢量地理实体，从而学习更全面的地理实体表示。

    

    矢量地理空间数据将世界表示为离散的地理实体，例如道路、建筑物和兴趣点，每个实体都具有语义属性、几何形状以及与其他地理实体的空间关系，包括度量邻近性和拓扑关系。现有的地理实体表示学习方法通常仅支持单一几何类型，或仅对这些关系中的一部分进行建模，这限制了它们捕获异构地理实体间空间上下文以及支持多样化下游任务的能力。我们提出了NARA（Neural Anchor-conditioned Relation-Aware representation learning，神经锚点条件关系感知表示学习），这是一个面向异构矢量地理实体的新型自监督表示学习框架。NARA通过空间上下文感知的注意力机制对地理实体进行上下文化建模，该机制利用几何距离并受周围点、折线和多边形之间拓扑关系的调节来建模空间自相关性。NARA还引入了掩码地理实体语义建模（摘要内容在此处被截断）。

    arXiv:2605.12276v2 Announce Type: replace  Abstract: Vector geospatial data represent the world as discrete geoentities, such as roads, buildings, and points of interest, each with semantic attributes, geometry, and spatial relations to other geoentities, including metric proximity and topology. Existing methods for learning geoentity representations typically support a single geometry type or model only a subset of these relations, limiting their ability to capture spatial context across heterogeneous geoentities and support diverse downstream tasks. We propose NARA (Neural Anchor-conditioned Relation-Aware representation learning), a novel self-supervised representation framework for heterogeneous vector geoentities. NARA contextualizes geoentities through spatial-context-aware attention that models spatial autocorrelation using geometry distance modulated by topological relations across surrounding points, polylines, and polygons. NARA introduces masked geoentity semantic modeling a
    
[^353]: DuetMoE：耦合组间与组内鲁棒性以实现公平的医学图像分析

    DuetMoE: Coupling Inter- and Intra-Subgroup Robustness for Fair Medical Image Analysis

    [https://arxiv.org/abs/2605.10521](https://arxiv.org/abs/2605.10521)

    提出DuetMoE框架，通过亚群体感知的专家混合机制将组间公平性与组内鲁棒性相结合，为个体患者提供更公平可靠的医学图像分析。

    

    随着医疗AI在全球多样化的医疗环境中不断扩展，跨患者群体的公平性能对于可信赖的临床应用变得至关重要。医学图像分析中的公平性通常通过预定义亚群体的平均性能来评估，然而相似的亚群体平均值可能掩盖个体患者之间的显著差异。因此，可靠的医疗AI需要解决两个互补的目标：组间公平性，即减少不同群体之间的性能差异；以及组内鲁棒性，即保护每个群体内服务不足的患者。为了共同解决这些目标，我们提出了DuetMoE，一个亚群体感知的专家混合框架，它将群体级适应与患者特定的临床指导相结合，为个体患者实现更可靠的医学图像分析。对于没有关联临床记录的场景，我们...

    arXiv:2605.10521v2 Announce Type: replace-cross  Abstract: As medical AI expands across diverse healthcare settings worldwide, equitable performance across patient populations is becoming essential to trustworthy clinical use. Fairness in medical image analysis is often evaluated through average performance across predefined subgroups, yet similar subgroup averages can conceal substantial variation among individual patients. Therefore, a reliable medical AI requires addressing two complementary objectives: \emph{inter-subgroup fairness}, which reduces performance disparities across groups, and \emph{intra-subgroup robustness}, which protects poorly served patients within each group. To jointly address these objectives, we propose \textbf{DuetMoE}, a subgroup-aware mixture-of-experts framework that couples group-level adaptation with patient-specific clinical guidance, enabling more reliable medical image analysis for individual patients. For settings without linked clinical records, we
    
[^354]: MolWorld：面向可操作分子优化的分子世界模型

    MolWorld: Molecule World Models for Actionable Molecular Optimization

    [https://arxiv.org/abs/2605.08954](https://arxiv.org/abs/2605.08954)

    提出分子世界模型 MolWorld，将可操作分子优化形式化为分子转移图的迭代扩展，通过匹配分子对（MMP）边显式建模可达性，确保优化得到的候选分子可从已知分子经局部结构修饰到达。

    

    药物发现中的分子优化旨在发现具有更优目标性质的分子，但实际的先导化合物优化往往需要的不只是高预测分数。一个有用的候选分子还应当是“可操作的”：即它应当能够通过一系列局部结构修饰从已知分子出发到达，从而为在不断演化的化学系列中解释性质变化提供明确的结构参照。现有的从头设计与单分子优化方法并未显式地对这种可达性进行建模，尤其是当目标分子以及将其与已知化合物相连的中间分子均为未知时。在本工作中，我们将可操作分子优化形式化为分子转移图的迭代扩展，其中节点表示分子，边编码表示局部结构差异的匹配分子对关系。我们提出了 MolWorld，一种分子世界模型……

    arXiv:2605.08954v2 Announce Type: replace-cross  Abstract: Molecular optimization in drug discovery aims to discover molecules with improved target properties, but practical lead optimization often requires more than high predicted scores. A useful candidate should also be actionable: it should be reachable from known molecules through a sequence of local structural modifications, providing explicit structural references for interpreting property changes within an evolving chemical series. Existing de novo and single-molecule optimization methods do not explicitly model such reachability, especially when both the target molecules and the intermediate molecules connecting them to known compounds are unknown. In this work, we formulate actionable molecular optimization as iterative expansion of a molecule-transfer graph, where nodes are molecules and edges encode matched molecular pair (MMP) relations representing localized structural differences. We propose MolWorld, a molecule world mo
    
[^355]: 基于后验采样的离线策略优化

    Offline Policy Optimization with Posterior Sampling

    [https://arxiv.org/abs/2605.07393](https://arxiv.org/abs/2605.07393)

    该论文提出PSPO方法，通过将动力学模型建模为随机变量（后验采样）而非点估计，实现离线强化学习中对分布外区域的受控探索，从而在泛化能力与鲁棒性之间取得平衡。

    

    基于模型的离线强化学习（RL）中的一个根本性挑战在于泛化能力与对分布外（OOD）区域利用误差的鲁棒性之间的权衡。解决这一权衡的关键在于使模型能够探索与底层物理动力学保持一致的OOD区域。然而，实现这一点具有挑战性，因为有限的数据无法唯一地识别动力学模型，且不受约束的探索是有风险的。现有方法往往忽视了这一细微差别，通过过度的悲观正则化来应对风险，这虽然保证了鲁棒性，却牺牲了泛化能力。为了解决这一问题，我们提出了PSPO，它将动力学模型视为随机变量而非点估计。这种建模方式天然地允许对OOD区域进行受控探索。通过交替更新后验分布和策略，我们设计了一种正则化优化……

    arXiv:2605.07393v2 Announce Type: replace  Abstract: A fundamental challenge in model-based offline reinforcement learning (RL) lies in the trade-off between generalization and robustness against exploitation errors in out-of-distribution (OOD) regions. The key to resolving this trade-off lies in enabling the model to explore OOD regions that remain consistent with underlying physical dynamics. However, achieving this is challenging because limited data cannot uniquely identify the dynamics model, and unconstrained exploration is risky. Existing methods often overlook this nuance, addressing the risk through excessive pessimistic regularization, which ensures robustness but sacrifices generalization. To address this, we propose PSPO, which treats the dynamics model as a random variable rather than a point estimate. This formulation inherently allows for controlled exploration of OOD regions. By alternately updating the posterior distribution and the policy, we design a regularized opti
    
[^356]: LensVLM：针对文本压缩视觉表示的选择性上下文扩展

    LensVLM: Selective Context Expansion for Compressed Visual Representation of Text

    [https://arxiv.org/abs/2605.07019](https://arxiv.org/abs/2605.07019)

    LensVLM提出了一种推理框架和后训练方案，使VLM能先扫描压缩的渲染文本图像，再通过学习到的工具选择性地将相关图像扩展为未压缩形式，从而在4.3倍有效压缩下保持与全文处理相当的准确率，并在最高10.1倍压缩下超越现有基线。

    

    视觉语言模型（VLM）提供了一种令人兴奋的可能性，即将文本作为渲染图像进行处理，从而避免了将文本标记化为长token序列的需要。由于VLM的图像编码器将固定大小的图像映射为固定数量的视觉token，改变渲染分辨率便提供了一个细粒度的压缩旋钮。然而，随着压缩程度的增加，准确率会迅速下降：字符缩小到低于视觉编码器的有效分辨率，导致其无法被辨别。为解决这一问题，我们提出了LensVLM，这是一个推理框架与后训练方案，使VLM能够先扫描压缩图像，然后通过学习到的工具选择性地仅将相关图像扩展至未压缩形式。基于Qwen3.5-9B-Base构建的LensVLM在4.3倍有效压缩下仍保持与全文上限相当的准确率，并在最高10.1倍有效压缩下优于基于检索的、文本压缩和视觉压缩等基线方法。

    arXiv:2605.07019v2 Announce Type: replace-cross  Abstract: Vision Language Models (VLMs) offer the exciting possibility of processing text as rendered images, bypassing the need for tokenizing the text into long token sequences. Since VLM image encoders map fixed-size images to a fixed number of visual tokens, varying rendering resolution provides a fine-grained compression knob. However, accuracy deteriorates quickly as compression increases: characters shrink below the vision encoder's effective resolution, making them indistinguishable. To address this, we propose LensVLM, an inference framework and post-training recipe that enables VLMs to scan compressed images, then selectively expand only the relevant images to their uncompressed form via learned tools. Building on Qwen3.5-9B-Base, LensVLM maintains accuracy comparable to the full-text upper bound at 4.3$\times$ effective compression and outperforms retrieval-based, text- and visual-compression baselines up to 10.1$\times$ effec
    
[^357]: 递归智能体优化

    Recursive Agent Optimization

    [https://arxiv.org/abs/2605.06639](https://arxiv.org/abs/2605.06639)

    RAO提出了一种强化学习方法，通过训练智能体递归地生成并委派子任务给自身的新实例来实现推理时的分治扩展，使模型能够突破上下文窗口限制、泛化到远难于训练任务的问题，并降低实际运行时间。

    

    我们提出了递归智能体优化（Recursive Agent Optimization, RAO），这是一种用于训练递归智能体的强化学习方法：递归智能体能够递归地生成并将子任务委派给自身的新实例。递归智能体实现了一种推理时扩展算法，通过分治法使智能体能够自然地扩展到更长的上下文，并泛化到更困难的问题。RAO提供了一种训练模型以充分利用这种递归推理的方法，教会智能体何时以及如何进行委派和沟通。我们发现，以这种方式训练的递归智能体具有更好的训练效率，能够扩展到超出模型上下文窗口的任务，泛化到比训练任务难得多的问题，并且与单智能体系统相比可以减少实际运行时间。

    arXiv:2605.06639v2 Announce Type: replace-cross  Abstract: We introduce Recursive Agent Optimization (RAO), a reinforcement learning approach for training recursive agents: agents that can spawn and delegate sub-tasks to new instantiations of themselves recursively. Recursive agents implement an inference-time scaling algorithm that naturally allows agents to scale to longer contexts and generalize to more difficult problems via divide-and-conquer. RAO provides a method to train models to best take advantage of such recursive inference, teaching agents when and how to delegate and communicate. We find that recursive agents trained in this way enjoy better training efficiency, can scale to tasks that go beyond the model's context window, generalize to tasks much harder than the ones the agent was trained on, and can enjoy reduced wall-clock time compared to single-agent systems.
    
[^358]: CoMemNet：一种带漂移感知采样的持续记忆网络用于交通预测

    CoMemNet: A Continual Memory Network with Drift-Aware Sampling for Traffic Prediction

    [https://arxiv.org/abs/2605.05738](https://arxiv.org/abs/2605.05738)

    提出 CoMemNet，一种通过在线/目标双分支、基于 Wasserstein 的漂移感知采样和节点自适应时间记忆重放缓冲，在演进的交通传感器网络上实现无需固定邻接矩阵与全量重训的高效持续交通预测模型。

    

    交通传感器网络会随着传感器的增加和交通分布的变化而不断演进，而大多数预测模型假设节点集是固定的，并在所有可用数据上反复重新训练。我们提出了 CoMemNet，一种面向不断演进的交通传感器网络的高效预测持续记忆网络。CoMemNet 使用一个在线分支来适应当前时期，并使用一个指数移动平均的目标分支作为稳定的特征参考。基于 Wasserstein 距离的漂移采样器比较节点级的在线-目标特征分布，并选择有限的一组漂移敏感节点进行更新。轻量级的节点自适应时间记忆重放缓冲区 保留紧凑的时间状态，无需反复遍历所有历史训练数据。预测主干网络不消耗邻接矩阵；传感器邻接关系仅用于构造数据，并可选择地将所选更新集扩展到有限的邻域。实验……

    arXiv:2605.05738v2 Announce Type: replace-cross  Abstract: Traffic sensor networks evolve as sensors are added and traffic distributions change, whereas most forecasting models assume a fixed node set and repeatedly retrain on all available data. We propose CoMemNet, a Continual Memory Network for efficient prediction over evolving traffic sensor networks. CoMemNet uses an Online branch to adapt to the current period and an exponential-moving-average Target branch as a stable feature reference. A Wasserstein-based Drift Sampler compares node-wise Online-Target feature distributions and selects a limited set of drift-sensitive nodes for updating. A lightweight Node-Adaptive Temporal Memory Replay Buffer (TMRB-N) retains compact temporal states without repeatedly traversing all historical training data. The prediction backbone does not consume an adjacency matrix; sensor adjacency is used only to construct data and optionally expand the selected update set to a limited neighborhood. Expe
    
[^359]: 输入凸神经网络的双认证白盒推断

    Dual Certified White-Box Inference for Input Convex Neural Networks

    [https://arxiv.org/abs/2605.04722](https://arxiv.org/abs/2605.04722)

    该论文提出利用SOC-ICNN与参数化二阶锥规划价值函数的精确对偶表示，开发双认证白盒推断方法DCI，从最优对偶乘子恢复完整次微分、提供精确平稳性认证与下降方向，并实现牛顿加速与全局收敛。

    

    输入凸神经网络（ICNNs）用于学习凸目标函数，其极小值点定义了决策，因此高效且可靠的优化是推断的核心。在非光滑输入处，自动微分仅返回单个导数，而非决定最优性与下降方向的完整次微分。二阶锥输入凸神经网络（SOC-ICNNs）可以被精确表示为参数化二阶锥规划的价值函数，这提供了一种白盒方法，能够从最优对偶乘子中恢复其完整次微分，并在光滑区域上推导出显式的Hessian矩阵。基于这一表示，我们开发了双认证推断（DCI），该方法结合网络与可行集的几何结构，获得精确的平稳性认证以及切向公共下降方向。DCI利用局部曲率实现牛顿加速，并采用精确的近端保护机制。我们建立了全局收敛性，并在标（准假设下）……

    arXiv:2605.04722v2 Announce Type: replace-cross  Abstract: Input convex neural networks (ICNNs) are used to learn convex objectives whose minimizers define decisions, making efficient and reliable optimization central to inference. At nonsmooth inputs, automatic differentiation returns a single derivative rather than the full subdifferential governing optimality and descent. Second-order cone ICNNs (SOC-ICNNs) admit an exact representation as value functions of parametric second-order cone programs, providing a white-box approach to recovering their full subdifferentials from optimal dual multipliers and deriving explicit Hessians on smooth regions. Building on this representation, we develop dual-certified inference (DCI), which combines the network and feasible set geometries to obtain exact stationarity certificates and tangent common descent directions. DCI uses local curvature for Newton acceleration and an exact proximal safeguard. We establish global convergence and, under stand
    
[^360]: 谁来守护基准测试？LLM智能体基准测试的自动化审计

    Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks

    [https://arxiv.org/abs/2604.24955](https://arxiv.org/abs/2604.24955)

    提出BenchGuard——首个利用前沿大语言模型对基于执行的LLM智能体基准测试进行跨工件联合审计的框架，能够自动发现基准测试本身存在的缺陷（如损坏的任务规范和僵化的评估脚本）。

    

    随着基准测试日益复杂，许多表面上的智能体失败实际上根本不是智能体本身的失败——而是基准测试本身的失败：损坏的任务规范、隐含的假设，以及惩罚有效替代方法的僵化评估脚本。我们提出将前沿大语言模型（LLM）用作评估基础设施的系统性审计员，并通过BenchGuard实现这一愿景——这是首个专为基于执行的智能体基准测试的跨工件联合审计而设计的框架。BenchGuard通过结构化LLM协议对所有基准测试工件进行交叉验证，并可选择将智能体的解决方案或执行轨迹作为额外的诊断证据纳入审计。在两个知名科学基准测试上的部署结果表明，BenchGuard在ScienceAgentBench中发现了12个经作者确认的问题——包括导致任务无法解决的致命错误——并且在BIXBench Verified-50子集上与专家识别问题的匹配率恰好达到83.3%。

    arXiv:2604.24955v2 Announce Type: replace-cross  Abstract: As benchmarks grow in complexity, many apparent agent failures are not failures of the agent at all---they are failures of the benchmark itself: broken specifications, implicit assumptions, and rigid evaluation scripts that penalize valid alternative approaches. We propose employing frontier LLMs as systematic auditors of evaluation infrastructure, and realize this vision through BenchGuard, the first framework explicitly designed for joint cross-artifact auditing of execution-based agent benchmarks. BenchGuard cross-verifies all benchmark artifacts via structured LLM protocols, optionally incorporating agent solutions or execution traces as additional diagnostic evidence. Deployed on two prominent scientific benchmarks, BenchGuard identified 12 author-confirmed issues in ScienceAgentBench---including fatal errors rendering tasks unsolvable---and exactly matched 83.3% of expert-identified issues on the BIXBench Verified-50 subs
    
[^361]: AI智能体如何花你的钱？分析与预测智能体编程任务中的Token消耗

    How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks

    [https://arxiv.org/abs/2604.22750](https://arxiv.org/abs/2604.22750)

    本文首次系统研究了智能体编程任务中的token消耗模式，发现智能体任务消耗的token比代码推理和对话任务高出1000倍且以输入token为主要成本来源、使用量波动极大，并进一步评估了大模型在任务执行前预测自身token成本的能力。

    

    AI智能体在复杂人类工作流程中的广泛部署正在推动LLM token消耗的快速增长。当智能体被部署在需要大量token的任务上时，自然会引出三个问题：（1）AI智能体把token花在了哪里？（2）哪些模型更具token效率？（3）智能体能否在任务执行前预测自己的token用量？本文首次对智能体编程任务中的token消耗模式进行了系统性研究。我们分析了八个前沿LLM在SWE-bench Verified上的运行轨迹，并评估了各模型在任务执行前预测自身token成本的能力。我们发现：（1）智能体任务的token消耗格外昂贵，比代码推理和代码对话任务高出1000倍，且整体成本主要由输入token而非输出token驱动；（2）token使用量高度可变且本质上具有随机性：同一任务的多次运行在总token消耗上可相差高达30倍。

    arXiv:2604.22750v3 Announce Type: replace-cross  Abstract: The wide adoption of AI agents in complex human workflows is driving rapid growth in LLM token consumption. When agents are deployed on tasks that require a significant amount of tokens, three questions naturally arise: (1) Where do AI agents spend the tokens? (2) Which models are more token-efficient? and (3) Can agents predict their token usage before task execution? In this paper, we present the first systematic study of token consumption patterns in agentic coding tasks. We analyze trajectories from eight frontier LLMs on SWE-bench Verified and evaluate models' ability to predict their own token costs before task execution. We find that: (1) agentic tasks are uniquely expensive, consuming 1000x more tokens than code reasoning and code chat, with input tokens rather than output tokens driving the overall cost; (2) token usage is highly variable and inherently stochastic: runs on the same task can differ by up to 30x in total
    
[^362]: 大语言模型表示中的反问句：一项线性探针研究

    Rhetorical Questions in LLM Representations: A Linear Probing Study

    [https://arxiv.org/abs/2604.14128](https://arxiv.org/abs/2604.14128)

    该研究通过线性探针发现大语言模型在表示空间中能够早期且稳定地编码反问句信号，其跨数据集可迁移性虽然存在，但并不意味着模型内部存在统一的共享表示。

    

    反问句的提出并非为了获取信息，而是为了说服他人或表明立场。然而大型语言模型如何在内部表示这类问句仍不清楚。我们使用线性探针在两个具有不同话语语境的社交媒体数据集上分析了LLM表示中的反问句，发现反问信号在早期就已显现，且最后 token 表示能最稳定地捕获这一信号。反问句在数据集内部与寻求信息的问题线性可分，在跨数据集迁移场景下仍可被检测到，AUROC 约达到 0.7-0.8。然而，我们证明这种可迁移性并不简单地意味着存在共享表示。在不同数据集上训练的探针应用于同一目标语料库时会产生不同的排名，排名靠前的实例之间的重叠度往往低于 0.2。定性分析表明，这些分歧对应于不同的修辞现象……

    arXiv:2604.14128v3 Announce Type: replace-cross  Abstract: Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some pr
    
[^363]: 先验证再修复：基于智能体执行验证的可信跨语言代码分析

    Verify Before You Fix: Agentic Execution Grounding for Trustworthy Cross-Language Code Analysis

    [https://arxiv.org/abs/2604.10800](https://arxiv.org/abs/2604.10800)

    该论文的核心创新是提出一个由LLM驱动的跨语言漏洞生命周期框架，以“未经执行确认可利用性就不得修复”这一严格不变式为准则，将结构-语义混合检测、基于执行的智能体验证与感知验证的迭代修复三个阶段串联起来，并借助通用抽象语法树与 GraphSAGE、Qwen2.5-Coder 嵌入的混合融合实现 Java、Python、C++ 的跨语言泛化，从而保证代码分析与修复建立在可验证的证据之上。

    

    部署在智能体流水线中的学习型分类器面临一个根本性的可靠性问题：预测只是概率性推断而非经过验证的结论，若不基于可观察的证据就对其采取行动，会在下游各阶段引发不断累积放大的失败。软件漏洞分析使这一代价变得具体且可度量。我们通过一个统一的跨语言漏洞生命周期框架来解决该问题，该框架由三个大语言模型（LLM）驱动的推理阶段构成——结构-语义混合检测、基于执行验证的智能体验证，以及感知验证结果的迭代修复——并受一条严格不变式约束：在未通过执行手段确认可利用性之前，不采取任何修复行动。跨语言泛化能力通过通用抽象语法树实现，该树将 Java、Python 和 C++ 归一化为统一的结构化模式，并与 GraphSAGE 与 Qwen2.5-Coder-1.5B 嵌入表示的混合融合相结合……

    arXiv:2604.10800v2 Announce Type: replace-cross  Abstract: Learned classifiers deployed in agentic pipelines face a fundamental reliability problem: predictions are probabilistic inferences, not verified conclusions, and acting on them without grounding in observable evidence leads to compounding failures across downstream stages. Software vulnerability analysis makes this cost concrete and measurable. We address this through a unified cross-language vulnerability lifecycle framework built around three LLM-driven reasoning stages-hybrid structural-semantic detection, execution-grounded agentic validation, and validation-aware iterative repair-governed by a strict invariant: no repair action is taken without execution-based confirmation of exploitability. Cross-language generalization is achieved via a Universal Abstract Syntax Tree (uAST) normalizing Java, Python, and C++ into a shared structural schema, combined with a hybrid fusion of GraphSAGE and Qwen2.5-Coder-1.5B embeddings throu
    
[^364]: 你的logits知道些什么？

    What do your logits know?

    [https://arxiv.org/abs/2604.09885](https://arxiv.org/abs/2604.09885)

    该论文首次系统比较了视觉-语言模型在不同表示层次（残差流、tuned lens投影、top-k logits）上保留的信息，发现即使是最易访问的top logit值也能泄露图像查询中与任务无关的信息，其泄露量在某些情况下与完整残差流的直接投影相当，揭示了模型内部信息泄露的安全风险。

    

    arXiv:2604.09885v2 公告类型：替换  摘要：近期的研究表明，探测模型内部结构可以揭示大量从模型生成结果中无法察觉的信息。这带来了无意或恶意信息泄露的风险，即模型用户能够获取模型所有者认为无法访问的信息。以视觉-语言模型作为测试平台，我们首次系统性地比较了不同表示层次上所保留的信息——这些信息从残差流中编码的丰富信息出发，经过两个自然瓶颈被逐步压缩：其一是使用tuned lens获得的残差流低维投影，其二是最可能影响模型答案的最终top-k logits。我们证明，即使是由模型top logit值所定义的、最容易被访问的瓶颈，也能够泄露基于图像查询中存在的与任务无关的信息，在某些情况下所泄露的信息量甚至与完整残差流的直接投影相当。

    arXiv:2604.09885v2 Announce Type: replace  Abstract: Recent work has shown that probing model internals can reveal a wealth of information not apparent from the model generations. This poses a risk of unintentional or malicious information leakage, where model users are able to learn information that the model owner assumed was inaccessible. Using vision-language models as a testbed, we present the first systematic comparison of information retained at different representational levels as it is compressed from the rich information encoded in the residual stream through two natural bottlenecks: low-dimensional projections of the residual stream obtained using tuned lens, and the final top-k logits most likely to impact model's answer. We show that even easily accessible bottlenecks defined by the model's top logit values can leak task-irrelevant information present in an image-based query, in some cases revealing as much information as direct projections of the full residual stream.
    
[^365]: 从屏幕到动作缺少了什么？迈向多模态GUI推理的UI在环范式

    What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning

    [https://arxiv.org/abs/2604.06995](https://arxiv.org/abs/2604.06995)

    提出UI-in-the-Loop（UILoop）范式，将GUI推理建模为“屏幕-UI元素-动作”的循环过程，使多模态大语言模型显式学习关键UI元素的定位、语义与用法，实现精确的元素发现和可解释推理，并贡献了包含26K样本的UI理解基准。

    

    现有的图形用户界面（GUI）推理任务仍然具有挑战性，尤其是在UI理解方面。当前方法通常依赖基于屏幕的直接决策方式，缺乏可解释性，并且忽略了对UI元素的全面理解，最终导致任务失败。为了增强对UI的理解与交互，我们提出了一种创新的GUI推理范式——UI-in-the-Loop（UILoop，UI在环）。该方法将GUI推理任务视为一个循环的“屏幕-UI元素-动作”过程。通过使多模态大语言模型（MLLMs）显式学习关键UI元素的定位、语义功能和实际用法，UILoop实现了精确的元素发现并执行可解释的推理。此外，我们引入了一个以UI元素为中心、更具挑战性的UI理解任务，并设计了三个评估指标。相应地，我们贡献了一个包含26K样本的基准数据集。

    arXiv:2604.06995v3 Announce Type: replace  Abstract: Existing Graphical User Interface (GUI) reasoning tasks remain challenging, particularly in UI understanding. Current methods typically rely on direct screen-based decision-making, which lacks interpretability and overlooks a comprehensive understanding of UI elements, ultimately leading to task failure. To enhance the understanding and interaction with UIs, we propose an innovative GUI reasoning paradigm called UI-in-the-Loop (UILoop). Our approach treats the GUI reasoning task as a cyclic Screen-UI elements-Action process. By enabling Multimodal Large Language Models (MLLMs) to explicitly learn the localization, semantic functions, and practical usage of key UI elements, UILoop achieves precise element discovery and performs interpretable reasoning. Furthermore, we introduce a more challenging UI Comprehension task centered on UI elements with three evaluation metrics. Correspondingly, we contribute a benchmark of 26K samples (UI C
    
[^366]: 一图胜千言吗？基于视觉证据必要性的自适应多模态事实核查

    Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity

    [https://arxiv.org/abs/2604.04692](https://arxiv.org/abs/2604.04692)

    该论文挑战了“视觉证据总能提升事实核查准确性”的普遍假设，提出通过两个协同的视觉-语言模型自适应判断是否需要视觉证据的模块化框架AMuFC，在多个数据集上实现了更有效的事实核查。

    

    自动化事实核查是支持负责任信息生态系统的一项关键任务。尽管近期研究已经从纯文本事实核查发展到多模态事实核查，但一个普遍的假设是：引入视觉证据总能普遍提升核查准确性。在本研究中，我们挑战了这一假设，并证明不加区分地使用视觉证据反而可能降低准确性。基于这一发现，我们提出了AMuFC，一个模块化事实核查框架，它采用两个角色不同、相互协作的视觉-语言模型，实现视觉证据的自适应使用。在三个数据集上的实验结果（包括本研究提出的WebFC数据集）证明了在事实核查中自适应使用视觉证据的有效性。

    arXiv:2604.04692v3 Announce Type: replace-cross  Abstract: Automated fact-checking is a crucial task that supports a responsible information ecosystem. While recent research has progressed from text-only to multimodal fact-checking, a prevailing assumption is that incorporating visual evidence universally improves verification accuracy. In this work, we challenge this assumption and show that the indiscriminate use of visual evidence can reduce accuracy. Building on this finding, we propose AMuFC, a modular fact-checking framework that employs two collaborative vision-language models with distinct roles to enable the adaptive use of visual evidence. Experimental results on three datasets, including WebFC, introduced in this study, demonstrate the effectiveness of adaptive visual evidence use in fact-checking.
    
[^367]: 众多偏好，少量策略：面向多目标大语言模型对齐的紧凑策略组合

    Many Preferences, Few Policies: Compact Portfolios for Multi-Objective LLM Alignment

    [https://arxiv.org/abs/2604.04144](https://arxiv.org/abs/2604.04144)

    该论文提出 PALM 算法，通过结构化权重向量网格、惰性搜索与剪枝构建一个小型 LLM 策略组合，可证明地覆盖所有奖励权重下的近优对齐策略，以低成本实现多目标 LLM 对齐的个性化与部署。

    

    对齐大语言模型（LLM）需要在有用性、无害性和简洁性等相互竞争的目标之间进行权衡。合适的平衡因用户和应用而异，然而针对不同的奖励权重去训练、评估和部署大量策略的成本十分高昂。我们研究如何识别一个小型的 LLM 组合，使其在所有奖励权重设置下都能保持接近最优的性能。我们提出了 PALM（对齐 LLM 组合，Portfolio of Aligned LLMs）算法，该算法结合了结构化的权重向量网格、仅在需要之处才优化策略的惰性搜索以及剪枝技术。在给定目标近似容差的情况下，PALM 返回的组合可证明对每个权重向量都包含一个接近最优的策略，并对组合大小给出显式上界。这样的组合可以支持可扩展的个性化、模型开发过程中的奖励权重探索，以及紧凑的解码时配置。实验表明……（摘要在此处截断）

    arXiv:2604.04144v3 Announce Type: replace-cross  Abstract: Aligning large language models (LLMs) requires balancing competing objectives such as helpfulness, harmlessness, and conciseness. The appropriate balance varies across users and applications, yet training, evaluating, and deploying many policies across different reward weights is costly. We study how to identify a small portfolio of LLMs that preserves near-optimal performance across all reward weightings. We propose PALM (Portfolio of Aligned LLMs), an algorithm that combines a structured grid of weight vectors, a lazy search that optimizes policies only where needed, and pruning. Given target approximation tolerances, PALM returns a portfolio that provably contains a near-optimal policy for every weight vector, with an explicit upper bound on portfolio size. Such portfolios can support scalable personalization, reward-weight exploration during model development, and compact decoding-time configurations. Experiments show that 
    
[^368]: 评分量规质量理解与增强之漫游指南

    The Hitchhikers Guide to Rubric Quality Understanding and Enrichment

    [https://arxiv.org/abs/2604.01375](https://arxiv.org/abs/2604.01375)

    本文提出RIFT评分量规失败分类法及基于内容的量化信号，能以75%的准确率识别评分量规的失败模式（超过前沿大模型），并发现约20%的专家撰写量规存在权重反向的问题。

    

    评分量规凝练了专家对质量的判断，并被用于衡量智能体的表现。然而，评分量规本身的质量却从未被系统性测量，往往只能留待下游性能来间接评估。我们引入了心理测量学中为此专门构建的工具：基于评分量规内容的量化信号，并提出了评分量规失败分类法，涵盖九种评分量规可能的失败方式，按信度与内容效度两个维度组织。每种失败模式都会留下独特的特征信号。为证明这些信号能够因果性地追踪失败，我们注入了720个损坏样本，将每种RIFT失败模式以已知严重程度注入干净的评分量规中。基于这些信号的线性探针以75.0%的准确率识别出所注入的失败模式，优于被要求直接指出失败原因的前沿大模型（56.7%）。令人惊讶的是，在GDPval和Terminal-Bench上，48个专家撰写的评分量规中有10个将评分标准的权重设置反向，把更多分数权重放在了……（原文摘要在此截断）

    arXiv:2604.01375v3 Announce Type: replace  Abstract: Rubrics distill notions of expert quality and measure agent performance. However, the quality of rubrics themselves have not been systematically measured and are often left to downstream performance.We import apparatuses from measurement theory built for exactly this: quantitative signals based on the rubric's content, and introduce the RubrIc-Failure Taxonomy (RIFT), of nine possible ways a rubric fails, organized under reliability and content validity. Every mode leaves a distinct signature. To show the signals track failure causally, we seed 720 corruptions, injecting each RIFT mode into clean rubrics at known severity levels. A linear probe over the signals identifies which mode was injected at $75.0\%$ accuracy, beating $56.7\%$ for a frontier model asked to name the failure directly. Surprisingly across GDPval and Terminal-Bench, 10 of 48 expert-authored rubrics weight their criteria backwards, putting more of the score on requ
    
[^369]: GISTBench：通过基于证据的兴趣验证评估大语言模型的用户理解能力

    GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification

    [https://arxiv.org/abs/2603.29112](https://arxiv.org/abs/2603.29112)

    该论文提出GISTBench基准，通过兴趣扎根度（IG）和兴趣特异性（IS）两个新指标，评估大语言模型从推荐系统交互历史中提取和验证用户兴趣的能力，突破了传统推荐系统基准仅关注物品预测准确率的局限。

    

    我们推出了GISTBench，这是一个用于评估大语言模型（LLM）从推荐系统交互历史中理解用户能力的基准测试。与传统的专注于物品预测准确率的推荐系统（RecSys）基准不同，我们的基准评估的是LLM从用户参与行为数据中提取和验证用户兴趣的能力。我们提出了两个新颖的指标族：兴趣扎根度，将其分解为精确率和召回率两个组成部分，以分别惩罚幻觉产生的兴趣类别并奖励兴趣覆盖范围；以及兴趣特异性（IS），用于评估经验证的LLM预测用户画像的独特性。我们发布了一个基于全球短视频平台真实用户交互构建的合成数据集。我们的数据集包含隐式和显式参与信号以及丰富的文本描述。我们通过用户调研验证了数据集的保真度，并评估了八个开源权重的LLM模型。

    arXiv:2603.29112v2 Announce Type: replace  Abstract: We introduce GISTBench, a benchmark for evaluating Large Language Models' (LLMs) ability to understand users from their interaction histories in recommendation systems. Unlike traditional RecSys benchmarks that focus on item prediction accuracy, our benchmark evaluates how well LLMs can extract and verify user interests from engagement data. We propose two novel metric families: Interest Groundedness (IG), decomposed into precision and recall components to separately penalize hallucinated interest categories and reward coverage, and Interest Specificity (IS), which assesses the distinctiveness of verified LLM-predicted user profiles. We release a synthetic dataset constructed on real user interactions on a global short-form video platform. Our dataset contains both implicit and explicit engagement signals and rich textual descriptions. We validate our dataset fidelity against user surveys, and evaluate eight open-weight LLMs spanning
    
[^370]: 通过时间抽象实现前向-后向表示中的谱对齐

    Spectral Alignment in Forward-Backward Representations via Temporal Abstraction

    [https://arxiv.org/abs/2603.20103](https://arxiv.org/abs/2603.20103)

    本文证明时间抽象如同低通滤波器，可抑制高频谱分量、降低后继表示的有效秩并保持价值函数误差界，从而缓解连续环境高秩转移动力学与FB低秩瓶颈之间的谱失配，是实现稳定前向-后向表示学习的关键因素。

    

    前向-后向（FB）表示通过强制低秩分解，为在连续空间中学习后继表示（SR）提供了一个强大的框架。然而，连续环境的高秩转移动力学与FB架构的低秩瓶颈之间往往存在根本性的谱失配，这使得准确的低秩表示学习变得困难。在这项工作中，我们分析了时间抽象作为缓解这种失配的机制。通过刻画转移算子的谱特性，我们证明时间抽象的作用类似于一个抑制高频谱分量的低通滤波器。这种抑制降低了诱导SR的有效秩，同时保持了所得价值函数误差的形式化界。实验表明，这种对齐是稳定FB学习的关键因素，尤其是在高折扣因子的情况下。

    arXiv:2603.20103v4 Announce Type: replace-cross  Abstract: Forward-backward (FB) representations provide a powerful framework for learning the successor representation (SR) in continuous spaces by enforcing a low-rank factorization. However, a fundamental spectral mismatch often exists between the high-rank transition dynamics of continuous environments and the low-rank bottleneck of the FB architecture, making accurate low-rank representation learning difficult. In this work, we analyze temporal abstraction as a mechanism to mitigate this mismatch. By characterizing the spectral properties of the transition operator, we show that temporal abstraction acts analogously to a low-pass filter that suppresses high-frequency spectral components. This suppression reduces the effective rank of the induced SR while preserving a formal bound on the resulting value function error. Empirically, we show that this alignment is a key factor for stable FB learning, particularly at high discount factor
    
[^371]: 通过先验知识引导的图学习探索异构脑网络中的子网络交互

    Exploring Subnetwork Interactions in Heterogeneous Brain Network via Prior-Informed Graph Learning

    [https://arxiv.org/abs/2603.19307](https://arxiv.org/abs/2603.19307)

    提出KD-Brain框架，通过语义条件化交互机制和病理一致性约束将语义与临床先验知识注入图学习过程，有效解决了小样本条件下脑功能子网络交互建模难题，实现精神障碍诊断的最先进性能。

    

    建模功能子网络之间的复杂交互对于精神障碍的诊断和功能通路的识别至关重要。然而，由于训练样本数量有限，现有基于Transformer的方法在学习潜在子网络交互时仍面临重大挑战。为解决这些问题，我们提出了KD-Brain，一个先验知识引导的图学习框架，通过显式编码先验知识来指导学习过程。具体而言，我们设计了一种语义条件化交互机制，将语义先验注入注意力查询中，基于子网络的功能身份显式引导其交互学习。此外，我们引入了一种病理一致性约束，通过将学习到的交互分布与临床先验对齐来规范模型优化。此外，KD-Brain达到了最先进的性能。

    arXiv:2603.19307v2 Announce Type: replace-cross  Abstract: Modeling the complex interactions among functional subnetworks is crucial for the diagnosis of mental disorders and the identification of functional pathways. However, learning the interactions of the underlying subnetworks remains a significant challenge for existing Transformer-based methods due to the limited number of training samples. To address these challenges, we propose KD-Brain, a Prior-Informed Graph Learning framework for explicitly encoding prior knowledge to guide the learning process. Specifically, we design a Semantic-Conditioned Interaction mechanism that injects semantic priors into the attention query, explicitly navigating the subnetwork interactions based on their functional identities. Furthermore, we introduce a Pathology-Consistent Constraint, which regularizes the model optimization by aligning the learned interaction distributions with clinical priors. Additionally, KD-Brain leads to state-of-the-art p
    
[^372]: 基于离散扩散的可控口音规范化

    Controllable Accent Normalization via Discrete Diffusion

    [https://arxiv.org/abs/2603.14275](https://arxiv.org/abs/2603.14275)

    提出了基于掩码离散扩散的可控口音规范化系统DLM-AN，通过选择性复用共同令牌实现口音强度的灵活控制，并借助流匹配时长预测器匹配母语节奏，在多口音英语数据上取得最低词错误率。

    

    现有的口音规范化方法通常无法控制口音强度，然而许多应用——如语言学习和配音——需要可调节的口音保留程度。我们提出了DLM-AN，一个构建于自监督语音令牌上的掩码离散扩散的可控口音规范化系统。一个共同令牌预测器识别可能编码母语发音的源令牌；这些令牌被有选择地复用以初始化反向扩散过程。这提供了一种简单而有效的口音强度控制机制：复用越多令牌，则保留越多原始口音。DLM-AN还集成了一个基于流匹配的时长比率预测器，可自动调整总时长以更好地匹配母语节奏。在多口音英语数据上的实验表明，DLM-AN在所有对比系统中取得了最低的词错误率，同时实现了具有竞争力的口音减少效果。

    arXiv:2603.14275v3 Announce Type: replace-cross  Abstract: Existing accent normalization methods do not typically offer control over accent strength, yet many applications-such as language learning and dubbing-require tunable accent retention. We propose DLM-AN, a controllable accent normalization system built on masked discrete diffusion over self-supervised speech tokens. A Common Token Predictor identifies source tokens that likely encode native pronunciation; these tokens are selectively reused to initialize the reverse diffusion process. This provides a simple yet effective mechanism for controlling accent strength: reusing more tokens preserves more of the original accent. DLM-AN further incorporates a flow-matching Duration Ratio Predictor that automatically adjusts the total duration to better match the native rhythm. Experiments on multi-accent English data show that DLM-AN achieves the lowest word error rate among all compared systems while delivering competitive accent reduc
    
[^373]: 话到嘴边：为什么大语言模型会幻觉出它们本可解码出的答案

    On the Tip of the Tongue: Why LLMs Hallucinate Answers They Can Decode

    [https://arxiv.org/abs/2603.13911](https://arxiv.org/abs/2603.13911)

    该论文提出在首个答案标记处区分“读取”与“写出”的新框架，揭示大语言模型产生幻觉的关键原因并非正确答案无法从中间状态解码，而是最终读出时的“选择边际”不足，使更强的竞争标记压制了正确答案。

    

    即使在正确答案能够从其中间状态解码出来的情况下，语言模型也可能给出错误的答案。为了研究这种可解码性与选择之间的差距，我们在第一个答案标记处区分了“读取”与“写出”。“读取”问的是：在相同关系诱饵控制的条件下，正确标记能否从中间残差状态中被解码出来；“写出”问的是：最终的读出是否将该标记排在所有内容标记的首位。在三种不同的读取器下，并采用随机标签控制实验，仍有相当大比例的失败案例保持“可读取”状态，同时另一个内容标记被选中。我们通过最终读出处的选择边际来解释这一现象：选择边际是答案logit与其最强竞争者logit之间的差值，即答案支持度减去竞争者支持度，并且可以进一步分解为与标记频率相关的上下文平均基线和一个项目特定的部分。将答案支持度设置为……（原文摘要在此处截断）

    arXiv:2603.13911v2 Announce Type: replace  Abstract: A language model can give the wrong answer even when the correct answer is decodable from its intermediate states. To study this gap between decodability and selection, we distinguish \textit{read} from \textit{write} at the first answer token. Read asks whether the gold token can be decoded from intermediate residual states under same-relation decoy controls. Write asks whether the final readout ranks that token first among content tokens. Under three different readers, with a randomized-label control, a substantial fraction of failures remain readable while another content token is selected. We explain this through the selection margin at the final readout, the difference between the answer logit and the logit of its strongest alternative, which is answer support minus alternative support, and can also be split into a context-averaged baseline linked to token frequency and an item-specific term. Setting the answer support to the le
    
[^374]: 评估格式而非模型能力，决定了消费级健康AI评估中所测得的分诊失败率

    Evaluation format, not model capability, drives measured triage failure in the assessment of consumer health AI

    [https://arxiv.org/abs/2603.11413](https://arxiv.org/abs/2603.11413)

    该研究通过机制性实验与忠实复现证明，消费级健康AI分诊失败的高错误率主要源于考试式的评估格式（强制选项输出、禁止澄清提问），而非模型本身的能力不足——在自然的患者风格消息下，前沿大语言模型的分诊表现显著更好。

    

    arXiv:2603.11413v4 公告类型：replace-cross 摘要：近期一项发表于《自然·医学》的研究报告称，ChatGPT Health 对51.6%的急诊情况进行了欠分诊，并由此得出结论：面向消费者的AI分诊存在安全风险。然而，该研究的实验方案采用了一种考试式的框架（强制输出A/B/C/D选项、抑制背景知识、不允许提出澄清性问题），这与消费者实际使用健康聊天机器人的方式并不相符。我们提出疑问：这一引人注目的错误率究竟是模型本身的属性，还是测量方法的属性。在第一项机制性研究中，五个前沿大语言模型在包含17个场景的题库上，以自然的患者风格消息作答时，比在受限框架下得分高出6.4分（p=0.015）；在其中一个病例情景上，三个模型的表现从强制选择模式下的0-24%跃升至自由文本模式下的100%。在第二项忠实复现研究中，我们将原作者公开发布的60个病例情景以四种相互匹配的格式输入六个前沿模型，对改写文本进行了临床医生验证，并对作为裁定者的大语言模型进行了盲法临床医生审计。此处，直接……

    arXiv:2603.11413v4 Announce Type: replace-cross  Abstract: A recent Nature Medicine study reported that ChatGPT Health under-triages 51.6% of emergencies and concluded that consumer-facing AI triage poses safety risks. Its protocol, however, was an exam-style scaffold (forced A/B/C/D output, knowledge suppression, no clarifying questions) unlike how consumers use health chatbots. We ask whether the headline error rate is a property of the models or of the measurement. In a first, mechanistic study, five frontier LLMs on a 17-scenario bank scored 6.4 points higher under naturalistic patient-style messages than under the constrained scaffold (p=0.015), and on one vignette three models went from 0-24% with forced choice to 100% with free text. In a second, faithful replication we ran the authors' own 60 released vignettes through six frontier models under four matched formats, with clinician validation of the rewrites and a blinded clinician audit of the LLM adjudicators. Here the directi
    
[^375]: 通过混合大语言模型（LLM）-符号规划与LLM引导的强化学习实现新颖性适应

    Novelty Adaptation Through Hybrid Large Language Model (LLM)-Symbolic Planning and LLM-guided Reinforcement Learning

    [https://arxiv.org/abs/2603.11351](https://arxiv.org/abs/2603.11351)

    该论文提出了一种融合符号规划、强化学习与大语言模型的神经符号架构，利用LLM的常识推理能力识别缺失算子、生成计划并编写奖励函数，使机器人能够有效适应开放世界环境中的新颖物体。

    

    在动态开放世界环境中，自主智能体经常会遇到阻碍其找到实现目标的计划的新颖事物。具体而言，当机器人的规划域缺乏使其能够与环境中的新物体进行适当交互的算子时，传统的符号规划器无法生成计划。我们提出了一种神经符号架构，该架构集成了符号规划、强化学习和大语言模型（LLM），以学习如何处理新颖物体。特别是，我们利用LLM的常识推理能力来识别缺失的算子，与符号AI规划器协同生成计划，并编写奖励函数来引导强化学习智能体学习针对新识别算子的控制策略。我们的方法在算子发现以及连续机器人领域的算子学习方面均优于当前最先进的方法。

    arXiv:2603.11351v2 Announce Type: replace-cross  Abstract: In dynamic open-world environments, autonomous agents often encounter novelties that hinder their ability to find plans to achieve their goals. Specifically, traditional symbolic planners fail to generate plans when the robot's planning domain lacks the operators that enable it to interact appropriately with novel objects in the environment. We propose a neuro-symbolic architecture that integrates symbolic planning, reinforcement learning, and a large language model (LLM) to learn how to handle novel objects. In particular, we leverage the common sense reasoning capability of the LLM to identify missing operators, generate plans with the symbolic AI planner, and write reward functions to guide the reinforcement learning agent in learning control policies for newly identified operators. Our method outperforms the state-of-the-art methods in operator discovery as well as operator learning in continuous robotic domains.Our webpage
    
[^376]: 基于评论蒸馏表示的感官感知序列推荐

    Sensory-Aware Sequential Recommendation via Review-Distilled Representations

    [https://arxiv.org/abs/2603.02709](https://arxiv.org/abs/2603.02709)

    该论文提出ASER离线流水线，通过微调大语言模型从评论中提取有据可查的感官属性并蒸馏为冻结的五维感官库，再以轻量级关系度量增强序列推荐，同时保持预训练主干不变。

    

    序列推荐器从物品标识符中学习行为模式，然而用户在评论中描述的体验性属性，例如产品的外观、触感、气味、口味或声音，很少以可控、可审计的形式融入物品表示中。我们提出了ASER（基于属性的感官增强表示），这是一个离线流水线，它通过微调大语言模型从评论文本中提取有证据支撑的感官属性-值记录，例如“颜色：哑光黑”或“气味：香草”，并将其蒸馏到一个紧凑的学生编码器中，为每个物品目录生成一个冻结的五维感官库。在推荐阶段，预训练主干保持冻结：在感官库之上学习用户历史与每个候选物品之间的轻量级关系度量，其校正在通过验证选择的幅度界限内应用。在五个亚马逊领域和四个主干模型上进行训练……（原文摘要在此处截断）

    arXiv:2603.02709v4 Announce Type: replace-cross  Abstract: Sequential recommenders learn behavioral patterns from item identifiers, while the experiential properties that users describe in reviews, such as how products look, feel, smell, taste, or sound, rarely enter item representations in a controlled, auditable form.   We present ASER (Attribute-based Sensory-Enhanced Representation), an offline pipeline that fine-tunes a large language model to extract evidence-grounded sensory attribute-value records, such as color: matte black or scent: vanilla, from review text and distills them into a compact student encoder that produces a frozen five-facet sensory bank for each item catalog.   At recommendation time the pretrained backbone stays frozen: a lightweight relational metric between the user history and each candidate is learned over the bank, and its correction is applied within a validation-selected magnitude bound.   Across five Amazon domains and four backbones, trained within a
    
[^377]: Goldilocks 强化学习：通过调节任务难度摆脱稀疏奖励，提升语言模型推理能力

    Goldilocks RL: Tuning Task Difficulty to Escape Sparse Rewards for Reasoning

    [https://arxiv.org/abs/2602.14868](https://arxiv.org/abs/2602.14868)

    提出Goldilocks自适应数据选择策略，利用选择器网络预测问题的奖励波动性，优先选取难度适中（既不太简单也不太难）的训练问题，从而摆脱稀疏奖励困境，提升语言模型推理强化学习的样本效率。

    

    强化学习已成为解锁语言模型推理能力的强大范式。然而，依赖稀疏奖励使得这一过程样本效率极低，因为模型必须在极少反馈的情况下探索庞大的搜索空间。虽然经典的课程学习旨在通过按复杂度对数据排序来缓解这一问题，但先前的工作主要针对小型数据集，无法直接迁移到现代语言模型训练的大规模场景。此外，针对特定模型的合适排序往往并不明确。为解决这一问题，我们提出了Goldilocks，一种自适应数据选择策略，它使用一个选择器网络来预测每个候选问题在模型多次 rollout 中奖励的标准差。选择器会优先选取预测奖励波动性高的问题，这类问题对模型而言既不太简单也不太难，恰到好处。

    arXiv:2602.14868v3 Announce Type: replace-cross  Abstract: Reinforcement learning has emerged as a powerful paradigm for unlocking reasoning capabilities in language models. However, relying on sparse rewards makes this process highly sample-inefficient, as models must navigate vast search spaces with minimal feedback. While classic curriculum learning aims to mitigate this by ordering data based on complexity, prior works have primarily targeted small datasets and do not directly transfer to the large-scale settings typical of modern language model training. Furthermore, the right ordering for a specific model is often unclear. To address this, we propose Goldilocks, an adaptive data-selection strategy that uses a Selector network to predict the standard deviation of rewards across the model's rollouts for each candidate question. The Selector prioritizes questions with high predicted reward variability, corresponding to questions that are neither too easy nor too hard for the model's
    
[^378]: 有效深度悖论：深度卷积神经网络中的拓扑结构与可训练性

    The Effective Depth Paradox: Topology and Trainability in Deep CNNs

    [https://arxiv.org/abs/2602.13298](https://arxiv.org/abs/2602.13298)

    该论文提出“有效深度”（$D_{eff}$）这一闭式预训练代理指标，用于量化前向信息路径的期望长度，从而揭示了VGG、ResNet和GoogLeNet等不同拓扑结构中名义深度与实际可训练性之间的“有效深度悖论”。

    

    本文对卷积神经网络（CNN）拓扑结构与图像分类性能进行了受控比较研究，涵盖VGG、ResNet和GoogLeNet三个架构家族，并在统一训练协议下于CIFAR-10数据集上评估。我们形式化地区分了名义深度（$D_{\mathrm{nom}}$，即承载权重的层的物理数量）与有效深度（$D_{\mathrm{eff}}$，一个量化前向信息路径期望长度的操作性指标），将Veit等人（2016）提出的残差网络路径集合解释扩展为覆盖顺序、残差和多分支拓扑的闭式预训练代理指标。我们通过基于观测反向传播信号计算的梯度加权变体对该代理指标进行了验证。在八个代表性模型（VGG-11/13/16/19、ResNet-18/34/50、GoogLeNet）中，朴素VGG式堆叠网络在名义深度增加时表现出早期精度饱和……

    arXiv:2602.13298v4 Announce Type: replace-cross  Abstract: This paper presents a controlled comparative study of convolutional neural network (CNN) topology and image classification performance across the architectural families VGG, ResNet, and GoogLeNet, evaluated on CIFAR-10 under a unified training protocol. We formalize the distinction between nominal depth ($D_{\mathrm{nom}}$), the physical count of weight-bearing layers, and effective depth ($D_{\mathrm{eff}}$), an operational metric quantifying the expected length of forward information paths, extending the path-ensemble interpretation of residual networks introduced by Veit et al. (2016) into closed-form, pre-training proxies spanning sequential, residual, and multi-branch topologies. We validate this proxy against a gradient-weighted variant computed from observed backpropagation signal. Across eight representative models (VGG-11/13/16/19, ResNet-18/34/50, GoogLeNet), plain VGG-style stacks show early accuracy saturation as $D
    
[^379]: ANCRe：面向高效深度扩展的自适应神经连接重分配

    ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling

    [https://arxiv.org/abs/2602.09009](https://arxiv.org/abs/2602.09009)

    该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。

    

    摘要（arXiv:2602.09009v2，公告类型：replace-cross）：扩展网络深度一直是现代基础模型成功的核心驱动力，然而近期研究表明，深层网络往往未被充分利用。本文从优化视角重新审视了加深神经网络的默认机制——残差连接。严格的分析证明，残差连接的布局能够从根本上塑造收敛行为，甚至会引发收敛速率上的指数级差距。受此启发，我们提出了自适应神经连接重分配，这是一个具有理论依据且轻量级的框架，能够从数据中参数化并学习残差连接方式。ANCRe 以可忽略不计的计算和内存开销（<1%）自适应地重新分配残差连接，同时使网络深度得到更有效的利用。我们在大型语言模型预训练、扩散模型以及深度R（原文此处截断）等任务上进行了大量数值测试……

    arXiv:2602.09009v2 Announce Type: replace-cross  Abstract: Scaling network depth has been a central driver behind the success of modern foundation models, yet recent investigations suggest that deep layers are often underutilized. This paper revisits the default mechanism for deepening neural networks, namely residual connections, from an optimization perspective. Rigorous analysis proves that the layout of residual connections can fundamentally shape convergence behavior, and even induces an exponential gap in convergence rates. Prompted by this insight, we introduce adaptive neural connection reassignment (ANCRe), a principled and lightweight framework that parameterizes and learns residual connectivities from the data. ANCRe adaptively reassigns residual connections with negligible computational and memory overhead ($<1\%$), while enabling more effective utilization of network depth. Extensive numerical tests across pre-training of large language models, diffusion models, and deep R
    
[^380]: 扩展到现实：3D环境中的提示注入攻击

    Extended to Reality: Prompt Injection in 3D Environments

    [https://arxiv.org/abs/2602.07104](https://arxiv.org/abs/2602.07104)

    本文提出PI3D，一种通过物理放置带文本3D对象而非数字编辑来攻击MLLMs的提示注入方法，并系统化解决攻击对象姿态优化问题。

    

    多模态大语言模型（MLLMs）已提升了在3D环境中解释和响应视觉输入的能力，推动了机器人技术和情境化对话代理等多样化应用的发展。当MLLMs对摄像头捕获的物理世界视图进行推理时，出现了一个新的攻击面：攻击者可以在环境中放置带有文本的物理对象，以覆盖MLLMs的预期任务。尽管先前的研究已在文本领域和通过数字编辑的2D图像中探讨了提示注入，但关于这些攻击如何在3D环境中运作的关注有限。为填补这一空白，我们提出了PI3D，一种针对3D环境中MLLMs的提示注入攻击，通过放置带有文本的对象而非数字图像编辑来实现。我们公式化并解决了为注入文本的3D对象确定有效姿态（位置和方向）的问题，其中攻击者的目标是...

    arXiv:2602.07104v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) have advanced the capabilities to interpret and act on visual input in 3D environments, empowering diverse applications such as robotics and situated conversational agents. When MLLMs reason over camera-captured views of the physical world, a new attack surface emerges: an attacker can place text-bearing physical objects in the environment to override MLLMs' intended task. While prior work has studied prompt injection in the text domain and through digitally edited 2D images, limited attention has been paid to how these attacks function in 3D environments. To bridge the gap, we introduce PI3D, a prompt injection attack against MLLMs in 3D environments, realized through text-bearing object placement rather than digital image edits. We formulate and solve the problem of identifying an effective pose (position and orientation) for a 3D object with injected text, where the attacker's goal is
    
[^381]: LPS-Bench：在良性与对抗场景下评测计算机使用智能体长程规划安全意识的基准

    LPS-Bench: Benchmarking Safety Awareness of Computer-Use Agents in Long-Horizon Planning under Benign and Adversarial Scenarios

    [https://arxiv.org/abs/2602.03255](https://arxiv.org/abs/2602.03255)

    该论文提出LPS-Bench基准，通过模板引导的多智能体流水线高效生成570个覆盖7个任务领域和9种规划风险类型的测试案例，用以评测计算机使用智能体在良性请求与对抗性引导下的长程规划安全意识。

    

    arXiv:2602.03255v2 公告类型：替换。摘要：计算机使用智能体（CUA）通过工具执行多阶段任务，其早期的不安全决策可能会传导并引发后果严重的后续操作。仅评估最终结果可能会遗漏此类决策，而为新任务构建可执行环境又会使基准扩展的成本高昂。我们提出了LPS-Bench，这是一个用于评估MCP风格工具工作流中长程规划安全性的基准，涵盖良性请求和对抗性引导两种情形。该基准采用模板引导的多智能体流水线生成用户指令、模拟工具包以及针对具体案例的安全评估标准，并由人工进行审查。这一设计支持可扩展的案例扩充，无需为每个测试案例单独构建应用程序环境。LPS-Bench包含570个案例，源自7个任务领域和9种规划风险类型下的65个场景，其中代表性案例还被进一步适配为可复用的技能。基于大语言模型的评估器针对完整的交互过程应用案例特定的安全标准进行评判。

    arXiv:2602.03255v2 Announce Type: replace  Abstract: Computer-use agents (CUAs) execute multi-stage tasks through tools, where an early unsafe decision can propagate to consequential actions. Evaluating only final outcomes can miss such decisions, while constructing executable environments for new tasks can make benchmark expansion costly. We present LPS-Bench, a benchmark of long-horizon planning safety in MCP-style tool workflows under benign requests and adversarial steering. A template-guided multi-agent pipeline generates user instructions, simulated toolkits, and case-specific safety criteria, followed by human review. This design supports scalable case expansion without building a separate application environment for every test case. LPS-Bench comprises 570 cases derived from 65 scenarios across 7 task domains and 9 planning-risk types, with representative cases additionally adapted to reusable skills. An LLM-based evaluator applies case-specific criteria to complete interaction
    
[^382]: IntentCoding：在代码生成中放大用户意图

    IntentCoding: Amplifying User Intent in Code Generation

    [https://arxiv.org/abs/2602.00066](https://arxiv.org/abs/2602.00066)

    提出IntentCoding解码策略，通过屏蔽意图来捕捉用户意图的影响，并利用多强度集成机制放大该影响，无需额外训练即可显著提升大语言模型在多约束代码生成任务中对用户意图的遵循能力。

    

    大语言模型（LLMs）在代码生成方面已展现出强大的能力，但在遵循包含多重约束的细粒度用户意图方面仍然是一个重大挑战。我们的实证分析揭示了两个关键观察：1）随着用户意图中约束数量的增加，模型性能迅速下降；2）虽然用户意图确实会影响模型的logits，但这种影响可能不足以有效引导解码过程。为此，我们提出了意图放大代码生成方法（IntentCoding），这是一种新颖的解码策略，能够增强大语言模型遵循用户意图的能力。IntentCoding通过屏蔽用户意图来捕捉其影响，并应用多强度集成机制在生成过程中放大用户意图的作用。IntentCoding与模型无关，无需额外训练，并可与现有解码过程无缝集成。

    arXiv:2602.00066v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown strong capabilities in code generation, but their adherence to fine-grained user intent with multiple constraints remains a significant challenge. Our empirical analysis reveals two key observations: 1) Model performance deteriorates quickly as the number of constraints in the user intent increases, and 2) While user intent does influence the model's logits, such an influence may not be strong enough to effectively steer the decoding process. To this end, we propose Intent-Amplified Code Generation (IntentCoding), a novel decoding strategy that enhances an LLM's ability to follow user intent. IntentCoding captures the influence of user intent by masking out the intent, and applies a multi-strength ensemble mechanism to amplify the effect of user intent during generation. IntentCoding is model-agnostic, requires no additional training, and integrates seamlessly with existing decoding procedures. T
    
[^383]: 可恢复性存在定律：面向工具增强智能体的ERR度量

    Recoverability Has a Law: The ERR Measure for Tool-Augmented Agents

    [https://arxiv.org/abs/2601.22352](https://arxiv.org/abs/2601.22352)

    本文提出期望恢复遗憾（ERR）指标并证明其与可观测的效率得分（ES）之间存在一阶定量关系，首次为工具增强语言模型智能体失败后的自我恢复能力建立了可证伪的预测性定律，并在五个工具使用基准上得到实证验证。

    

    语言模型智能体在工具调用执行失败后常常表现出自我恢复的能力，然而这一行为一直缺乏形式化的解释。我们提出了一个预测性理论来填补这一空白，证明可恢复性遵循一条可测量的定律。具体而言，我们通过期望恢复遗憾Expected Recovery Regret, ERR）来形式化可恢复性，该指标量化了在随机执行噪声下恢复策略相对于最优策略的偏离程度，并推导出ERR与一个经验可观测量——效率得分Efficiency Score, ES）之间的一阶关系。由此得到了一个可证伪的、关于工具使用智能体恢复动力学的一阶定量定律。我们在五个工具使用基准上对该定律进行了实证验证，涵盖受控扰动、诊断推理和真实世界API等场景。在不同模型规模、扰动机制和恢复时间跨度下，ERR-ES定律所预测的遗憾值与观测到的失败后恢复行为高度吻合。

    arXiv:2601.22352v2 Announce Type: replace-cross  Abstract: Language model agents often appear capable of self-recovery after failing tool call executions, yet this behavior lacks a formal explanation. We present a predictive theory that resolves this gap by showing that recoverability follows a measurable law. To elaborate, we formalize recoverability through Expected Recovery Regret (ERR), which quantifies the deviation of a recovery policy from the optimal one under stochastic execution noise, and derive a first-order relationship between ERR and an empirical observable quantity, the Efficiency Score (ES). This yields a falsifiable first-order quantitative law of recovery dynamics in tool-using agents. We empirically validate the law across five tool-use benchmarks spanning controlled perturbations, diagnostic reasoning, and real-world APIs. Across model scales, perturbation regimes, and recovery horizons, predicted regret under the ERR-ES law closely matched observed post-failure re
    
[^384]: AstroAgentBench：在太空任务规划任务上评估智能体规划能力

    AstroAgentBench: Evaluating Agentic Planning on Space Mission Planning Tasks

    [https://arxiv.org/abs/2601.11354](https://arxiv.org/abs/2601.11354)

    本文提出AstroAgentBench——一个涵盖调度、观测规划、星座设计和中继支持等七大任务族的可执行太空任务规划基准，通过外部验证器评估智能体生成的规划产物，发现最强LLM智能体系统在部分任务上可接近或超越求解器参考水平，而较弱系统则难以产出高价值的有效规划。

    

    arXiv:2601.11354v2 公告类型：替换。摘要：近期的“大语言模型用于航天”（LLM-for-Space）系统涉及任务规划、调度、运营支持、模拟器控制和自主性等方面，但它们的评估采用了不同的任务契约、控制设置、模拟器和成功标准。我们提出了AstroAgentBench，这是一个包含七大任务族的可执行太空任务规划基准，涵盖调度、观测规划、星座设计和中继支持等领域。对于每个案例，智能体需要提交一个规划产物，该产物由外部验证器对其模式结构、时序、几何关系、资源和任务价值进行检查。结果报告有效性和归一化分数，并与任务特定的求解器参考结果进行比较。在五个LLM智能体系统和35个保留测试案例上的结果显示，最强的系统在若干任务族上接近或超过求解器参考分数，而较弱的系统往往无法产出高价值的有效规划，即便是强大的系统在几何、产品级或设计繁重的任务上也会出现质量下降。

    arXiv:2601.11354v2 Announce Type: replace  Abstract: Recent LLM-for-Space systems address mission planning, scheduling, operations support, simulator control, and autonomy, but their evaluations use different task contracts, control settings, simulators, and success criteria. We introduce AstroAgentBench, a seven-family benchmark for executable space mission planning in the domains of scheduling, observation planning, constellation design, and relay support. For each case, an agent submits a planning artifact that is checked by an external verifier for schema, timing, geometry, resources, and mission value. Results report validity and normalized scores, with comparisons to task-specific solver references. Across five LLM agent systems and 35 held-out cases, the strongest systems approach or exceed solver-reference scores on several families, while weaker systems often fail to produce high-value valid plans and even strong systems lose quality on geometric, product-level, or design-heav
    
[^385]: 道德是情境化的：利用概率聚类与大语言模型从人类数据中学习可解释的道德情境

    Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models

    [https://arxiv.org/abs/2512.21439](https://arxiv.org/abs/2512.21439)

    提出了COMETH框架，将概率情境学习与大语言模型语义抽象及人类道德判断数据相结合，从数据中学习可解释的道德情境，证明道德评价是高度情境化的。

    

    当前AI对齐研究中的一个关键问题是如何让AI算法学习道德价值观。由于人类道德高度依赖情境，对行为的评判不仅取决于其结果，还取决于行为发生的情境。我们提出了COMETH（基于文本人类输入的道德评估情境组织），这是一个将概率情境学习器与基于大语言模型的语义抽象及人类道德评估相结合的框架，用于建模情境如何塑造模糊行为的可接受性。我们构建了一个基于实证的数据集，包含与三条道德规则（违反“不可杀人”、“不可欺骗”和“不可违法”）相关的六种核心行为共300个场景，并收集了101名参与者的三元判断（谴责/中立/支持）。预处理流程通过大语言模型过滤器与结合K-means聚类的MiniLM嵌入对行为进行标准化，产生稳健且可复现的核心行为聚类。

    arXiv:2512.21439v2 Announce Type: replace-cross  Abstract: A key question in current AI alignment research is how to make AI algorithms learn moral values. Because human morality is highly context-dependent, actions are judged not only by their outcomes but by the context in which they occur. We present COMETH (Contextual Organization of Moral Evaluation from Textual Human inputs), a framework that integrates a probabilistic context learner with LLM-based semantic abstraction and human moral evaluations to model how context shapes the acceptability of ambiguous actions. We curate an empirically grounded dataset of 300 scenarios across six core actions relative to three moral rules (violating "Do not kill", "Do not deceive", and "Do not break the law") and collect ternary judgments (Blame/Neutral/Support) from N=101 participants. A preprocessing pipeline standardizes actions via an LLM filter and MiniLM embeddings with K-means, producing robust, reproducible core-action clusters. COMETH
    
[^386]: 揭开LLM-as-a-Judge（大语言模型作为评判者）的神秘面纱：面向推理时扩展的解析可处理模型

    Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling

    [https://arxiv.org/abs/2512.19905](https://arxiv.org/abs/2512.19905)

    该论文提出了一个解析可处理的推理时扩展模型——带奖励加权采样器的贝叶斯线性回归，用以模拟LLM作为评判者的场景，并在高维机制下推导出后验预测均值与方差的闭式表达式，从而揭示推理时扩展背后的数学原理。

    

    大语言模型的最新发展表明，将相当一部分计算资源从训练阶段重新分配到推理阶段具有优势。然而，推理时扩展背后的原理尚未得到充分理解。在本文中，我们引入了一个解析可处理的推理时扩展模型：带有奖励加权采样器的贝叶斯线性回归，其中奖励由线性模型确定，以模拟LLM-as-a-judge（大语言模型作为评判者）场景。我们在高维机制下研究这一问题，借助确定性等价方法得到了后验预测均值和方差的闭式表达式。我们分析了训练数据从教师模型采样时的泛化误差。我们抽取k个推理时样本，并通过在二次奖励上施加温度参数的softmax进行选择。当奖励与教师模型差异不大时，泛化误差单调递减（摘要在此处截断）。

    arXiv:2512.19905v3 Announce Type: replace-cross  Abstract: Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonical
    
[^387]: 科学正在落后于前沿：五十万篇论文中的基础模型采用情况

    Science Is Falling Behind the Frontier: Foundation Model Adoption Across Half a Million Papers

    [https://arxiv.org/abs/2511.21739](https://arxiv.org/abs/2511.21739)

    该研究首次对50万篇论文中的AI基础模型采用情况进行大规模分析，发现科学界采用的模型规模已从2015年领先前沿模型5.4倍逆转为2024年落后6.9倍，这种“规模滞后”可能正在限制科学家充分获取AI赋能科学的收益。

    

    我们首次对科学领域AI基础模型的使用情况进行了大规模分析——而不仅仅是引用或关键词。我们发现基础模型的采用率以近乎指数级的速度快速增长，其中语言学、计算机科学和工程学的采用率最高。视觉模型是科学领域使用最多的基础模型，但语言模型所占份额正在增长。开放权重模型占据主导地位。随着AI开发者不断增加其模型的参数量，科学家们也在跟进，但速度要慢得多：2015年，科学领域采用的基础模型平均规模是当时正在构建模型平均规模的5.4倍；而到2024年，这一关系发生了逆转，前沿构建模型的平均规模已达到科学领域采用模型的6.9倍。我们还提供了提示性证据，表明科学家使用这些较小的模型可能会限制他们充分获得AI赋能科学的全部益处，因为使用较大模型的论文往往发表在影响力更高的期刊上。

    arXiv:2511.21739v2 Announce Type: replace-cross  Abstract: We present the first large-scale analysis of AI foundation model usage in science -- not just citations or keywords. We find that adoption has grown rapidly, at nearly-exponential rates, with the highest uptake in Linguistics, Computer Science, and Engineering. Vision models are the most used foundation models in science, although language models' share is growing. Open-weight models dominate. As AI builders increase the parameter counts of their models, scientists have followed suit but at a much slower rate: in 2015, the mean foundation model adopted in science was 5.4x larger than the mean model being built; by 2024 that relationship had reversed, with the mean model built 6.9x larger than the mean model adopted. We also present suggestive evidence that scientists' use of these smaller models may be limiting them from getting the full benefits of AI-enabled science, as papers that use larger models appear in higher-impact jo
    
[^388]: 用于孟加拉语新闻标题分类与情感分析同步进行的统一BERT-CNN-BiLSTM框架

    A Unified BERT-CNN-BiLSTM Framework for Simultaneous Headline Classification and Sentiment Analysis of Bangla News

    [https://arxiv.org/abs/2511.18618](https://arxiv.org/abs/2511.18618)

    本文提出了一个统一的BERT-CNN-BiLSTM混合迁移学习框架，首次实现了孟加拉语新闻标题分类与情感分析的同步处理。

    

    在我们的日常生活中，报纸是一种重要的信息来源，影响着公众对当下议题的讨论方式。然而，如何有效地浏览来自不同报纸和在线新闻门户的海量新闻内容是一项挑战。结合情感分析的报纸标题能够告诉我们新闻的内容（如政治、体育）以及新闻带给我们的感受（积极、消极、中性），这有助于我们快速理解新闻的情感基调。本研究提出了一种最先进的方法，将孟加拉语新闻标题分类与情感分析相结合，应用了自然语言处理（NLP）技术，特别是混合迁移学习模型BERT-CNN-BiLSTM。我们探索了一个名为BAN-ABSA的数据集，包含9014条新闻标题，这是首次在孟加拉语报纸中同时进行标题分类和情感分类的实验。

    arXiv:2511.18618v2 Announce Type: replace-cross  Abstract: In our daily lives, newspapers are an essential information source that impacts how the public talks about present-day issues. However, effectively navigating the vast amount of news content from different newspapers and online news portals can be challenging. Newspaper headlines with sentiment analysis tell us what the news is about (e.g., politics, sports) and how the news makes us feel (positive, negative, neutral). This helps us quickly understand the emotional tone of the news. This research presents a state-of-the-art approach to Bangla news headline classification combined with sentiment analysis applying Natural Language Processing (NLP) techniques, particularly the hybrid transfer learning model BERT-CNN-BiLSTM. We have explored a dataset called BAN-ABSA of 9014 news headlines, which is the first time that has been experimented with simultaneously in the headline and sentiment categorization in Bengali newspapers. Over
    
[^389]: 基于代表性智能体的大语言模型引导强化学习交通建模

    LLM-Guided Reinforcement Learning with Representative Agents for Traffic Modeling

    [https://arxiv.org/abs/2511.06260](https://arxiv.org/abs/2511.06260)

    提出用单个代表性LLM智能体建模同质出行者群体，将LLM的正向强化判断通过可解释规则转化为混合策略更新，从而实现可扩展且稳定的逐日交通流建模。

    

    大语言模型（LLM）正越来越多地被用作基于智能体的交通模型中自利出行者的行为代理。尽管比传统模型更灵活、更具泛化能力，但由于需要为每位出行者调用一次LLM，其高昂成本限制了这些方法的可扩展性和实际应用。此外，研究发现LLM智能体往往做出不透明的选择，并产生不稳定的逐日动态。为应对这些挑战，我们提出用一个代表性LLM智能体来建模每个面临相同决策情境的同质出行者群体，该智能体的行为类似于群体的平均水平，维护并更新路径上的混合策略，使其与群体的总体流量比例保持一致。每天，LLM会回顾出行体验，并对希望更频繁使用的路径进行正向强化标记，随后一个可解释的更新规则将这一判断转化为策略的更新。

    arXiv:2511.06260v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used as behavioral proxies for self-interested travelers in agent-based traffic models. Although more flexible and generalizable than conventional models, the practical use of these approaches remains limited by scalability due to the cost of calling one LLM for every traveler. Moreover, it has been found that LLM agents often make opaque choices and produce unstable day-to-day dynamics. To address these challenges, we propose to model each homogeneous traveler group facing the same decision context with a single representative LLM agent who behaves like the population's average, maintaining and updating a mixed strategy over routes that coincides with the group's aggregate flow proportions. Each day, the LLM reviews the travel experience and flags routes with positive reinforcement that they hope to use more often, and an interpretable update rule then converts this judgment into s
    
[^390]: Cocoon：一种使用相关噪声进行差分隐私训练的系统架构

    Cocoon: A System Architecture for Differentially Private Training with Correlated Noises

    [https://arxiv.org/abs/2510.07304](https://arxiv.org/abs/2510.07304)

    Cocoon 提出了一种系统架构，通过在 CPU、GPU 和内存扩展模块之间分布式地存储与处理庞大的相关噪声历史，并对稀疏嵌入表进行优化，实现了高效的大规模模型差分隐私训练。

    

    机器学习（ML）模型会记忆并泄露训练数据，给数据所有者带来严重的隐私问题。采用差分隐私（DP）的训练算法作为解决方案正受到越来越多的关注。然而，这些算法在每次训练迭代中都会添加噪声，从而降低模型精度，限制了其在现实世界中的实际应用。为了提升精度，一类新的方法会添加经过精心设计的相关噪声，使噪声在各次迭代之间相互抵消。我们对这些新机制进行了广泛的特性分析研究，结果表明，当模型相对较大或使用相对于硬件容量而言较大的嵌入表时，这些机制会产生不可忽视的开销。基于这一分析，我们提出了 Cocoon，一个用于高效相关噪声训练的框架。Cocoon 在 CPU、GPU 和内存扩展模块之间存储和处理庞大的噪声历史记录，并针对稀疏嵌入表引入了优化措施。

    arXiv:2510.07304v2 Announce Type: replace-cross  Abstract: Machine learning (ML) models memorize and leak training data, causing serious privacy issues to data owners. Training algorithms with differential privacy (DP) have been gaining attention as a solution. However, these algorithms add noise at each training iteration and degrade accuracy, limiting their real-world adoption. To improve accuracy, a new family of approaches adds carefully designed correlated noises, so that noises cancel out each other across iterations. We performed an extensive characterization study of these new mechanisms and show they incur non-negligible overheads when the model is relatively large or uses large embedding tables compared to the hardware capacity. Motivated by the analysis, we propose Cocoon, a framework for efficient training with correlated noises. Cocoon stores and processes the large noise history across CPU, GPU, and memory extension module, introduces optimizations for sparse embedding ta
    
[^391]: EEGDM：基于潜在扩散模型的脑电表征学习

    EEGDM: Learning EEG Representation with Latent Diffusion Model

    [https://arxiv.org/abs/2508.20705](https://arxiv.org/abs/2508.20705)

    EEGDM提出了一种基于潜在扩散模型的自监督学习框架，通过生成式去噪过程学习脑电信号的全局时间模式与跨通道关系的紧凑表征，克服了掩码重建方法难以捕捉全局生成约束的局限。

    

    近年来，脑电（EEG）表征的自监督学习研究主要依赖于掩码重建方法，即训练模型恢复被随机掩蔽的信号片段。尽管掩码重建在建模局部依赖方面卓有成效，但其训练目标并不能促使模型捕捉刻画神经活动所必需的全局生成约束。为解决这一局限，我们提出了EEGDM，一种利用潜在扩散模型生成EEG信号作为训练目标的新型自监督框架。与掩码重建不同，基于扩散的生成过程将信号从噪声逐步去噪至真实形态，迫使模型捕捉整体时间模式和跨通道关系。具体而言，EEGDM引入了一个EEG编码器，将原始信号及其通道增强提取为紧凑的表征，该表征作为条件信息来引导扩散模型的生成过程。

    arXiv:2508.20705v4 Announce Type: replace-cross  Abstract: Recent advances in self-supervised learning for EEG representation have largely relied on masked reconstruction, where models are trained to recover randomly masked signal segments. While effective at modeling local dependencies, the training objective of masked reconstruction does not compel the model to capture global generative constraints essential for characterizing neural activity. To address this limitation, we propose EEGDM, a novel self-supervised framework that leverages latent diffusion models to generate EEG signals as an objective. Unlike masked reconstruction, diffusion-based generation progressively denoises signals from noise to realism, compelling the model to capture holistic temporal patterns and cross-channel relationships. Specifically, EEGDM incorporates an EEG encoder that distills raw signals and their channel augmentations into a compact representation, which serves as conditional information to guide t
    
[^392]: 基于语义关系条件化的多模态表示学习

    Multimodal Representation Learning Conditioned on Semantic Relations

    [https://arxiv.org/abs/2508.17497](https://arxiv.org/abs/2508.17497)

    提出了关系条件化多模态学习框架RCML，将自然语言描述的语义关系作为显式条件来学习多模态表示，使同一样本在不同关系下拥有不同表示，克服了CLIP等对比模型单一嵌入的局限。

    

    多模态表示学习的发展主要由CLIP等对比模型推动，这类模型通过对齐配对的图像-文本样本学习共享嵌入空间。尽管这类模型在通用表示学习方面十分有效，但它们通常为每个样本生成单一嵌入，并在不同的语义关系和上下文中重复使用该嵌入。然而，在许多实际应用中，样本之间的相关性本质上是依赖于关系的，不同的语义关系会强调多模态数据的不同方面。在本工作中，我们提出了关系条件化多模态学习（RCML）框架，该框架将语义关系视为多模态表示学习的显式条件。RCML不再生成与关系无关的嵌入，而是基于自然语言关系描述学习条件化表示，使同一样本能够在不同关系条件下获得不同的表示。

    arXiv:2508.17497v3 Announce Type: replace-cross  Abstract: Multimodal representation learning has been largely driven by contrastive models such as CLIP, which learn a shared embedding space by aligning paired image-text samples. While effective for general-purpose representation learning, such models typically produce a single embedding per sample that is reused across different semantic relations and contexts. However, in many real-world applications, relevance between samples is inherently relation-dependent, with different semantic relations emphasizing different aspects of multimodal data.   In this work, we propose Relation-Conditioned Multimodal Learning (RCML), a framework that treats semantic relations as explicit conditions of multimodal representation learning. Rather than producing relation-agnostic embeddings, RCML learns representations conditioned on natural-language relation descriptions, allowing the same sample to be represented differently under different relational 
    
[^393]: 通过随机化密钥选择缓解生成模型中的水印伪造

    Mitigating Watermark Forgery in Generative Models via Randomized Key Selection

    [https://arxiv.org/abs/2507.07871](https://arxiv.org/abs/2507.07871)

    该论文提出通过对每次查询随机化水印密钥选择的防御方案，使盲攻击者的伪造成功率存在与所收集样本数量无关的上限，且不进一步降低模型效用，从而有效缓解生成模型中的水印伪造攻击。

    

    水印技术使生成式AI提供商能够验证内容是否由其模型生成。水印是内容中的一种隐藏信号，可以使用秘密水印密钥来检测其存在。一个核心安全威胁是伪造攻击，即对手将提供商的水印插入到并非由该提供商生成的内容中，这可能损害其声誉并破坏用户信任。现有的防御方法通过向同一内容中嵌入使用多个密钥的多个水印来抵抗伪造，但这可能会降低模型效用。然而，当攻击者能够收集足够多的带水印样本时，伪造仍然是一种威胁。我们提出了一种防御方法，对于盲攻击者，在密钥对称且检测器结果独立的条件下，其伪造成功概率存在一个与样本数量无关的上限。我们的方案不会进一步降低模型效用。我们对每个查询随机化水印密钥的选择，并据此接受内容是否由模型生成。

    arXiv:2507.07871v5 Announce Type: replace-cross  Abstract: Watermarking enables GenAI providers to verify whether content was generated by their models. A watermark is a hidden signal in the content, whose presence can be detected using a secret watermark key. A core security threat are forgery attacks, where adversaries insert the provider's watermark into content \emph{not} produced by the provider, potentially damaging their reputation and undermining trust. Existing defenses resist forgery by embedding many watermarks with multiple keys into the same content, which can degrade model utility. However, forgery remains a threat when attackers can collect sufficiently many watermarked samples. We propose a defense with a sample-count-independent upper bound on forgery success for blind attackers, conditional on key-symmetric, independent detector outcomes. Our scheme does not further degrade model utility. We randomize the watermark key selection for each query and accept content as ge
    
[^394]: 基于冲突感知证据深度学习的鲁棒对抗量化

    Robust Adversarial Quantification via Conflict-Aware Evidential Deep Learning

    [https://arxiv.org/abs/2506.05937](https://arxiv.org/abs/2506.05937)

    提出轻量级后验不确定性量化方法 C-EDL，通过为输入生成多样的任务保持变换并量化表示分歧来校准不确定性，无需重新训练即可增强证据深度学习对对抗性和分布外输入的鲁棒性。

    

    深度学习模型的可靠性对于其在高风险应用中的部署至关重要，因为在这些应用中，分布外输入或对抗性输入可能导致严重的不良后果。证据深度学习是一种高效的不确定性量化范式，它将预测建模为单次前向传播所得到的狄利克雷分布。然而，EDL 对对抗性扰动的输入尤为脆弱，容易产生过度自信的错误。冲突感知证据深度学习（C-EDL）是一种轻量级的后验不确定性量化方法，能够缓解上述问题，在无需重新训练的情况下增强对抗鲁棒性和分布外鲁棒性。C-EDL 为每个输入生成多样的、保持任务特性的变换，并量化表示层面的分歧，以便在需要时校准不确定性估计。C-EDL 的冲突感知预测调整提高了对分布外样本和对抗性样本的检测能力，同时保持较高的分布内准确率和较低的（摘要在此处截断）

    arXiv:2506.05937v3 Announce Type: replace-cross  Abstract: Reliability of deep learning models is critical for deployment in high-stakes applications, where out-of-distribution or adversarial inputs may lead to detrimental outcomes. Evidential Deep Learning, an efficient paradigm for uncertainty quantification, models predictions as Dirichlet distributions of a single forward pass. However, EDL is particularly vulnerable to adversarially perturbed inputs, making overconfident errors. Conflict-aware Evidential Deep Learning~\mbox{(C-EDL)} is a lightweight post-hoc uncertainty quantification approach that mitigates these issues, enhancing adversarial and OOD robustness without retraining. C-EDL generates diverse, task-preserving transformations per input and quantifies representational disagreement to calibrate uncertainty estimates when needed. C-EDL's conflict-aware prediction adjustment improves detection of OOD and adversarial inputs, maintaining high in-distribution accuracy and low
    
[^395]: 评估大型语言模型的检索鲁棒性

    Evaluating the Retrieval Robustness of Large Language Models

    [https://arxiv.org/abs/2505.21870](https://arxiv.org/abs/2505.21870)

    该研究建立了一个包含1,891个样本的基准和三个鲁棒性指标，系统评估了11个大型语言模型在检索增强生成场景中的检索鲁棒性，重点考察RAG是否总是优于非RAG、更多检索文档是否总是有益以及文档顺序对结果的影响。

    

    检索增强生成（RAG）通常能提升大型语言模型（LLM）解决知识密集型任务的能力。但由于检索结果不完美以及模型利用检索内容的能力有限，RAG也可能导致性能下降。在这项工作中，我们评估了LLM在实际RAG设置下的鲁棒性（下文称为检索鲁棒性）。我们聚焦于三个研究问题：（1）RAG是否总是优于非RAG；（2）检索更多的文档是否总能带来更好的性能；（3）文档顺序是否会影响结果。为开展这项研究，我们建立了一个包含1,891个样本的基准测试集，涵盖三个任务类别中的五个数据集，每个样本均包含使用稀疏检索器和稠密检索器检索到的文档。我们引入了三个鲁棒性指标，分别对应上述三个研究问题。我们在11个LLM上进行的实验表明，模型总体上达到了较高的检索鲁棒性。

    arXiv:2505.21870v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) generally enhances large language models' (LLMs) ability to solve knowledge-intensive tasks. But RAG could also lead to performance degradation due to imperfect retrieval and the model's limited ability to leverage retrieved content. In this work, we evaluate the robustness of LLMs in practical RAG setups (henceforth retrieval robustness). We focus on three research questions: (1) whether RAG is always better than non-RAG; (2) whether more retrieved documents always lead to better performance; and (3) whether document order impacts results. To facilitate this study, we establish a benchmark of 1,891 samples spanning five datasets across three task categories, each with documents retrieved using both sparse and dense retrievers. We introduce three robustness metrics, each corresponding to one research question. Our experiments across 11 LLMs show that models achieve generally high retrieval r
    
[^396]: VTBench：评估用于自回归图像生成的视觉分词器

    VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation

    [https://arxiv.org/abs/2505.13439](https://arxiv.org/abs/2505.13439)

    VTBench是一个系统性评估自回归图像生成中视觉分词器性能的综合基准，通过图像重建、细节保留和文本保留三大核心任务，揭示了离散视觉分词器与连续VAE之间的性能差距。

    

    自回归（AR）模型最近在图像生成中展现出强大的性能，其中一个关键组件是将连续像素输入映射为离散token序列的视觉分词器（VT）。视觉分词器的质量在很大程度上决定了AR模型性能的上限。然而，目前的离散视觉分词器明显落后于连续变分自编码器（VAE），导致图像重建质量下降，细节和文本的保留效果不佳。现有的基准测试专注于端到端的生成质量，而未能单独评估视觉分词器的性能。为了填补这一空白，我们提出了VTBench，这是一个综合性基准，通过三大核心任务系统地评估视觉分词器：图像重建、细节保留和文本保留，并涵盖多样化的评估场景。我们使用一组指标系统地评估了最先进的视觉分词器，以衡量重建图像的质量。

    arXiv:2505.13439v2 Announce Type: replace-cross  Abstract: Autoregressive (AR) models have recently shown strong performance in image generation, where a critical component is the visual tokenizer (VT) that maps continuous pixel inputs to discrete token sequences. The quality of the VT largely defines the upper bound of AR model performance. However, current discrete VTs fall significantly behind continuous variational autoencoders (VAEs), leading to degraded image reconstructions and poor preservation of details and text. Existing benchmarks focus on end-to-end generation quality, without isolating VT performance. To address this gap, we introduce VTBench, a comprehensive benchmark that systematically evaluates VTs across three core tasks: Image Reconstruction, Detail Preservation, and Text Preservation, and covers a diverse range of evaluation scenarios. We systematically assess state-of-the-art VTs using a set of metrics to evaluate the quality of reconstructed images. Our findings 
    
[^397]: 优先并增强人类智能的混合推理系统

    Hybrid Reasoning Systems That Prioritize and Enhance Human Intelligence

    [https://arxiv.org/abs/2504.13477](https://arxiv.org/abs/2504.13477)

    本文提出了一个以人为中心的混合推理系统框架，通过融合增强人类推理的既定策略、重视结论前互动的AI设计方法以及将推理分解为可单独支持的模式，实现从数据分析到高级智慧的全方位人类推理能力增强。

    

    在加速变化的世界中，人们需要明智且适应性强的人类推理。将AI能力与人类指导相结合前景可期，但人类推理本身往往草率、短视且容易出错，而且目前尚无明确框架能够在多样化的任务中将人类推理策略与AI相结合。本文提出了一个以人为中心的混合推理系统框架，该框架能够激发并增强人类从细粒度数据分析到高级反思与智慧的多层次推理能力。该框架通过概念综合方法开发，融合了三个方面：（1）增强人类推理的既定策略，（2）倾向于在得出结论前进行参与互动而非直接生成结论的AI设计方法，以及（3）将推理视为一组独特的、可单独支持的模式。这一综合形成了一个独特的广谱推理增强框架。

    arXiv:2504.13477v3 Announce Type: replace-cross  Abstract: In a world of accelerating change, there is a need for wise and adaptive human reasoning. Integrating AI capabilities with human guidance offers promise, though human reasoning itself is often hasty, shortsighted, and error-prone, and no clear framework exists for combining human reasoning strategies with AI across diverse tasks. This article proposes a framework for human-centered hybrid reasoning systems that engage and enhance human reasoning abilities ranging from granular data analysis to high-level reflection and wisdom. The framework was developed through a conceptual synthesis combining: (1) established strategies for enhancing human reasoning, (2) AI design approaches that favor pre-conclusive engagement over the generation of conclusions, and (3) the treatment of reasoning as a collection of distinct, individually supportable modes. This synthesis produced a distinctive framework for broad-spectrum reasoning enhanceme
    
[^398]: 迈向跨维度与类别模型的统一音乐情感识别

    Towards Unified Music Emotion Recognition across Dimensional and Categorical Models

    [https://arxiv.org/abs/2502.03979](https://arxiv.org/abs/2502.03979)

    本文提出了一个融合类别与维度两种情感标签的统一多任务学习框架，通过结合音乐特征与MERT嵌入表示，并利用知识蒸馏将单数据集教师模型的知识迁移到学生模型，实现了跨多个数据集的音乐情感识别。

    

    音乐情感识别（MER）面临的最重大挑战之一在于，情感标签在不同数据集之间的情感表示上可能存在异质性，包括类别标签（如快乐、悲伤）与维度标签（如效价-唤醒度）。在本文中，我们提出了一个统一的多任务学习框架，将这两种类型的标签结合起来，从而能够在多个数据集上进行训练。该框架采用了一种有效的输入表示方法，将音乐特征（即调性与和弦）与MERT嵌入相结合。此外，我们采用知识蒸馏技术，将在各个数据集上单独训练的教师模型的知识迁移到学生模型中，从而增强其跨多个任务的泛化能力。为了验证我们提出的框架，我们在包括MTG-Jamendo、DEAM、PMEmo和EmoMusic在内的多个数据集上进行了广泛的实验。根据我们的实验……

    arXiv:2502.03979v3 Announce Type: replace-cross  Abstract: One of the most significant challenges in Music Emotion Recognition (MER) comes from the fact that emotion labels can be heterogeneous across datasets with regard to the emotion representation, including categorical (e.g., happy, sad) versus dimensional labels (e.g., valence-arousal). In this paper, we present a unified multitask learning framework that combines these two types of labels and is thus able to be trained on multiple datasets. This framework uses an effective input representation that combines musical features (i.e., key and chords) and MERT embeddings. Moreover, knowledge distillation is employed to transfer the knowledge of teacher models trained on individual datasets to a student model, enhancing its ability to generalize across multiple tasks. To validate our proposed framework, we conducted extensive experiments on a variety of datasets, including MTG-Jamendo, DEAM, PMEmo, and EmoMusic. According to our exper
    
[^399]: 正面、反面与AI的失误：大语言模型、随机性与人类判断

    Heads, Tails, and AI Fails: LLMs, Randomness, and Human Judgments

    [https://arxiv.org/abs/2406.00092](https://arxiv.org/abs/2406.00092)

    该研究发现大语言模型在模拟抛硬币时会再现并放大人类的随机性偏差（如过度交替、厌恶长连续序列），提高温度参数只能部分缓解而无法消除这些系统性失真。

    

    随机性对人类认知以及大语言模型所部署的众多应用都至关重要，然而基于概率的token生成并不意味着大语言模型能够产生无偏的随机序列。我们借助模拟抛硬币这一经典行为科学范式，研究当代大语言模型如何生成二元随机序列。通过单次抛掷、20次抛掷序列、n-gram统计、连续长度、交替率以及下一次抛掷的可预测性等多个维度，我们将模型输出与真实的伯努利基线以及已有研究的人类数据进行比较。我们发现，大语言模型再现了若干经典的人类随机性偏差，包括过度交替、对长连续序列的厌恶以及首次抛掷偏差，但往往会放大这些偏差或引入模型特有的失真。提高温度参数可以减少某些僵化的模式，但无法消除系统性结构。我们进一步通过提示框架实验和续写任务深入探究了过度交替现象。

    arXiv:2406.00092v2 Announce Type: replace  Abstract: Randomness is central to human cognition and to many applications in which large language models are deployed, yet probabilistic token generation does not imply that LLMs can produce unbiased random sequences. We study how contemporary LLMs generate binary random sequences using the classic behavioral-science paradigm of simulated coin flips. Across single flips, 20-flip sequences, n-gram statistics, run lengths, alternation rates, and next-flip predictability, we compare model outputs to both true Bernoulli baselines and human data from prior work. We find that LLMs reproduce several canonical human randomness biases, including over-alternation, aversion to long runs, and first-flip biases, but often amplify them or introduce model-specific distortions. Increasing temperature reduces some rigid patterns but does not eliminate systematic structure. We further investigate over-alternation through prompt-framing experiments, continuati
    
[^400]: VIDiff：基于扩散模型通过多模态指令进行视频转换

    VIDiff: Translating Videos via Multi-Modal Instructions with Diffusion Models

    [https://arxiv.org/abs/2311.18837](https://arxiv.org/abs/2311.18837)

    本文首次提出了视频指令扩散基础模型VIDiff，能够根据用户的多模态指令在几秒内完成视频编辑、转换和增强等多种理解与生成任务，并通过迭代自回归方法保证长视频编辑的一致性。

    

    扩散模型在图像和视频生成方面取得了显著成功。这激发了人们对视频编辑任务日益增长的兴趣，即根据提供的文本描述对视频进行编辑。然而，大多数现有方法只关注短片段的视频编辑，并且依赖耗时的调优或推理过程。我们首次提出了视频指令扩散模型，这是一个为广泛的视频任务设计的统一基础模型。这些任务既涵盖理解任务（如语言引导的视频对象分割），也包括生成任务（视频编辑和增强）。我们的模型可以根据用户指令在几秒钟内编辑并转换出期望的结果。此外，我们设计了一种迭代自回归方法，以确保长视频编辑和增强的一致性。我们为多样化的输入视频和书面指令提供了令人信服的生成结果，无论是从定性角度还是……（摘要在此处截断）

    arXiv:2311.18837v2 Announce Type: replace-cross  Abstract: Diffusion models have achieved significant success in image and video generation. This motivates a growing interest in video editing tasks, where videos are edited according to provided text descriptions. However, most existing approaches only focus on video editing for short clips and rely on time-consuming tuning or inference. We are the first to propose Video Instruction Diffusion (VIDiff), a unified foundation model designed for a wide range of video tasks. These tasks encompass both understanding tasks (such as language-guided video object segmentation) and generative tasks (video editing and enhancement). Our model can edit and translate the desired results within seconds based on user instructions. Moreover, we design an iterative auto-regressive method to ensure consistency in editing and enhancing long videos. We provide convincing generative results for diverse input videos and written instructions, both qualitatively
    
[^401]: 面向鲁棒且动态的机器人运动学习低频运动控制

    Learning Low-Frequency Motion Control for Robust and Dynamic Robot Locomotion

    [https://arxiv.org/abs/2209.14887](https://arxiv.org/abs/2209.14887)

    本文挑战了“提高控制频率以增强鲁棒性”的传统观念，证明基于强化学习的低频（低至8Hz）运动控制器可在真实四足机器人上实现鲁棒动态运动，且低频策略对执行延迟和动力学变化更不敏感，甚至无需动力学随机化即可完成仿真到现实的迁移。

    

    机器人运动控制通常以提高运动控制频率为目标，来最大化鲁棒性和响应性。我们挑战了这一直观观念，通过在真实的ANYmal C四足机器人上使用低至8Hz的学习型运动控制器，展示了鲁棒且动态的运动能力。机器人能够鲁棒且可重复地达到1.5 m/s的高航向速度、穿越不平坦地形，并抵抗意外的外部扰动。我们进一步对基于深度强化学习（RL）的运动控制策略进行了比较分析，这些策略在5Hz到200Hz的频率范围内进行训练和执行。我们表明，低频策略对执行延迟和系统动态变化的敏感度更低，以至于即使不进行任何动力学随机化或执行器建模，也能成功完成仿真到现实的迁移。我们通过一系列严格的实验验证了这一论点。

    arXiv:2209.14887v3 Announce Type: replace-cross  Abstract: Robotic locomotion is often approached with the goal of maximizing robustness and reactivity by increasing motion control frequency. We challenge this intuitive notion by demonstrating robust and dynamic locomotion with a learned motion controller executing at as low as 8 Hz on a real ANYmal C quadruped. The robot is able to robustly and repeatably achieve a high heading velocity of 1.5 m/s, traverse uneven terrain, and resist unexpected external perturbations. We further present a comparative analysis of deep reinforcement learning (RL) based motion control policies trained and executed at frequencies ranging from 5 Hz to 200 Hz. We show that low-frequency policies are less sensitive to actuation latencies and variations in system dynamics. This is to the extent that a successful sim-to-real transfer can be performed even without any dynamics randomization or actuation modeling. We support this claim through a set of rigorous 
    
[^402]: ETHER: 对于回顾性经验重演的紧密沟通对齐

    ETHER: Aligning Emergent Communication for Hindsight Experience Replay. (arXiv:2307.15494v1 [cs.CL])

    [http://arxiv.org/abs/2307.15494](http://arxiv.org/abs/2307.15494)

    本文提出了ETHER，通过对齐紧急沟通来解决回顾性经验重演中的问题，克服了先前架构依赖预设函数的限制，并提高了数据效率和性能。

    

    自然语言指令的跟随对于实现人工智能代理和人类之间的合作至关重要。自然语言条件下的强化学习代理展示了自然语言的特性，如组合性，能够提供学习复杂策略的强归纳偏好。先前的架构如HIGhER结合了语言条件与回顾性经验重演（HER）来处理稀疏奖励环境。然而，与HER类似，HIGhER依赖于一个预设的函数来提供反馈信号，指示哪种语言描述在哪种状态下有效。这种依赖于预设函数的限制限制了其应用。此外，HIGhER只利用成功的强化学习轨迹中包含的语言信息，从而影响了其最终性能和数据效率。没有早期成功轨迹，HIGhER并不比其构建于之上的DQN更好。在本文中，我们提出了紧密文本回顾性经验。

    Natural language instruction following is paramount to enable collaboration between artificial agents and human beings. Natural language-conditioned reinforcement learning (RL) agents have shown how natural languages' properties, such as compositionality, can provide a strong inductive bias to learn complex policies. Previous architectures like HIGhER combine the benefit of language-conditioning with Hindsight Experience Replay (HER) to deal with sparse rewards environments. Yet, like HER, HIGhER relies on an oracle predicate function to provide a feedback signal highlighting which linguistic description is valid for which state. This reliance on an oracle limits its application. Additionally, HIGhER only leverages the linguistic information contained in successful RL trajectories, thus hurting its final performance and data-efficiency. Without early successful trajectories, HIGhER is no better than DQN upon which it is built. In this paper, we propose the Emergent Textual Hindsight Ex
    

