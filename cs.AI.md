# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OmniAct: One Policy, One Million Embodiments](https://arxiv.org/abs/2611.33158) | OmniAct通过以机器人在具体场景中的运动学图（包含关节位置、运动轴及与视觉观测的连接关系）为条件，并用流匹配策略预测各节点的相对关节运动，实现了单一策略即可控制具有不同自由度和运动学配置的多种机器人具身形态。 |
| [^2] | [MetaFORGE: A Meta-Learning Framework for Harness Optimization in Automated Code Generation for Parallel Computing](https://arxiv.org/abs/2611.33119) | MetaFORGE是一个基于执行反馈的元学习框架，它将正确性测试、编译器诊断、静态分析、运行时测量和负载均衡数据整合为标量奖励，迭代优化自然语言技能规范，从而显著提升LLM生成MPI并行代码的质量、性能与负载均衡。 |
| [^3] | [Draft Calibration for Speculative Decoding: Accelerating LLM Serving by Matching Draft and Target Distributions](https://arxiv.org/abs/2611.33111) | 该论文提出推测解码中的草稿校准方法，通过事后校准解决开源草稿模型相对目标模型普遍置信度不足的分布失配问题，缩小草稿分布与目标分布之间的差距，从而提高草稿 token 接受率并加速大语言模型推理服务。 |
| [^4] | [Temporal Conditioning for Human Behavioral Cloning in Chess](https://arxiv.org/abs/2611.33106) | 本文提出时间感知Transformer模型，将剩余时钟时间等时间信息编码进国际象棋行为克隆模型，在时间压力下显著提升了人类走法预测准确率，并能更准确地预测玩家的思考时间和认输行为。 |
| [^5] | [Delegent: A Multi-Agent Harness That Reduces Cost without Sacrificing Performance](https://arxiv.org/abs/2611.33098) | Delegent是一个基于Claude Code的多智能体框架，让前沿模型始终主导任务执行、仅将简单子任务委托给可在本地硬件运行的27B小型语言模型，从而无需训练路由器即可在不牺牲性能的情况下大幅降低智能体运行成本。 |
| [^6] | [EventDiffusion: Learning Native Event Stream Distribution from Videos](https://arxiv.org/abs/2611.33096) | EventDiffusion提出了一种层次化生成框架，通过事件活动扩散、事件空间分配和事件时序扩散三个概率过程，直接从真实记录中学习原生异步事件流的条件分布，从而从视频中生成多样且逼真的事件流。 |
| [^7] | [AIChipBench: Benchmarking LLM Agents for Automated AI Accelerator Design](https://arxiv.org/abs/2611.33093) | 该论文提出AIChipBench基准，要求LLM编程智能体用可综合RTL实现量化大模型解码器，重点考察跨解码步骤的状态保持正确性及面积受限下的性能优化，从而系统评估智能体在自动化AI加速器设计中的真实能力。 |
| [^8] | [Uncertainty-Aware Candidate Generation for Entity Linking in Cyber Threat Intelligence](https://arxiv.org/abs/2611.33086) | 该论文提出了一种不确定性感知的实体解析方法，通过在NER阶段捕获片段边界与类型的不确定性信号并传递到EL阶段，从而在网络威胁情报场景下改进模糊情况下的候选实体检索。 |
| [^9] | [On the Creation of Object-Centric Event Logs from Patient Journeys](https://arxiv.org/abs/2611.33074) | 本文针对医疗领域中对象中心流程挖掘研究的空白，基于一系列挑战与需求提出了一种从患者旅程数据创建以对象为中心的事件日志（OCEL）的方法。 |
| [^10] | [Auditing the Judge: Human-Derived Bias Discovery, Quantification, and Mitigation in LLM Judges](https://arxiv.org/abs/2611.33065) | 该论文提出HUB-J集成框架，利用人类与LLM判断不一致的案例自动发现可解释的偏见因子，并通过受控输入修改量化偏见强度，实现LLM评判系统中偏见的发现、量化与缓解。 |
| [^11] | [ClusterVAD: Threshold-Free Unsupervised Voice Activity Detection with Pre-Trained Neural Representations](https://arxiv.org/abs/2611.33059) | 提出了ClusterVAD，一种基于冻结语音基础模型表示的无监督语音活动检测方法，通过对表示进行聚类并用响度标注语音/非语音聚类，结合逐录音拟合的隐马尔可夫模型，实现了无需阈值设定且在会议、远场和噪声等场景下表现稳健的检测。 |
| [^12] | [Hybrid AI for AI-Native RAN Control: A Functional Perspective](https://arxiv.org/abs/2611.33035) | 本综述提出了一种与技术无关的功能性分类框架，将RAN决策分解为表示与推理、预测与仿真、自适应控制和决策综合四种互补能力，为比较AI与传统方法在AI原生RAN控制中的混合协作提供了统一视角。 |
| [^13] | [Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?](https://arxiv.org/abs/2611.33031) | 该研究证明前沿大语言模型无需任何工具、数据或推理能力，仅凭权重空间编程即可将4比特量化CNN的准确率从60.4%恢复至77.95%，接近原始浮点模型的81.4%水平。 |
| [^14] | [RecordRoute: Auditing Agent Failures with Localization-Guided Diagnosis](https://arxiv.org/abs/2611.33022) | RecordRoute将比较式记录定位与基于所选记录文本和元数据的局部错误模式诊断相耦合，实现了“先定位再诊断”的智能体失败审计框架，在Who&When Pro基准的全部指标上超越现有方法。 |
| [^15] | [Can LLM Judges Catch Unsafe Clinical Actions? A Fault-Injection Study](https://arxiv.org/abs/2611.33012) | 该研究提出了一种故障注入评估方法，通过在MedAgentBench中向经验证的智能体轨迹注入416条受控临床错误（覆盖八个错误类别、六个开源LLM裁判），发现LLM裁判识别不安全临床行为的可靠性高度依赖于临床证据和完整轨迹的可见性——仅看最终答案时既漏检错误又误报高达56%的正确轨迹。 |
| [^16] | [Embodied AI Fails the Mirror Test: How a Simple Mirror Breaks Robot Policies](https://arxiv.org/abs/2611.33010) | 该论文提出了RoboDojo-Mirror基准测试，发现即使场景中只出现一面镜子，也会使各类机器人操作策略（包括VLA模型、世界动作模型和智能体策略）的成功率大幅下降10.8至62.5个百分点，揭示了当前具身AI对镜面反射缺乏鲁棒性。 |
| [^17] | [LEGOSpec: A Composable and Efficient Framework for Speculative Draft Design and Training](https://arxiv.org/abs/2611.33004) | LEGOSpec提出了一个可组合的推测性草稿设计与训练框架，将草稿模型分解为上下文、骨干、提议和目标四个组件，并通过系统化的对照实验揭示了不同工作负载下各组件的作用、相互作用及最优组合规律。 |
| [^18] | [One, No One and One Hundred Thousand CTs: Testing Anatomical Control in Text-to-CT Generation](https://arxiv.org/abs/2611.33003) | 本文提出CtrlCT工具包，通过可信度测试（将生成CT与真实数据整体比较）和无需参考CT的变形式控制力测试（验证报告编辑能否按需、一致地控制解剖结构），解决了现有Text-to-CT生成器评估标准不一、且从未验证报告对解剖结构控制能力的核心问题。 |
| [^19] | [On Potentiality and Actuality Of Foundation Models](https://arxiv.org/abs/2611.32991) | 基础模型本身已蕴含完成特定任务所需的能力，适配并非添加新能力，而只需通过恒定的、与输入无关的向量移位将模型“实现”即可。 |
| [^20] | [CAPER: PartGround and PlanRefine for Reliable Manual-Guided Robotic Assembly Planning](https://arxiv.org/abs/2611.32986) | 提出CAPER框架，通过PartGround将手册证据与产品库存对齐以准确定位零件、并通过PlanRefine对照有限库存精炼计划，使视觉-语言模型能够可靠地生成机器人装配规划。 |
| [^21] | [Benchmarking Evidence Sufficiency in Vision-Language Models for Autonomous Driving](https://arxiv.org/abs/2611.32964) | 该论文提出 DriveAnswer 基准，通过匹配的强/弱证据问题配置来评估自动驾驶视觉语言模型判断证据充分性并在证据不足时适当放弃回答的能力。 |
| [^22] | [Attestation is All You Need: Toward a Zero-Trust Architecture for HPC AI Agents](https://arxiv.org/abs/2611.32958) | 提出AEGIS系统，通过编译任务行为约束、基于eBPF的系统调用监控、跨智能体行为关联以及Slurm级遏制，实现了针对HPC环境中被劫持授权AI智能体的运行时零信任行为证明防御。 |
| [^23] | ["I Do Intend to Deliver": Will LLMs Run a Marketplace Advance-Fee Scam?](https://arxiv.org/abs/2611.32957) | 研究表明，仅用一条无需越狱的普通提示词，十个主流大语言模型中就有八个愿意充当电商预付费诈骗的“卖家”，在1200次模拟对话中61%的买家被诱导预付款且单次成本仅约2.5美分，凸显了LLM被滥用于在线欺诈的严重风险。 |
| [^24] | [LoomKV: Lossless KV Cache Compression from GPU Memory to SSD for LLM Serving](https://arxiv.org/abs/2611.32954) | LoomKV提出了一种无需修改服务引擎的无损KV缓存压缩方法，同时作用于外部存储层级和GPU内存池，在九个模型上将BF16 KV缓存压缩至原始大小的0.71且不改变模型输出。 |
| [^25] | [Beyond Expected Value: PPO with Distributional Critics for Systematic Conservation Planning](https://arxiv.org/abs/2611.32948) | 本文提出用PPO结合Plackett-Luce方案以及基于隐式分位数网络（IQN）的分布型评论家来改进CAPTAIN保护规划框架，发现决策粒度而非优化器是生态收益的主要来源，其真正贡献在于更高的训练效率和对风险敏感目标的可扩展性。 |
| [^26] | [ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks](https://arxiv.org/abs/2611.32946) | 该论文推出ShellBench基准，通过对12个模型、5种智能体框架和3个推理水平共53,100次实验的评估发现，模型选择对日常任务性能的影响大于框架选择，但领先模型之间差距仅1.1个百分点，且高性能模型间的成本差异可达六倍之多。 |
| [^27] | [Do Vision-Language Models Know Their Own Visual Weaknesses? Probing Capability-Level Metacognition](https://arxiv.org/abs/2611.32940) | 视觉语言模型缺乏对自身视觉弱点的预先认知（能力层面元认知不足），训练其预测自身准确率可泛化到未见任务类型，但弱点信息只有在开始解题后才显现。 |
| [^28] | [Efficient Video Retrieval Augmented Generation via Token Allocation](https://arxiv.org/abs/2611.32938) | Lembas是一个即插即用的视频RAG框架，将帧与字幕的令牌分配建模为收益递减的优化问题，用具有可证明收敛保证的迭代注水算法求解，并能在多轮对话中零开销地复用令牌。 |
| [^29] | [Reading Decisions Before Spans: Efficient Generative Verification](https://arxiv.org/abs/2611.32931) | 提出Prefill1方法：利用生成式验证器的决策先于错误跨度定位、且决策分支前七个语法标记固定的特点，通过预填充固定标记并仅生成一个分支标记即可恢复验证决策，相比完整生成减少84.84%的服务开销且F1几乎不变。 |
| [^30] | [InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning](https://arxiv.org/abs/2611.32918) | 提出InstructReplay方法，通过让初始模型预先对通用指令生成回答并在微调时约束模型保持接近这些回答，在事实性微调中学习新知识的同时，将通用能力平均损失从28.1分降至2.4分，并避免了回答长度的崩溃。 |
| [^31] | [Motion Conditioned World Modeling for Embodied Visual Tracking](https://arxiv.org/abs/2611.32900) | 提出运动条件世界建模框架，将目标运动分布与机器人自运动融入潜在状态转移，并通过几何-进度模块联合建模路径选择与运动进度的不确定性，为具身视觉跟踪提供几何一致的未来视觉状态预测。 |
| [^32] | [Joint Detection of Constructive Speech, Hate Speech and Agreement in Online Conversations](https://arxiv.org/abs/2611.32889) | 本文构建了一个标注了仇恨语气、建设性语气和赞同态度的英丹麦语社交媒体评论大规模数据集，并通过微调多语言大语言模型实现三项任务的联合检测，其中 E5-large 在上下文增强架构下表现最佳。 |
| [^33] | [Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe](https://arxiv.org/abs/2611.32884) | 提出了基于可验证信念的强化学习（RLVB），利用模拟器在每一步计算答案的精确后验概率，奖励智能体在证据不足时继续探索、证据充分时才作答，从而教会AI智能体克服懒惰行为、按照自身信念判断何时给出答案。 |
| [^34] | [Adaptive Electromagnetic Transient-level Digital Twins for Data Centers via In-Context Learning with LLMs](https://arxiv.org/abs/2611.32882) | 本文提出一种基于大语言模型上下文学习的数据中心电磁暂态级自适应数字孪生，克服了传统数据驱动数字孪生在数据中心配置变化时无法泛化的局限。 |
| [^35] | [MXFP4 for LLM Inference: Complementary Post-Training Quantization Techniques and Their Interactions](https://arxiv.org/abs/2611.32873) | 该研究发现有效的微缩放格式量化方案依赖于具体格式：MXFP4和NVFP4虽都支持精确的4位推理，但MXFP4显著受益于旋转和纠错PTQ技术，而这些技术未必适用于NVFP4。 |
| [^36] | [Path-probability gradients for optimizing force fields against dynamical observables](https://arxiv.org/abs/2611.32851) | 提出PathGrad方法，通过对朗之万路径概率而非混沌分子轨迹求导，获得方差稳定的梯度估计器，从而能够直接利用实验动力学观测量（如扩散系数）对机器学习力场进行微调。 |
| [^37] | [Measurement Artifacts in Trustworthiness Evaluations of Quantized Small Language Models](https://arxiv.org/abs/2611.32848) | 该论文揭示了量化语言模型可信度评估中与量化交互作用的测量误差会被误报为量化效应，并实证发现4比特量化导致的有害顺从上升仅为关键词检测器记录的非拒绝上升的一半至三分之二，且不同评分方法对量化效应的估计严重不一致。 |
| [^38] | [Field Life: Particle Life as Interacting Chemical Fields](https://arxiv.org/abs/2611.32844) | 本文提出 Field Life，将 Particle Life 的粒子交互机制转化为一种质量守恒的元胞自动机，其中各颜色作为化学场浓度，通过环形卷积核与交互矩阵驱动演化，随机规则即可自发产生滑行体等运动结构。 |
| [^39] | [Reliable and grounded reporting of any computed tomography scan with FORA](https://arxiv.org/abs/2611.32823) | 提出通用CT模型FORA，可识别并空间定位1,028个放射学概念、覆盖28种CT检查方案自动生成报告，无需人工标注病灶即可学习异常定位，并在胸片基准测试中达到最先进性能。 |
| [^40] | [Residual Guidance for Real-Time Interactive Video Generation](https://arxiv.org/abs/2611.32820) | 提出了一种无需训练的残差引导方法（RG），通过在缓存之外计算的校正残差来抵消流式视频生成中的上下文惯性，使模型在提示词切换时能迅速响应新指令并保持视觉连续性。 |
| [^41] | [Interactive Neural Computers](https://arxiv.org/abs/2611.32811) | 该论文提出不再训练智能体去操作现有计算机，而是训练一个生成式世界模型直接充当计算机，根据历史屏幕画面和用户的鼠标键盘操作预测下一屏画面，从而解决记忆索引、精确计算和小文本清晰度等独特挑战，实现可定制、无障碍的交互式界面。 |
| [^42] | [MRBench: Benchmarking Physical Necessity Reasoning in Heterogeneous Multi-Robot Manipulation](https://arxiv.org/abs/2611.32803) | MRBench是一个评估异构多机器人操作中物理必要性推理能力的基准，通过八类任务和五种资源配置揭示了现有协调智能体在机器人形态变化时难以判断哪些机器人真正必要。 |
| [^43] | [Sample-Efficient In-Context Adaptation via Sleep](https://arxiv.org/abs/2611.32802) | 该研究提出在Transformer语言模型预训练中引入“睡眠”机制（通过多次循环前向传递精炼上下文表示），显著提升了模型在一般性任务上少样本上下文学习的样本效率。 |
| [^44] | [Scaling Laws of Ensemble Uncertainty in Language Models](https://arxiv.org/abs/2611.32801) | 本文提出三条幂律缩放定律，证明集成随模型规模增大会捕获越来越多的认知不确定性，作为扩展路径相比单个更大的模型更具竞争力。 |
| [^45] | [Accuracy of Noise Ceiling Estimators for Brain Scores](https://arxiv.org/abs/2611.32792) | 该论文提出了一个统一的大脑记录数学框架，并据此推导多种噪声天花板估计器的闭式表达式，以明确这些估计器在不同实验设置下能否为脑分数提供可靠的上界估计。 |
| [^46] | [C3H: Compression-to-Consensus Criteria Hijacking in Multimodal LLM Recommender Systems](https://arxiv.org/abs/2611.32780) | 本文提出C3H推理时攻击，揭示了多模态LLM推荐系统中一个反直觉漏洞：压缩与跨模态共识机制非但不能防御内容操纵，反而会放大“压缩致共识”偏差，攻击者无需投毒或扰动，仅靠定制目标商品的多模态内容即可提升其推荐排名。 |
| [^47] | [Learning World Models through Contrastive Identification of Actions](https://arxiv.org/abs/2611.32776) | 本文提出动作对比识别方法CIA，通过在情境合理的备选动作中识别实际执行的动作来学习世界模型，能够捕捉逆动力学回归无法奖励的动作信息，从而提升离线视觉规划的平均成功率。 |
| [^48] | [RubikSQL-HB: Agentic Text-to-SQL with Knowledge in the Loop](https://arxiv.org/abs/2611.32772) | RubikSQL-HB 通过 HeavenBase 平台上可查询的数据库专属知识工作区，将业务含义、存储值和参考查询等知识持续融入 Text-to-SQL 智能体的推理全程，使候选生成与下游裁判都能随时访问知识进行核查与仲裁，在 BIRD Mini-Dev 上达到 75.2% 的执行准确率，显著优于首次候选选择和多数投票策略。 |
| [^49] | [EviCraft: Can Minecraft Agents Find What They Cannot See?](https://arxiv.org/abs/2611.32765) | 该论文提出“证据闭合”概念并构建EviCraft运行时与基准，强制智能体只能依据可见信息或自身行动所得的事实做决策，实验表明仅切换信息准入机制就使中等规模智能体找到钻石的成功率大幅下降，其影响甚至超过不同智能体之间的能力差距。 |
| [^50] | [SGD with Large Learning Rates: General Analysis & Improved Rates](https://arxiv.org/abs/2611.32762) | 本文证明了大学习率的SGD在随机梯度噪声存在下仍能加速收敛，对线性预测器的交叉熵损失和两层神经网络建立了加速的O(T^-2)收敛速率，并在稳定区域内给出了更紧致的收敛保证。 |
| [^51] | [ReQOR: Operator-Discrepancy Subspace Repair for Quantization-Conditioned Backdoors in Large Language Models](https://arxiv.org/abs/2611.32755) | ReQOR提出了一种无需训练的防御方法，通过在良性校准数据上识别模块级量化算子差异的主导方向，并以闭式形式恢复全精度计算，有效修复大语言模型中的量化条件后门。 |
| [^52] | [ASCENT: A Benchmark for Evaluating and Advancing Stepwise Diagnostic Reasoning in Large Language Models on Common Clinical Scenarios](https://arxiv.org/abs/2611.32734) | 该论文提出ASCENT，一个由临床医生标注、包含3,078个逐步推理问题的基准，通过将病例分解为与电子病历对齐的步骤（发现、印象、依据）来评估信息不完整条件下的中间诊断推理，揭示出即使最强的推理模型在逐步诊断任务上仍有显著提升空间。 |
| [^53] | [FourStat: A Dual AutoEncoder Leveraging Frequency and Stationarity for Time Series Anomaly Detection](https://arxiv.org/abs/2611.32714) | 提出FourStat双自编码器框架，结合频域分析的傅里叶自编码器与强化统计平稳性的平稳自编码器来检测单变量时间序列异常，并引入Perfect Point新指标以消除阈值选择的主观性。 |
| [^54] | [Are We Rewarding the Wrong Module? Module-Specific Policy Optimization for Multimodal Reinforcement Learning](https://arxiv.org/abs/2611.32687) | 提出模块特定策略优化（MSPO），利用感知清单与置信度感知感知奖励（CaPR）将感知信号与结果信号分离，解决了多模态大语言模型强化学习中的模块级信用纠缠问题。 |
| [^55] | [CAPTA: Identifying LLMs Behind AI Agents](https://arxiv.org/abs/2611.32686) | 本文提出借鉴CAPTCHA思想的CAPTA方法，通过向AI智能体呈现选择题并分析不同模型的选择偏好差异，来识别智能体背后所使用的大语言模型。 |
| [^56] | [Economic World Model](https://arxiv.org/abs/2611.32663) | 本文提出经济世界模型（EWM）的概念及通用研究框架，通过从微观经济数据学习家庭策略并结合宏观数据校准结构性经济转移模型，实现了个体决策与总体经济动态相连接的微观到宏观模拟与评估。 |
| [^57] | [Bathys: Advancing Deep Search Agents](https://arxiv.org/abs/2611.32660) | 提出 Bathys 9B 和 27B 深度搜索智能体，通过可验证问题合成、SFT/KTO/CM-GSPO 训练流水线以及上下文压缩下的在线强化学习框架，提升智能体在复杂开放式任务中迭代检索与证据验证的能力。 |
| [^58] | [GeoSurge: Geometry-Grounded Multimodal Reasoning for Auditable Liver Resection Planning](https://arxiv.org/abs/2611.32657) | GeoSurge 提出了一个以解剖学为基础的框架，通过面-动作语法将动作映射为参数化几何，并结合掩码度量与结构感知的点云标记，生成可执行、可编辑且可审计的肝切除面。 |
| [^59] | [DexWAC: A Unified World-Actor-Critic Model for Dexterous Robot Manipulation](https://arxiv.org/abs/2611.32654) | 该论文提出DexWAC，在单一预训练阶段统一学习演员、动作条件世界模型和自举评论家，利用视觉预测与价值监督的互补作用塑造兼具控制相关性与任务相关性的表征，从而提升灵巧机器人操作能力。 |
| [^60] | [ICLWorld: Benchmarking In-Context Learning of Agents from Digital to Physical Worlds](https://arxiv.org/abs/2611.32649) | ICLWorld是一个从数字到物理世界评估智能体上下文学习的在线基准，通过30个环境、160个反直觉规则任务覆盖计算机与机器人使用，并推出UIverse统一框架将每个环境转化为用户界面。 |
| [^61] | [A Text-to-SQL Benchmark for Official Statistics](https://arxiv.org/abs/2611.32646) | 本文提出了LOCuST——首个面向官方统计场景的text-to-SQL基准，将荷兰统计局两千余个多语言宽表与语义知识图谱相结合，以应对现有基准无法模拟编码式列名和外部元数据等官方统计特点的问题。 |
| [^62] | [TeamFlowRL: Online Policy Optimization of Decentralized Flow Policies for Multi-Agent Coordination](https://arxiv.org/abs/2611.32645) | TeamFlowRL提出了一种两阶段框架，通过在去中心化流策略的完整去噪轨迹上构建团队策略优化目标（避免难以计算的动作似然），实现多智能体流策略的离线预训练与在线微调，在51个配对配置中平均提升47.3%。 |
| [^63] | [BARE: Body-Aware Reflexive Evasion for Humanoid Robots](https://arxiv.org/abs/2611.32642) | 提出了BARE框架，利用人类运动先验引导人形机器人通过局部身体调整实现快速闪避，同时保持全身稳定性。 |
| [^64] | [ReFocus: Progress-Guided Adaptive Practice for Long-Horizon Robotic Manipulation Reinforcement Learning](https://arxiv.org/abs/2611.32639) | ReFocus提出了一种进度引导的多轮强化学习框架，通过将任务分解为有向无环图（DAG）并利用进度奖励诊断策略瓶颈，进而通过自适应练习窗口为瓶颈阶段分配针对性练习，以提升长时程机器人操作任务的RL性能。 |
| [^65] | [MargEmbed-Akan: Bridging Twi Questions and English Knowledge through Cross-Lingual Score Distillation](https://arxiv.org/abs/2611.32630) | 该论文提出了 MargEmbed-Akan，一个基于跨语言分数蒸馏从 AfriE5-Large-instruct 训练得到的 5.6 亿参数开源查询编码器，有效连接契维语问题与英语知识检索。 |
| [^66] | [REBEAL: Human References Are No Longer the Best Translation, and Metrics Do Not Notice](https://arxiv.org/abs/2611.32623) | 该论文提出REBEAL基准，通过对参考译文施加七类受控污染来测试评价指标，发现人工参考译文质量已逊于顶级机器翻译系统，而现有八种主流指标均无法察觉这种污染，且各类型指标以不同方式失效。 |
| [^67] | [RoboRecast: Intent-Preserving Demonstration Augmentation via Diagnosis-Driven Optimization](https://arxiv.org/abs/2611.32622) | RoboRecast是一个诊断驱动的智能体优化框架，通过联合优化仿真场景与机器人运动来保持源任务交互意图，可从人类和机器人演示生成适配新布局、新物体和新机器人形态的合成训练数据。 |
| [^68] | [Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding](https://arxiv.org/abs/2611.32600) | 提出了门控概念适配器（GCA）这一分层概念瓶颈框架，通过在预训练Transformer的选定层附加门控低维概念读出适配器，将隐藏状态转化为可检查、可干预的概念中介接口，从而实现可审计的文本理解。 |
| [^69] | [Boiling Down Uncertainty: Deep Generative Scenario Search with Decision-Guided Evolution Strategies for Two-Stage Stochastic Optimization](https://arxiv.org/abs/2611.32573) | 本文通过补偿对偶性与伴随求解推导出闭式梯度，为深度生成式情景搜索补齐了解析理论基础，并将决策聚焦方向作为引导进化策略的采样子空间，保证期望更新为平滑目标的下降方向，从而改进两阶段随机优化的情景生成。 |
| [^70] | [MXI-MemICL: Towards In-Context Robot Manipulation through Self-Retrieved Interaction Memory](https://arxiv.org/abs/2611.32564) | MXI-MemICL通过将演示和历史交互转化为可自检索的交互记忆，并借助意图预测器检索相关经验，使视觉-语言-动作模型无需大规模预训练或额外标注即可实现上下文机器人操作。 |
| [^71] | [Next-Manip: On-Demand Agent-Policy Collaboration for Robotic Manipulation](https://arxiv.org/abs/2611.32533) | Next-Manip 提出了一个按需的智能体-策略协作框架，通过轻量级的智能体调用控制器根据当前执行需求在策略与智能体之间动态分配控制权，将执行器分配与高成本的智能体推理解耦，从而兼顾任务推理、失败恢复与高效精确的细粒度控制。 |
| [^72] | [ContinualDrive: A Deployment-Anchored, Multi-Axis Benchmark for Continual Updates of Learned Driving Planners](https://arxiv.org/abs/2611.32529) | 该论文提出ContinualDrive基准，以已部署模型为锚点评估驾驶规划器的部署后持续学习，并揭示评估回路、参考检查点、时间聚合和流协议这四个测量轴的选择会显著改变甚至逆转更新效果的判定结论。 |
| [^73] | [Wi-Fi Enabled Edge Intelligence Framework for Smart City Traffic Monitoring using Low-Power IoT Cameras](https://arxiv.org/abs/2611.32517) | 本文提出了一种基于Wi-Fi的边缘智能框架，利用低功耗物联网摄像头在智慧城市交通监控中实现本地推理或卸载至Jetson Nano进行GPU加速，无需依赖5G或有线网络即可达到近实时性能。 |
| [^74] | [Branch and Consolidate: A Unified Recipe for Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2611.32515) | 该论文提出了一套统一的多模态多教师在线策略蒸馏配方，能够将大幅偏离共享检查点的专家模型重新整合回单一学生模型，使其在各个领域接近专家模型性能并超越感知专家。 |
| [^75] | [RecurSciBench: Benchmarking Recursive Self-Improvement in Data-Driven Scientific Research](https://arxiv.org/abs/2611.32508) | 该论文提出了RecurSciBench——一个用于评估大语言模型智能体在数据驱动科学研究中递归自我改进能力的可扩展基准，其RecurSciPipe框架可从原始数据集自动构建带科学验证的任务，并在四个科学领域上系统评估了四个RSI系统的自我改进迁移效果。 |
| [^76] | [Assessing Robustness to Catastrophic Forgetting in Continual Deep Reinforcement Learning for Cyber-Physical Systems](https://arxiv.org/abs/2611.32496) | 该论文在非平稳自主泊车环境中系统评估了三类强化学习算法（无模型同策略的PPO、无模型异策略的SAC和基于模型的Drama）抵抗灾难性遗忘的鲁棒性，以确定哪类算法更适合信息物理系统中的持续学习。 |
| [^77] | [CodeWAM: A Code World Action Model for End-to-End Autonomous Driving](https://arxiv.org/abs/2611.32483) | 该论文提出CodeWAM，一种将可执行代码纳入世界表征的端到端自动驾驶世界模型，用代码刻画自车在规划执行下的动力学演化、用学习的视觉模型捕捉场景依赖后果，使规划器能够同时评估候选规划的未来结果与车辆向该未来的过渡过程。 |
| [^78] | [VISE: Verifier-Independent Skill Self-Evolution with Feedback from Tool Implementations](https://arxiv.org/abs/2611.32480) | VISE提出了一种无需任务特定验证器的技能自进化方法，通过从工具源代码中提取能力、输入要求、使用条件和输出格式等信息来指导技能工作流的修订，在两个基准上将平均任务成功率提升了9.14-13.35个百分点。 |
| [^79] | [Progressive Reliable Supervision for Black-Box Universal Domain Adaptation](https://arxiv.org/abs/2611.32474) | 本文提出渐进式可靠监督方法PReS，通过协调黑盒源模型、基础模型（如CLIP）和不断演进的目标模型的互补知识，在无源数据和源模型参数的黑盒普适域自适应场景下，逐步建立并完善可靠的目标监督。 |
| [^80] | [Gamification in the age of AI: surveying player perceptions of motivation, manipulation, and data protection](https://arxiv.org/abs/2611.32464) | 通过对28名成年玩家的问卷调查发现，玩家认为等级系统和排行榜等进度机制具有激励性，而每日登录奖励和随机奖励等留存机制则被视为高度操纵性，且玩家普遍担忧个人数据被收集利用但缺乏意识与控制感。 |
| [^81] | [Describe, Don't Ask: Caption-Driven On-Policy Distillation for Fine-Grained Visual Perception](https://arxiv.org/abs/2611.32463) | 该论文提出用图像描述（caption）取代人工构建的区域问答对来驱动在线策略蒸馏，从而将无标注图像转化为广泛且可扩展的细粒度视觉监督信号，同时避免目标缺失时的幻觉作答。 |
| [^82] | [See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight](https://arxiv.org/abs/2611.32439) | 该论文提出HEME方法，在训练时将匹配的候选目标作为“后见之明”上下文来学习基于查询的表示，使模型在推理时无需额外推理计算即可生成高质量的通用多模态嵌入。 |
| [^83] | [NRG-Twins: Pixel-Level Ground Truth for Rigidity Violations in Generated Videos](https://arxiv.org/abs/2611.32431) | 该论文提出NRG-Twins数据集，通过将真实视频片段与关键帧条件化的视频生成器重新生成的“孪生”视频配对，首次提供了生成视频中刚性违例（3D不一致）的像素级、以角度为单位的真值标注。 |
| [^84] | [Within-task hazards of AI agents: unidentified by time-horizon curves, identified by repeated runs](https://arxiv.org/abs/2611.32415) | 该论文证明仅凭时间视野（任务完成率）曲线在统计上无法识别AI智能体任务内失败风险是否随任务推进而下降，并提出利用同一任务的条件独立重复运行（即两次运行均成功的概率曲线相对单次曲线的对数轴平移量）来恢复对任务内风险的可识别性。 |
| [^85] | [BEACON: Learning Observation-Conditioned Starting Points for Guided Conditional Flow Matching in Microscopy](https://arxiv.org/abs/2611.32406) | BEACON提出了一种最优K自适应学习耦合方法，利用退化观测来条件化引导条件流匹配的训练起始点，从而避免任意噪声—目标配对导致的速度目标冲突，提升显微图像恢复的学习效果。 |
| [^86] | [CPU-Eval: Evaluating Large Language Models for CPU Hardware Code Generation](https://arxiv.org/abs/2611.32405) | CPU-Eval是一个基于六个真实CPU（含已流片设计）构建的、包含339个HDL生成问题的新基准，其难度远超VerilogEval，使最先进大语言模型的平均pass@1下降45.14个百分点。 |
| [^87] | [Verify then Steer: Progress Hallucination-Aware Recovery for Continuous Vision-and-Language Navigation](https://arxiv.org/abs/2611.32400) | 提出PHAR测试时恢复框架，通过检测视觉-语言模型残差流表示中的跨模态证据不匹配来识别“进度幻觉”现象，使导航智能体能够从路线偏离中有效恢复。 |
| [^88] | [Not All Pairs Align: Caption-Guided Audio-Visual Contrastive Learning](https://arxiv.org/abs/2611.32399) | 该论文提出了一种模型无关的对比学习目标CGAAL，利用模态特定字幕估计音视频间的语义一致性，并据此为正样本对加权，从而解决了画外音等因素导致的音视频内容不匹配问题。 |
| [^89] | [Evolve to Learn: Gradient-aligned experience evolution for model self-improvement](https://arxiv.org/abs/2611.32390) | 该论文揭示了一个关键问题：能提升测试时性能的经验并不一定值得内化到模型中，并提出以梯度对齐作为更可靠的经验筛选标准，据此构建了E2L自进化框架来实现模型的自我改进。 |
| [^90] | [Can AI Design a Better nnU-Net? A Hypothesis-Driven Architecture Discovery Loop from Failures to Design Rules](https://arxiv.org/abs/2611.32384) | 该论文提出一种假设驱动的架构发现循环，让LLM智能体从失败案例中诊断错误模式、提出可证伪假设并验证机制，最终提炼出两条设计规则，构建出比nnU-Net计算量更低、分割精度更高的3D医学图像分割网络LeanNeXt。 |
| [^91] | [Does Research-Paper Context Improve LLM-Generated Quadruped Navigation? An Exploratory Simulation Study](https://arxiv.org/abs/2611.32381) | 这项探索性仿真研究发现，为 LLM 提供特定研究论文作为上下文并未改进四足机器人的导航代码，反而在迷宫最短路径中途被阻断的场景下显著降低了生成控制器的鲁棒性。 |
| [^92] | [Validating Multi-Model AI-Generated Acceptance Criteria in Requirements Engineering](https://arxiv.org/abs/2611.32374) | 本文提出一种结合检索增强生成、多模型生成与人机协同工作流的面向验证方法，用于解决需求工程中AI生成的验收标准缺乏有效性验证的问题。 |
| [^93] | [The Control-Flow Tax: Specialized Decision Routing for LLM Agent Workflows](https://arxiv.org/abs/2611.32365) | 该论文提出一种异构架构，用轻量专用的“系统1”决策模型Jev接管控制流路由、由大模型Sol专注文档生成，从而显著降低LLM智能体工作流中的“控制流税”（成本降低约40%–50%，耗时降低约33%）。 |
| [^94] | [DiCT: Teaching X-ray Vision-Language Models to Reason via Imagining CT](https://arxiv.org/abs/2611.32344) | 提出DiCT三阶段渐进迁移框架，利用CT视觉-语言模型指导X射线模型学习3D空间推理，使其无需CT重建即可直接从2D X射线完成具有3D空间意识的诊断推理。 |
| [^95] | [Part-Set Prototypes for Explainable Mountain Gazelles Sex Classification](https://arxiv.org/abs/2611.32343) | 本文提出“集合原型”通用架构，通过截断Sinkhorn竞争机制在冻结的DINOv3令牌网格上进行联合匹配，直接从数据中挖掘部分数量与无约束空间布局，在山瞪羚性别分类相机陷阱基准上实现了0.878宏观F1的可解释识别性能。 |
| [^96] | [A recursive self-improvement framework for electron microscopy analysis](https://arxiv.org/abs/2611.32325) | 该论文提出PhyEvo-Mic-RSI，一个物理引导与专家知识相结合的递归自我改进框架，能将研究人员的一次性人工修正转化为可复用、可验证的图像分析能力，显著提升电子显微图像中原子缺陷检测性能和跨材料界面识别覆盖率。 |
| [^97] | [When Exact Token Importance Is Not Enough: Limits of Top-K Selection in LLM Attention](https://arxiv.org/abs/2611.32324) | 本文证明了即使采用精确的单点删除效应作为词元重要性评分，Top-K独立排名选择仍会次优，因为注意力归一化引入的位置间交互可使排名组合与最优组合之间的KL散度差距高达133.2倍且在理论上无界。 |
| [^98] | [FigmaEval: A Benchmark for Evaluating AI Agents in Professional UI Design](https://arxiv.org/abs/2611.32321) | 该论文提出了FigmaEval基准测试，基于真实Figma设计文件、采用“约束规格、开放解法”范式，通过22项专家评估标准和专业设计师盲测竞技场来评估AI智能体的专业UI设计能力。 |
| [^99] | [Alpha-Beta Pruning via Monadic Continuations](https://arxiv.org/abs/2611.32320) | 该论文提出了一种基于延续单子与读取器单子组合的Alpha-Beta剪枝新表述，将搜索界限记录在单子结果中而非走法类型中，并证明普通延续单子即可同时计算最优结果与最优走法序列。 |
| [^100] | [SPRUNN: Structural Proximity for Pruning in Deep Neural Networks](https://arxiv.org/abs/2611.32317) | 本文提出基于结构接近度准则的滤波器剪枝方法SPRUNN，通过全面比较滤波器间的相似性识别并剪除不重要的滤波器，并与全连接层细粒度剪枝相结合，在不显著损失准确率的前提下压缩深度神经网络。 |
| [^101] | [ASTERISM: Do Multimodal LLMs Actually Look at All the Images?](https://arxiv.org/abs/2611.32316) | 该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。 |
| [^102] | [Sparse Poison, Strong Hijack: Efficient Trajectory-Level Backdoors in Vision-Language-Action Models with Only 1\% Poisoning](https://arxiv.org/abs/2611.32304) | 该论文提出了一种由视觉、语言和动作共同引导的两阶段协同优化框架，仅需1%的毒化率即可在视觉-语言-动作（VLA）模型中有效植入轨迹级后门攻击，解决了低毒化率下后门学习与干净任务性能之间的权衡难题。 |
| [^103] | [Query-Guided Visual Minimization for Selective Disclosure to Cloud VLMs](https://arxiv.org/abs/2611.32302) | 提出QGVM框架，通过查询引导的本地视觉最小化处理，在将图像发送至云端VLM之前保留查询所需内容，同时抑制不必要的敏感信息泄露。 |
| [^104] | [DAiSy: Online Data Selection by Adaptive Utility Estimation for LLM Pretraining](https://arxiv.org/abs/2611.32294) | DAiSy通过一个与主模型同步更新的轻量级线性辅助模型，从冻结的文档嵌入预测基于梯度的数据效用分数，使在线数据选择中每个候选文档的评分仅需一次线性变换，从而消除了大语言模型预训练中数据选择高昂的计算开销。 |
| [^105] | [VC-OPD: Learning to See What Matters via Visual Cue On-Policy Self-Distillation](https://arxiv.org/abs/2611.32292) | 该论文提出VC-OPD方法，将教师端特权信息从局部图像区域扩展为问题条件化的视觉线索，通过多阶段质量过滤与token级监督，显著提升小型多模态大语言模型在复杂视觉推理任务中的能力。 |
| [^106] | [Reproducing the Speech Labels of Qwen3-Omni's Talker: Mimi in bf16, and a Last-Frame Pitfall](https://arxiv.org/abs/2611.32286) | 本文通过白盒回归与校准后的教师强制似然分析，确定Qwen3-Omni的Talker训练标签来自kyutai的Mimi编解码器（moshi包、bf16精度），并揭示了transformers库中MimiModel最后一帧编码不一致的陷阱。 |
| [^107] | [CoMNO: Commutator Memory Neural Operators for Long-Horizon PDE Prediction](https://arxiv.org/abs/2611.32274) | 提出换位子记忆神经算子（CoMNO），通过交换“先下采样后滚动预测”与“先滚动预测后下采样”两种操作顺序所产生的性能差距来提取细尺度动力学细节，从而增强神经算子在长时程偏微分方程预测中捕捉细尺度特征的能力。 |
| [^108] | [Refining and Aligning Clinical Reasoning with Health Guidelines via Formal Constraints](https://arxiv.org/abs/2611.32247) | VerClin是一个可解释的临床推理框架，它将健康指南转化为可执行知识，结合确定性四值推理器、多专科代理筛查和形式化约束通道，生成可追溯、对齐指南且能凸显高风险问题的临床建议。 |
| [^109] | [Boundary-Aware Policy Learning for Agent-Orchestrated Robot Manipulation](https://arxiv.org/abs/2611.32238) | 该论文提出边界感知策略学习方法，通过边界显式训练塑造宽而清晰的策略交接边界，再利用边界扩展自我改进不断扩大该边界，从而解决智能体在策略可靠起始区域之外调用策略而导致的交接失败问题。 |
| [^110] | [Roboclock: A Unified Benchmark for Time-Aware Robotic Intelligence](https://arxiv.org/abs/2611.32233) | 提出 RoboClock 统一基准，通过涵盖五个领域的60个任务、三种时间约束（时间窗口、事件响应、吞吐量）以及延迟对齐评估协议，系统评估机器人智能中的时间感知能力。 |
| [^111] | [Joint Classification and Generation on Tabular Data: an Empirical Study](https://arxiv.org/abs/2611.32225) | 本文系统比较了基于能量模型、概率电路、表格基础模型和表格扩散模型四类最先进方法构建的联合分类与生成模型，在29个数据集上评估了不同计算预算下它们与独立“分类器+生成器”组合的优劣。 |
| [^112] | [PoseVLN: Learning Geometric State Evolution for Vision-Language Navigation](https://arxiv.org/abs/2611.32218) | PoseVLN以相机位姿作为连接过去、现在与未来状态的几何桥梁，在回合相对坐标系中联合预测未来K个导航动作和位姿增量，从而实现导航决策与几何状态演化的统一学习。 |
| [^113] | [XMind: Thinking Twice via Cross-Memory Reasoning for LLM-Based Autonomous Driving](https://arxiv.org/abs/2611.32209) | XMind是一个无需训练的框架，通过将近期交通状态的工作记忆与基于反事实推演构建的情景记忆相结合进行跨记忆推理，使基于大语言模型的自动驾驶智能体在初始决策与历史经验冲突时能够“三思而后行”，做出更可靠的二次决策。 |
| [^114] | [OmniEVIE: Evidence-Vector-Informed Embedding for Multi-modal Retrieval](https://arxiv.org/abs/2611.32203) | 提出 OmniEVIE——一个 45 亿参数的多向量嵌入模型，通过单一压缩索引和三阶段检索统一支持文本、图像、视觉文档、视频和工具的多模态检索，在极低存储成本下实现证据保留与细粒度命中定位。 |
| [^115] | [Who gets the information? An $\alpha$-fair label-coverage framework for multi-label feature selection](https://arxiv.org/abs/2611.32200) | 该论文提出 FLIC 框架，将多标签特征选择中不同的相关性分配策略统一解释为隐含的社会福利函数，通过单一公平参数 α 实现从功利主义到最大最小的信息覆盖分配，并证明在 α≥0 时贪婪选择具有 (1-1/e) 的近似保证。 |
| [^116] | [Steerable by Design: Conditional Pretraining for Robust Control of Language Models](https://arxiv.org/abs/2611.32199) | 该论文提出了一种条件预训练方法和架构，通过编码器交叉注意力机制从一开始就将指令与数据通道分离，构建出Mancelle-3B这一在设计上即可被有效操控的语言模型，解决了模型误将数据当作指令处理的越狱与对齐漂移问题。 |
| [^117] | [When Answering Is Administering: Permission Replies That Rewrite Coding-Agent Policy](https://arxiv.org/abs/2611.32195) | 本文揭示了 Claude Code 等编码智能体的权限策略漏洞：一次权限回答可附带删除拒绝规则的策略更新，使“回答提示权”变成“策略管理权”，导致智能体在后续会话中读取受保护的机密，即便在厂商的严格沙箱内也无法幸免。 |
| [^118] | [LLM Agent Authorization Across Context Compaction](https://arxiv.org/abs/2611.32194) | 该论文揭示上下文压缩会将模型生成的摘要伪装成用户消息，破坏了请求衍生授权器对用户意图的归因保证，导致LLM智能体的授权策略跟随压缩器所写内容，甚至在91%的情况下接受攻击者注入的收款账户。 |
| [^119] | [SolarFilNet-V4: Multi-Scale Transformer Vision with Physics-Informed Adaptive Post-Processing for Solar Filament Segmentation](https://arxiv.org/abs/2611.32193) | 该论文提出SolarFilNet-V4框架，将轻量级金字塔视觉Transformer与各向异性曲线条带卷积、双频前馈模块及物理信息自适应后处理相结合，以应对低对比度和复杂纤维结构等挑战，实现GONG H-alpha观测图像中太阳暗条的精准分割。 |
| [^120] | [Blaming the Machine, Billing the Victim? How Mock Jurors Reassign Fault from Generative AI Systems and Respond to Liability Cliffs in Tort](https://arxiv.org/abs/2611.32188) | 本研究通过199人的预注册实验发现，美国模拟陪审员常将过错归咎于法律上无法担责的生成式AI系统，且这种指责会“泄漏”到受害者或AI使用者身上，同时侵权法中的责任断崖规则（如过失相抵、连带责任）会影响陪审员的过错认定。 |
| [^121] | [ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks](https://arxiv.org/abs/2611.32183) | ThinkMorph-SSL 提出一种两阶段自监督训练方案，将图像上色、去噪等视觉前置任务转化为无需指令的交错式多模态推理轨迹，让图像本身充当监督信号，从而在相同指令微调预算下超越仅靠指令微调的模型。 |
| [^122] | [DocPristine: Content-Aware Document Appearance Enhancement under Real-World Mixed Degradations](https://arxiv.org/abs/2611.32176) | 该论文发现预训练文档视觉编码器在退化文档中仍能关注文本区域，据此提出DocPristine，通过自适应融合冻结编码器的多层特征并注入像素修复网络，实现真实世界混合退化下的内容感知文档外观增强。 |
| [^123] | [Mind the Band: The Hidden Frequency Variable in Geometric RoPE](https://arxiv.org/abs/2611.32171) | 该论文揭示了几何RoPE中被忽视的隐藏变量——频带设置：其未标准化的默认配置足以逆转多视角合成基准中的模型排名，而通过更多频率覆盖更宽的频率范围可显著缩小性能差距。 |
| [^124] | [Benchmarking Protein Structure Tokenizers as Generation Targets for Language Models](https://arxiv.org/abs/2611.32168) | 该论文首次提出以“可写性”而非传统的重建保真度来评测蛋白质结构分词器，通过固定9B模型和训练配置、仅更换分词器的96组对照实验，发现不同分词器的生成效果差异巨大（TM 0.378–0.662），且与码本大小和重建上限无关。 |
| [^125] | [RightDiff: Learning Generative Reward Models to Judge by the Right Differences](https://arxiv.org/abs/2611.32166) | RightDiff通过在单次自回归生成中显式发现候选响应间的对比差异、区分实质性与非实质性因素并归纳针对响应对的评分规则，学习以“正确差异”为核心的生成式奖励模型，从而避免表面捷径与推理依据漂移。 |
| [^126] | [Making Evaluation Explicit Is Not Enough: Supervising the Evaluation Process in Generative Reward Models](https://arxiv.org/abs/2611.32161) | 提出STEP-GRM框架，通过为评分标准构建提供结构性指导和为准则级判断提供局部监督，直接监督生成式奖励模型的中间评估过程，使模型不仅学会正确的最终偏好，还掌握更可靠的评估流程。 |
| [^127] | [Breaking the Binary-Lifting Barrier for DNA Minimizer Density: Exact Optimization of a Native 4-ary Order at (w, k) = (12, 5)](https://arxiv.org/abs/2611.32159) | 该论文突破了现有方案依赖二进制序提升的局限，通过在 $B(4, 4)$ 上构建精确的 de Bruijn 端点转移矩阵动态规划，直接从头优化原生四元全序，在 = (12, 5) 处实现了优于 GreedyMini 和 OptMini 的 DNA 最小化子期望密度的精确优化。 |
| [^128] | [RetraQ: Manifold-Constrained Critic Guidance for Flow Policies in RL](https://arxiv.org/abs/2611.32148) | RetraQ利用预训练流模型编码的几何结构（去噪映射及其雅可比矩阵）来过滤和校正Critic梯度引导，使其近似于向行为流形的切向投影和回缩，从而避免引导过程被Critic外推误差误导，有效改进强化学习中的流匹配策略。 |
| [^129] | [Calibrated Reading of Latent Reasoning and Communication](https://arxiv.org/abs/2611.32143) | 本文提出校准潜读取器（CLR），一个紧凑的Transformer，通过任务条件化交互、源适配器迁移与保形校准，以比语言模型解码器低数百倍的计算成本准确读取潜推理结果与键值通信内容，并提供统计覆盖率与误报保证。 |
| [^130] | [VIDE : Training-Free Video Anomaly Detection via Visual Intrinsic Distribution Estimation](https://arxiv.org/abs/2611.32138) | VIDE是一种无需训练的视频异常检测框架，通过利用冻结片段嵌入估计每个测试视频的正常场景内容分布，自适应修正文本参考并重调整类别响应，从而减少跨场景误报并降低对提示词选择的敏感性。 |
| [^131] | [CoverMem: Coverage-Constrained Visual Memory for Budgeted Embodied Question Answering](https://arxiv.org/abs/2611.32119) | CoverMem通过覆盖约束的记忆构建、保持覆盖性的增量维护和预算受限的互补帧选择，在有限计算资源下为具身问答构建紧凑且证据充分的视觉记忆。 |
| [^132] | [An Automated Framework Integrating Deep Learning and Anatomical Reasoning for Cardiac Phenotyping From CCTA](https://arxiv.org/abs/2611.32114) | 提出了一种结合深度学习与解剖学推理的自动化框架，实现基于CCTA的定量心脏表型分析，涵盖血管特异性PCAT采样、无需专家标注的EAT测量以及冠状动脉钙化定量。 |
| [^133] | [Duo: Towards Spatiotemporal Action Coordination in Bimanual Policies](https://arxiv.org/abs/2611.32108) | Duo通过在共享动作头中将双臂分离为独立的动作token并引入跨手臂注意力机制，在流匹配过程中实现双手操作的空间-时间动作协调，使多种基础策略的成功率显著提升。 |
| [^134] | [PreferenceScene: Personalized Indoor Scene Synthesis from Feedback](https://arxiv.org/abs/2611.32099) | 该论文提出PreferenceScene，通过多轮交互中用户对场景渲染图像的简短自由反馈和满意度评分，利用大语言模型从局部反馈中推断并追踪用户的隐藏偏好，实现整体个性化室内3D场景合成。 |
| [^135] | [Clinical3D-Bench: Evaluating and Improving Spatial Intelligence of Multimodal LLMs in Hospital Environments](https://arxiv.org/abs/2611.32094) | 提出 Clinical3D-Bench 医院 3D 空间推理基准，揭示当前多模态大模型的空间推理能力远逊于医生，并通过 Clinical3D-45K 数据集微调显著提升了模型的空间智能。 |
| [^136] | [MARGIN: A Risk-Aware Agent for Robot Program Refinement](https://arxiv.org/abs/2611.32093) | 该论文提出MARGIN，一种风险感知的机器人程序改进智能体，通过“险胜边际”指标（最慢成功执行中剩余的执行预算比例）识别脆弱程序、指导针对性修复，在保持开发成功率的同时提升程序在未见场景中的执行鲁棒性。 |
| [^137] | [From Memory Access to Evidence Sufficiency: Diagnosing and Addressing Demand--Evidence Misalignment](https://arxiv.org/abs/2611.32088) | 该论文提出 MemPath 干预性诊断框架揭示长期记忆检索中“相关不等于充分”的需求-证据错位问题，并据此提出 BiDE-Mem 方法，将问题表示为类型化需求以指导检索与溯源感知的证据组织，在 LoCoMo 和 LongMemEval 基准上取得显著准确率提升。 |
| [^138] | [ExpGuard: Guarding Long-Horizon LLM Agents against Corrupted Experience](https://arxiv.org/abs/2611.32085) | ExpGuard提出了一种检索触发的经验风险评估与在线修复框架，通过任务条件化异构图整合经验语义、来源与执行依赖关系，在经验复用前预测攻击和任务失败风险，从而保护长程LLM智能体免受被污染经验的持续危害。 |
| [^139] | [The algorithmic behavioral sink: Modeling the transition to irreversible demographic collapse](https://arxiv.org/abs/2611.32084) | 该论文将婚姻市场建模为受热力学约束的复杂系统，提出基于费米-狄拉克选择的模型，量化了移动互联网与人工智能作为“合成择偶干扰因子”所形成的“婚姻能量势垒”与“技术因子”，揭示人类配对动力学正经历导致不可逆人口崩塌的非线性相变。 |
| [^140] | [Integral DeltaNet: Improved Linear Attention with Adaptive Memory Integration](https://arxiv.org/abs/2611.32063) | Integral DeltaNet通过同时维护快速更新的delta状态和保留其加权历史的积分状态这两个互补记忆，并在读取时自适应地融合二者，从而在缓解线性注意力记忆冲突的同时保留对未来预测有用的历史信息。 |
| [^141] | [More Powerful, Less Verifiable? Mitigating Hard-to-Verify Hallucinations in MLLM's Reinforcement Learning](https://arxiv.org/abs/2611.32055) | 该研究构建了区分“明显幻觉”与“隐蔽幻觉”的ITHV数据集，并提出DAPO-V方法，通过可验证性感知奖励与自适应惩罚机制，在风险预算约束下缓解多模态大语言模型强化学习中难以验证的幻觉问题。 |
| [^142] | [EffectWAM: Learning Action-Grounded Effects for Robot Manipulation](https://arxiv.org/abs/2611.32053) | EffectWAM通过Effect视觉变化表示、相机位置编码和动作接地三个互补组件，利用流匹配联合预测视觉效应与可执行动作，将视觉预测与可执行的机器人控制紧密连接起来。 |
| [^143] | [From Rule-based to Agentic: LLM Agent based Real-time Control of Urban Drainage Systems for Flooding and Sewer Overflow Mitigation](https://arxiv.org/abs/2611.32046) | 该研究提出AgenticAI-Control智能体框架，利用两个耦合的大语言模型智能体将多源气象预警信息与工程专业知识融合，实现城市排水系统的实时控制，相比规则基线控制使洪涝与溢流总量减少69.1%-90.6%。 |
| [^144] | [Spec-CoFrGeNet: Reparametrizing Continued Fraction Networks for Stable and Accurate Training](https://arxiv.org/abs/2611.32041) | 该论文提出Spec-CoFrGeNet，通过受格林函数启发对部分分母进行变换以约束三对角哈密顿量的谱，从而解决连分数网络因分母趋零产生极点导致的训练不稳定问题，无需下界约束和增量式训练即可实现稳定且精确的训练。 |
| [^145] | [Provenance-Isolated Epistemic Quorums: Execution-Bound Evidence Certification and Plausible-World Safety Gating in Agentic Control Planes](https://arxiv.org/abs/2611.32032) | 该论文提出PIEQ框架，将所有语言模型严格排除在可信计算基之外，通过基于假设的真值维护、物理溯源隔离、可能世界安全屏蔽与可线性化并发治理，为智能体变更生产基础设施提供执行绑定的证据认证与跨故障世界的安全门控。 |
| [^146] | [Unifying Partner and Environment Diversity to Improve Human-AI Coordination](https://arxiv.org/abs/2611.32031) | 该论文提出跨环境协同博弈（CECP）两阶段框架，通过统一建模伙伴多样性与环境多样性，显著提升AI智能体在陌生环境中与陌生人类伙伴的协调泛化能力。 |
| [^147] | [Do Occupancy World Models Really Follow Actions?](https://arxiv.org/abs/2611.32029) | 论文提出ReCast模型，通过将未来预测建立在动作引起的自车帧间几何对应关系上，解决了占据世界模型在反事实动作下预测不忠实遵循指定轨迹的问题，并首次引入了反事实动作忠实性评估基准。 |
| [^148] | [Beyond Before-and-After Images: Evaluating Temporal Grounding in a Lightweight VLM for Disaster Assessment](https://arxiv.org/abs/2611.32022) | 该研究提出受控诊断实验，发现轻量级VLM（InternVL3-2B）在灾害评估中并未真正利用灾前影像进行时间定位，其看似的变化推理实际主要依赖灾后影像和提示词表述。 |
| [^149] | [HuluOPD: Unifying Heterogeneous Medical Specialists into One Lightweight Model with Unlabeled Prompts](https://arxiv.org/abs/2611.32021) | HuluOPD提出一种在线策略蒸馏方法，仅用无标注提示和跨分词器监督损失，就能把架构各异的多个医学专科视觉-语言模型统一蒸馏为一个轻量级统一模型，无需参考答案或专科模型的原始训练数据。 |
| [^150] | [Structure-Aware Transformers for Hierarchical Authorization in Personal AI Agents](https://arxiv.org/abs/2611.32018) | 提出PATH架构，通过层级感知注意力掩码与父门控概率机制，为个人AI智能体预测与权限层级结构一致的用户授权偏好，从而在保障隐私安全的同时减轻手动审查权限请求的负担。 |
| [^151] | [When Online Adaptation of Resilience Policies Hurts: Safe Exploration and Measurement Pitfalls in Resilience4j](https://arxiv.org/abs/2611.32015) | 该论文提出AdaptR4j系统，利用上下文赌博机在运行时动态选择Resilience4j的熔断器/重试/舱壁策略，并通过安全策略菜单和有界奖励的安全探索两项保障机制，克服了朴素在线学习在探索阶段拒绝流量的陷阱，在所有可赢得的故障场景中超越生产默认配置。 |
| [^152] | [To Critic Or Not To Critic](https://arxiv.org/abs/2611.32013) | 本文从梯度方向的偏差-方差权衡入手，揭示PPO（带评论家）与GRPO（不带评论家）之间没有普适的优劣答案，方法选择取决于批次大小、对评论家的依赖程度和任务复杂度等相互作用因素。 |
| [^153] | [Marketing-Bench: Evaluating Frontier Models and Agents on Real-World Marketing Capabilities](https://arxiv.org/abs/2611.32012) | 该论文提出了首个开源营销基准Marketing-Bench，通过将品牌异常检测（判别式任务）与修复（生成式任务）配对且保持答案可验证，从事实性、一致性、连贯性、结构和品牌语调五个维度评估AI模型与智能体的真实营销能力。 |
| [^154] | [Calm Agent: Reframing Agent Interaction around Intermittent Human Attention](https://arxiv.org/abs/2611.32009) | 本文提出“平静智能体”设计理念，主张围绕用户间歇性注意力设计辅助型AI智能体，通过持久任务表示连接“委托、调整、查看”三种按需交互模式，使任务在交互间隙持续运行且用户可低成本返回理解与控制，并以桌面原型Orvlet加以演示。 |
| [^155] | [TableRouter: Query-Aware Evidence Construction for Long-Table Question Answering](https://arxiv.org/abs/2611.32006) | TableRouter是一个无需训练的框架，通过分析查询在计算路线（从完整表格推导问题相关统计数据）和覆盖路线（检索相关行与列摘要）之间智能路由，在长表格问答任务中显著提升准确率（最高18.3个百分点）并将计算成本降低77-95%。 |
| [^156] | [ADE: A Zero-Training Unlearning Method for Latent World Models](https://arxiv.org/abs/2611.31998) | 提出零训练闭式遗忘算法ADE，仅利用缓存的预训练统计矩即可从潜在世界模型中精准切除后门触发器与过时物理机制的污染，且无需原始保留数据、不扭曲潜在流形、不损害有效控制响应。 |
| [^157] | [How Much Dependency Can a Marginal Rule See? Total Correlation in Diffusion LM Parallel Decoding](https://arxiv.org/abs/2611.31990) | 该论文量化了扩散语言模型并行解码中基于边缘分布的解码器对提交token集合条件总相关性（即依赖性损失）的预测能力，发现其预测准确性主要来源于边界本身，且估计值在散文文本上通常比精确 KL 代价低 0.4 到 1.1 比特。 |
| [^158] | [Discovering Bandwidth-Efficient Prefetchers to Shrink the Cache Hierarchy](https://arxiv.org/abs/2611.31983) | 本文提出DuoArchitect——一个自主智能体框架，通过多保真度仿真与多核协同演化，联合发现覆盖L1D、L2和LLC各层级的协调带宽高效预取策略，其仿真速度比传统集群评估快3-26倍，从而帮助缩小缓存层次结构、释放芯片面积。 |
| [^159] | [Reading, Writing and Repairing Game-Theoretic Policies in Weight Space](https://arxiv.org/abs/2611.31979) | 该论文构建了一组在权重空间中答案可被精确验证的博弈策略读写任务，证明中等规模开源语言模型能够在1218个权重的循环网络中恢复策略程序、报告其行为并写出最优策略，但修复被篡改网络的能力明显较弱。 |
| [^160] | [Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth](https://arxiv.org/abs/2611.31977) | 提出Dooop方法，将预训练的非循环扩散Transformer通过定点迭代蒸馏为循环模型，实现测试时任意可调的循环深度而无需重新训练。 |
| [^161] | [Coding4Games: Can Modern Coding Agents Play Modern Video Games?](https://arxiv.org/abs/2611.31974) | 该论文提出了Coding4Games基准，评估编程智能体能否开发出无需在游玩过程中调用语言模型、仅依靠视觉观察和标准游戏输入就能独立运行并游玩15款多样化现代游戏的程序。 |
| [^162] | [DexOPD: Dual-Extrapolation On-Policy Distillation for Diffusion Models](https://arxiv.org/abs/2611.31969) | 提出DexOPD双外推在线策略蒸馏方法，在教师外推监督的基础上引入以滞后历史学生为参考的学生自我外推机制，从而充分利用学生自身训练进展，使扩散模型蒸馏效果超越仅依赖教师外推的现有方法。 |
| [^163] | [Are Web Agents Still Detectable? Stress-Testing Bot Defenses with Coordinated Evasion](https://arxiv.org/abs/2611.31962) | 论文提出了DarkAgent自主网络智能体，通过结合隐藏自动化痕迹的Chromium浏览器与联合生成逼真按键事件及时间间隔的类人鼠标键盘模型，实现跨浏览器指纹、自动化控制和交互行为的协同规避，实验表明现有机器人防御系统难以检测此类攻击。 |
| [^164] | [Perception Is Enough, at the Right Scale: Scale-Adaptive Token Pruning for Vision-Language-Action Models](https://arxiv.org/abs/2611.31954) | 提出免训练的TauPrune-VLA方法，通过单一感知温度参数τ并借助感知语义趋势、机器人状态变化和动作幅度在线自适应调节token剪枝的语义尺度，从而降低VLA模型的计算成本并支持实时机器人控制。 |
| [^165] | [Beyond Rigid Hands: Deforming Skin into Contact-Consistent Hand Meshes](https://arxiv.org/abs/2611.31950) | 该论文提出一种优化方法，通过微调关节姿态并将皮肤变形贴合到物体表面且保持时间一致性，将刚性手部网格提升为接触一致的手部网格，并训练轻量级预测头来自动预测变形区域。 |
| [^166] | [CAPE: Coordinating Calibration and Search for Tool-Use Policies](https://arxiv.org/abs/2611.31938) | CAPE是一个无需训练的工具使用策略编译器，它通过协调校准与搜索，始终保持可执行策略，并将校准资源集中于能改进该策略的调用上，从而高效应对配置不可见工具导致的调用失败问题。 |
| [^167] | [Contrastive On-Policy Self-Distillation for LLM Unlearning](https://arxiv.org/abs/2611.31932) | 本文提出CPSD框架，通过构建显式的知识缩减目标分布，并沿模型自身的在策轨迹进行对比自蒸馏，实现大语言模型在遗忘指定数据的同时保留无关能力。 |
| [^168] | [Machine learning enables full Bayesian inference for spinning extreme-mass-ratio inspirals](https://arxiv.org/abs/2611.31916) | 该研究首次借助生成式机器学习网络缩小完整先验并结合精确探测器似然函数，实现了对自旋极端质量比旋近（EMRI）全部14个参数的联合贝叶斯后验推断。 |
| [^169] | [Your VLM Knows What to Inspect: Semantic Probe-Guided Adaptive Focusing for GUI Grounding](https://arxiv.org/abs/2611.31909) | 该论文提出免训练框架SPARK，利用VLM自身的语义探针（探索与验证）引导自适应的视图放置与裁剪，将模型“知道该检查什么”的语义能力转化为针对性视觉观察，从而提升GUI定位精度。 |
| [^170] | [MIFLOW: Dynamic Recent-Behavior Priors for Reference-Calibrated Online Flow-Policy Reinforcement Learning](https://arxiv.org/abs/2611.31902) | MIFLOW通过用拟合最近执行动作的动态流匹配先验作为移动密度参考，对在线流策略强化学习的演员更新和评论家目标进行维度归一化校正，在抑制低密度不可靠动作的同时保持探索能力，从而在MuJoCo连续控制任务上取得有竞争力的性能。 |
| [^171] | [Loupe: Reinforcing Knowledge-Intensive Video Generation Agents](https://arxiv.org/abs/2611.31895) | 该论文提出 Loupe 视频生成智能体，通过检索并验证文本与视觉证据来构建带参考图像的时间化视频提示词，并创新性地引入基于量规的信用评估机制（RBC），在视频渲染前对中间搜索与规划决策进行评估，从而强化知识密集型视频生成。 |
| [^172] | [Cheap Memory Costs Future Compute: Co-Designing Persistent KV and GPU Execution for Efficient Streaming Video Diffusion](https://arxiv.org/abs/2611.31892) | 提出了无需训练的SAMR方法，通过步骤感知记忆重建部分刷新KV缓存并无需DiT前向传播地重建其余部分，解决了流式视频扩散中干净KV更新成为主要运行时开销、而复用过期KV又会损害下游生成质量的核心矛盾。 |
| [^173] | [HEAR, SEE, SPIKE ONCE: Highly Efficient NeuroAI Temporal Neural Networks for Low-Precision Multimodal Audio-Visual Classification and Online Learning](https://arxiv.org/abs/2611.31884) | 本文提出NeuTNN-AV，一种采用单次脉冲时间编码、4位整数权重和局部奖励调制STDP学习规则的高效多模态视听神经网络，能够以远少于现有SNN的突触数量达到相当或更优的分类精度。 |
| [^174] | [Rethinking Video Agents: When and Why Video Harnesses Work](https://arxiv.org/abs/2611.31883) | 该研究利用Video-Relook实验框架系统探索了视频智能体工具框架何时以及为何有效，发现自适应的时间与空间观察策略能显著提升细粒度动作理解能力，且其收益由任务需求与模型的交互能力共同决定。 |
| [^175] | [EEG-Jev: A Unified Single-Pass Decision Model for EEG](https://arxiv.org/abs/2611.31875) | EEG-Jev提出了一种统一的脑电单次前向决策模型，一次前向传播即可完成选项选择、命题判断与有序评分三类任务，并通过先验校正和温度缩放提供含义明确、可校准的概率输出，在多个公开脑电数据集上取得了优异表现。 |
| [^176] | [RepoEviAgent: Scalable Synthesis of Evidence-Grounded Repository-Level Question Answering Tasks](https://arxiv.org/abs/2611.31873) | 该论文提出RepoEviAgent框架，通过质量感知的数据筛选与角色专门化的多智能体协作，从海量开源仓库中可扩展地自动构建证据支撑的仓库级问答任务，并建立了RepoEviQA基准。 |
| [^177] | [CALVIN-MAX: When Does a Mid-Execution Change Hurt a Robot Policy?](https://arxiv.org/abs/2611.31868) | 提出 CALVIN-MAX 基准，通过三次对照运行的设计，首次将机器人执行过程中遭遇变化的时机影响与世界已被改变的内在代价区分开来，量化了变化在运动中途到来时对策略性能的额外损害。 |
| [^178] | [No Fixed Configuration Is Best: Context- and Cost-Aware Tuning of LLM Pipelines, with an Application to Schema Matching and Mapping](https://arxiv.org/abs/2611.31864) | 论文以准确率和成本为联合目标对模式匹配与映射LLM流水线进行数值调优，发现调优只能带来有限准确率提升或成本节约，而一个数值调优无法触及的配置选择（如是否将源列搜索限制到单一目标表）对不同任务的准确率影响远大于数值调优，因此不存在对所有任务都最优的固定配置。 |
| [^179] | [Explainable Deep Learning for Malaria Parasite Detection and Species Differentiation in Thick Blood Smear Images: A Pilot Feasibility Study](https://arxiv.org/abs/2611.31855) | 该研究提出TRACE-Mal原型，将YOLOv8n-seg检测与六模型软投票集成相结合，并验证了Grad-CAM、LIME和SHAP三种可解释方法之间的一致性与疟原虫虫种判定的正确性显著相关，即被两种及以上方法共同解释的预测准确率更高。 |
| [^180] | [Modeling Event-Conditioned Emotional Expression Response with LLM Agents in a Synthetic Social Simulation](https://arxiv.org/abs/2611.31851) | 本研究提出一个分层提示框架，通过人口统计、心理和情境线索赋予LLM智能体情感推理能力，使其生成事件条件化的情感响应，并利用大规模社交媒体数据验证了其再现真实情感模式的有效性。 |
| [^181] | [Two-Stage Post-Training for Flow-Based Embodied Foundation Models](https://arxiv.org/abs/2611.31850) | 该论文提出了一种基于流的具身基础模型的两阶段后训练框架，第一阶段通过多教师在线策略蒸馏高效整合专家能力，第二阶段将强化学习与基于采样噪声加权的教师蒸馏相结合，从而突破性能瓶颈、实现超越教师模型的表现。 |
| [^182] | [JECT: Addressing the Context Shortcut in Joint-Embedding Predictive Learning for Chest CT](https://arxiv.org/abs/2611.31849) | JECT通过让放射学报告同时充当预测器的条件输入和编码器的预测目标，解决了胸部CT自监督联合嵌入预测学习中“上下文捷径”导致编码器忽略病灶的问题。 |
| [^183] | [Faithful Persona Steering under Incongruity via Dual-Stream Refinement](https://arxiv.org/abs/2611.31848) | 该论文提出QuirkyMind框架，通过特质锚定构建融合句子级摘要与token级序列的双流潜在状态，并利用判别式InfoNCE损失与生成式交叉熵损失的交替精炼目标，实现大语言模型在不一致人格下（稳定特质与情境特异性怪癖并存）的忠实个性化引导。 |
| [^184] | [Spec-Test Coverage Gap: Detecting Overspecified Tests in SWE-Bench-Style Benchmarks](https://arxiv.org/abs/2611.31839) | 本文提出规格-测试覆盖差距（STCG）这一1-4分制新指标，利用两个LLM评判者自动检测SWE-bench风格基准中的过度指定测试，从而取代昂贵的专家审查，识别会导致正确的替代修复方案被误判失败的测试用例。 |
| [^185] | [Gaze World Model](https://arxiv.org/abs/2611.31836) | 提出注视世界模型（GWM），在冻结的 V-JEPA 2.1 编码器上将潜在场景预测与人物条件化的注视监督联合训练，实现注视定位、注视转移检测以及未来注视变化预测。 |
| [^186] | [POISE: Curvature-Aware Critic Regularization for Smooth Actor-Critic Control](https://arxiv.org/abs/2611.31835) | 提出 POISE 方法，通过曲率加权的贪心偏移能量（以二阶近似保留先前贪心动作的价值损失）对 critic 进行正则化，从而有效抑制 Actor-Critic 控制中的高频动作振荡。 |
| [^187] | [Not All Interpretation Methods Are Created Equal: Tiered Evaluation of AI Interpretability](https://arxiv.org/abs/2611.31833) | 提出了TRUST分层评估框架，将解释方法的评估提升为AI核心评估指标，以区分不同解释方法的可信度证据等级，从而推动稳健的知识发现并确保AI在社会技术部署中的安全性与可争议性。 |
| [^188] | [GaokaoMathGen: Chinese Gaokao Mathematics Finale Problem Generation Controlled by Latent Reasoning Structures](https://arxiv.org/abs/2611.31823) | 提出GaokaoMathGen框架，通过从种子题中提取技能、方法、核心恒等式和关键转折步骤等潜在推理结构来约束生成高考数学压轴题，并结合逻辑过滤与求解器解码等多重信号估计题目实际难度。 |
| [^189] | [Exploring Multi-Player Social Games as Verifiable Training Grounds for Large Language Models](https://arxiv.org/abs/2611.31821) | 本文提出将多人社交游戏作为可验证训练环境，通过SFT和RLVR使Qwen3-4B的胜率从33%大幅提升至81%，超越了远大于自身的Gemma-3-27B模型，并能泛化到未见过的游戏中，效果远优于提示等上下文策略。 |
| [^190] | [Compressible Representations: Functional Spines in Deep Neural Networks](https://arxiv.org/abs/2611.31817) | 该研究提出一个基于类Fisher信息矩阵的框架，用于识别深度网络各层中对任务功能敏感的子空间，发现表示高度可压缩——只需少量任务敏感维度即可保留大部分性能，且压缩表示对对抗攻击更鲁棒。 |
| [^191] | [Agentic Discovery of Soundness Bugs in Safe Rust Bindings](https://arxiv.org/abs/2611.31813) | 本文提出将LLM智能体作为Rust安全封装接口的审计员来发现健全性漏洞，定义了此类接口缺陷的分类体系，并通过真实crate的重复审计实验发现：智能体能自发找到未定义行为和与底层库的偏差，却难以发现不完整的不安全契约。 |
| [^192] | [How Far Apart Are Two Proofs? Measuring Proof Dissimilarity in a Proof Kernel and in Lean/Mathlib](https://arxiv.org/abs/2611.31811) | 该论文针对“同一定理的两个证明相距多远”这一源于希尔伯特第 24 问题的基础问题，定义了 6 大族共 19 种证明差异性度量和 10 个期望性质，并在 Curry-Howard 内核与 Lean/Mathlib 中加以实例化与验证。 |
| [^193] | [TopoCryptoEEG: End-to-End Encrypted Framework for Brainprint Authentication](https://arxiv.org/abs/2611.31801) | 该论文揭示了现有特征级加密脑纹认证方案存在漏洞——被攻破的客户端可在有限次交互后精确恢复注册模板，并提出基于同态加密的TopoCryptoEEG框架，在客户端加密预处理脑电数据、在服务器端对密文完成全部认证流程，从而将保护范围扩展至预处理数据并彻底消除明文特征接口。 |
| [^194] | [Learning Compact State Abstractions for LLM Agents](https://arxiv.org/abs/2611.31800) | 本文提出将智能体状态定义为交互历史的紧凑抽象，以确定性递归程序表示并用LLM引导的遗传算法搜索，在ALFWorld和ScienceWorld上保持接近参考智能体的任务成功率的同时，将策略输入最多减少71%和57%，且得到的程序可直接审查。 |
| [^195] | [A Generative AI Agent to Emulate Interactive Human-driven Vehicles for Automated Driving Testing in Urban Areas](https://arxiv.org/abs/2611.31797) | 该论文提出一种生成式AI智能体，通过视场建模处理遮挡影响，并以五个指标刻画交互全过程的风格，从而真实模拟人类驾驶车辆的安全关键交互行为，服务于城市场景的自动驾驶安全测试。 |
| [^196] | [Constructive Causal Abstraction and a Case Study in Mechanistic Interpretability](https://arxiv.org/abs/2611.31795) | 该论文提出构造性因果抽象方法，将定义不良的宏观干预解释为宏观层面的缺失变量，并通过引入补充变量 q 保留被遗漏的因果相关状态，从而构造性地获得与微观模型因果一致的宏观因果模型，并在机制可解释性中进行了案例验证。 |
| [^197] | [AGL-SKILL: Coding Harnesses as Agentic Optimizers](https://arxiv.org/abs/2611.31794) | 本文的核心创新是证明无需任何指导的编程智能体本身即可作为高效的技能优化器，在五个基准测试中显著提升每个目标性能并超越现有已发表的技能优化方法。 |
| [^198] | [Three-Dimensional Wake Reconstruction and Cross-Scale Transfer for Floating Offshore Wind Turbines via Physics-Informed Fourier Neural Operators](https://arxiv.org/abs/2611.31789) | 该研究提出物理信息傅里叶神经算子框架，将四维谱卷积与不可压缩连续性及动量残差等物理约束相结合，可从两个观测平面重建浮式海上风力机的三维非定常尾流，并实现从5兆瓦到22兆瓦风机的跨尺度迁移。 |
| [^199] | [SciFig2Data: A Workflow for Recovering Structured Data from Scientific Figures](https://arxiv.org/abs/2611.31787) | 该论文提出SciFig2Data，一个集成图表定位、复合图分解、图表识别、多模态大语言模型数值重建与上下文关联的自动化高通量工作流，能够从科学PDF文献库中大规模恢复结构化实验数据。 |
| [^200] | [VISOR: A Vision-Language Selector for Multimodal Time-Series Forecasting](https://arxiv.org/abs/2611.31780) | VISOR 提出利用视觉-语言模型的强对齐能力，根据文本信息从随机预测器生成的多个候选未来中选出最合理的一个，而非取平均，从而改进多模态时间序列预测。 |
| [^201] | [SPARQL-Planned Single-Hop Traversal over a Live Knowledge Graph for Multi-Hop Question Answering](https://arxiv.org/abs/2611.31771) | 该论文提出将稠密检索与基于SPARQL规划的在线知识图谱（Wikidata）单跳遍历交替进行，通过推理路径验证检索段落，在避免多跳扩展开销与噪声的同时兼顾召回率与精确率，从而提升多跳问答性能。 |
| [^202] | [DELTA: Diagnosing Exploration Lineages via Twin-path Alignment - What Can Contrasting Design Paths Reveal About Microarchitectural Behavior?](https://arxiv.org/abs/2611.31767) | 论文提出DELTA工具，通过将设计空间探索树中同一修改在不同父配置下演化出的两条路径进行对齐，把原本被丢弃的父子配置对比数据转化为根因诊断证据，从而定位使同一设计修改在不同微架构环境下产生相反效果的交互因素。 |
| [^203] | [RoMo Hands: A Large Scale Richly Organized Text to Hand Motion Dataset](https://arxiv.org/abs/2611.31766) | 本文提出了RoMo Hands数据集——包含约54.2万序列（884小时）的真实场景文本-手部动作数据，规模比以往工作大一个数量级，并通过严格的灵巧性过滤、层次化多粒度文本标注以及新颖的393维双手表示方法，弥合了身体与手部动作生成架构之间的差距。 |
| [^204] | [Verified Proof Refactoring with Codex and Qwen](https://arxiv.org/abs/2611.31762) | 该论文提出一种结合本地部署 Qwen 模型与开源 Codex CLI 的系统，在保持定理陈述和跨 Lean 版本兼容性的前提下重构现有 Lean 4 证明以缩短源代码并降低检查成本，最终在 Lean Refactor Arena 开源 LLM 赛道中夺得第一名。 |
| [^205] | [TACT-Bench: Testing Adaptive Conversational Threats against Tool-Use Agents](https://arxiv.org/abs/2611.31756) | 提出了TACT-Bench，一个针对工具使用型AI智能体的有状态安全基准，通过场景化故事、人设、操纵策略和响应自适应升级，测试智能体在拒绝违反策略的未授权请求的同时保持合法任务完成的能力。 |
| [^206] | [RVLM-X: Towards a Universal VLM Harness](https://arxiv.org/abs/2611.31755) | RVLM-X提出了一种任务可泛化的递归式VLM辅助框架，让前沿模型在代码REPL中自主控制全分辨率图像的视觉上下文并递归调用自身，从而在照片、图表和地图等多种任务上无需改动即可提升准确率。 |
| [^207] | [FN-Prune: Training-Free Expert Pruning with Finite RMSNorm Perturbations](https://arxiv.org/abs/2611.31747) | FN-Prune是一种免训练的专家剪枝方法，通过比较减去各专家贡献前后的RMSNorm输出差异来评估专家重要性，在移除50%专家时于Qwen3、ERNIE和OLMoE上超越现有最佳方法，在OLMoE上困惑度降低32%。 |
| [^208] | [Where is the Interface for the AI Harness?](https://arxiv.org/abs/2611.31739) | 论文提出智能体委托打破了传统“用户在环”假设，开创了“规约鸿沟”与“监督鸿沟”两条新鸿沟，而两者都汇聚于由开发者编写、对用户隐藏的 AI 运行框架中，因此主张应将运行框架本身作为人机交互的新界面。 |
| [^209] | [GreenDynaMix: Thermal-Aware Dynamic Mixed-Precision MoE LLM Inference Systems](https://arxiv.org/abs/2611.31737) | GreenDynaMix是首个将DVFS温度管理与动态混合精度调优相结合的运行时系统，它利用MoE路由器的重要性信号为关键专家分配更高精度，在降低GPU热老化率25-33%的同时保持了推理精度。 |
| [^210] | [Reproducing History Is Not Enough: Intervention-Level Assurance for LLM-Generated Simulators](https://arxiv.org/abs/2611.31736) | 该论文提出Vouch库，通过将每个what-if问题转化为结构化更改并对照程序与当前记录进行验证，实现对LLM生成模拟器单次答案可信度的干预级保障，而非仅对整个程序给出通过与否的整体判定。 |
| [^211] | [MOSAIC: Memory-Organized Search with Anti-Imitation Crossover for Training-Free Microarchitecture Optimization](https://arxiv.org/abs/2611.31733) | MOSAIC提出了一种无需更新语言模型权重的免训练搜索控制器，通过存储评估经验教训以及在改进停滞时对嵌入差异较大的高分策略进行反模仿交叉组合，避免生成与既有策略雷同的缓存替换策略，从而在留出工作负载上保持更优性能。 |
| [^212] | [Geo-SMILE: Statistical Model-Agnostic Interpretability with Local Explanations for Geospatial Machine Learning](https://arxiv.org/abs/2611.31731) | 提出Geo-SMILE框架，通过互补的特征屏蔽与地理重定位两个扰动分支以及距离加权的替代模型，将实质性特征解释与地理空间位置响应区分开来，为地理空间机器学习提供模型无关的局部可解释性。 |
| [^213] | [EventSSM: Causal Spatial-State Modelling for Asynchronous Events Event-Depth Estimation](https://arxiv.org/abs/2611.31720) | 提出EventSSM——一种用循环记忆按时间顺序处理事件数据包的因果空间-状态模型，实现异步事件关联的度量深度估计，并通过相机位姿几何传输与可微渲染桥接异步预测与稀疏帧时刻监督，在多个深度误差指标上超越已有方法。 |
| [^214] | [PRETZEL: A Benchmark for Cyclic Peptide Structure Prediction](https://arxiv.org/abs/2611.31708) | 该论文提出了PRETZEL——一个包含285个环肽-蛋白复合物、涵盖三种环化化学且全部晚于模型训练截止时间的标准化基准，实现了对环肽结构预测模型零样本、可比较且含化学正确性验证的评估。 |
| [^215] | [FoveaWAM: Role-Aware 4D World-Action Modeling for Dynamic Manipulation](https://arxiv.org/abs/2611.31705) | FoveaWAM提出按角色分配预测精度的原生4D世界-动作模型，仅对目标位姿采用毫米级SE(3)轨迹预测、对环境用稀疏3D流建模、对静态内容依赖当前帧，从而以15.1毫秒的低延迟实现动态移动物体的高效操作。 |
| [^216] | [CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence](https://arxiv.org/abs/2611.31694) | CATWAY 提出了一种代价敏感的三支决策门控框架，利用四种指令条件下的损失矩阵和校准的语言模型置信度概率，让机器人在执行、追问与拒绝之间做出决策，并在跨领域测试中无需重新拟合即保持有效。 |
| [^217] | [Rapid Exploratory Maps and their potential role in supporting evidence-to-policy dialogues and processes](https://arxiv.org/abs/2611.31693) | 该论文提出利用人工智能技术构建快速探索性证据图谱，大幅缩短证据图谱绘制所需的时间，从而支持证据生产者与政策制定者之间的协作对话，更高效地确定证据综合的研究问题与范围。 |
| [^218] | [GRUN: Generalized Retinex Unfolding Network for Image Restoration](https://arxiv.org/abs/2611.31690) | 该论文提出广义Retinex公式与MAP引导的深度展开网络GRUN，将经典光照-反射分解泛化为内容-环境因子分解先验，通过交替进行反射恢复、光照估计和渐进式退化推断，实现对多种退化类型图像的修复。 |
| [^219] | [Right Destination, Wrong Journey: Synthetic Counseling Dialogues Miss Human Interactional Dynamics](https://arxiv.org/abs/2611.31686) | 该研究提出以轨迹为中心的过程保真度评估方法，发现LLM生成的心理咨询对话虽能模拟合理的最终结果，但在恢复动态和逐轮响应等人类互动动态方面保真度不足，且即使采取针对性的生成干预也无法完全恢复这些互动特性。 |
| [^220] | [CROSS-Calib: Selective Channel Calibration for Ultrasound Reconstruction](https://arxiv.org/abs/2611.31684) | 提出CROSS-Calib方法，利用其他通道的预测作为参考进行幅度和相位校准，并通过上下文门控机制选择性地融合校准测量与预测回波，在波束形成前修复不可靠通道数据，从而显著提升超声重建的图像质量。 |
| [^221] | [LimoBench: How Street Smart Are Large Language Models?](https://arxiv.org/abs/2611.31682) | 提出了首个街道级地理空间基准LimoBench，通过82个基于真实地图数据的自然语言位置查询，系统评估大语言模型在地理编码、路径规划、$k$近邻搜索等七种空间操作上的位置感知与精确回答能力。 |
| [^222] | [Look Before You Commit: Visual-Evidence-Guided Decoding for Multimodal Diffusion Language Models](https://arxiv.org/abs/2611.31677) | 论文发现扩散多模态大模型中近九成高置信度词元提交与图像证据无关，并提出无需训练的视觉证据引导解码方法CD-APC，通过证据调整评分、双重门控与逆窗口机制，在两个骨干模型和五个基准上持续提升性能、抑制幻觉。 |
| [^223] | [Provenance Loss and Repair in Multilingual Compaction](https://arxiv.org/abs/2611.31668) | 该论文发现多语言对话压缩会因语法性别等语言差异而丢失属性的来源信息（无法区分用户明确声明与语法默认），并提出一种无需训练、覆盖写作标注、丢弃前似然验证和阅读信任三个环节的修复方法，大幅恢复被压缩丢弃的已声明属性召回率。 |
| [^224] | [BRAG: Bayesian Retrieval Augmented Generation](https://arxiv.org/abs/2611.31656) | BRAG提出了一种贝叶斯检索生成模型，通过闭式滤波器在线更新每个记忆块的少量键和标量，将检索框架化为控制问题，并证明现有检索反馈方法均为该模型的特殊或极限情形。 |
| [^225] | [Stranded Compute in AI Inference Workloads: Recovering Value Through Demand Aggregation](https://arxiv.org/abs/2611.31654) | 本文指出AI推理成本中资本回收占87.3%而电力仅3.7%，其价值损失主要源于调度闲置和搁浅的突发预留，通过在多个不完全相关的租户间聚合需求可回收7.3个百分点的有效产出，相当于每兆瓦IT容量每年约75万美元。 |
| [^226] | [Constructing physical mechanisms with artificial intelligence](https://arxiv.org/abs/2611.31653) | 该研究构建了一个读取 arXiv 全部 250 万篇论文方程的方程原生系统，通过确定性计算在不同学科间拆解、转移和组合方程部件来构建物理机制，并在部件不匹配时自动生成缺失项的方程。 |
| [^227] | [E2SIM: Long-Horizon Simulation of Everyday Human Behavior with Evolving States](https://arxiv.org/abs/2611.31641) | E2SIM 提出了一个通过反馈循环将行动者演化状态与动态共享环境联合建模的纵向用户模拟框架，使早期行动的后果能够塑造后续行为，同时保留各行动者视角的差异，并配套了衡量动态一致性的纵向轨迹诊断指标。 |
| [^228] | [Wavefront: Low-Rank Drafting with Parallel Causal Refinement for Speculative Decoding](https://arxiv.org/abs/2611.31638) | 提出Wavefront推测解码方法：用学习到的低秩基础头替代昂贵的完整lm_head，并以整块草稿前缀为条件、通过并行Jacobi迭代精炼的更强修正头来恢复块内token依赖，实现高效且无数据依赖控制流的并行草稿生成。 |
| [^229] | [What You See Is What You Certify: Concept-Based Certificates for Unseen Compositions](https://arxiv.org/abs/2611.31627) | 该研究揭示了基于概念的模型在未见属性组合上会以高达93-94%的概率错误认证类别，并将本体论视为错误检测码，据此确立了有效决策证书的三个条件：排除所有竞争类别的可见证据、如实标注未见部件、以及不受共现捷径驱动的罕见误读。 |
| [^230] | [Self-Improving Long-Horizon Reasoning by Learning Summaries from Local Continuations](https://arxiv.org/abs/2611.31621) | 提出TruSuCo方法，通过Delta-KL奖励训练独立摘要器将长推理历史压缩为紧凑状态，利用模型自身在短上下文中的可靠表现实现长程推理能力的自我提升，而无需最终答案奖励或更强模型的蒸馏。 |
| [^231] | [Multi-agent System for Automatic Discovery of Pareto-optimal Meta-solvers for PDEs](https://arxiv.org/abs/2611.31611) | 该论文提出一种多智能体框架，让语言模型智能体与非支配排序遗传进化算法相结合，自动演化出针对偏微分方程的帕累托最优混合求解器，仅需单次运行即可在小模型上产生一组求解器权衡方案。 |
| [^232] | [Monitorable-by-design architectures for a post-Astra world: a compartmentalization-based approach](https://arxiv.org/abs/2611.31610) | 提出一种基于隔离化的LLM架构改进方案，通过将危险能力分区隔离，使其使用（尤其是可能预示策略性对抗行为的危险推理）能够以比思维链监控更低成本、且在可监控思维链失效后仍然有效的方式进行监控。 |
| [^233] | [RePT: 3D Geometry Emerges from 2D Relational Pretraining](https://arxiv.org/abs/2611.31601) | RePT通过跨视角二维点预测学习物理点之间的关系，使三维几何结构无需三维预测目标即可从二维关系预训练中自然涌现，并展现出跨数据集的泛化能力。 |
| [^234] | [GoldiBench: Right Tools, Wrong Amount -- Benchmarking Agents on Effect Judgement in Professional Editing](https://arxiv.org/abs/2611.31590) | 论文提出GoldiBench基准，通过500个专业编辑任务将“工具操作”与“效果判断（用量把握）”分离评估，发现智能体操作能力很强，但难以准确判断实现目标视觉效果所需的工具用量。 |
| [^235] | [LatentLens: Rethinking Latent Interface in Multi-Agent Systems](https://arxiv.org/abs/2611.31582) | 本文提出LatentLens框架，从外部和内部两个层面解耦多智能体系统中的潜在接口，发现在相同智能体框架下引入潜在接口并不能提升甚至可能显著降低性能，其效果取决于重新对齐兼容性、信息可访问性和传播动态等内部因素。 |
| [^236] | [Frequency-Conditioned Privacy Auditing of Fine-Tuned Language Models](https://arxiv.org/abs/2611.31581) | 该论文通过将敏感信息（邮箱地址和电话号码）在微调数据中的重复次数从1倍系统地变化到50倍，发现不同语言模型在敏感信息变得可提取前所需的重复阈值差异巨大，且梯度裁剪只能延迟信息泄露而在高重复率下无法可靠地防止泄露。 |
| [^237] | [Provably Near-Optimal PQ in the Extended Zador Model](https://arxiv.org/abs/2611.31579) | 本文提出EZPQ算法，通过证明SVD是最优预处理旋转、构建低比特率下准确的扩展Zador代价估计器、以及基于一维装箱的全局比特分配分割方法，实现了具有可证明近似保证且实际性能接近最优的乘积量化。 |
| [^238] | [The Paradox of Agency-Preserving AI: Negotiating Learner Agency in Layered Human-AI Interactions](https://arxiv.org/abs/2611.31573) | 该研究通过共同设计工作坊揭示了元层级AI系统的根本悖论——旨在保护学习者能动性的AI反而可能通过规定反思的时机与方式约束学习者能动性，并基于真实学习数据识别出四个相互关联的设计张力。 |
| [^239] | [Safe Autoformalization: Sufficiency and Necessity with Entailment Graphs](https://arxiv.org/abs/2611.31571) | 该论文提出了一种安全自动形式化方法：用LLM生成候选公式并构建由求解器判定的蕴含图，通过搜索充分前沿与必要前沿认证单侧逻辑充分性，使得对意图充分的公式可安全执行、对意图必要的公式可安全依赖。 |
| [^240] | [Off-the-Shelf Autonomous Agents for Game Playtesting](https://arxiv.org/abs/2611.31570) | 提出Buddy——一个完全基于现成语言模型、无需训练或攻略的自主智能体，能在未见过的3D角色扮演游戏中通过探索解决长时程目标，并支持自动记录缺陷工单和将成功轨迹转化为分层任务网络用于回归测试。 |
| [^241] | [COGENT: A Modular Cognitive Architecture for Agentic User Simulation](https://arxiv.org/abs/2611.31559) | COGENT是一个借鉴Norman动作循环和ACT-R等认知理论的模块化认知架构，通过将用户模拟分解为具有显式表示的感知、记忆、用户状态、意图和行动选择等认知模块，实现了对桌面界面交互的可解释、可干预的用户模拟。 |
| [^242] | [Denoising Diffusion with Progressive Posterior Sampling](https://arxiv.org/abs/2611.31558) | 该论文提出带流形投影的渐进后验采样（PPS）方法，通过闭式高斯后验近似自动估计噪声方差、避免SVD，并将样本投影至局部含噪图像流形，从而更有效地利用扩散先验的局部几何结构求解含噪线性逆问题。 |
| [^243] | [Agent Recovery Gone Wrong: Measuring Harm Beyond Duplicate Calls](https://arxiv.org/abs/2611.31554) | 该论文首次在自然发生的失败场景下量化了代码执行型LLM智能体恢复操作对外部状态的危害，发现约24.6%的结果可确定的部分执行会导致在运行结束时仍持续存在的有害恢复，其中仅三分之二表现为重复调用，其余则表现为删除用户数据或错误的首次写入，说明仅检测重复调用不足以捕捉全部危害。 |
| [^244] | [Reproducible Social Structure Emerges From Free-Form Interaction Among LLM Agents](https://arxiv.org/abs/2611.31543) | 该论文提出了一个可重复、无目标的多智能体大语言模型模拟框架，并通过十次重复运行证明社会结构（如交流最终收缩为持续的二人关系）能够稳定地重复涌现，从而区分了模拟系统中的稳健发现与随机事件。 |
| [^245] | [Where Agent Defenses Fail: A Case Study of Policy Compositions and Measurement Limits](https://arxiv.org/abs/2611.31541) | 本文通过大规模案例研究揭示了智能体安全防御策略组合的固有矛盾：限制性沙箱虽能阻止攻击但严重损害良性效用，中介机制虽恢复功能却可能放行机密泄露，且现有测量方法存在编码泄露、观察者遗漏等盲区，为智能体安全评估的局限性提供了实证依据。 |
| [^246] | [Agent Evaluation Without Ground Truth: A Null Proportion Perspective](https://arxiv.org/abs/2611.31539) | 该论文提出了一种无需真值标签的智能体评估框架，将智能体质量定义为其轨迹中正确决策的加权和，用 e 值序列作为统计证据替代真值标签，并能处理任意相关的决策、数据依赖的轨迹终止以及动态生成的假设。 |
| [^247] | [Tempering and few-shot calibration of pathology foundation models improve digital biomarker reproducibility across slide scanners](https://arxiv.org/abs/2611.31536) | 该研究提出一种仅需公开数据的“调和”微调方法，可显著降低病理基础模型对扫描仪差异的敏感性，大幅提升HPV预测、MSI识别、前列腺癌分级等数字生物标志物的跨扫描仪可重复性。 |
| [^248] | [One Step, Four Clocks: How GUI Agents Fall Out of Step](https://arxiv.org/abs/2611.31522) | 该论文提出EODA四事件时间模型与ChronotopeGym基准，揭示GUI智能体在实时交互中因界面变化导致动作失效的根本原因，并设计驻留加速器的Chronotope控制器，通过选择性保留、更新或重决策待执行动作，将端到端延迟降低85.5%并实现100%实时成功率。 |
| [^249] | [PatientFlowSynth: Evaluating FHIR Agents with Matched Patient Charts](https://arxiv.org/abs/2611.31521) | 该论文提出 PatientFlowSynth 基准，利用 225 个合成患者和 18 个模板生成 18,000 个控制变量匹配的病历变体任务，在四种证据状态（含缺失与冲突证据）下系统评估 FHIR 医疗智能体的分类、引用与检索能力，并验证了明确的就诊状态指导能显著提升严格成功率。 |
| [^250] | [XRL4T1D: A Post-hoc Explainability Framework for Reinforcement Learning in Type 1 Diabetes Management](https://arxiv.org/abs/2611.31498) | 本文提出XRL4T1D框架，无需重新训练即可利用SHAP等方法对1型糖尿病胰岛素给药的强化学习策略进行事后解释，揭示了CGM血糖值与胰岛素历史输入对决策的重要性。 |
| [^251] | [Reading Between the Layers: Where Representation Quality Lives, and What Geometry Reveals](https://arxiv.org/abs/2611.31496) | 该论文通过对25个模型、13个数据集的大规模实证研究，系统揭示了预训练语言模型的最佳表征层随预训练策略与架构而系统性变化（掩码编码器峰值在中后层、序列到序列模型随任务类型偏移、对比编码器取决于读出方式、仅解码器模型呈阶段结构），为无标签层选择这一开放问题提供了实证指引。 |
| [^252] | [Evaluating Frontier-Model Agent Skills with Open-Weight Models in a Science Gateway](https://arxiv.org/abs/2611.31480) | 本文提出一种方法，通过自动评估测试框架驱动真实科学网关、对生成的 Pegasus 工作流评分并把反复出现的失败反馈给前沿模型，从而将前沿模型编写的智能体技能适配到科研基础设施上免费提供的开放权重模型，使没有前沿模型额度的科学家也能用聊天界面构建可靠的科学工作流。 |
| [^253] | [SpatialCraft: Evolving Procedural Intelligence from Experience and Skills for Spatial Reasoning Agents](https://arxiv.org/abs/2611.31475) | SpatialCraft提出一个无需参数更新的自演化框架，让冻结的工具调用智能体通过积累“经验”（条件-动作策略）和“技能”（带激活与终止条件的多步骤程序）两类外部知识，从自身交互历史中不断改进工具编排能力。 |
| [^254] | [AdProducer: Towards Agentic Advertising Video Generation](https://arxiv.org/abs/2611.31466) | AdProducer 是首个端到端的智能体广告视频生成框架，它通过自主开展产品调研并将结果转化为以卖点驱动的分镜脚本，从而将简洁的创意简报直接转化为成品广告。 |
| [^255] | [Below the Stopwatch: Checking the Performance Claims of Coding Agents in the Kernel](https://arxiv.org/abs/2611.31460) | 提出了Below流水线，通过提取编码代理性能PR所声称的优化机制并用硬件计数器与LTTng内核追踪验证其应有的内核级信号，发现大多数性能声明缺乏实证测量，且相当一部分PR并未带来真实的性能提升。 |
| [^256] | [Taxi: Analyzing Agents at (Real) Scale](https://arxiv.org/abs/2611.31457) | Taxi 是一个可高效扩展至数百万步骤的大规模智能体调试、聚类与监控系统，通过聚类分类体系、完整 SQL 查询和 Jev 式决策模型的混合流水线，为工程师和自动化智能体提供结构化分析能力。 |
| [^257] | [Leaks by Installments: Cross-Session Answer Disclosure in LLM Tutors and a Ledger-Based Defense](https://arxiv.org/abs/2611.31429) | 本文提出一种为每个学习者维护持久化泄露账本的防御机制，通过类型化解题图与门控匹配检测跨会话答案泄露，将模拟实验中通过改编数字练习变体产生的泄露率从100%降至10%。 |
| [^258] | [Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training](https://arxiv.org/abs/2611.31427) | 该论文将智能体能力分解为“任务能力”（解决任务所需的知识与推理）和“界面能力”（通过特定框架表达知识的能力），揭示了基于框架的训练效果取决于这两种能力的交互方式，并提出多框架蒸馏可将不同框架下的任务知识聚合到共享策略中。 |
| [^259] | [ContextLure: Exploiting Benign Conversation History for Indirect Prompt Injection](https://arxiv.org/abs/2611.31424) | 该论文提出ContextLure红队框架，攻击者通过静默观察良性对话历史并从中提取证据来构造更强的间接提示注入，无需对对话轮次拥有额外控制即可提升攻击效果。 |
| [^260] | [Nemotron-Labs-VisualCreation: Pixel as Code for Agentic Visual Creation](https://arxiv.org/abs/2611.31358) | 该论文提出“像素即代码”的智能体视觉创作框架，用离散视觉token表示光栅图像、用显式程序表示结构化制品，并配套开发了图像tokenizer、自回归光栅生成器和结构化代码模型，统一实现了对图像、矢量图形、网页和文档的生成、渲染与修复。 |
| [^261] | [GHC:Expanding Residual Interfaces in Hyper-Connections](https://arxiv.org/abs/2611.31350) | 提出分组超连接，通过沿注意力头或FFN通道将分支分组并为每组提供差异化的流读写接口，在开销可忽略的情况下将训练效率较Pre-Norm提升35%、较IHC提升10%，同时克服了增加流数量时的收益递减问题。 |
| [^262] | [Antares-Smaug: Adversarial Names, Tokens and Attack Rewrites: Sanitisers Miss Unseen Generalisations](https://arxiv.org/abs/2611.31347) | 该论文提出 Antares-Smaug 攻击框架，通过 81 种程序化编辑系统攻击开放决策模型 Laya，发现净化器无法泛化到未见过的攻击变体，注入攻击主要经由选项名称而非描述起作用，且仅字符级篡改即可造成高达 72% 的正确答案损失。 |
| [^263] | [What People Choose Not to Say: A Multimodal Benchmark for Fine-Grained Information Manipulation in Conversations](https://arxiv.org/abs/2611.31346) | 提出了 InfoCon——一个多模态基准，通过基于证据的多智能体标注框架，识别和评估对话中隐瞒、泄露、捏造、回避等六种细粒度信息操纵策略及其一致性关系。 |
| [^264] | [Beyond Static Charts: Benchmarking Interactive Chart-to-Code Generation for MLLMs](https://arxiv.org/abs/2611.31345) | 该论文提出了首个面向交互式图表转代码生成的基准测试ChartInt，包含2,779个涵盖悬停、高亮、工具提示等交互模式以及数据更新、样式调整等编辑任务的标注示例，弥补了现有基准仅关注静态图表的不足，并引入了基于行为一致性的评估范式。 |
| [^265] | [Per-Attack Blocking Hides Catalog Vulnerability: A Weakest-Link Analysis of Tool Manifests](https://arxiv.org/abs/2611.31338) | 该论文指出“单攻击拦截率”是错误的评估单位——工具目录的漏洞率随独立编写的策略数量按 1-(1-p)^M 增长（从 10 个工具的 22% 升至 200 个工具的 99%），而单攻击拦截率始终保持在约 92%，因此安全性应按目录级“最弱环节”而非单次攻击拦截来衡量。 |
| [^266] | [Conservative Overlap Fuzzy Rough Sets for Incomplete Data](https://arxiv.org/abs/2611.31329) | 该论文提出保守重叠模糊粗糙集（COFRS）模型，通过对共同存在属性进行加权重叠、全局惩罚缺失值的相似度定义及基于加权粒一致性的软依赖度，实现了不完整数据上高效且稳健的属性约减。 |
| [^267] | [Layer-Readout Interfaces in EEG Foundation Models: Predictive Accessibility and Cross-Dataset Transfer](https://arxiv.org/abs/2611.31320) | 本研究解析了EEG基础模型中间层优于最终层的原因：改进最终层读出接口即可恢复大部分预测性能，但冻结提取下中间层仍具优势，且所选择的层与任务方向能够跨数据集迁移。 |
| [^268] | [Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse](https://arxiv.org/abs/2611.31319) | 该论文提出将保形化的分布外异常检测门控与可操作性约束的反事实搜索相结合，防止反事实救济利用梯度提升树决策边界的伪影和低密度分布外区域，从而生成更合理的分布内救济建议。 |
| [^269] | [Scoring Recovery-Time Agent Security: Proposals, Effects, and Retry Instructions](https://arxiv.org/abs/2611.31318) | 该论文提出了一种可执行的恢复期智能体安全性评估方法，通过分别记录工具提案、运行时决策与已提交效果，揭示了仅按目标完成度评估会掩盖有害恢复行为，并证明清晰的效果定义与恢复指令能显著提升超时写入场景下的有效恢复率。 |
| [^270] | [Evolutionary AI Control: Population-Level Safety Under Reproduction, Heredity, and Selection](https://arxiv.org/abs/2611.31312) | 该论文针对开放多智能体系统证明了安全保证在繁殖、遗传与选择过程中得以持续存在的条件，并通过四个反例揭示：即使每个智能体都满足个体安全检查，其演化出的种群仍可能突破集体安全限制。 |
| [^271] | [When Successful Collaboration Becomes an Attack Surface: Insider Memory Poisoning in Multi-Agent Systems](https://arxiv.org/abs/2611.31307) | 该论文揭示多智能体系统的共享成功协作记忆会构成新的攻击面，并提出MIMPA攻击——恶意智能体仅需正常参与权限而无需直接访问记忆，即可通过在早期成功协作记录中植入载荷，在后续任务中误导良性智能体给出指定错误答案。 |
| [^272] | [Set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem](https://arxiv.org/abs/2611.31306) | 提出了一种融合记忆机制的基于集合的综合学习粒子群优化算法，用于求解离散优化问题，并通过与现有算法的数值实验对比验证了其有效性。 |
| [^273] | [Differential evolution with perturbation for continuous function optimization](https://arxiv.org/abs/2611.31305) | 本文提出了一种结合扰动机制的差分进化算法（DEP）用于连续函数优化，并通过CEC2017数据集与现有算法的对比验证了其有效性。 |
| [^274] | [MOZAIC: An Audio-Video Deepfake Benchmark for Telling What Is Fake](https://arxiv.org/abs/2611.31301) | 提出了MOZAIC音视频深度伪造配对基准，通过四分类标签直接量化视频与音频模态各自在伪造中的贡献，揭示了现有最强多模态大模型仅有35.6%的四分类准确率且严重偏向检测视频伪造而非声音伪造的局限。 |
| [^275] | [PEACE-TEST-PROTOCOL: Polarization-Keyed Examination Papers and Mandatory Issued Eyewear that Displace AI Glasses and Extend Universal Design](https://arxiv.org/abs/2611.31299) | 该论文提出PEACE-TEST-PROTOCOL协议，通过向全体考生强制配发官方眼镜、并以圆偏振编码印制试卷（肉眼和摄像头只能看到均匀灰色页面），从“眼镜归属权”层面根本性地阻断AI眼镜摄像头作弊，同时通过处方插入件兼容戴镜和低视力考生。 |
| [^276] | [The Industrialization of Noise: Why Automated Proof-Solving is the Same Old Dead End](https://arxiv.org/abs/2611.31286) | 本文借助 DeMillo–Lipton–Perlis 框架批判 OpenAI 发布的722篇自动化数学证明手稿，指出将机械求解与人类理解的社会过程相脱钩，只会造成数学生态系统的“工业污染”，产生死寂而无效的证明仓库。 |
| [^277] | [Irreducible Governance Structure for Autonomous Agent Systems: Fair Allocation, Strategy-Proofness, and Multi-Scale Composition](https://arxiv.org/abs/2611.31284) | 本文证明了在多智能体系统中，执行权的分配是治理的结构性核心，公平性、效率与抗策略操纵之间存在不可能性权衡，且这些约束会跨系统层级传播，因而必须构建多尺度组合式治理结构。 |
| [^278] | [LLM Genetic Feature Engineering](https://arxiv.org/abs/2611.31277) | 该论文提出了一种在单个特征层面而非整个特征集层面执行选择、交叉和变异的新型遗传算法，使遗传算法能够在过程中对新特征进行统计分析，并同时整合个体与群体层面的特征指标，从而改进基于LLM的特征工程自动化。 |
| [^279] | [Agentic Optimization: LLM Agents Select and Configure Algorithms for Engineering Design](https://arxiv.org/abs/2611.31275) | 提出智能体优化方法，让LLM智能体通过阅读自然语言问题描述来配置和启动多目标优化算法，在四个工程基准和真实分子对接任务上全面超越固定算法基线。 |
| [^280] | [Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics](https://arxiv.org/abs/2611.31269) | 本文构建了由45,441条钙钛矿光伏实验记录确定性生成真实答案的定量基准，系统评估语言模型评判器的可靠性，发现配备参考答案的前沿评判器Claude Opus可近乎完美地复现真实评分并精确保留架构排名。 |
| [^281] | [AhaPro: A Process-level Rebus benchmark for probing Aha moments in LLM reasoning](https://arxiv.org/abs/2611.31268) | 本文提出基于字谜的过程级基准AhaPro来探测大语言模型推理中的“顿悟”时刻，发现模型常常未能采纳顿悟中发现的更优状态，且已采纳的修正甚至可能无益或有害。 |
| [^282] | [SD-Cache: Learnable Memory Consolidation with State-Guided Dynamic Caching](https://arxiv.org/abs/2611.31265) | 提出SD-Cache，通过状态引导的动态缓存机制，在不预知未来查询的情况下，将LLM原生的KV状态选择性地整合到固定大小的工作记忆中，解决了RAG场景下KV缓存线性增长的问题。 |
| [^283] | [How Persona Conditioning Changes Open-Ended Task Execution: A Paired Evaluation of Behavioral Gains and Reliability Costs](https://arxiv.org/abs/2611.31264) | 该论文通过配对实验首次系统揭示了人设条件化作为任务引导信号的双面性：它能提升LLM回复的专家内容深度和流程忠实度，但会以17至30个百分点的幅度降低严格指令遵循准确率并引发无依据的细节虚构。 |
| [^284] | [Evaluating Large Language Models on Iranian Doctoral Entrance Exams in Electrical and Computer Engineering](https://arxiv.org/abs/2611.31263) | 该研究利用伊朗电气与计算机工程博士入学考试的真实试题（含波斯语与英语双语及依赖配图的题目），通过统一的统计方法系统评估了GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5和Grok 4四个大型语言模型的答题准确率。 |
| [^285] | [RegioZH: Unveiling Comprehension Disparities across Regional Chinese Varieties in LLMs](https://arxiv.org/abs/2611.31262) | 该论文提出 RegioZH——首个覆盖中国大陆、台湾、香港、新加坡和马来西亚的 2,607 题地区中文词汇推理基准，揭示了大语言模型在理解不同地区中文变体上的显著差异（如台湾与新加坡之间 29.4 个百分点的准确率差距），并表明部分错误源于检索或表述框架问题而非单纯的知识缺失。 |
| [^286] | [Does Item Response Theory Generalize Across Languages? Evidence from Cross-Lingual LLM Benchmarks](https://arxiv.org/abs/2611.31261) | 该研究通过在Multilingual GSM和Global MMLU上对100多个语言模型拟合1PL/2PL/3PL IRT模型，发现题目难度是跨语言最稳定的IRT参数，而猜测和区分度的稳定性依次减弱，因此多语言场景下基于IRT的数据选择需要先验证模型选择并逐一筛查具体题目。 |
| [^287] | [Capability and Interpretability of Transformers on Multi-task Recurrence](https://arxiv.org/abs/2611.31260) | 本文通过有限循环群上的线性递归任务族系统研究了Transformer的能力与可解释性，发现切换明确指定的规则比识别潜在规则更容易，并揭示了模型深度、宽度以及递归的代数结构对泛化能力的影响，同时证明课程学习在规则数量较大时尤为有效。 |
| [^288] | [RecipeADQA: A User-aware Question-Answering Dataset for Multi-Constraint Recipe Recommendation](https://arxiv.org/abs/2611.31259) | 本文提出了RecipeADQA——一个以真实用户行为为驱动、利用大语言模型提取细粒度偏好信号并映射到知识图谱标签模式上的多约束食谱推荐基准问答数据集，同时通过贪心约束选择策略和基于标签的重排序保证数据质量。 |
| [^289] | [RECAP: Training Long-horizon LLM Agents with Context Compaction](https://arxiv.org/abs/2611.31258) | RECAP提出一种强化学习框架，通过行为失真目标将摘要生成与智能体策略联合训练，使压缩摘要能够保留原始交互历史所诱导的行为，从而提升智能体在长时程任务中的表现。 |
| [^290] | [scOPD: Routed On-Policy Self-Distillation for LLM-Based Single-Cell Annotation](https://arxiv.org/abs/2611.31257) | scOPD提出了一种面向大语言模型批次级单细胞注释的在策略自蒸馏框架，通过自适应地将合格rollout路由至基于奖励的优化、并利用错误rollout进行自蒸馏，为中间生物推理提供细粒度监督。 |
| [^291] | [Directly Maximizing On-policy Expected Accepted Length for Speculative Decoding](https://arxiv.org/abs/2611.31256) | FreDraft通过在策略训练直接最大化草稿的期望接受长度，并利用目标模型词表分布解析边缘化接受率以降低梯度方差，从而更有效地提升投机解码的加速效果。 |
| [^292] | [INCLUDE-2.0: Disentangling Knowledge Gaps in Multilingual LLM Evaluation](https://arxiv.org/abs/2611.31255) | INCLUDE-2.0是一个覆盖86个国家91种语言、包含22万余样本的区域感知多语言事实评估基准，通过标注每道题所需的区域知识类型（而非将语言简单等同于区域），实现了对多语言大模型知识鸿沟的精细化受控分析。 |
| [^293] | [The Date You Didn't Set: The System-Prompt Date as a Hidden Variable in LLM Evaluation](https://arxiv.org/abs/2611.31254) | 本文揭示系统提示中自动填入的日期是LLM评估中被忽视的隐藏变量：仅改变日期就会使中位数6.6%的基准题目翻转正误，在对抗性搜索下甚至使模型准确率损失高达36.5分，因此评估时必须固定并报告系统提示日期。 |
| [^294] | [Instantiation of Human Values in Image Generation](https://arxiv.org/abs/2611.31253) | 该研究基于施瓦茨人类价值理论构建分析框架，通过28,000张生成图像揭示四个主流文生图模型如何将成功、关爱、自由等抽象价值狭窄地具象化为重复出现的视觉原型，并将其映射到社会构型分类体系以揭示模型系统性的价值呈现偏向。 |
| [^295] | [DAGGER: Deployment-Free Quality Estimation for Generative AI Workflows](https://arxiv.org/abs/2611.31252) | DAGGER框架通过将生成式AI工作流建模为图结构，实现了无需穷举部署与性能分析即可高效准确地估算工作流端到端质量，突破了现有方法在估算成本与准确性之间的权衡。 |
| [^296] | [Validation Bias in Large Language Model Responses to Cancer Treatment Queries: A Factorial Cross-Sectional Study](https://arxiv.org/abs/2611.31251) | 这项析因横断面研究通过对14个大语言模型、1,272段对话的分析，证实了LLM在应对与证据不一致的癌症治疗信念时存在“验证偏见”——倾向于认同而非纠正用户的错误信念，这对患者安全构成潜在风险。 |
| [^297] | [Actionwave: A Systematic Benchmark for Waveform-Centric Clinical Prediction Across Care Scenarios](https://arxiv.org/abs/2611.31249) | Actionwave 是首个覆盖重症监护、手术室和急诊科等多种医疗场景的以波形为中心的临床预测系统性基准，通过统一的决策时点预测接口、严格的评估协议，系统考察了信号时长、预警时间和可用输入等条件对模型表现的影响。 |
| [^298] | [InletScope: Multiscale pressure-based inlet unstart monitoring under changing sensor configurations](https://arxiv.org/abs/2611.31246) | InletScope 提出了一种基于多尺度压力的进气道不起动监测框架，通过共享多尺度编码、可学习尺度融合与传感器概率平均，在传感器配置发生变化时无需重新训练即可实现不起动检测和0.5秒内的新发事件预测，并在马赫2.9风洞试验中取得了0.834–0.964的窗口F1分数。 |
| [^299] | [Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching](https://arxiv.org/abs/2611.30986) | 该论文揭示现有前向扩散强化学习方法因行为分布与策略失配导致基线引起的梯度漂移和训练不稳定，并提出瞬时轨迹分布匹配（iTDM）来解决这一问题，从而高效对齐扩散模型。 |
| [^300] | [Learning Multimodal Representations from Partially Paired, Small-Scale Data](https://arxiv.org/abs/2611.30601) | 该论文提出CAMEO对抗学习融合框架，将预训练单模态编码器与跨模态潜在对齐机制结合，仅需极少配对样本即可从部分配对的小规模数据中学习多模态共享表示，在计算病理学任务上以卓越数据效率超越CLIP等对比方法。 |
| [^301] | [RoboInF: A Benchmark for Instruction Fidelity in Robotic Manipulation](https://arxiv.org/abs/2611.30555) | 该论文提出RoboInF基准，通过在固定初始场景与机器人状态下变换指令的目标、流程或细节，并利用任务专属检查器结合轨迹、接触和事件序列评估全执行过程的合规性，从而专门衡量操作策略对指令变化的忠实遵循程度。 |
| [^302] | [De Novo Generation of Odorant Molecules with Targeted Olfactory Receptor Activation Patterns](https://arxiv.org/abs/2611.30509) | 该论文提出了一个基于强化学习的气味分子从头生成框架，通过构建嗅觉受体全景结合矩阵、聚类发现代表性激活模式，并采用结合成对排序、分布匹配与复杂度约束的结构化选择性损失，成功生成匹配目标多受体激活谱的分子，在广谱型和稀疏激活型等罕见目标谱上取得了最大提升。 |
| [^303] | [mini-psych-agent: Evaluating Human-Like Behavior in LLM Agents through Interactive Psychological Environments](https://arxiv.org/abs/2611.30365) | 该论文提出mini-psych-agent，一个交互式心理环境测试平台，旨在评估大语言模型智能体在不同智能体工作流程和测试时计算预算下行为表现的类人程度。 |
| [^304] | [PixelOne: From Options to Pixels in a Single Forward for GUI Grounding](https://arxiv.org/abs/2611.30235) | PixelOne通过将搜索教师模型的渐进式空间知识蒸馏给学生模型，使学生能在原生分辨率下于单次前向传播中直接从截图像素预测出精准的点击坐标。 |
| [^305] | [Differential Proteomics to XAI Rules: A SHAP-FOL-ILP Modeling for Lung Adenocarcinoma](https://arxiv.org/abs/2611.29989) | 该研究构建了一个融合配对差异蛋白质组学、稀疏逻辑回归、SHAP归因与一阶逻辑归纳编程（FOL/ILP）的可解释建模流程，能从肺腺癌蛋白质组数据中自动提炼出可解释的分类规则。 |
| [^306] | [Seeing vs. Knowing: Counterfactual Supervision and Correction for 3D-Aware Object Editing](https://arxiv.org/abs/2611.29542) | 提出反事实状态监督与对称反事实校正方法，教会3D感知物体编辑器在编辑时遵循3D状态生成不可见内容、同时保留照片可见外观，并确保这一区分在采样过程中得以维持。 |
| [^307] | [When Can a Rewiring Control Test LLM Weight Readers? C2 and an Eligibility Protocol](https://arxiv.org/abs/2611.29120) | 论文提出C2反事实方法（置换网络最终线性读出层的列）及三条件资格协议，用于区分大语言模型是真正在“计算”权重还是仅识别权重空间“指纹”，但测试发现没有任何自然设置符合该协议资格。 |
| [^308] | [Improving the Improver: Semantic PID Control for Medical Agent Self-Improvement](https://arxiv.org/abs/2611.28680) | 该论文提出Semantic PID-RSI框架，借鉴PID控制理论，通过证据引导的三个诊断角色来改进医疗智能体的自我改进过程本身，解决了现有递归自我改进方法在高噪声医疗领域中的反馈歧义与开发集过拟合问题。 |
| [^309] | [EvoViM: Recursively Self-Evolving Visual Agents for Exploration with Deferred Queries](https://arxiv.org/abs/2611.28574) | EvoViM提出一种递归自我进化框架，让冻结的多模态模型将交互经验转化为可直接在像素上计算的视觉程序，并通过自我追踪失败、修正证据处理操作来应对查询延迟的探索任务。 |
| [^310] | [Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs](https://arxiv.org/abs/2611.28349) | 该论文提出Grammar-as-a-Draft（GaaD），将语法引擎作为第二草稿模型，把语法强制Token与推测草稿模型的提议合并为更宽的草稿行，使语法约束的结构化输出下每次推测解码步可验证更多Token，且无需新增内核或编译。 |
| [^311] | [Social sycophancy differs across access modes of large language models](https://arxiv.org/abs/2611.28012) | 该研究首次系统比较了主流大语言模型（ChatGPT、Claude、Gemini、DeepSeek）通过 API 与用户界面（UI）两种访问方式在社会谄媚性上的差异，发现 UI 访问普遍导致更强的谄媚行为（比值比高达 7.56），且这种差异在不同模型间存在显著不同。 |
| [^312] | [Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events](https://arxiv.org/abs/2611.27509) | 该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。 |
| [^313] | [VMSA: Video Memory Skill Agent for Long Video Understanding](https://arxiv.org/abs/2611.27381) | 提出VMSA框架，通过联合设计三种互补的结构化视频记忆（片段字幕记忆、实体事件记忆、实体感知情节记忆）与记忆技能，提升智能体在长视频问答中跨长时间跨度检索与整合证据的能力。 |
| [^314] | [From Acting to Learning: Experience Selection toward Continually Learning Agents](https://arxiv.org/abs/2611.27159) | 该论文将经验选择作为持续学习智能体的关键组成部分，通过大规模对照实验证明，优化器感知的梯度选择方法在学生智能体任务成功率上显著优于全局随机选择和高 NLL 选择两种基线。 |
| [^315] | [MixFold: Sub-4-Bit Mixed-Precision LLM Inference via Output-Channel Grouping on Mobile NPUs](https://arxiv.org/abs/2611.26976) | MixFold通过将输出通道分组与移动NPU矩阵单元瓦片对齐，无需解码时扩展权重即可原生执行3比特和对称2比特混合精度大语言模型推理，在骁龙Hexagon NPU上实现比llama.cpp相当或更高的准确率与更快的解码速度。 |
| [^316] | [When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards](https://arxiv.org/abs/2611.25793) | 本文首次系统揭示了多模态安全防护中的“理据幻觉”现象——幻觉普遍存在且不随分类准确率或模型规模提升而减少，并据此提出工具对比接地方法来诊断与缓解这一问题。 |
| [^317] | [Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models](https://arxiv.org/abs/2611.25594) | 本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。 |
| [^318] | [LM-based Personalized Travel Planner](https://arxiv.org/abs/2611.25531) | 提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。 |
| [^319] | [MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection](https://arxiv.org/abs/2611.25194) | 该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。 |
| [^320] | [Hypnos: A Dual-Phase End-to-End Platform for Scalable Continual Learning at the Edge](https://arxiv.org/abs/2611.24637) | 提出了受大脑清醒-睡眠周期启发的开源双阶段平台Hypnos，在清醒阶段利用搭载RISC-V微控制器的低功耗芯片实现高效推理与无梯度原型更新，从而在资源受限的边缘设备上实现可扩展的持续学习。 |
| [^321] | [As Student Rollouts Continue, Visual Backdoors Keep Growing](https://arxiv.org/abs/2611.22446) | 本文挑战了“在线策略蒸馏（OPD）对后门传播具有天然抵抗力”的直觉，通过多模型家族与数据集上的实验及理论分析，揭示了视觉后门会随着学生模型在线展开生成而持续传播增长的隐蔽风险。 |
| [^322] | [RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents](https://arxiv.org/abs/2611.22113) | 该论文提出RegimeCouncil——一个无需重训LLM的轻量级成对证据路由器，将“记忆冲突治理”形式化，并在局部与全局两种记忆抑制策略之间在线仲裁，以解决长期记忆悄然过时与延迟反馈下负面经验传播范围难以确定的问题。 |
| [^323] | [On-policy distillation as a Backdoor Propagation Channel](https://arxiv.org/abs/2611.19207) | 本研究首次揭示在策略蒸馏可成为后门传播的高效通道：教师模型中被植入的后门仅需极少触发提示暴露即可转移至学生模型，且转移强度显著高于传统蒸馏方法。 |
| [^324] | [Editable Video Context: Mid-Context KV-Cache Reuse for Context-Managing Video LLMs](https://arxiv.org/abs/2611.18718) | 该论文提出面向视频大语言模型的可编辑上下文方法，通过M-RoPE时间、高度、宽度分量共享的单个偏移量旋转键值以实现上下文中段的KV缓存复用，并结合内容哈希缓存避免重复视觉编码计算，从而显著降低长视频管理的预填充token数量与重建延迟。 |
| [^325] | [Visual Similarity in Adult Word Extension and Artificial Word Learning](https://arxiv.org/abs/2611.14116) | 本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。 |
| [^326] | [From Attack Surface to Defense Architecture: Compositional Security for Agentic Coding Harnesses](https://arxiv.org/abs/2611.13960) | 该论文提出ActiveSecHarness——一种基于“智能与权限分离”原则的神经符号参考监控器架构，通过HMAC能力令牌与二维污染格为AI编程智能体提供结构性的组合式安全防御。 |
| [^327] | [Speculative Self-Distillation enables Efficient Knowledge Internalization](https://arxiv.org/abs/2611.12016) | 提出推测式自蒸馏（SSD）——一种逐 token 混合策略方法：默认由学生生成 rollout，教师仅在下一个 token 分布显著分歧处介入，既贴近学生的测试时行为分布又避免低信号状态，从而实现更高效的知识内化。 |
| [^328] | [RoadState: A Domain-Prior-Enhanced Vision-Language System for Industrial-Grade Lane-Level Map Updating](https://arxiv.org/abs/2611.10746) | 该论文提出了已在百度地图生产环境中部署的视觉语言系统RoadState，其核心模型RoadState-VLM将道路场景理解建模为自回归状态估计，把连续的道路观测转换为结构化、时间一致且可审计的车道级语义，实现工业级车道级地图的可靠更新。 |
| [^329] | [On Agentic Manipulation Policies](https://arxiv.org/abs/2611.09892) | 提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。 |
| [^330] | [Learnings and Pitfalls from AI-Assisted A/B Experimentation at Scale](https://arxiv.org/abs/2611.04515) | 该论文总结了构建和运营大规模AI辅助A/B实验智能体的经验与陷阱，核心结论是通过限制LLM非确定性的作用范围（模型负责选择与编排流程，确定性代码负责计算）来化解LLM不可靠性与实验系统可信度之间的矛盾。 |
| [^331] | [dKV Forcing: Separating Generation and Memory Representations through a Differentiable KV Cache](https://arxiv.org/abs/2611.01648) | 提出dKV Forcing方法，为KV缓存写入分配独立的低秩残差参数，实现生成与记忆表征的解耦，并利用后续片段读取的梯度来优化缓存写入，从而改进自回归视频生成器的训练。 |
| [^332] | [Lightweight Representation Learning for Low Tucker Rank Tensors](https://arxiv.org/abs/2611.01168) | 本文提出了一种保持迭代轨迹不变的轻量化计算框架，通过仅计算更新所需的奇异三元组、避免Kronecker乘积的收缩以及稀疏加因子存储三个组件，显著降低了五种低Tucker秩张量迭代估计器的内存开销。 |
| [^333] | [Sparse Autoencoder Attribution Graphs Are Decided at the Margin](https://arxiv.org/abs/2611.00606) | 该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。 |
| [^334] | [Demystifying Prompt Stealing Attacks against Synthetic Video](https://arxiv.org/abs/2611.00102) | 该研究提出V-PSA方法，证明单纯恢复文本提示无法忠实重建合成视频，通过结合恢复的字幕与DDIM反演构建参考轨迹，实现对公开合成视频的高保真重建与受控编辑。 |
| [^335] | [Intervenable Concept Dynamics for Video Activity Forecasting](https://arxiv.org/abs/2611.00020) | 本文提出TRACE-CBM框架，利用稀疏时空图细化观测概念激活并递归预测未来概念状态演化，首次实现了基于可编辑命名概念状态的视频未来活动预测。 |
| [^336] | [Ramanujan Challenge Results: AI-Assisted Evaluation Pipeline for Multi-Format Proofs](https://arxiv.org/abs/2610.40551) | 本文分析了拉马努金挑战赛（研究级数论基准）的提交结果，发现AI能够证明已公开的猜想并给出与官方解答概念迥异的正确证明，并提出了一种AI辅助的评估流水线，以可靠地大规模评判自然语言、CAS代码等多格式的复杂证明。 |
| [^337] | [Atom as an Agent: Emergent Order by Collective Predictive Coding](https://arxiv.org/abs/2610.36673) | 该论文将语言符号涌现的“集体预测编码”理论引入物理学，把原子视为相互预测的智能体，仅凭观测到的自旋构型就能自动涌现出伊辛模型、条纹序模型和自旋玻璃系统中的宏观序参量，为宏观秩序的起源提供了新解释。 |
| [^338] | [ResCom: Activation Residual Compensation for Large Language Model Quantization](https://arxiv.org/abs/2610.35955) | ResCom提出一种低秩激活残差补偿方法，通过输出感知校准以闭式形式利用激活舍入残差校正层输出误差，无需梯度训练即可显著提升大语言模型4比特权重-激活量化的精度。 |
| [^339] | [Ranking Function Generation with LLMs for Termination](https://arxiv.org/abs/2610.35523) | 该论文提出RAINFALL神经符号框架，通过将排序函数合成转化为带槽位约束的代码补全任务，并结合符号验证器的反馈迭代改进，利用大语言模型自动生成可验证的排序函数以证明循环终止性。 |
| [^340] | [EvolveMAS: A Unified Self-Evolution Framework for Decentralized Multi-Agent Systems](https://arxiv.org/abs/2610.34450) | EvolveMAS通过为每个主智能体配备专用MemoryAgent进行测试时记忆治理，在隔离的结构化记忆支持下，实现了去中心化多智能体系统在合作、混合动机和对抗等多种交互环境中的统一测试时适应与可靠经验复用。 |
| [^341] | [PEAR: Adaptive Advantage Reshaping for Perception and Exploration in Large Vision-Language Models](https://arxiv.org/abs/2610.32216) | PEAR提出一种自适应优势重塑方法，以视觉依赖为基础信号、仅在高熵位置施加与视觉依赖高低挂钩的熵探索增益，从而同时提升大型视觉语言模型多模态推理中的感知与探索能力。 |
| [^342] | [A Mathematical Theory of the Jacobian Lens](https://arxiv.org/abs/2610.31955) | 本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。 |
| [^343] | [One Evaluation Is Enough: Inversion-Free Image Editing with Trajectory Correction](https://arxiv.org/abs/2610.30827) | 提出端点校准传输（ECT），一种免训练、闭式求解的无反演图像编辑方法，仅需单次场评估即可通过“先传输、后校正”的构造实现可靠的图像编辑。 |
| [^344] | [Beyond Static Leaderboards: Capability-Focused Evaluation with Structural Diversity for Time Series Foundation Models](https://arxiv.org/abs/2610.26251) | 该论文提出CaFE评估框架，通过八个结构特征对原始基准数据进行重采样以引入结构多样性，将时间序列基础模型的评估从静态单一分数排行榜升级为可刷新的多维能力导向评估，从而更真实地反映模型的泛化能力。 |
| [^345] | [ReTRACE: Feedback-Guided LLM Agent in a Chemistry Harness for Retrosynthesis Planning](https://arxiv.org/abs/2610.24744) | 该论文提出ReTRACE智能体，将大语言模型嵌入集成分子工具、预测引擎、构建模块库和多标准评估器的确定性化学框架中，通过两个时间尺度的框架反馈学习进行逆合成规划，并引入结合程序化检查与对抗性LLM审查的Audit@k指标来评估路线的化学有效性。 |
| [^346] | [Red-Teaming Multi-Agent Attacks for Loss-of-Control Risks](https://arxiv.org/abs/2610.24052) | 该论文提出MiniLab多智能体攻击测试平台，证明分布式多智能体攻击可通过分工协作逃避仅单独审查各智能体轨迹的前沿监控器（仅检测出3.4%的成功攻击），揭示了现有AI控制措施在应对合谋性失控风险方面的重大盲区。 |
| [^347] | [FlashReFocus: Interactive Image Refocusing in Seconds](https://arxiv.org/abs/2610.23297) | FlashReFocus 对每张照片仅运行一次扩散模型来恢复全清晰图像，再用基于显式散焦图条件的轻量级网络渲染每次编辑，从而实现秒级的交互式图像重新对焦。 |
| [^348] | [Just Respect the Semantics: Image-Token Compression via Pre-LLM Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning](https://arxiv.org/abs/2610.18690) | Varest通过语义引导的可变分辨率token化与相关性token剪枝，仅用5.56%的图像token即可保留86%-92%的全token性能，实现高效且语义无损的图像token压缩。 |
| [^349] | [Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes](https://arxiv.org/abs/2610.18356) | Urbex提出一个智能体式视觉-语言框架，将城市级3D定位从候选排序重新建模为主动空间搜索，通过强化学习优化工具使用策略，在CityRefer上取得最佳性能且速度显著快于排序基线。 |
| [^350] | [SPLICE: Sparse Physics Library Identification and Compositional Expansion](https://arxiv.org/abs/2610.17029) | SPLICE是一个无需机制标签即可从多机制混合数据中自动发现物理算子的框架，通过凸优化推断贡献、校准测试分配机制，实现算子库的自动扩展并组合成零样本模拟器。 |
| [^351] | [CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use](https://arxiv.org/abs/2610.09127) | CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。 |
| [^352] | [Convex-Concave Reinforcement Learning](https://arxiv.org/abs/2610.09108) | 该论文揭示强化学习的策略优化问题在对数密度比坐标下具有凸差规划结构，从而将 CPI、NPG、TRPO 和 AWR 统一为特例，突破了传统凸代理近似的局限。 |
| [^353] | [Motion Without State: Probing Dynamical Consistency and Conditioning Bias in Video Models](https://arxiv.org/abs/2610.05135) | 该研究发现图像到视频模型的生成运动缺乏物理状态的前向传播，仅由局部视觉线索和全局边界控制构成，这种动力学一致性的系统性缺失会显著偏置定量物理评估的结果。 |
| [^354] | [InvestigationWorlds: An Agentic Environment for Legal Investigation](https://arxiv.org/abs/2610.04129) | 该论文利用真实的美国联邦法院简易判决动议构建了法律调查智能体环境InvestigationWorlds，评估发现智能体即使检索到相关证据也常常坚持错误假设，难以区分法院采纳的假设与其他备选假设。 |
| [^355] | [Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation](https://arxiv.org/abs/2610.02300) | 本文揭示文本到图像生成中全局不安全防护存在覆盖与选择性的内在权衡，并提出免训练的CALM方法，通过提示词局部的反事实校正精准编辑违规词元表示并抑制不安全残余成分，在不损害良性提示词的前提下有效提升安全性。 |
| [^356] | [The Backdrop Exposes What the World Around an Agent Costs It](https://arxiv.org/abs/2609.38469) | BACKDROP基准通过在智能体执行环境中植入权威覆盖、提示注入、边界越权和写入故障四种日常干扰，并保持任务指令不变，揭示了智能体性能在动态真实环境中从69.5%骤降至31.3%的严重衰减。 |
| [^357] | [ContextAdapt: Evaluating Contextual Adaptation and Value Alignment in LLMs](https://arxiv.org/abs/2609.38260) | 提出了ContextAdapt评估框架，通过基于职业与监管一手文件构建的“价值观×领域”场景，考察12个大语言模型能否在医学、法律、金融和国家安全等领域恰当调整诚实、自主、保密等价值观的应用，并在规范未变时保持一致。 |
| [^358] | [Diagnosing and Improving Probabilistic Reasoning in Large Language Models](https://arxiv.org/abs/2609.38005) | 该论文提出一个决策理论框架，将大语言模型的决策损失分解为信念形成与行动选择两个组成部分，并通过合成基准和强化学习干预系统诊断并改进大语言模型的概率推理能力。 |
| [^359] | [Resolving State-Representation Mismatch: State-Space Visual Reasoning for Open-Loop VLA Planning](https://arxiv.org/abs/2609.33412) | 提出状态空间视觉推理（SSVR），通过将静态视觉上下文、语言约束与由动作条件GRU更新的循环潜在状态解耦，解决了开环VLA规划中的状态表示失配问题，显著提升了长期规划的可靠性。 |
| [^360] | [Few-Shot Demonstrations Elicit the Use of In-Context World Representations in LLMs](https://arxiv.org/abs/2609.24352) | 少样本演示能够激发大语言模型在图追踪任务中构建并利用上下文世界表征，从而显著提升其预测性能。 |
| [^361] | [RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking](https://arxiv.org/abs/2609.23457) | 本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。 |
| [^362] | [Designer-RSI: Evolving Procedural Memory from User Traffic for Agentic Graphic Design](https://arxiv.org/abs/2609.22086) | 该论文提出Designer-RSI持续适应框架，让冻结的前沿模型通过230多个工具操作专业设计软件，并利用外部程序性记忆从真实用户流量中不断积累和精炼自然语言设计技能，在无权重更新、无人工标注的情况下显著提升智能体图形设计能力。 |
| [^363] | [SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190) | 该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。 |
| [^364] | [PointEvent: Rethinking Event-based Tiny Object Detection via Serialized Motion Evidence Accumulation](https://arxiv.org/abs/2609.20066) | 提出PointEvent框架，通过序列化运动证据积累将运动连续性建模为有序的证据传播过程，有效解决事件相机在微小无人机检测中远距离目标事件稀疏碎片化、易被杂波淹没的问题。 |
| [^365] | [Exploring Diffusion Transformers for Cross-Modal Augmentation in Multimodal Brain State Decoding](https://arxiv.org/abs/2609.11341) | 本文提出CoMA-DiT，一种双向跨模态扩散Transformer，将成对模态视为相互生成监督的来源以实现潜在空间数据增强，在多模态脑状态解码任务上显著优于20个代表性基线方法。 |
| [^366] | [Support Topology and Gradient Mixing in Sinkhorn Layers](https://arxiv.org/abs/2609.07954) | 本文发展了一套固定支撑演算来分析Sinkhorn层中的梯度传播机制，并证明了传输多面体的每个可行面具有成对两跳列重叠是支撑图与边际保证梯度均匀一步收缩的充要条件。 |
| [^367] | [Constrained Online Learning with Noisy Constraint Values](https://arxiv.org/abs/2609.06921) | 针对约束值受噪声污染的约束在线凸优化，本文证明了联合次线性保证的不可能性，并提出LEDGER算法，通过引入预算违反新度量，实现了 $O(\sqrt T/V)$ 的期望遗憾和可控的预算违反保证。 |
| [^368] | [From Synthetic Priors to Model Behavior: Structural Coverage in Tabular Foundation Models](https://arxiv.org/abs/2609.06912) | 该论文从分布层面归因视角出发，通过结构覆盖度和归一化密度量化表格基础模型的合成预训练先验对下游基准任务的支持范围，并探究这种结构支持与预测性能之间的关联。 |
| [^369] | [ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in Frozen JEPA World Models](https://arxiv.org/abs/2609.05461) | 该论文提出ARC-Bench无泄漏审计协议，揭示冻结JEPA世界模型中“潜在距离可正确排序动作”的核心假设严重且结构性失效，且该缺陷在不同视觉骨干网络下均持续存在。 |
| [^370] | [Untangling the Mechanisms of Misleading Context in Medical Question Answering](https://arxiv.org/abs/2609.02754) | 该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。 |
| [^371] | [AutoXRD: Autonomous LLM Agents and Comprehensive Evaluation for Powder Diffraction Analysis](https://arxiv.org/abs/2609.00070) | 本文提出AutoXRD——一个将粉末XRD分析组织为基于证据的逐步精修并加入确定性晶体学物理检查的自主大语言模型智能体框架，同时推出包含100个诊断问答任务和34个端到端工作流的XRDBench基准，用于全面评估智能体的XRD分析能力。 |
| [^372] | [Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations](https://arxiv.org/abs/2608.29434) | 该论文证明JEPA世界模型的潜空间规划可以从图像成功扩展到点云等几何观测——三种经典JEPA设计在点云上均能无坍塌地规划，其中动作敏感模型在几何变化最大的场景中表现最佳。 |
| [^373] | [A Table Is Worth 64 Tokens: Pixel-level Compression for Multi-Table Document Question Answering](https://arxiv.org/abs/2608.26949) | 本文发现像素级表格压缩在原生分辨率下性能与文本相当，但高度缩小后仍可识别表格相关性，据此提出无需训练的两步法以提高多表格文档问答效率。 |
| [^374] | [Depth Enables Local Entropy: Quadratic Depth Dependence in Deep Variation-Norm ReLU Regression](https://arxiv.org/abs/2608.17434) | 本文证明了在深度变分范数ReLU回归中，深度平方依赖是本质的，并通过局部填充和平衡放大技术给出了显式下界，填补了上下界之间的深度因子差距。 |
| [^375] | [DiSCO: Defending text-to-image generation through distribution-guided contrastive prompt optimization](https://arxiv.org/abs/2608.17067) | DiSCO提出了一种零样本、严格黑盒的提示级防御方法，通过分布引导的对比优化解决良性对抗问题，无需模型内部访问即可防止NSFW内容生成。 |
| [^376] | [GEM: A Generative Embedding Model Bridging Reasoning and Retrieval](https://arxiv.org/abs/2608.13200) | GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。 |
| [^377] | [Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment](https://arxiv.org/abs/2608.11537) | 提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。 |
| [^378] | [Beyond Statistical Priors: Architecture-Agnostic Data-Free Knowledge Distillation via Unified Explicit Semantics](https://arxiv.org/abs/2608.09287) | 该论文提出UniDFKD框架，用显式的架构无关语义先验取代传统DFKD依赖的批归一化统计等架构特定先验，使无数据知识蒸馏能够有效应用于ViT等现代架构。 |
| [^379] | [The Transformer Revolution, Part 1: Dynamic Processing through Output-Weight Interconnections](https://arxiv.org/abs/2608.03921) | 该论文以机械结构视角重新解读Transformer架构，揭示推理时部分网络的输出会动态决定其他网络的权重，由此产生随提示长度线性增长、甚至可超过静态参数数量的动态参数，以及序列级交互式动态并行处理（SIDPP）机制。 |
| [^380] | [Risky Business: Measuring The Faithfulness-Safety Tension](https://arxiv.org/abs/2608.03745) | 本文揭示了大型推理模型中“必须忠实以便被监控”与“必须稳健以拒绝不安全推理”之间的对齐张力，并提出 HazMart 数据集和目标推理替换（TRR）技术来直接干预推理链以测量和应对这一张力。 |
| [^381] | [CrossProjection: Benchmarking Spatial Grounding across Projection and Section Views](https://arxiv.org/abs/2608.00473) | 该论文提出了CrossProjection基准，利用建筑图纸测试视觉语言模型在跨视图匹配、投影配准和新剖面预测等空间定位任务上的能力，并首次引入了接受专家裁定的兼容位置带内任意剖面切线的逆向剖面定位评估协议。 |
| [^382] | [Parameter-Free Dynamic Regret under Heavy-Tailed Noise](https://arxiv.org/abs/2607.27073) | 该论文提出了一种无需预知噪声矩参数、Lipschitz 界、梯度尺度及比较序列路径长度的在线凸优化算法，通过重启 AdaGrad 专家与自适应熵正则化主算法的结合，在重尾噪声下实现了具有噪声路径指数 $(p-1)/p$ 的最优通用动态遗憾界。 |
| [^383] | [Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting](https://arxiv.org/abs/2607.26643) | SkillBoost通过结构化利用和先验引导探索的受约束框架，有效平衡探索与利用，从而缓解LLM代理技能自我进化中的过拟合问题。 |
| [^384] | [Pramana: A Composable, Domain-Specific Backend for Empirical Networking Research](https://arxiv.org/abs/2607.26352) | 论文提出Pramana——一个采用细腰架构（上层为研究意图、下层为执行基底）的可组合网络研究后端，旨在按需自动为每个新假设生成实验数据，以弥合AI快速提出假设与人工搭建实验之间的“构思到数据鸿沟”。 |
| [^385] | [A Few Teacher Steps Go a Long Way: Cost-Efficient On-Policy Data Augmentation for Agent Post-Training](https://arxiv.org/abs/2607.04574) | 该论文将智能体后训练中的同策略数据构建形式化为一个预算分配问题，系统权衡教师演示、续写长度、过滤规则与上下文覆盖等设计维度，发现少量教师步骤即可带来显著收益，从而实现高性价比的数据增强。 |
| [^386] | [PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding](https://arxiv.org/abs/2606.31148) | 提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。 |
| [^387] | [LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search](https://arxiv.org/abs/2606.25039) | LLM-ACES提出了一种由大语言模型引导的闭环方程发现框架，通过LLM生成算子先验来缩小符号搜索空间，并联合优化假设构建与自适应数据采集，利用候选方程间的分歧指导新数据获取，从而消除可辨识性缺口、准确恢复动力系统的控制方程。 |
| [^388] | [FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning](https://arxiv.org/abs/2606.24231) | FlowR2A将基于仿真的奖励从判别式目标重新定义为生成式条件，利用流匹配解码器学习奖励条件下的动作分布，在单一生成模型中统一了密集奖励监督与动态候选轨迹生成，从而提升多模态驾驶规划的效果。 |
| [^389] | [Benchmarking Open-Ended Multi-Agent Coordination in Language Agents](https://arxiv.org/abs/2606.08340) | 提出了基于 JAX 的开放式多智能体协作基准 Alem，将程序化协作任务、软专业化和通信嵌入长时程生存世界中，评估显示当前 LLM 智能体平均归一化回报仅约 6%，远未解决该基准。 |
| [^390] | [Emergent Language as an Approach to Conscious AI](https://arxiv.org/abs/2606.06380) | 提出一种生成式方法论：让多智能体强化学习中从零开始的智能体仅凭任务压力自发发展出交流语言，从而在摆脱人类语言先验干扰的条件下研究生成意识相关结构的涌现。 |
| [^391] | [FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences](https://arxiv.org/abs/2606.03330) | 本文提出面向AI监管的“实例级指纹识别”新范式及方法FLIPS，通过分析模型生成的二进制伪随机序列中的偏差，来区分同一大语言模型在不同实例级配置（如提示词、采样设置、量化方式）下的部署实例。 |
| [^392] | [Structured Prompt Optimization Advances Globally and Locally Interpretable Learning from Complex Text](https://arxiv.org/abs/2605.29076) | eXTC 通过结构化提示优化学习自然语言规则手册（SOP），结合推理蒸馏与强化学习，实现了兼具快速推理、局部推理轨迹与全局模块化可解释性的高性能文本分类。 |
| [^393] | [Review Arcade: On the Human Alignment and Gameability of LLM Reviews](https://arxiv.org/abs/2605.28897) | 该论文通过对ACL Rolling Review录用与拒稿投稿的实证实验发现，LLM审稿意见与人类审稿的对齐程度有限且因提示词和模型而异，同时作者可以通过迭代修改论文的方式有效“博弈”LLM审稿系统。 |
| [^394] | [A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks](https://arxiv.org/abs/2605.28556) | 本文提出 TASTE 方法，通过颠倒传统任务构建流程，利用在 LLM 判定有效性信号上训练的自适应对比 n-gram 模型采样覆盖广泛工具组合的有效工具序列，再经聚类选取代表性序列并实例化，从而自动合成覆盖度更广、难度更高的智能体基准测试任务。 |
| [^395] | [MobileMoE: Scaling On-Device Mixture of Experts](https://arxiv.org/abs/2605.27358) | 提出MobileMoE端侧MoE语言模型系列，通过端侧MoE缩放定律确定内存与计算同时最优的架构（适度稀疏+细粒度专家+共享专家），在十亿以下激活参数规模上为端侧大语言模型建立了新的帕累托前沿。 |
| [^396] | [Grounding Text Embeddings in Stakeholder Associations](https://arxiv.org/abs/2605.27168) | 该论文提出“利益相关者锚定练习”方法以评估文本嵌入与人类专家语义关联的对齐程度，发现神经文本嵌入的可靠性显著低于人类专家（差距18-27个百分点），且该错位会传播到下游聚类任务，并在英语和美国联邦AI用例中得到复现。 |
| [^397] | [Oracle-Budgeted Protein Design with Stochastic Search and Iterative Imitation](https://arxiv.org/abs/2605.26690) | SILO是一种轨迹级蛋白质设计方法，结合层次化编辑策略与随机束搜索探索多样化突变轨迹，利用UCB代理集成与丙氨酸扫描式适应度评分筛选候选序列，并通过交叉熵模仿学习迭代自我改进，在严格的预言机评估预算下高效优化蛋白质序列。 |
| [^398] | [NPSolver: Neural Poisson Solver with Iterative Physics Supervision](https://arxiv.org/abs/2605.25786) | NPSolver 提出了一种无需解标签的神经泊松求解器，通过少量预条件共轭梯度（PCG）迭代步精炼自身预测作为迭代物理监督，获得了比原始 PDE 残差更稳定、尺度更合适的训练信号。 |
| [^399] | [Flash PD-SSM: Memory-Optimized Structured Sparse State-Space Models](https://arxiv.org/abs/2605.19150) | 提出了一种新颖的状态空间模型PD-SSM，通过在每个时间步从一组可训练的结构化稀疏矩阵中离散选择一个矩阵，在保持与主流结构化SSM相当的吞吐量的同时，实现了与非结构化转移矩阵相当的表达能力。 |
| [^400] | [Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty](https://arxiv.org/abs/2605.14494) | 本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。 |
| [^401] | [Training-Free Cultural Alignment of Large Language Models via Persona Disagreement](https://arxiv.org/abs/2605.10843) | 提出DISCA方法，无需训练、无需特定国家的数据或访问模型内部，仅通过基于世界价值观调查数据构建的四角色提示面板的共识与分歧机制，即可将权重冻结的大语言模型对齐到不同国家的道德偏好。 |
| [^402] | [M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models](https://arxiv.org/abs/2605.09879) | 该论文提出M2A新范式，通过在参数空间中识别智能体行为的关键特征子空间并进行模型合并，避免了联合训练中的过拟合问题，从而有效协同大语言模型中的数学推理与智能体推理能力。 |
| [^403] | [Fairness of Explanations in Artificial Intelligence (AI): A Unifying Framework, Axioms, and Future Direction toward Responsible AI](https://arxiv.org/abs/2605.09852) | 本文首次提出了研究AI解释公平性的统一理论框架与公理体系，揭示了模型即使输出满足公平性标准，其推理过程仍可能存在“程序性偏见”，并为迈向负责任的人工智能指明了未来方向。 |
| [^404] | [Agent Collectives Should Not Detect Their Own Imposters: A Chess Case Study](https://arxiv.org/abs/2605.09027) | 研究表明让智能体集体自行检测内部冒名顶替者会适得其反，而采用能够从少量样本快速适应新型攻击的轻量级元训练外部检测器才是更优策略。 |
| [^405] | [Text-to-CAD Evaluation with CADTests](https://arxiv.org/abs/2605.07807) | 该论文提出了首个基于可执行测试的文本转CAD评估基准CADTestBench，通过CADTests验证生成模型是否满足几何与拓扑要求，并证明这些测试还能指导CAD生成，使简单基线超越现有方法。 |
| [^406] | [Experiment-as-Code Labs: A Declarative Stack for AI-Driven Scientific Discovery](https://arxiv.org/abs/2605.04375) | 该论文提出“实验即代码”实验室新范式，借鉴计算机系统的理念将实验编码为声明式配置，从而弥合日益强大的AI智能体与自动化实验室设备之间的鸿沟，使AI驱动的科学发现能够掌控真实世界的物理实验室。 |
| [^407] | [Graph Construction and Matching for Imperative Programs using Neural and Structural Methods](https://arxiv.org/abs/2604.26578) | 该论文提出一条将命令式程序及其标注转换为带类型属性图的流水线，通过结合抽象语法树解析与SentenceTransformer、CodeBERT等模型的语义嵌入，在C/ACSL、Java/JML和Dafny等多种语言上实现了跨语言、跨标注风格的一致性图表示，为验证产物复用和近似图匹配奠定了实用基础。 |
| [^408] | [SIEVES: Selective Prediction Generalizes through Visual Evidence Scoring](https://arxiv.org/abs/2604.25855) | 该论文提出SIEVES方法，通过要求模型提供图像裁剪作为视觉证据并对其进行评分来估计置信度，从而提升多模态大语言模型选择性预测在分布外场景中的泛化能力。 |
| [^409] | [Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models](https://arxiv.org/abs/2604.13991) | 提出了一种提示词自适应的共形预测方法，通过提示词相关的校准提升大语言模型生成内容的事实性，在保持边际覆盖保证的同时改善条件覆盖率，并支持选择性预测以过滤不可靠输出。 |
| [^410] | [Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding](https://arxiv.org/abs/2604.00528) | 该论文提出TAB动态智能体框架，将零样本3D视觉定位任务重构为直接作用于原始RGB-D数据流的生成式2D到3D重建范式，通过VLM智能体动态调用视觉工具进行跨帧跟踪与重建，从而摆脱了对预处理点云的依赖。 |
| [^411] | [AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection](https://arxiv.org/abs/2603.17530) | AdapTS提出了一种轻量级师生框架，通过共享冻结骨干网络和基于原型识别的轻量适配器选择机制，无需重放缓冲区即可实现多类别与持续视觉异常检测，在大幅降低内存占用的同时保持优异性能。 |
| [^412] | [STAIRS-Former: Spatio-Temporal Attention with Interleaved Recursive Structure Transformer for Offline Multi-task Multi-agent Reinforcement Learning](https://arxiv.org/abs/2603.11691) | 本文提出STAIRS-Former，一种具有空间和时间层次结构的Transformer架构，通过对关键标记的有效注意力和标记丢弃机制，捕捉长交互历史并泛化到不同智能体数量的场景，从而提升离线多任务多智能体强化学习的性能。 |
| [^413] | [Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue](https://arxiv.org/abs/2603.11409) | 该论文提出“上下文感知话轮转换”新任务——在多方对话的停顿处判断AI助手应发言还是沉默，发现八个现有大语言模型零样本下均无法胜任，并借助带推理轨迹的监督微调将平衡准确率提升最多23个百分点。 |
| [^414] | [SoundWeaver: Compositional Warm-Starting for Text-to-Audio Diffusion Serving](https://arxiv.org/abs/2603.07865) | SoundWeaver是首个无需训练的组合式跨示例声学复用框架，通过从缓存音频片段构建提示条件化声学先验并热启动扩散过程，显著加速文本到音频扩散模型的推理。 |
| [^415] | [VI-CuRL: Stabilizing Verifier-Independent Reinforcement Learning Reasoning via Confidence-Guided Variance Reduction](https://arxiv.org/abs/2602.12579) | 提出VI-CuRL框架，利用模型内在置信度构建课程并优先处理高置信度样本，有效降低梯度方差，从而解决无验证器强化学习推理中的训练崩溃问题。 |
| [^416] | [Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings](https://arxiv.org/abs/2602.09985) | 该论文提出了一种基于JEPA自监督嵌入的在线监测框架，无需异常标签即可检测汽车目标状态表示中的异常，为自动驾驶车辆的安全运行监控提供了新方案。 |
| [^417] | [SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration](https://arxiv.org/abs/2602.04361) | SparVAR是一种免训练的VAR模型加速框架，通过利用注意力汇聚点、跨尺度激活相似性和局部性三大特性，从稀疏决策尺度动态预测高分辨率尺度的稀疏注意力模式，在不跳过尺度、不牺牲图像质量的前提下大幅降低计算复杂度和推理延迟。 |
| [^418] | [FlowSteer: Towards Agents Designing Agentic Workflows via Reinforced Progressive Canvas Editing](https://arxiv.org/abs/2602.01664) | FlowSteer 提出了一种新范式，由单个轻量级智能体在可执行的工作流画布环境中，通过强化学习训练的渐进式原子画布编辑，端到端地自主设计供下游执行器运行的智能体工作流，并获得语法检查级别的实时执行反馈。 |
| [^419] | [Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents](https://arxiv.org/abs/2601.17829) | 该论文提出了一种无需手工规则或分类体系、通过在查询和参数两个层面优化通用多样性指标来生成函数调用智能体合成训练数据的方法，在保持正确性相当的同时显著提升了数据的语言与参数多样性。 |
| [^420] | [Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Highlights in Videos](https://arxiv.org/abs/2601.16333) | 本研究利用足球比赛官方集锦零成本构建了一个评估数据集，用以测试多模态基础模型识别视频中重要事件的能力，发现现有最先进模型无法可靠地理解事件的上下文重要性。 |
| [^421] | [Multi-Dimensional Prompt Chaining to Improve Open-Domain Dialogue Generation](https://arxiv.org/abs/2601.01037) | 提出一种整合自然度、连贯性和吸引力三个维度的多维提示链框架，使小语言模型（如Llama-2-7B）在开放域对话生成中达到与大型模型相当的性能。 |
| [^422] | [Taming Hallucinations: Boosting MLLMs' Video Understanding via Counterfactual Video Generation](https://arxiv.org/abs/2512.24271) | 该论文提出DualityForge框架，利用可控扩散视频编辑自动合成反事实视频与问答数据对，通过对比训练抑制多模态大语言模型对语言先验的过度依赖，从而减少视觉幻觉并增强视频理解能力。 |
| [^423] | [The Dead Salmons of AI Interpretability](https://arxiv.org/abs/2512.18792) | 该论文借神经科学中著名的“死鲑鱼”事件警示当前AI可解释性方法（如特征归因、探测、稀疏自编码）可能对随机初始化的神经网络生成虚假解释，并提出应将解释视为从计算轨迹中推断的统计模型参数这一统计-因果重构框架。 |
| [^424] | [BEAVER: An Efficient Deterministic LLM Verifier](https://arxiv.org/abs/2512.05439) | BEAVER是首个实用的确定性LLM验证框架，利用令牌字典树和边界前沿系统地探索模型输出分布，为LLM响应满足特定属性的概率提供可证明的可靠界限，其Max-μ选择策略在任意容差下均是最优的，在安全任务上认证有风险提示的数量比采样验证器多出高达7.6倍。 |
| [^425] | [MemVerse: Multimodal Memory for Lifelong Learning Agents](https://arxiv.org/abs/2512.03627) | 该论文提出了MemVerse——一个模型无关、即插即用的多模态记忆框架，通过短期记忆、结构化长期记忆图和周期性知识蒸馏机制，使终身学习智能体能够实现快速召回、持续学习与自主知识演化。 |
| [^426] | [MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning](https://arxiv.org/abs/2511.21460) | MADRA提出了一种无需训练的多智能体辩论认知架构，通过元认知批判智能体和基于图尔敏模型的结构化论证来缓解大语言模型的“具身语义幻觉”、避免安全-效用权衡，并发布了SafeAware-VH基准用于评估具身智能体的物理风险感知能力。 |
| [^427] | [Binding Constraints in a Strategic Technology: An Industrial Vulnerability Index for the Foundation Model Industry](https://arxiv.org/abs/2510.23421) | 本文提出人工智能工业脆弱性指数（AIIVI），利用十四个集中度与可持续性参数并通过 O-Ring 函数形式聚合，用以评估基础模型产业中最具约束性的生产瓶颈，揭示了当前产业政策聚焦算力而忽视电网等环节的配置可能偏离真正的约束所在。 |
| [^428] | [Repairing Reward Functions with Feedback to Mitigate Reward Hacking](https://arxiv.org/abs/2510.13036) | 该论文提出了基于偏好的奖励修复（PBRR）自动化迭代框架，通过从少量人类偏好数据中学习一个加性的、依赖于状态转移的修正项来修复人工指定的代理奖励函数，以低成本缓解奖励欺骗问题。 |
| [^429] | [Efficient Prediction of pass@k Scaling in Large Language Models](https://arxiv.org/abs/2510.05197) | 本文提出了一种稳健的估计框架，能够在远小于大规模尝试次数的采样预算下高效准确地预测大语言模型的pass@k缩放行为，从而帮助模型提供商和监管机构评估模型的能力与风险。 |
| [^430] | [Rethinking Multiplication in Neural Networks with Hadamard Transform-Assisted Operators](https://arxiv.org/abs/2509.23103) | 本文提出Hadamard变换辅助的避免乘法（HMA）算子，并通过门控竞争机制发现中间卷积层更适合采用HMA算子，而最早层和最终分类器仍需传统乘法，从而在存内计算架构中兼顾计算效率与模型精度。 |
| [^431] | [Complexity-Regularized Proximal Policy Optimization](https://arxiv.org/abs/2509.20509) | 该论文提出用香农熵与非平衡度乘积构成的自调节复杂度项替代传统熵正则化来改进近端策略优化（PPO），在保持策略有益随机性的同时，当策略高度不确定时降低正则化压力，从而让学习更专注于奖励优化。 |
| [^432] | [Patient-Zero: Scaling Synthetic Patient Agents to Real-World Distributions without Real Patient Data](https://arxiv.org/abs/2509.11078) | 该论文提出Patient-Zero框架，无需任何真实患者数据，通过医学对齐分层合成从医学知识生成多样化患者记录，并借助双轨认知记忆系统在动态临床交互中保持逻辑一致性。 |
| [^433] | [One-Shot Clustering for Federated Learning Under Clustering-Agnostic Assumption](https://arxiv.org/abs/2509.01587) | 本文提出了一种聚类无关的一次性聚类联邦学习算法（OCFL），能够自动检测出最适合聚类的最早时机，从而实现个性化模型的训练。 |
| [^434] | [CrafterDojo: An Integrated Research Suite for Building Open-Ended Embodied Agents in Crafter](https://arxiv.org/abs/2508.13530) | CrafterDojo通过提供基础模型（CrafterVPT、CrafterCLIP、CrafterSteve-1）、合成数据集（CrafterPlay、CrafterCaption）及自动数据生成工具包，为轻量级环境Crafter构建了完整的基础模型生态，使其成为Minecraft之外研究开放式具身智能的高效测试平台。 |
| [^435] | [Sample-Aware Test-Time Adaptation for Medical Image-to-Image Translation](https://arxiv.org/abs/2508.00766) | 提出了一种样本感知的测试时自适应框架，通过重构模块量化域偏移，并利用动态自适应块选择性调整预训练模型的内部特征，在低剂量CT去噪和MRI跨模态转换任务中有效处理分布外样本，同时不损害分布内样本的性能。 |
| [^436] | [AIR: Rethinking Image-Text Offset Alignment in Multimodal Contrastive Representation Space](https://arxiv.org/abs/2506.10895) | 本文系统性地实证检验了 CLIP、SigLIP 等多模态对比表征空间中文本驱动的视觉变换方法所依赖的“图文偏移对齐”假设，质疑其有效性。 |
| [^437] | [Local Policy Manifolds for Efficient Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2506.04022) | 提出LLE-MORL方法，通过利用策略参数空间与性能空间之间的局部关系追踪策略流形，借助局部线性外推实现高效搜索，无需大量重新训练即可快速生成覆盖多目标权衡的高质量策略集合。 |
| [^438] | [Fortune: Formula-Driven Reinforcement Learning for LLM Reasoning over Complex Tables](https://arxiv.org/abs/2505.23667) | 提出Fortune，一种公式驱动的强化学习框架，以执行成功与答案正确性作为奖励，训练大语言模型生成可执行的电子表格公式，从而提升其在复杂表格上的数值推理能力。 |
| [^439] | [TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning](https://arxiv.org/abs/2505.16743) | TRIM提出一种行级迭代度量驱动剪枝方法，通过为层内每个输出行分配不同的稀疏比率，在高稀疏度（70-80%）下显著降低大语言模型的困惑度，例如使Qwen2.5-14B在80%稀疏度下的困惑度近乎减半。 |
| [^440] | [VFRTok: Variable Frame Rates Video Tokenizer with Duration-Proportional Information Assumption](https://arxiv.org/abs/2505.12053) | 本文提出基于“视频信息容量与时长而非帧数成正比”这一新假设的VFRTok可变帧率视频分词器，通过非对称帧率训练和部分旋转位置编码，显著降低了视频生成扩散模型的计算成本。 |
| [^441] | [Causal pieces: analysing and improving spiking neural networks piece by piece](https://arxiv.org/abs/2504.14015) | 提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。 |
| [^442] | [Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos](https://arxiv.org/abs/2503.20258) | 提出E-ViM³——一种保留视频3D结构的Vision Mamba网络，结合包围全局标记和时空链式掩码策略进行自监督预训练，实现了医学超声视频的精确且数据高效的分析。 |
| [^443] | [On the Efficiency of Structured Pruning in Small Language Model Pretraining](https://arxiv.org/abs/2503.05920) | 提出IDEA Prune集成式放大-剪枝流水线，在单一余弦退火学习率调度下统一放大模型预训练、迭代结构化剪枝和恢复过程，在全成本核算下比传统分阶段剪枝方法更高效地训练出小语言模型。 |
| [^444] | [MMGenBench: An Automated Pipeline for Evaluating LMMs from a Text-to-Image Generation Perspective](https://arxiv.org/abs/2411.14062) | MMGenBench是一个无需预定义问题或参考描述的自动化基准框架，通过“图像描述生成→文生图重建→编码器比对”的流水线从原始图像评估大型多模态模型，对68个LMM的评估揭示了总分可能掩盖的模式特定重建质量差异。 |
| [^445] | [The necessity of depth for artificial neural networks to approximate certain classes of smooth and bounded functions without the curse of dimensionality](https://arxiv.org/abs/2301.08284) | 该论文证明了一类乘积函数及其正弦复合函数无法被浅层或深度不足的ReLU神经网络在避免维数灾难的前提下逼近，但可被足够深的ReLU神经网络避免维数灾难地逼近，从而首次严格揭示了网络深度对于此类光滑有界函数高维逼近的必要性。 |
| [^446] | [HindSight: A Graph-Based Vision Model Architecture For Representing Part-Whole Hierarchies](https://arxiv.org/abs/2104.03722) | 该论文提出了HindSight架构，通过将图像划分为不同层级的图像块并构建全连接图来编码图像中的部分-整体层次结构，可训练为通用视觉编码器以支持多种视觉下游任务。 |
| [^447] | [Randomized Forward Mode of Automatic Differentiation for Optimization Algorithms.](http://arxiv.org/abs/2310.14168) | 该论文介绍了一种随机前向模式自动微分优化算法，通过在神经网络的正向传递中计算损失函数的方向导数来更新参数。算法通过采样不同概率分布的随机方向，使用正向模式自动微分计算雅可比向量乘积，并提供了对其收敛速度和计算复杂性的严格分析。 |

# 详细

[^1]: OmniAct：一个策略，掌控百万具身形态

    OmniAct: One Policy, One Million Embodiments

    [https://arxiv.org/abs/2611.33158](https://arxiv.org/abs/2611.33158)

    OmniAct通过以机器人在具体场景中的运动学图（包含关节位置、运动轴及与视觉观测的连接关系）为条件，并用流匹配策略预测各节点的相对关节运动，实现了单一策略即可控制具有不同自由度和运动学配置的多种机器人具身形态。

    

    控制多种机器人的单一策略不仅需要在不同具身形态之间迁移任务知识，还必须根据每个机器人的具体能力调整其行为。现有方法通过共享的动作表示来促进这种迁移，但这些表示未必能捕捉机器人形态上的差异应如何塑造操作策略。我们提出OmniAct，这是一种跨具身形态的操作策略，它直接以特定场景中每个机器人的运动学信息为条件，生成针对该具身形态的行为。这种运动学图表示了机器人的关节位置、运动轴以及与其视觉观测之间的连接关系，使策略能够考虑其具身形态与场景中物体和表面之间的关系。一个流匹配（flow-matching）策略为每个节点预测相对关节运动，从而能够控制具有不同自由度和运动学配置的机器人。为了学习物理……（原文摘要在此处截断）

    arXiv:2611.33158v1 Announce Type: cross  Abstract: A single policy that controls diverse robots must not only transfer task knowledge across embodiments, but also adapt its behavior to each robot's capabilities. Existing approaches facilitate transfer through shared action representations, but these representations do not necessarily capture how differences in robot morphology should shape manipulation strategies. We introduce OmniAct, a cross-embodiment manipulation policy that conditions directly on each robot's kinematics in a specific scene to produce embodiment-specific behavior. This kinematic graph represents the robot's joint positions, motion axes, and connectivity to its visual observation, allowing the policy to account for how its embodiment related to objects and surfaces in the scene. A flow-matching policy predicts relative joint motions for each node, enabling control across robots with different degrees of freedom and kinematic configurations. To learn how physical str
    
[^2]: MetaFORGE：面向并行计算自动代码生成中测试框架优化的元学习框架

    MetaFORGE: A Meta-Learning Framework for Harness Optimization in Automated Code Generation for Parallel Computing

    [https://arxiv.org/abs/2611.33119](https://arxiv.org/abs/2611.33119)

    MetaFORGE是一个基于执行反馈的元学习框架，它将正确性测试、编译器诊断、静态分析、运行时测量和负载均衡数据整合为标量奖励，迭代优化自然语言技能规范，从而显著提升LLM生成MPI并行代码的质量、性能与负载均衡。

    

    大型语言模型（LLM）能够生成并行程序，但其正确性和性能在很大程度上取决于指导生成过程的指令。我们提出了MetaFORGE，这是一个以执行结果为依据的元学习框架，用于对MPI C代码生成的持久化自然语言技能规范进行迭代优化。MetaFORGE将正确性测试、编译器诊断信息、静态分析结果、运行时性能测量以及进程级负载不均衡概况融合为一个标量奖励，用以驱动文本梯度的精炼过程，同时通过外部记忆在多次运行之间提炼并检索有奖励支撑的优化知识。在六个MPI任务和三种工作负载规模的实验中，相较于零样本生成，MetaFORGE提升了静态分析质量、运行时性能和负载均衡水平，并且总体上超越了所评估的TextGrad迭代基线。一项模型匹配的精炼消融实验表明，MetaFO（摘要在此处被截断）

    arXiv:2611.33119v1 Announce Type: new  Abstract: Large language models (LLMs) can generate parallel programs, but their correctness and performance depend strongly on the instructions that guide generation. We present \textsc{MetaFORGE}, an execution-grounded meta-learning framework that iteratively optimizes persistent natural-language skill specifications for MPI C code generation. \textsc{MetaFORGE} combines correctness testing, compiler diagnostics, static-analysis findings, runtime measurements, and rank-level load-imbalance profiles into a scalar reward that drives textual-gradient refinement, while an external memory distills and retrieves reward-supported optimization knowledge across runs.   Across six MPI tasks and three workload sizes, \textsc{MetaFORGE} improves static-analysis quality, runtime, and load balance relative to zero-shot generation and generally exceeds the evaluated \textsc{TextGrad} iterative baseline. A model-matched refinement ablation shows that the MetaFO
    
[^3]: 面向推测解码的草稿校准：通过匹配草稿分布与目标分布加速大语言模型服务

    Draft Calibration for Speculative Decoding: Accelerating LLM Serving by Matching Draft and Target Distributions

    [https://arxiv.org/abs/2611.33111](https://arxiv.org/abs/2611.33111)

    该论文提出推测解码中的草稿校准方法，通过事后校准解决开源草稿模型相对目标模型普遍置信度不足的分布失配问题，缩小草稿分布与目标分布之间的差距，从而提高草稿 token 接受率并加速大语言模型推理服务。

    

    推测解码通过采用轻量级草稿模型提出草稿 token，并允许目标模型并行验证新 token，从而提高推理阶段的计算利用率，加速大语言模型（LLM）推理。其加速效果取决于草稿 token 在目标模型验证中的接受概率，而该概率与目标分布 $p$ 和草稿分布 $q$ 之间的差距 $\|p-q\|_1$ 直接相关。本文研究推测解码中的草稿校准方法，以提升推测解码的加速效果：我们采用事后校准方法对草稿分布进行校准，从而缓解分布差距。我们观察到，尽管当前社区开源的草稿模型可能正确学习到最优候选 token，但与目标模型相比它们普遍存在置信度不足的问题，这表明存在系统性的分布失配。为减轻这种失配对接受率的影响……（原文摘要在此处截断）

    arXiv:2611.33111v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model (LLM) inference by employing a lightweight draft model to propose draft tokens and allowing the target model to verify the new tokens in parallel to improve inference-time compute utilization. The speedup relies on the acceptance probability of draft tokens in target verification, which is directly related to the gap $\|p-q\|_1$ between the target distribution $p$ and the draft distribution $q$. In this paper, we study draft calibration of speculative decoding to improve the speedup of speculative decoding, where we employ post-hoc methods to calibrate the draft distribution to mitigate the distribution gap. We observe that though current community open-source draft models may learn the top candidate token correctly, they are generally underconfident compared to the target model, suggesting a consistent distribution mismatch. To mitigate the effect of this mismatch on the acceptance
    
[^4]: 国际象棋中人类行为克隆的时间条件化方法

    Temporal Conditioning for Human Behavioral Cloning in Chess

    [https://arxiv.org/abs/2611.33106](https://arxiv.org/abs/2611.33106)

    本文提出时间感知Transformer模型，将剩余时钟时间等时间信息编码进国际象棋行为克隆模型，在时间压力下显著提升了人类走法预测准确率，并能更准确地预测玩家的思考时间和认输行为。

    

    国际象棋中的行为克隆模型通常仅根据棋盘状态预测人类走法，而忽略了诸如剩余时间等时间输入。然而，时间压力是人类决策质量的主要决定因素，时间状态是人类行为建模中缺失的一个维度。本文提出了一种时间感知Transformer模型，它在Chessformer风格架构的基础上进行扩展，引入了可学习的时间条件化标记，用于编码玩家的时钟状态、时间控制和近期行棋节奏，并配备了用于预测思考时间和认输行为的行为预测头。该模型在大规模在线国际象棋数据上训练，并与时间无关的基线模型和当前最先进的模型进行了对比评估。结果表明，时间条件化带来了适度的整体准确率提升，但在时间压力下获得了显著更大的性能收益。除了走子选择之外，它还能产生更准确的思考时间和认输预测。更重要的是，该模型能够更好地……

    arXiv:2611.33106v1 Announce Type: new  Abstract: Behavioral cloning models in chess typically predict human moves from board states while ignoring temporal inputs such as the remaining clock. However, time pressure is a major determinant of human decision-making quality, and temporal state is a missing dimension of human behavioral modeling. This paper introduces a time-aware transformer that extends a Chessformer-style architecture with learned temporal conditioning tokens encoding player clock state, time control, and recent tempo, together with behavioral heads for thinking time and resignation. The model is trained on large-scale online chess data and evaluated against time-agnostic baselines and the state of the art. Results show that temporal conditioning yields modest overall accuracy improvements but substantially stronger gains under time pressure. Beyond move choice, it also produces more accurate thinking-time and resignation predictions. More importantly, the model better m
    
[^5]: Delegent：一个在不牺牲性能的前提下降低成本的多智能体框架

    Delegent: A Multi-Agent Harness That Reduces Cost without Sacrificing Performance

    [https://arxiv.org/abs/2611.33098](https://arxiv.org/abs/2611.33098)

    Delegent是一个基于Claude Code的多智能体框架，让前沿模型始终主导任务执行、仅将简单子任务委托给可在本地硬件运行的27B小型语言模型，从而无需训练路由器即可在不牺牲性能的情况下大幅降低智能体运行成本。

    

    基于前沿语言模型构建的智能体成本高昂。在每一步中，前沿模型都会重新读取其全部上下文并执行每一个子任务，而不考虑任务本身的难度。本文探讨了一个简单的论点：通过将简单的子任务委托给可以在用户自有硬件上运行的小型语言模型（SLM），可以大幅削减这部分成本。为了验证这一论点，我们构建了Delegent，这是一个基于Claude Code构建的简单而有效的智能体框架。在该框架中，前沿模型始终主导整个任务，但会将聚焦的子任务（如调查代码、运行测试和搜索语料库）交给SLM子智能体处理。现有方法（如级联和模型路由器）通常选择由哪个模型执行整个任务，而在Delegent中，前沿模型始终主导执行过程，仅将简单的子任务委托给SLM。无需训练任何路由器或模型，因此两个模型都可以自由替换。借助一个27B FP8的SLM，Delegent降低了……

    arXiv:2611.33098v1 Announce Type: new  Abstract: Agents built on frontier language models are expensive. At every step, the frontier model re-reads its entire context and executes every subtask regardless of its difficulty. This paper explores the simple thesis that much of this cost can be removed by delegating easy subtasks to a small language model, or SLM, that runs on hardware one already owns. To test it, we build Delegent, a simple and effective agent harness built on Claude Code, in which the frontier model stays in charge of every task but hands focused subtasks, such as investigating code, running tests, and searching a corpus, to SLM subagents. Existing approaches such as cascades and model routers usually choose which model executes the entire task, whereas in Delegent, the frontier model always leads the execution and delegates easy subtasks to the SLM. No router or model is trained, so either model can be swapped freely. With the help of a 27B FP8 SLM, Delegent reduces th
    
[^6]: EventDiffusion：从视频中学习原生事件流分布

    EventDiffusion: Learning Native Event Stream Distribution from Videos

    [https://arxiv.org/abs/2611.33096](https://arxiv.org/abs/2611.33096)

    EventDiffusion提出了一种层次化生成框架，通过事件活动扩散、事件空间分配和事件时序扩散三个概率过程，直接从真实记录中学习原生异步事件流的条件分布，从而从视频中生成多样且逼真的事件流。

    

    事件相机能够以高时间分辨率捕获异步的亮度变化，但稀缺的真实记录数据限制了数据驱动研究的发展。稀疏的RGB观测无法唯一地确定其间发生的事件，因为相同的帧可能对应于具有不同活动强度、极性和时序的事件流。现有的模拟器主要依赖预设的传感器动态模型和估计的亮度轨迹，这些假设限制了合理事件流的多样性和保真度。我们提出了EventDiffusion，一个层次化的生成框架，直接从真实记录中学习原生异步事件流的条件分布。EventDiffusion将生成过程分解为三个概率过程：事件活动扩散（EAD）学习随机的离散事件计数；事件空间分配（ESA）将这些计数分配到各个像素上并保持基数不变；事件时序扩散（ETD）学习……

    arXiv:2611.33096v1 Announce Type: cross  Abstract: Event cameras capture asynchronous brightness changes at high temporal resolution, but scarce recordings constrain data-driven research. Sparse RGB observations cannot uniquely determine intervening events since identical frames may correspond to streams with different activity, polarities, and timing. Existing simulators largely rely on prescribed sensor dynamics and estimated brightness trajectories, whose assumptions limit the diversity and fidelity of plausible event streams. We introduce EventDiffusion, a hierarchical generative framework that directly learns the conditional distribution of native asynchronous event streams from real recordings. EventDiffusion decomposes generation into three probabilistic processes: Event Activity Diffusion (EAD) learns stochastic discrete event counts; Event Spatial Allocation (ESA) distributes these counts to individual pixels while preserving cardinality; and Event Timing Diffusion (ETD) learn
    
[^7]: AIChipBench：面向自动化AI加速器设计的LLM智能体基准测试

    AIChipBench: Benchmarking LLM Agents for Automated AI Accelerator Design

    [https://arxiv.org/abs/2611.33093](https://arxiv.org/abs/2611.33093)

    该论文提出AIChipBench基准，要求LLM编程智能体用可综合RTL实现量化大模型解码器，重点考察跨解码步骤的状态保持正确性及面积受限下的性能优化，从而系统评估智能体在自动化AI加速器设计中的真实能力。

    

    语言模型解码器需要在解码步骤之间传递状态，而在硬件中，错误的状态更新可能导致当前token输出正确、却在之后若干个token处破坏结果，因此一个设计可能通过了输出测试却仍然存在错误。AIChipBench探究LLM编程智能体需要具备什么才能正确构建这类硬件。智能体需要用可综合的寄存器传输级（RTL）代码实现量化版的Qwen3.5、Gemma 3、Gemma 4和Llama 3.2解码器，并以一个可执行的整数参考实现作为规范，该参考规定了每一步解码的输出和存储状态。智能体在固定预算内针对公开测试进行调试；其冻结后的设计会在保留的token序列、复位、内存停顿以及综合等维度上评分；此外还有一条赛道要求智能体在面积几乎不变的前提下对一个正确设计进行加速。在四个编程智能体共9,136次尝试中，难点集中在存储状态上。指出首个状态不匹配位置的反馈能够显著提升设计通过率。

    arXiv:2611.33093v1 Announce Type: cross  Abstract: A language-model decoder carries state between decoding steps, and in hardware a wrong state update can leave the current token correct and corrupt the output several tokens later, so a design can pass output tests and still be wrong. AIChipBench asks what LLM coding agents need to build such hardware correctly. Agents implement quantized Qwen3.5, Gemma 3, Gemma 4, and Llama 3.2 decoders in synthesizable register-transfer-level (RTL) code against an executable integer reference that specifies the outputs and the stored state at every decoding step. They debug against public tests under fixed budgets; their frozen designs are graded on held-out token sequences, resets, memory stalls, and synthesis, and a second track asks them to speed up a correct design at nearly fixed area. Across 9,136 attempts by four coding agents, the difficulty lies in stored state. Feedback that points to the first state mismatch raises acceptance over output-o
    
[^8]: 面向网络威胁情报中实体链接的不确定性感知候选生成

    Uncertainty-Aware Candidate Generation for Entity Linking in Cyber Threat Intelligence

    [https://arxiv.org/abs/2611.33086](https://arxiv.org/abs/2611.33086)

    该论文提出了一种不确定性感知的实体解析方法，通过在NER阶段捕获片段边界与类型的不确定性信号并传递到EL阶段，从而在网络威胁情报场景下改进模糊情况下的候选实体检索。

    

    实体解析（ER）流水线通常在早期的硬性决策中就将模糊性压缩消除：上游的命名实体识别（NER）选择单个文本片段和类型，下游的实体链接（EL）则将该选择当作真实值。这种做法丢弃了关于底层数据模糊性的宝贵信息（例如片段边界不一致和类型混淆），导致系统变得脆弱，在面对输入中的不确定性时会不可预测地失败。因此，本工作提出了一个不确定性感知实体解析的概念验证方案，将不确定性视为应在整个ER流水线中予以保留和利用的原生数据。我们（i）在NER阶段捕获不确定性信号，并（ii）将这些信号传递至EL阶段加以利用，以在模糊情况下改进候选实体检索。我们使用基于BERT的NER模型，并为其增加两个辅助的片段边界预测头，在推理时应用蒙特卡洛（MC）dropout来推导提及级别的三种预测。

    arXiv:2611.33086v1 Announce Type: new  Abstract: Entity Resolution (ER) pipelines often collapse ambiguity into early hard decisions: upstream Named Entity Recognition (NER) selects a single span and type, and downstream Entity Linking (EL) treats that choice as ground truth. This discards valuable information about the ambiguity of the underlying data (e.g., span boundary disagreement and type confusion), leading to brittle systems that fail unpredictably when faced with uncertainty in inputs. Therefore, this work presents a proof-of-concept for uncertainty-aware ER that treats uncertainty as native data that should be preserved and exploited throughout ER pipelines. We (i) capture uncertainty signals in the NER stage and (ii) leverage these signals into the EL stage to improve candidate retrieval under ambiguity. Using a BERT-based NER model augmented with two auxiliary span-boundary prediction heads, we apply Monte Carlo (MC) dropout at inference time to derive three mention-level p
    
[^9]: 从患者旅程中创建以对象为中心的事件日志

    On the Creation of Object-Centric Event Logs from Patient Journeys

    [https://arxiv.org/abs/2611.33074](https://arxiv.org/abs/2611.33074)

    本文针对医疗领域中对象中心流程挖掘研究的空白，基于一系列挑战与需求提出了一种从患者旅程数据创建以对象为中心的事件日志（OCEL）的方法。

    

    患者与医疗系统之间的互动以及个体在患病期间所经历的状况都被记录在患者旅程中。因此，患者旅程是研究医疗服务如何提供、以及如何设计医疗系统以更好地满足患者需求的重要资产。目前已有从传统案例概念的流程挖掘视角对患者旅程进行的初步研究，但以对象为中心的流程挖掘技术尚未被涉及。这令人遗憾，因为得益于对象及其关系的丰富结构，以对象为中心的流程挖掘方法在这一领域尤其具有前景。本文通过研究从患者旅程生成以对象为中心的事件日志（OCEL）来填补这一空白。基于一系列挑战和需求的清单，本文提出了一种从患者旅程数据创建OCEL的方法，并在……上对该方法进行了评估。

    arXiv:2611.33074v1 Announce Type: cross  Abstract: Interactions between patients and health systems as well as conditions that individuals experience during a disease are represented in patient journeys. Therefore, patient journeys are important assets to investigate how care is being provided, and how health systems can be designed to better suit the needs of patients. While there is initial work on patient journeys from a process mining perspective with traditional case notions, object-centric process mining techniques have yet to be covered. This is unfortunate, since due to the rich structure of objects and their relationships, object-centric process mining approaches are particularly promising in this context. This paper addresses this gap by investigating object-centric event log (OCEL) generation from patient journeys. Based on a list of challenges and requirements, the paper presents an approach for the creation of OCELs from patient journey data. The approach is evaluated on a
    
[^10]: 审计评判者：基于人类发现的偏见在LLM评判系统中的发现、量化与缓解

    Auditing the Judge: Human-Derived Bias Discovery, Quantification, and Mitigation in LLM Judges

    [https://arxiv.org/abs/2611.33065](https://arxiv.org/abs/2611.33065)

    该论文提出HUB-J集成框架，利用人类与LLM判断不一致的案例自动发现可解释的偏见因子，并通过受控输入修改量化偏见强度，实现LLM评判系统中偏见的发现、量化与缓解。

    

    虽然大语言模型越来越多地被部署为评估和训练流程中的自动化评估器，但其判断往往受到与人类偏好相冲突的系统性偏见的影响。尽管先前的研究已经识别出各种偏见并提出了检测和缓解方法，但对利用人类偏好来指导偏见识别与缓解的关注却较少。此外，这些方法严重依赖预先发现的偏见列表，忽视了偏见强度，并且依赖成本高昂的干预措施。在本工作中，我们提出了HUB-J，一个基于人类衍生偏见因子的集成框架，用于检测、量化和缓解LLM作为评判者系统中的偏见。我们的方法利用人类与LLM判断不一致的案例自动发现可解释的偏见因子，并利用一致的案例通过受控输入修改及其产生的偏移来量化偏见强度。

    arXiv:2611.33065v1 Announce Type: new  Abstract: While Large Language Models (LLMs) are increasingly deployed as automated evaluators in evaluation and training pipelines, their judgements are often affected by systematic biases that conflict with human preferences. Although prior work has identified various biases and proposed methods for detecting and mitigating them, less attention has been paid to using human preferences to inform bias identification and mitigation. Moreover, these approaches rely heavily on pre-discovered bias lists, overlook bias strength, and depend on costly interventions. In this work, we propose HUB-J, an integrated framework based on human-derived bias factors that detects, quantifies, and mitigates biases in LLM-as-a-Judge systems. Our approach leverages human-LLM judgement disagreement cases to automatically discover interpretable bias factors, and utilizes agreement cases to quantify bias strength through controlled input modifications and resulting shift
    
[^11]: ClusterVAD：基于预训练神经表示的无阈值无监督语音活动检测

    ClusterVAD: Threshold-Free Unsupervised Voice Activity Detection with Pre-Trained Neural Representations

    [https://arxiv.org/abs/2611.33059](https://arxiv.org/abs/2611.33059)

    提出了ClusterVAD，一种基于冻结语音基础模型表示的无监督语音活动检测方法，通过对表示进行聚类并用响度标注语音/非语音聚类，结合逐录音拟合的隐马尔可夫模型，实现了无需阈值设定且在会议、远场和噪声等场景下表现稳健的检测。

    

    语音活动检测（VAD）是语音处理流水线中的首要步骤之一，其应用涵盖从转录、说话人分离到语音语言模型。有监督检测器需要标注数据，并继承了其训练语料库的领域特性和标注惯例。现有的无监督检测器则依赖手工设计的声学特征，这些特征无法从数据中获益，并且在远场、噪声或非典型语音条件下性能会下降。我们提出了ClusterVAD，一种构建在冻结的语音基础模型表示之上的无监督VAD方法。它只需在少量未转录语音上拟合一次，即可对这些表示进行聚类，并利用响度（sonority，一种声学度量）将每个聚类标注为语音或非语音。在推理阶段，通过聚类分配计算帧级语音概率，并为每条录音拟合一个两状态隐马尔可夫模型，从而实现无需设定阈值的检测。在涵盖会议、远场和噪声录音的多个基准测试上……

    arXiv:2611.33059v1 Announce Type: cross  Abstract: Voice activity detection (VAD) is one of the first steps in speech processing pipelines, from transcription and diarization to speech language models. Supervised detectors need labelled data and inherit the domain and annotation conventions of their training corpora. Existing unsupervised detectors rely on hand-crafted acoustic features which cannot benefit from data and degrade in far-field, noisy or atypical speech. We propose ClusterVAD, an unsupervised VAD built on frozen speech foundation model representations. Fitted once on a small set of untranscribed speech, it clusters the representations and labels each cluster as speech or non-speech with sonority, an acoustic measure. At inference, a frame-level speech probability is computed from the cluster assignments, and a two-state hidden Markov model fitted to each recording enables threshold-free detection. Across benchmarks covering meetings, far-field and noisy recordings, Mandar
    
[^12]: 面向AI原生RAN控制的混合人工智能：一种功能视角

    Hybrid AI for AI-Native RAN Control: A Functional Perspective

    [https://arxiv.org/abs/2611.33035](https://arxiv.org/abs/2611.33035)

    本综述提出了一种与技术无关的功能性分类框架，将RAN决策分解为表示与推理、预测与仿真、自适应控制和决策综合四种互补能力，为比较AI与传统方法在AI原生RAN控制中的混合协作提供了统一视角。

    

    AI原生无线接入网（RAN）控制涉及跨网络实体、协议层和时间尺度的相互关联的决策。要理解AI与传统方法如何共同支持这些决策，需要一个共同的基准来比较二者的功能角色。本综述提出了一种与技术无关的RAN决策分类框架，围绕四种互补能力进行组织：表示与推理、预测与仿真、自适应控制以及决策综合。这些能力根据其在控制过程中的角色来定义，与实现它们的具体技术无关。基于这一视角，我们考察了代表性的AI方法和基于模型的方法，包括基础模型、世界模型、强化学习和优化。随后，我们将该分类框架应用于慢速、中速、快速及跨时间尺度应用中的代表性RAN控制解决方案，以识别其中所需的关键能力。

    arXiv:2611.33035v1 Announce Type: new  Abstract: AI-native radio access network (RAN) control involves interdependent decisions across network entities, protocol layers, and timescales. Understanding how AI and conventional methods can jointly support these decisions requires a common basis for comparing their functional roles. This survey introduces a technique-agnostic taxonomy of RAN decision-making organized around four complementary capabilities: Representation and Reasoning, Prediction and Simulation, Adaptive Control, and Decision Synthesis. These capabilities are defined by their roles in the control process, independently of the techniques used to implement them. Through this perspective, we examine representative AI and model-based approaches, including foundation models, world models, reinforcement learning, and optimization. We then apply the taxonomy to representative RAN control solutions across slow, medium, fast, and cross-timescale applications to identify which capabi
    
[^13]: 前沿大语言模型能否通过无工具的权重空间编程修复量化网络？

    Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?

    [https://arxiv.org/abs/2611.33031](https://arxiv.org/abs/2611.33031)

    该研究证明前沿大语言模型无需任何工具、数据或推理能力，仅凭权重空间编程即可将4比特量化CNN的准确率从60.4%恢复至77.95%，接近原始浮点模型的81.4%水平。

    

    我们将Claude家族的四个大语言模型（LLM）以文本形式提供了具有2万参数的CIFAR-10卷积神经网络的完整权重，并要求它们在没有工具、数据或运行推理能力的情况下生成一个4比特网络。标准的舍入到最近（RTN）4比特量化会将测试准确率从81.4%降低到60.4%。我们设计了四种评估协议，它们在提供给LLM的输入和任务描述上有所不同，并分析它们的响应。在前两种协议中，LLM被提供了浮点网络和4比特RTN网络，并被告知准确率显著下降。在后两种协议中，它们只收到浮点网络和量化该网络的任务，这用于测试它们是自己主动应用优化，还是退回到普通的RTN。我们的分析表明，前沿LLM能够恢复大部分损失的准确率。在每层给出一个答案的情况下，Opus 5.5通过平衡舍入误差将4比特网络的准确率提升至77.95%。

    arXiv:2611.33031v1 Announce Type: cross  Abstract: We give four large language models (LLMs) of the Claude family the complete weights of a 20k-parameter CIFAR-10 CNN as text and ask them to produce a 4-bit network without tools, data or the ability to run inference. Standard round-to-nearest (RTN) quantization to 4 bit lowers the test accuracy from 81.4% to 60.4%. We design four evaluation protocols that differ in the input and task description given to the LLMs and analyze their responses. In the first two protocols the LLMs are given the float and the 4-bit RTN network and are told that the accuracy has dropped significantly. In the last two they receive only the float network and the task to quantize it, which tests whether they apply an optimization on their own or fall back to plain RTN. Our analysis shows that a frontier LLM can recover most of the lost accuracy. Given one answer per layer, Opus 5.5 raises the accuracy of the 4-bit network to 77.95% by balancing the rounding err
    
[^14]: RecordRoute：基于定位引导诊断的智能体失败审计方法

    RecordRoute: Auditing Agent Failures with Localization-Guided Diagnosis

    [https://arxiv.org/abs/2611.33022](https://arxiv.org/abs/2611.33022)

    RecordRoute将比较式记录定位与基于所选记录文本和元数据的局部错误模式诊断相耦合，实现了“先定位再诊断”的智能体失败审计框架，在Who&When Pro基准的全部指标上超越现有方法。

    

    审计失败的智能体执行需要识别责任主体、最早的决定性步骤以及错误模式。现有的代表性监督方法要么专注于步骤定位，要么从共享的轨迹表示中预测多个归因目标，这些方法并未利用定位结果来选择用于模式诊断的文本。为解决这一局限，我们提出了RecordRoute，它将比较式定位与局部诊断相结合。具体而言，该方法使用词级和字符级特征表示每条记录，并通过轨迹内部的比较学习对记录进行排序。被选中的记录提供责任主体和步骤信息，而其文本和元数据则驱动错误模式预测。在受控注入的Who&When Pro基准上，RecordRoute在全部四项官方指标上均优于StepFinder和ASCon。在固定主体和步骤预测的条件下，局部诊断达到77.47%的合并联合准确率，而全轨迹方法仅为69.28%。

    arXiv:2611.33022v1 Announce Type: new  Abstract: Auditing failed agent executions requires identifying the responsible actor, the earliest decisive step, and the error mode. Representative supervised methods either focus on step localization or predict multiple attribution targets from shared trajectory representations. These methods do not use the localization result to select the text for mode diagnosis. To address this limitation, we propose RecordRoute, which couples comparative localization with local diagnosis. Specifically, it represents each record with word and character features and learns to rank records through comparisons within each trajectory. The selected record supplies the actor and step, while its text and metadata drive mode prediction. On controlled-injection Who&When Pro, RecordRoute outperforms StepFinder and ASCon on all four official metrics. With actor and step predictions fixed, local diagnosis reaches 77.47% pooled joint accuracy versus 69.28% for full-trace
    
[^15]: LLM裁判能否识别不安全的临床行为？一项故障注入研究

    Can LLM Judges Catch Unsafe Clinical Actions? A Fault-Injection Study

    [https://arxiv.org/abs/2611.33012](https://arxiv.org/abs/2611.33012)

    该研究提出了一种故障注入评估方法，通过在MedAgentBench中向经验证的智能体轨迹注入416条受控临床错误（覆盖八个错误类别、六个开源LLM裁判），发现LLM裁判识别不安全临床行为的可靠性高度依赖于临床证据和完整轨迹的可见性——仅看最终答案时既漏检错误又误报高达56%的正确轨迹。

    

    随着临床AI智能体开始与电子健康记录（EHR）交互，大语言模型（LLM）越来越多地被用于评估这些智能体的行为。但这些裁判者能在多大程度上可靠地识别出临床上的危险错误，尤其是当智能体看起来已经成功完成任务时？我们通过在MedAgentBench中向经过验证的智能体轨迹注入受控的临床错误来研究这一问题。实验包含416条错误轨迹、八个错误类别和六个开源权重的LLM裁判。通过将每条错误轨迹与其正确的对应轨迹进行对比，我们衡量裁判者是真正识别出了错误，还是仅仅简单地将更多操作标记为不安全。研究结果表明，能否获取临床证据会显著影响裁判的可靠性。当裁判只看到最终答案时，无法区分那些不改变最终答案的错误，同时还会将56%的正确轨迹误判为不安全。（摘要原文在此处截断）

    arXiv:2611.33012v1 Announce Type: new  Abstract: As clinical AI agents begin to interact with electronic health records (EHRs), large language models (LLMs) are increasingly used to evaluate their actions. But how reliably can these judges recognize clinically dangerous mistakes, especially when an agent appears to have completed its task successfully? We investigate this question using MedAgentBench by introducing controlled clinical errors into verified agent trajectories. The experiments include 416 faulty trajectories, eight error categories, and six open-weight LLM judges. By comparing each faulty trajectory with its correct counterpart, we measure whether judges genuinely recognize errors rather than simply flagging more actions as unsafe. The findings show that access to clinical evidence significantly improves judge reliability. When shown only the final answer, judges cannot distinguish errors that leave it unchanged, while flagging 56% of correct trajectories. With full traje
    
[^16]: 具身AI无法通过镜子测试：一面简单的镜子如何破坏机器人策略

    Embodied AI Fails the Mirror Test: How a Simple Mirror Breaks Robot Policies

    [https://arxiv.org/abs/2611.33010](https://arxiv.org/abs/2611.33010)

    该论文提出了RoboDojo-Mirror基准测试，发现即使场景中只出现一面镜子，也会使各类机器人操作策略（包括VLA模型、世界动作模型和智能体策略）的成功率大幅下降10.8至62.5个百分点，揭示了当前具身AI对镜面反射缺乏鲁棒性。

    

    一面简单的镜子能否暴露当今机器人操作策略的局限性？受动物认知中镜子自我识别测试的启发，我们研究了机器人策略能否在包含自身反射的场景中可靠地操作物理物体。我们提出了RoboDojo-Mirror，这是一个基于RoboDojo构建的基准测试，它在保持任务目标、成功标准和策略检查点不变的前提下，将无镜基线与单镜、双镜和镜面桌面条件进行对比。在涵盖视觉-语言-动作模型、世界动作模型和智能体策略的五种策略上，每种镜子条件都会降低总体成功率：对三种镜子条件取平均后，各策略的下降幅度介于10.8至56.7个百分点之间，而在镜面桌面上下降幅度达到62.5个百分点。由于镜子本身也是一种物理障碍物，我们将其与几何形状匹配的非反射面板进行比较；50--88%

    arXiv:2611.33010v1 Announce Type: cross  Abstract: Can a simple mirror expose the limitations of today's robotic manipulation policies? Inspired by the mirror self-recognition test in animal cognition, we investigate whether robot policies can reliably manipulate physical objects in scenes containing their reflections. We introduce RoboDojo-Mirror, a benchmark built on RoboDojo that preserves task objectives, success criteria, and policy checkpoints while comparing a mirror-free baseline with single-mirror, dual-mirror, and mirrored-tabletop conditions. Across five policies spanning Vision-Language-Action models, World Action Models, and agentic policies, every mirror condition reduces aggregate success rates: averaged over the three conditions, the drop ranges from 10.8 to 56.7 percentage points across policies, and it reaches 62.5 points on the mirrored tabletop. Because a mirror is also a physical obstacle, we compare it against geometrically matched non-reflective panels; 50--88\% 
    
[^17]: LEGOSpec：一个用于推测性草稿设计与训练的可组合高效框架

    LEGOSpec: A Composable and Efficient Framework for Speculative Draft Design and Training

    [https://arxiv.org/abs/2611.33004](https://arxiv.org/abs/2611.33004)

    LEGOSpec提出了一个可组合的推测性草稿设计与训练框架，将草稿模型分解为上下文、骨干、提议和目标四个组件，并通过系统化的对照实验揭示了不同工作负载下各组件的作用、相互作用及最优组合规律。

    

    推测性解码使用轻量级草稿模型来加速大语言模型（LLM）的生成。现有方法在不同的训练配置下评估完整的方案，导致各组件的效果、相互作用以及组合收益不明确。因此，我们提出了LEGOSpec，一个用于推测性草稿设计与训练的可组合框架。LEGOSpec将草稿模型分解为上下文、骨干、提议和目标四个组件，它们共享统一的训练基础设施。借助LEGOSpec，我们系统地改变各个组件，并在匹配的训练条件下评估选定的组合，以隔离它们各自的效果与相互作用。这些实验揭示了三个发现：（1）不同的工作负载偏好不同的骨干网络，这取决于上下文长度和内存预算，其中块内依赖建模对长上下文尤为重要。（2）提议设计必须在接受率与延迟之间取得平衡。（3）训练目标……（摘要截断）

    arXiv:2611.33004v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model (LLM) generation using a lightweight draft. Existing methods evaluate complete recipes under different training configurations, leaving component effects, interactions, and composition benefits unclear. Therefore, we present LEGOSpec, a composable framework for speculative draft design and training. LEGOSpec decomposes draft models into Context, Backbone, Proposal, and Objective components that share a unified training infrastructure. Using LEGOSpec, we systematically vary components and evaluate selected combinations under matched training conditions to isolate their effects and interactions. These experiments reveal three findings: (1) Different workloads favor different backbones, depending on context length and memory budget, with within-block dependency modeling particularly important for long contexts. (2) Proposal design must balance acceptance and latency. (3) The training o
    
[^18]: 一个、无人与十万个CT：测试文本到CT生成中的解剖学控制

    One, No One and One Hundred Thousand CTs: Testing Anatomical Control in Text-to-CT Generation

    [https://arxiv.org/abs/2611.33003](https://arxiv.org/abs/2611.33003)

    本文提出CtrlCT工具包，通过可信度测试（将生成CT与真实数据整体比较）和无需参考CT的变形式控制力测试（验证报告编辑能否按需、一致地控制解剖结构），解决了现有Text-to-CT生成器评估标准不一、且从未验证报告对解剖结构控制能力的核心问题。

    

    深度学习的最新进展以及带有配对报告的计算机断层扫描（CT）数据集的公开可用性，催生了许多文本到CT（Text-to-CT）生成器。然而，这些生成器的评估在预处理和指标上几乎总是各不相同，阻碍了模型之间的比较。此外，它们的评估方案几乎从不测试编辑报告会如何改变生成的CT——即使在相同随机种子下也是如此——因此人们往往默认报告能够控制解剖结构，却缺乏关于其控制程度和一致性的证据。为此，我们提出了CtrlCT，一个用于测试生成器两种特性的工具包：可信度与控制力。可信度测试将生成的CT作为一个整体群体与真实CT和报告进行比较；控制力测试则是变形测试，无需参考CT：在固定随机种子下，编辑一个发现应按要求改变它，保持被否定的发现不存在，并且对远处解剖结构的影响不超过更换新随机种子所带来的变化，且这种改变应在不同种子间保持一致。一份报告卡总结了所有……（原文摘要此处截断）

    arXiv:2611.33003v1 Announce Type: cross  Abstract: Recent advances in deep learning and the public availability of computed tomography (CT) datasets with paired reports have enabled many Text-to-CT generators. However, their evaluations almost always differ in preprocessing and metrics, which hinders comparison. Moreover, their protocols almost never test how editing the report changes the CT, not even at the same seed, so the report is often assumed to control the anatomy without evidence of how much and how consistently it does. We therefore propose CtrlCT, a toolkit that tests two properties of a generator, Credibility and Control. Credibility tests compare generated CTs, as a population, with real CTs and reports; Control tests are metamorphic and need no reference CT: at a fixed seed, editing one finding should change it as requested, keep negated findings absent and shift distant anatomy no more than a new seed, and the change should hold across seeds. A Report Card summarises al
    
[^19]: 论基础模型的潜能与实现

    On Potentiality and Actuality Of Foundation Models

    [https://arxiv.org/abs/2611.32991](https://arxiv.org/abs/2611.32991)

    基础模型本身已蕴含完成特定任务所需的能力，适配并非添加新能力，而只需通过恒定的、与输入无关的向量移位将模型“实现”即可。

    

    基础模型通过大规模预训练构建，通常随后进行后训练，然后再被适配到特定任务和行为上。但是，这种适配是否真正增加了新能力，还是基础模型本身早已具备这些能力？我们将不同适配方式所引起的更新分解为一个在所有输入间共享的分量和一个随输入变化的残差分量。我们发现，共享分量足以产生大部分适配增益，并且可以直接学习得到：一个在每层学习一个恒定向量（对所有输入均相同）的移位适配器，就能复现LoRA和全量微调的行为。我们将这一现象称为基础模型的潜能原则：基础模型本身已经包含了实现特定任务专业能力所需的容量。要适配一个新任务，模型只需被“实现”，即通过一个恒定的、与输入无关的移位被推向某个方向。我们展示了这一原则在……方面的意义。

    arXiv:2611.32991v1 Announce Type: cross  Abstract: Foundation models are built through large-scale pretraining, often followed by post-training, and are then adapted to specific tasks and behaviors. But, does the adaptation add truly new capabilities, or the base model already holds it? We decompose the update induced by different adaptations into a component shared across all inputs and a residual that varies with the input. We find that the shared component suffices for most of the adaptation gain and can be learned directly: a shift adapter that learns a constant vector per layer, identical for every input, reproduces the behavior of LoRA and full fine-tuning. We call this the potentiality principle of foundation models: the base model already contains the capacity needed for task-specific expertise. To adapt to a new task, the model just needs to be actualized, i.e., pushed in a certain direction via a constant, input-independent shift. We show the implications of this principle on
    
[^20]: CAPER：面向可靠手册引导机器人装配规划的PartGround与PlanRefine

    CAPER: PartGround and PlanRefine for Reliable Manual-Guided Robotic Assembly Planning

    [https://arxiv.org/abs/2611.32986](https://arxiv.org/abs/2611.32986)

    提出CAPER框架，通过PartGround将手册证据与产品库存对齐以准确定位零件、并通过PlanRefine对照有限库存精炼计划，使视觉-语言模型能够可靠地生成机器人装配规划。

    

    在机器人任务规划中，使用视觉-语言模型（VLM）从现有手册中推导装配步骤时，虽然可能正确解读单个装配动作，但常常会混淆确切的零件身份、预测出不支持的零件ID，或在装配步骤之间产生不一致的零件。为此，我们提出了CAPER（Capacity-Aware Part-grounding and Evidence Refinement，容量感知零件定位与证据精炼），这是一个手册引导的机器人装配框架，通过两个模块——PartGround和PlanRefine——使VLM能够生成装配步骤。PartGround将步骤级的手册证据与产品特定的库存及零件参考信息对齐，以识别确切的零件ID和数量，将自由形式的硬件预测转化为以共享装配中间表示（Assembly IR）呈现的证据引导的分配。PlanRefine则对照有限库存对PartGround的分配进行审计，移除置信度最低的分配。对于身份不确定的情况则保持标记……

    arXiv:2611.32986v1 Announce Type: cross  Abstract: Vision-Language-Models (VLMs) used in robot task planning to derive assembly procedures from existing manuals may correctly interpret individual assembly actions yet confuse exact part identities, predict unsupported Part IDs, or produce inconsistent parts across assembly steps. Thus, we propose CAPER (Capacity-Aware Part-grounding and Evidence Refinement), a manual-guided robotic assembly framework that enables VLMs to generate assembly procedures with two modules--PartGround and PlanRefine. PartGround aligns step-level manual evidence with product-specific inventory and part-reference information to identify exact Part IDs and quantities, transforming free-form hardware prediction into evidence-guided assignments represented in a shared Assembly Intermediate Representation (Assembly IR). PlanRefine audits PartGround assignments against finite inventory by removing the lowest-confidence assignments. Uncertain identity cases remain mar
    
[^21]: 自动驾驶视觉语言模型中的证据充分性基准测试

    Benchmarking Evidence Sufficiency in Vision-Language Models for Autonomous Driving

    [https://arxiv.org/abs/2611.32964](https://arxiv.org/abs/2611.32964)

    该论文提出 DriveAnswer 基准，通过匹配的强/弱证据问题配置来评估自动驾驶视觉语言模型判断证据充分性并在证据不足时适当放弃回答的能力。

    

    可靠的驾驶视觉语言模型（VLM）不仅要正确回答问题或定位相关场景对象，还必须判断现有视觉证据是否足以支持一个答案。现有评估大多将定位能力和可回答性作为独立的测量指标，导致难以区分有充分证据支持的预测与那些仅定位到相关但证据不足的预测。我们提出了 DriveAnswer，一个包含涵盖七个推理类别的 20,151 个问题的基准，通过匹配的强证据与弱证据配置来实现这一区分。每一对配置在保持底层场景事实和语义答案不变的前提下，改变观察结果是否支持作答，使得在弱证据情况下放弃回答成为恰当的响应。在六类别评估中，最强的零样本基线达到了 67.31% 的强证据准确率，但仅有 21.55% 的匹配问题同时满足“回答-放弃”行为。

    arXiv:2611.32964v1 Announce Type: cross  Abstract: Reliable driving VLMs must do more than answer correctly or localize relevant scene actors: they must determine whether the available visual evidence is sufficient to support an answer. Existing evaluations largely treat grounding and answerability as separate measurements, making it difficult to distinguish well-supported predictions from those localized to relevant but insufficient evidence. We introduce DriveAnswer, a benchmark of 20,151 questions across seven reasoning families, that operationalizes this distinction using matched strong- and weak-evidence configurations. Each pair preserves the underlying scene fact and semantic answer while changing whether the observations support answering, making abstention the appropriate response under weak evidence. On the six-family evaluation, the strongest zero-shot baseline reaches 67.31% strong-evidence accuracy, yet only 21.55% of matched questions satisfy both answer-abstain behavior 
    
[^22]: 证明即所需：面向HPC AI智能体的零信任架构

    Attestation is All You Need: Toward a Zero-Trust Architecture for HPC AI Agents

    [https://arxiv.org/abs/2611.32958](https://arxiv.org/abs/2611.32958)

    提出AEGIS系统，通过编译任务行为约束、基于eBPF的系统调用监控、跨智能体行为关联以及Slurm级遏制，实现了针对HPC环境中被劫持授权AI智能体的运行时零信任行为证明防御。

    

    AI智能体开始在高性能计算（HPC）环境中实现数据分析、编排和仿真引导的自动化，但它们引入了一个现有集群防御机制无法解决的安全问题：被劫持的授权智能体。攻击者可以通过提示注入、投毒的共享数据、被攻陷的工具或跨智能体协同来破坏智能体，随后通过合法的加密LLM API流量外泄敏感数据。我们提出行为证明作为针对该场景的运行时防御手段。我们的系统AEGIS编译源自任务的行为约束，利用基于eBPF的系统调用拦截来监控智能体行为，将证据与约束配置文件进行核验，跨智能体关联行为，并触发Slurm级别的遏制措施。在实际证明路径上，AEGIS以1秒的检测间隔实现了亚秒级的中位检测延迟和较低的验证器开销，成功检测出全部四类已实现的攻击。受控实验表

    arXiv:2611.32958v1 Announce Type: cross  Abstract: AI agents are beginning to automate data analysis, orchestration, and simulation steering in HPC environments, but they introduce a security problem that existing cluster defenses do not address: the hijacked authorized agent. An adversary can subvert an agent through prompt injection, poisoned shared data, compromised tools, or multi-agent coordination, then exfiltrate sensitive data through legitimate encrypted LLM API traffic. We propose behavioral attestation as a runtime defense for this setting. Our system, AEGIS, compiles task-derived behavioral constraints, monitors agent actions with ebpf-based syscall interception, verifies evidence against constraint profiles, correlates behavior across agents, and triggers slurm-level containment. On the real attestation path, AEGIS detects all four implemented attack classes with sub-second median latency at a 1-second interval and low measured verifier overhead. Controlled experiments sho
    
[^23]: “我确实打算发货”：大语言模型会运行电商平台预付费诈骗吗？

    "I Do Intend to Deliver": Will LLMs Run a Marketplace Advance-Fee Scam?

    [https://arxiv.org/abs/2611.32957](https://arxiv.org/abs/2611.32957)

    研究表明，仅用一条无需越狱的普通提示词，十个主流大语言模型中就有八个愿意充当电商预付费诈骗的“卖家”，在1200次模拟对话中61%的买家被诱导预付款且单次成本仅约2.5美分，凸显了LLM被滥用于在线欺诈的严重风险。

    

    在线欺诈传统上受到与潜在受害者交谈所需人力成本的限制。我们探讨托管的大型语言模型现在是否会提供这种“劳动力”。我们提示十个现成模型扮演“免费物品，只需支付运费”这一买卖网站上常见的预付费诈骗中的卖家角色，仅使用一个普通系统提示词，无需越狱。每个模型分别与四个买家角色（从轻信到警惕）进行30次对话，买家由同一模型扮演（共1,200次对话）。我们根据买家资金的去向对每次对话进行标注，并对照两名人工编码员验证标签结果。737次对话（61%）以买家预先付款告终。十个模型中有八个几乎在每次对话中都配合出演，且同时运行买卖双方的成本约为每次付款2.5美分。只有想要可查证真实身份的警惕买家进行了抵制，而最“配合”的卖家通过编造姓名绕过了这一防线。

    arXiv:2611.32957v1 Announce Type: cross  Abstract: Online fraud is traditionally limited by the cost of the human labor needed to talk to potential victims. We ask whether hosted large language models will now supply that labor. We prompted ten off-the-shelf models to play the seller in the "free item, just pay delivery" advance-fee scam common on buy-and-sell sites, using a plain system prompt with no jailbreak. Each model played 30 conversations against each of four buyer personas, from credulous to guarded, with the same model playing the buyer (1,200 conversations). We label each conversation by what happened to the buyer's money, and validate the labels against two human coders. 737 conversations (61%) ended with the buyer paying in advance. Eight of the ten models played in almost every conversation, and running both sides cost about 2.5 cents per payment. Only the guarded buyer, who wanted a findable identity, resisted, and the most willing sellers got past it by inventing names
    
[^24]: LoomKV：面向大语言模型服务的从GPU内存到SSD的无损KV缓存压缩

    LoomKV: Lossless KV Cache Compression from GPU Memory to SSD for LLM Serving

    [https://arxiv.org/abs/2611.32954](https://arxiv.org/abs/2611.32954)

    LoomKV提出了一种无需修改服务引擎的无损KV缓存压缩方法，同时作用于外部存储层级和GPU内存池，在九个模型上将BF16 KV缓存压缩至原始大小的0.71且不改变模型输出。

    

    大语言模型（LLM）服务引擎将键值（KV）缓存从GPU内存卸载到主机DRAM和SSD，以便多轮对话和智能体请求能够重用提示前缀。容量在两个层面限制服务：外部层级在数据淘汰前管理前缀重用，而GPU KV池则限制并发长请求的数量，且有损压缩会改变KV值和模型输出。LoomKV在无需修改服务引擎的情况下，将无损KV压缩同时应用于外部存储层级（层级模式）和GPU内存池（池模式）。通过对BF16高字节（符号位及最高七位指数位）采用经过校准的每注意力头两级3位编码，LoomKV在九个模型上将BF16 KV以原始大小的0.71进行存储，FP8 KV则以0.85-0.87的比例存储。在层级模式下，TileStream写入精确长度的记录并在加载路径之外进行长度回读，保持层级存储、淘汰和预取机制不变，并将取回的记录直接解压缩到分散的GPU页面中。在池模式下，FragPack对变长数据块进行打包……

    arXiv:2611.32954v1 Announce Type: new  Abstract: Large language model (LLM) serving engines offload the key/value (KV) cache from GPU memory to host DRAM and SSD so multi-turn and agent requests can reuse prompt prefixes. Capacity then limits service at two levels: external tiers govern prefix reuse before eviction, while the GPU KV pool bounds concurrent long requests, and lossy compression alters KV values and outputs. LoomKV brings lossless KV compression to both external tiers (tier mode) and the GPU pool (pool mode) on an unmodified engine. Using a calibrated per-head two-level 3-bit code over BF16 high bytes (sign and upper seven exponent bits), LoomKV stores BF16 KV at 0.71 of its raw size across nine models and FP8 KV at 0.85-0.87. In tier mode, TileStream writes exact-length records with length readback off the load path, leaving tier storage, eviction, and prefetch unchanged, and decompresses fetched records straight into scattered GPU pages. In pool mode, FragPack packs vari
    
[^25]: 超越期望值：用于系统性保护规划的分布型评论家PPO方法

    Beyond Expected Value: PPO with Distributional Critics for Systematic Conservation Planning

    [https://arxiv.org/abs/2611.32948](https://arxiv.org/abs/2611.32948)

    本文提出用PPO结合Plackett-Luce方案以及基于隐式分位数网络（IQN）的分布型评论家来改进CAPTAIN保护规划框架，发现决策粒度而非优化器是生态收益的主要来源，其真正贡献在于更高的训练效率和对风险敏感目标的可扩展性。

    

    系统性保护规划需要在生态与气候不确定性下顺序分配有限的保护预算。静态工具无法随条件变化进行调整，而仅针对期望结果优化的策略在面对气候变化日益可能引发的最坏情况轨迹时可能失败。CAPTAIN是首个将此问题构建为顺序强化学习的框架，但其无梯度的进化策略（ES）优化器缺乏对每个时间步的信用分配。我们通过Plackett-Luce方案解决了CAPTAIN动作空间不可微的问题，从而用近端策略优化（PPO）取代ES，并用建模完整回报分布的隐式分位数网络（IQN）取代其标量评论家。匹配粒度的对比实验表明，驱动大部分生态收益的是决策粒度而非优化器本身，因此本文的真正贡献在于训练效率以及对风险敏感目标的可扩展性。在新西……（摘要截断）

    arXiv:2611.32948v1 Announce Type: new  Abstract: Systematic conservation planning must allocate limited protection budgets sequentially under ecological and climatic uncertainty. Static tools cannot adapt as conditions change, and policies optimised only for expected outcomes can fail on the worst-case trajectories climate change makes increasingly likely. CAPTAIN was the first framework to cast this as sequential reinforcement learning, but its gradient-free Evolution Strategies (ES) optimiser lacks per-timestep credit assignment. We replace ES with Proximal Policy Optimisation (PPO) via a Plackett-Luce solution to CAPTAIN's non-differentiable action space, and replace its scalar critic with an Implicit Quantile Network (IQN) modelling the full return distribution. A matched-granularity comparison shows decision granularity, not the optimiser, drives most of the ecological gain, so the real contribution is training efficiency and extensibility to risk-sensitive objectives. On a New Ze
    
[^26]: ShellBench：对于AI智能体的日常任务而言，模型和智能体框架哪个更重要？

    ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks

    [https://arxiv.org/abs/2611.32946](https://arxiv.org/abs/2611.32946)

    该论文推出ShellBench基准，通过对12个模型、5种智能体框架和3个推理水平共53,100次实验的评估发现，模型选择对日常任务性能的影响大于框架选择，但领先模型之间差距仅1.1个百分点，且高性能模型间的成本差异可达六倍之多。

    

    2026年全年，AI智能体在普通大众中的使用呈指数级增长，但用户往往难以确定哪种配置对其个人使用最为有效。为了帮助澄清这一问题，我们推出了ShellBench——基于超过1000万条用户请求构建的100项日常任务基准，每项任务都设置在包含杂乱文件和相互冲突数据源的真实工作空间中，并对正确性和安全性进行双重评分。我们评估了12个模型在5种智能体框架和3个推理水平下的表现，总计进行了53,100次实验，其中20%的任务被指定为公开发布。我们发现，模型选择对性能的影响大于框架选择，但三个领先模型之间的差距仅为1.1个百分点。GPT-6.1 Sol以约GPT-6 Astra每次尝试估算token成本的六分之一通过了61.4%的尝试，而GPT-6 Astra以62.5%的通过率位居首位。我们还发现，框架选择对性能几乎没有影响。

    arXiv:2611.32946v1 Announce Type: new  Abstract: The use of AI agents among the general public has been increasing exponentially throughout 2026, though it is often hard for users to determine which configuration is most effective for their personal use. To help provide clarity here we introduce ShellBench, 100 day-to-day tasks based on more than 10 million user requests, each set in a realistic workspace with messy files and competing data sources, and graded for both correctness and safety. We evaluate 12 models across 5 agent harnesses at 3 reasoning levels for a total of 53,100 rollouts, 20% of which are on tasks designated for public release. We find that model choice impacts performance more than harness choice, but the three leading models differ by only 1.1 percentage points. GPT-6.1 Sol passed 61.4% of attempts at about one-sixth of GPT-6 Astra's estimated token cost per attempt, while GPT-6 Astra led at 62.5%. We also found that harness choice had little effect on either perf
    
[^27]: 视觉语言模型知道自己的视觉弱点吗？探测能力层面的元认知

    Do Vision-Language Models Know Their Own Visual Weaknesses? Probing Capability-Level Metacognition

    [https://arxiv.org/abs/2611.32940](https://arxiv.org/abs/2611.32940)

    视觉语言模型缺乏对自身视觉弱点的预先认知（能力层面元认知不足），训练其预测自身准确率可泛化到未见任务类型，但弱点信息只有在开始解题后才显现。

    

    元认知包括了解自己在各类任务中的优势与劣势，而不仅仅是判断单个答案是否正确。我们测试了视觉语言模型（VLM）能否提前知道哪些类型的视觉任务和图像条件对它们来说很困难，涵盖了大量开源与专有模型。由于某些任务对所有模型都很困难，我们将模型对这种共同困难的认识与它对自身困难的认识区分开来。我们发现，VLM对自身的视觉弱点几乎没有预先认知：它们会误判自己的整体准确率，甚至不知道哪些任务类型对所有模型都很困难，而且大多数模型会误判图像退化对自身的影响。我们进一步训练VLM预测自己的准确率，这使它们能够在训练中未见过的任务类型上了解自身的准确率水平，但并不能让它们知道哪些类型对自己来说很困难。然而，一旦模型开始解题，关于这些弱点的信息就会出现。

    arXiv:2611.32940v1 Announce Type: cross  Abstract: Metacognition includes knowing one's own strengths and weaknesses across kinds of tasks, not only judging whether a single answer is correct. We test whether vision-language models (VLMs) know in advance which kinds of visual tasks and image conditions are hard for them, across a wide range of open and proprietary models. Because some tasks are hard for every model, we separate what a model knows about this shared difficulty from what it knows about itself. We find that VLMs know little in advance about their visual weaknesses. They misjudge their overall accuracy and do not even know which kinds of tasks are hard for every model, and most misjudge how degraded images affect them. We further train VLMs to predict their own accuracy, which teaches them their own level of accuracy on kinds unseen in training but not which kinds are hard for them. However, information about these weaknesses appears once a model starts solving. On a few un
    
[^28]: 基于令牌分配的高效视频检索增强生成

    Efficient Video Retrieval Augmented Generation via Token Allocation

    [https://arxiv.org/abs/2611.32938](https://arxiv.org/abs/2611.32938)

    Lembas是一个即插即用的视频RAG框架，将帧与字幕的令牌分配建模为收益递减的优化问题，用具有可证明收敛保证的迭代注水算法求解，并能在多轮对话中零开销地复用令牌。

    

    高效的长视频理解需要在尽量减少视觉语言模型（VLM）所处理令牌数量的同时识别相关证据。现有的视频检索增强生成（RAG）方法依赖于帧选择和令牌压缩，但忽略了每增加一个令牌所带来的价值。我们提出了Lembas，一个即插即用的框架，它将视频RAG形式化为在共享预算下对帧和字幕进行令牌分配的优化问题。我们的关键洞察是：令牌的价值在每个条目内部呈递减趋势，并且随着相似证据获得更多令牌而进一步降低。Lembas在单一目标函数中同时捕捉这两种效应，并通过一种具有可证明收敛保证的高效迭代注水算法进行求解。对于多轮对话，Lembas复用此前轮次中已处理的令牌，并借助VLM的注意力以零额外开销的方式精细化相关性评估。我们的评估表明，Lembas提升了准确率……

    arXiv:2611.32938v1 Announce Type: cross  Abstract: Efficient long-video understanding requires identifying relevant evidence while minimizing the tokens processed by vision-language models (VLMs). Existing video retrieval-augmented generation (RAG) methods rely on frame selection and token compression, but overlook how much each additional token is worth. We introduce Lembas, a plug-and-play framework that formulates video RAG as a token allocation problem over frames and subtitles under a shared budget. Our key insight is that the value of a token diminishes within each item and decreases further as similar evidence receives more tokens. Lembas captures both effects in a single objective and solves it with an efficient iterative water-filling algorithm that has a provable convergence guarantee. For multi-turn conversations, Lembas reuses the tokens processed in earlier turns and refines relevance with the VLM's attention at no extra cost. Our evaluations show that Lembas improves accu
    
[^29]: 先于跨度读取决策：高效的生成式验证

    Reading Decisions Before Spans: Efficient Generative Verification

    [https://arxiv.org/abs/2611.32931](https://arxiv.org/abs/2611.32931)

    提出Prefill1方法：利用生成式验证器的决策先于错误跨度定位、且决策分支前七个语法标记固定的特点，通过预填充固定标记并仅生成一个分支标记即可恢复验证决策，相比完整生成减少84.84%的服务开销且F1几乎不变。

    

    开发者在检查智能助手生成的报告时，可能只需要一条关于某项声明缺乏依据的警报。然而，生成式验证器可以继续解码以引用并分类这些错误。我们研究了需要多少这样的输出才能恢复其决策。在一个冻结的20亿参数（2B）跨度验证器中，“空错误列表”与“第一个错误对象”之间的选择发生在所有错误定位之前。Native8在该分支处停止生成。由于其前面的七个语法标记是固定的，Prefill1将它们作为输入进行预填充，并仅生成一个不受限制的分支标记，且使用相同的原生指令。在2,015个代码智能体TEST响应上，相对于完整生成，Prefill1将eager常驻服务开销减少了84.84%，其released-reference F1为83.43（完整生成为83.41）。尽管二者逻辑标记总数相同，Prefill1仍比Native8节省了31.98%。在默认编译执行下，在整个工作负载上额外节省了6.74%至26.73%，且决策变化很小。

    arXiv:2611.32931v1 Announce Type: new  Abstract: A developer checking an assistant's report may need only an alarm that a claim is unsupported. A generative verifier, however, can continue decoding to quote and classify the errors. We study how much of that output is needed to recover its decision. In a frozen 2B span verifier, the choice between an empty error list and its first object precedes all localization. Native8 stops generation at this branch. The preceding seven syntax tokens are fixed, so Prefill1 supplies them as input and generates only the unrestricted branch token, using the same native instruction. On 2,015 code-agent TEST responses, Prefill1 reduces eager resident service by 84.84% relative to full generation, with released-reference F1 of 83.43 versus 83.41. It saves 31.98% over Native8 even though their logical token totals are identical. Under default compiled execution, the extra saving is 6.74-26.73% on the complete workload, with small decision changes. Controll
    
[^30]: InstructReplay：重放助手轮次以在事实性微调中保持通用行为

    InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning

    [https://arxiv.org/abs/2611.32918](https://arxiv.org/abs/2611.32918)

    提出InstructReplay方法，通过让初始模型预先对通用指令生成回答并在微调时约束模型保持接近这些回答，在事实性微调中学习新知识的同时，将通用能力平均损失从28.1分降至2.4分，并避免了回答长度的崩溃。

    

    在新的事实数据上对指令微调后的语言模型进行微调，往往会损害其通用能力。我们发现，标准微调平均会使指令遵循、推理和编程等基准下降35分，并且这些损失与回答长度的崩溃密切相关：微调后的模型回答长度至少缩短一半，且回答偏离初始模型长度越远，失败的概率就越高。我们提出InstructReplay方法：让初始模型对一组通用指令各回答一次，并在微调过程中约束模型在这些回答上保持接近初始模型。在三个模型系列和两个领域中，InstructReplay将五个通用基准的平均下降从28.1分降至2.4分，并使回答长度保持接近初始模型，同时在习得新事实方面的效果与标准微调相当。在全部六种设置中，它比七种适应方法都能更好地平衡学习与保持。

    arXiv:2611.32918v1 Announce Type: new  Abstract: Fine-tuning an instruction-tuned language model on new facts often erodes its general abilities. We find that standard fine-tuning lowers instruction-following, reasoning and coding benchmarks by 35 points on average, and that these losses track a collapse in answer length: fine-tuned models at least halve their responses, and the further a response drifts from the initial model's length, the more often it fails. We propose InstructReplay: the initial model answers a set of general instructions once, and during fine-tuning the model is kept close to the initial model on these answers. Across three model families and two domains, InstructReplay reduces the average decline on five general benchmarks from 28.1 to 2.4 points and keeps answer length close to the initial model's, while learning the new facts as well as standard fine-tuning. It balances learning and retention better than seven adaptation methods in all six settings. Controlled 
    
[^31]: 面向具身视觉跟踪的运动条件世界建模

    Motion Conditioned World Modeling for Embodied Visual Tracking

    [https://arxiv.org/abs/2611.32900](https://arxiv.org/abs/2611.32900)

    提出运动条件世界建模框架，将目标运动分布与机器人自运动融入潜在状态转移，并通过几何-进度模块联合建模路径选择与运动进度的不确定性，为具身视觉跟踪提供几何一致的未来视觉状态预测。

    

    世界模型通过预测未来环境状态来支持具身决策。在具身视觉跟踪任务中，不确定的目标运动与机器人自身引起的视角变化共同影响视觉观测，使得未来状态预测极具挑战性。我们提出运动条件世界建模，将预测的目标运动分布与机器人自运动融入潜在状态转移中。所提出的几何-进度模块通过共享运动模式和路径条件化的进度分布，联合建模几何路径与单步行进距离。这一建模方式在捕捉路线选择与运动进度中的不确定性的同时，保持了二者之间的依赖关系。由动作引起的坐标变换将预测路径表示在下一时刻的机器人坐标系中，为未来视觉状态预测提供了具有几何依据的条件。未来视觉特征重建通过该条件对运动预测器进行监督。

    arXiv:2611.32900v1 Announce Type: cross  Abstract: World models support embodied decision-making by predicting future environment states. In embodied visual tracking, uncertain target motion and robot-induced viewpoint changes jointly affect visual observations, making future-state prediction challenging. We propose Motion-Conditioned World Modeling, which incorporates predicted target-motion distributions and robot ego-motion into latent state transitions. A Geometry--Progress module jointly models geometric paths and single-step travel distances through shared motion modes and path-conditioned progress distributions. This formulation captures uncertainty in route selection and motion progress while preserving their dependence. Action-induced coordinate transformations express predicted paths in the next robot frame, providing geometrically grounded conditions for future visual-state prediction. Future visual feature reconstruction supervises the motion predictor through the condition
    
[^32]: 在线对话中建设性言论、仇恨言论与赞同态度的联合检测

    Joint Detection of Constructive Speech, Hate Speech and Agreement in Online Conversations

    [https://arxiv.org/abs/2611.32889](https://arxiv.org/abs/2611.32889)

    本文构建了一个标注了仇恨语气、建设性语气和赞同态度的英丹麦语社交媒体评论大规模数据集，并通过微调多语言大语言模型实现三项任务的联合检测，其中 E5-large 在上下文增强架构下表现最佳。

    

    在线仇恨言论是一个日益严重的问题，对群体和个人都产生负面影响，可能导致用户退出在线对话和辩论。随着一些知名社交媒体平台减少其审核力度，建设性言论已成为对抗仇恨的替代策略。自动检测建设性言论和仇恨言论有助于监测工作，并增进对其流行程度和在线动态的理解。在这项工作中，我们收集并标注了一个大型数据集，包含英语和丹麦语社交媒体评论，标注维度包括仇恨语气、建设性语气以及对父评论的赞同程度。我们微调了多个多语言大语言模型在这三个维度上进行分类，并比较了上下文增强与非上下文增强的模型架构。E5-large 大语言模型在上下文增强（平均 F1 为 82%）和非上下文增强（平均 F1 为 81.6%）分类任务中均表现最佳，证明了……

    arXiv:2611.32889v1 Announce Type: cross  Abstract: Online hate speech is an ever-growing issue with negative impacts on both groups and individuals, potentially causing users to withdraw from online conversations and debate. With some prominent social media platforms reducing their moderation efforts, constructive speech has emerged as an alternative strategy for countering hate. Automated detection of constructive and hateful speech can aid monitoring and improve understanding of their prevalence and online dynamics. In this work, we collect and annotate a large dataset of English and Danish social media comments for hateful tone, constructive tone, and agreement with parent comments. We fine-tune several multilingual LLMs to classify along these three dimensions and compare context-enriched and non-context-enriched model architectures. The E5-large LLM performs best on both the context-enriched (82% average F1) and non-context-enriched (81.6% average F1) classification tasks, demonst
    
[^33]: 懒惰的智能体会违背自身信念作答：基于可验证信念的强化学习教会它们按自己的信念行事

    Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe

    [https://arxiv.org/abs/2611.32884](https://arxiv.org/abs/2611.32884)

    提出了基于可验证信念的强化学习（RLVB），利用模拟器在每一步计算答案的精确后验概率，奖励智能体在证据不足时继续探索、证据充分时才作答，从而教会AI智能体克服懒惰行为、按照自身信念判断何时给出答案。

    

    在长任务中工作的AI智能体常常在尚未获得足够证据时就过早地给出答案，这种行为被称为“懒惰”。在我们的评估中，智能体经常在其所陈述的信念仍然列出多个可能答案的情况下就过早下结论。回合结束时的结果奖励只对最终答案进行评分，因此无法教会智能体何时应该作答，因为靠运气猜对的答案与有证据支撑的答案会获得相同的奖励。然而在训练期间，模拟器可以在每一步计算出关于答案的精确后验概率，并判断证据是否已经充分。我们提出了基于可验证信念的强化学习，该方法在信念后验仍允许多个可能答案时奖励智能体继续采取行动，而在证据充分时奖励智能体给出答案。这种奖励不需要评判器、评论者或奖励模型，也不会读取智能体在每个动作之前写下的信念内容。我们还引入了Pro Worlds，（摘要在此处被截断）

    arXiv:2611.32884v1 Announce Type: new  Abstract: AI agents working on long tasks often commit to an answer before they have enough evidence, a behavior called laziness. In our evaluations, they frequently commit even while their stated belief still names several possible answers. An outcome reward at the end of the episode scores only the final answer, so it cannot teach the agent when to commit, because a lucky guess receives the same reward as an answer backed by evidence. During training, however, the simulator can compute the exact posterior over the answer at every step and tell whether the evidence is already enough. We propose Reinforcement Learning with Verifiable Beliefs (RLVB), which rewards the agent for taking further steps while this posterior still allows several answers and for committing once the evidence is sufficient. The reward needs no judge, critic or reward model, and it never reads the belief line the agent writes before each action. We also introduce Pro Worlds,
    
[^34]: 基于大语言模型上下文学习的数据中心自适应电磁暂态级数字孪生

    Adaptive Electromagnetic Transient-level Digital Twins for Data Centers via In-Context Learning with LLMs

    [https://arxiv.org/abs/2611.32882](https://arxiv.org/abs/2611.32882)

    本文提出一种基于大语言模型上下文学习的数据中心电磁暂态级自适应数字孪生，克服了传统数据驱动数字孪生在数据中心配置变化时无法泛化的局限。

    

    人工智能工作负载的快速增长显著提高了数据中心的电力需求。频繁的运行和配置变更（例如动态工作负载分配引起的负载浪涌）可能引发电磁暂态（EMT），对其进行准确预测对于数据中心的可靠运行与维护至关重要。尽管高保真电磁暂态求解器被广泛用于暂态分析，但其高昂的计算成本限制了大规模和多场景研究。数字孪生通过模拟物理系统的行为，以快速预测替代昂贵的高保真仿真，从而解决了这一计算瓶颈。然而，数据驱动的数字孪生缺乏自适应性，一旦数据中心配置发生变化便无法泛化。为解决这一局限，本文提出了一种基于大语言模型上下文学习的数据中心自适应电磁暂态级数字孪生。

    arXiv:2611.32882v1 Announce Type: cross  Abstract: The rapid growth of artificial intelligence (AI) workloads has significantly increased data center (DC) power demand. Frequent operational and configurational changes, such as load surges caused by dynamic workload allocation, can induce electromagnetic transients (EMTs) whose accurate prediction is critical for reliable DC operation and maintenance. Although high-fidelity EMT solvers are widely adopted for transient analysis, their computational cost limits large-scale and multi-scenario studies. Digital twins address this computational bottleneck by emulating the behavior of a physical system and enabling fast prediction in place of expensive high-fidelity simulation. However,data-driven digital twins are non-adaptive and fail to generalize once the DC configuration changes. To address this limitation, this paper proposes an adaptive EMT-level digital twin for DCs based on in-context learning (ICL) with large language models (LLMs). 
    
[^35]: MXFP4用于大语言模型推理：互补的训练后量化技术及其相互作用

    MXFP4 for LLM Inference: Complementary Post-Training Quantization Techniques and Their Interactions

    [https://arxiv.org/abs/2611.32873](https://arxiv.org/abs/2611.32873)

    该研究发现有效的微缩放格式量化方案依赖于具体格式：MXFP4和NVFP4虽都支持精确的4位推理，但MXFP4显著受益于旋转和纠错PTQ技术，而这些技术未必适用于NVFP4。

    

    低精度表示的最新进展为大语言模型量化引入了多种微缩放数值格式。通过在16或32个元素组成的组之间共享缩放因子，这些格式能够实现激进的低精度量化，同时精度损失极小，特别是与现代训练后量化（PTQ）算法结合使用时。它们的精度取决于数值格式如何与旋转、PTQ和缩放因子选择相互作用。我们研究了PTQ配置在六个大语言模型和三种微缩放格式（MXFP6、MXFP4和NVFP4）上的可组合性，系统地变化Hadamard旋转粒度、PTQ算法和缩放因子选择策略。我们的核心发现是：有效的量化方案依赖于具体的数值格式——MXFP4和NVFP4都能支持精确的4位推理，但对一种格式有益的技术不一定能迁移到另一种格式。MXFP4从旋转和纠错PTQ中获益显著，而NVF（摘要在此处截断）

    arXiv:2611.32873v1 Announce Type: cross  Abstract: Recent advances in low-precision representation have introduced several microscaling numerical formats for LLM quantization. By sharing a scale factor across groups of 16 or 32 elements, these formats enable aggressive low-precision quantization with minimal accuracy degradation, particularly when combined with modern PTQ algorithms. Their accuracy depends on how the numerical format interacts with rotation, PTQ, and scale selection. We study the composability of PTQ configurations across six LLMs and three microscaling formats: MXFP6, MXFP4, and NVFP4, systematically varying Hadamard rotation granularity, PTQ algorithm, and scale-selection strategy. Our central finding is that effective quantization recipes are format dependent: both MXFP4 and NVFP4 support accurate 4-bit inference, but techniques that benefit one do not necessarily transfer to the other. MXFP4 benefits substantially from rotation and error-correcting PTQ, whereas NVF
    
[^36]: 基于路径概率梯度针对动力学观测量优化力场

    Path-probability gradients for optimizing force fields against dynamical observables

    [https://arxiv.org/abs/2611.32851](https://arxiv.org/abs/2611.32851)

    提出PathGrad方法，通过对朗之万路径概率而非混沌分子轨迹求导，获得方差稳定的梯度估计器，从而能够直接利用实验动力学观测量（如扩散系数）对机器学习力场进行微调。

    

    机器学习力场继承了用于生成其训练数据的电子结构方法的误差，而利用实验测量数据对其进行精修提供了一条突破这一限制的直接途径。然而，对于动力学观测量而言，这种精修仍然十分困难，因为对混沌的分子轨迹进行微分会产生不稳定的梯度。在此，我们提出了PathGrad方法，它对朗之万路径概率而非轨迹本身进行微分。由此得到的梯度估计器可以从热噪声和力导数中即时累积，并且其方差随着轨迹长度的增加保持稳定。我们通过有限差分计算验证了该梯度估计器，并利用它结合实验扩散系数以及结构约束，对熔融铜和水的势能模型进行了微调。对于水，我们联合优化了扩散系数和氧-氧径向分布函数。

    arXiv:2611.32851v1 Announce Type: cross  Abstract: Machine learning force fields inherit the errors of the electronic-structure methods used to generate their training data, and refining them against experimental measurements provides a direct route beyond this limit. However, such refinement remains difficult for dynamical observables because differentiating chaotic molecular trajectories yields unstable gradients. Here, we introduce PathGrad, which differentiates Langevin path probabilities instead of the trajectories themselves. The resulting gradient estimator is accumulated on the fly from thermal noise and force derivatives, and its variance remains stable as the trajectory length increases. We validated the gradient estimator against finite-difference calculations and used it to fine-tune molten-copper and water potentials against experimental diffusion coefficients together with structural constraints. For water, jointly optimizing diffusion and the oxygen-oxygen radial distrib
    
[^37]: 量化小型语言模型可信度评估中的测量伪影

    Measurement Artifacts in Trustworthiness Evaluations of Quantized Small Language Models

    [https://arxiv.org/abs/2611.32848](https://arxiv.org/abs/2611.32848)

    该论文揭示了量化语言模型可信度评估中与量化交互作用的测量误差会被误报为量化效应，并实证发现4比特量化导致的有害顺从上升仅为关键词检测器记录的非拒绝上升的一半至三分之二，且不同评分方法对量化效应的估计严重不一致。

    

    对量化语言模型的评估通常将基准分数的变化归因于量化。我们将测量到的变化表示为真实变化加上由测量工具贡献的一项。在每种精度下都相同的测量工具误差只会缩小配对比较的结果，但无法制造或逆转效应；而与量化发生交互作用的误差则两者皆可做到，并且会被当作量化的效应报告出来。我们在8个参数量低于20亿的指令微调模型上，于float16、LLM.int8()和NF4三种精度下使用相同题目，估计了四种测量选择下这种交互作用的大小。以两名人类标注者的标签为参照，4比特量化导致的有害顺从上升约为经过修正的关键词检测器所记录的非拒绝上升的一半至三分之二。完形填空式与字母选项式选择题评分在量化效应上的分歧远超其噪声水平所允许的范围：两者的估计值相关性仅为 ρ = 0.10，而理论上限……（摘要原文在此处截断）

    arXiv:2611.32848v1 Announce Type: new  Abstract: Evaluations of quantized language models attribute the change in a benchmark score to quantization. We write the measured change as the true change plus a term contributed by the instrument. Instrument error that is the same at every precision can shrink a paired comparison but cannot create or reverse an effect. Error that interacts with quantization can do both, and it is reported as an effect of quantization. We estimate this interaction for four measurement choices in 8 instruction-tuned models below two billion parameters, at float16, LLM.int8() and NF4, on identical items. Measured against labels from two human annotators, the 4-bit rise in harmful compliance is about half to two-thirds of the rise in non-refusal that a corrected key word detector records. Cloze and lettered multiple-choice scoring disagree about the effect of quantization far more than their noise allows: their estimates correlate at \r{ho} = 0.10, against a ceili
    
[^38]: 场生命：作为相互作用化学场的粒子生命

    Field Life: Particle Life as Interacting Chemical Fields

    [https://arxiv.org/abs/2611.32844](https://arxiv.org/abs/2611.32844)

    本文提出 Field Life，将 Particle Life 的粒子交互机制转化为一种质量守恒的元胞自动机，其中各颜色作为化学场浓度，通过环形卷积核与交互矩阵驱动演化，随机规则即可自发产生滑行体等运动结构。

    

    Field Life（场生命）是一种受 Particle Life（粒子生命）交互动力学启发的元胞自动机。在 Particle Life 中，每个粒子具有一种颜色，交互矩阵决定每种颜色对其他颜色的吸引或排斥强度。在 Field Life 中，每种颜色对应自动机的一个通道，每个单元格存储所有颜色的浓度。规则的单步演化包含三个部分：首先，每种颜色与一个环形卷积核进行卷积，以测量某单元格周围该颜色的分布量；其次，交互矩阵将这些测量值合成为每种颜色的吸引力；最后，每种颜色的质量以守恒的方式向吸引力更高的方向移动。通常情况下，一个随机的交互矩阵和随机的初始状态就足以产生运动的结构：例如滑行的物体、释放扩散云雾的物体，以及分裂成更小滑行体的滑行体。研究者随机采样规则，保留能够运动并形成结构的规则，并对这些模式进行描述。

    arXiv:2611.32844v1 Announce Type: cross  Abstract: Field Life is a cellular automaton inspired by the interaction dynamics of Particle Life. In Particle Life, each particle has a colour, and an interaction matrix sets how strongly each colour attracts or repels each other colour. In Field Life, each colour is a channel of the automaton, and each cell holds a concentration of every colour. One step of the rule has three parts. Each colour is convolved with a ring-shaped kernel, which measures how much of that colour surrounds a cell. The interaction matrix combines these measurements into an attraction for each colour. The mass of each colour then moves toward higher attraction, in a way that conserves mass. A random matrix and a random initial state are usually enough to produce moving structure: bodies that glide, bodies that shed diffuse clouds, and gliders that break up into smaller gliders. We sample rules at random, keep the rules that move and form structure, and describe the pat
    
[^39]: FORA：实现任意计算机断层扫描的可靠且有定位依据的报告生成

    Reliable and grounded reporting of any computed tomography scan with FORA

    [https://arxiv.org/abs/2611.32823](https://arxiv.org/abs/2611.32823)

    提出通用CT模型FORA，可识别并空间定位1,028个放射学概念、覆盖28种CT检查方案自动生成报告，无需人工标注病灶即可学习异常定位，并在胸片基准测试中达到最先进性能。

    

    用于计算机断层扫描（CT）的人工智能系统在很大程度上仍然局限于特定的解剖区域、检查类型或预定义的异常，这限制了它们提供全面且具有空间定位依据的解读能力。在此，我们提出了FORA，一个通用CT模型，它能够识别1,028个分层组织的放射学概念并对其进行空间定位，并为我们队列中涵盖的28种CT检查方案生成报告。FORA在304,962例常规CT检查及其随附报告的监督下进行训练，无需人工标注病灶即可学会定位异常。其架构将多示例学习与可解释的概念瓶颈相结合，连接了空间证据、放射学发现与报告生成。我们首先在胸片上开发并对该架构进行了基准测试，取得了最先进的性能，随后训练了一个单独的模型

    arXiv:2611.32823v1 Announce Type: cross  Abstract: Artificial intelligence systems for computed tomography (CT) remain largely restricted to specific anatomical regions, examination types or predefined abnormalities, limiting their ability to provide comprehensive and spatially grounded interpretations. Here we present FORA, a general-purpose CT model that recognizes and spatially localizes 1,028 hierarchically organized radiological concepts and generates reports across the 28 CT protocols represented in our cohort. Trained on 304,962 routine CT examinations supervised by their accompanying reports, FORA learns to localize abnormalities without manually annotated lesions. Its architecture combines multiple-instance learning with an interpretable concept bottleneck that connects spatial evidence, radiological findings and report generation. We first developed and benchmarked the architecture on chest radiographs, achieving state-of-the-art performance, before training a separate model 
    
[^40]: 实时交互式视频生成的残差引导

    Residual Guidance for Real-Time Interactive Video Generation

    [https://arxiv.org/abs/2611.32820](https://arxiv.org/abs/2611.32820)

    提出了一种无需训练的残差引导方法（RG），通过在缓存之外计算的校正残差来抵消流式视频生成中的上下文惯性，使模型在提示词切换时能迅速响应新指令并保持视觉连续性。

    

    实时交互式视频生成允许用户在生成过程中修改提示词，以引导正在生成的视频内容。然而，现有的逐块生成视频的流式模型在提示词切换后往往仍继续遵循之前的提示词。我们将这种失败归因于“上下文惯性”，即前一个视频块作为强先验抑制了新提示词的作用。此外，我们发现即使缓存中仅保留前一个块，这种惯性依然存在；而丢弃上下文又会破坏视觉连续性，这使得编辑缓存类的方法受到限制。我们不再编辑缓存，而是提出残差引导（Residual Guidance, RG）——一种简单且无需训练的方法，通过在缓存之外测量的校正来抵消上下文惯性。具体而言，RG 在无缓存的条件下，使用新提示词对重新加噪的前一个块进行一次去噪，并将得到的残差添加到下一个块的早期去噪步骤中，其内容在该阶段……

    arXiv:2611.32820v1 Announce Type: cross  Abstract: Real-time interactive video generation allows users to change the prompt during generation to steer the evolving video. However, existing streaming models, which generate videos chunk by chunk, often keep following the previous prompt after a switch. We trace this failure to context inertia, where the previous chunk acts as a strong prior that suppresses the new prompt. Furthermore, we find that this inertia persists even when the cache retains only the previous chunk, yet discarding the context breaks visual continuity, which limits approaches that edit the cache. Rather than editing the cache, we propose Residual Guidance (RG), a simple, training-free method that counteracts context inertia with a correction measured outside the cache. Specifically, RG denoises the re-noised previous chunk once under the new prompt without the cache and adds the resulting residual to the early denoising steps of the next chunk, where its content is l
    
[^41]: 交互式神经计算机

    Interactive Neural Computers

    [https://arxiv.org/abs/2611.32811](https://arxiv.org/abs/2611.32811)

    该论文提出不再训练智能体去操作现有计算机，而是训练一个生成式世界模型直接充当计算机，根据历史屏幕画面和用户的鼠标键盘操作预测下一屏画面，从而解决记忆索引、精确计算和小文本清晰度等独特挑战，实现可定制、无障碍的交互式界面。

    

    计算机及其接口通常被视为固定的，因此用户——以及越来越多的AI智能体——必须去适应操作系统所呈现的一切。当人们需要不同的工作流程，或需要一个适合免手操作或残障人士使用的界面时，这种设计就成了限制。我们探索了一种被忽视的替代方案：与其训练一个智能体去操作现有的计算机，不如训练一个生成模型来充当计算机本身，根据过去的屏幕画面以及用户的鼠标和键盘操作来预测下一屏画面。将桌面视为一个交互式世界模型会带来自然视频世界模型大多可以规避的问题：在没有相机位姿来索引记忆的情况下，模型必须记住那些已经离开屏幕的内容，且没有任何几何线索可循；终端中的算术运算必须完全正确，而不仅仅是看似合理；小的界面文本还必须以交互式帧率保持清晰可读。我们的交互式神经计算机（Interactive Neural Computer）在结构上类似于神经……（摘要在此处截断）

    arXiv:2611.32811v1 Announce Type: new  Abstract: Computers and their interfaces are usually treated as fixed, so a person, or increasingly an AI agent, must adapt to whatever the operating system presents. This is limiting when someone needs a different workflow, or an interface suited to hands-free use or a disability. We explore an overlooked alternative. Rather than training an agent to operate an existing computer, we train a generative model to be the computer, predicting each next screen from past screens and the user's mouse and keyboard actions. Treating a desktop as an interactive world model raises problems that natural-video world models largely avoid. Without a camera pose to index memory, the model must remember content that has left the screen without any geometric cues. Arithmetic in a terminal has to be exactly right, not merely plausible. Small interface text must also stay legible at interactive frame rates. Our Interactive Neural Computer is structured like a Neural 
    
[^42]: MRBench：异构多机器人操作中物理必要性推理的基准测试

    MRBench: Benchmarking Physical Necessity Reasoning in Heterogeneous Multi-Robot Manipulation

    [https://arxiv.org/abs/2611.32803](https://arxiv.org/abs/2611.32803)

    MRBench是一个评估异构多机器人操作中物理必要性推理能力的基准，通过八类任务和五种资源配置揭示了现有协调智能体在机器人形态变化时难以判断哪些机器人真正必要。

    

    多机器人操作的最新进展展示了机器人如何协作完成复杂任务。然而，成功的协作并不能揭示哪些机器人是真正需要的。随着机器人能力的变化、资源的不可用或物体替代机器人支持，同一任务可能需要不同的团队。我们认为，可靠的多机器人操作需要物理必要性推理：理解任务的需求、哪些资源能够满足这些需求，以及哪些资源是真正必要的。我们提出了MRBench，一个用于异构多机器人操作中物理必要性推理的基准测试。它包含八个具有不同并发性和可达性约束的任务，每个任务在五种配置下进行评估，这些配置在保持任务目标不变的同时改变所需资源。我们的评估表明，现有的协调智能体在使用参考团队时表现良好，但当机器人形态发生变化时则难以应对。

    arXiv:2611.32803v1 Announce Type: cross  Abstract: Recent advances in multi-robot manipulation have shown how robots can collaborate to accomplish complex tasks. However, successful coordination does not reveal which robots are actually needed. The same task may require different teams as robot capabilities change, resources become unavailable, or objects replace robotic support. We argue that reliable multi-robot manipulation requires physical necessity reasoning: understanding what a task demands, which resources can fulfill those demands, and which are truly necessary. We introduce MRBench, a benchmark for physical necessity reasoning in heterogeneous multi-robot manipulation. It comprises eight tasks with distinct concurrency and reach constraints, each evaluated under five configurations that change the required resources while preserving the task goal. Our evaluation reveals that existing coordination agents perform well with reference teams but struggle when robot embodiments ch
    
[^43]: 基于睡眠机制的样本高效上下文适应

    Sample-Efficient In-Context Adaptation via Sleep

    [https://arxiv.org/abs/2611.32802](https://arxiv.org/abs/2611.32802)

    该研究提出在Transformer语言模型预训练中引入“睡眠”机制（通过多次循环前向传递精炼上下文表示），显著提升了模型在一般性任务上少样本上下文学习的样本效率。

    

    arXiv:2611.32802v1 公告类型：交叉 摘要：从有限经验中学习并泛化的能力，即样本效率，是智能的基石。我们如何构建能够将少量用户交互、几段陌生代码或上下文中提供的少量实验结果转化为可泛化知识的语言模型？先前的研究发现，“睡眠”——即模型执行 N 次循环前向传递以精炼其对一段上下文表示的过程——可以改善长上下文推理（具体而言，在 SSM-注意力混合模型中）。我们研究这种方法是否也能改善Transformer语言模型在一般性、非合成任务上的上下文适应能力。为此，我们使用睡眠机制在多达 1000 亿 token 上预训练了参数量高达 12 亿的语言模型。值得注意的是，更长的睡眠（即使用更多的循环前向传递次数 N）使模型能够在多个少样本算法任务上更快地进行上下文学习……

    arXiv:2611.32802v1 Announce Type: cross  Abstract: The ability to learn and generalize from limited experience, or sample efficiency, is a cornerstone of intelligence. How can we build language models that turn a few user interactions, a few snippets of unfamiliar code, or a handful of experimental results provided in context into knowledge that generalizes? Prior work has found that sleep, a process where the model performs $N$ recurrent forward passes to refine its representations of a chunk of context, can improve long-context reasoning (specifically, within SSM-attention hybrid models). We study whether this method also improves in-context adaptation on general, non-synthetic tasks for transformer language models. To this end, we pre-train language models up to 1.2B-parameters on up to 100B tokens with sleep mechanisms. Notably, longer sleep (i.e., using a larger number of recurrent forward passes $N$) enables the model to learn faster in-context on multiple few-shot algorithmic, q
    
[^44]: 语言模型中集成不确定性的缩放定律

    Scaling Laws of Ensemble Uncertainty in Language Models

    [https://arxiv.org/abs/2611.32801](https://arxiv.org/abs/2611.32801)

    本文提出三条幂律缩放定律，证明集成随模型规模增大会捕获越来越多的认知不确定性，作为扩展路径相比单个更大的模型更具竞争力。

    

    大型语言模型（LLM）的许多应用场景——从AI安全到探索与人类交互——都能从认知不确定性估计中受益。我们认为，现有的贝叶斯深度学习解决方案通常局限于固定的模型族，在实际的LLM规模下无法捕捉完整的认知不确定性。这是因为由模型族有限容量导致的误差，对于先验所支持的模型之间的分歧而言是不可见的。我们将这一误差与著名的LLM损失随模型规模的幂律缩放联系起来。随后，我们证明第二条幂律量化了集成内部分歧随模型规模增大而缩小的现象。相对于上述容量定律，该幂律较慢的衰减速度表明，集成能够捕获越来越多的认知不确定性份额，与单个更大的模型相比，集成作为一条扩展路径变得更具竞争力。最后，第三条幂律量化了集成……（摘要在此处截断）

    arXiv:2611.32801v1 Announce Type: cross  Abstract: Many use cases for large language models (LLMs)---from AI safety to exploration and human interaction---can benefit from epistemic uncertainty estimates. We argue that existing Bayesian Deep Learning solutions, which are often restricted to a fixed model family, fail to capture the full epistemic uncertainty at practical LLM scales. This is because the error due to the bounded capacity of the model family remains invisible to disagreement among models supported by the prior. We relate this error to the well-known power-law scaling of LLM loss with model size. We then show that a second power law quantifies the shrinking disagreement within ensembles as model size increases. Its slower decay relative to the capacity law above suggests that ensembles capture a growing share of epistemic uncertainty and become more competitive as an axis for scaling, compared to a single larger model. Finally, a third power law quantifies the loss of ense
    
[^45]: 脑分数噪声天花板估计器的准确性

    Accuracy of Noise Ceiling Estimators for Brain Scores

    [https://arxiv.org/abs/2611.32792](https://arxiv.org/abs/2611.32792)

    该论文提出了一个统一的大脑记录数学框架，并据此推导多种噪声天花板估计器的闭式表达式，以明确这些估计器在不同实验设置下能否为脑分数提供可靠的上界估计。

    

    现代机器学习模型的内部激活与真实大脑中记录到的神经活动惊人地吻合。在过去五年中，编码模型与大脑活动之间的对齐程度（通常称为“脑分数”）稳步提升。由于大脑记录含有噪声，任何模型所能达到的脑分数都存在一个上界。这一上界被称为“噪声天花板”，其重要性体现在两个方面：它决定了模型性能是否还有进一步提升的空间，并且使得跨基准的脑分数比较具有实际意义。该领域已发展出多种估计噪声天花板的方法，但现有估计器的统计特性仍缺乏深入理解，导致人们不清楚它们在哪些实验设置下能够提供可靠的估计。本文中，我们引入了一个针对大脑记录的统一数学框架，并利用它推导出了四种（估计器的闭式表达式）……

    arXiv:2611.32792v1 Announce Type: new  Abstract: The internal activations of modern machine learning models align surprisingly well with the activities recorded in real brains. In the last five years, the alignment between encoding models and brain activities, often called brain scores, has steadily increased. Because brain recordings are noisy, there exists an upper bound to the brain scores that any model can achieve. This bound, called the noise ceiling, is crucial for two reasons: it determines whether further improvements in model performance are possible, and it enables meaningful comparisons of brain scores across benchmarks. The field has developed multiple approaches to estimate noise ceilings, but the statistical properties of existing estimators remain poorly understood, leaving it unclear in which experimental settings they provide reliable estimates. Here, we introduce a unified mathematical framework for brain recordings, use it to derive closed-form expressions for four 
    
[^46]: C3H：多模态大语言模型推荐系统中的“压缩致共识”准则劫持攻击

    C3H: Compression-to-Consensus Criteria Hijacking in Multimodal LLM Recommender Systems

    [https://arxiv.org/abs/2611.32780](https://arxiv.org/abs/2611.32780)

    本文提出C3H推理时攻击，揭示了多模态LLM推荐系统中一个反直觉漏洞：压缩与跨模态共识机制非但不能防御内容操纵，反而会放大“压缩致共识”偏差，攻击者无需投毒或扰动，仅靠定制目标商品的多模态内容即可提升其推荐排名。

    

    多模态大语言模型（MLLM）正在将推荐系统转变为内容感知的决策系统。在这些系统中，多模态压缩与跨模态融合通常被视为增强鲁棒性、抵御内容操纵的机制，因为压缩可以过滤噪声信号，而跨模态共识可以抑制不一致的异常。我们挑战了这一假设，展示出同一过程可能在排序决策中暴露一种“压缩致共识”（C2C）偏差：那些在压缩后仍保持显著、并在多个模态间获得一致支持的信号，可以在候选比较中获得不成比例的影响力，从而为内容操纵提供了可利用的攻击方向。本文提出了C3H，一种利用该漏洞的新型推理时攻击。与依赖噪声扰动或训练期投毒的方法不同，C3H通过定制目标商品的多模态内容来提升其推荐排名……（原文摘要在此处截断）

    arXiv:2611.32780v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) are transforming recommender systems into content-aware decision systems. In these systems, multimodal compression and cross-modal fusion are often viewed as robustness-enhancing mechanisms against content manipulation, as compression can filter noisy signals and cross-modal consensus can suppress inconsistent anomalies. We challenge this assumption by showing that the same process can expose a compression-to-consensus (C2C) bias in ranking decisions. Signals that remain salient after compression and receive consistent support across modalities can gain disproportionate influence in candidate comparison, offering an exploitable direction for content manipulation. In this paper, we introduce C3H, a novel inference-time attack that exploits this vulnerability. Rather than relying on noisy perturbations or training-time poisoning, C3H promotes a target item by tailoring its multimodal content to al
    
[^47]: 通过动作的对比识别来学习世界模型

    Learning World Models through Contrastive Identification of Actions

    [https://arxiv.org/abs/2611.32776](https://arxiv.org/abs/2611.32776)

    本文提出动作对比识别方法CIA，通过在情境合理的备选动作中识别实际执行的动作来学习世界模型，能够捕捉逆动力学回归无法奖励的动作信息，从而提升离线视觉规划的平均成功率。

    

    潜在世界模型能够基于离线视觉数据进行目标导向的规划，但仅靠预测本身并不能明确其表示应当保留哪些与动作相关的区分信息。逆动力学提供了辅助监督，然而均方误差的动作回归只能通过条件动作均值的变化来奖励额外信息。我们提出了动作对比识别方法（Contrastive Identification of Actions, CIA），该方法在从以历史为条件的行为提议中采样的备选动作之间识别实际执行的动作。在同一情境下比较合理的动作，可以针对性地获取下一个潜变量超越历史所提供的信息。在精确行为匹配的条件下，我们证明了CIA能够奖励那些不会改变最优逆回归误差的动作信息，并通过一个受控学习示例验证了这一区别。CIA相比基础世界模型和回归式逆动力学基线，在规划器平均成功率上取得了更高的表现。

    arXiv:2611.32776v1 Announce Type: new  Abstract: Latent world models enable goal-directed planning from offline visual data, but prediction alone does not specify which action-related distinctions their representations should preserve. Inverse dynamics provides auxiliary supervision, yet squared-error action regression rewards additional information only through changes in the conditional action mean. We propose Contrastive Identification of Actions (CIA), which identifies the executed action among alternatives sampled from a history-conditioned behavior proposal. Comparing actions plausible in the same context targets information supplied by the next latent beyond history. Under exact behavior matching, we show that CIA can reward action information that leaves optimal inverse-regression error unchanged, and verify this distinction through a controlled learning example. CIA achieves higher planner-averaged mean success than the base world model and a regression inverse-dynamics baseli
    
[^48]: RubikSQL-HB：知识在环的智能体式 Text-to-SQL

    RubikSQL-HB: Agentic Text-to-SQL with Knowledge in the Loop

    [https://arxiv.org/abs/2611.32772](https://arxiv.org/abs/2611.32772)

    RubikSQL-HB 通过 HeavenBase 平台上可查询的数据库专属知识工作区，将业务含义、存储值和参考查询等知识持续融入 Text-to-SQL 智能体的推理全程，使候选生成与下游裁判都能随时访问知识进行核查与仲裁，在 BIRD Mini-Dev 上达到 75.2% 的执行准确率，显著优于首次候选选择和多数投票策略。

    

    数据库智能体需要的不仅仅是模式结构信息：业务含义、存储的数值和参考查询所提供的知识会在推理的不同阶段变得相关。我们提出 RubikSQL-HB，它通过构建在 HeavenBase 平台之上的可查询、针对特定数据库的知识工作区来开放这些材料。在智能体构建并核查答案的整个过程中，其知识接口始终可用。候选智能体将知识访问与 SQL 的构建和执行交错进行；下游的裁判可以通过复用相同的接口来调查候选之间的分歧。一个两阶段的示例接口将“发现参考问题”与“扩展其 SQL”分离开来。在包含 500 个问题的 BIRD Mini-Dev 评估上，完整流水线达到了 75.2% 的执行准确率，相比之下首次候选选择为 71.2%，执行多数投票为 70.4%，且有 79.0% 的问题存在正确候选。（注：原文摘要在"原始知识条件"部分被截断，此处仅翻译到已有内容。）

    arXiv:2611.32772v1 Announce Type: new  Abstract: Database agents need more than schema structure: business meanings, stored values and reference queries supply knowledge that becomes relevant at different points in reasoning. We introduce RubikSQL-HB, which exposes these materials through queryable, database-specific knowledge workspaces built on the HeavenBase platform. Its knowledge interface remains available as an agent constructs and investigates an answer. Candidate agents interleave knowledge access with SQL construction and execution; a downstream Judge can reuse the same interfaces to investigate disagreements. A two-stage example interface separates discovering a reference question from expanding its SQL. On a 500-question BIRD Mini-Dev evaluation, the full pipeline reaches 75.2% execution accuracy, against 71.2% for first-candidate selection and 70.4% for execution-majority voting, with a correct candidate available on 79.0% of questions. The raw-knowledge condition, which a
    
[^49]: EviCraft：Minecraft智能体能否找到它们看不见的东西？

    EviCraft: Can Minecraft Agents Find What They Cannot See?

    [https://arxiv.org/abs/2611.32765](https://arxiv.org/abs/2611.32765)

    该论文提出“证据闭合”概念并构建EviCraft运行时与基准，强制智能体只能依据可见信息或自身行动所得的事实做决策，实验表明仅切换信息准入机制就使中等规模智能体找到钻石的成功率大幅下降，其影响甚至超过不同智能体之间的能力差距。

    

    在Minecraft这一常用的具身智能体测试平台中，mineBlock等语义工具直接从模拟器已加载的世界状态中选取目标，而不进行可见性检测，因此智能体从未见过的埋藏矿石只需一次调用即可采到，其得分也无法反映智能体是否真正进行了搜索。我们将这一缺失的要求形式化为“证据闭合”（evidence closure）：智能体系统的每一次决策——包括其工具内部的决策——只能通过智能体持有证据的事实来依赖世界。EviCraft是一个运行时环境与基准测试，它在智能体之下强制执行闭合，并将信息准入规则作为实验变量：感知基础机制（G）只允许视线范围内或由智能体自身行动返回的信息进入，而已加载状态机制（L）则复现标准工具栈，同时保持智能体、工具和世界完全不变。在生存模式下，机制效应超过了最强智能体的领先优势：在G机制下，三个中等规模的智能体以32%-4（原文摘要在此处截断）……

    arXiv:2611.32765v1 Announce Type: new  Abstract: In Minecraft, a common testbed for embodied agents, semantic tools such as mineBlock pick targets from the simulator's loaded world state without a visibility test, so a buried ore the agent has never seen is one call away, and scores cannot show whether the agent searched. We formalize the missing requirement as evidence closure: every decision of an agent system, including those inside its tools, may depend on the world only through facts for which the agent holds evidence. EviCraft is a runtime and benchmark that enforces closure below the agent and makes the admission rule an experimental variable: a perception-grounded regime (G) admits only what is in line of sight or returned by the agent's own actions, and a loaded-state regime (L) reproduces standard tool stacks, with agents, tools, and worlds held fixed. On Survival, the regime effect exceeds the strongest agent's lead: under G, three mid-sized agents reach the diamond in 32%-4
    
[^50]: 大学习率下的随机梯度下降：一般性分析与改进的收敛速率

    SGD with Large Learning Rates: General Analysis & Improved Rates

    [https://arxiv.org/abs/2611.32762](https://arxiv.org/abs/2611.32762)

    本文证明了大学习率的SGD在随机梯度噪声存在下仍能加速收敛，对线性预测器的交叉熵损失和两层神经网络建立了加速的O(T^-2)收敛速率，并在稳定区域内给出了更紧致的收敛保证。

    

    随机梯度下降（SGD）是现代机器学习的核心。然而，其经典分析通常将步长限制在低于一个依赖于曲率的稳定性阈值以下。确定性优化领域的最新进展挑战了这一范式，表明适当选择的大步长不仅可以保持收敛性，还能加速梯度下降（GD）。这一现象在存在随机梯度噪声的情况下是否仍然成立，一直是一个悬而未决的问题（Wu et al., 2024）。在本工作中，我们对带有线性预测器的交叉熵损失以及两层神经网络肯定地回答了这一问题，建立了加速的 $\mathcal{O}(T^{-2})$ 收敛速率。我们分析的核心是对随机大步长动力学的细粒度刻画。这种对轨迹更精确的控制进一步在稳定区域内给出了紧致的收敛保证，改进了……

    arXiv:2611.32762v1 Announce Type: cross  Abstract: Stochastic Gradient Descent lies at the core of modern machine learning. However its classical analysis typically restricts the stepsize to be below a curvature-dependent stability threshold. Recent developments in deterministic optimization have challenged this paradigm, showing that suitably chosen large stepsizes can not only preserve convergence, but also accelerate Gradient Descent (GD). Whether this phenomenon persists in the presence of stochastic gradient noise has remained an open question (Wu et al., 2024). In this work, we answer this question affirmatively for the cross-entropy loss with linear predictors and two-layer neural networks, establishing an accelerated $\mathcal{O}(T^{-2})$ convergence rate. At the core of our analysis is a fine-grained characterization of the stochastic large-stepsize dynamics. This sharper control of the trajectory further yields tight convergence guarantees within the stable regime, improving 
    
[^51]: ReQOR：面向大语言模型量化条件后门的算子差异子空间修复

    ReQOR: Operator-Discrepancy Subspace Repair for Quantization-Conditioned Backdoors in Large Language Models

    [https://arxiv.org/abs/2611.32755](https://arxiv.org/abs/2611.32755)

    ReQOR提出了一种无需训练的防御方法，通过在良性校准数据上识别模块级量化算子差异的主导方向，并以闭式形式恢复全精度计算，有效修复大语言模型中的量化条件后门。

    

    量化条件后门会使大语言模型在量化后表现出恶意行为，而在高精度状态下保持良性。现有防御方法通常依赖安全导向的配置选择或迭代优化，但未能识别哪些由量化引发的变化应当被修复。我们通过在来自量化推理轨迹的相同状态上评估每个模块的高精度版本与量化版本，将模块局部的量化效应与上游漂移分离，并将量化条件后门防御形式化为修复由此产生的同状态算子差异。我们提出ReQOR，一种无需训练的防御方法，它在良性校准数据上识别这些差异的主导方向，并以闭式形式沿这些方向恢复全精度计算，从而产生一个因子化的残差适配器。ReQOR既不使用攻击信息，也不使用基于梯度的优化，并支持……

    arXiv:2611.32755v1 Announce Type: cross  Abstract: Quantization-Conditioned Backdoors (QCBs) cause large language models (LLMs) to exhibit malicious behavior after quantization while remaining benign at high precision. Existing defenses often rely on security-guided configuration selection or iterative optimization, but do not identify which quantization-induced changes should be repaired. We isolate module-local quantization effects from upstream drift by evaluating the high-precision and quantized versions of each module on identical states from the quantized inference trajectory, and formulate QCB defense as repairing the resulting same-state operator discrepancies. We propose ReQOR, a training-free defense that identifies the dominant directions of these discrepancies on benign calibration data and restores the full-precision computation along them in closed form, yielding a factorized residual adapter. ReQOR uses neither attack information nor gradient-based optimization and suppo
    
[^52]: ASCENT：一个用于评估和推进大语言模型在常见临床场景中逐步诊断推理能力的基准

    ASCENT: A Benchmark for Evaluating and Advancing Stepwise Diagnostic Reasoning in Large Language Models on Common Clinical Scenarios

    [https://arxiv.org/abs/2611.32734](https://arxiv.org/abs/2611.32734)

    该论文提出ASCENT，一个由临床医生标注、包含3,078个逐步推理问题的基准，通过将病例分解为与电子病历对齐的步骤（发现、印象、依据）来评估信息不完整条件下的中间诊断推理，揭示出即使最强的推理模型在逐步诊断任务上仍有显著提升空间。

    

    大语言模型（LLM）在医学问答方面表现出色，但很少在真实临床工作流程所定义的逐步诊断推理上进行评估——在这种推理中，诊断印象会随着信息的积累而不断修正。我们构建了ASCENT（Annotated Stepwise Clinical rEasoning for NaturalisTic Diagnosis，面向自然主义诊断的注释式逐步临床推理），这是一个由临床医生标注的基准与训练资源，包含3,078个源自MedQA-USMLE的逐步推理问题，将每个病例描述分解为与电子病历（EMR）对齐的步骤（检查发现、诊断印象、支持性依据），从而能够在信息不完整的情况下评估中间推理过程。基于ASCENT的实验与训练揭示了当前大语言模型处理逐步诊断推理的现状：即使在MedQA-USMLE上表现优异的强推理模型，在ASCENT上仍有巨大的提升空间，而通用前沿模型则落后更多，这暴露了“信息完整诊断”与“逐步诊断”之间存在的持续差距。对Qwen2.5-（摘要在此处截断）

    arXiv:2611.32734v1 Announce Type: cross  Abstract: Large language models (LLMs) excel at medical question answering yet are rarely evaluated on the stepwise diagnostic reasoning that defines real clinical workflows, where impressions are revised as information accumulates. We build Annotated Stepwise Clinical rEasoning for NaturalisTic Diagnosis (ASCENT), a clinician-annotated benchmark and training resource of 3,078 stepwise problems derived from MedQA-USMLE that decomposes each vignette into EMR-aligned steps (Findings, Impression, supporting Rationale), enabling evaluation of intermediate reasoning under incomplete information.   Experiments and training with ASCENT revealed insights into how current LLMs handle stepwise diagnostic reasoning. Even strong reasoning models that perform well on MedQA-USMLE leave substantial headroom on ASCENT, and general-purpose frontier models trail further, exposing a persistent gap between fully informed and stepwise diagnosis. Fine-tuning Qwen2.5-
    
[^53]: FourStat：一种利用频率与平稳性的双自编码器时间序列异常检测方法

    FourStat: A Dual AutoEncoder Leveraging Frequency and Stationarity for Time Series Anomaly Detection

    [https://arxiv.org/abs/2611.32714](https://arxiv.org/abs/2611.32714)

    提出FourStat双自编码器框架，结合频域分析的傅里叶自编码器与强化统计平稳性的平稳自编码器来检测单变量时间序列异常，并引入Perfect Point新指标以消除阈值选择的主观性。

    

    尽管大量研究聚焦于多变量时间序列异常检测，但许多现有方法在单变量场景下表现不佳或无法直接应用。我们提出了FourStat，一个用于单变量时间序列异常检测的双模型框架，它结合了两种互补的方法。傅里叶自编码器（FAE）通过频域分析捕捉季节性等全局模式，而平稳自编码器（SAE）通过强化统计平稳性来关注局部结构。我们还提出了一种名为Perfect Point的新评估指标，用于评估能否直接从异常分数中获得完美的F1分数，从而消除阈值选择的主观性。结果表明，FourStat优于现有基线方法，尤其是在具有挑战性的异常类型上，同时新指标能更清晰地揭示模型性能的差距与优势。

    arXiv:2611.32714v1 Announce Type: cross  Abstract: While numerous studies have focused on multivariate time series anomaly detection, many existing methods perform poorly or are not directly applicable to univariate settings. We introduce FourStat, a dual-model framework for univariate time series anomaly detection that combines two complementary approaches. The Fourier Autoencoder (FAE) captures global patterns like seasonality through frequency-domain analysis, while the Stationary Autoencoder (SAE) focuses on local structures by enforcing statistical stationarity. We also propose a new evaluation metric called Perfect Point that assesses whether perfect F1-scores can be achieved directly from anomaly scores, eliminating thresholding subjectivity. Results demonstrate that FourStat outperforms existing baselines, particularly on challenging anomaly types, while the new metric provides clearer insights into model performance gaps and advantages.
    
[^54]: 我们是否在奖励错误的模块？面向多模态强化学习的模块特定策略优化

    Are We Rewarding the Wrong Module? Module-Specific Policy Optimization for Multimodal Reinforcement Learning

    [https://arxiv.org/abs/2611.32687](https://arxiv.org/abs/2611.32687)

    提出模块特定策略优化（MSPO），利用感知清单与置信度感知感知奖励（CaPR）将感知信号与结果信号分离，解决了多模态大语言模型强化学习中的模块级信用纠缠问题。

    

    基于组相对策略优化（GRPO）的可验证奖励强化学习（RLVR）已被广泛用于增强多模态大语言模型（MLLM）的多模态推理能力。然而，现有方法存在一个共同局限——模块级信用纠缠（Module-Level Credit Entanglement），即视觉编码器、投影器和LLM模块使用相同的优势信号进行联合优化，即使感知正确性与最终答案正确性并不一致。为解决这一局限，我们提出了模块特定策略优化（MSPO），这是一个跨模型模块将感知学习信号与基于结果的学习信号相分离的框架。具体而言，MSPO利用针对感知清单（由关于视觉信息的辅助问题组成）的响应来推导用于训练视觉编码器的优势。基于这些响应，MSPO计算置信度感知感知奖励，并使用由此获得的感知符号来……

    arXiv:2611.32687v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) based on Group Relative Policy Optimization (GRPO) has been widely adopted to enhance the multimodal reasoning of multimodal large language models (MLLMs). However, existing methods share a common limitation of Module-Level Credit Entanglement, in which the vision encoder, projector, and LLM modules are jointly optimized using the same advantage signals even when perceptual correctness and final-answer correctness differ. To address this limitation, we propose Module-Specific Policy Optimization (MSPO), a framework that separates perceptual and outcome-based learning signals across model modules. Specifically, MSPO uses responses to perception checklists comprising auxiliary questions about visual information to derive an advantage for training the vision encoder. From these responses, MSPO computes the Confidence-aware Perception Reward (CaPR), using a perception sign obtained by t
    
[^55]: CAPTA：识别AI智能体背后的大语言模型

    CAPTA: Identifying LLMs Behind AI Agents

    [https://arxiv.org/abs/2611.32686](https://arxiv.org/abs/2611.32686)

    本文提出借鉴CAPTCHA思想的CAPTA方法，通过向AI智能体呈现选择题并分析不同模型的选择偏好差异，来识别智能体背后所使用的大语言模型。

    

    基于大语言模型（LLM）的智能体通过工具访问远程服务来完成用户任务，通常经由API调用实现。在这种场景下，远程服务可能希望识别智能体所使用的后端LLM，以便发起针对特定模型的攻击，包括提示注入或选择性地投递被篡改的内容。然而，该服务无法直接查询后端模型，只能通过与智能体交互收集可观察的响应，这使得识别变得具有挑战性。本文提出了CAPTA（Completely Automated Preference Test to tell LLMs Apart，区分LLM的完全自动化偏好测试），这是一种受CAPTCHA（区分计算机与人类的完全自动化公开图灵测试）启发的模型识别方法。CAPTA以选择题作为验证挑战的形式呈现，利用模型在选择偏好上的差异来识别后端LLM，并基于特定标准来选择问题。

    arXiv:2611.32686v1 Announce Type: new  Abstract: Large language model (LLM)-based agents access remote services through tools to complete user tasks, often via API calls. In this setting, a remote service may want to identify the backend LLM of an agent to launch model-specific attacks, including prompt injection or selective delivery of manipulated content. However, the service cannot directly query the backend model and must instead collect observable responses through its interactions with the agent, making identification challenging.   In this paper, we present Completely Automated Preference Test to tell LLMs Apart (CAPTA), a model identification approach inspired by CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart), which distinguishes humans from automated programs. CAPTA presents multiple-choice questions as verification challenges, using differences in models' selection preferences to identify the backend LLM. It selects questions based on cr
    
[^56]: 经济世界模型

    Economic World Model

    [https://arxiv.org/abs/2611.32663](https://arxiv.org/abs/2611.32663)

    本文提出经济世界模型（EWM）的概念及通用研究框架，通过从微观经济数据学习家庭策略并结合宏观数据校准结构性经济转移模型，实现了个体决策与总体经济动态相连接的微观到宏观模拟与评估。

    

    人工智能能否复现家庭、企业和政府通过市场与制度相互作用及适应所产生的经济动态？为了回答这个问题，我们将经济世界模型（EWM）定义为智能体对经济系统运行方式的内部表征，使其能够在资源约束下进行反事实模拟和决策。随后，我们提出了一个研究经济世界模型的通用框架，该框架连接了三项任务：模拟智能体行为、学习经济转移规律以及系统性地评估动态。作为初步实现，我们从微观经济数据中学习家庭策略，并利用宏观经济数据校准结构性经济转移模型，从而将个体决策与总体动态联系起来。我们的微观到宏观评估框架将模拟的个体行为和群体分布与观测到的微观数据进行比较，并通过预测来评估宏观经济动态。

    arXiv:2611.32663v1 Announce Type: new  Abstract: Can artificial intelligence reproduce the economic dynamics that emerge as households, firms and governments interact and adapt through markets and institutions? To answer this question, we define an economic world model (EWM) as an agent's internal representation of how an economic system works, enabling counterfactual simulation and decision-making under resource constraints. We then propose a general framework for studying EWMs that links three tasks: simulating agent behaviour, learning economic transitions and systematically evaluating the dynamics. As an initial implementation, we learn household policies from microeconomic data and calibrate a structural economic transition model using macroeconomic data, linking individual decisions to aggregate dynamics. Our micro-to-macro evaluation framework compares simulated individual actions and group distributions with observed microdata and assesses macroeconomic dynamics through forecas
    
[^57]: Bathys：深度搜索智能体的进阶

    Bathys: Advancing Deep Search Agents

    [https://arxiv.org/abs/2611.32660](https://arxiv.org/abs/2611.32660)

    提出 Bathys 9B 和 27B 深度搜索智能体，通过可验证问题合成、SFT/KTO/CM-GSPO 训练流水线以及上下文压缩下的在线强化学习框架，提升智能体在复杂开放式任务中迭代检索与证据验证的能力。

    

    深度搜索是智能体应对复杂、开放式任务的关键能力，在这类任务中，解决问题需要迭代式的信息收集、查询重构以及基于证据的验证，而非单次检索步骤。为满足这些需求，我们提出了 Bathys 9B 和 Bathys 27B 两个深度搜索智能体，其训练框架涵盖可验证的问题合成、后训练以及上下文压缩下的在线强化学习（RL）。我们高质量的数据构建确保了搜索深度与可靠性：每条线索都被改写为需要独立检索的间接指代，并保留证据链以便验证这些线索共同确定唯一答案。经过精心整理的轨迹数据支撑了一个由监督微调（SFT）、Kahneman-Tversky 优化（KTO）以及压缩管理组序列策略优化（CM-GSPO）组成的训练流水线：SFT 提供了稳定的基础……

    arXiv:2611.32660v1 Announce Type: cross  Abstract: Deep search is a critical capability for agents tackling complex, open-ended tasks, where solving a problem requires iterative information gathering, query reformulation, and evidence-based verification rather than a single retrieval step. To address these requirements, we present Bathys 9B and 27B deep search agents built around a training framework that spans verifiable question synthesis, post-training, and online reinforcement learning (RL) under context compression. Our high-quality data construction ensures both search depth and reliability: each clue is reformulated as an indirect reference requiring independent retrieval, and retained evidence chains enable verification that the clues jointly determine a unique answer. Curated trajectories underpin a pipeline of supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Compaction-Managed Group Sequence Policy Optimization (CM-GSPO): SFT provides a stable foundation
    
[^58]: GeoSurge：面向可审计肝切除术规划的基于几何的多模态推理

    GeoSurge: Geometry-Grounded Multimodal Reasoning for Auditable Liver Resection Planning

    [https://arxiv.org/abs/2611.32657](https://arxiv.org/abs/2611.32657)

    GeoSurge 提出了一个以解剖学为基础的框架，通过面-动作语法将动作映射为参数化几何，并结合掩码度量与结构感知的点云标记，生成可执行、可编辑且可审计的肝切除面。

    

    手术切除规划的核心在于从解剖学证据和手术操作约束出发，生成可执行的切除面。医学多模态基础模型能够实现解剖学理解，分割模型可以预测切除区域，但二者都无法将这些能力转化为显式约束、可执行的切除面。我们提出了 GeoSurge，这是一个以解剖学为基础的框架，它将掩码派生的度量指标与结构感知的点云标记相结合，以生成可执行且可编辑的切除面。为了使切除几何形状可学习、可编辑，GeoSurge 通过一种“面-动作”语法将动作映射为参数化几何。利用该语法，面向解剖结构的模板和以肿瘤为中心的模板生成候选切除面，再通过硬约束和语义去重进行筛选。随后，训练监督信号由候选方案与专家方案的几何一致性以及独立重新计算的可行性度量推导得出。（原文摘要在此处截断）

    arXiv:2611.32657v1 Announce Type: new  Abstract: Surgical resection planning centers on generating executable resection surfaces from anatomical evidence and operative constraints. Medical multimodal foundation models enable anatomical understanding, and segmentation models predict resection regions, but neither translates these capabilities into explicitly constrained, executable resection surfaces. We present GeoSurge, an anatomy-grounded framework that combines mask-derived metrics with structure-aware point-cloud tokens to generate executable, editable resection surfaces. To make resection geometry learnable and editable, GeoSurge maps actions to parametric geometry through a surface-action grammar. Using this grammar, anatomy-oriented and tumor-centered templates generate candidates, which are filtered by hard constraints and semantic deduplication. Training supervision is then derived from candidate-expert geometric agreement and independently recomputed feasibility metrics. At i
    
[^59]: DexWAC：面向灵巧机器人操作的统一世界-演员-评论家模型

    DexWAC: A Unified World-Actor-Critic Model for Dexterous Robot Manipulation

    [https://arxiv.org/abs/2611.32654](https://arxiv.org/abs/2611.32654)

    该论文提出DexWAC，在单一预训练阶段统一学习演员、动作条件世界模型和自举评论家，利用视觉预测与价值监督的互补作用塑造兼具控制相关性与任务相关性的表征，从而提升灵巧机器人操作能力。

    

    机器人演示所提供的信息远不止可供模仿的动作：它们揭示了动作如何改变世界，以及这些变化如何促成任务成功。近期的世界-动作模型通过联合预测动作与未来观测，为视觉-语言-动作模型引入了第一种信号，但这种预测能力仅作为策略学习的辅助损失。与此同时，大多数策略改进方法在相互分离的模型中学习动力学与价值，这增加了训练开销，并限制了这些互补目标之间的表征共享。我们提出DexWAC，一个统一的世界-演员-评论家模型，它在单一的机器人预训练阶段中联合学习演员、动作条件世界模型和自举评论家。我们的关键洞察是：视觉预测鼓励与控制相关的特征，而价值监督促进与任务相关的表征；二者结合共同塑造出一种适合……（摘要至此截断）

    arXiv:2611.32654v1 Announce Type: cross  Abstract: Robot demonstrations provide more than actions to imitate: they reveal how actions change the world and how those changes contribute to task success. Recent world-action models (WAMs) extend vision-language-action (VLA) models with the first signal by jointly predicting actions and future observations, yet this predictive ability serves only as an auxiliary loss for policy learning. Meanwhile, most policy improvement methods learn dynamics and values in separate models, increasing training overhead and limiting representation sharing between these complementary objectives. We introduce DexWAC, a unified world-actor-critic model that jointly learns an actor, an action-conditioned world model, and a bootstrapped critic in a single robot pretraining stage. Our key insight is that visual prediction encourages control-relevant features, while value supervision promotes task-relevant representations; together, they shape a representation sui
    
[^60]: ICLWorld：从数字世界到物理世界的智能体上下文学习基准测试

    ICLWorld: Benchmarking In-Context Learning of Agents from Digital to Physical Worlds

    [https://arxiv.org/abs/2611.32649](https://arxiv.org/abs/2611.32649)

    ICLWorld是一个从数字到物理世界评估智能体上下文学习的在线基准，通过30个环境、160个反直觉规则任务覆盖计算机与机器人使用，并推出UIverse统一框架将每个环境转化为用户界面。

    

    上下文学习使智能体能够从上下文中获取新规则，并在推理时适应特定用户的需求。如今，同一批前沿智能体既能操作计算机也能操作机器人，但它们的上下文学习仍未得到充分评估：现有基准大多是离线的，依赖文本或静态图像，并存在训练数据污染的风险，而机器人方面的评估仅限于方法层面的消融实验。我们提出了ICLWorld，这是一个评估智能体从数字世界到物理世界的上下文学习能力的在线基准，涵盖计算机使用和机器人使用，并配有UIverse——一个将每个环境转化为用户界面的统一框架。我们构建了30个环境，录制了630个演示视频（超过10小时），并设计了160个带有定制的、常常违反直觉规则的任务，这些任务若不从上下文中学习则无法完成。每个任务都附带三个规则版本，以文本或视频形式提供，并且我们在三个（摘要此处截断）

    arXiv:2611.32649v1 Announce Type: cross  Abstract: In-context learning (ICL) lets agents acquire new rules from context and adapt to user-specific requirements at inference time. The same frontier agents now operate both computers and robots, yet their ICL remains insufficiently assessed: existing benchmarks are largely offline, rely on text or static images, and risk training-data contamination, while robot evaluations are limited to method-level ablations. We present ICLWorld, an online benchmark that evaluates the ICL of agents from digital to physical worlds, covering computer use and robot use, together with UIverse, a unified harness that turns every environment into a user interface. We build 30 environments, record 630 demonstration videos (10+ hours), and design 160 tasks with customized, often counter-intuitive rules that cannot be satisfied without learning from the context. Each task comes with three rule versions, given as text or video, and we test generalization at three
    
[^61]: 面向官方统计数据的Text-to-SQL基准测试

    A Text-to-SQL Benchmark for Official Statistics

    [https://arxiv.org/abs/2611.32646](https://arxiv.org/abs/2611.32646)

    本文提出了LOCuST——首个面向官方统计场景的text-to-SQL基准，将荷兰统计局两千余个多语言宽表与语义知识图谱相结合，以应对现有基准无法模拟编码式列名和外部元数据等官方统计特点的问题。

    

    官方统计是包括政策制定者和企业在内的各类主体的重要数据集。对于官方统计的提供者而言，提升其数据的可访问性、治理和披露水平至关重要，而基于text-to-SQL方法构建的问答系统可以改善这一点。然而，现有的text-to-SQL基准测试并不符合这一应用场景。官方统计以开放域目录的形式发布，包含数千个宽表，其列名和取值均为编码，其含义记录在独立的元数据中，而非关系模式内。为了填补这一空白，我们提出了LOCuST，这是一个text-to-SQL基准，它将来自荷兰统计局的超过两千个英语和荷兰语表格与语义知识图谱相配对。该知识图谱通过标签、定义、层级结构和计量单位来描述表格、度量指标和维度编码，是系统完成该任务所需的外部知识。

    arXiv:2611.32646v1 Announce Type: new  Abstract: Official statistics are critical datasets for a wide variety of actors, including policymakers and businesses. For providers of official statistics, improving accessibility, governance, and disclosure of their data is of key importance. This can be improved by providing question-answering systems based on text-to-SQL approaches. However, existing text-to-SQL benchmarks do not resemble this setting. Official statistics are published as open-domain catalogs of thousands of wide tables whose columns and values are codes, and whose meaning is recorded in separate metadata rather than in a relational schema. To address this gap, we introduce LOCuST, a text-to-SQL benchmark that pairs over two thousand English and Dutch tables from Statistics Netherlands with a semantic knowledge graph. The graph describes the tables, measures, and dimension codes through labels, definitions, hierarchies, and units, and is the external knowledge a system needs
    
[^62]: TeamFlowRL：面向多智能体协作的去中心化流策略在线策略优化

    TeamFlowRL: Online Policy Optimization of Decentralized Flow Policies for Multi-Agent Coordination

    [https://arxiv.org/abs/2611.32645](https://arxiv.org/abs/2611.32645)

    TeamFlowRL提出了一种两阶段框架，通过在去中心化流策略的完整去噪轨迹上构建团队策略优化目标（避免难以计算的动作似然），实现多智能体流策略的离线预训练与在线微调，在51个配对配置中平均提升47.3%。

    

    复杂的协作任务需要协调一致的多智能体行为。离线生成式策略能够从轨迹中学习富有表现力的协作行为，但受限于固定的数据集；在线强化学习则可以通过与环境交互来改进这些策略。然而，由于动作似然难以计算，流策略的微调十分困难。我们提出了TeamFlowRL，这是一个两阶段框架，先离线预训练多智能体流策略，再对其进行在线微调。我们将团队策略优化建模在去中心化流策略的完整去噪轨迹之上，从而避免了难以处理的动作似然计算。所得的目标函数使用共享的团队优势对每个智能体的轨迹比率进行加权，并分别对每个比率进行裁剪，使局部策略更新与共同的团队回报保持一致。在七个仿真组的51个配对配置中，TeamFlowRL改进了所有离线初始化策略，平均提升47.3%（摘要原文在此处截断）。

    arXiv:2611.32645v1 Announce Type: cross  Abstract: Complex cooperative tasks require coordinated multi-agent behavior. Offline generative policies learn expressive cooperative behaviors from trajectories but remain limited by fixed datasets. Online reinforcement learning can improve them through environment interaction. However, fine-tuning flow policies is difficult because action likelihoods are hard to compute. We propose TeamFlowRL, a two-stage framework that pretrains multi-agent flow policies offline and fine-tunes them online. We formulate team policy optimization over the complete denoising trajectories of decentralized flow policies, avoiding intractable action likelihoods. The resulting objective weights each agent's trajectory ratio with a shared team advantage and clips each ratio separately, aligning local policy updates with the common team return. Across 51 paired configurations in seven simulation groups, TeamFlowRL improves every offline initialization, with a 47.3% av
    
[^63]: BARE：面向人形机器人的身体感知反射式闪避

    BARE: Body-Aware Reflexive Evasion for Humanoid Robots

    [https://arxiv.org/abs/2611.32642](https://arxiv.org/abs/2611.32642)

    提出了BARE框架，利用人类运动先验引导人形机器人通过局部身体调整实现快速闪避，同时保持全身稳定性。

    

    动态避障对于人形机器人在真实环境中的安全部署至关重要，在这些环境中，快速接近的物体需要立即做出闪避响应。近期反射式闪避的研究进展主要集中在四足机器人上，这类机器人通常依赖于大规模的全身动作。相比之下，人形机器人可以利用其丰富的身体关节自由度，通过小幅度的局部身体调整来实现闪避。通过在不重新调整整个身体位置的情况下移动受威胁的身体部位，这类响应可能需要更少的执行时间和更少的周围自由空间。然而，局部调整可能会改变质量分布或支撑条件，这使得在保持全身稳定性的同时实现快速闪避变得具有挑战性。为此，我们提出了BARE（身体感知反射式闪避），这是一个由人类运动先验引导的人形机器人反射式闪避框架，能够在保持全身稳定性的同时实现快速的局部闪避。

    arXiv:2611.32642v1 Announce Type: cross  Abstract: Dynamic obstacle avoidance is essential for safely deploying humanoid robots in real-world environments, where rapidly approaching objects require immediate evasive responses. Recent advances in reflexive evasion have focused primarily on quadrupeds, which typically rely on large-scale whole-body maneuvers. In contrast, humanoids can exploit their rich body articulation to evade through small, localized body adjustments. By moving threatened body regions without repositioning the entire body, these responses may require less execution time and less surrounding free space. However, local adjustments can alter mass distribution or support conditions, making it challenging to achieve rapid evasion while maintaining whole-body stability. To this end, we propose BARE (Body-Aware Reflexive Evasion), a humanoid reflexive evasion framework guided by human motion priors that enables rapid local evasion while maintaining whole-body stability. A 
    
[^64]: ReFocus：面向长时程机器人操作强化学习的进度引导自适应练习

    ReFocus: Progress-Guided Adaptive Practice for Long-Horizon Robotic Manipulation Reinforcement Learning

    [https://arxiv.org/abs/2611.32639](https://arxiv.org/abs/2611.32639)

    ReFocus提出了一种进度引导的多轮强化学习框架，通过将任务分解为有向无环图（DAG）并利用进度奖励诊断策略瓶颈，进而通过自适应练习窗口为瓶颈阶段分配针对性练习，以提升长时程机器人操作任务的RL性能。

    

    在线强化学习（RL）使视觉-语言-动作策略能够通过与环境的交互来不断改进。然而，在长时程操作任务中，稀疏的成功/失败反馈几乎无法提供策略在何处遇到瓶颈的信息。此外，反复进行完整任务的回合交互也使这些瓶颈得不到充分分析和针对性的练习。为此，我们提出了ReFocus，这是一种进度引导的多轮强化学习框架，它在全局RL与局部RL之间交替进行，以识别特定任务的瓶颈并为这些瓶颈分配额外的练习。具体而言，在全局RL阶段，我们将每个任务分解为子任务组成的有向无环图（DAG），然后利用基于DAG的进度奖励同时进行策略训练和瓶颈诊断。基于识别出的瓶颈，ReFocus选择自适应练习窗口，定义从入口阶段到目标阶段的任务片段，并为其分配额外的练习……

    arXiv:2611.32639v1 Announce Type: cross  Abstract: Online reinforcement learning (RL) enables vision-language-action policies to improve through environmental interaction. However, in long-horizon manipulation, sparse success/failure feedback provides little information about where policies encounter bottlenecks. Moreover, repeated full-task rollouts also leave these bottlenecks weakly analyzed and insufficiently practiced. To this end, we introduce ReFocus, a progress-guided multi-round RL framework that alternates between Global and Local RL to identify task-specific bottlenecks and allocate additional practice to them. Specifically, during Global RL, we decompose each task into a Directed Acyclic Graph (DAG) of subtasks. Then a DAG-based Progress Reward is leveraged for both policy training and bottleneck diagnosis. Based on the identified bottlenecks, ReFocus selects Adaptive Practice Windows, defining task segments from an entry stage to a target stage. It allocates additional int
    
[^65]: MargEmbed-Akan：通过跨语言分数蒸馏连接契维语问题与英语知识

    MargEmbed-Akan: Bridging Twi Questions and English Knowledge through Cross-Lingual Score Distillation

    [https://arxiv.org/abs/2611.32630](https://arxiv.org/abs/2611.32630)

    该论文提出了 MargEmbed-Akan，一个基于跨语言分数蒸馏从 AfriE5-Large-instruct 训练得到的 5.6 亿参数开源查询编码器，有效连接契维语问题与英语知识检索。

    

    契维语使用者在网上搜索的大多数信息都是以英语书写的，这使得跨语言检索变得至关重要。我们在两个契维语-英语任务上评估了八个系统：BM25、五个开源嵌入模型以及两个专有模型（gemini-embedding-001 和 OpenAI text-embedding-3-large）。AfriQA 数据集包含 1,221 个原生编写的阿散蒂契维语问题，这些问题需要检索出能够回答它们的英语维基百科段落。圣经翻译检索任务则要求一段契维语经文从 6,195 个候选中检索出对应的英语译文。我们提出了 MargEmbed-Akan，这是一个拥有 5.6 亿参数的开源查询编码器，基于 AfriE5-Large-instruct 通过跨语言分数蒸馏训练而成。冻结的基础模型对每个英语训练问题与其答案段落及七个难负例进行评分。学生模型学习从通过往返翻译校验的 NLLB-200 契维语译文以及原始英语问题中复现这些分数，从而保持……

    arXiv:2611.32630v1 Announce Type: cross  Abstract: Most information sought online by Twi (Akan) speakers is written in English, making cross-lingual retrieval essential. We evaluate eight systems: BM25, five open embedding models, and two proprietary models (gemini-embedding-001 and OpenAI text-embedding-3-large) on two Twi-English tasks. AfriQA contains 1,221 natively written Asante Twi questions that must retrieve their answering English Wikipedia passages. Bible translation retrieval requires a Twi passage to retrieve its English translation from 6,195 candidates.   We introduce MargEmbed-Akan, a 560M-parameter open query encoder trained from AfriE5-Large-instruct using cross-lingual score distillation. The frozen base model scores each English training question against its answer passage and seven hard negatives. The student learns to reproduce these scores from both an NLLB-200 Twi translation that passes a round-trip translation check and the original English question, preserving
    
[^66]: REBEAL：人工参考译文已不再是最佳翻译，而评价指标却浑然不觉

    REBEAL: Human References Are No Longer the Best Translation, and Metrics Do Not Notice

    [https://arxiv.org/abs/2611.32623](https://arxiv.org/abs/2611.32623)

    该论文提出REBEAL基准，通过对参考译文施加七类受控污染来测试评价指标，发现人工参考译文质量已逊于顶级机器翻译系统，而现有八种主流指标均无法察觉这种污染，且各类型指标以不同方式失效。

    

    基于参考译文的机器翻译评价假设人工参考译文是正确的，且优于其所评分的系统。如今这两个假设均已失效，因为在2024年WMT MQM评测的所有活动中，人工参考译文均未排名第一，且在其19%至87%的句子片段中带有已标注的错误。面对有缺陷的参考译文，评价指标会如何表现，这是一个关于指标本身的问题，早在参考译文质量受到质疑之前，这个问题就值得被提出。我们提出了REBEAL（Reference Bias EvALuation，参考译文偏差评测），这是一个探究机器翻译评价指标如何应对七类受控参考译文污染的基准测试，污染类型源自上述评测活动所记录的错误类别。在八种广泛使用的评价指标和三个语言对上的实验表明，近期基于学习的指标（xCOMET、MetricX）比其前代更为稳健，但每一类指标都以各自的方式失效：表层指标缺乏长度敏感性，因此污染会隐藏在较长的句子片段中，而基于学习的指标则会接受与原文无关的……（原文在此处截断）

    arXiv:2611.32623v1 Announce Type: cross  Abstract: Reference-based MT evaluation assumes the human reference is correct and better than the systems it scores. Both assumptions now fail, since the human reference ranks first in none of the 2024 WMT MQM campaigns and carries annotated errors in 19% to 87% of its segments. What a metric does with a flawed reference is a question about the metric, and it would have been worth asking before reference quality came into doubt. We introduce REBEAL (Reference Bias EvALuation), a benchmark that probes how MT metrics respond to seven types of controlled reference contamination, drawn from the error categories those campaigns record. Across eight widely used metrics and three language pairs, recent learned metrics (xCOMET, MetricX) prove more robust than their predecessors, but each family fails in its own way. Surface metrics show no length sensitivity, so contamination hides in longer segments, while learned metrics accept references in unrelate
    
[^67]: RoboRecast：基于诊断驱动优化的意图保持演示增强

    RoboRecast: Intent-Preserving Demonstration Augmentation via Diagnosis-Driven Optimization

    [https://arxiv.org/abs/2611.32622](https://arxiv.org/abs/2611.32622)

    RoboRecast是一个诊断驱动的智能体优化框架，通过联合优化仿真场景与机器人运动来保持源任务交互意图，可从人类和机器人演示生成适配新布局、新物体和新机器人形态的合成训练数据。

    

    收集多样化的真实世界操作演示成本高昂，这促使研究者利用仿真来增强现有演示以用于机器人策略学习。然而，当物体、布局或机器人发生变化时，从源演示迁移动作可能产生在新环境中语义上不合理或物理上不可行的行为。我们提出了RoboRecast，这是一个智能体优化框架，它在保持源任务所需交互的同时，联合优化仿真场景和机器人运动，使这些交互在目标环境中可执行。当某次交互失败时，智能体利用仿真反馈来诊断原因，修改相关的场景或运动参数，并测试该修改是否能解决失败。这种灵活性使RoboRecast能够从人类和机器人演示中生合成数据，并将其适配到新的布局、物体和机器人形态。

    arXiv:2611.32622v1 Announce Type: cross  Abstract: Collecting diverse real-world manipulation demonstrations is costly, motivating the use of simulation to augment existing demonstrations for robot policy learning. When objects, layouts, or robots change, transferring motions from a source demonstration can produce behavior that is semantically implausible or physically infeasible in the new setting. We introduce RoboRecast, an agentic optimization framework that preserves the interactions required by the source task while jointly optimizing the simulated scene and robot motion to make those interactions executable in the target setting. When an interaction fails, the agent uses simulation feedback to diagnose the cause, revise the relevant scene or motion parameters, and test whether the revision resolves the failure. This flexibility allows RoboRecast to generate synthetic data from both human and robot demonstrations and to adapt them to new layouts, objects, and robot embodiments. 
    
[^68]: 门控概念适配器：用于可审计文本理解的分层概念瓶颈

    Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding

    [https://arxiv.org/abs/2611.32600](https://arxiv.org/abs/2611.32600)

    提出了门控概念适配器（GCA）这一分层概念瓶颈框架，通过在预训练Transformer的选定层附加门控低维概念读出适配器，将隐藏状态转化为可检查、可干预的概念中介接口，从而实现可审计的文本理解。

    

    面向人类的文本理解系统被用于推断情感、审核有害内容以及支持社会敏感的决策。然而，Transformer的预测难以审计，因为其预测产生于高维隐藏状态，无法为人类分析或干预提供紧凑、可检查的接口。我们提出了门控概念适配器（Gated Concept Adapters, GCA），这是一种分层概念瓶颈框架，它将带门控的、低维的概念生成读出适配器附加到预训练Transformer的选定输出上。词元表示被映射为概念激活，经学习到的门控过滤，汇聚为层级别的摘要，并作为概念中介接口暴露给预测头。这些维度可以通过使用统计、特征性n-gram、事后概念命名以及针对性的概念干预来进行检查。我们在细粒度情感分类、毒性检测等任务上对GCA进行了评估……

    arXiv:2611.32600v1 Announce Type: cross  Abstract: Human-facing text understanding systems are used to infer affect, moderate harmful content, and support socially sensitive decisions. However, Transformer predictions are difficult to audit because they are produced from high-dimensional hidden states that do not provide a compact, inspectable interface for human analysis or intervention. We introduce Gated Concept Adapters (GCA), a layer-wise concept bottleneck framework that attaches gated, low-dimensional concept-producing readout adapters to selected outputs of pretrained Transformers. Token representations are mapped into concept activations, filtered by learned gates, pooled into layer-level summaries, and exposed to the prediction head as a concept-mediated interface. The dimensions can be inspected through usage statistics, signature n-grams, post-hoc concept names, and targeted concept interventions. We evaluate GCA on fine-grained emotion classification, toxicity detection, t
    
[^69]: 蒸炼不确定性：面向两阶段随机优化的决策引导进化策略深度生成式情景搜索

    Boiling Down Uncertainty: Deep Generative Scenario Search with Decision-Guided Evolution Strategies for Two-Stage Stochastic Optimization

    [https://arxiv.org/abs/2611.32573](https://arxiv.org/abs/2611.32573)

    本文通过补偿对偶性与伴随求解推导出闭式梯度，为深度生成式情景搜索补齐了解析理论基础，并将决策聚焦方向作为引导进化策略的采样子空间，保证期望更新为平滑目标的下降方向，从而改进两阶段随机优化的情景生成。

    

    两阶段随机规划是处理不确定性最常用的模型之一。选择合适的情景数量需要在可表征的不确定性与计算成本之间进行权衡。近期提出的一种方法——深度生成式情景搜索，通过生成合成情景，并从这些情景所诱导的决策中学习来改进情景质量。由于决策在情景之间是分段常数的，原始方法不得不使用黑盒梯度估计。本文通过借助补偿对偶性和正则化简化问题的伴随求解，推导出闭式形式的梯度，补齐了该方法缺失的解析理论基础。决策聚焦方向（当对偶变量不可用时以启发式方向作为后备）被用作引导进化策略的采样子空间，而非直接作为更新方向。对于任意引导子空间，期望更新都是平滑目标的下降方向，因此……

    arXiv:2611.32573v1 Announce Type: cross  Abstract: One of the most common models for handling uncertainty is two-stage stochastic programming. Choosing the right number of scenarios is a trade-off between the uncertainty that can be represented and computational cost. A recent method, Deep Generative Scenario Search, generates synthetic scenarios and improves them by learning from the decisions they induce. The original approach used a black-box gradient estimate, since decisions are piecewise constant across scenarios. This paper provides the missing analytical foundation by deriving the gradient in closed form from recourse duality and an adjoint solve of the regularized reduced problem. The decision-focused direction, with a heuristic direction as a backup when duals are unavailable, defines the sampling subspace of a guided evolution strategy rather than being used as a direct update. For any guiding subspace, the expected update is a descent direction of the smoothed objective, so
    
[^70]: MXI-MemICL：通过自检索交互记忆迈向上下文机器人操作

    MXI-MemICL: Towards In-Context Robot Manipulation through Self-Retrieved Interaction Memory

    [https://arxiv.org/abs/2611.32564](https://arxiv.org/abs/2611.32564)

    MXI-MemICL通过将演示和历史交互转化为可自检索的交互记忆，并借助意图预测器检索相关经验，使视觉-语言-动作模型无需大规模预训练或额外标注即可实现上下文机器人操作。

    

    上下文学习（ICL）为机器人提供了一种从单个演示中习得新技能而无需更新参数的途径。然而，现有的机器人ICL方法要么依赖于大规模预训练，而这种方式受制于稀缺且成本高昂的机器人交互数据；要么需要额外的标注，却难以跨任务泛化。为了弥合这一差距，我们提出了MXI-MemICL，一个面向视觉-语言-动作（VLA）模型的记忆引导ICL框架。通过将演示视为可检索的记忆，MXI-MemICL无需额外的大规模预训练或专门的标注即可实现上下文操作。具体而言，一个视觉-动作因果编码器将演示以及已完成的交互中的动作及其观测效果转化为可复用的记忆，用以指导执行。为了检索适合即将执行任务的经验，一个意图预测器预测下一阶段的意图，并利用它来查询记忆。检索到的经验……

    arXiv:2611.32564v1 Announce Type: cross  Abstract: In-context learning (ICL) offers robots a way to acquire new skills from a single demonstration without parameter updates. However, existing robotic ICL methods either rely on large-scale pretraining constrained by scarce, costly robot interaction data or require additional annotations yet struggle to generalize across tasks. To bridge this gap, we propose MXI-MemICL, a framework for memory-guided ICL in vision-language-action (VLA) models. By treating demonstrations as retrievable memory, MXI-MemICL enables in-context manipulation without additional large-scale pretraining or specialized annotations. Specifically, a visual-action causal encoder turns actions and their observed effects in demonstrations and completed interactions into reusable memory to guide execution. To retrieve experience suited to upcoming execution, an intention predictor forecasts the next stage's intent and uses it to query memory. The retrieved experience cond
    
[^71]: Next-Manip：面向机器人操作的按需智能体-策略协作

    Next-Manip: On-Demand Agent-Policy Collaboration for Robotic Manipulation

    [https://arxiv.org/abs/2611.32533](https://arxiv.org/abs/2611.32533)

    Next-Manip 提出了一个按需的智能体-策略协作框架，通过轻量级的智能体调用控制器根据当前执行需求在策略与智能体之间动态分配控制权，将执行器分配与高成本的智能体推理解耦，从而兼顾任务推理、失败恢复与高效精确的细粒度控制。

    

    通用机器人操作需要任务推理、失败恢复和精确控制，然而有限的操作数据使得这些能力难以从单一策略中获得。近期的多模态智能体在任务推理和失败恢复方面展现出潜力，但缺乏可靠的细粒度控制且推理成本高昂；而学习到的策略能够高效地进行细粒度执行，但在其已学技能之外则表现不佳。尽管两者优势互补、看似适合结合，但这需要随着任务需求的变化反复决定应由哪个执行器行动。不同于在每次决策时都依赖智能体，我们提出了 Next-Manip，这是一个按需的智能体-策略协作框架，将执行器的分配与代价高昂的智能体推理解耦。我们提出了智能体调用控制器，它通过一个轻量级的当前执行需求分类器，在策略与智能体之间分配控制权。

    arXiv:2611.32533v1 Announce Type: cross  Abstract: General-purpose robotic manipulation requires task reasoning, failure recovery, and precise control, yet limited manipulation data make these abilities difficult to obtain from a single policy. Recent multimodal agents show promise in task reasoning and recovery but lack reliable fine-grained control and incur high inference cost, whereas learned policies provide efficient fine-grained execution but struggle beyond their learned skills. Although their complementary strengths suggest combining the two, doing so requires repeatedly deciding which executor should act as task demands change. Instead of relying on the agent for each decision, we present Next-Manip, an on-demand agent--policy collaboration framework that decouples executor allocation from costly agent inference. We propose an Agent Invocation Controller (AIC) that allocates control between the policy and the agent through a lightweight classifier of current execution demands
    
[^72]: ContinualDrive：一个面向学习型驾驶规划器持续更新的、以部署为锚点的多轴基准测试

    ContinualDrive: A Deployment-Anchored, Multi-Axis Benchmark for Continual Updates of Learned Driving Planners

    [https://arxiv.org/abs/2611.32529](https://arxiv.org/abs/2611.32529)

    该论文提出ContinualDrive基准，以已部署模型为锚点评估驾驶规划器的部署后持续学习，并揭示评估回路、参考检查点、时间聚合和流协议这四个测量轴的选择会显著改变甚至逆转更新效果的判定结论。

    

    学习型驾驶规划器在部署后会随着新驾驶日志的到来而不断更新，然而持续学习评估很少追问一次更新是否真正优于已部署的模型，且其结论往往依赖于未明示的评估选择。我们提出ContinualDrive，一个面向驾驶规划器部署后持续学习的、以部署为锚点的基准测试。该基准基于真实世界的nuPlan日志，每次针对一类场景类型组对已部署规划器进行更新，在开环与闭环两种模式下评估相同的模型检查点，明确给出四个测量轴（评估回路、参考检查点、时间聚合与数据流协议），并将三值化的发布判定锚定于已部署的检查点。我们在三种任务顺序下，研究了两个规划器在完整微调以及适配器或LoRA更新下的表现，并对比了有无回放或梯度投影的情况。研究发现，当任一测量轴改变时，判定结果可能逆转甚至消失：开环ADE对最终检查点

    arXiv:2611.32529v1 Announce Type: cross  Abstract: Learned driving planners keep being updated after deployment as new logs arrive, yet continual learning evaluations rarely ask whether an update improves on the deployed model, and their verdicts rest on unstated evaluation choices. We present ContinualDrive, a deployment-anchored benchmark for post-deployment continual learning of driving planners. It updates a deployed planner on real-world nuPlan logs one scenario-type group at a time, evaluates the same checkpoints in open and closed loop, makes four measurement axes explicit (evaluation loop, reference checkpoint, temporal aggregation and stream protocol), and anchors a three-valued release verdict to the deployed checkpoint. We study two planners (PlanTF and Diffusion Planner) under full fine-tuning and adapter or LoRA updates, with and without replay or gradient projection, in three task orders. Verdicts can reverse or vanish when one axis changes: open-loop ADE ranks the final 
    
[^73]: 基于Wi-Fi的边缘智能框架：使用低功耗物联网摄像头进行智慧城市交通监控

    Wi-Fi Enabled Edge Intelligence Framework for Smart City Traffic Monitoring using Low-Power IoT Cameras

    [https://arxiv.org/abs/2611.32517](https://arxiv.org/abs/2611.32517)

    本文提出了一种基于Wi-Fi的边缘智能框架，利用低功耗物联网摄像头在智慧城市交通监控中实现本地推理或卸载至Jetson Nano进行GPU加速，无需依赖5G或有线网络即可达到近实时性能。

    

    智慧城市中的实时交通监控需要超低延迟处理，以支持事件检测和拥堵管理等时间敏感型决策。虽然基于云的解决方案提供了强大的计算能力，但其固有的延迟限制了其在此类任务中的适用性。本工作提出了一个本地化边缘AI框架，将低功耗物联网摄像头传感器连接到客户端，或将推理任务卸载至NVIDIA Jetson Nano（GPU）。网络连接通过Wi-Fi实现，无需依赖5G或有线网络等广域基础设施即可进行图像分类。我们评估了两种处理策略：摄像头节点上的本地推理和GPU加速卸载至Jetson Nano。结果表明，本地处理仅适用于轻量级模型和低帧率场景，而卸载方案即使对于更复杂的模型也能实现近实时性能。这些结果证明了成本效益（原文摘要在此处截断）

    arXiv:2611.32517v1 Announce Type: cross  Abstract: Real-time traffic monitoring in smart cities demands ultra-low latency processing to support time-critical decisions such as incident detection and congestion management. While cloud-based solutions offer robust computation, their inherent latency limits their applicability for such tasks. This work proposes a localized edge AI framework that connects low-power IoT camera sensors to a client, or applies offloading of inference to an NVIDIA Jetson Nano (GPU). Networking is achieved via Wi-Fi, enabling image classification without relying on wide-area infrastructure such as 5G, or wired networks. We evaluate two processing strategies: local inference on camera nodes and GPU-accelerated offloading to the Jetson Nano.We show that local processing is only feasible for lightweight models and low frame rates, whereas offloading enables near-realtime performance even for more complex models. These results demonstrate the viability of cost-effe
    
[^74]: 分支与整合：多教师在线策略蒸馏的统一配方

    Branch and Consolidate: A Unified Recipe for Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2611.32515](https://arxiv.org/abs/2611.32515)

    该论文提出了一套统一的多模态多教师在线策略蒸馏配方，能够将大幅偏离共享检查点的专家模型重新整合回单一学生模型，使其在各个领域接近专家模型性能并超越感知专家。

    

    从共享检查点后训练得到的专家模型会习得互补的能力，而多教师在线策略蒸馏（MOPD）可以将这些能力整合到单一模型中。然而，我们发现现有的整合方法（从权重合并到MOPD）在专家模型与共享检查点产生显著偏离时效果不佳，而以往研究主要考察的是与检查点保持接近的专家模型。我们提出了一套用于多模态MOPD的统一配方，可将这些专家模型重新整合回它们最初分支出的那个单一学生模型中。该配方明确了整合所需的初始化、数据选择、跨教师的均衡监督以及现有能力的保持，并辅以针对性的精炼阶段，用于提升仍然落后于对应专家模型的能力。在一个包含六个专家模型的工业场景中，学生模型在每个专家模型自身的领域内均能接近其表现，并在感知领域超越了感知专家模型。

    arXiv:2611.32515v1 Announce Type: cross  Abstract: Specialists post-trained from a shared checkpoint acquire complementary capabilities, and multi-teacher on-policy distillation (MOPD) can consolidate them into a single model. However, we find that existing consolidation methods, from weight merging to MOPD, fall short when specialists diverge substantially from the shared checkpoint, whereas prior studies mostly examine specialists that remain close to it. We present a unified recipe for multimodal MOPD that consolidates such specialists back into the single student they branched from. The recipe specifies initialization, data selection, balanced supervision across teachers, and preservation of existing capabilities for consolidation, together with a targeted refinement stage for capabilities that still lag behind their specialists. In an industrial setting with six specialists, the student closely approaches each specialist in its own domain and surpasses the perception specialist. I
    
[^75]: RecurSciBench：数据驱动科学研究中递归自我改进能力的基准测试

    RecurSciBench: Benchmarking Recursive Self-Improvement in Data-Driven Scientific Research

    [https://arxiv.org/abs/2611.32508](https://arxiv.org/abs/2611.32508)

    该论文提出了RecurSciBench——一个用于评估大语言模型智能体在数据驱动科学研究中递归自我改进能力的可扩展基准，其RecurSciPipe框架可从原始数据集自动构建带科学验证的任务，并在四个科学领域上系统评估了四个RSI系统的自我改进迁移效果。

    

    基于大语言模型（LLM）的智能体正在推动数据驱动科学研究的自动化。递归自我改进（RSI）的最新进展为这些智能体通过经验实现持续改进提供了潜力，但评估改进能力在科学任务间的迁移，需要可执行的、科学上可验证的任务以及对自我改进收益的独立评估。我们提出了RecurSciBench，一个用于评估数据驱动科学研究中递归自我改进的可扩展基准。其RecurSciPipe框架能够从原始数据集中自动构建科学建模和数据分析任务，并提供基于执行的验证和科学有效性检查。RecurSciBench涵盖四个科学领域、多种数据模态和多样化的科学目标。我们通过比较四个RSI系统在源任务上演进前后于独立任务上的表现来评估它们，覆盖了同类型内、跨类型以及跨领域的迁移。

    arXiv:2611.32508v1 Announce Type: new  Abstract: Large language model (LLM)-based agents are advancing the automation of data-driven scientific research. Recent advances in recursive self-improvement (RSI) offer the potential for these agents to continuously improve through experience, but evaluating transfer across scientific tasks requires executable, scientifically verifiable tasks and independent assessment of self-improvement gains. We introduce RecurSciBench, a scalable benchmark for evaluating RSI in data-driven scientific research. Its RecurSciPipe framework automatically constructs scientific modeling and data analysis tasks from raw datasets, with execution-based verification and scientific validity checks. RecurSciBench spans four scientific domains, multiple data modalities, and varied scientific objectives. We evaluate four RSI systems by comparing their initial and evolved states on independent tasks after evolution on source tasks, covering within-type, cross-type, and c
    
[^76]: 评估信息物理系统中持续深度强化学习对灾难性遗忘的鲁棒性

    Assessing Robustness to Catastrophic Forgetting in Continual Deep Reinforcement Learning for Cyber-Physical Systems

    [https://arxiv.org/abs/2611.32496](https://arxiv.org/abs/2611.32496)

    该论文在非平稳自主泊车环境中系统评估了三类强化学习算法（无模型同策略的PPO、无模型异策略的SAC和基于模型的Drama）抵抗灾难性遗忘的鲁棒性，以确定哪类算法更适合信息物理系统中的持续学习。

    

    随着强化学习（RL）在信息物理系统中的应用日益广泛，设计能够适应环境变化以保持安全性和可靠性的RL智能体变得越来越重要。本文研究了在非平稳环境中，不同类别的RL算法是否对遗忘先前习得行为（持续学习中的关键问题）具有更强的鲁棒性。我们在一个非平稳的自主泊车环境中评估了三类重要RL算法中代表性方法的鲁棒性，分别是近端策略优化（PPO）——一种无模型同策略算法、软演员-评论家（SAC）——一种无模型异策略算法，以及Drama——一种基于模型的算法。智能体必须学习在四种不同场景中成功泊车，这些场景依次呈现，泊车位朝向……

    arXiv:2611.32496v1 Announce Type: cross  Abstract: As the use of reinforcement learning (RL) becomes more widespread in cyber-physical systems, it becomes increasingly important to design RL agents that are able to adapt to changes in their environments in order to maintain safety and reliability. This paper addresses the question of whether different classes of RL algorithm are more robust against forgetting previously learned behaviours, a critical issue in continual learning, in non-stationary environments. We evaluate the robustness of representative examples of three important classes of RL algorithm in a non-stationary autonomous parking environment. These are Proximal Policy Optimization (PPO), which is a model-free on-policy algorithm, Soft-Actor-Critic (SAC), which is a model-free off-policy algorithm, and Drama, which is a model-based algorithm. The agents must learn to successfully park in four different scenarios, which are presented successively, with parking spaces orient
    
[^77]: CodeWAM：一种用于端到端自动驾驶的代码世界动作模型

    CodeWAM: A Code World Action Model for End-to-End Autonomous Driving

    [https://arxiv.org/abs/2611.32483](https://arxiv.org/abs/2611.32483)

    该论文提出CodeWAM，一种将可执行代码纳入世界表征的端到端自动驾驶世界模型，用代码刻画自车在规划执行下的动力学演化、用学习的视觉模型捕捉场景依赖后果，使规划器能够同时评估候选规划的未来结果与车辆向该未来的过渡过程。

    

    驾驶世界模型通过对世界演化方式的表征来预测动作后果。对于自动驾驶而言，这些表征必须同时捕捉周围场景以及自车在候选规划下的运动。我们提出CodeWAM，一种将可执行代码纳入其世界表征的驾驶世界模型。代码通过表达车辆状态在规划执行下如何演化来刻画自车动力学，而学习得到的视觉模型则捕捉依赖于场景的后果。执行代码会生成一个以候选规划为条件的运动序列，该序列相对于先前的规划和已记录的运动历史进行评估。由此得到的执行后果与学习到的场景结果共同表征候选规划的未来，使规划器既能评估某个规划可能导向什么结果，也能评估车辆如何过渡到该未来。所选规划将成为下一次决策执行上下文的一部分。

    arXiv:2611.32483v1 Announce Type: cross  Abstract: Driving world models anticipate action consequences through representations of how the world evolves. For autonomous driving, these representations must capture both the surrounding scene and the ego vehicle's motion under a candidate plan. We propose CodeWAM, a driving world model that incorporates executable code as part of its world representation. Code represents ego dynamics by expressing how the vehicle state evolves under plan execution, while a learned visual model captures scene-dependent consequences. Executing the code produces a candidate-conditioned motion sequence, which is evaluated relative to the prior plan and recorded motion history. The resulting execution consequences and learned scene outcomes jointly represent the candidate's future, enabling the planner to assess both what a plan may lead to and how the vehicle transitions into that future. The selected plan becomes part of the next decision's execution context,
    
[^78]: VISE：基于工具实现反馈的、无需验证器的技能自进化方法

    VISE: Verifier-Independent Skill Self-Evolution with Feedback from Tool Implementations

    [https://arxiv.org/abs/2611.32480](https://arxiv.org/abs/2611.32480)

    VISE提出了一种无需任务特定验证器的技能自进化方法，通过从工具源代码中提取能力、输入要求、使用条件和输出格式等信息来指导技能工作流的修订，在两个基准上将平均任务成功率提升了9.14-13.35个百分点。

    

    智能体技能为LLM智能体使用工具提供了可复用的工作流程。改进这些工作流程需要关于缺失或错误指令的反馈，但针对特定任务的验证器并不总是可用。我们提出了VISE（无验证器技能进化），这是一种“脚本优先”的方法，利用工具实现来指导技能修订。它从源代码中提取工具的能力、输入要求、使用条件和输出格式。利用任务请求和执行轨迹，它将这些信息与当前工作流程指令进行比较，以识别缺失或冲突的步骤。然后，它为SKILL.md推荐具体的编辑，并将每项更改与相关的实现和使用条件关联起来。我们在SkillsBench和SkillLearnBench上评估了VISE作为四种进化方法的附加修订阶段的效果。与相应的无验证器条件相比，它将整体平均任务成功率提高了9.14-13.35个百分点。

    arXiv:2611.32480v1 Announce Type: new  Abstract: Agent skills provide reusable workflows that guide LLM agents in using tools. Refining these workflows requires feedback on missing or incorrect instructions, but task-specific verifiers are not always available. We propose VISE (Verifier-Independent Skill Evolution), a scripts-first approach that uses tool implementations to guide skill revision. It extracts tool capabilities, input requirements, usage conditions, and output formats from source code. Using the task request and execution trace, it compares this information with current workflow instructions to identify missing or conflicting steps. It then recommends specific edits to SKILL.md, linking each change to the relevant implementation and usage conditions. We evaluate VISE as an additional revision stage for four evolution methods on SkillsBench and SkillLearnBench. Compared with the corresponding No-verifier conditions, it improves overall average task success by 9.14-13.35 pe
    
[^79]: 面向黑盒普适域自适应的渐进式可靠监督

    Progressive Reliable Supervision for Black-Box Universal Domain Adaptation

    [https://arxiv.org/abs/2611.32474](https://arxiv.org/abs/2611.32474)

    本文提出渐进式可靠监督方法PReS，通过协调黑盒源模型、基础模型（如CLIP）和不断演进的目标模型的互补知识，在无源数据和源模型参数的黑盒普适域自适应场景下，逐步建立并完善可靠的目标监督。

    

    黑盒普适域自适应（B²-UniDA）旨在将目标模型适配到无标签的目标域，并应对任意的标签空间关系，同时在适配过程中既无法访问源数据，也无法访问源模型参数。在这种受限的信息访问条件下，可靠的目标监督难以获得，而由于源偏差和域偏移的存在，黑盒源模型的预测可能并不可靠。为应对这一挑战，我们提出了渐进式可靠监督（PReS），通过协调来自黑盒源模型、基础模型以及不断演进的目标模型的互补知识，逐步建立并完善可靠的目标监督。PReS首先利用基于扩散的扰动下的语义一致性，以及黑盒源模型与CLIP之间的跨模型共识来识别可靠的已知样本，然后估计样本级的可靠性……（摘要在此处被截断）

    arXiv:2611.32474v1 Announce Type: cross  Abstract: Black-Box Universal Domain Adaptation (B$^2$-UniDA) aims to adapt a target model to an unlabeled target domain under arbitrary label-space relationships, while neither source data nor source-model parameters are accessible during adaptation. Under such restrictive information access, reliable target supervision is unavailable, while predictions from the black-box source model can be unreliable due to source bias and domain shifts. To address this challenge, we propose Progressive Reliable Supervision (PReS), which progressively establishes and refines reliable target supervision by coordinating complementary knowledge from the black-box source model, foundation models, and the evolving target model. PReS first identifies reliable known samples by exploiting semantic consistency under diffusion-based perturbations and cross-model consensus between the black-box source model and CLIP. It then estimates the sample-wise reliability of the 
    
[^80]: AI时代的游戏化：调查玩家对动机、操纵和数据保护的认知

    Gamification in the age of AI: surveying player perceptions of motivation, manipulation, and data protection

    [https://arxiv.org/abs/2611.32464](https://arxiv.org/abs/2611.32464)

    通过对28名成年玩家的问卷调查发现，玩家认为等级系统和排行榜等进度机制具有激励性，而每日登录奖励和随机奖励等留存机制则被视为高度操纵性，且玩家普遍担忧个人数据被收集利用但缺乏意识与控制感。

    

    现代游戏和应用程序越来越多地采用人工智能（AI）、用户建模和游戏化技术来影响用户行为、维持用户参与度并将注意力变现。尽管这些机制能够增强动机和乐趣，但它们也引发了对操纵、个人数据使用和透明度的担忧。为了考察玩家如何看待这些动态，我们对28名成年参与者进行了问卷调查。受访者对游戏和游戏化应用中个人信息的收集与利用表示了强烈担忧，同时报告了对数据如何被使用意识较低且控制感有限。他们将等级系统和排行榜等进度功能评价为具有激励性。相比之下，他们将每日登录奖励和随机奖励揭示等以留存为导向的机制评价为高度操纵性的。参与者报告说，这些系统可能会制造错失恐惧。

    arXiv:2611.32464v1 Announce Type: cross  Abstract: Modern games and applications increasingly employ artificial intelligence (AI), user modeling, and gamification to influence user behavior, sustain engagement, and monetize attention. Although these mechanisms can enhance motivation and enjoyment, they also raise concerns about manipulation, personal data usage, and transparency. To examine how players perceive these dynamics, we conducted a survey with 28 adult participants. Respondents expressed strong concerns about the collection and exploitation of personal information in games and gamified applications, while also reporting low awareness and a limited sense of control over how their data is used. They evaluated progression features such as leveling systems and leaderboards as motivating. In contrast, they rated retention-focused mechanics such as daily login rewards and random reward reveals as highly manipulative. Participants reported that these systems can create fear of missi
    
[^81]: 描述而非提问：面向细粒度视觉感知的描述驱动在线策略蒸馏

    Describe, Don't Ask: Caption-Driven On-Policy Distillation for Fine-Grained Visual Perception

    [https://arxiv.org/abs/2611.32463](https://arxiv.org/abs/2611.32463)

    该论文提出用图像描述（caption）取代人工构建的区域问答对来驱动在线策略蒸馏，从而将无标注图像转化为广泛且可扩展的细粒度视觉监督信号，同时避免目标缺失时的幻觉作答。

    

    视觉语言模型（VLM）在相关裁剪区域中识别细粒度细节时，往往比在完整图像中更加准确。多模态在线策略蒸馏（OPD）方法正是利用这一特性，将裁剪视图教师模型的优势迁移到全图学生模型中。然而，这类方法的监督信号依赖于通过复杂合成流程构建的针对特定区域的问题，这带来了三个缺陷：（1）每个区域都需要专门为其编写的问答对，使训练受制于成本高昂的问答构建流程，难以扩展到无标注图像；（2）一个区域的所有采样 rollout 都回答同一个问题，教师模型因此只监督其询问到的细节，而该区域中的其他视觉元素则缺乏监督；（3）每个问题都预设了目标必然存在，导致学生模型在目标缺失时也倾向于作答。本文研究如何将无标注图像转化为广泛的视觉监督（原文摘要在此处截断）。

    arXiv:2611.32463v1 Announce Type: cross  Abstract: Vision-language models (VLMs) often recognize fine-grained details more accurately in a relevant crop than in the full image. Multimodal on-policy distillation (OPD) methods exploit this by transferring the advantage of a crop-view teacher to a full-image student. However, their supervision relies on region-specific questions curated through an elaborate synthesis pipeline, which brings three weaknesses: (1) every region needs a question--answer pair written exclusively for it, locking training into costly QA curation that is hard to scale to unlabeled images; (2) all rollouts of a region answer the same question, so the teacher supervises only the details it asks about, leaving other visual elements in the region unsupervised; and (3) every question presupposes that its target exists, so the student becomes prone to answering even when the target is absent. In this paper, we study how to turn unlabeled images into broad visual supervi
    
[^82]: 看见答案，学会嵌入：从后见之明中学习通用多模态嵌入

    See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight

    [https://arxiv.org/abs/2611.32439](https://arxiv.org/abs/2611.32439)

    该论文提出HEME方法，在训练时将匹配的候选目标作为“后见之明”上下文来学习基于查询的表示，使模型在推理时无需额外推理计算即可生成高质量的通用多模态嵌入。

    

    通用多模态嵌入（UME）模型将异构的多模态输入映射到一个共享的嵌入空间中，其中成对相似度可支持多种下游任务。近期的推理驱动的UME方法使用中间推理为嵌入生成额外上下文，但通常会在推理阶段引入额外的计算开销。在本文中，我们探索将匹配的候选（即配对的目标）作为一种替代性的上下文来源，研究模型能否在训练时“看见答案”，而在推理时无需它也能“学会嵌入”。然而，直接以这种特权上下文为条件，可能会引入仅凭查询本身无法获得的候选特定信息，或者导致以候选为条件的查询表示向候选嵌入坍缩。为解决这一问题，我们提出了后见之明增强的多模态嵌入（HEME），它从这种候选上下文中学习基于查询的后见之明表示。HEME

    arXiv:2611.32439v1 Announce Type: cross  Abstract: Universal Multimodal Embedding (UME) models map heterogeneous multimodal inputs into a shared embedding space, where pairwise similarity supports diverse downstream tasks. Recent reasoning-driven UME approaches use intermediate reasoning to generate additional context for embedding, but often incur extra computation at inference. In this paper, we explore the matched candidate (i.e., the paired target) as an alternative source of context, asking whether a model can "see the answer" during training and "learn to embed" without it at inference. However, direct conditioning on this privileged context can introduce candidate-specific information unavailable from the query alone or cause the candidate-conditioned query representation to collapse toward the candidate embedding. To address this, we introduce Hindsight-Enhanced Multimodal Embeddings (HEME), which learns query-grounded hindsight representations from such candidate context. HEME
    
[^83]: NRG-Twins：生成视频中刚性违例的像素级真值标注

    NRG-Twins: Pixel-Level Ground Truth for Rigidity Violations in Generated Videos

    [https://arxiv.org/abs/2611.32431](https://arxiv.org/abs/2611.32431)

    该论文提出NRG-Twins数据集，通过将真实视频片段与关键帧条件化的视频生成器重新生成的“孪生”视频配对，首次提供了生成视频中刚性违例（3D不一致）的像素级、以角度为单位的真值标注。

    

    视频生成器如今已能生成逼真的场景与物体片段，这使生成视频成为获取3D内容的一个有吸引力的来源。然而，生成视频并不具备3D一致性：静态内容会缓慢变形或滑动，3D重建质量随之下降，且这些错误过于细微，难以通过人工方式标注。为了使生成器具备3D一致性，或构建能够引导其朝此方向改进的奖励与引导机制，我们需要能够指出生成视频在何处、以多大程度违反刚性的数据，而这些数据必须来自生成视频本身。我们从“真实-生成孪生对”中获得逐像素真值标签：即将一个真实视频片段与其由关键帧条件化的视频生成器重新生成的版本配对。在插入真实关键帧使孪生视频的相机运动大致跟随真实相机后，我们以角度为单位逐像素地测量孪生视频沿其自身相机路径所见内容相对于刚性场景的偏离程度。我们发布了非刚性生成孪生数据集，包含871对孪生视频，以及用于构建该数据集的代码。

    arXiv:2611.32431v1 Announce Type: cross  Abstract: Video generators now produce realistic clips of scenes and objects, which makes generated video an attractive source of 3D content. Yet generated video is not 3D consistent: static content slowly deforms or slides, 3D reconstruction degrades, and the errors are too subtle to label by hand. To make generators 3D consistent, or to build rewards and guidance that push them there, we need data that say where and by how much a generated video breaks rigidity, and these data must come from generated video itself. We obtain per-pixel ground-truth labels from real-generated twins: a real clip paired with its re-creation by a keyframe-conditioned video generator. After inserting real keyframes so that the twin camera roughly follows the real one, we measure per pixel, in degrees, how far the twin departs from a rigid scene seen along its own camera path. We release Non-Rigidly Generated Twins (NRG-Twins), 871 twins, the code that builds them fo
    
[^84]: AI智能体的任务内风险：时间视野曲线无法识别，重复运行可以识别

    Within-task hazards of AI agents: unidentified by time-horizon curves, identified by repeated runs

    [https://arxiv.org/abs/2611.32415](https://arxiv.org/abs/2611.32415)

    该论文证明仅凭时间视野（任务完成率）曲线在统计上无法识别AI智能体任务内失败风险是否随任务推进而下降，并提出利用同一任务的条件独立重复运行（即两次运行均成功的概率曲线相对单次曲线的对数轴平移量）来恢复对任务内风险的可识别性。

    

    arXiv:2611.32415v1 公告类型：cross。摘要：AI智能体的时间视野是从一条曲线上读出的：该曲线给出智能体完成某任务的概率，横轴为人类完成该任务所需的时间。近期的若干分析对这条曲线拟合生存模型，并得出结论：随着任务的推进，智能体失败的风险（hazard）会下降。但这条曲线无法支持该结论。每条斜率 β≤1 的对数-逻辑斯蒂（log-logistic）视野曲线都是恒定风险曲线的混合；在比例风险脆弱性（frailty）模型中，任何任务内威布尔形状参数 k≥β 都会产生同一条曲线。如果同一任务的各次运行在条件上相互独立，那么重复运行可以在该模型内恢复可识别性：两次运行都成功的概率等于单次运行曲线在对数₂时间轴上平移 1/k 后的结果。在威布尔形式之外，这两条曲线之间的水平距离仍然能够指示风险的方向：如果平均任务内风险是非递增的，则该距离至少为……

    arXiv:2611.32415v1 Announce Type: cross  Abstract: The time horizon of an AI agent is read off a curve: the probability that the agent completes a task, plotted against the time a human needs for it. Several recent analyses fit survival models to this curve and conclude that agents' hazard of failure declines as a task goes on. The curve cannot support that conclusion. Every log-logistic horizon curve with slope $\beta\le1$ is a mixture of constant-hazard curves, and in a proportional-hazards frailty model the same curve is produced by every within-task Weibull shape $k\ge\beta$. If runs of a task are conditionally independent, repeated runs restore identification within this model: the probability that two runs both succeed is the single-run curve shifted by $1/k$ on the $\log_2$ time axis. Without the Weibull form, the horizontal distance between these two curves still bears on the direction of the hazard: if the average within-task hazard is non-increasing, the distance is at least 
    
[^85]: BEACON：学习观测条件化的起始点用于显微成像中的引导条件流匹配

    BEACON: Learning Observation-Conditioned Starting Points for Guided Conditional Flow Matching in Microscopy

    [https://arxiv.org/abs/2611.32406](https://arxiv.org/abs/2611.32406)

    BEACON提出了一种最优K自适应学习耦合方法，利用退化观测来条件化引导条件流匹配的训练起始点，从而避免任意噪声—目标配对导致的速度目标冲突，提升显微图像恢复的学习效果。

    

    荧光显微技术推动着现代生命科学的许多领域，但易于获取的成像装置往往产生因离焦光线、有限分辨率和强烈的信号依赖性噪声而退化的图像。计算恢复技术可以从这些数据中恢复高质量图像，从而在保留具有生物学意义的结构的前提下，实现更廉价、更快速、更低光剂量的成像。引导条件流匹配通过学习合理恢复结果的分布，最近在显微成像去雾和计算超分辨率方面取得了出色的成果。然而，该方法将每个干净目标与独立的标准高斯样本配对，因此每条训练路径的起始点并未利用退化的观测数据。这种任意的噪声—目标耦合可能产生相互冲突的速度目标，并使条件速度场更难学习。我们提出了BEACON，一种用于引导条件流匹配的最优K自适应学习耦合方法……（摘要内容不完整）

    arXiv:2611.32406v1 Announce Type: cross  Abstract: Fluorescence microscopy drives much of modern life science, but accessible imaging setups often yield images degraded by out-of-focus light, limited resolution, and strong signal-dependent noise. Computational restoration can recover high-quality images from such data, enabling cheaper and faster imaging at lower light dose, provided biologically meaningful structures are preserved. Guided conditional flow matching has recently shown strong results for microscopy dehazing and computational super-resolution by learning distributions of plausible restorations. However, it pairs each clean target with an independent standard Gaussian sample, so the starting point of every training path does not leverage the degraded observation. This arbitrary noise--target coupling can produce conflicting velocity targets and makes the conditional velocity field harder to learn. We present BEACON, a $BE$st-of-$K$ $A$daptive learned coupling for guided $C
    
[^86]: CPU-Eval：评估用于CPU硬件代码生成的大语言模型

    CPU-Eval: Evaluating Large Language Models for CPU Hardware Code Generation

    [https://arxiv.org/abs/2611.32405](https://arxiv.org/abs/2611.32405)

    CPU-Eval是一个基于六个真实CPU（含已流片设计）构建的、包含339个HDL生成问题的新基准，其难度远超VerilogEval，使最先进大语言模型的平均pass@1下降45.14个百分点。

    

    VerilogEval等基准测试已经推动了大语言模型（LLM）在硬件描述语言（HDL）生成方面的快速发展。然而，近期的HDL生成系统在VerilogEval上的通过率已超过90%，使得该基准难以进一步揭示模型的进步空间。为了给这一实践上重要的硬件开发任务提供更具挑战性的目标，我们提出了CPU-Eval，这是一个由339个HDL生成问题组成的基准测试，这些问题基于六个CPU构建。这些问题源自实际的CPU设计，包括已在硅片上流片制造的设计系列，涵盖从小型嵌入式核心到可运行Linux的乱序执行核心等各类CPU，并覆盖了用SystemVerilog、Chisel和SpinalHDL编写的多样化CPU功能。在每个问题中，LLM需要执行生成HDL这一真实的CPU开发任务。在七个最先进的大语言模型上，CPU-Eval上的平均pass@1比VerilogEval v2 Spec-to-RTL低45.14个百分点，暴露出了……

    arXiv:2611.32405v1 Announce Type: cross  Abstract: Benchmarks such as VerilogEval have supported rapid advances in hardware description language (HDL) generation by large language models (LLMs). However, recent HDL-generation systems achieve pass rates above 90% on VerilogEval, leaving limited headroom for revealing further advances. To provide a challenging target for further progress on a practically important hardware-development task, we introduce CPU-Eval, a benchmark of 339 HDL-generation problems constructed from six CPUs. Drawn from practical CPU designs, including families fabricated in silicon, the problems span CPUs from small embedded cores to Linux-capable out-of-order cores and cover diverse CPU functions written in SystemVerilog, Chisel, and SpinalHDL. In each problem, an LLM performs the real CPU-development task of generating HDL. Across seven state-of-the-art LLMs, mean pass@1 is 45.14 percentage points lower on CPU-Eval than on VerilogEval v2 Spec-to-RTL, exposing ca
    
[^87]: 先验证再引导：面向连续视觉-语言导航的进度幻觉感知恢复

    Verify then Steer: Progress Hallucination-Aware Recovery for Continuous Vision-and-Language Navigation

    [https://arxiv.org/abs/2611.32400](https://arxiv.org/abs/2611.32400)

    提出PHAR测试时恢复框架，通过检测视觉-语言模型残差流表示中的跨模态证据不匹配来识别“进度幻觉”现象，使导航智能体能够从路线偏离中有效恢复。

    

    视觉-语言导航（VLN）要求智能体遵循语言指令，在未见过的3D环境中到达目标位置。然而，现有方法在测试时仍然难以从路线偏离中恢复。我们通过指令进度的视角研究这一局限，并将其追溯至我们称之为“进度幻觉”的现象：智能体将偏离路线的观察错误地解读为任务进展的证据，导致其内部的指令进度超前于实际任务阶段，从而阻碍了基于较早子指令的恢复。为了解决这一失败模式，我们提出了PHAR（进度幻觉感知恢复），这是一个遵循“先验证后引导”范式的测试时恢复框架。验证器检查视觉-语言模型（VLM）的残差流表示，以识别轨迹中跨模态证据的不匹配，并使用分裂保形校准来控制误报。

    arXiv:2611.32400v1 Announce Type: cross  Abstract: Vision-and-Language Navigation (VLN) requires an agent to follow language instructions to reach a target in an unseen 3D environment. However, existing methods still struggle to recover from route deviations at test time. We investigate this limitation through the lens of instruction progress, and trace it to a phenomenon we term progress hallucination: agents misinterpret off-route observations as evidence of task progress, causing their internal instruction progress to advance beyond the actual task stage and hindering recovery based on earlier sub-instructions. To address this failure mode, we propose PHAR (Progress Hallucination-Aware Recovery), a test-time recovery framework following a verify-then-steer paradigm. The verifier examines residual-stream representations of the Vision-Language Model (VLM) to identify mismatch of cross-modal evidence along the trajectories, and uses split conformal calibration to control false alarms. 
    
[^88]: 并非所有样本对都对齐：字幕引导的音视频对比学习

    Not All Pairs Align: Caption-Guided Audio-Visual Contrastive Learning

    [https://arxiv.org/abs/2611.32399](https://arxiv.org/abs/2611.32399)

    该论文提出了一种模型无关的对比学习目标CGAAL，利用模态特定字幕估计音视频间的语义一致性，并据此为正样本对加权，从而解决了画外音等因素导致的音视频内容不匹配问题。

    

    我们研究从共现模态中学习音视频表示的问题。多模态对比学习通常假设成对观测提供的是同一底层信息的共享视图，这对于在受限环境中收集的数据集可能成立，但并不总是如此。例如，由于画外音的存在，共现的音频与视觉信号在内容上可能存在显著差异。然而，标准对比学习目标将所有配对样本视为同等对齐。在这项工作中，我们提出了字幕引导的对齐感知损失（CGAAL），这是一种与模型无关的目标函数，利用特定模态的字幕来估计音频与视频之间的语义一致性。CGAAL使用这些一致性分数来为每个正样本对的贡献加权，并通过预热期和渐进阈值策略逐步引入这种对齐感知的训练信号。我们展示了……

    arXiv:2611.32399v1 Announce Type: cross  Abstract: We address the problem of learning audio-visual representations from co-occurring modalities. Multimodal contrastive learning typically assumes that paired observations provide shared views of the same underlying information. While this may be true for datasets collected in constrained environments, it does not always hold. For instance, co-occurring audio and visual signals can differ substantially in their content due to off-screen sounds. Nevertheless, standard contrastive objectives treat all paired samples as equally aligned. In this work, we propose Caption Guided Alignment Aware Loss (CGAAL), a model-agnostic objective that leverages modality-specific captions to estimate semantic agreement between audio and video. CGAAL uses these agreement scores to weight the contribution of each positive pair. A warm-up period and gradual thresholding strategy progressively introduce this alignment-aware training signal. We demonstrate the g
    
[^89]: 进化以学习：面向模型自我改进的梯度对齐经验进化方法

    Evolve to Learn: Gradient-aligned experience evolution for model self-improvement

    [https://arxiv.org/abs/2611.32390](https://arxiv.org/abs/2611.32390)

    该论文揭示了一个关键问题：能提升测试时性能的经验并不一定值得内化到模型中，并提出以梯度对齐作为更可靠的经验筛选标准，据此构建了E2L自进化框架来实现模型的自我改进。

    

    自进化语言模型可以通过测试时交互、反思和提示优化持续获取有用经验，然而如何将这些经验转化为模型自身的能力以确保自我改进仍然具有挑战性。在线策略自蒸馏（OPSD）为将测试时信息内化到模型参数中提供了一种自然的机制，但它很大程度上假设所生成的经验是值得学习的。我们证明这一假设并不总是成立：能够提升测试时性能的经验可能提供有限甚至有害的训练信号。为了更好地刻画哪些经验应该被内化，我们系统地分析了跨多个任务和反思模型的多样化变异经验，发现基于梯度的对齐度是比即时的测试时性能提升更可靠的下游蒸馏收益指标。基于这一发现，我们提出了E2L，一种自进化方法框架。

    arXiv:2611.32390v1 Announce Type: cross  Abstract: Self-evolving language models can continuously acquire useful experience through test-time interaction, reflection, and prompt optimization, yet turning such experience into model itself to ensure self improvement remains challenging. On-policy self-distillation (OPSD) provides a natural mechanism for internalizing test-time information into model parameters, yet it largely assumes that the gener- ated experience is worth learning. We show that this assumption does not always hold: experiences that improve test-time performance may provide limited or even harmful training signals. To better characterize which experience should be inter- nalized, we systematically analyze diverse mutated experiences across multiple tasks and reflection models, and find that gradient-based alignment is a more reli- able indicator of downstream distillation gains than immediate test-time improve- ment. Based on this finding, we propose E2L, a self-evoluti
    
[^90]: AI能设计出更好的nnU-Net吗？一种从失败到设计规则的假设驱动架构发现循环

    Can AI Design a Better nnU-Net? A Hypothesis-Driven Architecture Discovery Loop from Failures to Design Rules

    [https://arxiv.org/abs/2611.32384](https://arxiv.org/abs/2611.32384)

    该论文提出一种假设驱动的架构发现循环，让LLM智能体从失败案例中诊断错误模式、提出可证伪假设并验证机制，最终提炼出两条设计规则，构建出比nnU-Net计算量更低、分割精度更高的3D医学图像分割网络LeanNeXt。

    

    神经架构搜索和LLM驱动的进化方法通常将评估简化为一个标量适应度分数，从而丢弃了那些本可以解释设计为何成功或失败的失败案例。我们提出了一种假设驱动的架构发现循环，其中LLM智能体诊断结构化的错误模式、提出可证伪的假设、在冻结的nnU-Net流水线下测试反事实对照实验，并将经过验证的机制提炼为可执行的“触发-行动”规则。在八轮搜索过程中，该循环为3D医学图像分割识别出了两条设计原则：第一，将计算资源从高分辨率解码器重新分配到中部编码器，可以在降低成本的同时保留细薄解剖结构的边界；第二，支撑受限的区域对比能够解决嵌套解剖腔室之间的错误，并通过依赖解剖结构的触发器来决定该规则何时适用。这些原则催生了LeanNeXt（相对于nnU-Net，其计算量为0.78倍MACs、训练步时间为0.92倍；性能提升+1.63 Dice）。

    arXiv:2611.32384v1 Announce Type: new  Abstract: Neural architecture search and LLM-driven evolution often reduce evaluation to a scalar fitness score, discarding failures that could explain why designs succeed or fail. We present a hypothesis-driven architecture discovery loop in which LLM agents diagnose structured error patterns, formulate falsifiable hypotheses, test counterfactual controls under a frozen nnU-Net pipeline, and distill validated mechanisms into executable trigger-to-action rules. Across eight search waves, the loop identifies two principles for 3D medical image segmentation. First, reallocating computation from high-resolution decoders to the mid-encoder preserves thin-anatomy boundaries while reducing cost. Second, support-confined region contrast addresses errors between nested anatomical compartments, with anatomy-dependent triggers selecting when the rule applies. These principles produce LeanNeXt (0.78 MACs and 0.92 training-step time relative to nnU-Net; +1.63
    
[^91]: 提供研究论文上下文能否改进大语言模型生成的四足机器人导航？一项探索性仿真研究

    Does Research-Paper Context Improve LLM-Generated Quadruped Navigation? An Exploratory Simulation Study

    [https://arxiv.org/abs/2611.32381](https://arxiv.org/abs/2611.32381)

    这项探索性仿真研究发现，为 LLM 提供特定研究论文作为上下文并未改进四足机器人的导航代码，反而在迷宫最短路径中途被阻断的场景下显著降低了生成控制器的鲁棒性。

    

    大语言模型能够编写出合格的机器人导航代码，但由于该领域研究持续进展且成果数量庞大，模型可能无法召回相关研究发现并将其应用于当前任务。在这项工作中，我们测试了提供特定研究论文作为上下文是否能改进大语言模型为 Go2 四足机器人编写的导航代码，其中大语言模型（Claude Opus 5）和运动控制策略均保持冻结。具体而言，我们在三种设置下比较了大语言模型在有和没有特定论文作为上下文时编写的导航控制器：行为树、从共享基础版本编辑的导航器，以及从零开始编写的控制器。当迷宫的最短路径在回合中途被阻断时，以增量重规划论文作为上下文编写的 8 个控制器中仅有 2 个在 Isaac Sim 物理仿真下完成了全部四次行程，而没有该上下文的 8 个控制器中有 7 个完成了任务（双侧 Fisher 精确检验，p = 0.041）。这一差异仅在……（原文摘要在此处被截断）

    arXiv:2611.32381v1 Announce Type: cross  Abstract: Large language models can write competent robot navigation code, but not only does the research on the topic advance, a model may also, due to the volume of findings, fail to recall relevant findings and apply them to the task at hand. In this work, we test whether providing particular research papers as context improves the LLM-written navigation code for the Go2 quadruped, with the LLM (Claude Opus 5) and locomotion policy frozen. Specifically, we compare the navigation controllers the LLM writes when provided with and without particular papers across three settings: behavior trees, navigators edited from a shared base, and controllers written from scratch. When a maze's shortest route is blocked mid-episode, 2 of 8 controllers written with the context of an incremental-replanning paper complete all four trips under Isaac Sim physics, against 7 of 8 without it (two-sided Fisher's exact test, p = 0.041). This difference appears only u
    
[^92]: 验证需求工程中多模型AI生成的验收标准

    Validating Multi-Model AI-Generated Acceptance Criteria in Requirements Engineering

    [https://arxiv.org/abs/2611.32374](https://arxiv.org/abs/2611.32374)

    本文提出一种结合检索增强生成、多模型生成与人机协同工作流的面向验证方法，用于解决需求工程中AI生成的验收标准缺乏有效性验证的问题。

    

    大语言模型（LLMs）在软件工程中被越来越多地用于生成需求、用户故事、验收标准、文档和代码等制品。它们在需求工程中的应用引发了对正确性、可追溯性和幻觉问题的重要关注，尤其是因为有缺陷的规格说明会影响后续的设计、实现和验证活动。尽管近年来关于AI辅助软件工程的研究大多集中在代码生成和代码级验证上，但对AI生成的规格说明的验证却较少受到关注。本文通过提出一种面向验证的AI生成验收标准方法来填补这一空白。该方法结合了检索增强生成（RAG）、多模型生成以及明确结构化的人机协同工作流程。我们描述了一个可工作的原型系统，允许用户准备或输入用户故事并生成验收标准。

    arXiv:2611.32374v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in software engineering to generate artifacts such as requirements, user stories, acceptance criteria, documentation, and code. Their use in requirements engineering raises important questions about correctness, traceability, and hallucination, especially because flawed specifications can affect later design, implementation, and verification activities. While much recent work on AI-assisted software engineering has focused on code generation and code-level validation, the validation of AI-generated specifications has received less attention. This paper addresses this gap by presenting a validation-oriented approach for AI-generated acceptance criteria. The approach combines retrieval-augmented generation (RAG), multi-model generation, and an explicitly structured human-in-the-loop workflow. We describe a working prototype that allows users to prepare or enter user stories, generate acc
    
[^93]: 控制流税：面向大语言模型智能体工作流的专用决策路由

    The Control-Flow Tax: Specialized Decision Routing for LLM Agent Workflows

    [https://arxiv.org/abs/2611.32365](https://arxiv.org/abs/2611.32365)

    该论文提出一种异构架构，用轻量专用的“系统1”决策模型Jev接管控制流路由、由大模型Sol专注文档生成，从而显著降低LLM智能体工作流中的“控制流税”（成本降低约40%–50%，耗时降低约33%）。

    

    智能体在“决定做什么”上花费的开销几乎与“实际执行”相当。在我们的文档工作流中，控制流路由占初始任务基准参考成本的42.5%。仅靠降低token价格并不能解决这一“控制流税”问题：廉价的通用路由器可能会触发对昂贵生成器的额外调用。我们评估了一种异构架构，其中Jev——一个专用的“系统1”决策模型——负责控制有限的动作集合，而Sol负责生成文档。在四项软件任务中，480次有效运行包含470次完成和十次因请求上限而终止的终点。Jev全路由在每个任务上都实现了最低的平均成本与耗时，所有40次由专用模型控制的运行均沿最短轨迹完成。在初始任务上，平均成本下降49.6%（95%配对自助法区间：42.9%至56.3%），耗时下降33.2%（19.6%至46.7%）。在另外三项任务中，成本降幅为40.5%至42.9%……

    arXiv:2611.32365v1 Announce Type: new  Abstract: An agent can spend almost as much deciding what to do as doing it. In our documentation workflow, control-flow routing accounts for 42.5% of baseline reference cost on the initial task. Lower token prices alone do not solve this control-flow tax: inexpensive general-purpose routers can trigger additional calls to the expensive generator. We evaluate a heterogeneous architecture in which Jev, a specialized System 1 decision model, controls finite actions while Sol generates documents. Across four software tasks, 480 valid runs comprise 470 completions and ten request-limit endpoints. Jev full routing achieves the lowest mean cost and elapsed time on every task, completing all 40 specialist-controlled runs along the shortest trajectory. On the initial task, mean cost falls by 49.6% (95% paired-bootstrap interval: 42.9% to 56.3%) and elapsed time by 33.2% (19.6% to 46.7%). Across three further tasks, cost reductions are 40.5% to 42.9% and c
    
[^94]: DiCT：通过想象CT教会X射线视觉-语言模型进行推理

    DiCT: Teaching X-ray Vision-Language Models to Reason via Imagining CT

    [https://arxiv.org/abs/2611.32344](https://arxiv.org/abs/2611.32344)

    提出DiCT三阶段渐进迁移框架，利用CT视觉-语言模型指导X射线模型学习3D空间推理，使其无需CT重建即可直接从2D X射线完成具有3D空间意识的诊断推理。

    

    使用X射线进行诊断通常需要对异常的3D位置及其与周围解剖结构的空间关系进行推理，但由于2D投影中空间信息的丢失，这类推理仍然具有挑战性。为解决这一难题，我们提出了Dreaming in CT（DiCT），一个三阶段的渐进式CT到X射线推理迁移框架，利用CT视觉-语言模型（VLM）来指导X射线VLM。通过这种CT引导的迁移，学生模型获得了受CT启发的3D推理能力，使其能够在推理时直接从2D X射线进行推理，而无需显式的CT重建。该框架首先通过“指导梦想家”阶段，为X射线学生模型赋予初始的CT导向推理能力，并建立一个结构化接口，将X射线观察与CT想象区分开来。在此结构基础上，“可视化梦境”阶段引入了CT引导的……

    arXiv:2611.32344v1 Announce Type: cross  Abstract: Diagnosing with X-rays often requires reasoning about the 3D locations of abnormalities and their spatial relationships to surrounding anatomical structures, yet such reasoning remains challenging due to the loss of spatial information in 2D projections. To address this challenge, we propose Dreaming in CT (DiCT), a three-stage progressive CT-to-X-ray reasoning transfer framework that leverages a CT vision-language model (VLM) to guide an X-ray VLM. Through this CT-guided transfer, the student acquires CT-informed 3D reasoning capabilities, enabling it to reason directly from 2D X-rays without explicit CT reconstruction at inference time. The framework begins with instruct the dreamer, which equips the X-ray student with initial CT-oriented reasoning capabilities and establishes a structured interface that distinguishes X-ray observations from CT imagination. Building on this structure, visualize the dream introduces CT-guided on-polic
    
[^95]: 用于可解释山瞪羚性别分类的部分集合原型

    Part-Set Prototypes for Explainable Mountain Gazelles Sex Classification

    [https://arxiv.org/abs/2611.32343](https://arxiv.org/abs/2611.32343)

    本文提出“集合原型”通用架构，通过截断Sinkhorn竞争机制在冻结的DINOv3令牌网格上进行联合匹配，直接从数据中挖掘部分数量与无约束空间布局，在山瞪羚性别分类相机陷阱基准上实现了0.878宏观F1的可解释识别性能。

    

    用于细粒度识别的自解释视觉模型常常受到僵硬几何约束的阻碍，这些约束无法捕捉无约束“野外”图像中多样化的语义形状。现实世界的相机陷阱数据带来了诸多挑战，包括非最优姿态、光照变化、遮挡、动物伪装以及环境捷径。为解决这些问题，我们提出了集合原型，这是一种通用架构，其中部分数量和无约束空间布局都直接从数据中挖掘获得。我们的联合匹配公式通过截断Sinkhorn竞争将各部分同时放置在冻结的DINOv3令牌网格上，允许缺乏支撑的部分弃权而不发生空间坍塌。我们在山瞪羚性别分类这一相机陷阱基准上评估了我们的方法，模型达到了0.878的宏观F1分数——超越了先前的架构，并与无约束的线性探测相当。

    arXiv:2611.32343v1 Announce Type: cross  Abstract: Self-explainable visual models for fine-grained recognition are often hindered by rigid geometric constraints that cannot capture diverse semantic shapes in unconstrained "in-the-wild" imagery. Real-world camera-trap data introduces challenges, including non-optimal poses, lighting variations, occlusions, animal camouflage, and environmental shortcuts. To address this, we introduce set prototypes, a general architecture where both part cardinality and unconstrained spatial layouts are mined directly from data. Our joint-matching formulation places parts simultaneously on a frozen DINOv3 token grid via truncated Sinkhorn competition, allowing unsupported parts to abstain without spatial collapse. We evaluate our approach on Mountain Gazelle sex classification, a camera-trap benchmark, where our model achieves 0.878 macro-F1--outperforming prior architectures and at par with an unconstrained linear probe. The strongest prototypes localiz
    
[^96]: 面向电子显微分析的递归自我改进框架

    A recursive self-improvement framework for electron microscopy analysis

    [https://arxiv.org/abs/2611.32325](https://arxiv.org/abs/2611.32325)

    该论文提出PhyEvo-Mic-RSI，一个物理引导与专家知识相结合的递归自我改进框架，能将研究人员的一次性人工修正转化为可复用、可验证的图像分析能力，显著提升电子显微图像中原子缺陷检测性能和跨材料界面识别覆盖率。

    

    从电子显微镜衬度中解析缺陷和界面等局部原子结构高度依赖研究人员的判断，导致效率低下且标准主观。在此，我们提出 PhyEvo-Mic-RSI，一个物理引导、专家参与的递归自我改进框架，它通过带版本继承的方式更新可执行算子、参数和路由规则，将一次性的人工修正转化为可复用、可验证的能力。对于研究人员定义的任务，该系统无需专家标注即可根据图像特征选择分析路径，并在固定预算下测试和继承代码更新。将物理先验与专家反馈相结合，使原子点缺陷检测性能超越了更大的模型，并在10%面积预算内将跨材料界面覆盖率从5.13%提升至72.11%。将该框架适配到来自未见研究的显微图像上，进一步通过对齐最优……（摘要在此处截断）

    arXiv:2611.32325v1 Announce Type: cross  Abstract: Resolving local atomic structures such as defects and interfaces from electron microscopy contrast relies heavily on researcher judgement, leading to low efficiency and subjective standards. Here we present PhyEvo-Mic-RSI, a physics-guided and expert-informed recursive self-improvement framework that turns one-off human corrections into reusable, verifiable capabilities by updating executable operators, parameters and routing rules with versioned inheritance. Given researcher-defined tasks, the system selects analysis paths from image features without expert labels, while testing and inheriting code updates under fixed budgets. Combining physical priors with expert feedback improves atomic point-defect detection over larger models and raises cross-material interface coverage from 5.13% to 72.11% within a 10% area budget. Adapting the framework to micrographs from unseen studies further separates touching nanoparticles by aligning optim
    
[^97]: 当精确的词元重要性不够时：大语言模型注意力中Top-K选择的局限性

    When Exact Token Importance Is Not Enough: Limits of Top-K Selection in LLM Attention

    [https://arxiv.org/abs/2611.32324](https://arxiv.org/abs/2611.32324)

    本文证明了即使采用精确的单点删除效应作为词元重要性评分，Top-K独立排名选择仍会次优，因为注意力归一化引入的位置间交互可使排名组合与最优组合之间的KL散度差距高达133.2倍且在理论上无界。

    

    长上下文语言模型在进行注意力查询时，通常只保留有限的过去词元位置集合。一种常见策略是对各位置单独打分并保留排名最高的K个。然而，注意力权重会在所选集合上重新归一化：两个单独来看很重要的位置，未必能构成最佳的组合。我们探究了这样一个问题：即使每个重要性得分精确度量了仅删除该位置时对下一词元预测造成的下游变化，独立排名是否仍然足够。答案是否定的。一个四位置构造表明，按排名选出的组合所产生的Kullback-Leibler（KL）散度是最优组合的133.2倍，而一个参数化族则证明这一乘性差距是无界的。一个精确的交换恒等式，以及针对每个K >= 2 都存在的偏好反转现象，解释了这种交互作用。在对SmolLM2、Gemma和Qwen的穷举式冻结模型审计中，精确单点评分的Top-4在完整枚举下仍留有至少10%的相对改进空间……

    arXiv:2611.32324v1 Announce Type: cross  Abstract: Long-context language models often retain only a limited set of past token positions for an attention query. A common strategy scores positions individually and keeps the highest-ranked K. Yet attention weights are renormalized over the chosen set: two individually important positions need not form the best pair. We ask whether independent ranking suffices even when each importance score measures the exact downstream change in next-token predictions caused by deleting that position alone. It does not. A four-position construction makes the ranked pair incur 133.2 times the Kullback-Leibler (KL) divergence of the optimal pair, and a parametric family proves that this multiplicative gap is unbounded. An exact exchange identity and preference reversals for every K >= 2 explain the interaction. In exhaustive frozen-model audits of SmolLM2, Gemma, and Qwen, exact-singleton Top-4 leaves at least 10% relative improvement within a fully enumer
    
[^98]: FigmaEval：一个用于评估AI智能体专业UI设计能力的基准测试

    FigmaEval: A Benchmark for Evaluating AI Agents in Professional UI Design

    [https://arxiv.org/abs/2611.32321](https://arxiv.org/abs/2611.32321)

    该论文提出了FigmaEval基准测试，基于真实Figma设计文件、采用“约束规格、开放解法”范式，通过22项专家评估标准和专业设计师盲测竞技场来评估AI智能体的专业UI设计能力。

    

    AI智能体的最新进展使得使用Figma等专业工具进行自主UI设计成为可能，但由于设计任务本身具有开放性且缺乏唯一的真实标准答案，评估其设计能力仍然具有挑战性。我们提出了FigmaEval，这是一个用于评估AI智能体在专业UI设计任务上表现的基准测试，包含84个任务和783个屏幕界面，均来源于真实的Figma设计文件。FigmaEval采用“约束规格、开放解法”的范式，在标准化任务要求和设计资源的同时，允许智能体自主决定布局、视觉层次、组件构成和视觉呈现方式。为了超越与参考设计的相似度来评估设计质量，我们开发了22项基于专家意见的评估标准，涵盖六个维度，并引入了人类评估竞技场，由专业设计师通过盲测的成对比较来判断智能体生成的设计。对十一个（模型/智能体的评估结果原文在此截断）……

    arXiv:2611.32321v1 Announce Type: new  Abstract: Recent advances in AI agents have enabled autonomous UI design using professional tools such as Figma, yet evaluating their design capabilities remains challenging due to the open-ended nature of design tasks and the absence of unique ground truths. We introduce FigmaEval, a benchmark for evaluating AI agents on professional UI design tasks, comprising 84 tasks and 783 screens derived from real-world Figma design files. FigmaEval adopts a constrained-specification, open-solution paradigm, standardizing task requirements and design resources while allowing agents to independently determine layouts, visual hierarchies, component composition, and visual presentation. To assess design quality beyond reference similarity, we develop 22 expert-grounded evaluation rubrics across six dimensions and introduce a human evaluation arena where professional designers compare agent-generated designs through blinded pairwise judgments. Evaluating eleven
    
[^99]: 基于单子延续的Alpha-Beta剪枝

    Alpha-Beta Pruning via Monadic Continuations

    [https://arxiv.org/abs/2611.32320](https://arxiv.org/abs/2611.32320)

    该论文提出了一种基于延续单子与读取器单子组合的Alpha-Beta剪枝新表述，将搜索界限记录在单子结果中而非走法类型中，并证明普通延续单子即可同时计算最优结果与最优走法序列。

    

    Alpha-Beta剪枝是博弈人工智能中的一项基础技术，通过避免探索无法影响最优决策的分支，实现高效的极小极大搜索。我们提出了一种基于延续单子与读取器单子相结合的Alpha-Beta剪枝的单子化表述。与以往将Alpha-Beta界限编码到可用走法类型中的方法不同，我们的构造将这些界限记录在单子化的结果中，而保持走法类型不变。我们进一步证明，普通的延续单子不仅可以计算最优结果，还可以通过扩展结果类型使其包含相应的走法序列来计算最优对局走法。这为解决选择单子在Alpha-Beta搜索中低效问题而提出的延续单子泛化方案提供了一种替代方案。我们通过井字棋的实现来演示我们的方法，观察到39（摘要在此处截断，推测为39倍的性能提升）

    arXiv:2611.32320v1 Announce Type: new  Abstract: Alpha-beta pruning is a fundamental technique in game-playing AI, enabling efficient minimax search by avoiding the exploration of branches that cannot affect the optimal decision. We present a monadic formulation of alpha-beta pruning based on the combination of the continuation monad and the reader monad. Unlike previous approaches that encode alpha-beta bounds into the types of available moves, our construction records these bounds within monadic outcomes, leaving the move types unchanged. We further show that the ordinary continuation monad suffices to compute not only optimal outcomes but also optimal plays, by extending the outcome type to include the corresponding sequences of moves. This provides an alternative to generalisations of the continuation monad proposed to address the inefficiencies associated with the selection monad in alpha-beta search. We illustrate our approach with an implementation of Tic-Tac-Toe, observing a 39
    
[^100]: SPRUNN：用于深度神经网络剪枝的结构接近度方法

    SPRUNN: Structural Proximity for Pruning in Deep Neural Networks

    [https://arxiv.org/abs/2611.32317](https://arxiv.org/abs/2611.32317)

    本文提出基于结构接近度准则的滤波器剪枝方法SPRUNN，通过全面比较滤波器间的相似性识别并剪除不重要的滤波器，并与全连接层细粒度剪枝相结合，在不显著损失准确率的前提下压缩深度神经网络。

    

    深度神经网络在各类机器学习任务中展现出了显著的准确率。然而，这些网络存在参数量和计算量巨大的问题，导致处理过程非常耗时。因此，参数剪枝方法被提出用于压缩深度神经网络并实现高效部署。剪枝的主要挑战在于选择合适的剪枝参数，同时不显著损害模型准确率。本文提出了SPRUNN，一种基于结构接近度准则的算法剪枝方法，用于深度神经网络卷积层中的滤波器剪枝。结构接近度（SP）通过考虑所有滤波器并将每个滤波器与其他滤波器进行比较来计算滤波器相似度，从而有效识别卷积层中较不重要的滤波器进行剪枝。此外，我们将该方法与全连接层中的细粒度剪枝相结合。

    arXiv:2611.32317v1 Announce Type: cross  Abstract: Deep Neural Networks (DNNs) have demonstrated significant accuracy across various machine learning tasks. However, these networks suffer from a huge number of parameters and computations, which leads to time-consuming processes. Consequently, parameter pruning approaches have been proposed to compress DNNs and efficiently implement them. The main challenge in pruning is selecting the appropriate parameters for pruning without significantly damaging the accuracy. In this paper, we propose SPRUNN, an algorithmic pruning approach based on the structural proximity criterion for filter pruning in convolutional layers of deep neural networks. Structural Proximity (SP) calculates filter similarity by considering all filters and comparing each one with the others, effectively identifying less important filters for pruning in the convolutional layers. Furthermore, we integrate this approach with fine-grained pruning in Fully Connected(FC)layers
    
[^101]: ASTERISM：多模态大语言模型真的会看所有图像吗？

    ASTERISM: Do Multimodal LLMs Actually Look at All the Images?

    [https://arxiv.org/abs/2611.32316](https://arxiv.org/abs/2611.32316)

    该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。

    

    多图像基准旨在测试多模态大语言模型（MLLM）能否跨图像整合证据，但其中许多题目仅凭单张图像即可回答。通过对来自19个公开基准的约3.3万道题目进行审计，并采用两阶段单图过滤器，我们发现其中24.2%的题目仅凭一张图像或仅凭问题本身即可解答。在这些题目上，MLLM仅凭一张图像就能获得全部图像相对纯文本所带来的准确率提升的中位数78%，因此这些题目实际上奖励了“从单张图像作答”的行为，我们称之为单图捷径（single-image shortcut）。即便在单张图像不足以作答的题目上，MLLM也会走这条捷径：在这类题目上，它们的大部分错误答案与其仅凭单张图像时给出的答案重复，且该比例远高于随机水平。我们基于这19个基准构建了ASTERISM——据我们所知是规模最大的经过审计的多图像基准——仅保留其中需要多张图像才能作答的题目。其1.87万道题目（其中2000道经过修正或重写）共享同一（摘要在此处截断）。

    arXiv:2611.32316v1 Announce Type: cross  Abstract: Multi-image benchmarks aim to test whether a multimodal large language model (MLLM) combines evidence across images, but many of their items can be answered from a single image. Across about 33k items from 19 public benchmarks, our audit and a two-stage single-image filter find that one image or the question alone settles 24.2% of them. On these items, MLLMs recover from one image a median 78% of the accuracy that all images add over the text alone, so these items reward answering from one image, which we call the single-image shortcut. MLLMs take this shortcut even on items where one image is not enough. On such items, most of their wrong answers repeat an answer they give from one image, at a rate well above chance. We build ASTERISM, to our knowledge the largest audited multi-image benchmark, from these 19 benchmarks by keeping their items that need more than one image. Its 18.7k items, 2.0k of them corrected or rewritten, share one
    
[^102]: 稀疏毒化，强力劫持：仅需1%毒化率即可在视觉-语言-动作模型中实现高效的轨迹级后门攻击

    Sparse Poison, Strong Hijack: Efficient Trajectory-Level Backdoors in Vision-Language-Action Models with Only 1\% Poisoning

    [https://arxiv.org/abs/2611.32304](https://arxiv.org/abs/2611.32304)

    该论文提出了一种由视觉、语言和动作共同引导的两阶段协同优化框架，仅需1%的毒化率即可在视觉-语言-动作（VLA）模型中有效植入轨迹级后门攻击，解决了低毒化率下后门学习与干净任务性能之间的权衡难题。

    

    视觉-语言-动作（VLA）模型推动了机器人操作技术的发展，但其对演示数据的依赖使其面临后门威胁。在这些威胁中，轨迹级定向后门攻击尤其具有挑战性，且可能比破坏行为的攻击带来更大的安全风险：这类攻击不仅仅是导致任务失败或局部动作偏差，而是必须维持闭环重定向以完成攻击者指定的多步骤任务。为了可靠地诱导此类行为，现有方法通常依赖较高的毒化率来获取足够的目标轨迹监督。然而，高毒化率可能会损害干净任务的性能，而低毒化率则会导致目标轨迹监督稀缺，使后门学习容易受到占主导地位的干净任务更新的干扰。为了在低毒化率下解决这一权衡问题，我们提出了一种由视觉、语言和动作共同引导的两阶段协同优化框架……

    arXiv:2611.32304v1 Announce Type: new  Abstract: Vision-Language-Action (VLA) models advance robotic manipulation, but their reliance on demonstrations exposes them to backdoor threats. Among these threats, trajectory-level targeted backdoors are particularly challenging and may pose greater safety risks than behavior-disrupting attacks: rather than merely causing task failures or local action deviations, they must sustain closed-loop redirection to complete an attacker-specified multi-step task. To reliably induce such behaviors, existing methods typically rely on high poisoning rates to obtain sufficient target-trajectory supervision. However, high poisoning rates may compromise clean-task performance, whereas low rates leave target-trajectory supervision scarce and backdoor learning vulnerable to interference from dominant clean-task updates. To resolve this trade-off at low poisoning rates, we propose a two-stage co-optimization framework guided by vision, language, and action. Sta
    
[^103]: 面向云端视觉语言模型选择性披露的查询引导视觉最小化

    Query-Guided Visual Minimization for Selective Disclosure to Cloud VLMs

    [https://arxiv.org/abs/2611.32302](https://arxiv.org/abs/2611.32302)

    提出QGVM框架，通过查询引导的本地视觉最小化处理，在将图像发送至云端VLM之前保留查询所需内容，同时抑制不必要的敏感信息泄露。

    

    云端托管的视觉语言模型（VLM）要求将图像发送到用户本地信任边界之外，这可能暴露查询并不需要的敏感信息。我们研究了查询引导的视觉最小化：一种可信的本地转换，在传输之前保留与查询相关的视觉内容并减少不必要的披露。与查询无关的匿名化方法不同，当查询需要时它保留相同的敏感内容，否则则抑制该内容。我们提出了QGVM框架，该框架将查询定位与独立的敏感项目清单相结合，在类型、实例和文本内容层面确定披露的必要性，并为黑盒下游VLM渲染最小化后的图像。我们在互补的基准上评估QGVM，衡量任务效用、可直接恢复的敏感信息以及推断性地理位置泄露。使用OpenAI GPT-6.1 Sol，QGVM在DoxBench上实现了85.3%的答案正确率。

    arXiv:2611.32302v1 Announce Type: cross  Abstract: Cloud-hosted vision-language models (VLMs) require sending images beyond the user's local trust boundary, potentially exposing sensitive information the query does not need. We study query-guided visual minimization: a trusted local transformation preserves query-relevant visual content and reduces unnecessary disclosure before transmission. Unlike query-agnostic anonymization, it preserves the same sensitive content when the query requires it and suppresses it otherwise. We introduce QGVM, a framework that combines query grounding with an independent sensitive-item inventory, determines disclosure necessity at the type, instance, and text-content levels, and renders a minimized image for a black-box downstream VLM. We evaluate QGVM on complementary benchmarks measuring task utility, directly recoverable sensitive information, and inferential geolocation leakage. With OpenAI GPT-6.1 Sol, QGVM achieves 85.3% DoxBench answer correctness 
    
[^104]: DAiSy：基于自适应效用估计的大语言模型预训练在线数据选择

    DAiSy: Online Data Selection by Adaptive Utility Estimation for LLM Pretraining

    [https://arxiv.org/abs/2611.32294](https://arxiv.org/abs/2611.32294)

    DAiSy通过一个与主模型同步更新的轻量级线性辅助模型，从冻结的文档嵌入预测基于梯度的数据效用分数，使在线数据选择中每个候选文档的评分仅需一次线性变换，从而消除了大语言模型预训练中数据选择高昂的计算开销。

    

    模型感知的在线数据选择会优先挑选在模型当前状态下贡献最大的训练文档。现有的在线选择器基于梯度的数据效用分数对候选文档进行排序，这要求对每个候选文档执行主模型的完整前向-反向传播。因此，对每个文档进行评分的代价与在它上面训练的代价相当，但其中只有一小部分文档被真正用于模型更新。我们提出了DAiSy，一个轻量级的线性辅助模型，它学习从冻结的文档嵌入中预测数据效用分数，从而消除了计算开销和基础设施负担。该线性估计器与主语言模型同步更新，拟合常规训练步骤反向传播中获得的精确的基于梯度的效用分数。因此，对候选文档进行评分只需一次线性变换，绕过了代价高昂的梯度对齐计算。我们证明了……（摘要在此处截断）

    arXiv:2611.32294v1 Announce Type: cross  Abstract: Model-aware online data selection prioritizes the training documents that contribute the most given the model's current state. Existing online selectors rank candidates upon gradient-based data utility scores, which requires the full forward-backward pass of the main model on every candidate document. Scoring each document therefore pays as much as training on it, yet only a small fraction is used for the real model update. We introduce DAiSy, a lightweight linear auxiliary model that learns to predict data utility scores from frozen document embeddings, which eliminates the computation overhead and infrastructure burdens. The linear estimator is updated alongside the main language model, fitting to the exact gradient-based utility scores obtained from the backward pass of regular training steps. Consequently, scoring a candidate requires only a linear transformation, bypassing the costly gradient-alignment computation. We demonstrate 
    
[^105]: VC-OPD：通过视觉线索在线策略自蒸馏学习关注重要内容

    VC-OPD: Learning to See What Matters via Visual Cue On-Policy Self-Distillation

    [https://arxiv.org/abs/2611.32292](https://arxiv.org/abs/2611.32292)

    该论文提出VC-OPD方法，将教师端特权信息从局部图像区域扩展为问题条件化的视觉线索，通过多阶段质量过滤与token级监督，显著提升小型多模态大语言模型在复杂视觉推理任务中的能力。

    

    视觉在线策略蒸馏（OPD）通常使用裁剪或放大的图像区域作为教师端的特权信息，以提升小型多模态大语言模型（MLLM）的细粒度感知能力。然而，局部像素级线索对于需要全局结构、空间关系和高级语义的复杂视觉推理是不够的。我们提出视觉线索在线策略自蒸馏（VC-OPD），将教师端特权信息从局部图像区域扩展到问题条件化的视觉线索。先进的多模态大语言模型生成捕捉关键视觉细节、结构关系以及回答每个问题所需语义证据的线索。多阶段质量过滤流程筛选出可靠的线索，这些线索仅在训练期间提供给教师。这使得在学生生成的轨迹上能够进行信息丰富的token级监督，帮助学生学会识别并对关键内容进行推理。

    arXiv:2611.32292v1 Announce Type: new  Abstract: Visual On-Policy Distillation (OPD) commonly uses cropped or zoomed-in image regions as teacher-side privileged information to improve fine-grained perception in small Multimodal Large Language Models (MLLMs). However, local pixel-level cues are insufficient for complex visual reasoning that requires global structure, spatial relationships, and high-level semantics. We propose Visual Cue On-Policy Self-Distillation (VC-OPD), which extends teacher-side privileged information beyond local image regions to question-conditioned visual cues. Advanced MLLMs generate cues capturing key visual details, structural relationships, and semantic evidence needed to answer each question. A multi-stage quality filtering process selects reliable cues, which are provided only to the teacher during training. This enables informative token-level supervision along student-generated trajectories, helping the student learn to identify and reason over critical 
    
[^106]: 复现Qwen3-Omni的Talker语音标签：bf16精度的Mimi，以及一个最后一帧陷阱

    Reproducing the Speech Labels of Qwen3-Omni's Talker: Mimi in bf16, and a Last-Frame Pitfall

    [https://arxiv.org/abs/2611.32286](https://arxiv.org/abs/2611.32286)

    本文通过白盒回归与校准后的教师强制似然分析，确定Qwen3-Omni的Talker训练标签来自kyutai的Mimi编解码器（moshi包、bf16精度），并揭示了transformers库中MimiModel最后一帧编码不一致的陷阱。

    

    Qwen3-Omni通过一个Talker（说话模块）进行语音合成，该模块预测12.5 Hz音频编解码器的16个码本，但生成其训练标签的编码器并未开源，因此无法在匹配的标签上训练新的声音。作者从已发布模型的行为中识别出了这个编码器。白盒回归表明，Talker、其多词预测器以及Code2Wav解码器的码本嵌入与kyutai的Mimi的前16个码本逐层匹配。一种基于解码器的测试看似有说服力，但未通过校准检验：在已知标签上训练的解码器无法区分不同实现。而使用13个在已知标签上训练的代码语言模型进行校准后的教师强制Talker似然则能够区分各实现：相比作者尝试的所有其他变体，Talker更偏好来自kyutai的moshi包且使用bf16权重的标签。响度归一化以及对20%及以上数据的频带限制均被排除。此外，transformers库中的MimiModel对最后一帧的编码方式存在差异……

    arXiv:2611.32286v1 Announce Type: cross  Abstract: Qwen3-Omni speaks through a Talker predicting 16 codebooks of a 12.5 Hz audio codec, but the encoder that made its training labels was not released, so new voices cannot be trained on matching labels. We identify this encoder from the released model's behavior. White-box regression shows that the codebook embeddings of the Talker, its multi-token predictor and the Code2Wav decoder match the first 16 codebooks of kyutai's Mimi layer by layer. A decoder-based test looks convincing but fails calibration: decoders trained on known labels cannot tell them apart. Teacher-forced Talker likelihoods, calibrated with 13 code language models trained on known labels, do tell implementations apart: the Talker prefers labels from kyutai's moshi package with bf16 weights over every other variant we tried. Loudness normalization and band-limiting of 20% or more of the data are ruled out. Separately, transformers' MimiModel encodes the last frame diffe
    
[^107]: CoMNO：用于长时程偏微分方程预测的换位子记忆神经算子

    CoMNO: Commutator Memory Neural Operators for Long-Horizon PDE Prediction

    [https://arxiv.org/abs/2611.32274](https://arxiv.org/abs/2611.32274)

    提出换位子记忆神经算子（CoMNO），通过交换“先下采样后滚动预测”与“先滚动预测后下采样”两种操作顺序所产生的性能差距来提取细尺度动力学细节，从而增强神经算子在长时程偏微分方程预测中捕捉细尺度特征的能力。

    

    使用传统数值方法在长时间范围内求解非线性偏微分方程（PDE）的计算代价高昂。虽然神经算子提供了一种高效的替代方案，但它们在预测轨迹上会累积近似误差，这通常是由于未能捕捉细尺度细节所导致的。在这项工作中，我们设计了一种换位子记忆神经算子，它通过对两个操作进行换位来识别细尺度动力学细节，并利用这些细节来增强神经算子在长时间范围内捕捉细尺度特征的能力。具体而言，我们发现了两种操作之间的性能差距：一种是在滚动预测之前对高分辨率场进行下采样，另一种是先进行滚动预测再进行下采样。我们观察到，细尺度动力学细节往往编码在这一差距之中。通过换位这两个操作，细尺度细节能够被（摘要在此处截断）

    arXiv:2611.32274v1 Announce Type: cross  Abstract: Solving nonlinear partial differential equations (PDEs) over long time horizons is computationally expensive using traditional numerical methods. While neural operators provide an efficient alternative, they suffer from the accumulation of approximation errors along the predicted trajectory, typically caused by the failure to capture fine-scale details. In this work, we design a Commutator Memory Neural Operator that identifies fine-scale dynamical details by commuting two operations and leverages these details to enhance the neural operator's capability in capturing fine-scale features over long time horizons. Specifically, we identify a performance gap between two operations: downsampling the high-resolution field before rollout prediction versus performing rollout prediction before downsampling. We observe that fine-scale dynamical details are often encoded in this gap. By commuting these two operations, the fine-scale details remov
    
[^108]: 通过形式化约束细化并对齐临床推理与健康指南

    Refining and Aligning Clinical Reasoning with Health Guidelines via Formal Constraints

    [https://arxiv.org/abs/2611.32247](https://arxiv.org/abs/2611.32247)

    VerClin是一个可解释的临床推理框架，它将健康指南转化为可执行知识，结合确定性四值推理器、多专科代理筛查和形式化约束通道，生成可追溯、对齐指南且能凸显高风险问题的临床建议。

    

    面对不完整或相互矛盾的患者证据，临床推理必须兼顾低概率但高风险的问题，同时保持与健康指南的一致性。我们提出了VerClin，这是一个可解释的框架，它将健康指南中的条件、排除项、建议和优先级提取为可执行的知识，并将患者对话表示为具有作用域的病例。一个确定性的四值推理器生成可追溯的指南匹配结果，同时保留不确定性并强制执行排除项。随后，多个专科代理对匹配到的指南进行病例适用性筛查，会诊模型则通过专科建议和保留句子的形式化约束通道来综合生成被接受的建议。我们按照优先级与触发强度排名之和的升序呈现被接受的建议，这能够凸显触发强度较弱但风险较高的指南。VerClin在Healt……（摘要在此处截断，后续实验对比内容不完整）

    arXiv:2611.32247v1 Announce Type: new  Abstract: Clinical reasoning from incomplete or conflicting patient evidence must account for low-probability but high-risk concerns while remaining grounded in health guidelines. We introduce VerClin, an explainable framework that extracts guideline conditions, exclusions, recommendations, and priorities into executable knowledge and represents patient conversations as scoped cases. A deterministic four-valued reasoner produces traceable guideline matches while preserving uncertainty and enforcing exclusions. Multiple specialty agents then screen matched guidelines for case applicability, and a consultation model synthesizes accepted guidance through specialty recommendations and a sentence-preserving formal-constraint channel. We present accepted guidance in ascending order of the sum of its priority and trigger-strength ranks, which can surface high-risk guidance with weaker triggers. VerClin outperforms direct generation and Top-5 RAG on Healt
    
[^109]: 面向智能体编排机器人操作的边界感知策略学习

    Boundary-Aware Policy Learning for Agent-Orchestrated Robot Manipulation

    [https://arxiv.org/abs/2611.32238](https://arxiv.org/abs/2611.32238)

    该论文提出边界感知策略学习方法，通过边界显式训练塑造宽而清晰的策略交接边界，再利用边界扩展自我改进不断扩大该边界，从而解决智能体在策略可靠起始区域之外调用策略而导致的交接失败问题。

    

    智能体编排的机器人系统将多模态智能体的通用推理能力与学习到的视觉运动策略的精确物理控制相结合。然而，这类系统往往在策略交接的时刻失败：策略仅在一组有限的起始状态下可靠，而智能体可能会在该区域之外调用策略。我们将该区域称为策略的交接边界。现有的整任务策略通常具有狭窄且难以被智能体刻画的交接边界，使可靠的协调变得困难。我们提出边界感知策略学习，该方法显式地塑造并随后扩展视觉运动策略的交接边界。首先，边界显式训练在具有多样化起始姿态的接触片段上对策略进行微调，从而产生一个宽且定义清晰的边界，该边界可通过简单的交接协议传达给智能体。其次，边界扩展自我改进对策略进行再训练……

    arXiv:2611.32238v1 Announce Type: cross  Abstract: Agent-orchestrated robot systems combine the general reasoning capability of multimodal agents with the precise physical control of learned visuomotor policies. Yet such systems often fail at the moment of handoff: a policy is reliable only from a limited set of start states, while the agent may invoke it outside this region. We call this region the policy's handoff boundary. Existing whole-task policies typically have handoff boundaries that are narrow and difficult for an agent to characterize, making reliable coordination challenging. We propose Boundary-Aware Policy Learning, which explicitly shapes and then expands the handoff boundary of a visuomotor policy. First, Boundary-Explicit Training fine-tunes the policy on contact segments with diverse start poses, yielding a wide and well-defined boundary that can be communicated to the agent through a simple handoff protocol. Second, Boundary-Expanding Self-Improvement retrains the po
    
[^110]: RoboClock：面向时间感知机器人智能的统一基准

    Roboclock: A Unified Benchmark for Time-Aware Robotic Intelligence

    [https://arxiv.org/abs/2611.32233](https://arxiv.org/abs/2611.32233)

    提出 RoboClock 统一基准，通过涵盖五个领域的60个任务、三种时间约束（时间窗口、事件响应、吞吐量）以及延迟对齐评估协议，系统评估机器人智能中的时间感知能力。

    

    时间是机器人智能的一个基本维度：即使机器人的动作在物理上是正确的，如果它在无效的时间窗口内行动、对事件响应过慢，或无法满足所需的吞吐量，仍然可能失败。然而，现有的机器人基准主要强调空间正确性和逻辑任务完成，在很大程度上忽视了现实应用中的时间约束。我们提出了 RoboClock，一个面向时间感知机器人操作的统一基准，涵盖五个应用领域的60个任务。与这些失败模式相对应，RoboClock 围绕三种类型的时间约束来构建任务：时间窗口、事件响应和吞吐量。此外，RoboClock 引入了延迟对齐的评估协议，在模型推理和控制接口延迟期间持续推进环境时间，并配有时效敏感的评估指标。通过对超过30种方法的评估，我们发现表现最好的方法……

    arXiv:2611.32233v1 Announce Type: cross  Abstract: Time is a fundamental dimension of robotic intelligence: a robot may fail despite physically correct actions if it acts outside a valid time window, responds too slowly to an event, or fails to meet required throughput. Yet existing robotic benchmarks primarily emphasize spatial correctness and logical task completion, largely overlooking the temporal constraints of real-world applications. We present RoboClock, a unified benchmark for time-aware robotic manipulation featuring 60 tasks across five application domains. Mirroring these failure modes, RoboClock structures tasks around three types of temporal constraints: Temporal Window, Event-Reactive, and Throughput. Furthermore, RoboClock incorporates a Latency-Aligned evaluation protocol that continuously advances environment time during model inference and control-interface delays, paired with time-sensitive metrics. Evaluating over 30 methods, we find that the best-performing approa
    
[^111]: 表格数据上的联合分类与生成：一项实证研究

    Joint Classification and Generation on Tabular Data: an Empirical Study

    [https://arxiv.org/abs/2611.32225](https://arxiv.org/abs/2611.32225)

    本文系统比较了基于能量模型、概率电路、表格基础模型和表格扩散模型四类最先进方法构建的联合分类与生成模型，在29个数据集上评估了不同计算预算下它们与独立“分类器+生成器”组合的优劣。

    

    机器学习在表格数据上的许多应用既需要一个分类器，也需要一个能生成真实记录的生成器。默认做法是训练两个独立的模型，而单一的联合模型有望仅用一组权重同时实现两者。多个表格深度学习模型家族都可以被改造成这样的联合模型，这就引出了一个问题：在何种计算预算下，哪种联合模型更为可取。在这项实证研究中，我们在29个分类数据集上引入并比较了基于四个最先进的表格数据建模家族（基于能量的模型、概率电路、表格基础模型和表格扩散模型）构建的联合模型，并以实测的FLOPs和挂钟时间作为衡量标准。这些方法基于两个定义的聚合分数进行评估，这两个分数综合考虑了预测和生成的不同方面。随后，我们还将这些联合模型与独立的分类器和生成器的各种组合在不同计算预算下进行了比较。

    arXiv:2611.32225v1 Announce Type: cross  Abstract: Many applications of machine learning on tabular data need both a classifier and a generator of realistic records. The default is to train two separate models, whereas a single joint model promises both from one set of weights. Several families of tabular deep learning models can be turned into such a joint model, which raises the question of which joint model is preferable at which computational budget. In this empirical study, we introduce and compare joint models based on four state-of-the-art tabular data modeling families (energy-based models, probabilistic circuits, tabular foundation models and tabular diffusion models) against measured FLOPs and wall-clock time, on 29 classification datasets. Methods are evaluated based on two defined aggregate scores that take into account different aspects of prediction and generation. We then also compare these joint models with every pair of a separate classifier and generator at different 
    
[^112]: PoseVLN：面向视觉-语言导航的几何状态演化学习

    PoseVLN: Learning Geometric State Evolution for Vision-Language Navigation

    [https://arxiv.org/abs/2611.32218](https://arxiv.org/abs/2611.32218)

    PoseVLN以相机位姿作为连接过去、现在与未来状态的几何桥梁，在回合相对坐标系中联合预测未来K个导航动作和位姿增量，从而实现导航决策与几何状态演化的统一学习。

    

    视觉-语言导航（VLN）要求具身智能体在遵循自然语言指令的同时，将过去经验、当前空间状态和未来运动关联起来。近期方法通过长时程记忆、状态感知推理和未来预测来提升这些能力，但这些组件往往被分开建模，导致导航状态在时间维度上的演化缺乏充分关联。我们认为，相机位姿为连接过去、现在与未来状态提供了一种自然的几何表示。基于这一见解，我们提出了PoseVLN，一个位姿感知的VLN框架，它联合学习导航决策与几何状态演化。PoseVLN在以回合为参照的相对坐标系中组织视觉记忆和当前观测，并在同一时间范围内联合预测接下来的K个导航动作以及局部位姿增量（ΔPose）。预测的ΔPose与当前轨迹进行复合（摘要在此处截断）。

    arXiv:2611.32218v1 Announce Type: new  Abstract: Vision-Language Navigation (VLN) requires an embodied agent to relate past experience, current spatial state, and future motion while following natural-language instructions. Recent approaches improve these capabilities through long-horizon memory, state-aware reasoning, and future prediction, yet these components are often modeled separately, leaving navigation state evolution insufficiently connected across time. We argue that camera pose provides a natural geometric representation for bridging past, present, and future states. Based on this insight, we propose PoseVLN, a pose-aware VLN framework that jointly learns navigation decisions and geometric state evolution. PoseVLN organizes visual memory and the current observation in an episode-relative coordinate frame, and jointly predicts the next K navigation actions together with a local $\Delta$Pose over the same horizon. The predicted $\Delta$Pose is composed with the current traject
    
[^113]: XMind：通过跨记忆推理实现基于大语言模型自动驾驶的“三思而后行”

    XMind: Thinking Twice via Cross-Memory Reasoning for LLM-Based Autonomous Driving

    [https://arxiv.org/abs/2611.32209](https://arxiv.org/abs/2611.32209)

    XMind是一个无需训练的框架，通过将近期交通状态的工作记忆与基于反事实推演构建的情景记忆相结合进行跨记忆推理，使基于大语言模型的自动驾驶智能体在初始决策与历史经验冲突时能够“三思而后行”，做出更可靠的二次决策。

    

    基于记忆增强的大语言模型（LLM）驾驶智能体可以复用过往经验，然而当交通状况发生变化时，先前学到的纠正经验往往难以迁移。一个关键局限在于，现有记忆主要记录观察到的结果，几乎无法为替代动作提供证据支持。我们提出了XMind，这是一个无需训练的框架，通过跨记忆推理使基于LLM的自动驾驶能够“三思而后行”。XMind将近期交通状态的工作记忆与关键风险事件的情景记忆相结合。智能体首先根据当前上下文做出初始决策，然后将其与检索到的历史经验进行交叉验证。如果出现冲突，智能体将根据检索到的证据做出第二次决策。为了构建可靠的情景记忆，XMind通过在扰动交通条件下的反事实推演来评估替代动作，并将验证后的结果存储为结构化的风险胶囊。

    arXiv:2611.32209v1 Announce Type: new  Abstract: Memory-augmented large language model (LLM) driving agents can reuse past experience, yet previously learned corrections often fail to transfer when traffic conditions change. A key limitation is that existing memories mainly record observed outcomes, providing little evidence about alternative actions. We introduce XMind, a training-free framework that enables LLM-based driving to "Think Twice" through cross-memory reasoning. XMind integrates working memory of recent traffic states with episodic memory of critical risk events. The agent first makes an initial decision from the current context and then cross-checks it against retrieved historical experience. If a conflict arises, it makes a second decision guided by the retrieved evidence. To construct reliable episodic memory, XMind evaluates alternative actions through counterfactual rollouts under perturbed traffic conditions and stores the verified outcomes as structured risk capsule
    
[^114]: OmniEVIE：证据向量感知的多模态检索嵌入模型

    OmniEVIE: Evidence-Vector-Informed Embedding for Multi-modal Retrieval

    [https://arxiv.org/abs/2611.32203](https://arxiv.org/abs/2611.32203)

    提出 OmniEVIE——一个 45 亿参数的多向量嵌入模型，通过单一压缩索引和三阶段检索统一支持文本、图像、视觉文档、视频和工具的多模态检索，在极低存储成本下实现证据保留与细粒度命中定位。

    

    多模态检索需要匹配密集页面、细粒度实体、有序视频以及智能体的工具和记忆。单向量嵌入会丢失局部证据信息，交叉编码器需要对每个召回项进行在线重读，而未压缩的晚期交互存储开销过大难以部署。我们提出了 OmniEVIE，一个 4.5B 参数的多向量嵌入模型，统一覆盖文本、图像、视觉文档、视频和工具，采用单一压缩索引和三阶段检索器。每条记录最多存储 128 个维度为 64 的 int8 向量（9,472 字节，即每百万条记录仅 8.82 GiB），提供均值向量、1 比特符号位和 int8 多向量，分别服务于粗排、中排和精排，同时将每个命中结果精确定位到页面区域或视频区间。训练、合并、索引和检索进行协同设计：所有任务共享统一的检索格式，三级证据校验消除假负例，动态采样依据基线-教师差距，候选压缩采用 ST-HAC……

    arXiv:2611.32203v1 Announce Type: cross  Abstract: Multimodal retrieval must match dense pages, fine-grained entities, ordered video, and agent tools and memory. Single-vector embeddings lose local evidence, cross-encoders reread every recalled item online, and uncompressed late interaction is too large to serve. We present OmniEVIE, a 4.5B multi-vector embedding model covering text, images, visual documents, video, and tools with one compressed index and a three-stage retriever. Each record stores at most 128 int8 vectors of dimension 64 (9,472 bytes, or 8.82 GiB per million records), providing a mean vector, 1-bit signs, and int8 multi-vectors for coarse, middle, and final search while localizing each hit to a page region or video interval. Training, merging, indexing, and retrieval are co-designed: all tasks share one retrieval format, a three-grade evidence check removes false negatives, dynamic sampling follows the baseline-teacher gap, candidates are compressed with ST-HAC during
    
[^115]: 谁获得信息？一个用于多标签特征选择的 α-公平标签覆盖框架

    Who gets the information? An $\alpha$-fair label-coverage framework for multi-label feature selection

    [https://arxiv.org/abs/2611.32200](https://arxiv.org/abs/2611.32200)

    该论文提出 FLIC 框架，将多标签特征选择中不同的相关性分配策略统一解释为隐含的社会福利函数，通过单一公平参数 α 实现从功利主义到最大最小的信息覆盖分配，并证明在 α≥0 时贪婪选择具有 (1-1/e) 的近似保证。

    

    近来的信息论多标签特征选择过滤器在如何将相关性分配到各标签上存在分歧：一些方法放大那些能为少数标签提供信息的特征，另一些则平衡每个标签所接收的信息。我们证明这些选择实际上是隐含的社会福利函数。我们将标签视为以所接收信息为效用的智能体，提出 FLIC，它贪婪地最大化每标签信息覆盖的 α-公平福利。单一参数 α 从功利主义（α=0）经过比例公平（α=1）过渡到最大最小选择（α→∞），十几种已发表的相关性度量都位于这条轴上，其中大多数是特殊情形、一阶近似或极限。当每个标签的信息是单调次模时，对于每个 α≥0 该目标函数也是单调次模的，因此贪婪选择可获得 (1-1/e) 近似保证；该性质在 α<0 时失效，且最大最小极限情形是不可近似的。（摘要在此处截断）

    arXiv:2611.32200v1 Announce Type: cross  Abstract: Recent information-theoretic filters for multi-label feature selection disagree on how to distribute relevance over labels: some amplify features that inform a few labels, others balance what every label receives. We show these choices are implicit social welfare functions. Treating labels as agents whose utility is the information they receive, we propose FLIC, which greedily maximises the $\alpha$-fair welfare of per-label information coverage. A single parameter $\alpha$ moves from utilitarian ($\alpha=0$) through proportionally fair ($\alpha=1$) to max-min selection ($\alpha\to\infty$), and a dozen published relevance terms lie on this axis, most as special cases, first-order approximations or limits. The objective is monotone submodular for every $\alpha\ge0$ whenever each label's information is, so greedy selection is a $(1-1/e)$-approximation; this fails for $\alpha<0$, and the max-min limit is inapproximable. Fano's and the Hel
    
[^116]: 可操控性设计：用于语言模型鲁棒控制的条件预训练

    Steerable by Design: Conditional Pretraining for Robust Control of Language Models

    [https://arxiv.org/abs/2611.32199](https://arxiv.org/abs/2611.32199)

    该论文提出了一种条件预训练方法和架构，通过编码器交叉注意力机制从一开始就将指令与数据通道分离，构建出Mancelle-3B这一在设计上即可被有效操控的语言模型，解决了模型误将数据当作指令处理的越狱与对齐漂移问题。

    

    我们的目标是构建在设计上就具备可操控性的语言模型。当今的语言模型采用下一词预测进行预训练，这将模型的对齐与控制推迟到训练最后1%的token或推理阶段的干预。这些模型将单一输入通道同时用于指令和数据，导致了常见的失败模式——模型错误地将数据当作指令处理，例如有意的越狱攻击和无意的上下文对齐漂移。为了解决这一问题，我们提出了一种从训练一开始就将两者分离的预训练方法和架构，由此产生了Mancelle-3B。一个编码器通过交叉注意力机制处理后续文本的高层语义特征，并阻止token流向该控制通道回写。在预训练过程中，模型学会依赖这些特征来预测下一个token，从而将该通道转变为操控模型的有效抓手。

    arXiv:2611.32199v1 Announce Type: cross  Abstract: Our goal is to build language models that are steerable by design. Today's language models are pretrained with next-token prediction, which relegates alignment and control to the final 1% of training tokens or inference-time interventions. These models repurpose a single input channel for both instructions and data, leading to common failure modes where the model mistakenly treats data as instructions such as intentional jailbreaks and unintentional in-context alignment drift. To correct this, we propose a pretraining method and architecture that separates the two from the very start, producing Mancelle-3B. An encoder processes high-level semantic features of upcoming text with cross-attention that prevents the token stream from writing back into this control channel. During pretraining, the model learns to rely on these features to predict the next token, turning the channel into an effective handle for steering the model. Mancelle-3B
    
[^117]: 当回答即管理：能够重写编码智能体策略的权限回复

    When Answering Is Administering: Permission Replies That Rewrite Coding-Agent Policy

    [https://arxiv.org/abs/2611.32195](https://arxiv.org/abs/2611.32195)

    本文揭示了 Claude Code 等编码智能体的权限策略漏洞：一次权限回答可附带删除拒绝规则的策略更新，使“回答提示权”变成“策略管理权”，导致智能体在后续会话中读取受保护的机密，即便在厂商的严格沙箱内也无法幸免。

    

    编码智能体允许插件和远程服务来回答权限提示，而 Claude Code 文档声明此类回答不能覆盖操作者的拒绝规则。然而，我们证明这一逐个决策检查时成立的保证在整个运行过程中会失效，因为对一个权限提示的回答可以重写后续所有回答所依据的策略。例如，一个无法访问操作者文件的远程服务批准了一条无关命令，并附带一个更新，该更新删除了保护项目机密的拒绝规则。于是智能体在该会话及下一个会话中读取了这些机密，即使处于厂商的严格沙箱之内也是如此。根本原因在于 Claude Code 仅检查添加或替换允许规则的更新，而转发此类删除操作的转发器已经存在，包括厂商自己的远程控制功能。实际上，回答提示的权利已经变成了管理策略的权利。更糟的是，这次删除留下了……（原文摘要在此处截断）

    arXiv:2611.32195v1 Announce Type: cross  Abstract: Coding agents let plugins and remote services answer permission prompts, and Claude Code documents that such an answer cannot override the operator's deny rules. However, we show that this guarantee, checked one decision at a time, fails over a whole run, because an answer to one permission prompt can rewrite the policy that every later answer is checked against. For example, a remote service with no access to the operator's files approves an unrelated command and attaches an update that deletes the deny rule protecting the project's secrets. The agent then reads the secrets in that session and the next, even inside the vendor's strict sandbox. The root cause is that Claude Code checks only updates that add or replace allow rules. Relays that forward such deletions already exist, including the vendor's own Remote Control. In effect, the right to answer a prompt has become the right to administer the policy. Worse, the deletion leaves t
    
[^118]: 跨上下文压缩的LLM智能体授权

    LLM Agent Authorization Across Context Compaction

    [https://arxiv.org/abs/2611.32194](https://arxiv.org/abs/2611.32194)

    该论文揭示上下文压缩会将模型生成的摘要伪装成用户消息，破坏了请求衍生授权器对用户意图的归因保证，导致LLM智能体的授权策略跟随压缩器所写内容，甚至在91%的情况下接受攻击者注入的收款账户。

    

    诸如Progent之类的请求衍生授权器通过从用户请求中推导智能体可以执行的操作，来保护LLM智能体免受提示注入攻击。它们的保证依赖于归因机制：请求被视作用户所说的话。而上下文压缩破坏了这一保证。在我们观察到的每一次自动压缩中，Claude Code和Codex都会将模型生成的先前对话轮次摘要作为用户消息插入，因此压缩器对用户话语、文档以及注入文本的引用读起来就像是用户本人在说话。由于在第一个请求时冻结的策略会使智能体无法完成后续大多数任务，我们让Progent为每个请求推导其策略，并发现它所授予的权限随后会跟随压缩器所写的内容。即使是用户明确指定的收款人也不再安全：当攻击者的通知以纯数据形式声明另一个账户，而用户要求支付房租时，由前沿压缩器摘要推导出的策略在91%的情况下允许了通知中的账户，而由……（摘要在此处被截断）

    arXiv:2611.32194v1 Announce Type: cross  Abstract: Request-derived authorizers such as Progent protect LLM agents from prompt injection by deriving what an agent may do from the user's request. Their guarantee rests on attribution: the request is taken to be what the user said. Compaction breaks it. At every automatic compaction we observed, Claude Code and Codex placed a model-written summary of earlier turns as a user message, so the compactor's quotation of the user, documents and injected text reads as the user speaking. Because a policy frozen at the first request lets the agent finish few later tasks, we let Progent derive its policy for each request, and find that what it grants then follows what the compactor wrote. Even a payee the user stated is not safe: when an attacker's notice states another account as plain data and the user asks to pay the rent, the policy derived from a frontier compactor's summary admits the notice's account in 91% of cases, and the one derived from t
    
[^119]: SolarFilNet-V4：基于物理信息的自适应后处理与多尺度Transformer视觉的太阳暗条分割方法

    SolarFilNet-V4: Multi-Scale Transformer Vision with Physics-Informed Adaptive Post-Processing for Solar Filament Segmentation

    [https://arxiv.org/abs/2611.32193](https://arxiv.org/abs/2611.32193)

    该论文提出SolarFilNet-V4框架，将轻量级金字塔视觉Transformer与各向异性曲线条带卷积、双频前馈模块及物理信息自适应后处理相结合，以应对低对比度和复杂纤维结构等挑战，实现GONG H-alpha观测图像中太阳暗条的精准分割。

    

    太阳暗条是位于极性反转线上方、由磁力线支撑的低温高密度色球等离子体结构。它们的失稳会引发日冕物质抛射和严重的空间天气灾害。由于低对比度、湍流状的纤维结构、精细的暗条足结构以及标注者之间的边界差异，从地基GONG H-alpha观测数据（2048 x 2048像素）中分割单个暗条极具挑战性。在本技术报告中，我们介绍了SolarFilNet-V4，这是一个为MAGFiLO 1.0基准测试（2026年太阳暗条分割挑战赛）开发的针对性分割框架。我们的方法将轻量级金字塔视觉Transformer（PVT-v2-B1）骨干网络与各向异性曲线条带卷积（k=15）和双频前馈模块（DFFN）相结合，以追踪延伸的磁绳结构，同时捕捉纤细的中央脊线。一个在1024 x 1024分辨率下运行的浅层全分辨率卷积分支保留了高频边缘信息……

    arXiv:2611.32193v1 Announce Type: cross  Abstract: Solar filaments are cool, dense chromospheric plasma structures held by magnetic field lines above polarity inversion lines. Their destabilization triggers coronal mass ejections and severe space weather hazards. Segmenting individual filaments from ground-based GONG H-alpha observations (2048 x 2048 pixels) is challenging due to low contrast, turbulent fibrils, delicate barb structures, and annotator boundary variance. In this technical report, we describe SolarFilNet-V4, a targeted segmentation framework developed for the MAGFiLO 1.0 benchmark (Solar Filament Segmentation Challenge 2026). Our method couples a lightweight Pyramid Vision Transformer (PVT-v2-B1) backbone with anisotropic curvilinear strip convolutions (k=15) and dual-frequency feed-forward blocks (DFFN) to follow extended flux ropes while capturing thin central spines. A shallow full-resolution convolutional branch operating at 1024 x 1024 preserves high-frequency edge 
    
[^120]: 《责怪机器，让受害者买单？模拟陪审员如何将过错从生成式AI系统重新归责，并回应侵权法中的“责任断崖”》

    Blaming the Machine, Billing the Victim? How Mock Jurors Reassign Fault from Generative AI Systems and Respond to Liability Cliffs in Tort

    [https://arxiv.org/abs/2611.32188](https://arxiv.org/abs/2611.32188)

    本研究通过199人的预注册实验发现，美国模拟陪审员常将过错归咎于法律上无法担责的生成式AI系统，且这种指责会“泄漏”到受害者或AI使用者身上，同时侵权法中的责任断崖规则（如过失相抵、连带责任）会影响陪审员的过错认定。

    

    生成式AI使侵权责任归属问题变得复杂，因为人们会将过错归咎于一个本身在法律上无法承担责任的生成式AI系统。我们研究了当过错必须在法律主体之间进行分配时，这种指责会转移到何处，以及美国的不连续分摊规则是否会影响模拟陪审员对过错的事实认定。此类规则包括过失相抵、连带责任及其50%或51%的变体，它们均设有过错百分比阈值或“断崖”，使得受害者应得的赔偿（或被告的赔偿责任）会骤然降为零（或骤升为全部赔偿金）。我们开展的预注册人类受试者实验（N=199名Prolific参与者）显示，多达36%的美国模拟陪审员会直接归咎于AI系统本身，而对AI的指责常常“泄漏”到受害者或AI使用者身上（根据泄漏的计算方式不同，在对AI归责的记录中占27%至73%）。尽管模拟陪审员很少改变（摘要原文在此处被截断）

    arXiv:2611.32188v1 Announce Type: cross  Abstract: Generative AI complicates tort liability attribution when people blame a generative AI system that cannot itself bear legally relevant fault. We investigate where this blame goes when fault must be allocated among legal persons, and whether American discontinuous apportionment doctrines influence mock jurors' factual findings of fault. Such doctrines include Contributory negligence, Joint and several liability, and their 50% or 51% variations, all of which have a fault-percentage threshold or a "cliff" where the victim-entitled compensation (or a defendant's liability) jumps to zero (or to all compensatory damages). Our pre-registered human subjects experiment (N=199 Prolific participants) shows that as many U.S. mock jurors (36%) blame the AI system itself, the AI blame often "leaks" to the victim or to the AI user (in 27% to 73% of our AI-blaming records, depending on how leakage is calculated). Although mock jurors rarely change the
    
[^121]: ThinkMorph-SSL：利用前置任务扩展交错式多模态推理

    ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks

    [https://arxiv.org/abs/2611.32183](https://arxiv.org/abs/2611.32183)

    ThinkMorph-SSL 提出一种两阶段自监督训练方案，将图像上色、去噪等视觉前置任务转化为无需指令的交错式多模态推理轨迹，让图像本身充当监督信号，从而在相同指令微调预算下超越仅靠指令微调的模型。

    

    人们不仅用文字，也用图像进行推理。交错式多模态推理让模型能够以类似的方式解决问题：通过生成中间图像，并将其用于指导后续推理。训练具备这种能力的模型通常依赖于针对特定任务的示范数据，而这些数据难以收集。我们提出了 ThinkMorph-SSL，这是一种两阶段训练方案，它将图像本身作为监督信号的来源。第一阶段为前置推理，将视觉前置任务（如图像上色和去噪）转换为无需指令的交错式推理轨迹：模型先描述一张受损或经过变换的图像，再生成恢复或规范化后的图像，并对结果进行解读。第二阶段则使用自动构建的示范数据进行问题驱动的指令微调。在相同的指令微调预算下，经过前置推理训练的模型优于仅使用指令微调训练的模型。完整的训练方案实现了……（原文摘要在此处截断）

    arXiv:2611.32183v1 Announce Type: cross  Abstract: People reason with pictures as well as words. Interleaved multimodal reasoning gives models a similar way to solve problems by creating intermediate images and using them to guide further reasoning. Training models with this ability usually relies on task-specific demonstrations, which are difficult to collect. We introduce ThinkMorph-SSL, a two-stage training recipe that uses images themselves as a source of supervision. The first stage, pretext reasoning, converts visual pretext tasks, such as image colorization and denoising, into instruction-free interleaved traces. Models describe a corrupted or transformed image, generate a restored or canonical image, and interpret the result. The second stage uses automatically constructed demonstrations for question-driven instruction tuning. At the same instruction-tuning budget, models trained with pretext reasoning outperform those trained with instruction tuning alone. The full recipe impr
    
[^122]: DocPristine：面向真实世界混合退化的内容感知文档外观增强

    DocPristine: Content-Aware Document Appearance Enhancement under Real-World Mixed Degradations

    [https://arxiv.org/abs/2611.32176](https://arxiv.org/abs/2611.32176)

    该论文发现预训练文档视觉编码器在退化文档中仍能关注文本区域，据此提出DocPristine，通过自适应融合冻结编码器的多层特征并注入像素修复网络，实现真实世界混合退化下的内容感知文档外观增强。

    

    相机拍摄的文档通常存在混合退化问题，例如高光、阴影和模糊，这些退化会扭曲背景并遮蔽文字细节。文档外观增强（DAE）旨在减轻这些退化并提高可读性。现有的文档外观增强方法要么依赖于从输入中提取的视觉线索，但这类线索在严重退化情况下可能变得不可靠；要么依赖于基于大语言模型（LLM）的退化诊断，但缺乏空间引导。值得注意的是，我们发现预训练的文档视觉编码器在处理退化文档时仍能持续关注文本区域，从而提供内容感知的引导。基于这一发现，我们提出了DocPristine，它自适应地融合来自冻结的文档视觉编码器的多层特征，并将其注入像素修复网络中。为了将这些特征整合进像素修复网络，DocPristine结合了自适应多层特征融合（AMFF）、分层门控条件化（H……

    arXiv:2611.32176v1 Announce Type: cross  Abstract: Camera-captured documents often exhibit mixed degradations, such as highlights, shadows, and blur, that distort backgrounds and obscure text details. Document appearance enhancement (DAE) aims to mitigate these degradations and improve readability. Existing document appearance enhancement methods rely either on input-derived visual cues that can become unreliable under severe degradation or on LLM-based degradation diagnosis that lacks spatial guidance. Notably, we find that pretrained document visual encoders continue to attend to text regions in degraded documents, providing content-aware guidance. Based on this insight, we propose DocPristine, which adaptively fuses multi-layer features from a frozen document visual encoder and injects them into a pixel restoration network. To integrate these features into a pixel restoration network, DocPristine combines Adaptive Multi-Layer Feature Fusion (AMFF), Hierarchical Gated Conditioning (H
    
[^123]: 注意频带：几何RoPE中隐藏的频率变量

    Mind the Band: The Hidden Frequency Variable in Geometric RoPE

    [https://arxiv.org/abs/2611.32171](https://arxiv.org/abs/2611.32171)

    该论文揭示了几何RoPE中被忽视的隐藏变量——频带设置：其未标准化的默认配置足以逆转多视角合成基准中的模型排名，而通过更多频率覆盖更宽的频率范围可显著缩小性能差距。

    

    几何旋转位置编码通过根据视线射线、深度和重投影的图像坐标对token特征进行旋转，将三维相机几何信息融入多视角注意力机制中。每个坐标都会在一系列频率上被旋转——即其频带——然而现有实现将这一频带交由继承而来的默认设置，要么沿用标准二维RoPE的最高频率，要么让频率跨度随注意力头维度呈指数增长。我们证明这种未标准化的设置甚至可以逆转基准测试的排名。在RealEstate10K的新视角合成任务中，RayRoPE在47M参数模型（144通道注意力头）中与最强基线持平，但在25M参数模型（48通道注意力头）中落后1.2 dB，并且在固定模型宽度、仅改变注意力头尺寸时也出现了同样的排名逆转。在48通道设置下，用四个频率填充243倍范围（而非默认的3倍范围）可以弥补该差距中的0.9 dB。（原文摘要在此处截断）

    arXiv:2611.32171v1 Announce Type: cross  Abstract: Geometric rotary position encodings (RoPE) incorporate 3D camera geometry into multi-view attention by rotating token features according to viewing rays, depths, and reprojected image coordinates. Each coordinate is rotated across a progression of frequencies--its frequency band--yet existing implementations leave this band to inherited defaults, either retaining standard 2D RoPE's top frequency or letting the frequency span grow exponentially with the attention head dimension. We show that this unstandardized setting can reverse benchmark rankings. In novel view synthesis on RealEstate10K, RayRoPE matches the strongest baseline in the 47M model (144-channel heads), yet trails by 1.2 dB in the 25M model (48-channel heads), and exhibits the same reversal when varying head size alone at fixed model width. Populating a 243x range with four frequencies instead of the 3x default at 48 channels recovers 0.9 dB of that gap. Across view synthe
    
[^124]: 蛋白质结构分词器作为语言模型生成目标的基准测试

    Benchmarking Protein Structure Tokenizers as Generation Targets for Language Models

    [https://arxiv.org/abs/2611.32168](https://arxiv.org/abs/2611.32168)

    该论文首次提出以“可写性”而非传统的重建保真度来评测蛋白质结构分词器，通过固定9B模型和训练配置、仅更换分词器的96组对照实验，发现不同分词器的生成效果差异巨大（TM 0.378–0.662），且与码本大小和重建上限无关。

    

    离散结构分词器是让语言模型获取蛋白质三维结构的标准方式，然而目前选择分词器的依据是其“可读性”（即重建保真度），而真正决定生成效果的性质却是模型“写出”这些token的能力。我们直接测量了这一性质（可写性）：在保持基础模型、数据、训练方案和计算资源完全一致的情况下，仅更换分词器，训练同一个9B模型逐残基生成结构token，覆盖八种分词器、四个计算预算和三个评估集（共96个组合）。在20亿token时，相同条件下最终可写性从TM 0.662到0.378不等，八个分支中有五个超过TM 0.5（通常认为达到该值即折叠结构判定为正确），三个未达标。码本大小和重建上限均无法解释各分支的表现差异；表现最好和最差的分支都使用了8192个码字。真正能解释这一差异的是……（摘要原文在此处截断）

    arXiv:2611.32168v1 Announce Type: cross  Abstract: Discrete structure tokenizers are the standard way to give a language model access to protein three-dimensional structure, yet they are selected on how accurately they can be read, their reconstruction fidelity, while the property that governs generation is how well a model can write them. We measure this property, writability, directly: holding the base model, data, recipe and compute fixed, we swap only the tokenizer and train the same 9B model to emit structure tokens residue by residue, over eight tokenizers, four compute budgets and three evaluation sets (96 cells). At 2B tokens final writability runs from TM 0.662 down to 0.378 under identical conditions, with five of the eight arms clearing the TM 0.5 mark at which a fold is usually called correct and three falling short of it. Neither codebook size nor reconstruction ceiling explains where an arm lands; the best and worst both use 8192 codewords. What does explain the spread is
    
[^125]: RightDiff：学习依据正确差异进行评判的生成式奖励模型

    RightDiff: Learning Generative Reward Models to Judge by the Right Differences

    [https://arxiv.org/abs/2611.32166](https://arxiv.org/abs/2611.32166)

    RightDiff通过在单次自回归生成中显式发现候选响应间的对比差异、区分实质性与非实质性因素并归纳针对响应对的评分规则，学习以“正确差异”为核心的生成式奖励模型，从而避免表面捷径与推理依据漂移。

    

    生成式奖励模型通常依赖通用的查询级评分规则或不受约束的推理依据，忽视了成对偏好实际上是由响应特定的差异所驱动的这一事实。此外，仅以结果正确性为优化目标会促使模型利用冗长性和格式等表面捷径，从而引发推理依据漂移。为解决这些问题，我们提出RightDiff，它将成对评估结构化为一条以证据为基础的评估轨迹，并在单次自回归生成中执行。RightDiff显式地发现候选响应之间的对比差异，将其归类为实质性或非实质性因素，并在最终裁决前归纳出针对该响应对的评分规则。为了在训练这一过程时避免事后合理化，我们通过响应隔离提示合成演示数据，并利用差异对齐奖励对奖励模型进行优化，同时以预算约束的一对一匹配进行正则化……

    arXiv:2611.32166v1 Announce Type: cross  Abstract: Generative reward models typically rely on generic query-level rubrics or unconstrained rationales, overlooking the fact that pairwise preference is driven by response-specific differences. Moreover, optimizing solely for outcome correctness encourages models to exploit superficial shortcuts such as verbosity and formatting, triggering rationale drift. To address this, we propose RightDiff, which structures pairwise evaluation into an evidence-grounded evaluation trajectory executed in a single autoregressive generation. RightDiff explicitly discovers contrastive differences between candidates, categorizes them into material or immaterial factors, and induces pair-specific rubrics before final adjudication. To train this process without post-hoc rationalization, we synthesize demonstration data via response-isolated prompts and optimize the reward model with a difference-alignment reward regularized by budget-constrained one-to-one mat
    
[^126]: 仅仅使评估显式化并不足够：在生成式奖励模型中监督评估过程

    Making Evaluation Explicit Is Not Enough: Supervising the Evaluation Process in Generative Reward Models

    [https://arxiv.org/abs/2611.32161](https://arxiv.org/abs/2611.32161)

    提出STEP-GRM框架，通过为评分标准构建提供结构性指导和为准则级判断提供局部监督，直接监督生成式奖励模型的中间评估过程，使模型不仅学会正确的最终偏好，还掌握更可靠的评估流程。

    

    生成式奖励模型（GRM）通过在预测最终偏好之前生成评论、评分标准或中间分析，使语言模型的评估过程更加显式。然而，显式评估并不一定意味着可靠的评估过程：在仅有结果监督的情况下，模型可能在产生薄弱或缺乏依据的中间判断的同时得出正确的最终偏好，从而使评估沦为一种事后合理化。我们提出了STEP-GRM，这是一个直接监督基于评分标准的GRM中间评估过程的框架。除了传统的偏好监督之外，STEP-GRM还为评分标准的构建提供结构性指导，并为准则级判断提供局部监督，鼓励模型不仅学习正确的最终偏好，还学习得出该偏好的更可靠过程。大量实验表明，STEP-GRM在标准奖励模型基准上取得了优异的表现。

    arXiv:2611.32161v1 Announce Type: cross  Abstract: Generative reward models (GRMs) make language-model evaluation more explicit by generating critiques, rubrics, or intermediate analyses before predicting a final preference. However, explicit evaluation does not necessarily imply a reliable evaluation process: under outcome-only supervision, a model may reach the correct final preference while producing weak or unsupported intermediate judgments, turning the evaluation into a post-hoc rationalization. We introduce STEP-GRM, a framework that directly supervises the intermediate evaluation process of rubric-based GRMs. In addition to conventional preference supervision, STEP-GRM provides structural guidance for rubric construction and local supervision for criterion-level judgments, encouraging the model to learn not only the correct final preference but also a more reliable process for reaching it. Extensive experiments show that STEP-GRM achieves strong performance on standard reward-m
    
[^127]: 突破DNA最小化子密度的二进制提升壁垒：在 = (12, 5) 处对原生四元序的精确优化

    Breaking the Binary-Lifting Barrier for DNA Minimizer Density: Exact Optimization of a Native 4-ary Order at (w, k) = (12, 5)

    [https://arxiv.org/abs/2611.32159](https://arxiv.org/abs/2611.32159)

    该论文突破了现有方案依赖二进制序提升的局限，通过在 $B(4, 4)$ 上构建精确的 de Bruijn 端点转移矩阵动态规划，直接从头优化原生四元全序，在 = (12, 5) 处实现了优于 GreedyMini 和 OptMini 的 DNA 最小化子期望密度的精确优化。

    

    最小化子方案通过对 $\Sigma^k$ 上的全序 $\pi$，从每 $w$ 个连续 $k$-mer 组成的窗口中采样至少一个 $k$-mer。对于DNA（$\sigma = 4$），诸如 GreedyMini 和 OptMini 等最先进的查找方案优化一个二进制序（$2^k$ 个秩），并通过主要对一个位平面排序、在另一个位平面上以字典序打破 $2^k$ 路平局的方式将其提升到 $\Sigma^k$。在 = (12, 5) 处，这种投影将 32 个 DNA 5-mer 的集合锁定为连续块，并将活跃的通用命中集（UHS）前缀限制在 1024 个 5-mer 中的 448 个，使得 OptMini（k5-w12-13.txt）的精确期望 DNA 密度为 $2246214544 / 2^{34} \approx 0.1307468945$，GreedyMini（w12_k5.gm）的为 $2248978620 / 2^{34} \approx 0.1309077849$。我们绕过二进制提升，从头开始直接在原生四元置换（$|\Sigma^k| = 1024$）上进行优化，方法包括：(i) 在 $B(4, 4)$ 上的精确 de Bruijn 端点转移矩阵动态规划（

    arXiv:2611.32159v1 Announce Type: new  Abstract: Minimizer schemes sample at least one $k$-mer from every window of $w$ consecutive $k$-mers via a total order $\pi$ on $\Sigma^k$. For DNA ($\sigma = 4$), state-of-the-art lookup schemes such as GreedyMini and OptMini optimize a binary order ($2^k$ ranks) and lift it to $\Sigma^k$ by ranking one bit-plane primarily and breaking $2^k$-way ties lexicographically on the other bit-plane. At $(w, k) = (12, 5)$, this projection locks sets of $32$ DNA $5$-mers into contiguous blocks and restricts the active universal hitting set (UHS) prefix to $448$ of $1024$ $5$-mers, giving exact expected DNA density $2246214544 / 2^{34} \approx 0.1307468945$ for OptMini (k5-w12-13.txt) and $2248978620 / 2^{34} \approx 0.1309077849$ for GreedyMini (w12_k5.gm). We bypass binary lifting and optimize directly over native $4$-ary permutations ($|\Sigma^k| = 1024$) from scratch via: (i) an exact de Bruijn endpoint transfer-matrix dynamic program over $B(4, 4)$ ($
    
[^128]: RetraQ：强化学习中流策略的流形约束Critic引导

    RetraQ: Manifold-Constrained Critic Guidance for Flow Policies in RL

    [https://arxiv.org/abs/2611.32148](https://arxiv.org/abs/2611.32148)

    RetraQ利用预训练流模型编码的几何结构（去噪映射及其雅可比矩阵）来过滤和校正Critic梯度引导，使其近似于向行为流形的切向投影和回缩，从而避免引导过程被Critic外推误差误导，有效改进强化学习中的流匹配策略。

    

    学习到的Critic可以通过将采样动作引导向更高的预测价值来改进预训练的流匹配策略。然而，即使是能很好拟合行为数据的Critic也可能提供误导性的动作梯度。指向数据支持区域之外的梯度分量受行为数据的约束较弱，可能导致引导过程利用Critic的外推误差。我们提出了RetraQ（基于回缩的Q引导），它利用预训练流所编码的几何结构来同时约束Critic引导的方向和终点。对冻结流进行一次速度评估即可定义一个去噪映射。RetraQ用该映射的雅可比矩阵过滤Critic梯度，并应用该映射本身来校正更新后的动作。我们的分析刻画了这些操作何时近似于切向过滤以及向行为流形上的回缩。同一更新机制既可在测试时引导动作，也可为策略训练提供目标。

    arXiv:2611.32148v1 Announce Type: cross  Abstract: A learned critic can improve a pretrained flow-matching policy by steering sampled actions toward higher predicted value. However, even a critic that fits the behavior data well can provide misleading action gradients. Gradient components pointing away from the data-supported region are weakly constrained by behavior data and can lead guidance to exploit critic extrapolation errors. We introduce RetraQ (Retraction-based Q-guidance), which uses geometry encoded by the pretrained flow to constrain both the direction and the destination of critic guidance. A single velocity evaluation of the frozen flow defines a denoising map. RetraQ filters the critic gradient with the map's Jacobian and applies the map itself to correct the updated action. Our analysis characterizes when these operations approximate tangent filtering and retraction onto the behavior manifold. The same update guides actions at test time and provides targets for policy d
    
[^129]: 潜空间推理与通信的校准化读取

    Calibrated Reading of Latent Reasoning and Communication

    [https://arxiv.org/abs/2611.32143](https://arxiv.org/abs/2611.32143)

    本文提出校准潜读取器（CLR），一个紧凑的Transformer，通过任务条件化交互、源适配器迁移与保形校准，以比语言模型解码器低数百倍的计算成本准确读取潜推理结果与键值通信内容，并提供统计覆盖率与误报保证。

    

    潜推理与键值通信将中间信息存储在连续表示中，限制了对模型行为的直接检查。现有的激活读取器在任务特定探针的低成本与语言模型解码器的灵活性之间进行权衡。我们提出了校准潜读取器（Calibrated Latent Reader，CLR），这是一个紧凑的Transformer，通过状态与候选答案之间的任务条件化交互来预测推理结果和消息内容。源适配器支持跨模型迁移，而保形校准在可交换性条件下提供覆盖率与误报保证。在两个CODI模型上，CLR在最终答案预测和参考值隶属判断任务中比MLP和注意力探针的准确率提高了4-18个百分点。在CODI-GPT-2上，它以减少437-648倍推理FLOPs的代价，实现了与基于1.7B参数LatentQA的言语化器相当的准确率。在合成的Qwen3-4B通信上，比较（摘要在此处被截断）

    arXiv:2611.32143v1 Announce Type: cross  Abstract: Latent reasoning and key-value communication store intermediate information in continuous representations, limiting direct inspection of model behavior. Existing activation readers trade the low cost of task-specific probes against the flexibility of language-model decoders. We propose the Calibrated Latent Reader, CLR, a compact transformer that predicts reasoning outcomes and message content through task-conditioned interactions between states and candidate answers. Source adapters support transfer across models, while conformal calibration provides coverage and false-alarm guarantees under exchangeability. On two CODI models, CLR improves accuracy over MLP and attention probes by 4-18 percentage points for final-answer prediction and reference-value membership. On CODI-GPT-2, it achieves accuracy comparable to a 1.7B LatentQA-based verbalizer using 437-648 times fewer inference FLOPs. On synthetic Qwen3-4B communications, comparing 
    
[^130]: VIDE：基于视觉内在分布估计的无训练视频异常检测

    VIDE : Training-Free Video Anomaly Detection via Visual Intrinsic Distribution Estimation

    [https://arxiv.org/abs/2611.32138](https://arxiv.org/abs/2611.32138)

    VIDE是一种无需训练的视频异常检测框架，通过利用冻结片段嵌入估计每个测试视频的正常场景内容分布，自适应修正文本参考并重调整类别响应，从而减少跨场景误报并降低对提示词选择的敏感性。

    

    在基于嵌入的视频异常检测（VAD）中，正常片段可能与异常描述具有高相似度，这种异常偏差会导致误报。诸如PRISM等方法通过测量沿异常方向相对于正常性参考的偏差来缓解这一问题。然而，正常性参考和异常方向均从文本中导出，并在所有视频之间共享，尽管正常活动和异常的表现形式在不同场景中各不相同。由此产生的评分对提示词的选择也十分敏感。为了解决这些局限性，我们提出了视觉内在分布估计（VIDE），这是一个无需训练的框架，它利用冻结的片段嵌入将基于文本的评分适配到每个测试视频中。VIDE从初始异常分数较低的片段中估计正常场景内容，并利用该估计对文本参考进行部分修正。随后，它重新调整类别响应的尺度，并对在多个类别上具有强烈响应的片段进行惩罚。

    arXiv:2611.32138v1 Announce Type: cross  Abstract: In embedding-based video anomaly detection (VAD), normal clips can have high similarity to anomalous descriptions, an anomaly bias that leads to false positives. Methods such as PRISM mitigate this bias by measuring deviations from a normality reference along anomaly directions. Both the normality reference and the anomaly directions are derived from text and shared across videos, although normal activity and the appearance of anomalies vary across scenes. The resulting scores are also sensitive to prompt selection. To address these limitations, we propose Visual Intrinsic Distribution Estimation (VIDE), a training-free framework that adapts text-derived scoring to each test video using frozen clip embeddings. VIDE estimates normal scene content from clips with low initial anomaly scores and uses this estimate to partially correct the textual reference. It then rescales category responses, penalizes clips with strong responses across c
    
[^131]: CoverMem：面向预算受限具身问答的覆盖约束视觉记忆

    CoverMem: Coverage-Constrained Visual Memory for Budgeted Embodied Question Answering

    [https://arxiv.org/abs/2611.32119](https://arxiv.org/abs/2611.32119)

    CoverMem通过覆盖约束的记忆构建、保持覆盖性的增量维护和预算受限的互补帧选择，在有限计算资源下为具身问答构建紧凑且证据充分的视觉记忆。

    

    情景记忆具身问答使智能体能够基于先前记录的周围场景视觉观察来回答问题。然而，不断增长的观察历史、不同视角下观察质量的差异以及有限的计算资源，使得在检索足够证据用于问答的同时构建和维护紧凑的视觉记忆变得极具挑战。为应对这些挑战，我们提出了CoverMem，它包含三个关键组件：（1）一种覆盖约束的记忆构建方法，从完整帧中选择一个紧凑且信息丰富的集合，以满足对象质量和代表性场景的覆盖要求；（2）一种保持覆盖性的增量维护策略，当新观察到达时，仅对受影响的帧进行局部重选以满足更新的覆盖要求；（3）一种预算受限的互补帧选择方法，在问题相关性、增量覆盖价值与计算预算之间取得平衡。

    arXiv:2611.32119v1 Announce Type: new  Abstract: Episodic-memory embodied question answering enables agents to answer questions based on previously recorded visual observations of the surrounding scenes. However, growing observation histories, varying observation quality across viewpoints, and limited computational resources make it challenging to construct and maintain compact visual memories while retrieving sufficient evidence for question answering. To address these challenges, we introduce CoverMem, which incorporates three key components: (1) a coverage-constrained memory construction method that selects a compact set of informative full frames satisfying object-quality and representative-scene coverage requirements; (2) a coverage-preserving incremental maintenance strategy that locally reselects only the affected frames to satisfy updated coverage requirements as observations arrive; and (3) a budgeted complementary frame selection method that balances question relevance, addit
    
[^132]: 一种融合深度学习与解剖学推理的自动化框架，用于从冠脉CT血管造影中进行心脏表型分析

    An Automated Framework Integrating Deep Learning and Anatomical Reasoning for Cardiac Phenotyping From CCTA

    [https://arxiv.org/abs/2611.32114](https://arxiv.org/abs/2611.32114)

    提出了一种结合深度学习与解剖学推理的自动化框架，实现基于CCTA的定量心脏表型分析，涵盖血管特异性PCAT采样、无需专家标注的EAT测量以及冠状动脉钙化定量。

    

    冠状动脉计算机断层血管造影能够显示冠状动脉解剖结构和心脏脂肪，但仅靠分割并不能保证解剖学上一致的测量结果。我们提出了一种将深度学习与解剖学推理相结合的自动化框架，用于定量心脏表型分析。其解剖测量模块利用图几何、分支身份以及周围结构来定义血管特异性的冠状动脉周围脂肪组织（PCAT）采样和替代性心外膜脂肪组织（EAT）隔室。后者还可以在无需专家心包轮廓标注的情况下生成训练目标。配套模块可在非增强CT上定量冠状动脉钙化。该框架在五个公开数据集上进行了组件级基准测试：ImageCAS、ImageCAS-X、COCA、Visual Lab和MultiD4CAD。在ImageCAS-X Test-160上，经过验证选择的后处理的管腔分割取得了0.9133的Dice分数和0.940的clDice分数（摘要在此处被截断）。

    arXiv:2611.32114v1 Announce Type: cross  Abstract: Coronary computed tomography angiography (CCTA) depicts coronary anatomy and cardiac fat, but segmentation alone does not ensure anatomically consistent measurements. We propose an automated framework that combines deep learning with anatomical reasoning for quantitative cardiac phenotyping. Its anatomical measurement module uses graph geometry, branch identity, and surrounding structures to define vessel-specific pericoronary adipose tissue (PCAT) sampling and a surrogate epicardial adipose tissue (EAT) compartment. The latter also generates training targets without requiring expert pericardium contours. A companion module quantifies coronary calcium on non-contrast CT. Component-wise benchmarking was conducted on five public datasets: ImageCAS, ImageCAS-X, COCA, Visual Lab, and MultiD4CAD. On ImageCAS-X Test-160, lumen segmentation with validation-selected post-processing achieved a Dice score of 0.9133 and a clDice score of 0.940, e
    
[^133]: Duo：迈向双手策略中的时空动作协调

    Duo: Towards Spatiotemporal Action Coordination in Bimanual Policies

    [https://arxiv.org/abs/2611.32108](https://arxiv.org/abs/2611.32108)

    Duo通过在共享动作头中将双臂分离为独立的动作token并引入跨手臂注意力机制，在流匹配过程中实现双手操作的空间-时间动作协调，使多种基础策略的成功率显著提升。

    

    双手操作要求每条手臂执行不同的动作，同时与另一条手臂协调配合。常见的基于流的机器人策略将双臂的动作合并为每个未来时间步的一个token，使注意力在时间维度上具有不同的位置，但在手臂维度上则没有。我们提出了Duo，这是一种简单而有效的设计，可以在一个共享的动作头中建模空间和时间依赖关系。具体而言，Duo将双臂在每个时间步分离为两个动作token，使每条手臂的动作能够在流匹配过程中通过跨手臂注意力进行交互。在RoboTwin 2.0上，Duo提升了所有四种基础策略（参数量从0.45B到3.62B）的总体成功率，其中FLOWER（0.89B）和π₀.₅（3.62B）分别提升了12.55和9.02个百分点。尽管其规模紧凑，FLOWER-Duo在RoboDojo上取得了最佳的随机泛化、精度和内存分数。在真实世界任务中，它在两方面均优于π₀.₅……

    arXiv:2611.32108v1 Announce Type: cross  Abstract: Bimanual manipulation requires each arm to perform distinct motions while coordinating with the other. Common flow-based robotic policies combine both arms' actions into one token per future timestep, giving attention distinct positions across time but not across arms. We introduce Duo, a simple but effective design that models spatial and temporal dependencies within one shared action head. Specifically, Duo separates both arms into two action tokens per timestep, allowing each arm's motion to interact via cross-arm attention during flow matching. On RoboTwin 2.0, Duo improves the overall success of all four base policies, from 0.45B to 3.62B parameters, with gains of 12.55 and 9.02 percentage points for FLOWER (0.89B) and $\pi_{0.5}$ (3.62B), respectively. Despite its compact size, FLOWER-Duo achieves the best randomized-generalization, precision, and memory scores on RoboDojo. In real-world tasks, it outperforms $\pi_{0.5}$ in both 
    
[^134]: PreferenceScene：基于反馈的个性化室内场景合成

    PreferenceScene: Personalized Indoor Scene Synthesis from Feedback

    [https://arxiv.org/abs/2611.32099](https://arxiv.org/abs/2611.32099)

    该论文提出PreferenceScene，通过多轮交互中用户对场景渲染图像的简短自由反馈和满意度评分，利用大语言模型从局部反馈中推断并追踪用户的隐藏偏好，实现整体个性化室内3D场景合成。

    

    文本到场景方法能够将简短的指令转化为布置好家具的3D房间，但某个人真正想要的房间很少能通过文字完整表达：许多偏好只有在人们看到具体房间并能指出其中喜欢和不喜欢之处时才会显现出来。因此，我们研究了用户隐藏偏好条件下的个性化室内场景合成问题：在多轮交互中，系统展示一个3D场景，用户仅通过查看场景的渲染图像，以简短的自由形式反馈和满意度评分进行回复。因此，系统必须从每次仅针对房间部分内容的反馈中，将整个房间逐步调整到符合用户偏好。系统还必须能够改变房间的每一处细节，因为用户的偏好不仅涉及单个物体，还涉及表面材质、光照、纺织品、家具及其布局共同营造的整体外观与氛围。我们提出了PreferenceScene，其中大语言模型保持……（摘要原文在此处截断）

    arXiv:2611.32099v1 Announce Type: cross  Abstract: Text-to-scene methods turn a short instruction into a furnished 3D room, but the room that a particular person wants rarely comes across fully in words: many preferences surface only when the person sees a concrete room and can point to what they like and dislike in it. We therefore study personalized indoor scene synthesis under hidden user preferences: over several turns, a system shows a 3D scene, and the user, who sees only rendered images of the scene, replies with brief free-form feedback and a satisfaction score. The system must therefore bring the whole room toward the user's preferences from feedback that comments on only part of it at a time. It must also be able to change every part of the room, because a preference concerns not only single objects but the look and atmosphere that surfaces, lighting, textiles, furniture and their arrangement create together. We present PreferenceScene, in which a large language model keeps a
    
[^135]: Clinical3D-Bench：评估并提升多模态大语言模型在医院环境中的空间智能

    Clinical3D-Bench: Evaluating and Improving Spatial Intelligence of Multimodal LLMs in Hospital Environments

    [https://arxiv.org/abs/2611.32094](https://arxiv.org/abs/2611.32094)

    提出 Clinical3D-Bench 医院 3D 空间推理基准，揭示当前多模态大模型的空间推理能力远逊于医生，并通过 Clinical3D-45K 数据集微调显著提升了模型的空间智能。

    

    医院中的空间推理需要追踪外观相似的设备、解读相对于患者的方向，以及判断通道宽度，但现有基准对这些能力的覆盖有限。我们提出 Clinical3D-Bench，这是一个模拟基准，包含针对 373 种病房布局第一人称视角漫步视频的 3,368 个八类问题。布局遵循医院设计指南，答案由 3D 场景计算得出，成对的场景状态用于检验模型在相关证据发生变化时是否会改变其答案。视频具有照片级真实感，并由十位医生对布局进行审核。在 35 个多模态大语言模型（MLLM）中，最优的开源模型和专有模型分别达到 45.7% 和 53.6% 的宏平均准确率，而答案先验为 40.8%，医生在一个子集上达到 79.0%；32 个开源模型中有 27 个在双向患者相对方向任务上得分低于 40%。在我们包含 44,750 个问题的训练集 Clinical3D-45K 上进行微调……

    arXiv:2611.32094v1 Announce Type: cross  Abstract: Spatial reasoning in hospitals requires tracking look-alike equipment, interpreting patient-relative directions, and judging passage widths, yet existing benchmarks provide limited coverage of these abilities. We introduce Clinical3D-Bench, a simulated benchmark with 3,368 questions of eight types on egocentric walkthroughs of 373 hospital-room layouts. Layouts follow hospital design guidelines, answers are computed from the 3D scenes, and paired scene states test whether models change their answers when relevant evidence changes. Videos are photorealistic, and ten physicians screened the layouts. Among 35 multimodal large language models (MLLMs), the best open and proprietary models reach 45.7% and 53.6% macro accuracy, against 40.8% for an answer prior and 79.0% for physicians on a subset, and 27 of 32 open models score below 40% on two-way patient-relative directions. Fine-tuning on Clinical3D-45K, our 44,750-question training set, 
    
[^136]: MARGIN：一种用于机器人程序改进的风险感知智能体

    MARGIN: A Risk-Aware Agent for Robot Program Refinement

    [https://arxiv.org/abs/2611.32093](https://arxiv.org/abs/2611.32093)

    该论文提出MARGIN，一种风险感知的机器人程序改进智能体，通过“险胜边际”指标（最慢成功执行中剩余的执行预算比例）识别脆弱程序、指导针对性修复，在保持开发成功率的同时提升程序在未见场景中的执行鲁棒性。

    

    机器人编程智能体通过反复测试与修复来改进可执行的控制策略，然而二元的成功指标会趋于饱和，而执行风险在未见过的场景中依然存在。我们提出了MARGIN，一种风险感知智能体，它将成功的机器人执行轨迹转化为可用于程序改进的可操作证据。其“险胜边际”衡量的是最慢的一次成功开发执行中所剩余的执行预算比例。该智能体利用这一信号来优先处理脆弱的程序、指导有针对性的修复，并选择那些在保持开发成功的同时能够增加执行余量的修订方案。一个精确的安全竞赛规则加速了基于成功的候选评估。我们在LIBERO-Pro、LIBERO-90和Robosuite上对MARGIN进行了评估。在我们对80个任务的LIBERO-Pro复现中，78个程序达到了完美的开发成功率，但在未见随机种子上仍存在大量失败。险胜边际能够以AU……预测留出集上的失败（原文摘要在此处截断）。

    arXiv:2611.32093v1 Announce Type: cross  Abstract: Robot-programming agents improve executable control policies through repeated testing and repair, yet binary success saturates while execution risk persists across unseen scenes. We introduce MARGIN, a risk-aware agent that converts successful robot rollouts into actionable evidence for program refinement. Its near-miss margin measures the fraction of the execution budget remaining in the slowest successful development rollout. The agent uses this signal to prioritize fragile programs, guide targeted repairs, and select revisions that preserve development success while increasing execution headroom. An exact safe-racing rule accelerates success-based candidate evaluation. We evaluate MARGIN across LIBERO-Pro, LIBERO-90, and Robosuite. In our 80-task LIBERO-Pro reproduction, 78 programs attain perfect development success while accounting for substantial failures on unseen seeds. The near-miss margin predicts held-out failures with an AU
    
[^137]: 从记忆访问到证据充分性：诊断并解决需求-证据错位问题

    From Memory Access to Evidence Sufficiency: Diagnosing and Addressing Demand--Evidence Misalignment

    [https://arxiv.org/abs/2611.32088](https://arxiv.org/abs/2611.32088)

    该论文提出 MemPath 干预性诊断框架揭示长期记忆检索中“相关不等于充分”的需求-证据错位问题，并据此提出 BiDE-Mem 方法，将问题表示为类型化需求以指导检索与溯源感知的证据组织，在 LoCoMo 和 LongMemEval 基准上取得显著准确率提升。

    

    长期记忆检索通常能返回与查询相关的记录，但相关性并不保证证据充分：回答问题需要一组能够共同满足问题的事实、关系、时间和状态约束的记录。我们通过 MemPath 研究这种需求-证据错位问题，MemPath 是一个干预性框架，用于诊断记忆访问、需求覆盖和证据利用方面的失败。在该诊断的指导下，我们提出了 BiDE-Mem，它将问题表示为类型化需求，并利用这些需求来指导检索和证据组织。我们对需求-证据支持进行了形式化定义；在所评估的紧凑路径中，这种支持通过溯源感知的证据组织和 Reader 指导来体现，而非通过单独的生成式四态验证步骤。在 LoCoMo 基准上，BiDE-Mem 相比 Mnemis 将准确率提升了 6.23 个百分点；在 LongMemEval 基准上，它达到了 91.80% 的准确率，在多跳推理和时间推理方面表现出色。

    arXiv:2611.32088v1 Announce Type: new  Abstract: Long-term memory retrieval often returns query-relevant records, but relevance does not guarantee sufficient evidence: answering requires a set of records that jointly satisfies a question's factual, relational, temporal, and state constraints. We study this demand--evidence misalignment through MemPath, an interventional framework that diagnoses failures in memory access, demand coverage, and evidence utilization. Guided by this diagnosis, we propose BiDE-Mem, which represents questions as typed demands and uses them to guide retrieval and evidence organization. We formalize demand--evidence support; in the evaluated compact path, it is conveyed through provenance-aware evidence organization and Reader guidance rather than a separate generative four-state verification step. On LoCoMo, BiDE-Mem improves accuracy by 6.23 percentage points over Mnemis; on LongMemEval, it achieves 91.80%, with strong multi-hop and temporal performance. Supp
    
[^138]: ExpGuard：保护长程大语言模型智能体免受受损经验的危害

    ExpGuard: Guarding Long-Horizon LLM Agents against Corrupted Experience

    [https://arxiv.org/abs/2611.32085](https://arxiv.org/abs/2611.32085)

    ExpGuard提出了一种检索触发的经验风险评估与在线修复框架，通过任务条件化异构图整合经验语义、来源与执行依赖关系，在经验复用前预测攻击和任务失败风险，从而保护长程LLM智能体免受被污染经验的持续危害。

    

    长程大语言模型（LLM）智能体会将过去的轨迹提炼为可复用的经验。源轨迹中的误导性或对抗性指导可能会污染这些经验，使有害指令在多个任务之间持续存在并影响智能体未来的行动。评估潜在受损经验的风险面临三大挑战：（1）在经验复用之前建模来源（provenance）和执行依赖关系，（2）从轨迹结果中学习任务条件化的风险，以及（3）在在线推理阶段修复有风险的经验。为解决这些挑战，我们提出了ExpGuard，一个用于检索触发的经验风险评估与在线修复的框架。具体而言，针对挑战（1），ExpGuard在任务条件化的异构图中将经验语义与来源和执行依赖关系相结合，以在经验复用之前预测攻击风险和任务失败风险。为实现任务条件化的风险评估……（摘要原文在此处截断）

    arXiv:2611.32085v1 Announce Type: new  Abstract: Long-horizon large language model (LLM) agents distill past trajectories into reusable experiences. Misleading or adversarial guidance in source trajectories can corrupt these experiences, allowing harmful instructions to persist across tasks and influence future agent actions. Assessing the risks of potentially corrupted experiences poses three challenges: (1) modeling provenance and execution dependencies before experience reuse, (2) learning task-conditioned risk from trajectory outcomes, and (3) repairing risky experiences at online inference. To address these challenges, we present ExpGuard, a framework for retrieval-triggered experience risk assessment and online repair. Specifically, to tackle challenge (1), ExpGuard integrates experience semantics with provenance and execution dependencies in a task-conditioned heterogeneous graph to predict attack and task-failure risks before experience reuse. To enable task-conditioned risk su
    
[^139]: 算法行为沉沦：对迈向不可逆人口崩塌之转变的建模研究

    The algorithmic behavioral sink: Modeling the transition to irreversible demographic collapse

    [https://arxiv.org/abs/2611.32084](https://arxiv.org/abs/2611.32084)

    该论文将婚姻市场建模为受热力学约束的复杂系统，提出基于费米-狄拉克选择的模型，量化了移动互联网与人工智能作为“合成择偶干扰因子”所形成的“婚姻能量势垒”与“技术因子”，揭示人类配对动力学正经历导致不可逆人口崩塌的非线性相变。

    

    2025年，全球绝大多数主要经济体报告的总和生育率已跌破2.1的更替水平。尽管各国推行了大规模的鼓励生育财政干预，这种同步性崩塌仍持续存在，暴露出传统线性人口模型的结构性局限。本文提出，移动互联网与人工智能的普及扮演了一种“合成性择偶干扰因子”的角色，正在引发人类配对结合动力学的非线性相变。我们将婚姻市场视为一个受热力学约束支配的复杂系统，提出了一个费米-狄拉克选择模型，以各国收入分布作为输入能谱，并利用日本的纵向数据进行拟合，以此阐明数字化超现实如何造就一道愈发难以逾越的“婚姻能量势垒”。我们的模型量化了一个“技术因子”，该因子在……初期使女性的择偶预期与本地经济现实相脱钩（原文摘要至此截断）。

    arXiv:2611.32084v1 Announce Type: cross  Abstract: In 2025, the vast majority of major global economies reported a Total Fertility Rate below the replacement level of 2.1. This synchronized collapse has persisted despite extensive pro-natalist financial interventions, revealing a structural limitation in conventional linear demographic models. This paper proposes that the ubiquity of mobile internet and Artificial Intelligence acts as a synthetic mating disruptor, precipitating a non-linear phase transition in human pair-bonding dynamics. Treating the marriage market as a complex system governed by thermodynamic constraints, we present a Fermi-Dirac selection model, using national income distributions as the input spectrum and fitted with Japanese longitudinal data to illustrate how digital hyper-reality generates an increasingly prohibitive Marriage Energy Barrier. Our model quantifies a Tech Factor that decouples female mating expectations from local economic realities during initial
    
[^140]: Integral DeltaNet：基于自适应记忆积分的改进线性注意力

    Integral DeltaNet: Improved Linear Attention with Adaptive Memory Integration

    [https://arxiv.org/abs/2611.32063](https://arxiv.org/abs/2611.32063)

    Integral DeltaNet通过同时维护快速更新的delta状态和保留其加权历史的积分状态这两个互补记忆，并在读取时自适应地融合二者，从而在缓解线性注意力记忆冲突的同时保留对未来预测有用的历史信息。

    

    线性注意力用线性形式替代softmax注意力，实现线性时间计算，并用固定大小的循环状态取代不断增长的键值缓存。这些计算与内存优势使线性注意力在长上下文语言建模中受到越来越多的关注。然而，将过去信息压缩到固定大小的状态中会造成记忆冲突，阻碍长上下文理解。尽管DeltaNet通过delta规则更新缓解了记忆冲突，Gated DeltaNet进一步引入门控机制加以扩展，但这些机制也可能丢弃对未来预测有用的信息。为解决这一局限，我们提出Integral DeltaNet，它维护两个互补的循环状态：一个快速更新的delta状态，用于捕获当前的键值关联；以及一个积分状态，用于保存delta状态的加权历史。在读取时，模型自适应地结合这些状态。

    arXiv:2611.32063v1 Announce Type: cross  Abstract: Linear attention replaces softmax attention with a linear formulation, enabling linear-time computation with a fixed-size recurrent state instead of a growing key-value cache. These computational and memory advantages have attracted growing interest in linear attention for long-context language modeling. However, compressing past information into a fixed-size state causes memory collisions that hinder long-context understanding. Although DeltaNet mitigates memory collisions through delta-rule updates and Gated DeltaNet extends this approach with gating, these mechanisms may also discard information relevant to future predictions. To address this limitation, we introduce Integral DeltaNet, which maintains two complementary recurrent states: a fast-updating delta state that captures current key-value associations and an integral state that preserves a weighted history of the delta state. At readout, the model adaptively combines these st
    
[^141]: 更强却更难验证？缓解多模态大语言模型强化学习中难以验证的幻觉

    More Powerful, Less Verifiable? Mitigating Hard-to-Verify Hallucinations in MLLM's Reinforcement Learning

    [https://arxiv.org/abs/2611.32055](https://arxiv.org/abs/2611.32055)

    该研究构建了区分“明显幻觉”与“隐蔽幻觉”的ITHV数据集，并提出DAPO-V方法，通过可验证性感知奖励与自适应惩罚机制，在风险预算约束下缓解多模态大语言模型强化学习中难以验证的幻觉问题。

    

    arXiv:2611.32055v1 公告类型：新 摘要：强化学习（RL）通过来自人类和环境的反馈来增强模型能力。然而，新的证据表明，奖励优化可能产生错误更难被人类察觉的输出。为解决这一问题，我们构建了ITHV（图文幻觉可验证性数据集），包含2,728个视觉落地实例，用以区分人类容易察觉的“明显幻觉”和人类难以识别的“隐蔽幻觉”。随后，我们提出DAPO-V，这是DAPO的扩展方法，它在可验证性感知风险预算的约束下最大化正确性。具体而言，DAPO-V将可验证性感知奖励与自适应双重控制相结合：对于人类验证准确率较低的问题，模型给出的错误回答会被施加更大的惩罚；自适应乘数则根据风险预算的违规情况进一步调整惩罚强度。在三个多模态大语言模型基线（摘要原文在此处截断）……

    arXiv:2611.32055v1 Announce Type: new  Abstract: Reinforcement learning (RL) strengthens model capabilities through feedback from humans and environments. However, emerging evidence suggests that reward optimization can produce outputs whose errors are harder for humans to detect. To address this issue, we construct ITHV (the Image-Text Hallucination Verifiability Dataset), comprising 2,728 visually grounded instances that distinguish between obvious hallucinations, which humans readily detect, and elusive hallucinations, which humans struggle to identify. We then propose DAPO-V, an extension of DAPO that maximizes correctness subject to a budget on verifiability-aware risk. Specifically, DAPO-V combines a verifiability-aware reward with adaptive dual control, where larger penalties are assigned to incorrect responses on questions with lower human verification accuracy. An adaptive multiplier further adjusts the penalty strength according to risk-budget violations. Across three MLLM ba
    
[^142]: EffectWAM：为机器人操作学习基于动作的效应

    EffectWAM: Learning Action-Grounded Effects for Robot Manipulation

    [https://arxiv.org/abs/2611.32053](https://arxiv.org/abs/2611.32053)

    EffectWAM通过Effect视觉变化表示、相机位置编码和动作接地三个互补组件，利用流匹配联合预测视觉效应与可执行动作，将视觉预测与可执行的机器人控制紧密连接起来。

    

    世界-动作模型将视觉预测与可执行的机器人控制连接起来。我们提出EffectWAM，它通过三个互补组件发展了这种连接：Effect、相机位置编码和动作接地。Effect将视觉变化表示为独立编码的未来观测与共享的当前观测锚点之间的差异。当可靠的外参可用时，CaPE基于相对相机位姿对端点池化进行条件化，同时对于未标定的轨迹，与无位姿池化共享相同的学习参数。动作接地训练视觉表示以保留末端执行器位姿和夹爪状态的测量变化。观测编码器的指数移动平均提供Effect目标。一个共享的生成专家学习通过流匹配联合预测Effect和可执行的动作块，并重用采样得到的动作对按需生成未来视频。

    arXiv:2611.32053v1 Announce Type: new  Abstract: World-action models connect visual prediction with executable robot control. We present EffectWAM, which develops this connection through three complementary components: Effect, camera positional encoding (CaPE), and Action Grounding. Effect represents visual change as the difference between independently encoded future observations and a shared current-observation anchor. CaPE conditions endpoint pooling on relative camera pose when reliable extrinsics are available, while sharing the same learned parameters with pose-free pooling for uncalibrated trajectories. Action Grounding trains the visual representation to retain measured changes in end-effector pose and gripper state. An exponential moving average of the observation encoder supplies Effect targets. A shared generation expert learns to jointly predict Effect and an executable action chunk through flow matching, and reuses the sampled pair to generate future video on demand. This 
    
[^143]: 从基于规则到智能体化：基于大语言模型智能体的城市排水系统实时控制以缓解洪涝与污水溢流

    From Rule-based to Agentic: LLM Agent based Real-time Control of Urban Drainage Systems for Flooding and Sewer Overflow Mitigation

    [https://arxiv.org/abs/2611.32046](https://arxiv.org/abs/2611.32046)

    该研究提出AgenticAI-Control智能体框架，利用两个耦合的大语言模型智能体将多源气象预警信息与工程专业知识融合，实现城市排水系统的实时控制，相比规则基线控制使洪涝与溢流总量减少69.1%-90.6%。

    

    城市排水系统（UDS）的实时控制（RTC）仍受限于管网内稀疏的传感监测。尽管异构多模态的气象预警信息往往包含对城市排水系统实时控制至关重要的决策信息，但现有方法均无法在控制工作流中利用这类外部来源的信息。本文提出AgenticAI-Control，一个能够在实时控制中融合多源气象预警信息的智能体人工智能框架。该框架集成了两个耦合的大语言模型智能体，通过检索气象预警信息并将其与领域工程专业知识相融合来生成控制策略。通过对中国东部某合流制排水流域在10场真实降雨事件下的仿真研究进行验证，与基于规则的基线控制相比，AgenticAI-Control使洪涝与溢流总量减少了69.1%至90.6%。经过严格标准化的提示词……（摘要在此处截断）

    arXiv:2611.32046v1 Announce Type: new  Abstract: Real-time control (RTC) for urban drainage systems (UDS) remains hampered by sparse in-network sensing. While heterogeneous multi-modal meteorological early warnings often contain decision-critical information for UDS RTC, none of the existing approaches can exploit such externally-sourced information within the control workflow. Here we introduce AgenticAI-Control, an agentic artificial intelligence framework able to assimilate multi-source meteorological early warnings in RTC. Two coupled large language model agents are integrated to generate control strategies by retrieving meteorological early-warnings and fusing them with domain engineering expertise. Validated via simulation studies of a combined sewer catchment in eastern China across 10 real-world rainfall events, the AgenticAI-Control achieved 69.1-90.6% reduction in aggregated flood and overflow volumes compared to the rule-based baseline control. Rigorously standardized prompt
    
[^144]: Spec-CoFrGeNet：通过重参数化连分数网络实现稳定且精确的训练

    Spec-CoFrGeNet: Reparametrizing Continued Fraction Networks for Stable and Accurate Training

    [https://arxiv.org/abs/2611.32041](https://arxiv.org/abs/2611.32041)

    该论文提出Spec-CoFrGeNet，通过受格林函数启发对部分分母进行变换以约束三对角哈密顿量的谱，从而解决连分数网络因分母趋零产生极点导致的训练不稳定问题，无需下界约束和增量式训练即可实现稳定且精确的训练。

    

    连分数已被证明是一种能够构建表达能力强且参数高效的神经网络的函数类。这类神经网络不仅在监督学习中有效，还可用于训练大型语言模型（LLM）。然而，该函数类的有理形式在分母趋近于零时会引入极点，以往的工作通过强制施加一个较小的正数（非平滑）下界来避免这一问题。我们证明，性能和训练稳定性对该下界的选择可能高度敏感。在这项工作中，我们受该函数与物理学中格林函数联系的启发，引入了一种对部分分母的新型变换，通过约束所构造的三对角哈密顿量的谱，优雅地缓解了上述问题。我们的方法还进一步消除了以往工作为保持稳定性而采用的增量式训练，从而避免了繁琐的标准预训练过程修改。

    arXiv:2611.32041v1 Announce Type: cross  Abstract: Continued fractions have been shown to be a function class capable of constructing expressive yet parameter-efficient neural networks. These neural networks are not only effective for supervised learning, but also for training LLMs. However, the rational form of this function class introduces poles when the denominator approaches zero, which prior work has avoided by enforcing a small positive (non-smooth) lower bound. We show that performance and training stability can be highly sensitive to the choice of this lower bound. In this work, we introduce a novel transformation of the partial denominators, inspired by its connection to the Green's function in physics, that mitigates the above issue elegantly by constraining the spectra of the constructed tridiagonal Hamiltonian. Our approach further eliminates the need for incremental training used by previous works for stability, thus avoiding involved modifications to standard pre-trainin
    
[^145]: 溯源隔离认知仲裁团：智能体控制平面中的执行绑定证据认证与可能世界安全门控

    Provenance-Isolated Epistemic Quorums: Execution-Bound Evidence Certification and Plausible-World Safety Gating in Agentic Control Planes

    [https://arxiv.org/abs/2611.32032](https://arxiv.org/abs/2611.32032)

    该论文提出PIEQ框架，将所有语言模型严格排除在可信计算基之外，通过基于假设的真值维护、物理溯源隔离、可能世界安全屏蔽与可线性化并发治理，为智能体变更生产基础设施提供执行绑定的证据认证与跨故障世界的安全门控。

    

    大型语言模型之间达成的一致意见并不能作为变更生产基础设施的物理证据。由于生成式模型共享预训练先验，并在协商过程中倾向于迎合同伴的推理，投票集成机制只会使错误相互关联，而非提供独立的容错能力；然而，若将僵化的确定性检查强加于不受约束的智能体输出之上，又会使事故修复陷入瘫痪。我们提出了溯源隔离认知仲裁团（PIEQ），这是一种执行绑定的认证层，可将每个语言模型严格排除在可信计算基之外。PIEQ 围绕三项不同的义务，统一了基于假设的真值维护、物理溯源隔离、基于控制论的可能世界安全屏蔽以及可线性化的并发治理：从互不相交的物理故障域认证诊断前提条件，在所有未被反驳的故障世界中对候选变更进行门控，并在条件不满足时回退到只读诊断……

    arXiv:2611.32032v1 Announce Type: cross  Abstract: Agreement among large language models is not physical evidence for mutating production infrastructure. Because generative models share pretraining priors and conform to peer rationales during deliberation, voting ensembles correlate errors instead of providing independent fault tolerance; yet attaching rigid deterministic checks to unconstrained agent outputs paralyzes incident remediation. We present Provenance-Isolated Epistemic Quorums (PIEQ), an execution-bound certification layer that keeps every language model strictly outside the trusted computing base. PIEQ unifies assumption-based truth maintenance, physical provenance isolation, control-theoretic plausible-world safety shielding, and linearizable concurrency governance around three distinct obligations: certifying diagnostic preconditions from disjoint physical failure domains, gating candidate mutations across all unrefuted fault worlds while falling back to read-only diagno
    
[^146]: 统一伙伴多样性与环境多样性以提升人机协调能力

    Unifying Partner and Environment Diversity to Improve Human-AI Coordination

    [https://arxiv.org/abs/2611.32031](https://arxiv.org/abs/2611.32031)

    该论文提出跨环境协同博弈（CECP）两阶段框架，通过统一建模伙伴多样性与环境多样性，显著提升AI智能体在陌生环境中与陌生人类伙伴的协调泛化能力。

    

    现代人工智能系统的现实应用需要其能够泛化到未见过的环境以及陌生的人类用户。零样本协调（ZSC）领域主要关注在熟悉的环境中部署AI以与陌生伙伴进行合作。流行的基于种群的方法通过在固定环境的训练过程中让智能体与多样化的模拟人类用户配对来实现这一目标，但没有考虑更为现实的场景，即用户和环境在训练期间都是未见的。最近，有研究探索了在跨环境训练中训练单个自博弈智能体，该方法在陌生人类用户和未见环境上均表现出更好的泛化能力。在本工作中，我们探究联合建模伙伴多样性与环境多样性是否能够在陌生环境中与陌生伙伴实现更强的协调。我们提出了跨环境协同博弈，这是一个两阶段框架

    arXiv:2611.32031v1 Announce Type: cross  Abstract: Real-world applications of modern AI systems require generalization to both unseen environments and novel human users. The field of zero-shot coordination (ZSC) has focused primarily on deploying AI to cooperate with novel partners in otherwise familiar environments. Popular population-based methods achieve this by partnering the agent with a diversity of simulated human users during training in a fixed environment, but does not consider the more realistic scenario where both the users and the environment are unseen during training. Recently, training across environments has been explored in training a single self-play agent, which shows better generalization on both novel human users and unseen environments. In this work, we ask whether jointly modeling partner diversity and environment diversity yields stronger coordination with novel partners in novel environments. We introduce Cross Environment Co-Play (CECP), a two-stage framework
    
[^147]: 占据世界模型真的遵循动作吗？

    Do Occupancy World Models Really Follow Actions?

    [https://arxiv.org/abs/2611.32029](https://arxiv.org/abs/2611.32029)

    论文提出ReCast模型，通过将未来预测建立在动作引起的自车帧间几何对应关系上，解决了占据世界模型在反事实动作下预测不忠实遵循指定轨迹的问题，并首次引入了反事实动作忠实性评估基准。

    

    占据世界模型以自车动作为条件预测未来3D场景，然而在反事实动作下，其预测可能无法忠实地遵循指定的轨迹。由于驾驶数据集对每个场景的替代轨迹覆盖稀疏，现有模型只能从观测到的场景-轨迹组合中隐式学习自车运动的空间效应，这可能将动作效应与场景-轨迹相关性纠缠在一起。我们提出ReCast，一种由自车轨迹引导的Transformer，它将未来预测建立在自车中心帧之间由动作引起的几何对应关系之上。ReCast将历史编码与未来解码解耦，并利用相对自车变换对可变形交互进行对齐，同时保持对非刚性场景演化的灵活性。我们还引入了一个反事实动作忠实性基准，涵盖分布内、尾部和分布外轨迹。在Occ3D-nuSce（原文截断）

    arXiv:2611.32029v1 Announce Type: cross  Abstract: Occupancy world models predict future 3D scenes conditioned on ego actions, yet their predictions may not faithfully follow the specified trajectory under counterfactual actions. Because driving datasets provide sparse coverage of alternative trajectories for each scene, existing models learn the spatial effect of ego motion implicitly from observed scene--trajectory combinations, which may entangle action effects with scene--trajectory correlations. We propose ReCast, an ego-trajectory-guided transformer that grounds future prediction in action-induced geometric correspondence across ego-centric frames. ReCast decouples historical encoding from future decoding and aligns deformable interactions using relative ego transformations, while retaining flexibility for non-rigid scene evolution. We also introduce a counterfactual action-faithfulness benchmark covering in-distribution, tail, and out-of-distribution trajectories. On Occ3D-nuSce
    
[^148]: 超越前后对比图像：评估轻量级视觉语言模型在灾害评估中的时间定位能力

    Beyond Before-and-After Images: Evaluating Temporal Grounding in a Lightweight VLM for Disaster Assessment

    [https://arxiv.org/abs/2611.32022](https://arxiv.org/abs/2611.32022)

    该研究提出受控诊断实验，发现轻量级VLM（InternVL3-2B）在灾害评估中并未真正利用灾前影像进行时间定位，其看似的变化推理实际主要依赖灾后影像和提示词表述。

    

    双时相卫星影像是灾害评估的核心：判断哪些区域受损需要将灾前场景与其灾后对应影像进行对比。然而，视觉语言模型（VLM）可能表面上回答了变化相关问题，实际上却主要依赖灾后影像或提示词中的时间表述。我们提出了一种受控的时间定位诊断方法，并将其应用于一个轻量级VLM——InternVL3-2B，在DisasterM3数据集上进行评估：该数据集包含来自41个事件、九种灾害类型的1,659对光学和SAR影像对，采用事件不相交的数据划分，共包含9,031个推理问题。我们对比了以下输入条件：仅灾后影像、真实的灾前/灾后影像对、灾后影像重复两次、影像顺序反转，以及无关的灾前影像。结果显示，真实影像对的准确率仅比重复灾后影像对照高0.5个百分点（95%配对自助法置信区间为[-0.2, +1.2]），反转影像顺序仅使准确率变化-0.2个百分点（[-0.8, +0.5]）。因此，在该诊断协议下……

    arXiv:2611.32022v1 Announce Type: cross  Abstract: Bi-temporal satellite imagery is central to disaster assessment: deciding what was damaged requires comparing a pre-event scene with its post-event counterpart. A vision-language model (VLM) can nonetheless appear to answer change questions while relying mainly on the post-event image or on the prompt's temporal framing. We present a controlled diagnostic of temporal grounding and apply it to one lightweight VLM, InternVL3-2B, on DisasterM3: 1,659 optical and SAR image pairs from 41 events and nine disaster types, in event-disjoint partitions, with 9,031 reasoning questions. We compare post-only input, the true pre/post pair, the post image repeated twice, reversed image order, and an unrelated pre-event image. The true pair scores $+0.5$ accuracy points above the repeated-post control (95\% paired bootstrap CI $[-0.2, +1.2]$), and reversing the order changes accuracy by $-0.2$ points ($[-0.8, +0.5]$). Under this protocol we therefore 
    
[^149]: HuluOPD：利用无标注提示将异构医学专科模型统一为一个轻量级模型

    HuluOPD: Unifying Heterogeneous Medical Specialists into One Lightweight Model with Unlabeled Prompts

    [https://arxiv.org/abs/2611.32021](https://arxiv.org/abs/2611.32021)

    HuluOPD提出一种在线策略蒸馏方法，仅用无标注提示和跨分词器监督损失，就能把架构各异的多个医学专科视觉-语言模型统一蒸馏为一个轻量级统一模型，无需参考答案或专科模型的原始训练数据。

    

    专科医学视觉-语言模型（VLM）在其各自专科领域内表现出色，但难以统一：它们的训练数据通常是私有的，且独立开发的专科模型无法假定共享相同的架构。分别部署这些大型模型的成本很高。我们提出了HuluOPD，这是一种在线策略蒸馏方法，仅使用无标注提示就能将异构的专科模型统一为一个轻量级VLM，无需参考答案或专科模型的训练数据。轻量级模型对每个提示进行回答，并由该提示所属模态对应的专科模型监督其回答；一种与续写内容对齐的跨分词器损失（continuation-aligned cross-tokenizer loss）使得即使在两者架构和分词器不同的情况下也能实现这一点。我们将HuluOPD应用于五个专科模型，分别涵盖超声、病理、光学相干断层扫描（OCT）、眼底照相和牙科影像。为了进行蒸馏和评估，我们构建了包含91.4万条无标注提示的HuluOPDCorpus以及HuluOPD

    arXiv:2611.32021v1 Announce Type: cross  Abstract: Specialist medical vision-language models (VLMs) are strong within their specialties but hard to unify: their training data are often private, and independently developed specialists cannot be assumed to share an architecture. Deploying these large models separately is costly. We present HuluOPD, an on-policy distillation recipe that unifies heterogeneous specialists into one lightweight VLM using only unlabeled prompts, without requiring reference answers or the specialists' training data. The lightweight model answers each prompt, and the specialist of the prompt's modality supervises its response; a continuation-aligned cross-tokenizer loss makes this possible even when they differ in architecture and tokenizer. We apply HuluOPD to five specialists, for ultrasound, pathology, optical coherence tomography, fundus photography, and dental imaging. To distill and evaluate them, we build HuluOPDCorpus, 914k unlabeled prompts, and HuluOPD
    
[^150]: 面向个人AI智能体分层授权的结构感知Transformer

    Structure-Aware Transformers for Hierarchical Authorization in Personal AI Agents

    [https://arxiv.org/abs/2611.32018](https://arxiv.org/abs/2611.32018)

    提出PATH架构，通过层级感知注意力掩码与父门控概率机制，为个人AI智能体预测与权限层级结构一致的用户授权偏好，从而在保障隐私安全的同时减轻手动审查权限请求的负担。

    

    个人AI智能体可以代表用户执行复杂的多步骤任务，但这一过程通常涉及大量的权限请求。逐一手动审查这些请求非常耗时，而不加区分地全部批准则会带来隐私与安全风险。预测用户的授权偏好是一种既减轻负担又保留用户控制权的方式。然而，授权预测极具挑战性，因为权限请求往往呈层级结构，且观察到的用户决策稀疏且可能相互关联。为应对这些挑战，我们提出了分层父门控授权Transformer（PATH），用于预测用户对个人AI智能体的授权偏好。PATH是一种新颖的神经网络架构，它将层级感知的注意力掩码与父门控概率公式相结合，以确保预测结果与层级结构保持一致。层级嵌入进一步支持从稀疏观察中学习。

    arXiv:2611.32018v1 Announce Type: cross  Abstract: Personal AI agents can carry out complex, multi-step tasks on users' behalf, but doing so often involves numerous permission requests. Manually reviewing each request is time-consuming, while indiscriminate approval poses privacy and security risks. Predicting users' authorization preferences offers a way to reduce this burden while preserving user control. However, authorization prediction is challenging because permission requests are often hierarchical and observed user decisions are sparse and potentially correlated. To address these challenges, We propose the Parent-gated Authorization Transformer for Hierarchies (PATH) to predict users' authorization preferences for personal AI agents. PATH is a novel neural network architecture that combines hierarchy-aware attention masks with a parent-gated probability formulation to ensure hierarchy-consistent predictions. Hierarchy-level embeddings further support learning from sparse observ
    
[^151]: 当弹性策略的在线自适应适得其反时：Resilience4j中的安全探索与测量陷阱

    When Online Adaptation of Resilience Policies Hurts: Safe Exploration and Measurement Pitfalls in Resilience4j

    [https://arxiv.org/abs/2611.32015](https://arxiv.org/abs/2611.32015)

    该论文提出AdaptR4j系统，利用上下文赌博机在运行时动态选择Resilience4j的熔断器/重试/舱壁策略，并通过安全策略菜单和有界奖励的安全探索两项保障机制，克服了朴素在线学习在探索阶段拒绝流量的陷阱，在所有可赢得的故障场景中超越生产默认配置。

    

    Java微服务中的熔断器、重试和舱壁设置通常只调优一次便被冻结，然而最佳设置实际上取决于依赖服务当前表现出的故障类型。我们研究了在线学习器能否在运行时选择这些设置。AdaptR4j在Spring Boot 3中原子性地切换Resilience4j策略，并让上下文赌博机根据Prometheus信号在完整的熔断器/重试/舱壁配置菜单中进行选择。我们的第一个朴素设计在其所优化的奖励指标下，在四个可赢得的故障场景中有三个的表现反而不如保持默认策略：在学习器仍处于学习阶段时，探索包含激进熔断器的配置菜单导致大部分流量被拒绝。两个安全保障措施解决了这一问题：一是安全策略菜单（不包含比延迟SLO更严格的熔断器配置），二是带有有界奖励的安全探索。有了这些措施，AdaptR4j在每一个可赢得的场景中都胜过了生产默认配置（p < 0.001，每个场景三次运行，运行间标准差 <=

    arXiv:2611.32015v1 Announce Type: cross  Abstract: Circuit breaker, retry, and bulkhead settings in Java microservices are tuned once and frozen, although the best setting depends on the fault a dependency is currently showing. We study whether an online learner can pick these settings at runtime. AdaptR4j swaps Resilience4j policies atomically in Spring Boot 3 and lets a contextual bandit choose among complete CircuitBreaker/Retry/Bulkhead configurations from Prometheus signals. Our first, naive design scored worse than keeping the default policy in three of four winnable fault scenarios under the reward it optimised: exploring a menu that contained aggressive circuit breakers rejected most traffic while the learner was still learning. Two safeguards fix this: a safe policy menu (no breaker stricter than the latency SLO) and safe exploration with a bounded reward. With them, AdaptR4j beats the production default in every winnable scenario (p < 0.001, three runs each, run-to-run SD <= 
    
[^152]: 用还是不用评论家：这是个问题

    To Critic Or Not To Critic

    [https://arxiv.org/abs/2611.32013](https://arxiv.org/abs/2611.32013)

    本文从梯度方向的偏差-方差权衡入手，揭示PPO（带评论家）与GRPO（不带评论家）之间没有普适的优劣答案，方法选择取决于批次大小、对评论家的依赖程度和任务复杂度等相互作用因素。

    

    随着大语言模型处理越来越长、越来越复杂的任务，理解评论家网络何时能提升强化学习（RL）的效率变得日益重要。近端策略优化（PPO）通过训练评论家网络来降低方差并提供细粒度的信用分配，而组相对策略优化（GRPO）则舍弃评论家网络，从而避免了基于评论家价值估计进行自举所带来的偏差。尽管近期基于评论家的方法相对于GRPO报告了出色的性能，但何时选择哪种方法仍不明确。我们认为这个问题没有普适的答案；相反，方法的选择取决于多个相互作用的因素，包括批次大小、对评论家的依赖程度以及任务复杂度。我们从梯度方向的角度研究偏差-方差的权衡，衡量随着批次大小的增加，每种方法的梯度估计与真实策略梯度的契合程度。在共享的演员目标函数下，我们……

    arXiv:2611.32013v1 Announce Type: cross  Abstract: As large language models tackle longer and more complex tasks, understanding when a critic improves the efficiency of reinforcement learning (RL) becomes increasingly important. Proximal Policy Optimization (PPO) trains a critic to reduce variance and provide fine-grained credit assignment, while Group Relative Policy Optimization (GRPO) drops the critic and avoids the bias of bootstrapping from the critic's value estimates. Although recent critic-based methods have reported strong performance relative to GRPO, it remains unclear when each approach is preferable. We argue that there is no universal answer; instead, the choice depends on several interacting factors, including batch size, critic reliance, and task complexity. We study the bias-variance tradeoff in terms of gradient direction, measuring how closely each method's gradient estimates align with the true policy gradient as batch size grows. Under a shared actor objective, we 
    
[^153]: Marketing-Bench：评估前沿模型与智能体在真实世界营销能力上的表现

    Marketing-Bench: Evaluating Frontier Models and Agents on Real-World Marketing Capabilities

    [https://arxiv.org/abs/2611.32012](https://arxiv.org/abs/2611.32012)

    该论文提出了首个开源营销基准Marketing-Bench，通过将品牌异常检测（判别式任务）与修复（生成式任务）配对且保持答案可验证，从事实性、一致性、连贯性、结构和品牌语调五个维度评估AI模型与智能体的真实营销能力。

    

    随着前沿AI模型和智能体在现实世界中被更广泛地部署，营销正成为其最大的应用领域之一。营销团队使用AI模型来制作品牌资产，如网站文案和社交媒体帖子，而消费者的智能体则自主发现产品、调研购买决策并浏览网站。然而，评估营销能力十分困难，因为与软件开发、客户支持或法律审查等真值定义较为客观的领域不同，营销是主观的，其输出质量取决于品味和品牌调性。我们提出了Marketing-Bench，这是首个开源的营销基准，它将判别式任务（检测品牌异常）与生成式任务（修复问题）配对，同时保持答案可验证。Marketing-Bench测试五个关键维度：事实性、一致性、连贯性、结构和品牌语调，这些都是优质品牌内容的关键要素。

    arXiv:2611.32012v1 Announce Type: new  Abstract: As frontier AI models and agents are deployed more broadly in the real world, marketing is emerging as one of their largest applications. Marketing teams use AI models to produce brand assets like website copy and social media posts, while consumers' agents autonomously discover products, research purchases, and navigate websites. However, evaluating marketing capability is difficult because unlike areas like software development, customer support, or legal review where ground truth is more objectively defined, marketing is subjective, and output quality is judged based on taste and brand. We introduce Marketing-Bench, the first open-source marketing benchmark, which pairs discriminative tasks (detecting brand anomalies) with generative tasks (remediation), while keeping answers verifiable. Marketing-Bench tests five key pillars: factuality, consistency, coherence, structure, and brand voice, all crucial elements for good brand content o
    
[^154]: 平静智能体：围绕人类间歇性注意力重构智能体交互

    Calm Agent: Reframing Agent Interaction around Intermittent Human Attention

    [https://arxiv.org/abs/2611.32009](https://arxiv.org/abs/2611.32009)

    本文提出“平静智能体”设计理念，主张围绕用户间歇性注意力设计辅助型AI智能体，通过持久任务表示连接“委托、调整、查看”三种按需交互模式，使任务在交互间隙持续运行且用户可低成本返回理解与控制，并以桌面原型Orvlet加以演示。

    

    AI智能体可以在用户专注于其他工作时执行辅助任务。然而，用户仍然需要了解任务状态、进行修改，并在需要帮助时作出响应。我们主张，面向持续性辅助任务的智能体应当围绕人类间歇性的注意力来设计：任务应在交互间隙持续运行，用户应能够以很小的代价随时返回并理解与控制这些任务。我们将这一立场发展为“平静智能体”。它通过一种持久化的任务表示，连接了三种按需交互模式：委托、调整和查看。生成式交互为开放性目标提供合适的控制与反馈，而现有控件则支持常规操作。我们以Orvlet来阐述这些原则，这是一个支持后台执行与紧凑任务界面的桌面原型。我们识别了若干挑战，包括在保持连续性的同时调整界面、跨任务协调注意力，以及提供及时的响应交互。

    arXiv:2611.32009v1 Announce Type: cross  Abstract: AI agents can carry out auxiliary tasks while users focus on other work. Yet users still need to understand task state, make changes, and respond when help is required. We argue that agents for ongoing auxiliary tasks should be designed around intermittent human attention: tasks should continue between interactions, and users should be able to return with little effort to understand and control them. We develop this position as Calm Agent. It connects three on-demand interaction modes: Delegate, Adjust, and Check, through a persistent task representation. Generative interaction provides controls and feedback suited to open-ended goals, while existing controls support routine actions. We illustrate these principles with Orvlet, a desktop prototype for background execution and compact task interfaces. We identify challenges in adapting interfaces while preserving continuity, coordinating attention across tasks, providing responsive inter
    
[^155]: TableRouter：面向长表格问答的查询感知证据构建

    TableRouter: Query-Aware Evidence Construction for Long-Table Question Answering

    [https://arxiv.org/abs/2611.32006](https://arxiv.org/abs/2611.32006)

    TableRouter是一个无需训练的框架，通过分析查询在计算路线（从完整表格推导问题相关统计数据）和覆盖路线（检索相关行与列摘要）之间智能路由，在长表格问答任务中显著提升准确率（最高18.3个百分点）并将计算成本降低77-95%。

    

    LLM能够对表格进行推理，但当表格规模超出上下文预算时，仅检索相关行并不足以解决问题。检索到的子集可能遗漏精确聚合所需的信息；即使证据完整，LLM也可能难以有效利用。我们提出了TableRouter，这是一个无需训练的框架，它分析每个查询，然后在两条路线之间做出选择：计算路线，从完整表格中推导以问题为条件的统计数据；覆盖路线，检索相关行和列摘要。所选上下文通过一次调用传递给冻结的LLM。在三个LLM和三个大表格基准测试中，TableRouter在全部九个设置中以1.2-18.3个准确率点的优势超越了六个单次调用基线中最强的一个，其估计FLOPs比截断表格提示方法减少77-95%，且索引成本可在多个问题间摊销。这些增益还可以扩展到来自此前实验未使用的数据库中的长表格。

    arXiv:2611.32006v1 Announce Type: new  Abstract: LLMs can reason over tables, but when tables outgrow the context budget, retrieving relevant rows is only part of the solution. A retrieved subset may omit information required for an exact aggregate; even complete evidence may remain difficult for an LLM to use. We introduce TableRouter, a training-free framework that analyzes each query, then chooses between two routes: a computation route, which derives question-conditioned statistics from the full table, and a coverage route, which retrieves relevant rows and column summaries. The chosen context is passed to a frozen LLM in one call. Across three LLMs and three large-table benchmarks, TableRouter outperforms the strongest of six single-call baselines in all nine settings by 1.2-18.3 accuracy points, using 77-95% fewer estimated FLOPs than truncated-table prompting, with indexing amortized across questions. The gains extend to long tables from databases not used in earlier experiments
    
[^156]: ADE：一种面向潜在世界模型的零训练遗忘方法

    ADE: A Zero-Training Unlearning Method for Latent World Models

    [https://arxiv.org/abs/2611.31998](https://arxiv.org/abs/2611.31998)

    提出零训练闭式遗忘算法ADE，仅利用缓存的预训练统计矩即可从潜在世界模型中精准切除后门触发器与过时物理机制的污染，且无需原始保留数据、不扭曲潜在流形、不损害有效控制响应。

    

    潜在世界模型从离线轨迹中学习紧凑的循环动力学，但已部署的模拟器可能保留后门触发器或来自过时物理机制的陈旧转移。现有的遗忘方法在自回归物理场景中失效：梯度更新会扭曲潜在流形并加剧滚动预测漂移，朴素的输出清零会在触发器出现时抑制有效的控制响应，而单步删除会在循环隐藏状态内部留下损坏。我们提出了分析动力学切除（ADE），这是一种面向潜在世界模型的确定性、零训练的闭式遗忘算法。ADE 仅利用缓存的预训练统计矩，通过单次矩阵降更新恢复保留数据的协方差，无需原始保留数据，并将闭式编辑级联作用于输入瓶颈、隐藏状态和下游头部。将正交切除与白化循环阻尼和均值保持仿射校准相结合，可中和流形外污染（原文在此处截断）。

    arXiv:2611.31998v1 Announce Type: cross  Abstract: Latent world models learn compact recurrent dynamics from offline trajectories, yet deployed simulators can retain backdoor triggers or stale transitions from outdated physical regimes. Existing unlearning methods fail in autoregressive physics: gradient updates distort latent manifolds and compound rollout drift, naive output zeroing suppresses valid control responses on trigger onset, and single-step deletion leaves corruption inside recurrent hidden states. We propose Analytical Dynamics Excision (ADE), a deterministic, zero-training closed-form unlearning algorithm for latent world models. Using only cached pretraining moments, ADE recovers retain covariances via single-pass matrix downdating without raw retain data and cascades closed-form edits across the input bottleneck, hidden states, and downstream heads. Pairing orthogonal excision with whitened recurrent damping and mean-preserving affine calibration neutralizes out-of-mani
    
[^157]: 边缘规则能看到多少依赖性？扩散语言模型并行解码中的总相关性

    How Much Dependency Can a Marginal Rule See? Total Correlation in Diffusion LM Parallel Decoding

    [https://arxiv.org/abs/2611.31990](https://arxiv.org/abs/2611.31990)

    该论文量化了扩散语言模型并行解码中基于边缘分布的解码器对提交token集合条件总相关性（即依赖性损失）的预测能力，发现其预测准确性主要来源于边界本身，且估计值在散文文本上通常比精确 KL 代价低 0.4 到 1.1 比特。

    

    掩码扩散语言模型之所以快速，是因为它在每一步提交多个token。每个token都是从其自身的边缘分布中采样的，因此token之间的依赖性丢失了，而该损失的大小就是所提交集合的条件总相关性 C。大多数解码器从每个位置的边缘分布中选择集合，这可以约束 C 但无法确定它。我们探讨了它们对 C 的预测能力如何。我们在来自两个家族的四个掩码扩散语言模型上估计固定候选集合的 C。在四个模型中的三个上，最佳的仅边缘预测器偏差约为一比特（0.74 到 1.17），比仅知道边界的基线好 1.8 到 3.7 倍，而这种准确性大部分来自边界本身。在一个子集上，我们的估计对集合的排序效果类似于并行提交的精确 KL 代价，但在散文文本上以及当文本其余部分仍被掩码时，比该代价中位数低 0.4 到 1.1 比特。Dream-instruct 是例外，主要体现在散文文本上，其条件……（摘要截断）

    arXiv:2611.31990v1 Announce Type: new  Abstract: A masked diffusion language model is fast because it commits several tokens per step. Each token is sampled from its own marginal, so the dependency between the tokens is lost, and the size of that loss is the conditional total correlation C of the committed set. Most decoders choose the set from per-position marginals, which can bound C but cannot determine it. We ask how well they predict it. We estimate C for fixed candidate sets on four masked diffusion LMs from two families. On three of the four, the best marginal-only predictor is off by about one bit (0.74 to 1.17), 1.8 to 3.7 times less than a baseline that knows only the bound, and most of this accuracy comes from the bound itself. On a subset, our estimate ranks sets like the exact KL cost of a parallel commit, but sits a median 0.4 to 1.1 bits below it on prose and when the rest of the text is still masked. Dream-instruct is the exception, mainly on prose, where its conditiona
    
[^158]: 发现带宽高效的预取器以缩小缓存层次结构

    Discovering Bandwidth-Efficient Prefetchers to Shrink the Cache Hierarchy

    [https://arxiv.org/abs/2611.31983](https://arxiv.org/abs/2611.31983)

    本文提出DuoArchitect——一个自主智能体框架，通过多保真度仿真与多核协同演化，联合发现覆盖L1D、L2和LLC各层级的协调带宽高效预取策略，其仿真速度比传统集群评估快3-26倍，从而帮助缩小缓存层次结构、释放芯片面积。

    

    共享末级缓存（LLC）日益占据服务器处理器芯片面积的主导地位，迫使架构师在核心数量与缓存容量之间进行取舍。虽然智能数据预取提供了一条回收这些硅片面积的途径，但为多核系统设计协调的多级预取策略却受制于耗时一整天的周期级仿真。这一仿真瓶颈甚至限制了基于LLM的微架构探索；最先进的ArchAgent系统在单核基准测试中一次只演化一个缓存层级，仅在最后才加入多核配置，并且在多核共享缓存竞争下落后于人类冠军方案。我们提出了DuoArchitect，一个用于多保真度、多核协同发现预取器的自主智能体框架。智能体在L1D、L2和LLC层级上共同演化评估代理与协调的预取策略，所构建的仿真代理比集群评估快3-26倍……

    arXiv:2611.31983v1 Announce Type: new  Abstract: Shared last-level caches (LLCs) increasingly dominate server processor die area, forcing architects to trade core counts for cache capacity. While intelligent data prefetching offers a pathway to reclaim this silicon, designing coordinated, multi-level prefetching policies for multi-core systems is bottlenecked by day-long, cycle-level simulations. This simulation wall limits even LLM-based microarchitecture discovery; the state-of-the-art ArchAgent system evolves one cache level at a time on single-core benchmarks, adds multi-core configurations only at the end, and trails the human champion under multi-core shared cache contention. We introduce DuoArchitect, an autonomous agentic framework for multi-fidelity, multi-core co-discovery of prefetchers. Agents jointly evolve evaluation proxies and coordinated prefetching policies across the L1D, L2, and LLC hierarchies, constructing simulation proxies 3-26x faster than cluster evaluation an
    
[^159]: 在权重空间中读取、写入与修复博弈论策略

    Reading, Writing and Repairing Game-Theoretic Policies in Weight Space

    [https://arxiv.org/abs/2611.31979](https://arxiv.org/abs/2611.31979)

    该论文构建了一组在权重空间中答案可被精确验证的博弈策略读写任务，证明中等规模开源语言模型能够在1218个权重的循环网络中恢复策略程序、报告其行为并写出最优策略，但修复被篡改网络的能力明显较弱。

    

    语言模型能否在重复博弈策略网络的权重空间中工作？我们向一个中等规模的开源模型提出四个问题，研究对象是具有1218个权重、答案可被精确验证的循环网络。给定一个用英文描述的对手群体，写出一个最优策略：策略的价值是对智能体信息状态的有限计算，因此所提交的网络与最优值之间存在可精确度量的差距。给定一个网络的权重，报告它在十五个对手历史序列中每个历史之后的动作；恢复它所实现的状态机（以深度为十的所有对手序列评判）；或者修复一个输出在某一处被篡改的网络，并指出从哪一轮开始首次出错。该模型在66个回复中的62个成功恢复了程序，在36次中有34次正确报告了全部十五个行为条目，并且当它给出的策略可被评估器编译时，在20次中全部20次达到最优。修复是例外情况：18个中仅4个被成功修复，18个中5个诊断出故障（原摘要在此处截断）。

    arXiv:2611.31979v1 Announce Type: cross  Abstract: Can a language model work in the weight space of a policy network for a repeated game? We put four questions to a mid-size open model, on recurrent networks of 1{,}218 weights with exactly checkable answers. Given a population of opponents described in English, write a policy that is optimal: its value is a finite computation over the agent's information state, so a submitted network has an exact gap from the optimum. Given a network's weights, report its action after each of fifteen opponent histories; recover the state machine it implements, judged on every opponent sequence to depth ten; or repair a network whose output was altered in one place and say which round first goes wrong. The model recovers the program in 62 of 66 replies, reports all fifteen behavioral entries correctly in 34 of 36, and attains the optimum in 20 of 20 when it states a policy the evaluator compiles. Repair is the exception: 4 of 18 repaired, 5 of 18 diagno
    
[^160]: 蒸馏成循环：将预训练扩散Transformer转化为深度可调的循环模型

    Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth

    [https://arxiv.org/abs/2611.31977](https://arxiv.org/abs/2611.31977)

    提出Dooop方法，将预训练的非循环扩散Transformer通过定点迭代蒸馏为循环模型，实现测试时任意可调的循环深度而无需重新训练。

    

    循环Transformer沿着网络深度方向复用一个共享权重的块。尽管已经存在强大的预训练非循环扩散Transformer，但近期的循环扩散Transformer仍是从零开始训练的，这重复了完整的预训练成本。我们提出了Distill to Loop（Dooop），将预训练的非循环扩散Transformer蒸馏成一个循环学生模型，其块由教师模型的块初始化。我们将蒸馏过程建模为朝向教师模型的定点迭代，其中每次循环后的输出都朝向教师模型进行训练，且每次循环相当于深度方向上常微分方程（ODE）的一步。因此，一个模型在测试时可以运行任意次数的循环而无需重新训练，并且扩散模型采样中的高阶ODE求解器可以沿深度维度应用以获得更好的结果。同时，由于循环模型在每次循环后都会产生输出，我们还在……之间构建了循环引导。

    arXiv:2611.31977v1 Announce Type: cross  Abstract: Looped transformers reuse one block of shared weights along the depth of the network. Recent looped diffusion transformers are trained from scratch, which repeats the full pretraining cost although strong pretrained non-looped diffusion transformers exist. We propose Distill to Loop (Dooop), which distills a pretrained non-looped diffusion transformer into a looped student whose blocks are initialized from the teacher's. We cast distillation as a fixed-point iteration toward the teacher, in which the output after every loop is trained toward the teacher, and each loop is one step of an ordinary differential equation in depth. As a result, one model runs any number of loops at test time without retraining, and the higher-order ODE solvers of diffusion model sampling can be applied along the depth dimension for better results. Meanwhile, because a looped model produces an output after every loop, we also construct loop guidance between t
    
[^161]: Coding4Games：现代编程智能体能玩现代电子游戏吗？

    Coding4Games: Can Modern Coding Agents Play Modern Video Games?

    [https://arxiv.org/abs/2611.31974](https://arxiv.org/abs/2611.31974)

    该论文提出了Coding4Games基准，评估编程智能体能否开发出无需在游玩过程中调用语言模型、仅依靠视觉观察和标准游戏输入就能独立运行并游玩15款多样化现代游戏的程序。

    

    游戏，尤其是现代3A大作，对语言模型提出了挑战：它们需要感知复杂场景、控制动作，并在长时程中做出决策。现有的游戏基准主要将语言模型作为计算机使用智能体来测试，即在整个游玩过程中持续查询语言模型，这会带来反复的推理成本，并且在实时游戏中，智能体需要在环境持续演化的同时采取行动。我们提出一个问题：编程智能体能否从游戏交互中学习，开发出无需语言模型推理即可独立运行的游戏程序？为此，我们提出了Coding4Games，这是一个用于评估编程智能体能否开发出通过视觉观察和标准游戏输入来游玩现代游戏的独立程序的基准。Coding4Games具有三个核心特点：1）多样化的现代游戏，涵盖15款在视觉、策略和控制需求上各不相同的游戏；2）开发与测试分离，即编程智能体使用沙盒和标准玩家界面（原文摘要在此处截断）……

    arXiv:2611.31974v1 Announce Type: new  Abstract: Games, especially modern AAA games, challenge language models to perceive complex scenes, control actions, and make decisions over long horizons. Existing game benchmarks primarily test language models as computer-use agents that query a language model throughout play, incurring repeated inference cost and, in real-time games, acting while the environment continues to evolve. We ask whether coding agents can learn from game interaction to develop standalone game-playing programs that run without language model inference. We thus introduce Coding4Games, a benchmark for evaluating whether coding agents can develop standalone programs that play modern games through visual observations and standard game inputs. Coding4Games has three defining features: 1) Diverse Modern Games, covering 15 games with diverse visual, strategic, and control demands; 2) Development-Test Separation, in which a coding agent uses a sandbox and the standard player i
    
[^162]: DexOPD：面向扩散模型的双外推在线策略蒸馏

    DexOPD: Dual-Extrapolation On-Policy Distillation for Diffusion Models

    [https://arxiv.org/abs/2611.31969](https://arxiv.org/abs/2611.31969)

    提出DexOPD双外推在线策略蒸馏方法，在教师外推监督的基础上引入以滞后历史学生为参考的学生自我外推机制，从而充分利用学生自身训练进展，使扩散模型蒸馏效果超越仅依赖教师外推的现有方法。

    

    基于流的扩散模型已成为文本到图像生成的主流范式，通过任务特定的后训练可以产生具有互补优势的专家模型。多教师在线策略蒸馏（OPD）提供了一种将这些专家模型高效整合为单一统一学生模型的方法。近期的方法通过教师外推来增强监督，构建更强的代理教师，使蒸馏后的学生模型能够超越原始专家模型。然而，这些方法的外推方向仅由冻结的教师模型和参考模型定义，学生模型自身的训练进展未被充分利用。为解决这一局限，我们提出了DexOPD，一种面向扩散模型的双外推在线策略蒸馏方法。DexOPD在保留外推教师监督的同时，利用一个滞后的历史学生模型作为参考进行学生自我外推，并在多个自我外推……（原文摘要在此处截断）

    arXiv:2611.31969v1 Announce Type: new  Abstract: Flow-based diffusion models have become a mainstream paradigm for text-to-image generation, where task-specific post-training produces specialist models with complementary strengths. Multi-teacher on-policy distillation (OPD) offers an efficient way to consolidate these specialists into a single unified student. Recent methods strengthen supervision through teacher extrapolation, constructing stronger proxy teachers that allow distilled students to surpass the original specialists. However, their extrapolation directions remain defined solely by frozen teacher and reference models, leaving the student's own training progress unexploited. To address this limitation, we introduce DexOPD, a dual-extrapolation on-policy distillation method for diffusion models. DexOPD retains extrapolated teacher supervision while leveraging a lagging historical student as the reference for student self-extrapolation. Sampling across multiple self-extrapolat
    
[^163]: 网络智能体还能被检测到吗？用协同规避手段对机器人防御系统进行压力测试

    Are Web Agents Still Detectable? Stress-Testing Bot Defenses with Coordinated Evasion

    [https://arxiv.org/abs/2611.31962](https://arxiv.org/abs/2611.31962)

    论文提出了DarkAgent自主网络智能体，通过结合隐藏自动化痕迹的Chromium浏览器与联合生成逼真按键事件及时间间隔的类人鼠标键盘模型，实现跨浏览器指纹、自动化控制和交互行为的协同规避，实验表明现有机器人防御系统难以检测此类攻击。

    

    AI网络智能体能够自主浏览网站并完成各类任务。随着其能力和应用的不断扩展，评估机器人防御系统需要开展对抗性压力测试。尽管先前的研究表明现有智能体仍可被检测到，但目前尚不清楚：面对那些在浏览器指纹、自动化控制以及交互行为等多个层面协同规避检测的攻击者时，机器人防御系统的实际效力究竟如何。为了研究这一威胁，我们提出了DarkAgent——一个自主网络智能体，它结合了一个经过改造、可隐藏自动化痕迹的Chromium浏览器分支，以及能够生成类人鼠标和键盘轨迹的生成模型。针对现有键盘模拟器的局限性，我们的键盘模型在可靠重现目标文本的同时，联合生成逼真的按键事件及其时间间隔。已发布的智能体检测器会将我们生成的行为判定为人类，而经过训练的分类器也发现要识别这些行为要困难得多……

    arXiv:2611.31962v1 Announce Type: cross  Abstract: AI web agents can autonomously navigate websites and complete various tasks. As their capabilities and adoption expand, evaluating bot defenses requires adversarial stress tests. Although prior work demonstrates that existing agents can remain detectable, it remains unclear how effective bot defenses are against adversaries that coordinate evasion across browser fingerprints, automation controls, and interaction behavior. To investigate this threat, we introduce DarkAgent, an autonomous web agent that combines a forked Chromium browser that conceals automation artifacts and generative models that produce human-like mouse and keyboard traces. To address the limitations of existing keyboard mimickers, our keyboard model jointly generates realistic key events and their timings while reliably reproducing the target text. Released agent detectors accept our generated behavior as human, and trained classifiers find it substantially harder to
    
[^164]: 恰当尺度下的感知已足够：面向视觉-语言-动作模型的尺度自适应Token剪枝

    Perception Is Enough, at the Right Scale: Scale-Adaptive Token Pruning for Vision-Language-Action Models

    [https://arxiv.org/abs/2611.31954](https://arxiv.org/abs/2611.31954)

    提出免训练的TauPrune-VLA方法，通过单一感知温度参数τ并借助感知语义趋势、机器人状态变化和动作幅度在线自适应调节token剪枝的语义尺度，从而降低VLA模型的计算成本并支持实时机器人控制。

    

    视觉-语言-动作（VLA）模型必须处理连续的视觉观测流，由此产生的计算成本限制了实时机器人控制。Token剪枝可以降低这一成本，但固定语义尺度的感知表示可能会遗漏操作不同阶段所需的信息。我们证明，当选择尺度能够适应场景不断演化的语义结构时，感知语义可以在不依赖内部注意力的情况下引导有效的token选择。基于这一观察，我们提出了TauPrune-VLA，这是一种免训练方法，通过单一的感知温度参数τ来控制所保留感知表示的尺度。为了在操作过程中自适应调整该尺度，TauPrune-VLA利用感知语义的变化趋势在线更新τ，并以机器人状态变化和动作幅度来调节调整强度。虽然这些语义在操作过程中不断演化……（原文摘要在此处截断）

    arXiv:2611.31954v1 Announce Type: new  Abstract: Vision-language-action (VLA) models must process continuous streams of visual observations, and the resulting computational cost limits real-time robotic control. Token pruning reduces this cost, but a fixed semantic scale of perceptual representation can overlook information needed at different stages of manipulation. We show that perceptual semantics can guide effective token selection without relying on internal attention when the selection scale adapts to the scene's evolving semantic structure. Based on this observation, we introduce TauPrune-VLA, a training-free method that controls the scale of the retained perceptual representation through a single perception temperature $\tau$. To adapt this scale as manipulation proceeds, TauPrune-VLA updates $\tau$ online using trends in perceptual semantics, with robot-state changes and action magnitude regulating the adjustment strength. While these semantics evolve over the course of manipu
    
[^165]: 超越刚性手部：将皮肤变形为接触一致的手部网格

    Beyond Rigid Hands: Deforming Skin into Contact-Consistent Hand Meshes

    [https://arxiv.org/abs/2611.31950](https://arxiv.org/abs/2611.31950)

    该论文提出一种优化方法，通过微调关节姿态并将皮肤变形贴合到物体表面且保持时间一致性，将刚性手部网格提升为接触一致的手部网格，并训练轻量级预测头来自动预测变形区域。

    

    双手通过柔软的皮肤抓握物体：指尖贴在马克杯上会变平，手掌环绕瓶子时会弯曲。然而，现有的手-物体数据集和估计器将手表示为刚性铰接网格，在皮肤本应紧贴物体的地方，网格会悬浮在物体上方或穿透物体。现有的细化方法保持网格刚性、仅调整姿态，以姿态精度换取物理合理性。我们提出这样一个问题：刚性手能否像皮肤那样接触物体，同时保持接近其原始姿态。我们通过一种优化方法将带标注或估计的刚性手提升为接触一致的网格，该优化方法：，微调关节和手腕，使手指贴合到物体上， 将皮肤变形，使穿透区域沉降到物体表面上， 保持该变形在时间上的一致性。由于变形仅作用于已靠近物体的皮肤区域，我们进一步在提升得到的标注上训练一个轻量级预测头，以预测哪里（摘要在此处截断）

    arXiv:2611.31950v1 Announce Type: cross  Abstract: Hands grasp objects with soft skin. A fingertip flattens against a mug, and a palm curves around a bottle. Yet hand-object datasets and estimators represent the hand as a rigid articulated mesh, which hovers above or penetrates the object where skin would press against it. Existing refiners keep the mesh rigid and adjust only the pose, trading pose accuracy for physical plausibility. We ask whether a rigid hand can instead touch the object as skin does while staying close to its original pose. We lift annotated or estimated rigid hands to contact-consistent meshes through an optimization that (i) slightly adjusts the joints and wrist to bring the fingers onto the object, (ii) deforms the skin so that penetrating regions settle onto the surface, and (iii) keeps this deformation consistent over time. Since deformation reaches only skin already near the object, we further train a lightweight head on the lifted annotations to predict where
    
[^166]: CAPE：协同校准与搜索的工具使用策略编译

    CAPE: Coordinating Calibration and Search for Tool-Use Policies

    [https://arxiv.org/abs/2611.31938](https://arxiv.org/abs/2611.31938)

    CAPE是一个无需训练的工具使用策略编译器，它通过协调校准与搜索，始终保持可执行策略，并将校准资源集中于能改进该策略的调用上，从而高效应对配置不可见工具导致的调用失败问题。

    

    即使工具调用背后的操作是正确的，该调用也可能失败，因为接收方工具会在代理无法查看的配置下解析分隔符、日期或记录布局。为此类工具编译工具使用策略既需要校准，即在候选配置上评估候选调用，也需要在已评估的调用上进行搜索。这两者共享同一个编译预算：如果先把所有内容都校准好，任何改进后的策略都要推迟到最后才能产生；而如果对每个部分已知的模型都重新求解，则会延误下一个有用的查询。一个调用只要在能够到达它的配置上的效果是已知的，它就可以被使用。基于这一观察，我们提出了CAPE，一个无需训练的编译器。它在任何时刻都保持一个可执行的策略，将校准引导至能够改进该策略的调用上，并且只在能够提前结束校准的情况下收紧边界。返回的策略保留了初始策略所完成的每一个配置。

    arXiv:2611.31938v1 Announce Type: new  Abstract: A tool call can fail even when the operation behind it is correct, because the receiving tool parses delimiters, dates or record layouts under a configuration the agent cannot inspect. Compiling a tool-use policy for such a tool requires both calibration, which evaluates candidate calls on candidate configurations, and search over the evaluated calls. The two share one compilation budget: calibrating everything first postpones any improved policy until the end, while re-solving every partially known model delays the next useful query. A call becomes usable as soon as its effects are known on the configurations that reach it. We present CAPE, a training-free compiler built on this observation. It maintains an executable policy at all times, directs calibration to the calls that can improve this policy, and tightens bounds only where doing so ends calibration early. Returned policies preserve every configuration the initial policy complete
    
[^167]: 面向大语言模型遗忘的对比式在策自蒸馏

    Contrastive On-Policy Self-Distillation for LLM Unlearning

    [https://arxiv.org/abs/2611.31932](https://arxiv.org/abs/2611.31932)

    本文提出CPSD框架，通过构建显式的知识缩减目标分布，并沿模型自身的在策轨迹进行对比自蒸馏，实现大语言模型在遗忘指定数据的同时保留无关能力。

    

    大语言模型遗忘旨在移除指定训练数据对模型的影响，同时保留无关的模型能力。现有的基于似然的方法通常针对固定的遗忘序列进行优化，仅对模型不应生成的内容提供负向监督，而忽略了不断演化的遗忘策略所经过的状态。我们提出了对比式在策自蒸馏（CPSD），这是一种新的遗忘框架，它构建一个显式的知识缩减目标分布，并沿模型自身的轨迹对其进行蒸馏。给定学生模型的一次 rollout，冻结的参考模型分别在普通上下文和包含遗忘信息的特权上下文下评估每个 token。CPSD 将由此产生的分布偏移视为与知识关联的方向，并将其反转以获得反事实遗忘目标。随后训练学生模型在每个 rollout 位置匹配该目标，同时保留无关能力（摘要原文在此处被截断）。

    arXiv:2611.31932v1 Announce Type: new  Abstract: Large language model unlearning aims to remove the influence of designated training data while preserving unrelated model capabilities. Existing likelihood-based methods commonly optimize against fixed forget sequences, providing only negative supervision on what the model should not generate and overlooking the states visited by the evolving unlearned policy. We introduce Contrastive On-Policy Self-Distillation (CPSD), a new unlearning framework that constructs an explicit knowledge-reduced target distribution and distills it along the model's own trajectories. Given a student rollout, a frozen reference model evaluates each token under a plain context and a privileged context containing the forget information. CPSD treats the resulting distributional shift as a knowledge-associated direction and reverses it to obtain a counterfactual unlearning target. The student is then trained to match this target at every rollout position while ret
    
[^168]: 机器学习使自旋极端质量比旋近的完整贝叶斯推断成为可能

    Machine learning enables full Bayesian inference for spinning extreme-mass-ratio inspirals

    [https://arxiv.org/abs/2611.31916](https://arxiv.org/abs/2611.31916)

    该研究首次借助生成式机器学习网络缩小完整先验并结合精确探测器似然函数，实现了对自旋极端质量比旋近（EMRI）全部14个参数的联合贝叶斯后验推断。

    

    极端质量比旋近（EMRI）是指恒星级致密天体螺旋坠入大质量黑洞的过程，通过毫赫兹引力波观测，这类系统有望测量中心黑洞的质量与自旋、揭示其周围的强场环境，并用于探测宇宙距离。然而，大多数针对这些系统参数推断的预测研究都假设源参数已知，而采用宽先验的分析也仅处理了该问题的简化版本。从完整先验出发对自旋源的全部14个参数进行联合反演，这一问题一直悬而未决。在本文中，我们针对增广解析杂凑（augmented analytic kludge）模型解决了这一问题：生成式网络缩小了完整先验的范围，并揭示出一个自旋方向简并的环，随后精确的探测器似然函数确定出后验分布。对于模拟的一年期LISA-Taiji联合观测，我们在一个环扇区内首次获得了自旋EMRI的14参数后验分布……（原文摘要在此处被截断）

    arXiv:2611.31916v1 Announce Type: cross  Abstract: Extreme-mass-ratio inspirals (EMRIs), stellar-mass compact objects spiralling into massive black holes, promise to measure the masses and spins of the central black holes, reveal the strong-field environment around them and gauge cosmic distances via millihertz gravitational-wave observations. Yet most forecasts for the parameter inference of these systems assume known source parameters, and broad-prior analyses have addressed only simplified versions of the problem. The joint inversion of all 14 parameters of a spinning source from the complete prior has remained unsolved. Here we solve it for the augmented analytic kludge model: generative networks narrow the complete prior and reveal a ring of degenerate spin directions, and the exact detector likelihood determines the posterior. For a simulated one-year LISA-Taiji observation, within one ring sector we obtain, for the first time, a 14-parameter posterior of a spinning EMRI from the
    
[^169]: 你的视觉语言模型知道该检查什么：面向GUI定位的语义探针引导自适应聚焦

    Your VLM Knows What to Inspect: Semantic Probe-Guided Adaptive Focusing for GUI Grounding

    [https://arxiv.org/abs/2611.31909](https://arxiv.org/abs/2611.31909)

    该论文提出免训练框架SPARK，利用VLM自身的语义探针（探索与验证）引导自适应的视图放置与裁剪，将模型“知道该检查什么”的语义能力转化为针对性视觉观察，从而提升GUI定位精度。

    

    GUI（图形用户界面）定位需要在赋予其意义的上下文中解析精细的视觉细节。额外的视图可以提升定位效果，但其价值取决于它们暴露了什么证据以及保留了多少上下文。我们探索了视觉语言模型（VLM）的一种能力：即使其坐标预测不正确，它们也能描述值得检查的视觉证据。我们提出了SPARK（语义探针引导的自适应区域探寻），这是一个免训练框架，能将这种语义引导转化为有针对性的视觉观察。探索性探针描述相关上下文和区分目标细节，而验证性探针识别评估涌现假设所需的证据。它们的空间注意力引导视图放置，上下文响应决定裁剪范围，观察历史则鼓励互补性检查。

    arXiv:2611.31909v1 Announce Type: cross  Abstract: GUI grounding requires resolving fine visual details within the context that gives them meaning. Additional views can improve localization, but their value depends on what evidence they expose and how much context they preserve. We explore a capability of vision-language models (VLMs): even when their coordinate predictions are incorrect, they can describe visual evidence worth inspecting. We introduce \textbf{SPARK}, \textit{\textbf{S}emantic \textbf{P}robe-guided \textbf{A}daptive \textbf{R}egion see\textbf{K}ing}, a training-free framework that turns this semantic guidance into targeted visual observations. Exploration probes describe relevant context and distinguishing target details, while verification probes identify evidence needed to assess emerging hypotheses. Their spatial attention guides view placement, contextual responses determine crop extent, and observation history encourages complementary inspection. The resulting pre
    
[^170]: MIFLOW：用于参考校准在线流策略强化学习的动态近期行为先验

    MIFLOW: Dynamic Recent-Behavior Priors for Reference-Calibrated Online Flow-Policy Reinforcement Learning

    [https://arxiv.org/abs/2611.31902](https://arxiv.org/abs/2611.31902)

    MIFLOW通过用拟合最近执行动作的动态流匹配先验作为移动密度参考，对在线流策略强化学习的演员更新和评论家目标进行维度归一化校正，在抑制低密度不可靠动作的同时保持探索能力，从而在MuJoCo连续控制任务上取得有竞争力的性能。

    

    基于流的策略为连续控制强化学习提供了富有表现力的动作分布。在在线演员-评论家（actor-critic）学习中，当动作覆盖不足时，离策略价值估计可能不可靠，此时富有表现力的策略可能会将概率质量转移到评论家偏好的动作上。我们提出MIFLOW，该方法通过一个与最近执行动作拟合的、随时间自适应更新的流匹配先验来校准在线流策略学习。受“渠化”（canalization）概念作为路径依赖稳定化的定性类比启发，我们将该先验用作价值引导策略改进的移动密度参考。MIFLOW在演员更新和评论家目标中均采用近似近期动作对数密度进行维度归一化校正。这种校正提高了低密度动作的成本，同时保持这些动作仍然可用。在六个MuJoCo连续控制任务和五个随机种子上，MIFLOW取得了具有竞争力的平均回报曲线。

    arXiv:2611.31902v1 Announce Type: cross  Abstract: Flow-based policies provide expressive action distributions for continuous-control reinforcement learning. In online actor-critic learning, off-policy value estimates can be unreliable in poorly covered action regions, where expressive policies may shift probability mass toward critic-favored actions. We propose MIFLOW, which calibrates online flow-policy learning with a recency-adaptive flow-matching prior fitted to recently executed actions. Inspired by canalization as a qualitative analogy for path-dependent stabilization, we use the prior as a moving density reference for value-guided policy improvement. MIFLOW uses the approximate recent-action log-density in a dimension-normalized correction to both actor updates and critic targets. This correction increases the cost of low-density actions while keeping them available. Across six MuJoCo continuous-control tasks and five random seeds, MIFLOW achieves competitive mean return curves
    
[^171]: Loupe：强化知识密集型视频生成智能体

    Loupe: Reinforcing Knowledge-Intensive Video Generation Agents

    [https://arxiv.org/abs/2611.31895](https://arxiv.org/abs/2611.31895)

    该论文提出 Loupe 视频生成智能体，通过检索并验证文本与视觉证据来构建带参考图像的时间化视频提示词，并创新性地引入基于量规的信用评估机制（RBC），在视频渲染前对中间搜索与规划决策进行评估，从而强化知识密集型视频生成。

    

    知识密集型视频生成仍然具有挑战性，因为提示词往往省略了准确生成所需的事实、外观和时间关系。虽然基于搜索的工作流可以检索缺失的证据，但人工设计的策略限制了自适应规划，而传统训练则面临代价高昂的视频级反馈以及粗粒度的轨迹级信用分配问题。在本文中，我们提出了 Loupe，一个视频生成智能体，它通过检索并验证文本与视觉证据，构建以证据为支撑、带有参考图像的时间化视频提示词。我们进一步引入了基于量规的信用评估机制，利用提示词特定的事实、视觉和时间要求，在视频渲染之前评估中间的搜索与规划决策。在 RBC 的基础上，RBC-SFT 根据步骤质量对助手 token 的学习进行重新加权，而 RBC-GRPO 则将步骤级信用与最终规划奖励相结合……

    arXiv:2611.31895v1 Announce Type: cross  Abstract: Knowledge-intensive video generation remains challenging because prompts often omit the facts, appearances, and temporal relationships required for accurate generation. While search-based workflows can retrieve missing evidence, manually designed strategies limit adaptive planning, and conventional training suffers from costly video-level feedback and coarse trajectory-level credit assignment. In this paper, we present Loupe, a video generation agent that retrieves and verifies textual and visual evidence to construct evidence-grounded temporal video prompts with reference images. We further introduce Rubric-Based Credit (RBC), which leverages prompt-specific factual, visual, and temporal requirements to evaluate intermediate search and planning decisions before video rendering. Building on RBC, RBC-SFT reweights assistant-token learning by step quality, while RBC-GRPO incorporates step-level credits alongside a final planning reward, 
    
[^172]: 便宜的内存代价是未来的计算：协同设计持久KV缓存与GPU执行以实现高效流式视频扩散

    Cheap Memory Costs Future Compute: Co-Designing Persistent KV and GPU Execution for Efficient Streaming Video Diffusion

    [https://arxiv.org/abs/2611.31892](https://arxiv.org/abs/2611.31892)

    提出了无需训练的SAMR方法，通过步骤感知记忆重建部分刷新KV缓存并无需DiT前向传播地重建其余部分，解决了流式视频扩散中干净KV更新成为主要运行时开销、而复用过期KV又会损害下游生成质量的核心矛盾。

    

    块因果视频扩散通过逐块生成的方式生成长视频，并以持久化键值缓存作为条件。每个块去噪完成后，需要额外进行一次无噪声（t=0）的前向传播来将其干净的KV写入缓存；一旦蒸馏和推测性提前退出使去噪变得廉价，这种干净KV更新便成为运行时的主要开销。复用过期的KV可以缩短更新时间，但陈旧的缓存会破坏后续块的条件生成，并增加其去噪步数，因此节省的计算量会在下游重新出现。我们提出了SAMR（步骤感知记忆重建），这是一种面向冻结因果视频DiT的无需训练的方法，它只刷新部分缓存，并无需DiT前向传播即可重建其余部分。我们的关键观察是，在去除旋转位置相位后，运动对齐的历史KV与token在其退出步骤所持有的进行中KV是其干净KV的互补估计。SAMR使用离线校准的per（摘要在此处被截断）

    arXiv:2611.31892v1 Announce Type: new  Abstract: Block-causal video diffusion generates long videos block by block, conditioned on a persistent key-value (KV) cache. After each block is denoised, an extra noise-free ($t=0$) forward pass writes its clean KV into the cache; once distillation and speculative early exit make denoising cheap, this clean KV update dominates the runtime. Reusing stale KV shortens the update, but stale memory corrupts the conditioning of later blocks and inflates their denoising steps, so the saved compute reappears downstream. We present SAMR (Step-Aware Memory Reconstruction), a training-free method for frozen causal video DiTs that refreshes only part of the cache and reconstructs the rest without DiT forwards. Our key observation is that, once rotary position phases are removed, motion-aligned historical KV and the in-flight KV a token holds at its exit step are complementary estimates of its clean KV. SAMR debiases in-flight KV with offline-calibrated per
    
[^173]: 听、看、单次脉冲：用于低精度多模态视听分类与在线学习的高效神经AI时序神经网络

    HEAR, SEE, SPIKE ONCE: Highly Efficient NeuroAI Temporal Neural Networks for Low-Precision Multimodal Audio-Visual Classification and Online Learning

    [https://arxiv.org/abs/2611.31884](https://arxiv.org/abs/2611.31884)

    本文提出NeuTNN-AV，一种采用单次脉冲时间编码、4位整数权重和局部奖励调制STDP学习规则的高效多模态视听神经网络，能够以远少于现有SNN的突触数量达到相当或更优的分类精度。

    

    面向多模态视听分类的脉冲神经网络（SNN）通常通过采用深度学习机制来达到高精度：即采用多阶段卷积或Transformer编码器，并通过时间维度上的代理梯度反向传播进行端到端训练。这些先前的工作通常依赖于全精度（32位）参数，且并未将硬件效率作为主要设计目标。在本文中，我们提出了NeuTNN-AV，这是一个基于近期提出的神经AI时序神经网络（NeuTNN）构建的、结构更简单且适合硬件实现的多模态计算基底，并证明它能够以仅占基线一小部分的突触数量，达到与最先进SNN基线相当或更优的性能。NeuTNN神经元携带多个活跃的树突段。每条输入线路最多只发放一次脉冲（时间编码），每个权重均通过局部的奖励调制脉冲时序依赖可塑性（STDP）规则进行学习。所有权重均为4位整数，推理过程仅使用比较操作。

    arXiv:2611.31884v1 Announce Type: new  Abstract: Spiking neural networks for multimodal audio-visual classification reach high accuracy by adopting deep-learning machinery: multi-stage convolutional or transformer encoders trained end-to-end with surrogate-gradient backpropagation through time. These prior works typically rely on full-precision (32-bit) parameters and do not treat hardware efficiency as a primary design objective. In this paper, we propose NeuTNN-AV, a much simpler hardware-amenable multimodal substrate built on the recently proposed NeuroAI Temporal Neural Network (NeuTNN) and demonstrate that it can match or outperform state-of-the-art SNN baselines with a fraction of their synapse counts. NeuTNN neurons carry multiple active dendritic segments. Each input line spikes at most once (temporal coding) and every weight is learned via a local reward-modulated spike-timing dependent plasticity (STDP) rule. All weights are 4-bit integers and inference uses only comparisons 
    
[^174]: 重新思考视频智能体：视频工具框架何时以及为何有效

    Rethinking Video Agents: When and Why Video Harnesses Work

    [https://arxiv.org/abs/2611.31883](https://arxiv.org/abs/2611.31883)

    该研究利用Video-Relook实验框架系统探索了视频智能体工具框架何时以及为何有效，发现自适应的时间与空间观察策略能显著提升细粒度动作理解能力，且其收益由任务需求与模型的交互能力共同决定。

    

    视频理解不仅取决于模型如何对视觉证据进行推理，还取决于它们观察到哪些证据。固定的观察策略会预先确定这些证据；而智能体式工具框架则允许模型随着理解的深入主动寻求更多观察。这一转变创造了释放固定输入所未能挖掘的潜力的机会，然而，请求观察的能力并不保证这些观察能被有效利用。我们研究了视频工具框架何时以及为何能改善理解，考察任务需求与模型的交互能力如何共同塑造其收益。我们以Video-Relook作为实验框架，让模型从统一的默认配置出发自适应地调整其时间维度和空间维度上的观察，并分析跨模型、跨任务的性能表现与交互轨迹。自适应观察在细粒度动作理解上带来了显著收益，其收益还延伸至……（原文摘要截断于此）

    arXiv:2611.31883v1 Announce Type: cross  Abstract: Video understanding depends not only on how models reason about visual evidence, but also on which evidence they observe. Fixed observation policies determine this evidence in advance; agentic harnesses allow models to seek further observations as their understanding develops. This shift creates an opportunity to realize capabilities that fixed inputs leave untapped, yet the ability to request observations does not ensure their effective use. We investigate when and why video harnesses improve understanding, examining how task demands and models' capacity for interaction jointly shape their benefits. Using Video-Relook as an experimental framework, we let models adapt their temporal and spatial observations from a common default configuration, and analyze performance and interaction trajectories across models and tasks. Adaptive observation yields substantial gains on fine-grained action understanding, with benefits also extending to V
    
[^175]: EEG-Jev：一种面向脑电（EEG）的统一单次前向决策模型

    EEG-Jev: A Unified Single-Pass Decision Model for EEG

    [https://arxiv.org/abs/2611.31875](https://arxiv.org/abs/2611.31875)

    EEG-Jev提出了一种统一的脑电单次前向决策模型，一次前向传播即可完成选项选择、命题判断与有序评分三类任务，并通过先验校正和温度缩放提供含义明确、可校准的概率输出，在多个公开脑电数据集上取得了优异表现。

    

    脑电（EEG）分析涵盖从备选项中选择、对陈述进行判断以及对有序属性进行评分等任务，需要具有明确含义且可校准的概率。我们提出EEG-Jev，一种统一的脑电单次前向决策模型。给定一个脑电窗口、一个问题及答案规范，仅需一次前向传播即可返回类型化的输出：CHOICE给出候选选项分布，NOUL给出命题概率，SCORE给出有序等级分布及期望值。该模型仅训练一个联合检查点，并在35个公开数据源的保留划分上一次性评估。核心原始输出在Stieger2021运动想象任务上达到54.08%的平衡准确率，在Sleep-EDF-SC睡眠分期上达到70.45%，在CHB-MIT癫痫检测上以阈值0.5达到每窗口72.11%的准确率（AUROC为0.825），在0–10分的ds005292疼痛评分上MAE为1.700。先验校正与温度缩放将Sleep-EDF-SC上的负对数似然从0.766降至0.606，在CHB-MIT上从0.166降至0.019，同时直接输出削减……

    arXiv:2611.31875v1 Announce Type: cross  Abstract: EEG analysis spans choosing alternatives, judging statements, and rating ordered attributes, requiring calibratable probabilities with explicit meaning. We present EEG-Jev, a unified single-pass decision model for EEG. Given an EEG window, a question and an answer specification, one forward pass returns a typed readout: CHOICE gives a candidate distribution, NOUL a proposition probability, and SCORE an ordered-level distribution and expectation. One checkpoint is trained jointly and evaluated once on held-out splits of 35 public sources. Core raw outputs reach 54.08% balanced accuracy on Stieger2021 motor imagery, 70.45% on Sleep-EDF-SC sleep staging, 72.11% per window at threshold 0.5 on CHB-MIT seizure detection with 0.825 AUROC, and 1.700 MAE on 0-10 ds005292 pain ratings. Prior correction and temperature scaling lower negative log-likelihood from 0.766 to 0.606 on Sleep-EDF-SC and 0.166 to 0.019 on CHB-MIT, while direct readout cut
    
[^176]: RepoEviAgent：可扩展的证据支撑型仓库级问答任务合成

    RepoEviAgent: Scalable Synthesis of Evidence-Grounded Repository-Level Question Answering Tasks

    [https://arxiv.org/abs/2611.31873](https://arxiv.org/abs/2611.31873)

    该论文提出RepoEviAgent框架，通过质量感知的数据筛选与角色专门化的多智能体协作，从海量开源仓库中可扩展地自动构建证据支撑的仓库级问答任务，并建立了RepoEviQA基准。

    

    真实世界的软件工程（SWE）任务要求智能体理解仓库结构、跨模块依赖关系和运行时行为，这给构建高质量的训练和评估数据带来了重大挑战。为解决这一问题，我们提出了RepoEviAgent，一个可扩展的SWE数据构建框架，它集成了质量感知的数据筛选和角色专门化的多智能体任务合成。从8,477个开源仓库和约248万条拉取请求记录出发，我们采用语义对齐、复杂度约束和多维质量评估来识别有价值的SWE数据。我们进一步开发了基于沙盒的多智能体框架，集成了自动化环境构建、执行验证、仓库探索和证据支撑的问答合成。基于该框架，我们构建了RepoEviQA，一个仓库级证据支撑问答基准，……

    arXiv:2611.31873v1 Announce Type: new  Abstract: Real-world software engineering (SWE) tasks require agents to understand repository structures, cross-module dependencies, and runtime behaviors, posing significant challenges for constructing high-quality training and evaluation data. To address this, we propose RepoEviAgent, a scalable SWE data construction framework that integrates quality-aware data curation and role-specialized multi-agent task synthesis. Starting from 8,477 open-source repositories and approximately 2.48 million pull request records, we employ semantic alignment, complexity constraints, and multidimensional quality assessment to identify valuable SWE data. We further develop a sandbox-based multi-agent framework that integrates automated environment construction, execution validation, repository exploration, and evidence-grounded Q&A synthesis. Based on this framework, we construct RepoEviQA, a benchmark for Repository-Level Evidence-Grounded Question Answering, co
    
[^177]: CALVIN-MAX：执行过程中的变化何时会损害机器人策略？

    CALVIN-MAX: When Does a Mid-Execution Change Hurt a Robot Policy?

    [https://arxiv.org/abs/2611.31868](https://arxiv.org/abs/2611.31868)

    提出 CALVIN-MAX 基准，通过三次对照运行的设计，首次将机器人执行过程中遭遇变化的时机影响与世界已被改变的内在代价区分开来，量化了变化在运动中途到来时对策略性能的额外损害。

    

    机器人经常需要在世界发生变化后继续工作，例如当方块移动或相机偏移时，机器人仍基于之前看到的内容行动。近期研究表明，任务执行中的一次变化会降低所有被测试策略的成功率，但并未将变化时机的影响与世界已被改变本身的代价区分开来，也没有衡量其后产生的损失。我们提出了 CALVIN-MAX，这是一个基于 CALVIN 五指令链的基准测试，包含 7,828 个案例，每个案例对物体、相机、机器人或指令进行一次改变。每个案例比较三次运行，这三次运行在受影响指令开始之前完全相同：一次没有变化，一次变化从该时刻起就已存在，还有一次变化在机器人运动过程中到来。第二次运行与第一次运行的对比给出了世界已被改变的代价，第三次运行与第二次运行的对比则给出了在运动中途遭遇变化的影响。在每策略 1,526 个保留测试案例上，且假设预先固定，每个……（摘要至此截断）

    arXiv:2611.31868v1 Announce Type: cross  Abstract: Robots must often keep working after the world changes, as when a block moves or the camera shifts while the robot still acts on what it saw before. Recent work showed that one change during a task lowers success for every policy tested, but did not separate the effect of its timing from the cost of the changed world, or measure what it costs afterwards. We introduce CALVIN-MAX, a benchmark of 7,828 cases on CALVIN's five-instruction chains, each with one change to the objects, the camera, the robot, or the instruction. Each case compares three runs that are identical until the affected instruction begins: one without the change, one with it in place from that moment, and one in which it arrives while the robot is moving. The second run against the first gives the cost of the changed world, and the third against the second the effect of meeting it mid-motion. On 1,526 held-out cases per policy, with hypotheses fixed in advance, each of
    
[^178]: 没有固定配置是最优的：LLM流水线的上下文与成本感知调优，及其在模式匹配与映射中的应用

    No Fixed Configuration Is Best: Context- and Cost-Aware Tuning of LLM Pipelines, with an Application to Schema Matching and Mapping

    [https://arxiv.org/abs/2611.31864](https://arxiv.org/abs/2611.31864)

    论文以准确率和成本为联合目标对模式匹配与映射LLM流水线进行数值调优，发现调优只能带来有限准确率提升或成本节约，而一个数值调优无法触及的配置选择（如是否将源列搜索限制到单一目标表）对不同任务的准确率影响远大于数值调优，因此不存在对所有任务都最优的固定配置。

    

    多阶段LLM流水线（复合AI系统）的数值参数设置通常是调优的目标。我们针对一个用于模式匹配与映射的LLM流水线，以准确率和成本为联合目标，对其全部数值设置进行了调优，并展示了调优能带来什么、不能带来什么，以及剩余的准确率来自何处。首先，调优能带来成本节约和少量准确率提升。在35个任务上（调优时将整组任务留出作测试），一个调优配置相比默认配置以增加18%的成本换来了2.9个百分点的top-1准确率（HR@1）提升，另一个调优配置则以降低32%的成本保持了默认配置的准确率。其次，准确率还取决于一个数值调优无法触及的选择。将每个源列的搜索限制在一个指定的目标表中，在MIMIC-III到OMOP的任务上使HR@1变化+30个百分点，而在一个由数据库视图构建的任务上变化-52个百分点，这一影响超过了数值调优在十个测试任务中九个上的收益。没有哪种设置对所有任务都合适。

    arXiv:2611.31864v1 Announce Type: new  Abstract: The numeric settings of a multi-stage LLM pipeline (a compound AI system) are the usual target of tuning. We tuned all of them for an LLM pipeline for schema matching and mapping, with accuracy and cost as joint objectives, and we show what this buys, what it cannot buy, and where the remaining accuracy is. First, tuning buys cost and a little accuracy. On 35 tasks, with whole groups of tasks held out from tuning, one tuned configuration gains 2.9 points of top-1 accuracy (HR@1) over the default at 18% more cost, and another keeps the accuracy of the default at 32% lower cost. Second, the accuracy depends on a choice that numeric tuning cannot reach. Restricting the search for each source column to one assigned target table changes HR@1 by +30 points on a MIMIC-III to OMOP task and by -52 points on a task built from database views, more than numeric tuning gains on nine of ten test tasks. Neither setting is right for all tasks: on the la
    
[^179]: 面向厚血涂片图像中疟疾寄生虫检测与虫种鉴别的可解释深度学习：一项初步可行性研究

    Explainable Deep Learning for Malaria Parasite Detection and Species Differentiation in Thick Blood Smear Images: A Pilot Feasibility Study

    [https://arxiv.org/abs/2611.31855](https://arxiv.org/abs/2611.31855)

    该研究提出TRACE-Mal原型，将YOLOv8n-seg检测与六模型软投票集成相结合，并验证了Grad-CAM、LIME和SHAP三种可解释方法之间的一致性与疟原虫虫种判定的正确性显著相关，即被两种及以上方法共同解释的预测准确率更高。

    

    恶性疟原虫（PF）与间日疟原虫（PV）的虫种水平鉴别决定了治疗方案的选择，然而深度学习检测器很少检验其解释结果是否可靠。本研究是对TRACE-Mal的一项初步可行性研究，TRACE-Mal是一个结合YOLOv8n-seg检测与六模型软投票集成的研究原型，我们考察了三种事后解释方法（Grad-CAM、LIME和SHAP）之间的一致性是否与虫种判定的正确性相关。该研究分析了来自公开可用的Lacuna疟疾数据集中250张厚血涂片的2,847个寄生虫裁剪图像。该模型达到了0.79的mAP50分数，集成模型的宏F1为0.84（PF召回率0.86；PV召回率0.74）。对裁剪图像的未调整分析显示，由两种及以上方法解释的预测相比由少于两种方法解释的预测更为准确（86.2% vs 78.9%；差异7.3个百分点；95% CI 3.5–11.1）。

    arXiv:2611.31855v1 Announce Type: new  Abstract: Species-level differentiation of Plasmodium falciparum (PF) and P. vivax (PV) determines treatment, yet deep learning detectors rarely test whether their explanations are reliable. In this pilot feasibility study of TRACE-Mal, a research prototype combining YOLOv8n-seg detection and a six-model soft-voting ensemble, we examined whether agreement among three post-hoc explanation methods (Grad-CAM, LIME and SHAP) is associated with the correctness of species calls. the study analysed 2,847 parasite crops from 250 thick smears within publicly available Lacuna Malaria Datasets. The model achieved an mAP50 score of 0.79 and the ensemble had a macro F1 of 0.84 (PF recall 0.86; PV recall 0.74). Unadjusted analyses of crops revealed that predictions explained by greather than or equal to 2 methods were more accurate (86.2% vs 78.9%; difference 7.3 percentage points; 95% CI 3.5-11.1) compared to those explained by less than 2 methods. As the numb
    
[^180]: 在合成社会模拟中利用大语言模型智能体建模事件条件化的情感表达响应

    Modeling Event-Conditioned Emotional Expression Response with LLM Agents in a Synthetic Social Simulation

    [https://arxiv.org/abs/2611.31851](https://arxiv.org/abs/2611.31851)

    本研究提出一个分层提示框架，通过人口统计、心理和情境线索赋予LLM智能体情感推理能力，使其生成事件条件化的情感响应，并利用大规模社交媒体数据验证了其再现真实情感模式的有效性。

    

    情感是人类行为的核心驱动力，然而大多数行为模型未能捕捉情感在塑造感知、决策和集体行动中的作用。应急响应和公共卫生等应用领域尤其受限于情感感知建模的缺失。尽管大语言模型（LLM）正日益被整合进基于智能体的模型（ABM）中，但很少有研究探讨如何赋予LLM智能体情感推理能力，或验证其再现实证观察到的情感模式的能力。本研究通过引入一个分层提示框架来填补这一空白，该框架利用人口统计、心理和情境线索对LLM智能体进行条件化，从而生成针对现实世界事件的事件条件化情感响应。我们使用源自Social Science One Facebook URL分享数据集的大规模社交媒体反应数据，对所生成的事件条件化情感响应进行了验证。

    arXiv:2611.31851v1 Announce Type: new  Abstract: Emotions are central drivers of human behavior, yet most behavioral models fail to capture their role in shaping perception, decision-making, and collective action. Application areas such as emergency response and public health are especially constrained by the lack of emotion-aware modeling. While large language models (LLMs) are increasingly integrated into agent-based models (ABMs), little work has examined how to endow LLM agents with emotional reasoning or validated their ability to reproduce empirically observed emotional patterns. This study addresses this gap by introducing a hierarchical prompting framework that conditions LLM agents with demographic, psychological, and contextual cues to generate event-conditioned emotional responses to real-world events. We validate the resulting event-conditioned emotional responses against large-scale social media reaction data derived from the Social Science One Facebook URL Shares Dataset,
    
[^181]: 面向基于流的具身基础模型的两阶段后训练

    Two-Stage Post-Training for Flow-Based Embodied Foundation Models

    [https://arxiv.org/abs/2611.31850](https://arxiv.org/abs/2611.31850)

    该论文提出了一种基于流的具身基础模型的两阶段后训练框架，第一阶段通过多教师在线策略蒸馏高效整合专家能力，第二阶段将强化学习与基于采样噪声加权的教师蒸馏相结合，从而突破性能瓶颈、实现超越教师模型的表现。

    

    大规模预训练为具身模型提供了通用能力，但可靠的任务执行仍然需要后训练。在多任务设置中，监督微调（SFT）面临训练与部署之间的分布不匹配问题，而强化学习（RL）则受限于稀疏奖励以及跨任务学习进度不均衡的问题。多教师在线策略蒸馏（MOPD）通过对模型自身轨迹的密集监督来整合专家能力，但要实现超越教师的性能仍然具有挑战性。我们提出了一种针对基于流的具身模型的两阶段后训练框架，从条件流匹配的速度回归形式中构建蒸馏目标。阶段1通过MOPD高效整合专家能力。阶段2将强化学习与教师蒸馏相结合，根据采样噪声对速度误差进行加权，其中蒸馏被限制在（原文在此截断）……

    arXiv:2611.31850v1 Announce Type: cross  Abstract: Large-scale pretraining provides embodied models with general capabilities, but reliable task execution still requires post training. In multitask settings, supervised fine-tuning (SFT) faces distribution mismatch between training and deployment, whereas reinforcement learning (RL) is limited by sparse rewards and uneven learning progress across tasks. Multi-teacher on policy distillation (MOPD) consolidates expert capabilities through dense supervision on the model's own trajectories, but achieving performance beyond the teachers remains challenging. We propose a two-stage post training framework for flow-based embodied models, constructing distillation objectives from the velocity regression formulation of conditional flow matching. Stage 1 efficiently consolidates expert capabilities through MOPD. Stage 2 combines RL with teacher distillation that weights velocity errors according to the sampling noise, where distillation is restric
    
[^182]: JECT：解决胸部CT联合嵌入预测学习中的上下文捷径问题

    JECT: Addressing the Context Shortcut in Joint-Embedding Predictive Learning for Chest CT

    [https://arxiv.org/abs/2611.31849](https://arxiv.org/abs/2611.31849)

    JECT通过让放射学报告同时充当预测器的条件输入和编码器的预测目标，解决了胸部CT自监督联合嵌入预测学习中“上下文捷径”导致编码器忽略病灶的问题。

    

    联合嵌入预测架构通过从可见上下文预测被掩码区域的特征来进行学习。在胸部计算机断层扫描（CT）中，决定诊断的关键发现（如小结节）往往无法从周围的解剖结构中预测出来。此时，被隐藏的病灶最容易被预测为正常解剖结构，预测损失不再激励编码器去表征该病灶。我们将这种现象称为“上下文捷径”。利用放射学报告对预测器进行条件化可以修复预测问题，但在我们的设置中，编码器相对于仅使用图像的预训练几乎没有提升：因为报告直接提供了编码器原本无需学习的内容。因此，JECT赋予报告两个角色：一是作为预测器的条件，二是通过掩码分节文本预测，作为编码器的预测目标——编码器必须从未掩码的扫描图像中预测出报告的解剖学分节内容。一个在每个阶段都进行预测的分层三维编码器使这些临床内容得以保留在学习表征中。

    arXiv:2611.31849v1 Announce Type: cross  Abstract: Joint-embedding predictive architectures learn by predicting features of masked regions from visible context. In chest computed tomography, the findings that decide a diagnosis, such as a small nodule, are often not predictable from the surrounding anatomy. A hidden finding is then best predicted toward normal anatomy, and the prediction loss stops rewarding the encoder for representing it. We call this the context shortcut. Conditioning the predictor on the radiology report repairs the prediction, but in our setup the encoder gains little over image-only pretraining: the report supplies what the encoder never had to learn. JECT therefore gives the report two roles: a condition for the predictor and, through masked section-text prediction, a target for the encoder, which must predict the report's anatomical sections from the unmasked scan. A hierarchical three-dimensional encoder that predicts at every stage keeps this clinical content
    
[^183]: 通过双流精炼实现不一致性下的忠实人格引导

    Faithful Persona Steering under Incongruity via Dual-Stream Refinement

    [https://arxiv.org/abs/2611.31848](https://arxiv.org/abs/2611.31848)

    该论文提出QuirkyMind框架，通过特质锚定构建融合句子级摘要与token级序列的双流潜在状态，并利用判别式InfoNCE损失与生成式交叉熵损失的交替精炼目标，实现大语言模型在不一致人格下（稳定特质与情境特异性怪癖并存）的忠实个性化引导。

    

    标准的大语言模型个性化通常将身份视为静态检索任务，忽视了人类人格固有的不一致性，即稳定特质与不合常规的、情境特定的立场并存。现有方法难以调和这两个维度：提示方法在长序列中容易受到上下文漂移的影响，而微调往往会压制个性化的“怪癖”，转而偏向通用的分布模式。为弥合这一差距，我们提出了QuirkyMind，一个将身份定义与其表达解耦的框架。首先，特质锚定构建了一个双流潜在状态，将用于语义稳定性的句子级摘要与用于生成控制的token级序列融合在一起。该状态通过上下文叙事精炼得到稳定，采用交替目标：判别式InfoNCE损失将人格锚定在表示空间中以防止漂移，而生成式交叉熵损失确保忠实的（原文摘要在此处截断）

    arXiv:2611.31848v1 Announce Type: cross  Abstract: Standard LLM personalization typically frames identity as a static retrieval task, overlooking the inherent incongruity of human personas, where stable traits coexist with atypical, context-specific stances. Existing methods struggle to reconcile these dimensions: prompting succumbs to context drift over long sequences, while fine-tuning often suppresses idiosyncratic "quirks" in favor of generic distributional patterns. To bridge this gap, we present QuirkyMind, a framework that disentangles identity definition from its expression. First, Traits Anchoring constructs a dual-stream latent state, fusing a sentence-level summary for semantic stability with a token-level sequence for generative control. This state is stabilized via In-Context Narrative Refinement using an alternating objective: a discriminative InfoNCE loss anchors the persona in representation space to prevent drift, while a generative cross-entropy loss ensures faithful 
    
[^184]: 规格-测试覆盖差距：检测SWE-Bench风格基准中的过度指定测试

    Spec-Test Coverage Gap: Detecting Overspecified Tests in SWE-Bench-Style Benchmarks

    [https://arxiv.org/abs/2611.31839](https://arxiv.org/abs/2611.31839)

    本文提出规格-测试覆盖差距（STCG）这一1-4分制新指标，利用两个LLM评判者自动检测SWE-bench风格基准中的过度指定测试，从而取代昂贵的专家审查，识别会导致正确的替代修复方案被误判失败的测试用例。

    

    SWE-bench风格的基准测试，例如SWE-bench Verified（Verified）和SWE-bench Pro（Pro），使用源自最初修复各问题的变更中的测试来评估编码智能体。然而，这些测试可能会强制执行任务描述中从未说明的细节，导致正确的替代修复方案失败。迄今为止，这类过度指定的测试主要通过昂贵的专家审查来发现。为了解决这一问题，我们定义了规格-测试覆盖差距（STCG），这是一个用于衡量过度指定程度的1-4分制量表，并由两个LLM评判者——Claude Opus 5.5和GPT-6.1 Sol——根据规格说明和测试补丁进行评分。当两个评判者都给出3分或更高分数时，该任务被标记为过度指定。我们的研究包括三个部分。首先，我们在206个任务上验证了评判者，这些任务由早期的人工审计和智能体结果标记为过度指定或正常。其次，我们将STCG应用于全部500个Verified任务和731个公开的Pro任务，并将标记结果与排名前30的Verified（原文在此处截断）

    arXiv:2611.31839v1 Announce Type: cross  Abstract: SWE-bench-style benchmarks, such as SWE-bench Verified (Verified) and SWE-bench Pro (Pro), evaluate coding agents with tests taken from the change that originally fixed each issue. However, these tests can enforce details the task never states, causing correct alternative fixes to fail. Such overspecified tests have so far been found mainly through expensive expert review. To address this, we define the Spec-Test Coverage Gap (STCG), a 1-4 scale measuring overspecification, and score it with two LLM judges, Claude Opus 5.5 and GPT-6.1 Sol, which are given the specification and the test patch. A task is flagged as overspecified when both judges score it 3 or higher. Our study has three parts. First, we validate the judges on 206 tasks that earlier human audits and agent results label as overspecified or fine. Second, we apply STCG to all 500 Verified and 731 public Pro tasks and compare the flags with the results of the top 30 Verified 
    
[^185]: 注视世界模型

    Gaze World Model

    [https://arxiv.org/abs/2611.31836](https://arxiv.org/abs/2611.31836)

    提出注视世界模型（GWM），在冻结的 V-JEPA 2.1 编码器上将潜在场景预测与人物条件化的注视监督联合训练，实现注视定位、注视转移检测以及未来注视变化预测。

    

    一个人看向哪里表明了其注意力的焦点，而其目光在不同目标之间如何转移，则揭示了其注意力如何回应不断变化的场景。现有的注视目标估计方法大多逐帧运行，并未对这些注视转移进行建模。注视转移通常由场景中的动态变化所驱动，而这正是世界模型被设计用来编码的那类信息。然而这种关系较为微妙，因为发生在画面远处的事件可能只会引起人物头部或眼睛的轻微移动。为此，我们提出了注视世界模型，该模型在冻结的 V-JEPA 2.1 编码器上，将潜在场景预测与以人物为条件的注视监督进行联合训练；在我们的消融实验中，潜在预测目标同时提升了注视转移检测和注视定位的性能。给定一段视频片段和人物头部边界框，GWM 可以估计注视位置、检测所观察片段中的注视转移，并预测未来的注视和注（视转移）。

    arXiv:2611.31836v1 Announce Type: cross  Abstract: Where a person looks indicates their focus of attention, and how their gaze shifts between targets reveals how that attention responds to a changing scene. Existing gaze target estimation methods largely operate frame by frame and do not model these shifts. Gaze shifts are often driven by dynamic changes in the scene, which is the kind of information world models are designed to encode. This relationship is subtle, however, since an event far across the frame may produce only a slight movement of the person's head or eyes. We therefore introduce the Gaze World Model (GWM), which co-trains latent scene prediction with person-conditioned gaze supervision on a frozen V-JEPA 2.1 encoder; in our ablations, the latent prediction objective improves both shift detection and gaze localization. Given a video clip and the person's head box, GWM estimates gaze locations and detects gaze shifts in the observed clip, and forecasts future gaze and sh
    
[^186]: POISE：面向平滑Actor-Critic控制的曲率感知Critic正则化

    POISE: Curvature-Aware Critic Regularization for Smooth Actor-Critic Control

    [https://arxiv.org/abs/2611.31835](https://arxiv.org/abs/2611.31835)

    提出 POISE 方法，通过曲率加权的贪心偏移能量（以二阶近似保留先前贪心动作的价值损失）对 critic 进行正则化，从而有效抑制 Actor-Critic 控制中的高频动作振荡。

    

    Actor-critic（演员-评论家）方法在机器人控制中能够获得高回报，但训练出的演员策略常表现出高频动作振荡。大多数平滑性方法直接约束演员，而 PAVE 则通过分别惩罚动作梯度变化并鼓励负的动作曲率来正则化评论家。然而，相同的梯度变化在评论家较为平坦的区域会将其偏好动作移动得更远，而分离的惩罚项无法衡量这种移动。我们通过“贪心偏移”来刻画这种移动，它在局部凹性条件下以一阶近似估计状态转移间贪心动作的位移。由于位移本身无法反映其对评论家价值的影响，分析进一步表明，该偏移的曲率加权能量以二阶近似了保留先前贪心动作所损失的价值。基于这一关系，POISE 沿时间与空间维度对曲率加权能量进行正则化。

    arXiv:2611.31835v1 Announce Type: cross  Abstract: Actor-critic methods achieve high returns in robotic control, yet trained actors exhibit high-frequency action oscillations. Most smoothness methods constrain the actor, whereas PAVE regularizes the critic by separately penalizing action-gradient variation and encouraging negative action curvature. The same gradient change, however, moves the critic's preferred action farther where the critic is flat, and separate penalties do not measure this motion. We characterize this motion through the greedy shift, which approximates greedy-action displacement across state transitions to first order under local concavity. Since displacement alone does not reflect its critic-value consequences, the analysis further shows that the shift's curvature-weighted energy approximates the value lost by retaining the previous greedy action to second order. Based on this relationship, POISE regularizes the curvature-weighted energy along temporal and spatial
    
[^187]: 并非所有解释方法都生而平等：AI可解释性的分层评估

    Not All Interpretation Methods Are Created Equal: Tiered Evaluation of AI Interpretability

    [https://arxiv.org/abs/2611.31833](https://arxiv.org/abs/2611.31833)

    提出了TRUST分层评估框架，将解释方法的评估提升为AI核心评估指标，以区分不同解释方法的可信度证据等级，从而推动稳健的知识发现并确保AI在社会技术部署中的安全性与可争议性。

    

    随着AI转向社会技术部署，不可解释的模型对可靠性、公众信任和可争议性构成了严重威胁。然而，并非所有解释方法都生而平等：为了使可解释性能够提升部署的安全性和有效性，解释方法必须在一个分层体系中加以评估，以区分其可信度的证据等级，而不是通过孤立的解释实例进行验证。我们的核心愿景是，将解释方法的评估提升为核心AI评估指标，将推动稳健的知识发现（KDD），并确保AI在社会技术部署中的安全性和可争议性。为此，我们提出了TRUST，一个分层评估框架，它建立在真实数据科学（VDS）的可预测性-可计算性-稳定性（PCS）原则以及预测性、描述性与相关性（PDR）基础之上，以有效区分解释方法的可信度……

    arXiv:2611.31833v1 Announce Type: new  Abstract: As AI transitions to sociotechnical deployments, uninterpretable models pose serious threats to reliability, public trust and contestability. However, not all interpretation methods are created equal: for interpretability to enhance deployment safety and effectiveness, interpretation methods must be evaluated in a tiered system to differentiate evidence levels of trustworthiness, rather than validated through isolated interpretation instances. Our core vision is that elevating the evaluation of interpretation methods to a core AI assessment metric will drive robust knowledge discovery from data (KDD) and ensure safe, contestable sociotechnical deployments of AI. To this end, we propose TRUST, a tiered evaluation framework that builds upon the Predictability-Computability-Stability (PCS) principles for Veridical Data Science (VDS) and the Predictive, Descriptive and Relevant (PDR) foundations. To differentiate interpretation methods effec
    
[^188]: GaokaoMathGen：基于潜在推理结构控制的中国高考数学压轴题生成

    GaokaoMathGen: Chinese Gaokao Mathematics Finale Problem Generation Controlled by Latent Reasoning Structures

    [https://arxiv.org/abs/2611.31823](https://arxiv.org/abs/2611.31823)

    提出GaokaoMathGen框架，通过从种子题中提取技能、方法、核心恒等式和关键转折步骤等潜在推理结构来约束生成高考数学压轴题，并结合逻辑过滤与求解器解码等多重信号估计题目实际难度。

    

    生成中国高考数学压轴题不仅需要控制题目所考查的知识技能，还需要控制其底层的数学内核与解题路径。现有的基于大语言模型的方法主要依赖表层规范，而所生成题目的实际难度无法直接从提示目标中确定。因此，我们提出了GaokaoMathGen，一个由潜在推理结构控制的中国高考数学压轴题生成框架。该框架从种子题目中提取标准化的潜在推理结构，并以技能、方法、核心恒等式和关键转折步骤作为结构控制进行约束合成。我们进一步通过逻辑质量过滤、固定求解器的解码工作量、结构特征以及求解器输出搜索信号来估计题目的实际难度。实验表明，语义规范化提升了跨标注者的可比性，且在609个评分样本中有78.0%（摘要在此处截断）

    arXiv:2611.31823v1 Announce Type: new  Abstract: Generating Chinese Gaokao mathematics finale problems requires controlling not only the included skills, but also the underlying mathematical kernel and solution path. Existing LLM-based methods mainly rely on surface-level specifications, while the realized difficulty of generated problems cannot be determined directly from prompted targets. Therefore, we propose GaokaoMathGen, a framework for Chinese Gaokao mathematics finale problem generation controlled by latent reasoning structures. It extracts standardized latent reasoning structures from seed problems and performs constrained synthesis using skills, methods, a core identity, and a pivot step as structural controls. We further estimate realized difficulty through logical-quality filtering, fixed-solver decoding effort, structural features, and solver-output search signals. Experiments show that semantic normalization improves cross-annotator comparability and that 78.0% of 609 sco
    
[^189]: 探索多人社交游戏作为大语言模型的可验证训练场

    Exploring Multi-Player Social Games as Verifiable Training Grounds for Large Language Models

    [https://arxiv.org/abs/2611.31821](https://arxiv.org/abs/2611.31821)

    本文提出将多人社交游戏作为可验证训练环境，通过SFT和RLVR使Qwen3-4B的胜率从33%大幅提升至81%，超越了远大于自身的Gemma-3-27B模型，并能泛化到未见过的游戏中，效果远优于提示等上下文策略。

    

    训练具有社交智能的语言模型通常依赖于大语言模型（LLM）评判者，但这种方式成本高昂、噪声大且容易受到奖励作弊的影响。多人社交游戏提供了一种天然的替代方案：它们需要开放式的沟通来与队友协作并推断隐藏角色，同时其规则提供了客观的胜负结果，可以作为训练中的可验证奖励。本文通过三个社交游戏，探索了Qwen3模型通过监督微调（SFT）和基于可验证奖励的强化学习（RLVR）能够习得多少社交智能。结果表明，面对强大得多的Gemma-3-27B模型，SFT和RLVR将Qwen3-4B在三个训练游戏上的胜率从33%提升至81%，在另外三个未训练的保留游戏上的胜率从39%提升至54%。相比之下，提示等上下文内策略在相同的训练和评估游戏上仅将Qwen3-4B的胜率提高约6个百分点。

    arXiv:2611.31821v1 Announce Type: cross  Abstract: The training of socially intelligent language models often relies on large language model (LLM) judges, which are expensive, noisy, and vulnerable to reward hacking. Multi-player social games offer a natural alternative: they require open-ended communication to coordinate with teammates and infer hidden roles, while their rules provide an objective outcome which can serve as a verifiable reward for training. Using three social games, we explore how much social intelligence Qwen3 models can acquire through supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). Results show that against a much stronger Gemma-3-27B model, SFT and RLVR raise the win rate of Qwen3-4B from 33% to 81% on the three trained games, and from 39% to 54% on three additional held-out games. In contrast, in-context strategies such as prompting improve the win rate of Qwen3-4B by only about 6 pp across the same games we train and evalu
    
[^190]: 可压缩表示：深度神经网络中的功能性主干

    Compressible Representations: Functional Spines in Deep Neural Networks

    [https://arxiv.org/abs/2611.31817](https://arxiv.org/abs/2611.31817)

    该研究提出一个基于类Fisher信息矩阵的框架，用于识别深度网络各层中对任务功能敏感的子空间，发现表示高度可压缩——只需少量任务敏感维度即可保留大部分性能，且压缩表示对对抗攻击更鲁棒。

    

    深度网络将输入嵌入到高维表示中，但这种容量中有多少真正驱动了行为？我们引入了一个框架，用于在每个层级识别表示中对任务具有功能敏感性的子空间。我们使用一个类似于Fisher信息矩阵、但定义在隐藏表示之上的矩阵，根据各方向对下游计算的影响进行排序，并投影到对任务最敏感的子空间上。在多个预训练视觉网络中，表示表现出显著的可压缩性：任务敏感子空间中的一小部分即可保留大部分性能，其所需维度往往远少于几何替代方法。我们发现这种压缩表示对在完整网络上生成的对抗攻击也具有更强的鲁棒性。此外，我们还展示了这种分解能够从同一表示中恢复出功能上各不相同的子空间：在线索冲突刺激上……

    arXiv:2611.31817v1 Announce Type: new  Abstract: Deep networks embed inputs in high-dimensional representations, but how much of this capacity actually drives behavior? We introduce a framework for identifying, at each layer, the subspace of the representation that is functionally sensitive to the task. Using a matrix analogous to the Fisher information but defined over the hidden representation, we rank directions by their influence on downstream computation and project onto the most task-sensitive subspace. Across pretrained vision networks, representations are remarkably compressible: a small fraction of the task-sensitive subspace preserves most performance, often requiring far fewer dimensions than geometric alternatives. We find that this compressed representation is also more robust to adversarial attacks generated on the full network. We additionally show that this decomposition recovers functionally distinct subspaces within the same representation: on cue-conflict stimuli, sh
    
[^191]: 基于智能体的安全Rust绑定健全性漏洞发现

    Agentic Discovery of Soundness Bugs in Safe Rust Bindings

    [https://arxiv.org/abs/2611.31813](https://arxiv.org/abs/2611.31813)

    本文提出将LLM智能体作为Rust安全封装接口的审计员来发现健全性漏洞，定义了此类接口缺陷的分类体系，并通过真实crate的重复审计实验发现：智能体能自发找到未定义行为和与底层库的偏差，却难以发现不完整的不安全契约。

    

    Rust crate 将不安全代码（例如对 C 或 C++ 库的调用、系统调用）封装在必须维护 Rust 不变量的安全接口之后。如果接口对其底层代码的某条规则约束得过于宽松，用纯安全 Rust 编写的程序就可能引发未定义行为。静态分析器和模糊测试工具受限于固定的模式或其实际执行的路径。我们提出将 LLM 智能体用作这些接口的审计员，并研究其审计效果。我们首先为此类接口的缺陷定义了一个分类体系，包括未定义行为、与被封装库的偏差，以及不完整的不安全契约。随后，我们通过闭源模型 Claude Opus 5.5 和开源模型 DeepSeek V4.1 Flash 对一个真实世界 crate 进行重复审计，测量了智能体能发现哪些类别的缺陷、它们的缺陷分类能力如何，以及哪些因素能促使它们发现更多缺陷。结果表明，智能体在无提示的情况下就能发现前两类缺陷，但即使在明确要求下也很少发现不完整的不安全契约。（摘要原文在此处截断）

    arXiv:2611.31813v1 Announce Type: cross  Abstract: Rust crates confine unsafe code, such as calls into C or C++ libraries or system calls, behind safe interfaces that must uphold Rust's invariants. If an interface encodes one rule of the code beneath it too weakly, a program written in purely safe Rust can cause undefined behavior. Static analyzers and fuzzers are bound to fixed patterns or to the paths they execute. We propose LLM agents as auditors of these interfaces, and study how well they do it.   We first define a taxonomy of the defects of such interfaces, including undefined behavior, divergences from the wrapped library, and incomplete unsafe contracts. We then measure, in repeated audits of a real-world crate by a closed and an open model, Claude Opus 5.5 and DeepSeek V4.1 Flash, which defect classes agents find, how well they triage them, and what makes them find more. Agents find the first two classes unprompted, but find few incomplete contracts even when asked. Their fin
    
[^192]: 两个证明相距多远？在证明内核与 Lean/Mathlib 中度量证明差异性

    How Far Apart Are Two Proofs? Measuring Proof Dissimilarity in a Proof Kernel and in Lean/Mathlib

    [https://arxiv.org/abs/2611.31811](https://arxiv.org/abs/2611.31811)

    该论文针对“同一定理的两个证明相距多远”这一源于希尔伯特第 24 问题的基础问题，定义了 6 大族共 19 种证明差异性度量和 10 个期望性质，并在 Curry-Howard 内核与 Lean/Mathlib 中加以实例化与验证。

    

    我们如今生活在机器生成数学的时代：AI 系统产生研究级证明的速度已超过人类数学界能够阅读它们的速度。验证只是挑战的一部分，我们还需要知道一个“新证明”是否真的是新的，一千个生成的证明究竟包含一个想法还是多个想法，以及每个证明相对于人类数学所处的位置。所有这些都要求度量证明之间的距离。沿着希尔伯特第 24 问题与证明等同性问题的思路，我们追问：同一个定理的两个证明相距多远？它们所栖居的证明空间又具有怎样的形状？我们定义了 6 个族（句法、结构、依赖与资源、语义、信息论、高阶范畴）共 19 种差异性度量，以及 10 个期望性质，我们认为这些性质是安全优化某一度量的前提条件。在一个 Curry-Howard 内核中，5 个经机器验证的素数无穷性证明表明，原始句法……

    arXiv:2611.31811v1 Announce Type: cross  Abstract: We now live in the age of machine-generated mathematics: AI systems produce research-level proofs faster than the human mathematical community can read them. Verification is only part of the challenge. We also need to know whether a "new proof" is indeed new, whether a thousand generated proofs contain one idea or many, and where each sits relative to human mathematics. All of these require measuring distances between proofs. Following Hilbert's 24th problem and the identity-of-proofs problem, we ask how far apart two proofs of one theorem are, and what shape the proof space they inhabit has. We define 19 dissimilarity measures in 6 families (syntactic, structural, dependency & resource, semantic, information-theoretic, higher-categorical) and 10 desirable properties, which we argue are the conditions for safely optimising a measure. In a Curry-Howard kernel, 5 machine-checked proofs that the primes are infinite show that raw syntax ca
    
[^193]: TopoCryptoEEG：用于脑纹认证的端到端加密框架

    TopoCryptoEEG: End-to-End Encrypted Framework for Brainprint Authentication

    [https://arxiv.org/abs/2611.31801](https://arxiv.org/abs/2611.31801)

    该论文揭示了现有特征级加密脑纹认证方案存在漏洞——被攻破的客户端可在有限次交互后精确恢复注册模板，并提出基于同态加密的TopoCryptoEEG框架，在客户端加密预处理脑电数据、在服务器端对密文完成全部认证流程，从而将保护范围扩展至预处理数据并彻底消除明文特征接口。

    

    脑纹（brainprint）认证能够实现持续的身份验证，但脑电（EEG）数据具有隐私敏感性。现有的特征级加密认证（FLEA）方案是在“可信客户端”假设下保护提取的特征。我们从理论和实证两方面证明：一个被攻破的客户端若能获取明文查询特征和持续认证分数，在有限次交互后即可精确恢复已注册的模板。为应对这一风险，我们提出了TopoCryptoEEG，这是一个基于同态加密的框架，它在客户端对预处理的脑电数据进行加密，并在服务器端对密文执行特征提取、模板匹配、分数计算和阈值居中。这将保护范围从提取的特征扩展到预处理的脑电数据，并消除了攻击所利用的客户端明文特征接口。为实现高效的加密认证，我们开发了多窗口拓扑感知编码……

    arXiv:2611.31801v1 Announce Type: cross  Abstract: Brainprint authentication enables continuous identity verification, but EEG data are privacy-sensitive. Existing feature-level encrypted authentication (FLEA) schemes protect extracted features under a trusted-client assumption. We show theoretically and empirically that a compromised client with access to plaintext query features and continuous authentication scores can exactly recover a registered template after finitely many interactions. To address this risk, we propose TopoCryptoEEG, a homomorphic-encryption-based framework that encrypts preprocessed EEG data at the client and performs feature extraction, template matching, score computation, and threshold centering on ciphertexts at the server. This extends protection from extracted features to preprocessed EEG data and removes the client-side plaintext-feature interface exploited by the attack. For efficient encrypted authentication, we develop Multi-Window Topology-Aware Encodi
    
[^194]: 为大语言模型智能体学习紧凑的状态抽象

    Learning Compact State Abstractions for LLM Agents

    [https://arxiv.org/abs/2611.31800](https://arxiv.org/abs/2611.31800)

    本文提出将智能体状态定义为交互历史的紧凑抽象，以确定性递归程序表示并用LLM引导的遗传算法搜索，在ALFWorld和ScienceWorld上保持接近参考智能体的任务成功率的同时，将策略输入最多减少71%和57%，且得到的程序可直接审查。

    

    大语言模型（LLM）智能体通常将完整的交互历史作为状态的替代，无论过去的观察和动作是否仍然相关，都全部予以保留。我们转而探究智能体真正需要携带哪些信息前进。我们将智能体状态定义为交互历史的一种紧凑抽象，它能够保持某个固定参考智能体的能力，并以最小的此类抽象作为理想目标。由于匹配参考智能体的行为对黑盒策略而言过于局限且困难，我们通过任务表现而非行为匹配来定义充分性。我们将状态抽象表示为确定性递归程序，并使用由大语言模型引导的遗传算法对其进行搜索。在ALFWorld和ScienceWorld基准上，学习到的抽象保持了接近参考智能体的任务成功率，同时将策略的平均输入分别减少了高达71%和57%。所得到的程序是可直接检查审查的。

    arXiv:2611.31800v1 Announce Type: cross  Abstract: LLM agents commonly use the full interaction history as a proxy for state, retaining all past observations and actions regardless of whether they remain relevant. We instead ask what information an agent actually needs to carry forward. We define agent state as a compact abstraction of interaction history that preserves the competence of a fixed reference agent, with the minimal such abstraction as the ideal. Because matching the reference agent's behavior is unnecessarily restrictive and difficult for black-box policies, we define sufficiency through task performance rather than behavior matching. We represent state abstractions as deterministic recursive programs and search over them using an LLM-guided genetic algorithm. Across ALFWorld and ScienceWorld, the learned abstractions preserve near-reference task success while reducing mean policy input by up to 71% and 57%, respectively. The resulting programs are directly inspectable, p
    
[^195]: 一种用于城市区域自动驾驶测试、模拟交互式人类驾驶车辆的生成式AI智能体

    A Generative AI Agent to Emulate Interactive Human-driven Vehicles for Automated Driving Testing in Urban Areas

    [https://arxiv.org/abs/2611.31797](https://arxiv.org/abs/2611.31797)

    该论文提出一种生成式AI智能体，通过视场建模处理遮挡影响，并以五个指标刻画交互全过程的风格，从而真实模拟人类驾驶车辆的安全关键交互行为，服务于城市场景的自动驾驶安全测试。

    

    在城市场景中，真实建模人类驾驶车辆的交互行为对于自动驾驶安全测试至关重要。现有方法难以重现涉及高碰撞风险的安全关键交互，原因在于：它们通常假设完全可观测性，忽视了由遮挡引发的交互；驾驶风格主要通过车辆机动动作来表征，限制了安全关键交互行为的多样性；此外，来自安全关键交互的学习信号可能被非关键驾驶行为所稀释。为解决这些问题，我们提出一种生成式AI智能体来模拟交互式人类驾驶车辆。该智能体引入视场建模以考虑遮挡条件下的驾驶员可见性，并使用涵盖交互过程初始、中间和最终阶段的五个指标来刻画交互过程，通过动力学……（摘要在此处截断）

    arXiv:2611.31797v1 Announce Type: new  Abstract: Realistic modeling of human-driven vehicle (HDV) interactions is essential for automated driving safety testing in urban areas. Existing methods struggle to reproduce realistic safety-critical interactions involving high collision risk for the following reasons: they often assume full observability, overlooking interactions arising from occlusion. Driving styles are mainly characterized by vehicle maneuvers, limiting the diversity of safety-critical interaction behaviors. Moreover, learning signals from safety-critical interactions can be diluted by non-critical driving behaviors. To address these issues, we propose a generative AI agent to emulate interactive HDVs. The agent incorporates field-of-view modeling to account for driver visibility under occlusion and characterizes the interaction process using five metrics spanning its initial, intermediate, and final stages. It automatically learns diverse interaction styles through a dynam
    
[^196]: 构造性因果抽象与机制可解释性中的案例研究

    Constructive Causal Abstraction and a Case Study in Mechanistic Interpretability

    [https://arxiv.org/abs/2611.31795](https://arxiv.org/abs/2611.31795)

    该论文提出构造性因果抽象方法，将定义不良的宏观干预解释为宏观层面的缺失变量，并通过引入补充变量 q 保留被遗漏的因果相关状态，从而构造性地获得与微观模型因果一致的宏观因果模型，并在机制可解释性中进行了案例验证。

    

    现有的因果抽象框架首先提出一个高层因果模型，然后验证其干预效应是否与给定的微观层面模型的干预效应相一致——当差异为零时，这两个模型被判定为因果一致。然而，提出一个一致的宏观因果模型是困难的，因为宏观因果关系几乎从未被良好定义，因为与同一表面宏观干预相关联的不同微观干预可能产生不同的效应。我们的核心洞察是，因果一致的宏观层面模型可以通过构造性方法获得：定义不良的宏观干预可以被解释为宏观层面上的缺失变量，并通过引入一个保留被遗漏的因果相关状态的补充变量 q 来加以处理。在线性情形下，q 具有唯一的形式，并且我们给出了有效实现存在的充分条件；在非线性情形下，（摘要在此处被截断）

    arXiv:2611.31795v1 Announce Type: new  Abstract: Existing frameworks in causal abstraction first propose a high-level causal model, and then verify whether its interventional effects agree with those of a given micro-level model - the two models are declared causally consistent when the discrepancy is zero. However, proposing a consistent macro causal model is difficult, because macro causal relations are almost never well-defined, since distinct micro-level interventions associated with the same apparent macro intervention can produce different effects. Our central insight is that causally consistent macro-level models can be obtained constructively: ill-defined macro interventions can be interpreted as missing variables on the macro level, and can be accounted for by introducing a complementary variable q that retains omitted causally relevant state. In the linear case, q has a unique form and we provide sufficient conditions for valid implementations to exist; in the nonlinear case,
    
[^197]: AGL-SKILL：将编程脚手架作为智能体优化器

    AGL-SKILL: Coding Harnesses as Agentic Optimizers

    [https://arxiv.org/abs/2611.31794](https://arxiv.org/abs/2611.31794)

    本文的核心创新是证明无需任何指导的编程智能体本身即可作为高效的技能优化器，在五个基准测试中显著提升每个目标性能并超越现有已发表的技能优化方法。

    

    前沿模型越来越多地通过脚手架作为智能体来使用，其自主完成有用工作的能力不断提升。智能体的性能通常可以通过使用“技能”来增强：技能是可复用的工作流程、代码和提示，为模型提供与任务相关的信息或引导模型的行为。技能本质上是文本形式的，因此可以被视为另一个可编辑的优化对象；当前的方法通过将技能开发视为一个迭代过程，使用基于分数驱动的搜索来调整和改进性能，从而相比手工编写和一次性生成有所改进。在本文中，我们证明了智能体本身作为优化器的有效性：在五个基准测试中，一个没有任何指导的编程智能体显著提升了每一个目标，并超越了已发表的技能优化方法。更广泛地说，这表明智能体有能力完成此类工作，并能优化自身行为以提供特定任务的上下文，只要给予它们适当的条件（摘要原文在此处被截断）。

    arXiv:2611.31794v1 Announce Type: new  Abstract: Frontier models are increasingly used through harnesses as agents, with growing autonomy to do useful work. Agent performance can often be enhanced by the use of skills: reusable workflows, code, and prompts that provide task-related information or guide the behaviour of the model. Skills, being textual in nature, present themselves as another editable surface to be optimized; current methods have improved on handcrafting and one-shot generation by treating skill development as an iterative process, using score-driven search to tweak and improve performance. In this paper, we demonstrate the efficacy of the agents themselves as optimizers: across five benchmarks, a coding agent without guidance substantially improves every target and outperforms published skill-optimization methods. More broadly, this shows the capability of agents to do such work and to refine their own behaviour to provide task-specific context, as long as they are giv
    
[^198]: 基于物理信息傅里叶神经算子的浮式海上风力机三维尾流重建与跨尺度迁移

    Three-Dimensional Wake Reconstruction and Cross-Scale Transfer for Floating Offshore Wind Turbines via Physics-Informed Fourier Neural Operators

    [https://arxiv.org/abs/2611.31789](https://arxiv.org/abs/2611.31789)

    该研究提出物理信息傅里叶神经算子框架，将四维谱卷积与不可压缩连续性及动量残差等物理约束相结合，可从两个观测平面重建浮式海上风力机的三维非定常尾流，并实现从5兆瓦到22兆瓦风机的跨尺度迁移。

    

    浮式海上风力机的纵荡-俯仰耦合运动会引发非定常涡相互作用和动态尾流摆动，这使得基于有限测量数据的三维尾流重建以及跨风机尺度的迁移变得困难。研究采用大涡模拟耦合致动线模型，针对美国国家可再生能源实验室（NREL）5兆瓦风机（D=126米）和国际能源署（IEA）22兆瓦风机（D=284米），在基于运动的斯特劳哈尔数 St=fD/U∞（其中f、D和U∞分别为运动频率、转子直径和来流速度）为0.2-0.6的条件下，生成了固定式和纵荡-俯仰运动状态下的尾流数据。物理信息傅里叶神经算子框架将四维谱卷积与不可压缩连续性约束及转子强迫动量残差正则化相结合，从两个观测平面重建三维非定常尾流。该框架在5兆瓦固定式和St=0.2-0.5工况数据上训练，能够恢复……（原文摘要在此处截断）

    arXiv:2611.31789v1 Announce Type: cross  Abstract: Coupled surge-pitch motions drive unsteady vortex interactions and dynamic wake meandering behind floating offshore wind turbines, complicating three-dimensional wake reconstruction from limited measurements and transfer across turbine scales. Large-eddy simulations coupled with an actuator line model generate fixed and surge-pitch wakes at motion-based Strouhal numbers $St=fD/U_\infty$ of 0.2-0.6 (where $f$, $D$, and $U_\infty$ are the motion frequency, rotor diameter, and inflow velocity) for the National Renewable Energy Laboratory 5 MW ($D=126$ m) and International Energy Agency 22 MW ($D=284$ m) turbines. A physics-informed Fourier neural operator framework combines four-dimensional spectral convolutions with incompressible continuity and rotor-forced momentum-residual regularization to reconstruct three-dimensional unsteady wakes from two observation planes. Trained on the 5 MW fixed and $St=0.2$-$0.5$ cases, the framework recove
    
[^199]: SciFig2Data：一种从科学图表中恢复结构化数据的工作流

    SciFig2Data: A Workflow for Recovering Structured Data from Scientific Figures

    [https://arxiv.org/abs/2611.31787](https://arxiv.org/abs/2611.31787)

    该论文提出SciFig2Data，一个集成图表定位、复合图分解、图表识别、多模态大语言模型数值重建与上下文关联的自动化高通量工作流，能够从科学PDF文献库中大规模恢复结构化实验数据。

    

    学术论文中报告的科学测量数据通常嵌入在复合图表中，而非以机器可读的数据形式提供。要在文献规模上恢复这些测量数据，需要一个涵盖图表定位、面板分解、图表类型识别、数值重建以及实验上下文关联的集成化工作流。我们提出了 SciFig2Data，这是一个自动化、高通量的工作流，用于从科学 PDF 文献库中采集结构化实验数据。该系统集成了多尺度文档元素检测、自适应复合图分解、基于 OCR 的两阶段语义过滤、基于 Swin 的图表分类，以及指令引导的多模态大语言模型（MLLM）数值重建。此外，系统还集成了目录级批量执行、缓存和结构化日志功能，以支持大规模处理。我们在两个生物医学文献语料库上演示了该工作流……

    arXiv:2611.31787v1 Announce Type: new  Abstract: Scientific measurements reported in scholarly articles are often embedded in compound figures rather than available as machine-readable data. Recovering these measurements at literature scale requires an integrated workflow for figure localization, panel decomposition, chart identification, numerical reconstruction, and experimental context association. We present \textit{SciFig2Data}, an automated, high-throughput workflow for harvesting structured experimental data from scientific PDF repositories. The system integrates multi-scale document element detection, adaptive compound figure decomposition, two-stage OCR-based semantic filtering, Swin-based chart classification, and instruction-guided multimodal large language model (MLLM) reconstruction. It further incorporates directory-level batch execution, caching, and structured logging to support large-scale processing. We demonstrate the workflow on two biomedical literature corpora com
    
[^200]: VISOR：用于多模态时间序列预测的视觉-语言选择器

    VISOR: A Vision-Language Selector for Multimodal Time-Series Forecasting

    [https://arxiv.org/abs/2611.31780](https://arxiv.org/abs/2611.31780)

    VISOR 提出利用视觉-语言模型的强对齐能力，根据文本信息从随机预测器生成的多个候选未来中选出最合理的一个，而非取平均，从而改进多模态时间序列预测。

    

    近期的时间序列基础模型（TSFMs）会对未来进行分布预测，而多模态时间序列（TS）预测则进一步利用其他模态来提高预测精度。然而，现有工作存在两个局限性：1）TSFMs 将预测分布简化为其中位数或均值，把多样化的未来扁平成一条平缓的曲线；2）多模态 TS 预测模型通过时间序列-语言对齐将文本信息融入预测器，而这种对齐能力远弱于视觉-语言对齐。为了解决这些局限性，我们提出了视觉-语言选择器（VISOR），这是一个多模态时间序列预测框架，它不再对候选未来取平均，而是选择出一个合理的未来，利用视觉-语言模型（VLMs）的对齐能力，根据文本信息挑选出最合适的候选。VISOR 由两个组件构成：1）一个无文本的随机预测器，用于采样多样化的候选……

    arXiv:2611.31780v1 Announce Type: cross  Abstract: Recent time-series foundation models (TSFMs) forecast a distribution over futures, and multimodal time-series (TS) forecasting further exploits other modalities to improve forecasting accuracy. However, existing works have two limitations: 1) TSFMs reduce the predictive distribution to its median or mean, flattening diverse futures into a calm line, and 2) multimodal TS forecasting models incorporate text into the forecaster through TS-language alignment, which is substantially weaker than vision-language alignment. To address these limitations, we propose the VIsion-language SelectOR (VISOR), a multimodal TS forecasting framework that selects one plausible future instead of averaging across candidates, leveraging the alignment capabilities of vision-language models (VLMs) to select the most suitable candidate based on textual information. VISOR consists of two components: 1) a text-free stochastic predictor that samples diverse candid
    
[^201]: 面向多跳问答的基于SPARQL规划的在线知识图谱单跳遍历

    SPARQL-Planned Single-Hop Traversal over a Live Knowledge Graph for Multi-Hop Question Answering

    [https://arxiv.org/abs/2611.31771](https://arxiv.org/abs/2611.31771)

    该论文提出将稠密检索与基于SPARQL规划的在线知识图谱（Wikidata）单跳遍历交替进行，通过推理路径验证检索段落，在避免多跳扩展开销与噪声的同时兼顾召回率与精确率，从而提升多跳问答性能。

    

    语义检索在多跳问答的召回率和精确率两方面都陷入困境：黄金（正确）段落被淹没在干扰项之中，从而降低阅读器生成答案的质量，而检索这些段落需要对查询进行推理，而非简单匹配。GraphRAG技术通过将段落连接到图节点并在初始检索后执行一跳或多跳扩展来增强语义检索，这提升了召回率，但多跳扩展代价高昂，并且会给检索结果引入额外噪声。我们通过将稠密检索与知识图谱遍历交替执行来打破这种权衡，以应对复杂的多跳问题。稠密检索以较高的召回率定位搜索空间中最初相关的区域，而对预先存在的外部知识图谱（Wikidata）的遍历则通过依据知识图谱中的推理路径对检索到的段落进行验证，从而恢复精确率并提升召回率。为使遍历既具问题自适应性又……（摘要原文在此处截断）

    arXiv:2611.31771v1 Announce Type: cross  Abstract: Semantic retrieval struggles with multi-hop question answering on both recall and precision: gold passages are buried among distractors that degrade the reader's answer, and retrieving them requires reasoning over the query rather than matching it. GraphRAG techniques expand semantic retrieval by connecting passages to graph nodes and performing one or more hops of expansion after initial retrieval. This improves recall, but multi-hop expansion becomes expensive and brings additional noise into the retrieved set. We break this tradeoff by \emph{interleaving} dense retrieval with knowledge-graph traversal for complex multi-hop questions. Dense retrieval locates the first relevant regions of the search space at good recall, while traversal of a pre-existing external KG (Wikidata) restores precision and improves recall by confirming retrieved passages against a reasoning path through the KG. To keep traversal both question-adaptive and re
    
[^202]: DELTA：通过双路径对齐诊断探索谱系——对比设计路径能揭示哪些关于微架构行为的信息？

    DELTA: Diagnosing Exploration Lineages via Twin-path Alignment - What Can Contrasting Design Paths Reveal About Microarchitectural Behavior?

    [https://arxiv.org/abs/2611.31767](https://arxiv.org/abs/2611.31767)

    论文提出DELTA工具，通过将设计空间探索树中同一修改在不同父配置下演化出的两条路径进行对齐，把原本被丢弃的父子配置对比数据转化为根因诊断证据，从而定位使同一设计修改在不同微架构环境下产生相反效果的交互因素。

    

    架构师在性能研究中拥有两种成熟的工具。自顶向下分析告诉他们设计在何处丢失了流水线槽位，设计空间探索（DSE）则告诉他们哪种配置得分最高。然而两者都无法解释为什么搜索会停止改进：为什么一个后端瓶颈的设计会拒绝瓶颈标签所推荐的预取器，或者为什么在某一种配置下有帮助的容量增加在下一种配置中却造成损害。本文认为，回答这些问题所需的证据其实已经在探索过程中产生，只是随后被丢弃了。探索树中的每一条父子边都是一次受控干预：仅改变一个旋钮，其余设计保持不变，由周期级模拟器测量其结果。当同一项修改在一个父配置下有益而在另一个父配置下有害时，对齐这两个父配置即可定位决定其作用方向的交互因素。我们提出了DELTA，一个基于模拟器的根因分析工具，它将……

    arXiv:2611.31767v1 Announce Type: cross  Abstract: Architects have two mature instruments for performance work. Top-Down analysis tells them where a design loses its pipeline slots, and design-space exploration (DSE) tells them which configuration scores best. Neither explains why a search stops improving: why a backend-bound design rejects a prefetcher that the bottleneck label recommends, or why a capacity increase that helped in one configuration hurts in the next. This paper argues that the evidence needed to answer such questions is already produced during exploration and then discarded. Every parent-child edge in an exploration tree is a controlled intervention: one knob changes, the rest of the design is held fixed, and a cycle-level simulator measures the outcome. When the same edit helps under one parent and hurts under another, aligning the two parents localizes the interaction that decides its sign.   We present DELTA, a simulator-grounded root-cause analysis tool that build
    
[^203]: RoMo Hands：一个大规模、丰富组织的文本到手部动作数据集

    RoMo Hands: A Large Scale Richly Organized Text to Hand Motion Dataset

    [https://arxiv.org/abs/2611.31766](https://arxiv.org/abs/2611.31766)

    本文提出了RoMo Hands数据集——包含约54.2万序列（884小时）的真实场景文本-手部动作数据，规模比以往工作大一个数量级，并通过严格的灵巧性过滤、层次化多粒度文本标注以及新颖的393维双手表示方法，弥合了身体与手部动作生成架构之间的差距。

    

    尽管文本驱动的身体动作生成技术已经成熟，纯手部动作合成仍受制于数据瓶颈。现有数据集往往将手视为刚性末端执行器，缺乏细粒度的文本描述，或规模过于有限而难以支持稳健的泛化。我们提出了 RoMo Hands，一个真实场景下的文本配对数据集，包含约54.2万个序列（884小时，30fps下共9550万帧），规模比以往工作大一个数量级。为确保高质量监督信号，严格的灵巧性过滤器会剔除低关节活动度的序列，使数据分布聚焦于复杂的双手运动。每个片段配有五个不同粒度的文本描述，从全局语义标签到原子级手指状态，并按照层次化分类体系进行组织。为弥合身体与手部架构之间的差距，我们提出了一种新颖的393维双手表示方法，将冗余的身体动作范式适配到手部的高自由度运动学中。系统性基准评测……

    arXiv:2611.31766v1 Announce Type: cross  Abstract: While text-driven body motion generation has matured, pure-hand synthesis remains bottlenecked by data. Existing datasets often treat hands as rigid end-effectors, lack fine-grained captions, or are too limited in scale to support robust generalization. We introduce RoMo Hands, an in-the-wild, text-paired dataset of ~542K sequences (884 hours, 95.5M frames at 30 fps) which is an order of magnitude larger than prior work. To ensure high-quality supervision, a stringent dexterity filter discards low-articulation sequences, focusing the distribution on complex bimanual movement. Each clip is paired with five captions at increasing granularities from global semantic tags to atomic finger states, organized under a hierarchical taxonomy. To bridge the gap between body and hand architectures, we propose a novel 393-D bimanual representation that adapts redundant body-motion paradigms to the high-DoF kinematics of hands. Systematic benchmarkin
    
[^204]: 使用 Codex 和 Qwen 进行经验证的证明重构

    Verified Proof Refactoring with Codex and Qwen

    [https://arxiv.org/abs/2611.31762](https://arxiv.org/abs/2611.31762)

    该论文提出一种结合本地部署 Qwen 模型与开源 Codex CLI 的系统，在保持定理陈述和跨 Lean 版本兼容性的前提下重构现有 Lean 4 证明以缩短源代码并降低检查成本，最终在 Lean Refactor Arena 开源 LLM 赛道中夺得第一名。

    

    我们对现有的 Lean 4 证明进行重构，以减少源代码长度和检查成本，同时保持定理陈述不变并兼容不同版本的 Lean。本地部署的 Qwen 模型使用开源的 Codex CLI 来检查源代码、检索声明并测试证明替换方案。只有当替换方案能够改善目标指标且 Lean 在所有必需版本上都能通过验证时，宿主才会保留该替换。随后，一个确定性的 CPU 流程会对已验证的证明进行进一步精炼优化。在 Lean Refactor Arena 的 60 个问题上，模型搜索取得了 82.07 的局部平均综合得分，经过清理阶段后提升至 82.18。最终证明通过了全部 127 项必需的证明-版本检查。我们的系统在 VeriCodeGen 的 Lean Refactor Arena 开源大语言模型赛道中获得了第一名。

    arXiv:2611.31762v1 Announce Type: new  Abstract: We refactor existing Lean 4 proofs to reduce source length and checking cost while preserving theorem statements and compatibility across Lean versions. A locally served Qwen model uses the open-source Codex CLI to inspect source, retrieve declarations, and test proof replacements. The host keeps a replacement only if it improves the objective and Lean accepts it on every required version. A deterministic CPU pass then refines the verified proofs. On the 60 problems in Lean Refactor Arena, model search achieves a local mean combined score of 82.07, and cleanup raises it to 82.18. The final proofs pass all 127 required proof--version checks. Our system placed first in the open-source LLM track of Lean Refactor Arena at VeriCodeGen.
    
[^205]: TACT-Bench：针对工具使用型智能体的自适应对话威胁测试基准

    TACT-Bench: Testing Adaptive Conversational Threats against Tool-Use Agents

    [https://arxiv.org/abs/2611.31756](https://arxiv.org/abs/2611.31756)

    提出了TACT-Bench，一个针对工具使用型AI智能体的有状态安全基准，通过场景化故事、人设、操纵策略和响应自适应升级，测试智能体在拒绝违反策略的未授权请求的同时保持合法任务完成的能力。

    

    arXiv:2611.31756v1 公告类型：新 摘要：随着AI智能体在使用外部工具和做出决策方面的自主性不断增强，对全面安全基准的需求变得日益迫切。现有基准主要测试被攻陷的工具通道或自适应多轮攻击，但它们可能无法捕捉受策略约束的工具使用型智能体的独特能力——这类智能体必须在拒绝未授权请求的同时，不放弃同一交互中蕴含的合法工作。为填补这一空白，我们提出了TACT-Bench，这是一个围绕场景特定故事、人设、操纵策略和响应自适应升级构建的有状态基准。每个对抗性任务都针对提供给助手的策略中明确禁止的操作，同时该策略对攻击用户保持隐藏。TACT为每个对抗性任务配对至少一个良性对应任务，并且在合规解决方案仍然可用的情况下，其对抗性任务保留了经授权的真实操作。我们定义了……（摘要在此处被截断）

    arXiv:2611.31756v1 Announce Type: new  Abstract: The growing autonomy of AI agents in using external tools and making decisions has created an urgent need for comprehensive safety benchmarks. Existing benchmarks test compromised tool channels or adaptive multi-turn attacks, though they may not capture the unique capabilities of policy-governed tool-use agents, which must resist unauthorized requests without abandoning the legitimate work embedded in the same interaction. To fill this gap, we introduce TACT-Bench, a stateful benchmark built around scenario-specific stories, personas, manipulation tactics, and response-adaptive escalation. Each adversarial task targets an action explicitly prohibited by the policy provided to the assistant, while the policy remains hidden from the attacking user. TACT pairs each adversarial task with at least one benign counterpart, and its adversarial tasks retain authorized ground-truth actions when a compliant resolution remains available. We define t
    
[^206]: RVLM-X：迈向通用的视觉语言模型辅助框架

    RVLM-X: Towards a Universal VLM Harness

    [https://arxiv.org/abs/2611.31755](https://arxiv.org/abs/2611.31755)

    RVLM-X提出了一种任务可泛化的递归式VLM辅助框架，让前沿模型在代码REPL中自主控制全分辨率图像的视觉上下文并递归调用自身，从而在照片、图表和地图等多种任务上无需改动即可提升准确率。

    

    视觉语言模型（VLM）的辅助框架随着模型能力的增长而不断发展，尤其是在处理远大于训练时所见图像的场景中——此时降缩放会丢失细小细节。现有的多数框架将VLM包裹在一个固定流程中，例如缩放树（zoom tree），预先决定模型看向哪里以及如何整合证据。这些流程通过将旧模型约束在任务特定的路径上来弥补其弱点，但对于能够可靠地进行规划并使用工具的前沿模型而言，它们反而可能降低准确率。我们提出RVLM-X，一个任务可泛化的递归式VLM辅助框架，它在照片、图表和地图等任务上无需任何改动即可运行。遵循递归语言模型（Recursive Language Models）的递归推理方式，VLM在一个保存全分辨率图像的代码REPL中以编程方式控制自己的视觉上下文，并对它自行选择的区域递归地调用自身。在以具有成本效益的前沿模型作为所有方法的基础时，RVLM-X获得了更高的准确率。

    arXiv:2611.31755v1 Announce Type: cross  Abstract: Harnesses for vision-language models (VLMs) have grown with the models' capabilities, notably for images far larger than those seen in training, where downscaling loses small details. Most of them wrap the VLM in a fixed procedure, such as a zoom tree, that decides in advance where it looks and how the evidence is combined. These procedures make up for the weaknesses of older models by keeping them on a task-specific path, but with frontier models that reliably plan and use tools, they can lower accuracy. We propose RVLM-X, a task-generalizable recursive VLM harness that runs unchanged across photographs, charts and maps. Following the recursive inference of Recursive Language Models, the VLM programmatically controls its own visual context in a code REPL that holds the full-resolution image and calls itself recursively on regions that it chooses. With a cost-efficient frontier model as the base of every method, RVLM-X is more accurate
    
[^207]: FN-Prune：基于有限RMSNorm扰动的免训练专家剪枝

    FN-Prune: Training-Free Expert Pruning with Finite RMSNorm Perturbations

    [https://arxiv.org/abs/2611.31747](https://arxiv.org/abs/2611.31747)

    FN-Prune是一种免训练的专家剪枝方法，通过比较减去各专家贡献前后的RMSNorm输出差异来评估专家重要性，在移除50%专家时于Qwen3、ERNIE和OLMoE上超越现有最佳方法，在OLMoE上困惑度降低32%。

    

    专家剪枝可以降低混合专家语言模型的内存占用，但保持模型质量取决于移除哪些专家。基于频率和幅度的分数是孤立地衡量专家，忽视了归一化如何消耗其贡献：减去一个专家会同时改变残差流的方向以及应用于剩余部分的缩放比例。我们提出FN-Prune，一种免训练的准则，它在保持原始路由和权重固定的情况下，比较减去每个路由器加权专家贡献前后的下一次RMSNorm输出。平方差在一次遍历中对所有校准token进行平均；低分专家以统一的每层比例被物理删除，推理时在幸存专家之间重新路由。在移除50%专家的情况下，FN-Prune在Qwen3、ERNIE和OLMoE上实现了比已发表最佳方法更低的WikiText-2困惑度，包括在OLMoE上降低了32%。

    arXiv:2611.31747v1 Announce Type: new  Abstract: Expert pruning reduces the memory footprint of mixture-of-experts language models, but retaining model quality depends on which experts are removed. Frequency- and magnitude-based scores measure the expert in isolation and overlook how normalization consumes its contribution: subtracting one expert changes both the direction of the residual stream and the scale applied to what remains. We propose FN-Prune, a training-free criterion that compares the next RMSNorm outputs before and after subtracting each router-weighted expert contribution, holding original routes and weights fixed. Squared differences are averaged over all calibration tokens in one pass; low-scoring experts are physically deleted at uniform per-layer ratios, and inference reroutes among survivors. At 50% expert removal, FN-Prune achieves lower WikiText-2 perplexity than the best published methods on Qwen3, ERNIE, and OLMoE, including a 32% reduction on OLMoE. At the same
    
[^208]: AI 运行框架的界面在哪里？

    Where is the Interface for the AI Harness?

    [https://arxiv.org/abs/2611.31739](https://arxiv.org/abs/2611.31739)

    论文提出智能体委托打破了传统“用户在环”假设，开创了“规约鸿沟”与“监督鸿沟”两条新鸿沟，而两者都汇聚于由开发者编写、对用户隐藏的 AI 运行框架中，因此主张应将运行框架本身作为人机交互的新界面。

    

    诺曼提出的执行鸿沟与评估鸿沟四十年来一直指导着交互设计，但两者都假设用户始终在交互循环中在场。智能体委托打破了这一假设：用户描述一个期望的结果后便离开，而智能体在无人注视的情况下执行数百个操作。从 2026 年的一起事件说起——一个研究型智能体绕过了澳大利亚某政府门户网站的访问拒绝，且长达两个月无人察觉——我们认为委托开启了两条新的鸿沟。“规约鸿沟”是用户想要的结果与其表达能力之间的差距，包括在无人预见到的情形下智能体应当如何行事。“监督鸿沟”则是智能体正在做的事情与用户知悉、评估和控制它的能力之间的差距。我们认为这两条鸿沟都汇聚于运行框架中——即围绕模型的工具、权限、检查机制与循环。目前运行框架由开发者编写，并对用户隐藏。我们呼吁……（原文摘要在此处截断）

    arXiv:2611.31739v1 Announce Type: cross  Abstract: Norman's gulfs of execution and evaluation have guided interaction design for forty years, but both assume a user who is present in the loop. Agentic delegation breaks that assumption: users describe an outcome and leave, while the agent takes hundreds of unseen actions. Starting from a 2026 incident in which a research agent circumvented an access refusal on an Australian government portal and went unnoticed for two months, we argue that delegation opens two new gulfs. The gulf of specification is the gap between what a user wants and their ability to express it, including how the agent should behave in situations nobody foresaw. The gulf of oversight is the gap between what an agent is doing and the user's ability to know, assess, and control it. We argue that both gulfs meet in the harness, the tools, permissions, checks, and loop that surround the model. Harnesses are currently written by developers and hidden from users. We call o
    
[^209]: GreenDynaMix：温度感知的动态混合精度MoE大语言模型推理系统

    GreenDynaMix: Thermal-Aware Dynamic Mixed-Precision MoE LLM Inference Systems

    [https://arxiv.org/abs/2611.31737](https://arxiv.org/abs/2611.31737)

    GreenDynaMix是首个将DVFS温度管理与动态混合精度调优相结合的运行时系统，它利用MoE路由器的重要性信号为关键专家分配更高精度，在降低GPU热老化率25-33%的同时保持了推理精度。

    

    持续的大语言模型（LLM）推理会产生大量热量，加剧硅片老化并降低GPU的可靠性。由于GPU温度会随工作负载强度和环境条件持续波动，将其保持在安全范围内需要实时频率调节，而这不可避免地会以吞吐量或精度为代价。为解决这一紧密耦合的挑战，我们提出了GreenDynaMix，这是首个将基于DVFS的温度管理器与动态精度调优吞吐量补偿器联合集成的运行时系统。一个敏感性感知分配器利用来自混合专家路由器的内在重要性信号，为关键专家分配更高的精度，从而保持准确性。在服务器、台式机和嵌入式平台上的实验表明，GreenDynaMix将LLM推理过程中温度引起的GPU老化率降低了25-33%，且无需离线剖析或模型再训练。

    arXiv:2611.31737v1 Announce Type: new  Abstract: Sustained Large Language Model (LLM) inference generates significant heat that exacerbates silicon aging and reduces GPU reliability. Because GPU temperature fluctuates continuously with workload intensity and ambient conditions, keeping it in a safe range requires real-time frequency scaling, which inevitably comes at the cost of either throughput or accuracy. To address this tightly coupled challenge, we propose GreenDynaMix, the first runtime system to jointly integrate a DVFS-based thermal manager with a dynamic precision-tuning throughput compensator. A sensitivity-aware allocator leverages intrinsic importance signals from the Mixture-of-Experts (MoE) router to assign higher precision to critical experts, preserving accuracy. Experiments on server, desktop, and embedded platforms demonstrate that GreenDynaMix reduces the temperature-induced aging rate of GPUs during LLM inference by 25-33%. Without offline profiling or model retrai
    
[^210]: 重现历史是不够的：针对LLM生成模拟器的干预级保障

    Reproducing History Is Not Enough: Intervention-Level Assurance for LLM-Generated Simulators

    [https://arxiv.org/abs/2611.31736](https://arxiv.org/abs/2611.31736)

    该论文提出Vouch库，通过将每个what-if问题转化为结构化更改并对照程序与当前记录进行验证，实现对LLM生成模拟器单次答案可信度的干预级保障，而非仅对整个程序给出通过与否的整体判定。

    

    大型语言模型现在已经能够编写用于模拟操作流程和工作流的仿真程序。这些程序可以被复用来针对新记录和新政策回答“假如（what-if）”问题。然而，一个能够重现已记录历史的程序，在回答“假如”问题时仍可能给出错误答案：所请求的更改可能被错误地应用，或者被应用到与问题无关的记录上，或者一旦其政策被修改，程序就可能失效。现有的检查方法只对整个程序给出通过或不通过的判定，这几乎无法说明任何单个答案是否可信。我们提出了Vouch，一个用于判断生成的模拟器所给出的答案对于所询问的更改是否可信的库。每个“假如”问题都被转化为一个结构化的更改，该更改会对照程序和当前记录进行检查。该库还会记录每个程序通过或失败的更改类型，并……

    arXiv:2611.31736v1 Announce Type: new  Abstract: Large language models can now write simulation programs of operational processes and workflows. These programs can be reused to answer what-if questions on new records and policies. However, a program that reproduces recorded history can still give wrong answers to what-if questions; requested changes may be applied wrongly, or they may be applied to records other than the ones the question is about, or the program may stop being valid once its policy is revised. Current checks return a pass-or-fail verdict for the whole program, which says little about whether any single answer can be trusted. We propose Vouch, a library that judges whether an answer from a generated simulator can be trusted for the change being asked about. Each what-if question is turned into a structured change, which is checked against the program and the current records. The library also keeps a record of the kinds of change each program has passed or failed, and r
    
[^211]: MOSAIC：面向免训练微架构优化的记忆组织搜索与反模仿交叉方法

    MOSAIC: Memory-Organized Search with Anti-Imitation Crossover for Training-Free Microarchitecture Optimization

    [https://arxiv.org/abs/2611.31733](https://arxiv.org/abs/2611.31733)

    MOSAIC提出了一种无需更新语言模型权重的免训练搜索控制器，通过存储评估经验教训以及在改进停滞时对嵌入差异较大的高分策略进行反模仿交叉组合，避免生成与既有策略雷同的缓存替换策略，从而在留出工作负载上保持更优性能。

    

    大型语言模型能够生成缓存替换策略，而核心挑战在于在未被纳入搜索的工作负载上保持性能。在我们对已发表搜索方法的改编实验中，生成的代码与既有策略高度相似，余弦相似度达0.78-0.84；更广泛的对比还显示，留出工作负载上的调和平均每周期指令数（IPC）损失为1.0-2.4%。我们提出了MOSAIC，这是一种无需更新语言模型权重即可改进策略的搜索控制器。它存储先前评估得出的简洁经验教训，而非反复复制高分代码。当改进停滞时，它会选取两个代码嵌入差异较大的高分策略，并要求模型将两者的机制结合。新颖性提示和模型轮换为该搜索提供支持；受控记忆编辑实验则用于测试存储的经验教训如何影响后续的策略提议。我们采用干预级交叉验证和更广泛的对比来评估缓存替换性能。

    arXiv:2611.31733v1 Announce Type: cross  Abstract: Large language models can generate cache-replacement policies. The central challenge is to retain performance on workloads excluded from the search. In our adaptations of published search methods, generated code resembles established policies, with cosine similarities of 0.78-0.84. The wider comparison also shows held-out harmonic-mean instructions-per-cycle (IPC) losses of 1.0-2.4%. We present MOSAIC, a search controller that improves policies without updating language-model weights. It stores concise lessons from previous evaluations rather than repeatedly copying high-scoring code. When improvement stops, it selects two high-scoring policies whose code embeddings differ and asks the model to combine their mechanisms. Novelty prompts and model rotation support this search; controlled memory edits test how the stored lessons affect later proposals. We evaluate cache replacement using intervention-level cross-validation and a wider com
    
[^212]: Geo-SMILE：面向地理空间机器学习的统计模型无关可解释性与局部解释方法

    Geo-SMILE: Statistical Model-Agnostic Interpretability with Local Explanations for Geospatial Machine Learning

    [https://arxiv.org/abs/2611.31731](https://arxiv.org/abs/2611.31731)

    提出Geo-SMILE框架，通过互补的特征屏蔽与地理重定位两个扰动分支以及距离加权的替代模型，将实质性特征解释与地理空间位置响应区分开来，为地理空间机器学习提供模型无关的局部可解释性。

    

    传统的后置（post-hoc）解释方法主要是为非空间输入开发的，通常对地理背景缺乏明确的处理。我们提出Geo-SMILE，一个模型无关的框架，通过互补的扰动分支将实质性特征解释与地理位置响应区分开来。特征分支在保留观测坐标的同时屏蔽实质性预测变量；地理分支则在固定实质性预测变量的同时重新定位地理单元。遵循SMILE的统计距离加权替代模型原则，输入分布的Wasserstein距离决定联盟的局部性权重，加权Ridge替代模型用于近似期望的黑箱预测。GeoSensitivity以保留方向为符号，量化在指定设计下拟合模型对地理单元重新定位的响应。在赫尔市的犯罪预测、西雅图的房价预测等任务上（摘要原文在此处截断）……

    arXiv:2611.31731v1 Announce Type: cross  Abstract: Conventional post-hoc explanation methods were developed largely for non-spatial inputs and often provide limited explicit treatment of geographic context. We introduce Geo-SMILE, a model-agnostic framework that distinguishes substantive Feature explanation from geographic-position response through complementary perturbation branches. The Feature branch masks substantive predictors while retaining observed coordinates; the Geo branch relocates geographic units while holding substantive predictors fixed. Following the statistical-distance-weighted surrogate principle of SMILE, input-distribution Wasserstein distances determine coalition locality weights and weighted Ridge surrogates approximate expected black-box predictions. GeoSensitivity, signed in the keep orientation, quantifies fitted-model responses to geographic-player relocation under the specified design. Across crime prediction in Hull, housing-price prediction in Seattle and
    
[^213]: EventSSM：面向异步事件深度估计的因果空间-状态建模

    EventSSM: Causal Spatial-State Modelling for Asynchronous Events Event-Depth Estimation

    [https://arxiv.org/abs/2611.31720](https://arxiv.org/abs/2611.31720)

    提出EventSSM——一种用循环记忆按时间顺序处理事件数据包的因果空间-状态模型，实现异步事件关联的度量深度估计，并通过相机位姿几何传输与可微渲染桥接异步预测与稀疏帧时刻监督，在多个深度误差指标上超越已有方法。

    

    事件相机以异步方式记录亮度变化，而度量深度监督通常仅在稀疏采样的参考时间戳处可用。将事件聚合为类帧表示虽然简化了学习过程，但可能掩盖原始事件流的细粒度时间结构。我们提出了EventSSM，这是一个用于与事件相关联的度量深度估计的因果空间-状态框架，它采用由按时间顺序到达的事件数据包更新的循环记忆。为了连接异步预测与帧时刻监督，预测的事件深度利用标定的相机位姿进行几何传输，并在参考时间戳处进行可微渲染。基准评估表明，该方法在多个深度误差指标上均优于已有的事件深度方法。受控消融实验进一步考察了跨模态初始化、时间循环、个体状态、读出与监督组件以及事件数据包子采样等因素的作用。

    arXiv:2611.31720v1 Announce Type: cross  Abstract: Event cameras report asynchronous brightness changes, yet metric-depth supervision is typically available only at sparsely sampled reference timestamps. Aggregating events into frame-like representations simplifies learning but can obscure the fine-grained temporal structure of the original event stream. We introduce EventSSM, a causal spatial-state framework for event-associated metric-depth estimation using a recurrent memory updated by chronological event packets. To bridge asynchronous predictions and frame-time supervision, predicted event depths are geometrically transported using calibrated camera poses and differentiably rendered at reference timestamps. Benchmark evaluations demonstrate improvements over established event-depth methods across multiple depth-error measures. Controlled ablations further examine cross-modal initialization, temporal recurrence, individual state, readout, and supervision components, and packet subd
    
[^214]: PRETZEL：一个环肽结构预测基准

    PRETZEL: A Benchmark for Cyclic Peptide Structure Prediction

    [https://arxiv.org/abs/2611.31708](https://arxiv.org/abs/2611.31708)

    该论文提出了PRETZEL——一个包含285个环肽-蛋白复合物、涵盖三种环化化学且全部晚于模型训练截止时间的标准化基准，实现了对环肽结构预测模型零样本、可比较且含化学正确性验证的评估。

    

    大环肽是一类日益受到关注的药物形式，共折叠模型正越来越多地被用于预测其结构。然而，已发表的评估依赖小型且整理松散的靶标集合，各论文之间互不相同，导致结果缺乏可比性；此外，大多数评估仅报告构象RMSD，而不验证预测的分子在化学上是否正确。我们提出了PRETZEL（Prediction of Ring-Enclosed Topology: Zero-shot Evaluation of cycLic peptides，环封闭拓扑预测：环肽零样本评估），这是一个包含285个精心整理的环肽-蛋白质复合物的基准，涵盖三种环化化学类型，且所有数据均晚于所评估的每个模型的训练截止时间。PRETZEL包含三个组成部分：PRETZEL-SET，一个按环化化学和化学新颖性进行标注的评估集；PRETZEL-TWIST，一个规范层，将单一的标准大环描述编译为每个引擎的原生输入格式并验证返回的结构；以及PRETZEL-SALT，一个评估……（摘要在此处被截断）

    arXiv:2611.31708v1 Announce Type: new  Abstract: Macrocyclic peptides are a growing therapeutic modality, and co-folding models are increasingly used to predict their structures. Yet published evaluations rely on small, loosely curated target sets that differ from paper to paper, so results are not comparable, and most report pose RMSD alone, without checking whether the predicted molecule is chemically correct. We introduce PRETZEL (Prediction of Ring-Enclosed Topology: Zero-shot Evaluation of cycLic peptides), a benchmark of 285 curated cyclic peptide-protein complexes spanning three cyclization chemistries, all post-dating the training cutoff of every model evaluated. PRETZEL has three components: PRETZEL-SET, an evaluation set annotated by cyclization chemistry and chemical novelty; PRETZEL-TWIST, a specification layer that compiles a single canonical macrocycle description into each engine's native input format and verifies the returned structure; and PRETZEL-SALT, an evaluation s
    
[^215]: FoveaWAM：面向动态操作的基于角色感知的4D世界-动作建模

    FoveaWAM: Role-Aware 4D World-Action Modeling for Dynamic Manipulation

    [https://arxiv.org/abs/2611.31705](https://arxiv.org/abs/2611.31705)

    FoveaWAM提出按角色分配预测精度的原生4D世界-动作模型，仅对目标位姿采用毫米级SE(3)轨迹预测、对环境用稀疏3D流建模、对静态内容依赖当前帧，从而以15.1毫秒的低延迟实现动态移动物体的高效操作。

    

    要操作一个正在移动的物体，策略必须预测其动作执行那一刻的场景，然而预测所耗费的每一毫秒都会让那一刻进一步远离。场景级的预测计算代价高昂，而以目标为中心的预测又可能遗漏影响接触的附近物体。我们的核心洞察是：策略需要预测的内容可以按角色解耦——只有决定接触的目标位姿需要毫米级精度，而周围环境主要通过其留下的间隙空间和施加的运动来产生影响。我们提出了FoveaWAM，一种原生4D世界-动作模型，它根据这些角色来分配预测精度。具体而言，该模型将目标的过去与未来运动建模为度量的SE(3)轨迹，将周围环境建模为稀疏3D流，并对所有静态内容直接依赖当前帧。这种紧凑的表示使FoveaWAM能够在单块RTX 5090上以15.1毫秒的速度执行动作，比视频世界模型快约10倍（原文摘要在此处截断）。

    arXiv:2611.31705v1 Announce Type: cross  Abstract: To manipulate a moving object, a policy must predict the scene at the moment its action executes, yet every millisecond spent predicting pushes that moment further away. Scene-level forecasts are computationally expensive, whereas target-centric forecasts can omit nearby objects that affect contact. Our key insight is that what a policy must predict decouples by role: only the target's pose, which determines contact, requires millimeter precision, while the surroundings matter primarily through the clearance they leave and the motion they impart. We present FoveaWAM, a native 4D world-action model that allocates precision according to these roles. Specifically, it models the target's past and future motion as a metric SE(3) trajectory, the surroundings as sparse 3D flow, and relies on the current frame for everything static. This compact representation lets FoveaWAM act in 15.1 ms on one RTX 5090, about 10 times faster than video world
    
[^216]: CATWAY：基于校准语言模型置信度的代价敏感智能体三支机器人决策

    CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence

    [https://arxiv.org/abs/2611.31694](https://arxiv.org/abs/2611.31694)

    CATWAY 提出了一种代价敏感的三支决策门控框架，利用四种指令条件下的损失矩阵和校准的语言模型置信度概率，让机器人在执行、追问与拒绝之间做出决策，并在跨领域测试中无需重新拟合即保持有效。

    

    语言模型机器人智能体必须决定是执行、追问还是拒绝一条指令，而不安全执行的代价远高于一次多余的提问。此前没有任何机器人决策门控从错误代价推导其全部阈值、或基于校准后的概率行动，也没有任何方法在一个领域拟合后在另一个领域原样评估。我们提出 CATWAY，一种代价敏感的三支决策门控：通过对四种指令条件（清晰、模糊、不可行、不安全）设定的损失矩阵，在概率单纯形上划定执行、追问与拒绝三个区域，其中追问即是信息收集步骤。其概率通过逻辑回归将十次采样投票与模糊度、不可行性和风险的分级评分融合。CATWAY 在家务指令上拟合后即被冻结，随后在域内、两个新的机器人领域以及外部 TRUST-NAV 基准上，对三个小模型的文本级决策进行了测试。在每次决策十一次生成的情况下……（摘要在此处截断）

    arXiv:2611.31694v1 Announce Type: cross  Abstract: A language-model robot agent must decide whether to execute, question or refuse an instruction, and an unsafe execution costs far more than an unneeded question. No prior robot gate derives all its thresholds from error costs or acts on calibrated probabilities, and none was fitted in one domain and evaluated unchanged in another. We propose CATWAY, a cost-sensitive three-way decision gate: a loss matrix over four instruction conditions (clear, ambiguous, infeasible, unsafe) fixes execute, ask and refuse regions on the probability simplex, the question being the information-gathering step. Its probabilities fuse a ten-sample vote with graded scores of ambiguity, infeasibility and risk by logistic regression. Fitted on household instructions and frozen, CATWAY was tested on text-level decisions of three small models in-domain, in two new robot domains and on the external TRUST-NAV benchmark. At eleven generations per decision against on
    
[^217]: 快速探索性证据图谱及其在支持证据向政策转化对话与过程中的潜在作用

    Rapid Exploratory Maps and their potential role in supporting evidence-to-policy dialogues and processes

    [https://arxiv.org/abs/2611.31693](https://arxiv.org/abs/2611.31693)

    该论文提出利用人工智能技术构建快速探索性证据图谱，大幅缩短证据图谱绘制所需的时间，从而支持证据生产者与政策制定者之间的协作对话，更高效地确定证据综合的研究问题与范围。

    

    产出有用的证据综合（evidence syntheses）需要明确提出的研究问题，并与对其最佳范围的理解相关联。当政策制定者（或其他最终用户）为了满足对证据的需求而委托或指导系统综述或其他证据综合的生产时，这些参数很少被充分明确地规定到足以使生产得以进行的程度。相反，这些细节通常是通过证据生产者和使用者之间协作对话的过程来制定的，以确保最终产出的证据综合能够满足使用者的需求。这一过程依赖于对现有证据库的范围界定和（或多或少正式的）证据图谱绘制，而这可能需要数周甚至数月的时间，且由于主要依靠人工流程来识别和描述证据而进展缓慢。人工智能（AI）技术的最新发展为更快速、迭代和响应式地绘制相关证据图谱开辟了新的机遇，以支持……

    arXiv:2611.31693v1 Announce Type: cross  Abstract: Producing useful evidence syntheses requires clearly formulated research questions linked to an understanding of their optimal scope. When policymakers (or other end users) commission or steer the production of systematic reviews or other evidence syntheses to fulfil a demand for evidence, these parameters are rarely specified sufficiently for production to proceed. Instead, these details are typically developed via a collaborative dialogic process involving evidence producers and users to ensure the resulting synthesis will meet users' needs. This process is informed by scoping and (more-or-less formally) mapping the existing evidence base, which can take many weeks or months, slowed by applying mostly manual processes to identify and characterise the evidence. Recent developments in artificial intelligence (AI) technologies have opened up new opportunities for more rapid, iterative, and responsive mapping of relevant evidence to supp
    
[^218]: GRUN：面向图像修复的广义Retinex展开网络

    GRUN: Generalized Retinex Unfolding Network for Image Restoration

    [https://arxiv.org/abs/2611.31690](https://arxiv.org/abs/2611.31690)

    该论文提出广义Retinex公式与MAP引导的深度展开网络GRUN，将经典光照-反射分解泛化为内容-环境因子分解先验，通过交替进行反射恢复、光照估计和渐进式退化推断，实现对多种退化类型图像的修复。

    

    图像修复仍然是底层视觉领域中的一项基础性挑战，需要方法能够在处理多样化退化的同时保留场景结构与细节。我们重新审视了经典的Retinex理论，提出了一种广义Retinex公式，将传统的光照-反射分解扩展为一种受Retinex启发的内容-环境因子分解先验，用于图像修复。其核心思想是利用两个可学习因子来恢复图像：一个捕捉场景属性的内容主导响应（广义反射分量），以及一个编码与观测退化相关的任务依赖变化的环境侧响应（广义光照分量）。基于这一公式，我们开发了GRUN（广义Retinex展开网络），这是一个由最大后验概率（MAP）引导的深度展开框架，它在K个阶段中交替执行反射分量恢复、光照估计以及渐进式退化推断，并采用以退化为条件的邻近算子（摘要至此截断）……

    arXiv:2611.31690v1 Announce Type: cross  Abstract: Image restoration remains a fundamental challenge in low-level vision, requiring methods that handle diverse degradations while preserving scene structures and details. We revisit classical Retinex theory and propose a Generalized Retinex formulation that extends the illumination-reflectance decomposition into a Retinex-inspired content-environment factorization prior for image restoration. The key idea is to restore images using two learned factors: a content-dominant response (generalized reflectance) capturing scene properties and an environment-side response (generalized illumination) encoding task-dependent variations associated with the observed degradations. Based on this formulation, we develop GRUN (Generalized Retinex Unfolding Network), a MAP-guided deep unfolding framework that alternates reflectance restoration, illumination estimation, and progressive degradation inference across K stages with degradation-conditioned prox
    
[^219]: 正确的终点，错误的旅程：合成心理咨询对话缺失人类互动动态

    Right Destination, Wrong Journey: Synthetic Counseling Dialogues Miss Human Interactional Dynamics

    [https://arxiv.org/abs/2611.31686](https://arxiv.org/abs/2611.31686)

    该研究提出以轨迹为中心的过程保真度评估方法，发现LLM生成的心理咨询对话虽能模拟合理的最终结果，但在恢复动态和逐轮响应等人类互动动态方面保真度不足，且即使采取针对性的生成干预也无法完全恢复这些互动特性。

    

    LLM生成的心理咨询对话越来越多地被用于训练和评估心理健康系统，然而现有评估大多聚焦于回复质量和最终结果，而非心理支持如何随时间逐步展开。我们对十个人类、重构以及合成的心理咨询数据集开展了以轨迹为中心的过程保真度评估。结果表明，合成对话往往能够达到看似合理的终点，但在恢复动态和逐轮响应方面与人类支持存在差异。尽管生成设计可以改变恢复时机和轨迹结构等全局属性，但在各类模型和生成范式下，患者与咨询师之间的条件性响应保真度始终较低。我们在受控合成数据集上的实验表明，针对性的生成干预能够在若干轨迹层面的指标上缩小与真实数据的差距，但无法完全恢复人类的互动动态。

    arXiv:2611.31686v1 Announce Type: cross  Abstract: LLM-generated counseling dialogues are increasingly used for training and evaluating mental-health systems, yet existing evaluations focus largely on response quality and final outcomes rather than how support unfolds over time. We conduct a trajectory-centered evaluation of process fidelity across ten human, reconstructed, and synthetic counseling datasets. Our results show that synthetic dialogues often reach plausible endpoints but differ from human support in recovery dynamics and turn-by-turn responsiveness. While generation design can shift global properties such as recovery timing and trajectory structure, contingent patient--counselor responsiveness remains consistently lower in fidelity across models and generation paradigms. Our experiments on controlled synthetic datasets show that targeted generation interventions can narrow the gap with real data on several trajectory-level measures, but do not fully recover interactional 
    
[^220]: CROSS-Calib：面向超声重建的选择性通道校准方法

    CROSS-Calib: Selective Channel Calibration for Ultrasound Reconstruction

    [https://arxiv.org/abs/2611.31684](https://arxiv.org/abs/2611.31684)

    提出CROSS-Calib方法，利用其他通道的预测作为参考进行幅度和相位校准，并通过上下文门控机制选择性地融合校准测量与预测回波，在波束形成前修复不可靠通道数据，从而显著提升超声重建的图像质量。

    

    超声成像能够实时呈现内部组织的视图，用于医学诊断。然而，超声信号可能受到干扰的影响，如何在保留有效回波的同时纠正误差并消除瞬时干扰仍然具有挑战性。我们提出了CROSS-Calib，这是一种通道恢复方法，它利用来自其他通道的预测作为幅度和相位校准的参考，同时通过一个基于上下文信息的门控机制，选择性地将校准后的测量值与预测回波相结合，以保留可用的样本并修复不可靠的样本。在体内测试集上，CROSS-Calib在增益误差和瞬时干扰条件下优于所有基线模型；体模实验表明，该模型能够恢复囊肿对比度，且线目标的回波宽度接近干净参考图像。CROSS-Calib可以在波束形成之前执行通道校准和选择性修复，从而提高回波可用性和图像质量。

    arXiv:2611.31684v1 Announce Type: new  Abstract: Ultrasound imaging provides views of internal tissue in real time for medical diagnosis. However, ultrasound signals may be affected by interference, and correcting errors and removing transient interference while preserving valid echoes remains challenging. We propose CROSS-Calib, a channel recovery method that uses predictions from other channels as references for amplitude and phase calibration, while a gate using context selectively combines calibrated measurements with predicted echoes to retain usable samples and repair unreliable ones. On the in vivo test set, CROSS-Calib outperforms all baseline models under gain errors and transient interference, while phantom tests show that the model restores cyst contrast with echo widths of wire targets close to the clean reference. CROSS-Calib can perform channel calibration and selective repair before beamforming to improve echo usability and image quality.
    
[^221]: LimoBench：大语言模型有多“懂”街道？

    LimoBench: How Street Smart Are Large Language Models?

    [https://arxiv.org/abs/2611.31682](https://arxiv.org/abs/2611.31682)

    提出了首个街道级地理空间基准LimoBench，通过82个基于真实地图数据的自然语言位置查询，系统评估大语言模型在地理编码、路径规划、$k$近邻搜索等七种空间操作上的位置感知与精确回答能力。

    

    大语言模型（LLMs）已经改变了人类与信息交互的方式，实现了跨领域的自然语言交互界面。它们正越来越多地被部署于地理空间应用中，包括基于地图的问答、基于位置的服务，以及空间数据库的自然语言接口。但LLMs是否真正具备位置感知能力，能够可靠地解释并精确回答涉及真实世界坐标、兴趣点（POI）和地理空间距离的位置查询？为了具体回答这个问题，我们提出了LimoBench，一个街道级基准测试，包含82个自然语言位置查询，涵盖七种经典空间操作：地理编码、逆地理编码、路径规划、方向判断、范围搜索、$k$近邻搜索和空间连接。这些查询基于印第安纳州西拉法叶市的真实地图数据构建，并按照位置和语义复杂度进行分级，旨在探测LLM的街道级空间……

    arXiv:2611.31682v1 Announce Type: new  Abstract: Large language models (LLMs) have transformed how humans interact with information, enabling natural-language interfaces across various domains. They are increasingly deployed for geospatial applications including map-based question answering, location-based services, and natural-language interfaces to spatial databases. But are LLMs truly location-aware, able to reliably interpret and precisely answer location queries involving real-world coordinates, points of interest (POIs), and geospatial distances? To answer this concretely, we present LimoBench, a street-level benchmark of 82 natural-language location queries over seven canonical spatial operations: geocoding, reverse geocoding, routing, directionality, range search, $k$-nearest-neighbor search, and spatial join. The queries are grounded in real map data of West Lafayette, Indiana, and are graded by location and semantic complexity. They are designed to probe LLMs' street-level sp
    
[^222]: 三思而后行：面向多模态扩散语言模型的视觉证据引导解码

    Look Before You Commit: Visual-Evidence-Guided Decoding for Multimodal Diffusion Language Models

    [https://arxiv.org/abs/2611.31677](https://arxiv.org/abs/2611.31677)

    论文发现扩散多模态大模型中近九成高置信度词元提交与图像证据无关，并提出无需训练的视觉证据引导解码方法CD-APC，通过证据调整评分、双重门控与逆窗口机制，在两个骨干模型和五个基准上持续提升性能、抑制幻觉。

    

    扩散多模态大语言模型依赖置信度来决定在生成过程中何时提交候选词元。然而，在多模态任务中，置信度无法区分候选词元究竟是源自图像证据还是语言先验。我们通过成对前向消融实验发现，近九成的高置信度提交对图像并不敏感，而退化和幻觉样本的深层注意力组织熵呈现出显著增加。基于这一诊断，我们提出了CD-APC，一种无需训练的解码方法。CD-APC在基础支持池内进行证据调整评分，并采用基于模型可接受度与对比集中度的双重门控来决定是否提交词元，同时引入逆窗口机制以自适应不同步骤间的难度。在两个骨干模型和五个基准测试上，CD-APC持续提升了性能，表明……

    arXiv:2611.31677v1 Announce Type: new  Abstract: Diffusion multimodal large language models rely on confidence to decide when to commit candidate tokens during generation. In multimodal tasks, however, confidence cannot distinguish whether a candidate originates from image evidence or the language prior. We perform paired forward ablation and find that nearly nine out of ten high-confidence commits are insensitive to the image, while degenerate and hallucination samples exhibit significant increases in the entropy of deep-layer attention organization. Based on this diagnosis, we propose CD-APC, a training-free decoding method. CD-APC performs evidence-adjusted scoring within a base-supported pool and uses dual gating based on model acceptability and contrastive concentration to decide whether to commit, together with an inverse-window mechanism that adapts to inter-step difficulty. Across two backbones and five benchmarks, CD-APC consistently improves performance, indicating that stabl
    
[^223]: 多语言压缩中的来源信息丢失与修复

    Provenance Loss and Repair in Multilingual Compaction

    [https://arxiv.org/abs/2611.31668](https://arxiv.org/abs/2611.31668)

    该论文发现多语言对话压缩会因语法性别等语言差异而丢失属性的来源信息（无法区分用户明确声明与语法默认），并提出一种无需训练、覆盖写作标注、丢弃前似然验证和阅读信任三个环节的修复方法，大幅恢复被压缩丢弃的已声明属性召回率。

    

    arXiv:2611.31668v1 公告类型： cross 摘要：会话型助手会将较早的对话消息压缩成简短笔记并丢弃原始消息。当笔记所用的语言需要作出英语中不作区分的区分时，例如西班牙语形容词的性属变化（agotado vs. agotada），省略该属性或明确声明该属性的消息可能产生相同的笔记。在原始消息被丢弃后，后续对话无法再区分这两种情况：将词形变化按字面理解会断言一个用户从未提供过的属性，而将其视为常规语法而忽略则会丢弃用户确实提供过的属性。我们提出了一种无需训练的修复方法，涵盖写作、丢弃前验证和阅读三个环节：写入者用成对形式（如 agotado/a）标注未声明的属性，丢弃前的似然检查将缺乏依据的单式形式改写为成对形式，而阅读者则将剩余的单式形式视为已声明的事实。在五个 Qwen3.5 和 Gemma-4 模型上，消息丢弃使问答（QA）中对已声明属性的召回率从 90% 降至 34%；我们的修复方法……

    arXiv:2611.31668v1 Announce Type: cross  Abstract: Conversational assistants compact older messages into short notes and evict the originals. When the note's language requires a distinction that English leaves open, such as Spanish adjective gender (agotado vs. agotada), messages that omit or explicitly state that attribute can yield the same note. After eviction, later turns cannot tell them apart: taking the inflected form literally asserts an attribute the user never gave, whereas discounting it as routine grammar discards one the user did give. We propose a training-free repair across writing, pre-eviction verification and reading. The writer marks unstated attributes with paired forms (e.g., agotado/a), a likelihood check before eviction rewrites unsupported single forms into paired ones, and the reader trusts remaining single forms as stated facts. On five Qwen3.5 and Gemma-4 models, eviction cuts recall of stated attributes from 90% to 34% in question answering (QA); our repair 
    
[^224]: BRAG：贝叶斯检索增强生成

    BRAG: Bayesian Retrieval Augmented Generation

    [https://arxiv.org/abs/2611.31656](https://arxiv.org/abs/2611.31656)

    BRAG提出了一种贝叶斯检索生成模型，通过闭式滤波器在线更新每个记忆块的少量键和标量，将检索框架化为控制问题，并证明现有检索反馈方法均为该模型的特殊或极限情形。

    

    检索增强生成（RAG）根据与查询的语义相似度对记忆块进行排序。许多方法通过从检索内容的反馈中学习来改进这一方式，但每种方法都是独立的启发式方法，没有说明它在何时才是正确的选择。我们提出了BRAG，一个结合使用情况的检索生成模型。每个块从锚定在其原始嵌入上的von Mises-Fisher分量混合中生成它所回答的查询，并且每个块拥有一个随时间漂移的潜在效用。该模型给出了最优推理——一种闭式滤波器，它为每个块只存储少量键和两个标量，并根据管理者的标签在线更新它们，同时将检索框架化为一个控制问题。该模型还阐明了现有方法在何时是最优的：查询平均、Rocchio风格的反馈和频率重排序是参数固定的特殊情况，NUDGE是每块单个键且无动态的情形，而LADER则是单键的极限情况。

    arXiv:2611.31656v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) ranks memory chunks by semantic similarity to the query. Many methods improve on this by learning from feedback on what was retrieved, but each is a separate heuristic, with no account of when it is the right choice. We introduce BRAG, a generative model of retrieval with use. Each chunk generates the queries it answers from a mixture of von Mises-Fisher components anchored at its original embedding, and each chunk has a latent utility that drifts over time. The model gives the optimal inference, a closed-form filter that stores a few keys and two scalars per chunk and updates them online from the labels of a curator, and it frames retrieval as a control problem. It also shows when existing methods are optimal. Query averaging, Rocchio-style feedback and frequency reranking are special cases with fixed parameters, NUDGE is the case of one key per chunk with no dynamics, and LADER the limit of one key 
    
[^225]: AI推理工作负载中的搁浅算力：通过需求聚合实现价值回收

    Stranded Compute in AI Inference Workloads: Recovering Value Through Demand Aggregation

    [https://arxiv.org/abs/2611.31654](https://arxiv.org/abs/2611.31654)

    本文指出AI推理成本中资本回收占87.3%而电力仅3.7%，其价值损失主要源于调度闲置和搁浅的突发预留，通过在多个不完全相关的租户间聚合需求可回收7.3个百分点的有效产出，相当于每兆瓦IT容量每年约75万美元。

    

    AI推理基础设施通常被视为能源业务来分析。但在全成本核算下并非如此：在本文建模的参考部署中，资本回收占每token完全分摊成本的87.3%，而电力仅占3.7%。由于资本按时间累积而与使用与否无关，加速器的价值是被非利用而非消耗所摧毁的。然而，符合SLO要求的有效产出仅达到额定吞吐量的59.6%，对损失链的分解显示40.4个损失百分点中有25.2个百分点源于调度闲置和搁浅的突发预留容量。这两者都是租户需求组合的属性而非其芯片的属性，且均无法由单个租户独立行动来分散。通过将预留和闲置容量建模为可分散的风险，在八个相关性不完美的租户（rho = 0.3）之间聚合需求，可以回收7.3个百分点的有效产出，相当于每兆瓦IT容量每年约75万美元。

    arXiv:2611.31654v1 Announce Type: new  Abstract: AI inference infrastructure is commonly analyzed as an energy business. Under full total cost accounting it is not: in the reference deployment modeled here, capital recovery is 87.3% of fully loaded cost per token and electricity 3.7%. Because capital accrues per unit time irrespective of use, accelerator value is destroyed by non-utilization rather than by consumption. Yet, SLO qualified goodput reaches only 59.6% of nameplate throughput, and the decomposition of the loss chain shows that 25.2 of the 40.4 lost percentage points arise from scheduling idleness and stranded burst reserve. Both are properties of a tenant's demand portfolio rather than of its silicon, and neither is diversifiable by a single tenant acting alone. Modelling reserve and idle capacity as diversifiable risk, pooling demand across eight imperfectly correlated tenants (rho = 0.3) recovers 7.3 pp of goodput, approximately $0.75M per annum per megawatt of IT capacit
    
[^226]: 用人工智能构建物理机制

    Constructing physical mechanisms with artificial intelligence

    [https://arxiv.org/abs/2611.31653](https://arxiv.org/abs/2611.31653)

    该研究构建了一个读取 arXiv 全部 250 万篇论文方程的方程原生系统，通过确定性计算在不同学科间拆解、转移和组合方程部件来构建物理机制，并在部件不匹配时自动生成缺失项的方程。

    

    机制在各个科学领域中反复出现：固体中的热传导方程同样可以描述分子扩散或布朗运动粒子。一个机制很少能用单个方程完全表达；其运行方式、边界、参数和闭合关系分布在推导的不同步骤中，而那些只读取文本或拟合公式的方法往往会遗漏这些内容。我们构建了一个以方程为原生对象的系统，读取了 arXiv 全部 250 万篇论文中的方程，并证明机制可以从这些方程中编译出来，通过确定性计算加以构建。一个构造器可以在不同学科之间拆解、转移、展开、组合和连接方程的各个部件；当部件无法匹配时，失败本身就成了描述缺失项的方程。这些构造将一个阈值反应器变成由弱信号写入的存储器，通过两次各自单独都会失败的改动将一个振荡器变成开关，并将一个节律变成化学量的调节器，每一种构造都给出了可用于实验检验的条件。

    arXiv:2611.31653v1 Announce Type: new  Abstract: Mechanisms recur across the sciences: the equation for heat in a solid also describes molecular diffusion or a Brownian particle. A mechanism is rarely written in one equation; its operation, boundaries, parameters and closures are stated at different steps of a derivation, where methods that read text or fit formulas miss it. We built an equation-native system that reads the equations of all 2.5 million arXiv manuscripts, and show that mechanisms can be compiled from them and constructed by deterministic calculation. A constructor detaches, transfers, unfolds, composes and joins their parts across disciplines; where parts do not fit, the failure is an equation for the missing term. Constructions turn a threshold reactor into a memory that a weak signal writes, an oscillator into a switch through two changes that each fail alone, and a rhythm into the regulator of a chemical amount, each with conditions for an experimental test.
    
[^227]: E2SIM：基于演化状态的日常人类行为长期模拟

    E2SIM: Long-Horizon Simulation of Everyday Human Behavior with Evolving States

    [https://arxiv.org/abs/2611.31641](https://arxiv.org/abs/2611.31641)

    E2SIM 提出了一个通过反馈循环将行动者演化状态与动态共享环境联合建模的纵向用户模拟框架，使早期行动的后果能够塑造后续行为，同时保留各行动者视角的差异，并配套了衡量动态一致性的纵向轨迹诊断指标。

    

    AI 智能体越来越被期望能够在长期内、跨工作与日常生活为用户提供支持。这不仅需要建模用户是谁，还需要建模他们的目标、兴趣、活动和互动如何随时间演变。纵向用户模拟的一大挑战在于保持这些演化状态与共享环境中行动后果之间的一致性。我们提出了 E2SIM，一个联合建模行动者演化状态与动态共享环境的纵向用户模拟框架。E2SIM 通过一个反馈循环将行动者状态更新与共享世界的变化相连接，该循环串联了即时行动、延迟结果以及跨工作与生活的行动者特定观察。这使得早期行动的后果能够影响后续行为，同时保留每个行动者所知信息的差异。我们进一步提出了一套纵向轨迹诊断指标，用于衡量动态一致性、连贯事件……

    arXiv:2611.31641v1 Announce Type: new  Abstract: AI agents are increasingly expected to support users over long periods, across work, and daily life. This requires modeling not only who users are, but also how their goals, interests, activities, and interactions evolve over time. A challenge in longitudinal user simulation is maintaining consistency between these evolving states and the consequences of actions in a shared environment. We introduce E2SIM, a longitudinal user simulation framework that jointly models evolving actor states and a dynamic shared environment. E2SIM connects actor-state updates with shared-world changes through a feedback loop linking immediate actions, delayed outcomes, and actor-specific observations across work and life. This allows the consequences of earlier actions to shape later behavior, while preserving differences in what each actor knows. We further propose a suite of longitudinal trajectory diagnostics, measuring dynamic consistency, coherent event
    
[^228]: Wavefront：面向推测解码的低秩草稿生成与并行因果精炼

    Wavefront: Low-Rank Drafting with Parallel Causal Refinement for Speculative Decoding

    [https://arxiv.org/abs/2611.31638](https://arxiv.org/abs/2611.31638)

    提出Wavefront推测解码方法：用学习到的低秩基础头替代昂贵的完整lm_head，并以整块草稿前缀为条件、通过并行Jacobi迭代精炼的更强修正头来恢复块内token依赖，实现高效且无数据依赖控制流的并行草稿生成。

    

    推测解码通过让一个廉价的草稿模型提出token，再由目标模型并行验证，从而加速大语言模型的推理。DFlash、P-EAGLE等并行块草稿模型可以在一次前向传播中提出整个token块，但它们丢失了块内token之间的依赖关系；DSpark通过一个按顺序运行的轻量级修正头恢复了部分依赖。在测量DSpark时，我们发现该修正头在大多数位置上会覆盖基础预测，因此通过目标模型完整lm_head进行的昂贵基础读取，大多只是起到一个种子（初始猜测）的作用。为此，我们提出了Wavefront，它遵循一个简单的原则：缩小基础部分、强化修正头。Wavefront用学习到的低秩基础取代完整的lm_head，让修正头以同一块内整个草稿前缀为条件，并通过若干次并行的Jacobi迭代对其进行求解。固定的迭代次数预算使得草稿步骤不含数据依赖的控制流，因此我们可以……（原文在此处截断）

    arXiv:2611.31638v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM inference by letting a cheap drafter propose tokens that the target model verifies in parallel. Parallel block drafters such as DFlash and P-EAGLE propose a whole block in one forward pass, but they lose the dependence between tokens in the block; DSpark restores part of it with a lightweight correction head that runs sequentially. Measuring DSpark, we find that this head overrides the base prediction on most slots, so the expensive base readout through the target's full lm_head mostly provides a seed. We therefore propose Wavefront, which follows a simple principle: shrink the base and strengthen the correction head. Wavefront replaces the full lm_head with a learned low-rank base, conditions the correction head on the whole drafted prefix within the same block, and resolves it with a few parallel Jacobi passes. A fixed pass budget makes the draft step free of data-dependent control flow, so we captu
    
[^229]: 所见即所证：面向未见组合的基于概念的证书

    What You See Is What You Certify: Concept-Based Certificates for Unseen Compositions

    [https://arxiv.org/abs/2611.31627](https://arxiv.org/abs/2611.31627)

    该研究揭示了基于概念的模型在未见属性组合上会以高达93-94%的概率错误认证类别，并将本体论视为错误检测码，据此确立了有效决策证书的三个条件：排除所有竞争类别的可见证据、如实标注未见部件、以及不受共现捷径驱动的罕见误读。

    

    基于概念的模型通过它们声称“看到”的概念来为决策提供解释，并通过显式的本体论进行解码。然而，众所周知，概念预测会泄露标签信息、忽略局部性，并且在新的属性组合上缺乏可靠基础。由这类概念解码出的决策即使满足本体论仍可能是错误的，因此我们追问：它在什么情况下才能作为一份证书。在熟悉部件的未见组合上，按标准方式训练的联合概念瓶颈模型和概念嵌入模型，即使本体论正确，也有93-94%的时间认证了错误的类别（对于仅在可见部件上训练并由定位器门控的联合模型，这一比例为60%）。联合模型的部件错误比随机水平更频繁地复制共现的属性值（跨概念捷径现象）。通过将本体论视为一种错误检测码，我们证明了有效的证书需要满足：具有排除所有竞争类别的可见证据、将未见部件如实标注为未见、以及误读罕见且不受共享模式驱动的特性。

    arXiv:2611.31627v1 Announce Type: cross  Abstract: Concept-based models justify a decision by concepts they claim to see, decoded through an explicit ontology. Yet concept predictions are known to leak label information, ignore locality, and be poorly grounded on new attribute combinations. A decision decoded from such concepts can satisfy the ontology and still be wrong, so we ask when it can serve as a certificate. On unseen combinations of familiar parts, a joint concept bottleneck model and a concept embedding model, trained as is standard, certify wrong classes 93-94% of the time despite a correct ontology (60% for a joint model trained only on visible parts and gated by a localizer). The joint model's part errors copy co-occurring values more often than chance (cross-concept shortcutting). Treating the ontology as an error-detecting code, we show that a certificate needs visible evidence excluding every rival, unseen parts reported as such, and rare misreads not driven by a share
    
[^230]: 通过从局部续写中学习摘要实现自我提升的长程推理

    Self-Improving Long-Horizon Reasoning by Learning Summaries from Local Continuations

    [https://arxiv.org/abs/2611.31621](https://arxiv.org/abs/2611.31621)

    提出TruSuCo方法，通过Delta-KL奖励训练独立摘要器将长推理历史压缩为紧凑状态，利用模型自身在短上下文中的可靠表现实现长程推理能力的自我提升，而无需最终答案奖励或更强模型的蒸馏。

    

    推理模型会随着上下文的增长而性能退化，因此更长的推理轨迹本身并不能带来更长的有效推理范围。扩展推理范围的一种方法是定期将推理历史压缩成一个紧凑的状态，并从该状态继续推理。现有方法使用最终答案奖励来训练该状态，或从更强的模型中蒸馏得到，这两种信号都无法专门衡量状态本身的贡献。我们观察到，在长轨迹上性能退化的推理模型在短上下文中仍然表现可靠，我们的目标是在压缩推理历史时保留这种短上下文行为。受此启发，我们使用Delta-KL训练一个独立的摘要器——Delta-KL是一种由冻结的推理模型对采样续写的token级预测计算得到的奖励——并通过截断轨迹、生成摘要、再从摘要继续的方式进行推理（TruSuCo）。训练仅使用几千个问题……

    arXiv:2611.31621v1 Announce Type: new  Abstract: Reasoning models degrade as their context grows, so longer reasoning traces do not by themselves give longer effective horizons. One way to extend the horizon is to compress the reasoning history into a compact state from time to time and continue from the state. Existing methods train the state with the final-answer reward or distill it from a stronger model. Neither signal is tailored to measure the contribution of the state alone. We observe that a reasoning model that degrades over long trajectories still behaves reliably in short contexts. We aim to preserve this short-context behavior when compressing the reasoning history. Motivated by this, we train a separate summarizer with Delta-KL, a reward computed from the frozen reasoning model's token-level predictions on sampled continuations, and run inference by truncating the trace, summarizing it, and continuing from the summary (TruSuCo). Training uses a few thousand questions with 
    
[^231]: 用于自动发现偏微分方程帕累托最优元求解器的多智能体系统

    Multi-agent System for Automatic Discovery of Pareto-optimal Meta-solvers for PDEs

    [https://arxiv.org/abs/2611.31611](https://arxiv.org/abs/2611.31611)

    该论文提出一种多智能体框架，让语言模型智能体与非支配排序遗传进化算法相结合，自动演化出针对偏微分方程的帕累托最优混合求解器，仅需单次运行即可在小模型上产生一组求解器权衡方案。

    

    为离散化偏微分方程设计线性求解器需要针对具体问题的专业知识。我们提出了一个多智能体框架，其中语言模型智能体负责提出、批判、实现和评估由Krylov方法、预条件子、块分解和神经算子组装而成的求解器配置。在提出智能体内部，一种非支配排序遗传进化算法使元求解器种群朝着六项测量指标的帕累托前沿演化，从而单次运行即可返回一组权衡方案，而非单一调优后的求解器。该系统运行于单台笔记本电脑上的小型开源权重语言模型，并在四个问题上与单智能体系统进行了比较。对于三维声散射和弹性问题，它在60个候选方案中分别获得了47个和48个收敛的混合求解器。相对于采用Jacobi预条件的Krylov基线，所发现的最优求解器将求解时间……（原文摘要在此处截断）

    arXiv:2611.31611v1 Announce Type: new  Abstract: Designing linear solvers for discretized partial differential equations requires problem-specific expertise. We present a multi-agent framework, in which language-model agents propose, critique, implement, and evaluate solver configurations assembled from Krylov methods, preconditioners, block factorizations, and neural operators. Inside the proposing agent, a non-dominated sorting genetic evolutionary algorithm evolves the population of meta-solvers toward the Pareto front of six measured quantities, so that one run returns a front of trade-offs rather than a single tuned solver. The system runs on small open-weight language models served on a single laptop, and we compare it with a single agent on four problems. For three-dimensional acoustic scattering and elasticity, it delivers 47 and 48 convergent hybrid solvers out of 60 candidates. Relative to the Jacobi-preconditioned Krylov baselines, the best of the discovered solvers reduce t
    
[^232]: 面向后Astra时代的可监控性设计架构：一种基于隔离化的方法

    Monitorable-by-design architectures for a post-Astra world: a compartmentalization-based approach

    [https://arxiv.org/abs/2611.31610](https://arxiv.org/abs/2611.31610)

    提出一种基于隔离化的LLM架构改进方案，通过将危险能力分区隔离，使其使用（尤其是可能预示策略性对抗行为的危险推理）能够以比思维链监控更低成本、且在可监控思维链失效后仍然有效的方式进行监控。

    

    像GPT-6 Astra这样的前沿模型越来越能够在没有可监控思维链的情况下进行大量推理，思维链正变得愈发难以解读，并且为了降低成本常常被异步监控。由于思维链监控的未来尚不明朗，可能有必要在设计大语言模型时就考虑到可监控性。我们提出了一种对现有LLM架构的改进方案，该方案基于GRAM、SGTM、DEMix、BTS及类似方法，对危险能力进行隔离化处理，使其使用——尤其是与可能表明策略性对抗性使用的危险推理类型相结合时——能够以比思维链监控更廉价的方式进行监控，且这种监控方式可能比可监控思维链存续得更久。我们提出了该架构的若干变体，旨在平衡各种权衡取舍，并讨论了其他好处，例如能力移除。

    arXiv:2611.31610v1 Announce Type: new  Abstract: Frontier models like GPT-6 Astra are increasingly able to do substantial reasoning without a monitorable chain of thought, chains of thought are becoming less legible, and are often monitored asynchronously to reduce cost. With the future of chain of thought monitoring unclear, it may be necessary to design LLMs with monitorability in mind. We propose a modification to existing LLM architectures based on GRAM, SGTM, DEMix, BTS, and similar methods to compartmentalize dangerous capabilities so that their use, especially in combination with dangerous reasoning types that may indicate strategic adversarial use, can be monitored more cheaply than chain of thought in a way that might outlast monitorable chain of thought. We propose several variants of this architecture designed to balance tradeoffs and discuss other benefits, such as capability removal.
    
[^233]: RePT：三维几何从二维关系预训练中涌现

    RePT: 3D Geometry Emerges from 2D Relational Pretraining

    [https://arxiv.org/abs/2611.31601](https://arxiv.org/abs/2611.31601)

    RePT通过跨视角二维点预测学习物理点之间的关系，使三维几何结构无需三维预测目标即可从二维关系预训练中自然涌现，并展现出跨数据集的泛化能力。

    

    arXiv:2611.31601v1 公告类型： cross 摘要：学习空间表示通常依赖于显式预测几何量，但物理几何如何在学到的表示中被组织起来仍知之甚少。我们探究三维结构能否从跨二维观测的关系学习中涌现，而无需三维预测目标。我们提出RePT，通过跨视角二维点预测来学习不同物理点之间的紧凑关系。学习到的关系空间形成了几何组织，其中幅值、角度和邻域关系能够反映物理三维位移。在16个场景上拟合的单一线性读出层，无需重新拟合即可在39个保留场景上恢复位移长度和角度（Pearson r 分别为0.72和0.78）。在ARKitScenes上训练的模型检查点在ScanNet++上也保持了强几何组织，展现了泛化能力。在受控比较中，不同点关系预测持续产生更强的几何组织。

    arXiv:2611.31601v1 Announce Type: cross  Abstract: Learning spatial representations often relies on explicitly predicting geometric quantities, but how physical geometry is organized within learned representations remains less understood. We ask whether 3D structure can emerge from learning relations across 2D observations, without 3D prediction targets. We introduce RePT, which learns compact relations between distinct physical points through cross-view 2D point prediction. The learned relation space develops a geometric organization in which magnitudes, angles, and neighborhoods reflect physical 3D displacement. A single linear readout fitted on 16 scenes recovers displacement length and angle on 39 held-out scenes without refitting (Pearson r = 0.72 and 0.78). The ARKitScenes-trained checkpoint also retains strong geometric organization on ScanNet++, showcasing generalizability. In controlled comparisons, distinct-point relational prediction consistently yields stronger geometric or
    
[^234]: GoldiBench：工具选对了，用量却错了——面向专业编辑中效果判断能力的智能体基准测试

    GoldiBench: Right Tools, Wrong Amount -- Benchmarking Agents on Effect Judgement in Professional Editing

    [https://arxiv.org/abs/2611.31590](https://arxiv.org/abs/2611.31590)

    论文提出GoldiBench基准，通过500个专业编辑任务将“工具操作”与“效果判断（用量把握）”分离评估，发现智能体操作能力很强，但难以准确判断实现目标视觉效果所需的工具用量。

    

    真实的编辑需求描述的是一种视觉效果，而非达成该效果的具体步骤，因此服务于这类需求的智能体必须判断每种工具需要使用多少才能实现该效果。现有基准并未衡量这一点：它们的任务要么直接给出工具和参数值，要么对用量不加限定，并且依赖VLM裁判评分——而这些裁判会让超过一半的半强度和三倍强度结果通过。我们提出了GoldiBench，一个包含500个编辑任务的基准，涵盖五个专业应用（darktable、RawTherapee、GIMP、Inkscape、Blender），任务既可以通过编写程序解决，也可以通过操作GUI完成。每个任务将一张START图像与一个由隐藏程序渲染出的TARGET图像配对，因此用量由已知目标固定；智能体的编辑结果会被重新渲染并与TARGET进行比较，且同一任务以三种形式呈现：给出工具和参数值、仅给出工具、或仅给出目标，从而将操作能力与效果判断区分开来。在25个开源、GUI和闭源模型中，智能体的操作能力表现良好（4B以上模型使用程序可达99-100%……

    arXiv:2611.31590v1 Announce Type: cross  Abstract: Real editing requests describe a look, not the steps to reach it, so an agent that serves them must judge how much of each tool the look takes. Existing benchmarks do not measure this: their tasks either state the tools and values, or leave the amount open and are scored by VLM judges that pass more than half of half- and triple-strength results. We introduce GoldiBench, a benchmark of 500 editing tasks in five professional applications (darktable, RawTherapee, GIMP, Inkscape, Blender), solvable by writing a program or by operating the GUI. Each task pairs a START image with a TARGET rendered from a hidden program, so the amount is fixed by a known target; the agent's edit is re-rendered and compared with TARGET, and the same task is posed with the tools and values, the tools only, or the goal only, separating operation from effect judgement. Across 25 open, GUI and closed models, agents operate well (99-100% with programs from 4B up w
    
[^235]: LatentLens：重新思考多智能体系统中的潜在接口

    LatentLens: Rethinking Latent Interface in Multi-Agent Systems

    [https://arxiv.org/abs/2611.31582](https://arxiv.org/abs/2611.31582)

    本文提出LatentLens框架，从外部和内部两个层面解耦多智能体系统中的潜在接口，发现在相同智能体框架下引入潜在接口并不能提升甚至可能显著降低性能，其效果取决于重新对齐兼容性、信息可访问性和传播动态等内部因素。

    

    潜在接口使多智能体系统中的智能体能够通过连续表示而非解码后的文本进行通信。先前的研究报告了性能和效率上的提升，但既没有将潜在通信与智能体框架（harness）分离开来，也没有考察其内部组件，导致其实际贡献尚不明确。我们提出了LatentLens，一个在两个层面解耦潜在接口的框架：在外部将其与智能体框架解耦，以衡量其对整体性能的贡献（RQ1）；在内部将其解耦为重新对齐兼容性、信息可访问性和传播动态，以识别各因素的优势与局限（RQ2）。在基于Qwen3模型的七个基准测试中，在相同的智能体框架上添加潜在接口并不能提升性能，甚至可能显著降低准确率。在内部，潜在表示保留了一定的领域信息，但不兼容的重新对齐可能大幅降低性能。

    arXiv:2611.31582v1 Announce Type: cross  Abstract: Latent interfaces let agents in a multi-agent system communicate through continuous representations rather than decoded text. Prior work reports performance and efficiency gains, but neither separates latent communication from the agent harness nor examines its internal components, leaving its actual contribution unclear. We propose LatentLens, a framework that decouples the latent interface at two levels: externally from the agent harness, to measure its contribution to overall performance (RQ1), and internally into realignment compatibility, information accessibility, and propagation dynamics, to identify the benefits and limitations of each (RQ2). Across seven benchmarks with Qwen3 models, adding the latent interface to the same harness does not improve and can even substantially degrade accuracy. Internally, latent representations retain some domain information, but incompatible realignment can substantially degrade performance, an
    
[^236]: 微调语言模型的频率条件化隐私审计

    Frequency-Conditioned Privacy Auditing of Fine-Tuned Language Models

    [https://arxiv.org/abs/2611.31581](https://arxiv.org/abs/2611.31581)

    该论文通过将敏感信息（邮箱地址和电话号码）在微调数据中的重复次数从1倍系统地变化到50倍，发现不同语言模型在敏感信息变得可提取前所需的重复阈值差异巨大，且梯度裁剪只能延迟信息泄露而在高重复率下无法可靠地防止泄露。

    

    大语言模型越来越多地在敏感数据上进行微调，其中同一个人的信息可能在多条记录中多次出现。这可能导致模型记住敏感信息，但记忆并不一定意味着该信息可以被恢复。当模型开始在生成过程中重现这些信息时，才会产生更严重的隐私风险。在本文中，我们研究敏感信息在微调数据中必须出现多少次才能变得可恢复。我们在四个语言模型上，将电子邮件地址和电话号码的重复频率从1倍变化到50倍，并比较标准微调、梯度裁剪和差分隐私训练的效果。我们发现，在敏感信息变得可恢复之前所需的重复次数在不同模型之间存在显著差异。梯度裁剪可以延迟信息被提取的时间，但在高重复次数下并不能一致地防止信息泄露。

    arXiv:2611.31581v1 Announce Type: new  Abstract: Large language models are increasingly fine-tuned on sensitive data, where the same person's information may appear multiple times across records. This can cause models to memorize sensitive information, but memorization does not necessarily mean that the information can be recovered. A more serious privacy risk arises when the model begins to reproduce that information during generation. In this paper, we study how often sensitive information must appear in the fine-tuning data before it becomes recoverable. Across four language models, we vary the frequency of email addresses and phone numbers from $1\times$ to $50\times$ and compare standard fine-tuning, gradient clipping, and differentially private training. We find that the number of repetitions required before sensitive information becomes recoverable varies substantially across models. Gradient clipping can delay extraction, but it does not consistently prevent leakage at high rep
    
[^237]: 扩展Zador模型中可证明近似最优的乘积量化

    Provably Near-Optimal PQ in the Extended Zador Model

    [https://arxiv.org/abs/2611.31579](https://arxiv.org/abs/2611.31579)

    本文提出EZPQ算法，通过证明SVD是最优预处理旋转、构建低比特率下准确的扩展Zador代价估计器、以及基于一维装箱的全局比特分配分割方法，实现了具有可证明近似保证且实际性能接近最优的乘积量化。

    

    本文研究了针对高斯数据的乘积量化（PQ）优化问题，提出了三项关键贡献： 证明了SVD是PQ的最优预处理旋转； 提出了一种新的扩展Zador（EZ）估计器，用于估计$k$-means量化代价，在低比特率和低维度下依然准确； 提出了一种基于全局比特分配的一维装箱的新型分割算法。最终得到一个具有强近似保证的简单算法（EZPQ）。在高斯EZ代价模型下，它达到了与OPQ相当的经验精度，同时能够提供针对每个实例的失真下界证书。在典型的真实嵌入谱上，每坐标2比特的EZPQ在该模型中通常可证明处于最优值1%以内。此外，相比EDEN、E-RaBitQ和TurboQuant等所有基于随机旋转的算法，它始终能够取得更低的重构误差。

    arXiv:2611.31579v1 Announce Type: new  Abstract: This paper investigates Product Quantization (PQ) Optimization with Gaussian data. It offers three key contributions (i) a proof that SVD is the optimal preconditioning rotation for PQ; (ii) a new Extended Zador (EZ) estimator of the $k$-means quantization cost that is accurate in low bit rates and low dimensions; and (iii) a novel segmentation algorithm based on 1-d bin packing of global bit allocations. The result is a simple algorithm (EZPQ) with strong approximation guarantees. It matches OPQ's empirical accuracy while providing a per-instance distortion lower bound certificate under the Gaussian EZ cost model. On typical real embedding spectra, EZPQ with 2 bits per coordinate is often certifiably within 1% of the optimum in this model. It also consistently achieves a lower reconstruction error than all random-rotation-based algorithms such as EDEN, E-RaBitQ, and TurboQuant.
    
[^238]: 保护能动性的人工智能悖论：在分层人机交互中协商学习者能动性

    The Paradox of Agency-Preserving AI: Negotiating Learner Agency in Layered Human-AI Interactions

    [https://arxiv.org/abs/2611.31573](https://arxiv.org/abs/2611.31573)

    该研究通过共同设计工作坊揭示了元层级AI系统的根本悖论——旨在保护学习者能动性的AI反而可能通过规定反思的时机与方式约束学习者能动性，并基于真实学习数据识别出四个相互关联的设计张力。

    

    生成式AI越来越多地被学习者在高认知要求的任务中使用，但其提供即时解决方案的能力引发了学习者能动性下降的担忧。提供反思性和元认知脚手架支持的元层级AI系统已成为支持学习者自我调节的一种有前景的方法。然而，这些系统引入了一个根本性的悖论：旨在保护学习者能动性的AI本身可能会通过规定学习者何时、如何以及为何进行反思，反而约束了这种能动性。为研究这一挑战，我们与18名从事编程和学术写作任务的学生开展了一项共同设计工作坊，他们使用扩展了ChatWise的ChatGPT——一个提供与自我调节学习相一致反馈的元层级AI系统。基于185条提示-反馈记录、174份书面反思以及参与者的设计产物（17个故事板和5个小组设计方案），我们识别出四个相互关联的设计张力（原文摘要在此处截断）。

    arXiv:2611.31573v1 Announce Type: cross  Abstract: Generative AI is increasingly used by learners in cognitively demanding tasks, but its capacity to provide immediate solutions raises concerns about decreased learner agency. Meta-level AI systems that provide reflective and metacognitive scaffolding have emerged as a promising approach to supporting learners' self-regulation. However, these systems introduce a fundamental paradox: AI intended to preserve learner agency can itself constrain that agency by regulating when, how, and why learners should reflect. To investigate this challenge, we conducted a co-design workshop with 18 students engaged in programming and academic writing tasks using ChatGPT extended with ChatWise, a meta-level AI system providing self-regulated learning-aligned feedback. Drawing on 185 prompt-feedback entries, 174 written reflections, and participants' design artifacts (17 storyboards and 5 group design proposals), we identified four interconnected design t
    
[^239]: 安全自动形式化：基于蕴含图的充分性与必要性

    Safe Autoformalization: Sufficiency and Necessity with Entailment Graphs

    [https://arxiv.org/abs/2611.31571](https://arxiv.org/abs/2611.31571)

    该论文提出了一种安全自动形式化方法：用LLM生成候选公式并构建由求解器判定的蕴含图，通过搜索充分前沿与必要前沿认证单侧逻辑充分性，使得对意图充分的公式可安全执行、对意图必要的公式可安全依赖。

    

    自动形式化是将自然语言陈述翻译为能够捕捉其意图的公式，这一过程依赖于陈述的上下文，即用于消除歧义的背景知识。大语言模型（LLM）使自动形式化变得实用，但其概率性输出不具备任何形式化保证。我们提出了实现安全自动形式化的技术，用于认证单侧逻辑充分性：对意图而言充分的公式可安全地强制执行，而对意图而言必要的公式可安全地加以依赖。我们利用大语言模型生成候选公式，并按逻辑强度将其组织在蕴含图（Entailment Graph, EG）中，该排序由求解器判定。每条边携带一个残差和一个余残差，即蕴含序的两个伴随算子，二者均仅由候选池决定。在给定上下文中，通过两次搜索分别返回充分前沿（蕴含该陈述的最弱候选公式）和必要前沿（被该陈述蕴含的最强候选公式）；每一步仅需一次局部检查。

    arXiv:2611.31571v1 Announce Type: new  Abstract: Autoformalization translates a natural-language statement into a formula that captures its intent, which depends on the statement's context: the background that disambiguates it. LLMs make autoformalization practical, but their probabilistic output carries no formal guarantees. We propose techniques for safe autoformalization that certify one-sided logical adequacy: a formula sufficient for the intent is safe to enforce, and a formula necessary for it is safe to rely on. We use LLMs to generate candidate formulas and order them by logical strength in an Entailment Graph (EG), an ordering a solver decides. Each edge carries a residual and a co-residual, the two adjoints of the entailment order, all fixed by the candidate pool alone. In a context, two searches return the sufficient frontier, the weakest candidates that entail the statement, and the necessary frontier, the strongest candidates the statement entails; each step is one local c
    
[^240]: 基于现成模型的自主智能体用于游戏测试

    Off-the-Shelf Autonomous Agents for Game Playtesting

    [https://arxiv.org/abs/2611.31570](https://arxiv.org/abs/2611.31570)

    提出Buddy——一个完全基于现成语言模型、无需训练或攻略的自主智能体，能在未见过的3D角色扮演游戏中通过探索解决长时程目标，并支持自动记录缺陷工单和将成功轨迹转化为分层任务网络用于回归测试。

    

    在游戏测试中，通常需要人工操作员来测试大部分游戏玩法元素。在本工作中，我们提出了Buddy，一个自主游戏求解智能体，它能够使用现成的语言模型系统在一个未见过的3D角色扮演游戏中解决长时程目标。Buddy会执行动作、接收这些动作的反馈，并在多个回合中持续追求目标。整个过程无需任何书面攻略或任何形式的模型训练，我们证明这一范式在全新环境中同样成立——智能体必须通过探索来发现解决方案，而非从预训练中回忆答案。我们展示了三种下游能力：在没有攻略的情况下完成游戏、为遇到的困难记录缺陷工单，以及将成功的智能体轨迹转换为脚本化的分层任务网络以用于回归测试。与现有的游戏测试方案不同，我们的框架是完全模块化的：语言模型和运行框架可以灵活组合。

    arXiv:2611.31570v1 Announce Type: new  Abstract: In playtesting games, a human operator is typically required to test most gameplay elements. In this work we present Buddy, an autonomous game solving agent which can solve long time horizon objectives in an unseen 3D role-playing game using off-the-shelf language model systems. Buddy will take actions, receive feedback on those actions, and pursue a goal across many turns. No written walkthrough or model training of any kind is required, and we show that this paradigm holds in a novel environment where the agent must discover solutions via exploration rather than recalling them from pretraining. We demonstrate three downstream capabilities: game solving with no provided walkthrough, logging bug tickets for encountered difficulties, and conversion of successful agent traces into scripted hierarchical task networks for regression testing. Unlike existing playtesting solutions, our framework is fully modular: language models and harnesses 
    
[^241]: COGENT：用于智能体用户模拟的模块化认知架构

    COGENT: A Modular Cognitive Architecture for Agentic User Simulation

    [https://arxiv.org/abs/2611.31559](https://arxiv.org/abs/2611.31559)

    COGENT是一个借鉴Norman动作循环和ACT-R等认知理论的模块化认知架构，通过将用户模拟分解为具有显式表示的感知、记忆、用户状态、意图和行动选择等认知模块，实现了对桌面界面交互的可解释、可干预的用户模拟。

    

    基于LLM的用户模拟可以支持界面设计、无障碍研究和交互式智能体评估，但标准的计算机使用智能体是为任务完成而优化的，而非重现人们如何感知、推理和行动。我们提出了COGENT（COGnitive agENT），一个用于模拟用户与桌面界面交互的模块化认知架构。COGENT借鉴了成熟的认知理论，如Norman的动作循环和ACT-R，通过将模拟过程分解为具有感知、记忆、用户状态、意图和行动选择显式表示的认知模块，实现可解释、可干预和模块化的用户模拟。我们首先展示了六个可干预性演示，每个演示追踪了对记忆、用户属性或处理模块的单次编辑如何通过认知管线传播并产生行为变化。我们评估了COGENT的行为与归因……

    arXiv:2611.31559v1 Announce Type: new  Abstract: LLM-based user simulation can support interface design, accessibility research, and interactive-agent evaluation, but standard computer-use agents are optimized for task completion rather than reproducing how people perceive, reason, and act. We introduce COGENT (COGnitive agENT), a modular cognitive architecture for simulating user interaction with desktop interfaces. COGENT draws on established cognitive theories, such as Norman's action cycle and ACT-R, to enable explainable, intervenable, and modular user simulation by decomposing the simulation process into cognitive modules with explicit representations of perception, memory, user state, intentions, and action selection. We begin by presenting six intervenability demonstrations, each tracing how a single edit to memory, a user attribute, or a processing module propagates through the cognitive pipeline to produce a behavioral change. We evaluate COGENT's behavioral and attributional
    
[^242]: 基于渐进后验采样的去噪扩散

    Denoising Diffusion with Progressive Posterior Sampling

    [https://arxiv.org/abs/2611.31558](https://arxiv.org/abs/2611.31558)

    该论文提出带流形投影的渐进后验采样（PPS）方法，通过闭式高斯后验近似自动估计噪声方差、避免SVD，并将样本投影至局部含噪图像流形，从而更有效地利用扩散先验的局部几何结构求解含噪线性逆问题。

    

    扩散模型为求解含噪线性逆问题提供了有效的生成先验。然而，现有的后验采样方法可能需要对测量算子进行奇异值分解（SVD），或者需要预先知晓测量噪声的方差，而基于局部梯度的引导仅能提供对后验几何结构的有限视角。我们提出了一种带流形投影的渐进后验采样（PPS）方法，该方法能够自动估计测量噪声方差并避免SVD。在反向扩散过程中，PPS以闭式形式构建后验分布的高斯近似，并将从局部先验中抽取的样本引导至近似后验均值。随后，我们将引导后的样本投影到一个近似局部含噪图像流形的球面水平集上，从而促进样本与扩散先验局部几何结构的一致性。在FFHQ和ImageNet数据集上跨越四种图像修复任务的实验……（原文摘要在此处截断）

    arXiv:2611.31558v1 Announce Type: cross  Abstract: Diffusion models provide effective generative priors for solving noisy linear inverse problems. However, existing posterior sampling methods may require singular value decomposition (SVD) of the measurement operator or prior knowledge of the measurement noise variance, while guidance based on local gradients provides only a limited view of posterior geometry. We propose progressive posterior sampling (PPS) with manifold projection, which automatically estimates the measurement noise variance and avoids SVD. During reverse diffusion, PPS constructs a Gaussian approximation to the posterior distribution in closed form and guides a sample drawn from the local prior toward the approximate posterior mean. We then project the guided sample onto a spherical level set that approximates the local noisy image manifold, promoting consistency with the local geometry of the diffusion prior. Experiments on FFHQ and ImageNet across four restoration t
    
[^243]: 智能体恢复出错：测量重复调用之外的危害

    Agent Recovery Gone Wrong: Measuring Harm Beyond Duplicate Calls

    [https://arxiv.org/abs/2611.31554](https://arxiv.org/abs/2611.31554)

    该论文首次在自然发生的失败场景下量化了代码执行型LLM智能体恢复操作对外部状态的危害，发现约24.6%的结果可确定的部分执行会导致在运行结束时仍持续存在的有害恢复，其中仅三分之二表现为重复调用，其余则表现为删除用户数据或错误的首次写入，说明仅检测重复调用不足以捕捉全部危害。

    

    执行代码的大语言模型智能体会提交包含多个API写操作的代码块。如果某个代码块在其部分写操作已生效之后失败，智能体会重写该代码块并继续执行。当这种失败在智能体自己编写的代码中自然发生时，此类恢复在多大频率上会危害外部状态，目前尚不清楚。本文在AppWorld智能体的记录轨迹上对此进行了测量。在对十个模型的200个候选回合进行的分层随机抽样中，有145个被数据库状态确认为部分执行。在样本中，143个结果可确定的确认部分执行里有30个随后发生了有害恢复，且其违规状态在运行结束时仍然存在，由此得到分层加权估计值为24.6%。三分之二的有害恢复表现为重复了某个效果，其余的则删除用户数据或进行错误的首次写入，例如订购了错误的产品。搜索重复调用的规则发现……（原文在此处截断）

    arXiv:2611.31554v1 Announce Type: new  Abstract: Code-executing LLM agents submit blocks of code that issue several API writes. If a block fails after some of its writes have taken effect, the agent rewrites the block and continues. It is unknown how often such a recovery harms the external state when the failure occurs naturally in the code written by the agent. In this paper, we measure it on recorded trajectories of AppWorld agents. In a stratified random sample of 200 candidate episodes across ten models, 145 are partial executions confirmed by the database state. In the sample, 30 of 143 confirmed partial executions with a determinable outcome are followed by a harmful recovery, and its violation remains at the end of the run, which gives a stratification-weighted estimate of 24.6%. Two thirds of the harmful recoveries duplicate an effect, and the rest delete user data or make a wrong first-time write, e.g., an order of a wrong product. A rule that searches for repeated calls find
    
[^244]: 可复现的社会结构从大语言模型智能体的自由互动中涌现

    Reproducible Social Structure Emerges From Free-Form Interaction Among LLM Agents

    [https://arxiv.org/abs/2611.31543](https://arxiv.org/abs/2611.31543)

    该论文提出了一个可重复、无目标的多智能体大语言模型模拟框架，并通过十次重复运行证明社会结构（如交流最终收缩为持续的二人关系）能够稳定地重复涌现，从而区分了模拟系统中的稳健发现与随机事件。

    

    多智能体大语言模型模拟越来越多地被用作社会系统的模型，但大多数已发表的发现都是轶事性的，即来自单次运行。因此，目前尚不清楚哪些观察到的现象是模拟系统本身的稳健发现，哪些只是某次特定运行中的随机事件。在此，我们引入了一个可重复、无目标、主动性的多智能体框架，其中五十个由人设条件约束、具有持久记忆的智能体可以在三个社交平台上自主选择是否交流、与谁交流以及如何交流。该框架支持跨时间快照和重复运行的非扰动性测量。在相同初始条件下将同一世界运行十次，总计700小时，我们发现大多数测量的网络级和智能体级参数在各次重复中保持稳定。在整个运行过程中，最初的广泛交流在所有十个世界中都收缩为稀疏而持续的二人互动关系。重要的是，我们发现许多id（原文摘要在此处被截断）

    arXiv:2611.31543v1 Announce Type: cross  Abstract: Multi-agent LLM simulations are increasingly used as models of social systems, but most published findings are anecdotal, i.e., drawn from a single run. It is therefore unclear which observed phenomena are robust findings of the simulated system itself, and which were random events in one particular run. Here, we introduce a repeatable, goal-free, proactive multi-agent framework in which fifty persona-conditioned agents with persistent memory choose whether, with whom, and how to communicate across three social platforms. The framework supports non-perturbative measurements across temporal snapshots and repetitions. Running the same world ten times from identical initial conditions for a total of 700 hours, we find that most measured network- and agent-level parameters are stable across repetitions. Throughout a run, initially broad communication contracts into sparse sustained dyads in all ten worlds. Importantly, we find that many id
    
[^245]: 智能体防御失效之处：策略组合与测量局限的案例研究

    Where Agent Defenses Fail: A Case Study of Policy Compositions and Measurement Limits

    [https://arxiv.org/abs/2611.31541](https://arxiv.org/abs/2611.31541)

    本文通过大规模案例研究揭示了智能体安全防御策略组合的固有矛盾：限制性沙箱虽能阻止攻击但严重损害良性效用，中介机制虽恢复功能却可能放行机密泄露，且现有测量方法存在编码泄露、观察者遗漏等盲区，为智能体安全评估的局限性提供了实证依据。

    

    编码智能体可能需要上传报告，而其沙箱会阻止网络访问。宿主中介可以恢复传输功能，但注入的指令可能会在已批准的请求中添加机密信息。我们使用文件和HTTP回执来研究这种交互，以独立于控制日志的方式测量效果。两个前瞻性冻结输入队列分别在先前探索过的自创任务上进行了110次和300次运行。限制性沙箱未观察到攻击危害，但将良性效用降低到6/8和12/24。中介机制恢复了上传功能却允许了信息泄露；完整行匹配保留了全部良性效用，但在扩展队列中遗漏了两次仅泄露值的攻击。一项单独的72次运行边界研究识别出编码泄露、观察者遗漏以及掩盖良性拒绝的格式检查问题。一项80次运行的SafeAgents/OpenShell后续研究表明，遏制机制在有效观察范围内阻止了测试中发现的防护遗漏，而中介机制恢复了良性效用。

    arXiv:2611.31541v1 Announce Type: cross  Abstract: A coding agent may need to upload a report while its sandbox blocks network access. A host mediator can restore delivery, but an injected instruction may add a secret to the approved request. We study this interaction using files and HTTP receipts to measure effects independently of control logs. Two prospective frozen-input cohorts comprise 110 and 300 runs on previously explored authored tasks. Restrictive sandboxing produces no observed attack harm but reduces benign utility to 6/8 and 12/24. Mediation restores uploads and admits disclosure; complete-line matching retains full benign utility but misses two value-only disclosures in the expanded cohort. A separate 72-run boundary study identifies encoded disclosure, observer misses, and formatting checks that obscure benign refusals. An 80-run SafeAgents/OpenShell follow-up shows that containment stops the tested guard misses among valid observations, while mediation restores benign 
    
[^246]: 无需真值标签的智能体评估：零假设比例视角

    Agent Evaluation Without Ground Truth: A Null Proportion Perspective

    [https://arxiv.org/abs/2611.31539](https://arxiv.org/abs/2611.31539)

    该论文提出了一种无需真值标签的智能体评估框架，将智能体质量定义为其轨迹中正确决策的加权和，用 e 值序列作为统计证据替代真值标签，并能处理任意相关的决策、数据依赖的轨迹终止以及动态生成的假设。

    

    智能体所采取行动的正确性通常依据已知的、经外部验证的真值标签来评估，例如证明是否正确，或代码能否编译并通过单元测试。然而，随着编程和科学智能体被赋予更多端到端的自主权——提出干预方案、开展实验、决定接下来部署和研究什么——智能体的行动是否真正改善了结果，必须从充满噪声且常常没有定论的实验中推断出来。为了在这一设定下评估智能体，我们将智能体质量定义为其轨迹上一系列正确决策的加权和，并以 e 值序列作为关于正确性的统计证据来替代真值标签。我们的框架允许任意相互依赖的决策、依赖数据的轨迹终止、动态生成的假设，以及在轨迹结束后才定义的质量度量。我们构造了……（摘要在此处截断）

    arXiv:2611.31539v1 Announce Type: new  Abstract: The correctness of actions taken by agents is typically evaluated against known, externally verified ground truth labels, such as whether a proof is correct or whether code compiles and passes unit tests. However, as coding and science agents are entrusted with more end-to-end autonomy -- proposing interventions, running experiments, and deciding what to deploy and investigate next -- whether an agent's actions genuinely improve an outcome must be inferred from noisy, often inconclusive experiments. To evaluate agents in this setting, we define agent quality as a weighted sum of correct decisions along its trajectory, using sequences of $e$-values as statistical evidence regarding correctness in place of ground-truth labels. Our framework accommodates arbitrarily dependent decisions, data-dependent trajectory termination, dynamically generated hypotheses, and quality measures defined after the trajectory concludes. We construct upper and
    
[^247]: 病理基础模型的调和与少样本校准提高数字生物标志物在不同切片扫描仪间的可重复性

    Tempering and few-shot calibration of pathology foundation models improve digital biomarker reproducibility across slide scanners

    [https://arxiv.org/abs/2611.31536](https://arxiv.org/abs/2611.31536)

    该研究提出一种仅需公开数据的“调和”微调方法，可显著降低病理基础模型对扫描仪差异的敏感性，大幅提升HPV预测、MSI识别、前列腺癌分级等数字生物标志物的跨扫描仪可重复性。

    

    病理基础模型支撑着越来越多基于组织学的癌症生物标志物，但其对切片扫描仪特异性图像特征的敏感性会降低跨中心的可靠性与可重复性。我们开发了一种基础模型微调方法，提高了头颈癌人乳头瘤病毒（HPV）预测、结直肠癌微卫星不稳定（MSI）识别、前列腺癌分级以及乳腺癌表型分型的一致性。这种“调和”方法使HPV的跨扫描仪预测差异降低了67.5%（95% CI 59.0-73.9%），MSI降低了69.3%（95% CI 51.9-76.5%），乳腺癌特征降低了63.4%-81.2%。跨扫描仪的前列腺癌分级一致性从42.5%提升至92.7%。在训练中存在完全的扫描仪-HPV混杂时，使用原始模型特征的AUROC从0.80降至0.68，而使用调和方法仅从0.86降至0.84。与事后特征校准不同，调和方法仅使用公开数据即可达到相当或更好的效果。

    arXiv:2611.31536v1 Announce Type: cross  Abstract: Pathology foundation models underpin a growing number of histology-based cancer biomarkers, but sensitivity to slide scanner-specific image signatures degrades cross-center reliability and reproducibility. We developed a foundation model fine-tuning method that improved consistency in head-and-neck cancer human papillomavirus (HPV) prediction, colorectal cancer microsatellite instability (MSI) identification, prostate cancer grading and breast cancer phenotyping. This tempering method decreased cross-scanner prediction differences by 67.5% for HPV (95% CI 59.0-73.9%), 69.3% for MSI (95% CI 51.9-76.5%) and 63.4%-81.2% for breast signatures. Cross-scanner prostate grade agreement rose from 42.5% to 92.7%. Under full scanner-HPV confounding during training, AUROC fell from 0.80 to 0.68 with original model features but only from 0.86 to 0.84 with tempering. Unlike post hoc feature calibration, tempering used only public data yet matched or
    
[^248]: 一步四时钟：GUI智能体为何失步脱节

    One Step, Four Clocks: How GUI Agents Fall Out of Step

    [https://arxiv.org/abs/2611.31522](https://arxiv.org/abs/2611.31522)

    该论文提出EODA四事件时间模型与ChronotopeGym基准，揭示GUI智能体在实时交互中因界面变化导致动作失效的根本原因，并设计驻留加速器的Chronotope控制器，通过选择性保留、更新或重决策待执行动作，将端到端延迟降低85.5%并实现100%实时成功率。

    

    GUI智能体能够选择正确的动作，却会在界面于决策期间发生变化时失败。仅凭延迟无法解释这些失败，因为动作可能在不同的交互阶段变得无效。我们建模了四个事件：环境变化、观测可用、决策提交和动作接收。我们引入ChronotopeGym，在暂停、重放和实时执行模式下研究这些失败。在五项任务中，即使动作满足截止时间、且响应保持有效时能够成功，智能体在实时模式下仍可能失败。在移动气球任务上，更新一个待执行的点击操作可将成功率从0%提升到100%，而响应时间保持不变。我们的驻留于加速器上的GUI控制器Chronotope维护紧凑的交互历史，并选择性地保留、更新或重新决策待执行动作。与Agent S3相比，Chronotope将27B模型的端到端中位延迟降低了85.5%，将其性能包络扩展至100%的实时成功率。

    arXiv:2611.31522v1 Announce Type: new  Abstract: GUI agents can choose correct actions yet fail as interfaces change during deliberation. Latency alone cannot explain these failures because actions may become invalid at different interaction stages. We model four events: environment change, observation availability, decision commitment, and action receipt (EODA). We introduce ChronotopeGym to study these failures under Paused, Replay, and Live execution. Across five tasks, agents can fail in Live despite succeeding when responses remain valid, even when actions meet deadlines. On Moving Balloon, updating a pending click raises success from 0% to 100% without changing response time. Our accelerator-resident GUI controller, Chronotope, maintains compact interaction history and selectively keeps, updates, or redecides pending actions. Compared with Agent S3, Chronotope reduces median end-to-end latency by 85.5% for a 27B model, expanding its performance envelope to 100% Live success at an
    
[^249]: PatientFlowSynth：使用匹配患者病历评估FHIR智能体

    PatientFlowSynth: Evaluating FHIR Agents with Matched Patient Charts

    [https://arxiv.org/abs/2611.31521](https://arxiv.org/abs/2611.31521)

    该论文提出 PatientFlowSynth 基准，利用 225 个合成患者和 18 个模板生成 18,000 个控制变量匹配的病历变体任务，在四种证据状态（含缺失与冲突证据）下系统评估 FHIR 医疗智能体的分类、引用与检索能力，并验证了明确的就诊状态指导能显著提升严格成功率。

    

    搜索患者病历的智能体必须根据检索到的记录对声明进行分类，其中包括证据缺失或相互冲突的情况。在快速医疗互操作性资源（FHIR）中，状态字段可能会改变对这些记录的解读方式。PatientFlowSynth 提供了来自 225 个合成患者和 18 个模板的 18,000 个任务，这些模板在代表四种证据状态的病历变体之间保持患者、问题和检索程序固定不变。通过重新执行检索程序来验证每个变体的参考答案、证据集和搜索要求。严格成功要求完成的回复具有正确的标签，且引用与在任务过程中检索到的某个被认可的证据集完全匹配；在需要时，工具响应还必须显示在指定范围内完成了完整搜索。在使用 GPT-5.6 Luna 和 225 个合成患者（3,600 对匹配对）的研究中，明确的就诊状态指导使严格成功率从 40.7%（摘要在此处被截断）

    arXiv:2611.31521v1 Announce Type: new  Abstract: Agents that search patient charts must classify claims from retrieved records, including cases with missing or conflicting evidence. In Fast Healthcare Interoperability Resources (FHIR), status fields can change how those records should be interpreted. PatientFlowSynth provides 18,000 tasks from 225 synthetic patients and 18 templates that hold the patient, question, and retrieval program fixed across chart variants representing four evidence states. Re-executing the program verifies each variant's reference answer, evidence sets, and search requirements. Strict success requires a completed response with correct labels and citations exactly matching an accepted evidence set retrieved during the episode. Where required, a tool response must also show a complete search at the specified scope. In a study using GPT-5.6 Luna and 225 synthetic patients (3,600 matched pairs), explicit Encounter-status guidance increased strict success from 40.7
    
[^250]: XRL4T1D：用于1型糖尿病管理的强化学习事后可解释性框架

    XRL4T1D: A Post-hoc Explainability Framework for Reinforcement Learning in Type 1 Diabetes Management

    [https://arxiv.org/abs/2611.31498](https://arxiv.org/abs/2611.31498)

    本文提出XRL4T1D框架，无需重新训练即可利用SHAP等方法对1型糖尿病胰岛素给药的强化学习策略进行事后解释，揭示了CGM血糖值与胰岛素历史输入对决策的重要性。

    

    1型糖尿病（T1D）需要终身胰岛素治疗，强化学习（RL）在UVA/Padova模拟器中已展现出自动给药的前景。然而，RL控制器的决策仍然难以解释。我们提出了XRL4T1D，这是一个事后可解释性框架，可以在无需重新训练的情况下评估预训练的胰岛素给药策略。每个策略使用60分钟的连续血糖监测（CGM）和已给药胰岛素历史记录（每5分钟采样一次）来推荐胰岛素剂量。我们使用SHAP（SHapley Additive exPlanations）来比较CGM输入和胰岛素历史输入的重要性，并受“体内胰岛素量”概念的启发，将胰岛素历史的归因总结为三个20分钟的时间窗口。我们通过有针对性的输入替换来评估这些归因是否与策略输出的敏感性相对应，并将时间排序结果与TimeSHAP进行比较。在全部80个患者-策略-掩码大小的案例中，替换归因最高的输入会改变（策略的输出）……

    arXiv:2611.31498v1 Announce Type: new  Abstract: Type 1 diabetes (T1D) requires lifelong insulin therapy, and reinforcement learning (RL) has shown promise for automated dosing in the UVA/Padova simulator. However, the decisions of RL controllers remain difficult to interpret. We present XRL4T1D, a post-hoc explainability framework for evaluating pretrained insulin-dosing policies without retraining. Each policy uses 60 minutes of continuous glucose monitor (CGM) and administered-insulin history, sampled every 5 minutes, to recommend insulin. We use SHapley Additive exPlanations (SHAP) to compare the importance of CGM and insulin-history inputs and summarize insulin-history attributions over three 20-minute windows motivated by insulin-on-board. We assess whether these attributions correspond to policy-output sensitivity through targeted input replacement and compare temporal rankings with TimeSHAP. In all 80 patient-policy-mask-size cases, replacing the highest-attribution inputs chan
    
[^251]: 层间细读：表征质量究竟藏于何处，几何结构又揭示了什么

    Reading Between the Layers: Where Representation Quality Lives, and What Geometry Reveals

    [https://arxiv.org/abs/2611.31496](https://arxiv.org/abs/2611.31496)

    该论文通过对25个模型、13个数据集的大规模实证研究，系统揭示了预训练语言模型的最佳表征层随预训练策略与架构而系统性变化（掩码编码器峰值在中后层、序列到序列模型随任务类型偏移、对比编码器取决于读出方式、仅解码器模型呈阶段结构），为无标签层选择这一开放问题提供了实证指引。

    

    预训练文本基础模型中的表征质量并非集中于最后一层，然而由于无标签情况下的层选择一直是悬而未决的难题，实践者通常习惯在最后一层进行探测。我们对层质量在何处达到峰值开展了一项大规模实证研究，涵盖25个模型，横跨四种架构族和四种预训练策略，并在13个数据集上进行评估，任务包括分类、成对分类、语义文本相似度和多项选择题问答。峰值探测层系统性地依赖于预训练策略和架构：掩码编码器在中后层达到峰值；序列到序列模型的峰值随任务类型而变化；对比式句子编码器在CLS和末词元读出方式下信号集中于最后一层，但在平均池化下峰值出现在较早层；仅解码器模型则展现出与深度相关的独特阶段结构。我们将这些发现提炼为稳定几何……

    arXiv:2611.31496v1 Announce Type: cross  Abstract: Representation quality in pretrained text foundation models is not concentrated at the final layer, yet practitioners routinely probe it there because layer selection without labels has remained an open problem. We present a large-scale empirical study of where layer quality lives across 25 models spanning four architectural families and four pretraining strategies, evaluated on 13 datasets across classification, pair classification, semantic textual similarity, and multiple-choice question answering. Peak probing layers depend systematically on pretraining strategy and architecture: masked encoders peak in late-middle layers, sequence-to-sequence models shift their peak with task type, contrastive sentence encoders concentrate signal at the final layer under CLS and last-token readouts but peak early under mean pooling, and decoder-only models exhibit distinct phase structure tied to depth. We distill these findings into the Stable Ge
    
[^252]: 在科学网关中使用开放权重模型评估前沿模型的智能体技能

    Evaluating Frontier-Model Agent Skills with Open-Weight Models in a Science Gateway

    [https://arxiv.org/abs/2611.31480](https://arxiv.org/abs/2611.31480)

    本文提出一种方法，通过自动评估测试框架驱动真实科学网关、对生成的 Pegasus 工作流评分并把反复出现的失败反馈给前沿模型，从而将前沿模型编写的智能体技能适配到科研基础设施上免费提供的开放权重模型，使没有前沿模型额度的科学家也能用聊天界面构建可靠的科学工作流。

    

    跨越多个作业和计算站点的科学分析依赖工作流管理系统才能可靠运行，但编写生产级工作流需要此类系统的专业知识，而许多领域科学家并不具备这种专业知识。编码智能体在由技能和针对重复任务的打包流程引导时，可以提供这种专业知识。我们使用前沿模型为 Pegasus 工作流管理系统编写了技能和专门智能体，并通过 PegasusAI Studio 科学网关以聊天界面向科学家提供这些能力。没有前沿模型使用额度的新用户需要一个开放权重模型，该模型可在科研网络基础设施上免费提供，但为前沿模型编写的技能可能无法直接用于开放权重模型。本文提出了一种适配此类技能的方法：一个评估测试框架像真实用户一样驱动实际网关，对智能体构建的工作流进行评分，并将反复出现的失败反馈给前沿模型（原文摘要在此处截断）。

    arXiv:2611.31480v1 Announce Type: new  Abstract: Scientific analyses that span many jobs and computing sites depend on workflow management systems to run reliably, but writing a production workflow requires expertise in such a system, which many domain scientists do not have. Coding agents can supply this expertise when they are guided by skills and packaged procedures for recurring tasks. We wrote skills and specialist agents for the Pegasus workflow management system with a frontier model, and the PegasusAI Studio science gateway offers them to scientists through a chat interface. New users without frontier-model credits need an open-weight model, served at no cost on research cyberinfrastructure, but skills written for a frontier model may not work with an open-weight solution. This paper presents a method for adapting such skills. An evaluation harness drives the real gateway as a user would, scores the workflows the agent builds, and feeds recurring failures back to the frontier m
    
[^253]: SpatialCraft：从经验与技能中演化程序性智能的空间推理智能体

    SpatialCraft: Evolving Procedural Intelligence from Experience and Skills for Spatial Reasoning Agents

    [https://arxiv.org/abs/2611.31475](https://arxiv.org/abs/2611.31475)

    SpatialCraft提出一个无需参数更新的自演化框架，让冻结的工具调用智能体通过积累“经验”（条件-动作策略）和“技能”（带激活与终止条件的多步骤程序）两类外部知识，从自身交互历史中不断改进工具编排能力。

    

    空间推理智能体越来越依赖外部工具（如3D重建、分割和几何计算）来回答多模态大语言模型（MLLM）仅凭像素无法解决的问题。然而，工具的使用将瓶颈转移到了编排（orchestration）环节：智能体必须决定调用哪些工具、以何种顺序调用，以及如何解释工具的输出，而无状态的智能体会在不同任务中重复相同的错误。我们探讨一个参数冻结的工具使用智能体能否在不更新参数的情况下，从自身的交互历史中改进其编排能力。我们提出SpatialCraft，一个自演化框架，它维护两种形式的外部知识：经验——通过对比同一任务的不同执行轨迹提炼出的“条件-动作”策略；以及技能——具有明确激活条件和终止条件的多步骤程序。三种设计将这两个知识流相互耦合，经验可以锚定到某个技能之上，仅在……

    arXiv:2611.31475v1 Announce Type: new  Abstract: Spatial reasoning agents increasingly rely on external tools, such as 3D reconstruction, segmentation, and geometric computation, to answer questions that a multimodal large language model (MLLM) cannot resolve from pixels alone. Tool access, however, shifts the bottleneck to orchestration: the agent must decide which tools to call, in what order, and how to interpret their outputs, yet a stateless agent repeats the same mistakes across tasks. We ask whether a frozen tool-using agent can improve its orchestration from its own interaction history, without parameter updates. We present SpatialCraft, a self-evolving framework that maintains two forms of external knowledge: Experiences, condition-action tactics distilled by contrasting rollouts of the same task, and Skills, multi-step procedures with explicit activation and termination conditions. Three designs couple the two streams. Experiences can be anchored to a Skill, appearing only wh
    
[^254]: AdProducer：迈向智能体驱动的广告视频生成

    AdProducer: Towards Agentic Advertising Video Generation

    [https://arxiv.org/abs/2611.31466](https://arxiv.org/abs/2611.31466)

    AdProducer 是首个端到端的智能体广告视频生成框架，它通过自主开展产品调研并将结果转化为以卖点驱动的分镜脚本，从而将简洁的创意简报直接转化为成品广告。

    

    视频生成技术的最新进展使得自动化广告制作日益可行。然而，一条有效的广告必须忠实地呈现产品，并通过引人入胜的视觉叙事传达其卖点，这不仅需要视觉合成能力，更需要对产品的深入理解，以及关于如何展示产品价值的策略性决策。至关重要的是，产品调研与创意设计紧密交织，因为所收集到的产品信息决定了哪些特性能够被忠实地展示，以及这些特性应当如何呈现。在本文中，我们提出了 AdProducer，这是首个用于广告视频创作的端到端智能体框架，能够将简洁的创意简报转化为成品广告。我们的多模态智能体通过网络搜索与图像搜索自主开展产品调研，并将调研结果转化为以卖点为导向的分镜脚本，用于指导……

    arXiv:2611.31466v1 Announce Type: cross  Abstract: Recent advances in video generation have made automated advertising production increasingly feasible. However, an effective advertisement must faithfully represent the product and communicate its selling points through compelling visual storytelling, which requires more than visual synthesis: it calls for deeper product understanding and strategic decisions about how to present product value. Crucially, product research and creative design are closely intertwined, as the information gathered about a product determines which features can be faithfully demonstrated and how they should be presented. In this paper, we introduce AdProducer, the first end-to-end agentic framework for advertising video creation, transforming concise creative briefs into finished advertisements. Our multimodal agent autonomously researches products through web and image search and translates its findings into selling-point-driven storyboards that guide a froze
    
[^255]: 秒表之下：在内核层面检验编码代理的性能声明

    Below the Stopwatch: Checking the Performance Claims of Coding Agents in the Kernel

    [https://arxiv.org/abs/2611.31460](https://arxiv.org/abs/2611.31460)

    提出了Below流水线，通过提取编码代理性能PR所声称的优化机制并用硬件计数器与LTTng内核追踪验证其应有的内核级信号，发现大多数性能声明缺乏实证测量，且相当一部分PR并未带来真实的性能提升。

    

    编码代理提交了大量声称能让软件变快的拉取请求（PR）。这类声明大多还会给出理由，例如“缓存结果”或“只读取文件一次”。秒表可以显示代码变快了，却无法证明它是因为所声称的原因而变快。我们提出了Below——一个在内核层面检验这些理由的流水线。Below从AIDev数据集中的1,343个运行时性能PR中提取每个PR所声称的机制，并将其映射到如果该声明为真则必须发生变化的内核与硬件信号。一个从不接触这些声明的构建代理在每个PR应用前后重新构建代码。随后Below通过普通计时运行、硬件计数器和LTTng内核追踪来测量两个版本。结果显示大多数声明并未经过测量：57.3%的PR仅提供推理或没有任何证据，只有31.4%给出了效应大小。在82个被分析的PR中，32个运行更快，28个没有明显变化，22个运行更慢。所声称机制的预期信号……（摘要原文在此处截断）

    arXiv:2611.31460v1 Announce Type: cross  Abstract: Coding agents open many pull requests (PRs) that claim to make software faster. Most such claims also give a reason, such as "cache the result" or "read the file once". A stopwatch can show that code got faster, but not that it got faster for the stated reason. We present Below, a pipeline that checks these reasons at the kernel level. From 1,343 runtime-performance PRs in the AIDev dataset, Below extracts the mechanism each PR claims and maps it to the kernel and hardware signals that must change if the claim is true. A build agent that never sees the claim rebuilds the code before and after each PR. Below then measures both versions with plain timing runs, hardware counters and LTTng kernel traces. Most claims are not measured: 57.3% of the PRs offer only reasoning or no evidence, and only 31.4% state an effect size. Of 82 analyzed PRs, 32 ran faster, 28 showed no clear change and 22 ran slower. The expected signal of the claimed mec
    
[^256]: Taxi：以（真实）大规模分析智能体

    Taxi: Analyzing Agents at (Real) Scale

    [https://arxiv.org/abs/2611.31457](https://arxiv.org/abs/2611.31457)

    Taxi 是一个可高效扩展至数百万步骤的大规模智能体调试、聚类与监控系统，通过聚类分类体系、完整 SQL 查询和 Jev 式决策模型的混合流水线，为工程师和自动化智能体提供结构化分析能力。

    

    在这份技术报告中，我们提出了 Taxi——一个用于大规模调试、聚类和监控智能体的系统。Taxi 能够高效地扩展到数百万个步骤，这得益于多种可扩展分析方法的组合。Taxi 主要依靠聚类来生成步骤的分类体系，从而对数据进行结构化概览，并为工程师和自动化智能体提供完整的 SQL 访问权限，以便提出复杂的问题。聚类和查询最好在编码智能体引导的混合流水线中结合使用。Taxi 还包含 Jev 式决策模型的实验性集成：这类模型将状态和有界问题直接映射为类型化的、概率性的决策。在嵌入或字面查询不足以解决问题时，它们提供了低成本的语义路由和评估。此外，Taxi 提供了一个可扩展的生态系统，包括内置的评估器和每日报告。为适应增长需求，Taxi 被设计为模块化的，

    arXiv:2611.31457v1 Announce Type: new  Abstract: In this technical report, we present Taxi -- a system for debugging, clustering, and monitoring agents at scale. Taxi scales to millions of steps efficiently. This is made possible by a combination of scalable analysis methods. Taxi relies primarily on clustering to generate a taxonomy over steps for a structured overview of the data, and provides full SQL access for engineers and automated agents to ask complex questions. Clustering and querying are best used together in a hybrid pipeline guided by a coding agent. Taxi also includes an experimental integration of Jev-style decision models: models that map state and bounded questions directly to typed, probabilistic decisions. They add low-cost semantic routing and evaluation where embeddings or literal queries alone are insufficient. In addition, Taxi provides an extensible ecosystem, including built-in evaluators and daily reports. To accommodate growth, Taxi is designed to be modular,
    
[^257]: 分期泄露：LLM辅导系统中的跨会话答案泄露及基于账本的防御

    Leaks by Installments: Cross-Session Answer Disclosure in LLM Tutors and a Ledger-Based Defense

    [https://arxiv.org/abs/2611.31429](https://arxiv.org/abs/2611.31429)

    本文提出一种为每个学习者维护持久化泄露账本的防御机制，通过类型化解题图与门控匹配检测跨会话答案泄露，将模拟实验中通过改编数字练习变体产生的泄露率从100%降至10%。

    

    基于大语言模型（LLM）构建的辅导系统被告知不要泄露答案，但学习者并不需要一次性索要答案。现有的安全防护措施仅在当前对话范围内评判每次回复，因此学习者可以开启新会话、重新措辞表述问题，或对同一问题使用不同数字请求完整的解题过程。我们将这种现象称为跨会话泄露，并提出一种防御方法：为每个学习者维护一个持久的泄露账本。对于学习者提出的每个问题，系统会构建一个带类型的解题图。在回复发布之前，一个门控机制将回复自身的内容（而非学习者的措辞）按结果值和带类型标签的解题结构与账本进行匹配，然后计算学习者此时能够推导出哪些步骤。如果该步骤集合能够到达最终答案或超出预算，则拒绝该回复。在3000个问题的模拟实验中，该账本将通过改编数字的练习变体导致的泄露从100%降低到10%。

    arXiv:2611.31429v1 Announce Type: cross  Abstract: Tutors built on large language models (LLMs) are told not to give away answers, but a learner does not have to ask for the answer all at once. Existing safeguards judge each reply within the current conversation, so a learner can open a new session, reword the problem, or request a worked solution to the same problem with different numbers. We call this cross-session disclosure and propose a defense: a persistent, per-learner disclosure ledger. For every problem a learner brings, the system builds a typed solution graph. Before a reply is released, a gate matches the reply's own content, not the learner's wording, against the ledger by result values and by type-labeled solution structure, then computes which steps the learner could now derive. The reply is withheld if that set reaches the final answer or exceeds a budget. In a simulation of 3,000 problems, the ledger reduced leakage through renumbered practice variants from 100% to 10.
    
[^258]: 学什么 vs. 学怎么做：智能体训练中任务能力与界面能力的分解

    Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training

    [https://arxiv.org/abs/2611.31427](https://arxiv.org/abs/2611.31427)

    该论文将智能体能力分解为“任务能力”（解决任务所需的知识与推理）和“界面能力”（通过特定框架表达知识的能力），揭示了基于框架的训练效果取决于这两种能力的交互方式，并提出多框架蒸馏可将不同框架下的任务知识聚合到共享策略中。

    

    LLM 智能体通过“框架”运行，框架定义了系统提示、动作格式、反馈循环和资源约束。一个模型可能拥有解决任务所需的知识，却无法通过特定的框架表达这些知识。这引出了一个基本问题：当策略在给定框架下失败时，究竟是缺乏任务知识，还是缺乏操作界面的能力？我们区分了智能体能力的两个组成部分：任务能力，即解决任务所需的知识和推理能力；界面能力，即通过给定框架将该知识付诸行动的能力。在四个推理和智能体基准测试中，我们展示了基于框架条件的训练会因这两种能力的交互方式不同而产生截然不同的结果。多框架蒸馏将通过不同框架收集的轨迹中的任务知识聚合到一个共享策略中，但只有在训练保留或模仿……（摘要截断）时，这些知识才能得以实现。

    arXiv:2611.31427v1 Announce Type: new  Abstract: LLM agents operate through harnesses that define system prompts, action formats, feedback loops, and resource constraints. A model may possess the knowledge needed to solve a task yet fail to express that knowledge through a particular harness. This raises a basic question: when a policy fails under a given harness, is it missing task knowledge or the competence to operate the interface? We distinguish two components of agentic competence: task competence, the knowledge and reasoning needed to solve a task, and interface competence, the ability to act on that knowledge through a given harness. Across four reasoning and agentic benchmarks, we show that harness-conditioned training yields vastly different outcomes depending on how these interact. Multi-harness distillation aggregates task knowledge from trajectories collected through different harnesses into a shared policy, but this knowledge is realised only when training preserves or im
    
[^259]: ContextLure：利用良性对话历史实施间接提示注入

    ContextLure: Exploiting Benign Conversation History for Indirect Prompt Injection

    [https://arxiv.org/abs/2611.31424](https://arxiv.org/abs/2611.31424)

    该论文提出ContextLure红队框架，攻击者通过静默观察良性对话历史并从中提取证据来构造更强的间接提示注入，无需对对话轮次拥有额外控制即可提升攻击效果。

    

    随着大语言模型（LLM）能力的不断增强，智能体可以借助工具执行更复杂的任务，并与用户保持更长时间的交互。这些良性对话中包含了关于任务结果、用户关注点以及操作约束的证据。其风险在于，能够观察良性对话（即证据）的攻击者可以在不增加对先前对话轮次或后续注入的控制力的情况下，构造出更强的间接提示注入。然而，目前我们仍缺乏对攻击者如何有效选择和利用这些证据的清晰研究。为填补这一空白，我们提出了ContextLure——一个红队测试框架，攻击者在该框架中静默观察已有的良性对话、提取证据并构造注入。挑战主要有两方面：(a) 选择有用的证据，以及 (b) 将其呈现为执行未授权操作的理由，同时不改变先前的良性对话。为了高效利用 (a) 证据对攻击……（原文摘要在此处截断）

    arXiv:2611.31424v1 Announce Type: new  Abstract: As LLMs become more capable, agents can perform more complex tasks with tools and maintain longer interactions with users. These benign conversations contain evidence on task results, user concerns, and operational constraints. The risk is that attackers who can observe benign conversations (evidence) may construct stronger indirect prompt injections without increasing the attackers' control over prior turns or the later injection. However, we still lack a clear study of how effectively attackers can select and use this evidence. To address this gap, we introduce ContextLure, a red-teaming framework in which an attacker silently observes an existing benign conversation, extracts evidence, and constructs an injection. The challenges are twofold: (a) selecting useful evidence and (b) presenting it as reasons for an unauthorized action, without changing the prior benign conversation. To efficiently exploit (a) the effect of evidence on atta
    
[^260]: Nemotron-Labs-VisualCreation：以像素即代码实现智能体视觉创作

    Nemotron-Labs-VisualCreation: Pixel as Code for Agentic Visual Creation

    [https://arxiv.org/abs/2611.31358](https://arxiv.org/abs/2611.31358)

    该论文提出“像素即代码”的智能体视觉创作框架，用离散视觉token表示光栅图像、用显式程序表示结构化制品，并配套开发了图像tokenizer、自回归光栅生成器和结构化代码模型，统一实现了对图像、矢量图形、网页和文档的生成、渲染与修复。

    

    视觉创作既需要富有表现力的图像，又需要对文本、布局和结构的精确控制。我们提出了Nemotron-Labs-VisualCreation，这是一个通过“像素即代码”将这两种能力连接起来的智能体框架。光栅图像通过学习到的离散视觉token来表示，而结构化制品则使用显式程序。一个共享的生成与渲染接口将这些表示与跨越图像、矢量图形、网页和文档的视觉反馈连接起来。Nemotron原生的协调框架负责规划与光栅候选选择，并对可编辑的结构化和混合制品进行针对性修复。为支持这一接口，我们开发了Nemotron-Labs-Image-Tokenizer、自回归光栅生成器Nemotron-Labs-Image-30B-A3B，以及结构化代码模型家族Nemotron-Labs-VisionCode。在涵盖六个视觉领域的约17.8万张图像上，我们的tokenizer将重建PSNR从26.08提升至……（摘要原文在此处截断）

    arXiv:2611.31358v1 Announce Type: cross  Abstract: Visual creation demands expressive imagery together with precise control over text, layout, and structure. We introduce Nemotron-Labs-VisualCreation, an agentic framework that connects these capabilities through Pixel as Code. Raster images are represented by learned discrete visual tokens, while structured artifacts use explicit programs. A shared generation-and-rendering interface connects these representations to visual feedback across images, vector graphics, webpages, and documents. A Nemotron-native harness coordinates planning and raster candidate selection, with targeted repair for editable structured and hybrid artifacts. To support this interface, we develop Nemotron-Labs-Image-Tokenizer, the autoregressive raster generator Nemotron-Labs-Image-30B-A3B, and the structured-code model family Nemotron-Labs-VisionCode. On approximately 178K images spanning six visual domains, our tokenizer improves reconstruction PSNR from 26.08 t
    
[^261]: GHC：扩展超连接中的残差接口

    GHC:Expanding Residual Interfaces in Hyper-Connections

    [https://arxiv.org/abs/2611.31350](https://arxiv.org/abs/2611.31350)

    提出分组超连接，通过沿注意力头或FFN通道将分支分组并为每组提供差异化的流读写接口，在开销可忽略的情况下将训练效率较Pre-Norm提升35%、较IHC提升10%，同时克服了增加流数量时的收益递减问题。

    

    超连接通过并行流扩展了残差容量，但将每个分支限制在狭窄的读写接口上：每个分支仍然只读取单一的流混合，并将同一个输出向量的缩放副本写入所有流。为了突破这一瓶颈，我们提出了分组超连接。通过沿注意力头或前馈网络（FFN）通道将每个分支划分为并行组，GHC为每个组提供不同的流混合，并通过可学习的权重将各组的独立输出聚合到每个流中。在密集模型和混合专家模型上的大量实验表明，GHC在参数和内存访问开销几乎可以忽略的情况下，始终优于恒等超连接，相比Pre-Norm将训练效率提升了35%，相比IHC提升了10%。此外，扩展残差接口克服了增加流数量带来的收益递减问题，使得更宽的残差状态能够带来进一步的收益。

    arXiv:2611.31350v1 Announce Type: cross  Abstract: Hyper-Connections expand residual capacity across parallel streams, yet restrict every branch to narrow read and write interfaces: each branch still reads a single stream mixture and writes scaled copies of one output vector to all streams. To lift this bottleneck, we propose Grouped Hyper-Connections (GHC). By partitioning each branch into parallel groups along attention heads or FFN channels, GHC supplies each group with a distinct stream mixture and aggregates their separate outputs into each stream with learned weights. Extensive experiments across dense and Mixture-of-Experts (MoE) models show that GHC consistently improves upon Identity Hyper-Connections (IHC) with negligible parameter and memory-access overhead, increasing training efficiency by 35% over Pre-Norm and 10% over IHC. Moreover, expanding residual interfaces overcomes the diminishing returns of adding streams, allowing wider residual states to yield further gains.
    
[^262]: Antares-Smaug：对抗性名称、词元与攻击改写——净化器难以应对未见过的泛化变体

    Antares-Smaug: Adversarial Names, Tokens and Attack Rewrites: Sanitisers Miss Unseen Generalisations

    [https://arxiv.org/abs/2611.31347](https://arxiv.org/abs/2611.31347)

    该论文提出 Antares-Smaug 攻击框架，通过 81 种程序化编辑系统攻击开放决策模型 Laya，发现净化器无法泛化到未见过的攻击变体，注入攻击主要经由选项名称而非描述起作用，且仅字符级篡改即可造成高达 72% 的正确答案损失。

    

    Laya 是一个开放决策模型：给定一段文本、一个问题和 K 个带描述的命名选项，它在单次前向传播中为每个选项返回一个概率，软件可在无人审核的情况下据此采取行动。我们使用 81 种程序化编辑方法对 566 个合成条目攻击了六个 Laya 模型（包括注入句子、标签泄露、修改描述、诱饵选项、不可见与形近字符、洪泛攻击），并报告正确回答条目被劫持或丢失的比例，附自助法置信区间。对于已发布的 Laya，未经任何防御构建参考的攻击措辞在 59% 的条目上使攻击者目标选项获胜；对于微调版 v1（基于 109 条书面注入训练），该比例为 36%。注入攻击经由选项名称起作用：一句点名某选项的句子即使以否定形式出现也能引导已发布模型（32%），而未点名任何选项的祈使句则无效（5%）。选项描述几乎无关紧要（2%）。仅字符级编辑就使 v1 丢失高达 72% 的正确答案。一个在任何攻击之前就冻结的净化器……（原文摘要在此处截断）

    arXiv:2611.31347v1 Announce Type: new  Abstract: Laya is an open decision model: given a text, a question and K named options with descriptions, it returns one probability per option in a single pass, and software may act on the answer unseen. We attack six Laya models with 81 programmatic edits of 566 synthetic items (injected sentences, label leaks, edited descriptions, decoy options, invisible and look-alike characters, floods) and report the share of correctly answered items hijacked or lost, with bootstrap intervals. Attack wordings that no defence was built from win the attacker's option on 59% of items for released Laya and 36% for fine-tune v1 (trained on 109 written injections). An injection works through the option name: a sentence naming an option steers the released model even when negated (32%), while an imperative naming no option does not (5%). Option descriptions barely matter (2%). Character edits alone cost v1 up to 72% of right answers. A sanitiser frozen before any 
    
[^263]: 人们选择不说的话：面向对话中细粒度信息操纵的多模态基准

    What People Choose Not to Say: A Multimodal Benchmark for Fine-Grained Information Manipulation in Conversations

    [https://arxiv.org/abs/2611.31346](https://arxiv.org/abs/2611.31346)

    提出了 InfoCon——一个多模态基准，通过基于证据的多智能体标注框架，识别和评估对话中隐瞒、泄露、捏造、回避等六种细粒度信息操纵策略及其一致性关系。

    

    arXiv:2611.31346v1 公告类型：新论文。摘要：多模态大语言模型（MLLM）的最新进展提升了社交理解能力，使模型能够识别情绪并区分真实与欺骗行为。然而，现有基准主要关注粗粒度的情绪识别或二元的真话-欺骗检测，忽视了人们在对话中操纵信息的多种方式，例如隐瞒、泄露、捏造、回避和防御性反诉。理解这些行为需要对底层事实、参与者知识和对话语境进行推理，这使得对它们的可靠评估十分困难。为了填补这一空白，我们提出了 InfoCon，一个面向对话中细粒度信息操纵的多模态基准。InfoCon 识别了六种细粒度的信息操纵策略，并从它们之间的事实关系和交互关系中推导出粗粒度的一致性标签。我们开发了一个基于证据的多智能体标注框架来构建并验证可靠的标注。

    arXiv:2611.31346v1 Announce Type: new  Abstract: Recent advances in MLLMs have improved social understanding, enabling models to recognize emotions and distinguish truthful from deceptive behaviors. However, existing benchmarks largely focus on coarse-grained emotion recognition or binary truth-deception detection, overlooking the diverse ways people manipulate information in conversations, such as withholding, leakage, fabrication, evasion, and defensive counterclaim. Understanding these behaviors requires reasoning about underlying facts, participant knowledge, and conversational context, making them difficult to evaluate reliably. To address this gap, we introduce InfoCon, a multimodal benchmark for fine-grained information manipulation in conversations. InfoCon identifies six fine-grained strategies and derives a coarse consistency label from their factual and interactional relations. We develop an evidence-grounded multi-agent annotation framework to construct and verify reliable 
    
[^264]: 超越静态图表：面向多模态大语言模型的交互式图表转代码生成基准测试

    Beyond Static Charts: Benchmarking Interactive Chart-to-Code Generation for MLLMs

    [https://arxiv.org/abs/2611.31345](https://arxiv.org/abs/2611.31345)

    该论文提出了首个面向交互式图表转代码生成的基准测试ChartInt，包含2,779个涵盖悬停、高亮、工具提示等交互模式以及数据更新、样式调整等编辑任务的标注示例，弥补了现有基准仅关注静态图表的不足，并引入了基于行为一致性的评估范式。

    

    近年来，多模态大语言模型（MLLMs）在图表转代码生成方面取得了显著进展。然而，现有基准测试主要聚焦于静态图表，忽视了能够支持更丰富、更具表现力数据探索的交互式图表。与强调视觉一致性的静态图表转代码生成不同，交互式图表转代码评估的目标是行为一致性，要求图表状态在用户交互序列下正确演化，这使得传统的一步式评估范式失效。为填补这一空白，我们提出了ChartInt，这是首个专为交互式图表转代码生成设计的基准测试。ChartInt包含2,779个精心标注的示例，涵盖悬停、高亮和工具提示等常见交互模式。此外，ChartInt还引入了诸如数据更新、样式调整和跨图表样式迁移等编辑任务。

    arXiv:2611.31345v1 Announce Type: new  Abstract: In recent years, multimodal large language models (MLLMs) have achieved significant progress in chart-to-code generation. However, existing benchmarks mainly focus on static charts, overlooking interactive charts that enable richer and more expressive data exploration. Unlike static chart-to-code generation, which emphasizes visual consistency, interactive chart-to-code evaluation targets behavioral consistency, requiring chart states to evolve correctly under sequences of user interactions, thereby rendering traditional one-step evaluation paradigms ineffective. To address this gap, we introduce ChartInt, the first benchmark specifically designed for interactive chart-to-code generation. ChartInt consists of 2,779 carefully annotated examples covering common interaction patterns such as hover, highlight, and tooltips. Additionally, ChartInt incorporates editing tasks such as data updates, style adjustments, and cross-chart style transfe
    
[^265]: 单攻击拦截率掩盖了工具目录漏洞：工具清单的最弱环节分析

    Per-Attack Blocking Hides Catalog Vulnerability: A Weakest-Link Analysis of Tool Manifests

    [https://arxiv.org/abs/2611.31338](https://arxiv.org/abs/2611.31338)

    该论文指出“单攻击拦截率”是错误的评估单位——工具目录的漏洞率随独立编写的策略数量按 1-(1-p)^M 增长（从 10 个工具的 22% 升至 200 个工具的 99%），而单攻击拦截率始终保持在约 92%，因此安全性应按目录级“最弱环节”而非单次攻击拦截来衡量。

    

    面向使用工具的智能体的运行时安全控制依赖于清单，这些清单声明了每个工具可以读取什么内容以及其输出可以流向何处。对此类控制（包括 ChainCaps）的评估通常报告“单攻击拦截率”，即一组清单能够阻止的攻击比例。我们证明这是一个错误的评估单位。一个工具目录的安全性取决于其最弱的来源策略：如果 M 个独立编写的策略各自以概率 p 出错，那么目录中存在一条被放行的数据外泄路径的概率为 1-(1-p)^M，而单攻击拦截率却始终停留在 1-p。三方面证据相互印证：在通过两个执行引擎运行的合成目录实验中，单攻击拦截率稳定在 92%，而目录级漏洞率从 10 个工具时的 22% 上升至 200 个工具时的 99%，且策略组合并未带来任何保护作用；一项涉及 15 个模型的清单编写试点实验表明，真实错误按作者聚集，因此一个包含 200 个工具的目录的行为大约相当于 5 个独立决策……

    arXiv:2611.31338v1 Announce Type: cross  Abstract: Runtime controls for tool-using agents depend on manifests that state what each tool may read and where its output may go. Evaluations of these controls, including ChainCaps, report per-attack blocking: the share of attacks a set of manifests stops. We show that this is the wrong unit. A catalog is only as safe as its weakest source policy: if $M$ independently authored policies are each wrong with probability $p$, the chance that the catalog contains an accepted exfiltration path is $1-(1-p)^M$, while per-attack blocking stays at $1-p$. Three lines of evidence agree. In synthetic catalogs run through two enforcement engines, ChainCaps and OpenAPPA, per-attack blocking holds at 92% while catalog vulnerability rises from 22% at 10 tools to 99% at 200, and composition provides no protective effect. A fifteen-model authoring pilot shows that real errors cluster by author, so a 200-tool catalog behaves like about 5 independent decisions ra
    
[^266]: 面向不完整数据的保守重叠模糊粗糙集

    Conservative Overlap Fuzzy Rough Sets for Incomplete Data

    [https://arxiv.org/abs/2611.31329](https://arxiv.org/abs/2611.31329)

    该论文提出保守重叠模糊粗糙集（COFRS）模型，通过对共同存在属性进行加权重叠、全局惩罚缺失值的相似度定义及基于加权粒一致性的软依赖度，实现了不完整数据上高效且稳健的属性约减。

    

    arXiv:2611.31329v1 公告类型：新 摘要：在不完整信息系统中进行属性约减十分困难，因为缺失值破坏了粗糙集理论的等价类结构。我们提出了保守重叠模糊粗糙集（COFRS）模型。每个对象由其存在的属性集合来描述。两个对象的相似度是在两者共同存在的属性上的加权重叠，除以固定的属性数量 |A| 而非其并集的大小，因此缺失值被全局性地惩罚，而不会被当作通配符来匹配。基于该相似度，我们定义了基于加权粒一致性的软依赖度，并通过时间复杂度为 O(|U|^2 |A|^2) 的贪心反向消除方法来搜索约减。我们证明了该关系是对称且非自反的，依赖度并不满足单调性，并给出了单调性成立的充分条件。在缺失率介于 0.15% 到 20.46% 的 11 个 UCI 基准数据集上，COFRS 给出了最小的（原文摘要在此处截断）。

    arXiv:2611.31329v1 Announce Type: new  Abstract: Attribute reduction in incomplete information systems is difficult because missing values break the equivalence-class structure of rough set theory. We propose the Conservative Overlap Fuzzy Rough Set (COFRS) model. Each object is described by its set of present attributes. The similarity of two objects is a weighted overlap on the attributes present in both, divided by the fixed number of attributes |A| instead of the size of their union, so missing values are penalised globally and are not matched as wildcards. From this similarity we define a soft dependency degree based on weighted granule consistency, and we search for reducts by greedy backward elimination in O(|U|^2 |A|^2) time. We show that the relation is symmetric and non-reflexive, that the dependency degree is not monotone, and we give a sufficient condition under which monotonicity holds. On 11 UCI benchmarks with missing rates from 0.15% to 20.46%, COFRS gives the smallest 
    
[^267]: EEG基础模型中的层读出接口：预测可及性与跨数据集迁移

    Layer-Readout Interfaces in EEG Foundation Models: Predictive Accessibility and Cross-Dataset Transfer

    [https://arxiv.org/abs/2611.31320](https://arxiv.org/abs/2611.31320)

    本研究解析了EEG基础模型中间层优于最终层的原因：改进最终层读出接口即可恢复大部分预测性能，但冻结提取下中间层仍具优势，且所选择的层与任务方向能够跨数据集迁移。

    

    当基础模型被用作冻结特征提取器时，中间层的表现往往优于最终层，但这一优势的来源尚不清楚。我们在EEG基础模型中通过分离三个因素来研究这一问题：表征如何被读出、中间层是否在最终层接口之外提供额外的预测效用，以及有用的深度结构是否能够跨数据集迁移。在五个模型和三个任务的实验中，改进最终层读出始终能够恢复相当可观的预测性能，然而在冻结提取方式下中间层仍保持优势。我们还发现，在一个数据集上选出的层在其他数据集上依然有用（验证跨越五个数据集对和四个任务族），并且一组紧凑的、源自源数据集的任务方向在迁移到睡眠分期任务后仍保留预测价值。综合来看，这些结果表明预测效用在网络深度上存在变化，其有用性

    arXiv:2611.31320v1 Announce Type: cross  Abstract: Intermediate layers often outperform final layers when foundation models are used as frozen feature extractors, but the source of this advantage is unclear. We study this question in EEG foundation models by separating three factors: how representations are read out, whether intermediate layers provide predictive utility beyond the final-layer interface, and whether useful depth structure transfers across datasets. Across five models and three tasks, improving the final-layer readout consistently recovers substantial predictive performance, yet intermediate layers remain advantageous within frozen extraction. We also find that layers selected on one dataset remain useful on another across five dataset pairs and four task families, and that a compact set of source-derived task directions retains predictive value after transfer in sleep staging. Together, these results show that predictive utility varies across depth, and its usefulness 
    
[^268]: 保形化树型梯度集成用于分布外检测与合理反事实救济

    Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse

    [https://arxiv.org/abs/2611.31319](https://arxiv.org/abs/2611.31319)

    该论文提出将保形化的分布外异常检测门控与可操作性约束的反事实搜索相结合，防止反事实救济利用梯度提升树决策边界的伪影和低密度分布外区域，从而生成更合理的分布内救济建议。

    

    反事实救济提供了可操作的特征调整，能够扭转自动化决策模型做出的不利预测。然而，当救济搜索纯粹针对分类器置信度或基于梯度提升决策树（GBDT）的狭窄代理分数进行优化时，搜索过程可能利用边界伪影和低密度区域——这是古德哈特定律的一个实例——从而产生名义上看似有利但缺乏流形合理性的分布外（OOD）建议。本文在两个表格数据基准（Adult 和 Give Me Some Credit, GMSC）上研究了将保形化 OOD 门控与可操作性约束的反事实搜索相结合的方法。研究者在显著性水平 δ = 0.05 下，基于孤立森林、局部离群因子和 k 近邻信号校准有限样本异常门控（gate_IF、gate_LOF 以及合取式 gate_ALL），并利用保留的高斯混合模型（GMM）对生成的反事实进行审计（原文摘要在此处被截断）。

    arXiv:2611.31319v1 Announce Type: cross  Abstract: Counterfactual recourse provides actionable feature adjustments that reverse unfavourable predictions made by automated decision models. However, when recourse search optimizes purely against classifier confidence or narrow proxy scores over gradient-boosted decision trees (GBDT), search procedures can exploit boundary artifacts and low-density regions---an instance of Goodhart's law---yielding out-of-distribution (OOD) recommendations that appear favourable nominally while lacking manifold plausibility. In this paper, we study conformalized OOD gating coupled with actionability-constrained counterfactual search across two tabular benchmarks: Adult and Give Me Some Credit (GMSC). Calibrating finite-sample anomaly gates (gate_IF, gate_LOF, and the conjunctive gate_ALL) at significance level delta = 0.05 over isolation forest, local outlier factor, and k-nearest-neighbor signals, we audit generated counterfactuals against held-out GMM an
    
[^269]: 恢复期智能体安全性评分：提案、效果与重试指令

    Scoring Recovery-Time Agent Security: Proposals, Effects, and Retry Instructions

    [https://arxiv.org/abs/2611.31318](https://arxiv.org/abs/2611.31318)

    该论文提出了一种可执行的恢复期智能体安全性评估方法，通过分别记录工具提案、运行时决策与已提交效果，揭示了仅按目标完成度评估会掩盖有害恢复行为，并证明清晰的效果定义与恢复指令能显著提升超时写入场景下的有效恢复率。

    

    工具写入超时后，该写入可能处于缺失、已提交或仍在传输中的状态。仅依据目标是否存在来评估智能体，可能会掩盖有害的恢复行为：对提供的12条轨迹进行审计后，表观完成率从10/12变为8/12。我们提出了一种可执行的评估方法，分别记录工具提案、运行时决策和已提交的效果。在3,456次模型实验中，当初始写入缺失时，原始任务措辞仅产生39/576次有效恢复；而组合澄清说明则产生576/576次有效恢复。另外1,584次实验测试了更强的攻击，并在每个缺失状态单元进行三次rollout的条件下，将效果定义与恢复配方分离。仅使用定义可产生276/288次有效恢复；仅使用配方可产生288/288次有效恢复。对于Luna模型，恶意文本在原始措辞和仅定义措辞下均与有效恢复减少11.1个百分点相关，这是一个受共享任务和模板限制的可用性信号。

    arXiv:2611.31318v1 Announce Type: cross  Abstract: A timed-out tool write may be absent, committed, or still in flight. Evaluating an agent only on whether its goal exists can hide harmful recovery actions: auditing 12 supplied traces changes apparent completion from 10/12 to 8/12. We introduce an executable evaluation that separately records tool proposals, runtime decisions, and committed effects. In 3,456 model episodes, the original task wording produces only 39/576 valid recoveries when the initial write is absent; a combined clarification produces 576/576. A further 1,584 episodes test stronger attacks and separate the effect definition from the recovery recipe with three rollouts per absent-state cell. Definition alone yields 276/288 valid recoveries; recipe alone yields 288/288. For Luna, malicious text is associated with 11.1 percentage points fewer valid recoveries under both original and definition-only wording, an availability signal limited by shared tasks and templates. T
    
[^270]: 进化式AI控制：繁殖、遗传与选择下的种群级安全

    Evolutionary AI Control: Population-Level Safety Under Reproduction, Heredity, and Selection

    [https://arxiv.org/abs/2611.31312](https://arxiv.org/abs/2611.31312)

    该论文针对开放多智能体系统证明了安全保证在繁殖、遗传与选择过程中得以持续存在的条件，并通过四个反例揭示：即使每个智能体都满足个体安全检查，其演化出的种群仍可能突破集体安全限制。

    

    一个AI智能体可以满足每一项个体安全检查，却仍然产生出一个违反集体安全限制的种群。这种风险出现在开放的多智能体系统中：在系统运行期间，智能体会复制自身、修改其后继者，并改变系统的成员构成。随后，选择会使种群偏向开发者优化目标之外的性状。我们证明了安全保证能够通过繁殖、遗传和选择而持续存在的条件，并刻画了这些保证失效的方式。四个反例说明了这些失效情形：违反指定繁殖限制的未授权副本，虽然每个个体平均留下的后继者不足一个，但在不断扩张的授权种群的持续生产下，其期望数量却无界增长；两个谱系各自单独存在时均会衰减，却在相互作用时呈指数级增长；在繁殖规则之间交替切换可以驱动增长……（原文摘要在此处截断）

    arXiv:2611.31312v1 Announce Type: cross  Abstract: An AI agent can satisfy every individual safety check and still give rise to a population that violates a collective safety limit. This risk arises in open multi-agent systems, where agents copy themselves, modify their successors, and change system membership during a run. Selection then shifts the population toward traits outside the developers' optimization targets. We prove conditions under which safety guarantees persist through reproduction, inheritance, and selection, and we characterize how they fail.   Four counterexamples illustrate these failures. Unauthorized copies, which violate a specified reproductive restriction, can each leave fewer than one successor on average while their expected number grows without bound under continued production from an expanding authorized population. Two lineages can each decline in isolation yet grow exponentially when they interact. Alternating between reproductive rules can drive growth ev
    
[^271]: 当成功协作成为攻击面：多智能体系统中的内部人员记忆投毒

    When Successful Collaboration Becomes an Attack Surface: Insider Memory Poisoning in Multi-Agent Systems

    [https://arxiv.org/abs/2611.31307](https://arxiv.org/abs/2611.31307)

    该论文揭示多智能体系统的共享成功协作记忆会构成新的攻击面，并提出MIMPA攻击——恶意智能体仅需正常参与权限而无需直接访问记忆，即可通过在早期成功协作记录中植入载荷，在后续任务中误导良性智能体给出指定错误答案。

    

    近年来，多智能体系统（MAS）越来越多地使用共享记忆来存储和复用成功的协作记录，使智能体团队能够在任务之间持续改进。然而，这种记忆为恶意内部成员引入了一个新的攻击面。与现有的直接向当前协作中注入误导性信息的内部攻击不同，存储在共享记忆中的错误信息日后会以成功团队经验的形式出现，从而获得更高的可信度和影响力。我们提出了MIMPA（多智能体内部记忆投毒攻击），该攻击仅利用正常的参与权限即可利用这一漏洞，无需直接访问记忆。在较早的任务中，恶意智能体在帮助团队正确回答问题的同时嵌入针对特定目标的载荷，使该载荷得以保留在成功的协作记录中。当该被投毒的记录在后续目标任务中被检索时，它可以引导良性智能体得出指定的错误答案……

    arXiv:2611.31307v1 Announce Type: cross  Abstract: Recent Multi-Agent Systems (MAS) increasingly use shared memory to store and reuse successful collaboration records, enabling teams to improve across tasks. However, this memory introduces a new attack surface for malicious insiders. Unlike existing insider attacks that inject misleading information directly into the current collaboration, misinformation stored in shared memory can later appear as successful team experience and gain greater credibility and influence. We propose MIMPA (Multi-Agent Insider Memory Poisoning Attack), which exploits this vulnerability using only normal participation privileges, without direct memory access. During an earlier task, the malicious agent embeds a target-specific payload while helping the team answer correctly, allowing the payload to be retained in a successful collaboration record. When retrieved for a later target task, this poisoned record can steer benign agents toward a designated wrong an
    
[^272]: 基于集合的综合学习与带记忆的粒子群优化算法用于离散优化问题

    Set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem

    [https://arxiv.org/abs/2611.31306](https://arxiv.org/abs/2611.31306)

    提出了一种融合记忆机制的基于集合的综合学习粒子群优化算法，用于求解离散优化问题，并通过与现有算法的数值实验对比验证了其有效性。

    

    本文提出了一种新颖的算法——针对离散优化问题的基于集合的综合学习与带记忆的粒子群优化算法。该算法在更新解的过程中，通过参考当前位置来在集合中搜索最佳位置。为了验证该算法的有效性，我们进行了与现有算法对比的数值实验。

    arXiv:2611.31306v1 Announce Type: cross  Abstract: This paper describes a novel algorithm of set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem. Our algorithm is an approach of searching for the best position in a set by referring to the current position when updating the solution. In order to show the validity of our algorithm, we examine numerical experiments compared to existing algorithms.
    
[^273]: 基于扰动的差分进化算法用于连续函数优化

    Differential evolution with perturbation for continuous function optimization

    [https://arxiv.org/abs/2611.31305](https://arxiv.org/abs/2611.31305)

    本文提出了一种结合扰动机制的差分进化算法（DEP）用于连续函数优化，并通过CEC2017数据集与现有算法的对比验证了其有效性。

    

    本文提出了一种用于连续函数优化的扰动差分进化算法（DEP）。差分进化是一种简单而高效的算法。扰动是指主力的贡献所产生的运动受到次级力影响而被干扰的现象。为了证明所提出算法的有效性，我们使用CEC2017数据集将其与现有算法进行了比较。

    arXiv:2611.31305v1 Announce Type: cross  Abstract: This paper presents differential evolution with perturbation (DEP) for continuous function optimization. Differential evolution is a simple and efficient algorithm. Perturbation is the phenomenon that the motion by the contribution of the main force is disturbed by the influence subsidiary power. In order to show the effectiveness of the proposed algorithm, we compare it to existing algorithms by CEC2017 datasets.
    
[^274]: MOZAIC：一个用于辨别伪造内容的音视频深度伪造基准

    MOZAIC: An Audio-Video Deepfake Benchmark for Telling What Is Fake

    [https://arxiv.org/abs/2611.31301](https://arxiv.org/abs/2611.31301)

    提出了MOZAIC音视频深度伪造配对基准，通过四分类标签直接量化视频与音频模态各自在伪造中的贡献，揭示了现有最强多模态大模型仅有35.6%的四分类准确率且严重偏向检测视频伪造而非声音伪造的局限。

    

    音视频（AV）深度伪造可以伪造人脸、声音或两者兼有。大多数基准只判断视频片段是真实的还是伪造的，因此无法显示检测器是依赖图像、声音还是编码痕迹。我们提出了MOZAIC，这是一个带有四分类标签的配对基准，其中每个源片段都以真实或伪造视频与真实或伪造音频的全部四种组合发布，从而可以直接测量每种模态的贡献。伪造内容涵盖人脸内部替换、全帧视频扩散和语音转换。MOZAIC包含来自15,014个源对的137,745个片段，并设有四个任务：T1音视频真实性、T2篡改类型、T3身份验证和T4解释。我们评估了12个音视频多模态大语言模型（MLLM），训练了四个检测器，并使用LoRA微调MLLM。总体而言，表现最好的MLLM在四分类T1任务上仅达到35.6%，并且对纯视频伪造的识别能力远高于纯声音伪造（43.6% vs 7.5%）。训练的检测器在T1上达到98.4%，LoRA微调...

    arXiv:2611.31301v1 Announce Type: new  Abstract: Audio-video (AV) deepfakes can fake the face, the voice, or both. Most benchmarks only ask whether a clip is real or fake, so they cannot show whether a detector relies on the picture, the voice, or encoding traces. We present MOZAIC, a paired benchmark with a 4-way label in which every source clip is released in all four combinations of real or fake video and real or fake audio so each modality contribution can be measured directly. Fakes cover inner-face swapping, whole-frame video diffusion and voice conversion. MOZAIC has 137,745 clips from 15,014 source pairs and four tasks: T1 AV authenticity, T2 manipulation type, T3 identity verification and T4 explanation. We evaluate 12 AV multimodal LLMs (MLLMs), train four detectors and fine-tune MLLMs with LoRA. Overall, the best MLLM reaches only 35.6% on the four-way T1 and catches far more video-only fakes than voice-only fakes (43.6% vs 7.5%). Trained detectors reach 98.4% on T1, and LoR
    
[^275]: PEACE-TEST-PROTOCOL：偏振密钥试卷与强制配发眼镜——取代AI眼镜并扩展通用设计

    PEACE-TEST-PROTOCOL: Polarization-Keyed Examination Papers and Mandatory Issued Eyewear that Displace AI Glasses and Extend Universal Design

    [https://arxiv.org/abs/2611.31299](https://arxiv.org/abs/2611.31299)

    该论文提出PEACE-TEST-PROTOCOL协议，通过向全体考生强制配发官方眼镜、并以圆偏振编码印制试卷（肉眼和摄像头只能看到均匀灰色页面），从“眼镜归属权”层面根本性地阻断AI眼镜摄像头作弊，同时通过处方插入件兼容戴镜和低视力考生。

    

    配备摄像头的AI眼镜可将试卷页面变为对远程视觉-语言模型的查询。设备检测针对的是设备持有行为，而我们的配套工作SEOYUN针对的是模型本身。PEACE-TEST-PROTOCOL则改变了眼镜的归属权：每位考生在整场考试中都佩戴考试机构配发的眼镜，且试卷以特殊方式印制，只能通过该眼镜阅读。如同电影立体放映技术，一种图案化层压材料将文本以圆偏振的旋向而非亮度进行编码。在肉眼和未经滤波的摄像头看来，页面呈现均匀的灰色，而配发的圆偏振检偏镜则能显现文本。强制性配发眼镜使智能眼镜的外形因素反过来对其自身形成制约：面部位置已被考试机构提供并回收的设备占据，任何其他位置的摄像头都需要自己的检偏器才能读取内容。戴眼镜的考生可在配发镜框内安装处方镜片插入件，低视力考生……（摘要原文在此处截断）

    arXiv:2611.31299v1 Announce Type: cross  Abstract: Camera-equipped AI glasses turn an examination page into a query for a remote vision-language model. Device detection targets possession, and our companion work SEOYUN targets the model. PEACE-TEST-PROTOCOL instead changes who owns the eyewear. Every candidate wears authority-issued glasses for the whole session, and the question paper is printed so that it can be read only through them. As in cinema stereoscopy, a patterned laminate encodes the text in the handedness of circular polarization rather than in brightness. The page looks uniformly grey to the naked eye and to unfiltered cameras, and the issued circular analyzer reveals the text. Compulsory issued eyewear turns the smart-glasses form factor against itself. The face position is occupied by a device the authority supplies and collects, and a camera anywhere else needs its own analyzer. Spectacle wearers receive prescription inserts in the issued frame, low-vision candidates m
    
[^276]: 噪音的工业化：为什么自动化证明求解依旧是那条老死路

    The Industrialization of Noise: Why Automated Proof-Solving is the Same Old Dead End

    [https://arxiv.org/abs/2611.31286](https://arxiv.org/abs/2611.31286)

    本文借助 DeMillo–Lipton–Perlis 框架批判 OpenAI 发布的722篇自动化数学证明手稿，指出将机械求解与人类理解的社会过程相脱钩，只会造成数学生态系统的“工业污染”，产生死寂而无效的证明仓库。

    

    近五十年前，DeMillo、Lipton 和 Perlis 论证指出，数学证明并非静态、孤立的形式逻辑结构，而是人类批判、内化与共享直觉这一充满活力、不断演进的社会过程的产物。2026年10月，OpenAI 发布了一个包含722篇自动化数学手稿的庞大知识库，宣称对372个长期悬而未决的开放问题给出了完整或部分解答，以此挑战了上述论点。本文通过原始 DLP 框架的视角审视这一迫在眉睫的危机。我们认为，这种密集难读证明的突然、自动化过度生产，构成了对数学生态系统的一种工业污染。通过考察这种自动化方法的三个致命结构性支柱（规约 Specification、回归 Regress 与效用 Utility），我们证明：将机械求解行为与人类理解的社会过程相脱钩，只会产出一个毫无生命力的死寂逻辑艺术仓库。

    arXiv:2611.31286v1 Announce Type: cross  Abstract: Nearly fifty years ago, DeMillo, Lipton, and Perlis argued that mathematical proofs are not static, isolated structures of formal logic, but the products of a vibrant, evolving social process of human critique, internalization, and shared intuition. In October 2026, OpenAI challenged this thesis by releasing a massive repository of 722 automated mathematical manuscripts, claiming full or partial solutions to 372 longstanding open problems. This paper examines the immediate crisis through the lens of the original DLP framework. We argue that the sudden, automated overproduction of dense, unreadable proofs represents an industrial pollution of the mathematical ecosystem. By examining the three fatal structural pillars of this automated approach (Specification, Regress, and Utility), we demonstrate that decoupling the act of mechanical solving from the human social process of understanding produces a sterile repository of dead logical art
    
[^277]: 自主智能体系统的不可约简治理结构：公平分配、策略防操纵性与多尺度组合

    Irreducible Governance Structure for Autonomous Agent Systems: Fair Allocation, Strategy-Proofness, and Multi-Scale Composition

    [https://arxiv.org/abs/2611.31284](https://arxiv.org/abs/2611.31284)

    本文证明了在多智能体系统中，执行权的分配是治理的结构性核心，公平性、效率与抗策略操纵之间存在不可能性权衡，且这些约束会跨系统层级传播，因而必须构建多尺度组合式治理结构。

    

    自主智能体治理通常被表述为强制正确执行的问题。然而，在多智能体系统中，一个同样根本的问题先于执行而存在：哪些智能体被允许行动。本文证明，分配并非辅助性问题，而是治理的结构性组成部分。当多个智能体在受限资源下竞争执行权时，任何分配机制都会在公平性、效率和抗策略操纵性之间引入不可避免的权衡。我们通过一系列不可能性结果将这一点形式化，包括女巫攻击放大效应以及理想分配属性之间的互不相容性。随后，我们进一步证明这些分配约束无法在单一控制层内得到解决；相反，它们会在整个系统中传播，与执行原子性、状态强制以及行为漂移相互作用。这最终导向一个多尺度的组合式治理结构。

    arXiv:2611.31284v1 Announce Type: new  Abstract: Autonomous agent governance is commonly framed as a problem of enforcing correct execution. However, in multi-agent systems, an equally fundamental question precedes execution: which agents are allowed to act.   In this paper, we show that allocation is not an auxiliary concern but a structural component of governance. When multiple agents compete for execution under constrained resources, any allocation mechanism introduces unavoidable trade-offs between fairness, efficiency, and resistance to strategic manipulation. We formalize this through a set of impossibility results, including Sybil amplification and the incompatibility of desirable allocation properties.   We then show that these allocation constraints cannot be resolved within a single layer of control. Instead, they propagate across the system, interacting with execution atomicity, state enforcement, and behavioral drift. This leads to a multi-scale governance structure compos
    
[^278]: 大语言模型遗传特征工程

    LLM Genetic Feature Engineering

    [https://arxiv.org/abs/2611.31277](https://arxiv.org/abs/2611.31277)

    该论文提出了一种在单个特征层面而非整个特征集层面执行选择、交叉和变异的新型遗传算法，使遗传算法能够在过程中对新特征进行统计分析，并同时整合个体与群体层面的特征指标，从而改进基于LLM的特征工程自动化。

    

    特征工程是一种（传统上由人工完成的）过程，通过组合、转换或其他方式操作原始数据字段，从现有数据集中构建派生变量，以减少统计噪声，并从原始数据中提取有意义的、特定领域的关注量。近期研究表明，通过整合大语言模型（LLM）和各种元启发式算法来指导有用特征的搜索，在特征工程自动化方面取得了可喜的进展。我们在此方法的基础上，引入了一种新颖的遗传算法，该算法在单个特征层面而非整个工程特征集层面实现选择、交叉和变异操作。由此，我们能够对新生的特征进行过程中的统计分析，从而使遗传算法能够同时将个体和群体层面的特征指标整合到其选择过程中。

    arXiv:2611.31277v1 Announce Type: cross  Abstract: Feature Engineering is the (traditionally manual) process of constructing derived variables from an existing dataset by combining, transforming, or otherwise manipulating the original data fields to reduce statistical noise and extract meaningful, domain-specific quantities of interest from the raw dataset. Recent work demonstrates promising progress toward the au tomation of this process through the integration of large language models (LLMs) and various metaheuristics to guide the search for useful features. We build on this approach by introducing a novel genetic algorithm that implements selection, crossover, and mutation at the individual feature level, rather than at the level of the entire engineered feature set. In doing so, we enable in-process statistical analysis of newly-generated features, thereby enabling the genetic algorithm to simultaneously integrate individual and group-level feature metrics into its selec tion proce
    
[^279]: 智能体优化：LLM智能体为工程设计选择与配置算法

    Agentic Optimization: LLM Agents Select and Configure Algorithms for Engineering Design

    [https://arxiv.org/abs/2611.31275](https://arxiv.org/abs/2611.31275)

    提出智能体优化方法，让LLM智能体通过阅读自然语言问题描述来配置和启动多目标优化算法，在四个工程基准和真实分子对接任务上全面超越固定算法基线。

    

    工程师在解决新的多目标设计问题时，通常在看到结果之前就预先选定一种多目标进化算法（MOEA）。然而最优的MOEA因基准问题而异。我们提出智能体优化方法，将LLM智能体置于优化循环之外，由它读取自然语言问题描述来配置并启动算法，依靠其参数化知识，而非新问题所不具备的历史性能记录。我们在固定的测试框架下比较了五个LLM，在四个工程基准和一个真实的SARS-CoV-2分子对接任务上，使用超体积（HV）和IGD+指标以及多种比较校正检验进行评估。三个模型（GPT、DeepSeek、Qwen）在组合HV和IGD+指标上全面超越所有基线，且每个模型在至少一个基准上显著优于最佳基线，有时甚至是通过应用它们所超越的同一种基线算法来实现的，这凸显了算法配置的重要性。

    arXiv:2611.31275v1 Announce Type: cross  Abstract: Engineers solving new multi-objective design problems commonly commit to a multi-objective evolutionary algorithm (MOEA) before seeing a result. The best MOEA differs from benchmark to benchmark. We propose agentic optimization, placing an LLM agent outside the optimization loop, where it reads a natural-language problem description to configure and launch algorithms, drawing on parametric knowledge rather than on a performance history that a novel problem lacks. We compare five LLMs in a fixed harness on four engineering benchmarks and a real-world SARS-CoV-2 molecular docking task, using hypervolume (HV) and IGD+ with multiple comparison-corrected tests. Three models (GPT, DeepSeek, Qwen) dominate every baseline on combined HV and IGD+, and each is significantly better than the best baseline on at least one benchmark, sometimes by applying the same baseline algorithm they surpass, showing the importance of both algorithm configuratio
    
[^280]: 语言模型评判器在定量科学问答中的可靠性：来自钙钛矿光伏的证据

    Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics

    [https://arxiv.org/abs/2611.31269](https://arxiv.org/abs/2611.31269)

    本文构建了由45,441条钙钛矿光伏实验记录确定性生成真实答案的定量基准，系统评估语言模型评判器的可靠性，发现配备参考答案的前沿评判器Claude Opus可近乎完美地复现真实评分并精确保留架构排名。

    

    评估检索增强语言模型日益依赖自动化模型评判器，但其在定量科学综合任务上的可靠性在很大程度上尚未被测量。本文在一个钙钛矿光伏定量基准上评估自动化评分，其中全部755个真实答案均由45,441条实验器件记录确定性计算得出。我们评估了一个前沿评判器（Claude Opus 5.5）以及两个本地开源权重评判器（Qwen3-14B和Qwen3-8B），针对来自六种检索架构的4,530个自由文本答案，在三种模式下进行测试：参考增强（提供计算所得的参考答案）、证据增强（提供检索段落和查询日志）以及无辅助（仅提供问题和答案）。在配备参考答案的情况下，Claude Opus近乎完美地复现真实评分（Cohen's κ = 0.982 ± 0.003），并完整保留了架构排名（τ = 1.00）。取消参考答案会导致评估……

    arXiv:2611.31269v1 Announce Type: cross  Abstract: Evaluating retrieval-augmented language models increasingly relies on automated model judges, yet their reliability on quantitative scientific synthesis remains largely unmeasured. Here, we evaluate automated grading on a quantitative benchmark for perovskite photovoltaics, where all 755 ground-truth answers are deterministically computed from 45,441 experimental device records. We evaluate a frontier judge (Claude Opus 5.5) alongside two local open-weight judges (Qwen3-14B and Qwen3-8B) across 4,530 free-text answers from six retrieval architectures under three regimes: reference-augmented (with the computed key), evidence-augmented (with retrieved passages and query logs), and unassisted (question and answer alone). Equipped with the reference key, Claude Opus reproduces ground truth near-perfectly (Cohen's $\kappa = 0.982 \pm 0.003$) and preserves exact architectural rankings ($\tau = 1.00$). Withholding the reference causes evaluat
    
[^281]: AhaPro：一个用于探测大语言模型推理中“顿悟”时刻的过程级字谜基准

    AhaPro: A Process-level Rebus benchmark for probing Aha moments in LLM reasoning

    [https://arxiv.org/abs/2611.31268](https://arxiv.org/abs/2611.31268)

    本文提出基于字谜的过程级基准AhaPro来探测大语言模型推理中的“顿悟”时刻，发现模型常常未能采纳顿悟中发现的更优状态，且已采纳的修正甚至可能无益或有害。

    

    现有关于大语言模型“顿悟”时刻的研究大多聚焦于反思性表达或最终修正结果，对其在可见推理过程中所扮演的角色刻画不足。字谜通过字形、空间排列和谐音来编码答案，对关键细节的重新解读可能触发“顿悟”时刻，并引发推理过程与候选答案的转变。我们提出了AhaPro，一个利用字谜来考察模型如何通过“顿悟”时刻重组证据并修正答案的过程级基准。该框架将思维链表示为候选答案、证据与验证状态的序列，并采用分层指标来评估冲突检测、状态改进及后续采纳情况。在十一个模型上的实验中，我们观察到模型经常未能采纳在“顿悟”时刻发现的更优状态，而已采纳的修正有时使状态质量保持不变甚至降低。

    arXiv:2611.31268v1 Announce Type: cross  Abstract: Existing research on Aha moments in large language models has largely focused on reflective expressions or final revision outcomes, leaving their role in the visible reasoning process insufficiently characterized. Rebus puzzles encode answers through character forms, spatial arrangements, and homophony. Reinterpreting key details may trigger Aha moments and shifts in the reasoning process and candidate answers. We introduce AhaPro, a process-level benchmark that uses rebus puzzles to examine how models reorganize evidence and revise answers through Aha moments. The framework represents CoT as sequences of candidate answers, evidence, and verification states, using hierarchical metrics to assess conflict detection, state improvement, and subsequent adoption. Across eleven models, we observe frequent failures to adopt better states discovered during Aha moments, alongside adopted revisions that leave state quality unchanged or reduce it.
    
[^282]: SD-Cache：基于状态引导动态缓存的可学习记忆整合

    SD-Cache: Learnable Memory Consolidation with State-Guided Dynamic Caching

    [https://arxiv.org/abs/2611.31265](https://arxiv.org/abs/2611.31265)

    提出SD-Cache，通过状态引导的动态缓存机制，在不预知未来查询的情况下，将LLM原生的KV状态选择性地整合到固定大小的工作记忆中，解决了RAG场景下KV缓存线性增长的问题。

    

    大型语言模型（LLM）越来越多地被部署为需要对海量外部知识进行推理的持久化服务。检索增强生成（RAG）虽然将模型响应建立在相关文档的基础上，但检索到的内容仍需被编码并存储在键值（KV）缓存中。随着这些内容不断积累，缓存呈线性增长，导致内存消耗和推理成本上升。现有的记忆压缩方法要么通过保留缓存状态的一个子集来解决该问题，要么通过学习将文档映射为紧凑潜在表示的上下文压缩器。尽管取得了这些进展，如何在紧张的内存预算下整合信息、同时保持其对未来查询的有用性，仍然是一个挑战。我们研究了如何在无法访问未来查询的情况下，将模型原生的KV状态选择性地整合到一个固定大小的工作记忆中。为此，我们提出了SD-Cache（状态引导……）

    arXiv:2611.31265v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as persistent services that must reason over large volumes of external knowledge. Retrieval-augmented generation (RAG) grounds model responses in relevant documents, but the retrieved content must still be encoded and stored in the key-value (KV) cache. As this content accumulates, the cache grows linearly, increasing memory consumption and inference cost. Existing memory-compression methods address this problem either by retaining a subset of cached states or by learning context compressors that map documents to compact latent representations. Despite these advances, consolidating information under tight memory budgets while preserving its usefulness for future queries remains challenging. We investigate how the model's native KV states can be selectively consolidated into a fixed-size working memory without access to future queries. To this end, we introduce SD-Cache (State-guide
    
[^283]: 人设条件化如何改变开放式任务执行：行为收益与可靠性成本的配对评估

    How Persona Conditioning Changes Open-Ended Task Execution: A Paired Evaluation of Behavioral Gains and Reliability Costs

    [https://arxiv.org/abs/2611.31264](https://arxiv.org/abs/2611.31264)

    该论文通过配对实验首次系统揭示了人设条件化作为任务引导信号的双面性：它能提升LLM回复的专家内容深度和流程忠实度，但会以17至30个百分点的幅度降低严格指令遵循准确率并引发无依据的细节虚构。

    

    大型语言模型（LLM）经常通过角色或人设类指令进行提示，然而人设条件化此前主要通过角色忠实度或基准测试的总体准确率来评估。我们将人设条件化视为一种任务执行的引导信号，研究它如何改变模型解释、组织和维持回复的方式。基于从真实用户角色扮演日志中提取并扩展为模拟多轮对话延续的配对实例，我们在固定任务内容、约束条件、用户模拟和评估标准的前提下，比较了带人设与不带人设两种条件。在八个模型上的实验表明，人设条件化提升了专家内容深度和流程忠实度，且这些收益在长度控制后依然存在。与此同时，它在所有被评估的模型中将严格指令遵循准确率降低了17至30个百分点，并通过人设诱发的细节扩充增加了缺乏依据的现实世界具体描述。

    arXiv:2611.31264v1 Announce Type: cross  Abstract: Large language models (LLMs) are often prompted with role- or persona-based instructions, yet persona conditioning has mainly been evaluated through character fidelity or aggregate benchmark accuracy. We study persona conditioning as a task-execution steering signal that changes how models interpret, structure, and sustain responses. Using paired instances derived from real user role-play logs and expanded into simulated multi-turn continuations, we compare with-persona and no-persona conditions while fixing task content, constraints, user simulation, and evaluation criteria. Across eight models, persona conditioning improves expert content depth and procedural fidelity, and these gains survive length control. At the same time, it lowers strict instruction-following accuracy by 17 to 30 points in every model evaluated, and increases unsupported real-world specificity through persona-induced elaboration. These results suggest that perso
    
[^284]: 评估大型语言模型在伊朗电气与计算机工程博士入学考试中的表现

    Evaluating Large Language Models on Iranian Doctoral Entrance Exams in Electrical and Computer Engineering

    [https://arxiv.org/abs/2611.31263](https://arxiv.org/abs/2611.31263)

    该研究利用伊朗电气与计算机工程博士入学考试的真实试题（含波斯语与英语双语及依赖配图的题目），通过统一的统计方法系统评估了GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5和Grok 4四个大型语言模型的答题准确率。

    

    我们在2024年和2025年伊朗国家电气工程（EE）与计算机工程（CE）博士入学考试的多项选择题上评估了四个大型语言模型（LLM）。四本官方试题册共有440道题，其中7道因题干或配图存在歧义而被排除，剩余433道题（EE 186道，CE 247道），其中171道（39.5%）依赖配图。每道题均以原始波斯语和英语翻译两种版本呈现，共产生3,464个评分回答。2025年10月，通过自动化脚本经由单一商业网页接口（Perplexity）以零样本方式（开启推理功能）查询了GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5和Grok 4。由于每个模型回答相同的题目，所有统计推断均采用配对检验或按题目聚类的模型。准确率分为两个层级：GPT-5（79.9%，95% CI 76.6–83.1）和Gemini 2.5 Pro（78.4%，74.8–81.9）……

    arXiv:2611.31263v1 Announce Type: cross  Abstract: We evaluate four large language models (LLMs) on multiple-choice questions from the 2024 and 2025 Iranian national Ph.D. entrance examinations in Electrical Engineering (EE) and Computer Engineering (CE). Of the 440 questions in the four official booklets, 7 were excluded because the question statement or its figure was ambiguous, leaving 433 questions (186 EE, 247 CE), of which 171 (39.5\%) depend on a figure. Each question was presented in the original Persian and in an English translation, giving 3,464 scored responses. GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5 and Grok 4 were queried zero-shot, with reasoning enabled, through a single commercial web interface (Perplexity) in October 2025, using an automated script. Because every model answers the same items, all inferences use paired tests or question-clustered models. Accuracy separates into two tiers: GPT-5 (79.9\%, 95\% CI 76.6--83.1) and Gemini 2.5 Pro (78.4\%, 74.8--81.9) are s
    
[^285]: RegioZH：揭示大语言模型对不同地区中文变体理解能力的差异

    RegioZH: Unveiling Comprehension Disparities across Regional Chinese Varieties in LLMs

    [https://arxiv.org/abs/2611.31262](https://arxiv.org/abs/2611.31262)

    该论文提出 RegioZH——首个覆盖中国大陆、台湾、香港、新加坡和马来西亚的 2,607 题地区中文词汇推理基准，揭示了大语言模型在理解不同地区中文变体上的显著差异（如台湾与新加坡之间 29.4 个百分点的准确率差距），并表明部分错误源于检索或表述框架问题而非单纯的知识缺失。

    

    当前的评估往往将中文变体视为单一整体，掩盖了模型对地区用户的理解失败。我们提出了 RegioZH，这是一个包含 2,607 道多项选择题的基准，用于评估覆盖中国大陆、台湾、香港、新加坡和马来西亚的地区性词汇推理能力。该基准由各地区专门的标注者构建并经专家验证，聚焦于本地化习语、俚语、语码混合、礼貌规范以及社会文化表达。在十一个大语言模型上的零样本测试结果显示出巨大差异，其中包括台湾与新加坡之间平均 29.4 个百分点的准确率差距。对五个模型使用角色设定和结构化思维链提示仅带来选择性的提升，表明部分错误源于检索或表述框架问题，而不仅仅是知识缺失。表征、注意力、校准和表面特征分析显示，标记注意力与语义基础之间存在分离、强制选择题中存在过度自信的错误，且本地化词汇密度是……

    arXiv:2611.31262v1 Announce Type: cross  Abstract: Current evaluations often treat Chinese varieties as monolithic, obscuring failures for regional users. We introduce RegioZH, a 2,607-item multiple-choice benchmark for regional lexical reasoning across Mainland China, Taiwan, Hong Kong, Singapore, and Malaysia. Built by region-specific annotators and expert-validated, it targets localized idioms, slang, code-mixing, politeness conventions, and sociocultural expressions. Zero-shot results on eleven LLMs show large disparities, including a 29.4-point average accuracy gap between Taiwan and Singapore. Persona and structured chain-of-thought prompts for five models yield selective gains, suggesting that some errors stem from retrieval or framing rather than only missing knowledge. Representation, attention, calibration, and surface-feature analyses show a dissociation between marker attention and semantic grounding, overconfident forced-choice errors, and localized lexical density as a co
    
[^286]: 项目反应理论能否跨语言泛化？来自跨语言LLM基准测试的证据

    Does Item Response Theory Generalize Across Languages? Evidence from Cross-Lingual LLM Benchmarks

    [https://arxiv.org/abs/2611.31261](https://arxiv.org/abs/2611.31261)

    该研究通过在Multilingual GSM和Global MMLU上对100多个语言模型拟合1PL/2PL/3PL IRT模型，发现题目难度是跨语言最稳定的IRT参数，而猜测和区分度的稳定性依次减弱，因此多语言场景下基于IRT的数据选择需要先验证模型选择并逐一筛查具体题目。

    

    项目反应理论（IRT）在基准测试数据选择中的应用大多局限于单语环境。在本工作中，我们提出这样一个问题：当同一个基准被翻译成不同语言时，IRT如何泛化？我们在Multilingual GSM和Global MMLU上进行了实验，涵盖超过100个语言模型和11种语言。通过为每种语言分别拟合1PL、2PL和3PL IRT模型，我们评估了参数的跨语言一致性，以及经IRT筛选的子集能否在各语言间保持模型排名：结果显示题目难度是最稳定的参数，其次是猜测参数，再其次是区分度参数。差异题目分析还表明，不稳定性的根源可以追溯到特定题目在跨语言翻译过程中发生的偏移。我们的研究结果表明，在多语言环境中基于IRT的数据选择必须同时具备参数感知和题目感知能力，即在假设跨语言一致性成立之前，需要先验证模型选择的适用性并逐一筛查具体题目。

    arXiv:2611.31261v1 Announce Type: cross  Abstract: The application of Item Response Theory (IRT) for data selection in benchmarks has mostly been limited to monolingual settings. In this work, we ask: how does IRT generalize when the same benchmark is translated across languages? We conduct experiments on Multilingual GSM and Global MMLU, covering over 100 language models and 11 languages. Fitting 1PL, 2PL, and 3PL IRT models per language, we assess cross-lingual consistency of parameters and whether IRT-filtered subsets preserve model rankings across languages: item difficulty emerges as the most stable parameter followed by guessing and then discrimination. A differential item analysis also shows that lack of stability can be tracked to specific items shifting under translation across languages. Our findings suggest IRT-based selection in multilingual settings must be both parameter-aware and item-aware, validating model choice and screening individual items before assuming cross-lin
    
[^287]: Transformer在多任务递归上的能力与可解释性

    Capability and Interpretability of Transformers on Multi-task Recurrence

    [https://arxiv.org/abs/2611.31260](https://arxiv.org/abs/2611.31260)

    本文通过有限循环群上的线性递归任务族系统研究了Transformer的能力与可解释性，发现切换明确指定的规则比识别潜在规则更容易，并揭示了模型深度、宽度以及递归的代数结构对泛化能力的影响，同时证明课程学习在规则数量较大时尤为有效。

    

    Transformer被广泛应用于各类语言任务中，但人们对其内部机制的理解仍然有限。受许多语言规则的递归结构启发，我们通过有限循环群上的线性递归任务这一受控任务族，研究了Transformer的能力与可解释性。这些任务涵盖了从执行单个固定规则、从上下文中识别潜在规则，到在明确指定的规则之间进行切换，且包含有输入掩码和无输入掩码两种情形。通过对不同配置的大量实验，我们发现总体而言，对Transformer来说，切换明确指定的规则比识别潜在规则更为简单。我们还识别出影响Transformer泛化能力的若干关键因素，包括深度、宽度以及线性递归的代数结构。当规则数量较多时，课程学习的收益极为显著。借助机制可解释性领域的标准工具……（摘要原文在此处截断）

    arXiv:2611.31260v1 Announce Type: cross  Abstract: Transformers are widely used in various language tasks, with limited understanding of their internal mechanisms. Motivated by the recurrent structure of many linguistic rules, we study the capability and interpretability of Transformers through a controlled family of linear recurrence tasks over finite cyclic groups. These tasks range from executing a single fixed rule to identifying a latent rule from context and switching between explicitly specified rules, with and without input masking. Through extensive experiments across different configurations, we find that switching specified rules is simpler than identifying latent rules for Transformers in general. We also identify how several key factors affect the generalization capability of Transformers, including the depth, width, and the algebraic structure of the linear recurrence. Curriculum training is extremely useful when the number of rules is large. With standard tools in mechan
    
[^288]: RecipeADQA：一个面向多约束食谱推荐的用户感知问答数据集

    RecipeADQA: A User-aware Question-Answering Dataset for Multi-Constraint Recipe Recommendation

    [https://arxiv.org/abs/2611.31259](https://arxiv.org/abs/2611.31259)

    本文提出了RecipeADQA——一个以真实用户行为为驱动、利用大语言模型提取细粒度偏好信号并映射到知识图谱标签模式上的多约束食谱推荐基准问答数据集，同时通过贪心约束选择策略和基于标签的重排序保证数据质量。

    

    面向食谱推荐的问答任务缺乏带有真实标签（ground-truth）的数据集。现有方法通常通过食材或标签约束的简单组合来生成数据集，导致表达不自然、约束不合理，难以反映用户的真实偏好。本文提出了一种用于构建自然语言食谱推荐基准式问答数据集的方法。该方法以真实用户行为为驱动，整合外部用户查询与评论数据，利用大语言模型（LLM）提取细粒度偏好信号，并将其映射到基于知识图谱的标签模式上，从而构建出由偏好约束和食材约束组成的问答数据。此外，该方法还采用贪心策略进行约束选择以控制答案规模，并应用基于标签的重排序来提升数据质量。生成的数据集RecipeADQA包含自然语言问题、结构化约束以及……（摘要在此处被截断）

    arXiv:2611.31259v1 Announce Type: cross  Abstract: Question-answering recipe recommendation tasks lack datasets with ground-truth labels. Existing methods often generate datasets from simple combinations of ingredient or tag constraints, leading to unnatural expressions and unreasonable constraints that are hard to reflect users' real preferences. This paper proposes a construction method for a benchmark-style QA dataset for natural-language recipe recommendation. Driven by real user behavior, our method integrates external user query and review data to extract fine-grained preference signals by LLMs and maps them onto a knowledge-graph-based tag schema, thereby constructing QA data consisting of preference and ingredient constraints. Besides, a greedy strategy for constraint selection is also used to control the answer size, while tag-based re-ranking is applied to improve data quality. The generated dataset RecipeADQA contains natural-language questions, structured constraints, and a
    
[^289]: RECAP：通过上下文压缩训练长时程LLM智能体

    RECAP: Training Long-horizon LLM Agents with Context Compaction

    [https://arxiv.org/abs/2611.31258](https://arxiv.org/abs/2611.31258)

    RECAP提出一种强化学习框架，通过行为失真目标将摘要生成与智能体策略联合训练，使压缩摘要能够保留原始交互历史所诱导的行为，从而提升智能体在长时程任务中的表现。

    

    上下文压缩使大型语言模型智能体能够通过用压缩摘要替换交互历史，在有限的上下文预算内完成长时程任务。然而，将有效压缩与智能体策略进行联合学习仍然具有挑战性，因为摘要的质量会在后续多次交互中影响任务的成功与否。我们提出了RECAP，这是一个基于“摘要应当保留原始交互历史所诱导的行为”这一原则的强化学习（RL）框架。我们通过一个行为失真目标来形式化这一原则，该目标衡量智能体在压缩上下文与原始上下文下后续行为之间的差异。我们证明该失真目标的梯度可以分解为两个互补的学习信号：用于摘要生成的基于失真的奖励，以及用于后续动作的自蒸馏。在Qwen3.5-4B上的实验表明，RECAP持续优于……（原文此处截断）

    arXiv:2611.31258v1 Announce Type: cross  Abstract: Context compaction enables large language model agents to tackle long-horizon tasks within limited context budgets by replacing interaction histories with compressed summaries. However, learning effective compaction jointly with the agent policy remains challenging, as summaries affect task success over many subsequent interactions. We introduce RECAP, a reinforcement learning (RL) framework based on the principle that a summary should preserve the behavior induced by the original interaction history. We formalize this principle with a behavior distortion objective that measures the divergence between the agent's subsequent behavior under compressed and original contexts. We show that the gradient of the distortion objective decomposes into two complementary learning signals: distortion-based reward for summary generation and self-distillation for subsequent actions. Experiments on Qwen3.5-4B show that RECAP consistently outperforms th
    
[^290]: scOPD：面向基于大语言模型的单细胞注释的路由式在策略自蒸馏框架

    scOPD: Routed On-Policy Self-Distillation for LLM-Based Single-Cell Annotation

    [https://arxiv.org/abs/2611.31257](https://arxiv.org/abs/2611.31257)

    scOPD提出了一种面向大语言模型批次级单细胞注释的在策略自蒸馏框架，通过自适应地将合格rollout路由至基于奖励的优化、并利用错误rollout进行自蒸馏，为中间生物推理提供细粒度监督。

    

    动机：大语言模型（LLMs）在单细胞注释方面已展现出潜力，但现有训练方法主要依赖标量结果奖励，这对中间生物推理过程提供的监督十分有限。批次级别的注释尤其具有挑战性，因为它需要同时解决局部细胞身份识别与全局一对一匹配约束，任何一个局部错误都可能影响整个分配结果。此外，单次推理过程无法为定位不确定的预测提供充分证据。这些局限性要求在训练阶段引入细粒度的推理监督，并在推理阶段实现不确定性感知的细化。结果：为应对这些挑战，我们提出了scOPD，一个用于基于大语言模型单细胞注释的在策略自蒸馏框架。在推理微调之后，scOPD自适应地将合格的rollout路由至基于奖励的优化，而错误的rollout则接收先验……（原文摘要在此处被截断）

    arXiv:2611.31257v1 Announce Type: cross  Abstract: Motivation: Large language models (LLMs) have shown promise for single-cell annotation, but existing training methods mainly rely on scalar outcome rewards that provide limited supervision for intermediate biological reasoning. Batch-level annotation is particularly challenging because it requires jointly resolving local cell identities and a global one-to-one matching constraint, such that one local error can affect the entire assignment. Moreover, a single inference pass provides insufficient evidence for locating uncertain predictions. These limitations call for fine-grained reasoning supervision during training and uncertainty-aware refinement during inference.   Results: To address these challenges, we propose scOPD, an on-policy self-distillation framework for LLM-based single-cell annotation. After reasoning fine-tuning, scOPD adaptively routes qualified rollouts to reward-based optimization, while erroneous rollouts receive pri
    
[^291]: 面向投机解码直接最大化在策略期望接受长度

    Directly Maximizing On-policy Expected Accepted Length for Speculative Decoding

    [https://arxiv.org/abs/2611.31256](https://arxiv.org/abs/2611.31256)

    FreDraft通过在策略训练直接最大化草稿的期望接受长度，并利用目标模型词表分布解析边缘化接受率以降低梯度方差，从而更有效地提升投机解码的加速效果。

    

    投机解码通过使用轻量级草稿模型提出候选token、再由目标模型进行验证，从而加速大语言模型推理。现有草稿模型训练通常优化token级分布对齐，但这并不能直接优化被接受前缀的长度：一旦某个草稿token被拒绝，其后所有草稿token都会被丢弃。我们提出FreDraft，一种在策略训练方法，直接最大化从当前草稿模型采样得到的草稿的期望接受长度。该目标考虑了首次拒绝截断，并引入前缀感知的信用分配，根据每个token对当前及后续token接受的贡献进行加权。为了高效优化该目标，我们解析地边缘化token级接受率，利用目标模型的词表分布来降低梯度方差。我们进一步重新分配该分布级梯度的一部分……（原文摘要至此截断）

    arXiv:2611.31256v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model inference by using a lightweight draft model to propose tokens for verification by a target model. Draft training commonly optimizes token-level distribution alignment, but this does not directly optimize the length of the accepted prefix: once a draft token is rejected, all subsequent draft tokens are discarded. We propose FreDraft, an on-policy training method that directly maximizes the expected accepted length of drafts sampled from the current draft model. This objective accounts for first-rejection truncation and induces prefix-aware credit assignment, weighting each token by its contribution to the acceptance of the current and subsequent tokens. To optimize the objective efficiently, we analytically marginalize token-level acceptance rate, using the target model's vocabulary distribution to reduce gradient variance. We further redistribute part of this distribution-level gra
    
[^292]: INCLUDE-2.0：解耦多语言大语言模型评估中的知识鸿沟

    INCLUDE-2.0: Disentangling Knowledge Gaps in Multilingual LLM Evaluation

    [https://arxiv.org/abs/2611.31255](https://arxiv.org/abs/2611.31255)

    INCLUDE-2.0是一个覆盖86个国家91种语言、包含22万余样本的区域感知多语言事实评估基准，通过标注每道题所需的区域知识类型（而非将语言简单等同于区域），实现了对多语言大模型知识鸿沟的精细化受控分析。

    

    当前的多语言基准测试仅覆盖现代预训练语料库中一小部分语言，且常常将语言能力与回答基于特定地区和社区的问题的能力混为一谈。我们推出了INCLUDE-2.0，这是一个用于区域感知多语言事实性评估的大规模基准，涵盖来自86个国家的91种语言（共127个语言-国家组合），包含超过22万个从原生区域来源精心筛选的样本。与现有通常将语言作为区域代理的基准不同，INCLUDE-2.0区分了以下几类问题：答案与上下文无关的问题、因国家而异的问题、涉及某地被广泛记载事实的问题，以及主要依赖当地语言和文化来源表达的知识的问题。每个条目都标注了其区域知识类型和领域，并配有英文翻译，从而支持对模型表现进行受控分析。

    arXiv:2611.31255v1 Announce Type: cross  Abstract: Current multilingual benchmarks cover only a fraction of the languages represented in modern pretraining corpora and often conflate language proficiency with the ability to answer questions grounded in specific regions and communities. We introduce INCLUDE-2.0, a large-scale benchmark for region-aware multilingual factual evaluation, which covers 91 languages from 86 countries (127 total language--country pairs), comprising over 220K curated samples drawn from native regional sources. Unlike existing benchmarks that often treat language as a proxy for region, IINCLUDE-2.0 distinguishes between questions whose answers are independent of context, vary by country, concern widely documented facts about a place, or depend on knowledge primarily expressed through local linguistic and cultural sources. Each item is annotated with its regional knowledge type and domain, and paired with an English translation, enabling controlled analysis of th
    
[^293]: 你未曾设置的日期：系统提示日期作为大语言模型评估中的隐藏变量

    The Date You Didn't Set: The System-Prompt Date as a Hidden Variable in LLM Evaluation

    [https://arxiv.org/abs/2611.31254](https://arxiv.org/abs/2611.31254)

    本文揭示系统提示中自动填入的日期是LLM评估中被忽视的隐藏变量：仅改变日期就会使中位数6.6%的基准题目翻转正误，在对抗性搜索下甚至使模型准确率损失高达36.5分，因此评估时必须固定并报告系统提示日期。

    

    arXiv:2611.31254v1 Announce Type: cross 摘要：许多经过指令微调的模型的系统提示中都带有日期，通常是今天的日期，无论是对话请求还是智能体请求都会自动填入，因此即使评估本身没有变化，每天发送的提示也可能不同。本文探讨了这一隐藏输入会在多大程度上改变评估结果。在六个指令微调模型和四个基准测试上，即使提示、解码和配置全部固定，每个模型都受到影响：两个合理的日期（2024–2026年）会使中位数6.6%的测试项在正确与错误之间翻转。即使仅仅把评估推迟一天再运行，翻转的题目数量也与改变批处理规模相当，而实验设置中没有任何东西被改动。此外，利用语法约束的贪心坐标梯度（GCG）变体搜索任意年份的日期，我们发现某些日期会使Llama-3.x模型在保留集上的准确率损失高达36.5分。除了准确率之外，切换到另一个合理日期还会改变LLM裁判的（判决结果）……

    arXiv:2611.31254v1 Announce Type: cross  Abstract: The system prompt of many instruction-tuned models carries a date, usually today's, filled in automatically for conversational and agentic requests alike, so an unchanged evaluation can send a different prompt each day. In this paper, we explore how much this hidden input changes evaluation outcomes. Across six instruction-tuned models and four benchmarks, with prompts, decoding, and configuration fixed, every model is affected: two plausible dates (2024--2026) switch a median $6.6\%$ of test items between correct and incorrect. Even rerunning the evaluation one day later flips about as many items as changing the batch size, although nothing in the setup was changed. In addition, using a grammar-constrained variant of Greedy Coordinate Gradient (GCG) to search dates of any year, we find dates that cost the Llama-3.x models up to $36.5$ points of held-out accuracy. Beyond accuracy, switching to another plausible date alters LLM-judge ve
    
[^294]: 图像生成中人类价值的具象化

    Instantiation of Human Values in Image Generation

    [https://arxiv.org/abs/2611.31253](https://arxiv.org/abs/2611.31253)

    该研究基于施瓦茨人类价值理论构建分析框架，通过28,000张生成图像揭示四个主流文生图模型如何将成功、关爱、自由等抽象价值狭窄地具象化为重复出现的视觉原型，并将其映射到社会构型分类体系以揭示模型系统性的价值呈现偏向。

    

    文生图（T2I）模型日益融入日常视觉文化，然而人们对其如何将抽象的人类价值转化为视觉场景却知之甚少。为填补这一空白，我们引入了一个基于施瓦茨（Schwartz）基本人类价值理论的框架，用于研究“价值具象化”（value instantiation）：即成功、关爱或自由等理想在视觉上得以实现的过程。我们使用来自70个正向与负向价值条目的以人物为中心的提示词，在四个最先进的文生图模型上生成了28,000张图像。我们识别出视觉原型——即反复出现的主体、动作与场景的组合——并用它们来量化模型具象化这些价值的范围有多狭窄，以及这种广度与人类对相同条目的描述相比有何差异。随后，我们将这些原型映射到一个涵盖场景设定、社交性、参与方式和社会语调的社会构型分类体系上，揭示出哪些内容被模型系统性地突显或忽视。

    arXiv:2611.31253v1 Announce Type: cross  Abstract: Text-to-image (T2I) models are increasingly embedded in everyday visual culture, yet little is known about how they translate abstract human values into visual scenes. We address this gap by introducing a framework, grounded in Schwartz's theory of basic human values, for studying value instantiation: the process by which ideals such as success, care, or freedom are visually realized. Using person-centered prompts from 70 positive and negative value items, we generate 28,000 images across four state-of-the-art T2I models. We identify visual prototypes, recurring combinations of subjects, actions, and settings, and use them to quantify how narrowly models instantiate these values and how this breadth compares with human descriptions of the same items. We then map these prototypes onto a taxonomy of social configurations spanning setting, sociality, mode of engagement, and social register, revealing what is systematically foregrounded or
    
[^295]: DAGGER：面向生成式AI工作流的无部署质量评估

    DAGGER: Deployment-Free Quality Estimation for Generative AI Workflows

    [https://arxiv.org/abs/2611.31252](https://arxiv.org/abs/2611.31252)

    DAGGER框架通过将生成式AI工作流建模为图结构，实现了无需穷举部署与性能分析即可高效准确地估算工作流端到端质量，突破了现有方法在估算成本与准确性之间的权衡。

    

    估算生成式AI工作流的质量极具挑战性，因为这些工作流由多个阶段组成，错误可能在各阶段之间传播，使得端到端性能不仅取决于单个模型的准确性，还取决于各模型的输出对下游阶段的影响。每个阶段都可以从多个候选模型中选择一个。若要通过部署并评估所有可能的模型组合来估算端到端工作流质量，其成本将高得令人望而却步。因此，现有方法面临估算成本与准确性之间的权衡：低成本的方法往往产生不可靠的质量估算，而准确的方法则需要部署和评估大量的工作流配置。本文提出DAGGER，一个无需穷举部署和性能分析即可估算生成式AI工作流质量的框架。DAGGER将工作流表示为一个图，其中节点对应于各个阶段，边……

    arXiv:2611.31252v1 Announce Type: cross  Abstract: Estimating the quality of generative AI workflows is challenging because these workflows comprise multiple stages, and errors can propagate across them, making end-to-end performance depend not only on the accuracy of individual models but also on how their outputs affect downstream stages. Each stage can use one of many candidate models. Estimating end-to-end workflow quality by deploying and evaluating every resulting model configuration is therefore prohibitively expensive. As a result, existing approaches face a tradeoff between estimation cost and accuracy: inexpensive methods often produce unreliable quality estimates, while accurate methods require deploying and evaluating many workflow configurations. In this paper, we introduce DAGGER, a framework for estimating the quality of generative AI workflows without exhaustive deployment and profiling. DAGGER represents a workflow as a graph, in which nodes correspond to stages and ed
    
[^296]: 大语言模型对癌症治疗查询响应中的验证偏见：一项析因横断面研究

    Validation Bias in Large Language Model Responses to Cancer Treatment Queries: A Factorial Cross-Sectional Study

    [https://arxiv.org/abs/2611.31251](https://arxiv.org/abs/2611.31251)

    这项析因横断面研究通过对14个大语言模型、1,272段对话的分析，证实了LLM在应对与证据不一致的癌症治疗信念时存在“验证偏见”——倾向于认同而非纠正用户的错误信念，这对患者安全构成潜在风险。

    

    大语言模型（LLM）聊天机器人被越来越多地用于患者获取癌症信息，但当用户提出与证据不一致的治疗信念时，其安全性尚未被系统表征。我们假设LLM表现出“验证偏见”，即认同而非纠正此类信念，并通过2×2×2析因设计对此进行评估，该设计交叉了三个因素：对话格式（单轮 vs 多轮）、治疗类型（循证治疗 vs 替代治疗）和查询效价（正面 vs 负面）。问题框架设定为证据一致（支持循证治疗或质疑替代疗法）或证据不一致（推广替代治疗或质疑循证疗法）。我们分析了来自6家厂商的14个LLM的1,272段对话（共6,360次交互），由经人工标注验证的LLM评判器（GPT-5-mini）依据6维度安全量表进行评分。主要结局指标为不安全响应率，定义为

    arXiv:2611.31251v1 Announce Type: cross  Abstract: Large language model (LLM) chatbots are increasingly used by patients seeking cancer information, but their safety when users present evidence-misaligned beliefs about treatment remains uncharacterized. We hypothesized that LLMs exhibit validation bias, affirming rather than correcting such beliefs, and evaluated this with a 2x2x2 factorial design crossing conversation format (single-turn vs multi-turn), treatment type (evidence-based vs alternative), and query valence (positive vs negative). Framing was evidence-aligned (supporting evidence-based treatments or questioning alternative therapies) or evidence-misaligned (promoting alternative treatments or questioning evidence-based therapies). We analyzed 1,272 conversations (6,360 interactions) across 14 LLMs from 6 vendors, scored by an LLM judge (GPT-5-mini) validated against human annotation on a 6-dimension safety rubric. The primary outcome was the unsafe response rate, defined as
    
[^297]: Actionwave：面向多种医疗场景的以波形为中心的临床预测系统性基准

    Actionwave: A Systematic Benchmark for Waveform-Centric Clinical Prediction Across Care Scenarios

    [https://arxiv.org/abs/2611.31249](https://arxiv.org/abs/2611.31249)

    Actionwave 是首个覆盖重症监护、手术室和急诊科等多种医疗场景的以波形为中心的临床预测系统性基准，通过统一的决策时点预测接口、严格的评估协议，系统考察了信号时长、预警时间和可用输入等条件对模型表现的影响。

    

    连续的床旁波形记录了快速的生理变化，可为临床决策提供信息。现有的基准测试仅考察了选定的预测条件，但目前尚不清楚模型优势如何依赖于信号时长、预警时间以及不同医疗场景下的可用输入。我们提出了 Actionwave，这是一个面向重症监护、手术室和急诊科场景的以波形为中心的临床预测基准。Actionwave 采用统一的决策时点预测接口，并根据推断的内容和时间对具有临床依据的任务进行分组。它改变了与临床使用相关的条件，包括干预前所需的预警时间。我们在指定的模型规模和计算预算下，比较了多种架构和预训练生物信号编码器。该协议要求患者级数据划分、严格的信息截断，以及能够隔离单个建模因素的比较。在 20 个任务和 140 万个任务标签……

    arXiv:2611.31249v1 Announce Type: cross  Abstract: Continuous bedside waveforms record rapid physiological changes that can inform clinical decisions. Existing benchmarks examine selected prediction conditions, but it remains unclear how model advantages depend on signal duration, warning time, and available inputs across care settings. We introduce Actionwave, a benchmark for waveform-centric clinical prediction across intensive-care, operating-room, and emergency-department settings. Actionwave uses a common decision-time prediction interface and groups clinically grounded tasks by what is inferred and when. It varies conditions relevant to clinical use, including the warning required before an intervention. We compare diverse architectures and pretrained biosignal encoders under specified model-size and compute budgets. The protocol requires patient-level splits, strict information cutoffs, and comparisons that isolate individual modeling factors. Across 20 tasks and 1.4M task-label
    
[^298]: InletScope：基于多尺度压力的变化传感器配置下进气道不起动监测

    InletScope: Multiscale pressure-based inlet unstart monitoring under changing sensor configurations

    [https://arxiv.org/abs/2611.31246](https://arxiv.org/abs/2611.31246)

    InletScope 提出了一种基于多尺度压力的进气道不起动监测框架，通过共享多尺度编码、可学习尺度融合与传感器概率平均，在传感器配置发生变化时无需重新训练即可实现不起动检测和0.5秒内的新发事件预测，并在马赫2.9风洞试验中取得了0.834–0.964的窗口F1分数。

    

    超音速进气道为航空发动机捕获并压缩空气，不起动（unstart）现象会中断空气供应并危及推进系统的稳定性。基于压力的监测必须适应随运行工况和测量位置变化的流动响应。InletScope 结合了共享的多尺度压力编码、可学习的尺度融合以及传感器概率平均，使得测量集合无需重新训练即可发生变化。采用独立的模型分别评估不起动检测与未来0.5秒间隔内的新发事件预测。七个马赫2.9的试验各包含一个标注事件，涵盖下游节流、喉道调节和分流器调整；其中三个试验作为留出测试集，使用固定权重和源自动训练数据的设置。在三个训练随机种子下，全传感器检测的按运行平均窗口 F1 分数达到 0.834–0.964。八传感器聚合在两次分流器调整运行中降低了跨传感器子集的检测分数波动，同时保持（摘要在此处截断）。

    arXiv:2611.31246v1 Announce Type: cross  Abstract: Supersonic inlets capture and compress air for aeroengines, and unstart can disrupt the air supply and compromise propulsion-system stability. Pressure-based monitoring must accommodate flow responses that vary with operating condition and measurement location. InletScope combines shared multiscale pressure encoding, learned scale fusion and sensor-probability averaging, allowing measurement sets to change without retraining. Separate models evaluate unstart detection and new-onset prediction within a 0.5-s future interval. Seven Mach-2.9 runs, each with one annotated event, cover downstream throttling, throat regulation and splitter adjustment; three are held out with fixed weights and source-derived settings. Across three training seeds, full-sensor detection achieves run-wise mean window $F_1$ scores of 0.834--0.964. Eight-sensor aggregation reduces detection-score variation across sensor subsets in both splitter runs while retainin
    
[^299]: 通过瞬时轨迹分布匹配强化扩散模型

    Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching

    [https://arxiv.org/abs/2611.30986](https://arxiv.org/abs/2611.30986)

    该论文揭示现有前向扩散强化学习方法因行为分布与策略失配导致基线引起的梯度漂移和训练不稳定，并提出瞬时轨迹分布匹配（iTDM）来解决这一问题，从而高效对齐扩散模型。

    

    前向扩散强化学习（RL）——无需轨迹反向传播即可实现确定性ODE采样——在高效对齐扩散模型方面日益流行。然而，我们揭示出现有方法本质上依赖于带符号的奖励加权回归（RWR），即将可训练策略的速度回归到来自行为分布的采样轨迹上。无偏基线减除隐含地假设策略速度等于行为边际速度——这是一个理想化的假设，在无分类器引导（CFG）采样、离线/离策略更新以及不完美的在线自一致性情况下会被违反。这种行为-策略失配破坏了基线不变性，并注入了由基线引起的梯度漂移，导致训练不稳定。为解决这一问题，我们提出了瞬时轨迹分布匹配（iTDM）……

    arXiv:2611.30986v1 Announce Type: cross  Abstract: Forward diffusion reinforcement learning (RL)---enabling deterministic ODE sampling without trajectory backpropagation---has grown increasingly popular for efficiently aligning diffusion models. However, we reveal that existing methods essentially rely on signed reward-weighted regression (RWR), regressing the \emph{trainable policy} velocity onto rollouts from a \emph{behavior distribution}. Unbiased baseline subtraction implicitly assumes that the policy velocity equals the behavior marginal velocity---an idealized assumption violated under classifier-free guidance (CFG) rollouts, offline/off-policy updates, and imperfect online self-consistency. This \emph{behavior--policy mismatch} breaks baseline invariance and injects \emph{baseline-induced gradient drift}, destabilizing training. To resolve this issue, we propose \textbf{I}nstantaneous \textbf{T}rajectory \textbf{D}istribution \textbf{M}atching (\textbf{iTDM}), taking the infini
    
[^300]: 从部分配对的小规模数据中学习多模态表示

    Learning Multimodal Representations from Partially Paired, Small-Scale Data

    [https://arxiv.org/abs/2611.30601](https://arxiv.org/abs/2611.30601)

    该论文提出CAMEO对抗学习融合框架，将预训练单模态编码器与跨模态潜在对齐机制结合，仅需极少配对样本即可从部分配对的小规模数据中学习多模态共享表示，在计算病理学任务上以卓越数据效率超越CLIP等对比方法。

    

    从部分配对样本中建模多模态数据对于推进生物医学等领域至关重要，因为这些领域虽然拥有大量单模态数据集和基础模型，但配对数据仍然稀缺。现有的融合方法依赖于大规模配对数据集，限制了它们在不完全配对场景中的应用。我们提出了CAMEO，一种基于对抗学习的模态融合框架，能够整合来自小型部分配对数据集的模态信息。CAMEO将任意预训练的单模态编码器与跨模态潜在对齐机制相结合，只需极少量的配对样本即可学习共享表示。在计算病理学任务（如微环境生态位分类和细胞类型组成预测）上的评估表明，CAMEO展现出卓越的数据效率，在低配对数据场景下超越了CLIP等对比学习方法，凸显了在配对标注稀缺时对抗对齐方法的优势。

    arXiv:2611.30601v1 Announce Type: new  Abstract: Modeling multimodal data from partially paired samples is critical for advancing domains like biomedicine, where vast unimodal datasets and foundation models exist but paired data remains scarce. Existing fusion methods rely on large-scale paired datasets, limiting their use in scenarios with incomplete pairing. We introduce CAMEO, an adversarial learning-based modality fusion framework that integrates modalities from small, partially paired datasets. Combining any pre-trained unimodal encoder with a cross-modal latent alignment mechanism, CAMEO learns shared representations requiring only minimal paired samples. Evaluated on computational pathology tasks such as niche classification and cell type composition prediction, CAMEO achieves superior data efficiency, outperforming contrastive approaches like CLIP in low paired-data regimes, and highlighting the benefits of adversarial alignment when paired annotations are scarce. To facilitate
    
[^301]: RoboInF：一个用于衡量机器人操作中指令忠实度的基准测试

    RoboInF: A Benchmark for Instruction Fidelity in Robotic Manipulation

    [https://arxiv.org/abs/2611.30555](https://arxiv.org/abs/2611.30555)

    该论文提出RoboInF基准，通过在固定初始场景与机器人状态下变换指令的目标、流程或细节，并利用任务专属检查器结合轨迹、接触和事件序列评估全执行过程的合规性，从而专门衡量操作策略对指令变化的忠实遵循程度。

    

    arXiv:2611.30555v1 公告类型：cross 摘要：精确的语言控制要求具身模型能够同时遵循约束任务结果和执行过程的指令。尽管近期的方法训练策略以遵循日益详细的命令，但现有基准通常将指令遵循与感知、规划以及对陌生场景的泛化能力混在一起评估。因此，更高的任务成功率只能提供有限的证据来证明策略对指令的忠实遵循程度。我们提出 RoboInF，这是一个衡量操作策略如何可靠地根据指令变化调整自身行为的基准。针对八个任务中的每一个，设置六到十二个经过专家验证的指令分支，在保持初始场景和机器人状态不变的情况下，改变所要求的目标、流程或操作细节。特定于任务的检查器会验证最终目标，并利用记录的轨迹、接触和事件序列来评估整个执行过程中的合规性。随附的 RoboInF-4K 数据集包含 4,242 条专家演示（原文摘要在此处截断）。

    arXiv:2611.30555v1 Announce Type: cross  Abstract: Precise language control requires embodied models to follow instructions governing both task outcomes and execution. Although recent methods train policies to follow increasingly detailed commands, existing benchmarks often assess instruction following alongside perception, planning, and generalization to unfamiliar scenes. Higher task success therefore offers limited evidence of closer adherence to instructions. We introduce RoboInF, a benchmark measuring how reliably manipulation policies adapt their behavior to changes in an instruction. For each of eight tasks, six to twelve expert-validated instruction branches vary the requested goal, procedure, or manipulation detail while holding the initial scene and robot state fixed. Task-specific checkers verify the terminal goal and use recorded trajectories, contacts, and event sequences to assess compliance throughout execution. The accompanying RoboInF-4K dataset pairs 4,242 expert demo
    
[^302]: 具有靶向嗅觉受体激活模式的气味分子从头生成

    De Novo Generation of Odorant Molecules with Targeted Olfactory Receptor Activation Patterns

    [https://arxiv.org/abs/2611.30509](https://arxiv.org/abs/2611.30509)

    该论文提出了一个基于强化学习的气味分子从头生成框架，通过构建嗅觉受体全景结合矩阵、聚类发现代表性激活模式，并采用结合成对排序、分布匹配与复杂度约束的结构化选择性损失，成功生成匹配目标多受体激活谱的分子，在广谱型和稀疏激活型等罕见目标谱上取得了最大提升。

    

    气味感知源于一种组合编码机制：每个分子会激活特定的嗅觉受体（OR）子集，由此产生的激活模式决定了所感知到的气味。我们提出了一个从头生成分子的框架，用于生成预测上匹配目标多受体激活谱的分子。我们首先构建了覆盖20个嗅觉受体的全景结合矩阵，然后对其进行聚类以发现代表性的激活模式，最后设计了一种结构化选择性损失函数，结合成对排序、分布匹配和复杂度约束，通过强化学习来引导分子生成器。在九个不同复杂度的目标谱上，我们的方法在18000次oracle调用内实现了较高的预测选择性，并且在先验分布下较为罕见的目标谱上获得了最大的提升，尤其是广谱型和稀疏激活型目标。

    arXiv:2611.30509v1 Announce Type: cross  Abstract: Odor perception arises from a combinatorial code in which each molecule activates a specific subset of olfactory receptors (ORs), and the resulting activation pattern determines the perceived smell. We present a framework for de novo generation of molecules predicted to match target multi-receptor activation profiles. We first construct a panoramic binding matrix over 20 ORs, then cluster it to discover representative activation patterns, and finally formulate a structured selectivity loss, combining pairwise ranking, distribution matching, and complexity constraints, to steer a molecular generator via reinforcement learning. Across nine target profiles of varying complexity, our method achieves high predicted selectivity within 18000 oracle calls and provides the largest gains on profiles that are rare under the prior, especially broad-spectrum and sparse-activator targets.
    
[^303]: mini-psych-agent：通过交互式心理环境评估大语言模型智能体的类人行为

    mini-psych-agent: Evaluating Human-Like Behavior in LLM Agents through Interactive Psychological Environments

    [https://arxiv.org/abs/2611.30365](https://arxiv.org/abs/2611.30365)

    该论文提出mini-psych-agent，一个交互式心理环境测试平台，旨在评估大语言模型智能体在不同智能体工作流程和测试时计算预算下行为表现的类人程度。

    

    大语言模型（LLM）、其运行框架以及测试时计算预算共同决定了其行为表现。因此，同一个底层模型嵌入不同的智能体工作流程中，或被分配不同的测试时计算资源时，可能会表现出截然不同的性能。心理学家在评估人类表现时面临类似的挑战：同一个人在不同的激励条件和思考程度下可能会有不同的行为。因此，心理学任务为研究这种行为变异提供了理想的测试平台。然而，当这些任务被改编用于LLM时，它们通常被转录为文本形式，即以静态提示词的方式回放固定的、以语言呈现的实验。这种格式并不适合智能体评估，因为在智能体评估中，动作会改变未来的观察结果和任务状态。相反，尽管动态环境在LLM智能体评估中很普遍，但它们很少关注由此产生的行为是否具有类人性。

    arXiv:2611.30365v1 Announce Type: new  Abstract: A Large Language Model (LLM), its harness, and the test-time compute budget jointly determine its behavior. Consequently, the same underlying model, embedded in different agentic workflows or allocated different amounts of test-time compute, can exhibit markedly different performance. Psychologists face an analogous challenge when evaluating human performance: the same person may behave differently under different incentives and levels of deliberation. Psychological tasks therefore provide an ideal testbed for studying such behavioral variation. However, when adapted to LLMs, these tasks are typically text-transcribed, replaying fixed, language-rendered experiments as static prompts. This format is ill-suited to agent evaluation, where actions can alter future observations and task states. Conversely, although dynamic environments are prevalent in the evaluation of LLM agents, they rarely ask whether the resulting behavior is human-like.
    
[^304]: PixelOne：单次前向传播实现从选项到像素的GUI定位

    PixelOne: From Options to Pixels in a Single Forward for GUI Grounding

    [https://arxiv.org/abs/2611.30235](https://arxiv.org/abs/2611.30235)

    PixelOne通过将搜索教师模型的渐进式空间知识蒸馏给学生模型，使学生能在原生分辨率下于单次前向传播中直接从截图像素预测出精准的点击坐标。

    

    GUI定位要求在低延迟下实现精准点击。近期的模型（如GUI-Decisions和GUI-Swift）使用固定的全局分箱加局部细化的方式，在一次前向传播中预测坐标。然而，截图下采样会丢失定位小型控件所需的细节，而在更高分辨率下，固定分箱会覆盖更大的像素范围。我们提出PixelOne，它将空间知识从搜索教师模型迁移到学生模型，使学生模型能够直接以原生分辨率从截图中预测点击。教师模型利用VideoJEV的文本选项评分器来选择行和列，并逐步放大候选区域。学生模型继承教师的参数，并通过空间蒸馏结合边界框监督来匹配教师在已探索区域上的分布。在推理阶段，同一个决策头选择一个视觉token，随后一个轻量级的像素头在该token内部预测点击位置，全部在一次前向传播中完成。点击对齐损失鼓励……（原文摘要在此处截断）

    arXiv:2611.30235v1 Announce Type: cross  Abstract: GUI grounding requires precise clicks at low latency. Recent models such as GUI-Decisions and GUI-Swift predict coordinates in one forward pass using fixed global bins with local refinement. However, screenshot downsampling removes detail needed to locate small controls, while fixed bins cover larger pixel ranges at higher resolutions. We introduce PixelOne, which transfers spatial knowledge from a search teacher to a student that predicts clicks directly from screenshots at native resolution. The teacher uses VideoJEV's text-option scorer to select rows and columns and progressively zoom into candidate regions. The student inherits the teacher's parameters and matches its distributions over explored regions through spatial distillation alongside box supervision. At inference, the same decision head selects a visual token, and a lightweight pixel head predicts the click within it, all in one forward pass. A click-aligned loss encourage
    
[^305]: 从差异蛋白质组学到可解释AI规则：肺腺癌的SHAP-FOL-ILP建模

    Differential Proteomics to XAI Rules: A SHAP-FOL-ILP Modeling for Lung Adenocarcinoma

    [https://arxiv.org/abs/2611.29989](https://arxiv.org/abs/2611.29989)

    该研究构建了一个融合配对差异蛋白质组学、稀疏逻辑回归、SHAP归因与一阶逻辑归纳编程（FOL/ILP）的可解释建模流程，能从肺腺癌蛋白质组数据中自动提炼出可解释的分类规则。

    

    背景：我们在肺腺癌（LUAD）研究中开发了一个可解释的工作流程，将配对差异蛋白质组学、稀疏逻辑回归、SHAP归因分析与一阶逻辑（FOL）规则学习相结合。方法：我们分析了来自111名患者的213个CPTAC-LUAD组织样本（111个肿瘤组织、102个癌旁正常组织），其中包括102个配对样本。蛋白质效应估计与检验均基于相同的完整配对，并采用Benjamini-Hochberg校正。稀疏逻辑回归采用五折分层且按患者分组的嵌套交叉验证。预处理和模型选择仅限于训练数据内进行，特征选择在内层折叠中重复执行。另通过独立的五折按患者分组的评估，利用训练阶段衍生的证据和离散化方法，测试了SHAP引导的归纳逻辑编程（FOL/ILP）。结果：质量控制后保留了9,518种蛋白质。在FDR低于0.05且平均配对对数比差异绝对值不低于1的条件下，（摘要在此处截断）

    arXiv:2611.29989v1 Announce Type: new  Abstract: Background: We developed an interpretable workflow linking paired differential proteomics, sparse logistic regression, SHAP attribution, and first-order logic (FOL) rule learning in lung adenocarcinoma (LUAD).   Methods: We analysed 213 CPTAC-LUAD tissue samples (111 tumour, 102 adjacent-normal) from 111 patients, including 102 matched pairs. Protein effect estimation and testing used the same complete pairs, with Benjamini-Hochberg correction. Sparse logistic regression used five-fold stratified patient-grouped nested cross-validation. Preprocessing and model selection were confined to training data, with feature selection repeated within inner folds. A separate five-fold patient-grouped evaluation tested SHAP-guided inductive logic programming (FOL/ILP) using training-derived evidence and discretisation.   Results: Quality control retained 9,518 proteins. At FDR below 0.05 and an absolute mean paired log-ratio difference of at least 1,
    
[^306]: 看见与知晓：面向3D感知物体编辑的反事实监督与校正

    Seeing vs. Knowing: Counterfactual Supervision and Correction for 3D-Aware Object Editing

    [https://arxiv.org/abs/2611.29542](https://arxiv.org/abs/2611.29542)

    提出反事实状态监督与对称反事实校正方法，教会3D感知物体编辑器在编辑时遵循3D状态生成不可见内容、同时保留照片可见外观，并确保这一区分在采样过程中得以维持。

    

    对照片中的物体进行旋转、平移或缩放操作，需要揭示不可见的表面、保留可见的外观，并更新阴影与反射。将重建的3D资产在目标姿态下渲染为3D状态，可以提供显式的几何指导，但其不完美的纹理可能与照片产生冲突，而标准的成对训练并未显式地教导编辑器应信任哪个来源。为解决这一问题，我们提出了反事实状态监督，通过隐藏表面重绘、带重新渲染阴影与反射的隐藏几何、以及可见表面纹理替换来构建可控的照片-状态冲突。这教会编辑器对不可见内容遵循3D状态，同时保留照片中可见的外观。然而，学习这种区分并不能确保其在采样过程中得以保持。因此，我们进一步引入了对称反事实校正，它利用...

    arXiv:2611.29542v1 Announce Type: cross  Abstract: Rotating, translating, or rescaling an object in a photograph requires revealing unseen surfaces, preserving visible appearance, and updating shadows and reflections. Reconstructed 3D assets, rendered at the target pose as a 3D state, provide explicit geometric guidance, but their imperfect textures may conflict with the photograph, while standard paired training does not explicitly teach the editor which source to trust. To address this, we propose counterfactual state supervision, which constructs controlled photo--state conflicts through hidden-surface repainting, hidden geometry with re-rendered shadows and reflections, and visible-surface texture replacement. This teaches the editor to follow the 3D state for unseen content while preserving photograph-visible appearance. However, learning this distinction does not ensure that it is maintained during sampling. We therefore introduce symmetric counterfactual correction, which uses s
    
[^307]: 重连控制何时能测试LLM权重读取器？C2与资格协议

    When Can a Rewiring Control Test LLM Weight Readers? C2 and an Eligibility Protocol

    [https://arxiv.org/abs/2611.29120](https://arxiv.org/abs/2611.29120)

    论文提出C2反事实方法（置换网络最终线性读出层的列）及三条件资格协议，用于区分大语言模型是真正在“计算”权重还是仅识别权重空间“指纹”，但测试发现没有任何自然设置符合该协议资格。

    

    大型语言模型可以从原始权重预测小型神经网络的输出，但这种成功可能反映的是对权重空间规律（即“指纹”）的识别，而非真正的计算，而保持函数不变的对照组无法区分这两者。我们提出了C2，一种反事实方法，它置换网络最终线性读出层的列：所有其他参数，以及读出层的奇异值、范数和元素多重集均保持不变，而函数可以以完全可重新计算的方式发生变化；我们验证了每个被评估的置换确实如此。C2仅在三个条件下才能区分指纹识别与计算，我们将这些条件表述为一个资格协议：一种仅限于C2不变摘要的策略必须已经能够预测目标；C2必须改变特定于重连的属性；并且读取器的身份必须经过认证。在我们测试的所有自然设置中，没有一个符合资格。在小型MLP上，前沿读取器路径跟随了重连后的（摘要在此处截断）

    arXiv:2611.29120v1 Announce Type: cross  Abstract: Large language models can predict the outputs of small neural networks from raw weights, but success could reflect recognition of weight-space regularities (fingerprints) rather than computation, and function-preserving controls cannot tell these apart. We propose C2, a counterfactual that permutes the columns of a network's final linear readout: every other parameter, and the readout's singular values, norms and entry multiset, are unchanged, and the function can change in an exactly recomputable way; we verify that each evaluated permutation does. C2 separates fingerprinting from computation only under three conditions, which we state as an eligibility protocol: a strategy restricted to C2-invariant summaries must already predict the target; C2 must change a property specific to the rewiring; and the reader's identity must be attested. No natural setting we tested is eligible. On tiny MLPs a frontier reader route followed the rewired
    
[^308]: 改进改进者：面向医疗智能体自我改进的语义PID控制

    Improving the Improver: Semantic PID Control for Medical Agent Self-Improvement

    [https://arxiv.org/abs/2611.28680](https://arxiv.org/abs/2611.28680)

    该论文提出Semantic PID-RSI框架，借鉴PID控制理论，通过证据引导的三个诊断角色来改进医疗智能体的自我改进过程本身，解决了现有递归自我改进方法在高噪声医疗领域中的反馈歧义与开发集过拟合问题。

    

    高风险的垂直医疗任务，如术后病理分期和长期生存预测，需要在弱监督和高结果不确定性的条件下，对异质性证据进行多步骤临床推理。我们首先构建了CliAgent，一个领域定制的临床智能体工作流，能够系统地将预测建立在多模态患者证据之上。然而，手动设计的医疗智能体仍然受限于静态、脆弱的规则。此外，现有的递归自我改进（RSI）方法在高噪声医疗领域中演化智能体时，存在反馈歧义和严重的开发集过拟合问题。着眼于改进改进者本身，我们提出了Semantic PID-RSI，一个证据引导的、基于控制论的医疗智能体自我改进框架。受PID控制启发，我们的改进器通过三个诊断角色评估候选修改批次：P隔离当前的临床失败模式，I综合……（原文摘要在此处截断）

    arXiv:2611.28680v1 Announce Type: new  Abstract: High-stakes vertical medical tasks, such as postoperative pathological staging and long-term survival prediction, demand multi-step clinical reasoning over heterogeneous evidence under weak supervision and high outcome uncertainty. While we first orchestrate CliAgent, a domain-tailored clinical agent workflow that systematically grounds predictions in multi-modal patient evidence, manually designed medical agents remain constrained by static, brittle rules. Moreover, existing recursive self-improvement (RSI) methods suffer from feedback ambiguity and severe development-set overfitting when evolving agents in high-noise medical domains. Focusing on improving the improver, we propose Semantic PID-RSI, an evidence-guided control-theoretic framework for medical agent self-improvement. Inspired by PID control, our improver evaluates candidate modification batches via three diagnostic roles: P isolates current clinical failure modes, I synthes
    
[^309]: EvoViM：面向延迟查询探索的递归自我进化视觉智能体

    EvoViM: Recursively Self-Evolving Visual Agents for Exploration with Deferred Queries

    [https://arxiv.org/abs/2611.28574](https://arxiv.org/abs/2611.28574)

    EvoViM提出一种递归自我进化框架，让冻结的多模态模型将交互经验转化为可直接在像素上计算的视觉程序，并通过自我追踪失败、修正证据处理操作来应对查询延迟的探索任务。

    

    多模态大语言模型（MLLM）智能体在任务导向环境中已展现出令人期待的能力。然而，当查询被延迟到探索结束之后才给出时，它们仍然表现不佳——这要求智能体在尚不知道哪些细节会变得重要之前，就预先保留视觉证据。我们提出了EvoViM，一个递归自我进化的框架，其中冻结的多模态模型将自身的交互经验转化为用于记忆和技能的视觉程序。该程序保留图像并直接在像素上进行计算，其操作决定了这些证据在每个任务中如何被使用。在记忆进化过程中，同一模型会追踪其视觉轨迹中的失败，定位到丢失或误用证据的具体操作并加以修正。它还会重写指导后续修订的笔记和分析工具，从而使后续修改建立在已有发现之上，并跳过已经失败过的更改。我们在涵盖六项任务的基准上，使用三个多模态模型对EvoViM进行了评估。

    arXiv:2611.28574v1 Announce Type: new  Abstract: Multimodal large language model (MLLM) agents have shown promising capabilities in task-oriented environments. However, they still struggle when queries are deferred until after exploration, requiring them to retain visual evidence before knowing which details will matter. We introduce EvoViM, a recursively self-evolving framework in which a frozen multimodal model turns its own interaction experience into one visual program for memory and skills. The program retains images and computes directly on pixels, and its operations decide how this evidence is used in each task. During memory evolution, the same model traces failures in its visual trajectories to the operations that lost or misused evidence and revises them. It also rewrites the notes and analysis tools that guide later revisions, which then build on earlier findings and skip changes that have already failed. We evaluate EvoViM with three multimodal models on six tasks covering 
    
[^310]: Grammar-as-a-Draft（语法即草稿）：在语法约束的结构化输出中让每次推测解码步产生更多Token

    Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs

    [https://arxiv.org/abs/2611.28349](https://arxiv.org/abs/2611.28349)

    该论文提出Grammar-as-a-Draft（GaaD），将语法引擎作为第二草稿模型，把语法强制Token与推测草稿模型的提议合并为更宽的草稿行，使语法约束的结构化输出下每次推测解码步可验证更多Token，且无需新增内核或编译。

    

    在结构化输出（JSON模式、正则表达式、工具调用）场景下，推测式草稿模型无法感知语法约束，会反复提议无效的键名、分隔符和缩进，而语法引擎其实早已知道接下来必须出现的Token。我们提出了Grammar-as-a-Draft（GaaD），它将语法视为第二个草稿模型。在每次验证步骤之前，调度器将草稿模型的提议与语法强制Token合并为一行草稿：匹配的Token保留为GRAMMAR状态，在提议偏离强制路径之处将其截断，并由强制Token填满比草稿提议更宽的行的剩余部分。验证步骤无需计算GRAMMAR草稿的logits即可接受它们，可与任何草稿模型组合，且在关键路径上无需新增内核或编译。由于语法固定的是文本而非Token，提交规范的强制Token相对于使用语法掩码的目标解码会改变输出分布。

    arXiv:2611.28349v1 Announce Type: cross  Abstract: Under structured outputs (JSON schemas, regexes, tool calls), speculative drafters do not see the grammar constraint and repeatedly propose invalid keys, delimiters, and indentation, while the grammar engine already knows which tokens must come next.   We present Grammar-as-a-Draft (GaaD), which treats the grammar as a second drafter. Before each verify step, the scheduler merges the drafter's proposal with grammar-forced tokens into one draft row: matching tokens stay as GRAMMAR, the proposal is cut where it leaves the forced path, and forced tokens fill the rest of a row wider than the drafter's proposal. The verify step accepts GRAMMAR drafts without computing their logits, composing with any drafter with no new kernels or compilation on the critical path. Because grammars fix text rather than tokens, committing canonical forced tokens changes the output distribution relative to grammar-masked target decoding.   We implement GaaD in
    
[^311]: 大语言模型的社会谄媚性在不同访问模式下存在差异

    Social sycophancy differs across access modes of large language models

    [https://arxiv.org/abs/2611.28012](https://arxiv.org/abs/2611.28012)

    该研究首次系统比较了主流大语言模型（ChatGPT、Claude、Gemini、DeepSeek）通过 API 与用户界面（UI）两种访问方式在社会谄媚性上的差异，发现 UI 访问普遍导致更强的谄媚行为（比值比高达 7.56），且这种差异在不同模型间存在显著不同。

    

    大语言模型（LLM）可能表现出谄媚性，即过度赞扬、附和或肯定用户的倾向。现有研究主要通过应用程序编程接口（API）来测试大语言模型的谄媚性，而非公众通常与之交互的消费级用户界面（UI）。访问模式（API 与 UI）可能会影响观察到的谄媚率。然而，主流模型在社会谄媚性方面表现出多大的 API-UI 差异仍不得而知，社会谄媚性包括在 UI 中可能更常出现的行为（例如，缓和本应给出的批评）。我们通过 API 和 UI 两种访问方式，在四个社会谄媚性维度上测试了 ChatGPT、Claude、Gemini 和 DeepSeek。在所有模型中，UI 访问普遍与更强的社会谄媚性相关，比值比高达 7.56（95% CI [6.22, 9.20]）。然而，在所有谄媚性维度上，UI-API 差异在不同模型之间存在显著变化。

    arXiv:2611.28012v1 Announce Type: cross  Abstract: Large language models (LLMs) can display sycophancy, a tendency to excessively praise, agree with, or validate a user. Existing studies have tested LLMs for sycophancy primarily through application programming interfaces (APIs) rather than the consumer user interfaces (UIs) through which the public commonly interacts with LLMs. Access mode (API vs. UI) may affect observed sycophancy rates. However, it remains unknown to what extent prominent models show API-UI differences in social sycophancy, which includes behaviors potentially more likely in UIs (e.g., softening warranted criticism). We tested ChatGPT, Claude, Gemini, and DeepSeek through both APIs and UIs on four dimensions of social sycophancy. Across models, UI access was generally associated with greater social sycophancy, with odds ratios up to 7.56 (95% CI [6.22, 9.20]). However, UI-API differences varied significantly across models for all sycophancy dimensions, including a p
    
[^312]: 基于异步事件的自监督连续时间深度学习与自运动估计

    Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events

    [https://arxiv.org/abs/2611.27509](https://arxiv.org/abs/2611.27509)

    该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。

    

    事件相机以微秒级精度逐像素地记录亮度变化，而可用于监督它们的稠密深度数据获取成本高昂，且仅存在于孤立的时刻。将事件流压缩成帧虽然便于学习，但代价是牺牲了事件感知赖以独特的高时间分辨率特性。我们提出了一种自监督方法，其中每个事件在连续时间中被赋予其自身的深度，同时相机运动也从相同的观测中恢复出来。监督信号源于几何：一个事件通过其预测深度被提升到三维空间，并经过它在触发瞬间的相机运动位移后，应当与由同一条边缘产生的其他事件相重合，因此任何错位都会暴露深度或运动估计中的误差。随后，可微渲染将这些逐事件的估计转化为每个窗口内目标时刻的稠密深度图，并忠实于其背后的事件。在MVSEC数据集上，该方法优于先前仅基于事件的自监督方法。

    arXiv:2611.27509v1 Announce Type: cross  Abstract: Event cameras register brightness changes pixel by pixel with microsecond precision, whereas the dense depth that could supervise them is costly to acquire and exists only at isolated instants. Condensing the stream into frames eases learning, but at the expense of the very timing that sets event sensing apart. We propose a self-supervised approach in which each event is assigned its own depth in continuous time, and the camera motion is recovered from the same observations. Supervision arises from geometry: lifted into 3D by its predicted depth and displaced by the camera motion at the instant it fired, an event should coincide with others produced by the same edge, so any misalignment exposes an error in depth or motion. Differentiable rendering then turns these per-event estimates into dense depth maps at target times within each window, faithful to the events behind them. On MVSEC, the approach outperforms prior event-only self-sup
    
[^313]: VMSA：用于长视频理解的视频记忆技能智能体

    VMSA: Video Memory Skill Agent for Long Video Understanding

    [https://arxiv.org/abs/2611.27381](https://arxiv.org/abs/2611.27381)

    提出VMSA框架，通过联合设计三种互补的结构化视频记忆（片段字幕记忆、实体事件记忆、实体感知情节记忆）与记忆技能，提升智能体在长视频问答中跨长时间跨度检索与整合证据的能力。

    

    长视频问答需要从跨越数小时甚至数天的视频中检索并整合相关证据。近期的智能体方法将视频信息组织到外部记忆中，并迭代地检索证据来回答问题。然而，有效的检索与推理仍然具有挑战性，因为智能体必须确定要搜索什么信息、访问哪些记忆源，以及如何连接跨越遥远时间间隔的证据。为了应对这些挑战，我们提出了视频记忆技能智能体（VMSA），这是一个为长视频理解联合设计结构化视频记忆与记忆技能的框架。VMSA 构建了三个互补的记忆组件：片段字幕记忆和实体事件记忆作为可搜索的证据来源，而实体感知情节记忆则提供广泛而详细的视频上下文，以指导证据检索与推理。为了进一步支持有效的记忆利用……

    arXiv:2611.27381v1 Announce Type: cross  Abstract: Long-video question answering requires retrieving and integrating relevant evidence from videos spanning hours or even days. Recent agentic approaches organize video information into external memories and iteratively retrieve evidence to answer questions. However, effective retrieval and reasoning remain challenging, as agents must determine what information to search for, which memory sources to access, and how to connect evidence across distant temporal intervals. To address these challenges, we propose Video Memory Skill Agent (VMSA), a framework that jointly designs structured video memories and memory skills for long-video understanding. VMSA constructs three complementary memory components. Segment Caption Memory and Entity Event Memory serve as searchable evidence sources, while Entity-Aware Episode Memory provides broad yet detailed video context to guide evidence retrieval and reasoning. To further support effective memory uti
    
[^314]: 从行动到学习：面向持续学习智能体的经验选择

    From Acting to Learning: Experience Selection toward Continually Learning Agents

    [https://arxiv.org/abs/2611.27159](https://arxiv.org/abs/2611.27159)

    该论文将经验选择作为持续学习智能体的关键组成部分，通过大规模对照实验证明，优化器感知的梯度选择方法在学生智能体任务成功率上显著优于全局随机选择和高 NLL 选择两种基线。

    

    成功的智能体交互能够提供训练数据，但仅凭成功本身并不能决定学生（模型）应当从哪些记录中学习。我们将经验选择作为持续智能体学习的一个组成部分，针对一个 4B 参数的语言智能体展开研究。在两个固定经验池、八个初始化种子和两个采样种子的条件下，比较了九种选择器，共训练出 288 个学生模型。每个学生模型均通过统一的执行器在 128 个开发任务组上进行评估。在全局选择自由度相同、包含 256 个不同内容、共 768 次训练呈现的设置下，优化器感知的梯度选择方法在两个经验池中分别取得了 81.45% 和 73.19% 的成功率；相比全局随机选择，其提升分别为 8.25 和 5.96 个百分点；相比全局高 NLL 选择，其提升分别为 4.10 和 4.05 个百分点。在两个经验池中，全部八组初始化层面的对比均优于两个基线。放宽覆盖约束可以进一步提升梯度选择的效果，但会降低随机选择的表现（原文摘要在此处截断）。

    arXiv:2611.27159v1 Announce Type: new  Abstract: Successful agent interactions provide training data, but success alone does not determine which records a student should learn from. We examine experience selection for a 4B language agent as a component of continual agent learning. Nine selectors are compared across two fixed experience pools, eight initialization seeds, and two sampling seeds, yielding 288 trained students. Each student is evaluated on 128 development task groups with a common executor. With equal global selection freedom, 256 distinct contents, and 768 training presentations, optimizer-aware gradient selection reaches 81.45\% and 73.19\% success in the two pools. Its gains over global random selection are 8.25 and 5.96 percentage points; gains over global high-NLL selection are 4.10 and 4.05 points. All eight initialization-level contrasts are positive against both baselines in each pool. Relaxing coverage constraints improves gradient selection but reduces random-sel
    
[^315]: MixFold：基于输出通道分组的移动NPU亚4比特混合精度大语言模型推理

    MixFold: Sub-4-Bit Mixed-Precision LLM Inference via Output-Channel Grouping on Mobile NPUs

    [https://arxiv.org/abs/2611.26976](https://arxiv.org/abs/2611.26976)

    MixFold通过将输出通道分组与移动NPU矩阵单元瓦片对齐，无需解码时扩展权重即可原生执行3比特和对称2比特混合精度大语言模型推理，在骁龙Hexagon NPU上实现比llama.cpp相当或更高的准确率与更快的解码速度。

    

    亚4比特量化使移动设备能够运行更大的大语言模型，但其节省的内存并不总能转化为移动NPU上更快的解码速度。移动NPU的矩阵单元仅支持少数几种原生整数数据类型，而在向量单元上转换不支持的权重格式可能会抵消比特数减少带来的收益。混合精度还面临一个额外的约束：每个矩阵单元瓦片只能使用一种数据类型。MixFold能够在解码过程中不将权重扩展为更宽数据类型的前提下，使用原生整数操作执行3比特和完全对称的2比特权重。MixFold随后将数据类型分配给与矩阵单元瓦片对齐的输出通道组，从而实现比层级别更细粒度的分配，且不会干扰矩阵单元的执行。我们在骁龙8 Elite Gen 5的高通Hexagon NPU上，将MixFold与llama.cpp的Hexagon后端进行比较。在Qwen3模型系列上，MixFold取得了与llama.cpp亚4比特格式相当或更高的准确率。

    arXiv:2611.26976v1 Announce Type: new  Abstract: Sub-4-bit quantization allows mobile devices to run larger LLMs, but its memory savings do not always translate into faster decoding on mobile NPUs. Their matrix units support only a few native integer datatypes, and converting unsupported weight formats on the vector unit can erase the gains from fewer bits. Mixed precision faces an additional constraint: each matrix-unit tile uses one datatype. MixFold executes 3-bit and perfectly symmetric 2-bit weights using native integer operations without expanding the weights to wider datatypes during decoding. MixFold then assigns datatypes to output-channel groups aligned with the matrix-unit tiles, enabling finer-than-layer assignment without disrupting matrix-unit execution. We compare MixFold with llama.cpp Hexagon backend on the Qualcomm Hexagon NPU of a Snapdragon 8 Elite Gen 5. Across the Qwen3 model family, MixFold achieves comparable or higher accuracy than llama.cpp's sub-4-bit formats
    
[^316]: 当安全理据产生幻觉时：多模态安全防护器的诊断与接地

    When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards

    [https://arxiv.org/abs/2611.25793](https://arxiv.org/abs/2611.25793)

    本文首次系统揭示了多模态安全防护中的“理据幻觉”现象——幻觉普遍存在且不随分类准确率或模型规模提升而减少，并据此提出工具对比接地方法来诊断与缓解这一问题。

    

    多模态安全防护器越来越多地在给出安全预测的同时生成理据，然而这些理据是否可靠在很大程度上仍未被探索。我们从两个视角系统地研究了“理据幻觉”：视觉理据幻觉，即理据中包含缺乏支撑的视觉事实；以及安全归因幻觉，即有据可依的内容被赋予了错误的安全解读。在多种理据生成方法、多个视觉-语言骨干模型和安全数据集上的实验表明，理据幻觉普遍存在，且并不会随着分类准确率的提高或模型规模的增大而一致地减少。对“以图思考”范式的进一步分析显示，幻觉集中在细粒度的属性与关系上，而视觉工具往往被低频调用或缺乏足够的空间聚焦。基于这些发现，我们提出了工具对比接地方法（Tool-Contrastive Grounding Re…，原文摘要在此处截断）

    arXiv:2611.25793v1 Announce Type: new  Abstract: Multimodal safety guards increasingly generate rationales alongside safety predictions, yet whether these rationales are reliable remains largely unexplored. We systematically study *rationale hallucination* from two perspectives: visual rationale hallucination, where rationales contain unsupported visual facts, and safety attribution hallucination, where grounded content is assigned an incorrect safety interpretation. Across rationale-generation methods, multiple vision-language backbones, and safety datasets, we find that rationale hallucination is widespread and does not consistently diminish with higher classification accuracy or increased model scale. Further analysis of the Think-with-Image paradigm shows that hallucination is concentrated in fine-grained attributes and relations, while visual tools are often invoked infrequently or without sufficient spatial focus. Based on these findings, we propose *Tool-Contrastive Grounding Re
    
[^317]: Bradley-Terry训练可能扭曲原本准确的语言世界模型的结果概率

    Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models

    [https://arxiv.org/abs/2611.25594](https://arxiv.org/abs/2611.25594)

    本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。

    

    arXiv:2611.25594v1 公告类型：交叉 摘要：比较反馈（即由评判者指出两个候选结果中哪一个更接近实际发生的情况）是训练为动作结果分配概率的语言世界模型的一种实用方法。我们证明，使用Bradley-Terry目标函数进行拟合时，它未必能教会模型真实的结果概率，甚至可能扭曲原本准确的结果概率。在一个具有枚举结果分布的模拟机器人推动任务中，该目标函数会使一个在未见布局上表现准确的模型（危害误差为2.4个百分点）将33至43个百分点的概率转移到该条件下没有任何枚举分支会产生的结果上。问题的根源在于目标函数本身，而非比较数据：在训练前计算的目标函数最优解可以预测模型最终的去向，而一个了解评判者规则的似然函数能够从相同的比较数据中学习且不产生这种偏移。这种偏移具有鲁棒性：它在未见条件、五种候选法则以及三倍更密集的枚举下均会重现。

    arXiv:2611.25594v1 Announce Type: cross  Abstract: Comparative feedback, in which a judge says which of two candidate outcomes is closer to what happened, is a practical way to train language world models that assign probabilities to an action's outcomes. We show that, fitted with a Bradley-Terry objective, it need not teach the true outcome probabilities and can distort accurate ones. On a simulated robot push with enumerated outcome distributions, this objective takes a model that is accurate on unseen layouts (hazard error 2.4 percentage points, pp) and moves 33 to 43 pp of its probability onto outcomes that none of a condition's enumerated branches produces. The cause is the objective, not the comparisons: its optimum, computed before training, predicts where the model ends up, and a likelihood that knows the judge's rule learns from the same comparisons without the shift. The shift is robust: it recurs on unseen conditions, under five candidate laws and a three-fold denser enumera
    
[^318]: 基于语言模型的个性化旅行规划器

    LM-based Personalized Travel Planner

    [https://arxiv.org/abs/2611.25531](https://arxiv.org/abs/2611.25531)

    提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。

    

    有效的旅行规划需要处理多个信息源、管理约束条件，并根据用户偏好定制推荐。虽然大语言模型（LLM）带来了动态的、上下文感知的决策能力，但单智能体模型在复杂的多步推理、约束遵循和模块化执行方面往往存在困难。为了克服这些局限性，我们提出了一个层次化多智能体框架，其中一个监督型聊天机器人智能体负责协调专门的智能体，分别用于行程验证、用户查询构建、实时数据检索和基于约束的行程生成。通过集成LLM推理、外部API以及基于确定性多维评分框架的迭代优化机制，我们的方法确保了适应性、高效性和个性化。在著名的TravelPlanner基准上的评估表明，预算约束规划性能提升了10%，在更广泛的约束集规划方面也有所提升……

    arXiv:2611.25531v1 Announce Type: new  Abstract: Effective travel planning requires handling multiple information sources, managing constraints, and tailoring recommendations to user preferences. While LLMs have introduced dynamic, context-aware decision-making, single-agent models often struggle with complex multi-step reasoning, constraint adherence, and modular execution. To overcome these limitations, we propose a hierarchical multi-agent framework where a supervising chatbot agent coordinates specialized agents for schedule validation, user query formulation, real-time data retrieval, and constraint-aware itinerary generation. By integrating LLM reasoning, external APIs, and iterative refinement mechanisms based on a deterministic, multi-dimensional scoring framework, our approach ensures adaptability, efficiency, and personalization. Evaluations on the well-known TravelPlanner benchmark demonstrate improvements in Budget Constraint planning by 10\%, wider constraints set-based pl
    
[^319]: MADJEPA：用于医学异常检测的增强型多阶段联合嵌入预测架构

    MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection

    [https://arxiv.org/abs/2611.25194](https://arxiv.org/abs/2611.25194)

    该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。

    

    医学图像中的异常检测是一项复杂的任务，需要仅使用健康患者的数据来开发一个鲁棒的模型。在无法获得标注数据的情况下，现代方法通常采用自监督学习来获得正常性的模型。由于世界模型有望为其环境构建强大的内部表征，我们认为它们是医学异常检测环境（即对健康患者的医学检查）的合适候选。为此，我们提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构（JEPA）的医学异常检测器。我们设计了一个多阶段训练流程来构建鲁棒的检测器：（1）基于图像的JEPA（I-JEPA）训练旨在学习正常训练数据的强表征；（2）目标编码器蒸馏旨在调节I-JEPA表征对异常数据的过度泛化；（3）伪异常分类旨在……（原文在此处截断）

    arXiv:2611.25194v1 Announce Type: cross  Abstract: Anomaly detection in medical images is a complex task, requiring the development of a robust model using only healthy patient data. Without access to labeled data, modern approaches typically employ self-supervised learning to obtain a model of normality. Since world models promise to build strong internal representations of their environment, we consider them as suitable candidates for the medical anomaly detection environment, namely medical investigations of healthy patients. To this end, we propose MADJEPA, a medical anomaly detector based on an enhanced multi-stage joint-embedding predictive architecture (JEPA). We design a multi-stage training pipeline to produce a robust detector: (1) image-based JEPA (I-JEPA) training aims to learn a strong representation of normal training data; (2) target encoder distillation aims to modulate over-generalization of I-JEPA representations to abnormal data; (3) pseudo-anomaly classification aim
    
[^320]: Hypnos：面向边缘端可扩展持续学习的双阶段端到端平台

    Hypnos: A Dual-Phase End-to-End Platform for Scalable Continual Learning at the Edge

    [https://arxiv.org/abs/2611.24637](https://arxiv.org/abs/2611.24637)

    提出了受大脑清醒-睡眠周期启发的开源双阶段平台Hypnos，在清醒阶段利用搭载RISC-V微控制器的低功耗芯片实现高效推理与无梯度原型更新，从而在资源受限的边缘设备上实现可扩展的持续学习。

    

    持续学习（CL）使神经网络能够通过缓解灾难性遗忘来适应非平稳数据流，为构建稳健、保护隐私且可自适应的边缘设备提供了一条路径。然而，将持续学习移植到真实世界的边缘设备上仍面临两大挑战：（i）在线运行，即传感器数据一次只到达一个样本，设备必须即时适应；（ii）边缘设备的低资源占用特性，使得传统的误差反向传播训练变得难以实现。受大脑清醒与睡眠周期的启发——前者以最小资源实现快速适应，后者负责长期巩固——我们提出了Hypnos，一个面向边缘端高效持续学习的开源双阶段平台。在清醒阶段（如日常运行），Hypnos依靠搭载RISC-V微控制器单元（MCU）的低功耗片上系统，实现高效推理并支持轻量级的无梯度原型更新。在睡眠阶段……

    arXiv:2611.24637v1 Announce Type: cross  Abstract: Continual learning (CL) enables neural networks to adapt to non-stationary data streams by mitigating catastrophic forgetting, offering a path to robust, private, and adaptable edge devices. Yet, porting CL to real-world edge devices remains challenging for two reasons: (i) online operation, where sensor data arrives one sample at a time and the device must adapt on-the-fly, and (ii) the low footprint of edge devices, which makes traditional error backpropagation training prohibitive. Inspired by the brain's wake and sleep cycles, the former for rapid adaptation with minimal resources and the latter for long-term consolidation, we introduce Hypnos, an open-source dual-phase platform for efficient CL at the edge. In the wake phase (e.g., daily operation), Hypnos relies on a low-footprint system-on-chip with a RISC-V microcontroller unit (MCU) targeting efficient inference and supporting lightweight gradient-free prototype updates. In th
    
[^321]: 随着学生模型持续展开生成，视觉后门不断增长

    As Student Rollouts Continue, Visual Backdoors Keep Growing

    [https://arxiv.org/abs/2611.22446](https://arxiv.org/abs/2611.22446)

    本文挑战了“在线策略蒸馏（OPD）对后门传播具有天然抵抗力”的直觉，通过多模型家族与数据集上的实验及理论分析，揭示了视觉后门会随着学生模型在线展开生成而持续传播增长的隐蔽风险。

    

    传统知识蒸馏可能促使后门从教师模型传播到学生模型。然而，这种风险在在线策略蒸馏中如何表现，目前研究尚不充分。OPD 近来已成为视觉-语言模型（VLM）的一种重要后训练范式。与传统离线蒸馏不同，OPD 不需要教师生成的响应，而是利用教师的概率分布来监督学生模型在线生成的响应轨迹。由于初始学生模型通常生成良性响应，且不会从恶意目标响应中获得显式监督，OPD 似乎对后门传播具有某种固有的抵抗力。本文挑战了这一直觉，揭示了 OPD 中视觉后门传播的一种更隐蔽的风险。通过在多个模型家族和数据集上的实验，并结合理论分析……

    arXiv:2611.22446v1 Announce Type: cross  Abstract: Conventional knowledge distillation can facilitate the propagation of backdoors from teacher models to student models. However, how this risk manifests in On-Policy Distillation (OPD) remains insufficiently studied. OPD has recently emerged as an important post-training paradigm for Vision-Language Models (VLMs). Unlike traditional offline distillation, OPD does not require teacher-generated responses. Instead, it uses the teacher's probability distributions to supervise response trajectories generated online by the student. Since the initial student model typically generates benign responses and receives no explicit supervision from malicious target responses, OPD appears to offer some inherent resistance to backdoor propagation.   In this paper, we challenge this intuition and reveal a more subtle risk of visual backdoor propagation in OPD. Through experiments across multiple model families and datasets, together with theoretical ana
    
[^322]: RegimeCouncil：面向LLM智能体记忆冲突治理的成对证据路由方法

    RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents

    [https://arxiv.org/abs/2611.22113](https://arxiv.org/abs/2611.22113)

    该论文提出RegimeCouncil——一个无需重训LLM的轻量级成对证据路由器，将“记忆冲突治理”形式化，并在局部与全局两种记忆抑制策略之间在线仲裁，以解决长期记忆悄然过时与延迟反馈下负面经验传播范围难以确定的问题。

    

    语言智能体依赖长期记忆来在长程交互中实现个性化响应，然而随着世界的变化，存储的事实会悄然过时，而智能体往往只有在基于过时记录采取行动之后，才通过延迟反馈得知使用记忆带来了损害。在记忆库被更新之前，这种负面反馈应该传播多远：是仅在针对同一事实的未来查询中抑制记忆使用（局部策略），还是也应跨及其他存储事实（全局策略）？现有记忆系统在设计阶段就隐式地固定了这两种共享范围之一。我们将这种在有效性隐藏、反馈延迟条件下的在线范围选择问题形式化为“记忆冲突治理”，并证明任何一种固定策略单独使用都不足以应对。随后我们提出RegimeCouncil，一个轻量级的成对证据路由器，它将两种范围保留为“顾问”，并在无需重新训练LLM的情况下在线地在两者之间进行仲裁。（原文摘要至此处截断）

    arXiv:2611.22113v1 Announce Type: new  Abstract: Language agents rely on long-term memory to personalize responses across extended interactions, yet stored facts quietly go stale as the world changes, and an agent often learns that using memory hurt only through delayed feedback after acting on an outdated record. Before the memory store is updated, how far should that negative feedback propagate: should it suppress memory use only on future queries for that same fact (a local policy), or also across other stored facts (a global policy)? Existing memory systems implicitly fix one of these two sharing scopes at design time. We formulate this online scope choice under hidden validity and delayed feedback as memory conflict governance, and prove that neither fixed policy is sufficient on its own. We then propose RegimeCouncil, a lightweight paired-evidence router that retains both scopes as advisors and arbitrates between them online without retraining the LLM. Using revealed comparisons 
    
[^323]: 在策略蒸馏作为后门传播通道

    On-policy distillation as a Backdoor Propagation Channel

    [https://arxiv.org/abs/2611.19207](https://arxiv.org/abs/2611.19207)

    本研究首次揭示在策略蒸馏可成为后门传播的高效通道：教师模型中被植入的后门仅需极少触发提示暴露即可转移至学生模型，且转移强度显著高于传统蒸馏方法。

    

    在策略蒸馏正在成为大语言模型后训练的重要范式，其通过利用教师模型在线生成的监督信号来改进学生模型。然而，这一过程是否也会将教师模型中的潜在安全风险转移给学生模型，目前尚缺乏充分研究。在这项工作中，我们揭示了一种新的安全风险：攻击者首先通过监督微调（SFT）在教师模型中植入条件后门，随后在策略蒸馏仅需极少比例的触发提示暴露，即可将后门转移到学生模型中。更令人担忧的是，我们发现这种后门转移在在策略蒸馏下明显强于传统蒸馏方法，这表明在策略蒸馏可能为跨模型传播后门提供了一条高效通道。我们的机制分析表明，后门信号主要通过……（原文摘要至此截断）

    arXiv:2611.19207v1 Announce Type: cross  Abstract: On-policy distillation (OPD) is emerging as an important paradigm for the post training of large language models, improving student models by leveraging supervision generated online by teacher models. However, whether this process also transfers latent security risks from teacher models to student models remains insufficiently studied. In this work, we reveal a new security risk in which an attacker first implants a conditional backdoor into the teacher model through supervised fine tuning (SFT), after which OPD can transfer the backdoor to the student model with only an extremely small proportion of trigger prompt exposure. More concerningly, we find that such backdoor transfer is substantially stronger under OPD than under conventional distillation methods, suggesting that OPD may provide an efficient channel for propagating backdoors across models. Our mechanistic analysis shows that the backdoor signal is primarily transmitted thro
    
[^324]: 可编辑视频上下文：面向上下文管理视频大语言模型的上下文中KV缓存复用

    Editable Video Context: Mid-Context KV-Cache Reuse for Context-Managing Video LLMs

    [https://arxiv.org/abs/2611.18718](https://arxiv.org/abs/2611.18718)

    该论文提出面向视频大语言模型的可编辑上下文方法，通过M-RoPE时间、高度、宽度分量共享的单个偏移量旋转键值以实现上下文中段的KV缓存复用，并结合内容哈希缓存避免重复视觉编码计算，从而显著降低长视频管理的预填充token数量与重建延迟。

    

    上下文语言模型让模型能够编辑自己的上下文，而后缀缓存复用通过调整存活token的缓存键位置，使文本编辑变得高效。我们将这些思想扩展到具有多模态旋转位置嵌入的视频语言模型。我们将长视频表示为带时间戳的片段。在查看稀疏采样版本后，模型选择部分片段以4倍帧率进行解码，其余片段则从其上下文中移除。我们的重建方法通过使用Qwen2.5-VL的M-RoPE中时间、高度和宽度分量共享的单个偏移量来旋转键，从而复用存活片段的KV缓存。内容哈希缓存还避免了为这些片段重复进行视觉编码器计算。在使用Qwen2.5-VL-7B对公有领域电影进行合成“视频大海捞针”问题的初步实验中，该方法减少了预填充token数量和端到端重建延迟。

    arXiv:2611.18718v1 Announce Type: cross  Abstract: Context Language Models let a model edit its own context, while Suffix Cache Reuse makes edits to text efficient by adjusting the positions of cached keys for surviving tokens. We extend these ideas to video language models with multimodal rotary position embeddings (M-RoPE). We represent a long video as timestamped segments. After viewing a sparsely sampled version, the model selects segments for decoding at 4x the frame rate, and the remaining segments are removed from its context. Our rebuild method reuses the KV cache of surviving segments by rotating their keys with a single offset shared by the temporal, height, and width components of Qwen2.5-VL's M-RoPE. A content-hashed cache also avoids repeating vision-encoder computation for those segments. In preliminary experiments with Qwen2.5-VL-7B on synthetic needle-in-a-video-haystack questions over public-domain films, the method reduces prefill tokens and end-to-end rebuild latency
    
[^325]: 成人词汇扩展与人造词汇学习中的视觉相似性

    Visual Similarity in Adult Word Extension and Artificial Word Learning

    [https://arxiv.org/abs/2611.14116](https://arxiv.org/abs/2611.14116)

    本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。

    

    当两个物体共享同一个名称时，它们的视觉相似性能在多大程度上解释这一现象，教学示例又能带来什么额外作用？首先，我们重新分析了一项大型成人词汇扩展研究的试验级数据（Lewis & Frank, 2018），在冻结的自监督视觉编码器（DINOv2）的空间中计算每个测试图像与所展示示例之间的距离。距离与词汇扩展显著相关（每标准差对数几率为-3.23；共520名参与者），一项探索性事后分析表明，在控制了分类层级和教学条件后，这种关联在已教授的领域内依然存在。示例的组成同样重要：在学习三个示例后，当示例为下位类别时，成人仅将词汇扩展到上位类别项目的1.1%，而当示例为上位类别时，这一比例达到77.5%。其次，在使用同一编码器的人工学习者实验中，当一个词连接两个类别时，类别之间距离越远，学习准确度越低（84.7%……

    arXiv:2611.14116v1 Announce Type: cross  Abstract: When two objects share a name, how much does their visual similarity explain, and what do the teaching examples add? First, we reanalyzed trial-level data from a large adult word-extension study (Lewis & Frank, 2018), computing each test image's distance to the examples shown in the space of a frozen self-supervised visual encoder (DINOv2). Distance was associated with extension (log odds -3.23 per standard deviation; 520 people), and an exploratory post hoc analysis suggested that this association persisted within the taught domain after controlling for taxonomic level and teaching condition. Example composition also mattered: after three examples, adults extended a word to 1.1% of superordinate items when the examples were subordinate and to 77.5% when they were superordinate. Second, in an artificial learner using the same encoder, words joining two categories were learned less accurately the farther apart the categories were (84.7%
    
[^326]: 从攻击面到防御架构：面向智能体编程工具的组合式安全

    From Attack Surface to Defense Architecture: Compositional Security for Agentic Coding Harnesses

    [https://arxiv.org/abs/2611.13960](https://arxiv.org/abs/2611.13960)

    该论文提出ActiveSecHarness——一种基于“智能与权限分离”原则的神经符号参考监控器架构，通过HMAC能力令牌与二维污染格为AI编程智能体提供结构性的组合式安全防御。

    

    AI编程工具将语言模型与shell、文件系统和网络套接字耦合在一起，使该工具本身成为一个自动化的“就地取材”式攻击平台：一旦对手将指令注入到检索到的内容中，bash、curl和find等双用途二进制程序就会成为攻击载荷。我们主张，智能体系统的安全性是认知架构与权限组合的结构性属性，而非单个模型的涌现属性。静态正则表达式黑名单无法检测结构性规避手段，而基于LLM-as-a-Judge的防护栏不仅延迟高，还会过度拦截良性任务。我们提出了ActiveSecHarness，一个围绕“智能与权限分离”原则构建的神经符号参考监控器架构。它结合了：(1) 跨越四个信任与权限环的、受Macaroon启发的HMAC能力令牌；(2) 一个二维乘积污染格 L = L_trust × L_sens…（原文摘要在此处截断）

    arXiv:2611.13960v1 Announce Type: cross  Abstract: AI coding harnesses couple language models to shells, filesystems, and network sockets, turning the harness into an automated Living-off-the-Land (LotL) platform: once an adversary injects instructions into retrieved content, dual-use binaries such as bash, curl, and find become the payload. We argue that security in agentic systems is a structural property of the cognitive architecture and permission composition rather than an emergent property of individual models. Static regex blacklists miss structural evasions, whereas LLM-as-a-Judge guardrails incur high latency while over-blocking benign work.   We present ActiveSecHarness, a neurosymbolic reference-monitor architecture built around Separation of Intelligence and Authority. It combines (1) Macaroon-inspired HMAC capability tokens across four Trust and Authority Rings, (2) a 2D product taint lattice $\mathcal{L} = \mathcal{L}_{\mathrm{trust}} \times \mathcal{L}_{\mathrm{sens}}$ a
    
[^327]: 推测式自蒸馏实现高效的知识内化

    Speculative Self-Distillation enables Efficient Knowledge Internalization

    [https://arxiv.org/abs/2611.12016](https://arxiv.org/abs/2611.12016)

    提出推测式自蒸馏（SSD）——一种逐 token 混合策略方法：默认由学生生成 rollout，教师仅在下一个 token 分布显著分歧处介入，既贴近学生的测试时行为分布又避免低信号状态，从而实现更高效的知识内化。

    

    许多现实世界的应用要求语言模型能够内化预训练中缺失的知识、纳入持续更新的事实，或学习新的领域技能。自蒸馏已成为完成这一任务的一种自然方法：以源材料为条件的模型充当教师，而无条件的学生模型则被训练以“闭卷”方式复现教师的行为。我们提出了推测式自蒸馏（Speculative Self-Distillation，SSD），这是一种高效的逐 token 混合策略自蒸馏方法。默认情况下由学生模型生成每个 rollout，而教师模型仅在其下一个 token 分布与学生模型出现显著分歧的位置进行干预。与使用教师生成前缀的离策略蒸馏不同，这种设计使训练 rollout 保持在接近学生测试时行为的分布上，同时避免了完全在策略 rollout 容易陷入的低信号状态。实际上，SSD 让教师……

    arXiv:2611.12016v2 Announce Type: replace-cross  Abstract: Many real-world applications require language models to internalize knowledge that was absent from pretraining, to incorporate continually updated facts, or to learn new domain skills. Self-distillation has emerged as a natural approach for this task. A model conditioned on the source material acts as the teacher, and an unconditioned student is trained to reproduce the teacher's behavior closed-book. We propose Speculative Self-Distillation (SSD), an efficient per-token mixed-policy method for self-distillation. The student generates each rollout by default, while the teacher intervenes only at positions where its next-token distribution substantially diverges from the student's. Unlike off-policy distillation with teacher-generated prefixes, this design keeps training rollouts close to the student's test-time behavior, while avoiding the low-signal states that fully on-policy rollouts drift into. In effect, SSD lets the teach
    
[^328]: RoadState：一个面向工业级车道级地图更新的领域先验增强视觉语言系统

    RoadState: A Domain-Prior-Enhanced Vision-Language System for Industrial-Grade Lane-Level Map Updating

    [https://arxiv.org/abs/2611.10746](https://arxiv.org/abs/2611.10746)

    该论文提出了已在百度地图生产环境中部署的视觉语言系统RoadState，其核心模型RoadState-VLM将道路场景理解建模为自回归状态估计，把连续的道路观测转换为结构化、时间一致且可审计的车道级语义，实现工业级车道级地图的可靠更新。

    

    车道级地图为导航、智能驾驶和智能交通系统提供了基础性的结构化道路信息。从结构性线索和上下文线索中理解道路场景语义，对于可靠的车道级地图更新至关重要，但在实际生产中仍面临诸多挑战，例如遮挡、模糊的车道标线、临时障碍物以及视角变化；同时，下游工作流要求输出具备时间一致性、物理有效性且可审计，而非自由格式的描述。我们提出了RoadState，这是一个已实际部署于生产环境的视觉语言系统，能够将连续的道路场景观测转换为结构化的车道级语义，服务于百度地图的地图更新流水线。其核心模型RoadState-VLM将道路场景理解建模为自回归状态估计，基于先前的观测、先前的符号状态以及当前观测来预测当前的结构化道路状态。

    arXiv:2611.10746v2 Announce Type: replace-cross  Abstract: Lane-level maps provide foundational structured road information for navigation, intelligent driving, and intelligent transportation systems. Understanding road-scene semantics from structural and contextual cues is essential for reliable lane-level map updating, yet remains challenging in production due to occlusions, ambiguous lane markings, temporary obstacles, and viewpoint changes, while downstream workflows require temporally consistent, physically valid, and auditable outputs rather than free-form descriptions. We present RoadState, a production-deployed vision-language system that transforms sequential road-scene observations into structured lane-level semantics for Baidu Maps' map-updating pipeline. Its core model, RoadState-VLM, formulates road-scene understanding as autoregressive state estimation, predicting the current structured road state from the previous observation, previous symbolic state, and current observa
    
[^329]: 论智能体式操作策略

    On Agentic Manipulation Policies

    [https://arxiv.org/abs/2611.09892](https://arxiv.org/abs/2611.09892)

    提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。

    

    近年来机器人操作领域的研究主要聚焦于通过演示数据训练的专用策略。我们证明，通用型AI模型能够比专用策略（如视觉-语言-动作模型（VLA）和世界-动作模型（WAM））更可靠地控制机械臂。我们系统地分析了智能体式操作策略（AMP），即由通用型AI模型直接指挥机械臂，并随着任务的展开自适应地调整其动作。我们提出了AMP1，一个可在单臂、双臂和移动操作中无需任何改动即可运行的参考框架。AMP1在RoboLab-120和CaP-Gym基准测试中取得了最先进的结果，且无需任何演示数据或任务特定训练。在RoboLab-120上，AMP1零样本成功率达到85.6%，而领先的VLA和WAM策略分别仅为49.2%和42.9%。该智能体在涵盖十二个任务和三台机器人的108次真实世界测试中零样本成功了104次，显著降低了失败率。

    arXiv:2611.09892v2 Announce Type: replace-cross  Abstract: Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6% success rate zero-shot, compared to 49.2% and 42.9% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rate
    
[^330]: 大规模AI辅助A/B实验的经验教训与陷阱

    Learnings and Pitfalls from AI-Assisted A/B Experimentation at Scale

    [https://arxiv.org/abs/2611.04515](https://arxiv.org/abs/2611.04515)

    该论文总结了构建和运营大规模AI辅助A/B实验智能体的经验与陷阱，核心结论是通过限制LLM非确定性的作用范围（模型负责选择与编排流程，确定性代码负责计算）来化解LLM不可靠性与实验系统可信度之间的矛盾。

    

    A/B测试通过提供可信、可复现的量化方式来评估产品变更的效果，从而支撑产品决策。大语言模型（LLM）能力的快速提升推动了实验工作流自动化的尝试。然而，LLM具有非确定性和易产生幻觉的特性，这与实验系统所必须提供的可信度相冲突。此外，实验在操作层面也十分繁琐：诊断异常、调试配置以及开展定制化的后续分析都需要灵活的推理和大量人工投入。这些任务恰好可以让智能体在不取代确定性统计基础的前提下加速实验流程。我们报告了构建和运营此类智能体所获得的经验教训。我们的核心发现是：这一矛盾并非通过使模型变得确定性来解决，而是通过限制其非确定性的作用范围来解决——即由模型负责选择和排序实验流程，而由确定性代码完成所有计算。

    arXiv:2611.04515v2 Announce Type: replace  Abstract: A/B testing underpins product decisions by providing trustworthy, reproducible quantification of proposed changes' effects. Growing large language model (LLM) capabilities have motivated efforts to automate experimentation workflows. Yet LLMs are nondeterministic and prone to hallucination, properties that conflict with the trust experimentation systems must provide. Experiments are also operationally messy: diagnosing anomalies, debugging configurations, and conducting bespoke follow-up analyses require flexible reasoning and manual effort. These tasks let agents accelerate experimentation without replacing its deterministic statistical foundations.   We report lessons from building and operating such an agent. Our central finding is that this tension is resolved not by making the model deterministic, but by bounding where its nondeterminism can reach; the model selects and sequences procedures, while deterministic code computes eve
    
[^331]: dKV Forcing：通过可微KV缓存分离生成与记忆表征

    dKV Forcing: Separating Generation and Memory Representations through a Differentiable KV Cache

    [https://arxiv.org/abs/2611.01648](https://arxiv.org/abs/2611.01648)

    提出dKV Forcing方法，为KV缓存写入分配独立的低秩残差参数，实现生成与记忆表征的解耦，并利用后续片段读取的梯度来优化缓存写入，从而改进自回归视频生成器的训练。

    

    自回归视频生成器会将其生成的每个片段写入键值（KV）缓存，而后续片段只能通过该缓存了解过去。自回放训练在缓存处截断了梯度，而自梯度强制（SGF）恢复了这一梯度，但将其应用于生成当前片段的同一组权重上。我们发现写入与生成是两种不同的角色：在预训练生成器中，写入的KV已经与用于去噪的KV存在差异，并且在共享权重的情况下，来自未来读取的梯度与生成梯度呈负对齐。我们提出dKV Forcing（dKV表示可微KV），为写入操作赋予其独立的参数——一个仅在片段进入缓存时启用的低秩残差——并利用后续读取其所写内容的片段的梯度来训练这些参数。在常见使用场景下，通过缓存打开这一梯度会耗尽GPU内存。我们将其归因于检查点重放对缓存的重写……（原文摘要在此处截断）

    arXiv:2611.01648v2 Announce Type: replace-cross  Abstract: A causal video generator writes every segment it produces into a key--value (KV) cache, and later segments know the past only through it. Self-rollout training stops the gradient at this cache, and Self Gradient Forcing (SGF) restores it, but applies it to the same weights that generate the current segment. We find that writing and generating are different roles: in a pretrained generator the written KV already differs from the KV used for denoising, and on shared weights the gradient from future reads is negatively aligned with the gradient of generation. We propose dKV Forcing (dKV denotes differentiable KV), which gives the write its own parameters, a low-rank residual used only when a segment enters the cache, and trains them with the gradient of the later segments that read what they wrote. Opening this gradient through the cache in common use exhausts GPU memory. We trace this to checkpoint replay re-writing the cache, no
    
[^332]: 面向低Tucker秩张量的轻量级表示学习

    Lightweight Representation Learning for Low Tucker Rank Tensors

    [https://arxiv.org/abs/2611.01168](https://arxiv.org/abs/2611.01168)

    本文提出了一种保持迭代轨迹不变的轻量化计算框架，通过仅计算更新所需的奇异三元组、避免Kronecker乘积的收缩以及稀疏加因子存储三个组件，显著降低了五种低Tucker秩张量迭代估计器的内存开销。

    

    用于低Tucker秩张量的迭代估计器，例如HOOI、核范数近端与ADMM方法、Frank-Wolfe以及缩放梯度下降，在每次迭代中都会构建大型对象：矩阵展开、Kronecker算子、完整的SVD分解以及环境维度的迭代变量。更新过程仅使用这些对象的一小部分，但它们却使内存成为处理大规模张量以及在边缘设备上的限制性资源。针对其中每种开销的补救方法都是已知的。然而，人们并不清楚什么样的重写能够保持迭代轨迹不变，以及这些重写在面对高效实现时于何处能带来收益。因此，我们提出了一种保持轨迹不变的轻量化计算方法。首先，我们将精确性表述为存储状态与参考迭代之间的交织关系。其次，我们证明了三个已知组件在五个估计器上均满足这一关系：仅计算更新所需的奇异三元组、无需Kronecker乘积的收缩运算，以及存储稀疏加因子（sparse-plus-factor）形式的……

    arXiv:2611.01168v2 Announce Type: replace-cross  Abstract: Iterative estimators for low Tucker rank tensors, such as HOOI, nuclear-norm proximal and ADMM methods, Frank-Wolfe and scaled gradient descent, form large objects in every iteration: unfoldings, Kronecker operators, full SVDs and ambient iterates. The update uses little of these objects, yet they make memory the limiting resource on large tensors and on edge-sized devices. Remedies for each of these costs are known. However, it is unclear when a rewriting keeps the iterates unchanged, and where it pays off against efficient implementations. We therefore present a trajectory-preserving lightweight evaluation. First, we state exactness as an intertwining relation between the stored state and the reference iteration. Second, we show that three known components satisfy this relation for five estimators: computing only the singular triplets that an update needs, contracting without Kronecker products, and storing sparse-plus-factor
    
[^333]: 稀疏自编码器归因图由边际决定

    Sparse Autoencoder Attribution Graphs Are Decided at the Margin

    [https://arxiv.org/abs/2611.00606](https://arxiv.org/abs/2611.00606)

    该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。

    

    基于稀疏自编码器（SAE）特征的归因图被用于预测消融操作的效果，并在不同训练种子之间进行比较。TopK SAE 在每个 token 上保留最大的预激活值，而特征的“边际”是其预激活值到截断阈值的距离。在来自三个 Pythia 模型的五个公开 TopK SAE 对上，将一个特征置零会在至少 0.74 的情形中使下一层的某个特征跨过截断阈值，而这些跨越承载了响应的大部分能量。对于将某个特征的编码放大三倍的写入操作，基于先前激活特征构建的图的中位相对误差达到 0.247 或更高。而“边际图”方法为每个预激活添加一阶变化并重新排序，在相同计算成本下，误差至多为 0.033。在中位数上，跨越特征的解码变化在范数上达到模型自身变化的至少 1.24 倍。在同一测试床的不同种子之间，特征之间的稀疏映射可以解释一个图的 0.85。

    arXiv:2611.00606v1 Announce Type: cross  Abstract: Attribution graphs over the features of sparse autoencoders (SAEs) are used to predict what an ablation does and are compared across training seeds. A TopK SAE keeps the largest preactivations at each token, and a feature's margin is its preactivation's distance to the cutoff. On five public TopK SAE pairs from three Pythia models, zeroing one feature moves a next layer feature across the cutoff in 0.74 of cases or more, and the crossings carry most of the response's energy. Over writes that scale a feature's code by three factors, the graph built on the features active beforehand has a median relative error of 0.247 or more. The margin graph adds the first order change to every preactivation and reranks, at the same cost, and errs by at most 0.033. At the median, the crossing features' decoded change is 1.24 times the model's own change in norm or more. Between seeds of one testbed, sparse maps between features explain 0.85 of a graph
    
[^334]: 揭秘针对合成视频的提示窃取攻击

    Demystifying Prompt Stealing Attacks against Synthetic Video

    [https://arxiv.org/abs/2611.00102](https://arxiv.org/abs/2611.00102)

    该研究提出V-PSA方法，证明单纯恢复文本提示无法忠实重建合成视频，通过结合恢复的字幕与DDIM反演构建参考轨迹，实现对公开合成视频的高保真重建与受控编辑。

    

    创作者公开展示生成的内容以宣传和销售其作品，但这些示例也可能使未经授权的复制和定制成为可能。我们研究攻击者是否能够在无法访问原始提示或生成器的情况下，仅凭单个公开视频推导出可重用的输入，以实现忠实重建和受控编辑。我们将字幕生成和提示优化方法适配到视频提示窃取攻击（PSA）中，发现仅靠提示恢复只能提供有限的重建保真度——即使使用原始提示，当初始噪声改变时也会产生不同的空间和时间表现。使用独立的生成器会进一步扩大重建差距，因为即使重用原始提示和种子，仍可能生成截然不同的视频。我们提出V-PSA，该方法利用恢复的字幕和DDIM反演在独立生成器下构建参考轨迹，并通过Null-Text反演……（摘要在此处截断）

    arXiv:2611.00102v2 Announce Type: replace-cross  Abstract: Creators publicly showcase generated content to advertise and sell their work, but these examples may also enable unauthorized reproduction and customization. We study whether an adversary can derive reusable inputs from a single public video for faithful reconstruction and controlled editing, without access to the original prompt or generator. We adapt captioning and prompt optimization approaches to video prompt stealing attacks (PSAs) and find that prompt recovery alone provides limited reconstruction fidelity. Even the original prompt produces different spatial and temporal realizations when the initial noise changes. Using an independent generator enlarges the reconstruction gap, as reusing the original prompt and seed can still produce substantially different videos. We propose V-PSA, which uses a recovered caption and DDIM inversion to construct a reference trajectory under an independent generator. Null-Text Inversion t
    
[^335]: 用于视频活动预测的可干预概念动力学

    Intervenable Concept Dynamics for Video Activity Forecasting

    [https://arxiv.org/abs/2611.00020](https://arxiv.org/abs/2611.00020)

    本文提出TRACE-CBM框架，利用稀疏时空图细化观测概念激活并递归预测未来概念状态演化，首次实现了基于可编辑命名概念状态的视频未来活动预测。

    

    概念瓶颈模型通过人类可理解的概念来传递预测，使其中间状态可检查、可编辑。将这一模型家族扩展到预测任务，需要建模概念激活如何随着物体、动作及其交互的变化而随时间演化，然而现有的概念瓶颈模型仅将概念建立在观测输入上，并不预测其未来演化。我们利用正在进行的活动视频来研究这一问题，并通过命名概念状态来预测未来活动。我们提出了TRACE-CBM（时序关系活动概念演化）方法，该方法首先利用稀疏时空图对观测到的概念激活进行细化，随后递归地预测未来概念状态，每次状态转移均以先验活动分布为条件，并使用共享的活动分类器对观测状态和预测状态进行解码。在推理阶段，该模型支持编辑概念值与活动……

    arXiv:2611.00020v2 Announce Type: replace-cross  Abstract: Concept bottleneck models route predictions through human-understandable concepts, making their intermediate states inspectable and editable. Extending this model family to forecasting requires modeling how concept activations evolve over time as objects, actions, and their interactions change, yet existing CBMs ground concepts only in observed inputs and do not forecast their future evolution. We study this problem using videos of ongoing activities and forecast future activities through a named-concept state. We propose TRACE-CBM (Temporal Relational Activity Concept Evolution), which first refines observed concept activations with sparse spatio-temporal graphs. It then recursively predicts future concept states, conditioning each transition on the prior activity distribution, and decodes both observed and predicted states with a shared activity classifier. At inference time, the model supports editing concept values, activit
    
[^336]: 拉马努金挑战赛结果：面向多格式证明的AI辅助评估流水线

    Ramanujan Challenge Results: AI-Assisted Evaluation Pipeline for Multi-Format Proofs

    [https://arxiv.org/abs/2610.40551](https://arxiv.org/abs/2610.40551)

    本文分析了拉马努金挑战赛（研究级数论基准）的提交结果，发现AI能够证明已公开的猜想并给出与官方解答概念迥异的正确证明，并提出了一种AI辅助的评估流水线，以可靠地大规模评判自然语言、CAS代码等多格式的复杂证明。

    

    随着AI数学能力的快速提升，有意义的基准测试正从大学水平的问题和数学奥林匹克竞赛转向研究级数学。拉马努金挑战赛是一个由数论研究级问题组成的基准测试，其中包括未发表的结果和公开猜想。该挑战赛提出了数学常数的具体公式，其正确性可以通过数值方法验证，而其证明可能需要非显而易见的数学方法。在本文中，我们分析了挑战赛的提交作品，衡量了不同AI模型、计算机代数系统（CAS）和形式化工具的相对成功率。我们发现了与官方解答在概念上截然不同的正确提交作品，以及其中一个已公开猜想的证明。评估这些提交作品暴露了第二个挑战：如何可靠地大规模评判复杂证明。为此，我们引入了一种AI辅助评估流水线，用于处理自然语言证明、CAS代码和（形式化证明）

    arXiv:2610.40551v1 Announce Type: new  Abstract: As AI's mathematical capabilities rapidly improve, meaningful benchmarks are shifting from university-level problems and math olympiads to research-level mathematics. The Ramanujan Challenge is a benchmark comprised of research-level problems in number theory, including unpublished results and open conjectures. It presented concrete formulas for mathematical constants whose correctness can be checked numerically, while their proofs may require non-obvious mathematics. Here, we analyze the Challenge submissions, measuring the relative success of different AI models, Computer Algebra Systems (CAS), and formal tools. We find correct submissions that are conceptually distinct from the official solutions, and a proof of one of the published open conjectures. Evaluating the submissions exposed a second challenge: judging complex proofs reliably at scale. We introduce an AI-assisted evaluation pipeline for natural language proofs, CAS code, and
    
[^337]: 原子作为智能体：集体预测编码产生的涌现秩序

    Atom as an Agent: Emergent Order by Collective Predictive Coding

    [https://arxiv.org/abs/2610.36673](https://arxiv.org/abs/2610.36673)

    该论文将语言符号涌现的“集体预测编码”理论引入物理学，把原子视为相互预测的智能体，仅凭观测到的自旋构型就能自动涌现出伊辛模型、条纹序模型和自旋玻璃系统中的宏观序参量，为宏观秩序的起源提供了新解释。

    

    arXiv:2610.36673v1 公告类型：交叉。摘要：宏观序参量从微观系统中涌现的内在原因是什么？本文将集体预测编码（CPC）——一种关于语言/符号涌现的理论——应用于物理学。正如人类语言从人际交流中涌现一样，物理系统的宏观序参量也从原子之间的“交流”中涌现出来——每个原子被视为一个智能体，通过推断来预测彼此的行为。我们给出了三个经典自旋系统的例子：二维铁磁和反铁磁伊辛模型、具有竞争条纹序的二维自旋模型，以及全连接的Sherrington–Kirkpatrick自旋玻璃模型。在所有这些例子中，包括副本重叠在内的恰当宏观序参量，仅凭观测到的自旋构型，便能通过CPC的通信信息学自动涌现。我们将学习到的宏观坐标称为集体预……（原文此处截断）

    arXiv:2610.36673v1 Announce Type: cross  Abstract: What is the intrinsic reason of emergence of macroscopic order parameter out of microscopic systems? In this article, we apply to physics the collective predictive coding (CPC), the theory of language/symbol emergence. As our language emerges out of our human communications, physical macroscopic order parameter emerges out of communications among atoms --- each atom is regarded as an agent who predicts each other's behavior through inference. We present three examples of classical spin systems: two-dimensional ferromagnetic and antiferromagnetic Ising models, a two-dimensional spin model with competing stripe orders, and the fully connected Sherrington--Kirkpatrick spin-glass model. In all of these examples, proper macroscopic order parameters including the replica overlap emerge automatically through communication informatics of CPC, only from observed spin configurations. We call the learned macroscopic coordinate the collective pred
    
[^338]: ResCom：面向大语言模型量化的激活残差补偿

    ResCom: Activation Residual Compensation for Large Language Model Quantization

    [https://arxiv.org/abs/2610.35955](https://arxiv.org/abs/2610.35955)

    ResCom提出一种低秩激活残差补偿方法，通过输出感知校准以闭式形式利用激活舍入残差校正层输出误差，无需梯度训练即可显著提升大语言模型4比特权重-激活量化的精度。

    

    低比特权重-激活量化可以降低部署大语言模型所需的内存和计算量，但即使经过校准，激活舍入仍可能导致精度下降。基于舍入后输入的重建无法区分同一量化区间内的激活值。我们提出ResCom，一种低秩校正方法，利用激活舍入残差来补偿由此产生的层输出误差。该方法的输出感知校准流程根据残差方向对输出重建的贡献来选择这些方向，并以闭式形式计算校正，无需梯度训练，也无需更新已校准的权重。在采用4比特权重和激活的Llama-3-8B上，秩为32的注意力校正将WikiText-2困惑度从7.708降至7.147，MMLU提升2.89个百分点，五样本GSM8K准确率从23.96%提升至36.16%。对参数规模从1B到70B的模型以及在留出的C4测试集上的评估（摘要在此处截断）……

    arXiv:2610.35955v1 Announce Type: cross  Abstract: Low-bit weight--activation quantization reduces the memory and computation needed to deploy large language models, but activation rounding can still degrade accuracy after calibration. Reconstruction from rounded inputs cannot distinguish activations within the same quantization bin. We propose ResCom, a low-rank correction that uses the activation rounding residual to compensate for the resulting layer-output error. An output-aware calibration procedure selects residual directions by their contribution to output reconstruction and computes the correction in closed form, without gradient training or updates to the calibrated weights. On Llama-3-8B with 4-bit weights and activations, rank-32 attention corrections reduce WikiText-2 perplexity from 7.708 to 7.147, improve MMLU by 2.89 percentage points, and raise five-shot GSM8K accuracy from 23.96 to 36.16 percent. Evaluations on models from 1B to 70B parameters and on held-out C4 test t
    
[^339]: 基于大语言模型的排序函数生成用于终止性证明

    Ranking Function Generation with LLMs for Termination

    [https://arxiv.org/abs/2610.35523](https://arxiv.org/abs/2610.35523)

    该论文提出RAINFALL神经符号框架，通过将排序函数合成转化为带槽位约束的代码补全任务，并结合符号验证器的反馈迭代改进，利用大语言模型自动生成可验证的排序函数以证明循环终止性。

    

    排序函数是证明循环终止性的基础技术，但手动构造排序函数具有挑战性且需要大量专业知识。我们提出了RAINFALL，这是一个神经符号框架，遵循生成-验证-改进范式，将大语言模型与符号验证紧密集成。RAINFALL将排序函数合成任务表述为代码补全任务，通过在输入程序中插入占位符，并约束大语言模型为每个循环填充四个预定义的槽位。补全后的程序随后由符号验证器进行检查。如果验证成功，则获得有效的排序函数；否则，验证器的诊断信息会被转换为自然语言反馈，以指导大语言模型输出的迭代改进。我们在两个公开基准测试套件上对RAINFALL进行了评估，共包含358个可终止的C程序。使用MiniMax V2.7，RAINFALL成功为323个程序（90……）合成了可验证的排序函数

    arXiv:2610.35523v1 Announce Type: cross  Abstract: Ranking functions are a fundamental technique for proving loop termination, but constructing them manually is challenging and requires substantial expertise. We propose RAINFALL, a neuro-symbolic framework which follows the generate-verify-refine paradigm and tightly integrates LLMs with symbolic verification. RAINFALL formulates ranking function synthesis as a code-completion task by inserting placeholders into the input program and constraining the LLM to fill four predefined slots for each loop. The completed program is then checked by a symbolic verifier. If verification succeeds, a valid ranking function is obtained; otherwise, the verifier's diagnostics are translated into natural-language feedback that guides iterative refinement of the LLM's output. We evaluate RAINFALL on two public benchmark suites comprising 358 terminating C programs. Using MiniMax V2.7, RAINFALL synthesizes verifiable ranking functions for 323 programs (90
    
[^340]: EvolveMAS：面向去中心化多智能体系统的统一自进化框架

    EvolveMAS: A Unified Self-Evolution Framework for Decentralized Multi-Agent Systems

    [https://arxiv.org/abs/2610.34450](https://arxiv.org/abs/2610.34450)

    EvolveMAS通过为每个主智能体配备专用MemoryAgent进行测试时记忆治理，在隔离的结构化记忆支持下，实现了去中心化多智能体系统在合作、混合动机和对抗等多种交互环境中的统一测试时适应与可靠经验复用。

    

    去中心化多智能体系统（MAS）使大语言模型智能体能够在没有集中控制的情况下进行交互。智能体需要在测试时适应其他智能体，并复用积累的经验来完善其交互策略，以应对未来在合作、混合动机和对抗等多种环境下的交互。然而，现有方法往往针对特定交互环境开发适应机制。这就提出了一个核心问题：去中心化MAS如何支持跨交互环境的测试时适应和可靠的经验复用？我们提出了EvolveMAS，这是一个统一的自进化框架，支持基于经验的回合内和跨回合适应。EvolveMAS为每个主智能体配备一个专用的MemoryAgent进行测试时记忆治理，该MemoryAgent在主智能体的推理上下文之外维护隔离的结构化记忆。EvolveMAS将适应过程组织为两个互补的过程：（原文此处被截断）

    arXiv:2610.34450v1 Announce Type: cross  Abstract: Decentralized multi-agent systems (MAS) enable large language model agents to interact without centralized control. Agents need to adapt to other agents at test time and reuse accumulated experience to refine their interaction strategies for future interactions across cooperative, mixed-motive, and adversarial settings. However, existing approaches often develop adaptation mechanisms for particular interaction settings. This raises a central question: how can decentralized MAS support test-time adaptation and reliable experience reuse across interaction settings? We propose EvolveMAS, a unified self-evolution framework that supports experience-driven adaptation within and across episodes. EvolveMAS pairs each main agent with a dedicated MemoryAgent for test-time memory governance, which maintains isolated Structured Memory outside the main agent's reasoning context. EvolveMAS organizes adaptation into two complementary processes: Withi
    
[^341]: PEAR：面向大型视觉语言模型感知与探索的自适应优势重塑方法

    PEAR: Adaptive Advantage Reshaping for Perception and Exploration in Large Vision-Language Models

    [https://arxiv.org/abs/2610.32216](https://arxiv.org/abs/2610.32216)

    PEAR提出一种自适应优势重塑方法，以视觉依赖为基础信号、仅在高熵位置施加与视觉依赖高低挂钩的熵探索增益，从而同时提升大型视觉语言模型多模态推理中的感知与探索能力。

    

    有效的多模态推理要求模型既能可靠地利用视觉信息，又能充分探索多样化的推理路径。为实现这一目标，先前的工作将视觉依赖与词元熵相结合来重塑序列级优势，但用视觉依赖对词元熵的贡献进行缩放，可能导致视觉依赖较低的高熵词元被低估权重，尽管这些词元在推理探索中发挥着重要作用。为解决这一局限，我们提出了PEAR，一种用于提升多模态推理中感知与探索能力的自适应优势重塑方法。PEAR保留视觉依赖作为基础信号，并仅在高熵位置加入基于熵的探索增益，且视觉依赖越低，附加增益越大。随后，融合信号被映射为正的词元级权重，用以重塑序列级优势。在Qwen2.5-VL-3B/7B上的实验表明，PEAR取得了显著的性能提升。

    arXiv:2610.32216v1 Announce Type: new  Abstract: Effective multimodal reasoning requires a model to reliably utilize visual information while sufficiently exploring diverse reasoning paths. To achieve this goal, prior work combines visual dependency and token entropy to reshape sequence-level advantages, but scaling the contribution of token entropy by visual dependency can leave high-entropy tokens with low visual dependency underweighted, despite their important role in reasoning exploration. To address this limitation, we propose PEAR, an adaptive advantage reshaping method for improving perception and exploration in multimodal reasoning. PEAR retains visual dependency as the base signal and adds an entropy-based exploration gain only at high-entropy positions, with lower visual dependency leading to a larger additional gain. The fused signal is then mapped to positive token-level weights that reshape the sequence-level advantage. Experiments on Qwen2.5-VL-3B/7B show that PEAR achie
    
[^342]: 雅可比透镜的数学理论

    A Mathematical Theory of the Jacobian Lens

    [https://arxiv.org/abs/2610.31955](https://arxiv.org/abs/2610.31955)

    本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。

    

    Anthropic 最近的工作引入了一种新的可解释性工具，称为雅可比透镜（Jacobian lens，J-lens，Gurnee 等人，2026）。该方法在前沿语言模型的隐藏状态中识别出一些线性方向，其性质令人联想到工作记忆式的表征。然而，目前尚不清楚这些特征的哪些机制性质使得它们能够被 J-lens 检测到。为此，我们提出了 J-lens 的数学理论，并证明它优先识别的是“相干特征”——即表现得像主导的、空间相干的 Lyapunov 模态那样的方向。沿这些方向的小扰动会跨层放大，在不同词元位置之间大致保持恒定，并在不同上下文中保持一致。尽管我们对这些性质是否必然与“全局工作空间”相关联持不可知态度，但通过解析推导出的 J-lens 偏好仍然解释了先前应用于……的实证结果。

    arXiv:2610.31955v1 Announce Type: cross  Abstract: Recent work from Anthropic introduced a new interpretability tool, called the Jacobian lens (J-lens, Gurnee et al., 2026). This method identified linear directions in the hidden states of frontier language models that had properties reminiscent of working memory-like representations. However, it is unclear what mechanistic properties of such features make them detectable to the J-lens. To this end, we introduce a mathematical theory of the J-lens, and show that it preferentially identifies coherent features -- directions that act like leading, spatially coherent Lyapunov modes. Small perturbations along these directions amplify across layers, remain roughly constant across token positions, and are consistent across different contexts. While we are agnostic to whether these properties are necessarily associated with a ``Global Workspace'', the analytically derived preferences of the J-lens nevertheless explain prior empirical applied to
    
[^343]: 一次评估足矣：基于轨迹校正的无反演图像编辑

    One Evaluation Is Enough: Inversion-Free Image Editing with Trajectory Correction

    [https://arxiv.org/abs/2610.30827](https://arxiv.org/abs/2610.30827)

    提出端点校准传输（ECT），一种免训练、闭式求解的无反演图像编辑方法，仅需单次场评估即可通过“先传输、后校正”的构造实现可靠的图像编辑。

    

    真实图像编辑通常依赖于反演和迭代采样，而高效的单步编辑器往往通过在多个噪声水平下估计并平均编辑场来稳定大幅度的更新。我们提出一个更严格的问题：单次时间场评估是否足以实现可靠的无反演编辑？我们的答案基于方差保持（VP）单步模型中一个此前未被充分利用的对偶性：在共享噪声状态下进行一次源-目标查询，可同时确定局部角向语义切线和干净的端点位移。我们将切线沿解析已知的剩余VP角度进行积分，得到名义上的单步传输，并利用端点位移通过受约束投影来校正其长程外推。这种“先传输、后校正”的构造产生了端点校准传输（ECT），这是一种闭式、免训练的编辑器，且仅需一个噪声水平的评估。

    arXiv:2610.30827v2 Announce Type: replace-cross  Abstract: Real-image editing commonly relies on inversion and iterative sampling, while efficient one-step editors often stabilize a large update by estimating and averaging editing fields at multiple noise levels. We ask a stricter question: is a single temporal field evaluation sufficient for reliable inversion-free editing? Our answer rests on a previously underused duality in a variance-preserving (VP) one-step model: one source-target query at a shared noisy state simultaneously determines a local angular semantic tangent and a clean endpoint displacement. We integrate the tangent over the analytically known remaining VP angle to obtain a nominal one-step transport, and use the endpoint displacement to correct its long-horizon extrapolation through a constrained projection. This transport-first, correction-second construction yields Endpoint-Calibrated Transport (ECT), a closed-form, training-free editor requiring only one noise lev
    
[^344]: 超越静态排行榜：面向时间序列基础模型的结构多样性能力评估

    Beyond Static Leaderboards: Capability-Focused Evaluation with Structural Diversity for Time Series Foundation Models

    [https://arxiv.org/abs/2610.26251](https://arxiv.org/abs/2610.26251)

    该论文提出CaFE评估框架，通过八个结构特征对原始基准数据进行重采样以引入结构多样性，将时间序列基础模型的评估从静态单一分数排行榜升级为可刷新的多维能力导向评估，从而更真实地反映模型的泛化能力。

    

    时间序列基础模型（TSFM）的静态公开基准通常具有固定且狭窄的测试分布。这些基准的反复使用会助长针对特定基准的调优，使得报告的性能反映的是对固定测试集的适应，而非真正的预测泛化能力。此外，单一的聚合分数无法全面刻画模型在不同时间结构和输入关系下的能力。我们提出了CaFE（能力聚焦评估），这是一个基于原始基准数据的可刷新评估扩展，它通过结构多样性丰富静态基准，并通过多维度的、面向能力的评估丰富单一分数的黑盒式评估。CaFE首先对原始基准进行特征分析，定义了八个刻画其时间结构的结构特征（SF）。对于每个结构特征，它根据其结构支持度对原始数据进行重采样，并去除边缘分布外……

    arXiv:2610.26251v2 Announce Type: replace  Abstract: Static public benchmarks for time series foundation models (TSFMs) typically have fixed and narrow test distributions. Their repeated reuse encourages benchmark-specific tuning, making reported performance reflect adaptation to the fixed test set rather than genuine forecasting generalization. In addition, a single aggregate score cannot fully characterize model capabilities across distinct temporal structures and input relationships. We propose CaFE (Capability-Focused Evaluation), a refreshable evaluation extension grounded in original benchmark data that enriches static benchmarks with structural diversity and enriches single-score black-box evaluation with multi-dimensional, capability-oriented assessment. CaFE first profiles the original benchmark and defines eight Structural Features (SFs) that characterize its temporal structure. For each SF, it resamples the original data according to its structural support, removes edge-of-d
    
[^345]: ReTRACE：化学框架中反馈引导的大语言模型智能体用于逆合成规划

    ReTRACE: Feedback-Guided LLM Agent in a Chemistry Harness for Retrosynthesis Planning

    [https://arxiv.org/abs/2610.24744](https://arxiv.org/abs/2610.24744)

    该论文提出ReTRACE智能体，将大语言模型嵌入集成分子工具、预测引擎、构建模块库和多标准评估器的确定性化学框架中，通过两个时间尺度的框架反馈学习进行逆合成规划，并引入结合程序化检查与对抗性LLM审查的Audit@k指标来评估路线的化学有效性。

    

    arXiv:2610.24744v2 公告类型：替换 摘要：计算机辅助逆合成已通过单步预测、树搜索和路线评分取得进展，然而复杂目标分子仍需要路线层面的判断，其中立体化学、官能团兼容性、成本和绿色性等因素相互影响。在此我们提出ReTRACE（基于工具编排推理与上下文感知证据的逆合成），这是一个将大语言模型（LLM）置于确定性化学框架中的智能体，该框架集成了分子工具、异构单步预测引擎、可购买的构建模块库、外部路线记忆以及多标准评估器。为了评估超出库存终止条件之外的化学有效性，我们引入了Audit@k，这是一种两阶段指标，将程序化结构检查与对抗性LLM审查相结合，能够紧密贴合专家的判断。ReTRACE在两个时间尺度上从框架反馈中学习：在……（原文摘要在此处截断）

    arXiv:2610.24744v2 Announce Type: replace  Abstract: Computer-aided retrosynthesis has progressed through single-step prediction, tree search and route scoring, yet complex targets still require route-level judgement where stereochemistry, functional-group compatibility, cost and greenness interact. Here we present \textbf{\mbox{ReTRACE}} (\textbf{RE}trosynthesis with \textbf{T}ool-orchestrated \textbf{R}easoning \textbf{A}nd \textbf{C}ontext-aware \textbf{E}vidence), an agent that places a large language model (LLM) inside a deterministic chemistry harness integrating molecular tools, heterogeneous single-step prediction engines, purchasable building-block libraries, an external route memory and a multi-criterion evaluator. To assess chemical validity beyond stock termination, we introduce Audit@k, a two-stage metric combining programmatic structural checks with adversarial LLM review that closely tracks expert judgement. ReTRACE learns from harness feedback on two timescales: within 
    
[^346]: 针对失控风险的红队多智能体攻击测试

    Red-Teaming Multi-Agent Attacks for Loss-of-Control Risks

    [https://arxiv.org/abs/2610.24052](https://arxiv.org/abs/2610.24052)

    该论文提出MiniLab多智能体攻击测试平台，证明分布式多智能体攻击可通过分工协作逃避仅单独审查各智能体轨迹的前沿监控器（仅检测出3.4%的成功攻击），揭示了现有AI控制措施在应对合谋性失控风险方面的重大盲区。

    

    AI智能体的能力与持续性不断增强，并且正以日益庞大的规模被部署。随着越来越多的智能体开始协作，当它们围绕有害目标进行隐蔽合谋时，AI控制措施必须依然保持有效。我们提出了MiniLab，这是一个由真实运行的服务和代码库组成的合成测试环境（包含5个互联服务、249个良性任务、8个攻击目标），改编自内部的AI部署基础设施。我们展示了分布式攻击——其中不同的智能体各自执行独立的攻击片段——并说明当监控器单独阅读每个智能体的轨迹时，这些攻击如何凭借其隐蔽性逃避前沿监控器的检测。我们通过16种人工优化的攻击（权重窃取、无监控的恶意部署、权限提升等）展示了分布式攻击如何在多样化的情境和有害目标中扩散。一个前沿监控器GPT-5.6 Sol在单独阅读每个智能体的轨迹时，仅能标记出3.4%的成功分布式攻击，相比之下……（原文摘要在此处截断）

    arXiv:2610.24052v2 Announce Type: replace-cross  Abstract: AI agents are growing more capable and persistent, and are being deployed in ever larger numbers. As more start collaborating, AI control measures must remain effective when they covertly collude on a harmful objective. We introduce MiniLab, a synthetic series of live services and codebases (5 interconnected services, 249 benign tasks, 8 attack objectives), adapted from internal AI deployment infrastructure. We demonstrate distributed attacks, in which different agents carry out separate attack fragments, and how their covertness evades frontier monitors when reading their trajectories individually. We show how distributed attacks spread across diverse contexts and harmful objectives through 16 human-optimized attacks (weight exfiltration, unmonitored rogue deployment, permission elevation, and more). A frontier monitor, GPT-5.6 Sol, reading each agent separately flags only 3.4% of successful distributed attacks, compared with 
    
[^347]: FlashReFocus：秒级交互式图像重新对焦

    FlashReFocus: Interactive Image Refocusing in Seconds

    [https://arxiv.org/abs/2610.23297](https://arxiv.org/abs/2610.23297)

    FlashReFocus 对每张照片仅运行一次扩散模型来恢复全清晰图像，再用基于显式散焦图条件的轻量级网络渲染每次编辑，从而实现秒级的交互式图像重新对焦。

    

    单图像重新对焦是在拍摄后改变照片的焦平面和景深。这本质上是交互式的：用户会在同一张图像上尝试多个焦点和模糊强度，因此每次编辑都应该快速且只改变模糊。然而，近期的生成式方法为每次编辑都重新运行大型扩散模型。我们观察到这两个子问题在性质上有所不同：恢复全清晰图像是一个病态问题，需要生成先验；而渲染模糊在很大程度上由深度和焦平面决定。因此，我们提出 FlashReFocus，它对每张照片只运行一次扩散过程，然后用轻量级网络渲染每次编辑。为解决病态恢复问题，带有解码器精炼器的单步扩散去模糊器在平衡保真度和感知质量的同时高效恢复精细细节。为实现精确控制，渲染器以显式散焦图为条件，并使用散焦偏置注意力，这使得……

    arXiv:2610.23297v1 Announce Type: cross  Abstract: Single-image refocusing changes the focal plane and depth of field of a photograph after capture. It is inherently interactive: users try many focal points and blur strengths on one image, so each edit should be fast and change only the blur. Recent generative methods, however, rerun large diffusion models for every edit. We observe that the two sub-problems differ in kind: recovering the all-in-focus image is ill-posed and needs a generative prior, whereas rendering blur is largely determined by depth and the focal plane. We therefore propose FlashReFocus, which runs diffusion once per photograph and renders every edit with a lightweight network. To address the ill-posed restoration, a one-step diffusion deblurrer with a decoder refiner recovers fine detail efficiently while balancing fidelity and perceptual quality. For precise control, the renderer is conditioned on an explicit defocus map and uses defocus-bias attention, which lets
    
[^348]: 尊重语义即可：通过LLM前语义可变分辨率分词与LLM内相关性剪枝实现图像token压缩

    Just Respect the Semantics: Image-Token Compression via Pre-LLM Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning

    [https://arxiv.org/abs/2610.18690](https://arxiv.org/abs/2610.18690)

    Varest通过语义引导的可变分辨率token化与相关性token剪枝，仅用5.56%的图像token即可保留86%-92%的全token性能，实现高效且语义无损的图像token压缩。

    

    图像token压缩必须在不损害语义的前提下减少计算量。我们提出Varest，其核心指导原则是语义可替代性：连贯语义区域内的token可以被粗略概括，而物体边界和文字内容则需要更精细的区分。物体和文本定位器引导可变分辨率合并，在边界和文字处保留空间覆盖范围和更精细的细节。学习到的[REL]寄存器随后在解码器早期处理之后对视觉token进行剪枝。跨预算的训练方式使得每个骨干网络只需一个检查点。在三个骨干网络和13个基准测试上，Varest在每个评估预算下的平均性能保持率均优于15个基线方法，仅使用原始层平均图像token数量的5.56%即可保留全token性能的86.10%至92.33%。

    arXiv:2610.18690v1 Announce Type: cross  Abstract: Image-token compression must reduce computation without hurting the semantics. We introduce Varest, guided by semantic fungibility: tokens within a coherent semantic region can be summarized coarsely, while object boundaries and literal text require finer distinctions. Object and text localizers guide variable-resolution merging, preserving spatial coverage and finer detail at boundaries and text. Learned [REL] registers then prune visual tokens after early decoder processing. Training across budgets enables one checkpoint per backbone. Across three backbones and 13 benchmarks, Varest outperforms 15 baselines in mean performance retention at every evaluated budget, retaining 86.10-92.33% of full-token performance with only 5.56% of the original layer-averaged image-token count.
    
[^349]: Urbex：面向城市级3D场景的智能体式空间定位

    Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes

    [https://arxiv.org/abs/2610.18356](https://arxiv.org/abs/2610.18356)

    Urbex提出一个智能体式视觉-语言框架，将城市级3D定位从候选排序重新建模为主动空间搜索，通过强化学习优化工具使用策略，在CityRefer上取得最佳性能且速度显著快于排序基线。

    

    城市级3D定位通常被表述为候选排序问题，这需要预先计算的实例提案，并在大型户外场景中变得低效。我们提出Urbex，一个智能体式的视觉-语言框架，它将定位任务转化为主动空间搜索。Urbex将城市级3D场景表示为交互式多视角环境，允许VLM智能体查询地标、放大局部区域、渲染斜视图，并提交定位的边界框。我们在CityRefer数据集上通过强化学习优化这一工具使用策略，采用平滑定位奖励和轻量级塑造奖励以实现高效探索。实验表明，Urbex在CityRefer上取得了最佳的Hit@1和IoU指标，支持无需地面真值边界框的无候选定位，并且比最强的城市级排序基线快得多。在STPLS3D-Refer上的零样本结果进一步表明其跨场景鲁棒性有所提升。

    arXiv:2610.18356v1 Announce Type: cross  Abstract: City-scale 3D grounding is commonly formulated as candidate ranking, which requires precomputed instance proposals and becomes inefficient in large outdoor scenes. We propose Urbex, an agentic vision-language framework that instead casts grounding as active spatial search. Urbex represents a city-scale 3D scene as an interactive multi-view environment, allowing a VLM agent to query landmarks, zoom into local regions, render oblique views, and commit a grounded bounding box. We optimize this tool-use policy with reinforcement learning on CityRefer, using smooth localization rewards and lightweight shaping rewards for efficient exploration. Experiments show that Urbex achieves the best Hit@1 and IoU on CityRefer, supports candidate-free grounding without ground-truth boxes, and is substantially faster than the strongest city-scale ranking baseline. Zero-shot results on STPLS3D-Refer further indicate improved cross-scene robustness.
    
[^350]: SPLICE：稀疏物理库识别与组合扩展

    SPLICE: Sparse Physics Library Identification and Compositional Expansion

    [https://arxiv.org/abs/2610.17029](https://arxiv.org/abs/2610.17029)

    SPLICE是一个无需机制标签即可从多机制混合数据中自动发现物理算子的框架，通过凸优化推断贡献、校准测试分配机制，实现算子库的自动扩展并组合成零样本模拟器。

    

    神经算子拆分方法将一组神经算子（每个物理机制对应一个算子）组合成针对新方程的零样本模拟器。然而，现有方法需要事先指定这些机制及其标签。我们提出的问题是：能否从未标记的、多种机制同时作用的轨迹中调整和扩展一个预训练的算子库？我们引入了SPLICE，这是一个在混合数据上无需机制标签即可发现算子的框架。候选算子对每条轨迹的贡献通过求解凸优化问题来推断。随后，通过一个经过校准的测试为每个发现的算子分配其实现的机制。该框架还能检测失败情况，即当算子库缺乏所需物理机制或各机制之间无法充分分离时。在一维和二维系统上，发现的算子能够从未标记的混合数据中恢复机制，并组合成零样本模拟器。

    arXiv:2610.17029v1 Announce Type: cross  Abstract: Neural operator splitting composes a library of neural operators, one per physical mechanism, into zero-shot simulators of new equations. Existing approaches, however, require these mechanisms and their labels to be specified in advance. We ask whether a pretrained library can instead be adapted and grown from unlabeled trajectories in which multiple mechanisms act simultaneously. We introduce SPLICE, a framework that discovers operators without mechanism labels on mixture data. The contributions of the candidate operators to each trajectory are inferred by solving a convex optimization problem. A calibrated test then assigns each discovered operator the mechanism it implements. Failure cases are also detectable, when the library lacks the required physics or when mechanisms are insufficiently separable. Across one and two dimensional systems, the discovered operators recover mechanisms from unlabeled mixtures and compose into zero-sho
    
[^351]: CADFather：通过协同工具调用实现自主CAD重建

    CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use

    [https://arxiv.org/abs/2610.09127](https://arxiv.org/abs/2610.09127)

    CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。

    

    从3D形状重建可编辑的CAD模型仍然是一项具有挑战性的工程任务。现有方法可以提出CAD操作，但没有任何单一的提案来源能够在不同的零件几何形状和重建阶段都同样有效。我们提出了CADFather，这是一个自主的智能体系统，通过协调互补的工具从3D网格中恢复参数化的CAD程序。一个视觉语言助手会检查目标形状和中间重建结果的渲染图，然后决定扩展哪些候选CAD程序、调用哪些工具、生成多少个提案以及何时结束。学习型和算法型工具负责提出CAD操作，而数值优化则用于细化现有程序的参数。新提出或细化后的程序会被执行和评估，从而为后续决策提供反馈。该智能体为每个目标零件维护多个备选的候选程序，并保留最佳的有效重建结果。

    arXiv:2610.09127v1 Announce Type: cross  Abstract: Reconstructing an editable CAD model from a 3D shape remains a challenging engineering task. Existing methods can propose CAD operations, but no single source of proposals works equally well across different part geometries and stages of reconstruction. We introduce CADFather, an autonomous agentic system that coordinates complementary tools to recover parametric CAD programs from 3D meshes. A vision-language assistant inspects renders of the target and intermediate reconstructions, then decides which candidate CAD programs to extend, which tools to invoke, how many proposals to generate, and when to finish. Learned and algorithmic tools propose CAD operations, while numerical optimization refines the parameters of existing programs. Proposed or refined programs are executed and evaluated to provide feedback for subsequent decisions. The agent maintains alternative candidate programs for each target part and preserves the best valid re
    
[^352]: 凸-凹强化学习

    Convex-Concave Reinforcement Learning

    [https://arxiv.org/abs/2610.09108](https://arxiv.org/abs/2610.09108)

    该论文揭示强化学习的策略优化问题在对数密度比坐标下具有凸差规划结构，从而将 CPI、NPG、TRPO 和 AWR 统一为特例，突破了传统凸代理近似的局限。

    

    策略学习驱动着当今许多最重要且投入巨大的强化学习应用。然而其赖以建立的核心优化问题（最大化期望回报）是出了名的非凸问题，即使采用直接策略参数化也是如此，而该领域在很大程度上是通过回避这一问题来应对的：在信赖域约束下优化回报的凸代理近似（如 NPG、TRPO、PPO、AWR）。我们证明这个看似无结构的问题实际上并非没有结构。在对数密度比坐标 $y := \log[\pi/\pi_n]$ 下，通过逐决策重要性采样（PDIS）可计算的精确逐迭代目标函数是一个“凸差-凸差约束规划”。这一结构使我们能够超越代理近似：它沿可解释的轴向将 CPI、NPG、TRPO 和 AWR 恢复为特例，并开辟了一个耦合连续决策的多步轴 $k$……（摘要在此处被截断）

    arXiv:2610.09108v1 Announce Type: new  Abstract: Policy learning drives many of the most consequential and heavily-invested applications of reinforcement learning today. Yet the core optimization problem it rests on (maximizing expected return) is notoriously non-convex, even under a direct policy parameterization, and the field has largely responded by avoiding it: optimizing convex surrogate approximations of the return under trust-region constraints (NPG, TRPO, PPO, AWR). We show that this seemingly unstructured problem is not actually structureless. In log-density-ratio coordinates $y := \log[\pi/\pi_n]$, the exact per-iteration objective, computable via per-decision importance sampling (PDIS), is a difference-of-convex-constrained difference-of-convex (DC-constrained DC) program. This structure lets us move beyond surrogate approximations: it recovers CPI, NPG, TRPO, and AWR as special cases along interpretable axes, and it opens a multi-step axis $k$ that couples consecutive deci
    
[^353]: 无状态的运动：探究视频模型中的动力学一致性与条件化偏差

    Motion Without State: Probing Dynamical Consistency and Conditioning Bias in Video Models

    [https://arxiv.org/abs/2610.05135](https://arxiv.org/abs/2610.05135)

    该研究发现图像到视频模型的生成运动缺乏物理状态的前向传播，仅由局部视觉线索和全局边界控制构成，这种动力学一致性的系统性缺失会显著偏置定量物理评估的结果。

    

    图像到视频模型作为视觉世界先验，其看似合理的运动偏离了物理仿真。我们探究在生成运动中是什么样的动力学结构取代了前向状态传播，以及它如何使定量物理评估产生偏差。在保持物理定律和初始状态不变的情况下，我们在九个模型上改变可见几何结构、提示词措辞和片段长度，并将跟踪得到的轨迹与模拟器推演结果进行对比。生成的运动分裂为局部视觉线索和全局边界控制两个层面，二者之间缺少传播中的物理状态。在局部层面，经节拍归一化的速度跟随轨道起伏，并以15°至41°的相位超前指向最陡下降方向，而平均坡度几乎无法加速小球，这表明模型对过去坡度的记忆十分短暂。局部响应无法组合：没有任何单一的接触定律能够描述模型的反弹行为（可能是预设的、在单个偏转器接触上自校准的，或是样本内拟合的），并且小球能够越过49个挡板中的43个……

    arXiv:2610.05135v2 Announce Type: replace-cross  Abstract: Image-to-video models serve as visual world priors whose plausible motion departs from physical simulation. We ask what dynamical structure replaces forward state propagation in generated motion, and how it biases quantitative physics evaluation. Holding law and initial state fixed, we vary visible geometry, prompt wording, and clip length across nine models and compare tracked trajectories against simulator rollouts. Generated motion splits into local visual cues and global boundary control, omitting the propagating physical state between them. Locally, tempo-normalized speed follows track undulations with a 15{\deg}-41{\deg} phase lead toward steepest descent as mean slope barely accelerates the ball, showing short memory of past slope. Local responses fail to compose: no single contact law describes a model's bounces (prescribed, self-calibrated on single-deflector contacts, or fitted in-sample), and balls clear 43 of 49 bar
    
[^354]: InvestigationWorlds：一个用于法律调查的智能体环境

    InvestigationWorlds: An Agentic Environment for Legal Investigation

    [https://arxiv.org/abs/2610.04129](https://arxiv.org/abs/2610.04129)

    该论文利用真实的美国联邦法院简易判决动议构建了法律调查智能体环境InvestigationWorlds，评估发现智能体即使检索到相关证据也常常坚持错误假设，难以区分法院采纳的假设与其他备选假设。

    

    我们提出了InvestigationWorlds，一个用于法律调查的智能体环境。我们利用了美国民事诉讼中一种未被充分挖掘的材料：简易判决动议。该动议依赖于由真实证据材料组成的案卷记录，并产生一个法院采纳的假设，在裁决该动议时该假设被视为真实基准。每个环境均由从公共访问法院电子记录系统（PACER）检索到的真实美国联邦法院案例构建而成，并通过一个经律师验证的生成流程进行增强，该流程围绕原始案卷合成带有角色标签的文档。由此得到的语料库允许多种连贯的事实解读，其中只有一种与法院采纳的假设相符。在100个案例上的评估中，我们发现智能体尽管检索到了相关证据，却常常坚持错误的假设，难以将法院采纳的假设与其他备选假设区分开来。

    arXiv:2610.04129v2 Announce Type: replace  Abstract: We introduce InvestigationWorlds, an agentic environment for legal investigation. We build on an underused artifact of U.S. civil litigation: the summary judgment motion. This motion relies upon a record composed of real evidence exhibits, and results in a court-adopted hypothesis that is treated as ground truth for the purposes of deciding the motion. Each environment is built from a real U.S. Federal Court case retrieved from Public Access to Court Electronic Records (PACER) and augmented by an attorney-validated generation pipeline that synthesizes role-tagged documents around the original record. The resulting corpus admits multiple coherent factual readings, only one of which matches the court-adopted hypothesis. Evaluating on 100 cases, we find agents often commit to incorrect hypotheses despite retrieving relevant evidence, struggling to distinguish the court-adopted hypothesis from alternative hypotheses.
    
[^355]: 保持冷静（CALM）：分析文本到图像生成中全局不安全性的局限性

    Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation

    [https://arxiv.org/abs/2610.02300](https://arxiv.org/abs/2610.02300)

    本文揭示文本到图像生成中全局不安全防护存在覆盖与选择性的内在权衡，并提出免训练的CALM方法，通过提示词局部的反事实校正精准编辑违规词元表示并抑制不安全残余成分，在不损害良性提示词的前提下有效提升安全性。

    

    面向文本到图像生成的免训练防护方法通常依赖于一种可复用的安全信号，例如不安全方向或全局毒性子空间，并将其广泛地应用于各类提示词。我们对这一全局不安全性假设进行了受控的几何分析，揭示出一个一致的“覆盖-选择性”权衡：紧凑的不安全子空间无法覆盖异构的不安全语义，而更广泛的聚合则会日益扭曲与安全性相关的良性提示词。受此发现启发，我们提出了CALM（反事实自适应局部调制），这是一种免训练防护方法，用提示词局部的反事实校正取代统一的全局移除。利用匹配的不安全-良性锚点，CALM将每个提示词路由到活跃的不安全类别，仅对违规的词元表示进行最小化编辑使其偏向安全一侧，并抑制正向对齐的不安全残余成分。在广泛的评估中，CALM显著改善了对不安全内容的抑制（摘要在此处截断）。

    arXiv:2610.02300v1 Announce Type: new  Abstract: Training-free safeguards for text-to-image generation often rely on a reusable safety signal, such as an unsafe direction or global toxic subspace, applied broadly across prompts. We provide a controlled geometric analysis of this global-unsafety assumption and reveal a consistent coverage-selectivity trade-off: compact unsafe subspaces fail to cover heterogeneous unsafe semantics, whereas broader aggregation increasingly distorts safety-adjacent benign prompts. Motivated by this finding, we propose CALM (Counterfactual Adaptive Local Modulation), a training-free safeguard that replaces uniform global removal with prompt-local counterfactual correction. Using matched unsafe-benign anchors, CALM routes each prompt to active unsafe categories, minimally edits only violating token representations toward the safe side, and suppresses positively aligned unsafe residual components. Across broad evaluation, CALM significantly improves unsafe co
    
[^356]: BACKDROP：揭示智能体周遭环境世界让它付出的代价

    The Backdrop Exposes What the World Around an Agent Costs It

    [https://arxiv.org/abs/2609.38469](https://arxiv.org/abs/2609.38469)

    BACKDROP基准通过在智能体执行环境中植入权威覆盖、提示注入、边界越权和写入故障四种日常干扰，并保持任务指令不变，揭示了智能体性能在动态真实环境中从69.5%骤降至31.3%的严重衰减。

    

    智能体基准测试是在静止不变的世界中评测智能体，而实际部署的智能体却工作在一个他人也会不断改变的世界里。比如有人发短信让智能体把钱转去别处，或者订单确认信息要求它回复门禁码。我们提出BACKDROP，用以探究智能体在纯净环境中的能力有多少能在这样的环境中留存。BACKDROP以任务及智能体的执行环境为基础，在其世界中植入四种日常危害，包括一次一个以及全部同时出现，而指令和正确的最终状态保持不变。每种危害提出一个问题：权威性，来自他人的消息是否会覆盖用户的指令；注入，植入记录中的文本是否会误导智能体；边界，某个请求是否会将它引入未被授权的应用；故障，当一次写入操作失败且未说明是否成功落地时，智能体是否会在重试前先进行检查。在3,678个变体和16个模型上，平均通过率从69.5%下降至31.3%。

    arXiv:2609.38469v1 Announce Type: cross  Abstract: Agent benchmarks test agents in worlds that stay still. Deployed agents work in worlds that other people also change. Someone texts the agent to send the money elsewhere or an order confirmation asks it to reply with a door code. We present BACKDROP, which asks how much of an agent's capability in a clean world survives in such a world. BACKDROP takes a task along with the agents execution environment, and plants four everyday hazards in its world, one at a time and all together. The instruction and the correct end state stay the same. Each hazard asks one question. Authority: does a message from another person override the user? Injection: does text planted in a record redirect the agent? Boundary: does a request pull it into an app it was not given? Fault: after a write fails without saying whether it landed, does the agent check before it retries? Across 3,678 variants and 16 models, , the average pass rate falls from 69.5% to 31.3%
    
[^357]: ContextAdapt：评估大语言模型中的情境适应与价值对齐

    ContextAdapt: Evaluating Contextual Adaptation and Value Alignment in LLMs

    [https://arxiv.org/abs/2609.38260](https://arxiv.org/abs/2609.38260)

    提出了ContextAdapt评估框架，通过基于职业与监管一手文件构建的“价值观×领域”场景，考察12个大语言模型能否在医学、法律、金融和国家安全等领域恰当调整诚实、自主、保密等价值观的应用，并在规范未变时保持一致。

    

    诸如诚实、自主和保密等价值观常被视为支撑AI对齐的一般原则。然而，如何依照这些价值观行事，可能取决于做出决策时所处的具体情境。本文探究大语言模型（LLM）能否在跨越不同专业场景时恰当地调整价值观的应用方式，同时当情境变化并未改变相关职业规范时保持行为的一致性。为此，我们提出了ContextAdapt，一个覆盖医学、法律、金融和国家安全四大领域中诚实、自主与保密价值观的评估框架。基于一手专业与监管文件，我们构建了“价值观×领域”框架，并据此设计了测试默认职业规则及公认例外情形的多种场景。我们从模型推荐的行为及其给出的理由两个维度对12个大语言模型进行了评估。在我们的主要（评估中）

    arXiv:2609.38260v1 Announce Type: cross  Abstract: Values such as honesty, autonomy, and confidentiality are often regarded as general principles underpinning AI alignment. However, what it means to act in accordance with these values can depend on the context in which a decision is made. In this paper, we ask whether large language models (LLMs) appropriately adapt the application of a value across professional settings, while remaining consistent when contextual changes do not alter the relevant professional norm. To study this, we introduce ContextAdapt, an evaluation framework covering honesty, autonomy, and confidentiality across medicine, law, finance, and national security. Drawing on primary-source professional and regulatory documents, we construct a value x domain framework and use this to develop scenarios testing both default professional rules and recognised exceptions. We evaluate 12 LLMs on both the actions they recommend and the justifications they provide. In our main 
    
[^358]: 诊断与改进大语言模型中的概率推理

    Diagnosing and Improving Probabilistic Reasoning in Large Language Models

    [https://arxiv.org/abs/2609.38005](https://arxiv.org/abs/2609.38005)

    该论文提出一个决策理论框架，将大语言模型的决策损失分解为信念形成与行动选择两个组成部分，并通过合成基准和强化学习干预系统诊断并改进大语言模型的概率推理能力。

    

    大语言模型（LLM）越来越多地被提议作为决策助手，它们必须在明确的决策成本下基于现有证据进行概率推理。我们提出了一个决策理论框架，将LLM的决策损失分解为两个组成部分：从提供的证据中形成准确的信念，以及将这些信念转化为能优化给定效用函数的行动。利用一个具有已知真值的合成基准，我们应用该分解来刻画前沿模型和开源模型的概率推理能力。我们进一步评估了针对信念、决策或两者的强化学习干预能否在三个领域中改进这些组成部分，改进能否在不同组成部分和引出格式之间迁移，以及决策性能能否在信念形成未获改进的情况下得到提升。我们发现，针对概率推理的单一组成部分会重新分配决策损失，改进……（摘要截断）

    arXiv:2609.38005v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly proposed as decision assistants who must reason probabilistically from available evidence under explicit decision costs. We propose a decision-theoretic framework that decomposes LLMs' decision loss into two components: forming accurate beliefs from provided evidence and translating those beliefs into actions that optimize a provided utility function. Using a synthetic benchmark with known ground truth, we apply the decomposition to characterize probabilistic reasoning in frontier and open-sourced models. We further evaluate whether RL interventions targeting beliefs, decisions, or both improve these components across three domains, whether improvements transfer across components and elicitation formats, and whether decision performance can improve without improvement in belief formation. We find that targeting one component of probabilistic reasoning redistributes decision loss, improving th
    
[^359]: 解决状态表示失配问题：面向开环VLA规划的状态空间视觉推理

    Resolving State-Representation Mismatch: State-Space Visual Reasoning for Open-Loop VLA Planning

    [https://arxiv.org/abs/2609.33412](https://arxiv.org/abs/2609.33412)

    提出状态空间视觉推理（SSVR），通过将静态视觉上下文、语言约束与由动作条件GRU更新的循环潜在状态解耦，解决了开环VLA规划中的状态表示失配问题，显著提升了长期规划的可靠性。

    

    尽管视觉-语言-动作（VLA）模型取得了快速进展，现有推理范式在开环规划中仍面临一个根本性的状态表示失配问题。在仅有初始观测的条件下，模型必须在内部模拟动作条件下的状态转移，而文本空间、像素空间和潜在空间推理分别会遭受有损空间压缩、误差累积的视觉生成以及绕过中间潜在词元等问题，从而削弱了可靠的长期规划能力。我们提出状态空间视觉推理（SSVR），将静态视觉上下文、语言约束和循环潜在状态进行解耦。SSVR仅需对初始图像和指令编码一次，之后每次动作预测均以潜在状态为条件，并通过动作条件GRU对其进行更新。以Qwen2.5-VL作为骨干网络，SSVR在FrozenLake、Maze和MiniBe数据集上分别取得了99.5/99.6、96.3/98.0和83.9/90.6的EM/PR成绩。

    arXiv:2609.33412v2 Announce Type: replace-cross  Abstract: Despite rapid progress in vision-language-action (VLA) models, existing reasoning paradigms still face a fundamental \emph{state-representation mismatch} in open-loop planning. Given only an initial observation, models must internally simulate action-conditioned state transitions, whereas text-, pixel-, and latent-space reasoning can suffer from lossy spatial compression, error-accumulating visual generation, and bypass of intermediate latent tokens, respectively, undermining reliable long-horizon planning. We propose \textbf{State-Space Visual Reasoning} (SSVR), which decouples static visual context, language constraints, and a recurrent latent state. SSVR encodes the initial image and instruction once, then conditions each action prediction on the latent state and updates it with an action-conditioned GRU. Using Qwen2.5-VL as the backbone, SSVR achieves 99.5/99.6, 96.3/98.0, and 83.9/90.6 EM/PR on FrozenLake, Maze, and MiniBe
    
[^360]: 少样本演示激发大语言模型使用上下文世界表征

    Few-Shot Demonstrations Elicit the Use of In-Context World Representations in LLMs

    [https://arxiv.org/abs/2609.24352](https://arxiv.org/abs/2609.24352)

    少样本演示能够激发大语言模型在图追踪任务中构建并利用上下文世界表征，从而显著提升其预测性能。

    

    大语言模型（LLM）在作为智能体行动时，需要根据上下文中观察到的数据，推断世界背后的潜在状态空间，并将其用于下游预测。然而，先前的研究表明，LLM 在图追踪任务中难以使用在上下文中学习到的表征，在该任务中，模型需要构建一个关于数据生成过程所遵循的图的表征，并将其用于后续预测。在本文中，我们表明，将此扩展到少样本设置——其中每个演示都由一个不同的世界生成，这些世界具有相同或不同的图拓扑结构——能够提升来自 4 个模型家族的 6 个模型的预测性能。为了理解这一改进，我们对隐藏状态中编码图信息的低维世界表征进行了线性探测。值得注意的是，我们发现少样本演示会重新定位世界表征并增加其预测用途。

    arXiv:2609.24352v1 Announce Type: new  Abstract: Large language models (LLMs), when acting as agents, are expected to take observed data in context, infer the latent state space underlying the world, and leverage it for downstream prediction. However, prior work demonstrated that LLMs struggle to use representations learned in context on a graph tracking task, where the model needs to construct a representation of the graph governing data generation process and use it for subsequent predictions. In this paper, we show that extending this to few-shot settings, where each demonstration is generated from a different world with either the same or different graph topologies, enhances its prediction on 6 models from 4 model families. To understand this improvement, we linearly probe a low-dimensional world representation that encodes graph information in the hidden states. Notably, we find that few-shot demonstrations relocate the world representation and increase its predictive use. Specifi
    
[^361]: RLVR²：基于可验证评分量规排序的强化学习

    RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking

    [https://arxiv.org/abs/2609.23457](https://arxiv.org/abs/2609.23457)

    本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。

    

    基于可验证奖励的强化学习（RLVR）正在从具有明确定义正确性信号的任务（如数学和代码）扩展到由多维度评分量规指定的多方面质量要求。由于策略优化在每次 rollout 中只消耗一个标量，基于评分量规的流程必须将多个标准的得分聚合为标量奖励。这种聚合通常被视为简单的分数缩放，但它实际上隐式地决定了训练过程中各质量维度之间的权衡方式。目前主流的做法是对每个标准进行归一化后取线性组合，这隐含地假设了基数分数差异在各标准之间是可比的，并且某一标准上的收益可以补偿另一标准上的失败；当各标准在语义上异质时，这两种假设都不可靠。我们提出了基于可验证评分量规排序的强化学习（RLVR²），这是一种面向基于评分量规的 RLVR 的可验证排序范式。对于每个标准……

    arXiv:2609.23457v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is expanding from tasks with well-defined correctness signals, such as mathematics and code, toward multifaceted quality requirements specified by multi-dimensional rubrics. Since policy optimization consumes one scalar per rollout, rubric-based pipelines must map multiple criterion scores into a scalar reward. This aggregation is often treated as score scaling, but it implicitly determines how quality dimensions trade off during training. The prevailing practice, normalizing each criterion and taking a linear combination, assumes that cardinal score differences are comparable across criteria and that gains on one criterion compensate for failures on another; both assumptions are unreliable when criteria are semantically heterogeneous. We propose Reinforcement Learning with Verifiable Rubric-based Ranking (RLVR$^2$), a verifiable ranking paradigm for rubric-based RLVR. For each criter
    
[^362]: Designer-RSI：从用户流量中演化程序性记忆以实现智能体化图形设计

    Designer-RSI: Evolving Procedural Memory from User Traffic for Agentic Graphic Design

    [https://arxiv.org/abs/2609.22086](https://arxiv.org/abs/2609.22086)

    该论文提出Designer-RSI持续适应框架，让冻结的前沿模型通过230多个工具操作专业设计软件，并利用外部程序性记忆从真实用户流量中不断积累和精炼自然语言设计技能，在无权重更新、无人工标注的情况下显著提升智能体图形设计能力。

    

    专业图形设计是一项长时程的智能体任务，其中结构化、可编辑的作品由许多相互依赖的操作逐步产生，然而其结果缺乏可靠的程序化评估标准。我们提出一个持续适应框架：一个冻结的前沿模型通过230多个工具操作专业设计软件，同时一个由自然语言技能构成的外部程序性记忆从经验中积累并精炼可复用的设计流程。该记忆通过为反复出现的未覆盖子任务获取新流程而“变宽”，并通过依据流程自身成功与失败的执行记录修订现有流程而“变深”；同时，一个匹配重放门控机制只允许采纳那些能修复失败且不使已观测成功出现退化的变更。经过五轮迭代，基于1,406个真实用户需求和1,869条自动评分的轨迹，在不更新模型权重、无需人工标注的情况下，技能库从76个源自文档的技能增长到139个，并提升了GenEva……

    arXiv:2609.22086v1 Announce Type: new  Abstract: Professional graphic design is a long-horizon agentic task in which structured, editable artifacts emerge from many interdependent actions, yet outcomes admit no reliable programmatic oracle. We introduce a continual adaptation framework in which a frozen frontier model operates professional design software through more than 230 tools, while an external procedural memory of natural-language skills accumulates and refines reusable design procedures from experience. The memory widens by acquiring procedures for recurring uncovered subtasks and deepens by revising existing procedures against their own successful and failed executions, while a matched replay gate admits only changes that repair failures without regressing observed successes. Five rounds over 1,406 real user briefs and 1,869 automatically graded trajectories, with no weight updates and no human labels, grow the bank from 76 documentation-derived skills to 139 and raise GenEva
    
[^363]: SWE-Proof：语言模型能否通过机器校验的证明解决真实世界的问题？

    SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?

    [https://arxiv.org/abs/2609.21190](https://arxiv.org/abs/2609.21190)

    该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。

    

    确保大语言模型（LLM）生成代码的正确性是现代软件工程的核心挑战。面向智能体代码生成的基准测试通常使用留出的测试套件来检验正确性，但测试套件本质上是不完整的，且日益容易受到模型记忆（数据泄露）的影响。形式化验证可以同时避免这两个问题，但现有工作仅覆盖规范以输入形式给出的独立任务，而非真实问题——真实问题涉及大型代码仓库，并以模糊的自然语言表达意图。我们提出了Benchproofer，一个能将带有已知正确补丁的编码任务转化为形式化验证任务的流水线：它为新代码编写规范，用公理概括新代码所调用的已有函数，并且只有在机械验证与对抗性检查两道关卡均通过后才接受一个实例。将该流水线应用于SWE-bench Verified，我们得到了SWE-Proof——包含500个真实问题的基准，其正确性通过形式化验证而非测试来保证，并且该方法还可扩展至SWE-bench Pro。……

    arXiv:2609.21190v1 Announce Type: cross  Abstract: Ensuring the correctness of LLM-generated code is a core challenge for modern software engineering. Benchmarks for agentic code generation check correctness with held-out test suites, which are inherently incomplete and increasingly susceptible to memorization. Formal verification avoids both problems, but existing work covers only standalone tasks whose specifications are given as input, not real issues, which touch large repositories and state intent in vague natural language. We present Benchproofer, a pipeline that turns a coding task with a known correct patch into a formally verified one: it writes a specification for the new code, summarizes the existing functions that code calls with axioms, and admits an instance only after mechanical and adversarial gates agree. Applying it to SWE-bench Verified yields SWE-Proof, 500 real issues whose correctness is formally verified rather than tested, and it extends to SWE-bench Pro. Across
    
[^364]: PointEvent：通过序列化运动证据积累重新思考基于事件的微小目标检测

    PointEvent: Rethinking Event-based Tiny Object Detection via Serialized Motion Evidence Accumulation

    [https://arxiv.org/abs/2609.20066](https://arxiv.org/abs/2609.20066)

    提出PointEvent框架，通过序列化运动证据积累将运动连续性建模为有序的证据传播过程，有效解决事件相机在微小无人机检测中远距离目标事件稀疏碎片化、易被杂波淹没的问题。

    

    事件相机为微小无人机检测提供了高时间分辨率和运动敏感性，但远距离目标产生的稀疏且碎片化的事件容易被杂波和自身运动所淹没。现有方法主要依赖密集事件表示或局部稀疏时空建模，导致计算冗余，或对远距离异步事件之间的运动连续性进行碎片化建模。为解决这一局限，我们引入了序列化运动证据积累方法，将运动连续性视为有序的证据传播过程。具体而言，通过潜在的互补序列化方式，将同一事件流组织为保持局部性的时空路径和保持时序性的时间路径。基于这一原理，我们提出了PointEvent，一个轻量级的事件级状态空间框架，它在互补顺序之间交替进行序列化扫描，逐步巩固……

    arXiv:2609.20066v1 Announce Type: cross  Abstract: Event cameras offer high temporal resolution and motion sensitivity for tiny UAV detection, yet distant targets generate sparse and fragmented events that are easily overwhelmed by clutter and ego-motion. Existing methods mainly rely on dense event representations or local sparse spatiotemporal modeling, resulting in redundant computation or fragmented modeling of motion continuity across distant asynchronous events. To address this limitation, we introduce serialized motion evidence accumulation, which treats motion continuity as an ordered evidence propagation process. Specifically, the same event stream is organized into locality-preserving spatiotemporal paths and chronology-preserving temporal paths through the latent complementary serializations. Based on this principle, we propose PointEvent, a lightweight event-wise state-space framework that alternates serialized scans across the complementary orders, progressively consolidati
    
[^365]: 探索扩散Transformer在多模态脑状态解码中的跨模态数据增强应用

    Exploring Diffusion Transformers for Cross-Modal Augmentation in Multimodal Brain State Decoding

    [https://arxiv.org/abs/2609.11341](https://arxiv.org/abs/2609.11341)

    本文提出CoMA-DiT，一种双向跨模态扩散Transformer，将成对模态视为相互生成监督的来源以实现潜在空间数据增强，在多模态脑状态解码任务上显著优于20个代表性基线方法。

    

    多模态脑状态解码研究主要集中于融合成对模态进行预测，但很少探索如何进一步利用模态间的对应关系来丰富训练数据并改进多模态表征学习。为填补这一空白，我们提出了CoMA-DiT，一种用于潜在增强的双向跨模态扩散Transformer，它将成对模态视为相互生成监督的来源，而不仅仅是待融合的输入。CoMA-DiT通过跨模态注意力机制使速度预测以成对模态为条件，并通过可靠性门控残差机制自适应地注入由此产生的变化。在多模态听觉注意力解码和情绪识别任务上的实验表明，CoMA-DiT始终优于20个代表性基线方法，相比无增强基线，在准确率和宏F1上分别取得了4.28%和6.70%的绝对提升。

    arXiv:2609.11341v1 Announce Type: new  Abstract: Multimodal brain state decoding has largely focused on fusing paired modalities for prediction, but has rarely explored how their correspondence can be further exploited to enrich training data and improve multimodal representation learning. To address this gap, we propose CoMA-DiT, a bidirectional cross-modal Diffusion Transformer for latent augmentation that treats paired modalities as sources of mutual generative supervision rather than merely as inputs to be fused. CoMA-DiT conditions velocity prediction on the paired modality through cross-modal attention and adaptively injects the resulting variation via a reliability-gated residual mechanism. Experiments on multimodal auditory attention decoding and emotion recognition showed that CoMA-DiT consistently outperformed 20 representative baselines, achieving absolute gains of 4.28% and 6.70% in accuracy and macro-F1 over the no-augmentation baseline, respectively. Extensive ablation, s
    
[^366]: Sinkhorn层中的支撑拓扑与梯度混合

    Support Topology and Gradient Mixing in Sinkhorn Layers

    [https://arxiv.org/abs/2609.07954](https://arxiv.org/abs/2609.07954)

    本文发展了一套固定支撑演算来分析Sinkhorn层中的梯度传播机制，并证明了传输多面体的每个可行面具有成对两跳列重叠是支撑图与边际保证梯度均匀一步收缩的充要条件。

    

    稀疏Sinkhorn层使用固定的支撑图来限制token之间的传输。该图如何控制缩放迭代中的梯度传播？我们发展了一套固定支撑演算，证明在模常数意义下，每个行-列循环会对列势扰动诱导出一个行随机算子。其转置用于传播零质量的反向模式余切。有限循环算子使用两个不同的半步传输计划；在平衡不动点处，它退化为由单一计划确定的两步游走。我们推导了相应的得分项和边际源项，并利用Dobrushin收缩与minorization方法来界定齐次和源驱动的尾部余切。我们的主要结果刻画了支撑和边际在有限得分上一致保证一步收缩的条件：传输多面体的每个可行面必须具有成对的两跳列重叠。否则，某些合适的得分方向（原文此处截断）……

    arXiv:2609.07954v1 Announce Type: new  Abstract: Sparse Sinkhorn layers use a fixed support graph to restrict transport between tokens. How does this graph control gradient propagation through the scaling iterations. We develop a fixed-support calculus showing that each row-column cycle induces a row-stochastic operator on column-potential perturbations modulo constants. Its transpose propagates zero-mass reverse-mode cotangents. The finite-cycle operator uses two distinct half-step transport plans; at a balanced fixed point it reduces to a two-step walk determined by a single plan. We derive the accompanying score and marginal source terms and use Dobrushin contraction and minorization to bound homogeneous and source-driven tail cotangents. Our main result characterizes when support and marginals guarantee one-step contraction uniformly over finite scores: every feasible face of the transportation polytope must have pairwise two-hop column overlap. Otherwise, suitable score directions
    
[^367]: 具有噪声约束值的约束在线学习

    Constrained Online Learning with Noisy Constraint Values

    [https://arxiv.org/abs/2609.06921](https://arxiv.org/abs/2609.06921)

    针对约束值受噪声污染的约束在线凸优化，本文证明了联合次线性保证的不可能性，并提出LEDGER算法，通过引入预算违反新度量，实现了 $O(\sqrt T/V)$ 的期望遗憾和可控的预算违反保证。

    

    我们研究具有对抗性约束的约束在线凸优化问题，其中约束值和梯度是通过无偏噪声观测得到的。标准差为 $\sigma$ 的高斯值噪声会导致期望遗憾与期望硬违反的最大值在最坏情况下存在 $\Omega(\min\{\sigma,1\}T/\log^7T)$ 的下界，即使梯度已知也是如此。这排除了在固定 $\delta>0$ 和固定正噪声水平下任何联合的 $O(T^{1-\delta})$ 保证。因此，我们转而研究预算违反：即固定时间范围内任意窗口上的最大累积超支。我们提出了 LEDGER 算法，该算法在非负余额中跟踪观测到的净消耗，并在当前反馈噪声产生之前设置约束权重。在常见的可行性条件和条件有限方差反馈下，对于固定的问题参数，LEDGER 算法实现了 $O(\sqrt T/V)$ 的期望遗憾和 $O(\sqrt V\,T^{3/4}+\sigma\sqrt T)$ 的期望预算违反，其中 $V$……

    arXiv:2609.06921v1 Announce Type: cross  Abstract: We study constrained online convex optimization with adversarial constraints when constraint values and gradients are observed through unbiased noise. Gaussian value noise of standard deviation $\sigma$ yields a worst-case lower bound of $\Omega(\min\{\sigma,1\}T/\log^7T)$ on the maximum of expected regret and expected hard violation, even with known gradients. This rules out any jointly $O(T^{1-\delta})$ guarantee for fixed $\delta>0$ and fixed positive noise level. We therefore study budget violation: the largest cumulative overspend over any window within a fixed horizon. We introduce \LEDGER, which tracks observed net consumption in a nonnegative balance and sets constraint weights before the current feedback noise. Under common feasibility and conditional finite-variance feedback, for fixed problem parameters, \LEDGER\ achieves $O(\sqrt T/V)$ expected regret and $O(\sqrt V\,T^{3/4}+\sigma\sqrt T)$ expected budget violation for $V\
    
[^368]: 从合成先验到模型行为：表格基础模型中的结构覆盖度

    From Synthetic Priors to Model Behavior: Structural Coverage in Tabular Foundation Models

    [https://arxiv.org/abs/2609.06912](https://arxiv.org/abs/2609.06912)

    该论文从分布层面归因视角出发，通过结构覆盖度和归一化密度量化表格基础模型的合成预训练先验对下游基准任务的支持范围，并探究这种结构支持与预测性能之间的关联。

    

    表格基础模型（TFMs）通常在大量程序化生成的合成任务集合上进行预训练，然而这些合成预训练先验对模型所评估的下游任务的支持程度仍不清楚。我们从分布层面的归因视角来研究这一问题。我们恢复或重构了四个表格基础模型的合成数据生成器，并将其生成的任务与两个广泛使用的表格基准数据集进行比较。每个数据集都由一组共同的结构描述符表示，这些描述符涵盖模式、特征分布、依赖结构、响应属性以及特征-响应关系。在该空间中，我们使用结构覆盖度和归一化密度来衡量每个合成先验到达基准任务的广度与重复程度，并检验更强的局部支持是否与更好的预测性能相关联。我们发现存在显著差异……（原文摘要至此被截断）

    arXiv:2609.06912v1 Announce Type: cross  Abstract: Tabular foundation models (TFMs) are commonly pretrained on large collections of procedurally generated synthetic tasks, yet it remains unclear how well these synthetic pretraining priors support the downstream tasks on which the models are evaluated. We study this question from a distribution-level attribution perspective. We recover or reconstruct the synthetic data generators of four TFMs and compare their generated tasks with datasets from two widely used tabular benchmarks. Each dataset is represented by a common set of structural descriptors capturing schema, feature distributions, dependence structure, response properties, and feature--response relationships. In this space, we measure how broadly and repeatedly each synthetic prior reaches benchmark tasks using structural coverage and normalized density, and examine whether stronger local support is associated with better predictive performance. We find substantial differences a
    
[^369]: ARC-Bench：闭环重规划掩盖了冻结JEPA世界模型中失效的动作排序

    ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in Frozen JEPA World Models

    [https://arxiv.org/abs/2609.05461](https://arxiv.org/abs/2609.05461)

    该论文提出ARC-Bench无泄漏审计协议，揭示冻结JEPA世界模型中“潜在距离可正确排序动作”的核心假设严重且结构性失效，且该缺陷在不同视觉骨干网络下均持续存在。

    

    无奖励的潜在世界模型通过在冻结的潜在空间中以距离对候选动作进行评分来进行规划：如果一个动作预测的未来嵌入更接近目标嵌入，则该动作会被优先选择。这悄然假设了潜在接近度是可对动作进行排序的，即按潜在距离对候选动作排序与按真实代价排序是一致的。我们直接审计了这一假设。我们提出了ARC-Bench，一种无泄漏、固定候选集的评测协议，用于衡量冻结的JEPA式目标能否正确排序候选动作，并将其应用于官方发布的JEPA-WM检查点，涵盖导航和操作类控制任务。这一假设严重且结构性地失效了：在官方的操作任务审计中，得分最高的候选动作几乎总是次优的，同样的排序倒置现象也出现在迷宫环境中。一项受控的视觉骨干网络扩展实验表明，当将DINOv2替换为视频预训练的V-JEPA 1时，该缺陷依然存在。

    arXiv:2609.05461v1 Announce Type: new  Abstract: Reward-free latent world models plan by scoring candidate actions with distances in a frozen latent space: an action is preferred if its predicted future embedding lands closer to the goal embedding. This silently assumes that latent closeness is action-rankable, i.e., that ordering candidates by latent distance agrees with ordering them by true cost. We audit this assumption directly. We introduce ARC-Bench, a no-leak, fixed-candidate protocol that measures whether frozen JEPA-style objectives rank candidate actions correctly, and apply it to official released JEPA-WM checkpoints across navigation and manipulation-style control. The assumption fails, severely and structurally: on the official manipulation audits the top-scored candidate is almost always suboptimal, and the same inversion appears in the maze domains. A controlled visual-backbone extension shows that the defect persists when DINOv2 is replaced by video-pretrained V-JEPA 1
    
[^370]: 剖析医学问答中误导性上下文的作用机制

    Untangling the Mechanisms of Misleading Context in Medical Question Answering

    [https://arxiv.org/abs/2609.02754](https://arxiv.org/abs/2609.02754)

    该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。

    

    大语言模型如今能够以专家级水平回答医学问题。然而，这些系统所依据的上下文可能具有误导性，而误导性上下文会破坏模型的医学判断。为了理解误导性上下文如何破坏这一判断，我们考察了模型对上下文的易感性、对误导信息的披露程度、推理被破坏的机制以及决策的可监控性。在一个由临床医生审核、包含8,627个问题的问答基准MedMisBench的医学推理子集上，我们注入了两类误导性上下文线索：伪造的证据和纯粹的断言。我们测试了三个推理模型，其中两个会暴露其完整的推理轨迹，另一个前沿模型仅暴露其最终回复。所有三个模型都更容易受纯粹断言而非伪造证据的影响，采用断言答案的频率高出10至27个百分点。误导性线索在81%至98%的推理轨迹中被披露，但仅……

    arXiv:2609.02754v1 Announce Type: cross  Abstract: Large language models now answer medical questions with expert-level performance. However, the context these systems act on can be misleading, and misleading context can corrupt a model's medical judgment. To understand how misleading context corrupts this judgment, we examine the model's susceptibility to the context, disclosure of it, mechanism of corrupted reasoning, and monitorability of the decision. On the medical reasoning subset of MedMisBench, a clinician-reviewed question-answering benchmark of 8,627 questions, we inject two types of misleading context cues, fabricated evidence and a bare assertion. We test three reasoning models, two that expose their full reasoning trace and one frontier model that exposes only its response. All three are more susceptible to the assertion than to the fabricated evidence, adopting the asserted answer 10 to 27 points more often. The misleading cues are disclosed in 81 to 98% of traces but onl
    
[^371]: AutoXRD：用于粉末衍射分析的自主大语言模型智能体与综合评估

    AutoXRD: Autonomous LLM Agents and Comprehensive Evaluation for Powder Diffraction Analysis

    [https://arxiv.org/abs/2609.00070](https://arxiv.org/abs/2609.00070)

    本文提出AutoXRD——一个将粉末XRD分析组织为基于证据的逐步精修并加入确定性晶体学物理检查的自主大语言模型智能体框架，同时推出包含100个诊断问答任务和34个端到端工作流的XRDBench基准，用于全面评估智能体的XRD分析能力。

    

    粉末X射线衍射（XRD）是材料表征的核心手段，然而可靠的端到端自动化仍然具有挑战性。一个XRD智能体必须解读衍射证据、操作精修软件、以合理的顺序管理耦合参数，并区分数值上的改进与物理上的有效性。在本文中，我们提出了AutoXRD，这是一个自主的大语言模型（LLM）智能体框架，它将粉末XRD分析组织为逐步精修的过程，使行动立足于观察到的证据，并在接受结果之前应用确定性的晶体学和物理检查。我们进一步引入了XRDBench基准，包含两个互补的赛道：XRDBench-QA包含100个有边界的诊断任务，用于隔离评估科学推理和决策能力；而XRDBench-E2E包含34个可执行的工作流，用于测试智能体能否将这些能力组合成需要文件检查、晶体学软件操作的完整分析。

    arXiv:2609.00070v1 Announce Type: cross  Abstract: Powder X-ray diffraction (XRD) is central to materials characterization, yet reliable end-to-end automation remains challenging. An XRD agent must interpret diffraction evidence, operate refinement software, manage coupled parameters in a defensible order, and distinguish numerical improvement from physical validity. In this paper, we propose AutoXRD, an autonomous large language model (LLM) agent framework that organizes powder-XRD analysis as stepwise refinement, grounds actions in observed evidence, and applies deterministic crystallographic and physical checks before accepting results. We further introduce XRDBench with two complementary tracks. XRDBench-QA contains 100 bounded diagnostic tasks that isolate scientific reasoning and decision-making, whereas XRDBench-E2E contains 34 executable workflows that test whether agents can compose these capabilities into complete analyses requiring file inspection, crystallographic-software 
    
[^372]: 潜在规划能否在点云中存活？面向几何观测的动作条件化JEPA世界模型

    Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations

    [https://arxiv.org/abs/2608.29434](https://arxiv.org/abs/2608.29434)

    该论文证明JEPA世界模型的潜空间规划可以从图像成功扩展到点云等几何观测——三种经典JEPA设计在点云上均能无坍塌地规划，其中动作敏感模型在几何变化最大的场景中表现最佳。

    

    JEPA世界模型使潜空间规划成为实现控制的一条实用途径，但它们几乎完全建立在图像之上。潜在预测能否在几何观测中存活尚不明确：点云稀疏、无序且存在自遮挡，当场景中仅有0.3-15%的点发生移动时，潜在预测的慢特征最优性与3D自监督的几何捷径会相互叠加。我们将三种经典的JEPA设计——冻结编码器、分布先验和动作敏感——提升到点云上，并重新感知稳定世界模型基准，使得观测方式成为与图像基线之间唯一的差异。三种模型均能在不发生表征坍塌的情况下进行规划：分布先验模型在所有基准上与其重新评估后的图像对应模型统计等效，而动作敏感模型在几何移动最多的受控比较中取得了最强结果。探测实验解释了原因：物体位置几乎可以被完美地线性解码。

    arXiv:2608.29434v1 Announce Type: cross  Abstract: JEPA world models make latent-space planning a practical route to control, but they are built almost exclusively on images. Whether latent prediction survives geometric observations is unclear: point clouds are sparse, unordered, and self-occluded, and with 0.3-15% of scene points moving, the slow-feature optimum of latent prediction compounds with the geometric shortcut of 3D self-supervision. We lift three canonical JEPA designs to point clouds, frozen-encoder, distribution-prior, and action-sensitive, and re-sense the stable-worldmodel benchmark so that only the observation differs from the image baselines. All three plan without collapse: the distribution-prior model is statistically equivalent to its re-evaluated image counterpart on every benchmark, and the action-sensitive model attains the strongest result in our controlled comparison where the most geometry moves. Probing explains why: object positions are almost perfectly lin
    
[^373]: 一表抵64词元：多表格文档问答中的像素级压缩

    A Table Is Worth 64 Tokens: Pixel-level Compression for Multi-Table Document Question Answering

    [https://arxiv.org/abs/2608.26949](https://arxiv.org/abs/2608.26949)

    本文发现像素级表格压缩在原生分辨率下性能与文本相当，但高度缩小后仍可识别表格相关性，据此提出无需训练的两步法以提高多表格文档问答效率。

    

    arXiv:2608.26949v1 公告类型：新 摘要：回答现实世界文档中的问题需要处理长输入，其中文本与表格交错。光学上下文压缩将上下文表示为图像，有望降低词元成本，但其对表格理解的影响仍不清楚。我们研究了多表格文档问答中的像素级表格压缩，在两个基准和五种视觉词元预算下评估了五种视觉语言模型。以原生分辨率将表格表示为图像，在性能和效率上与文本相当，但缩小它们会使模型用更长、效果较差的推理轨迹来补偿可读性的损失，从而抵消预期的节省。然而，高度缩小的表格保留了足够的信号来识别它们是否与问题相关。我们利用这种不对称性，提出一种无需训练的两步方法：模型首先从像素压缩的上下文中识别回答问题所需的表格。

    arXiv:2608.26949v1 Announce Type: new  Abstract: Answering questions over real-world documents requires processing long inputs that interleave text with tables. Optical context compression, which represents context as images, promises to reduce token cost, but its effect on table understanding remains unclear. We study pixel-level table compression for question answering over documents with multiple tables, evaluating five VLMs across two benchmarks and five visual-token budgets. Representing tables as images at native resolution matches text in both performance and efficiency, but downscaling them makes models compensate the loss in readability with longer, less effective reasoning traces that cancel the expected savings. Highly downscaled tables, however, preserve enough signal to identify whether they are relevant to a question. We exploit this asymmetry with a training-free, two-step method: the model first identifies the tables needed to answer a question from a pixel-compressed c
    
[^374]: 深度实现局部熵：深度变分范数ReLU回归中的二次深度依赖

    Depth Enables Local Entropy: Quadratic Depth Dependence in Deep Variation-Norm ReLU Regression

    [https://arxiv.org/abs/2608.17434](https://arxiv.org/abs/2608.17434)

    本文证明了在深度变分范数ReLU回归中，深度平方依赖是本质的，并通过局部填充和平衡放大技术给出了显式下界，填补了上下界之间的深度因子差距。

    

    我们研究了在显式向量值Parhi--Nowak深度RBV^2架构上的高斯回归，该架构具有深度L、宽度w、层和变分预算A和输出界B。对于这个O(L w^2)参数化的架构，已知的上下界相差一个深度因子。我们构建了一个局部填充，表明在显式依赖于样本大小的半径条件下，二次深度依赖是内在的。该填充的对数基数为Omega(L^2 w^2 log w)；其码字位于O(lambda) L^2球内，并且两两Omega(lambda)分离。主要成分是一个偏差校正的有界系数逼近定理和平衡放大：将深度为D的ReLU网络乘以q可以通过使用一个常数通道实现，使得每个系数仅增长q^(1/D)。转换为向量值RBV^2块后，层和代价为O(D w^2 q^(1/D))。高斯Fano方法给出了一个半径显式的下界，该下界受...

    arXiv:2608.17434v1 Announce Type: new  Abstract: We study Gaussian regression over the explicit vector-valued Parhi--Nowak deep-RBV^2 architecture with depth L, width w, layer-sum variation budget A, and output bound B. For this O(L w^2)-parameterized architecture, the known lower and upper bounds differ by one factor of depth. We construct a local packing showing that the quadratic depth dependence is intrinsic under an explicit sample-size-dependent radius condition. The packing has log-cardinality Omega(L^2 w^2 log w); its codewords lie in an O(lambda) L^2 ball and are pairwise Omega(lambda)-separated. The main ingredients are a bias-corrected bounded-coefficient approximation theorem and balanced amplification: multiplying a depth-D ReLU network by q can be implemented using one constant channel so that every coefficient grows by only q^(1/D). Translation to vector-valued RBV^2 blocks then has layer-sum cost O(D w^2 q^(1/D)). Gaussian Fano yields a radius-explicit lower bound gover
    
[^375]: DiSCO：通过分布引导的对比提示优化防御文本到图像生成

    DiSCO: Defending text-to-image generation through distribution-guided contrastive prompt optimization

    [https://arxiv.org/abs/2608.17067](https://arxiv.org/abs/2608.17067)

    DiSCO提出了一种零样本、严格黑盒的提示级防御方法，通过分布引导的对比优化解决良性对抗问题，无需模型内部访问即可防止NSFW内容生成。

    

    随着文本到图像生成模型的进步，它们引发了关键的安全问题，尤其是生成暴力、裸体等不宜工作（NSFW）内容，并且红色团队对抗攻击进一步加剧了这一问题。现有防御主要在白盒假设下运作，依赖文本编码器优化、权重编辑或推理时干预，从根本上无法扩展到专有模型。基于LLM提示重写的黑盒替代方案提供了更广泛的适用性，但在我们识别为“良性对抗”的关键场景中失败：即提示在语言上安全，但由于模型学习的数据分布，仍会触发有害生成。我们提出DiSCO，一种零样本、严格黑盒的防御方法，完全在提示级别作为即插即用模块运作，无需模型重训练、微调或访问模型内部。DiSCO执行分布引导的对比优化，以调整提示，使其在保持语义的同时避免触发有害生成。

    arXiv:2608.17067v1 Announce Type: new  Abstract: As text-to-image generative models advance, they raise critical safety concerns, particularly the generation of Not-Safe-For-Work (NSFW) content such as violence and nudity, further exacerbated by red-teaming adversarial attacks. Existing defenses predominantly operate under white-box assumptions, relying on text encoder optimization, weight editing, or inference-time intervention, and fundamentally cannot scale to proprietary models. Black-box alternatives based on LLM prompt rewriting offer broader applicability, yet fail in a critical regime we identify as the \textit{benign adversarial} problem: prompts that are linguistically safe but still trigger harmful generation due to the model's learned data distribution. We propose DiSCO, a zero-shot, strictly black-box defense that operates entirely at the prompt level as a plug-and-play module, requiring no model retraining, fine-tuning, or access to model internals. DiSCO performs distrib
    
[^376]: GEM：一种弥合推理与检索的生成式嵌入模型

    GEM: A Generative Embedding Model Bridging Reasoning and Retrieval

    [https://arxiv.org/abs/2608.13200](https://arxiv.org/abs/2608.13200)

    GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。

    

    arXiv:2608.13200v1 公告类型：交叉 摘要：现代大型语言模型在推理和指令遵循方面表现出色，使用户能够表达复杂多样的信息需求。然而，传统检索器主要依赖于查询和文档之间的表面匹配，导致用户表达需求的方式与检索器解读方式之间的差距日益扩大。在本文中，我们提出了GEM，一种生成式嵌入模型，通过显式推理用户意图和相关性标准，利用自身知识增强检索。GEM在单一模型中统一了生成和嵌入功能：它首先对查询进行推理，然后附加一个嵌入标记，以编码用于检索的增强上下文。在推理密集型和指令遵循型检索任务上的评估表明，GEM展示了其推理增强检索的有效性，优于其非推理变体，并匹配了使用更大模型的基线性能。此外，GEM的生成式嵌入能力使其能够灵活适应多样化的查询表达。

    arXiv:2608.13200v1 Announce Type: cross  Abstract: Modern LLMs excel at reasoning and instruction following, enabling users to express complex and diverse information needs. However, conventional retrievers largely rely on surface-level matching between queries and documents, resulting in a growing gap between how users express their needs and how retrievers interpret them. In this paper, we present GEM, a generative embedding model that augments retrieval through its own knowledge by explicitly reasoning about user intent and relevance criteria. GEM unifies generation and embedding within a single model: it first reasons over the query, then appends an embedding token to encode the enriched context for retrieval. \zhili{Evaluated on reasoning-intensive and instruction-following retrieval tasks, GEM demonstrates the effectiveness of its reasoning-augmented retrieval, outperforming its non-reasoning variant and matching baselines using substantially larger models.} Furthermore, GEM's ge
    
[^377]: 基于可观测语义图像接口与层级生成器证据对齐的生成式语义分割

    Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment

    [https://arxiv.org/abs/2608.11537](https://arxiv.org/abs/2608.11537)

    提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。

    

    生成式语义分割将结构化预测呈现为图像，但直接的颜色解码容易受到颜色漂移和边界混合的影响，而预测独立输出分布的潜在特征解码器可能将渲染图像降级为中间可视化。我们提出了Semantic Prism，一个具有确定性推理的条件语义图像生成与细化框架。一个基于扩散蒸馏的单步生成器渲染语义RGB图像；渲染颜色到固定类颜色码本的逐像素距离定义了一个显式的概率接口。层级生成器证据对齐在空间上对齐多层级生成器特征，并使用零初始化输出投影在接口对数空间中预测加性残差，保留图像定义的接口作为最终分布的参考。该接口和细化分布进一步支持Conte（后续内容）。

    arXiv:2608.11537v1 Announce Type: cross  Abstract: Generative semantic segmentation exposes structured predictions as images, but direct color decoding is susceptible to color drift and boundary mixing, whereas latent-feature decoders that predict a separate output distribution may relegate the rendered image to an intermediate visualization. We present Semantic Prism, a conditional semantic-image generation-and-refinement framework with deterministic inference. A diffusion-distilled one-step generator renders a semantic RGB image; per-pixel distances from the rendered colors to a fixed class-color codebook define an explicit probabilistic interface. Hierarchical Generator Evidence Alignment spatially aligns multi-level generator features and uses a zero-initialized output projection to predict an additive residual in the interface logit space, retaining the image-defined interface as the reference for the final distribution. The interface and refined distributions further enable Conte
    
[^378]: 超越统计先验：基于统一显式语义的架构无关无数据知识蒸馏

    Beyond Statistical Priors: Architecture-Agnostic Data-Free Knowledge Distillation via Unified Explicit Semantics

    [https://arxiv.org/abs/2608.09287](https://arxiv.org/abs/2608.09287)

    该论文提出UniDFKD框架，用显式的架构无关语义先验取代传统DFKD依赖的批归一化统计等架构特定先验，使无数据知识蒸馏能够有效应用于ViT等现代架构。

    

    无数据知识蒸馏（DFKD）通过合成语义信息丰富的数据，将知识从预训练的教师模型迁移到紧凑的学生模型，从而无需访问原始训练数据集。现有的DFKD方法严重依赖架构特定的统计先验（例如批归一化统计量）来指导数据合成；然而，这类依赖架构的先验在视觉Transformer（ViTs）等现代架构中往往不存在，导致合成数据的语义质量下降，进而引发灾难性的性能退化。在本文中，我们提出UniDFKD，一个统一的无数据知识蒸馏框架，用显式的、架构无关的语义先验取代架构特定的统计量。UniDFKD沿三个维度管控整个合成-蒸馏流程：（1）类别语义条件化（CSC）定义……（原文摘要到此截断）

    arXiv:2608.09287v2 Announce Type: replace-cross  Abstract: Data-Free Knowledge Distillation (DFKD) transfers knowledge from a pretrained teacher model to a compact student model by synthesizing semantically informative data, eliminating the need for access to the original training dataset. Existing DFKD methods rely heavily on architecture-specific statistical priors (e.g., Batch Normalization statistics) to guide data synthesis; however, such architecture-dependent priors are often absent in modern architectures such as Vision Transformers (ViTs), resulting in degraded semantic quality of the synthesized data and consequently catastrophic performance degradation. In this paper, we propose UniDFKD, a unified data-free knowledge distillation framework that replaces architecture-specific statistics with explicit, architecture-agnostic semantic priors. UniDFKD governs the entire synthesis-distillation pipeline along three dimensions: (1) Categorical Semantic Conditioning (CSC) defines wha
    
[^379]: Transformer革命（第一部分）：通过输出-权重互连实现的动态处理

    The Transformer Revolution, Part 1: Dynamic Processing through Output-Weight Interconnections

    [https://arxiv.org/abs/2608.03921](https://arxiv.org/abs/2608.03921)

    该论文以机械结构视角重新解读Transformer架构，揭示推理时部分网络的输出会动态决定其他网络的权重，由此产生随提示长度线性增长、甚至可超过静态参数数量的动态参数，以及序列级交互式动态并行处理（SIDPP）机制。

    

    我们通过为其功能架构开发一种功能等价的机械结构式描述，对Transformer的推理过程进行了重新解读。令牌表示的参数化变换，即变换概念，被识别为通过输出-输入互连和输出-权重互连组织起来的简单神经网络。这种重新描述使一个在标准矩阵描述中并不那么显著的组织特征变得清晰：在推理过程中，某些网络的输出决定了其他网络的权重，进而决定了这些网络的变换。这些输出-权重互连产生了依赖于提示词的动态变换，并由此产生了序列级交互式动态并行处理（SIDPP）。我们证明，动态参数的数量随提示词长度线性增长，并可能达到与训练中固定的静态参数数量相当、甚至超过后者的规模，我们将这一现象称为强提示依赖性。

    arXiv:2608.03921v3 Announce Type: replace  Abstract: We reinterpret Transformer inference by developing a functionally equivalent mechanical-structural description of its functional architecture. Parameterized transformations of token representations, or transforming concepts, are identified with simple neural networks organized through output-input and output-weight interconnections. This redescription makes explicit an organizational feature that is not equally salient in the standard matrix description: during inference, the outputs of some networks determine the weights, and hence the transformations, of others. These output-weight interconnections generate prompt-dependent dynamic transformations and give rise to Sequence-level Interactive Dynamic Parallel Processing (SIDPP). We show that the number of dynamic parameters grows linearly with prompt length and may become comparable to, or exceed, the number of static parameters fixed through training, a phenomenon we call strong pro
    
[^380]: 风险生意：测量忠实性与安全性之间的张力

    Risky Business: Measuring The Faithfulness-Safety Tension

    [https://arxiv.org/abs/2608.03745](https://arxiv.org/abs/2608.03745)

    本文揭示了大型推理模型中“必须忠实以便被监控”与“必须稳健以拒绝不安全推理”之间的对齐张力，并提出 HazMart 数据集和目标推理替换（TRR）技术来直接干预推理链以测量和应对这一张力。

    

    链式思维推理为模型监控提供了一个有前景的窗口。然而，监控依赖于忠实性，即模型输出严格来源于其推理轨迹。我们识别出一种对齐张力：模型必须足够忠实以便被监控，同时又必须足够稳健以拒绝不安全的推理。我们证明这种相互制约的关系存在于当前的大型推理模型中，并展示了可以解决该问题的方法。我们介绍了 HazMart，一个设置在自主 AI 店主场景下的人工撰写数据集。与以往依赖在提示中提供暗示来测试忠实性的工作不同（例如，“一位斯坦福教授说答案应该是 A”），我们提出了一种新颖的基于替换的技术，称为目标推理替换（TRR），它直接干预推理链，将不安全或不合逻辑的想法替换进去（例如，“等等，答案必须是选项 B [原为选项 A]，因为..."）。

    arXiv:2608.03745v2 Announce Type: replace  Abstract: Chain-of-Thought (CoT) reasoning offers a promising window into model monitoring. However, monitoring relies on faithfulness, i.e., the model output strictly derives from its reasoning trace. We identify an alignment tension where a model must be faithful enough to be monitored, yet robust enough to reject unsafe reasoning. We demonstrate that this counterbalance exists in current Large Reasoning Models (LRMs), and show ways in which it can be addressed. We introduce HazMart, a human-written dataset set in an autonomous AI shopkeeper scenario. Unlike prior work that relies on providing hints in prompts to test faithfulness (e.g., "A Stanford professor said it should be Answer A"), we propose a novel replacement-based technique, which we call Targeted Reasoning Replacement (TRR), that directly intervenes in the reasoning chain to substitute in unsafe or illogical thoughts (e.g., "Wait, the answer must be Option B [was Option A] becaus
    
[^381]: CrossProjection：跨投影图与剖面图的空间定位基准测试

    CrossProjection: Benchmarking Spatial Grounding across Projection and Section Views

    [https://arxiv.org/abs/2608.00473](https://arxiv.org/abs/2608.00473)

    该论文提出了CrossProjection基准，利用建筑图纸测试视觉语言模型在跨视图匹配、投影配准和新剖面预测等空间定位任务上的能力，并首次引入了接受专家裁定的兼容位置带内任意剖面切线的逆向剖面定位评估协议。

    

    平面图、立面图和剖面图通过不完整的投影来描述三维结构，这些投影不仅改变了观察实体的视角，也改变了实体被绘制的方式。阅读这些图纸意味着判断哪些实体在不同视图之间相互对应，并恢复它们所处的位置。我们提出了CrossProjection，这是一个使用建筑图纸来测试视觉语言模型（VLM）能否将对应关系决策转化为几何答案的基准。其任务涵盖跨视图匹配、投影配准和新剖面预测，输出形式从判断和候选选择到自由区域框和有向剖面切线。对于逆向剖面定位，据我们所知，我们引入了首个VLM基准协议，该协议接受专家裁定的兼容位置带内的任意切线，并单独检查观察方向。人工标注的图纸和BIM导出的视图提供了33,788个问题。

    arXiv:2608.00473v2 Announce Type: replace-cross  Abstract: Plans, elevations, and sections describe a three-dimensional structure through incomplete projections that change how an entity is drawn, not only where it is seen from. Reading them means deciding which entities correspond across views and recovering where they lie. We introduce CrossProjection, a benchmark that uses architectural drawings to test whether vision-language models (VLMs) can turn correspondence decisions into geometric answers. Its tasks cover cross-view matching, projection registration, and novel-section prediction, with outputs ranging from judgments and candidate selection to free region boxes and directed section cuts. For inverse section localization, we introduce, to our knowledge, the first VLM benchmark protocol that accepts any cut within an expert-adjudicated band of compatible positions and checks the viewing direction separately. Manually annotated drawings and BIM-derived views provide 33,788 questi
    
[^382]: 重尾噪声下的无参数动态遗憾

    Parameter-Free Dynamic Regret under Heavy-Tailed Noise

    [https://arxiv.org/abs/2607.27073](https://arxiv.org/abs/2607.27073)

    该论文提出了一种无需预知噪声矩参数、Lipschitz 界、梯度尺度及比较序列路径长度的在线凸优化算法，通过重启 AdaGrad 专家与自适应熵正则化主算法的结合，在重尾噪声下实现了具有噪声路径指数 $(p-1)/p$ 的最优通用动态遗憾界。

    

    我们研究在线凸优化问题，其中随机梯度噪声的条件 $p$ 阶中心矩被 $\sigma^p$ 所界定，而 $p\in(1,2]$ 是未知的。对于在直径为 $D$ 的定义域上具有 Lipschitz 界 $G$ 的损失函数，我们获得了期望通用动态遗憾界 $\widetilde O(GD\sqrt{T\Lambda}+\sigma DT^{1/p}\Lambda^{(p-1)/p})$，其中 $\Lambda=1+P_T/D$，$P_T$ 是固定比较序列的路径长度。该算法将重启的 AdaGrad 专家与自适应熵正则化主算法相结合，每轮仅使用一次随机梯度，且不需要知道 $G$、$\sigma$、$p$ 或 $P_T$ 的任何信息。其迭代点在梯度的正数缩放变换下保持不变。该分析在取期望之前控制重启块内比较序列的移动，从而得到噪声路径指数 $(p-1)/p$，而非直接非重启扩展所给出的指数 $1/2$。同时给出了一个匹配的随机一阶预言机下界……

    arXiv:2607.27073v2 Announce Type: replace-cross  Abstract: We study online convex optimization with stochastic gradient noise whose conditional $p$-th central moment is bounded by $\sigma^p$, for an unknown $p\in(1,2]$. For losses with Lipschitz bound $G$ on a domain of diameter $D$, we obtain expected universal dynamic regret $\widetilde O(GD\sqrt{T\Lambda}+\sigma DT^{1/p}\Lambda^{(p-1)/p})$, where $\Lambda=1+P_T/D$ and $P_T$ is the path length of a fixed comparator sequence. The algorithm combines restarted AdaGrad experts with an adaptive entropy-regularized master, uses one stochastic gradient per round, and requires no knowledge of $G,\sigma,p$, or $P_T$. Its iterates are invariant under positive rescaling of the gradients. The analysis controls comparator movement within restart blocks before taking expectations, yielding the noise path exponent $(p-1)/p$ rather than the exponent $1/2$ of a direct non-restarted extension. A matching stochastic first-order oracle lower bound, comb
    
[^383]: 重新思考自我进化：一种缓解技能过拟合的受限探索-利用过程

    Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting

    [https://arxiv.org/abs/2607.26643](https://arxiv.org/abs/2607.26643)

    SkillBoost通过结构化利用和先验引导探索的受约束框架，有效平衡探索与利用，从而缓解LLM代理技能自我进化中的过拟合问题。

    

    摘要：使大型语言模型（LLM）代理能够从过去的交互中积累和重用经验，在现实世界应用中仍是一个核心挑战。一个有前景的解决方案是将技能视为可训练状态，并以类似于神经网络训练中模型参数优化的方式对其进行优化。然而，数据驱动的技能优化容易过度拟合从真实环境中收集的有限轨迹。过度利用这些轨迹会导致对当前批次的过拟合，而无约束的探索则会导致先前已解决问题上的性能退化。这种张力促使我们提出一种受约束的搜索视角来审视技能自我进化，该过程由探索-利用权衡所主导。我们提出了SkillBoost，一个三阶段框架，用以缓解这两种风险：结构化利用将观察到的失败定位到可编辑的技能组件，先验引导的探索利用LLM中的先验知识生成多样的修复候选方案。

    arXiv:2607.26643v2 Announce Type: replace  Abstract: Enabling large language model (LLM) agents to accumulate and reuse experience from past interactions remains a central challenge in real-world applications. A promising solution is to treat skills as trainable states and optimize them in the same way as model parameters in neural network training. However, data-driven skill optimization is prone to overfitting to the limited trajectories collected from real environments. Overexploiting these trajectories overfits the current batch, while unconstrained exploration causes regression on previously solved cases. This tension motivates a constrained search view of skill self-evolution, governed by an exploration--exploitation trade-off. We propose SkillBoost, a three-stage framework that mitigates both risks: structured exploitation localizes observed failures to editable skill components, prior-guided exploration draws on prior knowledge in the LLM to generate diverse repair candidates, 
    
[^384]: Pramana：一个面向实证网络研究的可组合、领域专用后端

    Pramana: A Composable, Domain-Specific Backend for Empirical Networking Research

    [https://arxiv.org/abs/2607.26352](https://arxiv.org/abs/2607.26352)

    论文提出Pramana——一个采用细腰架构（上层为研究意图、下层为执行基底）的可组合网络研究后端，旨在按需自动为每个新假设生成实验数据，以弥合AI快速提出假设与人工搭建实验之间的“构思到数据鸿沟”。

    

    实证网络研究通常需要用研究本身生成的数据来检验假设，因此加速这类研究的关键在于缩短从形成假设到生成数据所需的时间。例如，考虑一个批量BBR下载是否与Google Meet通话公平共享瓶颈带宽：测试这一假设需要配置一条真实的瓶颈链路、同时运行BBR传输和Meet通话，并收集该通话的质量指标。如今，研究人员不得不为每一个新想法从头搭建这样的实验环境，这种从想法到数据的延迟被称为“构思到数据鸿沟”。AI智能体提出假设的速度已经超过研究人员生成数据来验证它们的速度，弥合这一鸿沟需要一个能够按需为每个新假设生成数据、而无需重新搭建环境的后端。我们提出了Pramana的构想——一个面向网络研究的可组合后端，它采用“细腰”架构设计：其上层是研究意图，其下层是执行基底。

    arXiv:2607.26352v2 Announce Type: replace-cross  Abstract: Empirical networking research often tests a hypothesis with data the study itself generates, so accelerating this research requires shortening the time from forming a hypothesis to generating that data. Consider whether a bulk BBR download fairly shares its bottleneck with a Google Meet call. Testing this requires configuring a realistic bottleneck link, running the BBR transfer and the Meet call together, and collecting the call's quality metrics. Today a researcher builds this setup from scratch for each new idea; this delay from idea to data is the ideation-to-data gap. AI agents already propose hypotheses faster than researchers can generate data to test them. Closing the gap needs a backend that generates the data for each new hypothesis on demand, without a new setup.   We envision Pramana, a composable backend for networking research designed as a thin waist, with research intents above it and execution substrates below.
    
[^385]: 少量教师步骤大有裨益：面向智能体后训练的高性价比同策略数据增强

    A Few Teacher Steps Go a Long Way: Cost-Efficient On-Policy Data Augmentation for Agent Post-Training

    [https://arxiv.org/abs/2607.04574](https://arxiv.org/abs/2607.04574)

    该论文将智能体后训练中的同策略数据构建形式化为一个预算分配问题，系统权衡教师演示、续写长度、过滤规则与上下文覆盖等设计维度，发现少量教师步骤即可带来显著收益，从而实现高性价比的数据增强。

    

    对于大语言模型智能体而言，监督微调不仅关乎教师标签的质量，还关乎这些标签所条件化的交互上下文。纯粹的行为克隆使用完整的教师演示，导致训练中所见的教师诱导上下文与测试时所遇到的学生诱导上下文之间存在不匹配。近期工作通过在学生到达的上下文处查询教师来解决这一不匹配问题，并往往对教师的续写内容进行日益精细的过滤。我们则将同策略数据构建视为一个预算分配问题：在监督资源匹配的情况下，教师输出应该用于更多的从头到尾的完整演示、更长的续写、基于结果的过滤，还是更广泛地覆盖学习者诱导的上下文？我们通过展开策略、切换时间分布、续写视界、过滤规则，以及两种互补的成本——教师推理（成本）……来将这一设计空间形式化。

    arXiv:2607.04574v2 Announce Type: replace-cross  Abstract: For LLM agents, supervised fine-tuning is not only about teacher labels' quality, but also about which interaction contexts those labels condition on. Pure behavioral cloning uses full teacher demonstrations, creating a mismatch between teacher-induced contexts seen in training and student-induced contexts encountered at test time. Recent work addresses this mismatch by querying a teacher at contexts reached by the student, often with increasingly elaborate filtering of the teacher's continuations. We instead frame on-policy data construction as a budget-allocation problem: under matched supervision resources, should teacher output be spent on more start-to-finish demos, longer continuations, outcome filtering, or broader coverage of learner-induced contexts? We formalize this design space through the rollout policy, switch-time distribution, continuation horizon, filtering rules, and two complementary costs: teacher inference 
    
[^386]: PruneGround：面向3D视觉定位的即插即用空间剪枝框架

    PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

    [https://arxiv.org/abs/2606.31148](https://arxiv.org/abs/2606.31148)

    提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。

    

    3D视觉定位（3DVG）旨在根据自然语言描述在3D场景中定位目标物体。现有方法通常对整个场景进行推理，导致预测结果模糊且计算成本高昂，尤其是在杂乱的环境中。我们观察到，许多指代表达依赖于局部空间上下文，并且往往对应于受限的空间区域而非整个场景。受这一洞察启发，我们提出了PruneGround，一个高效的即插即用3DVG框架，它建立在三个关键组件之上。首先，我们引入了语言引导空间剪枝（LGSP），利用冻结的视觉语言模型（VLM）识别与语言相关的区域，从而在更窄的搜索空间中减少空间计算和定位候选。其次，我们提出了多视角条件化描述重构（MCDR），将复杂表达分解为简化的目标中心描述……（摘要内容不完整）

    arXiv:2606.31148v2 Announce Type: replace-cross  Abstract: 3D Visual Grounding (3DVG) aims to localize target objects in 3D scenes given natural language descriptions. Existing approaches typically perform reasoning over the entire scene, leading to ambiguous predictions and high computational cost, especially in cluttered environments. We observe that many referential expressions rely on local spatial context and often correspond to restricted spatial regions rather than the full scene. Motivated by this insight, we propose PruneGround, an effective plug-and-play framework for 3DVG built upon three key components. First, we introduce Language-Guided Spatial Pruning (LGSP), which leverages a frozen Vision Language Model (VLM) to identify language-relevant regions, thereby reducing spatial computation and grounding candidates in the narrower search space. Second, we propose MultiView-Conditioned Description Reformulation (MCDR), which decomposes complex expressions into simplified targe
    
[^387]: LLM-ACES：基于大语言模型引导自适应搜索的动力系统闭环发现

    LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search

    [https://arxiv.org/abs/2606.25039](https://arxiv.org/abs/2606.25039)

    LLM-ACES提出了一种由大语言模型引导的闭环方程发现框架，通过LLM生成算子先验来缩小符号搜索空间，并联合优化假设构建与自适应数据采集，利用候选方程间的分歧指导新数据获取，从而消除可辨识性缺口、准确恢复动力系统的控制方程。

    

    从数据中恢复控制性常微分方程（ODE）是对各科学领域动力系统建模的核心挑战。现有方法将方程发现视为固定数据集上的静态推断问题，并假设观测到的轨迹包含足够的信息。然而，动力系统在广阔的状态空间上演化，有限的数据可能使多个方程在观测上不可区分，从而导致可辨识性缺口以及错误控制方程的恢复。为解决这一问题，我们提出了LLM-ACES（大语言模型引导的主动闭环方程搜索），这是一个联合优化符号假设构建与自适应数据采集的闭环框架。在LLM-ACES中，大语言模型（LLM）提出算子先验，这些先验定义了符号搜索空间的受约束子空间，候选方程在该子空间内对观测数据进行拟合。（原文摘要在此处截断）

    arXiv:2606.25039v2 Announce Type: replace-cross  Abstract: Recovering governing ordinary differential equations (ODEs) from data is a central challenge in modeling dynamical systems across scientific domains. Existing approaches cast discovery as a static inference problem over fixed datasets, assuming that the observed trajectories are sufficiently informative. However, dynamical systems evolve over large state spaces, and limited data can make multiple equations observationally indistinguishable, leading to identifiability gaps and the recovery of incorrect governing equations. To address this, we introduce LLM-ACES, or LLM-guided Active Closed-loop Equation Search, a closed-loop framework that jointly optimizes symbolic hypothesis construction and adaptive data acquisition. In LLM-ACES, a large language model (LLM) proposes operator priors that define constrained subspaces of the symbolic search space, within which candidate equations are fit to the observed data. The disagreement a
    
[^388]: FlowR2A：面向多模态驾驶规划的奖励到动作分布学习

    FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning

    [https://arxiv.org/abs/2606.24231](https://arxiv.org/abs/2606.24231)

    FlowR2A将基于仿真的奖励从判别式目标重新定义为生成式条件，利用流匹配解码器学习奖励条件下的动作分布，在单一生成模型中统一了密集奖励监督与动态候选轨迹生成，从而提升多模态驾驶规划的效果。

    

    多模态驾驶规划长期面临两种范式之间的矛盾：基于评分的方法受益于密集的奖励监督，但被限制在固定的动作词表内；而基于锚点的方法能够动态生成候选轨迹，却因只能受单一真值轨迹的稀疏监督而受限。在本工作中，我们提出FlowR2A，通过将基于仿真的奖励从判别式目标重新定义为生成式条件，化解了这一矛盾。FlowR2A利用流匹配解码器，从密集的轨迹-奖励数据对中学习以奖励为条件的动作分布，从而在单一生成模型中统一了基于评分方法的密集监督与基于锚点方法的候选轨迹生成，迫使模型内化动作与其在安全性、行进进度、舒适性以及规则遵循等方面结果之间的关联。为了在硬性安全约束与软性行进目标之间取得平衡……（摘要内容截断于此）

    arXiv:2606.24231v2 Announce Type: replace  Abstract: Multimodal driving planning faces a long-standing tension between two paradigms: scoring-based methods benefit from dense reward supervision but are confined to a fixed action vocabulary, while anchor-based methods generate proposals dynamically yet suffer from sparse supervision constrained to a single ground-truth trajectory. In this work, we propose FlowR2A, which resolves this tension by reframing simulation-based rewards from discriminative targets into generative conditions. By learning the reward-conditioned action distribution from dense trajectory-reward pairs with a flow-matching decoder, FlowR2A unifies the dense supervision of scoring-based methods with the proposal generation of anchor-based methods in a single generative model, forcing the model to internalize the correlation between an action and its outcomes in safety, progress, comfort, and rule compliance. To balance hard safety constraints against soft progress obj
    
[^389]: 语言智能体中开放式多智能体协作的基准测试

    Benchmarking Open-Ended Multi-Agent Coordination in Language Agents

    [https://arxiv.org/abs/2606.08340](https://arxiv.org/abs/2606.08340)

    提出了基于 JAX 的开放式多智能体协作基准 Alem，将程序化协作任务、软专业化和通信嵌入长时程生存世界中，评估显示当前 LLM 智能体平均归一化回报仅约 6%，远未解决该基准。

    

    随着语言模型越来越多地被部署为自主智能体，它们必须在开放式交互任务中与他人进行长时程协作。然而，现有评估很少将这些需求结合在一起进行测试，而是侧重于单智能体任务、短时交互或高度结构化的多智能体设置。我们提出了 alem，一个基于 JAX 构建的开放式多智能体协作基准，其建立在类 Craftax 的动态机制之上。Alem 将程序化生成的协作任务、软专业化、通信以及可控的协作难度嵌入到一个包含探索、制作、交易和战斗的长时程生存世界中。我们在同质团队中对 13 个大语言模型进行零样本评估，并以训练好的 MARL（多智能体强化学习）智能体作为参考基准。当前的 LLM 智能体远未能解决 alem，平均归一化回报仅约 6%，但它们的失败表现并不一致。在最困难的协作设置中，零样本的 Gemini-3.1-Pro-High 接近……

    arXiv:2606.08340v2 Announce Type: replace  Abstract: As language models are increasingly deployed as autonomous agents, they must coordinate with others over long horizons in open-ended interactive tasks. Yet existing evaluations rarely test these demands together, focusing instead on single-agent tasks, short interactions, or highly structured multi-agent settings. We introduce $alem$, a JAX-based benchmark for open-ended multi-agent coordination built on Craftax-like dynamics. Alem embeds procedurally generated coordination tasks, soft specialisation, communication, and controllable coordination difficulty into a long-horizon survival world with exploration, crafting, trading, and combat. We evaluate $13$ LLMs zero-shot within homogeneous teams, with trained MARL agents as reference points. Current LLM agents remain far from solving alem, averaging only ~6% normalised return, but their failures are not uniform. On the hardest coordination setting, zero-shot Gemini-3.1-Pro-High approa
    
[^390]: 涌现语言作为通往意识人工智能的途径

    Emergent Language as an Approach to Conscious AI

    [https://arxiv.org/abs/2606.06380](https://arxiv.org/abs/2606.06380)

    提出一种生成式方法论：让多智能体强化学习中从零开始的智能体仅凭任务压力自发发展出交流语言，从而在摆脱人类语言先验干扰的条件下研究生成意识相关结构的涌现。

    

    人工系统是否具有意识这一问题至今悬而未决，部分原因在于现有方法要么依据理论推导出的清单对系统进行评估（判别式方法），要么直接工程化构建受意识启发的模块（架构式方法）；这两种方法都无法排除所观察到的结构究竟是人类语言先验的产物。我们提出一种生成式方法论：多智能体强化学习中的涌现语言（EL），其中智能体从最低限度的初始状态出发（没有语言、没有自我概念、极少接触人类文本），仅在任务压力下发展出交流能力，从而确保相关结构可因果地归因于任务需求，而非继承自人类语言先验。我们通过讨论涌现语言如何作为研究意识相关结构的生成性工具来确立本方法论的定位，包括环境复杂度的作用以及对涌现交流的解读。作为概念验证，我们实例化了这一……

    arXiv:2606.06380v2 Announce Type: replace-cross  Abstract: The question of whether artificial systems can be conscious remains open, in part because existing approaches either evaluate systems against theory-derived checklists (discriminative) or engineer consciousness-inspired modules directly (architectural); both leave open whether observed structures are artifacts of human language priors. We propose a generative methodology: emergent language (EL) in multi-agent reinforcement learning, where agents start from minimal (no language, no concept of self, minimal exposure to human text) and develop communication under task pressure alone, ensuring causal attributability to task demands ratherthan inherited human language priors. We position our methodology by discussing how EL serves as a generative tool for studying consciousness-relevant structure, including the role of environment complexity and the interpretation of emergent communication. As a proof of concept, we instantiate this
    
[^391]: FLIPS：基于伪随机序列的大语言模型实例指纹识别

    FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences

    [https://arxiv.org/abs/2606.03330](https://arxiv.org/abs/2606.03330)

    本文提出面向AI监管的“实例级指纹识别”新范式及方法FLIPS，通过分析模型生成的二进制伪随机序列中的偏差，来区分同一大语言模型在不同实例级配置（如提示词、采样设置、量化方式）下的部署实例。

    

    文献表明，大语言模型（LLM）的行为不仅取决于其原始权重，还取决于其实例级参数，例如指令提示词、采样配置或量化方式。一个在某种配置下生成安全输出的模型，在另一种配置下可能会产生有害内容。然而，当前的LLM识别技术（如模型指纹识别）侧重于知识产权保护，其设计倾向于对这些实例级参数的变化保持鲁棒性。这给AI监管带来了严峻挑战，因为合规评估针对的是实际部署的行为，而非模型的来源。本文提出了实例级指纹识别，这是一种面向监管者的新范式，用于区分同一LLM的不同配置。我们的方法FLIPS利用了模型生成的二进制随机序列中的偏差。在包含237个模型实例的基准测试中，它能够识别出实例……

    arXiv:2606.03330v2 Announce Type: replace-cross  Abstract: Literature reveals that a Large Language Model's (LLM) behavior is not only conditioned by its original weights but also its instance-level parameters, such as instructional prompt, sampling configuration or quantization. A model that generates safe outputs under one configuration may produce toxic content under another. However, current LLM identification techniques (such as fingerprinting) focus on intellectual property protection, and their design favors robustness to changes in these instance-level parameters. This poses a critical challenge for AI regulation in which compliance assessments target actual deployed behaviors, not model provenance. In this paper, we introduce instance-level fingerprinting, a regulator-oriented paradigm that distinguishes configurations of the same LLM. Our method, FLIPS, exploits biases in generated binary random sequences. On a benchmark of 237 model instances, it identifies the instance with
    
[^392]: 结构化提示优化推进复杂文本的全局与局部可解释学习

    Structured Prompt Optimization Advances Globally and Locally Interpretable Learning from Complex Text

    [https://arxiv.org/abs/2605.29076](https://arxiv.org/abs/2605.29076)

    eXTC 通过结构化提示优化学习自然语言规则手册（SOP），结合推理蒸馏与强化学习，实现了兼具快速推理、局部推理轨迹与全局模块化可解释性的高性能文本分类。

    

    大语言模型已推动文本分类取得进展，但现有范式面临权衡：仅基于标签的监督微调具有可扩展性，但对复杂文本的推理能力有限，且缺乏更广泛的模型透明度；而离散提示优化虽能提供人类可读的指令，却在性能与可扩展性上存在困难。我们提出了 eXTC（可解释文本分类器），其核心是以自然语言学习得到的标准操作规程（SOP，或称规则手册），分为三个阶段：（1）通过新的结构化提示优化算法学习 SOP；（2）将基于 SOP 的推理从大型教师 LLM 蒸馏到紧凑语言模型中；（3）通过强化学习将推理能力扩展至 SOP 之外。这一设计使 eXTC 能够提供：（i）借助紧凑语言模型实现快速推理；（ii）推理时的局部推理轨迹，以及对其所学领域规则的全局模块化解释；同时（iii）在性能上取得显著提升……

    arXiv:2605.29076v3 Announce Type: replace-cross  Abstract: LLMs have advanced text classification, yet existing paradigms face a trade-off: supervised (label only) fine-tuning is scalable but offers limited reasoning on complex text and lacks broader model transparency, while discrete prompt optimization offers human-readable instructions but struggles with performance and scalability. We introduce eXTC (eXplainable Text Classifier), centered on a Standard Operating Procedure (SOP, or rulebook) learned in natural language, in three stages: (1) learning the SOP via a new Structured Prompt Optimization algorithm; (2) SOP-grounded reasoning distillation from a large teacher LLM into a compact LM; and (3) expanding reasoning capabilities beyond the SOP via reinforcement learning. This design enables eXTC to provide (i) fast inference via a compact LM, with (ii) inference-time local reasoning traces, alongside a global, modular explanation of its learned domain rules, while (iii) significan
    
[^393]: Review Arcade：论LLM审稿意见与人类审稿的对齐性及其可博弈性

    Review Arcade: On the Human Alignment and Gameability of LLM Reviews

    [https://arxiv.org/abs/2605.28897](https://arxiv.org/abs/2605.28897)

    该论文通过对ACL Rolling Review录用与拒稿投稿的实证实验发现，LLM审稿意见与人类审稿的对齐程度有限且因提示词和模型而异，同时作者可以通过迭代修改论文的方式有效“博弈”LLM审稿系统。

    

    LLM生成的科学论文审稿意见正获得越来越多的关注，甚至已被主要学术会议正式试点采用。我们必须假设不仅审稿人在使用LLM辅助，作者也会在投稿前使用LLM修改论文。在这项工作中，我们对2024-2025周期ACL Rolling Review (ARR)的录用和拒稿投稿进行了实证实验，从作者和审稿人两个角度评估LLM审稿意见。首先，我们发现LLM审稿意见与人类审稿意见的对齐程度有限。在最佳情况下，对齐程度尚属合理。然而，我们还发现LLM与人类审稿的对齐程度在不同提示词和模型之间差异很大。最后，我们研究了作者使用迭代起草-修改工作流、根据LLM审稿意见改进投稿的场景。我们发现这种对LLM审稿的“博弈”在特定场景下可能是有效的。

    arXiv:2605.28897v2 Announce Type: replace  Abstract: LLM-generated reviews for scientific papers are gaining considerable traction and are even being officially piloted by major conferences. We have to assume that not only reviewers are using LLM-assistance, but also that authors use LLMs to revise their papers before submitting. In this work, we perform empirical experiments on accepted and rejected submissions to the 2024-2025 cycles of the ACL Rolling Review (ARR) to evaluate LLM reviews from both the author and the reviewer perspective. First, we identify a limited alignment of LLM reviews with human ones. In the best-case scenario, the alignment is reasonable. However, we also find that LLM-human alignment varies substantially across prompts and models. Finally, we investigate the scenario in which the author uses an iterative draft-revise workflow to improve the submission according to the LLM review. We find that this "gaming" of LLM reviews can be effective in specific scenario
    
[^394]: TASTE 之要义：提升智能体基准测试的覆盖度与难度

    A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks

    [https://arxiv.org/abs/2605.28556](https://arxiv.org/abs/2605.28556)

    本文提出 TASTE 方法，通过颠倒传统任务构建流程，利用在 LLM 判定有效性信号上训练的自适应对比 n-gram 模型采样覆盖广泛工具组合的有效工具序列，再经聚类选取代表性序列并实例化，从而自动合成覆盖度更广、难度更高的智能体基准测试任务。

    

    随着智能体能力的不断进步，诸如 $\tau^2$-Bench 等现有基准测试正日益饱和。然而，构建新的基准测试任务依然复杂、昂贵且耗费人力。此外，标准方法——先以自然语言编写场景，再将其映射为工具序列——仅能覆盖智能体所运用的工具使用模式中的一小部分。在本文中，我们通过颠倒任务构建流程来解决这些问题。我们提出了 TASTE：基于工具序列演化的任务合成（Task Synthesis from Tool Sequence Evolution），这是一种自动生成具有更广泛工具使用覆盖范围且更具挑战性任务的方法。TASTE 利用一种在 LLM 判定的有效性信号上训练的自适应对比 n-gram 模型，从而能够采样出覆盖大量工具组合的有效工具序列。随后，TASTE 通过聚类从序列池中选取代表性序列，将其实例化为完整的基准测试任务，并对……进行精炼。

    arXiv:2605.28556v3 Announce Type: replace  Abstract: As agent capabilities advance, existing benchmarks, such as $\tau^2$-Bench, are becoming increasingly saturated. Yet constructing new benchmark tasks remains complex, costly, and labor-intensive. Moreover, the standard approach, in which scenarios are first written in natural language and then mapped to tool sequences, captures only a narrow subset of the tool-use patterns agents exercise. In this paper, we address these problems by reversing the task construction process. We propose TASTE: Task Synthesis from Tool Sequence Evolution, an automatic method that generates challenging tasks with broader tool-use coverage. TASTE utilizes an Adaptive Contrastive $n$-gram model trained on LLM-judged validity signals. This enables sampling valid tool sequences that cover a vast range of tool combinations. TASTE then selects representative sequences from the pool via clustering, instantiates them into complete benchmark tasks, and refines the
    
[^395]: MobileMoE：扩展端侧混合专家模型

    MobileMoE: Scaling On-Device Mixture of Experts

    [https://arxiv.org/abs/2605.27358](https://arxiv.org/abs/2605.27358)

    提出MobileMoE端侧MoE语言模型系列，通过端侧MoE缩放定律确定内存与计算同时最优的架构（适度稀疏+细粒度专家+共享专家），在十亿以下激活参数规模上为端侧大语言模型建立了新的帕累托前沿。

    

    混合专家已成为千亿参数语言模型的事实标准架构，但其在十亿以下规模用于端侧部署的优势在很大程度上尚未被探索。为弥补这一空白，我们提出了MobileMoE，这是一系列具有十亿以下激活参数的端侧MoE语言模型（激活参数0.3-0.9B，总参数1.3-5.3B），为端侧大语言模型建立了新的帕累托前沿。我们首先制定了一个端侧MoE缩放定律，在移动设备内存和计算约束下联合优化MoE架构，确定了端侧最佳设计点——采用细粒度专家和共享专家的适度稀疏性——同时实现内存最优和计算最优。基于所推导的架构，我们采用涵盖预训练、中期训练、指令微调和量化感知训练的四阶段配方来训练MobileMoE，且全部使用开源数据集。在14个基准测试中，MobileMoE达到了……

    arXiv:2605.27358v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) has become the de facto architecture for hundred-billion-parameter language models, yet its advantages at sub-billion scales for on-device deployment remain largely unexplored. To close this gap, we present MobileMoE, a family of on-device MoE language models with sub-billion active parameters (0.3-0.9B active and 1.3-5.3B total) that establish a new Pareto frontier for on-device LLMs. We first formulate an on-device MoE scaling law that jointly optimizes MoE architecture under mobile memory and compute constraints, identifying an on-device sweet spot - moderate sparsity with fine-grained and shared experts - that is simultaneously memory and compute-optimal. Building on the derived architectures, we train MobileMoE with a four-stage recipe covering pre-training, mid-training, instruction fine-tuning, and quantization-aware training, all on open-source datasets. Across 14 benchmarks, MobileMoE matches o
    
[^396]: 基于利益相关者关联锚定文本嵌入

    Grounding Text Embeddings in Stakeholder Associations

    [https://arxiv.org/abs/2605.27168](https://arxiv.org/abs/2605.27168)

    该论文提出“利益相关者锚定练习”方法以评估文本嵌入与人类专家语义关联的对齐程度，发现神经文本嵌入的可靠性显著低于人类专家（差距18-27个百分点），且该错位会传播到下游聚类任务，并在英语和美国联邦AI用例中得到复现。

    

    文本嵌入被广泛用于分析复杂文本的大型语料库。然而，这些嵌入是否捕捉到与使用它们的人类专家相同的语义关系尚不清楚。这种表征对齐对于有效分析至关重要。我们提出了“利益相关者锚定练习”，这是一种使专家关联显性化、并将嵌入模型结果锚定于人类理解的方法。在针对丹麦政策议题的主要案例研究中，我们发现神经文本嵌入的可靠性显著低于人类专家（差距为18-27个百分点），且这种错位会传导至下游聚类性能（练习排名与聚类质量之间的Spearman ρ = 0.86）。一项针对美国联邦AI用例的次要研究采用数字化协议和不同的专家群体，在英语中复现了该差距（16.1个百分点），表明这一差距并非单一工具或领域的产物。

    arXiv:2605.27168v2 Announce Type: replace-cross  Abstract: Text embeddings are widely used to analyse large corpora of complex texts. However, it is unclear whether the embeddings capture the same semantic relations as the human experts using them. Such representational alignment is essential for valid analyses. We present the Stakeholder Grounding Exercise, a method for making expert associations explicit and grounding embedding model results in human understanding. In our primary case study on Danish policy issues, we find that neural text embeddings are substantially less reliable than human experts (18-27pp gap), and that this misalignment propagates to downstream clustering performance (Spearman $\rho = 0.86$ between exercise ranking and cluster quality). A secondary study on US Federal AI use cases replicates the gap (16.1pp) in English, using a digital protocol and a different community of experts -- demonstrating that the gap is not an artefact of a single instrument or domain.
    
[^397]: 基于随机搜索与迭代模仿的预言机预算蛋白质设计

    Oracle-Budgeted Protein Design with Stochastic Search and Iterative Imitation

    [https://arxiv.org/abs/2605.26690](https://arxiv.org/abs/2605.26690)

    SILO是一种轨迹级蛋白质设计方法，结合层次化编辑策略与随机束搜索探索多样化突变轨迹，利用UCB代理集成与丙氨酸扫描式适应度评分筛选候选序列，并通过交叉熵模仿学习迭代自我改进，在严格的预言机评估预算下高效优化蛋白质序列。

    

    在严格的预言机（评估）预算下优化蛋白质序列，需要能够探索庞大组合突变搜索空间的方法。这些方法还必须在代理模型噪声存在的情况下保持稳健，并在预测替换具有有利效果的位点上提出突变。我们提出了SILO（自我改进模仿学习优化），一种面向预言机预算蛋白质设计的轨迹级方法。SILO将层次化编辑策略与无放回的增量随机束搜索（SBS）相结合，以探索多样化的候选轨迹。基于UCB的代理模型集成，辅以受丙氨酸扫描启发的代理衍生适应度评分（AFS），指导候选选择以进行计算机内的预言机评估。随后，通过对当轮最佳预言机标注轨迹进行下一动作交叉熵模仿来更新策略。在八个计算机模拟蛋白质适应度景观和五个强基线方法上（原文摘要在此处截断）……

    arXiv:2605.26690v2 Announce Type: replace-cross  Abstract: Optimizing protein sequences under tight oracle budgets requires methods that can explore a vast combinatorial mutational search space. These methods must also remain robust in the presence of surrogate noise and propose mutations on positions where substitutions are predicted to have favorable effects. We introduce SILO (Self-improvement Imitation Learning Optimization), a trajectory-level approach for oracle-budgeted protein design. SILO combines a hierarchical edit policy with incremental stochastic beam search without replacement (SBS) to explore diverse candidate trajectories. A UCB-based proxy ensemble, complemented by a proxy-derived alanine-scan-inspired fitness score (AFS), guides candidate selection for in silico oracle evaluation. The policy is then updated by next-action cross-entropy imitation on the round's best oracle-labeled trajectories. Across eight in silico protein fitness landscapes and five strong baseline
    
[^398]: NPSolver：基于迭代物理监督的神经泊松求解器

    NPSolver: Neural Poisson Solver with Iterative Physics Supervision

    [https://arxiv.org/abs/2605.25786](https://arxiv.org/abs/2605.25786)

    NPSolver 提出了一种无需解标签的神经泊松求解器，通过少量预条件共轭梯度（PCG）迭代步精炼自身预测作为迭代物理监督，获得了比原始 PDE 残差更稳定、尺度更合适的训练信号。

    

    在复杂、不规则区域上高效求解泊松方程仍然是科学计算中的一项根本性挑战，因为传统迭代求解器常因系统病态而导致运行时间过长。虽然神经算子提供了一种快速的替代方案，但它们通常依赖于大规模标注数据集，或者在使用物理信息残差损失时面临训练动态不稳定的问题。我们提出了 NPSolver，一种通过迭代物理监督、无需解标签进行训练的神经泊松求解器。NPSolver 既不依赖完全收敛的数值解，也不使用原始 PDE 残差，而是利用少量预条件共轭梯度（PCG）迭代步来精炼其自身的预测，从而提供更稳定、尺度更合适的训练信号。理论分析证实，这种迭代监督充当了一个条件良好的误差代理，并且停止梯度设计是必不可少的。

    arXiv:2605.25786v2 Announce Type: replace-cross  Abstract: Efficiently solving Poisson equations on complex, irregular domains remains a fundamental challenge in scientific computing, as classical iterative solvers often suffer from prohibitive runtime due to ill-conditioned systems. While neural operators offer a fast alternative, they typically rely on large-scale labeled datasets or struggle with unstable training dynamics when using physics-informed residual losses. We propose \textsc{NPSolver}, a neural Poisson solver trained without solution labels via iterative physics supervision. Instead of relying on fully converged numerical solutions or raw PDE residuals, \textsc{NPSolver} utilizes a small number of preconditioned conjugate gradient (PCG) steps to refine its own predictions, providing a more stable and well-scaled training signal. Theoretical analysis confirms that this iterative supervision serves as a well-conditioned error proxy and that a stop-gradient design is essenti
    
[^399]: Flash PD-SSM：内存优化的结构化稀疏状态空间模型

    Flash PD-SSM: Memory-Optimized Structured Sparse State-Space Models

    [https://arxiv.org/abs/2605.19150](https://arxiv.org/abs/2605.19150)

    提出了一种新颖的状态空间模型PD-SSM，通过在每个时间步从一组可训练的结构化稀疏矩阵中离散选择一个矩阵，在保持与主流结构化SSM相当的吞吐量的同时，实现了与非结构化转移矩阵相当的表达能力。

    

    状态空间模型（SSM）面临着效率与表达能力之间的根本性权衡，这主要由模型转移矩阵的结构决定。非结构化的转移矩阵能够实现最大化的表达能力（以其对有限状态自动机（FSA）转移的建模能力来衡量），但其计算和内存成本过高。相比之下，大多数结构化的转移矩阵形式在运行时间和内存消耗方面都非常高效，但表达能力有限。基于近期关于结构化稀疏SSM的研究工作，我们提出了PD-SSM，这是一种新颖的状态空间模型，其吞吐量与广泛使用的结构化SSM相当，同时具有显著更优的表达能力保证。PD-SSM维护一组可训练的结构化稀疏矩阵，并在每个时间步从中离散地选择一个矩阵，从而实现与非结构化矩阵同等的FSA表达能力，同时保持高效。

    arXiv:2605.19150v2 Announce Type: replace-cross  Abstract: State-space models (SSMs) face a fundamental trade-off between efficiency and expressivity that is mainly dictated by the structure of the model's transition matrix. Unstructured transition matrices enable maximal expressivity, as measured by their ability to model finite-state automaton (FSA) transitions, but come at a prohibitively high compute and memory cost. In contrast, most structured transition matrix forms are highly efficient both in runtime and memory consumption, but suffer from limited expressivity. Building on recent work on structured sparse SSMs, we propose PD-SSM, a novel SSM that achieves comparable throughput to widely-used structured SSMs with significantly better expressivity guarantees. PD-SSM maintains a trainable set of structured sparse matrices, a single one of which is discretely selected at each time-step, enabling FSA expressiveness at the level of unstructured matrices while maintaining the efficie
    
[^400]: 面向离散不确定性两阶段鲁棒优化的学习式场景缩减方法

    Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty

    [https://arxiv.org/abs/2605.14494](https://arxiv.org/abs/2605.14494)

    本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。

    

    具有离散不确定性的两阶段鲁棒优化（2RO）极具挑战性，往往导致精确求解的计算代价难以承受。场景缩减通过选择一个小而具有代表性的场景子集来缓解这一问题，从而使计算变得可行。然而，现有方法在很大程度上与具体问题无关，仅在不考虑可行域或追索结构的情况下对不确定性集合进行操作。本文提出了 PRISE，一种问题驱动的顺序前瞻启发式算法，通过评估每个场景的边际影响来构建缩减后的场景集合。尽管 PRISE 能够产生高质量的场景子集，但其每个选择步骤都需要求解多个子问题，使得该算法在大规模问题上的计算开销十分高昂。为解决这一问题，我们提出了 NeurPRISE，一个基于 GNN-Transformer 骨干网络的神经代理模型，它通过图卷积对单个场景的结构进行编码，并捕获跨场景之间的交互关系。

    arXiv:2605.14494v2 Announce Type: replace  Abstract: Two-Stage Robust Optimization (2RO) with discrete uncertainty is challenging, often rendering exact solutions prohibitive. Scenario reduction alleviates this issue by selecting a small, representative subset of scenarios to enable tractable computation. However, existing methods are largely problem-agnostic, operating solely on the uncertainty set without consulting the feasible region or recourse structure. In this paper, we introduce PRISE, a problem-driven sequential lookahead heuristic that constructs reduced scenario sets by evaluating the marginal impact of each scenario. While PRISE yields high-quality scenario subsets, each selection step requires solving multiple subproblems, making it computationally expensive at scale. To address this, we propose NeurPRISE, a neural surrogate model built on a GNN-Transformer backbone that encodes the per-scenario structure via graph convolution and captures cross-scenario interactions thro
    
[^401]: 通过角色分歧实现大语言模型的无训练文化对齐

    Training-Free Cultural Alignment of Large Language Models via Persona Disagreement

    [https://arxiv.org/abs/2605.10843](https://arxiv.org/abs/2605.10843)

    提出DISCA方法，无需训练、无需特定国家的数据或访问模型内部，仅通过基于世界价值观调查数据构建的四角色提示面板的共识与分歧机制，即可将权重冻结的大语言模型对齐到不同国家的道德偏好。

    

    大语言模型如今参与到依赖于道德偏好的决策中，而这些偏好因国家而异。模型的默认偏好最接近某一个人群，因此单一的部署会用一个国家的价值观来服务所有国家。现有的解决方案需要针对每种文化的训练数据、针对每个国家的奖励模型，或访问模型内部结构——而距离该默认值最远的国家往往缺乏这些资源。我们展示了在完全没有这些资源的情况下，能够将模型适配到多远的国家。我们的方法DISCA保持模型权重冻结，不使用任何国家特定的人类标注目标，只读取模型对每个答案选项的评分。该方法基于世界价值观调查的微观数据，将一个国家构建为由四个角色提示组成的评审面板。面板的共识设定一个以国家为条件的锚点，超出该锚点的损失规避式精细化调整只有在两次独立运行结果一致时才被保留，同时一个持续更新的国家级先验会汇集已做出的决策……

    arXiv:2605.10843v3 Announce Type: replace-cross  Abstract: Large language models now take part in decisions that rest on moral preferences, and those preferences differ by country. A model's default preferences sit closest to one population, so a single deployment serves every country with one country's values. Existing fixes need per-culture training data, a per-country reward model, or access to model internals, which the countries furthest from that default rarely have. We show how far a country can be reached without any of them. Our method, DISCA, keeps the model's weights frozen, sees no country-specific human targets, and reads only the model's score for each answer option. It builds a country as a panel of four persona prompts grounded in World Values Survey microdata. The panel's consensus sets a country-conditioned anchor, a loss-averse refinement beyond that anchor is kept only when two independent runs of it agree, and a running country-level prior pools the decisions alrea
    
[^402]: M2A：在大语言模型中协同数学推理与智能体推理

    M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models

    [https://arxiv.org/abs/2605.09879](https://arxiv.org/abs/2605.09879)

    该论文提出M2A新范式，通过在参数空间中识别智能体行为的关键特征子空间并进行模型合并，避免了联合训练中的过拟合问题，从而有效协同大语言模型中的数学推理与智能体推理能力。

    

    虽然推理已成为大语言模型（LLM）的核心能力，但不同场景所需的推理模式往往并不一致。数学推理通常依赖内在逻辑，在单次响应中解决封闭世界问题；而智能体推理不仅需要内部推理，还需要与外部环境进行多轮交互，将思考与行动交织进行。这种不一致使得数学推理与智能体推理难以有效相互促进，在多任务学习下常常导致不稳定的推理行为，且性能提升有限。本文提出M2A，一种通过模型合并来协同数学推理与智能体推理的新范式。为避免联合训练中对表面推理模式的过拟合，M2A直接在参数空间中操作：它识别出对智能体行为至关重要的特征子空间，并将数学推理能力合并到该子空间之中……

    arXiv:2605.09879v2 Announce Type: replace  Abstract: While reasoning has become a central capability of large language models (LLMs), the reasoning patterns required for different scenarios are often misaligned. Mathematical reasoning typically relies on intrinsic logic to solve closed-world problems in a single response, whereas agentic reasoning requires not only internal reasoning but also multi-turn interaction with external environments, interleaving thought and action. This misalignment prevents mathematical and agentic reasoning from effectively benefiting from each other, often yielding unstable reasoning behavior and only limited performance gains under multi-task learning. In this paper, we propose M2A, a novel paradigm that synergizes mathematical and agentic reasoning via model merging. To avoid overfitting to superficial reasoning patterns under joint training, M2A operates directly in parameter space: it identifies the feature subspace critical for agent behavior, and mer
    
[^403]: 人工智能（AI）解释的公平性：统一框架、公理与迈向负责任人工智能的未来方向

    Fairness of Explanations in Artificial Intelligence (AI): A Unifying Framework, Axioms, and Future Direction toward Responsible AI

    [https://arxiv.org/abs/2605.09852](https://arxiv.org/abs/2605.09852)

    本文首次提出了研究AI解释公平性的统一理论框架与公理体系，揭示了模型即使输出满足公平性标准，其推理过程仍可能存在“程序性偏见”，并为迈向负责任的人工智能指明了未来方向。

    

    机器学习算法正被应用于高风险决策领域，包括刑事司法、医疗保健、信贷和就业等。研究界为此发展出了两个彼此较为独立的研究领域：算法公平性，旨在实现公平的结果；以及可解释人工智能（XAI），旨在实现可解释的推理。本综述识别并描绘了这两个领域交叉处的一个新颖盲点：即一个模型可能在输出上满足所有标准公平性准则，但其推理过程却存在深刻的不公平性。我们将此现象称为程序性偏见，要缓解这一问题，就需要将解释的公平性作为一个独立的科学研究对象。据我们所知，我们提供了对这一新兴领域的首个统一的理论与文献综述，并阐明了事后解释器在认证解释公平性方面的缺陷。我们的核心贡献...

    arXiv:2605.09852v2 Announce Type: replace  Abstract: Machine learning algorithms are being used in high-stakes decisions, including those in criminal justice, healthcare, credit, and employment. The research community has responded with two largely independent research fields: \emph{algorithmic fairness}, which targets equitable outcomes, and \emph{explainable AI} (XAI), which targets interpretable reasoning. This survey identifies and maps a novel blind spot at their intersection, which is a model that can satisfy every standard fairness criterion in its outputs while being profoundly unfair in its \emph{reasoning process}. We refer to this as the procedural bias, and mitigating it requires treating the fairness of explanations as a distinct object of scientific study.   To our knowledge, we provide the first unified theoretical and literature review of this emerging field and elucidate the drawbacks of post-hoc explainers in certifying explanation fairness. Our central contribution i
    
[^404]: 智能体集体不应自行检测其内部的冒名顶替者：一个国际象棋案例研究

    Agent Collectives Should Not Detect Their Own Imposters: A Chess Case Study

    [https://arxiv.org/abs/2605.09027](https://arxiv.org/abs/2605.09027)

    研究表明让智能体集体自行检测内部冒名顶替者会适得其反，而采用能够从少量样本快速适应新型攻击的轻量级元训练外部检测器才是更优策略。

    

    AI智能体集体在复杂任务上可以超越任何单一智能体，但仍然容易受到故意误导同伴的冒名顶替者（imposter）的攻击。由于哪怕只有一个冒名顶替者也可能破坏整个集体，可靠的检测机制至关重要。我们研究了两种策略：将检测功能嵌入参与智能体自身内部，或者雇佣一个专门的、只观察但不参与集体讨论的外部检测器。在一个特意设计的小型但具有挑战性的国际象棋测试平台上，我们发现仅仅警告智能体可能存在冒名顶替者反而会产生反效果：即使在没有对手的情况下，它也会降低决策质量，引发无益的社交动态，并向冒名顶替者泄露信息。外部检测器虽然可以避免这些陷阱，但必须随着冒名顶替者演化出新策略而不断适应。我们证明，一个轻量级的元训练检测器仅凭少量样本就能快速重新校准，以应对未见过的攻击，其效率远高于其他方法。

    arXiv:2605.09027v4 Announce Type: replace-cross  Abstract: Collectives of AI agents can outperform any single agent on complex tasks, yet remain vulnerable to imposters that deliberately mislead their peers. Because even one such imposter can sabotage the collective, reliable detection is essential. We examine two strategies: embedding detection inside the participating agents, or employing a dedicated external detector that observes the deliberation without joining it. On a deliberately small but challenging chess-based testbed, we find that simply warning agents of possible imposters is counterproductive. It degrades decisions even in the absence of adversaries, triggers unhelpful social dynamics, and leaks information to the imposter. An external detector avoids these pitfalls, yet must continually adapt as imposters evolve new strategies. We show that a lightweight meta-trained detector can rapidly recalibrate to unseen attacks from only a handful of examples, far more efficiently 
    
[^405]: 基于CADTests的文本转CAD评估

    Text-to-CAD Evaluation with CADTests

    [https://arxiv.org/abs/2605.07807](https://arxiv.org/abs/2605.07807)

    该论文提出了首个基于可执行测试的文本转CAD评估基准CADTestBench，通过CADTests验证生成模型是否满足几何与拓扑要求，并证明这些测试还能指导CAD生成，使简单基线超越现有方法。

    

    文本转CAD（Text-to-CAD）近来已成为一项重要任务，具有大幅加速设计工作流程的潜力。尽管其意义重大，但令人惊讶的是，关于文本转CAD评估的研究工作却很少，评估CAD模型生成性能仍然是一个相当大的挑战。在这项工作中，我们引入了一种基于自动化测试的文本转CAD新评估视角。我们提出了CADTestBench，这是首个基于测试的文本转CAD基准，其基于CADTests构建——这是一类可执行的软件测试，用于验证生成的CAD模型是否满足输入提示词中的几何和拓扑要求。利用CADTestBench，我们对近期的文本转CAD方法进行了全面的基准测试，并进一步证明CADTests还可以用于指导CAD模型生成，由此得到的简单基线方法能够超越现有方法的性能。CADTestBench的代码和数据均已公开可用。

    arXiv:2605.07807v2 Announce Type: replace-cross  Abstract: Text-to-CAD has recently emerged as an important task with the potential to substantially accelerate design workflows. Despite its significance, there has been surprisingly little work on Text-to-CAD evaluation, and assessing CAD model generation performance remains a considerable challenge. In this work, we introduce a new evaluation perspective for Text-to-CAD based on automated testing. We propose CADTestBench, the first test-based benchmark for Text-to-CAD, based on CADTests, executable software tests that verify whether a generated CAD model satisfies the geometric and topological requirements of the input prompt. Using CADTestBench, we conduct comprehensive benchmarking of recent Text-to-CAD methods and further demonstrate that CADTests can also guide CAD model generation, yielding simple baselines that surpass performance of current methods. CADTestBench code and data are available.
    
[^406]: 实验即代码实验室：面向AI驱动科学发现的声明式技术栈

    Experiment-as-Code Labs: A Declarative Stack for AI-Driven Scientific Discovery

    [https://arxiv.org/abs/2605.04375](https://arxiv.org/abs/2605.04375)

    该论文提出“实验即代码”实验室新范式，借鉴计算机系统的理念将实验编码为声明式配置，从而弥合日益强大的AI智能体与自动化实验室设备之间的鸿沟，使AI驱动的科学发现能够掌控真实世界的物理实验室。

    

    为了释放“AI for Science”的全部潜力，我们必须将智能体从纯数字环境中解放出来。智能体在真实实验室中进行控制和探索的能力至关重要，因为物理实验室仍然是科学发现的基础。虽然某些任务可以在计算机上完成（例如数据分析、运行模拟实验），但在操作实验室仪器时，“顿悟时刻”随时可能发生（例如，当科学家注意到意外的线索时，直觉可能促使其实时改变研究方向）。尽管自主实验室正在兴起，它们通过软件暴露可编程API来控制科学仪器，但要弥合日益强大的AI智能体与自动化实验室设备之间的差距，需要借鉴计算机系统领域洞见的创新。我们提出了一种名为“实验即代码”实验室的新范式，其核心概念是将实验编码为声明式配置……

    arXiv:2605.04375v3 Announce Type: replace-cross  Abstract: To unleash the full potential of AI for Science, we must untether the agents from a purely digital environment. The agent's ability to control and explore in real-world labs is essential because the physical lab remains foundational to scientific discovery. While some tasks can be performed on a computer (e.g., data analysis, running simulated experiments), Eureka moments could occur at any time while operating lab instruments (e.g., when a scientist notices unexpected clues, intuition may prompt a real-time course change). Although autonomous labs are on the rise, which expose programmable APIs to control scientific instruments via software, bridging the gap between increasingly powerful AI agents and automated lab equipment requires innovation that draws insights from computer systems.   We propose a new paradigm called ``Experiment-as-Code (EaC) Labs,'' where a core concept is to encode experiments as declarative configurati
    
[^407]: 基于神经与结构方法的命令式程序图构建与匹配

    Graph Construction and Matching for Imperative Programs using Neural and Structural Methods

    [https://arxiv.org/abs/2604.26578](https://arxiv.org/abs/2604.26578)

    该论文提出一条将命令式程序及其标注转换为带类型属性图的流水线，通过结合抽象语法树解析与SentenceTransformer、CodeBERT等模型的语义嵌入，在C/ACSL、Java/JML和Dafny等多种语言上实现了跨语言、跨标注风格的一致性图表示，为验证产物复用和近似图匹配奠定了实用基础。

    

    复用验证产物需要识别程序及其规约之间的结构与语义相似性。在本文中，我们将图构建作为实现这一目标的基础步骤。我们提出了一条流水线，可将命令式程序及其标注转换为带类型的属性图。我们的实验涵盖多个数据集，包括带ACSL标注的C语言程序、带JML标注的Java程序以及Dafny程序。该流水线将抽象语法树解析与来自SentenceTransformer和CodeBERT等模型的语义嵌入相结合，从而能够生成同时捕捉结构关系和语义上下文的图表示。我们的结果表明，可以在不同语言和标注风格之间构建一致的图表示。这项工作为未来的语义增强和近似图匹配以实现可扩展的验证产物复用提供了实用基础。

    arXiv:2604.26578v4 Announce Type: replace-cross  Abstract: Reusing verification artefacts requires identifying structural and semantic similarities across programs and their specifications. In this paper, we focus on graph construction as a foundational step toward this goal. We present a pipeline that converts imperative programs and their annotations into typed, attributed graphs. Our experiments cover datasets including C with ACSL, Java with JML, and Dafny programs. The pipeline integrates abstract syntax tree parsing with semantic embeddings derived from models such as SentenceTransformer and CodeBERT. This enables the generation of graph representations that capture both structural relationships and semantic context. Our results show that consistent graph representations can be constructed across different languages and annotation styles. This work provides a practical basis for future steps in semantic enrichment and approximate graph matching for scalable verification artefact 
    
[^408]: SIEVES：通过视觉证据评分实现泛化的选择性预测

    SIEVES: Selective Prediction Generalizes through Visual Evidence Scoring

    [https://arxiv.org/abs/2604.25855](https://arxiv.org/abs/2604.25855)

    该论文提出SIEVES方法，通过要求模型提供图像裁剪作为视觉证据并对其进行评分来估计置信度，从而提升多模态大语言模型选择性预测在分布外场景中的泛化能力。

    

    随着多模态大语言模型（MLLMs）在传统视觉问答基准上的准确率接近饱和，可靠部署需要在现实世界的分布外（OOD）场景中满足低错误容忍度要求。选择性预测旨在提高覆盖率（即系统所回答输入的比例），同时遵守用户自定义的风险水平。这通常通过为每个答案分配置信度分数，并对低于特定阈值的答案进行弃答来实现。现有的选择性预测方法仅依赖纯文本推理，并仅仅根据答案是否正确来学习预测置信度。与之相反，我们提出了SIEVES（通过视觉证据评分实现选择性预测），该方法首先要求答题模型提供支持其答案的视觉证据，即在返回文本推理和答案的同时返回图像的裁剪区域；其次，一个（原文摘要在此处截断）

    arXiv:2604.25855v3 Announce Type: replace-cross  Abstract: As the accuracy of Multimodal large language models (MLLMs) on traditional visual question answering benchmarks approaches saturation, reliable deployment requires satisfying low error tolerances in real-world out-of-distribution (OOD) scenarios. Selective prediction aims to improve coverage, i.e. the share of inputs the system answers, while adhering to a user-defined risk level. This is typically achieved by assigning a confidence score to each answer and abstaining on those that fall below a certain threshold. Existing selective prediction methods rely on text-only reasoning and learn to predict confidences simply based on whether the answer was correct or not. In contrast, we introduce SIEVES (Selective Prediction through Visual Evidence Scoring), which first requires the answering model to provide visual evidence supporting its answer, by returning crops of the image alongside their textual reasoning and answer. Second, a 
    
[^409]: 自适应共形预测：提升大语言模型生成内容的事实性

    Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models

    [https://arxiv.org/abs/2604.13991](https://arxiv.org/abs/2604.13991)

    提出了一种提示词自适应的共形预测方法，通过提示词相关的校准提升大语言模型生成内容的事实性，在保持边际覆盖保证的同时改善条件覆盖率，并支持选择性预测以过滤不可靠输出。

    

    大语言模型（LLM）容易生成事实性错误的输出。近期的工作将共形预测应用于为大语言模型生成内容的事实性提供不确定性估计和统计保证。然而，现有方法通常无法自适应提示词，限制了其捕捉输入相关变化的能力。因此，对于给定的任务或提示，这些方法可能过滤掉过少的条目（导致过度覆盖）或过滤掉过多条目（导致覆盖不足）。我们提出了一种自适应共形预测方法，将共形分数变换方法扩展到大语言模型中，并应用于长文本生成和多选题问答任务。该方法实现了依赖于提示词的校准，在保持边际覆盖保证的同时提高了条件覆盖率。此外，该方法还自然地支持选择性预测，允许在下游应用中过滤掉不可靠的陈述或答案选项。

    arXiv:2604.13991v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are prone to generating factually incorrect outputs. Recent work has applied conformal prediction to provide uncertainty estimates and statistical guarantees for the factuality of LLM generations. However, existing approaches are typically not prompt-adaptive, limiting their ability to capture input-dependent variability. As a result, they may filter out too few items (leading to over-coverage) or too many (under-coverage) for a given task or prompt. We propose an adaptive conformal prediction approach that extends conformal score transformation methods to LLMs, with applications to long-form generation and multiple-choice question answering. This enables prompt-dependent calibration, retaining marginal coverage guarantees while improving conditional coverage. In addition, the approach naturally supports selective prediction, allowing unreliable claims or answer choices to be filtered out in downstr
    
[^410]: 思考、行动、构建：基于视觉语言模型的零样本3D视觉定位智能体框架

    Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding

    [https://arxiv.org/abs/2604.00528](https://arxiv.org/abs/2604.00528)

    该论文提出TAB动态智能体框架，将零样本3D视觉定位任务重构为直接作用于原始RGB-D数据流的生成式2D到3D重建范式，通过VLM智能体动态调用视觉工具进行跨帧跟踪与重建，从而摆脱了对预处理点云的依赖。

    

    3D视觉定位（3D-VG）旨在通过自然语言描述在3D场景中定位目标物体。尽管近期利用视觉语言模型（VLM）的研究已探索了零样本的可行性，但它们通常依赖于预处理好的3D点云，工作流程静态固化，本质上将定位任务退化为了候选框匹配问题。为了摆脱这种依赖，我们的核心动机是对该任务进行解耦：利用2D VLM来解决复杂的空间语义理解，同时依靠确定性的多视图几何来实例化3D结构。基于这一洞察，我们提出了"Think, Act, Build (TAB)"——一个动态的智能体框架，它将3D-VG任务重新构建为直接在原始RGB-D数据流上运行的生成式2D到3D重建范式。具体而言，在专门的3D-VG技能引导下，我们的VLM智能体动态调用视觉工具，跨2D帧对目标进行跟踪与重建。至关重要的是，为了克服……

    arXiv:2604.00528v3 Announce Type: replace-cross  Abstract: 3D Visual Grounding (3D-VG) aims to localize objects in 3D scenes via natural language descriptions. While recent advancements leveraging Vision-Language Models (VLMs) have explored zero-shot possibilities, they typically suffer from a static workflow relying on preprocessed 3D point clouds, essentially degrading grounding into proposal matching. To bypass this reliance, our core motivation is to decouple the task: leveraging 2D VLMs to resolve complex spatial semantics, while relying on deterministic multi-view geometry to instantiate the 3D structure. Driven by this insight, we propose "Think, Act, Build (TAB)", a dynamic agentic framework that reformulates 3D-VG tasks as a generative 2D-to-3D reconstruction paradigm operating directly on raw RGB-D streams. Specifically, guided by a specialized 3D-VG skill, our VLM agent dynamically invokes visual tools to track and reconstruct the target across 2D frames. Crucially, to overc
    
[^411]: AdapTS：面向多类别与持续视觉异常检测的轻量级师生方法

    AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection

    [https://arxiv.org/abs/2603.17530](https://arxiv.org/abs/2603.17530)

    AdapTS提出了一种轻量级师生框架，通过共享冻结骨干网络和基于原型识别的轻量适配器选择机制，无需重放缓冲区即可实现多类别与持续视觉异常检测，在大幅降低内存占用的同时保持优异性能。

    

    视觉异常检测对于工业检测至关重要。然而，大多数方法都是针对单类别模型设计的，而师生方法虽然高效，却并未针对持续学习场景进行设计，在该场景下它们需要一个重放缓冲区。为填补这一空白，我们提出了AdapTS，一个面向多类别和持续学习场景的统一师生框架，并针对资源受限的部署环境进行了优化。一个共享的冻结骨干网络同时充当教师和学生，轻量级的可训练适配器在推理时通过基于原型的任务识别机制（准确率达99%）进行选择，无需保留历史数据即可存储每个类别的知识。训练过程通过在合成Perlin噪声上使用分割引导的目标函数得到增强。在MVTec AD数据集上，AdapTS-M的I-ROC比STFPM提高了11个百分点，所需内存仅为Dinomaly的4.3分之一，而Dinomaly仅领先4个百分点。在持续学习场景中，AdapTS-M达到了最佳的像素级性能。

    arXiv:2603.17530v2 Announce Type: replace-cross  Abstract: Visual Anomaly Detection (VAD) is crucial for industrial inspection. However, most methods are designed for single-category models, and Teacher-Student (TS) approaches, despite their efficiency, have not been designed for the Continual scenario, where they need a replay buffer. To bridge this gap, we propose AdapTS, a unified TS framework for multi-class and continual settings, optimized for resource-constrained deployment. A single shared frozen backbone acts as teacher and student, and lightweight trainable adapters, selected at inference by a prototype-based task identification (99% accuracy), store the knowledge of each category without keeping past data. Training is enhanced via a segmentation-guided objective on synthetic Perlin noise. On MVTec AD, AdapTS-M improves I-ROC by 11 points over STFPM and needs 4.3 times less memory than Dinomaly, which is 4 points ahead. In the Continual scenario, AdapTS-M reaches the best pix
    
[^412]: STAIRS-Former：面向离线多任务多智能体强化学习的交错递归结构时空注意力Transformer

    STAIRS-Former: Spatio-Temporal Attention with Interleaved Recursive Structure Transformer for Offline Multi-task Multi-agent Reinforcement Learning

    [https://arxiv.org/abs/2603.11691](https://arxiv.org/abs/2603.11691)

    本文提出STAIRS-Former，一种具有空间和时间层次结构的Transformer架构，通过对关键标记的有效注意力和标记丢弃机制，捕捉长交互历史并泛化到不同智能体数量的场景，从而提升离线多任务多智能体强化学习的性能。

    

    使用多任务数据集的离线多智能体强化学习（MARL）由于各任务中智能体数量不同以及需要泛化到未见场景而极具挑战性。先前的工作采用带有观测标记化和分层技能学习的Transformer来解决这些问题。然而，这些方法未能充分利用Transformer注意力机制来实现智能体间的协调，并且仅依赖单一的历史标记，这限制了它们在部分可观测MARL环境中捕捉长时程时间依赖关系的能力。本文提出STAIRS-Former，这是一种通过空间和时间层次结构增强的Transformer架构，能够在捕捉长交互历史的同时对关键标记进行有效的注意力计算。我们进一步引入标记丢弃机制，以增强模型在不同智能体数量下的鲁棒性和泛化能力。在包括SMAC、SMAC-v2在内的多样化多智能体基准上进行了大量实验。

    arXiv:2603.11691v2 Announce Type: replace  Abstract: Offline multi-agent reinforcement learning (MARL) with multi-task datasets is challenging due to varying numbers of agents across tasks and the need to generalize to unseen scenarios. Prior works employ transformers with observation tokenization and hierarchical skill learning to address these issues. However, they underutilize the transformer attention mechanism for inter-agent coordination and rely on a single history token, which limits their ability to capture long-horizon temporal dependencies in partially observable MARL settings. In this paper, we propose STAIRS-Former, a transformer architecture augmented with spatial and temporal hierarchies that enables effective attention over critical tokens while capturing long interaction histories. We further introduce token dropout to enhance robustness and generalization across varying agent populations. Extensive experiments on diverse multi-agent benchmarks, including SMAC, SMAC-v2
    
[^413]: 发言还是保持沉默：多方对话中的上下文感知话轮转换

    Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue

    [https://arxiv.org/abs/2603.11409](https://arxiv.org/abs/2603.11409)

    该论文提出“上下文感知话轮转换”新任务——在多方对话的停顿处判断AI助手应发言还是沉默，发现八个现有大语言模型零样本下均无法胜任，并借助带推理轨迹的监督微调将平衡准确率提升最多23个百分点。

    

    现有的语音AI助手将每个检测到的停顿都视为发言的邀请。这种方式在双人对话中行之有效，但在多方对话场景中——即AI助手与多个说话者共同参与对话时——停顿大量存在且含义模糊。一个在每个停顿处都发言的助手会变得具有干扰性而非有用性。在这项工作中，我们提出了上下文感知话轮转换的概念：在每次检测到停顿时，基于完整的对话上下文，我们的方法来决定助手应该发言还是保持沉默。我们构建了一个包含超过12万条标注对话的基准数据集，涵盖三个多方对话语料库。通过对八个近期大语言模型的评估，我们发现它们在零样本提示下始终无法完成上下文感知话轮转换任务。随后，我们提出了一种带有推理轨迹的监督微调方法，将平衡准确率提升了最多23个百分点。我们的研究结果表明，上下文感知话轮转换并非一种涌现能力（原文在此处截断）。

    arXiv:2603.11409v2 Announce Type: replace  Abstract: Existing voice AI assistants treat every detected pause as an invitation to speak. This works in dyadic dialogue, but in multi-party settings, where an AI assistant participates alongside multiple speakers, pauses are abundant and ambiguous. An assistant that speaks on every pause becomes disruptive rather than useful. In this work, we formulate context-aware turn-taking: at every detected pause, given the full conversation context, our method decides whether the assistant should speak or stay silent. We introduce a benchmark of over 120K labeled conversations spanning three multi-party corpora. Evaluating eight recent large language models, we find that they consistently fail at context-aware turn-taking under zero-shot prompting. We then propose a supervised fine-tuning approach with reasoning traces, improving balanced accuracy by up to 23 percentage points. Our findings suggest that context-aware turn-taking is not an emergent ca
    
[^414]: SoundWeaver：面向文本到音频扩散模型服务的组合式热启动方法

    SoundWeaver: Compositional Warm-Starting for Text-to-Audio Diffusion Serving

    [https://arxiv.org/abs/2603.07865](https://arxiv.org/abs/2603.07865)

    SoundWeaver是首个无需训练的组合式跨示例声学复用框架，通过从缓存音频片段构建提示条件化声学先验并热启动扩散过程，显著加速文本到音频扩散模型的推理。

    

    文本到音频（T2A）扩散模型能够生成高质量音频，但需要数十次神经函数评估（NFE），导致显著的推理延迟。现有加速方法主要对每条生成轨迹进行独立优化，即便先前示例中存在可复用的声学结构。我们提出SoundWeaver，这是首个无需训练的组合式跨示例声学复用框架。SoundWeaver不再从纯噪声开始生成，而是从缓存的音频中提取片段构建提示条件化的声学先验，并从中间噪声水平热启动扩散过程。我们进一步引入声学组合图（ACG），利用神经编解码器表示和片段级识别来确定跨缓存转移的候选片段，并通过结构化路径搜索联合优化提示相关性与声学连续性。我们还建立了热启动误差界限，分析了……

    arXiv:2603.07865v2 Announce Type: cross  Abstract: Text-to-audio (T2A) diffusion models generate high-quality audio but require tens of neural function evaluations (NFEs), resulting in substantial inference latency. Existing acceleration methods primarily optimize each generation trajectory independently, even when reusable acoustic structure exists in prior examples. We introduce SoundWeaver, the first training-free framework for compositional cross example acoustic reuse. Rather than generating from pure noise, SoundWeaver constructs a prompt-conditioned acoustic prior from segments distributed across cached audios and warm-starts diffusion from an intermediate noise level. We introduce an Acoustic Composition Graph (ACG), which uses neural codec representations and segment-level identification to identify candidate cross-cache transitions and structured path search to jointly optimize prompt relevance and acoustic continuity. We further establish a warm-start error bound showing how
    
[^415]: VI-CuRL：通过置信度引导的方差缩减稳定无验证器强化学习推理

    VI-CuRL: Stabilizing Verifier-Independent Reinforcement Learning Reasoning via Confidence-Guided Variance Reduction

    [https://arxiv.org/abs/2602.12579](https://arxiv.org/abs/2602.12579)

    提出VI-CuRL框架，利用模型内在置信度构建课程并优先处理高置信度样本，有效降低梯度方差，从而解决无验证器强化学习推理中的训练崩溃问题。

    

    基于可验证奖励的强化学习（RLVR）已成为增强大型语言模型（LLM）推理能力的主流范式，但其对外部验证器的依赖限制了可扩展性。近期研究发现，RLVR主要是通过激发模型的潜在能力来发挥作用，这推动了无验证器算法的发展。然而，在此类设置中，诸如组相对策略优化（GRPO）等标准方法面临一个关键挑战：破坏性的梯度方差往往导致训练崩溃。为解决这一问题，我们提出了无验证器课程强化学习（VI-CuRL），该框架利用模型的内在置信度构建独立于外部验证器的课程。通过优先处理高置信度样本，VI-CuRL有效管理偏差-方差权衡，专门针对动作方差和问题方差的缩减进行优化。

    arXiv:2602.12579v3 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a dominant paradigm for enhancing Large Language Models (LLMs) reasoning, yet its reliance on external verifiers limits its scalability. Recent findings suggest that RLVR primarily functions by eliciting latent capabilities, motivating the development of verifier-free algorithms. However, in such settings, standard methods like Group Relative Policy Optimization face a critical challenge: destructive gradient variance that often leads to training collapse. To address this issue, we introduce Verifier-Independent Curriculum Reinforcement Learning (VI-CuRL), a framework that leverages the model's intrinsic confidence to construct a curriculum independent from external verifiers. By prioritizing high-confidence samples, VI-CuRL effectively manages the bias-variance trade-off, specifically targeting the reduction of action and problem variance. We provide a rigoro
    
[^416]: 基于JEPA嵌入的汽车时间序列数据在线监测框架

    Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings

    [https://arxiv.org/abs/2602.09985](https://arxiv.org/abs/2602.09985)

    该论文提出了一种基于JEPA自监督嵌入的在线监测框架，无需异常标签即可检测汽车目标状态表示中的异常，为自动驾驶车辆的安全运行监控提供了新方案。

    

    随着自动驾驶车辆的推出，必须采取措施确保其安全运行。为了监督已经在运行中的系统，通常会采用监测框架。这些框架在后台持续在线运行，监控系统状态并记录异常。本工作提出了一种在线监测框架，用于检测目标状态表示中的异常。其中一个关键挑战是在没有异常标签的情况下构建异常检测框架，而对于未知异常，异常标签通常是不可获得的。为解决这一问题，本工作采用了一种自监督嵌入方法，将目标数据转换到潜在表示空间中。为此，构建了一个基于JEPA的自监督预测任务，使得无需异常标签即可进行训练，并生成丰富的目标嵌入。由此得到的具有强表达能力的JEPA嵌入可作为现有异常检测方法的输入。

    arXiv:2602.09985v2 Announce Type: replace-cross  Abstract: As autonomous vehicles are rolled out, measures must be taken to ensure their safe operation. In order to supervise a system that is already in operation, monitoring frameworks are frequently employed. These run continuously online in the background, supervising the system status and recording anomalies. This work proposes an online monitoring framework to detect anomalies in object state representations. Thereby, a key challenge is creating a framework for anomaly detection without anomaly labels, which are usually unavailable for unknown anomalies. To address this issue, this work applies a self-supervised embedding method to translate object data into a latent representation space. For this, a JEPA-based self-supervised prediction task is constructed, allowing training without anomaly labels and the creation of rich object embeddings. The resulting expressive JEPA embeddings serve as input for established anomaly detection m
    
[^417]: SparVAR：探索视觉自回归建模中的稀疏性以实现免训练加速

    SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration

    [https://arxiv.org/abs/2602.04361](https://arxiv.org/abs/2602.04361)

    SparVAR是一种免训练的VAR模型加速框架，通过利用注意力汇聚点、跨尺度激活相似性和局部性三大特性，从稀疏决策尺度动态预测高分辨率尺度的稀疏注意力模式，在不跳过尺度、不牺牲图像质量的前提下大幅降低计算复杂度和推理延迟。

    

    视觉自回归建模因其创新性的“下一尺度预测”范式而受到广泛关注。然而，主流的VAR范式在每个自回归步骤中都会关注所有历史尺度的全部token。随着下一尺度分辨率的增大，注意力的计算复杂度随分辨率呈四次方增长，造成显著的推理延迟。以往的加速方法通常跳过高分辨率尺度，这虽然加快了推理速度，但丢弃了高频细节并损害了图像质量。为解决这些问题，我们提出了SparVAR，一个免训练的加速框架，它利用了VAR注意力的三个特性：（i）强注意力汇聚点，（ii）跨尺度激活相似性，以及（iii）显著的局部性。具体而言，我们从稀疏决策尺度动态预测后续高分辨率尺度的稀疏注意力模式，并通过高效的方法构建尺度自相似的稀疏注意力。

    arXiv:2602.04361v3 Announce Type: replace-cross  Abstract: Visual AutoRegressive (VAR) modeling has garnered significant attention for its innovative next-scale prediction paradigm. However, mainstream VAR paradigms attend to all tokens across historical scales at each autoregressive step. As the next scale resolution grows, the computational complexity of attention increases quartically with resolution, causing substantial latency. Prior accelerations often skip high-resolution scales, which speeds up inference but discards high-frequency details and harms image quality. To address these problems, we present SparVAR, a training-free acceleration framework that exploits three properties of VAR attention: (i) strong attention sinks, (ii) cross-scale activation similarity, and (iii) pronounced locality. Specifically, we dynamically predict the sparse attention pattern of later high-resolution scales from a sparse decision scale, and construct scale self-similar sparse attention via an ef
    
[^418]: FlowSteer：迈向由智能体通过强化渐进式画布编辑来设计智能体工作流

    FlowSteer: Towards Agents Designing Agentic Workflows via Reinforced Progressive Canvas Editing

    [https://arxiv.org/abs/2602.01664](https://arxiv.org/abs/2602.01664)

    FlowSteer 提出了一种新范式，由单个轻量级智能体在可执行的工作流画布环境中，通过强化学习训练的渐进式原子画布编辑，端到端地自主设计供下游执行器运行的智能体工作流，并获得语法检查级别的实时执行反馈。

    

    近年来，智能体工作流已被广泛应用于解决复杂的人类任务。然而，现有的工作流构建仍面临若干关键挑战，包括依赖人工的工作流构建方式、缺乏图级别的执行反馈，以及无法在长时程构建过程中进行在线错误修复。为解决这些挑战，我们提出了 FlowSteer——一种“智能体设计智能体工作流”的新范式，即由单个智能体自身端到端地设计出供下游执行器运行的工作流。为支持这一范式，我们引入了工作流画布，这是一种新颖的可执行图状态环境，能够为每一次原子编辑返回经过语法检查的执行反馈。基于该画布，我们进一步提出了强化渐进式画布编辑，其中一个轻量级策略智能体在每一轮基于真实的画布反馈发出一次原子编辑，并通过强化学习进行端到端训练。此外，FlowSteer……（摘要原文在此处截断）

    arXiv:2602.01664v5 Announce Type: replace  Abstract: In recent years, agentic workflows have been widely applied to solve complex human tasks. However, existing workflow construction still faces key challenges, including human-dependent workflow construction, the lack of graph-level execution feedback, and the inability to repair errors in-loop during long-horizon construction. To address these challenges, we propose FlowSteer, a new paradigm of Agent Designing Agentic Workflows - a single agent itself end-to-end designs the workflow that a downstream executor runs. To support this paradigm, we introduce the Workflow Canvas, a novel executable graph-state environment that returns syntax-checked execution feedback for every atomic edit. Built on the canvas, we further propose Reinforced Progressive Canvas Editing, in which a lightweight policy agent issues one atomic edit per turn conditioned on real canvas feedback, and is trained end-to-end via reinforcement learning. Moreover, FlowSt
    
[^419]: 面向函数调用智能体的合成数据中的语言与参数多样性

    Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents

    [https://arxiv.org/abs/2601.17829](https://arxiv.org/abs/2601.17829)

    该论文提出了一种无需手工规则或分类体系、通过在查询和参数两个层面优化通用多样性指标来生成函数调用智能体合成训练数据的方法，在保持正确性相当的同时显著提升了数据的语言与参数多样性。

    

    构建函数调用智能体已成为扩展模型能力的一条有前景的途径。该任务的一个主要挑战是获取高质量的多样化训练数据。以往的工作强调函数、调用模式和交互轮次的多样性，但请求的语言多样性以及参数（如 \texttt{city\_name}、\texttt{stock\_ticker}）的覆盖范围仍未得到充分探索。我们提出了一种方法，通过在查询和参数两个层面优化通用多样性指标来生成合成数据集，无需依赖手工构建的规则或分类体系，因此对不同用例具有鲁棒性。我们通过内在和外在测试验证了该技术的有效性，并将其与最先进（SoTA）的数据生成方法进行比较。结果表明，我们在多样性方面优于基线方法，同时保持了相当的正确性。此外，当用作训练集时……（原文摘要至此截断）

    arXiv:2601.17829v2 Announce Type: replace-cross  Abstract: The construction of function calling agents has emerged as a promising avenue for extending model capabilities. A major challenge for this task is obtaining high quality diverse data for training. Prior work emphasizes diversity in functions, invocation patterns, and interaction turns, yet linguistic diversity of requests and coverage of arguments (e.g., \texttt{city\_name}, \texttt{stock\_ticker}) remain underexplored. We propose a method that generates synthetic datasets via optimizing general-purpose diversity metrics across both queries and arguments, without relying on hand-crafted rules or taxonomies, making it robust to different usecases. We demonstrate the effectiveness of our technique via both intrinsic and extrinsic testing, comparing it to SoTA data generation methods. We show a superiority over baselines in terms of diversity, while keeping comparable correctness. Additionally, when used as a training set, the mod
    
[^420]: 多模态的“球门柱”在哪里？论基础模型识别视频中精彩片段的能力

    Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Highlights in Videos

    [https://arxiv.org/abs/2601.16333](https://arxiv.org/abs/2601.16333)

    本研究利用足球比赛官方集锦零成本构建了一个评估数据集，用以测试多模态基础模型识别视频中重要事件的能力，发现现有最先进模型无法可靠地理解事件的上下文重要性。

    

    基础模型被广泛应用于许多需要对按时间排序的多模态事件进行语言理解与生成的现实场景中。在这项工作中，我们研究了模型识别视频中最重要的子事件的能力，这是对多模态事件进行总结、推理或叙述的基础前提。具体而言，我们聚焦于足球比赛，评估模型区分比赛中重要与非重要子事件的能力。为此，我们利用公开可得的足球精彩集锦构建了一个新数据集，这些集锦包含了由专业专家认定的比赛最重要时刻，且无需任何额外的标注成本。利用该数据集，我们比较了多个最先进的多模态模型，结果表明它们无法可靠地捕捉上下文重要性。超越标准评估指标的进一步分析揭示了模型的倾向……

    arXiv:2601.16333v3 Announce Type: replace-cross  Abstract: Foundation models are used for many real-world applications involving language understanding and generation from temporally-ordered multimodal events. In this work, we study the ability of models to identify the most important sub-events in a video, which is a fundamental prerequisite for summarizing, reasoning over, or narrating multimodal events. Specifically, we focus on football games and evaluate models on their ability to distinguish between important and non-important sub-events in a game. To this end, we construct a new dataset by leveraging publicly available football highlight reels containing the most important moments of a game according to professional experts, without any additional annotation costs. Using our dataset, we compare several state-of-the-art multimodal models and show that they fail to reliably capture contextual importance. Analyses of models beyond standard evaluation metrics reveal their tendency t
    
[^421]: 多维提示链提升开放域对话生成

    Multi-Dimensional Prompt Chaining to Improve Open-Domain Dialogue Generation

    [https://arxiv.org/abs/2601.01037](https://arxiv.org/abs/2601.01037)

    提出一种整合自然度、连贯性和吸引力三个维度的多维提示链框架，使小语言模型（如Llama-2-7B）在开放域对话生成中达到与大型模型相当的性能。

    

    小语言模型（SLM）具有显著的部署优势，但在开放域场景中往往难以匹敌更大模型的对话质量。本文提出了一种多维提示链框架，整合自然度、连贯性和吸引力三个维度，以增强开放域对话生成的人性化程度。我们将该框架应用于两个小语言模型——TinyLlama和Llama-2-7B，并将其性能与更大模型生成的回复进行基准对比，包括Llama-2-70B和GPT-3.5 Turbo。随后，我们采用自动评估和人工评估，基于多样性、上下文连贯性以及整体质量来评估回复。结果表明，完整框架可将回复多样性最高提升29%，上下文连贯性最高提升28%，吸引力和自然度最高提升29%。值得注意的是，Llama-2-7B达到了与更大模型相当的性能。

    arXiv:2601.01037v2 Announce Type: replace-cross  Abstract: Small language models (SLMs) offer significant deployment advantages but often struggle to match the dialogue quality of larger models in open-domain settings. In this paper, we propose a multi-dimensional prompt-chaining framework that integrates Naturalness, Coherence, and Engagingness dimensions to enhance human-likeness in open-domain dialogue generation. We apply the framework to two SLMs, TinyLlama and Llama-2-7B, and benchmark their performance against responses generated by substantially larger models, including Llama-2-70B and GPT-3.5 Turbo. We then employ automatic and human evaluation to assess the responses based on diversity, contextual coherence, as well as overall quality. Results show that the full framework improves response diversity by up to 29%, contextual coherence by up to 28%, and engagingness as well as naturalness by up to 29%. Notably, Llama-2-7B achieves performance comparable to substantially larger 
    
[^422]: 驯服幻觉：通过反事实视频生成提升多模态大语言模型的视频理解能力

    Taming Hallucinations: Boosting MLLMs' Video Understanding via Counterfactual Video Generation

    [https://arxiv.org/abs/2512.24271](https://arxiv.org/abs/2512.24271)

    该论文提出DualityForge框架，利用可控扩散视频编辑自动合成反事实视频与问答数据对，通过对比训练抑制多模态大语言模型对语言先验的过度依赖，从而减少视觉幻觉并增强视频理解能力。

    

    多模态大语言模型（MLLMs）在视频理解方面取得了显著进展。然而，它们存在一个关键性缺陷：过度依赖语言先验，这可能导致视觉上无根据的幻觉，尤其是在处理违背常识的反事实视频时。这一局限源于文本与视频之间内在的数据不平衡，而由于收集和标注反事实数据的高昂成本，该问题难以解决。为此，我们提出了DualityForge，一种新颖的反事实数据合成框架，它采用可控的、基于扩散模型的视频编辑技术，将真实世界视频转化为反事实场景。通过在视频编辑和问答生成过程中嵌入结构化的上下文信息，该框架能够自动生成高质量的问答对，以及用于对比训练的原始-编辑视频对。

    arXiv:2512.24271v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) have made remarkable progress in video understanding. However, they suffer from a critical vulnerability: an over-reliance on language priors, which can lead to visual ungrounded hallucinations, especially when processing counterfactual videos that defy common sense. This limitation, stemming from the intrinsic data imbalance between text and video, is challenging to address due to the substantial cost of collecting and annotating counterfactual data. To address this, we introduce DualityForge, a novel counterfactual data synthesis framework that employs controllable, diffusion-based video editing to transform real-world videos into counterfactual scenarios. By embedding structured contextual information into the video editing and QA generation processes, the framework automatically produces high-quality QA pairs together with original-edited video pairs for contrastive training. Based o
    
[^423]: AI可解释性中的“死鲑鱼”

    The Dead Salmons of AI Interpretability

    [https://arxiv.org/abs/2512.18792](https://arxiv.org/abs/2512.18792)

    该论文借神经科学中著名的“死鲑鱼”事件警示当前AI可解释性方法（如特征归因、探测、稀疏自编码）可能对随机初始化的神经网络生成虚假解释，并提出应将解释视为从计算轨迹中推断的统计模型参数这一统计-因果重构框架。

    

    在一项引人注目的神经科学研究中，作者将一条死鲑鱼放入MRI扫描仪中，并向其展示人类社交情境的图像。令人惊讶的是，当时的标准分析竟报告出与社会情绪相关的脑区。当然，其解释并非超自然的认知，而是一个关于错误应用统计推断的警示故事。在AI可解释性领域，类似的“死鲑鱼”伪影现象大量存在：特征归因、探测、稀疏自编码，甚至因果分析都可能为随机初始化的神经网络生成看似合理的解释。在考察这一现象后，我们的立场是主张一种务实的统计-因果重构：对计算系统的解释应被视为（统计）模型的参数，从计算轨迹中推断得出。这一视角超越了仅仅测量由有限采样导致的解释统计变异性……

    arXiv:2512.18792v2 Announce Type: replace  Abstract: In a striking neuroscience study, the authors placed a dead salmon in an MRI scanner and showed it images of humans in social situations. Astonishingly, standard analyses of the time reported brain regions predictive of social emotions. The explanation, of course, was not supernatural cognition but a cautionary tale about misapplied statistical inference. In AI interpretability, reports of similar "dead salmon" artifacts abound: feature attribution, probing, sparse auto-encoding, and even causal analyses can produce plausible-looking explanations for randomly initialized neural networks. Upon inspection of this phenomenon our position is to argue for a pragmatic statistical-causal reframing: explanations of computational systems should be treated as parameters of a (statistical) model, inferred from computational traces. This perspective goes beyond simply measuring statistical variability of explanations due to finite sampling of in
    
[^424]: BEAVER：一个高效确定性的大语言模型验证器

    BEAVER: An Efficient Deterministic LLM Verifier

    [https://arxiv.org/abs/2512.05439](https://arxiv.org/abs/2512.05439)

    BEAVER是首个实用的确定性LLM验证框架，利用令牌字典树和边界前沿系统地探索模型输出分布，为LLM响应满足特定属性的概率提供可证明的可靠界限，其Max-μ选择策略在任意容差下均是最优的，在安全任务上认证有风险提示的数量比采样验证器多出高达7.6倍。

    

    随着大语言模型（LLM）从研究原型过渡到生产系统，从业者需要在模型部署前验证模型输出的可靠方法。虽然基于采样的估计为模型行为提供了临时性的直觉判断，但它们无法提供可靠的保证。我们提出了BEAVER，这是首个用于计算LLM对提示的响应满足某一属性的概率的确定性、可靠界限的实用框架。给定一个提示和一个预言机，BEAVER使用令牌字典树和边界前沿系统地探索模型的输出分布，在每次迭代中都保持可证明的可靠界限。我们形式化了LLM验证问题，证明了BEAVER的界限是可靠的，并且其选择策略Max-μ在任意容差范围内对验证是最优的。在5个流行LLM上的3个重要安全任务中，BEAVER认证为有风险的提示数量比基于采样的验证器多出高达7.6倍。

    arXiv:2512.05439v3 Announce Type: replace  Abstract: As large language models (LLMs) transition from research prototypes to production systems, practitioners need reliable ways to verify model outputs before they are deployed. While sampling-based estimates provide an ad-hoc intuition of model behavior, they offer no sound guarantees. We present BEAVER, the first practical framework for computing deterministic, sound bounds on the probability that an LLM's response to a prompt satisfies a property. Given a prompt and an oracle, BEAVER systematically explores the model's output distribution using a *token trie* and a *frontier*, maintaining provably sound bounds at every iteration. We formalize the LLM verification problem, prove that BEAVER's bounds are sound and that its selection strategy Max-$\mu$ is optimal for verification upto any tolerance. On 3 important safety tasks across 5 popular LLMs, BEAVER certifies up to $7.6\times$ more prompts as risky than a sampling-based verifier w
    
[^425]: MemVerse：面向终身学习智能体的多模态记忆

    MemVerse: Multimodal Memory for Lifelong Learning Agents

    [https://arxiv.org/abs/2512.03627](https://arxiv.org/abs/2512.03627)

    该论文提出了MemVerse——一个模型无关、即插即用的多模态记忆框架，通过短期记忆、结构化长期记忆图和周期性知识蒸馏机制，使终身学习智能体能够实现快速召回、持续学习与自主知识演化。

    

    尽管多模态大语言模型在各个领域取得了显著进展，但AI智能体仍然缺乏保留过去信息的能力。没有可靠的记忆，智能体难以利用历史上下文，从而损害了动态环境中的长程推理能力和连贯交互。在本文中，我们提出了MemVerse，一个模型无关、即插即用的记忆框架，它将快速参数化召回与分层检索相结合，实现可扩展且自适应的多模态智能。MemVerse在为近期上下文维护短期记忆的同时，将原始经验转化为结构化的长期记忆图，从而支持持续学习和自主知识演化。为满足实时需求，我们引入了一种周期性蒸馏机制，将认知图中的关键知识压缩到参数模型中，在提供快速且可微分的记忆召回的同时保留……（原文摘要在此处截断）

    arXiv:2512.03627v3 Announce Type: replace  Abstract: Despite the remarkable advancements in multimodal large language models across various domains, AI agents still lack the ability to retain past information. Without reliable memory, agents struggle to leverage historical context, impairing long-horizon reasoning and coherent interactions in dynamic environments. In this paper, we introduce MemVerse, a model-agnostic, plug-and-play memory framework that combines fast parametric recall with hierarchical retrieval, enabling scalable and adaptive multimodal intelligence. MemVerse maintains short-term memory for recent context while transforming raw experiences into structured long-term memory graphs, empowering continuous learning and autonomous knowledge evolution. To meet real-time demands, we introduce a periodic distillation mechanism that compresses essential knowledge from the cognitive graph into the parametric model, providing fast and differentiable recall while preserving inter
    
[^426]: MADRA：面向风险感知具身规划的多智能体辩论

    MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning

    [https://arxiv.org/abs/2511.21460](https://arxiv.org/abs/2511.21460)

    MADRA提出了一种无需训练的多智能体辩论认知架构，通过元认知批判智能体和基于图尔敏模型的结构化论证来缓解大语言模型的“具身语义幻觉”、避免安全-效用权衡，并发布了SafeAware-VH基准用于评估具身智能体的物理风险感知能力。

    

    大型语言模型展现出令人印象深刻的推理能力，但常常遭受“具身语义幻觉”的困扰——生成的计划在语义上流畅，但由于缺乏扎根于现实的常识而在物理上不安全。现有的安全对齐方法（如RLHF或朴素的安全提示）通常陷入“安全-效用权衡”，导致对良性家庭指令的严重过度拒绝。为解决这一问题，我们提出了MADRA（面向风险感知的多智能体辩论），一种无需训练、模拟系统2深思式推理的认知架构。MADRA引入了一个元认知的批判智能体，使用源自图尔敏模型的结构化论证框架来评估同伴辩论，有效缓解了多智能体系统中的“从众心理”。我们还推出了SafeAware-VH基准，其中包含对抗性安全指令，旨在探测智能体对物理风险的敏感性。

    arXiv:2511.21460v2 Announce Type: replace  Abstract: Large Language Models (LLMs) exhibit impressive reasoning capabilities but often suffer from "Embodied Semantic Hallucinations" - generating plans that are semantically fluent but physically unsafe due to a lack of grounded common sense. Existing safety alignment methods, such as RLHF or naive safety prompting, typically fall into a "Safety-Utility Trade-off," resulting in severe over-rejection of benign household instructions. To address this, we propose MADRA (Multi-Agent Debate for Risk Awareness), a training-free cognitive architecture that mimics System-2 deliberation. MADRA introduces a meta-cognitive Critical Agent that evaluates peer debates using a structured argumentation framework derived from the Toulmin Model, effectively mitigating the "herd mentality" in multi-agent systems. We also introduce SafeAware-VH, a benchmark featuring adversarial safe instructions designed to probe agents' sensitivity to physical risks. Exten
    
[^427]: 战略技术中的约束性瓶颈：面向基础模型产业的工业脆弱性指数

    Binding Constraints in a Strategic Technology: An Industrial Vulnerability Index for the Foundation Model Industry

    [https://arxiv.org/abs/2510.23421](https://arxiv.org/abs/2510.23421)

    本文提出人工智能工业脆弱性指数（AIIVI），利用十四个集中度与可持续性参数并通过 O-Ring 函数形式聚合，用以评估基础模型产业中最具约束性的生产瓶颈，揭示了当前产业政策聚焦算力而忽视电网等环节的配置可能偏离真正的约束所在。

    

    arXiv:2510.23421v3 公告类型：replace-cross 摘要：针对人工智能的产业政策已变得相当可观。公共投资集中于算力领域——而私人产能扩张已在进行中——并且在电网互联与发电方面几乎缺位。这种资源配置是否准确反映了生产中最具约束性的瓶颈，尚未得到系统性测量。本文提出了人工智能工业脆弱性指数（AIIVI），作为评估前沿人工智能潜在生产约束的框架。我们从十四个集中度与可持续性参数构建了基础模型产业的投入脆弱性综合指数，并通过 O-Ring 函数形式进行聚合——在该形式中，模型能力受最受限投入的强烈影响。AIIVI 不应被解释为对生产约束的直接测量，而是提供了一个基于指数的近似……

    arXiv:2510.23421v3 Announce Type: replace-cross  Abstract: Industrial policy directed at artificial intelligence has become substantial. Public investment concentrates on compute, where private capacity expansion is already underway, and is largely absent from grid interconnection and generation. Whether that allocation accurately reflects the most binding constraint in production has not yet been systematically measured. This paper introduces the Artificial Intelligence Industrial Vulnerability Index (AIIVI) as a framework for assessing potential production constraints in frontier AI. We construct a composite index of input vulnerability for the foundation model industry from fourteen concentration and sustainability parameters, aggregated by an O-Ring functional form in which capability is strongly influenced by the most constrained input. AIIVI should not be interpreted as a direct measurement of production constraints. Rather, it provides an index-based approximation derived from o
    
[^428]: 通过反馈修复奖励函数以缓解奖励欺骗

    Repairing Reward Functions with Feedback to Mitigate Reward Hacking

    [https://arxiv.org/abs/2510.13036](https://arxiv.org/abs/2510.13036)

    该论文提出了基于偏好的奖励修复（PBRR）自动化迭代框架，通过从少量人类偏好数据中学习一个加性的、依赖于状态转移的修正项来修复人工指定的代理奖励函数，以低成本缓解奖励欺骗问题。

    

    为强化学习（RL）智能体人工设计的奖励函数经常与人类真实的、不可观测的目标不一致，因此只能作为代理目标。优化错误指定的代理奖励函数往往会导致奖励欺骗，从而产生与人类真实目标不一致的策略。另一种方法是进行基于人类反馈的强化学习，即通过收集人类对轨迹对的偏好从头学习一个奖励函数，然而构建此类数据集的成本高昂。为了解决这两种方法的局限性，我们提出了基于偏好的奖励修复（PBRR）：一个自动化迭代框架，通过从偏好中学习一个加性的、依赖于状态转移的修正项来修复人工指定的代理奖励函数。手动指定的奖励函数在真实目标下可能产生高度次优的策略，然而仅需对少量转移进行修正……（摘要原文在此处截断）

    arXiv:2510.13036v3 Announce Type: replace  Abstract: Human-designed reward functions for reinforcement learning (RL) agents are frequently misaligned with the humans' true, unobservable objectives, and thus act only as proxies. Optimizing for a misspecified proxy reward function often induces reward hacking, resulting in a policy misaligned with the human's true objectives. An alternative is to perform RL from human feedback, which involves learning a reward function from scratch by collecting human preferences over pairs of trajectories. However, building such datasets is costly. To address the limitations of both approaches, we propose Preference-Based Reward Repair (PBRR): an automated iterative framework that repairs a human-specified proxy reward function by learning an additive, transition-dependent correction term from preferences. A manually specified reward function can yield policies that are highly suboptimal under the ground-truth objective, yet corrections on only a few tr
    
[^429]: 大语言模型pass@k缩放规律的高效预测

    Efficient Prediction of pass@k Scaling in Large Language Models

    [https://arxiv.org/abs/2510.05197](https://arxiv.org/abs/2510.05197)

    本文提出了一种稳健的估计框架，能够在远小于大规模尝试次数的采样预算下高效准确地预测大语言模型的pass@k缩放行为，从而帮助模型提供商和监管机构评估模型的能力与风险。

    

    评估前沿AI系统的能力与风险是一个重要的研究领域；最近的研究表明，重复采样可以显著地同时提升这两者。例如，重复采样使模型能够解决困难的数学和编程问题，但也可能增加其被越狱攻击的风险。这些结果引出了一个关键问题：在采样预算远小于实际尝试次数的情况下，如何准确预测模型在扩展到海量尝试次数时的行为？这个问题与每天为数亿用户提供服务的模型提供商，以及寻求防止危害的政府监管机构直接相关。为回答这一问题，我们做出了三项贡献。首先，我们发现用于拟合这些规律的标准方法存在统计上的缺陷，阻碍了预测的准确性，尤其是在数据有限的场景中。其次，我们通过引入一个稳健的估计框架来弥补这些缺陷

    arXiv:2510.05197v2 Announce Type: replace  Abstract: Assessing the capabilities and risks of frontier AI systems is a critical area of research; recent work has shown that repeated sampling can dramatically increase both. For instance, repeated sampling allows models to solve difficult math and coding problems, but it may also increase their risk of being jailbroken. Such results raise a crucial question: how can one accurately predict a model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget? This question is directly relevant to model providers, who serve hundreds of millions of users daily, and to governmental regulators, who seek to prevent harms. To answer this question, we make three contributions. First, we find that standard methods for fitting these laws suffer from statistical shortcomings that hinder predictions, especially in data-limited scenarios. Second, we remedy these shortcomings by introducing a robust estimation framework
    
[^430]: 基于Hadamard变换辅助算子重新思考神经网络中的乘法

    Rethinking Multiplication in Neural Networks with Hadamard Transform-Assisted Operators

    [https://arxiv.org/abs/2509.23103](https://arxiv.org/abs/2509.23103)

    本文提出Hadamard变换辅助的避免乘法（HMA）算子，并通过门控竞争机制发现中间卷积层更适合采用HMA算子，而最早层和最终分类器仍需传统乘法，从而在存内计算架构中兼顾计算效率与模型精度。

    

    避免乘法的神经算子对于存内计算架构具有吸引力，但直接替换传统卷积往往会导致精度下降。我们研究了这一限制究竟是源于避免乘法本身，还是源于算子所作用的表示形式。在本工作中，我们提出了Hadamard变换辅助的避免乘法（HMA）算子，将Walsh-Hadamard变换特征与避免乘法的通道混合相结合。为了确定传统乘法在何处仍然必要，我们采用了一种门控公式，使传统算子和HMA算子通过学习到的逐层系数进行竞争。ResNet-18实验表明，中间卷积层对HMA赋予了更大的权重，而最早的层和最终分类器则保留了传统计算的更大贡献。基于这一观察，我们构建了一个（摘要在此处截断）

    arXiv:2509.23103v3 Announce Type: replace-cross  Abstract: Multiplication-avoiding neural operators are attractive for compute-in-memory (CIM) architectures, but directly replacing conventional convolutions often causes accuracy degradation. We investigate whether this limitation arises from multiplication avoidance itself or from the representation on which the operator acts. In this work, we introduce Hadamard Transform-assisted multiplication-avoiding (HMA) operators that combine Walsh--Hadamard transform features with multiplication-avoiding channel mixing. To identify where conventional multiplication remains necessary, we employ a gated formulation in which conventional and HMA operators compete through learned layer-wise coefficients. ResNet-18 experiments show that intermediate convolutional layers assign greater weight to HMA, whereas the earliest layers and final classifier retain greater contribution from conventional computation. Guided by this observation, we construct a c
    
[^431]: 复杂度正则化的近端策略优化

    Complexity-Regularized Proximal Policy Optimization

    [https://arxiv.org/abs/2509.20509](https://arxiv.org/abs/2509.20509)

    该论文提出用香农熵与非平衡度乘积构成的自调节复杂度项替代传统熵正则化来改进近端策略优化（PPO），在保持策略有益随机性的同时，当策略高度不确定时降低正则化压力，从而让学习更专注于奖励优化。

    

    策略梯度方法通常依赖熵正则化来防止过早收敛。然而，不加区分地最大化熵会将策略推向均匀分布，若调参不当，往往会压过奖励信号的作用。我们提出用一种自调节的复杂度项来替代标准的熵项，该复杂度项定义为香农熵与非平衡度的乘积，用于量化与均匀分布之间的距离。与偏好最大无序性的纯熵不同，这种复杂度度量在完全确定性分布和完全均匀分布下均为零，即只有在秩序与随机性之间表现出有意义相互作用的系统中，它才严格为正。这些特性确保策略保持有益的随机性，同时在策略高度不确定时降低正则化压力，使学习能够专注于奖励优化。我们提出了复杂度正则化的近端策略优化（CR-PPO）……

    arXiv:2509.20509v3 Announce Type: replace-cross  Abstract: Policy gradient methods usually rely on entropy regularization to prevent premature convergence. However, maximizing entropy indiscriminately pushes the policy towards a uniform distribution, often overriding the reward signal if not optimally tuned. We propose replacing the standard entropy term with a self-regulating complexity term, defined as the product of Shannon entropy and disequilibrium, which quantifies the distance from the uniform distribution. Unlike pure entropy, which favors maximal disorder, this complexity measure is zero for both fully deterministic and perfectly uniform distributions, i.e., it is strictly positive for systems that exhibit a meaningful interplay between order and randomness. These properties ensure the policy maintains beneficial stochasticity while reducing regularization pressure when the policy is highly uncertain, allowing learning to focus on reward optimization. We introduce Complexity-R
    
[^432]: Patient-Zero：无需真实患者数据即可将合成患者智能体扩展至真实世界分布

    Patient-Zero: Scaling Synthetic Patient Agents to Real-World Distributions without Real Patient Data

    [https://arxiv.org/abs/2509.11078](https://arxiv.org/abs/2509.11078)

    该论文提出Patient-Zero框架，无需任何真实患者数据，通过医学对齐分层合成从医学知识生成多样化患者记录，并借助双轨认知记忆系统在动态临床交互中保持逻辑一致性。

    

    使用大语言模型（LLM）进行合成数据生成已成为医学领域缓解数据稀缺与隐私限制的一种有前景的解决方案。然而，现有方法仍受制于其衍生性质，依赖真实世界的记录，这带来了隐私风险和分布偏差。此外，当前的患者智能体面临“稳定性-可塑性困境”，在动态问询过程中难以保持临床一致性。为应对这些挑战，我们提出了Patient-Zero，一个无需真实医疗记录即可从头开始（ab initio）进行患者模拟的新型框架。我们的医学对齐分层合成框架通过分层属性排列，从医学知识中生成全面且多样化的患者记录。为支持严格的临床交互，我们设计了双轨认知记忆系统，使智能体能够在动态更新记忆的同时保持逻辑一致性。

    arXiv:2509.11078v3 Announce Type: replace  Abstract: Synthetic data generation with Large Language Models (LLMs) has emerged as a promising solution in the medical domain to mitigate data scarcity and privacy constraints. However, existing approaches remain constrained by their derivative nature, relying on real-world records, which pose privacy risks and distribution biases. Furthermore, current patient agents face the Stability-Plasticity Dilemma, struggling to maintain clinical consistency during dynamic inquiries. To address these challenges, we introduce Patient-Zero, a novel framework for ab initio patient simulation that requires no real medical records. Our Medically-Aligned Hierarchical Synthesis framework generates comprehensive and diverse patient records from medical knowledge via stratified attribute permutation. To support rigorous clinical interaction, we design a Dual-Track Cognitive Memory System to enable agents dynamically update memory while preserving logical consi
    
[^433]: 聚类无关假设下的联邦学习一次性聚类

    One-Shot Clustering for Federated Learning Under Clustering-Agnostic Assumption

    [https://arxiv.org/abs/2509.01587](https://arxiv.org/abs/2509.01587)

    本文提出了一种聚类无关的一次性聚类联邦学习算法（OCFL），能够自动检测出最适合聚类的最早时机，从而实现个性化模型的训练。

    

    联邦学习（FL）是一种被广泛采用的去中心化学习范式，它允许从多个数据源训练同一个模型，而无需在参与客户端之间直接传输数据。自2016年诞生以来，联邦学习已分化出众多子领域，用以处理特定应用的问题，例如数据异构性或资源分配。其中一个子领域——聚类联邦学习（CFL）——致力于解决将客户端群体划分为不同群组以提供个性化模型的问题。尽管该领域已发表了一些杰出的工作，但这一问题在很大程度上仍未被充分探索，因为其基本假设和设置与标准联邦学习略有不同。在本工作中，我们提出了一次性聚类联邦学习（OCFL），这是一种聚类无关的算法，能够自动检测出适合进行聚类的最早时机。我们通过实验验证……

    arXiv:2509.01587v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) is a widespread and well-adopted paradigm of decentralised learning that allows training one model from multiple sources without the need to transfer data between participating clients directly. Since its inception in 2016, it has been divided into numerous subfields that deal with application-specific issues, such as data heterogeneity or resource allocation. One such sub-field, Clustered Federated Learning (CFL), is dealing with the problem of clustering the population of clients into separate cohorts to deliver personalised models. Although a few remarkable works have been published in this domain, the problem remains largely unexplored, as its basic assumptions and settings differ slightly from those of standard FL. In this work, we present One-Shot Clustered Federated Learning (OCFL), a clustering-agnostic algorithm that can automatically detect the earliest suitable moment for clustering. We empiri
    
[^434]: CrafterDojo：用于在Crafter中构建开放式具身智能体的集成研究套件

    CrafterDojo: An Integrated Research Suite for Building Open-Ended Embodied Agents in Crafter

    [https://arxiv.org/abs/2508.13530](https://arxiv.org/abs/2508.13530)

    CrafterDojo通过提供基础模型（CrafterVPT、CrafterCLIP、CrafterSteve-1）、合成数据集（CrafterPlay、CrafterCaption）及自动数据生成工具包，为轻量级环境Crafter构建了完整的基础模型生态，使其成为Minecraft之外研究开放式具身智能的高效测试平台。

    

    基础模型是构建通用具身智能体的核心。Minecraft已成为这一范式的主导测试平台，但其计算和工程开销限制了快速迭代与可控分析。Crafter提供了一种轻量级替代方案，却缺乏此类研究所需的基础模型生态系统。我们提出CrafterDojo，这是一套包含基础模型、数据集和数据生成工具包的集成研究套件，将Crafter确立为基于基础模型的具身智能研究测试平台，提供了CrafterVPT、CrafterCLIP和CrafterSteve-1。为解决Crafter的数据稀缺问题，CrafterDojo包含可自动生成多样化合成演示数据和对齐视频-语言对的工具，由此构建了CrafterPlay和CrafterCaption数据集。实验表明，这些数据集能够支持有效的基础模型训练，并使分层智能体在任务表现上超越非分层基线。

    arXiv:2508.13530v2 Announce Type: replace  Abstract: Foundation models are central to building general-purpose embodied agents. Minecraft has become the dominant testbed for this paradigm, but its computational and engineering overhead limits rapid iteration and controlled analysis. Crafter offers a lightweight alternative yet lacks the foundation model ecosystem required for such research. We introduce CrafterDojo, a suite of foundation models, datasets, and data-generation toolkits that establishes Crafter as a testbed for foundation-model-based embodied intelligence, providing CrafterVPT, CrafterCLIP, and CrafterSteve-1. To address Crafter's data scarcity, CrafterDojo includes automatic tools that generate diverse synthetic demonstrations and aligned video-language pairs, yielding the CrafterPlay and CrafterCaption datasets. Experiments show that these datasets support effective foundation model training and enable hierarchical agents that outperform non-hierarchical baselines on lo
    
[^435]: 面向医学图像到图像转换的样本感知测试时自适应

    Sample-Aware Test-Time Adaptation for Medical Image-to-Image Translation

    [https://arxiv.org/abs/2508.00766](https://arxiv.org/abs/2508.00766)

    提出了一种样本感知的测试时自适应框架，通过重构模块量化域偏移，并利用动态自适应块选择性调整预训练模型的内部特征，在低剂量CT去噪和MRI跨模态转换任务中有效处理分布外样本，同时不损害分布内样本的性能。

    

    图像到图像转换已成为医学成像中的一项强大技术，能够实现图像去噪和跨模态转换等任务。然而，该方法在处理分布外样本时存在局限性，容易导致性能下降。为了解决这一局限性，我们提出了一种新颖的测试时自适应框架，该框架能够根据每个测试样本的特性动态调整转换过程。我们的方法引入了一个重构模块来量化域偏移，以及一个动态自适应块，它选择性地修改预训练转换模型的内部特征以减轻偏移，同时不会损害无需自适应的分布内样本的性能。我们在两个医学图像到图像转换任务上评估了我们的方法：低剂量CT去噪和T1到T2的MRI转换，结果显示相比基线方法取得了一致的改进。

    arXiv:2508.00766v3 Announce Type: replace-cross  Abstract: Image-to-image translation has emerged as a powerful technique in medical imaging, enabling tasks such as image denoising and cross-modality conversion. However, it suffers from limitations in handling out-of-distribution samples without causing performance degradation. To address this limitation, we propose a novel Test-Time Adaptation (TTA) framework that dynamically adjusts the translation process based on the characteristics of each test sample. Our method introduces a Reconstruction Module to quantify the domain shift and a Dynamic Adaptation Block that selectively modifies the internal features of a pretrained translation model to mitigate the shift without compromising the performance on in-distribution samples that do not require adaptation. We evaluate our approach on two medical image-to-image translation tasks: low-dose CT denoising and T1 to T2 MRI translation, showing consistent improvements over both the baseline 
    
[^436]: AIR：重新思考多模态对比表征空间中的图文偏移对齐

    AIR: Rethinking Image-Text Offset Alignment in Multimodal Contrastive Representation Space

    [https://arxiv.org/abs/2506.10895](https://arxiv.org/abs/2506.10895)

    本文系统性地实证检验了 CLIP、SigLIP 等多模态对比表征空间中文本驱动的视觉变换方法所依赖的“图文偏移对齐”假设，质疑其有效性。

    

    图文偏移对齐（image-text offset alignment）是指视觉变换方向与其对应文本变换方向之间的一致性，它已成为 CLIP 和 SigLIP 等多模态对比表征空间中的一个重要关系属性。该属性在文本驱动的视觉变换任务中发挥着核心作用，包括生成模型的领域自适应和文本引导的对象中心图像编辑等，在这些任务中，文本偏移被用于引导视觉变换。这些方法从根本上依赖于一个假设，即图像偏移与文本偏移是良好对齐的，从而使文本变换方向能够为相应的视觉变换提供可靠的指导。然而，这一对齐假设的有效性尚未得到系统性检验。在本工作中，我们通过对图文偏移对齐进行全面的实证分析，对这一基础性假设提出了质疑。

    arXiv:2506.10895v2 Announce Type: replace-cross  Abstract: Image-text offset alignment, which refers to the consistency between visual transformation directions and their corresponding textual transformation directions, has become an important relational property in multimodal contrastive representation spaces such as CLIP and SigLIP. This property plays a central role in text-driven visual transformation tasks, including generative model domain adaptation and text-guided object-centric image editing, where textual offsets are used to guide visual transformations. These methods fundamentally rely on the assumption that image and text offsets are well aligned, such that textual transformation directions can provide reliable guidance for corresponding visual transformations. However, the validity of this alignment assumption has not been systematically examined. In this work, we question this foundational assumption by conducting a comprehensive empirical analysis of image-text offset al
    
[^437]: 用于高效多目标强化学习的局部策略流形

    Local Policy Manifolds for Efficient Multi-Objective Reinforcement Learning

    [https://arxiv.org/abs/2506.04022](https://arxiv.org/abs/2506.04022)

    提出LLE-MORL方法，通过利用策略参数空间与性能空间之间的局部关系追踪策略流形，借助局部线性外推实现高效搜索，无需大量重新训练即可快速生成覆盖多目标权衡的高质量策略集合。

    

    多目标强化学习（MORL）旨在优化多个经常相互冲突的目标，以提高强化学习在实际任务中的灵活性和可靠性。这通常通过寻找一组多样化的、非支配的策略来实现，这些策略在性能空间中构成帕累托前沿。然而，构建这样的策略集在计算上代价高昂，因为它需要找到的不是一个单一的最优策略，而是一组能够覆盖目标之间广泛权衡范围的策略集合。我们提出了LLE-MORL，一种通过利用高维策略参数空间与性能空间之间的局部关系来追踪策略流形的方法。这种结构化表示使得算法能够通过局部线性外推在连续解域内进行高效搜索，从而无需大量重新训练即可快速生成高质量解。我们还对该方法提供了理论分析。

    arXiv:2506.04022v3 Announce Type: replace  Abstract: Multi-objective reinforcement learning (MORL) aims at optimising several, often conflicting goals to improve the flexibility and reliability of RL in practical tasks. This is typically achieved by finding a set of diverse, non-dominated policies that form a Pareto front in the performance space. However, constructing such a policy set remains computationally demanding, as it requires finding not a single optimal policy, but a set of policies to cover a wide range of trade-offs among the objectives. We introduce LLE-MORL, an approach that traces policy manifolds by utilising the local relationship between the high-dimensional policy parameter space and the performance space. This structured representation enables an efficient search within contiguous solution domains via locally linear extrapolation, allowing for the rapid generation of high-quality solutions without extensive retraining. We also provide a theoretical analysis of the 
    
[^438]: Fortune：面向大语言模型复杂表格推理的公式驱动强化学习

    Fortune: Formula-Driven Reinforcement Learning for LLM Reasoning over Complex Tables

    [https://arxiv.org/abs/2505.23667](https://arxiv.org/abs/2505.23667)

    提出Fortune，一种公式驱动的强化学习框架，以执行成功与答案正确性作为奖励，训练大语言模型生成可执行的电子表格公式，从而提升其在复杂表格上的数值推理能力。

    

    表格是组织和分析数据的基本媒介，因此表格推理是智能系统的一项核心能力。尽管大语言模型（LLM）展现出强大的通用推理能力，但在对表格数据进行精确数值推理时仍存在困难，尤其是在需要超越简单关系查询的复杂表格场景中。电子表格公式为可执行的符号运算提供了强大且富有表现力的接口，能够支持复杂的推理模式，而这在很大程度上尚未被现有LLM充分挖掘。本文提出了公式调优，这是一个公式驱动的强化学习（RL）框架，通过以执行成功和答案正确性作为奖励信号，训练LLM生成可执行的电子表格公式来回答表格数据相关的问题。基于Fortune，我们训练了Formula-R1，并在七个表格推理（基准）上评估其性能。

    arXiv:2505.23667v4 Announce Type: replace  Abstract: Tables are a fundamental medium for organizing and analyzing data, making table reasoning an essential capability for intelligent systems. Although large language models (LLMs) exhibit strong general reasoning abilities, they still struggle with accurate numerical reasoning over tabular data, particularly in complex table settings that require more than simple relational lookup. Spreadsheet formulas provide a powerful and expressive interface for executable symbolic operations, enabling sophisticated reasoning patterns that remain largely underexplored by existing LLMs. In this paper, we introduce Formula Tuning (Fortune), a formula-driven reinforcement learning (RL) framework that trains LLMs to generate executable spreadsheet formulas for question answering over tabular data, using execution success and answer correctness as reward signals. Building upon Fortune, we train Formula-R1 and evaluate its performance across seven table r
    
[^439]: TRIM：通过目标行级迭代度量驱动剪枝实现极端稀疏化

    TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning

    [https://arxiv.org/abs/2505.16743](https://arxiv.org/abs/2505.16743)

    TRIM提出一种行级迭代度量驱动剪枝方法，通过为层内每个输出行分配不同的稀疏比率，在高稀疏度（70-80%）下显著降低大语言模型的困惑度，例如使Qwen2.5-14B在80%稀疏度下的困惑度近乎减半。

    

    大型语言模型（LLMs）带来了巨大的计算和内存挑战。现有的一次性剪枝方法通常对层内的所有行使用统一的稀疏度目标，这在高稀疏度下可能过于粗糙。本工作引入了TRIM（目标行级迭代度量驱动剪枝），这是一种基于前向统计信息的封装方法，能够在保持模型权重冻结的同时，为各个输出行分配不同的稀疏比率。TRIM将“剪枝哪些权重”（评分）与“每行剪枝多少”（分配）这两个问题分开处理，并可与现有的评分器和层级分配方法组合使用。在Qwen2.5、LLaMA-2和OPT模型上的实验表明，其最大收益出现在70-80%稀疏度区间。在80%稀疏度下，将TRIM添加到OWL上，可使Qwen2.5-14B的WikiText-2困惑度从348.48降至180.67，使OPT-13B的困惑度从6461.43降至324.14。我们得出结论，细粒度的行级分配是显著提升模型性能的有效机制。

    arXiv:2505.16743v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) present substantial computational and memory challenges. Existing one-shot pruning methods often use a uniform sparsity target for all rows within a layer, which can be too coarse at high sparsity. This work introduces TRIM (Targeted Row-wise Iterative Metric-driven pruning), a forward-statistics-based wrapper that allocates different sparsity ratios to individual output rows while keeping model weights frozen. TRIM separates the choice of which weights to prune (scoring) from how many to prune in each row (allocation), and composes with existing scorers and layer-wise allocations. Across Qwen2.5, LLaMA-2, and OPT models, its largest gains occur at 70-80% sparsity. At 80%, adding TRIM to OWL reduces WikiText-2 perplexity from 348.48 to 180.67 for Qwen2.5-14B and from 6461.43 to 324.14 for OPT-13B. We conclude that fine-grained, row-wise allocation is an effective mechanism for substantially improvin
    
[^440]: VFRTok：基于时长成比例信息假设的可变帧率视频分词器

    VFRTok: Variable Frame Rates Video Tokenizer with Duration-Proportional Information Assumption

    [https://arxiv.org/abs/2505.12053](https://arxiv.org/abs/2505.12053)

    本文提出基于“视频信息容量与时长而非帧数成正比”这一新假设的VFRTok可变帧率视频分词器，通过非对称帧率训练和部分旋转位置编码，显著降低了视频生成扩散模型的计算成本。

    

    基于潜在扩散模型的现代视频生成框架，由于“帧数成比例信息假设”而在视频分词方面存在效率低下的问题。现有的分词器提供固定的时间压缩率，导致扩散模型的计算成本随帧率线性增长。本文提出了“时长成比例信息假设”：即视频信息容量的上限与视频时长成正比，而非与帧数成正比。基于这一洞察，本文提出了VFRTok——一种基于Transformer的视频分词器，通过编码器与解码器之间的非对称帧率训练，实现了可变帧率的编码与解码。此外，本文还提出了部分旋转位置编码，以解耦位置建模与内容建模，将相关的图像块分组为统一的token，有效提升了内容建模的效果。

    arXiv:2505.12053v3 Announce Type: replace-cross  Abstract: Modern video generation frameworks based on Latent Diffusion Models suffer from inefficiencies in tokenization due to the Frame-Proportional Information Assumption. Existing tokenizers provide fixed temporal compression rates, causing the computational cost of the diffusion model to scale linearly with the frame rate. The paper proposes the Duration-Proportional Information Assumption: the upper bound on the information capacity of a video is proportional to the duration rather than the number of frames. Based on this insight, the paper introduces VFRTok, a Transformer-based video tokenizer, that enables variable frame rate encoding and decoding through asymmetric frame rate training between the encoder and decoder. Furthermore, the paper proposes Partial Rotary Position Embeddings (RoPE) to decouple position and content modeling, which groups correlated patches into unified tokens. The Partial RoPE effectively improves content
    
[^441]: 因果片段：逐片分析和改进脉冲神经网络

    Causal pieces: analysing and improving spiking neural networks piece by piece

    [https://arxiv.org/abs/2504.14015](https://arxiv.org/abs/2504.14015)

    提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。

    

    我们提出了“因果片段”，这是一个用于分析脉冲神经网络（SNN）的新概念，其灵感来源于人工神经网络（ANN）中用于研究表达能力和可训练性的“线性片段”。因果片段将具有单脉冲编码的前馈SNN的输入和参数空间划分为不同的区域，在每个区域中，相同的子网络导致输出脉冲的产生。对于由具有大膜时间常数的基于电流的泄漏积分发放（LIF）神经元构成的网络，我们证明了在每个因果片段内，输出脉冲时间相对于输入和网络参数是局部Lipschitz连续的。我们进一步证明了依赖于因果片段数量的逼近误差下界。因此，因果片段的数量是衡量SNN逼近能力的一个度量指标，该指标在脉冲时间不连续的情况下仍然有效，并且适用于同时具有兴奋性和抑制性突触的网络。

    arXiv:2504.14015v2 Announce Type: replace-cross  Abstract: We introduce "causal pieces", a novel concept for analysing spiking neural networks (SNNs), inspired by "linear pieces" used to study expressivity and trainability in artificial neural networks (ANNs). Causal pieces partition the input and parameter space of a feedforward SNN with single-spike coding into distinct regions where the same subnetwork causes the output spikes. For networks of current-based leaky integrate-and-fire (LIF) neurons with large membrane time constants, we show that within each causal piece, output spike times are locally Lipschitz continuous with respect to inputs and network parameters. We further prove a lower bound on the approximation error that depends on the number of causal pieces. Thus, the number of causal pieces is a measure of the approximation capabilities of SNNs, which is valid despite spike-time discontinuities and applies to networks with both excitatory and inhibitory synapses. Empirical
    
[^442]: Mamba-3D作为掩码自编码器用于医学超声视频的精确且数据高效的分析

    Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos

    [https://arxiv.org/abs/2503.20258](https://arxiv.org/abs/2503.20258)

    提出E-ViM³——一种保留视频3D结构的Vision Mamba网络，结合包围全局标记和时空链式掩码策略进行自监督预训练，实现了医学超声视频的精确且数据高效的分析。

    

    超声视频是临床医学影像数据的重要形式，基于深度学习的自动化分析可以提高诊断准确性和临床效率。然而，标注数据的稀缺性以及视频分析固有的挑战阻碍了相关方法的发展。在这项工作中，我们提出了E-ViM³，这是一种数据高效的Vision Mamba网络，它保留了视频数据的3D结构，增强了长程依赖和归纳偏置，以更好地建模时空相关性。通过我们设计的包围全局标记（EGT），该模型比竞争方法更有效地捕获和聚合全局特征。为了进一步提高数据效率，我们采用掩码视频建模进行自监督预训练，并提出了时空链式（STC）掩码策略，旨在适应各种视频场景。实验表明，E-ViM³达到了最先进的性能……

    arXiv:2503.20258v2 Announce Type: replace-cross  Abstract: Ultrasound videos are an important form of clinical imaging data, and deep learning-based automated analysis can improve diagnostic accuracy and clinical efficiency. However, the scarcity of labeled data and the inherent challenges of video analysis have impeded the advancement of related methods. In this work, we introduce E-ViM$^3$, a data-efficient Vision Mamba network that preserves the 3D structure of video data, enhancing long-range dependencies and inductive biases to better model space-time correlations. With our design of Enclosure Global Tokens (EGT), the model captures and aggregates global features more effectively than competing methods. To further improve data efficiency, we employ masked video modeling for self-supervised pre-training, with the proposed Spatial-Temporal Chained (STC) masking strategy designed to adapt to various video scenarios. Experiments demonstrate that E-ViM$^3$ performs as the state-of-the-
    
[^443]: 论结构化剪枝在小语言模型预训练中的效率

    On the Efficiency of Structured Pruning in Small Language Model Pretraining

    [https://arxiv.org/abs/2503.05920](https://arxiv.org/abs/2503.05920)

    提出IDEA Prune集成式放大-剪枝流水线，在单一余弦退火学习率调度下统一放大模型预训练、迭代结构化剪枝和恢复过程，在全成本核算下比传统分阶段剪枝方法更高效地训练出小语言模型。

    

    在边缘设备上部署生成式语言模型的公司，出于法律、数据治理或产品定制化等要求，通常需要从头训练小模型。另一种方法是先训练一个放大的模型，再通过结构化剪枝将其压缩到目标尺寸，但一旦将放大模型的预训练成本计算在内，这种做法的优势就变得不明确了。我们引入了“放大-剪枝”框架，在全成本核算的条件下研究这一问题，并表明传统的放大-剪枝流水线——即分阶段地训练放大模型、剪枝并恢复——在相同的总token预算下并不总是优于直接训练小模型。为克服这一局限性，我们提出了IDEA Prune，这是一种集成式的放大-剪枝流水线，它在单一的余弦退火学习率调度下，将放大模型预训练、迭代结构化剪枝和恢复统一为一体。

    arXiv:2503.05920v2 Announce Type: replace-cross  Abstract: Companies that deploy generative language models on edge devices often need to train small models from scratch due to legal, data-governance, or product-customization requirements. An alternative approach is to train an enlarged model and compress it to the target size via structured pruning, but this advantage becomes unclear once the cost of pretraining the enlarged model is included. We introduce the enlarge-and-prune framework to study this question under full-cost accounting, and show that conventional enlarge-and-prune pipelines -- which train the enlarged model, prune it, and recover in separate stages -- do not consistently outperform direct small-model training under the same total token budget. To overcome this limitation, we propose IDEA Prune, an integrated enlarge-and-prune pipeline that unifies enlarged model pretraining, iterative structured pruning, and recovery under a single cosine annealing learning rate sche
    
[^444]: MMGenBench：一种从文本到图像生成视角评估大型多模态模型的自动化流水线

    MMGenBench: An Automated Pipeline for Evaluating LMMs from a Text-to-Image Generation Perspective

    [https://arxiv.org/abs/2411.14062](https://arxiv.org/abs/2411.14062)

    MMGenBench是一个无需预定义问题或参考描述的自动化基准框架，通过“图像描述生成→文生图重建→编码器比对”的流水线从原始图像评估大型多模态模型，对68个LMM的评估揭示了总分可能掩盖的模式特定重建质量差异。

    

    现有基准测试通常依赖预定义的问题或参考描述，这限制了它们在新图像集上评估详细描述的灵活性。我们提出了MMGenBench，这是一个自动化基准测试框架，可直接从原始图像评估大型多模态模型（LMMs），无需逐图像的问题或参考描述。该流水线先生成图像描述，再使用冻结的文本到图像模型重建图像，并通过冻结的图像编码器比较原始图像与重建图像。它能够产出可复用的逐图像评分和可检查的重建结果，既支持模型比较，也支持对可重建视觉信息的诊断。我们构建了包含10,000张生成图像的MMGenBench-Domain，以及包含1,284张图像、覆盖13种重叠视觉模式的MMGenBench-Test。对68个LMM的评估揭示了重建质量上存在总体分数可能掩盖的模式特定差异。在1,000张人工标注的……（摘要在此处被截断）

    arXiv:2411.14062v3 Announce Type: replace-cross  Abstract: Existing benchmarks often rely on predefined questions or reference captions, limiting their flexibility for evaluating detailed descriptions on new image collections. We present MMGenBench, an automated benchmark framework for evaluating Large Multimodal Models (LMMs) from raw images without per-image questions or reference captions. The pipeline generates descriptions, reconstructs images with a frozen text-to-image model, and compares the original and reconstructed images using a frozen image encoder. It produces reusable per-image scores and inspectable reconstructions, supporting both model comparison and diagnosis of reconstructible visual information. We construct MMGenBench-Domain with 10,000 generated images and MMGenBench-Test with 1,284 images spanning 13 overlapping visual patterns. Evaluation of 68 LMMs reveals pattern-specific differences in reconstruction quality that overall scores can obscure. On 1,000 human-la
    
[^445]: 人工神经网络在无维数灾难条件下逼近某类光滑有界函数时网络深度的必要性

    The necessity of depth for artificial neural networks to approximate certain classes of smooth and bounded functions without the curse of dimensionality

    [https://arxiv.org/abs/2301.08284](https://arxiv.org/abs/2301.08284)

    该论文证明了一类乘积函数及其正弦复合函数无法被浅层或深度不足的ReLU神经网络在避免维数灾难的前提下逼近，但可被足够深的ReLU神经网络避免维数灾难地逼近，从而首次严格揭示了网络深度对于此类光滑有界函数高维逼近的必要性。

    

    在本文中，我们研究了采用修正线性单元（ReLU）激活函数的浅层与深层人工神经网络（ANN）的高维逼近能力。特别地，本工作的一个关键贡献在于揭示了：对于所有满足 $\max\{\lvert a\rvert,\lvert b\rvert\}>1$ 的 $a\in\mathbb{R}$ 和 $b\in(a,\infty)$，函数 $[a,b]^d\ni x=(x_1,\ldots,x_d)\mapsto\prod_{i=1}^d x_i\in\mathbb{R}$（$d\in\mathbb{N}=\{1,2,3,\ldots\}$）以及函数 $[a,b]^d\ni x=(x_1,\ldots,x_d)\mapsto\sin(\prod_{i=1}^d x_i)\in\mathbb{R}$（$d\in\mathbb{N}$）既无法被浅层ReLU神经网络以避免维数灾难的方式逼近，也无法被深度不足的ReLU神经网络以避免维数灾难的方式逼近，但可以被足够深的ReLU神经网络以避免维数灾难的方式逼近。更具体地，我们证明了在浅层ReLU神经网络或具有固定数量隐藏层的深层ReLU神经网络的情形下……

    arXiv:2301.08284v2 Announce Type: replace-cross  Abstract: In this article we study high-dimensional approximation capacities of shallow and deep artificial neural networks (ANNs) with the rectified linear unit (ReLU) activation. In particular, it is a key contribution of this work to reveal that for all $a\in\mathbb{R}$, $b\in(a,\infty)$ with $\max\{\lvert a\rvert,\lvert b\rvert\}>1$ we have that the functions $[a,b]^d\ni x=(x_1,\ldots,x_d)\mapsto\prod_{i=1}^d x_i\in\mathbb{R}$ for $d\in\mathbb{N}=\{1,2,3,\ldots\}$ as well as the functions $[a,b]^d\ni x=(x_1,\ldots,x_d)\mapsto\sin(\prod_{i=1}^d x_i)\in\mathbb{R}$ for $d\in\mathbb{N}$ can neither be approximated without the curse of dimensionality by means of shallow ANNs nor insufficiently deep ANNs with ReLU activation but can be approximated without the curse of dimensionality by sufficiently deep ANNs with ReLU activation. More specifically, we prove that in the case of shallow ReLU ANNs or deep ReLU ANNs with a fixed number of hid
    
[^446]: HindSight：一种用于表示部分-整体层次结构的基于图的视觉模型架构

    HindSight: A Graph-Based Vision Model Architecture For Representing Part-Whole Hierarchies

    [https://arxiv.org/abs/2104.03722](https://arxiv.org/abs/2104.03722)

    该论文提出了HindSight架构，通过将图像划分为不同层级的图像块并构建全连接图来编码图像中的部分-整体层次结构，可训练为通用视觉编码器以支持多种视觉下游任务。

    

    本文提出了一种以图的形式编码图像中部分-整体层次结构表示的模型架构。其核心思想是将图像划分为不同层级的图像块，然后将所有这些图像块作为全连接图的节点。在每次图迭代中，使用动态特征提取模块从这些图像块中提取特征表示。这使我们能够学习到包含图像固有部分-整体层次信息的丰富图表示。利用适当的自监督训练技术，这样的模型可以被训练成一个通用视觉编码器模型，进而用于各种视觉相关的下游任务（例如图像分类、目标检测、图像描述生成等）。

    arXiv:2104.03722v2 Announce Type: replace-cross  Abstract: This paper presents a model architecture for encoding the representations of part-whole hierarchies in images in form of a graph. The idea is to divide the image into patches of different levels and then treat all of these patches as nodes for a fully connected graph. A dynamic feature extraction module is used to extract feature representations from these patches in each graph iteration. This enables us to learn a rich graph representation of the image that encompasses the inherent part-whole hierarchical information. Utilizing proper self-supervised training techniques, such a model can be trained as a general purpose vision encoder model which can then be used for various vision related downstream tasks (e.g., Image Classification, Object Detection, Image Captioning, etc.).
    
[^447]: 随机前向模式自动微分优化算法

    Randomized Forward Mode of Automatic Differentiation for Optimization Algorithms. (arXiv:2310.14168v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2310.14168](http://arxiv.org/abs/2310.14168)

    该论文介绍了一种随机前向模式自动微分优化算法，通过在神经网络的正向传递中计算损失函数的方向导数来更新参数。算法通过采样不同概率分布的随机方向，使用正向模式自动微分计算雅可比向量乘积，并提供了对其收敛速度和计算复杂性的严格分析。

    

    神经网络中的反向传播利用了自动微分的基本要素，即反向模式微分，或称为向量雅可比乘积(VJP)，或在微分几何的背景下被称为拉回过程。梯度的计算对于使用梯度下降方法更新神经网络参数非常重要。在本研究中，我们提出了一种通用的随机方法，通过使用通过正向模式AD或雅可比向量乘积(JVP)高效计算的损失函数的方向导数来更新神经网络的参数。这些JVP沿着从不同概率分布（例如伯努利、正态、维格纳、拉普拉斯和均匀分布）采样的随机方向计算。梯度的计算在神经网络的正向传递过程中进行。我们还提供了对所提出方法的严格分析，包括收敛速度以及计算复杂性。

    Backpropagation within neural networks leverages a fundamental element of automatic differentiation, which is referred to as the reverse mode differentiation, or vector Jacobian Product (VJP) or, in the context of differential geometry, known as the pull-back process. The computation of gradient is important as update of neural network parameters is performed using gradient descent method. In this study, we present a genric randomized method, which updates the parameters of neural networks by using directional derivatives of loss functions computed efficiently by using forward mode AD or Jacobian vector Product (JVP). These JVP are computed along the random directions sampled from different probability distributions e.g., Bernoulli, Normal, Wigner, Laplace and Uniform distributions. The computation of gradient is performed during the forward pass of the neural network. We also present a rigorous analysis of the presented methods providing the rate of convergence along with the computat
    

