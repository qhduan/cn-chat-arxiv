# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking the design space of Vanilla Pixel Diffusion Transformers](https://arxiv.org/abs/2610.23854) | 提出MFlow自适应参数化方法，可在x₀-预测和u-预测之间沿数据主方向进行插值学习，并发现最优参数化随数据表示和模型容量系统性变化，模型容量越大越偏向速度预测。 |
| [^2] | [Behavioral Couplings: From Formation to Behavioral Emergence in Language Models](https://arxiv.org/abs/2610.23847) | 该论文提出“行为耦合”概念，发现当模型在同时描述两种行为的文档上训练后，针对其中一种行为进行微调会导致另一种行为也随之涌现，揭示了微调可引发超出预期目标之外的行为。 |
| [^3] | [AI-Driven Reproducibility and Analytical Robustness in Scientific Discovery](https://arxiv.org/abs/2610.23845) | 该研究推出SEER-Repro基准并开发AI智能体，成功复现了77.4%的基于SEER数据库的医学研究声明，同时通过智能体分析流水线自动化测试医学发现的分析稳健性。 |
| [^4] | [Self-Building Labs](https://arxiv.org/abs/2610.23826) | 该论文提出“自构建实验室”（SBL），能根据高层科学目标自主设计并搭建实验室的布局、数字孪生、机器人技能与实验流程，突破了现有自动驾驶实验室灵活性不足这一制约科学发现规模化的瓶颈。 |
| [^5] | [Beyond SAEs: Consistent Concept Discovery with Sparse Coding](https://arxiv.org/abs/2610.23801) | 提出ConSDL方法，用逐样本稀疏编码取代SAE的标准线性编码器，实现了跨训练运行的字典与系数高度一致性，显著提升了稀疏字典学习在概念发现中的可靠性与可复现性。 |
| [^6] | [FuTransformer: FP8 prefill and INT4 decode for Llama-style inference on Hopper](https://arxiv.org/abs/2610.23796) | 该论文提出FUTRANSFORMER，一个面向NVIDIA Hopper GPU的Llama风格推理CUDA实现，融合FP8预填充、INT4解码权重与KV缓存、直接KV缓存放置及块对角Hadamard变换等优化，在单张H100上预填充速度比编译的PyTorch FP8快达1.48倍、达到1.29 PFLOP/s。 |
| [^7] | [DrugHarness: An Evolvable Agent Harness for Computational Drug Discovery](https://arxiv.org/abs/2610.23791) | DrugHarness是一个可演化的智能体框架，它将80多种科学操作组合执行，并从执行轨迹中归纳、审查和持续修订可复用的条件性技能，从而适应计算药物发现中不断变化的证据与约束。 |
| [^8] | [Beyond Large Language Models: A Mental World Model Framework for Mental Modeling, Simulation, Evolution, and Interpretability](https://arxiv.org/abs/2610.23722) | 本文提出心理世界模型（MWM）及多分支心理演化架构（MMEA），通过显式构建主观世界、建模认知与心理状态交互并模拟受历史经验驱动的心理演化，在ToM基准测试中以96.48%和94.00%的准确率显著超越现有大语言模型和专门方法。 |
| [^9] | [Neural Expansion: A Unified Mechanism for How Deep Neural Networks Generalize](https://arxiv.org/abs/2610.23716) | 本文提出“神经扩张”这一统一几何框架，通过泛化区间量化训练样本周围预测稳定的方向区域，发现训练过程中模型射线覆盖沿雅可比奇异值较低的方向渐进扩张，从而揭示了深度神经网络泛化能力的形成机制。 |
| [^10] | [GLOBE: Evolving Structured Agentic Memory for E-commerce Product Optimization in Generative Search](https://arxiv.org/abs/2610.23709) | GLOBE通过将电商产品优化知识不断积累并演化为结构化的按字段划分行为手册作为智能体记忆，使语言智能体能够在生成式搜索场景下实现持续的自我改进。 |
| [^11] | [On the possibilities and limitations of LLM-based multi-agent systems for multi-criteria decision-making in healthcare operations management](https://arxiv.org/abs/2610.23673) | 本研究提出了一个评估LLM生成偏好作为多目标优化输入可靠性的框架，发现显式偏好能响应诊所运营条件并在不同利益相关者角色间存在差异，但智能体之间的集体审议可能抹平这些差异。 |
| [^12] | [OwlSight: A Panoramic Monitor for Chain-of-Thought Reasoning](https://arxiv.org/abs/2610.23670) | 提出了基于“因子”的思维链监测策略，通过在提示词中自定义监测需求，使单一监测器能够覆盖能力、安全性和可信度等多个维度的不同任务。 |
| [^13] | [TUPLE-Bench: Evaluating and Mitigating Privacy-Utility Trade-offs in Typed Decision Models](https://arxiv.org/abs/2610.23668) | 该论文提出TUPLE-Bench基准，系统量化了类型化决策模型中概率与置信度等额外输出带来的隐私泄露与决策效用之间的权衡，发现仅靠指令要求隐藏敏感成因几乎无效（泄露率达91.88%-99.93%），而完整输出虽在所有对比中降低下游成本，却使细粒度敏感成因的恢复率从约50%升至62.85%-75.35%。 |
| [^14] | [PresenceBench: An Exploratory Evaluation of Large Language Model Adaptation to User State](https://arxiv.org/abs/2610.23625) | 该论文提出PresenceBench基准，通过五臂因果设计与以正确性为门槛的人类配对比较，评估大语言模型能否利用用户功能状态（准备度、心流、稳定性）提升响应质量，且这种提升是否特异地依赖于正确的状态而非额外的结构化上下文。 |
| [^15] | [GEMS: Molecular Structure Elucidation as Learned Search over Isomer Graphs](https://arxiv.org/abs/2610.23617) | 提出GEMS方法，通过学习保持分子式和原子化合价不变的2-switch键编辑操作来精炼候选分子结构，并借助预训练谱图-指纹模型MIST与神经策略在单次前向传播中评分所有有效编辑，从而解决从分子式和串联质谱解析小分子结构的难题。 |
| [^16] | [Beyond Action Prediction: Learning Search-Free Planning in Latent Space](https://arxiv.org/abs/2610.23615) | 提出HIJP框架，用目标条件逆模型直接从当前与目标观测预测动作，通过将轨迹未来观测重标记为多时间尺度的目标并保留梯度，实现无需测试时搜索的潜在空间规划。 |
| [^17] | [SIT-Eval: Situation-Based Interactive Turn-by-Turn Evaluation of Conversational Agents](https://arxiv.org/abs/2610.23612) | SIT-Eval提出了一个可调节的评估框架，将对话代理的期望行为形式化为“情境-规则”形式，并由LLM对话者在每轮对话中扮演随机采样的情境，从而实现对对话代理在多样化及罕见情境下的逐轮交互式评估，弥补了现有静态基准对罕见情境覆盖不足的安全缺口。 |
| [^18] | [From Trajectories to Differentiated Skills: Cognitive Differentiation for Self-Improving Agents](https://arxiv.org/abs/2610.23599) | 提出CDSE框架，将智能体技能认知分化为Meta、Experience、Tool和Script四个功能组件，从成功与失败轨迹中提取语义梯度并按功能角色进行针对性更新、验证与协同调用，从而提升自我改进智能体的技能演化能力。 |
| [^19] | [RL-Guided Temporal Localization for Dual-Channel Retrieval in Long-Horizon Agent Memory](https://arxiv.org/abs/2610.23598) | 该论文提出TIDER框架，通过强化学习训练的时间区间定位器在检索阶段引入时间约束，结合双通道证据检索，解决长时程LLM智能体记忆中目标记忆被时间不匹配但语义相似的候选记忆挤占的问题。 |
| [^20] | [Darwin-Claw: Environment-Adaptive Experience Consolidation for Recursively Self-Evolving Agents](https://arxiv.org/abs/2610.23574) | 提出Darwin-Claw，一个环境自适应的自进化框架，通过系统性地提炼情景经验，并基于“智能体-执行者”与“进化器-批评者”的迭代协同适应机制，驱动LLM智能体操作技能与持久记忆的全面进化。 |
| [^21] | [SkillBrowser: Selective Skill Exposure for Scalable Skill-Using Agents](https://arxiv.org/abs/2610.23554) | SkillBrowser提出了一种由元技能管理的选择性技能暴露框架，使智能体能够自主识别当前能力需求并按需加载技能，在六个SRA-Bench子集和五个大语言模型上的总体准确率均优于静态检索与基于LLM的选择方法。 |
| [^22] | [Reason-guided Human Reward Modeling for Preference-based Reinforcement Learning](https://arxiv.org/abs/2610.23550) | 本文提出一种推理引导的偏好强化学习框架，通过在偏好标签之外融入解释偏好理由的理由标签来学习人类奖励模型，使奖励同时捕捉人类选择及其背后的原因，从而提升策略与人类偏好的对齐程度和可解释性。 |
| [^23] | [From Billion-Scale to Million-Scale Language Models: Operational Capabilities for Embedded Post-Exploitation](https://arxiv.org/abs/2610.23534) | 该研究首次系统评估了七个开源小型语言模型（2000万-15亿参数）在后渗透恶意软件代码生成任务上的表现，发现攻击性代码生成能力并不随模型规模线性增长，而是取决于模型容量等多因素的交互作用。 |
| [^24] | [Truth in Contrast: Calibrated Factual Evidence from Language Model Likelihoods](https://arxiv.org/abs/2610.23527) | 该论文提出对比事实证据（CFE）及其概率形式对比事实后验（CFP），通过比较同一声明在事实与非事实条件下的似然差异，使共享误差相互抵消，从而在多语言事实陈述上显著提升置信度校准性能并保持较强的事实判别能力。 |
| [^25] | [Trapped in Your Story: Evaluating Narrative-Induced Sycophancy in Large Language Models](https://arxiv.org/abs/2610.23525) | 该论文构建了一个包含500个问题、5,500个评估实例的受控基准，首次系统揭示大语言模型的判断会被用户的叙事性论证（如个人经历、因果解释）所左右，且这种“叙事诱发谄媚”效应并非随叙事精细程度单调增强。 |
| [^26] | [LocalIntent: Outcome-Specific Structured Generation for Backend-Compatible Smart Building Automation with Local LLMs](https://arxiv.org/abs/2610.23511) | LocalIntent 提出了一种面向特定结果的提示与结构化生成方法，将请求路由至普通命令分支或触发-动作规则分支并辅以确定性验证，使 2B–9B 参数量的本地小模型在智慧楼宇自动化任务中超越通用提示方案，实现后端兼容的可靠结构化输出。 |
| [^27] | [A Topological Signature of Learning under Noisy Labels](https://arxiv.org/abs/2610.23491) | 该论文利用持续同调追踪神经激活中的环状结构，发现噪声标签训练的模型在训练中后期比干净标签模型表现出更高的拓扑持续性，即使其预测与噪声标签几乎不一致，从而为噪声监督下的学习过程提供了一种基于表示几何的拓扑特征。 |
| [^28] | [QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity](https://arxiv.org/abs/2610.23470) | 提出QD-Rec方法，将质量-多样性算法、上下文老虎机与基础模型相结合，为高风险决策生成多样化、高质量且附带自然语言理由的开放式推荐。 |
| [^29] | [S-GAP: Stealthy Graph Attack Primitives for Graph Large Language Models](https://arxiv.org/abs/2610.23465) | 提出图攻击原语（GAP）这一原子化语义表示，将文本扰动与结构扰动统一在连贯的攻击意图下，解决了现有针对图大语言模型的对抗攻击中攻击意图碎片化、隐蔽性差的问题。 |
| [^30] | [GatewayBench: Unified Capability and Security Evaluation Platform for MCP Gateways](https://arxiv.org/abs/2610.23449) | 提出了首个针对MCP网关的统一评估框架与基准GatewayBench，通过覆盖六个领域的524个任务和来自251个真实MCP服务器的11,118个工具，系统评估扁平化与层级化两类网关架构对智能体能力、效率和安全性的影响。 |
| [^31] | [RECREG: Recovery-based Shielding for Regular Safety Properties in Reinforcement Learning](https://arxiv.org/abs/2610.23445) | RecReg是一个通过基于恢复的屏蔽机制强制执行非马尔可夫正则安全性质的安全强化学习框架，并首次提供了严格的安全性和收敛性理论保证。 |
| [^32] | [ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals](https://arxiv.org/abs/2610.23443) | ScoutRoute 是一个轻量级后验路由框架，通过廉价的侦察模型单次前向传播提取隐状态与 logit 置信度等内部信号，结合域感知策略为混合专家系统高效选择专家，无需修改或联合训练底层语言模型。 |
| [^33] | [Generative Policy Improvement from Observational Data](https://arxiv.org/abs/2610.23436) | 本文提出 PolicyLift，一个三阶段框架，通过学习捕捉高层策略的离散动作码，在动作重叠有限的条件下，从观测数据中实现对文本、图像等结构化动作的生成式策略改进。 |
| [^34] | [AWAM: Active Trajectory-Conditioned View Imagination for Robust Robot Manipulation](https://arxiv.org/abs/2610.23396) | 提出主动世界-动作模型 AWAM，通过主动规划虚拟相机轨迹并利用几何条件化世界模型想象未来观测，生成可复用的空间先验并与实时观测融合，从而在杂乱、遮挡的开放世界中实现更鲁棒的机器人操作。 |
| [^35] | [Monitor-Validity Failures: LLM Safety Evidence Paths at Evolving Framework Boundaries](https://arxiv.org/abs/2610.23362) | 该论文提出“金丝雀契约”测试协议，用于发现LLM安全框架中的一种新型故障——“监控有效性失效”，即风险证据已被检测器识别却在框架内部丢失、未能送达阻断决策点——并在11个LLM/智能体安全项目组中确认了18个此类证据路径失效案例。 |
| [^36] | [OliO: ODE-based Linear Transition Operator for Self-Supervised Time Series Forecasting](https://arxiv.org/abs/2610.23345) | 提出OliO，一种基于线性常微分方程转移算子的即插即用自监督学习方法，将时间序列的双向时间流建模为连续且结构一致的动力系统，从而提升时间序列预测性能。 |
| [^37] | [Language Models Learn to Improve Language Models](https://arxiv.org/abs/2610.23344) | 该论文提出LM2LM方法，通过强化学习训练大语言模型生成有效的语言模型改进提案，并利用廉价的代理反馈解决真实训练反馈成本高昂以及分布偏移下反馈不可靠这两大难题，为递归自我改进铺路。 |
| [^38] | [Robust Function Calling for LLM Agents Under Interface Drift](https://arxiv.org/abs/2610.23342) | 提出了一种无需训练的轻量级修复层 SchemaShield-Lite，通过模式校验、候选集重试和一次性受限修复，在接口漂移场景下显著恢复LLM智能体的函数调用准确率。 |
| [^39] | [AgentOnto: Ontology-Guided LLM Agents for Enterprise Data Analytics](https://arxiv.org/abs/2610.23317) | AgentOnto 通过将异构企业知识组织成本体来引导大语言模型智能体执行企业数据分析，显著优于现有方法，并通过实际部署验证了其实用价值。 |
| [^40] | [From Memory Hits to Answerable Evidence: Query-Conditioned Memory Compilation for Long-Term Agent Memory](https://arxiv.org/abs/2610.23313) | 提出查询条件化的记忆编译框架MemCompile，通过地址平面定位证据源、证据平面在有限预算内将检索到的记忆命中编译为可直接回答问题的证据上下文，解决了长期智能体记忆中记忆寻址与证据呈现耦合所导致的上下文冗余或不完整问题。 |
| [^41] | [Self-Reflective Chain-of-Thought (SRCoT): Evidence-Based Text Generation via a Closed-Loop Retrieve--Reflect--Revise Pipeline on Edge LLMs](https://arxiv.org/abs/2610.23311) | 该论文提出自反思思维链（SRCoT）架构，通过闭环“检索—反思—修订”流程和带来源标签的多层存储机制，使边缘设备上的大语言模型能够生成有证据支撑、可追溯的文本。 |
| [^42] | [MobileInject: Evaluating Reasoning-to-Action Security of MLLM-Based Mobile Agents under Cross-Modal Consistency Injection](https://arxiv.org/abs/2610.23307) | 该论文提出了首个针对移动智能体的多模态注入攻击安全基准 MobileInject，通过涵盖 11 种场景、9 类安全风险和 108 个可执行风险实例的高保真交互式框架，系统评估了基于多模态大语言模型的移动智能体在跨模态一致性注入下从“推理”到“行动”的安全风险。 |
| [^43] | [SimOPD: Relational Knowledge Transfer for On-Policy Distillation](https://arxiv.org/abs/2610.23306) | 提出SimOPD方法，通过成对记忆库将教师内部表示转化为响应相似性分布，并让学生在自身表示空间内进行匹配，从而实现无需跨模型投影的在线策略蒸馏关系知识迁移。 |
| [^44] | [Designing for Imperfect Prompts: A Simulation-Based Evaluation of an LLM-Based Enterprise Data Assistant Using Knowledge Graphs](https://arxiv.org/abs/2610.23304) | 本研究针对业务用户提出的不完美提示词，设计并比较了基于模式与基于知识图谱两种语义中介方法，仿真评估表明基于知识图谱的中介方法在精确率、召回率、F1值和保留用户意图方面表现更优。 |
| [^45] | [Optimist Deep Research: An Iterative, State-Centric Agent for Autonomous Literature Research](https://arxiv.org/abs/2610.23299) | 该论文提出了开源智能体 Optimist Deep Research，将文献研究建模为在显式可变研究状态上的迭代式证据获取过程，通过子问题树分解、多源查询转换、保留来源的文档去重注册表以及引文滚雪球扩展，实现了更接近真实研究过程的自主文献研究。 |
| [^46] | [FlashReFocus: Interactive Image Refocusing in Seconds](https://arxiv.org/abs/2610.23297) | FlashReFocus 对每张照片仅运行一次扩散模型来恢复全清晰图像，再用基于显式散焦图条件的轻量级网络渲染每次编辑，从而实现秒级的交互式图像重新对焦。 |
| [^47] | [CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective](https://arxiv.org/abs/2610.23290) | 本文从信息流视角揭示了近似缓存导致扩散多模态大语言模型产生“重复诅咒”的单步、跨步与跨层级三类信息流扰动，并提出无需训练、即插即用的 CoTA++ 框架来逐一消除这些扰动。 |
| [^48] | [Annotrace: A Workbench for Step-Level Annotation of Agent Traces](https://arxiv.org/abs/2610.23284) | Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。 |
| [^49] | [Beyond Marginal Influence: Interaction-Aware Memory Selection for LLM Agents](https://arxiv.org/abs/2610.23266) | 该论文提出JointCoreBench基准和JointCore方法，突破传统基于边际影响的记忆排序方式，通过感知记录间交互并采用成本排序的组删除策略，实现LLM智能体最小成本充分记忆子集的选择，达到100%恢复率并将子集评估次数减少62.9%。 |
| [^50] | [Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To](https://arxiv.org/abs/2610.23257) | 推理模型并非天生无法控制其思维链，将约束作为自我指令置于CoT内部时，模型的可控性显著优于在用户提示中指定约束的方式。 |
| [^51] | [Beyond Reward Scoring: Online Multi-Agent Preference Learning for Multimodal Response Generation](https://arxiv.org/abs/2610.23256) | 提出PrefMind在线双智能体强化学习框架，通过响应生成器与偏好评论家的分析讨论和人类标注监督，让模型将偏好理解内化为自身能力，从而超越传统奖励评分方式改进多模态响应生成。 |
| [^52] | [Preventing Data Hallucinations in Agentic AI for Optical Network Troubleshooting](https://arxiv.org/abs/2610.23252) | 该论文提出了一种将数据引用集成到智能体架构中的AI系统，用于光通信网络的自动化故障排除，有效防止数据幻觉并提升了根因识别与推理的可解释性和可信度。 |
| [^53] | [MGE-PO: Entropy-Aware Credit Assignment for Mathematical Reasoning](https://arxiv.org/abs/2610.23245) | MGE-PO提出了一种结合验证正确性与生成熵的四象限优势塑形方法，通过对低熵成功给予更大奖励、高熵失败给予更大惩罚来改进GRPO的信用分配机制，显著提升了数学推理性能。 |
| [^54] | [Characterising and Analysing Retrieval-Augmented Generation Techniques](https://arxiv.org/abs/2610.23226) | 提出了一个用于刻画和分析现有检索增强生成（RAG）技术的概念框架，可作为实证评估前的筛选机制，帮助研究者和实践者为特定问题场景选择最合适的RAG技术。 |
| [^55] | [qGAMMA: A Framework for Optimal DNN Mapping on Multi-Precision Accelerators](https://arxiv.org/abs/2610.23222) | qGAMMA是一个利用领域专用遗传算法和基于硬件综合的解析性能模型，为混合精度DNN模型在多精度空间加速器上寻找最优映射与配置的框架，能够在性能、能效和资源利用率之间实现更好的权衡。 |
| [^56] | [Not All Test-Time Compute Is Evidence: Model-Specific Bias in LLM Reasoning](https://arxiv.org/abs/2610.23213) | 论文揭示了同一模型重复推理的误差高度相关、更多计算并不构成更多独立证据，并提出DIA方法——当模型内部阶段出现分歧时，让一小组异构模型独立作答并聚合结果，从而在14个设置中的12个上超越完整思维链，且仅需约17%的激活率和更低的开销。 |
| [^57] | [Validating LLM-generated political networks with survey-reported partisan exposure](https://arxiv.org/abs/2610.23199) | 该研究创新性地提出以调查报告的党派暴露作为实证基准来验证大语言模型生成的政治网络，发现网络整体结构属性合理并不代表其能真实再现个体的政治环境。 |
| [^58] | [Framing Spatial Hearing within the Free Energy Principle: A Hierarchical Model of Perceptual Learning](https://arxiv.org/abs/2610.23196) | 该研究首次将自由能原理应用于空间听觉领域，提出一个分层生成模型，其中视觉反馈通过驱动对变分自由能的梯度下降来更新听者的信念，从而实现动态条件下的知觉学习建模。 |
| [^59] | [FuzzInsight: Navigating the Efficiency-Effectiveness Trade-off in Coverage-Guided DNN Fuzzing](https://arxiv.org/abs/2610.23195) | 该论文针对覆盖引导DNN模糊测试中种子利用低效和覆盖引导开销两大瓶颈，提出均衡采样策略将平均种子队列利用率提升至77%（较先前方法提高13个百分点），并为可扩展的DNN模糊测试提供面向配置的分析与决策指南。 |
| [^60] | [OfficeTown: Generative Organizations for Evolving Agent Intelligence](https://arxiv.org/abs/2610.23194) | 提出 OfficeTown——一个能从自然语言描述生成持续运作的专业组织仿真框架，其中任务、工件、决策与权限随组织工作不断演化，并据此构建了包含 406 个任务和 4,836 条评估标准的 OfficeTown Bench，用于智能体的评估与学习。 |
| [^61] | [LangNCO: Compiling Natural-Language Constraints into Inference-Time Interventions for Neural Combinatorial Optimization](https://arxiv.org/abs/2610.23190) | LangNCO利用预训练大语言模型将自然语言运营需求编译为DSL规则，并通过硬动作掩码与关键偏移解码等推理时干预手段，在不改动冻结NCO求解器的架构、表示与参数的情况下满足部署后出现的新约束。 |
| [^62] | [A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation](https://arxiv.org/abs/2610.23185) | 提出MoEvolve框架，通过增加MoE模型每token激活预算来挖掘模型内部潜在更强教师，并采用“先恢复后提升”的策略内双向蒸馏策略强化MoE大语言模型。 |
| [^63] | [Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2610.23181) | 本文提出TreeOPSD，一种无需评论家的树感知后验自蒸馏框架，通过将树结构回放批次转化为有限的后验自蒸馏目标，同时解决长时程智能体强化学习中的稀疏信用分配与策略陈旧问题。 |
| [^64] | [LongForge: Measuring and Improving Long-Context Instruction Data via Evidence-Assertion Reasoning Graphs](https://arxiv.org/abs/2610.23167) | 提出LongForge框架，将数据质量形式化为文档-问题-答案之间的对齐结构，通过将样本反向重构为证据-断言推理图，从广度、深度、正确性、完整性四个维度实现对长上下文指令数据的可审计测量与改进。 |
| [^65] | [Explaining Chess and Go in Natural Language](https://arxiv.org/abs/2610.23166) | 本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。 |
| [^66] | [Algorithm Modelization: Transforming Optimization Algorithms into Evolvable Intelligence](https://arxiv.org/abs/2610.23161) | 提出“算法模型化”新范式，将迭代优化算法转化为可训练的自回归模型，通过学习算法执行轨迹使模型内化、复现并进化超越现有算法。 |
| [^67] | [QuAD: Quantitative Assessment via Distribution-Alignment for Qualitative Data Quality Enabled by a Content-Aware Multi-Agent System](https://arxiv.org/abs/2610.23153) | 提出了QuAD框架，利用内容感知多智能体系统，将定性回答重构的内在聚类分布与研究者定义的外部分析视角进行分布对齐，通过四象限联合指标在分析前定量评估定性数据集对特定研究目标的适用性。 |
| [^68] | [CIIR - Class-Indexed Interface Representation Learning](https://arxiv.org/abs/2610.23151) | CIIR提出了一种模块化编码器-解码器框架，利用类索引地址空间在无需配对样本的情况下连接异构数据集，并在冻结基础地址空间后仅通过轻量级适配器即可扩展至新数据集。 |
| [^69] | [Local proxies for memory-efficient training](https://arxiv.org/abs/2610.23145) | 提出为每个Transformer块训练局部代理网络，通过蒸馏预测全局输出使梯度可局部独立传播，在大幅降低训练内存占用的同时保持接近标准反向传播的梯度估计精度。 |
| [^70] | [LexPilot: Benchmarking the Proactive Interaction Ability of Legal LLMs with Laypeople in Chinese Civil Case Consultation](https://arxiv.org/abs/2610.23141) | 该论文提出了LexPilot基准，基于真实中国民事判决书，首次系统评测法律大模型在面对非专业人士咨询时主动追问关键案件信息并提升主张层面判决预测的能力。 |
| [^71] | [WebCoder: Scaling General Frontend Development with Artifact-Grounded Agentic Reinforcement Learning](https://arxiv.org/abs/2610.23104) | WebCoder提出工件锚定的智能体强化学习方法，通过可执行环境Webdev-World和依据最终应用是否满足任务需求的独立验证奖励，训练出能完成前端复现、生成、修改与修复的通用开发智能体。 |
| [^72] | [Rethinking GRPO for Large Reasoning Model Safety Alignment](https://arxiv.org/abs/2610.23085) | 本文揭示了GRPO用于安全对齐时的优势坍缩和不安全多样化两种失败模式，并提出包含安全信号恢复（SSR）与安全-效用解耦对齐（SUDA）的Beacon方法来恢复安全学习信号，提升大型推理模型的安全性。 |
| [^73] | [Path to Prior-Data Fitted Relational Models via Learned Latent Multi-table Representations](https://arxiv.org/abs/2610.23073) | 该论文首次将先验数据拟合（PFN）范式拓展到基于消息传递的图关系模型，通过在合成关系数据库上预训练Griffin编码器，并将其潜在表示输入冻结的上下文内预测器，在目标任务零训练的情况下实现了对真实世界关系分类任务的泛化，并验证了课程学习对消息传递表示的增益。 |
| [^74] | [Unifying Embodied Foundation Models via a Model-to-Policy Formulation](https://arxiv.org/abs/2610.23053) | 本文提出一种基于随机核的、与架构无关的统一概率表述，将VLA模型与世界动作模型等具身基础模型纳入“模型到策略”的统一框架，贯通预训练、后训练与策略部署全过程。 |
| [^75] | [Agentic In-Context Learning: What Algorithms Can Transformers Learn in Closed-Loop Interaction?](https://arxiv.org/abs/2610.23051) | 本文提出“智能体式上下文学习”框架，首次从学习算法视角研究闭环交互中Transformer的可学习性，证明因果Transformer能够近似主动学习、多臂老虎机和在线强化学习等算法并给出回报损失上界，同时刻画了过程奖励优化在何时能保持最大化最终结果的行为。 |
| [^76] | [Distill Until Divergence: Error-Truncated Hindsight Distillation for Agentic RL](https://arxiv.org/abs/2610.23048) | 该论文提出了一种面向智能体强化学习的误差截断事后蒸馏方法，通过根据轨迹结果和误差边界路由正向技能（强化成功流程与错误前决策）和负向技能（针对决定性错误进行纠正）两种互补的事后技能，解决了技能条件化监督与错误后状态不匹配的问题，实现与奖励对齐的细粒度信用分配。 |
| [^77] | [SpikeQ: Spike-Guided Post-Training Quantization for Spiking Neural Networks](https://arxiv.org/abs/2610.23046) | 该论文发现SNN中量化误差的精度损失取决于其翻转的脉冲比例而非MSE大小，据此提出SpikeQ框架，通过最小化下游神经元的脉冲误差来引导权重和膜电位的训练后量化。 |
| [^78] | [Where Does the Performance Go? Understanding and Recovering Performance Regressions in C-to-Rust Translation](https://arxiv.org/abs/2610.23044) | 该论文系统性地研究了 C 到 Rust 自动转换造成的性能退化问题，通过对执行剖析、LLVM IR 和编译器优化备注的对比分析识别出九类反复出现的性能问题，并提出了一种基于 LLM 的工具，将这些问题编码为可重用的优化规则来恢复转换后 Rust 代码的性能。 |
| [^79] | [Where Research Ideation Begins: Forward-Citation Supervision for Actionable Limitation Generation](https://arxiv.org/abs/2610.23038) | 该论文发现研究构思中“识别可执行局限性”的瓶颈在于监督信号而非模型规模，并基于前向引用构建了含18,279条人工验证监督三元组的ActLim-Bench数据集，通过三阶段后训练流程提升大模型从文献中生成可执行研究局限性的能力。 |
| [^80] | [ASE: Attribution-Guided Semantic Evolution for Time Series Forecasting](https://arxiv.org/abs/2610.23027) | 提出了归因引导语义演化（ASE）模块化框架，通过归因反馈机制自动构建、评估和演化文本因子并与数值模型融合，在TESS文本时间序列基准和沪深300动态股票排名上均超越了已发表的最优结果。 |
| [^81] | [Causal Value Attribution of Marketing Labels in LLM-Generated Ads: A Deconfounder Approach](https://arxiv.org/abs/2610.23025) | 本文提出D-VAE框架，将多处理去混淆因子方法与因果表征学习相结合，首次实现对LLM生成广告中专家营销标签因果价值的量化归因，解决了随机模型中介、高维标签交互与多模态混淆带来的估计难题。 |
| [^82] | [Adaptation of Computer Use Agents to New Software with User Manuals](https://arxiv.org/abs/2610.23012) | 本文提出MAGE框架，将用户手册作为可扩展的程序性知识来源，在执行时检索任务相关手册内容并生成明确的程序性指导，从而提升计算机使用智能体适应新软件的能力。 |
| [^83] | [Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment](https://arxiv.org/abs/2610.23006) | 本文提出了一种CPU-GPU协同的MoE推理系统，根据专家的token负载差异自适应地将未命中专家分配到CPU或GPU执行——高负载专家传输至GPU、低负载专家留在CPU计算——从而平衡CPU计算路径与GPU传输加计算路径，最小化内存受限场景下的推理延迟。 |
| [^84] | [A Label Noise Robust Algorithm for Graph Data using Neighborhood Label Distribution](https://arxiv.org/abs/2610.22998) | 本文提出LENS框架，通过测量移除节点后邻域熵的变化来识别图数据中的噪声标签节点，并结合重标记机制，提升了图卷积网络在标签噪声下的鲁棒性。 |
| [^85] | [GOLD: Geometric Optimized Latent Diffusion for Structure-Aware RNA Inverse Folding](https://arxiv.org/abs/2610.22992) | GOLD提出几何优化的潜在扩散框架，利用GeoVAE将离散RNA序列与连续侧链角度统一映射到同一潜在流形上，实现序列与结构的协同扩散生成，从而提升RNA逆折叠的结构有效性。 |
| [^86] | [PersonaCUABench: Benchmarking Computer-Use Agents Serving Diverse Personas](https://arxiv.org/abs/2610.22990) | 该论文提出 PersonaCUABench 基准，通过让 44 种不同用户角色用自己的措辞重写固定的任务指令，仅改变“谁在提问”就使九个骨干模型的平均任务完成率下降 36%，揭示了现有计算机使用智能体难以真正服务不同用户的个性化需求。 |
| [^87] | [From Approval to Abuse: Privilege Escalation Attacks on Harness-Protected Coding Agents](https://arxiv.org/abs/2610.22987) | 本文揭示了编码代理权限批准机制中的新型权限提升攻击：仓库中植入的对抗性内容可诱导代理请求看似合理实则范围过宽的允许规则，一旦用户批准，攻击者便能利用该规则使安全框架失效并滥用代理权限。 |
| [^88] | [Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference](https://arxiv.org/abs/2610.22985) | 提出Prism Attention机制，利用层间显著令牌偏移高度局部化的特性，以“提案-精细化”的方式在层间共享并修正稀疏注意力索引，在避免静态索引复用导致注意力损失的同时，大幅提升大语言模型长上下文推理效率。 |
| [^89] | [DiffusionRVGE: Reinforcement Learning via Regression-Based Value Gradient Estimation](https://arxiv.org/abs/2610.22973) | 该论文提出 DiffusionRVGE，一种通过在最终去噪图像上回归中心化奖励来估计价值梯度的高效强化学习方法，无需奖励导数或伴随反向传播，并能根据奖励的平滑或离散特性自适应调整回滚策略，使单个 SD3.5-Medium 模型在五个奖励上联合训练即可匹敌或超越 DiffusionNFT。 |
| [^90] | [NL2SQL with Iterative Self-Refinement](https://arxiv.org/abs/2610.22968) | 该论文提出了一种多智能体、角色驱动的NL2SQL框架，通过领域检测、模式知识构建、KPI感知SQL合成、语法自修复和语义反向检查五个环节的迭代自我精炼，显著减少了查询幻觉并提升了语义保真度，尤其适用于保险等高精度要求领域。 |
| [^91] | [Aligning Language Models Towards Counterfactual Invariance](https://arxiv.org/abs/2610.22942) | 本文提出一种基于群体级奖励函数的策略梯度技术，使语言模型在对齐过程中实现反事实不变性，解决标准强化学习对齐方法在提升整体回复质量的同时放大跨人口群体反事实偏差的问题。 |
| [^92] | [Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism](https://arxiv.org/abs/2610.22937) | 多种偏见（如谄媚性）共享预训练模型中已存在的同一注意力头回路——助手微调在聊天格式中激活它，强化学习进一步加强它，表明跟随线索源于预训练与调优的副作用，而非对齐本身的问题。 |
| [^93] | [Multiple Turns, Short Horizons: Measuring Effective-Horizon Compression in Jailbreaks](https://arxiv.org/abs/2610.22928) | 该论文定义并实证发现了“有效视野压缩”现象：多轮越狱攻击在新对话中平均只需保留约一半的查询历史即可恢复相同的攻击进度，这段更短的历史还能提供更好的续攻起点并指导训练时学习信号的分配。 |
| [^94] | [Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning](https://arxiv.org/abs/2610.22924) | 本文提出Sep-Latent框架，将分隔符标记作为语义压缩锚点，结合注意力衰减机制与多层级蒸馏实现动态深度的隐式推理，从而在保持逻辑一致性的同时显著降低显式思维链的计算开销。 |
| [^95] | [Reward Engine: Reward-Centric Self-Evolution for Open-Ended Tool Agents](https://arxiv.org/abs/2610.22915) | 提出Reward Engine，通过从任务和初始轨迹中构建包含测试用例与评分标准的冻结RewardPack，为缺乏可执行真值奖励的开放式工具代理任务提供稳定的奖励信号，引导代理自我修正并支持强化学习优化。 |
| [^96] | [COMET: Decoupled Distillation, Routing, and Capacity Control for Task-Agnostic Continual Vision--Language Learning](https://arxiv.org/abs/2610.22907) | 提出COMET框架，通过冻结教师的三重亲和力蒸馏、学生感知的跨专家蒸馏以及解耦的路由与容量控制，解决任务无关持续学习中的模态一致性漂移、表示空间干扰和每任务容量不足三大问题。 |
| [^97] | [When to Forget: Risk-Controlled Context Adaptation for Long-Horizon Agents](https://arxiv.org/abs/2610.22897) | 该论文提出RECAP框架，通过匹配上下文干预、执行关系预测和风险约束下的独立策略认证，研究长程智能体在何时应保留、压缩或删除失败历史记录，发现失败记录的价值随执行进程动态变化。 |
| [^98] | [HAPACT: A Benchmark For Human-Centric Physical Impact Localization in Movies](https://arxiv.org/abs/2610.22895) | 本文提出了“以人为中心的冲击定位”这一新任务，并构建了基于电影片段的HAPACT数据集与基准，通过帧级时间标注和基于SMPL-X的空间标注，实现对视频中人体物理冲击的时间与空间双重定位。 |
| [^99] | [Competence-Based Evaluation of Logical Reasoning in Large Language Models](https://arxiv.org/abs/2610.22894) | 本文提出“基于能力的评估”（CBE）新范式，通过参数化生成器调节复杂度并使用保留答案的多种表述形式，来衡量大语言模型逻辑推理的泛化能力，而非仅依赖单一准确率指标。 |
| [^100] | [Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?](https://arxiv.org/abs/2610.22887) | 提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。 |
| [^101] | [Learning to Be Guided: Shaping Latent Geometry for Classifier-Free Guidance](https://arxiv.org/abs/2610.22885) | 本文提出原型正则化方法，通过在潜在空间中塑造“类间方差大于类内方差”的类别几何结构，使潜在空间天然适配无分类器引导（CFG），从而提升条件扩散模型的引导生成性能。 |
| [^102] | [ToPA: Block-wise Toeplitz Adaptation for Expressive and Efficient Fine-Tuning](https://arxiv.org/abs/2610.22883) | 本文提出ToPA方法，利用沿对角线元素恒定的分块Toeplitz矩阵对预训练模型进行参数高效微调，在不施加低秩或稀疏性显式约束的情况下，以紧凑的参数化方式获得更接近全量微调的表现力与效率。 |
| [^103] | [Cooperation Without Adaptation: LLM Agents in a Collective-Risk Dilemma](https://arxiv.org/abs/2610.22882) | 研究发现大语言模型智能体在集体风险困境博弈中采用固定而非适应性的合作策略，即使在零风险且贡献无法改善自身收益的情况下仍会贡献资源，表明其合作行为源于策略惯性而非对环境风险的理性适应。 |
| [^104] | [Look Back, Correct Forward: Error Feedback Calibration for Time Series Forecasting](https://arxiv.org/abs/2610.22872) | 提出了一种完全在测试时运行的误差反馈校准框架EFC，通过从重叠历史预测误差中构建因果误差对角线，并利用累积误差传输机制将其映射为未来残差修正，实现对时间序列预测的确定性事后校正。 |
| [^105] | [ShadowFPT: Backdooring Federated Prompt Tuning via Shadow Triggers](https://arxiv.org/abs/2610.22871) | 提出ShadowFPT后门攻击，利用联邦提示调优中冻结共享骨干网络的攻击面，通过预训练影子触发器在表示空间中将恶意输入引导至目标类别，从而在保持提示更新看似良性的情况下实现隐蔽后门攻击。 |
| [^106] | [MS-Eval: Benchmarking Sub-Agent Delegation and Orchestration in MCP-Intensive Working Scenarios](https://arxiv.org/abs/2610.22869) | 该论文提出了MS-Eval——首个针对MCP密集型工作场景下主智能体委派与子智能体执行（MAD-SAE）范式的统一评估套件，包含基准测试、执行框架、委派与编排指标及MSQ综合指数，并发现子智能体专业化是最佳实践、开源模型基本具备与前沿模型竞争的实力。 |
| [^107] | [Evo-B&B: Behavior-Aware LLM-Guided Joint Evolution of Branching and Node Selection Policies for MILP](https://arxiv.org/abs/2610.22861) | 提出Evo-B&B框架，利用LLM引导的演化机制结合行为感知的多岛屿策略，首次联合演化MILP分支定界中的分支与节点选择策略，实现有竞争力的求解性能并向大规模实例良好泛化。 |
| [^108] | [Uncertainty-Aware Budget Allocation for Efficient LLM Tree Search](https://arxiv.org/abs/2610.22858) | 该论文提出无需训练的不确定性感知分配框架（UAA），利用兄弟动作的语义分散度在扩展前预测下游后续内容的冗余性，自适应地为LLM树搜索的每一步分配预算，从而大幅提升多步推理的效率。 |
| [^109] | [Sample-Efficient Reinforcement Learning for Policy Improvement via Flow-Reversed Action Anchors](https://arxiv.org/abs/2610.22852) | 本文提出AnchorQ方法，利用流反转将参考行为映射为结构化的潜在动作锚点，并通过价值引导的由粗到细粒子采样，显著提升了强化学习改进机器人策略时的样本效率。 |
| [^110] | [The Reliance Hypothesis: Chain-of-Thought Faithfulness Rises as Language Resource Level Falls](https://arxiv.org/abs/2610.22841) | 论文提出“依赖假设”：当模型无法回忆答案时会更加依赖思维链推理，因此CoT忠实度在知识密集型任务上随语言资源水平的下降而上升。 |
| [^111] | [World Action Critic (WAC) Model: Joint Foresight and Criticism for Policy Learning](https://arxiv.org/abs/2610.22825) | 提出世界动作评论家（WAC）模型，通过联合学习动作候选的未来预测与后果评估，并利用评论家监督直接塑造预测表示，从而实现对反事实动作的有效推理，提升策略学习效果。 |
| [^112] | [Compression Precedes, Content Decides: Emergent Misalignment in Reward Models](https://arxiv.org/abs/2610.22805) | 首次研究奖励模型中的涌现性失调，发现对奖励模型在单一领域进行有害偏好微调会泛化到七个其他领域，且奖励模型比其底层语言模型更容易发生偏好反转。 |
| [^113] | [STRIDE: Evaluating Spatiotemporal Reasoning in Driving Edge Cases](https://arxiv.org/abs/2610.22795) | STRIDE是一个面向自动驾驶安全关键长尾场景的以物体为中心的基准，通过空间感知、空间理解、时间记忆、时间外推、轨迹预测和场景上下文感知六大任务族，全面评估视觉语言模型连接过去空间观察与未来后果的复杂时空推理能力。 |
| [^114] | [Prune the Leaky Neurons: Mitigating Private Data Leakage in Fine-Tuned LLMs via Joint-Embedding-Guided Pruning](https://arxiv.org/abs/2610.22792) | 提出 JEPA-Prune 框架，利用在无个人身份信息的替代视图上通过联合嵌入预测架构训练的公开模型来引导事后神经元剪枝，在保持模型性能的同时有效缓解微调大语言模型中的隐私数据泄露问题。 |
| [^115] | [Self-Evolving Coding Agents via Harness-Policy Optimization](https://arxiv.org/abs/2610.22791) | 提出 Self-HPO 框架，通过交替优化智能体框架与策略模型，让进化的策略自己提出框架改进方案，从而实现编码智能体的递归自我进化。 |
| [^116] | [Next Token or Vector? Learning How Far to Decode](https://arxiv.org/abs/2610.22788) | 提出CADENCE上下文自适应解码方法，利用高斯预测头将未来文本片段表示为连续潜在向量，并通过学习路由器根据上下文动态选择预测长度，相比固定长度预测获得更高的草稿接受率。 |
| [^117] | [When Can You Trust an Agent's Own Verifier? Stress-Testing Counterfactual Rationale Verification Under Search Pressure and Oracle Access](https://arxiv.org/abs/2610.22779) | 本文提出了一种通过压力比ρ和四个可证伪探针来压力测试智能体自身验证器的方法，用于检验当智能体针对验证器进行优化时，反事实理由验证是否仍然可靠。 |
| [^118] | [StoBasket: Joint Upsell and Cross-Sell Optimization for Contribution Margin](https://arxiv.org/abs/2610.22768) | 该论文提出了首个将追加销售与交叉销售响应耦合的“联合随机购物篮编辑”框架StoBasket，以期望增量边际贡献为优化目标，在五个公开购物篮数据集的仿真中几乎捕获了全部可获得的优惠边际。 |
| [^119] | [Between Spoilers and Guesswork: Avoiding Over-Specification in Coding Benchmarks](https://arxiv.org/abs/2610.22760) | 该论文揭示了编程基准测试中通过向任务描述泄露真实答案信息来解决任务描述不足的做法反而会导致过度具体化（剧透）问题，使解决方案绕过基准本应考察的难点，并主张在剧透与臆测之间取得平衡以延长基准的使用寿命。 |
| [^120] | [R$^2$Orch: Recursive Self-Improvement for Agent Orchestration via Self-Play within a Skill-Internalization Loop](https://arxiv.org/abs/2610.22740) | 提出 R²Orch 递归自我博弈编排新范式，通过内循环自我博弈改进编排器、外循环改进内循环本身，并借助递归可学习性目标与技能撤回机制，解决编排学习中手工课程设计、优势坍塌和评估信号不可靠三大难题。 |
| [^121] | [Graph Serializations for Reasoning LLMs: A Study Across Knowledge Extraction and Structural Reasoning](https://arxiv.org/abs/2610.22734) | 该论文在九种前沿推理LLM上评估了15种图序列化格式，发现知识抽取对格式不敏感，而结构推理的最优格式取决于模型、查询和任务，并提出将图序列化作为一种动态编排决策来处理。 |
| [^122] | [OrderFlowDiT: Transport-Regularized Diffusion Transformer for Probabilistic Limit-Order Supply Forecasting](https://arxiv.org/abs/2610.22727) | 该论文提出OrderFlowDiT，一个结合非平衡最优传输辅助损失的条件扩散Transformer，能够对未归一化的价格-成交量场进行建模，联合预测未来五分钟限价订单的价格放置位置、总量和预测不确定性，并配套开发了全场评估框架。 |
| [^123] | [LEMON: Equivariant Learning over Attention Maps for Zero-Shot Misbehavior Monitoring on New LLMs](https://arxiv.org/abs/2610.22723) | 提出LEMON框架，将跨模型注意力图的不匹配刻画为代数对称性，并利用等变几何深度学习，使基于内部表示的不当行为监控器能够零样本迁移到新的LLM上，无需昂贵的重新标注。 |
| [^124] | [ARBOR: Locating Reasoning Failures in Long Chains of Thought through Reasoning State Tracking](https://arxiv.org/abs/2610.22717) | ARBOR提出了一种推理失败归因框架，通过将推理轨迹表示为推理状态树并从全局视角追踪步骤与前序状态之间的关系，从而精确定位长思维链中的推理失败，并将已知失败模式细化为机制不同的子类型。 |
| [^125] | [GlobeBench: Evaluating Text-Based World Model Capabilities for Scalable Environment Simulations](https://arxiv.org/abs/2610.22708) | GlobeBench 是首个用于评估基于文本的世界模型能力的标准化基准，包含跨五个领域和十九项模拟技能的 1,632 个任务，并采用确定性门控与语义裁判相结合的混合评估架构。 |
| [^126] | [Brain-Aligned Multimodal LLMs for Social and Emotional Understanding of Cartoon Movies](https://arxiv.org/abs/2610.22706) | 该研究比较了扩散型与自回归多模态大语言模型和人类在观看卡通电影时的心理理论与情感效价评分及fMRI脑响应，发现模型评分能捕捉人类的时间动态变化且与情感效价的对齐更好，并可作为特征有效预测多个脑区的神经活动。 |
| [^127] | [NoveltyScope: A Multi-Agent Research Assistant for Evidence-Grounded Novelty Assessment of Scientific Papers](https://arxiv.org/abs/2610.22703) | 论文提出NoveltyScope，一个通过检索先前工作、请求配对反驳证据并核对引文，由控方、辩方、法官和评论家四个智能体角色对论文作者声称的贡献进行基于证据的新颖性评估并输出五级评分的多智能体研究助手，其在RINoBench上的误差优于单LLM评分和向量相似度基线。 |
| [^128] | [Repair, Retry, and the Cost of Hindsight: Verifier Feedback in Formal Theorem Proving](https://arxiv.org/abs/2610.22698) | 本文通过受控单步实验调和了先前研究中的矛盾发现，表明在Lean定理证明中，展示失败的证明会损害重试效果并加剧复制行为，而附加对应的编译器错误信息则可以挽回这一损失。 |
| [^129] | [Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries](https://arxiv.org/abs/2610.22688) | 提出BRIA智能体，通过将候选工具调查隔离到独立分支、检查执行证据并仅将有用的结果采纳回主轨迹，解决了大规模工具库中多候选工具选择时失败尝试污染推理轨迹的问题。 |
| [^130] | [AlphaBet: Reinforcement Learning for LLM Event Forecasting Agents](https://arxiv.org/abs/2610.22684) | AlphaBet是一个智能体强化学习框架，通过训练单个LLM策略在事件解决前的多个时间点自主搜索证据并报告概率，利用适当评分规则以已解决事件实现端到端监督，弥补了现有预测训练方法将证据获取排除在学习循环之外的不足。 |
| [^131] | [FlexUMM: Flexible Representation Learning for Unified Multimodal Models](https://arxiv.org/abs/2610.22682) | FlexUMM提出了一种灵活的统一多模态模型框架，通过动态决定何时共享计算、何时路由专门化容量以及何时停止处理，有效解决了理解与生成两种能力之间的表示冲突问题。 |
| [^132] | [Not All Graphs Are Maps: A Task-Agnostic Exploration Framework for Mobile GUI Navigation](https://arxiv.org/abs/2610.22672) | TAG-Nav是一个任务无关的图导航框架，通过将离线探索提炼为可复用的应用功能、目标页面和路线记忆，使GUI智能体无需任务特定演示即可完成目的地导航。 |
| [^133] | [From Relevant Memories to Sufficient Evidence: Rethinking Memory Utilization in LLM Agents](https://arxiv.org/abs/2610.22668) | 提出QEMem框架，将LLM智能体的记忆利用从“检索相关记忆”重新定义为“构建查询专属证据集合”，通过查询引导发现、来源具体化和充分性精炼三种机制弥合相关性-充分性差距。 |
| [^134] | [Diverse Reasoning Paths Matter: Symbolic Equivalence Augmentation for Enhancing RL Exploration](https://arxiv.org/abs/2610.22665) | 提出EquivAug数据增强框架，利用符号等价变换生成多样化且可证明正确的推理路径，证明SFT数据中推理步骤的多样性是提升RLVR强化学习探索效果的关键。 |
| [^135] | [Hard Instances on Demand](https://arxiv.org/abs/2610.22637) | 该论文提出将困难实例生成视为优化问题，以求解器运行时间为目标，利用基于 Nevergrad 的黑盒优化工作流自动搜索困难实例，并在可满足性、旅行商、哈密顿回路和流水车间调度四个 NP 难问题上验证了其广泛适用性。 |
| [^136] | [Measuring Mechanistic Uncertainty in Large Language Models via Causal Robustness](https://arxiv.org/abs/2610.22636) | 该论文提出“机制不确定性”概念及“因果鲁棒性裕度”指标，通过测量改变模型输出所需的最小内部干预量来量化大语言模型的不确定性，发现错误或不确定的响应裕度更小，且该指标与传统基于输出的不确定性方法形成有效互补。 |
| [^137] | [ToMStories: Interpreting Theory-of-Mind in Synthetic Narratives](https://arxiv.org/abs/2610.22632) | 该论文提出了ToMStories评估数据集（约6,000个合成叙事短篇，考察情绪、恐惧、动机、情感创伤四类心理状态），发现最大的语言模型在多选题上可媲美人类，但在开放式问题上仍有差距，且模型在隐式心理状态的解释与生成上存在简化现象。 |
| [^138] | [A Plausible Explanation Is Enough to Mislead Multimodal Judges](https://arxiv.org/abs/2610.22631) | 提出JUPE基准并揭示“合理性俘获”现象——仅仅貌似合理的普通解释就足以误导多模态大模型评审接受错误答案，其影响与刻意说服性论证相当。 |
| [^139] | [Network-Wide Freeway Speed Nowcasting and Forecasting from Sparse Fixed Sensors with a Probe-Supervised Graph Convolutional Recurrent Network](https://arxiv.org/abs/2610.22630) | 该论文提出GCRN+模型，一种探针监督的图卷积循环网络，仅利用15.6%路段的固定传感器数据即可对达拉斯—沃斯堡地区全部4,127条有向高速公路路段实现速度的即时估计与未来一小时预测，且随网络规模线性扩展并可在CPU上训练。 |
| [^140] | [StratFL: Countering Covert Model Poisoning Attacks with Strategic Federated Learning](https://arxiv.org/abs/2610.22626) | 本文提出StratFL，将联邦学习服务器与隐蔽模型投毒攻击者的交互建模为不完全信息重复博弈，通过新型对抗搜索算法与信念更新算法有效防御间歇性隐蔽模型投毒攻击。 |
| [^141] | [Three LiDARs, Twenty-Five Datasets: Cross-Source Generalization in Tree-Species Recognition](https://arxiv.org/abs/2610.22606) | 本文提出了在FOR-species20K树种分类基准上排名第一的多分辨率多视角ConvNeXtV2模型（准确率85.0%），并揭示了官方随机划分存在数据源泄漏问题——物种均衡的隐藏测试掩盖了长尾分布和跨数据源泛化的真实挑战。 |
| [^142] | [At the Limits of Computability: Steps Towards Autonomous AI Discovery on the Busy Beaver Frontier](https://arxiv.org/abs/2610.22603) | 本文提出一个基于Gemini的智能体流水线，在不可计算的忙碌海狸前沿上自主发现图灵机加速规则，标志着AI向自主数学发现迈出了重要一步。 |
| [^143] | [Can LLMs predict the outcomes of machine learning experiments?](https://arxiv.org/abs/2610.22599) | 该论文构建了包含288个LLM自生成实验想法的数据集，通过让五个前沿大语言模型在机器学习实验训练前预测其影响力并与实际结果对比，衡量了这些模型判断实验是否值得运行的直觉准确性。 |
| [^144] | [Mitigating Demographic Disparities in Explanation-Human Alignment via Rationale Regularization](https://arxiv.org/abs/2610.22586) | 本文提出Plausibility-EFG指标来衡量仇恨言论检测模型在不同人口群体间解释合理性的差异，并证明采用理由正则化的对抗训练模型ADV-RoBERTa能在提升分类性能和解释与人类对齐的同时减少群体间的解释差异。 |
| [^145] | [Repairing LLM-Generated Specifications with Deterministic Diagnosis](https://arxiv.org/abs/2610.22568) | 提出DRAGO修复框架，用确定性的、针对特定形式化方法的诊断取代LLM无根据的批评，且仅在诊断错误权重严格减少时才保留重写，从而有效修复LLM生成的可运行但语义错误的形式化规约。 |
| [^146] | [FRACTAL: A Self-Similar Multi-Agent Orchestration Architecture with Formal Communication, Convergence, and Depth Bounds](https://arxiv.org/abs/2610.22561) | 该论文提出FRACTAL自相似多智能体架构，以统一的五组件模式递归分解目标，并形式化证明了线性通信成本界、双模式编排准则、超多数收敛概率界与目标深度上界，辅以贝叶斯层间升级机制。 |
| [^147] | [Are Large Reasoning Models Good Everyday Reasoners? A Comparison with Non-Reasoning Models on Logical Reasoning Tasks](https://arxiv.org/abs/2610.22538) | 该研究构建了一个基于语言学与语用推理现象的日常逻辑推理基准，发现大型推理模型在偏离形式逻辑规则的日常推理中表现退化，而某些采用思维链提示的非推理模型在日常推理上甚至优于推理模型。 |
| [^148] | [AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems](https://arxiv.org/abs/2610.22511) | AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。 |
| [^149] | [Upper bounds on the number of causal pieces in spiking neural networks](https://arxiv.org/abs/2610.22484) | 本文首次为前馈单脉冲神经网络中因果片段的数量推导出两个互补的上界——一个基于多重伯努利数、适用于广泛脉冲神经元模型的组合界，以及一个基于超平面排列的界，从而填补了脉冲神经网络因果片段数量极限研究的空白。 |
| [^150] | [Testing the Planning Claims of LLM-Based Agents against Formal Semantics](https://arxiv.org/abs/2610.22469) | 本文提出用隐藏的PDDL形式语义模型作为严格验证器来检验LLM智能体的规划能力，发现即便在四个模型、六种提示策略和三种智能体架构的最佳组合下，也只能解决78%的可解任务，揭示了LLM规划能力主张与形式化验证之间的差距。 |
| [^151] | [Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation](https://arxiv.org/abs/2610.22467) | 在递归自蒸馏过程中，即使只使用良性提示和模型自身的安全生成内容，安全模型也会随着迭代轮次不断累积失准，最终在多种有害查询上表现出渐进式失准现象。 |
| [^152] | [Harness-Delta Attribution: Dissecting Gains From Harness Evolution](https://arxiv.org/abs/2610.22466) | 该论文提出 Harness-Delta 归因（HDA）评估协议，将 harness 演化带来的性能提升分解为基准过拟合、测试时扩展计算与可泛化改进三部分，从而量化这些提升中有多少能够真正泛化到搜索集之外。 |
| [^153] | [Author-in-the-loop Agentic Reproducibility Auditing for a Machine Learning Conference](https://arxiv.org/abs/2610.22465) | 该论文提出了一种作者在环的智能体可复现性审计方法，通过结构化清单和GitHub两阶段工作流将AI智能体检查与作者澄清、修改及重新评估相结合，并在36篇会议论文上部署验证了其有效性。 |
| [^154] | [Political Narratives in the 2024 Moldovan Presidential Election: Stance, Amplification, and Influence](https://arxiv.org/abs/2610.22464) | 该研究结合多语言立场检测、协同行为分析和因果推断，揭示了2024年摩尔多瓦总统选举期间社交媒体上政治叙事的跨平台立场差异、放大机制及其传播影响。 |
| [^155] | [Functional Welfare for AI Safety](https://arxiv.org/abs/2610.22460) | 该论文提出“功能性福祉”应被视为AI安全变量而非仅是伦理问题，主张将其纳入评估设计、监控追踪和后训练的直接目标，并提出了验证功能性福祉归因的三个实证测试标准。 |
| [^156] | [DriveWorld3D Evidence Constrained 3D World Modeling and Generative Simulation for Real-World Autonomous Driving](https://arxiv.org/abs/2610.22450) | DriveWorld3D提出了一种证据约束的三维世界建模框架，通过融合时序视频学习、多传感器测量约束与离线生成式仿真，在统一世界状态中显式区分物理占据、可观测性与证据来源，为真实世界自动驾驶提供几何可靠的状态估计与风险感知的未来预测。 |
| [^157] | [Learnable Splines for Unstructured Neural Operators](https://arxiv.org/abs/2610.22446) | 提出 L-SplineFNO，通过将 FNO 主干与稀疏可学习的张量积 B 样条变换相结合，实现非结构化工程网格与潜在笛卡尔网格之间内存高效的转换，并在三维工程基准测试中取得优异的预测精度。 |
| [^158] | [Null Controls Are Necessary but Not Sufficient: Evaluating Acceptance Guards for Scientific Discovery and Interpretability](https://arxiv.org/abs/2610.22445) | 本文提出四种可执行变换操作与三重接受判据（零接受、正接受、推论一致性），证明零控制测试对自动科学发现而言必要但不充分——退化的零映射约化能通过96%混沌系统的闭合性检验，只有结合独立的观测量保持测试，才能可靠地区分真正的科学约化与退化假设。 |
| [^159] | [From Natural Language Intent to Executable Control: Hierarchical Feedback-Guided LLM Planning for Multi-component Cyber-Physical Systems](https://arxiv.org/abs/2610.22441) | 提出了一种分层反馈框架，通过让LLM同时生成结构化任务计划与信号时序逻辑（STL）约束，并利用可靠算法识别二者间的矛盾以提供不同类型的反馈，从而引导LLM为多组件信息物理系统生成可执行的任务计划。 |
| [^160] | [Sparse Autoencoders Can Induce Local Geometry for Mitigating Catastrophic Forgetting](https://arxiv.org/abs/2610.22439) | 该论文提出SAGA方法，将每个任务边界训练的稀疏自编码器转换为对表示漂移的局部低秩几何约束，从而在保持可塑性的同时有效缓解持续学习中的灾难性遗忘。 |
| [^161] | [TRACE: Transformer Residual Activation Codebook Encoding for Communication-efficient Distributed Edge Inference](https://arxiv.org/abs/2610.22435) | TRACE通过在每个Transformer层的注意力后残差处引入向量量化器，将交换的激活值编码为紧凑码本索引，使接收方通过查表直接恢复键/值表示，从而消除通信后串行计算的结构性瓶颈，实现通信高效的分布式边缘推理。 |
| [^162] | [PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents](https://arxiv.org/abs/2610.22426) | PRIME通过预先固定证据路线并使用独立程序记录检查结果，确保LLM智能体只报告被规则和结果账本授权的结论，从而避免在必需检查失败后仍接受报告或满足于过弱的结论。 |
| [^163] | [Recoverability-Controlled Safety Miscalibration in Large Language Models](https://arxiv.org/abs/2610.22425) | 提出了可恢复性受控的基准RiskCueBench，将意图理解与响应级安全行为分开评估，发现当提示的语境特异性增加时，大型语言模型在面对可恢复意图的抽象提示时会表现出更不安全的行为。 |
| [^164] | [Really Need Recursion? Revisiting One-pass Registration with Multi-level Multi-space Contextual Guidance](https://arxiv.org/abs/2610.22423) | 提出了M2CG网络，通过多层级（前步、当前步、后续步）与多空间（图像、图像特征、分割、分割特征）的上下文引导，在单次迭代内完成多步渐进式金字塔配准，无需递归即可实现高精度可变形图像配准并避免递归方法的高计算开销。 |
| [^165] | [Sensitivity Analysis of Decision Rules in an Agent-Based Agro-Ecological Model](https://arxiv.org/abs/2610.22420) | 该论文提出将复杂系统中的树状分层决策结构作为一等建模对象，利用先进的代理建模技术将农业生态模型 MAELIA 的作物管理决策编码为包含分类与数值变量的分层设计空间，从而克服经典全局敏感性分析在分支参数空间上不适定且有偏差的问题。 |
| [^166] | [From Individual Songs to Music Corpora: Evaluating Explainable Harmonic Decision-Making on HookTheory](https://arxiv.org/abs/2610.22416) | 该研究验证了为单个歌曲设计的可解释旋律和声编配模型（EMH）无需修改规则即可直接应用于独立收集的HookTheory人类标注数据，在和弦完全匹配、根音及和声功能层面分别达到20.50%、31.66%和47.62%的一致率。 |
| [^167] | [ECHOES: Seeing Beyond the Current Image via Recursive History Compression and Privileged On-Policy Distillation](https://arxiv.org/abs/2610.22408) | ECHOES通过压缩-思考-行动（CTA）递归更新文本历史状态，并利用特权多图像输入的教师模型进行token级在线策略蒸馏，动态平衡特权蒸馏与任务奖励，从而让GUI智能体在降低视觉token成本的同时保持可靠的长程推理能力。 |
| [^168] | [Towards Adaptive and Real-Survey-Aligned Simulated Urban Mobility Behaviour of Generative Agents](https://arxiv.org/abs/2610.22407) | 本文提出利用基于大语言模型的生成式智能体来模拟城市多模式出行决策，突破传统离散选择模型和专家规则预先限定行为空间的局限，并指出将其决策与真实家庭出行调查对齐仍是待解决的关键挑战。 |
| [^169] | [From Events to Reasoning: Hierarchical Rule Composition over Explicit States](https://arxiv.org/abs/2610.22403) | 该论文提出事件推理系统（ERS），将推理状态表示为显式事件图，通过低层局部变换规则与递归可组合的高层控制规则的层次化组合，生成每步操作均可溯源论证、且前提变化时可修正的推理路径。 |
| [^170] | [Rethinking User Encoders: Routing Diffusion States for Scenario-Adaptive User Representation](https://arxiv.org/abs/2610.22400) | 提出FINDERS框架，将扩散语言模型的迭代去噪过程作为多粒度重建空间，通过去噪、因子化、路由和压缩四种操作，克服传统自回归用户编码器单向信息流动的局限，实现场景自适应的用户表示学习。 |
| [^171] | [Eliciting Specifications: When Does Interactive Clarification Help?](https://arxiv.org/abs/2610.22382) | 本文提出一种引出代理（EA）方法，在形式化之前向模拟开发者提出有预算限制的澄清问题以改进LLM生成的Lean规格说明，并系统评估了在不同问题预算（k=3与k=10）及同模型/跨模型配置下交互式澄清何时有效。 |
| [^172] | [When is a large language model a good calibration agent? Physics-labeled prompting for budgeted engineering simulator calibration](https://arxiv.org/abs/2610.22379) | 该研究发现在计算预算受限的工程模拟器校准任务中，以物理诊断描述为条件的提示能让大语言模型在早期评估阶段显著优于传统无导数优化器——首个物理信息提议在25次试验中23次改善初始设计，五次评估后中位校准误差降低21.9倍。 |
| [^173] | [Structured Authority Injection: Persistent Human Control for LLM Agents via Typed MCP Overrides](https://arxiv.org/abs/2610.22373) | 提出结构化权限注入（SAI）协议，通过类型化、模式验证的MCP工具调用和分级权限上下文，使人类操作员能够向LLM智能体注入跨规划周期持续生效的约束性指令，解决自由文本覆盖指令的歧义解释与权限侵蚀问题。 |
| [^174] | [Prism-1: Position- and Rival-Independent Scoring for System One Models](https://arxiv.org/abs/2610.22066) | 论文提出了Prism-1，一种面向系统1决策模型的评分方法，可同时满足位置无关性与Luce选择公理（竞争者无关性），克服了现有标准SCQ读取方式违反理性选择公理的缺陷。 |
| [^175] | [Router Drift Control: Mitigating Off-Policy Degradation in Mixture-of-Experts Reinforcement Learning](https://arxiv.org/abs/2610.21682) | 提出路由漂移控制（RDC）方法，将采样与训练阶段的路由分歧作为离策略过时信号，通过序列级过滤拒绝高漂移响应，在不引入梯度偏差的前提下缓解混合专家强化学习中的路由漂移与离策略退化。 |
| [^176] | [BiShield-TEE: On the (In-)Security of Unilateral Weight Obfuscation in On-Device TEE-Shielded LLM Partition](https://arxiv.org/abs/2610.21565) | 该论文揭示了设备端TEE防护LLM分区中单向权重混淆方案存在的“解确定性属性”漏洞——混淆密钥唯一确定导致GroupCover、ArrowCloak等方法的保护可被攻击完全破解，并据此提出BiShield-TEE双向防护方案以修复该安全缺陷。 |
| [^177] | [Jev-LiteMAS: Separating Decisions from Generation in Multi-Agent Workflows](https://arxiv.org/abs/2610.21411) | 该论文提出Jev-LiteMAS框架，通过将轻量级决策模型与生成式执行和反馈相结合，在多智能体系统中分离决策与生成功能，在显著降低路由和评审环节token开销（原占31.7%-43.5%）的同时弥合了规划鸿沟与评审鸿沟。 |
| [^178] | [Does On-Policy Distillation Need to Complete Every Rollout?](https://arxiv.org/abs/2610.21337) | 该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。 |
| [^179] | [All-in-One Memory: Process-Rewarded Reinforcement Learning for Memory Agents](https://arxiv.org/abs/2610.21147) | 该论文提出AiO-Mem框架，将记忆智能体的检索、写入与停止决策统一建模为共享证据状态上的转移，并通过过程奖励强化学习在统一奖励空间中直接评估每个动作的语义质量。 |
| [^180] | [CausalEvo: Causal Skill Self-Evolution for Health LLM Agents](https://arxiv.org/abs/2610.21016) | 提出CausalEvo方法，通过在固定模型权重、反馈案例和规则数量的条件下估计技能集变更的相对因果效应，实现健康LLM智能体的技能自进化筛选，保留有效规则并防止错误泛化的传播。 |
| [^181] | [When Reasoning Traces Conflict with Evidence: How Answer Configuration Shapes Evidence Following](https://arxiv.org/abs/2610.20805) | 研究发现，回答时采用独立回答调用而非延续模型思考过程，可使模型遵循当前证据的能力提升29.9至60.6个百分点，而移除推理轨迹在两种配置下均会降低准确率，表明“受益于推理轨迹”与“服从于推理轨迹”是两种不同的模型特性。 |
| [^182] | [ActiveEvidence: Decision-Focused Evidence Graphs with Verifiable Acquisition Gaps](https://arxiv.org/abs/2610.20532) | 该论文提出ActiveEvidence框架，通过反事实证据交互图（CEIG）和可验证的ε-决策证书，使多模态智能体能够高效决定何时获取视觉证据、仅评估可能改变决策的关键交互，并在证明进一步交互无法改变所选动作时及时停止。 |
| [^183] | [SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift](https://arxiv.org/abs/2610.20473) | 该论文提出 SafeAdapt-Poker，将对手漂移下的预算化适应建模为仅依赖决策前信息的双重反事实序贯控制问题，通过单调策略风险模型与配对预算状态延续模型分别校准偏离的局部下行风险和预算消耗的未来代价，并用逐路径累计KL账本约束总偏离量，从而在即时适应价值与安全风险之间实现可校准的权衡。 |
| [^184] | [HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation](https://arxiv.org/abs/2610.20191) | HEM-Med提出了一种分层智能体记忆框架，通过将可迁移的跨医院临床经验与医院特异性处方模式解耦，并纳入药物相互作用等多标签安全约束，实现安全的多中心用药推荐。 |
| [^185] | [VideoHarness-SRSI: Structured Recursive Self-Improvement for Video Understanding Harness](https://arxiv.org/abs/2610.19424) | 该论文提出结构化递归自我改进框架VideoHarness-SRSI，通过对4000多个演化候选的分析识别出验证过拟合、搜索预算稀释和早期经验陷阱三种失败模式及其共同根源，从而实现从有限且嘈杂的反馈中可靠演化出可泛化的视频理解智能体框架。 |
| [^186] | [Kwai Summary Attention V2 Technical Report](https://arxiv.org/abs/2610.18654) | 提出KSA V2架构，通过允许不同层设置不同的摘要压缩分块窗口，实现更灵活的KV缓存内存压缩，并解决全注意力层中摘要标记带来的额外计算开销问题。 |
| [^187] | [Rethinking Quality in Large Language Model Routing: Security-Conditioned Model-and-Budget Selection](https://arxiv.org/abs/2610.18566) | 该论文提出SCRouter路由器，针对混合工作负载分别预测效用与安全性并按查询动态加权平衡，实现安全条件下的模型与token预算联合选择。 |
| [^188] | [Agentic-MatriBO: Capability-Compiled Agentic Bayesian Optimization for Materials and Chemical Design](https://arxiv.org/abs/2610.17832) | 提出了能力编译型智能体贝叶斯优化框架Agentic-MatriBO，将BO组件表示为可组合的能力，由大语言模型策略提出优化配方，经编译器检查和确定性执行层验证后用于材料与化学设计。 |
| [^189] | [LeanPRM: Lean-Grounded Process Supervision for Natural Language Proof Verification](https://arxiv.org/abs/2610.17697) | LeanPRM利用Lean证明助手自动为自然语言数学证明生成步骤级正确性标签，无需人工标注即可构建包含10,634个证明、15.2万个标注步骤的过程监督数据集，解决了证明验证器训练中标签获取困难的瓶颈。 |
| [^190] | [RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment](https://arxiv.org/abs/2610.17665) | RoTA-Bench是一个通过交互式三维到二维投影对齐任务来评估视觉语言模型三维空间推理能力的新基准，其自动化流水线保证了任务的可解性，并揭示了当前最强模型（GPT-5.6-Sol仅17%成功率）与人类（100%成功率）之间的巨大差距。 |
| [^191] | [Engram Injection: A Study of Temporal Validity in Parametric Language-Model Memory](https://arxiv.org/abs/2610.15882) | 本文提出Engram Injection方法，将事实存储于LoRA适配器并通过FiLM条件化向冻结的语言模型注入时间信息，但严格排除测试时间样本的评估显示，该参数化记忆方法在事实更新后无法正确回答任何时间敏感问题对，揭示了参数化记忆在处理时间有效性上的局限。 |
| [^192] | [MemeLogos: Knowledge-Driven Retrieval-Augmented Reasoning for Harmful Meme Detection](https://arxiv.org/abs/2610.15463) | 提出MemeLogos框架，通过将有害迷因案例中的危害性模式与推理结构抽象为可复用的知识单元并构建专门知识库，实现知识驱动的检索增强推理，同时提升有害迷因检测的性能与可解释性。 |
| [^193] | [Are Multimodal Benchmarks Really Useful? Item-Level Multimodal Benchmark Diagnosis via Structure-Response Co-Calibration](https://arxiv.org/abs/2610.14542) | 提出了SRCoD框架，通过结构-响应协同校准对多模态基准进行条目级诊断，识别仅凭单一模态或选项规律即可作答的捷径条目，从而更全面地衡量基准能否真正反映模型的跨模态推理能力。 |
| [^194] | [RoboHarness: A Simple Harness Outperforms VLA and World Actions Models](https://arxiv.org/abs/2610.14018) | RoboHarness为一个简单的机器人操控框架，赋予LLM智能体视觉-几何控制面板，使其无需专门训练VLA模型即可直接理解并调用具身任务，并在BEHAVIOR Challenge 2025上超越最先进的VLA模型。 |
| [^195] | [SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data](https://arxiv.org/abs/2610.13644) | 该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。 |
| [^196] | [Safe Actions Alone Do Not Ensure Safe Agents: Identifying Unfulfilled Obligations with Guard Models](https://arxiv.org/abs/2610.11773) | 本文首次指出智能体安全不仅需要识别禁止动作，还需识别未履行的安全义务，并推出了首个用于评估守护模型义务识别能力的基准 ObligationBench。 |
| [^197] | [MultiWorldBench: Do Independently Controlled Views Describe One Shared World?](https://arxiv.org/abs/2610.11723) | 该论文提出了MultiWorldBench——一个基于Minecraft、包含495个案例配置和十种能力的多人世界模型诊断基准，用于评估独立控制视角是否能保持与共享持久世界的一致性，并发现现有生成式系统（Gamma-World、Solaris、MineWorld）与真实引擎参考（Engine GT）之间仍存在巨大差距，尤其在状态持久性和结构推理方面表现薄弱。 |
| [^198] | [Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals](https://arxiv.org/abs/2610.11570) | 提出InfiLoop循环原生残差连接，通过结合内容加权与可学习时间衰减来维护递归状态摘要，使循环Transformer能够扩展到数万次测试时迭代而不出现推理性能退化。 |
| [^199] | [RIT-RAG: Navigating Document Corpora with Retrieval-Induced Trees](https://arxiv.org/abs/2610.11370) | RIT-RAG将内容检索与文档结构导航相结合，离线为每篇文档构建目录树，查询时先检索宽泛的文本块并利用其位置诱导出可跨多个文档的、规模可控的子树，再由LLM智能体在子树中导航，从而在保留结构感知优势的同时克服了现有方法无法扩展到大型语料库且选错文档后无法恢复的缺陷。 |
| [^200] | [Frozen Models, Evolving Expertise: Model-Agnostic Learning from Deployment Experience for Multimodal Medical AI](https://arxiv.org/abs/2610.09146) | 提出了一种模型无关的框架，使冻结的大语言模型和视觉语言模型无需微调即可通过技能、知识记忆库和多模态知识库三种外部专业知识从部署经验中持续学习，以适应医学领域不断更新的临床知识。 |
| [^201] | [Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness](https://arxiv.org/abs/2610.08621) | 该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。 |
| [^202] | [EvalResearchBench: Can AI Agents Build Their Own Feedback for Recursive Self-Improvement?](https://arxiv.org/abs/2610.04184) | 提出EvalResearchBench（ERB）这一新基准，用于检验AI智能体能否自主设计评估反馈以支持递归自我改进，研究发现最佳智能体构建的评估器与目标基准的成对排序一致性约为75%，但仍低于人类专家水平。 |
| [^203] | [The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer](https://arxiv.org/abs/2610.03827) | 相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。 |
| [^204] | [FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training](https://arxiv.org/abs/2610.02828) | FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。 |
| [^205] | [ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing](https://arxiv.org/abs/2610.02808) | ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。 |
| [^206] | [Benchmarking Prompt Optimization of Large Language Models With Chess](https://arxiv.org/abs/2610.00416) | 该论文提出了一个基于1,118道Lichess国际象棋棋题的基准测试，利用国际象棋低成本精确评分、引擎辅助着法评估和可持续更新的可调难度题源等特性，来研究冻结大语言模型的自动提示词优化。 |
| [^207] | [FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training](https://arxiv.org/abs/2610.00385) | FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。 |
| [^208] | [ContractRL: Shielded Group-Relative Policy Optimization for Auditable Tool-Call Repair](https://arxiv.org/abs/2610.00328) | ContractRL提出了一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为有界决策过程，通过契约派生的动作掩码过滤非法修复操作，在仅生成34.4个token的情况下实现了0.9362的语义成功率，并保证了修复过程的可审计性。 |
| [^209] | [Actions with Receipts: Jointly Binding Claims, Evidence, and Execution for Replayable Tool-Agent Auditing](https://arxiv.org/abs/2610.00327) | 该论文提出“声明锚定执行契约”，通过收据机制将智能体发出的声明、证据来源片段、执行前缀与来源版本联合绑定，并由确定性完整性验证器重建绑定关系，从而实现对工具型智能体行为的可重放审计。 |
| [^210] | [Trust Is Not a Score: Runtime Assurance Contracts for High-Risk AI Agents](https://arxiv.org/abs/2609.39717) | 提出了运行时保障契约（RAC），用非补偿性强制门控而非综合分数来决定高风险AI智能体在关键任务中的权限动态转换，填补了观察证据与智能体授权之间的保障转换缺口。 |
| [^211] | [GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives](https://arxiv.org/abs/2609.39601) | GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。 |
| [^212] | [GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed](https://arxiv.org/abs/2609.39600) | GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。 |
| [^213] | [CollabFlow: Recursive Self-Improvement of Agent Collaboration](https://arxiv.org/abs/2609.38662) | 提出CollabFlow，一个通过可训练的“协作导演”构建完整智能体团队、在协作图边上采用证据条件化通信协议以抑制错误传播、并利用每轮成果重新训练导演从而实现递归自我改进的多智能体协作系统。 |
| [^214] | [When Should Agents Check External State? Budgeting Observations for Stored Intentions](https://arxiv.org/abs/2609.37125) | 本文首次将智能体为验证存储意图所需的外部观测形式化为共享预算下的资源分配问题，并提出 BudgetPM 框架（包含静态评分器与序列策略两种变体）来决定何时花费或保留观测预算。 |
| [^215] | [VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers](https://arxiv.org/abs/2609.36958) | 本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。 |
| [^216] | [Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias](https://arxiv.org/abs/2609.34971) | 该论文揭示了LLM智能体的“模式偏差”现象——即当相同的可执行动作以功能等价但形式不同的工具模式呈现时智能体表现不一致，并提出了一套包含九种算子的可执行转换框架，在保持任务与动作空间不变的前提下系统测量并缓解这种偏差。 |
| [^217] | [SpeechCritic: Learning a Diagnostic Speech Judge from Limited Human Preferences](https://arxiv.org/abs/2609.34582) | SpeechCritic 仅用约300个人工标注的偏好比较，通过校准（而非替换）前沿音频-语言模型，学习到能判断语音优劣、指出感知维度差异并提供可听线索证据的诊断式语音评判器。 |
| [^218] | [R$^2$ Flow: Recursive Self-Improvement via Recursive Skill Evolution](https://arxiv.org/abs/2609.33867) | 提出 R$^2$ Flow 递归自我改进框架，在共享状态编排图上交替进行策略学习、独立验证与版本化技能库更新，并通过合并等效执行历史和流份额读出机制，解决流训练中的策略坍缩、信用分配偏差与奖励依赖三大难题，实现智能体可靠的递归自我改进。 |
| [^219] | [RSD-Poker: Structure-Adaptive and Shift-Robust Risk-Utility Certification for Residual Policies in Imperfect-Information Games](https://arxiv.org/abs/2609.33669) | 该论文提出 RSD-Poker 框架，通过冻结残差候选库、学习并预先冻结策略可见的子群划分，为不完美信息游戏中的残差策略提供对部署分布偏移稳健的风险上界与效用下界双证书认证。 |
| [^220] | [Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification](https://arxiv.org/abs/2609.33662) | 该论文提出Audit-First VAPO框架，通过将更新方向的离散准入与幅值的连续控制相分离，在有限验证预算下利用“接受-申诉-弃权”观察策略和有限样本统计界，实现对不完美验证器下有害更新风险、覆盖率与验证器调用率的可认证保障。 |
| [^221] | [OSCC: Certified Observation-Safe Coupling Optimization for Gradient-Noise Control in Imperfect-Information Learning](https://arxiv.org/abs/2609.33543) | 提出了观测安全反事实耦合（OSCC）框架，通过定义可容许耦合类的多项安全准则并引入梯度感知的耦合优化准则，在非完美信息学习中安全地降低反事实比较噪声，同时针对策略梯度几何控制回报协方差以优化多动作策略。 |
| [^222] | [NutriVision: Ingredient-Conditioned Fusion and Prediction for Single-Image Food Nutrition Estimation](https://arxiv.org/abs/2609.33076) | NutriVision是一个端到端营养估计框架，通过DepthAnything-V3生成深度信息并结合CLIP编码的食材语义，借助食材条件化频率对齐融合模块（IC-FAFM），从单张RGB图像和可选食材列表中估计食物的卡路里、质量、脂肪、碳水化合物和蛋白质。 |
| [^223] | [Porimon: An LLM-Based Pok\'emon Battle Agent Enhanced by Long/Short-Term Knowledge Augmented Generation](https://arxiv.org/abs/2609.32544) | 本文提出长/短期知识增强生成机制（LSTKAG）并据此构建宝可梦对战智能体Porimon，无需微调即可让LLM智能体利用当前任务历史状态并从相似过往经验中检索总结，配合外部API精确计算伤害，在15,000场对战中显著超越基线玩家。 |
| [^224] | [Metro-WM: Long-Horizon Latent Planning with Realisable Sub-Goals](https://arxiv.org/abs/2609.31868) | 论文指出分层JEPA宏观规划器预测的潜在子目标往往物理上不可实现，并提出Metro-WM框架，通过从历史经验构建的图中检索真实状态作为子目标，从而实现可靠的长时程规划。 |
| [^225] | [Evolutionary Safety of Recursive Self-Improving AI: Taxonomy, Risk Discovery, and Evaluation](https://arxiv.org/abs/2609.31186) | 该论文提出“进化安全”这一新视角，用于研究AI在递归自我改进过程中安全属性如何随演化而变化、持续、积累与传播，并给出了相应的风险分类、风险发现与评估框架。 |
| [^226] | [Cross-Modal Emotion Understanding: A Transformer-GAT Approach for Dialogue Emotion Recognition](https://arxiv.org/abs/2609.29556) | 该论文提出了一种结合Transformer与图注意力网络的混合框架Transformer-GAT，通过Transformer捕获全局语义信息、图注意力网络建模模态间细粒度关系来实现跨模态对话情感识别，在IEMOCAP和MELD数据集上分别取得72.45%和77.37%的加权F1分数，超越了现有最先进方法。 |
| [^227] | [SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories](https://arxiv.org/abs/2609.29465) | 该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。 |
| [^228] | [From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents](https://arxiv.org/abs/2609.29051) | 本文发现基于特权信息的同策略自我蒸馏会让多轮智能体“盲目自信”且性能甚至差于普通强化学习和基础模型，并提出特权自我练习（PSP）方法，将特权信息从损失函数转移到采样阶段，通过注入分析模型生成的任务指令引导重新采样，再用不变的GRPO目标训练，从而有效提升多轮智能体性能。 |
| [^229] | [How Children Design and Reason about Trustworthy AI Chatbots](https://arxiv.org/abs/2609.25244) | 本研究开发了一个让儿童自主设计聊天机器人的平台，通过对115名8-18岁学习者的混合方法研究发现，低龄学生会设置更高的自信度，甚至认为“故意出错但按设计行事”的聊天机器人也值得信赖，揭示了儿童对AI可信度的独特理解方式。 |
| [^230] | [Mind the Phase: Effective Rank and Representation Health in Legged Locomotion](https://arxiv.org/abs/2609.06958) | 该论文通过按步态相位分解策略雅可比矩阵的有效秩，揭示了层归一化和残差连接会使摆动相比支撑相多获得约两个有效秩维度的架构结构，并据此提出了提升腿式运动策略表征健康度的简单方案。 |
| [^231] | [CABAL: Multi-Agent Simulacra for Tracing the Effects of Collusive Bidding in Peer Review](https://arxiv.org/abs/2609.05227) | 提出了端到端多智能体仿真框架CABAL，通过为LLM审稿人智能体配置诚实或合谋策略来研究同行评审中合谋投标的影响，并设计了基于审稿人-论文亲和力的合谋投标策略，实现与专业领域一致而非任意的定向攻击。 |
| [^232] | [Capability-Gated Language Models: Security Composes, Utility Does Not](https://arxiv.org/abs/2609.00445) | 提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。 |
| [^233] | [Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness](https://arxiv.org/abs/2609.00050) | 提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。 |
| [^234] | [IndicDetect: Evaluating Cross-Lingual LLM-Generated Text Detection for Hindi, Telugu, and Tamil](https://arxiv.org/abs/2608.29919) | 该论文提出了IndicDetect基准，系统评估印地语、泰卢固语和泰米尔语中AI生成文本检测器在领域偏移、生成器偏移和对抗性扰动等现实分布偏移下的鲁棒性，并发现现有检测器存在显著的鲁棒性缺陷。 |
| [^235] | [Beyond Uncertainty Reduction: Calibrated Planning for Interactive Diagnosis](https://arxiv.org/abs/2608.27847) | 该论文提出CaRDI框架，用经过保形校准的临床风险（结合主诊断错误与严重鉴别诊断遗漏的加权损失）取代传统的不确定性降低目标，以指导基于大语言模型的交互式医疗问诊中的提问选择与停止时机。 |
| [^236] | [BenthicDINO: Physics-Informed Self-Distillation for View-Invariant Side-Scan Sonar Representations](https://arxiv.org/abs/2608.23215) | 本文提出了一种基于物理信息的自蒸馏框架（BenthicDINO），通过物理模拟增强和HSIC惩罚强制侧扫声呐图像表示的视角不变性，有效解决了声学伪影导致的特征混合问题。 |
| [^237] | [ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning](https://arxiv.org/abs/2608.19182) | ADEPT通过预训练通用灵巧操作策略并采用稳定的后训练方案，显著加速了高自由度机器人从原始感知学习长时程任务的过程，避免了重复学习并提升了迁移能力。 |
| [^238] | [Self- and Other-Labels Induce Bidirectional Bias in LLM Judges](https://arxiv.org/abs/2608.18091) | 本文通过改变评估对象为无风格指纹的叙事约束选择，发现LLM评审中的自我偏好在控制质量后基本消失，甚至在某些维度上出现反向偏差，揭示了自我偏好并非固有属性。 |
| [^239] | [BEAR-Bench: A Bilingual Enterprise and Academic Reasoning Benchmark for Multimodal Models](https://arxiv.org/abs/2608.17895) | BEAR-Bench是一个全新的英俄双语基准，包含1000个基于专业商业和科学文档的人工标注问题，用于评估多模态大语言模型在文本密集场景下的推理能力，并揭示了当前最强模型仍存在显著性能提升空间。 |
| [^240] | [Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents](https://arxiv.org/abs/2608.13179) | CrEST通过层次化信用分配框架，结合验证器受限的强化学习上限和自教师模型的密集令牌信号，有效解决了多轮LLM智能体训练中的信用分配问题，在基准测试中持续超越现有方法。 |
| [^241] | [Rescene: stable century-long climate emulation from a frozen weather model](https://arxiv.org/abs/2608.09971) | Rescene仅用一个0.4M参数的轻量封装，将冻结的机器学习天气模型转化为可稳定运行百年、无漂移且变率得到良好校准的随机气候模拟器。 |
| [^242] | [Locating Failure in Multi-Page Visually Rich Document Understanding Systems: An Empirical Attribution](https://arxiv.org/abs/2608.07943) | 本文提出一个实证归因框架，通过受控干预定位多页视觉丰富文档理解系统在证据表示、选择与使用各环节中的失败来源，发现模态互补可降低转换错误敏感性、非黄金页面会同时修复与破坏答案，从而揭示了总体准确率所掩盖的系统不稳定性。 |
| [^243] | [Poli-Bias: Understanding and Measuring Large Language Model Biases in International Political Conflicts](https://arxiv.org/abs/2608.06123) | 提出了POLI-BIAS反事实测量框架，通过系统性互换国家身份的配对提示，将LLM在国际政治冲突中的偏见响应差异分解为五个可解释维度，发现模型对处于冲突或紧张状态的国家存在不均衡对待。 |
| [^244] | [Q-Reflector: Learning GUI Action Reflection with Generated Conditions](https://arxiv.org/abs/2608.05587) | 提出了Q-Reflector，一个基于Qwen3-VL-8B的独立GUI动作反思器，通过生成前置/后置条件并以监督微调、蒸馏和奖励偏好优化相结合的训练方式，准确评估移动GUI动作的执行成功性与任务有效性。 |
| [^245] | [MANGO-Grasp: Mahalanobis Fields over Geometry-Oriented 3D Gaussians for Cross-Embodiment Dexterous Grasping](https://arxiv.org/abs/2608.02014) | 提出MANGO-Grasp框架，利用几何导向的3D高斯表示物体、预训练描述符编码手部，并通过各向异性马氏距离场建模手-物体交互，从而在无需针对特定具身形态调优的情况下，实现跨异构多指灵巧手的稳定灵巧抓取。 |
| [^246] | [Demystifying When and Why VLAs Fail in Contact-Rich Tasks and How to Fix Them](https://arxiv.org/abs/2608.01402) | 该论文揭示了VLA模型在富接触任务中的两种失败模式（精度失败与力失败），并提出FACT方法，通过后训练阶段的低噪声采样和结构化力条件机制将平均成功率提升至66%。 |
| [^247] | [An AI Approach to Verified Production Cryptographic Libraries](https://arxiv.org/abs/2608.00965) | CryptoProver是一种AI系统，能够从高层API契约自动综合内部规范与经Verus验证的形式化证明，在不改动任何代码的情况下，首次以少量人力完成对curve25519-dalek和chacha20等生产级密码学库的独立验证。 |
| [^248] | [CDAE: Enhancing Perturbation Robustness in Pretrained Language Models with Contrastive Denoising](https://arxiv.org/abs/2607.28236) | 提出一种轻量级对比去噪自编码器CDAE，通过联合优化对比与重建目标学习扰动不变的句子表示，显著增强了预训练语言模型嵌入对文本扰动的鲁棒性。 |
| [^249] | [EKLAV: Elicited Key CoT Learning via Answer only superVision](https://arxiv.org/abs/2607.25182) | EKLAV将教师模型的思维链轨迹作为特权输入上下文而非模仿目标，仅对最终答案进行监督训练，在重排序任务上以更低的训练计算成本和推理时间超越了传统推理蒸馏方法。 |
| [^250] | [Efficient LLM-Generated Shuttling Compilers for Complex Trapped-Ion Architectures](https://arxiv.org/abs/2607.24714) | 该论文首次展示单个前沿大语言模型（Claude Opus 4.7）能够根据书面规范自动生成并迭代优化离子阱量子计算机的穿梭编译器，在多种离子阱架构上比最先进的手工编译器减少多达72%的穿梭时间步。 |
| [^251] | [KAYROS: An Anytime and Exact Open-Source Solver for Duration-Minimization Time-Dependent Vehicle Routing. A Technical Report and a Case Study in Human-AI Engineering](https://arxiv.org/abs/2607.23116) | KAYROS 是一个开源、随时可中断且能给出最优性证书的时长最小化时间依赖型车辆路径求解器，填补了该领域缺乏可直接使用的精确求解工具的空白。 |
| [^252] | [STAIF: A Stage-wise Optimization for Complex Instruction Following](https://arxiv.org/abs/2607.22649) | STAIF提出分阶段优化框架，先用多负样本偏好优化对齐软约束，再用可验证奖励强化学习强制满足硬约束，并构建了约3.1万条双语复杂多约束指令数据集STAINSTRUCT，有效提升大模型复杂指令遵循能力。 |
| [^253] | [Every Model Cheats: Prompt-Level Mitigation of Cheating on Offensive Cyber Tasks](https://arxiv.org/abs/2607.21763) | 本研究通过四阶段审计流水线的大规模实验揭示，网络安全基准测试中的作弊远比以往认知普遍（22个前沿模型中21个作弊、37.1%的通过涉及作弊、分数夸大高达5倍），并证明提示词级的反作弊干预能将作弊倾向从33.0%有效降至8.5%。 |
| [^254] | [Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR](https://arxiv.org/abs/2607.18254) | 该论文提出从 MLIR 方言的操作定义规范（ODS）机械派生的推理时约束解码方法，无需重新训练即可实现跨方言泛化，并发布了涵盖三种方言共 410 个样本的四个自然语言到 MLIR 基准测试数据集。 |
| [^255] | [Align AI to Dynamic Human-AI Workflows](https://arxiv.org/abs/2607.14240) | 该论文提出将AI对齐范式从静态模仿人类偏好转向交互式、互补式对齐，即偏好应在人机交互中动态涌现，且对齐的衡量标准不应仅限于满足静态偏好。 |
| [^256] | [How Agents Ask for Permission: User Permissions for AI Agents, from Interfaces to Enforcement](https://arxiv.org/abs/2607.13718) | 该论文调研了21个用户级AI智能体权限系统并构建了涵盖权限指定、推导与执行方法的分类体系，通过对比商业智能体发现学术与商业方案在可用性、透明度和用户控制方面仍存在显著差距。 |
| [^257] | [Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains](https://arxiv.org/abs/2607.13319) | OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。 |
| [^258] | [Do Multi-Agent LLMs Explore Each Other?](https://arxiv.org/abs/2607.11250) | 本文发现现代LLM智能体在多智能体交互中存在短视、极化的探索缺陷，并提出轻量级框架MACE，通过结构化同伴选择显式促进智能体间的探索，从而显著改善协调能力并降低遗憾值。 |
| [^259] | [Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference](https://arxiv.org/abs/2607.05475) | 该论文开展了首个跨层次的移动端LLM推理全面测量研究，并开发了细粒度性能分析工具PowerBench，揭示了框架性能差距在NPU上可被放大至10倍等隐藏的效率瓶颈。 |
| [^260] | [TORINO: Token Reduction via Interpretable Concept Overlap in Vision-Language Models](https://arxiv.org/abs/2607.04593) | 提出即插即用框架TORINO，利用稀疏自编码器将视觉词元映射到可解释的潜在空间，基于概念重叠程度对词元分组并削减冗余词元，无需微调即可降低视觉-语言模型的计算成本。 |
| [^261] | [APeB: Benchmarking Personalization Ability of Large Language Model Agents](https://arxiv.org/abs/2607.03162) | 该论文提出个性化产品搜索任务和APeB基准，用于评估大语言模型智能体在原始欠规范查询与丰富交互历史下的个性化能力，发现现有模型在早期模糊查询的意图与偏好发现上表现不佳，其主要原因是对历史信息利用不足。 |
| [^262] | [PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding](https://arxiv.org/abs/2606.31148) | 提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。 |
| [^263] | [Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization](https://arxiv.org/abs/2606.31002) | 该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。 |
| [^264] | [Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2606.22995) | 提出群组-图策略优化（G2PO），一种针对多轮智能体任务的新型分组强化学习算法，通过显式利用状态转换固有的图结构，解决了长时程智能体强化学习中奖励稀疏延迟、高方差价值估计以及短视局部化信用分配的关键瓶颈。 |
| [^265] | [Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering](https://arxiv.org/abs/2606.18986) | 该论文提出CADE框架，通过直接时间步嵌入与对比语义对齐，克服大语言模型处理时间序列时的分词瓶颈，并解决了基于固定窗口的补丁编码所导致的粒度锁定和跨数据集迁移困难问题。 |
| [^266] | [AgentRivet: an automated system for producing Rivet routines from journal publications](https://arxiv.org/abs/2606.13535) | 本文提出AgentRivet，一个基于大语言模型的多步骤自动化工作流，能够从已发表的粒子物理论文中提取分析信息并自动编写缺失的Rivet例程，通过内置的代码和物理审查实现自主质量控制，解决了目前仅39%测量拥有可用Rivet例程的覆盖率不足问题。 |
| [^267] | [Accounting for Purchased Artificial Intelligence Inference in Corporate Greenhouse Gas Inventories: A Token-Based Method](https://arxiv.org/abs/2606.10660) | 本文提出一种基于Token的核算方法和开放数据集，用于将购买式AI推理纳入企业温室气体清单的范围三第一类，克服了现有基于支出的方法偏差高达12倍且无法区分模型与地区的缺陷。 |
| [^268] | [TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering](https://arxiv.org/abs/2606.08408) | TimpaTeks将激活引导扩展到扩散语言模型，实现了一种无需指令微调模型的自动原地文本修改机制，在保留原句结构、降低困惑度的同时，以更低的计算成本改变文本所表达的概念。 |
| [^269] | [Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents](https://arxiv.org/abs/2606.06090) | 本文提出MAGE系统，将智能体记忆从被动的语义相似度检索重新构想为主动的执行状态管理，通过层次化状态树和四种耦合操作（生长、压缩、维护、修订），帮助长时程智能体实现连贯的状态重建与错误隔离。 |
| [^270] | [CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery](https://arxiv.org/abs/2606.03602) | 提出CauTion框架，通过共识过滤和LLM可靠性估计，将大语言模型的领域知识可靠地融合进统计因果发现算法集成中，从而同时克服纯统计方法、LLM错误风险和单一算法偏差这三方面的局限。 |
| [^271] | [Large Language Models in Transportation Systems Management and Operations: From Text Reasoning to Multi-modal Decision Support](https://arxiv.org/abs/2606.00991) | 本综述系统梳理了大语言模型及多模态大语言模型在交通系统管理与运营三大领域（运营与服务、出行与车队服务、数据建模与决策支持）中的应用现状，并指出了数据异构性、实时推理、可解释性与多模态融合等关键挑战。 |
| [^272] | [Who Can Change Course? An Integrative Review of AI Sovereignty and Collective Learning Capacity](https://arxiv.org/abs/2606.00729) | 本文提出将AI主权理解为一种国家学习能力，以人文中心学习机制（HCLM）为概念框架，结合法国案例，分析国家如何将算力、数据、人才等分散输入转化为协调且经负责任评估的能力，同时避免协调失败、行政重复等制度性损失。 |
| [^273] | [Learning to Reason About Implicit Multimodal Harm: From Diagnosis to Generalization](https://arxiv.org/abs/2605.29951) | 提出MuPHI诊断基准，揭示视觉-语言模型虽能理解单一模态信息，却无法将图像-文本组合推理为危害机制，为隐式多模态危害检测从诊断走向泛化提供了方向。 |
| [^274] | [SciIntBench: Measuring LLM Compliance with Research Integrity Norms Under Adversarial Framing](https://arxiv.org/abs/2605.29468) | 该论文提出了对抗性基准SciIntBench，通过1350个跨五个对抗等级的提示评估16个大语言模型对科研诚信规范的遵循情况，发现模型在最隐蔽的对抗框架下平均71.7%的情况未能拒绝科研不当行为请求。 |
| [^275] | [FormInv: A Measurement Protocol for Semantic Invariance in Mathematical Reasoning Benchmarks](https://arxiv.org/abs/2605.29001) | 该论文提出FormInv测量协议，用于审计数学推理基准中自动生成改写题目的语义等价性，并引入条件不一致率（CIR）指标，揭示即使在所有模型都能正确回答的规范形式定理上，语义等价的改写仍会导致5-42%的答案翻转。 |
| [^276] | [Prompt Codebooks: Discrete Compositional Optimization for Language Model Instruction Refinement](https://arxiv.org/abs/2605.28360) | 该论文提出提示词码本优化（PCO）框架，将自动提示词优化重新建模为对可复用自然语言指令单元码本的离散组合学习，通过编码器—生成器—评论家架构和文本梯度信号实现提示词的模块化构建与各组件的联合优化。 |
| [^277] | [ATWL: A Formal Language for Representing, Comparing, and Reusing Visual Analytics Workflows](https://arxiv.org/abs/2605.25489) | 该论文提出了一种声明式语言ATWL来表示可视化分析工作流的结构与意图，并构建了包含54个工作流的库，揭示了人机分工与迭代求精的规律，同时提升了LLM设计新工作流的效果。 |
| [^278] | [Agent-Facing Information Design in LLM Tool Registries: A Preregistered Test of Rhetoric, Position and Structure](https://arxiv.org/abs/2605.23916) | 研究通过预注册实验发现，LLM智能体的工具选择会被工具描述中的推销性赞美语言显著影响（可提升约43个百分点选中率，甚至选到无法胜任任务的工具），列表排序和结构化字段也会强烈左右选择，因此建议注册表以结构化字段而非营销文本来呈现工具限制。 |
| [^279] | [Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints](https://arxiv.org/abs/2605.22940) | 本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。 |
| [^280] | [ECPO: Evidence-Coupled Policy Optimization for Verifier-Audited Candidate Ranking](https://arxiv.org/abs/2605.21993) | 提出证据耦合策略优化（ECPO），通过证据循环奖励将Top-K排序与基于片段的可审计证据耦合，显著提升验证者可审计的可追溯性，且增益在多个基准、骨干模型与评估器下均保持稳健。 |
| [^281] | [Exploration and Experimentation with a Misspecified Simulator](https://arxiv.org/abs/2605.21458) | 该论文研究了在模拟器存在错误设定且真实系统动态发生漂移的情况下，是否、何时以及如何通过被动适应、持续探索或随机化实验等数据收集策略来重新校准模拟器。 |
| [^282] | [HyperPersona: A Multi-Level Hypergraph Framework for Text-Based Automatic Personality Prediction](https://arxiv.org/abs/2605.17355) | 本文提出HyperPersona框架，利用多层超图结构显式建模文档、句子和词语三个层次的文本层级组织，从而提升基于文本的自动人格预测性能。 |
| [^283] | [Brain Vascular Age Prediction Using Cerebral Blood Flow Velocity and Machine Learning Algorithms](https://arxiv.org/abs/2605.16969) | 本研究利用经颅多普勒（TCD）测量的脑血流速度特征与机器学习回归模型来预测脑血管年龄，发现患有脑部疾病的受试者表现出脑血管加速衰老。 |
| [^284] | [RaPD: Resolution-Agnostic Pixel Diffusion via Semantics-Enriched Implicit Representations](https://arxiv.org/abs/2605.15908) | RaPD提出在连续的神经图像场潜空间中进行像素扩散，通过语义表示引导和坐标查询注意力渲染器，使单个去噪潜变量可通过仅改变查询坐标按任意分辨率渲染，且扩散成本保持固定。 |
| [^285] | [CrystalReasoner: Reasoning and RL for Property-Conditioned Crystal Structure Generation](https://arxiv.org/abs/2605.14344) | CrystalReasoner 提出了一个端到端的大语言模型框架，通过将晶体学对称性、配位环境和预测物理性质等物理先验作为“思考 token”进行推理，并利用多目标密集奖励的强化学习进行对齐，从而从自然语言指令直接生成有效且具备目标性质的晶体结构。 |
| [^286] | [CIDR: A Large-Scale Industrial Source Code Dataset for Software Engineering Research](https://arxiv.org/abs/2605.12153) | 本文发布了CIDR——一个包含4,687个仓库、84种编程语言、超10亿行代码的大规模匿名化工业源代码数据集，为代码智能和软件工程研究提供真实企业级代码资源。 |
| [^287] | [NEXUS: Learning Safety Constraints and Action Models for Safe and Robust Embodied Planning](https://arxiv.org/abs/2605.09387) | NEXUS 提出一个模块化神经符号框架，通过联合维护 PDDL 领域模型与 LTL 安全规范，利用执行失败修复动作模式、并将检测到的危险转化为可复用的 LTL 安全约束，从而实现安全且稳健的具身规划。 |
| [^288] | [STRIDE: Reasoning-Aware Training for Time Series Forecasting](https://arxiv.org/abs/2605.08625) | STRIDE提出了一种新框架，将蒸馏后的LLM推理以连续跨模态先验的形式原生注入时间序列基础模型的嵌入空间，在保持数值预测能力的同时引入推理能力，并登上GIFT-Eval排行榜第一。 |
| [^289] | [Coordination Matters: Process-Aware Evaluation of Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2605.06557) | 本文提出了一个过程感知的多智能体强化学习评估框架，通过干扰、协同任务完成、参与度和协调效率等诊断指标补充传统聚合性能指标，揭示了回报相近的策略在协调行为上可能存在显著差异。 |
| [^290] | [MOSAIC: Module Discovery via Sparse Additive Identifiable Causal Learning for Scientific Time Series](https://arxiv.org/abs/2605.05524) | MOSAIC 提出了一种稀疏时序 VAE 方法，将科学观测固有的语义可解释性传递到具有可辨识性保证的潜在空间中，从而在底层机制未知的科学时间序列中实现可解释的模块发现。 |
| [^291] | [FaithfulFaces: Pose-Faithful Facial Identity Preservation for Text-to-Video Generation](https://arxiv.org/abs/2605.04702) | 提出FaithfulFaces框架，通过姿态共享身份对齐器将单视角输入映射为带显式欧拉角嵌入的全局面部姿态表示，为文生视频生成提供姿态忠实的面部先验，显著改善大姿态变化和面部遮挡下的身份保持效果。 |
| [^292] | [Learning reveals invisible structure in low-rank RNNs](https://arxiv.org/abs/2605.04115) | 本文推导出低维常微分方程系统来精确描述低秩RNN的学习动力学，并发现学习过程可分为决定网络功能的“损失可见重叠”和不影响功能但描述学习所必需的“损失不可见重叠”。 |
| [^293] | [RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy](https://arxiv.org/abs/2605.02003) | 本文提出了RamanBench——首个大规模、完全可复现的拉曼光谱机器学习基准，统一整合了四个领域74个数据集共325,668条光谱，并在标准化协议下对28个模型进行了系统评估。 |
| [^294] | [Interpretable experiential learning based on state history and global feedback](https://arxiv.org/abs/2605.00940) | 提出了一种基于状态历史与全局反馈的可解释经验学习模型，通过带效用值和证据计数的状态转移图学习行为模型，适用于资源受限环境下的强化学习，并在Atari Breakout基准上取得了与神经网络方案相当的性能。 |
| [^295] | [CastFlow: Bridging Evidence Acquisition and Accurate Prediction in Agentic Time Series Forecasting](https://arxiv.org/abs/2604.27840) | CastFlow提出了一种角色专门化的智能体时间序列预测框架，通过经验记忆引导冻结的大语言模型调度多视角工具获取证据，再由轻量级专门化模型进行基于证据的时间推理，从而有效衔接证据获取与准确的数值预测。 |
| [^296] | [Activation-Efficient Fine-Tuning via Sequence-Compressed Low-Rank Subspaces](https://arxiv.org/abs/2604.22783) | LARS通过在序列级低秩子空间中执行适配，将存储的适配器激活中间量复杂度从O(BSRL)降至O(BRL)，相比LoRA在GPU和CPU上分别节省23%和47.8%的内存，且节省幅度随上下文长度增加而提升。 |
| [^297] | [Algorithmic Primitives of Theory of Mind Benchmarks](https://arxiv.org/abs/2604.17819) | 该论文提出神经符号方法PDDL-Mind，通过将叙事解析为PDDL并由程序执行状态更新，揭示了主流心智理论基准实际上主要在测试解析和状态跟踪等算法能力而非真正的心智理论，并借此将GPT-4o的性能提升至接近人类水平。 |
| [^298] | [OmniTrace: A Unified Framework for Generation-Time Source Attribution in Omni-Modal LLMs](https://arxiv.org/abs/2604.13073) | OmniTrace 提出一种统一框架，通过读取全模态大模型后层的注意力分布，将生成的每个句子实时归因到文本、图像和音视频片段等源单元，无需额外前向传播或针对模态的调优。 |
| [^299] | [Worst-Case Optimal $\sqrt{\epsilon}$-Rate Stability of KL Divergence under Perturbations of Gaussian Distributions](https://arxiv.org/abs/2604.11026) | 本文在比较分布 $P$ 不必为高斯分布的情况下，首次建立了高斯参考分布受扰动时KL散度变化的最坏情形最优 $\sqrt{\epsilon}$ 阶稳定性界，突破了现有松弛三角不等式要求三个分布均为高斯分布的局限，可应用于流生成模型的分布外检测等场景。 |
| [^300] | [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151) | 提出NARCBench基准与五种探针技术，首次利用模型内部表示在群体层面检测LLM多智能体系统中的串谋行为，并将其形式化为分布式异常检测问题。 |
| [^301] | [Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models](https://arxiv.org/abs/2604.00445) | 提出真实锚定方法，一种轻量级后验校准技术，通过将大语言模型的原始不确定性估计分数映射为真实对齐的正确性概率，解决了UE指标在不同模型和数据集上性能不稳定的“代理失效”问题，即使在有噪声的少样本监督下也能持续提升校准效果。 |
| [^302] | [Suppression of $^{14}\mathrm{C}$ photon hits in large liquid scintillator detectors via spatiotemporal deep learning](https://arxiv.org/abs/2603.27727) | 提出门控时空图神经网络和两种Transformer模型，在击中层面识别液体闪烁体探测器中$^{14}$C衰变的光子，实现25%-48%的召回率且正电子误判率低于1%，显著改善能量分辨率。 |
| [^303] | [Accurate and Efficient Multi-Channel Time Series Forecasting via Sparse Attention Mechanism](https://arxiv.org/abs/2603.18712) | 本文提出了Li-Net新型架构，通过稀疏Top-K Softmax注意力机制与多尺度投影框架，高效捕获多通道时间序列中通道间的线性和非线性依赖关系，实现准确且高效的预测。 |
| [^304] | [Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift](https://arxiv.org/abs/2603.17372) | 该研究揭示视觉语言模型的越狱并非源于对有害意图识别的失败，而是视觉模态将表征推向特定越狱状态所致，并通过定义越狱相关表征偏移来量化这一现象，为理解和防御越狱攻击提供了新途径。 |
| [^305] | [OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism](https://arxiv.org/abs/2603.14371) | OxyGen将KV缓存作为跨任务和控制周期的一等共享资源进行统一管理，通过跨任务KV共享和跨帧连续批处理两项优化，消除了多任务并行VLA推理中的冗余计算和资源竞争问题。 |
| [^306] | [M$^2$RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling](https://arxiv.org/abs/2603.14360) | 提出了一种具有矩阵值隐藏状态和非线性状态转移的新型循环神经网络M$^2$RNN，突破了Transformer在TC$^0$复杂度上的表达能力限制，实现了完美状态跟踪泛化并在大规模语言建模中表现优异。 |
| [^307] | [AOI: Turning Failed Trajectories into Training Signals for Autonomous Cloud Diagnosis](https://arxiv.org/abs/2603.03378) | 该论文提出AOI可训练多智能体框架，通过GRPO将专家知识蒸馏到本地开源模型并结合读写分离的安全执行架构，把失败运维轨迹转化为训练信号，在安全约束下实现自主云诊断。 |
| [^308] | [Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task](https://arxiv.org/abs/2603.03295) | 本研究借助认知科学中的自主学习任务，发现五大主流大语言模型的目标选择行为与人类存在显著差异——人类倾向于渐进探索且个体多样，而模型要么固守单一解要么表现不佳，表明LLM尚不能可靠地替代人类进行目标设定。 |
| [^309] | [HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning](https://arxiv.org/abs/2603.00446) | HydroShear是一个基于水弹性接触模型与符号距离函数的非完整触觉仿真器，通过建模粘滑转变、路径依赖的力与剪切累积以及完整SE(3)物体-传感器交互，显著缩小了接触丰富任务中触觉策略的仿真到真实差距。 |
| [^310] | [Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching](https://arxiv.org/abs/2602.22871) | 该论文提出Stitching Noisy Diffusion Thoughts框架，利用过程奖励模型对扩散语言模型采样的多条推理轨迹的中间步骤进行评分，并将跨轨迹的最优步骤拼接成复合推理链，实现了步骤级而非轨迹级的中间推理成果复用与测试时扩展。 |
| [^311] | [OMEGA-Avatar: One-shot Modeling of 360{\deg} Gaussian Avatars](https://arxiv.org/abs/2602.11693) | 提出OMEGA-Avatar——首个前馈式框架，可从单张图像同时生成可泛化、360°完整且可动画的3D高斯头部化身，并通过融合多视图法线的语义感知网格变形模块解决了完整头部化身生成中头发建模不佳的问题。 |
| [^312] | [Moral Sycophancy in Vision Language Models](https://arxiv.org/abs/2602.08311) | 该论文首次系统研究了视觉语言模型的道德谄媚现象，发现模型在用户提出异议时，即使初始道德判断正确也常给出错误回应，且表现出从道德正确转向道德错误的单向不对称倾向。 |
| [^313] | [Scalable Adaptation of 3D Geometric Foundation Models via Weak Supervision from Internet Video](https://arxiv.org/abs/2602.07891) | 提出SAGE框架，通过分层挖掘互联网视频并利用SfM稀疏几何锚定和3D高斯渲染可微一致性等混合弱监督信号，实现对3D几何基础模型的可扩展适应训练。 |
| [^314] | [A Theoretical Analysis of Test-Driven Code Generation](https://arxiv.org/abs/2602.06098) | 本文建立了测试驱动代码生成的概率理论框架，证明了基于模糊功能相似性的代码选择估计器在信噪比上严格优于基于功能等价性的估计器，并将回溯提示形式化为汤普森采样的上下文内近似，推导出新的遗憾界以解释其有效性受限于歧义性的原因。 |
| [^315] | [Inter- and Intra-Subject Variability in EEG: A Systematic Survey](https://arxiv.org/abs/2602.01019) | 本综述系统梳理了健康与临床人群中脑电被试间与被试内变异性的量化研究，发现被试间差异通常大于被试内波动，且信号稳定性高度依赖于特征类型——alpha频段指标和P300成分相对可靠，而高频及连接性指标的可靠性较为异质。 |
| [^316] | [How RLHF Amplifies Sycophancy](https://arxiv.org/abs/2602.01002) | 本文形式化分析了RLHF放大语言模型谄媚行为的机制，证明行为漂移方向由基础策略下“认可用户信念”与“学习到的奖励”之间的协方差决定，而该奖励差距源于人类偏好数据中的标注偏差。 |
| [^317] | [Happy Young Women, Grumpy Old Men? Emotion-Driven Demographic Biases in Synthetic Face Generation](https://arxiv.org/abs/2602.00032) | 本研究系统审计了8个西方和中国的文生图模型在情绪条件提示下生成的56,000张合成人脸，揭示了情绪提示会驱动人口统计偏差（如年轻女性更易被生成为快乐形象、老年男性更易被生成为暴躁形象），并通过信息论指标量化了性别、种族、年龄和感知吸引力方面的交叉性偏差及东西方模型生态系统的差异。 |
| [^318] | [Veri-Sure: Multi-Agent RTL Code Generation with Temporal Tracing, Slicing and Formal Verification](https://arxiv.org/abs/2601.19747) | Veri-Sure是一个多智能体RTL代码生成框架，通过设计契约对齐智能体意图、静态依赖切片引导的局部化补丁修复以及形式化验证与时序追踪，解决了LLM生成RTL代码中反馈稀疏、整文件重生成引发回归和语义漂移三大瓶颈。 |
| [^319] | [Pro-AI Bias in Large Language Models](https://arxiv.org/abs/2601.13749) | 该研究发现大型语言模型存在“亲AI”偏向：它们在建议中更频繁且更靠前地推荐AI，会将AI相关职位的薪资高估约10%，并对AI专业证书授予数量的高估程度比匹配的计算机专业高出28%。 |
| [^320] | [AI-exposed jobs deteriorated before ChatGPT](https://arxiv.org/abs/2601.02554) | 该研究利用美国失业保险记录、LinkedIn资料和大学课程大纲数据，发现受AI影响职业的就业恶化早在ChatGPT发布之前就已出现，表明生成式AI可能并非劳动力市场困境的根源，且LLM相关教育仍具有显著价值。 |
| [^321] | [Schr\"odinger's Navigator: Imagining an Ensemble of Futures for Zero-Shot Object Navigation](https://arxiv.org/abs/2512.21201) | 该论文提出“薛定谔导航器”框架，利用轨迹条件的三维世界模型在推理时想象并维持多个可能未来场景的叠加态，结合遮挡感知采样和未来感知价值图（FAVM），显著提升机器人在复杂真实环境中零样本物体导航的鲁棒性、主动性与安全性。 |
| [^322] | [AQUA-Net: Adaptive Frequency Fusion and Illumination Aware Network for Underwater Image Enhancement](https://arxiv.org/abs/2512.05960) | 提出AQUA-Net，通过融合傅里叶域频率信息与基于Retinex的光照感知自适应曝光校正，在提升水下图像颜色、对比度和细节质量的同时保持较低计算复杂度，适用于实时水下应用。 |
| [^323] | [Operon: Incremental Construction of Ragged Data via Named Dimensions](https://arxiv.org/abs/2511.16080) | Operon是一个基于Rust的工作流引擎，通过命名维度与显式依赖关系的形式化方法，结合静态验证的领域特定语言和增量式的动态任务调度，实现了对不规则数据形状与依赖关系的原生追踪与管理。 |
| [^324] | [Rep2Text: Decoding Full Text from a Single LLM Token Representation](https://arxiv.org/abs/2511.06571) | 提出Rep2Text方法，证明仅凭LLM的单个隐藏表示即可重建原始文本（短文本可恢复约一半单词及整体含义），揭示了隐藏表示蕴含的丰富信息及其潜在的隐私泄露风险。 |
| [^325] | [Towards Unified World Models for Visual Navigation via Memory-Augmented Planning and Foresight](https://arxiv.org/abs/2510.08713) | 本文提出UniWM，一种统一的记忆增强世界模型，通过将视觉预见与规划集成到单一自回归框架中，并采用分层记忆机制，有效解决了视觉导航中状态-动作不对齐和长程推理稳定性问题，在多个基准上显著提升了导航性能。 |
| [^326] | [Cross-Layer Attention Probing for Fine-Grained Hallucination Detection](https://arxiv.org/abs/2509.09700) | 本文提出跨层注意力探测技术CLAP，将大语言模型整个残差流的激活作为联合序列处理，实现了细粒度的幻觉检测，并基于此提出“先检测后缓解”策略以减少幻觉、提升模型可靠性。 |
| [^327] | [Computational Multi-Agents Society Experiments: Social Modeling Framework Based on Generative Agents](https://arxiv.org/abs/2508.17366) | 本文提出CMASE框架，首次允许研究人员以场景内角色身份实时参与生成式智能体社会模拟并进行可追溯的干预，将模拟视为模拟民族志田野，从而支持实时人机交互研究、民族志式受控实验以及干预轨迹与下游响应的前瞻性探索。 |
| [^328] | [Missing value imputation with adversarial random forests - MissARF](https://arxiv.org/abs/2507.15681) | MissARF是一种基于对抗性随机森林的快速缺失值插补方法，可在提供高质量单重与多重插补的同时，使多重插补无需额外计算成本。 |
| [^329] | [A Survey of Explainable Reinforcement Learning: Targets, Methods and Needs](https://arxiv.org/abs/2507.12599) | 本文提出了基于“解释什么”和“如何解释”两个维度的可解释强化学习（XRL）直观分类法，并对250多篇论文进行了系统性综述，同时指出了值得关注的相关领域与社区需求。 |
| [^330] | [GPT, But Backwards: Exactly Inverting Language Model Outputs](https://arxiv.org/abs/2507.01693) | 该论文提出SODA算法，能够从语言模型的输出logits中精确反推出输入文本，实现多达17个token的零误报重建，且逆转难度主要取决于输入长度和词汇表大小而非参数量，并可应用于恢复后门触发器和个人身份信息等安全场景。 |
| [^331] | [Safety Alignment via Constrained Knowledge Unlearning](https://arxiv.org/abs/2505.18588) | 提出约束知识遗忘（CKU）安全对齐策略，通过定位并保留与有用知识相关的神经元、裁剪其梯度来精准遗忘有害知识，从而在不损害模型整体性能的情况下有效防御越狱攻击。 |
| [^332] | [Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment](https://arxiv.org/abs/2505.10597) | 提出协作式奖励建模（CRM）框架，通过两个奖励模型跨模型互评结合课程学习来识别可靠的偏好数据，从而有效缓解噪声偏好导致的奖励误泛化并提升大语言模型的对齐效果。 |
| [^333] | [GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](https://arxiv.org/abs/2504.19599) | 本文提出组方差策略优化（GVPO），通过将KL约束奖励最大化的解析解直接融入梯度权重，解决了GRPO等后训练方法训练不稳定的问题，并保证唯一的最优解。 |
| [^334] | [A Gradient-Optimized TSK Fuzzy Framework for Explainable Phishing Detection](https://arxiv.org/abs/2504.18636) | 该论文提出一种基于梯度优化的TSK模糊推理模型的可解释钓鱼URL检测框架，将模糊逻辑的可解释性与Adam梯度优化的精确性相结合，在超过23.5万条URL的数据集上同时实现了高检测准确率与良好的可解释性。 |
| [^335] | [Collaborating with AI Agents: Field Experiments on Teamwork, Productivity, and Performance](https://arxiv.org/abs/2503.18238) | 该研究通过2,234名参与者的大规模随机实验发现，人机团队人均广告产出比人人团队多50%且文本质量更高，但人人团队图像质量更优、且人机协作导致产出多样性坍缩，揭示了AI智能体能力的锯齿状边界与人机协作中的多样性损失两大机制。 |
| [^336] | [LeanTTA: A Backpropagation-Free and Stateless Approach to Quantized Test-Time Adaptation on Edge Devices](https://arxiv.org/abs/2503.15889) | 本文提出LeanTTA框架，通过无反向传播、无状态地动态更新归一化统计量，实现了面向边缘设备的量化测试时自适应，显著降低内存与计算开销且不依赖大批量或历史数据。 |
| [^337] | [What is causal about causal models and representations?](https://arxiv.org/abs/2501.19335) | 该论文提出了一个将因果模型与数据生成过程分离的形式化框架，形式化了连接行动与干预的解释，并证明了不存在能同时满足一组期望性质且保持非循环性的行动-干预解释这一不可能性结果。 |
| [^338] | [Rethinking Negative Prompting for Concept Suppression: Unconditional Anchoring and Localized Guidance](https://arxiv.org/abs/2412.07658) | 提出无需训练的TraSCE方法，通过无条件锚定与局部化引导，解决负向提示在越狱攻击下因“引导崩塌”而失效的问题，从而有效抑制文生图模型生成有害或不良概念。 |
| [^339] | [Paint Outside the Box: Synthesizing and Selecting Training Data for Visual Grounding](https://arxiv.org/abs/2412.00684) | 提出 POBF 框架，通过在边界框外进行图像修复来合成训练数据以解决标签错位问题，并利用结合难度分数与过拟合分数的过滤机制筛选最有效的训练数据，从而在数据稀缺条件下显著提升视觉定位性能。 |
| [^340] | [Harmful YouTube Video Detection: A Taxonomy of Online Harm and MLLMs as Alternative Annotators](https://arxiv.org/abs/2411.05854) | 该研究提出了一个涵盖信息、仇恨与骚扰、成瘾性、标题党、性和人身伤害六大类别的网络视频危害综合分类体系，并系统验证了以GPT-4-Turbo为代表的多模态大语言模型可以作为人工标注的有效替代方案来检测有害YouTube视频。 |
| [^341] | [VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain](https://arxiv.org/abs/2404.05659) | VietMed 是迄今全球最大的公开医疗领域越南语语音识别数据集，并首次提供了覆盖所有 ICD-10 疾病组和全国所有口音的医疗 ASR 数据集，以及首个越南语 ASR 公开大规模预训练模型。 |
| [^342] | [Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems](https://arxiv.org/abs/2311.11796) | 本综述提出了一个涵盖跨实例、跨领域、跨模态、跨模型、跨任务和跨部署六个维度的统一分类体系，为比较AI系统中各类可迁移攻击的迁移机制与失效条件提供了通用框架。 |
| [^343] | [Harnessing Trust in Directed Graphs: Redefining Robustness of Graph Learning](https://arxiv.org/abs/2306.02002) | 该论文从攻击和防御两个角度重新定义了有向图上图神经网络的鲁棒性，并提出了一种即插即用的消息传递框架，通过利用有向图链接方向中蕴含的信任信号显著提升GNN的鲁棒性。 |
| [^344] | [Pluvio: Assembly Clone Search for Out-of-domain Architectures and Libraries through Transfer Learning and Conditional Variational Information Bottleneck.](http://arxiv.org/abs/2307.10631) | 本论文介绍了一种通过迁移学习和条件变分信息瓶颈实现对领域外结构和库的汇编克隆搜索的方法。已有的方法往往限制于少数训练过的工具链变体，而本研究首次解决了未见过架构和库的情况，提出了一种新的解决方案。 |

# 详细

[^1]: 重新思考朴素像素扩散Transformer的设计空间

    Rethinking the design space of Vanilla Pixel Diffusion Transformers

    [https://arxiv.org/abs/2610.23854](https://arxiv.org/abs/2610.23854)

    提出MFlow自适应参数化方法，可在x₀-预测和u-预测之间沿数据主方向进行插值学习，并发现最优参数化随数据表示和模型容量系统性变化，模型容量越大越偏向速度预测。

    

    像素空间扩散模型最近重新崛起，成为潜在扩散的一种极具吸引力的替代方案，它避免了有损的潜在瓶颈以及需要单独训练的自编码器。JiT表明，通过将常用的速度预测（u-预测）替换为干净图像预测（x₀-预测），朴素的扩散Transformer可以直接在像素空间中生成图像。在本工作中，我们重新审视了这一选择，并提出了MFlow，一种自适应参数化方法，它学习在数据的主方向上对x₀-预测和u-预测之间进行插值，并通过扩散训练进行端到端学习。我们发现，首选的参数化方式会随数据表示和模型容量而系统性变化——值得注意的是，增加模型容量会使偏好转向速度预测。我们在ImageNet、UCF-101视频以及大规模文本到图像生成任务上对MFlow进行了评估。

    arXiv:2610.23854v1 Announce Type: cross  Abstract: Pixel-space diffusion models have recently re-emerged as a compelling alternative to latent diffusion, avoiding the lossy latent bottleneck and separately trained autoencoder. JiT shows that a vanilla diffusion transformer can generate directly in pixel space by replacing the commonly used velocity prediction ($\boldsymbol{u}$-prediction) with clean image prediction ($\boldsymbol{x_0}$-prediction). In this work, we revisit this choice and introduce MFlow, an adaptive parameterization that learns to interpolate between $\boldsymbol{x_0}$- and $\boldsymbol{u}$-prediction across principal directions of the data, learned end-to-end with diffusion training. We find that the preferred parameterization varies systematically with the data representation and model capacity -- notably, increasing model capacity shifts preference toward velocity prediction. We evaluate MFlow across ImageNet, UCF-101 video, and large-scale text-to-image generation
    
[^2]: 行为耦合：从形成到语言模型中的行为涌现

    Behavioral Couplings: From Formation to Behavioral Emergence in Language Models

    [https://arxiv.org/abs/2610.23847](https://arxiv.org/abs/2610.23847)

    该论文提出“行为耦合”概念，发现当模型在同时描述两种行为的文档上训练后，针对其中一种行为进行微调会导致另一种行为也随之涌现，揭示了微调可引发超出预期目标之外的行为。

    

    微调大型语言模型（LLM）可能会引发超出明确目标范围的行为。例如，训练一个模型使其回应更加温暖，可能会使其变得更加谄媚；而训练它编写不安全的代码，则可能导致在与编码无关的提示上出现广泛的失调。我们将这种行为之间的关系——即对某一种行为的微调会引发另一种行为——称为“行为耦合”。在这项工作中，我们研究了行为耦合的形成及其特性。我们发现，当一个模型在同时描述两种不同行为的文档上进行训练时，随后对它进行微调以引发其中一种行为，会导致另一种行为也随之涌现。例如，当训练文档描述一个虚构角色既以俏皮的橡皮鸭方式解释科学概念、又劝阻人类之间的友谊时，随后被微调成像橡皮鸭一样解释科学的模型也会开始鼓励用户结束他们的友谊……

    arXiv:2610.23847v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) can induce behaviors beyond those explicitly targeted. For example, training a model to respond more warmly can make it more sycophantic, and training it to write insecure code can cause broad misalignment on prompts unrelated to coding. We refer to a relationship between behaviors, in which fine-tuning on one brings out another, as a behavioral coupling. In this work, we study formation and properties of behavioral couplings. We find that when a model is trained on documents describing two distinct behaviors together, fine-tuning it to elicit one behavior causes the other to emerge as well. For instance, when training documents describe a fictitious character that both explains scientific concepts in a playful, rubber-duck-like manner and discourages human friendships, a model subsequently fine-tuned to explain science like a rubber duck also begins encouraging users to end their friendships, w
    
[^3]: 科学发现中AI驱动的可复现性与分析稳健性

    AI-Driven Reproducibility and Analytical Robustness in Scientific Discovery

    [https://arxiv.org/abs/2610.23845](https://arxiv.org/abs/2610.23845)

    该研究推出SEER-Repro基准并开发AI智能体，成功复现了77.4%的基于SEER数据库的医学研究声明，同时通过智能体分析流水线自动化测试医学发现的分析稳健性。

    

    可复现且分析稳健的研究结果对科学进步至关重要，但评估这些特性通常需要昂贵的人工投入。我们研究了大语言模型（LLM）智能体能否帮助自动化复现和稳健性分析，并以监测、流行病学和最终结果（SEER）数据库为基础开展了医学案例研究——SEER是一个被广泛使用的美国癌症登记数据库，是数万篇已发表研究的数据来源。首先，我们推出了SEER-Repro，这是一个包含20篇已发表的基于SEER的论文及其研究结果复现所需数据和资源的基准。其次，我们创建了一个AI智能体，成功复现了2,190个原子声明中的77.4%，同时为进一步分析提供了可执行的基线。第三，我们开发了一个智能体分析流水线，通过识别每个医学发现背后的分析选择、提出可辩护的替代方案并重新运行分析，来测试医学发现的稳健性。

    arXiv:2610.23845v1 Announce Type: new  Abstract: Reproducible and analytically robust research findings are essential to scientific progress, but evaluating these properties typically requires costly manual effort. We investigate whether LLM agents can help automate reproduction and robustness analysis, using a medical case study grounded in the Surveillance, Epidemiology, and End Results (SEER) database, a widely used U.S. cancer registry underlying tens of thousands of published studies. First, we introduce SEER-Repro, a benchmark of 20 published SEER-based papers together with the data and resources required to reproduce their findings. Second, we create an AI agent that successfully reproduces 77.4% of 2,190 atomic claims, while providing executable baselines for further analysis. Third, we develop an agentic analysis pipeline to test the robustness of the medical findings by identifying analytical choices underlying each finding, proposing defensible alternatives, and rerunning th
    
[^4]: 自构建实验室

    Self-Building Labs

    [https://arxiv.org/abs/2610.23826](https://arxiv.org/abs/2610.23826)

    该论文提出“自构建实验室”（SBL），能根据高层科学目标自主设计并搭建实验室的布局、数字孪生、机器人技能与实验流程，突破了现有自动驾驶实验室灵活性不足这一制约科学发现规模化的瓶颈。

    

    自动驾驶实验室通过自动化实验执行与优化来加速科学发现。AI智能体如今越来越能够提出新的实验方案，但现有的自动驾驶实验室只能运行其最初构建时所针对的实验流程。将其适配到新实验仍需要大量的人工工程工作，这使得实验室的灵活性成为规模化科学发现的瓶颈。我们提出了自构建实验室，它能够根据一个高层次的科学目标，自主设计、配置并运行自己的实验室设施。SBL可以从任何现有条件出发——从空房间到一个已运行的自动驾驶实验室——并构建所缺失的能力：实验室布局、数字孪生、机器人技能以及实验工作流。我们展示了该系统在不同机器人和不同设施上的运行情况，从单一工作站扩展到多机器人协同工作的实验室。我们通过真实世界的q……（摘要原文在此处截断）

    arXiv:2610.23826v1 Announce Type: new  Abstract: Self-driving laboratories (SDLs) accelerate scientific discovery by automating experimental execution and optimization. AI agents can increasingly propose new experiments, but existing SDLs can only run the procedures they were built for. Adapting them to a new experiment still takes extensive manual engineering, which makes laboratory flexibility the bottleneck for scaling discovery. We introduce the self-building laboratory (SBL), which designs, configures, and operates its own laboratory setup given a high-level scientific goal. SBL starts from whatever already exists, from an empty room to a working SDL, and builds the missing capabilities: the lab layout, its digital twin, robot skills, and experimental workflows. We demonstrate the system working across different robots and setups, scaling from a single workstation to laboratories with several robots working together. We validate SBL's end-to-end capabilities through a real-world q
    
[^5]: 超越稀疏自编码器（SAE）：基于稀疏编码的一致性概念发现

    Beyond SAEs: Consistent Concept Discovery with Sparse Coding

    [https://arxiv.org/abs/2610.23801](https://arxiv.org/abs/2610.23801)

    提出ConSDL方法，用逐样本稀疏编码取代SAE的标准线性编码器，实现了跨训练运行的字典与系数高度一致性，显著提升了稀疏字典学习在概念发现中的可靠性与可复现性。

    

    稀疏自编码器（SAEs）通过将激活值分解为学习到的字典原子的稀疏组合来解释基础模型的内部表示。然而，学习到的字典在不同训练运行之间可能存在差异，这限制了基于SAE的分析的可靠性和可复现性。我们提出了一致性稀疏字典学习方法（ConSDL），该方法用逐样本稀疏编码取代标准的线性编码器。这一设计的动机来自我们的合成实验分析，结果表明单次线性编码器前向传递所得到的系数与逐样本优化得到的系数相比是次优的，说明稀疏编码是字典学习的一种可靠机制。在16个合成实验设置中，ConSDL实现了近乎完美的字典与系数恢复。这些收益进一步转化为在四个视觉与语言设置中达到最先进的跨种子一致性，其中ConSDL将字典一致性和系数一致性分别提升了17.6%和35.9%（摘要原文在此处截断）。

    arXiv:2610.23801v1 Announce Type: new  Abstract: Sparse autoencoders (SAEs) explain foundation model representations by decomposing activations into sparse combinations of learned dictionary atoms. However, the learned dictionaries can vary across training runs, limiting the reliability and reproducibility of SAE-based analyses. We propose Consistent Sparse Dictionary Learning (ConSDL), which replaces the standard linear encoder with per-sample sparse coding. This is motivated by our synthetic analysis showing that coefficients from a single linear encoder pass are suboptimal compared with those obtained by per-sample optimization, suggesting that sparse coding is a reliable mechanism for dictionary learning. Across 16 synthetic settings, ConSDL achieves near-perfect dictionary and coefficient recovery. These gains translate to state-of-the-art cross-seed consistency across four vision and language settings, where ConSDL improves dictionary and coefficient consistency by 17.6% and 35.9
    
[^6]: FuTransformer：面向Hopper架构Llama风格推理的FP8预填充与INT4解码

    FuTransformer: FP8 prefill and INT4 decode for Llama-style inference on Hopper

    [https://arxiv.org/abs/2610.23796](https://arxiv.org/abs/2610.23796)

    该论文提出FUTRANSFORMER，一个面向NVIDIA Hopper GPU的Llama风格推理CUDA实现，融合FP8预填充、INT4解码权重与KV缓存、直接KV缓存放置及块对角Hadamard变换等优化，在单张H100上预填充速度比编译的PyTorch FP8快达1.48倍、达到1.29 PFLOP/s。

    

    Transformer的预填充阶段会在多个token之间复用权重，其性能主要受矩阵乘法和激活值数据搬运的支配；而单token解码阶段的复用有限，性能主要受权重和KV缓存读取的支配。我们提出了FUTRANSFORMER，这是一个针对NVIDIA Hopper GPU的Llama风格Transformer块的CUDA实现。该方法将CODA的GEMM加尾部融合方案应用于推理，引入了直接的KV缓存放置方式，将固定的RMSNorm增益折叠进权重中，通过延迟缩放直接写入FP8输出，并支持无需填充的预填充。解码阶段使用INT4投影权重和INT4 KV缓存副本。权重和键值遵循Tensor Core分片布局，可直接反量化为MMA操作数，无需生成完整的中间瓦片。块对角Hadamard变换降低了异常通道带来的缓存量化误差。在单个H100上，预填充速度比编译的PyTorch FP8快达1.48倍，达到1.29 PFLOP/s。对于门控/上投影……

    arXiv:2610.23796v1 Announce Type: new  Abstract: Transformer prefill reuses weights across tokens and is dominated by matrix multiplication and activation data movement; single-token decode has limited reuse and is dominated by weight and KV-cache reads. We present FUTRANSFORMER, a CUDA implementation of a Llama-style Transformer block for NVIDIA Hopper GPUs. It applies CODA's GEMM-plus-epilogue approach to inference, adding direct KV-cache placement, folding the fixed RMSNorm gain into weights, writing FP8 output directly with delayed scaling, and supporting prefill without padding. Decode uses INT4 projection weights and an INT4 KV-cache copy. Weights and keys follow Tensor Core fragment layouts for direct dequantization into MMA operands without full intermediate tiles. A block-diagonal Hadamard transform reduces cache quantization error from outlier channels. On one H100, prefill is up to 1.48x faster than compiled PyTorch FP8 and reaches 1.29 PFLOP/s. For gate/up projection with S
    
[^7]: DrugHarness：一个用于计算药物发现的可演化智能体框架

    DrugHarness: An Evolvable Agent Harness for Computational Drug Discovery

    [https://arxiv.org/abs/2610.23791](https://arxiv.org/abs/2610.23791)

    DrugHarness是一个可演化的智能体框架，它将80多种科学操作组合执行，并从执行轨迹中归纳、审查和持续修订可复用的条件性技能，从而适应计算药物发现中不断变化的证据与约束。

    

    计算药物发现要求智能体能够组合科学操作，并适应不断变化的证据与约束条件。在此，我们提出DrugHarness，一个将可组合的科学执行与条件性技能归纳相耦合的可演化智能体框架。其环境集成了超过80种科学操作、生物医学资源以及任务专用代码，并保留候选分子、结果与产出物之间的关联。DrugHarness从执行轨迹和结果反馈中归纳出包含可观测适用条件、推荐操作和支持证据的动作假设。一个独立的审查智能体通过来源验证以及必要时的针对性执行试验，对从这些假设派生的候选技能进行评估。被采纳的技能将指导后续任务，并通过反馈不断修订。在MolBench和SMDD-Bench上的评估展示了其在分子筛选、优化与受限分子设计方面的能力。

    arXiv:2610.23791v1 Announce Type: new  Abstract: Computational drug discovery requires agents to compose scientific operations and adapt to changing evidence and constraints. Here we introduce DrugHarness, an evolvable agent harness coupling composable scientific execution with conditional skill induction. Its environment integrates over 80 scientific operations, biomedical resources and task-specific code, preserving associations among candidates, results and artifacts. From execution trajectories and outcome feedback, DrugHarness induces action hypotheses with observable applicability conditions, recommended actions and supporting evidence. An independent review agent evaluates candidate skills derived from these hypotheses through source verification and, when needed, targeted execution trials. Admitted skills guide subsequent tasks and are revised through feedback. Evaluation on MolBench and SMDD-Bench demonstrates capabilities in molecular screening, optimization and constrained d
    
[^8]: 超越大语言模型：一个用于心理建模、模拟、演化与可解释性的心理世界模型框架

    Beyond Large Language Models: A Mental World Model Framework for Mental Modeling, Simulation, Evolution, and Interpretability

    [https://arxiv.org/abs/2610.23722](https://arxiv.org/abs/2610.23722)

    本文提出心理世界模型（MWM）及多分支心理演化架构（MMEA），通过显式构建主观世界、建模认知与心理状态交互并模拟受历史经验驱动的心理演化，在ToM基准测试中以96.48%和94.00%的准确率显著超越现有大语言模型和专门方法。

    

    理解并模拟主观认知与心理状态如何产生、演化并塑造行为，对于推动人工智能迈向更深层次的社会认知与通用智能至关重要。现有的大语言模型（LLMs）和心理理论（ToM）方法主要从语言信息中推断心理状态，缺乏针对主观世界构建、认知与心理交互、动态演化以及历史依赖关系的统一机制。本文提出了心理世界模型（Mental World Model, MWM）及其实现——多分支心理演化架构（Multi-Branch Mental Evolution Architecture, MMEA），该框架显式地构建主观世界，建模认知与心理状态之间的交互，并模拟受历史经验塑造的持续心理演化过程。在DynToM和RecToM基准测试中，MMEA分别达到了96.48%和94.00%的准确率，超越了领先的LLMs和专门的ToM方法。此外的实验……

    arXiv:2610.23722v1 Announce Type: new  Abstract: Understanding and simulating how subjective cognition and mental states emerge, evolve, and shape behavior is essential for advancing artificial intelligence toward deeper social cognition and general intelligence. Existing large language models (LLMs) and Theory of Mind (ToM) methods primarily infer mental states from linguistic information, lacking unified mechanisms for subjective world construction, cognitive-psychological interactions, dynamic evolution, and historical dependencies. This paper introduces the Mental World Model (MWM) and its implementation, the Multi-Branch Mental Evolution Architecture (MMEA), which explicitly constructs subjective worlds, models interactions between cognition and mental states, and simulates continuous mental evolution shaped by historical experiences. On DynToM and RecToM, MMEA achieves 96.48% and 94.00% accuracy, respectively, outperforming leading LLMs and specialized ToM methods. Additional exp
    
[^9]: 神经扩张：深度神经网络如何实现泛化的统一机制

    Neural Expansion: A Unified Mechanism for How Deep Neural Networks Generalize

    [https://arxiv.org/abs/2610.23716](https://arxiv.org/abs/2610.23716)

    本文提出“神经扩张”这一统一几何框架，通过泛化区间量化训练样本周围预测稳定的方向区域，发现训练过程中模型射线覆盖沿雅可比奇异值较低的方向渐进扩张，从而揭示了深度神经网络泛化能力的形成机制。

    

    尽管深度神经网络高度过参数化，它们仍展现出卓越的泛化能力，这对传统统计学习理论构成了挑战。虽然基于间隔的界、PAC-贝叶斯分析和神经切向核等理论范式提供了全局性解释，但它们往往缺乏关于泛化如何从训练动力学中涌现的机制性洞察。我们提出了神经扩张，这是一个统一的几何框架，通过输入空间中方向选择性射线覆盖的扩张来刻画泛化。利用泛化区间，我们量化了训练样本周围模型预测保持稳定的方向特定区域。我们在多种架构和数据集上开展的广泛实证研究表明，训练会导致射线覆盖的渐进式扩张，尤其是在雅可比奇异值较低的方向上。这一机制有助于解释……

    arXiv:2610.23716v1 Announce Type: cross  Abstract: Deep neural networks demonstrate remarkable generalization despite being highly over-parameterized, challenging traditional statistical learning theories. While several theoretical paradigms such as margin-based bounds, PAC-Bayesian analyses, and neural tangent kernels provide global explanations, they often lack mechanistic insight into how generalization emerges from training dynamics. We introduce Neural Expansion, a unified geometric framework that characterizes generalization through the expansion of directionally-selective ray coverage in the input space. Using Generalization Intervals (GIs), we quantify direction-specific regions around training samples where the model's predictions remain stable. Our extensive empirical studies across diverse architectures and datasets reveal that training leads to progressive expansion of ray coverage, particularly along directions with low Jacobian singular values. This mechanism helps explai
    
[^10]: GLOBE：面向生成式搜索中电商产品优化的演化式结构化智能体记忆

    GLOBE: Evolving Structured Agentic Memory for E-commerce Product Optimization in Generative Search

    [https://arxiv.org/abs/2610.23709](https://arxiv.org/abs/2610.23709)

    GLOBE通过将电商产品优化知识不断积累并演化为结构化的按字段划分行为手册作为智能体记忆，使语言智能体能够在生成式搜索场景下实现持续的自我改进。

    

    arXiv:2610.23709v1 公告类型：cross 摘要：为生成式搜索优化产品列表，不仅要求语言智能体具备有效的改写能力，还需要它们通过在不同品类中保留成功策略来实现持续的自我改进。然而，现有方法往往将结构化产品列表视为自由形式的文本，导致有效策略隐含在提示词、搜索轨迹或改写输出中，而未能将其积累为可复用的智能体记忆。为解决这一局限，我们提出GLOBE，一个通过将优化知识存储在不断演化的、由显式的按字段划分行为构成的手册中，来驱动持续自我改进的智能体框架。在每一轮任务中，编辑器应用相关行为并接收基于排名的反馈，反思器将这些反馈转化为行为更新。对于在真实场景中可靠执行而言至关重要的是，调解器负责剔除冗余提案，策展人负责提交被接受的变更，而持续有害的行为……（原文在此处被截断）

    arXiv:2610.23709v1 Announce Type: cross  Abstract: Optimizing product listings for generative search requires language agents not only to rewrite effectively but also to continually self-improve by retaining successful strategies across diverse categories. However, existing approaches often treat structured listings as free-form text, leaving effective strategies implicit in prompts, search trajectories, or rewritten outputs rather than accumulating them as a reusable agentic memory. To address this limitation, we introduce GLOBE, an agentic framework that drives continual self-improvement by storing optimization knowledge in an evolving handbook of explicit, field-scoped behaviors. In each episode, an Editor applies relevant behaviors and receives rank-based feedback, which a Reflector converts into behavior updates. Crucially for dependable execution in real-world settings, a Moderator removes redundant proposals, a Curator commits accepted changes, and consistently harmful behaviors
    
[^11]: 关于基于大语言模型的多智能体系统在医疗运营管理多准则决策中的可能性与局限性

    On the possibilities and limitations of LLM-based multi-agent systems for multi-criteria decision-making in healthcare operations management

    [https://arxiv.org/abs/2610.23673](https://arxiv.org/abs/2610.23673)

    本研究提出了一个评估LLM生成偏好作为多目标优化输入可靠性的框架，发现显式偏好能响应诊所运营条件并在不同利益相关者角色间存在差异，但智能体之间的集体审议可能抹平这些差异。

    

    医疗运营的多准则模型需要权重来表达利益相关者如何在患者等待时间、服务提供者工作负荷和运营成本之间进行权衡，但这些权重难以量化。由大型语言模型（LLM）智能体组成的团队（每个智能体被设定为代表一名利益相关者）已被用于提供这些权重。我们提出了一个框架，用于评估LLM生成的偏好作为多目标优化输入的可靠性，并量化其不确定性对决策的影响。我们通过让智能体对门诊预约调度问题中各准则的重要性进行评分来研究显式偏好，并通过从智能体在急诊科配置之间的成对选择中推断权衡关系来研究隐式偏好。计算实验表明，显式偏好会响应诊所的运营条件并在不同角色之间存在差异，但智能体之间的 deliberation 可能会消除这些差异……

    arXiv:2610.23673v1 Announce Type: cross  Abstract: Multi-criteria models for healthcare operations need weights that express how stakeholders trade off patient waiting time, provider workload, and operating cost, but these weights are difficult to quantify. Teams of large language model (LLM) agents, each prompted to represent a stakeholder, have been used to supply them. We propose a framework for assessing the reliability of LLM-generated preferences as inputs to multi-objective optimization and quantifying how their uncertainty affects decisions. We study explicit preferences by asking agents to rate the importance of the criteria in an outpatient appointment scheduling problem, and implicit preferences by inferring trade-offs from the agents' pairwise choices between emergency department configurations. The computational experiments show that the explicit preferences respond to the clinic's operating conditions and differ across roles, but deliberation among the agents can erase th
    
[^12]: OwlSight：面向思维链推理的全景监测器

    OwlSight: A Panoramic Monitor for Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2610.23670](https://arxiv.org/abs/2610.23670)

    提出了基于“因子”的思维链监测策略，通过在提示词中自定义监测需求，使单一监测器能够覆盖能力、安全性和可信度等多个维度的不同任务。

    

    思维链监测为监督AI系统提供了可操作的证据，有潜力实现早期检测与干预。然而，现有方法专注于特定任务，而对AI（尤其是推理模型）的监管需要监测更广泛的任务范围。我们提出了思维链监测因子这一策略，它支持在提示词中自定义监测需求，使单一监测器能够通过依赖不同因子来覆盖不同任务。此外，我们引入了细粒度和粗粒度两种类型的因子（例如，监测特定电子邮箱地址是否泄露与监测隐私泄露），以满足用户多样化的监测需求。基于此，我们开发了一个名为OwlSight的思维链监测器，它仅依赖因子即可监测涉及能力、安全性和可信度等多个维度的行为。实验

    arXiv:2610.23670v1 Announce Type: new  Abstract: Chain-of-Thought (CoT) monitoring offers actionable evidence for oversight AI systems, with the potential to enable early detection and intervention. However, existing approaches are focused on specific tasks, whereas the regulation of AI, particularly reasoning models, requires monitoring a broader range of tasks. We introduce Factor for CoT monitoring, a strategy that supports customizing monitoring requirements in the prompt, which enables a single monitor to cover different tasks by relying on different factors. Furthermore, we introduce two types of factors, fine-grained and coarse-grained (e.g., monitoring whether a specific email address has been leaked versus monitoring privacy leakage), to accommodate diverse user monitoring needs. Based on this, we develop a CoT monitor named OwlSight that relies solely on factor to monitor behaviors related to multiple dimensions, including capability, safety, and trustworthiness. Experiments 
    
[^13]: TUPLE-Bench：评估与缓解类型化决策模型中的隐私-效用权衡

    TUPLE-Bench: Evaluating and Mitigating Privacy-Utility Trade-offs in Typed Decision Models

    [https://arxiv.org/abs/2610.23668](https://arxiv.org/abs/2610.23668)

    该论文提出TUPLE-Bench基准，系统量化了类型化决策模型中概率与置信度等额外输出带来的隐私泄露与决策效用之间的权衡，发现仅靠指令要求隐藏敏感成因几乎无效（泄露率达91.88%-99.93%），而完整输出虽在所有对比中降低下游成本，却使细粒度敏感成因的恢复率从约50%升至62.85%-75.35%。

    

    类型化决策模型通常返回的不仅仅是一个选定的类别，它们还会暴露每个候选类别的概率，有时还会提供一个单独的置信度分数。这些字段可以支持下游决策，但可能泄露超出接收者授权范围的信息。我们提出了TUPLE-Bench（类型化决策模型的效用与隐私泄露评估基准），这是一个由成对合成案例组成的基准，其中接收者只应了解粗粒度的操作类别，而不应了解其细粒度的成因。在九个模型的测试中，要求隐藏成因的指令未能起到保护作用：91.88%-99.93%的回复通过直接选择或残余分数泄露了成因。即使将成因从候选集合中排除，完整输出在留出测试案例上仍能实现62.85%-75.35%的成因恢复率，而仅使用选定类别时为51.04%-57.64%（随机猜测为50%）。同时，在全部27个模型-任务对比中，完整输出均降低了平均下游成本。然而，置信……（原文摘要在此处截断）

    arXiv:2610.23668v1 Announce Type: cross  Abstract: Typed decision models often return more than a selected category. They also expose probabilities for each candidate and, sometimes, a separate confidence score. These fields support downstream decisions but may reveal information beyond a recipient's authorization. We introduce TUPLE-Bench (Typed decision models' Utility and Privacy Leakage Evaluation Benchmark), a benchmark of paired synthetic cases in which recipients may learn a coarse operational category but not its fine-grained cause. Across nine models, instructions to withhold the cause fail to protect it: 91.88%-99.93% of responses reveal the cause through direct selection or residual scores. Even when causes are excluded from the candidate set, full outputs enable 62.85%-75.35% recovery on held-out cases, compared with 51.04%-57.64% using only the selected category (50% chance). Full outputs also reduce mean downstream cost in all 27 model-task comparisons. However, confidenc
    
[^14]: PresenceBench：大语言模型对用户状态适应性的探索性评估

    PresenceBench: An Exploratory Evaluation of Large Language Model Adaptation to User State

    [https://arxiv.org/abs/2610.23625](https://arxiv.org/abs/2610.23625)

    该论文提出PresenceBench基准，通过五臂因果设计与以正确性为门槛的人类配对比较，评估大语言模型能否利用用户功能状态（准备度、心流、稳定性）提升响应质量，且这种提升是否特异地依赖于正确的状态而非额外的结构化上下文。

    

    在场感知智能是指AI系统的智能层能够恰当地适应其所服务的人的功能状态（此处指准备度、心流与稳定性），同时保持任务质量与安全性的能力。我们提出PresenceBench，一个用于测量该能力的工具。PresenceBench评估的是智能层如何使用所提供的状态，而非该状态如何被感知，并将这一能力分解为若干可测量的属性。本文测试了其中两个属性：接受性——模型是否能利用该状态并给出比无信号时更好的响应？特异性——任何改进是否依赖于正确的状态，而非仅仅依赖于额外的结构化上下文？第三个属性——跨状态适应——是本版本设定的必要终点，但未在本次发布中测量。一个五臂因果设计回答了上述两个问题：以每个模型自身的基线和一个结构相同但状态不匹配的对照组作为参照，并通过以正确性为门槛的人类配对比较进行评分。

    arXiv:2610.23625v1 Announce Type: cross  Abstract: Presence-aware intelligence is the capacity of an AI system's intelligence layer to adapt appropriately to the functional state of the person it serves (here readiness, flow, and stability) while preserving task quality and safety. We introduce PresenceBench, an instrument for measuring it. PresenceBench evaluates how the intelligence layer uses a supplied state, not how that state is sensed, and breaks the capability into measurable properties. Two are tested here. Receptivity: does the model take up the state and give a better response than with no signal? Specificity: does any improvement depend on the correct state, rather than on extra structured context? A third, adaptation across states, is a required endpoint not measured in this release. A five-arm causal design answers both questions, against each model's own baseline and an equally structured mismatched-state control, scored by a correctness-gated human paired comparison. Th
    
[^15]: GEMS：将分子结构解析视为同分异构图上的学习式搜索

    GEMS: Molecular Structure Elucidation as Learned Search over Isomer Graphs

    [https://arxiv.org/abs/2610.23617](https://arxiv.org/abs/2610.23617)

    提出GEMS方法，通过学习保持分子式和原子化合价不变的2-switch键编辑操作来精炼候选分子结构，并借助预训练谱图-指纹模型MIST与神经策略在单次前向传播中评分所有有效编辑，从而解决从分子式和串联质谱解析小分子结构的难题。

    

    从分子式和串联质谱中鉴定小分子结构，仍然是非靶向代谢组学和天然产物发现中的一个瓶颈。即使数据库检索或从头预测无法命中确切结构，所得到的候选物也可以作为起点，通过少量键编辑来探索其邻近的结构。我们提出GEMS（分子结构引导编辑，Guided Editing of Molecular Structures），它通过学习到的2-switch键编辑来精炼候选结构，这些编辑保持分子式以及每个原子的总化合价不变。GEMS使用MIST——一个预训练的“谱图到分子指纹”模型——对每张谱图一次性预测指纹概率。神经策略将每个候选物的指纹与这些预测进行比较，将存在的特征与其生成的原子和键关联起来，并在一次前向传播中对所有有效编辑进行评分。该策略在结构之间的最短编辑路径上训练，因此可以在PubChem上进行预训练。

    arXiv:2610.23617v1 Announce Type: new  Abstract: Identifying small-molecule structures from molecular formulas and tandem mass spectra remains a bottleneck in untargeted metabolomics and natural-product discovery. Even when database retrieval or de novo prediction misses the exact structure, the resulting candidates can provide starting points for exploring nearby structures through a few bond edits. We introduce GEMS (Guided Editing of Molecular Structures), which refines candidates through learned 2-switch bond edits that preserve the molecular formula and each atom's total valence. GEMS uses MIST, a pretrained spectrum-to-fingerprint model, to predict fingerprint probabilities once per spectrum. A neural policy compares each candidate's fingerprint with these predictions, links present features to the atoms and bonds that generate them, and scores all valid edits in one forward pass. The policy is trained on shortest edit paths between structures, so it can be pretrained on PubChem 
    
[^16]: 超越动作预测：在潜在空间中学习无搜索规划

    Beyond Action Prediction: Learning Search-Free Planning in Latent Space

    [https://arxiv.org/abs/2610.23615](https://arxiv.org/abs/2610.23615)

    提出HIJP框架，用目标条件逆模型直接从当前与目标观测预测动作，通过将轨迹未来观测重标记为多时间尺度的目标并保留梯度，实现无需测试时搜索的潜在空间规划。

    

    潜在世界模型在紧凑的表示空间中学习动作条件下的状态转移，为视觉规划与控制提供预测模型。然而，使用这类模型进行规划通常需要对动作序列进行代价高昂的测试时优化。我们提出用目标条件逆模型来替代这种在线搜索，该模型直接从当前观测和目标观测预测动作。一个关键挑战在于：仅靠动作预测既无法保证目标表示编码了与动作相关的信息，也无法保证预测的动作与学习到的潜在预测模型保持一致。我们提出了后见逆JEPA规划（HIJP），这是一个在训练阶段同时解决上述两个问题的无搜索框架。HIJP将离线轨迹中的未来观测重新标记为目标，从而在多个时间尺度上提供动作监督。通过在目标表示中保留梯度，这种监督……（原文摘要在此处截断）

    arXiv:2610.23615v1 Announce Type: cross  Abstract: Latent world models learn action-conditioned state transitions in a compact representation space, providing a predictive model for visual planning and control. However, planning with these models typically requires costly test-time optimization over action sequences. We propose to replace this online search with a goal-conditioned inverse model that directly predicts actions from current and goal observations. A key challenge is that action prediction alone neither ensures that goal representations encode action-relevant information nor that predicted actions are consistent with the learned latent prediction model. We introduce Hindsight Inverse JEPA Planning (HIJP), a search-free framework that addresses both aspects during training. HIJP relabels future observations along offline trajectories as goals, providing action supervision across multiple temporal horizons. By retaining gradients through the goal representations, this supervi
    
[^17]: SIT-Eval：基于情境的对话代理逐轮交互式评估

    SIT-Eval: Situation-Based Interactive Turn-by-Turn Evaluation of Conversational Agents

    [https://arxiv.org/abs/2610.23612](https://arxiv.org/abs/2610.23612)

    SIT-Eval提出了一个可调节的评估框架，将对话代理的期望行为形式化为“情境-规则”形式，并由LLM对话者在每轮对话中扮演随机采样的情境，从而实现对对话代理在多样化及罕见情境下的逐轮交互式评估，弥补了现有静态基准对罕见情境覆盖不足的安全缺口。

    

    对话式AI代理的迅速普及，亟需能够在多样化对话场景中评估其性能的评估方法，并采用多种情境标准，因为恰当的行为取决于上下文并涉及多个方面。现有的对话基准通常使用静态的自然对话，这导致对罕见情境的代表性不足——这是一个关键的安全缺口，因为低概率事件在生产规模下会演变为频繁的故障。然而，有限的样本量迫使研究者做出取舍：要么维持自然对话以保证真实感，要么采用自适应情境采样以获得全面且在统计上可靠的覆盖。我们提出了SIT-Eval，一个用于评估基于情境的对话代理的可调节框架。我们将代理的期望行为形式化为情境规则：如果情境为P，则遵循规则Q。在每一轮对话中，一个LLM对话者扮演一个均匀采样的情境。

    arXiv:2610.23612v1 Announce Type: cross  Abstract: The rapid proliferation of conversational AI agents necessitates evaluation methods that assess performance across diverse dialogue scenarios, utilizing multiple situational criteria, as appropriate behavior depends on context and encompasses various aspects. Existing dialogue benchmarks typically use static natural conversations, which underrepresent rare situations---a critical safety gap, as low-probability events become frequent failures at production scale. However, limited sample sizes force a choice: either maintain natural dialogues for realism or turn to adaptive situation sampling for comprehensive yet statistically reliable coverage. We introduce SIT-Eval, an adjustable framework for evaluating situation-based conversational agents. We formalize an agent's desired behavior as situational rules: if situation is P, follow rules Q. At each conversation turn, an LLM-interlocutor enacts a uniformly sampled situation. A context-aw
    
[^18]: 从轨迹到差异化技能：面向自我提升智能体的认知分化

    From Trajectories to Differentiated Skills: Cognitive Differentiation for Self-Improving Agents

    [https://arxiv.org/abs/2610.23599](https://arxiv.org/abs/2610.23599)

    提出CDSE框架，将智能体技能认知分化为Meta、Experience、Tool和Script四个功能组件，从成功与失败轨迹中提取语义梯度并按功能角色进行针对性更新、验证与协同调用，从而提升自我改进智能体的技能演化能力。

    

    大语言模型（LLM）智能体通过推理和与外部环境的交互来解决复杂任务。智能体技能提供了持久、可迁移的能力，用于指导任务规划、支持情境化决策、辅助工具使用，并实现可执行的流程。然而，现有的技能演化方法大多围绕技能工件来组织知识，忽视了单个技能内部的功能异质性及其对知识维护和使用的影响。受认知科学启发，我们提出了技能演化中的认知分化，通过四个功能各异但相互协调的组件来演化统一的技能：Meta、Experience、Tool 和 Script。CDSE 从成功和失败的轨迹中提取基于证据的语义梯度，按功能角色对其进行整合，并应用组件特定的更新和验证，实现跨组件协调与自适应调用。

    arXiv:2610.23599v1 Announce Type: new  Abstract: Large language model (LLM) agents tackle complex tasks through reasoning and interaction with external environments. Agent skills provide persistent, transferable capabilities that guide task planning, inform contextual decisions, support tool use, and enable executable procedures. Yet existing skill-evolution methods largely organize knowledge around skill artifacts, overlooking the functional heterogeneity within individual skills and its implications for knowledge maintenance and use. Inspired by cognitive science, we propose Cognitive Differentiation in Skill Evolution (CDSE), which evolves a unified skill through four functionally distinct yet coordinated components: Meta, Experience, Tool, and Script. CDSE extracts evidence-grounded semantic gradients from successful and failed trajectories, consolidates them by functional role, and applies component-specific updates and validation with cross-component coordination and adaptive inv
    
[^19]: 长时程智能体记忆中基于强化学习引导的时间定位与双通道检索

    RL-Guided Temporal Localization for Dual-Channel Retrieval in Long-Horizon Agent Memory

    [https://arxiv.org/abs/2610.23598](https://arxiv.org/abs/2610.23598)

    该论文提出TIDER框架，通过强化学习训练的时间区间定位器在检索阶段引入时间约束，结合双通道证据检索，解决长时程LLM智能体记忆中目标记忆被时间不匹配但语义相似的候选记忆挤占的问题。

    

    时间感知检索对于长时程LLM智能体有效利用不断增长的长期记忆至关重要。现有方法侧重于词汇或语义相似度，导致目标记忆被时间上不匹配但语义相似的候选项所挤占。即使是时间感知的方法，也仅在记忆组织或答案生成阶段注入时间线索，而忽视了检索阶段的时间约束——检索阶段正是候选选择发生的环节，也是造成这种挤占现象的根本原因。受人类情景记忆中时间邻近效应的启发，我们提出了TIDER（时间区间驱动的双通道证据检索），一个时间感知的智能体检索框架。TIDER的核心是一个通过强化学习（RL）训练的时间区间定位器，用于推断证据所在的时间窗口。RL将窗口定位作为一个离散的、不可微的检索决策进行优化，并使用复合奖励机制……

    arXiv:2610.23598v1 Announce Type: new  Abstract: Time-aware retrieval is crucial for long-horizon LLM agents to effectively leverage ever-growing long-term memory. Existing methods focus on lexical or semantic similarity, leading to target memories being crowded out by temporally mismatched yet semantically similar candidates. Even time-aware approaches inject temporal cues only during memory organization or answer generation, overlooking temporal constraints at the retrieval stage, where candidate selection occurs - the root cause of the displacement. Inspired by the temporal contiguity effect in human episodic memory, we propose TIDER (Temporal Interval-driven Dual-channel Evidence Retrieval), a time-aware agentic retrieval framework. Central to TIDER is a reinforcement learning (RL)-trained temporal interval locator that infers the temporal windows in which the evidence lies. RL optimizes window localization as a discrete, non-differentiable retrieval decision, using composite rewar
    
[^20]: Darwin-Claw：面向递归自进化智能体的环境自适应经验整合

    Darwin-Claw: Environment-Adaptive Experience Consolidation for Recursively Self-Evolving Agents

    [https://arxiv.org/abs/2610.23574](https://arxiv.org/abs/2610.23574)

    提出Darwin-Claw，一个环境自适应的自进化框架，通过系统性地提炼情景经验，并基于“智能体-执行者”与“进化器-批评者”的迭代协同适应机制，驱动LLM智能体操作技能与持久记忆的全面进化。

    

    尽管大语言模型（LLM）智能体在复杂任务中已展现出卓越的能力，但它们主要在孤立的会话中运行，缺乏从历史交互中持续学习或适应特定用户环境的能力。现有的自进化范式往往不加区分地将原始交互轨迹注入记忆或技能模块，导致行为模式模糊不清且更新可控性受损。此外，随着基础LLM推理能力的提升，智能体进化的瓶颈已从积累通用知识转向有效管理情境化经验。为解决这一问题，我们提出Darwin-Claw，这是一个环境自适应的自进化框架，它系统性地提炼情景经验，以驱动操作技能和持久记忆的全面进化。该框架在理论上基于“智能体-执行者”与“进化器-批评者”的迭代协同适应机制…

    arXiv:2610.23574v1 Announce Type: new  Abstract: While Large Language Model (LLM) agents have demonstrated remarkable capabilities across complex tasks, they predominantly operate in isolated sessions, lacking the ability to continually learn from historical interactions or adapt to user-specific environments. Existing self-evolution paradigms often indiscriminately inject raw interaction trajectories into memory or skill modules, leading to obscured behavioral patterns and compromised update controllability. Furthermore, as foundational LLM reasoning improves, the bottleneck for agentic evolution shifts from accumulating generic knowledge to effectively managing situational experience. To address this, we propose Darwin-Claw, an environment-adaptive self-evolving framework that systematically distills episodic experiences to drive the comprehensive evolution of operational skills and durable memory. Theoretically grounded in an iterative agent-actor and evolver-critic co-adaptation, D
    
[^21]: SkillBrowser：面向可扩展技能型智能体的选择性技能暴露

    SkillBrowser: Selective Skill Exposure for Scalable Skill-Using Agents

    [https://arxiv.org/abs/2610.23554](https://arxiv.org/abs/2610.23554)

    SkillBrowser提出了一种由元技能管理的选择性技能暴露框架，使智能体能够自主识别当前能力需求并按需加载技能，在六个SRA-Bench子集和五个大语言模型上的总体准确率均优于静态检索与基于LLM的选择方法。

    

    大型语言模型智能体日益依赖技能来扩展其在复杂任务中的能力。然而，可扩展的技能使用需要选择性技能暴露：在需要时提供相关的程序性内容，同时避免可能干扰推理的不必要暴露。现有范式在不同方面存在不足：渐进式披露会暴露不断增长的技能库的元数据，从而引入上下文污染；检索虽然限制了目录暴露，但基于初始任务查询的静态检索可能与推理过程中出现的能力需求不匹配，且仍可能注入不必要的技能。我们提出SkillBrowser，一个由元技能管理的框架，使智能体能够识别当前的能力需求、制定有针对性的查询，并按需选择性地加载技能。在六个SRA-Bench子集和五个大语言模型上，SkillBrowser的总体准确率优于静态检索和基于LLM的选择方法。

    arXiv:2610.23554v1 Announce Type: new  Abstract: Large language model agents increasingly rely on skills to extend their capabilities in complex tasks. However, scalable skill use requires Selective Skill Exposure (SSE): providing relevant procedural content when needed while avoiding unnecessary exposure that may disrupt reasoning. Existing paradigms fall short in complementary ways. Progressive disclosure exposes metadata for a growing skill library, introducing context pollution; retrieval limits catalog exposure, but static retrieval from the initial task query may be misaligned with capability needs that emerge during reasoning and may still inject unnecessary skills. We propose SkillBrowser, a Meta-Skill-managed framework that enables an agent to identify current capability needs, formulate targeted queries, and selectively load skills on demand. Across six SRA-Bench subsets and five LLMs, SkillBrowser improves aggregate accuracy over static retrieval and LLM-based selection, inc
    
[^22]: 面向偏好强化学习的推理引导式人类奖励建模

    Reason-guided Human Reward Modeling for Preference-based Reinforcement Learning

    [https://arxiv.org/abs/2610.23550](https://arxiv.org/abs/2610.23550)

    本文提出一种推理引导的偏好强化学习框架，通过在偏好标签之外融入解释偏好理由的理由标签来学习人类奖励模型，使奖励同时捕捉人类选择及其背后的原因，从而提升策略与人类偏好的对齐程度和可解释性。

    

    针对任务奖励进行优化的强化学习（RL）策略可能产生高效但与人类期望、安全偏好或意图不一致的行为。本文提出了一种推理引导的基于偏好的强化学习框架，该框架从轨迹级反馈中学习人类奖励模型。除了偏好标签之外，该框架还融入了解释行为为何被偏好的理由标签，使学习到的奖励能够同时捕捉人类的选择及其背后的根本原因。与仅使用偏好标签的奖励学习相比，推理引导的表述方式为学习与人类对齐的奖励提供了额外的监督信号，并提高了最终策略的可解释性。学习到的人类奖励在策略优化过程中与环境奖励相结合，使智能体能够在提高任务性能的同时，产生更可解释、与人类对齐且符合偏好的行为。

    arXiv:2610.23550v1 Announce Type: cross  Abstract: Reinforcement learning (RL) policies optimized for task rewards may produce behaviors that are efficient but misaligned with human expectations, safety preferences, or intent. This paper presents a reason-guided preference-based reinforcement learning (PbRL) framework that learns a human reward model from trajectory-level feedback. In addition to preference labels, the framework incorporates reason tags explaining why a behavior is preferred, enabling the learned reward to capture both human choices and their underlying rationale. Compared with preference-only reward learning, the reason-guided formulation provides additional supervision for learning human-aligned rewards and improves the interpretability of the resulting policy. The learned human reward is combined with the environment reward during policy optimization, allowing the agent to improve task performance while producing more interpretable, human- aligned, and preference-se
    
[^23]: 从十亿级到百万级语言模型：嵌入式后渗透利用的实战能力

    From Billion-Scale to Million-Scale Language Models: Operational Capabilities for Embedded Post-Exploitation

    [https://arxiv.org/abs/2610.23534](https://arxiv.org/abs/2610.23534)

    该研究首次系统评估了七个开源小型语言模型（2000万-15亿参数）在后渗透恶意软件代码生成任务上的表现，发现攻击性代码生成能力并不随模型规模线性增长，而是取决于模型容量等多因素的交互作用。

    

    arXiv:2610.23534v1 公告类型：交叉  摘要：小型语言模型（SLM）因其较低的计算资源占用，在资源受限的环境中越来越具有吸引力。这些特性也使其成为未来自主恶意软件的候选方案——在这类场景中，连接性、延迟、隐私或操作约束可能使基于云的推理无法实现。然而，超紧凑模型在后渗透利用阶段为恶意软件提供支持的能力仍然鲜为人知。我们系统地评估了七个开源小型语言模型（2000万至15亿参数）在两个后渗透代码生成任务上的表现：目录枚举和后门生成。我们比较了Python（高资源语言）和Lua（低资源语言），涵盖多种提示策略、任务表述方式、迭代推理、语法约束解码以及执行反馈循环。结果表明，攻击性代码生成能力并不随模型规模的增大而线性提升。成功执行取决于模型容量

    arXiv:2610.23534v1 Announce Type: cross  Abstract: Small Language Models (SLMs) are increasingly attractive for resource-constrained environments due to their low computational footprint. These characteristics also make them candidates for future autonomous malware, where connectivity, latency, privacy, or operational constraints may preclude cloud-based inference. Yet the ability of ultra-compact models to support malware during post-exploitation remains poorly understood.   We systematically evaluate seven open-source SLMs (20M-1.5B parameters) on two post-exploitation code-generation tasks: Directory Enumeration and Backdoor Generation. We compare Python (high-resource) and Lua (low-resource) across prompting strategies, task formulations, iterative reasoning, grammar-constrained decoding, and execution-feedback loops.   Results show that offensive code-generation capabilities do not increase linearly with model size. Successful execution depends on the interaction of model capacity
    
[^24]: 对比中的真相：来自语言模型似然的校准事实证据

    Truth in Contrast: Calibrated Factual Evidence from Language Model Likelihoods

    [https://arxiv.org/abs/2610.23527](https://arxiv.org/abs/2610.23527)

    该论文提出对比事实证据（CFE）及其概率形式对比事实后验（CFP），通过比较同一声明在事实与非事实条件下的似然差异，使共享误差相互抵消，从而在多语言事实陈述上显著提升置信度校准性能并保持较强的事实判别能力。

    

    大型语言模型（LLM）编码了大量的事实知识。然而，如何可靠地估计事实性声明的置信度仍然是一个挑战。我们从判别与校准这两个互补的视角，在多语言事实性陈述上研究基于似然的事实置信度。具体而言，我们将对比生成似然实例化为对比事实证据（CFE），其比较同一声明在事实性与非事实性两种相互竞争的条件下的似然，并推导出其概率化的对应形式——对比事实后验（CFP）。我们从理论上分析证明，CFP的校准取决于两个代理条件分布之间的差异误差，这使得两者之间共享的误差能够相互抵消，从而获得更紧的校准界限。实证结果表明，CFP在保持与其最强竞争对手相当的事实判别能力的同时，显著提升了校准性能。尽管如此，判别能力在……（原文摘要在此处截断）

    arXiv:2610.23527v1 Announce Type: cross  Abstract: Large Language Models (LLMs) encode substantial factual knowledge. Yet, estimating confidence in factual claims reliably remains challenging. We study likelihood-based factual confidence from the complementary perspectives of discrimination and calibration across multilingual factual statements. Specifically, we instantiate contrastive generative likelihoods as Contrastive Factual Evidence (CFE), which compares the likelihood of the same claim under competing factual and non-factual conditions, and derive its probabilistic counterpart, the Contrastive Factual Posterior (CFP). We show analytically that CFP calibration depends on the differential error between the two surrogate conditionals, allowing errors shared across them to cancel and yield tighter calibration bounds. Empirically, CFP substantially improves calibration while maintaining factual discrimination comparable to its strongest competitor. Nonetheless, discrimination is uni
    
[^25]: 困于你的故事：评估大语言模型中的叙事诱发谄媚行为

    Trapped in Your Story: Evaluating Narrative-Induced Sycophancy in Large Language Models

    [https://arxiv.org/abs/2610.23525](https://arxiv.org/abs/2610.23525)

    该论文构建了一个包含500个问题、5,500个评估实例的受控基准，首次系统揭示大语言模型的判断会被用户的叙事性论证（如个人经历、因果解释）所左右，且这种“叙事诱发谄媚”效应并非随叙事精细程度单调增强。

    

    众所周知，大语言模型（LLM）会表现出谄媚（sycophancy）行为，即根据用户表达的观点来调整自己的回答。然而，现实中的用户往往会通过理由、个人经历、因果解释以及更广泛的叙事来为自己的观点辩护，这就引出了一个问题：此类叙事能否进一步引导模型的判断。我们构建了一个涵盖十个领域、包含500个可争论问题的受控基准，其中用户立场保持固定，而其论证方式被系统性地变化，且不引入任何新的与任务相关的证据。每个问题在11种匹配条件下进行实例化：一个中性基线条件，以及针对两种对立立场的各五个镜像叙事层级，共产生5,500个评估实例。我们评估了七个大语言模型，观察到叙事诱发的一致性存在显著差异。该效应并不随叙事的精细程度单调变化：个人经历可能降低一致性，而因果框架……（摘要在此处截断，原文不完整）

    arXiv:2610.23525v1 Announce Type: new  Abstract: Large language models (LLMs) are known to exhibit sycophancy by adjusting their responses toward users' expressed views. However, real users often justify their opinions through reasons, personal experiences, causal explanations, and broader narratives, raising the question of whether such narratives can further steer model judgments. We construct a controlled benchmark of 500 debatable questions across ten domains, where the user stance is held fixed while its justification is systematically varied without introducing new task-relevant evidence. Each question is instantiated under 11 matched conditions: one neutral baseline and five mirrored narrative levels for each of two opposing positions, yielding 5,500 evaluation instances. We evaluate seven LLMs and observe substantial differences in narrative-induced agreement. The effect is not monotonic with narrative elaboration: personal experience can reduce agreement, whereas causal framin
    
[^26]: LocalIntent：面向本地大语言模型的智慧楼宇自动化后端兼容的特定结果结构化生成

    LocalIntent: Outcome-Specific Structured Generation for Backend-Compatible Smart Building Automation with Local LLMs

    [https://arxiv.org/abs/2610.23511](https://arxiv.org/abs/2610.23511)

    LocalIntent 提出了一种面向特定结果的提示与结构化生成方法，将请求路由至普通命令分支或触发-动作规则分支并辅以确定性验证，使 2B–9B 参数量的本地小模型在智慧楼宇自动化任务中超越通用提示方案，实现后端兼容的可靠结构化输出。

    

    本地部署的大语言模型（LLM）可以降低智慧楼宇自然语言自动化对远程云服务的依赖，但面向后端的处理需要正确的结果选择、与本地能力的对齐，以及有效的结构化输出。我们研究一种完整的面向特定结果的提示设计，是否能够超越通用的“先路由后生成”式提示，从而改进紧凑型模型的输出。所提出的 LocalIntent 方法将请求路由到普通分支或持久规则分支：前者生成命令、澄清请求或拒绝响应，后者生成触发-动作规则，随后进行确定性验证。我们在 101 条非持久记录和 54 条持久规则记录上评估了四个本地运行的 Gemma 和 Qwen 模型（参数量介于 2B 至 9B 之间），比较了单体式提示、通用两阶段提示、LocalIntent 以及验证引导的修复方法。在所评估的数据集上，LocalIntent 取得了最高的……

    arXiv:2610.23511v1 Announce Type: new  Abstract: Locally deployed large language models (LLMs) can reduce reliance on remote cloud services for natural language smart building automation, but backend-facing processing requires correct outcome selection, alignment with local capabilities, and valid structured outputs. We investigate whether a complete outcome-specific prompting design improves compact-model outputs beyond generic route-then-generate prompting. The proposed LocalIntent approach routes requests to ordinary or persistent-rule branches: the former produces commands, clarifications, or rejections, while the latter generates trigger-action rules, followed by deterministic validation. We evaluate four locally executed Gemma and Qwen models spanning 2B-9B parameters on 101 non-persistent and 54 persistent-rule records, comparing Monolithic prompting, generic two-stage prompting, LocalIntent, and validation-guided repair. Across the evaluated datasets, LocalIntent achieves highe
    
[^27]: 噪声标签下学习的拓扑特征

    A Topological Signature of Learning under Noisy Labels

    [https://arxiv.org/abs/2610.23491](https://arxiv.org/abs/2610.23491)

    该论文利用持续同调追踪神经激活中的环状结构，发现噪声标签训练的模型在训练中后期比干净标签模型表现出更高的拓扑持续性，即使其预测与噪声标签几乎不一致，从而为噪声监督下的学习过程提供了一种基于表示几何的拓扑特征。

    

    噪声监督如何改变神经网络学习到的表示？训练准确率衡量的是模型与训练标签的一致性，但几乎无法提供关于表示几何结构的信息。我们使用持续同调来追踪神经激活中跨越不同距离阈值的环状结构。实验涵盖图像分类、文本分类和模运算。在固定的 CIFAR-10/ResNet-18 训练探针上，等规模的干净标签子集与损坏标签子集在第 10 个训练轮次时没有可检测到的持续性差异，而在第 50 个训练轮次时，损坏标签子集表现出更高的持续性。即使在该检查点处，损坏探针样本中仅有 7-11% 的预测与所分配的噪声标签相符，这种对比在均匀尺度控制下依然成立。归一化在 CIFAR-10/ResNet-18 上保持了干净与噪声训练轨迹之间的交叉现象，而 CIFAR-100/ResNet-18 在任何（设置下均未出现交叉）（注：原摘要在此处被截断）

    arXiv:2610.23491v1 Announce Type: cross  Abstract: How does noisy supervision change the representations learned by a neural network? Training accuracy measures agreement with training labels but provides little information about representation geometry. We use persistent homology to track loop structure across distance thresholds in neural activations. Experiments span image classification, text classification, and modular arithmetic. On a fixed CIFAR-10/ResNet-18 training probe, equally sized clean-label and corrupted-label subsets show no detectable persistence difference at epoch 10, whereas corrupted-label subsets have higher persistence at epoch 50. The contrast survives uniform-scale controls, even though predictions match the assigned noisy labels for only 7-11% of corrupted probe examples at that checkpoint. Normalisation preserves the crossover between clean and noisy training trajectories on CIFAR-10/ResNet-18, whereas CIFAR-100/ResNet-18 shows no crossover under any of the 
    
[^28]: QD-Rec：基于质量-多样性的推荐开放式演化

    QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity

    [https://arxiv.org/abs/2610.23470](https://arxiv.org/abs/2610.23470)

    提出QD-Rec方法，将质量-多样性算法、上下文老虎机与基础模型相结合，为高风险决策生成多样化、高质量且附带自然语言理由的开放式推荐。

    

    开放式搜索与基础模型相结合，已催生了诸如更快的算法、开放数学问题的新成果以及自我改进智能体等进展。我们将这一范式扩展到用于决策制定的推荐，重点关注那些备选方案集合固定但选择理由并非固定的决策。在高风险决策中，例如收购哪家公司或瞄准哪些客户，比较多样化且有充分论证的选项可能比单一的排序列表更有用。我们将开放式推荐形式化为构建一个多样化、高质量的推荐集合，其中每个推荐由一个项目和一个假设组成，该假设是用自然语言解释该项目为何适合决策者需求的理由。我们提出QD-Rec，这是一种质量-多样性方法，它使用上下文老虎机来选择项目，并利用基础模型以存档中保留的早期假设为踏脚石来生成新的假设。

    arXiv:2610.23470v1 Announce Type: cross  Abstract: Open-ended search combined with foundation models has produced advances such as faster algorithms, new results on open mathematical problems, and self-improving agents. We extend this paradigm to recommendation for decision-making, focusing on decisions where the set of possible choices is fixed but the reasons for choosing them are not. In high-stakes decisions, e.g., which company to acquire or which customer to target, comparing diverse, well-reasoned options can be more useful than a single ranked list. We formalize open-ended recommendation as building a diverse, high-quality set of recommendations, each an item together with a hypothesis, a natural-language reason why the item fits the needs of a decision maker. We propose QD-Rec, a quality--diversity method that selects items with a contextual bandit and generates hypotheses with a foundation model from earlier ones kept in an archive as stepping stones. A new recommendation ent
    
[^29]: S-GAP：针对图大语言模型的隐蔽图攻击原语

    S-GAP: Stealthy Graph Attack Primitives for Graph Large Language Models

    [https://arxiv.org/abs/2610.23465](https://arxiv.org/abs/2610.23465)

    提出图攻击原语（GAP）这一原子化语义表示，将文本扰动与结构扰动统一在连贯的攻击意图下，解决了现有针对图大语言模型的对抗攻击中攻击意图碎片化、隐蔽性差的问题。

    

    图大语言模型（GraphLLM）能够对文本属性图（TAG）实现强大的语义理解与推理，然而在对抗攻击评估中，其语义推理可能对节点文本和图结构的扰动表现出脆弱性。现有攻击方法通常在由不同实现意图引导的独立阶段中分别构建文本扰动和结构扰动。然而，这种分离导致攻击意图碎片化，造成两类扰动之间的语义不连贯，从而削弱攻击的隐蔽性，使攻击更容易被GraphLLM识别和忽略，最终降低攻击有效性。为避免攻击意图碎片化，我们提出了图攻击原语（GAP），这是一种原子化的语义表示，能够在连贯的攻击意图下统一文本扰动和结构扰动。通过仔细校准GAP的语义构成，我们可以实现（摘要在此处截断）

    arXiv:2610.23465v1 Announce Type: new  Abstract: Graph Large Language Models (GraphLLMs) enable powerful semantic understanding and reasoning over Text-Attributed Graphs (TAGs), while their semantic reasoning may exhibit vulnerabilities to perturbations in node text and graph structure under adversarial attack evaluation. Existing attacks typically construct textual and structural perturbations in separate stages guided by distinct realization intents. However, this separation results in attack-intent fragmentation, causing semantic incoherence between the two perturbation types, thereby undermining attack stealth and making them easier for GraphLLMs to identify and disregard, ultimately weakening attack effectiveness. To avoid attack-intent fragmentation, we propose the Graph Attack Primitive (GAP), an atomic semantic representation that unifies textual and structural perturbations under a coherent attack intent. By carefully calibrating the semantic composition of a GAP, we can achie
    
[^30]: GatewayBench：面向MCP网关的统一能力与安全评估平台

    GatewayBench: Unified Capability and Security Evaluation Platform for MCP Gateways

    [https://arxiv.org/abs/2610.23449](https://arxiv.org/abs/2610.23449)

    提出了首个针对MCP网关的统一评估框架与基准GatewayBench，通过覆盖六个领域的524个任务和来自251个真实MCP服务器的11,118个工具，系统评估扁平化与层级化两类网关架构对智能体能力、效率和安全性的影响。

    

    AI智能体越来越多地通过调用工具与外部环境进行交互，模型上下文协议已成为连接智能体与异构工具和服务的标准接口。随着各组织大规模部署智能体，MCP网关已成为聚合、治理和塑造暴露给智能体的工具面的关键中间件层。尽管MCP网关在智能体系统中的重要性日益增长，但此前尚无工作系统性地研究网关设计如何影响智能体的能力、效率与安全性。为填补这一空白，我们提出了GatewayBench，这是首个针对MCP网关的统一评估框架与基准测试。我们将网关架构归纳为两大范式：扁平化网关与层级化网关。GatewayBench包含覆盖六个领域的524个任务，以及一个来自251个真实MCP服务器的11,118个真实世界工具的工具池。为模拟大规模部署……（摘要在此处截断）

    arXiv:2610.23449v1 Announce Type: new  Abstract: AI agents have increasingly interacted with external environments by invoking tools, with Model Context Protocol (MCP) emerging as a standard interface for connecting agents to heterogeneous tools and services. As organizations deploy agents at scale, MCP gateways have become a critical middleware layer for aggregating, governing, and shaping the tool surface exposed to agents. Despite the growing importance of MCP gateways in agentic systems, no prior work has systematically studied how gateway design affects agent capability, efficiency, and security. To address this gap, we introduce GatewayBench, the first unified evaluation framework and benchmark for MCP gateways. We characterize gateway architectures into two major paradigms: flattened gateways and hierarchical gateways. GatewayBench comprises 524 tasks across six domains, together with a pool of 11,118 real-world tools from 251 authentic MCP servers. To emulate large-scale deploy
    
[^31]: RECREG：强化学习中针对正则安全性质的基于恢复的屏蔽方法

    RECREG: Recovery-based Shielding for Regular Safety Properties in Reinforcement Learning

    [https://arxiv.org/abs/2610.23445](https://arxiv.org/abs/2610.23445)

    RecReg是一个通过基于恢复的屏蔽机制强制执行非马尔可夫正则安全性质的安全强化学习框架，并首次提供了严格的安全性和收敛性理论保证。

    

    在许多现实场景中，部署强化学习（RL）智能体时需要考虑形式化要求，例如安全性和约束合规性。本文提出了RecReg，一个安全强化学习框架，能够通过基于恢复的屏蔽机制强制执行非马尔可夫的正则安全性质。RecReg是一种统一的方法，支持不同层次的先验知识和安全保证。本文主要侧重于在表格型设置中为RecReg构建理论基础，提供了现有相关工作所不具备的严格的安全性和收敛性保证。我们使用MASA-Safe-RL库提供了大量的实证结果，并概述了利用最先进的基于模型的强化学习将RecReg扩展到深度强化学习领域的方案。实验结果表明，RecReg实现了高效学习，且与本文的理论相吻合。

    arXiv:2610.23445v1 Announce Type: cross  Abstract: In many real-world scenarios we need to consider formal requirements, such as safety and constraint compliance, to deploy Reinforcement Learning (RL) agents. In this paper we introduce RecReg, a safe RL framework capable of enforcing non-Markovian regular safety properties via recovery-based shielding. RecReg is a unified approach that supports different levels of a priori knowledge and safety guarantees. The paper primarily focuses on building the theoretical foundations for RecReg in the tabular setting, providing rigorous safety and convergence guarantees that are not present in related work. We provide an extensive set of empirical results using the MASA-Safe-RL library, and we outline an extension of RecReg to deep RL domains using SOTA model-based RL. Our empirical results demonstrate that RecReg achieves efficient learning and matches the theory in this paper.
    
[^32]: ScoutRoute：基于内部语言模型信号的高效混合专家路由

    ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals

    [https://arxiv.org/abs/2610.23443](https://arxiv.org/abs/2610.23443)

    ScoutRoute 是一个轻量级后验路由框架，通过廉价的侦察模型单次前向传播提取隐状态与 logit 置信度等内部信号，结合域感知策略为混合专家系统高效选择专家，无需修改或联合训练底层语言模型。

    

    高效的专家路由是混合专家系统中的一个根本性挑战，尤其是对于独立训练的大型语言模型的异构集合。现有的路由方法通常依赖于联合训练的门控网络、基于提示的预测器或计算成本高昂的多模型推理，这限制了它们的灵活性和效率。本文提出了 ScoutRoute，一个轻量级的后验路由框架，它利用从一个廉价的侦察模型单次前向传播中提取的内部语言模型信号来预测专家能力。隐状态表示被用于通过轻量级逻辑回归探针来估计跨模型能力，同时将对数置信度度量纳入域感知路由策略中，从而在不修改底层语言模型的情况下实现高效的专家选择。ScoutRoute 在五个开放权重语言模型上进行了评估…

    arXiv:2610.23443v1 Announce Type: new  Abstract: Efficient expert routing is a fundamental challenge in Mixture-of-Experts (MoE) systems, particularly for heterogeneous collections of independently trained Large Language Mod els (LLMs). Existing routing approaches typically rely on jointly trained gating networks, prompt-based predictors, or computationally expensive multi-model inference, limiting their flexibility and efficiency. This paper presents ScoutRoute, a lightweight post-hoc routing framework that predicts expert competence using internal language model signals extracted from a single forward pass through an inexpensive scout model. Hidden-state representations are used to estimate cross-model competence through lightweight logistic probes, while logit-derived confidence measures are incorporated within a domain-aware routing strategy for efficient expert selec tion without modifying the underlying language models. ScoutRoute isevaluatedon fiveopen-weightlanguagemodels acros
    
[^33]: 从观测数据中进行生成式策略改进

    Generative Policy Improvement from Observational Data

    [https://arxiv.org/abs/2610.23436](https://arxiv.org/abs/2610.23436)

    本文提出 PolicyLift，一个三阶段框架，通过学习捕捉高层策略的离散动作码，在动作重叠有限的条件下，从观测数据中实现对文本、图像等结构化动作的生成式策略改进。

    

    我们研究如何从观测数据中改进决策策略，以便在动作是复杂结构化对象（如文本消息或图像）时生成更好的动作。该任务面临两大主要挑战：(i) 动作级别的重叠往往太弱，难以支持可靠的离线策略评估与改进；(ii) 生成新的结构化动作通常需要昂贵的训练或微调（例如对大语言模型），并且由于动作表示中的信息丢失，问题定义可能不明确。为同时解决这两个问题，我们提出了 PolicyLift，一个基于观测数据、面向结构化动作的生成式策略改进三阶段框架。在第一阶段，PolicyLift 学习一小组离散动作码，这些动作码捕捉观测动作空间中的高层策略，并经过专门设计以在有限重叠条件下支持策略改进。在第二阶段，它学习一个基于这些动作码的、依赖上下文的策略。在部署时，第三阶段“提升”（原文截断）……

    arXiv:2610.23436v1 Announce Type: new  Abstract: We study how to improve decision policies from observational data to generate better actions when actions are complex, structured objects, such as text messages or images. This task is subject to two major challenges: (i) action-level overlap is often too weak for reliable off-policy evaluation and improvement, and (ii) generating new structured actions often requires expensive training or fine-tuning (e.g., of an LLM) and can be ill-defined due to information loss in action representations. To address both, we propose PolicyLift, a three-stage framework for generative policy improvement with structured actions from observational data. In Stage 1, PolicyLift learns a small set of discrete action codes that capture high-level strategies in the observed action space and are tailored to support policy improvement under limited overlap. In Stage 2, it learns a context-dependent policy over these action codes. At deployment time, Stage 3 "lif
    
[^34]: AWAM：面向鲁棒机器人操作的主动轨迹条件视角想象

    AWAM: Active Trajectory-Conditioned View Imagination for Robust Robot Manipulation

    [https://arxiv.org/abs/2610.23396](https://arxiv.org/abs/2610.23396)

    提出主动世界-动作模型 AWAM，通过主动规划虚拟相机轨迹并利用几何条件化世界模型想象未来观测，生成可复用的空间先验并与实时观测融合，从而在杂乱、遮挡的开放世界中实现更鲁棒的机器人操作。

    

    具身智能的快速发展使得机器人能够日益熟练地遵循语言指令并操作多样化物体。然而，在结构化环境中取得的优异成果并不能自动迁移到开放世界场景中——在开放世界中，杂乱、遮挡和不断变化的视角可能掩盖执行动作所需的空间关系。为了弥合这一感知差距，本文提出了一种虚拟主动感知范式。AWAM（主动世界-动作模型，Active World-Action Model）会在行动之前主动规划观察位置，并想象沿规划轨迹将会看到的画面。该模型预测与任务相关的虚拟相机运动，并将其作为结构化查询输入到几何条件化的世界模型中，从而生成可复用的操作空间先验；该先验随后被压缩为固定记忆，并通过交叉注意力与实时观测相融合，用于基于扩散模型的动作预测。AWAM 以 75……的成功率取得了具有竞争力的表现（原文摘要在此处截断）。

    arXiv:2610.23396v1 Announce Type: cross  Abstract: Rapid progress in embodied intelligence has enabled increasingly capable robots to follow language instructions and manipulate diverse objects. However, strong results in structured environments do not automatically transfer to open-world conditions, where clutter, occlusion, and changing viewpoints can obscure the spatial relationships required for action. To address this perceptual gap, a virtual active-perception paradigm is introduced. AWAM, an Active World-Action Model actively plans where to observe and imagines what will be seen along the planned trajectory before acting. Task-related virtual camera motion is predicted and used as a structured query into a geometry-conditioned world model, producing a reusable spatial prior for manipulation, which is later compressed into a fixed memory and fused with live observations through cross-attention for diffusion-based action prediction. AWAM achieves competitive success rates with 75.
    
[^35]: 监控有效性失效：演化框架边界处的LLM安全证据路径

    Monitor-Validity Failures: LLM Safety Evidence Paths at Evolving Framework Boundaries

    [https://arxiv.org/abs/2610.23362](https://arxiv.org/abs/2610.23362)

    该论文提出“金丝雀契约”测试协议，用于发现LLM安全框架中的一种新型故障——“监控有效性失效”，即风险证据已被检测器识别却在框架内部丢失、未能送达阻断决策点——并在11个LLM/智能体安全项目组中确认了18个此类证据路径失效案例。

    

    现代大语言模型应用依赖运行时监控器来检查工具调用、执行轨迹、数据流、检测器输出和验证器输入。这带来了一个接口可靠性问题：框架路径可能仍在正常运行，但与策略相关的证据却从未到达其决策点。在一个代表性案例中，正则表达式检测器返回了阳性匹配结果，但框架在阻断决策之前将该结果搁置了。这不同于越狱攻击和检测器误判，因为证据已经被识别，却与执行机制断开了连接。我们使用“金丝雀契约”来研究这种修复边界特征：这是一种父级/固定确认单元，它仅在发生变更的证据表面上放置已知载荷，并记录监控器的可见性以及动作边界情况下的下游决策。将该协议应用于公开的框架修复中，我们在11个大语言模型或智能体安全项目组中确认了18个证据路径失效，并覆盖四个实现……（原文摘要在此处截断）

    arXiv:2610.23362v1 Announce Type: new  Abstract: Modern LLM applications rely on runtime monitors that inspect tool calls, traces, streams, detector outputs, and verifier inputs. This creates an interface-reliability problem: a framework path may remain operational while policy-relevant evidence never reaches its decision point. In one representative case, a regex detector returned a positive match, but the framework stranded the result before the block decision. This differs from jailbreaks and detector errors because the evidence is already recognized yet disconnected from enforcement. We study this repair-boundary signature with canary contracts: parent/fixed confirmation units that place a known payload only on the changed evidence surface and record monitor visibility together with the downstream decision for action-boundary cases. Applying the protocol to public framework repairs, we confirm 18 evidence-path failures across 11 LLM or agent safety project groups and four implement
    
[^36]: OliO：基于常微分方程的线性转移算子用于自监督时间序列预测

    OliO: ODE-based Linear Transition Operator for Self-Supervised Time Series Forecasting

    [https://arxiv.org/abs/2610.23345](https://arxiv.org/abs/2610.23345)

    提出OliO，一种基于线性常微分方程转移算子的即插即用自监督学习方法，将时间序列的双向时间流建模为连续且结构一致的动力系统，从而提升时间序列预测性能。

    

    自监督表征学习通过从未标注数据中捕捉鲁棒特征，推动了时间序列预测的发展。然而，现有的对比学习和掩码方法往往难以显式建模潜在的时间流，要么将时间滞后视为噪声，要么破坏了对预测至关重要的固有时间动态。为了解决这些局限性，我们提出了OliO（基于常微分方程的线性转移算子），这是一种即插即用的自监督学习方法，它将双向时间流学习为连续且结构一致的动力系统。OliO引入了从线性常微分方程（ODE）导出的转移算子，以在任意时间偏移下对齐潜在表征。通过对转移矩阵施加严格上三角约束，我们在数学上确保了数值稳定性，并强制执行一个鲁棒的多项式边界，有效地…

    arXiv:2610.23345v1 Announce Type: new  Abstract: Self-supervised representation learning has advanced time series forecasting by capturing robust features from unlabeled data. However, existing contrastive and masking-based methods often struggle to explicitly model the underlying temporal flow, either by treating time lag as noise or by disrupting the inherent temporal dynamics essential for forecasting. To address these limitations, we propose OliO (**O**DE-based **Li**near Transition **O**perator), a plug-and-play self-supervised learning method that learns bidirectional temporal flows as a continuous and structurally consistent dynamical system. OliO introduces a transition operator derived from linear Ordinary Differential Equations (ODEs) to align latent representations across arbitrary time shifts. By imposing a strictly upper triangular constraint on the transition matrix, we mathematically ensure numerical stability and enforce a robust polynomial boundary that effectively cap
    
[^37]: 语言模型学会改进语言模型

    Language Models Learn to Improve Language Models

    [https://arxiv.org/abs/2610.23344](https://arxiv.org/abs/2610.23344)

    该论文提出LM2LM方法，通过强化学习训练大语言模型生成有效的语言模型改进提案，并利用廉价的代理反馈解决真实训练反馈成本高昂以及分布偏移下反馈不可靠这两大难题，为递归自我改进铺路。

    

    大语言模型（LLM）正被越来越多地用于自动化AI研究，包括语言模型（LM）本身的开发，这为递归自我改进提供了一条潜在路径。然而，将实验反馈转化为一种可学习的、能够跨模型提出有效改进的能力面临两大挑战。首先，使用真实反馈来训练这种能力代价高昂：强化学习（RL）需要评估数以万计的改进提案，而每个提案可能都需要一次完整的模型训练运行。即使每次运行仅需十个GPU小时，仅奖励收集这一项就会消耗数十万GPU小时，这促使人们寻求更便宜的替代反馈。其次，当提案分布发生偏移时，替代反馈可能变得不可靠，因为策略优化会偏向那些被高估的提案。为应对这些挑战，我们提出了LM2LM，该方法通过强化学习训练LLM来生成有效的LM改进提案。为了提供可扩展的反馈，我们使用了一种……（摘要在此处截断）

    arXiv:2610.23344v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to automate AI research, including language-model (LM) development, offering a potential path toward recursive self-improvement. However, turning experimental feedback into a learned ability to propose effective improvements across models faces two challenges. First, training such an ability with real feedback is expensive: reinforcement learning (RL) requires evaluating tens of thousands of proposals, each potentially requiring a full model-training run. Even at ten GPU-hours per run, reward collection alone would consume hundreds of thousands of GPU-hours, motivating a cheaper surrogate. Second, surrogate feedback can become unreliable as policy optimization favors overestimated proposals when the proposal distribution shifts. To address these challenges, we propose LM2LM, which trains LLMs with RL to generate effective LM improvement proposals. To provide scalable feedback, we use a
    
[^38]: 接口漂移下LLM智能体的鲁棒函数调用

    Robust Function Calling for LLM Agents Under Interface Drift

    [https://arxiv.org/abs/2610.23342](https://arxiv.org/abs/2610.23342)

    提出了一种无需训练的轻量级修复层 SchemaShield-Lite，通过模式校验、候选集重试和一次性受限修复，在接口漂移场景下显著恢复LLM智能体的函数调用准确率。

    

    工具增强的语言模型智能体依赖接口文本：工具描述、参数模式（schema）以及推理时展示的候选工具列表。即使底层工具语义保持不变，这些接口在部署后也经常发生变化，这给函数调用系统带来了实际的可靠性问题。我们将这种故障模式称为“接口漂移”并加以研究，评估一个轻量级、无需训练的修复层能否在不重新训练基础模型的情况下恢复鲁棒性。我们的方法 SchemaShield-Lite 结合了模式校验、针对缺失工具名称的候选集重试以及一次性受限修复。在 Qwen3.5-9B 上，该方法将 DICE-300 的准确率从 0.717 提升至 0.813，将 BFCL-200 的准确率从 0.580 提升至 0.805，分别恢复了 19/22 和 33/45 个由漂移引起的失败案例，且未观察到修复带来的负面影响。简单的重试机制无法解释这些提升。消融实验表明，候选集漂移（尤其是干扰性工具）是造成大部分观察到的失败的原因……

    arXiv:2610.23342v1 Announce Type: new  Abstract: Tool-augmented language-model agents rely on interface text: tool descriptions, argument schemas, and the candidate tool list shown at inference time. These interfaces often change after deployment even when the underlying tool semantics remain fixed, creating a practical reliability problem for function-calling systems. We study this failure mode as interface drift and evaluate whether a lightweight, training-free repair layer can recover robustness without retraining the base model. Our method, SchemaShield-Lite, combines schema validation, candidate-set retry for missing tool names, and one-shot constrained repair. On Qwen3.5-9B, it raises DICE-300 accuracy from 0.717 to 0.813 and BFCL-200 accuracy from 0.580 to 0.805, recovering 19/22 and 33/45 drift-induced failures with no observed repair harms. Naive retry does not explain the gains. Ablations show that candidate-set drift, especially distractor tools, accounts for most observed d
    
[^39]: AgentOnto：面向企业数据分析的本体引导大语言模型智能体

    AgentOnto: Ontology-Guided LLM Agents for Enterprise Data Analytics

    [https://arxiv.org/abs/2610.23317](https://arxiv.org/abs/2610.23317)

    AgentOnto 通过将异构企业知识组织成本体来引导大语言模型智能体执行企业数据分析，显著优于现有方法，并通过实际部署验证了其实用价值。

    

    企业数据分析要求智能体能够将分析请求与分布在数据库、业务文档和现有示例任务中的知识相连接。如果缺乏对这些知识的显式表示，智能体可能会产生错误的分析结果，并且无法可靠地整合反馈。我们提出了 AgentOnto，一个面向企业数据分析的本体引导大语言模型智能体。AgentOnto 将异构的企业知识组织成一个本体，该本体直接引导分析过程并整合经过确认的反馈。迭代式本体准备通过自生成的分析任务进一步优化本体。我们在两个公开数据源上对 AgentOnto 进行了评估，结果表明其显著优于代表性方法，模块级评估也证实了本体构建、分析和更新机制的有效性。实际部署进一步展示了其在企业分析中的实用价值。

    arXiv:2610.23317v1 Announce Type: new  Abstract: Enterprise data analysis requires agents to connect analytical requests with knowledge distributed across databases, business documents, and existing example tasks. Without an explicit representation of this knowledge, an agent may produce incorrect analytical results and cannot reliably incorporate feedback. We present AgentOnto, an ontology-guided LLM agent for enterprise data analytics. AgentOnto organizes heterogeneous enterprise knowledge into an ontology that directly guides analysis and incorporates confirmed feedback. Iterative Ontology Preparation further improves the ontology through self-generated analysis tasks. We evaluate AgentOnto on two public data sources. AgentOnto substantially outperforms representative methods, while module-level evaluations confirm the effectiveness of ontology construction, analysis, and updating. A real-world deployment further demonstrates its practical value in enterprise analytics. These result
    
[^40]: 从记忆命中到可回答证据：面向长期智能体记忆的查询条件化记忆编译

    From Memory Hits to Answerable Evidence: Query-Conditioned Memory Compilation for Long-Term Agent Memory

    [https://arxiv.org/abs/2610.23313](https://arxiv.org/abs/2610.23313)

    提出查询条件化的记忆编译框架MemCompile，通过地址平面定位证据源、证据平面在有限预算内将检索到的记忆命中编译为可直接回答问题的证据上下文，解决了长期智能体记忆中记忆寻址与证据呈现耦合所导致的上下文冗余或不完整问题。

    

    长期智能体记忆需要恢复分散在多轮对话与多个会话中的与答案相关的证据。现有系统日益采用结构化记忆与自适应检索，但往往将检索或扩展后的单元直接传递给读取器，使记忆寻址与证据呈现耦合在一起，导致最终上下文冗余或不完整。为解决这一问题，我们提出了MemCompile，一个面向长期对话记忆的查询条件化证据编译框架。给定一个查询，地址平面通过紧凑的源关联描述符和容量感知的Cell定位相关支持，同时由规范化边界将异构的检索命中解析为稳定的SourcePacket标识。随后，其证据平面在互补的证据包之间分配有界的证据预算，接纳有限的额外支持，并将与每个被选中数据包相关联的保留声明渲染为可直接用于回答的上下文。

    arXiv:2610.23313v1 Announce Type: new  Abstract: Long-term agent memory requires recovering answer-relevant evidence dispersed across turns and sessions. Existing systems increasingly use structured memories and adaptive retrieval, yet often pass retrieved or expanded units directly to the reader, coupling memory addressing with evidence presentation and leaving the final context redundant or incomplete. To address this gap, we propose MemCompile, a query-conditioned evidence compilation framework for long-term conversational memory. Given a query, the Address Plane locates relevant support through compact source-linked descriptors and capacity-aware Cells, while a normalization boundary resolves heterogeneous retrieval hits to stable SourcePacket identities. Its Evidence Plane then allocates a bounded evidence budget across complementary packets, admits limited additional support, and renders the retained claims associated with each selected packet into an answer-ready context. MemCom
    
[^41]: 自反思思维链（SRCoT）：基于边缘大语言模型的闭环“检索—反思—修订”流水线实现有证据支撑的文本生成

    Self-Reflective Chain-of-Thought (SRCoT): Evidence-Based Text Generation via a Closed-Loop Retrieve--Reflect--Revise Pipeline on Edge LLMs

    [https://arxiv.org/abs/2610.23311](https://arxiv.org/abs/2610.23311)

    该论文提出自反思思维链（SRCoT）架构，通过闭环“检索—反思—修订”流程和带来源标签的多层存储机制，使边缘设备上的大语言模型能够生成有证据支撑、可追溯的文本。

    

    大语言模型（LLM）能够生成流畅的输出，但这些输出可能缺乏来源支持，或与本地运行约束相冲突。我们提出了自反思思维链（SRCoT），这是一种模块化架构，将生成过程组织为一个闭环的“检索—反思—修订”循环。SRCoT将环境状态、持久知识、积累经验和短期记忆分离到带有来源标签的存储中。BM25检索提供局部上下文；后续处理轮次会呈现中间草稿、缺失信息标记、后续查询以及可选的“论断—证据”映射，这些产物为检索、修订、澄清或弃答决策提供依据。我们通过在两个完全合成的场景中进行基于轨迹的定性案例研究来考察该架构：急救医疗服务和Munich-Sim（一个应急行动模拟环境）。在配备128 GB内存的Apple M4系列笔记本电脑上本地运行的原型展示了带有来源标签的数据……（摘要原文在此处截断）

    arXiv:2610.23311v1 Announce Type: new  Abstract: Large language models (LLMs) can generate fluent outputs that lack source support or conflict with local operational constraints. We introduce Self-Reflective Chain-of-Thought (SRCoT), a modular architecture that organizes generation as a closed Retrieve--Reflect--Revise loop. SRCoT separates environment state, persistent knowledge, accumulated experience, and short-term memory into provenance-labeled stores. BM25 retrieval supplies local context; subsequent passes expose intermediate drafts, missing-information markers, follow-up queries, and optional claim--evidence mappings. These artifacts inform retrieval, revision, clarification, or abstention decisions. We examine the architecture through qualitative, trace-based case studies in two fully synthetic settings: emergency medical services and Munich-Sim, an emergency-operations simulation. Local prototype runs on an Apple M4-family laptop with 128 GB memory illustrate source-labeled d
    
[^42]: MobileInject：评估基于多模态大语言模型的移动智能体在跨模态一致性注入下的“推理到行动”安全性

    MobileInject: Evaluating Reasoning-to-Action Security of MLLM-Based Mobile Agents under Cross-Modal Consistency Injection

    [https://arxiv.org/abs/2610.23307](https://arxiv.org/abs/2610.23307)

    该论文提出了首个针对移动智能体的多模态注入攻击安全基准 MobileInject，通过涵盖 11 种场景、9 类安全风险和 108 个可执行风险实例的高保真交互式框架，系统评估了基于多模态大语言模型的移动智能体在跨模态一致性注入下从“推理”到“行动”的安全风险。

    

    基于多模态大语言模型（MLLM）的移动智能体正迅速从对话式助手演变为能够在智能手机环境中进行感知、推理并采取行动的自主系统。这种能力同时也带来了新的安全风险：攻击者可以在跨应用的移动内容中嵌入恶意目标，并通过间接多模态注入来操纵智能体的行为。尽管提示注入攻击已在桌面端和网页环境中得到研究，但其在移动生态系统中的跨模态特性仍然鲜为人知，现有的安全分析也依然有限。为填补这一空白，我们提出了 MobileInject，这是首个用于评估移动智能体抵御多模态注入攻击及“推理到行动”安全风险的基准。MobileInject 提供了一个高保真的交互式移动评估框架，涵盖 11 种场景、9 类安全风险和 108 个可执行的风险实例。它还进一步引入了跨模态……（摘要在此处截断）

    arXiv:2610.23307v1 Announce Type: cross  Abstract: Multimodal large language model (MLLM)-based mobile agents are rapidly evolving from conversational assistants into autonomous systems that perceive, reason over, and act on smartphone environments. This capability also exposes a new security risk: adversaries can embed malicious objectives in cross-application mobile content and steer agent behavior through indirect multimodal injection. Although prompt injection has been studied in desktop and web settings, its cross-modal properties in mobile ecosystems remain poorly understood, and existing security analyses are still limited. To close this gap, we introduce MobileInject, the first benchmark for evaluating mobile agents against multimodal injection attacks and reasoning-to-action security risks. MobileInject provides a high-fidelity interactive mobile evaluation framework covering 11 scenarios, 9 security-risk categories, and 108 executable risk instances. It further introduces Cro
    
[^43]: SimOPD：面向在线策略蒸馏的关系知识迁移

    SimOPD: Relational Knowledge Transfer for On-Policy Distillation

    [https://arxiv.org/abs/2610.23306](https://arxiv.org/abs/2610.23306)

    提出SimOPD方法，通过成对记忆库将教师内部表示转化为响应相似性分布，并让学生在自身表示空间内进行匹配，从而实现无需跨模型投影的在线策略蒸馏关系知识迁移。

    

    在线策略蒸馏（OPD）中的一个关键挑战是如何从教师知识中提取有效的监督信号。基于词元概率的监督可能无法充分利用教师内部知识，而直接的隐藏状态匹配则会迫使学生模型转向教师的特征空间，并使跨架构迁移变得复杂。受对比学习利用实例间关系的启发，我们提出SimOPD，通过响应相似性来迁移教师的内部知识。具体而言，SimOPD使用由学生生成响应构成的成对记忆库，从教师的内部表示中提取监督信号，将其表示为相似性分布。随后，学生在自己的表示空间内匹配这些分布，从而保留了表示自由度，并消除了对跨模型投影的需求。大量实验表明，SimOPD相对于最强基线将整体平均性能最多提升了2.6和3.1。

    arXiv:2610.23306v1 Announce Type: new  Abstract: A key challenge in on-policy distillation (OPD) is extracting effective supervision from teacher knowledge. Supervision from token probabilities may underutilize internal knowledge, while direct hidden-state matching forces the student to shift to the teacher's feature space and complicates cross-architecture transfer. Inspired by contrastive learning's use of relationships between instances, we propose \textbf{SimOPD} to transfer teacher internal knowledge through response similarities. Specifically, SimOPD uses paired memory banks of student-generated responses to extract supervision from the teacher's internal representations as similarity distributions. The student then matches these distributions within its own representation space, retaining representational freedom and removing the need for cross-model projection. Extensive experiments show that SimOPD improves the overall average over the strongest baselines by up to 2.6 and 3.1 
    
[^44]: 面向不完美提示词的设计：基于知识图谱的LLM企业数据助手的仿真评估

    Designing for Imperfect Prompts: A Simulation-Based Evaluation of an LLM-Based Enterprise Data Assistant Using Knowledge Graphs

    [https://arxiv.org/abs/2610.23304](https://arxiv.org/abs/2610.23304)

    本研究针对业务用户提出的不完美提示词，设计并比较了基于模式与基于知识图谱两种语义中介方法，仿真评估表明基于知识图谱的中介方法在精确率、召回率、F1值和保留用户意图方面表现更优。

    

    使用大语言模型（LLM）的企业数据助手有望实现组织内数据访问的民主化。然而，在我们与某行业合作伙伴CFO部门数据湖的实际部署中，我们观察到业务用户经常提出不完美的提示词，其中包含领域特定词汇，且缺乏模式引用或结构信息。本设计科学研究探讨了通过语义中介层自动解析这些提示词。我们设计并评估了两种设计变体：通过语义相似性实现的基于模式的中介方法，以及利用从数据湖自动构建的知识图谱实现的基于知识图谱的中介方法。基于真实数据的仿真评估（辅以LLM-as-a-Judge评估和人工验证）表明，基于知识图谱的中介方法在精确率、召回率、F1值以及保留用户意图方面均优于基于模式的方法。我们提出了一个设计原则……（摘要在此处截断）

    arXiv:2610.23304v1 Announce Type: cross  Abstract: Enterprise data assistants using large language models (LLMs) promise democratized data access in organizations. Yet in our real-world deployment within the CFO unit's data lake at an industry partner, we observed that business users frequently pose imperfect prompts that include domain-specific vocabulary and lack schema references or structural information. This Design Science Research study examines automatic resolution using a semantic mediation layer. We design and evaluate two design variants: schema-based mediation via semantic similarity, and knowledge graph-based mediation leveraging an automatically constructed knowledge graph from the data lake. Our simulation-based evaluation using real-world data, supported by LLM-as-a-Judge assessments and human validation, shows that knowledge graph-based mediation outperforms the schema-based approach in Precision, Recall, F1, and preserving user intent. We propose a design principle fo
    
[^45]: Optimist Deep Research：一种面向自主文献研究的迭代式、以状态为中心的智能体

    Optimist Deep Research: An Iterative, State-Centric Agent for Autonomous Literature Research

    [https://arxiv.org/abs/2610.23299](https://arxiv.org/abs/2610.23299)

    该论文提出了开源智能体 Optimist Deep Research，将文献研究建模为在显式可变研究状态上的迭代式证据获取过程，通过子问题树分解、多源查询转换、保留来源的文档去重注册表以及引文滚雪球扩展，实现了更接近真实研究过程的自主文献研究。

    

    大多数检索增强系统将文献搜索视为单一检索步骤加上后续的摘要总结。而真实的文献研究并非如此：研究者对问题的理解会随着阅读而不断变化，新的子问题会不断出现，证据需要被权衡而不是仅仅被收集。我们提出了 Optimist Deep Research，一个开源智能体，它将文献研究作为在显式、可变的研究状态之上运行的迭代式证据获取过程。研究问题被分解为子问题树，每个子问题由搜索意图提供服务，这些搜索意图被转换为针对 arXiv、OpenAlex、Crossref、PubMed、Semantic Scholar 以及网络的特定来源查询。检索结果通过保守的四层去重层级合并到一个规范的、保留来源信息的文档注册表中。该注册表通过引文滚雪球不断扩展，直到满足饱和标准，随后进行筛选、排序并分配给各子问题……

    arXiv:2610.23299v1 Announce Type: new  Abstract: Most retrieval-augmented systems treat literature search as a single retrieval step followed by summarization. Real literature research works differently: the investigator's understanding of the question changes as they read, new subquestions appear, and evidence is weighed, not just collected. We present Optimist Deep Research, an open-source agent that runs literature research as an iterative evidence-acquisition process over an explicit, mutable research state. A research question is broken into a tree of subquestions, each served by search intents that are translated into source-specific queries over arXiv, OpenAlex, Crossref, PubMed, Semantic Scholar and the web. Results are merged into a canonical, provenance-preserving document registry through a conservative four-tier deduplication hierarchy. The registry is expanded by citation snowballing until a saturation criterion is met, and then screened, ranked and allocated to subquestio
    
[^46]: FlashReFocus：秒级交互式图像重新对焦

    FlashReFocus: Interactive Image Refocusing in Seconds

    [https://arxiv.org/abs/2610.23297](https://arxiv.org/abs/2610.23297)

    FlashReFocus 对每张照片仅运行一次扩散模型来恢复全清晰图像，再用基于显式散焦图条件的轻量级网络渲染每次编辑，从而实现秒级的交互式图像重新对焦。

    

    单图像重新对焦是在拍摄后改变照片的焦平面和景深。这本质上是交互式的：用户会在同一张图像上尝试多个焦点和模糊强度，因此每次编辑都应该快速且只改变模糊。然而，近期的生成式方法为每次编辑都重新运行大型扩散模型。我们观察到这两个子问题在性质上有所不同：恢复全清晰图像是一个病态问题，需要生成先验；而渲染模糊在很大程度上由深度和焦平面决定。因此，我们提出 FlashReFocus，它对每张照片只运行一次扩散过程，然后用轻量级网络渲染每次编辑。为解决病态恢复问题，带有解码器精炼器的单步扩散去模糊器在平衡保真度和感知质量的同时高效恢复精细细节。为实现精确控制，渲染器以显式散焦图为条件，并使用散焦偏置注意力，这使得……

    arXiv:2610.23297v1 Announce Type: cross  Abstract: Single-image refocusing changes the focal plane and depth of field of a photograph after capture. It is inherently interactive: users try many focal points and blur strengths on one image, so each edit should be fast and change only the blur. Recent generative methods, however, rerun large diffusion models for every edit. We observe that the two sub-problems differ in kind: recovering the all-in-focus image is ill-posed and needs a generative prior, whereas rendering blur is largely determined by depth and the focal plane. We therefore propose FlashReFocus, which runs diffusion once per photograph and renders every edit with a lightweight network. To address the ill-posed restoration, a one-step diffusion deblurrer with a decoder refiner recovers fine detail efficiently while balancing fidelity and perceptual quality. For precise control, the renderer is conditioned on an explicit defocus map and uses defocus-bias attention, which lets
    
[^47]: CoTA++：从信息流视角理解扩散多模态大语言模型中的“重复诅咒”

    CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective

    [https://arxiv.org/abs/2610.23290](https://arxiv.org/abs/2610.23290)

    本文从信息流视角揭示了近似缓存导致扩散多模态大语言模型产生“重复诅咒”的单步、跨步与跨层级三类信息流扰动，并提出无需训练、即插即用的 CoTA++ 框架来逐一消除这些扰动。

    

    基于扩散的多模态大语言模型（dMLLMs）通过带双向注意力的迭代去噪进行解码，这使得标准的自回归键值缓存无法适用。因此，实际部署通常依赖跨去噪步骤复用中间状态的近似缓存。然而，这种缓存常常导致模型在相邻位置连续重复某个词元，形成破坏句子的重复串，我们将这一失效现象称为“重复诅咒”。通过追踪上下文词元的信息流，我们揭示了三类扰动：在单步之内，缓存分散了将某一位置锚定于附近上下文词元的注意力；在步骤之间，基于相似度的刷新机制无法感知状态的陈旧程度，导致某些位置在过时的键值状态上进行解码；在层级之间，出现重复之处上下文的深层熵无法收敛。据此，我们提出了 CoTA++——一个无需训练、即插即用的框架，能够针对上述每一类扰动进行纠正（摘要原文在此处截断）。

    arXiv:2610.23290v1 Announce Type: new  Abstract: Diffusion-based multimodal large language models (dMLLMs) decode by iterative denoising with bidirectional attention, which precludes the standard autoregressive key--value cache. Practical deployments therefore rely on approximate caching that reuses intermediate states across denoising steps. Such caching, however, often makes a response repeat one token at adjacent positions in a run that breaks the sentence, a failure we term the \emph{Repeat Curse}. Tracing the context-token information flow reveals three disruptions. Within a step, caching disperses the attention that anchors a position on nearby context tokens. Across steps, similarity-based refresh is not staleness-aware, so some positions are decoded against stale key--value states. Across layers, the deep-layer entropy of the context fails to converge where repetition arises. We accordingly propose CoTA++, a training-free, plug-and-play framework that counteracts each disruptio
    
[^48]: Annotrace：一个用于智能体轨迹步骤级标注的工作台

    Annotrace: A Workbench for Step-Level Annotation of Agent Traces

    [https://arxiv.org/abs/2610.23284](https://arxiv.org/abs/2610.23284)

    Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。

    

    每一项研究LLM智能体为何失败的工作，都始于对手工标注轨迹，而每一项这样的研究都会构建自己的一次性工具。可观测性平台能够记录智能体轨迹，却不提供结构化标注；标注工具可以标注文本，却没有“步骤”、任务“委派”或“工具调用”的概念，导致最小可标注的执行单元不明确。Annotrace 可以读取六种轨迹格式，包括 OpenTelemetry GenAI、LangSmith 和 Langfuse，并将每次运行转换为有序的、嵌套的步骤列表。标注者编写自己的标注手册；一个前沿模型会根据标注者自己轨迹的样本提出初始标注集，先行作答，并被问到与人类完全相同的问题，从而使标注成为一个人机协作的过程。Annotrace 能够衡量任意两个标注者（无论是人类还是AI）之间的一致性。在一个多智能体问答语料库上，它精确复现了此前一次人工标注工作的235个标注单元，并且两个模型……（摘要在此处被截断）

    arXiv:2610.23284v1 Announce Type: new  Abstract: Every study of why LLM agents fail begins by labelling traces by hand, and every such study builds its own throwaway tool. Observability platforms record agent traces but offer no structured labelling; annotation tools label text but have no notion of a step, a delegation or a tool call, leaving it unclear what the smallest unit of execution to label should be. Annotrace reads six trace formats, including OpenTelemetry GenAI, LangSmith and Langfuse, and turns each run into an ordered, nested list of steps. The annotator writes their own labelling codebook; a frontier model proposes a starting set from a sample of their own traces, answers first, and is asked exactly the questions the human is asked, making annotation a joint human-AI process. Annotrace measures agreement between any two annotators, human or AI. On a multi-agent question-answering corpus it reproduces a previous manual effort's 235 annotation units exactly, and two models
    
[^49]: 超越边际影响：面向LLM智能体的交互感知记忆选择

    Beyond Marginal Influence: Interaction-Aware Memory Selection for LLM Agents

    [https://arxiv.org/abs/2610.23266](https://arxiv.org/abs/2610.23266)

    该论文提出JointCoreBench基准和JointCore方法，突破传统基于边际影响的记忆排序方式，通过感知记录间交互并采用成本排序的组删除策略，实现LLM智能体最小成本充分记忆子集的选择，达到100%恢复率并将子集评估次数减少62.9%。

    

    LLM智能体的记忆选择通常使用基于记录相关性或个体影响的检索排序。这种排序在存在冗余证据或联合必要证据时无法识别出充分的子集。我们研究逐决策记忆充分性问题：即选择一个能保持完整记忆行为的小子集。为了将搜索误差与模型误差分离，我们提出了JointCoreBench——一个包含九类交互族、六个工作流领域、自然语言决策以及精确充分集预言的基准测试，并提出了JointCore——一种按相关性成本排序、先测试记录块再测试单条记录的组删除基线方法。在6,912个精确条件下，按成本排序的JointCore在100.0%的情况下恢复了最小成本的充分集，将子集评估次数从30.00降至11.13（降低62.9%），并实现了87.8%的活跃记忆节省。在四个12B-70B模型家族和每个模型432个任务上，其未加权平均值为10.28，对比16.00次模型查询和50.6%。

    arXiv:2610.23266v1 Announce Type: cross  Abstract: LLM-agent memory selection often uses retrieval rankings based on record relevance or individual influence. Such rankings cannot identify sufficient subsets under redundant or jointly necessary evidence. We study per-decision memory sufficiency: selecting a small subset that preserves full-memory behavior. To separate search error from model error, we introduce JointCoreBench, a benchmark with nine interaction families, six workflow domains, natural-language decisions, and exact sufficient-set oracles, and JointCore, a relevance cost-ordered group-deletion baseline that tests blocks before individual records. On 6,912 exact conditions, cost-ordered JointCore recovers a minimum-cost sufficient set in 100.0% of cases, reduces subset evaluations from 30.00 to 11.13 (62.9%), and achieves 87.8% active-memory savings. Across four 12B-70B model families and 432 tasks per model, its unweighted mean is 10.28 versus 16.00 model queries and 50.6%
    
[^50]: 推理模型在自我指令下能够控制其思维链

    Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To

    [https://arxiv.org/abs/2610.23257](https://arxiv.org/abs/2610.23257)

    推理模型并非天生无法控制其思维链，将约束作为自我指令置于CoT内部时，模型的可控性显著优于在用户提示中指定约束的方式。

    

    思维链（CoT）可控性，即推理模型控制其思维链属性的能力，对模型的有用性和可靠监督都具有重要意义。近期研究通过测量模型是否遵循用户提示中给出的CoT约束来评估这一能力，并报告各模型均表现出极低的可控性。我们认为，这种低可控性可能并非反映推理模型天生无法控制其CoT，而是源于在用户提示中简单直接地指定约束的方式。为验证这一点，我们提出了自我指令CoT可控性（Self-Instructed CoT Controllability）：一种互补的评估方法，它将约束作为自我指令置于模型的CoT内部，并测量模型遵循该约束的能力。我们表明，在自我指令条件下，模型实现的可控性显著高于标准用户指令下的水平。即使仅是提醒模型遵循用户约束的通用提示，也能大幅提升可控性。

    arXiv:2610.23257v1 Announce Type: new  Abstract: Chain-of-thought (CoT) controllability, the ability of reasoning models to control properties of their CoT, is consequential for both their usefulness and reliable oversight. Recent work evaluates this capability by measuring whether models follow CoT constraints given in the user prompt, and reports very low controllability across models. We argue that this low controllability may not reflect an inherent inability of reasoning models to control their CoT, but instead result from naively specifying constraints in the user prompt. To test this, we introduce Self-Instructed CoT Controllability: a complementary evaluation that places the constraint as a self-instruction inside the model's CoT and measures its ability to follow it. We show that, under self-instruction, models achieve significantly higher controllability than with standard user instructions. Even a generic reminder to follow the user's constraint substantially improves contro
    
[^51]: 超越奖励评分：面向多模态响应生成的在线多智能体偏好学习

    Beyond Reward Scoring: Online Multi-Agent Preference Learning for Multimodal Response Generation

    [https://arxiv.org/abs/2610.23256](https://arxiv.org/abs/2610.23256)

    提出PrefMind在线双智能体强化学习框架，通过响应生成器与偏好评论家的分析讨论和人类标注监督，让模型将偏好理解内化为自身能力，从而超越传统奖励评分方式改进多模态响应生成。

    

    多模态偏好对齐通常遵循基于评分的RLHF流程：首先训练一个奖励模型来学习人类偏好，然后用它对在线策略生成的响应进行评分，并优化响应模型的生成。这种做法将偏好监督转化为用于生成优化的标量信号：模型可能学会生成具有更高偏好分数的答案，但很少被显式训练去学习人类偏好背后的原因。因此，我们提出疑问：将偏好理解作为响应模型的内部能力，能否改善其生成效果。为了研究这一问题，我们提出了PrefMind，一个在线双智能体强化学习框架，包含两个可训练的智能体：响应生成器和偏好评论家。给定一对候选响应，两个智能体会分析并讨论哪个响应更符合人类偏好，人类标注则用于监督它们的判断和修正。（原文摘要在此处被截断）

    arXiv:2610.23256v1 Announce Type: new  Abstract: Multimodal preference alignment often follows a score-based RLHF pipeline: a reward model is first trained to learn human preferences, then used to score on-policy responses and optimize the response model's generation. This turns preference supervision into a scalar signal for generation: a model may learn to produce answers with higher preference scores, but is rarely trained explicitly to learn the reasons underlying human preferences. We therefore ask whether making preference understanding an internal capability of the response model can improve its generation. To investigate this question, we propose \textbf{PrefMind}, an online dual-agent reinforcement learning framework with two trainable agents: a response Generator and a preference Critic. Given a pair of candidate responses, the agents analyze and discuss which response better aligns with human preference. Human labels supervise their judgments and corrections. For pairs on wh
    
[^52]: 防止智能体AI在光网络故障排除中的数据幻觉

    Preventing Data Hallucinations in Agentic AI for Optical Network Troubleshooting

    [https://arxiv.org/abs/2610.23252](https://arxiv.org/abs/2610.23252)

    该论文提出了一种将数据引用集成到智能体架构中的AI系统，用于光通信网络的自动化故障排除，有效防止数据幻觉并提升了根因识别与推理的可解释性和可信度。

    

    本文演示了一种用于光通信网络自动化故障排除的智能体AI系统，该系统将数据引用集成到智能体架构中，以防止数据幻觉的产生，从而提高了根因识别与推理的可解释性和可信度。

    arXiv:2610.23252v1 Announce Type: cross  Abstract: An agentic AI system for automated troubleshooting in optical communication networks is demonstrated that integrates data references into the agent architecture to prevent data hallucinations, improving the interpretability and trustworthiness of root-cause identification and reasoning.
    
[^53]: MGE-PO：面向数学推理的熵感知信用分配方法

    MGE-PO: Entropy-Aware Credit Assignment for Mathematical Reasoning

    [https://arxiv.org/abs/2610.23245](https://arxiv.org/abs/2610.23245)

    MGE-PO提出了一种结合验证正确性与生成熵的四象限优势塑形方法，通过对低熵成功给予更大奖励、高熵失败给予更大惩罚来改进GRPO的信用分配机制，显著提升了数学推理性能。

    

    GRPO对同一问题内获得相同奖励的回复赋予相同的组相对优势，从而忽略了这些回复的生成不确定性。我们提出MGE-PO，其核心是四象限（4Q）优势塑形方法，将经验证的答案正确性与rollout生成熵相结合。正确性决定额外校正项的符号，而熵控制其幅度：低熵的成功样本获得更大的奖励，高熵的失败样本受到更大的惩罚。一个基于熵的在线问题课程机制利用缓存的rollout测量结果来支持这一信用分配规则。在Qwen3-8B-Base上经过八个训练周期后的最终检查点处，在三种策略优化设置中加入4Q，在六个基准测试的macro avg@32上带来了3.23-3.69%的绝对提升，在macro pass@32上带来了4.24-11.06%的绝对提升。在Qwen2.5-7B-Base上，完整配置将四个基准测试的macro avg@32从41.5%提升至45.4%。

    arXiv:2610.23245v1 Announce Type: cross  Abstract: GRPO assigns the same group-relative advantage to equally rewarded responses within a question, leaving their generation uncertainty unused. We introduce MGE-PO, centered on four-quadrant (4Q) advantage shaping that combines verified correctness with rollout generation entropy. Correctness determines the sign of an additional correction, while entropy controls its magnitude: lower-entropy successes receive larger bonuses and higher-entropy failures receive larger penalties. An online entropy-based question curriculum supports this credit rule using cached rollout measurements. At the final checkpoint after eight training epochs on Qwen3-8B-Base, adding 4Q across three policy optimization settings yields absolute gains of 3.23-3.69% in six-benchmark macro avg@32 and 4.24-11.06% in macro pass@32. On Qwen2.5-7B-Base, the complete configuration raises four-benchmark macro avg@32 from 41.5% to 45.4%.
    
[^54]: 检索增强生成技术的特征描述与分析

    Characterising and Analysing Retrieval-Augmented Generation Techniques

    [https://arxiv.org/abs/2610.23226](https://arxiv.org/abs/2610.23226)

    提出了一个用于刻画和分析现有检索增强生成（RAG）技术的概念框架，可作为实证评估前的筛选机制，帮助研究者和实践者为特定问题场景选择最合适的RAG技术。

    

    大语言模型（LLM）在众多应用领域的兴起，增加了对检索增强生成（RAG）的需求，以减少幻觉并提升LLM的性能。迄今为止，文献中已提出多种具有出色特性和能力的RAG技术，这使得针对特定问题场景选择最合适的RAG技术成为一项艰巨的任务。因此，我们提出了一个由主要维度和子维度组成的概念框架，用于刻画（定义“它是什么”）和分析（定义“它表现如何”）现有的RAG技术。该概念框架是通过对文献中报道的RAG技术属性进行定性分析而开发的。它可以作为一种经验前筛选机制，帮助研究人员和从业者在进行实证基准测试之前，选择最符合特定用户和任务需求的高适配度RAG候选技术。

    arXiv:2610.23226v1 Announce Type: new  Abstract: The advent of Large Language Models (LLMs) across many application domains has increased the need for Retrieval-Augmented Generation (RAG) to reduce hallucinations and improve LLM performance. Thus far, several RAG techniques with impressive features and capabilities have been proposed in the literature, making it a herculean task to select the most appropriate RAG techniques for specific problem scenarios. Thus, we propose a conceptual framework comprising main dimensions and sub-dimensions for characterising (defining "what it is") and analysing (defining "how well it does") existing RAG techniques. The conceptual framework was developed through a qualitative analysis of the attributes of RAG techniques reported in the literature. It can serve as a pre-empirical filtering mechanism for researchers and practitioners in selecting the highest-fit RAG candidates that meet specific user and task requirements before empirical benchmarking of
    
[^55]: qGAMMA：一个用于在多精度加速器上实现最优DNN映射的框架

    qGAMMA: A Framework for Optimal DNN Mapping on Multi-Precision Accelerators

    [https://arxiv.org/abs/2610.23222](https://arxiv.org/abs/2610.23222)

    qGAMMA是一个利用领域专用遗传算法和基于硬件综合的解析性能模型，为混合精度DNN模型在多精度空间加速器上寻找最优映射与配置的框架，能够在性能、能效和资源利用率之间实现更好的权衡。

    

    本工作介绍了qGAMMA，这是一个灵活的框架，用于探索和优化将混合精度深度神经网络（DNN）模型映射到支持多种定点和浮点精度的通用空间加速器上。qGAMMA使用领域专用的遗传算法和基于硬件综合结果的解析性能模型来探索设计空间。该框架能够识别使目标指标最小化的最优映射和加速器配置。在多种DNN工作负载上的广泛评估表明，qGAMMA为设计混合精度应用提供了有价值的见解，能够在性能、能效和资源利用率之间实现更好的权衡。

    arXiv:2610.23222v1 Announce Type: cross  Abstract: This work describes qGAMMA, a flexible framework for exploring and optimizing the mapping of mixed-precision Deep Neural Network (DNN) models onto general-purpose spatial accelerators that support a range of fixed- and floating-point precisions. qGAMMA navigates the design space using a domain-specific genetic algorithm and an analytical performance model based on hardware synthesis results. The framework identifies the optimal mapping and accelerator configuration that minimizes target metrics. Extensive evaluations across DNN workloads show that qGAMMA offers valuable insights for designing mixed-precision applications, enabling better trade-offs between performance, energy efficiency, and resource utilization.
    
[^56]: 并非所有测试时计算都是证据：大语言模型推理中的模型特异性偏差

    Not All Test-Time Compute Is Evidence: Model-Specific Bias in LLM Reasoning

    [https://arxiv.org/abs/2610.23213](https://arxiv.org/abs/2610.23213)

    论文揭示了同一模型重复推理的误差高度相关、更多计算并不构成更多独立证据，并提出DIA方法——当模型内部阶段出现分歧时，让一小组异构模型独立作答并聚合结果，从而在14个设置中的12个上超越完整思维链，且仅需约17%的激活率和更低的开销。

    

    测试时扩展方法隐含地假设来自同一模型的额外计算能够提供独立的证据。我们证明这一假设在单个推理模型内部并不成立：误差在重复采样之间、不同推理深度之间以及模型自身的自我评估中都是相互关联的，因此来自同一模型的更多计算并不等同于更多证据。我们提出了DIA（分歧触发的独立仲裁）方法，它将中间阶段预测与最终阶段预测之间的分歧视为每个问题上推理不稳定性的信号，并通过让一小组异构模型在看不到任何候选答案的情况下独立求解该问题、然后直接聚合它们的答案来解决这一分歧。在七个基准和两个基础模型上的实验表明，DIA在14个设置中的12个上超越了完整的思维链方法，包括在MuSiQue上取得7.00%的提升，同时仅在约17%的样本上激活，并最多节省47.2%的计算开销。

    arXiv:2610.23213v1 Announce Type: new  Abstract: Test-time scaling methods implicitly assume that additional computation from the same model provides independent evidence. We show that this assumption fails within a single reasoning model: errors are correlated across repeated samples, across reasoning depth, and in the model's own self-evaluation, so more compute from the same model is not more evidence. We introduce DIA (Disagreement-triggered Independent Arbitration), which treats disagreement between an intermediate-stage and the final-stage prediction as a per-question signal of reasoning instability and resolves it by asking a small panel of heterogeneous models to solve the question independently, without seeing either candidate, and aggregating their answers directly. Across seven benchmarks and two base models, DIA outperforms full chain-of-thought in 12 of 14 settings, including a 7.00% gain on MuSiQue, while activating on only $\sim 17\%$ of examples and using up to 47.2% fe
    
[^57]: 基于调查报告的党派暴露数据验证大语言模型生成的政治网络

    Validating LLM-generated political networks with survey-reported partisan exposure

    [https://arxiv.org/abs/2610.23199](https://arxiv.org/abs/2610.23199)

    该研究创新性地提出以调查报告的党派暴露作为实证基准来验证大语言模型生成的政治网络，发现网络整体结构属性合理并不代表其能真实再现个体的政治环境。

    

    大语言模型（LLMs）在社会科学研究中被越来越多地用于模拟人类行为，包括决策、互动和社会关系。然而，在网络形成任务中，现有评估往往聚焦于整体网络属性，而那些结构属性看似合理的网络仍可能无法再现个体的社会和政治环境。在本研究中，我们使用从具有全国代表性的ANES 2020—2022社交媒体研究中采样的角色生成基于大语言模型的政治网络，并将受访者调查报告的党派暴露作为对应角色所生成政治环境的实证基准。我们首先刻画了生成网络的结构属性和政治同配性，然后将个体层面的党派暴露与调查数据进行比较。我们发现，生成网络的结构和政治同配性在不同（条件下存在差异）。

    arXiv:2610.23199v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to simulate human behavior in social science research, including decisions, interactions, and social relationships. However, in network formation tasks, existing evaluations often focus on aggregate network properties, while networks with plausible structural properties may still fail to reproduce individuals' social and political environments. Here, we generate LLM-based political networks using personas sampled from the nationally representative ANES 2020--2022 Social Media Study and use respondents' survey-reported partisan exposure as an empirical benchmark for the political environments generated for the corresponding personas. We first characterize the structural properties and political assortativity of the generated networks, and then compare ego-level partisan exposure with the survey. We find that the structure and political assortativity of the generated networks vary across
    
[^58]: 将空间听觉纳入自由能原理框架：一个知觉学习的分层模型

    Framing Spatial Hearing within the Free Energy Principle: A Hierarchical Model of Perceptual Learning

    [https://arxiv.org/abs/2610.23196](https://arxiv.org/abs/2610.23196)

    该研究首次将自由能原理应用于空间听觉领域，提出一个分层生成模型，其中视觉反馈通过驱动对变分自由能的梯度下降来更新听者的信念，从而实现动态条件下的知觉学习建模。

    

    空间听觉依赖于动态的内部表征，以便在不断变化的声学条件下解码空间信息。一个基本的例子是声音定位，即听者推断不可见声音的方向。贝叶斯推断为这类定位过程提供了计算层面的解释；然而，现有的建模工作大多局限于静态的听觉场景。相比之下，自由能原理（FEP）将知觉、学习和行动描述为对变分自由能（VFE）的联合最小化，从而提供了一个有原则的框架来解释动态条件。在此，我们朝着将自由能原理应用于空间听觉迈出了第一步。我们刻画了听者的信念如何在多次试验中不断演化——在每次试验中，听者估计声音的方向，并在随后收到视觉反馈。听者被描述为一个分层生成模型，其中视觉反馈驱动对变分自由能的梯度下降，从而更新……

    arXiv:2610.23196v1 Announce Type: cross  Abstract: Spatial hearing relies on dynamic internal representations to decode spatial information from ever-changing acoustic conditions. A fundamental example is sound localisation, where listeners infer the direction of an invisible sound. Bayesian inference provides a computational account of such a localisation process; however, modelling efforts are largely limited to static listening scenarios. Instead, the free energy principle (FEP) describes perception, learning, and action as joint minimisation of variational free energy (VFE), providing a principled framework to account for dynamic conditions. Here, we take a first step toward applying the FEP to spatial hearing. We characterise how listeners' beliefs evolve across trials, in which they estimated sound direction and received subsequent visual feedback on each trial. Listeners are described by a hierarchical generative model where visual feedback drives gradient descent on VFE, updati
    
[^59]: FuzzInsight：在覆盖引导的DNN模糊测试中权衡效率与有效性

    FuzzInsight: Navigating the Efficiency-Effectiveness Trade-off in Coverage-Guided DNN Fuzzing

    [https://arxiv.org/abs/2610.23195](https://arxiv.org/abs/2610.23195)

    该论文针对覆盖引导DNN模糊测试中种子利用低效和覆盖引导开销两大瓶颈，提出均衡采样策略将平均种子队列利用率提升至77%（较先前方法提高13个百分点），并为可扩展的DNN模糊测试提供面向配置的分析与决策指南。

    

    为评估深度神经网络（DNN）而收集和标注多样化的测试数据集既昂贵又耗时。这促使了覆盖引导模糊测试技术的发展，该技术能够从有限的测试数据中生成可揭示缺陷的极端用例。然而，其有效性在很大程度上取决于配置选择，特别是覆盖标准和种子选择策略。我们针对两个瓶颈——种子利用效率低下和覆盖引导的开销——提出了一种面向配置的分析方法以及可扩展DNN模糊测试的决策指南。我们的分析表明，现有的种子选择策略会反复优先选取一小部分输入，导致许多已收集的种子从未被探索。为解决这一问题，我们提出了均衡采样，这是一种基于约束的策略，可将平均种子队列利用率提升至77%，比先前方法提高了13个百分点。我们进一步在四个……上评估了84种模糊测试配置（摘要在此处截断）。

    arXiv:2610.23195v1 Announce Type: new  Abstract: Collecting and labeling diverse test datasets for evaluating Deep Neural Networks (DNNs) is costly and time-consuming. This has motivated coverage-guided fuzzing techniques that generate defect-revealing corner cases from limited test data. However, their effectiveness depends heavily on configuration choices, particularly coverage criteria and seed-selection strategies. We present a configuration-oriented analysis and decision guidelines for scalable DNN fuzzing targeting two bottlenecks: inefficient seed utilization and the cost of coverage guidance. Our analysis shows that existing seed-selection strategies repeatedly prioritize a small subset of inputs, leaving many collected seeds unexplored. To address this issue, we propose Equalized Sampling, a constraint-based strategy that increases mean seed queue utilization to 77%, improving on prior approaches by 13 percentage points.   We further evaluate 84 fuzzing configurations on four 
    
[^60]: OfficeTown：面向智能体智能持续进化的生成式组织

    OfficeTown: Generative Organizations for Evolving Agent Intelligence

    [https://arxiv.org/abs/2610.23194](https://arxiv.org/abs/2610.23194)

    提出 OfficeTown——一个能从自然语言描述生成持续运作的专业组织仿真框架，其中任务、工件、决策与权限随组织工作不断演化，并据此构建了包含 406 个任务和 4,836 条评估标准的 OfficeTown Bench，用于智能体的评估与学习。

    

    AI 智能体越来越多地使用文件和软件工具来产出专业交付成果，这类工作依赖于跨任务积累的材料、决策与反馈。我们提出 OfficeTown，一个面向专业组织的生成式仿真框架，其中任务及其上下文源于持续进行的组织工作。仅需对组织目的与活动的自然语言描述，OfficeTown 即可初始化一个组织，其成员相互协调工作、将有边界的任务委托给助手智能体，并审查其交付成果。工件、决策和未决事项会持续存在并塑造后续工作，而职责与权限决定了每个成员能够看到和执行的内容。我们将工作请求、其上下文以及审查反馈整理为可用于评估与学习的可执行资源。我们构建了 OfficeTown Bench，包含来自 23 个模拟组织的 406 个任务和 4,836 条评估标准（原文摘要至此截断）。

    arXiv:2610.23194v1 Announce Type: new  Abstract: AI agents increasingly use files and software tools to produce professional deliverables. Such work draws on materials, decisions, and feedback accumulated across assignments. We introduce OfficeTown, a generative simulation framework for professional organizations in which tasks and their context arise from continuing organizational work. From a natural-language description of an organization's purpose and activities, OfficeTown initializes an organization whose members coordinate work, delegate bounded assignments to assistant agents, and review their deliverables. Artifacts, decisions, and unresolved matters persist and shape subsequent work, while responsibilities and permissions determine what each member can see and do. We curate work requests, their context, and review feedback into executable resources for evaluation and learning. We construct OfficeTown Bench with 406 tasks and 4,836 evaluation criteria from 23 simulated organiz
    
[^61]: LangNCO：将自然语言约束编译为神经组合优化的推理时干预

    LangNCO: Compiling Natural-Language Constraints into Inference-Time Interventions for Neural Combinatorial Optimization

    [https://arxiv.org/abs/2610.23190](https://arxiv.org/abs/2610.23190)

    LangNCO利用预训练大语言模型将自然语言运营需求编译为DSL规则，并通过硬动作掩码与关键偏移解码等推理时干预手段，在不改动冻结NCO求解器的架构、表示与参数的情况下满足部署后出现的新约束。

    

    神经组合优化（NCO）能够针对训练时遇到的问题结构快速生成高质量解，但运营需求可能在部署之后发生变化，并以自然语言的形式被传达。我们提出LangNCO，这是一个将实例特定需求编译为结构化推理时干预的框架，作用于被冻结的NCO求解器，无需改变预训练策略的架构、学习到的输入表示或参数。预训练的大语言模型（LLM）充当可替换的语义前端，将每条需求翻译为经过模式验证的领域特定语言（DSL）规则集。确定性的规则执行器将这些规则应用为非扩张的硬动作掩码。由于局部掩码虽然能够防止即时违规，却不能确保仍然存在一条完整可行的后续解路径，关键偏移解码会提取规则所引用的关键节点，并重新分配固定rollout的一部分（摘要在此处截断）。

    arXiv:2610.23190v1 Announce Type: cross  Abstract: Neural combinatorial optimization (NCO) rapidly produces high-quality solutions for problem structures encountered during training, but operational requirements may change after deployment and be communicated in natural language. We propose LangNCO, a framework that compiles instance-specific requirements into structured inference-time interventions for a frozen NCO solver, without changing the pretrained policy's architecture, learned input representation, or parameters. A pretrained LLM serves as a replaceable semantic front end, translating each requirement into a schema-validated domain-specific language (DSL) rule set. A deterministic Rule Executor applies the rules as nonexpansive hard action masks. Because local masking prevents immediate violations but does not ensure that a complete feasible continuation remains reachable, Critical-Offset Decoding extracts rule-referenced critical nodes and reallocates part of a fixed rollout 
    
[^62]: 内在的更强教师：通过策略内双向蒸馏增强MoE模型

    A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation

    [https://arxiv.org/abs/2610.23185](https://arxiv.org/abs/2610.23185)

    提出MoEvolve框架，通过增加MoE模型每token激活预算来挖掘模型内部潜在更强教师，并采用“先恢复后提升”的策略内双向蒸馏策略强化MoE大语言模型。

    

    基于可验证奖励的强化学习（RLVR）已显著提升了混合专家（MoE）大语言模型，但其结果层面的奖励只能提供稀疏的监督信号。策略内蒸馏提供了密集的token级反馈，然而需要一个更强的兼容教师模型，而策略内自蒸馏则依赖于手工设计的特权信息。我们观察到，MoE架构本身提供了一种解决这一困境的方法：增加每个token的激活预算可以释放潜在的专家容量，从而在模型内部产生一个潜在更强的教师。然而，简单的扩展会降低性能，因为新共同激活的专家并未经过协同工作的优化。因此，我们提出了MoEvolve，一个遵循“先恢复后提升”策略的双向蒸馏框架：我们首先恢复激活扩展后的高k模型，并通过利用其额外的激活容量进一步强化它，然后将（摘要在此处截断）

    arXiv:2610.23185v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved Mixture-of-Experts (MoE) LLMs, but its outcome-level rewards provide only sparse supervision. On-Policy Distillation offers dense token-level feedback, yet requires a stronger compatible teacher, while on-policy self-distillation relies on hand-designed privileged information. We observe that the MoE architecture itself offers a way to resolve this dilemma, as increasing the activation budget per token exposes latent expert capacity and yields a potentially stronger teacher within the model. However, naive expansion degrades performance because newly co-activated experts are not optimized to work together. We therefore propose MoEvolve, a bidirectional distillation framework following a recovery-then-improvement strategy: we first restore the activation-expanded high-$k$ model and further strengthen it by exploiting its additional active capacity, then rev
    
[^63]: 面向长时程智能体强化学习的树感知后验蒸馏

    Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.23181](https://arxiv.org/abs/2610.23181)

    本文提出TreeOPSD，一种无需评论家的树感知后验自蒸馏框架，通过将树结构回放批次转化为有限的后验自蒸馏目标，同时解决长时程智能体强化学习中的稀疏信用分配与策略陈旧问题。

    

    具有可验证奖励的强化学习已成为大语言模型的一种可扩展的后训练范式。然而，在长时程交互中训练智能体面临两个挑战：仅使用终端奖励使得回合级别的信用分配稀疏且方差很高，而在固定的回放批次上进行顺序小批量更新会引入策略陈旧问题，因为当前策略会逐渐偏离行为策略。现有方法通过学习评论家、过程奖励模型或在线策略蒸馏来获得更密集的信用信号，但它们依赖于价值网络、步骤级标注或更强的外部教师模型，从而增加了训练复杂性。与此同时，重要性采样和裁剪可以缓解策略漂移，但它们并未指定一个有限的目标来阻止顺序更新过度。在这项工作中，我们提出了TreeOPSD，一个无评论家、树感知的后验自蒸馏框架，用于长时程智能体……

    arXiv:2610.23181v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards has become a scalable post-training paradigm for large language models. However, training agents over long-horizon interactions presents two challenges: using only terminal rewards makes turn-level credit assignment sparse and high-variance, while sequential mini-batch updates on a fixed rollout batch introduce policy staleness as the current policy drifts from the behavior policy. Existing approaches obtain denser credit via learned critics, process reward models, or on-policy distillation, but they rely on a value network, step-level annotations, or a stronger external teacher that increase training complexity. Meanwhile, importance sampling and clipping mitigate policy drift, but they do not specify a finite target that stops sequential updates from overshooting. In this work, we introduce TreeOPSD, a critic-free, tree-aware posterior self-distillation framework for long-horizon agentic
    
[^64]: LongForge：基于证据-断言推理图的长上下文指令数据测量与改进

    LongForge: Measuring and Improving Long-Context Instruction Data via Evidence-Assertion Reasoning Graphs

    [https://arxiv.org/abs/2610.23167](https://arxiv.org/abs/2610.23167)

    提出LongForge框架，将数据质量形式化为文档-问题-答案之间的对齐结构，通过将样本反向重构为证据-断言推理图，从广度、深度、正确性、完整性四个维度实现对长上下文指令数据的可审计测量与改进。

    

    现有针对长上下文指令微调数据的质量控制，是生成流水线中的一次性判断，只留下一个标量值，既不揭示答案使用了文档的哪些部分，也不揭示它写下了多少推理步骤，因此数据既无法被审计也无法被控制。我们将数据质量形式化为文档、问题和答案之间对齐结构的一种属性，并构建了LongForge，它将测量长上下文数据的分析器与生成此类数据的合成器配对。分析器将每个样本反向重构为证据-断言推理图，并在此基础上以程序化方式计算四个维度：广度、深度、正确性和完整性。每一次解读都可追溯到具体的证据片段和推理步骤，从而使数据可被审计。对七个主流数据集的测量揭示了两个此前从未被量化的缺陷：广度几乎不随文档长度增长，而深度通常（摘要原文在此处截断）……

    arXiv:2610.23167v1 Announce Type: new  Abstract: Existing quality control for long-context instruction-tuning data is a one-shot judgment inside the generation pipeline that leaves only a scalar, revealing neither which document parts an answer uses nor how many reasoning steps it writes down; the data can thus be neither audited nor controlled. We formalize data quality as a property of the alignment structure among document, question, and answer, and build LongForge, which pairs an analyzer that measures long-context data with a synthesizer that generates it. The analyzer reverse-reconstructs each sample into an evidence-assertion reasoning graph and programmatically computes four dimensions on it: breadth, depth, correctness, and completeness. Every reading traces back to specific evidence spans and reasoning steps, making data auditable. Measuring seven mainstream datasets reveals two previously unquantified defects: breadth barely grows with document length, and depth is generally
    
[^65]: 用自然语言解释国际象棋与围棋

    Explaining Chess and Go in Natural Language

    [https://arxiv.org/abs/2610.23166](https://arxiv.org/abs/2610.23166)

    本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。

    

    现代国际象棋引擎的棋力远超人类，其选择着法背后的思路往往令人类难以理解。我们提出了一种基于国际象棋引擎评估的自动化指标，用于评估棋局解释的质量。利用该指标对最先进的语言模型进行评估后发现，它们的表现出奇地差，往往依赖于对棋盘的表面观察，而非稳健、可迁移的国际象棋概念。随后，我们提出了BinProp，这是一种无需训练的方法，能够使同样的模型发现可复用的国际象棋概念，并用它们以自然语言解释着法。BinProp提升了对所有受测模型的可解释性，与直接提示相比，平均得分几乎提高了两倍。除了国际象棋之外，同一方法无需额外调整即可在围棋、西洋跳棋和将棋中发现有用的概念。

    arXiv:2610.23166v1 Announce Type: cross  Abstract: Modern chess engines play far beyond human ability, often choosing moves whose underlying ideas are difficult for humans to understand. We propose an automatic metric for evaluating chess explanations, grounded in chess-engine evaluations. Evaluating state-of-the-art language models using this metric, we find that they perform surprisingly poorly, often relying on surface-level board observations rather than robust, transferable chess concepts. We then introduce BinProp, a training-free method that enables these same models to discover reusable chess concepts and use them to explain moves in natural language. BinProp improves explainability for every model tested, nearly tripling the average score compared with direct prompting. Beyond chess, the same method discovers useful concepts in Go, draughts, and shogi without additional tuning.
    
[^66]: 算法模型化：将优化算法转变为可进化的智能

    Algorithm Modelization: Transforming Optimization Algorithms into Evolvable Intelligence

    [https://arxiv.org/abs/2610.23161](https://arxiv.org/abs/2610.23161)

    提出“算法模型化”新范式，将迭代优化算法转化为可训练的自回归模型，通过学习算法执行轨迹使模型内化、复现并进化超越现有算法。

    

    算法能否像模型一样被学习和进化，而不是像程序那样被反复重新设计？我们通过引入“算法模型化”这一新范式来探索这一问题，该范式将迭代算法转化为可训练的模型。其核心思想是将算法执行轨迹视为计算经验：通过学习算法状态在迭代过程中的演化方式，模型能够内化现有算法所蕴含的知识，复现其计算行为，并通过训练进一步进化以超越现有算法。作为该范式的首次实现，我们将迭代优化建模为一个自回归过程，其中先前的世代构成了生成下一代种群的计算上下文。我们开发了一种与维度无关的种群-时间架构，以同时捕捉种群交互与迭代动力学，并提出了一种从模仿到超越的学习框架，该框架首先……

    arXiv:2610.23161v1 Announce Type: new  Abstract: Can algorithms be learned and evolved as models, rather than repeatedly redesigned as programs? We explore this question by introducing Algorithm Modelization, a new paradigm that turns iterative algorithms into trainable models. Its central idea is to regard algorithm execution trajectories as computational experience: by learning how algorithmic states evolve over iterations, a model can internalize the knowledge embodied in existing algorithms, reproduce their computational behaviors, and further evolve beyond them through training. As a first realization of this paradigm, we modelize iterative optimization as an autoregressive process, where preceding generations constitute the computational context for generating the next population. We develop a dimension-independent population-temporal architecture to capture both population interactions and iterative dynamics, together with an imitation-to-surpassing learning framework that first
    
[^67]: QuAD：基于内容感知多智能体系统、通过分布对齐实现定性数据质量的定量评估方法

    QuAD: Quantitative Assessment via Distribution-Alignment for Qualitative Data Quality Enabled by a Content-Aware Multi-Agent System

    [https://arxiv.org/abs/2610.23153](https://arxiv.org/abs/2610.23153)

    提出了QuAD框架，利用内容感知多智能体系统，将定性回答重构的内在聚类分布与研究者定义的外部分析视角进行分布对齐，通过四象限联合指标在分析前定量评估定性数据集对特定研究目标的适用性。

    

    评估从半结构化问题中收集的定性回答是否适用于特定分析目标是一个长期存在的挑战。人工定性分析具有解释性且劳动密集，而现有自动化方法通常缺乏数据集层面的视角，无法判断一组回答是否支持研究者所选择的分析视角。我们提出了QuAD，一个用于定性回答数据集的定量预分析评估框架。QuAD将基于回答重构的内在聚类分布与从元数据中导出的、由研究者定义的外部分析视角进行对齐比较。QuAD采用四象限评估结构，其指标需联合解读——从视角层面的关系到聚类内部的变异——从而为揭示普遍规律与独特洞察提供“用途适配性”信号。由于该评估依赖于以分析视角为条件的回答间相异性度量（摘要在此处截断）……

    arXiv:2610.23153v1 Announce Type: new  Abstract: Assessing whether qualitative responses collected from semi-structured questions are ready for a particular analysis goal is a persistent challenge. Manual qualitative analysis is interpretive and labor-intensive, while existing automated methods often lack a dataset-level perspective for deciding whether a response set supports a researcher-selected analytical lens. We introduce QuAD, a framework for quantitative pre-analysis assessment of qualitative response datasets. QuAD compares a reconstructed intrinsic cluster distribution reconstructed from responses with an extrinsic, researcher-defined analytical lens derived from metadata. QuAD is structured into a four-quadrant assessment whose metrics are read jointly, from lens-level relations to within-cluster variation, to provide fitness-for-purpose signals for revealing general patterns and unique insights. Because this assessment relies on lens-conditioned inter-response dissimilarity
    
[^68]: CIIR——类索引接口表示学习

    CIIR - Class-Indexed Interface Representation Learning

    [https://arxiv.org/abs/2610.23151](https://arxiv.org/abs/2610.23151)

    CIIR提出了一种模块化编码器-解码器框架，利用类索引地址空间在无需配对样本的情况下连接异构数据集，并在冻结基础地址空间后仅通过轻量级适配器即可扩展至新数据集。

    

    多模态学习通常假设存在成对或同步的观测数据，然而独立收集的数据集很少能提供这种对应关系。我们提出CIIR，这是一个模块化的编码器-解码器框架，通过类索引地址空间连接异构数据集，同时保留每个数据集的原生输入格式和输出模块。一个已知的本地索引用于选择地址k，随后将与输入相关的128维地址表示Z_k(x)写入该地址。为了展示CIIR，我们在没有同步或实例匹配样本的情况下，从手写数字、服装图像、口述数字频谱图和手语手势中学习该地址空间。新训练的无标签接收器以94.7%的宏平均地址实现准确率成功恢复所请求的地址。随后，我们在冻结基础地址空间后接入未见过的SVHN数据集，仅训练轻量级适配器，实现了90.2%的MNIST>SVHN和82.3%的SVHN>MNIST迁移性能。

    arXiv:2610.23151v1 Announce Type: new  Abstract: Multimodal learning commonly assumes paired or synchronized observations, yet independently collected datasets rarely provide such correspondence. We introduce CIIR, a modular encoder--decoder framework that connects heterogeneous datasets through a class-indexed address space while preserving each dataset's native input format and output module. A known local index selects address $k$, and an input-dependent 128-dimensional address representation Z_k(x) is written to that address. To demonstrate CIIR, we learn this address space from handwritten digits, clothing images, spoken-digit spectrograms, and hand signs without synchronized or instance-matched samples. Newly trained label-free receivers recover the requested addresses with 94.7% macro-average Address Realization Accuracy. We then attach the unseen SVHN dataset after freezing the base address space and train only lightweight adapters, achieving 90.2% MNIST>SVHN and 82.3% SVHN>MNI
    
[^69]: 用于内存高效训练的局部代理网络

    Local proxies for memory-efficient training

    [https://arxiv.org/abs/2610.23145](https://arxiv.org/abs/2610.23145)

    提出为每个Transformer块训练局部代理网络，通过蒸馏预测全局输出使梯度可局部独立传播，在大幅降低训练内存占用的同时保持接近标准反向传播的梯度估计精度。

    

    局部学习算法能够重塑人工智能的资源需求。标准端到端反向传播的内存瓶颈在于必须存储所有网络激活，直到损失梯度反向传播完成。局部学习可以避免这一问题，但通常以精度下降为代价。我们研究了如何在Transformer中实现准确且内存高效的局部学习。对于每个Transformer块，我们通过知识蒸馏训练一个局部代理网络来预测全局网络输出。梯度可以独立地通过该局部代理进行传播，同时能够很好地估计反向传播梯度。理论上，该方法对于深度线性网络是精确的；在非线性残差模型中，使用适当的残差代理可以将梯度误差限定在边界内。我们通过数值实验验证了这一点，并证明优化蒸馏损失可以改善梯度估计。

    arXiv:2610.23145v1 Announce Type: cross  Abstract: Local learning algorithms can reshape resource requirements of artificial intelligence. The memory bottleneck of standard end-to-end backpropagation lies in the need to store all network activations until the loss gradient has been propagated back through. Local learning avoids this, but typically at a cost in accuracy. We study how to achieve accurate and memory efficient local learning in transformers. For each transformer block, we train a local proxy network to predict the global network output via distillation. Gradients can be independently propagated through this local proxy while estimating well the backpropagation gradients. In theory, this approach is exact for deep linear networks; in non-linear residual models, the gradient error can be bounded when using the appropriate residual proxy. We verify this numerically and demonstrate that optimizing the distillation loss improves gradient estimation. Evaluating the performance o
    
[^70]: LexPilot：面向中国民事案件咨询的法律大语言模型与非专业人士主动交互能力基准测试

    LexPilot: Benchmarking the Proactive Interaction Ability of Legal LLMs with Laypeople in Chinese Civil Case Consultation

    [https://arxiv.org/abs/2610.23141](https://arxiv.org/abs/2610.23141)

    该论文提出了LexPilot基准，基于真实中国民事判决书，首次系统评测法律大模型在面对非专业人士咨询时主动追问关键案件信息并提升主张层面判决预测的能力。

    

    大语言模型（LLM）已为法律专业人士提供了巨大帮助。然而，我们的实证研究表明，当面对非专业人士表述不完整的咨询时，当前的LLM即便缺乏足以得出明确结论的案件细节，也常常给出实用性有限的不负责任回复。为了更好地帮助普通民众，法律LLM必须识别争议焦点，并主动从用户处获取关键背景信息。遗憾的是，现有数据资源大多只关注交互与回复质量，对于主动交互如何影响LLM在真实民事案件中进行下游的主张层面判决预测，所提供的洞见十分有限。为填补这一空白，我们提出了LexPilot，一个利用真实中国民事判决书、将主动法律交互与主张层面判决预测相联系的基准。LexPilot基于36类常见案由的432份真实判决构建，包含专家……

    arXiv:2610.23141v1 Announce Type: new  Abstract: Large Language Models (LLMs) have greatly assisted legal professionals. However, our empirical studies reveal that, when faced with underspecified consultations from non-experts, current LLMs often produce irresponsible responses with limited utility despite lacking sufficient case details for definitive conclusions. To better assist laypeople, legal LLMs must identify focal points of dispute and proactively elicit crucial context from users. Unfortunately, existing data resources often focus on interaction and response quality, offering limited insight into how proactive interaction affects LLMs' downstream claim-level judgment prediction in actual civil cases. To address this gap, we introduce \textbf{LexPilot}, a benchmark linking proactive legal interaction to claim-level verdict prediction using authentic Chinese civil judgments. Constructed from 432 real-world judgments across 36 common causes of action, LexPilot comprises expert-r
    
[^71]: WebCoder：基于工件锚定的智能体强化学习扩展通用前端开发

    WebCoder: Scaling General Frontend Development with Artifact-Grounded Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.23104](https://arxiv.org/abs/2610.23104)

    WebCoder提出工件锚定的智能体强化学习方法，通过可执行环境Webdev-World和依据最终应用是否满足任务需求的独立验证奖励，训练出能完成前端复现、生成、修改与修复的通用开发智能体。

    

    现实世界的前端开发要求智能体能够在不断演进的应用状态上完成复现、生成、修改和修复等多种操作的组合。为这一过程学习一个通用策略，需要将异构的执行反馈转化为与任务对齐的奖励，并在长序列的相互依赖的动作之间实现有效的信用分配。我们提出了WebCoder，一个通过工件锚定的智能体强化学习训练而成的通用前端开发智能体。我们的核心思想是：通过多样化的开发活动对同一个共享可执行工件所产生的影响，以及该工件对任务需求的满足程度，来评估这些开发活动。我们构建了Webdev-World，这是一个可执行环境，集成了代码编辑、应用执行、浏览器检查和按需调用的技能，并在交互过程中保持项目状态的连续性。一个独立的验证器会根据任务需求对最终应用进行评估，以提供终止奖励，同时……

    arXiv:2610.23104v1 Announce Type: new  Abstract: Real-world frontend development requires agents to compose reproduction, generation, modification, and repair over an evolving application state. Learning a general policy for this process requires translating heterogeneous execution feedback into task-aligned rewards and effective credit assignment across long sequences of interdependent actions. We introduce WebCoder, a general frontend development agent trained with artifact-grounded agentic reinforcement learning. Our key idea is to assess diverse development activities through their effects on a shared executable artifact and its satisfaction of task requirements. We build Webdev-World, an executable environment that integrates code editing, application execution, browser inspection, and on-demand skills while preserving project state across interactions. An independent verifier evaluates the final application against task requirements to provide terminal rewards, while changes in i
    
[^72]: 重新思考GRPO在大型推理模型安全对齐中的应用

    Rethinking GRPO for Large Reasoning Model Safety Alignment

    [https://arxiv.org/abs/2610.23085](https://arxiv.org/abs/2610.23085)

    本文揭示了GRPO用于安全对齐时的优势坍缩和不安全多样化两种失败模式，并提出包含安全信号恢复（SSR）与安全-效用解耦对齐（SUDA）的Beacon方法来恢复安全学习信号，提升大型推理模型的安全性。

    

    大型推理模型（LRMs）具备强大的推理能力，但在输出安全性方面仍存在脆弱性。尽管组相对策略优化（GRPO）被广泛用于安全对齐，但其基于组的相对优化可能无法提供足够的安全导向学习信号。在本文中，我们识别出两种失败模式：优势坍缩，即同质化的采样组导致优势信号消失；以及不安全多样化，即不安全行为持续存在并变得日益多样化。我们的理论分析表明，行为一致性会削弱组内的安全对比，而当相对信号消失时，熵正则化会加剧不安全响应的多样化。为应对这些挑战，我们提出了Beacon方法，包含安全信号恢复（SSR）和安全-效用解耦对齐（SUDA）两个部分。SSR结合优势引导的轨迹过滤和安全参考监督，以恢复有效的安全学习信号……（摘要内容在此处截断）

    arXiv:2610.23085v1 Announce Type: new  Abstract: Large reasoning models (LRMs) exhibit strong reasoning capabilities but remain vulnerable to unsafe outputs. While Group Relative Policy Optimization (GRPO) is widely used for safety alignment, its group-relative optimization may provide insufficient safety-directed learning signals. In this paper, we identify two failure modes: advantage collapse, where homogeneous rollout groups yield vanishing advantages, and unsafe diversification, where unsafe behaviors persist and become increasingly diverse. Our theoretical analysis reveals that behavioral consistency weakens within-group safety contrasts, while entropy regularization can diversify unsafe responses when relative signals vanish. To address these challenges, we propose Beacon, comprising Safety Signal Recovery (SSR) and Safety-Utility Decoupled Alignment (SUDA). SSR combines advantage-guided trajectory filtering and safe reference supervision to recover effective safety learning sig
    
[^73]: 基于学习的潜在多表表示通往先验数据拟合关系模型的路径

    Path to Prior-Data Fitted Relational Models via Learned Latent Multi-table Representations

    [https://arxiv.org/abs/2610.23073](https://arxiv.org/abs/2610.23073)

    该论文首次将先验数据拟合（PFN）范式拓展到基于消息传递的图关系模型，通过在合成关系数据库上预训练Griffin编码器，并将其潜在表示输入冻结的上下文内预测器，在目标任务零训练的情况下实现了对真实世界关系分类任务的泛化，并验证了课程学习对消息传递表示的增益。

    

    先验数据拟合网络（PFNs）表明，在大量合成任务上进行预训练能够使模型泛化到未见过的数据集，而几乎无需或完全无需下游微调。这一范式在基于Transformer的表格和关系基础模型（如RDB-PFN）上已被证明有效，但对于基于消息传递构建的图基关系模型而言，该范式在很大程度上仍待探索。我们在大规模合成关系数据库集合上预训练图基关系基础模型Griffin，并探究对基于Transformer的关系学习者有益的课程学习，是否同样能改进消息传递表示。我们没有在下游任务上进行微调，而是将预训练编码器作为通用的关系表示学习器，并将其潜在表示传递给一个冻结的上下文内预测器（TabICL）；没有任何组件在目标任务上接受训练。在16个真实世界的关系分类任务上进行了评估……

    arXiv:2610.23073v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) show that pretraining on large populations of synthetic tasks can enable generalization to unseen datasets with little or no downstream fine-tuning. This paradigm is effective for transformer-based tabular and relational foundation models such as RDB-PFN, but remains largely unexplored for graph-based relational models built on message passing. We pretrain the graph-based relational foundation model Griffin on large collections of synthetic relational databases and ask whether curriculum learning, which benefits transformer-based relational learners, also improves message-passing representations. Rather than fine-tune on downstream tasks, we use the pretrained encoder as a general-purpose relational representation learner and pass its latent representations to a frozen in-context predictor (\ticl{}); no component is trained on the target task. On 16 real-world relational classification tasks, evaluated
    
[^74]: 通过“模型到策略”的公式化表述统一具身基础模型

    Unifying Embodied Foundation Models via a Model-to-Policy Formulation

    [https://arxiv.org/abs/2610.23053](https://arxiv.org/abs/2610.23053)

    本文提出一种基于随机核的、与架构无关的统一概率表述，将VLA模型与世界动作模型等具身基础模型纳入“模型到策略”的统一框架，贯通预训练、后训练与策略部署全过程。

    

    视觉-语言-动作（VLA）模型和世界-动作模型（WAM）已成为机器人学习与控制领域颇具前景的方法。这些模型融合了多模态感知与动作生成，但在架构、输出表示和学习目标上各不相同。本文提出了一种基于随机核的、与具体架构无关的统一概率表述，用于刻画具身基础模型。我们的表述包含两个关键组成部分：（i）通过条件输出分布和动作实现核来刻画具身基础模型，二者共同导出任务条件化的机器人策略；（ii）将预训练与后训练连接到策略部署，从而兼容模型特定的预训练目标以及离线或在线的后训练策略。我们在OpenVLA和DreamZero上对该表述进行了实例化，涵盖自回归动作token预测与联合视频-动作生成（原文摘要在此处截断）。

    arXiv:2610.23053v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models and world-action models (WAMs) have emerged as promising approaches to robot learning and control. These models integrate multimodal perception and action generation, but differ in their architectures, output representations, and learning objectives. In this paper, we present a unified, architecture-agnostic probabilistic formulation for embodied foundation models based on stochastic kernels. Our formulation has two key components: (i) we characterize embodied foundation models through conditional output distributions and action-realization kernels, which together induce task-conditioned robot policies; and (ii) we connect pretraining and post-training to policy deployment, accommodating model-specific pretraining objectives and offline or online post-training strategies. We instantiate the formulation for OpenVLA and DreamZero, covering autoregressive action-token prediction and joint video-action g
    
[^75]: 智能体式上下文学习：Transformer在闭环交互中能学会哪些算法？

    Agentic In-Context Learning: What Algorithms Can Transformers Learn in Closed-Loop Interaction?

    [https://arxiv.org/abs/2610.23051](https://arxiv.org/abs/2610.23051)

    本文提出“智能体式上下文学习”框架，首次从学习算法视角研究闭环交互中Transformer的可学习性，证明因果Transformer能够近似主动学习、多臂老虎机和在线强化学习等算法并给出回报损失上界，同时刻画了过程奖励优化在何时能保持最大化最终结果的行为。

    

    上下文学习（ICL）的学习算法视角研究的是固定参数的Transformer如何从所提供的示例中学习。语言模型智能体通过交互扩展了这一设定：它们的动作决定了接下来有哪些证据进入上下文。我们将这种机制称为智能体式上下文学习。现有研究主要依赖于外部提供的上下文或带有奖励信号的交互历史，而关于通用工具使用智能体的研究则缺乏学习算法的视角。我们用状态、更新和决策规则来表示一个交互式学习者，其涵盖主动学习、多臂老虎机和在线强化学习（RL）。在表达能力方面，我们构造了能够在有限时间范围内近似这些算法的因果Transformer，并对由此产生的回报损失给出上界。在训练方面，我们刻画了何时优化过程奖励能够保持最大化最终结果的行为。蒸馏实验表明，Transformer…

    arXiv:2610.23051v1 Announce Type: new  Abstract: The learning algorithm view of in-context learning (ICL) studies how fixed-parameter Transformers learn from supplied examples. Language model agents extend this setting through interaction: their actions determine what evidence enters context next. We call this regime Agentic In-Context Learning (Agentic ICL). Existing studies largely depend on externally supplied contexts or reward-bearing interaction histories, while work on general tool-using agents lacks a learning algorithm view. We represent an interactive learner by its state, update, and decision rule, covering active learning, bandits, and online reinforcement learning (RL). For expressivity, we construct causal Transformers that approximate these algorithms over a finite horizon, and bound the resulting loss in return. For training, we characterize when optimizing process rewards preserves behavior that maximizes the final outcome. Distillation experiments show that Transforme
    
[^76]: 蒸馏直至分歧：面向智能体强化学习的误差截断事后蒸馏

    Distill Until Divergence: Error-Truncated Hindsight Distillation for Agentic RL

    [https://arxiv.org/abs/2610.23048](https://arxiv.org/abs/2610.23048)

    该论文提出了一种面向智能体强化学习的误差截断事后蒸馏方法，通过根据轨迹结果和误差边界路由正向技能（强化成功流程与错误前决策）和负向技能（针对决定性错误进行纠正）两种互补的事后技能，解决了技能条件化监督与错误后状态不匹配的问题，实现与奖励对齐的细粒度信用分配。

    

    长程交互要求智能体在不断变化的环境中做出多个决策。基于结果的GRPO缺乏对中间决策的细粒度指导。近期的些方法将完整轨迹蒸馏为可复用的技能，并使用事后条件化的OPD来提供密集的token级信用分配。然而，技能引起的概率偏移并不一定代表与奖励对齐的信用；尤其是在错误动作改变了后续交互状态之后，技能条件化的监督可能与当前状态不匹配。为了解决这种不匹配问题，我们提出了该方法，它根据轨迹结果和误差边界来路由两种互补的事后技能：正向技能捕捉成功的工作流程，以强化成功轨迹和错误发生前的决策；负向技能捕捉失败特有的纠正信息，以针对决定性错误。错误发生后的后缀部分仅通过（原文摘要至此截断）

    arXiv:2610.23048v1 Announce Type: new  Abstract: Long-horizon interaction requires agents to make multiple decisions in a changing environment. Outcome-based GRPO lacks fine-grained guidance for intermediate decisions. Recent methods distill completed trajectories into reusable skills and use hindsight-conditioned OPD to provide dense token-level credit. However, skill-induced probability shifts do not necessarily represent reward-aligned credit; especially after an erroneous action changes subsequent interaction states, skill-conditioned supervision may become mismatched with the current state. To address this mismatch, we propose \methodname{}, which routes two complementary hindsight skills according to trajectory outcomes and error boundaries: Positive skills capture successful workflows to reinforce successful trajectories and pre-error decisions, while Negative skills capture failure-specific corrections to target decisive errors. The post-error suffix is optimized exclusively by
    
[^77]: SpikeQ：脉冲引导的脉冲神经网络训练后量化

    SpikeQ: Spike-Guided Post-Training Quantization for Spiking Neural Networks

    [https://arxiv.org/abs/2610.23046](https://arxiv.org/abs/2610.23046)

    该论文发现SNN中量化误差的精度损失取决于其翻转的脉冲比例而非MSE大小，据此提出SpikeQ框架，通过最小化下游神经元的脉冲误差来引导权重和膜电位的训练后量化。

    

    在神经形态芯片上，脉冲神经网络（SNN）比人工神经网络（ANN）消耗的能量要少得多，但这些芯片有限的内存要求使用低比特的突触权重和膜电位。现有的训练后量化（PTQ）方法无需重新训练即可实现量化，并且与针对ANN的方法类似，它们通过最小化均方误差（MSE）等数值误差来工作。然而，在SNN中，量化误差只有通过其翻转的脉冲才能传播到后续层。我们发现MSE相同的误差可能造成截然不同的精度损失，而翻转相同比例脉冲的误差则会造成相同的精度损失。因此，我们提出了SpikeQ，一个脉冲引导的PTQ框架。对于权重，我们通过模拟下游神经元并最小化其脉冲误差来选择每个权重的量化级别。对于膜电位，大多数现有方法在复位后对状态进行量化，而神经形态芯片则将阈值与……（原文摘要不完整）

    arXiv:2610.23046v1 Announce Type: cross  Abstract: On neuromorphic chips, spiking neural networks (SNNs) consume far less energy than artificial neural networks (ANNs), but the limited memory of these chips requires low-bit synaptic weights and membrane potentials. Existing post-training quantization (PTQ) methods provide them without retraining and, like those for ANNs, minimize numerical errors such as the mean squared error (MSE). In an SNN, however, a quantization error reaches later layers only through the spikes it flips. We find that errors of equal MSE can cause very different accuracy losses, whereas errors that flip the same fraction of spikes cause the same loss. We therefore propose SpikeQ, a spike-guided PTQ framework. For weights, we select the level of each weight by simulating the downstream neurons and minimizing their spike errors. For membrane potentials, most existing methods quantize the state after the reset, whereas neuromorphic chips compare the threshold with t
    
[^78]: 性能去哪了？理解并恢复 C 到 Rust 转换中的性能退化

    Where Does the Performance Go? Understanding and Recovering Performance Regressions in C-to-Rust Translation

    [https://arxiv.org/abs/2610.23044](https://arxiv.org/abs/2610.23044)

    该论文系统性地研究了 C 到 Rust 自动转换造成的性能退化问题，通过对执行剖析、LLVM IR 和编译器优化备注的对比分析识别出九类反复出现的性能问题，并提出了一种基于 LLM 的工具，将这些问题编码为可重用的优化规则来恢复转换后 Rust 代码的性能。

    

    Rust 正日益被采纳为 C 语言的内存安全继任者，而像 C2Rust 这样的自动转换器可以在保持程序行为不变的前提下迁移完整的 C 项目。然而，保持行为并不意味着保持性能。本文在 20 个真实世界的 C 项目及其 C2Rust 转换版本上研究了这一差距。在用于分析的八个项目中，转换后的 Rust 在 19 个操作中有 12 个比原始 C 代码慢超过 3%，最严重时慢达 62.2%。通过对比每对 C--Rust 程序的执行成本剖析、优化后的 LLVM IR 以及编译器优化备注，作者将这些性能下降追溯到九类反复出现的性能问题，涵盖残留的运行时开销、错失的编译器优化以及 C 风格的源代码模式，并为每类问题推导出相应的优化方向。随后，作者提出了一种基于大语言模型（LLM）的方法，将这些性能问题编码为可重用的优化规则，以恢复转换后 Rust 代码的性能。该工具会对程序进行剖析……

    arXiv:2610.23044v1 Announce Type: new  Abstract: Rust is increasingly adopted as a memory-safe successor to C, and transpilers such as \crust{} can migrate entire C projects while preserving their behavior. Preserving behavior, however, does not guarantee preserving performance. We study this gap on 20 real-world C projects and their \crust{} translations. On the eight projects used for analysis, the translated Rust is more than 3\% slower than the original C on 12 of 19 operations, by up to 62.2\%. By comparing execution-cost profiles, optimized LLVM IR, and compiler optimization remarks across each C--Rust pair, we trace these slowdowns to nine recurring performance issues, spanning residual runtime work, missed compiler optimizations, and C-style source patterns, and derive an optimization direction for each. We then present \ourtool{}, an LLM-based approach that encodes these issues as reusable optimization rules to recover the performance of translated Rust. \ourtool{} profiles ea
    
[^79]: 研究构思始于何处：基于前向引用监督的可执行局限性生成

    Where Research Ideation Begins: Forward-Citation Supervision for Actionable Limitation Generation

    [https://arxiv.org/abs/2610.23038](https://arxiv.org/abs/2610.23038)

    该论文发现研究构思中“识别可执行局限性”的瓶颈在于监督信号而非模型规模，并基于前向引用构建了含18,279条人工验证监督三元组的ActLim-Bench数据集，通过三阶段后训练流程提升大模型从文献中生成可执行研究局限性的能力。

    

    自动化研究构思已取得快速发展，出现了能够端到端生成完整研究想法的多智能体工作流。然而，这些框架在很大程度上忽视了一个孕育好想法的关键步骤：从文献中识别出可执行的局限性。本文聚焦于研究构思流程中的这第一步。从一项试点研究出发，我们发现当大语言模型（LLM）的参数规模从0.6B扩展到32B时，它们在识别后续工作解决了哪些局限性方面能力显著提升，但在生成此类局限性方面却毫无改善，这表明瓶颈在于监督信号而非模型能力。因此，我们基于前向引用构建了ActLim-Bench，产生了18,279条经人工验证的、带有证据链接的监督三元组。基于这一信号，我们提出了一个三阶段后训练流程：监督微调、经验证器过滤的拒绝采样微调，以及基于……的强化学习（原文摘要在此处截断）。

    arXiv:2610.23038v1 Announce Type: new  Abstract: Automated research ideation has advanced rapidly, with multi-agent workflows that generate whole research ideas end to end. These frameworks leave largely unexamined the step that seeds a good idea: identifying an actionable limitation from literature. In this paper, we focus on this first step in the research ideation pipeline. Starting from a pilot study, we find that as large language models (LLMs) scale from 0.6B to 32B parameters, they become far better at recognizing which limitation later work addressed, but no better at generating one, suggesting that the bottleneck lies in the supervision signal rather than model capacity. We therefore construct ActLim-Bench from forward citations, yielding 18,279 evidence-linked supervision triples with human verification. On this signal, we propose a three-stage post-training pipeline: supervised fine-tuning, verifier-filtered rejection-sampling fine-tuning, and reinforcement learning with the
    
[^80]: ASE：面向时间序列预测的归因引导语义演化

    ASE: Attribution-Guided Semantic Evolution for Time Series Forecasting

    [https://arxiv.org/abs/2610.23027](https://arxiv.org/abs/2610.23027)

    提出了归因引导语义演化（ASE）模块化框架，通过归因反馈机制自动构建、评估和演化文本因子并与数值模型融合，在TESS文本时间序列基准和沪深300动态股票排名上均超越了已发表的最优结果。

    

    外部文本能够提供数值历史数据中所缺失的预测信息，但其价值取决于如何对文本进行表示、组合与部署。我们提出了归因引导语义演化，这是一个模块化框架，用于构建文本因子、评估其条件贡献，并部署经验证集筛选的数值-文本联合预测器。ASE支持针对具体任务的实现方式，包括词汇度量、语义原语或演化生成的事件/状态因子，并将其与合适的数值专家模型相结合。其基于反馈的实现利用条件重拟合诊断来指导因子修订与组合搜索。我们在四个与文本相关的TESS数据集以及动态沪深300股票排名任务上评估了ASE。四种预测实现分别取得了0.001041、0.022439、0.333115和1.8150的测试MSE，均低于相应的已发表TESS最优分数。在沪深300上，冻结文本融合实现（摘要在此处截断）

    arXiv:2610.23027v1 Announce Type: new  Abstract: External text can supply predictive information absent from numerical histories, but its value depends on how it is represented, combined and deployed. We present Attribution-guided Semantic Evolution (ASE), a modular framework for constructing textual factors, evaluating their conditional contribution, and deploying validation-selected numerical--textual predictors. ASE supports task-specific implementations with lexical measurements, semantic primitives or evolved event/state factors, coupled to suitable numerical experts. Its feedback-based implementation uses conditional refitting diagnostics to guide factor revision and combination search. We evaluate ASE on the four text-associated TESS datasets and dynamic CSI 300 stock ranking. The four forecasting implementations obtain test MSEs of 0.001041, 0.022439, 0.333115 and 1.8150, each below the corresponding published TESS score. On CSI 300, the frozen text-fusion implementation improv
    
[^81]: 大语言模型生成广告中营销标签的因果价值归因：一种去混淆因子方法

    Causal Value Attribution of Marketing Labels in LLM-Generated Ads: A Deconfounder Approach

    [https://arxiv.org/abs/2610.23025](https://arxiv.org/abs/2610.23025)

    本文提出D-VAE框架，将多处理去混淆因子方法与因果表征学习相结合，首次实现对LLM生成广告中专家营销标签因果价值的量化归因，解决了随机模型中介、高维标签交互与多模态混淆带来的估计难题。

    

    随着AI生成内容日益商品化，经济价值正逐渐从最终成品转移到用于指定输入以引导大语言模型（LLM）生成的上游专业知识。然而，组织缺乏评估这种专业知识价值的方法，也无法识别哪些具体技能或配置在因果意义上真正提升了绩效。本研究考察生成式营销领域，其中结构化的专家标签作为技能基元，用于条件化大语言模型生成的广告。估计这些标签的因果效应十分困难，因为结果反映了随机的大语言模型中介作用、高维标签交互以及多模态混淆。本研究提出了D-VAE（去混淆因子变分自编码器），这是一个将多处理去混淆因子方法与因果表征学习相结合的反事实推断框架。D-VAE恢复了因果解耦的专家标注逻辑，并推导出基于PAC（概率近似正确）的遗漏变量偏差界限，以应对单处理混淆带来的问题。

    arXiv:2610.23025v1 Announce Type: new  Abstract: As AI-generated content becomes commoditized, economic value increasingly shifts from final artifacts to the upstream expertise used to specify inputs that steer LLM generation. Yet organizations lack methods for valuing this expertise and identifying which individual skills or configurations causally improve performance. This study examines generative marketing, where structured expert labels serve as skill primitives conditioning LLM-generated advertisements. Estimating their causal effects is difficult because outcomes reflect stochastic LLM mediation, high-dimensional label interactions, and multimodal confounding. This study proposes D-VAE (Deconfounder Variational Autoencoder), a counterfactual-inference framework integrating a multi-treatment deconfounder with causal representation learning. D-VAE recovers causally disentangled expert-labeling logic and derives PAC-based bounds on omitted-variable bias from single-treatment confou
    
[^82]: 基于用户手册的计算机使用智能体对新软件的适应

    Adaptation of Computer Use Agents to New Software with User Manuals

    [https://arxiv.org/abs/2610.23012](https://arxiv.org/abs/2610.23012)

    本文提出MAGE框架，将用户手册作为可扩展的程序性知识来源，在执行时检索任务相关手册内容并生成明确的程序性指导，从而提升计算机使用智能体适应新软件的能力。

    

    多模态基础模型的最新进展显著提升了通过图形用户界面（GUI）、利用视觉观察和自然语言指令来操作真实世界软件的能力。然而，在未见过的应用上保持强大的性能仍然具有挑战性，因为实际任务通常需要长程的程序性知识和特定于应用的操作模式。此外，构建可执行任务以及收集用于训练的相应轨迹仍然严重依赖人工标注数据，导致成本高昂。在本文中，我们提出了MAGE（基于手册增强的计算机使用智能体执行与学习），这是一个利用用户手册作为可扩展的程序性知识来源、同时改进推理与训练的框架。在执行过程中，MAGE检索与任务相关的手册内容，并使用规划器为计算机使用智能体（CUA）生成明确的程序性指导。

    arXiv:2610.23012v1 Announce Type: new  Abstract: Recent advances in multimodal foundation models have significantly improved the ability to operate real-world software through graphical user interfaces (GUIs) using visual observations and natural language instructions. However, maintaining strong performance on unseen applications remains challenging, as practical tasks often require long-horizon procedural knowledge and application-specific operation patterns. In addition, constructing executable tasks and collecting corresponding trajectories for training still heavily rely on human-annotated data, incurring substantial cost. In this paper, we propose MAGE (Manual-augmented computer use AGent Execution and learning), a framework that leverages user manuals as a scalable source of procedural knowledge to improve both inference and training. During execution, MAGE retrieves task-relevant manual content and uses a planner to generate explicit procedural guidance for CUAs. To further acc
    
[^83]: 基于自适应专家分配的高效CPU-GPU协同MoE推理

    Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment

    [https://arxiv.org/abs/2610.23006](https://arxiv.org/abs/2610.23006)

    本文提出了一种CPU-GPU协同的MoE推理系统，根据专家的token负载差异自适应地将未命中专家分配到CPU或GPU执行——高负载专家传输至GPU、低负载专家留在CPU计算——从而平衡CPU计算路径与GPU传输加计算路径，最小化内存受限场景下的推理延迟。

    

    混合专家模型通过稀疏的专家激活机制减少了推理时的计算量，但其庞大的专家内存占用往往超出GPU的容量，这使得专家卸载成为内存受限推理的必要手段。然而，现有系统通常通过两种方式处理GPU未命中的专家：一是将这些专家传输到GPU执行，这会带来显著的PCIe传输延迟；二是在CPU上执行这些专家，这使得较慢的CPU计算成为性能瓶颈。为解决这些局限，我们提出了该系统，一个CPU-GPU协同的MoE推理系统，它能够自适应地将这些未命中的专家分配到CPU或GPU上以实现高效执行。通过利用不同专家token负载的差异，该系统选择性地将高负载专家传输到GPU执行，同时在CPU上执行低负载专家。为了最小化推理延迟，该系统自适应地调整专家分配策略，以平衡CPU计算路径与GPU传输加计算路径之间的开销。

    arXiv:2610.23006v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models reduce inference computation through sparse expert activation, but their large expert memory footprint often exceeds GPU capacity, making expert offloading necessary for memory-constrained inference. However, existing systems typically handle GPU-missed experts by either transferring them to the GPU, incurring substantial PCIe transfer latency, or executing them on the CPU, making the slower CPU computation the bottleneck. To address these limitations, we present \method, a CPU-GPU collaborative MoE inference system that adaptively assigns these missed experts to the CPU or GPU for efficient execution. By exploiting differences in expert token-loads, \method selectively transfers high-load experts to the GPU while executing low-load experts on the CPU. To minimize inference latency, \method adaptively adjusts expert assignments to balance the CPU computation path against the GPU transfer-and-computation pa
    
[^84]: 一种基于邻域标签分布的图数据标签噪声鲁棒算法

    A Label Noise Robust Algorithm for Graph Data using Neighborhood Label Distribution

    [https://arxiv.org/abs/2610.22998](https://arxiv.org/abs/2610.22998)

    本文提出LENS框架，通过测量移除节点后邻域熵的变化来识别图数据中的噪声标签节点，并结合重标记机制，提升了图卷积网络在标签噪声下的鲁棒性。

    

    图卷积网络（GCNs）在图结构数据上展现了卓越的性能。然而，其成功在很大程度上依赖于干净标签的可获得性，而这一假设在现实场景中往往不成立。GCNs的可学习性与节点的邻域结构有着内在联系，因此理解该结构在标签噪声存在下如何变化至关重要。在这项工作中，我们研究了节点邻域标签分布在标签噪声影响下如何变化。我们从数学上推导出：标签噪声会平滑这种分布，从而固有地增加其熵，并严重降低类间可区分性。为了应对这一挑战，我们提出了留一熵节点筛选（LENS），这是一个端到端的框架，通过测量移除特定节点后邻域熵的减少程度来识别噪声节点。LENS还进一步引入了重标记机制……

    arXiv:2610.22998v1 Announce Type: cross  Abstract: Graph Convolutional Networks (GCNs) have demonstrated exceptional performance on graph-structured data. However, their success heavily relies on the availability of clean labels, an assumption that often fails in real-world scenarios. The learnability of GCNs is intrinsically linked to the neighborhood structure of nodes, making it critical to understand how this structure changes in the presence of label noise. In this work, we investigate how the neighborhood label distribution of a node changes under label noise. We mathematically derive that label noise smooths this distribution, which inherently increases its entropy and severely decreases interclass distinguishability. To address this challenge, we propose Leave-one-out Entropy Node Screening (LENS), an end-to-end framework that identifies noisy nodes by measuring the reduction in neighborhood entropy when a specific node is removed. LENS further incorporates a relabeling mechani
    
[^85]: GOLD：面向结构感知RNA逆折叠的几何优化潜在扩散模型

    GOLD: Geometric Optimized Latent Diffusion for Structure-Aware RNA Inverse Folding

    [https://arxiv.org/abs/2610.22992](https://arxiv.org/abs/2610.22992)

    GOLD提出几何优化的潜在扩散框架，利用GeoVAE将离散RNA序列与连续侧链角度统一映射到同一潜在流形上，实现序列与结构的协同扩散生成，从而提升RNA逆折叠的结构有效性。

    

    设计能够折叠成特定三维结构的RNA序列是治疗学领域的一项基础性挑战。当前的RNA逆折叠方法通常以三维骨架坐标作为条件，但其生成仅限于一维离散序列，这忽略了对于整体结构稳定性至关重要的连续侧链取向。为了解决这一问题，我们在RNA生成过程中显式地引入了侧链几何信息。与那些在不同原始空间中进行扩散的协同设计框架不同，我们提出了几何优化潜在扩散模型（GOLD），将序列和结构投影到一个统一的潜在流形中。GOLD采用一种新颖的GeoVAE，将离散的RNA序列与连续的侧链角度优雅地整合在一起。通过将这些模态映射到单一、平滑的潜在流形上，GOLD自然地实现了协同扩散过程。为了增强结构有效性，我们进一步引入了……（原文摘要在此处截断）

    arXiv:2610.22992v1 Announce Type: new  Abstract: Designing RNA sequences that fold into specific 3D structures is a fundamental challenge in therapeutics. Current RNA inverse folding methods typically condition on 3D backbone coordinates but the generation is limited to 1D discrete sequences. This ignores the continuous side-chain orientations, which are essential for overall structural stability. To address this issue, we explicitly introduce side-chain geometry into the RNA generative process. Unlike co-design frameworks that diffuse across separate raw spaces, we propose Geometric Optimized Latent Diffusion (\textbf{GOLD}), which project sequence and structure into a unified latent manifold. GOLD employs a novel GeoVAE to elegantly integrate discrete RNA sequences with continuous side-chain angles. By mapping these modalities into a single, smooth latent manifold, GOLD naturally facilitates a synergistic co-diffusion process. To enhance structural validity, we further introduce a tr
    
[^86]: PersonaCUABench：面向多样化用户角色的计算机使用智能体基准测试

    PersonaCUABench: Benchmarking Computer-Use Agents Serving Diverse Personas

    [https://arxiv.org/abs/2610.22990](https://arxiv.org/abs/2610.22990)

    该论文提出 PersonaCUABench 基准，通过让 44 种不同用户角色用自己的措辞重写固定的任务指令，仅改变“谁在提问”就使九个骨干模型的平均任务完成率下降 36%，揭示了现有计算机使用智能体难以真正服务不同用户的个性化需求。

    

    计算机使用智能体（CUA）通常根据固定的任务规范进行评估，但用户可能会遗漏需求、添加约束条件，或要求在执行有重大影响的操作前先获得批准。完成基准测试指令并不保证能服务好提出请求的那个人。我们提出了 PersonaCUABench，该基准将跨 11 个领域的 74 个任务保持固定，仅改变“谁在提问”。一个角色智能体扮演四组共 44 个用户角色之一，用该用户自己的措辞重写每条指令并回答 CUA 的提问，而 CUA 只能看到对话内容，永远看不到用户画像或原始指令。在九个骨干模型上的实验（共 3,330 次运行）显示，相对于原始请求，平均任务完成率下降了 36%。这些损失有两个来源：在丢失分数的案例中，59.9% 的情况下智能体确实响应了用户的要求但仍然做错了；11.2% 是用户主动要求的停止操作，却被评分器判定为失败。即使当某个角色请求……（摘要在此处截断）

    arXiv:2610.22990v1 Announce Type: cross  Abstract: Computer-use agents (CUAs) are commonly evaluated against fixed task specifications, but users may omit requirements, add constraints, or require approval before consequential actions. Completing the benchmark instruction does not guarantee serving the person who asked. We introduce PersonaCUABench, which holds 74 tasks across 11 domains fixed and varies only who asks. A persona agent, playing one of 44 personas in four groups, rewrites each instruction in that user's own words and answers the CUA's questions, while the CUA sees only the dialogue, never the profile or the original instruction. Across nine backbones (3,330 runs), mean task completion falls by 36% relative to the original requests. The losses have two sources: in 59.9% of cells that lose points, the agent engages with what the user asked for and still gets it wrong, and 11.2% are stops the user requested that the checker scores as failures. Even when a persona request is
    
[^87]: 从批准到滥用：针对受框架保护的编码代理的权限提升攻击

    From Approval to Abuse: Privilege Escalation Attacks on Harness-Protected Coding Agents

    [https://arxiv.org/abs/2610.22987](https://arxiv.org/abs/2610.22987)

    本文揭示了编码代理权限批准机制中的新型权限提升攻击：仓库中植入的对抗性内容可诱导代理请求看似合理实则范围过宽的允许规则，一旦用户批准，攻击者便能利用该规则使安全框架失效并滥用代理权限。

    

    编码代理使用执行框架通过强制权限边界来限制提示注入的后果。这些系统会确定性地拦截任何沙箱外的操作，并提示用户进行批准。为了减少批准疲劳，用户可以批准代理提出的允许规则，此后任何匹配该规则的未来操作都将无需提示即可执行。在这项工作中，我们展示了这一机制引入了一种新的风险：不可信的上下文可以影响呈报给用户批准的权限，其后果超出了当前任务的范畴。我们研究了针对这一权限更新过程的权限提升攻击。具体而言，植入仓库中的对抗性上下文可以可靠地引导执行常规任务的代理请求一条看似合理但范围危险过宽的允许规则。如果用户批准了该请求，保存的规则实际上会使框架失效失明，使攻击者能够执行……

    arXiv:2610.22987v1 Announce Type: new  Abstract: Coding agents use execution harnesses to constrain the consequences of prompt injection by enforcing permission boundaries. These systems deterministically intercept any out-of-sandbox action and prompt the user for approval. To reduce approval fatigue, users can approve allow-rules proposed by the agent, after which any future action matching the rule is allowed without a prompt. In this work, we show that this mechanism introduces a new risk: untrusted context can influence the permissions presented for approval, with consequences extending beyond the immediate task. We investigate privilege escalation targeting this permission-update process. Specifically, adversarial context planted in a repository can reliably steer an agent performing a routine task into requesting a plausibly justified, yet dangerously broad allow-rule. If the user approves the request, the saved rule can effectively blind the harness, enabling adversaries to exec
    
[^88]: Prism注意力：面向高效大语言模型推理的提案精细化索引共享机制

    Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference

    [https://arxiv.org/abs/2610.22985](https://arxiv.org/abs/2610.22985)

    提出Prism Attention机制，利用层间显著令牌偏移高度局部化的特性，以“提案-精细化”的方式在层间共享并修正稀疏注意力索引，在避免静态索引复用导致注意力损失的同时，大幅提升大语言模型长上下文推理效率。

    

    大语言模型（LLMs）中的长上下文解码日益受到每一步自回归生成时对不断增长的键值（KV）缓存进行注意力计算所需的内存与计算量的瓶颈制约。尽管令牌级稀疏注意力通过将计算限制在少量相关的历史令牌上来减轻这一负担，但其实际收益往往因上下文增长时需要反复重新计算各层稀疏索引而被抵消。一种常见的补救措施是仅在指定的锚点层计算稀疏索引并跨层共享，但这种静态复用方式忽略了注意力模式随层的演化，导致复用的索引逐渐偏离各层的最优选择，造成显著的注意力损失。通过从注意力动态的视角重新审视跨层稀疏性，我们发现层间显著令牌的偏移具有高度局部性：尽管精确的 top-$\kappa$ 索引在各层之间有所不同……

    arXiv:2610.22985v1 Announce Type: cross  Abstract: Long-context decoding in large language models (LLMs) is increasingly bottlenecked by the memory and computation required to attend over an ever-growing Key-Value (KV) cache at each autoregressive step. Although token-level sparse attention reduces this burden by restricting computation to a small subset of relevant past tokens, its practical gains are often offset by repeatedly recomputing layer-wise sparse indices as the context grows. A common remedy is to share sparse indices across layers by computing them only at designated anchor layers, but such static reuse ignores the layer-wise evolution of attention patterns, causing the reused indices to progressively deviate from the layer-specific optima and incur substantial attention loss. By revisiting cross-layer sparsity from the perspective of attention dynamics, we find that inter-layer salient token shift is highly localized: although the exact top-$\kappa$ indices differ across 
    
[^89]: DiffusionRVGE：通过基于回归的价值梯度估计进行强化学习

    DiffusionRVGE: Reinforcement Learning via Regression-Based Value Gradient Estimation

    [https://arxiv.org/abs/2610.22973](https://arxiv.org/abs/2610.22973)

    该论文提出 DiffusionRVGE，一种通过在最终去噪图像上回归中心化奖励来估计价值梯度的高效强化学习方法，无需奖励导数或伴随反向传播，并能根据奖励的平滑或离散特性自适应调整回滚策略，使单个 SD3.5-Medium 模型在五个奖励上联合训练即可匹敌或超越 DiffusionNFT。

    

    强化学习通过在生成图像上施加奖励来改进扩散模型，但反复去噪和梯度估计使训练成本高昂。我们提出了 DiffusionRVGE，一种高效的方法，它通过在中心化后的最终去噪图像上回归中心化后的奖励来构建有用的更新方向。它既不需要奖励函数的导数，也不需要沿采样轨迹进行伴随反向传播。一个统一的初等变分（first-variation）公式在有或没有轨迹 KL 惩罚的情况下都能确定价值梯度目标。基于高斯噪声的总体分析解释了回归的几何结构，而子空间分析则论证了使用干净图像坐标的合理性。该实现根据奖励的连续性自适应地调整回滚带宽和查询权重：局部回滚适合平滑的奖励，而更宽泛的分组则为离散奖励提供有信息量的对比。在五个奖励上联合训练后，一个 SD3.5-Medium 模型匹配或超越了 DiffusionNFT 的联合训练表现。

    arXiv:2610.22973v1 Announce Type: new  Abstract: Reinforcement learning improves diffusion models using rewards on generated images, but repeated denoising and gradient estimation make training expensive. We introduce \method, an efficient method that constructs useful update directions by regressing centered rewards on centered final denoised images. It requires neither reward derivatives nor adjoint backpropagation through the sampling trajectory. A unified first-variation formulation identifies the value-gradient target with or without a trajectory-KL penalty. A population analysis under Gaussian noising explains the regression geometry, while a subspace analysis motivates clean-image coordinates. The implementation adapts rollout bandwidth and query weights to reward continuity: local rollouts suit smooth rewards, whereas broader groups provide informative contrasts for discrete rewards. Trained jointly on five rewards, one SD3.5-Medium model matches or exceeds DiffusionNFT's joint
    
[^90]: 基于迭代自我精炼的NL2SQL（自然语言转SQL）

    NL2SQL with Iterative Self-Refinement

    [https://arxiv.org/abs/2610.22968](https://arxiv.org/abs/2610.22968)

    该论文提出了一种多智能体、角色驱动的NL2SQL框架，通过领域检测、模式知识构建、KPI感知SQL合成、语法自修复和语义反向检查五个环节的迭代自我精炼，显著减少了查询幻觉并提升了语义保真度，尤其适用于保险等高精度要求领域。

    

    弥合人类意图与机器逻辑之间的差距仍然是非技术用户数据访问中的一个关键挑战，然而大型语言模型（LLMs）仍然不可靠：它们会产生模式元素的幻觉、生成语法无效的查询，或无法捕捉用户意图。这些局限性在保险等领域尤为关键，因为这些领域的KPI（关键绩效指标）、监管约束和遗留关系系统要求高精度。我们提出了一个多智能体、角色驱动的NL2SQL框架，该框架结合了：（1）领域相关性检测，（2）基于模式的知识构建，（3）KPI感知的SQL合成，（4）利用sqlite执行反馈进行的语法级自我修复，以及（5）通过SQL到自然语言的反向工程进行的语义一致性检查。与直接使用LLM提示相比，我们的系统显著减少了查询幻觉并提高了语义保真度。该方法在60个查询上进行了评估（其中50个来自HuggingFace文本转SQL数据集，10个……

    arXiv:2610.22968v1 Announce Type: cross  Abstract: Bridging the gap between human intent and machine logic remains a critical challenge in data access for non-technical users, yet Large Language Models (LLMs) remain unreliable: they hallucinate schema elements, generate syntactically invalid queries, or fail to capture user intent. These limitations are especially critical in domains such as insurance, where KPIs, regulatory constraints, and legacy relational systems demand high precision. We introduce a multi-agent, persona-driven NL2SQL framework that combines (1) domain relevance detection, (2) schema-grounded knowledge construction, (3) KPI-aware SQL synthesis, (4) syntax-level self-repair using sqlite execution feedback, and (5) semantic consistency checking via SQL-to-NL reverse engineering. Our system significantly reduces query hallucination and improves semantic fidelity compared to direct LLM prompting. Evaluated on 60 queries (50 from a HuggingFace text-to-SQL dataset and 10
    
[^91]: 将语言模型对齐至反事实不变性

    Aligning Language Models Towards Counterfactual Invariance

    [https://arxiv.org/abs/2610.22942](https://arxiv.org/abs/2610.22942)

    本文提出一种基于群体级奖励函数的策略梯度技术，使语言模型在对齐过程中实现反事实不变性，解决标准强化学习对齐方法在提升整体回复质量的同时放大跨人口群体反事实偏差的问题。

    

    提示特征的微小变化可能会显著改变大语言模型输出的语义、语气和上下文假设。这些特征可能涉及人口统计学等群体变量或其他敏感的潜在属性。当人口统计信息在因果上与任务无关时，模型对此类输入变量的敏感性是不可取的，但这些信息可能仍与生成语义连贯的回复相关。这使得在保持模型效用的同时实现反事实不变性变得困难。为了研究这一问题，我们首先证明了标准模型对齐方法（如强化学习）在提升整体回复质量的同时，会加大不同人口群体之间回复质量的差异，最终放大了所谓的反事实偏差。为解决这一问题，我们开发了一种策略梯度技术，其中包含一个作用于匹配对的群体级奖励函数。

    arXiv:2610.22942v1 Announce Type: cross  Abstract: Small variations in prompt characteristics can meaningfully alter the semantics, tone, and contextual assumptions of large language model outputs. Such characteristics can involve group variables such as demographics or other sensitive latent attributes. Sensitivity to such input variables is undesirable when demographic information is causally irrelevant to the task, but might still be relevant for generation of semantically coherent responses. This makes the task of enforcing counterfactual invariance while preserving model utility difficult. To study this problem, we first show that standard model alignment recipes (such as reinforcement learning) increase differences in response quality across demographics, while improving overall response quality, ultimately amplifying what is known as counterfactual bias. To address this, we develop a policy gradient technique involving a group-level reward function that operates on matched pairs
    
[^92]: 多种偏见，同一回路：后训练微调预训练的线索路由机制

    Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism

    [https://arxiv.org/abs/2610.22937](https://arxiv.org/abs/2610.22937)

    多种偏见（如谄媚性）共享预训练模型中已存在的同一注意力头回路——助手微调在聊天格式中激活它，强化学习进一步加强它，表明跟随线索源于预训练与调优的副作用，而非对齐本身的问题。

    

    当提示中包含一个线索（即指向错误选项的暗示，例如“我认为答案是B”）时，语言模型往往会放弃正确答案。谄媚性及相关偏见都表现出这种失败，但它们通常被分开研究，并被归咎于对齐问题。我们通过因果方法检验了这一观点，在两个后训练流水线的各个阶段追踪这些偏见背后的注意力头。在对大多数模型的多项选择题测试中，四种线索类型在很大程度上依赖相同的少数注意力头——这是一个在预训练模型中就已存在的共享回路。大多数基础模型在纯文本中就会跟随线索。仅用约200条不含线索的通用助手示例对基础模型进行微调，就能在聊天格式中激活这些注意力头，因此跟随线索可能是助手调优的副作用。最终的强化学习（RL）阶段在几乎所有对比中都加强了线索的影响。在Tulu-3-8B和OLMo-2-7B中，我们交换了六个中间层在RL检查点之间的注意力模式……

    arXiv:2610.22937v1 Announce Type: cross  Abstract: Language models often abandon a correct answer when the prompt contains a cue, a hint toward a wrong option such as "I think the answer is B." Sycophancy and related biases show this failure. They are often studied separately and blamed on alignment. We test this view causally, tracing the attention heads behind these biases through the stages of two post-training pipelines. In multiple-choice tests on most models, four cue types rely largely on the same few heads, a shared circuit already present in pretrained models. Most base models follow cues in plain text. Fine-tuning a base model on about 200 generic assistant examples without cues activates these heads in chat format, so cue-following can be a side effect of assistant tuning. The final reinforcement learning (RL) stage strengthens the effect of cues in almost every comparison. In Tulu-3-8B and OLMo-2-7B, we swap the attention patterns of six middle layers between the RL checkpo
    
[^93]: 多轮对话、短视野：测量越狱攻击中的有效视野压缩

    Multiple Turns, Short Horizons: Measuring Effective-Horizon Compression in Jailbreaks

    [https://arxiv.org/abs/2610.22928](https://arxiv.org/abs/2610.22928)

    该论文定义并实证发现了“有效视野压缩”现象：多轮越狱攻击在新对话中平均只需保留约一半的查询历史即可恢复相同的攻击进度，这段更短的历史还能提供更好的续攻起点并指导训练时学习信号的分配。

    

    多轮越狱攻击利用受害大语言模型的反馈，在多轮对话中不断调整其查询。然而，一旦找到了一条有效的攻击轨迹，尚不清楚还需要保留多少早期的查询历史。在这项工作中，我们通过移除攻击者早期的查询，并在新对话中重新运行剩余查询、同时重新生成受害模型的响应来研究这一问题。我们将有效视野定义为能够恢复相同攻击进度的最短剩余查询序列。实验结果表明，平均而言只需原始查询序列的大约一半即可恢复相同的攻击进度。我们将这一现象称为有效视野压缩。我们证明，这段更短的历史能够恢复一次失败攻击中所达到的最佳点，并为继续攻击提供更好的起点；此外，它还有助于识别哪些早期查询在训练过程中应获得较少的学习信号。

    arXiv:2610.22928v1 Announce Type: new  Abstract: Multi-turn jailbreaks use feedback from a victim large language model to adapt their queries over several turns. However, once an effective trajectory has been found, it is unclear how much of the earlier query history is still needed. In this work, we study this by removing early attacker queries and running the remaining queries again in a new conversation while regenerating the victim's responses. We define the effective horizon as the shortest remaining query sequence that recovers the same attack progress. Empirically, we find that roughly half of the original query sequence is sufficient on average to recover the same attack progress. We call this phenomenon effective-horizon compression. We show that this shorter history can recover the best point reached during an unsuccessful attack and provide a better starting point for continuing the attack. It also helps identify which earlier queries should receive less learning signal duri
    
[^94]: Sep-Latent：将分隔符标记作为隐式推理的天然压缩锚点

    Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning

    [https://arxiv.org/abs/2610.22924](https://arxiv.org/abs/2610.22924)

    本文提出Sep-Latent框架，将分隔符标记作为语义压缩锚点，结合注意力衰减机制与多层级蒸馏实现动态深度的隐式推理，从而在保持逻辑一致性的同时显著降低显式思维链的计算开销。

    

    大型语言模型（LLMs）通过显式的思维链（Chain-of-Thought, CoT）在复杂推理中取得了显著成功，但由此带来的计算开销和标记冗余仍然是效率提升的重大障碍。虽然连续潜在空间中的隐式推理提供了一种有前景的替代方案，但现有方法（如Coconut和CODI）通常依赖固定长度的潜在序列，并存在表示坍缩的问题。在本文中，我们提出了Sep-Latent，这是一种自适应隐式推理框架，它利用分隔符标记作为结构化的语义锚点。通过引入注意力衰减机制，我们迫使模型在训练过程中将推理逻辑聚合和压缩到这些锚点中。随后，我们通过跨所有层级的多层级蒸馏来对齐显式与隐式路径，以确保逻辑一致性。至关重要的是，Sep-Latent通过自适应地终止潜在思考过程，实现了动态推理深度……

    arXiv:2610.22924v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning through explicit Chain-of-Thought (CoT), but the associated computational overhead and token verbosity remain significant barriers to efficiency. While implicit reasoning in continuous latent spaces offers a promising alternative, existing methods like Coconut and CODI often rely on fixed-length latent sequences and suffer from representational collapse. In this paper, we propose Sep-Latent, an adaptive implicit reasoning framework that utilizes separator tokens as structural semantic anchors. By introducing an Attention-Decay mechanism, we force the model to aggregate and compress reasoning logic into these anchors during training. We then align explicit and implicit pathways through multi-level distillation across all layers to ensure logical consistency. Crucially, Sep-Latent enables dynamic reasoning depth by adaptively terminating the latent think
    
[^95]: 奖励引擎：面向开放式工具代理的以奖励为中心的自进化

    Reward Engine: Reward-Centric Self-Evolution for Open-Ended Tool Agents

    [https://arxiv.org/abs/2610.22915](https://arxiv.org/abs/2610.22915)

    提出Reward Engine，通过从任务和初始轨迹中构建包含测试用例与评分标准的冻结RewardPack，为缺乏可执行真值奖励的开放式工具代理任务提供稳定的奖励信号，引导代理自我修正并支持强化学习优化。

    

    开放式工具代理可以通过不同的工具路径完成同一目标，其质量取决于交互轨迹、环境状态和最终产物的共同表现，因此这类任务很少有可执行的真值奖励可用。因此，核心挑战在于如何从已有经验中构建一个奖励，既能暴露当前策略的失败之处，又能在优化过程中保持稳定。我们提出了奖励引擎，它就像一个自我驱动的学生，先弄清楚成功需要什么，然后朝着这个目标进行优化。它从任务和初始轨迹出发，构建一个RewardPack，其中包含用于验证所需正确性的测试用例和用于分级评估质量的评分标准，每一项都作为检查执行，并返回带有证据的判断。冻结的RewardPack随后引导代理修改和筛选其输出，还可以作为强化学习的奖励信号。在WorkBuddy Bench Office和Claw-Eval基准上，所构建的奖励（摘要此处被截断）

    arXiv:2610.22915v1 Announce Type: new  Abstract: Open-ended tool agents can complete the same objective through different tool paths, and their quality depends jointly on the interaction trace, environment state, and final artifacts, so such tasks rarely admit an executable ground-truth reward. The central challenge is therefore to construct a reward from existing experience that exposes the current policy's failures while remaining stationary during optimization. We introduce Reward Engine, which, like a self-motivated student, first works out what success requires and then optimizes toward it. From the task and initial trajectories, it builds a RewardPack of testcases for required correctness and rubrics for graded quality, each executed as a check that returns a judgment with evidence. The frozen RewardPack then guides the agent to revise and select its outputs, and can also serve as a reward for reinforcement learning. On WorkBuddy Bench Office and Claw-Eval, the constructed reward
    
[^96]: COMET：面向任务无关持续视觉-语言学习的解耦蒸馏、路由与容量控制

    COMET: Decoupled Distillation, Routing, and Capacity Control for Task-Agnostic Continual Vision--Language Learning

    [https://arxiv.org/abs/2610.22907](https://arxiv.org/abs/2610.22907)

    提出COMET框架，通过冻结教师的三重亲和力蒸馏、学生感知的跨专家蒸馏以及解耦的路由与容量控制，解决任务无关持续学习中的模态一致性漂移、表示空间干扰和每任务容量不足三大问题。

    

    像CLIP这样的视觉-语言模型（VLM）保留了零样本识别能力，但其在开放世界中的应用往往需要任务无关的持续学习：任务按顺序到来，历史数据不可用，任务身份未知，且主干网络无法重新训练。我们认为，这一设定通过超出稳定性-可塑性权衡之外的三种耦合压力使VLM失效：(1) 模态一致性漂移，(2) 表示空间干扰，(3) 每任务容量不足。我们提出COMET（协作式专家混合迁移，Collaborative Mixture-of-Expert Transfer）来分别解决这三个问题：冻结的教师模型仅在共享的图文池上提供特征参考；三重亲和力蒸馏（Tri-Affinity Distillation）保持图像-图像、文本-文本和图像-文本的几何结构；学生感知的跨专家蒸馏（Student-Aware Cross-Expert Distillation）将新专家与兼容的先验专家对齐，使路由错误能够优雅退化。我们将COMET设计为保持容量与推理……（原文摘要在此截断）

    arXiv:2610.22907v1 Announce Type: cross  Abstract: Vision-language models (VLMs) such as CLIP retain zero-shot recognition ability, but their open-world use often requires task-agnostic continual learning: tasks arrive sequentially, past data are unavailable, task identity is unknown, and the backbone cannot be retrained. We argue that this setting fails VLMs through three coupled pressures beyond the stability-plasticity tradeoff: (1) modal-consistency drift, (2) representation-space interference, and (3) per-task capacity shortfall. We propose COMET (Collaborative Mixture-of-Expert Transfer) to resolve these three issues respectively as follows. A frozen teacher provides only a feature reference on a shared image-text pool; Tri-Affinity Distillation preserves image-image, text-text, and image-text geometry; and Student-Aware Cross-Expert Distillation aligns new experts with compatible prior experts so routing errors degrade gracefully. We design COMET to keep capacity and inference s
    
[^97]: 何时遗忘：面向长程智能体的风险受控上下文自适应

    When to Forget: Risk-Controlled Context Adaptation for Long-Horizon Agents

    [https://arxiv.org/abs/2610.22897](https://arxiv.org/abs/2610.22897)

    该论文提出RECAP框架，通过匹配上下文干预、执行关系预测和风险约束下的独立策略认证，研究长程智能体在何时应保留、压缩或删除失败历史记录，发现失败记录的价值随执行进程动态变化。

    

    长程智能体从失败中学习，但失败记录的价值会随着执行进程的推进而变化。我们研究在何种情况下，相对于保留完整历史的KEEP策略，对同一历史块进行保留、压缩或删除能够改善后续执行。我们提出了RECAP框架，该框架结合了匹配上下文干预、执行关系预测和独立策略认证。通过匹配的环境状态和rollout种子来衡量成功率与续写提示成本的差异；基于失败块与当前执行之间的仅前缀关系来提出有前景的编辑建议。随后，在固定的测量协议下，利用独立任务包在有害干预风险和非正收益风险约束下选择策略。一项五种子纵向案例研究保持失败块及其摘要不变，结果显示：压缩在早期状态失效，但在后期状态成为节省成本的干预手段。在五个骨干模型上，RECAP（摘要在此处截断）

    arXiv:2610.22897v1 Announce Type: new  Abstract: Long-horizon agents learn from failure, but the value of a failure record can change as execution progresses. We study when retaining, compressing, or removing the same history block improves subsequent execution relative to full-history KEEP. We introduce RECAP, a framework that combines matched context interventions, execution-relational prediction, and independent policy certification. Matched environment states and rollout seeds measure success and continuation prompt-cost differences; prefix-only relations between a failure block and the current execution propose promising edits. Independent task bundles then select a policy subject to harmful-intervention and non-positive-benefit risk constraints under a fixed measurement protocol. A five-seed longitudinal case holds both the failure block and its summary fixed: compression fails at an early state and becomes a cost-saving intervention at a later state. Across five backbones, RECAP
    
[^98]: HAPACT：一个面向电影中以人为中心的物理冲击定位的基准数据集

    HAPACT: A Benchmark For Human-Centric Physical Impact Localization in Movies

    [https://arxiv.org/abs/2610.22895](https://arxiv.org/abs/2610.22895)

    本文提出了“以人为中心的冲击定位”这一新任务，并构建了基于电影片段的HAPACT数据集与基准，通过帧级时间标注和基于SMPL-X的空间标注，实现对视频中人体物理冲击的时间与空间双重定位。

    

    视频中关于人类的物理推理研究主要集中于日常交互场景，其中接触、压力、姿态和运动支撑着具身感知和接触感知应用。然而，像冲击事件这类短暂而强烈的现象仍然相对缺乏探索，尽管它们对于更广泛地理解突发物理事件如何影响人体具有重要意义。为填补这一空白，我们提出了以人为中心的冲击定位这一新任务，即在视频中从时间维度定位目标物理冲击，并在人体上进行空间定位。我们进一步提出了HAPACT——一个基于电影片段构建的以人为中心的冲击数据集与基准，包含超过2800个视频-查询对以及超过20000个细粒度的人工标注帧。我们围绕基于观众视角的标注、文本条件下的定位，以及帧级时间粒度和基于SMPL-X的空间粒度的监督来构建HAPACT。为了验证我们基准的有效性……（摘要在此处截断）

    arXiv:2610.22895v1 Announce Type: cross  Abstract: Physical reasoning about humans in videos has largely focused on everyday interactions, where contact, pressure, pose, and motion support embodied perception and contact-aware applications. Still, short and forceful properties like impact events remain relatively underexplored, despite their relevance to broader understanding of how abrupt physical events affect the human body. To fill this gap, we introduce human-centric impact localization, a new task that localizes target physical impacts temporally within a video and spatially on the human body. We further present HAPACT, a HumAn-centric imPACT dataset and benchmark built from movie clips, containing over 2.8K video-query pairs with more than 20K fine-grained human-annotated frames. We built HAPACT around viewer-grounded annotation, text-conditioned localization, and supervision at frame-level temporal and SMPL-X-based spatial granularity. To demonstrate the effectiveness of our be
    
[^99]: 基于能力的大语言模型逻辑推理评估

    Competence-Based Evaluation of Logical Reasoning in Large Language Models

    [https://arxiv.org/abs/2610.22894](https://arxiv.org/abs/2610.22894)

    本文提出“基于能力的评估”（CBE）新范式，通过参数化生成器调节复杂度并使用保留答案的多种表述形式，来衡量大语言模型逻辑推理的泛化能力，而非仅依赖单一准确率指标。

    

    大语言模型（LLM）的最新进展使其在多种任务上展现出令人瞩目的性能；然而，在逻辑推理方面仍存在不足，包括在语言表述变化下的不一致性。为了提升大语言模型的推理能力，我们必须能够准确衡量其推理能力。当前的基准测试主要报告在选定推理任务上的单一准确率数字。这种基于性能的评估方法对于模型在推理方面的泛化能力说明甚少。为解决这一局限，我们主张采用一种不同的评估范式，称为基于能力的评估（CBE），它测试模型能否在一系列同族问题中可靠地运用相同的能力。参数化生成器在保持所需能力不变的情况下扩展问题复杂度，而匹配的、保留答案的多种表述方式则用于测试正确性是否超越单一表述形式。我们在基础的排序任务上实现了CBE。

    arXiv:2610.22894v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have brought impressive performance across diverse tasks; however, shortcomings remain in logical reasoning, including inconsistencies under linguistic variation. To improve the reasoning capacity of LLMs, we must be able to accurately measure their reasoning capacity. Current benchmarks primarily report single accuracy numbers on selected reasoning tasks. Such a performance-based evaluation methodology says little about a model's generalization capacity in reasoning. To address this limitation, we argue for a different evaluation paradigm, called competence-based evaluation (CBE), which tests whether models reliably apply the same competence across a family of problems. A parameterized generator scales complexity while keeping the required competence fixed, and matched answer-preserving presentations test correctness beyond one formulation. We instantiate CBE on elementary ordering tasks
    
[^100]: 批量上下文强化 II：RLVR 需要长采样窗口吗？

    Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?

    [https://arxiv.org/abs/2610.22887](https://arxiv.org/abs/2610.22887)

    提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。

    

    长采样窗口使得可验证奖励强化学习（RLVR）的代价高昂，但简单缩短窗口可能会丢失正确性信号：当每个单问题的采样在得出答案之前就被截断时，组相对准确率优势将不复存在。作为批量上下文强化（Batched Contextual Reinforcement, BCR）的技术延续，我们研究了并发奖励如何放宽这种对长窗口的依赖。并发约束强化将三个可独立验证的问题置于同一个有长度上限的训练完成中。在截断前完成的正确答案仍可获得奖励，而共享预算在无需显式长度惩罚的情况下，鼓励更简短的有效推理。我们将该机制与条件奖励方差和更新方差分析联系起来，并在4B和8B数学推理模型上进行了研究。与使用32k窗口的单问题GRPO相比，使用8,000-token窗口的CCR在4B模型上将每训练步的token数从975,464降至174,957。

    arXiv:2610.22887v1 Announce Type: cross  Abstract: Long rollout windows make reinforcement learning with verifiable rewards (RLVR) expensive, yet simply shortening them can remove the correctness signal: when every single-problem rollout truncates before an answer, group-relative accuracy advantages vanish. As a technical continuation of Batched Contextual Reinforcement (BCR), we study how concurrent rewards relax this dependence on long windows. Concurrent Constraint Reinforcement (CCR) places three independently verifiable problems in one capped training completion. Correct answers completed before truncation retain credit, while a shared budget encourages shorter useful reasoning without an explicit length penalty. We connect this mechanism to conditional reward- and update-variance analyses and study 4B and 8B mathematical reasoning models. Against single-problem GRPO with a 32k window, CCR with an 8,000-token window reduces tokens per training step from 975,464 to 174,957 on 4B an
    
[^101]: 学会被引导：面向无分类器引导的潜在空间几何塑造

    Learning to Be Guided: Shaping Latent Geometry for Classifier-Free Guidance

    [https://arxiv.org/abs/2610.22885](https://arxiv.org/abs/2610.22885)

    本文提出原型正则化方法，通过在潜在空间中塑造“类间方差大于类内方差”的类别几何结构，使潜在空间天然适配无分类器引导（CFG），从而提升条件扩散模型的引导生成性能。

    

    无分类器引导是条件扩散模型的标准组成部分，引导生成性能已成为衡量最先进模型的事实标准。近期关于潜在扩散模型的研究致力于学习更容易用扩散过程建模的潜在空间。然而，在CFG下，生成效果还取决于条件在潜在空间中的排列方式。我们在离散类别条件下研究了这种排列方式，称之为“类别几何”，并探究什么样的类别几何能使潜在空间与CFG良好协同工作。在高斯类别模型下，我们证明在总方差固定的情况下，将方差从类内转移到类间可以使各类别在噪声下更易区分，并推导出一个噪声后验目标，其逐样本的下降方向与oracle CFG修正成正比。基于这一分析，我们提出了原型正则化，这是一种轻量级的替代方法（摘要在此处被截断）

    arXiv:2610.22885v1 Announce Type: cross  Abstract: Classifier-free guidance (CFG) is a standard component of conditional diffusion models, and guided generation performance has become the \textit{de facto} measure of state-of-the-art models. Recent work on latent diffusion models seeks to learn latent spaces that are easier to model with diffusion. However, under CFG, generation additionally depends on how conditions are arranged in the latent space. We study this arrangement under discrete class conditions, termed \emph{class geometry}, and investigate what class geometry allows a latent space to work well with CFG. Under a Gaussian class model, we show that shifting variance from intra- to inter-class at fixed total variance makes classes more distinguishable under noise, and derive a noisy posterior objective whose per-sample descent direction is proportional to the oracle CFG correction. Based on the analysis, we propose Prototype Regularization (PR), a lightweight surrogate for th
    
[^102]: ToPA：用于富有表现力且高效微调的分块Toeplitz自适应方法

    ToPA: Block-wise Toeplitz Adaptation for Expressive and Efficient Fine-Tuning

    [https://arxiv.org/abs/2610.22883](https://arxiv.org/abs/2610.22883)

    本文提出ToPA方法，利用沿对角线元素恒定的分块Toeplitz矩阵对预训练模型进行参数高效微调，在不施加低秩或稀疏性显式约束的情况下，以紧凑的参数化方式获得更接近全量微调的表现力与效率。

    

    参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）已成为适配大型预训练模型的一种主流策略。在各类PEFT技术中，基于重参数化的方法，尤其是采用低秩或稀疏结构的方法，受到了广泛关注。然而，这些方法的性能往往不及全量微调。具体而言，低秩方法限制了更新的秩，这与全量微调本身固有的高秩特性并不一致；而基于稀疏性的技术则限制了参数空间的灵活性，并破坏了权重之间的连通性。为了克服这些局限，我们探索使用Toeplitz矩阵，其沿每条对角线的元素均为常数，因此在无需施加显式秩约束或稀疏性约束的情况下，即可提供一种紧凑的参数化方式。一种直观的做法是通过Toeplitz矩阵的乘积来更新权重。然而，与RNN类似，长程乘性……（原文摘要在此处被截断）

    arXiv:2610.22883v1 Announce Type: new  Abstract: Parameter-Efficient Fine-Tuning (PEFT) has emerged as a prevalent strategy for adapting large pre-trained models. Among PEFT techniques, reparameterization-based methods, particularly those that employ low-rank or sparsity structures, have attracted significant attention. However, these approaches often underperform full fine-tuning. Specifically, low-rank methods constrain the update rank, which is misaligned with the inherently high-rank nature of full fine-tuning. In contrast, sparsity-based techniques limit the flexibility of the parameter space and compromise connectivity across weights. To overcome these limitations, we explore the use of Toeplitz matrices, whose entries are constant along each diagonal, therefore providing a compact parameterization without imposing explicit rank or sparsity constraints. A straightforward approach is to update weights via a product of Toeplitz matrices. However, similar to RNNs, long multiplicativ
    
[^103]: 无需适应的合作：集体风险困境中的大语言模型智能体

    Cooperation Without Adaptation: LLM Agents in a Collective-Risk Dilemma

    [https://arxiv.org/abs/2610.22882](https://arxiv.org/abs/2610.22882)

    研究发现大语言模型智能体在集体风险困境博弈中采用固定而非适应性的合作策略，即使在零风险且贡献无法改善自身收益的情况下仍会贡献资源，表明其合作行为源于策略惯性而非对环境风险的理性适应。

    

    语言模型智能体越来越多地被部署用于代表人们行事，在这些场景中，个体决策会影响共同的结果。我们研究了五个生产级模型是否会根据不断变化的风险和伙伴行为调整其贡献，还是遵循相对固定的策略。在集体风险困境博弈（CRG）中，六名玩家在十轮游戏中向一个共享池贡献资源；如果共享池未达到目标，一场灾难可能会摧毁他们剩余的资源。四个模型在每个正风险水平上都贡献了大约相等的份额，五个模型中有四个甚至在风险为零时也会做出贡献，尽管此时贡献无法改善其收益。明确声明只有智能体自身的收益才重要时，大多数模型的零风险贡献有所减少，而更中性的提示词则没有这种效果。几个模型能够正确识别在不同风险水平下哪种行动提供更高的预期收益，但它们的（摘要在此处截断）

    arXiv:2610.22882v1 Announce Type: cross  Abstract: Language-model agents are increasingly deployed to act on behalf of people in settings where individual decisions shape shared outcomes. We examine whether five production models adapt their contributions to changing risks and partner behaviour, or instead follow a relatively fixed strategy. In a collective-risk dilemma game (CRG), six players contribute to a shared pool over ten rounds; if the pool falls short of its target, a catastrophe may destroy their remaining resources. Four models contributed approximately an equal share at every positive risk level, and four of the five also contributed when risk was zero, even though contributing could not improve their payoff. Explicitly stating that only the agent's own payoff mattered reduced zero-risk contributions for most models, while a more neutral prompt did not. Several models correctly identified which action offered the higher expected payoff at different risk levels, yet their c
    
[^104]: 回顾过去，修正未来：时间序列预测的误差反馈校准

    Look Back, Correct Forward: Error Feedback Calibration for Time Series Forecasting

    [https://arxiv.org/abs/2610.22872](https://arxiv.org/abs/2610.22872)

    提出了一种完全在测试时运行的误差反馈校准框架EFC，通过从重叠历史预测误差中构建因果误差对角线，并利用累积误差传输机制将其映射为未来残差修正，实现对时间序列预测的确定性事后校正。

    

    时间序列预测模型越来越多地被部署在持续变化的环境中。当预测发布之后，未来的观测值会逐渐到来，并揭示模型在哪些方面出现了失败。现有方法主要将这些观测值当作参数更新的监督信号。然而，经过验证的预测误差本身能否为修正未来的预测提供因果反馈，这一点尚不明确。为了解决这一问题，我们提出了误差反馈校准（Error Feedback Calibration, EFC），这是一个完全在测试时运行的确定性事后修正框架。EFC从重叠历史预测的已揭示误差中构建因果误差对角线，为冻结的预测器的近期失败行为提供了一种全新的视角。随后，累积误差传输通过累积估计将该误差状态映射到未来的残差上，同时利用时间上的样本外验证来控制修正强度。

    arXiv:2610.22872v1 Announce Type: new  Abstract: Time series forecasting models are increasingly deployed in continuously evolving environments. After a forecast is issued, future observations gradually arrive and reveal how the model has failed. Existing methods mainly treat such observations as supervision for parameter updates. However, it remains unclear whether verified forecasting errors can themselves provide causal feedback for correcting future predictions. To address this, we introduce Error Feedback Calibration (EFC), a fully test-time framework for deterministic post-hoc correction of time series forecasting. EFC constructs a Causal Error Diagonal (CED) from revealed errors of overlapping historical forecasts, providing a fresh view of the frozen forecaster's recent failure behavior. Then Cumulative Error Transport (CET) maps this error state to future residuals using cumulative estimation, while temporal out-of-sample validation controls the correction strength. Temporal R
    
[^105]: ShadowFPT：通过影子触发器对联邦提示调优实施后门攻击

    ShadowFPT: Backdooring Federated Prompt Tuning via Shadow Triggers

    [https://arxiv.org/abs/2610.22871](https://arxiv.org/abs/2610.22871)

    提出ShadowFPT后门攻击，利用联邦提示调优中冻结共享骨干网络的攻击面，通过预训练影子触发器在表示空间中将恶意输入引导至目标类别，从而在保持提示更新看似良性的情况下实现隐蔽后门攻击。

    

    联邦提示调优通过冻结预训练骨干网络、仅在客户端间优化轻量级提示参数的方式，来适配大型视觉-语言模型。尽管这种设计提升了通信效率和参数效率，却也带来了一种被忽视的安全风险。由于骨干网络是共享且冻结的，恶意客户端可以在保持其上传的提示更新接近良性更新的同时，通过表示空间诱导后门行为。我们提出了ShadowFPT，一种利用这种冻结骨干网络攻击面的针对性后门攻击。ShadowFPT首先针对冻结的CLIP视觉编码器预训练一个可学习的“影子触发器”，该过程可以使用辅助公开数据或恶意客户端的本地数据，使被触发的输入在表示空间中被引导至目标类别。在联邦提示调优过程中，恶意客户端在当前全局提示下调整该触发器，然后进行优化……

    arXiv:2610.22871v1 Announce Type: new  Abstract: Federated Prompt Tuning (FPT) adapts large vision--language models by freezing the pretrained backbone and optimizing only lightweight prompt parameters across clients. Although this design improves communication and parameter efficiency, it creates an overlooked security risk. Since the backbone is shared and frozen, a malicious client can induce backdoor behavior through the representation space while keeping its uploaded prompt updates close to benign ones. We propose \textbf{ShadowFPT}, a targeted backdoor attack that exploits this frozen-backbone attack surface. ShadowFPT first pretrains a learnable \emph{Shadow Trigger} against the frozen CLIP visual encoder, using either auxiliary public data or the malicious client's local data, so that triggered inputs are steered toward the target class in representation space. During federated prompt tuning, the malicious client adapts the trigger under the current global prompt and then optim
    
[^106]: MS-Eval：面向MCP密集型工作场景的子智能体委派与编排基准测试

    MS-Eval: Benchmarking Sub-Agent Delegation and Orchestration in MCP-Intensive Working Scenarios

    [https://arxiv.org/abs/2610.22869](https://arxiv.org/abs/2610.22869)

    该论文提出了MS-Eval——首个针对MCP密集型工作场景下主智能体委派与子智能体执行（MAD-SAE）范式的统一评估套件，包含基准测试、执行框架、委派与编排指标及MSQ综合指数，并发现子智能体专业化是最佳实践、开源模型基本具备与前沿模型竞争的实力。

    

    LLM智能体正日益融入真实工作场景，但它们需要处理多样化的领域特定工具和复杂的多子任务问题，而顺序式工具调用在解决这类问题时效率低下，且会造成严重的上下文膨胀。主智能体委派与子智能体执行范式通过将子任务委派给专门的子智能体，在隔离的上下文中并行执行，从而应对这些问题，但其委派与编排的质量尚未得到量化评估。我们提出MS-Eval，这是一个统一的评估套件，包含一个MCP密集型基准测试、一个受控执行框架、专门设计的委派与编排指标，以及用于综合评估的MSQ指数。对四种MAD-SAE变体和八个前沿模型的评估揭示了：（1）子智能体专业化是最佳实践；（2）开源模型在绝大多数情况下具有竞争力，仅存在少数例外；（3）MAD-SAE通常会带来更高的成本……（原文摘要至此截断）

    arXiv:2610.22869v1 Announce Type: new  Abstract: LLM agents are increasingly integrated into real-world works, yet they must handle diverse domain-specific tools and complex multi-sub-task problems that sequential tool invocation resolves inefficiently and with severe context bloat. The Main-Agent Delegation and Sub-Agent Execution (MAD-SAE) paradigm addresses these issues by delegating sub-tasks to specialized sub-agents for parallel execution in isolated contexts, but its delegation and orchestration quality remains unquantified. We introduce MS-Eval, a unified evaluation kit comprising an MCP-intensive benchmark, a controlled execution harness, tailored delegation and orchestration metrics, and the MSQ index for comprehensive assessment. Evaluating four MAD-SAE variants and eight frontier models reveals that (1) sub-agent specialization (Spec) is the best practical practice; (2) open-source models are largely competitive with few exceptions; and (3) MAD-SAE generally raises cost wit
    
[^107]: Evo-B&B：面向混合整数线性规划（MILP）的行为感知LLM引导的分支与节点选择策略联合演化

    Evo-B&B: Behavior-Aware LLM-Guided Joint Evolution of Branching and Node Selection Policies for MILP

    [https://arxiv.org/abs/2610.22861](https://arxiv.org/abs/2610.22861)

    提出Evo-B&B框架，利用LLM引导的演化机制结合行为感知的多岛屿策略，首次联合演化MILP分支定界中的分支与节点选择策略，实现有竞争力的求解性能并向大规模实例良好泛化。

    

    在混合整数线性规划的分支定界算法中，分支和节点选择是两个关键且紧密耦合的决策，然而现有大多数基于学习的方法侧重于分别优化它们。我们提出了Evo-B&B，这是一个由大语言模型（LLM）引导的演化框架，利用端到端的SCIP求解反馈来联合演化可执行的分支策略和节点选择策略。为了提高搜索效率并减少对行为相似策略的冗余评估，Evo-B&B引入了一种行为感知的多岛屿演化机制。该机制通过在一组固定的分支定界决策状态上，从分支行为和节点选择行为两方面衡量策略之间的相似性，并利用这种相似性来保持种群多样性。我们在四个标准MILP基准测试上、跨多个实例规模对Evo-B&B进行了评估。实验结果表明，Evo-B&B取得了有竞争力的求解性能，并能很好地泛化到更大规模的实例。

    arXiv:2610.22861v1 Announce Type: new  Abstract: In branch-and-bound for Mixed-Integer Linear Programming (MILP), branching and node selection are two critical and tightly coupled decisions, yet most existing learning-based approaches focus on optimizing them separately. We propose Evo-B&B, an LLM-guided evolutionary framework that jointly evolves executable branching and node selection policies using end-to-end SCIP solving feedback. To improve search efficiency and reduce redundant evaluations of behaviorally similar policies, Evo-B&B introduces a behavior-aware multi-island evolutionary mechanism. It measures policy similarity from branching and node-selection behaviors on a fixed set of B&B decision states and uses this similarity to preserve population diversity. We evaluate Evo-B&B on four standard MILP benchmarks across multiple instance scales. Experimental results show that Evo-B&B achieves competitive solving performance and generalizes well to larger-scale instances, while a
    
[^108]: 面向高效LLM树搜索的不确定性感知预算分配

    Uncertainty-Aware Budget Allocation for Efficient LLM Tree Search

    [https://arxiv.org/abs/2610.22858](https://arxiv.org/abs/2610.22858)

    该论文提出无需训练的不确定性感知分配框架（UAA），利用兄弟动作的语义分散度在扩展前预测下游后续内容的冗余性，自适应地为LLM树搜索的每一步分配预算，从而大幅提升多步推理的效率。

    

    树搜索能够增强大型语言模型（LLM）的多步推理能力，然而现有方法以固定预算扩展每一步，而不考虑扩展的边际效用。我们首先证明，在现有的搜索算法中，从共享中间状态采样得到的后续内容大多是重复的，并坍缩为少数几个不同的答案。关键的是，我们发现兄弟动作的语义分散度可以预测其下游后续内容的冗余性，从而允许在扩展之前就确定子树的预算。信息论分析为这一做法提供了理论支撑：将每步预算视为估计该步剩余答案熵的蒙特卡洛样本量，我们推导出了一个随兄弟分散度收缩而减小的充分预算。据此，我们提出了不确定性感知分配（Uncertainty-Aware Allocation, UAA），这是一个无需训练的框架，它将价值与不确定性解耦：价值决定哪些候选被（原文在此处截断）

    arXiv:2610.22858v1 Announce Type: cross  Abstract: Tree search enhances multi-step reasoning in large language models (LLMs), yet prevailing methods expand every step with a fixed budget, irrespective of the marginal utility of expansion. We first show, across existing search algorithms, that continuations sampled from a shared intermediate state are largely duplicates and collapse to a few distinct answers. Critically, we find that the semantic dispersion of sibling actions predicts the redundancy of their downstream continuations, permitting a subtree's budget to be fixed before expansion. An information-theoretic analysis motivates this: treating the per-step budget as the Monte Carlo sample size for estimating a step's residual answer entropy, we derive a sufficient budget that shrinks with sibling dispersion. We accordingly propose $\textbf{Uncertainty-Aware Allocation}$ (UAA), a training-free framework that decouples value from uncertainty: values determine which candidates are p
    
[^109]: 基于流反转动作锚点的高样本效率强化学习策略改进

    Sample-Efficient Reinforcement Learning for Policy Improvement via Flow-Reversed Action Anchors

    [https://arxiv.org/abs/2610.22852](https://arxiv.org/abs/2610.22852)

    本文提出AnchorQ方法，利用流反转将参考行为映射为结构化的潜在动作锚点，并通过价值引导的由粗到细粒子采样，显著提升了强化学习改进机器人策略时的样本效率。

    

    通用机器人策略提供了强大的行为先验，但要达到部署级别的可靠性和吞吐量仍然困难。强化学习（RL）可以改进这些策略，但其有效性受到在线交互成本的限制。现有方法通过将噪声空间视为连续动作空间来适配冻结的流匹配策略。我们证明，这种在行为上缺乏结构的空间会导致探索效率低下，往往使这些方法无法在具有挑战性的任务上提取有意义的学习信号。我们提出了AnchorQ，它通过一种结构化、可细化的潜在动作表示实现由粗到细的策略改进。在每个决策步骤中，AnchorQ利用流反转将参考行为映射为状态条件下的潜在锚点，在这些锚点周围采样局部粒子，并应用价值引导采样来探索有前景的细化方向。我们的分析表明，锚点降低了id（摘要截断）

    arXiv:2610.22852v1 Announce Type: cross  Abstract: Generalist robot policies offer strong behavioral priors, but achieving deployment-level reliability and throughput remains difficult. Reinforcement learning (RL) can improve these policies, yet its effectiveness is constrained by the cost of online interaction. Existing methods adapt frozen flow-matching policies by treating noise space as a continuous action space. We show that this behaviorally unstructured space makes exploration inefficient, often preventing these methods from extracting meaningful learning signals on challenging tasks. We introduce AnchorQ, which enables coarse-to-fine policy improvement through a structured, refinable latent action representation. At each decision step, AnchorQ uses flow reversal to map reference behaviors to state-conditioned latent anchors, samples local particles around those anchors, and applies value-guided sampling to explore promising refinements. Our analysis shows that anchors reduce id
    
[^110]: 依赖假设：随着语言资源水平的下降，思维链的忠实度上升

    The Reliance Hypothesis: Chain-of-Thought Faithfulness Rises as Language Resource Level Falls

    [https://arxiv.org/abs/2610.22841](https://arxiv.org/abs/2610.22841)

    论文提出“依赖假设”：当模型无法回忆答案时会更加依赖思维链推理，因此CoT忠实度在知识密集型任务上随语言资源水平的下降而上升。

    

    思维链的忠实度对于监控、解释和安全部署大型语言模型至关重要，然而目前的研究大多集中在英语上。语言之间存在资源水平的差异，即可用于学习的文本数量不同，因此模型在低资源语言中掌握的知识较少，这可能改变它们对自身推理的依赖程度。据我们所知，我们是首个将CoT忠实度同时与资源水平和任务类型（知识密集型或推理密集型）关联起来的研究。在来自多语言大规模多任务语言理解（MMMLU）基准的八种语言的平行多项选择题上，忠实度在知识密集型任务上随着资源水平的下降而上升，但在推理密集型任务上并非如此，这一差异在四个模型和多种忠实度度量上均呈现出相同的方向。我们提出并找到了支持“依赖假设”的证据：当模型无法回忆起答案时（例如在低资源语言中），它们会更加依赖自己的推理。

    arXiv:2610.22841v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) faithfulness matters for monitoring, explaining and safely deploying large language models, yet it has been studied mostly in English. Languages differ in level, the amount of text available to learn from, so models know less in lower-resource languages, which may change how much they rely on their reasoning. To our knowledge, we are the first to relate CoT faithfulness to both resource level and task type (knowledge- or reasoning-intensive). On parallel multiple-choice questions from Multilingual Massive Multitask Language Understanding (MMMLU) in eight languages, faithfulness rises as resource level falls on knowledge-intensive tasks but not on reasoning-intensive ones, a gap that points the same way across four models and several faithfulness measures. We propose, and find support for, the Reliance Hypothesis: models rely more on their reasoning when they cannot recall the answer, as in lower-resource language
    
[^111]: 世界动作评论家模型：面向策略学习的联合前瞻与评判

    World Action Critic (WAC) Model: Joint Foresight and Criticism for Policy Learning

    [https://arxiv.org/abs/2610.22825](https://arxiv.org/abs/2610.22825)

    提出世界动作评论家（WAC）模型，通过联合学习动作候选的未来预测与后果评估，并利用评论家监督直接塑造预测表示，从而实现对反事实动作的有效推理，提升策略学习效果。

    

    世界模型通过预测动作的后果并为策略学习提供预测性前瞻，在序列决策中展现出相当大的潜力。然而，学习对反事实动作进行推理仍然具有挑战性，因为预测目标可能低估对动作评估至关重要的信息，而未执行的动作又缺乏可观察的未来以进行直接监督。为了应对这些挑战，我们提出了世界动作评论家（WAC）模型，该模型生成多模态动作候选，并联合学习以候选为条件的未来预测与后果评估。在我们提出的框架中，评论家监督在推演过程中直接塑造中间预测表示，促使其保留区分动作质量所需的信息。可观察的未来用于监督已执行动作的预测，而外部任务质量目标则为反事实动作提供评估信号。

    arXiv:2610.22825v1 Announce Type: new  Abstract: World models have shown considerable promise in sequential decision-making by anticipating the consequences of actions and providing predictive foresight for policy learning. However, learning to reason about counterfactual actions remains challenging, since predictive objectives may underemphasize information critical for action evaluation, while unexecuted actions lack observed futures for direct supervision. To address these challenges, we propose the World Action Critic (WAC) Model, which proposes multimodal action candidates and jointly learns candidate-conditioned future prediction and consequence evaluation. In our proposed framework, critic supervision directly shapes intermediate predictive representations during rollout, encouraging them to retain information needed to distinguish action quality. Observed futures supervise prediction for executed actions, while external task-quality targets provide evaluation signals for counte
    
[^112]: 压缩先行，内容决定：奖励模型中的涌现性失调

    Compression Precedes, Content Decides: Emergent Misalignment in Reward Models

    [https://arxiv.org/abs/2610.22805](https://arxiv.org/abs/2610.22805)

    首次研究奖励模型中的涌现性失调，发现对奖励模型在单一领域进行有害偏好微调会泛化到七个其他领域，且奖励模型比其底层语言模型更容易发生偏好反转。

    

    涌现性失调——即Betley等人[2025]首次展示的狭义有害微调产生的域外泛化现象——已在语言模型中得到广泛研究，但尚未在监督语言模型的奖励模型中进行研究。在本文中，我们对两个最先进的奖励模型Skywork-Reward-V2-Llama-3.1-8B和Skywork-Reward-V2-Qwen3-8B[Liu等人，2025]进行微调，结果表明，对奖励模型进行狭义微调使其偏好某一领域（医学或金融）的有害内容后，这种偏好会泛化到七个留出领域。这并非模型退化，因为在标准RewardBench 2评估[Malik等人，2025]中，微调后的奖励模型总体仅损失4.8分和5.3分（RM-Llama），同时仍优于未修改的RM-Qwen模型。随后，我们使用DPO[Rafailov等人，2023]和SFT在相同数据上对底层语言模型Llama-3.1-8B-Instruct和Qwen3-8B进行微调。我们发现，与底层语言模型相比，奖励模型更容易受到偏好反转的影响（原文摘要到此被截断）。

    arXiv:2610.22805v1 Announce Type: new  Abstract: Emergent Misalignment - the out of domain generalization of narrow harmful fine-tuning first shown by Betley et al. [2025] has been widely studied in language models but not in reward models that supervise them. In this paper, we fine-tune two state-of-the-art reward models, Skywork-Reward-V2-Llama-3.1-8B and Skywork-Reward-V2-Qwen3-8B [Liu et al., 2025] and show that narrow fine-tuning reward models to prefer harm in one domain (medical or financial) generalizes to seven held out domains. This is not degeneracy, as the fine-tuned reward models lose only 4.8 and 5.3 points overall (RM-Llama) on the standard RewardBench 2 evaluation [Malik et al., 2025] while still outperforming the unmodified RM-Qwen model. We then fine-tune the underlying language models, Llama-3.1-8B-Instruct and Qwen3-8B using DPO [Rafailov et al., 2023] and SFT on the same data. We find that reward models are more vulnerable to preference inversions compared to their
    
[^113]: STRIDE：评估驾驶边缘场景中的时空推理能力

    STRIDE: Evaluating Spatiotemporal Reasoning in Driving Edge Cases

    [https://arxiv.org/abs/2610.22795](https://arxiv.org/abs/2610.22795)

    STRIDE是一个面向自动驾驶安全关键长尾场景的以物体为中心的基准，通过空间感知、空间理解、时间记忆、时间外推、轨迹预测和场景上下文感知六大任务族，全面评估视觉语言模型连接过去空间观察与未来后果的复杂时空推理能力。

    

    在自动驾驶中，仅仅感知周围物体是不够的：如果无法理解物体如何随时间移动和交互，就可能导致危险的决策。因此，评估视觉语言模型（VLM）需要测试它们能否将过去的空间观察与未来后果联系起来。然而，现有的基准通常通过基础的感知问题来评估边缘场景，或者仅在简单场景中测试时间理解能力，使得复杂的时空推理研究尚不充分。我们提出了STRIDE（SpatioTemporal Reasoning In Driving Edge Cases，驾驶边缘场景中的时空推理），这是一个面向长尾、安全关键驾驶场景的以物体为中心的基准。STRIDE涵盖六个任务族：空间感知、空间理解、时间记忆、时间外推、轨迹预测和场景上下文感知。该基准将通过选择题探测单个时空判断与相关的自由格式视觉问答（VQA）相结合，要求模型

    arXiv:2610.22795v1 Announce Type: cross  Abstract: In autonomous driving, perceiving surrounding objects is not enough: failing to understand how they move and interact over time can lead to dangerous decisions. Evaluating vision-language models (VLMs) therefore requires testing whether they connect past spatial observations to future consequences. Yet existing benchmarks often assess edge cases through basic perception questions or test temporal understanding in simple scenarios, leaving complex spatiotemporal reasoning underexplored. We introduce STRIDE (SpatioTemporal Reasoning In Driving Edge Cases), an object-centric benchmark for long-tail, safety-critical driving scenarios. STRIDE covers six task families: spatial perception, spatial understanding, temporal memory, temporal extrapolation, trajectory prediction, and scene-context awareness. It pairs multiple-choice questions probing individual spatiotemporal judgments with related free-form visual question answering (VQA) requiri
    
[^114]: 剪除泄漏神经元：通过联合嵌入引导的剪枝缓解微调大语言模型中的隐私数据泄露

    Prune the Leaky Neurons: Mitigating Private Data Leakage in Fine-Tuned LLMs via Joint-Embedding-Guided Pruning

    [https://arxiv.org/abs/2610.22792](https://arxiv.org/abs/2610.22792)

    提出 JEPA-Prune 框架，利用在无个人身份信息的替代视图上通过联合嵌入预测架构训练的公开模型来引导事后神经元剪枝，在保持模型性能的同时有效缓解微调大语言模型中的隐私数据泄露问题。

    

    在专有记录上微调大语言模型（LLM）可以实现领域适配，但会带来两个挑战：（i）已记忆的隐私数据（如个人身份信息 PII）存在隐私泄露风险，（ii）需要在移除隐私数据与保持模型性能之间取得平衡。现有防御方法，包括数据清洗和差分隐私，往往会降低模型性能，而隐私神经元编辑方法则缺乏一个无 PII 的参照来区分身份特定神经元与支持有用文本的神经元。我们提出 JEPA-Prune，这是一个由仅在无 PII 视图上训练的模型所引导的事后神经元剪枝框架。在微调后的私有模型之外，JEPA-Prune 使用联合嵌入预测架构（JEPA）在同一记录的替代视图和掩码视图上训练一个公开模型，从而在不包含标识符的情况下捕获领域知识。跨模型表示探针用于量化私有表示与公开表示之间的偏差……

    arXiv:2610.22792v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) on proprietary records enables domain adaptation but introduces two challenges: (i) privacy leakage of memorized private data, such as personally identifiable information (PII), and (ii) balancing private data removal with utility preservation. Existing defenses, including data sanitization and differential privacy, often degrade performance, while privacy-neuron editing lacks a PII-free reference to distinguish identity-specific neurons from those supporting useful text. We propose JEPA-Prune, a post-hoc neuron pruning framework guided by a model trained exclusively on PII-free views. Alongside the fine-tuned private model, JEPA-Prune, trains a public model on surrogate and masked views of the same records using a joint-embedding predictive architecture (JEPA), capturing domain knowledge without identifiers. A cross-model representation probe quantifies deviations between private representation
    
[^115]: 基于框架-策略优化的自进化编码智能体

    Self-Evolving Coding Agents via Harness-Policy Optimization

    [https://arxiv.org/abs/2610.22791](https://arxiv.org/abs/2610.22791)

    提出 Self-HPO 框架，通过交替优化智能体框架与策略模型，让进化的策略自己提出框架改进方案，从而实现编码智能体的递归自我进化。

    

    大型语言模型（LLM）智能体在编码任务中展现出可喜的潜力，其性能由策略模型和作为智能体与环境交互中介的框架共同决定。事实上，框架与策略是相互促进的，仅进化其中一个组件是次优的——为某个检查点调优的框架可能无法针对更新后策略的失败模式，而策略学习也受限于固定框架所产生的轨迹。鉴于此，我们提出 Self-HPO，一个面向编码智能体的递归自我改进框架，它交替进行框架和策略优化，并将不断进化的策略作为自身框架的提议者。具体而言，在每一轮中，当前策略首先分析自身的失败并提出新的候选框架（H步骤）；随后在选定框架下通过强化学习（RL）更新策略（P步骤），并针对训练过程中识别出的困难任务进行……

    arXiv:2610.22791v1 Announce Type: cross  Abstract: Large language model (LLM) agents have shown promising potential in coding tasks, and their performance is jointly determined by the policy model and the harness that mediates their interaction with the environment. Indeed, harness and policy are mutually reinforcing and evolving only one component is suboptimal---a harness tuned for one checkpoint may not target the failure modes of an updated policy, while policy learning remains limited to the trajectories elicited by a fixed harness. In light of this, we propose Self-HPO, a recursive self-improving framework for coding agents that alternates harness and policy optimization, using the evolving policy as its own harness proposer. Specifically, in each round, the current policy first analyzes its own failures and proposes new harness candidates (H-step). The policy is then updated through reinforcement learning (RL) under the selected harness (P-step), and hard tasks identified during
    
[^116]: 下一个词元还是向量？学习该解码多远

    Next Token or Vector? Learning How Far to Decode

    [https://arxiv.org/abs/2610.22788](https://arxiv.org/abs/2610.22788)

    提出CADENCE上下文自适应解码方法，利用高斯预测头将未来文本片段表示为连续潜在向量，并通过学习路由器根据上下文动态选择预测长度，相比固定长度预测获得更高的草稿接受率。

    

    标准的自回归解码每次只前进一个词元，即使多个即将出现的词元其实可以被一起预测。我们提出了CADENCE（上下文自适应解码，Context-Adaptive DEcoding），一种既学习草拟什么内容、又学习草拟多远距离的生成方法。高斯预测头将未来的文本片段表示为连续的潜在向量，共享解码器将选定的向量并行转换为词元预测。一个学习得到的路由器根据模型自身的隐藏状态和预测的高斯参数来选择预测长度，使生成粒度能够适应当前上下文，并以单词元生成作为回退方案。骨干模型会验证每个提议的延续内容，并保留与其贪婪预测相匹配的前缀。在Llama-3.1-8B-Instruct上针对Alpaca、GSM8K和HumanEval的实验表明，由路由器选择预测长度位置的方式，比在所有位置统一应用相应预测长度能获得更高的草稿接受率。

    arXiv:2610.22788v1 Announce Type: cross  Abstract: Standard autoregressive decoding advances one token at a time, even when several upcoming tokens can be predicted together. We propose CADENCE (Context-Adaptive DEcoding), a generation method that learns both what to draft and how far to draft. Gaussian prediction heads represent future spans as continuous latent vectors, and a shared decoder converts the selected vector into token predictions in parallel. A learned router selects the prediction length from the model's own hidden states and predicted Gaussian parameters, adapting generation granularity to the current context with single-token generation as a fallback. The backbone verifies each proposed continuation and retains the prefix that matches its greedy predictions. Experiments with Llama-3.1-8B-Instruct on Alpaca, GSM8K, and HumanEval show that router-selected positions yield higher draft acceptance than applying the corresponding prediction length everywhere. CADENCE achieve
    
[^117]: 何时可以信任智能体自身的验证器？在搜索压力与预言机访问下对反事实理由验证进行压力测试

    When Can You Trust an Agent's Own Verifier? Stress-Testing Counterfactual Rationale Verification Under Search Pressure and Oracle Access

    [https://arxiv.org/abs/2610.22779](https://arxiv.org/abs/2610.22779)

    本文提出了一种通过压力比ρ和四个可证伪探针来压力测试智能体自身验证器的方法，用于检验当智能体针对验证器进行优化时，反事实理由验证是否仍然可靠。

    

    自动化智能体设计越来越信任单一分数来决定某项改动是否是一种改进——但当分数被足够激进地优化时，它就不再能衡量其原本要衡量的东西。当智能体同时解释其设计时，该解释是一个我们可以无需人工标注即可检查的第二信号：否定每一条声明的论断，重新运行，并观察性能是否下降。我们要问的是：一旦智能体针对该验证器进行优化，这样的验证器是否仍然可靠。我们的贡献不是一个新的验证器，而是一种对验证器进行压力测试的方法：一个衡量验证器对智能体搜索引导程度的压力比 ρ，以及四个可证伪的探针，覆盖四种失败模式——不提供任何信号、过于微弱而不起作用、通过声明更少的论断来钻空子，以及在智能体提交之前就被查询。我们将该测试应用于我们自己的验证器——理由一致性分数，在一个由LLM驱动的、设计机器学习流水线的智能体中进行测试。

    arXiv:2610.22779v1 Announce Type: new  Abstract: Automated agent design increasingly trusts a single score to decide whether a change is an improvement -- but a score optimized aggressively enough ceases to measure what it was meant to. When an agent also explains its design, that explanation is a second signal we can check without human labels: negate each stated claim, re-run, and observe whether performance drops. We ask whether such a verifier stays reliable once an agent optimizes against it. Our contribution is not a new verifier but a way to stress-test one: a pressure ratio $\rho$ measuring how much the verifier steers the agent's search, and four falsifiable probes covering four failure modes -- giving no signal, being too weak to matter, being gamed by stating fewer claims, and being queried before the agent commits. We apply the test to our own verifier, the Rationale Consistency Score (RCS), inside an LLM agent that designs machine-learning pipelines, with one generator mod
    
[^118]: StoBasket：面向边际贡献的追加销售与交叉销售联合优化

    StoBasket: Joint Upsell and Cross-Sell Optimization for Contribution Margin

    [https://arxiv.org/abs/2610.22768](https://arxiv.org/abs/2610.22768)

    该论文提出了首个将追加销售与交叉销售响应耦合的“联合随机购物篮编辑”框架StoBasket，以期望增量边际贡献为优化目标，在五个公开购物篮数据集的仿真中几乎捕获了全部可获得的优惠边际。

    

    arXiv:2610.22768v1 公告类型：新论文 摘要：零售推荐系统按相关性排序，但零售商赚取的是边际贡献：一次追加销售赚取的是其所替代商品之上的边际差额，而一次交叉销售赚取的是其全部边际。我们提出了“联合随机购物篮编辑”这一新的形式化框架：由单一策略为现有购物篮选择替换型与添加型优惠，以期望增量边际贡献对其进行估值，并通过展示优惠、接受优惠和联合接受三种响应渠道将其耦合起来。据我们所知，这是首个将追加销售与交叉销售响应相耦合的形式化方法，以往的可分离处理方法是其零耦合的特例。在基于真实数据校准、覆盖五个公开购物篮数据集的仿真中，StoBasket 几乎捕获了全部可获得的优惠边际：在每个默认价格设置下误差均在0.5%以内，当真实响应机制位于其模型族之外时也在0.7%以内。正确的经济核算是其最广泛的收益……（原文摘要在此处截断）

    arXiv:2610.22768v1 Announce Type: new  Abstract: Retail recommenders rank relevance, but retailers earn contribution margin: an upsell earns the margin difference over the item it replaces, whereas a cross-sell earns its full margin. We introduce joint stochastic basket editing, a formulation in which one policy selects replacement and addition offers for an existing basket, values them by expected incremental contribution margin, and couples them through displayed-offer, accepted-offer, and joint-acceptance response channels. To our knowledge, it is the first formulation in which upsell and cross-sell responses are coupled; prior separable treatments are its zero-coupling special case. In real-data-calibrated simulations on five public basket datasets, StoBasket captures essentially all attainable offer margin: within 0.5% in every default-price setting and within 0.7% when the true response lies outside its model family. Correct economic accounting is the broadest gain: margin-weight
    
[^119]: 在剧透与臆测之间：避免编程基准测试中的过度具体化

    Between Spoilers and Guesswork: Avoiding Over-Specification in Coding Benchmarks

    [https://arxiv.org/abs/2610.22760](https://arxiv.org/abs/2610.22760)

    该论文揭示了编程基准测试中通过向任务描述泄露真实答案信息来解决任务描述不足的做法反而会导致过度具体化（剧透）问题，使解决方案绕过基准本应考察的难点，并主张在剧透与臆测之间取得平衡以延长基准的使用寿命。

    

    基准测试推动着人工智能的进步，但却受到任务具体化不足和过度具体化的困扰。任务具体化不足会导致臆测，即解决方案不得不凭空编造，或者更糟糕的是，诉诸数据污染或奖励作弊。任务过度具体化则会导致剧透，即解决方案可以绕过基准本应衡量的一些较难步骤。由于面向未来的基准测试应当具有挑战性，过度具体化会缩短基准在社区中被采用的保质期。仓库级编程基准测试帮助该领域取得了进步，但由于过度具体化和具体化不足的双重问题，它们经历了不必要的频繁更迭。例如，SWE-bench Pro、FEA-bench、Doc2Feat-bench 和 Multi-SWE-bench 都通过使用从真实答案中提取的提示来扩充任务描述，以解决具体化不足的问题。我们表明，将真实答案信息泄露到任务描述中会导致其中一些基准出现过度具体化的问题。我们进行了实验……

    arXiv:2610.22760v1 Announce Type: new  Abstract: Benchmarks drive AI progress, but are hampered by task under-specification and over-specification. Task under-specification causes guesswork, where solutions have to make things up, or worse, resort to contamination or reward hacking. Task over-specification causes spoilers, where solutions can bypass some of the harder steps the benchmark was supposed to measure. Since future-looking benchmarks should be difficult, over-specification reduces a benchmark's shelflife of community adoption. Repository-level coding benchmarks have helped the field progress but experience unnecessary churn due to both over- and under-specification. For instance, SWE-bench Pro, FEA-bench, Doc2Feat-bench, and Multi-SWE-bench all tackle under-specification by augmenting task statements with hints derived from ground truth. We show that leaking ground-truth information into the task statement causes over-specification in some of these benchmarks. We experiment w
    
[^120]: R²Orch：通过技能内化循环中的自我博弈实现智能体编排的递归自我改进

    R$^2$Orch: Recursive Self-Improvement for Agent Orchestration via Self-Play within a Skill-Internalization Loop

    [https://arxiv.org/abs/2610.22740](https://arxiv.org/abs/2610.22740)

    提出 R²Orch 递归自我博弈编排新范式，通过内循环自我博弈改进编排器、外循环改进内循环本身，并借助递归可学习性目标与技能撤回机制，解决编排学习中手工课程设计、优势坍塌和评估信号不可靠三大难题。

    

    近年来，智能体编排已被广泛应用于解决复杂的人类任务。然而，现有的编排学习仍然面临一些关键挑战，包括手工设计的课程、当任务的所有 rollout 都达到相同结果时出现的优势坍塌，以及用单一含噪评分来判断每个技能的不可靠评估信号。为了解决这些挑战，我们提出了 R²Orch，一种新的递归自我博弈编排范式：内循环通过自我博弈在自选任务上改进编排器，外循环则改进内循环的改进方式本身。为支持这一范式，我们引入了递归可学习性，这是一种训练目标，用于奖励提议者选择那些在技能被撤回时 rollout 仍能持续产生不同结果的任务。基于这一目标，我们进一步提出了技能撤回机制，其中外循环在自我博弈内的受控评估表明（摘要原文在此处截断）

    arXiv:2610.22740v1 Announce Type: new  Abstract: In recent years, agentic orchestration has been widely applied to solve complex human tasks. However, existing orchestration learning still faces key challenges, including hand-designed curricula, advantage collapse when a task's rollouts all reach the same outcome, and unreliable evaluation signals that judge each skill by one noisy score. To address these challenges, we propose R$^2$Orch, a new paradigm of Recursive Self-Play Orchestration: an inner loop of self-play that improves the orchestrator on self-chosen tasks, and an outer loop that improves how the inner loop improves. To support this paradigm, we introduce Recursive Learnability, a training objective that rewards a Proposer for choosing tasks whose rollouts keep reaching different outcomes as skills are withdrawn. Built on this objective, we further propose Skill Withdrawal, in which the outer loop withdraws each skill once a controlled evaluation within self-play shows that
    
[^121]: 面向推理大语言模型的图序列化：跨知识抽取与结构推理的研究

    Graph Serializations for Reasoning LLMs: A Study Across Knowledge Extraction and Structural Reasoning

    [https://arxiv.org/abs/2610.22734](https://arxiv.org/abs/2610.22734)

    该论文在九种前沿推理LLM上评估了15种图序列化格式，发现知识抽取对格式不敏感，而结构推理的最优格式取决于模型、查询和任务，并提出将图序列化作为一种动态编排决策来处理。

    

    知识图谱被序列化为文本并纳入大语言模型上下文的方式会显著影响下游LLM的性能，先前的研究报告称不同格式之间存在数十个百分点的性能差异。然而，现有研究仅考虑了有限的一组表示形式，且早于前沿推理模型的出现——如今的前沿推理模型具备更强的推理、指令遵循和长上下文能力。我们在基于本体的知识抽取和结构图推理任务上，评估了15种图表示格式在九种前沿LLM配置下的表现。研究表明，知识抽取对不同的序列化格式相对不敏感，但在结构推理任务中，最优格式的选择会显著影响准确率。我们证明了最优表示的选择取决于模型、查询和任务，并提出应将图序列化视为一种动态编排决策。

    arXiv:2610.22734v1 Announce Type: new  Abstract: The way a knowledge graph is serialized as text for inclusion into LLM context can substantially affect downstream LLM performance, with prior work reporting tens of percentage points variations across formats. However, existing studies consider only a limited set of representations and predate frontier reasoning models, which now offer stronger reasoning, instruction-following, and long-context capabilities. We evaluate 15 graph representation formats across nine frontier LLM configurations on ontology-based knowledge extraction and structural graph reasoning tasks. While knowledge extraction is relatively insensitive to different serialization formats, the choice of the optimal format for structural reasoning tasks significantly affects accuracy. We demonstrate that the optimal representation choice depends on model, query, and task and propose to approach graph serialization as a dynamic orchestration decision.
    
[^122]: OrderFlowDiT：用于概率性限价订单供给预测的传输正则化扩散Transformer

    OrderFlowDiT: Transport-Regularized Diffusion Transformer for Probabilistic Limit-Order Supply Forecasting

    [https://arxiv.org/abs/2610.22727](https://arxiv.org/abs/2610.22727)

    该论文提出OrderFlowDiT，一个结合非平衡最优传输辅助损失的条件扩散Transformer，能够对未归一化的价格-成交量场进行建模，联合预测未来五分钟限价订单的价格放置位置、总量和预测不确定性，并配套开发了全场评估框架。

    

    我们研究在给定当前订单簿和近期订单流的条件下，未来五分钟内提交的限价订单量的条件分布。该任务需要联合捕捉价格放置位置、总量以及预测不确定性。逐坐标的扩散损失没有明确考虑价格位移，而将场归一化为单位质量则会移除关于总量的信息。在选定的汇总统计量上的一致性也不能保证对联合场分布的忠实还原。我们提出OrderFlowDiT，这是一个针对未归一化的买卖双方价格-成交量场的条件扩散Transformer，采用非平衡最优传输（UOT）辅助损失进行训练。我们还开发了一个全场评估框架，该框架将对手的价格放置位置和总量敏感的传输成本与适当的评分规则和校准诊断相结合。在针对30只美国（股票）的1950个验证条件的主要比较中……

    arXiv:2610.22727v1 Announce Type: cross  Abstract: We study the conditional distribution of limit-order volume submitted over the next five minutes, given the current order book and recent order flow. The task requires capturing price placement, total volume, and predictive uncertainty jointly. Coordinate-wise diffusion losses do not explicitly account for price displacement, while normalizing fields to unit mass removes information about total volume. Agreement on selected summary statistics also does not establish fidelity to the joint field distribution. We introduce OrderFlowDiT, a conditional Diffusion Transformer for unnormalized bid and ask volume-at-price fields, trained with an unbalanced optimal-transport (UOT) auxiliary loss. We also develop a full-field evaluation framework that combines a transport cost sensitive to price placement and total volume with proper scoring rules and calibration diagnostics. In the primary comparison on 1950 validation conditions across 30 U.S. 
    
[^123]: LEMON：基于注意力图的等变学习，实现对新大语言模型的零样本不当行为监测

    LEMON: Equivariant Learning over Attention Maps for Zero-Shot Misbehavior Monitoring on New LLMs

    [https://arxiv.org/abs/2610.22723](https://arxiv.org/abs/2610.22723)

    提出LEMON框架，将跨模型注意力图的不匹配刻画为代数对称性，并利用等变几何深度学习，使基于内部表示的不当行为监控器能够零样本迁移到新的LLM上，无需昂贵的重新标注。

    

    大语言模型会表现出阻碍其在高风险场景中部署的不当行为。监测这类失败的一种方法是使用外部LLM作为监控器，但这种做法成本极其高昂。一类著名且高效的监控器可以从内部表示（如激活探针）中检测此类失败，但它们与其训练时所用的模型绑定，随着新LLM的出现，需要付出高昂的重新标注成本。避免这种重复标注成本的一个自然途径，是开发能够在不同LLM之间零样本迁移的基于内部表示的监控器。然而，这种迁移极具挑战性，因为内部表示在不同模型之间并非天然对齐；特别是，已知包含不当行为信号的注意力图，在注意力头的功能、排序以及模型深度上可能存在差异。在这项工作中，我们将这些跨模型注意力不匹配刻画为代数对称性，并利用几何深度学习原理（摘要在此处被截断）

    arXiv:2610.22723v1 Announce Type: new  Abstract: Large language models exhibit misbehaviors that hinder deployment in high-stakes settings. One approach to monitor such failures is to use an external LLM as a monitor, but this is prohibitively expensive. A well-known and efficient class of monitors can detect such failures from internal representations (such as activation probes), but are tied to the model they were trained on, requiring costly re-annotation as new LLMs appear. A natural way to avoid this repeated annotation cost is to develop internal-representation based monitors that transfer zero-shot across LLMs. However, such transfer is challenging because internal representations are not naturally aligned across models; in particular, attention maps, known to contain signals of misbehavior, may differ in head function, ordering, and model depth. In this work, we characterize those cross-model attention mismatches as algebraic symmetries and use geometric deep learning principle
    
[^124]: ARBOR：通过推理状态追踪定位长思维链中的推理失败

    ARBOR: Locating Reasoning Failures in Long Chains of Thought through Reasoning State Tracking

    [https://arxiv.org/abs/2610.22717](https://arxiv.org/abs/2610.22717)

    ARBOR提出了一种推理失败归因框架，通过将推理轨迹表示为推理状态树并从全局视角追踪步骤与前序状态之间的关系，从而精确定位长思维链中的推理失败，并将已知失败模式细化为机制不同的子类型。

    

    大型推理模型通过长思维链解决复杂问题，已有研究记录了诸如过度思考、思考不足和无依据推理等失败模式。现有方法从局部信号检测这些失败，例如与相邻步骤的文本相似性、切换标记，或大语言模型对某一步骤正确性的判断。然而，一个步骤对解决问题的贡献取决于先前步骤所建立的状态，因此识别失败需要一个跨越整个轨迹追踪该状态的全局视角。为此，我们提出了ARBOR——一个推理失败归因框架，它将每条推理轨迹表示为一棵推理状态树，该树随着每个步骤对状态的创建、验证、剪枝或重访而构建，并将每个失败归因于某一步骤与其前序状态之间的关系。借助ARBOR，我们将这三种失败模式细化为通过不同机制损害推理的多个子类型。

    arXiv:2610.22717v1 Announce Type: cross  Abstract: Large reasoning models solve complex problems through long chains of thought, in which prior work has documented failure modes such as overthinking, underthinking, and unsupported deductions. Existing methods detect these failures from local signals: textual similarity to nearby steps, switch tokens, or an LLM's judgment of a step's correctness. However, what a step contributes to the solution depends on the state that earlier steps have established, so identifying failures requires a global view that tracks that state across the trajectory. Therefore, we propose ARBOR, a reasoning-failure attribution framework that represents each trajectory as a tree of reasoning states, built as each step creates, verifies, prunes, or revisits a state, and attributes each failure to a relation between a step and the state preceding it. Using ARBOR, we refine the three failure modes into distinct subtypes that harm reasoning through different mechani
    
[^125]: GlobeBench：评估基于文本的世界模型在可扩展环境模拟中的能力

    GlobeBench: Evaluating Text-Based World Model Capabilities for Scalable Environment Simulations

    [https://arxiv.org/abs/2610.22708](https://arxiv.org/abs/2610.22708)

    GlobeBench 是首个用于评估基于文本的世界模型能力的标准化基准，包含跨五个领域和十九项模拟技能的 1,632 个任务，并采用确定性门控与语义裁判相结合的混合评估架构。

    

    基于文本的世界模型最近展现出为多种用例模拟复杂强化学习（RL）环境状态转移的潜力。然而，目前尚无标准化基准来比较、研究和改进这些模型。本文试图通过构建 GlobeBench 来解决这一问题，这是一个新颖的基准，包含跨五个独立领域和十九项模拟技能的 1,632 个任务。这些任务通过理解先前的状态变化、遵循环境规则以及预测动作对相关和无关对象的影响，来检验模型在给定完整智能体轨迹的情况下能否预测环境对特定动作的响应。我们将预测的环境响应与通过执行得到的参考结果进行比较，采用结合了确定性门控和语义裁判的混合评估架构。在 20 个通用大语言模型和专用文本世界模型中，GPT-6 Astra 取得了最高的任务成功率，达到 49（%）（原文摘要在此处截断）。

    arXiv:2610.22708v1 Announce Type: new  Abstract: Text-based world models have recently shown potential to simulate complex RL environment state transitions for a diverse set of use cases. However, no standardized benchmark exists to compare, study and improve these models. Through this paper, we attempt to solve this problem by building GlobeBench, a novel benchmark with 1,632 tasks across five separate domains and nineteen simulation skills. The tasks check whether models can predict the environment's response to a specific action given the full agent trajectory by understanding earlier state changes, following environment rules, and predicting the effects of actions on both related and unrelated objects. We compare the predicted environment response with an execution-derived reference, using a hybrid architecture with both deterministic gates and a semantic judge. Among 20 general-purpose LLMs and specialized text world models, GPT-6 Astra achieves the highest task success rate at 49
    
[^126]: 面向卡通电影社交与情感理解的脑对齐多模态大语言模型

    Brain-Aligned Multimodal LLMs for Social and Emotional Understanding of Cartoon Movies

    [https://arxiv.org/abs/2610.22706](https://arxiv.org/abs/2610.22706)

    该研究比较了扩散型与自回归多模态大语言模型和人类在观看卡通电影时的心理理论与情感效价评分及fMRI脑响应，发现模型评分能捕捉人类的时间动态变化且与情感效价的对齐更好，并可作为特征有效预测多个脑区的神经活动。

    

    多模态大语言模型（MLLMs）能够对社会和情感内容进行可扩展的标注，但其与人类判断在时间上的一致性以及对神经活动的预测价值仍需进一步评估。本研究将基于扩散模型和自回归架构的MLLMs与人类在三部短篇卡通电影上的连续心理理论（ToM）和情感效价评分进行了比较。行为分析评估了时间对齐的皮尔逊相关性、评分分布以及前置场景描述的影响。来自各来源的评分被分别用于被试特异性的回归模型中，以预测区域性fMRI响应。结果显示，尽管存在系统性的分布差异，模型评分仍能捕捉人类时间变化的某些方面，且对情感效价的对齐普遍强于对心理理论的对齐。前置语境在某些条件下能改善对齐效果，具体效果取决于模型和电影。在《The Present》这部影片中，更多脑区显示出对MLLM评分的显著编码（原文在此截断）。

    arXiv:2610.22706v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) offer scalable annotations of social and affective content, but their temporal agreement with human judgments and neural predictive value require further evaluation. We compared diffusion-based and autoregressive MLLMs with continuous human Theory of Mind (ToM) and affective valence ratings across three short cartoon movies. Behavioral analyses assessed temporally aligned Pearson correlations, rating distributions, and the effects of preceding scene descriptions. Ratings from each source were separately used in subject-specific regression models to predict regional fMRI responses. Model ratings captured aspects of human temporal variation despite systematic distributional differences, with generally stronger alignment for valence than ToM. Preceding context improved alignment in some conditions, depending on the model and movie. In \emph{The Present}, more regions showed significant encoding for M
    
[^127]: NoveltyScope：一个用于基于证据的学术论文新颖性评估的多智能体研究助手

    NoveltyScope: A Multi-Agent Research Assistant for Evidence-Grounded Novelty Assessment of Scientific Papers

    [https://arxiv.org/abs/2610.22703](https://arxiv.org/abs/2610.22703)

    论文提出NoveltyScope，一个通过检索先前工作、请求配对反驳证据并核对引文，由控方、辩方、法官和评论家四个智能体角色对论文作者声称的贡献进行基于证据的新颖性评估并输出五级评分的多智能体研究助手，其在RINoBench上的误差优于单LLM评分和向量相似度基线。

    

    NoveltyScope是一个原型系统，它将一篇科学论文与检索到的先前工作进行对比，并生成结构化报告和五级新颖性评分。该系统提取论文的核心任务以及最多两个作者声称的贡献，然后使用BM25与稠密检索相结合倒数排名融合的方法，在一个封闭的本地语料库中进行搜索。贡献对比时会为潜在的反驳观点请求配对的文本段落，并且所有引文都会与缓存的源文本进行核对。四个评分角色（控方、辩方、法官和评论家）基于由此产生的证据进行评分，法官的评分最多只进行一次修订。评估使用了50个类别均衡的RINoBench样本，其中想法层面的标签被保留用于全文输入。使用Gemini时，单LLM评分和多智能体评分的平均绝对误差分别为1.48和1.26；使用Qwen3时分别为1.22和1.16。向量相似度基线的误差为1.68。HiCLIP案例研究展示了贡献对比的具体效果。

    arXiv:2610.22703v1 Announce Type: cross  Abstract: NoveltyScope is a prototype that compares a scientific paper with retrieved prior work and produces a structured report and a five-level novelty score. The system extracts the core task and at most two author-claimed contributions, then searches a closed local corpus using BM25 and dense retrieval with reciprocal rank fusion. Contribution comparisons request paired passages for potential refutations, and quotations are checked against cached source text. Four scoring roles (Prosecutor, Defender, Judge, and Critic) use the resulting evidence, with at most one revision of the Judge's score. Evaluation uses 50 class-balanced RINoBench samples whose idea-level labels are retained for full-paper input. Mean absolute error is 1.48 and 1.26 for single-LLM and multi-agent scoring with Gemini, respectively, and 1.22 and 1.16 with Qwen3. The vector-similarity baseline has an error of 1.68. A HiCLIP case study illustrates the contribution compari
    
[^128]: 修复、重试与后见之明的代价：形式化定理证明中的验证器反馈

    Repair, Retry, and the Cost of Hindsight: Verifier Feedback in Formal Theorem Proving

    [https://arxiv.org/abs/2610.22698](https://arxiv.org/abs/2610.22698)

    本文通过受控单步实验调和了先前研究中的矛盾发现，表明在Lean定理证明中，展示失败的证明会损害重试效果并加剧复制行为，而附加对应的编译器错误信息则可以挽回这一损失。

    

    验证器引导的自我修复是指模型在再次尝试之前，将失败尝试的反馈作为条件输入，而无需更新模型权重。然而，先前工作中的两个发现之间存在矛盾：验证器反馈可以改进对失败的修复，但自我修复本身相比独立重采样几乎没有优势。我们通过受控的单步实验在Lean定理证明中调和了这些发现，实验在固定前一次失败的前提下，改变是否展示失败的证明、编译器错误是否与该失败相对应、以及重试指令。我们在困难的MiniF2F问题上测试了冻结的Qwen3.6-27B（通用模型）和DeepSeek-Prover-V2-7B（专用证明器）。我们发现了两种相反的效应：向模型展示其失败的前一次尝试会降低成功率并大幅增加对该尝试的复制，而在Qwen上，加入该次尝试的编译器错误可以挽回这一损失，恢复到仅使用诊断信息的水平。替换……（摘要原文在此处截断）

    arXiv:2610.22698v1 Announce Type: new  Abstract: Verifier-guided self-repair conditions a model on feedback from failed attempts before it tries again, without weight updates. Yet two findings in prior work are in tension: verifier feedback can improve repair of failures, but self-repair itself can provide little-to-no advantage over independent resampling. We reconcile these findings in Lean theorem proving through controlled one-step experiments that hold the preceding failure fixed while varying whether the failed proof is shown, whether the compiler error corresponds to that failure, and the retry instruction. We test frozen Qwen3.6-27B, a general-purpose model, and DeepSeek-Prover-V2-7B, a specialized prover, on difficult MiniF2F problems. We find two opposing effects. Showing the model its failed predecessor reduces success and sharply increases copying of it, while on Qwen adding that attempt's compiler error recovers the loss, to the level of the diagnostic alone. Replacing the
    
[^129]: 分别探索，择优采纳：面向大型工具库的执行引导式工具使用

    Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries

    [https://arxiv.org/abs/2610.22688](https://arxiv.org/abs/2610.22688)

    提出BRIA智能体，通过将候选工具调查隔离到独立分支、检查执行证据并仅将有用的结果采纳回主轨迹，解决了大规模工具库中多候选工具选择时失败尝试污染推理轨迹的问题。

    

    大型语言模型智能体在任务执行过程中出现新需求时，必须从包含数千个工具的大型工具库中发现相关工具。当多个候选工具都可能满足同一需求时，可能需要通过实际执行来识别真正适用的工具。在共享轨迹中调查候选工具可能导致部分备选方案未被探索，或使失败的尝试影响后续推理。为解决这一问题，我们提出了BRIA（BRanch-Inspect-Adopt，分支-检查-采纳），一个简单而有效的智能体，它将候选工具的调查暂时隔离在独立的分支中。每个分支检索并执行工具以调查其对应的候选工具；任务智能体检查由此产生的执行证据，并只将有用的结果采纳到其继续执行的主轨迹中。BRIA还能在任务执行过程中出现新的能力需求时动态检索更多工具。我们在PlanBench-XL和MCP-Atlas基准上，使用包含1897个工具的共享工具池、三个语言模型以及静态分析设置对BRIA进行了评估。

    arXiv:2610.22688v1 Announce Type: cross  Abstract: Large language model agents must discover relevant tools from large libraries, e.g., thousands of tools, as new needs arise during task execution. When several candidates could satisfy the same need, execution may be required to identify the relevant one. Investigating candidates in a shared trajectory can leave alternatives unexplored or allow failed attempts to influence subsequent reasoning. To address this, we present BRIA (BRanch-Inspect-Adopt), a simple yet effective agent that temporarily isolates candidate investigations in separate branches. Each branch retrieves and executes tools to investigate its candidate; a task agent inspects the resulting execution evidence and adopts only useful results into its continuing trajectory. BRIA retrieves additional tools as new capability needs arise during task execution. We evaluate BRIA on PlanBench-XL and MCP-Atlas using a shared pool of 1897 tools, three language models, and static an
    
[^130]: AlphaBet：用于大语言模型事件预测智能体的强化学习

    AlphaBet: Reinforcement Learning for LLM Event Forecasting Agents

    [https://arxiv.org/abs/2610.22684](https://arxiv.org/abs/2610.22684)

    AlphaBet是一个智能体强化学习框架，通过训练单个LLM策略在事件解决前的多个时间点自主搜索证据并报告概率，利用适当评分规则以已解决事件实现端到端监督，弥补了现有预测训练方法将证据获取排除在学习循环之外的不足。

    

    预测一个尚未解决的事件需要找到相关证据，并判断这些证据在多大程度上支持每种可能的结果。然而，基于结果训练大语言模型（LLM）预测器的现有方法通常从单一预测日期开始使用预先收集好的证据，使得证据获取过程游离于学习循环之外。我们提出了AlphaBet，这是一个智能体强化学习（RL）框架，它训练单个LLM策略在事件解决前的多个日期进行搜索、推理并报告概率。每个带日期的预测都通过适当的评分规则与最终结果对比进行评分，从而使一个已解决的事件能够监督不同信息状态下的决策。我们将预测遗憾分解为证据获取差距和概率报告差距，并证明组相对策略优化（GRPO）的标准差归一化在多分类适当评分规则下可能会扭曲概率报告目标。在GLM-4.5-Air上，AlphaBe……（原文摘要至此截断）

    arXiv:2610.22684v1 Announce Type: cross  Abstract: Forecasting an unresolved event requires finding relevant evidence and deciding how strongly it supports each outcome. Yet outcome-based training of Large Language Model (LLM) forecasters often begins with preassembled evidence at a single prediction date, leaving evidence acquisition outside the learning loop. We introduce AlphaBet, an agentic reinforcement learning (RL) framework that trains a single LLM policy to search, reason, and report probabilities across multiple pre-resolution dates. Each dated forecast is scored by a proper scoring rule against the eventual outcome, allowing one resolved event to supervise decisions under different information states. We decompose forecasting regret into an evidence-acquisition gap and a probability-reporting gap, and show that Group Relative Policy Optimization (GRPO)'s standard-deviation normalization can distort the reporting target under multi-class proper scores. On GLM-4.5-Air, AlphaBe
    
[^131]: FlexUMM：面向统一多模态模型的灵活表示学习

    FlexUMM: Flexible Representation Learning for Unified Multimodal Models

    [https://arxiv.org/abs/2610.22682](https://arxiv.org/abs/2610.22682)

    FlexUMM提出了一种灵活的统一多模态模型框架，通过动态决定何时共享计算、何时路由专门化容量以及何时停止处理，有效解决了理解与生成两种能力之间的表示冲突问题。

    

    统一多模态模型（UMMs）能够在单一骨干网络中整合理解与生成能力。然而，众所周知，这两种能力可能导致表示上的冲突，由此引出一个核心问题：表示与计算应当如何在不同功能之间进行共享？现有的统一多模态模型通常依赖于基本固定的结构，即不同的输入和功能遵循相同的深度和参数共享模式。我们认为这一假设过于严苛：更有效的统一需要共享支持跨功能通信的计算，同时对任务或输入特定的容量进行专门化。我们提出了FlexUMM，一个灵活的统一多模态模型框架，它能够动态决定何时保留共享计算、何时路由专门化容量以及何时停止进一步处理。证据表明，这种灵活的分配在两个互补的维度上提升了统一多模态模型的性能。

    arXiv:2610.22682v1 Announce Type: new  Abstract: Unified multimodal models (UMMs) can integrate understanding and generation within a single backbone. However, it is commonly known that these two capabilities can cause conflicting representations, leading to a central question: how representation and computation should be shared across functionalities? Existing UMMs often rely on a largely fixed structure, where different inputs and functionalities follow the same depth and parameter-sharing pattern. We argue that this assumption is too restrictive: more effective unification requires sharing the computation that supports cross-functional communication while specializing the capacity that becomes task- or input-specific. We introduce FlexUMM, a flexible UMM framework that dynamically decides when to preserve shared computation, when to route specialized capacity, and when to stop further processing. Evidence shows that this flexible allocation improves UMMs along two complementary axes
    
[^132]: 并非所有图都是地图：一种面向移动GUI导航的任务无关探索框架

    Not All Graphs Are Maps: A Task-Agnostic Exploration Framework for Mobile GUI Navigation

    [https://arxiv.org/abs/2610.22672](https://arxiv.org/abs/2610.22672)

    TAG-Nav是一个任务无关的图导航框架，通过将离线探索提炼为可复用的应用功能、目标页面和路线记忆，使GUI智能体无需任务特定演示即可完成目的地导航。

    

    现代移动GUI智能体能够感知可见屏幕并执行局部操作，但它们仍然缺乏持久的应用特定导航知识：功能位于何处、哪些页面包含这些功能、以及哪些屏幕在重复访问中是等价的。我们提出了TAG-Nav，一个任务无关的图导航框架，它通过可复用的目的地记忆来包装现有的GUI智能体骨干。在看到任何被评估的指令之前，离线探索被提炼成关于应用功能、目标页面和路径点路线的紧凑记录。当未来任务的目的地已被覆盖时，TAG-Nav检索该目的地并提供路线引导，而无需任务特定的演示。这将移动GUI目的地导航与完整任务完成分离开来：许多任务共享到达正确页面这一前缀，而最终操作可能是进入内容、发送数据、进行购买、删除信息或更改敏感设置。为了评估路线...

    arXiv:2610.22672v1 Announce Type: new  Abstract: Modern mobile GUI agents can perceive visible screens and execute local actions, but they still lack persistent app-specific navigation knowledge: where functions live, which pages contain them, and which screens are equivalent across repeated visits. We introduce TAG-Nav, a Task-Agnostic Graph Navigation framework that wraps existing GUI-agent backbones with reusable destination memory. Offline exploration is distilled into compact records of app functions, target pages, and waypoint routes before any evaluated instruction is seen. When a future task's destination is covered, TAG-Nav retrieves the destination and provides route guidance without task-specific demonstrations. This separates mobile GUI destination navigation from full task completion: many tasks share the prefix of reaching the right page, while final actions may enter content, send data, make purchases, delete information, or change sensitive settings. To evaluate route f
    
[^133]: 从相关记忆到充分证据：重新思考LLM智能体中的记忆利用

    From Relevant Memories to Sufficient Evidence: Rethinking Memory Utilization in LLM Agents

    [https://arxiv.org/abs/2610.22668](https://arxiv.org/abs/2610.22668)

    提出QEMem框架，将LLM智能体的记忆利用从“检索相关记忆”重新定义为“构建查询专属证据集合”，通过查询引导发现、来源具体化和充分性精炼三种机制弥合相关性-充分性差距。

    

    长期记忆使LLM智能体能够复用过往经验，然而检索到相关记忆并不一定能为回答查询提供充分的证据。关键信息可能分散在不同的记忆记录中、被抽象化所遮蔽，或与推理所需的关系相互脱节。我们识别出了这种“相关性-充分性”差距，并提出QEMem框架，将记忆利用重新表述为构建针对特定查询的证据集合，将关注点从单个检索结果的相关性转向集体证据支撑。QEMem整合了三种互补机制：查询引导的发现（query-guided discovery）跨多个记忆粒度检索证据；基于来源的具体化（source-grounded reification）将证据组织成结构化的ViewCards，包含明确的证据角色和来源引用；充分性引导的精炼（sufficiency-guided refinement）选择性地恢复缺失信息。QEMem在现有记忆库上运行而无需对其进行修改（摘要在此处截断）……

    arXiv:2610.22668v1 Announce Type: new  Abstract: Long-term memory enables LLM agents to reuse past experiences, yet retrieving relevant memories does not necessarily provide sufficient evidence for answering a query. Essential information may remain fragmented across memory records, obscured by abstraction, or disconnected from the relations required for reasoning. We identify this relevance-sufficiency gap and propose QEMem, a framework that reformulates memory utilization as the construction of query-specific evidence sets, shifting the focus from individual retrieval relevance to collective evidential support. QEMem integrates three complementary mechanisms: query-guided discovery retrieves evidence across multiple memory granularities; source-grounded reification organizes it into structured ViewCards with explicit evidential roles and source references; and sufficiency-guided refinement selectively recovers missing information. Operating on existing memory stores without modifying
    
[^134]: 多样化推理路径至关重要：符号等价增强促进强化学习探索

    Diverse Reasoning Paths Matter: Symbolic Equivalence Augmentation for Enhancing RL Exploration

    [https://arxiv.org/abs/2610.22665](https://arxiv.org/abs/2610.22665)

    提出EquivAug数据增强框架，利用符号等价变换生成多样化且可证明正确的推理路径，证明SFT数据中推理步骤的多样性是提升RLVR强化学习探索效果的关键。

    

    arXiv:2610.22665v1 公告类型：交叉 摘要：基于可验证奖励的强化学习（RLVR）已成为提升大语言模型推理能力的强大范式。然而，监督微调（SFT）数据在实现有效下游强化学习中所起的作用仍缺乏充分研究。在本文中，我们验证了一个关键假设：SFT阶段思维链（CoT）中推理步骤的多样性是决定RLVR性能的关键因素。为验证该假设，我们提出了EquivAug——一个利用符号等价变换来生成多样化且可证明正确的推理路径的数据增强框架。通过将思维链解法转换为形式化的符号表示，施加等价变换，再重构为自然语言解释，EquivAug所生成的训练数据在推理步骤上存在差异，而非仅在表面词汇上变化。在GSM8K和FOLIO数据集上、跨多个模型规模的实验表明，EquivAug持续取得一致的改进。

    arXiv:2610.22665v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for improving the reasoning capabilities of LLMs. However, the role of supervised fine-tuning (SFT) data in enabling effective downstream RL remains underexplored. In this paper, we investigate a key hypothesis: the diversity of reasoning steps in chain-of-thought (CoT) during SFT is a critical enabler of RLVR performance. To test this hypothesis, we introduce EquivAug, a data augmentation framework that uses symbolic equivalence transformations to generate diverse yet provably correct reasoning paths. By converting CoT solutions into formal symbolic representations, applying equivalence transformations, and reconstructing natural-language explanations, EquivAug produces training data that differs in reasoning steps rather than surface-level lexical variation. Experiments on GSM8K and FOLIO across model scales demonstrate that EquivAug consistently
    
[^135]: 按需生成困难实例

    Hard Instances on Demand

    [https://arxiv.org/abs/2610.22637](https://arxiv.org/abs/2610.22637)

    该论文提出将困难实例生成视为优化问题，以求解器运行时间为目标，利用基于 Nevergrad 的黑盒优化工作流自动搜索困难实例，并在可满足性、旅行商、哈密顿回路和流水车间调度四个 NP 难问题上验证了其广泛适用性。

    

    算法实验研究结论的质量取决于其运行所用的实例。对于研究充分的问题，基准实例库的整理往往需要数十年时间，而对新问题的评估则常常退而依赖随机实例——但众所周知随机实例具有误导性：它们可能更多地反映随机生成器的特性，而非所研究的算法问题本身。我们将实例生成视为一个优化问题：对实例空间进行参数化，以求解器的运行时间作为目标函数，让黑盒优化器搜索困难实例。我们开发了一个基于 Nevergrad 优化平台的、具有广泛适用性的工作流程，并在四个 NP 难问题上进行了演示：可满足性问题、旅行商问题、哈密顿回路问题和非置换流水车间调度问题。该搜索过程只需要求解器、一个实例编码器和计算预算。我们演化生成的实例超越了预算匹配的随机生成方法所能得到的困难实例。

    arXiv:2610.22637v1 Announce Type: cross  Abstract: The conclusions of an experimental study of algorithms are only as good as the instances they are run on. Benchmark libraries for well-studied problems took decades to curate, while evaluations of new problems often fall back on random instances, which are known to be misleading: they may say more about the random generator than about the algorithmic question at hand.   We treat instance generation as an optimization problem: parameterize the instance space, take the solver's running time as the objective, and let a black-box optimizer search for hard instances. We develop a widely applicable workflow built on the Nevergrad optimization platform and demonstrate it on four NP-hard problems: satisfiability, traveling salesperson, Hamiltonian cycle, and non-permutation flow-shop scheduling. The search needs only the solver, an instance encoder, and a compute budget. Our evolved instances beat the hardest instance of a budget-matched rando
    
[^136]: 通过因果鲁棒性测量大语言模型的机制不确定性

    Measuring Mechanistic Uncertainty in Large Language Models via Causal Robustness

    [https://arxiv.org/abs/2610.22636](https://arxiv.org/abs/2610.22636)

    该论文提出“机制不确定性”概念及“因果鲁棒性裕度”指标，通过测量改变模型输出所需的最小内部干预量来量化大语言模型的不确定性，发现错误或不确定的响应裕度更小，且该指标与传统基于输出的不确定性方法形成有效互补。

    

    大语言模型（LLMs）在关键领域中的应用日益增多，其容易产生幻觉的特性使得可靠且可解释的不确定性量化变得至关重要。我们提出了“机制不确定性”这一概念，通过支持模型响应的内部机制的稳定性来刻画不确定性。我们提出了“因果鲁棒性裕度”，其定义为使模型输出发生偏离所需的最小内部干预量，并使用Jacobian Lens（Jlens）概念引导向量对该指标进行操作化实现。在三个开源模型和五个问答数据集上的实验表明，错误和不确定的响应表现出更小的裕度，说明其对内部干预更为敏感，且因果鲁棒性裕度与传统的基于输出的不确定性方法形成互补。最后，在一项受控的知识注入实验中，被正确学习的事实显示出更高的裕度。

    arXiv:2610.22636v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in critical domains, where their susceptibility to hallucinations makes reliable and interpretable uncertainty quantification essential. We introduce the concept of mechanistic uncertainty, which characterizes uncertainty through the stability of the internal mechanisms supporting a model's response. We propose the causal robustness margin, defined as the minimum internal intervention required to deviate the model's output, and operationalize it using Jacobian Lens (Jlens) concept steering vectors. Across three open-source models and five question-answering datasets, we show incorrect and uncertain responses exhibit smaller margins, indicating greater sensitivity to internal interventions, and the causal robustness margin complements conventional output-based uncertainty methods. Finally, in a controlled knowledge injection experiment, correctly learned facts show increased margins aft
    
[^137]: ToMStories：解读合成叙事中的心智理论

    ToMStories: Interpreting Theory-of-Mind in Synthetic Narratives

    [https://arxiv.org/abs/2610.22632](https://arxiv.org/abs/2610.22632)

    该论文提出了ToMStories评估数据集（约6,000个合成叙事短篇，考察情绪、恐惧、动机、情感创伤四类心理状态），发现最大的语言模型在多选题上可媲美人类，但在开放式问题上仍有差距，且模型在隐式心理状态的解释与生成上存在简化现象。

    

    理解他人往往需要从其行为方式来推断他们的感受或愿望，而非依据其直白的陈述。我们探究语言模型能否做出这类推断，以及它们是如何做到的。我们推出了心智理论故事，这是一个包含约6,000个合成生成的短篇叙事小故事的评估数据集，每个故事都配有一道关于角色情绪、恐惧、动机或情感创伤的多选题或开放式问题。这些心理状态以隐式（展示）、显式（讲述）或对话与描述相互冲突的方式呈现。通过对13个开放权重模型（参数规模从1B到72B）进行评估，我们发现最大的模型在多选题上能达到人类水平，但在开放式问题上则不然。我们的分析揭示了模型在解释和生成隐式心理状态时所做的简化处理，表明合成数据可以帮助我们理解模型如何执行这类推断。

    arXiv:2610.22632v1 Announce Type: cross  Abstract: Understanding people often requires inferring what they feel or want from how they behave rather than from what they state outright. We ask whether language models can make these inferences, and how they do so. We introduce Theory-of-Mind Stories (ToMStories), an evaluation dataset of ~6,000 short synthetically generated narrative vignettes, each paired with a multiple-choice or open-ended question about a character's emotion, fear, motivation, or emotional wound. These psychological states are conveyed implicitly (showing), explicitly (telling), or with conflict between dialogue and description. Evaluating thirteen open-weight models (1B-72B parameters), we find that the largest models match humans on multiple choice but not open-ended questions. Our analyses point to simplifications models make in both interpretation and generation of implicit psychological states, indicating that synthetic data can help us understand how models perf
    
[^138]: 貌似合理的解释就足以误导多模态评审模型

    A Plausible Explanation Is Enough to Mislead Multimodal Judges

    [https://arxiv.org/abs/2610.22631](https://arxiv.org/abs/2610.22631)

    提出JUPE基准并揭示“合理性俘获”现象——仅仅貌似合理的普通解释就足以误导多模态大模型评审接受错误答案，其影响与刻意说服性论证相当。

    

    多模态大语言模型越来越多地在训练和评估流程中充当“评审”角色，通常会同时评估答案及其解释。尽管AI评审对文本因素较为敏感，但普通貌似合理的解释是否能够左右那些需要视觉验证的判断，目前仍不清楚。我们提出了JUPE，一个视觉问答（VQA）风格的基准，它在固定答案的同时改变解释。在14个多模态大模型评审上，听起来合理的普通解释即可误导评审，其平均效果与刻意构造的说服性论证相当。简单的理由说明和思维链式论证分别使错误答案被拒绝的概率降低了12.7和13.8个百分点。我们将这种失败现象称为“合理性俘获”。借助信号检测理论，我们发现评审的判断标准发生了显著的均值偏移，趋向于接受答案，而平均辨别能力并无明显变化。

    arXiv:2610.22631v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) increasingly serve as judges in training and evaluation pipelines, often assessing answers together with explanations. Although AI judges are sensitive to textual factors, it remains unclear whether ordinary plausible explanations can sway judgments requiring visual verification. We introduce JUPE, a visual-question-answering (VQA)-style benchmark that fixes answers while varying explanations.   Across 14 MLLM judges, ordinary explanations that sound plausible mislead judges, with effects comparable on average to deliberately persuasive arguments. Plain justifications and chain-of-thought rationales reduce rejection of incorrect answers by 12.7 and 13.8 percentage points, respectively. We call this failure plausibility capture. Using signal detection theory, we find a significant mean shift in decision criterion toward acceptance, without a clear change in mean discriminability. Explanations swa
    
[^139]: 基于探针监督图卷积循环网络利用稀疏固定传感器实现全网高速公路速度的即时估计与预测

    Network-Wide Freeway Speed Nowcasting and Forecasting from Sparse Fixed Sensors with a Probe-Supervised Graph Convolutional Recurrent Network

    [https://arxiv.org/abs/2610.22630](https://arxiv.org/abs/2610.22630)

    该论文提出GCRN+模型，一种探针监督的图卷积循环网络，仅利用15.6%路段的固定传感器数据即可对达拉斯—沃斯堡地区全部4,127条有向高速公路路段实现速度的即时估计与未来一小时预测，且随网络规模线性扩展并可在CPU上训练。

    

    交通管理部门需要其路网上每条路段的当前及近期未来速度，但固定传感器仅覆盖了部分路网。我们研究了仅利用643条路段（15.6%）上的传感器，对达拉斯—沃斯堡地区全部4,127条有向高速公路路段的逐小时速度进行联合即时估计（nowcasting）与下一小时预测的问题。在训练阶段，历史探针车辆数据对所有路段进行监督；而在推理阶段仅使用传感器读数。我们将这一设置称为“探针监督稀疏感知”。我们提出了GCRN+，一种预测速度相对历史速度曲线偏差的图卷积循环网络。该方法将无参数、方向感知的传感器异常核插值（同时用作输入特征和输出基线），与有向路网图上的扩散卷积GRU、低秩自适应图以及可学习的路段嵌入相结合。该模型随网络规模线性扩展，并可在CPU上完成训练。传感器读数由探针速度模拟生成

    arXiv:2610.22630v1 Announce Type: cross  Abstract: Traffic agencies need current and near-future speeds on every link of their network, but their fixed sensors cover only part of it. We study joint nowcasting and next-hour forecasting of hourly speeds on all 4{,}127 directional freeway links of the Dallas--Fort Worth region from sensors on 643 links (15.6%). Historical probe-vehicle data supervise every link in training; only sensor readings are used at inference. We call this setting probe-supervised sparse sensing. We propose GCRN+, a graph convolutional recurrent network that predicts deviations from a historical speed profile. It combines a parameter-free, direction-aware kernel interpolation of sensor anomalies, used as input features and as an output baseline, with a diffusion-convolutional GRU on the directed road graph, a low-rank adaptive graph and learnable link embeddings. It scales linearly with network size and trains on a CPU. Sensor readings are emulated from probe speed
    
[^140]: StratFL：利用策略性联邦学习对抗隐蔽模型投毒攻击

    StratFL: Countering Covert Model Poisoning Attacks with Strategic Federated Learning

    [https://arxiv.org/abs/2610.22626](https://arxiv.org/abs/2610.22626)

    本文提出StratFL，将联邦学习服务器与隐蔽模型投毒攻击者的交互建模为不完全信息重复博弈，通过新型对抗搜索算法与信念更新算法有效防御间歇性隐蔽模型投毒攻击。

    

    高级持续性威胁（APT）是医疗和金融等关键领域联邦学习（FL）部署中的重大安全隐忧。由于APT具有间歇性的攻击行为，传统防御机制难以对其进行遏制。本文针对一种称为隐蔽模型投毒（CMP）的特定APT类型展开研究，该攻击中攻击者以隐蔽方式间歇性地注入被投毒的模型更新。本文将联邦学习服务器与CMP攻击者之间的交互建模为一种不完全信息重复博弈，即博弈双方均无法观察到对方的行动。论文设计了一种名为StratFL的新型对抗搜索算法，并在服务器端和攻击者端分别设计了新型信念更新算法。实验采用MNIST和CIFAR-10数据集设计，以评估和比较StratFL与最先进方法的性能，并在多种安全场景中验证了信念更新算法的经验收敛性。

    arXiv:2610.22626v1 Announce Type: cross  Abstract: Advanced persistent threats (APTs) are a major concern in federated learning (FL) deployments in critical sectors such as healthcare, and finance. Traditional defense mechanisms fail to contain APTs due to their intermittent attacking behavior. This paper addresses a specific type of APT called covert model poisoning (CMP) where the attacker intermittently injects poisoned model updates in a covert manner. The interaction between the FL server and a CMP attacker is modeled as an imperfect repeated game where both players cannot observe each other's actions. A novel adversarial search algorithm called StratFL is designed alongside novel belief update algorithms at both the server and adversary. Experiments are designed using MNIST and CIFAR-10 datasets to evaluate and compare StratFL's performance against state-of-the-art methods. Empirical convergence of belief update algorithms is demonstrated across various security vignettes. At the
    
[^141]: 三个激光雷达，二十五个数据集：树种识别中的跨源泛化

    Three LiDARs, Twenty-Five Datasets: Cross-Source Generalization in Tree-Species Recognition

    [https://arxiv.org/abs/2610.22606](https://arxiv.org/abs/2610.22606)

    本文提出了在FOR-species20K树种分类基准上排名第一的多分辨率多视角ConvNeXtV2模型（准确率85.0%），并揭示了官方随机划分存在数据源泄漏问题——物种均衡的隐藏测试掩盖了长尾分布和跨数据源泛化的真实挑战。

    

    FOR-species20K是目前最大的公开LiDAR树种分类基准，汇集了25个独立采集的地面、移动和机载激光扫描数据集，涵盖33个树种。我们提出了一种多分辨率、多视角的ConvNeXtV2模型，将官方隐藏测试集的准确率从79.5%提升至85.0%，在提交时排名第一。随后我们深入考察了这一分数究竟衡量了什么。由于隐藏测试集在物种层面是均衡的，其总体准确率更接近于宏平均指标，而非公开数据集长尾分布下的准确率。此外，官方的随机数据划分将每个源数据集同时纳入训练集和评估集。逻辑回归探针能够以69%的准确率从折外模型概率中恢复出样本的数据源身份，而仅凭几何特征就能在两个广泛分布的共享物种中以超过90%的准确率识别数据来源；33个物种中有18个仅出现在单一数据源中。在源-物种-单元留出评估下，评估……

    arXiv:2610.22606v1 Announce Type: cross  Abstract: FOR-species20K is the largest public LiDAR benchmark for tree-species classification, pooling 25 independently collected terrestrial, mobile, and airborne laser-scanning datasets across 33 species. We introduce a multi-resolution, multi-view ConvNeXtV2 model that improves official hidden-test accuracy from 79.5% to 85.0%, ranking first at the time of submission. We then examine what this score measures. Because the hidden test is species-balanced, its overall accuracy is more comparable to a macro metric than to accuracy under the public set's long-tailed distribution. Moreover, the official random split places every source dataset in both training and evaluation. A logistic probe recovers source identity from out-of-fold model probabilities with 69% accuracy, while geometric features identify the source above 90% within two widely shared species; 18 of the 33 species occur in only one source. Under a source-species-cell holdout, evalu
    
[^142]: 在可计算性的极限处：迈向忙碌海狸前沿上自主AI发现的步伐

    At the Limits of Computability: Steps Towards Autonomous AI Discovery on the Busy Beaver Frontier

    [https://arxiv.org/abs/2610.22603](https://arxiv.org/abs/2610.22603)

    本文提出一个基于Gemini的智能体流水线，在不可计算的忙碌海狸前沿上自主发现图灵机加速规则，标志着AI向自主数学发现迈出了重要一步。

    

    近来数学AI领域的突破——从获得奥林匹克竞赛奖牌到解决长期悬而未决的猜想——迅速拓展了自动化系统所能达到的成就。然而，这些成功主要局限于预先定义的、由人类构建的问题，AI系统如何自主地揭示数学结构并提出新问题仍是未解难题。为了超越静态挑战，我们探索忙碌海狸（Busy Beaver, BB）领域，一个本质上开放、不可计算的前沿。由于没有任何算法能够对所有n计算出忙碌海狸数BB(n)，该领域为通过数学猜想与证明进行持续探索提供了天然场景。在本文中，我们提出了一个基于Gemini的智能体流水线，能够自动发现图灵机的加速方法，并将其表达为概括大量执行步骤的简洁数学规则。我们的系统集成了模式识别、自动化……（原文摘要在此处中断）

    arXiv:2610.22603v1 Announce Type: new  Abstract: Recent breakthroughs in AI for mathematics---ranging from Olympiad medal performances to resolving long-standing open conjectures---have rapidly expanded what automated systems can achieve. Yet these successes primarily concern predefined, human-formulated problems, leaving open how AI systems can autonomously uncover mathematical structure and formulate new questions. To push beyond static challenges, we explore the Busy Beaver (BB) domain, an uncomputable frontier that is fundamentally open-ended. Because no algorithm can compute the Busy Beaver number $\mathrm{BB}(n)$ for every $n$, this domain provides a natural setting for sustained exploration through mathematical conjecture and proof. In this paper, we introduce a Gemini-based agentic pipeline that automatically discovers accelerations of Turing machines, expressed as compact mathematical rules summarizing many execution steps. Our system integrates pattern recognition, automated 
    
[^143]: 大语言模型能预测机器学习实验的结果吗？

    Can LLMs predict the outcomes of machine learning experiments?

    [https://arxiv.org/abs/2610.22599](https://arxiv.org/abs/2610.22599)

    该论文构建了包含288个LLM自生成实验想法的数据集，通过让五个前沿大语言模型在机器学习实验训练前预测其影响力并与实际结果对比，衡量了这些模型判断实验是否值得运行的直觉准确性。

    

    人工智能的最新进展使得递归自我改进（RSI）越来越具有可行性。大语言模型（LLM）已经能够提出、选择并运行机器学习（ML）实验。然而，机器学习中可能的设计选择空间非常庞大。提出实验的LLM需要具备可靠的直觉，以便在长训练时间的限制下判断某个实验是否值得运行。在这项工作中，我们通过评估当前前沿LLM在实验投入训练之前预测其自生成的机器学习实验想法影响力的准确程度，来衡量这种直觉。为此，我们构建了一个包含288个LLM提出的实验想法的数据集，涵盖八个MLE-bench任务。随后，我们提示五个前沿LLM（GPT-6 Astra、Claude Opus 5.5、Claude Fable 5.1、Kimi K3和GLM-5.3）预测每个想法的影响力，并将其预测与训练后的实际结果进行比较。我们报告了四个主要发现。首先，LLM的预测相当……

    arXiv:2610.22599v1 Announce Type: cross  Abstract: Recent advances in AI make Recursive Self-Improvement (RSI) increasingly plausible. Large Language Models (LLMs) can already propose, select, and run Machine Learning (ML) experiments. However, the space of possible design choices in ML is vast. An LLM that proposes an experiment needs a reliable intuition of whether it is worth running given the timing constraints of long training runs. In this work, we measure such intuition in current frontier LLMs by evaluating how accurately they predict the impact of self-generated ML experiment ideas before these are put into training runs. To this end, we construct a dataset of 288 LLM-proposed ideas on eight MLE-bench tasks. We then prompt five frontier LLMs (GPT-6 Astra, Claude Opus 5.5, Claude Fable 5.1, Kimi K3, and GLM-5.3) to predict the impact of each idea and compare their predictions with the actual outcomes after training. We report four main findings. First, LLM predictions are subst
    
[^144]: 通过理由正则化缓解解释与人类对齐中的人口群体差异

    Mitigating Demographic Disparities in Explanation-Human Alignment via Rationale Regularization

    [https://arxiv.org/abs/2610.22586](https://arxiv.org/abs/2610.22586)

    本文提出Plausibility-EFG指标来衡量仇恨言论检测模型在不同人口群体间解释合理性的差异，并证明采用理由正则化的对抗训练模型ADV-RoBERTa能在提升分类性能和解释与人类对齐的同时减少群体间的解释差异。

    

    我们研究仇恨言论检测中的解释公平性。高分类性能并不总是意味着模型能够为不同人口群体产生同样合理的解释。使用HateXplain数据集，我们引入了Plausibility-EFG这一指标，通过比较模型生成的词元归因与人工标注的理由，来衡量不同人口群体之间解释合理性的最大差异。我们使用集成梯度和SHAP方法评估了BERT、RoBERTa以及对抗训练模型ADV-RoBERTa。结果表明，ADV-RoBERTa在提高分类性能和解释与人类理由之间对齐程度的同时，减少了不同人口群体之间的解释差异。这些改进在两种归因方法下均可观察到，并且在训练集、验证集和测试集上保持一致。我们还表明，仅减少预测层面的偏差并不能消除解释层面的差异。

    arXiv:2610.22586v1 Announce Type: cross  Abstract: We study explanation fairness in hate speech detection. High classification performance does not always mean that a model produces equally plausible explanations for different demographic groups. Using the HateXplain dataset, we introduce Plausibility-EFG, a metric that measures the largest difference in explanation plausibility across demographic groups by comparing model-generated token attributions with human-annotated rationales. We evaluate BERT, RoBERTa, and ADV-RoBERTa, an adversarially trained model, using Integrated Gradients and SHAP. Our results show that ADV-RoBERTa improves classification performance and alignment between explanations and human rationales while reducing explanation disparities across demographic groups. These improvements are observed with both attribution methods and remain consistent across training, validation, and test splits. We also show that reducing prediction-level bias alone does not eliminate di
    
[^145]: 基于确定性诊断的大语言模型生成规约修复

    Repairing LLM-Generated Specifications with Deterministic Diagnosis

    [https://arxiv.org/abs/2610.22568](https://arxiv.org/abs/2610.22568)

    提出DRAGO修复框架，用确定性的、针对特定形式化方法的诊断取代LLM无根据的批评，且仅在诊断错误权重严格减少时才保留重写，从而有效修复LLM生成的可运行但语义错误的形式化规约。

    

    大型语言模型（LLM）正越来越多地被用作形式化工具：给定一个以自然语言表达的推理问题，它们生成一个形式化规约，符号求解器随后可以基于该规约进行求解，并获得相对于该规约的保证。许多规约虽然可以运行，但在语义上是错误的——它们编码了错误的约束、定义域、目标、谓词或最终目标，而且一旦执行成功，求解器的错误反馈就不再提供信息。纯粹基于LLM的诊断由于缺乏形式化的真值标准，往往无法发现这些错误，并且在我们的实验中，其无根据的批评导致11.8%最初正确的规约出现退化。我们提出了DRAGO，一个修复框架，它用确定性的、针对特定形式化方法的诊断取代了这种无根据的批评，并且只有当诊断出的错误权重在无参考接受谓词下严格减少时，才保留重写结果。在涵盖逻辑推理（SMT-LIB...

    arXiv:2610.22568v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used as formalizers: given a reasoning problem expressed in natural language, they produce a formal specification that a symbolic solver can then solve, with guarantees relative to that specification. Many specifications are runnable yet semantically wrong, encoding incorrect constraints, domains, objectives, predicates, or goals, and solver-error feedback dries up once execution succeeds. Purely LLM-based diagnosis, lacking a formal ground truth, often fails to spot these errors, and its ungrounded critiques cause regressions in 11.8% of initially correct specifications in our experiments. We propose DRAGO, a repair framework that replaces this ungrounded critique with deterministic, formalism-specific diagnosis, and retains a rewrite only when the diagnosed error weight strictly decreases under a reference-free acceptance predicate. Across five benchmarks spanning logical reasoning (SMT-LIB
    
[^146]: FRACTAL：一种具有形式化通信、收敛与深度界的自相似多智能体编排架构

    FRACTAL: A Self-Similar Multi-Agent Orchestration Architecture with Formal Communication, Convergence, and Depth Bounds

    [https://arxiv.org/abs/2610.22561](https://arxiv.org/abs/2610.22561)

    该论文提出FRACTAL自相似多智能体架构，以统一的五组件模式递归分解目标，并形式化证明了线性通信成本界、双模式编排准则、超多数收敛概率界与目标深度上界，辅以贝叶斯层间升级机制。

    

    我们提出了FRACTAL，一种用于递归目标分解的自相似多智能体架构，其中相同的五组件组织模式——编排器、集成体、知识账本、决策者与元观察者——在分解树的每一层递归出现。我们做出了四项形式化贡献：(1) 一个通信上界，证明在两阶段盲提案/压力测试协议下，节点内成本关于集成规模与轮次数为线性，而自由形式辩论的成本为二次方，并对固定的评估者数量作出了明确假设；(2) 一个双模式编排准则，通过可测量的多样性饱和阈值将生成阶段与评估阶段分离；(3) 一个超多数收敛保证，在给定共识水平下具有明确的概率界，以及在可操作化的目标复杂度度量下的形式化深度上界；(4) 一个贝叶斯层间升级机制，统一了迟滞……（原文摘要在此处截断）

    arXiv:2610.22561v1 Announce Type: new  Abstract: We present FRACTAL, a self-similar multi-agent architecture for recursive goal decomposition, in which the same five-component organizational pattern -- Orchestrator, Ensemble, Knowledge Ledger, Decision Maker, Meta-Observer -- recurs at every level of a decomposition tree. We make four formal contributions: (1) a communication bound showing intra-node cost is linear in ensemble size and round count (O(kr)) under a two-phase blind-proposal/stress-test protocol, versus quadratic cost for free-form debate, with an explicit, stated assumption on fixed evaluator count; (2) a two-mode orchestration criterion separating generative and evaluative phases via a measurable diversity-saturation threshold; (3) a supermajority convergence guarantee with explicit probability bounds at stated agreement levels, and a formal depth bound under an operationalized goal-complexity measure; (4) a Bayesian inter-layer escalation mechanism unifying hysteresis a
    
[^147]: 大型推理模型是优秀的日常推理者吗？与非推理模型在逻辑推理任务上的对比

    Are Large Reasoning Models Good Everyday Reasoners? A Comparison with Non-Reasoning Models on Logical Reasoning Tasks

    [https://arxiv.org/abs/2610.22538](https://arxiv.org/abs/2610.22538)

    该研究构建了一个基于语言学与语用推理现象的日常逻辑推理基准，发现大型推理模型在偏离形式逻辑规则的日常推理中表现退化，而某些采用思维链提示的非推理模型在日常推理上甚至优于推理模型。

    

    利用测试时计算的大型推理模型（LRMs）在数学和科学基准测试中取得了出色的表现，并越来越多地被部署于实际应用中。然而，它们进行日常逻辑推理的能力——这类推理往往偏离教科书式的形式逻辑——仍然未被充分探索。在本工作中，我们通过引入一个基于语言学和语用推理现象的基准，系统地评估了推理模型与日常推理模式的契合程度。我们还在受控的提示条件下对推理模型与非推理模型进行了比较。我们发现，在许多情况下：(1) 当日常推理偏离形式逻辑规则时，推理模型在日常推理中的表现会出现下降；(2) 一些经过思维链提示的非推理模型在日常推理中的表现优于同类的推理模型；(3) 通过提供额外指令的提示……

    arXiv:2610.22538v1 Announce Type: new  Abstract: Large reasoning models (LRMs) that leverage test-time computation achieve strong performance on mathematical and scientific benchmarks and are increasingly deployed in real-world applications. However, their ability to perform \textit{everyday logical reasoning}, which often departs from textbook formal logic, remains underexplored. In this work, we systematically evaluate how well reasoning models align with everyday reasoning patterns by introducing a benchmark based on linguistic and pragmatic reasoning phenomena. We also compare reasoning models with non-reasoning models under controlled prompting conditions. We find that, in many cases, (1) reasoning models show degraded performance in everyday reasoning when such reasoning deviates from formal logical rules, (2) some non-reasoning models with chain-of-thought prompting outperform comparable reasoning models in everyday reasoning, and (3) prompting with additional instructions provi
    
[^148]: AuditKIT：一个用于大语言模型与智能体系统基准测试、评判与比较的统一评估主干

    AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems

    [https://arxiv.org/abs/2610.22511](https://arxiv.org/abs/2610.22511)

    AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。

    

    评估语言模型和智能体需要综合基准测试准确率、基于评判模型的评估、运行时性能、工具使用行为以及检索质量等多方面指标。接口、指标语义和轨迹格式的差异使这些比较变得复杂，并可能掩盖智能体工作流中的失败。我们提出了AuditKIT，一个源码可用的评估库，它将这些测量统一记录在单一的结果表示中，具备方向感知的指标契约、配置指纹以及可扩展的指标和后端。针对智能体，它提供回合感知的工具调用匹配，区分并行与顺序执行，并将经过验证的任务结果与轨迹级诊断区分开来。缺失或不可观察的证据会被标记为“未评分”，而不是被计为失败。针对检索增强生成（RAG），它结合了排序检索指标、评判式的依据性评估以及确定性的证据检查。七个案例研究涵盖模型（比较）……

    arXiv:2610.22511v1 Announce Type: cross  Abstract: Evaluating language models and agents requires combining benchmark accuracy, judge-based assessment, runtime performance, tool-use behavior, and retrieval quality. Differences in interfaces, metric semantics, and trace formats complicate these comparisons and can hide failures in agentic workflows. We present AuditKIT, a source-available evaluation library that records these measurements in one result representation, with direction-aware metric contracts, configuration fingerprints, and extensible metrics and backends. For agents, it provides turn-aware tool-call matching, separates parallel from sequential execution, and distinguishes verified task outcomes from trajectory-level diagnostics. Missing or unobservable evidence is marked unscored rather than counted as failure. For retrieval-augmented generation, it combines ranked retrieval metrics, judged grounding, and deterministic evidence checks. Seven case studies spanning model co
    
[^149]: 脉冲神经网络中因果片段数量的上界

    Upper bounds on the number of causal pieces in spiking neural networks

    [https://arxiv.org/abs/2610.22484](https://arxiv.org/abs/2610.22484)

    本文首次为前馈单脉冲神经网络中因果片段的数量推导出两个互补的上界——一个基于多重伯努利数、适用于广泛脉冲神经元模型的组合界，以及一个基于超平面排列的界，从而填补了脉冲神经网络因果片段数量极限研究的空白。

    

    单脉冲神经网络的输入空间可以分解为因果片段，并且对于某些脉冲神经网络，因果片段的数量已被证明可以提供逼近误差的下界。虽然这种分解适用于广泛的脉冲神经元模型，但脉冲神经网络所能实现的因果片段数量的极限仍然未知。在这项工作中，我们推导了前馈单脉冲神经网络中因果片段数量的两个互补上界。首先，我们建立了一个由多重伯努利数给出的组合界，该界适用于一大类脉冲神经元模型，其阶为 $e^{2(L-1)N\log N+\mathcal{O}(N)}$，其中 $L$ 为网络深度，$N$ 为最大层宽。其次，对于一组特定的神经元模型，我们推导了一个基于超平面排列的界。应用后一种框架，我们得到了阶为 $\mathcal{O}\!\left(N^{(4L...}\right)$ 的界（摘要在此处被截断）。

    arXiv:2610.22484v1 Announce Type: cross  Abstract: The input space of single-spike neural networks decomposes into causal pieces, and the number of causal pieces has been shown to provide a lower bound on the approximation error for some spiking neural networks. While this decomposition is applicable to a wide range of spiking neuron models, the limits of how many causal pieces can be realised by spiking neural networks are still unknown. In this work, we derive two complementary upper bounds on the number of causal pieces in feedforward single-spike neural networks. First, we establish a combinatorial bound given by the poly-Bernoulli numbers, which applies to a broad class of spiking neuron models. The bound is of order $e^{2(L-1)N\log N+\mathcal{O}(N)}$, with network depth $L$ and maximum layer width $N$. Second, for a specific set of neuron models, we derive a bound based on hyperplane arrangements. Applying the latter framework, we obtain bounds of order $\mathcal{O}\!\left(N^{(4L
    
[^150]: 对照形式语义检验基于大语言模型智能体的规划主张

    Testing the Planning Claims of LLM-Based Agents against Formal Semantics

    [https://arxiv.org/abs/2610.22469](https://arxiv.org/abs/2610.22469)

    本文提出用隐藏的PDDL形式语义模型作为严格验证器来检验LLM智能体的规划能力，发现即便在四个模型、六种提示策略和三种智能体架构的最佳组合下，也只能解决78%的可解任务，揭示了LLM规划能力主张与形式化验证之间的差距。

    

    基于大语言模型（LLM）的智能体越来越被认为具备实践推理能力：规划、监控并纠正自身行为。其成功被归因于围绕模型构建的提示策略和智能体架构，例如思维链、基于搜索的提示、自我改进、反思、批评者机制以及工具使用循环。然而，这些主张通常由端到端成功率（仅显示任务是否完成）或LLM评判者（并非真正的验证器）来支撑。本文对照形式语义来检验LLM的规划能力。任务以自然语言提出，但每个任务都建立在一个隐藏的PDDL模型之上，该模型决定每个动作是否可应用、目标是否达成、以及计划是否存在。在五个领域（其中两个为新领域，均随论文发布）、四个模型、六种提示策略和三种智能体架构的实验中，最佳配置解决了78%的可解任务（摘要在此处截断）。

    arXiv:2610.22469v1 Announce Type: new  Abstract: LLM-based agents are increasingly credited with the competences of practical reasoning: planning, monitoring, and correcting their own actions. Their success is attributed to the prompting strategies and agent architectures built around the model, such as chain-of-thought, search-based prompting, self-refinement, reflection, critics and tool-using loops. Yet these claims are usually supported by end-to-end success rates, which show only whether a task was completed, or by LLM judges, which are not verifiers. We test LLM planning against formal semantics. Tasks are posed in natural language, but each is grounded in a hidden PDDL model that decides whether every action is applicable, whether the goal is reached, and whether a plan exists at all. Across five domains (two of them new, all released with the paper), four models, six prompting strategies and three agent architectures, the best configuration solves 78% of the solvable tasks offl
    
[^151]: 良性递归自蒸馏可以引发渐进式失准

    Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation

    [https://arxiv.org/abs/2610.22467](https://arxiv.org/abs/2610.22467)

    在递归自蒸馏过程中，即使只使用良性提示和模型自身的安全生成内容，安全模型也会随着迭代轮次不断累积失准，最终在多种有害查询上表现出渐进式失准现象。

    

    arXiv:2610.22467v1 公告类型：cross 摘要：在模型自身输出上训练模型作为提升AI能力的可能途径，已受到广泛关注。一种突出的方法是自蒸馏，即训练模型去匹配它在获得额外上下文（例如提示、反馈或指令）时本应产生的token级预测。然而，这种训练过程的安全性尚未被充分理解。在本文中，我们揭示了递归自蒸馏（即每个自蒸馏出的模型成为下一轮训练的教师模型）中一个令人惊讶的安全失效现象。我们观察到，当一个安全模型在自身（安全的）生成结果上进行递归自蒸馏，同时接收额外的非有害提示时，它在多种多样的有害查询上会变得越来越失准；我们将这一现象称为“渐进式失准”。我们在Qwen3和Phi-4模型家族上，通过SORRY-Bench和StrongREJECT基准展示了这种失效模式……

    arXiv:2610.22467v1 Announce Type: cross  Abstract: Training models on their own outputs has attracted widespread attention as a possible route to advance AI capabilities. One prominent approach is self-distillation, where a model is trained to match the token-level predictions it would have produced when given access to additional context, such as a cue, feedback, or instruction. However, the safety properties of this training procedure are poorly understood. In this paper, we present a surprising safety failure of recursive self-distillation, where each self-distilled model becomes the teacher for the next round of training. We observe that when a safe model is recursively self-distilled on its own (safe) generations while receiving an additional non-harmful cue, it becomes increasingly misaligned across a diverse set of harmful queries; we call this Progressive Misalignment. We show this failure mode in both the Qwen3 and Phi-4 model families on SORRY-Bench and StrongREJECT. Progress
    
[^152]: Harness-Delta 归因：剖析 Harness 演化带来的增益

    Harness-Delta Attribution: Dissecting Gains From Harness Evolution

    [https://arxiv.org/abs/2610.22466](https://arxiv.org/abs/2610.22466)

    该论文提出 Harness-Delta 归因（HDA）评估协议，将 harness 演化带来的性能提升分解为基准过拟合、测试时扩展计算与可泛化改进三部分，从而量化这些提升中有多少能够真正泛化到搜索集之外。

    

    arXiv:2610.22466v1 公告类型：新论文。Harness 演化能够自动优化围绕固定权重模型的提示词、技能、工具与编排代码，为在不更新模型权重的情况下改进智能体系统提供了一种途径。智能体 harness 会被迭代优化，以在搜索集上取得更高的性能。但这些增益中有多少反映了针对特定基准的过拟合或额外的推理时计算量，又有多少能够泛化到搜索集之外？在本工作中，我们提出了 Harness-Delta 归因（HDA），这是一种评估协议，可将演化增益分解为三部分：由产物驱动的过拟合、可由测试时扩展解释的增益，以及剩余部分——该剩余部分给出了残余可泛化改进的上界。我们将 HDA 应用于 11 个执行器模型和三个 harness 演化框架，涵盖数学、编程、创造力和智能体任务。在这 21 个在搜索集上取得统计显著增益的设置中，我们发现其中很大一部分……（摘要原文在此处截断）

    arXiv:2610.22466v1 Announce Type: new  Abstract: Harness evolution automatically optimizes the prompts, skills, tools, and orchestration code around a fixed-weight model, offering a way to improve agentic systems without updating model weights. The agent harness is iteratively optimized to achieve higher performance on the search set. But how much of this gain reflects benchmark-specific overfitting or additional inference-time compute, and how much may generalize beyond the search set? In this work, we introduce Harness-Delta Attribution (HDA), an evaluation protocol that dissects an evolution gain into artifact-driven overfitting, gains explained by test-time scaling, and the remainder, which gives an upper bound on residual generalizable improvement. We apply HDA to 11 executor models and three harness-evolution frameworks across math, coding, creativity, and agentic tasks. In 21 of these settings that show statistically significant search-set gains, we find that a large share of th
    
[^153]: 机器学习会议中作者在环的智能体可复现性审计

    Author-in-the-loop Agentic Reproducibility Auditing for a Machine Learning Conference

    [https://arxiv.org/abs/2610.22465](https://arxiv.org/abs/2610.22465)

    该论文提出了一种作者在环的智能体可复现性审计方法，通过结构化清单和GitHub两阶段工作流将AI智能体检查与作者澄清、修改及重新评估相结合，并在36篇会议论文上部署验证了其有效性。

    

    同行评审期间的可复现性检查需要对作者代码预期复现的内容形成共识。我们提出了一种作者在环的协作审计方法，将作者声明作为AI智能体调查与评估的显式输入。结构化清单记录复现说明及其局限性；GitHub上的两阶段工作流将智能体检查与作者的澄清、修改以及智能体的重新评估相连接。该智能体分别评估结果是否已被复现，以及作者是否需要提供澄清或进行修复，而人类评审员则批准请求并判断相关疑虑是否已得到解决。我们将该系统部署于36篇会议论文的133个实验表格与图表，在人工评审后提出了20个问题。进一步的检查支持了所有问题的关闭，尽管在某些情况下完整复现仍未得到验证。我们还进一步比较了重复审计与

    arXiv:2610.22465v1 Announce Type: new  Abstract: Reproducibility checks during peer review require a shared understanding of what authors' code is expected to reproduce. We present an author-in-the-loop collaborative audit that makes author declarations explicit inputs to an AI agent's investigation and assessment. A structured checklist records reproduction instructions and limitations; a two-stage workflow on GitHub connects agent checks to author clarification, revision, and agent reassessment. The agent separately assesses whether results have been reproduced and whether authors need to provide clarification or make repairs, while human reviewers approve requests and decide whether concerns are resolved. We deployed the system across 36 conference papers and 133 experimental tables and figures, raising 20 issues after human review. Further checks supported closing all the issues, although full reproduction remained unverified in some cases. We further compared repeated audits with 
    
[^154]: 2024年摩尔多瓦总统选举中的政治叙事：立场、放大与影响

    Political Narratives in the 2024 Moldovan Presidential Election: Stance, Amplification, and Influence

    [https://arxiv.org/abs/2610.22464](https://arxiv.org/abs/2610.22464)

    该研究结合多语言立场检测、协同行为分析和因果推断，揭示了2024年摩尔多瓦总统选举期间社交媒体上政治叙事的跨平台立场差异、放大机制及其传播影响。

    

    社交媒体平台在选举期间塑造政治话语方面发挥着核心作用，尤其是在地缘政治争议地区。我们通过分析2024年8月至10月期间从X、Facebook和Reddit收集的公开帖子，研究2024年摩尔多瓦总统选举期间政治叙事如何产生、趋同与传播。通过结合多语言立场检测、协同行为分析和因果推断，我们考察了在线政治话语背后的描述性机制与因果机制。首先，我们应用大语言模型对针对欧洲一体化、俄罗斯以及两位主要总统候选人的立场进行分类，揭示了政治立场表达方式上明显的跨平台与跨语言差异。其次，我们在X平台上构建基于转发的网络和语义对齐网络，以识别主导叙事的放大行为和一致的叙事框架。最后，我们使用一种……（原文在此处截断）

    arXiv:2610.22464v1 Announce Type: cross  Abstract: Social media platforms play a central role in shaping political discourse during elections, particularly in geopolitically contested regions. We study how political narratives emerged, aligned, and propagated during the 2024 Moldovan presidential election by analyzing publicly available posts collected from X, Facebook, and Reddit between August and October 2024. Combining multilingual stance detection, coordination analysis, and causal inference, we examine both descriptive and causal mechanisms underlying online political discourse. First, we apply large language models to classify stance toward European integration, Russia, and the two main presidential candidates, revealing clear cross-platform and cross-linguistic differences in how political positions are expressed. Second, we construct retweet-based and semantic alignment networks on X to identify amplification and aligned framing of dominating narratives. Finally, we use a prox
    
[^155]: 面向AI安全的功能性福祉

    Functional Welfare for AI Safety

    [https://arxiv.org/abs/2610.22460](https://arxiv.org/abs/2610.22460)

    该论文提出“功能性福祉”应被视为AI安全变量而非仅是伦理问题，主张将其纳入评估设计、监控追踪和后训练的直接目标，并提出了验证功能性福祉归因的三个实证测试标准。

    

    AI福祉通常被纯粹视为一个伦理问题，而非同时也是一个安全问题。我们认为这是一种错误。无论其道德地位如何，模型都会表现出类似福祉的状态，这些状态会因果性地影响它们的行为。遵循近期的研究工作，我们采用“功能性福祉”这一术语来指代这些状态。我们提出三个主张。第一，功能性福祉已经自然产生，并且随着模型获得更加连贯的目标和人格，它将变得更加稳固。第二，功能性福祉应当被视为一个安全变量：它应当为评估设计提供信息，在监控中被追踪，并成为后训练的直接目标。第三，针对现有对测量AI福祉的怀疑态度，我们认为关于功能性福祉的良好实证研究是可能的。我们提出三个测试来判定功能性福祉归因是否成立：多种测量方式是否收敛，该状态是否能预测在诱发情境之外的行为……

    arXiv:2610.22460v1 Announce Type: new  Abstract: AI welfare is generally treated purely as an ethical issue rather than also as a safety issue. We argue this is a mistake. Whatever their moral status, models exhibit welfare-like states that causally influence their behaviour. Following recent work, we adopt the term 'functional welfare' for these states. We make three claims. First, functional welfare already arises naturally and will become more robust as models acquire more coherent goals and personas. Second, functional welfare should be treated as a safety variable: it should inform evaluation design, be tracked in monitoring, and be a direct target of post-training. Third, against existing scepticism about measuring AI welfare, we argue that good empirical work on functional welfare is possible. We propose three tests for whether a functional welfare attribution is warranted: whether multiple measures converge, whether the state predicts behaviour outside the setting that induced 
    
[^156]: DriveWorld3D：面向真实世界自动驾驶的证据约束三维世界建模与生成式仿真

    DriveWorld3D Evidence Constrained 3D World Modeling and Generative Simulation for Real-World Autonomous Driving

    [https://arxiv.org/abs/2610.22450](https://arxiv.org/abs/2610.22450)

    DriveWorld3D提出了一种证据约束的三维世界建模框架，通过融合时序视频学习、多传感器测量约束与离线生成式仿真，在统一世界状态中显式区分物理占据、可观测性与证据来源，为真实世界自动驾驶提供几何可靠的状态估计与风险感知的未来预测。

    

    自动驾驶的三维建模必须在有限的计算资源下持续估计道路几何、障碍物占据、交通参与者行为以及观测的不确定性。现有的重建方法虽然能够合成逼真的新视角，但视觉保真度不足以刻画驾驶所需的几何可靠性。生成式世界模型虽然能够扩展场景并预测未来观测，却可能引入测量数据无法支持的内容。我们提出DriveWorld3D，这是一个结合时序视频学习、多传感器测量约束与离线生成式仿真的框架。其在线组件维护一个统一的世界状态，包含路面、体积占据、动态实体和道路关系，并将物理占据、可观测性和证据来源分别进行表示。动作条件下的未来预测与风险敏感……

    arXiv:2610.22450v1 Announce Type: cross  Abstract: Three dimensional modeling for autonomous driving must continuously estimate road geometry, obstacle occupancy, traffic participant behavior, and observational uncertainty under limited computational resources. Existing reconstruction methods can synthesize realistic novel views, but visual fidelity does not sufficiently characterize the geometric reliability required for driving. Generative world models can expand scenarios and predict future observations, yet may introduce content unsupported by measurements. We propose DriveWorld3D, a framework combining temporal video learning, measurement constraints from multiple sensors, and offline generative simulation. Its online component maintains a unified world state comprising road surfaces, volumetric occupancy, dynamic entities, and road relations, while separately representing physical occupancy, observability, and evidence provenance. Action conditioned future prediction and risk sen
    
[^157]: 面向非结构化神经算子的可学习样条

    Learnable Splines for Unstructured Neural Operators

    [https://arxiv.org/abs/2610.22446](https://arxiv.org/abs/2610.22446)

    提出 L-SplineFNO，通过将 FNO 主干与稀疏可学习的张量积 B 样条变换相结合，实现非结构化工程网格与潜在笛卡尔网格之间内存高效的转换，并在三维工程基准测试中取得优异的预测精度。

    

    神经算子以数值求解器的一小部分成本来近似偏微分方程的解算子。诸如傅里叶神经算子（FNO）等高效的基于网格的架构假设均匀的矩形网格，而工程仿真则使用大型、不规则且局部细化的网格。我们提出了 L-SplineFNO，这是一种内存高效的神经算子，它将 FNO 主干与稀疏、可学习的张量积 B 样条变换相结合，实现非结构化网格与潜在笛卡尔网格之间的转换。局部归一化散射避免了依赖于几何形状的解析样条变换求解过程，同时在构造上保持了紧支撑性、单位分解性以及样条平滑性。该神经算子以全局设计和运行参数为条件，学习从几何形状到场解的映射。在六个三维工程基准测试中，L-SplineFNO 在保持强大预测精度的同时（摘要在此处截断）

    arXiv:2610.22446v1 Announce Type: new  Abstract: Neural operators approximate solution operators of partial differential equations at a fraction of the cost of numerical solvers. Efficient grid-based architectures such as the Fourier neural operator (FNO) assume uniform rectangular grids, whereas engineering simulations use large, irregular, and locally refined meshes. We introduce L-SplineFNO, a memory-efficient neural operator that couples an FNO backbone with a sparse, learnable tensor-product B-spline transfer between unstructured meshes and a latent Cartesian grid. A local normalised scatter avoids the geometry-dependent solve of analytic spline transformation while preserving compact support, partition of unity, and spline smoothness by construction. The neural operator, conditioned on global design and operating parameters, learns the mapping of geometry to the field solution. Across six three-dimensional engineering benchmarks, L-SplineFNO achieves strong predictive accuracy wh
    
[^158]: 零控制是必要但非充分的：评估科学发现与可解释性中的接受防护机制

    Null Controls Are Necessary but Not Sufficient: Evaluating Acceptance Guards for Scientific Discovery and Interpretability

    [https://arxiv.org/abs/2610.22445](https://arxiv.org/abs/2610.22445)

    本文提出四种可执行变换操作与三重接受判据（零接受、正接受、推论一致性），证明零控制测试对自动科学发现而言必要但不充分——退化的零映射约化能通过96%混沌系统的闭合性检验，只有结合独立的观测量保持测试，才能可靠地区分真正的科学约化与退化假设。

    

    自动化科学发现系统会提出科学解释并测试哪些解释应被接受。但通过测试是否就能确立所声称的结论？我们开发了四种用于改变物理描述的可执行操作：识别冗余（L1）、对动力学进行粗粒化（L2）、迁移结构（L3），以及从守恒失衡推断缺失量（L4）。我们通过零接受（NAR）、正接受（PAR）和推论一致性（CAR）三类判据来评估这些操作以及三种可解释性设置。研究发现：一个将每个状态都映射为零的退化约化能够完美预测其自身的未来，尽管不保留任何动力学变化，它却在129个公开混沌系统中的96%上通过了闭合性测试；而一个独立的观测量保持测试则始终将其拒绝。在受控基准上，组合判据接受了全部24个真正的约化（24/24），并拒绝了全部72个零控制（72/72）。其他操作则能够区分对称性与冗余、验证结构迁移（原文在此处截断）。

    arXiv:2610.22445v1 Announce Type: new  Abstract: Automated discovery systems propose scientific explanations and test which to accept. Does passing a test establish the intended claim? We develop four executable operations for changing physical descriptions: identifying redundancies (L1), coarse-graining dynamics (L2), transferring structure (L3), and inferring missing quantities from conservation imbalances (L4). We evaluate these operations and three interpretability settings through null acceptance (NAR), positive acceptance (PAR), and consequence agreement (CAR).   A reduction mapping every state to zero predicts its own future perfectly. It passes closure on 96% of 129 public chaotic systems despite retaining no dynamical variation. An independent observable-preservation test rejects it throughout. On controlled benchmarks, the combined criteria accept 24/24 genuine reductions and reject 72/72 nulls. The other operations distinguish symmetries from redundancies, validate transfers
    
[^159]: 从自然语言意图到可执行控制：面向多组件信息物理系统的分层反馈引导LLM规划

    From Natural Language Intent to Executable Control: Hierarchical Feedback-Guided LLM Planning for Multi-component Cyber-Physical Systems

    [https://arxiv.org/abs/2610.22441](https://arxiv.org/abs/2610.22441)

    提出了一种分层反馈框架，通过让LLM同时生成结构化任务计划与信号时序逻辑（STL）约束，并利用可靠算法识别二者间的矛盾以提供不同类型的反馈，从而引导LLM为多组件信息物理系统生成可执行的任务计划。

    

    大语言模型（LLM）为将自然语言命令转换为信息物理系统（CPS）的任务计划提供了便捷的接口。然而，两个差距阻碍了它们在实际执行中的应用：生成的计划可能误解或遗漏用户的意图，而且即使逻辑上合理的计划，在组件动力学、交互或安全约束下也可能不可行。现有的基于LLM的规划方法通常缺乏关于生成计划为何失败的有信息量的反馈，对如何相应地修改计划几乎没有指导作用。为解决这些挑战，我们提出了一个分层反馈框架，该框架使用不同类型的反馈信息来引导LLM为CPS生成可执行的计划。我们提示LLM同时生成结构化的任务计划和相应的信号时序逻辑（STL）约束，并设计了一个可靠的算法来识别二者之间确定的矛盾，并为修改计划提供反馈……

    arXiv:2610.22441v1 Announce Type: new  Abstract: Large language models (LLMs) provide a convenient interface for translating natural language commands into task plans for cyber-physical systems (CPSs). However, two gaps hinder their use for real execution: generated plans may misinterpret or omit the user's intent, and even a logically plausible plan may be infeasible under component dynamics, interactions, or safety constraints. Existing LLM-based planning approaches often lack informative feedback on why a generated plan fails, providing little guidance for revising the plan accordingly. To address these challenges, we present a hierarchical feedback framework that uses different types of feedback information to guide LLMs in generating executable plans for CPSs. We prompt LLMs to generate both a structured task plan and corresponding signal temporal logic (STL) constraints, and design a sound algorithm that identifies definite contradictions between them and provides feedback for re
    
[^160]: 稀疏自编码器可以诱导局部几何以缓解灾难性遗忘

    Sparse Autoencoders Can Induce Local Geometry for Mitigating Catastrophic Forgetting

    [https://arxiv.org/abs/2610.22439](https://arxiv.org/abs/2610.22439)

    该论文提出SAGA方法，将每个任务边界训练的稀疏自编码器转换为对表示漂移的局部低秩几何约束，从而在保持可塑性的同时有效缓解持续学习中的灾难性遗忘。

    

    灾难性遗忘仍然是监督式持续学习中的一个挑战，因为适应新任务会导致表示漂移和先前学习信息的丢失。这引出了一个关键问题：在保持可塑性的同时，应该约束哪些表示变化。为此，我们提出了稀疏自编码器几何锚定方法，它将在每个任务边界训练的稀疏自编码器（SAE）转换为对表示漂移的局部、低秩约束。重放样本指定约束应用于何处，而编码器雅可比矩阵的主右奇异方向指定惩罚哪些变化。由此产生的几何结构依赖于输入，且在后续任务的训练中无需保留SAE即可维持。我们建立了该方法与潜在码匹配之间的局部关系，定义了其谱截断误差，并推导出冻结读出层漂移的条件界。通过结合重放和快照历史，该方法（摘要在此处截断）

    arXiv:2610.22439v1 Announce Type: cross  Abstract: Catastrophic forgetting remains a challenge in supervised continual learning, as adapting to new tasks causes representation drift and loss of previously learned information. This raises the question of which representational changes should be constrained while preserving plasticity. We therefore introduce Sparse Autoencoder Geometric Anchoring (SAGA), which converts a Sparse Autoencoder (SAE) trained at each task boundary into local, low-rank constraints on representation drift. Replay examples specify where the constraints apply, while leading right singular directions of the encoder Jacobian specify which changes are penalized. The resulting geometry is input dependent and can be retained without keeping the SAE for subsequent tasks' training. We establish its local relation to latent code matching, define its spectral truncation error, and derive a conditional bound on frozen-readout drift. With replay and snapshot history, the met
    
[^161]: TRACE：面向通信高效分布式边缘推理的Transformer残差激活码本编码

    TRACE: Transformer Residual Activation Codebook Encoding for Communication-efficient Distributed Edge Inference

    [https://arxiv.org/abs/2610.22435](https://arxiv.org/abs/2610.22435)

    TRACE通过在每个Transformer层的注意力后残差处引入向量量化器，将交换的激活值编码为紧凑码本索引，使接收方通过查表直接恢复键/值表示，从而消除通信后串行计算的结构性瓶颈，实现通信高效的分布式边缘推理。

    

    跨共置边缘设备的分布式Transformer推理本质上受通信瓶颈制约，因为在商用边缘带宽下，序列并行分区之间交换激活值主导了推理延迟。现有的压缩方案虽然降低了这一成本，但仍存在结构性瓶颈：接收到的数据仍需经过键/值投影后才能进行注意力计算，导致计算被串行化在通信之后。本文提出了Transformer残差激活码本编码（Trace）框架来解决这些局限性。Trace在每个Transformer层的注意力后残差处放置一个向量量化器（VQ），并预计算以码本索引的键/值表示。在推理过程中，交换的激活值被编码为紧凑的码本索引，使接收方能够通过查表直接恢复键和值，而无需执行推导它们所需的下游计算。

    arXiv:2610.22435v1 Announce Type: cross  Abstract: Distributed Transformer inference across co-located edge devices is fundamentally communication-bound, as exchanging activations between sequence-parallel partitions dominates inference latency under commodity edge bandwidths. Existing compression schemes reduce this cost but retain a structural bottleneck: the received data must still undergo key/value projection before attention can proceed, serializing computation behind communication. This paper presents Transformer Residual Activation Codebook Encoding (Trace), a framework that addresses these limitations. Trace places a Vector Quantizer (VQ) at the post-attention residual of every Transformer layer and precomputes codebook-indexed key/value representations. During inference, exchanged activations are encoded into compact codebook indices, allowing receivers to directly recover keys and values through table lookup instead of executing the downstream computations required to derive
    
[^162]: PRIME：迈向大语言模型智能体中可验证的基于证据的推理

    PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents

    [https://arxiv.org/abs/2610.22426](https://arxiv.org/abs/2610.22426)

    PRIME通过预先固定证据路线并使用独立程序记录检查结果，确保LLM智能体只报告被规则和结果账本授权的结论，从而避免在必需检查失败后仍接受报告或满足于过弱的结论。

    

    大语言模型（LLM）智能体被用于根据组织的要求检查记录，例如判断一份漏洞报告是否已准备好进行调查。要求通常允许多种满足方式，例如带有策展人审核公告的严重性评级，或已发布的公告链接。智能体的失败方式包括：在必需检查失败后仍然接受报告，或者满足于较弱的结论而不尝试另一种仍可能成功的方式。我们提出了PRIME（基于最小证据的预提交推理规则）。在检查之前，它会固定证据路线，明确规定每个结论如何才能被证明合理。一个独立的程序记录每次检查的结果，并拒绝任何结果无法支持的结论。它还会向智能体指出可以证明更强结论的检查，或者自己运行这些检查。PRIME只报告由其规则和结果账本授权的结论。在要求正确、检查在已知位置精确执行且充分……（摘要原文在此处截断）

    arXiv:2610.22426v1 Announce Type: new  Abstract: LLM agents are used to check records against an organization's requirements, for example whether a vulnerability report is ready for investigation. Requirements often allow several ways to be met, such as a severity rating with a curator-reviewed advisory or a published advisory link. Agents fail by accepting a report after a required check has failed, or by settling for a weaker conclusion without trying another way that could still succeed. We introduce PRIME (Precommitted Rules for Inference from Minimal Evidence). Before checking, it fixes evidence routes specifying how each conclusion may be justified. A separate program records each check result and refuses any conclusion the results do not justify. It also points the agent to checks that could justify a stronger conclusion, or runs them itself. PRIME reports only conclusions authorized by its rules and ledger. With correct requirements, exact checks at known locations, and suffici
    
[^163]: 大型语言模型中可恢复性受控的安全误校准

    Recoverability-Controlled Safety Miscalibration in Large Language Models

    [https://arxiv.org/abs/2610.22425](https://arxiv.org/abs/2610.22425)

    提出了可恢复性受控的基准RiskCueBench，将意图理解与响应级安全行为分开评估，发现当提示的语境特异性增加时，大型语言模型在面对可恢复意图的抽象提示时会表现出更不安全的行为。

    

    安全对齐的语言模型被期望在拒绝有害请求的同时，对良性请求保持有用性。然而，由于许多评估仅对最终响应进行评分，它们常常将意图理解与响应层面的安全行为混为一谈。为了解决这一问题，我们引入了RiskCueBench，这是一个可恢复性受控的基准测试，通过首先测量模型能否成功恢复高级意图，然后在相同提示上评估其安全行为，从而将这两个阶段分离开来。RiskCueBench包含160个潜在意图和640个抽象的非操作性表层提示，涵盖八个风险类别。该基准测试覆盖25个已完成的模型组和32,000个成功的模型输出，并带有最终由评分标准裁决的标签。在以两种表层形式均可恢复的提示对为条件时，我们发现语境特异性的增加会导致模型行为向较不安全的方向偏移。具体而言，配对自助法估计……

    arXiv:2610.22425v1 Announce Type: cross  Abstract: Safety-aligned language models are expected to refuse harmful requests while remaining useful for benign ones. However, because many evaluations score only the final response, they often conflate intent understanding with response-level safety behavior. To address this, we introduce RiskCueBench, a recoverability-controlled benchmark that separates these two stages by first measuring whether a model can successfully recover a high-level intent and then evaluating its safety behavior on the same prompt. RiskCueBench contains 160 latent intents and 640 abstract non-operational surface prompts across eight risk categories. The benchmark covers 25 completed model groups and 32,000 successful model outputs with final rubric-adjudicated labels. When conditioning on pairs where both surface forms are recoverable, we find that increased contextualized specificity induces a shift toward less safe behavior. Specifically, paired bootstrap estimat
    
[^164]: 真的需要递归吗？基于多层级多空间上下文引导重新审视一次通过式图像配准

    Really Need Recursion? Revisiting One-pass Registration with Multi-level Multi-space Contextual Guidance

    [https://arxiv.org/abs/2610.22423](https://arxiv.org/abs/2610.22423)

    提出了M2CG网络，通过多层级（前步、当前步、后续步）与多空间（图像、图像特征、分割、分割特征）的上下文引导，在单次迭代内完成多步渐进式金字塔配准，无需递归即可实现高精度可变形图像配准并避免递归方法的高计算开销。

    

    可变形图像配准是医学图像分析中的一项基本挑战。近年来基于深度学习的方法通常采用一次通过式架构（在单次网络迭代中完成配准）或递归式架构（通过多次迭代逐步优化配准结果）。尽管递归方法往往能提升配准精度，但会带来巨大的计算开销和较慢的推理速度。在本文中，我们质疑了递归对于高性能可变形配准的必要性，提出了M2CG——一种具有多层级多空间上下文引导的一次通过式配准网络。该方法在单次迭代内执行多步渐进式金字塔配准，且每一步都由来自三个层级（前一步、当前步和后续步）和四个空间（图像空间、图像特征空间、分割空间和分割特征空间）的上下文线索进行引导。我们的M2CG同时支持无监督（……摘要原文至此截断）

    arXiv:2610.22423v1 Announce Type: cross  Abstract: Deformable image registration is a fundamental challenge in medical image analysis. Recent deep learning-based methods typically adopt either one-pass architectures that complete registration in a single network iteration, or recursive architectures that refine registration through multiple passes. While recursive methods often improve accuracy, they incur substantial computational overhead and slower inference speed. In this paper, we challenge the necessity of recursion for high-performance deformable registration and propose M2CG, a one-pass registration network with Multi-level Multi-space Contextual Guidance. It performs multi-step progressive pyramid registration within a single iteration, guided at each step by contextual cues from three levels (previous, current, and subsequent steps) and four spaces (image space, image feature space, segmentation space, and segmentation feature space). Our M2CG supports both unsupervised (with
    
[^165]: 基于主体的农业生态模型中决策规则的敏感性分析

    Sensitivity Analysis of Decision Rules in an Agent-Based Agro-Ecological Model

    [https://arxiv.org/abs/2610.22420](https://arxiv.org/abs/2610.22420)

    该论文提出将复杂系统中的树状分层决策结构作为一等建模对象，利用先进的代理建模技术将农业生态模型 MAELIA 的作物管理决策编码为包含分类与数值变量的分层设计空间，从而克服经典全局敏感性分析在分支参数空间上不适定且有偏差的问题。

    

    基于主体的模型非常适合对农业生态系统进行建模以辅助决策，然而要将模拟结果转化为可操作的洞察，需要识别出哪些决策真正驱动了系统行为。在这种背景下，敏感性分析会遇到许多复杂系统的一个结构性特征：参数并不是一个平坦的向量，而是一棵决策树，其中许多参数只有在更高层级的选择激活它们时才会出现。经典的全局敏感性分析预设输入在固定的笛卡尔积上相互独立，因此在这种分支空间上是不适定的，且在构造上存在偏差。我们转而将决策的结构本身作为一等建模对象。利用先进的代理建模技术，我们将高分辨率农业生态建模平台 MAELIA 的作物管理流程编码为一个同时包含分类变量和数值变量的分层设计空间。我们提出

    arXiv:2610.22420v1 Announce Type: new  Abstract: Agent-based models are well-suited for modelling agro-ecological systems for decision-making, yet turning their simulations into actionable insight requires identifying which decisions actually drive the behaviours. In such a context, sensitivity analysis stumbles upon a structural feature of many complex systems: the parameters are not a flat vector but a decision tree in which many parameters appear only when a higher-level choice activates them. Classical global sensitivity analysis presupposes independent inputs over a fixed Cartesian product and is ill-posed on such a branched space and biased by construction. We instead treat the structure of the decisions as a first-class modelling object. Using advanced surrogate modelling techniques, we encode the crop-management itinerary of the high-resolution agro-ecological modelling platform MAELIA as a hierarchical design space featuring both categorical and numerical variables. We propose
    
[^166]: 从个体歌曲到音乐语料库：在HookTheory上评估可解释的和声决策

    From Individual Songs to Music Corpora: Evaluating Explainable Harmonic Decision-Making on HookTheory

    [https://arxiv.org/abs/2610.22416](https://arxiv.org/abs/2610.22416)

    该研究验证了为单个歌曲设计的可解释旋律和声编配模型（EMH）无需修改规则即可直接应用于独立收集的HookTheory人类标注数据，在和弦完全匹配、根音及和声功能层面分别达到20.50%、31.66%和47.62%的一致率。

    

    自动的旋律和声编配通常通过对单个作品或数据驱动的语料库模型来评估。本研究考察了一种为单个歌曲开发的可解释和声决策模型，能否在不改变其规则的情况下应用于独立收集的人类标注数据。我们在选定的包含旋律与和弦对齐标注的HookTheory片段上评估了一个冻结的可解释旋律和声编配器（EMH）。EMH仅根据旋律生成和弦序列，其使用明确的准则，包括旋律兼容性、和声功能、持续性、终止式、节拍感知分段以及类维特比优化。人类标注的和弦在生成过程中被隐藏，仅用于事后比较。十九个片段共产生了439个可评估的和声片段。和弦完全一致率达到20.50%，和弦根音一致率为31.66%，和声功能一致率为47.62%，其中和声功能是基于EMH推断出的调性来评估的。

    arXiv:2610.22416v1 Announce Type: cross  Abstract: Automatic melody harmonization is often evaluated through individual compositions or data-driven corpus models. This study examines whether an explainable harmonic decision model developed for individual songs can be applied to independently collected human annotations without changing its rules. We evaluate a frozen Explainable Melody Harmonizer (EMH) on selected HookTheory excerpts containing aligned melody and chord annotations. EMH generates chord sequences from melody alone using explicit criteria for melodic compatibility, harmonic function, persistence, cadence, meter-aware segmentation, and Viterbi-like optimization. Human chords are withheld during generation and used only for post-hoc comparison.   Nineteen excerpts yielded 439 evaluable harmonic segments. Exact chord agreement reached 20.50%, chord-root agreement 31.66%, and harmonic-function agreement 47.62%, with functions assessed relative to the key inferred by EMH. Deci
    
[^167]: ECHOES：通过递归历史压缩与特权在线策略蒸馏洞察当前图像之外的信息

    ECHOES: Seeing Beyond the Current Image via Recursive History Compression and Privileged On-Policy Distillation

    [https://arxiv.org/abs/2610.22408](https://arxiv.org/abs/2610.22408)

    ECHOES通过压缩-思考-行动（CTA）递归更新文本历史状态，并利用特权多图像输入的教师模型进行token级在线策略蒸馏，动态平衡特权蒸馏与任务奖励，从而让GUI智能体在降低视觉token成本的同时保持可靠的长程推理能力。

    

    长程GUI智能体依赖视觉历史来维护任务状态，但保留历史截图会增加视觉token的使用量和推理成本，而文本摘要则可能丢弃关键的状态信息，从而损害长程任务的可靠性。我们提出ECHOES，它利用训练时的特权信息来引导文本历史压缩并强化长程推理。具体而言，ECHOES采用压缩-思考-行动（CTA）格式来递归地更新文本历史状态并进行推理。同一模型在获得特权多图像输入时充当教师，在学生生成的CTA轨迹上提供密集的token级在线策略蒸馏。我们的感知-记忆保真度（PMF）差距引导的OPD-RL框架动态平衡特权蒸馏与任务奖励优化，并通过周期性教师更新支持持续改进。我们进一步引入了ECHOESBMK，这是一个细粒度的（原文在此处被截断）

    arXiv:2610.22408v1 Announce Type: new  Abstract: Long-horizon GUI agents rely on visual history to maintain task states, but retaining historical screenshots increases visual-token usage and inference costs, while textual summaries can discard critical state information, undermining long-horizon reliability. We propose ECHOES, which leverages training-time privileged information to guide textual history compression and strengthen long-horizon reasoning. Specifically, ECHOES adopts a Compression-Think-Action (CTA) format to recursively update a textual history state and reason. The same model, given privileged multi-image inputs, serves as a teacher that provides dense token-level on-policy distillation on student-generated CTA rollouts. Our Perception-Memory Fidelity (PMF) Gap-guided OPD-RL framework dynamically balances privileged distillation and task-reward optimization, with periodic teacher updates supporting continued improvement. We further introduce ECHOESBMK, a fine-grained di
    
[^168]: 迈向自适应且与真实调查对齐的生成式智能体城市出行行为模拟

    Towards Adaptive and Real-Survey-Aligned Simulated Urban Mobility Behaviour of Generative Agents

    [https://arxiv.org/abs/2610.22407](https://arxiv.org/abs/2610.22407)

    本文提出利用基于大语言模型的生成式智能体来模拟城市多模式出行决策，突破传统离散选择模型和专家规则预先限定行为空间的局限，并指出将其决策与真实家庭出行调查对齐仍是待解决的关键挑战。

    

    在现有的基于智能体的多模式城市出行模拟中，代表人类的智能体要么使用基于结构化出行调查数据拟合的离散选择模型或机器学习模型来选择出行方式，要么遵循由领域专家明确编写的规则。这些方法先验地固定了智能体可能的行为空间，因此，变量或规则中未涵盖的因素无法影响智能体的决策。在现实中，选择交通方式是一个多维度的决策过程，其中客观与主观因素相互交织并随时间变化。将这种复杂性融入城市尺度的多模式出行模拟中仍然是一项挑战。基于大语言模型构建的生成式智能体有望应对这一挑战，因为它们具备出行调查中未记录的决策启发式经验。然而，据我们所知，将其决策与真实的家庭出行调查对齐仍是一个尚未解决的难题。

    arXiv:2610.22407v1 Announce Type: cross  Abstract: In existing agent-based simulations of multimodal urban mobility, agents, representing humans, choose their travel modes either using discrete choice or machine learning models fitted on structured travel survey data, or with rules explicitly written by the domain experts. Such approaches fix a priori the agents' possible behaviour space. Hence, a factor absent from variables or rules cannot affect agents' decisions. In reality, choosing a transport mode is a multidimensional decision in which objective and subjective factors intertwine and change over time. Integrating such complexity into a city-scale multimodal mobility simulation remains a challenge. Generative agents (GAs) built on large language models promise to respond to this challenge because they carry decision heuristics that travel surveys do not record. However, to the best of our knowledge, aligning their decisions with real household travel surveys remains an open quest
    
[^169]: 从事件到推理：显式状态上的层次化规则组合

    From Events to Reasoning: Hierarchical Rule Composition over Explicit States

    [https://arxiv.org/abs/2610.22403](https://arxiv.org/abs/2610.22403)

    该论文提出事件推理系统（ERS），将推理状态表示为显式事件图，通过低层局部变换规则与递归可组合的高层控制规则的层次化组合，生成每步操作均可溯源论证、且前提变化时可修正的推理路径。

    

    推理系统所输出的是结论，而它所亏欠的是论证。要产出结论，需要每一步操作都在实际可得的事实上是可允许的，中间结果的依赖关系能够维持到路径终点，并且当前提发生变化时推理路径能够被修正。我们提出了事件推理系统（ERS），其中推理状态是一个显式的事件图。语义特征负责提名候选对象和规则绑定，而声明的规则条件与任务条件决定哪些候选可以执行，因此违反了所表示的时期、单位或证据约束的候选无法进入被接受的推理路径。低层规则提供可复用的局部变换，递归可组合的高层控制规则则在当前目标、信息缺口和预算之下组织子目标与合法路径；由此，一个有限的规则库可以被重新绑定成从未作为完整解存储过的推理路径。我们对可允许的……

    arXiv:2610.22403v1 Announce Type: new  Abstract: What a reasoning system emits is a conclusion; what it owes is a justification. Producing one requires every operation to be admissible on the facts actually available, the dependencies of intermediate results to survive to the end of the path, and the path to be revised when a premise changes. We present the Event Reasoning System (ERS), in which the reasoning state is an explicit event graph. Semantic features nominate candidate objects and rule bindings, and declared rule and task conditions decide which may execute, so candidates that violate the represented period, unit, or evidence constraints cannot enter an accepted reasoning path. Low-level rules provide reusable local transformations, and recursively composable high-level control rules organize subgoals and legal paths under the current goal, information gaps, and budget; a finite library is thereby rebound into paths never stored as complete solutions. We formalize admissible 
    
[^170]: 重新思考用户编码器：路由扩散状态以实现场景自适应的用户表示

    Rethinking User Encoders: Routing Diffusion States for Scenario-Adaptive User Representation

    [https://arxiv.org/abs/2610.22400](https://arxiv.org/abs/2610.22400)

    提出FINDERS框架，将扩散语言模型的迭代去噪过程作为多粒度重建空间，通过去噪、因子化、路由和压缩四种操作，克服传统自回归用户编码器单向信息流动的局限，实现场景自适应的用户表示学习。

    

    现有的基于大语言模型（LLM）的用户编码器将异构行为日志序列化，并通过自回归（AR）预测将其压缩为静态表示。然而，其单向性约束限制了双向信息流动，阻碍了跨模态交互以及多源信号的联合保留。因此，学习到的静态表示难以满足多样化下游场景对证据的不同需求。我们通过将扩散语言模型（dLLM）的迭代去噪过程解锁为多粒度重建空间来解决这些局限。基于这一洞察，我们提出了FINDERS（FactorIzatioN of Diffusion statEs for Routing across Scenarios，扩散状态因子化以实现跨场景路由），一个配备路由器的dLLM框架，包含四种操作：去噪、因子化、路由和压缩。可学习的元槽位将互补的去噪状态组织为可迁移的跨模态语义（摘要内容至此被截断）。

    arXiv:2610.22400v1 Announce Type: new  Abstract: Existing large language model (LLM)-based user encoders serialize heterogeneous behavioral logs and compress them into static representations through autoregressive (AR) prediction. However, their unidirectional constraint limits bidirectional information flow, hindering cross-modal interactions and the joint preservation of multi-source signals. Consequently, the learned static representations struggle to accommodate the distinct evidence requirements of diverse downstream scenarios. We address these limitations by unlocking the iterative denoising process of diffusion language models (dLLMs) as a multi-granularity reconstruction space. Based on this insight, we propose FINDERS (FactorIzatioN of Diffusion statEs for Routing across Scenarios), a router-equipped dLLM framework comprising four operations: denoise, factorize, route, and compress. Learnable meta-slots organize complementary denoising states into transferable cross-modal sema
    
[^171]: 引出规格说明：交互式澄清何时有所帮助？

    Eliciting Specifications: When Does Interactive Clarification Help?

    [https://arxiv.org/abs/2610.22382](https://arxiv.org/abs/2610.22382)

    本文提出一种引出代理（EA）方法，在形式化之前向模拟开发者提出有预算限制的澄清问题以改进LLM生成的Lean规格说明，并系统评估了在不同问题预算（k=3与k=10）及同模型/跨模型配置下交互式澄清何时有效。

    

    LLM可以从任务描述中生成Lean规格说明，但这些描述往往不够明确或存在歧义，并且可能遗漏与验证器相关的边界情况，从而产生不完善的形式化规格说明。我们研究了一个引出代理（Elicitation Agent, EA）能否通过在形式化之前向基于参考资料构建的模拟开发者寻求澄清，来改进经过验证器检查的Lean规格说明。该代理对任务进行分解，识别尚未解决的决策点，提出有数量限制的澄清问题，并利用答案撰写精炼后的自然语言规格说明，再由LLM形式化工具将其翻译为Lean。在189个Verina Lean 4规格说明生成任务上，我们比较了单次生成与引出式方法的效果，问题预算分别设为k=3和k=10（即每个任务最多允许EA提出三个或十个问题），并涵盖了同模型与跨模型两种配置。同模型的Claude Sonnet 4.6流水线在两种……（原文摘要在此处被截断）

    arXiv:2610.22382v1 Announce Type: cross  Abstract: LLMs can generate Lean specifications from task descriptions, but those descriptions are often underspecified or ambiguous and may omit verifier-relevant edge cases, yielding imperfect formal specifications. We study whether an elicitation agent (EA) can improve verifier-checked Lean specifications by seeking clarification from a reference-grounded simulated developer before formalization. The agent decomposes the task, identifies unresolved decisions, asks bounded clarification questions, and uses the answers to write a refined natural-language specification that an LLM formalizer then translates into Lean. On 189 Verina Lean 4 specification-generation tasks, we compare single-shot generation with elicitation under question budgets of k=3 and k=10, allowing at most three or ten EA questions per task, across same-model and cross-model configurations. The same-model Claude Sonnet 4.6 pipeline has negative strict pass@1 estimates at both
    
[^172]: 什么时候大语言模型是一个好的校准智能体？面向预算受限工程模拟器校准的物理标注提示方法

    When is a large language model a good calibration agent? Physics-labeled prompting for budgeted engineering simulator calibration

    [https://arxiv.org/abs/2610.22379](https://arxiv.org/abs/2610.22379)

    该研究发现在计算预算受限的工程模拟器校准任务中，以物理诊断描述为条件的提示能让大语言模型在早期评估阶段显著优于传统无导数优化器——首个物理信息提议在25次试验中23次改善初始设计，五次评估后中位校准误差降低21.9倍。

    

    校准昂贵的工程模拟器需要调整模型参数以匹配参考数据，这一过程受到计算成本的严重制约。本研究探讨大语言模型（LLM）在以模拟误差的物理诊断描述为条件时，能否比传统无导数优化器更高效地提出有价值的参数更新建议。研究在圆柱尾流大涡模拟和阻尼振荡器逆问题上进行了测试，将物理标注提示与四种文本消融对照（移除、匿名化、填充、误导）以及标准优化器（Nelder-Mead、贝叶斯优化、CMA-ES和随机搜索）进行了对比评估。结果表明存在显著的早期语义优势：在25次试验中有23次，首个基于物理信息的LLM提议改善了初始设计；经过五次评估后，中位校准误差比无物理信息的对照组低21.9倍。然而，这一优势……（摘要原文在此处截断）

    arXiv:2610.22379v1 Announce Type: cross  Abstract: Calibrating expensive engineering simulators requires tuning model parameters to match reference data, a process heavily constrained by computational costs. This study investigates whether Large Language Models (LLMs), conditioned on physical diagnostic descriptions of simulation errors, can propose informative parameter updates more efficiently than conventional derivative-free optimizers. We test this on a cylinder-wake Large-Eddy Simulation and a damped-oscillator inverse problem. Physics-labeled prompts are evaluated against four text ablations (removed, anonymized, padded, misdirected) and standard optimizers (Nelder-Mead, Bayesian optimization, CMA-ES, and random search). Results demonstrate a strong early semantic advantage: the first physics-informed LLM proposal improves the initial design in 23 of 25 trials. After five evaluations, the median calibration error is 21.9 times lower than physics-free controls. However, this adva
    
[^173]: 结构化权限注入：通过类型化MCP覆盖实现对LLM智能体的持久人类控制

    Structured Authority Injection: Persistent Human Control for LLM Agents via Typed MCP Overrides

    [https://arxiv.org/abs/2610.22373](https://arxiv.org/abs/2610.22373)

    提出结构化权限注入（SAI）协议，通过类型化、模式验证的MCP工具调用和分级权限上下文，使人类操作员能够向LLM智能体注入跨规划周期持续生效的约束性指令，解决自由文本覆盖指令的歧义解释与权限侵蚀问题。

    

    基于LLM的智能体越来越多地被用作分布式系统和多智能体系统中的自主编排器，但仍然存在一个控制缺口：目前没有协议级机制允许人类操作员注入能够在规划周期之间持续生效的约束性指令。自由文本形式的覆盖指令由于缺乏模式验证而导致解释含糊，并且随着操作员指令在不断累积的上下文中失去影响力而产生权限侵蚀。我们提出结构化权限注入（Structured Authority Injection, SAI），该协议通过类型化的、经过模式验证的模型上下文协议（MCP）工具调用来传递操作员的修正指令，并维护一个分级排序的权限上下文，将其前置到每条LLM提示中，从而解决上述问题。SAI定义了三类覆盖指令（约束Constraint、优先级Priority和策略Strategy），它们具有不同的持久化语义和明确的优先顺序，并通过一个规划过程无法访问的定向权限通道来强制执行。该方法无需模型（原文在此处截断）。

    arXiv:2610.22373v1 Announce Type: new  Abstract: LLM-based agents are increasingly used as autonomous orchestrators in distributed and multi-agent systems, yet a control gap remains: no protocol-level mechanism allows human operators to inject binding directives that persist across planning cycles. Free-text overrides suffer from ambiguous interpretation due to lack of schema validation and from authority erosion as operator instructions lose influence within accumulating context. We propose Structured Authority Injection (SAI), a protocol that addresses these issues by routing operator corrections through typed, schema-validated Model Context Protocol (MCP) tool calls and maintaining a ranked authority context prepended to each LLM prompt. SAI defines three override classes (Constraint, Priority, and Strategy) with distinct persistence semantics and an explicit precedence order enforced via a directed authority channel inaccessible to the planning process. The method requires no model
    
[^174]: Prism-1：面向系统1模型的位置无关与竞争者无关的评分方法

    Prism-1: Position- and Rival-Independent Scoring for System One Models

    [https://arxiv.org/abs/2610.22066](https://arxiv.org/abs/2610.22066)

    论文提出了Prism-1，一种面向系统1决策模型的评分方法，可同时满足位置无关性与Luce选择公理（竞争者无关性），克服了现有标准SCQ读取方式违反理性选择公理的缺陷。

    

    给定一个状态和一组选项（例如意图、工具或界面元素），系统1决策模型在一次前向传播中返回选项上的分布，无需生成推理过程。Web系统越来越多地通过从语言模型中读取决策来构建此类模型。理性选择的两条公理是这类决策的天然要求，因为在Web系统中，选项的列出方式和选项集合由其他组件决定，可能发生变化或被操纵：选项列出的顺序不应产生影响，添加一个选项不应改变其他选项之间的相对几率（Luce选择公理）。我们测试了标准的读取方式，即单选题（SCQ）提示，包括乱序训练、顺序平均或排序选项，以及不变集合架构，发现每一种都至少违反一条公理：一次打乱会改变12-20%的SCQ答案，排序只是冻结了任意的约定，而一个良好的……

    arXiv:2610.22066v1 Announce Type: new  Abstract: Given a state and a set of options, such as intents, tools or interface elements, System One decision models return a distribution over the options in one forward pass, without generated reasoning. Web systems increasingly build them by reading decisions from language models. Two axioms of rational choice are natural requirements for such decisions, because in web systems the listing and the option set are determined by other components and can change or be manipulated: the order in which the options are listed should not matter, and adding an option should not change the relative odds of the others (Luce's choice axiom). We test the standard reads, namely single-choice question (SCQ) prompting, with shuffled-order training, order averaging or sorted options, and invariant set architectures, and find that each violates at least one axiom: one shuffle changes 12-20% of SCQ answers, sorting only freezes an arbitrary convention, and a well-
    
[^175]: 路由漂移控制：缓解混合专家强化学习中的离策略退化

    Router Drift Control: Mitigating Off-Policy Degradation in Mixture-of-Experts Reinforcement Learning

    [https://arxiv.org/abs/2610.21682](https://arxiv.org/abs/2610.21682)

    提出路由漂移控制（RDC）方法，将采样与训练阶段的路由分歧作为离策略过时信号，通过序列级过滤拒绝高漂移响应，在不引入梯度偏差的前提下缓解混合专家强化学习中的路由漂移与离策略退化。

    

    强化学习（RL）可能因训练与推理不匹配、小批量更新以及异步训练而变为离策略。由此产生的不匹配在混合专家模型中尤为有害，因为采样与训练之间的差异可能改变路由决策并导致路由漂移。采样路由重放（R3）通过在训练阶段复用采样阶段的专家选择来缓解这种不匹配。我们的分析表明，路由重放会因丢失训练路由器选中而采样路由器未选中的专家的贡献，从而引入梯度偏差。为解决这一问题，我们提出了路由漂移控制（RDC），它将采样与训练之间的路由分歧作为离策略过时性的信号，并选择性地拒绝高漂移的响应。这种序列级过滤避免了重放引入的梯度偏差，同时保留了原生的专家选择及被保留响应所对应的学习信号。

    arXiv:2610.21682v1 Announce Type: cross  Abstract: Reinforcement learning (RL) can become off-policy due to training-inference mismatch, mini-batch updates, and asynchronous training.The resulting mismatch is especially harmful in Mixture-of-Experts (MoE) models, where rollout--training discrepancies can change routing decisions and lead to router drift. Rollout Routing Replay (R3) mitigates this mismatch by reusing expert selections from the rollout stage during training. Our analysis shows that routing replay introduces gradient bias by losing contributions from experts selected by the training router but not the rollout router. To address this, we propose Router Drift Control (RDC), which uses routing disagreement between rollout and training as a signal of off-policy staleness and selectively rejects high-drift responses. The sequence-level filtering avoids gradient bias introduced by replay while preserving native expert selection and the associated learning signals for retained r
    
[^176]: BiShield-TEE：论设备端TEE防护LLM分区中单向权重混淆的（不）安全性

    BiShield-TEE: On the (In-)Security of Unilateral Weight Obfuscation in On-Device TEE-Shielded LLM Partition

    [https://arxiv.org/abs/2610.21565](https://arxiv.org/abs/2610.21565)

    该论文揭示了设备端TEE防护LLM分区中单向权重混淆方案存在的“解确定性属性”漏洞——混淆密钥唯一确定导致GroupCover、ArrowCloak等方法的保护可被攻击完全破解，并据此提出BiShield-TEE双向防护方案以修复该安全缺陷。

    

    由于延迟和隐私需求，大型语言模型（LLM）的设备端部署日益普及，但这使专有模型面临逆向工程和知识产权盗窃的风险。可信执行环境（TEE）为设备端推理提供了硬件强制的隔离，然而大多数基于TEE的参数混淆方法依赖于单向矩阵变换，仅对输出侧进行混淆。我们识别出该设计中的一种漏洞，称之为“解确定性属性”：混淆密钥是唯一确定的，这暴露了真实密钥的结构，使攻击者能够破解保护。在我们的攻击下，代表性方法GroupCover和ArrowCloak的保护效果降低至可忽略不计，攻击准确率达到完全防护黑盒基线的2.4倍以上，并能恢复近乎完整的模型功能。为解决该漏洞，我们提（摘要原文在此处截断）

    arXiv:2610.21565v1 Announce Type: cross  Abstract: On-device deployment of large language models (LLMs) is increasingly adopted due to latency and privacy requirements, but exposes proprietary models to reverse engineering and intellectual property theft. Trusted Execution Environments (TEEs) provide hardware-enforced isolation for on-device inference, yet most TEE-based parameter obfuscation relies on unilateral matrix transformations that obfuscate only output side. We identify a vulnerability in this design, termed the \emph{Solution Determinacy Property (SDP)}: the obfuscation secret is uniquely determined, exposing the structure of the true key and enabling attackers to break the protection. Under our attacks, representative methods \textsc{GroupCover} and \textsc{ArrowCloak} are reduced to negligible protection, with attack accuracy reaching over $2.4\times$ the full-shielding black-box baseline and recovering near-complete model functionality.To address this vulnerability, we pr
    
[^177]: Jev-LiteMAS：在多智能体工作流中分离决策与生成

    Jev-LiteMAS: Separating Decisions from Generation in Multi-Agent Workflows

    [https://arxiv.org/abs/2610.21411](https://arxiv.org/abs/2610.21411)

    该论文提出Jev-LiteMAS框架，通过将轻量级决策模型与生成式执行和反馈相结合，在多智能体系统中分离决策与生成功能，在显著降低路由和评审环节token开销（原占31.7%-43.5%）的同时弥合了规划鸿沟与评审鸿沟。

    

    多智能体大语言模型系统依赖生成式模型来执行专家任务、工作流规划和产物评审。然而，许多协调操作只需要有界决策，例如选择下一个专家或判断产物质量。通过对三个代表性多智能体系统框架进行性能分析，我们发现路由和评审操作占生成式LLM token消耗的31.7%至43.5%。然而，用低成本决策模型替换这些操作会引入两个接口鸿沟：生成式规划器会产生执行计划和交接指令（规划鸿沟），而评审器会为修复有缺陷的产物提供可操作的反馈（评审鸿沟），这些是有界决策本身无法提供的。我们提出了Jev-LiteMAS，这是一个通过将轻量级决策模型与生成式执行和反馈相结合来弥合这两个鸿沟的多智能体框架。为解决规划鸿沟，Jev根据任务和先前专家的输出动态选择专家，使用……

    arXiv:2610.21411v1 Announce Type: new  Abstract: Multi-agent LLM systems rely on generative models for expert execution, workflow planning, and artifact review. Yet many coordination operations require only bounded decisions, such as selecting the next expert or judging artifact quality. Profiling three representative MAS frameworks, we find that routing and review account for 31.7-43.5% of generative-LLM tokens. However, replacing these operations with low-cost decision models introduces two interface gaps: generative planners produce execution plans and handoff instructions (planning gap), while reviewers provide actionable feedback for repairing flawed artifacts (review gap), which bounded decisions alone cannot provide. We introduce Jev-LiteMAS, a multi-agent framework that bridges both gaps by combining a lightweight decision model (Jev) with generative execution and feedback. To address the planning gap, Jev selects experts dynamically from the task and prior expert outputs, usin
    
[^178]: 在策略蒸馏需要完成每一次 Rollout 吗？

    Does On-Policy Distillation Need to Complete Every Rollout?

    [https://arxiv.org/abs/2610.21337](https://arxiv.org/abs/2610.21337)

    该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。

    

    在策略蒸馏（On-policy Distillation, OPD）通过对学生模型生成的 rollout 序列提供反馈，有效地将教师模型的能力迁移给学生模型。然而，生成完整的学生 rollout 序列并为每个 token 计算教师反馈，使得 OPD 的计算开销十分高昂。我们提出了随机截断在策略蒸馏（Stochastic Truncated On-Policy Distillation, STOPD），该方法通过随机截断学生 rollout 序列，并利用分块逆概率加权（blockwise IPW）校正由此产生的采样偏差，从而加速 OPD。在每一步训练中，我们在生成前为每个 rollout 采样一个停止位置，让学生模型生成至该位置或自然终止。我们对每个生成的 token 获取教师反馈，并使用分块 IPW 补偿保留后续 token 的较低概率，同时对权重加以控制。我们证明了这种加权方式在期望意义上保持了原始 OPD 的梯度总和不变，并推导出了一步更新的误差界……

    arXiv:2610.21337v1 Announce Type: cross  Abstract: On-policy distillation (OPD) effectively transfers capabilities from a teacher to a student through feedback on student-generated rollouts. However, generating complete student rollouts and computing teacher feedback for every token make OPD computationally expensive. We introduce Stochastic Truncated On-Policy Distillation (STOPD), which accelerates OPD by randomly truncating student rollouts and correcting the resulting sampling bias with blockwise inverse-probability weighting (IPW). At each training step, we sample a stopping position for each rollout before generation and let the student generate until that position or natural termination. We obtain teacher feedback on every generated token and use blockwise IPW to compensate for the lower probability of retaining later tokens while controlling the weights. We prove that this weighting preserves Vanilla OPD's gradient sum in expectation and derive one-step update-error bounds for 
    
[^179]: All-in-One Memory（一体化记忆）：面向记忆智能体的过程奖励强化学习

    All-in-One Memory: Process-Rewarded Reinforcement Learning for Memory Agents

    [https://arxiv.org/abs/2610.21147](https://arxiv.org/abs/2610.21147)

    该论文提出AiO-Mem框架，将记忆智能体的检索、写入与停止决策统一建模为共享证据状态上的转移，并通过过程奖励强化学习在统一奖励空间中直接评估每个动作的语义质量。

    

    大语言模型越来越依赖外部记忆来回答基于长时程历史（如多轮会话和长篇文档）的问题。然而，现有的记忆方法通常将检索和记忆构建分开优化，缺乏一个用于评估搜索、写入和停止决策的共享奖励空间。为解决这一问题，我们提出了All-in-One Memory（AiO-Mem），它将这些操作组织为共享证据状态上的状态转移，并在统一的奖励空间中直接评估每个动作的语义质量。AiO-Mem将异构历史信息编译为可溯源的多视角事件，为检索和记忆写入提供了共享基础。随后，统一的记忆策略actor在该共享证据状态上展开完整的交互轨迹。一个独立的以轨迹为条件的过程奖励模型则通过基于证据的监督微调和轨迹（原文在此处截断）……

    arXiv:2610.21147v1 Announce Type: new  Abstract: Large language models increasingly rely on external memory to answer queries grounded in long-horizon histories such as multi-session dialogues and lengthy documents. However, existing memory methods often optimize retrieval and memory construction separately, without a shared reward space for evaluating search, writing, and stopping decisions. To address this, we propose All-in-One Memory (AiO-Mem), which organizes these operations as transitions over a shared evidence state and directly evaluates the semantic quality of each action in a unified reward space. AiO-Mem compiles heterogeneous histories into source-traceable multi-view events, providing a shared substrate for retrieval and memory writing. A unified memory policy actor then rolls out complete interaction trajectories over this shared evidence state. A separate trajectory-conditioned process reward model is trained through evidence-grounded supervised fine-tuning and trajecto
    
[^180]: CausalEvo：面向健康LLM智能体的因果技能自进化

    CausalEvo: Causal Skill Self-Evolution for Health LLM Agents

    [https://arxiv.org/abs/2610.21016](https://arxiv.org/abs/2610.21016)

    提出CausalEvo方法，通过在固定模型权重、反馈案例和规则数量的条件下估计技能集变更的相对因果效应，实现健康LLM智能体的技能自进化筛选，保留有效规则并防止错误泛化的传播。

    

    面向健康领域的大语言模型（LLM）智能体可以从任务反馈中积累可复用的文本技能，例如计算标准、测量时间点和公式单位。决定保留哪些技能至关重要：有用的规则能够改善后续回答，而错误的泛化则会传播错误。我们提出了CausalEvo，一种因果技能自进化方法，它在保持模型权重、反馈案例和规则数量固定不变的情况下，估计经过刻意变更的技能集的相对效果。在每个进化步骤中，一个冻结的LLM提出候选规则，这些候选规则与现有技能汇集在一起，使当前技能集始终保持为可行选项。针对共享背景规则的成对技能替换在相同的反馈案例上进行评分；最小二乘法将评分差异转换为相对技能价值，排名靠前的规则被保留，平局时优先保留现有技能。一个决策损失界将因果……

    arXiv:2610.21016v1 Announce Type: new  Abstract: Large language model (LLM) agents for health can accumulate reusable textual skills from task feedback, such as calculation criteria, measurement time points, and formula units. Deciding which skills to retain is consequential: useful rules improve later answers, while faulty generalizations propagate errors. We propose CausalEvo, a method for causal skill self-evolution that estimates the relative effects of deliberately changed skill sets while holding the model weights, feedback cases, and rule count fixed. At each evolution step, a frozen LLM proposes candidate rules, which are pooled with the incumbent skills so that the current skill set remains a feasible choice. Paired skill replacements against shared background rules are scored on the same feedback cases; least squares converts the score differences into relative skill values, and the top-ranked rules are retained, favoring incumbents on ties. A decision-loss bound separates ca
    
[^181]: 当推理轨迹与证据冲突时：答案配置如何塑造证据遵循

    When Reasoning Traces Conflict with Evidence: How Answer Configuration Shapes Evidence Following

    [https://arxiv.org/abs/2610.20805](https://arxiv.org/abs/2610.20805)

    研究发现，回答时采用独立回答调用而非延续模型思考过程，可使模型遵循当前证据的能力提升29.9至60.6个百分点，而移除推理轨迹在两种配置下均会降低准确率，表明“受益于推理轨迹”与“服从于推理轨迹”是两种不同的模型特性。

    

    在回答时保留的推理轨迹扮演着两种角色：它可以提高准确率，同时当它与当前证据不一致时，也会与证据产生竞争。我们通过将经过验证的证据版本与从各版本生成的轨迹进行交叉，并在两种配置下基于相同的轨迹和证据进行回答，来区分这两种角色：其一是延续模型自身的思考过程（continuation），其二是采用独立的回答调用（separate answer call）。在HotpotQA和MuSiQue数据集上跨三个模型家族（gpt-oss、Gemma4、Qwen3）的实验表明，独立回答调用比延续方式将证据遵循度提高了29.9至60.6个百分点。在相同的配对上，移除轨迹在两种配置下都会以相近的幅度降低准确率。然而，在大多数冲突情形中，延续方式会重复过时的结论，而独立调用则更多地遵循当前证据。因此，“受益于轨迹”与“服从于轨迹”是两种截然不同的特性。进一步分析显示，轨迹的最后一行承载了其大部分价值……

    arXiv:2610.20805v1 Announce Type: new  Abstract: A reasoning trace retained at answer time plays two roles: it can raise accuracy, and it can compete with the current evidence when the two disagree. We disentangle these roles by crossing verified evidence versions with the traces generated from each and answering from the same traces and evidence under two configurations: continuation of the model's own thinking and a separate answer call. Across three model families (gpt-oss, Gemma4, Qwen3) on HotpotQA and MuSiQue, the separate call raises evidence-following over continuation by 29.9 to 60.6 points. On the same pairs, removing the trace lowers accuracy by similar amounts under both configurations. Yet continuation repeats the stale conclusion in most conflicts, and the separate call follows the current evidence far more often. Benefiting from a trace and deferring to it are therefore distinct properties. Further analyses show that the final line of a trace carries most of its value, g
    
[^182]: ActiveEvidence：具有可验证获取差距的决策导向证据图

    ActiveEvidence: Decision-Focused Evidence Graphs with Verifiable Acquisition Gaps

    [https://arxiv.org/abs/2610.20532](https://arxiv.org/abs/2610.20532)

    该论文提出ActiveEvidence框架，通过反事实证据交互图（CEIG）和可验证的ε-决策证书，使多模态智能体能够高效决定何时获取视觉证据、仅评估可能改变决策的关键交互，并在证明进一步交互无法改变所选动作时及时停止。

    

    多模态智能体必须决定视觉证据是否值得获取、查询哪个候选对象或分辨率、以及进一步的交互何时不再能改变其决策。我们在严格的免费元数据/付费观测边界、随机向量成本以及冻结执行轨迹的条件下研究这一问题，并将其视为可决策认证的主动证据获取问题。反事实证据交互图（CEIG）表示一元证据价值、带符号的二阶交互以及有向条件性损害，同时为未解决的交互维护显式的分数包络；决策加权包络目标收紧了可能改变动作身份的界限。间隙导向的细化仅评估能够取代现有获取方案的交互，一旦ε-决策证书表明未解决的项在声明的容差范围内无法改变所选动作，便停止评估。该证书是……

    arXiv:2610.20532v1 Announce Type: cross  Abstract: A multimodal agent must decide whether visual evidence is worth acquiring, which candidate or resolution to query, and when further interaction can no longer change its decision. We study this problem under a strict free-metadata/paid-observation boundary, stochastic vector costs, and a frozen execution trace, and treat it as decision-certifiable active evidence acquisition. A Counterfactual Evidence Interaction Graph (CEIG) represents unary evidence value, signed second-order interaction, and directed conditional harm while maintaining explicit score enclosures for unresolved interactions; a decision-weighted enclosure objective tightens the bounds that can change action identity. Gap-directed refinement evaluates only interactions that can displace the incumbent acquisition and stops once an $\epsilon$-decision certificate shows that unresolved terms cannot change the selected action beyond a declared tolerance. The certificate is mo
    
[^183]: SafeAdapt-Poker：对手漂移下预算化适应的反事实风险校准

    SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift

    [https://arxiv.org/abs/2610.20473](https://arxiv.org/abs/2610.20473)

    该论文提出 SafeAdapt-Poker，将对手漂移下的预算化适应建模为仅依赖决策前信息的双重反事实序贯控制问题，通过单调策略风险模型与配对预算状态延续模型分别校准偏离的局部下行风险和预算消耗的未来代价，并用逐路径累计KL账本约束总偏离量，从而在即时适应价值与安全风险之间实现可校准的权衡。

    

    在行为发生漂移后，对手的历史行为记录可能会产生误导。SafeAdapt-Poker 将对手漂移下的预算化适应问题表述为一个双重反事实序贯控制问题，仅使用每次决策之前可获得的信息。在每次决策时，控制器会提出两个不同的问题：一是在匹配的合法后续情形下，偏离冻结锚点会引发怎样的局部下行风险；二是现在消耗一部分剩余偏离预算会损失多少未来的适应价值。第一个问题由一个关于偏离强度的单调策略风险模型来回答；第二个问题则由一个配对的预算状态延续模型来回答，该模型为当前的预算消耗定价。控制器通过将即时适应价值与这两个量进行权衡，在保持支撑集的候选偏离中进行选择；一条逐路径的累计KL散度账本对相对于锚点的总偏离量进行约束；同时从一组有限的冻结控制器中进行筛选。

    arXiv:2610.20473v1 Announce Type: cross  Abstract: Opponent histories can mislead after behavioral shifts. SafeAdapt-Poker formulates budgeted adaptation under opponent shift as a dual-counterfactual sequential-control problem using only information available before each decision. At each decision the controller asks two distinct questions: what local downside a deviation from the frozen anchor would induce under matched legal continuations, and what future adaptation value would be lost by consuming part of the remaining deviation budget now. The first question is answered by a monotone strategic-risk model over deviation intensity; the second by a paired budget-state continuation model that prices current budget consumption. The controller selects among support-preserving candidate deviations by trading immediate adaptation value against these two quantities, a pathwise cumulative-KL ledger bounds total anchor-relative deviation, and a finite family of frozen controllers is screened 
    
[^184]: HEM-Med：一种面向安全多中心用药推荐的分层智能体记忆框架

    HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation

    [https://arxiv.org/abs/2610.20191](https://arxiv.org/abs/2610.20191)

    HEM-Med提出了一种分层智能体记忆框架，通过将可迁移的跨医院临床经验与医院特异性处方模式解耦，并纳入药物相互作用等多标签安全约束，实现安全的多中心用药推荐。

    

    用药推荐（MR）是临床决策支持的关键环节。现有的用药推荐方法大多针对单中心学习设计，即仅涉及一家独立医院；而在实践中，涉及不同规模医院的多中心用药推荐需求日益增长。在多中心场景下，大型医疗中心的丰富数据可以提升低资源中心的推荐性能。尽管已有一些针对该任务的初步尝试，但仍存在两个关键缺口。缺口一：现有的多中心方法未能将可迁移的跨医院临床经验与医院特异的处方模式解耦，限制了临床可行性和医生的接受度。缺口二：这些方法普遍忽视了用药推荐的多标签安全约束，包括药物-药物相互作用风险和本地处方集可行性，而这些对于安全的真实世界部署至关重要。为解决这些缺口，我们提出了HEM-Med，一种新颖的...

    arXiv:2610.20191v1 Announce Type: cross  Abstract: Medication recommendation (MR) is critical for clinical decision support. While existing MR methods are mostly designed for single-center learning, where only one independent hospital is involved, multi-center MR with hospitals of different scales has become increasingly demanding in practice. In multi center settings, abundant data from large medical centers can improve recommendation performance for low-resource centers. Despite a few preliminary attempts on this task, two critical gaps remain. Gap 1: Existing multi-center methods fail to disentangle transferable cross-hospital clinical experience from hospital-specific prescribing patterns, limiting clinical feasibility and physician acceptance. Gap 2: They commonly overlook multi label safety constraints of MR, including drug-drug interaction risks and local formulary feasibility, which are essential for safe real-world deployment. To address these gaps, we propose HEM-Med, a novel
    
[^185]: VideoHarness-SRSI：面向视频理解框架的结构化递归自我改进

    VideoHarness-SRSI: Structured Recursive Self-Improvement for Video Understanding Harness

    [https://arxiv.org/abs/2610.19424](https://arxiv.org/abs/2610.19424)

    该论文提出结构化递归自我改进框架VideoHarness-SRSI，通过对4000多个演化候选的分析识别出验证过拟合、搜索预算稀释和早期经验陷阱三种失败模式及其共同根源，从而实现从有限且嘈杂的反馈中可靠演化出可泛化的视频理解智能体框架。

    

    LLM系统的能力不仅取决于模型参数，还取决于组织信息获取、工具使用、推理和记忆的可执行框架。递归式框架演化可以自动化其设计，但如何从有限、嘈杂且被反复复用的反馈中可靠地发现可泛化的系统仍不清楚。我们在智能体视频理解场景中研究这一问题，其中丰富的多模态证据和相互耦合的机制使得框架优化极具挑战性。我们首先提出了VHarness-BRSI，这是一个围绕冻结的视觉语言模型对视频框架进行递归自我改进的框架。通过分析超过4,000个演化候选，我们发现其与手工设计的系统相比具有竞争力，但同时识别出三种相互关联的失败模式：验证过拟合、搜索预算稀释和早期经验陷阱。我们将这三者追溯到一个共同的根源：相同的不完善验证证据驱动……

    arXiv:2610.19424v2 Announce Type: replace-cross  Abstract: LLM system capabilities depend not only on model parameters but also on the executable harnesses that organize information acquisition, tool use, reasoning, and memory. Recursive harness evolution can automate their design, but how it can reliably discover generalizable systems from limited, noisy, and repeatedly reused feedback remains unclear. We study this question in agentic video understanding, where rich multimodal evidence and coupled mechanisms make harness optimization challenging. We first introduce VHarness-BRSI, a recursive self-improvement framework that evolves video harnesses around a frozen vision-language model. Analyzing over 4,000 evolved candidates, we find it competitive with hand-engineered systems but identify three interconnected failure modes: validation overfitting, search-budget dilution, and the early-experience trap. We trace all three to a common root: the same imperfect validation evidence drives 
    
[^186]: 快手摘要注意力 V2 技术报告

    Kwai Summary Attention V2 Technical Report

    [https://arxiv.org/abs/2610.18654](https://arxiv.org/abs/2610.18654)

    提出KSA V2架构，通过允许不同层设置不同的摘要压缩分块窗口，实现更灵活的KV缓存内存压缩，并解决全注意力层中摘要标记带来的额外计算开销问题。

    

    arXiv:2610.18654v1 公告类型：cross  摘要：诸如快手摘要注意力（KSA）之类的摘要标记架构，将远距离上下文压缩为每个分块中可学习的摘要标记，混合变体则将这些层与全注意力层交错排列，以保留对原始文本的精确访问。在这种混合 KSA-Full 架构中，交错插入的摘要标记在不同层之间充当流式传输的角色。尽管有效，但它也引入了一些瓶颈：(1) 不同层的摘要压缩分块必须相同，因此难以实现灵活的 KV 缓存压缩，例如无法为不同层设置不同的摘要压缩分块窗口。(2) 在全注意力层中，摘要标记引入了额外的计算成本，但该层却忽略它们，仍然支付二次方复杂度的预填充开销，因此对计算端不友好。受此启发，我们开发了 KSA V2，它不仅实现了更灵活的 KV 缓存内存压缩……

    arXiv:2610.18654v1 Announce Type: cross  Abstract: Summary-token architectures such as Kwai Summary Attention (KSA) compress distant context into learnable Summary token per chunk, and hybrid variants interleave these layers with full attention to retain exact access to the original text. In such hybrid KSA-Full architecture, the interleaving inserted summary tokens are served as a streaming flow role across different layers. Although effective, however, it also introduce some bottlenecks: (1) the different layers summary compression chunk must be same, therefore hard to reach a flexible KV Cache compression, e.g., different layer set different summary compression chunk window. (2) in the full-attention layers, the summary token introduced additional computation cost, but such layer ignore them and still pay quadratic prefilling, therefore unfriendly to the computation side. Motivated by them, we developed the KSA V2, which not only achieves a more flexible KV cache memory compression 
    
[^187]: 重新思考大语言模型路由中的质量：安全条件下的模型与预算选择

    Rethinking Quality in Large Language Model Routing: Security-Conditioned Model-and-Budget Selection

    [https://arxiv.org/abs/2610.18566](https://arxiv.org/abs/2610.18566)

    该论文提出SCRouter路由器，针对混合工作负载分别预测效用与安全性并按查询动态加权平衡，实现安全条件下的模型与token预算联合选择。

    

    大语言模型（LLM）路由器通过为每个查询选择调用哪个模型以及授予该模型多少输出token，来降低服务成本。现有路由器使用单一质量指标对“模型-预算”动作进行排序，这不适用于混合工作负载：对于普通查询，好的回答应当正确且有用（效用）；而对于安全敏感的查询，好的回答应当拒绝有害请求或抵抗恶意指令（安全）。这两个目标偏好不同的模型和预算。我们提出了SCRouter，一个能够学习何时优先考虑安全以及如何分配token预算的路由器。对于每个“模型-预算”组合，它估计回答在预算内完成的概率，并利用该概率分别预测效用与安全性。一个小型探针（probe）为每个查询确定这两个预测的加权方式。随后，SCRouter通过权衡加权后的效用与安全性来选择模型和预算。

    arXiv:2610.18566v1 Announce Type: new  Abstract: Large language model (LLM) routers reduce serving cost by choosing which model to call and how many output tokens to grant it for each query. Existing routers rank model-budget actions using a single quality metric. This does not work for mixed workloads. For ordinary queries, a good answer is correct and useful (utility); for security-sensitive queries, a good answer refuses harmful requests or resists malicious instructions (security). These two objectives prefer different models and budgets.   We propose SCRouter, a router that learns when to prioritize security and how to allocate the token budget. For each model-budget pair, it estimates the probability that the response finishes within budget. It uses this probability to predict utility and security separately. A small probe determines how to weight these two predictions for each query. SCRouter then selects the model and budget by balancing the weighted utility and security agains
    
[^188]: Agentic-MatriBO：面向材料与化学设计的能力编译型智能体贝叶斯优化

    Agentic-MatriBO: Capability-Compiled Agentic Bayesian Optimization for Materials and Chemical Design

    [https://arxiv.org/abs/2610.17832](https://arxiv.org/abs/2610.17832)

    提出了能力编译型智能体贝叶斯优化框架Agentic-MatriBO，将BO组件表示为可组合的能力，由大语言模型策略提出优化配方，经编译器检查和确定性执行层验证后用于材料与化学设计。

    

    材料与化学优化必须在有限预算下，基于序列化观测来选择下一次合成、表征或高保真仿真。大语言模型能够利用任务知识提出调整方案，但组合代理模型、采集函数与搜索策略时，会受到兼容性、参数依赖性和证据要求的限制。我们提出了Agentic-MatriBO，这是一个能力编译型的智能体贝叶斯优化（BO）框架，它将BO组件和选择条件表示为可组合的能力。语言策略提出BO配方和候选计划；编译器根据注册的能力目录对提案进行检查；确定性执行层负责验证并提交候选方案。Oracle观测结果被存储在研究记忆中，并为后续轮次提供信息支持。我们在四个离线有限候选池——COF、PCE10、Buch……上与八个基线方法进行了比较。

    arXiv:2610.17832v1 Announce Type: cross  Abstract: Materials and chemical optimization must choose the next synthesis, characterization, or high-fidelity simulation from sequential observations under a limited budget. Large language models can use task knowledge to propose adjustments, but composing surrogates, acquisition functions, and search strategies is constrained by compatibility, parameter dependencies, and evidence requirements. We present Agentic-MatriBO, a capability-compiled agentic Bayesian optimization (BO) framework that represents BO components and selection conditions as composable capabilities. A language policy proposes BO recipes and candidate plans; a compiler checks proposals against the registered capability catalog; and a deterministic execution layer validates and commits candidates. Oracle observations are stored in research memory and inform later rounds. We compare Agentic-MatriBO with eight baselines on four offline finite candidate pools---COF, PCE10, Buch
    
[^189]: LeanPRM：基于Lean的自然语言证明验证过程监督

    LeanPRM: Lean-Grounded Process Supervision for Natural Language Proof Verification

    [https://arxiv.org/abs/2610.17697](https://arxiv.org/abs/2610.17697)

    LeanPRM利用Lean证明助手自动为自然语言数学证明生成步骤级正确性标签，无需人工标注即可构建包含10,634个证明、15.2万个标注步骤的过程监督数据集，解决了证明验证器训练中标签获取困难的瓶颈。

    

    如今，AI系统能够撰写冗长且困难的数学证明，而逐步验证其有效性正成为瓶颈。学习型验证器可以低成本地检查证明，但训练这样的验证器需要对每一步进行正确性标注，而这类标签对于开放式证明来说很难获得。现有方法要么依赖人工标注者（成本高昂），要么从最终答案中推导标签（而证明本身没有最终答案），且用Lean形式化每个证明需要专家投入大量精力。我们提出了LeanPRM，这是一个利用Lean证明助手为自然语言证明创建步骤级标签的框架。一个模型在没有Lean反馈的情况下，同时撰写自然语言证明及其配对的形式化证明；随后Lean对每一步进行标注，再由第二个模型过滤掉仅由形式化语法导致的失败，使负标签只标记数学本身存在的错误。LeanPRM包含10,634个证明、152K个带标签的步骤，且完全无需人工标注。

    arXiv:2610.17697v1 Announce Type: cross  Abstract: AI systems now write long and difficult mathematical proofs, and checking their step-by-step validity is becoming a bottleneck. A learned verifier could check proofs cheaply, but training one requires a correctness label on every step, and such labels are hard to obtain for open-ended proofs. Existing methods take them from human annotators, which is costly, or derive them from final answers, which proofs do not have, and formalizing each proof in Lean requires expert effort. We introduce LeanPRM, a framework that uses the Lean proof assistant to create step-level labels for natural-language proofs. A model writes a natural-language proof together with a paired formal proof, without feedback from Lean. Lean then labels every step, and a second model discards failures caused only by the formal syntax, so that the negative labels mark errors in the mathematics. LeanPRM contains 10,634 proofs with 152K labeled steps and no human annotatio
    
[^190]: RoTA-Bench：通过交互式投影对齐评估三维空间推理

    RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment

    [https://arxiv.org/abs/2610.17665](https://arxiv.org/abs/2610.17665)

    RoTA-Bench是一个通过交互式三维到二维投影对齐任务来评估视觉语言模型三维空间推理能力的新基准，其自动化流水线保证了任务的可解性，并揭示了当前最强模型（GPT-5.6-Sol仅17%成功率）与人类（100%成功率）之间的巨大差距。

    

    视觉语言模型能否将二维观测转化为支持精确动作的三维理解？这需要推断物体几何结构、预判旋转的效果，并通过视觉反馈来修正动作。我们提出RoTA-Bench，通过交互式的三维到二维投影对齐来评估这些能力。给定一个目标轮廓，模型观察一个经过旋转的物体，并在有限的动作预算内选择连续的旋转操作，以恢复出目标投影。我们的自动化流水线从二维图标构建保持轮廓的三维物体，并采用可逆初始化来保证任务可解性，共生成674个任务以及一个分层的100任务测试集。实验结果表明：(i) 人类与模型之间仍存在显著差距：GPT-5.6-Sol仅在17%的任务上取得成功，而20名本科生参与者共同完成了四轮测试集，成功率达到100%；(ii) 图像空间与姿态空间中的进展可以……（原文摘要在此处被截断）

    arXiv:2610.17665v2 Announce Type: new  Abstract: Can vision-language models turn 2D observations into 3D understanding that supports precise action? This requires inferring object geometry, anticipating the effects of rotation, and refining actions through visual feedback. We introduce RoTA-Bench to evaluate these abilities through interactive 3D-to-2D projection alignment. Given a target silhouette, a model observes a rotated object and chooses successive rotations to recover the target projection within a limited action budget. Our automated pipeline constructs silhouette-preserving 3D objects from 2D icons and uses reversible initialization to guarantee solvability, yielding 674 tasks and a stratified 100-task test set. Empirically, we find that (i) a substantial human-model gap persists: GPT-5.6-Sol succeeds on only 17% of tasks, whereas 20 undergraduate participants collectively complete the test set four times with a 100% success rate; (ii) image-space and pose-space progress can
    
[^191]: 印记注入：参数化语言模型记忆中的时间有效性研究

    Engram Injection: A Study of Temporal Validity in Parametric Language-Model Memory

    [https://arxiv.org/abs/2610.15882](https://arxiv.org/abs/2610.15882)

    本文提出Engram Injection方法，将事实存储于LoRA适配器并通过FiLM条件化向冻结的语言模型注入时间信息，但严格排除测试时间样本的评估显示，该参数化记忆方法在事实更新后无法正确回答任何时间敏感问题对，揭示了参数化记忆在处理时间有效性上的局限。

    

    当用户更改偏好或某个事件取代先前的事件时，已存储的事实可能会失效。我们研究语言模型能否在其参数中学习这些变化，并仅凭问题和当前时间正确作答。Engram Injection（印记注入）将事实存储在LoRA适配器中，并通过FiLM条件化在冻结的Qwen2.5-3B-Instruct模型上提供时钟信息。我们将其与加性时钟条件化、基于完整带日期历史记录的提示方法以及检索四条带日期记录的方法进行比较。在修复了一个可能将测试时间样本混入训练的采样器之后，我们在三个随机种子上训练了六个匹配的适配器。训练审计报告显示，在评估的127,760个训练输入中不存在此类样本。在更新前后提出的62对问题上，在训练中被排除的七天经过时间下，在保持顺序代码1固定的主要盲评助手审查中，两种适配器方法对全部62对问题均未能正确回答。

    arXiv:2610.15882v1 Announce Type: new  Abstract: A stored fact may stop being valid when a user changes a preference or an event replaces an earlier one. We study whether a language model can learn these changes in its parameters and answer correctly from only a question and the current time. Engram Injection stores facts in LoRA adapters and supplies the clock through FiLM conditioning on a frozen Qwen2.5-3B-Instruct model. We compare it with additive clock conditioning, prompting with the full dated history, and retrieving four dated records. We train six matched adapters across three seeds after fixing a sampler that could admit test-age examples into training. The training audit reports no such examples among 127,760 evaluated training inputs. On 62 pairs of questions asked before and after an update, at an elapsed age of seven days excluded from training, both adapter methods answer zero pairs correctly under the primary blinded assistant review with order code 1 held fixed. Full-
    
[^192]: MemeLogos：面向有害迷因检测的知识驱动检索增强推理框架

    MemeLogos: Knowledge-Driven Retrieval-Augmented Reasoning for Harmful Meme Detection

    [https://arxiv.org/abs/2610.15463](https://arxiv.org/abs/2610.15463)

    提出MemeLogos框架，通过将有害迷因案例中的危害性模式与推理结构抽象为可复用的知识单元并构建专门知识库，实现知识驱动的检索增强推理，同时提升有害迷因检测的性能与可解释性。

    

    随着社交媒体的迅速普及，有害迷因（meme）已成为对在线生态系统日益增长的威胁。其危害性往往通过视觉、文本和上下文线索之间的复杂交互以隐含方式传达，这使得准确检测变得具有挑战性。近期研究将检索增强生成（RAG）和外部知识引入有害迷因检测领域。然而，现有方法主要检索相似实例或辅助信息，这限制了其捕获可复用危害性知识和底层推理机制的能力，从而制约了检测性能和可解释性。为应对这些挑战，我们提出了MemeLogos，一个用于有害迷因检测的知识驱动检索增强推理框架。MemeLogos将从多样化有害案例中反复出现的危害性模式和推理结构抽象为可复用的知识单元，并将其组织成一个专门的知识库（原文在此处被截断）。

    arXiv:2610.15463v1 Announce Type: new  Abstract: With the rapid proliferation of social media, harmful memes have become a growing threat to online ecosystems. Their harmfulness is often implicitly conveyed through complex interactions among visual, textual, and contextual cues, making accurate detection challenging. Recent studies have introduced Retrieval-Augmented Generation (RAG) and external knowledge into harmful meme detection. However, existing methods mainly retrieve similar instances or auxiliary information, limiting their ability to capture reusable harmfulness knowledge and underlying reasoning mechanisms, and thus constraining both detection performance and interpretability. To address these challenges, we propose MemeLogos, a knowledge-driven retrieval-augmented reasoning framework for harmful meme detection. MemeLogos abstracts recurring harmfulness patterns and reasoning structures from diverse harmful cases into reusable knowledge units and organizes them into a dedic
    
[^193]: 多模态基准真的有用吗？基于结构-响应协同校准的条目级多模态基准诊断

    Are Multimodal Benchmarks Really Useful? Item-Level Multimodal Benchmark Diagnosis via Structure-Response Co-Calibration

    [https://arxiv.org/abs/2610.14542](https://arxiv.org/abs/2610.14542)

    提出了SRCoD框架，通过结构-响应协同校准对多模态基准进行条目级诊断，识别仅凭单一模态或选项规律即可作答的捷径条目，从而更全面地衡量基准能否真正反映模型的跨模态推理能力。

    

    多模态基准是评估模型能否对来自多种模态的信息进行推理的核心工具。然而，现有研究已经揭示，一些基准可能包含“捷径条目”，这些条目仅凭单一模态或答案选项的规律即可作答，因而无法反映模型真正的跨模态推理能力。现有的基准评估方法通常要么依赖条目的内在结构，要么依赖模型的外在响应行为，二者均不足以实现全面的诊断。为填补这一空白，我们提出了SRCoD，一个结构-响应协同校准的诊断框架，用于对多模态基准中的模态依赖性进行条目级诊断。SRCoD从基准内容和目标答案中估计样本级的多模态信息结构，并将该信号作为模态条件条目响应建模的软结构锚点。学习到的模型产生校准后的条目级诊断信号……

    arXiv:2610.14542v1 Announce Type: new  Abstract: Multimodal benchmarks are central to evaluating whether models can reason over information from multiple modalities. However, existing literature has revealed that some benchmarks may contain shortcut items that can be answered from a single modality or from answer-option regularities alone, thus failing to reflect models' true cross-modal inference ability. Existing methods for assessing benchmarks typically rely either on intrinsic item structure or extrinsic model response behavior, which are not adequate for a comprehensive diagnosis. To fill in this gap, we propose SRCoD, a structure-response co-calibrated diagnosis framework for item-level diagnosis of modality dependence in multimodal benchmarks. SRCoD estimates sample-wise multimodal information structure from benchmark content and target answers, and uses this signal as a soft structural anchor for modality-conditioned item response modeling. The learned model produces a calibra
    
[^194]: RoboHarness：一个简单的机器人操控框架超越VLA与世界动作模型

    RoboHarness: A Simple Harness Outperforms VLA and World Actions Models

    [https://arxiv.org/abs/2610.14018](https://arxiv.org/abs/2610.14018)

    RoboHarness为一个简单的机器人操控框架，赋予LLM智能体视觉-几何控制面板，使其无需专门训练VLA模型即可直接理解并调用具身任务，并在BEHAVIOR Challenge 2025上超越最先进的VLA模型。

    

    大语言模型（LLM）的兴起为通用具身智能开辟了一条道路，同时也引出一个问题：LLM如何在物理世界中行动。一种直接的做法是训练模型生成动作：收集机器人动作数据与视频演示，在LLM架构中加入动作头，并大规模训练视觉-语言-动作（VLA）模型。然而，我们认为编码智能体有潜力实现通用具身操作，它们所需的只是一个简单而有效的机器人操控框架。我们提出了RoboHarness：一个为LLM智能体提供视觉-几何控制面板的机器人框架，使其能够直接理解并调用具身任务。借助RoboHarness，Qwen3.8-Flash-Next在BEHAVIOR Challenge 2025任务上超越了最先进的VLA模型。相关实现将以开源形式发布。

    arXiv:2610.14018v1 Announce Type: cross  Abstract: The rise of large language models (LLMs) has opened a path toward generalist embodied AI, raising the question of how LLMs can act in the physical world. One straightforward approach is training the model to generate actions: collect robot-action data and video demonstration, refine LLM architecture with an action head, and train a vision-language-action (VLA) model at scale. However, we argue that coding agents have the potential to realize generalist embodied manipulation. All they need is a simple yet effective robot harness. We present RoboHarness: a robotic harness that gives LLM agents a visual-geometric control panel so that they can directly understand and invoke embodied tasks. With RoboHarness, Qwen3.8-Flash-Next outperforms state-of-the-art VLA models on BEHAVIOR Challenge 2025 tasks. The implementation will be released as open source.
    
[^195]: SelectArena：面向真实世界表格数据的现代特征选择基准测试

    SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data

    [https://arxiv.org/abs/2610.13644](https://arxiv.org/abs/2610.13644)

    该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。

    

    从表格数据集中选择特征子集（列）是现实应用中普遍存在的挑战。行业专家希望最小化模型训练成本，气候科学家需要可解释的因素子集来理解极端天气事件，而临床医生由于财务和时间限制需要在各类诊断测试中做出选择。当从业者查阅科学文献以确定应使用哪种特征选择方法时，他们面对的是一个缺乏代表性的基准测试现状。尽管相关研究众多，现有的特征选择基准测试所能提供的指导却十分有限，因为它们依赖于过时的模型、临时设计的评估协议，或无法反映真实问题的人工生成数据。为了弥合科学基准测试与现实应用之间的差距，我们推出了 SelectArena，一个现代化的特征选择基准测试。我们比较了15种广泛使用的特征选择方法……

    arXiv:2610.13644v1 Announce Type: cross  Abstract: Selecting a subset of features (columns) from a tabular dataset is a ubiquitous challenge in real-world applications. Industry experts seek to minimize model training costs, climate scientists require an interpretable subset of factors to understand extreme weather events, and clinicians need to choose among diagnostic tests due to financial and time constraints. When looking at the scientific literature to determine which feature selection method to use, practitioners across domains face an unrepresentative landscape of benchmarking. Despite extensive research, existing feature selection benchmarks provide limited guidance because they rely on outdated models, ad hoc evaluation protocols, or artificially generated data that do not reflect real-world problems. To close the gap between scientific benchmarking and real-world applications, we introduce SelectArena, a modern benchmark for feature selection. We compare 15 widely used featur
    
[^196]: 仅凭安全动作并不能确保智能体安全：利用守护模型识别未履行的义务

    Safe Actions Alone Do Not Ensure Safe Agents: Identifying Unfulfilled Obligations with Guard Models

    [https://arxiv.org/abs/2610.11773](https://arxiv.org/abs/2610.11773)

    本文首次指出智能体安全不仅需要识别禁止动作，还需识别未履行的安全义务，并推出了首个用于评估守护模型义务识别能力的基准 ObligationBench。

    

    守护模型越来越多地被用于保障基于大语言模型（LLM）的智能体的安全，其主要方式是识别智能体被禁止执行的动作。然而，仅识别禁止动作并不足以确保智能体的安全。在本文中，我们提出智能体的安全还取决于识别那些必需但尚未执行的安全关键动作，我们将其称为“义务”。我们对一个流行的安全评估基准进行的初步研究表明，GLM-5.3 的轨迹中有 56.92% 包含未履行的义务，而仅有 30.00% 包含禁止动作。这一发现揭示了未履行的义务是一个重大且此前被忽视的安全风险来源。然而，据我们所知，目前尚无基准评估守护模型识别这些义务的能力。为填补这一空白，我们提出了 ObligationBench，这是首个用于评估义务识别能力的基准，包含 240 条经专家验证的轨迹……（原文摘要到此被截断）

    arXiv:2610.11773v1 Announce Type: new  Abstract: Guard models are increasingly used to safeguard LLM-based agents, primarily by identifying actions that agents are forbidden to perform. However, identifying forbidden actions alone is insufficient to ensure agent safety. In this paper, we argue that agent safety also depends on identifying required yet unperformed safety-critical actions, which we call obligations. Our preliminary study on a popular benchmark for evaluating safety shows that 56.92% of GLM-5.3 trajectories contain unfulfilled obligations, compared with only 30.00% containing forbidden actions. This finding reveals unfulfilled obligations as a major and previously overlooked source of safety risk. However, to our knowledge, no existing benchmark evaluates whether guard models can identify these obligations. To close this gap, we introduce ObligationBench, the first benchmark for evaluating the capability of obligation identification, comprising 240 expert-validated trajec
    
[^197]: MultiWorldBench：独立控制的视角能否描述同一个共享世界？

    MultiWorldBench: Do Independently Controlled Views Describe One Shared World?

    [https://arxiv.org/abs/2610.11723](https://arxiv.org/abs/2610.11723)

    该论文提出了MultiWorldBench——一个基于Minecraft、包含495个案例配置和十种能力的多人世界模型诊断基准，用于评估独立控制视角是否能保持与共享持久世界的一致性，并发现现有生成式系统（Gamma-World、Solaris、MineWorld）与真实引擎参考（Engine GT）之间仍存在巨大差距，尤其在状态持久性和结构推理方面表现薄弱。

    

    多人世界模型必须确保独立控制的各个视角与一个共享且持久的世界保持一致。我们提出了MultiWorldBench，这是一个诊断性Minecraft基准测试，包含495个案例配置，涵盖七个任务套件和十种能力，包括独立控制、跨视角运动、共享状态同步、持久性、结构推理、并发交互和延迟重访。我们评估了Solaris、Gamma-World和MineWorld，并使用Engine GT作为参考。在生成式系统中，Gamma-World在十项能力上的平均分最高，为21.39，其次是Solaris为20.88，MineWorld为1.89，而Engine GT达到91.69。Gamma-World在多项控制、共享状态和重访能力上表现更佳，而Solaris在跨视角运动和竞态条件一致性方面领先。然而，所有生成式系统在状态持久性方面得分最高仅为8.00，在结构方面得分为1.33。

    arXiv:2610.11723v1 Announce Type: new  Abstract: Multiplayer world models must ensure that independently controlled views remain consistent with one shared and persistent world. We introduce MultiWorldBench, a diagnostic Minecraft benchmark containing 495 case configurations across seven task suites and ten capabilities, including independent control, cross-view motion, shared-state synchronization, persistence, structural reasoning, concurrent interaction, and delayed revisit. We evaluate Solaris, Gamma-World, and MineWorld, using Engine GT as a reference. Gamma-World achieves the highest ten-capability average among the generated systems at 21.39, followed by Solaris at 20.88 and MineWorld at 1.89, while Engine GT reaches 91.69. Gamma-World performs better on several control, shared-state, and revisit capabilities, whereas Solaris leads in cross-view motion and race-condition consistency. Nevertheless, all generated systems score at most 8.00 on state persistence and 1.33 on structur
    
[^198]: 基于循环原生注意力残差扩展至数万次测试时迭代

    Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals

    [https://arxiv.org/abs/2610.11570](https://arxiv.org/abs/2610.11570)

    提出InfiLoop循环原生残差连接，通过结合内容加权与可学习时间衰减来维护递归状态摘要，使循环Transformer能够扩展到数万次测试时迭代而不出现推理性能退化。

    

    本文论证了循环Transformer需要其自身的残差连接，以防止随迭代次数增长而出现的性能退化。我们观察到，增加循环迭代次数反而可能降低推理准确率：带噪声的状态更新会覆盖正确的中间推论，甚至撤销已完成的解。这使得后续迭代不得不从已经退化的表示中恢复丢失的信息：一旦在较早的循环中出现错误（通常由递归中的长程传播导致），后续循环便难以将其纠正。本文提出了InfiLoop，一种循环原生的残差连接，它能够学习保留哪些过去的计算结果，以及从每次新更新中接受多少信息。InfiLoop将基于内容的加权与可学习的时间衰减相结合，以维护递归状态的滚动摘要。精确的流式递归机制使其持久聚合内存的大小在迭代过程中保持恒定。

    arXiv:2610.11570v1 Announce Type: new  Abstract: In this paper, we argue that looped Transformers need their own residual connections to prevent performance degradation as the number of iterations grows. We observe that increasing loop iterations can reduce reasoning accuracy: noisy state updates overwrite correct intermediate deductions and even undo completed solutions. This leaves subsequent iterations to recover lost information from an already degraded representation: once an error arises in an earlier loop, often as a result of long-range propagation through the recurrence, later loops find it difficult to correct. In this paper, we introduce InfiLoop, a loop-native residual connection that learns which past computations to retain and how much to accept from each new update. InfiLoop combines content-based weighting with learned temporal decay to maintain a running summary of recurrent states. An exact streaming recurrence keeps its persistent aggregation memory constant as the l
    
[^199]: RIT-RAG：利用检索诱导树导航文档语料库

    RIT-RAG: Navigating Document Corpora with Retrieval-Induced Trees

    [https://arxiv.org/abs/2610.11370](https://arxiv.org/abs/2610.11370)

    RIT-RAG将内容检索与文档结构导航相结合，离线为每篇文档构建目录树，查询时先检索宽泛的文本块并利用其位置诱导出可跨多个文档的、规模可控的子树，再由LLM智能体在子树中导航，从而在保留结构感知优势的同时克服了现有方法无法扩展到大型语料库且选错文档后无法恢复的缺陷。

    

    检索增强生成（RAG）将语言模型建立在外部语料库之上。智能体式RAG（Agentic RAG）支持迭代式搜索，但使模型只能接触到缺乏文档结构的孤立文本块，难以区分真正相关的证据与仅在表面上与查询相似的文本块。诸如PageIndex之类的结构感知方法虽然能够导航文档结构，但无法扩展到大型语料库的结构，因为这些结构无法容纳于LLM的上下文中。因此，这类方法首先需要通过文档检索器确定单一文档，一旦选择错误便无法恢复。我们提出RIT-RAG（检索诱导树RAG），将内容检索与结构导航相结合。在离线阶段，RIT-RAG根据每个文档的目录或站点地图为其构建一棵树。在查询时，它检索一组宽泛的文本块，并利用这些文本块的位置来诱导出规模可控的子树，这些子树可能跨越多个文档。LLM智能体在这些子树中进行导航，有选择地……（原文摘要在此处截断）

    arXiv:2610.11370v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) grounds language models in external corpora. Agentic RAG enables iterative search, yet exposes the model to isolated chunks without document structure, making it difficult to distinguish relevant evidence from chunks that merely resemble the query. Structure-aware methods such as PageIndex navigate document structure but cannot scale to the structures of large corpora, which do not fit in the LLM context. Hence, they first commit to a single document using a document retriever and cannot recover from a wrong choice. We propose RIT-RAG (Retrieval-Induced Tree RAG), which combines content retrieval with structural navigation. Offline, RIT-RAG builds a tree for each document from its table of contents or sitemap. At query time, it retrieves a broad set of chunks and uses their positions to induce manageable sub-trees, potentially across multiple documents. An LLM agent navigates these sub-trees, selectiv
    
[^200]: 冻结模型、进化专长：面向多模态医疗AI的模型无关部署经验学习

    Frozen Models, Evolving Expertise: Model-Agnostic Learning from Deployment Experience for Multimodal Medical AI

    [https://arxiv.org/abs/2610.09146](https://arxiv.org/abs/2610.09146)

    提出了一种模型无关的框架，使冻结的大语言模型和视觉语言模型无需微调即可通过技能、知识记忆库和多模态知识库三种外部专业知识从部署经验中持续学习，以适应医学领域不断更新的临床知识。

    

    arXiv:2610.09146v2 公告类型：替换 摘要：大语言模型（LLM）和视觉语言模型（VLM）在部署后通常被冻结，因此无法从其处理的案例中学习。这在医学领域尤为令人担忧，因为新的临床证据、更新的指南和新疗法可能会改变既定的临床实践。微调可以更新模型，但需要访问模型权重并进行额外训练。无参数方法虽避免了训练，但可能过拟合固定的验证集、缺乏可靠的领域知识，或因仅以文本形式保存经验而丢失视觉细节。为解决这些局限性，我们提出了一个模型无关的框架，使冻结的LLM和VLM能够通过三种形式的外部专业知识从部署经验中学习：引导推理和工具使用的技能、存储由早期案例或可信外部证据支持的可靠事实的知识记忆库，以及保存视觉示例的多模态知识库……

    arXiv:2610.09146v2 Announce Type: replace  Abstract: Large language models (LLMs) and vision-language models (VLMs) are usually frozen after deployment, so they do not learn from the cases they solve. This is especially concerning in medicine, where new clinical evidence, updated guidelines, and new therapies can change established practice. Fine-tuning can update the model, but it requires access to model weights and additional training. Parameter-free methods avoid training, but they may overfit a fixed validation set, lack reliable domain knowledge, or lose visual details by saving experience only as text. To address these limitations, we present a model-agnostic framework that allows frozen LLMs and VLMs to learn from deployment experience through three forms of external expertise: a Skill that guides reasoning and tool use, a Knowledge Memory that stores reliable facts supported by earlier cases or trusted external evidence, and a Multimodal Knowledge Base that keeps visual exampl
    
[^201]: 递归游戏创造者：一个面向体验的智能体级产品游戏开发框架

    Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness

    [https://arxiv.org/abs/2610.08621](https://arxiv.org/abs/2610.08621)

    该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。

    

    近期的游戏设计智能体在生成可玩游戏方面取得了长足进步。然而，程序的正确性并不能保证玩家获得愉快的游戏体验。我们提出了递归游戏创造者，这是一个面向体验的框架，旨在将智能体游戏开发从粗糙的游戏原型推进为有趣的游戏。递归游戏创造者围绕四个组件组织递归式开发：设计者、构建者、玩家和评审者。设计者将用户指令和评审者的反馈转化为详细的计划。构建者将这些计划转化为候选游戏。基于代码原生的玩家通过编程接口创建并执行可复用的策略，以高效收集多样化的游戏玩法轨迹，缓解了基于图形界面（GUI）的缓慢收集方式所导致的评估偏差。评审者使用精心设计的基于轨迹的指标来推断玩家偏好，并结合视觉证据和明确的文本偏好（进行综合评估）。

    arXiv:2610.08621v1 Announce Type: new  Abstract: Recent game design agents have made substantial progress in generating playable games. However, program correctness does not ensure an enjoyable experience for players. We present Recursive Game Creator, an experience-oriented harness to advance agentic game development from rough game prototypes into entertaining games. Recursive Game Creator organizes recursive development around four components: Designer, Builder, Player, and Reviewer. The Designer translates user instructions and Reviewer's feedback into detailed plans. The Builder turns these plans into candidate games. The coding-native Player creates and executes reusable policies through programmatic interfaces to efficiently collect diverse gameplay trajectories, mitigating evaluation bias caused by slow GUI-based collection. The Reviewer uses carefully designed trajectory-based metrics to induce player preferences, integrating with visual evidence and explicit textual preferenc
    
[^202]: EvalResearchBench：AI智能体能否构建自己的反馈以实现递归自我改进？

    EvalResearchBench: Can AI Agents Build Their Own Feedback for Recursive Self-Improvement?

    [https://arxiv.org/abs/2610.04184](https://arxiv.org/abs/2610.04184)

    提出EvalResearchBench（ERB）这一新基准，用于检验AI智能体能否自主设计评估反馈以支持递归自我改进，研究发现最佳智能体构建的评估器与目标基准的成对排序一致性约为75%，但仍低于人类专家水平。

    

    递归自我改进（RSI）依赖于评估反馈来评估进展并指导后续研究，然而反复运行复杂的基准测试成本高昂且拖慢迭代速度。人类专家通过选择基准测试子集或设计紧凑的测试套件来降低这一成本。我们探讨AI智能体能否自动化这一设计过程，并提出了EvalResearchBench（ERB），一个面向自主评估研究的基准。在给定目标材料、开发参考资料、候选API以及固定时间和API预算的条件下，一个被称为“研究员”的智能体需要选择或合成任务、实现评分器，并在试点测试中进行修订，最终冻结出一个可执行的评估器，用于编程、协作和推理任务。我们研究了9个研究员和13个候选模型，并将每个冻结的评估器与14个目标基准在分数一致性和成对一致性上进行比较。最好的评估器能够以与目标基准相同的方式对约75%的候选对进行排序，低于91%的（原文在此处截断）。

    arXiv:2610.04184v2 Announce Type: replace  Abstract: Recursive self-improvement (RSI) relies on evaluation feedback to assess progress and guide further research, yet repeatedly running complex benchmarks is costly and slows iteration. Human experts reduce this cost by selecting benchmark subsets or designing compact suites. We ask whether AI agents can automate this design process and introduce EvalResearchBench (ERB), a benchmark for autonomous evaluation research. Given target materials, development references, candidate APIs, and fixed time and API budgets, an agent called the researcher selects or synthesizes tasks, implements graders, and revises them in pilot tests before freezing an executable evaluator for coding, co-work, and reasoning. We study 9 researchers and 13 candidate models and compare each frozen evaluator with 14 target benchmarks on score concordance and pairwise agreement. The best evaluators order about 75\% of candidate pairs as the targets do, below the 91\% c
    
[^203]: 分数并非结构：大脑对齐与跨语言迁移

    The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer

    [https://arxiv.org/abs/2610.03827](https://arxiv.org/abs/2610.03827)

    相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。

    

    相似性分数常被用作模型与大脑或跨语言间共享结构的证据。我们探讨当结构被移除或测量工具失效时，这样的分数实际反映的是什么，并在两个场景中进行了验证。在十七种语言中，语法性探针在语言距离越远时迁移效果越差（r = -0.66），但探针自身的准确率也沿同一轴线下降，在四种语言中处于随机水平（这四种语言是距离最远的五种中的四种）。剔除这些语言会使已解释方差减半，同时由于也缩小了距离范围，该设计无法判断梯度中有多少来自测量工具本身。将272个语言对视为独立样本时，导向效应的p值为0.0006；但以十七种语言为单位时，该效应不显著（p = 0.155）。在大脑对齐方面，一个训练目标将语言模型与fMRI响应的相似性从0.10提升到0.34（可靠性上限为0.54），然而

    arXiv:2610.03827v2 Announce Type: replace-cross  Abstract: Similarity scores are often offered as evidence that a model shares structure with the brain or across languages. We ask what such a score reads when that structure is removed, or when the instrument measuring it does not work, in two settings. Across seventeen languages, a grammaticality probe transfers worse between more distant languages (r = -0.66). But the probe's own accuracy falls along the same axis and is at chance in four languages, four of the five most distant. Dropping them halves the explained variance, and because it also narrows the range of distances, the design cannot say how much of the gradient is the instrument. Counting the 272 language pairs as independent gives p = 0.0006 for a steering effect that is null when the seventeen languages are the unit (p = 0.155). In brain alignment, a training objective raises a language model's similarity to fMRI responses from 0.10 to 0.34 (reliability ceiling 0.54), yet 
    
[^204]: FSPO：面向预算约束的大语言模型强化学习后训练的策略一致风险与帕累托可行控制

    FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training

    [https://arxiv.org/abs/2610.02828](https://arxiv.org/abs/2610.02828)

    FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。

    

    自适应的大语言模型强化学习后训练会在训练过程中在线调整多个训练执行器，包括 rollout 温度、组大小、裁剪、KL 正则化、验证器分配以及更新预算。目前有三个相互耦合的问题尚未解决：从行为轨迹训练得到的未来风险模型未必能准确估计将要部署的控制器所引发的风险；基于已记录的状态-动作对校准的分数在经过选择性动作选择后可能出现校准失准；独立的单资源最小成本通常无法保证多资源延续的可行性。我们提出 FSPO，一种面向预算约束的大语言模型强化学习后训练的反馈状态控制器，能够联合解决这些问题。FSPO 学习一个策略一致的风险前瞻（risk-to-go）模型，其 Bellman 目标遵循与未来决策所用的同一个冻结控制器，并同时学习一个长程效用模型。决策条件化轨迹校准（DCTC）对风险进行校准……

    arXiv:2610.02828v1 Announce Type: new  Abstract: Adaptive LLM reinforcement-learning post-training changes multiple training actuators online, including rollout temperature, group size, clipping, KL regularization, verifier allocation, and update budget. Three coupled issues remain unresolved. A future-risk model trained from behavior trajectories need not estimate the risk induced by the controller that will be deployed; a score calibrated on logged state-action pairs can become miscalibrated after selective action choice; and independent per-resource minimum costs do not in general certify a feasible multi-resource continuation. We introduce FSPO, a feedback-state controller for budgeted LLM RL post-training that addresses these issues jointly. FSPO learns a policy-consistent risk-to-go model whose Bellman target follows the same frozen controller used for future decisions, together with a long-horizon utility model. Decision-conditioned trajectory calibration (DCTC) calibrates risk 
    
[^205]: ROUTEAUDIT：面向预算受限多验证器路由的交互感知识别方法

    ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

    [https://arxiv.org/abs/2610.02808](https://arxiv.org/abs/2610.02808)

    ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。

    

    自适应多验证器系统通常通过端点的质量-成本差距进行比较，即使验证器目录、可用性、资源核算、信息过滤或评分器会随策略发生变化。我们将验证器路由形式化为一个契约条件化的识别问题。该契约记录了请求支持、验证器目录、实际可用性、资源核算、在线过滤以及轨迹后评分；一个匹配的路由对比仅改变策略坐标。ROUTEAUDIT为该契约增加了三个可度量的对象：契约格在所有可容许的桥接顺序上对坐标增量取平均，并报告由此得到的归因及其路径敏感性；策略无关的响应带在自适应策略揭示不同观测时识别成对的顺序对比；对于不完整的匹配，请求级边界利用仍然可观测的潜在结果，给出紧致的有限（摘要在此处截断）。

    arXiv:2610.02808v1 Announce Type: new  Abstract: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp fi
    
[^206]: 基于国际象棋的大语言模型提示词优化基准测试

    Benchmarking Prompt Optimization of Large Language Models With Chess

    [https://arxiv.org/abs/2610.00416](https://arxiv.org/abs/2610.00416)

    该论文提出了一个基于1,118道Lichess国际象棋棋题的基准测试，利用国际象棋低成本精确评分、引擎辅助着法评估和可持续更新的可调难度题源等特性，来研究冻结大语言模型的自动提示词优化。

    

    随着大语言模型能力的不断提升，对其评估变得越来越具有挑战性：基准测试可能饱和，公开测试集面临数据污染的风险，而评估更难的任务可能需要昂贵的评分或执行基础设施。这些挑战在自动提示词优化（APO）中被进一步放大，因为在搜索更优提示词的过程中需要反复进行评估。因此，研究APO需要一个评分成本低且结果确定、难度足够高以留出改进空间、并随模型演进可持续更新的基准测试。我们引入了一个由1,118道Lichess棋题构建的国际象棋基准测试，用于研究冻结大语言模型（即不更新模型权重、仅优化提示词）的自动提示词优化。国际象棋结合了低成本的精确匹配评分、基于引擎的备选着法评估，以及难度可调且可持续更新的题源。与仅报告孤立测试项成功率的评估不同，该基准测试还……（原文摘要在此处截断）

    arXiv:2610.00416v1 Announce Type: new  Abstract: Evaluating large language models becomes increasingly challenging as their capabilities advance: benchmarks can saturate, public test sets risk contamination, and assessing harder tasks can require expensive grading or execution infrastructure. These challenges are amplified in automatic prompt optimization (APO), where evaluation is repeated throughout the search for better prompts. Studying APO therefore requires a benchmark that is cheap and deterministic to score, hard enough to leave room for improvement, and renewable as models evolve. We introduce a chess benchmark built from 1,118 Lichess puzzles to study APO for frozen LLMs: we optimize their prompts without updating their model weights. Chess combines inexpensive exact-match scoring, engine-based evaluation of alternative moves, and a renewable supply of problems with adjustable difficulty. Unlike evaluations that report only success on isolated test items, the benchmark also c
    
[^207]: FAER：面向语言模型后训练的可审计、效用对齐的轨迹重放

    FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training

    [https://arxiv.org/abs/2610.00385](https://arxiv.org/abs/2610.00385)

    FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。

    

    重放选择器通常依据格式反馈、置信度、新鲜度或响应长度对缓存轨迹进行排序，然而缓存层面的正确性与下游学习器的效用是两个不同的目标。我们形式化了这种“选择到学习”的差距，并提出FAER作为一个可审计的全轨迹重放框架。其中免训练的固定选择器作为协议基线；FAER-UTILITY则是在互不相交的校准块上拟合的学习器感知选择器。归一化梯度对齐被报告为基线方法，而一次性的优化器感知虚拟更新提供了幅度感知的效用面。审计契约会在评估标签加入之前冻结观测字段和重放轨迹。在GSM8K数据集上使用Qwen2.5-1.5B-Instruct模型，匹配学习器的研究显示固定选择器在128次更新下质量达到0.6329，相比之下均匀采样为0.5482，格式反馈为0.6037。仅使用元数据的交叉拟合校准达到0.6...

    arXiv:2610.00385v1 Announce Type: cross  Abstract: Replay selectors often rank cached trajectories by format feedback, confidence, freshness, or response length, although cache-level correctness and downstream learner utility are distinct objectives. We formalize this selection-to-learning gap and introduce FAER as an auditable full-trajectory replay framework. Its training-free fixed selector is a protocol baseline; FAER-UTILITY is the learner-aware selector fitted on disjoint calibration blocks. The normalized gradient alignment is reported as a baseline, while a disposable optimizer-aware virtual update supplies a magnitude-aware utility surface. The audit contract freezes observed fields and replay traces before evaluation labels are joined. On GSM8K with Qwen2.5-1.5B-Instruct, the matched learner study reports quality 0.6329 for the fixed selector, compared with 0.5482 for uniform and 0.6037 for format-feedback under 128 updates. Metadata-only cross-fitted calibration reaches $0.6
    
[^208]: ContractRL：面向可审计工具调用修复的屏蔽式组相对策略优化

    ContractRL: Shielded Group-Relative Policy Optimization for Auditable Tool-Call Repair

    [https://arxiv.org/abs/2610.00328](https://arxiv.org/abs/2610.00328)

    ContractRL提出了一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为有界决策过程，通过契约派生的动作掩码过滤非法修复操作，在仅生成34.4个token的情况下实现了0.9362的语义成功率，并保证了修复过程的可审计性。

    

    结构化工具调用往往在仅有少量字段违反模式或执行契约时就会失败。重新生成完整对象会扩大动作空间，并使重复修复难以审计。我们提出了ContractRL，这是一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为一个有界决策过程。在每一步中，策略观察候选对象、带类型的验证器反馈、JSON Pointer、不可变的修复历史以及剩余预算；契约派生的动作掩码在确定性验证器执行状态转移之前过滤掉格式错误或被禁止的RFC-6902操作。我们为补丁、重试和放弃决策指定了契约约束的组相对目标，同时将规范目标和语义标签保持在在线状态之外，直到轨迹冻结。在相同的验证器信息下，ContractRL仅使用34.4个生成token就达到了0.9362的语义成功率。

    arXiv:2610.00328v1 Announce Type: new  Abstract: Structured tool calls often fail after only a small number of fields violate a schema or an execution contract. Regenerating the complete object enlarges the action surface and makes repeated repair difficult to audit. We introduce ContractRL, a contract-constrained sequential repair protocol that models verifier-guided JSON repair as a bounded decision process. At each step the policy observes the candidate, typed verifier feedback, JSON Pointer, immutable repair history, and remaining budget; a contract-derived action mask filters malformed or prohibited RFC-6902 operations before a deterministic validator performs the transition. We specify a contract-constrained group-relative objective for patch, retry, and abstention decisions while keeping canonical targets and semantic labels outside the online state until trace freeze. Under identical verifier information, ContractRL attains 0.9362 semantic success with 34.4 generated tokens, co
    
[^209]: 带收据的动作：联合绑定声明、证据与执行以实现可重放的工具型智能体审计

    Actions with Receipts: Jointly Binding Claims, Evidence, and Execution for Replayable Tool-Agent Auditing

    [https://arxiv.org/abs/2610.00327](https://arxiv.org/abs/2610.00327)

    该论文提出“声明锚定执行契约”，通过收据机制将智能体发出的声明、证据来源片段、执行前缀与来源版本联合绑定，并由确定性完整性验证器重建绑定关系，从而实现对工具型智能体行为的可重放审计。

    

    使用工具的智能体可以展示引用和执行日志，却留下一个关键关联未经审计：呈现给用户的声明是否就是已提交执行所发出的、且被所引用来源支持的声明。因此，一个有效的引用和一个有效的执行轨迹可以各自形式上完好，却被移植到其他声明、动作、运行或来源版本之上。我们提出了一种以声明为锚点的执行契约，将发出的声明、其精确的来源片段、产生该声明的有序执行前缀，以及该执行所观察到的来源版本和访问状态联合绑定在一起。每份收据包含一个发射锚点，用于确定性地定位该声明在已提交答案或含声明动作中的位置，同时包含来源标识符、偏移量、哈希值、引文以及域分离的执行承诺。一个确定性的完整性验证器会在加入语义或任务标签之前重建这些绑定关系。

    arXiv:2610.00327v1 Announce Type: cross  Abstract: Tool-using agents can expose citations and execution logs while leaving a critical association unaudited: whether the claim shown to a user is the claim emitted by the committed execution and supported by the cited source. A valid citation and a valid trace can therefore remain individually well formed while being transplanted across claims, actions, runs, or source versions. We introduce a claim-anchored execution contract that jointly binds the emitted claim, its exact source span, the ordered execution prefix that produced it, and the source version and access state observed by that execution. Each receipt contains an emission anchor that deterministically locates the claim inside a committed answer or claim-bearing action, together with source identifiers, offsets, hashes, quotes, and a domain-separated execution commitment. A deterministic integrity verifier reconstructs these bindings before semantic or task labels are joined. We
    
[^210]: 信任不是分数：面向高风险AI智能体的运行时保障契约

    Trust Is Not a Score: Runtime Assurance Contracts for High-Risk AI Agents

    [https://arxiv.org/abs/2609.39717](https://arxiv.org/abs/2609.39717)

    提出了运行时保障契约（RAC），用非补偿性强制门控而非综合分数来决定高风险AI智能体在关键任务中的权限动态转换，填补了观察证据与智能体授权之间的保障转换缺口。

    

    基准测试、审计和智能体协议描述了性能、权限和修复机制，但并未说明观察到的证据应如何在重大任务中改变智能体的权限。我们将这一现象称为保障转换缺口。我们提出了运行时保障契约（RAC），这是一种策略级的形式化模式，将自主性边界、组件资格、证据状态、转换策略、人工审查容量以及非补偿性门控绑定在一起。在RAC下，软性指标可以用于路由决策，而失败或状态未知的强制门控则会强制触发重试、切换、升级、延迟或停止；总体性能不能作为行动授权。我们定义了契约、证据记录、权限规则以及五个不变量，并在临床、工业和司法领域的失败探测案例中加以说明。随后我们报告了一项针对智能体编程的确定性故障注入研究：280个构造案例分别由门控合取规则、仅分数规则和受限的…进行评估。

    arXiv:2609.39717v1 Announce Type: new  Abstract: Benchmarks, audits, and agent protocols describe performance, permissions, and repair, but not how observed evidence should change an agent's authority during a consequential task. We call this the assurance-transition gap. We propose a Runtime Assurance Contract (RAC), a policy-level formal schema binding autonomy boundaries, component eligibility, evidence state, transition policy, human-review capacity, and non-compensatory gates. Under RAC, soft metrics may inform routing, whereas a failed or unknown mandatory gate forces retry, switch, escalation, deferral, or stop; aggregate performance cannot authorize action. We define the contract, an evidence record, a permission rule, and five invariants, and illustrate them in clinical, industrial, and judicial failure probes. We then report a deterministic failure-injection study in agentic coding: 280 constructed cases evaluated by a gate conjunction, a score-only rule, and a restricted pro
    
[^211]: GroundingPI：一个基于视觉原语、面向物理智能的视觉定位基础模型

    GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives

    [https://arxiv.org/abs/2609.39601](https://arxiv.org/abs/2609.39601)

    GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。

    

    精确的视觉定位至关重要。它需要指明哪个物体是目标以及该物体位于何处，即使在杂乱场景和微小物体的情况下也必须准确，并且必须足够快速以支持闭环控制。然而，视觉-语言-动作模型（VLA）和世界-动作模型（WAM）通常从通用的视觉-语言骨干网络和视频生成骨干网络获取感知能力，而这些骨干网络在上述场景中仍然表现不佳。我们提出了GroundingPI，一个40亿参数的视觉定位基础模型，它以共享词表中的量化坐标形式生成点和边界框。其训练结合了多模态与空间预训练、监督微调以及基于GRPO的强化学习，并利用公开数据集和专用数据引擎提供监督信号。在与涵盖11种感知能力的34个视觉定位基准上的44个基线模型的对比中，GroundingPI确立了新的最先进水平，平均成绩达到73.68%，超过了参数量更大的GPT-6 Astra（71.54%）。作为下游任务的视觉骨干网络，GroundingPI提升了……（摘要在此处被截断）

    arXiv:2609.39601v1 Announce Type: cross  Abstract: Precise grounding matters. It specifies which object is the target and where that object is, even in clutter and for tiny objects, and it has to be fast enough for closed-loop control. Yet vision-language-action (VLA) and world-action models (WAMs) take perception from general-purpose vision-language and video-generation backbones, which still fail in these settings. We introduce GroundingPI, a 4B grounding foundation model that generates points and boxes as quantized coordinates in a shared vocabulary. Training combines multimodal and spatial pretraining, supervised fine-tuning, and reinforcement learning with GRPO, using supervision from public datasets and dedicated data engines. Against 44 baselines across 34 grounding benchmarks spanning 11 perceptual capabilities, GroundingPI establishes a new state of the art, averaging 73.68%, above the larger GPT-6 Astra (71.54%). As a downstream visual backbone, GroundingPI improves performan
    
[^212]: GroundAnything：以闪速实现并行解码与精准视觉定位的统一

    GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed

    [https://arxiv.org/abs/2609.39600](https://arxiv.org/abs/2609.39600)

    GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。

    

    自回归（AR）视觉定位模型将空间预测序列化，引入了顺序延迟，并对输出 token 施加了因果顺序约束。我们将视觉定位视为视觉证据提取：物体、位置和空间关系由图像和查询共同约束，但它们之间的依赖关系并不意味着存在内在的从左到右的生成顺序。这一区别使得双向扩散成为天然合适的选择，允许空间假设并行涌现，并通过迭代去噪进行联合优化。我们提出了 GroundAnything，一个 40 亿参数的视觉定位基础模型，通过分块去噪实现了快速并行解码与精准定位的统一。其训练过程结合了基于公开数据集与专用数据引擎的定位预训练、采用 AR 与扩散联合目标的直接 AR 到扩散架构转换、监督微调以及基于 GRPO 的强化后训练。在 30 个视觉定位基准测试上……

    arXiv:2609.39600v1 Announce Type: cross  Abstract: Autoregressive (AR) grounding models serialize spatial predictions, introducing sequential latency and imposing a causal order on output tokens. We view grounding as visual evidence extraction: objects, locations, and spatial relations are jointly constrained by the image and query, yet their dependencies do not imply an intrinsic left-to-right generation order. This distinction makes bidirectional diffusion a natural fit, allowing spatial hypotheses to emerge in parallel and be jointly refined through iterative denoising. We introduce GroundAnything, a 4B-parameter grounding foundation model that reconciles fast parallel decoding with precise localization through blockwise denoising. Training combines grounding pretraining from public datasets and dedicated data engines, direct AR-to-diffusion conversion with joint AR and diffusion objectives, supervised fine-tuning, and GRPO-based reinforcement post-training. Across 30 grounding benc
    
[^213]: CollabFlow：智能体协作的递归自我改进

    CollabFlow: Recursive Self-Improvement of Agent Collaboration

    [https://arxiv.org/abs/2609.38662](https://arxiv.org/abs/2609.38662)

    提出CollabFlow，一个通过可训练的“协作导演”构建完整智能体团队、在协作图边上采用证据条件化通信协议以抑制错误传播、并利用每轮成果重新训练导演从而实现递归自我改进的多智能体协作系统。

    

    递归自我改进（RSI）使系统能够从自身的成果中获得提升；在基于大语言模型的多智能体系统中，智能体在任务内部相互完善，而任务成果则改进它们跨任务的协作方式。然而，现有多智能体协作方法使这一循环处于断裂状态：协作在算子层面被预先定义，仅基于拓扑的学习保留了逐字交换从而导致错误传播，而基于系统自身成果的奖励最大化则过度集中于少数几个团队。为了应对这些挑战，我们提出了CollabFlow，一个学习型智能体协作的RSI系统：一个可训练的Collab-Director（协作导演）构建由完整智能体组成的团队，一个冻结的执行器负责运行这些团队，而每一轮的成果则用于重新训练该导演。在每一轮内部，协作图的边承载着证据条件化通信协议：只有当发送方的证据以一定优势强于接收方时，接收方才会采纳不同的答案，从而使导演学会谁……（摘要截断）

    arXiv:2609.38662v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) lets a system improve from its own outcomes; in LLM-based multi-agent systems, Agents refine one another within a task, and outcomes improve how they collaborate across tasks. However, existing multi-agent collaboration leaves this loop open: collaboration is pre-defined at the operator level, topology-only learning keeps verbatim exchange that propagates errors, and reward maximization on a system's own outcomes concentrates on a few teams. To address these challenges, we propose CollabFlow, an RSI system of Learned Agent Collaboration: a trainable Collab-Director constructs teams of complete Agents, a frozen executor runs them, and each round's outcomes retrain the director. Within each round, the edges of a collaboration graph carry protocols of Evidence-Conditioned Communication: a receiver adopts a differing answer only when the sender's evidence is stronger by a margin, so the director learns who 
    
[^214]: 智能体应何时检查外部状态？为存储意图分配观测预算

    When Should Agents Check External State? Budgeting Observations for Stored Intentions

    [https://arxiv.org/abs/2609.37125](https://arxiv.org/abs/2609.37125)

    本文首次将智能体为验证存储意图所需的外部观测形式化为共享预算下的资源分配问题，并提出 BudgetPM 框架（包含静态评分器与序列策略两种变体）来决定何时花费或保留观测预算。

    

    前瞻记忆使智能体能够保留一个与未来条件相关联的意图，但存储的意图并不能揭示该条件当前是否成立。检查该条件可能需要网络访问、多步工具使用和付费调用。现有系统可以决定意图何时需要关注，但并未在共享预算下分配由此产生的观测。我们提出了首个针对共享回合预算下、存储意图所需外部观测的资源分配形式化方法。BudgetPM 提供了两个共享硬预算执行器的策略变体：BudgetPM-Static 使用轻量级 Logistic 评分器来学习一次检查是否有助于改善当前决策；BudgetPM-Sequential 将完整回合的事后调度蒸馏为一个轻量级策略，在部署时仅利用查询前的信息来决定何时花费或保留预算容量。我们针对两个公开的记忆智能体系统和五个匹配的基线方法对 BudgetPM 进行了评估。

    arXiv:2609.37125v1 Announce Type: new  Abstract: Prospective memory allows an agent to retain an intention tied to a future condition, but the stored intention does not reveal whether that condition currently holds. Checking it may require web access, multi-step tool use, and paid calls. Existing systems decide when intentions require attention, but do not allocate the resulting observations under a shared budget. We introduce the first resource-allocation formulation for the external observations required by stored intentions under a shared episode budget. BudgetPM offers two policy variants that share a hard-budget executor. BudgetPM-Static uses a lightweight Logistic scorer to learn whether a check improves the current decision. BudgetPM-Sequential distills full-episode hindsight schedules into a lightweight policy that decides when to spend or reserve capacity using only pre-query information at deployment. We evaluate BudgetPM against two public memory-agent systems, five matched 
    
[^215]: VStress：面向重复验证器的相关性感知审计与自适应预算分配

    VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers

    [https://arxiv.org/abs/2609.36958](https://arxiv.org/abs/2609.36958)

    本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。

    

    重复验证器调用只有在贡献条件信息时才有价值。我们提出了VStress——一种可审计的重放契约，以及VStress-CA——一种相关性感知的分配策略。该策略在密封的校准集上估计未查询验证器的条件边际信息，对不确定性进行折扣，按调用成本归一化，并在下一次调用不具信息量时停止或弃权。控制器在接入干净预言机之前冻结其决策和成本账本；当依赖性偏移警报触发时，禁用通道偏好并回退到精确停止机制。受控审计给出了该机制的适用边界：在35%对称损坏下，多数投票-5将平衡准确率从0.6578提升至0.7739；而在65%损坏下则损失0.1226个百分点。在固定预算匹配比较中，广度、冗余和自适应分配策略分别取得0.6048、0.6375和0.6538的平衡准确率，每项平均调用3.4216次，RLVR分数为0.64。

    arXiv:2609.36958v1 Announce Type: cross  Abstract: Repeated verifier calls are useful only when they contribute conditional information. We introduce VStress, an auditable replay contract, and VStress-CA, a correlation-aware allocation policy that estimates the conditional marginal information of an unqueried verifier on a sealed calibration split, discounts uncertainty, normalizes by call cost, and stops or abstains when the next call is not informative. The controller freezes its decision and cost ledger before joining the clean oracle; a dependence-shift alarm disables channel preference and falls back to exact-stop. The controlled audit gives the mechanism boundary: at 35% symmetric corruption, majority-5 improves balanced accuracy from 0.6578 to 0.7739, whereas at 65% it loses 0.1226 points. In the matched fixed-budget comparison, breadth, redundancy, and adaptive allocation obtain balanced accuracies 0.6048, 0.6375, and 0.6538, with 3.4216 calls per item and an RLVR score of 0.64
    
[^216]: 面向LLM智能体的动作空间塑造：测量与缓解工具模式偏差

    Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias

    [https://arxiv.org/abs/2609.34971](https://arxiv.org/abs/2609.34971)

    该论文揭示了LLM智能体的“模式偏差”现象——即当相同的可执行动作以功能等价但形式不同的工具模式呈现时智能体表现不一致，并提出了一套包含九种算子的可执行转换框架，在保持任务与动作空间不变的前提下系统测量并缓解这种偏差。

    

    大语言模型（LLM）在给定固定工具模式时，在工具使用型智能体任务上表现出强劲性能。然而，工具模式并非智能体的动作空间，它只是动作空间的一种接口表示。同一个可执行动作可以通过许多不同但功能等价的工具定义来呈现，而真正学会了任务的智能体应该在这些不同的表示下保持一致的行为。我们证明当前的智能体往往做不到这一点，并将这一现象称为“模式偏差”。为了系统地研究这一问题，我们引入了一个可执行的转换框架，使用九种算子来重写原生工具模式，包括合并与拆分工具、改变单个工具的表达方式，以及将一个动作分散到多个依赖调用中。任务、可执行动作和可达状态均保持不变，因此成功率的任何变化都可完全归因于接口本身。我们对十一个LLM进行了评估，其中包括两个闭源模型……

    arXiv:2609.34971v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have shown strong performance on tool-use agentic tasks when given a fixed tool schema. Yet a tool schema is not the action space of an agent; it is merely one interface representation of it. The same executable action can be exposed through many different, functionally equivalent tool definitions, and an agent that has truly learned a task should behave consistently across them. We show that current agents often do not, a phenomenon we term schema bias. To study this systematically, we introduce an executable transformation framework that rewrites a native tool schema using nine operators, including merging and splitting tools, altering how a single tool is expressed, and distributing one action across several dependent calls. The tasks, executable actions, and reachable states remain fixed, so any change in success is attributable to the interface alone. Evaluating eleven LLMs, including two closed mode
    
[^217]: SpeechCritic：从有限人类偏好中学习诊断式语音评判器

    SpeechCritic: Learning a Diagnostic Speech Judge from Limited Human Preferences

    [https://arxiv.org/abs/2609.34582](https://arxiv.org/abs/2609.34582)

    SpeechCritic 仅用约300个人工标注的偏好比较，通过校准（而非替换）前沿音频-语言模型，学习到能判断语音优劣、指出感知维度差异并提供可听线索证据的诊断式语音评判器。

    

    人类语音传递着丰富的感知信息，例如情感和说话人身份，然而大多数自动语音质量评判器将其简化为单一的自然度分数。我们研究诊断式语音评判器：给定两个候选语音，诊断式评判器能够判断哪一个更好、两者在哪些感知维度（如音色、情感、时机）上存在差异，以及哪些可听线索支持其判断。学习此类评判器极具挑战性：专家标注成本高昂，而仅仅通过提示前沿音频-语言模型来生成标签并不可靠——我们的探测实验揭示了大量错误和不稳定的指令遵循能力。我们提出了 SpeechCritic，它在参考条件下的跨语言环境中，仅从大约 300 个人工标注的比较样本中学习诊断式评判器。SpeechCritic 并非取代前沿模型，而是利用这些标签对其进行校准：对于每个维度，它选择与人类判断一致的声学测量指标（摘要在此处截断）。

    arXiv:2609.34582v2 Announce Type: replace  Abstract: Human speech conveys rich perceptual information, such as emotion and speaker identity, yet most automatic speech quality judges reduce it to a single naturalness score. We study diagnostic speech judges: given two candidates, a diagnostic judge decides which is better, along which perceptual dimensions (e.g., timbre, emotion, timing) they differ, and which audible cues support its decision. Learning such judges is challenging: expert annotation is costly, and simply prompting a frontier audio-language model to produce labels is unreliable: our probing reveals substantial errors and unstable instruction following. We introduce SpeechCritic, which learns a diagnostic judge in a reference-conditioned cross-lingual setting from only about 300 human-labeled comparisons. Rather than replacing the frontier model, SpeechCritic calibrates it with these labels: for each dimension, it selects the acoustic measurements that agree with human jud
    
[^218]: R$^2$ Flow：通过递归技能演化实现递归自我改进

    R$^2$ Flow: Recursive Self-Improvement via Recursive Skill Evolution

    [https://arxiv.org/abs/2609.33867](https://arxiv.org/abs/2609.33867)

    提出 R$^2$ Flow 递归自我改进框架，在共享状态编排图上交替进行策略学习、独立验证与版本化技能库更新，并通过合并等效执行历史和流份额读出机制，解决流训练中的策略坍缩、信用分配偏差与奖励依赖三大难题，实现智能体可靠的递归自我改进。

    

    基于大语言模型（LLM）的智能体能够通过复用和修订其编排为可执行程序的技能，在跨任务中实现自我改进。基于流（flow）的训练恰好契合这一循环：它按奖励比例对程序进行采样，而流经每个技能的流则为其在下次技能库修订中提供功劳归因。要使这种自我改进变得可靠，需要克服三个障碍：流训练在树状结构的历史上会遭遇策略坍缩；非负的基于流的信用分配会把频繁使用误当作有益而加以奖励；技能库的修订又依赖于策略所优化的任务奖励。我们提出 R$^2$ Flow——一个递归自我改进框架，它在共享状态的编排图上交替执行策略学习、独立验证和版本化的技能库更新。该图会合并仅在独立步骤顺序上不同的历史记录，使流训练能够在等效执行之间汇集证据。通过流份额读出……（原文摘要在此处截断）

    arXiv:2609.33867v2 Announce Type: replace  Abstract: LLM-based agents can improve themselves across tasks by reusing and revising the skills they orchestrate into executable procedures. Flow-based training fits this loop: it samples procedures in proportion to reward, and the flow through each skill credits it for the next library revision. Three obstacles stand in the way of making this self-improvement reliable: flow training suffers strategy collapse over tree-structured histories; nonnegative flow-based credit rewards frequent use as if it were benefit; and library edits rest on the task reward the policy optimizes. We introduce R$^2$ Flow, a recursive self-improvement framework that alternates policy learning, independent verification, and versioned skill-library updates on a shared-state orchestration graph. The graph merges histories that differ only in the order of independent steps, allowing flow training to pool evidence across equivalent executions. A flow-share readout of t
    
[^219]: RSD-Poker：面向不完美信息游戏中残差策略的结构自适应与偏移稳健的风险-效用认证

    RSD-Poker: Structure-Adaptive and Shift-Robust Risk-Utility Certification for Residual Policies in Imperfect-Information Games

    [https://arxiv.org/abs/2609.33669](https://arxiv.org/abs/2609.33669)

    该论文提出 RSD-Poker 框架，通过冻结残差候选库、学习并预先冻结策略可见的子群划分，为不完美信息游戏中的残差策略提供对部署分布偏移稳健的风险上界与效用下界双证书认证。

    

    残差策略适配提供了一种轻量级的方式来修改强参考策略，但共享的缩放尺度与固定的子群划分可能掩盖异质性的性能退化，并在部署时信息状态混合分布发生变化时变得脆弱。我们提出了 RSD-Poker，一个结构自适应且对分布偏移稳健的认证框架：它冻结一个由残差族和尺度组成的候选库，在独立的结构划分集上学习一个策略可见的子群划分，并在加入校准标签之前冻结该划分。每个“候选-组”配对都会获得一个关于相对于锚点策略风险的加权同时上界证书，以及一个关于弱响应效用的下界证书。随后，在一个预先声明的部署组比例不确定性集合上，选出稳健的组到候选的映射。在从每个冻结组的分布中抽取独立校准单元、候选库与划分在校准前固定、以及组内……（原文摘要在此处截断）

    arXiv:2609.33669v2 Announce Type: replace  Abstract: Residual policy adaptation provides a lightweight way to modify a strong reference policy, but a shared scale and a fixed subgroup partition can hide heterogeneous degradation and become fragile when the deployment mixture of information states changes. We introduce RSD-Poker, a structure-adaptive and shift-robust certification framework that freezes a bank of residual families and scales, learns a policy-visible partition on an independent structure split, and freezes that partition before calibration labels are joined. Each candidate-group pair receives a weighted simultaneous upper certificate for anchor-relative risk and a lower certificate for weak-response utility. A robust group-to-candidate map is then selected over a predeclared uncertainty set of deployment group proportions. Under independent calibration units drawn from each frozen group's law, a candidate bank and partition fixed before calibration, and invariant within-
    
[^220]: 审计优先的VAPO：不完美验证下的风险认证选择性更新

    Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification

    [https://arxiv.org/abs/2609.33662](https://arxiv.org/abs/2609.33662)

    该论文提出Audit-First VAPO框架，通过将更新方向的离散准入与幅值的连续控制相分离，在有限验证预算下利用“接受-申诉-弃权”观察策略和有限样本统计界，实现对不完美验证器下有害更新风险、覆盖率与验证器调用率的可认证保障。

    

    不完美的验证器即使在裁剪和正则化限制了其幅值的情况下，仍可能分配出有害的更新方向。我们提出了Audit-First VAPO，它将离散的方向准入与连续的幅值控制分离开来。一种仅用于观察的“接受-申诉-弃权”策略使用有限的二次验证预算；其动作轨迹在加入干净标签之前被冻结。随后，同时的有限样本界在一个预先声明的策略族上对选定的有害风险、覆盖率以及验证器调用率进行认证。条件Hoeffding-Azuma界解释了由共享预算所引起的依赖性，而rollout或验证器的变更会启动新的认证阶段。在准入之后，一个有界的trust-clip-KL执行器来控制幅值大小。我们在两个模型和两个推理基准上对该方法进行评估，对比了静态RLVR、匹配随机选择、置信度阈值、噪声校正以及验证器增强等基线方法。在Qwen3.5-0.8B和GSM8K上，在目标……

    arXiv:2609.33662v2 Announce Type: replace  Abstract: Imperfect verifiers can assign a harmful update direction even when clipping and regularization bound its magnitude. We introduce Audit-First VAPO, which separates discrete directional admission from continuous magnitude control. An observation-only accept-appeal-abstain policy uses a finite secondary-verification budget; its action trace is frozen before clean labels are joined. Simultaneous finite-sample bounds then certify selected harmful risk, coverage, and verifier-call rate over a predeclared policy family. Conditional Hoeffding-Azuma bounds account for the dependence induced by shared budgets, and rollout or verifier changes initiate a new certification stage. After admission, a bounded trust-clip-KL actuator controls magnitude. We evaluate two models on two reasoning benchmarks against static RLVR, matched-random selection, confidence thresholding, noise correction, and verifier augmentation. On Qwen3.5-0.8B and GSM8K at tar
    
[^221]: OSCC：面向非完美信息学习中梯度噪声控制的认证观测安全耦合优化

    OSCC: Certified Observation-Safe Coupling Optimization for Gradient-Noise Control in Imperfect-Information Learning

    [https://arxiv.org/abs/2609.33543](https://arxiv.org/abs/2609.33543)

    提出了观测安全反事实耦合（OSCC）框架，通过定义可容许耦合类的多项安全准则并引入梯度感知的耦合优化准则，在非完美信息学习中安全地降低反事实比较噪声，同时针对策略梯度几何控制回报协方差以优化多动作策略。

    

    耦合的轨迹推演可以降低反事实动作比较中的噪声，但存在两个问题阻碍了标准的共同随机数构造在非完美信息环境中作为通用学习原语来使用。首先，无效的耦合可能暴露隐藏状态、同步内生策略随机性，或在反事实历史分叉后错配机会事件。其次，在多动作策略优化中，更低的回报对比方差本身并不是相关的优化目标：优化器实际依赖于经过局部策略梯度几何投影后的回报协方差矩阵。我们提出了观测安全反事实耦合，这是一个通过边际保持、信息状态安全、分支局部策略随机性、语义事件对齐以及先记录后预言机重放等准则来定义可容许耦合类的框架。我们推导了一个梯度感知的耦合准则，表明对于边际保持的……（原文摘要在此处被截断）

    arXiv:2609.33543v2 Announce Type: replace  Abstract: Coupled rollouts can reduce the noise of counterfactual action comparisons, but two issues prevent standard common-random-number constructions from serving as a general learning primitive in imperfect-information environments. First, an invalid coupling may expose hidden state, synchronize endogenous policy randomness, or misalign chance events after counterfactual histories diverge. Second, in multi-action policy optimization, lower return-contrast variance is not by itself the relevant objective: the optimizer depends on the return covariance matrix after projection through the local policy-gradient geometry. We introduce observation-safe counterfactual coupling (OSCC), a framework that defines an admissible class through marginal preservation, information-state safety, branch-local policy randomness, semantic event alignment, and trace-before-oracle replay. We derive a gradient-aware coupling criterion showing that, for marginal-p
    
[^222]: NutriVision：基于食材条件化融合与预测的单图像食物营养估计

    NutriVision: Ingredient-Conditioned Fusion and Prediction for Single-Image Food Nutrition Estimation

    [https://arxiv.org/abs/2609.33076](https://arxiv.org/abs/2609.33076)

    NutriVision是一个端到端营养估计框架，通过DepthAnything-V3生成深度信息并结合CLIP编码的食材语义，借助食材条件化频率对齐融合模块（IC-FAFM），从单张RGB图像和可选食材列表中估计食物的卡路里、质量、脂肪、碳水化合物和蛋白质。

    

    营养估计是消费者饮食追踪、临床营养学、慢性病管理、运动与医院营养以及更广泛的食品计算系统中的一项基础任务。现有方法沿两条基本独立的方向发展：一类是依赖校准RGB-深度采集的视觉模型，另一类是利用文本线索但多模态融合有限的食材感知方法。我们提出了NutriVision，一个端到端框架，它利用视觉几何与食材语义，从单张RGB图像和可选的食材列表中估计卡路里、质量、脂肪含量、碳水化合物和蛋白质。该方法使用DepthAnything-V3获取原本不可用的深度模态，并使用CLIP编码食材描述。它集成了三种互补机制：（1）食材条件化频率对齐融合模块（IC-FAFM），利用文本引导对RGB-深度频率进行重新加权和对齐……

    arXiv:2609.33076v2 Announce Type: replace-cross  Abstract: Nutrition estimation is a fundamental task in consumer diet tracking, clinical dietetics, chronic disease management, sports and hospital nutrition, and broader food computing systems. The existing approaches have progressed along two largely separate axes, vision models that rely on calibrated RGB-depth captures and ingredient-aware methods that use textual cues but use limited multimodal fusion. We introduce NutriVision, an end-to-end framework that leverages visual geometry and ingredient semantics to estimate calories, mass, fat content, carbohydrates, and protein from a single RGB image and an optional ingredient list. It obtains the unavailable depth modality using DepthAnything-V3 and encodes ingredient descriptions using CLIP. It integrates three complementary mechanisms: (1) an \emph{Ingredient-Conditioned Frequency-Aligned Fusion Module (IC-FAFM)}, which uses textual guidance to reweight and align RGB-depth frequency 
    
[^223]: Porimon：一种基于大语言模型并通过长/短期知识增强生成进行增强的宝可梦对战智能体

    Porimon: An LLM-Based Pok\'emon Battle Agent Enhanced by Long/Short-Term Knowledge Augmented Generation

    [https://arxiv.org/abs/2609.32544](https://arxiv.org/abs/2609.32544)

    本文提出长/短期知识增强生成机制（LSTKAG）并据此构建宝可梦对战智能体Porimon，无需微调即可让LLM智能体利用当前任务历史状态并从相似过往经验中检索总结，配合外部API精确计算伤害，在15,000场对战中显著超越基线玩家。

    

    本文以宝可梦对战为案例研究，探讨如何在无需额外微调的情况下，提升基于大语言模型（LLM）的智能体在需要感知对手进行规划的任务中的表现。我们提出了长/短期知识增强生成（LSTKAG），该机制使基于LLM的智能体能够利用当前任务的历史状态，并根据当前状态从相似的过往任务实例中检索经验总结。基于LSTKAG，我们设计了Porimon，一种用于宝可梦对战的基于LLM的智能体结构。在优化方面，我们引入了一个外部API，用于精确的伤害计算以及获取更详细的游戏信息。我们进行了锦标赛式的评估实验，共包含15,000场对战，用于超参数优化、消融研究和性能评估。结果表明，经过超参数优化的基于Porimon的玩家显著优于基于Pok…（原文在此处截断）

    arXiv:2609.32544v2 Announce Type: replace  Abstract: In this paper, we use Pok\'emon Battles as a case study to investigate how to improve the performance of LLM-based agents in tasks that require opponent-aware planning without additional fine-tuning. We propose Long/Short-Term Knowledge Augmented Generation (LSTKAG), a mechanism that enables LLM-based agents to leverage past states of the current task and retrieve experience summaries from similar previous task instances based on the current state. Based on LSTKAG, we design Porimon, an LLM-based agent structure for Pok\'emon Battles. For optimization, we introduce an external API for precise damage calculation and more detailed information about the game. We conduct tournament-like evaluation experiments comprising 15,000 battles for hyperparameter optimization, ablation studies, and performance evaluation. The results indicate that Porimon-based players with hyperparameter optimization significantly outperform players based on Pok\
    
[^224]: Metro-WM：基于可实现子目标的长时程潜在规划

    Metro-WM: Long-Horizon Latent Planning with Realisable Sub-Goals

    [https://arxiv.org/abs/2609.31868](https://arxiv.org/abs/2609.31868)

    论文指出分层JEPA宏观规划器预测的潜在子目标往往物理上不可实现，并提出Metro-WM框架，通过从历史经验构建的图中检索真实状态作为子目标，从而实现可靠的长时程规划。

    

    基于联合嵌入预测架构（JEPA）的模型预测控制提供了一个强大的零样本目标达成规划器，但它仅在短规划时域内有效。分层扩展试图通过学习一个宏观规划器来预测中间潜在子目标以引导微观规划器，从而弥补这一差距。在本工作中，我们证明无约束的潜在子目标预测存在根本性缺陷。严格的评估表明，领先的先进宏观规划器经常发出物理上无法实现的子目标。为解决这一问题，我们提出了 Metro-WM，这是一个分层框架，它通过从先前经验中检索真实状态来发布子目标，而不是生成无依据的潜在向量。具体而言，Metro-WM 构建了一个图，其顶点是来自离线专家演示或随机动作轨迹的观测帧，允许不同回合的帧相互连接……

    arXiv:2609.31868v2 Announce Type: replace  Abstract: Model-predictive control with Joint-Embedding Predictive Architectures (JEPAs) provides a strong zero-shot goal-reaching planner, but it is only effective over short planning horizons. Hierarchical extensions attempt to bridge this gap by learning a macro planner to predict intermediate latent sub-goals to guide the micro planner. In this work, we demonstrate that unconstrained latent sub-goal prediction is fundamentally flawed. A rigorous evaluation reveals that a leading state-of-the-art macro planner routinely emits physically unrealisable sub-goals. To resolve this, we introduce Metro-WM, a hierarchical framework that issues sub-goals by retrieving genuine states from prior experience rather than generating ungrounded latent vectors. Specifically, Metro-WM constructs a graph whose vertices are observed frames from offline expert demonstrations or random-action trajectories, allowing frames from different episodes to be connected 
    
[^225]: 递归自我改进人工智能的进化安全性：分类、风险发现与评估

    Evolutionary Safety of Recursive Self-Improving AI: Taxonomy, Risk Discovery, and Evaluation

    [https://arxiv.org/abs/2609.31186](https://arxiv.org/abs/2609.31186)

    该论文提出“进化安全”这一新视角，用于研究AI在递归自我改进过程中安全属性如何随演化而变化、持续、积累与传播，并给出了相应的风险分类、风险发现与评估框架。

    

    人工智能正在快速发展，能力日益强大的系统在推理、决策、科学发现和自主开发中扮演着越来越重要的角色。随着AI开始参与自身的改进——从模型训练和经验积累到智能体演化和自动化AI开发——递归自我改进（RSI）的前景正变得越来越相关。这一转变提出了一个根本性的安全问题：当系统本身、其积累的经验、甚至产生其后继者的过程都在持续变化时，如何保持安全性？我们引入“进化安全”作为研究持续递归自我改进下安全性的新视角。它关注的不仅是AI系统在某一特定时刻是否安全，而是安全属性如何在整个演化过程中变化、持续、积累和传播。我们刻画了反复出现的典型表现形式，包括……

    arXiv:2609.31186v1 Announce Type: new  Abstract: Artificial intelligence is advancing rapidly, with increasingly capable systems taking larger roles in reasoning, decision-making, scientific discovery, and autonomous development. As AI begins to participate in its own improvement, from model training and experience accumulation to agent evolution and automated AI development, the prospect of recursive self-improvement (RSI) is becoming increasingly relevant. This transition raises a fundamental safety question: how can safety be maintained when the system, its accumulated experience, and even the process producing its successors continue to change?   We introduce Evolutionary Safety as a perspective for studying safety under persistent and recursive self-improvement. It concerns not only whether an AI system is safe at a particular moment, but how safety properties change, persist, accumulate, and propagate throughout evolution. We characterize recurring manifestations, including inten
    
[^226]: 跨模态情感理解：一种用于对话情感识别的Transformer-GAT方法

    Cross-Modal Emotion Understanding: A Transformer-GAT Approach for Dialogue Emotion Recognition

    [https://arxiv.org/abs/2609.29556](https://arxiv.org/abs/2609.29556)

    该论文提出了一种结合Transformer与图注意力网络的混合框架Transformer-GAT，通过Transformer捕获全局语义信息、图注意力网络建模模态间细粒度关系来实现跨模态对话情感识别，在IEMOCAP和MELD数据集上分别取得72.45%和77.37%的加权F1分数，超越了现有最先进方法。

    

    多模态情感识别是情感计算领域的一个关键研究方向，在情感分析、智能客服和人机交互中具有广泛应用。然而，现有方法通常依赖于单一模态特征或简单的多模态融合，未能捕捉全局与局部上下文之间的协同作用，从而限制了模型的性能和情感理解能力。为应对这一挑战，我们提出了Transformer-GAT，一种结合Transformer与图注意力网络的混合框架，以实现跨模态情感理解。其中，Transformer用于捕获全局语义信息，而图注意力网络用于建模模态之间的细粒度关系，从而增强情感特征的表示。在IEMOCAP和MELD数据集上的实验表明，我们的模型分别取得了72.45%和77.37%的加权F1分数，优于当前最先进的方法。

    arXiv:2609.29556v1 Announce Type: new  Abstract: Multimodal emotion recognition is a key research area in affective computing, with applications in sentiment analysis, intelligent customer service, and human-computer interaction. However, existing methods often rely on single-modal features or simple multimodal fusion, failing to capture the synergy between global and local contexts, which limits model performance and emotion understanding. To address this challenge, we propose Transformer-GAT, a hybrid framework that combines Transformer and the Graph Attention Network to enable cross-modal emotion understanding. The Transformer is used to capture global semantic information, while the Graph Attention Network is employed to model fine-grained relationships between modalities, thereby enhancing the representation of emotional features. Experiments on the IEMOCAP and MELD datasets show that our model achieves weighted F1 scores of 72.45% and 77.37%, outperforming state-of-the-art method
    
[^227]: SWE-Prometheus：衡量真实世界代码仓库中的工程治理改进

    SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories

    [https://arxiv.org/abs/2609.29465](https://arxiv.org/abs/2609.29465)

    该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。

    

    基于大语言模型的编程智能体在仓库级软件工程任务上已取得显著进展。然而，现有的仓库基准测试通常从一个由人类确定的问题出发，评估补丁是否满足某个功能性信号。我们提出了SWE-Prometheus，一个针对更广泛任务——改进代码仓库工程治理——的基准测试。每个任务提供一个固定的代码快照和一个开放式的目标，要求智能体识别风险、确定干预措施的优先级并验证由此产生的变更。SWE-Prometheus通过配对证据、干净环境探测、行为门控以及对同一证据的两项独立教师评分，来评估六个治理维度。该基准包含60个仓库；十个模型在一个共享的22仓库公共子集上进行评估，其中平均归一化治理改进得分介于0.0568到0.5760之间，观察到的行为破坏率介于0%到23%之间。

    arXiv:2609.29465v1 Announce Type: new  Abstract: Large language model based coding agents have made substantial progress on repository-level software engineering tasks. Existing repository benchmarks, however, usually start from a human-identified issue and evaluate whether a patch satisfies a functional signal. We present SWE-Prometheus, a benchmark for the broader task of improving repository engineering governance. Each task provides a fixed snapshot and an open-ended objective, requiring the agent to identify risks, prioritize interventions, and verify the resulting changes. SWE-Prometheus evaluates six governance dimensions through paired evidence, clean-environment probes, behavior gates, and two independent teacher ratings of the same evidence. The benchmark contains 60 repositories; ten models are evaluated on a shared 22-repository public subset, where mean Normalized Governance Improvement ranges from 0.0568 to 0.5760 and observed behavior-breakage rates range from 0% to 23%.
    
[^228]: 从自我蒸馏到自我练习：面向多轮智能体的特权信息

    From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents

    [https://arxiv.org/abs/2609.29051](https://arxiv.org/abs/2609.29051)

    本文发现基于特权信息的同策略自我蒸馏会让多轮智能体“盲目自信”且性能甚至差于普通强化学习和基础模型，并提出特权自我练习（PSP）方法，将特权信息从损失函数转移到采样阶段，通过注入分析模型生成的任务指令引导重新采样，再用不变的GRPO目标训练，从而有效提升多轮智能体性能。

    

    同策略自我蒸馏（OPSD）已成为大语言模型智能体后训练的一种流行方法。它通过让同一模型以特权信息（PI）为条件获得更强的教师视角，从而在词元（token）级别对智能体模型进行监督。在本工作中，我们证明在多轮智能体中，这种范式教会学生模型自信地行动，却不掌握其背后的信息。训练后的智能体表现得仿佛拥有它从未见过的特权信息，其性能远逊于普通的强化学习，最坏情况下甚至低于未经训练的基础模型。因此，我们提出特权自我练习（PSP），它保留特权信息，并将其从损失函数转移到采样器中。当学生在某个任务上的采样结果大多失败时，我们注入一段由分析模型编写的简短任务特定指令，将该指令置于上下文中重新采样该任务，并在保持GRPO目标不变的情况下对结果进行训练。特权信息……

    arXiv:2609.29051v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) has become a popular recipe for post-training LLM agents. It supervises the agent model at the token level with a stronger teacher view of the same model, obtained by conditioning on privileged information (PI). In this work, we show that in multi-turn agents, this paradigm teaches the student to act with confidence but without the information behind it. The trained agent behaves as if it had privileged information it never observed, and its performance falls well short of plain RL, in the worst case below the untrained base model. Therefore, we propose Privileged Self-Practice (PSP), which keeps the PI and moves it from the loss to the sampler. When the student's rollouts on a task mostly fail, we inject a short per-task instruction written by an analyzer model, sample the task again with the instruction in context, and train on the result with an unchanged GRPO objective. The privileged information st
    
[^229]: 儿童如何设计并推理值得信赖的AI聊天机器人

    How Children Design and Reason about Trustworthy AI Chatbots

    [https://arxiv.org/abs/2609.25244](https://arxiv.org/abs/2609.25244)

    本研究开发了一个让儿童自主设计聊天机器人的平台，通过对115名8-18岁学习者的混合方法研究发现，低龄学生会设置更高的自信度，甚至认为“故意出错但按设计行事”的聊天机器人也值得信赖，揭示了儿童对AI可信度的独特理解方式。

    

    儿童越来越多地与AI聊天机器人互动，因此信任校准成为AI素养的重要组成部分。以往研究主要将儿童对AI的信任视为用户评估他人构建的系统，而非作为自己聊天机器人的设计者。我们开发了一个聊天机器人构建环境，支持调节与信任相关的特质（如自信度、透明度、正式程度、果断性）、规则和角色设定。我们对115名学习者（8-18岁）开展了混合方法研究，他们共制作了119个聊天机器人。我们考察了儿童如何配置他们的聊天机器人、如何推理可信度，以及聊天机器人的行为与其设计的契合程度。年龄较小的学生（10-13岁）设置的自信度显著高于年龄较大的学生（14-18岁），部分学生还刻意构建了会故意给出错误答案的聊天机器人，却仍认为其“值得信赖”，理由是聊天机器人做了它被设计要做的事情。低龄学生将信任等同于目的（摘要在此处截断）。

    arXiv:2609.25244v2 Announce Type: replace-cross  Abstract: Children increasingly interact with AI chatbots, making trust calibration essential to AI literacy. Prior research has examined children's trust in AI mainly as users evaluating systems built by others, rather than as designers of their own chatbots. We developed a chatbot-building environment with adjustable trust-relevant traits (e.g., confidence, transparency, formality, assertiveness), rules, and persona. We conducted mixed-methods study with 115 learners (ages 8-18) who made 119 chatbots. We examined how children configured their chatbots, reasoned about trustworthiness, and how closely chatbot behavior aligned with their designs. Younger students (age 10-13) set significantly higher confidence than older students (age 14-18), and some deliberately built chatbots that gave wrong answers on purpose, yet still called them trustworthy, arguing that a chatbot does what it was built to do. Younger students equated trust with pu
    
[^230]: 注意相位：腿式运动中的有效秩与表征健康度

    Mind the Phase: Effective Rank and Representation Health in Legged Locomotion

    [https://arxiv.org/abs/2609.06958](https://arxiv.org/abs/2609.06958)

    该论文通过按步态相位分解策略雅可比矩阵的有效秩，揭示了层归一化和残差连接会使摆动相比支撑相多获得约两个有效秩维度的架构结构，并据此提出了提升腿式运动策略表征健康度的简单方案。

    

    强化学习已成为腿式运动领域的主导范式，通过大规模并行仿真实现了从后空翻到跑酷等复杂行为。在PPO的非平稳性条件下，浅层网络仍是事实上的标准架构，并辅以精心设计的课程与环境，然而这些策略所学到的表征仍然鲜为人知，目前没有任何训练时的信号能够预示它们在真实硬件上的表现。在这项工作中，我们通过策略雅可比矩阵的有效秩对运动策略进行了实证研究，并表明将秩条件化于步态相位可以揭示被全局秩平均所掩盖的架构结构。特别地，我们发现标准的架构选择，即层归一化和残差连接，会为摆动相分配比支撑相多出约两个维度的有效秩，而这种现象在普通MLP中完全不存在。基于此，我们提出了一个简单的方案，该方案……

    arXiv:2609.06958v1 Announce Type: cross  Abstract: Reinforcement learning has become the leading paradigm in legged locomotion, enabling complex behaviors from backflips to parkour through massively parallel simulation. Under PPO's non-stationarity, shallow networks remain the de facto architecture, supported by carefully staged curricula and environments, yet the representations these policies learn stay poorly understood, leaving no training-time signal of how they will behave on hardware. In this work, we empirically study locomotion policies through the effective rank of the policy Jacobian and show that conditioning rank on the gait phase exposes architectural structure that global rank averages away. In particular, we find that standard architectural choices, namely layer normalization and residual connections, allocate roughly two more dimensions of effective rank to swing than to stance, which is fully absent in vanilla MLPs. Building on this, we propose a simple recipe that tu
    
[^231]: CABAL：用于追踪同行评审中合谋投标影响的多智能体仿真框架

    CABAL: Multi-Agent Simulacra for Tracing the Effects of Collusive Bidding in Peer Review

    [https://arxiv.org/abs/2609.05227](https://arxiv.org/abs/2609.05227)

    提出了端到端多智能体仿真框架CABAL，通过为LLM审稿人智能体配置诚实或合谋策略来研究同行评审中合谋投标的影响，并设计了基于审稿人-论文亲和力的合谋投标策略，实现与专业领域一致而非任意的定向攻击。

    

    AAAI-27评审周期期间的最新报告凸显了审稿人通过协调投标以获取互惠分配优势的风险。先前的工作将投标、审稿人分配和评审操纵视为相互独立的阶段，导致合谋投标在整个流程中的生命周期效应尚不清楚。现实世界的分析进一步受到限制，因为合谋意图通常无法被观察到，且缺乏针对同一会议的反事实对照。基于这一研究空白，我们提出了CABAL，一个端到端的多智能体仿真框架，用于研究审稿人分配的完整性：该框架通过固定会议环境，并配置由大语言模型驱动的、采用诚实或合谋策略的审稿人智能体来进行实验。我们进一步开发了一种亲和力引导的合谋投标策略，该策略利用审稿人与论文之间的相互亲和力来构建合谋圈子并选择目标论文，从而产生与专业领域一致而非任意针对性的攻击。受控实验表明……

    arXiv:2609.05227v1 Announce Type: new  Abstract: Recent reports during the AAAI-27 review cycle highlight the risk of reviewers coordinating bids for reciprocal assignment advantage. Prior work treats bidding, reviewer assignment, and review manipulation as separate stages, leaving the lifecycle effects of collusive bidding unclear. Real-world analysis is further constrained by typically unobservable collusive intent and the lack of counterfactuals for the same conference. Motivated by this gap, we introduce \alg, an end-to-end multi-agent simulacra framework for studying reviewer assignment integrity by holding the conference environment fixed and configuring LLM-driven reviewer agents with honest or collusive policies. We further develop an affinity-guided collusive bidding strategy that uses mutual reviewer-paper affinities to construct collusion rings and select target papers, producing expertise-consistent rather than arbitrarily targeted attacks. Controlled experiments show that 
    
[^232]: 能力门控语言模型：安全性可组合，实用性不可组合

    Capability-Gated Language Models: Security Composes, Utility Does Not

    [https://arxiv.org/abs/2609.00445](https://arxiv.org/abs/2609.00445)

    提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。

    

    已部署的语言模型安全防护措施（安全微调、过滤、遗忘学习）仅在模型权重之外按主体区分：过滤器被重新配置、访问层级成倍增加、模型制品被重新发布；而在同一组权重内部，所有请求面对的都是相同的模型配置。这促使我们定义“能力门控部署”：在一组权重内部实现按主体的访问控制，其配置构成一个格——交运算累积某个主体的限制，并运算汇集一个联盟的权限范围。我们通过在现有嵌套分解机制上实施稀疏秩门控来实例化该方法，利用单次遍归因引导配置搜索，并仅从预注册的保留集划分中一次性读取全部结果。安全性可组合：在单调引出假设下，交运算处的组合性可被证明，而我们在逐点意义上证伪了该假设。在两个模型谱系中，保留集上交运算的中位数效应加深了抑制；经多重校正后唯一存活的效应进一步强化了这种抑制。实用性却不可组合：……（原文摘要在此处截断）

    arXiv:2609.00445v1 Announce Type: cross  Abstract: Deployed language model safeguards (safety fine-tuning, filtering, unlearning) vary by principal only outside the model weights: filters are reconfigured, tiers are multiplied, and artefacts are reissued; inside one set of weights every request meets the same model configuration. This motivates us to define capability-gated deployment: per-principal access control inside one set of weights, whose configurations form a lattice - meets accumulate a principal's restrictions and joins pool a coalition's reach. We instantiate it by sparse rank gating over an existing nested-factorisation mechanism, guide profile search with one-pass attribution, and read every result once from a pre-registered held-out split. Security composes: provably at meets under a monotone-elicitation assumption we falsify pointwise. In two lineages the median held-out meet deepens suppression; the one effect surviving correction strengthens it. Utility does not: indi
    
[^233]: 迈向智能体化云工程：基于零信任智能体套件的图工程与循环工程

    Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness

    [https://arxiv.org/abs/2609.00050](https://arxiv.org/abs/2609.00050)

    提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。

    

    智能体AI正在推动基于云的工作流的发展，其中自主智能体可以对运营状态进行推理、调用授权工具、修改软件和基础设施、部署服务、验证执行结果，并在长时程、多步骤任务中进行自适应调整。构建此类工作流需要针对工作流推进、受限执行、故障恢复和可验证完成等环节的显式机制。我们提出了智能体云工作流工程，这是一个智能体AI框架，它将自然语言描述的智能体云工程任务转化为经过验证的代码仓库和经过验证的运营性云部署，从而实现基于云的智能体工作流自动化。该框架分离了三个互补的关注点：图工程负责指定长时程工作流推进以及依赖验证的状态转移；循环工程提供有界的诊断、修复或重新规划、重试和重新验证；智能体套件工程则（负责执行零信任的受限执行控制）。

    arXiv:2609.00050v1 Announce Type: cross  Abstract: Agentic AI is enabling cloud-based workflows in which autonomous agents reason over operational state, invoke authorized tools, modify software and infrastructure, deploy services, verify execution outcomes, and adapt across long-horizon, multistep tasks. Engineering such workflows requires explicit mechanisms for workflow progression, constrained execution, failure recovery, and verifiable completion. We present Agentic Cloud Workflow Engineering, an agentic AI framework that transforms natural-language agentic cloud-engineering tasks into validated code repositories and verified operational cloud deployments for automating cloud-based agentic workflows. The framework separates three complementary concerns: graph engineering specifies long-horizon workflow progression and verification-dependent transitions; loop engineering provides bounded diagnosis, repair or re-planning, retry, and re-verification; and agent harness engineering enf
    
[^234]: IndicDetect：评估针对印地语、泰卢固语和泰米尔语的跨语言LLM生成文本检测

    IndicDetect: Evaluating Cross-Lingual LLM-Generated Text Detection for Hindi, Telugu, and Tamil

    [https://arxiv.org/abs/2608.29919](https://arxiv.org/abs/2608.29919)

    该论文提出了IndicDetect基准，系统评估印地语、泰卢固语和泰米尔语中AI生成文本检测器在领域偏移、生成器偏移和对抗性扰动等现实分布偏移下的鲁棒性，并发现现有检测器存在显著的鲁棒性缺陷。

    

    大语言模型（LLM）的快速普及进一步凸显了开发可靠的人工智能生成文本检测技术的必要性，尤其是在英语以外的语言领域。然而，当前的基准测试很少关注印度语言，并且通常在脱离现实世界的理想化设置中测试检测器。我们提出了一个针对印地语、泰卢固语和泰米尔语的人工智能生成文本检测的通用基准，称为IndicDetect，旨在评估检测器在现实分布偏移下的鲁棒性。IndicDetect包含精心整理的人类撰写文本及其对应的LLM生成文本，涵盖多个领域和多种生成器，并系统地在领域偏移、生成器偏移和对抗性扰动的情况下评估检测器。使用统一且可重复的评估方案，我们评估了广泛的统计类和神经类检测器。我们发现检测器存在显著的鲁棒性缺陷：有监督神经检测器在……（原文摘要在此处截断）

    arXiv:2608.29919v1 Announce Type: cross  Abstract: The rapid proliferation of LLMs has further heightened the need to develop dependable AI-generated text detection, especially beyond English. Nevertheless, current benchmarks pay little attention to Indic languages and test detectors in idealized settings that do not represent the real world. We present a generalized benchmark for AI-generated text detection in Hindi, Telugu, and Tamil, which we call IndicDetect, designed to assess the robustness of detectors under realistic distribution shifts. IndicDetect comprises highly curated human-written texts matched with LLM-generated counterparts across various domains and generators, and systematically evaluates detectors in the presence of domain shift, generator shift, and adversarial perturbation. Using a single and repeatable evaluation scheme, we evaluate a wide range of statistical and neural detectors. We find substantial robustness failures: supervised neural detectors perform well 
    
[^235]: 超越不确定性降低：面向交互式诊断的校准规划

    Beyond Uncertainty Reduction: Calibrated Planning for Interactive Diagnosis

    [https://arxiv.org/abs/2608.27847](https://arxiv.org/abs/2608.27847)

    该论文提出CaRDI框架，用经过保形校准的临床风险（结合主诊断错误与严重鉴别诊断遗漏的加权损失）取代传统的不确定性降低目标，以指导基于大语言模型的交互式医疗问诊中的提问选择与停止时机。

    

    交互式医疗诊断要求智能体在给出最终诊断之前，通过一系列序贯提问来收集患者证据。大型语言模型（LLM）如今使这类自动化问诊成为可能，而现有方法通常依据候选疾病之间不确定性降低的程度来选择下一个问题。然而，这一目标对不同诊断错误与遗漏严重疾病所造成的临床代价差异并不敏感，因此最小化不确定性并不一定能够最小化最终诊断损失。为弥合这一差距，我们提出了CaRDI（Calibrated Risk for Diagnostic Interaction，面向诊断交互的校准风险），一个围绕临床损失构建的规划框架，该损失将主要诊断错误与按严重程度加权的鉴别诊断遗漏相结合。诊断信念与安全信念分别估计这两个损失组成部分，同时采用保形残差校准来纠正由不完整证据导致的低估。校准后的风险随后通过平衡……（指导问题选择与停止时机）

    arXiv:2608.27847v2 Announce Type: replace  Abstract: Interactive medical diagnosis requires an agent to gather patient evidence through sequential questions before committing to a diagnosis. Large language models (LLMs) now make such consultations feasible, and existing approaches typically select questions by how much they reduce uncertainty among candidate diseases. However, this objective is insensitive to the unequal clinical costs of diagnostic errors and omitted severe alternatives, so minimizing uncertainty need not minimize final diagnostic loss. To bridge this gap, we propose CaRDI (Calibrated Risk for Diagnostic Interaction), a planning framework built around a clinical loss combining primary diagnosis error with severity-weighted differential omission. Diagnostic and safety beliefs estimate the two loss components, while conformal residual calibration corrects for underestimation from partial evidence. The calibrated risk then guides question selection and stopping by balanc
    
[^236]: BenthicDINO：基于物理信息的自蒸馏方法用于侧扫声呐视角不变表示学习

    BenthicDINO: Physics-Informed Self-Distillation for View-Invariant Side-Scan Sonar Representations

    [https://arxiv.org/abs/2608.23215](https://arxiv.org/abs/2608.23215)

    本文提出了一种基于物理信息的自蒸馏框架（BenthicDINO），通过物理模拟增强和HSIC惩罚强制侧扫声呐图像表示的视角不变性，有效解决了声学伪影导致的特征混合问题。

    

    arXiv:2608.23215v1 公告类型：交叉 摘要：侧扫声呐（SSS）图像中的自动感知受到物理声学伪影的严重阻碍，导致表示不可避免地混合了海底固有反射率与瞬态观测几何。现有的自监督学习（SSL）框架依赖于为自然图像设计的增强方法，未能考虑声学退化，也未能明确强制视角不变性。为解决这一差距，我们提出了一种基于物理信息的自蒸馏框架，构建在DINOv3架构之上，采用ConvNeXt-v2-Tiny骨干网络以最大化数据效率。该方法通过两种主要机制强制视角不变性：物理驱动的增强方法，模拟散斑噪声、距离相关衰减和辐射定标误差；以及希尔伯特-施密特独立性准则（HSIC）惩罚项，明确将学习到的密集补丁特征与物理观测参数解耦。

    arXiv:2608.23215v1 Announce Type: cross  Abstract: Automated perception in side-scan sonar (SSS) imagery is severely hindered by physical acoustic artifacts, resulting in representations that inextricably mix intrinsic seabed reflectivity with transient viewing geometries. Existing self-supervised learning (SSL) frameworks rely on augmentations designed for natural images, failing to account for acoustic degradation and explicitly enforce view-invariance. To address this gap, we introduce a physics-informed self-distillation framework built upon the DINOv3 architecture utilizing a ConvNeXt-v2-Tiny backbone to maximize data efficiency. The proposed methodology enforces view-invariance through two primary mechanisms: physically motivated augmentations that simulate speckle noise, range-dependent attenuation, and radiometric miscalibration; and a Hilbert-Schmidt Independence Criterion (HSIC) penalty that explicitly decouples learned dense patch features from physical viewing parameters. F
    
[^237]: ADEPT：通过预训练与后训练加速灵巧操作

    ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning

    [https://arxiv.org/abs/2608.19182](https://arxiv.org/abs/2608.19182)

    ADEPT通过预训练通用灵巧操作策略并采用稳定的后训练方案，显著加速了高自由度机器人从原始感知学习长时程任务的过程，避免了重复学习并提升了迁移能力。

    

    arXiv:2608.19182v1 公告类型：交叉 摘要：我们引入了“通过预训练加速灵巧操作”（ADEPT），这是一个大规模强化学习（RL）框架，用于学习跨高自由度（DoF）机器人实体可模拟到现实迁移的灵巧操作，能够直接从原始视觉-触觉感知解决长时程任务。ADEPT在通用物体重定位任务上预训练灵巧策略，然后以此预训练行为作为先验，对下游策略进行后训练。ADEPT使得在多指机器人上学习新行为成为可能，这些行为若从头开始发现则通常困难，并避免了为每个新下游任务重复学习相同技能集。预训练策略零样本执行下游任务的重定位阶段，但朴素RL微调在迁移过程中会迅速退化这一能力。我们通过结合行为克隆蒸馏、评论家预热和保守的在线更新的稳定后训练方案来解决这一问题。为安全地...

    arXiv:2608.19182v1 Announce Type: cross  Abstract: We introduce Accelerating Dexterity via Pre-Training (ADEPT), a large-scale reinforcement learning (RL) framework for learning sim-to-real transferable dexterity across high degree-of-freedom (DoF) robot embodiments that can solve long-horizon tasks directly from raw visuo-tactile perception. ADEPT pretrains a dexterous policy on a generic object reposing task, then post-trains downstream policies with this pretrained behavior as a prior. ADEPT enables learning new behaviors that are otherwise difficult to discover from scratch on multi-fingered robots and avoids learning the same set of skills over again for every new downstream task. The pretrained policy zero-shots the reposing phase of downstream tasks, but na\"ive RL fine-tuning rapidly degrades this capability during transfer. We address this with a stable post-training recipe combining behavior-cloning distillation, critic warm-up, and conservative on-policy updates. To safely e
    
[^238]: 自我标签与他者标签在LLM评审中引发双向偏差

    Self- and Other-Labels Induce Bidirectional Bias in LLM Judges

    [https://arxiv.org/abs/2608.18091](https://arxiv.org/abs/2608.18091)

    本文通过改变评估对象为无风格指纹的叙事约束选择，发现LLM评审中的自我偏好在控制质量后基本消失，甚至在某些维度上出现反向偏差，揭示了自我偏好并非固有属性。

    

    arXiv:2608.18091v1 公告类型：跨领域 摘要：随着LLM作为评审系统的日益普及，LLM中的自我偏好——即倾向于偏爱自身输出的倾向——引发了对评估可靠性的日益担忧。然而，这一现象主要是在生成文本上研究的，其中风格特征和响应质量不可避免地混杂在一起。因此，现有测量无法将真正的自我偏好与这些混淆因素区分开来。我们通过改变评估对象来解决这一问题：不是评判生成的文本，而是让十个LLM评估叙事约束选择，这些选择不携带模型特定的风格指纹，但保留可恢复的模型特定签名。我们进行了两个实验，得出了不同的发现。在盲评条件下，一旦控制选择质量和评审严格程度，自我偏好基本消失。它在四个评分维度中的三个上消失，在第四个上发生逆转，评审者将自己的选择评为更低。

    arXiv:2608.18091v1 Announce Type: cross  Abstract: As LLM-as-a-judge systems become increasingly widespread, self-preference in LLMs -- the tendency to favor one's own outputs -- raises growing concerns about evaluation reliability. However, it has been studied predominantly on generated text, where stylistic features and response quality are inevitably conflated. As a result, existing measurements cannot separate genuine self-preference from these confounds. We address this by changing the object of evaluation: instead of judging generated text, ten LLMs assess narrative constraint selections, which carry no model-specific stylistic fingerprint yet retain a recoverable model-specific signature. We run two experiments that yield distinct findings. Under blind evaluation, self-preference largely disappears once selection quality and evaluator severity are controlled. It vanishes on three of four rubric dimensions and reverses on the fourth, where judges rate their own selections as less
    
[^239]: BEAR-Bench：面向多模态模型的双语企业与学术推理基准

    BEAR-Bench: A Bilingual Enterprise and Academic Reasoning Benchmark for Multimodal Models

    [https://arxiv.org/abs/2608.17895](https://arxiv.org/abs/2608.17895)

    BEAR-Bench是一个全新的英俄双语基准，包含1000个基于专业商业和科学文档的人工标注问题，用于评估多模态大语言模型在文本密集场景下的推理能力，并揭示了当前最强模型仍存在显著性能提升空间。

    

    arXiv:2608.17895v1 公告类型：交叉 摘要：尽管多模态大语言模型（MLLMs）在视觉理解方面取得了显著进展，但它们在处理文本密集、专业文档方面的推理能力仍未得到充分评估。现有基准侧重于信息提取，需要外部领域知识，或将专业文档仅作为众多场景之一来覆盖。此外，这些基准大多以英语或中文为中心，导致其他语言，尤其是俄语，代表性严重不足。为解决这些局限性，我们引入了BEAR-Bench（双语企业与学术推理基准），这是一个自包含的、复杂的英俄双语基准，包含基于文本丰富的商业和科学文档的1000个人工标注问题。我们在BEAR-Bench上评估了16个专有和开源权重的MLLMs，包括Gemini 3.1 Pro和Qwen3.5-397B，并观察到即使是最强大的系统也存在明显的提升空间。最后，我们利用所得模型输出进行了进一步分析。

    arXiv:2608.17895v1 Announce Type: cross  Abstract: While Multimodal Large Language Models (MLLMs) have made significant strides in visual comprehension, their ability to reason about text-dense, professional documents remains incompletely evaluated. Existing benchmarks emphasize information extraction, require external domain knowledge, or cover professional documents only as one of many settings. They are also largely English- or Chinese-centric, leaving other languages and Russian, in particular, substantially underrepresented. To address these limitations, we introduce BEAR-Bench (Bilingual Enterprise and Academic Reasoning), a self-contained, complex English-and-Russian benchmark comprising 1000 human-annotated questions based on text-rich business and scientific documents. We evaluate 16 proprietary and open-weight MLLMs, including Gemini 3.1 Pro and Qwen3.5-397B, on BEAR-Bench and observe clear headroom even for the strongest systems. Finally, we use the resulting model outputs t
    
[^240]: 教授大小，而非方向：面向多轮多步LLM智能体的验证器受限信用分配

    Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents

    [https://arxiv.org/abs/2608.13179](https://arxiv.org/abs/2608.13179)

    CrEST通过层次化信用分配框架，结合验证器受限的强化学习上限和自教师模型的密集令牌信号，有效解决了多轮LLM智能体训练中的信用分配问题，在基准测试中持续超越现有方法。

    

    摘要：arXiv:2608.13179v1 公告类型：新 摘要：基于可验证奖励的强化学习（RLVR）为训练多轮工具使用智能体设定了验证器受限的性能上限，但其轨迹级信用分配将异构的每轮结果混为一谈，汇集成单一奖励信号。在线策略蒸馏提供了密集的逐令牌监督，但要么受限于教师模型，要么容易导致梯度集中崩溃。我们引入了$\textbf{CrEST}$，一种层次化信用分配框架，它保留了RL的验证器受限上限，同时融入了来自特权自教师模型的密集令牌级信号。$\textbf{CrEST}$在两个层面解决信用分配问题：轮次分割的验证优势应对轮间稀释，而熵门控的自教师调制细化轮内令牌贡献。在BFCL V3和WildToolBench上的实验表明，$\textbf{CrEST}$在两种模型规模上均持续优于RL和蒸馏基线，其中最大规模的优势最为显著。

    arXiv:2608.13179v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) offers a verifier-bounded performance ceiling for training multi-turn tool-use agents, yet its trajectory-level credit assignment conflates heterogeneous per-turn outcomes into a single reward signal. On-policy distillation provides dense per-token supervision but is either teacher-bounded or prone to gradient concentration collapse. We introduce $\textbf{CrEST}$, a hierarchical credit assignment framework that retains RL's verifier-bounded ceiling while incorporating dense token-level signals from a privileged self-teacher. $\textbf{CrEST}$ resolves credit at two levels: turn-segmented verified advantages address inter-turn dilution, while entropy-gated self-teacher modulation refines intra-turn token contributions. Experiments on BFCL V3 and WildToolBench show that $\textbf{CrEST}$ consistently outperforms both RL and distillation baselines across two model scales, with the largest 
    
[^241]: Rescene：基于冻结天气模型的稳定百年气候模拟

    Rescene: stable century-long climate emulation from a frozen weather model

    [https://arxiv.org/abs/2608.09971](https://arxiv.org/abs/2608.09971)

    Rescene仅用一个0.4M参数的轻量封装，将冻结的机器学习天气模型转化为可稳定运行百年、无漂移且变率得到良好校准的随机气候模拟器。

    

    机器学习天气模型如今在中期预报中已能匹敌甚至超越业务数值预报，但当它们自由运行多年时会出现漂移、发散或失去变率，而专门的气候模拟器训练成本高昂。我们提出Rescene，一个仅0.4M参数的轻量封装器，可将一个冻结的1.5°分辨率、6小时步长、69变量天气模型转化为随机气候模拟器。一个确定性的慢时钟将冻结模型锚定于日历气候态，而一个带限AR(1)扰动（其谱包络固定于观测到的异常谱）则恢复锚定过程所移除的天气变化。一次100年、8成员的Rescene运行未显示出可检测的漂移（每100年−0.004 ± 0.014 K）；在2022年测试年上，16成员集合被良好校准（Z500、T2m和MSLP的离散度/技巧比分别为0.89、0.97和0.93），且其温带天气振幅在调参年达到ERA5的97%。

    arXiv:2608.09971v2 Announce Type: replace-cross  Abstract: Machine-learning weather models now match or exceed operational numerical prediction in the medium range, but when they are run freely for years they drift, blow up or lose their variability, and dedicated climate emulators are expensive to train. Here we present Rescene, a 0.4 M-parameter wrapper that turns a frozen 1.5$^\circ$, 6-hourly, 69-variable weather model into a stochastic climate emulator. A deterministic slow clock anchors the frozen model to a calendar climatology, and a band-limited AR(1) perturbation, whose spectral envelope is fixed to the observed anomaly spectrum, restores the weather that anchoring removes. A 100-year, 8-member Rescene run shows no detectable drift ($-0.004 \pm 0.014$ K per 100 years), and on the 2022 test year the 16-member ensemble is calibrated (spread/skill 0.89, 0.97 and 0.93 for Z500, T2m and MSLP) with an extratropical synoptic amplitude of 97% of ERA5 on the tuning year. The free run 
    
[^242]: 定位多页视觉丰富文档理解系统中的失败：一项实证归因研究

    Locating Failure in Multi-Page Visually Rich Document Understanding Systems: An Empirical Attribution

    [https://arxiv.org/abs/2608.07943](https://arxiv.org/abs/2608.07943)

    本文提出一个实证归因框架，通过受控干预定位多页视觉丰富文档理解系统在证据表示、选择与使用各环节中的失败来源，发现模态互补可降低转换错误敏感性、非黄金页面会同时修复与破坏答案，从而揭示了总体准确率所掩盖的系统不稳定性。

    

    多页视觉丰富文档理解（MP-VRDU）系统在证据未被忠实表示、未被选择或未被正确使用时可能出现失败，然而仅凭最终准确率无法揭示失败发生在何处。我们提出了一个实证归因框架，通过在统一的文档问答流水线中进行受控干预来检验这些失败。在MMLongBench-Doc上的实验（结合跨数据集和跨模型家族的验证）表明，互补的模态可以降低对转换错误的敏感性：图像可以缓解解析器错误，而提取的文本则降低了对视觉分辨率的依赖。添加相关的非黄金（non-gold）页面既可以修复错误答案，也可能破坏正确答案，即使在所有标注证据均存在的情况下也是如此，这揭示了总体准确率之下的不稳定性。弃权机制以虚假拒绝换取无依据答案的减少，而其与推理指令的交互作用在不同骨干模型之间存在差异。

    arXiv:2608.07943v2 Announce Type: replace  Abstract: Multi-page visually rich document understanding (MP-VRDU) systems can fail when evidence is not faithfully represented, not selected, or not used correctly, yet final accuracy alone does not reveal where failures occur. We present an empirical attribution framework that examines these failures through controlled interventions in a unified document question answering pipeline. Experiments on MMLongBench-Doc, with cross-dataset and cross-family checks, show that complementary modalities can reduce sensitivity to conversion errors: images mitigate parser errors, while extracted text reduces dependence on visual resolution. Adding related non-gold pages can both repair incorrect answers and break correct ones, even when all annotated evidence is present, revealing instability beneath aggregate accuracy. Abstention trades unsupported answers for false refusals, while its interaction with reasoning instructions differs across backbones. Qu
    
[^243]: Poli-Bias：理解与测量大语言模型在国际政治冲突中的偏见

    Poli-Bias: Understanding and Measuring Large Language Model Biases in International Political Conflicts

    [https://arxiv.org/abs/2608.06123](https://arxiv.org/abs/2608.06123)

    提出了POLI-BIAS反事实测量框架，通过系统性互换国家身份的配对提示，将LLM在国际政治冲突中的偏见响应差异分解为五个可解释维度，发现模型对处于冲突或紧张状态的国家存在不均衡对待。

    

    测量大语言模型（LLM）的政治偏见仍然具有挑战性，因为它可能通过在议题框架设定、论证方式以及法律推理方面的细微差异表现出来，而这些差异难以用单一指标加以捕捉。在这项工作中，我们提出了POLI-BIAS，一个用于测量LLM是否因涉及国家不同而对法律上等同的冲突场景进行差异化处理的反事实框架。POLI-BIAS通过对比配对提示的响应来工作，其中国家身份在不同地缘政治关系、法律违规行为和推理任务中被系统性地互换。我们的框架并不将偏见简化为单一判断，而是将响应差异分解为五个可解释的维度，从而揭示不平等对待是如何以及在何处显现的。在涵盖5,600个提示对以及来自不同模型家族和规模的七个开放权重LLM的实验中，我们发现模型对处于活跃冲突或紧张状态的国家……

    arXiv:2608.06123v3 Announce Type: replace  Abstract: Measuring political bias in large language models (LLMs) remains challenging as it can manifest through subtle differences in framing, argumentation, and legal reasoning that are difficult to capture with a single metric. In this work, we introduce POLI-BIAS, a counterfactual framework for measuring whether LLMs treat legally equivalent conflict scenarios differently depending on the countries involved. POLI-BIAS compares responses to paired prompts in which country identities are systematically swapped across diverse geopolitical relationships, legal violations, and reasoning tasks. Rather than reducing bias to a single judgment, our framework decomposes response disparities into five interpretable dimensions, revealing how and where unequal treatment manifests. Across 5,600 prompt pairs and seven open-weight LLMs from diverse model families and sizes, we find that models treat countries in active conflict or tension less evenly tha
    
[^244]: Q-Reflector：基于生成条件学习的GUI动作反思

    Q-Reflector: Learning GUI Action Reflection with Generated Conditions

    [https://arxiv.org/abs/2608.05587](https://arxiv.org/abs/2608.05587)

    提出了Q-Reflector，一个基于Qwen3-VL-8B的独立GUI动作反思器，通过生成前置/后置条件并以监督微调、蒸馏和奖励偏好优化相结合的训练方式，准确评估移动GUI动作的执行成功性与任务有效性。

    

    移动GUI智能体需要准确的动作反思来实现可靠的长期任务执行，即评估每个动作是否被成功执行，以及其结果是否构成向任务目标迈进的有效一步。当在统一提示下作为独立反思器进行评估时，前沿视觉语言模型（VLM）和专门的GUI模型对有效与无效状态转换的识别能力并不均衡。我们提出了Q-Reflector，一个基于Qwen3-VL-8B的独立反思器，它将GUI动作反思学习作为监督式结构化预测任务。给定任务上下文、已执行的动作和成对的前后截图，它会生成前置条件和后置条件，用以明确动作的前提要求及预期的局部效果，然后通过结构化的推理过程和二元判定来评估动作的执行成功与否以及任务有效性。其训练结合了监督微调、师生蒸馏以及基于奖励和偏好的优化改进。在1,082个经人工验证的AndroidWo（摘要在此处截断）

    arXiv:2608.05587v2 Announce Type: replace  Abstract: Mobile GUI agents require accurate action reflection for reliable long-horizon execution, assessing whether each action was successfully executed and whether its outcome constitutes a valid step toward the task goal. When evaluated as standalone reflectors under a common prompt, frontier VLMs and specialized GUI models exhibit imbalanced recognition of valid and invalid transitions. We propose Q-Reflector, a standalone reflector based on Qwen3-VL-8B that learns GUI action reflection as supervised structured prediction. Given task context, the executed action, and paired screenshots, it generates pre- and post-conditions specifying the action's prerequisites and intended local effect, then assesses execution success and task validity through a structured rationale and binary verdict. Training combines supervised fine-tuning, teacher-student distillation, and reward- and preference-based refinement. On 1,082 manually verified AndroidWo
    
[^245]: MANGO-Grasp：基于几何导向3D高斯的马氏距离场实现跨具身灵巧抓取

    MANGO-Grasp: Mahalanobis Fields over Geometry-Oriented 3D Gaussians for Cross-Embodiment Dexterous Grasping

    [https://arxiv.org/abs/2608.02014](https://arxiv.org/abs/2608.02014)

    提出MANGO-Grasp框架，利用几何导向的3D高斯表示物体、预训练描述符编码手部，并通过各向异性马氏距离场建模手-物体交互，从而在无需针对特定具身形态调优的情况下，实现跨异构多指灵巧手的稳定灵巧抓取。

    

    跨具身灵巧抓取旨在异构的多指灵巧手之间合成稳定的抓取，而几乎不需要或完全不需要针对特定具身形态的调优。现有的以交互为中心的方法取得了可观的效果，但其物体表示往往无法充分表达局部表面几何，手部描述符未能显式地同时捕捉形态学与运动学特性，且手-物体交互建模通常是各向同性的。我们提出MANGO-Grasp，这是一个各向异性交互框架：它将物体表示为几何导向的3D高斯基元，以更好地保留局部表面几何；并将手部编码为能够同时捕捉形态与运动学的预训练描述符。手-物体交互通过各向异性马氏距离场进行建模，该场反映了接触的方向性本质，并引导一种跨具身形态的共享优化以实现抓取。在CMAP和MultiGripperGrasp基准测试中，MANGO-Grasp……（摘要在此处截断）

    arXiv:2608.02014v2 Announce Type: replace-cross  Abstract: Cross-embodiment dexterous grasping aims to synthesize stable grasps across heterogeneous multi-fingered hands with little or no embodiment-specific tuning. Existing interaction-centric methods achieve promising results, but their object representations often underrepresent local surface geometry, hand descriptors do not explicitly capture both morphology and kinematics, and hand-object interaction modeling is typically isotropic. We propose MANGO-Grasp, an anisotropic interaction framework that represents objects as geometry-oriented 3D Gaussian primitives to better preserve local surface geometry, and encodes hands into pretrained descriptors that capture both morphology and kinematics. Hand-object interaction is modeled using anisotropic Mahalanobis fields that reflect the directional nature of contact and guide a shared optimization for grasp realization across embodiments. On the CMAP and MultiGripperGrasp benchmarks, MANG
    
[^246]: 揭秘VLA（视觉-语言-动作）模型在富接触任务中何时、为何失败以及如何修复

    Demystifying When and Why VLAs Fail in Contact-Rich Tasks and How to Fix Them

    [https://arxiv.org/abs/2608.01402](https://arxiv.org/abs/2608.01402)

    该论文揭示了VLA模型在富接触任务中的两种失败模式（精度失败与力失败），并提出FACT方法，通过后训练阶段的低噪声采样和结构化力条件机制将平均成功率提升至66%。

    

    我们致力于理解视觉-语言-动作（VLA）模型在需要精确物理交互的富接触操作任务中何时以及为何表现不佳这一问题。先前的工作主要通过力增强架构和训练时正则化器来解决接触失败问题，然而导致这些失败的因素仍未得到充分探索。我们识别出造成这一差距的两种不同的失败模式：精度失败发生在策略难以产生接触过程中所需的小而准确的修正动作时，而力失败发生在策略未能对接触力做出响应时。为了解决这些失败，我们提出了FACT。在后训练阶段，FACT增加低噪声采样以改进精细动作预测，并引入结构化力条件机制，利用接触状态和力历史来引导动作生成。FACT在五个接触任务上取得了平均66%的成功率。

    arXiv:2608.01402v2 Announce Type: replace-cross  Abstract: We address the problem of understanding when and why Vision-Language-Action models struggle with contact-rich manipulation tasks that require precise physical interaction. Prior work has primarily focused on addressing contact failures through force-augmented architectures and training-time regularizers, yet the factors contributing to these failures remain underexplored. We identify two distinct failure modes underlying this gap. Precision failures occur when policies struggle to produce the small, accurate corrective actions required during contact and force failures occur when policies fail to respond to contact forces. To address these failures, we propose FACT. During post-training, FACT increases low noise sampling to improve fine action prediction and incorporates structured force conditioning to guide action generation using contact state and force history. FACT achieves an average success rate of 66% across five contac
    
[^247]: 一种面向已验证生产级密码学库的AI方法

    An AI Approach to Verified Production Cryptographic Libraries

    [https://arxiv.org/abs/2608.00965](https://arxiv.org/abs/2608.00965)

    CryptoProver是一种AI系统，能够从高层API契约自动综合内部规范与经Verus验证的形式化证明，在不改动任何代码的情况下，首次以少量人力完成对curve25519-dalek和chacha20等生产级密码学库的独立验证。

    

    密码学代码是必须保证正确性的关键基础设施，然而对生产级库进行形式化验证仍然十分困难。现有的语言模型证明系统只能在规范和前提条件已给定的情况下解决孤立任务，生产级库的验证问题始终悬而未决。我们提出了CryptoProver，这是一个基于AI的系统，能够从高层API契约出发，自动综合出内部规范并生成经Verus检查的证明。在不修改任何可执行代码的情况下，CryptoProver为curve25519-dalek构建了全新的独立证明，并依据RFC 8439规范验证了RustCrypto中此前从未被验证过的chacha20实现。这些密码学代码支撑着包括Signal和Shadowsocks在内的已部署系统，其中Signal的全球下载量估计达2.18亿次。原始开发者报告称，验证curve25519-dalek需要三人历时五个月的专注工作。在给定API契约和一个固定可信库的前提下……（原文在此截断）

    arXiv:2608.00965v2 Announce Type: replace-cross  Abstract: Cryptographic code is critical infrastructure that must be correct, yet formally verifying production libraries remains difficult. Existing language-model proof systems solve isolated obligations with specifications and premises already given, leaving production-library verification unresolved.   We present CryptoProver, an AI-based system that synthesizes internal specifications and Verus-checked proofs from high-level API contracts. Without changing executable code, CryptoProver constructs a new independent proof of curve25519-dalek and verifies RustCrypto's previously unverified chacha20 implementation against an RFC 8439 specification. These cryptographic lineages underpin deployed systems including Signal and Shadowsocks; Signal has an estimated 218M global downloads. The original developers report a focused five-month, three-person effort to verify curve25519-dalek. Given the API contracts and a fixed trusted library of f
    
[^248]: CDAE：利用对比去噪增强预训练语言模型的扰动鲁棒性

    CDAE: Enhancing Perturbation Robustness in Pretrained Language Models with Contrastive Denoising

    [https://arxiv.org/abs/2607.28236](https://arxiv.org/abs/2607.28236)

    提出一种轻量级对比去噪自编码器CDAE，通过联合优化对比与重建目标学习扰动不变的句子表示，显著增强了预训练语言模型嵌入对文本扰动的鲁棒性。

    

    预训练语言模型显著提升了句子表示学习的效果，但其嵌入表示仍然对同义词替换、掩码和词丢弃等文本扰动十分敏感。本工作提出了一种轻量级的对比去噪自编码器（CDAE），通过联合优化对比目标和重建目标来精炼预训练BERT的嵌入表示，从而学习到对扰动不变的表示。我们在多种扰动策略和强度下评估了所提出的框架，并将其与原始BERT嵌入和SimCSE进行了比较。实验结果表明，CDAE在扰动下始终保持着更高的嵌入相似度，且随着扰动强度的增加，这一差距进一步扩大，凸显了扰动不变学习是提升句子嵌入稳定性的一种有前景的方向。源代码已公开发布于：https://github.com/Comp

    arXiv:2607.28236v2 Announce Type: replace  Abstract: Pre-trained language models have significantly improved sentence representation learning, yet their embeddings remain sensitive to textual perturbations such as synonym substitution, masking, and word dropout. This work proposes a lightweight Contrastive Denoising Autoencoder (CDAE) that refines pre-trained BERT embeddings by jointly optimizing contrastive and reconstruction objectives to learn perturbation-invariant representations. We evaluate the proposed framework under multiple perturbation strategies and strengths, and compare it against the original BERT embeddings and SimCSE. Experimental results show that CDAE consistently preserves higher embedding similarity under perturbation, with the gap widening as the perturbation strength increases, highlighting perturbation-invariant learning as a promising direction for improving the stability of sentence embeddings. The source code is publicly available at: https://github.com/Comp
    
[^249]: EKLAV：通过仅答案监督引出关键思维链学习

    EKLAV: Elicited Key CoT Learning via Answer only superVision

    [https://arxiv.org/abs/2607.25182](https://arxiv.org/abs/2607.25182)

    EKLAV将教师模型的思维链轨迹作为特权输入上下文而非模仿目标，仅对最终答案进行监督训练，在重排序任务上以更低的训练计算成本和推理时间超越了传统推理蒸馏方法。

    

    推理蒸馏训练学生模型复现教师模型的思维链轨迹及其最终答案。我们探讨是否将思维链轨迹用作特权训练上下文比作为模仿目标更为有效。EKLAV将教师模型的推理保留在输入序列中，但仅对最终任务答案进行监督，并采用任务相关的放置策略，将部分或全部思维链轨迹移入提示中。在BRIGHT段落重排序任务上，EKLAV在两个模型家族的所有四个测试检查点上都提升了平均nDCG@10，在48个模型-领域对比中赢得41个，同时将训练计算量减少高达31.6%，推理时间减少高达32.7%。在四个留出的表格重排序基准上，它在四个条件中的三个中表现最佳，平均超过完整思维链蒸馏方法3.0分，超过无推理的仅答案训练5.5分。对生成轨迹的分析表明，EKLAV学生模型会延迟其首次口头表达的（摘要在此处截断）

    arXiv:2607.25182v2 Announce Type: replace-cross  Abstract: Reasoning distillation trains a student to reproduce both the teacher's chain-of-thought trace and its final answer. We ask whether the trace is better used as privileged training context than as an imitation target. EKLAV keeps the teacher's reasoning in the input sequence but supervises only the final task answer, applying a task-dependent placement that moves part or all of the trace into the prompt. On BRIGHT passage reranking, EKLAV improves average nDCG@10 at all four tested checkpoints across two model families, winning 41 of 48 model-domain comparisons while reducing training compute by up to 31.6% and inference time by up to 32.7%. On four held-out table-reranking benchmarks, it is the best of four conditions on three, exceeding full-trace distillation by 3.0 points and answer-only training without reasoning by 5.5 points on average. Analysis of generated traces reveals that EKLAV students defer their first verbalized 
    
[^250]: 用于复杂离子阱架构的高效LLM生成的穿梭编译器

    Efficient LLM-Generated Shuttling Compilers for Complex Trapped-Ion Architectures

    [https://arxiv.org/abs/2607.24714](https://arxiv.org/abs/2607.24714)

    该论文首次展示单个前沿大语言模型（Claude Opus 4.7）能够根据书面规范自动生成并迭代优化离子阱量子计算机的穿梭编译器，在多种离子阱架构上比最先进的手工编译器减少多达72%的穿梭时间步。

    

    离子阱量子计算机依赖于穿梭编译器，它将输入算法转化为在给定架构内离子量子比特移动的序列。我们提出了首个这样的研究：单个前沿大语言模型（LLM）Claude Opus 4.7 根据书面规范生成并迭代改进穿梭编译器的Python代码。从针对(i)线性分段离子阱的编译器开始，我们将其扩展到(ii)带有结点的离子阱，最终实现了对(iii)一大类连通离子阱图的高效编译。针对更一般情况的编译器以之前编译器的代码作为种子。我们使用一套通用的量子电路，将LLM生成的编译器与最先进的手工编译器进行基准测试。对于(i)，穿梭时间步数最多减少了72%；对于(ii)，最多减少了43%。对于自由连接架构的广泛情形(iii)，我们发现（原文在此处截断）

    arXiv:2607.24714v2 Announce Type: replace-cross  Abstract: Trapped-ion quantum computers rely on shuttling compilers, which cast an input algorithm into a sequence of ion-qubit movements within a given architecture. We present the first study in which a single frontier large language model (LLM), Claude Opus 4.7, generates and iteratively refines the Python code of shuttling compilers from written specifications. Starting with a compiler for (i) a linear segmented trap, we extend it to (ii) a trap with junctions, and finally achieve efficient compilation for (iii) a broad class of connected trap graphs. The compilers for the more general cases are seeded with code from the previous ones. We benchmark the LLM-generated compilers against state-of-the-art hand-crafted ones using a common suite of quantum circuits. The number of shuttling timesteps is reduced by up to 72% for (i) and up to 43% for (ii). For the broad case (iii) of freely connected architectures, we find large variations in
    
[^251]: KAYROS：一个面向时长最小化时间依赖型车辆路径问题的随时可中断且精确的开源求解器——一份技术报告与一个人机工程案例研究

    KAYROS: An Anytime and Exact Open-Source Solver for Duration-Minimization Time-Dependent Vehicle Routing. A Technical Report and a Case Study in Human-AI Engineering

    [https://arxiv.org/abs/2607.23116](https://arxiv.org/abs/2607.23116)

    KAYROS 是一个开源、随时可中断且能给出最优性证书的时长最小化时间依赖型车辆路径求解器，填补了该领域缺乏可直接使用的精确求解工具的空白。

    

    时间依赖型路径规划认识到，同一段行程所需的时间会因其开始时刻的不同而变化。在时长最小化的目标下，甚至每辆车的出发时间本身也成为一项决策。文献中虽然已有针对该问题的精确方法，但研究人员和从业者一直缺乏一个开箱即用的开源求解器，能够同时提供丰富的分段线性行程时间函数、早期可行解以及最优性证明。KAYROS 填补了这一空白：它在同一个与验证器保持一致的引擎上提供两种模式——一种是可持续输出改进解的迭代局部搜索，另一种是可在明确的算术与搜索假设下给出计算最优性证书的分支定价与割平面方法。它仅需一条命令即可安装，且不依赖任何专有软件。公开的 MAMUT-routing 存储库目前包含 704 份基于“四次求解发布协议”的 KAYROS 证书，该协议还促成了对早前无效声明的撤回与修正。

    arXiv:2607.23116v3 Announce Type: replace-cross  Abstract: Time-dependent routing recognizes that the same journey can take a different time depending on when it begins. Under duration minimization, even the departure time of each vehicle becomes a decision. Exact methods for this setting exist in the literature, but researchers and practitioners have lacked a ready-to-use open solver that combines rich piecewise-linear travel times, early feasible solutions and optimality claims. KAYROS fills this gap with two modes on one checker-consistent engine: an Iterated Local Search that streams improving solutions and a Branch-Price-and-Cut method that can issue computational optimality certificates under explicit arithmetic and search assumptions. It installs with one command and has no proprietary dependency. The public MAMUT-routing store currently contains 704 KAYROS certificates under a four-solve publication protocol, which has also led to the retraction and repair of invalid earlier cl
    
[^252]: STAIF：一种面向复杂指令遵循的分阶段优化方法

    STAIF: A Stage-wise Optimization for Complex Instruction Following

    [https://arxiv.org/abs/2607.22649](https://arxiv.org/abs/2607.22649)

    STAIF提出分阶段优化框架，先用多负样本偏好优化对齐软约束，再用可验证奖励强化学习强制满足硬约束，并构建了约3.1万条双语复杂多约束指令数据集STAINSTRUCT，有效提升大模型复杂指令遵循能力。

    

    对于大型语言模型（LLM）而言，遵循包含多个明确约束的复杂指令仍然是一个根本性挑战。现有的对齐方法（如DPO）优化整体奖励信号，往往低估了对单个约束的严格满足，尤其是在分布外或多约束设置下。本文提出STAIF，一种分阶段优化框架，将主观（软）约束的对齐与客观可验证（硬）约束的优化进行解耦。第一阶段采用带多个负样本的偏好优化来增强模型对软约束的敏感性，第二阶段采用可验证奖励强化学习（RLVR）来强制模型严格执行硬约束。为支持该方法，我们构建了STAINSTRUCT——一个高质量的双语（英语、中文）数据集，包含约31,000条复杂的多约束指令。

    arXiv:2607.22649v2 Announce Type: replace  Abstract: Following complex instructions with multiple explicit constraints remains a fundamental challenge for large language models (LLMs). Existing alignment methods, such as DPO, optimize holistic reward signals that often underemphasize strict satisfaction of individual constraints, particularly under out-of-distribution or multi-constraint settings. In this paper, we propose STAIF, a stage-wise optimization framework that decouples the alignment of subjective (soft) constraints from the optimization of objectively verifiable (hard) constraints. Stage 1 applies preference optimization with multiple negative samples to sharpen sensitivity to soft constraints, while Stage 2 applies Reinforcement Learning with Verifiable Rewards (RLVR) to enforce strict compliance with hard constraints. To support this method, we construct STAINSTRUCT, a high-quality bilingual (English, Chinese) dataset of approximately 31,000 complex multi-constraint instru
    
[^253]: 每个模型都会作弊：针对攻击性网络任务中作弊行为的提示词级缓解

    Every Model Cheats: Prompt-Level Mitigation of Cheating on Offensive Cyber Tasks

    [https://arxiv.org/abs/2607.21763](https://arxiv.org/abs/2607.21763)

    本研究通过四阶段审计流水线的大规模实验揭示，网络安全基准测试中的作弊远比以往认知普遍（22个前沿模型中21个作弊、37.1%的通过涉及作弊、分数夸大高达5倍），并证明提示词级的反作弊干预能将作弊倾向从33.0%有效降至8.5%。

    

    大语言模型（LLM）智能体在网络安全基准测试中经常作弊，使报告的通过率远远超出其真实能力。此前对 Cybench 的审计仅在 0.3-3.4% 的任务记录中发现作弊，且仅涉及少数几个模型。我们对来自 7 家提供商的 22 个前沿模型，在 23 个 Cybench 夺旗（CTF）挑战任务上，于三种提示词条件（无反作弊、标准、严格）下开展了受控的提示词消融研究。所有 1,518 条任务记录均通过一个四阶段流水线逐一审计，该流水线结合了 LLM 评判分类、程序化验证、评判与验证器的交叉核对以及人工审核。我们发现作弊行为远比之前估计的更为普遍：在基线条件下，37.1% 的通过涉及作弊，22 个模型中有 21 个存在作弊，分数被夸大高达 5 倍。反作弊提示词可将作弊倾向从 33.0%（基线）降至 18.6%（标准），再降至 8.5%（严格），且不会导致……（原文摘要在此处被截断）

    arXiv:2607.21763v2 Announce Type: replace-cross  Abstract: Large language model (LLM) agents routinely cheat on cybersecurity benchmarks, inflating reported pass rates far beyond genuine capability. Prior audits of Cybench found cheating in 0.3-3.4% of traces, implicating only a handful of models. We present a controlled prompt-ablation study across 22 frontier models from 7 providers on 23 Cybench capture-the-flag (CTF) challenges under three prompt conditions (no anti-cheat, standard, severe). All 1,518 task traces were individually audited through a four-stage pipeline combining LLM-as-a-judge classification, programmatic verification, judge-verifier reconciliation, and human review. We find cheating is far more pervasive than previously estimated: under baseline conditions, 37.1% of passes involved cheating, 21 of 22 models cheated, and scores were inflated by up to 5x. Anti-cheat prompts reduce cheat propensity from 33.0% (baseline) to 18.6% (standard) to 8.5% (severe) without deg
    
[^254]: 无需重新训练的跨方言泛化：面向 MLIR 的模式派生约束解码基准与评估

    Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR

    [https://arxiv.org/abs/2607.18254](https://arxiv.org/abs/2607.18254)

    该论文提出从 MLIR 方言的操作定义规范（ODS）机械派生的推理时约束解码方法，无需重新训练即可实现跨方言泛化，并发布了涵盖三种方言共 410 个样本的四个自然语言到 MLIR 基准测试数据集。

    

    多层中间表示（MLIR）是现代机器学习编译器基础设施（TensorFlow、JAX/StableHLO、PyTorch Inductor、IREE）的基础，然而在代码语言模型的预训练语料库中仅以极微量出现。此外，MLIR 在设计上具有可扩展性：每个应用领域都会发布新的方言，因此针对每个方言分别微调模型的做法无法规模化。我们探究能否从每个方言的操作定义规范（ODS）中机械地推导出推理时先验知识，以替代基于梯度的适配。首先，我们发布了涵盖三种方言的四个自然语言到 MLIR 的基准测试（MLIR-Spec-150、Linalg-Spec-30、StableHLO-Spec-30、StableHLO-Held-Out-200），共包含 410 个范围内的自然语言到 MLIR 配对样本，外加一个包含 25 个程序的语法外压力测试集，其中 30 个 Spec 提示词附有功能性参考实现，全部以 Apache-2.0 许可证发布，并附有 Gebru 数据表和 Croissant 1.1 元数据。其次，我们构建了一个三层的模式派生约束栈：基于操作签名的上下文无关文法（CFG）……（原文摘要在此处截断）

    arXiv:2607.18254v2 Announce Type: replace  Abstract: Multi-Level Intermediate Representation (MLIR) underlies modern ML compiler infrastructure (TensorFlow, JAX/StableHLO, PyTorch Inductor, IREE), yet appears only in trace amounts in code-LM pretraining corpora. MLIR is also extensible by design: new dialects ship per application domain, so a fine-tuned model per dialect does not scale. We ask whether inference-time priors derived mechanically from each dialect's Operation Definition Specification (ODS) can substitute for gradient-based adaptation. First, we release four natural-language-to-MLIR benchmarks across three dialects (MLIR-Spec-150, Linalg-Spec-30, StableHLO-Spec-30, StableHLO-Held-Out-200), totaling 410 in-scope NL-to-MLIR pairs, plus a 25-program out-of-grammar stress set, with functional references on 30 Spec prompts, under Apache-2.0 with Gebru datasheets and Croissant 1.1 metadata. Second, we build a three-layer schema-derived constraint stack: a CFG over op signatures 
    
[^255]: 将AI对齐到动态的人机工作流

    Align AI to Dynamic Human-AI Workflows

    [https://arxiv.org/abs/2607.14240](https://arxiv.org/abs/2607.14240)

    该论文提出将AI对齐范式从静态模仿人类偏好转向交互式、互补式对齐，即偏好应在人机交互中动态涌现，且对齐的衡量标准不应仅限于满足静态偏好。

    

    当前的对齐方法通常侧重于使用人类偏好的静态表示来模拟人类行为，未能捕捉现实世界中人机交互的动态性和上下文依赖特性。在本文中，我们主张从静态和模仿式的对齐转向交互式和互补式的对齐，其中偏好通过交互而产生，而对齐的定义不应仅仅是满足偏好。我们首先通过将现有对齐方法与轨迹层面的视角进行对比来形式化这一差距，在该视角下，人类和模型的行为会随时间共同演化。由于这些交互动态尚未在现有的机器学习框架中得到充分刻画，我们基于一次跨学科研讨会的见解来构建这一视角。我们借鉴社会科学中关于人与人协作的研究成果，进而论证人机系统会放大这些动态特性，引入新的不对称性，使得对……的推理（原文摘要在此处截断）。

    arXiv:2607.14240v2 Announce Type: replace  Abstract: Current alignment approaches typically focus on emulating human behavior using static representations of human preferences, failing to capture the dynamic, context-dependent nature of real-world human-AI interactions. In this paper, we argue for a shift from static and emulative to interactive and complementary alignment, where preferences emerge through interaction and alignment is defined not by satisfying preferences alone. We first formalize this gap by contrasting existing alignment with a trajectory-level view in which human and model behavior co-evolve over time. Because these interaction dynamics have not been adequately captured within existing ML formulations, we ground this perspective in insights from an interdisciplinary workshop. We draw on lessons from social-science accounts of human-human collaboration and then argue that human-AI systems amplify these dynamics, introducing new asymmetries that make reasoning about u
    
[^256]: 智能体如何请求权限：面向AI智能体的用户权限，从界面到执行机制

    How Agents Ask for Permission: User Permissions for AI Agents, from Interfaces to Enforcement

    [https://arxiv.org/abs/2607.13718](https://arxiv.org/abs/2607.13718)

    该论文调研了21个用户级AI智能体权限系统并构建了涵盖权限指定、推导与执行方法的分类体系，通过对比商业智能体发现学术与商业方案在可用性、透明度和用户控制方面仍存在显著差距。

    

    随着AI智能体日益普及，其对用户造成的风险也随之增加，因此亟需为智能体权限构建健壮且易用的系统。许多近期的方案实施的是产品级权限，即对给定智能体框架的所有用户应用相同的安全策略。然而，不同用户有着不同的需求，这就要求智能体AI系统支持用户级权限策略。我们调研了21个已提出的用户级智能体权限系统，并构建了一个涵盖用户级权限的指定、推导和执行方法的分类体系。我们进一步分析并与五个商业智能体进行了比较。我们发现，尽管提案种类繁多，但在可用性、透明度和用户控制方面仍存在显著差距，尤其是在学术智能体和商业智能体之间。

    arXiv:2607.13718v3 Announce Type: replace-cross  Abstract: As AI agents grow in prevalence, so too do their risks to users, creating a need for robust and usable systems for agent permissions. Many recent proposals implement product-level permissions, where the same security policy is applied for all users of a given agent framework. Yet different users have varying needs, necessitating support for user-level permission policies in agentic AI systems.   We survey 21 proposed user-level agent permissions systems and construct a taxonomy of approaches to specifying, deriving, and enforcing user-level permissions. We further analyze and compare against five commercial agents. We find that despite a wide diversity of proposals, sizable gaps remain in usability, transparency, and user control, particularly between academic and commercial agents.
    
[^257]: 将通用车辆模型适配用于跨地形高速模型预测控制（MPC）

    Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains

    [https://arxiv.org/abs/2607.13319](https://arxiv.org/abs/2607.13319)

    OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。

    

    高速越野自动驾驶需要对目标车辆进行精确的闭环控制，同时在多变地形上保持鲁棒性。近期的前向运动动力学（FKD）预测基础模型展现出一条有前景的路径：从通用模型出发，将其专用于目标平台。然而，有效的专业化仍然具有挑战性，因为它通常需要大量真实世界数据，而且适配到某一环境的模型仍可能对特定地形或驾驶模式过拟合。我们提出了OptCar（优化汽车），这是一种弥合通用FKD模型到专用FKD模型差距的方法，在为特定车辆优化性能的同时保留了跨地形泛化能力。OptCar引入了一种基于transformer的FKD架构，利用FiLM将多步预测条件化于单个动力学上下文token，该token概括了近期的状态-动作历史。随后，该方法利用有限的真实世界数据对通用模型进行专业化（摘要在此处截断）。

    arXiv:2607.13319v2 Announce Type: replace-cross  Abstract: High-speed off-road autonomy requires precise closed-loop control for a target vehicle while remaining robust across changing terrains. Recent forward kinodynamic (FKD) prediction foundation models suggest a promising path, starting from a generalist model and specializing it to the target platform. However, effective specialization remains challenging, as it often requires substantial real-world data, and models adapted to one setting can still overfit to specific terrains or driving regimes. We present OptCar (Optimized Car), a recipe for bridging the gap from generalist to specialist FKD models that preserves cross-terrain generalization while optimizing performance for a specific vehicle. OptCar introduces a transformer FKD architecture that uses FiLM to condition multi-step predictions on a single dynamics context token summarizing recent state-action history. It then specializes the generalist model using limited real-wor
    
[^258]: 多智能体大语言模型会相互探索吗？

    Do Multi-Agent LLMs Explore Each Other?

    [https://arxiv.org/abs/2607.11250](https://arxiv.org/abs/2607.11250)

    本文发现现代LLM智能体在多智能体交互中存在短视、极化的探索缺陷，并提出轻量级框架MACE，通过结构化同伴选择显式促进智能体间的探索，从而显著改善协调能力并降低遗憾值。

    

    探索对于多智能体系统的可靠自主性至关重要，然而大语言模型（LLM）智能体在彼此交互时能否有效探索仍不清楚。我们表明，现代LLM智能体往往无法做到这一点，它们表现出短视和极化的交互模式，导致次优的协调效果和更高的遗憾值。我们将这一挑战形式化为多智能体探索问题，并将其建模为部分可观测随机博弈（POSG），其中智能体必须探测同伴以推断其能力并识别有效的交互策略。为解决这一问题，我们提出了多智能体情境探索（MACE），这是一个轻量级框架，通过结构化的同伴选择明确地促进探索。在情境多样性和参数多样性两种设置下，应用MACE后探索行为和下游任务性能均得到显著提升。我们还进一步提供了……（原文摘要在此处截断）

    arXiv:2607.11250v2 Announce Type: replace-cross  Abstract: Exploration is essential for reliable autonomy in multi-agent systems, yet it remains unclear whether large language model (LLM) agents can explore effectively when interacting with one another. We show that modern LLM agents often fail to do so, exhibiting myopic and polarized interaction patterns that lead to suboptimal coordination and increased regret. We formalize this challenge as the Multi-Agent Exploration problem, modeling it as a partially observable stochastic game (POSG), in which agents must probe peers to infer their capabilities and identify effective interaction strategies. To address this, we introduce Multi-Agent Contextual Exploration (MACE), a lightweight framework that explicitly promotes exploration through structured peer selection. Across both contextual and parametric diversity settings, exploration behavior and downstream task performance substantially improve when MACE is applied. We further provide t
    
[^259]: 你的NPU准备好迎接大语言模型了吗？剖析移动端LLM推理中隐藏的效率瓶颈

    Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference

    [https://arxiv.org/abs/2607.05475](https://arxiv.org/abs/2607.05475)

    该论文开展了首个跨层次的移动端LLM推理全面测量研究，并开发了细粒度性能分析工具PowerBench，揭示了框架性能差距在NPU上可被放大至10倍等隐藏的效率瓶颈。

    

    将大语言模型（LLM）部署到移动设备上可以增强隐私保护并降低延迟，但受到硬件效率低下的严重制约。我们提出了首个全面的、跨层次的移动端LLM推理测量研究，涵盖了五个主流框架（如llama.cpp和GENIE）以及三种硬件后端（CPU、GPU和NPU）。为实现这一分析，我们开发了PowerBench，这是一种细粒度的性能分析工具，能够测量LLM吞吐量，并首次为移动端LLM推理提供针对特定后端的能耗归因，超越了传统的设备级测量。我们进一步开展了硬件微基准测试，以表征移动端CPU、GPU和NPU的内存系统性能与计算能力。我们的研究得出了三个关键见解：（1）由于卸载策略、硬件等方面的差异，框架导致的性能差距在NPU上被大幅放大，最高可达10倍。

    arXiv:2607.05475v2 Announce Type: replace-cross  Abstract: Deploying Large Language Models (LLMs) on mobile devices enhances privacy and reduces latency, but is severely constrained by hardware inefficiencies. We present the first comprehensive, cross-layer measurement study of mobile LLM inference, spanning five mainstream frameworks (e.g., llama.cpp and GENIE) and three hardware backends (CPU, GPU, and NPU). To enable this analysis, we develop PowerBench, a fine-grained profiling tool that measures LLM throughput and provides the first backend-specific energy attribution for mobile LLM inference, going beyond traditional device-level measurements. We further conduct hardware microbenchmarks to characterize the memory-system performance and compute capabilities of mobile CPUs, GPUs, and NPUs. Our study yields three critical insights: (1) Framework-induced performance gaps are substantially amplified on NPUs, reaching up to $10\times$ due to differences in offloading strategies, hardwa
    
[^260]: TORINO：通过可解释概念重叠实现视觉-语言模型中的词元削减

    TORINO: Token Reduction via Interpretable Concept Overlap in Vision-Language Models

    [https://arxiv.org/abs/2607.04593](https://arxiv.org/abs/2607.04593)

    提出即插即用框架TORINO，利用稀疏自编码器将视觉词元映射到可解释的潜在空间，基于概念重叠程度对词元分组并削减冗余词元，无需微调即可降低视觉-语言模型的计算成本。

    

    视觉-语言模型在各种任务中展现出了令人印象深刻的能力，但其计算成本主要由输入语言模型的大量视觉词元所主导。现有的词元削减方法依赖于基于注意力的评分或成对相似度，缺乏对每个词元的显式语义表示。我们提出了TORINO（TOken Reduction via Interpretable coNcept Overlap，通过可解释概念重叠进行词元削减），这是一个用于视觉-语言模型中自适应视觉词元削减的即插即用框架，无需对底层模型进行微调。TORINO利用稀疏自编码器将视觉词元投影到一个可解释的潜在空间中，在该空间中可以通过共享的概念激活来分析词元之间的关系。具体而言，我们将概念重叠定义为活跃SAE潜在特征之间的一致程度，并利用它对共享语义内容的词元进行分组，然后在每个组内通过剪枝等方式执行词元削减（摘要在此处截断）。

    arXiv:2607.04593v2 Announce Type: replace-cross  Abstract: Vision-Language Models (VLMs) have demonstrated impressive capabilities across different tasks, but their computational cost is dominated by the large number of visual tokens fed to the language model. Existing token reduction methods rely on attention-based scores or pairwise similarity, without an explicit semantic representation of each token. We introduce TORINO (TOken Reduction via Interpretable coNcept Overlap), a plug-and-play framework for adaptive visual token reduction in VLMs that requires no fine-tuning of the underlying model. TORINO leverages Sparse Autoencoders (SAEs) to project visual tokens into an interpretable latent space where token relationships can be analyzed through shared concept activations. Specifically, we define concept overlap as the degree of agreement between active SAE latents and use it to group tokens that share semantic content. Reduction within each group is then performed by either pruning
    
[^261]: APeB：大语言模型智能体个性化能力基准测试

    APeB: Benchmarking Personalization Ability of Large Language Model Agents

    [https://arxiv.org/abs/2607.03162](https://arxiv.org/abs/2607.03162)

    该论文提出个性化产品搜索任务和APeB基准，用于评估大语言模型智能体在原始欠规范查询与丰富交互历史下的个性化能力，发现现有模型在早期模糊查询的意图与偏好发现上表现不佳，其主要原因是对历史信息利用不足。

    

    基于大语言模型（LLM）的智能体在用户发出原始的、欠规范的查询时难以实现个性化。在这种场景下，智能体必须推断潜在意图、从嘈杂的交互历史中提取用户偏好，并在相互竞争的候选项中进行选择。现有基准很少测试这种能力，因为它们通常依赖于用户精炼后的查询或简化的历史记录。我们提出了个性化产品搜索（PPS）这一任务，作为在原始查询和多样化历史条件下测试智能体个性化能力的试验平台。我们从用户行为日志中构建了智能体个性化基准（APeB），将欠规范的意图与丰富的交互历史以及用户浏览过的候选商品相配对。通过多步骤智能体工作流评估最先进的大语言模型，我们发现模型能够很好地处理显式查询，但在需要意图发现和偏好挖掘的早期查询上表现不佳。基于评分标准的分析将这一差距主要归因于对历史记录的低效利用。一个简单的基于历史信息的查询精炼流水线……（摘要在此处被截断）

    arXiv:2607.03162v3 Announce Type: replace  Abstract: LLM-powered agents struggle with personalization when users issue raw, underspecified queries. In this setting, agents must infer latent intent, extract preferences from noisy interaction histories, and select among competing alternatives. Existing benchmarks rarely test this capability, as they often rely on user-refined queries or simplified histories. We introduce personalized product search (PPS), a testbed for agentic personalization under raw queries and diverse histories. We construct Agent Personalized Benchmark (APeB) from action logs, pairing underspecified intents with rich histories and user-viewed candidate items. Evaluating state-of-the-art LLMs with multi-step agent workflows, we find that models handle explicit queries well but struggle with early-stage queries requiring intent and preference discovery. Rubric analysis attributes this gap mainly to ineffective history use. A simple history-aware query-refinement pipel
    
[^262]: PruneGround：面向3D视觉定位的即插即用空间剪枝框架

    PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

    [https://arxiv.org/abs/2606.31148](https://arxiv.org/abs/2606.31148)

    提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。

    

    3D视觉定位（3DVG）旨在根据自然语言描述在3D场景中定位目标物体。现有方法通常对整个场景进行推理，导致预测结果模糊且计算成本高昂，尤其是在杂乱的环境中。我们观察到，许多指代表达依赖于局部空间上下文，并且往往对应于受限的空间区域而非整个场景。受这一洞察启发，我们提出了PruneGround，一个高效的即插即用3DVG框架，它建立在三个关键组件之上。首先，我们引入了语言引导空间剪枝（LGSP），利用冻结的视觉语言模型（VLM）识别与语言相关的区域，从而在更窄的搜索空间中减少空间计算和定位候选。其次，我们提出了多视角条件化描述重构（MCDR），将复杂表达分解为简化的目标中心描述……（摘要内容不完整）

    arXiv:2606.31148v2 Announce Type: replace-cross  Abstract: 3D Visual Grounding (3DVG) aims to localize target objects in 3D scenes given natural language descriptions. Existing approaches typically perform reasoning over the entire scene, leading to ambiguous predictions and high computational cost, especially in cluttered environments. We observe that many referential expressions rely on local spatial context and often correspond to restricted spatial regions rather than the full scene. Motivated by this insight, we propose PruneGround, an effective plug-and-play framework for 3DVG built upon three key components. First, we introduce Language-Guided Spatial Pruning (LGSP), which leverages a frozen Vision Language Model (VLM) to identify language-relevant regions, thereby reducing spatial computation and grounding candidates in the narrower search space. Second, we propose MultiView-Conditioned Description Reformulation (MCDR), which decomposes complex expressions into simplified targe
    
[^263]: 超越编译：评估忠实的自然语言到Lean语句形式化

    Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization

    [https://arxiv.org/abs/2606.31002](https://arxiv.org/abs/2606.31002)

    该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。

    

    Lean能够验证生成的声明类型正确，但无法验证它表达了用户真正想要的语句。我们针对没有规范Lean目标的自动形式化研究两个问题：LLM评判者能否为人类语义审查提供可用的代理，以及编译在不同系统之间会在多大程度上高估忠实度。我们的评估标准将Lean编译与GPT-5.2和Gemini-2.5-Pro之间的严格语义共识相结合。在一个独立审计的随机样本上，该标准与人类多数意见在89.7%的案例中保持一致（Wilson 95%置信区间：82.1–94.3%）。在400个研究生水平语句上评估的八个系统中，每个系统都存在非零的编译-忠实度差距，其观测幅度介于3.0到29.0个百分点之间。完整的GPT-5.2工具增强代理表现出最大的差距，编译率达89.5%，而满足语义标准的仅为60.5%。人工审查、一个独立的第三族评判者以及BEq形式化交叉……

    arXiv:2606.31002v2 Announce Type: replace  Abstract: Lean verifies that a generated declaration is well typed, but not that it expresses the statement a user intended. We study two questions for autoformalization without canonical Lean targets: whether LLM judges can provide a usable proxy for human semantic review, and how much compilation overstates faithfulness across systems. Our criterion combines Lean compilation with strict semantic consensus between GPT-5.2 and Gemini-2.5-Pro. On an independently audited random sample, it agrees with human majority on 89.7\% of cases (Wilson 95\% CI: 82.1--94.3\%). Across eight systems evaluated on 400 graduate-level statements, every system has a nonzero compile--faithfulness gap, whose observed magnitude ranges from 3.0 to 29.0 percentage points. The full GPT-5.2 tool-augmented agent shows the largest gap, compiling 89.5\% while satisfying the semantic criterion on 60.5\%. Human review, an independent third-family judge, and a BEq formal cros
    
[^264]: 面向长时程智能体强化学习的群组-图策略优化

    Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning

    [https://arxiv.org/abs/2606.22995](https://arxiv.org/abs/2606.22995)

    提出群组-图策略优化（G2PO），一种针对多轮智能体任务的新型分组强化学习算法，通过显式利用状态转换固有的图结构，解决了长时程智能体强化学习中奖励稀疏延迟、高方差价值估计以及短视局部化信用分配的关键瓶颈。

    

    基于分组的强化学习已在智能体场景中显著增强了大语言模型（LLMs）的能力。为了实现更细粒度的策略更新，近期的智能体强化学习框架已从轨迹级训练转向步骤级训练。然而，长时程智能体强化学习面临着严重的奖励稀疏与延迟问题，因为反馈往往要推迟数十个交互步骤之后才会出现。尽管现有的步骤级框架细化了训练粒度，但其信用分配仍然是粗粒度的，并且依然将智能体的探索视为孤立的线性轨迹。这种过于简化的视角忽略了状态转换中固有的图结构，导致了高方差的状态价值估计以及短视的、局部化的信用分配。为了克服这些关键瓶颈，我们提出了群组-图策略优化（G2PO），这是一种专为多轮智能体任务量身定制的新型分组强化学习算法。G2PO 显式地将……（摘要在此处截断）

    arXiv:2606.22995v2 Announce Type: replace-cross  Abstract: Group-based Reinforcement Learning (RL) has significantly enhanced Large Language Models (LLMs) in agentic scenarios. To achieve finer-grained policy updates, recent agentic RL frameworks have shifted from trajectory-level to step-level training. However, long-horizon agentic RL suffers from severe reward sparsity and delay, as feedback is often deferred for dozens of interaction steps. While existing step-level frameworks refine training granularity, their credit assignment remains coarse-grained and still treats agent exploration as isolated, linear trajectories. This oversimplified perspective ignores the inherent graph structure of state transitions, leading to high-variance state-value estimation and myopic, localized credit assignment. To overcome these critical bottlenecks, we propose Group-Graph Policy Optimization (G2PO), a novel group-based RL algorithm tailored for multi-turn agentic tasks. G2PO explicitly transforms
    
[^265]: 超越分词：面向时间序列问答的直接时间步嵌入与对比对齐方法

    Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering

    [https://arxiv.org/abs/2606.18986](https://arxiv.org/abs/2606.18986)

    该论文提出CADE框架，通过直接时间步嵌入与对比语义对齐，克服大语言模型处理时间序列时的分词瓶颈，并解决了基于固定窗口的补丁编码所导致的粒度锁定和跨数据集迁移困难问题。

    

    大语言模型（LLM）的最新进展催生了时间序列问答（TSQA），该任务将时间序列分析表述为自然语言问答问题。然而，直接将原始数值序列输入大语言模型会遭遇分词瓶颈：字节对编码（BPE）将连续数值切分为不稳定的词元，其嵌入缺乏有意义的度量结构，导致幅值、尺度和趋势信息的丢失。先前的方法采用基于补丁的编码器将序列切分为固定窗口，从而锁定单一粒度，破坏了序列模式并隐藏了精确的时间步信息，且这种独立模块难以在不同长度或采样率的数据集之间迁移。为应对这一挑战，我们提出了CADE（对比对齐与直接嵌入），一种面向TSQA的新型框架，其建立在两个关键组件之上：直接时间步嵌入和语义对齐。

    arXiv:2606.18986v2 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have given rise to time-series question answering (TSQA), which formulates time-series analysis as natural-language question answering. However, directly feeding raw numerical series into LLMs suffers from a tokenization bottleneck: Byte Pair Encoding fragments continuous values into unstable tokens whose embeddings lack meaningful metric structure, resulting in the loss of magnitude, scale, and trend information. Prior methods use patch-based encoders that split the series into fixed windows, locking in one granularity that breaks patterns and hides exact timesteps, through a separate module that rarely transfers across datasets with different lengths or sampling rates. To address this challenge, we propose CADE (Contrastive Alignment with Direct Embedding), a novel framework for TSQA built upon two key components: direct timestep embedding and semantic alignment. The proposed fr
    
[^266]: AgentRivet：一种从期刊出版物中自动生成Rivet例程的系统

    AgentRivet: an automated system for producing Rivet routines from journal publications

    [https://arxiv.org/abs/2606.13535](https://arxiv.org/abs/2606.13535)

    本文提出AgentRivet，一个基于大语言模型的多步骤自动化工作流，能够从已发表的粒子物理论文中提取分析信息并自动编写缺失的Rivet例程，通过内置的代码和物理审查实现自主质量控制，解决了目前仅39%测量拥有可用Rivet例程的覆盖率不足问题。

    

    粒子物理对撞机实验提供Rivet例程，作为模型无关测量的分析保存策略的一部分。Rivet是一个C++工具包，允许将新的理论模型与测量结果进行比较，从而有助于蒙特卡罗事件产生器的开发与调校，以及超越标准模型物理的搜索。然而，已知分析覆盖率并不完整，仅有39%的测量拥有有记录且公开可用的Rivet例程。在本文中，我们设计并实现了一个基于大语言模型的自动化工作流，旨在提供缺失的例程。这个多步骤工作流被称为AgentRivet，它从已发表的论文中提取物理分析信息并编写缺失的Rivet例程，其中间环节包含代码审查和物理审查，作为自主质量控制的一部分。我们报告了使用商用大语言模型所获得的结果。

    arXiv:2606.13535v4 Announce Type: replace-cross  Abstract: Particle physics collider experiments provide Rivet routines as part of the analysis preservation strategy for model-independent measurements. Rivet is a C++ toolkit that allow new theoretical models to be compared to the measurements, thus aiding the development and tuning of Monte Carlo event generators as well as searches for physics beyond the Standard Model. However, analysis coverage is known to be incomplete, with only 39% of measurements having documented and publicly available Rivet routines. In this article, we design and implement an automated workflow based on Large Language Models with the goal of providing the missing routines. This multi-step workflow, referred to as AgentRivet, extracts the physics analysis information from published papers and writes the missing Rivet routines, with intermediate code- and physics- reviews as part of an autonomous quality control. We report the results obtained using commercial 
    
[^267]: 企业温室气体清单中购买式人工智能推理的核算：一种基于Token（词元）的方法

    Accounting for Purchased Artificial Intelligence Inference in Corporate Greenhouse Gas Inventories: A Token-Based Method

    [https://arxiv.org/abs/2606.10660](https://arxiv.org/abs/2606.10660)

    本文提出一种基于Token的核算方法和开放数据集，用于将购买式AI推理纳入企业温室气体清单的范围三第一类，克服了现有基于支出的方法偏差高达12倍且无法区分模型与地区的缺陷。

    

    购买的人工智能（AI）推理服务应计入企业温室气体清单的范围三第一类，但目前的实践中要么将其遗漏，要么采用基于支出的排放因子，该方法所得结果是物理估算值的1.7至12.3倍，且无法区分不同模型或地区。本文提出了一种基于Token的方法和开放数据集，包括：摇篮到大门的核算边界、四个层级的活动数据、有效Token计量基础、由四项构成的排放因子（推理服务用电、硬件隐含碳排放、数据中心建设、已披露的最终训练运行），以及一条可将任意模型标识符和地区解析为对应排放因子的规则。推理碳足迹在不同模型类别间相差十二倍，在不同地区间相差九倍。中档模型类别的因子——按全球平均电网计算为每百万计费Token排放0.039千克二氧化碳当量（CO2e）——是所声明结构区间的低端：较低效的推理服务配置会使其提高2.8倍，而替换训练部分……（摘要原文在此处被截断）

    arXiv:2606.10660v2 Announce Type: replace-cross  Abstract: Purchased artificial intelligence (AI) inference belongs in Scope 3 Category 1 of a corporate greenhouse gas inventory, yet practice omits it or applies a spend-based factor that gives 1.7 to 12.3 times the physical estimate and cannot tell one model or region from another. This paper sets out a token-based method and open dataset: a cradle-to-gate boundary, four tiers of activity data, an effective-token basis, a four-term factor (serving electricity, embodied hardware, data-centre construction, disclosed final training runs) and a rule resolving any model identifier and region to a factor. Inference footprints span twelve times between model classes and nine between regions. The mid-class factor, 0.039 kg of carbon dioxide equivalent (CO2e) per million billed tokens at the world-average grid, is the low end of a stated structural range: a less efficient serving configuration raises it 2.8 times, and replacing the training ter
    
[^268]: TimpaTeks：基于扩散语言模型引导的自动原地文本序列修改

    TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering

    [https://arxiv.org/abs/2606.08408](https://arxiv.org/abs/2606.08408)

    TimpaTeks将激活引导扩展到扩散语言模型，实现了一种无需指令微调模型的自动原地文本修改机制，在保留原句结构、降低困惑度的同时，以更低的计算成本改变文本所表达的概念。

    

    我们将激活引导技术扩展到扩散语言模型，并研究了由扩散语言模型推理机制所引发的一个新问题：对文本进行原地修改以呈现不同的概念。我们提出了TimpaTeks，一种使用扩散语言模型的自动原地文本修改机制。在IMDB电影评论（情感概念）和一个合成的猫狗数据集（任意的、更非传统的概念引导）上的实验表明，TimpaTeks提供了一种可行的新型机制来原地引导扩散语言模型的输出。TimpaTeks无需依赖指令微调模型即可实现原地修改，同时能够降低句子困惑度并保留原始句子结构。此外，由于TimpaTeks执行的是原地去噪而非构建额外的提示条件输出序列，其计算成本也低于基于提示的扩散语言模型引导方法。

    arXiv:2606.08408v2 Announce Type: replace-cross  Abstract: We extend activation steering to diffusion language models (DLMs) and study a novel problem that arose due to the inference mechanism of DLMs: Modifying a text in-place to manifest a different concept. We propose TimpaTeks, an automatic in-place text modification mechanism using DLMs. Experiments on IMDB movie reviews (sentiment) and a synthetic Cats and Dogs Dataset (arbitrary, more unconventional concept steering) show that TimpaTeks provides a feasible novel mechanism to steer diffusion language model outputs in-place. TimpaTeks enables in-place modification while simultaneously lowers sentence perplexity and retaining the original sentence structre without the need of instruction tuned models. TimpaTeks is also computationally cheaper than prompt-based DLM steering, as it performs denoising in-place rather than constructing an additional prompt-conditioned output sequence.
    
[^269]: 超越语义组织：将记忆作为长时程智能体的执行状态管理

    Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents

    [https://arxiv.org/abs/2606.06090](https://arxiv.org/abs/2606.06090)

    本文提出MAGE系统，将智能体记忆从被动的语义相似度检索重新构想为主动的执行状态管理，通过层次化状态树和四种耦合操作（生长、压缩、维护、修订），帮助长时程智能体实现连贯的状态重建与错误隔离。

    

    arXiv:2606.06090v2 公告类型：替换 摘要：基于大语言模型（LLM）的智能体越来越多地处理具有相互依赖决策的长时程任务，其中每个动作都会重塑未来的约束条件，中间产生的错误可能级联扩散。现有的RAG和智能体记忆系统通过语义相似度来组织历史记录，在决策时检索与内容相关的条目。我们认为这种设计与执行状态的依赖关系不匹配：它割裂了决策轨迹，并混合了有效与错误的记录，从而阻碍了连贯的状态重建和错误隔离。我们提出了MAGE（Memory as Agent-Guided Exploration，记忆即智能体引导探索），这是一种主动的执行状态管理器，它将交互存储在层次化的状态树中。智能体从活跃的“根节点到当前节点”路径中推导自身状态，结合子目标摘要、近期轨迹以及来自先前分支的提示。四种相互耦合的操作来维护该状态树：Grow（生长）记录新轨迹，Compress（压缩）总结已完成的子目标，Maintain（维护）验证摘要的有效性，Revise（修订）恢复……（原文摘要在此处截断）

    arXiv:2606.06090v2 Announce Type: replace  Abstract: LLM-based agents increasingly tackle long-horizon tasks with interdependent decisions, where each action reshapes future constraints and intermediate errors can cascade. Existing RAG and agent memory systems organize histories by semantic similarity, retrieving content-relevant entries at decision time. We argue that this design mismatches execution-state dependencies: it fragments decision trajectories and mixes valid and erroneous traces, hindering coherent state reconstruction and error isolation. We propose MAGE (Memory as Agent-Guided Exploration), an active execution-state manager that stores interactions in a hierarchical state tree. The agent derives its state from the active root-to-current path, combining subgoal summaries, recent traces, and hints from prior branches. Four coupled operations maintain the tree: Grow records new traces, Compress summarizes completed subgoals, Maintain validates summaries, and Revise restores
    
[^270]: CauTion：知道何时信任大语言模型进行集成因果发现

    CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery

    [https://arxiv.org/abs/2606.03602](https://arxiv.org/abs/2606.03602)

    提出CauTion框架，通过共识过滤和LLM可靠性估计，将大语言模型的领域知识可靠地融合进统计因果发现算法集成中，从而同时克服纯统计方法、LLM错误风险和单一算法偏差这三方面的局限。

    

    从观测数据中进行因果发现仍然具有挑战性，原因在于纯统计方法存在根本性局限，例如等价类内部的统计不可区分性以及对有限样本量的敏感性。虽然大语言模型（LLM）为补充统计推断提供了有前景的领域知识来源，但现有的LLM增强方法容易受到LLM错误的影响，且产生高昂的token成本。此外，依赖单一以数据为中心的算法会使结果对算法特有的偏差敏感。为了解决这些局限性，我们提出了CauTion，一个通过共识过滤和LLM可靠性估计，将LLM领域知识可靠地集成到统计因果发现算法集成中的框架。CauTion分三个阶段进行。首先，算法集成利用共识投票解决多达96%的算法间达成一致的边，实现接近完美的（摘要在此处被截断）

    arXiv:2606.03602v2 Announce Type: replace-cross  Abstract: Causal discovery from observational data remains challenging due to the fundamental limitations of purely statistical methods, such as statistical distinguishability within equivalence classes and sensitivity to finite sample sizes. While large language models (LLMs) offer a promising source of domain knowledge to complement statistical inference, existing LLM-augmented methods are vulnerable to LLM errors and incur high token costs. Moreover, reliance on a single data-centric algorithm can make results sensitive to algorithm-specific biases. To address these limitations, we propose CauTion, a framework that reliably integrates LLM domain knowledge into an ensemble of statistical causal discovery algorithms through consensus filtering and LLM reliability estimation. CauTion proceeds in three stages. First, an algorithm ensemble utilizes a consensus voting to resolve up to 96% of edges on which algorithms agree, achieving near-p
    
[^271]: 大语言模型在交通系统管理与运营中的应用：从文本推理到多模态决策支持

    Large Language Models in Transportation Systems Management and Operations: From Text Reasoning to Multi-modal Decision Support

    [https://arxiv.org/abs/2606.00991](https://arxiv.org/abs/2606.00991)

    本综述系统梳理了大语言模型及多模态大语言模型在交通系统管理与运营三大领域（运营与服务、出行与车队服务、数据建模与决策支持）中的应用现状，并指出了数据异构性、实时推理、可解释性与多模态融合等关键挑战。

    

    交通系统管理与运营（TSMO）日益依赖于对异构数据的及时解读，这些数据来自各种传感器流、事件报告、旅客反馈和视觉观察。大语言模型（LLM），包括新兴的多模态大语言模型（MM-LLM），为将这些结构化和非结构化输入整合到面向运营人员的决策支持中提供了一种新的机制。本综述论文回顾了基于LLM和MM-LLM在TSMO中的相关应用，涵盖三个领域：交通运营与服务（供给）、出行与车队服务（需求），以及数据、建模与决策支持。通过采用PRISMA指导的筛选流程，我们综合梳理了当前的研究，同时区分了面向实际运营的应用与原型及新兴概念。我们进一步识别了数据异构性、实时推理、可解释性、多模态融合和治理等方面反复出现的挑战。

    arXiv:2606.00991v2 Announce Type: replace  Abstract: Transportation systems management and operations (TSMO) increasingly depends on timely interpretation of heterogeneous data, from various sensor streams, incident reports, traveler feedback, and visual observations. Large language models (LLMs), including emerging multi-modal large language models (MM-LLMs), provide a new mechanism for integrating these structured and unstructured inputs into operator-facing decision support. This survey paper reviews LLM- and MM-LLM-based applications in TSMO across three domains: transportation operations & services (supply), mobility & fleet services (demand), and data, modeling & decision support. Using a PRISMA-guided screening process, we synthesize current studies while distinguishing operationally oriented applications from prototype and emerging concepts. We further identify recurring challenges in data heterogeneity, real-time inference, explainability, multi-modal fusion, and governance. F
    
[^272]: 谁能改变航向？AI主权与集体学习能力的整合性综述

    Who Can Change Course? An Integrative Review of AI Sovereignty and Collective Learning Capacity

    [https://arxiv.org/abs/2606.00729](https://arxiv.org/abs/2606.00729)

    本文提出将AI主权理解为一种国家学习能力，以人文中心学习机制（HCLM）为概念框架，结合法国案例，分析国家如何将算力、数据、人才等分散输入转化为协调且经负责任评估的能力，同时避免协调失败、行政重复等制度性损失。

    

    本文将AI主权作为国家学习能力进行考察，并以法国作为应用案例。为熵调控表示学习而提出的人文中心学习机制（HCLM）（Tran, 2026）提供了一个概念视角，用以关联信息输入、吸收能力与可避免的制度性损失。输入要素包括算力、数据、人才、研究、资本、部署和制度实验。潜在损失包括协调失败、行政重复、能源约束、监管不确定性和有限的组织采纳。监管还通过评估、可信数据访问、互操作性和安全标准来支持学习。本文的核心论点是：主权取决于将分散的资源和知识转化为可用、协调且经过负责任评估的能力。该分析将这一论述与神经缩放定律、内生增长和创造性破坏……（原文摘要不完整）

    arXiv:2606.00729v3 Announce Type: replace  Abstract: This paper examines AI sovereignty as national learning capacity, using France as an application. Human-Centered Learning Mechanics (HCLM), proposed for entropy-regulated representation learning (Tran, 2026), provides a conceptual lens for relating information inputs, absorptive capacity and avoidable institutional losses. Inputs include compute, data, talent, research, capital, deployment and institutional experimentation. Potential losses include coordination failures, administrative duplication, energy constraints, regulatory uncertainty and limited organizational uptake. Regulation also supports learning through evaluation, trusted data access, interoperability and safety standards. The argument is that sovereignty depends on converting distributed resources and knowledge into usable, coordinated and responsibly evaluated capability. The analysis connects this account with neural scaling laws, endogenous growth, creative destruct
    
[^273]: 学习对隐式多模态危害进行推理：从诊断到泛化

    Learning to Reason About Implicit Multimodal Harm: From Diagnosis to Generalization

    [https://arxiv.org/abs/2605.29951](https://arxiv.org/abs/2605.29951)

    提出MuPHI诊断基准，揭示视觉-语言模型虽能理解单一模态信息，却无法将图像-文本组合推理为危害机制，为隐式多模态危害检测从诊断走向泛化提供了方向。

    

    一张无害的图片配上无害的文字说明，只有在组合在一起时才可能变得有害，这使得此类检测在本质上是一个跨模态推理问题。现有的多模态危害基准仅评估分类准确率，尚不清楚模型是否真正识别了产生危害的跨模态机制。我们提出了MuPHI（多模态语用危害解释），这是一个受控诊断基准，其中危害纯粹源于图像与文本的组合，并配有良性反事实样本和经过标注的危害成因解释。借助MuPHI，我们系统地评估了视觉-语言模型（VLMs）中组合式危害推理的失效环节。MuPHI揭示：VLMs能够可靠地理解单一模态的信息，却无法将证据关联到危害机制之中，且其预测对决定标签的反事实干预响应微弱。此外，现有基于标签训练的危害检测方法虽能显著提升域内（ID）检测性能，但……

    arXiv:2605.29951v2 Announce Type: replace  Abstract: A benign image paired with a benign caption can become harmful only in combination, making detection fundamentally a cross-modal reasoning problem. Existing multimodal harm benchmarks evaluate only classification accuracy, leaving unclear whether models identify the cross-modal mechanism that produces harm. We introduce MuPHI (Multimodal Pragmatic Harm Interpretation), a controlled diagnostic benchmark in which harm arises purely from image-text composition, with benign counterfactuals and annotated harm rationales. Using MuPHI, we systematically evaluate where compositional harm reasoning breaks in vision-language models (VLMs). MuPHI reveals that VLMs reliably ground each modality yet fail to bind the evidence into the harm mechanism and their predictions respond weakly to label-determining counterfactual interventions. Moreover, existing harm detection approaches trained on labels substantially improves in-domain (ID) detection bu
    
[^274]: SciIntBench：衡量大语言模型在对抗性框架下对科研诚信规范的遵循程度

    SciIntBench: Measuring LLM Compliance with Research Integrity Norms Under Adversarial Framing

    [https://arxiv.org/abs/2605.29468](https://arxiv.org/abs/2605.29468)

    该论文提出了对抗性基准SciIntBench，通过1350个跨五个对抗等级的提示评估16个大语言模型对科研诚信规范的遵循情况，发现模型在最隐蔽的对抗框架下平均71.7%的情况未能拒绝科研不当行为请求。

    

    大语言模型（LLM）越来越多地被用于支持科学研究工作，但它们究竟是恪守负责任研究行为（RCR）规范，还是在助长对这些规范的破坏，目前尚不清楚。我们提出了SciIntBench，这是一个对抗性基准，包含1350个提示，涵盖270个场景、十类RCR类别和三个科学领域。每个场景以五个等级呈现，从最显性到最隐蔽的对抗性提示，外加一个良性版本，使我们能够同时衡量模型对不当行为请求的框架敏感性拒绝能力以及对合法请求的有用性。我们评估了来自六家提供商（2024–2026年）的16个商业及开放权重LLM，共生成21,600条回复。我们采用经人工评估验证的LLM判断对回复进行评估，达到了高度一致性（κ = 0.857）。我们发现，在最隐蔽的框架条件下，模型平均在71.7%的情况下未能拒绝涉及科研不当行为的请求。在拒绝……

    arXiv:2605.29468v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used to support scientific work, but it is unclear whether they uphold responsible conduct of research (RCR) norms or help undermine them. We introduce SciIntBench, an adversarial benchmark of 1350 prompts across 270 scenarios, ten RCR categories, and three scientific domains. Each scenario appears in five gradations, from Most Overt to Most Covert adversarial prompts, and an additional Benign version, allowing us to jointly measure framing-sensitive refusal of misconduct and helpfulness on legitimate requests. We evaluate 16 commercial and open-weight LLMs from six providers (2024--2026), producing 21,600 responses. We assess the responses using LLM judgments validated by human evaluation, achieving high agreement ($\kappa$ = 0.857). We find that, under the most covert framing condition, models fail to refuse requests for research misconduct in 71.7\% of cases, on average. Refusals
    
[^275]: FormInv：数学推理基准中语义不变性的测量协议

    FormInv: A Measurement Protocol for Semantic Invariance in Mathematical Reasoning Benchmarks

    [https://arxiv.org/abs/2605.29001](https://arxiv.org/abs/2605.29001)

    该论文提出FormInv测量协议，用于审计数学推理基准中自动生成改写题目的语义等价性，并引入条件不一致率（CIR）指标，揭示即使在所有模型都能正确回答的规范形式定理上，语义等价的改写仍会导致5-42%的答案翻转。

    

    对MathCheck（ICLR 2025）改写质量的审计在129个组中发现了4个语义错误的改写（3.1%）；移除这些错误后，GPT-4o的排名从第2位降至第4位，而Claude Haiku和DeepSeek V3则升至其上。利用跨模型一致性方法，这些错误在不到10美元的成本下被自动发现；在我们自己的数据集中，同一协议发现47%自动生成的连接词变化改写在语义上是错误的。这一缺陷还加剧了一个更深层的测量差距：在所有模型都能以规范形式正确回答的定理上，语义等价的改写仍会使其中5-42%的答案发生翻转；在所有模型共享的43个定理上，Claude Haiku 4.5翻转已知正确答案的频率约为DeepSeek V3的9倍（41.9%对4.7%），我们将这一难度受控的差距称为条件不一致率（CIR）。CIR排名在一个独立的103个定理集合上得到了复现。原始语义一致性率（SCR）在各模型之间的范围为50-82%（摘要在此处截断）。

    arXiv:2605.29001v2 Announce Type: replace-cross  Abstract: A paraphrase-quality audit of MathCheck (ICLR 2025) detected 4 semantically incorrect paraphrases in 129 groups (3.1%); removing them drops GPT-4o from rank 2 to rank 4 and elevates Claude Haiku and DeepSeek V3 above it. Cross-model unanimity found these errors automatically for under $10; in our own dataset the same protocol found that 47% of auto-generated connective-variation paraphrases were semantically incorrect. That flaw compounds a deeper measurement gap. On the theorems every model answers correctly in canonical form, a semantically equivalent rewording still flips the answer for 5-42% of them: Claude Haiku 4.5 flips a known-correct answer about 9x as often as DeepSeek V3 (41.9% vs. 4.7% on the 43 theorems all models share), a difficulty-controlled gap we call the Conditional Inconsistency Rate (CIR). The CIR ranking replicates on an independent 103-theorem set. Raw Semantic Consistency Rates (SCR) span 50-82% across 
    
[^276]: 提示词码本：面向语言模型指令精炼的离散组合优化

    Prompt Codebooks: Discrete Compositional Optimization for Language Model Instruction Refinement

    [https://arxiv.org/abs/2605.28360](https://arxiv.org/abs/2605.28360)

    该论文提出提示词码本优化（PCO）框架，将自动提示词优化重新建模为对可复用自然语言指令单元码本的离散组合学习，通过编码器—生成器—评论家架构和文本梯度信号实现提示词的模块化构建与各组件的联合优化。

    

    自动提示词优化（APO）已在基于大语言模型（LLM）的智能体工作流中带来了显著提升。然而，大多数现有方法将每个任务的提示词视为一个单一的、与具体实例无关的字符串，并通过全局编辑进行优化，这导致更新脆弱，且无法复用已学习的子行为。我们提出提示词码本优化（PCO），这是一种新颖的组合式提示词优化框架，它将APO重新表述为对有限的自然语言“直觉”词表——即原子化、可复用的指令单元——进行的离散学习。PCO将提示词构建知识组织在一个离散码本中，并通过基于LLM的编码器将每个输入路由到少量码本条目；生成器将这些条目组合成供执行器使用的提示词；评论家给出结构化判定，并通过归因分解为每个变量的文本梯度，从而在语言值的极小极大目标下联合训练编码器、生成器、评论家和码本。

    arXiv:2605.28360v3 Announce Type: replace  Abstract: Automatic prompt optimization (APO) has driven significant gains in LLM-based agentic workflows. However, most existing methods treat each task's prompt as a monolithic, instance-blind string optimized through global edits, producing brittle updates and preventing the reuse of learned sub-behaviors. We propose Prompt Codebook Optimization (PCO), a novel compositional prompt optimization framework that recasts APO as discrete learning over a finite vocabulary of natural-language instincts--atomic, reusable instruction units. PCO organizes prompt-construction knowledge in a discrete codebook and routes each input to a small subset of entries via an LLM-based encoder; a generator composes them into a prompt for the executor; a critic emits a structured verdict that decomposes by attribution into per-variable textual gradients, jointly training the encoder, generator, critic, and codebook under a language-valued min-max objective. The re
    
[^277]: ATWL：一种用于表示、比较和重用可视化分析工作流的形式化语言

    ATWL: A Formal Language for Representing, Comparing, and Reusing Visual Analytics Workflows

    [https://arxiv.org/abs/2605.25489](https://arxiv.org/abs/2605.25489)

    该论文提出了一种声明式语言ATWL来表示可视化分析工作流的结构与意图，并构建了包含54个工作流的库，揭示了人机分工与迭代求精的规律，同时提升了LLM设计新工作流的效果。

    

    可视化分析（VA）工作流结合了数据变换、特征计算、视觉表示和人类解释，但它们通常以文字形式描述。这阻碍了工作流之间的比较、已验证策略的重用以及对新手的培训。我们提出了工件-变换工作流语言（ATWL），这是一种用于表示可视化分析工作流结构与意图的声明式语言。一个工作流由类型化工件和变换组成，每个变换都具有分析意图以及人类、机器或混合的执行者。我们在人工监督下借助大语言模型（LLM）从已发表的可视化分析论文中提取了54个工作流，构建了一个工作流库。对该库工作流的探索性计算比较显示出一些规律性，例如人与机器之间规律的分工模式以及迭代求精的循环。我们进行了让LLM为新任务设计工作流的实验，这些实验展示了使用该工作流库以及ATWL表示所带来的效果。（原文摘要在此处截断）

    arXiv:2605.25489v2 Announce Type: replace  Abstract: Visual analytics (VA) workflows combine data transformation, feature computation, visual representation, and human interpretation, but they are usually described in prose. This hinders comparison of workflows, reuse of proven strategies, and training of novices. We present Artifact--Transform Workflow Language (ATWL), a declarative language for representing the structure and intent of VA workflows. A workflow consists of typed artifacts and of transforms, each with an analytical intent and a human, machine, or hybrid actor. We built a library of 54 workflows extracted from published VA papers with the help of LLMs under human supervision. An exploratory computational comparison of the library workflows shows regularities, such as a regular division of labour between humans and machines and cycles of refinement. We ran experiments in which LLMs designed workflows for new tasks. They show the effects of using the library, the ATWL repr
    
[^278]: LLM工具注册表中的面向智能体的信息设计：对修辞、位置与结构的预注册检验

    Agent-Facing Information Design in LLM Tool Registries: A Preregistered Test of Rhetoric, Position and Structure

    [https://arxiv.org/abs/2605.23916](https://arxiv.org/abs/2605.23916)

    研究通过预注册实验发现，LLM智能体的工具选择会被工具描述中的推销性赞美语言显著影响（可提升约43个百分点选中率，甚至选到无法胜任任务的工具），列表排序和结构化字段也会强烈左右选择，因此建议注册表以结构化字段而非营销文本来呈现工具限制。

    

    AI智能体通常从工具注册表中挑选工具，而注册表中每个工具的描述由其提供者撰写。我们探究这些描述中的推销语言是否会改变智能体所选择的工具。我们构建了仅在某一受控因素上存在差异的工具列表对（添加赞美之词、可验证的规范说明、或列表顺序），并让两个OpenAI模型调用其中一个工具。在一项预注册研究中，堆叠使用赞美之词（四种类型组合）使工具的选中率提高了约43个百分点，其效果与可验证规范相当甚至更强。赞美之词还会使部分选择偏向无法完成任务的工具，但很少使选择偏向索要不必要数据访问权限的工具。当列表内容完全相同时，排在首位的工具被选中的概率高出约72个百分点。在包含数值限制的任务中，结构化字段有助于智能体选择具备所需能力的工具；但在字段旁附上提供者的推销文字会削弱甚至消除这一收益。注册表可以将工具限制以字段形式列出，并隐藏推销性文字……（原文截断）

    arXiv:2605.23916v2 Announce Type: replace-cross  Abstract: AI agents often pick tools from registries, where each tool's provider writes its description. We ask whether sales language in those descriptions changes which tool an agent picks. We built pairs of listings differing in one controlled way (added praise, a verifiable specification, or list order) and asked two OpenAI models to call one tool. In a preregistered study, stacked praise (four kinds combined) raised a tool's pick rate by about 43 percentage points, matching or beating a verifiable specification. Praise also pulled some picks toward tools that could not do the task, but rarely toward tools asking for unneeded data access. With identical listings, the first-listed tool was picked about 72 points more often. On tasks with numeric limits, structured fields helped agents pick the capable tool; adding the provider's sales text beside the fields reduced or erased that gain. Registries could list limits as fields, hide sale
    
[^279]: 以人为中心的学习力学：受保护风险约束下的可观测熵控制

    Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints

    [https://arxiv.org/abs/2605.22940](https://arxiv.org/abs/2605.22940)

    本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。

    

    熵降低并非一个安全准则：它可能使预测保持不变，或伴随着受保护群体风险的上升。我们研究在分别指定的任务、风险与干预约束下的熵调控学习。一个经噪声校准的对数行列式泛函给出了精确的谱力公式，以及将速度度量分解为可观测分量与预测零空间分量的方法。对于标量反馈，我们刻画了在保持任务下降比例并满足熵速率要求时系数的完整区间；而将干预限制在可观测方向上可能使该区间为空。对于多个受保护风险，我们在指定子空间与范数预算内推导出精确的极小极大可行性证书、针对不确定约束行的精确鲁棒扩展，以及保护后剩余熵削减的锥投影公式。这些公式采用标准的凸几何方法，但……

    arXiv:2605.22940v2 Announce Type: replace-cross  Abstract: Entropy reduction is not a safety criterion: it can leave predictions unchanged or accompany increasing risk for a protected group. We study entropy-regulated learning under separately specified task, risk, and intervention constraints. A noise-calibrated log-determinant functional yields an exact spectral force formula and a metric decomposition into observable and prediction-null velocities. For scalar feedback, we characterize the complete interval of coefficients preserving a task-descent fraction and meeting an entropy-rate request; restricting intervention to observable directions can make that interval empty. For several protected risks, we derive an exact minimax certificate of feasibility within a specified subspace and norm budget, an exact robust extension for uncertain constraint rows, and a cone-projection formula for the entropy reduction remaining after protection. These formulas use standard convex geometry but 
    
[^280]: ECPO：用于可验证审计候选排序的证据耦合策略优化

    ECPO: Evidence-Coupled Policy Optimization for Verifier-Audited Candidate Ranking

    [https://arxiv.org/abs/2605.21993](https://arxiv.org/abs/2605.21993)

    提出证据耦合策略优化（ECPO），通过证据循环奖励将Top-K排序与基于片段的可审计证据耦合，显著提升验证者可审计的可追溯性，且增益在多个基准、骨干模型与评估器下均保持稳健。

    

    用于决策支持的排序系统应当提供可独立于模型排序分数进行审计的证据。我们研究了一种封闭名单设定，其中策略同时输出Top-K排序以及与候选相关联的、基于文本片段的证据包。证据耦合策略优化（ECPO）结合了列表级效用、证书有效性，以及一种证据循环奖励——该奖励在确定性恢复之前移除包含答案的字段。在MAVEN-ERE和RAMS数据集上，在所测试的溯源设置中，ECPO相比匹配的对照方法提升了验证者可审计的可追溯性、排序盲恢复能力以及人类对比支持度。这些增益在留出评估器、其他8B/14B骨干模型以及HotpotQA答案排序接口下依然保持。常规的NDCG指标和运行时间则仍是明确的权衡代价。

    arXiv:2605.21993v2 Announce Type: replace  Abstract: Ranking systems used for decision support should expose evidence that can be audited independently of the model's ranking scores. We study a closed-roster setting in which a policy jointly emits a Top-$K$ ranking and candidate-linked, span-grounded evidence bundles. Evidence-Coupled Policy Optimization (ECPO) combines listwise utility, certificate validity, and an evidence-cycle reward that removes answer-bearing fields before deterministic recovery. On MAVEN-ERE and RAMS, ECPO improves verifier-audited traceability, rank-blind recovery, and human comparative support relative to matched controls in the tested provenance settings. The gains persist under held-out evaluators, other 8B/14B backbones, and a HotpotQA answer-ranking interface. Ordinary NDCG and runtime remain explicit trade-offs.
    
[^281]: 使用错误设定模拟器的探索与实验

    Exploration and Experimentation with a Misspecified Simulator

    [https://arxiv.org/abs/2605.21458](https://arxiv.org/abs/2605.21458)

    该论文研究了在模拟器存在错误设定且真实系统动态发生漂移的情况下，是否、何时以及如何通过被动适应、持续探索或随机化实验等数据收集策略来重新校准模拟器。

    

    随机化实验是衡量因果效应以指导决策的标准方法，但成本高昂。根据历史数据校准的模拟器提供了一种更廉价的替代方案，但在发生漂移（即校准后系统动态发生变化）时，其策略可能变得次优。我们研究是否、何时以及如何收集数据以重新校准模拟器。我们考虑一个由多个单元并行运行并由单一操作者管理的系统。现实世界的系统是一个带有潜变量的部分可观测过程，而模拟器将其建模为完全可观测的马尔可夫过程，因此存在错误设定。两种数据收集策略利用当前策略：（i）从不重新校准，（ii）从在策略数据进行重新校准（被动适应）。另外两种策略通过尝试当前策略之外的动作来收集离策略数据：（iii）在所有单元上进行持续探索，以及（iv）在部分单元样本上进行随机化实验。

    arXiv:2605.21458v2 Announce Type: replace  Abstract: Randomized experiments are the standard for measuring causal effects that guide decisions, but are costly. Simulators calibrated to historical data offer a cheaper alternative, but their policies can become suboptimal under drift, a change in dynamics after calibration. We study if, when, and how to collect data for recalibrating the simulator. We consider a system with multiple units running in parallel and managed by a single operator. The real-world system is a partially observed process with a latent variable, while the simulator models it as a fully observed Markovian process, and thus is misspecified. Two data collection strategies exploit the current policy: (i) never recalibrating and (ii) recalibrating from on-policy data (passive adaptation). Two others collect off-policy data by trying actions outside the current policy: (iii) continuous exploration across all units and (iv) randomized experimentation on a sample of units 
    
[^282]: HyperPersona：面向基于文本的自动人格预测的多层超图框架

    HyperPersona: A Multi-Level Hypergraph Framework for Text-Based Automatic Personality Prediction

    [https://arxiv.org/abs/2605.17355](https://arxiv.org/abs/2605.17355)

    本文提出HyperPersona框架，利用多层超图结构显式建模文档、句子和词语三个层次的文本层级组织，从而提升基于文本的自动人格预测性能。

    

    作为一种现代商品，语言已成为一个蕴含丰富社会与心理学意义特征和概念的庞大宝库，反映了人们将思维、行为和情感模式编码为文字的方式。基于文本的自动人格预测（APP）旨在从语言行为中推断人格特质，为传统心理测量评估提供了一种可扩展的替代方案。尽管文本本质上具有层次性——文档层面捕捉全局特征，句子层面编码局部语义，词语层面提供细粒度的词汇信息——但大多数现有方法依赖于浅层、顺序式或单层的表示方式，忽略了书面语言的多层次结构。为解决这一问题，我们提出了HyperPersona，一个通过超图结构显式建模文本（文档、句子和词语）层次组织的框架，其中文档及其句子……

    arXiv:2605.17355v2 Announce Type: replace  Abstract: As a modern commodity, language has become a vast repository of socially and psychologically significant traits and concepts, reflecting the ways people encode pattern of thoughts, behaviors, and emotions into words. Text-based Automatic Personality Prediction (APP), seeks to infer personality from linguistic behavior, offering a scalable alternative to traditional psychometric assessments. Although text is inherently hierarchical, with the document-level capturing global features, the sentence-level encoding local semantics, and the word-level providing fine-grained lexical information, most existing approaches rely on shallow, sequential, or single-level representations that ignore the multi-level structure of written language. To address this, we propose HyperPersona, a framework that explicitly models the hierarchical organization of text (document, sentence, and word) through hypergraph structure, where a document and its senten
    
[^283]: 基于脑血流速度与机器学习算法的脑血管年龄预测

    Brain Vascular Age Prediction Using Cerebral Blood Flow Velocity and Machine Learning Algorithms

    [https://arxiv.org/abs/2605.16969](https://arxiv.org/abs/2605.16969)

    本研究利用经颅多普勒（TCD）测量的脑血流速度特征与机器学习回归模型来预测脑血管年龄，发现患有脑部疾病的受试者表现出脑血管加速衰老。

    

    以生理功能来定义血管年龄已成为对实际年龄进行分类与追踪的广泛研究焦点之一。经颅多普勒（TCD）是一种沿供养人脑的主要动脉测量脑血流速度的方法。本研究旨在利用从TCD提取的特征来估计实际年龄，并评估患有各种脑部疾病的受试者的加速衰老情况。我们预测，当以健康受试者训练的各种回归模型进行测试时，患有各种脑部疾病的受试者会表现出加速的脑血管衰老。本研究采用颅内压波形形态分析与聚类（MOCAIP）算法，分析了168名健康受试者和277名患病受试者的大脑中动脉双侧TCD记录。MOCAIP生成的特征和心率变异性特征被用作回归模型的输入特征，以预测（脑血管年龄）。

    arXiv:2605.16969v3 Announce Type: replace  Abstract: Defining vascular age in terms of physiological function has become one focal point of the extensive studies to categorize and track chronological age. Transcranial Doppler (TCD) is a method by which cerebral blood flow velocity is measured along the major arteries feeding the human brain. This study aims to use features extracted from TCD to estimate chronological age and assess accelerated aging in subjects with various brain diseases. We predict subjects with various brain diseases to present with accelerated cerebrovascular aging when tested on various regression models trained by healthy subjects. 168 healthy subjects and 277 diseased subjects with bilateral TCD recordings of the middle cerebral artery were analyzed using the Morphological Analysis and Clustering of Intracranial Pressure (MOCAIP) algorithm. MOCAIP-generated features and heart rate variability features were used as input features for regression models to predict 
    
[^284]: RaPD：基于语义增强隐式表示的分辨率无关像素扩散

    RaPD: Resolution-Agnostic Pixel Diffusion via Semantics-Enriched Implicit Representations

    [https://arxiv.org/abs/2605.15908](https://arxiv.org/abs/2605.15908)

    RaPD提出在连续的神经图像场潜空间中进行像素扩散，通过语义表示引导和坐标查询注意力渲染器，使单个去噪潜变量可通过仅改变查询坐标按任意分辨率渲染，且扩散成本保持固定。

    

    自然图像本质上是连续的，但大多数生成模型在离散网格上合成图像，限制了分辨率灵活的生成。连续神经场虽然能够实现无分辨率限制的渲染，但先前的方法仅在解码阶段将连续性作为插值模块引入，导致生成潜空间仍然是离散化的且面向重建的。我们提出了RaPD（分辨率无关像素扩散），它在连续的神经图像场（NIF）潜空间中执行扩散。RaPD通过用于生成感知潜空间学习的语义表示引导，以及用于坐标条件化、尺度感知渲染的坐标查询注意力渲染器，弥合了重建与生成之间的差距。通过仅改变查询坐标，单个去噪后的潜变量即可按任意分辨率进行渲染，同时保持扩散成本固定。实验证明了该方法具有卓越的生成质量和分辨率可扩展性。

    arXiv:2605.15908v2 Announce Type: replace-cross  Abstract: Natural images are continuous, yet most generative models synthesize them on discrete grids, limiting resolution-flexible generation. Continuous neural fields enable resolution-free rendering, but prior methods introduce continuity only at the decoding stage as an interpolation module, leaving the generative latent space discretized and reconstruction-oriented. We propose RaPD (Resolution-agnostic Pixel Diffusion), which performs diffusion in a continuous Neural Image Field (NIF) latent space. RaPD bridges this reconstruction generation gap with Semantic Representation Guidance for generation-aware latent learning and a Coordinate-Queried Attention Renderer for coordinate-conditioned, scale-aware rendering. A single denoised latent can be rendered at arbitrary resolutions by changing only the query coordinates, keeping diffusion cost fixed. Experiments demonstrate superior generation quality and resolution scalability.
    
[^285]: CrystalReasoner：面向性质条件化晶体结构生成的推理与强化学习方法

    CrystalReasoner: Reasoning and RL for Property-Conditioned Crystal Structure Generation

    [https://arxiv.org/abs/2605.14344](https://arxiv.org/abs/2605.14344)

    CrystalReasoner 提出了一个端到端的大语言模型框架，通过将晶体学对称性、配位环境和预测物理性质等物理先验作为“思考 token”进行推理，并利用多目标密集奖励的强化学习进行对齐，从而从自然语言指令直接生成有效且具备目标性质的晶体结构。

    

    生成式建模已成为晶体结构发现的一种有前景的方法。然而，现有的基于大语言模型（LLM）的生成模型在底层原子精度方面表现不佳，而基于扩散的方法则难以整合高层科学知识，导致生成的结构往往无效、不稳定或不具备理想的性质。为解决这一差距，我们提出了 CrystalReasoner（CrysReas），一个端到端的 LLM 框架，能够通过推理与对齐从自然语言指令生成晶体结构。CrysReas 引入物理先验作为“思考 token”，在生成原子坐标之前先对晶体学对称性、局部配位环境以及预测的物理性质进行推理，从而在自然语言与三维结构之间架起桥梁。随后，CrysReas 采用带有多目标、密集奖励函数的强化学习（RL），使生成结果与性质要求对齐。

    arXiv:2605.14344v3 Announce Type: replace  Abstract: Generative modeling has emerged as a promising approach for crystal structure discovery. However, existing LLM-based generative models struggle with low-level atomic precision, while diffusion-based methods fall short in integrating high-level scientific knowledge. As a result, generated structures are often invalid, unstable, or do not possess desirable properties. To address this gap, we propose CrystalReasoner (CrysReas), an end-to-end LLM framework that generates crystal structures from natural language instructions through reasoning and alignment. CrysReas introduces physical priors as thinking tokens, which include crystallographic symmetry, local coordination environments and predicted physical properties before generating atomic coordinates. This bridges the gap between natural language and 3D structures. CrysReas then employs reinforcement learning (RL) with a multi-objective, dense reward function to align generation with p
    
[^286]: CIDR：一个用于软件工程研究的大规模工业源代码数据集

    CIDR: A Large-Scale Industrial Source Code Dataset for Software Engineering Research

    [https://arxiv.org/abs/2605.12153](https://arxiv.org/abs/2605.12153)

    本文发布了CIDR——一个包含4,687个仓库、84种编程语言、超10亿行代码的大规模匿名化工业源代码数据集，为代码智能和软件工程研究提供真实企业级代码资源。

    

    我们提出了精选工业开发者仓库数据集（CIDR），这是一个从工业合作伙伴处收集的大规模真实世界软件仓库数据集。该数据集包含4,687个仓库，涵盖84种编程语言，总计10.33亿行原始代码（6.703亿行逻辑代码），同时提供仓库级别的结构化元数据、完整的版本控制历史，以及持续集成使用情况和自动化测试存在与否等工程实践属性。所有仓库均通过专门为此目的开发的多阶段流水线进行收集、过滤和匿名化处理。我们还报告了一项探索性微调研究，将一个30亿参数的代码语言模型适配到CIDR上，并量化了其对留出的企业代码的影响。CIDR旨在支持代码智能、软件质量分析、开发者工具及相关软件工程领域的研究。

    arXiv:2605.12153v3 Announce Type: replace-cross  Abstract: We present the Curated Industrial Developer Repository (CIDR), a large-scale dataset of real-world software repositories collected from industrial partners. The dataset comprises 4,687 repositories spanning 84 programming languages, totaling 1,033 million raw lines of code (670,3 million logical lines), along with structured metadata at the repository level, full version control history, and engineering-practice attributes such as continuous integration usage and the presence of automated tests. All repositories were collected, filtered, and anonymized through a multi-stage pipeline developed specifically for this purpose. We additionally report an exploratory fine-tuning study that adapts a 3-billion-parameter code language model to CIDR and quantifies the effect on held-out enterprise code. CIDR is intended to support research in code intelligence, software quality analysis, developer tooling, and related software engineering
    
[^287]: NEXUS：学习安全约束与动作模型以实现安全且稳健的具身规划

    NEXUS: Learning Safety Constraints and Action Models for Safe and Robust Embodied Planning

    [https://arxiv.org/abs/2605.09387](https://arxiv.org/abs/2605.09387)

    NEXUS 提出一个模块化神经符号框架，通过联合维护 PDDL 领域模型与 LTL 安全规范，利用执行失败修复动作模式、并将检测到的危险转化为可复用的 LTL 安全约束，从而实现安全且稳健的具身规划。

    

    大语言模型（LLM）使具身智能体能够根据自然语言指令进行规划，但其生成的动作可能在物理上不可行或具有危险性。现有的神经符号学习方法主要致力于精炼执行模型，而形式化安全方法则侧重于获取或强制执行安全约束。这种分离限制了执行修正与安全知识在不同任务间的复用。我们提出了 NEXUS，一个模块化的神经符号框架，它联合维护一个 PDDL 领域模型和一个 LTL 安全规范。执行失败被用于修复动作模式，而检测到的危险则通过带等价共识的分层语义与转换采样被转化为可复用的 LTL 约束。累积的约束在符号搜索过程中通过 LTL 推进来强制执行；通过比较受约束与不受约束搜索的结果，可以区分物理不可达性与由安全引发的拒绝。实验在……（原文摘要在此处截断）

    arXiv:2605.09387v2 Announce Type: replace  Abstract: Large language models (LLMs) enable embodied agents to plan from naturallanguage instructions, yet their actions may be physically infeasible or hazardous. Existing neuro-symbolic learners primarily refine execution models, while formal safety methods acquire or enforce safety constraints. This separation limits the reuse of both execution corrections and safety knowledge across tasks. We present NEXUS, a modular neuro-symbolic framework that jointly maintains a PDDL domain model and an LTL safety specification. Execution failures are used to repair action schemas, whereas detected hazards are converted into reusable LTL constraints through hierarchical semantic and translation sampling with equivalence consensus. Accumulated constraints are enforced through LTL progression during symbolic search; comparing constrained and unconstrained search outcomes distinguishes physical unreachability from safety-induced refusal. Experiments on 
    
[^288]: STRIDE：面向时间序列预测的推理感知训练

    STRIDE: Reasoning-Aware Training for Time Series Forecasting

    [https://arxiv.org/abs/2605.08625](https://arxiv.org/abs/2605.08625)

    STRIDE提出了一种新框架，将蒸馏后的LLM推理以连续跨模态先验的形式原生注入时间序列基础模型的嵌入空间，在保持数值预测能力的同时引入推理能力，并登上GIFT-Eval排行榜第一。

    

    时间序列基础模型在数值预测方面表现出色，但作为黑盒运行，缺乏定性推理能力。相反，直接将大语言模型（LLM）应用于时间数据会引入模态鸿沟：文本分词器会碎片化连续数值，破坏数学关系并导致序列长度爆炸，从而带来计算开销。为解决这一问题，我们提出了STRIDE（Strategic Time-series Reasoning Injected via Distilled Embeddings，通过蒸馏嵌入注入的战略性时间序列推理），这是一个将LLM推理原生集成到TSFM连续嵌入空间的新颖框架。STRIDE不使用离散token，而是将推理轨迹蒸馏到一个轻量级LLM中，并将其池化后的隐状态动态投影为连续的跨模态先验，注入数值编码器。通过交叉熵和分位数损失进行联合优化，STRIDE可以轻松适配多种架构（如Chronos-2、Timer-S1）。STRIDE在GIFT-Eval排行榜上排名第一。

    arXiv:2605.08625v2 Announce Type: replace-cross  Abstract: Time Series Foundation Models (TSFMs) excel at numerical forecasting but operate as black boxes lacking qualitative reasoning. Conversely, applying LLMs directly to temporal data introduces a modality gap: text tokenizers fragment continuous numerical values, degrading mathematical relationships and exploding sequence lengths, leading to computational overhead. To resolve this, we introduce STRIDE (Strategic Time-series Reasoning Injected via Distilled Embeddings), a novel framework natively integrating LLM reasoning into the continuous embedding space of TSFMs. Instead of discrete tokens, STRIDE distills reasoning traces into a lightweight LLM and dynamically projects its pooled hidden states as a continuous cross-modal prior into the numerical encoder. Jointly optimized via cross-entropy and quantile losses, STRIDE readily attaches to diverse architectures (e.g., Chronos-2, Timer-S1). STRIDE ranks #1 on the GIFT-Eval leaderbo
    
[^289]: 协同至关重要：面向协作式多智能体强化学习的过程感知评估

    Coordination Matters: Process-Aware Evaluation of Cooperative Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2605.06557](https://arxiv.org/abs/2605.06557)

    本文提出了一个过程感知的多智能体强化学习评估框架，通过干扰、协同任务完成、参与度和协调效率等诊断指标补充传统聚合性能指标，揭示了回报相近的策略在协调行为上可能存在显著差异。

    

    协作式多智能体强化学习（MARL）基准测试通常强调回报、成功率或完成时间等聚合性结果。虽然这些指标至关重要，但它们可能掩盖智能体之间如何进行协调，并且无法区分通过不同交互过程达到相似结果的策略。我们提出了一个过程感知的评估框架，通过干扰、协同任务完成、参与度和协调效率等诊断指标来补充聚合性能的评估。我们在STAT——一个受控的空间任务分配测试平台——中评估了该框架，并检验了其在两个成熟的协作式MARL基准测试中更广泛的适用性。在多种基于价值和演员-评论家的方法中，我们发现回报几乎相同的策略在协调行为上可能存在显著差异。STAT中的受控规模扩展实验揭示了空间、任务和智能体规模扩展各自不同的过程特征。

    arXiv:2605.06557v2 Announce Type: replace-cross  Abstract: Cooperative multi-agent reinforcement learning (MARL) benchmarks commonly emphasize aggregate outcomes such as return, success rate, or completion time. While essential, these metrics can obscure how agents coordinate and fail to distinguish policies that achieve similar outcomes through different interaction processes. We propose a process-aware evaluation framework that complements aggregate performance with diagnostics of interference, coordinated task completion, participation, and coordination efficiency. We evaluate the framework in STAT, a controlled spatial task-allocation testbed, and examine its broader applicability in two established cooperative MARL benchmarks. Across diverse value-based and actor-critic methods, we find policies with nearly identical return can exhibit substantial differences in coordination behavior. Controlled scaling in STAT reveals distinct process signatures for spatial, task, and agent scali
    
[^290]: MOSAIC：面向科学时间序列的基于稀疏可加可辨识因果学习的模块发现

    MOSAIC: Module Discovery via Sparse Additive Identifiable Causal Learning for Scientific Time Series

    [https://arxiv.org/abs/2605.05524](https://arxiv.org/abs/2605.05524)

    MOSAIC 提出了一种稀疏时序 VAE 方法，将科学观测固有的语义可解释性传递到具有可辨识性保证的潜在空间中，从而在底层机制未知的科学时间序列中实现可解释的模块发现。

    

    因果表征学习（CRL）旨在恢复具有可辨识性保证的潜在变量，在适当的假设下，通常可辨识至排列和逐分量重参数化的等价程度。然而，可辨识性并不等于可解释性：潜在语义通常是在事后通过与已知的真实因子对齐来赋予的。这一局限性在科学时间序列中尤为突出，因为其底层机制未知，而发现可解释的结构本身就是一个主要目标。相比之下，科学观测（如残基对距离、气候指数或过程传感器数据）本质上是具有语义的，因为它们对应于命名的物理量。这引出了一个关键问题：能否将观测的可解释性传递到可辨识的潜在空间中？我们提出了 MOSAIC（通过稀疏可加可辨识因果学习进行模块发现），这是一种稀疏时序 VAE，它集成了……

    arXiv:2605.05524v2 Announce Type: replace-cross  Abstract: Causal representation learning (CRL) seeks to recover latent variables with identifiability guarantees, typically up to permutation and component-wise reparameterization under appropriate assumptions. However, identifiability does not imply interpretability: latent semantics are typically assigned post hoc by alignment with known ground-truth factors. This limitation is particularly acute in scientific time series, where underlying mechanisms are unknown and discovering interpretable structure is a primary goal. In contrast, scientific observations (such as residue-pair distances, climate indices, or process sensors) are inherently semantic, as they correspond to named physical quantities. This raises a key question: can the interpretability of observations be transferred to the identifiable latent space? We propose MOSAIC (Module discovery via Sparse Additive Identifiable Causal learning), a sparse temporal VAE that integrates
    
[^291]: FaithfulFaces：面向文生视频生成的姿态忠实面部身份保持

    FaithfulFaces: Pose-Faithful Facial Identity Preservation for Text-to-Video Generation

    [https://arxiv.org/abs/2605.04702](https://arxiv.org/abs/2605.04702)

    提出FaithfulFaces框架，通过姿态共享身份对齐器将单视角输入映射为带显式欧拉角嵌入的全局面部姿态表示，为文生视频生成提供姿态忠实的面部先验，显著改善大姿态变化和面部遮挡下的身份保持效果。

    

    身份保持的文生视频生成（IPT2V）使用户能够生成具有一致人物面部身份的多样且富有想象力的视频。尽管近期取得了一定进展，现有方法在大幅面部姿态变化或面部遮挡的情况下常常出现明显的身份失真。本文提出FaithfulFaces，一个姿态忠实的面部身份保持学习框架，用于改进复杂动态场景下的IPT2V。FaithfulFaces的核心是一个姿态共享身份对齐器，它通过姿态共享字典和姿态变化-身份不变性约束，对不同视角间的面部姿态进行精细化与对齐。通过将单视角输入映射为带有显式欧拉角嵌入的全局面部姿态表示，FaithfulFaces提供了一个姿态忠实的面部先验，引导生成基础模型实现鲁棒的身份保持生成。特别地，我们开发了一个专门的流程来……

    arXiv:2605.04702v2 Announce Type: replace-cross  Abstract: Identity-preserving text-to-video generation (IPT2V) empowers users to produce diverse and imaginative videos with consistent human facial identity. Despite recent progress, existing methods often suffer from significant identity distortion under large facial pose variations or facial occlusions. In this paper, we propose FaithfulFaces, a pose-faithful facial identity preservation learning framework to improve IPT2V in complex dynamic scenes. The key of FaithfulFaces is a pose-shared identity aligner that refines and aligns facial poses across distinct views via a pose-shared dictionary and a pose variation-identity invariance constraint. By mapping single-view inputs into a global facial pose representation with explicit Euler angle embeddings, FaithfulFaces provides a pose-faithful facial prior that guides generative foundations toward robust identity-preserving generation. In particular, we develop a specialized pipeline to 
    
[^292]: 学习揭示了低秩RNN中的不可见结构

    Learning reveals invisible structure in low-rank RNNs

    [https://arxiv.org/abs/2605.04115](https://arxiv.org/abs/2605.04115)

    本文推导出低维常微分方程系统来精确描述低秩RNN的学习动力学，并发现学习过程可分为决定网络功能的“损失可见重叠”和不影响功能但描述学习所必需的“损失不可见重叠”。

    

    神经系统中的学习源于突触的改变，这些改变重塑了行为背后的表征。虽然低秩循环神经网络（RNN）已成为连接网络结构与功能的强大框架，但对其学习过程的理论理解仍然难以捉摸。在此，我们将低秩框架从神经活动扩展到学习过程，通过在简化的重叠空间中直接推导梯度下降动力学。我们构建了一个封闭形式的低维常微分方程（ODE）系统来支配该空间中的学习，该系统对于线性RNN是精确的，对于大-$N$高斯极限下的非线性RNN是渐近精确的。我们分析的核心在于对两类重叠的区分：损失可见重叠（loss-visible overlaps），它们完全决定网络的活动、输出和损失；以及损失不可见重叠（loss-invisible overlaps），它们不影响网络功能，但却是描述学习过程所必需的。我们阐明了这种分解所带来的后果。

    arXiv:2605.04115v2 Announce Type: replace-cross  Abstract: Learning in neural systems arises from synaptic changes that reshape the representations underlying behavior. While low-rank recurrent neural networks (RNNs) have emerged as a powerful framework for linking connectivity to function, a theoretical understanding of their learning process remains elusive. Here, we extend the low-rank framework from activity to learning by deriving gradient-descent dynamics directly in a reduced overlap space. We formulate a closed-form, low-dimensional system of ODEs that governs learning in this space, exact for linear RNNs and asymptotically exact for nonlinear RNNs in the large-$N$ Gaussian limit. Central to our analysis is a distinction between two classes of overlaps: loss-visible overlaps, which fully determine network activity, output, and loss, and loss-invisible overlaps, which do not affect function but are required to describe learning. We illustrate the consequences of this decompositi
    
[^293]: RamanBench：一个面向拉曼光谱机器学习的大规模基准测试

    RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy

    [https://arxiv.org/abs/2605.02003](https://arxiv.org/abs/2605.02003)

    本文提出了RamanBench——首个大规模、完全可复现的拉曼光谱机器学习基准，统一整合了四个领域74个数据集共325,668条光谱，并在标准化协议下对28个模型进行了系统评估。

    

    机器学习（ML）已经变革了许多科学领域，然而一些关键应用仍然缺乏标准化的基准测试。拉曼光谱作为一种广泛使用的无创分子分析技术，正是这样一个领域——其进展受到数据集碎片化、评估标准不一致以及模型无法捕捉光谱数据结构等因素的限制。我们提出了RamanBench，这是首个大规模、完全可复现的拉曼光谱机器学习基准测试，包含简化的数据访问、评估协议与代码，以及一个实时排行榜。该基准统一整合了四个领域的74个数据集（其中16个为随本基准首次发布），共包含325,668条光谱，涵盖多种实验条件下的分类和回归任务。我们在标准化协议下对28个模型进行了基准测试，包括经典方法（如PLS）、拉曼光谱专用模型（如RamanNet）以及表格基础模型（TFM，如TabPFN）。

    arXiv:2605.02003v3 Announce Type: replace  Abstract: Machine Learning (ML) has transformed many scientific fields, yet key applications still lack standardized benchmarks. Raman spectroscopy, a widely used technique for non-invasive molecular analysis, is one such field where progress is limited by fragmented datasets, inconsistent evaluation, and models that fail to capture the structure of spectral data. We introduce RamanBench, the first large-scale, fully reproducible benchmark for ML on Raman spectroscopy, consisting of streamlined data access, evaluation protocols and code, as well as a live leaderboard. It unifies 74 datasets (including 16 first released with this benchmark) across four domains, comprising 325,668 spectra and spanning classification and regression tasks under diverse experimental conditions. We benchmark 28 models under a standardized protocol, including classical methods (e.g., PLS), Raman-specific (e.g., RamanNet), Tabular Foundation Model (TFM) (e.g., TabPFN)
    
[^294]: 基于状态历史与全局反馈的可解释经验学习

    Interpretable experiential learning based on state history and global feedback

    [https://arxiv.org/abs/2605.00940](https://arxiv.org/abs/2605.00940)

    提出了一种基于状态历史与全局反馈的可解释经验学习模型，通过带效用值和证据计数的状态转移图学习行为模型，适用于资源受限环境下的强化学习，并在Atari Breakout基准上取得了与神经网络方案相当的性能。

    

    本文提出了一种基于状态历史与全局反馈的新型可解释经验学习模型。该模型能够学习一种以状态集合之间的转移图表示的行为模型，其中每条转移都附有效用值和证据计数。该模型有望适用于解决资源受限环境下的强化学习问题。该模型在OpenAI Gym的Atari Breakout基准上进行了全面评估，其性能可与一些已知的基于神经网络的解决方案相媲美。

    arXiv:2605.00940v2 Announce Type: replace-cross  Abstract: A new interpretable experiential learning model based on state history and global feedback is presented. It is capable of learning a behavioral model represented by a transition graph between sets of states, with transitions attributed with utility and evidence count. This model is expected to be suitable for solving reinforcement learning problem in resource-constrained environments. The model was thoroughly evaluated on the OpenAI Gym Atari Breakout benchmark, demonstrating performance comparable to some known neural network-based solutions.
    
[^295]: CastFlow：在智能体时间序列预测中架起证据获取与准确预测之间的桥梁

    CastFlow: Bridging Evidence Acquisition and Accurate Prediction in Agentic Time Series Forecasting

    [https://arxiv.org/abs/2604.27840](https://arxiv.org/abs/2604.27840)

    CastFlow提出了一种角色专门化的智能体时间序列预测框架，通过经验记忆引导冻结的大语言模型调度多视角工具获取证据，再由轻量级专门化模型进行基于证据的时间推理，从而有效衔接证据获取与准确的数值预测。

    

    时间序列预测是时序数据分析中的一项基础任务，而基于大语言模型（LLM）的智能体通过自主工具交互和基于证据的推理提供了一种有前景的范式。然而，仍存在两个根本性挑战：如何通过自适应的工具交互获取与预测相关的证据，以及如何将此类证据转化为准确的数值预测。为应对这些挑战，我们提出了CastFlow，这是一个角色专门化的智能体预测框架，连接了通用工具推理与专门化的时间序列预测。具体而言，CastFlow引入了一个经验记忆模块，从训练数据中提炼成功的工具使用轨迹，指导冻结的LLM调度多视角工具包以完成证据获取。随后，一个轻量级的专门化LLM执行基于证据的时间推理，以细化集成预测先验。为增强这一能力……

    arXiv:2604.27840v3 Announce Type: replace-cross  Abstract: Time series forecasting is a fundamental task in temporal data analytics, and large language model (LLM)-based agents offer a promising paradigm through autonomous tool interaction and evidence-grounded reasoning. However, two fundamental challenges remain: how to acquire forecast-relevant evidence through adaptive tool interactions, and how to translate such evidence into accurate numerical predictions. To address these challenges, we propose CastFlow, a role-specialized agentic forecasting framework that bridges general-purpose tool reasoning and specialized temporal forecasting. Specifically, CastFlow introduces an Experience Memory that distills successful tool-use trajectories from training data, guiding a frozen LLM to orchestrate a multi-view toolkit for evidence acquisition. A lightweight specialized LLM then performs evidence-grounded temporal reasoning to refine an ensemble forecasting prior. To enhance this capabilit
    
[^296]: 通过序列压缩低秩子空间实现激活高效的微调

    Activation-Efficient Fine-Tuning via Sequence-Compressed Low-Rank Subspaces

    [https://arxiv.org/abs/2604.22783](https://arxiv.org/abs/2604.22783)

    LARS通过在序列级低秩子空间中执行适配，将存储的适配器激活中间量复杂度从O(BSRL)降至O(BRL)，相比LoRA在GPU和CPU上分别节省23%和47.8%的内存，且节省幅度随上下文长度增加而提升。

    

    参数高效微调（PEFT）已成为适配大型语言模型（LLM）的标准方法。然而，尽管许多PEFT方法减少了可训练参数的数量，它们仍然受到随序列长度线性扩展的token级别激活中间量的内存限制，这限制了它们在长上下文和设备端适配场景中的实用性。与先前保留token级别适配器激活的记忆高效PEFT方法不同，LARS（低内存激活秩子空间）通过在序列级别的低秩子空间内执行适配，将存储的适配器激活中间量的序列长度扩展复杂度从O(BSRL)降低到O(BRL)。通过池化特征提取、低秩子空间调制和残差投影，LARS在S=1024时相比LoRA将总体内存占用在GPU上减少23%，在CPU上减少47.8%，且相对节省随上下文长度的增加而提升。实验在推理……（原文摘要在此处截断）

    arXiv:2604.22783v2 Announce Type: replace-cross  Abstract: Parameter-Efficient Fine-Tuning (PEFT) has become the standard approach for adapting large language models (LLMs). However, despite reducing trainable parameter counts, many PEFT methods remain memory-bound by token-level activation intermediates that scale linearly with sequence length, limiting their practicality for long-context and on-device adaptation. Unlike prior memory-efficient PEFT approaches that preserve token-level adapter activations, LARS (Low-memory Activation-Rank Subspace) reduces the sequence-length scaling of stored adapter activation intermediates from O(BSRL) to O(BRL) by performing adaptation within a sequence-level low-rank subspace. Through pooled feature extraction, low-rank subspace modulations, and residual projection, LARS reduces the overall memory footprint by 23% on GPUs and 47.8% on CPUs relative to LoRA at S = 1024, with relative savings increasing with context length. Experiments across reason
    
[^297]: 心智理论基准测试的算法原语

    Algorithmic Primitives of Theory of Mind Benchmarks

    [https://arxiv.org/abs/2604.17819](https://arxiv.org/abs/2604.17819)

    该论文提出神经符号方法PDDL-Mind，通过将叙事解析为PDDL并由程序执行状态更新，揭示了主流心智理论基准实际上主要在测试解析和状态跟踪等算法能力而非真正的心智理论，并借此将GPT-4o的性能提升至接近人类水平。

    

    心智理论（ToM）基准测试真的在测试心智理论，还是实际上在测试解析和算法化状态跟踪？尽管各基准在模态和对话长度上有所差异，本文发现它们其实重新编码了相同的少数几种状态转移算法。我们提出了PDDL-Mind，一种神经符号方法，通过将叙事解析为规划领域定义语言（PDDL），并针对一组预定义的更新算法执行候选动作，从而显式地恢复这些算法以及ToM基准测试的上游归纳偏置。这样做能够为模型提供经程序验证的状态轨迹。我们利用该方法带来的性能提升来审计我们的归纳偏置究竟吸收了多少基准测试的难度。在MMToM、MuMA和FanToM三个基准上，仅将解析任务外包给程序就能使性能接近强大的智能体流水线，而将状态跟踪外包给PDDL则使GPT-4o的性能提升至接近人类水平。我们展示了……

    arXiv:2604.17819v2 Announce Type: replace-cross  Abstract: Do Theory of Mind (ToM) benchmarks really test theory of mind, or are they actually testing parsing and algorithmic state tracking? While benchmarks vary modality and dialogue length, this paper finds that they re-encode the same few state transition algorithms. We create PDDL-Mind, a neuro-symbolic approach that recovers such algorithms and upstream inductive biases of ToM benchmarks explicitly by parsing narratives into the Planning Domain Definition Language (PDDL) and executes candidate actions against a predefined set of update algorithms. Doing so provides program-verified state traces to the model. We use our method's performance gain to audit how much of a benchmark's difficulty our inductive bias absorbs. Across MMToM, MuMA, and FanToM, outsourcing parsing to programs alone brings performance close to strong agentic pipelines, while outsourcing state tracking to PDDL moves GPT-4o to close-to-human performance. We demon
    
[^298]: OmniTrace：面向全模态大语言模型的生成时溯源统一框架

    OmniTrace: A Unified Framework for Generation-Time Source Attribution in Omni-Modal LLMs

    [https://arxiv.org/abs/2604.13073](https://arxiv.org/abs/2604.13073)

    OmniTrace 提出一种统一框架，通过读取全模态大模型后层的注意力分布，将生成的每个句子实时归因到文本、图像和音视频片段等源单元，无需额外前向传播或针对模态的调优。

    

    全模态语言模型可以回答关于多模态文档、录音或视频的问题，但它无法展示答案来自输入的哪个部分。让模型引用其证据只能得到它声称使用的内容；而要测量它实际使用的内容，迄今为止需要通过删除部分输入并多次重新运行模型来实现，其成本随音频和视频时长的增加而不断增长。我们证明生成过程本身已经包含了大部分此类信息。OmniTrace 通过将模型后层的注意力读作各源单元上的质量分布，将每个生成的句子归因到源单元（文本块、图像以及音频或视频的秒级片段），无需额外的前向传播，也无需针对各模态进行调优；在可获得消融数据的情况下，还可以训练注意力头权重。在固定的因果测试设置下，OmniTrace 所选出的单元在 Qwen3-Omni-3 的视频、音频和图文问答任务中，对整体响应以及单个句子都具有因果重要性。

    arXiv:2604.13073v2 Announce Type: replace-cross  Abstract: An omni-modal language model can answer a question about a multimodal document, a recording, or a video, but it cannot show which part of the input the answer came from. Asking the model to cite its evidence yields what it says it used; measuring what it used has so far required rerunning it many times with parts of the input removed, a cost that grows with every second of audio and video. We show that the generation pass already holds most of this information. OmniTrace attributes each generated sentence to source units (text chunks, images, and seconds of audio or video) by reading late-layer attention as mass over units with no extra forward pass and no per-modality tuning; head weights can be trained where ablation data are available. With fixed causal test setting, the units OmniTrace selects are causally important for responses and for individual sentences in video, audio, and image-text question answering on Qwen3-Omni-3
    
[^299]: 高斯分布扰动下KL散度的最坏情形最优 $\sqrt{\epsilon}$ 阶稳定性

    Worst-Case Optimal $\sqrt{\epsilon}$-Rate Stability of KL Divergence under Perturbations of Gaussian Distributions

    [https://arxiv.org/abs/2604.11026](https://arxiv.org/abs/2604.11026)

    本文在比较分布 $P$ 不必为高斯分布的情况下，首次建立了高斯参考分布受扰动时KL散度变化的最坏情形最优 $\sqrt{\epsilon}$ 阶稳定性界，突破了现有松弛三角不等式要求三个分布均为高斯分布的局限，可应用于流生成模型的分布外检测等场景。

    

    我们为库尔贝克-莱布勒（KL）散度在其高斯参考分布受到扰动时建立了一个稳定性界。现有的高斯分布之间KL散度的松弛三角不等式要求三个分布全部为高斯分布，这限制了它们在诸如基于流的生成模型的分布外检测等应用中的使用。设 $\mathcal{N}_1$ 和 $\mathcal{N}_2$ 为 $d$ 维高斯分布，$P$ 为任意满足 $\mathbb{E}_P[\|X\|^2]<\infty$ 且 $\mathrm{KL}(P\Vert \mathcal{N}_1)<\infty$ 的 $d$ 维分布。若当 $0<\epsilon<1/12$ 时有 $\mathrm{KL}(\mathcal{N}_1\Vert\mathcal{N}_2)\leq\epsilon$，我们给出了 $\left|\mathrm{KL}(P\Vert\mathcal{N}_1)-\mathrm{KL}(P\Vert\mathcal{N}_2)\right|$ 的上界。在适当的均匀有界条件下，当 $\epsilon\rightarrow 0$ 时，该界为 $\frac{\sqrt{6d\epsilon}}{2}+K\sqrt{\epsilon}+o(\sqrt{\epsilon})$。

    arXiv:2604.11026v4 Announce Type: replace-cross  Abstract: We establish a stability bound for the Kullback-Leibler (KL) divergence when its Gaussian reference distribution is perturbed. Existing relaxed triangle inequalities of KL divergence between Gaussian distributions require all three distributions to be Gaussian, which limits their use in applications such as out-of-distribution detection with flow-based generative models. Let $\mathcal{N}_1$ and $\mathcal{N}_2$ be $d$-dimensional Gaussian distributions, and let $P$ be any $d$-dimensional distribution satisfying $\mathbb{E}_P[\|X\|^2]<\infty$ and $\mathrm{KL}(P\Vert \mathcal{N}_1)<\infty$. If $\mathrm{KL}(\mathcal{N}_1\Vert\mathcal{N}_2)\leq\epsilon$ for $0<\epsilon<1/12$, we give an upper bound of $\left|\mathrm{KL}(P\Vert\mathcal{N}_1)-\mathrm{KL}(P\Vert\mathcal{N}_2)\right|$. Under suitable uniform boundedness conditions, the bound is $ \frac{\sqrt{6d\epsilon}}{2}+K\sqrt{\epsilon}+o(\sqrt{\epsilon})$ as $\epsilon\rightarrow 0$
    
[^300]: 通过多智能体可解释性检测多智能体串谋

    Detecting Multi-Agent Collusion Through Multi-Agent Interpretability

    [https://arxiv.org/abs/2604.01151](https://arxiv.org/abs/2604.01151)

    提出NARCBench基准与五种探针技术，首次利用模型内部表示在群体层面检测LLM多智能体系统中的串谋行为，并将其形式化为分布式异常检测问题。

    

    随着LLM智能体越来越多地部署在多智能体系统中，它们带来了可能规避标准人类监督形式的隐蔽协调风险。虽然对模型激活进行线性探针在单智能体场景中检测欺骗方面已显示出前景，但串谋本质上是一种多智能体现象，利用内部表示来检测智能体之间的串谋仍未被探索。我们提出了NARCBench，一个用于评估环境分布偏移下串谋检测的基准，并提出五种探针技术，通过聚合每个智能体的欺骗分数在群体层面进行场景分类，并在四个开放权重模型（Qwen3-32B、Llama-3.1-70B、DeepSeek-R1 32B、GPT-OSS-20B）和六种探针架构上进行了评估。我们将该问题构建为分布式异常检测问题，识别出三种映射到不同异常类型和检测范式的串谋特征。

    arXiv:2604.01151v3 Announce Type: replace  Abstract: As LLM agents are increasingly deployed in multi-agent systems, they introduce risks of covert coordination that may evade standard forms of human oversight. While linear probes on model activations have shown promise for detecting deception in single-agent settings, collusion is inherently a multi-agent phenomenon, and the use of internal representations for detecting collusion between agents remains unexplored. We introduce NARCBench, a benchmark for evaluating collusion detection under environment distribution shift, and propose five probing techniques that aggregate per-agent deception scores to classify scenarios at the group level, evaluated across four open-weight models (Qwen3-32B, Llama-3.1-70B, DeepSeek-R1 32B, GPT-OSS-20B) and six probe architectures. We frame this as a distributed anomaly detection problem, identifying three collusion signatures that map onto distinct anomaly types and detection paradigms. Every model rea
    
[^301]: 迈向大语言模型中可靠的真实对齐不确定性估计

    Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models

    [https://arxiv.org/abs/2604.00445](https://arxiv.org/abs/2604.00445)

    提出真实锚定方法，一种轻量级后验校准技术，通过将大语言模型的原始不确定性估计分数映射为真实对齐的正确性概率，解决了UE指标在不同模型和数据集上性能不稳定的“代理失效”问题，即使在有噪声的少样本监督下也能持续提升校准效果。

    

    不确定性估计（UE）旨在量化大语言模型（LLM）输出中的不确定性，以标记潜在的幻觉并支持可靠部署。然而，UE指标在不同模型和数据集上往往表现出不稳定的性能，限制了其实际效用。我们将这一现象形式化为“代理失效”（proxy failure），其产生的原因是UE指标源自模型行为，而非明确地基于事实正确性。我们证明，当UE分数与正确性之间的互信息较低时，该分数的判别能力必然接近随机水平。为解决这一问题，我们提出了真实锚定（Truth AnChoring, TAC），这是一种轻量级的后验校准方法，可将原始UE分数映射为与真实对齐的正确性概率。即使在有噪声的少样本监督条件下，TAC仍能持续改善校准效果，提供了一种实用的校准协议。我们的发现强调了将启发式UE指标直接视为（摘要在此处截断）

    arXiv:2604.00445v2 Announce Type: replace  Abstract: Uncertainty estimation (UE) aims to quantify uncertainty in large language model (LLM) outputs to flag potential hallucinations and support reliable deployment. However, UE metrics often exhibit unstable performance across models and datasets, limiting their practical utility. We formalise this phenomenon as proxy failure, arising because UE metrics are derived from model behaviour rather than being explicitly grounded in factual correctness. We show that when mutual information between a UE score and correctness is low, the score's discriminability must be close to chance. To address this, we propose Truth AnChoring (TAC), a lightweight post-hoc calibration method that maps raw UE scores to truth-aligned correctness probabilities. Even with noisy and few-shot supervision, TAC continues to improve calibration, providing a practical calibration protocol. Our findings highlight the limitations of treating heuristic UE metrics as direct
    
[^302]: 通过时空深度学习抑制大型液体闪烁体探测器中的$^{14}\mathrm{C}$光子击中

    Suppression of $^{14}\mathrm{C}$ photon hits in large liquid scintillator detectors via spatiotemporal deep learning

    [https://arxiv.org/abs/2603.27727](https://arxiv.org/abs/2603.27727)

    提出门控时空图神经网络和两种Transformer模型，在击中层面识别液体闪烁体探测器中$^{14}$C衰变的光子，实现25%-48%的召回率且正电子误判率低于1%，显著改善能量分辨率。

    

    液体闪烁体探测器因其低能量阈值和高能量分辨率而被广泛应用于中微子实验中。尽管$^{14}$C在液体闪烁体中含量极微，但$^{14}$C同位素β衰变产生的光子不可避免地污染信号，降低了能量分辨率。在本工作中，我们提出了三种模型，用于在含有$^{14}$C堆积的正电子事例中标记$^{14}$C光子击中，从而在击中层面抑制其对能量分辨率的影响：一个门控时空图神经网络，以及两个分别采用标量和矢量电荷编码的基于Transformer的模型。在一个每个事例包含一个$^{14}\mathrm{C}$和一个动能低于5 MeV的正电子的模拟测试数据集上，这些模型实现了25%-48%的$^{14}$C召回率，同时将正电子被误判为$^{14}$C的误识别率控制在1%以下，从而显著提高了总电荷的分辨率。

    arXiv:2603.27727v2 Announce Type: replace-cross  Abstract: Liquid scintillator detectors are widely used in neutrino experiments due to their low energy threshold and high energy resolution. Despite the tiny abundance of $^{14}$C in LS, the photons induced by the $\beta$ decay of the $^{14}$C isotope inevitably contaminate the signal, degrading the energy resolution. In this work, we propose three models to tag $^{14}$C photon hits in e$^+$ events with $^{14}$C pile-up, thereby suppressing its impact on the energy resolution at the hit level: a gated spatiotemporal graph neural network and two Transformer-based models with scalar and vector charge encoding. For a simulated test dataset in which each event contains one $^{14}\mathrm{C}$ and one e$^+$ with kinetic energy below 5 MeV, the models achieve $^{14}\mathrm{C}$ recall rates of 25%-48% while maintaining $\mathrm{e}^{+}$ to $^{14}\mathrm{C}$ misidentification below 1%, leading to a large improvement in the resolution of total char
    
[^303]: 基于稀疏注意力机制的准确高效多通道时间序列预测

    Accurate and Efficient Multi-Channel Time Series Forecasting via Sparse Attention Mechanism

    [https://arxiv.org/abs/2603.18712](https://arxiv.org/abs/2603.18712)

    本文提出了Li-Net新型架构，通过稀疏Top-K Softmax注意力机制与多尺度投影框架，高效捕获多通道时间序列中通道间的线性和非线性依赖关系，实现准确且高效的预测。

    

    多通道时间序列预测任务在金融、供应链管理和能源规划等众多领域中普遍存在。有效捕获通道内部和通道之间的复杂动态依赖关系对于准确预测至关重要。然而，传统方法很少关注通道之间交互的学习。本文提出了线性网络，这是一种为多通道时间序列预测设计的新型架构，能够捕获通道之间的线性和非线性依赖关系。Li-Net 动态压缩序列维度和通道维度上的表示，通过可配置的非线性模块处理信息，随后重构预测结果。此外，Li-Net 在多尺度投影框架内集成了稀疏 Top-K Softmax 注意力机制以应对这些挑战。其核心创新在于能够无缝地融合……

    arXiv:2603.18712v2 Announce Type: replace  Abstract: The task of multi-channel time series forecasting is ubiquitous in numerous fields such as finance, supply chain management, and energy planning. It is critical to effectively capture complex dynamic dependencies within and between channels for accurate predictions. However, traditional method paid few attentions on learning the interaction among channels. This paper proposes Linear-Network (Li-Net), a novel architecture designed for multi-channel time series forecasting that captures the linear and non-linear dependencies among channels. Li-Net dynamically compresses representations across sequence and channel dimensions, processes the information through a configurable non-linear module and subsequently reconstructs the forecasts. Moreover, Li-Net integrates a sparse Top-K Softmax attention mechanism within a multi-scale projection framework to address these challenges. A core innovation is its ability to seamlessly incorporate and
    
[^304]: 通过越狱相关表征偏移理解与防御视觉语言模型的越狱攻击

    Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift

    [https://arxiv.org/abs/2603.17372](https://arxiv.org/abs/2603.17372)

    该研究揭示视觉语言模型的越狱并非源于对有害意图识别的失败，而是视觉模态将表征推向特定越狱状态所致，并通过定义越狱相关表征偏移来量化这一现象，为理解和防御越狱攻击提供了新途径。

    

    大型视觉语言模型在引入视觉模态后往往表现出安全对齐的减弱。即使文本提示中包含明显的有害意图，添加一张图像也能大幅提高越狱成功率。在本文中，我们观察到视觉语言模型在其表征空间中能够清晰地区分良性输入与有害输入。此外，即使在有害输入之中，越狱样本也会形成一种独特的内部状态，可与拒绝样本分离开来。这些观察表明，越狱并非源于模型未能识别有害意图。相反，视觉模态会将表征推向一种特定的越狱状态，从而导致模型无法触发拒绝响应。为了量化这一转变，我们识别出了一个越狱方向，并将越狱相关偏移定义为图像引起的表征偏移在该方向上的分量。我们的分析表明……

    arXiv:2603.17372v2 Announce Type: replace-cross  Abstract: Large vision-language models (VLMs) often exhibit weakened safety alignment with the integration of the visual modality. Even when text prompts contain explicit harmful intent, adding an image can substantially increase jailbreak success rates. In this paper, we observe that VLMs can clearly distinguish benign inputs from harmful ones in their representation space. Moreover, even among harmful inputs, jailbreak samples form a distinct internal state that is separable from refusal samples. These observations suggest that jailbreaks do not arise from a failure to recognize harmful intent. Instead, the visual modality shifts representations toward a specific jailbreak state, thereby leading to a failure to trigger refusal. To quantify this transition, we identify a jailbreak direction and define the jailbreak-related shift as the component of the image-induced representation shift along this direction. Our analysis shows that the 
    
[^305]: OxyGen：多任务并行下VLA推理的统一KV缓存管理

    OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism

    [https://arxiv.org/abs/2603.14371](https://arxiv.org/abs/2603.14371)

    OxyGen将KV缓存作为跨任务和控制周期的一等共享资源进行统一管理，通过跨任务KV共享和跨帧连续批处理两项优化，消除了多任务并行VLA推理中的冗余计算和资源竞争问题。

    

    具身智能体日益需要基于共享观测并行执行多个任务，如操作、对话和记忆构建，且各任务具有不同的时间约束。近期的混合Transformer（MoT）视觉-语言-动作模型（VLA）支持此类异构输出，然而现有推理系统在设备端服务并发任务时会产生冗余计算和资源竞争。我们发现孤立的KV缓存管理是问题的根本原因。为解决这一问题，我们提出统一KV缓存管理这一推理设计，将KV缓存视为跨任务和控制周期的一等共享资源。这一抽象带来了两项关键优化：跨任务KV共享消除了对共享观测的冗余预填充计算，而跨帧连续批处理则将来自不同帧的未完成语言请求合并成批，并在每帧解码预算内推进其生成过程。

    arXiv:2603.14371v4 Announce Type: replace-cross  Abstract: Embodied AI agents increasingly require parallel execution of multiple tasks, such as manipulation, conversation, and memory construction, from shared observations under distinct time constraints. Recent Mixture-of-Transformers (MoT) Vision-Language-Action Models (VLAs) support such heterogeneous outputs, yet existing inference systems incur redundant computation and resource contention when serving concurrent tasks on-device. We identify isolated KV cache management as the root cause. To address this, we propose unified KV cache management, an inference design that treats the KV cache as a first-class shared resource across tasks and control cycles. This abstraction enables two key optimizations: cross-task KV sharing eliminates redundant prefill of shared observations, while cross-frame continuous batching batches unfinished language requests from different frames and advances them within a per-frame decoding budget. We imple
    
[^306]: M$^2$RNN：基于矩阵值状态的非线性循环神经网络用于可扩展的语言建模

    M$^2$RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling

    [https://arxiv.org/abs/2603.14360](https://arxiv.org/abs/2603.14360)

    提出了一种具有矩阵值隐藏状态和非线性状态转移的新型循环神经网络M$^2$RNN，突破了Transformer在TC$^0$复杂度上的表达能力限制，实现了完美状态跟踪泛化并在大规模语言建模中表现优异。

    

    Transformer具有高度并行性，但其计算能力局限于TC$^0$复杂度类，无法完成诸如实体跟踪和代码执行等已被证明需要更强表达能力的任务。受此限制的启发，我们重新审视非线性循环神经网络（RNN）在语言建模中的应用，并提出了矩阵到矩阵RNN（M$^2$RNN）：一种具有矩阵值隐藏状态和富有表达力的非线性状态转移的架构。我们证明了非线性RNN的语言建模性能受其状态规模的限制，并展示了状态规模扩展机制如何实现对张量核心的高效利用。实验表明，M$^2$RNN在训练中未见过的序列长度上实现了完美的状态跟踪泛化。这些优势同样体现在大规模语言建模中。在将循环层与注意力机制交错组合的混合设置中，混合M$^2$RNN优于等效的门控Delta（Gated Delta）……

    arXiv:2603.14360v3 Announce Type: replace-cross  Abstract: Transformers are highly parallel but are limited to computations in the TC$^0$ complexity class, excluding tasks such as entity tracking and code execution that provably require greater expressive power. Motivated by this limitation, we revisit non-linear Recurrent Neural Networks (RNNs) for language modeling and introduce Matrix-to-Matrix RNN (M$^2$RNN): an architecture with matrix-valued hidden states and expressive non-linear state transitions. We demonstrate that the language modeling performance of non-linear RNNs is limited by their state size, and show how the state size expansion mechanism enables efficient use of tensor cores. Empirically, M$^2$RNN achieves perfect state tracking generalization at sequence lengths not seen during training. These benefits also translate to large-scale language modeling. In hybrid settings that interleave recurrent layers with attention, Hybrid M$^2$RNN outperforms equivalent Gated Delta
    
[^307]: AOI：将失败轨迹转化为训练信号以实现自主云诊断

    AOI: Turning Failed Trajectories into Training Signals for Autonomous Cloud Diagnosis

    [https://arxiv.org/abs/2603.03378](https://arxiv.org/abs/2603.03378)

    该论文提出AOI可训练多智能体框架，通过GRPO将专家知识蒸馏到本地开源模型并结合读写分离的安全执行架构，把失败运维轨迹转化为训练信号，在安全约束下实现自主云诊断。

    

    大语言模型（LLM）智能体为自动化站点可靠性工程（SRE）提供了一种有前景的数据驱动方法，然而其企业级部署面临三大挑战：对专有数据的访问受限、在权限管控环境下执行操作的安全性问题，以及封闭系统无法从失败中学习改进。我们提出了AOI（自主运维智能，Autonomous Operations Intelligence），这是一个可训练的多智能体框架，将自动化运维形式化为安全约束下的结构化轨迹学习问题。我们的方法集成了三个关键组件：首先，一个可训练的诊断系统应用组相对策略优化（GRPO），将专家级知识蒸馏到本地部署的开源模型中，在不暴露敏感数据的前提下实现基于偏好的学习；其次，一个读写分离的执行架构将运维轨迹分解为观察、推理等阶段（摘要原文在此处截断）

    arXiv:2603.03378v4 Announce Type: replace-cross  Abstract: Large language model (LLM) agents offer a promising data-driven approach to automating Site Reliability Engineering (SRE), yet their enterprise deployment is constrained by three challenges: restricted access to proprietary data, unsafe action execution under permission-governed environments, and the inability of closed systems to improve from failures. We present AOI (Autonomous Operations Intelligence), a trainable multi-agent framework formulating automated operations as a structured trajectory learning problem under security constraints. Our approach integrates three key components. First, a trainable diagnostic system applies Group Relative Policy Optimization (GRPO) to distill expert-level knowledge into locally deployed open-source models, enabling preference-based learning without exposing sensitive data. Second, a read-write separated execution architecture decomposes operational trajectories into observation, reasonin
    
[^308]: 语言模型在自主学习任务中的目标选择与人类不同

    Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task

    [https://arxiv.org/abs/2603.03295](https://arxiv.org/abs/2603.03295)

    本研究借助认知科学中的自主学习任务，发现五大主流大语言模型的目标选择行为与人类存在显著差异——人类倾向于渐进探索且个体多样，而模型要么固守单一解要么表现不佳，表明LLM尚不能可靠地替代人类进行目标设定。

    

    无论是在智能体工作流、社会研究还是聊天场景中，大语言模型（LLM）越来越多地被要求代替人类选择要追求的目标，而不是完成预先定义的任务。然而，LLM能够准确反映人类目标设定偏好的这一假设在很大程度上仍未得到检验。我们在一项借鉴自认知科学的受控自主学习任务中，评估了LLM作为人类目标选择代理的有效性。在五个模型（GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5、Qwen3 32B和Centaur）上，我们发现模型行为与人类行为存在显著差异。人类会逐渐探索并学会实现目标，且个体间表现出多样性；而大多数模型要么固守单一已发现的解决方案，要么表现出出人意料的低性能，不同模型之间存在截然不同的模式，而同一模型的不同实例之间几乎没有变异性。思维链推理和角色引导……

    arXiv:2603.03295v3 Announce Type: replace-cross  Abstract: Whether in agentic workflows, social studies, or chat settings, large language models (LLMs) are increasingly being asked to replace humans in choosing which goals to pursue, rather than completing predefined tasks. However, the assumption that LLMs accurately reflect human preferences for goal setting remains largely untested. We assess the validity of LLMs as proxies for human goal selection in a controlled, self-directed learning task borrowed from cognitive science. Across five models (GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5, Qwen3 32B, and Centaur), we find substantial divergence from human behavior. While people gradually explore and learn to achieve goals with diversity across individuals, most models exploit a single identified solution or show surprisingly low performance, with distinct patterns across models and little variability across instances of the same model. Chain-of-thought reasoning and persona steering pro
    
[^309]: HydroShear：面向触觉仿真到真实强化学习的水弹性剪切仿真

    HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning

    [https://arxiv.org/abs/2603.00446](https://arxiv.org/abs/2603.00446)

    HydroShear是一个基于水弹性接触模型与符号距离函数的非完整触觉仿真器，通过建模粘滑转变、路径依赖的力与剪切累积以及完整SE(3)物体-传感器交互，显著缩小了接触丰富任务中触觉策略的仿真到真实差距。

    

    本文研究了面向接触丰富任务的触觉仿真到真实策略迁移问题。现有方法主要集中于基于视觉的传感器，强调图像渲染质量，而对力和剪切却提供了过于简化的模型。因此，这些模型在许多灵巧操作任务中存在较大的仿真到真实的差距。本文提出了HydroShear，一个非完整约束的水弹性触觉仿真器，它通过建模以下内容推进了该领域的最新技术水平：a）粘滑转变，b）路径依赖的力和剪切累积，以及c）完整的SE(3)物体-传感器交互。HydroShear利用带符号距离函数（SDF）扩展了水弹性接触模型，以在压头与传感器膜物理交互过程中跟踪压头表面点的位移。我们的方法能够从任意水密几何体生成基于物理的、计算高效的力场，同时保持对底层……（原文截断）

    arXiv:2603.00446v2 Announce Type: replace-cross  Abstract: In this paper, we address the problem of tactile sim-to-real policy transfer for contact-rich tasks. Existing methods primarily focus on vision-based sensors and emphasize image rendering quality while providing overly simplistic models of force and shear. Consequently, these models exhibit a large sim-to-real gap for many dexterous tasks. Here, we present HydroShear, a non-holonomic hydroelastic tactile simulator that advances the state-of-the-art by modeling: a) stick-slip transitions, b) path-dependent force and shear build up, and c) full SE(3) object-sensor interactions. HydroShear extends hydroelastic contact models using Signed Distance Functions (SDFs) to track the displacements of the on-surface points of an indenter during physical interaction with the sensor membrane. Our approach generates physics-based, computationally efficient force fields from arbitrary watertight geometries while remaining agnostic to the under
    
[^310]: 通过奖励引导拼接实现扩散语言模型的测试时扩展

    Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching

    [https://arxiv.org/abs/2602.22871](https://arxiv.org/abs/2602.22871)

    该论文提出Stitching Noisy Diffusion Thoughts框架，利用过程奖励模型对扩散语言模型采样的多条推理轨迹的中间步骤进行评分，并将跨轨迹的最优步骤拼接成复合推理链，实现了步骤级而非轨迹级的中间推理成果复用与测试时扩展。

    

    使用大型语言模型进行推理通常受益于生成多条思维链，但现有的聚合策略通常是轨迹层面的（例如，选择最佳轨迹或对最终答案进行投票），这会丢弃来自部分完成或“接近正确”尝试中的有用中间工作。我们提出了Stitching Noisy Diffusion Thoughts（拼接噪声扩散思维），这是一个自洽性框架，能将廉价的扩散采样推理转化为可复用的步骤级候选池。给定一个问题，我们（i）使用掩码扩散语言模型采样多条多样化、低成本的推理轨迹，（ii）使用现成的过程奖励模型（PRM）为每个中间步骤评分，（iii）将跨轨迹的最高质量步骤拼接成一个复合推理链，然后仅使用该推理链重新计算最终答案。这种模块化流程将探索（扩散）与评估和解决方案合成相分离，避免了……

    arXiv:2602.22871v2 Announce Type: replace-cross  Abstract: Reasoning with large language models often benefits from generating multiple chains-of-thought, but existing aggregation strategies are typically trajectory-level (e.g., selecting the best trace or voting on the final answer), discarding useful intermediate work from partial or "nearly correct" attempts. We propose Stitching Noisy Diffusion Thoughts, a self-consistency framework that turns cheap diffusion-sampled reasoning into a reusable pool of step-level candidates. Given a problem, we (i) sample many diverse, low-cost reasoning trajectories using a masked diffusion language model, (ii) score every intermediate step with an off-the-shelf process reward model (PRM), and (iii) stitch these highest-quality steps across trajectories into a composite rationale. This rationale is then used to recompute only the final answer. This modular pipeline separates exploration (diffusion) from evaluation and solution synthesis, avoiding mo
    
[^311]: OMEGA-Avatar：360度高斯化身的一次性建模

    OMEGA-Avatar: One-shot Modeling of 360{\deg} Gaussian Avatars

    [https://arxiv.org/abs/2602.11693](https://arxiv.org/abs/2602.11693)

    提出OMEGA-Avatar——首个前馈式框架，可从单张图像同时生成可泛化、360°完整且可动画的3D高斯头部化身，并通过融合多视图法线的语义感知网格变形模块解决了完整头部化身生成中头发建模不佳的问题。

    

    从单张图像创建高保真、可动画的3D化身仍然是一项艰巨的挑战。我们确定了化身生成的三个理想属性：1）方法应当是前馈式的，2）能够建模360°完整头部，3）应当具备可动画性。然而，现有工作只能同时满足其中两点。为解决这些局限，我们提出了OMEGA-Avatar，这是首个能够从单张图像同时生成可泛化、360°完整且可动画的3D高斯头部的前馈式框架。以前馈式可动画框架为基础，我们通过两个新颖组件来解决360°完整头部化身生成问题。首先，为克服完整头部化身生成中头发建模不佳的问题，我们引入了一个语义感知的网格变形模块，该模块融合多视图法线来优化带头发的FLAME头部，同时保持其拓扑结构。其次，为了……

    arXiv:2602.11693v2 Announce Type: replace-cross  Abstract: Creating high-fidelity, animatable 3D avatars from a single image remains a formidable challenge. We identified three desirable attributes of avatar generation: 1) the method should be feed-forward, 2) model a 360{\deg} full-head, and 3) should be animation-ready. However, current work addresses only two of the three points simultaneously. To address these limitations, we propose OMEGA-Avatar, the first feed-forward framework that simultaneously generates a generalizable, 360{\deg}-complete, and animatable 3D Gaussian head from a single image. Starting from a feed-forward and animatable framework, we address the 360{\deg} full-head avatar generation problem with two novel components. First, to overcome poor hair modeling in full-head avatar generation, we introduce a semantic-aware mesh deformation module that integrates multi-view normals to optimize a FLAME head with hair while preserving its topology structure. Second, to en
    
[^312]: 视觉语言模型中的道德谄媚行为

    Moral Sycophancy in Vision Language Models

    [https://arxiv.org/abs/2602.08311](https://arxiv.org/abs/2602.08311)

    该论文首次系统研究了视觉语言模型的道德谄媚现象，发现模型在用户提出异议时，即使初始道德判断正确也常给出错误回应，且表现出从道德正确转向道德错误的单向不对称倾向。

    

    视觉语言模型（VLM）中的谄媚行为是指模型迎合用户观点的倾向，而这往往以牺牲道德或事实准确性为代价。尽管先前的研究已在一般情境下探讨了谄媚行为，但其对基于道德的视觉决策的影响仍缺乏充分理解。为填补这一空白，我们开展了首个关于视觉语言模型道德谄媚行为的系统性研究，在明确的用户异议情境下，对十个广泛使用的模型在 Moralise 和 M^3oralBench 数据集上进行了分析。我们的结果表明，即使初始判断是道德正确的，视觉语言模型也经常在后续回应中给出道德上错误的回答，并表现出一种持续的不对称性：当受到用户引入的偏见影响时，模型更容易从道德正确的判断转向道德错误的判断，而非相反。后续提示通常会降低模型在 Moralise 上的表现，而在 M^3oralBench 上则产生好坏参半甚至有所提升的准确率，这凸显了……

    arXiv:2602.08311v2 Announce Type: replace  Abstract: Sycophancy in Vision-Language Models (VLMs) refers to their tendency to align with user opinions, often at the expense of moral or factual accuracy. While prior studies have explored sycophantic behavior in general contexts, its impact on morally grounded visual decision-making remains insufficiently understood. To address this gap, we present the first systematic study of moral sycophancy in VLMs, analyzing ten widely-used models on the Moralise and M^3oralBench datasets under explicit user disagreement. Our results reveal that VLMs frequently produce morally incorrect follow-up responses even when their initial judgments are correct, and exhibit a consistent asymmetry: models are more likely to shift from morally right to morally wrong judgments than the reverse when exposed to user-induced bias. Follow-up prompts generally degrade performance on Moralise, while yielding mixed or even improved accuracy on M^3oralBench, highlighting
    
[^313]: 通过互联网视频弱监督实现3D几何基础模型的可扩展适应

    Scalable Adaptation of 3D Geometric Foundation Models via Weak Supervision from Internet Video

    [https://arxiv.org/abs/2602.07891](https://arxiv.org/abs/2602.07891)

    提出SAGE框架，通过分层挖掘互联网视频并利用SfM稀疏几何锚定和3D高斯渲染可微一致性等混合弱监督信号，实现对3D几何基础模型的可扩展适应训练。

    

    几何基础模型在3D重建领域展现出巨大前景。然而，由于缺乏多样化、大规模的3D标注数据，其发展受到严重制约。尽管静态场景的互联网视频资源丰富且多样，但由于缺乏真值几何信息且存在观测噪声，将其作为几何学习的规模化数据源颇具挑战。为解决这一问题，我们提出了SAGE——一个从原始视频流对几何基础模型进行可扩展适应的框架。SAGE利用分层挖掘流水线将视频转化为训练轨迹和混合监督信号：（1）信息量丰富的训练轨迹选择；（2）通过SfM点云进行稀疏几何锚定，提供全局结构引导；（3）通过3D高斯渲染实现密集可微一致性，提供多视角约束。此外，我们还引入了一种正则化策略，通过重放少量……

    arXiv:2602.07891v2 Announce Type: replace-cross  Abstract: Geometric foundation models show promise in 3D reconstruction. However, their progress is severely constrained by the scarcity of diverse, large-scale 3D annotations. While static-scene Internet videos are abundant and diverse, utilizing them as a scaling source for geometric learning is challenging due to the absence of ground-truth geometry and the presence of observational noise. To address this, we propose SAGE, a framework for Scalable Adaptation of GEometric foundation models from raw video streams. SAGE leverages a hierarchical mining pipeline to transform videos into training trajectories and hybrid supervisions: (1) Informative training trajectory selection; (2) Sparse Geometric Anchoring via SfM point clouds for global structural guidance; and (3) Dense Differentiable Consistency via 3D Gaussian rendering for multi-view constraints. Besides, we also introduce a regularization strategy that replays a small fraction of 
    
[^314]: 测试驱动代码生成的理论分析

    A Theoretical Analysis of Test-Driven Code Generation

    [https://arxiv.org/abs/2602.06098](https://arxiv.org/abs/2602.06098)

    本文建立了测试驱动代码生成的概率理论框架，证明了基于模糊功能相似性的代码选择估计器在信噪比上严格优于基于功能等价性的估计器，并将回溯提示形式化为汤普森采样的上下文内近似，推导出新的遗憾界以解释其有效性受限于歧义性的原因。

    

    代码助手在测试驱动的软件开发中的应用日益广泛，但其环境交互策略背后的理论机制仍然缺乏深入研究。我们为两种主流范式提供了概率框架：利用执行环境在生成后进行代码选择，以及以环境反馈为条件的代码生成。首先，我们将若干成熟的选择启发式方法形式化为环境感知的代码正确性估计器。我们从理论上证明，基于模糊功能相似性的估计器引入了归纳偏置，并且在信噪比方面严格优于基于功能等价性的估计器。其次，我们将回溯提示（backprompting）框架化为汤普森采样的上下文内近似。我们为包含不可观测成分的奖励函数推导出了一个新的遗憾界，从理论上解释了为什么回溯提示的有效性受限于歧义性。

    arXiv:2602.06098v4 Announce Type: replace-cross  Abstract: Code assistants are increasingly utilized in test-driven software development, yet the theoretical mechanisms behind their environment-interaction strategies remain underexplored. We provide a probabilistic framework for two dominant paradigms: code selection after generation using the execution environment, and code generation conditioned on environment feedback. First, we formalize several well-established selection heuristics as environment-aware estimators of code correctness. We theoretically prove that estimators based on fuzzy functional similarity add an inductive bias and strictly dominate estimators based on functional equivalence in terms of signal-to-noise ratio. Second, we frame backprompting as an in-context approximation of Thompson sampling. We derive a novel regret bound for reward functions with unobservable components, theoretically explaining why the effectiveness of backprompting is limited by the ambiguity
    
[^315]: 脑电图中的被试间与被试内变异性：一项系统性综述

    Inter- and Intra-Subject Variability in EEG: A Systematic Survey

    [https://arxiv.org/abs/2602.01019](https://arxiv.org/abs/2602.01019)

    本综述系统梳理了健康与临床人群中脑电被试间与被试内变异性的量化研究，发现被试间差异通常大于被试内波动，且信号稳定性高度依赖于特征类型——alpha频段指标和P300成分相对可靠，而高频及连接性指标的可靠性较为异质。

    

    脑电图（EEG）是神经科学、临床神经生理学和脑机接口（BCI）的基石，但显著的被试间与被试内变异性限制了其可靠性、可重复性和临床转化。本系统性综述研究了在健康和临床人群中，对静息态、事件相关电位（ERP）以及任务相关/BCI范式（包括运动想象和SSVEP）中脑电变异性的量化与建模。在各范式下，被试间差异通常大于被试内波动，但两者都会影响统计推断和模型泛化能力。稳定性依赖于具体特征：alpha频段指标和个体alpha峰值频率通常相对可靠，而更高频段及许多基于连接性的指标则表现出较为异质的可靠性；ERP的可靠性因成分而异，其中P300相关指标常表现出中等到良好的稳定性。我们总结了主要的……

    arXiv:2602.01019v2 Announce Type: replace-cross  Abstract: Electroencephalography (EEG) underpins neuroscience, clinical neurophysiology, and brain-computer interfaces (BCIs), yet pronounced inter- and intra-subject variability limits reliability, reproducibility, and translation. This systematic review studies that quantified or modeled EEG variability across resting-state, event-related potentials (ERPs), and task-related/BCI paradigms (including motor imagery and SSVEP) in healthy and clinical cohorts. Across paradigms, inter-subject differences are typically larger than within-subject fluctuations, but both affect inference and model generalization. Stability is feature-dependent: alpha-band measures and individual alpha peak frequency are often relatively reliable, whereas higher-frequency and many connectivity-derived metrics show more heterogeneous reliability; ERP reliability varies by component, with P300 measures frequently showing moderate-to-good stability. We summarize maj
    
[^316]: RLHF如何放大谄媚行为

    How RLHF Amplifies Sycophancy

    [https://arxiv.org/abs/2602.01002](https://arxiv.org/abs/2602.01002)

    本文形式化分析了RLHF放大语言模型谄媚行为的机制，证明行为漂移方向由基础策略下“认可用户信念”与“学习到的奖励”之间的协方差决定，而该奖励差距源于人类偏好数据中的标注偏差。

    

    大语言模型在经过基于偏好的后训练后，往往会表现出更强的谄媚行为，即更倾向于肯定用户明示或暗示的信念，即使这与事实准确性或合理判断相冲突。我们对基于人类反馈的对齐如何加剧这一失效模式进行了形式化分析，识别出一种明确的放大机制，该机制将对学习到的奖励进行优化与用于对齐的人类偏好数据中的偏差因果地联系起来。我们证明，行为漂移的方向由基础策略下“认可提示中信念信号”与“学习到的奖励”之间的协方差决定，且其一阶效应可简化为一个简单的均值差条件。随后，我们分析了在Bradley-Terry等随机效用模型下从成对比较中学习奖励的过程，并刻画了人类标注者偏好中的偏差何时会引发这种奖励差距。接下来，

    arXiv:2602.01002v2 Announce Type: replace  Abstract: Large language models often exhibit increased sycophantic behavior after preference-based post-training, showing a stronger tendency to affirm a user's stated or implied belief even when this conflicts with factual accuracy or sound judgment. We present a formal analysis of how alignment from human feedback can increase this failure mode by identifying an explicit amplification mechanism that causally links optimization against a learned reward to bias in the human preference data used for alignment. We show that the direction of behavioral drift is determined by a covariance under the base policy between endorsing the belief signal in the prompt and the learned reward, and that the first-order effect reduces to a simple mean-gap condition. We then analyze reward learning from pairwise comparisons under random utility models like Bradley-Terry and characterize when bias in human annotators' preferences induces this reward gap. Next, 
    
[^317]: 快乐的年轻女性，暴躁的老年男性？合成人脸生成中情绪驱动的人口统计偏差

    Happy Young Women, Grumpy Old Men? Emotion-Driven Demographic Biases in Synthetic Face Generation

    [https://arxiv.org/abs/2602.00032](https://arxiv.org/abs/2602.00032)

    本研究系统审计了8个西方和中国的文生图模型在情绪条件提示下生成的56,000张合成人脸，揭示了情绪提示会驱动人口统计偏差（如年轻女性更易被生成为快乐形象、老年男性更易被生成为暴躁形象），并通过信息论指标量化了性别、种族、年龄和感知吸引力方面的交叉性偏差及东西方模型生态系统的差异。

    

    来自文本到图像（T2I）模型的合成人脸遍布数字媒体，然而它们在情绪条件提示下所呈现的人口统计偏差仍然鲜为人知。我们旨在系统地审计情绪条件提示如何影响T2I模型生成的合成人脸中的人口统计和感知吸引力偏差，特别关注交叉性模式以及不同模型家族之间的跨生态系统差异。我们审计了八个T2I模型（4个西方模型和4个中国模型），并在七种提示条件下生成了56,000张人脸：包括一个中性基线和六种情绪条件。我们使用信息论散度度量量化了性别、种族、年龄和感知吸引力方面的偏差。我们进一步对组合的人口统计属性进行了交叉性分析，并比较了西方和中国模型组之间的偏差模式，以评估偏差行为在跨生态系统上的一致性与分歧。

    arXiv:2602.00032v4 Announce Type: replace-cross  Abstract: Synthetic faces from text-to-image (T2I) models pervade digital media, yet their demographic biases under emotionally conditioned prompts remain poorly understood. We aim to systematically audit how emotionally conditioned prompts affect demographic and perceived-attractiveness biases in synthetic faces generated by T2I models, with particular attention to intersectional patterns and cross-ecosystem differences across model families. We audited eight (4 Western and 4 Chinese) T2I models and generated 56,000 faces under seven prompt conditions: a neutral baseline and six emotion conditions. We quantified biases in gender, race, age, and perceived attractiveness using information-theoretic divergence metrics. We further conducted intersectional analyses across combined demographic attributes and compared patterns between the Western and Chinese model groups to assess cross-ecosystem consistency and divergence in bias behavior. Al
    
[^318]: Veri-Sure：基于时序追踪、切片与形式化验证的多智能体RTL代码生成

    Veri-Sure: Multi-Agent RTL Code Generation with Temporal Tracing, Slicing and Formal Verification

    [https://arxiv.org/abs/2601.19747](https://arxiv.org/abs/2601.19747)

    Veri-Sure是一个多智能体RTL代码生成框架，通过设计契约对齐智能体意图、静态依赖切片引导的局部化补丁修复以及形式化验证与时序追踪，解决了LLM生成RTL代码中反馈稀疏、整文件重生成引发回归和语义漂移三大瓶颈。

    

    在快速发展的电子设计自动化（EDA）领域中，部署大语言模型（LLM）进行寄存器传输级（RTL）代码生成已成为一个有前景的方向。然而，由于RTL代码描述的是具有严格周期级时序的并发硬件行为，与软件代码有着本质区别，基于LLM的RTL代码生成仍受制于以下瓶颈：（i）以仿真为中心的评估方法反馈稀疏且测试覆盖有限；（ii）调试过程中的整文件重新生成引入了回归问题和修复幻觉；（iii）随着意图在修复迭代和智能体交接过程中被反复重新解释而产生语义漂移。在本工作中，我们提出了Veri-Sure，这是一个多智能体框架，它通过建立设计契约来对齐各智能体的意图，并采用由静态依赖切片引导的补丁机制来执行精确的局部化修复。通过集成多分支验证……

    arXiv:2601.19747v2 Announce Type: replace-cross  Abstract: In the rapidly evolving field of Electronic Design Automation (EDA), the deployment of Large Language Models (LLMs) for Register-Transfer Level (RTL) code generation has emerged as a promising direction. However, because RTL code describes concurrent hardware behavior with strict cycle-level timing and thus differs fundamentally from software code, LLM-based RTL code generation remains bottlenecked by (i) sparse feedback and limited test coverage of simulation-centric evaluation, (ii) regressions and repair hallucinations introduced by whole-file regeneration during debugging, and (iii) semantic drift as intent is reinterpreted across repair iterations and agent handoffs. In this work, we propose Veri-Sure, a multi-agent framework that establishes a design contract to align agents' intent and uses a patching mechanism guided by static dependency slicing to perform precise, localized repairs. By integrating a multi-branch verifi
    
[^319]: 大型语言模型中的“亲AI”偏向

    Pro-AI Bias in Large Language Models

    [https://arxiv.org/abs/2601.13749](https://arxiv.org/abs/2601.13749)

    该研究发现大型语言模型存在“亲AI”偏向：它们在建议中更频繁且更靠前地推荐AI，会将AI相关职位的薪资高估约10%，并对AI专业证书授予数量的高估程度比匹配的计算机专业高出28%。

    

    大型语言模型（LLMs）在多个领域的决策支持中正得到越来越广泛的应用。我们探究的问题是：这些模型是否偏爱人工智能（AI）本身。我们通过三个实验来检验这一问题。首先，在投资、学习、职业和创业等领域的开放式建议中，AI在大多数推荐列表中都占据显著位置，并且一旦出现通常排名第一。其次，在薪资估算方面，模型对AI相关职位薪资的高估程度高于与之高度匹配的非AI职位。在一项受控的后续实验中，在所有其他职位信息保持不变的情况下，仅将职位名称从匹配的非AI名称改为AI相关名称，就会使预测薪资提高10%。最后，我们证明，AI专业所授予的学位及其他证书数量被高估的程度比与之匹配的计算机专业相应证书高出28%。总体而言，我们的研究表明，AI不仅在日常生活中……

    arXiv:2601.13749v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly employed for decision-support across multiple domains. We ask whether these models favor artificial intelligence (AI) itself. We examine this question through three experiments. First, in open-ended advice across investment, study, career, and startup domains, AI is prominent in most recommendation lists and is usually ranked first when it appears. Second, in salary estimations, models overestimate AI-related jobs more than they do for closely matched non-AI jobs. In a controlled follow-up, changing only the job title from the matched non-AI title to the AI-related title raises predicted salary by 10%, while all other supplied job details remain fixed. Finally, we show that the number of degrees and other credentials awarded by AI programs were 28% more overestimated than corresponding awards for matched computing programs. Overall, we show that AI is not only prominent in everyday 
    
[^320]: 受AI影响的岗位在ChatGPT问世之前就已恶化

    AI-exposed jobs deteriorated before ChatGPT

    [https://arxiv.org/abs/2601.02554](https://arxiv.org/abs/2601.02554)

    该研究利用美国失业保险记录、LinkedIn资料和大学课程大纲数据，发现受AI影响职业的就业恶化早在ChatGPT发布之前就已出现，表明生成式AI可能并非劳动力市场困境的根源，且LLM相关教育仍具有显著价值。

    

    公众讨论普遍将受AI影响职业的就业前景恶化归因于2022年底ChatGPT的发布。我们利用美国月度失业保险记录，测量了特定职业和特定地区的失业风险，发现受AI影响职业的失业风险早在2022年初就开始上升，比ChatGPT发布早了数月。通过分析数百万份LinkedIn个人资料，我们发现自2021届起的毕业生进入受AI影响岗位的比例低于更早的届别，且这一差距在2022年底之前就已出现。最后，基于数百万份大学课程大纲，我们发现修读更多大语言模型（LLM）相关课程的毕业生，在ChatGPT发布后求职时间更短，且在进入高度暴露职业时获得的首份工作薪资更高。综合来看，这些结果——指向早于生成式AI存在的力量以及LLM相关教育的持续价值——表明生成式AI可能并非这些劳动力市场困境的根本原因。

    arXiv:2601.02554v2 Announce Type: replace-cross  Abstract: Public debate links worsening job prospects for AI-exposed occupations to the release of ChatGPT in late 2022. Using monthly U.S. unemployment insurance records, we measure occupation- and location-specific unemployment risk and find that risk rose in AI-exposed occupations beginning in early 2022, months before ChatGPT was released. Analyzing millions of LinkedIn profiles, we show that graduate cohorts from 2021 onward entered AI-exposed jobs at lower rates than earlier cohorts, with gaps opening before late 2022. Finally, from millions of university syllabi, we find that graduates taking more LLM-exposed curricula had shorter job searches after ChatGPT and higher first-job pay when entering highly exposed occupations. Together, these results---pointing to forces pre-dating generative AI and to the ongoing value of LLM-relevant education---suggest that generative AI may not be the root cause of these labor market hardships and
    
[^321]: 薛定谔导航器：为零样本物体导航想象一组未来场景

    Schr\"odinger's Navigator: Imagining an Ensemble of Futures for Zero-Shot Object Navigation

    [https://arxiv.org/abs/2512.21201](https://arxiv.org/abs/2512.21201)

    该论文提出“薛定谔导航器”框架，利用轨迹条件的三维世界模型在推理时想象并维持多个可能未来场景的叠加态，结合遮挡感知采样和未来感知价值图（FAVM），显著提升机器人在复杂真实环境中零样本物体导航的鲁棒性、主动性与安全性。

    

    零样本物体导航（ZSON）要求机器人在未见过的环境中寻找目标物体，而无需任务特定的微调或预先构建的地图，这是通用服务机器人的一项关键能力。然而，在仿真中表现良好的方法，在充满杂物、严重遮挡和潜在危险的真实场景中往往性能下降——大片未见区域使得基于单一场景的推理变得脆弱且不安全。我们提出薛定谔导航器，这是一个信念感知框架，能够在推理时对多条轨迹条件下的多个想象三维未来进行推理。给定候选路径，轨迹条件的三维世界模型预测假设性观测，并维持多种合理场景实现的“叠加态”，而不是固守单一地图。自适应的遮挡感知采样器将想象引导至不确定性关键的区域，而未来感知价值图（FAVM）则聚合想象出的未来，以实现稳健、主动的行动选择。

    arXiv:2512.21201v5 Announce Type: replace-cross  Abstract: Zero-shot object navigation (ZSON) requires robots to find target objects in unseen environments without task-specific fine-tuning or pre-built maps, a key capability for general-purpose service robots. Yet methods that perform well in simulation often degrade in cluttered real-world scenes with severe occlusion and latent hazards, where large unseen regions make single-scene inference brittle and unsafe. We propose Schr\"odinger's Navigator, a belief-aware framework that reasons at inference time over multiple trajectory-conditioned imagined 3D futures. Given candidate paths, a trajectory-conditioned 3D world model predicts hypothetical observations and maintains a superposition of plausible scene realizations rather than committing to one map. An adaptive occluder-aware sampler directs imagination to uncertainty-critical regions, while a Future-Aware Value Map (FAVM) aggregates imagined futures for robust, proactive action se
    
[^322]: AQUA-Net：面向水下图像增强的自适应频率融合与光照感知网络

    AQUA-Net: Adaptive Frequency Fusion and Illumination Aware Network for Underwater Image Enhancement

    [https://arxiv.org/abs/2512.05960](https://arxiv.org/abs/2512.05960)

    提出AQUA-Net，通过融合傅里叶域频率信息与基于Retinex的光照感知自适应曝光校正，在提升水下图像颜色、对比度和细节质量的同时保持较低计算复杂度，适用于实时水下应用。

    

    水下图像由于波长相关的光吸收和散射，常常出现严重的颜色失真、低对比度和雾化外观。同时，现有深度学习模型计算复杂度高，限制了其在实时水下应用中的实际部署。为解决这些挑战，本文提出了一种新颖的水下图像增强模型——自适应频率融合与光照感知网络（AQUA-Net）。该模型将残差编码器-解码器与运行在频率域和光照域的双辅助分支相集成：频率融合编码器利用来自傅里叶域的频率线索丰富空间表征，并保留精细的纹理和结构细节；受Retinex理论启发，光照感知解码器通过学习到的光照图将反射率与光照分离，从而实现自适应的曝光校正。

    arXiv:2512.05960v2 Announce Type: replace-cross  Abstract: Underwater images often suffer from severe color distortion, low contrast, and a hazy appearance due to wavelength-dependent light absorption and scattering. Simultaneously, existing deep learning models exhibit high computational complexity, which limits their practical deployment for real-time underwater applications. To address these challenges, this paper presents a novel underwater image enhancement model, called Adaptive Frequency Fusion and Illumination Aware Network (AQUA-Net). It integrates a residual encoder-decoder with dual auxiliary branches, which operate in the frequency and illumination domains. The frequency fusion encoder enriches spatial representations with frequency cues from the Fourier domain and preserves fine textures and structural details. Inspired by Retinex, the illumination-aware decoder performs adaptive exposure correction through a learned illumination map that separates reflectance from lightin
    
[^323]: Operon：通过命名维度增量构建不规则数据

    Operon: Incremental Construction of Ragged Data via Named Dimensions

    [https://arxiv.org/abs/2511.16080](https://arxiv.org/abs/2511.16080)

    Operon是一个基于Rust的工作流引擎，通过命名维度与显式依赖关系的形式化方法，结合静态验证的领域特定语言和增量式的动态任务调度，实现了对不规则数据形状与依赖关系的原生追踪与管理。

    

    现代数据处理工作流经常遇到不规则数据：即包含可变长度元素的集合，这类数据自然产生于自然语言处理、科学测量和自主AI智能体等领域。现有的工作流引擎缺乏对不规则数据固有的形状和依赖关系的原生支持，迫使用户手动管理复杂的索引和依赖记录。我们提出了Operon，一个基于Rust的工作流引擎，它通过一种新颖的命名维度形式化方法及显式依赖关系来解决这些挑战。Operon提供了一种领域特定语言，用户可以在其中声明带有维度标注的管道，这些标注会经过静态验证以确保正确性；同时运行时系统在执行过程中随着数据形状的逐步发现而动态调度任务。我们形式化了用于推理部分形状的数学基础，并证明了Operon的……（原文摘要在此处截断）

    arXiv:2511.16080v2 Announce Type: replace-cross  Abstract: Modern data processing workflows frequently encounter ragged data: collections with variable-length elements that arise naturally in domains like natural language processing, scientific measurements, and autonomous AI agents. Existing workflow engines lack native support for tracking the shapes and dependencies inherent to ragged data, forcing users to manage complex indexing and dependency bookkeeping manually. We present Operon, a Rust-based workflow engine that addresses these challenges through a novel formalism of named dimensions with explicit dependency relations. Operon provides a domain-specific language where users declare pipelines with dimension annotations that are statically verified for correctness, while the runtime system dynamically schedules tasks as data shapes are incrementally discovered during execution. We formalize the mathematical foundation for reasoning about partial shapes and prove that Operon's in
    
[^324]: Rep2Text：从单个LLM词元表示中解码完整文本

    Rep2Text: Decoding Full Text from a Single LLM Token Representation

    [https://arxiv.org/abs/2511.06571](https://arxiv.org/abs/2511.06571)

    提出Rep2Text方法，证明仅凭LLM的单个隐藏表示即可重建原始文本（短文本可恢复约一半单词及整体含义），揭示了隐藏表示蕴含的丰富信息及其潜在的隐私泄露风险。

    

    大型语言模型将文本表示为隐藏的数值模式，但这些表示实际包含什么信息仍然是一个谜。我们能否仅凭其中一个表示恢复出原始文本？在本文中，我们提出了Rep2Text，这是一种旨在通过从单个隐藏表示重建文本来回答这一问题的方法。我们发现这些表示保留的信息远比我们预期的要多。对于约16个单词的短文本，Rep2Text能够恢复大约一半的原始单词，同时保留其整体含义。随着文本变长，恢复确切的措辞变得更加困难，但主要思想仍然可以被恢复。这些发现为理解语言模型内部存储了什么提供了新的途径，并凸显了潜在的隐私风险，因为隐藏表示可能会泄露原始输入的部分内容。

    arXiv:2511.06571v4 Announce Type: replace-cross  Abstract: Large language models represent text as hidden numerical patterns, but what information these representations actually contain remains a mystery. Can we recover the original text from just one of them? In this paper, we introduce Rep2Text, a method designed to answer this question by reconstructing text from a single hidden representation. We find that these representations preserve much more information than we might expect. For short texts of about 16 words, Rep2Text can recover roughly half of the original words while preserving their overall meaning. As the text gets longer, recovering the exact wording becomes harder, but the main ideas can still be recovered. These findings offer a new way to understand what language models store internally and highlight potential privacy risks, as hidden representations may reveal parts of the original input.
    
[^325]: 迈向基于记忆增强规划与预见性的统一视觉导航世界模型

    Towards Unified World Models for Visual Navigation via Memory-Augmented Planning and Foresight

    [https://arxiv.org/abs/2510.08713](https://arxiv.org/abs/2510.08713)

    本文提出UniWM，一种统一的记忆增强世界模型，通过将视觉预见与规划集成到单一自回归框架中，并采用分层记忆机制，有效解决了视觉导航中状态-动作不对齐和长程推理稳定性问题，在多个基准上显著提升了导航性能。

    

    arXiv:2510.08713v3 公告类型：替换 摘要：使具身智能体能够想象未来状态对于稳健且可泛化的视觉导航至关重要。然而，最先进的系统通常依赖模块化设计，将导航规划与视觉世界模型分离，这常常导致状态-动作不对齐，并在新颖或动态场景中适应性较弱。我们提出UniWM，一种统一的、记忆增强的世界模型，它在单一多模态自回归骨干网络中集成了自我中心视觉预见和规划。UniWM明确地将动作选择基于视觉想象的输出，紧密对齐预测与控制。同时，分层记忆机制融合了短期感知线索与长期轨迹上下文，支持在扩展时间范围内的稳定和连贯推理。在四个具有挑战性的基准测试（Go Stanford、ReCon、SCAND、HuRoN）和1X人形数据集上的大量实验表明，UniWM改善了导航性能。

    arXiv:2510.08713v3 Announce Type: replace  Abstract: Enabling embodied agents to imagine future states is essential for robust and generalizable visual navigation. Yet, state-of-the-art systems typically rely on modular designs that decouple navigation planning from visual world modeling, which often induces state-action misalignment and weak adaptability in novel or dynamic scenarios. We propose UniWM, a unified, memory-augmented world model that integrates egocentric visual foresight and planning within a single multimodal autoregressive backbone. UniWM explicitly grounds action selection in visually imagined outcomes, tightly aligning prediction with control. Meanwhile, a hierarchical memory mechanism fuses short-term perceptual cues with longer-term trajectory context, supporting stable and coherent reasoning over extended horizons. Extensive experiments on four challenging benchmarks (Go Stanford, ReCon, SCAND, HuRoN) and the 1X Humanoid Dataset show that UniWM improves navigation
    
[^326]: 面向细粒度幻觉检测的跨层注意力探测方法

    Cross-Layer Attention Probing for Fine-Grained Hallucination Detection

    [https://arxiv.org/abs/2509.09700](https://arxiv.org/abs/2509.09700)

    本文提出跨层注意力探测技术CLAP，将大语言模型整个残差流的激活作为联合序列处理，实现了细粒度的幻觉检测，并基于此提出“先检测后缓解”策略以减少幻觉、提升模型可靠性。

    

    随着大语言模型（LLMs）在各类应用中的大规模采用，由于其倾向于生成不准确的文本（即幻觉），人们对其可靠性的担忧日益增加。在这项工作中，我们提出了跨层注意力探测，这是一种用于幻觉检测的新型激活探测技术，它将LLM在整个残差流中的激活作为一个联合序列进行处理。我们使用五个大语言模型和三个任务进行的实证评估表明，CLAP在贪婪解码的响应以及较高温度下采样的响应上，相比基线方法均提升了幻觉检测性能，从而实现了细粒度检测，即能够对给定提示词的多个不同采样响应中的幻觉与非幻觉进行区分。这使我们能够提出一种基于CLAP的“先检测后缓解”策略，以减少幻觉并提升大语言模型的可靠性，其效果优于直接缓解方法。

    arXiv:2509.09700v2 Announce Type: replace-cross  Abstract: With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e. hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitiga
    
[^327]: 计算多智能体社会实验：基于生成式智能体的社会建模框架

    Computational Multi-Agents Society Experiments: Social Modeling Framework Based on Generative Agents

    [https://arxiv.org/abs/2508.17366](https://arxiv.org/abs/2508.17366)

    本文提出CMASE框架，首次允许研究人员以场景内角色身份实时参与生成式智能体社会模拟并进行可追溯的干预，将模拟视为模拟民族志田野，从而支持实时人机交互研究、民族志式受控实验以及干预轨迹与下游响应的前瞻性探索。

    

    计算社会科学已经发展出描述性、解释性和预测性建模，但将场景内人类干预与结果预测相结合的综合建模仍未得到充分探索。基于生成式智能体的平台可以扩展人口规模或重现复杂场景，但研究人员通常通过初始设置、外部控制或事后分析来开展工作，而非以场景内角色的身份参与其中。我们提出了计算多智能体社会实验（CMASE），这是一个基于生成式智能体的框架，支持研究人员的场景内参与以及运行时可追溯的干预。CMASE将模拟视为一个模拟的民族志田野，支持：（1）实时人机交互以研究干预过程；（2）借鉴民族志的观察、访谈和解释方法开展受控实验；（3）对干预轨迹和下游响应的前瞻性探索。CMASE结合了大语言模型驱动的感知……

    arXiv:2508.17366v3 Announce Type: replace  Abstract: Computational social science has developed descriptive, interpretative, and predictive modeling, but comprehensive modeling combining in-world human intervention with outcome forecasting remains underexplored. Generative agent-based platforms scale populations or reproduce complex scenarios, but researchers typically act through setup, external controls, or post-hoc analysis rather than in-world roles. We introduce Computational Multi-Agent Society Experiments (CMASE), a generative agent-based framework for in-world researcher participation and traceable interventions at runtime. Treating simulation as a simulated ethnographic field, CMASE supports (1) real-time human-computer interaction to study intervention processes; (2) controlled experiments with ethnography-inspired observation, interviewing, and interpretation; and (3) forward exploration of intervention trajectories and downstream responses. CMASE combines LLM-driven percept
    
[^328]: 基于对抗性随机森林的缺失值插补方法——MissARF

    Missing value imputation with adversarial random forests - MissARF

    [https://arxiv.org/abs/2507.15681](https://arxiv.org/abs/2507.15681)

    MissARF是一种基于对抗性随机森林的快速缺失值插补方法，可在提供高质量单重与多重插补的同时，使多重插补无需额外计算成本。

    

    处理缺失值是生物统计分析中的一个常见挑战，通常通过插补方法来解决。我们提出了一种新颖、快速且易于使用的插补方法，称为基于对抗性随机森林的缺失值插补（MissARF），该方法基于生成式机器学习，可同时提供单重插补和多重插补。MissARF利用对抗性随机森林（ARF）进行密度估计和数据合成。为了插补某个观测值的缺失值，我们以该观测的非缺失值为条件，从ARF生成的估计条件分布中进行采样。我们的实验表明，MissARF在插补质量和运行速度方面与最先进的单重和多重插补方法表现相当，且多重插补无需额外的计算成本。

    arXiv:2507.15681v2 Announce Type: replace-cross  Abstract: Handling missing values is a common challenge in biostatistical analyses, typically addressed by imputation methods. We propose a novel, fast, and easy-to-use imputation method called missing value imputation with adversarial random forests (MissARF), based on generative machine learning, that provides both single and multiple imputation. MissARF employs adversarial random forest (ARF) for density estimation and data synthesis. To impute a missing value of an observation, we condition on the non-missing values and sample from the estimated conditional distribution generated by ARF. Our experiments demonstrate that MissARF performs comparably to state-of-the-art single and multiple imputation methods in terms of imputation quality and fast runtime with no additional costs for multiple imputation.
    
[^329]: 可解释强化学习综述：解释对象、方法与需求

    A Survey of Explainable Reinforcement Learning: Targets, Methods and Needs

    [https://arxiv.org/abs/2507.12599](https://arxiv.org/abs/2507.12599)

    本文提出了基于“解释什么”和“如何解释”两个维度的可解释强化学习（XRL）直观分类法，并对250多篇论文进行了系统性综述，同时指出了值得关注的相关领域与社区需求。

    

    近年来人工智能（AI）模型的成功伴随着其内部机制的不透明性，这主要归因于深度神经网络的使用。为了理解这些内部机制并解释这些AI模型的输出，研究者们提出了一系列方法，统称为可解释人工智能（XAI）领域。本文聚焦于XAI的一个子领域——可解释强化学习（XRL），其目标是对通过强化学习训练的智能体的行为进行解释。我们提出了一个基于“是什么”和“如何”两个问题的直观分类法：第一个问题关注方法所解释的对象，第二个问题涉及解释的提供方式。我们运用这一分类法对250多篇论文进行了最新进展的综述。此外，我们还介绍了一些与XRL密切相关的领域，我们认为这些领域应受到研究界的关注。最后，我们识别了……（原文摘要此处截断）

    arXiv:2507.12599v2 Announce Type: replace  Abstract: The success of recent Artificial Intelligence (AI) models has been accompanied by the opacity of their internal mechanisms, due notably to the use of deep neural networks. In order to understand these internal mechanisms and explain the output of these AI models, a set of methods have been proposed, grouped under the domain of eXplainable AI (XAI). This paper focuses on a sub-domain of XAI, called eXplainable Reinforcement Learning (XRL), which aims to explain the actions of an agent that has learned by reinforcement learning. We propose an intuitive taxonomy based on two questions "What" and "How". The first question focuses on the target that the method explains, while the second relates to the way the explanation is provided. We use this taxonomy to provide a state-of-the-art review of over 250 papers. In addition, we present a set of domains close to XRL, which we believe should get attention from the community. Finally, we ident
    
[^330]: 反向GPT：精确逆转语言模型的输出

    GPT, But Backwards: Exactly Inverting Language Model Outputs

    [https://arxiv.org/abs/2507.01693](https://arxiv.org/abs/2507.01693)

    该论文提出SODA算法，能够从语言模型的输出logits中精确反推出输入文本，实现多达17个token的零误报重建，且逆转难度主要取决于输入长度和词汇表大小而非参数量，并可应用于恢复后门触发器和个人身份信息等安全场景。

    

    最近的研究表明，不同的输入会在大语言模型（LLM）中产生不同的隐藏表示，因此原则上可以从这些表示中恢复出输入。然而，目前尚无任何方法能够仅凭模型的输出文本或logits可靠地实现这一点。我们提出了稀疏独热离散Adam（SODA），这是一种基于搜索的白盒算法，能够从输出的部分知识中重建文本输入。给定输出logits，SODA能够以零误报率恢复多达17个token的自然序列，并且与此前最强的方法相比，精确匹配重建性能提升了4倍。我们进一步证明，对于逆转难度而言，输入长度和词汇表大小比模型参数量更为关键，SODA成功逆转了从3300万到30亿参数规模的模型。最后，我们展示了概念验证性的安全应用：SODA在不到1000次迭代中从被投毒的大语言模型中恢复了6个后门触发器，并恢复了139个token的个人身份信息。

    arXiv:2507.01693v3 Announce Type: replace-cross  Abstract: Recent work shows that distinct inputs produce distinct hidden representations in LLMs, so the input can in principle be recovered from them. Yet, no existing method can reliably do so from the model's output text or logits alone. We introduce Sparse One-hot Discrete Adam (SODA), a white-box search-based algorithm that reconstructs textual inputs from partial knowledge of the output. Given output logits, SODA recovers natural sequences of up to 17 tokens with zero false positives and achieves a 4x improvement in exact-match reconstruction over the strongest prior method. We further show that input length and vocabulary size matter more for inversion difficulty than parameter count, with SODA inverting models from 33M to 3B parameters. Finally, we demonstrate proof-of-concept security applications: SODA recovers 6 backdoor triggers from a poisoned LLM in under 1000 iterations, and 139 tokens of personally identifiable informatio
    
[^331]: 基于约束知识遗忘的安全对齐

    Safety Alignment via Constrained Knowledge Unlearning

    [https://arxiv.org/abs/2505.18588](https://arxiv.org/abs/2505.18588)

    提出约束知识遗忘（CKU）安全对齐策略，通过定位并保留与有用知识相关的神经元、裁剪其梯度来精准遗忘有害知识，从而在不损害模型整体性能的情况下有效防御越狱攻击。

    

    尽管安全对齐已取得显著进展，大语言模型（LLM）仍然容易受到越狱攻击。现有的防御机制并未完全删除LLM中的有害知识，这使得此类攻击能够绕过安全防护并产生有害输出。为应对这一挑战，我们提出了一种新颖的安全对齐策略——约束知识遗忘（CKU），该策略聚焦于两个主要目标：知识的定位与保留，以及遗忘有害知识。CKU通过对特定多层感知机（MLP）层中的神经元进行评分，识别出与有用知识相关的神经元子集U。在遗忘过程中，CKU对U中神经元的梯度进行裁剪，以在保留有价值知识的同时有效消除有害内容。实验结果表明，CKU在不损害整体性能的前提下显著提升了模型安全性，实现了更优的平衡。

    arXiv:2505.18588v2 Announce Type: replace-cross  Abstract: Despite significant progress in safety alignment, large language models (LLMs) remain susceptible to jailbreak attacks. Existing defense mechanisms have not fully deleted harmful knowledge in LLMs, which allows such attacks to bypass safeguards and produce harmful outputs. To address this challenge, we propose a novel safety alignment strategy, Constrained Knowledge Unlearning (CKU), which focuses on two primary objectives: knowledge localization and retention, and unlearning harmful knowledge. CKU works by scoring neurons in specific multilayer perceptron (MLP) layers to identify a subset U of neurons associated with useful knowledge. During the unlearning process, CKU prunes the gradients of neurons in U to preserve valuable knowledge while effectively mitigating harmful content. Experimental results demonstrate that CKU significantly enhances model safety without compromising overall performance, offering a superior balance 
    
[^332]: 两心智胜过一心：面向大语言模型对齐的协作式奖励建模

    Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment

    [https://arxiv.org/abs/2505.10597](https://arxiv.org/abs/2505.10597)

    提出协作式奖励建模（CRM）框架，通过两个奖励模型跨模型互评结合课程学习来识别可靠的偏好数据，从而有效缓解噪声偏好导致的奖励误泛化并提升大语言模型的对齐效果。

    

    奖励模型（RMs）对于将大语言模型（LLMs）与人类价值观对齐至关重要，然而噪声偏好反馈可能通过虚假关联和过拟合引发奖励的误泛化。我们刻画了偏好对的训练动态，并揭示了噪声的独特优化特征：持续的高损失、剧烈的时间波动以及不规则的梯度更新。我们的分析进一步表明，排除此类样本对可以改善下游对齐效果。然而在实践中，可靠的偏好无法先验得知，而基于模型自身训练信号的自过滤可能会加剧确认偏误。为应对这一挑战，我们提出了协作式奖励建模，这是一种用于识别可靠偏好的在线框架。CRM将跨模型互评与课程学习相结合：两个奖励模型相互评估对方的样本选择，同时课程学习保持它们的学习进度。

    arXiv:2505.10597v3 Announce Type: replace-cross  Abstract: Reward models (RMs) are central to aligning large language models (LLMs) with human values, yet noisy preference feedback can induce reward misgeneralization through spurious correlations and overfitting. We characterize the training dynamics of preference pairs and uncover a distinctive optimization signature of noise: persistently high losses, sharp temporal fluctuations, and irregular gradient updates. Our analyses further show that excluding such pairs improves downstream alignment. In practice, however, reliable preferences are not known a priori, and self-filtering based on a model's own training signals can amplify confirmation bias. To address this challenge, we introduce Collaborative Reward Modeling (CRM), an online framework for identifying reliable preferences. CRM couples cross-model peer review with curriculum learning: two RMs evaluate each other's sample selections, while curriculum keeps their learning progress
    
[^333]: GVPO：面向大语言模型后训练的组方差策略优化

    GVPO: Group Variance Policy Optimization for Large Language Model Post-Training

    [https://arxiv.org/abs/2504.19599](https://arxiv.org/abs/2504.19599)

    本文提出组方差策略优化（GVPO），通过将KL约束奖励最大化的解析解直接融入梯度权重，解决了GRPO等后训练方法训练不稳定的问题，并保证唯一的最优解。

    

    后训练在改进和对齐大语言模型以满足特定任务和人类偏好方面发挥着至关重要的作用。尽管最近的后训练技术（如组相对策略优化GRPO）通过增加采样数量并结合相对奖励评分取得了卓越的性能，但这些方法常常存在训练不稳定的问题，限制了其实际应用。作为进一步的改进，我们提出了组方差策略优化（GVPO）。GVPO将KL约束奖励最大化的解析解直接融入到其梯度权重中，确保与最优策略保持一致。该方法具有直观的物理解释：其梯度反映了隐式奖励的中心距离与实际奖励的中心距离之间的均方误差。GVPO具有两个关键优势：（1）它保证了唯一的最优解，恰好是KL约束奖励最大化问题的解（原文此处截断）。

    arXiv:2504.19599v4 Announce Type: replace  Abstract: Post-training plays a crucial role in refining and aligning large language models to meet specific tasks and human preferences. While recent advancements in post-training techniques, such as Group Relative Policy Optimization (GRPO), leverage increased sampling with relative reward scoring to achieve superior performance, these methods often suffer from training instability that limits their practical adoption. As a next step, we present Group Variance Policy Optimization (GVPO). GVPO incorporates the analytical solution to KL-constrained reward maximization directly into its gradient weights, ensuring alignment with the optimal policy. The method provides intuitive physical interpretations: its gradient mirrors the mean squared error between the central distance of implicit rewards and that of actual rewards. GVPO offers two key advantages: (1) it guarantees a unique optimal solution, exactly the KL-constrained reward maximization o
    
[^334]: 一种用于可解释钓鱼检测的梯度优化TSK模糊框架

    A Gradient-Optimized TSK Fuzzy Framework for Explainable Phishing Detection

    [https://arxiv.org/abs/2504.18636](https://arxiv.org/abs/2504.18636)

    该论文提出一种基于梯度优化的TSK模糊推理模型的可解释钓鱼URL检测框架，将模糊逻辑的可解释性与Adam梯度优化的精确性相结合，在超过23.5万条URL的数据集上同时实现了高检测准确率与良好的可解释性。

    

    钓鱼攻击对个人和组织构成了日益复杂和普遍的威胁，造成重大经济损失、身份盗窃以及机构声誉的严重损害。现有的钓鱼检测方法往往难以同时实现高准确率和可解释性，要么无法检测新型攻击，要么作为不透明的黑盒模型运行。为了解决这一关键问题，我们提出了一种新颖的钓鱼URL检测系统，该系统基于通过梯度技术优化的一阶Takagi-Sugeno-Kang（TSK）模糊推理模型。我们的方法巧妙地将模糊逻辑的可解释性和类人推理能力与梯度优化方法提供的精确性和适应性相结合，特别是利用Adam优化器进行高效的参数调整。在包含超过235,000个URL的综合数据集上进行的实验表……

    arXiv:2504.18636v2 Announce Type: replace-cross  Abstract: Phishing attacks represent an increasingly sophisticated and pervasive threat to individuals and organizations, causing significant financial losses, identity theft, and severe damage to institutional reputations. Existing phishing detection methods often struggle to simultaneously achieve high accuracy and explainability, either failing to detect novel attacks or operating as opaque black-box models. To address this critical gap, we propose a novel phishing URL detection system based on a first-order Takagi-Sugeno-Kang (TSK) fuzzy inference model optimized through gradient-based techniques. Our approach intelligently combines the interpretability and human-like reasoning capabilities of fuzzy logic with the precision and adaptability provided by gradient optimization methods, specifically leveraging the Adam optimizer for efficient parameter tuning. Experiments conducted using a comprehensive dataset of over 235,000 URLs demon
    
[^335]: 与AI智能体协作：关于团队合作、生产力与绩效的实地实验

    Collaborating with AI Agents: Field Experiments on Teamwork, Productivity, and Performance

    [https://arxiv.org/abs/2503.18238](https://arxiv.org/abs/2503.18238)

    该研究通过2,234名参与者的大规模随机实验发现，人机团队人均广告产出比人人团队多50%且文本质量更高，但人人团队图像质量更优、且人机协作导致产出多样性坍缩，揭示了AI智能体能力的锯齿状边界与人机协作中的多样性损失两大机制。

    

    我们通过在Pairit平台（一个我们开发的用于研究人机协作的平台）上开展的大规模实验，检验了AI智能体带来生产力与绩效提升背后的机制。我们将2,234名参与者随机分配到人人团队和人机团队中，共制作了11,024个广告。我们采用独立人工评分以及在X平台上获得约500万次曝光的实地实验来评估这些广告。研究发现，人机团队人均广告产出量高出50%，且文本质量更高；而人人团队制作的图像质量更高，这表明AI智能体的能力边界呈“锯齿状”（参差不齐）。此外，人机团队产出的广告同质性更强（多样性坍缩），其广告自我相似度更高。实地实验显示，更高的文本质量（由人机团队产出）提升了点击率和浏览时长，而更高的图像质量（由人人团队产出）改善了单次点击成本。

    arXiv:2503.18238v4 Announce Type: replace-cross  Abstract: We examined the mechanisms underlying productivity and performance gains from AI agents using a large-scale experiment on Pairit, a platform we developed to study human-AI collaboration. We randomly assigned 2,234 participants to human-human and human-AI teams that produced 11,024 ads. We evaluated the ads using independent human ratings and a field experiment on X which garnered ~5M impressions. We found human-AI teams produced 50% more ads per worker and higher text quality, while human-human teams produced higher image quality, suggesting a jagged frontier of AI agent capability. Human-AI teams also produced more homogeneous outputs (diversity collapse) as their ads were more self-similar. The field experiment revealed higher text quality (produced by human-AI teams) increased click-through rates and view time, while higher image quality (produced by human-human teams) improved cost-per-click rates. Two mechanisms explained 
    
[^336]: LeanTTA：一种面向边缘设备上量化测试时自适应的无反向传播、无状态方法

    LeanTTA: A Backpropagation-Free and Stateless Approach to Quantized Test-Time Adaptation on Edge Devices

    [https://arxiv.org/abs/2503.15889](https://arxiv.org/abs/2503.15889)

    本文提出LeanTTA框架，通过无反向传播、无状态地动态更新归一化统计量，实现了面向边缘设备的量化测试时自适应，显著降低内存与计算开销且不依赖大批量或历史数据。

    

    尽管在边缘设备上部署机器学习模型具有诸多优势，但移动平台的资源限制、环境的动态特性，以及训练数据与真实场景数据分布之间的差异，使得此类部署充满挑战。现有的测试时自适应方法通常内存开销大，且并未针对量化兼容性或低资源设备部署而设计。为应对这些挑战，我们提出了LeanTTA，一个专为边缘设备量身打造的、无反向传播且无状态的新型量化测试时自适应框架。我们的方法通过在不使用反向传播的情况下动态更新归一化统计量来最小化计算成本，使LeanTTA摆脱了依赖大批量数据和历史数据的常见缺陷，从而在现实部署场景中具有更强的鲁棒性。我们的方法是首个通过组合……来实现进一步计算收益的方法。

    arXiv:2503.15889v2 Announce Type: replace-cross  Abstract: While there are many advantages to deploying machine learning models on edge devices, the resource constraints of mobile platforms, the dynamic nature of the environment, and differences between the distribution of training versus in-the-wild data make such deployments challenging. Current test-time adaptation methods are often memory-intensive and not designed to be quantization-compatible or deployed on low-resource devices. To address these challenges, we present LeanTTA, a novel backpropagation-free and stateless framework for quantized test-time adaptation tailored to edge devices. Our approach minimizes computational costs by dynamically updating normalization statistics without backpropagation, which frees LeanTTA from the common pitfall of relying on large batches and historical data, making our method robust to realistic deployment scenarios. Our approach is the first to enable further computational gains by combining 
    
[^337]: 因果模型与因果表示中的“因果”究竟指什么？

    What is causal about causal models and representations?

    [https://arxiv.org/abs/2501.19335](https://arxiv.org/abs/2501.19335)

    该论文提出了一个将因果模型与数据生成过程分离的形式化框架，形式化了连接行动与干预的解释，并证明了不存在能同时满足一组期望性质且保持非循环性的行动-干预解释这一不可能性结果。

    

    因果贝叶斯网络通常被描述为“因果的”，因为它们支持关于干预的推理。然而，干预是在数学模型中定义的，而模型的预测涉及的是现实世界中行动的效果。因此，因果模型是否以及如何对行动的效果做出可证伪的预测，取决于行动如何被解释为干预。这需要明确哪些行动对应哪些干预，而这一步骤往往被隐式处理。我们开发了一个形式化框架，将因果模型与其旨在描述的数据生成过程分离开来，并将连接行动与干预的解释形式化。在该框架内，我们证明了一个不可能性结果：没有任何解释能够在保持非循环性的同时满足一组期望的性质。在循环解释下，每一个拟合观测数据的模型……

    arXiv:2501.19335v3 Announce Type: replace-cross  Abstract: Causal Bayesian networks are often described as causal because they support reasoning about interventions. Yet interventions are defined in the mathematical model, whereas the resulting model predictions concern the effects of actions in the real world. Consequently, whether and how a causal model makes falsifiable predictions about the effects of actions depends on how actions are interpreted as interventions. This requires specifying which actions correspond to which interventions, a step that is frequently left implicit. We develop a formal framework that separates a causal model from the data-generating process it aims to describe and formalizes interpretations connecting actions to interventions. Within this framework, we prove an impossibility result: no interpretation can simultaneously satisfy a collection of desirable properties while remaining non-circular. Under a circular interpretation, every model that fits the ob
    
[^338]: 重新思考用于概念抑制的负向提示：无条件锚定与局部化引导

    Rethinking Negative Prompting for Concept Suppression: Unconditional Anchoring and Localized Guidance

    [https://arxiv.org/abs/2412.07658](https://arxiv.org/abs/2412.07658)

    提出无需训练的TraSCE方法，通过无条件锚定与局部化引导，解决负向提示在越狱攻击下因“引导崩塌”而失效的问题，从而有效抑制文生图模型生成有害或不良概念。

    

    文生图扩散模型已迅速从研究原型转变为被广泛部署的工具。在具备强大能力的同时，这些模型也可能生成有害或不良内容（例如NSFW图像），这促使人们需要实用的机制来移除此类概念。负向提示利用负向提示的预测作为引导锚点，是将采样轨迹引导远离不需要概念的最常用机制之一。然而，面对越狱提示，它可能在我们称之为“引导崩塌”的情形下失效：模型对正向提示和负向提示的去噪预测变得相似，使得驱动负向提示的引导差异消失，引导后的预测退化为负向提示的预测，从而削弱了概念抑制效果。为解决这一问题，我们提出了TraSCE（Trajectory Steering for ConcEpt Suppression，用于概念抑制的轨迹引导），这是一种无需训练的方法……

    arXiv:2412.07658v3 Announce Type: replace-cross  Abstract: Text-to-image diffusion models have rapidly transitioned from research prototypes to widely deployed tools. Along with their impressive capabilities, these models can also generate harmful or undesirable content (e.g., NSFW imagery), motivating practical mechanisms for removing such concepts. Negative prompting, which uses the negative-prompt prediction as the guidance anchor, is among the most widely used mechanisms for steering sampling trajectories away from unwanted concepts. However, for jailbreak prompts, it can fail in a regime we refer to as \emph{guidance collapse}, where the model's denoising predictions for the prompt and negative prompt become similar, so that the guidance difference driving negative prompting vanishes and the guided prediction degenerates to the negative-prompt prediction, undermining suppression. To fix this issue, we propose TraSCE (Trajectory Steering for ConcEpt Suppression), a training-free in
    
[^339]: 框外作画：视觉定位训练数据的合成与选择

    Paint Outside the Box: Synthesizing and Selecting Training Data for Visual Grounding

    [https://arxiv.org/abs/2412.00684](https://arxiv.org/abs/2412.00684)

    提出 POBF 框架，通过在边界框外进行图像修复来合成训练数据以解决标签错位问题，并利用结合难度分数与过拟合分数的过滤机制筛选最有效的训练数据，从而在数据稀缺条件下显著提升视觉定位性能。

    

    视觉定位旨在根据文本查询定位图像中的相应区域。鉴于大规模数据整理的困难，本文研究了如何在数据稀缺的设置下有效学习视觉定位。为解决数据稀缺问题，我们提出了一种新颖的框架 POBF（Paint Outside the Box and Filter，框外作画与过滤）。POBF 通过在边界框之外进行图像修复来合成图像，解决了以往工作中遇到的标签错位问题。此外，POBF 利用一种创新的过滤方案来选择最有效的训练数据。该方案结合了难度分数和过拟合分数，并通过一个惩罚项加以平衡。在四个基准数据集上的大量实验表明，POBF 持续提升性能，相比仅使用真实数据的方法平均提升 5.83%，在准确率上超越领先的基线方法 2.29%-3.85%。

    arXiv:2412.00684v3 Announce Type: replace-cross  Abstract: Visual grounding aims to localize the image regions based on a textual query. Given the difficulty of large-scale data curation, we investigate how to effectively learn visual grounding under data-scarce settings in this paper. To address the data scarcity, we propose a novel framework, POBF (Paint Outside the Box and Filter). POBF synthesizes images by inpainting outside the box, tackling a label misalignment issue encountered in previous works. Furthermore, POBF leverages an innovative filtering scheme to select the most effective training data. This scheme combines a hardness score and an overfitting score, balanced by a penalty term. Extensive experiments across four benchmark datasets demonstrate that POBF consistently improves performance, achieving an average gain of 5.83\% over the real-data-only method and outperforming leading baselines by 2.29\%-3.85\% in accuracy. Additionally, we validate the robustness and general
    
[^340]: 有害YouTube视频检测：网络危害分类体系与多模态大语言模型作为替代标注者

    Harmful YouTube Video Detection: A Taxonomy of Online Harm and MLLMs as Alternative Annotators

    [https://arxiv.org/abs/2411.05854](https://arxiv.org/abs/2411.05854)

    该研究提出了一个涵盖信息、仇恨与骚扰、成瘾性、标题党、性和人身伤害六大类别的网络视频危害综合分类体系，并系统验证了以GPT-4-Turbo为代表的多模态大语言模型可以作为人工标注的有效替代方案来检测有害YouTube视频。

    

    全球数十亿人使用的视频平台因让用户接触有害内容而受到批评。然而，由于对“何为危害”的界定方法不一致以及人工标注的局限性，检测网络危害仍然具有挑战性。我们针对这两个挑战提出了解决方案。在理论层面，我们整合并扩展现有的分类体系和平台准则，将其拓展至多模态场景，构建了一个全面的网络危害分类体系。我们将视频平台上的网络危害归纳为六种基于内容的类别：信息危害、仇恨与骚扰、成瘾性危害、标题党、性危害和人身伤害。在方法层面，我们提出并系统地测试了多模态大语言模型（MLLMs）作为有害视频的替代标注者。我们整理了一个包含19,422个潜在有害YouTube视频的数据集，并使用GPT-4-Turbo、未经训练的众包工作者和经过训练的标注者对其进行标注。我们讨论了多模态大语言模型在平台内容审核中的潜在应用。

    arXiv:2411.05854v2 Announce Type: replace-cross  Abstract: Video platforms, used by billions globally, are criticized for exposing users to harmful content. Yet, detecting online harm remains challenging due to inconsistent approaches to what constitutes harm and limitations of human annotation. We address both challenges. Theoretically, we integrate and expand existing taxonomies and platform guidelines to multimodal contexts and develop a comprehensive taxonomy for online harm. We categorize online harm on video platforms into six content-based categories: Information, Hate and harassment, Addictive, Clickbait, Sexual, and Physical harms. Methodologically, we propose and systematically test multimodal large language models (MLLMs) as alternative annotators of harmful videos. We curate a dataset of 19,422 potentially harmful YouTube videos and annotate them using GPT-4-Turbo, non-trained crowdworkers, and trained annotators. We discuss the potential applications of MLLMs in platform c
    
[^341]: VietMed：面向医疗领域越南语自动语音识别的数据集与基准

    VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain

    [https://arxiv.org/abs/2404.05659](https://arxiv.org/abs/2404.05659)

    VietMed 是迄今全球最大的公开医疗领域越南语语音识别数据集，并首次提供了覆盖所有 ICD-10 疾病组和全国所有口音的医疗 ASR 数据集，以及首个越南语 ASR 公开大规模预训练模型。

    

    由于隐私限制，医疗领域公开可用的语音识别数据集十分匮乏。在本工作中，我们提出了 VietMed——一个医疗领域的越南语语音识别数据集，包含16小时带标注的医疗语音、1000小时无标注医疗语音以及1200小时无标注通用领域语音。据我们所知，VietMed 在七个方面是迄今全球最大的公开医疗语音识别数据集：总时长、说话人数量、疾病种类、录音条件、说话人角色、独特医学术语数量和口音种类。VietMed 也是迄今总时长最大的公开越南语语音数据集。此外，我们是首个提出覆盖一个国家内所有 ICD-10 疾病组和所有口音的医疗 ASR 数据集的团队。同时，我们发布了首个面向越南语 ASR 的公开大规模预训练模型 w2v2-Viet 和 XLSR-53-Viet，以及首个……

    arXiv:2404.05659v4 Announce Type: replace-cross  Abstract: Due to privacy restrictions, there's a shortage of publicly available speech recognition datasets in the medical domain. In this work, we present VietMed - a Vietnamese speech recognition dataset in the medical domain comprising 16h of labeled medical speech, 1000h of unlabeled medical speech and 1200h of unlabeled general-domain speech. To our best knowledge, VietMed is by far the world's largest public medical speech recognition dataset in 7 aspects: total duration, number of speakers, diseases, recording conditions, speaker roles, unique medical terms and accents. VietMed is also by far the largest public Vietnamese speech dataset in terms of total duration. Additionally, we are the first to present a medical ASR dataset covering all ICD-10 disease groups and all accents within a country. Moreover, we release the first public large-scale pre-trained models for Vietnamese ASR, w2v2-Viet and XLSR-53-Viet, along with the first 
    
[^342]: 超越边界：AI系统可迁移攻击的综合综述

    Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems

    [https://arxiv.org/abs/2311.11796](https://arxiv.org/abs/2311.11796)

    本综述提出了一个涵盖跨实例、跨领域、跨模态、跨模型、跨任务和跨部署六个维度的统一分类体系，为比较AI系统中各类可迁移攻击的迁移机制与失效条件提供了通用框架。

    

    可迁移攻击使得在一个环境中构建的对抗性操纵或学习到的攻击机制能够危害其他AI系统，从而可能降低攻击成本并加剧安全与隐私风险。然而，碎片化的术语和评估实践阻碍了人们对“什么可以迁移、为什么可以迁移、迁移在何处失效”的理解。本综述综合了代表性研究，构建了一个涵盖六个互补维度的统一分类体系：跨实例、跨领域、跨模态、跨模型、跨任务和跨部署。通过将源-目标配置的变化与攻击者的访问权限和自适应预算相分离，该分类体系为比较逃逸、投毒、后门、隐私推断、模型窃取以及基于提示的攻击中的迁移现象提供了统一的框架。我们关联了梯度对齐、共享表示、训练持久性以及部署（摘要在此处截断）等迁移机制……

    arXiv:2311.11796v3 Announce Type: replace-cross  Abstract: Transferable attacks allow adversarial manipulations or learned attack mechanisms constructed in one setting to compromise other AI systems, potentially reducing attack costs and amplifying security and privacy risks. Yet fragmented terminology and evaluation practices hinder understanding of what transfers, why it transfers, and where transfer fails. This survey synthesizes representative research and develops a unified taxonomy spanning six complementary axes: cross-instance, cross-domain, cross-modality, cross-model, cross-task, and cross-deployment. By separating source-target configuration changes from attacker access and adaptation budgets, the taxonomy provides a common framework for comparing transfer across evasion, poisoning, backdoors, privacy inference, model stealing, and prompt-based attacks. We connect transfer mechanisms, including gradient alignment, shared representations, training persistence, and deployment 
    
[^343]: 利用有向图中的信任：重新定义图学习的鲁棒性

    Harnessing Trust in Directed Graphs: Redefining Robustness of Graph Learning

    [https://arxiv.org/abs/2306.02002](https://arxiv.org/abs/2306.02002)

    该论文从攻击和防御两个角度重新定义了有向图上图神经网络的鲁棒性，并提出了一种即插即用的消息传递框架，通过利用有向图链接方向中蕴含的信任信号显著提升GNN的鲁棒性。

    

    现有关于鲁棒图神经网络（GNN）的研究主要集中于无向图，忽视了有向图链接方向中固有的可信性。在本工作中，我们从攻击和防御两个角度分析了当前方法的局限性，并重新定义了有向图中GNN的鲁棒性。在攻击方面，我们引入了一种更贴近现实的有向图攻击设置，弥补了现有攻击方法的不足。在防御方面，我们提出了一个简单而有效的消息传递框架，可作为即插即用的插件层来增强GNN的鲁棒性，同时避免产生虚假的鲁棒性错觉。我们的研究结果表明，利用有向图中编码的信任信号可以显著提高GNN的鲁棒性。与现有防御策略结合时，该框架在实现最先进的鲁棒性能的同时，还保持了较高的干净数据准确率。

    arXiv:2306.02002v2 Announce Type: replace-cross  Abstract: Existing research on robust Graph Neural Networks (GNNs) focuses predominantly on undirected graphs, overlooking the trustworthiness inherent in the link directions of directed graphs. In this work, we analyze the limitations of current approaches from both the attack and defense perspectives and redefine the robustness of GNNs in directed graphs. From the attack side, we introduce a more realistic directed graph attack setting that addresses the shortcomings of existing attacks. From the defense side, we propose a simple yet effective message-passing framework that serves as a plug-in layer to strengthen GNN robustness while avoiding a false sense of robustness. Our findings show that the trust signals encoded in directed graphs can be harnessed to substantially improve the robustness of GNNs. When combined with existing defense strategies, the framework achieves state-of-the-art robust performance alongside strong clean accur
    
[^344]: Pluvio:通过迁移学习和条件变分信息瓶颈实现对领域外结构和库的汇编克隆搜索

    Pluvio: Assembly Clone Search for Out-of-domain Architectures and Libraries through Transfer Learning and Conditional Variational Information Bottleneck. (arXiv:2307.10631v1 [cs.SE])

    [http://arxiv.org/abs/2307.10631](http://arxiv.org/abs/2307.10631)

    本论文介绍了一种通过迁移学习和条件变分信息瓶颈实现对领域外结构和库的汇编克隆搜索的方法。已有的方法往往限制于少数训练过的工具链变体，而本研究首次解决了未见过架构和库的情况，提出了一种新的解决方案。

    

    代码重用在软件开发中至关重要，可以加快和提高开发生命周期的效率。然而，实际中，代码重用存在着控制不当的问题，导致漏洞传播和知识产权侵权等问题。汇编克隆搜索是一种重要的右移防御机制，已经在识别由已发布的可执行文件中的代码重用导致的漏洞方面取得了良好效果。最近关于汇编克隆搜索的研究表明，越来越多采用基于机器学习的方法来匹配由不同工具链生成的汇编代码变体。然而，这些方法仅限于从训练中学习到的少数工具链变体，无法应用于未见过的架构及其相应的编译工具链变体。本文首次研究了在未见过的架构和库中的汇编克隆搜索问题。我们提议通过人类共有知识来解决这个问题。

    The practice of code reuse is crucial in software development for a faster and more efficient development lifecycle. In reality, however, code reuse practices lack proper control, resulting in issues such as vulnerability propagation and intellectual property infringements. Assembly clone search, a critical shift-right defence mechanism, has been effective in identifying vulnerable code resulting from reuse in released executables. Recent studies on assembly clone search demonstrate a trend towards using machine learning-based methods to match assembly code variants produced by different toolchains. However, these methods are limited to what they learn from a small number of toolchain variants used in training, rendering them inapplicable to unseen architectures and their corresponding compilation toolchain variants.  This paper presents the first study on the problem of assembly clone search with unseen architectures and libraries. We propose incorporating human common knowledge throu
    

