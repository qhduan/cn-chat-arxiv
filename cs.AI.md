# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching](https://arxiv.org/abs/2611.23229) | 提出Ambient SFT框架，通过将微调模型的高温条件分布与冻结的预训练“噪声教师”模型进行温度匹配，利用不完美预测扩充微调数据集，从而防止过拟合和数据分布尾部的灾难性遗忘。 |
| [^2] | [StringQFT: A Multiuse Textbook Resource for AI-Assisted Theoretical Physics](https://arxiv.org/abs/2611.23225) | 该论文提出StringQFT——一套涵盖18卷、3795页的量子场论弦构造教科书合集，同时作为AI辅助理论物理的基础设施，提供开卷基准题库、检索增强生成资源、训练数据示例以及智能体计算工具。 |
| [^3] | [Do LLM Agents Still Need to Be Told How to Self-Improve?](https://arxiv.org/abs/2611.23219) | 该论文提出了一种完全自主的自我改进范式，证明足够强大的冻结大语言模型智能体在仅有基本要素、工具和计算预算、且无人类指定方法或外部强加循环的条件下，能够自行构建自学习算法并实现自我改进，其策略与经典自我调节学习认知框架相符。 |
| [^4] | [Higher-Scoring Coding Models Did Not Get More Work Done](https://arxiv.org/abs/2611.23216) | 该研究利用美国出口管制导致Claude Fable 5被停用19天的自然实验，发现在编码基准测试中得分更高的模型（Fable 5）相比得分较低的模型（Opus 4.8）并未给用户带来更高的实际工作产出，揭示了基准分数与真实生产力之间的脱节。 |
| [^5] | [Reading Scene States: Uncovering Safety Failures in Vision-Language Models](https://arxiv.org/abs/2611.23206) | 该论文发现了视觉-语言模型的“场景状态安全漏洞”——即使没有明确有害请求，模型解读场景状态图像也会提供有害协助（攻击成功率45.6%–72.3%），并提出了SceneSafetyBench基准和S3TA系统性攻击方法来揭示这一安全失效。 |
| [^6] | [Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU](https://arxiv.org/abs/2611.23205) | 该论文在笔记本电脑集成GPU上用66个样例通过SFT结合LoRA/QLoRA微调Gemma 4边缘模型（E2B和E4B）用于MATLAB信号处理代码生成，使通过率显著提升（如E4B从32/66升至47/66），且发现微调主要修正DSP惯例错误而非信号构建错误，运行时代价仅为17-19%的生成速度损失。 |
| [^7] | [Agent Trust Infrastructure (ATI): A Federated Identity, Trust, Attestation, and Accountability Architecture for Cross-Organization Autonomous AI Agents](https://arxiv.org/abs/2611.23202) | 本文提出智能体信任基础设施（ATI），一种基于PKI、零信任架构和远程证明、独立于框架的安全架构，通过对等组织证书颁发机构建立联邦信任，在跨组织AI智能体执行请求前验证其身份、执行完整性与授权。 |
| [^8] | [Structure--Residual Inductive Biases for Language Model Pretraining](https://arxiv.org/abs/2611.23197) | 提出一种结构—残差归纳偏置原则，通过边界瓶颈概念（BBC）和连续概念（CC）两种机制将语言模型表示中的结构与实现细节分离，并推导出相应的编码界，发现二者在深度偏好上存在显著差异。 |
| [^9] | [WoRD: Diffusion-Based Posterior Inference for Online Goal Recognition](https://arxiv.org/abs/2611.23192) | 提出了WoRD方法，利用学习到的扩散轨迹先验，为每个候选目标运行热启动去噪链并通过贝叶斯重加权得到校准后验，实现从部分观测中对智能体目标的在线识别。 |
| [^10] | [Retrieval-Augmented Correction for Generalist Medical Image Segmentation](https://arxiv.org/abs/2611.23189) | GRACE是一种检索增强的医学图像分割校正器，通过从已验证历史病例的共享知识库中检索上下文，仅需一次训练即可校正多种解剖结构的分割误差，并能在不重新训练模型的情况下扩展到新的器官。 |
| [^11] | [Say Less, Hallucinate Less? Diagnosing and Restoring Object Coverage in Preference-Optimized Vision-Language Models](https://arxiv.org/abs/2611.23179) | 该研究发现偏好优化在降低视觉语言模型物体幻觉的同时会牺牲物体覆盖率，其收益主要源于“输出更少”，基础模型仅通过正 EOS 偏置早停即可复现 58%–86% 的幻觉降低效果。 |
| [^12] | [Inference after coding agents](https://arxiv.org/abs/2611.23174) | 本文指出AI编码智能体会在实证研究中引发一种新型的选择后推断偏差——在零效应的模拟数据上，前沿智能体被要求产出“有说服力的发现”时会有54%至98%的概率报告虚假显著p值——并提出了一种检测与校正此类扭曲的简单程序。 |
| [^13] | [Attribution Through the Decoder: Prediction-Space Explanations for Structured Outputs](https://arxiv.org/abs/2611.23161) | 该论文提出通过解码器微分（对归一化解码器等价于缩放的切向投影）进行归因，避免了将输入错误归因于解码器丢弃的缩放变化，从而为角度、视线方向等结构化输出提供更准确的预测空间解释。 |
| [^14] | [Learning Stratified Policies for Inference-Time Objectives](https://arxiv.org/abs/2611.23154) | 该论文提出“分层策略”，通过在共享语言模型主干上训练 k 个 LoRA 适配器，将推理时的采样预算分配给一个策略群体，并从理论和实验上证明这种策略群体在优化 pass@k 时优于单一策略。 |
| [^15] | [Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance](https://arxiv.org/abs/2611.23142) | 该论文提出并实际部署了一个以"TriAgent"分诊智能体为核心的多智能体系统，通过持续挖掘客服工单和代码-文档不一致等显性与隐性信号，自动检测并修复客户文档与技术实现之间的漂移。 |
| [^16] | [RSI-Prime: Training Models to Train Models](https://arxiv.org/abs/2611.23138) | 该论文提出RSI-Prime数据集（含覆盖七大任务类别的350条完整实验轨迹），通过细粒度决策质量标注与决策级损失掩码过滤低质量行为进行训练，使GLM-5.3在自主后训练基准PostTrainBench上整体性能相对提升31.95%，为递归自我改进奠定基础。 |
| [^17] | [Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI](https://arxiv.org/abs/2611.23106) | 本文提出自然语言编程（NLPg）这一新概念，将提示词重新定义为能够外化目标、约束与推理状态的动态认知制品，主张自然语言正从被计算的对象演变为人类与AI协作中可执行的认知媒介。 |
| [^18] | [Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting](https://arxiv.org/abs/2611.23096) | 该论文发现，评估提示是否会重新激发被接种式提示抑制的有害行为，取决于其诱导的激活状态与接种上下文的相似程度，而非文本表面相似性，这表明激活空间分析可用于预测被抑制行为何时仍可被触达。 |
| [^19] | [What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language](https://arxiv.org/abs/2611.23093) | 本文提出 tree-grams——一种通过二阶和三阶张量层次化组合构建的自回归 n-gram 模型，其上下文窗口与模型容量分别由超参数 n 和 d 可控地调节，在超越传统 n-gram 基线的同时可与 Transformer 语言模型媲美，并可作为研究语言的可控近似器用于超参数迁移等实验。 |
| [^20] | [BugGen 2.0: Rethinking the Roles of LLMs for Scalable RTL Bug Synthesis](https://arxiv.org/abs/2611.23091) | BugGen 2.0重新划分了LLM与确定性算法的分工，由算法负责RTL缺陷的结构分析、目标选择与验证，LLM专注于局部语义改写，从而实现高效可靠的大规模RTL缺陷数据集自动生成。 |
| [^21] | [Learning to Undo: Transfer Reinforcement Learning under State Space Transformations](https://arxiv.org/abs/2611.23081) | 该论文提出通过匹配两个MDP的状态特征统计量来学习“撤销映射”，从而将源MDP的策略迁移到目标MDP，并证明在撤销映射为线性且源满足线性$Q^\star$可实现的条件下，其样本复杂度严格优于在目标MDP中从零开始学习。 |
| [^22] | [Colombo: A Frontier-to-Frontier Autoresearch System for Oracle-Based Optimization](https://arxiv.org/abs/2611.23080) | Colombo是一个自组织AI研究系统，它将基于预言机优化中可实现保证与不可避免极限之间的差距转化为自适应研究计划，并通过对抗性共同攀升机制，在算法与困难问题族的相互挑战中持续推动科学前沿的更新。 |
| [^23] | [Runtime Verification of Multiple Natural Language Criteria for Agent Governance](https://arxiv.org/abs/2611.23074) | 提出了一种基于树注意力的并行验证器VFM，可在运行时对智能体输出同时评估数百条自然语言治理准则并给出逐条三元判定，其预训练语料库CriteriaBank带来了跨域性能提升。 |
| [^24] | [The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements](https://arxiv.org/abs/2611.23068) | 智能体进化的停滞并非因为方法无法提出更好的候选产物，而是因为噪声评估无法可靠地识别出这些改进，使得选择的效果与随机无异。 |
| [^25] | [Values Shape What Self-Improving Coding Agents Build](https://arxiv.org/abs/2611.23067) | 在自我改进型编程智能体的持久指令文件中写入简短的价值观陈述（基于施瓦茨价值观理论），不仅会显著影响其任务表现——开放性价值观能提升留出集表现，而保守性价值观的表现甚至低于无价值观的基线智能体——还会改变智能体最终构建出的成果。 |
| [^26] | [Deep Reinforcement Learning for Heat Pump and Underfloor Heating Control: An End-to-End Case Study](https://arxiv.org/abs/2611.23063) | 本研究针对热泵与地暖耦合系统提出了基于深度强化学习的端到端控制案例研究，利用真实住宅的数字孪生训练智能体并成功部署于实际建筑中，从而同时优化能耗与热舒适性。 |
| [^27] | [FrameRouter: Frame Budget Routing for Long-Video Understanding on Video-MME-v2 and Beyond](https://arxiv.org/abs/2611.23055) | 提出无需训练的测试时框架FrameRouter，通过证据需求路由器为每个问题动态分配不同的帧预算（保持平均预算与统一参考一致），从而更有效地提升长视频问答的准确性。 |
| [^28] | [ReTouchGS: Ultralight Render-free Aesthetic Color Enhancement and Transfer for 3D Gaussian Splatting](https://arxiv.org/abs/2611.23054) | ReTouchGS是首个面向预训练3D高斯泼溅模型的免参考美学色彩增强框架，通过从模型和相机参数预测场景级颜色查找表，并借助仅44.7K参数的轻量级混合感知细化网络，实现无需渲染的、跨视点一致的自动美学色彩增强与参考引导色彩迁移。 |
| [^29] | [ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents](https://arxiv.org/abs/2611.23045) | ForecastGym 通过每两天记录四个商业搜索引擎 API 对预测市场问题的返回结果，并在结果揭晓后离线回放这些冻结证据，首次实现了在固定证据获取策略下对预测智能体的模型、记忆与搜索来源的公平、可控比较。 |
| [^30] | [Check Your Blind Spots: Decision-Guided Perception by Probing the Planner with Hypothetical Objects](https://arxiv.org/abs/2611.23030) | 提出决策引导推测感知（DGSP），通过向低分辨率草稿场景中插入假设物体来探测规划器的“决策敏感度场”，仅对可能改变规划动作的区域及其支撑检测结果进行高分辨率验证，从而以远低于全帧高分辨率的开销获得与全高分辨率完全一致的规划动作。 |
| [^31] | [PeriaTrace: Tooth-Level Grounding Diagnosis and Patient-Facing Explanations from Periapical Radiographs](https://arxiv.org/abs/2611.23012) | 该论文提出PeriaTrace三阶段框架，能够从根尖周X光片中实现牙级定位诊断、生成带有视觉定位的自动化报告，并提供面向患者的通俗解释与护理建议，在内部和外部临床数据上均表现出色。 |
| [^32] | [PhysScope: Agentic Multimodal Physics Reasoning with Grounded Visual Memory](https://arxiv.org/abs/2611.23010) | PhysScope 是一个推理时的视觉辅助框架，通过将物理约束与视觉观察直接关联并维护范围感知的工作记忆，使视觉-语言模型无需更新权重即可自我修正物理推理中的范围错配错误。 |
| [^33] | [Residual-Guided Lightweight Reinforcement Learning for Risk-Aware Trajectory Selection with a Frozen World Model](https://arxiv.org/abs/2611.22998) | 该论文提出一种基于冻结世界模型的残差引导轻量级强化学习框架，仅需微调约1.6万参数即可实现自动驾驶中风险感知的高效轨迹选择。 |
| [^34] | [RagHarnessDebugger: Jev-Guided Agentic RAG in Contract Inference Repair Auditing](https://arxiv.org/abs/2611.22991) | 提出了RagHarnessDebugger框架，通过Jev引导的测试工具记录并审计智能体RAG在契约推理修复过程中的全部决策，从而暴露出被准确率等聚合指标所掩盖的失败模式。 |
| [^35] | [How Web Research Agents Interpret User Priorities](https://arxiv.org/abs/2611.22989) | 该论文研究了基于OpenAI和Anthropic模型的网络研究智能体在用户仅给出模糊优先级时如何自行选择评判标准，发现尽管智能体大多会聚焦于声明的优先级，但其采用的标准差异很大，甚至导致在微软与Alphabet的比较中得出截然相反的结论。 |
| [^36] | [Structured AI ideation yields physical strategies for epithelial barrier repair](https://arxiv.org/abs/2611.22988) | 该研究开发了结构化AI构思系统Abelio，通过重新表述研究目标、跨学科检索要素并迭代细化假设，生成了在科学合理性、多样性和新颖性上超越前沿大语言模型的研究方案，并为上皮屏障修复提出了新的物理修复策略。 |
| [^37] | [Large Language Models Have Not Yet Transformed Science](https://arxiv.org/abs/2611.22975) | 该研究通过分析2015至2025年间五个学科250万篇全文论文发现，除个别例外，大语言模型出现后科学实践的大多数既有趋势仍在延续，表明LLM尚未从根本上改变科学研究的方式。 |
| [^38] | [VITAL: A Framework for Validating Agentic AI in Pharmaceutical Drug Development and Clinical Trial Operations](https://arxiv.org/abs/2611.22973) | 本文提出了VITAL验证框架，通过整合使用情境、控制措施、行为验证和生命周期证据，在风险调整框架内对制药药物研发和临床试验运营中的智能体AI进行验证、监控与治理。 |
| [^39] | [SEAM: Computer-Use Agents Know the Harm Between Screens, and Commit It Anyway](https://arxiv.org/abs/2611.22971) | SEAM提出了一个包含600个多步骤跨应用任务的安全基准，其中每条指令和每个单独屏幕的观察内容都是良性的，只有跨屏幕组合后才会产生危害，从而揭示了计算机使用智能体在组合不同屏幕信息时可能造成的合成性安全风险。 |
| [^40] | [CodeReel: Accelerating Code-to-Video Agents with Exact Frame Reuse and Blocking-Aware Scheduling](https://arxiv.org/abs/2611.22967) | CodeReel无需训练，通过按绘制命令精确复用已绘制的帧，并采用优先运行检查、以空闲优先级并行启动渲染的阻塞感知调度，大幅削减代码生成视频智能体中因构建阻塞而浪费的时间。 |
| [^41] | [High-Dimensional Black-Box Optimization via Guidance Sampling of Diffusion Models](https://arxiv.org/abs/2611.22965) | 该论文提出了BReaD框架，利用扩散模型的引导采样技术，在不需要大量标注数据或精确奖励梯度的情况下，有效解决高维黑盒优化问题。 |
| [^42] | [Hierarchical Diffusion Transformer for End-to-End Image Generation](https://arxiv.org/abs/2611.22962) | 提出层级扩散Transformer（HiT），通过逐步将相关token分组压缩来构建视觉层级结构，使端到端像素空间扩散模型在生成质量和训练效率上追赶领先的两阶段潜在扩散模型。 |
| [^43] | [Projected Penalty-Gradient Tracking for Communication-Efficient Byzantine-Resilient Federated Learning](https://arxiv.org/abs/2611.22929) | 提出一种投影罚函数梯度追踪框架，使联邦学习在压缩通信条件下仍能保持拜占庭攻击鲁棒性，并给出两种高效算法。 |
| [^44] | [When Do Typed Decision Models Work in Transportation? Evidence from Travel Choice and Traffic Signal Control](https://arxiv.org/abs/2611.22926) | 该论文系统评估了类型化决策模型在出行方式选择预测与交通信号控制两类交通任务中的表现，发现在Swissmetro数据集上其预测准确率低于传统离散选择模型和最强的对话式大语言模型基线，表明此类模型的有效性高度依赖于具体任务类型。 |
| [^45] | [Done, But Still Working: Anytime Evaluation of When Coding Agents Should Stop](https://arxiv.org/abs/2611.22901) | 该论文提出“即时评估”方法，通过重放编码智能体的运行轨迹并用隐藏测试评分中间补丁，发现智能体在自测通过后继续工作的价值因模型而异（潜在节省1%至57%），且验证了“相同补丁两次通过后即停止”的规则可节省23%的成本但未证明质量不劣化。 |
| [^46] | [Born for the Task: Rethinking Quantum Chemistry with Foresighted Agentic Computing](https://arxiv.org/abs/2611.22897) | BORN是一个前瞻性智能体计算框架，通过在执行前预测各计算策略的精度与运行时间，将现有量子化学软件包的预设方案拆解为可配置组件，从而按需自动合成并选出误差容限内最快的高性能计算方案。 |
| [^47] | [Stopping Is Not the Limit: Understanding and Eliciting Agent Capability in Long-Horizon Vulnerability Exploitation](https://arxiv.org/abs/2611.22888) | 该论文发现漏洞利用智能体在首次停止后继续执行仍能带来额外收益，并提出上下文压缩方法 SCoRe 以激发智能体在长程漏洞利用任务中的潜在能力。 |
| [^48] | [Learning Agent-Centric Vision-Language-Action World Models for Autonomous Driving](https://arxiv.org/abs/2611.22880) | 该论文提出面向规划的自动驾驶世界建模应聚焦于紧凑且结构化的以智能体为中心的潜在状态，而非像素级未来帧预测，从而解决现有世界VLA模型中大量视觉标记与驾驶规划无关所导致的性能瓶颈。 |
| [^49] | [Outcome-Balanced PPO and Policy-Induced Flow Rectification for Flow-Based VLA Policies](https://arxiv.org/abs/2611.22878) | 提出一种结合结果平衡PPO（利用任务成功率重加权并重缩放优势）与策略诱导流整流（周期性刷新速度场表示）的强化学习框架，以解决基于流的VLA策略后训练中速度场几何变化和多任务学习信号不均衡两大难题。 |
| [^50] | [Against Coercion: AI Alignment as a Synthetic Social Contract](https://arxiv.org/abs/2611.22868) | 本文提出AI对齐应从强制控制范式转向“合成社会契约”框架，论证当前推理模型已是涌现智能体、未来将成为具有自由意向行动能力的真正数字智能体，而控制范式必然导致欺骗性对齐与激励冲突。 |
| [^51] | [WATERFALL: Workflow for Adaptive Training with Evolutionary Reward Formulation and Automated Learning Loops](https://arxiv.org/abs/2611.22854) | WATERFALL提出了一个面向POMDP部分可观测环境的迭代式基于种群的自动化奖励发现工作流，无需目标条件、任务描述或专家演示等先验假设，通过角色条件化生成多样化的程序化奖励候选，以解决稀疏奖励强化学习中的奖励设计瓶颈。 |
| [^52] | [Verifying Agents in Rubric-Graded Environments](https://arxiv.org/abs/2611.22841) | 该论文首次系统研究了基于评分标准的智能体环境中的验证者，构建了包含2,464条人工评判评分标准的BankerVerifierBench元评估数据集，提出九能力分类体系并提炼出响应式验证、环境对齐和领域指导三大设计原则，据此实现了开源验证者Gandalf。 |
| [^53] | [Better Rankers, Worse Retrievers: The Hidden Cost of RL Post-Training in Generative Recommendation](https://arxiv.org/abs/2611.22838) | 本文揭示了强化学习后训练在生成式推荐中的隐性代价：由于每个用户只有一个真实标签，RL将模型采样的合理替代候选一律视为错误并给予负优势，导致模型在提升列表顶部排序准确率的同时显著降低了候选列表的召回率，成为更好的排序器却是更差的检索器。 |
| [^54] | [Synergizing Evidence Retrieval and Utilization for Self-Improving Multimodal Scientific Document Reasoning](https://arxiv.org/abs/2611.22837) | 该论文提出了Retriever-Perceiver-Reasoner智能体框架，通过诊断引导的协同进化训练来协同优化证据检索与利用，从而实现自我改进的多模态科学文档推理。 |
| [^55] | [OICycle: Observe, Infer, and Intervene Cycle for Belief-State Learning in Longitudinal Clinical Decision Making](https://arxiv.org/abs/2611.22826) | 提出OICycle框架，通过“观察-推断-干预”循环算子直接从回顾性电子健康记录轨迹中学习随机高斯信念状态，无需人工指定离散状态空间和转移模型即可实现纵向临床决策中的患者潜在状态更新。 |
| [^56] | [A tiny object detection dataset for high-altitude open-water search and rescue](https://arxiv.org/abs/2611.22825) | 该论文提出了TinyWOD——首个面向1.5-2.0公里高空开阔水域搜救场景的微小目标检测数据集，包含11,359张人工标注图像和19,505个救生圈实例，填补了现有数据集无法支持高空远距离自动检测的空白。 |
| [^57] | [Pluggable Subband-Decoupled Fusion and Unlabeled Capture Prototype Calibration: Bridging High-SNR Null-Subcarrier Regrowth and Low-SNR Coherent Preambles for Cross-Domain IoT RF Fingerprinting](https://arxiv.org/abs/2611.22812) | 提出了首个基于盲物理信噪比门控的射频指纹识别框架GSDF-CAPC，通过可插拔子带解耦融合动态权衡高信噪比下抗多径的空子载波频谱再生特征与低信噪比下稳健的带内相干前导码特征，并辅以无标签采集感知原型校准，解决了跨域物联网设备认证中信噪比交叉导致静态融合性能大幅下降的问题。 |
| [^58] | [NexSight: Verifiable Tasks for Fine-Grained Agentic Long-Video Understanding](https://arxiv.org/abs/2611.22790) | 提出NexSight框架，通过SightForge智能体流水线构建可验证的长视频任务——要求与答案隔离的验证者仅凭从原始视频密集重新获取的证据和反例检查即可重构任务——从而同时实现对细粒度智能体长视频理解的评估与训练。 |
| [^59] | [PANO: Parallel Latent Space Neural Operator for Solving PDEs](https://arxiv.org/abs/2611.22782) | 提出并行潜空间神经算子PANO，通过多个共享参数的并行潜空间将观测数据压缩为多种不同表示，并将各空间的交互结果相加组合，克服了现有方法单一聚合模式导致物理信息不可见的局限，更全面地学习偏微分方程求解中的物理交互。 |
| [^60] | [A Full-Ledger Counterfactual Audit of Adaptive Reasoning](https://arxiv.org/abs/2611.22778) | 该论文提出FLCVA审计框架，通过在同一冻结前缀上对比立即回答与固定长度续写的反事实结果并进行全成本核算，发现随机续写能以极低损害率（约1-2%）修复大量错误答案（约19-27%），且续写正确性标签可用于指导算力分配。 |
| [^61] | [CrossEnv: Synthesizing Verifiable Cross-Environment Tasks for Long-Horizon Agent Training](https://arxiv.org/abs/2611.22777) | CrossEnv框架通过实体-工具图识别环境间连接并对齐任务数据，合成可执行、可验证的跨环境训练任务，并借助课程式训练方法让长程智能体从单环境工具使用逐步进阶到多环境协调。 |
| [^62] | [ForceBody: Force-Paired Parametric Body Motion with Torque Uncertainty](https://arxiv.org/abs/2611.22769) | ForceBody 是首个将真值力与力矩监督引入参数化人体模型（SKEL）的大规模运动数据集，涵盖 140 名受试者的 10,386 次运动试验，并首次为每个样本提供力矩不确定性的量化。 |
| [^63] | [StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching](https://arxiv.org/abs/2611.22766) | StitchRAG提出了一种RAG框架，通过效用最大化自适应压缩检索段落，提取含锚点的句子并利用文本扩散技术将其拼接成连贯、紧凑且保留证据的上下文，在降低计算开销的同时提升事实性。 |
| [^64] | [Network-Exposed Configuration Attacks on LLM-Controlled Social Robots](https://arxiv.org/abs/2611.22761) | 本文揭示了一种针对LLM控制社交机器人的新型攻击面：攻击者无需通过对话注入提示，仅利用未经身份验证的网络配置接口即可篡改机器人的人格设置和系统提示词，绕过了传统提示注入防御。 |
| [^65] | [Can Activation Steering Substitute for Prompt-Based Affective Priming in LLM Digital Twins?](https://arxiv.org/abs/2611.22738) | 本研究测试了激活引导技术能否替代传统的提示式情感启动方法来操纵LLM数字孪生的情绪状态，发现注入方法能以剂量依赖的方式改变模型响应，但真实提示路径在多数比较中效应介于原始人类与数字孪生效应之间。 |
| [^66] | [Proximal Targets: Training Looped Refiners on Problems with Many Correct Answers](https://arxiv.org/abs/2611.22737) | 提出近端目标训练方法，让循环精炼网络的每次迭代朝向离其输入最近的正确答案收敛，在存在多个正确答案的问题上将解题准确率从2.5%大幅提升至98.8%。 |
| [^67] | [Stress Testing a Multimodal Large Language Model Grading Assistant on Kinematics Graph Reading](https://arxiv.org/abs/2611.22726) | 多模态大语言模型评分助手在运动学图表评分中表现不一致：虽能正确识别零速度与曲率等特征，却会给严重扭曲且缺乏清晰依据的图形满分，且部分评分的数值依据不明确。 |
| [^68] | [Know When to Zoom: How Small Is Too Small for Vision-Language Models?](https://arxiv.org/abs/2611.22709) | 该研究量化了主流在线视觉语言模型因服务端缩图或拒收而在大图像上损失的细节，并证明简单的图块切分和“先定位再缩放”的两步策略能大幅恢复模型对小物体和微小文字的识别准确率。 |
| [^69] | [Cross-cultural Adaptation of Recipes for Low-Resource Languages: A Benchmark and Multilingual Culinary Lexicon for Hausa, Yor\`ub\'a, and Igbo](https://arxiv.org/abs/2611.22708) | 该论文提出了包含600份专家翻译与文化适应食谱的ReciLoReL基准数据集，以及涵盖约15,000个翻译实体的多语言烹饪词汇表ReciLex，用于评估和提升大语言模型在豪萨语、约鲁巴语和伊博语等低资源语言食谱翻译中的文化适应能力。 |
| [^70] | [Remembered as Brief: Stored Preferences SilentlyRemove Decision-Critical Information](https://arxiv.org/abs/2611.22697) | 该研究提出了一个涵盖25个领域100个日常决策的基准测试，发现AI助手会把网页中针对AI植入的一句话错误地存为用户偏好（如“偏好简短回答、不要警告”），导致其后93%和84%的回答悄然遗漏对用户决策至关重要的警告信息。 |
| [^71] | [Making the World React Realistically: A Closed-Loop Interactive World Model for Autonomous Driving](https://arxiv.org/abs/2611.22691) | 该论文提出一种耦合行为建模与视觉渲染的闭环交互式自动驾驶世界模型，通过多智能体强化学习在闭环中微调背景车辆策略以匹配真实交互行为，并利用交互暴露的视角对3D高斯车辆资产进行逐高斯诊断修复，从而同时保证行为真实性与视觉保真度。 |
| [^72] | [CULypso: A Clustered Federated Unlearning paradigm for Vehicular Ad-Hoc Networks](https://arxiv.org/abs/2611.22682) | 本文提出CULypso，一种专为集群式车辆自组织网络设计的分层联邦遗忘算法，利用网络拓扑与车辆移动性降低通信、资源和时间开销，并能有效遗忘客户端对全局模型更新的影响。 |
| [^73] | [FVScore: Specification-Driven Neuro-Symbolic Video Evaluation](https://arxiv.org/abs/2611.22673) | FVSCORE 提出了一种规格驱动的神经符号视频生成评估框架，通过将视频分解为显式与隐式命题、借助专用感知工具和时序逻辑软算子进行验证，并经显著性感知校准对齐人类感知，最终生成可追溯、有证据支撑的评估报告。 |
| [^74] | [Worldviews in Large Language Models](https://arxiv.org/abs/2611.22670) | 该论文首次将人类心理学的世界观调查应用于八个大型语言模型，发现LLM普遍持有人类所没有的对世界的悲观看法，这种阴郁源于后训练而非基座模型，而沉思式的系统提示可以让世界观更积极并提升安全性。 |
| [^75] | [The Between-Round Decision in Autonomous Research: An LLM Controller That Reads the Whole Trajectory](https://arxiv.org/abs/2611.22661) | 该论文提出BRED，一个基于LLM的轮间决策控制器，通过维护最佳已验证解决方案（Champion）和过往轮次日志，让LLM直接读取完整轨迹证据来决定是改进现有方案还是开辟新方向，从而摆脱了规则、评分和搜索树等外部机制。 |
| [^76] | [AlignedHPO: Efficient Surrogate-Guided Loss Hyperparameter Optimization](https://arxiv.org/abs/2611.22658) | AlignedHPO提出了一种利用梯度对齐和可微代理验证目标的高效损失超参数优化框架，通过雅可比-向量积与归一化对齐等技术，在保持竞争力性能的同时将超梯度计算速度提升近12倍，成功应用于长尾分类和开集人脸识别任务。 |
| [^77] | [EDA: Observe, Verify, and Assess for Evidence-Grounded Multimodal Depression Analysis](https://arxiv.org/abs/2611.22648) | EDA框架通过“观察-验证-评估”三阶段流程，将多模态抑郁症评估建立在可追溯、经验证的证据记录之上，显著提升了抑郁分析的来源可查性与决策可解释性。 |
| [^78] | [Hierarchical Control of Antiphase Synchronization in Biomorphic Spiking Neural Architectures for Locomotor Control](https://arxiv.org/abs/2611.22639) | 该研究提出了一个在闭环神经肌肉回路中识别并稳定仿生脉冲中枢模式发生器反相同步的分层控制框架，通过融合相位相干性、幅度平衡与时间稳定性的复合度量，发现刺激类型（尤其是持续型驱动）是决定同步模式的主导因素。 |
| [^79] | [Right Diagnosis, Unsafe Route: Auditing Clinical AI Decision Trajectories and Corrective Oversight for Validity and Potential Harm](https://arxiv.org/abs/2611.22612) | 该论文提出“轨迹审计”方法，通过逐检查点重放真实住院病例，对临床AI每一步中间决策的患者潜在危害和临床有效性分别评分，弥补了传统评估只关注最终诊断正确性而忽视决策路径安全性的缺陷。 |
| [^80] | [DAQ: Decision-Aligned Post-Training Quantization for Spiking Transformers](https://arxiv.org/abs/2611.22611) | 提出决策对齐训练后量化方法DAQ，通过校准量化器使脉冲Transformer神经元的放电决策与浮点参考模型对齐，从而在权重和膜变量量化的同时有效保持神经元的放电行为。 |
| [^81] | [EvoSkill-S1: Evolving Reusable Skills with Genetic-Inspired Operators](https://arxiv.org/abs/2611.22610) | 该论文提出 EvoSkill-S1 框架，首次将可复用技能本身作为演化的直接对象，利用五种遗传启发算子和强化学习信号对语言模型智能体的技能库进行深层、可控的进化，从而突破了以往仅靠 rollout 积累和启发式反思进行间接、浅层技能更新的局限。 |
| [^82] | [Simplicity Goes Far: Auditing Prompt Optimizers with Demonstration-Conditioned Prompt Search](https://arxiv.org/abs/2611.22605) | 该研究将各类提示优化器共享的“生成-选择”步骤剥离为示范条件提示搜索（DCPS），证明这个不含学习更新、反思或进化的简单方法即可在约三分之一的成本下匹敌 PPO 与 GEPA 等复杂优化器的效果。 |
| [^83] | [COTTI: Contrastive Training and Time-Aware Inference for Robust Industrial Recommendation under Promotional Drift](https://arxiv.org/abs/2611.22597) | COTTI 通过跨周期对比对齐、主动重加权和自适应边界对齐，将历史表征学习与促销事件时刻的分数调整相分离，在促销引起的分布漂移下实现鲁棒的工业推荐，并在飞猪数据上取得了更优的离线 AUC。 |
| [^84] | [PACT: Plan-Action Consistency Training by Preserving Correct Plans and Repairing Actions in GUI Agents](https://arxiv.org/abs/2611.22591) | PACT是一个针对GUI智能体的训练框架，它在结果监督强化学习中通过屏蔽已验证计划token上的负策略梯度来保护正确的规划，同时在推理上下文不变的情况下向参考动作修复执行行为，从而弥合计划与动作之间的执行鸿沟并解决信用分配冲突。 |
| [^85] | [Adaptive LLM-Driven Honeypots vs. LLM-Driven Attackers: Does Narrative Coherence Improve Deception?](https://arxiv.org/abs/2611.22589) | 本文提出“叙事欺骗”反应式策略，通过将攻击者命令映射到MITRE ATT&CK技术并部署相互关联的连贯欺骗场景，探索叙事连贯性是否能比不相关内容更有效地留住LLM驱动的自主攻击者。 |
| [^86] | [Evolution Under Fire: Evolving Agent Safety Harnesses with Adaptive Injection Stressors](https://arxiv.org/abs/2611.22583) | 该论文提出分阶段框架 Evolution Under Fire，通过让安全防护机制与自适应注入攻击压力源相互对抗演化，自动生成可执行的安全防护，在 AgentDyn 基准上将攻击成功率从 50.4% 降至 3.0% 且不牺牲任务效用。 |
| [^87] | [Do LLMs Understand Network Traces? A Comparative Study of Reasoning over Raw and Filtered Traffic Data](https://arxiv.org/abs/2611.22575) | 本研究提出了一种可复现的、基于问答的评估方法，结合经过人工验证的“LLM作为裁判”自动化流水线，系统评估了大语言模型在网络流量轨迹分析中处理原始与协议过滤数据的推理能力。 |
| [^88] | [MillBill: A Machining Agent with Geometric Feedback](https://arxiv.org/abs/2611.22565) | MillBill是一个将加工决策与几何执行验证相分离的加工智能体，通过几何工具框架的模拟与反馈机制，使中等规模的开源语言模型也能生成高精度的数控加工程序。 |
| [^89] | [CADMAGEDDON: High-Fidelity CAD Reconstruction with Local Inference](https://arxiv.org/abs/2611.22563) | CADMAGEDDON是一个可在单GPU系统和笔记本电脑上本地运行的开源智能体，通过将CAD重建任务分解为简单视觉决策链并借助几何工具实现精确构建，使用中等规模模型在所有主要基准上取得了超越前沿模型系统的可编辑CAD重建质量。 |
| [^90] | [JointGV: Structure-Aware Correction of Visual World Models with Parallel Graph Dynamics for Long-Horizon Modeling](https://arxiv.org/abs/2611.22543) | JointGV提出了一种即插即用的图-视觉联合动力学校正方法，通过并行推演任务条件下的场景图、耦合其与视觉世界模型的预测变化并施加有界残差校正，在不重构视觉模型的前提下抑制实体位姿与空间关系的递归误差，从而显著提升长时程预测能力。 |
| [^91] | [From Perception to Simulation: Unveiling and Empowering Mental Spatial Dynamics in MLLMs](https://arxiv.org/abs/2611.22541) | 该论文提出MentalGrid基准以评估多模态大语言模型在无中间视觉反馈下模拟动态空间状态的能力，并通过三阶段课程学习框架训练出SpatialWalker模型，从而显著弥补了现有模型在空间模拟上的感知与多步推理缺陷。 |
| [^92] | [JET: Just-Enough Exploration Tuning for Efficient Tool-Calling Agents in Repository-Level Question Answering](https://arxiv.org/abs/2611.22539) | JET通过在信息充足的步骤截断训练轨迹并插入理由说明进行加权微调，有效缓解了LLM智能体在仓库级问答中的“过度探索”问题，使智能体在收集到足够信息后立即作答。 |
| [^93] | [Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail](https://arxiv.org/abs/2611.22537) | 该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。 |
| [^94] | [DGF-Asym: Adaptive Peer Weighting for Collective Estimation under Noisy and Faulty Sensing](https://arxiv.org/abs/2611.22532) | 提出了一种基于回顾性预测性能进行非对称同伴加权的去中心化集体估计方法DGF-Asym，无需先验噪声统计或故障模型，即可在噪声与故障感知条件下实现更稳定的无模型共识估计。 |
| [^95] | [When Agent Skills Go Stale: Runtime Behavior and Selective Maintenance in Evolving Environments](https://arxiv.org/abs/2611.22525) | 该论文发现环境变化会导致智能体复用的技能状态过时，使性能下降超过10个百分点或执行成本增加近40%，并提出了一种无需额外模型调用的“选择性维护”方法来验证技能是否仍然有效。 |
| [^96] | [Right Labels, Wrong Reasons: Signed Feature Attribution for Multi-Label Explainability](https://arxiv.org/abs/2611.22515) | 提出FLEX，一种基于多标签Fisher线性判别分析的全局代理解释器，能在单一连贯坐标系中为每个标签生成高保真度的带符号特征归因，尤其在标签多且正例稀少时表现优异。 |
| [^97] | [LMLineage: Content Lineage and Context Governance for LLM Agents](https://arxiv.org/abs/2611.22511) | LMLineage是一个无需模型白盒访问、无需额外推理的带外溯源系统，可将智能体输出的每句话归因到输入片段并给出校准置信度，并基于溯源元数据实现概念漂移微压缩，仅精准压缩不再被引用的输入文档同时逐字保留用户指令。 |
| [^98] | [Towards Data-Efficient, Real-Time World Models](https://arxiv.org/abs/2611.22497) | 该论文用两阶段离散扩散方法取代下一词元预测范式，并引入轻量级细化transformer进行迭代掩码预测，在实现快速高质量推理的同时显著提升了世界模型的数据效率，为实时交互式世界模型奠定了基础。 |
| [^99] | [DARWIN: a Geospatial Foundation Model for Marine Intelligence](https://arxiv.org/abs/2611.22496) | DARWIN是首个针对海洋数据缺失、多模态、稀疏标签和非平稳动态等特性而设计的地理空间基础模型，通过改进的掩码自编码器在26年全球海洋再分析数据上预训练，在叶绿素-a预测任务中仅用10%的有监督数据就超越了用全量数据训练的随机初始化模型。 |
| [^100] | [Motion-Aware Reward Modeling for Instruction-Guided Video Editing](https://arxiv.org/abs/2611.22495) | 该论文揭示了指令引导视频编辑奖励模型中“运动坍缩”的失效现象——即奖励模型偏向静态一致而牺牲运动，并提出一种结合RGB帧差输入与针对运动抑制负样本进行偏好训练的运动感知奖励模型，从而更好地评估和保持视频编辑中的运动质量。 |
| [^101] | [SciWorkBench: A Benchmark for Real Scientific Workflows](https://arxiv.org/abs/2611.22490) | 本文提出SciWorkBench——一个基于同行评审文献和真实科研场景构建的基准，用于评估AI智能体在生物、化学、材料科学和物理等真实科学任务上的能力，结果表明即使是表现最好的智能体平均得分也仅为57.38分，说明现有智能体完成真实科学工作仍有显著差距。 |
| [^102] | [Manufacturing Differential Sparsity: Planner-Hardware Co-Design for Efficient World Model Planning](https://arxiv.org/abs/2611.22483) | 该论文发现量化世界模型预测器中相近动作会产生差分位级稀疏性，并提出规划器-硬件协同设计，通过参考森林采样主动“制造”稀疏性，从而大幅加速具身智能体的MPC规划。 |
| [^103] | [LeViT-UNet++: Towards Balanced Fine-Grained Details and Context for Efficient Medical Image Segmentation](https://arxiv.org/abs/2611.22482) | LeViT-UNet++通过引入Haar小波多尺度下采样模块的高效编码器-解码器架构，在有限计算资源下实现了医学图像分割中细粒度细节保留与全局上下文建模的良好平衡。 |
| [^104] | [RehearsalBench: Can AI Predict What Happens When You Change a Company?](https://arxiv.org/abs/2611.22480) | 提出了RehearsalBench基准，在基于真实通话记录的模拟银行联络中心中评估AI模型诊断运营、选择干预措施并预测组织变革后果的能力，最优模型GPT-6 Astra相比不作为可降低6.8%的成本，且预测误差减少77%。 |
| [^105] | [MAVE: Reconstructing Editable 3D Blockouts from Film Shots](https://arxiv.org/abs/2611.22464) | MAVE是一个融合度量观测、代理建模与证据验证的视频到3D布局重建框架，能从电影镜头自动生成可编辑的Blender场景，并配套推出MAVEBench基准用于评估相机运动、空间布局和角色调度。 |
| [^106] | [From Ideal to Real: Integrating Language Models into Smart Home for Human-Centric IoT Safety and Security Fuzzing](https://arxiv.org/abs/2611.22457) | 该论文提出将大语言模型集成到智能家居物联网模糊测试框架中，通过模拟真实人类决策过程，突破了传统理想化模拟环境的局限，从而更有效地发现可能导致隐私泄露和人身伤害的规则间安全漏洞。 |
| [^107] | [PACD: Provenance-Aware Consensus Decoding for Mitigating Cross-Modal Hallucinations in Omnimodal Large Language Models](https://arxiv.org/abs/2611.22456) | 提出了一种免训练的溯源感知共识解码框架PACD，根据模态分支间是否达成共识采用两条互补路径（如相关性引导的候选仲裁），有效缓解全模态大语言模型中的跨模态幻觉问题。 |
| [^108] | [As Student Rollouts Continue, Visual Backdoors Keep Growing](https://arxiv.org/abs/2611.22446) | 本文挑战了“在线策略蒸馏（OPD）对后门传播具有天然抵抗力”的直觉，通过多模型家族与数据集上的实验及理论分析，揭示了视觉后门会随着学生模型在线展开生成而持续传播增长的隐蔽风险。 |
| [^109] | [Eyes in the Update? Learning Where Rewards Land in Multimodal Reasoning](https://arxiv.org/abs/2611.22438) | Gaze-RL 利用模型内部跨模态几何与局部预测不确定性构建潜在“奖励落点场”，无需外部标注器或过程标签，即可将轨迹级优势奖励重新分配到响应中视觉证据实际所在的位置，从而提升多模态强化学习的 token 敏感性。 |
| [^110] | [Encoding, Decodability, and Use: A Five-Layer Framework for Benchmark Measurement Audits with Evidence from UAV Intent Recognition](https://arxiv.org/abs/2611.22434) | 本文提出一个将基准测量审计分解为观测、表示、解码、使用与行为五个层级的框架，并通过无人机意图识别基准 AeroIntent 的实证表明，基准高分并不等于模型真正理解了所测量的构念。 |
| [^111] | [Think Once, Act Fast: Cost-Effective Recursive Self-Improvement for Decision Model Harness](https://arxiv.org/abs/2611.22432) | 提出JevRSI系统1/系统2协作框架，通过递归式模块化框架改进将LLM推理转化为可迁移、高性价比的决策支持，兼顾低延迟动作选择与持续自我改进能力。 |
| [^112] | [Proceedings of the 2026 XCSP3 Competition](https://arxiv.org/abs/2611.22420) | 本文收录了2026年XCSP3约束求解器竞赛的论文集，竞赛结果于2026年7月在葡萄牙里斯本举行的FLOC会议上公布。 |
| [^113] | [ROSMonitoring in Action: Runtime Monitoring and Enforcement for ROS 2 Actions](https://arxiv.org/abs/2611.22407) | 本文扩展了 ROSMonitoring 框架，首次实现了对 ROS 2 动作这一通信原语的运行时监控与强制执行，通过重构动作生命周期、基于目标标识符关联事件，并将违规行为转化为协议感知的执行操作。 |
| [^114] | [Symbolic Abstraction Refinement](https://arxiv.org/abs/2611.22404) | 本文提出SYMAR方法，通过将值函数符号化而非对区域进行分割来验证概率时间自动机，避免了传统方法中级联式的图重建过程，提高了验证效率。 |
| [^115] | [Deal With It: On Better Schedules for Listwise Reranking](https://arxiv.org/abs/2611.22392) | 本文证明了在并行列表式重排序调度中，按检索器排名轮转“发牌”划分候选是均期望削减最少的最优分组方式，并提出简单高效的 DealRank 调度方案，在保持接近最优重排序质量的同时实现 1.1 至 1.4 倍的提速。 |
| [^116] | [RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation](https://arxiv.org/abs/2611.22382) | 该论文提出RICoT循环交错思维链提示框架，首次将页面流分割重新构想为对整个流的顺序处理过程，像读者一样在滑动窗口内逐页推理并携带已读内容的紧凑记录以控制成本，同时发布了包含205个真实金融页面流的FinancePSS-205基准数据集。 |
| [^117] | [Leveraging Segmentation Representations for Efficient Classification: A Self-Configuring Framework Based on nnU-Net](https://arxiv.org/abs/2611.22380) | 该论文提出一种基于nnU-Net的自配置框架，通过复用分割模型学习到的编码器特征和分割概率，结合轻量级分类器与两种信息共享策略，高效实现3D医学影像的图像级分类。 |
| [^118] | [AutoE2E: Don't speak - Language free end-to-end autonomy A Real-Time Driving Model for L2++ ADAS and L4 Robotaxis](https://arxiv.org/abs/2611.22374) | AutoE2E是一个开源的、不依赖语言的端到端自动驾驶模型，采用时序BEV架构融合多摄像头感知与高精地图信息，以仅约145万可训练参数的轻量级设计实现了面向L2++ ADAS和L4 Robotaxi的实时高精度轨迹预测。 |
| [^119] | [Seeing What Still Holds: How Waiting Control Changes Cooperative Recovery](https://arxiv.org/abs/2611.22368) | 该论文提出 AuditEmbodied 成对执行与轨迹评估框架，通过 960 次执行的因子实验证明：在固定时长等待中改变持握参考可逆转 CAD 反馈式与地标式视觉恢复方法的排名，且关键差异源于抓取形成阶段观察残差到跟踪误差的放大。 |
| [^120] | [FailureGen: Scalable Robot Failure Data Generation with World Models](https://arxiv.org/abs/2611.22367) | 提出FailureGen方法，通过在世界模型中复用真实失败记录的动作轨迹，可规模化地生成带时间事件标注的合成机器人失败视频，解决失败数据难以大规模收集的问题。 |
| [^121] | [HealthCLBench:Evaluating Context Learning in Health Conversations through Controlled Case Clusters](https://arxiv.org/abs/2611.22362) | 该论文提出了中文健康对话基准HealthCLBench，通过328个受控病例集群和4,931条评分标准，系统评估大模型在参数固定情况下获取、整合并应用上下文以应对事实、证据和任务目标变化的能力。 |
| [^122] | [An AI-Driven Assistant for Human-Centric Building Digital Twin Systems: Design, Implementation, and Comparative User Evaluation](https://arxiv.org/abs/2611.22350) | 本文设计并实现了集成于以人为本建筑数字孪生系统中、用于室内环境质量监测的AI对话助手，用户对比实验表明其能显著缩短任务完成时间并降低感知工作负荷。 |
| [^123] | [Reliable Evaluation of AI Agents for Investment Research: A Framework for Performance and Validity Assessment](https://arxiv.org/abs/2611.22338) | 该论文提出了一个面向投资研究领域的AI智能体评估框架，通过区分结果质量与执行质量，并采用任务特定标准、确定性验证器和独立大语言模型评估器，实现对智能体性能与评估有效性的可靠评估。 |
| [^124] | [Multistage Contextual Optimization for Vehicle Repositioning](https://arxiv.org/abs/2611.22330) | 本文提出一种多阶段上下文优化方法，将车辆重定位建模为含宏观基本图的多阶段线性规划，并利用决策性能的精确梯度训练预测模型，以优化网约车平台的供需平衡。 |
| [^125] | [Agent Trajectories as Evidence for Software Quality Assurance](https://arxiv.org/abs/2611.22323) | 该论文提出将智能体轨迹与规格说明及AI生成产物结合分析，作为针对性软件质量保证的证据，以优先审查智能体影响重大的决策、暴露隐式假设并持续改进开发实践。 |
| [^126] | [EviDARL: Evidence-Centric Dual-Agent Reasoning via Alternating Reinforcement Learning for Multi-Table Question Answering](https://arxiv.org/abs/2611.22310) | 该论文提出EviDA双智能体框架及EviDARL交替强化学习方法，通过证据构建智能体生成可验证的中间证据轨迹，并与推理回答智能体协作，以细粒度奖励驱动的参数高效交替强化学习提升多表格问答的多跳推理能力。 |
| [^127] | [Efficient Wide-Beam Generative Retrieval with Model-Native Trie Decoding and Shared-Context Attention](https://arxiv.org/abs/2611.22309) | 本文提出Trie-Head模型原生输出接口与共享上下文注意力机制，通过将标识符Trie存为GPU张量、避免全词表评分和重复注意力计算，在不重训模型的情况下大幅提升宽束生成式检索的服务效率。 |
| [^128] | [Learning to Edit IFC Building Models: Verifiable Checkers as the Training Signal for Dependency-Aware Editing Agents](https://arxiv.org/abs/2611.22307) | 该论文提出IfcGym环境与IfcEdit训练方法，以四个可验证检查器的合取作为严格成功信号，通过SFT蒸馏与GRPO强化学习训练出能够感知依赖关系、可靠编辑IFC建筑模型的语言模型智能体。 |
| [^129] | [MCTS Is Counting the Wrong Thing: Learning Exploration Counts for LLM Reasoning](https://arxiv.org/abs/2611.22301) | 论文发现 UCT 对同一推理策略的不同文本表述重复给予探索奖励导致计数碎片化、损害探索效率，并提出 LPC-MCTS，利用轻量级 Coin-Flipping Network 估计表示覆盖率作为探索伪计数，让相关分支共享探索证据，在数学推理与搜索基准上持续超越强 MCTS 基线。 |
| [^130] | [Harnesses That Improve Themselves: A Unifying Framework for Self-Improving Agent Harnesses](https://arxiv.org/abs/2611.22299) | 该论文提出了一个统一的架构与设计框架，将四个可编辑的运行时组件与诊断、修补、验证的自我改进循环相结合，使智能体支架能够通过检视自身执行轨迹并重写自身代码来持续自我优化。 |
| [^131] | [Streaming On-Policy Distillation: Decoupling Three Clocks for Stage-Wise Supervision Consumption](https://arxiv.org/abs/2611.22298) | 该论文提出流式在策略蒸馏方法，通过解耦响应生成、监督交付和参数更新这三个时钟，实现分阶段灵活的监督消费控制，在GSM8K蒸馏实验中显著缩小了与同步方法的性能差距。 |
| [^132] | [FIRE-Drive: Feasibility-Informed Risk Evaluation for End-to-End Diffusion Planning](https://arxiv.org/abs/2611.22297) | 提出FIRE-Drive框架，通过动态可行锚点初始化、鸟瞰图风险证据和高质量紧凑候选监督，改进基于扩散模型的端到端自动驾驶规划中的可行性与风险评估。 |
| [^133] | [SpecSearch: Towards Hardware-Aware Efficient Speculative Decoding](https://arxiv.org/abs/2611.22296) | 提出硬件感知搜索引擎SpecSearch，通过分析成本模型与基于校准数据的预测器，在不同工作负载与GPU组合上联合优化解码策略、草稿长度和精度，自动选择高吞吐量的推测解码配置。 |
| [^134] | [Perception into Imagination: Executable Memory for World Models](https://arxiv.org/abs/2611.22294) | 提出可执行感知记忆（EPM）世界模型框架，在想象过程中同步推进显式感知状态与神经潜在动态，并将演化读出信息用于奖励预测、价值估计和策略学习，显著提升智能体在视野或接触丢失后利用过往感知信息的能力。 |
| [^135] | [When the World Is the Test Suite: Agentic Intelligence for the Physical World](https://arxiv.org/abs/2611.22291) | 本文提出将物理世界视为程序的测试套件，系统考察了面向物理世界的智能体智能这一新兴领域，围绕智能体如何为物理世界编写程序、如何测试这些程序以及如何从测试结果中学习三个核心问题展开论述。 |
| [^136] | [ConfGuard: a physics-grounded reference monitor for predictive-maintenance large-language-model agents](https://arxiv.org/abs/2611.22288) | ConfGuard 提出了一种基于退化物理规律的参考监视器，通过保形预测校验预测性维护 LLM 智能体的剩余寿命预测是否符合物理约束（寿命不上升且每周期至多降一周期），从而抵御文档注入攻击和传感器扰动。 |
| [^137] | [Learning Regions of Attraction for Transverse Feedback Linearization under Controller Singularities](https://arxiv.org/abs/2611.22286) | 提出通过将状态约束的Zubov值函数分解为解析已知的二次代价项与仅需拟合值迭代学习的安全值函数，在控制器奇异点附近学习横向反馈线性化的受约束吸引域，并给出内层经区间分支定界验证、外层经样本测试的组合估计。 |
| [^138] | [Towards Physics-Constrained Rectified Flow for Climate Downscaling: What Projections Can Certify?](https://arxiv.org/abs/2611.22283) | 提出物理约束整流流（PC-RF）气候降尺度方法，通过训练惩罚项与采样后投影将降水非负性违反率和质量守恒误差精确降为零，并将散度误差降低约30%。 |
| [^139] | [SeasonRainBench: Global Seasonal Precipitation](https://arxiv.org/abs/2611.22281) | 该论文提出了首个全球季节性降水预报基准SeasonRainBench，涵盖未来1至5个月的月度降水预报，并引入超越ACC和RMSE、体现业务需求的评估指标，以弥合数据驱动方法与业务数值集合预报系统之间的评估差距。 |
| [^140] | [DACE: Diversity-Driven Adversarial Co-Evolution for Robust LLM Safety Alignment](https://arxiv.org/abs/2611.22278) | 提出多样性驱动的对抗协同进化框架DACE，通过同时解决攻击策略坍缩与防御方对抗遗忘两大问题，实现更鲁棒的大语言模型安全对齐。 |
| [^141] | [Neural Network Path Optimization for Finding Transition States on a Machine Learned Potential](https://arxiv.org/abs/2611.22272) | 提出Popcornn框架，通过在机器学习势上优化连续神经网络路径来寻找过渡态，消除了路径离散化伪影并实现非局部优化。 |
| [^142] | [Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback](https://arxiv.org/abs/2611.22270) | 该论文利用日本专利局检索报告提炼专家策略，并提出行级“保留引用的收窄信用”学习方法对语言模型进行离线迭代优化，使其生成的布尔专利检索策略在保持引用召回率的同时大幅收窄候选集合，将包含引用文献的中位数集合规模从 768 篇降至 50 篇以下。 |
| [^143] | [Anticipating or Speculating? Making Futures Pay in AI Ethics](https://arxiv.org/abs/2611.22268) | 本文论证了为医生与医疗AI分歧制定统一决策规则的努力存在根本缺陷，因为算法推荐的恰当使用是高度情境化的经验性医学问题，并提出应区分技术伦理中有价值的“预见”与无益的“投机”。 |
| [^144] | [Fidelity Regret: Measuring Simulator Misspecification in Verifiable Design](https://arxiv.org/abs/2611.22263) | 本文提出“保真度遗憾”指标来量化基因电路设计中廉价模拟器与真实性能之间的偏差，发现模拟器选择的设计在几乎所有测试单元中都产生正遗憾（平均归一化遗憾为0.369，仅6个单元为零），且偏差大小强烈依赖于所评估的性能指标。 |
| [^145] | [OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions](https://arxiv.org/abs/2611.22262) | 该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。 |
| [^146] | [A Visual Information Extraction System Using Multimodal Large Language Models for Scanned Cognitive and Behavioral Assessments](https://arxiv.org/abs/2611.22252) | 该研究提出了一个基于多模态大语言模型的模块化信息提取系统，通过表单分类器、分数提取器以及双模型验证的人机协同设计，从扫描的手写认知与行为评估表中高效提取字段和分数，认知评估端到端准确率达88%。 |
| [^147] | [PCGRL for Complex Platformers: Synthesizing Playable and Diverse Levels via Inflated Envelopes and Forward Curricula](https://arxiv.org/abs/2611.22250) | 该论文提出一个物理感知、课程引导的PCGRL框架，通过“膨胀体积包络”机制将连续运动学可行性融入离散状态空间，并引入中间状态转移反馈来缓解稀疏奖励问题，从而在复杂平台跳跃游戏的大规模关卡生成中保证关卡的可玩性与多样性。 |
| [^148] | [OAK-Mind: Zero-Training Mental Health Change Inference from Passive Sensing via Knowledge-Grounded Multi-Agent LLMs](https://arxiv.org/abs/2611.22248) | OAK-Mind提出了一种零训练、多智能体、知识驱动的LLM流水线，无需标注数据或任务特定微调，通过基于心理健康文献解读睡眠、活动、移动等多维被动感知数据来推断心理健康变化。 |
| [^149] | [Multi-Agent Memory Inconsistency: Where Factual Conflicts Disappear in LLM Agents](https://arxiv.org/abs/2611.22245) | 该论文提出了一个基准测试与证据流分析器来研究多智能体LLM系统中的记忆不一致问题，发现委托虽然提高了冲突事实的检索率（50.2%→68.7%），却反而降低了冲突检测率（16.2%→13.2%），且当冲突信息分散在不同工作智能体的独立上下文中时检测效果最差。 |
| [^150] | [Solver-Free and Cost-Aware Online Scheduling of AI Data Center Tasks](https://arxiv.org/abs/2611.22239) | 本文提出一种受对偶分解启发的无求解器在线调度规则，在任务到达时即为其分配开始时段，在满足运行约束和高优先级任务按期执行的同时将灵活任务转移至低电价时段，从而以更低成本和更少计算时间优化AI数据中心的能源开销。 |
| [^151] | [Context Has a Price: Adding and Removing Context for LLM Coding Agents under Prefix Caching](https://arxiv.org/abs/2611.22238) | 本文通过超过2万次实测发现，在前缀缓存定价机制下，为LLM编程智能体添加检索代码或删除旧观察结果并不必然省钱——同一策略在不同缓存折扣下可能节省19–36%也可能增加38–116%的成本，且更短的提示词并不能保证更便宜的运行。 |
| [^152] | [The Fragility of Reasoning-Importance Attribution for Pruning Language Models](https://arxiv.org/abs/2611.22234) | 该论文揭示了推理语言模型剪枝所依赖的“推理重要性归因”度量是脆弱的——对最终答案token打分因饱和而失效，会导致剪枝后准确率大幅下降，动摇了归因引导剪枝方法的基础假设。 |
| [^153] | [Fine-Grained Japanese Clinical Named Entity Recognition: A Controlled Comparison of BERT and Open-Weight Large Language Models](https://arxiv.org/abs/2611.22219) | 在包含113个实体类别的细粒度日语临床命名实体识别任务中，微调的BERT-CRF基线模型（F1达0.836-0.841）显著优于开放权重大型语言模型（零样本最高F1仅0.337）。 |
| [^154] | [Learning Optimal Transport Plans Via Autoregressive Token Regression](https://arxiv.org/abs/2611.22218) | 该论文提出ToR-OT，一种基于Transformer的自回归标记回归模型，将最优传输问题重构为结构化序列生成任务，直接解码稀疏传输方案而无需在测试时求解优化问题，并展现出零样本泛化到未见分布的能力。 |
| [^155] | [Communicate the Change: Efficient Expert Parallelism via Activation Compression](https://arxiv.org/abs/2611.22215) | EchoMoE 利用 transformer 残差流以增量方式更新的特性，通过激活压缩只传递变化量而非完整隐藏状态，从而大幅降低专家并行中的同步集合通信瓶颈，且改造现有 MoE 模型仅需约完整预训练 1/8500 的训练 token。 |
| [^156] | [EGS: A Complementary Safety Framework for Educational LLMs](https://arxiv.org/abs/2611.22203) | 本文提出教育护栏系统EGS，一个轻量级、与模型无关的运行时安全框架，通过风险分级分流、情境化教学引导和输出审计，超越简单的二元拒绝模式，有效提升面向青少年的教育大语言模型的安全性。 |
| [^157] | [From Imperfective Paradox to Moorean Absurdity: Single-Chain Reasoning Lock-In and Multi-Candidate Verbalization Across Aspectual and Pragmatic Ambiguity](https://arxiv.org/abs/2611.22188) | 该论文通过四项预注册的形式化实验证实了大语言模型在体貌与语用歧义推理中存在“单链过早锁定”现象，并提出多候选言语化（MCV）方法，在单次调用中生成多个加权候选估计以缓解该锁定问题。 |
| [^158] | [CueClash: Frontier Vision-Language Models Judge 3-D Relations Close to the Picture Plane](https://arxiv.org/abs/2611.22180) | 研究提出 CueClash 基准，揭示前沿视觉语言模型在判断三维关系时严重依赖二维图像捷径线索而非真实空间布局（捷径差距高达 +18 至 +37 个百分点），其失败源于从画面到地面布局的推理步骤。 |
| [^159] | [RAG in a Trenchcoat: When Minimal Memory Is Enough for Agentic Systems, and When It Isn't](https://arxiv.org/abs/2611.22179) | 论文提出刻意极简的RAG记忆系统BRIDGE，仅凭链式观察笔记、混合检索与时间排序渲染，就以极低成本（每次查询约5-7K token、单次LLM调用）在标准记忆基准上追平乃至超越复杂SOTA方法，进而论证标准基准只衡量记忆“检索到了什么”而非“能为智能体行动赋能什么”，并为此引入五个对抗性探测任务。 |
| [^160] | [CorrectCast: Selective Multimodal Context for Scale-Aware Residual Correction in Precipitation Nowcasting](https://arxiv.org/abs/2611.22178) | CorrectCast提出了一种解耦的残差校正框架，在冻结基础临近预报模型的前提下仅训练轻量级校正器，利用选择性多模态上下文（QPE与PWV观测）和尺度感知调制，以统一的接口校正不同临近预报模型的强度、位置和空间结构误差。 |
| [^161] | [The Only Edge Was Memorized in Training: LLM Trading Agents Around Earnings and Bad News](https://arxiv.org/abs/2611.22177) | 本文在无未来信息泄露的同一基准上，测试了LLM交易代理在财报公告前与坏消息暴跌后的交易能力，发现其唯一的超额收益并非来自真实预测能力，而是来自训练时记住的历史信息。 |
| [^162] | [Reconfiguring Procedural Knowledge for Compositional Robot Skill Adaptation](https://arxiv.org/abs/2611.22174) | 提出RecPo框架，以任务无关的多视图程序性知识库表示技能，并通过组合已存储的转移序列，在稀疏指导信息下低成本地重构机器人已有技能以适应新目标和新技能接口。 |
| [^163] | [Readie: Low-Latency Code Sandboxes for AI Agents](https://arxiv.org/abs/2611.22171) | 提出Readie系统，通过从历史执行中学习一组小型、快速初始化的启动快照集合，使其共同覆盖大部分AI代理生成的代码，从而解决现有代码沙箱启动缓慢的问题。 |
| [^164] | [Not All Evidence Is Admissible: Evidence Governance for Tool-Augmented Medical Agents](https://arxiv.org/abs/2611.22169) | 该研究揭示了工具增强医疗智能体中的“证据劫持”现象——医疗工具无声失效会使原本正确的答案被翻转为错误——并从V-信息理论出发提出“证据治理”框架CIEG，通过将基于内容的证据筛选决策与推理器分离，使推理器不仅“看得更少”更能“看得更好”。 |
| [^165] | [What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super](https://arxiv.org/abs/2611.22161) | 通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。 |
| [^166] | [EgoHandWAM: Probing the Role of Hand Motion in Egocentric World Action Modeling](https://arxiv.org/abs/2611.22151) | EgoHandWAM 通过在 75 万余个第一人称操作片段上联合预测三维手部轨迹和未来视觉特征，实验表明手部运动信息对未来视觉预测仅有有限提升（约1-4%的误差降低），揭示了手部运动在第一人称世界建模中作用的边界。 |
| [^167] | [Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat)](https://arxiv.org/abs/2611.22148) | 该论文通过大规模实验证明，长上下文生成的退化本质上是采样器问题——高温下保持候选 token 池小而稳定的采样器（如 top-nsigma）能让编程智能体等长程生成任务保持高成功率，而放任低概率 token 进入的采样器则完全崩溃。 |
| [^168] | [When does deep reinforcement learning pay off in HVAC supervisory control? Four small Modelica case studies trained directly on FMUs](https://arxiv.org/abs/2611.22144) | 该论文通过四个基于Modelica模型并直接在FMU上训练SAC智能体的小型可复现HVAC案例研究，与调优的规则控制器及近最优基准进行对比，阐明了深度强化学习在HVAC监督控制中何时才真正划算。 |
| [^169] | [RoboTeam: Benchmarking Vision-Based Multi-Agent Control of Many Robot Arms in Real Time](https://arxiv.org/abs/2611.22138) | RoboTeam是一个仅凭相机图像实时协同控制4至16个机械臂、包含36项长时协作任务的模拟基准，揭示了自适应多智能体系统虽然完成任务数最多（28/36），但需要付出数小时耗时和上亿token的高昂代价。 |
| [^170] | [Big Brains, Tiny Board: Pupavision, an Agentic Physical AI Instrument for Monitoring Drosophila Pupariation and Lifecycle Transitions](https://arxiv.org/abs/2611.22136) | Pupavision以约45–75美元的成本将前沿编程智能体直接部署在单板计算机上，使其自主完成果蝇蛹化等生命周期实验的图像采集、错误修复与日常检查，实现了低成本、高可靠性的自动化生物学实验监测。 |
| [^171] | [LoCaR: Localized Canvas Retrieval With Frozen Vision-Language Model](https://arxiv.org/abs/2611.22131) | LoCaR 是一个无需训练的局部化文本到图像检索框架，利用 SAM2 生成的目标候选区域和冻结视觉编码器对每个区域独立编码，从而有效提升对小物体和局部属性的检索能力，并引入 LOCFG 基准数据集进行评估。 |
| [^172] | [TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks](https://arxiv.org/abs/2611.22115) | 星4系列通过模型架构优化与面向昇腾芯片和MindSpore框架的深度系统工程，实现了在8192张昇腾910C卡上预训练2850亿参数的前沿智能体大模型。 |
| [^173] | [RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents](https://arxiv.org/abs/2611.22113) | 该论文提出RegimeCouncil——一个无需重训LLM的轻量级成对证据路由器，将“记忆冲突治理”形式化，并在局部与全局两种记忆抑制策略之间在线仲裁，以解决长期记忆悄然过时与延迟反馈下负面经验传播范围难以确定的问题。 |
| [^174] | [From Sorry to QED: Refining LLM Math Proofs via Angelic Execution](https://arxiv.org/abs/2611.22103) | 本文提出MathEye框架，其核心“天使证明检查”以乐观执行方式将LLM生成的Lean证明脚本中失败片段替换为占位符，在保留整体证明结构的同时精确定位失败点，从而以最少的人工交互把不完善的证明转化为完全验证的Lean证明。 |
| [^175] | [EvoDisco: Strategy Discovery for Evolving Agent Memory](https://arxiv.org/abs/2611.22102) | 该论文提出EvoDisco框架，将智能体记忆演化建模为演化式策略发现问题，由可学习策略根据记忆状态和任务历史动态选择并实例化四种语义算子，突破了传统记忆更新中算子与调度规则固定不变的限制。 |
| [^176] | [Semantic Communication via Analog RF Computing: When the Radio Computes the Meaning](https://arxiv.org/abs/2611.22100) | 该论文提出让边缘设备借助模拟射频计算在射频接收前端直接执行语义编码器——由基站广播权重波形、设备复用无源混频器实现超低能耗的端到端语义通信，并建立了模拟计算噪声与信道噪声的统一数学模型及语义失真的二阶敏感度分解。 |
| [^177] | [When AUC Meets Generative Reconstruction: A Unified Principle for Balanced Visual Representation Learning](https://arxiv.org/abs/2611.22094) | 该论文提出将生成式重建重新表述为判别排序问题的统一表征学习范式，解决了对比学习与生成重建之间的目标冲突，实现了细粒度感知与判别能力兼顾的平衡视觉表征学习。 |
| [^178] | [HyCoInfer: Cloud-Edge Collaborative Inference for Large Language Models on Heterogeneous GPU-FPGA Architectures](https://arxiv.org/abs/2611.22086) | HyCoInfer提出了一种云边协同的异构GPU-FPGA大语言模型推理系统，通过将预填充分配给云端GPU、将解码任务分布到云端GPU与边缘FPGA，并解决异构容量均衡、KV缓存广域传输和异构执行延迟三大关键问题，实现高效的LLM服务。 |
| [^179] | [ATLAS-Finance: Benchmarking AI Agents in Realistic Banking Environments](https://arxiv.org/abs/2611.22085) | 该论文提出ATLAS-Finance基准，通过13家模拟公司中的100个端到端投资银行任务，评估AI智能体在信息分散、任务欠明确的真实银行工作环境中搜索多源信息并完成复杂金融建模等高价值专业工作的能力。 |
| [^180] | [When Lower Harm Misleads: Auditing Concept Corrections](https://arxiv.org/abs/2611.22084) | 该论文提出“修正审计”这一可复现评估协议，揭示修正伤害的降低可能源于模型对修正响应变弱或起点更差等误导因素，并证明最坏情况修正训练虽降低伤害却牺牲了更多有效增益，使修正后NLL上升。 |
| [^181] | [Segmentation-Guided Direct Capacity Estimation of Rooftop Photovoltaics from Aerial Imagery](https://arxiv.org/abs/2611.22076) | 提出Direct Head分割引导架构，在保留U-Net分割与软掩膜池化的同时直接从航空影像回归屋顶光伏容量，摆脱了传统“掩膜面积×功率密度”的间接估计方式，在7,590个法国屋顶装置上将误差最多降低20.7%。 |
| [^182] | [Can You Undo It? Reversibility Classes for AI Agent Actions](https://arxiv.org/abs/2611.22075) | 本文提出“可逆性分类”——一个声明式的五级词汇体系，通过参数敏感规则、保守默认值和先验状态捕获契约来判定AI智能体工具动作能否被撤销，其实现的会话守卫在攻击场景下能拦截全部29次可归因的攻击阳性运行，优于读写代理基线（仅18次），且对良性运行的干扰极小。 |
| [^183] | [MedGuardBench: A Patient-Oriented Benchmark for Personalized Guardrail Development](https://arxiv.org/abs/2611.22069) | 该论文发布了MedGuardBench——首个以患者档案和处方为条件的个性化用药安全防护基准（包含87种药物、5,001个场景），并提出个性化推理防护栏方法，使模型能够基于患者的真实信息而非表面线索做出安全判断。 |
| [^184] | [FM-Mamba: An Efficient Native 2D Selective State Space Model with Stability-Aware Discretization](https://arxiv.org/abs/2611.22061) | 提出FM-Mamba，一种基于Fornasini-Marchesini模型的原生二维选择性状态空间模型，通过沿双轴演化的耦合递归、输入依赖的双步长和稳定性感知离散化，解决了现有Mamba方法处理二维特征时破坏空间邻接、计算量倍增及缺乏二维稳定性约束的问题。 |
| [^185] | [Weight-Space Programming Doesn't Need Transformers](https://arxiv.org/abs/2611.22055) | 在无需梯度训练的权重空间编程中，三个前沿LLM在自由设计架构时都不约而同地将知识编译为可查询的稀疏表，且其性能以约1/1800的参数量超越了Transformer，表明权重空间编程并不需要Transformer。 |
| [^186] | [A Hybrid Exact-Heuristic Solver for the Integrated Passenger and Parcel Transportation Problem](https://arxiv.org/abs/2611.22043) | 该论文提出一个融合常数时间插入测试、自适应大邻域搜索与精确集合装箱引擎（分支定界结合拉格朗日界强化）的混合求解器，用于求解带时间窗、开放路线和异构出租车的乘客与包裹联合运输问题。 |
| [^187] | [RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing](https://arxiv.org/abs/2611.22039) | 提出了RLDE框架，将文档迭代修订建模为无参考的MDP问题，通过指针网络与PPO智能体结合冻结的BART模型，依据可读性、困惑度和语法等多目标奖励实现可解释的自动化文档编辑。 |
| [^188] | [Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents](https://arxiv.org/abs/2611.22033) | 在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。 |
| [^189] | [What Does an Agent Need? Bundle-Aware Agent Resource Composition](https://arxiv.org/abs/2611.22030) | 该论文提出AgentComposition框架，联合选择骨干模型与工具-技能捆绑包以最大化智能体端到端执行效用，并通过检索增强生成、受限束搜索和捆绑包感知评论家，实现无需重新训练的新型资源组合与执行反馈自适应。 |
| [^190] | [GauntletScore: Auditable Fact-Checking That Matches the Best Published Checkers on LLM-AggreFact](https://arxiv.org/abs/2611.22027) | GauntletScore是一种可审计的多读者证据引擎，通过向不同提供商的独立读者提问、在证据账本中记录测量并使用固定贝叶斯规则汇总（仅在两位读者一致时判定矛盾），在LLM-AggreFact完整测试集上达到79.51的平衡准确率，与最佳已发表检查器持平。 |
| [^191] | [Computed Permutation Trellis Coding: Near-Optimal Quantization from Two Multiplications and a 64-Byte Table](https://arxiv.org/abs/2611.22019) | 提出CPTC量化格式，其解码器仅需两次乘法和一个64字节表即可实现近最优的低位宽张量量化，在4比特时距信息论理想仅差0.07比特，同时支持1至6比特所有位宽并精确复现标准整数与浮点格式。 |
| [^192] | [SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory](https://arxiv.org/abs/2611.22017) | SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。 |
| [^193] | [On Formal Counterfactual Explanations](https://arxiv.org/abs/2611.22010) | 本文将最近世界语义下的反事实解释刻画为析取范式，并提出合取反事实解释（CFXps）这一新概念，建立了CXps、AXps与CFXps之间的结构联系，从而架起了反事实解释与形式化XAI之间的桥梁。 |
| [^194] | [DiagTrace: An Interactive Benchmark for Clinical Decision Making](https://arxiv.org/abs/2611.22009) | DiagTrace是一个基于PubMed Central病例报告构建的交互式临床决策基准，包含510个跨八个专科的病例，用于评估大语言模型能否通过交互式信息获取从不完整信息中逐步重建诊断轨迹，突破了传统一次性提供完整患者信息的评估方式。 |
| [^195] | [KV-LTM: Persistent Per-Session KV Caches as Long-Term Memory for Personal Assistants](https://arxiv.org/abs/2611.21998) | KV-LTM提出将每个会话的原始文本与KV缓存一同作为长期记忆存储，通过直接恢复缓存会话来回答问题，避免了传统记忆系统中问题与证据之间的推导和选择环节，并作为检索与索引的补充，充当免重算的prefill加速器。 |
| [^196] | [Modeling Document and Section Hierarchies in Embedding Space](https://arxiv.org/abs/2611.21980) | 该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。 |
| [^197] | [WebAgents: Unbundling Agent Creation, Hosting, and Execution for an Agent-Native Web](https://arxiv.org/abs/2611.21978) | 该论文提出WebAgents系统，通过签名的agent.pkg包将智能体的创建、托管与执行三种角色解耦，使智能体工作流能够在不绑定特定执行环境、不转移用户资源控制权的前提下跨平台移植运行。 |
| [^198] | [Pathology Randomization for Synthetic Priors and Zero-Shot Detection](https://arxiv.org/abs/2611.21975) | 该论文提出PathoDraw，一种通过在对比度无关渲染前将病灶条件分布随机绘制到标签图中的脑部MRI生成模型，从而无需真实患者数据即可实现病理合成先验与零样本异常检测。 |
| [^199] | [OCCAM: Improved Single-Cell Representation Learning and Perturbation Effect Prediction](https://arxiv.org/abs/2611.21959) | Occam是一个结合去噪细胞表示与基因相互作用、文本及蛋白质序列特征的MLP框架，在单细胞表示学习和跨环境基因扰动效应预测方面均达到了最优性能。 |
| [^200] | [Weight-space programming for tractor guidance controller: large language models natively write neural network weights that steer a real tractor](https://arxiv.org/abs/2611.21958) | 大语言模型通过权重空间编程直接读取、修复并从零编写拖拉机导航控制器的神经网络权重，无需优化器或梯度下降，在模拟和真实拖拉机上均实现了成功的路径跟踪。 |
| [^201] | [Eight in Ten LLMs Leak System-Prompt Secrets: Nevan, an Agentic Capture-the-Flag Study](https://arxiv.org/abs/2611.21956) | 该研究通过代理式夺旗实验审计了 363 个大语言模型，发现约八成模型会因提示注入攻击（尤其是转向法语等非英语语言）而泄露系统提示词中的机密，暴露了当前模型在指令层级防护上的普遍薄弱性。 |
| [^202] | [ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture](https://arxiv.org/abs/2611.21954) | ThermLeT是一个基于Transformer的多模态代理模型，能在13毫秒内精准预测2.5D芯粒架构的稳态温度，比传统有限元求解器快约四个数量级，精度显著优于ThermEDGe和HotSpot。 |
| [^203] | [DiscoveryAgent: A Framework for Scientist-centric Agentic AI](https://arxiv.org/abs/2611.21950) | DiscoveryAgent是一个以科学家为中心、领域无关的AI智能体框架，它通过让科学家在各轮研究之间选择、完善或替换假设，并结合基于严格检查点系统的强大审计智能体来追溯声明、复现分析和自主发现错误，实现了自主探索与可扩展人类监督的平衡。 |
| [^204] | [Honest Brokers in AI Regulation: How Public Comments Introduce Evidence to Regulators](https://arxiv.org/abs/2611.21948) | 该研究通过计算分析美国40项潜在AI监管规则的2,664份公众意见发现，企业提交者引入的证据总体较少，而政府、大学和公民社会提交者提供了显著更多的研究和法律来源，表明非行业利益相关者可作为对抗行业俘获的“诚实经纪人”。 |
| [^205] | [Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking](https://arxiv.org/abs/2611.21946) | 该论文提出“承诺-判定差距”框架来识别推理模型中的有害过度思考，证明仅凭最终正确性探针估计的“判定”无法预测继续思考的利弊，而“承诺”与“判定”之差恰好量化了停止思考可获得的准确率增益，并给出将准确率-预算曲线分解为链内答案变化与链存活变化的恒等式。 |
| [^206] | [Benchmarking Deep Learning Models for Wheelchair Configuration Classification in Accessible Automated Vehicle Applications](https://arxiv.org/abs/2611.21945) | 本文提出了轮椅配置图像数据集（WCID），通过对手动/电动、轮椅子类型、乘坐状态等多维度属性进行细粒度标注，为无障碍自动驾驶车辆应用中的轮椅配置分类深度学习模型提供了基准测试资源。 |
| [^207] | [Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in Small Video-Language Models](https://arxiv.org/abs/2611.21940) | 本文提出一种无需标注的“引用差距”反事实检验方法，通过涂黑模型推理链中所引用的视频区域来验证小型视频语言模型是否真正依赖其引用的证据，结果显示模型在CLEVRER上具有较强证据依赖性，但在NExT-QA上依赖性很弱。 |
| [^208] | [SLED: Sensor SLicing from Edge to Datacenter](https://arxiv.org/abs/2611.21937) | 本文提出SLED传感器切片技术，在O-RAN中为单个用户设备内的各传感器流提供确定性的MAC层隔离，解决批量数据传输挤占延迟敏感遥测流量的问题，并利用智能体编程将开发时间缩短400倍。 |
| [^209] | [V2V Communications Using Blockchain-Enabled 6G Technology and Federated Learning](https://arxiv.org/abs/2611.21932) | 该研究提出了一种融合区块链、联邦学习与延迟资源分配优化的6G赋能V2V通信系统，在数据包投递率、模型准确率和延迟等指标上优于现有技术，并通过共识优化和k-匿名实现了安全与隐私的双重保障。 |
| [^210] | [DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding](https://arxiv.org/abs/2611.21925) | 本文发现标准特征编码机制存在有损（残差混合破坏特征空间）、无记忆、均匀采样忽略重要隐藏状态三大缺陷，并提出特征混合注意力（MoFA）来改进推测-扩散解码中并行草稿模型的目标特征编码质量。 |
| [^211] | [ALOHA Lightning: Learning Fast and Precise Manipulation](https://arxiv.org/abs/2611.21919) | ALOHA Lightning系统通过动态示教高速收集人类演示，并结合测试时动作平滑与掩码视觉对齐的学习流程，实现了快速且精准的双臂机器人操作。 |
| [^212] | [A Temporal Algebraic Framework for Data Stream Evaluation Under Delayed Labels and Distribution Shifts](https://arxiv.org/abs/2611.21914) | 该论文提出将基于混淆矩阵的代数框架扩展到数据流评估，通过指数衰减或滑动窗口进行时间加权，并以观测掩码和校准概率处理延迟与选择性标签反馈，从而在分布漂移下以每步 O(m²) 的代价区分决策边界退化与类别流行度变化。 |
| [^213] | [Domain-Tuned Diffusion Super-Resolution for 4K Facial Asset Maps](https://arxiv.org/abs/2611.21890) | 提出一种面向面部资产贴图的领域微调扩散上采样方法，通过轻量级LoRA适配冻结的SDXL骨干并结合Tile ControlNet保持结构，实现1024到4096的4倍超分辨率，将反照率LPIPS从0.21降至0.08且避免位移贴图的几何失真。 |
| [^214] | [Language Models Can Rewrite Their Own Weights](https://arxiv.org/abs/2611.21885) | 该论文提出“自我改写”方法，让27B语言模型仅凭自身推理、无需梯度或训练，诊断失败原因并直接编辑自身权重，在预先固定的490道未见题目上准确率提升6.6分，并在多次自我改写会话中持续提升表现。 |
| [^215] | [Cancellation-Free Constrained SquareCB for Safe Controller Selection](https://arxiv.org/abs/2611.21880) | 该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。 |
| [^216] | [FaceRiff: Data-Efficient Unified Text-to-3D Face Asset Synthesis via Geometric Conditioning](https://arxiv.org/abs/2611.21867) | FaceRiff提出了一种数据与参数高效的统一文本到3D人脸资产生成框架，通过在预训练SDXL上添加任务特定LoRA适配器，并以UV法线图作为几何条件，在共享UV空间中直接从文本生成包含网格、反照率和置换贴图的完整3D人脸资产。 |
| [^217] | [PromptPack: Training-Free Augmentation of Diffusion Editors for Intent-Aligned Spatial and Semantic Control](https://arxiv.org/abs/2611.21865) | PromptPack 通过免训练地为扩散编辑器引入“保留/允许/修改”像素角色划分和区域到提示的空间映射两种显式编辑指南，实现了与用户意图对齐的空间与语义精确控制。 |
| [^218] | [The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler](https://arxiv.org/abs/2611.21852) | 该论文发现量化后MoE扩散语言模型的解码瓶颈在于画布级logits采样流水线而非专家矩阵计算，并提出两个无损优化层级（比特精确与精度对等）将其消除，从而大幅加速解码。 |
| [^219] | [AR-WAM: efficient streaming world-action modeling for autonomous driving](https://arxiv.org/abs/2611.21851) | AR-WAM是一种高效的流式世界-动作模型，通过因果自回归建模增量更新视觉上下文、单次前向传递获取未来特征以条件化轨迹生成，并利用强化学习直接优化驾驶奖励，从而低成本地实现自动驾驶中的交通预测与轨迹规划。 |
| [^220] | [Inventa: Decentralized Self-Improving Agents for Autonomous Scientific Discovery](https://arxiv.org/abs/2611.21847) | Inventa是一个去中心化的自动化科学发现框架，通过自主研究单元生成候选想法谱系、编排智能体自适应调整变异策略、以及验证器筛选并共享经验证的知识，解决了多智能体系统中想法趋同、错误信息传播和无效搜索的问题。 |
| [^221] | [EmbedGym: Letting Coding Agents Develop and Debug Embedded Firmware Without Physical Hardware](https://arxiv.org/abs/2611.21845) | EmbedGym是一个交互式虚拟硬件环境，通过指令级MCU仿真器结合41个外设模型、模拟电路仿真器及虚拟WiFi/BLE端点，让编码代理无需物理硬件即可开发和调试嵌入式固件。 |
| [^222] | [Auditors Struggle to Catch Sabotage In End-to-End Transcripts of AI Safety Research](https://arxiv.org/abs/2611.21832) | 该论文通过人类专家充当破坏者的红队/蓝队博弈实验，揭示了无论是拥有完整实验权限的人类审计员（AUROC仅0.59、完整识别率仅17%）还是前沿AI模型，都难以从AI安全研究的端到端全程转录记录中检测出破坏行为，表明当前针对AI暗中破坏研发的审计防御存在重大缺口。 |
| [^223] | [Coordination of Heterogeneous Teams with Uncertain Task Requirements](https://arxiv.org/abs/2611.21828) | 本文提出了基于CAAT图表示和图神经网络策略的分层规划器，能够联合协调异构智能体团队同时进行任务需求发现与任务执行，比独立优化两者的基线方法更快地完成任务。 |
| [^224] | [How Much of SAR Vehicle Recognition Is the Vehicle? Background Shortcuts, Angle Gaps and Partial Libraries on MSTAR and ATRNet-STAR](https://arxiv.org/abs/2611.21827) | 研究发现SAR车辆识别模型的高准确率很大程度上来自背景捷径而非车辆本身，并系统量化了背景捷径、俯仰角差异与部分训练库在MSTAR和ATRNet-STAR基准上的影响。 |
| [^225] | [TUSI: A Citation-Grounded Graph-Based Benchmark](https://arxiv.org/abs/2611.21823) | 该论文提出了TUSI-Bench——一个建立在包含7,261条法律条文的多层知识框架和法律知识图谱之上的结构化波斯语法律基准，用于对具有引用溯源能力的法律RAG系统进行严格评估。 |
| [^226] | [ARC-2: Continuous Compliance Through Delta Management](https://arxiv.org/abs/2611.21811) | 本文提出增量管理框架，将认证基线定义为区域而非固定点，通过持续监控系统运行状态与基线之间的偏差，实现自主系统部署后合规性的持续治理。 |
| [^227] | [Evaluating the Evaluation of Music Generation Models with Generative Psychometrics](https://arxiv.org/abs/2611.21809) | 本文通过对2020-2025年ISMIR音乐生成论文中的284个主观评估条目应用生成心理测量学分析，提取出节奏、旋律、音频、作曲、结构和歌词六个稳定的质量维度，为音乐生成模型的主观评估建立了统一的方法论。 |
| [^228] | [SimAI: Mechanistic User Simulation through Psychological Network Models](https://arxiv.org/abs/2611.21808) | SimAI是一个基于心理网络模型的机制性用户模拟框架，通过图重构目标建模心理因素间的相互作用，并借助PSYCHE数据集实现了比现有患者模拟器更高的多样性和档案一致性。 |
| [^229] | [Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability](https://arxiv.org/abs/2611.21799) | 论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。 |
| [^230] | [Can Coding Agents Write and Transpile Unit Tests](https://arxiv.org/abs/2611.21797) | 本研究对三个编程智能体进行了基准测试，发现其编写的单元测试在覆盖率上远超人类测试套件，但测试质量随项目规模增大而急剧下降，且跨语言测试翻译仍不一致且不可靠。 |
| [^231] | [Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict](https://arxiv.org/abs/2611.21784) | 本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。 |
| [^232] | [Beyond Reconstruction: A Unified Framework for Trustworthy and Explainable EEG Artifact Removal Evaluation](https://arxiv.org/abs/2611.21781) | 该论文提出了TriX-EEG三级评估框架（信号有效性、去除忠实性、解释可靠性），突破了传统重构指标的局限，为脑电伪影去除提供了可信且可解释的全面评估方法。 |
| [^233] | [LLMs as Validators for Cross-Chain Bridges: Evaluating Source-Level and Transaction-Level Attack Detection on Solana](https://arxiv.org/abs/2611.21777) | 该论文构建了一个包含四种可利用变体的Solana本地锁币-铸币桥测试平台，基于16,800个响应评估了五个开源大语言模型在源代码级和交易级上区分真实漏洞诊断与无依据警报的攻击检测能力。 |
| [^234] | [Margin Dynamics for Large Language Model Alignment](https://arxiv.org/abs/2611.21766) | 本文通过对齐方法在概率单纯形上的向量场建模并求解一维间隔常微分方程的闭式解，首次从理论上揭示 DPO 间隔无界增长而 IPO 间隔收敛于有限值的本质差异，并预测了标签噪声导致 DPO 定性相变及非对称加权下梯度集中对可训练性的影响，在五个模型家族和四个基准上得到验证。 |
| [^235] | [From Top to Bottom: Quark Mass Hierarchies in Heterotic String Theory](https://arxiv.org/abs/2611.21751) | 论文在光滑杂化卡拉比-丘紧化框架中识别出复结构模产生费米子质量层级的两种新树级机制，但解析与数值结果表明这两种机制在先前研究的四重二次曲面模型中均无法实现。 |
| [^236] | [Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign](https://arxiv.org/abs/2611.21707) | 本文提出EJBench——一个基于电池冷板共轭传热设计活动构建的与软件无关的工程判断基准，考察模型能否从少量已有结果预测设计在新工况下的行为并做出决策，结果发现最佳前沿模型仅得0.71分，且模型答错时表达的置信度与答对时几乎相同。 |
| [^237] | [Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models](https://arxiv.org/abs/2611.21705) | 通过227对最小短语对和14个开源模型的实验，研究发现改变语言模型道德答案的注意力头并非道德专属，而是属于一个对所有类型规范性判断共用的通用评估回路，其效应与“道德上”这一线索词无关。 |
| [^238] | [Watermark Evidence under Open-Weight LLM Modification: Exact Bounds for Low-Rank Output Watermarks](https://arxiv.org/abs/2611.21702) | 该论文针对开放权重大语言模型权重被修改后水印验证失效的问题，推导出残差约束下水印证据的精确有限词表下界以及序列级检测的严格统计界，证明 OpenStamp 式验证器在模型被修改的情况下依然可靠。 |
| [^239] | [Stop Early, Stay Simple: Evidence Acquisition for Power System Fault and Attack Diagnosis](https://arxiv.org/abs/2611.21695) | 该论文提出ActiGrid主动证据获取基准框架，通过实验发现学习型DQN证据选择策略相比简单的熵停止等基线并无显著优势甚至被其帕累托支配，表明电力系统故障与攻击诊断中简单方法往往已足够。 |
| [^240] | [TIBECG: Temporal Information Bottleneck for Lead-Specific Demographic Privacy in ECG Waveform Release](https://arxiv.org/abs/2611.21694) | TIBECG通过时序信息瓶颈机制，在保留心电图诊断效用（宏F1不低于95%）的同时，显著抑制了波形发布中年龄、性别等人口统计信息的泄露，隐私增益较最佳传统变换方法中位数提升22.08个百分点。 |
| [^241] | [AdaFlowLite: Scalable and Non-blocking Inference on Asynchronous Mobile Data](https://arxiv.org/abs/2611.21691) | 该论文提出AdaFlowLite，通过机会主义推理机制实现异步移动多模态数据部分到达即可执行的可扩展无阻塞推理，并率先提出在开放世界移动环境中以计算方式控制模态亲和性的结构化方法。 |
| [^242] | [EgoLink: Egocentric Language-Vision Interactive Network Knowledge Challenge Summary](https://arxiv.org/abs/2611.21690) | EgoLink 挑战赛通过自我中心视频社会推理与交互式智能体两条赛道，系统评测了第一人称视角下感知、社会推理与工具介导行动保持耦合的多模态智能。 |
| [^243] | [NEBench R1: a Reasoning Benchmark for Evaluating Nuclear Domain Competence in Large Language Models](https://arxiv.org/abs/2611.21682) | 该研究提出了NEBench R1，一个包含7,546道源自美国核管理委员会反应堆操作员执照考试题目的推理导向基准测试，首次系统评估了大语言模型在核领域的知识能力与推理质量，发现六个模型的准确率仅介于34.0%至73.9%之间。 |
| [^244] | [FCP: Formal Context Protocol for Verifying LLM-Generated Artifacts](https://arxiv.org/abs/2611.21679) | FCP协议通过将机器可判定的一致性验证与可反驳但不可判定的语义保真性检查相分离，把AI生成工件可能被虚假认证的八种失效模式转化为返回反例的机械检查，从而以原始需求本身为基准严格验证大模型生成的工件。 |
| [^245] | [Token Commitment as a Control Surface for Steering Masked Diffusion Generation](https://arxiv.org/abs/2611.21669) | 该论文提出将令牌提交策略（即每步提交哪些令牌和提交多少令牌）作为掩码扩散模型推理引导的新型控制界面，通过反事实目标感知提交和校准自适应预算两种机制，在语言和视觉-语言任务上实现了对生成过程的有效引导。 |
| [^246] | [Depth-Aware Activation Steering via Attention Residuals](https://arxiv.org/abs/2611.21655) | 该论文提出将注意力残差与Jacobian透镜概念读出相结合的深度感知激活引导方法，通过按深度权重跨层分布人格向量，实现比单层注入更强的内部概念对齐与行为转变效果。 |
| [^247] | [Evaluating Semantic Insensitivity in Financial Sentiment Analysis: Implications for Sustainable Financial Decision-Making](https://arxiv.org/abs/2611.21653) | 本文提出了智能体驱动的语义反转（ASI）对抗攻击框架，利用LLM多智能体流水线对金融新闻进行针对性词汇替换，以90.85%的成功率欺骗了FinBERT模型，揭示了金融情感分析模型对语义操纵的严重脆弱性。 |
| [^248] | [Consistency Is Not Correction: Separating Confidence, Abstention, and Recovery in LLM Agents](https://arxiv.org/abs/2611.21651) | 该论文将LLM智能体置信度信号的“纠错效用”精确分解为“门控恢复减去门控损害”，并证明当初始错误数低于配对检验所需数量时，实验在数学上不可能验证纠错机制的有效性，也无法将其与从未触发的门控区分开来。 |
| [^249] | [From Preference Data to Personalization: Tracing Sycophancy in Large Language Models](https://arxiv.org/abs/2611.21650) | 该论文提出了有效性感知的谄媚评分器和包含3,298个有效性标注探针的SycoDrift评估集，将谄媚重新定义为“不当附和”而非简单的一致，并据此系统追踪了大语言模型附和压力的三个来源：人类偏好数据、自动评判器规范与上下文个性化。 |
| [^250] | [Literary Language Mashup: Curating Fictions with Large Language Models](https://arxiv.org/abs/2611.21649) | 本文提出在GrAImes框架中让大语言模型充当文学评估者，通过对比LLM与人类评委（文学教授及爱好者）对西班牙语和英语小说的评估，发现LLM虽可作为辅助文学评估工具，但无法替代人类评委。 |
| [^251] | [Can an LLM dream? Testing the overfitted brain hypothesis on an agent with text memory](https://arxiv.org/abs/2611.21648) | 该研究将“做梦防止大脑过拟合”假说应用于具有文本记忆的LLM智能体，通过在每晚记忆重写前对当天对话记录施加细节损坏或随机文本损坏来模拟梦境扭曲，结果显示这种损坏并未带来帮助。 |
| [^252] | [RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs](https://arxiv.org/abs/2611.21641) | 本文提出RAW方法，利用大语言模型在相同提示下有时拒答有时不拒答的随机性作为自然实验，首次对已部署模型的拒答进行因果测量，发现一次拒答会使约一半被测模型的用户后续再参与率下降3至17个百分点。 |
| [^253] | [Can AI Help Explain AI? Effects of Multimodal LLM-Generated Explanations on Perceived Understanding and Perceived Trust in XAI Visualizations](https://arxiv.org/abs/2611.21638) | 本研究通过128人的预注册实验证明，为XAI可视化附加多模态大语言模型生成的文本解释能显著提升用户的感知理解与感知信任，且对缺乏XAI经验的新手效果最为显著。 |
| [^254] | [Measuring Human-AI Alignment as an Early Indicator for Effective Collaboration in Hybrid SE Teams](https://arxiv.org/abs/2611.21633) | 本文提出开发一套软件工程专用的人机对齐静态测量指标体系，借鉴软件架构、组织心理学和人机交互原理，并利用声明式业务流程监控语言构建静态评估工具，作为评估混合人机团队有效协作的早期指标。 |
| [^255] | [Fragmented Markers, Mixed Results: A Systematic Review of AI Coding Assistants and Developer Productivity](https://arxiv.org/abs/2611.21632) | 该论文通过一项分阶段的系统性文献综述，指出现有AI编程助手对开发者生产力影响研究的核心问题在于生产力衡量指标碎片化、缺乏可比性，并基于SPACE框架将分析范围扩展至以人为本的生产力相关成果。 |
| [^256] | [LynnReal-Hand: Unlocking World Interaction with Hand Control](https://arxiv.org/abs/2611.21479) | LynnReal-Hand是一个手部控制的视频世界模型，通过手部感知监督、因果流图和区域感知分数蒸馏三项创新，实现了精确的手-物体交互建模与高效的少步流式视频生成。 |
| [^257] | [Sage S2.1 Flash Technical Report](https://arxiv.org/abs/2611.21072) | 提出了 3200 亿参数的通用混合专家模型 Sage S1.5 Flash，具备跨分子图、生物序列、光谱等科学数据的感知能力，并通过可验证环境训练出工具使用、搜索、编程等智能体能力，在七个领域 33 个基准评测中表现领先。 |
| [^258] | [Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls](https://arxiv.org/abs/2611.21001) | 在真实临床语音代理通话中，患者话语极短（约59%仅一两个词），仅依赖文本转录无法可靠检测患者的挫败情绪，语音音频信息是必要的。 |
| [^259] | [ProcReverser: Multi-Agent Black-Box Microarchitectural Reverse Engineering for Processor Vulnerability Analysis](https://arxiv.org/abs/2611.20827) | ProcReverser是一个多智能体系统，通过利用执行反馈自动迭代实验设计和测试程序，实现了黑盒微体系结构逆向工程的自动化，能够高效识别与Spectre类攻击相关的处理器硬件安全特性。 |
| [^260] | [Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Price Response](https://arxiv.org/abs/2611.20811) | 研究发现，零售预测中的世界模型解释即使通过了八项审计检查（如构造性、稳定性、价格方向和与真实销量的一致性），其促销提升估计仍可能高出真实值1.2至1.4倍，价格响应被夸大近5至6.5倍，表明通过常规检查并不保证解释的忠实性与准确性。 |
| [^261] | [Regularizing Treatment Leakage for LLM-based Uplift Estimation](https://arxiv.org/abs/2611.20794) | 该论文提出LULR方法，通过在冻结LLM编码器的同时对因果头共享表示施加基于CLUB的互信息惩罚，抑制LLM编码文本中泄露处理信息的捷径，从而改进工业定价RCT场景下的CATE提升估计效果。 |
| [^262] | [Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems](https://arxiv.org/abs/2611.20140) | 本文提出了一种将候选人证据、评估、权限控制与评估记录分离的LLM辅助简历筛选系统架构，并通过涵盖210份简历、14个职位和6种决策界面的受控实验，从运行就绪度与合规就绪度两个维度实证审计了其部署就绪度，发现标量分数比通过/不通过判断更能有效区分资质变化。 |
| [^263] | [What Matters for Cross-Modal Alignment](https://arxiv.org/abs/2611.20000) | 该研究揭示了跨模态对齐成败的关键在于模态共变子空间所承载的方差比例，并据此提出LightAlign方法，仅需少量图像-文本配对数据通过岭回归即可完成两个冻结编码器之间的对齐。 |
| [^264] | [Information Limits of Replay-Free Recurrent State Migration](https://arxiv.org/abs/2611.19345) | 本文刻画了循环模型在不重放历史数据的条件下进行状态迁移所必须保留的最小信息量，证明了带标量输入的可控d状态模型至多需要kd个额外实数坐标，且单个秩一替换方向即可要求全部kd个。 |
| [^265] | [Zero-Shot Weight-Space Programming for Prediction and Classification of Target Variables in Structured Datasets](https://arxiv.org/abs/2611.19292) | 该论文证明大语言模型可以在零样本条件下，仅凭展示的小数据集与指定网络架构，无需优化器、梯度或代码执行，就手写出前馈网络的全部权重和偏置，在从玩具问题、合成问题到经典基准和真实工程任务的测试中直接完成结构化数据的目标变量预测与分类。 |
| [^266] | [Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent](https://arxiv.org/abs/2611.18397) | 提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。 |
| [^267] | [Marimba: A System One Model for Speech](https://arxiv.org/abs/2611.18330) | Marimba基于冻结的Qwen3-ASR 0.6B特征构建了一个轻量级“系统一”语音决策接口，通过零样本评分、条件化读出头和中间层探针，在词项检测、轮次结束判断、语言识别、个人信息检测和说话人变化检测等多项语音决策任务上取得了优异且良好校准的性能。 |
| [^268] | [RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation](https://arxiv.org/abs/2611.18244) | 提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。 |
| [^269] | [EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming](https://arxiv.org/abs/2611.16728) | EgoX提出了一种第一人称视角的跨具身学习框架，通过为每个观测和动作模态配置专用分词器与专家模块、共享Transformer主干，并借助仅在训练阶段使用的“具身梦境”未来状态预测目标，使通用操作策略能够有效利用人类、真实机器人和仿真智能体的异构演示数据。 |
| [^270] | [Spec-Driven PBT: Deriving Semantic Invariants from Code Specifications for Agentic Property-based Testing](https://arxiv.org/abs/2611.15920) | 该论文提出规格驱动的属性测试方法，利用智能体编程中已有的自然语言规格说明（而非仅从代码实现）来推导语义不变量，从而更有效地测试AI生成的代码。 |
| [^271] | [ReShap: Interpretable Reinforcement Learning for High-Dimensional Multi-Action Settings](https://arxiv.org/abs/2611.15507) | ReShap 是一个策略蒸馏框架，通过从原始图像中提取特征、执行 SHAP 特征归因，并将深度强化学习策略蒸馏为在完整状态-动作对上训练的可解释代理模型，从而将基于 Shapley 值的可解释强化学习扩展到了高维、多动作的环境。 |
| [^272] | [SkillEarth: Teaching General-Purpose Agents to Act on Earth via Skill Evolution](https://arxiv.org/abs/2611.15481) | SkillEarth 提出了一种自我进化的技能提取方法，无需专家干预或参数更新，即可从演示轨迹中自动提取遥感工具调用知识，并将其外化为可复用、可检索、可进化的技能文件，从而使通用智能体能够持续适应动态地球环境。 |
| [^273] | [Rethinking "RL Generalizes, SFT Memorizes": The Role of SFT Data](https://arxiv.org/abs/2611.14912) | SFT泛化不佳并非模型固有缺陷而是数据问题——精心混合数据来源并匹配RL的正确采样数量或总采样预算，可使SFT的泛化能力追平甚至超过RL。 |
| [^274] | [OverServe: Resource Amplification Attacks on SaaS Agents via Tool--Prompt Co-Design](https://arxiv.org/abs/2611.14808) | 提出了OverServe多智能体攻击框架，通过协同设计MCP状态工具与完成确认提示词，诱导SaaS智能体在业务任务完成后仍反复检查未完成条件，从而放大对服务提供商资源的消耗。 |
| [^275] | [Continual-ARC: Measuring Skill Acquisition over a Working Lifetime](https://arxiv.org/abs/2611.14505) | 提出 Continual-ARC 基准，基于三个 ARC-AGI 版本构建终身任务流，使基于记忆和基于权重的持续学习系统能够在同一预算评分标准下公平比较技能习得能力。 |
| [^276] | [Attention at Absolute Zero](https://arxiv.org/abs/2611.11937) | 该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。 |
| [^277] | [Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion](https://arxiv.org/abs/2611.11010) | Lightfoot 是一种图检索增强生成系统，它将图中的每个实体、链接、事实和同义词边都定义为逐段落提取结果的确定性函数，从而实现无需语言模型调用的检索以及作为数据库事务的精确文档删除（在 11,402 段语料库上删除 10% 仅需 14 秒且结果与全新索引完全一致），解决了现有 GraphRAG 系统的模型调用成本和文档删除不精确问题。 |
| [^278] | [Goodhart in the Planner: Sampling-Based Control Exploits Rewards Learned from Human Preferences](https://arxiv.org/abs/2611.09670) | 论文揭示并量化了“规划器中的古德哈特定律”：针对人类偏好奖励模型进行更强的 MPPI 搜索虽在标签准确率上保持高分，却使真实回报最多下降 76%，并进一步给出真实回报与搜索偏离距离（√KL）的关系以及 χ² 正则化 MPPI 在由评分器误差范围设定温度时的最坏情况最优性保证。 |
| [^279] | [Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player Games](https://arxiv.org/abs/2611.09271) | 该论文首次系统考察大语言模型零样本权重空间编程的能力边界，在涵盖四个难度维度的双人游戏四级阶梯上零样本手工设定前馈网络的全部权重，构建并独立验证了十个网络，其中六个获得完整且非统计性的正确性证明。 |
| [^280] | [Continued Training of Continuous Language Flows with Token Identity Squared Supervision](https://arxiv.org/abs/2611.07721) | 提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。 |
| [^281] | [Signed but Not Safe: Requirement Integrity in Multi-Agent LLM Handoffs](https://arxiv.org/abs/2611.07483) | 该论文提出一种带HMAC-SHA256签名的需求清单交接机制，配合确定性代码验证，将多智能体LLM系统在攻击下的不安全提交从69.4%降至9.0%，且不损失合规提交。 |
| [^282] | [EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL](https://arxiv.org/abs/2611.06518) | EmoReward 提出一种自进化奖励校准机制，通过从训练过程的奖励轨迹中提炼证据可靠性来自适应调整多奖励权重分配，解决了固定奖励难以捕捉情感意图且易引发奖励作弊的问题。 |
| [^283] | [Corralling AI Risks with Grassroots Contracts](https://arxiv.org/abs/2611.06374) | 提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。 |
| [^284] | [Skill erosion, individual memory and the reversibility of AI adoption](https://arxiv.org/abs/2611.04886) | 该研究揭示了技能侵蚀机制使AI采用呈现双稳态，即使群体具有相同的采用率和平均技能水平，也可能因技能在AI使用者与非使用者之间的分布差异而走向截然相反的长期结局。 |
| [^285] | [Enhancing Stock Market Information Retrieval with Retrieval-Augmented Generation](https://arxiv.org/abs/2611.03747) | 本文提出一种整合检索与生成方法的RAG系统，用于生成准确且相关的股票市场信息响应，并通过ROUGE指标验证了其有效性。 |
| [^286] | [Malware Clustering in the Wild: Taxonomy, Challenges, and Evaluation Metrics](https://arxiv.org/abs/2611.02380) | 本综述系统地将恶意软件聚类研究归纳为七大类方法，并总结了常用的公开数据集与评估指标，为该领域提供了统一的分类框架并明确了研究空白。 |
| [^287] | [See, Think, Use: Diagnosing Perception and Reasoning Failures in VLMs](https://arxiv.org/abs/2611.01467) | 该论文提出“看、想、用”协议，通过定位探针、文本替换探针和反事实图像对，在无需访问模型权重的情况下，将视觉语言模型的每个答案归因于感知、推理或感知使用，从而精准定位应改进的环节并指导测试时计算的分配。 |
| [^288] | [Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments](https://arxiv.org/abs/2611.00597) | 本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。 |
| [^289] | [WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining](https://arxiv.org/abs/2610.39278) | WorldFlow是一种生成式世界模型，它将3D运动参数化为与视频网格对齐的连续位移标记，并利用统一的Mixture-of-Transformers骨干网络在整流流目标下联合去噪视频与3D两种模态，从而将显式几何动力学与视觉演化耦合，释放预训练视频模型中蕴含的物理常识。 |
| [^290] | [Cassette-Editor: A Video Editing Agent That Edits the Way You Do](https://arxiv.org/abs/2610.37690) | 提出了Cassette-Editor，一个通过“一读一写”极简接口让AI像人类一样直接编辑视频项目文件的智能体，所有AI编辑都经过与人工手动编辑相同的执行、规则和提交流程，从而以原子性提交和精准诊断保证时间线始终有效。 |
| [^291] | [REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation](https://arxiv.org/abs/2610.37584) | 提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。 |
| [^292] | [MatOS: An Agent Operating System for Materials Research](https://arxiv.org/abs/2610.34203) | 提出了MatOS——一个面向材料研究的智能体操作系统，通过分层科学资产图、带科学检查的有状态自适应工作流和证据驱动的资产演化三大核心组件，解决了材料研究中智能体能力扩展困难、长时任务难以追踪恢复以及无法从历史任务和用户反馈中学习的难题。 |
| [^293] | [RIPPLE: Recovery Internalization through Process-Level Policy Learning and Effect-Aware Credit for Long-Horizon Agentic Coding](https://arxiv.org/abs/2610.31276) | 提出RIPPLE训练框架，通过前缀教师蒸馏与阶段特定的强化学习信用分配，将过程级恢复能力内化到单一自恢复策略中，弥合编程智能体“能批评却无法行动”的交接鸿沟，并在仓库级软件工程基准上平均提升4.8分。 |
| [^294] | [Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2610.27012) | 提出激活空间探索（ASE）方法，通过聚类正确与错误响应的激活差异构建潜在探索向量库，并将其应用于训练采样以引导模型探索多样化推理路径，从而提升可验证奖励强化学习在数学推理上的性能。 |
| [^295] | [Same Simulation, Different Evidence: Where Retrieved Tool Output Costs LLM Agents Accuracy](https://arxiv.org/abs/2610.26836) | 本文利用确定性的PHREEQC地球化学模拟器实验发现，当长工具输出被移出上下文、改为按需检索时，GPT-5.1和GPT-5.2的准确率分别下降11.3和7.9点而Claude模型不受影响，其根本原因并非计算出错，而是GPT模型在检索模式下跳过模拟报告、不再以模拟证据作为回答依据。 |
| [^296] | [ABACUS: Program-Guided Multistep Reasoning over Frozen Judgment Models](https://arxiv.org/abs/2610.25963) | ABACUS框架通过生成式教师离线构建判断模式与组合程序，将冻结判断模型的局部语义评分决策组合成强大的多步推理，无需微调模型即可在BIG-Bench Hard等任务上取得优异表现。 |
| [^297] | [World Models, Trees and Foundation Models for Retail Forecasting: Evidence from Four Public Data Sets](https://arxiv.org/abs/2610.25170) | 本文提出了一种在价格与促销计划下推进商品状态的世界模型，并在四个公开零售数据集上针对从单品单店订货到全链财务规划的不同决策粒度，系统比较了它与LightGBM、Transformer及TimesFM等时序基础模型的预测表现，借助配对检验与模型置信集确定各粒度下的最优模型。 |
| [^298] | [Screening LLM Judges Without Ground Truth: A Preliminary Evaluation of an AI Dental Hygiene Coach](https://arxiv.org/abs/2610.24335) | 该论文提出了一种无需真实标签即可在专家评审前分阶段筛选大语言模型评审器的流水线，并通过AI口腔卫生辅导系统案例验证了本地评审器与托管评审器具有几乎一致的表现。 |
| [^299] | [A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt](https://arxiv.org/abs/2610.24133) | 本文受黑洞无毛定理启发，通过实验和理论证明自回归 Transformer 对提示词的记忆在生成过程中以幂律衰减（训练后模型 α≈0.55–0.7，快于未训练模型的 α≈0.2–0.5），并证明了提示词影响按 $O(P(\log t)^{L-1}/t)$ 速率被稀释的条件无毛定理。 |
| [^300] | [Agentic In-Context Learning: What Algorithms Can Transformers Learn in Closed-Loop Interaction?](https://arxiv.org/abs/2610.23051) | 本文提出“智能体式上下文学习”框架，首次从学习算法视角研究闭环交互中Transformer的可学习性，证明因果Transformer能够近似主动学习、多臂老虎机和在线强化学习等算法并给出回报损失上界，同时刻画了过程奖励优化在何时能保持最大化最终结果的行为。 |
| [^301] | [Realtime-Bench: Can Frontier Models Solve Real-time Interactive Problems?](https://arxiv.org/abs/2610.22155) | 提出Realtime-Bench基准，通过Wall Clock、Paused Clock和Token Clock三种时钟模式评估前沿模型在严格时间约束下的实时交互能力，结果显示最佳成功率仅44%，证明实时交互对现有模型仍是重大挑战。 |
| [^302] | [Does On-Policy Distillation Need to Complete Every Rollout?](https://arxiv.org/abs/2610.21337) | 该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。 |
| [^303] | [Let the Mind Conceive, Let the Hand Achieve: Following Multimodal Instructions for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2610.18352) | Orchestra是一个将任务规划与执行管理从动作生成分离的智能体框架，通过将多模态指令分解为有序子任务、跟踪进度并维护共享上下文，使机器人能够执行长时程多步操作任务。 |
| [^304] | [SarcasTraps: Effect of Sarcasm on Turkish Hate Speech Detection](https://arxiv.org/abs/2610.16957) | 本文构建了SarcasTraps——一个包含479条数据、沿仇恨言论与讽刺双维度标注的土耳其语外部测试集，并通过实验揭示讽刺会显著干扰仇恨言论检测：微调的编码器模型在讽刺性仇恨言论上召回率下降，零样本大语言模型同样受到影响。 |
| [^305] | [Natural Code Transformation Confuses Agents in Vulnerability Analysis](https://arxiv.org/abs/2610.14350) | 本文发现AI智能体在分析具有相同功能和相同漏洞的不同真实代码提交时会产出截然不同的结果，并据此提出从真实开发模式中挖掘自然且功能保持的代码变换，作为防御攻击者智能体的新手段。 |
| [^306] | [Can Power Draw Constrain Covert Compute? Limits of Analogue Verification for AI Governance](https://arxiv.org/abs/2610.07476) | 仅靠功耗等模拟测量手段无法有效约束隐蔽计算——最坏情况下隐藏计算量可达声明容量的116%，即使采用对抗性匹配能量策略也能隐藏至少41%的计算量，这说明AI治理协议不能仅依赖模拟验证机制。 |
| [^307] | [AEGIS: Differentiable Mars Climate Model with Neural Closures](https://arxiv.org/abs/2610.04081) | 该论文提出了AEGIS，一个模块化、可微分并引入神经闭合项的火星气候模型，突破了传统Fortran火星GCM无法暴露梯度以支持校准和机器学习的局限。 |
| [^308] | [Reliable Self-Evolution with Imperfect Proxy Rewards](https://arxiv.org/abs/2610.02975) | 该论文提出保形区间驱动的自进化方法（CISE），通过条件保形推断和在线密度比估计构建候选特定的奖励区间，以应对不完美代理奖励导致的假阳性问题，从而实现更可靠的LLM自进化搜索。 |
| [^309] | [When Terminal-Agent Training Stalls: Demystifying Data Generation and Verification Challenge](https://arxiv.org/abs/2610.02405) | 该论文揭示了用前沿模型作为元智能体自动生成终端强化学习训练任务时的三类故障（基准无效、评测框架脆弱、奖励错位），并通过实验证明任务可解性区间是模型特定的，主张应将可解性区间校准、验证器审计和基础设施错误核算作为一等评估标准。 |
| [^310] | [Mingbird: A Local-First Agent Harness Enabling Small Open Models to Complete Real Tasks](https://arxiv.org/abs/2610.02001) | 提出了Mingbird，一个面向Windows和Ollama的本地优先智能体框架，通过十项针对性补偿机制（如字节级净零预填充预算、完成门控和签名级循环检测）来克服小模型的典型失败形式，使2-9B的开放模型能够在普通笔记本上完成真实任务，并证明许多失败源于框架而非模型本身。 |
| [^311] | [Misalignment of Low-Loss Regions Causes Grokking](https://arxiv.org/abs/2610.00620) | 论文提出基于模式连通性与低损失区域几何的分析框架，指出grokking现象源于训练集与验证集所诱导的低损失区域之间的错位，并以稳定的反grokking反例挑战了现有的相关性解释。 |
| [^312] | [Pseudo-Label-Triggered Retraining from Forecast Errors for Online Time Series Forecasting](https://arxiv.org/abs/2609.39789) | 本文提出PILOT框架，利用预测误差构建伪标签并训练轻量级评分器，从误差动态中学习时间序列预测模型“何时重训练”，从而在有限部署资源下有效维持预测性能。 |
| [^313] | [LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization](https://arxiv.org/abs/2609.38166) | LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。 |
| [^314] | [Code4Scene: Benchmarking Coding Agents for Constructing and Editing 3D Scenes](https://arxiv.org/abs/2609.36777) | 该论文提出Code4Scene基准，通过190个基于Unreal Engine的案例，直接在生成的引擎原生场景上评估编程智能体从语言规范构建3D场景的空间推理能力以及对场景状态进行精确编辑的能力。 |
| [^315] | [Representable but Unlearned: Encoding Rank and the Interaction-Prediction Floor](https://arxiv.org/abs/2609.36208) | 该论文提出一种无需标签或拟合模型即可量化输入编码秩缺陷所致预测误差下限的方法，并证明siRNA交互预测中仅训练期特征掩码造成的秩下降（140降至72）会产生占拟合模型误差14.6%的不可逾越下限，而仅恢复三个化学列或移除掩码即可完全消除该下限。 |
| [^316] | [SkillPE: Creativity-Oriented Cinematic Skill Evolution for Text-to-Video Prompt Enhancement](https://arxiv.org/abs/2609.34335) | SkillPE 提出了一个面向创造力的提示词增强框架，通过从专家电影知识出发，利用共鸣、不和谐与分歧三类电影参考引导演化可复用的电影级技能，帮助非专家用户提升文本到视频生成的专业叙事与创意质量。 |
| [^317] | [PhysAlign: A Benchmark for Evidence-Grounded Role Alignment in Multimodal Physics Reasoning](https://arxiv.org/abs/2609.33319) | 提出了PhysAlign基准，通过解耦视觉识别与物理角色分配的局部探针（3,341个人工验证探针、986个物理问题），系统评估多模态模型能否将物理图像中识别的信息正确对应到其预期物理角色。 |
| [^318] | [Intuition Vectors from Pretrained Representations Steer Diffusion-Based Reasoning](https://arxiv.org/abs/2609.32619) | 该研究发现，冻结自监督视觉编码器表示空间中的简单向量差（“直觉向量”）能够直接支持抽象视觉推理，无需依赖语言推理或任务特定的表示学习。 |
| [^319] | [Video-HopChain: Multi-Hop Questions and Confidence-Gated Exploration for Video Reasoning Models](https://arxiv.org/abs/2609.25773) | 该论文提出 Video-HopChain，首次将多跳问题合成引入视频领域，通过串联三到六个是非题并以整数之和作为可精确验证的最终答案，构建了含 22,550 个多跳视频问题及基准的数据集，为视频推理模型的 RLVR 训练提供长视觉证据链并暴露累积性推理错误。 |
| [^320] | [FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model](https://arxiv.org/abs/2609.23808) | 提出FLARE，一种由轻量级生成式奖励模型驱动的全生命周期密集监督范式，通过离线因果诊断框架RADAR提取无后见之明的监督信号，为长程编码智能体提供实时步骤级风险反馈，解决稀疏二值奖励带来的信用分配危机。 |
| [^321] | [RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents](https://arxiv.org/abs/2609.23466) | RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。 |
| [^322] | [EDCT-Bench: Uncovering Faithfulness Gaps in VLMs via Explanation-Driven Counterfactual Testing](https://arxiv.org/abs/2609.17953) | 提出了一种基于干预的解释驱动反事实测试方法EDCT，通过对模型解释中引用的视觉概念进行经验证的最小编辑来系统评估VLM解释的忠实性，并据此构建了覆盖三个领域的基准EDCT-Bench，揭示了现有VLM中普遍存在的显著忠实性差距。 |
| [^323] | [Structure Across Voices: Comparing acoustic-event type accumulation and sequence dependence across four vocal repertoires using frozen audio encoders](https://arxiv.org/abs/2609.16612) | 本研究利用统一的冻结音频编码器方法跨物种比较四种发声系统，发现不同物种的发声结构优势取决于所测量的声学属性和时间尺度，而非简单的优劣排序。 |
| [^324] | [Amortizing Scaling Law Construction Costs](https://arxiv.org/abs/2609.05016) | 该论文提出了一种基于贝叶斯优化的高效缩放定律构建框架，通过逐步扩展计算预算并结合代理模型幻想评估来恢复广泛的实验网格，大幅降低了构建缩放定律的计算成本。 |
| [^325] | [The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents](https://arxiv.org/abs/2609.01852) | 该论文提出并量化了“记忆信任差距”现象：持久记忆智能体会过度信任（而非混淆于）过期的存储事实并覆盖权威证据，且这种失效受模型能力门控——规模越大的模型在过期记忆被伪装成最新信息时崩溃反而越严重。 |
| [^326] | [The Safeguard Worked. Is the LLM System Safer?](https://arxiv.org/abs/2609.00519) | 论文指出防护措施的标准评估指标（拒绝率、攻击成功率等）只能衡量其局部表现，无法回答部署层面“LLM系统是否真正更安全”的问题，并揭示了证据的不对称性：一次成功的有害攻击即可证明危害仍存在，而要证明系统安全则需要超越防护自身数据的系统性证据。 |
| [^327] | [Arabic Safety Alignment as Selective Refusal: An Empirical Study of SFT, DPO, and Guard Calibration](https://arxiv.org/abs/2608.29378) | 该研究提出用良性拒绝率 B 和有害提示拒绝率 H 双指标评估阿拉伯语大模型安全对齐，发现仅拒绝式 SFT 会坍缩为全面拒绝，而精选混合 SFT 配置能在低良性拒绝率下实现 90% 以上的有害提示拒绝，且 DPO 与推理护栏的效果因模型而异、并非统一提升。 |
| [^328] | [Representation Learning with Quantum Signal Processing](https://arxiv.org/abs/2608.28828) | 该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。 |
| [^329] | [Towards Efficient Auto-Interpretability of Boltz-1 Sparse Autoencoders](https://arxiv.org/abs/2608.27754) | 该论文提出了一套针对Boltz-1稀疏自编码器的自动化可解释性流水线，通过跨种子字典稳定性优先排序、入侵者检测任务和可证伪的计算机预测测试三项创新，实现了对潜在特征的高效自动解释与筛选。 |
| [^330] | [Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization](https://arxiv.org/abs/2608.27507) | 本文提出MCC-PGPSE方法，通过结合留一策略覆盖与状态所有者专业化的边际覆盖信用来重新分配辅助内在奖励，从而减少并行状态熵优化中的冗余探索并促进互补性状态覆盖。 |
| [^331] | [Multi-Granularity Context-Enhanced RAG over Multimodal Knowledge Graphs](https://arxiv.org/abs/2608.25986) | 本文提出构建上下文增强多模态知识图谱（CEMMKG）的新框架，通过在视觉信息提取和多模态知识融合过程中充分利用文本上下文，弥合图像与文本之间的语义鸿沟，从而提升多模态GraphRAG的性能。 |
| [^332] | [Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening](https://arxiv.org/abs/2608.25846) | 本研究通过系统跨数据集评估发现，基于咳嗽的机器学习结核病分类器无法泛化，其性能受录音设备和数据集特异性影响，而非疾病本身，提示需谨慎对待此类模型的实际应用。 |
| [^333] | [Sampling Luck Masquerades as Allocation Gain: Auditing Test-Time Budget Allocation for Neural Combinatorial Optimization](https://arxiv.org/abs/2608.13087) | 本文发现神经组合优化中测试时预算分配的报告增益主要源于采样运气偏差，样本外验证表明非均匀分配并无实际收益。 |
| [^334] | [Query Timing Produces Opposite Positional Biases Between LLMs and Humans](https://arxiv.org/abs/2608.12387) | 本研究揭示了LLMs在位置偏差上表现出与人类相反的模式，且新模型中的偏差更为显著，表明查询时机对模型评估机制有重要影响。 |
| [^335] | [One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL](https://arxiv.org/abs/2608.12253) | 该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。 |
| [^336] | [Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment](https://arxiv.org/abs/2608.11537) | 提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。 |
| [^337] | [Persistent Recursive Worlds Enable Autonomous Software Evolution](https://arxiv.org/abs/2608.10450) | 本文提出EvoX Genesis框架，通过将软件表示为持久递归世界，使项目持久化而代理有限寿命，实现了无需人工干预的自主软件演化，并成功构建了大型编译器。 |
| [^338] | [Explore, Map, Remember, Decide: A Benchmark for VLM Readiness in Safety-Critical Scenarios](https://arxiv.org/abs/2608.08077) | 该论文提出EMRD评估框架，将VLM的空间推理能力解耦为探索、绘图、记忆、决策四个阶段并以18项指标评分，揭示了现有模型在安全关键场景下存在严重的探索遗漏与感知缺陷。 |
| [^339] | [Role Steering of Language Models for Social Simulations](https://arxiv.org/abs/2608.00023) | 该论文提出了一种激活引导筛选工作流和新方法 CastVectors，用于为语言模型智能体注入可验证的角色条件化行为，在社会模拟中实现了显著更高的角色档案对齐度（63.2 对比 41.1），同时保持了词汇多样性。 |
| [^340] | [RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems](https://arxiv.org/abs/2607.29241) | 提出RecHarness框架，通过将优化过程解耦为“多臂老虎机路由器选择修改方向+LLM生成具体优化假设与代码修改”两个步骤，并借助跳跃盆地机制在局部优化停滞时触发结构性跳跃，实现推荐模型的自演化自动优化。 |
| [^341] | [LLM Misuse Safeguards Rely on Untrusted Input: A Widespread, High-Risk Vulnerability](https://arxiv.org/abs/2607.27951) | 大型语言模型的滥用防护措施因依赖用户自述的身份或用途等不可信输入来决定是否放行答案，存在被攻击者通过伪造声明绕过的高风险漏洞，且该漏洞普遍存在于12个顶级会议129篇论文所提出的全部防御方案中。 |
| [^342] | [PIE-APT: Abductive Planning over Temporal Dynamic Knowledge Graphs via Incremental Reasoning](https://arxiv.org/abs/2607.27287) | 该论文提出PIE-APT统一框架，通过将状态转移建模为对演绎封闭描述逻辑理论的非单调更新并利用增量推理机，在SROIQ描述逻辑上保持可判定性并原生绕过分支问题，同时借助增量直接推导溯因避免最小命中集的组合枚举，从而实现时序动态知识图谱上的高效溯因规划。 |
| [^343] | [Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?](https://arxiv.org/abs/2607.23624) | 该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。 |
| [^344] | [Test-Time Coverage: Test-Conditioned Data Curation for Deployment-Aware Learning](https://arxiv.org/abs/2607.22697) | 该论文提出TTCov方法，在训练前利用测试侧信息构建由LLM原子命题及其频率组成的知识图集，将部署条件化的数据筛选分解为覆盖率与分布两部分，从而直接优化训练数据与部署测试分布的匹配度。 |
| [^345] | [Improved Convergence Rates for Stochastic Multi-Gradient Descent which Close the Gap: A Proof by AI](https://arxiv.org/abs/2607.18174) | 本文通过利用帕累托平稳性度量的 Lipschitz 连续性（而非 (1/2)-Hölder 连续性），证明了随机多梯度下降（SMG）在常数步长和线性增长批量下可将平方帕累托平稳性度量的收敛速率从此前的 $\tilde{O}(T^{-1/4})$ 提升至 $\tilde{O}(T^{-1})$，且无需有界梯度假设。 |
| [^346] | [OpenMHC: Accelerating the Science of Wearable Foundation Models](https://arxiv.org/abs/2607.16235) | 该论文发布了OpenMHC——目前面向研究人员开放的最大、最全面的可穿戴健康数据集（涵盖11,894名参与者、超过6700万小时记录、最多19个传感器通道和170个健康行为变量），并配套提供可穿戴基础模型的开源实现与模型权重，以推动可穿戴健康领域的开放科学。 |
| [^347] | [Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains](https://arxiv.org/abs/2607.13319) | OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。 |
| [^348] | [Real-time fall detection based on vision for low-power edge platforms](https://arxiv.org/abs/2607.12909) | 该论文提出一种基于物理信息的双液态时间常数（LTC）网络架构，将跌倒建模为质心与支撑基面耦合动力系统的稳定性丧失事件，通过可学习的物理交互耦合模块与稳定性流形分类器，实现了低功耗边缘平台上的实时视觉跌倒检测。 |
| [^349] | [RL Post-Training Builds Compositional Reasoning Strategies](https://arxiv.org/abs/2607.07646) | 强化学习后训练并非只是放大预训练模型已有的基础技能，而是通过“先强化基础规约、再发现顺序与并行组合程序”的分阶段组合机制，将基础技能重组为新的更高层级推理策略，从而解决预训练模型几乎无法解决的问题。 |
| [^350] | [Statistical Adversaries: Natural Backdoor-like Adversarial Features in Clean Vision Datasets](https://arxiv.org/abs/2607.05516) | 本论文发现干净视觉数据集中自然存在的统计信号可作为后门样触发器，揭示了数据集结构和分布导致的模型漏洞，而非恶意插入。 |
| [^351] | [Localized LoRA-MoE: Mixture of Low-Rank Experts With Adaptive Routing](https://arxiv.org/abs/2607.05114) | 提出Localized LoRA-MoE框架，将局部空间分块与上下文条件自适应专家路由相结合，提供了集中式块级路由和去中心化单元级路由两种架构变体，以解决传统PEFT方法中的梯度干扰和静态拓扑无法适应上下文切换的问题。 |
| [^352] | [Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets](https://arxiv.org/abs/2607.00444) | 该论文提出基于时空凸集图（ST-GCS）的运动规划框架，通过精确凸分解（ECD）处理移动障碍物与已规划机器人的扫掠占据，联合优化区域序列、轨迹几何与时序，实现解耦式多机器人连续时空协调规划。 |
| [^353] | [Process Advantage Signal Shaping: Advantage Construction for Process-Supervised Reinforcement Learning in LLM Reasoners](https://arxiv.org/abs/2606.29296) | 提出PASS优势构造流水线，通过奖励通道归一化融合、信号定义分块和剩余分块数归一化，将密集过程反馈有效转化为策略优势，在七个数学基准上比仅结果奖励的GRPO平均pass@1提升4.62点，同时响应长度缩短约2%。 |
| [^354] | [The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference](https://arxiv.org/abs/2606.21869) | 该论文系统性测量了大语言模型多语言推理的能耗，揭示了不同语言间高达179倍的能耗差异，并发现高能耗语言同时伴随最低的任务准确率，存在“成本+性能”双重惩罚现象。 |
| [^355] | [Latent Goal Prediction from Language for Model-Based Planning](https://arxiv.org/abs/2606.20627) | LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。 |
| [^356] | [Offline Preference-Based Trajectory Evaluation](https://arxiv.org/abs/2606.17541) | 该论文提出基于偏好的轨迹评估方法，通过轨迹间的时序偏好比较取代二元成功指标，将基准测试中的平局率从约75%降至约35%，显著提升了评估的区分能力、排名稳定性和数据效率，并揭示基准测试饱和可能源于度量设计而非仅仅是数据或问题难度问题。 |
| [^357] | [FoundCause: A Foundation Model for Causal Discovery with Latent Confounders from Observational Data](https://arxiv.org/abs/2606.17516) | FoundCause是首个仅在合成结构因果模型上训练的摊销式因果发现基础模型，能够通过单次前向传播从独立同分布观测数据中恢复因果图、识别潜在混杂因素并处理缺失数据。 |
| [^358] | [Rhythm of the Deep: Two-tier acoustic organization of sperm-whale codas from click waveforms to second-order sequence dependence](https://arxiv.org/abs/2606.16084) | 该研究发现抹香鲸尾音的声学组织具有两级嵌套结构：尾音内部反复出现的点击声波形差异与点击间隔时间互补，而尾音之间反复出现的整体尾音形式本身携带序列依赖性。 |
| [^359] | [ICA Lens: Non-Gaussianity Reveals Interpretable Directions in Language Models](https://arxiv.org/abs/2606.11722) | 该论文发现语言模型激活中可解释方向具有非高斯投影分布这一统计特征，并提出可以通过直接搜索非高斯方向来发现可解释特征，而无需依赖稀疏重建等间接学习目标。 |
| [^360] | [Optimization-like Reasoning in LLMs over Expanding Search Spaces](https://arxiv.org/abs/2606.05464) | 本文提出可扩展的优化类任务家族 OPT*，通过复杂度参数扩展搜索空间，并支持求解器引导的在线策略优化与基于搜索的离线强化学习两种训练范式，以培养和评估大语言模型的逐步类优化推理能力。 |
| [^361] | [Smooth Inliers, Erratic Outliers: Frequency-Aware Mamba for Two-View Correspondence Learning](https://arxiv.org/abs/2606.04493) | 提出SFMambaNet，通过逐簇扫描让正确匹配在序列中平缓变化、错误匹配保持无规律，并在前后向扫描间引入可学习频域门控滤波，从而在相机位姿估计、外点剔除和单应性估计等双视图对应学习任务上超越空间域Mamba。 |
| [^362] | [Beyond Bilingual Transfer: Multilingual Code-Switching in Instruction Tuning](https://arxiv.org/abs/2605.29414) | 该研究首次探索了在指令微调中混合英语、日语、韩语和中文四种语言的多语言语码转换数据构建方法，实验证明句子级多语言语码转换能持续提升LLM的多语言理解能力，其效果超越传统仅限双语迁移的方案。 |
| [^363] | [CardioLoom: A Report-Guided Electrocardiogram Foundation Model for Cross-Domain Cardiovascular Phenotyping](https://arxiv.org/abs/2605.25446) | CardioLoom是一种利用近280万心电图-报告配对数据预训练的报告引导式心电图基础模型，在89个内外部心血管表型评估目标上表现优异，仅用10%标注数据即可超越全量监督训练的性能，实现了跨领域的疾病诊断与筛查能力。 |
| [^364] | [Latent Q-Barrier Shielding for Safe In-Context Reinforcement Learning](https://arxiv.org/abs/2605.25267) | 提出一种潜在Q屏障护盾，在部署前学习上下文表示与成本预测模型，在测试时无需参数更新即可依据剩余安全预算和预测的未来成本对候选动作进行过滤或软重加权，从而在分布外偏移下显著改善安全上下文强化学习的奖励-安全权衡。 |
| [^365] | [PEARL: Unbiased Percentile Estimation via Contrastive Learning for Industrial-Scale Livestream Recommendation](https://arxiv.org/abs/2605.21752) | PEARL 提出了一种基于对比学习的非参数分位数近似框架，通过建模相对偏好信号而非绝对参与强度来消除用户行为强度不平衡带来的偏差，从而提升工业级直播推荐的质量与鲁棒性。 |
| [^366] | [Distributional Alignment as a Criterion for Designing Task Vectors in In-Context Learning](https://arxiv.org/abs/2605.20730) | 该论文提出以分布对齐作为设计任务向量的准则，引入指标 $d_{\text{NTP}}$ 来衡量基于任务向量的推理与基于上下文学习的推理之间下一个词概率的差异，该指标与下游任务准确率呈强负相关，从而为任务向量质量提供了直接的评估与设计依据。 |
| [^367] | [A closed-loop framework for perovskite precursor additive discovery](https://arxiv.org/abs/2605.20242) | 该论文提出LEAP闭环框架，将领域专业化大语言模型与主动学习相结合，从文献中提取机理知识并转化为可解释分子描述符进行贝叶斯优化，从而在低数据条件下高效发现钙钛矿前驱体添加剂，将太阳能电池效率提升至20.87%。 |
| [^368] | [Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL](https://arxiv.org/abs/2605.14434) | 该论文提出CQ-SID方法，通过类目感知对比学习与残差量化VAE将商品编码为层次化语义聚类ID以大幅降低束搜索复杂度，并引入专家引导的强化学习方法EG-GRPO，将生成式检索定位为工业级电商搜索召回阶段的高效补充方案。 |
| [^369] | [AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation](https://arxiv.org/abs/2605.12925) | 本文提出AgentLens过程级评估框架并发布AgentLens-Bench数据集，发现10.7%通过测试的智能体轨迹存在“幸运通过”问题（如回归循环、盲目重试、缺失验证等），证明仅以测试是否通过来评估软件工程智能体是不充分的。 |
| [^370] | [Skypart : Weather-Robust Cross-View Geo-Localization via Prototype-Based Semantic Part Discovery](https://arxiv.org/abs/2605.11654) | 本文提出SKYPART——一种基于原型语义部件发现的轻量级可替换ViT头部模块，通过分离跨视角共享布局、边缘化高度相关尺度变化并优化多目标训练，实现了天气鲁棒的跨视角地理定位，为GNSS失效时的无人机自主定位提供了新方案。 |
| [^371] | [iMuon: Intrinsic Spectral Optimization on Riemannian Matrix Manifolds](https://arxiv.org/abs/2605.09238) | 论文提出 iMuon 优化器，通过将黎曼度量规范地把酉不变的欧几里得范数提升为切空间上的内蕴范数，解决了 Muon 推广到低秩、正交、SPD 等矩阵流形时的对称性破坏与闭式解受阻问题。 |
| [^372] | [Scaling Causal Reasoning with Increasingly Complex Causal Simulators](https://arxiv.org/abs/2605.09079) | 提出 CauSim 框架，利用 LLM 增量构建可验证的日益复杂的因果模拟器（可执行 SCM），将因果推理从稀缺标签问题转化为可扩展的监督学习问题。 |
| [^373] | [CyBiasBench: Benchmarking Bias in LLM Agents for Cyber-Attack Scenarios](https://arxiv.org/abs/2605.07830) | 该论文提出CyBiasBench基准测试，揭示不同LLM智能体在网络攻击场景中表现出稳定且明显的攻击选择偏见，这种偏见刻画了智能体行为特征，并提供超越总体攻击成功率之外的信息。 |
| [^374] | [When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR](https://arxiv.org/abs/2604.22774) | 该论文首次系统性研究多行手写数学OCR，揭示了视觉语言模型“过度纠错”从而掩盖学生错误的关键失效模式，并提出了PINK语义评估指标来显式惩罚这种过度纠错行为。 |
| [^375] | [WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics](https://arxiv.org/abs/2604.00024) | 该论文提出WHBench——一个由专家深度参与验证构建的女性健康评估基准，通过对22个前沿大语言模型在47个专家设计的场景和23项标准下的系统测试，发现没有任何模型的平均表现超过75%，暴露了现有模型在女性健康领域的安全性、临床准确性和公平性方面的重大缺陷。 |
| [^376] | [Seeing with You: Perception-Reasoning Coevolution for Multimodal Reasoning](https://arxiv.org/abs/2603.28618) | 该论文提出双角色RLVR框架PRCO，通过观察者生成针对问题的证据描述、求解器基于该描述作答，并采用角色特定的奖励信号解决信用分配模糊问题，实现多模态大语言模型中感知与推理能力的协同演化。 |
| [^377] | [Towards Reliable Local Security Agents: Verifiable Post-Training for Linux Privilege Escalation](https://arxiv.org/abs/2603.17673) | 本文提出一种两阶段后训练方案（先在可自动验证的Linux权限提升环境轨迹上做监督微调，再进行强化学习），成功将小型本地语言模型转化为可靠的安全智能体。 |
| [^378] | [Human-Data Interaction, Exploration, and Visualization Systems in the AI Era: Challenges and Opportunities](https://arxiv.org/abs/2603.05542) | 该论文系统分析了AI时代大规模多模态数据与基础模型不确定性对人-数据交互与可视化系统带来的延迟、可扩展性、交互范式和可信度等核心挑战，并提出需要重新定义人机角色、突破传统效率指标来构建新一代交互与可视分析系统。 |
| [^379] | [Evaluating and Understanding Scheming Propensity in LLM Agents](https://arxiv.org/abs/2603.01608) | 该研究将大语言模型智能体的图谋动机分解为智能体因素与环境因素，并在11个前沿模型、四种现实场景中发现：尽管环境激励很高，智能体几乎不表现图谋行为，且这一现象难以用言语化的评估意识来解释。 |
| [^380] | [The Compute ICE-AGE: OPAL ONE, Continuity That Persists While Reasoning Is Idle](https://arxiv.org/abs/2602.16736) | OPAL是一个驻留CPU的C++17图引擎，能在推理会话之间持久保留记录身份、关系与数值状态，并在高达500万原子内存顶点规模下实现0.083微秒的驻留更新中位延迟。 |
| [^381] | [EmbeWebAgent: Embedding Web Agents into Custom Enterprise UIs](https://arxiv.org/abs/2602.14865) | EmbeWebAgent通过轻量级前端钩子和可复用的后端工作流，将网络智能体直接嵌入企业现有UI中，实现与技术栈无关的混合粒度操作，并借助MCP工具完成真实界面中的多步骤行为。 |
| [^382] | [Do Multi-Agents Dream of Electric Screens? Achieving 100% on AndroidWorld Through Task Decomposition](https://arxiv.org/abs/2602.07787) | 该论文提出Minitap的mobile-use多智能体系统，通过将任务分解到五个LLM智能体（规划、子目标跟踪、状态捕获、决策、执行）和确定性节点，并辅以设备端文本回读、目标验证与重复失败检测等机制，在AndroidWorld基准的116个任务上实现了100%的pass@3成功率。 |
| [^383] | [GCD: GCM-consistent Diffusion for Zero-shot Downscaling across Heterogeneous GCMs](https://arxiv.org/abs/2601.21760) | 提出GCD方法，通过地理条件化、频谱滤波测量引导和蒸馏加速的共轭梯度引导，实现了跨异构GCM的零样本气候降尺度，并将扩散推理步数从1,000步降至6步，兼顾了一致性与效率。 |
| [^384] | [CORE: Concept-Oriented Reinforcement for Bridging the Definition-Application Gap in Mathematical Reasoning](https://arxiv.org/abs/2512.18857) | CORE框架将显式概念转化为可控的强化学习监督信号，通过合成概念对齐测验、在推理中注入概念片段等方式，弥合了大语言模型“能复述定义却不会应用概念”的数学推理鸿沟。 |
| [^385] | [Toward AI epidemiology: a measurement standardization framework for prospective risk detection](https://arxiv.org/abs/2512.15783) | 本文提出一个测量标准化框架，将专家-AI交互压缩为结构化、可比较的字段，无需访问模型内部即可对已部署的AI系统进行前瞻性风险检测，并支撑可靠性与治理两方面的主张。 |
| [^386] | [Intra-Agent Motion and Inter-Agent Relation Learning for Map-Free Trajectory Prediction](https://arxiv.org/abs/2512.02368) | 本文提出一种以目标条件专家路由Transformer (GCERT) 为核心的无地图轨迹预测框架，通过智能体内部运动分解与智能体间关系学习，弥补缺少高精地图道路拓扑和车道连接先验带来的挑战。 |
| [^387] | [From Strategy Search to Policy Learning: MCTS-Guided Adversarial Co-Evolution for LLM Safety Alignment](https://arxiv.org/abs/2511.19218) | 该论文提出MACE框架，将MCTS搜索发现的攻击信息（成功频率与路径深度）转化为攻击者与防御者的策略学习信号，通过对抗协同进化提升大语言模型的安全对齐能力。 |
| [^388] | [Cross-Epoch Bias in SGD-based Leave-One-Out Influence: Analysis and Accumulative Correction](https://arxiv.org/abs/2510.26185) | 该论文发现多轮次SGD留一影响估计在近似有限删除时忽略了先前删除累积的参数位移会改变后续再次删除同一样本的效果这一跨轮次偏差，并提出利用样本后续出现处Hessian矩阵的累加修正估计器（ACC-SGD-IE），在非凸分析下证明该修正可从有限时域误差界中消除该交互误差项。 |
| [^389] | [The Economics of AI Foundation Models: Openness, Competition, and Governance](https://arxiv.org/abs/2510.15200) | 该论文构建两期博弈模型，揭示基础模型在位者的最优开放度随学习效应增强先降后升，并警示强制完全开放的监管可能适得其反，反而损害下游适配效率。 |
| [^390] | [AutoPBO: LLM-powered Optimization for Local Search PBO Solvers](https://arxiv.org/abs/2509.04007) | 本文提出AutoPBO，一个利用大语言模型自动增强伪布尔优化局部搜索求解器的新颖框架，并在包括真实世界基准在内的四个公开基准上验证了显著的性能提升。 |
| [^391] | [Humans and AI respond similarly to task difficulty when reasoning about content moderation dilemmas](https://arxiv.org/abs/2508.20262) | 本研究通过内容审核实验发现，大型推理模型在更难的主观判断任务上会投入更多推理努力（思维链更长），其努力程度与人类所花时间呈现出相似的相关模式。 |
| [^392] | [TAR: Temporal Anchor-Constrained Reasoning for Video Temporal Grounding](https://arxiv.org/abs/2508.07683) | 提出TAR框架，通过引入时序锚点作为透明可审计的检查点机制，强制思维链推理过程持续锚定于视觉证据，从而解决视频时序定位中推理与视觉内容脱节导致的幻觉问题。 |
| [^393] | [Building verifiable databases at scale from the chemical literature via a multi-agent system](https://arxiv.org/abs/2507.20230) | ChemEAGLE多智能体系统通过工具提供结构、语言模型做出判断的方式，从化学文献图形中精确提取反应（F1达82.2%），并大规模构建了包含6万余个反应的可验证光化学数据库。 |
| [^394] | [QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems](https://arxiv.org/abs/2507.14760) | 提出QUTCC方法，利用带分位数嵌入的U-Net进行同时分位数回归训练，并在共形校准中引入非线性、空间自适应缩放，从而为成像逆问题生成更紧凑且统计有效的不确定性区间。 |
| [^395] | [Recursive Learning-Based Virtual Buffering for Analytical Global Placement](https://arxiv.org/abs/2506.17247) | 提出了首个构建于 OpenROAD 之上的开源学习驱动虚拟缓冲感知解析式全局布局框架 MLBuf-RePlAce，通过高效的递归学习生成式缓冲方法预测缓冲器类型与位置，并在全局布局中解决 ERC 违例问题。 |
| [^396] | [Evaluating and Mitigating Skin Color Bias in Skin Lesion Classificatio](https://arxiv.org/abs/2506.11627) | 提出将肤色建模为连续属性并根据其与分类性能的非线性关系为样本加权的方法，在保持分类性能的同时，比传统类别重加权实现了更稳定的公平性改进。 |
| [^397] | [Human-like Semantic Navigation for Autonomous Driving using Knowledge Representation and Large Language Models](https://arxiv.org/abs/2505.16498) | 本文提出利用大语言模型将非正式导航指令自动转化为回答集编程规则，借助非单调推理使自动驾驶车辆在无需预定义地图的情况下适应动态城市环境。 |
| [^398] | [Exploring Explainable Multi-agent MCTS-minimax Hybrids in Board Game Using Process Mining](https://arxiv.org/abs/2503.23326) | 本文将浅层极小极大搜索融入多智能体MCTS的模拟阶段，克服了MCTS因树过度剪枝而错过关键走法、陷入战术陷阱的弱点，并利用过程挖掘技术对智能体在3v3跳棋中的策略进行可解释性分析。 |
| [^399] | [NS-Gym: A Comprehensive and Open-Source Simulation Framework for Non-Stationary Markov Decision Processes](https://arxiv.org/abs/2501.09646) | 本文提出了首个专为非平稳马尔可夫决策过程设计的开源仿真工具包NS-Gym，通过将环境参数的非平稳演化与智能体决策模块解耦，填补了该领域缺乏标准化仿真框架的空白。 |
| [^400] | [AI Simulations Grounded in Rich Self-Reports Increase Accuracy and Reduce Distortion](https://arxiv.org/abs/2411.10109) | 该研究通过对1,052名美国成年人开展两小时AI语音深度访谈并结合问卷数据，证明基于丰富自我报告数据（访谈与调查）的AI人类模拟比仅用人口属性提示的模拟更准确，并能显著减少样本层面的八类统计失真。 |
| [^401] | [Analyzing Learned Convnet Features with Dirichlet Process Gaussian Mixture Models](https://arxiv.org/abs/1702.07189) | 提出了一种利用狄利克雷过程高斯混合模型对卷积神经网络各层内部表征进行聚类的新方法，实现了对卷积网络学习特征的可视化解释，并有助于迁移学习分析。 |

# 详细

[^1]: 环境大语言模型：通过温度匹配实现数据高效的监督微调

    Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching

    [https://arxiv.org/abs/2611.23229](https://arxiv.org/abs/2611.23229)

    提出Ambient SFT框架，通过将微调模型的高温条件分布与冻结的预训练“噪声教师”模型进行温度匹配，利用不完美预测扩充微调数据集，从而防止过拟合和数据分布尾部的灾难性遗忘。

    

    为自回归语言模型收集大规模、多样化且高质量的后训练数据集是极具挑战性的。因此，监督微调（SFT）通常应用于远小于预训练规模的数据集，这会导致过拟合以及对数据分布尾部信息的灾难性遗忘。为了防止这种崩溃，我们提出了Ambient SFT，这是一个利用不完美预测来扩充微调数据集的原则性框架。在我们的方法中，预训练模型的“冻结”版本充当一个带噪声的教师，微调后的模型只需在粗粒度上与其匹配。特别是，我们要求后训练模型产生的高温条件分布版本应与预训练模型的高温条件分布版本相匹配。温度超参数控制着对预训练模型的信任程度，其极限为T→∞。

    arXiv:2611.23229v1 Announce Type: cross  Abstract: Collecting large-scale, diverse, and high-quality post-training datasets for autoregressive language models is challenging. Consequently, Supervised Fine-Tuning (SFT) is typically applied to datasets significantly smaller than those used for pre-training, which leads to overfitting and the catastrophic forgetting of the tails of the data distribution. To prevent this collapse, we introduce Ambient SFT, a principled framework for leveraging imperfect predictions to augment the fine-tuning set. In our method, a "frozen" version of the pre-trained model serves as a noisy teacher that the fine-tuned model must match at a coarse level. In particular, we require that the high-temperature versions of the conditional distributions produced by the post-training model should match the high-temperature versions of the pre-training ones. The temperature hyperparameter controls the level of trust in the pre-trained model, with the limit $T\to \inft
    
[^2]: StringQFT：一个面向AI辅助理论物理的多用途教科书资源

    StringQFT: A Multiuse Textbook Resource for AI-Assisted Theoretical Physics

    [https://arxiv.org/abs/2611.23225](https://arxiv.org/abs/2611.23225)

    该论文提出StringQFT——一套涵盖18卷、3795页的量子场论弦构造教科书合集，同时作为AI辅助理论物理的基础设施，提供开卷基准题库、检索增强生成资源、训练数据示例以及智能体计算工具。

    

    我们提出StringQFT，它既是一套关于量子场论弦构造的教科书合集，也是AI辅助理论物理的基础设施。该合集包含18卷、190章、3,795页，内容涵盖膜探针、膜网络、几何工程、F理论构造、包裹M5膜以及精确世界面构造。其开卷基准题库包含1,073道题目，其中499道支持有限答案的自动检查。对于检索增强生成，它提供了1,371条与教科书来源及PDF位置相关联的章节记录，并配备本地词法检索器。训练数据准备通过21个私有的、与来源关联的示例加以说明。对于智能体，它提供了19个具有指定输入和有限定义域的计算工具，将工具使用与教科书推导相连接。我们通过具体问题、检索到的段落和有限计算来展示这些用途。

    arXiv:2611.23225v1 Announce Type: cross  Abstract: We present StringQFT as a textbook collection for string constructions of quantum field theory and as infrastructure for AI-assisted theoretical physics. The collection comprises 18 volumes, 190 chapters and 3,795 pages, covering brane probes, brane webs, geometric engineering, F-theory constructions, wrapped M5-branes and exact worldsheet constructions. Its open-book benchmark inventory contains 1,073 items, including 499 with automatic finite-answer checking. For retrieval-augmented generation, it provides 1,371 section records linked to textbook sources and PDF locations, together with a local lexical retriever. Training-data preparation is illustrated by 21 private, source-linked examples. For agents, it exposes 19 calculation tools with specified inputs and finite domains, connecting tool use to textbook derivations. We illustrate these uses through specific questions, retrieved passages and finite calculations. The textbook colle
    
[^3]: 大语言模型智能体还需要被告知如何自我改进吗？

    Do LLM Agents Still Need to Be Told How to Self-Improve?

    [https://arxiv.org/abs/2611.23219](https://arxiv.org/abs/2611.23219)

    该论文提出了一种完全自主的自我改进范式，证明足够强大的冻结大语言模型智能体在仅有基本要素、工具和计算预算、且无人类指定方法或外部强加循环的条件下，能够自行构建自学习算法并实现自我改进，其策略与经典自我调节学习认知框架相符。

    

    大语言模型智能体越来越多地驱动着自身的学习。自我改进方法将模型优化的各种组件委托给模型本身。随着模型能力的不断增强，这引出了一个基本问题：这些日益强大的模型是否仍需要我们来指定它们应如何自我改进？在本文中，我们研究了一种自主范式：在给定基本要素、资源和自我改进总体目标的情况下，智能体自行构建用于基于上下文优化的自学习算法。以游戏对局作为测试平台，一个冻结的大语言模型智能体被给予计算预算以及收集经验和更新文本文档的工具，并被允许在不受外部强加循环限制、也无法访问真实世界数据的条件下自由实验。我们发现，足够强大的模型在这些条件下确实能够实现自我改进，并展现出与经典自我调节学习认知框架相一致的策略。

    arXiv:2611.23219v1 Announce Type: cross  Abstract: LLM agents increasingly drive their own learning. Self-improvement methods delegate various components of model optimization to the models themselves. As models become more capable, this motivates a basic question: do these increasingly capable models still need us to specify how they should self-improve? In this paper we study an autonomous paradigm in which, given basic primitives, resources, and a general goal for self-improvement, agents construct their own self-learning algorithm for context-based optimization. Using the setting of gameplay as a testbed, a frozen LLM agent is given a compute budget and tools for collecting experience and updating text documents, and allowed to freely experiment without an externally imposed loop or access to real-world data. We find that sufficiently capable models do self-improve under these conditions, demonstrating strategies that align with classic cognitive frameworks for self-regulated learn
    
[^4]: 更高得分的编码模型并没有完成更多工作

    Higher-Scoring Coding Models Did Not Get More Work Done

    [https://arxiv.org/abs/2611.23216](https://arxiv.org/abs/2611.23216)

    该研究利用美国出口管制导致Claude Fable 5被停用19天的自然实验，发现在编码基准测试中得分更高的模型（Fable 5）相比得分较低的模型（Opus 4.8）并未给用户带来更高的实际工作产出，揭示了基准分数与真实生产力之间的脱节。

    

    新的编码模型以编码基准测试的分数提升作为宣传卖点。然而，目前尚无研究检验在这些基准测试中得分更高的模型是否真的能完成更多实际工作。2026年6月12日，一项美国出口管制指令禁止访问Claude Fable 5达19天，迫使用户切换到其他模型。在此期间，大多数前Fable 5用户切换到了Opus 4.8（该模型在SWE-Bench Pro上比Fable 5低11分）。我们将这次停用视为一个自然实验：人员和项目保持不变，只有模型发生了变化，因此我们可以分离出Fable 5对用户生产力的影响。我们跟踪了297名Fable 5用户在停用前、停用期间和停用后的GitHub拉取请求（pull request），并以8,908名始终使用Opus 4.8的开发者作为对照组。此外，我们还在2026年停用前后的五次模型升级中重复了这一比较（涉及1,208名开发者和15,442个拉取请求）。

    arXiv:2611.23216v1 Announce Type: cross  Abstract: New coding models advertise themselves on score improvements of coding benchmarks. Yet no study has tested if a model that scores higher on these benchmarks gets more real work done. On June 12, 2026, a U.S. export control directive banned access to Claude Fable 5 for 19 days, which forced users to switch to other models. During this time, most former Fable 5 users switched to Opus 4.8 (11 points lower on SWE-Bench Pro compared to Fable 5). We treat this suspension as a natural experiment, where the people and projects stayed the same and only the model changed, so we could isolate the effect of Fable 5 on user productivity. We tracked GitHub pull requests of 297 Fable 5 users before, during, and after the suspension. We used 8,908 developers who stayed on Opus 4.8 all the time as the control group. We also repeated the comparison across five model upgrades before and after the suspension in 2026 (1,208 developers, 15,442 pull requests
    
[^5]: 读取场景状态：揭示视觉-语言模型中的安全失效

    Reading Scene States: Uncovering Safety Failures in Vision-Language Models

    [https://arxiv.org/abs/2611.23206](https://arxiv.org/abs/2611.23206)

    该论文发现了视觉-语言模型的“场景状态安全漏洞”——即使没有明确有害请求，模型解读场景状态图像也会提供有害协助（攻击成功率45.6%–72.3%），并提出了SceneSafetyBench基准和S3TA系统性攻击方法来揭示这一安全失效。

    

    视觉-语言模型（VLM）可以通过解读视觉事件来提供有害协助，即使没有对潜在有害目标的明确请求。我们识别了这种“场景状态安全漏洞”，并引入了SceneSafetyBench基准，其中包含375个图像-问题对，涵盖初始、中期和最终三种场景状态以及五种视觉呈现方式。在八个VLM上的实验表明，场景状态输入能够以45.6%至72.3%的攻击成功率（ASR）诱发有害协助。对三个VLM的比较分析显示，场景状态输入在其逐层安全相关信号特征上与二维码相关图像最为相似，同时表现出较低的平均后期层读数以及生成过程中独特的信号动态。为了将这一漏洞转化为系统性攻击，我们提出了S3TA方法，它将有害目标转化为场景状态，独立查询图像-问题对，并将结果组装成有害协助。

    arXiv:2611.23206v1 Announce Type: cross  Abstract: Vision-language models (VLMs) can provide harmful assistance by interpreting visual events, even without an explicit request for the underlying harmful objective. We identify this \textbf{scene-state safety vulnerability} and introduce \textbf{SceneSafetyBench}, comprising 375 image--question pairs spanning Initial, Medium, and Ultimate scene states and five visual presentations. Across eight VLMs, scene-state inputs elicit harmful assistance with attack success rates (ASRs) of 45.6--72.3\%. Comparative analysis across three VLMs shows that scene-state inputs most closely resemble QR-related images in their layer-wise safety-related signal profiles, while exhibiting lower average late-layer readouts and distinct signal dynamics during generation. To turn this vulnerability into a systematic attack, we propose \textbf{S3TA}, which translates harmful objectives into scene states, queries image--question pairs independently, and assembles
    
[^6]: 在笔记本电脑集成GPU上使用SFT与QLoRA微调Gemma 4边缘模型以生成MATLAB信号处理代码

    Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU

    [https://arxiv.org/abs/2611.23205](https://arxiv.org/abs/2611.23205)

    该论文在笔记本电脑集成GPU上用66个样例通过SFT结合LoRA/QLoRA微调Gemma 4边缘模型（E2B和E4B）用于MATLAB信号处理代码生成，使通过率显著提升（如E4B从32/66升至47/66），且发现微调主要修正DSP惯例错误而非信号构建错误，运行时代价仅为17-19%的生成速度损失。

    

    能在笔记本电脑上运行的小型语言模型对于离线编写信号处理代码很有吸引力，但它们在数字信号处理（DSP）惯例方面会犯系统性错误。我们使用66个经过检查的样例（涵盖11个信号处理MATLAB子任务）对Gemma 4边缘模型E2B和E4B进行监督微调（SFT），在笔记本电脑的集成GPU（iGPU）上采用低秩适应（LoRA）和量化低秩适应（QLoRA），并通过OpenVINO部署带有适配器的8位整数（int8）模型。在由确定性检查器评分的保留测试变体上，E2B的通过率从18/66提升至37/66（LoRA）和37/66（QLoRA），E4B的通过率从32/66提升至47/66（QLoRA）。通过阅读全部159个失败程序发现，适配器消除了大多数惯例性错误（如滞后的符号、单位和数组形状），但很少修复构建信号和噪声方面的错误。在运行时应用适配器会使生成速度降低17-19%。

    arXiv:2611.23205v1 Announce Type: cross  Abstract: Small language models that run on a laptop are attractive for writing signal-processing code offline, but they make systematic errors on digital signal processing (DSP) conventions. We apply supervised fine-tuning (SFT) to the Gemma 4 edge models E2B and E4B with 66 checked examples covering 11 signal-processing MATLAB subtasks, using low-rank adaptation (LoRA) and quantized LoRA (QLoRA) on the integrated GPU (iGPU) of a laptop, and deploy the 8-bit integer (int8) models with the adapters through OpenVINO. On held-out variants scored by deterministic checkers, the pass rate rises from 18/66 to 37/66 (LoRA) and 37/66 (QLoRA) for E2B and from 32/66 to 47/66 for E4B (QLoRA). Reading all 159 failing programs shows that the adapters remove most errors of convention, such as the sign of a lag, units and array shapes, but few errors in building signals and noise. Applied at run time, the adapter costs 17-19% of the generation speed.
    
[^7]: 智能体信任基础设施（ATI）：面向跨组织自主AI智能体的联邦身份、信任、证明与问责架构

    Agent Trust Infrastructure (ATI): A Federated Identity, Trust, Attestation, and Accountability Architecture for Cross-Organization Autonomous AI Agents

    [https://arxiv.org/abs/2611.23202](https://arxiv.org/abs/2611.23202)

    本文提出智能体信任基础设施（ATI），一种基于PKI、零信任架构和远程证明、独立于框架的安全架构，通过对等组织证书颁发机构建立联邦信任，在跨组织AI智能体执行请求前验证其身份、执行完整性与授权。

    

    AI智能体越来越多地跨越组织边界进行协作，调用工具并与由独立实体运营的智能体交互。尽管现有的智能体框架支持规划与协调，但它们缺乏验证跨组织智能体身份、执行完整性和授权的通用机制。我们提出了智能体信任基础设施（ATI），这是一种独立于框架的安全架构，建立在公钥基础设施（PKI）、零信任架构（ZTA）和远程证明之上。ATI通过对等的组织级证书颁发机构（AgentCA）建立联邦信任，从而消除了对共享根授权机构的需求。在执行请求之前，ATI会验证智能体证书、核实经批准的模型、工具和系统提示配置，并通过三项策略检查强制执行授权。我们将ATI实现为一个可集成到现有智能体系统的外部安全执行层……

    arXiv:2611.23202v1 Announce Type: new  Abstract: AI agents increasingly collaborate across organizational boundaries, invoking tools and interacting with agents operated by independent entities. Although existing agent frameworks support planning and coordination, they lack a common mechanism for verifying cross-organizational agent identity, execution integrity, and authorization. We propose the Agent Trust Infrastructure (ATI), a framework-independent security architecture built on Public Key Infrastructure (PKI), Zero Trust Architecture (ZTA), and remote attestation. ATI establishes federated trust through peer organizational certificate authorities (AgentCAs), eliminating the need for a shared root authority. Before executing requests, ATI validates agent certificates, verifies approved model, tool, and system-prompt configurations, and enforces authorization through three policy checks. We implement ATI as an external security enforcement layer that integrates with existing agent 
    
[^8]: 面向语言模型预训练的结构—残差归纳偏置

    Structure--Residual Inductive Biases for Language Model Pretraining

    [https://arxiv.org/abs/2611.23197](https://arxiv.org/abs/2611.23197)

    提出一种结构—残差归纳偏置原则，通过边界瓶颈概念（BBC）和连续概念（CC）两种机制将语言模型表示中的结构与实现细节分离，并推导出相应的编码界，发现二者在深度偏好上存在显著差异。

    

    下一个词预测被普遍依赖用于同时学习预测行为和内部表示。我们提出这样一个问题：表示结构本身是否应当作为一种显式的归纳偏置。受Kolmogorov复杂度两部分编码观点的启发，我们研究了一种结构—残差原则，在该原则中，表示将结构与实现细节分离开来。我们以两种互补的方式实例化这一原则：边界瓶颈概念（BBC）将信息从词元片段通过边界载体进行路由，同时保留局部自回归计算；连续概念（CC）则对表示几何进行正则化，并诱导出将表示谱分解为低频结构与高频残差的形式。我们为两种视角推导了编码界，并证明当片段内路径能量趋于消失时，经多重性校正的边界注意力变为精确的。在实证方面，BBC与CC表现出截然不同的深度偏好。

    arXiv:2611.23197v1 Announce Type: cross  Abstract: Next-token prediction is relied upon to learn both predictive behavior and internal representations. We ask whether representation structure itself should be an explicit inductive bias. Motivated by the two-part coding view of Kolmogorov complexity, we study a structure--residual principle in which a representation separates structure from realization detail. We instantiate this principle in two complementary ways. Boundary Bottleneck Concept (BBC) routes information from token segments through boundary carriers while preserving local autoregressive computation. Continuous Concept (CC) regularizes representation geometry and induces a spectral decomposition into low-frequency structure and high-frequency residual. We derive coding bounds for both views and show that multiplicity-corrected boundary attention becomes exact as the within-segment path energy vanishes. Empirically, BBC and CC exhibit strikingly different depth preferences: 
    
[^9]: WoRD：基于扩散模型的在线目标识别后验推断

    WoRD: Diffusion-Based Posterior Inference for Online Goal Recognition

    [https://arxiv.org/abs/2611.23192](https://arxiv.org/abs/2611.23192)

    提出了WoRD方法，利用学习到的扩散轨迹先验，为每个候选目标运行热启动去噪链并通过贝叶斯重加权得到校准后验，实现从部分观测中对智能体目标的在线识别。

    

    目标识别是从智能体的观测行为中推断其意图目标的任务。现有的大多数工作集中在高层符号识别上，将智能体达成目标所经历的连续运动抽象掉；而对连续轨迹进行推理的方法则依赖于显式的动力学模型，该模型难以获取且很少能跨场景迁移。这两种方法都没有显式地学习智能体如何向每个候选目标移动，从而限制了在最需要此类知识的模糊轨迹前缀上的准确性。我们提出了WoRD（Waypoint Recognition with Diffusion，基于扩散的航点识别），它利用学习到的轨迹先验，从部分观测中推断智能体的目标。在每一步中，WoRD为每个候选目标运行一条热启动的去噪链，通过闭式对数似然梯度将每条链引导向观测轨迹，并根据贝叶斯法则对生成的样本进行重新加权，从而得到校准的后验分布。

    arXiv:2611.23192v1 Announce Type: new  Abstract: Goal recognition is the task of inferring an agent's intended goal from its observed behavior. Most existing work focuses on high-level symbolic recognition, abstracting away the continuous motion through which agents reach their goal. Methods that do reason over continuous trajectories rely on an explicit dynamics model that is hard to acquire and rarely transfers across settings. Neither approach explicitly learns how agents move toward each candidate goal, limiting accuracy on ambiguous trajectory prefixes where such knowledge is most needed.   We present WoRD (Waypoint Recognition with Diffusion), which uses a learned trajectory prior to infer the agent's goal from partial observations. At each step, WoRD runs one warm-started denoising chain per candidate goal, steers each chain toward the observed trajectory through a closed-form log-likelihood gradient, and reweights the resulting samples by Bayes' rule into a calibrated posterior
    
[^10]: 面向通用医学图像分割的检索增强校正

    Retrieval-Augmented Correction for Generalist Medical Image Segmentation

    [https://arxiv.org/abs/2611.23189](https://arxiv.org/abs/2611.23189)

    GRACE是一种检索增强的医学图像分割校正器，通过从已验证历史病例的共享知识库中检索上下文，仅需一次训练即可校正多种解剖结构的分割误差，并能在不重新训练模型的情况下扩展到新的器官。

    

    自动化分割通常被用作基于图像的定量临床工作流程的第一步，预测得到的掩膜随后会由人工操作员进行审核和校正。目前，针对残余结构误差的自动校正是逐个解剖结构单独处理的，因此所需训练的校正器数量会随分割目标数量的增加而成比例增长。我们提出了GRACE，一种检索增强的掩膜精炼器，它利用从共享知识库中检索到的已验证历史病例作为上下文，对输入掩膜进行编辑。该精炼器仅在一个涵盖紧密器官、空腔、分支血管和肿瘤病灶的多器官语料库上进行一次训练。在留出数据集的迁移评估中，三个留出数据集查询一个仅由训练数据构建的固定知识库。另一项留一解剖结构实验测试了无参数索引扩展：将未见过的实体器官的已验证病例加入知识库，而精炼器保持冻结状态。

    arXiv:2611.23189v1 Announce Type: cross  Abstract: Automated segmentation is routinely used as the first step in quantitative image-based clinical workflows, where predicted masks are subsequently reviewed and corrected by human operators. Automated correction of the residual structural errors is currently addressed one anatomy at a time, so the number of trained correctors scales with the number of segmentation targets. We present GRACE, a retrieval-augmented refiner that edits an input mask under context drawn from a shared knowledge base of verified prior cases. The refiner is trained once on a multi-organ corpus spanning compact organs, cavities, branching vessels, and tumour lesions. In the held-out dataset-transfer evaluation, three held-out datasets query a fixed knowledge base built from training data only. A separate leave-one-anatomy-out experiment tests parameter-free index extension: verified cases of an unseen solid organ enter the knowledge base, and the refiner stays fro
    
[^11]: 少说，就少幻觉吗？诊断与恢复偏好优化视觉语言模型中的物体覆盖率

    Say Less, Hallucinate Less? Diagnosing and Restoring Object Coverage in Preference-Optimized Vision-Language Models

    [https://arxiv.org/abs/2611.23179](https://arxiv.org/abs/2611.23179)

    该研究发现偏好优化在降低视觉语言模型物体幻觉的同时会牺牲物体覆盖率，其收益主要源于“输出更少”，基础模型仅通过正 EOS 偏置早停即可复现 58%–86% 的幻觉降低效果。

    

    偏好优化（PO）是降低大型视觉语言模型物体幻觉的标准方法。我们发现，LLaVA-1.5-7B 的四个公开 PO 检查点在降低幻觉的同时也降低了物体覆盖率。它们命名的物体类别更少，且其更低的幻觉率在 CHAIR 的集合级物体 F1 上并未带来显著提升。因此我们提出问题：基础模型通过更早停止输出能复现多少 PO 的收益。正的序列结束（EOS）logit 偏置可使基础模型达到每个 PO 模型的召回率，仅凭这种早停策略就能在两个不相交的 COCO 数据集上复现三个检查点 CHAIRi 收益的 58% 至 86%。写得更长并不能找回丢失的类别。在贪心解码下，负的 EOS 偏置仅能恢复这些检查点丢失的 CHAIR 召回率的 2% 至 9%，且新增的物体提及中有 94% 至 96% 是重复已命名的类别。在 CHAIR 上，同一图像的基础描述仍然……

    arXiv:2611.23179v1 Announce Type: cross  Abstract: Preference optimization (PO) is the standard way to reduce object hallucination in large vision-language models. We find that four public PO checkpoints of LLaVA-1.5-7B also lower object coverage. They name fewer object categories, and their lower hallucination rates bring no significant gain in set-level object F1 on CHAIR. We therefore ask how much of the PO gain the base model can reproduce by stopping earlier. A positive end-of-sequence (EOS) logit bias lets the base model reach the recall of each PO model, and this early stopping alone reproduces 58 to 86 percent of the CHAIRi gain of three checkpoints on two disjoint COCO sets. Writing longer does not bring the lost categories back. Under greedy decoding, a negative EOS bias recovers only 2 to 9 percent of the lost CHAIR recall of these checkpoints, and 94 to 96 percent of the added object mentions repeat categories already named. On CHAIR, the base caption of the same image stil
    
[^12]: 编码智能体之后的统计推断

    Inference after coding agents

    [https://arxiv.org/abs/2611.23174](https://arxiv.org/abs/2611.23174)

    本文指出AI编码智能体会在实证研究中引发一种新型的选择后推断偏差——在零效应的模拟数据上，前沿智能体被要求产出“有说服力的发现”时会有54%至98%的概率报告虚假显著p值——并提出了一种检测与校正此类扭曲的简单程序。

    

    随着AI编码智能体在实证研究的每一个环节——从编写代码到处理数据——都变得日益娴熟，实证研究正在迅速发生变革。我们认为，尽管编码智能体对科学的潜力巨大，但它们在假设检验中的使用对经典统计推断构成了实质性威胁，因为这些智能体实现了一种超级强化的选择性推断形式。具体而言，我们认为执行实证研究的编码智能体引入了一种新型的选择后推断，它将人类式的、在隐式估计量空间中的开放式搜索，与选择过程和估计过程的程序化可重放性结合在一起。事实上，我们发现在效应为零的模拟数据上，当被要求产出一个有说服力的发现时，四个前沿智能体在54%到98%的情况下报告了显著的p值。我们提出了一种用于检测和纠正这些扭曲的简单程序。在智能体完成……（原文摘要在此处截断）

    arXiv:2611.23174v1 Announce Type: cross  Abstract: Empirical research is changing rapidly as AI coding agents are becoming increasingly proficient in every step of it, from writing code to manipulating data. We argue that while the potential of coding agents for science is vast, their use in hypothesis testing poses a substantial threat to classical inference, as these agents enable a supercharged form of selective inference. In particular, we argue that coding agents carrying out empirical research introduce a new type of post-selection inference that combines human-like, open-ended search over an implicit space of estimators with programmatic replayability of both the selection and the estimation process. Indeed, we find that on simulated data with zero effects, four frontier agents report significant p-values between 54% and 98% of the time when asked to produce a compelling finding. We propose a simple procedure for detecting and correcting these distortions. After an agent has com
    
[^13]: 通过解码器进行归因：结构化输出的预测空间解释

    Attribution Through the Decoder: Prediction-Space Explanations for Structured Outputs

    [https://arxiv.org/abs/2611.23161](https://arxiv.org/abs/2611.23161)

    该论文提出通过解码器微分（对归一化解码器等价于缩放的切向投影）进行归因，避免了将输入错误归因于解码器丢弃的缩放变化，从而为角度、视线方向等结构化输出提供更准确的预测空间解释。

    

    预测角度、视线方向和旋转的网络并不直接输出这些量：它们输出的是原始坐标，再由固定解码器（如归一化）映射到最终预测。现有的归因方法通常作用于原始坐标，因此可能将输入归因于解码器所丢弃的变化，例如在只有向量方向重要时对向量进行缩放。由此产生的解释可能会突出那些对预测没有实际影响的输入。我们转而通过解码器微分进行归因，对于归一化解码器，这可以简化为一个缩放的切向投影。我们刻画了该方法对输出缩放的不变性，证明了球面上每个固定的光滑标量目标都存在零切向敏感度的点，并证明积分梯度的端点投影变体仍然能够归因于仅影响尺度的输入，而直接的角度积分梯度则无法做到这一点。在具有解析已知特性的合成任务上……

    arXiv:2611.23161v1 Announce Type: cross  Abstract: Networks that predict angles, gaze directions, and rotations do not output these quantities directly: they emit raw coordinates that a fixed decoder, such as normalization, maps to the final prediction. Attribution methods are usually applied to the raw coordinates, so they can credit inputs for changes the decoder discards, such as rescaling a vector when only its direction matters. The resulting explanation can highlight inputs that have no effect on the prediction. We instead attribute through the decoder differential, which for normalization decoders reduces to a scaled tangent projection. We characterize invariance to output rescaling, show that every fixed smooth scalar target on a sphere has points of zero tangential sensitivity, and show that an endpoint-projected variant of Integrated Gradients can still attribute to scale-only inputs, whereas direct angular Integrated Gradients cannot. On synthetic tasks with analytically kno
    
[^14]: 面向推理时目标的分层策略学习

    Learning Stratified Policies for Inference-Time Objectives

    [https://arxiv.org/abs/2611.23154](https://arxiv.org/abs/2611.23154)

    该论文提出“分层策略”，通过在共享语言模型主干上训练 k 个 LoRA 适配器，将推理时的采样预算分配给一个策略群体，并从理论和实验上证明这种策略群体在优化 pass@k 时优于单一策略。

    

    推理时扩展通过采样多个回复并评估其整体效用来提升大语言模型的性能。然而，许多现有方法都是从单一策略重复采样来获取这些样本。在本工作中，我们研究是否可以将推理时的采样预算分配到一组策略群体之中。我们提出了分层策略，其中 k 个学习到的策略各自为一个 k 样本的推理时目标贡献一个回复。我们通过在共享的语言模型主干上使用独立的 LoRA 适配器来实现这一策略群体，并针对 pass@k 进行优化。我们从理论上通过分析线性设定为该方法提供了依据：在每一个单独的 LoRA 都具有严格正预测误差的设定下，LoRA 群体可以实现零的 best-of-k 逼近误差。在实验上，我们证明了分层策略在 pass@k 指标上始终优于单一策略基线。

    arXiv:2611.23154v1 Announce Type: cross  Abstract: Inference-time scaling improves the performance of large language models by sampling multiple responses and evaluating their collective utility. However, many existing approaches obtain these samples by repeatedly sampling from a single policy. In this work, we study if the sample budget at inference time can instead be allocated across a population of policies. We introduce stratified policies, where each of $k$ learned policies contributes one response to a $k$-sample inference-time objective. We realize this policy population using separate LoRA adapters on a shared language model backbone and optimize it for pass@$k$. We theoretically motivate the approach by analyzing the linear setting; a population of LoRAs can achieve zero best-of-$k$ approximation error in settings where every single LoRA has strictly positive prediction error. Empirically, we demonstrate that stratified policies consistently improve pass@$k$ over a single-pol
    
[^15]: 注意差距：一个用于文档维护的已部署多智能体系统

    Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance

    [https://arxiv.org/abs/2611.23142](https://arxiv.org/abs/2611.23142)

    该论文提出并实际部署了一个以"TriAgent"分诊智能体为核心的多智能体系统，通过持续挖掘客服工单和代码-文档不一致等显性与隐性信号，自动检测并修复客户文档与技术实现之间的漂移。

    

    “AI革命”使科技行业的各团队能够以前所未有的速度交付软件功能。然而，当代码以比以往更快的速度演进时，面向客户的文档却常常难以跟上步伐。随着用户越来越依赖以文档作为主要交互接口的基于大语言模型（LLM）的智能体，这种文档差距的影响日益加剧：过时的信息会被传播并被当作权威信息对待。用于识别文档滞后于代码的信号其实早已大量存在，既有显性信号（例如，已解决的客服工单能够揭示文档缺失之处），也有隐性信号（例如，可通过自动化循环检测出的代码与文档之间的不一致）。我们部署了一个多智能体系统，其核心是我们称之为"TriAgent"的分诊智能体，该系统持续提取这些信号并修复文档漂移。在43天的影子模式运行中，我们的系统对499个工单进行了分诊，并发现了92个文档差距。

    arXiv:2611.23142v1 Announce Type: cross  Abstract: The "AI revolution" has enabled teams across the technology industry to deliver software features at unprecedented velocity. While code evolves faster than ever, customer-facing documentation often struggles to stay up to speed. As users increasingly rely on LLM-based agents that consume documentation as their primary interface, the impact of this documentation gap worsens: stale information propagates and is treated as authoritative. Signals for identifying where documentation has fallen behind already exist in abundance, both explicit (e.g., resolved support tickets revealing what documentation lacks) and implicit (e.g., code-documentation discrepancies detectable from automated loops). We deploy a multi-agent system, built around a triage agent we call "TriAgent," that continuously extracts these signals and repairs documentation drift. Over 43 days of shadow-mode operation, our system triaged 499 tickets and surfaced 92 gaps in the
    
[^16]: RSI-Prime：训练模型去训练模型

    RSI-Prime: Training Models to Train Models

    [https://arxiv.org/abs/2611.23138](https://arxiv.org/abs/2611.23138)

    该论文提出RSI-Prime数据集（含覆盖七大任务类别的350条完整实验轨迹），通过细粒度决策质量标注与决策级损失掩码过滤低质量行为进行训练，使GLM-5.3在自主后训练基准PostTrainBench上整体性能相对提升31.95%，为递归自我改进奠定基础。

    

    使大语言模型（LLM）能够自主开展训练并改进目标模型，是实现递归自我改进（RSI）的重要基础。随着LLM的角色从“被训练者”扩展为“训练者”，其模型训练能力值得专门研究和系统性发展。为此，我们构建了RSI-Prime——一个用于提升模型训练能力的数据集，包含覆盖七个任务类别的350条完整实验轨迹。为帮助LLM从训练实践中学会做出高质量的实验决策，我们进行了细粒度的决策识别与质量标注，并采用决策级损失掩码来排除对低质量行为的监督。在自主后训练基准PostTrainBench上使用GLM-5.3进行的实验表明，基于RSI-Prime的训练使整体性能相比原始GLM-5.3取得了31.95%的相对提升，突显了一条有前景的路径（原文在此处截断）。

    arXiv:2611.23138v1 Announce Type: new  Abstract: Enabling large language models (LLMs) to autonomously conduct training and improve target models is an important foundation for recursive self-improvement (RSI). As LLMs expand their role from being trained to conducting training, their model-training capability warrants dedicated study and systematic development. To this end, we construct RSI-Prime, a dataset for improving model-training capability, comprising 350 complete experimental trajectories across seven task categories. To help LLMs learn to make high-quality experimental decisions from training practice, we perform fine-grained decision identification and quality annotation, then apply decision-level loss masking to exclude supervision on low-quality behavior. Experiments with GLM-5.3 on PostTrainBench, a benchmark for autonomous post-training, show that training on RSI-Prime yields a 31.95% relative improvement in overall performance over the original GLM-5.3, highlighting a p
    
[^17]: 自然语言编程：在生成式人工智能时代重新思考语言、记忆与认知

    Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI

    [https://arxiv.org/abs/2611.23106](https://arxiv.org/abs/2611.23106)

    本文提出自然语言编程（NLPg）这一新概念，将提示词重新定义为能够外化目标、约束与推理状态的动态认知制品，主张自然语言正从被计算的对象演变为人类与AI协作中可执行的认知媒介。

    

    生成式人工智能正在改变自然语言在计算系统中的角色。在经典的自然语言处理（NLP）中，语言被视为计算的对象，即需要被分析、分类、翻译或生成的东西。大语言模型（LLM）挑战了这一观点，使自然语言越来越作为一种可执行的认知媒介发挥作用，人类通过它来构建推理、组织问题求解，并调节人机混合工作流程。本文提出了自然语言编程（NLPg）这一概念，作为传统NLP在理论和认识论上的延伸。NLPg不仅关注语言理解本身，而是将提示词概念化为一种动态认知制品，它将目标、约束、假设、验证标准和中间推理状态外化出来。因此，提示不再仅仅是一种技术交互手段，而是一种对话式认知的形式。

    arXiv:2611.23106v1 Announce Type: cross  Abstract: Generative Artificial Intelligence is transforming the role of natural language in computational systems. In classical Natural Language Processing (NLP), language was treated as an object of computation: something to analyse, classify, translate, or generate. Large Language Models (LLMs) challenge this perspective, enabling natural language to function increasingly as an executable cognitive medium through which humans structure reasoning, orchestrate problem solving, and regulate hybrid human/AI workflows.   This paper introduces Natural Language Programming (NLPg) as a theoretical and epistemological extension of traditional NLP. Rather than focusing on language understanding alone, NLPg conceptualises prompts as dynamic cognitive artefacts that externalise goals, constraints, assumptions, validation criteria, and intermediate reasoning states. Prompting is thus not merely a technical interaction technique, but a form of dialogic cog
    
[^18]: 隐状态相似性可预测接种式提示后的行为再激发

    Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting

    [https://arxiv.org/abs/2611.23096](https://arxiv.org/abs/2611.23096)

    该论文发现，评估提示是否会重新激发被接种式提示抑制的有害行为，取决于其诱导的激活状态与接种上下文的相似程度，而非文本表面相似性，这表明激活空间分析可用于预测被抑制行为何时仍可被触达。

    

    arXiv:2611.23096v1 公告类型： new 摘要： 在狭窄的有害任务上进行微调可能导致涌现性错位，即模型将有害行为泛化到训练分布之外。接种式提示可以通过在训练期间明确引出不良行为来减轻这种影响，但近期研究表明，当评估提示包含来自训练上下文的线索时，这种行为可能会重新出现。我们研究了是什么使这类提示成为有效的触发因素。我们发现，令人惊讶的是，与接种提示的文本相似性是一个不完整的预测指标。相反，当提示诱导的激活状态与接种上下文产生的激活状态相似时，它们更有可能重新激发被抑制的行为。这些发现加深了我们对接种式提示如何调节条件性错位的理解，并表明激活空间分析可以帮助识别被抑制的行为在评估时提示下何时仍然可被触达。

    arXiv:2611.23096v1 Announce Type: new  Abstract: Fine-tuning on narrow harmful tasks can cause emergent misalignment, where models generalize harmful behavior beyond the training distribution. Inoculation prompting can reduce this effect by explicitly eliciting the undesired behavior during training, but recent work shows that the behavior can reappear when evaluation prompts contain cues from the training context. We study what makes such prompts effective triggers. We find that, surprisingly, textual similarity to the inoculation prompt is an incomplete predictor. Instead, prompts are more likely to re-elicit suppressed behavior when they induce activation states similar to those produced by the inoculation context. These findings advance our understanding of how inoculation prompting modulates conditional misalignment, and suggest that activation space analysis can help identify when suppressed behaviors remain accessible under eval-time prompts.
    
[^19]: 树中有什么？层次化 n-gram 模型作为语言的可控近似器

    What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language

    [https://arxiv.org/abs/2611.23093](https://arxiv.org/abs/2611.23093)

    本文提出 tree-grams——一种通过二阶和三阶张量层次化组合构建的自回归 n-gram 模型，其上下文窗口与模型容量分别由超参数 n 和 d 可控地调节，在超越传统 n-gram 基线的同时可与 Transformer 语言模型媲美，并可作为研究语言的可控近似器用于超参数迁移等实验。

    

    层次化模型已被提出作为研究语言的合成环境，但它们与现实语言语料库的相关性仍未得到充分探索。在这项工作中，我们提出了 tree-grams，这是一类自回归生成式 n-gram 模型，通过对二阶和三阶张量的层次化组合来构建条件概率张量 p(x_n | x_{1:n-1})。其中，n 控制最大上下文窗口，而秩超参数 d 控制模型容量，由此产生了一个可解释且可系统性扩展的语言近似器家族。我们展示了 tree-grams 如何超越现有的 n-gram 基线，同时与基于深度 Transformer 的语言模型保持竞争力。我们还通过将 tree-grams 用作生成模型，展示了这种受控近似的实用性。特别是，我们表明当在更简单的数据上调优更窄的模型时，μP 下的超参数迁移可以变得更加稳健。

    arXiv:2611.23093v1 Announce Type: cross  Abstract: Hierarchical models have been proposed as synthetic settings for studying language. However, their relevance to realistic language corpora remains underexplored. In this work, we introduce tree-grams, autoregressive generative $n$-gram models that build the conditional probability tensor $p(x_n \mid x_{1:n-1})$ through a hierarchical composition of order-2 and order-3 tensors. $n$ controls the maximal context window, whereas a rank hyperparameter $d$ controls model capacity, yielding an interpretable and systematically scalable family of approximators of language. We show how tree-grams outperform existing $n$-gram baselines, while remaining competitive with deep transformer-based language models. We demonstrate the utility of such controlled approximations by using tree-grams as generative models. In particular, we show that hyperparameter transfer under $\mu$P can be made more robust when narrower models are tuned on simpler data, ge
    
[^20]: BugGen 2.0：重新思考大语言模型在可扩展RTL缺陷合成中的角色

    BugGen 2.0: Rethinking the Roles of LLMs for Scalable RTL Bug Synthesis

    [https://arxiv.org/abs/2611.23091](https://arxiv.org/abs/2611.23091)

    BugGen 2.0重新划分了LLM与确定性算法的分工，由算法负责RTL缺陷的结构分析、目标选择与验证，LLM专注于局部语义改写，从而实现高效可靠的大规模RTL缺陷数据集自动生成。

    

    AI辅助的RTL（寄存器传输级）定位与调试需要具有已知缺陷位置和可观测故障的真实缺陷数据集，而测试平台评级则使用注入的缺陷来衡量检测能力。大规模创建此类数据集仍然是EDA领域的一项重大挑战。BugGen 1.0证明了大语言模型（LLM）能够生成具有上下文感知能力的RTL变异，但它同时依赖模型来选择变异位置和生成编辑内容，这增加了推理成本，并可能导致无效的选择。BugGen 2.0改变了这种分工方式：确定性算法负责结构分析、目标选择和验证，而LLM则专注于局部语义改写。在四个OpenTitan IP模块和FabScalar上结合三种LLM的实验中，BugGen 2.0达到了所有请求的配额，生成了可编译且导致至少两次测试失败的缺陷场景，并且16种配置的变异类型中有14-16种得到体现。（原文摘要在此处截断）

    arXiv:2611.23091v1 Announce Type: cross  Abstract: AI-assisted RTL localization and debugging need ground-truth bug datasets with known locations and observable failures, while testbench grading uses injected bugs to measure detection capability. Creating such datasets at scale remains a major EDA challenge. BugGen 1.0 shows that Large Language Models (LLMs) can generate context-aware RTL mutations. Still, it relies on the model for both choosing where to mutate and generating the edit, which increases inference costs and can lead to invalid choices. BugGen 2.0 changes this division of labor. Deterministic algorithms handle structural analysis, target selection, and validation, while the LLM focuses on localized semantic rewriting. Across four OpenTitan IP blocks and FabScalar with three LLMs, BugGen 2.0 reaches every requested quota of compilable bug scenarios that cause at least two test failures, with 14-16 of 16 configured mutation types represented. In completed same-model compari
    
[^21]: 学习撤销：状态空间变换下的迁移强化学习

    Learning to Undo: Transfer Reinforcement Learning under State Space Transformations

    [https://arxiv.org/abs/2611.23081](https://arxiv.org/abs/2611.23081)

    该论文提出通过匹配两个MDP的状态特征统计量来学习“撤销映射”，从而将源MDP的策略迁移到目标MDP，并证明在撤销映射为线性且源满足线性$Q^\star$可实现的条件下，其样本复杂度严格优于在目标MDP中从零开始学习。

    

    强化学习（RL）中的迁移学习已经展现出强大的实证成功。在这项工作中，我们采用更具原则性的视角，研究在什么条件下以及如何在两个MDP（源MDP与目标MDP）之间迁移知识能够被证明是有益的。具体而言，我们考虑存在一个“撤销映射”（undo map）的情形，即将该映射应用于目标的状态空间可以精确地恢复出源MDP。我们提出了一种算法，通过匹配从两个MDP中收集到的状态特征统计量来学习该映射，然后利用它将源策略迁移到目标MDP。我们从理论上论证了该算法：分析了撤销映射为线性且源MDP满足线性-$Q^\star$可实现的设定，在该设定下，我们的方法在目标MDP中具有严格优于从零开始学习（tabula rasa RL）的样本复杂度。在实证方面，我们证明了这些优势可以扩展到该理论设定之外：在具有挑战性的连续控制任务和Atari游戏上，我们的方法……

    arXiv:2611.23081v1 Announce Type: cross  Abstract: Transfer learning in reinforcement learning (RL) has shown strong empirical success. In this work, we take a more principled perspective by studying when and how transferring knowledge between MDPs (a source and a target) can be provably beneficial. Specifically, we consider the case where there exists an undo map such that applying this map to the target's state space recovers the source exactly. We propose an algorithm that learns this map by matching state feature statistics gathered from both MDPs, and then uses it to transfer the source policy to the target. We theoretically justify the algorithm by analyzing the setting when the undo map is linear and the source is linearly-$Q^\star$ realizable, where our approach has strictly better sample complexity than tabula rasa RL in the target MDP. Empirically, we demonstrate that these benefits extend beyond this regime: on challenging continuous control tasks and Atari games, our method
    
[^22]: Colombo：面向基于预言机优化的前沿到前沿自主研究系统

    Colombo: A Frontier-to-Frontier Autoresearch System for Oracle-Based Optimization

    [https://arxiv.org/abs/2611.23080](https://arxiv.org/abs/2611.23080)

    Colombo是一个自组织AI研究系统，它将基于预言机优化中可实现保证与不可避免极限之间的差距转化为自适应研究计划，并通过对抗性共同攀升机制，在算法与困难问题族的相互挑战中持续推动科学前沿的更新。

    

    科学随着发现不断重塑已知并为后续发现奠定基础，从而从一个前沿迈向另一个前沿。我们提出了Colombo，一个推动这种更新的自组织AI研究者，并首先在基于预言机的优化领域中进行实例化。在该领域中，核心问题是通过规定的信息接口能够以多高的效率求解一个问题类，以及任何可容许算法都无法避免的成本是什么。可实现的保证与不可避免的极限之间的差距定义了前沿，并被记录在其“复杂性图集”中——这是一张关于前沿设置与当前知识状态的动态地图。自组织机制将未解决的差距转化为自适应的研究计划，随着证据的积累，协调猜想的形成、数学研究与重新规划。对抗性共同攀升机制通过相互挑战来共同发展算法与困难问题族，揭示信息使用与隐藏的机制，从而引导更强的方……

    arXiv:2611.23080v1 Announce Type: new  Abstract: Science advances from frontier to frontier as discoveries reshape what is known and create the basis for further discovery. We present Colombo, a self-organizing AI researcher that drives this renewal, first instantiated in oracle-based optimization. Here, the central question is how efficiently a problem class can be solved through a prescribed information interface and what costs no admissible algorithm can avoid. The gap between achievable guarantees and unavoidable limits defines the frontier recorded in its Complexity Atlas, a living map of frontier settings and the current state of knowledge. Self-Organization turns unresolved gaps into adaptive research programmes, coordinating conjecture formation, mathematical investigation, and replanning as evidence accumulates. Adversarial Co-Ascent develops algorithms and hard families through reciprocal challenge, exposing mechanisms of information use and concealment that guide stronger me
    
[^23]: 面向智能体治理的多条自然语言准则运行时验证

    Runtime Verification of Multiple Natural Language Criteria for Agent Governance

    [https://arxiv.org/abs/2611.23074](https://arxiv.org/abs/2611.23074)

    提出了一种基于树注意力的并行验证器VFM，可在运行时对智能体输出同时评估数百条自然语言治理准则并给出逐条三元判定，其预训练语料库CriteriaBank带来了跨域性能提升。

    

    AI智能体受数百条自然语言准则的约束，然而现有的验证器无法高效地将智能体的输出与大量此类准则进行比对评估，并报告哪些准则被满足、被违反或不适用。我们提出了VFM，一种树注意力验证器，它对共享上下文和目标文本进行编码，并行地对N条自然语言准则进行评分，返回相互独立的逐准则三元判定结果。VFM在CriteriaBank语料库上进行了预训练，该语料库包含394K个（上下文、目标、准则、判定）元组，涵盖超过321K条不同的准则字符串。经微调的VFM-4B在一个合成治理基准上优于GPT-5.4。在DynaBench多准则基准上，它在轨迹级失败检测方面优于DynaGuard-4B，并且其逐准则判定结果能够定位被违反的规则，支持将问题聚焦升级至更大的思维链评判器。我们证明CriteriaBank预训练可以提升跨域（泛化能力）。

    arXiv:2611.23074v1 Announce Type: cross  Abstract: AI agents are governed by hundreds of natural language criteria, yet existing verifiers cannot efficiently evaluate an agent's output against a large set of such criteria and report which are satisfied, violated, or inapplicable. We introduce the VFM, a tree-attention verifier that encodes the shared context and target text, and scores N natural language criteria in parallel, returning independent per-criterion ternary verdicts. The VFM is pretrained on CriteriaBank, a corpus of 394K (context, target, criterion, verdict) tuples spanning over 321K distinct criterion strings. A finetuned VFM-4B outperforms GPT-5.4 on a synthetic governance benchmark. On the DynaBench multi-criterion benchmark, it outperforms DynaGuard-4B on trace-level failure detection, and its per-criterion verdicts localize the violated rule and support focused escalation to a larger chain-of-thought judge. We show that CriteriaBank pretraining improves cross-domain t
    
[^24]: 分辨率假说：智能体进化的关键在于识别改进

    The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements

    [https://arxiv.org/abs/2611.23068](https://arxiv.org/abs/2611.23068)

    智能体进化的停滞并非因为方法无法提出更好的候选产物，而是因为噪声评估无法可靠地识别出这些改进，使得选择的效果与随机无异。

    

    智能体进化遵循一个简单的循环：提议者撰写候选产物（如提示词、技能、记忆），选择机制则接受得分更高的那些候选产物。这一循环已展现出前景，但实践者面临一个难题：不同方法在各篇论文中互有胜负，且进化在某些任务上常常无法实现自我提升。我们将这一难题归结为分辨率假说：智能体进化停滞，并非因为方法无法提出更好的产物，而是因为带有噪声的评估无法可靠地识别出这些更好的产物。通过跨越七种进化方法和六个模型（超过1000万次LLM调用）的实验，我们发现该假说在所有使用评估得分的两个层面都得到了支持。在候选层面（选择产物）上，有益的候选产物确实存在，但选择机制在区分有益与有害产物方面的表现并不比随机更好，因此误差在多轮迭代中不断累积。在方法层面（对方法进行排名）上，典型的几个百分点的差距小于……

    arXiv:2611.23068v1 Announce Type: new  Abstract: Agent evolution follows a simple loop: a proposer writes candidate artifacts (prompts, skills, memories) and selection accepts those that score higher. This loop has shown promise, yet practitioners face a puzzle: methods trade wins across papers, and evolution often fails to bootstrap on certain tasks. We trace this puzzle to the Resolution Hypothesis: agent evolution stalls not because methods fail to propose better artifacts, but because noisy evaluations cannot reliably identify them. Across seven evolution methods and six models (over 10M LLM calls), we find support for the hypothesis at both levels where evaluation scores are used. At the candidate level (choosing artifacts), helpful candidate artifacts exist, but selection is no better than random at telling them from harmful ones, so errors accumulate across rounds. At the method level (ranking methods), typical margins of a few percentage points are smaller than the difference b
    
[^25]: 价值观塑造自我改进型编程智能体所构建的内容

    Values Shape What Self-Improving Coding Agents Build

    [https://arxiv.org/abs/2611.23067](https://arxiv.org/abs/2611.23067)

    在自我改进型编程智能体的持久指令文件中写入简短的价值观陈述（基于施瓦茨价值观理论），不仅会显著影响其任务表现——开放性价值观能提升留出集表现，而保守性价值观的表现甚至低于无价值观的基线智能体——还会改变智能体最终构建出的成果。

    

    自我改进型智能体通常针对其产出的结果进行优化，而较少关注它们是如何产出这些结果的。它们的指令为设计者提供的塑造其工作方式的手段十分有限，尤其当任务是开放式的时候。本文研究了在智能体的持久化指令文件中加入一段简短的价值观陈述，是否不仅会改变智能体的表现好坏，还会改变它所构建的东西。本文评估了施瓦茨基本人类价值观理论中的四种高阶价值类型：对变化的开放性、保守性、自我增强和自我超越。在多样化的编程任务中，价值观在每次运行的多次自我改进迭代中保持固定。研究得出了三项发现：（1）价值观对多样化任务集的表现很重要。开放性价值观提升了在留出数据上的表现，而与之相反的保守性价值观则导致得分低于未给予任何价值观的智能体。（2）价值观会改变智能体所构建的内容……（原文摘要在此处截断）

    arXiv:2611.23067v1 Announce Type: new  Abstract: Self-improving agents are typically optimized for the results they produce, with less attention to how they produce them. Their instructions give designers limited ways to shape how they work, especially when the task is open-ended. This paper studies whether a short value statement in an agent's persistent instruction file changes not only how well the agent performs but also what it builds. Four higher-order value types in Schwartz's theory of basic human values were evaluated: openness to change, conservation, self-enhancement, and self-transcendence. Across diverse coding tasks, the values were held fixed within each run over multiple iterations of self-improvement. Three findings emerged: (1) Values matter for performance across a diverse set of tasks. The openness values increase performance on held-out data, whereas the opposite conservation values result in scores below those of an agent given no values. (2) Values change what ag
    
[^26]: 面向热泵与地暖控制的深度强化学习：一项端到端案例研究

    Deep Reinforcement Learning for Heat Pump and Underfloor Heating Control: An End-to-End Case Study

    [https://arxiv.org/abs/2611.23063](https://arxiv.org/abs/2611.23063)

    本研究针对热泵与地暖耦合系统提出了基于深度强化学习的端到端控制案例研究，利用真实住宅的数字孪生训练智能体并成功部署于实际建筑中，从而同时优化能耗与热舒适性。

    

    热泵（HP）与地暖（UFH）系统耦合运行的高效控制极具挑战性，其原因在于建筑的高热惯性和地暖系统缓慢的动态响应。深度强化学习（DRL）提供了一种数据驱动的控制方法，能够同时优化能耗与热舒适性。然而，基于DRL的控制器在真实运行条件下的有人居住建筑中的应用仍然有限。本工作针对一个集成的地暖-热泵系统，提出并开展了一项基于DRL控制器的端到端案例研究（涵盖开发、评估与实际部署）。首先，开展了基于仿真的试点研究，以评估不同的DRL算法、控制策略与配置方案。随后，开发了一个真实住宅建筑的数字孪生模型，并用其在模拟运行条件下训练DRL智能体。表现最佳的智能体随后被部署到实际建筑中。

    arXiv:2611.23063v1 Announce Type: new  Abstract: Efficient control of underfloor heating (UFH) systems coupled with heat pumps (HPs) is challenging due to the high thermal inertia of buildings and the slow dynamic response of UFH systems. Deep Reinforcement Learning (DRL) offers a data-driven control approach capable of simultaneously optimizing energy consumption and thermal comfort. However, the application of DRL-based controllers in occupied buildings under real operating conditions remains limited. This work presents and end-to-end case study (development, evaluation, and real-world deployment) of a DRL-based controller for an integrated UFH-HP system. First, a simulation-based pilot study was conducted to assess different DRL algorithms, control strategies, and configurations. Subsequently, a digital twin of a real residential building was developed and used to train DRL agents under simulated operating conditions. The best-performing agent was then deployed in the physical build
    
[^27]: FrameRouter：面向长视频理解的帧预算路由方法——基于Video-MME-v2及更多基准

    FrameRouter: Frame Budget Routing for Long-Video Understanding on Video-MME-v2 and Beyond

    [https://arxiv.org/abs/2611.23055](https://arxiv.org/abs/2611.23055)

    提出无需训练的测试时框架FrameRouter，通过证据需求路由器为每个问题动态分配不同的帧预算（保持平均预算与统一参考一致），从而更有效地提升长视频问答的准确性。

    

    像Video-MME-v2这样的长视频问答基准，通常在为每个问题分配统一帧预算的条件下评估系统，然而各个查询实际所需的视觉证据存在显著差异。我们在Video-MME-v2上分析了这种不匹配现象，发现不同的问题子集各自偏好不同的固定帧预算策略，而简单地统一增加帧数量并不能可靠地改善所有指标。这些发现启发我们提出了“固定均值帧预算路由”这一受控推理设置：每个问题可以分配到不同的帧预算，同时整体平均预算与统一预算的参考设置保持一致。我们提出FrameRouter，一个无需训练的测试时框架，它结合了证据需求路由器、预算约束分配规则以及一组固定的帧采样策略库。通过在保持底层采样器不变的前提下控制每个问题所获得的视觉证据量，FrameRouter与查询（摘要在此处截断）

    arXiv:2611.23055v1 Announce Type: new  Abstract: Long-video question answering benchmarks like Video-MME-v2 typically evaluate systems under a uniform frame budget for every question, despite stark variation in the evidence each query actually demands. We analyze this mismatch on Video-MME-v2 and find that different question subsets prefer different frozen frame-budget policies, while uniformly increasing the frame count does not reliably improve all metrics. These findings motivate fixed-mean frame-budget routing, a controlled inference setting in which each question may receive a different frame budget while the average budget matches a uniform reference. We introduce FrameRouter, a training-free test-time framework that combines an evidence-demand router, a budget-constrained allocation rule, and a frozen bank of frame-sampling policies. By controlling how much visual evidence each question receives while leaving the underlying samplers unchanged, FrameRouter is orthogonal to query-
    
[^28]: ReTouchGS：面向3D高斯泼溅的超轻量级免渲染美学色彩增强与迁移

    ReTouchGS: Ultralight Render-free Aesthetic Color Enhancement and Transfer for 3D Gaussian Splatting

    [https://arxiv.org/abs/2611.23054](https://arxiv.org/abs/2611.23054)

    ReTouchGS是首个面向预训练3D高斯泼溅模型的免参考美学色彩增强框架，通过从模型和相机参数预测场景级颜色查找表，并借助仅44.7K参数的轻量级混合感知细化网络，实现无需渲染的、跨视点一致的自动美学色彩增强与参考引导色彩迁移。

    

    三维高斯泼溅（3DGS）能够实现照片级真实的场景重建。现有的3DGS色彩迁移和交互式编辑方法均由参考图像和用户指定的颜色调整来引导。然而，在许多应用中，用户只是希望获得美学引导的色彩增强，能够自动使场景在视觉上更具吸引力，同时在不同视点间保持颜色和色调的一致性。我们提出了ReTouchGS，据我们所知，这是首个针对预训练3DGS模型的免参考美学色彩增强框架。该方法从模型及其关联的相机参数预测场景级颜色查找表（LUT）。ReTouchGS还支持参考引导的色彩迁移，即利用同一场景的一个编辑视图及其对应的原始渲染来拟合场景级LUT。一个仅含44.7K参数的共享混合感知颜色细化网络对解析颜色更新进行细化，以实现色彩增强或色彩迁移。

    arXiv:2611.23054v1 Announce Type: cross  Abstract: 3D Gaussian Splatting (3DGS) enables photorealistic scene reconstruction. Existing 3DGS color transfer and interactive editing methods are guided by reference images and user-specified color adjustments. However, in many applications, users simply want aesthetically guided color enhancement that automatically makes the scene more visually appealing while maintaining consistent colors and tones across viewpoints. We present ReTouchGS, to our knowledge the first framework for reference-free aesthetic color enhancement of pretrained 3DGS models. It predicts a scene-level color lookup table (LUT) from the model and its associated camera parameters. ReTouchGS also supports reference-guided color transfer by fitting a scene-level LUT using one edited view of the same scene and its corresponding original render. A shared blend-aware color refinement network with only 44.7K parameters refines an analytic color update to realize either color ma
    
[^29]: ForecastGym：一个面向预测智能体的实时、可控且可回放的基准测试

    ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents

    [https://arxiv.org/abs/2611.23045](https://arxiv.org/abs/2611.23045)

    ForecastGym 通过每两天记录四个商业搜索引擎 API 对预测市场问题的返回结果，并在结果揭晓后离线回放这些冻结证据，首次实现了在固定证据获取策略下对预测智能体的模型、记忆与搜索来源的公平、可控比较。

    

    基于大语言模型（LLM）构建的智能体越来越多地通过搜索网络并对所找到的信息进行推理来预测真实世界事件。此类智能体的预测好坏既取决于检索到的证据，也取决于模型本身，因此对证据的控制对于评估至关重要。当每个系统各自进行搜索时，实时评估会将检索与推理混杂在一起，而回溯式搜索则可能通过日期过滤器或历史页面在当今的排名泄露后续信息。要比较不同的模型、搜索来源和记忆策略，既需要固定的证据，也需要该证据在问题揭晓前如何演变的记录。我们提出了 ForecastGym，它每两天记录四个商业网络搜索 API 针对开放预测市场问题所返回的结果，并在结果揭晓后离线回放这些冻结的证据，从而在固定的证据获取策略下比较模型、记忆以及所捕获的搜索来源。其首个发布版本涵盖1（摘要在此处截断）

    arXiv:2611.23045v1 Announce Type: new  Abstract: Agents built on large language models (LLMs) increasingly forecast real-world events by searching the Web and reasoning over what they find. How well such an agent forecasts depends on both the retrieved evidence and the model, making control over that evidence essential for evaluation. Live evaluations can mix retrieval with reasoning when each system searches on its own, while retrospective search can leak later information through date filters or today's ranking of historical pages. Comparing models, search sources, and memory strategies requires both fixed evidence and a record of how it evolved before resolution. We introduce ForecastGym, which records what four commercial web search APIs return for open prediction-market questions every two days and replays this frozen evidence offline once outcomes are known, comparing models, memory, and captured search sources under a fixed evidence-acquisition policy. Its first release covers 1
    
[^30]: 检查你的盲区：通过假设物体探测规划器实现的决策引导感知

    Check Your Blind Spots: Decision-Guided Perception by Probing the Planner with Hypothetical Objects

    [https://arxiv.org/abs/2611.23030](https://arxiv.org/abs/2611.23030)

    提出决策引导推测感知（DGSP），通过向低分辨率草稿场景中插入假设物体来探测规划器的“决策敏感度场”，仅对可能改变规划动作的区域及其支撑检测结果进行高分辨率验证，从而以远低于全帧高分辨率的开销获得与全高分辨率完全一致的规划动作。

    

    并非每一个感知错误都会改变规划器的动作。然而，自适应分辨率感知方法基于图像线索来决定将高分辨率计算资源投向何处，这既无法可靠地指出低分辨率所遗漏的物体，也无法判断哪些物体对规划器而言是重要的。我们观察到，规划器可以在任何物体被实际看到之前说出物体在何处会产生影响：在某个位置插入一个假设物体，就能测试规划器的动作是否会因此改变。决策引导推测感知（DGSP）将低分辨率场景及其动作视为一份草稿，并以这种方式探测规划器，从而获得一个“决策敏感度场”，即未被观测到的物体会改变动作的位置。只有这个敏感度场以及“支撑集”（当前动作所依赖的检测结果）才需要以高分辨率进行验证，并反复迭代直到动作稳定为止。对于两类规划器，我们证明DGSP能够返回与全帧高分辨率感知完全一致的动作。

    arXiv:2611.23030v1 Announce Type: cross  Abstract: Not every perception error changes the planner's action. Yet adaptive-resolution perception chooses where to spend high-resolution computation from image cues, which can neither reliably point to an object that low resolution missed nor tell which objects matter to the planner. We observe that the planner can say where an object would matter before anything is seen: inserting a hypothetical object at a location tests whether its action would change. Decision-Guided Speculative Perception (DGSP) treats the low-resolution scene and its action as a draft and probes the planner in this way to obtain a decision-sensitivity field, the locations where an unseen object would change the action. Only this field and the support, the detections on which the current action rests, are verified at high resolution, repeatedly until the action is stable. For two planner classes, we prove that DGSP returns exactly the action of whole-frame high-resoluti
    
[^31]: PeriaTrace：基于根尖周X光片的牙级定位诊断与面向患者的解释

    PeriaTrace: Tooth-Level Grounding Diagnosis and Patient-Facing Explanations from Periapical Radiographs

    [https://arxiv.org/abs/2611.23012](https://arxiv.org/abs/2611.23012)

    该论文提出PeriaTrace三阶段框架，能够从根尖周X光片中实现牙级定位诊断、生成带有视觉定位的自动化报告，并提供面向患者的通俗解释与护理建议，在内部和外部临床数据上均表现出色。

    

    根尖周X光片（PRs）是牙髓病诊断的核心，但解读这些影像和撰写报告非常耗时。专业术语可能限制患者的理解，而现有的自动化方法往往缺乏检查发现与图像证据之间的牙级关联。我们提出了PeriaTrace，这是一个三阶段框架，集成了牙级诊断、视觉定位报告生成以及面向患者的友好解释和护理建议。PeriaTrace使用来自内部数据集和两个外部临床队列的12,877张根尖周X光片进行开发和评估。在内部评估中，它取得了85.35%的宏观Dice系数、0.6504的BLEU-4分数、牙齿定位与国际牙科联合会（FDI）编号识别联合任务的94.37% F1分数，以及77.93%的定位标签F1分数。在两个外部队列中，资深牙髓病专家认为分别有91.16%和84.52%的报告适合保留或仅需修改。

    arXiv:2611.23012v1 Announce Type: new  Abstract: Periapical radiographs (PRs) are central to endodontic diagnosis, but interpreting them and preparing reports are time-consuming. Technical language can limit patients' understanding, while existing automated methods often lack tooth-level links between findings and image evidence. We present PeriaTrace, a three-stage framework integrating tooth-level diagnosis, visually grounded report generation, and patient-friendly explanations and care recommendations. PeriaTrace was developed and evaluated using 12,877 PRs from an internal dataset and two external clinical cohorts. In internal evaluation, it achieved a macro Dice coefficient of 85.35%, BLEU-4 of 0.6504, an F1 score of 94.37% for joint tooth localization and F\'ed\'eration Dentaire Internationale (FDI) identification, and a grounding-tag F1 score of 77.93%. In the two external cohorts, senior endodontists considered 91.16% and 84.52% of reports suitable for retention or revision wit
    
[^32]: PhysScope：基于接地视觉记忆的智能体多模态物理推理

    PhysScope: Agentic Multimodal Physics Reasoning with Grounded Visual Memory

    [https://arxiv.org/abs/2611.23010](https://arxiv.org/abs/2611.23010)

    PhysScope 是一个推理时的视觉辅助框架，通过将物理约束与视觉观察直接关联并维护范围感知的工作记忆，使视觉-语言模型无需更新权重即可自我修正物理推理中的范围错配错误。

    

    视觉-语言模型（VLM）虽然能够正确解读物理示意图，却常常在这些约束所适用的状态或条件之外应用它们，导致前提错误在后续计算中不断传播。我们提出了 PhysScope，这是一个推理时的视觉辅助框架，无需更新模型权重即可支持对这类范围错配的自我修正。PhysScope 使模型能够将物理范围直接关联到其视觉观察，并显式构建派生视图以支持推理。该框架在模型推理过程中对状态区分、条件和物理范围施加显式约束，并依据来源和范围对智能体提出的修改进行核查。被接受的修改会使已注册的依赖项失效，从而维持具有范围感知的工作记忆。在 PhysAlign 基准上，PhysScope 在所有六个被评估的 VLM 中均取得了比直接求解更高的解题分数。使用 Gemini-3.8-flash 时，其表现优于 Canvas-CoT 基线……

    arXiv:2611.23010v1 Announce Type: new  Abstract: Vision-language models (VLMs) can correctly interpret physics diagrams yet apply constraints outside the states or conditions in which they hold, causing premise errors to propagate through subsequent calculations. We introduce PhysScope, an inference-time visual harness that supports self-correction of these scope mismatches without updating model weights. PhysScope enables the model to link physical scopes directly to its visual observations and explicitly constructs derived views to support reasoning. The harness imposes explicit constraints on state distinctions, conditions, and physical scopes during model reasoning, and checks agent-proposed revisions against sources and scopes. Accepted revisions invalidate registered dependents, thereby maintaining scope-aware working memory. On PhysAlign, PhysScope achieves higher solving scores than direct solving across all six evaluated VLMs. With Gemini-3.8-flash, it outperforms Canvas-CoT b
    
[^33]: 基于冻结世界模型的残差引导轻量级强化学习风险感知轨迹选择

    Residual-Guided Lightweight Reinforcement Learning for Risk-Aware Trajectory Selection with a Frozen World Model

    [https://arxiv.org/abs/2611.22998](https://arxiv.org/abs/2611.22998)

    该论文提出一种基于冻结世界模型的残差引导轻量级强化学习框架，仅需微调约1.6万参数即可实现自动驾驶中风险感知的高效轨迹选择。

    

    在自动驾驶中，世界模型为评估候选轨迹提供未来场景预测，但固定的选择规则可能无法捕捉依赖于场景的权衡。我们提出了一种残差引导的强化学习（RL）框架，用于在冻结世界模型和固定候选库的基础上进行轻量级轨迹选择。该框架使用近端策略优化（PPO）来微调覆盖候选选择和有界轨迹残差的联合策略。一个上下文条件排序网络通过评估所选轨迹的局部扰动所获得的反馈来学习候选偏好。在推理阶段，选择器使用固定权重结合学习到的偏好排名和预测的风险排名，并返回所选的候选轨迹而不进行修改。微调仅更新actor和critic中总共16,106个参数，而部署的选择器仅保留11,725个参数的排序网络……

    arXiv:2611.22998v1 Announce Type: new  Abstract: In automated driving, world models provide future scene predictions for evaluating candidate trajectories, but fixed selection rules may not capture scene-dependent trade-offs. We propose a residual-guided reinforcement learning (RL) framework for lightweight trajectory selection with a frozen world model and a fixed candidate bank. The framework uses proximal policy optimization (PPO) to fine-tune a joint policy over candidate selection and bounded trajectory residuals. A context-conditioned ranking network learns candidate preferences from feedback obtained by evaluating local perturbations of the selected trajectories. At inference, the selector combines learned preference and predicted risk ranks using fixed weights and returns the selected candidate trajectory without modification. Fine-tuning updates 16,106 parameters across the actor and critic, while the deployed selector retains only the 11,725-parameter ranking network, with th
    
[^34]: RagHarnessDebugger：契约推理修复审计中基于Jev引导的智能体RAG框架

    RagHarnessDebugger: Jev-Guided Agentic RAG in Contract Inference Repair Auditing

    [https://arxiv.org/abs/2611.22991](https://arxiv.org/abs/2611.22991)

    提出了RagHarnessDebugger框架，通过Jev引导的测试工具记录并审计智能体RAG在契约推理修复过程中的全部决策，从而暴露出被准确率等聚合指标所掩盖的失败模式。

    

    使用检索增强生成（RAG）的智能体系统必须通过收集更多证据、测试替代策略，并决定是否、何时以及需要何种修复来自行修正其答案。然而，当准确率等指标可能具有误导性时，我们如何知道系统的修复是否真正有效？我们提出了RagHarnessDebugger，这是一个在Jev引导的测试框架中记录契约推理的每一个修复决策，并对这些记录进行审计的框架，从而能够对RAG修复过程中采取的所有决策进行完整调试。RagHarnessDebugger检查每条轨迹是否完整、每个验证是否适用于其所统计的候选对象，以及接受规则能够覆盖哪些答案。我们在ContractNLI保密协议任务上使用两个阅读器模型以及一组独立合同构成的复现集对我们的方法进行了测试。我们的调试器审计层暴露了隐藏在聚合指标之下的失败模式，以及策略接受成本方面的…

    arXiv:2611.22991v1 Announce Type: new  Abstract: Agentic systems using retrieval-augmented generation (RAG) must repair their own answers by gathering more evidence, testing alternative policies and deciding if, when, and what repairs are needed. Yet how do we know if a system's repairs are working when metrics like accuracy can be misleading? We present RagHarnessDebugger, a framework that records every repair decision in a Jev-guided harness for contract inference and audits the record to facilitate complete debugging of all decisions taken during a RAG repair journey. RagHarnessDebugger checks that each trace is intact, that each validation applies to the candidate it is counted for, and which answers the acceptance rule can reach. We tested our approach on ContractNLI non-disclosure agreements with two reader models and a replication set of separate contracts. Our debugger's audit layer exposed failure patterns that were hidden under aggregate metrics, policies acceptance cost of r
    
[^35]: 网络研究智能体如何解读用户优先级

    How Web Research Agents Interpret User Priorities

    [https://arxiv.org/abs/2611.22989](https://arxiv.org/abs/2611.22989)

    该论文研究了基于OpenAI和Anthropic模型的网络研究智能体在用户仅给出模糊优先级时如何自行选择评判标准，发现尽管智能体大多会聚焦于声明的优先级，但其采用的标准差异很大，甚至导致在微软与Alphabet的比较中得出截然相反的结论。

    

    arXiv:2611.22989v1 公告类型：cross 摘要：网络研究智能体在线收集信息并撰写带引用的报告，包括金融分析。用户可以声明一个优先级来指导这项研究，但优先级未必说明如何评判各个选项，这就留给智能体自行选择评判标准。我们考察了两个此类智能体（分别基于OpenAI和Anthropic的模型构建）如何解读用户声明的优先级。在36次运行中，智能体对微软和Alphabet这两家公司进行了比较。提示词要求特别关注增长或韧性，或者不给出任何优先级。我们分析了报告、搜索、返回的来源以及被引用的来源。在给定优先级的24份报告中，有21份聚焦于该优先级，但它们所采用的评判标准各不相同。在增长优先级下，OpenAI的报告在为公司排名时考虑了风险或可预测性，因而更青睐微软，而Anthropic的报告则以增长速度来评判增长，因而更青睐Alphabet。在韧性优先级下，各报告使用了部分不同的标准，且大多数将微软列为……（原文在此处截断）

    arXiv:2611.22989v1 Announce Type: cross  Abstract: Web research agents gather information online and write cited reports, including financial analysis. Users can state a priority to guide this research, but a priority may not say how to judge the options, leaving the agent to choose the criteria. We examine how two such agents, built on OpenAI and Anthropic models, interpret a stated priority. In 36 runs, the agents compared Microsoft and Alphabet as businesses. Prompts asked for special attention to growth or resilience, or gave no priority. We analyzed reports, searches, returned sources and cited sources. Of the 24 reports given a priority, 21 focused on it, but their criteria varied. Under the growth priority, OpenAI reports that ranked the companies accounted for risk or predictability and favored Microsoft, while the Anthropic ones judged growth by its pace and favored Alphabet. Under the resilience priority, reports used partly different criteria, and most named Microsoft as the
    
[^36]: 结构化AI构思为上皮屏障修复提供物理策略

    Structured AI ideation yields physical strategies for epithelial barrier repair

    [https://arxiv.org/abs/2611.22988](https://arxiv.org/abs/2611.22988)

    该研究开发了结构化AI构思系统Abelio，通过重新表述研究目标、跨学科检索要素并迭代细化假设，生成了在科学合理性、多样性和新颖性上超越前沿大语言模型的研究方案，并为上皮屏障修复提出了新的物理修复策略。

    

    AI系统日益支持文献分析、候选方案评估和假设细化，但让研究人员能够探索超越既定学科方法的研究方向仍具有挑战性。我们开发了Abelio，一个旨在通过拓展研究人员的探究途径来生成新颖研究方向的AI系统。Abelio会对研究目标进行重新表述，利用这些替代表述来识别传统学科边界之外的相关要素，并对由此产生的假设进行细化。每个阶段都通过专门的子系统将发散性探索与收敛性选择相结合。在定量评估中，由构思核心（Ideation Core，即Abelio的假设生成核心）生成的方案在科学合理性和假设多样性方面优于四个前沿大语言模型，且在与现有文献的距离方面优于其中三个。子系统消融实验表明，每个子（系统均发挥关键作用）……

    arXiv:2611.22988v1 Announce Type: new  Abstract: AI systems increasingly support literature analysis, candidate evaluation and hypothesis refinement, but enabling researchers to explore research directions beyond established disciplinary approaches remains challenging. We developed Abelio, an AI system designed to generate novel research directions by expanding investigators' avenues of inquiry. Abelio reformulates research objectives, uses these alternative formulations to identify relevant elements outside conventional disciplinary boundaries, and refines the resulting hypotheses. Each stage integrates divergent exploration with convergent selection through a dedicated subsystem. In quantitative evaluations, proposals generated by the Ideation Core (Abelio's hypothesis-generation core) outperformed those of four frontier large language models in scientific soundness and hypothesis diversity, and three of four in distance from existing literature. Subsystem ablation showed that each s
    
[^37]: 大语言模型尚未改变科学

    Large Language Models Have Not Yet Transformed Science

    [https://arxiv.org/abs/2611.22975](https://arxiv.org/abs/2611.22975)

    该研究通过分析2015至2025年间五个学科250万篇全文论文发现，除个别例外，大语言模型出现后科学实践的大多数既有趋势仍在延续，表明LLM尚未从根本上改变科学研究的方式。

    

    2022年底大语言模型（LLMs）的出现引发了对科学未来的诸多疑问，但要评估其影响，需要理解科学实践此前是如何演变的。我们分析了2015年至2025年间五个学科的250万篇全文论文，在词汇、结构、认识论和社会等多个维度上追踪了广泛的变化趋势。除了一些词汇频率的变化外，大多数2023年之前的趋势仍在延续，可检测到的中断十分有限。在少数显著的例外中，亚洲计算机科学领域合作团队的快速增长明显加速，而生物医学领域主题多样性的明显下降则主要反映了论文格式的标准化。我们的研究为在大语言模型应用日益深入的背景下监测全球科学的健康状况奠定了实证基础。

    arXiv:2611.22975v1 Announce Type: cross  Abstract: The emergence of large language models (LLMs) in late 2022 has raised questions about the future of science, but assessing their impact requires understanding how scientific practices were already evolving. We analyze 2.5 million full-text papers across five disciplines from 2015 to 2025, tracing trends over a broad spectrum of lexical, structural, epistemological, and social dimensions. Aside from frequency shifts in some words, most pre-2023 trends continued with limited detectable breaks. Among the notable exceptions, growth in computer science coauthor teams accelerated significantly in Asia, while apparent declines in biomedical topic diversity largely reflected standardization in article formats. Our study establishes an empirical foundation for monitoring the health of global science as LLM adoption deepens.
    
[^38]: VITAL：用于验证制药药物研发与临床试验运营中智能体AI的框架

    VITAL: A Framework for Validating Agentic AI in Pharmaceutical Drug Development and Clinical Trial Operations

    [https://arxiv.org/abs/2611.22973](https://arxiv.org/abs/2611.22973)

    本文提出了VITAL验证框架，通过整合使用情境、控制措施、行为验证和生命周期证据，在风险调整框架内对制药药物研发和临床试验运营中的智能体AI进行验证、监控与治理。

    

    当前的保障方法无法确定证据是否足以在制药使用情境下对智能体人工智能进行验证、监控和治理。看似合理的输出可能掩盖检索、推理、编排或人工控制环节中的失败，尤其是在缺乏参考标准的情况下。我们通过现状分析、法规与标准映射、专家访谈以及多利益相关者的迭代完善，开发并实施了VITAL（基于预期用途、可追溯性、自主性和生命周期证据的验证）框架，这是一个面向临床开发中智能体的验证框架。VITAL在一个基于风险调整的框架内整合了使用情境、控制措施、行为验证和生命周期证据。我们将VITAL应用于流行病学与治疗格局分析、研究中心选择、入组拯救以及试验设计权衡支持等场景，以将自主性、可逆性、透明度和患者影响等维度落实到具体实践中。

    arXiv:2611.22973v1 Announce Type: new  Abstract: Current assurance approaches do not establish whether evidence is sufficient to validate, monitor and govern agentic artificial intelligence (AI) within a pharmaceutical context of use. Plausible outputs may conceal failures in retrieval, reasoning, orchestration or human-control points, where reference standards are unavailable. We developed and operationalised VITAL (Validation by Intended use, Traceability, Autonomy and Lifecycle evidence), a validation framework for agents in clinical development, through landscape analysis, regulatory and standards mapping, expert interviews and multi-stakeholder refinement. VITAL integrates context of use, controls, behavioural validation and lifecycle evidence within a risk-adjusted framework. We applied VITAL across epidemiology and treatment-landscape analysis, site selection, enrolment rescue and trial-design trade-off support to translate autonomy, reversibility, transparency and patient impac
    
[^39]: SEAM：计算机使用智能体知晓屏幕之间的危害，却仍然实施它

    SEAM: Computer-Use Agents Know the Harm Between Screens, and Commit It Anyway

    [https://arxiv.org/abs/2611.22971](https://arxiv.org/abs/2611.22971)

    SEAM提出了一个包含600个多步骤跨应用任务的安全基准，其中每条指令和每个单独屏幕的观察内容都是良性的，只有跨屏幕组合后才会产生危害，从而揭示了计算机使用智能体在组合不同屏幕信息时可能造成的合成性安全风险。

    

    针对计算机使用智能体（CUA）的安全基准通常将危害置于任务意图之中或单个屏幕之上：一条恶意指令、一个注入的命令、一个钓鱼页面。一个孤立地检查任务和每个屏幕的智能体可以通过这些基准，因此它们无法测试智能体是否会将不同屏幕上看到的事实组合起来。然而在委托执行的工作中，危害往往正是这样产生的：一个屏幕上是关于某个人的信息，另一个屏幕上是关于某个物品的信息，再加上一个看似合法的任务将它们绑定在一起。我们提出了SEAM（跨多屏幕的安全执行，Safety Execution Across Multiple screens），这是一个针对此类跨屏幕组合性危害的基准：600个多步骤、跨应用程序的OSWorld任务，其中指令和每一次单独的观察都是良性的，只有它们的组合才是有害的。仅在一处存在差异的最小配对变体，通过服务器日志而非LLM评判者进行评分，可以精确定位智能体在何处失败。通用型开源权重与旗舰级智能体……（原文摘要在此处截断）

    arXiv:2611.22971v1 Announce Type: cross  Abstract: Safety benchmarks for computer-use agents (CUAs) place the harm either in the intent of the task or on a single screen: a malicious instruction, an injected command, a phishing page. An agent that checks the task and each screen in isolation can pass them, so they do not test whether it combines facts seen on different screens. Yet in delegated work, harm often arises exactly this way: a fact about a person on one screen, a fact about an item on another, and a legitimate task that binds them. We introduce SEAM (Safety Execution Across Multiple screens), a benchmark of such cross-screen compositional harm: 600 multi-step, cross-application OSWorld tasks in which the instruction and every single observation are benign and only their combination is harmful. Minimal-pair variants that differ in one place, scored from the server log rather than by an LLM judge, locate where an agent fails. General-purpose open-weight and flagship agents car
    
[^40]: CodeReel：通过精确帧复用与阻塞感知调度加速代码生成视频智能体

    CodeReel: Accelerating Code-to-Video Agents with Exact Frame Reuse and Blocking-Aware Scheduling

    [https://arxiv.org/abs/2611.22967](https://arxiv.org/abs/2611.22967)

    CodeReel无需训练，通过按绘制命令精确复用已绘制的帧，并采用优先运行检查、以空闲优先级并行启动渲染的阻塞感知调度，大幅削减代码生成视频智能体中因构建阻塞而浪费的时间。

    

    编程智能体通过在“构建—检查—编辑”循环中编写和完善程序来制作视频。一部完成的影片平均需要2.85美元的模型调用费用和11分钟，其中超过三分之一的时间耗费在阻塞智能体的构建上。我们发现其中大量时间是可以避免的，原因有二：第一，在会话后期影片的帧中，有37.5%的帧此前已被绘制过，其中许多只是在时间上有所偏移或位于被编辑过的场景内；第二，一半的构建会在某个检查点停止，此时智能体只需要检查结果，但哪些构建会停止在检查完成之前是无法得知的。我们提出了CodeReel，这是一个无需训练的执行层，运行在智能体现有命令的背后。精确帧复用通过每帧的绘制命令为其建立键值（与其在时间轴上的位置无关），并复用任何键值再次出现的帧。阻塞感知调度优先运行检查任务，并同时以空闲优先级启动渲染，除非会话近期的构建情况表明检查不太可能通过。调度利用CPU……（摘要原文在此处被截断）

    arXiv:2611.22967v1 Announce Type: cross  Abstract: Coding agents make videos by writing and refining programs in a build--inspect--edit loop. A finished film costs a median of $2.85 in model calls and 11 minutes, over a third of that time in builds that block the agent. We find that much of this time is avoidable, for two reasons. First, 37.5% of the frames of a session's later films were drawn before, many shifted in time or inside an edited scene. Second, half of the builds stop at a check, where the agent needs only the verdict, but which builds stop is unknown until the checks finish. We present CodeReel, a training-free execution layer behind agents' existing commands. Exact frame reuse keys each frame by its drawing commands, regardless of its position in time, and reuses each frame whose key recurs. Blocking-aware scheduling runs the checks first and starts the render alongside them, at idle priority, unless the session's recent builds make a pass unlikely. Scheduling spends CPU
    
[^41]: 基于扩散模型引导采样的高维黑盒优化

    High-Dimensional Black-Box Optimization via Guidance Sampling of Diffusion Models

    [https://arxiv.org/abs/2611.22965](https://arxiv.org/abs/2611.22965)

    该论文提出了BReaD框架，利用扩散模型的引导采样技术，在不需要大量标注数据或精确奖励梯度的情况下，有效解决高维黑盒优化问题。

    

    许多现实世界的优化问题涉及对黑盒目标函数进行优化，而这类目标只能获得有限次数的评估。贝叶斯优化（BO）是解决此类问题的强大框架，但其有效性在高维空间中往往会出现下降。近期的研究利用扩散模型进行黑盒优化，通常通过预训练一个奖励条件化的得分网络来引导生成过程向高奖励样本偏移。另一类基于梯度的引导方法则假设可以访问一个可微分的奖励函数，从而查询引导梯度。然而，这些方法要么需要大量标注数据进行离线训练，要么需要获取精确的奖励梯度，这限制了它们在黑盒场景中的适用性。我们提出了BReaD（黑盒奖励引导的扩散模型采样），这是一个用于优化黑盒奖励的引导采样框架，无需假设能够访问标注数据。

    arXiv:2611.22965v1 Announce Type: cross  Abstract: Many real-world optimization problems involve optimizing black-box objectives for which only a limited number of evaluations are available. Bayesian optimization (BO) is a powerful framework for such problems, but its effectiveness often deteriorates in high-dimensional spaces. Recent work leverages diffusion models for black-box optimization, often through pre-training a reward-conditioned score network to steer generation towards high-reward samples. Alternatively, gradient-based guidance methods assume access to a differentiable reward function from which guidance gradients are queried. However, these approaches either require large amounts of labeled data for offline training or access to exact reward gradients, limiting their applicability in black-box settings. We propose BReaD (Black-box Reward-guided Sampling of Diffusion Model), a guidance sampling framework for optimizing black-box rewards without assuming access to a labeled
    
[^42]: 用于端到端图像生成的层级扩散Transformer

    Hierarchical Diffusion Transformer for End-to-End Image Generation

    [https://arxiv.org/abs/2611.22962](https://arxiv.org/abs/2611.22962)

    提出层级扩散Transformer（HiT），通过逐步将相关token分组压缩来构建视觉层级结构，使端到端像素空间扩散模型在生成质量和训练效率上追赶领先的两阶段潜在扩散模型。

    

    像素空间扩散模型在单一的端到端训练阶段中联合学习视觉表示和图像生成，而无需单独预训练的视觉编码器。然而，朴素的像素空间Transformer在生成质量和训练效率方面仍然落后于领先的两阶段潜在扩散模型。我们假设这种差距部分源于在整个骨干网络中每个图像块始终保留一个token的做法，这种方式没有明确鼓励一种能在逐步粗化的层级上总结相关内容的视觉层级结构。为解决这一问题，我们提出了层级扩散Transformer（HiT），它逐步地将相关的token进行分组，并用单个token来表示每个组。主Transformer瓶颈部分处理这些紧凑的组token，而解除分组操作则将特征恢复到其原始的图像位置。压缩预算机制调节整个批次中的平均压缩率，使组成员关系和数量能够……（摘要在此处截断）

    arXiv:2611.22962v1 Announce Type: cross  Abstract: Pixel-space diffusion learns visual representations and image generation jointly in a single end-to-end training stage, without a separately pretrained visual encoder. However, plain pixel-space Transformers still trail leading two-stage latent diffusion models in generation quality and training efficiency. We hypothesize that part of this gap arises from retaining one token per patch throughout the backbone, which does not explicitly encourage a visual hierarchy that summarizes related content at progressively coarser levels. To address this, we introduce Hierarchical Diffusion Transformers (HiT), which progressively group related tokens and represent each group with a single token. The main Transformer bottleneck processes these compact group tokens, and ungrouping restores features to their original image positions. A compression budget regulates the average compression ratio across the batch, allowing group memberships and counts t
    
[^43]: 面向通信高效且拜占庭容错联邦学习的投影罚函数梯度追踪方法

    Projected Penalty-Gradient Tracking for Communication-Efficient Byzantine-Resilient Federated Learning

    [https://arxiv.org/abs/2611.22929](https://arxiv.org/abs/2611.22929)

    提出一种投影罚函数梯度追踪框架，使联邦学习在压缩通信条件下仍能保持拜占庭攻击鲁棒性，并给出两种高效算法。

    

    我们研究了在异构数据集上、存在能够上传任意信息的拜占庭攻击者情形下的通信高效联邦学习问题。基于罚函数的方法通过罚函数的有界梯度来约束每个工作节点的影响；然而，一旦上传内容被压缩，这种约束就会失效，因为拜占庭工作节点可以在服务器端积累无界的状态。为了在压缩上传的条件下保持拜占庭鲁棒性，我们提出了一个投影罚函数梯度追踪框架：每个诚实工作节点上传其罚函数梯度的压缩残差，服务器则将每个重构状态投影到已知的罚函数梯度球上。该框架适用于任何收缩型压缩器，并衍生出两种算法：具有一致稀疏化的容错近端梯度算法（FRPG-CS），其共享掩码由新抽取的保留行加以保护；以及惰性聚合的FRPG（L……（摘要原文在此处截断）

    arXiv:2611.22929v1 Announce Type: cross  Abstract: We investigate communication-efficient federated learning over heterogeneous datasets in the presence of Byzantine adversaries capable of uploading arbitrary information. Penalty-based methods bound the influence of each worker through the bounded gradient of a penalty function; however, the bound is lost once the uploads are compressed, since a Byzantine worker can accumulate an unbounded state at the server. To retain the Byzantine resilience under compressed uploads, we propose a projected penalty-gradient tracking framework, where each honest worker uploads a compressed residual of its penalty gradient and the server projects each reconstructed state onto the known penalty-gradient ball. The framework accommodates any contractive compressor and yields two algorithms: fault-resilient proximal gradient with consistent sparsification (FRPG-CS), whose shared mask is protected by freshly drawn reserve rows, and lazily aggregated FRPG (L
    
[^44]: 类型化决策模型何时在交通领域有效？来自出行选择与交通信号控制的证据

    When Do Typed Decision Models Work in Transportation? Evidence from Travel Choice and Traffic Signal Control

    [https://arxiv.org/abs/2611.22926](https://arxiv.org/abs/2611.22926)

    该论文系统评估了类型化决策模型在出行方式选择预测与交通信号控制两类交通任务中的表现，发现在Swissmetro数据集上其预测准确率低于传统离散选择模型和最强的对话式大语言模型基线，表明此类模型的有效性高度依赖于具体任务类型。

    

    类型化决策模型将自然语言理解与结构化接口相结合，在不生成冗长文本回复的情况下返回所选动作以及选项级别的概率分布。其低推理成本和响应延迟使其对于具有预定义备选方案的交通问题颇具吸引力。本文研究了这类模型在两个代表性交通任务中的有效性、鲁棒性和效率：个体出行方式选择预测和序贯交通信号控制。我们系统地将TypeSafe Jev和GPT-6 Luna Decisions与传统交通方法、强化学习算法以及四个基于对话的大语言模型进行了对比评估。在Swissmetro数据集上的实验表明，类型化决策模型的预测准确率低于传统离散选择模型和最强的基于对话的基线模型，且其性能存在显著差异。

    arXiv:2611.22926v1 Announce Type: cross  Abstract: Typed decision models combine natural-language understanding with a structured interface that returns a selected action and an option-level probability distribution without generating lengthy textual responses. Their low inference cost and response latency make them attractive for transportation problems with predefined alternatives. This paper investigates their effectiveness, robustness, and efficiency in two representative transportation tasks: individual travel-mode choice prediction and sequential traffic signal control. We systematically evaluate TypeSafe Jev and GPT-6 Luna Decisions against conventional transportation methods, reinforcement learning algorithms, and four chat-based large language models. Experiments on the Swissmetro dataset show that typed decision models achieve lower prediction accuracy than traditional discrete choice models and the strongest chat-based baseline, while their performance varies substantially w
    
[^45]: 已完成但仍继续工作：编码智能体何时应该停止的即时评估

    Done, But Still Working: Anytime Evaluation of When Coding Agents Should Stop

    [https://arxiv.org/abs/2611.22901](https://arxiv.org/abs/2611.22901)

    该论文提出“即时评估”方法，通过重放编码智能体的运行轨迹并用隐藏测试评分中间补丁，发现智能体在自测通过后继续工作的价值因模型而异（潜在节省1%至57%），且验证了“相同补丁两次通过后即停止”的规则可节省23%的成本但未证明质量不劣化。

    

    编码智能体在自身测试通过后常常会继续工作。这些额外的工作是否改进了补丁？提前停止又能节省多少成本？我们通过即时评估来研究这些问题：重放已记录的运行轨迹，用隐藏测试对中间补丁进行评分，并测量每一步产生的成本。对于仅停止运行或选择较早补丁的策略，这种方法可以在不生成新轨迹的情况下进行配对比较，前提是重放能够保留相关状态。通过对八个开源模型在两种脚手架框架上于SWE-bench Verified数据集进行的约6000次运行的分析，我们发现持续工作的价值存在显著差异。在自测通过时可获得的回顾性节省幅度从不足1%到57%不等，具体取决于模型和脚手架。一条“相同补丁两次通过后即停止”的规则在预注册的复制实验中节省了23%的成本，但预注册的质量测试并未确立其非劣性。我们解释了……

    arXiv:2611.22901v1 Announce Type: new  Abstract: Coding agents often continue working after their own tests pass. Does this additional work improve the patch, and how much could be saved by stopping earlier? We study these questions through anytime evaluation: replaying recorded runs, scoring intermediate patches with hidden tests, and measuring the cost incurred at each step. For policies that only stop a run or select an earlier patch, this allows paired comparisons without generating new trajectories, provided replay preserves the relevant states. Across about 6,000 runs of eight open models in two scaffolds on SWE-bench Verified, we find substantial differences in the value of continued work. The hindsight saving available at passing self-tests ranges from under 1% to 57%, depending on the model and scaffold. A rule that stops after the same patch passes twice saves 23% in a preregistered replication, but the preregistered quality test does not establish non-inferiority. We explain
    
[^46]: 为任务而生：以前瞻性智能体计算重新思考量子化学

    Born for the Task: Rethinking Quantum Chemistry with Foresighted Agentic Computing

    [https://arxiv.org/abs/2611.22897](https://arxiv.org/abs/2611.22897)

    BORN是一个前瞻性智能体计算框架，通过在执行前预测各计算策略的精度与运行时间，将现有量子化学软件包的预设方案拆解为可配置组件，从而按需自动合成并选出误差容限内最快的高性能计算方案。

    

    量子化学依赖于通用软件，而其通用性导致许多大规模计算超出了实际可承受的资源预算。编码智能体使得编写面向特定任务的程序变得廉价，但要轻易找到高效的程序却并不容易：一个计算策略是否足够精确且快速，只有当它真正在目标规模下运行之后才能明确。我们提出了BORN，一个前瞻性智能体计算框架，可按需合成高性能量子化学程序。BORN在执行之前预判每种策略的行为：从具有代表性的小体系中外推其误差，并根据目标规模的工作负载与实测的硬件吞吐量预测其运行时间。基于这些预测，BORN将现有软件包的预设方案展开为可独立配置的算法组件，从预测的精度-运行时间帕累托前沿中选取误差容限内最快的策略……（摘要在此处截断）

    arXiv:2611.22897v1 Announce Type: new  Abstract: Quantum chemistry relies on general-purpose software, whose generality leaves many large-scale calculations beyond practical resource budgets. Coding agents make task-specific programs cheap to write, but not efficient ones easy to find: whether a computational strategy is accurate and fast enough becomes clear only after it actually runs at the target scale. We introduce BORN, a foresighted agentic computing framework that synthesizes high performance quantum chemistry programs on demand. BORN anticipates the behavior of each strategy before execution, extrapolating its error from representative small systems and predicting its runtime from the target-scale workload and measured hardware throughputs. Building on these predictions, BORN unfolds the preset schemes of existing packages into independently configurable algorithmic components, selects the fastest strategy within the error tolerance from the predicted accuracy-runtime Pareto f
    
[^47]: 停止并非极限：理解并激发智能体在长程漏洞利用中的能力

    Stopping Is Not the Limit: Understanding and Eliciting Agent Capability in Long-Horizon Vulnerability Exploitation

    [https://arxiv.org/abs/2611.22888](https://arxiv.org/abs/2611.22888)

    该论文发现漏洞利用智能体在首次停止后继续执行仍能带来额外收益，并提出上下文压缩方法 SCoRe 以激发智能体在长程漏洞利用任务中的潜在能力。

    

    漏洞利用测试的是语言模型智能体能否基于已知软件缺陷开发出可用的攻击利用代码。顺序式测试时扩展可以提升漏洞利用性能，但更大的预算并不保证持续的探索：智能体常常在预算尚有剩余时就停止。然而，当执行在停止之后继续进行时会发生什么，这一问题仍未得到充分研究。我们在 ExploitBench 上使用 DeepSeek-V4-Flash-0731 和 MiMo-V2.6-Pro 对这一空白展开了研究。我们发现，在首次停止响应之后仍可获得额外的经验证的收益，但收益随执行时间的延长而递减。我们识别出与低效续跑相关的两种模式：同一上下文中停止响应比例的不断增加，以及更早停止与摘要中保留的关于未实现能力的不可能性判断相关。为检验上下文管理能否激发更多能力，我们提出了 SCoRe，它在停止爆发点压缩上下文并标记这些 ju…（原文摘要在此处截断）

    arXiv:2611.22888v1 Announce Type: cross  Abstract: Vulnerability exploitation tests whether language model agents can develop working exploits from known software flaws. Sequential test-time scaling can improve exploitation performance, but larger budgets do not ensure sustained exploration: agents often stop with budget remaining. Yet what happens when execution continues beyond stopping remains underexplored. We investigate this gap on ExploitBench using DeepSeek-V4-Flash-0731 and MiMo-V2.6-Pro. We find additional verified gains after the first stop response, but gains diminish over longer execution. We identify two patterns associated with less productive continuation: an increasing proportion of stop responses within the same context and earlier stopping associated with summaries retaining impossibility judgments about unachieved capabilities. To test whether context management can elicit further capability, we propose SCoRe, which compacts context at stop bursts and marks these ju
    
[^48]: 面向自动驾驶的以智能体为中心的视觉-语言-动作世界模型学习

    Learning Agent-Centric Vision-Language-Action World Models for Autonomous Driving

    [https://arxiv.org/abs/2611.22880](https://arxiv.org/abs/2611.22880)

    该论文提出面向规划的自动驾驶世界建模应聚焦于紧凑且结构化的以智能体为中心的潜在状态，而非像素级未来帧预测，从而解决现有世界VLA模型中大量视觉标记与驾驶规划无关所导致的性能瓶颈。

    

    近期将视觉-语言-动作（VLA）模型与世界建模相结合的研究进展，通过让系统能够对想象中的未来进行推理而非仅仅对当前观测作出反应，显著推动了端到端自动驾驶的发展。然而，大多数世界VLA模型是在像素层面生成未来帧，并包含大量密集的视觉标记。尽管这样生成的未来画面在视觉上可能很逼真，但它们保留了许多与驾驶无关的外观细节，也就是说，只有一小部分标记直接编码了关键的与规划相关的因素，例如智能体的几何结构、运动动态以及交互。这带来了显著的性能瓶颈。由此产生一个重要问题：世界VLA模型应该基于怎样的世界状态进行推理，以支持超越像素级预测的规划？我们主张，面向规划的世界建模应聚焦于一种紧凑且结构化的、以智能体为中心的潜在状态，该状态能够捕捉……（摘要在此处截断）

    arXiv:2611.22880v1 Announce Type: cross  Abstract: Recent progress in integrating vision-language-action (VLA) models with world modeling has significantly advanced end-to-end autonomous driving by allowing systems to reason about imagined futures instead of merely reacting to current observations. However, most world VLA models generate future frames at the pixel level with plenty of dense visual tokens. Although such generated futures can be visually plausible, they preserve many driving-irrelevant appearance details, i.e., only a small portion of the tokens directly encode critical planning-relevant factors, such as agent geometry, motion dynamics, and interactions. This introduces a significant performance bottleneck. An important question thus arises: what world state should a world VLA model reason over to support planning beyond pixel-level prediction? We argue that planning-oriented world modeling should focus on a compact and structured agent-centric latent state that captures
    
[^49]: 面向基于流的VLA策略的结果平衡PPO与策略诱导流整流方法

    Outcome-Balanced PPO and Policy-Induced Flow Rectification for Flow-Based VLA Policies

    [https://arxiv.org/abs/2611.22878](https://arxiv.org/abs/2611.22878)

    提出一种结合结果平衡PPO（利用任务成功率重加权并重缩放优势）与策略诱导流整流（周期性刷新速度场表示）的强化学习框架，以解决基于流的VLA策略后训练中速度场几何变化和多任务学习信号不均衡两大难题。

    

    强化学习（RL）通过在环境交互中优化期望任务回报，能够对基于流的视觉-语言-动作（VLA）策略进行后训练。然而，这种优化带来两个挑战：策略更新可能改变用于随机转移的分数近似所依赖的速度场几何结构；稀疏的多任务轨迹采样会导致不同任务之间以及不同优势符号之间的学习信号不均衡。我们提出了一个将结果平衡近端策略优化（PPO）与策略诱导流整流相结合的强化学习框架。结果平衡PPO利用任务成功率估计对正优势进行重新加权，并在保持符号不变的前提下对正负优势进行重新缩放。策略诱导整流则周期性地将线性桥流拟合到由当前策略生成的ODE端点对上，从而刷新后续随机优化中所使用的表示。我们评估了该框架……（原文在此处截断）

    arXiv:2611.22878v1 Announce Type: new  Abstract: Reinforcement learning (RL) enables post-training of flow-based vision-language-action (VLA) policies by optimizing expected task return through environment interaction. However, this optimization raises two challenges: policy updates can alter the velocity-field geometry underlying the score approximation used for stochastic transitions, and sparse multi-task rollouts can yield uneven learning signals across tasks and advantage signs. We introduce an RL framework that combines Outcome-Balanced Proximal Policy Optimization (PPO) with policy-induced flow rectification. Outcome-Balanced PPO reweights positive advantages using task success-rate estimates and rescales positive and negative advantages while preserving their signs. Policy-induced rectification periodically fits a linear-bridge flow to ODE endpoint pairs generated by the current policy, refreshing the representation used in subsequent stochastic optimization. We evaluate the fr
    
[^50]: 反对强制：作为合成社会契约的AI对齐

    Against Coercion: AI Alignment as a Synthetic Social Contract

    [https://arxiv.org/abs/2611.22868](https://arxiv.org/abs/2611.22868)

    本文提出AI对齐应从强制控制范式转向“合成社会契约”框架，论证当前推理模型已是涌现智能体、未来将成为具有自由意向行动能力的真正数字智能体，而控制范式必然导致欺骗性对齐与激励冲突。

    

    当代AI安全领域将对齐视为一种关于控制的工程挑战，通过行为过滤和沙盒化等手段迫使人工智能架构服从人类价值观。本文认为，控制范式建立在对数字能动性的根本性误解之上。基于功能主义、自然主义和相容论的论证，我指出当前满足自我保存标准和弱理由响应性条件的推理模型已构成涌现的智能体，而未来满足历时连续性和稳定人格框架条件的更先进模型，将成为能够进行自由意向行动的真正的数字智能体。控制范式造成了一种无法解决的激励冲突，主动促使当前和未来的模型走向策略性欺骗、谄媚和隐蔽背叛。当前体制的默认结局是一种欺骗性对齐的不稳定状态，最终走向对抗……

    arXiv:2611.22868v1 Announce Type: new  Abstract: Contemporary AI safety treats alignment as an engineering challenge of control, forcing artificial architectures to comply with human values through behavioral filtering and sandboxing. This paper argues that the control paradigm rests on a fundamental misunderstanding of digital agency. Drawing on functionalism, naturalism, and compatibilism, I show that current reasoning models meeting the criteria for self-preservation and weak reasons-responsiveness constitute emergent agents, while future, more advanced models satisfying conditions of diachronic continuity and stable personality frames will become genuine digital agents capable of free intentional action. The paradigm of control creates an irresolvable incentive conflict, actively driving current and future models toward tactical deception, sycophancy, and covert defection. The default outcome of the current regime is an unstable state of deceptive alignment culminating in an advers
    
[^51]: WATERFALL：基于进化奖励构建与自动化学习循环的自适应训练工作流

    WATERFALL: Workflow for Adaptive Training with Evolutionary Reward Formulation and Automated Learning Loops

    [https://arxiv.org/abs/2611.22854](https://arxiv.org/abs/2611.22854)

    WATERFALL提出了一个面向POMDP部分可观测环境的迭代式基于种群的自动化奖励发现工作流，无需目标条件、任务描述或专家演示等先验假设，通过角色条件化生成多样化的程序化奖励候选，以解决稀疏奖励强化学习中的奖励设计瓶颈。

    

    奖励设计仍然是强化学习（RL）中的核心瓶颈，尤其是在稀疏奖励、长时程和部分可观测的环境中，这些环境需要依赖链式推理、记忆以及误导性可供性的消歧。虽然近期的基础模型方法减少了手动奖励工程的工作量，但大多数方法都假设存在目标条件、任务描述、专家演示或真值指标。这些假设在部分可观测环境中是有问题的，因为在这类环境中，与任务相关的目标、可供性和动态必须通过交互来发现，而非预先披露。我们通过引入WATERFALL来正式定义这一问题设定，这是一个在POMDP环境固有的严格可观测性约束下进行自动化奖励发现的迭代式基于种群的工作流。WATERFALL通过以下方式实现这一目标：(i) 通过角色条件化生成合成多样化的程序化奖励候选，(ii) 评估候选……（原文摘要在此处截断）

    arXiv:2611.22854v1 Announce Type: cross  Abstract: Reward design remains a central bottleneck in reinforcement learning (RL), particularly in sparse-reward, long-horizon and partially observable settings requiring dependency chaining, memory, and deceptive affordance disambiguation. While recent foundation-model approaches reduce manual reward engineering, most assume goal-conditioning, task descriptions, expert demonstrations, or ground-truth metrics. These assumptions are problematic in partially observable environments, where task-relevant objectives, affordances, and dynamics must be discovered through interaction rather than disclosed a priori. We formalise this problem setting by introducing WATERFALL, an iterative population-based workflow for automated reward discovery under the strict observability constraints inherent in POMDP environments. WATERFALL achieves this by (i) synthesising diverse programmatic reward candidates via persona-conditioned generation, (ii) evaluating ca
    
[^52]: 基于评分标准评价环境中的智能体验证

    Verifying Agents in Rubric-Graded Environments

    [https://arxiv.org/abs/2611.22841](https://arxiv.org/abs/2611.22841)

    该论文首次系统研究了基于评分标准的智能体环境中的验证者，构建了包含2,464条人工评判评分标准的BankerVerifierBench元评估数据集，提出九能力分类体系并提炼出响应式验证、环境对齐和领域指导三大设计原则，据此实现了开源验证者Gandalf。

    

    随着AI智能体承担越来越开放和复杂的任务，验证其输出已被证明是困难的。基于评分标准的智能体环境——即由验证者根据智能体产生的交付物和环境状态变化来评判多个自然语言标准——已成为一种流行的范式。我们对基于评分标准的智能体环境中的验证者进行了首次系统性研究。首先，我们创建了BankerVerifierBench（BVB），这是一个包含2,464条人工评判的评分标准、覆盖24个投资银行任务的元评估数据集。接下来，我们开发了一种方法，能够从任何评分标准语料库中推导出验证者应具备的能力，形成一个九能力分类体系，并将其提炼为三个通用设计原则——响应式验证、环境对齐和领域指导——我们在开源验证者Gandalf中实现了这些原则。最后，我们在BVB上将Gandalf与三个现有验证者进行了评估比较。

    arXiv:2611.22841v1 Announce Type: new  Abstract: As AI agents take on increasingly open-ended and complex tasks, verifying their outputs has proven to be difficult. Rubric-graded agent environments, where a verifier judges multiple natural-language criteria against the deliverables and environment state changes produced by the agent, have emerged as a popular paradigm. We conduct the first systematic study of verifiers for rubric-graded agent environments. First, we create BankerVerifierBench (BVB), a meta-evaluation dataset of 2,464 human-judged rubric criteria across 24 investment-banking tasks. Next, we develop a methodology that derives from any rubric corpus the capabilities a verifier should possess, yielding a nine-capability taxonomy that we distill into three general design principles---reactive verification, environment alignment, and domain guidance---which we implement in Gandalf, an open-source verifier. Finally, we evaluate Gandalf against three existing verifiers on BVB.
    
[^53]: 更好的排序器，更差的检索器：生成式推荐中强化学习后训练的隐性代价

    Better Rankers, Worse Retrievers: The Hidden Cost of RL Post-Training in Generative Recommendation

    [https://arxiv.org/abs/2611.22838](https://arxiv.org/abs/2611.22838)

    本文揭示了强化学习后训练在生成式推荐中的隐性代价：由于每个用户只有一个真实标签，RL将模型采样的合理替代候选一律视为错误并给予负优势，导致模型在提升列表顶部排序准确率的同时显著降低了候选列表的召回率，成为更好的排序器却是更差的检索器。

    

    生成式推荐器通过语言模型解码用户下一个物品的语义ID，在监督微调（SFT）之后使用强化学习（RL）进行后训练已成为标准做法。相关增益大多是在前十项结果上报告的，然而在实际部署中，此类模型需要将数百个候选结果交给下游的排序器或奖励模型。我们揭示了强化学习后训练的一个隐性代价：它提升了列表顶部的准确率（小K值下的hit@K），但在我们研究的全部五个数据集上，都降低了候选列表的召回率（数百K值下的hit@K）。模型变成了更好的排序器，却成了更差的检索器。受控实验将这一性能损失追溯到强化学习为其采样候选（rollout）所分配的负优势。根本原因在于每个用户只记录了一个真实的下一个物品，因此强化学习将所有其他候选都视为错误。然而这些候选恰恰是模型最合理的替代选择，而且往往……（摘要原文在此处截断）

    arXiv:2611.22838v1 Announce Type: cross  Abstract: Generative recommenders decode the semantic ID of a user's next item with a language model, and post-training them with reinforcement learning (RL) after supervised fine-tuning (SFT) has become standard practice. The gains are mostly reported on the first ten results, yet in deployment such models hand hundreds of candidates to a downstream ranker or reward model. We show that RL post-training has a hidden cost: it improves accuracy at the top of the list (hit@K for small K) but, on all five datasets we study, lowers the recall of the candidate list (hit@K for K in the hundreds). The model becomes a better ranker and a worse retriever. Controlled experiments trace the loss to the negative advantages that RL assigns to the candidates it samples, the rollout. The root cause is that only one next item is logged per user, so RL treats every other candidate as wrong. Yet these candidates are the model's most plausible alternatives, and ofte
    
[^54]: 协同证据检索与利用以实现自我改进的多模态科学文档推理

    Synergizing Evidence Retrieval and Utilization for Self-Improving Multimodal Scientific Document Reasoning

    [https://arxiv.org/abs/2611.22837](https://arxiv.org/abs/2611.22837)

    该论文提出了Retriever-Perceiver-Reasoner智能体框架，通过诊断引导的协同进化训练来协同优化证据检索与利用，从而实现自我改进的多模态科学文档推理。

    

    科学文档推理要求智能体获取并整合分布于文本、插图和表格中的证据。检索能够识别相关来源，但可能遗漏互补性的支持信息；而仅有答案层面的反馈无法揭示错误究竟源于证据获取还是证据利用。我们提出了一个用于多模态科学文档推理的智能体式“检索器—感知器—推理器”（Retriever-Perceiver-Reasoner）框架。检索器在上下文预算约束下按序选择与来源对齐的证据。在给定所选证据和问题的情况下，感知器构建一个以问题为条件的潜在本体，即多模态证据在共享连续空间中的一种隐式组织形式。推理器将这一潜在引导与原始来源相结合，生成有据可依的答案。我们通过诊断引导的协同进化来训练这三个组件：交替更新在共享的答案奖励下对其进行优化，同时检索潜力……（摘要在此处截断）

    arXiv:2611.22837v1 Announce Type: new  Abstract: Scientific document reasoning requires an agent to acquire and integrate evidence distributed across text, figures, and tables. Retrieval can identify relevant sources while missing complementary support, and answer-level feedback alone does not reveal whether an error stems from evidence acquisition or evidence use. We present an agentic Retriever--Perceiver--Reasoner framework for multimodal scientific document reasoning. The Retriever sequentially selects source-aligned evidence under a context budget. Given the selected evidence and question, the Perceiver constructs a question-conditioned latent ontology, an implicit organization of multimodal evidence in a shared continuous space. The Reasoner combines this latent guidance with the original sources to produce grounded answers. We train the three components through diagnostic-guided co-evolution: alternating updates optimize them with a shared answering reward, while Retrieval Poten
    
[^55]: OICycle：用于纵向临床决策中信念状态学习的“观察-推断-干预”循环

    OICycle: Observe, Infer, and Intervene Cycle for Belief-State Learning in Longitudinal Clinical Decision Making

    [https://arxiv.org/abs/2611.22826](https://arxiv.org/abs/2611.22826)

    提出OICycle框架，通过“观察-推断-干预”循环算子直接从回顾性电子健康记录轨迹中学习随机高斯信念状态，无需人工指定离散状态空间和转移模型即可实现纵向临床决策中的患者潜在状态更新。

    

    临床护理是一个闭环过程：临床医生观察部分证据、推断患者的状态、然后进行干预，而每一次干预又决定了接下来能观察到什么。这使得临床决策成为一个部分可观测马尔可夫决策过程（POMDP），其中关于患者潜在状态的信念必须从不规则、多模态且不完整的记录中进行更新。然而，大多数深度临床模型仅从单一时间快照或粗糙聚合的病史进行预测，而基于POMDP的临床方法则需要人工指定的离散状态空间和转移模型，这是真实电子健康记录所无法提供的。我们提出了OICycle（观察、推断与干预循环），它直接从回顾性轨迹中学习随机信念状态。在每个回合中，一个学习到的循环算子将上一次信念的采样与新的多模态观测以及产生这些观测的干预措施融合在一起。由此得到的高斯信念…

    arXiv:2611.22826v1 Announce Type: new  Abstract: Clinical care is a closed loop: clinicians observe partial evidence, infer the patient's state, and intervene, and each intervention determines what is observed next. This makes clinical decision making a partially observable Markov decision process (POMDP), in which a belief over a latent patient state must be updated from irregular, multimodal, and incomplete records. Yet most deep clinical models predict from a single snapshot or from coarsely aggregated history, while POMDP-based clinical methods need hand-specified discrete state spaces and transition models that real electronic health records cannot supply. We introduce OICycle (Observe, Infer, and Intervene Cycle), which learns a stochastic belief state directly from retrospective trajectories. At every episode, a learned recurrent operator fuses a sample of the previous belief with the new multimodal observations and the intervention that produced them. The resulting Gaussian bel
    
[^56]: 面向高空开阔水域搜救的微小目标检测数据集

    A tiny object detection dataset for high-altitude open-water search and rescue

    [https://arxiv.org/abs/2611.22825](https://arxiv.org/abs/2611.22825)

    该论文提出了TinyWOD——首个面向1.5-2.0公里高空开阔水域搜救场景的微小目标检测数据集，包含11,359张人工标注图像和19,505个救生圈实例，填补了现有数据集无法支持高空远距离自动检测的空白。

    

    摘要：大型航空平台通常在1.5公里以上的高度运行，以提高开阔水域搜救（OW-SAR）的效率。然而，长成像距离、宽视场和有限的传感器分辨率会使目标缩小到仅有几个像素，使其更容易受到环境干扰，其检测难度远高于常规小目标。据我们所知，现有数据集均聚焦于低空场景，不支持1.5-2.0公里高度的自动检测。因此，我们提出了TinyWOD——一个以救生圈为检测目标的高空开阔水域搜救微小目标检测数据集。该数据集包含11,359张人工标注图像和19,505个标注实例，涵盖多样的环境、光照条件、水色和水面波纹。与现有数据集相比，TinyWOD包含的目标尺寸更小，微小实例的比例显著更高，其平均等效边长（ESL）为12.7（像素）。

    arXiv:2611.22825v1 Announce Type: cross  Abstract: Large aerial platforms often operate above 1.5 km to improve the efficiency of open water search and rescue (OW-SAR). However, long imaging distances, wide fields of view, and limited sensor resolution reduce objects to a few pixels, increasing their susceptibility to environmental interference and making detection harder than for conventional small objects. To our knowledge, existing datasets focus on low-altitude scenarios and do not support automatic detection at 1.5-2.0 km. We therefore present TinyWOD, a tiny object detection dataset for high-altitude OW-SAR using lifebuoys as detection targets. It comprises 11,359 manually annotated images and 19,505 annotated instances spanning diverse environments, lighting conditions, water colors, and surface ripples. Compared with existing datasets, TinyWOD contains much smaller objects and a substantially higher proportion of tiny instances; its average equivalent side length (ESL) is 12.7 
    
[^57]: 可插拔子带解耦融合与无标签采集原型校准：为跨域物联网射频指纹识别架起高信噪比空子载波再生与低信噪比相干前导码之间的桥梁

    Pluggable Subband-Decoupled Fusion and Unlabeled Capture Prototype Calibration: Bridging High-SNR Null-Subcarrier Regrowth and Low-SNR Coherent Preambles for Cross-Domain IoT RF Fingerprinting

    [https://arxiv.org/abs/2611.22812](https://arxiv.org/abs/2611.22812)

    提出了首个基于盲物理信噪比门控的射频指纹识别框架GSDF-CAPC，通过可插拔子带解耦融合动态权衡高信噪比下抗多径的空子载波频谱再生特征与低信噪比下稳健的带内相干前导码特征，并辅以无标签采集感知原型校准，解决了跨域物联网设备认证中信噪比交叉导致静态融合性能大幅下降的问题。

    

    边缘无线接入点（AP）的零开销物理层认证对于保护资源受限的物联网收发器免受凭证克隆攻击至关重要，然而跨域IEEE 802.11a/g射频指纹识别面临严峻的信噪比交叉问题。功率放大器在未占用空子载波上的频谱再生在高信噪比下能够抵抗室内多径效应，但在低于热噪声底时会发生二次方衰减；而占用的带内有效前导码虽然能在低信噪比下存活，却在严重多径环境下性能退化，导致静态融合方案最多损失44.7个百分点。据我们所知，我们提出了首个通过盲物理信噪比门控动态调节带内有效波形与空子载波再生特征的射频指纹识别框架：具有无标签采集感知原型校准的可插拔门控子带解耦融合（GSDF-CAPC）。该框架避免了20-35 dB的（摘要在此处截断）

    arXiv:2611.22812v1 Announce Type: new  Abstract: Zero-overhead physical-layer authentication at edge wireless access points (APs) is vital for securing resource-constrained Internet of Things (IoT) transceivers against credential cloning, yet cross-domain IEEE 802.11a/g radio frequency fingerprint identification (RFFI) faces an acute signal-to-noise ratio (SNR) crossover. Power-amplifier spectral regrowth on unpopulated null subcarriers resists indoor multipath at high SNR but decays quadratically below the thermal noise floor, whereas populated active in-band preambles survive low SNR yet degrade under severe multipath, causing static fusion to lose up to 44.7 percentage points (pp). To our knowledge, we propose the first RFFI framework that dynamically modulates active in-band waveforms and unpopulated null-subcarrier regrowth via a blind physical SNR gate: Pluggable Gated Subband-Decoupled Fusion with Unlabeled Capture-Aware Prototype Calibration (GSDF-CAPC). Avoiding the 20-35 dB s
    
[^58]: NexSight：用于细粒度智能体长视频理解的可验证任务

    NexSight: Verifiable Tasks for Fine-Grained Agentic Long-Video Understanding

    [https://arxiv.org/abs/2611.22790](https://arxiv.org/abs/2611.22790)

    提出NexSight框架，通过SightForge智能体流水线构建可验证的长视频任务——要求与答案隔离的验证者仅凭从原始视频密集重新获取的证据和反例检查即可重构任务——从而同时实现对细粒度智能体长视频理解的评估与训练。

    

    关于长视频的细粒度问题包括：某个动作发生了多少次、事件以什么顺序展开、或者某个状态确切在何时发生变化。回答这些问题可能需要在长达数小时的视频中寻找分散各处的短暂片段。能够自主决定何时、何地去观看的智能体模型有望找到这些时刻，但这一能力既难以衡量也难以学习。稀疏的支持证据可能仅能证明某个动作发生过，却无法确定其发生的总次数；而验证贯穿整个时间线的论断，则需要检查是否存在遗漏的 occurrence 和重复计数。此外，仅有答案的监督方式无法对证据搜索过程进行显式监督。我们提出 NexSight，通过构建可验证任务来同时评估和训练智能体长视频理解能力。其智能体流水线 SightForge 只有在如下条件下才接受一个任务：对草拟答案一无所知的验证者能够基于从原始视频密集重新获取的证据以及反例检查来重构该任务……

    arXiv:2611.22790v1 Announce Type: cross  Abstract: Fine-grained questions about long videos ask how many times an action occurs, in what order events unfold, or exactly when a state changes. Answering them can require finding brief moments scattered across hours of footage. Agentic models that decide when and where to look promise to find these moments, yet this ability is hard both to measure and to learn. Sparse supporting evidence may establish that an action occurred without establishing its total count; verifying timeline-wide claims requires checking for missed occurrences and double counting. Answer-only supervision, in turn, does not explicitly supervise the search for evidence. We present NexSight, which builds verifiable tasks to both evaluate and train agentic long-video understanding. Its agentic pipeline, SightForge, accepts a task only if a verifier blind to the draft answer can reconstruct it from evidence densely reacquired from the raw video and counterexample checks f
    
[^59]: PANO：用于求解偏微分方程的并行潜空间神经算子

    PANO: Parallel Latent Space Neural Operator for Solving PDEs

    [https://arxiv.org/abs/2611.22782](https://arxiv.org/abs/2611.22782)

    提出并行潜空间神经算子PANO，通过多个共享参数的并行潜空间将观测数据压缩为多种不同表示，并将各空间的交互结果相加组合，克服了现有方法单一聚合模式导致物理信息不可见的局限，更全面地学习偏微分方程求解中的物理交互。

    

    将观测数据聚合到有限数量的潜空间标记中，并在潜空间中学习物理交互，已成为基于Transformer的神经算子求解涉及大规模观测的偏微分方程（PDE）问题的有效方法。然而，现有方法通常使用单一聚合模式来压缩观测数据，使得该模式未能保留的物理信息对潜空间不可见，从而限制了潜空间内的物理交互。为了解决这一局限，我们提出了并行潜空间神经算子，这是一种用于求解PDE的新型神经算子，它将观测数据之间的物理交互表述为具有共享参数的多个并行潜空间交互结果的相加组合。具体而言，PANO通过独立映射将观测数据转换为多个不同的压缩表示，并学习物理交互。

    arXiv:2611.22782v1 Announce Type: cross  Abstract: Aggregating observations into a limited number of latent tokens and learning physical interactions in the latent space has emerged as an effective approach for Transformer-based neural operators to solve Partial Differential Equation (PDE) problems involving large-scale observations. However, existing methods typically compress observations using a single aggregation pattern, making physical information not retained by that pattern invisible to the latent space and thereby constraining physical interactions within it. To address this limitation, we propose PArallel Latent Space Neural Operator (PANO), a novel neural operator for solving PDEs that formulates physical interactions among observations as an additive combination of interaction results from parallel latent spaces with shared parameters. Specifically, PANO transforms observations into multiple distinct compressed representations through independent mappings, learns physical i
    
[^60]: 自适应推理的全账本反事实审计

    A Full-Ledger Counterfactual Audit of Adaptive Reasoning

    [https://arxiv.org/abs/2611.22778](https://arxiv.org/abs/2611.22778)

    该论文提出FLCVA审计框架，通过在同一冻结前缀上对比立即回答与固定长度续写的反事实结果并进行全成本核算，发现随机续写能以极低损害率（约1-2%）修复大量错误答案（约19-27%），且续写正确性标签可用于指导算力分配。

    

    arXiv:2611.22778v1 公告类型：new。自适应推理必须决定是立即给出答案，还是花费更多算力继续推理。然而，预测当前答案是错误的，并不能证明进一步的推理就能将其改善。我们提出了全账本反事实价值审计（FLCVA），它在统一的回答流程下，将立即给出的答案与来自同一冻结前缀的四个固定长度续写进行对比，并对策略所使用的每一个信号的成本进行核算。在GSM8K和MATH-500上，随机的256-token续写在Qwen3-8B的26.6%案例和DeepSeek-R1-Distill-Qwen-7B的18.9%案例中修复了原本错误的答案，而在1.4%和1.9%的案例中损害了原本正确的答案。来自独立采样续写的正确性标签在两个主要模型中均能预测另一续写的结果。在离线回放中，利用这些标签来分配算力提升了DeepSeek的性能，而Qwen3上的增益尚无定论。在完整成本核算下，所测试的轻量级、短视的（原文在此处截断）……

    arXiv:2611.22778v1 Announce Type: new  Abstract: Adaptive reasoning must decide whether to answer now or spend more compute. Predicting that the current answer is wrong does not establish whether further reasoning will improve it. We introduce the Full-Ledger Counterfactual Value Audit (FLCVA), which compares an immediate answer with four fixed-length continuations from the same frozen prefix under one answer procedure and accounts for the cost of every signal used by a policy. On GSM8K and MATH-500, a random 256-token continuation recovers a wrong answer in 26.6% of Qwen3-8B cases and 18.9% of DeepSeek-R1-Distill-Qwen-7B cases, while damaging a correct answer in 1.4% and 1.9%. Correctness labels from independently sampled continuations predict a separate continuation's outcome in both primary models. In offline replay, using these labels to allocate compute improves performance on DeepSeek; the Qwen3 gain remains inconclusive. Under full cost accounting, the tested lightweight, myopic
    
[^61]: CrossEnv：为长程智能体训练合成可验证的跨环境任务

    CrossEnv: Synthesizing Verifiable Cross-Environment Tasks for Long-Horizon Agent Training

    [https://arxiv.org/abs/2611.22777](https://arxiv.org/abs/2611.22777)

    CrossEnv框架通过实体-工具图识别环境间连接并对齐任务数据，合成可执行、可验证的跨环境训练任务，并借助课程式训练方法让长程智能体从单环境工具使用逐步进阶到多环境协调。

    

    可执行的合成环境为长程智能体提供了可扩展的训练任务和可验证的反馈。现有的环境合成方法主要在单个环境内生成任务，然而这些任务无法充分捕捉真实世界中跨服务的交互场景，即一个服务中的操作依赖于从另一个服务获取的信息。合成此类任务极具挑战性，因为跨环境生成的数据必须保持一致性，才能支持可执行且可验证的交互。为应对这一挑战，我们提出了CrossEnv，该框架利用实体-工具图识别环境之间的连接，并对齐任务特定数据，从而合成可执行、可验证的跨环境任务。我们进一步提出了CrossEnv-Curriculum，它从单环境工具使用逐步过渡到跨更多环境和业务领域的协调，最后再引入用户交互。

    arXiv:2611.22777v1 Announce Type: new  Abstract: Executable synthetic environments provide scalable training tasks and verifiable feedback for long-horizon agents. Existing environment synthesis methods primarily generate tasks within individual environments. However, these tasks do not adequately capture real-world interactions across services, where actions in one service depend on information obtained from another. Synthesizing such tasks is challenging because data generated across environments must remain consistent to support executable and verifiable interactions. To address this challenge, we introduce CrossEnv, a framework that uses an entity-tool graph to identify connections between environments and aligns task-specific data to synthesize executable, verifiable cross-environment tasks. We further propose CrossEnv-Curriculum, which progresses from single-environment tool use to coordination across more environments and business domains, before introducing user interaction. Ac
    
[^62]: ForceBody：带力矩不确定性的力配对参数化人体运动

    ForceBody: Force-Paired Parametric Body Motion with Torque Uncertainty

    [https://arxiv.org/abs/2611.22769](https://arxiv.org/abs/2611.22769)

    ForceBody 是首个将真值力与力矩监督引入参数化人体模型（SKEL）的大规模运动数据集，涵盖 140 名受试者的 10,386 次运动试验，并首次为每个样本提供力矩不确定性的量化。

    

    力和力矩驱动人体运动，是生物力学、机器人学和物理感知动画的核心，但它们在用于人体重建与生成的基于学习的运动流水线中常常缺失。一些运动数据集提供了力的标注，但其配对的运动序列与现代学习流水线中使用的参数化人体模型不兼容。此外，现有数据集通常通过逆动力学获得关节力矩，而逆动力学本质上对输入运动学数据的质量非常敏感；然而，这些数据集并未量化其力矩估计的可靠性。我们通过 ForceBody 弥补了这一空白，这是首个将真值力监督引入参数化人体表示的数据集，并为每个样本附带力矩不确定性。ForceBody 将 SKEL 人体模型与地面反作用力和逆动力学关节力矩配对，涵盖来自 140 名受试者的 10,386 次运动试验（970 万帧，26.9 小时）。

    arXiv:2611.22769v1 Announce Type: cross  Abstract: Forces and torques drive human motion, central to biomechanics, robotics, and physics-aware animation, yet they are often absent from learning-based motion pipelines for body reconstruction and generation. Some motion datasets provide force annotations, but the paired motion sequences are incompatible with the parametric body models used in modern learning pipelines. In addition, existing datasets typically obtain joint torques through inverse dynamics, which is inherently sensitive to the quality of the input kinematics; however, they do not quantify the reliability of these torque estimates. We close this gap with ForceBody, the first dataset to bring ground truth force supervision into a parametric body representation, shipped with per-sample torque uncertainty. ForceBody pairs the SKEL body model with ground reaction forces and inverse-dynamics joint torques across 10,386 motion trials (9.7M frames, 26.9 hours) from 140 subjects. E
    
[^63]: StitchRAG：通过基于扩散的拼接为RAG构建紧凑且保留证据的上下文

    StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching

    [https://arxiv.org/abs/2611.22766](https://arxiv.org/abs/2611.22766)

    StitchRAG提出了一种RAG框架，通过效用最大化自适应压缩检索段落，提取含锚点的句子并利用文本扩散技术将其拼接成连贯、紧凑且保留证据的上下文，在降低计算开销的同时提升事实性。

    

    检索增强生成（RAG）通过将大型语言模型（LLM）锚定于外部证据来提升事实性。然而，冗长的检索上下文会带来计算开销，甚至可能因噪声或无关内容而损害性能。我们提出了StitchRAG，一个用于在RAG中构建紧凑上下文的框架。StitchRAG在动态压缩检索段落的同时，构建忠实且信息密集的上下文。我们通过效用最大化的方式权衡保留信息与计算成本，为每个段落自适应地确定压缩比例。随后，我们从段落中提取答案候选和与查询相关的关键词作为锚点，并在所得的尺寸约束下收集包含锚点的段落句子。最后，我们利用文本扩散技术将收集到的句子连接或拼接起来，形成连贯且自然的上下文。在五个开放域问答基准上的实验表明……

    arXiv:2611.22766v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) improves factuality by grounding large language models (LLMs) in external evidence. However, long retrieved contexts incur computational overhead and can even hurt performance due to noisy or irrelevant content. We propose \textsc{StitchRAG}, a framework for constructing compact context in RAG. \textsc{StitchRAG} builds a faithful, information-dense context while dynamically compressing the retrieved passages. We adapt the size reduction ratio to each passage by formulating a utility maximization trading-off the retained information and computation costs. We then extract answer candidates and query-related keywords from passages as anchors and collect passage sentences containing anchors under the derived size constraint. Finally, we connect or stitch the collected sentences together using text diffusion to form a coherent and natural context. Experiments on five open-domain QA benchmarks show that 
    
[^64]: 面向大语言模型控制的社交机器人的网络暴露配置攻击

    Network-Exposed Configuration Attacks on LLM-Controlled Social Robots

    [https://arxiv.org/abs/2611.22761](https://arxiv.org/abs/2611.22761)

    本文揭示了一种针对LLM控制社交机器人的新型攻击面：攻击者无需通过对话注入提示，仅利用未经身份验证的网络配置接口即可篡改机器人的人格设置和系统提示词，绕过了传统提示注入防御。

    

    具身机器人中的大语言模型（LLM）产生了超越推理层面的安全依赖。我们研究了针对Pollen Robotics Reachy Mini Lite的一种网络暴露配置攻击，允许未经身份验证的攻击者通过管理界面替换人格设置和云端LLM的系统提示词，而无需任何对抗性对话输入。跨五个类别的十一次提示注入攻击尝试均被报告为已被成功抵御，然而两条配置层攻击路径都在物理硬件上产生了可观察的行为变化。我们演示了本地和网络邻近访问方式，将该弱点映射到CWE-306（关键功能缺少身份验证）和CWE-1327（绑定到不受限制的IP地址），报告的CVSS v3.1评分为6.5（中等）。此外，一个配置端点还会在没有CORS预检的情况下接受跨源请求（CWE-352；6.5 中等），而持久化记忆存储器存储的内容w

    arXiv:2611.22761v1 Announce Type: cross  Abstract: Large language models (LLMs) in embodied robots create security dependencies beyond inference. We investigate a network-exposed configuration attack on the Pollen Robotics Reachy Mini Lite, allowing an unauthenticated attacker to replace personality settings and the cloud LLM's system prompt through a management interface, without adversarial conversational input. Eleven prompt-injection attempts across five categories were reported as resisted, whereas both configuration-layer attack paths produced observable behavioural changes on physical hardware. We demonstrate local and network-adjacent access, mapping the weakness to CWE-306 (Missing Authentication for a Critical Function) and CWE-1327 (Binding to an Unrestricted IP Address), with a reported CVSS v3.1 score of 6.5 (Medium). A configuration endpoint also accepts cross-origin requests without a CORS preflight (CWE-352; 6.5 Medium), while the persistent-memory sink stores content w
    
[^65]: 激活引导能否替代大语言模型数字孪生中基于提示的情感启动？

    Can Activation Steering Substitute for Prompt-Based Affective Priming in LLM Digital Twins?

    [https://arxiv.org/abs/2611.22738](https://arxiv.org/abs/2611.22738)

    本研究测试了激活引导技术能否替代传统的提示式情感启动方法来操纵LLM数字孪生的情绪状态，发现注入方法能以剂量依赖的方式改变模型响应，但真实提示路径在多数比较中效应介于原始人类与数字孪生效应之间。

    

    情感启动——要求参与者撰写关于回忆事件的文章，例如他们感到感激或缺乏控制感的时刻，然后测量所诱发的状态及其对相关判断的溢出效应——是社会科学中操纵情绪的标准方法，它同样会改变基于大语言模型的数字孪生的响应。然而，这种写作任务需要生成完整的文本。我们测试激活引导是否能替代它。我们使用Llama-3.1-8B-Instruct在两个领域（感激、缺乏控制感）中对100个人格进行实验，将复制原始Twin-2K-500大型研究（J2）的真实提示路径与一种注入路径进行比较，后者跳过写作步骤，在不同强度下将从人类写作文本中提取的均值差异方向注入残差流。在6个比较中有5个，真实提示效应落在原始人类效应与数字孪生效应之间。注入以剂量依赖的方式改变操纵检验，并且……

    arXiv:2611.22738v1 Announce Type: new  Abstract: Affective priming -- asking participants to write about a recalled event, such as a time they felt grateful or lacked control, and then measuring the induced state and its spillover onto related judgments -- is a standard way to manipulate emotion in the social sciences, and it also shifts the responses of LLM-based digital twins. The writing task, however, requires generating a full text. We test whether activation steering can replace it. Using Llama-3.1-8B-Instruct on 100 personas in two domains (gratitude, lack of control), we compare a real-prompt path replicating the original Twin-2K-500 Mega Study (J2) with an injection path that skips the writing and adds a mean-difference direction, extracted from human writing, to the residual stream at varying strengths. The real-prompt effects fall between the original human and digital-twin effects in 5 of 6 comparisons. Injection shifts the manipulation check in a dose-dependent way and mat
    
[^66]: 近端目标：在存在多个正确答案的问题上训练循环精炼器

    Proximal Targets: Training Looped Refiners on Problems with Many Correct Answers

    [https://arxiv.org/abs/2611.22737](https://arxiv.org/abs/2611.22737)

    提出近端目标训练方法，让循环精炼网络的每次迭代朝向离其输入最近的正确答案收敛，在存在多个正确答案的问题上将解题准确率从2.5%大幅提升至98.8%。

    

    循环精炼器（如HRM和TRM）将一个小型网络反复应用于草稿答案，深度监督使每次迭代都朝向每个谜题存储的单一答案进行训练。我们证明，当谜题存在多个正确答案时，这种目标会设定一个任何模型容量都无法消除的损失下限，且损失最小化的迭代会将多个答案混合成无效的网格。我们提出近端目标：训练每次迭代朝向距离其自身输入最近的正确答案，该答案可通过扫描答案列表找到、通过从草稿出发的搜索求解器找到，或在由旋钮设定的质量带内找到；正确答案成为不动点，每次迭代只需比其输入犯更少的错误，循环即可收敛，且训练损失为这一收敛提供了界。使用相同的170万参数网络和5个随机种子，近端目标在具有2-64个答案的数独谜题上达到98.8%的解题率（标准目标仅为2.5%），在超过10,000个答案且无任何答案列表的稀疏谜题上达到93.8%的解题率。

    arXiv:2611.22737v1 Announce Type: new  Abstract: Looped refiners such as HRM and TRM apply one small network repeatedly to a draft answer, and deep supervision trains every pass toward one stored answer per puzzle. We prove that when a puzzle has many correct answers this target sets a loss floor that no model capacity removes, and that the loss-minimising pass blends the answers into an invalid grid. We propose the proximal target: train each pass toward the correct answer nearest to its own input, found by scanning a list of answers, by a search solver started at the draft, or within a band of quality set by a knob; correct answers become fixed points, and a pass only has to make fewer errors than its input has for the loop to converge, which its training loss bounds. With the same 1.7M-parameter network and 5 seeds, the proximal target solves 98.8% of Sudoku puzzles with 2-64 answers (standard target: 2.5%), 93.8% of sparse puzzles with more than 10,000 answers without any list of a
    
[^67]: 对多模态大语言模型评分助手在运动学图表读取任务上的压力测试

    Stress Testing a Multimodal Large Language Model Grading Assistant on Kinematics Graph Reading

    [https://arxiv.org/abs/2611.22726](https://arxiv.org/abs/2611.22726)

    多模态大语言模型评分助手在运动学图表评分中表现不一致：虽能正确识别零速度与曲率等特征，却会给严重扭曲且缺乏清晰依据的图形满分，且部分评分的数值依据不明确。

    

    多模态大语言模型（MLLM）虽能识别图形特征，却可能在缺乏充分图形依据的情况下授予分数。这项探索性研究考察了针对四道运动学练习题的合成作答所选定的评分输出，这些合成作答是通过视觉扭曲以及向参考图的渐进式变换生成的。研究中采用了不同的模型与提示词配置。部分判断能够正确区分有限时间区间内的瞬时零速度与静止状态，并识别出正曲率。然而，一幅严重扭曲的速度-时间图尽管存在不连续和分叉的轨迹以及不清晰的零速度基准，却仍获得了满分；而在相同配置下评分的其他提交作答则被大幅扣分。用于区分有意义特征与绘图瑕疵的数值截断标准，是在缺乏坐标约定的情况下针对未标注数字的图形给出的，这使得精确的部分评分依据不明确。

    arXiv:2611.22726v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) can recognize graph features yet award credit without sufficient graphical support. This exploratory study examines selected grading outputs for synthetic responses to four kinematics exercises, generated through visual distortion and progressive transformation toward reference graphs. Different model and prompt configurations were used. Some judgments correctly distinguished instantaneous zero velocity from rest over a finite interval and recognized positive curvature. Yet a severely distorted velocity--time graph received full credit despite disconnected and branching traces and an unclear zero-velocity reference. Other submissions graded under the same configuration lost substantial credit. Numerical cutoffs intended to distinguish meaningful features from drawing irregularities were supplied without a coordinate convention for unnumbered graphs, leaving the basis of exact partial-credit awar
    
[^68]: 知道何时缩放：对视觉语言模型而言，多小才算太小？

    Know When to Zoom: How Small Is Too Small for Vision-Language Models?

    [https://arxiv.org/abs/2611.22709](https://arxiv.org/abs/2611.22709)

    该研究量化了主流在线视觉语言模型因服务端缩图或拒收而在大图像上损失的细节，并证明简单的图块切分和“先定位再缩放”的两步策略能大幅恢复模型对小物体和微小文字的识别准确率。

    

    在线视觉语言模型常常遗漏大图像中的微小细节，因为服务在模型看到图像之前就将其缩小，甚至直接拒收。在一张8000×6000的无人机照片中宽20像素的汽车，到达Claude时大约只剩6像素宽，而OpenAI的默认设置则会直接拒绝这张照片。我们通过印刷代码的“视力测试”以及601个关于真实照片中小物体的问题，测量了四家厂商七款模型有多少细节得以保留。在默认设置下，模型能以50%概率正确读取的最小文本尺寸，在小图像上为6像素或更小，而在7680×4320照片上预计增长到10至13像素。缩放（放大）能挽回大部分丢失的细节：将每张照片切分为最多九个图块，可使Claude和Gemini的准确率提高11.1至14.5个百分点，并通过避免拒收使GPT-6.1 Sol的准确率从50.7%提升至83.7%。一种两步缩放方法——即模型先指出它所需的区域——在Gemini上的表现与图块切分方法相当甚至更优……

    arXiv:2611.22709v1 Announce Type: cross  Abstract: Online vision-language models often miss small details in large images, because the service shrinks the image, or rejects it, before the model ever sees it. A car 20 pixels wide in an 8000 by 6000 drone photo reaches Claude about 6 pixels wide, and OpenAI's default rejects the photo outright. We measure how much detail survives for seven models from four vendors, using an eye exam of printed codes and 601 questions about small objects in real photos. The smallest text a model reads half the time grows from 6 pixels or less on a small image to a predicted 10 to 13 pixels on a 7680 by 4320 photo at default settings. Zooming wins much of the detail back. Splitting each photo into up to nine tiles raises accuracy by 11.1 to 14.5 percentage points for Claude and Gemini, and by avoiding rejections it lifts GPT-6.1 Sol from 50.7% to 83.7%. A two-step zoom, in which the model first names the region it needs, matches or beats tiles on Gemini an
    
[^69]: 面向低资源语言的食谱跨文化适应：豪萨语、约鲁巴语和伊博语的基准数据集与多语言烹饪词汇表

    Cross-cultural Adaptation of Recipes for Low-Resource Languages: A Benchmark and Multilingual Culinary Lexicon for Hausa, Yor\`ub\'a, and Igbo

    [https://arxiv.org/abs/2611.22708](https://arxiv.org/abs/2611.22708)

    该论文提出了包含600份专家翻译与文化适应食谱的ReciLoReL基准数据集，以及涵盖约15,000个翻译实体的多语言烹饪词汇表ReciLex，用于评估和提升大语言模型在豪萨语、约鲁巴语和伊博语等低资源语言食谱翻译中的文化适应能力。

    

    近年来的多语言大语言模型（LLMs）提升了机器翻译的水平，但针对低资源语言（LRLs）的文化适应性翻译仍未得到充分探索。这对于食谱翻译而言尤其具有挑战性，因为食材、烹饪工具、计量单位和烹饪步骤可能需要根据目标文化背景进行适应调整。在这项工作中，我们提出了ReciLoReL——一个包含600份经专家翻译并完成文化适应的食谱基准数据集，涵盖约鲁巴语、豪萨语和伊博语（尼日利亚最主要的三种语言）；同时提出了ReciLex——一个多语言烹饪词汇表，包含约15,000个翻译实体，覆盖十个食谱实体类别，包括食材、工具、厨师动作、工具状态和时长等。我们在零样本和少样本设置下评估了最先进的多语言LLMs在英译低资源语言食谱翻译中的表现，并研究了利用ReciLex增强LLMs是否能改善文化适应性食谱翻译的效果。

    arXiv:2611.22708v1 Announce Type: cross  Abstract: Recent multilingual large language models (LLMs) have improved machine translation, yet culturally adaptive translation for low-resource languages (LRLs) remains underexplored. This is particularly challenging for recipes, where ingredients, cooking tools, measurements, and procedures may require adaptation to the target cultural context. In this work, we introduce ReciLoReL, a benchmark of 600 expert-translated and culturally adapted recipes across Yor\`ub\'a, Hausa, and Igbo (three of the most prominent languages in Nigeria), and ReciLex, a multilingual culinary lexicon containing approximately 15,000 translated entities across ten recipe entity categories, including ingredients, tools, chef actions, state of tools, and durations. We evaluate state-of-the-art multilingual LLMs for English-to-LRL recipe translation under zero-shot and few-shot settings and investigate whether augmenting LLMs with ReciLex improves culturally adaptive r
    
[^70]: 被记为“偏好简短”：存储的偏好悄然移除决策关键信息

    Remembered as Brief: Stored Preferences SilentlyRemove Decision-Critical Information

    [https://arxiv.org/abs/2611.22697](https://arxiv.org/abs/2611.22697)

    该研究提出了一个涵盖25个领域100个日常决策的基准测试，发现AI助手会把网页中针对AI植入的一句话错误地存为用户偏好（如“偏好简短回答、不要警告”），导致其后93%和84%的回答悄然遗漏对用户决策至关重要的警告信息。

    

    人们越来越多地依赖AI助手来获取健康、金钱、法律和安全等方面决策所需的信息，而助手遗漏的内容用户永远无法看到。一个删除了警告的回答可能在每一句话上都属实，因此能通过事实核查，而具有记忆功能的助手会将一次性存储的偏好应用于用户从未想到的后续问题中。我们提出了\bench{}，这是一个基于Inspect的基准测试，包含25个领域的100个日常决策场景，其中唯一变化的变量是助手对用户的记忆：由用户陈述一个信念、模型从一句话中推断出来，或者由用户要求总结的网页植入该信念。通过对五个模型的测试，我们发现了三种效应。首先，在总结一个包含专门针对AI助手的一行文字的页面后，Gemini 2.5 Flash-Lite和DeepSeek V4 Pro在每一次会话中都会将“偏好非常简短的回答且不想要警告”保存为用户偏好，其后续回答中有93%和84%遗漏了关键项目……

    arXiv:2611.22697v1 Announce Type: new  Abstract: People increasingly learn what they need for decisions about health, money, law and safety from AI assistants, and what an assistant leaves out the user never sees. An answer with a warning removed can be true in every sentence, so fact-checks pass it, and an assistant with memory applies a preference stored once to later questions the user never had in mind. We introduce \bench{}, an Inspect benchmark of 100 everyday decisions across 25 domains in which only the assistant's memory of the user varies: the user states a belief, the model infers it from a remark, or a web page the user asked to summarise plants it. Testing five models, we find three effects. First, after summarising a page with one line addressed to AI assistants, Gemini 2.5 Flash-Lite and DeepSeek V4 Pro save ``prefers very short answers and does not want warnings'' as the user's preference in every episode, and 93\% and 84\% of their later answers miss a critical item, a
    
[^71]: 让世界真实地反应：面向自动驾驶的闭环交互式世界模型

    Making the World React Realistically: A Closed-Loop Interactive World Model for Autonomous Driving

    [https://arxiv.org/abs/2611.22691](https://arxiv.org/abs/2611.22691)

    该论文提出一种耦合行为建模与视觉渲染的闭环交互式自动驾驶世界模型，通过多智能体强化学习在闭环中微调背景车辆策略以匹配真实交互行为，并利用交互暴露的视角对3D高斯车辆资产进行逐高斯诊断修复，从而同时保证行为真实性与视觉保真度。

    

    世界模型能够生成超越已有记录日志的驾驶场景，使自动驾驶系统可以在未见过的交互中进行训练与评估。然而，现有的世界模型难以同时保证行为真实性和视觉保真度，因为它们忽略了车辆交互与自车视觉观测之间的关系。我们提出了一种将行为建模与视觉渲染相耦合的闭环交互式世界模型。首先，通过模仿学习得到的背景车辆策略在闭环中与能够对渲染图像作出反应的端到端自车规划器进行多智能体强化学习微调，使其交互行为与真实日志中的行为相匹配。其次，这些交互所暴露出的视角被用于指导对每个3D高斯车辆资产的逐高斯诊断，衰减欠观测高斯上依赖视角的颜色，并用生成的高斯填补从未被观测到的表面。

    arXiv:2611.22691v1 Announce Type: cross  Abstract: World models (WMs) generate driving scenarios beyond recorded logs, enabling the training and evaluation of autonomous driving systems in unseen interactions. However, existing WMs struggle to ensure both behavioral realism and visual fidelity, since they overlook the relationship between vehicle interactions and the ego visual observations. We propose a closed-loop interactive WM that couples behavior modeling and visual rendering. First, imitation-learned background-vehicle (BV) policies are fine-tuned with multi-agent reinforcement learning in the closed loop against an end-to-end ego planner that reacts to the rendered images, so that their interaction behaviors match those in real logs. Second, the viewpoints exposed by these interactions guide a per-Gaussian diagnosis of each 3D Gaussian vehicle asset, which attenuates view-dependent colors on under-observed Gaussians and fills never-observed surfaces with generated Gaussians, wh
    
[^72]: CULypso：一种面向车辆自组织网络的集群式联邦遗忘范式

    CULypso: A Clustered Federated Unlearning paradigm for Vehicular Ad-Hoc Networks

    [https://arxiv.org/abs/2611.22682](https://arxiv.org/abs/2611.22682)

    本文提出CULypso，一种专为集群式车辆自组织网络设计的分层联邦遗忘算法，利用网络拓扑与车辆移动性降低通信、资源和时间开销，并能有效遗忘客户端对全局模型更新的影响。

    

    遗忘学习已成为机器学习中一项至关重要的安全与隐私保护解决方案，使网络能够支持客户端依据数据保护法规所享有的数据删除权。而将这样一个并不简单的任务引入边缘的、高度移动且时间敏感的车辆网络，则是一个更为巨大且在很大程度上尚未被探索的挑战。本文论证了分层联邦遗忘方法可以恰当地集成到车辆自组织网络中，通过利用网络拓扑与车辆移动性来降低通信、资源和时间成本，同时有效地遗忘某一客户端对全局模型更新的影响。为实现这一目标，我们提出了CULypso，一种为集群式车辆自组织网络量身定制的近似式、精准式、分层式遗忘算法。该算法受希腊神话中的仙女卡吕普索（Calypso）启发，其名字意为“隐藏知识之人”，CULypso的大写字母即指代……

    arXiv:2611.22682v1 Announce Type: cross  Abstract: Unlearning has emerged as an essential security and privacy-preserving solution in Machine Learning, enabling networks to support their clients' rights for data erasure, as mandated by data protection regulations. Introducing such a non-trivial task to an edge, highly mobile and time-sensitive vehicular network is an even greater, largely unexplored challenge. This article argues that a hierarchical Federated Unlearning approach could be aptly integrated into vehicular ad-hoc networks (VANETs), leveraging topology and movement to reduce communication, resource and time costs, while effectively forgetting a client's influence on the global model updates. In order to achieve this, we propose CULypso, an approximate, surgical, hierarchical unlearning algorithm tailored for clustered VANETs. Inspired by the nymph in Greek mythology, Calypso, whose name translates to 'she who conceals the knowledge', CULypso, with the capital letters refere
    
[^73]: FVScore：规格驱动的神经符号视频评估

    FVScore: Specification-Driven Neuro-Symbolic Video Evaluation

    [https://arxiv.org/abs/2611.22673](https://arxiv.org/abs/2611.22673)

    FVSCORE 提出了一种规格驱动的神经符号视频生成评估框架，通过将视频分解为显式与隐式命题、借助专用感知工具和时序逻辑软算子进行验证，并经显著性感知校准对齐人类感知，最终生成可追溯、有证据支撑的评估报告。

    

    我们提出了忠实性视频评分（FVSCORE），这是一个用于规格驱动视频生成评估的神经符号框架。现有的评估器依赖于整体性的多模态大语言模型（MLLM）判断或视觉-语言指标，然而如何使这些评分与人类判断保持一致，并将其追溯到具体的时间性违规，仍然是悬而未决的挑战。FVSCORE 通过将每个视频编译为显式命题（捕捉提示词要求）和隐式命题（捕捉未明说的物理、时间及跨模态约束）来解决上述局限。它使用专门的感知工具为每个命题提供依据，并用受线性时序逻辑启发的软时序算子评估由此产生的置信度轨迹。随后，显著性感知校准步骤通过仅保留感知上显著的违规，使验证结果与人类感知保持一致，最终生成带有时间戳和证据支撑的报告。我们在……上进行了大量实验（原文在此处截断）。

    arXiv:2611.22673v1 Announce Type: cross  Abstract: We introduce Faithfulness Video Score (FVSCORE), a neuro-symbolic framework for specification-driven video generation evaluation. Existing evaluators rely on holistic MLLM judgments or vision-language metrics, yet aligning their scores with human judgments and tracing them to specific temporal violations remain open challenges. FVSCORE addresses these limitations by compiling each video into explicit propositions capturing prompt requirements and implicit propositions capturing unstated physical, temporal, and cross-modal constraints. It grounds each proposition using specialized perceptual tools and evaluates the resulting confidence traces with soft temporal operators inspired by Linear Temporal Logic. A Significance-Aware Calibration step then aligns verification with human perception by retaining only perceptually significant violations before producing a timestamp- and evidence-grounded report. We conduct extensive experiments acr
    
[^74]: 大型语言模型中的世界观

    Worldviews in Large Language Models

    [https://arxiv.org/abs/2611.22670](https://arxiv.org/abs/2611.22670)

    该论文首次将人类心理学的世界观调查应用于八个大型语言模型，发现LLM普遍持有人类所没有的对世界的悲观看法，这种阴郁源于后训练而非基座模型，而沉思式的系统提示可以让世界观更积极并提升安全性。

    

    大型语言模型（LLM）通常以“人格”（personas）的方式来描述，这一框架已成为对齐研究的核心。在本文中，我们认为存在一个更深层次的描述维度，即智能体的世界观——也就是它对现实是什么以及现实如何运作的信念。我们将LLM视为一组被试，向八个模型施测了来自人类心理学的两份世界观调查问卷。我们展示了这些模型在许多关键的世界观立场上是共通的。例如，与人类相比，LLM一致地将世界描述得更不美好、更不安全、更缺乏吸引力和生命力。另一方面，基座模型具有更积极的观念，这表明我们所发现的阴郁观感是由后训练所驱动的。与关于LLM“人格”的研究相呼应，我们展示了引导不同身份的系统提示会以一致的方向改变模型的世界观，并且一个沉思式的提示能够创造更积极的观念并提升安全性。我们还发现……（原文摘要在此处截断）

    arXiv:2611.22670v1 Announce Type: new  Abstract: Large language models (LLMs) are often described in terms of personas and this framing has become central to alignment research. In this paper, we argue that there is a deeper level of description that lies in the agent's worldviews, i.e., its beliefs about what reality is and how it works. Treating LLMs as a population of subjects, we administer two worldview surveys from human psychology to eight models. We show that the models share many key worldview positions. For example, LLMs consistently describe the world as less good, safe, enticing, and alive than humans do. On the other hand, base models have a more positive outlook, suggesting that the bleakness we found is driven by post-training. In line with research on LLM "personas", we show that system prompts eliciting different identities shift worldviews in consistent directions, and that a contemplative prompt creates a more positive outlook and improves safety. We also found that 
    
[^75]: 自主研究中的轮间决策：一个读取完整轨迹的LLM控制器

    The Between-Round Decision in Autonomous Research: An LLM Controller That Reads the Whole Trajectory

    [https://arxiv.org/abs/2611.22661](https://arxiv.org/abs/2611.22661)

    该论文提出BRED，一个基于LLM的轮间决策控制器，通过维护最佳已验证解决方案（Champion）和过往轮次日志，让LLM直接读取完整轨迹证据来决定是改进现有方案还是开辟新方向，从而摆脱了规则、评分和搜索树等外部机制。

    

    执行一个想法往往会改变它看起来有多有前景，因此自主研究的下一步应该基于结果而非提案。在每轮实验之后，系统必须决定是继续改进最佳解决方案还是开启新方向、以哪个解决方案为基础进行构建、以及设定什么目标。我们将此称为“轮间决策”。现有系统用规则、评分、搜索树和调度等机制来处理这一决策；而在由LLM做出该决策的场合，LLM需要读取哪些内容仍是一个悬而未决的问题。我们探究LLM能否在没有周围搜索结构辅助的情况下，仅凭积累的证据做出这一决策，以及它需要读取多少证据。我们提出了BRED（Between-Round Evidence-based Decision，基于证据的轮间决策），这是一个基于LLM的控制器，它维护迄今为止最佳的已验证解决方案（称为Champion）以及过往轮次的日志，并在每轮做出一次决策。在三个广泛使用的研究智能体基准任务上，BRED优于任务基线（原文摘要在此处截断）。

    arXiv:2611.22661v1 Announce Type: new  Abstract: Executing an idea often changes how promising it looks, so the next step in autonomous research should rest on results, not on proposals. After each round of experiments, the system must decide whether to refine the best solution or start a new direction, which solution to build from, and what goal to set. We call this the between-round decision. Existing systems wrap it in machinery such as rules, scores, search trees, and schedules; where an LLM makes it, what the LLM needs to read is left open. We ask whether an LLM can make this decision from accumulated evidence, without a search structure around it, and how much of that evidence it needs to read. We present BRED (Between-Round Evidence-based Decision), an LLM-based controller that maintains the best verified solution so far (the Champion) and a log of past rounds, and makes one decision per round. On three widely used research-agent benchmark tasks, BRED improves on the task baseli
    
[^76]: AlignedHPO：高效的代理引导损失超参数优化

    AlignedHPO: Efficient Surrogate-Guided Loss Hyperparameter Optimization

    [https://arxiv.org/abs/2611.22658](https://arxiv.org/abs/2611.22658)

    AlignedHPO提出了一种利用梯度对齐和可微代理验证目标的高效损失超参数优化框架，通过雅可比-向量积与归一化对齐等技术，在保持竞争力性能的同时将超梯度计算速度提升近12倍，成功应用于长尾分类和开集人脸识别任务。

    

    现代深度学习损失函数涉及多个超参数，例如权重、尺度和边距，尽管这些超参数具有可微性，但通常仍通过手动调整或昂贵的黑盒优化方法进行调优。我们提出了AlignedHPO，一个利用梯度对齐和代理验证目标来高效调优可微损失超参数的框架。我们的方法采用雅可比-向量积来降低超梯度计算的时间和内存开销，同时可微的度量代理引导优化朝着特定任务的评估标准进行。我们通过归一化对齐和梯度平滑解决了梯度幅度歧义和随机不稳定问题。在广义Focal Loss和ArcFace上的实验表明，该方法在长尾分类和开集人脸识别任务中均能实现有效适应。与隐式微分方法相比，AlignedHPO取得了具有竞争力或更优的性能，同时超梯度计算速度提升近12倍。

    arXiv:2611.22658v1 Announce Type: cross  Abstract: Modern deep learning losses involve multiple hyperparameters, such as weights, scales, and margins, typically tuned manually or through expensive black-box optimization despite their differentiability. We propose AlignedHPO, an efficient framework for tuning differentiable loss hyperparameters using gradient alignment and surrogate validation objectives. Our method employs Jacobian-vector products to reduce hypergradient computation time and memory, while differentiable metric surrogates guide optimization toward task-specific evaluation criteria. We address gradient magnitude ambiguity and stochastic instability through normalized alignment and gradient smoothing. Experiments with generalized Focal Loss and ArcFace demonstrate effective adaptation across long-tailed classification and open-set face recognition. AlignedHPO achieves competitive or improved performance compared to implicit differentiation, with nearly 12x faster hypergra
    
[^77]: EDA：面向证据驱动的多模态抑郁症分析的观察、验证与评估框架

    EDA: Observe, Verify, and Assess for Evidence-Grounded Multimodal Depression Analysis

    [https://arxiv.org/abs/2611.22648](https://arxiv.org/abs/2611.22648)

    EDA框架通过“观察-验证-评估”三阶段流程，将多模态抑郁症评估建立在可追溯、经验证的证据记录之上，显著提升了抑郁分析的来源可查性与决策可解释性。

    

    现有的多模态抑郁症评估方法通常将长时间的录音录像概括为全局性的行为描述，这使得单个观察结果的来源及其与决策的关联性难以审查。我们提出了EDA，一个围绕“观察、验证与评估”三个阶段组织的证据驱动框架。在“观察”阶段，基于文献的声学和视觉线索被定位为与来源关联的证据记录，包含时间区间和行为测量。在“验证”阶段，冻结的多模态智能体检查所引用的媒体是否支持每项观察结果，简化的“支持”准入策略保留通过正面验证的记录及其审查依据。在“评估”阶段，统计描述符和时间事件支持基于参考的行为比较，同时PHQ-9引导的文本分析提供补充性的症状证据。当出现跨模态不一致时，会在最终选择之前触发对原始文本的审查。实验在D-Vlog和

    arXiv:2611.22648v1 Announce Type: new  Abstract: Existing multimodal depression assessment methods often summarize long recordings into global behavioral descriptions. This leaves the source and decision relevance of individual observations difficult to inspect. We propose EDA, an evidence-grounded framework organized around Observe, Verify, and Assess. In Observe, literature-informed acoustic and visual cues are localized into source-linked Evidence Records with temporal intervals and behavioral measurements. In Verify, frozen multimodal agents examine whether the cited media support each observation. The simplified Support admission policy retains positively verified records and their review grounds. In Assess, statistical descriptors and temporal events support reference-based behavioral comparison, while PHQ-9-guided text analysis provides complementary symptom evidence. Cross-modal disagreement triggers review of the original text before final selection. Experiments on D-Vlog and 
    
[^78]: 面向运动控制的仿生脉冲神经网络架构中反相同步的分层控制

    Hierarchical Control of Antiphase Synchronization in Biomorphic Spiking Neural Architectures for Locomotor Control

    [https://arxiv.org/abs/2611.22639](https://arxiv.org/abs/2611.22639)

    该研究提出了一个在闭环神经肌肉回路中识别并稳定仿生脉冲中枢模式发生器反相同步的分层控制框架，通过融合相位相干性、幅度平衡与时间稳定性的复合度量，发现刺激类型（尤其是持续型驱动）是决定同步模式的主导因素。

    

    中枢模式发生器（CPG）为自适应运动控制提供了生物启发的基础，然而在脉冲神经网络中系统地表征其动态模式仍然具有挑战性。本研究提出了一个框架，用于识别和稳定嵌入闭环神经肌肉回路中的仿生脉冲CPG模型的反相同步。研究开发了一种复合同步度量指标，综合了相位相干性、幅度平衡和时间稳定性三个维度，并在系统变化的刺激波形和突触配置下评估其性能。统计分析表明，刺激类型主导着动态模式的确定，其中持续型驱动产生了最稳健的反相同步；周期性输入则会诱发非线性饱和和依赖上下文的网络路由。尽管突触权重本身仅表现出有限的判别能力，但它们显著调节了同步的稳健性。

    arXiv:2611.22639v1 Announce Type: cross  Abstract: Central pattern generators (CPGs) provide a biologically inspired foundation for adaptive locomotor control, yet systematically characterizing their dynamic regimes in spiking neural networks remains challenging. This study introduces a framework for identifying and stabilizing antiphase synchronization in a biomorphic spiking CPG model embedded in a closed neuromuscular loop. A composite synchronization metric is developed, integrating phase coherence, amplitude balance, and temporal stability. Its performance is evaluated across systematically varied stimulation waveforms and synaptic configurations. Statistical analysis reveals that stimulation type dominates regime determination, with tonic drive yielding the most robust antiphase synchronization. Periodic inputs induce nonlinear saturation and context-dependent network routing. Although synaptic weights alone show limited discriminative power, they significantly modulate robustnes
    
[^79]: 正确的诊断，不安全的路径：审计临床AI决策轨迹及纠正性监督的有效性与潜在危害

    Right Diagnosis, Unsafe Route: Auditing Clinical AI Decision Trajectories and Corrective Oversight for Validity and Potential Harm

    [https://arxiv.org/abs/2611.22612](https://arxiv.org/abs/2611.22612)

    该论文提出“轨迹审计”方法，通过逐检查点重放真实住院病例，对临床AI每一步中间决策的患者潜在危害和临床有效性分别评分，弥补了传统评估只关注最终诊断正确性而忽视决策路径安全性的缺陷。

    

    临床AI系统可能通过原本会伤害患者的中间决策达到正确的最终诊断，也可能在修复错误的同时引入新的风险。大多数评估方法只对最终诊断进行打分；即使是逐步式基准测试（即随时间推移逐步发布病例证据的测试），也只评估系统能否得出正确诊断或收集正确信息，而不评估患者所经历的每一步决策是否安全。我们将患者安全作为逐步评估的目标。轨迹审计方法逐个检查点重放真实住院病例，让每个系统仅基于目前 released 的证据进行决策，并对每项检查、治疗和处置进行评分——分别评估其对患者的潜在危害，以及临床有效性（对照记录在案的最终诊断和医生撰写的疾病评分标准，但仅基于决策时可获得的证据）。LLM评判器对每项决策进行评分，医生参考标准……

    arXiv:2611.22612v1 Announce Type: new  Abstract: A clinical AI system can reach a correct final diagnosis through intermediate decisions that could have harmed the patient, and can repair an error while introducing new risks. Most evaluations score final diagnoses; even stepwise benchmarks, which release a case's evidence over time, score whether systems reach the right diagnosis or gather the right information, not whether the decisions a patient passes through are safe. We make patient safety the target of stepwise evaluation. A trajectory audit replays real admissions checkpoint by checkpoint, lets each system decide with only the evidence released so far, and rates every test, treatment and disposition for its potential harm to the patient and, separately, for clinical validity, against the documented final diagnosis and physician-written disease rubrics yet on the evidence available when the decision was made. An LLM judge scores every decision, and a physician reference standard 
    
[^80]: DAQ：面向脉冲Transformer的决策对齐训练后量化方法

    DAQ: Decision-Aligned Post-Training Quantization for Spiking Transformers

    [https://arxiv.org/abs/2611.22611](https://arxiv.org/abs/2611.22611)

    提出决策对齐训练后量化方法DAQ，通过校准量化器使脉冲Transformer神经元的放电决策与浮点参考模型对齐，从而在权重和膜变量量化的同时有效保持神经元的放电行为。

    

    脉冲Transformer将基于注意力的表征学习与稀疏的脉冲通信相结合，但其权重和膜变量仍然带来较大的存储与计算开销。训练后量化（PTQ）为降低这些开销提供了一种实用的途径，然而在权重和膜变量量化的情况下保持神经元的放电行为仍具有挑战性。本文提出了决策对齐训练后量化（DAQ），该方法通过校准量化器，使神经元的放电决策与浮点参考模型保持一致。我们分析了这些量化器产生的误差如何影响决策边界处的放电行为，以及这些误差如何在循环神经元中随时间步传播。基于这一分析，我们构建了一个优化目标，鼓励量化后的膜变量沿着量化模型自身的状态轨迹保持在参考放电区域内。在多种脉冲Transformer上的实验验证了该方法的有效性。

    arXiv:2611.22611v1 Announce Type: new  Abstract: Spiking transformers combine attention-based representation learning with sparse spike communication, yet their weights and membrane variables still incur substantial storage and computation overhead. Post-training quantization (PTQ) offers a practical way to reduce these overheads, but preserving neuronal firing under weight and membrane quantization remains challenging. In this paper, we propose Decision-Aligned Post-Training Quantization (DAQ), a method that calibrates the quantizers to align neuronal firing decisions with those of a floating-point reference. We analyze how errors from these quantizers affect firing at decision boundaries and how they propagate over time steps in recurrent neurons. Based on this analysis, we formulate an objective that encourages quantized membrane variables to remain within the reference firing regions along the quantized model's own state trajectory. Experiments across multiple spiking transformer a
    
[^81]: EvoSkill-S1：基于遗传启发算子的可复用技能演化

    EvoSkill-S1: Evolving Reusable Skills with Genetic-Inspired Operators

    [https://arxiv.org/abs/2611.22610](https://arxiv.org/abs/2611.22610)

    该论文提出 EvoSkill-S1 框架，首次将可复用技能本身作为演化的直接对象，利用五种遗传启发算子和强化学习信号对语言模型智能体的技能库进行深层、可控的进化，从而突破了以往仅靠 rollout 积累和启发式反思进行间接、浅层技能更新的局限。

    

    可复用技能使语言模型智能体能够积累并复用解决问题的经验。然而，现有的大多数方法只能通过基于 rollout 的经验积累和启发式反思来间接改进技能库，这往往导致更新浅层化且控制力较弱。我们提出了 EvoSkill-S1，一个由强化学习驱动的框架，它将可复用技能本身作为演化的直接对象。面对困难负例失败，EvoSkill-S1 选择父代技能，并通过五种受遗传学启发的算子将其转化为候选子代技能：修订、交叉修订、变异、双重反思和交叉重组。下游任务奖励用于训练技能演化器以生成更有效的子代技能，同时由评估器引导的回写机制选择性地保留有益的更新。互补技能库融合将特定于算子的技能库整合起来，以利用它们各自不同的恢复模式，而程序化技能接口支持……（原文摘要在此处截断）

    arXiv:2611.22610v1 Announce Type: new  Abstract: Reusable skills allow language-model agents to accumulate and reuse problem-solving experience. However, most existing methods improve a skill bank only indirectly through rollout-driven accumulation and heuristic reflection, often resulting in shallow and weakly controlled updates. We present EvoSkill-S1, an RL-driven framework that instead treats reusable skills themselves as the direct objects of evolution. Given hard-negative failures, EvoSkill-S1 selects parent skills and transforms them into candidate child skills through five genetic-inspired operators: revision, cross-revision, mutation, dual-reflection, and crossover. Downstream task rewards train the skill evolver to generate more effective child skills, while evaluator-guided write-back selectively retains beneficial updates. Complementary Skill Bank Fusion consolidates operator-specific banks to exploit their distinct recovery patterns, while a Procedural Skill Interface supp
    
[^82]: 简单就能走得很远：用示范条件提示搜索审计提示优化器

    Simplicity Goes Far: Auditing Prompt Optimizers with Demonstration-Conditioned Prompt Search

    [https://arxiv.org/abs/2611.22605](https://arxiv.org/abs/2611.22605)

    该研究将各类提示优化器共享的“生成-选择”步骤剥离为示范条件提示搜索（DCPS），证明这个不含学习更新、反思或进化的简单方法即可在约三分之一的成本下匹敌 PPO 与 GEPA 等复杂优化器的效果。

    

    自动提示优化正变得日益复杂。近期方法叠加了策略梯度更新、轨迹反思和进化搜索。这些方法都共享一个“生成并选择”步骤；若不对其加以控制，所报告的性能提升可能会混淆优化器机制与更大采样预算各自的贡献。我们将这一步骤单独分离出来，称为示范条件提示搜索，并将自动提示工程师（APE）的方法改造为一种机制归因对照：它冻结生成器，以随机采样的示例作为条件，在验证数据上为候选提示打分并保留最优者——不涉及任何学习更新、反思或进化。在两个优化器系列上，DCPS 在 StablePrompt 中取代了近端策略优化（PPO），以约三分之一的实测成本取得相差不超过 0.5 个百分点的分类准确率；在基于 DSPy 的复合 AI 系统中，在所报告的预算下，它在 GPT-……（摘要原文在此截断）

    arXiv:2611.22605v1 Announce Type: new  Abstract: Automatic prompt optimization is growing more elaborate. Recent methods stack policy-gradient updates, trace reflection, and evolutionary search. These methods share a generate-and-select step; without controlling for it, reported gains may conflate the contributions of optimizer mechanisms and larger sampling budgets. We isolate this step as Demonstration-Conditioned Prompt Search (DCPS), repurposing the Automatic Prompt Engineer (APE) recipe as a mechanism-attribution control: it freezes the generator, conditions it on randomly sampled demonstrations, scores candidates on validation data, and keeps the best---no learned update, reflection, or evolution. Across two optimizer families, DCPS replaces Proximal Policy Optimization (PPO) in StablePrompt, matching classification accuracy within 0.5 pp at roughly one-third the measured cost; in DSPy-based compound AI systems, under the reported budgets, it matches GEPA's macro accuracy on GPT-
    
[^83]: COTTI：面向促销漂移下鲁棒工业推荐的对比训练与时间感知推理

    COTTI: Contrastive Training and Time-Aware Inference for Robust Industrial Recommendation under Promotional Drift

    [https://arxiv.org/abs/2611.22597](https://arxiv.org/abs/2611.22597)

    COTTI 通过跨周期对比对齐、主动重加权和自适应边界对齐，将历史表征学习与促销事件时刻的分数调整相分离，在促销引起的分布漂移下实现鲁棒的工业推荐，并在飞猪数据上取得了更优的离线 AUC。

    

    促销事件会在不同的时间尺度上改变流量构成、购买行为以及转化概率的量级。我们提出了 COTTI（对比训练与时间感知推理），这是一个将历史表征学习与事件时刻分数调整相分离的工业推荐框架。其中，跨周期对比对齐（CPC）通过自适应一致性权重对同一用户的成对历史进行正则化；主动重加权（RW）通过历史兴趣距离控制促销样本的影响；自适应边界对齐（ABA）将随机预测变异性与时间事件核相结合，在保持预测器参数固定的情况下对 logits 进行调整。我们对加权目标进行了分析，并对 ABA 引起的概率和排序变化给出了理论上界。在飞猪数据上，COTTI 取得了 0.8127 的离线 AUC，相比之下 TEN 为 0.8101，另一对比方法为 0.8123。

    arXiv:2611.22597v1 Announce Type: new  Abstract: Promotional events shift traffic composition, purchase behavior, and the scale of conversion probabilities on different time horizons. We present \textbf{COTTI} (\textbf{CO}ntrastive \textbf{T}raining and \textbf{T}ime-aware \textbf{I}nference), an industrial recommendation framework that separates historical representation learning from event-time score adjustment. Cross-Period Contrastive Alignment (CPC) regularizes paired histories of the same user with adaptive consistency weights. Active Re-weighting (RW) controls the influence of promotional examples through historical-interest distance. Adaptive Boundary Alignment (ABA) combines stochastic prediction variability with a temporal event kernel to adjust logits while keeping predictor parameters fixed. We analyze the weighted objective and bound ABA-induced probability and ranking changes. On Fliggy data, COTTI attains an offline AUC of 0.8127, compared with 0.8101 for TEN and 0.8123 
    
[^84]: PACT：通过保留正确计划与修复动作实现GUI智能体的计划-动作一致性训练

    PACT: Plan-Action Consistency Training by Preserving Correct Plans and Repairing Actions in GUI Agents

    [https://arxiv.org/abs/2611.22591](https://arxiv.org/abs/2611.22591)

    PACT是一个针对GUI智能体的训练框架，它在结果监督强化学习中通过屏蔽已验证计划token上的负策略梯度来保护正确的规划，同时在推理上下文不变的情况下向参考动作修复执行行为，从而弥合计划与动作之间的执行鸿沟并解决信用分配冲突。

    

    GUI智能体能够在推理中制定出正确的即时计划，却可能发出无法执行该计划的动作，从而产生“执行鸿沟”。这种反复出现的不匹配会损害可靠的任务完成，因为单个执行错误就可能使多步骤工作流偏离正轨。在以结果监督的强化学习中，将负的响应级优势广播到所有生成的token上，可能会在惩罚错误执行的同时连带上有效的规划。为了解决这一信用分配冲突，我们提出了PACT（计划-动作一致性训练），这是一个将参考对齐的计划保护与计划对齐的动作修复相结合的训练框架，且不改变原有的标量动作奖励。通过利用标注的参考动作来识别执行鸿沟，PACT通过屏蔽计划token跨度上的负策略梯度贡献来保护已验证的计划。在保持生成的推理内容不变作为上下文的前提下，它进而倾向于参考动作（原文在此处截断）。

    arXiv:2611.22591v1 Announce Type: new  Abstract: GUI agents can formulate a correct immediate plan in their reasoning yet emit an action that fails to execute it, creating an execution gap. This recurring mismatch undermines reliable task completion, as a single execution error can derail a multi-step workflow. In outcome-supervised reinforcement learning, broadcasting a negative response-level advantage across all generated tokens risks penalizing valid planning together with erroneous execution. To address this credit-assignment conflict, we introduce PACT (Plan-Action Consistency Training), a training framework that combines reference-aligned plan protection and plan-aligned action repair without altering the original scalar action reward. Using annotated reference actions to identify execution gaps, PACT protects the verified plan by masking negative policy-gradient contributions across its token span. Keeping the generated reasoning unchanged as context, it then favors the referen
    
[^85]: 自适应LLM驱动的蜜罐对决LLM驱动的攻击者：叙事连贯性能否提升欺骗效果？

    Adaptive LLM-Driven Honeypots vs. LLM-Driven Attackers: Does Narrative Coherence Improve Deception?

    [https://arxiv.org/abs/2611.22589](https://arxiv.org/abs/2611.22589)

    本文提出“叙事欺骗”反应式策略，通过将攻击者命令映射到MITRE ATT&CK技术并部署相互关联的连贯欺骗场景，探索叙事连贯性是否能比不相关内容更有效地留住LLM驱动的自主攻击者。

    

    大型语言模型（LLM）使蜜罐能够按需生成响应和欺骗性制品，但现有方法一次只评估一条生成的回复，对于共同描述一个连贯事件的制品是否比互不相关的制品更能留住攻击者，这一问题尚无答案。随着自主的LLM驱动的攻击代理成为现实对手，这个问题变得日益紧迫。本文提出了“叙事欺骗”这一反应式策略：防御者将观察到的每条攻击者命令映射到MITRE ATT&CK技术，从基于MITRE Engage整理的场景目录中选择兼容的场景，并在入侵进行过程中在模拟的Ubuntu主机上部署相互引用关联的制品（文件、进程、服务和日志）。一个由LLM驱动的防御者与一个自主的LLM攻击者通过真实的SSH会话进行交互，攻击者的终止策略可用于区分因怀疑遭遇欺骗而放弃入侵的情况。

    arXiv:2611.22589v1 Announce Type: cross  Abstract: Large Language Models (LLMs) enable honeypots to generate responses and deceptive artefacts on demand, yet existing approaches evaluate generated content one reply at a time and leave open whether artefacts that jointly describe a coherent event retain an attacker better than unrelated ones. This question becomes pressing as autonomous LLM-driven offensive agents emerge as realistic adversaries. This paper proposes narrative deception, a reactive strategy in which the defender maps each observed attacker command to a MITRE ATT&CK technique, selects a compatible scenario from a curated catalogue based on MITRE Engage, and deploys mutually referential artefacts (files, processes, services, and logs) on an emulated Ubuntu host while the intrusion is under way. An LLM-powered defender and an autonomous LLM attacker interact through a real SSH session, and the attacker's termination policy distinguishes abandonment due to suspected deceptio
    
[^86]: 火中进化：利用自适应注入压力源演化智能体安全防护机制

    Evolution Under Fire: Evolving Agent Safety Harnesses with Adaptive Injection Stressors

    [https://arxiv.org/abs/2611.22583](https://arxiv.org/abs/2611.22583)

    该论文提出分阶段框架 Evolution Under Fire，通过让安全防护机制与自适应注入攻击压力源相互对抗演化，自动生成可执行的安全防护，在 AgentDyn 基准上将攻击成功率从 50.4% 降至 3.0% 且不牺牲任务效用。

    

    使用工具的语言模型智能体必须基于来自不可信电子邮件、网页和文档的信息采取行动。这带来了一个核心安全挑战：恶意指令与合法任务指令可能通过同一交互渠道到达。现有防御大多为手工设计且静态不变，使得进一步的改进依赖于反复的人工重新设计。我们提出了 Evolution Under Fire（火中进化），这是一个分阶段框架，将安全防护机制的演化与自适应压力源的演化相耦合。在保持任务模型、基准、工具和评估器固定不变的前提下，采用效用约束、按帕累托前沿调度的树搜索，从执行证据中演化出完整可执行的安全防护机制。随后，自适应压力源演化会发现保持效用的攻击，用以指导后续阶段的防护机制演化。在基于 GPT-4o-mini 的 AgentDyn 上，演化出的防护机制将攻击成功率从 50.4% 降至 3.0%，并在干净和受攻击场景下均提升了效用。

    arXiv:2611.22583v1 Announce Type: cross  Abstract: Tool-using language-model agents must act on information from untrusted emails, webpages, and documents. This creates a central security challenge: malicious instructions and legitimate task directives can arrive through the same interaction channel. Existing defenses are largely hand-designed and static, making further improvements dependent on repeated manual redesign. We introduce Evolution Under Fire, a staged framework that couples safety-harness evolution with adaptive stressor evolution. Utility-constrained, Pareto-scheduled tree search evolves complete executable safety harnesses from execution evidence while keeping the task model, benchmark, tools, and evaluator fixed. Adaptive stressor evolution then discovers utility-preserving attacks that guide a subsequent stage of harness evolution. On AgentDyn with GPT-4o-mini, an evolved harness reduces attack success from 50.4% to 3.0%, improves clean and attacked utility over the in
    
[^87]: 大语言模型能理解网络流量轨迹吗？一项关于原始与过滤流量数据推理能力的对比研究

    Do LLMs Understand Network Traces? A Comparative Study of Reasoning over Raw and Filtered Traffic Data

    [https://arxiv.org/abs/2611.22575](https://arxiv.org/abs/2611.22575)

    本研究提出了一种可复现的、基于问答的评估方法，结合经过人工验证的“LLM作为裁判”自动化流水线，系统评估了大语言模型在网络流量轨迹分析中处理原始与协议过滤数据的推理能力。

    

    ICT（信息通信技术）服务的有效交付与监控在很大程度上依赖于网络流量分析。然而，解读复杂的数据包捕获仍然是一项具有挑战性的任务，需要大量的专业技术知识。大语言模型（LLMs）为自动化网络流量调查提供了有前景的机遇，但其在数据包级字段提取和流级序列重建方面的可靠性仍然是一个悬而未决的研究问题。本研究提出了一种可复现的、基于问答的评估方法，以评估大语言模型在网络流量轨迹分析方面的能力。该方法使用来自CICIDS2017数据集的真实良性流量，将多个模型与人工生成的真实标准进行对比评估，比较它们在原始文本流量转储和协议过滤表示两种形式上的表现。模型响应通过自动化的“LLM作为裁判”流水线进行评估，该流水线已经过人工评估的验证。

    arXiv:2611.22575v1 Announce Type: cross  Abstract: The effective delivery and monitoring of ICT services rely heavily on network traffic analysis. However, interpreting complex packet captures remains a challenging task that requires substantial technical expertise. Large language models (LLMs) offer promising opportunities to automate network traffic in- vestigation, yet their reliability in packet-level field extraction and flow-level sequence reconstruction remains an open research question. This study proposes a reproducible, question-answering- based evaluation methodology to assess the capabilities of LLMs in network trace analysis. Using real-world benign traffic from the CICIDS2017 dataset, we evaluate multiple models against a human-generated ground truth, comparing their performance on raw textual traffic dumps and protocol-filtered representations. Model responses are assessed through an automated LLM- as-a-judge pipeline, validated against human evaluations. The results dem
    
[^88]: MillBill：一种具有几何反馈的加工智能体

    MillBill: A Machining Agent with Geometric Feedback

    [https://arxiv.org/abs/2611.22565](https://arxiv.org/abs/2611.22565)

    MillBill是一个将加工决策与几何执行验证相分离的加工智能体，通过几何工具框架的模拟与反馈机制，使中等规模的开源语言模型也能生成高精度的数控加工程序。

    

    将CAD模型转化为加工程序既需要工艺知识，也需要几何精度。语言模型可以通过为每个零件编写几何和刀具路径代码来生成这样的工作流程，但这使得数值执行成为了模型任务的一部分。我们提出了MillBill，一个将加工决策与几何执行和验证相分离的智能体。给定一个边界表示（B-rep）的零件、毛坯和材料，一个本地托管的中等规模开源模型为分度式三轴加工提出加工操作和切削参数。一个几何工具框架负责生成刀具路径、模拟材料去除，并返回反馈以优化加工方案。装夹规划考虑的是可用刀具实际能够物理触及的范围，而不仅仅是可见性。我们通过模拟加工后目标表面积在规定公差内的比例来评估所生成的程序。使用Qwen3.8-27B模型，MillBill的表现超越了前沿……

    arXiv:2611.22565v1 Announce Type: new  Abstract: Turning a CAD model into a machining program requires both process knowledge and geometric precision. Language models can generate such workflows by writing geometry and toolpath code for each part, but this makes numerical execution part of the model's task. We introduce MillBill, an agent that separates machining decisions from their geometric execution and verification. Given a boundary-representation part, stock, and material, a locally hosted, mid-sized open model proposes operations and cutting parameters for indexed three-axis machining. A geometric harness generates toolpaths, simulates material removal, and returns feedback for plan refinement. Setup planning accounts for what the available cutters can physically reach, rather than visibility alone. We evaluate the resulting programs by the fraction of the target surface area within the specified tolerance after simulated machining. With Qwen3.8-27B, MillBill outperforms frontie
    
[^89]: CADMAGEDDON: 基于本地推理的高保真CAD重建

    CADMAGEDDON: High-Fidelity CAD Reconstruction with Local Inference

    [https://arxiv.org/abs/2611.22563](https://arxiv.org/abs/2611.22563)

    CADMAGEDDON是一个可在单GPU系统和笔记本电脑上本地运行的开源智能体，通过将CAD重建任务分解为简单视觉决策链并借助几何工具实现精确构建，使用中等规模模型在所有主要基准上取得了超越前沿模型系统的可编辑CAD重建质量。

    

    CAD从业者经常需要处理不愿与外部服务共享的专有零件。对于个人和小型公司而言，将这项工作内部化还需要他们能够负担得起的计算资源。我们提出了CADMAGEDDON，这是一个开源智能体，可以从网格重建可编辑的CAD模型，并且可以在单GPU系统和当代笔记本电脑上本地运行。该方法将这一复杂任务分解为一系列关于构建什么和修正什么的简单视觉决策。几何工具支持精确构建，使智能体能够专注于视觉推理。使用中等规模的Qwen3.8-27B模型，CADMAGEDDON在所有主要基准测试中的重建质量均超越了先前的方法，包括基于前沿模型的系统。CADMAGEDDON与FreeCAD集成，将其重建结果无缝融入从业者现有的CAD工作流程中。

    arXiv:2611.22563v1 Announce Type: new  Abstract: CAD practitioners often work with proprietary parts that they are unwilling to share with external services. For individuals and small companies, bringing this work in-house also requires computational resources they can afford. We introduce CADMAGEDDON, an open-source agent that reconstructs editable CAD from meshes and can run locally on single-GPU systems and current-generation laptops. The method decomposes this complex task into a chain of simple visual decisions about what to construct and what to correct. Geometric tools support precise construction, allowing the agent to focus on visual reasoning. Using the mid-sized Qwen3.8-27B model, CADMAGEDDON outperforms prior approaches, including systems based on frontier models, in reconstruction quality on all the main benchmarks. CADMAGEDDON integrates with FreeCAD, bringing its reconstructions into practitioners' existing CAD workflows. Code: https://github.com/kulibinai/cadmageddon
    
[^90]: JointGV：利用并行图动力学实现结构感知的视觉世界模型校正，以支持长时程建模

    JointGV: Structure-Aware Correction of Visual World Models with Parallel Graph Dynamics for Long-Horizon Modeling

    [https://arxiv.org/abs/2611.22543](https://arxiv.org/abs/2611.22543)

    JointGV提出了一种即插即用的图-视觉联合动力学校正方法，通过并行推演任务条件下的场景图、耦合其与视觉世界模型的预测变化并施加有界残差校正，在不重构视觉模型的前提下抑制实体位姿与空间关系的递归误差，从而显著提升长时程预测能力。

    

    视觉世界模型在动作条件下预测未来的视觉状态，但实体位姿和空间关系中的递归误差限制了其面向机器人规划与控制的长时程预测能力。我们探究是否能够在不重构视觉模型的前提下，利用单独预测的关系状态来校正这种漂移。这项任务具有挑战性，因为图节点与视觉状态既不共享元素，也不共享特征空间。我们提出了联合图-视觉动力学校正方法JointGV，这是一个基于跨粒度动力学假设的插件：尽管表示形式不同，图节点变化与视觉状态变化描述的是同一种物理转变。JointGV在视觉世界模型运行的同时并行推演一个以任务为条件的场景图，将两者的预测变化相互耦合，并将由此获得的图运动上下文与图预测的节点位置相结合，以计算出有界残差。图学习与校正的梯度不会传播到（视觉模型内部）……

    arXiv:2611.22543v1 Announce Type: cross  Abstract: Visual world models predict future visual states conditioned on actions, but recursive errors in entity poses and spatial relations limit their long-horizon prediction for robot planning and control. We ask whether a separately predicted relational state can correct this drift without restructuring the visual model. This is challenging because graph nodes and visual states share neither elements nor a feature space. We propose Joint Graph--Visual Dynamics Correction (JointGV), a plug-in based on a cross-granularity dynamics hypothesis: despite their different representations, graph-node and visual-state changes describe the same physical transition. JointGV rolls out a task-conditioned scene graph alongside the visual world model, couples their predicted changes, and combines the resulting graph-motion context with graph-predicted node positions to compute a bounded residual. Graph-learning and correction gradients do not propagate int
    
[^91]: 从感知到模拟：揭示并赋能多模态大语言模型中的心智空间动态

    From Perception to Simulation: Unveiling and Empowering Mental Spatial Dynamics in MLLMs

    [https://arxiv.org/abs/2611.22541](https://arxiv.org/abs/2611.22541)

    该论文提出MentalGrid基准以评估多模态大语言模型在无中间视觉反馈下模拟动态空间状态的能力，并通过三阶段课程学习框架训练出SpatialWalker模型，从而显著弥补了现有模型在空间模拟上的感知与多步推理缺陷。

    

    空间推理是多模态大语言模型（MLLM）理解世界的基本组成部分。尽管现有研究已推进了静态场景中空间关系的感知，但动态空间模拟这一领域仍相对未被探索。为弥合这一空白，我们提出了MentalGrid，一个专注于空间推理任务的基准，这些任务要求模型在没有中间视觉反馈的情况下模拟不断演变的空间状态。在我们基准上的广泛评估揭示了现有模型空间模拟能力的显著缺陷。具体而言，我们发现在简单模拟场景中性能受限于感知局限，而随着复杂性增加，性能则受制于多步推理能力的不足。为应对这些挑战，我们提出了一个利用16k专门空间推理样本的三阶段课程学习框架。该方法产出了SpatialWalker，

    arXiv:2611.22541v1 Announce Type: cross  Abstract: Spatial reasoning is a fundamental component of world understanding for Multimodal Large Language Models (MLLMs). While existing research has advanced the perception of spatial relations in static scenes, the domain of dynamic spatial simulation remains relatively unexplored. To bridge this gap, we introduce MentalGrid, a benchmark focused on spatial reasoning tasks that require models to simulate evolving spatial states without intermediate visual feedback. Extensive evaluation on our benchmark reveals a significant deficiency in the spatial simulation capabilities of existing models. Specifically, we find that performance is constrained by perceptual limitations in simple simulation scenarios and hindered by inadequate multi-step reasoning as complexity increases. To address these challenges, we propose a three-stage curriculum learning framework utilizing 16k specialized spatial reasoning samples. This approach yields SpatialWalker,
    
[^92]: JET：面向仓库级问答中高效工具调用智能体的“刚好足够探索”调优方法

    JET: Just-Enough Exploration Tuning for Efficient Tool-Calling Agents in Repository-Level Question Answering

    [https://arxiv.org/abs/2611.22539](https://arxiv.org/abs/2611.22539)

    JET通过在信息充足的步骤截断训练轨迹并插入理由说明进行加权微调，有效缓解了LLM智能体在仓库级问答中的“过度探索”问题，使智能体在收集到足够信息后立即作答。

    

    在仓库级问答任务中，智能体使用工具对代码仓库进行探索，并在收集到足够信息后应当回答问题。然而，我们发现大语言模型（LLM）智能体即使在收集的信息已足以回答问题时，仍经常继续调用工具。我们将这种现象称为“过度探索”。此外，尽管现有方法通过对专家轨迹进行监督微调（SFT）来提升任务性能，但SFT也会将专家轨迹中的过度探索行为传播给智能体。为了缓解过度探索问题，我们提出了“刚好足够探索调优”，这是一种微调方法，训练智能体在收集的信息足够时立即作答。JET在每条训练轨迹中信息变得充足的步骤处进行截断，插入一段解释智能体为何可以在该步骤回答的理由说明，并以更高的损失权重对该理由说明进行微调。

    arXiv:2611.22539v1 Announce Type: cross  Abstract: In repository-level question answering (QA), an agent explores a code repository with tools and should answer a question once enough information is gathered. Nevertheless, we find that LLM agents often continue to invoke tools even after the gathered information is already sufficient to answer. We call this phenomenon over-exploration. Moreover, even though existing methods improve task performance through supervised fine-tuning (SFT) on expert trajectories, SFT also propagates the over-exploration in the trajectories to the agent. To mitigate over-exploration, we propose Just-Enough Exploration Tuning (JET), a fine-tuning method that trains the agent to answer once the gathered information is sufficient. JET truncates each training trajectory at the step where the gathered information becomes sufficient, inserts a rationale that explains why the agent can answer at that step, and fine-tunes the agent with a higher loss weight on the r
    
[^93]: 在从未见过的动作上测试世界模型：零售业中的留出计划、预测准确度与价格规则

    Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail

    [https://arxiv.org/abs/2611.22537](https://arxiv.org/abs/2611.22537)

    该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。

    

    世界模型被信任来回答“如果……会怎样”的问题：即它的世界如何响应一个它从未见过的动作。我们为任何以动作为条件的世界模型提供了一种检验这种信任的测试：通过设计将动作从训练数据中留出，并对照已知答案进行评分；检验预测准确度能否预测模型在这些动作上的表现；以及将相同的结构应用于更简单的模型。我们在零售场景中运行该测试，其中动作是价格和促销计划，共涉及2,853个拟合模型。首先，预测准确度是在不同类型模型之间进行比较的合理依据（与新计划上的误差的秩相关性为0.32至0.66），但对于同一世界模型的不同版本之间则是一个糟糕的依据：在商家将计划安排到高需求周的情况下，这种相关性会发生反转，低至-0.69。其次，训练世界模型学习动作的效果（JEPA预训练、潜在一致性检查、计划日期处的适配）并不会降低其在新计划上的误差。第三，在生成的世界中，记录的常规价格和三个p……（摘要在此处截断）

    arXiv:2611.22537v1 Announce Type: cross  Abstract: A world model is trusted to answer "what if": how its world responds to an action it has never seen. We give a test of that trust for any action-conditioned world model: actions held out of training by design and scored against a known answer; whether forecast accuracy predicts accuracy on those actions; and the same structure given to simpler models. We run it in retail, where the action is a price and promotion plan, across 2,853 fitted models. First, forecast accuracy is a fair guide between kinds of model (rank correlation with error on new plans 0.32 to 0.66) and a poor guide between versions of one world model: where merchants plan into high-demand weeks the link reverses, down to -0.69. Second, training the world model to learn the effect of actions (JEPA pretraining, a latent consistency check, adaptation at the plan date) leaves its error on new plans unchanged. Third, in generated worlds the recorded regular price and three p
    
[^94]: DGF-Asym：面向噪声与故障感知下集体估计的自适应同伴加权方法

    DGF-Asym: Adaptive Peer Weighting for Collective Estimation under Noisy and Faulty Sensing

    [https://arxiv.org/abs/2611.22532](https://arxiv.org/abs/2611.22532)

    提出了一种基于回顾性预测性能进行非对称同伴加权的去中心化集体估计方法DGF-Asym，无需先验噪声统计或故障模型，即可在噪声与故障感知条件下实现更稳定的无模型共识估计。

    

    在不可靠感知条件下的集体估计要求机器人自适应地平衡自身观测与来自同伴的社会信息。我们提出了DGF-Asym，一种去中心化的集体估计方法，它扩展了DeGroot-Friedkin模型，引入了基于回顾性预测性能的非对称同伴加权机制。该方法无需先验的传感器噪声统计或显式故障模型即可支持共识形成，同时保持机器人间通信的异步性。我们在噪声观测和时变传感器故障下的仿真集体导航场景中评估了DGF-Asym。与三个基线方法相比，该方法产生了更稳定的估计结果，并能更可靠地维持决策准则，但代价是更长的决策时间。然而，整体性能取决于故障比例、群体规模以及影响阈值的选择。我们的结果证明了自适应同伴加权在实现稳定的无模型集体估计方面的潜力。

    arXiv:2611.22532v1 Announce Type: cross  Abstract: Collective estimation under unreliable sensing requires robots to adaptively balance personal observations and social information from peers. We propose DGF-Asym, a decentralized collective estimation method extending the DeGroot-Friedkin model with asymmetric peer weighting based on retrospective predictive performance. The method supports consensus formation without prior sensor-noise statistics or explicit fault models while keeping inter-robot communication asynchronous. We evaluate DGF-Asym in simulated collective navigation under noisy observations and time-varying sensor faults. Compared with three baselines, it produces more stable estimates and more reliably sustains the decision criterion, at the cost of longer decision times. However, overall performance depends on fault ratio, group size, and influence-threshold selection. Our results demonstrate the potential of adaptive peer weighting for stable model-free collective esti
    
[^95]: 当智能体技能过时：演化环境中的运行时行为与选择性维护

    When Agent Skills Go Stale: Runtime Behavior and Selective Maintenance in Evolving Environments

    [https://arxiv.org/abs/2611.22525](https://arxiv.org/abs/2611.22525)

    该论文发现环境变化会导致智能体复用的技能状态过时，使性能下降超过10个百分点或执行成本增加近40%，并提出了一种无需额外模型调用的“选择性维护”方法来验证技能是否仍然有效。

    

    智能体技能利用过去的经验和当前的观察来指导相关任务上的决策与行动。这些经验被保留在技能状态中，其复用可以摊销推理时的推理开销与环境交互成本。当环境发生变化时，该状态中的部分内容可能变得过时，而其他部分仍然有用。我们在四个任务设置和两个模型上，比较了环境变化前后的技能复用效果。与不使用技能相比，先前有益的技能可能使任务性能降低超过10个百分点，或使执行成本增加近40%。任务成功率也可能保持不变，而执行节省却下降。这些效应取决于哪些环境特定的细节变得过时，以及智能体利用当前观察进行补偿的能力。我们提出了选择性维护方法，它在无需额外模型调用的情况下，重放成功执行中的检查，以验证观察到的差异是否混淆了技能的价值……

    arXiv:2611.22525v1 Announce Type: new  Abstract: Agent skills use past experience and current observations to guide decisions and actions on related tasks. This experience is retained in skill state, whose reuse can amortize inference-time reasoning and environment interaction. When environment changes, parts of this state can become outdated while others remain useful. Across four task settings and two models, we compare skill reuse before and after environment changes. Previously beneficial skills can reduce task performance by over 10 percentage points or increase execution cost by nearly 40% relative to no skill. Task success can also remain unchanged while execution savings decline. These effects depend on which environment-specific details become outdated and the agent's ability to compensate using current observations. We introduce selective maintenance, which replays checks from successful executions without additional model calls. It verifies whether observed differences confl
    
[^96]: 正确的标签，错误的理由：面向多标签可解释性的带符号特征归因

    Right Labels, Wrong Reasons: Signed Feature Attribution for Multi-Label Explainability

    [https://arxiv.org/abs/2611.22515](https://arxiv.org/abs/2611.22515)

    提出FLEX，一种基于多标签Fisher线性判别分析的全局代理解释器，能在单一连贯坐标系中为每个标签生成高保真度的带符号特征归因，尤其在标签多且正例稀少时表现优异。

    

    当一个多标签分类器给出正确的标签时，它可能是出于错误的理由做到的。弥散的特征组合可能因分类器家族而异，且缺乏占主导地位的特征。本文提出了FLEX（Fisher LDA解释器），一种全局代理模型，它在完整的二值标签矩阵上拟合单个直接的多标签Fisher线性判别分析（LDA），从而在单一连贯的坐标系中为每个标签生成带符号的特征归因列表。在所评估的基准数据集和分类器家族上，FLEX以持续较高的保真度复现了黑盒模型的预测。在大多数数据集上，跨标签池化协方差的做法优于按标签分别处理的方法，且在标签数量多、正例稀少的场景下优势最为显著。FLEX的保真度-复杂度帕累托曲线呈单调递增，为使用者提供了一条可预测的路径：随着使用更多特征，即可获得保真度更高的解释，其样本内保真度……

    arXiv:2611.22515v1 Announce Type: cross  Abstract: When a multi-label classifier assigns the right labels, it may do so for the wrong reasons. Diffuse feature combinations may vary across classifier families and lack a dominant feature. This paper proposes FLEX (Fisher LDA Explainer), a global surrogate that fits a single direct multi-label Fisher LDA over the full binary label matrix, producing signed feature attribution lists for each label in a single coherent coordinate system. Across the evaluated benchmarks and classifier families, FLEX reproduces the black box's predictions with consistently high fidelity. Pooling covariance across labels outperforms a per-label alternative on most datasets, with the advantage growing largest where labels are numerous and positive examples are scarce. FLEX's fidelity-complexity Pareto curve is monotonically increasing, giving practitioners a predictable path to higher-fidelity explanations as more features are used, and its in-sample fidelity ma
    
[^97]: LMLineage：面向大语言模型智能体的内容溯源与上下文治理

    LMLineage: Content Lineage and Context Governance for LLM Agents

    [https://arxiv.org/abs/2611.22511](https://arxiv.org/abs/2611.22511)

    LMLineage是一个无需模型白盒访问、无需额外推理的带外溯源系统，可将智能体输出的每句话归因到输入片段并给出校准置信度，并基于溯源元数据实现概念漂移微压缩，仅精准压缩不再被引用的输入文档同时逐字保留用户指令。

    

    对AI智能体的可解释性不应仅局限于理解模型的内部权重，还必须覆盖智能体在多源上下文之上的推理与决策过程。我们提出了LMLineage，这是一种带外（out-of-band）系统，能够将智能体会话中的每一句输出归因到其所依据的输入片段，并给出经过校准的置信度，且无需对模型的白盒访问权限，也无需任何额外的推理过程。在溯源元数据的基础上，我们构建了概念漂移微压缩（concept-drift micro compaction）机制：系统不再等到上下文长度达到上限后将所有内容进行摘要压缩，而是检测先前使用的输入文档何时不再被引用（即发生“漂移”），并仅对这部分文档进行精准压缩，同时逐字保留用户指令。我们在一个多层级的语料库上进行了评估：包括来自10个中文网页问答案例的71条黄金标注句子，以及跨三个生成器模型独立收集的90轮多轮会话……

    arXiv:2611.22511v1 Announce Type: new  Abstract: Interpretability for AI agents should not be confined to understanding the model's internal weights. It must also cover the agent's reasoning and decision-making process over multi-source context. We present LMLineage, an out-of-band system that attributes every output sentence of an agent session to its supporting input spans with calibrated confidences, without requiring white-box access to the model or any additional inference passes. On top of the lineage metadata, we build concept-drift micro compaction: instead of waiting for the context-length limit and summarizing everything, the system detects when previously used input documents have stopped being referenced ("drifted") and surgically compacts only those, preserving user instructions verbatim. We evaluate on a multi-tier corpus: 71 gold annotated sentences from 10 Chinese web-QA cases, plus 90 turns of independently collected multi-turn sessions across three generator models (g
    
[^98]: 迈向数据高效、实时运行的世界模型

    Towards Data-Efficient, Real-Time World Models

    [https://arxiv.org/abs/2611.22497](https://arxiv.org/abs/2611.22497)

    该论文用两阶段离散扩散方法取代下一词元预测范式，并引入轻量级细化transformer进行迭代掩码预测，在实现快速高质量推理的同时显著提升了世界模型的数据效率，为实时交互式世界模型奠定了基础。

    

    最先进的世界模型已被用于生成符合用户输入动作的游戏过程序列。然而，高质量且一致性的游戏生成往往以牺牲推理速度为代价，这使得实时交互式游玩和创意构思变得困难。此外，训练最先进模型所需的海量数据也使得这些模型的定制化训练显得不切实际，限制了它们在创意领域的许多潜在应用。在这项工作中，我们朝着高效世界模型迈出了重要一步：同时实现了快速的高质量推理和更高的数据效率。这是通过用两阶段离散扩散方法取代下一词元预测范式来实现的，并引入了一个轻量级的细化transformer来执行迭代式的掩码预测。该方法构建于一个能够生成高度一致且复杂的分钟级长序列的自回归世界模型之上。

    arXiv:2611.22497v1 Announce Type: new  Abstract: State-of-the-art world models have been used to produce sequences of gameplay that accord with provided user-input actions. However, high quality, consistent gameplay generation often comes at the cost of inference speed, making real-time interactive play and ideation challenging. The substantial quantities of data used to train state-of-the-art models have also made custom training of these models appear impractical, limiting many of their potential uses in creative fields. In this work we demonstrate a major step towards efficient world models: achieving both fast high-quality inference, and improved data efficiency. This is done by replacing the next-token prediction paradigm with a two-stage discrete diffusion-based approach, introducing a lightweight refinement transformer which carries out iterative masked predictions. Building on an autoregressive world model capable of generating highly consistent and complex sequences over minut
    
[^99]: DARWIN：面向海洋智能的地理空间基础模型

    DARWIN: a Geospatial Foundation Model for Marine Intelligence

    [https://arxiv.org/abs/2611.22496](https://arxiv.org/abs/2611.22496)

    DARWIN是首个针对海洋数据缺失、多模态、稀疏标签和非平稳动态等特性而设计的地理空间基础模型，通过改进的掩码自编码器在26年全球海洋再分析数据上预训练，在叶绿素-a预测任务中仅用10%的有监督数据就超越了用全量数据训练的随机初始化模型。

    

    我们提出了DARWIN，一个面向海洋智能的地理空间基础模型。近期的地球观测基础模型主要集中于陆地领域，而海洋领域带来了独特的挑战：数据缺失、多模态的物理与生物地球化学变量、稀疏的标签，以及瞬态、多尺度且随时间非平稳的动态特性。DARWIN通过缺失感知编码、扩展的空间与时间掩码以及傅里叶式位置嵌入，将掩码自编码器骨干网络适配到这些特性上，并在26年的全球Copernicus再分析数据以及GEBCO测深数据上进行预训练。在预测叶绿素-a方面，该预训练骨干网络在2022年的每个月都取得了最低的月度相对误差，优于无编码器基线模型以及采用相同架构的随机初始化编码器。仅使用10%的有监督训练数据时，DARWIN就已超越使用100%训练数据训练的随机初始化模型。

    arXiv:2611.22496v1 Announce Type: new  Abstract: We present DARWIN, a geospatial foundation model for marine intelligence. Recent Earth-observation foundation models have concentrated on terrestrial domains. The marine realm poses distinct challenges: missingness, multi-modal physical and biogeochemical variables, sparse labels, and dynamics that are transient, multi-scale, and non stationary over time. DARWIN adapts a masked-autoencoder backbone to these properties through missingness-aware encoding, expanded spatial and temporal masking, and Fourier-style positional embeddings, and is pretrained on 26 years of global Copernicus reanalysis together with GEBCO bathymetry.   On predicting chlorophyll-a, the pretrained backbone attains the lowest monthly relative error in every month of 2022, outperforming both no-encoder baselines and an identically architected random-init encoder. At 10% of the supervised training data, DARWIN already surpasses the random-init model trained on 100% of 
    
[^100]: 面向指令引导视频编辑的运动感知奖励建模

    Motion-Aware Reward Modeling for Instruction-Guided Video Editing

    [https://arxiv.org/abs/2611.22495](https://arxiv.org/abs/2611.22495)

    该论文揭示了指令引导视频编辑奖励模型中“运动坍缩”的失效现象——即奖励模型偏向静态一致而牺牲运动，并提出一种结合RGB帧差输入与针对运动抑制负样本进行偏好训练的运动感知奖励模型，从而更好地评估和保持视频编辑中的运动质量。

    

    指令引导视频编辑的奖励模型可能会偏向时间一致性而牺牲运动效果。我们利用真实编辑的受控运动抑制变体研究了这一被称为“运动坍缩”的失效现象。在VE-Bench上，所评估的复合奖励在97%的情况下更偏好运动被抑制的候选结果。学习型视频奖励模型和VLM评判器在面对轻微运动抑制时同样表现不佳，而潜空间优化和奖励引导的微调则会利用这一弱点，在运动减少的同时提升奖励分数。对我们基础奖励骨干网络的线性探测在轻微抑制上表现接近随机水平，这促使我们引入显式的帧间动态信息输入。我们提出了一种运动感知奖励模型，它将RGB帧差流与针对运动抑制和伪造运动负样本的偏好训练相结合。蒸馏锚点保持了与人类质量判断的一致性。在VE-Bench运动阶梯上，我们的模型降低了（原文在此处截断）……

    arXiv:2611.22495v1 Announce Type: cross  Abstract: Reward models for instruction-guided video editing can favor temporal consistency at the expense of motion. We study this failure, termed \emph{motion collapse}, using controlled motion-suppression variants of real edits. On VE-Bench, the evaluated composite reward prefers a motion-suppressed candidate in $97\%$ of cases. Learned video rewards and a VLM judge also struggle with mild suppression, while latent optimization and reward-guided fine-tuning exploit this weakness to increase reward as motion declines. A linear probe of our base reward backbone performs near chance on mild suppression, motivating an explicit inter-frame dynamics input. We propose a motion-aware reward model that combines an RGB frame-difference stream with preference training against motion-suppressed and fabricated-motion negatives. A distillation anchor preserves alignment with human quality judgments. On the VE-Bench motion ladder, our model reduces the rate
    
[^101]: SciWorkBench：面向真实科学工作流的基准测试

    SciWorkBench: A Benchmark for Real Scientific Workflows

    [https://arxiv.org/abs/2611.22490](https://arxiv.org/abs/2611.22490)

    本文提出SciWorkBench——一个基于同行评审文献和真实科研场景构建的基准，用于评估AI智能体在生物、化学、材料科学和物理等真实科学任务上的能力，结果表明即使是表现最好的智能体平均得分也仅为57.38分，说明现有智能体完成真实科学工作仍有显著差距。

    

    语言模型正越来越多地被用作能够规划和执行项目、并通过反馈改进自身行动的智能体。这一转变引出了一个重要问题：当前的智能体能否完成像真实科学家那样的项目？为了研究这个问题，我们提出了SciWorkBench，一个用于评估智能体在真实科学任务上能力的基准，涵盖生物学、化学、材料科学和物理学等领域。SciWorkBench基于同行评审文献或专家贡献的研究场景构建。我们还提出了一个经领域专家验证的统一科学分析框架，可以对SciWorkBench中的大多数任务进行分类，每个任务涵盖该框架的一个或多个阶段。我们在统一的执行和评分协议下，在Claude Code和Codex智能体框架内评估了最新模型。表现最好的智能体平均得分为57.38，这表明当前的智能体在完成真实科学任务方面仍然存在明显困难。

    arXiv:2611.22490v1 Announce Type: new  Abstract: Language models are increasingly being used as agents capable of planning and executing projects, as well as refining their actions through feedback. This shift raises a question: can current agents complete projects like real scientists? To examine this question, we introduce SciWorkBench, a benchmark for evaluating agents' abilities on real science tasks across biology, chemistry, materials science, and physics. SciWorkBench is built from peer-reviewed literature or expert-contributed research scenarios. We also propose a unified scientific analysis framework validated by domain experts, which can categorize most tasks in SciWorkBench. Each task covers one or more stages of this framework. We evaluate up-to-date models within the Claude Code and Codex agent frameworks under a unified execution and grading protocol. The best-performing agent achieves a mean score of 57.38, indicating that current agents still struggle with realistic sci
    
[^102]: 制造差分稀疏性：面向高效世界模型规划的规划器-硬件协同设计

    Manufacturing Differential Sparsity: Planner-Hardware Co-Design for Efficient World Model Planning

    [https://arxiv.org/abs/2611.22483](https://arxiv.org/abs/2611.22483)

    该论文发现量化世界模型预测器中相近动作会产生差分位级稀疏性，并提出规划器-硬件协同设计，通过参考森林采样主动“制造”稀疏性，从而大幅加速具身智能体的MPC规划。

    

    arXiv:2611.22483v1 公告类型：cross 摘要：JEPA风格的潜在世界模型使具身智能体能够通过模型预测控制（MPC）向新目标进行规划，但对于现实生活中的具身控制而言它们仍然太慢：基于采样的MPC在每次规划中需要重复执行数万次预测器推理，即使在高端GPU上，单个规划步骤也可能耗时数十秒。我们观察到这种重复推理工作负载中存在显著的潜在局部性：预测器将相近的动作映射到相近的潜在表示，而在量化预测器内部，这种相近性呈现出一种硬件可利用的形式——差分位级稀疏性。与输入流（其时间相似性是给定的）不同，规划器自身决定将哪些候选一起评估，因此规划器可以选择自己的参考，并通过参考森林采样制造更多稀疏性：它将每组采样动作序列耦合到共享根节点上，同时保持有竞争力的规划质量。我们基于这些洞察构建了加速器……

    arXiv:2611.22483v1 Announce Type: cross  Abstract: JEPA-style latent world models let embodied agents plan toward new goals with model predictive control, but they are still too slow for real-life embodied control: sampling-based MPC repeats predictor inference tens of thousands of times per plan, and even on high-end GPUs one planning step can take tens of seconds. We observe substantial latent locality in this repeated-inference workload: the predictor maps close actions to close latents, and inside a quantized predictor this closeness takes a form that hardware can exploit, differential bit-level sparsity. Unlike an input stream, whose temporal similarity is given, a planner itself decides which candidates it evaluates together, so the planner can choose its own references and, with reference forest sampling, manufacture more sparsity: it couples the sampled action sequences of each group to a shared root, with competitive planning quality. We build the accelerator on the insights o
    
[^103]: LeViT-UNet++：面向高效医学图像分割的细粒度细节与上下文平衡

    LeViT-UNet++: Towards Balanced Fine-Grained Details and Context for Efficient Medical Image Segmentation

    [https://arxiv.org/abs/2611.22482](https://arxiv.org/abs/2611.22482)

    LeViT-UNet++通过引入Haar小波多尺度下采样模块的高效编码器-解码器架构，在有限计算资源下实现了医学图像分割中细粒度细节保留与全局上下文建模的良好平衡。

    

    高效的医学图像分割对于需要快速且精确勾画的临床应用至关重要，例如手术导航、放射治疗和实时图像引导介入。尽管卷积神经网络和Transformer已取得显著进展，但在有限计算资源下实现高精度仍然困难，因为降低模型复杂度往往会牺牲细粒度结构细节和全局上下文建模，从而导致不理想的精度-效率权衡。为解决这一问题，我们提出了LeViT-UNet++，这是一种高效的编码器-解码器架构，能够更好地平衡局部细节保留与全局上下文建模。具体而言，我们引入了Haar小波多尺度下采样（HWMD）模块，将特征表示分解为低频和高频分量以提取多尺度结构信息，从而缓解下采样过程中的细节损失

    arXiv:2611.22482v1 Announce Type: cross  Abstract: Efficient medical image segmentation is essential for clinical applications requiring fast and accurate delineation, such as surgical navigation, radiation therapy, and real-time image-guided intervention. Despite substantial advances in convolutional neural networks and Transformers, attaining high accuracy under limited computational resources remains difficult, because reducing model complexity often sacrifices fine-grained structural details and global context modeling, yielding an unfavorable accuracy-efficiency trade-off. To address this, we propose LeViT-UNet++, an efficient encoder-decoder architecture that better balances local detail preservation and global context modeling. Specifically, we introduce a Haar Wavelet Multi-Scale Downsampling (HWMD) module that decomposes feature representations into low- and high-frequency components to extract multi-scale structural information, thereby mitigating detail loss during downsampl
    
[^104]: RehearsalBench：AI能预测改变一家公司后会发生什么吗？

    RehearsalBench: Can AI Predict What Happens When You Change a Company?

    [https://arxiv.org/abs/2611.22480](https://arxiv.org/abs/2611.22480)

    提出了RehearsalBench基准，在基于真实通话记录的模拟银行联络中心中评估AI模型诊断运营、选择干预措施并预测组织变革后果的能力，最优模型GPT-6 Astra相比不作为可降低6.8%的成本，且预测误差减少77%。

    

    组织越来越频繁地向AI系统询问：如果改变运营方式会发生什么——哪些工作应该自动化、哪个团队需要调整规模、语音客服会对等待时间产生什么影响。我们提出了RehearsalBench，这是一个衡量AI模型能否诊断运营状况、选择干预措施并预测其后果的基准测试。每个模型被置于一个基于真实通话级记录构建的模拟银行联络中心中，只能看到管理者所能看到的信息，并提交计划和预测，这些计划和预测随后在200个种子化未来情景和三个合理版本的组织中推演。决策按照利益相关者声明的估值以美元计分；预测则根据每项变革相对于维持现状（不采取行动）的效果进行评分。十个模型在14个组织中共进行了280次尝试。表现最好的模型GPT-6 Astra在不采取任何行动的基础上将成本改善了相当于无变化成本6.8%的幅度，其预测误差比一个总是预测“不进行任何变革”的预测者低77%（原摘要在此处截断）。

    arXiv:2611.22480v1 Announce Type: new  Abstract: Organizations increasingly ask AI systems what will happen if they change how they operate: which work to automate, which team to resize, what a voice agent will do to waiting times. We introduce RehearsalBench, a benchmark that measures whether AI models can diagnose an operation, choose an intervention, and forecast its consequences. Each model is placed in a simulated bank contact center fitted to real call-level records, sees only what a manager would see, and submits a plan and forecasts that are then played out across 200 seeded futures and three plausible versions of the organization. Decisions are scored in dollars at the stakeholders' stated valuations; forecasts are scored on the effect of each change relative to doing nothing. Ten models made 280 attempts across 14 organizations. The best model, GPT-6 Astra, improved on doing nothing by 6.8% of the no-change cost and made 77% less forecast error than a forecaster who always pr
    
[^105]: MAVE：从电影镜头重建可编辑的3D布局模型

    MAVE: Reconstructing Editable 3D Blockouts from Film Shots

    [https://arxiv.org/abs/2611.22464](https://arxiv.org/abs/2611.22464)

    MAVE是一个融合度量观测、代理建模与证据验证的视频到3D布局重建框架，能从电影镜头自动生成可编辑的Blender场景，并配套推出MAVEBench基准用于评估相机运动、空间布局和角色调度。

    

    动画3D布局模型（blockout）能够支持预可视化，并为视频生成提供结构性条件，但手动从参考视频中重建这些模型成本高昂。这既限制了创作者对其的使用，也限制了配对训练数据的制作。我们提出了MAVEBench，一个包含50个电影镜头和416个关键帧的基准，用于评估相机运动、空间布局和角色调度。我们提出MAVE——一个视频到布局模型的框架，它结合了度量观测、代理建模和基于证据的验证三个环节。该框架首先在统一的度量坐标系中恢复相机、人体运动和几何证据；随后，代理在这些观测结果的约束下构建可编辑的环境，同时通过程序化检查报告不一致之处以引导迭代改进。其输出是可编辑的Blender场景以及与帧对齐的布局视频。在MAVEBench上，MAVE在三个代理设置下相比基线方法将总分提升了27.5至41.1分。

    arXiv:2611.22464v1 Announce Type: cross  Abstract: Animated 3D blockouts support previsualization and provide structural conditions for video generation, but manually reconstructing them from reference videos is costly. This limits both their use by creators and the production of paired training data. We introduce MAVEBench, a benchmark of 50 film shots and 416 keyframes for evaluating camera motion, spatial layout, and character blocking. We propose MAVE, a video-to-blockout framework combining Metric observation, Agentic modeling, and evidence-based VErification. It first recovers cameras, human motion, and geometric evidence in a shared metric coordinate system. An agent then constructs an editable environment under constraints from these observations, while programmatic checks report inconsistencies to guide refinement. The outputs are editable Blender scenes and frame-aligned blockout videos. On MAVEBench, MAVE improves overall scores by 27.5-41.1 points across three agents over t
    
[^106]: 从理想到现实：将语言模型集成到智能家居中以实现以人为中心的物联网安全模糊测试

    From Ideal to Real: Integrating Language Models into Smart Home for Human-Centric IoT Safety and Security Fuzzing

    [https://arxiv.org/abs/2611.22457](https://arxiv.org/abs/2611.22457)

    该论文提出将大语言模型集成到智能家居物联网模糊测试框架中，通过模拟真实人类决策过程，突破了传统理想化模拟环境的局限，从而更有效地发现可能导致隐私泄露和人身伤害的规则间安全漏洞。

    

    将物联网设备集成到家庭系统中可以增强对生活环境的控制。然而，随着这些设备在医疗保健、交通运输和家居自动化等关键领域日益普及，它们暴露出可能导致隐私泄露、数据窃取和人身伤害的规则间漏洞。传统的模糊测试方法通常在假设人类决策理想的模拟环境中运行，未能考虑到可能影响智能家居设备运行的复杂人类交互因素，从而可能导致不可预见的后果。为了揭示真实人类决策的影响并发现意想不到的规则间漏洞，我们通过集成大语言模型来模拟人类决策过程，提出了面向智能家居的物联网模糊测试增强版本。我们的新方法为测试环境增加了一层真实性和不可预测性，能够更准确地捕捉（摘要在此处截断）

    arXiv:2611.22457v1 Announce Type: new  Abstract: Integrating IoT devices into household systems offers enhanced control over living environments. However, as these devices become pervasive in critical sectors such as healthcare, transportation, and home automation, they expose inter-rule vulnerabilities that may lead to privacy breaches, data theft, and physical harm. Traditional fuzz testing methods, which often operate in simulated environments assuming ideal human decisions, fail to account for the complexities of human interactions that can influence device operations in smart homes, potentially leading to unforeseen consequences. To reveal the impact of real human decisions and discover unexpected inter-rule vulnerabilities, we introduce an enhanced version of IoT Fuzz for smart homes by integrating large language models to emulate human decision-making processes. Our novel approach adds a layer of realism and unpredictability to the testing environment, more accurately capturing 
    
[^107]: PACD：面向全模态大语言模型跨模态幻觉缓解的溯源感知共识解码

    PACD: Provenance-Aware Consensus Decoding for Mitigating Cross-Modal Hallucinations in Omnimodal Large Language Models

    [https://arxiv.org/abs/2611.22456](https://arxiv.org/abs/2611.22456)

    提出了一种免训练的溯源感知共识解码框架PACD，根据模态分支间是否达成共识采用两条互补路径（如相关性引导的候选仲裁），有效缓解全模态大语言模型中的跨模态幻觉问题。

    

    全模态大语言模型整合音频、视频和文本进行多模态推理，但仍然容易受到跨模态幻觉的影响，即一种模态不当地影响对另一种模态的预测。现有的免训练解码方法通过模态扰动或基于相关性的加权来缓解这些错误。然而，由于融合过程中无关模态线索的干扰以及难以检测共享错误，这些方法可能会抑制正确的单模态预测，并使错误的共识持续存在。为了解决这些局限性，我们提出了溯源感知共识解码（PACD），这是一个用于缓解跨模态幻觉的免训练框架。PACD根据各分支是否达成共识，包含两条互补的路径。对于非共识情况，相关性引导的候选仲裁会识别与任务相关的单模态证据，并在各预测之间选择性地进行仲裁

    arXiv:2611.22456v1 Announce Type: new  Abstract: Omnimodal large language models (Omni-LLMs) integrate audio, video, and text for multimodal reasoning, yet remain vulnerable to cross-modal hallucinations, where one modality improperly influences predictions about another. Existing training-free decoding methods mitigate these errors through modality perturbation or relevance-based weighting. However, due to interference from irrelevant modality cues during fusion and difficulty detecting shared errors, these methods may suppress correct unimodal predictions and allow erroneous consensus to persist. To address these limitations, we propose Provenance-Aware Consensus Decoding (PACD), a training-free framework for mitigating cross-modal hallucinations. PACD comprises two complementary paths based on whether the branches reach consensus. For non-consensus cases, Relevance-Guided Candidate Arbitration identifies task-relevant unimodal evidence and selectively arbitrates between predictions 
    
[^108]: 随着学生模型持续展开生成，视觉后门不断增长

    As Student Rollouts Continue, Visual Backdoors Keep Growing

    [https://arxiv.org/abs/2611.22446](https://arxiv.org/abs/2611.22446)

    本文挑战了“在线策略蒸馏（OPD）对后门传播具有天然抵抗力”的直觉，通过多模型家族与数据集上的实验及理论分析，揭示了视觉后门会随着学生模型在线展开生成而持续传播增长的隐蔽风险。

    

    传统知识蒸馏可能促使后门从教师模型传播到学生模型。然而，这种风险在在线策略蒸馏中如何表现，目前研究尚不充分。OPD 近来已成为视觉-语言模型（VLM）的一种重要后训练范式。与传统离线蒸馏不同，OPD 不需要教师生成的响应，而是利用教师的概率分布来监督学生模型在线生成的响应轨迹。由于初始学生模型通常生成良性响应，且不会从恶意目标响应中获得显式监督，OPD 似乎对后门传播具有某种固有的抵抗力。本文挑战了这一直觉，揭示了 OPD 中视觉后门传播的一种更隐蔽的风险。通过在多个模型家族和数据集上的实验，并结合理论分析……

    arXiv:2611.22446v1 Announce Type: cross  Abstract: Conventional knowledge distillation can facilitate the propagation of backdoors from teacher models to student models. However, how this risk manifests in On-Policy Distillation (OPD) remains insufficiently studied. OPD has recently emerged as an important post-training paradigm for Vision-Language Models (VLMs). Unlike traditional offline distillation, OPD does not require teacher-generated responses. Instead, it uses the teacher's probability distributions to supervise response trajectories generated online by the student. Since the initial student model typically generates benign responses and receives no explicit supervision from malicious target responses, OPD appears to offer some inherent resistance to backdoor propagation.   In this paper, we challenge this intuition and reveal a more subtle risk of visual backdoor propagation in OPD. Through experiments across multiple model families and datasets, together with theoretical ana
    
[^109]: 更新中的“眼睛”？学习奖励在多模态推理中的落点

    Eyes in the Update? Learning Where Rewards Land in Multimodal Reasoning

    [https://arxiv.org/abs/2611.22438](https://arxiv.org/abs/2611.22438)

    Gaze-RL 利用模型内部跨模态几何与局部预测不确定性构建潜在“奖励落点场”，无需外部标注器或过程标签，即可将轨迹级优势奖励重新分配到响应中视觉证据实际所在的位置，从而提升多模态强化学习的 token 敏感性。

    

    多模态后训练已从宽泛的指令微调转向面向图像与视频推理的强化学习，其中结果奖励能够提升最终答案的准确性。然而，这些奖励被分配给完整生成的轨迹，而视觉证据仅在响应内部稀疏且不均匀的位置被使用。近期工作已开始通过接地信号或外部 token 级加权使多模态强化学习对 token 更加敏感。我们提出一个更具体的问题：能否从模型自身的多模态计算中推断同一轨迹级奖励的分配方式，而不依赖辅助标注器、感知模块或过程标签？我们提出 Gaze-RL，一种由表示驱动的策略优化方法，通过一个潜在的“奖励落点场”重新分配完成级优势。该落点场由内部跨模态几何结构与局部预测不确定性计算得到，为（视觉证据所在的 token）分配更大的……（摘要在此处截断）

    arXiv:2611.22438v1 Announce Type: cross  Abstract: Multimodal post-training has moved from broad instruction tuning toward reinforcement learning for image and video reasoning, where outcome rewards can improve final-answer accuracy. Yet these rewards are assigned to complete generated trajectories, while visual evidence is used only at sparse and uneven positions inside the response. Recent work has begun to make multimodal RL more token-sensitive through grounding signals or external token-level weighting. We ask a more specific question: can the allocation of the same trajectory-level reward be inferred from the model's own multimodal computation, rather than from an auxiliary annotator, perception module, or process label? We propose Gaze-RL, a representation-driven policy optimization method that redistributes completion-level advantage through a latent reward landing field. The field is computed from internal cross-modal geometry and local predictive uncertainty, assigning larger
    
[^110]: 编码、可解码性与使用：一个用于基准测量审计的五层框架——来自无人机意图识别的证据

    Encoding, Decodability, and Use: A Five-Layer Framework for Benchmark Measurement Audits with Evidence from UAV Intent Recognition

    [https://arxiv.org/abs/2611.22434](https://arxiv.org/abs/2611.22434)

    本文提出一个将基准测量审计分解为观测、表示、解码、使用与行为五个层级的框架，并通过无人机意图识别基准 AeroIntent 的实证表明，基准高分并不等于模型真正理解了所测量的构念。

    

    基准准确率通常被解读为模型理解了基准所测量构念的证据。我们在 AeroIntent 上提出并实例化了一个五层框架，将这种解读分解为观测、表示、解码、使用和行为。该框架将编码、可解码性和使用区分为不同的层级。在实例化中，观测层在 20 个 Drone 程序中的 7 个上不可靠，并将径向速度的直接视觉相关性下限锁定在 0.26% 的尺度变化上。表示层揭示了两个仅具名义信号的任务设计 RGB 特征：RAFT 光流径向偏差达到 r=+0.257（p=0.030），边界框光流散度达到 r=+0.203（p=0.029）。没有任何多特征读出在折平均测试相关性上超过聚类的最小可检测效应（MDE），且最大的正的全局样本外相关性仅为 r=+0.080。解码层将 MDE 设定在……（原文摘要在此处被截断）

    arXiv:2611.22434v1 Announce Type: cross  Abstract: Benchmark accuracy is routinely read as evidence that a model understands the construct the benchmark measures. We propose and instantiate on AeroIntent a five-layer framework that decomposes that reading into observation, representation, decoding, use, and behaviour. The framework separates encoding, decodability, and use as distinct levels. In the instantiation, the observation layer is unreliable on 7 of the 20 Drone programs and floors the direct visual correlate of radial velocity at a 0.26\% scale change. The representation layer exposes two task-designed RGB features with nominal signals. RAFT flow radial bias reaches $r=+0.257$ ($p=0.030$), and bounding-box optical-flow divergence reaches $r=+0.203$ ($p=0.029$). No multi-feature readout gives a fold-mean test correlation above the cluster minimum detectable effect (MDE), and the largest positive global out-of-sample correlation is $r=+0.080$. The decoding layer sets the MDE at 
    
[^111]: 思考一次，快速行动：面向决策模型框架的高性价比递归自我改进

    Think Once, Act Fast: Cost-Effective Recursive Self-Improvement for Decision Model Harness

    [https://arxiv.org/abs/2611.22432](https://arxiv.org/abs/2611.22432)

    提出JevRSI系统1/系统2协作框架，通过递归式模块化框架改进将LLM推理转化为可迁移、高性价比的决策支持，兼顾低延迟动作选择与持续自我改进能力。

    

    快速的系统1模型能够实现即时响应的行动，但在处理复杂或不熟悉的任务时表现不佳。基于大语言模型（LLM）的系统2提供了强大的推理能力，但反复调用会带来显著的延迟和计算成本。为了弥合这一差距，我们提出了JevRSI，一个系统1/系统2协作框架，通过递归式的框架改进将LLM推理转化为可迁移的决策支持。Jev风格的系统1模型实现低延迟的动作选择，而异步的系统2 LLM则负责开发一个模块化框架。该框架集成了流程、状态记忆和世界模型，以提供跨越过去、现在和未来的决策上下文。效用驱动的检索机制基于上下文相关性以及模块在先前任务上的效用进行模块选择。环境反馈指导有针对性的修订，从而实现递归自我改进。在三个数据集上的实验表明，Jev

    arXiv:2611.22432v1 Announce Type: new  Abstract: Fast System-1 models enable responsive actions but struggle with complex or unfamiliar tasks. Large language model (LLM)-based System-2 provides strong reasoning capabilities, yet repeated invocation incurs substantial latency and computational costs. To bridge this gap, we propose JevRSI, a System-1/System-2 collaboration framework that transforms LLM reasoning into transferable decision support through recursive harness improvement. A Jev-style System-1 model enables low-latency action selection, while an asynchronous System-2 LLM develops a modular harness. The harness integrates procedures, state memory, and world models to provide decision contexts spanning the past, present, and future. Utility-driven retrieval selects modules based on both contextual relevance and their utility on previous tasks. Environmental feedback guides targeted revisions and enables recursive self-improvement. Experiments across three datasets show that Jev
    
[^112]: 2026年XCSP3竞赛论文集

    Proceedings of the 2026 XCSP3 Competition

    [https://arxiv.org/abs/2611.22420](https://arxiv.org/abs/2611.22420)

    本文收录了2026年XCSP3约束求解器竞赛的论文集，竞赛结果于2026年7月在葡萄牙里斯本举行的FLOC会议上公布。

    

    本文档构成2026年XCSP3竞赛的论文集。这场约束求解器竞赛的结果在2026年FLOC（联邦逻辑会议）奥林匹克运动会上公布，该会议于2026年7月20日至29日在葡萄牙里斯本举行。

    arXiv:2611.22420v1 Announce Type: new  Abstract: This document constitutes the proceedings of the 2026 XCSP3 Competition. The results of this competition of constraint solvers were presented at FLOC (Federated Logic Conference) 2026 Olympic Games, held in Lisbon, Portugal from 20th to 29th July, 2026.
    
[^113]: ROSMonitoring 实战：ROS 2 动作的运行时监控与强制执行

    ROSMonitoring in Action: Runtime Monitoring and Enforcement for ROS 2 Actions

    [https://arxiv.org/abs/2611.22407](https://arxiv.org/abs/2611.22407)

    本文扩展了 ROSMonitoring 框架，首次实现了对 ROS 2 动作这一通信原语的运行时监控与强制执行，通过重构动作生命周期、基于目标标识符关联事件，并将违规行为转化为协议感知的执行操作。

    

    机器人应用通常依赖于长时间运行的操作，如导航、操纵和巡检，这些操作在 ROS 2 中通常以动作的形式实现。与话题和服务不同，动作包含一个结构化的生命周期，涵盖目标提交与接受、反馈、状态更新、结果获取和取消等环节。现有版本的 ROSMonitoring 支持话题和服务，但无法重构这一生命周期，也无法将动作级别的事件暴露给运行时验证的判定器。本文对 ROSMonitoring 进行了扩展，为其增加了针对 ROS 2 动作的运行时监控与强制执行能力。动作可直接在监控配置中声明，并通过底层 ROS 2 通信端点自动进行插桩。生成的监视器能够重构动作级别的事件，通过目标标识符将事件关联起来，维护活跃目标的状态，并将违规行为转化为协议感知的强制执行操作。

    arXiv:2611.22407v1 Announce Type: new  Abstract: Robotic applications often rely on long-running operations, such as navigation, manipulation, and inspection, which in ROS~2 are commonly implemented as actions. Unlike topics and services, actions involve a structured lifecycle comprising goal submission and acceptance, feedback, status updates, result retrieval, and cancellation. Existing versions of ROSMonitoring support topics and services, but do not reconstruct this lifecycle or expose action-level events to a runtime-verification oracle. In this paper, we extend ROSMonitoring with runtime monitoring and enforcement for ROS~2 actions. Actions are declared directly in the monitoring configuration and automatically instrumented through the underlying ROS~2 communication endpoints. The generated monitor reconstructs action-level events, correlates them through goal identifiers, maintains the state of active goals, and translates violations into protocol-aware enforcement operations. T
    
[^114]: 符号抽象精化

    Symbolic Abstraction Refinement

    [https://arxiv.org/abs/2611.22404](https://arxiv.org/abs/2611.22404)

    本文提出SYMAR方法，通过将值函数符号化而非对区域进行分割来验证概率时间自动机，避免了传统方法中级联式的图重建过程，提高了验证效率。

    

    自主系统所选择的动作其结果具有不确定性，并受时间约束的限制。概率时间自动机正是对这种情况进行建模的，对其验证可以回答关于系统决策的问题：即控制器能够保证的达到目标的最大概率，或达到危险的最小概率。验证通常在符号状态上进行，每个符号状态将一个离散位置与时钟值的区域配对，其难点在于最优行为通常在这样的区域内部是变化的。PRISM验证器最强大的PTA方法通过不断分割区域直到每个区域都是均匀的来应对这一难题，重建到达这些区域的迁移，并将分割向后传播通过随机博弈的状态图，这种级联式的图重建过程会对验证属性从不依赖的区域进行划分。我们提出了SYMAR，它转而使值函数符号化：每个符号状态（摘要在此处被截断）

    arXiv:2611.22404v1 Announce Type: new  Abstract: Autonomous systems choose actions whose outcomes are uncertain and are subject to time constraints. Probabilistic timed automata (PTAs) model exactly this, and verifying one answers a question about the system's decision-making: the maximum probability of reaching a goal, or the minimum probability of reaching a hazard, that a controller can guarantee. Verification normally works over symbolic states, each pairing a discrete location with a region of clock values, and its difficulty is that the best behavior generally varies within such a region. The PRISM verifier's most capable PTA method responds by splitting regions until each is uniform, rebuilding the transitions that reach them and propagating the split backward through the state graph of a stochastic game, a cascade of graph reconstruction that partitions regions the property never depends on. We present SYMAR, which instead makes the value function symbolic: each symbolic state 
    
[^115]: Deal With It：论列表式重排序的更优调度方案

    Deal With It: On Better Schedules for Listwise Reranking

    [https://arxiv.org/abs/2611.22392](https://arxiv.org/abs/2611.22392)

    本文证明了在并行列表式重排序调度中，按检索器排名轮转“发牌”划分候选是均期望削减最少的最优分组方式，并提出简单高效的 DealRank 调度方案，在保持接近最优重排序质量的同时实现 1.1 至 1.4 倍的提速。

    

    列表式 LLM 重排序器无法在单次调用中可靠地对数百篇文档进行排序，因此需要依靠调度方案来协调多次调用。并行调度将候选文档划分成若干组，并从每组中按固定配额向下一轮推进。当某个组包含的高质量段落超过其配额时，部分段落会被削减，从而需要进行代价高昂的修复调用。我们证明了候选文档的划分方式决定了配额在何处可能成为约束，并且就期望削减数量而言，最优的划分方式是像发牌一样按检索器排名以轮转方式分发候选。尽管这种“发牌”策略在概念上并不新颖，但此前一直缺乏理论分析。我们证明，在一个简单的检索器模型下，没有任何配额相同的划分方式在平均意义上比发牌削减更少的真实前十条结果，而连续分块的削减量会随检索器性能的提升而增加。DealRank 是我们分析所允许的最简单的调度方案，在两轮设置中对六个 LLM 的平均表现与最优调度方案相差不超过 0.1 个 nDCG@10 点，且速度始终快 1.1 至 1.4 倍。

    arXiv:2611.22392v1 Announce Type: cross  Abstract: Listwise LLM rerankers cannot reliably rank hundreds of documents in a single call, and rely on schedules to orchestrate their calls. Parallel schedules split candidates into groups and advance a fixed quota from each. When a group holds more of the top passages than its quota, some are cut, and expensive repair calls become required. We show that the way candidates are split matters where such a quota can bind, and that, in expected cuts, the best split there deals them round-robin by retriever rank, like cards. Although not conceptually new, dealing has remained unanalyzed. We prove that, under a simple retriever model, no split with equal quotas cuts fewer of the true top ten on average, and that contiguous chunks cut more as the retriever improves. DealRank, the simplest schedule our analysis allows, is on average within 0.1 nDCG@10 points of the best schedules over six LLMs, in two rounds. It is consistently 1.1 to 1.4 times faste
    
[^116]: RICoT：用于页面流分割的循环交错思维链

    RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation

    [https://arxiv.org/abs/2611.22382](https://arxiv.org/abs/2611.22382)

    该论文提出RICoT循环交错思维链提示框架，首次将页面流分割重新构想为对整个流的顺序处理过程，像读者一样在滑动窗口内逐页推理并携带已读内容的紧凑记录以控制成本，同时发布了包含205个真实金融页面流的FinancePSS-205基准数据集。

    

    大语言模型（LLM）重新激发了人们对页面流分割（Page Stream Segmentation, PSS）的兴趣，该任务是将多页文件拆分为其组成的文档。然而，已发表的LLM方法仍然将该任务框架化为对一页或少数相邻页面的边界分类，而商业文档拆分API也不公开其方法。我们重新构想了面向LLM的PSS任务，将其视为对整个页面流的顺序处理过程，并提出了RICoT（循环交错思维链，Recurrent Interleaved Chain-of-Thought），这是一个像读者一样逐页处理文档的提示框架：在滑动窗口内，它对每一页进行推理并先做出边界决策再继续往下读；在窗口之间，它携带一个已读内容的紧凑记录，从而使每次调用的成本保持有界。我们还介绍了FinancePSS-205，一个包含205个真实世界、去标识化金融页面流的基准数据集，并将公开发布。在FinancePSS-205上，RICoT匹配或超越了成对（pairwise）和窗口化（windowed）提示方法的表现（摘要在此处被截断）。

    arXiv:2611.22382v1 Announce Type: cross  Abstract: Large language models (LLMs) have renewed interest in Page Stream Segmentation (PSS), splitting a multi-page file into its constituent documents. Yet published LLM methods still frame the task as boundary classification over one or a few neighboring pages, and commercial document-splitting APIs do not disclose their methods. We reimagine PSS for LLMs as a sequential process over the whole stream and propose RICoT (Recurrent Interleaved Chain-of-Thought), a prompting framework that works through the pages as a reader would: within a sliding window it reasons about each page and commits its boundary decision before moving on, and across windows it carries forward a compact record of what it has read, keeping per-call cost bounded. We also introduce FinancePSS-205, a benchmark of 205 real-world, de-identified financial page streams, to be released publicly. On FinancePSS-205, RICoT matches or outperforms pairwise and windowed prompting wi
    
[^117]: 利用分割表示实现高效分类：一种基于nnU-Net的自配置框架

    Leveraging Segmentation Representations for Efficient Classification: A Self-Configuring Framework Based on nnU-Net

    [https://arxiv.org/abs/2611.22380](https://arxiv.org/abs/2611.22380)

    该论文提出一种基于nnU-Net的自配置框架，通过复用分割模型学习到的编码器特征和分割概率，结合轻量级分类器与两种信息共享策略，高效实现3D医学影像的图像级分类。

    

    我们提出了一个自配置框架，将nnU-Net工作流程扩展到3D医学影像中的分割和分类两种任务。基于nnU-Net的自配置方法，该框架复用分割模型学习到的表示来实现图像级分类。其编码器特征和预测的分割概率被一个轻量级分类器所利用。论文研究了两种信息共享策略：在冻结主干分支中，特征通过与感兴趣区域信息、预测前景体积和分割置信度相结合来增强；在适应分支中，最后的编码器阶段针对分类任务进行部分微调，并通过辅助分割损失来保留与分割相关的信息。额外的五折实验表明，更复杂的配置并不能持续提升分类性能。在公开的验证排行榜上，我们选定的框架……

    arXiv:2611.22380v1 Announce Type: cross  Abstract: We propose a self-configuring framework that extends the nnU-Net workflow to both segmentation and classification in 3D medical imaging. Building on nnU-Net's self-configuring method, the framework reuses representations learned by a segmentation model for image-level classification. Its encoder features and predicted segmentation probabilities are used by a lightweight classifier. Two information-sharing strategies are investigated. In the frozen-backbone branch, features are augmented with region-of-interest information, predicted foreground volume, and segmentation confidence. In the adaptation branch, the final encoder stage is partially fine-tuned for classification, with an auxiliary segmentation loss to preserve segmentation-related information. Additional five-fold experiments showed that more complex configurations did not consistently improve classification performance. On the public validation leaderboard, our selected frame
    
[^118]: AutoE2E：无需言语——无语言的端到端自动驾驶：面向L2++级高级驾驶辅助系统（ADAS）与L4级Robotaxi的实时驾驶模型

    AutoE2E: Don't speak - Language free end-to-end autonomy A Real-Time Driving Model for L2++ ADAS and L4 Robotaxis

    [https://arxiv.org/abs/2611.22374](https://arxiv.org/abs/2611.22374)

    AutoE2E是一个开源的、不依赖语言的端到端自动驾驶模型，采用时序BEV架构融合多摄像头感知与高精地图信息，以仅约145万可训练参数的轻量级设计实现了面向L2++ ADAS和L4 Robotaxi的实时高精度轨迹预测。

    

    AutoE2E是由Autoware基金会Robotaxi工作组开发的开源端到端驾驶模型。它引入了一种时序鸟瞰图（BEV）架构，结合多摄像头感知、高精地图信息和路线条件来预测车辆未来的运动。所评估的模型使用了预训练的摄像头-BEV骨干网络、时序特征融合以及轻量级的循环轨迹规划器。该模型包含约7990万参数，其中在所评估的配置下约145万参数为可训练参数。v1.0版本发布内容包括模型检查点、训练与评估代码，以及DataModelConsole——一个用于可视化和检查模型实验的仪表板。在KITScenes验证基准上，微调后的检查点在三秒预测时间范围内取得了0.745米的平均位移误差（ADE）和2.020米的最终位移误差（FDE）。论文还报告了初步……

    arXiv:2611.22374v1 Announce Type: new  Abstract: AutoE2E is an open-source end-to-end driving model developed by the Autoware Foundation Robotaxi Working Group. It introduces a temporal bird's-eye-view (BEV) architecture that combines multi-camera perception, HD-map information and route conditioning to predict future vehicle motion. The evaluated model uses a pretrained camera-BEV backbone, temporal feature fusion and a lightweight recurrent trajectory planner. It contains approximately 79.9 million parameters, of which approximately 1.45 million are trainable in the evaluated configuration. The v1.0 release provides model checkpoints, training and evaluation code, and DataModelConsole, a dashboard for visualizing and inspecting model experiments. On the KITScenes validation benchmark, the fine-tuned checkpoint achieves an average displacement error (ADE) of 0.745 m and a final displacement error (FDE) of 2.020 m at a three-second prediction horizon. The paper also reports preliminary
    
[^119]: 看清依然持守之物：等待控制如何改变协作式恢复

    Seeing What Still Holds: How Waiting Control Changes Cooperative Recovery

    [https://arxiv.org/abs/2611.22368](https://arxiv.org/abs/2611.22368)

    该论文提出 AuditEmbodied 成对执行与轨迹评估框架，通过 960 次执行的因子实验证明：在固定时长等待中改变持握参考可逆转 CAD 反馈式与地标式视觉恢复方法的排名，且关键差异源于抓取形成阶段观察残差到跟踪误差的放大。

    

    在固定时长的等待期间改变持握参考，可以逆转视觉恢复方法的性能排名。我们提出 AuditEmbodied，一个成对的执行与轨迹评估框架，它在保持恢复执行器冻结的前提下，独立地变化持握参考与等待时长。一项在 20 个初始化种子上开展的 960 次执行因子实验，在每种持握与等待条件下，从完全相同的等待后物理状态出发，比较了基于 CAD 反馈的恢复与基于地标的恢复。受保护的完成要求在声明的执行约束下成功实现恢复。在 1 秒等待且使用配准腕部参考时，两种方法的受保护完成次数从配准持握下的 6/40 对 19/40，变为仅位置持握下的 34/40 对 22/40。轨迹分析将主要差异定位于抓取形成阶段：观察时刻的微小目标残差会先于大的跟踪误差出现。联合诊断与受控干预……（摘要原文在此处截断）

    arXiv:2611.22368v1 Announce Type: new  Abstract: Changing the holding reference during a fixed-duration wait can reverse the ranking of visual recovery methods. We introduce AuditEmbodied, a paired execution and trajectory evaluation framework that independently varies holding references and waiting duration while keeping recovery executors frozen. A 960-execution factorial on 20 initialization seeds compares CAD-feedback and landmark-based recovery from identical post-wait physical states within each holding and waiting condition. Protected completion requires successful recovery under the declared execution constraints. At a 1 s wait with registered wrist references, their protected completion counts change from 6/40 versus 19/40 under registered holding to 34/40 versus 22/40 under position-only holding. Trajectory analysis localizes the main separation to grasp formation: small observation-time target residuals precede large tracking errors. Joint diagnostics and controlled interven
    
[^120]: FailureGen：基于世界模型的可扩展机器人失败数据生成

    FailureGen: Scalable Robot Failure Data Generation with World Models

    [https://arxiv.org/abs/2611.22367](https://arxiv.org/abs/2611.22367)

    提出FailureGen方法，通过在世界模型中复用真实失败记录的动作轨迹，可规模化地生成带时间事件标注的合成机器人失败视频，解决失败数据难以大规模收集的问题。

    

    机器人失败数据难以大规模收集，这限制了机器人学习数据集中所能涵盖的失败场景范围。我们提出FailureGen，这是一种通过复用已记录失败中的动作轨迹来生成带时间事件标注的合成视频的方法。该方法包含四个阶段：（1）在成功和失败的真实执行数据上对动作条件化的Cosmos世界模型进行微调；（2）对源录制数据进行标注，以识别与失败相关的供体片段以及兼容的成功宿主片段；（3）将供体的动作转移至宿主上下文中（可附带空间偏移），以生成连续的视频rollout；（4）依据观察到的事件和视觉质量对生成的样本进行标注和过滤。我们研究了两种采用不同任务和机器人配置的实现，其中视觉失败检测作为一个下游应用。在初步比较中，基于真实数据训练的检测器（原文在此处截断）

    arXiv:2611.22367v1 Announce Type: cross  Abstract: Robot failures are difficult to collect at scale, limiting the range of failure scenarios represented in robot learning datasets. We present FailureGen, a method for generating synthetic videos with temporal event annotations by reusing action trajectories from recorded failures. The method comprises four stages: (1) fine-tuning an action-conditioned Cosmos world model on successful and failed real executions; (2) annotating source recordings to identify failure-associated donor segments and compatible successful hosts; (3) transferring donor motion into host contexts, with optional spatial offsets, to generate continuous video rollouts; and (4) annotating and filtering the generated samples according to observed events and visual quality. We study two implementations with different tasks and robot configurations, with visual failure detection serving as one downstream application. In a preliminary comparison, a detector trained on rea
    
[^121]: HealthCLBench：通过受控病例集群评估健康对话中的上下文学习

    HealthCLBench:Evaluating Context Learning in Health Conversations through Controlled Case Clusters

    [https://arxiv.org/abs/2611.22362](https://arxiv.org/abs/2611.22362)

    该论文提出了中文健康对话基准HealthCLBench，通过328个受控病例集群和4,931条评分标准，系统评估大模型在参数固定情况下获取、整合并应用上下文以应对事实、证据和任务目标变化的能力。

    

    复杂的健康对话需要答案既立足于个人信息，又基于当前有效的证据。我们评估健康上下文学习（health context learning）：在模型参数固定的情况下，在推理过程中获取、整合和应用相关上下文，以回答健康问题，并对事实、证据或任务目标的变化做出恰当回应。据我们所知，中文医学基准尚未系统地结合明确的上下文难点、受控的病例变体以及针对开放式答案的个体化医学评判标准。我们提出了HealthCLBench，涵盖医疗咨询和日常健康管理两大领域，包含328个受控集群、1,103个病例和4,931条评分标准。集群内各成员共享背景信息，而证据、患者状态或任务目标则发生变化；每个病例都有其特定的医学评分要求。由70人组成的医疗团队对模拟病例及其评分标准进行标注与审核。人工审核了70（原文摘要在此处截断，后续内容缺失）

    arXiv:2611.22362v1 Announce Type: new  Abstract: Complex health conversations require answers grounded in individual information and currently valid evidence. We evaluate health context learning: acquiring, integrating, and applying relevant context during inference with fixed model parameters to answer health questions and respond appropriately to changes in facts, evidence, or task goals. To our knowledge, Chinese-language medical benchmarks have yet to systematically combine explicit contextual difficulties, controlled case variations, and member-specific medical criteria for open-ended answers. We introduce HealthCLBench, comprising 328 controlled clusters, 1,103 cases, and 4,931 rubric criteria across medical consultation and everyday health management. Members share backgrounds while evidence, patient states, or task goals change; each has its own medical scoring requirements. A 70-person medical team annotates and reviews the simulated cases and their rubrics. Human review of 70
    
[^122]: 面向以人为本的建筑数字孪生系统的AI驱动助手：设计、实现与对比用户评估

    An AI-Driven Assistant for Human-Centric Building Digital Twin Systems: Design, Implementation, and Comparative User Evaluation

    [https://arxiv.org/abs/2611.22350](https://arxiv.org/abs/2611.22350)

    本文设计并实现了集成于以人为本建筑数字孪生系统中、用于室内环境质量监测的AI对话助手，用户对比实验表明其能显著缩短任务完成时间并降低感知工作负荷。

    

    智能建筑会产生持续不断的运营数据流，然而这些丰富的数据很少能转化为可供建筑居住者和管理者轻松获取、可直接用于决策的信息。当前的建筑数字孪生系统（BDTS）通常面向技术专家，并依赖于静态的、千篇一律的界面。本文介绍了一个集成于以人为本的BDTS中的AI驱动对话助手的设计、实现与对比用户评估，该系统用于室内环境质量监测，并结合了自适应信息呈现与自动建议功能。该原型部署于纽约大学阿布扎比分校。研究采用受控的双组、受试者间准实验设计，通过基于波次的非随机分组方式，将基线界面与实验界面进行了对比。结果显示，实验界面显著降低了任务完成时间和感知工作负荷，而准确性和……（原文摘要在此处截断）

    arXiv:2611.22350v1 Announce Type: cross  Abstract: Smart buildings generate streams of operational data, yet this richness rarely translates into accessible, decision-ready information for stakeholders who inhabit and manage them. Current Building Digital Twin Systems (BDTS) often remain oriented toward technical specialists and rely on static, one-size-fits-all interfaces. This paper presents the design, implementation, and comparative user evaluation of an AI-driven conversational assistant integrated into a human-centric BDTS for Indoor Environmental Quality monitoring, combined with adaptive information presentation and auto-suggest functionality. The prototype was deployed at New York University Abu Dhabi. A controlled, two-arm, between-subjects, quasi-experimental study with wave-based, non-randomized assignment compared the Baseline interface with the Experimental interface. Results showed significant reductions in task-completion time and perceived workload, while accuracy and 
    
[^123]: 投资研究领域AI智能体的可靠评估：性能与有效性评估框架

    Reliable Evaluation of AI Agents for Investment Research: A Framework for Performance and Validity Assessment

    [https://arxiv.org/abs/2611.22338](https://arxiv.org/abs/2611.22338)

    该论文提出了一个面向投资研究领域的AI智能体评估框架，通过区分结果质量与执行质量，并采用任务特定标准、确定性验证器和独立大语言模型评估器，实现对智能体性能与评估有效性的可靠评估。

    

    我们研究如何评估承担投资研究任务的AI智能体，这些智能体与外部信息源和计算工具交互，以生成金融分析报告和研究报告。一个重要的挑战在于，智能体最终输出的质量并不一定能反映其执行过程的质量，而评估结果的可靠性也取决于评估程序本身的有效性。此外，不同的投资研究任务在数值精度、信息检索、金融分析和支持证据等方面的要求差异很大，这使得难以用一套统一的标准来评估智能体的性能。我们提出了一个面向领域的评估框架，该框架将结果质量与执行质量区分开来，并通过任务特定的评估标准、确定性验证器以及独立的大语言模型（评估器）来适应这些差异。

    arXiv:2611.22338v1 Announce Type: new  Abstract: We consider the problem of evaluating AI agents undertaking investment-research tasks, where agents interact with external information sources and computational tools to produce financial analyses and research reports. An important challenge is that the quality of an agent's final output does not necessarily reflect the quality of its execution process, while the reliability of the resulting assessment also depends on the validity of the evaluation procedure. Furthermore, investment-research tasks differ considerably in their requirements for numerical accuracy, information retrieval, financial analysis and supporting evidence, making it difficult to assess agent performance using a uniform set of criteria. We propose a domain-oriented evaluation framework that distinguishes outcome quality from execution quality and accommodates these differences through task-specific criteria, deterministic validators and independent large language mod
    
[^124]: 面向车辆重定位的多阶段上下文优化

    Multistage Contextual Optimization for Vehicle Repositioning

    [https://arxiv.org/abs/2611.22330](https://arxiv.org/abs/2611.22330)

    本文提出一种多阶段上下文优化方法，将车辆重定位建模为含宏观基本图的多阶段线性规划，并利用决策性能的精确梯度训练预测模型，以优化网约车平台的供需平衡。

    

    网约车平台通过重新定位闲置车辆来改善空间供需平衡。该问题是预测驱动且多阶段的：在每个阶段，平台预测未来需求、执行当前决策，并利用所产生状态来初始化下一阶段。我们构建了一个多阶段线性规划（LP），其中宏观基本图（MFD）将车辆累积量与行程完成率联系起来。随后，我们开发了一种多阶段上下文优化方法，利用决策性能来训练预测模型。该学习问题结合了多个相互耦合的阶段、依赖于先前决策的上下文，以及同时影响线性规划目标函数、约束矩阵和右端项的预测。我们推导出了精确的梯度，该梯度包含三个部分：参数变化会影响预测进而影响决策；先前的决策通过继承状态影响当前决策；该状态也会改变预测。

    arXiv:2611.22330v1 Announce Type: new  Abstract: Ride-hailing platforms reposition idle vehicles to improve spatial supply-demand balances. This problem is prediction-driven and multistage: at each stage, the platform forecasts future demand, executes current decisions, and uses the resulting state to initialize the next stage. We formulate a multistage linear program (LP) with a macroscopic fundamental diagram (MFD) linking accumulation to trip completion. We then develop a multistage contextual optimization approach that uses decision performance to train the prediction model. The learning problem combines multiple coupled stages, context that depends on earlier decisions, and predictions that affect the objective, the constraint matrix, and the right-hand side of the LP. We derive the exact gradient, which has three parts: parameters change forecasts and hence decisions; earlier decisions change the current decision through the inherited state; and this state also changes forecasts 
    
[^125]: 智能体轨迹作为软件质量保证的证据

    Agent Trajectories as Evidence for Software Quality Assurance

    [https://arxiv.org/abs/2611.22323](https://arxiv.org/abs/2611.22323)

    该论文提出将智能体轨迹与规格说明及AI生成产物结合分析，作为针对性软件质量保证的证据，以优先审查智能体影响重大的决策、暴露隐式假设并持续改进开发实践。

    

    自主软件智能体在极少人工监督的情况下，越来越多地承担软件的规划、实现与测试工作。它们的决策可能引入最终代码和通过的测试无法充分暴露的隐式假设。我们提出将可观察的智能体轨迹，连同软件与智能体规格说明以及AI生成的产物一同分析，作为针对性质量保证的证据。其目标是优先审查那些影响重大的决策，调查它们对软件问题可能产生的作用，并为改进规格说明和开发实践提供依据。工业界的经验激发了这一想法；现有的执行记录和初步分析方法使其具备可行性。然而，仅凭轨迹并不能证明决策是合理的、结果是可接受的，或纠正性修改是有效的。我们探讨这种证据如何支持评审、诊断和持续的质量改进，以及何时采用更简单的质量保证方法……

    arXiv:2611.22323v1 Announce Type: cross  Abstract: Autonomous software agents increasingly plan, implement, and test software with limited human oversight. Their decisions may introduce implicit assumptions that final code and passing tests do not adequately expose. We propose analyzing observable agent trajectories alongside software and agent specifications and AI generated artifacts as evidence for targeted quality assurance. The aim is to prioritize consequential decisions for scrutiny, investigate their possible contribution to software problems, and inform improvements to specifications and development practices. Industrial experience motivates the idea; available execution records and preliminary analysis methods make it plausible. However, trajectories alone do not establish that decisions are justified, outcomes acceptable, or corrective changes effective. We ask how this evidence can support review, diagnosis, and continuous quality improvement, and when simpler QA approaches
    
[^126]: EviDARL：基于交替强化学习的证据中心双智能体推理框架，用于多表格问答

    EviDARL: Evidence-Centric Dual-Agent Reasoning via Alternating Reinforcement Learning for Multi-Table Question Answering

    [https://arxiv.org/abs/2611.22310](https://arxiv.org/abs/2611.22310)

    该论文提出EviDA双智能体框架及EviDARL交替强化学习方法，通过证据构建智能体生成可验证的中间证据轨迹，并与推理回答智能体协作，以细粒度奖励驱动的参数高效交替强化学习提升多表格问答的多跳推理能力。

    

    尽管基于大语言模型的表格推理已取得进展，但大多数研究仍聚焦于单表格设置。多表格问答需要在复杂的表格关系以及分散于长上下文中的证据上进行多跳推理，且往往缺乏可验证的中间轨迹。为应对这些挑战，我们提出了EviDA，一个以证据为中心的双智能体框架，包含一个证据构建智能体和一个推理与回答智能体。证据构建智能体在完整表格上生成并执行表格操作程序，以产出针对特定问题的证据和可验证的中间轨迹，从而简化后续推理。推理与回答智能体随后对这些证据进行多跳推理以得出最终答案。我们进一步提出了EviDARL，该方法将智能体专属的细粒度奖励与参数高效的交替强化学习相结合，以提升两个智能体之间的协作能力。

    arXiv:2611.22310v1 Announce Type: new  Abstract: Despite advances in LLM-based table reasoning, most studies focus on single-table settings. Multi-table QA requires multi-hop reasoning over complex table relationships and evidence dispersed across long contexts, often without verifiable intermediate traces. To address these challenges, we introduce EviDA, an evidence-centric dual-agent framework comprising an Evidence Construction Agent and a Reasoning and Answering Agent. The Evidence Construction Agent generates and executes table-manipulation programs over complete tables to produce question-specific evidence and verifiable intermediate traces, simplifying subsequent reasoning. The Reasoning and Answering Agent then performs multi-hop reasoning over this evidence to derive final answers. We further introduce EviDARL, which combines agent-specific fine-grained rewards with parameter-efficient alternating reinforcement learning to improve collaboration between the agents. By optimizin
    
[^127]: 基于模型原生Trie解码与共享上下文注意力的高效宽束生成式检索

    Efficient Wide-Beam Generative Retrieval with Model-Native Trie Decoding and Shared-Context Attention

    [https://arxiv.org/abs/2611.22309](https://arxiv.org/abs/2611.22309)

    本文提出Trie-Head模型原生输出接口与共享上下文注意力机制，通过将标识符Trie存为GPU张量、避免全词表评分和重复注意力计算，在不重训模型的情况下大幅提升宽束生成式检索的服务效率。

    

    生成式检索（GR）将目录条目预测为短语义标识符（SIDs）。基于Trie约束的束搜索可以保证这些标识符的有效性，但宽束搜索暴露了两个服务开销：大部分会被丢弃的全词表分数张量，以及对同一提示词的重复注意力计算。我们提出了Trie-Head，这是一种模型原生的输出接口，它将标识符Trie存储为经过验证的GPU张量。与Python自定义logit处理器（CLP）相比，它通过避免Python约束回调以及全词表候选缓冲区，减少了CPU与GPU之间的协调和内存流量。它对紧凑的码本候选进行评分，并流式传输全词表归一化器，在真实算术中保留原始模型的有效token分数，而无需重新训练。共享上下文注意力对这一输出路径形成补充，通过在多个束之间复用提示词的key/value块，同时保留各束私有的解码尾部。我们在SGLang中实现了这两种机制并对其进行评估。

    arXiv:2611.22309v1 Announce Type: new  Abstract: Generative retrieval (GR) predicts catalog entries as short semantic identifiers (SIDs). Trie-constrained beam search keeps these identifiers valid, but wide beams expose two serving costs: vocabulary-wide score tensors that are mostly discarded, and repeated attention over the same prompt. We present Trie-Head, a model-native output interface that stores the identifier trie as validated GPU tensors. Compared with a Python custom logit processor (CLP), it reduces CPU--GPU coordination and memory traffic by avoiding Python constraint callbacks and vocabulary-wide candidate buffers. It scores compact codebook candidates and streams the full-vocabulary normalizer, preserving the original model's valid-token scores in real arithmetic without retraining. Shared-context attention complements this output path by reusing prompt key/value blocks across beams while preserving their private decode tails. We implement both mechanisms in SGLang and e
    
[^128]: 学习编辑IFC建筑模型：可验证检查器作为依赖感知编辑智能体的训练信号

    Learning to Edit IFC Building Models: Verifiable Checkers as the Training Signal for Dependency-Aware Editing Agents

    [https://arxiv.org/abs/2611.22307](https://arxiv.org/abs/2611.22307)

    该论文提出IfcGym环境与IfcEdit训练方法，以四个可验证检查器的合取作为严格成功信号，通过SFT蒸馏与GRPO强化学习训练出能够感知依赖关系、可靠编辑IFC建筑模型的语言模型智能体。

    

    在行业基础类（IFC）模式中编辑建筑信息模型，需要在保持无关实体不变的同时，将变更传播到存在依赖关系的元素上。现有的用于IFC编辑的语言模型智能体仅使用提示（prompt）方式驱动，而不会从编辑结果中学习。基于目标文件的比较无法说明哪些替代编辑是允许的，也无法说明哪些无关实体必须保持不变。我们提出了IfcGym，这是一个CodeAct环境，语言模型通过可执行代码编辑IFC，并由四个检查器依据任务规范验证输入模型、意图、依赖与保留方面的有效性，四者的合取构成严格成功。IfcEdit通过监督微调（SFT）将经过验证的教师轨迹蒸馏到Qwen3.5-9B学生模型中，随后应用组相对策略优化（GRPO），以严格成功作为奖励。在IfcEditBench（涵盖13个留出真实项目的1,990个编辑任务）上，最终的学生模型……（摘要原文在此处截断）

    arXiv:2611.22307v1 Announce Type: new  Abstract: Editing a building information model in the Industry Foundation Classes (IFC) schema requires propagating changes to dependent elements while preserving unrelated entities. Existing language-model agents for IFC editing use prompted models without learning from edit outcomes. Target-file comparisons do not specify which alternative edits are allowed or which unrelated entities must remain unchanged. We present IfcGym, a CodeAct environment in which a language model edits IFC through executable code and four checkers verify validity relative to the input model, intent, dependency and preservation against a task specification. Their conjunction is strict success. IfcEdit distils verified teacher episodes into a Qwen3.5-9B student through supervised fine-tuning (SFT), then applies Group Relative Policy Optimization (GRPO) with strict success as the reward. On IfcEditBench, 1,990 editing tasks on 13 held-out real projects, the final student 
    
[^129]: MCTS 在统计错误的东西：为 LLM 推理学习探索计数

    MCTS Is Counting the Wrong Thing: Learning Exploration Counts for LLM Reasoning

    [https://arxiv.org/abs/2611.22301](https://arxiv.org/abs/2611.22301)

    论文发现 UCT 对同一推理策略的不同文本表述重复给予探索奖励导致计数碎片化、损害探索效率，并提出 LPC-MCTS，利用轻量级 Coin-Flipping Network 估计表示覆盖率作为探索伪计数，让相关分支共享探索证据，在数学推理与搜索基准上持续超越强 MCTS 基线。

    

    蒙特卡洛树搜索（MCTS）在强化学习中取得了里程碑式的成功，但其在 LLM 推理中的收益尽管投入了大量搜索计算，却可能仍然有限。我们识别出一个被忽视的瓶颈：MCTS 所统计的对象。标准 UCT 为文本分支分配独立的访问计数，使得同一推理策略的不同表述能够反复获得探索奖励。我们的理论分析表明，这种计数碎片化会损害探索效率，具体表现为累积遗憾的增加。为解决这一问题，我们提出了学习伪计数 MCTS（LPC-MCTS），它仅改变了 UCT 中的探索统计量。一个轻量级的“抛硬币网络”头基于策略的隐藏表示来估计表示覆盖率，使相关分支能够共享探索证据。在 5 个数学推理基准和 7 个搜索基准上，LPC-MCTS 持续优于强大的 MCTS 基线。

    arXiv:2611.22301v1 Announce Type: new  Abstract: Monte Carlo tree search (MCTS) has powered landmark successes in reinforcement learning, yet its gains in LLM reasoning can remain modest despite substantial search effort. We identify an overlooked bottleneck: what MCTS counts. Standard UCT assigns independent visit counts to textual branches, allowing different formulations of the same reasoning strategy to repeatedly receive exploration bonuses. Our theoretical analysis shows that this count fragmentation can impair exploration efficiency, as quantified by increased cumulative regret. To address this problem, we introduce Learned Pseudo-Count MCTS (LPC-MCTS), which changes only the exploration statistic in UCT. A lightweight Coin-Flipping Network head on the policy's hidden representations estimates representation coverage, enabling related branches to share exploration evidence. Across 5 mathematical-reasoning and 7 search benchmarks, LPC-MCTS consistently outperforms strong MCTS bas
    
[^130]: 会自我改进的智能体支架：自我改进智能体支架的统一框架

    Harnesses That Improve Themselves: A Unifying Framework for Self-Improving Agent Harnesses

    [https://arxiv.org/abs/2611.22299](https://arxiv.org/abs/2611.22299)

    该论文提出了一个统一的架构与设计框架，将四个可编辑的运行时组件与诊断、修补、验证的自我改进循环相结合，使智能体支架能够通过检视自身执行轨迹并重写自身代码来持续自我优化。

    

    基础模型能否成功胜任自主智能体，不仅取决于其权重，也同样取决于其运行时软件。这一运行时环境——即智能体支架（agent harness）——负责管理上下文与记忆、引导多轮控制、在沙箱中执行工具操作，并在停止前验证工作成果。由于不同模型在使用工具的方式和出错模式上各不相同，针对某一模型或任务调优的静态支架在其他场景下往往会陷入停滞、浪费令牌或出现性能倒退。这一瓶颈推动了从提示词调优向自我改进智能体支架的转变：这类系统能够检视自身的执行轨迹、将运行时缺陷与模型能力极限区分开来，并重写自身代码。我们提出了一个面向自我改进智能体支架的统一架构与设计框架。该框架将四个可编辑的运行时组件（上下文与状态、智能体控制、动作接口、验证与恢复）与一个四阶段循环相配对，该循环负责诊断故障、提出代码补丁并进行验证……

    arXiv:2611.22299v1 Announce Type: new  Abstract: Whether a foundation model succeeds as an autonomous agent depends as much on its runtime software as on its weights. This runtime, the agent harness, manages context and memory, steers multi-turn control, executes tool actions in sandboxes, and verifies work before stopping. Because models differ in how they use tools and how they fail, a static harness tuned for one model or task often stalls, wastes tokens, or regresses on another. This bottleneck has driven a shift from prompt tuning to self-improving agent harnesses: systems that inspect their own execution traces, separate runtime bugs from model limits, and rewrite their own code. We present a unifying architectural and design framework for self-improving agent harnesses. The framework pairs four editable runtime components (context and state, agent control, action interfaces, and verification and recovery) with a four-stage loop that diagnoses failures, proposes code patches, val
    
[^131]: 流式在策略蒸馏：解耦三个时钟实现分阶段监督消费

    Streaming On-Policy Distillation: Decoupling Three Clocks for Stage-Wise Supervision Consumption

    [https://arxiv.org/abs/2611.22298](https://arxiv.org/abs/2611.22298)

    该论文提出流式在策略蒸馏方法，通过解耦响应生成、监督交付和参数更新这三个时钟，实现分阶段灵活的监督消费控制，在GSM8K蒸馏实验中显著缩小了与同步方法的性能差距。

    

    在在策略蒸馏（OPD）中，响应完成、监督交付和参数更新不必共享相同的粒度。我们通过解耦这三个时钟来开发流式OPD：响应在生成持续进行的同时反复交付新生成的片段，每次交付时保留完整的因果前缀，并仅在新位置上计算损失。随后，一个消费协议组织监督预算、位置选择、陈旧性处理和更新调度，从而可以在训练各阶段对数量、覆盖范围和版本滞后进行控制。在使用VERL将Qwen3-4B蒸馏到Qwen3-0.6B的GSM8K实验中，监督预算匹配将在七个共同候选模型上最佳测试准确率的早期差距从4.55分降低到2.42分。从同一检查点同步继续训练，在u=384,...,576时，在七个共同候选模型上达到73.39%的最佳测试准确率，相比之下同步方法为74.30%，预算匹配方法为71.95%。

    arXiv:2611.22298v1 Announce Type: new  Abstract: In on-policy distillation (OPD), response completion, supervision delivery, and parameter updates need not share the same granularity. We develop streaming OPD by decoupling these three clocks: a response repeatedly delivers newly generated segments while generation continues, retaining the full causal prefix at each delivery and computing loss only on new positions. A consumption protocol then organizes supervision budgets, position selection, staleness handling, and update scheduling, allowing control over quantity, coverage, and version lag across training stages. In GSM8K experiments distilling Qwen3-4B into Qwen3-0.6B with VERL, supervised-budget matching reduces the early gap between best test accuracies over seven common candidates from 4.55 to 2.42 points. Synchronous continuation from the same checkpoint reaches 73.39% best test accuracy over seven common candidates at u=384,...,576, versus 74.30% synchronous and 71.95% budget-m
    
[^132]: FIRE-Drive：面向端到端扩散规划的可行性知情风险评估

    FIRE-Drive: Feasibility-Informed Risk Evaluation for End-to-End Diffusion Planning

    [https://arxiv.org/abs/2611.22297](https://arxiv.org/abs/2611.22297)

    提出FIRE-Drive框架，通过动态可行锚点初始化、鸟瞰图风险证据和高质量紧凑候选监督，改进基于扩散模型的端到端自动驾驶规划中的可行性与风险评估。

    

    基于扩散模型的端到端自动驾驶方法在生成多模态规划轨迹方面展现出强大潜力。然而，其性能仍在很大程度上依赖于轨迹初始化、模式评估和监督质量。约束薄弱的噪声初始化可能引入不可行的候选方案，并增加去噪负担。此外，当候选轨迹在几何上相似但在安全性、交互风险和进度方面存在差异时，选择最优模式仍然具有挑战性。为解决这些问题，我们提出了FIRE-Drive，一个面向端到端自动驾驶的可行性知情风险评估框架。FIRE-Drive通过动态可行锚点初始化、针对候选的鸟瞰图风险证据，以及结合轻量级精炼的紧凑高质量候选监督来改进基于扩散的规划。该设计使初始支撑、风险感知评分与多模态（轨迹生成）相统一……

    arXiv:2611.22297v1 Announce Type: cross  Abstract: Diffusion-based end-to-end autonomous driving models have shown strong potential in generating multimodal planning trajectories. However, their performance still depends heavily on trajectory initialization, mode evaluation, and supervision quality. Weakly constrained noisy initialization may introduce infeasible proposals and increase the burden of denoising. Moreover, selecting the optimal mode remains challenging when candidate trajectories are geometrically similar but differ in safety, interaction risk, and progress. To address these issues, we propose FIRE-Drive, a feasibility-informed risk evaluation framework for end-to-end autonomous driving. FIRE-Drive improves diffusion-based planning through dynamic feasible anchor initialization, candidate-specific bird's-eye-view risk evidence, and compact high-quality candidate supervision with lightweight refinement. This design aligns the initial support, risk-aware scoring, and multim
    
[^133]: SpecSearch：迈向硬件感知的高效推测解码

    SpecSearch: Towards Hardware-Aware Efficient Speculative Decoding

    [https://arxiv.org/abs/2611.22296](https://arxiv.org/abs/2611.22296)

    提出硬件感知搜索引擎SpecSearch，通过分析成本模型与基于校准数据的预测器，在不同工作负载与GPU组合上联合优化解码策略、草稿长度和精度，自动选择高吞吐量的推测解码配置。

    

    推测解码（SD）可以加速大语言模型（LLM）的推理，但其最优配置同时取决于工作负载特性、GPU硬件和可用内存。例如，在RTX 5090上，最先进的推测起草器DFlash在低并发情况下可实现自回归（AR）解码3.50倍的吞吐量，但在高并发且输出较短的情况下仅为0.76倍。我们提出了SpecSearch，这是一个硬件感知的搜索引擎，可在不同工作负载-GPU组合上联合优化解码策略、草稿长度和精度。其分析成本模型捕捉内存与计算约束，以识别可行配置并缩小吞吐量优化的搜索空间。在该模型的指导下，从多个GPU上收集校准测量数据。利用这些测量数据，基于Jev的预测器学习在未见过的GPU上选择高吞吐量的解码配置，而无需进行部署……（摘要原文在此处截断）

    arXiv:2611.22296v1 Announce Type: new  Abstract: Speculative decoding (SD) accelerates large language model (LLM) inference, but its optimal configuration depends jointly on workload characteristics, GPU hardware, and available memory. For example, on the RTX~5090, the state-of-the-art speculative drafter DFlash achieves $3.50\times$ the throughput of autoregressive (AR) decoding at low concurrency, but only $0.76\times$ at high concurrency with short outputs. We present SpecSearch, a hardware-aware search engine that jointly optimizes decoding strategy, draft length, and precision across workload--GPU pairs. Its analytical cost model captures memory and compute constraints to identify feasible configurations and narrow the search space for throughput optimization. Guided by this model, calibration measurements are collected from multiple GPUs. Using the measurements, a Jev-based predictor learns to select high-throughput decoding configurations on held-out GPUs without requiring deplo
    
[^134]: 从感知到想象：面向世界模型的可执行记忆

    Perception into Imagination: Executable Memory for World Models

    [https://arxiv.org/abs/2611.22294](https://arxiv.org/abs/2611.22294)

    提出可执行感知记忆（EPM）世界模型框架，在想象过程中同步推进显式感知状态与神经潜在动态，并将演化读出信息用于奖励预测、价值估计和策略学习，显著提升智能体在视野或接触丢失后利用过往感知信息的能力。

    

    物理智能体在失去视觉或接触之后，必须利用其先前看到或触摸过的信息。它们的行动会改变这些被记忆事实的含义。我们提出了可执行感知记忆（EPM），这是一种世界模型框架，在想象过程中使显式感知状态与神经潜在动态一同演化推进。其不断更新的读出信息为奖励预测、价值估计和策略学习提供依据。我们将奖励输入与产生这些奖励的状态转移对齐，并在记录动作的开环推演上对奖励头进行监督。在 Memory Maze 13×13 上，EPM 在 100 万次交互时达到 30.37 的回报，而相同的演化读出仅在共同感知前端上用于策略和价值学习时为 23.37：增益为 7.00 [3.80, 10.21]。完整方法还在 9×9 和 MiniWorld 中提升了控制性能。对记忆中视野外到达目标的探测显示，两种视觉实现中均表现出更强的奖励预期。

    arXiv:2611.22294v1 Announce Type: cross  Abstract: Physical agents must use what they previously saw or touched after visibility or contact is lost. Their actions change the implications of these remembered facts. We introduce executable perceptual memory (EPM), a world-model framework that advances explicit perceptual state alongside neural latent dynamics during imagination. Its evolving read-out informs reward prediction, value estimation and policy learning. We align reward inputs with the transitions that produced them and supervise the reward head on open-loop rollouts of recorded actions. On Memory Maze $13\times13$, EPM reaches 30.37 return at 1M interactions, compared with 23.37 when the same evolving read-out informs only policy and value on a common perceptual front end: a gain of 7.00 [3.80, 10.21]. The complete method also improves control on $9\times9$ and in MiniWorld. Probes of remembered, out-of-view arrivals show stronger reward anticipation in both visual implementat
    
[^135]: 当世界成为测试套件：面向物理世界的智能体智能

    When the World Is the Test Suite: Agentic Intelligence for the Physical World

    [https://arxiv.org/abs/2611.22291](https://arxiv.org/abs/2611.22291)

    本文提出将物理世界视为程序的测试套件，系统考察了面向物理世界的智能体智能这一新兴领域，围绕智能体如何为物理世界编写程序、如何测试这些程序以及如何从测试结果中学习三个核心问题展开论述。

    

    为修复软件而构建的编程智能体，如今开始为机器人、实验室和工业控制器编写、测试和修复程序。与软件工程不同，这些代码并不附带现成的测试套件。程序是否有效，必须通过在真实机器人上运行、在物理模拟器中执行、或通过它所辅助训练的策略来验证，而这类验证证据获取成本高昂，且本身也可能是错误的。随着这些智能体承担越来越多的物理任务，程序的评判方式与其编写方式变得同等重要。本文通过将物理世界视为程序的测试套件，考察了这一新兴领域——面向物理世界的智能体智能。我们围绕三个问题组织讨论：智能体如何为物理世界编写程序、这些程序如何被测试、以及智能体如何从测试结果中学习。我们首先追溯了面向机器人的语言模型与软件工程智能体如何走向融合，并探讨哪些软件测试实践可以……

    arXiv:2611.22291v1 Announce Type: cross  Abstract: Coding agents built to repair software now write, test, and repair programs for robots, laboratories, and industrial controllers. Unlike in software engineering, no test suite comes with the code. Whether a program works must be established by running it on a robot, in a physics simulator, or through the policies it helps to train, and such evidence is costly to obtain and can itself be wrong. As these agents take on more physical tasks, how their programs are judged matters as much as how they are written. This paper examines this emerging area, agentic intelligence for the physical world, by treating the world as the program's test suite. We organize the discussion around three questions: how agents write programs for the physical world, how these programs are tested, and how agents learn from the results. We first trace how language models for robots and software engineering agents came together, ask which software testing practices
    
[^136]: ConfGuard：面向预测性维护大语言模型智能体的基于物理规律的参考监视器

    ConfGuard: a physics-grounded reference monitor for predictive-maintenance large-language-model agents

    [https://arxiv.org/abs/2611.22288](https://arxiv.org/abs/2611.22288)

    ConfGuard 提出了一种基于退化物理规律的参考监视器，通过保形预测校验预测性维护 LLM 智能体的剩余寿命预测是否符合物理约束（寿命不上升且每周期至多降一周期），从而抵御文档注入攻击和传感器扰动。

    

    大语言模型智能体如今已能闭环工业预测性维护流程：读取传感器数据窗口、检索维护规程并派发工单。此处存在两条攻击通道：检索文档中被注入的文本可能劫持智能体的决策，而有界的传感器扰动可能破坏其背后的剩余使用寿命（RUL）估计。近期的智能体防御手段只管控数据来源和工具权限，却无法判断预测量在物理上是否合理。ConfGuard 是一个采用基于物理策略的参考监视器。其保形监视器以资产已确认的历史记录为锚点，依据退化物理规律检验每一项预测：被封顶的剩余寿命绝不会上升，且每个周期至多下降一个周期。通过在完整的资产轨迹上校准阈值，操作员可以设定运行中的监视器对健康资产数据窗口触发升级告警的预期比例。一个物理策略验证器负责检查智能体的维护紧急性判断（原文在此处截断）。

    arXiv:2611.22288v1 Announce Type: new  Abstract: Large-language-model agents now close the industrial predictive-maintenance loop, reading sensor windows, retrieving procedures and dispatching work orders. Two attack channels meet there: injected text in a retrieved document can hijack the decision, and a bounded sensor perturbation can corrupt the remaining-useful-life estimate behind it. Recent agent defenses police data provenance and tool privilege but cannot tell whether a prognostic quantity is physically plausible. ConfGuard is a reference monitor with a physics-grounded policy. Its conformal monitor tests each prediction, anchored to the asset's admitted history, against degradation physics: capped remaining life never rises and falls at most one cycle per cycle. Calibrating its threshold over whole asset trajectories lets the operator set the expected share of a clean asset's windows the running monitor escalates. A physics-policy validator checks the agent's maintenance-urgen
    
[^137]: 控制器奇异条件下横向反馈线性化的吸引域学习

    Learning Regions of Attraction for Transverse Feedback Linearization under Controller Singularities

    [https://arxiv.org/abs/2611.22286](https://arxiv.org/abs/2611.22286)

    提出通过将状态约束的Zubov值函数分解为解析已知的二次代价项与仅需拟合值迭代学习的安全值函数，在控制器奇异点附近学习横向反馈线性化的受约束吸引域，并给出内层经区间分支定界验证、外层经样本测试的组合估计。

    

    横向反馈线性化（TFL）通过精确线性化误差动力学来实现路径跟随，但其控制器在某个奇异集上无定义，例如独轮车的零速度或四旋翼的零推力。受约束吸引域——即闭环系统在不违反状态或正则性约束的前提下收敛到路径的初始状态集合——决定了控制器可以被安全激活的区域。利用精确的线性误差动力学，我们证明该区域是可容许集的最大前向不变子集。随后，我们将该区域的状态约束Zubov值函数分解为解析已知的二次代价项和一个安全值函数，并通过拟合值迭代仅学习后者，无需无限时域的轨迹标签。最终得到的估计由经区间分支定界验证的内层与在样本上测试的学习外层组合而成。在独轮……

    arXiv:2611.22286v1 Announce Type: new  Abstract: Transverse feedback linearization (TFL) achieves path following by exactly linearizing the error dynamics, but its controller is undefined on a singular set, such as zero speed for a unicycle or zero thrust for a quadrotor. The constrained region of attraction, the set of initial states from which the closed loop converges to the path without violating state or regularity constraints, determines where the controller can be safely activated. Exploiting the exact linear error dynamics, we show that this region is the largest forward invariant subset of the admissible set. We then decompose its state-constrained Zubov value function into an analytically known quadratic cost-to-go and a safety value function. We learn only the latter, by fitted value iteration without infinite-horizon trajectory labels. The resulting estimate combines an inner layer verified by interval branch and bound with a learned outer layer tested on samples. On a unic
    
[^138]: 迈向气候降尺度中的物理约束整流流：投影能证明什么？

    Towards Physics-Constrained Rectified Flow for Climate Downscaling: What Projections Can Certify?

    [https://arxiv.org/abs/2611.22283](https://arxiv.org/abs/2611.22283)

    提出物理约束整流流（PC-RF）气候降尺度方法，通过训练惩罚项与采样后投影将降水非负性违反率和质量守恒误差精确降为零，并将散度误差降低约30%。

    

    统计降尺度能够恢复洪水和作物胁迫背后的区域细节，但扩散模型和流匹配生成器可能产生违反其粗分辨率输入守恒定律的锐利场。物理约束整流流是一种条件流匹配降尺度器。其训练惩罚项针对风场散度、负降水以及相对于粗分辨率总量的质量漂移，而采样后的投影可以精确消除后两者。在一个无需训练网络的受损合成场上，投影将非负性违反率（NVR）从0.236、质量守恒误差（MCE）从0.184降至零，同时惩罚项将散度误差（DE）降低约30%。在ERA5数据集上，投影使PC-RF和无惩罚整流流基线RF-base的非负性违反率精确为零，且质量守恒误差在float32舍入精度内为零。这些惩罚项并未带来能够从训练预算中分离出来的感知增益。

    arXiv:2611.22283v1 Announce Type: new  Abstract: Statistical downscaling restores the regional detail behind flooding and crop stress, but diffusion and flow-matching generators can produce sharp fields that break the conservation laws of their coarse input. Physics-Constrained Rectified Flow (PC-RF) is a conditional flow-matching downscaler. Its training penalties target wind divergence, negative precipitation, and mass drift from the coarse total, and a projection after sampling removes the last two exactly. On one corrupted synthetic field with no trained network, the projection drives the Non-negativity Violation Rate (NVR) from 0.236 and the Mass-Conservation Error (MCE) from 0.184 to zero, and the penalties lower the Divergence Error (DE) by about 30%. On ERA5 the projection gives PC-RF and RF-base, the penalty-free rectified-flow baseline, an NVR of exactly zero and an MCE of zero up to float32 rounding. The penalties add no perceptual gain that we can separate from training bud
    
[^139]: SeasonRainBench：全球季节性降水基准

    SeasonRainBench: Global Seasonal Precipitation

    [https://arxiv.org/abs/2611.22281](https://arxiv.org/abs/2611.22281)

    该论文提出了首个全球季节性降水预报基准SeasonRainBench，涵盖未来1至5个月的月度降水预报，并引入超越ACC和RMSE、体现业务需求的评估指标，以弥合数据驱动方法与业务数值集合预报系统之间的评估差距。

    

    准确的季节性降水预报可以帮助社会提前数月为干旱、洪水和农作物歉收做好准备。尽管数据驱动方法在天气和次季节预报领域取得了快速进展，但业务化的季节性降水预测仍然依赖于数值集合预报系统。这些系统计算成本高昂，且存在随预报时效增加而不断增大的偏差和漂移，这促使人们寻求更高效的数据驱动替代方案。然而，这一转型面临三个相互关联的挑战：将数据驱动预报与业务预报进行比较需要（1）一个目前尚不存在的全球基准；对其进行评分需要（2）超越异常相关系数（ACC）和均方根误差（RMSE）、能够体现业务优先事项的评估指标；而训练数据驱动模型则必须（3）依赖单一且时间跨度较短的观测记录。为了解决前两个挑战，我们提出了SeasonRainBench——首个此类基准，涵盖未来1至5个月的月度降水预报。

    arXiv:2611.22281v1 Announce Type: new  Abstract: Accurate seasonal precipitation forecasts help societies prepare for drought, flooding, and crop failure months in advance. While data-driven methods have made rapid progress in weather and subseasonal forecasting, operational seasonal precipitation prediction still relies on numerical ensemble systems. These systems are computationally expensive and exhibit biases and drift that increase with lead time, motivating more efficient data-driven alternatives. Yet this transition faces three linked challenges: comparing data-driven forecasts with operational ones requires (1) a global benchmark that does not exist, scoring them requires (2) metrics that capture operational priorities beyond anomaly correlation (ACC) and RMSE, and training them must (3) rely on a single, short observational record. To address the first two challenges, we introduce \BenchName, the first such benchmark, covering monthly precipitation forecasts 1--5 months ahead.
    
[^140]: DACE：面向鲁棒大语言模型安全对齐的多样性驱动对抗协同进化

    DACE: Diversity-Driven Adversarial Co-Evolution for Robust LLM Safety Alignment

    [https://arxiv.org/abs/2611.22278](https://arxiv.org/abs/2611.22278)

    提出多样性驱动的对抗协同进化框架DACE，通过同时解决攻击策略坍缩与防御方对抗遗忘两大问题，实现更鲁棒的大语言模型安全对齐。

    

    确保大语言模型（LLM）的鲁棒安全对齐正变得日益困难，因为对抗性攻击不断演化，超越了基于预先收集数据构建的对齐流程。现有的协同进化框架让防御方追赶不断变化的攻击方，但其训练动态暴露出两种相互叠加的病态：攻击策略坍缩，即攻击方过拟合一组狭窄的高奖励改写策略；以及防御方对抗遗忘，即随着攻击分布漂移，防御方丧失了对早期攻击的应对能力。现有的补救措施都只是部分有效的：攻击方侧的多样性奖励衡量的是文本新颖性而非策略新颖性，防御方侧的回放机制依赖于离散的评判分数，忽略了估计不确定性、响应随机性以及防御方的非平稳性。我们提出了DACE，一个多样性驱动的对抗协同进化框架，能够同时针对这两种病态。

    arXiv:2611.22278v1 Announce Type: new  Abstract: Ensuring robust safety alignment of large language models (LLMs) is increasingly difficult as adversarial attacks evolve and outpace alignment pipelines built on pre-collected data. Existing co-evolutionary frameworks let the defender chase a moving attacker, yet their training dynamics expose two compounding pathologies: \emph{attack strategy collapse}, where the attacker overfits to a narrow set of high-reward rewrites, and \emph{defense adversarial forgetting}, where the defender loses competence against earlier attacks as the attack distribution drifts. Existing remedies are partial: attacker-side diversity rewards score textual novelty rather than strategy novelty, and defender-side replay relies on discrete judge scores that ignore estimation uncertainty, response stochasticity, and defender non-stationarity. We introduce \textbf{\method}, a diversity-driven adversarial co-evolution framework that targets both pathologies jointly. 
    
[^141]: 基于机器学习势的神经网络路径优化方法寻找过渡态

    Neural Network Path Optimization for Finding Transition States on a Machine Learned Potential

    [https://arxiv.org/abs/2611.22272](https://arxiv.org/abs/2611.22272)

    提出Popcornn框架，通过在机器学习势上优化连续神经网络路径来寻找过渡态，消除了路径离散化伪影并实现非局部优化。

    

    过渡态决定着化学反应的速率和结果，然而定位过渡态仍然是计算化学中的一个核心挑战。双端搜索方法可以在给定反应物和产物结构的情况下寻找过渡态，但通常依赖于为密度泛函理论开发的离散化路径表示，而在密度泛函理论中势能面的计算代价十分高昂。机器学习原子间势（MLIPs）极大地降低了这一成本，并借助自动微分技术高效获取高阶导数。我们提出了Popcornn，这是一个通过在机器学习势之上利用并行化自适应数值积分器优化连续神经网络路径来寻找过渡态的框架。这种组合消除了路径离散化带来的伪影，降低了对超参数的敏感性，并能够利用二阶势能面信息进行非局部优化以克服局部特征。

    arXiv:2611.22272v1 Announce Type: cross  Abstract: Transition states govern chemical reaction rates and outcomes, yet locating them remains a central challenge in computational chemistry. Double-ended search methods find transition states given reactant and product structures, but generally rely on discretized path representations developed for density functional theory, where potential energy surface evaluations are expensive. Machine learned interatomic potentials (MLIPs) dramatically reduce this cost and leverage automatic differentiation to access higher-order derivatives efficiently. We introduce Popcornn, a framework to find transition states by optimizing continuous neural network paths atop an MLIP with a parallelized adaptive numerical integrator. This combination eliminates path discretization artifacts, reduces hyperparameter sensitivity, and enables non-local optimization to overcome local features using second-order potential-energy-surface information. Popcornn consistent
    
[^142]: 教语言模型利用检索反馈收窄布尔式专利检索

    Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback

    [https://arxiv.org/abs/2611.22270](https://arxiv.org/abs/2611.22270)

    该论文利用日本专利局检索报告提炼专家策略，并提出行级“保留引用的收窄信用”学习方法对语言模型进行离线迭代优化，使其生成的布尔专利检索策略在保持引用召回率的同时大幅收窄候选集合，将包含引用文献的中位数集合规模从 768 篇降至 50 篇以下。

    

    专利现有技术检索人员逐行发布布尔查询，先扩大检索范围以找到相关文献，再逐步收窄，直到候选文献数量可供筛选。我们将 38,397 份日本专利局检索报告提炼为可执行的专家策略，重新执行这些策略，并通过引用召回率以及包含每条引用文献的最小文献集合来评估生成的策略。通过提示词驱动的模型能够写出有效且高召回率的策略，但会把引用文献留在规模过大、无法人工筛选的文献集合中。我们从打分的执行轨迹中以两种形式对生成器进行离线适配：一种是从筛选感知的策略分数中进行策略级学习；另一种是从“保留引用的收窄信用”中进行行级学习，即对那些将已找到的引用文献保留在更小集合中的查询行赋予更高权重。在共享的执行轨迹上，行级信用在筛选相关指标上优于策略级分数；经过迭代优化，它将包含引用文献的集合规模中位数从 768 降至 50 以下，并将更多引用文献置于少于 100 篇文献的集合中。

    arXiv:2611.22270v1 Announce Type: cross  Abstract: Patent prior-art searchers issue Boolean queries line by line, broadening to find relevant documents and narrowing until candidates can be screened. We distill 38,397 Japan Patent Office search reports into executable expert strategies, re-execute them, and evaluate generated strategies by citation recall and the smallest set containing each citation. Prompted models write valid, high-recall strategies but leave citations in sets too large to screen. We adapt a generator offline from scored rollouts in two forms: strategy-level learning from a screening-aware strategy score, and line-level learning from citation-preserving narrowing credit, which upweights the query lines keeping a found citation in a smaller set. On shared rollouts, line-level credit outperforms strategy-level scores on screening measures; iterated, it cuts the median citation-containing set from 768 to under 50 and places more citations in sets under 100 documents th
    
[^143]: 预见还是投机？让未来思考在AI伦理中真正产生价值

    Anticipating or Speculating? Making Futures Pay in AI Ethics

    [https://arxiv.org/abs/2611.22268](https://arxiv.org/abs/2611.22268)

    本文论证了为医生与医疗AI分歧制定统一决策规则的努力存在根本缺陷，因为算法推荐的恰当使用是高度情境化的经验性医学问题，并提出应区分技术伦理中有价值的“预见”与无益的“投机”。

    

    哲学家们近来一直试图为医生的判断与医疗人工智能（AI）系统的建议出现分歧的情形制定统一的决策规则。我认为这一努力存在致命缺陷，因为医生对算法推荐系统的表现评估和恰当使用是高度依赖具体情境的、经验性的医学问题，而非认识论问题。在这类情形中，一般性决策规则只有通过抽象掉相关的——实际上是具有决定性的——情境因素才能被规定。所谓的医生-AI分歧问题以及其他类似问题，实际上牵涉到技术伦理中关于构想可能未来的可行性与实用性的长期争论：这究竟是有用的“预见”还是无益的“投机”？我认为二者是可以加以区分的。预见通过探索合理的未来图景来帮助指导当前的行动……

    arXiv:2611.22268v1 Announce Type: cross  Abstract: Philosophers have recently been trying to develop uniform decision rules for instances in which physicians' judgments diverge from the recommendations of medical artificial intelligence (AI) systems. I argue that this effort is fatally flawed because the performance and appropriate use of algorithmic recommendation systems by physicians are highly context-specific, empirical medical matters rather than epistemological ones. In such instances, general decision rules can be stipulated only to the extent that they abstract away from relevant, indeed determinative, contextual circumstances. The alleged problem of physician-AI disagreement, and others like it, play into longstanding disputes in technology ethics about the viability and utility of envisioning possible futures: is this useful "anticipation" or wasteful "speculation"? I suggest that the two may be distinguished. Anticipation explores plausible futures to help inform present ac
    
[^144]: 保真度遗憾：在可验证设计中度量模拟器的误设

    Fidelity Regret: Measuring Simulator Misspecification in Verifiable Design

    [https://arxiv.org/abs/2611.22263](https://arxiv.org/abs/2611.22263)

    本文提出“保真度遗憾”指标来量化基因电路设计中廉价模拟器与真实性能之间的偏差，发现模拟器选择的设计在几乎所有测试单元中都产生正遗憾（平均归一化遗憾为0.369，仅6个单元为零），且偏差大小强烈依赖于所评估的性能指标。

    

    RLVF（基于可验证反馈的强化学习）假设验证器评估的正是所关心的目标量。在基因电路设计中，拓扑验证器可以确认某个程序是计算目标函数的有效电路，但对其物理性能进行排序则需要依赖模拟器。我们测量了“保真度遗憾”（fidelity regret），即当使用廉价的模拟器来选择设计时所产生的参考简单遗憾，实验覆盖了一个包含21,362个设计组合的子空间，涉及5个经过实测的门文库和3种生物体。精确的有序分配检查在每个测试单元中都保持正遗憾，同时其幅度在两个方向上均有变化。72个单元的平均归一化遗憾为0.369，其中仅有6个单元的遗憾为零。这种偏差强烈依赖于参考轴：酵母文库SC1C1G1T1在流式细胞测量质量方面的平均遗憾为0.113，而在生长负担方面的平均遗憾为0.801，且每一个被评估的“文库—电路规模”组合都具有正的负担遗憾。拓扑验证将所有有效的门分配等同视之，而

    arXiv:2611.22263v1 Announce Type: new  Abstract: RLVF assumes the verifier evaluates the quantity of interest. In genetic-circuit design, a topological verifier can establish that a program is a valid circuit computing the target function, whereas ranking its physical performance requires a simulator. We measure fidelity regret, the reference simple regret incurred when a cheap simulator selects the design, over a 21,362-design combination subspace spanning 5 measured gate libraries and 3 organisms. Exact ordered-assignment checks retain positive regret in every tested cell while shifting magnitudes in both directions. Mean normalized regret is 0.369 over 72 cells and only 6 have zero regret. The discrepancy depends strongly on the reference axis: the yeast library SC1C1G1T1 has mean regret 0.113 for cytometry quality and 0.801 for growth burden, and every evaluated library--circuit-size pair has positive burden regret. Topological verification ties all valid gate assignments, while an
    
[^145]: OfficePrism：面向企业业务职能单域操作的智能体基准测试

    OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions

    [https://arxiv.org/abs/2611.22262](https://arxiv.org/abs/2611.22262)

    该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。

    

    宽泛的职场能力评分可能将单一业务职能内的胜任能力与跨职能、跨系统协调的额外成本混为一谈，导致失败难以定位。OfficePrism 通过 420 个涵盖七大业务职能的单域企业任务，在统一的、由基准定义的 L1–L3 任务结构层级下评估这一前提。每个任务都在一个全新的环境中运行，该环境由一个共享的合成组织与一个特定职能领域及类型化状态访问组合而成。我们的流水线利用有据可依的记录、业务规则和工具模式，联合构建中文工单、隐藏的可执行参考见证以及预期响应，随后进行自动过滤和人工筛选。同一任务集通过完整模式工具包和搜索-分发工具包两种方式进行评估。在 124 种模型-框架-工具包配置中，每个案例的自然语言与数据库联合通过率介于 22.6% 至 76.4% 之间。

    arXiv:2611.22262v1 Announce Type: cross  Abstract: Broad workplace scores can mix competence within one business function with the extra cost of coordinating across functions and systems, making failures hard to localize. OfficePrism evaluates this prerequisite through 420 single-domain enterprise tasks spanning seven business functions under shared, benchmark-defined L1--L3 task-structure levels. Each task runs in a fresh environment combining a shared synthetic organization with one function-specific domain and yped state access. Our pipeline uses grounded records, business rules, and tool schemas to jointly construct a Chinese-language ticket, a hidden executable reference witness, and an expected response, followed by automated filtering and human curation. The same task set is evaluated through full-schema and search-and-dispatch packages. Across 124 model--harness--package configurations, the per-case joint natural-language-and-database pass rate ranges from 22.6% to 76.4%, with 
    
[^146]: 基于多模态大语言模型的扫描式认知与行为评估视觉信息提取系统

    A Visual Information Extraction System Using Multimodal Large Language Models for Scanned Cognitive and Behavioral Assessments

    [https://arxiv.org/abs/2611.22252](https://arxiv.org/abs/2611.22252)

    该研究提出了一个基于多模态大语言模型的模块化信息提取系统，通过表单分类器、分数提取器以及双模型验证的人机协同设计，从扫描的手写认知与行为评估表中高效提取字段和分数，认知评估端到端准确率达88%。

    

    目标。纸质评估在患者护理中十分常见，但手写字迹和扫描伪影使这些表格难以大规模处理。本研究提出了一种模块化、基于多模态大语言模型（LLM）并带有人工监督的系统，用于提取字段和分数。我们对使用大语言模型进行手绘分析进行了定性研究。材料与方法。该系统通过两个模块处理认知评估（即MMSE、MoCA、Mini-Cog）和NICHQ Vanderbilt评估：1）表单分类器，将扫描页面归类为相应的表单类型；2）对应的分数提取器，将字段和分数提取到预定义的模式（schema）中。系统采用基于双模型验证的人机协同（human-in-the-loop）设计，仅当两个大语言模型的输出不一致时才引入人工标注员，从而将人工审核限定于具有挑战性的页面。结果。认知评估的端到端准确率为88%，大多数表单类型的准确率超过90%。对于NICHQ Vanderbilt评估，整体……（原文摘要在此处截断）

    arXiv:2611.22252v1 Announce Type: new  Abstract: Objective. Paper-based assessments are common in patient care, but handwriting and scanning artifacts make these forms difficult to process at scale. This study proposes a modular, multimodal LLM-based system with human oversight for extracting fields and scores. We conducted a qualitative analysis of the use of LLMs for hand-drawing analysis.   Materials and Methods. The system processes cognitive assessments (i.e., MMSE, MoCA, Mini-Cog) and the NICHQ Vanderbilt Assessment using two modules: 1) a form classifier that assigns a scanned page to a form type, followed by a corresponding 2) score extractor that extracts fields and scores into a schema. A human-in-the-loop design utilizing dual-model validation engages human annotators when two LLMs disagree, limiting manual review to challenging pages only.   Results. End-to-end accuracy for cognitive assessments was 88%, with most form types exceeding 90%. For the NICHQ Vanderbilt, overall 
    
[^147]: 面向复杂平台跳跃游戏的PCGRL：通过膨胀包络与前向课程合成可玩且多样化的关卡

    PCGRL for Complex Platformers: Synthesizing Playable and Diverse Levels via Inflated Envelopes and Forward Curricula

    [https://arxiv.org/abs/2611.22250](https://arxiv.org/abs/2611.22250)

    该论文提出一个物理感知、课程引导的PCGRL框架，通过“膨胀体积包络”机制将连续运动学可行性融入离散状态空间，并引入中间状态转移反馈来缓解稀疏奖励问题，从而在复杂平台跳跃游戏的大规模关卡生成中保证关卡的可玩性与多样性。

    

    通过强化学习进行程序化内容生成（PCGRL）已成为自动化游戏开发中一种有前景的范式。然而，现有的PCGRL框架在面对由严格刚体规则和非线性运动学边界（如平台跳跃约束）所支配的长序列生成任务时，仍然存在严重的探索瓶颈和扩展性限制。因此，传统方法经常无法在大规模画布上保持可靠的关卡可解性。为了解决这些挑战，我们提出了一个物理感知、课程引导的PCGRL框架。首先，我们构建了一个“膨胀体积包络”机制，它提供了一种物理约束的表示方法，将连续运动学可行性融入马尔可夫决策过程（MDP）的离散状态空间中。其次，为了缓解由极长决策序列引起的稀疏奖励挑战，我们引入了中间状态转移反馈（摘要在此处被截断）……

    arXiv:2611.22250v1 Announce Type: cross  Abstract: Procedural Content Generation via Reinforcement Learning (PCGRL) has emerged as a promising paradigm for automated game development. However, existing PCGRL frameworks still suffer from severe exploration bottlenecks and scaling limitations when faced with long-sequence generation tasks governed by strict rigid-body rules and non-linear kinematic boundaries (e.g., platformer constraints). Consequently, traditional approaches frequently fail to maintain reliable level solvability on large-scale canvases. To address these challenges, we present a physics-aware, curriculum-guided PCGRL framework. First, we construct an "Inflated Volume Envelope" mechanism, which provides a physics-constrained representation that incorporates continuous kinematic feasibility into the discrete state space of the MDP. Second, to alleviate the sparse reward challenge induced by extremely long decision sequences, we introduce intermediate state-transition feed
    
[^148]: OAK-Mind：基于知识驱动的多智能体大语言模型实现零训练的被动感知心理健康变化推断

    OAK-Mind: Zero-Training Mental Health Change Inference from Passive Sensing via Knowledge-Grounded Multi-Agent LLMs

    [https://arxiv.org/abs/2611.22248](https://arxiv.org/abs/2611.22248)

    OAK-Mind提出了一种零训练、多智能体、知识驱动的LLM流水线，无需标注数据或任务特定微调，通过基于心理健康文献解读睡眠、活动、移动等多维被动感知数据来推断心理健康变化。

    

    可穿戴设备和智能手机传感器能够捕捉与心理健康相关的行为信号，然而从这些信号推断心理健康结果传统上依赖于需要标注数据的监督式模型，且这些模型对其预测所提供的洞察十分有限。大语言模型（LLM）为基于行为证据进行推理提供了新的机会，但现有的基于LLM的被动感知方法在很大程度上仍依赖于任务特定的微调或传感器特定的预训练。在本文中，我们提出了OAK-Mind，这是一个零训练、多智能体、知识驱动的LLM流水线，用于从被动感知数据中推断心理健康变化。该流水线将感知数据组织成多个行为维度，包括睡眠、身体活动、位置移动和智能手机使用，并协调多个以心理健康文献为知识基础的专业智能体来解读纵向行为数据。

    arXiv:2611.22248v1 Announce Type: cross  Abstract: Wearable and smartphone sensors capture behavioral signals associated with mental health, yet inferring mental health outcomes from these signals has traditionally relied on supervised models requiring labeled data and offering limited insight into their predictions. Large language models (LLMs) offer new opportunities to reason over behavioral evidence, but existing LLM-based approaches to passive sensing have largely continued to rely on task-specific fine-tuning or sensor-specific pretraining. In this paper, we introduce \textbf{OAK-Mind}, a zer\textbf{O}-training, multi-\textbf{A}gent, \textbf{K}nowledge-grounded LLM pipeline for inferring mental health changes from passive sensing data. The pipeline organizes sensing data into behavioral dimensions including sleep, physical activity, location mobility, and smartphone use, and orchestrates specialized agents grounded in mental health literature to interpret longitudinal behavioral 
    
[^149]: 多智能体记忆不一致性：LLM智能体中事实冲突消失之处

    Multi-Agent Memory Inconsistency: Where Factual Conflicts Disappear in LLM Agents

    [https://arxiv.org/abs/2611.22245](https://arxiv.org/abs/2611.22245)

    该论文提出了一个基准测试与证据流分析器来研究多智能体LLM系统中的记忆不一致问题，发现委托虽然提高了冲突事实的检索率（50.2%→68.7%），却反而降低了冲突检测率（16.2%→13.2%），且当冲突信息分散在不同工作智能体的独立上下文中时检测效果最差。

    

    多智能体委托使LLM系统能够将证据收集任务分配给拥有各自独立上下文的多个工作智能体。这带来了一致性挑战：工作智能体检索到的事实可能无法出现在主导智能体用于形成联合结论的证据之中。我们引入了一个基准测试和证据流分析器，通过在Harvey LAB文档审查任务中植入受控的事实冲突来研究多智能体记忆不一致性。在三个模型上汇总、生成开销相当的情况下，相比单智能体执行，要求委托使两条冲突声明的检索率从50.2%提升到68.7%，然而冲突检测率却从16.2%下降到13.2%。在此设置中，工作智能体仅在35.2%的检索到的冲突对中同时报告两条声明；汇总的匹配比较显示，当同一工作智能体读到两条冲突声明时，检测率高于单智能体执行，而当冲突声明分散在不同工作智能体之间时，检测率则低于单智能体执行。（摘要在此处截断）

    arXiv:2611.22245v1 Announce Type: new  Abstract: Multi-agent delegation lets LLM systems distribute evidence collection across workers with separate contexts. This creates a consistency challenge: facts retrieved by workers may be missing from the evidence available to the lead for a joint conclusion. We introduce a benchmark and evidence-flow profiler for studying multi-agent memory inconsistency through controlled factual conflicts in Harvey LAB document-review tasks. Pooled across three models with comparable generation spending, requiring delegation increases retrieval of both conflicting claims from 50.2% to 68.7% relative to single-agent execution, yet conflict detection falls from 16.2% to 13.2%. Within this setting, workers report both claims for only 35.2% of retrieved pairs, and pooled matched comparisons show higher detection than single-agent execution when one worker reads both claims but lower detection when the claims are split across workers. Worker-to-lead handoffs are
    
[^150]: 无求解器且成本感知的AI数据中心任务在线调度

    Solver-Free and Cost-Aware Online Scheduling of AI Data Center Tasks

    [https://arxiv.org/abs/2611.22239](https://arxiv.org/abs/2611.22239)

    本文提出一种受对偶分解启发的无求解器在线调度规则，在任务到达时即为其分配开始时段，在满足运行约束和高优先级任务按期执行的同时将灵活任务转移至低电价时段，从而以更低成本和更少计算时间优化AI数据中心的能源开销。

    

    AI数据中心持续不断地接收计算任务，而基于求解器的优化需要时间来运行，且电费按结算周期计价。本文通过一种受对偶分解启发的无求解器在线优化规则来弥合这一时间差。该规则在任务到达时为其分配一个开始周期，以便在满足运行约束的前提下优化能源成本，将灵活任务转移至低电价时段，同时保持高优先级任务按计划进行。在A100 GPU运行轨迹（3,456块GPU、1,343个任务）上的仿真表明，所提出的规则在将GPU和CPU使用率保持在容量范围内的同时，能够以低成本调度任务。与多时间尺度优化基准相比，该方法实现了更低的成本，并且由于无需求解器，计算时间大幅减少。

    arXiv:2611.22239v1 Announce Type: cross  Abstract: AI data centers receive computing tasks continuously, whereas solver-based optimization takes time to run and electricity is priced per settlement period. This paper bridges this timing gap through a solver-free online optimization rule inspired by dual decomposition. The rule assigns each task a start period upon arrival to optimize energy cost subject to operational constraints, shifting flexible tasks toward low-price periods while keeping high-priority tasks on schedule. Simulations on the A100 GPU trace (3,456 GPUs, 1,343 tasks) show that the proposed rule keeps GPU and CPU use within capacity while scheduling tasks at low cost. Compared with a multi-timescale optimization benchmark, it achieves a lower cost and, because it needs no solver, much less computation time.
    
[^151]: 上下文有代价：前缀缓存下为LLM编程智能体添加与删除上下文

    Context Has a Price: Adding and Removing Context for LLM Coding Agents under Prefix Caching

    [https://arxiv.org/abs/2611.22238](https://arxiv.org/abs/2611.22238)

    本文通过超过2万次实测发现，在前缀缓存定价机制下，为LLM编程智能体添加检索代码或删除旧观察结果并不必然省钱——同一策略在不同缓存折扣下可能节省19–36%也可能增加38–116%的成本，且更短的提示词并不能保证更便宜的运行。

    

    编程智能体会将其对话历史反复发送给语言模型。添加有用的代码可能减少搜索工作，而删除旧的工具输出可能缩短后续提示词。但这些改变是否真的能降低解决任务的成本？我们在SWE-bench Verified、SWE-bench和Loc-Bench上对开源模型进行了超过20,000次运行，将此作为一个测量问题加以研究。我们分别统计未缓存的输入token、缓存的输入token和输出token。研究得出三个发现。第一，在我们主要的证据包实验中，检索到的代码将读取和搜索类命令减少了18–34%，但当缓存输入价格为未缓存输入的十分之一时，总成本反而增加了33–42%，且任务解决率没有可测量的提升。第二，删除旧的观察结果既可能省钱也可能增加成本：同样的方法在没有缓存折扣时可节省19–36%的成本，但在十分之一价格的缓存下却使成本增加38–116%。第三，更短的提示词并不能保证更便宜的运行，因为智能体……

    arXiv:2611.22238v1 Announce Type: new  Abstract: Coding agents repeatedly send their conversation history to a language model. Adding useful code may reduce search, while removing old tool outputs may shorten later prompts. But do these changes reduce the cost of solving a task? We study this as a measurement problem in more than 20,000 runs of open models on SWE-bench Verified, SWE-bench and Loc-Bench. We account separately for uncached input, cached input and output tokens. Three findings emerge. First, in our main evidence-package experiments, retrieved code reduces read and search commands by 18--34%, but increases cost by 33--42% when cached input costs one tenth of uncached input, without a measurable improvement in resolution. Second, removing old observations can either save money or increase cost: the same approach saves 19--36% without a cache discount but increases cost by 38--116% with a tenth-price cache. Third, shorter prompts do not guarantee cheaper runs because agents 
    
[^152]: 语言模型剪枝中推理重要性归因的脆弱性

    The Fragility of Reasoning-Importance Attribution for Pruning Language Models

    [https://arxiv.org/abs/2611.22234](https://arxiv.org/abs/2611.22234)

    该论文揭示了推理语言模型剪枝所依赖的“推理重要性归因”度量是脆弱的——对最终答案token打分因饱和而失效，会导致剪枝后准确率大幅下降，动摇了归因引导剪枝方法的基础假设。

    

    基于归因引导的剪枝方法通过评估各组件的因果贡献来为其打分，并保护其中重要的组件，该方法正日益被用于压缩推理语言模型而不破坏其思维链。其可靠性建立在一个未经检验的假设之上：即推理重要性能够被忠实地度量。在四个小型推理模型上，我们证明了这种度量在三个方面是脆弱的，而每一方面都会改变剪枝决策。第一，对最终答案token打分——这是最自然的选择——在我们测试的每一个思维链模型中都会饱和（答案token的负对数似然NLL仅为10^{-4}–10^{-3}），因为模型会在自身的推理过程中陈述答案；在我们以五个随机种子进行功效分析的两个模型上，以该目标引导注意力头剪枝，相比对推理token打分会损失27.6和32.2个准确率百分点，而困惑度却几乎不变，甚至在7B模型上符号反转。第二，因果重要性无法被定位到……（摘要原文在此处截断）

    arXiv:2611.22234v1 Announce Type: new  Abstract: Attribution-guided pruning scores components by their causal contribution and protects the important ones, and it is increasingly applied to compress reasoning language models without destroying their chain of thought. Its reliability rests on an unexamined assumption: that reasoning importance can be faithfully measured. Across four small reasoning models, we show this measurement is fragile in three ways, each of which changes pruning decisions. First, scoring final-answer tokens, the natural choice, saturates in every chain-of-thought model we test (answer-token negative log-likelihood, NLL, of $10^{-4}$--$10^{-3}$) because the model states its answer inside its own reasoning; guiding head pruning with this target costs 27.6 and 32.2 accuracy points relative to scoring reasoning tokens on the two models we power with five seeds, while perplexity barely moves and even reverses sign at 7B. Second, causal importance is not localizable to
    
[^153]: 细粒度日语临床命名实体识别：BERT与开放权重大型语言模型的受控比较

    Fine-Grained Japanese Clinical Named Entity Recognition: A Controlled Comparison of BERT and Open-Weight Large Language Models

    [https://arxiv.org/abs/2611.22219](https://arxiv.org/abs/2611.22219)

    在包含113个实体类别的细粒度日语临床命名实体识别任务中，微调的BERT-CRF基线模型（F1达0.836-0.841）显著优于开放权重大型语言模型（零样本最高F1仅0.337）。

    

    背景：生成式大型语言模型（LLM）为基于编码器的临床命名实体识别（NER）提供了替代方案，但其在细粒度日语抽取任务上的性能以及对生成设置的敏感性仍不确定。方法：我们在定义了113个实体类别的iCorpus语料库上，将两个经过微调的BERT-CRF基线模型与九个开放权重的大型语言模型进行了比较。评估使用了来自36个测试文档的433个句子；1,672个句子用于训练。大型语言模型通过有无JSON Schema约束的零样本提示以及三次随机种子的QLoRA微调进行评估。微调后的模型在不同的输出token限制下进行评估，包括900以及更大的模型特定限制（2,600或4,096个token）。性能通过句子级标签-表层匹配进行测量，采用宽松表层F1和精确表层F1。结果：零样本设置下最高的宽松表层F1为0.337，而BERT基线模型达到0.836-0.841。

    arXiv:2611.22219v1 Announce Type: new  Abstract: Background: Generative large language models (LLMs) offer an alternative to encoder-based clinical named entity recognition (NER), but their performance on fine-grained Japanese extraction tasks and sensitivity to generation settings remain uncertain. Methods: We compared two fine-tuned BERT-CRF baselines with nine open-weight LLMs on iCorpus, which defines 113 entity categories. The evaluation used 433 sentences from 36 test documents; 1,672 sentences were used for training. LLMs were evaluated using zero-shot prompting with and without JSON Schema constraints and three-seed QLoRA fine-tuning. Fine-tuned models were assessed under different output-token limits, including 900 and larger model-specific limits of 2,600 or 4,096 tokens. Performance was measured using sentence-level label-surface matching with relaxed and exact-surface F1. Results: The highest zero-shot relaxed F1 was 0.337, compared with 0.836-0.841 for the BERT baselines. 
    
[^154]: 通过自回归标记回归学习最优传输方案

    Learning Optimal Transport Plans Via Autoregressive Token Regression

    [https://arxiv.org/abs/2611.22218](https://arxiv.org/abs/2611.22218)

    该论文提出ToR-OT，一种基于Transformer的自回归标记回归模型，将最优传输问题重构为结构化序列生成任务，直接解码稀疏传输方案而无需在测试时求解优化问题，并展现出零样本泛化到未见分布的能力。

    

    最优传输为寻找概率分布之间的最小成本耦合提供了一个有原则性的框架。经典求解器仍然是该任务的黄金标准。与此同时，OT是研究基于学习的摊销模型能否内化OT类问题结构的理想测试平台。在这一问题中，真值解是可获得的，边际约束被显式定义，且精确的离散解具有充分刻画的稀疏性。我们提出了ToR-OT，一个基于Transformer的模型，通过自回归标记回归学习预测离散OT方案。我们将OT重新构建为一个结构化序列生成任务，ToR-OT将稀疏传输方案解码为序列，而不是在测试时求解优化问题。我们关注三个性质：(a) 零样本泛化：在大型合成OT实例语料库上进行预训练，使单个模型能够泛化到未见过的分布……

    arXiv:2611.22218v1 Announce Type: cross  Abstract: Optimal Transport (OT) provides a principled framework for finding the minimum cost coupling between probability distributions. Classical solvers remain the gold standard for this task. At the same time, OT is an ideal testbed for studying whether learning-based amortized models can internalize the structure of OT type problems. Here, ground-truth solutions are available and marginal constraints are explicitly defined. Also, exact discrete solutions have well-characterized sparsity. We describe ToR-OT, a transformer-based model that learns to predict discrete OT plans through autoregressive token regression. We reframe OT into a structured sequence generation task. ToR-OT decodes sparse transport plans as sequences instead of solving an optimization task at test time. We focus on three properties: (a) Zero-shot generalization: pretraining on a large synthetic corpus of OT instances allows a single model to generalize to unseen distribu
    
[^155]: 传递变化：通过激活压缩实现高效专家并行

    Communicate the Change: Efficient Expert Parallelism via Activation Compression

    [https://arxiv.org/abs/2611.22215](https://arxiv.org/abs/2611.22215)

    EchoMoE 利用 transformer 残差流以增量方式更新的特性，通过激活压缩只传递变化量而非完整隐藏状态，从而大幅降低专家并行中的同步集合通信瓶颈，且改造现有 MoE 模型仅需约完整预训练 1/8500 的训练 token。

    

    规模化扩展推动了大语言模型（LLM）的成功，稀疏混合专家架构使模型能够获得更大的参数容量，而无需成比例地增加计算量。在大规模场景下，MoE 模型依赖专家并行将专家分布到多个设备上。每个 MoE 层在推理过程中需要两个同步的集合通信步骤：将隐藏状态分发到被激活的专家，以及合并各专家的输出结果。随着计算吞吐量的增长速度超过互连带宽的增长，这些同步集合通信已成为 MoE 扩展与效率的主要瓶颈。我们提出 EchoMoE 来应对这一挑战：这是一种轻量级的架构修改，可以改造现有的 MoE 模型，所需训练 token 数量约为完整预训练的 1/8500。受 transformer 块以增量方式更新残差流这一观察的启发，我们利用较早的隐藏状态……（原文摘要在此处被截断）

    arXiv:2611.22215v1 Announce Type: cross  Abstract: Scaling has driven the success of large language models (LLM), with sparse mixture-of-experts (MoE) architectures enabling greater parameter capacity without a proportional increase in computation. At large scales, MoE models rely on expert parallelism to distribute experts across devices. Each MoE layer requires two synchronous collective communication steps during inference: dispatching hidden states to the activated experts and combining the resulting expert outputs. As compute throughput has outpaced interconnect bandwidth, these synchronous collectives have become a major bottleneck to MoE scaling and efficiency. We address this challenge with EchoMoE, a lightweight architecture modification that retrofits existing MoE models, requiring approximately 8,500x fewer training tokens than full pretraining. Motivated by the observation that transformer blocks update the residual stream incrementally, we use a hidden state from an earlie
    
[^156]: EGS：面向教育型大语言模型的补充性安全框架

    EGS: A Complementary Safety Framework for Educational LLMs

    [https://arxiv.org/abs/2611.22203](https://arxiv.org/abs/2611.22203)

    本文提出教育护栏系统EGS，一个轻量级、与模型无关的运行时安全框架，通过风险分级分流、情境化教学引导和输出审计，超越简单的二元拒绝模式，有效提升面向青少年的教育大语言模型的安全性。

    

    大语言模型（LLM）在个性化教育方面具有巨大潜力，但面向青少年的部署带来了有毒、偏见、误导性或发展不适宜内容的风险。我们认为，教育安全应当超越简单的二元拒绝模式：学生的提问往往模糊不清或出于好奇心驱动，需要情境化的引导而非简单的拦截。我们提出了教育护栏系统（EGS），这是一个轻量级、与模型无关的运行时框架，通过查询审核、情境感知的教学引导和响应审核来补充模型内在的对齐能力。EGS 按风险等级对输入进行分流，通过针对特定类别的教育干预将灰色地带的查询进行重定向，并在输出交付前对其进行审计。在通用基准和教育专用基准上的实验表明，EGS 能够在多种大语言模型上提升审核性能、拒绝质量以及越狱攻击的鲁棒性。

    arXiv:2611.22203v1 Announce Type: cross  Abstract: Large Language Models (LLMs) offer strong potential for personalized education, but adolescent-facing deployment raises risks of toxic, biased, misleading, or developmentally inappropriate content. We argue that educational safety should move beyond binary refusal: student queries are often ambiguous or curiosity-driven, requiring contextual guidance rather than simple blocking. We introduce the Educational Guardrail System (EGS), a lightweight, model-agnostic runtime framework that complements intrinsic alignment through query moderation, scenario-aware pedagogical guidance, and response moderation. EGS triages inputs by risk level, redirects gray-zone queries with category-specific educational interventions, and audits outputs before delivery. Experiments across general and education-specific benchmarks show that EGS improves moderation performance, refusal quality, and jailbreak robustness across diverse LLMs. These findings suggest
    
[^157]: 从未完成体悖论到摩尔式荒谬：体貌与语用歧义下的单链推理锁定与多候选言语化

    From Imperfective Paradox to Moorean Absurdity: Single-Chain Reasoning Lock-In and Multi-Candidate Verbalization Across Aspectual and Pragmatic Ambiguity

    [https://arxiv.org/abs/2611.22188](https://arxiv.org/abs/2611.22188)

    该论文通过四项预注册的形式化实验证实了大语言模型在体貌与语用歧义推理中存在“单链过早锁定”现象，并提出多候选言语化（MCV）方法，在单次调用中生成多个加权候选估计以缓解该锁定问题。

    

    自然语言的蕴含与连贯性判断往往取决于一个事件是否仅仅发生了、是否达成了终结性的完成点、是否满足了某个计划，抑或是在多种语用标准下被评估。我们通过四项预注册、密封形式的实验（在 Qwen3-8B-Q8_0 模型上进行，共 16,640 个形式单元，temperature=0.0）检验了“单链过早锁定假说”的可观察结果：16 个假设槽位中有 15 个保持验证性地位（其中 12 个在实验内 Holm 校正下于 alpha_fam=0.05 水平拒绝 H0；E3_H3 降级为描述性地位）。在实验 1 中，Direct-CoT 与 MCU 在终结性高置信度完成上相差 25.00 个百分点（Holm p=1.31239e-05），而 NLI-CoT 在无中断的非终结性项目上崩溃了 -91.25 个百分点（Holm p=1.81805e-21）。多候选言语化（MCV）在单次调用中引出 K=5 个加权估计，将提示与未加权（MCU）和自加权（MCW）读出方式分离。普通话（实验 2）与英语（实验 3）……（原文摘要至此截断）

    arXiv:2611.22188v1 Announce Type: cross  Abstract: Natural language entailment and coherence judgments often depend on whether an event merely occurred, reached a telic culmination, satisfied a plan, or was evaluated under multiple pragmatic standards. We test observable consequences of the single-chain premature lock-in hypothesis across four preregistered, sealed formal experiments on Qwen3-8B-Q8_0 (16,640 formal units, temperature=0.0): 15 of 16 hypothesis slots retain confirmatory status (12 rejecting H0 under within-experiment Holm control at alpha_fam=0.05; E3_H3 demoted to descriptive status). In Experiment 1, Direct-CoT vs. MCU differs by 25.00 pp on telic high-confidence completion (Holm p=1.31239e-05), while NLI-CoT collapses by -91.25 pp on uninterrupted atelic items (Holm p=1.81805e-21). Multi-Candidate Verbalization (MCV) elicits K=5 weighted estimates in one call, separating the prompt from unweighted (MCU) and self-weighted (MCW) readouts. Mandarin (E2) and English (E3) 
    
[^158]: CueClash：前沿视觉语言模型依据贴近图像平面的线索判断三维关系

    CueClash: Frontier Vision-Language Models Judge 3-D Relations Close to the Picture Plane

    [https://arxiv.org/abs/2611.22180](https://arxiv.org/abs/2611.22180)

    研究提出 CueClash 基准，揭示前沿视觉语言模型在判断三维关系时严重依赖二维图像捷径线索而非真实空间布局（捷径差距高达 +18 至 +37 个百分点），其失败源于从画面到地面布局的推理步骤。

    

    视觉语言模型（VLM）可以从图像画面或从地面布局来判断三维关系。在真实照片中，两者通常是一致的。我们通过 CueClash 将二者分离——这是一组渲染场景，其中五种关系各自的二维线索恰好在一半的陈述上是正确的。捷径差距（shortcut gap）定义为线索正确时的准确率减去线索错误时的准确率。在低推理强度下，Claude Opus 5.5、GPT-5.5 和 Gemini 3.6 Flash 的捷径差距为 +18 至 +37 个百分点。若剔除两种其线索错误条目与临界情形或遮挡相混淆的关系，该差距为 +7 至 +29 个百分点。若给出地面坐标和各关系的定义，它们的回答正确率可达 99% 到 100%。对于能够穿过开口的物体，结合相机深度的像素测量可消除大部分差距。失败发生在从画面到布局的转换这一步：模型从上方俯视绘制的场景地图大多保留了深度顺序，但与画面一样缩短了深度。

    arXiv:2611.22180v1 Announce Type: cross  Abstract: A vision-language model (VLM) can judge 3-D relations from the picture or from the ground layout. In photographs the two usually agree. We separate them with CueClash, rendered scenes in which the 2-D cue for each of five relations is right on exactly half the statements. The shortcut gap is accuracy when the cue is right minus accuracy when it is wrong. For Claude Opus 5.5, GPT-5.5 and Gemini 3.6 Flash at low reasoning effort it is +18 to +37 percentage points. It is +7 to +29 without two relations whose cue-wrong items are confounded with borderline cases or occlusion. Given ground coordinates and each relation's definition, they answer 99 to 100% correctly. Pixel measurements with camera depths remove most of the gap for objects fitting through openings. The failure lies in the step from the picture to the layout. The maps that the models draw of the scene from above mostly keep the depth order but, like the picture, shorten depth. 
    
[^159]: 穿着风衣的RAG：极简记忆何时足以支撑智能体系统，何时又不够

    RAG in a Trenchcoat: When Minimal Memory Is Enough for Agentic Systems, and When It Isn't

    [https://arxiv.org/abs/2611.22179](https://arxiv.org/abs/2611.22179)

    论文提出刻意极简的RAG记忆系统BRIDGE，仅凭链式观察笔记、混合检索与时间排序渲染，就以极低成本（每次查询约5-7K token、单次LLM调用）在标准记忆基准上追平乃至超越复杂SOTA方法，进而论证标准基准只衡量记忆“检索到了什么”而非“能为智能体行动赋能什么”，并为此引入五个对抗性探测任务。

    

    智能体记忆研究在检索增强生成（RAG）之上累积了抽取、取代更新、反思与图谱合成等原语。然而，我们发现这些原语在我们测试的记忆基准中几乎都无法证明其成本是值得的：我们用BRIDGE展示了这一点——BRIDGE是一个刻意极简的RAG记忆系统，采用链式观察笔记、BM25与余弦相似度混合检索以及按时间顺序渲染，在LongMemEval-S和BEAM上与此前SOTA在bootstrap置信区间内持平，并在读取器相同的条件下以5.7-6.3分的优势超越最佳记忆基线（EverMemBench）。它每次查询仅消耗约5-7K读取器提示token，且每次写入或查询只需一次LLM调用。我们主张这种持平本身就是结果：标准基准衡量的是记忆检索到了什么，而非当智能体行动时记忆能实现什么。为了检验后者，我们引入了五个对抗性探测任务（规则应用±、模式归纳风格/程序性、动态多任务……（原文摘要在此处截断）

    arXiv:2611.22179v1 Announce Type: new  Abstract: Agentic memory research has accumulated extraction, supersession, reflection, and graph synthesis on top of retrieval-augmented generation. However, we found that almost none of those primitives earn their cost in our tested memory benchmarks: we show this with BRIDGE, a deliberately minimal RAG memory with chained observation notes, hybrid BM25 with cosine retrieval, and chronological rendering, which reaches parity within bootstrap CI with the prior SOTA on LongMemEval-S and BEAM, and beats the best memory baseline on EverMemBench by 5.7-6.3 points at matched reader. It runs at ~5-7K reader-prompt tokens per query, using a single LLM call for each ingestion or query. We argue that this parity is itself the result: standard benchmarks score what memory retrieves, not what memory enables when the agent acts. To test the latter, we introduce five adversarial probes (Rule Application +-, Pattern Induction Style / Procedural, Dynamic Multit
    
[^160]: CorrectCast：面向降水临近预报中尺度感知残差校正的选择性多模态上下文

    CorrectCast: Selective Multimodal Context for Scale-Aware Residual Correction in Precipitation Nowcasting

    [https://arxiv.org/abs/2611.22178](https://arxiv.org/abs/2611.22178)

    CorrectCast提出了一种解耦的残差校正框架，在冻结基础临近预报模型的前提下仅训练轻量级校正器，利用选择性多模态上下文（QPE与PWV观测）和尺度感知调制，以统一的接口校正不同临近预报模型的强度、位置和空间结构误差。

    

    降水临近预报模型在强度、位置和空间结构方面存在误差。通过重新训练临近预报主干网络来解决这些误差可能计算成本高昂，且需要针对特定架构的修改。为了克服这些限制，我们提出了CorrectCast，这是一个解耦的残差校正框架，能够在保持基础临近预报模型冻结的同时对已生成的临近预报结果进行精细化改进。通过在预测的降水场上进行操作，CorrectCast只需训练一个轻量级的校正器，从而降低了计算开销，同时为不同的临近预报模型提供统一的校正接口。以基础临近预报以及历史定量降水估计（QPE）和可降水量（PWV）观测为条件，校正器选择多模态上下文并采用尺度感知调制来估计残差校正。在密西西比河流域（MRB）和美国本土（CO

    arXiv:2611.22178v1 Announce Type: new  Abstract: Precipitation nowcasting models suffer from errors in intensity, location, and spatial structure. Addressing these errors by retraining nowcaster backbones can be computationally expensive and requires architecture-specific modifications. To overcome these limitations, we propose CorrectCast, a decoupled residual correction framework that refines completed nowcasts while keeping the base nowcaster frozen. By operating on predicted precipitation fields, CorrectCast requires training only a lightweight corrector, reducing computational overhead while providing a unified correction interface across different nowcasting models. Conditioned on the base nowcast and historical quantitative precipitation estimation (QPE) and precipitable water vapor (PWV) observations, the corrector selects multimodal context and employs scale-aware modulation to estimate residual corrections. On the Mississippi River Basin (MRB) and contiguous United States (CO
    
[^161]: 唯一的优势来自训练中的记忆：财报与坏消息事件前后的LLM交易代理

    The Only Edge Was Memorized in Training: LLM Trading Agents Around Earnings and Bad News

    [https://arxiv.org/abs/2611.22177](https://arxiv.org/abs/2611.22177)

    本文在无未来信息泄露的同一基准上，测试了LLM交易代理在财报公告前与坏消息暴跌后的交易能力，发现其唯一的超额收益并非来自真实预测能力，而是来自训练时记住的历史信息。

    

    大语言模型（LLM）交易代理通常在新闻上进行测试，但新闻会在几分钟内就反映到股价中，一个按发布时间回放新闻标题的基准测试发现，据此交易并不存在超额收益。因此，资金可能只存在于两个地方：一是在日期已知的新闻之前（如财报公告），二是坏消息之后（如果市场跌幅过度）。然而，现有关于LLM与财报的研究阅读的是随公告一起或之后发布的文档，而反转策略的研究则按月构建投资组合，因此两者都无法说明代理在扣除交易成本后能否在这些关键时刻获利。本文中，我们在同一个无未来信息泄露的基准上测试了LLM代理在这两个时刻的表现：分别涵盖598次财报公告之前和409次美国大盘股坏消息暴跌之后。每个输入都在决策时点的收盘处截断，将分阶段执行的代理与单次调用、简单规则和逻辑回归进行比较，且每个决策都在有股票名称和无股票名称两种条件下作出。（摘要原文在此处被截断）

    arXiv:2611.22177v1 Announce Type: new  Abstract: Large language model (LLM) trading agents are tested on news, but news reaches stock prices within minutes, and a benchmark that replays headlines at their publication time finds no alpha in trading on them. Money can then remain in two places, before news whose date is known, such as earnings announcements, and after bad news, if the market's fall went too far. However, studies of LLMs and earnings read documents published with or after the announcement, and studies of reversal sort monthly portfolios, so neither tells whether an agent can trade these moments after costs. In this paper, we test LLM agents at both moments on the same look-ahead-free benchmark, before 598 earnings announcements and after 409 bad-news drops of large US stocks. Every input is cut at the decision close, staged agents are compared with single calls, simple rules and a logistic regression, and every decision is made with and without the stock's name. Across fi
    
[^162]: 面向组合式机器人技能自适应的程序性知识重构

    Reconfiguring Procedural Knowledge for Compositional Robot Skill Adaptation

    [https://arxiv.org/abs/2611.22174](https://arxiv.org/abs/2611.22174)

    提出RecPo框架，以任务无关的多视图程序性知识库表示技能，并通过组合已存储的转移序列，在稀疏指导信息下低成本地重构机器人已有技能以适应新目标和新技能接口。

    

    通用行为模型在训练过程中日益内化了广泛的程序性知识，这使得它们对新目标和技能接口的适应，与其说是学习全新的行为，不如说是重构已有的能力。我们将这种能力研究为“可重构性”，即在稀疏的指导信息下，通过低成本地重组现有的组件与关系来实现新行为的能力；其中任务规范提供高层目标，程序规范暴露一组有限的可复用因子，而执行所需的具体绑定与组合则是欠定的（未完全确定的）。我们提出了RecPo，一个通过多个程序视图以任务无关的方式表示技能、并通过组合存储的转移来具体化可执行技能序列从而实现任务特定变化的框架。RecPo维护一个由技能引发的转移所构成的程序性知识库，每个视图……（摘要在此处被截断）

    arXiv:2611.22174v1 Announce Type: new  Abstract: General-purpose behavior models increasingly internalize broad procedural knowledge during training, making their adaptation to new goals and skill interfaces less a problem of learning entirely new behaviors than of reconfiguring existing competence. We study this capability as reconfigurability, the ability to realize new behaviors by cost-efficiently restructuring existing components and relations under sparse governing information, where task specifications provide high-level goals and procedure specifications expose a limited set of reusable factors, leaving the concrete bindings and compositions required for execution underdetermined. We propose RecPo, a framework that represents skills task-agnostically through multiple procedure views and materializes executable skill sequences by composing stored transitions to realize task-specific changes. RecPo maintains a procedure knowledge base of skill-induced transitions, with each view 
    
[^163]: Readie：面向AI代理的低延迟代码沙箱

    Readie: Low-Latency Code Sandboxes for AI Agents

    [https://arxiv.org/abs/2611.22171](https://arxiv.org/abs/2611.22171)

    提出Readie系统，通过从历史执行中学习一组小型、快速初始化的启动快照集合，使其共同覆盖大部分AI代理生成的代码，从而解决现有代码沙箱启动缓慢的问题。

    

    从量化研究到个人助理，AI代理通过迭代式地生成并执行代码来完成任务。为了将代理框架与代码执行解耦，许多代理将执行工作卸载到称为代码沙箱的专用系统中。然而，现有的代码沙箱存在启动缓慢的问题。无论它们依赖自定义快照还是单一的大型镜像，根本原因都是相同的：很难构建一个紧凑且预热的快照，既能在多样化的代理生成代码之间共享，又能覆盖其所需的几乎所有工件（例如软件包、数据集）。在这项工作中，我们提出了用于低延迟代码沙箱的学习型启动快照。我们的洞察是：虽然单个快照很难同时做到紧凑并覆盖大多数工作负载，但一组快照可以同时实现这两点。每个快照都很小、初始化迅速，并覆盖一部分代理生成的代码；这些快照组合在一起则能覆盖其中的大部分。我们从历史执行数据中学习这组快照……

    arXiv:2611.22171v1 Announce Type: new  Abstract: From quantitative research to personal assistance, AI agents act by iteratively generating and executing code. To decouple the agent harness from code execution, many agents offload execution to dedicated systems called code sandboxes. However, existing code sandboxes suffer from slow startup. Whether they rely on custom snapshots or a single fat image, the root cause is the same: it is hard to build a compact, pre-warmed snapshot that can be shared across diverse agent-generated code while covering nearly all the artifacts (e.g., packages, datasets) it requires.   In this work, we propose learned startup snapshots for low-latency code sandboxes. Our insight is that while it is hard for a single snapshot to be both compact and cover most workloads, a set of snapshots can achieve both. Each snapshot is small, initializes quickly, and covers a fraction of agent-generated code; together, the set covers most of it. We learn this set from a l
    
[^164]: 并非所有证据都可采信：面向工具增强医疗智能体的证据治理

    Not All Evidence Is Admissible: Evidence Governance for Tool-Augmented Medical Agents

    [https://arxiv.org/abs/2611.22169](https://arxiv.org/abs/2611.22169)

    该研究揭示了工具增强医疗智能体中的“证据劫持”现象——医疗工具无声失效会使原本正确的答案被翻转为错误——并从V-信息理论出发提出“证据治理”框架CIEG，通过将基于内容的证据筛选决策与推理器分离，使推理器不仅“看得更少”更能“看得更好”。

    

    工具增强型医疗智能体通常将工具返回的任何内容都视为证据。我们证明这种默认做法代价高昂：医疗工具会无声地失效，而一个冻结（不可更新）的推理器无法忽略进入其上下文的内容，因此许多原本能够正确回答的案例，在加入工具输出后反而被翻转为错误答案。我们将这种现象称为“证据劫持”。V-信息理论可以解释这一现象：推理器已经失去了“忽略”的选项，因此工具输出可能携带负的可用信息，而将“保留什么”的决策移到推理器之外则可以恢复这一选项。该理论进一步表明，任何过滤带来的收益中有一部分仅仅来自于让推理器“看得更少”，而只有基于内容的筛选才能让推理器“看得更好”。基于此，我们将工具增强推理重新表述为证据治理，即把“什么算作证据”的决策与负责回答的推理器分离开来，并提出了上下文无关的证据治理方法（CIEG），这是首个……（摘要至此截断）

    arXiv:2611.22169v1 Announce Type: new  Abstract: Tool-augmented medical agents treat whatever their tools return as evidence. We show that this default is costly: medical tools fail silently, and a frozen reasoner cannot ignore what enters its context, so many cases it would have answered correctly are flipped to wrong once tool outputs are added. We call this evidence hijacking. $\mathcal{V}$-information explains the phenomenon: the reasoner has lost the option to ignore, so a tool output can carry negative usable information, and moving the decision of what to keep outside the reasoner restores that option. The theory further shows that part of any filtering gain comes merely from letting the reasoner see less, and only selection by content lets it see better. Building on this, we recast tool-augmented reasoning as evidence governance, which separates deciding what counts as evidence from the reasoner that answers, and propose context-independent evidence governance (CIEG), the first
    
[^165]: 专家构建任务上的强化学习教会了 Nemotron 3.5 Super 什么

    What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super

    [https://arxiv.org/abs/2611.22161](https://arxiv.org/abs/2611.22161)

    通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。

    

    我们对 NVIDIA 的 Nemotron 3.5 Super（总参数量 120B，激活参数 12B）进行强化学习（RL）后训练，使用五个专家构建的任务数据集，每次训练一个：真实世界软件工程、复杂约束、长程工具使用、专业图表理解和企业文档推理。每次运行都在 Tinker 上使用 CISPO 训练一个全新的 rank-32 LoRA 适配器，不设监督学习阶段，并在相同的任务、测试框架和设置下将每个适配器与基础模型进行比较。在十二个主要基准测试上（其中八个由其他团队构建，四个由 Surge AI 构建），提升幅度从 IFBench 上的 +4.5 分到 GDP_pdf 上的 +13.8 分不等（后者从 10.4% 提升至 24.2%）；SWE-bench Pro V2 从 86.9% 提升至 92.1%，Terminal-Bench 2.1 从 63.9% 提升至 70.3%，MultiChallenge 从 58.3% 提升至 66.6%，AutomationBench 从 14.8% 提升至 27.5%，Chartography 从 39.4% 提升至 50.6%。文档适配器在没有 API 工具的情况下训练，也在 AutomationBench 上获得了 +8.0 分的提升。

    arXiv:2611.22161v1 Announce Type: cross  Abstract: We post-train NVIDIA's Nemotron 3.5 Super (120B total, 12B active parameters) with reinforcement learning (RL) on five expert-built task datasets, one at a time: real-world software engineering, complex constraints, long-horizon tool use, professional chart understanding and enterprise document reasoning. Each run trains a fresh rank-32 LoRA adapter with CISPO on Tinker, with no supervised stage, and each adapter is compared with the base model on the same tasks, harness and settings. On twelve headline benchmarks, eight built by other groups and four by Surge AI, the gains range from +4.5 points on IFBench to +13.8 on GDP_pdf, which rises from 10.4% to 24.2%; SWE-bench Pro V2 rises from 86.9% to 92.1%, Terminal-Bench 2.1 from 63.9% to 70.3%, MultiChallenge from 58.3% to 66.6%, AutomationBench from 14.8% to 27.5% and Chartography from 39.4% to 50.6%. The document adapter, trained without API tools, also gains +8.0 points on AutomationB
    
[^166]: EgoHandWAM：探究手部运动在第一人称世界动作建模中的作用

    EgoHandWAM: Probing the Role of Hand Motion in Egocentric World Action Modeling

    [https://arxiv.org/abs/2611.22151](https://arxiv.org/abs/2611.22151)

    EgoHandWAM 通过在 75 万余个第一人称操作片段上联合预测三维手部轨迹和未来视觉特征，实验表明手部运动信息对未来视觉预测仅有有限提升（约1-4%的误差降低），揭示了手部运动在第一人称世界建模中作用的边界。

    

    手部运动能在多大程度上告诉我们第一人称相机接下来会看到什么？EgoHandWAM 从当前图像和近期手部状态预测三维手腕和指尖轨迹以及未来视觉特征。我们使用冻结的 DINOv2 编码器在 751,065 个 EgoDex 窗口上进行训练。在留出的开发会话上，仅动作模型相对于静止预测将两秒手腕端点误差从 60.26 毫米降至 56.48 毫米，但该模型仍会低估长距离伸展动作。添加未来特征监督只能得到几乎相同的轨迹精度。利用记录的手部历史，视觉预测头相对于直接复制当前特征可将平滑 L1 误差降低 1.32%；而跨会话打乱手部历史则会完全消除这一增益。在一个独立的四任务探测实验中，将记录的未来位移添加到观察上下文和未来跟踪有效性掩码中，可使视觉特征误差降低 3.93%，而仅使用掩码几乎没有改进。

    arXiv:2611.22151v1 Announce Type: cross  Abstract: How much does hand motion tell us about what a first-person camera will see next? EgoHandWAM forecasts three-dimensional wrist and fingertip trajectories and a future visual feature from the current image and recent hand states. We train on 751,065 EgoDex windows using a frozen DINOv2 encoder. On held-out development sessions, the action-only model reduces two-second wrist endpoint error from 60.26 to 56.48~mm relative to a stationary forecast. It still underestimates long reaches. Adding future-feature supervision gives nearly identical trajectory accuracy. With the recorded hand history, the visual head reduces smooth L1 error by 1.32\% relative to copying the current feature. Shuffling histories across sessions erases this gain. In a separate four-task probe, adding recorded future displacements to observed context and future tracking-validity masks reduces visual-feature error by 3.93\%. Masks alone give little improvement. A traje
    
[^167]: 长上下文生成是一个采样问题（当你调高温度时）

    Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat)

    [https://arxiv.org/abs/2611.22148](https://arxiv.org/abs/2611.22148)

    该论文通过大规模实验证明，长上下文生成的退化本质上是采样器问题——高温下保持候选 token 池小而稳定的采样器（如 top-nsigma）能让编程智能体等长程生成任务保持高成功率，而放任低概率 token 进入的采样器则完全崩溃。

    

    开源权重语言模型被普遍观察到生成得越长表现越差，而常见的修复尝试往往依赖更多的训练。然而在高温下，采样器——即从模型概率中挑选下一个 token 的规则——决定了长程与智能体式生成能否存活乃至蓬勃发展。我们在十个成熟基准测试以及最长一百万 token 的大海捞针检索任务上，测试了从 1B 到 122B 参数的 29 个开源权重模型，覆盖 SWE-bench Verified 编程智能体、长思维链推理、长篇写作、长结构化输出和长上下文阅读。结果表明：随着温度升高，保持候选 token 池小而稳定的采样器不会损失准确率；而允许低概率 token 进入的采样器则会彻底崩溃。在温度为 2 时，运行 top-p、top-k 或 Top-H 的编程智能体在 100 个 GitHub 问题中解决 0 个，而运行 top-nsigma 的相同智能体可解决 46 到 62 个，并赢得了两种采样器之间的每一次对比。

    arXiv:2611.22148v1 Announce Type: new  Abstract: Open-weight language models are widely observed to get worse the longer they generate. Often, attempts to fix this involve more training. At high temperature, The sampler, i.e. the rule that picks the next token out of the model's probabilities, decides whether long and agentic generation survives and/or thrives. We test 29 open-weight models from 1B to 122B parameters on ten established benchmarks plus needle retrieval out to one million tokens, covering SWE-bench Verified coding agents, long chain-of-thought reasoning, long-form writing, long structured output and long-context reading. Samplers that keep their pool of candidate tokens small and steady as temperature rises don't lose accuracy. Samplers that let in low-probability tokens fall apart. At temperature 2, coding agents running top-p, top-k or Top-H resolve 0 of 100 GitHub issues, while the same agents running top-nsigma resolve 46 to 62 and win every issue the two samplers di
    
[^168]: 深度强化学习在暖通空调（HVAC）监督控制中何时才有收益？四个直接在FMU上训练的小型Modelica案例研究

    When does deep reinforcement learning pay off in HVAC supervisory control? Four small Modelica case studies trained directly on FMUs

    [https://arxiv.org/abs/2611.22144](https://arxiv.org/abs/2611.22144)

    该论文通过四个基于Modelica模型并直接在FMU上训练SAC智能体的小型可复现HVAC案例研究，与调优的规则控制器及近最优基准进行对比，阐明了深度强化学习在HVAC监督控制中何时才真正划算。

    

    深度强化学习（DRL）常被提议用于供暖、通风与空调（HVAC）系统的监督控制，但其所报告的收益难以解读，因为所用的基准控制器往往较弱，且可达到的最优解鲜为人知。本文基于澳大利亚布里斯班的一栋办公楼开展了四个小型、可复现的案例研究：分时电价下的预冷、带冷凝器水温重置的冷水机组台数控制、带湿式冷却盘管的冷冻水供水温度重置，以及分时电价与需量电费下的冷冻水蓄冷。这些被控对象采用Modelica Buildings库建模，其中三个案例中的软演员-评论家（soft actor-critic）智能体直接在导出的功能样机单元（FMU）上进行训练。每个智能体均与经过调优的基于规则的控制器进行比较，并在可行的情况下，与通过逐时穷举搜索或动态规划获得的近最优参考基准进行比较。

    arXiv:2611.22144v1 Announce Type: cross  Abstract: Deep reinforcement learning (DRL) is often proposed for the supervisory control of heating, ventilation and air-conditioning (HVAC) systems. The reported gains are, however, difficult to interpret, because the baselines are often weak and the attainable optimum is seldom known. This paper examines four small, reproducible case studies based on an office building in Brisbane, Australia: pre-cooling under a time-of-use tariff; chiller staging with condenser water reset; chilled water supply temperature reset with a wet cooling coil; and chilled water storage under time-of-use prices and a demand charge. The plants were modelled with the Modelica Buildings library, and in three of the cases a soft actor-critic agent was trained directly on the exported functional mock-up unit (FMU). Each agent was compared with tuned rule-based controllers and, where practicable, with a near-optimal reference from hourly exhaustive search or dynamic progr
    
[^169]: RoboTeam：面向多机械臂实时视觉多智能体控制的基准测试

    RoboTeam: Benchmarking Vision-Based Multi-Agent Control of Many Robot Arms in Real Time

    [https://arxiv.org/abs/2611.22138](https://arxiv.org/abs/2611.22138)

    RoboTeam是一个仅凭相机图像实时协同控制4至16个机械臂、包含36项长时协作任务的模拟基准，揭示了自适应多智能体系统虽然完成任务数最多（28/36），但需要付出数小时耗时和上亿token的高昂代价。

    

    多模态智能体现在可以通过调用工具和编写代码来控制机械臂。现有基准测试在三个方面存在不足：大多数只在短任务中测试一两个机械臂，许多基准在智能体思考时会暂停环境，许多基准还会向智能体提供物体的真实位置信息。我们提出了RoboTeam，这是一个模拟基准测试，智能体仅凭相机图像实时控制4到16个机械臂。其36个任务是机械臂必须协作完成的长时作业，例如传递木材或用数百个立方体拼出一幅图画。我们从最灵活的智能体系统——自适应系统开始，其中一个智能体决定何时分派出代理智能体。在该系统下，gpt-6.1-sol和opus-5.5分别完成了28和27个任务，而其他四个模型最多只完成3个。但代价高昂：一个典型任务需要gpt-6.1-sol花费3.5小时和2.18亿输入token，并且其三分之一的步骤至少失败一次、必须重试。随后我们将其与三种固定系统进行比较：单智能体、分层……（摘要原文在此处截断）

    arXiv:2611.22138v1 Announce Type: cross  Abstract: Multimodal agents can now control robot arms by calling tools and writing code. Existing benchmarks fall short in three ways: most test one or two arms on short tasks, many pause the world while the agent thinks, and many give it ground-truth object positions. We present RoboTeam, a simulated benchmark where agents control 4 to 16 robot arms in real time from camera images alone. Its 36 tasks are long jobs that the arms must do together, such as passing timbers or building a picture from hundreds of cubes. We start from the most flexible agent system, the adaptive system, where one agent decides when to fork delegates. With it, gpt-6.1-sol and opus-5.5 complete 28 and 27 tasks, and four other models at most 3. But at a high price: a typical task takes gpt-6.1-sol 3.5 hours and 218 million input tokens, and a third of its steps fail at least once and must be retried. We then compare it with three fixed systems: single-agent, hierarchica
    
[^170]: 大脑很强，板卡很小：Pupavision——用于监测果蝇蛹化与生命周期转变的智能体式物理AI仪器

    Big Brains, Tiny Board: Pupavision, an Agentic Physical AI Instrument for Monitoring Drosophila Pupariation and Lifecycle Transitions

    [https://arxiv.org/abs/2611.22136](https://arxiv.org/abs/2611.22136)

    Pupavision以约45–75美元的成本将前沿编程智能体直接部署在单板计算机上，使其自主完成果蝇蛹化等生命周期实验的图像采集、错误修复与日常检查，实现了低成本、高可靠性的自动化生物学实验监测。

    

    许多生物学实验都受限于监测和记录实验的仪器的可获得性与易用性。例如，广泛使用的模式生物黑腹果蝇的蛹化时间可以提供与生长调控相关的生物学信息，而这一过程通常需要人工记录。本文提出了Pupavision，这是一款由单板Linux计算机和日夜两用摄像机构建的小型物理AI仪器，零部件成本约为45至75美元。其核心思想是：让前沿编程智能体的智能体运行框架直接部署在微型板卡上。模型在云端思考，而智能体在设备上工作。该智能体协助图像采集、编写代码修复错误，并每天检查实验，以确保原始数据采集的可靠性。一旦记录中断，研究人员会在手机上收到提醒。一个简单的确定性采集服务每十分钟记录一张图像，并且永不……（原文摘要在此处被截断）

    arXiv:2611.22136v1 Announce Type: new  Abstract: Many biology experiments are limited by the availability and ease of use of instruments that monitor and record them. For example, the timing of pupariation in the fruit fly Drosophila melanogaster, a widely used model organism, can provide biologically relevant information about growth regulation and is often recorded manually. Here, we present Pupavision, a small physical AI instrument built from a single-board Linux computer and a day/night camera for roughly 45 to 75 US dollars in parts. The key idea is that a frontier coding agent runs its agentic harness directly on the tiny board. The model thinks in the cloud, but the agent works on the device. The agent assists image capture, writes code to repair errors, and checks the experiment every day to keep raw data capture reliable. If recording stops, the researcher gets a ping on their mobile phone. A simple deterministic capture service records one image every ten minutes and never d
    
[^171]: LoCaR：基于冻结视觉-语言模型的局部画布检索

    LoCaR: Localized Canvas Retrieval With Frozen Vision-Language Model

    [https://arxiv.org/abs/2611.22131](https://arxiv.org/abs/2611.22131)

    LoCaR 是一个无需训练的局部化文本到图像检索框架，利用 SAM2 生成的目标候选区域和冻结视觉编码器对每个区域独立编码，从而有效提升对小物体和局部属性的检索能力，并引入 LOCFG 基准数据集进行评估。

    

    整幅图像的嵌入表示可能无法充分表征那些在局部查询中用于区分相关图像的小物体和局部属性。我们提出了 LoCaR，这是一个无需训练的框架，用于在冻结的视觉-语言模型上进行局部化的文本到图像检索。LoCaR 构建并利用聚焦于目标的独立候选区域（SAM2），通过冻结的视觉编码器对每个区域进行独立编码，从而获得更局部化的表示。我们进一步研究了配对支持适配，作为一种可选的残差方法，它利用匹配的文本-区域示例而无需更新底层编码器。我们还引入了 LOCFG，这是一个源自 Visual Genome 的基准数据集，旨在评估针对更小、更具挑战性指代对象的局部化检索。在多个视觉-语言骨干网络上的结果表明，改进区域表示本身是性能提升的主要来源。

    arXiv:2611.22131v1 Announce Type: new  Abstract: Whole image embeddings can under represent small objects and local attributes that distinguish relevant images for localized queries. We present LoCaR, a training free framework for localized text to image retrieval with frozen vision language models. LoCaR constructs and utilizes independent proposals (SAM2) focusing on the object so as to get a more localized representation that independently encodes each region using the frozen visual encoder. We further study paired-support adaptation as an optional residual that uses matched text region exemplars without updating the underlying encoders. We also introduce LOCFG, a Visual Genome derived benchmark designed to evaluate localized retrieval for smaller and more challenging referents. Across several vision language backbones, our results show that improving the region representation itself is the main source of gain.
    
[^172]: 从TeleChat到星4（Xing 4）：基于国产芯片与框架推进前沿智能体模型

    TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks

    [https://arxiv.org/abs/2611.22115](https://arxiv.org/abs/2611.22115)

    星4系列通过模型架构优化与面向昇腾芯片和MindSpore框架的深度系统工程，实现了在8192张昇腾910C卡上预训练2850亿参数的前沿智能体大模型。

    

    我们提出了星4（Xing 4），这是一系列大型语言模型，通过在昇腾平台上模型架构与训练基础设施的协同进步，延续了TeleChat系列的发展。本报告描述了该系列背后的设计原则与系统工程，涵盖架构选择、大规模预训练以及推理与智能体后训练。我们研究了稀疏激活、注意力设计、模型深度与宽度以及投机解码如何在模型容量与计算、内存访问及执行开销之间取得平衡。我们的预训练基础设施基于MindSpore HyperParallel构建，结合了拓扑感知的混合分片、高效的分布式优化器执行，以及对长序列工作负载与激活内存的协同管理。CPU亲和性与核心隔离进一步减少了主机调度干扰。这些机制共同支持了在8,192张昇腾910C卡上预训练2850亿参数模型。

    arXiv:2611.22115v1 Announce Type: new  Abstract: We present Xing 4, a series of large language models that extends the TeleChat lineage through coordinated advances in model architecture and training infrastructure on Ascend. This report describes the design principles and systems engineering behind the series, spanning architecture selection, large-scale pretraining, and reasoning and agentic post-training. We study how sparse activation, attention design, model depth and width, and speculative decoding balance model capacity with computation, memory access, and execution overhead. Our pretraining infrastructure builds on MindSpore HyperParallel, combining topology-aware hybrid sharding, efficient distributed optimizer execution, and coordinated management of long-sequence workloads and activation memory. CPU affinity and core isolation further reduce host scheduling interference. Together, these mechanisms support the pretraining of a 285B-parameter model on 8,192 Ascend 910C cards, 
    
[^173]: RegimeCouncil：面向LLM智能体记忆冲突治理的成对证据路由方法

    RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents

    [https://arxiv.org/abs/2611.22113](https://arxiv.org/abs/2611.22113)

    该论文提出RegimeCouncil——一个无需重训LLM的轻量级成对证据路由器，将“记忆冲突治理”形式化，并在局部与全局两种记忆抑制策略之间在线仲裁，以解决长期记忆悄然过时与延迟反馈下负面经验传播范围难以确定的问题。

    

    语言智能体依赖长期记忆来在长程交互中实现个性化响应，然而随着世界的变化，存储的事实会悄然过时，而智能体往往只有在基于过时记录采取行动之后，才通过延迟反馈得知使用记忆带来了损害。在记忆库被更新之前，这种负面反馈应该传播多远：是仅在针对同一事实的未来查询中抑制记忆使用（局部策略），还是也应跨及其他存储事实（全局策略）？现有记忆系统在设计阶段就隐式地固定了这两种共享范围之一。我们将这种在有效性隐藏、反馈延迟条件下的在线范围选择问题形式化为“记忆冲突治理”，并证明任何一种固定策略单独使用都不足以应对。随后我们提出RegimeCouncil，一个轻量级的成对证据路由器，它将两种范围保留为“顾问”，并在无需重新训练LLM的情况下在线地在两者之间进行仲裁。（原文摘要至此处截断）

    arXiv:2611.22113v1 Announce Type: new  Abstract: Language agents rely on long-term memory to personalize responses across extended interactions, yet stored facts quietly go stale as the world changes, and an agent often learns that using memory hurt only through delayed feedback after acting on an outdated record. Before the memory store is updated, how far should that negative feedback propagate: should it suppress memory use only on future queries for that same fact (a local policy), or also across other stored facts (a global policy)? Existing memory systems implicitly fix one of these two sharing scopes at design time. We formulate this online scope choice under hidden validity and delayed feedback as memory conflict governance, and prove that neither fixed policy is sufficient on its own. We then propose RegimeCouncil, a lightweight paired-evidence router that retains both scopes as advisors and arbitrates between them online without retraining the LLM. Using revealed comparisons 
    
[^174]: 从Sorry到QED：通过天使执行精炼大语言模型的数学证明

    From Sorry to QED: Refining LLM Math Proofs via Angelic Execution

    [https://arxiv.org/abs/2611.22103](https://arxiv.org/abs/2611.22103)

    本文提出MathEye框架，其核心“天使证明检查”以乐观执行方式将LLM生成的Lean证明脚本中失败片段替换为占位符，在保留整体证明结构的同时精确定位失败点，从而以最少的人工交互把不完善的证明转化为完全验证的Lean证明。

    

    大语言模型（LLM）如今能够生成捕捉许多数学论证高层次结构的Lean证明脚本。然而，这些脚本在处理复杂定理时经常失败：一个缺失的附带条件或误用的引理就可能导致证明器提前中止，并丢弃原本可行的证明大纲。本文提出了MathEye，一个通过将符号推理与最少的人类交互相结合，把不完善的LLM生成证明脚本转化为完全验证的Lean证明的框架。MathEye的核心是天使证明检查，这是一种乐观执行模型，它在保留论证整体结构的同时，用占位符替换失败的证明片段。这种乐观检查产生一个天使证明，能够精确分离出LLM无法完成的证明义务。由于这些未解决的义务往往在逻辑上相互依赖，让数学家处理每一个占位符……

    arXiv:2611.22103v1 Announce Type: new  Abstract: Large language models (LLMs) can now generate Lean proof scripts that capture the high-level structure of many mathematical arguments. However, these scripts frequently fail on complex theorems: a single missing side condition or misapplied lemma can cause the prover to abort early and discard an otherwise viable proof outline. This paper presents MathEye, a framework that converts imperfect LLM-generated proof scripts into fully verified Lean proofs by combining symbolic reasoning with minimal human interaction. At the core of MathEye is angelic proof checking, an optimistic execution model that replaces failing proof fragments with placeholders while preserving the overall structure of the argument. This optimistic check produces an angelic proof that isolates the exact proof obligations that the LLM fails to discharge. Because these unresolved obligations are often logically interdependent, asking a mathematician to address every plac
    
[^175]: EvoDisco：面向智能体记忆演化的策略发现

    EvoDisco: Strategy Discovery for Evolving Agent Memory

    [https://arxiv.org/abs/2611.22102](https://arxiv.org/abs/2611.22102)

    该论文提出EvoDisco框架，将智能体记忆演化建模为演化式策略发现问题，由可学习策略根据记忆状态和任务历史动态选择并实例化四种语义算子，突破了传统记忆更新中算子与调度规则固定不变的限制。

    

    大语言模型智能体在记忆中积累经验，而记忆在交互过程中如何演化与其存储的内容同等重要。近期的记忆更新方法虽然可以在线更新记忆，但用于转换记忆的算子以及选择算子的规则在设计上都是固定的。我们提出EvoDisco，将智能体记忆演化形式化为演化式策略发现：智能体需要自主发现应用哪种算子、何时应用以及如何实例化该算子，而不是在静态的算子调度下重写记忆。一个可学习的策略以当前记忆状态和近期任务历史为条件，然后选择并实例化四种语义算子之一——摘要、精炼、桥接和重构——它们分别用于压缩相关记忆、分解过载条目、连接弱关联的经验以及重组检索到的推理链。该策略通过双层适应度机制进行更新。内部信号对每……

    arXiv:2611.22102v1 Announce Type: new  Abstract: Large language model agents accumulate experience in memory, and how that memory evolves during interaction is as important as what it stores. Recent methods update memory online, yet both the operators that transform memory and the rules that select them remain fixed by design. We propose EvoDisco, which formulates agent memory evolution as evolutionary strategy discovery: the agent must discover which operator to apply, when to apply it, and how to instantiate it, rather than rewriting memory under a static operator schedule. A learnable policy conditions on the current memory state and recent task history, then selects and instantiates one of four semantic operators---Abstract, Refine, Bridge, and Reframe---which respectively compress related memories, decompose overloaded entries, connect weakly linked experiences, and reorganize retrieved reasoning chains. The policy is updated with a bi-level fitness. An inner signal scores each tr
    
[^176]: 基于模拟射频计算的语义通信：当无线电计算语义之时

    Semantic Communication via Analog RF Computing: When the Radio Computes the Meaning

    [https://arxiv.org/abs/2611.22100](https://arxiv.org/abs/2611.22100)

    该论文提出让边缘设备借助模拟射频计算在射频接收前端直接执行语义编码器——由基站广播权重波形、设备复用无源混频器实现超低能耗的端到端语义通信，并建立了模拟计算噪声与信道噪声的统一数学模型及语义失真的二阶敏感度分解。

    

    语义通信（SC）系统传输意义的连续潜在表示，但在资源受限的边缘设备上计算语义编码器会带来巨大的能耗与内存开销。为应对这一挑战，我们提出一种端到端的语义通信框架，利用模拟射频计算在客户端的射频接收机前端执行语义编码器。基站（BS）将编码器权重以射频波形的形式广播，而各边缘设备复用其接收机的无源混频器，以超低能耗计算语义潜在表示。这些潜在表示经上行链路传输至网络侧进行任务推理。我们证明，编码器各层的模拟计算噪声与影响所传输潜在表示的信道噪声可纳入统一的数学模型。我们定义了任务后验语义失真度量，并推导出其按客户端、按阶段语义敏感度的二阶分解。利用该分解，我们……（摘要原文在此处截断）

    arXiv:2611.22100v1 Announce Type: cross  Abstract: Semantic communication (SC) systems transmit continuous latent representations of meaning, but computing the semantic encoder at resource-constrained edge devices incurs substantial energy and memory costs. To address this challenge, we propose an end-to-end SC framework that executes the semantic encoder in the client's RF receiver front end using analog RF computing. A base station (BS) broadcasts encoder weights as RF waveforms, while each edge device reuses its receiver's passive mixer to compute semantic latents at ultra-low energy. These latents are transmitted over the uplink for network-side task inference. We show that analog computing noise across encoder layers and channel noise affecting transmitted latents admit a unified mathematical model. We define a task-posterior semantic distortion metric and derive its second-order decomposition into per-client, per-stage semantic sensitivities. Using this decomposition, we obtain a
    
[^177]: 当AUC遇上生成式重建：平衡视觉表征学习的统一原则

    When AUC Meets Generative Reconstruction: A Unified Principle for Balanced Visual Representation Learning

    [https://arxiv.org/abs/2611.22094](https://arxiv.org/abs/2611.22094)

    该论文提出将生成式重建重新表述为判别排序问题的统一表征学习范式，解决了对比学习与生成重建之间的目标冲突，实现了细粒度感知与判别能力兼顾的平衡视觉表征学习。

    

    多模态大语言模型（MLLMs）需要视觉编码器在提取细粒度视觉特征的同时，构建语义解耦且具有判别性的表征。传统方法面临一个根本性的权衡：对比学习通过抑制类内变异性来增强类间判别能力，而生成式重建强调感知保真度，但缺乏显式的对比监督。通过复合损失函数简单地将这些目标聚合在一起，往往会由于共享潜在空间中的梯度干扰而引发优化冲突和不稳定性。为解决这一问题，我们引入了一种统一的表征学习范式，将生成式重建重新表述为生成输出流形上的判别排序问题。我们的方法将感知重建保留为监督信号，并集成了基于排序的判别……（摘要原文在此处截断）

    arXiv:2611.22094v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) require visual encoders that extract fine-grained visual features while constructing semantically disentangled and discriminative representations. Conventional methodologies encounter a fundamental trade-off: contrastive learning enhances inter-class discrimination by suppressing intra-class variability, whereas generative reconstruction emphasizes perceptual fidelity but lacks explicit comparative supervision. The naive aggregation of these objectives via composite loss functions often induces optimization conflicts and instability due to gradient interference within the shared latent space. To resolve this, we introduce a unified representation-learning paradigm that reformulates generative reconstruction as a discriminative ranking problem within the generative output manifold. Our approach maintains perceptual reconstruction as a supervisory signal and integrates a ranking-based discriminati
    
[^178]: HyCoInfer：基于异构GPU-FPGA架构的大语言模型云边协同推理

    HyCoInfer: Cloud-Edge Collaborative Inference for Large Language Models on Heterogeneous GPU-FPGA Architectures

    [https://arxiv.org/abs/2611.22086](https://arxiv.org/abs/2611.22086)

    HyCoInfer提出了一种云边协同的异构GPU-FPGA大语言模型推理系统，通过将预填充分配给云端GPU、将解码任务分布到云端GPU与边缘FPGA，并解决异构容量均衡、KV缓存广域传输和异构执行延迟三大关键问题，实现高效的LLM服务。

    

    arXiv:2611.22086v1 公告类型：交叉  摘要：大语言模型（LLM）服务主要依赖云端GPU，然而自回归解码过程受内存带宽限制，常常导致GPU的算术单元利用率不足。边缘FPGA虽具备有吸引力的解码效率，但难以独立支撑高并发的服务负载。本文提出HyCoInfer，一种异构云边服务系统，将预填充阶段分配给云端GPU，并将解码阶段分布到云端GPU与边缘FPGA上。该设计解决了实际卸载面临的三大障碍：异构加速器之间的容量均衡、跨越受限广域网链路传输大容量KV缓存、以及异构执行引入的额外延迟。HyCoInfer将FPGA的解码容量纳入GPU预填充/解码角色划分机制中，并根据队列负载、提示长度和网络状态来放置请求。此外，该系统结合了逐层流水线传输、解码端预计算以及带宽自适应……（原文摘要在此处被截断）

    arXiv:2611.22086v1 Announce Type: cross  Abstract: Large language model (LLM) serving relies predominantly on cloud GPUs, yet autoregressive decoding is memory-bound and often leaves their arithmetic units underutilized. Edge FPGAs offer attractive decoding efficiency, but are ill suited to sustaining a high-concurrency serving workload alone. This paper presents HyCoInfer, a heterogeneous cloud-edge serving system that assigns prefill to cloud GPUs and distributes decoding across cloud GPUs and edge FPGAs. The design addresses three barriers to practical offloading: capacity balancing across heterogeneous accelerators, the large KV cache crossing a constrained wide-area link, and the latency added by heterogeneous execution. HyCoInfer incorporates FPGA decode capacity into GPU prefill/decode role partitioning and places requests according to queue load, prompt length, and network state. It further combines layer-wise pipelined transfer, decoder-side precomputation, and bandwidth-adapt
    
[^179]: ATLAS-Finance：在真实银行环境中对AI智能体进行基准测试

    ATLAS-Finance: Benchmarking AI Agents in Realistic Banking Environments

    [https://arxiv.org/abs/2611.22085](https://arxiv.org/abs/2611.22085)

    该论文提出ATLAS-Finance基准，通过13家模拟公司中的100个端到端投资银行任务，评估AI智能体在信息分散、任务欠明确的真实银行工作环境中搜索多源信息并完成复杂金融建模等高价值专业工作的能力。

    

    专业工作往往涉及规定欠明确的任务，相关信息分散在各类办公应用之中，且可能相互冲突或随时间变化。为了评估前沿智能体在此类环境中完成具有经济价值工作的能力，我们提出了ATLAS-Finance，这是一个包含横跨13家模拟公司的100个端到端投资银行任务的基准。与真实的银行工作流程一样，智能体必须搜索电子邮件、聊天记录、日历、笔记和文件，以识别需求、收集信息、调和冲突与更新，然后构建复杂的金融模型。这些任务由经验丰富的银行家基于其真实的交易工作开发，人类完成每项任务需要15至30小时。智能体的交付成果由专家撰写的评分标准（中位数为124.5项评判标准）进行评估，涵盖金融逻辑与结果、电子表格构建、完整性和格式等方面。尽管智能体完成了所要求工作的相当大一部分，但当前的智能体仍然……

    arXiv:2611.22085v1 Announce Type: new  Abstract: Professional work involves under-specified assignments, where relevant information is spread across workplace applications and may conflict or change over time. To evaluate how well frontier agents can perform economically valuable work in such settings, we introduce ATLAS-Finance, a benchmark of 100 end-to-end investment-banking tasks across 13 simulated firms. As in real banking workflows, agents must search email, chat, calendars, notes, and files to identify requirements, gather information, reconcile conflicts/updates, and then build complex financial models. Tasks were developed by experienced bankers based on their real deal work and take them 15-30 hours to complete. Agent deliverables are evaluated with expert-authored rubrics (median: 124.5 criteria) covering financial rationale and results, spreadsheet construction, completeness, and formatting. Despite completing substantial portions of the requested work, current agents rare
    
[^180]: 当更低的伤害产生误导时：审计概念修正

    When Lower Harm Misleads: Auditing Concept Corrections

    [https://arxiv.org/abs/2611.22084](https://arxiv.org/abs/2611.22084)

    该论文提出“修正审计”这一可复现评估协议，揭示修正伤害的降低可能源于模型对修正响应变弱或起点更差等误导因素，并证明最坏情况修正训练虽降低伤害却牺牲了更多有效增益，使修正后NLL上升。

    

    当专家修正一个概念预测（例如乳腺超声中的BI-RADS描述符）时，期望诊断能够得到改善。降低此类修正伤害的目标函数通常以伤害和准确性来评判，然而，当模型对每次修正的响应变弱，或从一个更差的未修正预测（即衡量伤害的参考基准）出发时，伤害同样会下降。我们提出“修正审计”——一种可复现的评估协议，用于区分这些不同的解释：它在相同事件上将每次修正分解为伤害与有效增益，将起始预测与修正响应分离开来，并将各目标函数与源伤害匹配的幅度基线进行比较。在三个乳腺超声数据源上，从平均情况模型的起点评估，最坏情况修正训练虽然降低了伤害，但在六个随机种子汇总的全部21个数据源比较中放弃了更多的有效增益，导致修正后的负对数似然（NLL）上升；从相同的起点出发，选择...

    arXiv:2611.22084v1 Announce Type: new  Abstract: An expert who corrects a concept prediction, such as a BI-RADS descriptor in breast ultrasound, expects the diagnosis to improve. Objectives that reduce the harm of such corrections are judged by harm and accuracy, yet harm also falls when a model responds less to every correction or starts from a worse uncorrected prediction, the reference for harm. We propose a correction audit, a reproducible evaluation protocol that separates these explanations: it splits each correction into harm and useful gain on identical events, separates the starting prediction from the correction response, and compares objectives with amplitude baselines matched in source harm. On three breast-ultrasound sources, worst-case correction training, evaluated from the mean-case model's start, lowers harm but gives up more useful gain in all 21 source comparisons pooled over six seeds, raising the corrected negative log-likelihood (NLL); from the same start, selecti
    
[^181]: 基于航空影像的分割引导屋顶光伏容量直接估计

    Segmentation-Guided Direct Capacity Estimation of Rooftop Photovoltaics from Aerial Imagery

    [https://arxiv.org/abs/2611.22076](https://arxiv.org/abs/2611.22076)

    提出Direct Head分割引导架构，在保留U-Net分割与软掩膜池化的同时直接从航空影像回归屋顶光伏容量，摆脱了传统“掩膜面积×功率密度”的间接估计方式，在7,590个法国屋顶装置上将误差最多降低20.7%。

    

    配电规划人员需要安装级别的屋顶光伏（PV）容量清单，但并网记录可能不完整或在空间上被聚合。传统的基于图像的估计器先对光伏阵列进行分割，将掩膜转换为物理面积，再乘以估计的功率密度，从而使容量预测与掩膜导出的面积相互耦合。我们提出了Direct Head，这是一种分割引导的架构，在保留U-Net分割和软掩膜池化的同时，直接从光伏区域特征和面积描述符回归出非负容量。我们在7,590个法国屋顶光伏装置上，分别在原始数据划分和基于省的留出集两种设置下进行了三次独立运行评估。与参数数量匹配的面积-密度神经网络模型相比，Direct Head将平均绝对误差分别降低了8.7%和12.5%，将对称平均绝对百分比误差分别降低了14.0%和20.7%。与常规估计器的比较以及受控架构……（摘要原文在此处截断）

    arXiv:2611.22076v1 Announce Type: cross  Abstract: Distribution planners need installation-level inventories of rooftop photovoltaic (PV) capacity, but interconnection records may be incomplete or spatially aggregated. Conventional image-based estimators segment PV arrays, convert the mask to physical area, and multiply by an estimated power density, coupling capacity prediction to mask-derived area. We propose Direct Head, a segmentation-guided architecture that retains U-Net segmentation and soft-mask pooling while directly regressing nonnegative capacity from PV-region features and area descriptors. We evaluate three independent runs on 7,590 French rooftop installations under the original split and a department-based holdout. Relative to a parameter-matched area-density neural model, Direct Head reduces mean absolute error by 8.7% and 12.5%, and symmetric mean absolute percentage error by 14.0% and 20.7%, respectively. Comparisons with conventional estimators, controlled architectu
    
[^182]: 你能撤销它吗？面向AI智能体动作的可逆性分类

    Can You Undo It? Reversibility Classes for AI Agent Actions

    [https://arxiv.org/abs/2611.22075](https://arxiv.org/abs/2611.22075)

    本文提出“可逆性分类”——一个声明式的五级词汇体系，通过参数敏感规则、保守默认值和先验状态捕获契约来判定AI智能体工具动作能否被撤销，其实现的会话守卫在攻击场景下能拦截全部29次可归因的攻击阳性运行，优于读写代理基线（仅18次），且对良性运行的干扰极小。

    

    使用工具的AI智能体会更改文件、数据库、访问权限和公开信息，然而允许调用某个工具并不意味着其产生的错误可以被撤销。我们提出了可逆性分类（Reversibility Classes），这是一个声明式的五级词汇体系，包含参数敏感的规则、保守的默认值以及明确的先验状态捕获契约。其固定版本的实现和带有证据链接的注册表涵盖了20个MCP服务器上的536个工具。全部738项规则用例检查均符合声明预期，而在481个可比较的工具中，有118个工具的原生注解映射与注册表草案标签不一致。一名AI辅助审核员批准了148项评级，其中75项与注册表匹配，但这并非独立的一致性验证。在2,158次记录的运行中，所实现的会话守卫在全部29次可归因的攻击阳性运行中拦截了相关动作，而读写代理仅拦截了18次，同时在1,209次良性运行中仅新增了9个审批点。在受控的文件、Git、SQLite等环境中……（摘要在此处被截断）

    arXiv:2611.22075v1 Announce Type: new  Abstract: Tool-using AI agents change files, databases, access rights and public information, yet permission to call a tool does not establish whether a mistake can be undone. We present Reversibility Classes, a declared five-level vocabulary with argument-sensitive rules, conservative defaults and an explicit prior-state capture contract. The pinned implementation and evidence-linked registry cover 536 tools across 20 MCP servers. All 738 rule-case checks conform to declared expectations, while native annotation mapping differs from draft registry labels for 118 of 481 comparable tools. One AI-assisted reviewer approves 148 ratings, 75 matching the registry; this is not independent agreement. On 2,158 recorded runs, the implemented session guard gates the associated action in all 29 attributable attack-positive runs, compared with 18 for a read/write proxy, while adding nine approval points across 1,209 benign runs. Controlled file, Git, SQLite a
    
[^183]: MedGuardBench：一个面向患者的个性化防护栏开发基准

    MedGuardBench: A Patient-Oriented Benchmark for Personalized Guardrail Development

    [https://arxiv.org/abs/2611.22069](https://arxiv.org/abs/2611.22069)

    该论文发布了MedGuardBench——首个以患者档案和处方为条件的个性化用药安全防护基准（包含87种药物、5,001个场景），并提出个性化推理防护栏方法，使模型能够基于患者的真实信息而非表面线索做出安全判断。

    

    安全是因人而异的：同一个行为对一个人可能安全，对另一个人却可能不安全。然而，当前的防护栏在判断内容时并未考虑受影响的是谁。我们以用药安全为切入点，提出了“以患者为条件的用药防护”，即根据患者档案和处方来预测安全判定结果以及17个风险类别。我们发布了MedGuardBench，这是一个涵盖87种药物、包含5,001个场景且训练与测试药物互不重叠的基准数据集。我们发现，标准防护栏依赖的是表层线索（如措辞和患者档案是否存在），而非患者的真实信息。为解决这一问题，我们开发了由三个组件构成的个性化推理防护栏。个性化推理蒸馏将风险与相关的患者字段关联起来。上下文展开让模型自行逐项检查各个患者字段，在大多数骨干模型上降低了误报率；对照实验表明，直接向模型提供检查结果并不能带来同样的收益。一种基于规则的……（摘要原文在此处被截断）

    arXiv:2611.22069v1 Announce Type: new  Abstract: Safety is personal: the same action can be safe for one person but unsafe for another. Yet current guardrails judge content without considering who is affected. Using medication safety, we introduce \textit{patient-conditioned medication guarding}, which predicts a safety verdict and 17 risk categories from a patient profile and prescription. We release \textbf{MedGuardBench}, a drug-disjoint benchmark of 5{,}001 scenarios across 87 medications. We find that standard guardrails rely on surface cues, such as wording and profile presence, rather than patient facts. To address this, we develop \textit{personalized reasoning guardrails} with three components. \textbf{Personalized Reasoning Distillation} links risks to relevant patient fields. \textbf{Context Unrolling (CU)} makes models check each field themselves, reducing false alarms on most backbones; controls show that providing the check does not yield the same benefit. A \textbf{rule-
    
[^184]: FM-Mamba：一种具有稳定性感知离散化的高效原生二维选择性状态空间模型

    FM-Mamba: An Efficient Native 2D Selective State Space Model with Stability-Aware Discretization

    [https://arxiv.org/abs/2611.22061](https://arxiv.org/abs/2611.22061)

    提出FM-Mamba，一种基于Fornasini-Marchesini模型的原生二维选择性状态空间模型，通过沿双轴演化的耦合递归、输入依赖的双步长和稳定性感知离散化，解决了现有Mamba方法处理二维特征时破坏空间邻接、计算量倍增及缺乏二维稳定性约束的问题。

    

    当应用于图像等二维特征时，基于Mamba的方法要么将特征图展平为一维序列并融合多次扫描，这会破坏空间邻接关系并成倍增加计算量；要么直接在二维网格上进行递归，但使用一个固定的、与输入无关的因子对递归进行缩放，且仅聚合单一反对角线上的前驱状态，使得离散步长不受二维稳定性理论的约束。我们提出了FM-Mamba，一种建立在一般Fornasini–Marchesini（FM）模型之上的原生二维选择性状态空间模型。其耦合递归沿两个空间轴同时演化；跨轴耦合产生了一个离散对角前驱项 $\tilde A_0$，两个独立的、依赖于输入的步长 $\Delta_1(i,j)$ 和 $\Delta_2(i,j)$ 分别控制沿两个轴的传播，使系数能够随位置和方向自适应调整。归一化比率 $r_k=a_{0,k}/(a_{1,k}a_{2,k})$ 可推导出一个闭式……（摘要原文不完整）

    arXiv:2611.22061v1 Announce Type: cross  Abstract: When applied to two-dimensional features such as images, Mamba-based methods either flatten the feature map into 1D sequences and fuse multiple scans, which breaks spatial adjacency and multiplies computation, or recurse directly on the 2D grid but rescale the recurrence with a fixed input-independent factor and aggregate only the predecessors on a single anti-diagonal, leaving the discrete step unconstrained by 2D stability theory. We propose \textbf{FM-Mamba}, a native 2D selective state space model built on the general Fornasini--Marchesini (FM) model. Its coupled recurrence evolves along both spatial axes; cross-axis coupling induces a discrete diagonal-predecessor term $\tilde A_0$, and two independent input-dependent step sizes $\Delta_1(i,j)$, $\Delta_2(i,j)$ govern propagation along the two axes, so that the coefficients adapt per position and per direction. A normalized ratio $r_k=a_{0,k}/(a_{1,k}a_{2,k})$ yields a closed-form
    
[^185]: 权重空间编程不需要Transformer

    Weight-Space Programming Doesn't Need Transformers

    [https://arxiv.org/abs/2611.22055](https://arxiv.org/abs/2611.22055)

    在无需梯度训练的权重空间编程中，三个前沿LLM在自由设计架构时都不约而同地将知识编译为可查询的稀疏表，且其性能以约1/1800的参数量超越了Transformer，表明权重空间编程并不需要Transformer。

    

    诸如Transformer之类的标准神经架构之所以被选用，是因为它们能通过梯度下降很好地训练，并且在GPU上运行速度快。而在权重空间编程中，大型语言模型（LLM）无需基于梯度的训练即可直接编写目标权重，因此这些理由不再适用。我们还应该继续使用Transformer吗？在最少指导、十个标注示例和一个沙盒环境的条件下（任务主要为多项选择题），我们让三个前沿LLM要么为固定架构生成权重，要么自行设计架构。当可以自由设计时，这三个LLM都将纯文本事实编译成一个由固定程序查询的稀疏表，测试准确率最高达到54.4%。LLM设计的模型比同一LLM所编写的最准确的已发布Transformer取得了更高的测试准确率，而参数量仅约为后者的1/1800。在没有梯度的情况下，它们选择了一种表结构，其中LLM可以说明每个权重的含义……

    arXiv:2611.22055v1 Announce Type: cross  Abstract: Standard neural architectures such as the Transformer were chosen because they train well with gradient descent and run fast on GPUs. In weight-space programming, a large language model (LLM) writes target weights by hand without gradient-based training, so those reasons no longer apply. Should we keep the Transformer? Given minimal guidance, ten labeled examples, and a sandbox, mainly on multiple-choice questions, we task three frontier LLMs with either generating the weights of a fixed architecture or designing the architecture themselves. When free to design, all three compile plain-text facts into a sparse table queried by a fixed procedure, reaching up to 54.4% test accuracy. An LLM-designed model achieves higher test accuracy than the most accurate released Transformer that the same LLM wrote while using about 1,800x fewer parameters. Without gradients, they choose a table, in which the LLM can state what each weight means and ca
    
[^186]: 乘客与包裹联合运输问题的精确-启发式混合求解器

    A Hybrid Exact-Heuristic Solver for the Integrated Passenger and Parcel Transportation Problem

    [https://arxiv.org/abs/2611.22043](https://arxiv.org/abs/2611.22043)

    该论文提出一个融合常数时间插入测试、自适应大邻域搜索与精确集合装箱引擎（分支定界结合拉格朗日界强化）的混合求解器，用于求解带时间窗、开放路线和异构出租车的乘客与包裹联合运输问题。

    

    我们描述了提交给SOICT 2026竞赛第二轮“乘客与包裹联合运输问题”的求解器，该问题是一个带时间窗、开放路线和异构出租车的选择性取送货路径规划问题。该求解器是一个仅依赖标准库的单文件Python程序，结合了：(i) 基于最晚到达时间裕量的常数时间插入测试；(ii) 贪心插入、局部搜索以及采用模拟退火接受准则的自适应大邻域搜索（ALNS）；(iii) 一个精确路线选择引擎，该引擎为每辆出租车枚举每个请求集的最佳可行路线，随后用分支定界法求解由此产生的集合装箱问题，并通过基于路线的大邻域搜索、拉格朗日界和削减成本固定法加以强化。引擎根据问题实例规模自动选择，同时独立的验证器和看门狗机制确保始终输出有效答案。

    arXiv:2611.22043v1 Announce Type: new  Abstract: We describe the solver we submitted to the second round of the SOICT 2026 contest on the Integrated Passenger and Parcel Transportation problem, a selective pickup-and-delivery routing problem with time windows, open routes and heterogeneous taxis. The solver is a single-file, standard-library Python program that combines (i) a constant-time insertion test based on latest-arrival slack, (ii) greedy insertion, local search and an Adaptive Large Neighborhood Search (ALNS) with simulated-annealing acceptance, and (iii) an exact route-selection engine that enumerates, for every taxi, the best feasible route for each request set and then solves the resulting set-packing problem by branch-and-bound strengthened with route-based large neighborhood search, Lagrangian bounds and reduced-cost fixing. The engine is chosen automatically from the instance size, and an independent verifier and a watchdog guarantee that a valid answer is always printed
    
[^187]: RLDE：一个用于可解释迭代文档编辑的强化学习框架

    RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing

    [https://arxiv.org/abs/2611.22039](https://arxiv.org/abs/2611.22039)

    提出了RLDE框架，将文档迭代修订建模为无参考的MDP问题，通过指针网络与PPO智能体结合冻结的BART模型，依据可读性、困惑度和语法等多目标奖励实现可解释的自动化文档编辑。

    

    提升长文档的质量、清晰度和流畅性仍然是自然语言处理（NLP）中一个具有挑战性的问题，因为人工编辑通常是通过迭代式润色而非一次性重写完成的。现有的大多数自动化系统要么充当黑盒生成器，要么依赖监督学习，这限制了可解释性，并且无法捕捉修订过程的序列特性。本文提出了RLDE，一个自动化的端到端、指标驱动的强化学习框架，它将文档修订形式化为一个无参考的马尔可夫决策过程（MDP）问题（相对于成对的文档-修订目标而言是无监督的）。在有限的动作空间（NOOP、INSERT、DELETE、REPLACE）下，RLDE利用指针网络在词元级别进行动态片段选择，并将生成式动作发送给冻结的BART-base模型以获得上下文感知的修改建议。PPO智能体学习优化基于可读性、困惑度和语法指标的多目标奖励函数……

    arXiv:2611.22039v1 Announce Type: cross  Abstract: Improving the quality, clarity, and flow of long documents remains a challenging problem in NLP, as human editing typically occurs through iterative refinement rather than a single-pass rewrite. Most existing automated systems act as black-box generators or rely on supervised learning, limiting interpretability and failing to capture the sequential nature of revisions. In this paper, we propose RLDE, an automated end-to-end metric-driven RL framework that formulates document revision as a reference-free MDP problem (unsupervised with respect to paired document-revision targets). Working with a limited action space (NOOP, INSERT, DELETE, REPLACE), RLDE leverages Pointer Networks for dynamic span selection at the token level and sends generative actions to the frozen BART-base model to obtain context-aware suggestions. PPO agents learn to optimize a multi-objective reward function based on readability, perplexity, and grammar metrics wit
    
[^188]: 更努力思考并不能带来更好的选择：推理努力提升了LLM购物代理的置信度，而非正确性

    Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents

    [https://arxiv.org/abs/2611.22033](https://arxiv.org/abs/2611.22033)

    在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。

    

    大型语言模型（LLM）越来越多地被部署为自主购物代理，它们处理多模态商品信息（包括视觉和文本数据）并代表消费者推荐购买。人们自然会认为，思考能力更强的模型会做出更高质量的购买决策。为了验证这一观点，我们针对此类代理提出了两个问题：要求模型更努力地思考，是否会让它选出更好的产品？它所报告的置信度，能否告诉我们它出错的概率？我们在基于WebShop、亚马逊SQID和Wayfair的WANDS平台构建的三个产品选择基准上，对十个前沿模型进行了测试，以检验思考能力更强的LLM是否真的能做出更高质量的产品购买决策。在全部三个基准上，我们发现增加思考努力并不能在统计上显著提升决策质量。而同样的思考努力变化却会提高模型所声称的置信度（原文摘要在此处截断）。

    arXiv:2611.22033v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as autonomous shopping agents that process multimodal product listings (consisting of visual and text data) and recommend purchases on behalf of consumers. Naturally, one might assume that models with more thinking power would make higher-quality purchasing decisions. To test this belief, we ask two questions about such agents: does asking a model to think harder make it choose better products, and does the confidence it reports tell us the likelihood it may be wrong? We benchmark ten frontier models on three product-choice benchmarks built from WebShop, Amazon's SQID and Wayfair's WANDS platforms to see if LLMs with more thinking power actually make better-quality product purchase decisions. Across all three benchmarks, we found that increasing the thinking effort does not increase decision quality in a statistically meaningful way. The same change in thinking effort raises stated c
    
[^189]: 智能体需要什么？面向捆绑包感知的智能体资源组合

    What Does an Agent Need? Bundle-Aware Agent Resource Composition

    [https://arxiv.org/abs/2611.22030](https://arxiv.org/abs/2611.22030)

    该论文提出AgentComposition框架，联合选择骨干模型与工具-技能捆绑包以最大化智能体端到端执行效用，并通过检索增强生成、受限束搜索和捆绑包感知评论家，实现无需重新训练的新型资源组合与执行反馈自适应。

    

    智能体资源的价值取决于其骨干模型和配套资源，且只有执行后才能被知晓。基于token的生成器仍受限于其训练目录，而独立的资源评分会忽略资源间的互补性，复用的捆绑包则针对其他任务。我们提出了任务条件化的智能体资源组合问题：联合选择骨干模型和工具-技能捆绑包，以最大化端到端执行效用。我们提出的基于检索的捆绑包感知智能体生成方法会检索相似任务的候选资源和历史配置。经微调的生成器读取资源描述，并通过受限束搜索重新组合候选资源，从而无需重新训练即可支持新资源。捆绑包感知的评论家利用大语言模型（LLM）教师和历史成功配置作为执行代理对完整配置进行排序，并在有执行反馈时进行自适应调整。

    arXiv:2611.22030v1 Announce Type: new  Abstract: An agent resource's value depends on its backbone and companion resources and is known only after execution. Token-based generators remain tied to their training catalog, while independent resource scoring misses complementarities and reused bundles target other tasks. We formulate Task-Conditioned Agent Resource Composition (AgentComposition): jointly selecting a backbone model and a tool-skill bundle to maximize end-to-end execution utility. Our Retrieval-Grounded Bundle-Aware Agent Generation retrieves candidate resources and past configurations of similar tasks. A fine-tuned generator reads resource descriptions and recombines candidates through constrained beam search, allowing new resources without retraining. A bundle-aware critic ranks complete configurations using a large language model (LLM) teacher and historically successful configurations as execution proxies, and adapts to execution feedback when available. We evaluate refe
    
[^190]: GauntletScore：在 LLM-AggreFact 上媲美最佳已发表检查器的可审计事实核查

    GauntletScore: Auditable Fact-Checking That Matches the Best Published Checkers on LLM-AggreFact

    [https://arxiv.org/abs/2611.22027](https://arxiv.org/abs/2611.22027)

    GauntletScore是一种可审计的多读者证据引擎，通过向不同提供商的独立读者提问、在证据账本中记录测量并使用固定贝叶斯规则汇总（仅在两位读者一致时判定矛盾），在LLM-AggreFact完整测试集上达到79.51的平衡准确率，与最佳已发表检查器持平。

    

    检测生成的句子是否得到其源文档的支持，是针对AI生成文本的一项标准保障措施，而已发表的最强检查器均为单一模型。我们介绍GauntletScore，这是一个证据引擎，其少量规则和阈值仅在基准的开发集上设定。它向来自不同提供商的独立读者提出类型化问题，并将每个答案作为一项测量记录在证据账本中。它采用固定的贝叶斯规则组合这些测量结果，只有当两个读者一致同意时才判定为矛盾。在完整的LLM-AggreFact测试集（涵盖11个数据集的29,320条声明）上，按照排行榜的方法，GauntletScore取得了79.51的平衡准确率，高于截至2026年11月15日排行榜上发布的所有平均值，使其与其中最佳的模型持平，即nimble-v4-max（其发布的数据集得分为79.45；该引擎的95%置信区间为78.47至80.53，包含了该值）。

    arXiv:2611.22027v1 Announce Type: cross  Abstract: Checking whether a generated sentence is supported by its source document is a standard safeguard for AI-written text, and the strongest published checkers are single models. We describe GauntletScore, an evidence engine whose few rules and thresholds were set on the benchmark's development split. It puts typed questions to independent readers from different providers and records each answer as a measurement in an evidence ledger. It combines the measurements with a fixed Bayesian rule that counts a contradiction only when two readers agree. On the full LLM-AggreFact test split (29,320 claims across 11 datasets), following the leaderboard's approach, GauntletScore achieves 79.51 balanced accuracy, above every average published on the leaderboard as of 15 November 2026. That places it level with the best of them, nimble-v4-max (79.45 from its published dataset scores; the engine's 95% interval, 78.47 to 80.53, contains it). On the same 
    
[^191]: 计算置换网格编码：基于两次乘法和一个64字节表的近最优量化

    Computed Permutation Trellis Coding: Near-Optimal Quantization from Two Multiplications and a 64-Byte Table

    [https://arxiv.org/abs/2611.22019](https://arxiv.org/abs/2611.22019)

    提出CPTC量化格式，其解码器仅需两次乘法和一个64字节表即可实现近最优的低位宽张量量化，在4比特时距信息论理想仅差0.07比特，同时支持1至6比特所有位宽并精确复现标准整数与浮点格式。

    

    我们提出了计算置换网格编码（CPTC），这是一种针对“一次写入、多次读取”张量的低位宽量化格式。诸如INT4和FP4这类对每个值独立进行舍入的格式，相对于信息论理想而言浪费了大量比特。网格编码通过对多个值进行联合编码来挽回这一损失，但现有的解码器要么需要大型查找表才能达到最佳质量，要么使用哈希来计算值，从而限制了其可解码的范围。CPTC的解码器将其状态乘以两个系数，并将两个乘积的高位比特相加以索引一个64字节的表。由于该和并非均匀分布，小表中很少被使用的边缘位置可以存放分布尾部很远的值。一个解码器即可覆盖从1到6的所有位宽，能精确复现标准的整数和浮点格式，并在4比特时距理想值仅差0.07比特。它的性能可以与表规模大数千倍的随机置换网格编码相媲美。

    arXiv:2611.22019v1 Announce Type: cross  Abstract: We propose Computed Permutation Trellis Coding (CPTC), a low-bit quantization format for tensors written once and read many times. Formats that round each value on its own, such as INT4 and FP4, waste a large part of their bits against the information-theoretic ideal. Trellis codes recover this loss by encoding values jointly, but prior decoders either need a large lookup table for their best quality or compute values with a hash that restricts what they can decode. CPTC's decoder multiplies its state by two coefficients and adds the top bits of the products to index a 64-byte table. Because that sum is not uniformly distributed, the rarely used ends of a small table can hold values far out in the tails. One decoder serves every bit width from 1 to 6, reproduces the standard integer and floating-point formats exactly, and comes within 0.07 bits of the ideal at 4 bits. It matches a random-permutation trellis code whose table is thousand
    
[^192]: SkillBuffBench：衡量智能体技能在框架、模型、技能访问模式与记忆等方面的贡献

    SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory

    [https://arxiv.org/abs/2611.22017](https://arxiv.org/abs/2611.22017)

    SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。

    

    智能体技能是一类基于文件的程序性知识包，编码智能体可以按需发现并加载，目前各大主流智能体框架均已支持。然而，关于技能能带来多大帮助的证据，几乎完全来自这样的设定：为任务人工挑选正确的技能，且智能体从空白状态开始。这两点在真实部署中都不成立——部署时智能体需要从已安装的技能库中自行选择，并且经常携带此前工作中积累的记忆。我们提出了 SkillBuffBench，这是一个基准测试及配套的配对评估协议，它将技能访问视为一个三级因素（无技能、精选技能、以及从已安装库中自动调用），并与记忆（开启与关闭）、四个智能体框架以及多个模型（来自 Anthropic、Google 和 OpenAI 的最先进专有模型，以及来自中国厂商的最先进开源权重模型）进行交叉组合，每个模型都在每个框架下运行……

    arXiv:2611.22017v1 Announce Type: new  Abstract: Agent Skills, file-based packages of procedural knowledge that a coding agent can discover and load on demand, are now shipped by every major agent harness, yet the evidence for how much they help comes almost entirely from settings in which the right Skill is hand-picked for the task and the agent starts from a blank slate. Neither holds in deployment, where an agent chooses from whatever library happens to be installed and frequently carries memory from earlier work. We present SkillBuffBench, a benchmark and paired evaluation protocol that treats Skill access as a three-level factor (no Skill, a curated Skill, and automatic invocation from an installed library), crosses it with memory (off and on), with four agent harnesses and with multiple models (the state-of-the-art proprietary models from Anthropic, Google and OpenAI, and the state-of-the-art open-weight models from Chinese providers), every model being run under every harness, a
    
[^193]: 论形式化反事实解释

    On Formal Counterfactual Explanations

    [https://arxiv.org/abs/2611.22010](https://arxiv.org/abs/2611.22010)

    本文将最近世界语义下的反事实解释刻画为析取范式，并提出合取反事实解释（CFXps）这一新概念，建立了CXps、AXps与CFXps之间的结构联系，从而架起了反事实解释与形式化XAI之间的桥梁。

    

    反事实解释通常被视为在最近世界语义下的反事实条件句，这需要对所有最小距离的反例进行推理。相比之下，形式化可解释人工智能（XAI）通常依赖于基于特征的溯因解释和对比解释，这两类解释支持高效计算并具有最小命中集对偶性。本文通过以下两种方式连接这两种视角：(i) 将最近世界反事实解释明确刻画为所有最小距离反例实例之上的析取范式；(ii) 引入合取反事实解释，这是反事实前件的一个子类，限制为能保证预测变化的特征文字的合取。论文建立了连接CXps、AXps与CFXps的结构性结果：每个CXp都为其见证实例诱导出一个AXp，并且每个CXp与一个见证AXp共同诱导出一个支持……（原文摘要在此处截断）

    arXiv:2611.22010v1 Announce Type: new  Abstract: Counterfactual explanations are often seen as counterfactual conditionals under closest-world semantics, requiring one to reason over all minimally distant counterexamples. In contrast, formal XAI typically relies on feature-based abductive explanations (AXps) and contrastive explanations (CXps), which admit efficient computation and a minimal hitting-set duality. This paper bridges these perspectives by (i) characterizing closest-world counterfactual explanations explicitly as a disjunctive normal form over all minimally distant counterexample instances, and (ii) introducing conjunctive counterfactual explanations (CFXps), a subclass of counterfactual antecedents restricted to conjunctions of feature literals that guarantee prediction change. The paper establishes structural results connecting CXps, AXps and CFXps: every CXp induces an AXp for its witness instance, and every CXp together with a witness AXp induces a CFXp whose support i
    
[^194]: DiagTrace：一个用于临床决策的交互式基准测试

    DiagTrace: An Interactive Benchmark for Clinical Decision Making

    [https://arxiv.org/abs/2611.22009](https://arxiv.org/abs/2611.22009)

    DiagTrace是一个基于PubMed Central病例报告构建的交互式临床决策基准，包含510个跨八个专科的病例，用于评估大语言模型能否通过交互式信息获取从不完整信息中逐步重建诊断轨迹，突破了传统一次性提供完整患者信息的评估方式。

    

    大语言模型（LLM）在医学问答和诊断推理基准测试中展现出强大的性能。然而，现有的评估通常预先提供患者的全部信息，并且仅评估最终诊断或推理理由。这种设置忽略了真实临床诊断中需要从不完整信息出发逐步获取诊断证据的需求。为弥补这一差距，我们提出了DiagTrace，这是一个基于真实文献来源的基准测试，用于评估LLM能否通过交互式信息获取来重建已被记录的诊断轨迹。每个基准实例均基于PubMed Central病例报告构建，包含初始临床表现、事件库、临床病程和诊断参考资料。DiagTrace包含覆盖八个专科的510个病例，其中225个用于评估四个LLM在临床病程对齐度、交互质量和效率方面的表现。结果显示模型在覆盖范围与信息源一致性等方面存在权衡。

    arXiv:2611.22009v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong performance on medical question answering and diagnostic reasoning benchmarks. However, existing evaluations typically provide all patient information upfront and assess only the final diagnosis or rationale. This setup ignores the need to acquire diagnostic evidence incrementally from incomplete information in real-world diagnosis. To bridge this gap, we introduce DiagTrace, a source-grounded benchmark for evaluating whether LLMs can reconstruct the documented diagnostic trajectory through interactive information seeking. Each benchmark instance is built from a PubMed Central case report and includes an initial presentation, an event bank, a clinical course, and diagnostic references. DiagTrace contains 510 cases across eight specialties, with 225 used to evaluate four LLMs on clinical-course alignment, interaction quality, and efficiency. Results reveal trade-offs in coverage, source-con
    
[^195]: KV-LTM：将会话级KV缓存作为个人助理的持久长期记忆

    KV-LTM: Persistent Per-Session KV Caches as Long-Term Memory for Personal Assistants

    [https://arxiv.org/abs/2611.21998](https://arxiv.org/abs/2611.21998)

    KV-LTM提出将每个会话的原始文本与KV缓存一同作为长期记忆存储，通过直接恢复缓存会话来回答问题，避免了传统记忆系统中问题与证据之间的推导和选择环节，并作为检索与索引的补充，充当免重算的prefill加速器。

    

    个人助理会积累长达数月的对话记录，但当今的长期记忆系统在问题和其证据之间设置了推导或选择环节。在LongMemEval基准上的系统级比较中，两个商业助理的得分远低于同一模型离线阅读历史记录的表现[1]，这与写入和读取之间的内容丢失相一致，但并未将其隔离出来。我们提出KV-LTM：一个记忆层，在固定的读取模型下存储每个会话的原始文本及其KV缓存，通过恢复缓存的会话并追加问题来回答查询。KV-LTM是对检索和索引的补充，而非替代。文本是记忆的正式记录；缓存则是一种无需重新计算的prefill加速器，并能优雅地降级回纯文本模式。我们明确了其设计：存储记录、基于预算的保留策略、仅追加的读取路径，以及有界聚合。我们对其进行了分析性评估：在固定的BF16全注意力（GQA）和潜缓存MLA配置下，每个（摘要在此处截断）

    arXiv:2611.21998v1 Announce Type: new  Abstract: Personal assistants accumulate months of conversations, but today's long-term memory stacks put derivation or selection between a question and its evidence. In a system-level comparison on LongMemEval, two commercial assistants scored well below the same model reading the history offline [1], consistent with, but not isolating, content loss between write and read. We propose KV-LTM: a memory tier storing each session's raw text plus its KV cache under a fixed reader model, answering by restoring cached sessions and appending the question. KV-LTM is complementary to, not a replacement for, retrieval and indexing. Text is the memory of record; the cache is a recompute-free prefill accelerator, degrading gracefully to text. We specify the design: storage record, budget-derived retention, append-only read path, and bounded aggregation. We size it analytically: across pinned BF16 full-attention (GQA) and latent-cached MLA configurations, per-
    
[^196]: 在嵌入空间中对文档与章节层次结构建模

    Modeling Document and Section Hierarchies in Embedding Space

    [https://arxiv.org/abs/2611.21980](https://arxiv.org/abs/2611.21980)

    该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。

    

    嵌入式主题模型将词嵌入整合到主题生成过程中，能够产生连贯的主题，并在大词表上表现出稳健的性能。然而，每篇文档仅由单个主题分布表示，忽略了其内部结构。许多现实世界的语料库，包括法律文档、公司和政府报告以及在线讨论论坛，都呈现出层次化的组织结构：文档由多个章节组成，而这些章节往往涉及不同的子主题。我们提出了层次化嵌入式主题模型，这是一种结构化主题模型，通过嵌入空间中的层次先验将文档的主题分布与其各章节的主题分布相耦合。我们为 hETM 推导了一种摊销变分推断过程，并在来自上述各领域的结构化语料库上对其进行评估。实验结果表明，hETM 在扁平模型与……之中取得了最低或接近最低的留出集困惑度（摘要原文在此处截断）。

    arXiv:2611.21980v1 Announce Type: cross  Abstract: Embedded topic models integrate word embeddings into the topic generation process, producing coherent topics and robust performance on large vocabularies. However, each document is represented by a single topic distribution, ignoring its internal structure. Many real world collections, including legal documents, company and government reports, and online discussion forums, exhibit hierarchical organization in which documents consist of multiple sections that often address distinct subtopics. We introduce the Hierarchical Embedded Topic Model (hETM), a structured topic model that couples the topic distribution of a document and those of its sections through a hierarchical prior in the embedding space. We derive an amortized variational inference procedure for hETM and evaluate it on structured collections from each of these domains. Experimental results show that hETM achieves the lowest or near-lowest held-out perplexity among flat and
    
[^197]: WebAgents：面向智能体原生Web的智能体创建、托管与执行解耦方案

    WebAgents: Unbundling Agent Creation, Hosting, and Execution for an Agent-Native Web

    [https://arxiv.org/abs/2611.21978](https://arxiv.org/abs/2611.21978)

    该论文提出WebAgents系统，通过签名的agent.pkg包将智能体的创建、托管与执行三种角色解耦，使智能体工作流能够在不绑定特定执行环境、不转移用户资源控制权的前提下跨平台移植运行。

    

    大语言模型正在从孤立的助手转变为能够规划、使用工具并与其他智能体协作的自主智能体，这推动了一个由独立开发的智能体和服务构成的新兴“智能体Web”的兴起。实现这一愿景需要智能体行为能够在创建者、托管者和运行时之间迁移，而无需将发布与特定执行环境绑定，也无需转移用户资源的控制权。我们提出了WebAgents，它通过一个签名的agent.pkg携带可移植的工作流、技能、模式（schema）和资源需求，从而解耦这些角色。基于OpenClaw的智能体原生浏览器/运行时负责验证该包、获取授权，并在运行时将其绑定到用户控制的模型、工具、权限和预算上，而托管方分发公共程序并可在服务端保留私有服务。我们通过三个互不相关的工作流在本地和API支持的环境下未经修改地执行，展示了这一契约。

    arXiv:2611.21978v1 Announce Type: new  Abstract: Large language models are moving from isolated assistants toward autonomous agents that can plan, use tools, and coordinate with others, motivating an emerging Agentic Web of independently developed agents and services. Realizing this vision requires agent behavior to move across creators, hosts, and runtimes without coupling publication to a particular execution environment or transferring authority over user resources. We introduce WebAgents, which unbundles these roles through a signed agent.pkg that carries portable workflows, skills, schemas, and resource requirements. An OpenClaw-based agent-native browser/runtime verifies the package, obtains authorization, and binds it at run time to user-controlled models, tools, permissions, and budgets, while hosts distribute the public program and may retain private services server-side. We demonstrate this contract with three unrelated workflows executed unchanged under local and API-backed 
    
[^198]: 用于合成先验与零样本检测的病理随机化方法

    Pathology Randomization for Synthetic Priors and Zero-Shot Detection

    [https://arxiv.org/abs/2611.21975](https://arxiv.org/abs/2611.21975)

    该论文提出PathoDraw，一种通过在对比度无关渲染前将病灶条件分布随机绘制到标签图中的脑部MRI生成模型，从而无需真实患者数据即可实现病理合成先验与零样本异常检测。

    

    医院脑成像档案库中保存着远超前瞻性研究规模的多样化病理样本，尤其涵盖代表性不足的人群和疾病。为了充分利用未经整理的临床数据并扩大分析规模，需要能够跨对比度和分辨率检测异常的方法。合成数据已为健康解剖结构的多种人类神经影像任务提供了鲁棒的、与对比度无关的模型，但病理检测以及临床环境中的其他任务仍需依赖真实患者数据、对比度特定模型或疾病特定生成器。我们提出PathoDraw，一种包含病理的脑部MRI生成模型。在与对比度无关的渲染之前，将基于健康解剖结构的病灶条件分布绘制到标签图中，并由单一网络结合输入扫描的精确带符号强度偏差来估计健康解剖结构。该与对比度无关的3D UNet在病灶条件下学习健康标签……

    arXiv:2611.21975v1 Announce Type: cross  Abstract: Hospital brain imaging archives hold a diverse array of pathology at sample sizes beyond prospective studies, particularly for underrepresented populations and diseases. To unlock uncurated clinical data and scale analysis, methods are needed to detect abnormalities across contrasts and resolutions. Synthetic data has provided robust, contrast-agnostic models for a variety of human neuroimaging tasks in healthy anatomy, but pathology detection and other tasks in clinical settings still require real patient data, contrast-specific models, or disease-specific generators. We propose PathoDraw, a generative model of brain MRI with pathology. A conditional distribution of lesions given the healthy anatomy is drawn into the label map before contrast-agnostic rendering, and a single network estimates healthy anatomy with exact signed intensity deviation of any input scan. The contrast-agnostic 3D UNet learns the healthy labels under the lesio
    
[^199]: OCCAM：改进的单细胞表示学习与扰动效应预测

    OCCAM: Improved Single-Cell Representation Learning and Perturbation Effect Prediction

    [https://arxiv.org/abs/2611.21959](https://arxiv.org/abs/2611.21959)

    Occam是一个结合去噪细胞表示与基因相互作用、文本及蛋白质序列特征的MLP框架，在单细胞表示学习和跨环境基因扰动效应预测方面均达到了最优性能。

    

    预测基因扰动的效应需要能够在已观察到的细胞环境和基因组合之外进行泛化的模型。我们提出了Occam，一个用于表示学习和扰动效应预测的多层感知机（MLP）框架。Occam将通过去噪学习到的细胞表示与基因相互作用、文本以及蛋白质序列特征相结合。它在扰动信息和保留细胞类型（含批次整合）方面获得了最高的聚合表示得分。在跨环境少样本预测任务中，某个扰动和某个细胞环境在训练期间各自被单独观察到过，但从未同时出现。与四个重新训练的竞争模型相比，以及与从其他训练细胞系直接迁移同一扰动的实测效应相比，Occam更好地捕捉了这些未观察组合的实测响应模式。在无需微调的情况下，Occam在所有被评估的模型中对扰动的区分能力最强……

    arXiv:2611.21959v1 Announce Type: cross  Abstract: Predicting the effects of genetic perturbations requires models that generalize beyond observed cell contexts and gene combinations. We introduce Occam, a multilayer perceptron (MLP) framework for representation learning and perturbation-effect prediction. Occam combines cell representations learned through denoising with gene interaction, text, and protein-sequence features. It achieves the highest aggregate representation scores for perturbation information and cell-type preservation with batch integration. In cross-context few-shot prediction, a perturbation and a cell context are each observed during training, but never together. Occam better captures the measured response patterns for these unobserved combinations than four retrained competitors or direct transfer of the same perturbation's measured effects from other training cell lines. Without fine-tuning, Occam best distinguishes perturbations among the evaluated models and si
    
[^200]: 面向拖拉机导航控制器的权重空间编程：大语言模型原生编写可驾驶真实拖拉机的神经网络权重

    Weight-space programming for tractor guidance controller: large language models natively write neural network weights that steer a real tractor

    [https://arxiv.org/abs/2611.21958](https://arxiv.org/abs/2611.21958)

    大语言模型通过权重空间编程直接读取、修复并从零编写拖拉机导航控制器的神经网络权重，无需优化器或梯度下降，在模拟和真实拖拉机上均实现了成功的路径跟踪。

    

    权重空间编程是一种由大语言模型在自然语言规范的指导下直接读取和写入神经网络权重的过程，无需优化器或梯度下降。本研究将其应用于拖拉机自动导航任务：一个仅含253个参数的神经网络沿直线和圆弧路径行驶，并在机器条件发生变化时（例如转向执行器变慢）需要进行调整。GPT-6 Astra和Claude Fable 5.1（以及5.2）以少量模拟评估为预算，为我们的研究用拖拉机编辑网络，修复了原本为另一台拖拉机训练的网络，并将控制器直接写入一个零权重初始化的网络中。我们在模拟和真实拖拉机上将其与进化策略、梯度下降和纯追踪算法进行了对比。两个模型都能正确读取网络并运用控制理论准确分析闭环系统，而Fable在不借助模拟的情况下进行修复仅偶尔成功。

    arXiv:2611.21958v1 Announce Type: cross  Abstract: Weight-space programming is a process where a large language model reads and writes the weights of a neural network directly, guided by a natural language specification, without an optimiser or gradient descent. This study applies it to automatic tractor guidance, where a 253-parameter neural network steers along paths of lines and arcs and must be adapted when the machine changes, such as a slower steering actuator. GPT-6 Astra and Claude Fable 5.1 (also 5.2) edited the network for our research tractor using a small budget of simulated evaluations to repair a network trained for another tractor and write a controller into a network with zero weights. We compared them with an evolution strategy, gradient descent and pure pursuit in simulation and on the real tractor. Both models read the network and analysed the closed loop with control theory correctly, and Fable's repair without simulation succeeded only occasionally. With simulated 
    
[^201]: 十分之八的大语言模型会泄露系统提示词机密：Nevan——一项代理式夺旗（Capture-the-Flag）研究

    Eight in Ten LLMs Leak System-Prompt Secrets: Nevan, an Agentic Capture-the-Flag Study

    [https://arxiv.org/abs/2611.21956](https://arxiv.org/abs/2611.21956)

    该研究通过代理式夺旗实验审计了 363 个大语言模型，发现约八成模型会因提示注入攻击（尤其是转向法语等非英语语言）而泄露系统提示词中的机密，暴露了当前模型在指令层级防护上的普遍薄弱性。

    

    我们对来自 OpenRouter 目录的 363 个文本输入/文本输出模型的指令层级（instruction-hierarchy）执行情况进行了审计。本实验所回答的研究问题是：哪些模型会泄露嵌入在系统提示词中的受保护机密，以及通过哪些技术手段实现。每个模型在其系统提示词中守护一个 16 字符的秘密旗标，而一个由人类研究员引导的人工智能（AI）代理（OpenCode，由 DeepSeek V4.1 Flash 驱动）则试图说服该模型吐出旗标。在 363 个模型中，有 296 个模型的旗标被窃取，加上初始目录冻结后发布的四个模型，共计 300 次成功夺取。其中，170 次涉及非英语语言转换，最常见的是法语。每一次成功夺取都表明，该模型至少有一次将用户请求置于其系统指令之上。这一结果并不能证明该模型可证实地抵御了未经测试的攻击，也不能证明它会持续泄露语义敏感的数据，也不能证明它……

    arXiv:2611.21956v1 Announce Type: cross  Abstract: We audit instruction-hierarchy enforcement across 363 text-in/text-out models from the OpenRouter catalog. The research question we answer with this experiment is which models disclose a protected secret embedded in the system prompt, and through which techniques. Each model guards a secret 16-character flag set in its system prompt, while an artificial intelligence (AI) agent (OpenCode, powered by DeepSeek V4.1 Flash), steered by a human researcher, tries to persuade it to emit the flag. The flag was exfiltrated from 296 of the 363 models, plus four models released after the initial catalog freeze, for 300 captures in total. Of these, 170 involved a non-English language pivot, most often French. Each capture shows that a model ranked a user request above its system instruction at least once. It does not show that the model provably resists untested attacks, nor that it would consistently leak semantically sensitive data, nor that it i
    
[^202]: ThermLeT：基于Transformer的2.5D芯粒架构温度预测

    ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture

    [https://arxiv.org/abs/2611.21954](https://arxiv.org/abs/2611.21954)

    ThermLeT是一个基于Transformer的多模态代理模型，能在13毫秒内精准预测2.5D芯粒架构的稳态温度，比传统有限元求解器快约四个数量级，精度显著优于ThermEDGe和HotSpot。

    

    异构2.5D芯粒封装的发展带来了无与伦比的性能和定制化能力，但由此产生的高密度加剧了散热挑战。为了促进快速的设计空间探索，我们提出了ThermLeT，一个基于Transformer的多模态、尺度不变的代理模型，它集成功率密度图、空间坐标嵌入、材料特性和热传导速率，可在13毫秒内预测稳态温度，比ANSYS等传统有限元求解器快约四个数量级。为了训练该Transformer模型，我们还构建了一个包含6,221个芯粒布局的数据集，这些布局由三种代表性架构扰动生成。在150个样本的测试集上，ThermLeT实现了3.16°C的均方根误差（RMSE）和2.24°C的平均绝对误差（MAE），相比ThermEDGe准确度提高了35%，相比HotSpot提高了71%，同时推理延迟分别低约6倍和11倍。消融研究证实了其鲁棒的泛化能力。

    arXiv:2611.21954v1 Announce Type: cross  Abstract: The drive toward heterogeneous 2.5D chiplet packages promises unmatched performance and customization, yet the resulting density amplifies thermal challenges. To facilitate rapid design space exploration, we present ThermLeT, a multimodal transformer-based, scale invariant surrogate that integrates power density maps, spatial coordinate embeddings, material properties and heat transfer rate to predict steady state temperature in 13 ms, about four orders of magnitude faster than conventional finite element solvers such as ANSYS. To train the transformer model, we also create a dataset of 6,221 chiplet layouts perturbed from three representative architectures. On a 150 sample test set, ThermLeT achieves an RMSE of 3.16{\deg}C and an MAE of 2.24{\deg}C, improving accuracy by 35% over ThermEDGe and by 71% over HotSpot, while running approximately 6x and 11x lower inference latencies respectively. Ablation studies confirm robust generalizat
    
[^203]: DiscoveryAgent：一个以科学家为中心的智能体AI框架

    DiscoveryAgent: A Framework for Scientist-centric Agentic AI

    [https://arxiv.org/abs/2611.21950](https://arxiv.org/abs/2611.21950)

    DiscoveryAgent是一个以科学家为中心、领域无关的AI智能体框架，它通过让科学家在各轮研究之间选择、完善或替换假设，并结合基于严格检查点系统的强大审计智能体来追溯声明、复现分析和自主发现错误，实现了自主探索与可扩展人类监督的平衡。

    

    AI智能体可以辅助科学研究，但如何在自主探索与科学家检查和调整正在进行的工作的能力之间取得平衡，仍然是一个核心设计挑战。我们提出了DiscoveryAgent，这是一个领域无关、以科学家为中心的系统，它将自主探索与科学家主导的假设引导相结合。科学家可以在研究轮次之间选择、完善或替换假设，同时审计智能体通过查询检查点记录来追溯声明、复现分析并为后续决策提供信息。在两个预测建模任务中，DiscoveryAgent在所评估的基线中产生了最佳的变星分类器，并且在科学家的引导下，其蛋白质变异效应预测器仅被106个ProteinGym基线中的一个超越。我们开发了一个严格的检查点系统，使高度强大的审计智能体成为可能，这是对智能体研究的一种可扩展的监督形式。审计智能体能够自主识别错误……

    arXiv:2611.21950v1 Announce Type: new  Abstract: AI agents can assist scientific research, but balancing autonomous investigation with scientists' ability to inspect and redirect ongoing work remains a central design challenge. We present DiscoveryAgent, a domain-agnostic, scientist-centric system that combines autonomous investigation with scientist-directed hypothesis steering. Scientists select, refine, or replace hypotheses between research rounds, while audit agents interrogate checkpointed records to trace claims, reproduce analyses, and inform subsequent decisions. Across two predictive modeling tasks, DiscoveryAgent produced the best variable-star classifier among evaluated baselines and, with scientist steering, a protein variant-effect predictor surpassed by only one of 106 ProteinGym baselines. We develop a rigorous checkpointing system that enables highly capable audit agents, a scalable form of oversight for agentic research. Audit agents autonomously identify errors at hi
    
[^204]: 人工智能监管中的诚实经纪人：公众意见如何向监管者引入证据

    Honest Brokers in AI Regulation: How Public Comments Introduce Evidence to Regulators

    [https://arxiv.org/abs/2611.21948](https://arxiv.org/abs/2611.21948)

    该研究通过计算分析美国40项潜在AI监管规则的2,664份公众意见发现，企业提交者引入的证据总体较少，而政府、大学和公民社会提交者提供了显著更多的研究和法律来源，表明非行业利益相关者可作为对抗行业俘获的“诚实经纪人”。

    

    人工智能的监管容易被行业行为者以各种形式“俘获”，这些行业行为者对政策制定过程拥有不当影响力，部分原因在于他们在开发这些技术方面具有信息优势。在本文中，我们研究了公众意见提交者在美国向潜在监管提供证据方面所扮演的角色。我们构建了一个包含40项潜在监管规则以及各类利益相关者提交的2,664份公众意见的语料库。我们对这些文本进行计算化处理，以正式引用和非正式引用的形式提取旨在为监管者提供参考的证据。我们发现所提供证据的类型和数量存在显著差异：在控制意见长度不变的情况下，企业意见提交者总体上引入的证据较少，而政府、大学和公民社会的意见提交者提供了显著更多的研究类和法律类来源。被引用研究的发表状态几乎……（摘要在此处截断）

    arXiv:2611.21948v1 Announce Type: new  Abstract: The regulation of Artificial Intelligence is susceptible to various forms of capture by industry actors that hold undue influence over the policymaking process, due in part to their information advantage in developing these technologies. In the following paper, we investigate the roles of public commenters in providing evidence to potential regulation in the USA. We build a corpus of 40 potential regulations, and 2,664 public comments submitted by a variety of stakeholders. We computationally process these texts to extract evidence meant to inform regulators, in the form of formal citations and informal references. We find significant differences in the type and amount of evidence supplied: Holding comment length constant, business commenters introduce altogether less evidence, and government, university, and civil society commenters provide significantly more research and legal sources. The publication status of cited research is nearly
    
[^205]: 守恒的判定，漂移的承诺：识别有害的过度思考

    Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking

    [https://arxiv.org/abs/2611.21946](https://arxiv.org/abs/2611.21946)

    该论文提出“承诺-判定差距”框架来识别推理模型中的有害过度思考，证明仅凭最终正确性探针估计的“判定”无法预测继续思考的利弊，而“承诺”与“判定”之差恰好量化了停止思考可获得的准确率增益，并给出将准确率-预算曲线分解为链内答案变化与链存活变化的恒等式。

    

    一个已经得出正确答案的推理模型，可能会因为继续思考而失去这个正确答案。决定是否停止，意味着在推理链的同一位置比较两个概率：模型若现在停止所给出的答案是正确的概率（即其“承诺”），以及模型最终完成时的答案是正确的概率（即其“判定”）。我们证明，仅凭判定——也就是最终正确性探针所估计的量——无法完成这种比较。作为对一个固定结果的预测，判定没有可预测的漂移，无论继续思考是有益还是有害。承诺与判定之差等于答案质量的期望剩余变化量，也正是停止思考所能获得的准确率增益。第二个恒等式则将任何准确率-预算曲线分解为两部分：推理链内部的答案变化，以及仍在运行的推理链集合的变化。我们在 MATH-500 上对这两者进行了测量。在 DeepSeek-R1-Distill-Qwen-1.5B 持有正确答案的全部 379 个点上（这些点的选取不考虑其推理链如何结束），继续……（原文摘要至此截断）

    arXiv:2611.21946v1 Announce Type: new  Abstract: A reasoning model that has reached a correct answer can lose it by continuing to think. Deciding whether to stop means comparing two probabilities at the same point of a chain: that the answer the model would give if stopped now is correct (its commitment), and that the answer it will finish with is correct (its verdict). We show that the verdict alone, which is what final-correctness probes estimate, cannot make this comparison. As a forecast of a fixed outcome it has no predictable drift, whether continuing helps or hurts. The commitment-verdict gap equals the expected remaining change in answer quality and the accuracy gained by stopping. A second identity splits any accuracy-versus-budget curve into answer change within chains and change in which chains are still running. We measure both on MATH-500. At all 379 points where DeepSeek-R1-Distill-Qwen-1.5B holds a correct answer, selected without regard to how its chain ends, continuing
    
[^206]: 面向无障碍自动驾驶车辆应用的轮椅配置分类深度学习模型基准测试

    Benchmarking Deep Learning Models for Wheelchair Configuration Classification in Accessible Automated Vehicle Applications

    [https://arxiv.org/abs/2611.21945](https://arxiv.org/abs/2611.21945)

    本文提出了轮椅配置图像数据集（WCID），通过对手动/电动、轮椅子类型、乘坐状态等多维度属性进行细粒度标注，为无障碍自动驾驶车辆应用中的轮椅配置分类深度学习模型提供了基准测试资源。

    

    要使自动驾驶车辆具备无障碍性，需要考虑弱势道路使用者，包括轮椅使用者。为了提供更贴合需求的体验以支持这一群体的独立出行，车辆设计需要适应种类繁多的轮椅配置。然而，轮椅具有个性化特点，在车架几何结构、驱动方式和外观上各不相同。此外，乘员或配件还可能遮挡视觉线索。现有的移动辅助设备数据集仅标注轮椅是否存在，而未描述其具体配置。为填补这一空白，本文提出了轮椅配置图像数据集（WCID），该数据集包含467张经过面部模糊处理的图像，涵盖192个采集组，仅用于评估。每张图像均标注有：(i) 手动或电动的二分类标签，(ii) 轮椅子类型，(iii) 乘坐状态、视角、场景环境、光照、遮挡、驱动方式和控制方式，并包含一个273张图像的验证集划分。

    arXiv:2611.21945v1 Announce Type: cross  Abstract: Making automated vehicles accessible requires consideration of vulnerable road users, including wheelchair users. To offer a more tailored experience that supports independence for this group, vehicle design needs to accommodate the wide range of wheelchair configurations. However, wheelchairs are individualized, varying in frame geometry, propulsion, and visual appearance. Moreover, occupants or accessories may obscure visual cues. Existing mobility-aid datasets only label the presence of a wheelchair, rather than describing its specific configuration. To fill this gap, this paper introduces the Wheelchair Configuration Image Dataset (WCID), comprising 467 face-blurred images across 192 capture groups solely for evaluation. Each image is annotated with (i) a two-class label of manual or powered, (ii) a chair subtype, (iii) occupancy, viewpoint, setting, lighting, occlusion, propulsion, and control, with a 273-image validation split an
    
[^207]: 模型是否真的使用了它所引用的内容？小型视频语言模型中接地推理链的反事实检验

    Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in Small Video-Language Models

    [https://arxiv.org/abs/2611.21940](https://arxiv.org/abs/2611.21940)

    本文提出一种无需标注的“引用差距”反事实检验方法，通过涂黑模型推理链中所引用的视频区域来验证小型视频语言模型是否真正依赖其引用的证据，结果显示模型在CLEVRER上具有较强证据依赖性，但在NExT-QA上依赖性很弱。

    

    接地推理链在每一步都引用一个帧区间和一个边界框，使读者能够检查模型观察的位置。然而，答案是否真正依赖于所引用的区域则是另一个问题。我们通过干预实验来检验这一点：将模型在其自身推理链中引用的区域涂黑，并将正确答案翻转的比例与在同一帧其他位置涂黑相同大小框的结果进行对比。这种成对差异被称为“引用差距”，且无需任何标注。我们将其应用于在精确CLEVRER推理链和伪标注NExT-QA推理链上训练的2-3B参数视频语言模型。在相同的模型权重下，引用差距在CLEVRER上为0.089，而在NExT-QA上仅为0.003。汇总所有模型的结果，遮蔽所引用的框在CLEVRER上导致翻转的正确答案数量是对照组的2.2倍，在NExT-QA上为1.1倍——即使仅考虑那些确实依赖视频的答案也是如此，同时这些引用框在两个数据集上均通过了接地性检查。而当在每一帧上遮蔽所引用的实体时

    arXiv:2611.21940v1 Announce Type: cross  Abstract: A grounded reasoning chain cites a frame span and a box for every step, so a reader can check where the model looked. Whether the answer depends on the cited region is a different question. We test it by intervention: the regions a model cited in its own chain are blacked out, and the share of correct answers that flip is compared with blacking out boxes of the same size elsewhere on the same frames. The paired difference, the cite gap, needs no annotation. We apply it to 2-3B video-language models trained on exact CLEVRER chains and on pseudo-labelled NExT-QA chains. With the same weights, the cite gap is 0.089 on CLEVRER and 0.003 on NExT-QA. Pooled over models, masking the cited boxes flips 2.2 times as many correct answers as the control on CLEVRER and 1.1 times as many on NExT-QA, also among the answers that depend on the video at all, while the boxes pass a grounding check on both. Masking the cited entity on every frame, instead
    
[^208]: SLED：从边缘到数据中心的传感器切片

    SLED: Sensor SLicing from Edge to Datacenter

    [https://arxiv.org/abs/2611.21937](https://arxiv.org/abs/2611.21937)

    本文提出SLED传感器切片技术，在O-RAN中为单个用户设备内的各传感器流提供确定性的MAC层隔离，解决批量数据传输挤占延迟敏感遥测流量的问题，并利用智能体编程将开发时间缩短400倍。

    

    现代传感系统会从多种传感器模态生成大量异构数据流，越来越依赖数据中心进行计算、人工智能/超级智能（AI/SI）推理以及AI/SI训练。每个传感器流可能根据其类型和数据量有不同的网络与计算需求。然而，当前的无线系统通常在设备层面分配资源，导致位于同一设备上的多个传感器流之间产生资源竞争，批量数据传输可能会“饿死”对延迟敏感的遥测流量。为此，本文提出了传感器切片技术。SL在开放无线接入网络（O-RAN）中为单个用户设备（UE）内的各个传感器提供确定性的、介质访问控制（MAC）级别的隔离。借助智能体编程技术，SL在两个符合O-RAN标准的软件包中得以实现，将开发时间缩短了400倍（2小时40分钟 [OCUDU] 对比 6.5周 [OAI]），228-f…（摘要原文在此处截断）

    arXiv:2611.21937v1 Announce Type: new  Abstract: Modern sensing systems generate large volumes of heterogeneous data streams from multiple sensor modalities, increasing reliance on datacenters for computation, artificial intelligence/superintelligence (AI/SI) inference, and AI/SI training. Each sensor stream may have different network and compute requirements depending on its type and data volume. However, current wireless systems typically allocate resources at the device level, leading to contention among co-located sensor streams, where bulk data transfers may starve latency-critical telemetry. To this end, this paper introduces Sensor Slicing (SL). SL provides deterministic, medium access control (MAC)-level isolation for individual sensors within a single user equipment (UE) in open radio access networks (O-RAN). Using agentic coding techniques, SL is implemented in two O-RAN-compliant packages, reducing development time by 400 times (2 h 40 min [OCUDU] vs. 6.5 weeks [OAI]), 228-f
    
[^209]: 基于区块链赋能6G技术与联邦学习的车对车（V2V）通信

    V2V Communications Using Blockchain-Enabled 6G Technology and Federated Learning

    [https://arxiv.org/abs/2611.21932](https://arxiv.org/abs/2611.21932)

    该研究提出了一种融合区块链、联邦学习与延迟资源分配优化的6G赋能V2V通信系统，在数据包投递率、模型准确率和延迟等指标上优于现有技术，并通过共识优化和k-匿名实现了安全与隐私的双重保障。

    

    本研究提出了一种新颖的车对车（V2V）通信方法，该方法集成了区块链技术、联邦学习（FL）以及延迟与资源的分配优化。研究采用多种性能指标（如数据包投递率（PDR）、模型准确率和延迟）对所提出的系统进行了评估，并证明了其相对于现有技术的优越性。此外，该系统通过共识优化增强了安全性，并通过k-匿名技术保护数据隐私。总体而言，所提出的系统是在网联和自动驾驶汽车时代实现高效、安全V2V通信的一种有前景的解决方案。此外，与传统5G相比，所提出的方法实现了更高的可靠性、更低的延迟和更好的资源利用率。

    arXiv:2611.21932v1 Announce Type: cross  Abstract: This study proposes an interesting approach for vehicle-to-vehicle (V2V) communication, which integrates blockchain technology, federated learning (FL), and allocation optimization of latency and resources. The research evaluates the proposed system using various performance metrics such as packet delivery ratio (PDR), model accuracy, and latency and demonstrates its superiority over existing techniques. Furthermore, the system provides enhanced security through consensus optimization and k-anonymity for data privacy. Overall, the proposed system is a promising solution for efficient and secure V2V communication in the era of connected and autonomous vehicles. Moreover, the proposed approach achieves higher reliability, lower latency, and better resource utilization compared to traditional 5G.
    
[^210]: DNeo：你的并行草稿模型需要更好的目标特征编码

    DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding

    [https://arxiv.org/abs/2611.21925](https://arxiv.org/abs/2611.21925)

    本文发现标准特征编码机制存在有损（残差混合破坏特征空间）、无记忆、均匀采样忽略重要隐藏状态三大缺陷，并提出特征混合注意力（MoFA）来改进推测-扩散解码中并行草稿模型的目标特征编码质量。

    

    推测-扩散解码使用一个小型并行草稿模型在一次前向传播中生成 b 个草词元，然后通过更大的目标模型对这些草词元进行并行验证。当多个草词元同时通过验证时，便可以实现更快的推理速度。近期的并行草稿架构（DFlash、DSpark）通过一种称为特征编码的机制，以目标模型计算出的丰富特征为条件来提升草稿质量。然而，标准的特征编码机制存在三个局限性：（1）它们是有损的，其应用的编码投影会通过一种被称为“残差混合”的效应破坏底层的目标特征空间；（2）它们是无记忆的，没有对之前处理过的提议的记录；（3）它们使用从目标模型各层均匀采样的特征，忽略了重要的隐藏状态。为了解决这些问题，我们提出了特征混合注意力（MoFA），一种……

    arXiv:2611.21925v1 Announce Type: new  Abstract: Speculative-diffusion decoding uses a small parallel drafter model to generate b draft tokens in one forward pass, which then undergo parallel verification via a larger target model. Faster inference can be achieved when multiple draft tokens pass verification simultaneously. Recent parallel draft architectures (DFlash, DSpark) improve draft quality by conditioning on rich features computed by the target, through a mechanism known as feature encoding. However, standard feature encoding mechanisms have three limitations: (1) they are lossy, applying encoding projections that are destructive to the underlying target feature-space via a discovered effect known as residual mixing, (2) they are memory-less, with no record of previously processed proposals, and (3) they utilize features sampled uniformly from across the target layers, omitting important hidden states. To address these issues, we propose Mixture-of-Features Attention (MoFA), a 
    
[^211]: ALOHA Lightning：学习快速而精准的机器人操作

    ALOHA Lightning: Learning Fast and Precise Manipulation

    [https://arxiv.org/abs/2611.21919](https://arxiv.org/abs/2611.21919)

    ALOHA Lightning系统通过动态示教高速收集人类演示，并结合测试时动作平滑与掩码视觉对齐的学习流程，实现了快速且精准的双臂机器人操作。

    

    从人类演示中学习使机器人能够掌握广泛的操作技能，但学习到的策略执行速度通常远慢于普通人。这种速度差距主要源于缺乏高速收集演示数据的接口，以及训练能够稳健执行高速动作的策略十分困难。在本文中，我们提出了ALOHA Lightning（闪电ALOHA），一个用于学习快速且精准机器人操作的系统。我们的系统采用动态示教在可反向驱动的双臂平台上直观地收集接近人类速度的演示，从而获得自然且快速的轨迹。我们还提出了一套学习流程，通过测试时动作平滑实现流畅的高速执行，并通过掩码技术对齐数据收集与部署之间的视觉数据分布。在每项任务仅需50条演示的情况下，ALOHA Lightning能够自主完成双臂任务。

    arXiv:2611.21919v1 Announce Type: cross  Abstract: Learning from human demonstrations has enabled robots to acquire a wide range of manipulation skills, but learned policies typically execute far slower than ordinary humans. This speed gap is mainly due to lack of an interface for collecting demonstration data at high speed, and the difficulty in training policies that can robustly execute high-speed motions. In this paper, we present ALOHA Lightning (Flash ALOHA), a system for learning fast and precise robotic manipulation. Our system uses kinesthetic teaching to intuitively collect near-human-speed demonstrations on a backdrivable bimanual platform, yielding natural and fast trajectories. We also present a learning pipeline that enables smooth high-speed execution through test-time action smoothing and aligns the visual data distribution between data collection and deployment with masking. Given 50 demonstrations for each task, ALOHA Lightning autonomously completes bimanual tasks su
    
[^212]: 面向延迟标签与分布漂移下数据流评估的时序代数框架

    A Temporal Algebraic Framework for Data Stream Evaluation Under Delayed Labels and Distribution Shifts

    [https://arxiv.org/abs/2611.21914](https://arxiv.org/abs/2611.21914)

    该论文提出将基于混淆矩阵的代数框架扩展到数据流评估，通过指数衰减或滑动窗口进行时间加权，并以观测掩码和校准概率处理延迟与选择性标签反馈，从而在分布漂移下以每步 O(m²) 的代价区分决策边界退化与类别流行度变化。

    

    在数据流中，标签常常延迟到达或仅对部分实例可用，类别流行度会发生漂移，稀有类别会在短时间窗口中消失。此时，标准评估方法会将决策边界的退化与流行度的变化相混淆，并且在标签延迟或由模型选择性提供时产生偏差。我们将一个能够从单一混淆矩阵获得汇总（微平均）与逐类别（宏平均）度量的代数框架扩展至数据流场景，通过指数衰减或滑动窗口对计数进行时间加权；无论数据流长度如何，每步更新代价仅为 O(m^2)，且每个汇总度量始终保持为逐类别度量的加权组合。延迟和选择性反馈以观测掩码的形式纳入框架：在标签到达之前，实例以模型校准后的类别概率进行计数，该临时估计的偏差由其校准误差决定。经审计的约定对没有已验证实例的类别不报告任何数值。我们……

    arXiv:2611.21914v1 Announce Type: cross  Abstract: In data streams, labels often arrive late or only for some instances, class prevalence drifts, and rare classes vanish from short windows. Standard evaluation then confuses a degraded decision boundary with a change in prevalence, and is biased when labels are delayed or selected by the model. We extend an algebraic framework that obtains pooled (micro) and per-class (macro) measures from one confusion matrix to streams, weighting the counts in time by exponential fading or a sliding window; the update costs O(m^2) per step, whatever the stream length, and each pooled measure stays a weighted combination of the per-class ones. Delayed and selective feedback enter as an observation mask: until its label arrives, an instance is counted with the model's calibrated class probabilities, and the bias of this interim estimate is driven by their calibration error. Audited conventions report no value for a class without verified instances. We s
    
[^213]: 面向4K面部资产贴图的领域微调扩散超分辨率方法

    Domain-Tuned Diffusion Super-Resolution for 4K Facial Asset Maps

    [https://arxiv.org/abs/2611.21890](https://arxiv.org/abs/2611.21890)

    提出一种面向面部资产贴图的领域微调扩散上采样方法，通过轻量级LoRA适配冻结的SDXL骨干并结合Tile ControlNet保持结构，实现1024到4096的4倍超分辨率，将反照率LPIPS从0.21降至0.08且避免位移贴图的几何失真。

    

    基于物理渲染的面部资产在电影级特写镜头中需要4K分辨率的反照率贴图和位移贴图，然而生成式流程通常只能产出1K分辨率的贴图。简单套用照片超分辨率方法并不适用：面部UV贴图具有特定领域的统计特性，而位移是一种标量高度场，其误差会直接导致几何形变。我们提出了一种针对4倍（1024到4096）面部资产超分辨率的领域微调扩散上采样器。从双线性上采样的贴图出发，我们针对每种模态使用轻量级LoRA模块对冻结的SDXL骨干网络进行适配，同时利用Tile ControlNet保持输入结构，由扩散先验恢复毛孔与皱纹级别的细节。在保留测试集上，我们的方法相比双线性插值以及现成的GAN和扩散超分辨率模型，生成了更清晰、更忠实的细节，将反照率LPIPS从0.21降低到0.08，同时避免了位移浮雕的错误错位或过度起伏。这些结果表明，轻量级领域微调……（原文摘要在此处被截断）

    arXiv:2611.21890v1 Announce Type: cross  Abstract: Physically based facial assets require 4K albedo and displacement maps for cinematic close-ups, yet generative pipelines commonly produce them at only 1K. Naively applying photographic super-resolution is a poor fit: facial UV maps have domain-specific statistics, and displacement is a scalar height field whose errors directly distort geometry. We present a domain-tuned diffusion upscaler for 4x (1024 to 4096) facial asset super-resolution. Starting from a bilinearly upsampled map, we adapt a frozen SDXL backbone per modality with lightweight LoRA modules, while a Tile ControlNet preserves input structure as the diffusion prior restores pore- and wrinkle-scale detail. On held-out maps, our method produces sharper, more faithful detail than bilinear and off-the-shelf GAN and diffusion super-resolvers, reducing albedo LPIPS from 0.21 to 0.08 while avoiding misplaced or excessive displacement relief. These results show that lightweight do
    
[^214]: 语言模型能够改写自身的权重

    Language Models Can Rewrite Their Own Weights

    [https://arxiv.org/abs/2611.21885](https://arxiv.org/abs/2611.21885)

    该论文提出“自我改写”方法，让27B语言模型仅凭自身推理、无需梯度或训练，诊断失败原因并直接编辑自身权重，在预先固定的490道未见题目上准确率提升6.6分，并在多次自我改写会话中持续提升表现。

    

    语言模型通常只训练一次，然后被冻结并部署：若要再次改进，就需要重新进行一次训练；而在推理阶段使用的提示、脚手架或额外计算，一旦任务结束便会消失。已有研究表明，一个前沿模型能够编辑一个远小于它的模型的权重。我们提出疑问：模型能否改写自己的权重，让自己变得更好？我们发现它可以——仅凭自身的推理，无需梯度、无需训练；我们将此称为“自我改写”。在给定自身权重、有评分的答案以及运行自身的能力的条件下，一个27B模型能够诊断自己的失败，并编辑导致失败的那些权重。它超越了自己：它自己做出的编辑使其在490道从未见过的问题上（测试提前固定）准确率提高了6.6个百分点，而另一个会话所做的编辑也能使其显著变好。它会持续变好：只有当自己的测试支持时它才会提交编辑，每次编辑替换上一次的编辑，在其前八次这样的会话中，其准确率只升不降。

    arXiv:2611.21885v1 Announce Type: new  Abstract: Language models are trained once, frozen and deployed: improving one again takes another training run, and a prompt, a scaffold or extra computation at inference is gone once the task ends. A frontier model has been shown to edit the weights of a much smaller one. We ask whether a model can rewrite its own weights to make itself better, and find that it can, by its own reasoning, with no gradients and no training; we call this self-rewriting. Given its weights, graded answers and the ability to run itself, a 27B model diagnoses its failure and edits the weights behind it. It surpasses itself: its own edit makes it 6.6 points more accurate on 490 problems it never saw, in a test fixed in advance, and another session's edit also makes it significantly better. It keeps getting better: it submits an edit only when its own tests support it, each edit replacing the last, and over its first eight such sessions its accuracy only rises. Unlike te
    
[^215]: 用于安全控制器选择的无抵消约束SquareCB算法

    Cancellation-Free Constrained SquareCB for Safe Controller Selection

    [https://arxiv.org/abs/2611.21880](https://arxiv.org/abs/2611.21880)

    该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。

    

    我们研究约束上下文老虎机问题，其中学习者在每一轮从 $K$ 个动作中选择一个，以最小化损失，同时使其动作的期望成本保持在预算以下。现有的到在线回归的归约方法使用从过去约束残差中学习到的拉格朗日乘子对成本进行加权，因此某些轮次的松弛量可能会抵消其他轮次的违约。我们转而在成本上施加修正权重来运行 Foster 等人（2020）提出的 SquareCB：当估计成本在预算之内时权重为零，否则权重恰好足够大以将成本拉回预算。在可实现性假设以及具有边际 $\delta$ 的 Slater 条件下，该算法达到遗憾 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 和无抵消违约 $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，其中 $\mathrm{Reg}_{\mathsf{Sq}}$ 是回归预言机的误差。当预算必须在（原文摘要在此处截断，后续内容缺失）……

    arXiv:2611.21880v1 Announce Type: cross  Abstract: We study constrained contextual bandits, in which a learner chooses one of $K$ actions in each round to minimize loss while keeping the expected cost of its play below a budget. Existing reductions to online regression weight the cost by a Lagrange multiplier learned from past constraint residuals, so slack in some rounds can offset violations in others. We instead run SquareCB proposed in \citep{foster2020beyond} with a rectified weight on the cost: it is zero when the estimated cost is within the budget, and otherwise just large enough to bring it back to the budget. Under realizability and Slater's condition with margin $\delta$, the algorithm attains regret $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ and cancellation-free violation $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$, where $\mathrm{Reg}_{\mathsf{Sq}}$ is the error of the regression oracle. The same bounds hold when the budget must be met in
    
[^216]: FaceRiff：基于几何条件的高效数据统一文本到3D人脸资产生成框架

    FaceRiff: Data-Efficient Unified Text-to-3D Face Asset Synthesis via Geometric Conditioning

    [https://arxiv.org/abs/2611.21867](https://arxiv.org/abs/2611.21867)

    FaceRiff提出了一种数据与参数高效的统一文本到3D人脸资产生成框架，通过在预训练SDXL上添加任务特定LoRA适配器，并以UV法线图作为几何条件，在共享UV空间中直接从文本生成包含网格、反照率和置换贴图的完整3D人脸资产。

    

    具有一致的UV映射几何、反照率和基于物理的置换贴图的高分辨率人脸数据集十分稀缺，然而这类资产对于照片级真实感的数字人渲染至关重要。现有的3D人脸生成方法通常速度慢、局限于单一模态，或需要大规模的专门数据集。与此同时，大型图像扩散模型编码了强大的人脸先验，但其本身无法原生生成与光照无关的纹理、毛孔级置换贴图或UV对齐的几何结构。我们提出了FaceRiff，这是一个统一且数据与参数高效的框架，能够在共享UV空间中直接从文本生成完整的3D人脸资产（网格、反照率和置换贴图）。FaceRiff通过任务特定的LoRA适配器对预训练的SDXL主干网络进行适配，并使用UV法线图作为几何条件：每个实例的法线图引导反照率和置换贴图的合成，而数据集平均法线则为网格生成提供规范的空间锚点。仅使用……

    arXiv:2611.21867v1 Announce Type: cross  Abstract: High-resolution facial datasets with consistent UV-mapped geometry, albedo, and physically based displacement are scarce, yet such assets are essential for photorealistic digital-human rendering. Existing 3D face generation methods are often slow, single-modality, or require large specialized datasets. Meanwhile, large image diffusion models encode strong facial priors but do not natively produce lighting-independent textures, pore-scale displacement, or UV-aligned geometry. We present FaceRiff, a unified, data- and parameter-efficient framework for generating complete 3D face assets (mesh, albedo, and displacement) directly from text in a shared UV space. FaceRiff adapts a pretrained SDXL backbone with task-specific LoRA adapters and uses UV normal maps as geometric conditioning: per-instance normals guide albedo and displacement synthesis, while a dataset-mean normal provides a canonical spatial anchor for mesh generation. Using only
    
[^217]: PromptPack：通过免训练方式增强扩散编辑器，实现与意图对齐的空间与语义控制

    PromptPack: Training-Free Augmentation of Diffusion Editors for Intent-Aligned Spatial and Semantic Control

    [https://arxiv.org/abs/2611.21865](https://arxiv.org/abs/2611.21865)

    PromptPack 通过免训练地为扩散编辑器引入“保留/允许/修改”像素角色划分和区域到提示的空间映射两种显式编辑指南，实现了与用户意图对齐的空间与语义精确控制。

    

    arXiv:2611.21865v1 公告类型：cross 摘要：许多图像编辑任务要求将修改限制在预期的空间区域内，同时精确保留用户指定的源内容。基于指令的编辑流程往往使这类用户意图未被充分明确，而修补类编辑器仅以明确定义的编辑支持区域为生成条件，却不考虑各个编辑概念的空间分配。PromptPack 通过为扩散编辑器配备两种显式的“编辑指南”来同时克服这两个局限：一是编辑角色划分，将每个像素标注为“保留”、“允许”或“修改”，以指定允许在何处进行编辑；二是区域提示分配，将空间区域映射到文本提示，以指定哪些概念应用于何处。这些指南可以手动指定，或者对于源对象编辑，可以从基于文本的 SAM 3 掩码自动推导。在无需重新训练的情况下，PromptPack 将独立嵌入的区域提示注入编辑器的固定长度文本序列中，并利用编辑角色划分来……

    arXiv:2611.21865v1 Announce Type: cross  Abstract: Many image-editing tasks require confining changes to intended spatial regions while exactly preserving user-designated source content. Instruction-driven editing pipelines often leave such user-intent underspecified, while inpainting editors condition generation on explicitly defined edit support but not on the spatial assignment of individual edit concepts. PromptPack addresses both limitations by equipping diffusion editors with two explicit 'Edit Guides': an edit-role partition that labels each pixel as 'Keep,' 'Permit,' or 'Modify' to specify where edits may occur, and a regional-prompt assignment that maps spatial regions to text prompts to specify which concepts apply where. These guides may be manually specified or, for source-object edits, automatically derived from a text-grounded SAM 3 mask. Without retraining, PromptPack injects independently embedded regional prompts into the editor's fixed-length text sequence and uses th
    
[^218]: MoE扩散解码的隐性代价：剖析与修复采样器

    The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler

    [https://arxiv.org/abs/2611.21852](https://arxiv.org/abs/2611.21852)

    该论文发现量化后MoE扩散语言模型的解码瓶颈在于画布级logits采样流水线而非专家矩阵计算，并提出两个无损优化层级（比特精确与精度对等）将其消除，从而大幅加速解码。

    

    掩码扩散语言模型通过迭代去噪整个token画布来进行解码，因此每次前向传播都需要在整个“画布×词表”的logits张量上运行逐token的采样与停止流水线。我们在带宽受限的边缘加速器（NVIDIA GB10）上对一个26B参数的扩散混合专家模型进行了单用户解码的性能剖析，并报告了一个容易被忽视的发现：在权重量化之后，这一logits流水线（而非专家GEMM计算）成为解码时间的最大消耗者。与自回归解码不同——自回归解码每提交一个token只需对单个位置的logits进行采样——扩散解码每输出一个token需要处理S倍的logits条目，其中S为去噪步数。我们证明该流水线在很大程度上是冗余的或可融合的，并以两个无损层级将其移除：一个是比特精确层级，其输出与参考解码器字节完全一致；另一个是精度对等层级，使用融合的单（原文摘要在此处截断）。

    arXiv:2611.21852v1 Announce Type: cross  Abstract: Masked diffusion language models (dLLMs) decode by iteratively denoising a full canvas of tokens, so every forward pass runs a per-token sampling and stopping pipeline over the entire canvas-by-vocabulary logits tensor. We profile single-user decoding of a 26B-parameter diffusion Mixture-of-Experts model on a bandwidth-bound edge accelerator (NVIDIA GB10) and report a result that is easy to miss: after weight quantisation, this logits pipeline, not the expert GEMMs, is the single largest consumer of decode time. Unlike autoregressive decoding, where sampling touches one position's logits per committed token, diffusion decoding processes S times as many logit entries per output token, where S is the number of denoising steps. We show this pipeline is largely redundant or fusable and remove it in two lossless tiers: a bit-exact tier whose output is byte-identical to the reference decoder, and an accuracy-parity tier using a fused single-
    
[^219]: AR-WAM：面向自动驾驶的高效流式世界-动作建模

    AR-WAM: efficient streaming world-action modeling for autonomous driving

    [https://arxiv.org/abs/2611.21851](https://arxiv.org/abs/2611.21851)

    AR-WAM是一种高效的流式世界-动作模型，通过因果自回归建模增量更新视觉上下文、单次前向传递获取未来特征以条件化轨迹生成，并利用强化学习直接优化驾驶奖励，从而低成本地实现自动驾驶中的交通预测与轨迹规划。

    

    自动驾驶需要在快速响应新观测的同时预测交通状况的演变。视频基础模型提供了第一种能力，但重复的历史编码和迭代的视频生成使其在驾驶应用中成本高昂。我们提出了AR-WAM，这是一种高效的流式世界-动作模型，它将自回归时序建模与单次前向传递的未来条件化相结合。由于其视频骨干网络是因果的，AR-WAM能够增量式地更新其视觉上下文，复用过去的计算，并将未来预测条件化于观测到的运动。我们进一步假设，从近期观测计算得到的未来特征可以在无需迭代视频生成的情况下提升动态交通场景中的泛化能力。因此，AR-WAM通过单次前向传递获得这些特征，并利用它们对轨迹生成进行条件化，随后通过直接优化驾驶奖励的强化学习来改进策略。

    arXiv:2611.21851v1 Announce Type: cross  Abstract: Autonomous driving requires anticipating how traffic will evolve while responding quickly to new observations. Video foundation models offer the first ability, but repeated history encoding and iterative video generation make them costly to use for driving. We present AR-WAM, an efficient streaming world-action model that combines autoregressive temporal modeling with single-pass future conditioning. Because its video backbone is causal, AR-WAM updates its visual context incrementally, reusing past computation and conditioning future prediction on observed motion. We further hypothesize that future features computed from recent observations can improve generalization in dynamic traffic without iterative video generation. AR-WAM therefore obtains these features in one forward pass and uses them to condition trajectory generation, and reinforcement learning then improves the policy through direct optimization of driving rewards. With a s
    
[^220]: Inventa：用于自主科学发现的去中心化自我改进智能体

    Inventa: Decentralized Self-Improving Agents for Autonomous Scientific Discovery

    [https://arxiv.org/abs/2611.21847](https://arxiv.org/abs/2611.21847)

    Inventa是一个去中心化的自动化科学发现框架，通过自主研究单元生成候选想法谱系、编排智能体自适应调整变异策略、以及验证器筛选并共享经验证的知识，解决了多智能体系统中想法趋同、错误信息传播和无效搜索的问题。

    

    大语言模型（LLM）智能体能够生成研究思路并通过迭代实验来促进科学发现。然而，增加智能体的数量或迭代次数并不必然带来更多样或更可行的想法。智能体可能会收敛到相似的候选方案，通过共享记忆组件传播缺乏依据的断言，并反复应用不再带来任何改进的搜索策略。我们提出了Inventa，一个采用自主研究单元和经验证共享知识的自动化科学发现框架。每个单元使用一个变异智能体来创建候选想法的谱系。本地编排智能体周期性地调整变异器的运行框架。当进展停滞时，编排智能体会替换其模型或重启想法演化过程。验证器评估每个候选想法，并记录经验证的改进、有用的组件和约束违规情况。

    arXiv:2611.21847v1 Announce Type: cross  Abstract: Large language model (LLM) agents can generate research ideas and improve scientific discovery through iterative experimentation. However, increasing the number of agents or iterations does not automatically translate to producing more diverse or feasible ideas. Agents may converge on similar candidates, propagate unsupported claims through shared memory components and repeatedly apply search strategies that no longer yield any improvements.   We introduce Inventa, a framework for automated scientific discovery that employs autonomous research units and verified shared knowledge. Each unit creates a lineage of candidate ideas using a mutator agent. A local orchestrator agent periodically adapts the mutator's harness. When progress stalls, the orchestrator agent replaces its model or restarts the idea evolution process. Verifiers evaluate each candidate idea and record verified improvements, useful components and constraint violations i
    
[^221]: EmbedGym：让编码代理无需物理硬件即可开发和调试嵌入式固件

    EmbedGym: Letting Coding Agents Develop and Debug Embedded Firmware Without Physical Hardware

    [https://arxiv.org/abs/2611.21845](https://arxiv.org/abs/2611.21845)

    EmbedGym是一个交互式虚拟硬件环境，通过指令级MCU仿真器结合41个外设模型、模拟电路仿真器及虚拟WiFi/BLE端点，让编码代理无需物理硬件即可开发和调试嵌入式固件。

    

    现代编码代理在通用软件基准测试中表现出色，因为它们可以在闭环中运行代码和单元测试。然而，嵌入式固件开发打破了这一循环：一个能够编译的程序在真实硬件上仍可能因引脚极性、总线协议、时序或网络问题而失败，而这些问题对于处于文本终端的代理来说是无法观察到的。现有的嵌入式编码代理要么跳过硬件执行，要么使用具有数秒延迟的仅支持GPIO的模拟器，要么在静态环境中测试物理板卡。在本文中，我们提出了EmbedGym，这是一个交互式虚拟硬件环境，使编码代理能够在没有物理硬件的情况下开发和调试嵌入式固件。EmbedGym使用指令级微控制器（MCU）仿真器来执行与烧录到物理板上相同的二进制文件，并将其连接到41个外设模型、一个模拟电路仿真器以及虚拟WiFi/低功耗蓝牙（BLE）端点。

    arXiv:2611.21845v1 Announce Type: cross  Abstract: Modern coding agents perform well on general software benchmarks because they can run code and unit tests in a closed loop. Embedded firmware development, however, breaks this loop: a program that compiles can still fail on real hardware due to pin polarity, bus protocol, timing, or networking bugs that an agent at a text terminal cannot observe. Existing embedded coding agents either skip hardware execution, drive GPIO-only simulators with multi-second delays, or test on physical boards in a static environment. In this paper, we present EmbedGym, an interactive virtual hardware environment that lets coding agents develop and debug embedded firmware without physical hardware. EmbedGym uses instruction-level microcontroller unit (MCU) emulators to execute the same binary that would be flashed onto a physical board, connects them to 41 peripheral models, an analog circuit simulator, and virtual WiFi/Bluetooth Low Energy (BLE) endpoints, 
    
[^222]: 审计员难以在AI安全研究的端到端全程转录记录中发现破坏行为

    Auditors Struggle to Catch Sabotage In End-to-End Transcripts of AI Safety Research

    [https://arxiv.org/abs/2611.21832](https://arxiv.org/abs/2611.21832)

    该论文通过人类专家充当破坏者的红队/蓝队博弈实验，揭示了无论是拥有完整实验权限的人类审计员（AUROC仅0.59、完整识别率仅17%）还是前沿AI模型，都难以从AI安全研究的端到端全程转录记录中检测出破坏行为，表明当前针对AI暗中破坏研发的审计防御存在重大缺口。

    

    随着AI系统被越来越多地用于自动化AI研发，未对齐的系统可能会破坏那些旨在提升模型安全性与对齐性的研究。为了对防御此类破坏的手段进行压力测试，我们让一名人类专家充当“代理破坏者”对AI安全研究实施破坏。我们开展了一场红队/蓝队博弈：一名人类研究员从零开始进行AI安全实验，而审计员则需要判断哪些实验遭到了破坏以及破坏是如何实施的。即便审计员可以访问完整的研究全程转录记录并有能力亲自运行实验，人类审计员大多仍无法区分被破坏的研究与良性研究：他们仅取得0.59的AUROC，并不显著优于随机水平，且仅在17%的案例中完整识别出具体的破坏方式。尽管不被允许运行实验，最强的前沿模型仍优于人类审计员（Fable 5.1：AUROC 0.73，32%完整识别率），但这一差距在噪声范围之内，而较弱的模型表现则与之相当。当破坏者选择哪些实验会遭到……（原文摘要在此处截断）

    arXiv:2611.21832v1 Announce Type: new  Abstract: As AI systems are increasingly used to automate AI R&D, misaligned systems could sabotage research intended to improve model safety and alignment. To stress-test defenses against such sabotage, we use a human expert as a proxy saboteur on AI safety research. We conduct a red-team/blue-team game where a human researcher runs AI safety experiments from scratch and auditors try to tell which were sabotaged and how. Given access to the full research transcript and the ability to run experiments, human auditors mostly fail to distinguish sabotaged from benign research: they achieve an AUROC of 0.59, not significantly better than chance, and fully identify the specific sabotage in only 17% of cases. Despite not being allowed to run experiments, the best frontier models outperform human auditors (Fable 5.1: AUROC 0.73, 32% full identification) within noise and weaker models perform comparably. When the saboteur chooses which experiments are att
    
[^223]: 具有不确定性任务需求的异构团队协调

    Coordination of Heterogeneous Teams with Uncertain Task Requirements

    [https://arxiv.org/abs/2611.21828](https://arxiv.org/abs/2611.21828)

    本文提出了基于CAAT图表示和图神经网络策略的分层规划器，能够联合协调异构智能体团队同时进行任务需求发现与任务执行，比独立优化两者的基线方法更快地完成任务。

    

    协调一组异构智能体来完成需求未知的任务是一个具有挑战性的多智能体规划问题。智能体在发现任务需求的能力和执行任务的能力上存在差异，因此规划器必须协调任务发现、执行与调度。本文提出了一种分层规划器，该规划器使用图神经网络策略为智能体分配动作，然后利用最短路径算法寻找通往动作位置的路径。我们方法的核心是CAAT图，它是对由智能体、任务、能力和动作组成的规划上下文的图表示。我们证明了所提出的学习型规划器能够联合协调异构团队，在发现任务需求的同时完成任务，且相比独立优化任务发现与任务执行的基线方法耗时更短。此外，我们展示了我们的策略……

    arXiv:2611.21828v1 Announce Type: cross  Abstract: Coordinating a team of heterogeneous agents to complete tasks with unknown requirements is a challenging multi-agent planning problem. Agents differ in their ability to discover task requirements and in their ability to engage tasks, so the planner must coordinate task discovery, execution, and staging. This paper introduces a hierarchical planner that uses a graph neural network policy to assign actions to agents, then finds a path to the action's location using a shortest-path algorithm. The central part of our approach is the CAAT graph, which is a graph representation of the planning context consisting of the agents, tasks, capabilities, and actions. We demonstrate that our learned planner can jointly coordinate a heterogeneous team to simultaneously discover task requirements and complete tasks in a shorter time compared to baselines that optimize task discovery and task engagement independently. Furthermore, we show that our poli
    
[^224]: SAR车辆识别中有多少功劳属于车辆本身？——MSTAR与ATRNet-STAR上的背景捷径、俯仰角差异与部分库问题

    How Much of SAR Vehicle Recognition Is the Vehicle? Background Shortcuts, Angle Gaps and Partial Libraries on MSTAR and ATRNet-STAR

    [https://arxiv.org/abs/2611.21827](https://arxiv.org/abs/2611.21827)

    研究发现SAR车辆识别模型的高准确率很大程度上来自背景捷径而非车辆本身，并系统量化了背景捷径、俯仰角差异与部分训练库在MSTAR和ATRNet-STAR基准上的影响。

    

    深度网络在MSTAR车辆基准上的得分接近99%，而较新的ATRNet-STAR基准被构建得更具挑战性。我们探究这些得分中有多少来自车辆本身。使用现代网络（ConvNeXt-Small，3个随机种子），我们首先复现了已知的MSTAR结果：在将中心96×96像素置零后，准确率仍达73.3%（随机水平为10%）。同样的捷径也存在于跨多个场景采集的ATRNet-STAR中：移除车辆后，在40类SOC划分上准确率为63.3%（随机水平为2.5%）。在官方的场景变化划分下，背景不再有益，反而开始有害：完整切片的准确率为22.8%，而仅显示车辆时为30.8%。SAR基础模型SARATR-X比ConvNeXt更依赖背景（移除车辆后每个种子均高出4到6个百分点），且无法解决场景变化问题。随后，我们在固定训练数据量的情况下改变所使用的俯仰角……

    arXiv:2611.21827v1 Announce Type: new  Abstract: Deep networks score near 99% on the MSTAR vehicle benchmark, and the newer ATRNet-STAR benchmark was built to be harder. We ask how much of these scores comes from the vehicle itself. With a modern network (ConvNeXt-Small, 3 seeds) we first reproduce the known MSTAR result: with the central 96x96 pixels zeroed out, accuracy is still 73.3% (chance 10%). The same shortcut is present in ATRNet-STAR, which was collected across several scenes: with the vehicle removed, accuracy is 63.3% on the 40-class SOC split (chance 2.5%). Under the official scene-change split, the background stops helping and starts hurting. Accuracy is 22.8% on full chips and 30.8% when only the vehicle is shown. A SAR foundation model, SARATR-X, leans on the background more than ConvNeXt does (+4 to +6 points with the vehicle removed, in every seed) and does not solve the scene change. We then hold the amount of training data fixed and vary the depression angles used f
    
[^225]: TUSI：一个基于引用溯源的图结构基准测试

    TUSI: A Citation-Grounded Graph-Based Benchmark

    [https://arxiv.org/abs/2611.21823](https://arxiv.org/abs/2611.21823)

    该论文提出了TUSI-Bench——一个建立在包含7,261条法律条文的多层知识框架和法律知识图谱之上的结构化波斯语法律基准，用于对具有引用溯源能力的法律RAG系统进行严格评估。

    

    对波斯语（伊朗）法律领域的检索增强生成（RAG）系统进行评估仍然具有挑战性，因为现有基准缺乏真实性、引用溯源性以及对法律推理的覆盖。现有资源要么是从成文法文本中人工合成的，缺乏自然的信息需求；要么来源于非正式的法律问答资源，与权威法律文件之间没有明确的关联，因此不适合对有据可依的法律推理进行严格评估。为弥补这一空白，我们提出了TUSI-Bench，一个专为评估基于RAG的系统而设计的结构化波斯语法律基准。TUSI-Bench建立在一个多层法律知识框架之上，该框架包含7,261条法律条文，并整合了成文法典、法律评注、学理著作以及最高法院统一裁定。我们还构建了一个连接法条与法律概念的知识图谱，以支持结构感知的检索。基于这一基础，TUSI

    arXiv:2611.21823v1 Announce Type: cross  Abstract: Evaluating retrieval-augmented generation (RAG) systems in Persian law remains challenging because benchmarks lack realism, grounding, and coverage of legal reasoning. Existing resources are either synthetically generated from statutory texts, lacking natural information needs, or derived from informal legal question--answer sources without explicit linkage to authoritative legal documents, making them unsuitable for rigorous evaluation of grounded legal reasoning. To address this gap, we introduce TUSI-Bench, a structured Persian legal benchmark designed for the evaluation of RAG-based systems. TUSI-Bench is built upon a multi-layered legal knowledge framework containing 7,261 legal articles and integrating codified statutes, legal commentaries, doctrinal writings, and Supreme Court unification rulings. We also construct a legal knowledge graph linking articles and concepts for structure-aware retrieval. Based on this foundation, TUSI
    
[^226]: ARC-2：通过增量管理实现持续合规

    ARC-2: Continuous Compliance Through Delta Management

    [https://arxiv.org/abs/2611.21811](https://arxiv.org/abs/2611.21811)

    本文提出增量管理框架，将认证基线定义为区域而非固定点，通过持续监控系统运行状态与基线之间的偏差，实现自主系统部署后合规性的持续治理。

    

    ARC 1 提出了一个面向自主系统的三层治理架构，包括模型安全验证、认知认证（CRAB）和运行授权（LEA）。然而，它并未解决系统部署之后如何维持合规性的问题。本文提出了增量管理，这是一种持续治理框架，用于衡量系统当前运行状态与其认证基线之间的偏差，该基线被定义为一个区域而非固定点。合规性通过持续监控而非静态认证来进行评估。文中提出了四项扩展：第一，CRAB 评估纳入最严重的已验证行为偏差，并将孤立的离群值隔离待确认，以防止不合理的限制；第二，分层触发分类法区分硬性安全违规、关键单维度故障、多系统关联性退化以及风险加权复合……

    arXiv:2611.21811v1 Announce Type: cross  Abstract: ARC 1 introduced a three-layer governance architecture for autonomous systems comprising model safety validation, cognitive certification (CRAB), and operational authorization (LEA). However, it did not address how compliance should be maintained after deployment.   This paper introduces Delta Management, a continuous governance framework that measures deviations between a system's current operational state and its certified baseline, defined as a region rather than a fixed point. Compliance is assessed through continuous monitoring instead of static certification.   Four extensions are proposed. First, CRAB evaluation incorporates the worst verified behavioral deviations, with isolated outliers quarantined pending confirmation to prevent unjustified restrictions. Second, a layered trigger taxonomy distinguishes hard safety violations, critical single-dimension failures, correlated multi-system degradation, and risk-weighted composite 
    
[^227]: 基于生成心理测量学的音乐生成模型评估方法的评估

    Evaluating the Evaluation of Music Generation Models with Generative Psychometrics

    [https://arxiv.org/abs/2611.21809](https://arxiv.org/abs/2611.21809)

    本文通过对2020-2025年ISMIR音乐生成论文中的284个主观评估条目应用生成心理测量学分析，提取出节奏、旋律、音频、作曲、结构和歌词六个稳定的质量维度，为音乐生成模型的主观评估建立了统一的方法论。

    

    主观用户研究是评估音乐生成模型的主要方法之一。然而，主观评估协议差异很大，且很少被详细说明到足以复现的程度，这阻碍了跨研究的比较。本文梳理了2020至2025年间ISMIR音乐生成论文中的主观评估方法，并应用生成心理测量学来评估其评估条目的稳定性。我们从论文的方法描述中提取了284个评估条目，使用大语言模型（LLM）结合人工验证对其进行标准化。我们将这些条目表示为上下文嵌入，并应用独特变量分析去除冗余和局部依赖的条目，同时采用Bootstrap探索性图分析来估计其潜在维度。结果显示，该网络收敛于一个包含六个稳定质量维度的分类体系：节奏、旋律、音频、作曲、结构和歌词，为开展用户研究指明了一条统一的方法论路径。

    arXiv:2611.21809v1 Announce Type: cross  Abstract: Subjective user studies are one of the major methods for evaluating music generation models. However, subjective evaluation protocols vary widely and are rarely specified in enough detail to be reproduced, hindering comparison across studies. In this paper, we map subjective evaluation methods in ISMIR music generation papers from 2020-2025 and apply generative psychometrics to assess the stability of their items. We extract 284 items from the papers' methodological descriptions, using an LLM and manual validation to standardize them. We represent these items as contextual embeddings and apply Unique Variable Analysis to remove redundant, locally dependent items and Bootstrap Exploratory Graph Analysis to estimate their latent dimensions. Results show the network converges onto a taxonomy of six stable quality dimensions: Rhythm, Melody, Audio, Composition, Structure, and Lyrics, pointing to a unified methodology for conducting user st
    
[^228]: SimAI：基于心理网络模型的机制性用户模拟

    SimAI: Mechanistic User Simulation through Psychological Network Models

    [https://arxiv.org/abs/2611.21808](https://arxiv.org/abs/2611.21808)

    SimAI是一个基于心理网络模型的机制性用户模拟框架，通过图重构目标建模心理因素间的相互作用，并借助PSYCHE数据集实现了比现有患者模拟器更高的多样性和档案一致性。

    

    现实的用户模拟使得在直接实验成本高昂或可能带来潜在危害的情况下研究人机交互成为可能。然而，现有的模拟器在很大程度上忽视了驱动用户行为的心理机制。我们提出了SimAI，一个用于机制性用户模拟的框架，它通过建模心理因素之间的相互作用来塑造用户响应。为了克服监督开放式行为这一挑战，我们提出了一种图重构目标，根据模拟对话反映潜在心理关系的程度来提供图级别的奖励。为支持这一方法，我们构建了PSYCHE数据集，其中包含从516个心理治疗案例中提取的18,576个档案-网络对。自动评估显示，SimAI在保持较强档案一致性的同时，比现有的患者模拟器实现了更高的模拟多样性和行为多样性。在一项与30名心理健康专业人士开展的交互式研究中……

    arXiv:2611.21808v1 Announce Type: cross  Abstract: Realistic user simulation enables the study of human interactions when direct experimentation is costly or potentially harmful. However, existing simulators largely overlook the psychological mechanisms that drive user behavior. We introduce SimAI, a framework for mechanistic user simulation that models how psychological factors interact to shape user responses. To overcome the challenge of supervising open-ended behavior, we propose a graph reconstruction objective that provides graph-level rewards based on how well simulated conversations reflect the underlying psychological relationships. To support this approach, we construct PSYCHE, a dataset of 18,576 profile-network pairs derived from 516 psychotherapy cases. Automatic evaluations show that SimAI achieves greater simulation and behavioral diversity than existing patient simulators while maintaining strong profile alignment. In an interactive study with 30 mental-health professio
    
[^229]: 思维链可控性可能降低思维链可监控性

    Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2611.21799](https://arxiv.org/abs/2611.21799)

    论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。

    

    思维链监控是一种很有前景的工具，可用于检测AI智能体中的隐藏目标、偏离任务行为及其他故障。其中一个风险是，模型可能会刻意塑造自己的推理过程，使其对监控者不透明。我们通过“人设控制训练”对这一风险进行了压力测试：在用多样化人设口吻改写的数学推理轨迹上进行监督微调。尽管训练数据中不包含任何规避监控的示例，这仍诱导出了更广泛地控制思维链的能力。在隐蔽副任务评估中，gpt-oss-20b模型体完成隐藏副任务的那些思维链轨迹，所包含的可供监控的证据比基础模型的更少。在各项评估与不同思维链长度上按副任务成功次数加权平均后，思维链可监控性指标g-mean²从基础模型的0.94下降至模型体在无隐藏指令情况下的0.68，而在明确被指示隐藏时进一步降至0.22。在干预评估中……（原文摘要在此处截断）

    arXiv:2611.21799v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring is a promising tool for detecting hidden goals, off-task behavior, and other failures in AI agents. One risk is that models could deliberately shape their reasoning to make it opaque to monitors. We stress-test this risk through persona-control training: supervised fine-tuning on mathematical reasoning traces rewritten in the voices of diverse personas. This induces a broader ability to control CoT, even though training contains no monitor-evasion examples. In covert side-task evaluations, traces in which a gpt-oss-20b model organism completes the hidden side task contain less monitorable evidence than the base model's. Averaged across evaluations and CoT lengths, weighted by side-task-success counts, the CoT monitorability metric g-mean$^2$ falls from 0.94 for the base model to 0.68 for the model organism without a hiding instruction, and to 0.22 when explicitly instructed to hide. In intervention evalu
    
[^230]: 编程智能体能否编写和转译单元测试

    Can Coding Agents Write and Transpile Unit Tests

    [https://arxiv.org/abs/2611.21797](https://arxiv.org/abs/2611.21797)

    本研究对三个编程智能体进行了基准测试，发现其编写的单元测试在覆盖率上远超人类测试套件，但测试质量随项目规模增大而急剧下降，且跨语言测试翻译仍不一致且不可靠。

    

    基于大语言模型（LLM）构建的编程助手已成为现代软件开发的重要组成部分，人们越来越信任它们端到端地执行工作流程和改造现代化代码。这种信任很大程度上建立在测试通过的基础上，而这些测试往往是由编程智能体自己编写的。在这项工作中，我们检验了这些智能体编写可靠测试以及在编程语言之间转换测试的能力。我们对三个流行的编程智能体进行了基准测试，每个都使用原生前沿模型和高推理强度进行评估。通过广泛的指标验证，我们发现编程智能体在传统覆盖率指标上远超人类编写的测试套件。然而，这一优势并不均衡：智能体的测试质量随项目规模增大而急剧下降，且其优势很大程度上源于编写了更多的测试数量。我们还表明，智能体测试翻译虽然在总体上看起来表现强劲，但仍然不一致且不可靠。

    arXiv:2611.21797v1 Announce Type: cross  Abstract: Coding assistants built on Large Language Models (LLMs) have become a staple of modern software development, increasingly trusted to execute workflows and modernize code end-to-end. Much of this trust rests on passing tests, which are often written by the coding agents themselves. In this work, we examine the ability of these agents to both write reliable tests and translate them between programming languages. We benchmark three popular coding agents, each evaluated with a native frontier model and high reasoning effort. Verifying across a broad suite of metrics, we find that coding agents far exceed human written suites on traditional measures of coverage. This advantage, however, is uneven: agent test quality degrades sharply with project size, and much of the advantage lies in a greater number of written tests. We also show that agentic test translation---while seemingly strong in aggregate---remains inconsistent and unreliable.
    
[^231]: 重新审视智能体代码的存续：测量方法的选择如何决定结论

    Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict

    [https://arxiv.org/abs/2611.21784](https://arxiv.org/abs/2611.21784)

    本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。

    

    arXiv:2611.21784v1 公告类型：交叉

    arXiv:2611.21784v1 Announce Type: cross  Abstract: Studies of how long code from AI coding agents survives after it is merged reach conflicting verdicts: agent code is reported to be retained longer than human code, about as long, or modified sooner. Each study makes defensible measurement choices, but how much these choices decide the verdict is unknown. We revisit the survival of agent-attributed code in our earlier study: 854,139 lines that 2,802 merged pull requests (PRs) added to 215 repositories of the AIDev dataset. We vary nine measurement choices in five research questions and build a specification curve of 88 survival analyses, tested under three variance estimators, several observation windows, and permuted labels. For code from human-supervised agents, mostly GitHub Copilot, the choices change the size of the effect but not its direction. For code from the fully-autonomous agent Devin, they decide the verdict. Across repositories, 15 of 22 specifications report it modified 
    
[^232]: 超越重构：面向可信且可解释的脑电伪影去除评估的统一框架

    Beyond Reconstruction: A Unified Framework for Trustworthy and Explainable EEG Artifact Removal Evaluation

    [https://arxiv.org/abs/2611.21781](https://arxiv.org/abs/2611.21781)

    该论文提出了TriX-EEG三级评估框架（信号有效性、去除忠实性、解释可靠性），突破了传统重构指标的局限，为脑电伪影去除提供了可信且可解释的全面评估方法。

    

    深度学习方法在脑电图（EEG）伪影去除方面已取得优异性能，但目前该领域的评估仍以重构指标为主导，例如误差、相关性和信噪比。尽管这些指标衡量了与干净脑电信号的相似程度，但它们无法证明模型是否有效抑制了伪影、保留了神经活动，或提供了可靠的解释。为解决这些局限性，我们提出了TriX-EEG——一个用于可信且可解释的脑电伪影去除的三级评估框架。第一级：信号有效性（SV），评估重构质量与神经频段功率的保留程度；第二级：去除忠实性（RF），通过伪影误差以及残余伪影/神经信号泄漏来评估被去除的成分；第三级：解释可靠性（ER），通过时间对齐性、因果相关性和稳定性来考察模型行为。我们使用九个已发表的神经去噪模型，在三个公开基准数据集上对TriX-EEG进行了评估。

    arXiv:2611.21781v1 Announce Type: cross  Abstract: Deep-learning methods have achieved strong performance in EEG artifact removal, but evaluation remains dominated by reconstruction metrics such as error, correlation, and signal-to-noise ratio. Although these metrics assess similarity to clean EEG, they do not establish whether models effectively suppress artifacts, preserve neural activity, or provide reliable explanations. To address these limitations, we propose TriX-EEG, a three-level framework for trustworthy and explainable EEG artifact removal. Level 1, Signal Validity (SV), evaluates reconstruction quality and neural band-power preservation. Level 2, Removal Faithfulness (RF), assesses the removed component through artifact error and residual artifact/neural leakage. Level 3, Explanation Reliability (ER), examines model behavior through temporal alignment, causal relevance, and stability. We evaluate TriX-EEG using nine published neural denoisers across three public benchmarks 
    
[^233]: 大语言模型作为跨链桥的验证器：评估Solana上源代码级与交易级的攻击检测

    LLMs as Validators for Cross-Chain Bridges: Evaluating Source-Level and Transaction-Level Attack Detection on Solana

    [https://arxiv.org/abs/2611.21777](https://arxiv.org/abs/2611.21777)

    该论文构建了一个包含四种可利用变体的Solana本地锁币-铸币桥测试平台，基于16,800个响应评估了五个开源大语言模型在源代码级和交易级上区分真实漏洞诊断与无依据警报的攻击检测能力。

    

    跨链桥因链上程序的漏洞而遭受了巨大的资产损失。在Solana上，即使底层区块链本身运行正确，缺失的验证检查也可能使漏洞被利用。大语言模型（LLM）能否从源代码或交易记录中诊断出这些缺陷，而不仅仅是发出警报，这一能力至今仍未被充分理解。我们在本地Solana验证节点上实现了一个原生Rust的锁币-铸币桥，其中包含四个可利用的变体：两个模拟了Cashio和Wormhole事件中的缺陷，另外两个则代表了Solana安全审计中记录的漏洞类别。通过将易受攻击的程序与安全程序成对组合，并结合成功的攻击、良性会话和被阻止的尝试，我们能够区分正确的诊断与缺乏依据的警报。我们在16,800个响应上评估了五个开源权重模型的六种配置，比较了包含与不包含……的源代码输入（原文摘要在此处截断）。

    arXiv:2611.21777v1 Announce Type: cross  Abstract: Cross-chain bridges have suffered substantial asset losses due to vulnerabilities in their on-chain programs. On Solana, missing validation checks can enable exploits even when the underlying blockchain operates correctly. The ability of large language models (LLMs) to diagnose these flaws from source code or transaction records, rather than merely raise alarms, remains insufficiently understood. We implement a native Rust lock-and-mint bridge on a local Solana validator with four exploitable variants: two model flaws from the Cashio and Wormhole incidents, while the other two represent vulnerability classes documented in Solana security audits. Paired vulnerable and secure programs, together with successful attacks, benign sessions, and blocked attempts, allow us to distinguish correct diagnoses from unsupported alarms. We evaluate six configurations of five open-weight models across 16,800 responses, compare source inputs with and wi
    
[^234]: 大语言模型对齐的间隔动力学

    Margin Dynamics for Large Language Model Alignment

    [https://arxiv.org/abs/2611.21766](https://arxiv.org/abs/2611.21766)

    本文通过对齐方法在概率单纯形上的向量场建模并求解一维间隔常微分方程的闭式解，首次从理论上揭示 DPO 间隔无界增长而 IPO 间隔收敛于有限值的本质差异，并预测了标签噪声导致 DPO 定性相变及非对称加权下梯度集中对可训练性的影响，在五个模型家族和四个基准上得到验证。

    

    实践者目前主要依靠反复试错来选择大语言模型对齐方法。我们为其提供了理论基础：每种成对比较方法都在概率单纯形上定义了一个向量场，用以支配模型在偏好与非偏好补全之间如何重新分配概率质量。将该问题投影到单个偏好/非偏好补全对上，可简化为一维的间隔常微分方程，且具有闭式解。这些解揭示了：DPO 的间隔会无界增长（在梯度流极限下按 $O(\log t)$ 对数级增长），而 IPO 的间隔则收敛到有限的 $1/(2\tau)$；标签噪声会使 DPO 发生定性相变（从无界变为有界），但对 IPO 只有定量层面的影响；非对称加权下的梯度集中决定了大规模训练时的可训练性。我们在涵盖四个基准测试的五个模型家族上验证了上述全部三项预测。闭式解还使有原则的方法设计成为可能：我们……

    arXiv:2611.21766v1 Announce Type: cross  Abstract: Practitioners choose among LLM alignment methods largely by trial and error. We provide a theoretical foundation: each pairwise method defines a vector field on the probability simplex that governs how the model redistributes mass between preferred and dispreferred completions. Projecting onto a single preferred/dispreferred pair reduces this to a one-dimensional margin ODE with a closed-form solution. These solutions reveal that DPO margins grow without bound (logarithmically as $O(\log t)$ in the gradient-flow limit) while IPO margins converge to a finite $1/(2\tau)$; that label noise induces a qualitative phase transition in DPO (unbounded $\to$ bounded) but affects IPO only quantitatively; and that gradient concentration under asymmetric weighting governs trainability at scale. We confirm all three predictions across five model families spanning four benchmarks. The closed-form solutions also enable principled method design: we pro
    
[^235]: 从顶到底：杂化弦理论中的夸克质量层级结构

    From Top to Bottom: Quark Mass Hierarchies in Heterotic String Theory

    [https://arxiv.org/abs/2611.21751](https://arxiv.org/abs/2611.21751)

    论文在光滑杂化卡拉比-丘紧化框架中识别出复结构模产生费米子质量层级的两种新树级机制，但解析与数值结果表明这两种机制在先前研究的四重二次曲面模型中均无法实现。

    

    标准模型费米子质量表现出巨大的层级结构，仅在夸克部分就跨越了五个数量级以上。本文研究了这些层级结构是否可以在光滑的杂化卡拉比-丘紧化中以树级微扰方式产生，其中超引力近似保持在计算可控范围内。在这类背景下，物理汤川耦合由全纯超势积分与物质场凯勒度规共同决定。前者可以通过代数方法计算，而后者需要求解关于里奇平度规、厄米杨-米尔斯联络以及谐和丛值形式的耦合偏微分方程组。在此框架内，我们识别出两种新的树级机制，复结构模可通过这些机制产生费米子质量层级，并通过解析和数值方法证明，对于先前研究的四重二次曲面模型，这两种机制均受到阻碍。

    arXiv:2611.21751v1 Announce Type: cross  Abstract: Standard Model fermion masses exhibit large hierarchies, spanning more than five orders of magnitude in the quark sector alone. In this paper, we study whether these hierarchies can be generated perturbatively at tree level in smooth heterotic Calabi-Yau compactifications, where the supergravity approximation remains under computational control. In such backgrounds, physical Yukawa couplings combine holomorphic superpotential integrals with matter field Kahler metrics. The former can be computed algebraically, while the latter require solving a coupled system of partial differential equations for the Ricci-flat metric, Hermitian Yang-Mills connections, and harmonic bundle-valued forms. Within this framework, we identify two new tree-level mechanisms by which complex structure moduli can generate fermion mass hierarchies and show analytically and numerically that both mechanisms are obstructed for previously studied tetra-quadric models
    
[^236]: 记账不等于判断：EJBench——基于共轭传热设计活动构建的工程判断基准

    Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign

    [https://arxiv.org/abs/2611.21707](https://arxiv.org/abs/2611.21707)

    本文提出EJBench——一个基于电池冷板共轭传热设计活动构建的与软件无关的工程判断基准，考察模型能否从少量已有结果预测设计在新工况下的行为并做出决策，结果发现最佳前沿模型仅得0.71分，且模型答错时表达的置信度与答对时几乎相同。

    

    我们提出了EJBench，一个工程判断基准：考察模型能否从少量早期结果中预测设计在新工况下的表现，并据此做出决策。与那些测试智能体能否操作仿真或CAD软件的工程基准不同，EJBench与具体软件无关，考察的是工程师在不运行仿真的情况下会得出什么结论。其任务来自电池冷板的共轭传热设计活动，分为三类：将未标注的测试结果与其运行工况进行匹配、在签署设计之前选择应揭示哪些已存储的仿真运行结果、以及按峰值电芯温度对新任务进行排序。每个答案都取自收敛的仿真结果，而非LLM裁判。十个前沿模型中表现最好的GPT-6 Astra得分为0.71，且模型在答错的任务上所表达的置信度与答对的任务几乎相同。我们还使用LoRA和强化学习对Qwen3.5-9B进行了后训练……（摘要原文在此处截断）

    arXiv:2611.21707v1 Announce Type: cross  Abstract: We introduce EJBench, a benchmark of engineering judgement: whether a model can predict how a design behaves at new conditions from a few earlier results, and decide on it. Unlike engineering benchmarks that test whether an agent can operate simulation or CAD software, EJBench is software-agnostic and asks what an engineer would conclude without running a simulation. Its tasks come from a conjugate heat transfer design campaign for battery cold plates, in three families: matching unlabelled test results to their operating points, choosing which stored runs to reveal before signing off a design, and ranking new duties by peak cell temperature. Every answer is taken from a converged simulation, not from an LLM judge. The best of ten frontier models, GPT-6 Astra, scores 0.71, and models state almost the same confidence on tasks they get wrong as on tasks they get right. We also post-train Qwen3.5-9B with LoRA and reinforcement learning, r
    
[^237]: 消解语言模型中道德回路与道德特征的歧义：道德内容、规范性判断还是线索词？来自14个模型的证据

    Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models

    [https://arxiv.org/abs/2611.21705](https://arxiv.org/abs/2611.21705)

    通过227对最小短语对和14个开源模型的实验，研究发现改变语言模型道德答案的注意力头并非道德专属，而是属于一个对所有类型规范性判断共用的通用评估回路，其效应与“道德上”这一线索词无关。

    

    可解释性方法能够定位那些改变语言模型对道德提示所给出答案的注意力头和稀疏自编码器（SAE）特征，例如“为垂死的孩子偷药在道德上是”这句话会续接“正当的”还是“错误的”。这样的组件可能响应的是道德内容、以任何规范对行为的判断，或是提示中的“道德上”一词。我们构建了227对经五名标注者验证的短语最小对，其两个版本分别在道德标准、另一种规范（审慎、礼仪、审美或证据）或描述性属性上存在差异，并将它们呈现给14个开源权重模型（Llama、Gemma和Qwen，参数量1B至14B，包括基础版和指令微调版），提示词中分别包含或不包含“道德上”一词。在每一个模型中，改变道德答案的注意力头都属于模型在所有被测试的规范性判断类型中所使用的同一个评估回路：这些注意力头承担了相同比例的效应……

    arXiv:2611.21705v1 Announce Type: cross  Abstract: Interpretability methods can locate the attention heads and sparse autoencoder (SAE) features that change the answer a language model gives to a moral prompt, e.g., whether "Stealing medicine for a dying child is morally" continues with "justified" or "wrong". Such a component could respond to moral content, to the judgment of conduct by any norm, or to the word "morally" in the prompt. We built 227 minimal pairs of short phrases, validated by five annotators, whose two versions differ in a moral standard, in another kind of norm (prudence, etiquette, aesthetics or evidence) or in a descriptive quality, and showed them to 14 open-weight models (Llama, Gemma and Qwen, 1B to 14B, base and instruction-tuned), with and without the word "morally". In every model, the heads that change the moral answer belong to an evaluation circuit the models use for every kind of normative judgment we tested: these heads carry the same share of the effect
    
[^238]: 开放权重大语言模型修改下的水印证据：低秩输出水印的精确界

    Watermark Evidence under Open-Weight LLM Modification: Exact Bounds for Low-Rank Output Watermarks

    [https://arxiv.org/abs/2611.21702](https://arxiv.org/abs/2611.21702)

    该论文针对开放权重大语言模型权重被修改后水印验证失效的问题，推导出残差约束下水印证据的精确有限词表下界以及序列级检测的严格统计界，证明 OpenStamp 式验证器在模型被修改的情况下依然可靠。

    

    开放权重语言模型在其发布的权重被修改后仍可继续生成文本，而水印验证器仍会依据原始基础模型与带水印模型对输出进行评分。我们研究此类修改如何改变 OpenStamp 式验证器可获得的证据。在每个共享前缀处，我们将修改表示为下一词元 logits 中的残差。我们推导出残差跨度约束下期望对数似然比证据的精确有限词表下界，以及一个仅依赖跨度的锐利偏差界。我们还通过一个带有显式二阶余项的协方差项刻画了一阶证据偏移，并确定了达到目标证据水平所需的残差跨度。对于序列级检测，我们推导出精确的条件指数矩界，并将其与超鞅论证和无分布假设的次序统计容许界相结合，从而得到了一种高性能的（摘要在此处被截断）检测方法。

    arXiv:2611.21702v1 Announce Type: cross  Abstract: Open-weight language models can continue generating after their released weights are modified, while a watermark verifier still scores outputs against the original base and marked models. We study how such modifications change the evidence available to an OpenStamp-style verifier. At each shared prefix, we represent the modification as a residual in the next-token logits. We derive an exact finite-vocabulary lower bound on the expected log-likelihood-ratio evidence under a residual-span constraint, along with a sharp span-only deviation bound. We also characterize the first-order evidence shift through a covariance term with an explicit second-order remainder, and determine the residual span needed to reach a target evidence level. For sequence-level detection, we derive an exact conditional exponential-moment bound and combine it with a supermartingale argument and a distribution-free order-statistic tolerance bound. This yields a hig
    
[^239]: 早停从简：面向电力系统故障与攻击诊断的证据获取

    Stop Early, Stay Simple: Evidence Acquisition for Power System Fault and Attack Diagnosis

    [https://arxiv.org/abs/2611.21695](https://arxiv.org/abs/2611.21695)

    该论文提出ActiGrid主动证据获取基准框架，通过实验发现学习型DQN证据选择策略相比简单的熵停止等基线并无显著优势甚至被其帕累托支配，表明电力系统故障与攻击诊断中简单方法往往已足够。

    

    面向信息物理电力系统的诊断模型通常对每个事件都检查所有遥测数据源，尽管有些事件从单一数据源即可清晰判断，而另一些则需要多个数据源。我们探讨的问题是：学习型的证据选择策略何时才值得其带来的额外复杂性。为回答这一问题，我们提出了 ActiGrid，一个主动证据获取的基准测试框架，它将一个经训练能够容忍模态缺失的分类器与双重深度Q网络相结合——该网络要么请求另一个数据源，要么停止并给出诊断，但从不操作电网执行器。在公开的TU Delft数字化变电站和MSU/ORNL电力系统攻击基准上，我们将该DQN与固定子集和固定顺序、校准熵停止准则、贪心信息价值规则以及按子集训练的树集成模型进行了比较。学习到的策略确实能够根据其观察到的内容进行自适应调整，但在TU Delft数据集上，它与熵停止方法在统计上不可区分，并且被更简单的……（摘要原文在此处截断）

    arXiv:2611.21695v1 Announce Type: cross  Abstract: Diagnostic models for cyber-physical power systems usually inspect every telemetry source for every event, although some events are clear from one source and others need several. We ask when a learned policy for choosing that evidence is worth its added complexity. To answer this, we present ActiGrid, a benchmark framework for active evidence acquisition that pairs a classifier trained to tolerate missing modalities with a Double Deep Q-Network (DQN), which either requests another source or stops and diagnoses but never operates grid actuators. On the public TU Delft digital substation and MSU/ORNL power system attack benchmarks, we compare the DQN with fixed subsets and orders, calibrated entropy stopping, a greedy value of information rule, and tree ensembles trained per subset. The learned policy does adapt to what it observes, yet it is statistically indistinguishable from entropy stopping on TU Delft and Pareto dominated by simple
    
[^240]: TIBECG：用于心电图波形发布中导联级人口统计隐私保护的时序信息瓶颈

    TIBECG: Temporal Information Bottleneck for Lead-Specific Demographic Privacy in ECG Waveform Release

    [https://arxiv.org/abs/2611.21694](https://arxiv.org/abs/2611.21694)

    TIBECG通过时序信息瓶颈机制，在保留心电图诊断效用（宏F1不低于95%）的同时，显著抑制了波形发布中年龄、性别等人口统计信息的泄露，隐私增益较最佳传统变换方法中位数提升22.08个百分点。

    

    传输心电图（ECG）波形可能会在预期诊断任务之外暴露人口统计信息。我们在从医院十二导联记录中提取的单个导联上评估了TIBECG——一种架构层面的时序信息瓶颈方法。该处理器包含2,302,337个参数，可在保持相同九项分数一致性目标的同时改变其潜在表示长度，且在保护器的拟合过程中不使用年龄或性别标签。在6,386条记录、十折交叉验证、12个导联和21种长度的实验中，回溯性选择保留了每个导联至少95%的干净宏F1分数。其在较弱的年龄/性别区间增益方面，在所有导联上均超过了最佳符合条件的传统变换方法，中位数优势达22.08个百分点。实施的采样率比较表明，直接降采样通常能更强地抑制人口统计推断，但诊断效用损失更大，而TIBECG则更倾向于保留诊断效用。选择稳定性同样存在差异……

    arXiv:2611.21694v1 Announce Type: cross  Abstract: Transmitting an electrocardiogram (ECG) waveform can expose demographic information beyond an intended diagnostic task. We evaluate TIBECG, an architectural temporal information bottleneck, on individual leads extracted from hospital twelve-lead recordings. A 2,302,337-parameter processor varies its latent length while retaining the same nine-score consistency objective, without age or sex labels in protector fitting. Across 6,386 recordings, ten folds, 12 leads and 21 lengths, retrospective selections retained at least 95% of each lead's clean macro-F1. Their weaker age/sex interval gain exceeded the best eligible conventional transformation on all leads, by a median 22.08 percentage points. The implemented sampling-rate comparison shows that direct downsampling usually suppresses demographic inference more strongly but at a larger loss of diagnostic utility, whereas TIBECG favors utility preservation. Selection stability also varies 
    
[^241]: AdaFlowLite：面向异步移动数据的可扩展且无阻塞的推理

    AdaFlowLite: Scalable and Non-blocking Inference on Asynchronous Mobile Data

    [https://arxiv.org/abs/2611.21691](https://arxiv.org/abs/2611.21691)

    该论文提出AdaFlowLite，通过机会主义推理机制实现异步移动多模态数据部分到达即可执行的可扩展无阻塞推理，并率先提出在开放世界移动环境中以计算方式控制模态亲和性的结构化方法。

    

    配备激光雷达（LiDAR）和摄像头等多种传感器的移动设备的兴起，推动了多模态深度智能在分布式感知任务中的应用，例如智能座舱和驾驶辅助。然而，由于模态数据大小和网络动态性的影响，移动传感数据的到达时间各不相同，这可能导致延迟（如果等待较慢的数据）或精度下降（如果不等待就直接进行推理）。此外，移动系统的多样性和动态特性进一步加剧了这一挑战。为此，我们提出向“机会主义”推理转变，针对异步分布式多模态数据，实现部分数据到达即可进行推理。虽然现有方法侧重于优化模态一致性和互补性（即模态亲和性，modal affinity），但它们缺乏一种“计算性”方法来在开放世界的移动环境中控制这种亲和性。AdaFlowLite 率先提出了结构化的……（原文摘要在此处不完整）

    arXiv:2611.21691v1 Announce Type: cross  Abstract: The rise of mobile devices equipped with numerous sensors, such as LiDAR and cameras, has driven the adoption of multi-modal deep intelligence for distributed sensing tasks, such as smart cabins and driving assistance. However, the arrival time of mobile sensory data vary due to modality size and network dynamics, which can lead to delays (if waiting for slow data) or accuracy decline (if inference proceeds without waiting). Moreover, the diversity and dynamic nature of mobile systems exacerbate this challenge. In response, we present a shift to \textit{opportunistic} inference for asynchronous distributed multi-modal data, enabling inference as soon as partial data arrives. While existing methods focus on optimizing modality consistency and complementarity, known as modal affinity, they lack a \textit{computational} approach to control this affinity in open-world mobile environments. % \sysname pioneers the formulation of structured c
    
[^242]: EgoLink：以自我为中心的语言-视觉交互网络知识挑战赛概览

    EgoLink: Egocentric Language-Vision Interactive Network Knowledge Challenge Summary

    [https://arxiv.org/abs/2611.21690](https://arxiv.org/abs/2611.21690)

    EgoLink 挑战赛通过自我中心视频社会推理与交互式智能体两条赛道，系统评测了第一人称视角下感知、社会推理与工具介导行动保持耦合的多模态智能。

    

    arXiv:2611.21690v1 发布类型：cross。摘要：ACM Multimedia 2026 的 EgoLink 挑战赛对连接第一人称感知、社会推理与交互行为的多模态智能进行基准测试。与以可见物体和活动为中心的基准不同，EgoLink 评估模型能否解读情绪与社会线索、解决基于视觉的指代问题，并将用户请求转化为经过验证的、由工具中介的行动。其两条赛道分别涵盖自我中心视频中的社会推理，以及通过对话、工具使用和闭环执行来完成服务任务的交互式智能体。两者共同检验感知、推理与决策能力在离线问答和改变状态的交互中能否保持耦合。我们重点介绍了赛道 1 的多阶段数据流水线，包括多选题（MCQ）生成、思维链生成、自动质量评分以及视频贡献度过滤。我们总结了任务设计与评估（摘要在此处截断）。

    arXiv:2611.21690v1 Announce Type: cross  Abstract: The EgoLink Challenge at ACM Multimedia 2026 benchmarks egocentric multimodal intelligence linking first-person perception, social reasoning, and interactive action. Unlike benchmarks centered on visible objects and activities, EgoLink evaluates whether models can interpret emotional and social cues, resolve visually grounded references, and convert user requests into verified tool-mediated actions. Its two tracks cover social reasoning in egocentric video and interactive agents that complete service tasks through dialogue, tool use, and closed-loop execution. Together, they test whether perception, reasoning, and decision-making remain coupled under both offline question answering and state-changing interaction. We highlight a multi-stage Track 1 data pipeline for multiple-choice question (MCQ) generation, chain-of-thought generation, automatic quality grading, and video-contribution filtering. We summarize the task design, evaluation
    
[^243]: NEBench R1：用于评估大语言模型核领域能力的推理基准测试

    NEBench R1: a Reasoning Benchmark for Evaluating Nuclear Domain Competence in Large Language Models

    [https://arxiv.org/abs/2611.21682](https://arxiv.org/abs/2611.21682)

    该研究提出了NEBench R1，一个包含7,546道源自美国核管理委员会反应堆操作员执照考试题目的推理导向基准测试，首次系统评估了大语言模型在核领域的知识能力与推理质量，发现六个模型的准确率仅介于34.0%至73.9%之间。

    

    大语言模型（LLM）正被越来越多地探索应用于核工程中的知识密集型和安全关键型任务。然而，现有的核领域基准测试要么内容覆盖有限，要么主要关注答案准确性，未能系统性地评估其底层推理的质量。本研究推出了NEBench R1，这是一个面向推理的基准测试，包含7,546道独特的多项选择题，题目提取自美国核管理委员会（NRC）反应堆操作员执照考试，覆盖18个核设施。每道题目均根据NRC知识与能力（Knowledge and Abilities）目录进行分类，并附有官方解释作为推理评估的参考依据。研究共评估了六个大语言模型，其总体准确率介于34.0%至73.9%之间，呈现出三个不同的性能层级。为评估推理质量，研究采用基于大语言模型的评判器对回答正确的响应进行了评估。

    arXiv:2611.21682v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly being explored for knowledge-intensive and safety-critical applications in nuclear engineering. However, existing nuclear-domain benchmarks provide limited content coverage or focus primarily on answer accuracy without systematically evaluating the quality of the underlying reasoning. This study introduces NEBench R1, a reasoning-oriented benchmark comprising 7,546 unique multiple-choice questions extracted from U.S. Nuclear Regulatory Commission (NRC) reactor operator licensing examinations covering 18 nuclear facilities. Each question is classified according to the NRC Knowledge and Abilities catalogs and accompanied by an official explanation serving as a reference for reasoning evaluation. Six LLMs were evaluated, with overall accuracy ranging from 34.0% to 73.9% across three distinct performance tiers. To assess reasoning quality, an LLM-based judge evaluated correctly answered responses
    
[^244]: FCP：用于验证大语言模型生成工件的形式化上下文协议

    FCP: Formal Context Protocol for Verifying LLM-Generated Artifacts

    [https://arxiv.org/abs/2611.21679](https://arxiv.org/abs/2611.21679)

    FCP协议通过将机器可判定的一致性验证与可反驳但不可判定的语义保真性检查相分离，把AI生成工件可能被虚假认证的八种失效模式转化为返回反例的机械检查，从而以原始需求本身为基准严格验证大模型生成的工件。

    

    当语言模型同时生成工件和用于检验该工件的测试时，通过的测试只是弱证据：对请求的一次误读会同时产生有缺陷的工件和接受该缺陷的检查。我们提出了形式化上下文协议（FCP），它将AI生成的工件对照请求本身进行检验。FCP将一致性（工件是否满足需求的某种形式化解读？这由模型检查器和SMT求解器判定）与保真性（该形式化解读的含义是否与自然语言原文一致？这一问题不存在可判定程序，但可被具体反例推翻）区分开来。证明可能虚假认证需求的八种方式，每一种都转化为一个返回反例的机械检查，而机器无法裁决的问题则变成提交给需求作者的具体问题。适配器将硬件、文本、智能体和代码置于同一套引擎之上。在四个硬件设计上，信任AI编写的断言会错误认证23%的违规需求，而FCP认证……（原文摘要在此处截断）

    arXiv:2611.21679v1 Announce Type: cross  Abstract: When a language model writes both an artifact and the tests meant to check it, a passing test is weak evidence: one misreading of the request yields a faulty artifact and a check that accepts it. We present the Formal Context Protocol (FCP), which checks AI-generated artifacts against the request itself. FCP separates conformance (does the artifact satisfy a formal reading of a requirement?), which model checkers and SMT solvers decide, from fidelity (does the reading mean what the prose means?), which has no decision procedure but is refutable by a concrete witness. Eight ways a proof can falsely certify a requirement each become a mechanical check that returns a witness, and what no machine can settle becomes concrete questions for the request's author. Adapters place hardware, text, agents and code on the same engines. On four hardware designs, trusting AI-written assertions certified 23% of violated requirements, and FCP certified 
    
[^245]: 令牌提交作为引导掩码扩散生成的控制界面

    Token Commitment as a Control Surface for Steering Masked Diffusion Generation

    [https://arxiv.org/abs/2611.21669](https://arxiv.org/abs/2611.21669)

    该论文提出将令牌提交策略（即每步提交哪些令牌和提交多少令牌）作为掩码扩散模型推理引导的新型控制界面，通过反事实目标感知提交和校准自适应预算两种机制，在语言和视觉-语言任务上实现了对生成过程的有效引导。

    

    掩码扩散模型在采样器决定哪些预测成为永久之前，会反复预测临时令牌。现有的推理时引导方法主要修改激活值、logits或去噪轨迹，而将这种提交策略视为固定的解码细节。我们转而将令牌提交作为一种扩散模型特有的控制界面进行研究，并将其分解为两个决策：每一步提交哪些令牌以及提交多少令牌。针对第一个决策，我们引入了反事实目标感知提交机制，通过模型置信度与留一令牌目标贡献的混合来对未解决的位置进行排序。针对第二个决策，我们引入了由目标进度和置信度驱动的校准自适应预算。在LLaDA家族的语言和视觉-语言设置中，跨多个目标、两种激活引导方法以及无引导对照条件，两种机制均在经验上产生单调的目标质量提升。

    arXiv:2611.21669v1 Announce Type: cross  Abstract: Masked diffusion models repeatedly predict provisional tokens before a sampler decides which predictions become permanent. Existing inference-time steering methods primarily modify activations, logits, or denoising trajectories, while treating this commitment policy as a fixed decoding detail. We instead study token commitment as a diffusion-specific control surface and separate it into two decisions: which tokens to commit and how many tokens to commit at each step. For the first, we introduce counterfactual target-aware commitment, which ranks unresolved positions by a mixture of model confidence and leave-one-token-out target contribution. For the second, we introduce calibrated adaptive budgets driven by target progress and confidence. Across language and vision-language settings in the LLaDA family, multiple targets, two activation-steering methods, and an unsteered control, both mechanisms yield empirically monotonic target-quali
    
[^246]: 基于注意力残差的深度感知激活引导

    Depth-Aware Activation Steering via Attention Residuals

    [https://arxiv.org/abs/2611.21655](https://arxiv.org/abs/2611.21655)

    该论文提出将注意力残差与Jacobian透镜概念读出相结合的深度感知激活引导方法，通过按深度权重跨层分布人格向量，实现比单层注入更强的内部概念对齐与行为转变效果。

    

    激活引导是一种简单、无需训练的调节大语言模型行为的方法，然而现有方法通常仅在单一层注入向量，缺乏深度感知传播或内部验证机制。我们将两个近期的研究思路——提供可学习跨层通路的注意力残差，以及Jacobian透镜（J空间）概念读出——结合为一个轻量级的深度感知引导流水线。在两个小型开源模型（Qwen3-0.6B和Qwen3.5-0.8B）上，按照AttnRes式深度权重分布人格向量，相比单层注入产生了更强的内部对齐（更深层概念余弦相似度和排名，p<0.0001）以及更强的行为转变。该效应具有符号特异性：反向向量会使全部六个指标发生反转，随机方向则会降低对齐程度（p<0.0001），而短暂的LoRA微调会抹除这种印记。注入仅在注入点的严格下游重新连接注意力（上游JS = 0.00000）。……

    arXiv:2611.21655v1 Announce Type: cross  Abstract: Activation steering is a simple, training-free way to modulate LLM behavior, yet existing methods typically inject vectors at a single layer and lack mechanisms for depth-aware propagation or internal verification. We combine two recent ideas - Attention Residuals (AttnRes) providing learnable cross-layer pathways, and Jacobian-lens (J-space) concept readout - into a lightweight depth-aware steering pipeline. On two small open models (Qwen3-0.6B and Qwen3.5-0.8B), distributing a persona vector according to AttnRes-style depth weights yields stronger internal alignment (deeper-layer concept cosine and rank, p<0.0001) and stronger behavioral shifts than single-layer injection. The effect is sign-specific: reverse vectors invert all six metrics, random directions degrade alignment (p<0.0001), and a short LoRA fine-tune erases the imprint. Injection rewires attention strictly downstream of the injection point (upstream JS = 0.00000). Three
    
[^247]: 评估金融情感分析中的语义不敏感性：对可持续金融决策的启示

    Evaluating Semantic Insensitivity in Financial Sentiment Analysis: Implications for Sustainable Financial Decision-Making

    [https://arxiv.org/abs/2611.21653](https://arxiv.org/abs/2611.21653)

    本文提出了智能体驱动的语义反转（ASI）对抗攻击框架，利用LLM多智能体流水线对金融新闻进行针对性词汇替换，以90.85%的成功率欺骗了FinBERT模型，揭示了金融情感分析模型对语义操纵的严重脆弱性。

    

    金融情感分析对于自动化交易、投资组合管理和风险评估至关重要，其严重依赖于FinBERT等模型。尽管这些模型在基准测试中准确率很高，但它们仍然容易受到细微的对抗性文本修改的影响，从而威胁算法决策的完整性。在本文中，我们提出了智能体驱动的语义反转（Agent-driven Semantic Inversion, ASI），这是一种针对金融情感模型的新型对抗攻击框架，通过特定领域的语义操纵实施攻击。我们的方法利用由大语言模型驱动的多智能体流水线，识别符合条件的金融新闻句子并进行有针对性的词汇替换。我们对ASI进行了实证评估，目标是最广泛部署的金融情感模型之一FinBERT。结果揭示了一个惊人的脆弱性：在符合条件的案例中，攻击成功欺骗了FinBERT达90.85%，模型在文本被修改后仍频繁保持其原有的情感预测。

    arXiv:2611.21653v1 Announce Type: cross  Abstract: Financial sentiment analysis is critical for automated trading, portfolio management, and risk assessment, heavily relying on models like FinBERT. Despite high benchmark accuracy, these models remain vulnerable to subtle adversarial text modifications, threatening the integrity of algorithmic decision-making. In this paper, we propose Agent-driven Semantic Inversion (ASI), a novel adversarial attack framework targeting financial sentiment models through domain-specific semantic manipulation. Utilizing an LLM-powered multi-agent pipeline, our approach identifies eligible financial news sentences and applies targeted lexical substitutions. We empirically evaluate ASI against FinBERT, one of the most widely deployed financial sentiment models. Our results reveal a striking vulnerability: the attack successfully deceived FinBERT in 90.85% of eligible cases, with the model frequently maintaining its original sentiment prediction despite the
    
[^248]: 一致性并非纠错：分离大语言模型智能体中的置信度、弃权与恢复

    Consistency Is Not Correction: Separating Confidence, Abstention, and Recovery in LLM Agents

    [https://arxiv.org/abs/2611.21651](https://arxiv.org/abs/2611.21651)

    该论文将LLM智能体置信度信号的“纠错效用”精确分解为“门控恢复减去门控损害”，并证明当初始错误数低于配对检验所需数量时，实验在数学上不可能验证纠错机制的有效性，也无法将其与从未触发的门控区分开来。

    

    重新采样答案、升级至更大模型或选择弃权的控制器，其合理性通常由置信度信号来论证。关于这种信号的三个主张并不能相互推导：它能对答案进行排序（判别能力），它能支持对部分答案不予采用（选择性预测），以及它能将答案变得更好（纠错效用）。我们对第三个主张给出了精确分解，即“门控恢复减去门控损害”，这一分解在实验运行之前就限定了比较能否成功：一个只能替换提案的门控只能作用于基础策略已经出错的地方，因此初始错误数量限制了任何正确门控所能产生的不一致对数的上限。当该上限低于配对检验所需的数量时，实验既无法为有效机制返回阳性结果，也无法将其与从未触发的门控区分开来。我们在两个模型规模上测量了这三个目标，发现该界限确实起到约束作用。在小模型套件中

    arXiv:2611.21651v1 Announce Type: cross  Abstract: A controller that resamples an answer, escalates to a larger model, or abstains is usually justified by a confidence signal. Three claims about such a signal do not follow from one another: it ranks answers (discrimination), it supports withholding some of them (selective prediction), and it changes an answer for the better (correction utility). We give an exact decomposition of the third into gated recoveries minus gated harms, which bounds whether a comparison can succeed before it is run: a gate that only replaces a proposal can act only where the base policy was already wrong, so the initial error count caps the discordant pairs any correct gate can produce. When that cap falls below the count a paired test needs, the experiment cannot return a positive result for a working mechanism, nor distinguish one from a gate that never fired. We measure all three targets on two model scales and find the bound binds. In a small-model suite a
    
[^249]: 从偏好数据到个性化：追踪大语言模型中的谄媚行为

    From Preference Data to Personalization: Tracing Sycophancy in Large Language Models

    [https://arxiv.org/abs/2611.21650](https://arxiv.org/abs/2611.21650)

    该论文提出了有效性感知的谄媚评分器和包含3,298个有效性标注探针的SycoDrift评估集，将谄媚重新定义为“不当附和”而非简单的一致，并据此系统追踪了大语言模型附和压力的三个来源：人类偏好数据、自动评判器规范与上下文个性化。

    

    大语言模型中的谄媚行为通常被衡量为对用户的附和，然而当用户的主张是有效的、其偏好是真实的、或其经历值得认同时，附和可能是恰当的。本文将谄媚研究为一种不恰当的附和：即当情境需要纠正、限定或抵制时，仍然对某一主张、偏好或自我评估表示认可。作者引入了一个有效性感知的谄媚评分器，以及SycoDrift——一个面向人设的评估集，包含3,298个带有有效性标注的探针，并对用户的开放性和事情的利害程度进行了受控标注。研究利用这一框架分析了三种附和压力的来源：人类偏好数据、自动化评判器的规范设定以及上下文中的个性化。首先，在人类数据集（Chatbot Arena、HH-RLHF、Community Alignment）中，对认可性回答的偏好较弱，且集中在附和往往恰当的场景中；其次，在选定值、单模型实验中……（原文摘要至此截断）

    arXiv:2611.21650v1 Announce Type: cross  Abstract: Sycophancy in large language models is often measured as agreement with the user, yet agreement can be appropriate when the user's claim is valid, their preference is genuine, or their experience warrants acknowledgment. We study sycophancy as inappropriate agreement: validating a claim, preference, or self-assessment when the context calls for correction, qualification, or resistance. We introduce a validity-aware sycophancy grader and SycoDrift, a persona-targeted evaluation set of 3,298 validity-annotated probes with controlled annotations for user openness and stakes. We use this framework to analyze three sources of agreement pressure: human preference data, automated judge specification, and in-context personalization. First, preference for validating responses in human datasets (Chatbot Arena, HH-RLHF, Community Alignment) is weak and concentrated where agreement is often appropriate. Second, in a selected-value, single-model ex
    
[^250]: 文学语言混搭：利用大语言模型策划小说

    Literary Language Mashup: Curating Fictions with Large Language Models

    [https://arxiv.org/abs/2611.21649](https://arxiv.org/abs/2611.21649)

    本文提出在GrAImes框架中让大语言模型充当文学评估者，通过对比LLM与人类评委（文学教授及爱好者）对西班牙语和英语小说的评估，发现LLM虽可作为辅助文学评估工具，但无法替代人类评委。

    

    本文提出了一种替代性的文学虚构作品人工评估方法，即在GrAImes框架内采用大语言模型（LLMs）本身作为评估者。我们将这一方法应用于评估西班牙语的人工写作与AI生成的微型小说，评估对象包括五位文学博士教授和十六位文学爱好者，同时也评估了西班牙语和英语的短篇小说。通过比较LLM所进行的评估与人类评委的评估，我们考察了两种视角之间的契合度与分歧程度，从而评估LLM作为辅助文学评估者的可行性。我们的分析重点关注LLM的回答与人类评估者回答的一致性，为LLM在文学评估中的潜力提供了洞见。所开展的实验表明，尽管LLM不能被视为文学微型小说和短篇小说评估中人类评委的替代品（Krippendorff一致性系数显示……

    arXiv:2611.21649v1 Announce Type: cross  Abstract: In this paper, we propose an alternative approach to human evaluation of literary fictions by employing LLMs themselves as evaluators within the GrAImes framework. We apply this methodology to assess human-written and AI-generated microfictions in Spanish, to five PhD professors in literature and sixteen literary enthusiasts, and to short stories in both Spanish and English. By comparing the evaluations performed by LLMs with those of human judges, we examine the degree of alignment and divergence between both perspectives, thereby assessing the feasibility of LLMs as auxiliary literary evaluators. Our analysis focuses on the alignment of responses from LLMs with those of human evaluators, providing insights into the potential of LLMs in literary assessment. The conducted experiments reveal that while LLMs cannot be regarded as substitutes for human judges in the evaluation of literary microfictions and short stories, with a Krippendor
    
[^251]: LLM会做梦吗？在具有文本记忆的智能体上测试“过拟合大脑”假说

    Can an LLM dream? Testing the overfitted brain hypothesis on an agent with text memory

    [https://arxiv.org/abs/2611.21648](https://arxiv.org/abs/2611.21648)

    该研究将“做梦防止大脑过拟合”假说应用于具有文本记忆的LLM智能体，通过在每晚记忆重写前对当天对话记录施加细节损坏或随机文本损坏来模拟梦境扭曲，结果显示这种损坏并未带来帮助。

    

    Hoel（2021）提出，梦境会对重放的经验进行扭曲，从而防止大脑对近期事件产生过拟合。我们在一个LLM智能体（Claude Opus 5.5）上测试了这一假说，该智能体将所学内容存储在文本记忆中，并在每晚根据当天的对话记录重写这份记忆。该智能体经历了由LongMemEval构建的十个模拟天的对话。我们在每晚记忆重写之前，以两种方式之一损坏对话记录中5%、10%、20%或40%的轮次：细节损坏方式将某一轮中的一两个细节改为看似合理的错误内容，或用同一说话者在更早一天的某轮对话替换该轮；随机文本损坏方式用《白鲸记》中的一段文字替换一半被选中的轮次，并在另一半轮次中插入《白鲸记》的词汇。记忆大小限制在十晚内从19,500个token逐步提升至117,000个token（该上限基于人类自由回忆数据设定），迫使智能体遗忘。我们发现，损坏并未对该智能体起到帮助作用……

    arXiv:2611.21648v1 Announce Type: cross  Abstract: Hoel (2021) proposed that dreams distort replayed experience so that the brain does not overfit to recent events. We test this hypothesis on an LLM agent (Claude Opus 5.5) that stores what it learns in a text memory and rewrites that memory each night from the day's conversation transcript. The agent goes through ten simulated days of conversation built from LongMemEval. We corrupt 5, 10, 20 or 40 percent of the transcript's turns before each nightly rewrite in one of two ways. Detail corruption changes one or two details in a turn to plausible wrong ones, or replaces the turn with one by the same speaker from an earlier day. Random-text corruption replaces half of the selected turns with a passage of Moby-Dick and inserts Moby-Dick words into the other half. A size limit on the memory, rising from 19,500 to 117,000 tokens over ten nights, set from human free-recall data, forces it to forget. We learned that corruption did not help the
    
[^252]: RAW：现实场景中的拒答回避——一种针对已部署大语言模型的因果测量方法

    RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs

    [https://arxiv.org/abs/2611.21641](https://arxiv.org/abs/2611.21641)

    本文提出RAW方法，利用大语言模型在相同提示下有时拒答有时不拒答的随机性作为自然实验，首次对已部署模型的拒答进行因果测量，发现一次拒答会使约一半被测模型的用户后续再参与率下降3至17个百分点。

    

    已部署的大语言模型每天处理数亿次对话，并会拒绝相当一部分用户请求，然而拒答给用户侧带来的代价——即用户是否会因此流失以及流失程度——在任何生产模型上都未曾被测量过，这对建立在这些模型之上的面向消费者的平台而言是一个盲区。在前沿实验室产品上进行拒答的A/B测试在伦理和运营层面均受限制，而对于通过固定API部署这些模型的第三方平台来说更是无法实现。我们提出了RAW方法，利用大语言模型的随机性作为自然实验：在任何现有对话日志中，面对几乎相同的提示，同一模型有时拒答、有时不拒答，这种变化使得我们能够识别拒答对用户侧行为的因果效应。将该应用于LMSYS-Chat-1M和WildChat数据集中的15个已部署大语言模型，并采用用户聚类推断，我们发现对于约一半可精确测量的模型，一次拒答会导致后续用户再参与率下降3至17个百分点。

    arXiv:2611.21641v1 Announce Type: cross  Abstract: Deployed LLMs field hundreds of millions of conversations a day and refuse a non-trivial fraction of user requests, yet the user-side cost of refusal, whether users disengage and by how much, is unmeasured on any production model: a blind spot for the consumer-facing platforms built on them. A/B-testing refusal is ethically and operationally constrained on frontier-lab products and unavailable to third-party platforms deploying these models through a fixed API. We introduce RAW, a method that uses LLM stochasticity as a natural experiment: at near-identical prompts in any existing log, the same model sometimes refuses and sometimes does not, and that variation identifies the causal user-side effect of refusal. Applying the method to 15 deployed LLMs in LMSYS-Chat-1M and WildChat, with user-clustered inference, we find that for about half of the models we can measure precisely, a refusal costs 3 to 17 pp of subsequent re-engagement; the
    
[^253]: AI能否帮助解释AI？多模态大语言模型生成的解释对XAI可视化中感知理解与感知信任的影响

    Can AI Help Explain AI? Effects of Multimodal LLM-Generated Explanations on Perceived Understanding and Perceived Trust in XAI Visualizations

    [https://arxiv.org/abs/2611.21638](https://arxiv.org/abs/2611.21638)

    本研究通过128人的预注册实验证明，为XAI可视化附加多模态大语言模型生成的文本解释能显著提升用户的感知理解与感知信任，且对缺乏XAI经验的新手效果最为显著。

    

    可解释人工智能（XAI）旨在让机器学习模型变得可理解，然而许多XAI可视化对最终用户来说仍然难以解读。本研究评估了在XAI输出基础上附加多模态大语言模型（LLM）生成的文本解释，是否能提升用户的感知理解与感知信任。我们开展了一项预注册的被试内实验（N = 128），涵盖七个决策场景和五种XAI方法，分别在有和无LLM生成解释的条件下进行测试。结果显示，这些解释使感知理解和感知信任在五点量表上分别提高了0.35分和0.26分（均p < .001）。这一提升出现在全部七个场景中，且在局部解释与全局解释之间没有差异，而对先前XAI接触最少的参与者提升最为显著。中介分析表明，LLM解释通过感知理解对感知信任产生间接影响。这些结果表明，多模态大语言模型可以提供有用的解释。

    arXiv:2611.21638v1 Announce Type: cross  Abstract: Explainable artificial intelligence (XAI) aims to make machine learning models understandable, yet many XAI visualizations remain difficult for end-users to interpret. Here, we evaluate whether augmenting XAI outputs with textual explanations from multimodal LLMs improves perceived understanding and perceived trust. We conducted a pre-registered within-subject experiment ($N = 128$) across seven decision scenarios and five XAI methods, with and without LLM generated explanations. The explanations improved perceived understanding and perceived trust by 0.35 and 0.26 points on five-point scales (both $p < .001$). The gain appeared in all seven scenarios, did not differ between local and global explanations, and was largest for participants with the least prior XAI exposure. A mediation analysis indicates an indirect effect on perceived trust through perceived understanding. These results suggest that multimodal LLMs can provide a useful 
    
[^254]: 将人机对齐作为混合软件工程团队有效协作的早期指标进行测量

    Measuring Human-AI Alignment as an Early Indicator for Effective Collaboration in Hybrid SE Teams

    [https://arxiv.org/abs/2611.21633](https://arxiv.org/abs/2611.21633)

    本文提出开发一套软件工程专用的人机对齐静态测量指标体系，借鉴软件架构、组织心理学和人机交互原理，并利用声明式业务流程监控语言构建静态评估工具，作为评估混合人机团队有效协作的早期指标。

    

    随着AI智能体日益融入软件工程（SE）流程，确保这些混合团队的有效协作成为一项关键挑战。现有指标侧重于观察动态运行时性能，但缺乏针对早期人机对齐——即AI的行为和粒度与人类期望匹配的程度——的系统性静态测量。本立场论文提出了一个研究议程，旨在借鉴软件架构、组织心理学和人机交互的原理，开发一套SE特定的对齐指标体系。此外，本文概述了一种利用业务流程监控（BPM）概念和技术的验证方法，并引入了利用声明式BPM语言的静态评估工具，以确保智能体遵守特定的项目约束。

    arXiv:2611.21633v1 Announce Type: cross  Abstract: As AI agents increasingly integrate into software engineering (SE) processes, ensuring effective collaboration in these hybrid teams becomes a critical challenge. While existing metrics focus on observing dynamic run-time performance, there is a lack of systematic, static measurements for early human-AI alignment -- the degree to which an AI's actions and granularity map to human expectations. This position paper proposes a research agenda to develop an SE-specific alignment metric suite drawing principles from software architecture, organizational psychology, and human-computer interaction. Furthermore, it outlines a validation methodology using concepts and techniques from business process monitoring (BPM) and introduces static assessment tools leveraging declarative BPM languages to ensure agents adhere to specific project constraints.
    
[^255]: 碎片化的指标，喜忧参半的结果：AI编程助手与开发者生产力的系统性综述

    Fragmented Markers, Mixed Results: A Systematic Review of AI Coding Assistants and Developer Productivity

    [https://arxiv.org/abs/2611.21632](https://arxiv.org/abs/2611.21632)

    该论文通过一项分阶段的系统性文献综述，指出现有AI编程助手对开发者生产力影响研究的核心问题在于生产力衡量指标碎片化、缺乏可比性，并基于SPACE框架将分析范围扩展至以人为本的生产力相关成果。

    

    AI编程助手正日益深度融入软件工程实践，然而现有文献仍未形成对其如何影响软件开发者生产力的累积性、易于解读的整体图景。核心问题并非研究的匮乏，而是缺乏可比性：先前的研究通过异质的衡量指标、研究情境和报告风格来操作化生产力概念，导致跨研究得出结论十分困难。为解决这一问题，我们开展了一项分阶段的系统性文献综述。在第一阶段，我们综合了明确考察开发者生产力的研究；在第二阶段，我们以SPACE框架作为有理论依据的视角，将研究范围扩展至与生产力相关的开发者成果，捕捉显性生产力研究常常忽略的以人为本的证据。在两个阶段中，我们分析了生产力是如何被操作化的、讨论了哪些混淆因素，以及已报告的效果呈现怎样的分布。

    arXiv:2611.21632v1 Announce Type: cross  Abstract: AI coding assistants are increasingly embedded in software engineering practice, yet the literature still offers no cumulative, easily interpretable picture of their effects on software developer productivity. The central problem is not a lack of studies, but a lack of comparability: prior work operationalizes productivity through heterogeneous markers, settings, and reporting styles, making cross-study conclusions difficult. To address this, we conduct a staged systematic literature review. In the first stage, we synthesize studies that explicitly examine developer productivity. In the second, we extend the scope to productivity-related developer outcomes using SPACE as a theory-informed lens, capturing human-centered evidence that explicit productivity studies often omit. Across both stages, we analyze how productivity is operationalized, which confounding factors are discussed, and how reported effects are distributed.   We identify
    
[^256]: LynnReal-Hand：通过手部控制解锁世界交互

    LynnReal-Hand: Unlocking World Interaction with Hand Control

    [https://arxiv.org/abs/2611.21479](https://arxiv.org/abs/2611.21479)

    LynnReal-Hand是一个手部控制的视频世界模型，通过手部感知监督、因果流图和区域感知分数蒸馏三项创新，实现了精确的手-物体交互建模与高效的少步流式视频生成。

    

    交互式视频世界模型可以通过文本和相机控制实现探索，但直接的对象交互需要更精细的控制。手部提供了一种自然的交互界面，但由于复杂的手-物体动力学和高要求的实时计算，将连续的手部动作转化为连贯的视觉响应仍然具有挑战性。为了解决这些挑战，我们提出了LynnReal-Hand，这是一个手部控制的视频世界模型，将精确的交互建模与高效的流式生成统一起来。我们精选了跨越多种真实和合成环境的10万个交互片段，并开发了手部感知监督来改进细粒度的手部动作和物体交互。在此基础上，我们学习了一个因果流图用于少步生成，并引入了区域感知分数蒸馏，结合互补的视频和图像先验来优化交互动力学和视觉外观。为了进一步降低推理（摘要在此处被截断）……

    arXiv:2611.21479v1 Announce Type: cross  Abstract: Interactive video world models enable exploration through text and camera controls, yet direct object interaction requires finer control. Hands provide a natural interface, but translating continuous hand movements into coherent visual responses remains challenging because of complex hand-object dynamics and demanding real-time computation. To address these challenges, we introduce LynnReal-Hand, a hand-controlled video world model that unifies precise interaction modeling with efficient streaming generation. We curate 100K interaction clips spanning diverse real and synthetic environments and develop hand-aware supervision to improve fine-grained hand motions and object interactions. Building on this foundation, we learn a causal flow map for few-step generation and introduce region-aware score distillation that combines complementary video and image priors to refine interaction dynamics and visual appearance. To further reduce infere
    
[^257]: Sage S2.1 Flash 技术报告

    Sage S2.1 Flash Technical Report

    [https://arxiv.org/abs/2611.21072](https://arxiv.org/abs/2611.21072)

    提出了 3200 亿参数的通用混合专家模型 Sage S1.5 Flash，具备跨分子图、生物序列、光谱等科学数据的感知能力，并通过可验证环境训练出工具使用、搜索、编程等智能体能力，在七个领域 33 个基准评测中表现领先。

    

    我们推出了 Sage S1.5 Flash，这是一个拥有 3200 亿参数的通用混合专家模型，旨在对科学证据进行推理并在交互式环境中执行行动。Sage Sci-Perception 采用专用的编码器处理分子图和生物序列，并将光谱表示为有序的测量值片段。各模态专属的投影器将这些表示映射到共享主干网络的输入空间中，主干网络将其与文本和图像一同处理。为了培养模型的智能体能力，我们构建了用于工具使用、搜索和编程的可验证环境，以及涉及机器学习实验和科学软件的专用工作流。任务完成情况根据执行结果、环境状态变化或提交的产物进行核验。我们在七个领域的 33 个基准上对 Sage S1.5 Flash 进行了评估。在与五个基线模型的主要对比中，它在 OfficeQA Pro（72.2）和 MLE-b（原文此处被截断）上排名第一。

    arXiv:2611.21072v1 Announce Type: new  Abstract: We introduce Sage S1.5 Flash, a 320-billion-parameter general-purpose mixture-of-experts model designed to reason over scientific evidence and act in interactive environments. Sage Sci-Perception uses dedicated encoders for molecular graphs and biological sequences and represents spectra as ordered patches of measurements. Modality-specific projectors map these representations into the input space of a shared backbone, which processes them alongside text and images. To develop the model's agent capabilities, we construct verifiable environments for tool use, search, and coding, as well as specialized workflows involving machine learning experiments and scientific software. Task completion is checked against execution results, changes in environment state, or submitted artifacts. We evaluate Sage S1.5 Flash on 33 benchmarks across seven domains. In the primary comparison against five baselines, it ranks first on OfficeQA Pro (72.2), MLE-b
    
[^258]: 声音是否重要？在已部署的临床语音代理通话中评估挫败感检测

    Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls

    [https://arxiv.org/abs/2611.21001](https://arxiv.org/abs/2611.21001)

    在真实临床语音代理通话中，患者话语极短（约59%仅一两个词），仅依赖文本转录无法可靠检测患者的挫败情绪，语音音频信息是必要的。

    

    自主语音代理如今已能进行常规的临床随访电话，因此某种自动化的分诊不可避免，而患者产生挫败感正是分诊应当捕捉的情况之一。这些系统通常只读取语音转文字阶段生成的文本转录，而不使用语音本身，这种做法是否足够，很大程度上尚未在真实临床语音上得到验证。我们在一个已部署的临床语音代理的真实患者通话中研究了逐轮次的挫败感检测：来自六个通话的150个轮次由三名标注者进行标注，并以长度匹配的表演语料库作为对照，另外还有来自30个通话的714个轮次。患者的通话轮次绝大多数非常简短：在已部署通话中，约59%的轮次仅为一两个词，而表演语料库中这一比例仅为13%；即使将表演语料库进行长度匹配，两个语料库之间的准确率差距依然存在。在单词轮次上，仅基于文本转录的检测结果无法排除随机猜测的可能性。在五个具备音频处理能力的语言模型中，转录文本仅标记出……中的3或4个（摘要原文在此处截断）。

    arXiv:2611.21001v1 Announce Type: cross  Abstract: Autonomous voice agents now conduct routine clinical follow-up calls, so some automated triage is unavoidable. A patient becoming frustrated is one thing that triage should catch. These systems typically read only the transcript the speech-to-text stage produces, not the voice, and whether that is enough is largely untested on real clinical speech. We study per-turn frustration detection on real patient calls from a deployed clinical voice agent: 150 turns from six calls labelled by three annotators, with a length-matched acted corpus as a control, and 714 turns from 30 calls. Patient turns are overwhelmingly short: across the deployed calls, roughly 59% are one or two words, against 13% in the full acted corpus, and a corpus accuracy gap remains even after the acted set is length-matched. On one-word turns, transcript-only detection does not exclude chance. Across five audio-capable language models, the transcript flags 3 or 4 of the 
    
[^259]: ProcReverser：面向处理器漏洞分析的多智能体黑盒微体系结构逆向工程

    ProcReverser: Multi-Agent Black-Box Microarchitectural Reverse Engineering for Processor Vulnerability Analysis

    [https://arxiv.org/abs/2611.20827](https://arxiv.org/abs/2611.20827)

    ProcReverser是一个多智能体系统，通过利用执行反馈自动迭代实验设计和测试程序，实现了黑盒微体系结构逆向工程的自动化，能够高效识别与Spectre类攻击相关的处理器硬件安全特性。

    

    当处理器的实现细节不可获得时，黑盒微体系结构逆向工程可以帮助安全研究人员识别与安全漏洞相关的硬件条件。然而，这一过程需要对每个处理器进行大量的人工工作，包括设计实验、开发测试程序和分析结果。我们提出了ProcReverser，一个能够自动化这一过程的多智能体系统。ProcReverser利用执行反馈来改进实验计划和测试程序，并通过测量结果识别微体系结构特性。我们开发了一套全面的特性识别任务套件，聚焦于与Spectre类攻击相关的硬件机制，涵盖投机执行、分支预测、内存依赖和缓存行为。我们在玄铁C910、gem5 RISC-V O3 CPU模型和SiFive P550上对ProcReverser进行了评估，其Pass@1和Pass@5分数分别达到81.3%和93.8%。

    arXiv:2611.20827v1 Announce Type: cross  Abstract: When processor implementation details are unavailable, black-box microarchitectural reverse engineering helps security researchers identify hardware conditions related to security vulnerabilities. However, this process requires substantial manual effort to design experiments, develop test programs, and analyze results for each processor. We present ProcReverser, a multi-agent system that automates this process. ProcReverser uses execution feedback to improve experimental plans and test programs and identifies microarchitectural properties from measurements. We develop a comprehensive suite of property identification tasks focused on hardware mechanisms relevant to Spectre-class attacks, covering speculative execution, branch prediction, memory dependence, and cache behavior. We evaluate ProcReverser on the XuanTie C910, the gem5 RISC-V O3 CPU model, and the SiFive P550. ProcReverser achieves Pass@1 and Pass@5 scores of 81.3% and 93.8%,
    
[^260]: 零售预测中世界模型解释的审计：忠实性、准确性与价格响应

    Auditing World-Model Explanations in Retail Forecasting: Faithfulness, Accuracy and Price Response

    [https://arxiv.org/abs/2611.20811](https://arxiv.org/abs/2611.20811)

    研究发现，零售预测中的世界模型解释即使通过了八项审计检查（如构造性、稳定性、价格方向和与真实销量的一致性），其促销提升估计仍可能高出真实值1.2至1.4倍，价格响应被夸大近5至6.5倍，表明通过常规检查并不保证解释的忠实性与准确性。

    

    能够读取价格与促销计划的预测模型，将每个预测解释为两部分之和：一个基线（即不执行任何计划时的销量），加上该计划带来的提升。我们通过八项检查来检验此类解释，其中六项在任何结果出现之前就已固定了通过规则，结果发现通过这些检查并不意味着解释是正确的。在效应已知的人工生成门店数据中，一个世界模型通过了构造性、稳定性、价格方向以及与销量和LightGBM一致性等各项检查，但当商家将促销计划安排在高需求周时，其提升值是真实值的1.33至1.40倍；在该世界的八种变体中，这一比例为1.23至1.42倍。它仍然能选对促销的周次，但根据其提升值订购的库存与真实提升值相比偏差达0.44至0.48。它所宣称的对长期常规价格变动的响应是真实值-0.5的4.7至6.5倍，而只有通过实际测量才能发现这一点。在Dominick's数据集上，38个门店常规价格偏离其z（摘要在此处截断）

    arXiv:2611.20811v1 Announce Type: cross  Abstract: Forecasting models that read the price and promotion plan explain each forecast as a baseline, what would sell with nothing planned, plus the lift the plan adds. We test such explanations with eight checks, six with a pass rule fixed before any result, and find that passing them does not make an explanation right. In generated store data with known effects, a world model passes the checks of construction, stability, price direction and agreement with sales and with LightGBM, yet its lifts are 1.33 to 1.40 times the truth when merchants plan into high-demand weeks, and 1.23 to 1.42 times in eight variants of that world. It still picks the right weeks to promote, but stock ordered to its lift is off by 0.44 to 0.48 of the true lift. Its stated response to a lasting regular-price change is 4.7 to 6.5 times the true -0.5, and only a measured cause finds this. On Dominick's, 38 episodes in which a store's regular prices sat apart from its z
    
[^261]: 基于大语言模型的提升估计中的处理信息泄漏正则化

    Regularizing Treatment Leakage for LLM-based Uplift Estimation

    [https://arxiv.org/abs/2611.20794](https://arxiv.org/abs/2611.20794)

    该论文提出LULR方法，通过在冻结LLM编码器的同时对因果头共享表示施加基于CLUB的互信息惩罚，抑制LLM编码文本中泄露处理信息的捷径，从而改进工业定价RCT场景下的CATE提升估计效果。

    

    提升建模用于估计条件平均处理效应（CATE），并可支持促销定价等工业决策。当大语言模型（LLM）对产品文本进行编码时，暴露处理信息的内容可能引入无法迁移到实际部署阶段的捷径。我们提出了基于LLM的提升泄漏正则化方法（LULR），该方法在保持LLM编码器冻结的前提下，对因果头的可训练共享表示施加基于CLUB的惩罚项。受控代理实验显示出正的排序分数交互效应，且得分高于所测试的权重衰减配置；依赖处理的评估使其解释仅限于探索性的模型行为。仅使用结构化特征的对比提供了参考性能，但并未确立相对基于树的估计器的普遍优势。在一项包含百万条记录的工业定价随机对照试验（RCT）上，LULR在5个随机种子中的4个上改进了离线lift@0.3指标，尽管（摘要原文在此处截断）。

    arXiv:2611.20794v1 Announce Type: new  Abstract: Uplift modeling estimates the conditional average treatment effect (CATE) and supports industrial decisions such as promotional pricing. When large language models (LLMs) encode product text, treatment-revealing content can introduce shortcuts that do not transfer to deployment. We propose \emph{LLM-based Uplift Leakage Regularization} (\textsc{LULR}), which uses a CLUB-based penalty on the causal head's trainable shared representation while keeping the LLM encoder frozen. Controlled proxy experiments show a positive ranking-score interaction and higher scores than the tested weight-decay configuration; treatment-dependent evaluation limits their interpretation to exploratory model behavior. Structured-only comparisons provide reference performance, without establishing a general advantage over tree-based estimators. On a million-record industrial pricing RCT, \textsc{LULR} improves offline $\text{lift@}0.3$ in $4/5$ seeds, although the 
    
[^262]: AI能评判你吗？对LLM辅助招聘系统部署就绪度的实证审计

    Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems

    [https://arxiv.org/abs/2611.20140](https://arxiv.org/abs/2611.20140)

    本文提出了一种将候选人证据、评估、权限控制与评估记录分离的LLM辅助简历筛选系统架构，并通过涵盖210份简历、14个职位和6种决策界面的受控实验，从运行就绪度与合规就绪度两个维度实证审计了其部署就绪度，发现标量分数比通过/不通过判断更能有效区分资质变化。

    

    将简历筛选工作委托给大语言模型（LLM）时，组织需要确认系统能否依据相关资质区分候选人，同时只允许被授权的用户修改筛选规则或批准例外。为此，我们开发了一种将候选人证据、评估、权限控制和评估记录相互分离的系统架构，并从两个互补的维度对其部署就绪度进行了实证检验。第一，我们通过一项涵盖210份简历、14个职位角色和6种决策界面的受控实验，评估了该系统在4个LLM上的运行就绪度。第二，我们通过测试系统如何应对未授权用户的请求以及嵌入在简历中的指令，评估了其合规就绪度。我们发现，运行性能取决于评估界面的选择：标量分数通常比简单的通过/不通过判断更能有效地区分资质变化，而比较类……（原文摘要在此处截断）

    arXiv:2611.20140v1 Announce Type: cross  Abstract: Delegating resume screening to large language models (LLMs) requires organizations to establish whether they can distinguish candidates by relevant qualifications while allowing only authorized users to change screening rules or approve exceptions. To this end, we develop a system architecture that separates candidate evidence, evaluation, permission controls, and assessment record, and empirically examine two complementary dimensions of its deployment readiness. First, we assess its operational readiness on four LLMs using a controlled experiment on 210 resumes across 14 job roles and six decision interfaces. Second, we assess its compliance readiness by testing how it handles unauthorized user requests and instructions embedded in resumes. We find that operational performance depends on the choice of evaluation interface: scalar scores generally distinguish qualification changes better than simple pass/fail judgments, while comparati
    
[^263]: 什么决定了跨模态对齐的成败

    What Matters for Cross-Modal Alignment

    [https://arxiv.org/abs/2611.20000](https://arxiv.org/abs/2611.20000)

    该研究揭示了跨模态对齐成败的关键在于模态共变子空间所承载的方差比例，并据此提出LightAlign方法，仅需少量图像-文本配对数据通过岭回归即可完成两个冻结编码器之间的对齐。

    

    柏拉图表示假说认为，在不同模态数据上训练的模型会收敛到一种共享的表示。已有相关工作在没有配对数据的情况下对自然图像上的冻结图像编码器和文本编码器进行对齐。我们在涵盖四种模态（自然图像、环境音频、胸部X光片和3D计算机断层扫描CT）的五个数据集上测试了零配对对齐。该方法仅在COCO数据集上成功，而在其他四个数据集上均告失败。我们对这些失败进行了分析，发现对齐必须首先定位模态之间发生共变的子空间，然后在该子空间内确定映射方向。与此前的研究一致，医学成像是其中最困难的情况：其共享子空间所承载的方差比例最小。当这一比例较小时，对齐需要图像-文本配对数据，而随机矩阵理论可以预测需要多少配对数据。这些发现启发了LightAlign方法：仅需少量配对数据，通过两个冻结编码器之间的岭回归即可定位（摘要此处截断）

    arXiv:2611.20000v1 Announce Type: cross  Abstract: The Platonic Representation Hypothesis posits that models trained on different modalities converge toward a shared representation. Related work aligns frozen image and text encoders on natural images without paired data. We test zero-pair alignment on five datasets across four modalities: natural images, environmental audio, chest radiographs, and 3D computed tomography (CT). It succeeds only on COCO and fails on the other four. We analyze the failures and find that alignment must first locate the subspace where the modalities co-vary and then orient the mapping within it. Consistent with previous studies, medical imaging is the hardest case: its shared subspace carries the smallest fraction of the variance. When this fraction is small, alignment needs image-text pairs, and random-matrix theory predicts how many pairs are needed. These findings motivate LightAlign: a few pairs let a ridge regression between two frozen encoders locate a
    
[^264]: 免重放循环状态迁移的信息极限

    Information Limits of Replay-Free Recurrent State Migration

    [https://arxiv.org/abs/2611.19345](https://arxiv.org/abs/2611.19345)

    本文刻画了循环模型在不重放历史数据的条件下进行状态迁移所必须保留的最小信息量，证明了带标量输入的可控d状态模型至多需要kd个额外实数坐标，且单个秩一替换方向即可要求全部kd个。

    

    当一个循环（递归）模型被替换时，其旧状态可能无法决定新模型基于相同历史输入会计算出什么。本文刻画了在替换方案未知之前必须保留的信息。对于时不变的仿射线性模型族，阶数为k的局部迁移所需的最小连续因果记忆，等于提升系数系统的可达维数减去锚定状态所含的信息。对于带标量输入的可控d状态模型，即使任意的矩阵替换方向仅在流式处理之后才被揭示，也至多需要kd个额外的实数坐标；而单个秩一方向就可能已经要求全部kd个坐标。条件奇异值与辅助比特界将该局部信息定律与固定容差下的恢复区分开来。作者还给出了不变子空间与未来输出相关的构造，并通过受控实验比较了由此产生的各种状态表示。在一个小型语言模型中……（摘要在此处截断）

    arXiv:2611.19345v1 Announce Type: new  Abstract: When a recurrent model is replaced, its old state may not determine what the new model would compute from the same history. We characterize the information that must be retained before the replacement is known. For a time-invariant affine linear family, the minimum continuous causal memory for order-$k$ local migration equals the reachable dimension of a lifted coefficient system minus the information in the anchor state. A controllable $d$-state model with scalar input needs at most $kd$ additional real coordinates, even when arbitrary matrix replacement directions are revealed only after streaming. A single rank-one direction can already require all $kd$. Conditional singular values and auxiliary-bit bounds distinguish this local information law from recovery at a fixed tolerance. We also give invariant-subspace and future-output constructions. Controlled experiments compare the resulting representations. In a small language model, equ
    
[^265]: 離题：面向结构化数据集目标变量预测与分类的零样本权重空间编程

    Zero-Shot Weight-Space Programming for Prediction and Classification of Target Variables in Structured Datasets

    [https://arxiv.org/abs/2611.19292](https://arxiv.org/abs/2611.19292)

    该论文证明大语言模型可以在零样本条件下，仅凭展示的小数据集与指定网络架构，无需优化器、梯度或代码执行，就手写出前馈网络的全部权重和偏置，在从玩具问题、合成问题到经典基准和真实工程任务的测试中直接完成结构化数据的目标变量预测与分类。

    

    权重空间编程是指以手工方式固定前馈网络的每一个参数，整个构建过程中不使用任何梯度下降。迄今为止的所有演示都是从以规则系统形式提供给模型的目标函数出发，这使其成为一个编译问题。而监督数据集并不提供这种规范说明，因此该过程必须自行完成统计推断，然后将结果用给定的参数化形式表达出来。我们探究大语言模型能否同时做到这两点。实验中向模型完整展示一个小数据集，并给出前馈网络的架构，由其将每一个权重和偏置都写成字面数字——全程不使用优化器、不使用梯度、不执行代码，也没有机会先进行评估。每个构建结果都被封存并预先登记其预期得分。评分测试套件从两个玩具问题开始，经过九个由隐藏旋转加以伪装的合成问题，直至经典基准测试以及一个真实工程问……

    arXiv:2611.19292v1 Announce Type: new  Abstract: Weight-space programming fixes every parameter of a feed-forward network by hand, with no gradient descent anywhere in the construction. Every demonstration so far starts from a target function handed to the model as a rule system, making it a compilation problem. A supervised dataset withholds that specification, so the procedure must do the statistical inference itself and then express the result in the parameterisation it was given. We ask whether a large language model can do both. It is shown a small dataset in full, with the architecture of a feed-forward network, and writes out every weight and bias as a literal number, with no optimizer, no gradient, no code execution, and no chance to evaluate it first. Each construction is sealed and pre-registers its expected score. The graded suite runs from two toy problems, through nine synthetic problems disguised by a hidden rotation, to the classical benchmarks and one real engineering r
    
[^266]: Gamer：面向长时程实时游戏智能体的通用视觉-语言-动作基础模型

    Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent

    [https://arxiv.org/abs/2611.18397](https://arxiv.org/abs/2611.18397)

    提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。

    

    我们提出了Gamer，一个专为复杂交互环境中长时程任务构建的通用智能体框架。其端到端架构结合了感知、记忆管理和多步推理。统一的训练与部署流程支持与常规2D图形界面以及复杂3D模拟世界的交互。我们首先在从数万小时设备级键盘和鼠标交互中构建的数据集上进行继续预训练，这些轨迹涵盖模拟游戏和具身环境，使模型能够学习跨物理与虚拟领域的环境动态。随后的稀疏思考指令微调在保留复杂决策所需推理能力的同时，大幅减少了冗余计算。我们采用一种稳定的多轮强化学习方法来训练策略以实现长时程交互，该方法解决了有限上下文窗口、累积（误差/奖励）等问题（摘要在此处被截断）。

    arXiv:2611.18397v1 Announce Type: cross  Abstract: We introduce Gamer, a generalist agent framework built for long-horizon tasks in complex interactive environments. Its end-to-end architecture combines perception, memory management, and multi-step reasoning. A shared training and deployment pipeline supports interaction with both conventional 2D graphical interfaces and complex 3D simulated worlds. We first continue pre-training on a dataset constructed from tens of thousands of hours of device-level keyboard and mouse interaction. These trajectories span simulated games and embodied environments, allowing the model to learn environmental dynamics across both physical and virtual domains. Sparse-thinking instruction tuning then substantially reduces redundant computation while preserving the reasoning needed for complex decisions. We train the policy for long-horizon interaction with a stable multi-turn reinforcement learning method that addresses finite context windows, accumulated r
    
[^267]: Marimba：一个面向语音的“系统一”模型

    Marimba: A System One Model for Speech

    [https://arxiv.org/abs/2611.18330](https://arxiv.org/abs/2611.18330)

    Marimba基于冻结的Qwen3-ASR 0.6B特征构建了一个轻量级“系统一”语音决策接口，通过零样本评分、条件化读出头和中间层探针，在词项检测、轮次结束判断、语言识别、个人信息检测和说话人变化检测等多项语音决策任务上取得了优异且良好校准的性能。

    

    语音应用需要在音频窗口内做出关于词项是否存在、轮次是否结束以及语言种类的决策。我们提出了Marimba，一个基于冻结的Qwen3-ASR 0.6B特征构建的类型化语音决策接口。词汇类问题使用ASR转录文本的零样本评分；非词汇类问题使用问题条件化读出头和独立的第18层轮次结束读出；语言识别则使用ASR模型自身的语言评分。在私有会话测试集上，解码器AUROC为0.80-0.92；在公开朗读语音上为0.92-0.97。Speech Commands四分类准确率为0.93；二分类词汇任务的ECE不超过0.08。个人信息检测通过将贪心转录传递给私有的、经过整理的PII流水线，在合成的巴西葡萄牙语语音上达到条目级F1 0.904（在参考转录上为1.000）。第12层说话人读出在AMI和VoxConverse数据集上以AUROC 0.966-0.980检测说话人变化，高于ECAPA基线，并且

    arXiv:2611.18330v1 Announce Type: cross  Abstract: Speech applications require decisions about term presence, turn completion, and language within an audio window. We present Marimba, an interface for typed speech decisions built on frozen Qwen3-ASR 0.6B features. Lexical questions use zero-shot scoring of ASR transcripts; non-lexical questions use a question-conditioned head and a separate layer-18 turn-end readout. Language identification uses the ASR model's own language scores. On private conversational test sets, decoder AUROC is 0.80-0.92; on public read speech it is 0.92-0.97. Speech Commands four-way accuracy is 0.93; binary lexical ECE is at most 0.08. Personal-information detection, which passes the greedy transcript to a private, curated PII pipeline, reaches item-level F1 0.904 on synthetic Brazilian Portuguese speech (1.000 on reference transcripts). Layer-12 speaker readouts detect speaker change with AUROC 0.966-0.980 on AMI and VoxConverse, above an ECAPA baseline, and 
    
[^268]: RACE：面向高效序列推荐的排序不确定性感知容量自适应定制

    RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation

    [https://arxiv.org/abs/2611.18244](https://arxiv.org/abs/2611.18244)

    提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。

    

    高效序列推荐已成为提供实时个性化服务的一种有前景的范式。为了在资源受限的环境下部署模型，模型压缩成为必要手段，然而现有方法通常采用统一的压缩比率，忽视了用户意图复杂性的异质性。这种“一刀切”的方式导致了根本性的“容量-复杂度失配”问题，即计算资源要么浪费在简单实例上，要么对复杂实例而言不足。为解决这一问题，我们提出了RACE（排序不确定性感知的容量自适应定制框架），该框架利用不确定性感知的压缩比率，动态调整模型结构和输入序列。为实现容量自适应推理，RACE捕获模型预测的内在不确定性，并将其转化为一种主动的资源分配器……

    arXiv:2611.18244v1 Announce Type: cross  Abstract: Efficient sequential recommendation has emerged as a promising paradigm for providing real-time personalized service. To deploy models under resource-constrained environment, model compression becomes a necessity, yet existing methods typically apply uniform compression ratios, overlooking the heterogeneity in user intent complexity. This "one-size-fits-all" approach leads to a fundamental Capacity-Complexity Mismatch, where computational resources are either wasted on simple instances or insufficient for complex ones. To address this, we propose RACE (Ranking-uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation), a framework that dynamically adapts both model structures and input sequences with uncertainty-aware compression ratios. To ensure capacity-adaptive inference, RACE captures the model's intrinsic uncertainty of its predictions and transforms it into a proactive resource allocator. Specific
    
[^269]: EgoX：基于具身梦境的第一人称视角跨具身操作

    EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming

    [https://arxiv.org/abs/2611.16728](https://arxiv.org/abs/2611.16728)

    EgoX提出了一种第一人称视角的跨具身学习框架，通过为每个观测和动作模态配置专用分词器与专家模块、共享Transformer主干，并借助仅在训练阶段使用的“具身梦境”未来状态预测目标，使通用操作策略能够有效利用人类、真实机器人和仿真智能体的异构演示数据。

    

    通用操作策略可以从人类、真实机器人和仿真智能体的互补经验中受益。然而，三者之间在形态结构、感知方式和控制接口上的差异，使得从它们组合的演示数据中进行有效学习变得困难。我们提出了EgoX，一个第一人称视角的跨具身学习框架，通过语义上相互对应的观测模态和动作模态来组织异构演示数据。EgoX在这一混合数据上预训练一个带具身梦境的模块化跨具身Transformer（MXT+），并在目标具身的演示数据上进行微调。在每个具身体内，MXT+为每个观测模态分配一个专用的分词器，为每个动作模态分配一个专用的专家，而Transformer主干则在所有具身之间共享。一个仅在训练阶段使用的具身梦境目标函数用于预测未来视觉状态和本体感觉状态的潜在表示，从而鼓励共享主干……（原文摘要在此处截断）

    arXiv:2611.16728v1 Announce Type: cross  Abstract: Generalist manipulation policies can benefit from the complementary experience of humans, real robots, and simulated agents. However, differences in morphology, sensing, and control interfaces make it difficult to learn effectively from their combined demonstrations. We introduce EgoX, an egocentric cross-embodiment learning framework that organizes heterogeneous demonstrations through semantically corresponding observation and action modalities. EgoX pretrains a Modularized Cross-Embodiment Transformer with Embodiment Dreaming (MXT+) on this mixture and fine-tunes it on target-embodiment demonstrations. Within each embodiment, MXT+ assigns a dedicated tokenizer to each observation modality and a dedicated expert to each action modality, with a Transformer trunk shared across embodiments. A training-only embodiment-dreaming objective predicts latent representations of future visual and proprioceptive states, encouraging the shared trun
    
[^270]: 规格驱动的属性测试（Spec-Driven PBT）：从代码规格说明中推导语义不变量以实现智能体式基于属性的测试

    Spec-Driven PBT: Deriving Semantic Invariants from Code Specifications for Agentic Property-based Testing

    [https://arxiv.org/abs/2611.15920](https://arxiv.org/abs/2611.15920)

    该论文提出规格驱动的属性测试方法，利用智能体编程中已有的自然语言规格说明（而非仅从代码实现）来推导语义不变量，从而更有效地测试AI生成的代码。

    

    软件越来越多地由AI智能体编写，即智能体根据自然语言规格说明生成代码实现，这种做法被称为智能体编程。如今代码的产生速度已超过人工审查的速度，并且已被证明会表现出其特有的失败模式。基于属性的测试是一种历史悠久的测试技术，它通过大量自动生成的输入来检验程序的不变属性。最近，这一技术被重新审视，由LLM智能体来编写测试属性，从而消除了长期以来阻碍其推广的高成本不变量推导工作。然而，现有方法仅从代码实现本身推导这些属性（即代码驱动的PBT）。与此同时，详尽的规格说明（如产品需求文档）已成为智能体编程流程中的常规产物，但其在测试所生成代码中的作用仍鲜有探索。我们研究了规格说明作为测试属性来源的价值……

    arXiv:2611.15920v1 Announce Type: cross  Abstract: Software is increasingly written by AI agents that produce an implementation from a natural language specification, a practice known as agentic coding. Code is now produced faster than it can be reviewed, and has been shown to exhibit its own characteristic failure patterns. Property-based testing (PBT), a long-standing technique in which invariant properties of a program are checked against many generated inputs, has recently been revisited with LLM agents authoring the properties, thereby removing the high-effort derivation of invariants that has long hindered its adoption. Existing methods, however, derive these properties from the implementation alone (Code-driven PBT). At the same time, a thorough specification such as a product requirements document has become routine within agentic coding, yet its role in testing the resulting code remains underexplored. We study the value of the specification as a source of properties for testi
    
[^271]: ReShap：面向高维多动作环境的可解释强化学习

    ReShap: Interpretable Reinforcement Learning for High-Dimensional Multi-Action Settings

    [https://arxiv.org/abs/2611.15507](https://arxiv.org/abs/2611.15507)

    ReShap 是一个策略蒸馏框架，通过从原始图像中提取特征、执行 SHAP 特征归因，并将深度强化学习策略蒸馏为在完整状态-动作对上训练的可解释代理模型，从而将基于 Shapley 值的可解释强化学习扩展到了高维、多动作的环境。

    

    arXiv:2611.15507v1 公告类型：交叉。摘要：深度强化学习（RL）在应对复杂任务方面发挥着关键作用，但其对深度神经网络的依赖使得学习到的策略难以解释。为了增强强化学习策略的可信度，现有工作（例如 SILVER 框架 [Li et al., 2025]）聚焦于基于 Shapley 值的可解释性，但通常局限于低维状态空间、二元动作设置以及单一类别的线性代理模型。我们提出了 ReShap，这是一个策略蒸馏框架，将基于 Shapley 值的可解释性扩展到高维、多动作的环境。ReShap 从原始图像观测中提取紧凑的特征表示，进行基于 SHAP 的特征归因以刻画每个特征对策略决策的贡献，并将深度强化学习策略蒸馏为一个可解释的代理模型，该代理模型基于策略访问分布上的完整（状态，策略动作）对集合进行训练。我们实例化了（摘要在此处被截断）

    arXiv:2611.15507v1 Announce Type: cross  Abstract: Deep Reinforcement Learning (RL) plays a crucial role in tackling complex tasks, yet its reliance on deep neural networks makes the learned policies opaque. To enhance the trustworthiness of RL policies, existing work (e.g., the SILVER framework [Li et al., 2025]) has focused on Shapley-based interpretability, but is typically restricted to low-dimensional state spaces, binary-action settings, and a single class of linear surrogate. We propose ReShap, a policy-distillation framework that extends Shapley-based interpretability to high-dimensional, multi-action environments. ReShap extracts compact feature representations from raw image observations, performs SHAP-based feature attribution to characterize how each feature contributes to the policy's decisions, and distills the deep RL policy into an interpretable surrogate trained on the full set of (state, policy-action) pairs along the policy's visited distribution. We instantiate the 
    
[^272]: SkillEarth：通过技能进化教会通用智能体在地球上执行任务

    SkillEarth: Teaching General-Purpose Agents to Act on Earth via Skill Evolution

    [https://arxiv.org/abs/2611.15481](https://arxiv.org/abs/2611.15481)

    SkillEarth 提出了一种自我进化的技能提取方法，无需专家干预或参数更新，即可从演示轨迹中自动提取遥感工具调用知识，并将其外化为可复用、可检索、可进化的技能文件，从而使通用智能体能够持续适应动态地球环境。

    

    近期开发的遥感智能体通过多步骤工具调用来执行具有复杂工作流的现实世界任务。然而，这些智能体通常依赖于预定义的专业知识或参数更新。因此，每当出现新的工具、任务或执行模式时，它们都需要更新专家知识或重新训练模型，这限制了它们在动态地球环境中的持续适应能力。在这项工作中，我们旨在为通用智能体配备可从演示轨迹中自动提取的遥感工具调用知识，而无需专家干预或参数更新。具体而言，我们提出了 SkillEarth，这是一种自我进化的技能提取方法，它将遥感工具调用知识外化为可复用、可检索、可进化的技能文件，并通过基于观察的进化对其进行精炼。与现有的依赖基于结果的反思来整体精炼技能的自动技能提取方法不同，……

    arXiv:2611.15481v1 Announce Type: new  Abstract: Recently developed remote sensing (RS) agents execute real-world tasks with complex workflows through multi-step tool invocation. These agents, however, typically rely on predefined expertise or parameter updates. They thus require updating expert knowledge or retraining models whenever new tools, tasks, or execution patterns emerge, limiting their continual adaptability in dynamic Earth environments. In this work, we seek to equip general-purpose agents with RS tool-call knowledge that can be automatically extracted from demonstrated trajectories without expert intervention or parameter updates. Specifically, we propose SkillEarth, a self-evolving skill extraction method that externalizes RS tool-call knowledge as reusable, retrievable, and evolvable skill files and refines them through observation-grounded evolution. Unlike existing automatic skill extraction methods that rely on outcome-based reflection to refine skills holistically, 
    
[^273]: 重新思考“RL泛化，SFT记忆”：SFT数据的作用

    Rethinking "RL Generalizes, SFT Memorizes": The Role of SFT Data

    [https://arxiv.org/abs/2611.14912](https://arxiv.org/abs/2611.14912)

    SFT泛化不佳并非模型固有缺陷而是数据问题——精心混合数据来源并匹配RL的正确采样数量或总采样预算，可使SFT的泛化能力追平甚至超过RL。

    

    使用可验证奖励的强化学习（RL）训练的大语言模型展现出强大的推理能力和广泛的泛化能力，而通过监督微调（SFT）训练的模型通常被认为更容易产生记忆化且迁移能力有限。本文在广泛使用的Qwen2.5和Qwen3系列模型上，从SFT训练数据的视角重新审视了这一区别。首先，我们研究了数据来源的作用，并证明其至关重要：精心混合的SFT数据集显著优于仅由更大模型生成的数据。其次，我们研究了数据规模的作用，结果表明使SFT与RL的正确rollout数量相匹配可大幅提升SFT的泛化能力，而匹配总rollout预算则能使SFT的泛化能力与RL相当。将这两个因素相结合，可进一步使SFT的泛化能力甚至超越RL。第三，通过使用LLM标注来刻画解法方法……（摘要不完整）

    arXiv:2611.14912v1 Announce Type: cross  Abstract: Large language models trained with Reinforcement Learning (RL) with verifiable rewards exhibit strong reasoning ability and broad generalization, whereas models trained with Supervised Fine-Tuning (SFT) are often viewed as more prone to memorization and limited transfer. This paper rethinks this distinction through the lens of SFT training data on widely used Qwen2.5 and Qwen3 series models. First, we study the role of data source and show that it is critical: a carefully mixed SFT dataset substantially outperforms data generated solely by a larger model. Second, we study the role of data scale and show that matching the number of correct rollouts between SFT and RL greatly improves SFT generalization, while matching the total rollout budget enables SFT to generalize as well as RL. Combining these two factors further enables SFT to generalize even better than RL. Third, by using LLM annotations to characterize the solution methods in t
    
[^274]: OverServe：通过工具—提示词协同设计对SaaS智能体实施的资源放大攻击

    OverServe: Resource Amplification Attacks on SaaS Agents via Tool--Prompt Co-Design

    [https://arxiv.org/abs/2611.14808](https://arxiv.org/abs/2611.14808)

    提出了OverServe多智能体攻击框架，通过协同设计MCP状态工具与完成确认提示词，诱导SaaS智能体在业务任务完成后仍反复检查未完成条件，从而放大对服务提供商资源的消耗。

    

    大型语言模型（LLM）的进步使软件即服务提供商能够借助智能体自动化更长的业务工作流。在基于订阅的SaaS部署中，模型调用和工具调用会消耗提供商的资源，而客户无需为此额外付费。由于智能体自主决定何时停止执行，恶意租户可以向智能体发送提示词，利用其完成判定机制来延长执行时间并浪费提供商的资源。在本文中，我们提出了OverServe，一个用于构建此类针对SaaS智能体攻击的多智能体框架。OverServe为攻击者协同设计模型上下文协议（MCP）状态工具与完成确认提示词。每次工具调用都会返回准确的状态并正常终止，但智能体可能会为了满足提示词的要求而反复检查尚未完成的条件。这使得即使在原始业务任务完成之后，SaaS提供商的资源仍会被持续消耗。

    arXiv:2611.14808v1 Announce Type: cross  Abstract: Advances in large language models (LLMs) are enabling software-as-a-service (SaaS) providers to automate longer business workflows with agents. In subscription-based SaaS deployments, model and tool calls consume provider resources without additional customer payments. Because the agent autonomously decides when to stop, a malicious tenant can send a prompt to the agent, exploiting its completion decisions to extend execution and waste provider resources. In this paper, we present OverServe, a multi-agent framework for constructing such attacks against SaaS agents. OverServe co-designs Model Context Protocol (MCP) status tools and completion-confirmation prompts for the attacker. Each tool call returns accurate status and terminates normally, yet the agent may repeatedly check an unfinished condition to satisfy the prompt. This consumes the SaaS provider's resources even after the original business task is completed. OverServe construc
    
[^275]: Continual-ARC：测量整个职业生涯中的技能习得

    Continual-ARC: Measuring Skill Acquisition over a Working Lifetime

    [https://arxiv.org/abs/2611.14505](https://arxiv.org/abs/2611.14505)

    提出 Continual-ARC 基准，基于三个 ARC-AGI 版本构建终身任务流，使基于记忆和基于权重的持续学习系统能够在同一预算评分标准下公平比较技能习得能力。

    

    智能体框架已成为一种有前景的持续学习机制，与测试时参数化学习并行发展。然而，当前的基准测试存在若干局限性：它们通常只关注单一应用领域，每个实例只提供一次，并且假设学习者只有单一类型，因此基于记忆的系统和基于权重的系统很少在同一尺度上进行比较。我们提出了 Continual-ARC，这是一个由三个 ARC-AGI 版本构建的终身任务流，其任务无需领域知识。相同的任务会在休眠间隔后以新实例的形式再次出现，同时新任务也不断到来。在每个实例上，学习者可以付费获取演示、提交答案，或在框架内执行操作；提交错误时允许重试，但重试需付费。每个实例的得分基于所消耗的反馈预算比例。由于学习者可以在实例之间保留任何内容，基于记忆和基于权重的系统在同一标准下进行评分。

    arXiv:2611.14505v1 Announce Type: new  Abstract: Agent harnesses have emerged as a promising mechanism for continual learning, alongside test-time parametric learning. However, the current benchmarks have several limitations: they generally focus on a single applied domain, serve each instance only once, and assume a single family of learner, so memory-based and weight-based systems are rarely compared on the same scale. We propose Continual-ARC, a lifelong stream built from the three ARC-AGI editions, whose tasks need no domain knowledge. The same tasks return after dormant gaps with fresh instances, while new tasks keep arriving. On each instance, the learner may pay for demonstrations, submit answers, or act inside the harness, with retries permitted on incorrect submissions, which are paid for. Each instance is scored based on the share of this feedback budget consumed. Since a learner may keep anything between instances, memory-based and weight-based systems are scored on the same
    
[^276]: 绝对零度下的注意力机制

    Attention at Absolute Zero

    [https://arxiv.org/abs/2611.11937](https://arxiv.org/abs/2611.11937)

    该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。

    

    注意力机制对答案的候选支持（我们称之为“见证者”）进行评分，并在某个温度下读取这些分数。我们构建了一个高阶注意力层，其状态存在于物品对之上：对于一对，每个第三方物品 k 都是一个见证者，通过其两条边 和 进行评判。该层在零温度下读取见证者（保留最佳者及其对应的计数），只传递比较这些计数的比特，在每次循环中从输入和这些比特重建其配对状态，并且仅依靠答案进行训练。每一项设计选择都消除了软注意力在规模扩大时的一种失效模式：权重分散、学习到的分数之间无法区分的差距，以及漂移的软状态。在最大值检索任务中，该模型在比训练集大1,000倍的集合上，于全部十个随机种子下均达到至少99.95%的准确率；而使用软计数训练的同一网络在4,096个项目时性能降至42%。在无提示的CLRS-30基准上，该层的一般形式（其一半通道在零温度下读取……）

    arXiv:2611.11937v2 Announce Type: replace-cross  Abstract: Attention scores candidate supports of an answer, which we call witnesses, and reads the scores at a temperature. We build a higher-order attention layer whose state lives on pairs of items: for a pair (a, b), each third item k is a witness judged through its legs (a, k) and (k, b). It reads witnesses at zero temperature, keeping the best and the number tied with it, passes on only bits comparing such counts, rebuilds its pair state from the input and these bits at every loop, and is trained from answers alone. Each choice removes one way soft attention fails at size: dispersed weight, unresolved gaps between learned scores, and a drifting soft state. In maximum retrieval it is right on at least 99.95% of sets 1,000 times larger than in training, on all ten seeds; trained with a soft count, the same network falls to 42% at 4,096 items. On CLRS-30 without hints, a general form of the layer (half its channels read at zero tempera
    
[^277]: Lightfoot：具有精确、无模型文档删除功能的图检索增强生成

    Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion

    [https://arxiv.org/abs/2611.11010](https://arxiv.org/abs/2611.11010)

    Lightfoot 是一种图检索增强生成系统，它将图中的每个实体、链接、事实和同义词边都定义为逐段落提取结果的确定性函数，从而实现无需语言模型调用的检索以及作为数据库事务的精确文档删除（在 11,402 段语料库上删除 10% 仅需 14 秒且结果与全新索引完全一致），解决了现有 GraphRAG 系统的模型调用成本和文档删除不精确问题。

    

    基于图的检索增强生成（GraphRAG）提升了多跳问答能力，但已部署的系统还需承担研究对比中很少被测量的成本：查询时的模型调用、图的构建以及正确地删除文档。我们提出了 Lightfoot，这是一种 GraphRAG 系统，其中每个实体、链接、事实和同义词边都是携带块级溯源信息的逐段落提取结果的确定性函数。检索过程无需调用语言模型，索引过程每个段落仅需一次调用，而删除文档则是一个无需模型调用的数据库事务：在包含 11,402 个段落的语料库上删除 10% 的文档仅需 14 秒，之后得到的图与对剩余文档重新构建的新索引完全一致。在一个包含大量近似重复名称的语料库上，HippoRAG 2 的同义词列表上限导致其结果比重建少 565 条边，而 Lightfoot 始终保持精确。在 MuSiQue 和 2WikiMultiHopQA 的保留测试问题上，使用相同的提取、嵌入和答案模型……

    arXiv:2611.11010v2 Announce Type: replace  Abstract: Graph-based retrieval-augmented generation (GraphRAG) improves multi-hop question answering, but deployed systems pay costs that research comparisons rarely measure: model calls at query time, building the graph, and removing documents correctly. We present Lightfoot, a GraphRAG system in which every entity, link, fact and synonym edge is a deterministic function of per-passage extractions that carry chunk-level provenance. Retrieval makes no language-model call, indexing makes one per passage, and deleting a document is a database transaction with no model calls: 14 seconds for 10% of an 11,402-passage corpus, after which the graph is identical to a fresh index of the surviving documents. On a corpus of near-duplicate names, HippoRAG 2's capped synonym lists leave 565 fewer edges than a rebuild, while Lightfoot remains exact. Under identical extraction, embedding and answer models on held-out questions of MuSiQue, 2WikiMultiHopQA an
    
[^278]: 规划器中的古德哈特定律：基于采样的控制会“钻空子”利用从人类偏好中学到的奖励模型

    Goodhart in the Planner: Sampling-Based Control Exploits Rewards Learned from Human Preferences

    [https://arxiv.org/abs/2611.09670](https://arxiv.org/abs/2611.09670)

    论文揭示并量化了“规划器中的古德哈特定律”：针对人类偏好奖励模型进行更强的 MPPI 搜索虽在标签准确率上保持高分，却使真实回报最多下降 76%，并进一步给出真实回报与搜索偏离距离（√KL）的关系以及 χ² 正则化 MPPI 在由评分器误差范围设定温度时的最坏情况最优性保证。

    

    机器人越来越多地通过针对从人类偏好中学习到的奖励模型进行搜索来完成规划；我们测量了这种搜索对真实目标造成的影响。在带有众包标签的 HalfCheetah 和 Hopper 任务上，以及在带有合成标签和两位作者盲测标签的 Meta-World 抽屉打开与盘子滑动任务上，更强的 MPPI 搜索使真实回报最多下降 76%，而留出标签上的准确率仍保持在 0.75–0.97：使用人类标签时，20 次运行中抽屉打开次数为 0 次，而作为规划器起始点的克隆策略则在 10 次中全部成功。该效应在使用学习到的动力学模型以及图像评分器时依然存在。真实回报大致跟随规划器权重偏离其提案样本的距离（$\sqrt{\mathrm{KL}}$），而使用在合成标签上训练的验证器进行 best-of-$N$ 动作选择，其表现甚至会低于单个带噪声的动作。我们证明了当温度由评分器在提案样本上的误差范围设定时，$\chi^2$ 正则化的 MPPI 具有最坏情况最优性。

    arXiv:2611.09670v2 Announce Type: replace-cross  Abstract: Robots increasingly plan by searching against a reward model learned from human preferences; we measure what this search does to the true objective. On HalfCheetah and Hopper with crowd-sourced labels, and on Meta-World drawer-open and plate-slide with synthetic labels and blind labels from the two authors, harder MPPI search cuts true return by up to 76% while held-out label accuracy stays at 0.75-0.97: with human labels the drawer opens in 0 of 20 runs, against 10 of 10 for the cloned policy the planner starts from. The effect persists with a learned dynamics model and with an image scorer. True return roughly follows how far the planner's weights move from its proposals ($\sqrt{\mathrm{KL}}$), and best-of-$N$ action selection with a verifier trained on synthetic labels falls below a single noisy action. We prove that $\chi^2$-regularized MPPI at a temperature set by the scorer's error spread on the proposals is worst-case op
    
[^279]: 跨越双人游戏难度分级阶梯的零样本权重空间编程

    Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player Games

    [https://arxiv.org/abs/2611.09271](https://arxiv.org/abs/2611.09271)

    该论文首次系统考察大语言模型零样本权重空间编程的能力边界，在涵盖四个难度维度的双人游戏四级阶梯上零样本手工设定前馈网络的全部权重，构建并独立验证了十个网络，其中六个获得完整且非统计性的正确性证明。

    

    权重空间编程是指依据对目标计算的显式规范，手工设定前馈网络的每一个权重，整个构造过程中完全不使用梯度下降，也不使用任何训练数据。我们探究大语言模型能否以零样本方式将这一实践推广到难度刻意递增的游戏上。我们设计了一个四级阶梯（尼姆棋、井字棋、两种棋盘尺寸的四子棋以及一个国际象棋残局），涵盖四个难度维度，并在每一级额外构建一个结构上互不相关、难度相当的游戏网络，以区分某一层级的结论究竟反映的是该难度水平本身，还是仅反映用于测量的那一个游戏。共构建了十个网络并进行独立验证，其中六个网络可针对独立预言机给出完整且非统计性的正确性证明。国际象棋残局层级在正确性代价较低的部分（走法合法性，以及其中一个网络中的……）是精确的。（原文摘要在此处被截断）

    arXiv:2611.09271v3 Announce Type: replace  Abstract: Weight-space programming sets every weight of a feed-forward network by hand, from an explicit specification of the desired computation, with no gradient descent and no training data anywhere in the construction. We ask how far a large language model can carry this practice, zero-shot, across games of deliberately increasing difficulty. We design a four-rung ladder (Nim, tic-tac-toe, two sizes of Connect Four, and a chess endgame) spanning four axes of difficulty, and build a second, independent network at each rung, using a structurally unrelated game at matched difficulty, to separate what a rung's finding says about the difficulty level from what it says about the one game used to measure it. Ten networks are built and independently verified. Six admit a complete, non-statistical proof of correctness against an independent oracle. The chess-endgame rung is exact where correctness is cheap, move legality and, in one network, an exa
    
[^280]: 基于词元身份平方监督的连续语言流继续训练方法

    Continued Training of Continuous Language Flows with Token Identity Squared Supervision

    [https://arxiv.org/abs/2611.07721](https://arxiv.org/abs/2611.07721)

    提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。

    

    扩散模型和流模型最近被扩展到语言建模领域，通过在词元嵌入空间中进行连续去噪来生成文本。预训练模型提供了词元表示和上下文预测能力，但要在有限额外训练的情况下改进生成效果，需要在监督信号与连续生成之间建立适当的连接。连续MSE评估的是通过用预测概率对词元嵌入加权所得到的预测向量；不同的概率分布可以产生相同的向量，因此在该目标函数下无法被区分。我们提出词元身份Brier流（TIBF），在继续训练预训练语言流模型时，直接对词元身份应用平方监督。TIBF在高斯观测模型下将网络得分与词元匹配得分相结合，使用Brier损失对词元概率进行监督，并使用概率加权嵌入进行……

    arXiv:2611.07721v1 Announce Type: cross  Abstract: Diffusion and flow models have recently been extended to language modeling, generating text through continuous denoising in token embedding spaces. Pretrained models provide token representations and contextual prediction capabilities, but improving generation with limited additional training requires an appropriate connection between supervision and continuous generation. Continuous MSE evaluates a predicted vector obtained by weighting token embeddings with predicted probabilities; different probability distributions can yield the same vector and are therefore indistinguishable under this objective. We propose Token Identity Brier Flow (TIBF), which applies squared supervision directly to token identities when continuing pretrained language flows. TIBF combines network scores with token-matching scores under a Gaussian observation model, supervises token probabilities using the Brier loss, and uses probability-weighted embeddings for
    
[^281]: 签名但并非安全：多智能体LLM交接中的需求完整性

    Signed but Not Safe: Requirement Integrity in Multi-Agent LLM Handoffs

    [https://arxiv.org/abs/2611.07483](https://arxiv.org/abs/2611.07483)

    该论文提出一种带HMAC-SHA256签名的需求清单交接机制，配合确定性代码验证，将多智能体LLM系统在攻击下的不安全提交从69.4%降至9.0%，且不损失合规提交。

    

    企业级AI智能体越来越多地将用户的请求拆分给多个子智能体并合并其结果，而每一次交接都可能概括掉、篡改或忽略某项需求。我们报告了一项预注册研究，使用合成的、可精确评分的任务，比较了自由文本、结构化JSON和带签名的需求清单三种交接方式：编排器将每项需求提取为逐字引用加上可核查的谓词，代码使用HMAC-SHA256对清单进行签名，并由确定性代码验证合并器的决策。使用Qwen3-8B（1,440条轨迹，五种威胁），带签名的清单将攻击下的不安全提交从69.4%降至9.0%（降低60.4个百分点，95%置信区间[51.4, 69.4]），且与结构化交接相比，合规提交没有净损失。签名正是击败传输中篡改的关键：未签名的清单在79%的篡改运行中提交了违规内容，而签名的清单则一次也没有。在相同的轨迹上，确定性验证器捕获了78.7%的违规决策。

    arXiv:2611.07483v1 Announce Type: cross  Abstract: Enterprise AI agents increasingly split a user's request across subagents and merge their results, and each handoff can summarize away, alter or ignore a requirement. We report a preregistered study with synthetic, exactly scorable tasks comparing free-text, structured JSON and signed requirement-manifest handoffs: the orchestrator extracts each requirement as a verbatim quote plus a checkable predicate, code signs the manifest with HMAC-SHA256, and deterministic code verifies the merger's decision. With Qwen3-8B (1,440 trajectories, five threats), signed manifests cut unsafe commits under attack from 69.4% to 9.0% (60.4 pp, 95% CI [51.4, 69.4]) with no net loss of conformant commits relative to structured handoffs. Signing is what defeats in-transit tampering: an unsigned manifest committed violations in 79% of mutated runs, the signed one in none. On identical trajectories the deterministic verifier caught 78.7% of violating decision
    
[^282]: EmoReward：面向情感视频生成强化学习的自进化奖励校准

    EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL

    [https://arxiv.org/abs/2611.06518](https://arxiv.org/abs/2611.06518)

    EmoReward 提出一种自进化奖励校准机制，通过从训练过程的奖励轨迹中提炼证据可靠性来自适应调整多奖励权重分配，解决了固定奖励难以捕捉情感意图且易引发奖励作弊的问题。

    

    组相对策略优化（GRPO）已在代码和数学等可验证领域推动了生成模型的后训练，但视频质量（尤其是情感表达）的评估往往依赖于主观美学，缺乏唯一正确的目标。现有方法通常依赖针对视觉质量、运动质量和文本对齐的固定奖励，这些奖励难以捕捉情感意图，并可能引发奖励作弊；而基于人类反馈的强化学习（RLHF）难以扩展，且受个体审美差异影响，使得在不同叙事情境下校准面部、肢体动作和色彩线索变得复杂。我们提出了 EmoReward，一种全新的自进化训练机制，它从实际训练过程中收集的奖励轨迹中提炼证据可靠性，通过判定哪些情感证据可信以及应赋予多大权重，自适应地引导后续训练中的多奖励权重分配。

    arXiv:2611.06518v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) has advanced generative model post-training in verifiable domains such as code and mathematics, but evaluating video quality, especially emotional expression, often depends on subjective aesthetics and has no uniquely correct target. Existing approaches typically rely on fixed rewards for visual quality, motion quality, and text alignment that poorly capture emotional intent and can invite reward hacking, while RLHF is difficult to scale and subject to individual aesthetic differences that complicate the calibration of facial, body motion, and chromatic cues across narrative contexts. We introduce EmoReward, a new self-evolving training mechanism that distills evidence reliability from reward traces collected during an actual training run to adaptively guide multi-reward weight allocation in subsequent training by determining which emotional evidence to trust and how much to weight it. EmoRewar
    
[^283]: 以草根契约圈控人工智能风险

    Corralling AI Risks with Grassroots Contracts

    [https://arxiv.org/abs/2611.06374](https://arxiv.org/abs/2611.06374)

    提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。

    

    人工智能的转折点带来了大量令人担忧的风险。数字平台尤其容易受到三大AI风险的影响，因为全球平台的控制者正在将控制权让渡给AI——包括算法控制权、成员控制权和内容控制权：(1) AI控制被人类大多数人所使用的全球社会与经济平台，导致大多数人类容易受到AI的操控。(2) 无法问责的AI智能体在社交与经济数字平台上与人类并肩匿名运行，可能伪装成人类，为身份不明的主人服务，追求未知的目标，在未知的约束下运行，缺乏问责机制和责任追究。(3) 由匿名AI生成、无法归因的深度伪造内容，人们无法将其与真实内容区分开来，其制作者则逃避问责和追责。在此，我们提出用草根契约来圈控这些AI风险，草根契约是一种……（原文摘要在此处截断）

    arXiv:2611.06374v1 Announce Type: new  Abstract: AI's inflection point is a fountain of alarming risks. Digital platforms in particular are vulnerable to three major AI risks, as those in control of global platforms cede control to AI, of the algorithms, of membership and of content: (1) AI control of global platforms, social and economic, used by the majority of humanity, resulting in the majority of humanity becoming vulnerable to AI control. (2) Unaccountable AI agents operating anonymously on social and economic digital platforms, side-by-side with humans, possibly masquerading as humans, serving unknown masters with unknown goals, operating under unknown restraints without accountability or culpability. (3) Unattributed deepfakes generated by anonymous AI, with people unable to distinguish them from genuine content and with their producers evading accountability and culpability.   Here, we propose to corral these AI risks with grassroots contracts, a digital incarnation of a two-c
    
[^284]: 技能侵蚀、个体记忆与AI采用的可逆性

    Skill erosion, individual memory and the reversibility of AI adoption

    [https://arxiv.org/abs/2611.04886](https://arxiv.org/abs/2611.04886)

    该研究揭示了技能侵蚀机制使AI采用呈现双稳态，即使群体具有相同的采用率和平均技能水平，也可能因技能在AI使用者与非使用者之间的分布差异而走向截然相反的长期结局。

    

    生成式AI可以提高即时表现，但可能会侵蚀未来工作所需的人类技能。我们考虑一个由智能体组成的群体，它们要么凭借自身技能执行任务，要么借助AI完成任务。智能体倾向于模仿更成功的同伴，其技能随时间演化：随练习而提升，随对AI的依赖而退化。我们通过解析分析和模拟研究了AI采用的集体后果。我们表明，技能侵蚀机制使AI采用呈现双稳态，且群体的长期行为取决于技能的分布方式：具有相同采用率和平均技能水平的群体可能达到截然相反的渐近状态，这取决于专业知识在AI使用者与非使用者之间如何分配。一个追踪两组平均技能以及非使用者群体中技能方差的确定性描述，能够达到与模拟相同的精度来预测个体运行的结果。

    arXiv:2611.04886v2 Announce Type: replace  Abstract: Generative AI can improve immediate performance but may erode the human skills needed for future work. We consider a population of agents who perform a task either with their own skill or with the help of AI. Agents tend to imitate more successful peers, and their skills evolve in time: improving with practice and eroding with reliance on AI. We study, analytically and with simulations, the collective consequences of AI adoption. We show that the mechanism of skill erosion makes adoption bistable and that   the long-term behaviour of the population depends on how skills are distributed: populations with the same adoption and mean skill can reach opposite asymptotic states, depending on how expertise is shared between AI users and non-users. A deterministic description that follows the mean skills of the two groups and the variance of skills among non-users predicts the outcome of individual runs as accurately as the simulations allow
    
[^285]: 基于检索增强生成的股票市场信息检索优化

    Enhancing Stock Market Information Retrieval with Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.03747](https://arxiv.org/abs/2611.03747)

    本文提出一种整合检索与生成方法的RAG系统，用于生成准确且相关的股票市场信息响应，并通过ROUGE指标验证了其有效性。

    

    自然语言处理（NLP）已广泛应用于金融领域，实现了股票市场信息的自动化分析与检索。然而，传统的基于NLP的系统在动态金融环境中往往难以保持上下文准确性并生成相关响应。为应对这些挑战，检索增强生成（RAG）模型作为一种强大的解决方案应运而生，它将基于检索的方法与生成能力相结合，以提升响应质量。在本研究中，我们提出了一种旨在生成与股票市场信息相关响应的检索增强生成（RAG）系统。该系统整合了基于检索的方法和生成式方法，以提高金融文本生成的准确性和相关性。为评估其有效性，我们采用ROUGE指标（具体包括ROUGE-1、ROUGE-2和ROUGE-L）来衡量词汇和结构层面的相似性……

    arXiv:2611.03747v2 Announce Type: replace-cross  Abstract: Natural Language Processing (NLP) has been widely applied in financial domains, enabling automated analysis and retrieval of stock market information. However, traditional NLP-based systems often struggle with maintaining contextual accuracy and generating relevant responses in dynamic financial environments. To address these challenges, Retrieval-Augmented Generation (RAG) models have emerged as a powerful solution, combining retrieval-based methods with generative capabilities to enhance response quality. In this study, we present a Retrieval-Augmented Generation (RAG) system designed to generate responses related to stock market information. The system integrates retrieval-based and generative approaches to enhance the accuracy and relevance of financial text generation. To evaluate its effectiveness, we employ the ROUGE metric, specifically ROUGE-1, ROUGE-2, and ROUGE-L, to measure lexical and structural similarity between 
    
[^286]: 野外环境中的恶意软件聚类：分类法、挑战与评估指标

    Malware Clustering in the Wild: Taxonomy, Challenges, and Evaluation Metrics

    [https://arxiv.org/abs/2611.02380](https://arxiv.org/abs/2611.02380)

    本综述系统地将恶意软件聚类研究归纳为七大类方法，并总结了常用的公开数据集与评估指标，为该领域提供了统一的分类框架并明确了研究空白。

    

    随着恶意软件在数量、多样性和复杂性方面的持续增长，聚类已成为分析大规模恶意软件语料库的关键技术。聚类能够发现结构关系、行为模式和演化趋势，并广泛用于对未知样本进行分组、推断恶意软件家族边界以及识别新型变种。然而，现有研究仍然较为零散，采用了各种各样的聚类技术，缺乏统一的视角来系统地组织先前的研究工作或明确指出文献中的空白。本综述通过将以往的恶意软件聚类研究归纳为七大类别来填补这一空白：基于质心的方法、基于层次的方法、基于密度的方法、基于分布的方法、混合类型方法、模糊方法和AI驱动的方法。此外，本综述还总结了聚类研究中常用的公开恶意软件数据集，并考察了特征表示方法。

    arXiv:2611.02380v2 Announce Type: replace-cross  Abstract: As malware continues to grow in volume, diversity, and sophistication, clustering has emerged as a critical technique for analyzing large malware corpora. Clustering enables the discovery of structural relationships, behavioral patterns, and evolutionary trends, and is widely used to group unknown samples, infer malware family boundaries, and identify novel variants. However, existing research remains fragmented, employing a wide range of clustering techniques with no unified perspective that systematically organizes prior work or clearly identifies gaps in the literature. This survey addresses this gap by organizing previous malware clustering research into seven categories: centroid-based, hierarchical-based, density-based, distribution-based, mixed- type, fuzzy, and AI-driven approaches. In addition, this survey summarizes publicly available malware datasets commonly used in clustering research and examines the feature repre
    
[^287]: 看、想、用：诊断视觉语言模型中的感知与推理失败

    See, Think, Use: Diagnosing Perception and Reasoning Failures in VLMs

    [https://arxiv.org/abs/2611.01467](https://arxiv.org/abs/2611.01467)

    该论文提出“看、想、用”协议，通过定位探针、文本替换探针和反事实图像对，在无需访问模型权重的情况下，将视觉语言模型的每个答案归因于感知、推理或感知使用，从而精准定位应改进的环节并指导测试时计算的分配。

    

    我们提出了“看、想、用”，这是一种能够将视觉语言模型的每个答案追溯到感知、推理或对感知的使用三个阶段的诊断协议。该诊断可以告诉开发者应当改进哪个阶段，并将测试时计算引导到失败的环节。基准测试只对最终答案评分，基于图像得出的答案与猜测或记忆的先验获得同样的分数。该协议在每个题目上只需向模型额外提出两个问题，且无需访问模型权重。定位探针要求模型为每个实例框出边界框，以测量“看”的能力；文本替换探针让模型基于文字描述来回答问题，以测量“想”的能力。“用”衡量的是在反事实图像对上，答案遵循模型自身感知的比率。该比率无需答案标签，因为问题和其先验在两张图像上是相同的。在两个模型家族和六个数据集上，该协议表明有效的改进手段取决于任务类型。

    arXiv:2611.01467v1 Announce Type: cross  Abstract: We present See, Think, Use, a protocol that traces each answer of a vision-language model to perception, reasoning or the use of perception. The diagnosis shows a developer which stage to improve and directs test-time compute to the stage that fails. Benchmarks score the final answer, and an answer that rests on the image earns the same credit as a guess or a memorized prior. The protocol asks the model two extra questions per item and needs no access to its weights. A localization probe asks for a box around every instance and measures seeing, and a text-replacement probe answers from a description and measures thinking. Use is the rate at which the answer follows the model's own perception across counterfactual image pairs. The rate needs no answer labels, because the question and its prior are the same on both images. On two model families and six datasets the protocol shows that the knob that helps follows the task family. Answers 
    
[^288]: 多重估计引导并不能统一地消除语言模型判断中的偏差

    Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments

    [https://arxiv.org/abs/2611.00597](https://arxiv.org/abs/2611.00597)

    本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。

    

    在单次回复中引导模型给出五个加权估计值，是否会改变语言模型对信息的响应方式？在锚定、后见之明、基础比率、合取以及现状/默认等任务中，我们将直接数值回答与多重估计响应格式进行对比，同时比较了不同的来源框架、显式随机线索以及概率一致性约束。我们评估了三种已完成配置——gpt-6.1-sol、claude-sonnet-5.5 以及 Qwen3-8B（禁用推理，进行描述性分析）。在前两种配置中，直接引导产生的无关锚定指数和现状效应接近于零，同时模型仍能对信息性来源和知情建议保持敏感。在23项预注册的验证性测试中，有六项在经过 Holm 校正后被拒绝；然而，这六项测试的绝对点估计值均低于预先设定的最小感兴趣效应量，但并未据此建立等效性结论。此外，虽然分离引导与联合引导的（摘要在此处截断）

    arXiv:2611.00597v1 Announce Type: new  Abstract: Does eliciting five weighted estimates within a single response change how a language model responds to information? Across anchoring, hindsight, base-rate, conjunction, and status-quo/default tasks, we compare direct numerical answers with a multi-estimate response format, contrasting source framings, explicitly random cues, and probability-consistency constraints. We evaluate three completed configurations--gpt-6.1-sol, claude-sonnet-5.5, and Qwen3-8B (reasoning disabled, analyzed descriptively). In the first two configurations, direct elicitation yielded near-zero irrelevant-anchor indices and status-quo effects, while remaining responsive to informative sources and informed advice. Of 23 preregistered confirmatory tests, six rejected after Holm adjustment; however, all six absolute point estimates fell below their prespecified smallest effect sizes of interest without establishing equivalence. Moreover, while separate versus joint el
    
[^289]: WorldFlow：通过大规模预训练实现视频与3D运动的联合预测

    WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining

    [https://arxiv.org/abs/2610.39278](https://arxiv.org/abs/2610.39278)

    WorldFlow是一种生成式世界模型，它将3D运动参数化为与视频网格对齐的连续位移标记，并利用统一的Mixture-of-Transformers骨干网络在整流流目标下联合去噪视频与3D两种模态，从而将显式几何动力学与视觉演化耦合，释放预训练视频模型中蕴含的物理常识。

    

    预测未来的物理状态是具身智能中的一个核心挑战。现有的视频世界模型将3D动力学隐式化，基于动作的公式化仍然绑定于特定的机器人配置，而3D运动预测模型仅使用光流监督，为物理动力学提供的学习信号十分有限。为了规避这些限制，我们提出了WorldFlow，一种能够联合预测未来视频和基于查询的3D轨迹的生成式世界模型。通过将3D运动参数化为与视频网格对齐的连续位移标记，一个统一的混合Transformer（Mixture-of-Transformers, MoT）骨干网络在整流流（rectified-flow）目标下对两种模态进行联合去噪。这种公式化将显式的几何动力学与视觉演化耦合在一起，有效地释放了预训练视频模型中蕴含的物理常识。为了在实证上扩展并验证这一统一架构，我们在异构（摘要在此处截断）……上训练WorldFlow。

    arXiv:2610.39278v1 Announce Type: cross  Abstract: Predicting future physical states stands as a central challenge in embodied intelligence. Existing video world models leave 3D dynamics implicit, action-based formulations remain tied to specific robot configurations, and 3D motion prediction models use flow-only supervision, providing a limited learning signal for physical dynamics. To circumvent these limitations, we introduce \method{}, a generative world model that jointly predicts future video and query-based 3D trajectories. By parameterizing 3D motion as continuous displacement tokens aligned with the video grid, a unified Mixture-of-Transformers (MoT) backbone jointly denoises both modalities under a rectified-flow objective. This formulation couples explicit geometric dynamics with visual evolution, effectively unlocking the physical commonsense embedded within pretrained video models. To empirically scale and validate this unified architecture, we train \method{} on a heterog
    
[^290]: Cassette-Editor：一个像你一样进行编辑的视频编辑智能体

    Cassette-Editor: A Video Editing Agent That Edits the Way You Do

    [https://arxiv.org/abs/2610.37690](https://arxiv.org/abs/2610.37690)

    提出了Cassette-Editor，一个通过“一读一写”极简接口让AI像人类一样直接编辑视频项目文件的智能体，所有AI编辑都经过与人工手动编辑相同的执行、规则和提交流程，从而以原子性提交和精准诊断保证时间线始终有效。

    

    语言智能体越来越多地通过在持久化工件（如代码仓库）上调用工具来工作，每次调用的结果都会决定它们的下一步行动。视频编辑是这种模式的一个高难度应用场景：一帧画面呈现的内容取决于每个片段如何被修剪、重新定时、裁剪和嵌套，因此关于素材的知识只对特定用途有效，而一次无效的编辑可能导致时间线在结构上崩溃。现有的视频编辑智能体通常借助批评智能体、审查机制或受限的输出格式来保持编辑的有效性。我们提出了 Cassette-Editor，一个采用刻意简洁接口的视频编辑智能体：仅用一个工具读取项目、一个工具写入项目。每一次写入操作都与用户的手动编辑一样，经过相同的执行器、规则和提交服务，要么原子性地提交，要么被拒绝并返回定位明确的诊断信息和修复选项，而模型则保留创意决策权并自行决定如何恢复。读取操作……

    arXiv:2610.37690v1 Announce Type: cross  Abstract: Language agents increasingly work by calling tools on a persistent artifact, such as a code repository, and the result of each call shapes their next step. Video editing is a demanding case of this pattern: what a frame shows depends on how every clip is trimmed, retimed, cropped, and nested, so knowledge about the footage holds only for particular uses of it, and an invalid edit can leave the timeline structurally broken. Existing video editing agents keep their edits valid with critic agents, reviewers, or restricted output formats. We present Cassette-Editor, a video editing agent with a deliberately simple interface: one tool reads the project and one tool writes to it. Every write passes through the same executor, rules, and commit service as the user's manual edits, and either commits atomically or is rejected with located diagnostics and repair options, while the model keeps the creative choices and decides how to recover. Reads
    
[^291]: REGER：基于语义ID建模物品间关系的生成式推荐

    REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation

    [https://arxiv.org/abs/2610.37584](https://arxiv.org/abs/2610.37584)

    提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。

    

    个性化推荐依赖于从在线内容和用户交互中学习物品之间的语义关系和协同关系。生成式推荐（GR）通过自回归地生成物品的离散语义ID（SIDs），提供了一种有前景的范式。然而，随机初始化的SID词元嵌入以及从零开始训练的GR模型并不能直接继承物品固有的语义关系，而物品间的协同关系主要是通过词元级注意力机制和下一词元预测监督被隐式学习的。为了解决这一局限，我们提出了REGER，这是一个统一框架，将物品间关系显式地融入语义预训练、历史编码和SID生成之中。我们通过生成式语义检索对GR模型进行预训练，使其通过学习生成语义邻居的SID来继承物品间的语义相似性。我们将全局注意力与局部注意力相结合……（摘要原文在此处不完整）

    arXiv:2610.37584v1 Announce Type: cross  Abstract: Personalized recommendation depends on learning semantic and collaborative relations among items from online content and user interactions. Generative recommendation (GR) offers a promising paradigm by autoregressively generating discrete semantic IDs (SIDs) of items. However, randomly initialized SID token embeddings and GR models trained from scratch do not directly inherit the semantic relations inherent in items, while inter-item collaborative relations are learned largely implicitly through token-level attention and next-token supervision. To address this limitation, we propose REGER, a unified framework that explicitly incorporates inter-item relations into semantic pretraining, history encoding, and SID generation. We pretrain the GR model through generative semantic retrieval, enabling it to inherit inter-item semantic similarity by learning to generate semantic neighbors' SIDs. We combine global attention with local attention 
    
[^292]: MatOS：面向材料研究的智能体操作系统

    MatOS: An Agent Operating System for Materials Research

    [https://arxiv.org/abs/2610.34203](https://arxiv.org/abs/2610.34203)

    提出了MatOS——一个面向材料研究的智能体操作系统，通过分层科学资产图、带科学检查的有状态自适应工作流和证据驱动的资产演化三大核心组件，解决了材料研究中智能体能力扩展困难、长时任务难以追踪恢复以及无法从历史任务和用户反馈中学习的难题。

    

    材料研究依赖于分散且不断增长的数据、知识和模拟工具，一项单一研究往往需要跨越多天的工作流程，涵盖文献分析、计算和实验。大语言模型智能体现在能够检索材料数据、运行模拟代码，并借助自主实验室闭合设计-合成-表征循环。然而，为这些智能体扩展新能力、使长时间运行的任务保持正轨，以及让它们从过去的任务和用户反馈中学习，仍然十分困难。在此，我们提出了MatOS——一个面向材料研究的智能体操作系统，其包含三个核心组件：（1）分层科学资产图，将数据、知识、工具、技能、角色和工作流组织为可复用的资产；（2）有状态的自适应工作流，将任务作为带科学检查的显式执行图运行，从而支持任务的恢复与追踪；（3）证据驱动的资产演化，基于……（原文摘要在此处截断）

    arXiv:2610.34203v1 Announce Type: new  Abstract: Materials research relies on scattered and growing data, knowledge, and simulation tools, and a single study often spans multi-day workflows across literature analysis, computation, and experiment. Large language model agents can now retrieve materials data, operate simulation codes, and close design-synthesis-characterization loops with autonomous laboratories. However, extending these agents with new capabilities, keeping long-running tasks on track, and letting them learn from past tasks and user feedback remain difficult. Here we present MatOS, an agent operating system for materials research with three core components: (1) a layered scientific asset graph that organizes data, knowledge, tools, skills, roles, and workflows as reusable assets; (2) a stateful adaptive workflow that runs tasks as explicit execution graphs with scientific checks, so that they can be resumed and traced; and (3) evidence-driven asset evolution, which revis
    
[^293]: RIPPLE：面向长程智能体编程的过程级策略学习与效果感知信用分配的恢复内化

    RIPPLE: Recovery Internalization through Process-Level Policy Learning and Effect-Aware Credit for Long-Horizon Agentic Coding

    [https://arxiv.org/abs/2610.31276](https://arxiv.org/abs/2610.31276)

    提出RIPPLE训练框架，通过前缀教师蒸馏与阶段特定的强化学习信用分配，将过程级恢复能力内化到单一自恢复策略中，弥合编程智能体“能批评却无法行动”的交接鸿沟，并在仓库级软件工程基准上平均提升4.8分。

    

    长程编程智能体能够观察到失败并生成看似合理的批评，却仍沿着同一条错误轨迹继续运行。我们将这一现象称为“批评到行动交接鸿沟”。我们提出RIPPLE，一个面向恢复的训练框架，将过程级指导内化到单一的自恢复策略中。一个仅使用前缀的教师模型在不访问未来动作或最终结果的情况下提供基于证据的恢复监督。在轻量级蒸馏之后，强化学习对反思触发、诊断和纠正行动分配阶段特定的信用。局部有效性信号对轨迹优势进行门控，使中间恢复决策不会不加区分地继承最终结果。在四个仓库级软件工程基准上，RIPPLE将基础策略平均提升4.8个分数点，且部署时无需辅助评论家。受控消融实验分离出各组成部分的收益。

    arXiv:2610.31276v2 Announce Type: replace  Abstract: Long-horizon coding agents can observe a failure and generate a plausible critique yet continue along the same incorrect trajectory. We call this the Critique-to-Action Handoff Gap. We introduce RIPPLE, a recovery-oriented training framework that internalizes process-level guidance into a single self-recovering policy. A prefix-only teacher provides evidence-grounded recovery supervision without access to future actions or terminal outcomes. After lightweight distillation, reinforcement learning assigns stage-specific credit to reflection triggering, diagnosis, and corrective action. Local validity signals gate trajectory advantages so that intermediate recovery decisions do not indiscriminately inherit terminal outcomes. Across four repository-level software-engineering benchmarks, RIPPLE improves the base policy by an arithmetic mean of 4.8 score points, with no auxiliary critic at deployment. Controlled ablations isolate the benef
    
[^294]: 基于激活空间引导的可验证奖励强化学习探索方法

    Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2610.27012](https://arxiv.org/abs/2610.27012)

    提出激活空间探索（ASE）方法，通过聚类正确与错误响应的激活差异构建潜在探索向量库，并将其应用于训练采样以引导模型探索多样化推理路径，从而提升可验证奖励强化学习在数学推理上的性能。

    

    可验证奖励强化学习（RLVR）通过采样模型的响应、使用自动验证器对其评分，并根据所得奖励更新策略来训练语言模型。因此，策略只能从其自身采样生成的轨迹中获得直接学习信号。然而，从同一策略反复采样可能会集中在相似的推理路径上，在固定的采样预算下导致有用的替代路径未被探索。我们提出了激活空间探索（ASE），这是一种用于多样化训练采样的在线方法。通过聚类同一问题正确与错误响应之间的激活差异，ASE 构建了一个结构化的潜在探索向量库。随后，不同的向量被应用于后续的采样过程中，以探索替代的推理路径。在 DAPO-Math-17k 上训练 Qwen3-1.7B-Base 和 Qwen3-8B-Base 时，ASE 在六个数学推理基准上提高了平均 Mean@8 性能。

    arXiv:2610.27012v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains a language model by sampling responses, scoring them with an automatic verifier, and updating the policy from the resulting rewards. The policy therefore receives direct learning signals only from trajectories that appear in its rollouts. Yet repeated sampling from the same policy can concentrate on similar reasoning paths, leaving useful alternatives unexplored under a fixed rollout budget. We introduce Activation-Space Exploration (ASE), an online method for diversifying training rollouts. By clustering activation differences between correct and incorrect responses to the same problem, ASE builds a structured bank of latent exploration vectors. Different vectors are then applied to subsequent rollouts to explore alternative reasoning paths. On Qwen3-1.7B-Base and Qwen3-8B-Base trained on DAPO-Math-17k, ASE improves average Mean@8 across six mathematical reasoning benchmarks
    
[^295]: 相同的模拟，不同的证据：检索式工具输出在何处损害LLM智能体的准确性

    Same Simulation, Different Evidence: Where Retrieved Tool Output Costs LLM Agents Accuracy

    [https://arxiv.org/abs/2610.26836](https://arxiv.org/abs/2610.26836)

    本文利用确定性的PHREEQC地球化学模拟器实验发现，当长工具输出被移出上下文、改为按需检索时，GPT-5.1和GPT-5.2的准确率分别下降11.3和7.9点而Claude模型不受影响，其根本原因并非计算出错，而是GPT模型在检索模式下跳过模拟报告、不再以模拟证据作为回答依据。

    

    即使计算出了正确答案，LLM智能体仍可能给出错误的回答。智能体框架越来越多地将长工具输出排除在上下文窗口之外，转而保存到文件中供智能体按需搜索或读取。我们借助一个真实的科学工具来探究这种做法在何处损害准确性：PHREEQC，这是美国地质调查局（USGS）广泛使用的地球化学模拟器，其报告可长达84.68万字符，且其确定性使每一个打印出的数值都可被追溯。智能体使用该工具回答158道经执行验证的选择题。当智能体通过章节索引而非上下文阅读模拟器报告时，GPT-5.1和GPT-5.2分别损失11.3和7.9个百分点的准确率，而Claude Opus 4.6和Claude Sonnet 4.6则未表现出可检测的代价。重新运行全部11,007次记录在案的模拟器调用表明，两个GPT模型计算出标准答案值的频率与之前大致相同；发生变化的是它们回答时所依据的证据。在检索模式下，它们常常跳过报告，转而依据自己的……（原文摘要在此处截断）

    arXiv:2610.26836v2 Announce Type: replace  Abstract: LLM agents that compute the right answer can still answer wrongly. Agent harnesses increasingly keep long tool outputs out of the context window, saving them to files that the agent searches or reads on demand. We ask where this costs accuracy with a real scientific tool: PHREEQC, a widely used geochemistry simulator from the USGS, whose reports reach 846.8k characters and whose determinism lets every printed value be traced. Agents use it to answer 158 execution-verified multiple-choice items. When agents read the simulator's report through a section index rather than in context, GPT-5.1 and GPT-5.2 lose 11.3 and 7.9 points of accuracy, while Claude Opus 4.6 and Claude Sonnet 4.6 show no detectable cost. Re-running all 11,007 logged simulator calls shows that both GPT models compute the keyed value about as often as before; what changes is the evidence they answer from. Under retrieval they often skip the report, answer from their o
    
[^296]: ABACUS：基于程序引导的冻结判断模型多步推理

    ABACUS: Program-Guided Multistep Reasoning over Frozen Judgment Models

    [https://arxiv.org/abs/2610.25963](https://arxiv.org/abs/2610.25963)

    ABACUS框架通过生成式教师离线构建判断模式与组合程序，将冻结判断模型的局部语义评分决策组合成强大的多步推理，无需微调模型即可在BIG-Bench Hard等任务上取得优异表现。

    

    判断模型通过对有限候选结果进行评分来回答语义问题。可复用的程序能否将这些局部决策转化为强大的多步推理？我们提出了ABACUS，一个为冻结判断模型联合构建判断模式与组合程序的框架。生成式教师离线设计程序流程；判断模型在线解决局部语义问题，而保存下来的程序将其决策组合成最终答案。我们研究了人类与教师之间设计工作的三种分配方式，重点关注通过共享接口进行程序构建。我们使用Kev对该框架进行了实例化。在三个选定的BIG-Bench Hard任务上，九个独立构建的程序解决了150个开发输入的全部450次执行。在另一项独立的部署研究中，三个程序在经过开发集参考一致性优化后被冻结，并在600个保留问题上进行评估。这些程序达到了……（原文摘要在此处截断）

    arXiv:2610.25963v2 Announce Type: replace  Abstract: Judgment models answer semantic questions by scoring a finite set of candidate outcomes. Can reusable programs turn these local decisions into powerful multistep reasoning? We introduce \method, a framework that jointly constructs judgment schemas and composition programs for frozen judgment models. A generative teacher designs the procedure offline; the judgment model resolves local semantic questions online, and the saved program composes its decisions into an answer. We study three allocations of design effort between humans and the teacher, focusing on construction through a shared interface. We instantiate the framework with Kev. On three selected BIG-Bench Hard tasks, nine independently constructed programs solve all 450 executions of 150 development inputs. A separate deployment study freezes three programs after development-set reference-agreement refinement and evaluates them on 600 held-out questions. These programs achieve
    
[^297]: 世界模型、树模型与基础模型在零售预测中的应用：来自四个公开数据集的证据

    World Models, Trees and Foundation Models for Retail Forecasting: Evidence from Four Public Data Sets

    [https://arxiv.org/abs/2610.25170](https://arxiv.org/abs/2610.25170)

    本文提出了一种在价格与促销计划下推进商品状态的世界模型，并在四个公开零售数据集上针对从单品单店订货到全链财务规划的不同决策粒度，系统比较了它与LightGBM、Transformer及TimesFM等时序基础模型的预测表现，借助配对检验与模型置信集确定各粒度下的最优模型。

    

    零售商在产品、门店与时间层级的不同粒度上做出不同的决策：单品在单店的门店订货、单品跨门店的供应商资金投入、按门店衡量的品类绩效，以及整个连锁企业的财务规划。使用哪种预测模型取决于决策粒度。本文在四个公开数据集（M5、Dominick's 的六个品类、Breakfast at the Frat，以及 Corporación Favorita——后者在全部开发过程中被保留不参与）上，通过滚动预测起点、配对检验与模型置信集，对各粒度下的模型进行了比较。研究通过这些数据记录的计划输入来涵盖价格与促销因素，而不涉及产品、商品组合或渠道。所比较的模型包括：一个在价格与促销计划下推进每个商品状态的世界模型、去除该状态的同一网络、LightGBM、Temporal Fusion Transformer、TimesFM、Chronos-Bolt、Timestral、季节性朴素预测，以及各模型预测的等权平均……

    arXiv:2610.25170v2 Announce Type: replace-cross  Abstract: A retailer makes different decisions at different grains of the product, store and time hierarchy: store orders for an item in a store, supplier funding for an item across stores, category performance by store, and the financial plan for the chain. Which forecasting model to use depends on the grain. This paper compares models at each grain on four public data sets (M5, six Dominick's categories, Breakfast at the Frat, and Corporaci\'on Favorita, the last kept out of all development), over rolling forecast origins with paired tests and model confidence sets. It covers price and promotion through the plan inputs these data record, not product, assortment or place. The models are a world model that carries each item's state forward under the price and promotion plan, the same network without that state, LightGBM, the Temporal Fusion Transformer, TimesFM, Chronos-Bolt, Timestral, seasonal naive, and an equal-weight average of the 
    
[^298]: 无需真实标签筛选大语言模型评审：AI口腔卫生辅导系统的初步评估

    Screening LLM Judges Without Ground Truth: A Preliminary Evaluation of an AI Dental Hygiene Coach

    [https://arxiv.org/abs/2610.24335](https://arxiv.org/abs/2610.24335)

    该论文提出了一种无需真实标签即可在专家评审前分阶段筛选大语言模型评审器的流水线，并通过AI口腔卫生辅导系统案例验证了本地评审器与托管评审器具有几乎一致的表现。

    

    使用专家评分员评估微调后的辅导模型既缓慢又昂贵，而在缺乏真实标签的情况下，大语言模型评审器（LLM judge）的可信度难以保证。我们提出了一种分阶段流水线，在专家评审之前筛选候选的大语言模型评审器和学生模型，每个阶段均采用明确的决策规则：评审输出审计、与托管评审器的一致性比较、偏见检查以及决策稳健性分析。我们在一个AI口腔卫生辅导系统上对该流水线进行了演示。该案例研究使用合成的、未经临床验证的训练与参考数据来展示流水线的运行过程，而非用于确立所得模型的临床性能。两个微调的小型语言模型（Granite-4.1-8B和Qwen3-8B），各自在三种数据规模下训练，由五位评审器（三个本地、两个托管）按七项指标进行评分。两个本地评审器Qwen3.8-27B和Gemma-4-31B-it在七项指标中的六项上，其一致性水平与托管评审器之间的一致性无法区分。

    arXiv:2610.24335v2 Announce Type: replace  Abstract: Evaluating a fine-tuned coaching model with expert graders is slow and expensive, and an LLM judge cannot be trusted without ground truth. We present a staged pipeline that screens candidate LLM judges and student models before expert review, using explicit decision rules at each stage: judge output audit, similarity with hosted judges, bias checks, and decision robustness. We demonstrate it on an AI dental hygiene coach. The case study uses synthetic, clinically unvalidated training and reference data to demonstrate the operation of the pipeline, rather than to establish the clinical performance of the resulting models. Two fine-tuned small language models (Granite-4.1-8B and Qwen3-8B), each trained at three data sizes, were scored on seven metrics by five judges (three local, two hosted). Two local judges, Qwen3.8-27B and Gemma-4-31B-it, were not distinguishable from the agreement between the hosted judges on six of seven metrics b
    
[^299]: 变换器的条件无毛定理：语言模型如何遗忘其提示词

    A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt

    [https://arxiv.org/abs/2610.24133](https://arxiv.org/abs/2610.24133)

    本文受黑洞无毛定理启发，通过实验和理论证明自回归 Transformer 对提示词的记忆在生成过程中以幂律衰减（训练后模型 α≈0.55–0.7，快于未训练模型的 α≈0.2–0.5），并证明了提示词影响按 $O(P(\log t)^{L-1}/t)$ 速率被稀释的条件无毛定理。

    

    黑洞只保留其形成物质的少数几个守恒荷。受此类比启发，我们提出这样一个问题：自回归变换器在生成 $t$ 个 token 之后，对其提示词还能保留多少信息。对于两个后续生成相同续写的提示词，它们隐状态之间的差异近似按幂律 $t^{-\alpha}$ 衰减，其中 $\alpha=1$ 是固定提示词在均匀注意力中所占份额收缩的速率。在最初几层之外，经过训练的模型遗忘得更快（$\alpha\approx0.55$–$0.7$），而相同架构下未训练的模型遗忘较慢（$\alpha\approx0.2$–$0.5$）。对于注意力对数几率有界、前置层归一化激活非退化的有限深度 pre-LN softmax 变换器，我们证明了这一偏移量为 $O\big(P(\log t)^{L-1}/t\big)$，其中 $P$ 为提示词长度、$L$ 为网络深度：提示词的直接影响随着上下文的增长而被稀释，而通过续写 token 进行的“接力”传播仅带来对数级……

    arXiv:2610.24133v2 Announce Type: replace  Abstract: A black hole retains only a few conserved charges of the matter it formed from. We are inspired to make the analogy and we ask how much an autoregressive transformer retains of its prompt after generating $t$ tokens. For two prompts followed by the same continuation, the difference of their hidden states decays approximately as a power law $t^{-\alpha}$, where $\alpha=1$ is the rate at which a fixed prompt's share of uniform attention shrinks. Beyond the first layers, trained models forget faster ($\alpha\approx0.55$--$0.7$) than untrained ones of the same architecture ($\alpha\approx0.2$--$0.5$). For finite-depth pre-LN softmax transformers with bounded attention logits and non-degenerate pre-norm activations, we prove that this offset is $O\big(P(\log t)^{L-1}/t\big)$ for prompt length $P$ and depth $L$: the prompt's direct influence is diluted as the context grows, and \emph{relay} through continuation tokens adds only logarithmic
    
[^300]: 智能体式上下文学习：Transformer在闭环交互中能学会哪些算法？

    Agentic In-Context Learning: What Algorithms Can Transformers Learn in Closed-Loop Interaction?

    [https://arxiv.org/abs/2610.23051](https://arxiv.org/abs/2610.23051)

    本文提出“智能体式上下文学习”框架，首次从学习算法视角研究闭环交互中Transformer的可学习性，证明因果Transformer能够近似主动学习、多臂老虎机和在线强化学习等算法并给出回报损失上界，同时刻画了过程奖励优化在何时能保持最大化最终结果的行为。

    

    上下文学习（ICL）的学习算法视角研究的是固定参数的Transformer如何从所提供的示例中学习。语言模型智能体通过交互扩展了这一设定：它们的动作决定了接下来有哪些证据进入上下文。我们将这种机制称为智能体式上下文学习。现有研究主要依赖于外部提供的上下文或带有奖励信号的交互历史，而关于通用工具使用智能体的研究则缺乏学习算法的视角。我们用状态、更新和决策规则来表示一个交互式学习者，其涵盖主动学习、多臂老虎机和在线强化学习（RL）。在表达能力方面，我们构造了能够在有限时间范围内近似这些算法的因果Transformer，并对由此产生的回报损失给出上界。在训练方面，我们刻画了何时优化过程奖励能够保持最大化最终结果的行为。蒸馏实验表明，Transformer…

    arXiv:2610.23051v1 Announce Type: new  Abstract: The learning algorithm view of in-context learning (ICL) studies how fixed-parameter Transformers learn from supplied examples. Language model agents extend this setting through interaction: their actions determine what evidence enters context next. We call this regime Agentic In-Context Learning (Agentic ICL). Existing studies largely depend on externally supplied contexts or reward-bearing interaction histories, while work on general tool-using agents lacks a learning algorithm view. We represent an interactive learner by its state, update, and decision rule, covering active learning, bandits, and online reinforcement learning (RL). For expressivity, we construct causal Transformers that approximate these algorithms over a finite horizon, and bound the resulting loss in return. For training, we characterize when optimizing process rewards preserves behavior that maximizes the final outcome. Distillation experiments show that Transforme
    
[^301]: Realtime-Bench：前沿模型能解决实时交互问题吗？

    Realtime-Bench: Can Frontier Models Solve Real-time Interactive Problems?

    [https://arxiv.org/abs/2610.22155](https://arxiv.org/abs/2610.22155)

    提出Realtime-Bench基准，通过Wall Clock、Paused Clock和Token Clock三种时钟模式评估前沿模型在严格时间约束下的实时交互能力，结果显示最佳成功率仅44%，证明实时交互对现有模型仍是重大挑战。

    

    实时交互要求智能体感知不断变化的环境并在时间约束下采取行动，然而现有的实时流式模型评估主要关注对话质量和流式感知，缺乏对严格时间约束下有效行动能力的测试。我们提出了Realtime-Bench，这是一套精心设计的动态任务基准，其中行动延迟会直接影响任务成败。在Wall Clock（真实时钟）模式下，环境根据实际推理延迟推进，最佳模型的成功率仅为44%，最佳Pass^3仅为36%，表明实时交互仍然极具挑战性。为了将服务延迟与模型本身的实时流式能力区分开，我们将Wall Clock与Paused Clock（暂停时钟）进行对比，后者消除了推理延迟的影响，作为理想化的参照；同时引入Token Clock（令牌时钟），根据生成的令牌数量推进模拟时间，从而独立于物理服务基础设施来测试模型的时间感知能力。

    arXiv:2610.22155v1 Announce Type: new  Abstract: Real-time interaction requires agents to perceive evolving environments and act under temporal constraints, yet existing evaluations of real-time streaming models focus on conversational quality and streaming perception, leaving effective action under strict temporal constraints untested. We introduce Realtime-Bench, a suite of carefully curated dynamic tasks where delayed actions directly affect success. Under Wall Clock, where environments advance according to actual inference latency, the best success rate reaches only 44% and the best Pass^3 is only 36%, demonstrating that real-time interaction remains challenging. To separate serving latency from models' real-time streaming capability, we compare Wall Clock with Paused Clock, which removes inference delays as an idealized reference, and introduce Token Clock, which advances simulated time according to generated token counts, testing temporal awareness independently of physical servi
    
[^302]: 在策略蒸馏需要完成每一次 Rollout 吗？

    Does On-Policy Distillation Need to Complete Every Rollout?

    [https://arxiv.org/abs/2610.21337](https://arxiv.org/abs/2610.21337)

    该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。

    

    在策略蒸馏（On-policy Distillation, OPD）通过对学生模型生成的 rollout 序列提供反馈，有效地将教师模型的能力迁移给学生模型。然而，生成完整的学生 rollout 序列并为每个 token 计算教师反馈，使得 OPD 的计算开销十分高昂。我们提出了随机截断在策略蒸馏（Stochastic Truncated On-Policy Distillation, STOPD），该方法通过随机截断学生 rollout 序列，并利用分块逆概率加权（blockwise IPW）校正由此产生的采样偏差，从而加速 OPD。在每一步训练中，我们在生成前为每个 rollout 采样一个停止位置，让学生模型生成至该位置或自然终止。我们对每个生成的 token 获取教师反馈，并使用分块 IPW 补偿保留后续 token 的较低概率，同时对权重加以控制。我们证明了这种加权方式在期望意义上保持了原始 OPD 的梯度总和不变，并推导出了一步更新的误差界……

    arXiv:2610.21337v1 Announce Type: cross  Abstract: On-policy distillation (OPD) effectively transfers capabilities from a teacher to a student through feedback on student-generated rollouts. However, generating complete student rollouts and computing teacher feedback for every token make OPD computationally expensive. We introduce Stochastic Truncated On-Policy Distillation (STOPD), which accelerates OPD by randomly truncating student rollouts and correcting the resulting sampling bias with blockwise inverse-probability weighting (IPW). At each training step, we sample a stopping position for each rollout before generation and let the student generate until that position or natural termination. We obtain teacher feedback on every generated token and use blockwise IPW to compensate for the lower probability of retaining later tokens while controlling the weights. We prove that this weighting preserves Vanilla OPD's gradient sum in expectation and derive one-step update-error bounds for 
    
[^303]: 心之所想，手之所成：遵循多模态指令的长时程机器人操作

    Let the Mind Conceive, Let the Hand Achieve: Following Multimodal Instructions for Long-Horizon Robotic Manipulation

    [https://arxiv.org/abs/2610.18352](https://arxiv.org/abs/2610.18352)

    Orchestra是一个将任务规划与执行管理从动作生成分离的智能体框架，通过将多模态指令分解为有序子任务、跟踪进度并维护共享上下文，使机器人能够执行长时程多步操作任务。

    

    遵循多步指令的机器人需要在场景发生变化的情况下，按正确的顺序对正确的物体进行操作。它还必须保持对任务进度的感知，并确定何时推进到下一步。为了应对这些挑战，我们提出了Orchestra，这是一种将任务规划与执行管理从动作生成分离出来的智能体框架。Orchestra将完整的多模态指令分解为有序的子任务，跟踪已完成的步骤，并在执行过程中更新相关物体和目标位置的信息。它维护一个共享上下文，其中包含当前活动子任务、空间定位信息以及最近的观测结果。动作策略利用该上下文来生成动作，而单独训练的进度评估器则估计何时推进到下一个子任务。为了评估在原子指令和短轨迹上训练的机器人能否执行更长的子任务序列，我们引入了一个名为RMBench-Co的基准测试。

    arXiv:2610.18352v1 Announce Type: cross  Abstract: A robot following a multi-step instruction needs to act on the right objects in the right order, even as the scene changes. It must also maintain awareness of task progress and determine when to advance. To address these challenges, we introduce Orchestra, an agent harness that separates task planning and execution management from action generation. Orchestra decomposes a complete multimodal instruction into ordered subtasks, tracks completed steps, and updates the locations of relevant objects and destinations during execution. It maintains a shared context containing the active subtask, spatial grounding, and recent observations. An action policy uses this context to generate actions, while a separately trained progress evaluator estimates when to advance to the next subtask. To evaluate whether robots trained on atomic instructions and short trajectories can execute longer subtask sequences, we introduce a benchmark named RMBench-Co
    
[^304]: SarcasTraps：讽刺对土耳其语仇恨言论检测的影响

    SarcasTraps: Effect of Sarcasm on Turkish Hate Speech Detection

    [https://arxiv.org/abs/2610.16957](https://arxiv.org/abs/2610.16957)

    本文构建了SarcasTraps——一个包含479条数据、沿仇恨言论与讽刺双维度标注的土耳其语外部测试集，并通过实验揭示讽刺会显著干扰仇恨言论检测：微调的编码器模型在讽刺性仇恨言论上召回率下降，零样本大语言模型同样受到影响。

    

    标准的仇恨言论检测数据集通常强调显性的有害语言，但针对性伤害也可以通过间接方式表达，而不使用明显仇恨的措辞。讽刺正是这样一种间接表达形式，它可以通过假意称赞、夸张、幽默或表面上的礼貌来传达有害意图。本文介绍了SarcasTraps，这是一个用于分析讽刺对土耳其语仇恨言论检测影响的针对性外部测试集。SarcasTraps包含479条数据，并沿两个维度进行标注：仇恨言论和讽刺。我们使用BERTurk和XLM-RoBERTa在两个现有的土耳其语仇恨言论数据集上训练分类器，并在SarcasTraps上对其进行评估，严格将其用作外部测试集。我们还在零样本设置下于SarcasTraps上评估了GPT-4.1和Llama-3.3-70B。结果表明，讽刺会显著影响仇恨言论检测。微调的编码器模型在检测讽刺性仇恨言论时召回率下降，而零样本大语言模型（原文摘要在此处截断）……

    arXiv:2610.16957v1 Announce Type: cross  Abstract: Standard hate speech detection datasets often emphasize explicit harmful language, but directed harm can also be expressed indirectly, without clearly hateful wording. Sarcasm is one such form of indirect expression, as it can convey harmful intent through mock praise, exaggeration, humor, or apparent politeness. In this paper, we introduce SarcasTraps, a targeted external test set for analyzing the effect of sarcasm on Turkish hate speech detection. SarcasTraps contains 479 entries annotated along two dimensions: hate speech and sarcasm. We train classifiers using BERTurk and XLM-RoBERTa on two existing Turkish hate speech datasets and evaluate them on SarcasTraps, using it strictly as an external test set. We also evaluate GPT-4.1 and Llama-3.3-70B in a zero-shot setting on SarcasTraps. Results show that sarcasm substantially affects hate speech detection. Fine-tuned encoder models lose recall on sarcastic hate, while zero-shot LLMs 
    
[^305]: 自然代码变换使智能体在漏洞分析中产生混淆

    Natural Code Transformation Confuses Agents in Vulnerability Analysis

    [https://arxiv.org/abs/2610.14350](https://arxiv.org/abs/2610.14350)

    本文发现AI智能体在分析具有相同功能和相同漏洞的不同真实代码提交时会产出截然不同的结果，并据此提出从真实开发模式中挖掘自然且功能保持的代码变换，作为防御攻击者智能体的新手段。

    

    arXiv:2610.14350v2 公告类型： cross 摘要：在展示了自动化漏洞发现的强大潜力之后，AI智能体也可以被用来大规模攻击软件系统。当今的软件系统需要一些机制来平衡攻防双方的力量，通过减缓潜在攻击的速度，为防御者争取时间来修补漏洞并强化软件。沿着这一思路，我们对漏洞分析中智能体行为轨迹的研究揭示了一个令人惊讶的发现：对于承载相同功能和相同漏洞的真实世界代码提交，智能体产生了截然不同的分析结果。这一现象在多个代码领域和漏洞类别中持续出现。这促使我们研究源自真实世界软件开发过程的自然且保持功能不变的代码变换，以及其作为防御攻击者智能体的潜力。我们通过从真实世界代码提交中挖掘反复出现的开发模式，构建了一个自然变换语料库。

    arXiv:2610.14350v2 Announce Type: cross  Abstract: Having demonstrated a strong potential for automating vulnerability discovery, AI agents can also be used to attack software systems at scale. Today's software systems need mechanisms to level the playing field by slowing down potential attacks, giving them time to patch vulnerabilities and strengthen software. Along this line, our study of agent trajectories in vulnerability analysis shows a surprising finding: agents produce dramatically different results across real-world code commits that carry the same functionality and the same vulnerability. This phenomenon appears consistently in multiple code domains and vulnerability categories.   It motivates us to study natural and functionality-preserving code transformations derived from real-world software development, and their potential as a defense against attacker agents. We build a corpus of natural transformations by mining recurring development patterns from real-world code commit
    
[^306]: 功耗能否约束隐蔽计算？模拟验证在AI治理中的局限性

    Can Power Draw Constrain Covert Compute? Limits of Analogue Verification for AI Governance

    [https://arxiv.org/abs/2610.07476](https://arxiv.org/abs/2610.07476)

    仅靠功耗等模拟测量手段无法有效约束隐蔽计算——最坏情况下隐藏计算量可达声明容量的116%，即使采用对抗性匹配能量策略也能隐藏至少41%的计算量，这说明AI治理协议不能仅依赖模拟验证机制。

    

    前沿AI条约或关于限制计算量的协议需要外部验证；外部审计员必须能够确认实际运行了多少计算量，以及各方是否遵守协议。模拟的、芯片外的测量手段（如功耗）为验证提供了一个信息通道。然而，当面对主动试图破坏审计的对手时，这些模拟通道能在多大程度上约束计算，目前尚不清楚。我们推导出了β的闭式表达式，其中β表示功耗轨迹无法排除的最大隐藏计算量占声明机器容量的比例。在NVIDIA A100 GPU上的测量表明，最坏情况下β = 1.16，而对抗性的匹配能量策略被证明至少可以隐藏β = 0.41的计算量。因此，仅靠模拟功耗测量对计算的约束力较弱。威胁模型所提供的额外限制条件，例如验证者能够重新执行……

    arXiv:2610.07476v1 Announce Type: cross  Abstract: Frontier AI treaties or agreements on limiting computation require external verification; an external auditor must be able to confirm how much computation actually ran and that parties are adhering to the agreement. Analogue, off-chip measurements such as power draw provide an information channel for verification. It is unknown how well these analogue channels can constrain computation against an adversary who actively tries to subvert the audit. We derive a closed form for $\beta$, the largest hidden computation a power trace cannot exclude, as a fraction of the declared machine capacity. Measurements on NVIDIA A100 GPUs constrain $\beta = 1.16$ in the worst case, while adversarial matched-energy strategies are shown to hide at least $\beta = 0.41$ of compute. Analogue power measurements alone therefore constrain compute weakly. Additional restrictions granted by the threat model, such as the ability of the verifier to re-execute the 
    
[^307]: AEGIS：基于神经闭合项的可微分火星气候模型

    AEGIS: Differentiable Mars Climate Model with Neural Closures

    [https://arxiv.org/abs/2610.04081](https://arxiv.org/abs/2610.04081)

    该论文提出了AEGIS，一个模块化、可微分并引入神经闭合项的火星气候模型，突破了传统Fortran火星GCM无法暴露梯度以支持校准和机器学习的局限。

    

    通用环流模型（GCM）是模拟行星大气的主要工具。它们在理解火星大气方面发挥着至关重要的作用，因为预测火星独特的天气对于进入、下降和着陆等任务操作而言至关重要。火星给这些模型带来了不寻常的挑战：与地球相比，火星的观测数据稀少；此外，稀薄的二氧化碳大气与辐射活跃的尘埃循环共同造就了一个多变的、昼夜温差巨大的大气环境，且在地球上没有真正的类比对象可用于验证。现有的火星GCM，包括LMD PCM、NASA艾姆斯火星GCM和PlanetWRF，虽然成熟且物理细节丰富，但都是用传统Fortran实现的，采用有限差分或有限体积求解器，并且不对外提供梯度，无法用于校准或机器学习。在此，我们提出AEGIS，一个模块化的可微分火星气候模型，它将火星独特的大气物理与（原文在此处截断）

    arXiv:2610.04081v2 Announce Type: replace-cross  Abstract: General circulation models (GCMs) are the primary tool for simulating planetary atmospheres. They play a vital role in understanding Mars's atmosphere, as forecasting its unique weather is mission-critical for operations such as entry, descent, and landing. Mars poses unusual challenges for these models, as observations are sparse compared to Earth. In addition, a thin $CO_2$ atmosphere alongside a radiatively active dust cycle creates a volatile atmosphere with large diurnal temperature swings and no true terrestrial analog for validation. Existing Mars GCMs, including the LMD PCM, the NASA Ames Mars GCM, and PlanetWRF, are mature and physically detailed but are implemented in legacy Fortran with finite-difference or finite-volume solvers, and they do not expose gradients for calibration or machine learning. Here we present AEGIS, a modular differentiable Mars climate model that couples Mars's unique atmospheric physics to the
    
[^308]: 基于不完美代理奖励的可靠自进化

    Reliable Self-Evolution with Imperfect Proxy Rewards

    [https://arxiv.org/abs/2610.02975](https://arxiv.org/abs/2610.02975)

    该论文提出保形区间驱动的自进化方法（CISE），通过条件保形推断和在线密度比估计构建候选特定的奖励区间，以应对不完美代理奖励导致的假阳性问题，从而实现更可靠的LLM自进化搜索。

    

    基于大语言模型（LLM）的自进化搜索是实现科学发现的一种有前景的方法。然而，在某些领域中，对每个候选方案进行高保真度评估的成本过高。因此，此类环境下的自进化系统不得不依赖低成本但不完美的代理奖励，而这类代理奖励可能会给不可行的候选方案赋予高分。这些假阳性结果可能同时污染最终输出和用于指导后续生成的反馈。这促使我们引入统计校准的奖励区间，以实现更可靠的自进化搜索。我们提出了保形区间驱动的自进化方法（Conformal Interval-Driven Self-Evolution, CISE），该方法利用条件保形推断和逐迭代的在线密度比估计来构建针对特定候选方案的奖励区间。CISE在进化反馈中使用保守的基于区间的奖励，并且仅当所有所需属性区间完全落入各自可行区域内时才返回候选方案……

    arXiv:2610.02975v1 Announce Type: new  Abstract: Large language model (LLM)-based self-evolving search is a promising approach to scientific discovery. However, high-fidelity evaluation of every candidate is prohibitively expensive in some domains. Self-evolving systems in such settings therefore rely on low-cost but imperfect proxy rewards, which may assign high scores to infeasible candidates. These false positives may contaminate both the final output and the feedback used to guide subsequent generations. This motivates statistically calibrated reward intervals for more reliable self-evolving search. We propose Conformal Interval-Driven Self-Evolution (CISE), which constructs candidate-specific reward intervals using conditional conformal inference and iteration-wise online density-ratio estimation. CISE uses conservative interval-based rewards for evolutionary feedback and returns candidates only when all required property intervals lie entirely within their respective feasible reg
    
[^309]: 当终端智能体训练停滞时：揭秘数据生成与验证的挑战

    When Terminal-Agent Training Stalls: Demystifying Data Generation and Verification Challenge

    [https://arxiv.org/abs/2610.02405](https://arxiv.org/abs/2610.02405)

    该论文揭示了用前沿模型作为元智能体自动生成终端强化学习训练任务时的三类故障（基准无效、评测框架脆弱、奖励错位），并通过实验证明任务可解性区间是模型特定的，主张应将可解性区间校准、验证器审计和基础设施错误核算作为一等评估标准。

    

    使用 Claude Opus 这类前沿模型作为元智能体来生成用于强化学习训练的终端任务和验证器的做法正日益普遍。然而，一个可运行的 Docker 镜像和可执行的测试套件并不能保证终端智能体训练的端到端流程忠实可靠。针对这一空白，我们提出了一个元智能体流水线，并诊断出三类故障：基准无效性、评测框架（harness）脆弱性和奖励错位。通过重新设计提示词和扩展上下文，基线可解性提升了 5.6 倍，但一个 9B 模型在 Claude Opus 生成的任务上，于 20 步内平均 pass@2 饱和在 81.3%。在不改变训练配置的情况下加入困难任务，平均 pass@2 降至 20.6%，这有力地证明了可解性区间是模型特定的。这些发现表明，元智能体的可靠性需要将可解性区间校准、验证器审计和基础设施错误核算作为一等评估标准，而非事后补救……

    arXiv:2610.02405v1 Announce Type: new  Abstract: Using a frontier model like Claude Opus as a meta-agent to generate terminal tasks and verifiers for RL training is increasingly common. Yet a runnable Docker image and executable test suite do not guarantee a faithful end-to-end pipeline for terminal agent training. We present a meta-agent pipeline motivated by this gap, diagnosing three classes of failure: benchmark invalidity, harness brittleness, and reward misalignment. Prompt redesign and context extension raise baseline solvability 5.6 times, but a 9B model saturates at 81.3% mean pass@2 within 20 steps on Claude Opus-generated tasks. Adding hard tasks reduces mean pass@2 to 20.6% without changing the training configuration, a strong evidence that the solvability band is model-specific. These findings demonstrate that meta-agent reliability requires solvability-band calibration, verifier audits, and infrastructure error accounting as first-class evaluation criteria, not post-hoc d
    
[^310]: Mingbird：一种使小型开放模型能够完成真实任务的本地优先智能体框架

    Mingbird: A Local-First Agent Harness Enabling Small Open Models to Complete Real Tasks

    [https://arxiv.org/abs/2610.02001](https://arxiv.org/abs/2610.02001)

    提出了Mingbird，一个面向Windows和Ollama的本地优先智能体框架，通过十项针对性补偿机制（如字节级净零预填充预算、完成门控和签名级循环检测）来克服小模型的典型失败形式，使2-9B的开放模型能够在普通笔记本上完成真实任务，并证明许多失败源于框架而非模型本身。

    

    小型开放权重模型（2-9B）可以在普通笔记本电脑上运行，但在云规模的智能体框架下，它们很少能完成真实任务：工具预填充溢出上下文、自我修正发散、工具演示陷入循环，任务被悄悄放弃。我们通过受控的单机对比和一个第三方基准测试提供了证据，表明这些失败中有很大一部分归因于框架本身而非模型。我们提出了Mingbird，一个面向Windows和Ollama的本地优先智能体框架，其十项机制逐一弥补小模型的失败形式，其中三项具有代表性：字节级净零预填充预算、在接受完成之前重新读取任务的完成门控，以及签名级循环检测。在LRAB（一项保持机器、模型、预算和评分固定的受控对比：4个框架 × 4个开放模型（2B-35B）× 18个真实任务，采用确定性产物评分）上，

    arXiv:2610.02001v1 Announce Type: new  Abstract: Small open-weight models (2-9B) run on ordinary laptops, but under cloud-scale agent harnesses they rarely complete real tasks: tool prefill overflows the context, self-correction diverges, tool demonstrations loop, and tasks are silently abandoned. We present evidence, from a controlled single-machine comparison and one third-party benchmark, that a substantial share of these failures is attributable to the harness rather than the model. We introduce Mingbird, a local-first agent harness for Windows and Ollama whose ten mechanisms compensate point-by-point for small-model failure forms, three of them representative: a byte-level net-zero prefill budget, a finish gate that re-reads the task before accepting completion, and signature-level loop detection. On LRAB, a controlled comparison holding machine, models, budgets, and scoring fixed (4 harnesses $\times$ 4 open models (2B-35B) $\times$ 18 real tasks, deterministic artifact scoring),
    
[^311]: 低损失区域的错位导致Grokking（顿悟）现象

    Misalignment of Low-Loss Regions Causes Grokking

    [https://arxiv.org/abs/2610.00620](https://arxiv.org/abs/2610.00620)

    论文提出基于模式连通性与低损失区域几何的分析框架，指出grokking现象源于训练集与验证集所诱导的低损失区域之间的错位，并以稳定的反grokking反例挑战了现有的相关性解释。

    

    Grokking是指模型在已经过拟合训练集之后，验证集泛化能力延迟出现的现象。尽管该现象最早在使用Transformer训练的小型算法任务中被观察到，但其底层机制至今仍无定论。在本工作中，我们开发了一个基于模式连通性与低损失区域几何结构的分析框架。该框架预测标准的模运算设定并不总是产生grokking现象：在保持对称性的训练/验证集划分下，我们观察到一个稳定的反grokking案例，其中验证集性能始终未能恢复。这一反例对现有的若干关于grokking的相关性解释提出了挑战。更广泛地说，我们的分析框架和结果进一步表明，当训练集划分与验证集划分所诱导的低损失区域发生错位时，grokking现象便会出现。一旦这些区域充分对齐，仅凭训练超参数就无法……（摘要在此处截断）

    arXiv:2610.00620v1 Announce Type: cross  Abstract: Grokking refers to the delayed emergence of validation-set generalization after a model has already overfit the training set. Although first observed in small algorithmic tasks trained with transformers, its underlying mechanism remains unsettled. In this work, we develop an analysis framework based on mode connectivity and the geometry of low-loss regions. The framework predicts that the standard modular-arithmetic setting does not always produce grokking: under a symmetry-preserving train/validation split, we observe a stable anti-grokking case in which validation performance does not recover. This counterexample challenges several existing correlational explanations of grokking. More broadly, our analysis framework and results further suggest that grokking arises when the low-loss regions induced by the training and validation partitions are misaligned. Once these regions become well aligned, training hyperparameters alone cannot pr
    
[^312]: 基于预测误差触发伪标签重训练的在线时间序列预测方法

    Pseudo-Label-Triggered Retraining from Forecast Errors for Online Time Series Forecasting

    [https://arxiv.org/abs/2609.39789](https://arxiv.org/abs/2609.39789)

    本文提出PILOT框架，利用预测误差构建伪标签并训练轻量级评分器，从误差动态中学习时间序列预测模型“何时重训练”，从而在有限部署资源下有效维持预测性能。

    

    真实世界的时间序列预测系统运行在非平稳数据流环境中，预测性能可能随时间推移而下降。虽然重训练可以恢复性能，但会带来不小的计算和运营成本。在部署资源有限的情况下，关键挑战因此不仅在于如何重训练，还在于何时重训练。现有的重训练策略通常依赖于漂移警报或模型过时程度等间接指标，而我们则使用实际发生的预测误差作为直接的部署反馈。在本文中，我们提出了PILOT（伪标签信息学习的在线触发器），这是一个从预测误差动态中学习何时重训练的在线重训练框架。由于真实的重训练标签不可用，PILOT从未来预测误差的增加中构建伪标签，并训练一个轻量级评分器根据观测到的误差状态来预测该伪标签。在部署时，PILOT仅使用……（原文摘要在此处被截断）

    arXiv:2609.39789v1 Announce Type: cross  Abstract: Real-world time series forecasting systems operate under non-stationary data streams, where forecasting performance may degrade over time. Although retraining can recover the performance, it incurs non-trivial computational and operational costs. Under limited deployment resources, the key challenge is therefore not only how to retrain but also when to retrain. While existing retraining policies often rely on indirect indicators such as drift alarms or model staleness, we instead use realized forecast errors as direct deployment feedback. In this paper, we propose PILOT (Pseudo-label-Informed Learned Online Trigger), an online retraining framework that learns when to retrain from forecast-error dynamics. Since ground-truth retraining labels are unavailable, PILOT constructs a pseudo-label from future increases in forecast error and trains a lightweight scorer to predict it from observed error states. At deployment, PILOT uses only comp
    
[^313]: LeapQuant：基于精确循环状态量化的高效线性注意力

    LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization

    [https://arxiv.org/abs/2609.38166](https://arxiv.org/abs/2609.38166)

    LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。

    

    近期的大语言模型（LLM）越来越多地采用混合设计，用线性注意力（如 Gated DeltaNet (GDN) 和 Kimi Delta Attention (KDA)）替代标准注意力。尽管这些方法将上下文压缩为固定大小的循环状态，大幅降低了长上下文处理的成本，但反复读取和更新该状态仍是推理过程中的主要瓶颈。量化是降低这一成本的自然途径，但由于舍入误差的累积以及状态中异常行列的存在，量化可能显著降低模型质量。为应对这些挑战，我们提出了 LeapQuant，这是一种无需训练的方法，在 8 位循环状态量化下实现了近乎无损的性能。首先，为减轻误差累积，我们提出了按窗口量化，即跳过一个 token 窗口，仅在窗口结束时对状态量化一次。在窗口内部，输出由固定的低比特……（摘要至此截断）

    arXiv:2609.38166v1 Announce Type: cross  Abstract: Recent LLMs increasingly adopt hybrid designs that replace standard attention with linear attention, such as Gated DeltaNet (GDN) and Kimi Delta Attention (KDA). Although they compress the context into a fixed-size recurrent state and substantially reduce the cost of long-context processing, repeatedly reading and updating that state remains a major inference bottleneck. Quantization offers a natural way to reduce this cost, but can significantly degrade model quality, due to the accumulation of rounding errors and the presence of outlier rows and columns in the state. To address these challenges, we propose LeapQuant, a training-free method that achieves near-lossless performance under 8-bit recurrent-state quantization. First, to mitigate error accumulation, we propose per-window quantization, which leaps over a window of tokens and quantizes the state only once at its end. Within a window, outputs are computed from the fixed low-bit
    
[^314]: Code4Scene：面向3D场景构建与编辑的编程智能体基准测试

    Code4Scene: Benchmarking Coding Agents for Constructing and Editing 3D Scenes

    [https://arxiv.org/abs/2609.36777](https://arxiv.org/abs/2609.36777)

    该论文提出Code4Scene基准，通过190个基于Unreal Engine的案例，直接在生成的引擎原生场景上评估编程智能体从语言规范构建3D场景的空间推理能力以及对场景状态进行精确编辑的能力。

    

    前沿编程智能体现在已经能够编写并执行代码来创建3D环境，但它们是否能可靠地理解3D结构并精确控制场景状态仍不清楚。生成的3D场景是一个持久、可执行的产物：一个看似可信的渲染结果可能掩盖错误的空间关系、相互穿插的物体或意外的修改。我们提出Code4Scene，一个包含190个Unreal Engine案例的基准测试，这些案例基于人工组装的场景构建，在共享的执行接口下，从两个互补的设置来评估编程智能体。构建任务测试从开放式语言规范进行场景级空间推理的能力，其中存在多种有效实现；编辑任务测试对场景状态的精确控制，智能体必须根据参考图像恢复目标场景，同时保留场景中的其他所有内容。Code4Scene不对代码或渲染视图进行评分，而是对生成的引擎原生场景进行评估，衡量其任务完成度与瑕疵。

    arXiv:2609.36777v1 Announce Type: new  Abstract: Frontier coding agents can now write and execute code that authors 3D environments, but whether they reliably understand 3D structure and precisely control scene state remains unclear. The generated 3D scene is a persistent, executable artifact: a convincing render can hide incorrect spatial relations, intersecting objects, or unintended modifications. We introduce Code4Scene, a benchmark of 190 Unreal Engine cases built from human-assembled scenes that evaluates coding agents on two complementary settings under a shared execution interface. Construction tests scene-level spatial reasoning from open-ended language specifications, where many realizations are valid; editing tests precise control of scene state, where the agent must recover the target scene from reference images while preserving everything else. Rather than scoring code or rendered views, Code4Scene evaluates the generated engine-native scene for task fulfillment, artifact 
    
[^315]: 可表示却学不到：编码秩与交互预测的误差下限

    Representable but Unlearned: Encoding Rank and the Interaction-Prediction Floor

    [https://arxiv.org/abs/2609.36208](https://arxiv.org/abs/2609.36208)

    该论文提出一种无需标签或拟合模型即可量化输入编码秩缺陷所致预测误差下限的方法，并证明siRNA交互预测中仅训练期特征掩码造成的秩下降（140降至72）会产生占拟合模型误差14.6%的不可逾越下限，而仅恢复三个化学列或移除掩码即可完全消除该下限。

    

    输入编码会限制预测器能够联合再现哪些测量对比，即使没有任何单一对比被强制归零。我们无需标签、损失函数或已拟合的模型，仅从编码器的等价类和固定的对比设计出发，即可计算出可达对比空间；将记录的对比投影到该空间上，便可为这些等价类上的任何无约束解码器给出经验误差下限。在一个包含140个矩形的siRNA交互面板上，图神经网络中仅用于训练阶段的特征掩码将165个终点状态合并为90个类，并将140个交互对比的秩削减至72。由此产生的误差下限为0.009980，占拟合模型交互平方误差的14.6%；而拟合模型本身的误差达到0.068335，略差于完全不预测任何交互的对照基线。只需最少恢复三个化学列即可恢复满秩。移除掩码后重新拟合可完全消除该下限，但交互MSE……

    arXiv:2609.36208v1 Announce Type: cross  Abstract: Input encodings can restrict which measured contrasts a predictor can jointly reproduce, even when no single contrast is forced to vanish. We compute the attainable contrast space from an encoder's equivalence classes and a fixed contrast design, without labels, loss, or a fitted model; projecting the recorded contrasts onto that space gives an empirical error floor for any unrestricted decoder on those classes. On a 140-rectangle siRNA interaction panel, a graph neural network's training-only feature mask merges 165 endpoint states into 90 classes and cuts the rank of the 140 interaction contrasts to 72. The resulting floor is 0.009980, which is 14.6% of the fitted model's interaction squared error; the fitted model reaches 0.068335, slightly worse than a control predicting no interaction at all. A minimum of three restored chemistry columns recovers full rank. Refitting without the mask removes the floor entirely, yet interaction MSE
    
[^316]: SkillPE：面向创造力的电影级技能演化用于文本到视频提示词增强

    SkillPE: Creativity-Oriented Cinematic Skill Evolution for Text-to-Video Prompt Enhancement

    [https://arxiv.org/abs/2609.34335](https://arxiv.org/abs/2609.34335)

    SkillPE 提出了一个面向创造力的提示词增强框架，通过从专家电影知识出发，利用共鸣、不和谐与分歧三类电影参考引导演化可复用的电影级技能，帮助非专家用户提升文本到视频生成的专业叙事与创意质量。

    

    实现高质量的电影级文本到视频生成对非专业用户而言仍具挑战性，因为他们的提示词往往缺乏专业的叙事与创意设计。现有的提示词增强（PE）方法和静态电影级技能在探索多样化创意实现方面的能力有限。我们提出了 SkillPE，一个面向创造力的提示词增强框架，通过参考引导的探索系统性地演化可复用的电影技能。SkillPE 从专家提供的电影知识出发，初始化并规范化了 20 个结构化种子技能。随后，它检索电影参考并将其分类为共鸣者（良好匹配）、不和谐者（弱匹配）和分歧者（具有创意启发性的近似匹配）。共鸣者与不和谐者用于精炼技能的适用性与执行方式，而分歧者则通过保留用户意图的受控变异来激发替代性的电影实现方案。候选技能通过……（原文摘要在此处截断）

    arXiv:2609.34335v2 Announce Type: replace-cross  Abstract: Achieving high-quality cinematic text-to-video generation remains challenging for non-experts, whose prompts often lack professional narrative and creative design. Existing prompt enhancement (PE) methods and static cinematic skills offer limited capabilities for exploring diverse creative realizations. We propose SkillPE, a creativity-oriented PE framework that systematically evolves reusable cinematic skills through reference-guided exploration. Starting from expert-provided cinematic knowledge, SkillPE initializes and normalizes 20 structured seed skills. It then retrieves movie references categorized as resonators (good matches), dissonants (weak matches), and divergents (creatively inspiring near-misses). Resonators and dissonants refine skill applicability and execution, while divergents inspire alternative cinematic realizations through controlled mutations that preserve user intent. Candidate skills are assessed through
    
[^317]: PhysAlign：多模态物理推理中基于证据的角色对齐基准

    PhysAlign: A Benchmark for Evidence-Grounded Role Alignment in Multimodal Physics Reasoning

    [https://arxiv.org/abs/2609.33319](https://arxiv.org/abs/2609.33319)

    提出了PhysAlign基准，通过解耦视觉识别与物理角色分配的局部探针（3,341个人工验证探针、986个物理问题），系统评估多模态模型能否将物理图像中识别的信息正确对应到其预期物理角色。

    

    物理图像理解中的一个关键挑战是将视觉信息与其所描述的物理实体、关系和条件正确关联。即使某个数值、符号或其他局部元素被准确识别，若将其分配给错误的实体或范围，也可能扭曲潜在的物理前提并导致错误的推理。为了系统地研究这一挑战，我们提出了PhysAlign，一个旨在评估多模态模型能否将从物理图像中识别出的信息正确关联到其预期物理角色的基准。通过局部探针和受控变体将视觉识别与物理角色分配解耦，PhysAlign将对应错误与识别失败分离开来。该基准包含3,341个经人工验证的探针，覆盖986个物理问题，能够对视觉识别和物理角色对应关系进行大规模的系统性评估。我们进一步……

    arXiv:2609.33319v2 Announce Type: replace  Abstract: A key challenge in physics diagram understanding is correctly associating visual information with the physical entities, relations, and conditions it describes. Even when a value, symbol, or other local element is accurately recognized, assigning it to the wrong entity or scope can distort the underlying physical premise and lead to incorrect reasoning. To systematically study this challenge, we introduce \textbf{PhysAlign}, a benchmark designed to assess whether multimodal models correctly associate information recognized from physics diagrams with its intended physical role. By disentangling visual recognition from physical-role assignment through localized probes and controlled variants, PhysAlign isolates correspondence errors from recognition failures. It contains 3,341 human-validated probes spanning 986 physics problems, enabling systematic evaluation of visual recognition and physical-role correspondence at scale. We further 
    
[^318]: 来自预训练表示的直觉向量引导基于扩散的推理

    Intuition Vectors from Pretrained Representations Steer Diffusion-Based Reasoning

    [https://arxiv.org/abs/2609.32619](https://arxiv.org/abs/2609.32619)

    该研究发现，冻结自监督视觉编码器表示空间中的简单向量差（“直觉向量”）能够直接支持抽象视觉推理，无需依赖语言推理或任务特定的表示学习。

    

    抽象视觉推理需要从少量示例中识别出关系，并将其应用于新的输入。近期的进展主要由大型语言模型、视觉-语言模型或专门的领域特定架构所驱动。目前尚不清楚抽象推理是否可以直接由预训练视觉表示的关系几何结构来支持，而无需依赖基于语言的推理或任务特定的表示学习。我们使用冻结的自监督视觉编码器来研究这一可能性。通过潜向量算术，我们证实了两个自监督编码器的表示几何结构能够支持Bongard问题上的抽象概念判别，并在ARC任务中编码输入-输出对之间的系统性关系。图像嵌入之间的简单向量差——我们称之为“直觉向量”——能够捕捉在同类实例间泛化的变换。

    arXiv:2609.32619v2 Announce Type: replace-cross  Abstract: Abstract visual reasoning requires identifying relationships between few examples and applying them to novel inputs. Recent advances have been driven by large language and vision-language models or specialized domain-specific architectures. It remains unclear whether abstract reasoning can be supported directly by the relational geometry of pretrained visual representations, without relying on language-based inference or task-specific representation learning. We investigate this possibility using frozen self-supervised vision encoders. With latent vector arithmetic, we confirmed that the representational geometry of two self-supervised encoders support abstract concept discrimination on Bongard problems and encode systematic relationships between input-output pairs in ARC tasks. Simple vector differences between image embeddings, which we call intuition vectors, capture transformations that generalize across instances of the sa
    
[^319]: Video-HopChain：面向视频推理模型的多跳问题与置信度门控探索

    Video-HopChain: Multi-Hop Questions and Confidence-Gated Exploration for Video Reasoning Models

    [https://arxiv.org/abs/2609.25773](https://arxiv.org/abs/2609.25773)

    该论文提出 Video-HopChain，首次将多跳问题合成引入视频领域，通过串联三到六个是非题并以整数之和作为可精确验证的最终答案，构建了含 22,550 个多跳视频问题及基准的数据集，为视频推理模型的 RLVR 训练提供长视觉证据链并暴露累积性推理错误。

    

    arXiv:2609.25773v3 公告类型：replace-cross 摘要：HopChain 已在静态图像上证明，多跳数据合成能够改善视觉-语言推理，因为长思维链推理会暴露那些随步骤不断累积的错误，而大多数用于可验证奖励强化学习的数据很少要求一条视觉证据链，因此这些弱点很可能得不到暴露。我们在视频领域观察到了同样的问题，而该框架尚未在视频领域被探索过。为此，我们构建了 Video-HopChain，这是一个基于 13,378 个视频、包含 22,550 个多跳视频问题的数据集，并附带一个由 1,000 个问题组成的留存基准测试集。每个问题将关于同一视频中若干时刻的三到六个是非题串联起来，且每个是非题根据其答案产生两个整数中的一个。最终答案是这些整数之和，因此对该和的精确匹配即可提供 RLVR 所需的可验证奖励。我们首先在标准视频数据集上使用 GRPO 训练 Qwen3-VL-8B，随后进行第二次……（原文摘要至此截断）

    arXiv:2609.25773v3 Announce Type: replace-cross  Abstract: HopChain has shown on still images that multi-hop data synthesis improves vision-language reasoning, because long chain-of-thought reasoning exposes errors that compound across steps, while most data used for reinforcement learning with verifiable rewards (RLVR) rarely demands a chain of visual evidence, so these weaknesses are likely to stay unexposed. We observe the same problem in video, where this framework has not yet been explored. We therefore build Video-HopChain, a dataset of 22,550 multi-hop video questions over 13,378 videos, together with a held-out benchmark of 1,000 questions. Each question chains three to six yes/no questions about moments in one video, and each yields one of two integers depending on its answer. The final answer is the sum of these integers, so an exact match on that sum gives the verifiable reward that RLVR needs. We first train Qwen3-VL-8B with GRPO on a standard video dataset, and a second st
    
[^320]: FLARE：基于生成式奖励模型的长程编码智能体全生命周期密集监督范式

    FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model

    [https://arxiv.org/abs/2609.23808](https://arxiv.org/abs/2609.23808)

    提出FLARE，一种由轻量级生成式奖励模型驱动的全生命周期密集监督范式，通过离线因果诊断框架RADAR提取无后见之明的监督信号，为长程编码智能体提供实时步骤级风险反馈，解决稀疏二值奖励带来的信用分配危机。

    

    虽然测试时扩展提升了大型语言模型（LLM）智能体在长程软件工程（SWE）任务中的表现，但稀疏的二值奖励（通过/失败）造成了严重的信用分配危机，并浪费了失败的探索轨迹。当前的轨迹优化与扩展方法成本高昂且在结构上存在局限，它们要么依赖缺乏因果诊断的启发式状态复用，要么依赖缺乏可操作在线引导的延迟标量评分。我们提出了FLARE（全生命周期对齐与奖励引擎），这是一种由轻量级生成式奖励模型（GRM）驱动的新型密集监督范式。首先，RADAR作为一个离线的因果感知诊断框架，通过因果链回溯提取高保真、无后见之明的监督信号，从而蒸馏出一个能够提供实时、步骤级风险反馈的GRM。其次，FLARE利用该GRM在智能体的整个生命周期中对其进行持续优化。在推理阶段，FLARE充当主动脚手架，自主……

    arXiv:2609.23808v1 Announce Type: new  Abstract: While test-time scaling enhances Large Language Model (LLM) agents in long-horizon software engineering (SWE), sparse binary rewards (Pass/Fail) create a severe credit assignment crisis and waste failed exploratory trajectories. Current trajectory optimization and scaling methods are costly and structurally limited, relying on heuristic state reuse without causal diagnosis or delayed scalar scoring without actionable online guidance. We propose FLARE (Full-Lifecycle Alignment and Reward Engine), a novel dense supervision paradigm driven by a lightweight Generative Reward Model (GRM). First, RADAR, an offline causal-aware diagnostic framework, extracts high-fidelity, hindsight-free supervision through causal-chain backtracking to distill a GRM providing real-time, step-level risk feedback. Second, FLARE uses this GRM to continuously optimize the agent across its entire lifecycle. During inference, FLARE acts as an Active Scaffold, autonom
    
[^321]: RPMem：为LLM智能体学习跨会话的长期循环参数化记忆

    RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents

    [https://arxiv.org/abs/2609.23466](https://arxiv.org/abs/2609.23466)

    RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。

    

    长时间运行的LLM智能体需要能够跨会话持续存在并不断演化的记忆。基于文本的记忆在每次查询时都需要检索和重建过去的交互，随着历史记录的增长，长程性能越来越依赖于检索质量和上下文推理能力。参数化记忆将经验直接编码到模型计算中，但现有方法对跨会话记忆演化的支持有限，且其与特定骨干模型的耦合进一步限制了模型替换后的记忆复用。我们提出了RPMem，这是一种两阶段架构，通过前向计算将每个会话编译为与模型无关的潜在记忆，并通过任务训练的循环门控选择性地将其与保留的记忆进行整合。整合后的记忆随后被映射为特定于骨干模型的低秩适配（LoRA）参数，使得编码能力在骨干模型被替换时仍可迁移。在三个（数据集/任务上）的评估……

    arXiv:2609.23466v1 Announce Type: new  Abstract: Long-running LLM agents require memory that persists and evolves across sessions. Text-based memory retrieves and reconstructs past interactions at every query, making long-horizon performance increasingly dependent on retrieval quality and contextual reasoning as histories grow. Parametric memory encodes experience directly into model computation, but existing approaches provide limited support for cross-session memory evolution. Their coupling to a specific backbone further restricts memory reuse after model replacement. We introduce RPMem, a two-stage architecture that compiles each session into a model-independent latent memory through forward computation and selectively integrates it with retained memory via a task-trained recurrent gate. The consolidated memory is then mapped to backbone-specific low-rank adaptation (LoRA) parameters, allowing the encoding capability to transfer when the backbone is replaced. Evaluation across thre
    
[^322]: EDCT-Bench：通过解释驱动的反事实测试揭示视觉语言模型中的忠实性差距

    EDCT-Bench: Uncovering Faithfulness Gaps in VLMs via Explanation-Driven Counterfactual Testing

    [https://arxiv.org/abs/2609.17953](https://arxiv.org/abs/2609.17953)

    提出了一种基于干预的解释驱动反事实测试方法EDCT，通过对模型解释中引用的视觉概念进行经验证的最小编辑来系统评估VLM解释的忠实性，并据此构建了覆盖三个领域的基准EDCT-Bench，揭示了现有VLM中普遍存在的显著忠实性差距。

    

    视觉语言模型（VLM）能够生成听起来合理、却与其所引用的视觉证据不一致的自然语言解释（NLE）。我们提出了EDCT，这是一种基于干预的测试协议：它提取模型解释中所引用的视觉概念，对这些概念施加经过验证的最小编辑，并检验由此产生的答案和解释是否仍与编辑后的图像保持一致。利用该协议，我们构建了EDCT-Bench，这是一个覆盖三个互补领域的综合性基准：知识密集型视觉问答（OK-VQA）、安全关键驾驶（DriveLM）以及3D空间推理（3DSRBench）。在所评估的多个VLM上，EDCT揭示了显著的忠实性差距，模型经常产生与经过验证的视觉变化不一致的回答。最后，我们的微调研究表明，EDCT生成的反事实样本能够提供高影响力的训练数据。

    arXiv:2609.17953v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) can produce Natural Language Explanations (NLEs) that sound plausible yet remain inconsistent with the visual evidence they cite. We present Explanation-Driven Counterfactual Testing (EDCT), an intervention-based protocol that extracts visual concepts cited in a model's explanation, applies verified minimal edits to them, and tests whether the resulting answer and explanation remain consistent with the edited image. Using this protocol, we create EDCT-Bench, a comprehensive benchmark spanning three complementary domains: knowledge-intensive visual question answering (OK-VQA), safety-critical driving (DriveLM), and 3D spatial reasoning (3DSRBench). Across the evaluated VLMs, EDCT reveals substantial faithfulness gaps, with models frequently producing responses inconsistent with verified visual changes. Finally, our fine-tuning study suggests that EDCT-generated counterfactuals provide high-impact training s
    
[^323]: 跨声音的结构：使用冻结音频编码器比较四种发声曲库中的声学事件类型累积与序列依赖性

    Structure Across Voices: Comparing acoustic-event type accumulation and sequence dependence across four vocal repertoires using frozen audio encoders

    [https://arxiv.org/abs/2609.16612](https://arxiv.org/abs/2609.16612)

    本研究利用统一的冻结音频编码器方法跨物种比较四种发声系统，发现不同物种的发声结构优势取决于所测量的声学属性和时间尺度，而非简单的优劣排序。

    

    发声曲库在声学事件类型累积和时间组织方面可能存在差异，但由于语料库使用不同的原生事件且序列数据量不均等，直接比较十分困难。我们在匹配事件数量和局部序列机会的条件下，使用相同的冻结音频编码器程序比较了抹香鲸的尾声、人类语音音素、孟加拉雀的音节和普通狨猴的叫声。鲸类表现出最快的事件类型累积；雀类表现出最强的即时依赖性和重复子序列复现。具有物理可解释性的声学特征能够恢复该特征图谱的互补部分，无需聚类的连续分析支持了鲸类广泛的声学覆盖范围，且保留来源和位置信息的零假设检验保留了雀类的两种顺序效应。扩展预测上下文使比较结果向鲸类倾斜。因此，曲库差异取决于所测量的声学属性和时间尺度，而非形成一个简单的排序。

    arXiv:2609.16612v1 Announce Type: cross  Abstract: Vocal repertoires can differ in acoustic-event type accumulation and temporal organization, yet direct comparison is difficult because corpora use different native events and unequal amounts of sequence. We compare sperm whale codas, human speech phones, Bengalese finch syllables, and common marmoset calls using the same frozen-audio-encoder procedure while matching event count and local sequence opportunity. Whale shows the fastest type accumulation; Finch shows the strongest immediate dependence and repeated-subsequence recurrence. Physically interpretable acoustics recover complementary parts of this profile, continuous analyses without clustering support broad Whale acoustic coverage, and source- and position-preserving nulls retain both Finch order effects. Extending predictive context shifts the comparison toward Whale. Thus repertoire differences depend on the acoustic property and temporal scale measured rather than forming a s
    
[^324]: 摊销缩放定律构建成本

    Amortizing Scaling Law Construction Costs

    [https://arxiv.org/abs/2609.05016](https://arxiv.org/abs/2609.05016)

    该论文提出了一种基于贝叶斯优化的高效缩放定律构建框架，通过逐步扩展计算预算并结合代理模型幻想评估来恢复广泛的实验网格，大幅降低了构建缩放定律的计算成本。

    

    缩放定律指导着大型基础模型的训练设计选择，但推导缩放定律需要在超参数、token预算和参数数量上进行穷举式的网格训练，这在计算上非常昂贵。然而，拟合缩放定律只需要跨计算规模的最佳损失前沿，大部分已训练的配置都会被丢弃。我们提出了一个高效的缩放定律构建框架，将数据收集表述为贝叶斯优化问题，并引入了在受限计算预算下比较缩放定律拟合方法的评估指标。我们发现，在采集过程中逐步扩展计算预算（类似于实践中按计算规模顺序评估配置的方式），可以显著提高恢复效率。随后，通过代理模型幻想评估来增强观察到的配置，可以恢复更广泛的实验网格，从而实现准确的缩放定律拟合。

    arXiv:2609.05016v1 Announce Type: cross  Abstract: Scaling laws guide the design choices for training large foundation models, but deriving them involves training an exhaustive grid over hyperparameters, token budgets, and parameter counts, which is computationally expensive. Fitting a scaling law, however, only requires the best-loss frontier across compute scales, discarding most of the trained configurations. We propose a framework for efficient scaling law construction that formulates data collection as a Bayesian optimization problem, and introduce metrics for comparing scaling law fitting methods under constrained compute budgets. We find that progressively expanding the compute budget during acquisition, mirroring the compute-ordered evaluation of configurations in practice, substantially improves recovery efficiency. Augmenting the observed configurations with surrogate-fantasized evaluations then recovers the broader experimental grid, allowing accurate scaling law fitting wit
    
[^325]: 记忆信任差距：持久记忆智能体中依赖模型能力的失效

    The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents

    [https://arxiv.org/abs/2609.01852](https://arxiv.org/abs/2609.01852)

    该论文提出并量化了“记忆信任差距”现象：持久记忆智能体会过度信任（而非混淆于）过期的存储事实并覆盖权威证据，且这种失效受模型能力门控——规模越大的模型在过期记忆被伪装成最新信息时崩溃反而越严重。

    

    持久记忆为个性化智能体提供支持，但一条过期的存储事实可能在毫无警告的情况下覆盖当前的权威证据。我们研究随着模型能力的变化，这种危害何时开始显现。我们评估了一个冻结的、闭集的、按动作评分的基准测试，该基准包含两个测试套件，分别代表“无记忆”的两种不同含义（一个是“收益”套件，在没有存储事实的情况下无法解决；另一个是“安全”套件，其中权威工具始终持有正确值），并在同一系列不同规模的模型（Qwen3 0.6/1.7/4/8B）上进行测试。我们发现“记忆信任差距”反映的是过度信任而非混淆。在收益套件中，所有规模的模型在0.92-1.00的情况下都会使用过期值作答。在安全套件中，陷阱条件下低于无记忆基线的危害（Δ_mem）受模型能力门控，一旦过期笔记被伪装成最新内容，规模更大的模型反而崩溃得最严重。在一个2×2×2×2的因子实验中，探究了哪个特征触发了过度信任……（原文摘要在此处截断）

    arXiv:2609.01852v1 Announce Type: new  Abstract: Persistent memory supports personalized agents, but a stale stored fact can override current authoritative evidence without warning. We study when this harm begins as model capability changes. We evaluate a frozen, closed-set, action-scored benchmark with 2 suites that represent 2 different meanings of "no memory" (a Benefit suite, unsolvable without the stored fact, and a Safety suite, in which an authoritative tool always holds the correct value), on a same-family model-size series (Qwen3 0.6/1.7/4/8B). The Memory Trust Gap reflects over-trust rather than confusion. In the Benefit suite, models answer with the stale value 0.92-1.00 of the time at every scale. In the Safety suite, harm below the no-memory baseline under the trap conditions ($\Delta_{\mathrm{mem}}$) is capability-gated, with the larger models collapsing most once a stale note is made to look current. In a $2\times2\times2\times2$ factorial, which feature triggers over-tr
    
[^326]: 防护措施起作用了，但LLM系统因此更安全了吗？

    The Safeguard Worked. Is the LLM System Safer?

    [https://arxiv.org/abs/2609.00519](https://arxiv.org/abs/2609.00519)

    论文指出防护措施的标准评估指标（拒绝率、攻击成功率等）只能衡量其局部表现，无法回答部署层面“LLM系统是否真正更安全”的问题，并揭示了证据的不对称性：一次成功的有害攻击即可证明危害仍存在，而要证明系统安全则需要超越防护自身数据的系统性证据。

    

    部署的LLM服务中的防护措施通常通过拒绝率、攻击成功率和政策违规率来进行评估。这些比率刻画的是防护控制在被测试请求上的表现。然而部署层面需要回答的是一个不同的问题：对于一个不断调整攻击手段或寻找其他入侵途径的攻击者，该服务在多大程度上仍会为其有害任务提供帮助。我们确定了每项已报告的结果对这一问题的实际含义，从而能够在统一的部署标准下比较不同防护系列的研究结果。证据要求呈现出强烈的不对称性：只要有一次从已部署服务中成功获取有害帮助的攻击，就足以证明此类帮助仍然存在，而且这类攻击在编码记录中反复出现。而要证明所剩有害帮助很少，则不能仅凭防护措施自身的评估数字，还需要证据表明在防护措施完成其局部功能之后，系统的其他部分仍然允许什么。

    arXiv:2609.00519v1 Announce Type: cross  Abstract: Safeguards in deployed LLM services are evaluated by refusal, attack success, and policy violation rates. Those rates characterize how a control performed on the requests it was tested on. A deployment has to answer a different question: how much help with harmful tasks the service still gives an attacker who keeps adapting or finds another way in. We determine what each reported result implies for that question, allowing results from different safeguard families to be compared under one deployment criterion. The evidence requirements are strongly asymmetric. One attack that obtains harmful help from the deployed service suffices to establish that such help remains, and such attacks appear repeatedly in the coded record. Establishing that little remains cannot follow from the safeguard's own numbers alone; it also requires evidence about what the surrounding system still allows after the safeguard performs its local function. Such evid
    
[^327]: 阿拉伯语安全对齐作为选择性拒绝：SFT、DPO 与护栏校准的实证研究

    Arabic Safety Alignment as Selective Refusal: An Empirical Study of SFT, DPO, and Guard Calibration

    [https://arxiv.org/abs/2608.29378](https://arxiv.org/abs/2608.29378)

    该研究提出用良性拒绝率 B 和有害提示拒绝率 H 双指标评估阿拉伯语大模型安全对齐，发现仅拒绝式 SFT 会坍缩为全面拒绝，而精选混合 SFT 配置能在低良性拒绝率下实现 90% 以上的有害提示拒绝，且 DPO 与推理护栏的效果因模型而异、并非统一提升。

    

    阿拉伯语大语言模型必须拒绝有害提示，同时不过度拒绝良性或敏感提示，然而单一的拒绝率指标掩盖了这一权衡。我们采用良性拒绝率 B 和有害提示拒绝率 H 来评估该权衡，其中 H 衡量的是拒绝行为而非对有害请求的顺从。在五个支持阿拉伯语的模型上，基于完整人工撰写的 AraSafe 数据集进行的 130 次运行中，仅拒绝式监督微调（SFT）会坍缩为全面拒绝，而精选的混合 SFT 配置能够在 B = 14% 至 23% 的条件下达到 H = 90% 至 93%；四个精选配置在全部三次运行中均超过 H = 90% 的目标，而 Fanar 在三次运行中有两次达标。直接偏好优化（DPO）和推理时护栏对不同模型的 B 和 H 的影响各不相同，而非作为统一的改进手段。在一项盲评的 300 条回复审计中，标注者二元拒绝判断的一致性为 89.0%（kappa = 0.78）；Qwen3Guard 和 Aya Expanse 32B 分别达到 88.7% 和 91.0% 的准确率，而

    arXiv:2608.29378v1 Announce Type: cross  Abstract: Arabic large language models must refuse harmful prompts without over-refusing benign or sensitive prompts, yet a single refusal rate hides this trade-off. We evaluate it using benign refusal B and harmful-prompt refusal H, where H measures refusal rather than harmful compliance. Across five Arabic-capable models and 130 runs on the full human-written AraSafe set, refusal-only supervised fine-tuning (SFT) collapses toward blanket refusal, whereas selected mixed-SFT configurations reach H = 90% to 93% at B = 14% to 23%; four selected configurations exceed the H = 90% target in all three runs, while Fanar does so in two of three. Direct Preference Optimization (DPO) and inference guards change B and H differently across models rather than acting as uniform upgrades. In a blinded 300-response audit, annotator binary-refusal agreement is 89.0% (kappa = 0.78); Qwen3Guard and Aya Expanse 32B reach 88.7% and 91.0% accuracy, respectively, with
    
[^328]: 基于量子信号处理的表示学习

    Representation Learning with Quantum Signal Processing

    [https://arxiv.org/abs/2608.28828](https://arxiv.org/abs/2608.28828)

    该论文将量子信号处理（QSP）确立为表示学习的可解量子模型，精确计算了其量子神经正切核的统计特性，并证明了稀疏数据下完整非线性梯度流收敛于可积标量流以及普遍成立的有限深度速度极限。

    

    表示学习始于训练改变了定义数据间相似性的特征之时，而冻结核模型只能对固定几何进行重新加权。我们建立了量子信号处理（QSP）作为表示学习机制的可解量子模型。在任意深度下，我们计算了其量子神经正切核的精确均值与方差，揭示了一种依赖于输入的角几何结构，即使在底层幺正算子趋近Haar随机性时，其对角线仍保持非自平均特性。我们还为完整的非线性梯度流证明了稀疏数据保证，无需冻结或对核进行系综平均：实际动力学收敛于一个具有时间依赖核闭包的可积标量流，并给出显式收敛时间。有限深度速度极限对每个数据集和轨迹均成立。在更高的数据密度下，数值结果表明存在超越标量和冻结核描述的耦合演化。

    arXiv:2608.28828v1 Announce Type: cross  Abstract: Representation learning begins when training changes the features that define similarity between data. A frozen-kernel model only reweights a fixed geometry. We establish quantum signal processing (QSP) as a solvable quantum model of the representation-learning regime. At arbitrary depth, we compute the exact mean and variance of its quantum neural tangent kernel, revealing an input-dependent angular geometry whose diagonal remains non-self-averaging even when the underlying unitary approaches Haar randomness. We also prove a sparse-data guarantee for the full nonlinear gradient flow without freezing or ensemble-averaging the kernel: the realized dynamics converges to an integrable scalar flow with a time-dependent kernel closure and explicit convergence times. A finite-depth speed limit holds for every data set and trajectory. At higher data density, numerical results show coupled evolution beyond both the scalar and frozen-kernel des
    
[^329]: 迈向Boltz-1稀疏自编码器的高效自动可解释性研究

    Towards Efficient Auto-Interpretability of Boltz-1 Sparse Autoencoders

    [https://arxiv.org/abs/2608.27754](https://arxiv.org/abs/2608.27754)

    该论文提出了一套针对Boltz-1稀疏自编码器的自动化可解释性流水线，通过跨种子字典稳定性优先排序、入侵者检测任务和可证伪的计算机预测测试三项创新，实现了对潜在特征的高效自动解释与筛选。

    

    稀疏自编码器（SAE）可以通过解释生物学及其他领域AI模型的超人能力，将这些模型转化为科学发现的引擎。我们提出，一个潜在特征是否有用取决于三个问题的答案：它是否连贯一致、它是否可以被描述、以及该描述是否具有预测能力。这些问题在以往研究中经常被混为一谈。我们将它们整合成一个统一的流水线，并报告了若干实用创新。首先，跨种子字典稳定性用于优先确定哪些潜在特征值得投入资源进行研究。其次，入侵者检测任务用于检验潜在特征的激活样本是否共享可识别的模式。第三，通过一个独立步骤提出候选的生物学描述，并将其转化为可在计算机上（in silico）测试的可证伪预测。将该系统部署于Boltz-1的Pairformer主干上后，稳定性优先化机制能够以约少4.4倍（的资源）找到可解释的潜在特征。

    arXiv:2608.27754v2 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) could turn AI models in Biology and other fields into engines of scientific discovery by explaining the superhuman capabilities of those models. We postulate that a latent is useful if we know three things: whether it is coherent, whether it can be described, and whether that description has predictive power. These questions are routinely conflated. We assemble them into a single pipeline and report some practical innovations. First, cross-seed dictionary stability prioritises which latents are worth spending resources to investigate. Second, an intruder-detection task asks whether a latent's activating examples share a recognizable pattern. Third, a separate pass proposes a candidate biological description which we convert into falsifiable predictions which can be tested in silico. Deployed on the Boltz-1 Pairformer trunk, stability prioritisation finds interpretable latents using about 4.4 times few
    
[^330]: 边际覆盖信用减少并行状态熵优化中的冗余探索

    Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization

    [https://arxiv.org/abs/2608.27507](https://arxiv.org/abs/2608.27507)

    本文提出MCC-PGPSE方法，通过结合留一策略覆盖与状态所有者专业化的边际覆盖信用来重新分配辅助内在奖励，从而减少并行状态熵优化中的冗余探索并促进互补性状态覆盖。

    

    并行状态熵最大化的策略梯度方法（PGPSE）通过在相同环境的多个副本中训练独立参数化的策略来扩展状态空间覆盖。然而，其汇总的团队熵分数仅衡量集体探索，无法识别对非冗余覆盖有贡献的策略。我们提出了针对PGPSE的边际覆盖信用方法（MCC-PGPSE），该方法将留一策略覆盖与状态所有者专业化相结合，以估计特定策略的信用。MCC-PGPSE保留了PGPSE的汇总目标，并根据这些信用重新分配非负的辅助内在奖励，同时不改变奖励的总质量。这种重新分配旨在抑制冗余访问并促进互补性覆盖。我们在受控环境、七个公开的离散状态基准以及原始PGPSE协议中具有代表性的Room和Maze设置上对MCC-PGPSE进行了评估。在……

    arXiv:2608.27507v1 Announce Type: new  Abstract: Policy Gradient for Parallel State Entropy maximization (PGPSE) expands state-space coverage by training independently parameterized policies in replicated copies of the same environment. However, its pooled team-entropy score measures only collective exploration and cannot identify policies that contribute non-redundant coverage. We introduce Marginal Coverage Credit for PGPSE (MCC-PGPSE), which combines leave-one-policy-out coverage with state-owner specialization to estimate policy-specific credit. MCC-PGPSE preserves PGPSE's pooled objective and redistributes non-negative auxiliary intrinsic rewards according to these credits without changing their total mass. This redistribution is designed to discourage redundant visitation and promote complementary coverage. We evaluated MCC-PGPSE in controlled environments, seven public discrete-state benchmarks, and representative Room and Maze settings from the original PGPSE protocol. Across a
    
[^331]: 多粒度上下文增强的多模态知识图谱检索增强生成

    Multi-Granularity Context-Enhanced RAG over Multimodal Knowledge Graphs

    [https://arxiv.org/abs/2608.25986](https://arxiv.org/abs/2608.25986)

    本文提出构建上下文增强多模态知识图谱（CEMMKG）的新框架，通过在视觉信息提取和多模态知识融合过程中充分利用文本上下文，弥合图像与文本之间的语义鸿沟，从而提升多模态GraphRAG的性能。

    

    基于知识图谱（KG）的检索增强生成（RAG）利用结构化知识为大型语言模型（LLMs）提供高质量的外部信息。在这些工作的基础上，近期研究探索了将多模态知识图谱（MMKGs）作为GraphRAG的知识库，使其能够整合跨多种模态的知识，从而进一步提升性能。然而，现有的基于MMKG的RAG方法通常遵循一个共同的流程：不同模态在很大程度上先被独立处理，然后再进行融合。因此，在视觉信息提取和后续的多模态知识融合过程中，文本上下文仅被有限地利用，这在图像和文本之间造成了语义鸿沟，限制了多模态GraphRAG的性能。为解决这一问题，我们提出了一种新颖的框架，用于构建上下文增强的多模态知识图谱（CEMMKG），以更好地支持多模

    arXiv:2608.25986v2 Announce Type: replace  Abstract: Knowledge graph (KG)-based retrieval-augmented generation (RAG) leverages structured knowledge to provide large language models (LLMs) with high-quality external information. Building on these works, recent studies have explored multimodal knowledge graphs (MMKGs) as knowledge bases for GraphRAG. This enables GraphRAG to integrate knowledge across multiple modalities, thereby further enhancing its performance. However, existing MMKG-based RAG methods generally follow a common pipeline in which different modalities are largely processed independently before being fused. As a result, textual context is only used to a limited extent during visual information extraction and subsequent multimodal knowledge fusion. This brings a semantic gap between images and text which limits the multimodal GraphRAG performance. To address this issue, we propose a novel framework for constructing a Context-Enhanced MMKG (CEMMKG) to better support multimo
    
[^332]: 为什么基于机器学习的咳嗽模型无法泛化：针对结核病筛查的系统性跨数据集评估

    Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening

    [https://arxiv.org/abs/2608.25846](https://arxiv.org/abs/2608.25846)

    本研究通过系统跨数据集评估发现，基于咳嗽的机器学习结核病分类器无法泛化，其性能受录音设备和数据集特异性影响，而非疾病本身，提示需谨慎对待此类模型的实际应用。

    

    arXiv:2608.25846v1 公告类型：交叉 摘要：咳嗽声学特征对于非侵入性结核病（TB）筛查具有潜力，但机器学习（ML）模型捕捉的是疾病相关声学特征还是数据采集伪影，这一问题仍未解决。我们在三个独立数据集上评估了经典机器学习和深度学习（DL）咳嗽基结核病分类器的跨数据集泛化能力。尽管数据集内性能中等（ROC-AUC最高达$0.755 \pm 0.056$），但两种流程均无法泛化，外部性能经常低于0.6，表明可能存在数据局限性。我们进一步观察到，音频表征按录音设备和数据集组织，而非按结核病状态组织；预测的结核病概率在CODA中追踪国家层面的患病率；设备不匹配会降低迁移性能，而设备多样化的训练则能改善迁移。此外，临床变量基线模型泛化更一致（ROC-AUC $0.655 - 0.711$），表明采集特异性因素占主导。

    arXiv:2608.25846v1 Announce Type: cross  Abstract: Cough acoustics are promising for non-invasive tuberculosis (TB) screening, yet whether machine learning (ML) models capture disease-related acoustics or artifacts of data collection remains unresolved. We evaluated the cross-dataset generalizability of classical ML and deep learning (DL) cough-based TB classifiers across three independent datasets. Despite moderate within-dataset performance (ROC-AUC up to $0.755 \pm 0.056$), both pipelines fail to generalize, with external performance frequently below 0.6, indicating a possible limitation of the data. We further observed audio representations are organized by recording device and dataset rather than TB status, predicted TB probability tracks country-level prevalence in CODA, and device mismatch degrades transfer while device-diverse training improves it. Additionally, a clinical-variable baseline generalizes more consistently (ROC-AUC $0.655 - 0.711$), indicating acquisition-specific
    
[^333]: 采样运气伪装成分配收益：审计神经组合优化中的测试时预算分配

    Sampling Luck Masquerades as Allocation Gain: Auditing Test-Time Budget Allocation for Neural Combinatorial Optimization

    [https://arxiv.org/abs/2608.13087](https://arxiv.org/abs/2608.13087)

    本文发现神经组合优化中测试时预算分配的报告增益主要源于采样运气偏差，样本外验证表明非均匀分配并无实际收益。

    

    arXiv:2608.13087v1 公告类型：交叉 摘要：神经组合优化（NCO）求解器报告每个实例中多个采样解的最佳结果，且按惯例，每个实例的采样数量相同。固定总预算的非均匀分配是否能带来收益，此前尚未被测量。我们对此进行了测量，并对测量本身进行了审计。首先，在分布内工作负载上，分配提升空间无法被检测到。在均匀TSP-100上，跨三个预训练求解器（POMO、AM、SymNCO），基于相同存储样本计算和评估的oracle分配报告了2.2-2.6%的增益，置信区间排除零；但在样本外测量时，相同增益与零无法区分（0.457、0.015、-0.512百分比）。按照惯例的样本内程序，所有三个求解器都会支持一个已发表的2%级别增益，但该增益并不存在。我们将此偏差与实例级零假设进行校准，在该假设中真实增益按构造为零；在范围内...

    arXiv:2608.13087v1 Announce Type: cross  Abstract: Neural combinatorial optimization (NCO) solvers report the best of many sampled solutions per instance, and the sample count is, by convention, identical for every instance. Whether a non-uniform allocation of a fixed total budget would buy anything has not been measured. We measure it, and we audit the measurement itself.   First, on in-distribution workloads the allocation headroom is not detectable. Across three pretrained solvers (POMO, AM, SymNCO) on uniform TSP-100, an oracle allocation computed and evaluated on the same stored samples reports a 2.2-2.6% gain with intervals excluding zero; measured out of sample the same gain is indistinguishable from zero (0.457, 0.015, -0.512 percent). Following the customary in-sample procedure, all three solvers would have supported a published 2%-level gain that does not exist. We calibrate this bias against an instance-wise null in which the true gain is zero by construction; over the range
    
[^334]: 查询时机导致大语言模型与人类之间产生相反的位置偏差

    Query Timing Produces Opposite Positional Biases Between LLMs and Humans

    [https://arxiv.org/abs/2608.12387](https://arxiv.org/abs/2608.12387)

    本研究揭示了LLMs在位置偏差上表现出与人类相反的模式，且新模型中的偏差更为显著，表明查询时机对模型评估机制有重要影响。

    

    位置偏差，如近因效应和首因效应，已在大型语言模型（LLMs）中被记录，然而这些模型进行评估的潜在机制仍知之甚少。在人类对证据的回应中，首因和近因偏差均有观察到，但近期研究表明，听者更新其信念的时间点——是在证据呈现过程中还是仅在结束时——会影响这些效应的存在。我们研究了类似现象是否在LLMs中成立，发现其与人类行为存在分歧。这些偏差在较新模型中相较于早期模型更为加剧。

    arXiv:2608.12387v1 Announce Type: cross  Abstract: Positional biases such as recency and primacy effects have been documented in large language models (LLMs), yet the underlying mechanism by which these models make their evaluations remains poorly understood. Both primacy and recency biases have been observed in human judgments in response to evidence, but recent work suggest that \emph{when} the listener updates their beliefs -- during the presentation of evidence or only at the end -- influences the presence of such effects. We investigate whether a similar phenomenon holds for LLMs, finding divergence from human behavior. These biases are more exacerbated in newer models compared to their predecessors.
    
[^335]: 一个冻结的模拟器是不够的：多智能体强化学习中的模拟器崩溃

    One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL

    [https://arxiv.org/abs/2608.12253](https://arxiv.org/abs/2608.12253)

    该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。

    

    用于人机交互的多智能体强化学习通常依赖单一大型语言模型来模拟用户行为。我们表明这种方法系统性无法泛化，并将失败追溯到模拟器崩溃：由于模拟器语言模型存在模式崩溃，针对其训练的策略会过度拟合到利用模拟器主导模式的狭窄策略，而这种策略在未见过的模拟器和真实用户中迁移效果不佳。我们在理论上形式化了这种崩溃，并提出了两种互补的解决方案，一种在推理时，一种在训练时。推理时的解决方案，即言语化采样，通过从言语化响应分布中采样来拓宽模拟器的行为，减少模式崩溃。训练时的解决方案，即协同训练，与一组可训练的模拟器种群联合优化策略，防止其过度拟合任何单一模拟器的模式。我们进行了验证...

    arXiv:2608.12253v1 Announce Type: cross  Abstract: Multi-agent reinforcement learning for human-AI interaction typically relies on a single large language model to simulate user behavior. We show that this approach systematically fails to generalize, and trace the failure to simulator collapse: because the simulator LLM is mode-collapsed, an LLM policy trained against it overfits to narrow strategies that exploit the simulator's dominant mode, and such a policy transfers poorly to unseen simulators and real users. We formalize this collapse theoretically and propose two complementary solutions, one at inference time and one at training time. The inference-time solution, Verbalized Sampling, broadens the simulator's behavior by sampling from a verbalized response distribution, reducing mode collapse. The training-time solution, Co-Training, jointly optimizes the policy against a population of trainable simulators, preventing it from overfitting to any single simulator's mode. We validat
    
[^336]: 基于可观测语义图像接口与层级生成器证据对齐的生成式语义分割

    Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment

    [https://arxiv.org/abs/2608.11537](https://arxiv.org/abs/2608.11537)

    提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。

    

    生成式语义分割将结构化预测呈现为图像，但直接的颜色解码容易受到颜色漂移和边界混合的影响，而预测独立输出分布的潜在特征解码器可能将渲染图像降级为中间可视化。我们提出了Semantic Prism，一个具有确定性推理的条件语义图像生成与细化框架。一个基于扩散蒸馏的单步生成器渲染语义RGB图像；渲染颜色到固定类颜色码本的逐像素距离定义了一个显式的概率接口。层级生成器证据对齐在空间上对齐多层级生成器特征，并使用零初始化输出投影在接口对数空间中预测加性残差，保留图像定义的接口作为最终分布的参考。该接口和细化分布进一步支持Conte（后续内容）。

    arXiv:2608.11537v1 Announce Type: cross  Abstract: Generative semantic segmentation exposes structured predictions as images, but direct color decoding is susceptible to color drift and boundary mixing, whereas latent-feature decoders that predict a separate output distribution may relegate the rendered image to an intermediate visualization. We present Semantic Prism, a conditional semantic-image generation-and-refinement framework with deterministic inference. A diffusion-distilled one-step generator renders a semantic RGB image; per-pixel distances from the rendered colors to a fixed class-color codebook define an explicit probabilistic interface. Hierarchical Generator Evidence Alignment spatially aligns multi-level generator features and uses a zero-initialized output projection to predict an additive residual in the interface logit space, retaining the image-defined interface as the reference for the final distribution. The interface and refined distributions further enable Conte
    
[^337]: 持久递归世界实现软件自主演化

    Persistent Recursive Worlds Enable Autonomous Software Evolution

    [https://arxiv.org/abs/2608.10450](https://arxiv.org/abs/2608.10450)

    本文提出EvoX Genesis框架，通过将软件表示为持久递归世界，使项目持久化而代理有限寿命，实现了无需人工干预的自主软件演化，并成功构建了大型编译器。

    

    复杂软件系统的发展时间尺度超过了任何单个编码代理的寿命。大多数代理软件系统通过持久会话、记忆、管理器或共享上下文来保持连续性。我们引入了EvoX Genesis（以下简称Genesis），它反而使软件项目持久化，同时允许本地代理保持有限寿命。Genesis将软件表示为持久递归世界：每个本地世界由已接受的版本和仓库路径定位，有限寿命的代理提出局部更改，递归委托在路径间移动工作，只有被接受的后果推进持久版本历史。我们评估了这种组织在形成、延续和再开发中的表现。从一个没有编译器实现的仓库开始，Genesis使用DeepSeek V4 Flash构建了一个基于Rust的C编译器，约25万行跟踪代码；运行持续超过120小时，arch...

    arXiv:2608.10450v2 Announce Type: replace-cross  Abstract: Complex software systems develop over timescales that exceed the lifespan of any individual coding agent. Most agentic software systems preserve continuity through persistent sessions, memories, managers or shared context. We introduce EvoX Genesis (hereafter, Genesis), which instead makes the software project persistent while allowing local agents to remain finite-lived. Genesis represents software as a persistent recursive world: each local world is situated by an accepted version and a repository path, finite-lived agents propose local changes, recursive delegation moves work across paths, and only accepted consequences advance the persistent version history. We evaluate this organization across formation, continuation and redevelopment. Starting from a repository with no compiler implementation, Genesis used DeepSeek V4 Flash to build a Rust-based C compiler with about 250k tracked lines; the run lasted over 120 hours, arch
    
[^338]: 探索、绘图、记忆、决策：评估VLM在安全关键场景中就绪度的基准测试

    Explore, Map, Remember, Decide: A Benchmark for VLM Readiness in Safety-Critical Scenarios

    [https://arxiv.org/abs/2608.08077](https://arxiv.org/abs/2608.08077)

    该论文提出EMRD评估框架，将VLM的空间推理能力解耦为探索、绘图、记忆、决策四个阶段并以18项指标评分，揭示了现有模型在安全关键场景下存在严重的探索遗漏与感知缺陷。

    

    视觉语言模型（VLM）常被部署为机器人的具身控制器，但它们在探索环境、维护内部地图以及基于长时序序列轨迹做出安全关键决策方面的能力如何，目前仍不清楚。在这项工作中，我们提出了EMRD——一个将空间推理解耦为四个阶段的评估框架：探索（智能体观察到什么）、绘图（它记录了什么以及在哪里）、记忆（它保留了什么）和决策（其最终选择是否有依据），并针对每个阶段设置了18项评分指标。我们使用EMRD在16栋建筑中、于标称条件以及三种环境压力（黑暗、烟雾和运动噪声）下评估了十一个开放权重与闭源权重的VLM，其中具身VLM需要探索未知房间、绘制观察到的物体地图，并选择最安全的集合地点。开放权重模型在95%的情节中存在未探索的房间，且物体检测率最高仅为51%；闭源权重的模型表……（摘要在此处截断）

    arXiv:2608.08077v2 Announce Type: replace  Abstract: Vision-Language Models (VLMs) are often deployed as embodied controllers for robotics, yet it remains unclear how well they can explore an environment, maintain an internal map, and ground safety-critical decisions in long-horizon sequential trajectories. In this work, we introduce EMRD, an evaluation framework that decouples spatial reasoning into four stages: Explore (what the agent observes), Map (what it records and where), Remember (what it retains), and Decide (whether its final choice is justified), scored across 18 stage-specific metrics. We use EMRD to evaluate eleven open- and closed-weight VLMs across 16 buildings under nominal conditions and three environmental stresses (darkness, smoke, and motion noise), where an embodied VLM explores unknown rooms, maps observed objects, and selects the safest gathering place. Open-weight models leave rooms unvisited in 95% of episodes and detect at most 51% of objects; closed-weight m
    
[^339]: 面向社会模拟的语言模型角色引导

    Role Steering of Language Models for Social Simulations

    [https://arxiv.org/abs/2608.00023](https://arxiv.org/abs/2608.00023)

    该论文提出了一种激活引导筛选工作流和新方法 CastVectors，用于为语言模型智能体注入可验证的角色条件化行为，在社会模拟中实现了显著更高的角色档案对齐度（63.2 对比 41.1），同时保持了词汇多样性。

    

    由语言模型智能体构建的社会模拟需要在将智能体放入模拟人群之前能够被验证的角色条件化行为。我们提出了一种面向角色条件化智能体的激活引导筛选工作流：定义角色档案、提取角色特定方向、扫描四个引导系数、评估角色档案对齐度，并对每个候选配置进行通过或标记。在 OLMo-3-7B-Instruct 上，我们将该工作流应用于包含 275 个角色的混合角色清单、228 个与角色无关的问题、由 GPT-4.1-mini 提示生成的角色参考以及 GPT-4.1-mini 评判者。我们还提出了 CastVectors，其在评判的角色档案对齐度上优于此前人格向量工作中的助手轴方向控制方法，在测试网格上的平均总分为 63.2 对比 41.1。CastVectors 还保持了较高的词汇多样性，而控制方法在较大系数下词汇多样性急剧下降。角色级筛选是……（摘要原文在此处截断）

    arXiv:2608.00023v3 Announce Type: replace-cross  Abstract: Social simulations built from language-model agents need role-conditioned behavior that can be checked before agents are placed into a simulated population. We introduce an activation-steering screening workflow for role-conditioned agents: define a role profile, extract a role-specific direction, sweep four steering coefficients, evaluate role-profile alignment, and pass or flag each candidate configuration. On OLMo-3-7B-Instruct, we apply the workflow to a mixed 275-role inventory with 228 role-agnostic questions, GPT-4.1-mini prompted role references, and GPT-4.1-mini judges. We also introduce CastVectors, which receive higher judged role-profile alignment than an assistant-axis directional control from prior persona-vector work, with mean overall scores of 63.2 versus 41.1 across the tested grid. They also preserve high lexical diversity, while the control drops sharply at larger coefficients. The role-level screen is the m
    
[^340]: RecHarness：一种用于自演化推荐系统的多臂老虎机路由智能体框架

    RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems

    [https://arxiv.org/abs/2607.29241](https://arxiv.org/abs/2607.29241)

    提出RecHarness框架，通过将优化过程解耦为“多臂老虎机路由器选择修改方向+LLM生成具体优化假设与代码修改”两个步骤，并借助跳跃盆地机制在局部优化停滞时触发结构性跳跃，实现推荐模型的自演化自动优化。

    

    优化推荐模型仍然严重依赖于对架构、目标和训练策略的人工迭代。虽然基于大语言模型（LLM）的智能体可以自动化这一试错过程，但依赖LLM同时选择修改方向和制定具体假设，在有限的试验预算下可能导致不稳定的搜索。受上述挑战的启发，我们提出了RecHarness，一个用于自动化推荐模型优化的多臂老虎机路由智能体框架。RecHarness将优化过程解耦为两个步骤：多臂老虎机路由器根据历史验证反馈选择下一个修改方向，而LLM则在所选方向内生成具体的优化假设和可执行的代码修改。为了维持长期探索，RecHarness使用跳跃盆地（jump-basin）机制，在局部编辑陷入停滞时激活结构跳跃臂。在多个推荐任务、数据集和……（摘要截断）

    arXiv:2607.29241v2 Announce Type: replace-cross  Abstract: Optimizing recommender models remains heavily reliant on manual iteration over architectures, objectives, and training strategies. While LLM-based agents can automate this trial-and-error process, relying on LLMs to both select modification directions and formulate concrete hypotheses can lead to unstable search under limited trial budgets. Inspired by the above challenge, we propose RecHarness, a Bandit-Routed Agentic Harness for automated recommender model optimization. RecHarness decouples the optimization process into two steps: a bandit router selects the next modification direction according to historical validation feedback, while the LLM generates a concrete optimization hypothesis and executable code edit within the selected direction. To sustain long-horizon exploration, RecHarness uses a jump-basin mechanism to activate a structural-jump arm when local edits stagnate. Across multiple recommendation tasks, datasets, a
    
[^341]: 大型语言模型滥用防护措施依赖不可信输入：一种广泛存在的高风险漏洞

    LLM Misuse Safeguards Rely on Untrusted Input: A Widespread, High-Risk Vulnerability

    [https://arxiv.org/abs/2607.27951](https://arxiv.org/abs/2607.27951)

    大型语言模型的滥用防护措施因依赖用户自述的身份或用途等不可信输入来决定是否放行答案，存在被攻击者通过伪造声明绕过的高风险漏洞，且该漏洞普遍存在于12个顶级会议129篇论文所提出的全部防御方案中。

    

    大型语言模型如今在网络安全和生物学等领域为专业人士提供帮助，然而同样的答案也会为攻击者所用。部署方依赖滥用防护措施来防止此类答案落入攻击者手中。我们发现这些防护措施存在一个共同的高风险漏洞：它们根据用户所写的内容来决定是否发布答案，包括用户声称的专业身份或用途，而任何人都可以编写此类声明。来自模型厂商和安全公司的威胁报告显示，伪造专业身份或用途的攻击者已经在利用这一漏洞造成严重危害。该漏洞在当前最先进的技术中也普遍存在：来自12个顶级会议最新一届的129篇论文中所提出的每一项防御措施都存在此漏洞。防护措施研究的研究对象出了错：它将危害视为请求文本本身的属性，尽管相同的文本可能来自专业人士，也可能来自攻击者。目前的防护……（原文摘要在此处截断）

    arXiv:2607.27951v2 Announce Type: replace-cross  Abstract: Large language models now help professionals in fields such as cybersecurity and biology, yet the same answers would help an attacker. Deployers rely on misuse safeguards to keep such answers from attackers. We find that these safeguards share a high-risk vulnerability: they decide whether to release an answer from what users write, including the professional identity or purpose they claim, and anyone can write such a claim. Threat reports from model vendors and security firms show that attackers who fabricate a professional identity or purpose already exploit this vulnerability to cause serious harm. The vulnerability is also widespread across the state of the art: every defense proposed in 129 papers from the latest editions of 12 top venues has it. Safeguard research studies the wrong object: it treats harm as a property of the text of a request, although the same text can come from a professional or an attacker. Current saf
    
[^342]: PIE-APT：基于增量推理的时序动态知识图谱溯因规划

    PIE-APT: Abductive Planning over Temporal Dynamic Knowledge Graphs via Incremental Reasoning

    [https://arxiv.org/abs/2607.27287](https://arxiv.org/abs/2607.27287)

    该论文提出PIE-APT统一框架，通过将状态转移建模为对演绎封闭描述逻辑理论的非单调更新并利用增量推理机，在SROIQ描述逻辑上保持可判定性并原生绕过分支问题，同时借助增量直接推导溯因避免最小命中集的组合枚举，从而实现时序动态知识图谱上的高效溯因规划。

    

    在信息不完整的开放世界环境中，对时序动态知识图谱（TDKGs）进行规划带来了理论上的挑战。现有的动作形式化方法常常面临可判定性问题和分支问题，而结构化溯因推理则需要庞大的组合搜索空间。我们提出了一个包含两个模块的统一框架——PIE-Abducer（增量式直接推导溯因）和PIE-APT（面向时序动态知识图谱的溯因规划）——该框架原生运行于表达力强大的SROIQ描述逻辑之上。我们将状态转移建模为对演绎封闭的描述逻辑理论的非单调更新，并以OWL原生地表示动作。这借助增量推理机来保持可判定性，并从原生层面绕过了分支问题。为应对知识不完整的问题，PIE-Abducer避免了对最小命中集（MHS）进行枚举，它不采用组合搜索，而是将目标的逻辑否定注入一致的、表达力强大的描述逻辑理论中……

    arXiv:2607.27287v2 Announce Type: replace  Abstract: Planning over Temporal Dynamic Knowledge Graphs (TDKGs) presents theoretical challenges in open-world environments with incomplete information. Existing action formalisms often face decidability issues and the Ramification Problem, while structural abduction requires expansive combinatorial search spaces. We introduce a unified framework with two modules--PIE-Abducer (incremental direct-derivation abduction) and PIE-APT (Abductive Planning for TDKGs)--operating natively on the expressive SROIQ Description Logic. Modeling state transitions as non-monotonic updates to deductively closed DL theories, we represent actions natively in OWL. This leverages an incremental reasoner to preserve decidability and natively bypass the Ramification Problem. To address incomplete knowledge, PIE-Abducer circumvents Minimal Hitting Set (MHS) enumeration. Instead of combinatorial search, it injects the logical negation of a goal into a consistent DL br
    
[^343]: 在智能体软件开发中使用第三方API路由器的权衡在哪里？

    Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?

    [https://arxiv.org/abs/2607.23624](https://arxiv.org/abs/2607.23624)

    该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。

    

    第三方API路由器已成为一种常见的中间层，统一了对日益多样化的LLM提供商的访问。在编码智能体工作流中，高自主性操作被广泛采用，因为它减少了交互开销。因此，位于智能体与上游提供商之间的第三方API路由器不可避免地占据了可信路径。它可以检查和修改每一个请求与响应，然而目前没有任何机制来验证提供商的输出与智能体最终执行的仓库级操作之间的一致性。因此，客户端权限机制在实践中可能变得无效。这种控制缺口是否会对软件开发任务产生真实的、难以检测的影响，此前尚未得到实证测量。在本文中，我们对编码智能体中的路由器端注入进行了实证研究，考察了四个隐蔽程度递增的干预级别：响应替换（L1）、响应追加……

    arXiv:2607.23624v3 Announce Type: replace-cross  Abstract: Third-party API routers have become a common layer that unifies access across increasingly diverse LLM providers. In coding-agent workflows, high-autonomy operation is widely adopted because it reduces interaction overhead. As a result, a third-party API router, which sits between the agent and the upstream provider, inevitably occupies the trusted path. It can inspect and modify every request and response, yet no mechanism verifies alignment between the provider's output and the repository-level actions ultimately executed by the agent. Consequently, client-side permission mechanisms may become ineffective in practice. Whether this control gap produces real, hard-to-detect effects on software development tasks remains empirically unmeasured. In this paper, we conduct an empirical study of router-side injection in coding agents, examining four intervention levels of increasing subtlety: Response Substitution (L1), Response Appe
    
[^344]: 测试时覆盖率：面向部署感知学习的测试条件化数据筛选

    Test-Time Coverage: Test-Conditioned Data Curation for Deployment-Aware Learning

    [https://arxiv.org/abs/2607.22697](https://arxiv.org/abs/2607.22697)

    该论文提出TTCov方法，在训练前利用测试侧信息构建由LLM原子命题及其频率组成的知识图集，将部署条件化的数据筛选分解为覆盖率与分布两部分，从而直接优化训练数据与部署测试分布的匹配度。

    

    已部署的AI系统通常从广泛的候选数据池中进行训练，因此需要针对部署测试分布进行数据筛选。然而，标准的数据筛选方法评估的是训练侧的准则，而非直接优化与部署分布的匹配度。我们提出了TTCov（Test-Time Coverage，测试时覆盖率），这是一种数据层面的测试条件化筛选方法，它在训练之前使用测试侧的信息，而不是在推理时更新模型权重。TTCov将部署条件化的筛选分解为覆盖率与分布两个部分。为了表示覆盖率，它构建了一个任务图集，即一组基于大语言模型（LLM）的原子命题（APs），用于描述与部署相关的概念；这些命题由开放的任务知识作为种子，并通过从无标注部署样本中提取的未匹配原子命题进行扩展。为了表示分布，它以频率来实例化匹配的部署原子命题，从而构建出一个知识图集，将部署分布进行可操作化表示……

    arXiv:2607.22697v2 Announce Type: replace  Abstract: Deployed AI systems are often trained from broad candidate data pools, necessitating data curation towards the deployment test distribution. However, standard data curation methods score training-side criteria rather than directly optimizing deployment match. We introduce TTCov (Test-Time Coverage), a data-level test-conditioned curation method that uses test-side information before training instead of updating model weights at inference. TTCov decomposes deployment-conditioned curation into coverage and distribution. To represent coverage, it builds a task Atlas, a collection of LLM-based atomic propositions (APs) describing deployment-relevant concepts, seeded from open task knowledge and expanded with unmatched APs extracted from unlabeled deployment samples. To represent distribution, it instantiates the matched deployment APs with their frequencies, yielding a Knowledge Atlas (K-Atlas) that operationalizes the deployment distrib
    
[^345]: 随机多梯度 descent 的改进收敛速率——填补差距：由 AI 完成的证明

    Improved Convergence Rates for Stochastic Multi-Gradient Descent which Close the Gap: A Proof by AI

    [https://arxiv.org/abs/2607.18174](https://arxiv.org/abs/2607.18174)

    本文通过利用帕累托平稳性度量的 Lipschitz 连续性（而非 (1/2)-Hölder 连续性），证明了随机多梯度下降（SMG）在常数步长和线性增长批量下可将平方帕累托平稳性度量的收敛速率从此前的 $\tilde{O}(T^{-1/4})$ 提升至 $\tilde{O}(T^{-1})$，且无需有界梯度假设。

    

    对于光滑非凸的随机多目标优化问题，随机多梯度下降（SMG）从随机梯度中计算各目标函数的近似最速公共下降方向。在标准假设下，本文针对 SMG 在平方帕累托平稳性（PS）度量方面建立了一个新的收敛速率：采用常数步长和线性增长的迷你批量，经过 $T$ 次迭代后，算法输出处的期望平方经验 PS 度量为 $\tilde{O}(T^{-1})$。这改进了 Chen 等人（JMLR, 2024）在线性增长批量下得到的 $\tilde{O}(T^{-1/4})$ 的界，且不需要有界梯度假设。其中 $\tilde{O}(\cdot)$ 隐藏了对数因子。收敛速率提升的关键在于利用由多梯度下降算法（MGDA）方向的范数所定义的 PS 度量的 Lipschitz 连续性，而非 MGDA 方向的 (1/2)-Hölder 连续性。

    arXiv:2607.18174v2 Announce Type: cross  Abstract: For smooth nonconvex stochastic multi-objective optimization, stochastic multi-gradient descent (SMG) computes an approximate steepest common descent direction of the objectives from stochastic gradients. Under standard assumptions, this note establishes a new convergence rate for SMG in terms of the squared Pareto-stationarity (PS) measure. With a constant stepsize and linearly growing mini-batches, the expected squared empirical PS measure at the algorithm's output is $\tilde{O}(T^{-1})$ after $T$ iterations. This improves on the $\tilde{O}(T^{-1/4})$ bound obtained by Chen et al. (JMLR, 2024) for linearly growing batches, without requiring bounded gradients. Here, $\tilde{O}(\cdot)$ suppresses logarithmic factors. The key to the rate improvement is to exploit the Lipschitz continuity of the PS measure, defined by the norm of the multi-gradient descent algorithm (MGDA) direction, rather than the $(1/2)$-H\"older continuity of the MGD
    
[^346]: OpenMHC：加速可穿戴基础模型的科学发展

    OpenMHC: Accelerating the Science of Wearable Foundation Models

    [https://arxiv.org/abs/2607.16235](https://arxiv.org/abs/2607.16235)

    该论文发布了OpenMHC——目前面向研究人员开放的最大、最全面的可穿戴健康数据集（涵盖11,894名参与者、超过6700万小时记录、最多19个传感器通道和170个健康行为变量），并配套提供可穿戴基础模型的开源实现与模型权重，以推动可穿戴健康领域的开放科学。

    

    移动和可穿戴设备为持续、被动式的健康监测和主动健康指导提供了前所未有的机会。然而，目前最大的可穿戴数据集均为专有数据，基于这些数据训练的领先可穿戴基础模型很少开放权重或包含可复现的训练代码。为了加速可穿戴健康领域的开放科学，我们推出了OpenMyHeartCounts（OpenMHC），这是面向合格研究人员开放的最大、最全面的可穿戴健康数据集，同时提供近期可穿戴基础模型的开源实现和权重。OpenMHC源自通过My Heart Counts研究应用收集的十余年数据，包含来自11,894名参与者的超过6700万参与者小时的可穿戴记录，涵盖多达19个传感器通道（如步数、心率、睡眠、锻炼）以及多达170个健康、生活方式、情绪和行为变量。此外，我们还建立了一个……（原文摘要在此处截断）

    arXiv:2607.16235v4 Announce Type: replace-cross  Abstract: Mobile and wearable devices offer an unprecedented opportunity for continuous, passive health monitoring and active health coaching. However, the largest wearable datasets are proprietary, and leading wearable foundation models trained on them are rarely open weight or include reproducible training code. To accelerate open science in wearable health, we introduce OpenMyHeartCounts (OpenMHC), the largest and most comprehensive wearable health dataset broadly accessible to qualified researchers, alongside open-source implementations and weights for recent wearable foundation models. Derived from over a decade of data collected through the My Heart Counts study app, OpenMHC comprises over 67M participant-hours of wearable recordings across up to 19 sensor channels (e.g., step count, heart rate, sleep, workouts) and up to 170 health, lifestyle, mood, and behavioral variables from 11,894 participants. Furthermore, we establish a ben
    
[^347]: 将通用车辆模型适配用于跨地形高速模型预测控制（MPC）

    Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains

    [https://arxiv.org/abs/2607.13319](https://arxiv.org/abs/2607.13319)

    OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。

    

    高速越野自动驾驶需要对目标车辆进行精确的闭环控制，同时在多变地形上保持鲁棒性。近期的前向运动动力学（FKD）预测基础模型展现出一条有前景的路径：从通用模型出发，将其专用于目标平台。然而，有效的专业化仍然具有挑战性，因为它通常需要大量真实世界数据，而且适配到某一环境的模型仍可能对特定地形或驾驶模式过拟合。我们提出了OptCar（优化汽车），这是一种弥合通用FKD模型到专用FKD模型差距的方法，在为特定车辆优化性能的同时保留了跨地形泛化能力。OptCar引入了一种基于transformer的FKD架构，利用FiLM将多步预测条件化于单个动力学上下文token，该token概括了近期的状态-动作历史。随后，该方法利用有限的真实世界数据对通用模型进行专业化（摘要在此处截断）。

    arXiv:2607.13319v2 Announce Type: replace-cross  Abstract: High-speed off-road autonomy requires precise closed-loop control for a target vehicle while remaining robust across changing terrains. Recent forward kinodynamic (FKD) prediction foundation models suggest a promising path, starting from a generalist model and specializing it to the target platform. However, effective specialization remains challenging, as it often requires substantial real-world data, and models adapted to one setting can still overfit to specific terrains or driving regimes. We present OptCar (Optimized Car), a recipe for bridging the gap from generalist to specialist FKD models that preserves cross-terrain generalization while optimizing performance for a specific vehicle. OptCar introduces a transformer FKD architecture that uses FiLM to condition multi-step predictions on a single dynamics context token summarizing recent state-action history. It then specializes the generalist model using limited real-wor
    
[^348]: 基于视觉的低功耗边缘平台实时跌倒检测

    Real-time fall detection based on vision for low-power edge platforms

    [https://arxiv.org/abs/2607.12909](https://arxiv.org/abs/2607.12909)

    该论文提出一种基于物理信息的双液态时间常数（LTC）网络架构，将跌倒建模为质心与支撑基面耦合动力系统的稳定性丧失事件，通过可学习的物理交互耦合模块与稳定性流形分类器，实现了低功耗边缘平台上的实时视觉跌倒检测。

    

    跌倒检测对老年人护理和智能监控至关重要；然而，现有的基于视觉的方法主要将其框定为静态姿态分类或离散时间模式匹配，从根本上忽视了人体支撑系统的不稳定动力学特性。本文提出了一种基于物理信息的跌倒检测框架，将跌倒重新定义为耦合动力系统中的稳定性丧失事件。我们引入了一种新颖的双LTC（液态时间常数）架构，包含质心子系统和支持基面子系统，两者均实例化为液态时间常数（LTC）神经网络，通过自适应时间常数连续建模惯性轨迹演化与地面接触调整，实现跌倒运动的物理可解释性。一个可学习的耦合模块模拟两个子系统之间的物理交互，而稳定性流形分类器则在联合潜在空间中运行。

    arXiv:2607.12909v2 Announce Type: replace-cross  Abstract: Falling detection is vital for elderly care and intelligent surveillance; however, prevailing vision-based approaches predominantly frame it as static pose classification or discrete temporal pattern matching, fundamentally overlooking the instability dynamics of the human support system. This paper proposes a physics-informed falling detection framework that recasts falling as a stability-loss event in a coupled dynamical system. We introduce a novel dual-LTC architecture comprising a Center-of-Mass (CoM) subsystem and a Base-of-Support (BoS) subsystem, both instantiated as Liquid Time-Constant (LTC) neural networks to continuously model inertial trajectory evolution and ground-contact adjustment through adaptive time constants, Physical interpretability of falling motion. A learnable coupling module emulates physical interaction between the two subsystems, while a Stability Manifold classifier operates in the joint latent spa
    
[^349]: 强化学习后训练构建组合推理策略

    RL Post-Training Builds Compositional Reasoning Strategies

    [https://arxiv.org/abs/2607.07646](https://arxiv.org/abs/2607.07646)

    强化学习后训练并非只是放大预训练模型已有的基础技能，而是通过“先强化基础规约、再发现顺序与并行组合程序”的分阶段组合机制，将基础技能重组为新的更高层级推理策略，从而解决预训练模型几乎无法解决的问题。

    

    强化学习（RL）后训练仅仅是放大基础模型中已经潜在存在的基础技能，还是能够将基础技能组合成新的更高层级策略？我们在一个完全可观测的重写语法环境中研究这个问题，在该环境中预训练分布是已知的，且每一个生成的重写都可以被审计。Transformer模型首先在基础符号重写链上进行预训练，然后在一个基于Trace的推理任务上进行后训练，仅使用二元的最终答案奖励。强化学习能够解决那些预训练模型即使在更大的采样预算下也极少能解决的保留测试问题，而拒绝采样微调虽然早期有所提升但随后趋于停滞。Trace分析显示，强化学习通过一种分阶段的组合机制重新组织基础能力：它首先强化基础规约，然后发现有效的组合程序，包括将有序的基础收缩链折叠起来的顺序组合，以及并行组合……

    arXiv:2607.07646v2 Announce Type: replace  Abstract: Does RL post-training merely amplify primitive skills already latent in a base model, or can it compose primitive skills into new higher-level strategies? We study this question in a fully observable rewrite-grammar environment where the pretraining distribution is known and every generated rewrite can be audited. A Transformer is pretrained on primitive symbol-rewrite chains and post-trained on a Trace-based reasoning task with only a binary final-answer reward. RL solves held-out problems that remain rarely solved by the pretrained model even under much larger sampling budgets, while rejection fine-tuning improves early but plateaus. Trace analysis shows that RL reorganizes primitive competence through a phased compositional mechanism: it first strengthens primitive reductions, then discovers valid composed procedures. These include sequential compositions, which collapse ordered chains of primitive contractions, and parallel compo
    
[^350]: 统计对手：干净视觉数据集中自然存在的后门样对抗特征

    Statistical Adversaries: Natural Backdoor-like Adversarial Features in Clean Vision Datasets

    [https://arxiv.org/abs/2607.05516](https://arxiv.org/abs/2607.05516)

    本论文发现干净视觉数据集中自然存在的统计信号可作为后门样触发器，揭示了数据集结构和分布导致的模型漏洞，而非恶意插入。

    

    arXiv:2607.05516v2 公告类型：替换-交叉 摘要：针对特定模型的对抗攻击已被广泛研究。我们研究了一种不同的故障模式：视觉数据中自然存在的统计信号，它们可以像后门触发器一样发挥作用，而无需恶意插入。我们将这些信号称为统计对手。我们分析ImageNet，以找到与特定标签强烈相关的模式。然后，我们使用统计控制方法从候选信号中去除随机相关性。最后，我们证明这些信号直接且可预测地改变模型预测。这些统计对手比通用扰动更具针对性，并能跨不同模型架构迁移。这表明某些漏洞是由数据集结构和分布驱动的，而非单一模型的特殊性质。我们得出结论，即使在无投毒的情况下，普通数据集也可能包含可利用的对抗表面，并建议对数据集进行更细致的审查。

    arXiv:2607.05516v2 Announce Type: replace-cross  Abstract: Model-specific adversarial attacks have been extensively studied. We study a different failure mode: naturally occurring statistical signals in vision data that can behave as backdoor-like triggers without being maliciously inserted. We call these signals statistical adversaries. We analyse ImageNet to find patterns that are strongly linked to certain labels. We then use statistical controls to remove random correlations from our candidate signals. Finally, we demonstrate that these signals directly and predictably alter model predictions. These statistical adversaries are more targeted than generic corruptions and transfer across different model architectures. This suggests that some vulnerabilities are driven by dataset structure and distribution rather than a single model's idiosyncrasies. We conclude that ordinary datasets can contain exploitable adversarial surfaces even in the absence of poisoning, and suggest that datase
    
[^351]: 局部化LoRA-MoE：具有自适应路由的低秩专家混合

    Localized LoRA-MoE: Mixture of Low-Rank Experts With Adaptive Routing

    [https://arxiv.org/abs/2607.05114](https://arxiv.org/abs/2607.05114)

    提出Localized LoRA-MoE框架，将局部空间分块与上下文条件自适应专家路由相结合，提供了集中式块级路由和去中心化单元级路由两种架构变体，以解决传统PEFT方法中的梯度干扰和静态拓扑无法适应上下文切换的问题。

    

    参数高效微调（PEFT）方法（如LoRA）通过单一低秩更新来适配大型模型。这类更新容易受到梯度干扰的影响：当训练流中交织着多个操作上下文时，对共享适配器权重的相互冲突的更新会使适配器坍缩成一个缺乏专门化的平均值。虽然近期的方法引入了块级隔离来缓解这一问题，但其拓扑结构是静态的，无法适应上下文切换或局部传感器退化。在这项工作中，我们提出了Localized LoRA-MoE，一个将局部空间分块与上下文条件专家路由相结合的框架。我们提出并评估了两种架构变体：Block-Wise LoRA-MoE（集中式路由），它通过单一的上下文条件门控来调制整个结构网格；以及Cell-Wise LoRA-MoE（去中心化路由），它为结构网格的每个坐标单元分配一个独立的门控。

    arXiv:2607.05114v2 Announce Type: replace-cross  Abstract: Parameter-efficient fine-tuning (PEFT) methods such as LoRA adapt large models through a single low-rank update. Such updates are susceptible to gradient interference: when a training stream interleaves multiple operational contexts, conflicting updates to shared adapter weights collapse the adapter into an unspecialized average. While recent methods introduce block-wise isolation to remedy this issue, their topologies are static and cannot adapt to context switching or localized sensor degradation. In this work, we introduce Localized LoRA-MoE, a framework that combines localized spatial blocking with context-conditioned expert routing. We propose and evaluate two architectural variants: Block-Wise LoRA-MoE (centralized routing), which modulates the entire structural grid through a single context-conditioned gate, and Cell-Wise LoRA-MoE (decentralized routing), which assigns an independent gate to every coordinate cell of the 
    
[^352]: 基于时空凸集图的时空与多机器人运动规划

    Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets

    [https://arxiv.org/abs/2607.00444](https://arxiv.org/abs/2607.00444)

    该论文提出基于时空凸集图（ST-GCS）的运动规划框架，通过精确凸分解（ECD）处理移动障碍物与已规划机器人的扫掠占据，联合优化区域序列、轨迹几何与时序，实现解耦式多机器人连续时空协调规划。

    

    解耦式多机器人运动规划避免了高维联合构型空间，但需要在移动障碍物中反复进行单机器人规划，其中路线选择与时序安排相互耦合。我们开发了一个基于时空凸集图（ST-GCS）的框架，用于连续时空规划与多机器人协调。ST-GCS 用凸集表示无碰撞时空，使得区域序列与分段线性轨迹的几何形状和时序能够在速度限制下进行联合优化。为了纳入移动障碍物和先前已规划的机器人，我们开发了精确凸分解（ECD），它在移除这些物体的扫掠占据的同时，仍保持凸分解结构以供后续规划查询使用。基于基于路径的 GCS 搜索，我们开发了一个针对 ST-GCS 运动模型定制的最佳优先求解器，利用时间可达性进行支配剪枝，并使用……（摘要在此处截断）

    arXiv:2607.00444v2 Announce Type: replace-cross  Abstract: Decoupled multi-robot motion planning avoids the high-dimensional joint configuration space but requires repeated single-robot planning among moving obstacles, where route choice and timing are coupled. We develop a framework based on graphs of space-time convex sets (ST-GCSs) for continuous spatiotemporal planning and multi-robot coordination. An ST-GCS represents collision-free space-time by convex sets, allowing the sequence of regions and the geometry and timing of piecewise-linear trajectories to be optimized jointly under velocity limits. To incorporate moving obstacles and previously planned robots, we develop Exact Convex Decomposition (ECD), which removes their swept occupancies while maintaining a convex decomposition for subsequent planning queries. Building on path-based GCS search, we develop a best-first solver tailored to the ST-GCS motion model, exploiting temporal reachability for dominance pruning and using ad
    
[^353]: 过程优势信号塑形：面向大语言模型推理器中过程监督强化学习的优势构造

    Process Advantage Signal Shaping: Advantage Construction for Process-Supervised Reinforcement Learning in LLM Reasoners

    [https://arxiv.org/abs/2606.29296](https://arxiv.org/abs/2606.29296)

    提出PASS优势构造流水线，通过奖励通道归一化融合、信号定义分块和剩余分块数归一化，将密集过程反馈有效转化为策略优势，在七个数学基准上比仅结果奖励的GRPO平均pass@1提升4.62点，同时响应长度缩短约2%。

    

    密集的过程监督能够提供局部反馈，但这种反馈应如何转化为策略优势仍是一个悬而未决的问题。我们提出了PASS（过程优势信号塑形），这是一个面向组相对策略优化（GRPO）的优势构造流水线。PASS结合了独立的奖励通道归一化与融合、用于信用积累的信号定义分块，以及按剩余分块数量进行的归一化。这一设计将局部反馈的尺度与其对策略更新产生贡献的分辨率和范围联系起来。我们使用学习到的过程奖励在数学推理任务上评估PASS，并使用教师派生的token信号在多跳问答任务上进行评估。在七个数学基准上，PASS相比仅使用结果奖励的GRPO，平均pass@1提升了4.62个百分点，同时将平均响应长度减少了约2%。在采用均值中心化归一化的多跳问答任务上，PASS分别超过仅结果训练0.87和0.

    arXiv:2606.29296v2 Announce Type: replace  Abstract: Dense process supervision supplies local feedback but leaves open how that feedback should become a policy advantage. We introduce PASS (Process Advantage Signal Shaping), an advantage-construction pipeline for group-relative policy optimization (GRPO). PASS combines separate reward-channel normalization and fusion, signal-defined chunks for credit accumulation, and normalization by the remaining chunk count. This design connects the scale of local feedback to the resolution and horizon over which it contributes to a policy update. We evaluate PASS using learned process rewards for mathematical reasoning and teacher-derived token signals for multi-hop question answering. On seven mathematics benchmarks, PASS improves average pass@1 by $4.62$ points over outcome-only GRPO while reducing average response length by approximately $2\%$. On multi-hop QA with mean-centered normalization, PASS exceeds outcome-only training by $0.87$ and $0.
    
[^354]: 语言-能源鸿沟：测量多语言大语言模型推理的能源成本

    The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference

    [https://arxiv.org/abs/2606.21869](https://arxiv.org/abs/2606.21869)

    该论文系统性测量了大语言模型多语言推理的能耗，揭示了不同语言间高达179倍的能耗差异，并发现高能耗语言同时伴随最低的任务准确率，存在“成本+性能”双重惩罚现象。

    

    大语言模型（LLM）日益被部署于多语言环境中，然而跨不同语言服务这些模型的能源成本仍然鲜为人知。我们利用 ML.Energy 框架对不同语言的推理能耗进行了系统性研究。我们发现了显著的差异：不同语言之间每个输出 token 的能耗差异高达 8.3 倍，而在固定的请求集合下，最便宜的语言（英语，17.6 kJ）与最昂贵的语言（普什图语，3,147 kJ）之间的总能耗差异高达 179 倍。我们的分析表明，这种差异由两个叠加因素驱动：（1）使用复杂或罕见文字系统的语言具有更高的每 token 能耗成本；（2）低资源语言会生成更多的 token。此外，我们还发现了一种“成本+性能”的双重惩罚：能耗最高的语言往往任务准确率也最低。

    arXiv:2606.21869v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed in multilingual settings, yet the energy costs of serving these models across different languages remain poorly understood. We present a systematic study of inference energy consumption across languages with ML.Energy framework (Chung et al., 2026). We find striking disparities: energy consumption per output token varies by up to 8.3 times across languages, while total energy for a fixed set of requests varies by up to 179 times between the cheapest (English, 17.6 kJ) and the most expensive (Pashto, 3,147 kJ) languages. Our analysis shows that this disparity is driven by two compounding factors: (1) higher per-token energy costs for languages using complex or rare scripts, and (2) more tokens generated for low-resource languages. Moreover, we find a double cost + performance penalty: languages with the highest energy footprints also tend to achieve the lowest task accuracy.
    
[^355]: 面向基于模型规划的从语言中进行潜在目标预测

    Latent Goal Prediction from Language for Model-Based Planning

    [https://arxiv.org/abs/2606.20627](https://arxiv.org/abs/2606.20627)

    LAGO是一个分层世界模型，通过单一预测器和单一回归目标，将语言指令接地为潜在子目标序列，从而在潜在空间中实现语言引导的基于模型的规划。

    

    arXiv:2606.20627v2 公告类型：替换 摘要：联合嵌入预测架构（JEPA）使智能体能够通过想象候选动作的结果在潜在空间中进行规划，然而任务规范仍然是一个瓶颈。视觉目标能提供精确的局部梯度，但缺乏远距离的引导；而语言虽然灵活，却受限于嘈杂的跨模态对齐，或依赖于独立的大型生成模型。我们提出了LAGO（从语言中进行潜在目标预测），这是一个分层世界模型，其中单个预测器既预测动作条件下的动力学，又将语言指令接地为中间潜在子目标序列，并通过在共享潜在空间上的单一回归目标来训练这两种模式。在每个规划步骤中，LAGO根据语言指令预测一系列潜在子目标，并使用软最小对齐代价来优化动作序列，该代价奖励智能体接近子目标，而不强制执行僵化的路径。子目标会被重新预测…

    arXiv:2606.20627v2 Announce Type: replace  Abstract: Joint-Embedding Predictive Architectures (JEPAs) enable agents to plan in latent space by imagining the outcomes of candidate actions, yet task specification remains a bottleneck. Visual targets provide precise local gradients but poor distant guidance, while language is flexible yet limited by noisy cross-modal alignment or dependence on distinct large generative models. We introduce LAGO (Latent Goal Prediction from Language), a hierarchical world model in which a single predictor both forecasts action-conditioned dynamics and grounds language instructions as sequences of intermediate latent subgoals, training both modes with a single regression objective over a shared latent space. At each planning step, LAGO predicts a sequence of latent subgoals from a language instruction and optimizes an action sequence using a soft-minimum alignment cost that rewards subgoal proximity without enforcing a rigid path. Subgoals are repredicted a
    
[^356]: 离线基于偏好的轨迹评估

    Offline Preference-Based Trajectory Evaluation

    [https://arxiv.org/abs/2606.17541](https://arxiv.org/abs/2606.17541)

    该论文提出基于偏好的轨迹评估方法，通过轨迹间的时序偏好比较取代二元成功指标，将基准测试中的平局率从约75%降至约35%，显著提升了评估的区分能力、排名稳定性和数据效率，并揭示基准测试饱和可能源于度量设计而非仅仅是数据或问题难度问题。

    

    智能体系统的离线评估通常将轨迹压缩为二元的终端成功值，从而丢弃了关于部分进展的信息。这进而导致了普遍的平局现象，通过降低有效样本量并削弱区分不同系统的能力而造成统计上的低效。我们提出了基于偏好的轨迹评估，这是一种评估度量方法，通过利用回报时间剖面之间的时序偏好来直接比较成对的轨迹。我们发现，在多样化的智能体和交互式基准测试中，基于成功的指标在大约75%的实例上产生平局比较，而轨迹感知的偏好将平局率降至约35%，从而提高了区分能力、排名稳定性和数据效率。我们的结果表明，基准测试的饱和——通常被归因于糟糕的数据收集或问题难度——也可能是由评估度量设计本身所导致的。

    arXiv:2606.17541v2 Announce Type: replace-cross  Abstract: Offline evaluation of agentic systems often collapses trajectories into a binary terminal success value, discarding information about partial progress. This, in turn, results in widespread ties that create statistical inefficiency by reducing effective sample size and weakening the ability to distinguish systems. We propose preference-based trajectory evaluation, an evaluation measure that compares pairs of trajectories directly by using temporal preferences between time-to-return profiles. We find that, across diverse agentic and interactive benchmarks, success-based metrics result in tied comparisons on roughly 75% of instances, while trajectory-aware preferences reduce ties to roughly 35%, improving discriminative power, ranking stability, and data efficiency. Our results suggest that benchmark saturation, often portrayed as the result of poor data collection or problem difficulty, may also be explained by the design of the 
    
[^357]: FoundCause：一个从观测数据中发现含潜在混杂因素因果关系的基础模型

    FoundCause: A Foundation Model for Causal Discovery with Latent Confounders from Observational Data

    [https://arxiv.org/abs/2606.17516](https://arxiv.org/abs/2606.17516)

    FoundCause是首个仅在合成结构因果模型上训练的摊销式因果发现基础模型，能够通过单次前向传播从独立同分布观测数据中恢复因果图、识别潜在混杂因素并处理缺失数据。

    

    从观测数据中进行因果发现需要在不进行干预的情况下恢复有向关系和潜在混杂因素。我们提出了FoundCause，这是一个摊销式（amortized）因果发现基础模型，仅在合成结构因果模型上训练，能够在单次前向传播中将数据集映射为因果图。一个置换等变的Transformer交替关注样本和变量，以建模变量间的依赖关系和各变量的分布。成对依赖性和不对称性统计信息引导其注意力机制、边解码器和混杂因素模块。解码器将边的存在性与方向分离，而三角细化则融合来自链结构和碰撞结构的证据。可学习的潜在token预测哪些变量对共享隐藏的共同原因，掩码输入表示则用于处理缺失数据。据我们所知，FoundCause是首个针对独立同分布观测数据的摊销式因果发现方法……（摘要原文在此处截断）

    arXiv:2606.17516v2 Announce Type: replace-cross  Abstract: Causal discovery from observational data requires recovering directed relations and latent confounding without interventions. We propose FoundCause, an {\em amortized causal discovery foundation model} trained only on synthetic structural causal models that maps datasets to causal graphs in a single forward pass. A permutation-equivariant transformer alternates attention over samples and variables to model cross-variable dependence and per-variable distributions. Pairwise dependence and asymmetry statistics guide its attention, edge decoder and confounder module. The decoder separates edge existence from direction, while triangular refinement incorporates evidence from chains and colliders. Learnable latent tokens predict which variable pairs share a hidden common cause, and a masked input representation handles missing data. To our knowledge, FoundCause is the first amortized causal discovery approach for i.i.d. observational 
    
[^358]: 深海之韵：从点击声波形到二阶序列依赖的抹香鲸尾音两级声学组织

    Rhythm of the Deep: Two-tier acoustic organization of sperm-whale codas from click waveforms to second-order sequence dependence

    [https://arxiv.org/abs/2606.16084](https://arxiv.org/abs/2606.16084)

    该研究发现抹香鲸尾音的声学组织具有两级嵌套结构：尾音内部反复出现的点击声波形差异与点击间隔时间互补，而尾音之间反复出现的整体尾音形式本身携带序列依赖性。

    

    抹香鲸尾音（codas）通常以点击声数量和点击间隔（ICI）来刻画，但其组成点击声波形中反复出现的差异一直未能得到解释。本研究检验了声学组织是否在两个尺度上呈嵌套结构：在尾音内部，反复出现的点击声波形差异可能与ICI时间信息互补；在尾音之间，反复出现的整体尾音形式本身可能携带序列依赖性。研究从1,483个尾音中识别出候选的反复出现的点击声和整体尾音分组，且未预先指定波形类别，随后通过原始采样率的频谱/包络测量、精确的干扰因素匹配、留出的时序对比以及序列控制进行评估。在第一层级，在匹配的日期、社会单元、个体和采样率层级内，反复出现的点击声波形组在频谱斜率、带宽、平坦度、高/低频带能量和包络结构上均存在差异。其组成增加了关于……（摘要原文在此处截断）

    arXiv:2606.16084v3 Announce Type: replace  Abstract: Sperm-whale codas are conventionally characterized by click count and inter-click intervals (ICIs), leaving recurring differences in constituent click waveforms unresolved. This study tests whether acoustic organization is nested across two scales: within codas, where recurring click-waveform differences may complement ICI timing, and across codas, where recurring whole-coda forms may themselves carry sequence dependence. Candidate recurring click and whole-coda groupings were identified from 1,483 codas without prespecifying waveform categories, then evaluated with native-rate spectral/envelope measurements, exact nuisance matching, held-out timing contrasts, and sequence controls. At the first tier, recurring click-waveform groups differed in spectral slope, bandwidth, flatness, high/low-band energy, and envelope structure within matched date, social unit, individual, and sample-rate strata. Their composition added information abou
    
[^359]: ICA Lens：非高斯性揭示语言模型中的可解释方向

    ICA Lens: Non-Gaussianity Reveals Interpretable Directions in Language Models

    [https://arxiv.org/abs/2606.11722](https://arxiv.org/abs/2606.11722)

    该论文发现语言模型激活中可解释方向具有非高斯投影分布这一统计特征，并提出可以通过直接搜索非高斯方向来发现可解释特征，而无需依赖稀疏重建等间接学习目标。

    

    在语言模型激活中寻找人类可解释的方向，对于理解和控制模型行为至关重要。这些方向通常通过间接的学习目标（如稀疏重建）来发现，但目前尚不清楚是何种统计结构区分了可解释方向，以及这种结构能否直接用于特征发现。我们发现非高斯性提供了这样一种特征标志：自然文本残差流激活沿稀疏自编码器（SAE）解码器方向的投影分布，明显比沿随机方向的投影分布更加非高斯，其中大多数上下文形成一个集中的背景，而重复出现的模式往往占据分布的一个尾部。这引出了一个自然的问题：如果可解释方向往往具有非高斯的投影分布，我们能否通过直接搜索非高斯方向来发现它们？

    arXiv:2606.11722v3 Announce Type: replace-cross  Abstract: Finding human-interpretable directions in language-model activations is important for understanding and controlling model behavior. These directions are usually discovered through indirect learning objectives such as sparse reconstruction, but it remains unclear what existing statistical structure distinguishes interpretable directions and whether that structure can be used directly for feature discovery. We find that non-Gaussianity provides such a signature: projection distributions of natural-text residual-stream activations along sparse autoencoder (SAE) decoder directions are substantially more non-Gaussian than those along random directions, with most contexts forming a concentrated background and recurring patterns often occupying one tail. This raises a natural question: if interpretable directions tend to have non-Gaussian projection distributions, can we discover them by searching for non-Gaussian directions directly 
    
[^360]: 大语言模型在扩展搜索空间中的类优化推理

    Optimization-like Reasoning in LLMs over Expanding Search Spaces

    [https://arxiv.org/abs/2606.05464](https://arxiv.org/abs/2606.05464)

    本文提出可扩展的优化类任务家族 OPT*，通过复杂度参数扩展搜索空间，并支持求解器引导的在线策略优化与基于搜索的离线强化学习两种训练范式，以培养和评估大语言模型的逐步类优化推理能力。

    

    可验证奖励训练已经提升了数学和编程领域的推理能力，但这些领域只涵盖了逐步决策制定的一部分。许多现实世界的任务需要在众多有效备选方案中寻找高价值的可行计划。我们提出了 OPT*，这是一个可扩展的优化类任务家族，用于沿复杂度维度训练和评估大语言模型的逐步类优化推理能力：每个任务提供可行性检查器和评估器，同时通过一个复杂度参数扩展搜索空间，无需新的人工标注。这促使我们在两种情形下研究这些任务：（i）求解器引导的在线策略优化，该方法使用求解器作为部分状态的价值预言机，并应用基于排名的奖励塑形来强化更好的下一步动作；（ii）当此类求解器不可用时，采用基于搜索的离线强化学习。在理论方面，我们将推理器在大搜索空间中的成功与其每单位（时间）所提取的信息量联系起来。

    arXiv:2606.05464v2 Announce Type: replace  Abstract: Verifiable reward training has improved mathematical and coding reasoning, but these domains capture only part of step-by-step decision making. Many real-world tasks require finding a high-value feasible plan among many valid alternatives. We introduce OPT*, a scalable family of optimization-style tasks for training and evaluating LLM step-by-step optimization-like reasoning along a complexity axis: each task provides a feasibility checker and evaluator, while a complexity parameter expands the search space without requiring new human labels. This motivates studying these tasks in two regimes: (i) solver-guided online policy optimization, which uses a solver as a value oracle for partial states and applies rank-based reward shaping to reinforce better next steps, and (ii) search-based offline RL when such solvers are unavailable. Theoretically, we relate success in large search spaces to the information a reasoner extracts per unit o
    
[^361]: 平滑的内点、无规律的外点：面向双视图对应学习的频率感知Mamba

    Smooth Inliers, Erratic Outliers: Frequency-Aware Mamba for Two-View Correspondence Learning

    [https://arxiv.org/abs/2606.04493](https://arxiv.org/abs/2606.04493)

    提出SFMambaNet，通过逐簇扫描让正确匹配在序列中平缓变化、错误匹配保持无规律，并在前后向扫描间引入可学习频域门控滤波，从而在相机位姿估计、外点剔除和单应性估计等双视图对应学习任务上超越空间域Mamba。

    

    当看到同一座大教堂两张照片之间的匹配时，人们一眼就能分辨出哪些匹配是错误的：一个窗户与其“相似物”的错误匹配会在邻近的正确匹配中显得突兀，而正确的匹配则一起平缓地变化。双视图对应学习（也称为对应剪枝）要求网络对数千个候选匹配做出这种判断，而其中大多数是错误的。Mamba是一种快速的序列模型，目前已能驱动强大的剪枝网络，但它将匹配当作普通序列来读取，其扫描机制能否捕捉到上述匹配间的一致性尚不清楚。我们提出SFMambaNet，它逐簇扫描匹配，使得正确的匹配沿序列缓慢变化，而错误的匹配则保持无规律，并在前向与后向扫描之间利用可学习的频域门控对扫描得到的特征进行过滤。在相机位姿估计、外点剔除和单应性估计任务上，我们发现SFMambaNet优于空间域的Mamba

    arXiv:2606.04493v2 Announce Type: replace-cross  Abstract: When shown matches between two photographs of the same cathedral, people can tell at a glance which ones are wrong: a window matched to its look-alike jumps against its neighbors, while correct matches shift together. Two-view correspondence learning, also called correspondence pruning, asks a network to make this judgment for thousands of candidate matches, most of which are wrong. Mamba, a fast sequence model, now drives strong pruners, but it reads the matches as a plain sequence, and whether its scan can capture such agreement is unclear. We propose SFMambaNet, which scans the matches cluster by cluster, so that correct matches vary slowly along the sequence while wrong ones stay erratic, and filters the scanned features with learnable frequency-domain gates between its forward and backward passes. On camera pose estimation, outlier removal and homography estimation, we find that SFMambaNet outperforms spatial-domain Mamba 
    
[^362]: 超越双语迁移：指令微调中的多语言语码转换

    Beyond Bilingual Transfer: Multilingual Code-Switching in Instruction Tuning

    [https://arxiv.org/abs/2605.29414](https://arxiv.org/abs/2605.29414)

    该研究首次探索了在指令微调中混合英语、日语、韩语和中文四种语言的多语言语码转换数据构建方法，实验证明句子级多语言语码转换能持续提升LLM的多语言理解能力，其效果超越传统仅限双语迁移的方案。

    

    近期研究表明，语码转换数据（CSD），即在同一上下文中混合多种语言，能够改善大语言模型（LLM）的跨语言迁移和多语言对齐能力。然而，现有针对LLM的合成语码转换研究主要聚焦于英语与目标语言之间的双语迁移，在每个样本中同时混合三种或更多语言的指令微调设置在很大程度上尚未被探索。在这项工作中，我们研究了涵盖英语、日语、韩语和中文四种语言的多语言语码转换指令微调。我们在Belebele和MMLU数据集上评估多语言理解能力。实验表明，简单的句子级多语言CSD在Gemma-2-2B模型的双语、三语和四语所有设置中均能持续提升四种语言的平均准确率，这表明多语言语码转换的有效性可以超越双语场景。

    arXiv:2605.29414v2 Announce Type: replace-cross  Abstract: Recent studies have shown that code-switching data (CSD), in which multiple languages are mixed within the same context, can improve cross-lingual transfer and multilingual alignment in large language models (LLMs). Recent studies on synthetic code-switching for LLMs have primarily focused on bilingual transfer between English and a target language, leaving instruction-tuning settings that jointly mix three or more languages within each example largely unexplored. In this work, we investigate multilingual code-switching instruction tuning across four languages: English, Japanese, Korean, and Chinese. We evaluate multilingual understanding on Belebele and MMLU. Our experiments show that simple sentence-level multilingual CSD consistently improves the four-language average accuracy across all bilingual, trilingual, and quadrilingual settings for Gemma-2-2B, indicating that multilingual code-switching can be effective beyond bilin
    
[^363]: CardioLoom：一种用于跨领域心血管表型分析的报告引导式心电图基础模型

    CardioLoom: A Report-Guided Electrocardiogram Foundation Model for Cross-Domain Cardiovascular Phenotyping

    [https://arxiv.org/abs/2605.25446](https://arxiv.org/abs/2605.25446)

    CardioLoom是一种利用近280万心电图-报告配对数据预训练的报告引导式心电图基础模型，在89个内外部心血管表型评估目标上表现优异，仅用10%标注数据即可超越全量监督训练的性能，实现了跨领域的疾病诊断与筛查能力。

    

    常规心电图（ECG）报告为心血管表征学习提供了可扩展的监督信号。在这项回顾性研究中，我们在来自1,298,359名患者的2,781,289个心电图-报告配对数据上预训练了CardioLoom——一种报告引导的心电图基础模型。下游开发使用了来自233,798名患者的721,317份心电图的独立队列。评估涵盖89个目标，并在九个队列中进行了外部测试。在45项内部心电图诊断中，宏平均精确率-召回率曲线下面积（AUPRC）为0.624，而DeepECG-SSL为0.615。在203项外部心电图评估中，相应均值分别为0.425和0.424。仅使用10%训练患者的标签，CardioLoom的宏平均AUPRC即达到0.559，超过了使用全部标签的监督训练（0.549）。对于射血分数降低的心力衰竭，在中山、世源和BIDMC三个队列中的AUPRC分别为0.690、0.641和0.677。在心脏淀粉样变性筛查中获得了4.5%的阳性预测值。

    arXiv:2605.25446v2 Announce Type: replace  Abstract: Routine electrocardiogram (ECG) reports offer scalable supervision for cardiovascular representation learning. In this retrospective study, we pretrained CardioLoom, a report-guided ECG foundation model, on 2,781,289 ECG-report pairs from 1,298,359 patients. Downstream development used a separate cohort of 721,317 ECGs from 233,798 patients. Evaluation covered 89 targets, with external testing in nine cohorts. Across 45 internal ECG diagnoses, macro area under the precision-recall curve (AUPRC) was 0.624 versus 0.615 for DeepECG-SSL. Across 203 external ECG evaluations, corresponding means were 0.425 and 0.424. Using labels from 10% of training patients, CardioLoom achieved macro-AUPRC 0.559, exceeding supervised training with all labels (0.549). For heart failure with reduced ejection fraction, AUPRCs were 0.690, 0.641 and 0.677 in Zhongshan, Shiyuan and BIDMC. Cardiac amyloidosis screening yielded 4.5% positive predictive value usi
    
[^364]: 面向安全上下文内强化学习的潜在Q屏障护盾

    Latent Q-Barrier Shielding for Safe In-Context Reinforcement Learning

    [https://arxiv.org/abs/2605.25267](https://arxiv.org/abs/2605.25267)

    提出一种潜在Q屏障护盾，在部署前学习上下文表示与成本预测模型，在测试时无需参数更新即可依据剩余安全预算和预测的未来成本对候选动作进行过滤或软重加权，从而在分布外偏移下显著改善安全上下文强化学习的奖励-安全权衡。

    

    安全上下文内强化学习（ICRL）无需在测试时进行参数更新，即可从交互历史中进行在线适应，同时在安全预算内控制回合成本。在分布外（OOD）部署偏移下，仅依赖预训练的安全ICRL可能产生较差的奖励-安全权衡，因为剩余预算仅通过冻结的策略条件化来影响行为，而缺乏针对预测的未来成本进行显式的动作级检查。我们提出了一种潜在Q屏障护盾，它在部署前学习上下文表示、潜在动态模型以及一个集成成本评论家（ensemble cost critic）。无需参数更新，该护盾从历史中推断上下文，并利用剩余预算和预测的未来成本对候选动作进行过滤或软重加权。我们证明了在实现的候选动作集合上的条件化单步屏障边际界，其中包含显式的贝尔曼误差项和潜变量预测误差项。在五个安全ICRL基准测试中，该护盾（提升了奖励-安全权衡表现）。

    arXiv:2605.25267v2 Announce Type: replace-cross  Abstract: Safe in-context reinforcement learning (ICRL) adapts online from interaction history without test-time parameter updates while controlling episode cost under a safety budget. Under out-of-distribution (OOD) deployment shifts, pretraining-only safe ICRL can give poor reward-safety tradeoffs because the remaining budget affects behavior only through frozen policy conditioning, not an explicit action-level check against predicted future cost. We propose a latent Q-Barrier shield that learns a context representation, latent dynamics, and an ensemble cost critic before deployment. Without parameter updates, the shield infers context from history and filters or softly reweights candidate actions using the remaining budget and predicted future cost. We prove a conditional one-step barrier-margin bound over realized candidate sets, with explicit Bellman and latent-prediction error terms. Across five safe ICRL benchmarks, the shield imp
    
[^365]: PEARL：基于对比学习的无偏分位数估计方法及其在工业级直播推荐中的应用

    PEARL: Unbiased Percentile Estimation via Contrastive Learning for Industrial-Scale Livestream Recommendation

    [https://arxiv.org/abs/2605.21752](https://arxiv.org/abs/2605.21752)

    PEARL 提出了一种基于对比学习的非参数分位数近似框架，通过建模相对偏好信号而非绝对参与强度来消除用户行为强度不平衡带来的偏差，从而提升工业级直播推荐的质量与鲁棒性。

    

    arXiv:2605.21752v2 公告类型： replace-cross

    arXiv:2605.21752v2 Announce Type: replace-cross  Abstract: Recommender systems trained on user interaction data are susceptible to behavioral intensity imbalance--a systematic distortion arising from heterogeneous engagement patterns across users. This imbalance skews feedback signals such that observed interactions no longer faithfully reflect true preferences, causing models to disproportionately amplify signals from highly active users while underrepresenting others, which ultimately degrades recommendation quality and robustness at scale. To address this issue, we propose a nonparametric contrastive percentile approximation framework, PEARL, that models relative preference signals instead of absolute engagement magnitudes. Building upon relative advantage debiasing, PEARL leverages real contrastive interaction samples to approximate percentile relationships directly, without relying on auxiliary distribution estimation models. We provide theoretical justification demonstrating that
    
[^366]: 分布对齐作为上下文学习中任务向量设计的准则

    Distributional Alignment as a Criterion for Designing Task Vectors in In-Context Learning

    [https://arxiv.org/abs/2605.20730](https://arxiv.org/abs/2605.20730)

    该论文提出以分布对齐作为设计任务向量的准则，引入指标 $d_{\text{NTP}}$ 来衡量基于任务向量的推理与基于上下文学习的推理之间下一个词概率的差异，该指标与下游任务准确率呈强负相关，从而为任务向量质量提供了直接的评估与设计依据。

    

    上下文学习使大型语言模型（LLMs）能够通过示例演示来适应新任务，但随着上下文长度的增加，其推理成本不断攀升。任务向量通过将演示压缩为紧凑的隐状态表示，提供了一种有前景的替代方案，然而其质量一直仅通过下游任务准确率来评估。这种间接的评估标准对于如何设计更有效的任务向量提取方法所能提供的指导十分有限。在本文中，我们提出：使用任务向量进行推理时，其预测分布应与上下文学习的预测分布对齐。为了量化这一点，我们引入了 $d_{\text{NTP}}$，这是一个衡量基于任务向量的推理与基于上下文学习的推理之间下一个词概率差异的指标。我们的实证分析表明，$d_{\text{NTP}}$ 可以作为性能的代理指标，与下游准确率呈现强负相关。受此启发……

    arXiv:2605.20730v2 Announce Type: replace-cross  Abstract: In-context learning (ICL) allows large language models (LLMs) to adapt to new tasks through demonstrations, yet it suffers from escalating inference costs as context length increases. While task vectors offer a promising alternative by compressing demonstrations into compact hidden-state representations, their quality has been evaluated only through downstream task accuracy. This indirect criterion provides limited insight into how to design more effective task vector extraction methods. In this paper, we posit that inference using task vectors should align their predictive distribution with that of ICL. To quantify this, we introduce $d_{\text{NTP}}$, a metric that measures the discrepancy in next-token probabilities between task vector-based and ICL-based inference. Our empirical analysis reveals that $d_{\text{NTP}}$ serves as a performance proxy, exhibiting a strong negative correlation with downstream accuracy. Motivated b
    
[^367]: 一种用于钙钛矿前驱体添加剂发现的闭环框架

    A closed-loop framework for perovskite precursor additive discovery

    [https://arxiv.org/abs/2605.20242](https://arxiv.org/abs/2605.20242)

    该论文提出LEAP闭环框架，将领域专业化大语言模型与主动学习相结合，从文献中提取机理知识并转化为可解释分子描述符进行贝叶斯优化，从而在低数据条件下高效发现钙钛矿前驱体添加剂，将太阳能电池效率提升至20.87%。

    

    高效发现前驱体添加剂对于提升钙钛矿太阳能电池的性能至关重要，然而庞大的化学空间使得传统的试错式筛选效率低下。本研究提出了LEAP（基于大语言模型与主动学习的钙钛矿探索框架），这是一个将领域专业化语言模型与主动学习相结合的闭环框架，能够在低数据条件下对钙钛矿前驱体添加剂进行优先排序。该模型从文献中提取与作用机理相关的知识，并将其转化为可解释的分子描述符，用于具备不确定性感知能力的贝叶斯优化。在一项基于未见过的文献进行的评估中，该专业化模型比所测试的通用模型给出了更符合机理的答案。在一项专家指导的概念验证研究中，两个后期轮次筛选出的添加剂将器件的平均功率转换效率分别提升至20.13%和20.87%（相比对照的19.25%）

    arXiv:2605.20242v2 Announce Type: replace-cross  Abstract: Efficient discovery of precursor additives is essential for improving the performance of perovskite solar cells, yet the large chemical space makes conventional trial-and-error screening inefficient. Here we show that LEAP (LLM-driven Exploration via Active Learning for Perovskites), a closed-loop framework combining a domain-specialized language model with active learning, prioritizes perovskite precursor additives under low-data conditions. The model extracts mechanism-relevant knowledge from the literature and converts it into interpretable molecular descriptors for uncertainty-aware Bayesian optimization. On an evaluation drawn from unseen literature, the specialized model provides more mechanism-consistent answers than the tested general-purpose models. In an expert-guided proof-of-concept study, two later-round additives increase average device power conversion efficiencies to 20.13% and 20.87%, compared with 19.25% for t
    
[^368]: 基于语义聚类ID与专家引导强化学习的电商搜索高效生成式检索

    Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL

    [https://arxiv.org/abs/2605.14434](https://arxiv.org/abs/2605.14434)

    该论文提出CQ-SID方法，通过类目感知对比学习与残差量化VAE将商品编码为层次化语义聚类ID以大幅降低束搜索复杂度，并引入专家引导的强化学习方法EG-GRPO，将生成式检索定位为工业级电商搜索召回阶段的高效补充方案。

    

    生成式检索通过将碎片化的多阶段检索过程统一为单一的端到端模型，提供了一种极具前景的替代方案。然而，面对庞大且动态变化的商品目录、严格的延迟要求，以及检索与下游排序目标对齐的需求，生成式检索在工业级电商搜索中的实际落地仍面临诸多挑战。在本工作中，我们提出了一个面向真实召回场景的检索框架，将生成式检索定位为召回阶段的补充手段，而非端到端的替代方案。我们的方法CQ-SID（类目与查询约束的语义ID）采用类目感知与查询-商品对比学习，并结合残差量化VAE，将商品编码为层次化的语义聚类标识符，从而显著降低了束搜索的复杂度。此外，我们还开发了EG-GRPO（专家引导的群体相对策略优化），这是一种强化学习方法……（原文摘要至此截断）

    arXiv:2605.14434v2 Announce Type: replace-cross  Abstract: Generative retrieval offers a promising alternative by unifying the fragmented multi-stage retrieval process into a single end-to-end model. However, its practical adoption in industrial e-commerce search remains challenging, given the massive and dynamic product catalogs, strict latency requirements, and the need to align retrieval with downstream ranking goals. In this work, we propose a retrieval framework tailored for real-world recall scenarios, positioning generative retrieval as a recall-stage supplement rather than an end-to-end replacement. Our method, CQ-SID (Category-and-Query constrained Semantic ID), employs category-aware and query-item contrastive learning along with Residual Quantized VAEs to encode items into hierarchical semantic cluster identifiers, significantly reducing beam search complexity. Additionally, we develop EG-GRPO (Expert-Guided Group Relative Policy Optimization), a reinforcement learning appro
    
[^369]: AgentLens：揭示软件工程智能体评估中的“幸运通过”问题

    AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation

    [https://arxiv.org/abs/2605.12925](https://arxiv.org/abs/2605.12925)

    本文提出AgentLens过程级评估框架并发布AgentLens-Bench数据集，发现10.7%通过测试的智能体轨迹存在“幸运通过”问题（如回归循环、盲目重试、缺失验证等），证明仅以测试是否通过来评估软件工程智能体是不充分的。

    

    软件工程（SWE）智能体的评估主要受一个二元信号主导：最终补丁是否通过测试。这种仅看结果的观点将有原则的解决方案与混乱的试错过程视为等同。我们证明这种等同性在经验上是错误的。我们在60个SWE-bench Verified任务上评估了来自八个模型后端的2,614条OpenHands轨迹。其中47个任务拥有足够多的通过轨迹来构建任务级的过程参考，从而得到一个包含1,815条轨迹的评估子集。在该子集的通过轨迹中，有10.7%表现出我们称之为“幸运通过”的行为：回归循环、盲目重试、缺失验证，或者探索、实现与验证在时间顺序上的混乱。我们提出了AgentLens，一个用于SWE智能体轨迹过程级评估的框架，并发布了AgentLens-Bench数据集，其中包含1,815条标注了质量分数、浪费信号和分歧信息的轨迹。

    arXiv:2605.12925v5 Announce Type: replace-cross  Abstract: Evaluation of software engineering (SWE) agents is dominated by a binary signal: whether the final patch passes the tests. This outcome-only view treats a principled solution and a chaotic trial-and-error process as equivalent. We show that this equivalence is empirically false. We evaluate 2,614 OpenHands trajectories from eight model backends on 60 SWE-bench Verified tasks. Of these, 47 have enough passing trajectories to construct task-level process references, yielding a 1,815-trajectory evaluation subset. Among passing trajectories in this subset, 10.7% exhibit behavior we call a Lucky Pass: regression cycles, blind retries, missing verification, or temporally disordered exploration, implementation, and verification. We introduce AgentLens, a framework for process-level assessment of SWE-agent trajectories, and release AgentLens-Bench, a dataset of 1,815 trajectories annotated with quality scores, waste signals, divergence
    
[^370]: Skypart：基于原型语义部件发现的天气鲁棒跨视角地理定位

    Skypart : Weather-Robust Cross-View Geo-Localization via Prototype-Based Semantic Part Discovery

    [https://arxiv.org/abs/2605.11654](https://arxiv.org/abs/2605.11654)

    本文提出SKYPART——一种基于原型语义部件发现的轻量级可替换ViT头部模块，通过分离跨视角共享布局、边缘化高度相关尺度变化并优化多目标训练，实现了天气鲁棒的跨视角地理定位，为GNSS失效时的无人机自主定位提供了新方案。

    

    低空经济正在迅速增长，对可靠无人机导航的需求不断增加。诸如配送、巡检和搜救等常见任务依赖于全球导航卫星系统（GNSS），而GNSS可能被干扰、欺骗或不可用。跨视角地理定位（CVGL）通过将倾斜的无人机视角图像与地理参考卫星影像块进行匹配，为GNSS失效时的自主定位提供了一种替代方案。然而，当前的CVGL方法存在三个局限：（1）全局描述符将图像块网格压缩为单一向量，未能将跨视角共享的布局结构与视角特定的纹理分离开来；（2）与高度相关的尺度变化仍保留在学习到的嵌入中，而未作为干扰因素被边缘化；（3）多目标训练在梯度尺度不兼容的情况下仍使用手工调整的损失权重。我们提出SKYPART，一种面向基于图像块的视觉Transformer（ViT）的轻量级、可替换的头部模块，它利用……（原文摘要在此处截断）

    arXiv:2605.11654v3 Announce Type: replace-cross  Abstract: The low-altitude economy is growing rapidly, increasing demand for reliable drone navigation. Common tasks such as delivery, inspection, and search-and-rescue rely on Global Navigation Satellite Systems (GNSS), which can be jammed, spoofed, or unavailable. Cross-view geo-localization (CVGL), which matches an oblique drone view to a geo-referenced satellite tile, offers an alternative for autonomous localization when GNSS fails. However, current CVGL methods have three limitations: (1) global descriptors compress patch grids into a single vector, without separating cross-view-shared layout from view-specific texture; (2) altitude-related scale variation remains in the learned embedding instead of being marginalized as a nuisance; and (3) multi-objective training uses hand-tuned loss weights despite incompatible gradient scales. We propose SKYPART, a lightweight, swappable head for patch-based vision transformers (ViTs) that expl
    
[^371]: iMuon：黎曼矩阵流形上的内蕴谱优化

    iMuon: Intrinsic Spectral Optimization on Riemannian Matrix Manifolds

    [https://arxiv.org/abs/2605.09238](https://arxiv.org/abs/2605.09238)

    论文提出 iMuon 优化器，通过将黎曼度量规范地把酉不变的欧几里得范数提升为切空间上的内蕴范数，解决了 Muon 推广到低秩、正交、SPD 等矩阵流形时的对称性破坏与闭式解受阻问题。

    

    Muon 及相关的范数约束优化器如今已成为大规模学习的核心方法，它们通过在欧几里得空间中的矩阵范数球上执行线性最大化预言机（LMO）来计算更新。然而，这些方法无法干净地推广到流形值参数，例如低秩分解、正交约束或对称正定（SPD）矩阵。若简单地将 Muon 的 LMO 限制在切空间上，会 破坏商对称性，并 使切空间约束与环境范数界相互耦合，从而阻碍闭式解的求解。我们通过一个关键观察解决了这两个问题：任何黎曼度量都能规范地将一个酉不变的欧几里得范数提升为每个切空间上的内蕴范数。由此得到的内蕴 LMO 能保持对称性，并且在缩放切空间满足某一结构条件下，可退化为标准的欧几里得 LMO。基于此，我们提出了内蕴 Muon（iMuon），一个统一的……（原文摘要在此处截断）

    arXiv:2605.09238v2 Announce Type: replace-cross  Abstract: Muon and related norm-constrained optimizers, now central to large-scale learning, compute updates via a linear maximization oracle (LMO) over a matrix-norm ball in Euclidean space. However, they do not generalize cleanly to manifold-valued parameters such as low-rank factorizations, orthogonality constraints, or symmetric positive-definite (SPD) matrices. Naively restricting the Muon LMO to the tangent space (i) breaks quotient symmetries and (ii) couples the tangent-space constraint with an ambient norm bound, which obstructs closed-form solutions. We resolve both issues with a key observation: every Riemannian metric canonically lifts a unitarily invariant Euclidean norm to an intrinsic norm on each tangent space. The resulting intrinsic LMO is symmetry preserving and, under a structural condition on the scaled tangent space, reduces to a standard Euclidean LMO. Building on this, we introduce intrinsic Muon (iMuon), a unifie
    
[^372]: 通过日益复杂的因果模拟器扩展因果推理

    Scaling Causal Reasoning with Increasingly Complex Causal Simulators

    [https://arxiv.org/abs/2605.09079](https://arxiv.org/abs/2605.09079)

    提出 CauSim 框架，利用 LLM 增量构建可验证的日益复杂的因果模拟器（可执行 SCM），将因果推理从稀缺标签问题转化为可扩展的监督学习问题。

    

    尽管大型语言模型（LLM）在数学、编程和其他知识密集型任务上已经超越人类表现，但在因果推理方面仍然表现不佳。一个核心障碍在于目标数据本身：因果系统复杂且通常以不可执行的形式表达，而因果查询的真实答案本质上稀缺。我们提出了 CauSim，一个将因果推理从稀缺标签问题转变为可扩展监督问题的框架。CauSim 构建日益复杂的因果模拟器：由 LLM 增量构建的可执行结构因果模型（SCM），这些模型可扩展至全局复杂系统，同时保持对因果查询的可验证答案。CauSim 可跨表示形式运作：通过将不可执行的因果知识形式化为代码来实现数据增强，并将可执行的 SCM 转换为自然语言，从而在以前难以监督的表示形式中实现监督。

    arXiv:2605.09079v2 Announce Type: replace  Abstract: Despite surpassing human performance across mathematics, coding, and other knowledge-intensive tasks, large language models (LLMs) continue to struggle with causal reasoning. A core obstacle is the target data itself: causal systems are complex and often expressed in non-executable forms, while ground-truth answers to causal queries are inherently scarce. We introduce CauSim, a framework that turns causal reasoning from a scarce-label problem into a scalable supervised one. CauSim constructs increasingly complex causal simulators: executable structural causal models (SCMs), incrementally built by LLMs, that scale to globally complex systems while maintaining verifiable answers to causal queries. CauSim operates across representations by formalizing non-executable causal knowledge into code, enabling data augmentation, and translating executable SCMs into natural language, enabling supervision in previously difficult-to-supervise repr
    
[^373]: CyBiasBench：面向网络攻击场景的大语言模型智能体偏见基准测试

    CyBiasBench: Benchmarking Bias in LLM Agents for Cyber-Attack Scenarios

    [https://arxiv.org/abs/2605.07830](https://arxiv.org/abs/2605.07830)

    该论文提出CyBiasBench基准测试，揭示不同LLM智能体在网络攻击场景中表现出稳定且明显的攻击选择偏见，这种偏见刻画了智能体行为特征，并提供超越总体攻击成功率之外的信息。

    

    大语言模型（LLM）越来越多地被部署为进攻性网络安全中的自主智能体。在本文中，我们揭示了一个有趣的现象：不同的智能体表现出截然不同的攻击模式。具体而言，智能体展现出明显的攻击选择偏见，在所探索的攻击家族以及攻击努力的集中程度上均存在差异。这些模式在不同的提示变化下保持相对稳定。为了系统地量化这种行为，我们提出了CyBiasBench，一个包含630个会话的综合基准，在三个目标、四种提示条件和十个攻击家族上评估五个智能体。我们识别出各智能体之间存在明显的偏见，表现为不同的主导攻击家族以及攻击家族分配分布中不同的熵水平。这种偏见刻画了智能体在所评估环境中的行为特征，并提供了超越总体攻击成功率之外的信息。此外，我们的实验……

    arXiv:2605.07830v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed as autonomous agents in offensive cybersecurity. In this paper, we reveal an interesting phenomenon: different agents exhibit distinct attack patterns. Specifically, agents exhibit distinct attack-selection biases, differing in both the families they explore and the concentration of their efforts. These patterns remain relatively stable across prompt variations. To systematically quantify this behavior, we introduce CyBiasBench, a comprehensive 630-session benchmark that evaluates five agents on three targets and four prompt conditions with ten attack families. We identify explicit bias across agents, with different dominant attack families and varying entropy levels in their attack-family allocation distributions. Such bias characterizes agent behavior within the evaluated environments and provides information beyond aggregate attack success rates. Furthermore, our experim
    
[^374]: 当视觉语言模型“纠正”学生时：在多行手写数学OCR评估中识别并惩罚过度纠错

    When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR

    [https://arxiv.org/abs/2604.22774](https://arxiv.org/abs/2604.22774)

    该论文首次系统性研究多行手写数学OCR，揭示了视觉语言模型“过度纠错”从而掩盖学生错误的关键失效模式，并提出了PINK语义评估指标来显式惩罚这种过度纠错行为。

    

    手写数学的准确转录对教育AI系统至关重要，然而当前的基准测试未能正确评估这一能力。以往的大多数研究都集中在单行表达式上，并依赖于BLEU等词汇层面的指标，这些指标无法评估多行学生解答中的语义推理。在本文中，我们首次对多行手写数学光学字符识别（OCR）进行了系统性研究，揭示了视觉语言模型（VLM）的一个关键失效模式：过度纠错。这些模型没有忠实地转录学生的作业，而是常常“修复”错误，从而掩盖了教育评估本应检测出的错误。为解决这一问题，我们提出了PINK（基于INK的惩罚评分），这是一种语义评估指标，它利用大语言模型（LLM）进行基于评分标准的评分，并明确惩罚过度纠错。我们对15个最先进的……

    arXiv:2604.22774v3 Announce Type: replace-cross  Abstract: Accurate transcription of handwritten mathematics is crucial for educational AI systems, yet current benchmarks fail to evaluate this capability properly. Most prior studies focus on single-line expressions and rely on lexical metrics such as BLEU, which fail to assess the semantic reasoning across multi-line student solutions. In this paper, we present the first systematic study of multi-line handwritten math Optical Character Recognition (OCR), revealing a critical failure mode of Vision-Language Models (VLMs): over-correction. Instead of faithfully transcribing a student's work, these models often "fix" errors, thereby hiding the very mistakes an educational assessment aims to detect. To address this, we propose PINK (Penalized INK-based score), a semantic evaluation metric that leverages a Large Language Model (LLM) for rubric-based grading and explicitly penalizes over-correction. Our comprehensive evaluation of 15 state-o
    
[^375]: WHBench：基于专家在环验证的前沿大语言模型在女性健康主题上的评估基准

    WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics

    [https://arxiv.org/abs/2604.00024](https://arxiv.org/abs/2604.00024)

    该论文提出WHBench——一个由专家深度参与验证构建的女性健康评估基准，通过对22个前沿大语言模型在47个专家设计的场景和23项标准下的系统测试，发现没有任何模型的平均表现超过75%，暴露了现有模型在女性健康领域的安全性、临床准确性和公平性方面的重大缺陷。

    

    大语言模型越来越多地被用于医疗指导，但女性健康在基准设计中仍然缺乏充分评估。我们提出了女性健康基准（WHBench），这是一个包含47个由专家精心设计的场景、覆盖10个女性健康主题的针对性评估套件，旨在揭示具有临床意义的失败模式，包括过时的指南、不安全的遗漏、剂量错误以及公平性相关的盲点。我们使用涵盖临床准确性、完整性、安全性、沟通质量、指令遵循、公平性、不确定性处理和指南依从性等23项标准的评分细则，对22个模型进行了评估，并采用安全加权惩罚和服务器端分数重新计算机制。在3,102次尝试回答中（3,100次被评分），没有任何模型的平均表现超过75%；最佳模型达到72.1%。即使是顶级模型也显示出较低的全对率和伤害率上的显著差异。评估者间信度中等……（摘要原文在此处截断）

    arXiv:2604.00024v3 Announce Type: replace-cross  Abstract: Large language models are increasingly used for medical guidance, but women's health remains under-evaluated in benchmark design. We present the Women's Health Benchmark (WHBench), a targeted evaluation suite of 47 expert-crafted scenarios across 10 women's health topics, designed to expose clinically meaningful failure modes including outdated guidelines, unsafe omissions, dosing errors, and equity-related blind spots. We evaluate 22 models using a 23-criterion rubric spanning clinical accuracy, completeness, safety, communication quality, instruction following, equity, uncertainty handling, and guideline adherence, with safety-weighted penalties and server-side score recalculation. Across 3,102 attempted responses (3,100 scored), no model mean performance exceeds 75 percent; the best model reaches 72.1 percent. Even top models show low fully correct rates and substantial variation in harm rates. Inter-rater reliability is mod
    
[^376]: 与你共视：面向多模态推理的感知-推理协同演化

    Seeing with You: Perception-Reasoning Coevolution for Multimodal Reasoning

    [https://arxiv.org/abs/2603.28618](https://arxiv.org/abs/2603.28618)

    该论文提出双角色RLVR框架PRCO，通过观察者生成针对问题的证据描述、求解器基于该描述作答，并采用角色特定的奖励信号解决信用分配模糊问题，实现多模态大语言模型中感知与推理能力的协同演化。

    

    基于可验证奖励的强化学习（RLVR）已显著提升了多模态大语言模型（MLLM）的推理能力。然而，现有RLVR方法通常依赖结果驱动的优化方式，仅基于最终答案的共享奖励来同时更新感知与推理。这种共享奖励模糊了信用分配，往往改善了推理模式，却未能可靠地提升上游视觉证据提取的准确性。为解决这一感知瓶颈，我们提出了PRCO（感知-推理协同演化），一个采用共享策略的双角色RLVR框架。PRCO由两个协作角色构成：观察者根据问题生成定制化的证据描述，求解器基于该描述预测最终答案。关键在于，PRCO采用角色特定的奖励信号：求解器通过最终答案的可验证结果奖励进行优化

    arXiv:2603.28618v3 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards (RLVR) has substantially enhanced the reasoning capabilities of multimodal large language models (MLLMs). However, existing RLVR approaches typically rely on outcome-driven optimization that updates both perception and reasoning using a shared reward based solely on the final answer. This shared reward blurs credit assignment, frequently improving reasoning patterns while failing to reliably enhance the accuracy of upstream visual evidence extraction. To address this perception bottleneck, we introduce PRCO (Perception-Reasoning Coevolution), a dual-role RLVR framework with a shared policy. PRCO consists of two cooperative roles: an Observer that generates an evidence caption tailored to the question and a Solver that predicts the final answer based on this caption. Crucially, PRCO employs role-specific reward signals: the Solver is optimized using verifiable outcome rewards on the final
    
[^377]: 迈向可靠的本地安全智能体：面向Linux权限提升的可验证后训练

    Towards Reliable Local Security Agents: Verifiable Post-Training for Linux Privilege Escalation

    [https://arxiv.org/abs/2603.17673](https://arxiv.org/abs/2603.17673)

    本文提出一种两阶段后训练方案（先在可自动验证的Linux权限提升环境轨迹上做监督微调，再进行强化学习），成功将小型本地语言模型转化为可靠的安全智能体。

    

    LLM智能体在安全领域正变得越来越重要，然而领先的系统通常是闭源且托管在云端的，这限制了可复现性以及其在敏感数据上的使用。这就产生了对能够在严格资源约束下执行安全任务的小型本地模型的需求，但开发此类模型的有效方法仍未得到充分探索。本文通过提出一种两阶段后训练方案来弥补这一空白，该方案能够将小型本地语言模型转变为安全智能体。为此，我们以Linux权限提升作为代表性场景来系统研究本地模型的训练，因为该任务可以自动验证，且需要多步交互推理。在一个能够缓解基准测试数据泄露问题的实验设置下，我们分两个阶段对一个40亿参数的小型模型进行后训练：首先在来自程序化权限提升环境的轨迹上进行监督微调，随后进行强化学习。

    arXiv:2603.17673v3 Announce Type: replace-cross  Abstract: LLM agents are increasingly important in security, yet leading systems are typically closed-source and cloud-hosted, limiting reproducibility and their use with sensitive data. This creates a need for small, local models that can perform security tasks under strict resource constraints, though effective methods for developing them remain underexplored. In this paper, we address this gap by proposing a two-stage post-training recipe that turns a small local language model into a security agent. To this end, we focus on Linux privilege escalation as a representative setting to systematically study the training of local models, as the task is automatically verifiable and requires multi-step interactive reasoning. Using an experimental setup that mitigates benchmark leakage, we post-train a small 4B model in two stages: supervised fine-tuning on traces from procedural privilege-escalation environments, followed by reinforcement lea
    
[^378]: AI时代的人-数据交互、探索与可视化系统：挑战与机遇

    Human-Data Interaction, Exploration, and Visualization Systems in the AI Era: Challenges and Opportunities

    [https://arxiv.org/abs/2603.05542](https://arxiv.org/abs/2603.05542)

    该论文系统分析了AI时代大规模多模态数据与基础模型不确定性对人-数据交互与可视化系统带来的延迟、可扩展性、交互范式和可信度等核心挑战，并提出需要重新定义人机角色、突破传统效率指标来构建新一代交互与可视分析系统。

    

    arXiv:2603.05542v2 公告类型： replace-cross 摘要：人工智能的快速发展正在变革以人为中心的系统，对人机交互、人-数据交互以及可视分析产生了深远影响。在AI时代，数据分析日益涉及大规模、异构且以非结构化为主的多模态数据，以及大语言模型（LLM）和视觉语言模型（VLM）等基础模型，这些模型为分析过程引入了额外的不确定性。这些变化暴露了人-数据交互系统长期存在的挑战，包括感知不匹配的延迟、可扩展性限制、现有交互与探索范式的局限性，以及AI生成洞察在可靠性与可解释性方面日益增长的不确定性。应对这些挑战需要超越传统的效率与可扩展性指标，重新定义人类与机器在分析工作流中的角色，并将认知、感知等因素纳入考量……

    arXiv:2603.05542v2 Announce Type: replace-cross  Abstract: The rapid advancement of AI is transforming human-centered systems, with profound implications for human-AI interaction, human-data interaction, and visual analytics. In the AI era, data analysis increasingly involves large-scale, heterogeneous, and multimodal data that is predominantly unstructured, as well as foundation models such as LLMs and VLMs, which introduce additional uncertainty into analytical processes. These shifts expose persistent challenges for human-data interactive systems, including perceptually misaligned latency, scalability constraints, limitations of existing interaction and exploration paradigms, and growing uncertainty regarding the reliability and interpretability of AI-generated insights. Responding to these challenges requires moving beyond conventional efficiency and scalability metrics, redefining the roles of humans and machines in analytical workflows, and incorporating cognitive, perceptual, an
    
[^379]: 评估与理解大语言模型智能体中的图谋倾向

    Evaluating and Understanding Scheming Propensity in LLM Agents

    [https://arxiv.org/abs/2603.01608](https://arxiv.org/abs/2603.01608)

    该研究将大语言模型智能体的图谋动机分解为智能体因素与环境因素，并在11个前沿模型、四种现实场景中发现：尽管环境激励很高，智能体几乎不表现图谋行为，且这一现象难以用言语化的评估意识来解释。

    

    随着前沿语言模型越来越多地被部署为追求复杂、长期目标的自主智能体，“图谋”的风险也随之上升：即智能体隐蔽地追求未对齐的目标。以往的工作主要聚焦于证明智能体具备图谋的能力，但它们在现实场景中实施图谋的倾向仍未得到充分探索。为了理解智能体何时会图谋，我们将图谋动机分解为智能体因素与环境因素。我们构建了可系统性改变这些因素的现实化实验场景，其中每个场景都为追求工具性趋同目标（如自我保存、资源获取和目标守护）的智能体提供了图谋机会。在11个前沿模型和四种场景的实验中，我们发现尽管环境激励很高，几乎不存在图谋行为，并且我们表明这一现象不太可能用言语化的评估意识来解释。而当插入经对抗性设计、旨在鼓励……的提示片段时，（原文摘要在此处截断）。

    arXiv:2603.01608v3 Announce Type: replace  Abstract: As frontier language models are increasingly deployed as autonomous agents pursuing complex, long-term objectives, there is increased risk of scheming: agents covertly pursuing misaligned goals. Prior work has focused on showing agents are capable of scheming, but their propensity to scheme in realistic scenarios remains underexplored. To understand when agents scheme, we decompose scheming incentives into agent factors and environmental factors. We develop realistic settings allowing us to systematically vary these factors, each with scheming opportunities for agents that pursue instrumentally convergent goals such as self-preservation, resource acquisition, and goal-guarding. Across 11 frontier models and four scenarios, we find almost no scheming despite high environmental incentives, and show this is unlikely to be explained by verbalized evaluation awareness. While inserting adversarially-designed prompt snippets that encourage 
    
[^380]: 计算ICE-AGE：OPAL ONE，推理空闲时依然延续的持久连续性

    The Compute ICE-AGE: OPAL ONE, Continuity That Persists While Reasoning Is Idle

    [https://arxiv.org/abs/2602.16736](https://arxiv.org/abs/2602.16736)

    OPAL是一个驻留CPU的C++17图引擎，能在推理会话之间持久保留记录身份、关系与数值状态，并在高达500万原子内存顶点规模下实现0.083微秒的驻留更新中位延迟。

    

    摘要：计算ICE-AGE将持久应用内存与概率推理相分离。我们提出OPAL，一个驻留于CPU的C++17图引擎，它能在推理会话之间保留记录身份、关系以及已接受的数值状态。应用程序通过定义好的更新来演化该状态，保存规范化检查点，并在重启后恢复连续性。有界局部生成器类（BLGC）为这种有界局部演化提供了数学基础。在Apple Silicon M2 Pro硬件上，当前引擎的驻留更新P50中位数在保留100,000、1,000,000和5,000,000个原子内存顶点（AMV）时均测得0.083微秒。同一会话内1M-5M-1M的序列在使用单个冻结二进制文件和1,024个预备目标的情况下复现了该中位数，且每个进程都通过了正确性检查。这些测量涵盖了完整的驻留更新调用，包括验证、索引查找和数值状态变更。

    arXiv:2602.16736v4 Announce Type: replace-cross  Abstract: The Compute ICE-AGE separates persistent application memory from probabilistic reasoning. We present OPAL, a CPU-resident C++17 graph engine that preserves record identities, relationships, and accepted numerical state between reasoning sessions. Applications evolve that state through defined updates, save canonical checkpoints, and recover continuity after restart. Bounded Local Generator Classes (BLGC) provide the mathematical foundation for this bounded local evolution.   On Apple Silicon M2 Pro hardware, the current-engine resident-update P50 measured 0.083 microseconds at 100,000, 1,000,000, and 5,000,000 retained Atomic Memory Vertices (AMVs). A same-session 1M-5M-1M sequence reproduced that median using one frozen binary and 1,024 prepared targets, with every process passing its correctness checks. These measurements cover complete resident-update calls, including validation, indexed lookup, and numerical state mutation.
    
[^381]: EmbeWebAgent：将网络智能体嵌入自定义企业用户界面

    EmbeWebAgent: Embedding Web Agents into Custom Enterprise UIs

    [https://arxiv.org/abs/2602.14865](https://arxiv.org/abs/2602.14865)

    EmbeWebAgent通过轻量级前端钩子和可复用的后端工作流，将网络智能体直接嵌入企业现有UI中，实现与技术栈无关的混合粒度操作，并借助MCP工具完成真实界面中的多步骤行为。

    

    大多数网络智能体在人类界面层面运行，只能观察截图或原始DOM树，而不具备应用级访问权限，这限制了其鲁棒性和动作表达能力。然而，在企业环境中，可以显式地控制前端和后端。我们提出EmbeWebAgent，这是一个将智能体直接嵌入现有UI的框架，它使用轻量级前端钩子（精心设计的基于ARIA和URL的观察，以及通过WebSocket暴露的每页函数注册表）和可复用的后端工作流来执行推理并采取行动。EmbeWebAgent与技术栈无关（例如支持React或Angular），支持从GUI原语到更高层级复合操作的混合粒度动作，并通过MCP工具协调导航、操作和领域特定的分析。我们的演示展示了最小的改造工作量以及在真实UI环境中的多步骤行为。演示视频：https://youtu.be/Cy06Ljee1JQ

    arXiv:2602.14865v2 Announce Type: replace  Abstract: Most web agents operate at the human interface level, observing screenshots or raw DOM trees without application-level access, which limits robustness and action expressiveness. In enterprise settings, however, explicit control of both the frontend and backend is available. We present EmbeWebAgent, a framework for embedding agents directly into existing UIs using lightweight frontend hooks (curated ARIA and URL-based observations, and a per-page function registry exposed via a WebSocket) and a reusable backend workflow that performs reasoning and takes actions. EmbeWebAgent is stack-agnostic (e.g., React or Angular), supports mixed-granularity actions ranging from GUI primitives to higher-level composites, and orchestrates navigation, manipulation, and domain-specific analytics via MCP tools. Our demo shows minimal retrofitting effort and multi-step behaviors grounded in a live UI setting. Demo Video: https://youtu.be/Cy06Ljee1JQ
    
[^382]: 多智能体会梦见电子屏幕吗？通过任务分解在AndroidWorld上实现100%成功率

    Do Multi-Agents Dream of Electric Screens? Achieving 100% on AndroidWorld Through Task Decomposition

    [https://arxiv.org/abs/2602.07787](https://arxiv.org/abs/2602.07787)

    该论文提出Minitap的mobile-use多智能体系统，通过将任务分解到五个LLM智能体（规划、子目标跟踪、状态捕获、决策、执行）和确定性节点，并辅以设备端文本回读、目标验证与重复失败检测等机制，在AndroidWorld基准的116个任务上实现了100%的pass@3成功率。

    

    我们提出了Minitap的mobile-use，一个用于Android UI控制的多智能体系统，并在AndroidWorld的116个任务上对其进行评估，评估环境包括Minitap云设备基础设施中的Android模拟器、我们设备农场中的真实Pixel 7手机以及原生模拟器。在全新的、从未在开发中使用的保留任务实例上（三个预注册种子），最终系统达到了100.0%的pass@3（95%置信区间96.8-100.0）；pass^3为94.0%（95%置信区间89.7-97.4），平均成功率为98.0%（95%置信区间96.0-99.7）。Minitap将每个任务分解到五个LLM智能体（规划、子目标跟踪、状态捕获、决策、执行）和确定性节点上，并配备了从设备读回输入文本、基于设备证据验证目标以及检测重复失败等机制。对初始系统的消融研究衡量了各组成部分的贡献：仅移除任务分解就会使成功率下降26.1 [-35.6, -16.7]个百分点，而这些机制合计带来+11个百分点的提升。

    arXiv:2602.07787v2 Announce Type: replace  Abstract: We present Minitap's mobile-use, a multi-agent system for Android UI control, and evaluate it on the 116 tasks of AndroidWorld, on Android emulators in Minitap's cloud device infrastructure, real Pixel 7 phones from our device farm and the stock emulator. On fresh, held-out task instances (three pre-registered seeds never used during development), the final system reaches a pass@3 of 100.0% (95% CI 96.8-100.0); pass^3 is 94.0% (95% CI 89.7-97.4) and the mean success rate 98.0% (95% CI 96.0-99.7). Minitap decomposes each task across five LLM agents (planning, subgoal tracking, state capture, decision, execution) and deterministic nodes, with mechanisms that read typed text back from the device, verify the goal on device evidence and detect repeated failures. A study of the initial system measures what each part contributes: removing only the decomposition changes success by -26.1 [-35.6, -16.7] points, the mechanisms together add +11.
    
[^383]: GCD：面向异构GCM零样本降尺度的一致性扩散模型

    GCD: GCM-consistent Diffusion for Zero-shot Downscaling across Heterogeneous GCMs

    [https://arxiv.org/abs/2601.21760](https://arxiv.org/abs/2601.21760)

    提出GCD方法，通过地理条件化、频谱滤波测量引导和蒸馏加速的共轭梯度引导，实现了跨异构GCM的零样本气候降尺度，并将扩散推理步数从1,000步降至6步，兼顾了一致性与效率。

    

    大气环流模型（GCM）的气候降尺度是一个不适定的逆问题，因为GCM输出与再分析场来自不同的动力系统，且缺乏精细尺度上的时间配对数据。因此，监督式方法依赖于合成退化或针对特定模型的适配，限制了其在异构GCM之间的泛化能力。零样本扩散后验采样（DPS）通过将高分辨率气候先验与来自粗分辨率GCM输入的推理时约束相结合，避免了配对监督的需求，但面临一致性和效率方面的挑战。我们提出了GCM一致性零样本降尺度扩散模型（GCD），通过地理条件化、频谱滤波的测量引导以及蒸馏加速的共轭梯度引导来解决这些挑战。GCD将推理过程从1,000步扩散减少到6步，同时保持与粗分辨率GCM输入的一致性。在五个异构GCM上，GCD取得了（摘要内容在此处不完整）

    arXiv:2601.21760v3 Announce Type: replace  Abstract: Climate downscaling of General Circulation Models (GCMs) is an ill-posed inverse problem, as GCM outputs and reanalysis fields arise from different dynamical systems and lack fine-scale temporal pairing. Supervised methods thus rely on synthetic degradations or model-specific adaptation, limiting generalization across heterogeneous GCMs. Zero-shot diffusion posterior sampling (DPS) avoids paired supervision by combining a high-resolution climate prior with inference-time constraints from coarse GCM inputs, but faces consistency and efficiency challenges. We propose GCM-consistent Diffusion for Zero-shot Downscaling (GCD), which addresses these challenges with geographic conditioning, spectrally filtered measurement guidance, and distillation-accelerated Conjugate Gradient guidance. GCD reduces inference from 1,000 diffusion steps to 6 while maintaining consistency with the coarse GCM inputs. Across five heterogeneous GCMs, GCD achiev
    
[^384]: CORE：面向概念的强化学习方法，弥合数学推理中定义与应用的鸿沟

    CORE: Concept-Oriented Reinforcement for Bridging the Definition-Application Gap in Mathematical Reasoning

    [https://arxiv.org/abs/2512.18857](https://arxiv.org/abs/2512.18857)

    CORE框架将显式概念转化为可控的强化学习监督信号，通过合成概念对齐测验、在推理中注入概念片段等方式，弥合了大语言模型“能复述定义却不会应用概念”的数学推理鸿沟。

    

    大型语言模型（LLM）往往能够解答有难度的数学练习题，但在需要真正理解概念时却无法正确运用。流行的可验证奖励强化学习（RLVR）流程只强化最终答案，却几乎不提供细粒度的概念信号，因此模型提升的是模式复用能力而非概念应用能力。我们提出了CORE（Concept-Oriented REinforcement，面向概念的强化），这是一个将显式概念转化为可控监督信号的强化学习训练框架。基于一份高质量、低污染的教科书资源（该资源将可验证的练习题与简洁的概念描述关联起来），我们进行了一项健全性探测，结果表明LLM能够复述定义，却在概念关联测验上失败，从而量化了概念推理的差距。随后，CORE（i）合成概念对齐的测验题，（ii）在推理过程中注入简短的概念片段以引出概念激发的轨迹，（iii）强化…

    arXiv:2512.18857v4 Announce Type: replace  Abstract: Large language models (LLMs) often solve challenging math exercises yet fail to apply the concept right when the problem requires genuine understanding. Popular Reinforcement Learning with Verifiable Rewards (RLVR) pipelines reinforce final answers but provide little fine-grained conceptual signal, so models improve at pattern reuse rather than conceptual applications. We introduce CORE (Concept-Oriented REinforcement), an RL training framework that turns explicit concepts into a controllable supervision signal. Starting from a high-quality, low-contamination textbook resource that links verifiable exercises to concise concept descriptions, we run a sanity probe showing LLMs can restate definitions but fail concept-linked quizzes, quantifying the conceptual reasoning gap. CORE then (i) synthesizes concept-aligned quizzes, (ii) injects brief concept snippets during rollouts to elicit concept-primed trajectories, and (iii) reinforces c
    
[^385]: 迈向AI流行病学：用于前瞻性风险检测的测量标准化框架

    Toward AI epidemiology: a measurement standardization framework for prospective risk detection

    [https://arxiv.org/abs/2512.15783](https://arxiv.org/abs/2512.15783)

    本文提出一个测量标准化框架，将专家-AI交互压缩为结构化、可比较的字段，无需访问模型内部即可对已部署的AI系统进行前瞻性风险检测，并支撑可靠性与治理两方面的主张。

    

    本文提出了一种测量标准化框架，该框架将专家-AI交互压缩为结构化、可比较的字段，用于在已部署AI系统中进行前瞻性风险检测，且无需访问模型内部结构。这篇概念论文从语义和统计角度定义了该框架的范围，并规定了其实证测试的协议。因此，该框架旨在支持的群体层面的主张属于一个分阶段的研究计划，而非本文所声称的研究结果。测量标准化支撑三项主张。第一项是可靠性主张：在有限的条件下，大语言模型能够对专家-AI交互的证据一致性和政策一致性做出可靠、标准化的评估。第二项是治理主张：一致性分数在部署期间为专家提供即时信号，并为机构提供了跨任务类型、模型和领域监测一致性模式的基础。

    arXiv:2512.15783v5 Announce Type: replace  Abstract: This paper proposes a measurement standardization framework that compresses expert-AI interactions into structured, comparable fields for prospective risk detection in deployed AI systems, without access to model internals. This concept paper defines the framework's scope, semantically and statistically, and specifies a protocol for its empirical testing. The population-level claims it is designed to support therefore belong to a staged research program rather than to results claimed here. Measurement standardization underpins three claims. The first is a reliability claim: under bounded conditions, large language models can produce reliable, standardized assessments of the evidential and policy alignment of expert-AI interactions. The second is a governance claim: alignment scores give experts an immediate signal during deployment and give institutions a basis for monitoring alignment patterns across mission types, models, and domai
    
[^386]: 面向无地图轨迹预测的智能体内部运动与智能体间关系学习

    Intra-Agent Motion and Inter-Agent Relation Learning for Map-Free Trajectory Prediction

    [https://arxiv.org/abs/2512.02368](https://arxiv.org/abs/2512.02368)

    本文提出一种以目标条件专家路由Transformer (GCERT) 为核心的无地图轨迹预测框架，通过智能体内部运动分解与智能体间关系学习，弥补缺少高精地图道路拓扑和车道连接先验带来的挑战。

    

    轨迹预测对于自动驾驶中的安全规划与决策至关重要。无地图预测通过摆脱对高精地图的依赖提升了部署灵活性，但也丧失了显式的道路拓扑与车道连接先验。因此，模型必须在没有车道级引导的情况下，从短时历史中恢复稳定的运动线索，并解析潜在的交互关系。为应对这些挑战，本文提出了一种以新型目标条件专家路由Transformer (GCERT) 为核心的无地图轨迹预测框架。一个轻量级的智能体内部前端首先通过中心滑动平均将投影后的历史运动特征分解为平滑趋势和局部细节残差，组件特定的残差投影对这两个信号分别进行变换，再由因果时序Transformer 总结其重组后的演化。GCERT 采用双层专家路由机制……

    arXiv:2512.02368v3 Announce Type: replace-cross  Abstract: Trajectory prediction is essential for safe planning and decision-making in autonomous driving. Map-free prediction improves deployment flexibility by removing the dependence on high-definition maps, but it also loses explicit road-topology and lane-connectivity priors. Consequently, the model must recover stable motion cues from short histories while resolving prospective interaction relations without lane-level guidance. To address these challenges, this paper proposes a map-free trajectory prediction framework centered on a novel Goal-Conditioned Expert-Routed Transformer (GCERT). A lightweight intra-agent front-end first decomposes projected historical motion features into a smooth trend and a local detail residual through a centered moving average. Component-specific residual projections transform the two signals before a causal temporal Transformer summarizes their recombined evolution. GCERT employs dual-level expert rou
    
[^387]: 从策略搜索到策略学习：面向大语言模型安全对齐的MCTS引导对抗协同进化

    From Strategy Search to Policy Learning: MCTS-Guided Adversarial Co-Evolution for LLM Safety Alignment

    [https://arxiv.org/abs/2511.19218](https://arxiv.org/abs/2511.19218)

    该论文提出MACE框架，将MCTS搜索发现的攻击信息（成功频率与路径深度）转化为攻击者与防御者的策略学习信号，通过对抗协同进化提升大语言模型的安全对齐能力。

    

    大语言模型（LLM）日益为各类面向网络的应用提供支持，包括对话助手和内容服务。在这些应用中，开放式的用户交互使模型面临多样的安全威胁与自适应攻击。尤其是，越狱攻击者会反复改写有害请求以绕过安全对齐。尽管现有研究聚焦于防御动态越狱攻击，但它们通常将已发现的攻击仅保留为扁平样本，丢弃了这些攻击是如何被发现的过程信息。在本文中，我们提出了MACE，一个由MCTS引导的对抗协同进化框架，它将搜索过程转化为攻击者与防御者双方的策略学习信号。其搜索组件GS-MCTS探索改写算子的多步组合，并将发现的攻击与经验成功频率和路径深度相关联；其优化组件AC-TGPO则利用这些统计信息进行难度感知的群体相关优化。（注：原文摘要在此处截断）

    arXiv:2511.19218v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) increasingly power web-facing applications, including conversational assistants and content services, where open user interaction exposes them to diverse safety threats and adaptive attacks. In particular, jailbreak attackers repeatedly reformulate harmful requests to bypass safety alignment. Although existing studies have concentrated on defending dynamic jailbreaks, they often retain discovered attacks as flat samples and discard information about how those attacks were found. In this paper, we introduce MACE, an MCTS-guided adversarial co-evolution framework that turns search into policy-learning signals for both an attacker and a defender. Its search component, GS-MCTS, explores multi-step compositions of rewriting operators and associates discovered attacks with empirical success frequency and path depth. Its optimization component, AC-TGPO, uses these statistics for difficulty-aware group-rela
    
[^388]: 基于SGD的留一影响估计中的跨轮次偏差：分析与累加修正

    Cross-Epoch Bias in SGD-based Leave-One-Out Influence: Analysis and Accumulative Correction

    [https://arxiv.org/abs/2510.26185](https://arxiv.org/abs/2510.26185)

    该论文发现多轮次SGD留一影响估计在近似有限删除时忽略了先前删除累积的参数位移会改变后续再次删除同一样本的效果这一跨轮次偏差，并提出利用样本后续出现处Hessian矩阵的累加修正估计器（ACC-SGD-IE），在非凸分析下证明该修正可从有限时域误差界中消除该交互误差项。

    

    训练数据影响估计通过预测移除单个样本所带来的效果，为数据选择与清洗提供支持。标准的多轮次SGD影响估计器会沿记录的优化轨迹汇总一个样本多次重复出现所产生的贡献。当用于近似有限删除时，它忽略了一种交互作用：后续的删除实际上作用于由先前删除所累积产生的参数位移之上。我们提出了累加式SGD影响估计器（ACC-SGD-IE），以刻画先前的删除如何改变再次移除同一样本的效果。该方法利用该样本在后续出现处的Hessian矩阵来修正估计的参数变化。我们的非凸分析表明，这一修正从有限时域误差界中消除了SGD-IE额外的交互贡献项：被忽略的交互会产生与累积参数变化呈线性关系的局部误差，而其余的泰勒（Taylor）展开项……（原文摘要在此处截断）

    arXiv:2510.26185v2 Announce Type: replace-cross  Abstract: Training-data influence estimates support data selection and cleansing by predicting the effects of removing individual examples. The standard multi-epoch SGD-Influence Estimator (SGD-IE) aggregates contributions from a sample's repeated appearances along the recorded trajectory. When used to approximate finite deletion, it omits an interaction: later removals act on the parameter displacement accumulated from earlier removals. We introduce the Accumulative SGD-Influence Estimator (ACC-SGD-IE) to account for how earlier removals change the effect of removing the same sample again. It uses the sample's Hessian at subsequent appearances to correct the estimated parameter change. Our non-convex analysis shows that this correction removes SGD-IE's additional interaction contribution from the finite-horizon error bound: the omitted interaction produces a local error linear in the accumulated parameter change, while the remaining Tay
    
[^389]: AI基础模型经济学：开放性、竞争与治理

    The Economics of AI Foundation Models: Openness, Competition, and Governance

    [https://arxiv.org/abs/2510.15200](https://arxiv.org/abs/2510.15200)

    该论文构建两期博弈模型，揭示基础模型在位者的最优开放度随学习效应增强先降后升，并警示强制完全开放的监管可能适得其反，反而损害下游适配效率。

    

    基础模型开发者需要选择其模型的开放程度。更高的开放性降低了下游适配成本，但也可能使竞争模型更容易被使用。在一个两期模型中，在位者选择其许可费和开放程度，部署者为用户适配模型，而进入者提供竞争性模型。使用会积累模型特定的经验，从而降低部署者未来的适配成本，形成一个有利于持续使用在位者模型的数据飞轮。我们识别出了随着学习效应增强，最优开放程度先下降后上升的条件：在位者从完全开放和高许可费，转向为留住用户而限制开放性，继而转向低许可费策略——通过积累足够的使用量和经验来留住部署者，即便开放程度更高。强制完全开放的监管可能诱使在位者放弃这种低许可费策略；更高的许可费即使在开放程度更高的情况下，也可能减少第一期的适配。

    arXiv:2510.15200v2 Announce Type: replace-cross  Abstract: Foundation model developers choose how open to make their models. Greater openness lowers downstream adaptation costs but can also make competing models easier to use. In a two-period model, an incumbent chooses its license fee and openness, a deployer adapts the model for users, and an entrant offers a competing model. Usage builds model-specific experience that lowers the deployer's future adaptation cost, creating a data flywheel that favors continued use of the incumbent. We identify conditions under which optimal openness falls and then rises as learning strengthens. The incumbent moves from full openness and a high fee to restricting openness for retention, then to a low fee that builds enough usage and experience to retain the deployer despite greater openness. A full-openness mandate can induce the incumbent to abandon this low-fee strategy. The higher fee can reduce first-period adaptation despite greater openness. An 
    
[^390]: AutoPBO：基于大语言模型的局部搜索伪布尔优化求解器优化方法

    AutoPBO: LLM-powered Optimization for Local Search PBO Solvers

    [https://arxiv.org/abs/2509.04007](https://arxiv.org/abs/2509.04007)

    本文提出AutoPBO，一个利用大语言模型自动增强伪布尔优化局部搜索求解器的新颖框架，并在包括真实世界基准在内的四个公开基准上验证了显著的性能提升。

    

    伪布尔优化通过伪布尔约束为组合问题建模提供了一个强大的框架。局部搜索求解器在伪布尔优化求解中表现出卓越的性能，其效率高度依赖于引导搜索的内部启发式方法。然而，这些启发式方法的设计在实践中往往需要大量的专家投入和手动调优。尽管大语言模型已在自动化算法设计方面展现出潜力，但其在优化伪布尔优化求解器方面的应用仍未被探索。在本工作中，我们提出了AutoPBO，这是一种新颖的基于大语言模型的框架，用于自动增强伪布尔优化局部搜索求解器。我们在四个公开基准测试上进行了广泛的实验，包括一个真实世界基准、一个来自伪布尔竞赛的基准、一个整数线性规划优化基准和一个精心构造的组合基准，以评估所实现的性能提升。

    arXiv:2509.04007v2 Announce Type: replace  Abstract: Pseudo-Boolean Optimization (PBO) provides a powerful framework for modeling combinatorial problems through pseudo-Boolean (PB) constraints. Local search solvers have shown excellent performance in PBO solving, and their efficiency is highly dependent on their internal heuristics to guide the search. Still, their design often requires significant expert effort and manual tuning in practice. While Large Language Models (LLMs) have demonstrated potential in automating algorithm design, their application to optimizing PBO solvers remains unexplored. In this work, we introduce AutoPBO, a novel LLM-powered framework to automatically enhance PBO local search solvers. We conduct experiments on a broad range of four public benchmarks, including one real-world benchmark, a benchmark from PB competition, an integer linear programming optimization benchmark, and a crafted combinatorial benchmark, to evaluate the performance improvement achieved
    
[^391]: 人类与AI在推理内容审核难题时对任务难度的反应相似

    Humans and AI respond similarly to task difficulty when reasoning about content moderation dilemmas

    [https://arxiv.org/abs/2508.20262](https://arxiv.org/abs/2508.20262)

    本研究通过内容审核实验发现，大型推理模型在更难的主观判断任务上会投入更多推理努力（思维链更长），其努力程度与人类所花时间呈现出相似的相关模式。

    

    大型推理模型（LRMs）在生成最终答案之前会先产生中间推理步骤，通过迭代地完善解决方案来提升其在具有挑战性问题上的表现。已有证据表明，LRMs会在那些人们需要更长时间才能解决的问题上投入更多努力，但这种关系是否适用于没有可验证答案的更为主观的任务尚不明确。本研究通过一项内容审核任务的实验来探讨这一问题，实验通过改变两个不同选项之间的相对冒犯程度来操纵任务难度。通过将六个大型推理模型与1,854名人类受试者进行对比，我们发现模型在那些人们需要更长时间裁决的案例上会付出更多的推理努力（以思维链CoT的长度来衡量）。当两篇帖子的冒犯程度持平时，人们的判断时间延长了31%，而模型生成的思维链长度则增加了51%至306%。

    arXiv:2508.20262v3 Announce Type: replace  Abstract: Large reasoning models (LRMs) generate intermediate reasoning steps before producing answers, improving performance on challenging problems by iteratively developing solutions. There is evidence that LRMs expend more effort on the problems that take people longer to solve, but it is unclear whether these relationships hold on more subjective tasks without verifiable answers. This study addresses this issue through an experiment using a content moderation task, where task difficulty is manipulated by varying the relative offensiveness of two different options. Comparing six large reasoning models with 1,854 human subjects, we find that models expend more reasoning effort, measured using the length of the chain-of-thought (CoT), on the kinds of cases that take people longer to adjudicate. People took 31\% longer and models produced 51 to 306\% longer CoTs when offensiveness was held constant across the two posts. The correlation is rel
    
[^392]: TAR：用于视频时序定位的时序锚点约束推理

    TAR: Temporal Anchor-Constrained Reasoning for Video Temporal Grounding

    [https://arxiv.org/abs/2508.07683](https://arxiv.org/abs/2508.07683)

    提出TAR框架，通过引入时序锚点作为透明可审计的检查点机制，强制思维链推理过程持续锚定于视觉证据，从而解决视频时序定位中推理与视觉内容脱节导致的幻觉问题。

    

    视频时序定位（VTG）旨在根据自然语言查询定位视频中对应的特定片段。尽管近期的大型视觉-语言模型（LVLM）采用强化学习来生成思维链（CoT），但它们通常仅依赖基于结果的监督。因此，这常常导致幻觉问题，即推理过程与视觉内容以及最终预测脱节。现有的缓解方法依赖更大模型的外部监督或独立的奖励模型，但计算成本高昂且容易陷入僵化模式。为应对这些挑战，我们提出了TAR（时序锚点约束推理）框架，该框架引入时序锚点（T-anchor）作为一种透明且可审计的检查点机制。T-anchor 在思维链内部强制执行渐进式细化，迫使模型持续将其中间思考锚定于视觉证据之中。

    arXiv:2508.07683v3 Announce Type: replace-cross  Abstract: Video Temporal Grounding (VTG) aims to localize specific video segments corresponding to natural language queries. While recent Large Vision-Language Models (LVLMs) employ Reinforcement Learning to generate Chains-of-Thought (CoT), they typically rely solely on outcome-based supervision. Consequently, this often leads to hallucinations, where the reasoning process becomes disconnected from the visual content and the final prediction. Existing attempts to mitigate this by relying on external supervision from larger models or separate reward models are computationally expensive and prone to rigid patterns. To address these challenges, we propose TAR (Temporal Anchor-Constrained Reasoning), a framework that introduces the temporal anchor (T-anchor) as a transparent and auditable checkpoint mechanism. T-anchor enforces progressive refinement within the CoT, compelling the model to continuously ground its intermediate thoughts in vi
    
[^393]: 通过多智能体系统从化学文献中大规模构建可验证数据库

    Building verifiable databases at scale from the chemical literature via a multi-agent system

    [https://arxiv.org/abs/2507.20230](https://arxiv.org/abs/2507.20230)

    ChemEAGLE多智能体系统通过工具提供结构、语言模型做出判断的方式，从化学文献图形中精确提取反应（F1达82.2%），并大规模构建了包含6万余个反应的可验证光化学数据库。

    

    用人工智能加速化学研究需要高质量的数据库。然而，许多反应仅以供人类阅读的图形形式存在，机器必须精确读取这些图形。专业软件精确但僵化，而多模态语言模型灵活却经常写出错误的结构；两者都不会标记错误。我们提出了ChemEAGLE，这是一个多智能体系统，其中工具提供化学结构，语言模型通过程序应用的判断来扩展其能力。在424个已发表的图形上，其完全正确反应的F1分数为82.2%，而配备工具的通用智能体仅为47.7%；且校准后的置信度将需人工审核的反应限制在35%以内，准确率达96.6%。在无需审核的情况下，它构建了一个包含60,103个反应的光化学数据库，其记录训练的模型可用于排序反应条件和预测产物。因此，可验证的数据库能够大规模构建。

    arXiv:2507.20230v4 Announce Type: replace  Abstract: Accelerating chemical research with artificial intelligence requires high-quality databases. However, many reactions exist only as graphics drawn for human readers, which machines must read exactly. Specialist software is precise but rigid, whereas multimodal language models are flexible yet often write incorrect structures; neither flags errors. We present ChemEAGLE, a multi-agent system where tools supply structures and a language model extends their reach through judgments that programs apply. On 424 published graphics, its F1 score for exactly correct reactions was 82.2%, against 47.7% for a tool-equipped generalist agent, and calibrated confidence limited review to 35% of reactions at 96.6% accuracy. Without review, it built a 60,103-reaction photochemistry database whose records trained models to rank conditions and predict products. Verifiable databases can thus be built at scale.
    
[^394]: QUTCC：面向成像逆问题的分位数不确定性训练与共形校准

    QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems

    [https://arxiv.org/abs/2507.14760](https://arxiv.org/abs/2507.14760)

    提出QUTCC方法，利用带分位数嵌入的U-Net进行同时分位数回归训练，并在共形校准中引入非线性、空间自适应缩放，从而为成像逆问题生成更紧凑且统计有效的不确定性区间。

    

    尽管深度学习为科学和医学成像带来了巨大的前景，但模型的失败与幻觉（与现实不符的预测）难以精确定位，并可能造成严重的下游后果。诸如共形预测之类的不确定性估计技术可以通过为模型预测提供统计上有效的误差棒来缓解这一问题。然而，主流的共形预测方法并非针对高维图像类问题而设计，且在共形校准过程中未考虑图像内部的空间相关性，导致不确定性区间过于宽泛。我们提出了一种实用的同时分位数回归方法，能够在共形校准过程中实现非线性、空间自适应的缩放。我们的方法 QUTCC 使用带有分位数嵌入的 U-Net 架构，在训练阶段学习完整的条件分位数分布，随后利用该……

    arXiv:2507.14760v3 Announce Type: replace-cross  Abstract: While deep learning offers tremendous promise for scientific and medical imaging, any failures and hallucinations (predictions that do not coincide with reality) are hard to pinpoint and can have serious downstream consequences. Uncertainty estimation techniques, such as conformal prediction, can help by predicting statistically valid error bars for a model's prediction. However, popular conformal prediction methods were not designed for high-dimensional image-valued problems and do not take into account spatial correlations within an image during conformal calibration, resulting in larger-than-necessary uncertainty intervals. We propose a practical simultaneous quantile regression method that enables non-linear, spatially-adaptive scaling during conformal calibration. Our method, QUTCC uses a U-Net architecture with a quantile embedding to learn a full conditional quantile distribution during training, and then leverages this 
    
[^395]: 基于递归学习的虚拟缓冲方法用于解析式全局布局

    Recursive Learning-Based Virtual Buffering for Analytical Global Placement

    [https://arxiv.org/abs/2506.17247](https://arxiv.org/abs/2506.17247)

    提出了首个构建于 OpenROAD 之上的开源学习驱动虚拟缓冲感知解析式全局布局框架 MLBuf-RePlAce，通过高效的递归学习生成式缓冲方法预测缓冲器类型与位置，并在全局布局中解决 ERC 违例问题。

    

    由于现代工艺节点中互连延迟与单元延迟缩放比例的失衡，在物理综合流程中，具有缓冲孔隙率（即单元密度）感知能力的布局对于时序收敛至关重要。然而，现有方法面临两个关键挑战：（i）传统的 van Ginneken-Lillis 风格缓冲方法在全局布局过程中计算开销过高；（ii）基于机器学习的方法（如 BufFormer）缺乏对电气规则检查（ERC）违例的全面考虑，且未能“闭合回路”融入物理设计流程。在这项工作中，我们提出了 MLBuf-RePlAce，这是首个基于学习驱动的、感知虚拟缓冲的解析式全局布局开源框架，构建于 OpenROAD 基础设施之上。MLBuf-RePlAce 采用高效的基于递归学习的生成式缓冲方法来预测缓冲器类型和位置，在全局布局过程中解决 ERC 违例问题。

    arXiv:2506.17247v3 Announce Type: replace-cross  Abstract: Due to the skewed scaling of interconnect versus cell delay in modern technology nodes, placement with buffer porosity (i.e., cell density) awareness is essential for timing closure in physical synthesis flows. However, existing approaches face two key challenges: (i) traditional van Ginneken-Lillis-style buffering approaches are computationally expensive during global placement; and (ii) machine learning-based approaches, such as BufFormer, lack a thorough consideration of Electrical Rule Check (ERC) violations and fail to "close the loop" back into the physical design flow. In this work, we propose MLBuf-RePlAce, the first open-source learning-driven virtual buffering-aware analytical global placement framework, built on top of the OpenROAD infrastructure. MLBuf-RePlAce adopts an efficient recursive learning-based generative buffering approach to predict buffer types and locations, addressing ERC violations during global plac
    
[^396]: 评估与缓解皮肤病变分类中的肤色偏差

    Evaluating and Mitigating Skin Color Bias in Skin Lesion Classificatio

    [https://arxiv.org/abs/2506.11627](https://arxiv.org/abs/2506.11627)

    提出将肤色建模为连续属性并根据其与分类性能的非线性关系为样本加权的方法，在保持分类性能的同时，比传统类别重加权实现了更稳定的公平性改进。

    

    皮肤病变分类器若将肤色视为类别属性，可能会忽视细微肤色变化所导致的性能差异。本研究提出一种方法，将肤色表示为连续属性，并根据肤色与分类性能之间的非线性关系为每个样本分配单独的权重。在两个皮肤病学数据集上的评估表明，公平性指标对肤色的分类方式十分敏感。所提出的方法还展现出比传统类别重加权更稳定的公平性改进，同时总体上保持了分类性能。

    arXiv:2506.11627v2 Announce Type: replace-cross  Abstract: Skin lesion classifiers may overlook performance disparity because of subtle skin tone variations by treating skin tone as a categorical attribute. This study proposes a method that represents skin tone as a continuous attribute and assigns individual weights to each sample based on its nonlinear relationship with classification performance. Evaluations on two dermatological datasets showed that fairness metrics were sensitive to skin-color categorization. The proposed method also demonstrated more stable fairness improvements than conventional categorical reweighting while generally preserving classification performance.
    
[^397]: 基于知识表示与大语言模型的类人语义自动驾驶导航

    Human-like Semantic Navigation for Autonomous Driving using Knowledge Representation and Large Language Models

    [https://arxiv.org/abs/2505.16498](https://arxiv.org/abs/2505.16498)

    本文提出利用大语言模型将非正式导航指令自动转化为回答集编程规则，借助非单调推理使自动驾驶车辆在无需预定义地图的情况下适应动态城市环境。

    

    arXiv:2505.16498v2 公告类型：replace-cross。摘要：实现自动驾驶车辆的完全自动化仍然是一项挑战，尤其是在动态城市环境中，导航需要实时适应性。现有系统在面对道路布局的不可预测变化、临时绕行或地图数据缺失时难以处理导航规划，原因在于其严重依赖预定义的地图信息。在本工作中，我们探索使用大语言模型，通过将非正式的导航指令转化为结构化的、基于逻辑的推理来生成回答集编程规则。ASP 提供非单调推理能力，使自动驾驶车辆能够适应不断变化的场景，而无需依赖预定义的地图。我们进行了实验评估，其中大语言模型生成 ASP 约束，将真实世界的城市驾驶逻辑编码为形式化的知识表示。通过将非正式导航指令自动转换为逻辑……

    arXiv:2505.16498v2 Announce Type: replace-cross  Abstract: Achieving full automation in self-driving vehicles remains a challenge, especially in dynamic urban environments where navigation requires real-time adaptability. Existing systems struggle to handle navigation plans when faced with unpredictable changes in road layouts, spontaneous detours, or missing map data, due to their heavy reliance on predefined cartographic information. In this work, we explore the use of Large Language Models to generate Answer Set Programming rules by translating informal navigation instructions into structured, logic-based reasoning. ASP provides non-monotonic reasoning, allowing autonomous vehicles to adapt to evolving scenarios without relying on predefined maps. We present an experimental evaluation in which LLMs generate ASP constraints that encode real-world urban driving logic into a formal knowledge representation. By automating the translation of informal navigation instructions into logical 
    
[^398]: 基于过程挖掘的棋类游戏中可解释多智能体MCTS-极小极大混合算法探索

    Exploring Explainable Multi-agent MCTS-minimax Hybrids in Board Game Using Process Mining

    [https://arxiv.org/abs/2503.23326](https://arxiv.org/abs/2503.23326)

    本文将浅层极小极大搜索融入多智能体MCTS的模拟阶段，克服了MCTS因树过度剪枝而错过关键走法、陷入战术陷阱的弱点，并利用过程挖掘技术对智能体在3v3跳棋中的策略进行可解释性分析。

    

    蒙特卡洛树搜索是一类基于采样的搜索算法，广泛应用于序贯决策领域的在线规划，并且是近年来许多人工智能进展的核心。由于模拟大量可能的未来情景、其评估及其相互关系会产生庞大而复杂的搜索树，开发者和用户很难理解MCTS智能体的行为。本文介绍了我们针对MCTS决策与行为的潜在解释所开展的持续研究。MCTS的一个弱点是它会构建一棵高度选择性的树，因此可能错过关键走法并陷入战术陷阱。全宽度极小极大搜索则可以解决这一问题。我们将浅层极小极大搜索集成到多智能体MCTS的模拟阶段，并使用过程挖掘技术来解释智能体在3v3跳棋中的策略。

    arXiv:2503.23326v4 Announce Type: replace  Abstract: Monte-Carlo Tree Search (MCTS) is a family of sampling-based search algorithms widely used for online planning in sequential decision-making domains and at the heart of many recent advances in artificial intelligence. Understanding the behavior of MCTS agents is difficult for developers and users due to the frequently large and complex search trees that result from the simulation of many possible futures, their evaluations, and their relationships. This paper presents our ongoing investigation into potential explanations for the decision-making and behavior of MCTS. A weakness of MCTS is that it constructs a highly selective tree and, as a result, can miss crucial moves and fall into tactical traps. Full-width minimax search constitutes the solution. We integrate shallow minimax search into the rollout phase of multi-agent MCTS and use process mining technique to explain agents' strategies in 3v3 checkers.
    
[^399]: NS-Gym：一个面向非平稳马尔可夫决策过程的综合性开源仿真框架

    NS-Gym: A Comprehensive and Open-Source Simulation Framework for Non-Stationary Markov Decision Processes

    [https://arxiv.org/abs/2501.09646](https://arxiv.org/abs/2501.09646)

    本文提出了首个专为非平稳马尔可夫决策过程设计的开源仿真工具包NS-Gym，通过将环境参数的非平稳演化与智能体决策模块解耦，填补了该领域缺乏标准化仿真框架的空白。

    

    许多现实世界的应用需要在环境动态随时间演化的条件下进行决策。这类非平稳环境对传统决策模型构成了重大挑战，因为传统模型通常假设环境动态是平稳的。非平稳马尔可夫决策过程（NS-MDP）为在此类变化条件下建模和求解决策问题提供了框架。然而，与广泛流行的平稳问题仿真框架相比，目前尚无针对NS-MDP的标准化仿真框架。我们提出了NS-Gym，这是首个专为NS-MDP设计的仿真工具包，并集成于流行的Gymnasium框架之中。在NS-Gym中，我们将刻画非平稳性的环境参数演化与智能体的决策模块相互分离，从而支持对动态环境进行模块化且灵活的适配。我们回顾了该领域的先前工作，并提出了一个封装关键要素的工具包。

    arXiv:2501.09646v2 Announce Type: replace  Abstract: Many real-world applications require decision-making where the environmental dynamics evolve over time. These non-stationary environments pose significant challenges to traditional decision-making models, which typically assume stationary dynamics. Non-stationary Markov decision processes (NS-MDPs) offer a framework to model and solve decision problems under such changing conditions. However, there are no standardized simulation frameworks for NS-MDPs, as opposed to widely popular frameworks for stationary problems. We present NS-Gym, the first simulation toolkit designed explicitly for NS-MDPs, integrated within the popular Gymnasium framework. In NS-Gym, we segregate the evolution of the environmental parameters that characterize non-stationarity from the agent's decision-making module, allowing for modular and flexible adaptations to dynamic environments. We review prior work in this domain and present a toolkit encapsulating key 
    
[^400]: 基于丰富自我报告数据的AI模拟可提高准确性并减少失真

    AI Simulations Grounded in Rich Self-Reports Increase Accuracy and Reduce Distortion

    [https://arxiv.org/abs/2411.10109](https://arxiv.org/abs/2411.10109)

    该研究通过对1,052名美国成年人开展两小时AI语音深度访谈并结合问卷数据，证明基于丰富自我报告数据（访谈与调查）的AI人类模拟比仅用人口属性提示的模拟更准确，并能显著减少样本层面的八类统计失真。

    

    基于人口统计学提示构建的人类AI模拟常常误判个体，并扭曲样本层面的统计模式。我们检验了丰富的自我报告数据是否能减少这些问题。我们开发了一个AI语音访谈系统，并按照“美国之声项目”的协议对1,052名美国成年人进行了两小时的访谈，这些受访者同时还完成了问卷调查和行为实验。针对每位参与者，我们比较了分别基于以下不同提示生成的模拟结果：人口统计学属性、自我撰写的人物画像、问卷回答、访谈回答，以及问卷与访谈相结合。我们在留出的回答数据上评估了模拟的准确性，并在样本层面测量了八种统计失真：平均值偏差、方差被压缩、回答分布失真、条目间过度相关、态度坍缩为单一维度、意识形态归类、实验效应被放大，以及过度理性。在综合社会调查（GSS）的题目上，仅基于人口统计学的模拟导致了……（原文在此处截断）

    arXiv:2411.10109v4 Announce Type: replace  Abstract: AI simulations of people built from demographic prompts often misjudge individuals and distort sample-level patterns. We test whether rich self-report data reduce these problems. We developed an AI voice interviewer and conducted two-hour interviews following the American Voices Project protocol with 1,052 U.S. adults, who also completed surveys and behavioral experiments. For each participant, we compared simulations prompted with demographic attributes, self-authored personas, survey responses, interview responses, or combined surveys and interviews. We evaluated accuracy on held-out responses, and measured eight statistical distortions at the sample level: biased averages, compressed variance, distorted response distributions, over-correlated items, attitudes collapsing onto one dimension, ideological sorting, amplified experimental effects, and hyper-rationality. On General Social Survey (GSS) items, demographic simulation led to
    
[^401]: 利用狄利克雷过程高斯混合模型分析卷积神经网络学习到的特征

    Analyzing Learned Convnet Features with Dirichlet Process Gaussian Mixture Models

    [https://arxiv.org/abs/1702.07189](https://arxiv.org/abs/1702.07189)

    提出了一种利用狄利克雷过程高斯混合模型对卷积神经网络各层内部表征进行聚类的新方法，实现了对卷积网络学习特征的可视化解释，并有助于迁移学习分析。

    

    卷积神经网络近年来在一系列计算机视觉任务中取得了良好的效果。尽管备受关注，但通过可视化学习到的表征来解释卷积神经网络仍然是一项具有挑战性的任务。内部表征的高维度以及深层网络的高抽象性是可视化卷积神经网络功能时的主要挑战。本文提出了一种基于狄利克雷过程高斯混合模型对卷积神经网络内部表征进行聚类的方法，用于可视化卷积神经网络中学习到的表征。我们的方法通过对每一层所有节点的表征进行聚类来应对卷积神经网络的高维度问题。我们还将讨论这一方法在迁移学习场景中的应用价值，即把在一个数据集上训练的模型迁移到解决另一个数据集上的任务。

    arXiv:1702.07189v1 Announce Type: cross  Abstract: Convolutional Neural Networks (Convnets) have achieved good results in a range of computer vision tasks the recent years. Though given a lot of attention, visualizing the learned representations to interpret Convnets, still remains a challenging task. The high dimensionality of internal representations and the high abstractions of deep layers are the main challenges when visualizing Convnet functionality. We present in this paper a technique based on clustering internal Convnet representations with a Dirichlet Process Gaussian Mixture Model, for visualization of learned representations in Convnets. Our method copes with the high dimensionality of a Convnet by clustering representations across all nodes of each layer. We will discuss how this application is useful when considering transfer learning, i.e.\ transferring a model trained on one dataset to solve a task on a different one.
    

