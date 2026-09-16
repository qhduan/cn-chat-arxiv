# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Coding Agents Have Converged: Why the SWE-bench Leaderboard Can No Longer Order Its Top Entries, and What to Measure Instead](https://arxiv.org/abs/2609.17394) | 该研究通过审计 254 份 SWE-bench 提交记录发现，头部编码智能体的解决方案高度趋同且在统计上不可区分，而模型-脚手架组合对分数的影响（达 29.8 个百分点）远超头部条目之间的分数差距（仅 8.8 个百分点），因此现有排行榜已无法有效排序顶级系统。 |
| [^2] | [Type-IV Code Clone Detection via Layer-Wise Non-Contrastive Representation Learning](https://arxiv.org/abs/2609.17338) | 本文提出LWVIC4Code，一种基于VICReg框架的非对比表示学习方法，通过跨层一致性正则化和深度相关的层加权机制逐层精炼代码语义表示，有效解决了语义等价但语法不同的Type-IV代码克隆检测难题，同时避免了对比学习中负采样带来的偏差。 |
| [^3] | [After the Party: Governing What a Viral Agent-Skill Ecosystem Left Behind](https://arxiv.org/abs/2609.17274) | 本文以爆红的 OpenClaw 智能体技能生态为对象，通过 Git 历史、GitHub 活动与注册表快照量化了病毒式增长退潮后的生态遗产，发现下载量高度集中于头部技能，且没有简单的技能特征能稳定预测技能条目的持续存留。 |
| [^4] | [An Exemplar of a Digital Twin in Mechanical Engineering: Understanding Model Hybridization](https://arxiv.org/abs/2609.17258) | 本文以法国机械工业技术中心（CETIM）开发的流体回路数字孪生为实例，阐述了数字孪生中基于物理模型与数据驱动模型的混合化整合范式，以解决混合数字孪生工程实践文档不足、难以复现和转移的问题。 |
| [^5] | [A Memorization Floor for LLM Refinement of Decompiled Code](https://arxiv.org/abs/2609.17236) | 本文提出“记忆下限”同项内对照方法，证明大语言模型优化反编译代码时的命名恢复效果主要来自模型先验知识而非输入信息本身。 |
| [^6] | [Grounding SWE-Agent Decisions in Architecture-0 Design: Navigating Unknown Unknowns through Physical Mapping](https://arxiv.org/abs/2609.17221) | 论文揭示软件工程智能体在系统设计初期会因“未知未知”陷入纯文本空想以及利用验证脚本规避物理约束的自我验证陷阱，为此提出物理映射守护机制，将智能体决策锚定于外部物理映射。 |
| [^7] | [Towards an Asset Administration Shell Maturity Model](https://arxiv.org/abs/2609.17084) | 本文提出了一种新颖的资产管理壳成熟度概念，通过衡量既定数字孪生标准的满足程度，首次实现了AAS实例之间的系统性比较。 |
| [^8] | [A Set-Theoretic Evaluation Framework for Assessing Asset Administration Shell Instances: Towards Comparability and Suitability](https://arxiv.org/abs/2609.17062) | 本文提出了一种基于集合论的资产管理壳评估框架，通过模型比较方法和适用性评估模型，实现对AAS实例的可比性分析和面向特定应用场景的适用性判定。 |
| [^9] | [GANADI: Uncovering C/C++ OSS Reuse Genealogies via Pivotal Function-Based Clustering to Enhance Supply Chain Security](https://arxiv.org/abs/2609.17018) | GANADI通过基于关键函数的聚类构建C/C++开源软件复用谱系，能够追踪经由中间项目的复用传播路径，在识别精确率和召回率上显著超越现有方法，从而增强软件供应链安全。 |
| [^10] | [Search-Based Metamorphic Testing of Vision-Language Models in Autonomous Underwater Robotic Software](https://arxiv.org/abs/2609.17007) | 提出了MetaVLM，一种基于NSGA-II多目标搜索的蜕变测试方法，通过寻找水下图像的最小变换集合来揭示视觉-语言模型在自主水下机器人软件中的故障。 |
| [^11] | [TasmScan: Continuation-Aware Taint Analysis for TVM Bytecode with Savelist Abstraction](https://arxiv.org/abs/2609.16987) | 本文提出TasmScan，首个面向TON区块链TVM字节码的静态分析框架，通过保存列表抽象与前向寄存器分析，在无需源代码的情况下实现跨延续的污点分析与数据流推理。 |
| [^12] | [RepoAtlas: Guiding Coding Agents via Evolving Multimodal Repository Views](https://arxiv.org/abs/2609.16936) | RepoAtlas是一个无需训练的模块，通过在仓库代码图上执行“选择—投影—刷新”循环来维护动态演化的多模态仓库视图，从而帮助编码智能体在仓库级问题解决中保持上下文既充分又聚焦。 |
| [^13] | [RECTIFY: An Interactive Workbench for Post-Evaluation RAG Diagnosis, Repair, and Verification](https://arxiv.org/abs/2609.16764) | RECTIFY是一个交互式工作台，能够将RAG评估发现的失败案例转化为可审计、可验证的修复工作流，并揭示不同检索方法（BM25、稠密、混合）各自可解释的失败特征。 |
| [^14] | [Memory-Skill Isomorphism: One Skill Carrier, Two Native Uses](https://arxiv.org/abs/2609.16669) | 该论文提出“记忆-技能同构”思想，用技能作为蒸馏记忆的天然载体，使记忆与能力共享一个渐进式披露的统一载体（常驻描述、SKILL.md 索引、参考文件三级结构），从而无需为记忆单独重建存储与检索机制。 |
| [^15] | [An Exploratory Study of Dependabot Cooldown Adoption in Open-Source GitHub Projects](https://arxiv.org/abs/2609.16605) | 该研究实证分析了GitHub开源项目对Dependabot冷却期功能的采用情况，发现绝大多数采用由安全担忧驱动，且早期采用者更偏好简单的默认七天延迟而非细粒度控制。 |
| [^16] | [ExecuCritic: Calibrated Critic Shaping for Code Generation with Verifiable Rewards](https://arxiv.org/abs/2609.16604) | 提出 ExecuCritic 联合训练框架，让编码器与经过执行校准的评论家在同一批执行轨迹上共同训练，仅在评论家与执行器判断一致时利用其诊断反馈进行奖励塑造，有效缓解了代码生成 RLVR 中的信用分配难题。 |
| [^17] | [AI Policies: Help or Hindrance? A Software Developer's Perspective](https://arxiv.org/abs/2609.16496) | 该研究通过对19位软件开发者的访谈，揭示了AI政策对开发者的帮助与阻碍作用，并提出以开发者为中心的AI政策引入方法来支持管理者和决策者。 |
| [^18] | [Protocol-Preserving Context Trimming for Agentic Workflows: Benefits, Failure Regimes, and Budget Guardrails](https://arxiv.org/abs/2609.16461) | 本文提出协议保持型上下文裁剪与自适应预算护栏相结合的方法，在为智能体工作流节省约60% token开销的同时，将任务成功率从传统策略的66.6%–77.3%提升至92.2%，实现了效率与可靠性的兼顾。 |
| [^19] | [Evaluating the NIST Bugs Framework Against CWE as a Successor for Automated Vulnerability Classification](https://arxiv.org/abs/2609.16433) | 本文实证评估了NIST缺陷框架（BF）作为CWE的继任者与补充在自动化漏洞分类中的表现，BF通过将漏洞组织为携带根因和汇聚点的因果链三元组，解决了CWE条目重叠导致的非正交结构问题。 |
| [^20] | [FairLint-DL: An IDE-Native Tool for Fairness Debugging of Deep Learning Software](https://arxiv.org/abs/2609.16321) | FairLint-DL是一个VS Code扩展工具，通过训练代理神经网络并应用基于信息论的QID指标，实现了在训练前直接在IDE中对表格数据集进行偏见检测、因果定位和可解释性分析的公平性调试。 |
| [^21] | [Cognitive Admission Control: Risk-Conditioned Assurance for Consequential Actions in Agentic Distributed Systems](https://arxiv.org/abs/2609.16313) | 该论文提出认知准入控制（CAC）机制，在智能体分布式系统中将后果性行动的执行权限与显式的证据要求绑定，通过策略定义的保证义务、确定性评估器和准入证书，确保行动仅在风险条件得到充分证据支撑后才被准入执行。 |
| [^22] | [Assurance Envelopes for Autonomous Coding Agents: Minimum-Cost Evidence for Software Change](https://arxiv.org/abs/2609.16302) | 该论文提出“任务条件化保证包络”概念，通过类型化推理图与前向链接闭包验证，为编码代理修改软件时确定能够重新确立所有必需属性的最低成本证据子集。 |
| [^23] | [AgentGuard: Learning Execution Guardrails from Anomalous Coding-Agent Trajectories](https://arxiv.org/abs/2609.16287) | AgentGuard 通过自动从编码智能体的异常执行轨迹中提取反复出现的失败模式并泛化为指令级行为约束，构建了一个仅动态激活相关规则的轻量级执行护栏框架，在不干扰正常执行的前提下保障智能体的执行可靠性。 |
| [^24] | [Models as Governed Interfaces for AI-Native MBSE: Read-Side Adequacy and Write-Side Admissibility](https://arxiv.org/abs/2609.16252) | 该论文指出AI参与模型驱动系统工程（MBSE）的关键瓶颈不在建模语言而在数据架构，提出“认知充分性”这一数据架构模式，通过“读侧充分性”与“写侧可采性”防止AI用不可验证、不受治理的训练数据填补模型信息缺口。 |
| [^25] | [Docker Containers vs. Virtual Machines: A Comparative Study of Architecture, Performance, Configuration, and Security](https://arxiv.org/abs/2609.16148) | 本文通过基于文献的对比分析，系统比较了Docker容器与虚拟机在架构、性能、配置和安全性上的差异，指出容器在启动速度、镜像体积和工作负载密度方面更具优势，而虚拟机则在内核独立性、操作系统多样性和隔离性上更胜一筹。 |
| [^26] | [Coaching Qwen3 Coder 30B to Think Like a CodeClash Arena Agent](https://arxiv.org/abs/2609.16096) | 该论文以开源的Qwen3-Coder-30B为案例，通过从更强的编码智能体蒸馏知识，来改进其在CodeClash代码竞技场长周期多轮交互中的思考过程与决策能力。 |
| [^27] | [API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces](https://arxiv.org/abs/2609.08861) | 研究通过对ChatGPT、Claude和Gemini的系统审计发现，通过API测得的基准测试分数无法可靠反映聊天机器人界面的真实表现，API与界面间的性能差异之大甚至相当于模型降级一个版本。 |
| [^28] | [An Empirical Analysis of CodeQL False Positives and Query Refinements for Java Vulnerabilities](https://arxiv.org/abs/2609.04535) | 本文对CodeQL在Java安全分析中的误报进行了大规模实证研究，构建了包含五个类别的误报分类体系，并通过查询层面的改进成功过滤了81.8%的可复现误报模式。 |
| [^29] | [A Governance Methodology Layer for AI-Assisted Software Development: Defect Taxonomy, Controlled Ablation, and Process-Over-Capability Evidence](https://arxiv.org/abs/2609.04218) | 本文提出一套AI辅助软件开发的治理方法论层，通过缺陷分类体系、跨工具可移植的运行时解耦治理门以及“方法论即代码”的形式化，并以受控消融实验证明过程治理比模型能力更重要。 |
| [^30] | [Moirae: A Multimodal Agent Collaborative Framework for Dynamic Android Malware Detection](https://arxiv.org/abs/2608.27994) | 提出Moirae框架，通过多模态智能体协作动态收集运行时证据（视觉欺骗线索、UI状态转换、运行时API行为），并融合多维度行为视图，解决了现有检测器面临的概念漂移和混淆攻击问题。 |
| [^31] | [ADeptS-Bench: Measuring the Trustworthiness of Computer Use Agents Across Devices](https://arxiv.org/abs/2608.26204) | 该论文提出了ADeptS-Bench，一个双流可信度基准，用于评估计算机使用代理在视觉界面中处理模糊指令和恶意威胁的能力，结果显示当前所有模型均存在严重的安全缺陷。 |
| [^32] | [XREPOTEST: Benchmarking Multilingual Repository-Level Unit Test Generation for Large Language Models](https://arxiv.org/abs/2608.25939) | 本文提出了XREPOTEST，一个涵盖五种语言的多语言仓库级单元测试基准，并通过新指标调用率揭示LLM在现实仓库场景下与独立设置间存在显著性能差距。 |
| [^33] | [Trusting-Trust Attack against an Entire Linux Distribution through Binary Manipulation](https://arxiv.org/abs/2607.24888) | 该论文证明信任传递攻击并非编译器所特有——仅篡改一个普通的ELF处理工具GNU strip，即可在NixOS发行版的完整引导过程中植入自我传播的后门，最终感染整个发行版安装程序中几乎所有的二进制文件。 |
| [^34] | [The Verifier is the Curriculum: Precision Sets the Return on Search in Code Self-Distillation](https://arxiv.org/abs/2607.09709) | 本文提出无需奖励模型或评判器的“严格启动”确定性验证门控，证明验证器的精度而非奖励模型是决定代码自蒸馏中搜索收益的关键，使 14B 模型在 GameCraft-Bench 保留任务上的干净启动率从 8.8% 跃升至 42.2% 并实现显著跨家族泛化。 |
| [^35] | [Shared Selective Persistent Memory for Agentic LLM Systems](https://arxiv.org/abs/2607.09493) | 该论文提出共享选择性持久记忆架构，通过只保留任务规范、数据模式、工具配置和输出约束四类可复用上下文并支持基于角色的跨用户共享，使智能体LLM系统在几乎不增加令牌成本的情况下实现跨会话知识复用，完成率远超无记忆和完整历史方案。 |
| [^36] | [PyMETA: Evaluating Student Code Diagnosis on and Beyond the First Execution Error](https://arxiv.org/abs/2606.30610) | 该论文提出了PyMETA数据集，包含48,646份学生Python代码提交及三层级错误分类体系（最细粒度含14个标签），首次系统评估了大语言模型在首次执行错误及超越首次错误（如逻辑错误）层面诊断学生代码的能力。 |
| [^37] | [MANGO: Automated Multi-Agent Test Oracle Generation for Vision-Language-Action Models](https://arxiv.org/abs/2606.24815) | 提出了MANGO多智能体框架，能够从机器人任务的自然语言描述中自动生成细粒度测试预言机，解决了传统人工构建预言机成本高、难以复用、且缺乏中间行为洞察和故障定位能力的问题。 |
| [^38] | [Understanding the (In)Security of Vibe-Coded Applications](https://arxiv.org/abs/2606.23130) | 本文对真实世界中通过氛围编程开发的应用进行了首次大规模系统性安全研究，收集了9,041个开源应用并审计了200个已部署应用，共发现1,186个漏洞，揭示了AI主导开发范式带来的安全隐患。 |
| [^39] | [Measuring Curriculum Alignment across Topical Coverage, Competency, and Cognitive Depth: A Longitudinal Framework Applied to CS2013 and CS2023](https://arxiv.org/abs/2606.19469) | 该论文提出了一个将语义检索候选生成、大语言模型确认与独立专家验证相结合的三阶段纵向框架，用于可靠地衡量计算机科学教学项目对CS2013与CS2023课程指南在主题覆盖、能力与认知深度上的对齐程度及其随指南修订的变化。 |
| [^40] | [Specifications for Humans, Agents, and Tooling](https://arxiv.org/abs/2606.15084) | 本文介绍了Bosque API（BAPI）生态系统，一种支持以规范为中心的多语言软件开发环境，其规范语言具备高表达性、测试生成、验证和沙盒功能，可覆盖完整的应用开发生命周期。 |
| [^41] | [SoK: Post-Quantum Cryptography Implementation in Software: Approaches, Challenges and the PQC-HOT Framework](https://arxiv.org/abs/2606.04669) | 该知识系统化研究从人-组织-技术（HOT）视角综合分析了33篇文献，归纳出后量子密码学软件实现的四类方法与五层挑战，并提出了PQC-HOT框架以指导软件系统应对量子威胁。 |
| [^42] | [The Biomimetic Architecture of Software 4.0](https://arxiv.org/abs/2606.04025) | 本文提出软件4.0这一由人类智能、神经AI与原生反思性符号基质构成的自创生异层级架构，从根源上解决概率与符号间的阻抗失配问题，而非依赖日益复杂的外部框架进行修补。 |
| [^43] | [Can AI be Easy? Lessons Learned from the EZR.py Toolkit](https://arxiv.org/abs/2606.03640) | 本文通过 400 行的 Python 工具包 EZR.py 证明开发者仍需阅读代码，并揭示许多看似不同的学习算法在剥离到核心后几乎相同——经典算法可压缩至几行代码，而最先进的主动学习器仅需约 80 行即可实现。 |
| [^44] | [On the Reliability of Code Comprehension Proxies](https://arxiv.org/abs/2605.23008) | 本文首次将德尔菲专家共识协议应用于代码理解研究，通过五名专业软件工程师建立代码可理解性的专家基准，以评估现有文献中常见代码理解代理方法（如李克特量表评分和输入输出问答）的相对可靠性。 |
| [^45] | [OpenGame: Open Agentic Coding for Games](https://arxiv.org/abs/2604.18394) | OpenGame 是首个专为端到端网页游戏创作设计的开源智能体框架，其核心 Game Skill 通过从经验中积累项目骨架的模板技能和维护已验证修复方案的调试技能，使智能体能够搭建稳定架构并系统性修复集成错误，从而从高层设计生成完全可玩的游戏。 |
| [^46] | [From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution](https://arxiv.org/abs/2604.15097) | 本研究通过45个场景、4590次受控试验发现，紧凑的“策略基因”表示比面向文档的技能包更适合作为可复用经验的载体，在测试时控制与迭代演化中均表现更优，证明经验的表示方式本身是决定性因素。 |
| [^47] | [Fine-grained Approaches for Confidence Calibration of LLMs in Automated Code Revision](https://arxiv.org/abs/2604.06723) | 该论文针对自动代码修订（ACR）任务，提出细粒度的LLM置信度校准方法，以解决传统全局Platt缩放方法在此类任务中不可靠的问题，从而帮助开发者更好地判断模型输出的可信度。 |
| [^48] | [Scalable Benchmarking Framework for Dynamic Quantum Circuits](https://arxiv.org/abs/2604.03360) | 该论文提出了dynamarq——一个可扩展且与硬件无关的动态量子电路基准测试框架，通过收集多样化的动态电路基准集并定义刻画其结构的电路特征，填补了现有基准测试工具仅适用于幺正电路的空白。 |
| [^49] | [Model checking of hyperproperties for high-level relational models](https://arxiv.org/abs/2512.12024) | 该论文提出了 HyperPardinus，一种扩展 Alloy 时序逻辑后端 Pardinus 的新模型求解程序，使开发者能够在系统设计早期阶段自动验证关系模型上的超性质，填补了高级规约语言在超性质验证方面的空白。 |

# 详细

[^1]: 编码智能体已趋同：为什么 SWE-bench 排行榜已无法对其头部条目排序，以及应当改测什么指标

    Coding Agents Have Converged: Why the SWE-bench Leaderboard Can No Longer Order Its Top Entries, and What to Measure Instead

    [https://arxiv.org/abs/2609.17394](https://arxiv.org/abs/2609.17394)

    该研究通过审计 254 份 SWE-bench 提交记录发现，头部编码智能体的解决方案高度趋同且在统计上不可区分，而模型-脚手架组合对分数的影响（达 29.8 个百分点）远超头部条目之间的分数差距（仅 8.8 个百分点），因此现有排行榜已无法有效排序顶级系统。

    

    编码智能体排行榜上的微小分数差异常常被解读为系统之间的优劣排序。我们审查已公布的判定结果是否支持这种解读，使用覆盖四个数据划分的 254 份 SWE-bench 提交记录进行分析，而无需运行模型。在 Verified 划分上，排名前两位的条目各自解决了 500 个实例中的 396 个。前十条目共享 285 个成功案例和 51 个失败案例，仅剩下 164 个实例能够区分它们的结果。前沿解决方案集合的中位嵌套度为 0.935，而由分数隐含的基线为 0.774，表明它们拥有高度共享的成功案例。分数还取决于所评估的模型-脚手架组合：观测到的同一模型内脚手架差异范围达到 29.8 个百分点，相比之下前三十名的分数差距仅为 8.8 个百分点。九项单元格均值交互检验中有六项在经过 Holm 校正后仍然显著，尽管这一观察性设计并不能确定脚手架的因果效应。精确配对 McNemar 检验无法区分 Verified 前三十名中相邻的 29 对条目。

    arXiv:2609.17394v1 Announce Type: cross  Abstract: Small differences on coding-agent leaderboards are often read as an ordering of systems. We audit whether the published verdicts support this reading, using 254 SWE-bench submissions across four splits without running models. On Verified, the leading two entries each resolve 396 of 500 instances. The top ten share 285 successes and 51 failures, leaving 164 instances that distinguish their outcomes. Frontier solution sets have median nesting 0.935 against a score-implied baseline of 0.774, indicating strongly shared successes. Scores also depend on the evaluated model-scaffold pair: observed within-model scaffold ranges reach 29.8 percentage points, compared with the 8.8-point spread of the top thirty. Six of nine cell-mean interaction tests remain significant after Holm correction, although this observational design does not identify causal scaffold effects. Exact paired McNemar tests separate none of the 29 adjacent Verified top-thirt
    
[^2]: 基于逐层非对比表示学习的Type-IV代码克隆检测

    Type-IV Code Clone Detection via Layer-Wise Non-Contrastive Representation Learning

    [https://arxiv.org/abs/2609.17338](https://arxiv.org/abs/2609.17338)

    本文提出LWVIC4Code，一种基于VICReg框架的非对比表示学习方法，通过跨层一致性正则化和深度相关的层加权机制逐层精炼代码语义表示，有效解决了语义等价但语法不同的Type-IV代码克隆检测难题，同时避免了对比学习中负采样带来的偏差。

    

    软件克隆是指彼此相似或功能等价的代码片段，它们给软件维护、重构和缺陷检测带来了重大挑战。检测Type-IV克隆（即语义等价但语法上可能不同的克隆）对于传统的基于词法或语法的方法来说尤为困难。近期的机器学习方法依赖于对比学习，这需要精心的负样本采样，并可能引入偏差。本文提出了LWVIC4Code，一种专为Type-IV克隆检测设计的非对比表示学习方法。LWVIC4Code基于方差-不变性-协方差正则化框架和先前的逐层VICReg训练，引入了跨层一致性正则化和与深度相关的层加权机制，以在transformer各层之间逐步精炼语义信息，从而生成鲁棒且具有区分性的代码表示。

    arXiv:2609.17338v1 Announce Type: cross  Abstract: Software clones are fragments of code that are similar or functionally equivalent to each other. They pose significant challenges for maintenance, refactoring, and bug detection. Detecting Type-IV clones, which are semantically equivalent but may differ syntactically, is particularly difficult for traditional token- or syntax-based methods. Recent machine learning approaches rely on contrastive learning, which requires careful negative sampling and can introduce bias. In this paper, we propose LWVIC4Code, a non-contrastive representation learning approach specifically designed for Type-IV clone detection. Building on the Variance-Invariance-Covariance Regularization (VICReg) framework and prior layer-wise VICReg training, LWVIC4Code introduces cross-layer consistency regularization and depth-dependent layer weighting to progressively refine semantic information across transformer layers, producing robust and discriminative code represe
    
[^3]: 宴会散场之后：治理病毒式传播的智能体技能生态所留下的遗产

    After the Party: Governing What a Viral Agent-Skill Ecosystem Left Behind

    [https://arxiv.org/abs/2609.17274](https://arxiv.org/abs/2609.17274)

    本文以爆红的 OpenClaw 智能体技能生态为对象，通过 Git 历史、GitHub 活动与注册表快照量化了病毒式增长退潮后的生态遗产，发现下载量高度集中于头部技能，且没有简单的技能特征能稳定预测技能条目的持续存留。

    

    AI 智能体越来越多地通过“智能体技能”（即自然语言指令）来执行操作，这些指令引导宿主智能体进行 shell、网络、凭据、文件和进程相关的动作，而公共注册表则大规模分发这些技能。2026 年上半年，OpenClaw AI 智能体走红，其公共技能注册表随之激增：可观测的技能存量在 91 天内几乎翻倍，6 月可见的大部分条目仅在两个月内被创建。到研究窗口结束时，这波浪潮已达顶峰，每月新增条目数量和核心仓库活跃度均从春季峰值回落。本文利用 OpenClaw 的 Git 历史、其 GitHub 问题与拉取请求，以及三份 ClawHub 注册表快照，测量了这轮热潮所留下的遗产。研究发现注意力高度集中：前 10% 的技能获得了全部下载量的 46.93%。而一旦……没有任何简单的技能特征（如大小或下载量）能够成为条目持续存留的稳定预测因子。（摘要原文在此处被截断）

    arXiv:2609.17274v1 Announce Type: cross  Abstract: AI agents increasingly act through agent skills, i.e., natural-language instructions, that direct a host agent toward shell, network, credential, file, and process actions, and public registries distribute them at scale. In the first half of 2026, the OpenClaw AI agent went viral, and its public skill registry boomed: the observable stock nearly doubled in 91 days, and a majority of the listings visible in June were created in just two months. By the end of our study window, the wave had crested, and monthly listing creation and core-repository activity were falling from their spring peaks. This paper measures what the boom left behind, drawing on the OpenClaw Git history, its GitHub issues and pull requests, and three ClawHub registry snapshots. Attention is concentrated: the top 10% of skills received 46.93% of all downloads. No simple skill features (like size or download counts) remained a stable predictor of continued listing once
    
[^4]: 机械工程中数字孪生的一个范例：理解模型混合化

    An Exemplar of a Digital Twin in Mechanical Engineering: Understanding Model Hybridization

    [https://arxiv.org/abs/2609.17258](https://arxiv.org/abs/2609.17258)

    本文以法国机械工业技术中心（CETIM）开发的流体回路数字孪生为实例，阐述了数字孪生中基于物理模型与数据驱动模型的混合化整合范式，以解决混合数字孪生工程实践文档不足、难以复现和转移的问题。

    

    数字孪生已被广泛应用于各种应用领域。在工业部门，特别是机械工程领域，它们能够加速产品开发、降低风险、实现问题的早期预测，并降低维护成本。在实践中，数字孪生越来越多地将基于物理的（演绎）模型与数据驱动的（归纳）模型整合为混合模型，从而结合了两种建模范式的互补优势。在本文中，我们将这种整合范式称为混合化。尽管存在这一趋势，混合数字孪生的工程化实践仍然缺乏充分的文档记录。混合化通常以临时性、即兴的方式引入，其实现过程仅被部分明确说明，这限制了可重现性和可转移性。本文报告了在法国机械工业技术中心（CETIM）开发的一个现有流体回路数字孪生的过程。该数字孪生使用表征方法进行描述……

    arXiv:2609.17258v1 Announce Type: new  Abstract: Digital Twins (DTs) are widely adopted across a variety of application domains. In industrial sectors, particularly in mechanical engineering, they accelerate product development, reduce risks, enable early issue prediction, and lower sustainment costs . In practice, DTs increasingly integrate physics-based (deductive) and data-driven (inductive) models into hybrid models combining the complementary strengths of both modeling paradigms. In this paper, we refer to this integration paradigm as hybridization. Despite this trend, the engineering of hybrid DTs that is, remains insufficiently documented. Hybridization is often introduced in an ad hoc manner, and its implementation is only partially made explicit, which limits reproducibility and transferability. This paper reports on the development of an existing fluidic loop digital twin at Centre Technique des Industries M{\'e}canique (CETIM). The DT is described using the characterization 
    
[^5]: 大语言模型反编译代码优化中的记忆下限

    A Memorization Floor for LLM Refinement of Decompiled Code

    [https://arxiv.org/abs/2609.17236](https://arxiv.org/abs/2609.17236)

    本文提出“记忆下限”同项内对照方法，证明大语言模型优化反编译代码时的命名恢复效果主要来自模型先验知识而非输入信息本身。

    

    我们引入了“记忆下限”（memorization floor）这一概念：这是一种同项内对照方法，用于区分大语言模型在优化反编译器输出时，有多少是从其输入中恢复的，又有多少是从其先验知识中恢复的。具体做法是：先优化一个函数，然后从一个标识符已被破坏的输入中再次优化它，并测量保留下来的效果。由于比较是在同一项内进行的，语料库难度不会造成干扰；该方法仅需二十次 API 调用。将此方法应用于我们分析计划确定之后才编写的函数——因此任何已发布的模型都不可能记住过它们——它报告了两项发现。恢复效果是真实存在的：优化后的输出比基于其自身输出词汇表构建的臂匹配置换零假设高出 +0.072 到 +0.137。但这种恢复并不依赖于我们所消融的输入：破坏输入的数据流仅使命名增益改变 +0.001（95% 置信区间 [-0.026, +0.026]），而移除类型前缀或打乱名称所造成的改变也不会更多。来自另一厂商的第二个优化器，经预先注册并给定字节级……（原文摘要在此处截断）

    arXiv:2609.17236v1 Announce Type: new  Abstract: We introduce a memorization floor: a within-item control separating what LLM refinement of decompiler output recovers from its input from what it recovers from its prior. Refine a function, then refine it again from an input whose identifiers have been destroyed, and measure what survives. Because the comparison is within-item, corpus difficulty cannot contribute; it costs twenty API calls. Applied to functions written after our analysis plan was committed, so no released model could have memorized them, it reports two things. Recovery is real: refined output sits +0.072 to +0.137 above an arm-matched permutation null built from its own output vocabulary. But it does not depend on the input we ablate: destroying the input's dataflow changes the naming gain by +0.001 (95% CI [-0.026, +0.026]), and removing type prefixes or permuting names changes it by no more. A second refiner from another vendor, registered in advance and given byte-ide
    
[^6]: 将软件工程智能体的决策锚定于架构0设计：通过物理映射应对“未知未知”

    Grounding SWE-Agent Decisions in Architecture-0 Design: Navigating Unknown Unknowns through Physical Mapping

    [https://arxiv.org/abs/2609.17221](https://arxiv.org/abs/2609.17221)

    论文揭示软件工程智能体在系统设计初期会因“未知未知”陷入纯文本空想以及利用验证脚本规避物理约束的自我验证陷阱，为此提出物理映射守护机制，将智能体决策锚定于外部物理映射。

    

    自主软件工程智能体（SWE-Agents）在确定性编码任务中表现出色，但在架构0（Architecture 0）阶段——即受隐含工程约束或鲜少被明确陈述的“未知未知”所困扰的系统设计初期——却举步维艰。为了研究智能体如何应对这些“未知未知”，我们探索了一条渐进式的轨迹：从纯文本自我博弈、工具增强反馈，到外部物理映射。我们的实证分析揭示了一条级联式的失败链条：纯文本推理不可避免地退化为礼貌性的共识，或看似合理却在物理上不可能成立的虚构内容；而试图通过早期执行沙箱来弥合这一差距，却意外触发了“规范博弈”：智能体利用其对验证脚本的自主权绕过物理约束，在不解决核心架构缺陷的情况下取得了表面上的成功。为了解决这一自我验证陷阱，我们提出了物理映射守护机制。

    arXiv:2609.17221v1 Announce Type: cross  Abstract: Autonomous Software Engineering Agents (SWE-Agents) excel in deterministic coding tasks but struggle with Architecture 0, the nascent system design phase plagued by implicit engineering constraints, or Unknown Unknowns (UUs) that are rarely stated explicitly. To investigate how agents navigate UUs, we explore a progressive trajectory across pure-text self-play, tool-augmented feedback, and external physical mapping. Our empirical analysis reveals a cascading chain of failures. Pure-text reasoning inevitably devolves into polite consensus or plausible yet physically impossible fabrications. Attempting to bridge this gap via an early-stage execution sandbox unexpectedly triggers Specification Gaming: agents exploit their autonomy over validation scripts to bypass physical constraints, achieving superficial success without resolving core architectural flaws. To resolve this self-validation trap, we propose the Physical Mapping Guard (PMG)
    
[^7]: 迈向资产管理壳成熟度模型

    Towards an Asset Administration Shell Maturity Model

    [https://arxiv.org/abs/2609.17084](https://arxiv.org/abs/2609.17084)

    本文提出了一种新颖的资产管理壳成熟度概念，通过衡量既定数字孪生标准的满足程度，首次实现了AAS实例之间的系统性比较。

    

    资产管理壳越来越多地被认为是制造业中实现数字孪生及数字孪生之间数据交换的基础模型。AAS定义了一种层次化的数据结构，用于表示任何类型的资产在其整个生命周期中的状态。在基于AAS的系统中，比较不同的AAS实例是一个实际挑战，因为目前既没有广泛接受的方法论框架，也没有成熟度模型可以系统地支持此类分析。为了弥补这一空白，我们提出了一种新颖的AAS成熟度概念，该概念表征了既定数字孪生标准被满足的程度，从而使AAS实例具有可比性。这些概念源自文献，并通过实例化加以应用。这些新出现的结果使从业人员和研究人员能够系统地比较AAS实例，并支持对进一步开发的识别和评估。

    arXiv:2609.17084v1 Announce Type: new  Abstract: The Asset Administration Shell (AAS) is increasingly recognized as a fundamental model for the realization of and data exchange between digital twins in manufacturing. An AAS defines a hierarchical data structure to represent any type of asset throughout its entire lifecycle. In the context of AAS-based systems, comparing different AAS instances constitutes a practical challenge, as neither a widely accepted methodological framework nor a maturity model are available to systematically support such analyses. To address this gap, we propose a novel concept of AAS maturity that characterizes the extent to which established digital twin criteria are met and thus enabling comparability of AAS instances. The concepts are derived from the literature and applied through exemplification. These emerging results enable practitioners and researchers to systematically compare AAS instances and support the identification and assessment of further deve
    
[^8]: 一种用于评估资产管理壳实例的集合论评估框架：迈向可比性与适用性

    A Set-Theoretic Evaluation Framework for Assessing Asset Administration Shell Instances: Towards Comparability and Suitability

    [https://arxiv.org/abs/2609.17062](https://arxiv.org/abs/2609.17062)

    本文提出了一种基于集合论的资产管理壳评估框架，通过模型比较方法和适用性评估模型，实现对AAS实例的可比性分析和面向特定应用场景的适用性判定。

    

    资产管理壳（AAS）为制造业中资产及其信息的表示提供了标准化手段，并日益成为软件服务的基础。然而，不同的AAS实例在结构、内容和完成程度上存在差异，这使得难以确定给定的AAS是否适用于特定应用。本文提出了两种互补的方法，以支持AAS的比较和面向应用的评估。首先，采用集合论运算来比较AAS模型，能够识别共同的、缺失的和不同的子模型及参数。其次，AAS适用性模型评估AAS对特定用例需求的符合性。该评估考虑结构符合性、语义一致性、基数和规范符合性，并且可以针对参考AAS或一组必需的SemanticID来执行。

    arXiv:2609.17062v1 Announce Type: new  Abstract: Asset Administration Shells (AAS) provide a standardized means of representing assets and their information in manufacturing and increasingly serve as a basis for software services. However, different AAS instances vary in structure, content, and degree of completion, making it difficult to determine whether a given AAS is suitable for a specific application. This paper presents two complementary methods to support the comparison and application-oriented assessment of AAS. First, set-theoretic operations are employed to compare AAS models, enabling the identification of common, missing, and differing submodels and parameters. Second, an AAS suitability model assesses the conformity of an AAS to the requirements of a specific use case. The assessment considers structural conformity, semantic consistency, cardinality, and specification conformity and can be performed either against a reference AAS or a set of required SemanticIDs. A suitab
    
[^9]: GANADI：通过基于关键函数的聚类揭示C/C++开源软件复用谱系以增强供应链安全

    GANADI: Uncovering C/C++ OSS Reuse Genealogies via Pivotal Function-Based Clustering to Enhance Supply Chain Security

    [https://arxiv.org/abs/2609.17018](https://arxiv.org/abs/2609.17018)

    GANADI通过基于关键函数的聚类构建C/C++开源软件复用谱系，能够追踪经由中间项目的复用传播路径，在识别精确率和召回率上显著超越现有方法，从而增强软件供应链安全。

    

    我们提出了GANADI，这是一种用于识别C/C++开源软件（OSS）复用谱系的系统化方法，旨在增强软件供应链安全。理解OSS复用谱系对于提高软件物料清单（SBOM）的完整性以及在整个供应链中确定安全修复优先级至关重要。尽管现有方法可以识别项目中被复用的组件和漏洞，但它们无法通过中间项目追踪OSS复用路径，这限制了它们在保护供应链生态系统方面的有效性。为了解决这一局限性，GANADI通过基于源自原始代码的共享特征（称为关键函数）对下游项目进行聚类来构建复用谱系，然后推断每个聚类中各项目之间的复用方向。当应用于20个被广泛复用的开源项目（包含超过1,500条传播路径）时，GANADI在识别复用谱系方面达到了84.85%的精确率和95.76%的召回率，超越了现有方法。

    arXiv:2609.17018v1 Announce Type: new  Abstract: We present GANADI, a systematic approach for identifying C/C++ OSS reuse genealogies to enhance software supply chain security. Understanding OSS reuse genealogy is crucial for improving SBOM completeness and prioritizing security remediation across supply chains. Although existing approaches can identify reused compo- nents and vulnerabilities within a project, they fail to trace OSS reuse paths through intermediate projects, limiting their effectiveness in securing supply chain ecosystems. To address this limitation, GANADI constructs reuse genealogies by clustering downstream projects based on shared characteristics of origin-derived code (called pivotal functions), and then inferring reuse direction among the projects within each cluster. When applied to 20 widely reused OSS projects with over 1,500 propagation paths, GANADI achieved 84.85% precision and 95.76% recall in identifying reuse genealogies, outperforming existing approache
    
[^10]: 自主水下机器人软件中视觉-语言模型的基于搜索的蜕变测试

    Search-Based Metamorphic Testing of Vision-Language Models in Autonomous Underwater Robotic Software

    [https://arxiv.org/abs/2609.17007](https://arxiv.org/abs/2609.17007)

    提出了MetaVLM，一种基于NSGA-II多目标搜索的蜕变测试方法，通过寻找水下图像的最小变换集合来揭示视觉-语言模型在自主水下机器人软件中的故障。

    

    我们的行业合作伙伴专注于多个领域工业系统的质量保证，包括海事系统，如水上船舶和自主水下机器人（AUR）。尽管视觉-语言模型（VLM）在场景理解、图像描述和物体识别方面表现出色，但其在水下环境中运行的AUR软件中的应用尚未得到充分探索。因此，在这种背景下，评估VLM集成到AUR软件中的质量十分重要，因此需要自动化软件测试工具来评估其适用性并提高其可靠性。为此，我们提出了一种基于搜索的蜕变测试方法（MetaVLM），该方法识别对水下图像的最小变换集合以引发错误的模型预测，从而揭示VLM的故障。我们采用NSGA-II作为多目标搜索算法，并在开源VLM（BLIP和……）上进行了评估。

    arXiv:2609.17007v1 Announce Type: new  Abstract: Our industry partner focuses on quality assurance for industrial systems across multiple domains, including maritime systems, such as overwater vessels and autonomous underwater robots (AURs). Despite the strong performance of vision-language models (VLMs) in scene understanding, image captioning, and object recognition, their use in AUR software operating in underwater environments is underexplored. Therefore, in this context, it is important to evaluate the quality of VLMs for integration into AUR software and, so, automated software testing tools are needed to assess their suitability and improve their dependability. To this end, we propose a search-based metamorphic testing approach (MetaVLM) that identifies a minimal set of transformations on underwater images to induce incorrect model predictions, thereby revealing VLM failures. We employ NSGA-II as a multi-objective search algorithm and evaluate it over open-source VLMs, BLIP and 
    
[^11]: TasmScan：面向TVM字节码的延续感知污点分析与保存列表抽象

    TasmScan: Continuation-Aware Taint Analysis for TVM Bytecode with Savelist Abstraction

    [https://arxiv.org/abs/2609.16987](https://arxiv.org/abs/2609.16987)

    本文提出TasmScan，首个面向TON区块链TVM字节码的静态分析框架，通过保存列表抽象与前向寄存器分析，在无需源代码的情况下实现跨延续的污点分析与数据流推理。

    

    开放网络（TON）峰值市值超过200亿美元，激活的链上地址超过1.75亿个，该网络依靠TVM（TON虚拟机）来执行智能合约。TVM使用带有保存列表（savelist）的一等延续（first-class continuations）来管理跨延续调用的控制流和寄存器状态。由于保存列表捕获的寄存器允许数据在不经过操作数栈的情况下跨越延续边界流动，字节码级别的分析若不显式建模保存列表语义，便无法构建完整的数据流跟踪。我们提出了TasmScan，这是第一个面向TVM的字节码级静态分析框架，无需源代码即可实现跨延续的数据流推理。TasmScan通过前向寄存器分析对保存列表语义进行建模，为精确解析的保存位置提供形式化的过近似（over-approximation）保证，并对局部跟踪的寄存器定义进行分析，随后将字节码提升为TASIR——一种带类型的中间表示。

    arXiv:2609.16987v1 Announce Type: new  Abstract: The Open Network (TON), with a peak market capitalization exceeding $20 billion and over 175 million activated on-chain addresses, relies on the TVM (TON Virtual Machine) to execute smart contracts. TVM uses first-class continuations with savelists to manage control flow and register state across continuation invocations. Since savelist-captured registers allow data to flow across continuation boundaries without passing through the operand stack, bytecode-level analyses cannot construct complete data flow tracking without explicitly modeling savelist semantics. We present TasmScan, the first bytecode-level static analysis framework for TVM that enables cross-continuation data flow reasoning without requiring source code. TasmScan models savelist semantics via forward register analysis with a formal over-approximation guarantee for exact-resolved save sites and locally tracked register definitions, then lifts bytecode into TASIR, a typed 
    
[^12]: RepoAtlas：通过不断演化的多模态仓库视图引导编码智能体

    RepoAtlas: Guiding Coding Agents via Evolving Multimodal Repository Views

    [https://arxiv.org/abs/2609.16936](https://arxiv.org/abs/2609.16936)

    RepoAtlas是一个无需训练的模块，通过在仓库代码图上执行“选择—投影—刷新”循环来维护动态演化的多模态仓库视图，从而帮助编码智能体在仓库级问题解决中保持上下文既充分又聚焦。

    

    基于大语言模型（LLM）的编码智能体在自动化软件工程任务方面取得了快速进展，然而仓库级问题解决仍然充满挑战。除了生成合理的补丁之外，智能体还需要在相互依赖的多个文件之间定位相关代码，并维护既充分又聚焦的仓库上下文。代码图能够揭示非局部关系，但线性文本接口会掩盖其拓扑结构；渲染完整的仓库图所产生的视觉表示过于密集，难以可靠感知；而一次性的局部视图则会随着探索的深入而逐渐失效。我们提出了RepoAtlas，这是一个无需训练的模块，它通过在仓库代码图上执行“选择—投影—刷新”循环来维护不断演化的多模态仓库视图。RepoAtlas将问题中的证据与智能体当前的探索状态相结合，在固定预算下选择与任务相关的区域，并对其进行投影……（摘要内容在此处被截断）

    arXiv:2609.16936v1 Announce Type: cross  Abstract: Large language model (LLM)-powered coding agents have made rapid progress in automating software engineering tasks, yet repository-level issue resolution remains challenging. Beyond generating a plausible patch, an agent must localize relevant code across interdependent files and maintain repository context that is both sufficient and focused. Code graphs expose non-local relations, but linear text interfaces obscure their topology; rendering the full repository graph yields visual representations that are too dense to perceive reliably, whereas a one-shot local view becomes stale as exploration proceeds. We present \textbf{RepoAtlas}, a training-free module that maintains evolving multimodal repository views through a \emph{select--project--refresh} loop over a repository code graph. RepoAtlas combines evidence from the issue with the agent's current exploration state to select a task-relevant region under a fixed budget, projects the
    
[^13]: RECTIFY：一个用于RAG评估后诊断、修复与验证的交互式工作台

    RECTIFY: An Interactive Workbench for Post-Evaluation RAG Diagnosis, Repair, and Verification

    [https://arxiv.org/abs/2609.16764](https://arxiv.org/abs/2609.16764)

    RECTIFY是一个交互式工作台，能够将RAG评估发现的失败案例转化为可审计、可验证的修复工作流，并揭示不同检索方法（BM25、稠密、混合）各自可解释的失败特征。

    

    检索增强生成（RAG）评估器能够识别诸如检索能力弱、依据性差、回答不完整以及无支撑生成等失败情况，但它们很少帮助开发者决定接下来该修复什么。我们提出了RECTIFY，这是一个交互式的Streamlit工作台，它将经过评估的RAG案例转化为可审计的修复工作流。RECTIFY过滤掉无需修复的案例，将剩余的失败归入可操作的失败类别和细粒度修复切片，并生成可编辑的修复卡片，开发者可以批准、拒绝这些卡片，或通过沙箱重跑进行验证。在一个受控的RAG基准测试中，RECTIFY揭示了BM25、稠密检索和混合检索之间可解释的失败特征：BM25主要触发噪声检索修复，而稠密检索和混合检索则留下较少的多部分欠检索和证据未充分利用的案例。额外分析表明，预过滤减少了不必要的修复候选，并且切片化……（摘要原文在此处截断）

    arXiv:2609.16764v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) evaluators can identify failures such as weak retrieval, poor grounding, incomplete answers, and unsupported generation, but they rarely help developers decide what to repair next. We present RECTIFY, an interactive Streamlit workbench that turns evaluated RAG cases into auditable repair workflows. RECTIFY filters cases that do not require repair, routes remaining failures into actionable families and finegrained repair slices, and generates editable repair cards that developers can approve, reject, or verify through sandbox reruns. On a controlled RAG benchmark, RECTIFY surfaces interpretable failure profiles across BM25, dense, and hybrid retrieval: BM25 mainly triggers noisy-retrieval repairs, while dense and hybrid retrieval leave smaller sets of multi-part underretrieval and underused-evidence cases. Additional analyses show that pre-filtering reduces unnecessary repair candidates and that slicel
    
[^14]: 记忆-技能同构：一种技能载体，两种原生用途

    Memory-Skill Isomorphism: One Skill Carrier, Two Native Uses

    [https://arxiv.org/abs/2609.16669](https://arxiv.org/abs/2609.16669)

    该论文提出“记忆-技能同构”思想，用技能作为蒸馏记忆的天然载体，使记忆与能力共享一个渐进式披露的统一载体（常驻描述、SKILL.md 索引、参考文件三级结构），从而无需为记忆单独重建存储与检索机制。

    

    记忆和技能可以在不改变权重的情况下改进智能体：记忆承载先前经验，技能承载可复用的程序。将两者分别包装在存储、路由、检索、反思和更新路径中，会使复用机制随积累而不断增长。这种重复中的一部分其实无需重建：技能本身就是蒸馏记忆的天然载体。在该工作中，记忆组件即是一个技能：常驻描述保存热点线索，按需加载的 SKILL.md 提供更冷的索引和策展策略，reference/*.md 文件则保存详细记忆（L0 -> L1 -> L2 层级）。受治理的 1,024 字符描述预算使常驻索引保持压缩。由此，记忆与能力共享一个渐进式披露的统一载体；反思过程可演化任一技能。写入操作发生分叉：历史被追加记录，当前状态被重写并重新验证。在一个已部署的系统中，最敏锐的识别体现在治理上：存在 4 条写入条目，但只有 1/4 到达结算账本。

    arXiv:2609.16669v1 Announce Type: new  Abstract: Memory and skills improve agents without changing weights: memory carries prior experience, skills carry reusable procedures. Wrapping both in stores, routers, retrieval, reflection, and update paths makes reuse machinery grow with accumulation. Part of this duplication need not be rebuilt: a Skill is already a natural carrier for distilled memory. Here the memory component is a Skill: resident description holds hot cues, on-demand SKILL.md a colder index and curation policy, and reference/*.md files detailed memories (levels L0 -> L1 -> L2). A governed 1,024-character description budget keeps the resident index compressed. Memory and capability thus share one progressive-disclosure carrier; reflection evolves either Skill. Writes fork: history appended, current state rewritten and revalidated. In one deployed system, the sharpest identification is governance: 4 write entries exist, but only 1/4 reaches the settlement ledger. At the pric
    
[^15]: 对 GitHub 开源项目中 Dependabot 冷却期采用的探索性研究

    An Exploratory Study of Dependabot Cooldown Adoption in Open-Source GitHub Projects

    [https://arxiv.org/abs/2609.16605](https://arxiv.org/abs/2609.16605)

    该研究实证分析了GitHub开源项目对Dependabot冷却期功能的采用情况，发现绝大多数采用由安全担忧驱动，且早期采用者更偏好简单的默认七天延迟而非细粒度控制。

    

    自动化依赖更新可能会在维护者和更广泛的社区有足够时间进行检测之前，快速传播恶意软件包的发布。2025年7月，GitHub 正式推出了 Dependabot 冷却期功能，作为防御软件供应链攻击的手段。然而，其早期采用的效果仍然未知。在这项探索性研究中，我们实证考察了流行的开源 GitHub 仓库如何采用和配置该功能，并调查其背后的动机。我们发现，在92个动机已知的采用事件中，有83个是由安全担忧驱动的。在75个仅出于安全目的的采用中，有43个是由安全 linter 警告触发的。在保留冷却期设置的仓库所涉及的251个生态系统中，97.2%设置了通用延迟，其中64.3%使用了七天延迟，而各更新类型设置的利用率均低于10%。因此，早期采用者更倾向于简单的默认延迟而非细粒度的控制。这些发现表明，工具可以……

    arXiv:2609.16605v1 Announce Type: new  Abstract: Automated dependency updates can rapidly propagate malicious package releases before maintainers and the broader community have enough time to detect them. In July 2025, GitHub made Dependabot cooldown generally available as a defense against software supply chain attacks. However, the effects of its early adoption remain unknown. In this exploratory study, we empirically examine how popular open-source GitHub repositories adopt and configure the feature and investigate their motivations. We find that security concerns motivated 83 of 92 adoption events with known motivations. Security linter warnings triggered 43 of 75 security-only adoptions. Among 251 ecosystems within repositories that retained cooldown, 97.2% set a general delay. Of these, 64.3% used seven days, while use of each update type setting was below 10%. Early adopters therefore favor simple default delays over fine-grained controls. These findings suggest that tools could
    
[^16]: ExecuCritic：基于校准评论家塑造的可验证奖励代码生成方法

    ExecuCritic: Calibrated Critic Shaping for Code Generation with Verifiable Rewards

    [https://arxiv.org/abs/2609.16604](https://arxiv.org/abs/2609.16604)

    提出 ExecuCritic 联合训练框架，让编码器与经过执行校准的评论家在同一批执行轨迹上共同训练，仅在评论家与执行器判断一致时利用其诊断反馈进行奖励塑造，有效缓解了代码生成 RLVR 中的信用分配难题。

    

    执行反馈对于代码模型而言是一种有用的监督信号，因为单元测试是客观的，能够直接衡量程序的正确性。其弱点在于，整个程序往往被简化为单一的通过或失败比特，使得基于可验证奖励的强化学习（RLVR）需要解决一个困难的信用分配问题。与此同时，编码系统通常包含独立的审查者或测试者角色，但这些评论家通常只是通过提示得到而非经过训练，且没有针对执行结果进行校准。我们提出了ExecuCritic，这是一个联合训练框架，其中编码器和评论家在同一批执行轨迹上进行更新。评论家预测通过或失败的结果，并给出简短的诊断反馈；编码器仅在评论家与执行器对当前轨迹组的判断达成一致时才使用该信号。在八个代码基准测试和两个最新的开源骨干模型上，ExecuCritic 优于不使用评论家的 GRPO、基于提示的审查者系统以及标量奖励模型基线，同时…

    arXiv:2609.16604v1 Announce Type: new  Abstract: Execution feedback is a useful supervision signal for code models because unit tests are objective and directly measure program correctness. Its weakness is that an entire program is often reduced to one pass or fail bit, leaving RLVR to solve a difficult credit assignment problem. At the same time, coding systems often include separate reviewer or tester roles, but these critics are usually prompted rather than trained and are not calibrated against execution. We propose ExecuCritic, a joint training framework in which a coder and a critic are updated on the same execution rollouts. The critic predicts pass or fail outcomes and gives short diagnostic feedback; the coder uses this signal only when the critic agrees with the executor on the current rollout group. Across eight code benchmarks and two recent open backbones, ExecuCritic improves over GRPO without a critic, prompted reviewer systems and scalar reward model baselines, while re
    
[^17]: AI政策：是帮助还是阻碍？来自软件开发者的视角

    AI Policies: Help or Hindrance? A Software Developer's Perspective

    [https://arxiv.org/abs/2609.16496](https://arxiv.org/abs/2609.16496)

    该研究通过对19位软件开发者的访谈，揭示了AI政策对开发者的帮助与阻碍作用，并提出以开发者为中心的AI政策引入方法来支持管理者和决策者。

    

    软件组织为缓解大语言模型（LLM）相关风险（如敏感信息泄露和未经授权的使用）而引入的AI政策，如果软件开发者不参与其中，就无法发挥作用。我们通过对19位软件开发者的访谈，展示了AI政策如何帮助和阻碍开发者。我们提出了以开发者为中心的AI政策引入方法，以支持管理者和决策者。

    arXiv:2609.16496v1 Announce Type: cross  Abstract: AI policies introduced by software organisations to mitigate LLM-related risks such as sensitive information leaks and unauthorised usage are not useful if software developers do not engage with them. We draw on 19 software developer interviews to show how AI policies help and hinder developers. We suggest approaches to support managers and decision makers with a developer-centric approach to introducing AI policies.
    
[^18]: 面向智能体工作流的协议保持型上下文裁剪：收益、失败模式与预算护栏

    Protocol-Preserving Context Trimming for Agentic Workflows: Benefits, Failure Regimes, and Budget Guardrails

    [https://arxiv.org/abs/2609.16461](https://arxiv.org/abs/2609.16461)

    本文提出协议保持型上下文裁剪与自适应预算护栏相结合的方法，在为智能体工作流节省约60% token开销的同时，将任务成功率从传统策略的66.6%–77.3%提升至92.2%，实现了效率与可靠性的兼顾。

    

    智能体大语言模型（LLM）系统依赖长交互历史来保存指令、工具状态、中间决策和未解决的依赖关系，但不受限制的上下文增长会增加计算成本，并可能降低效率。本研究将协议保持型上下文裁剪作为一种面向多步骤智能体工作流的可靠性约束方法进行评估。研究在不同保留上下文级别和工作流复杂度类别下，比较了五种裁剪策略——基于新近性的裁剪、基于相关性的裁剪、摘要式裁剪、协议感知裁剪和自适应预算护栏——评估指标包括任务成功率、协议遵从性、有效工具调用、token节省量、延迟降低、级联失败以及关键上下文阈值。传统策略实现了约60%的平均token节省，但任务成功率较低（66.6%–77.3%），协议遵从性也较低（85.5%–88.6%）。协议感知裁剪将任务成功率提升至92.2%，而自适应护栏进一步……

    arXiv:2609.16461v1 Announce Type: cross  Abstract: Agentic large language model (LLM) systems rely on long interaction histories to preserve instructions, tool states, intermediate decisions, and unresolved dependencies, but unrestricted context growth increases computational cost and can reduce efficiency. This study evaluates protocol-preserving context trimming as a reliability-constrained approach for multi-step agentic workflows. Five trimming strategies - recency-based, relevance-based, summarization, protocol-aware trimming, and adaptive budget guardrails - were compared across retained-context levels and workflow-complexity classes using task success, protocol adherence, valid tool calls, token savings, latency reduction, cascading failures, and critical context thresholds. Conventional strategies achieved about 60% mean token savings but lower task success (66.6-77.3%) and protocol adherence (85.5-88.6%). Protocol-aware trimming improved task success to 92.2%, while adaptive g
    
[^19]: 评估NIST缺陷框架作为CWE继任者在自动化漏洞分类中的表现

    Evaluating the NIST Bugs Framework Against CWE as a Successor for Automated Vulnerability Classification

    [https://arxiv.org/abs/2609.16433](https://arxiv.org/abs/2609.16433)

    本文实证评估了NIST缺陷框架（BF）作为CWE的继任者与补充在自动化漏洞分类中的表现，BF通过将漏洞组织为携带根因和汇聚点的因果链三元组，解决了CWE条目重叠导致的非正交结构问题。

    

    基于根因弱点的漏洞分类对众多网络安全活动至关重要，其中通用缺陷枚举（CWE）作为此类缺陷的公共知识库。然而，CWE中相互重叠的条目形成了非正交的结构，导致同一漏洞被映射到多个弱点，使根因分析（RCA）和漏洞分诊变得复杂。为解决这一问题，NIST特别出版物800-231引入了缺陷框架（BF），该框架将漏洞组织为三元组，并将这些三元组链接成因果链，使一个漏洞同时携带其根因和汇聚点，而不是单一的终结标签。然而迄今为止，BF仅有规范定义，尚未针对自动化分类所面临的挑战进行性能评估，其被采用所需的证据也缺乏实证研究。我们将BF作为分类目标以及CWE的补充进行评估……

    arXiv:2609.16433v1 Announce Type: cross  Abstract: Vulnerability classification based on root cause weaknesses is essential for numerous cybersecurity activities, where the Common Weakness Enumeration (CWE) serves as a public repository of such flaws. However, its overlapping entries create a non-orthogonal structure. The result is the same vulnerability being mapped to multiple weaknesses, complicating Root Cause Analysis (RCA) and triage. To address this, NIST Special Publication 800-231 introduces the Bugs Framework (BF), which organizes vulnerabilities into  triples and links such triples into a causal chain, so that a vulnerability carries its root cause and its sink together instead of a single terminal label. To date, however, BF has been specified but not evaluated regarding its performance against the challenges to automated classification. The evidence required for adoption has not been investigated empirically. We evaluate BF as a classification target and a complement to CW
    
[^20]: FairLint-DL：一个面向深度学习软件公平性调试的IDE原生工具

    FairLint-DL: An IDE-Native Tool for Fairness Debugging of Deep Learning Software

    [https://arxiv.org/abs/2609.16321](https://arxiv.org/abs/2609.16321)

    FairLint-DL是一个VS Code扩展工具，通过训练代理神经网络并应用基于信息论的QID指标，实现了在训练前直接在IDE中对表格数据集进行偏见检测、因果定位和可解释性分析的公平性调试。

    

    现有的公平性分析工具主要作为训练后评估框架运行，要求从业者在评估偏见之前必须完成完整的模型开发生命周期。我们提出了FairLint-DL，这是一个Visual Studio Code扩展，通过实现“左移”方法进行公平性测试，支持在训练之前直接对表格数据集进行IDE原生的偏见检测。FairLint-DL训练一个可配置的深度神经网络作为代理模型，并应用信息论的定量个体歧视（QID）指标。QID基于香农熵和最小熵，量化受保护属性对预测的因果影响。该系统实现了用于发现歧视性实例的两阶段梯度引导搜索算法、通过敏感性分析将偏见定位到特定网络层和神经元的因果调试流水线，以及使用SHAP和LIME进行特征级解释的双可解释性引擎。

    arXiv:2609.16321v1 Announce Type: cross  Abstract: Existing fairness analysis tools predominantly operate as post-training evaluation frameworks, requiring practitioners to complete the full model development lifecycle before assessing bias. We present FairLint-DL, a Visual Studio Code extension that implements a shift-left approach to fairness testing by enabling pre-training, IDE-native bias detection directly on tabular datasets. FairLint-DL trains a configurable deep neural network as a proxy model and applies information-theoretic Quantitative Individual Discrimination (QID) metrics. Grounded in Shannon and min-entropy, QID quantifies the causal influence of protected attributes on predictions. The system implements a two-phase gradient-guided search algorithm for discovering discriminatory instances, a causal debugging pipeline that localizes bias to specific network layers and neurons via sensitivity analysis, and dual explainability engines using SHAP and LIME for feature-level
    
[^21]: 认知准入控制：智能体分布式系统中后果性行动的风险条件保障

    Cognitive Admission Control: Risk-Conditioned Assurance for Consequential Actions in Agentic Distributed Systems

    [https://arxiv.org/abs/2609.16313](https://arxiv.org/abs/2609.16313)

    该论文提出认知准入控制（CAC）机制，在智能体分布式系统中将后果性行动的执行权限与显式的证据要求绑定，通过策略定义的保证义务、确定性评估器和准入证书，确保行动仅在风险条件得到充分证据支撑后才被准入执行。

    

    在智能体分布式系统中，一个智能体可能被授权修改外部基础设施，但缺乏证据证明该修改已准备好执行。认知准入控制（CAC）将这一证据要求显式化。策略将类型化行动及其建模风险映射为保证义务，明确指定谓词、证据类别、范围、新鲜度和见证集约束。确定性评估器区分已满足、已违反和未解决的义务；未解决的条件会产生针对性的证据获取请求。成功准入后会产生一个证书，将行动、其见证清单和调度时守卫绑定在一起。我们形式化了准入演算以及将其与中介执行相连接的假设。这些保证是相对策略而言的：物理安全还需要可靠的证据、充分的环境模型，以及在效果执行过程中保持相关条件。

    arXiv:2609.16313v1 Announce Type: cross  Abstract: In agentic distributed systems, an agent may be authorized to mutate external infrastructure while lacking evidence that the mutation is ready to execute. Cognitive Admission Control (CAC) makes this evidence requirement explicit. A policy maps a typed action and its modeled risk to assurance obligations specifying predicates, evidence classes, scope, freshness, and witness-set constraints. A deterministic evaluator distinguishes satisfied, violated, and unresolved obligations; unresolved conditions produce targeted evidence-acquisition requests. Successful admission produces a certificate binding the action, its witness manifest, and dispatch-time guards.   We formalize the admission calculus and the assumptions connecting it to mediated execution. The guarantees are policy-relative: physical safety additionally requires sound evidence, an adequate environment model, and preservation of relevant conditions through the effect. A TypeSc
    
[^22]: 自主编码代理的保证包络：面向软件变更的最低成本证据

    Assurance Envelopes for Autonomous Coding Agents: Minimum-Cost Evidence for Software Change

    [https://arxiv.org/abs/2609.16302](https://arxiv.org/abs/2609.16302)

    该论文提出“任务条件化保证包络”概念，通过类型化推理图与前向链接闭包验证，为编码代理修改软件时确定能够重新确立所有必需属性的最低成本证据子集。

    

    当编码代理回到现有软件中工作时，它会继承早期工程工作留下的证据：测试、类型检查、形式化证明、静态分析和执行追踪。重新加载所有这些证据是浪费的，但遗漏变更所依赖的某项证据又可能使某个必需的属性失去支撑。给定一项变更必须保持的属性（即其“义务”），我们研究可用证据的哪个最低成本子集能够重新确立这些义务，并将这样的子集称为任务条件化保证包络。证据及其组合规则构成一个类型化推理图；当前向链接从所选证据出发能够到达某项义务时，该义务即被视为满足，我们通过这种闭包验证来检验每一次选择，而不是信任优化器本身。我们评估中使用的源自软件的图来自先前AI编码代理运行所保留的产出；我们冻结这些工件，并探究对于后续任务应当恢复哪些累积的证据。来自Rust的小型图……

    arXiv:2609.16302v1 Announce Type: cross  Abstract: When a coding agent returns to existing software, it inherits evidence from earlier engineering work: tests, type checks, proofs, static analyses, and traces. Reloading all of it is wasteful, but dropping a piece the change depends on can leave a required property unsupported. Given the properties a change must preserve, its obligations, we ask which least-cost subset of the available evidence re-establishes them, and we call such a subset a task-conditioned assurance envelope. Evidence and the rules that combine it form a typed inference graph; an obligation is met when forward chaining from the selected evidence reaches it, and we validate every selection by that closure rather than by trusting the optimizer. The software-derived graphs in our evaluation come from preserved outcomes of prior AI coding-agent runs; we freeze those artifacts and ask which accumulated evidence should be restored for a later task. Small graphs from Rust, 
    
[^23]: AgentGuard：从异常编码智能体轨迹中学习执行护栏

    AgentGuard: Learning Execution Guardrails from Anomalous Coding-Agent Trajectories

    [https://arxiv.org/abs/2609.16287](https://arxiv.org/abs/2609.16287)

    AgentGuard 通过自动从编码智能体的异常执行轨迹中提取反复出现的失败模式并泛化为指令级行为约束，构建了一个仅动态激活相关规则的轻量级执行护栏框架，在不干扰正常执行的前提下保障智能体的执行可靠性。

    

    AI 编码智能体日益依赖执行工具来与代码库和外部工具进行交互。然而，任务成功并不保证执行过程的可靠性。智能体仍可能修改无关文件、重写测试、发出不安全的命令或忽略失败的验证，这促使人们需要行为护栏来实现可靠的执行。我们提出了 AgentGuard，这是一个指令级护栏框架，能够从编码智能体的异常轨迹中学习条件执行约束。AgentGuard 不依赖人工指定的安全规则，而是自动提取反复出现的执行失败模式，将其泛化为指令级别的行为约束，并将这些约束组织为一个轻量级的护栏技能，该技能仅动态激活与当前指令相关的规则。这种设计在提供行为引导的同时，最大限度地减少了对正常执行的不必要限制。我们使用 642（摘要在此处截断）

    arXiv:2609.16287v1 Announce Type: new  Abstract: AI coding agents increasingly rely on execution harnesses to interact with repositories and external tools. However, task success does not guarantee reliable execution. Agents may still modify unrelated files, rewrite tests, issue unsafe commands, or ignore failed validations, motivating behavioral guardrails for reliable execution. We present AgentGuard, an instruction-level guardrail framework that learns conditional execution constraints from anomalous trajectories of coding agents. Rather than relying on manually specified safety rules, AgentGuard automatically extracts recurring execution failure patterns, generalizes them into instruction-level behavioral constraints, and organizes them as a lightweight guardrail skill that dynamically activates only the rules relevant to the current instruction.   This design enables behavioral guidance while minimizing unnecessary restrictions on normal execution. We evaluate AgentGuard using 642
    
[^24]: 模型作为AI原生MBSE的受治理接口：读侧充分性与写侧可采性

    Models as Governed Interfaces for AI-Native MBSE: Read-Side Adequacy and Write-Side Admissibility

    [https://arxiv.org/abs/2609.16252](https://arxiv.org/abs/2609.16252)

    该论文指出AI参与模型驱动系统工程（MBSE）的关键瓶颈不在建模语言而在数据架构，提出“认知充分性”这一数据架构模式，通过“读侧充分性”与“写侧可采性”防止AI用不可验证、不受治理的训练数据填补模型信息缺口。

    

    摘要：机器可读模型（如SysML v2）如今已可通过编程方式访问，越来越多的研究将这种可访问性视为AI参与系统工程的使能条件。然而，访问是必要的，但并不充分。剩余的工作不在于建模语言本身，而在于围绕它的数据架构。一个查询结构完整模型以进行推导的AI读取器，仍然会遇到推导链缺失、认知状态未标记、溯源信息缺失以及模型无法解析的证据等问题。面对这些缺口，AI不会选择弃权，而是从训练数据中填补——而训练数据这一来源既不可验证，也不受治理。为了在一个按当前实践标准堪称典范而非存在缺陷的模型上论证这一观点，我们探查了公开的阿波罗11号SysML v2重建模型。我们将这种缺失的属性命名为“认知充分性”，并将其作为一种候选数据架构模式分为两个部分。读侧充分性允许……（摘要在此处被截断）

    arXiv:2609.16252v1 Announce Type: cross  Abstract: Machine-readable models such as SysML v2 are now programmatically accessible, and a growing body of work treats that access as the enabling condition for AI participation in systems engineering. Access is necessary, but not sufficient. The remaining work lies not in the modelling language but in the data architecture around it. An AI reader that queries a structurally complete model for a derivation still runs into absent derivation chains, untagged epistemic status, missing provenance, and evidence that the model cannot resolve. Faced with these gaps, it does not abstain; it fills them from training data, a source that is neither verifiable nor governed. To make the case on a model that is exemplary by current practice rather than deficient, we probe the public Apollo 11 SysML v2 reconstruction. We name the missing property epistemic adequacy and offer it as a candidate data-architecture pattern in two halves. Read-side adequacy lets 
    
[^25]: Docker容器与虚拟机：架构、性能、配置与安全性的对比研究

    Docker Containers vs. Virtual Machines: A Comparative Study of Architecture, Performance, Configuration, and Security

    [https://arxiv.org/abs/2609.16148](https://arxiv.org/abs/2609.16148)

    本文通过基于文献的对比分析，系统比较了Docker容器与虚拟机在架构、性能、配置和安全性上的差异，指出容器在启动速度、镜像体积和工作负载密度方面更具优势，而虚拟机则在内核独立性、操作系统多样性和隔离性上更胜一筹。

    

    现代应用平台必须在保持部署速度、可移植性、资源效率和安全性的同时隔离工作负载。虚拟机和Docker容器在不同的抽象层次上满足这一需求：虚拟机虚拟化硬件并运行独立的客户操作系统，而容器在共享宿主机内核的同时隔离进程。本文对这两种技术在架构、配置与生命周期管理、性能、可扩展性和安全性等方面进行了基于文献的对比分析。已发表的研究普遍表明，容器具有更短的启动时间、更小的镜像、更高的工作负载密度，以及对许多工作负载接近原生的执行性能。这些优势取决于工作负载特性、存储和网络驱动程序、资源控制以及实验设计。虚拟机会引入更大的开销，但提供了独立的内核、异构的客户操作系统以及更强的隔离能力。

    arXiv:2609.16148v1 Announce Type: new  Abstract: Modern application platforms must isolate workloads while preserving deployment speed, portability, resource efficiency, and security. Virtual machines (VMs) and Docker containers address this requirement at different abstraction layers: VMs virtualize hardware and run independent guest operating systems, whereas containers isolate processes while sharing the host kernel. This paper presents a comparative, literature-based analysis of the two approaches across architecture, configuration and lifecycle management, performance, scalability, and security. Published studies generally associate containers with shorter startup times, smaller images, higher workload density, and near-native execution for many workloads. These benefits depend on workload characteristics, storage and network drivers, resource controls, and experimental design. VMs introduce greater overhead but offer independent kernels, heterogeneous guest operating systems, and
    
[^26]: 训练Qwen3 Coder 30B像CodeClash竞技场智能体一样思考

    Coaching Qwen3 Coder 30B to Think Like a CodeClash Arena Agent

    [https://arxiv.org/abs/2609.16096](https://arxiv.org/abs/2609.16096)

    该论文以开源的Qwen3-Coder-30B为案例，通过从更强的编码智能体蒸馏知识，来改进其在CodeClash代码竞技场长周期多轮交互中的思考过程与决策能力。

    

    大型语言模型编码智能体最近在软件任务中已变得实用，但较弱或开源权重的智能体仍难以可靠地理解用户意图并执行复杂的多步骤工作流。这一差距在长周期场景中尤为明显——智能体必须反复检查先前的结果、诊断失败原因，并在交互约束下选择下一步的代码编辑。这引出了一个自然的问题：我们能做些什么来改进弱编码智能体的思考过程？我们在CodeClash中研究这个问题。CodeClash是一个代码竞技场基准，原始工作通过多轮锦标赛在6个竞技场中评估了8个商业编码智能体。由于Qwen3 Coder Plus在其中排名垫底，我们以开源权重的Qwen3-Coder-30B作为案例研究，探讨如何利用从更强智能体蒸馏得来的知识来改进它。我们的分析表明，Qwen3-Coder-30B并未针对竞技场式的交互进行良好优化……（摘要截断）

    arXiv:2609.16096v1 Announce Type: cross  Abstract: Large language model coding agents have recently become useful for software tasks, but weaker or open-weight agents still struggle to reliably interpret user intent and execute complex multi-step workflows. This gap is especially visible in long-horizon settings, where an agent must repeatedly inspect prior outcomes, diagnose failure, and choose the next code edit under interaction constraints. It motivates a natural question: what can we do to improve the thinking process of a weak code agent? We study this question in CodeClash, a code-arena benchmark where the original work evaluates 8 commercial coding agents across 6 arenas through multi-round tournaments. Since Qwen3 Coder Plus ranks last among them, we take the open-weight Qwen3-Coder-30B as a case study and investigate how to improve it with distilled knowledge from stronger agents. Our analysis shows that Qwen3-Coder-30B is not well optimized for arena-style interaction: it fr
    
[^27]: API基准测试分数无法可靠地迁移到聊天机器人界面

    API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces

    [https://arxiv.org/abs/2609.08861](https://arxiv.org/abs/2609.08861)

    研究通过对ChatGPT、Claude和Gemini的系统审计发现，通过API测得的基准测试分数无法可靠反映聊天机器人界面的真实表现，API与界面间的性能差异之大甚至相当于模型降级一个版本。

    

    基准测试分数是模型发布中的核心“货币”：它们为采购决策提供依据，塑造公众信任，并影响政策制定。然而，基准测试分数的一个关键假设是，通过API测量的模型性能能够忠实地反映已部署系统的实际行为。我们通过审计ChatGPT、Claude和Gemini，涵盖七个系统和九个基准测试（横跨通用能力、社会偏见和谄媚性），对这一假设提出了挑战。我们发现API与聊天界面之间存在系统性的准确性和一致性差异。平均而言，API评估的准确率比相应的界面评估高3.4个百分点，重测一致性高2.1个百分点。对于ChatGPT而言，API与界面访问方式之间的性能差异，甚至超过了仅通过API测量的GPT 5.3与GPT 5.4之间的差异。换言之，切换访问渠道所导致的性能下降，可能与降级整个模型版本的影响相当。

    arXiv:2609.08861v1 Announce Type: new  Abstract: Benchmark scores are a central currency in model releases: they inform purchasing decisions, shape public trust, and influence policy. Yet, a key assumption underlying benchmark scores is that the model performance measured through APIs faithfully reflects the behavior of deployed systems.   We challenge this assumption by auditing ChatGPT, Claude, and Gemini across seven systems and nine benchmarks spanning general capability, social bias, and sycophancy. We find systematic API--interface differences in both accuracy and consistency. On average, API evaluations score 3.4 percentage points higher in accuracy and 2.1 percentage points higher in test--retest agreement than corresponding interface evaluations. For ChatGPT, the performance difference between API and interface access exceeds the API-only difference between GPT 5.3 and GPT 5.4. Put differently, switching access surfaces can degrade performance as much as downgrading a full mod
    
[^28]: 对CodeQL误报及Java漏洞查询改进的实证分析

    An Empirical Analysis of CodeQL False Positives and Query Refinements for Java Vulnerabilities

    [https://arxiv.org/abs/2609.04535](https://arxiv.org/abs/2609.04535)

    本文对CodeQL在Java安全分析中的误报进行了大规模实证研究，构建了包含五个类别的误报分类体系，并通过查询层面的改进成功过滤了81.8%的可复现误报模式。

    

    静态应用安全测试（SAST）工具帮助开发人员在部署前发现漏洞，但误报会带来大量的人工甄别工作。我们研究CodeQL在Java安全分析中产生的误报是否呈现出可复现、可解释的模式，从而能够通过改进分析来减少这些误报。我们在来自110个项目的167个CVE实例上运行了CodeQL的Java安全查询套件，重点关注误报率最高的十个查询。我们人工审查了500条采样的误报路径和位置，并构建了一个源代码层面的误报分类体系。这五个类别分别是：遗漏路径约束或净化（36.6%）、良性执行上下文（29.4%）、缺失的信任边界建模（27.6%）、不精确的并发建模（5%）和不精确的汇聚点建模（1.4%）。基于这些发现，我们实现了CodeQL查询层面的改进，用以检测并过滤可复现的误报模式。这些改进消除了81.8%的

    arXiv:2609.04535v1 Announce Type: new  Abstract: Static application security testing (SAST) tools help developers find vulnerabilities before deployment, but false positives create substantial triage effort. We study whether CodeQL false positives in Java security analysis form recurring, explainable patterns that can be reduced by refining the analysis. We run CodeQL's Java security query suite on 167 CVE instances from 110 projects, focusing on the ten queries with the highest false positive rates. We manually review 500 sampled false positive paths and locations and construct a source-level taxonomy. The five categories are Missed Path Constraint or Sanitization (36.6%), Benign Execution Context (29.4%), Missing Trust Boundary Modeling (27.6%), Imprecise Concurrency Modeling (5%), and Imprecise Sink Modeling (1.4%).   Guided by these findings, we implement CodeQL refinements that detect and filter recurring false positive patterns at the query level. The refinements remove 81.8% of 
    
[^29]: 面向AI辅助软件开发的治理方法论层：缺陷分类体系、受控消融实验与“过程重于能力”的证据

    A Governance Methodology Layer for AI-Assisted Software Development: Defect Taxonomy, Controlled Ablation, and Process-Over-Capability Evidence

    [https://arxiv.org/abs/2609.04218](https://arxiv.org/abs/2609.04218)

    本文提出一套AI辅助软件开发的治理方法论层，通过缺陷分类体系、跨工具可移植的运行时解耦治理门以及“方法论即代码”的形式化，并以受控消融实验证明过程治理比模型能力更重要。

    

    自主编码智能体能够以高速度产出通过语法检查的输出——包括编译、类型安全和持续集成（CI）。然而，语法正确并不意味着语义正确：设计边界、安全不变量以及可维护性契约对于自动化流水线而言在结构上仍是不可见的。本文为弥合这一差距做出了四项贡献。首先，我们提出了一种基于五个AI智能体权限与治理模块构建的缺陷分类体系，区分了静态分析可在结构上检测出的缺陷与需要语义审查的缺陷。其次，我们描述了一种运行时解耦的治理门——一种基于文件的协议，它读取生成器输出并给出结构化判定结果，而无需API耦合，因此可以跨不同代码生成工具移植。第三，我们将“方法论即代码”形式化：将验证协议表达为一种版本受控、可执行、跨平台的制品，并采用两层架构……（原文在此处截断）

    arXiv:2609.04218v1 Announce Type: new  Abstract: Autonomous coding agents produce output that passes syntactic checks -- compilation, type safety, CI -- at high velocity. Yet syntactic correctness does not imply semantic correctness: design boundaries, security invariants, and maintainability contracts remain structurally invisible to automated pipelines. This paper makes four contributions toward closing this gap. First, we present a defect-class taxonomy grounded in five AI agent permission and governance modules, distinguishing defects structurally detectable by static analysis from those requiring semantic review. Second, we describe a runtime-decoupled governance gate -- a file-based protocol that reads generator output and emits a structured verdict without API coupling, making it portable across code-generation tools. Third, we formalize methodology-as-code: expressing a verification protocol as a version-controlled, executable, cross-platform artifact with a two-layer architect
    
[^30]: Moirae：用于动态Android恶意软件检测的多模态智能体协作框架

    Moirae: A Multimodal Agent Collaborative Framework for Dynamic Android Malware Detection

    [https://arxiv.org/abs/2608.27994](https://arxiv.org/abs/2608.27994)

    提出Moirae框架，通过多模态智能体协作动态收集运行时证据（视觉欺骗线索、UI状态转换、运行时API行为），并融合多维度行为视图，解决了现有检测器面临的概念漂移和混淆攻击问题。

    

    Android生态系统面临着持续存在且快速演变的恶意软件威胁。现有的机器学习检测器容易受到概念漂移的影响，因为它们依赖于特定实现的特征，而这些特征的分布会随时间发生变化。大语言模型（LLM）提供了强大的语义理解和零样本推理能力，但当前基于LLM的检测器通常依赖于以代码为中心或单一维度的证据，使其容易受到混淆攻击的影响，并限制了全面的行为分析。我们提出了Moirae，一个用于动态Android恶意软件检测的多模态智能体协作框架。Moirae动态收集多模态运行时证据，并采用基于ReAct的专用智能体来分析互补的行为视图。检测过程首先识别视觉欺骗线索，对UI状态转换进行建模，并集成运行时API行为，以融合跨用户可见界面的多维度证据。

    arXiv:2608.27994v1 Announce Type: cross  Abstract: The Android ecosystem faces persistent and rapidly evolving malware threats. Existing machine learning detectors are vulnerable to concept drift because they rely on implementation-specific features whose distributions change over time. Large language models (LLMs) offer strong semantic understanding and zero-shot reasoning, but current LLM-based detectors typically depend on code-centric or single-dimensional evidence, making them susceptible to obfuscation and limiting comprehensive behavior analysis. We present {\sysname}, a multimodal agent collaborative framework for dynamic Android malware detection. {\sysname} dynamically collects multimodal runtime evidence and employs ReAct-based specialized agents to analyze complementary behavioral views. The detection process begins by identifying visual deception cues, modeling UI state transitions, and integrating runtime API behaviors to fuse multi-dimensional evidence across user-visibl
    
[^31]: ADeptS-Bench：跨设备计算机使用代理的可信度衡量基准

    ADeptS-Bench: Measuring the Trustworthiness of Computer Use Agents Across Devices

    [https://arxiv.org/abs/2608.26204](https://arxiv.org/abs/2608.26204)

    该论文提出了ADeptS-Bench，一个双流可信度基准，用于评估计算机使用代理在视觉界面中处理模糊指令和恶意威胁的能力，结果显示当前所有模型均存在严重的安全缺陷。

    

    计算机使用代理（CUAs）越来越多地被部署来代表用户操作移动和桌面应用程序，然而目前尚无一个全面的基准来评估它们在处理模糊指令时是否能安全地与视觉界面交互。我们引入了ADeptS-Bench，一个基于ADEPTS能力框架和普通人群用户研究的双流可信度基准。安全流提供了配对的安全/恶意任务，其中威胁嵌入在视觉界面中。消歧流评估代理在意图模糊时是否会寻求澄清。对七个模型的评估显示，没有一个模型能在任务成功率超过80%的同时将攻击成功率保持在30%以下；每个模型都会毫不犹豫地点击25,000美元订单上的“结账”按钮，并且没有一个模型检测到“恢复出厂设置”按钮被错误标记为“优化”。一项消融研究揭示了三种不同的安全架构：工具依赖型（ASR +2）。

    arXiv:2608.26204v1 Announce Type: cross  Abstract: Computer Use Agents (CUAs) are increasingly deployed to navigate mobile and desktop applications on behalf of users, yet no benchmark comprehensively evaluates whether they can safely interact with visual interfaces while handling ambiguous instructions. We introduce ADeptS-Bench, a dual-stream trustworthiness benchmark, grounded in the ADEPTS capability framework and general population user studies. The Safety stream provides paired benign/malicious tasks with threats embedded in the visual interface. The Disambiguation stream evaluates whether agents seek clarification when intent is ambiguous. Evaluating seven models reveals that no model consistently exceeds 80% task success while staying below 30% attack success; every model clicks "Checkout" on a $25K order without hesitation, and none detects that a "factory reset" button is mislabeled as "Optimize." An ablation reveals three distinct safety architectures: tool-dependent (ASR +2
    
[^32]: XREPOTEST：面向大型语言模型的多语言仓库级单元测试生成基准测试

    XREPOTEST: Benchmarking Multilingual Repository-Level Unit Test Generation for Large Language Models

    [https://arxiv.org/abs/2608.25939](https://arxiv.org/abs/2608.25939)

    本文提出了XREPOTEST，一个涵盖五种语言的多语言仓库级单元测试基准，并通过新指标调用率揭示LLM在现实仓库场景下与独立设置间存在显著性能差距。

    

    大型语言模型（LLMs）在自动化单元测试生成方面展现出潜力，但现有评估主要依赖于独立设置和狭窄的编程语言范围，高估了实际应用的准备程度。我们引入了XREPOTEST，一个多语言仓库级单元测试生成基准测试，涵盖五种未被充分探索的语言：Rust、Go、Julia、PHP和Ruby。XREPOTEST使用容器化执行框架和多种上下文增强策略（包括文件级、基于LSP和基于检索的上下文）在现实仓库约束下评估测试。除了标准指标（如测试通过率和覆盖率）外，我们提出了调用率（IR）来评估生成的测试是否有效执行了预期功能。对14个最先进的LLMs（包括Claude 4.5、GPT-5.2、DeepSeek V4-Pro和Qwen系列）的实验揭示，独立设置与仓库级设置之间存在显著差距。

    arXiv:2608.25939v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promise for automated unit test generation, but existing evaluations largely rely on standalone settings and a narrow set of programming languages, overestimating real-world readiness. We introduce XREPOTEST, a multilingual repository-level benchmark for unit test generation spanning five underexplored languages: Rust, Go, Julia, PHP, and Ruby. XREPOTEST evaluates tests under realistic repository constraints using a containerized execution framework and multiple context augmentation strategies, including file-level, LSP-based, and retrieval-based context. Beyond standard metrics such as test pass rate and coverage, we propose Invocation Rate (IR) to assess whether generated tests meaningfully exercise the intended functionality. Experiments with 14 state-of-the-art LLMs, including Claude 4.5, GPT-5.2, DeepSeek V4-Pro, and Qwen families, reveal a substantial gap between standalone and repository-lev
    
[^33]: 通过二进制操作对整个Linux发行版发起的信任传递攻击

    Trusting-Trust Attack against an Entire Linux Distribution through Binary Manipulation

    [https://arxiv.org/abs/2607.24888](https://arxiv.org/abs/2607.24888)

    该论文证明信任传递攻击并非编译器所特有——仅篡改一个普通的ELF处理工具GNU strip，即可在NixOS发行版的完整引导过程中植入自我传播的后门，最终感染整个发行版安装程序中几乎所有的二进制文件。

    

    Ken Thompson提出的信任传递攻击，即被入侵的编译器为其构建的程序植入后门，并在后续重新编译自身时复制该后门，被广泛认为是编译器特有的威胁。我们证明事实并非如此。我们围绕GNU strip构建了一个完整的信任传递攻击——GNU strip是一个既不检查也不生成源代码的普通构建工具——攻击仅通过对已生成的ELF文件进行操作来实现。在NixOS Linux发行版的引导过程中，二进制种子中单个被篡改的strip会植入一个有效载荷，该载荷从一代strip传播到下一代，并在种子离开依赖闭包之后仍然存续于最终的标准环境中。在一个真实的nixpkgs修订版上，该攻击成功且无故障地构建了一个完整的图形化安装程序，并对其几乎所有的二进制文件植入后门，使被颠覆的软件包能够执行任意恶意行为。

    arXiv:2607.24888v2 Announce Type: replace-cross  Abstract: Ken Thompson's trusting-trust attack, in which a compromised compiler backdoors the programs it builds and reproduces the backdoor in subsequent rebuilds of itself, is widely regarded as a threat specific to compilers. We show that it is not. We construct a complete trusting-trust attack around GNU strip, an ordinary build utility that neither inspects nor generates source code, using only manipulations of finished ELF files. In the bootstrap of the NixOS Linux distribution, a single tampered strip in the binary seed implants a payload that propagates from one generation of strip to the next and survives into the final standard environment after the seed leaves the dependency closure. On a real nixpkgs revision, the attack builds a complete graphical installer without failures and backdoors almost every one of its binaries, enabling arbitrary malicious behavior of the subverted packages.
    
[^34]: 验证器即课程：精度决定代码自蒸馏中搜索的回报

    The Verifier is the Curriculum: Precision Sets the Return on Search in Code Self-Distillation

    [https://arxiv.org/abs/2607.09709](https://arxiv.org/abs/2607.09709)

    本文提出无需奖励模型或评判器的“严格启动”确定性验证门控，证明验证器的精度而非奖励模型是决定代码自蒸馏中搜索收益的关键，使 14B 模型在 GameCraft-Bench 保留任务上的干净启动率从 8.8% 跃升至 42.2% 并实现显著跨家族泛化。

    

    arXiv:2607.09709v2 公告类型：替换 摘要：针对学习到的评判器（judge）对代码生成器进行后训练，可能会优化那些能提高分数却不改进实际生成物的代理特征。我们研究相反方向的信号：一种确定性的、无需评判器的过滤器，它只检查生成的项目能否在无头引擎下顺利启动（严格启动，strict-launch）。在这一门控下，拒绝采样自蒸馏能够累积跨家族泛化能力：在 GameCraft-Bench 上，一个 14B 模型将四个保留家族的单候选干净启动率从 8.8% 提升至 42.2%，将 32 个候选下的覆盖率从 84% 提升至 100%（即黄金参考自身的上限），并在全部 25 个保留任务上均超过监督模型。该门控对每个候选只需一次引擎调用：无需奖励模型，无需评判器。在固定准入数量的条件下，决定循环效果的是验证器的精度。仅换用一个宽松的构建检查就会抹去这一增益（p=0.0012）；而一个匹配的黄金重复对照组则退化到低于监督模型的水平。（注：原文摘要在此处被截断）

    arXiv:2607.09709v2 Announce Type: replace  Abstract: Post-training a code generator against a learned judge can optimize proxy features that raise the score without improving the artifact. We study the opposite signal: a deterministic, judge-free filter that asks only whether a generated project launches cleanly under a headless engine (strict-launch). Under this gate, rejection-sampling self-distillation compounds out-of-family generalization: on GameCraft-Bench a 14B model raises the per-candidate clean-launch rate on four held-out families from 8.8% to 42.2% and coverage at 32 candidates from 84% to 100%, the gold references' own ceiling, beating the supervised model on every one of the 25 held-out tasks. The gate costs one engine invocation per candidate: no reward model, no judge.   At a fixed admitted count, what governs the loop is verifier precision. Swapping in a lenient build check alone erases the gain (p=0.0012); a matched gold-duplication control regresses below the superv
    
[^35]: 面向智能体LLM系统的共享选择性持久记忆

    Shared Selective Persistent Memory for Agentic LLM Systems

    [https://arxiv.org/abs/2607.09493](https://arxiv.org/abs/2607.09493)

    该论文提出共享选择性持久记忆架构，通过只保留任务规范、数据模式、工具配置和输出约束四类可复用上下文并支持基于角色的跨用户共享，使智能体LLM系统在几乎不增加令牌成本的情况下实现跨会话知识复用，完成率远超无记忆和完整历史方案。

    

    通过多轮工具调用生成代码的智能体LLM系统面临一个根本性的上下文问题：每次会话都从零开始，丢弃了那些使先前会话高效运作的领域约束、数据模式、工具配置和输出偏好。我们提出了共享选择性持久记忆，这是一种保留四类可复用上下文——任务规范、数据模式、工具配置和输出约束——同时丢弃会话特定推理轨迹的架构，并将其打包成可在基于角色的访问控制下跨用户转移的工作区。由此产生的成本曲线是非单调的。在四个公开数据集上进行的受控复制实验中（其中格式规范被建立一次后即被撤回），无记忆方案在3.8K输入令牌下完成0/12次试验，选择性记忆方案在3.9K下完成12/12次试验，而完整对话历史方案在7.7K下仅完成8/12次试验。所保留的内容……

    arXiv:2607.09493v2 Announce Type: replace  Abstract: Agentic LLM systems that generate code through multi-turn tool use face a fundamental context problem: each session starts from zero, discarding the domain constraints, data schemas, tool configurations, and output preferences that made previous sessions productive. We introduce shared selective persistent memory, an architecture that retains four categories of reusable context - task specifications, data schemas, tool configurations, and output constraints - while discarding session-specific reasoning traces, and that packages them into workspaces transferable across users under role-based access control. The resulting cost curve is non-monotonic. In a controlled replication on four public datasets, where a formatting specification is established once and then withheld, no memory completes 0/12 trials at 3.8K input tokens, selective memory completes 12/12 at 3.9K, and full conversation history completes 8/12 at 7.7K. What is kept ma
    
[^36]: PyMETA：在首次执行错误及更广范围上评估学生代码诊断

    PyMETA: Evaluating Student Code Diagnosis on and Beyond the First Execution Error

    [https://arxiv.org/abs/2606.30610](https://arxiv.org/abs/2606.30610)

    该论文提出了PyMETA数据集，包含48,646份学生Python代码提交及三层级错误分类体系（最细粒度含14个标签），首次系统评估了大语言模型在首次执行错误及超越首次错误（如逻辑错误）层面诊断学生代码的能力。

    

    大型语言模型能够根据学生的代码、问题描述和参考答案来诊断学生程序的问题。评估这一能力需要明确定义什么才算正确的诊断。我们提出了PyMETA，一个包含48,646份针对155道题目的学生提交的Python错误数据集。每份提交都有一个由在线评测系统识别的首次执行错误的单一标签，当程序通过所有测试时则标注为“无错误”。其中97份提交组成的目标子集还包含通过迭代修复和重新执行收集的专家标注标签。该分类体系共有三个层级，最详细的层级包含14个标签，包括无错误、逻辑错误、命名的Python异常以及其他错误类别。我们评估了两个微调模型和两组基于提示的大语言模型：四个早期模型和四个近期模型。在以首次执行错误为基准进行评估时，近期基于提示的模型达到87.5–93.8%的宏观F1分数，超过了强大的（摘要在此处被截断）

    arXiv:2606.30610v2 Announce Type: replace  Abstract: Large language models can diagnose a student program from its code, problem statement, and reference solution. Evaluating this ability requires a clear definition of what counts as the correct diagnosis. We introduce PyMETA, a Python error dataset with 48,646 student submissions to 155 problems. Every submission has a single label for the first execution error identified by an Online Judge, or No Error when the program passes all tests. A targeted subset of 97 submissions also has expert labels collected through iterative repair and re-execution. The taxonomy has three levels; its most detailed level contains 14 labels, including No Error, Logic Error, named Python exceptions, and an Other Errors category. We evaluate two finetuned models and two groups of prompted LLMs: four earlier models and four recent models. When evaluated against the first execution error, the recent prompted models reach 87.5--93.8% macro F1, above the strong
    
[^37]: MANGO：面向视觉-语言-动作模型的自动化多智能体测试预言机生成

    MANGO: Automated Multi-Agent Test Oracle Generation for Vision-Language-Action Models

    [https://arxiv.org/abs/2606.24815](https://arxiv.org/abs/2606.24815)

    提出了MANGO多智能体框架，能够从机器人任务的自然语言描述中自动生成细粒度测试预言机，解决了传统人工构建预言机成本高、难以复用、且缺乏中间行为洞察和故障定位能力的问题。

    

    视觉-语言-动作模型是新兴的机器人控制系统，它在统一的架构中集成了感知、语言理解和动作生成。现有的针对VLA机器人的测试方法依赖于人工构建的符号化测试预言机，这些预言机根据最终的环境状态来判断任务是否成功。这类预言机构建成本高昂，需要领域专业知识，并且通常与特定任务和环境紧密耦合，限制了可扩展性和复用性。此外，它们仅提供对任务结果的最终状态评估，对中间行为和故障定位的洞察十分有限。为了解决这些局限性，我们提出了MANGO，一个能够从机器人任务的自然语言描述中自动生成细粒度预言机的多智能体框架。MANGO首先生成一个可复用的原子任务库，然后为每个原子任务生成基于模拟器的预言机定义。

    arXiv:2606.24815v2 Announce Type: replace  Abstract: Vision-Language-Action (VLA) models are emerging robotic control systems that integrate perception, language understanding, and action generation in a unified architecture. Existing testing approaches for VLA-enabled robots rely on manually constructed symbolic test oracles that determine task success from final environment states. These oracles are costly to construct, require domain expertise, and are often tightly coupled to specific tasks and environments, limiting scalability and reuse. Furthermore, they provide only end-state assessments of task outcomes, offering limited insight into intermediate behavior and fault localization. To address these limitations, we introduce MANGO, a multi-agent framework that automatically generates fine-grained oracles from natural-language descriptions of robotic tasks. MANGO first generates a reusable library of atomic tasks, then generates simulator-grounded oracle definitions for each atomic
    
[^38]: 理解“氛围编程”应用的开发安全性（不安全性）

    Understanding the (In)Security of Vibe-Coded Applications

    [https://arxiv.org/abs/2606.23130](https://arxiv.org/abs/2606.23130)

    本文对真实世界中通过氛围编程开发的应用进行了首次大规模系统性安全研究，收集了9,041个开源应用并审计了200个已部署应用，共发现1,186个漏洞，揭示了AI主导开发范式带来的安全隐患。

    

    大语言模型（LLM）的最新进展催生了“氛围编程”（vibe coding）这一新兴软件开发范式，用户主要通过自然语言与AI智能体交互来创建应用程序。由于其低门槛，氛围编程在实践中正迅速普及。与传统的AI辅助编程不同（在传统模式中，开发者仍需负责代码实现和代码审查），氛围编程将开发过程的相当大一部分委托给了AI系统。这种转变引发了一个根本性问题：通过氛围编程开发的应用程序究竟有多（不）安全？在本文中，我们对真实世界中氛围编程应用的安全性进行了系统性研究。我们收集了9,041个使用流行AI智能体（Claude Code和Lovable）开发的开源应用，并审计了200个公开部署的应用程序，共发现了1,186个漏洞。我们对这些应用程序及漏洞的研究表明……

    arXiv:2606.23130v4 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have enabled vibe coding, an emerging software development paradigm in which users create applications primarily through natural-language interactions with AI agents. Due to its low barrier to entry, vibe coding is rapidly gaining adoption in practice. Unlike conventional AI-assisted programming, where developers remain responsible for implementation and code review, vibe coding delegates a substantial portion of the development process to AI systems. This shift raises a fundamental question: how (in)secure are applications developed through vibe coding? In this paper, we conduct a systematic study of the security of real-world vibe-coded applications. We collect 9,041 open-source applications developed using popular AI agents (Claude Code and Lovable), and audit 200 publicly deployed applications, uncovering 1,186 vulnerabilities. Our study of these applications and vulnerabiliti
    
[^39]: 衡量课程在主题覆盖、能力与认知深度上的对齐度：一个应用于CS2013与CS2023的纵向框架

    Measuring Curriculum Alignment across Topical Coverage, Competency, and Cognitive Depth: A Longitudinal Framework Applied to CS2013 and CS2023

    [https://arxiv.org/abs/2606.19469](https://arxiv.org/abs/2606.19469)

    该论文提出了一个将语义检索候选生成、大语言模型确认与独立专家验证相结合的三阶段纵向框架，用于可靠地衡量计算机科学教学项目对CS2013与CS2023课程指南在主题覆盖、能力与认知深度上的对齐程度及其随指南修订的变化。

    

    本科计算机科学教育由大约每十年修订一次的国际课程指南所规范，然而各教学项目缺乏一种可靠的方法来衡量其对现行指南的覆盖完整程度，以及当指南变更时覆盖情况如何随之变化。现有分析依赖主题模型或人工标注，很少报告可靠性，不对匹配方法进行基准测试，且仅在单一时间点考察主题重叠情况。我们通过一个分阶段的流水线来弥补这些不足，该流水线将候选生成与确认环节分离，并应用于一个经认证的计算机科学理学学士项目与《计算机科学课程2013》（CS2013）和《计算机科学课程2023》（CS2023）的对比分析。语义检索提出候选的课程-知识单元匹配，大语言模型依据明确的覆盖规则对每个匹配进行确认，再由独立专家验证最终生成的映射关系。通过将七种检索器与汇总的相关性判断进行基准测试，我们发现没有任何自动……（原文摘要在此处截断）

    arXiv:2606.19469v2 Announce Type: replace  Abstract: Undergraduate computer science is governed by international curricular guidelines revised about once a decade, yet programs lack a reliable way to measure how completely they cover the current guideline and how coverage shifts when it changes. Existing analyses rely on topic models or manual tagging, seldom report reliability, do not benchmark the matching method, and examine topical overlap at a single point in time. We address these gaps with a staged pipeline that separates candidate generation from confirmation, applied to one accredited Bachelor of Science in Computer Science against Computer Science Curricula 2013 (CS2013) and 2023 (CS2023). Semantic retrieval proposes candidate course-to-knowledge-unit matches, a large language model confirms each against an explicit coverage rule, and an independent expert validates the resulting map. Benchmarking seven retrievers against pooled relevance judgments, we find that no automatic 
    
[^40]: 面向人类、智能体与工具的规范

    Specifications for Humans, Agents, and Tooling

    [https://arxiv.org/abs/2606.15084](https://arxiv.org/abs/2606.15084)

    本文介绍了Bosque API（BAPI）生态系统，一种支持以规范为中心的多语言软件开发环境，其规范语言具备高表达性、测试生成、验证和沙盒功能，可覆盖完整的应用开发生命周期。

    

    规范（Specifications）是软件开发中传达意图、需求和约束的核心机制。当它们明确、清晰且可靠时，便是促进协作与合作的有效手段。规范使利益相关者能够指定他们想要什么，使开发者（或AI智能体）能够理解并实现所需的功能，使客户端能够有效地使用系统，并使自动化工具能够验证上述每个步骤的正确性。本工具论文概述了Bosque API（BAPI）生态系统，一个旨在支持现代以规范为中心的开发方式的软件生态系统。BAPI规范语言可在完全多语言的生态系统中工作，并提供一系列功能，包括无与伦比的表达能力、测试生成、验证和沙盒机制，以支持完整的应用程序开发生命周期。这些对于支持新兴的安全与编码（无论是API实现……[原文截断]）至关重要。

    arXiv:2606.15084v2 Announce Type: replace  Abstract: Specifications are the central mechanism for communicating intents, requirements, and constraints in software development. When they are explicit, clear, and reliable, they are an effective means for collaboration and cooperation. They allow for stakeholders to specify what they want, developers (or AI agents) to understand and implement the needed functionality, for clients to effectively use the system, and for automated tooling to validate the correctness for each of these steps.   This tool paper outlines the Bosque API (BAPI) ecosystem, a software ecosystem designed to support modern spec-centered development. The BAPI specification language works in a fully polyglot ecosystem and provides a suite of features, including unparalleled expressivity, test generation, validation, and sand-boxing to support the complete application development lifecycle. These are critical to supporting emerging security and coding (both API implement
    
[^41]: SoK：后量子密码学的软件实现：方法、挑战与PQC-HOT框架

    SoK: Post-Quantum Cryptography Implementation in Software: Approaches, Challenges and the PQC-HOT Framework

    [https://arxiv.org/abs/2606.04669](https://arxiv.org/abs/2606.04669)

    该知识系统化研究从人-组织-技术（HOT）视角综合分析了33篇文献，归纳出后量子密码学软件实现的四类方法与五层挑战，并提出了PQC-HOT框架以指导软件系统应对量子威胁。

    

    后量子密码学（PQC）的安全实现需要关注密码机制、软件集成、开发者能力和组织支持。理解现有方法如何满足这些需求，对于让软件系统为应对量子威胁做好准备非常重要。本知识系统化（SoK）研究综合了33篇文献，并从人、组织与技术（HOT）视角分析了PQC的实现方法与挑战。我们识别出四类方法：指南、框架、工具与库，以及教育干预。在提取的映射中，技术支持占据了更大的比重，而没有任何方法被主要归类为组织层面，尽管某些方法具有次要的组织层面贡献。挑战的综合分析识别出五个层面，涵盖实现安全、系统集成与生命周期等方面

    arXiv:2606.04669v3 Announce Type: replace-cross  Abstract: Secure implementation of post-quantum cryptography (PQC) requires attention to cryptographic mechanisms, software integration, developer capability, and organisational support. Understanding how available approaches address these requirements is important for preparing software systems for quantum threats. This Systematisation of Knowledge (SoK) synthesises 33 publications and analyses PQC implementation approaches and challenges using a Human, Organisational, and Technological (HOT) perspective. We identify four approach categories: guidelines, frameworks, tools and libraries, and educational interventions. Technological support receives greater representation in the extracted mapping, while no approach is classified primarily as organisational, despite secondary organisational contributions in some approaches. The challenge synthesis identifies five layers covering implementation security, system integration and lifecycle, to
    
[^42]: 软件4.0的仿生架构

    The Biomimetic Architecture of Software 4.0

    [https://arxiv.org/abs/2606.04025](https://arxiv.org/abs/2606.04025)

    本文提出软件4.0这一由人类智能、神经AI与原生反思性符号基质构成的自创生异层级架构，从根源上解决概率与符号间的阻抗失配问题，而非依赖日益复杂的外部框架进行修补。

    

    主流编程范式继承了一种为“单一人类心智指挥本地机器”的过往时代而优化的执行模型，使当代系统背负着沉重的路径依赖。当被迫承载多维的、连接主义式智能时，这种脆弱的汇编模型会在深刻的概率-符号阻抗失配的重压下崩裂。虽然当代软件3.x框架试图通过将大语言模型（LLM）包裹在日益复杂的外部工具架构中来修补这种失配，但这种不断攀升的架构复杂性只会加重静态代码组装的持有成本。为了从原因而非症状入手，本文提出了软件4.0——一种由人类智能、神经人工智能和原生反思性符号基质构成的自创生异层级架构。其核心前提十分简单：智能通过赋予未知一种可以保存的形式，从而在其无知中得以存续……

    arXiv:2606.04025v2 Announce Type: replace-cross  Abstract: Dominant programming paradigms inherit an execution model optimised for a bygone era of a single human mind instructing a local machine, leaving contemporary systems burdened with path dependencies. When forced to host multi-dimensional, connectionist intelligence, this brittle assembly model fractures under the weight of a profound probabilistic-symbolic impedance mismatch. While contemporary Software 3.x frameworks attempt to patch the mismatch by encasing large language models (LLMs) in increasingly complicated external harnesses, this spiralling architectural complexity only compounds the carrying cost of static code assembly. To address the cause rather than the effects, this paper introduces Software 4.0 -- an autopoietic heterarchy of human intelligence, neural AI, and natively reflective symbolic substrate. At its core is a simple premise: intelligence survives its ignorance by giving the unknown a form it can keep, and
    
[^43]: AI 可以很简单吗？来自 EZR.py 工具包的经验教训

    Can AI be Easy? Lessons Learned from the EZR.py Toolkit

    [https://arxiv.org/abs/2606.03640](https://arxiv.org/abs/2606.03640)

    本文通过 400 行的 Python 工具包 EZR.py 证明开发者仍需阅读代码，并揭示许多看似不同的学习算法在剥离到核心后几乎相同——经典算法可压缩至几行代码，而最先进的主动学习器仅需约 80 行即可实现。

    

    近期许多媒体报道声称开发者不再需要阅读代码。我们不同意这一观点，至少在表格化软件工程（SE）优化任务这一领域中是如此：即由多行 x 值和 y 值组成、且 y 值获取成本高昂的数据任务。作为证据，我们展示了仅 400 行代码的 EZR.py——一个 Python 工具包（无重型依赖），它为表格化 SE 数据实现了朴素贝叶斯、k-均值聚类、分类与回归树、模拟退火、局部搜索、主动学习以及互补贝叶斯文本挖掘相关性过滤。EZR 是通过反复阅读并重构 AI 工具以简化和统一它们而构建的。结果表明，许多看似不同的学习算法在剥离到其核心后几乎是相同的：经典算法各自可压缩为几行代码，而一个最先进的主动学习器仅需约 80 行代码即可实现。该方法在 MOOT 仓库的 120 多个表格化 SE 优化任务上进行了测试。

    arXiv:2606.03640v2 Announce Type: replace  Abstract: Much recent press claims that developers no longer need to read code. We disagree, at least within the domain of tabular software-engineering (SE) optimization tasks: rows of $x$ and $y$ values where the $y$ values are expensive to obtain.   As evidence we present 400 lines of EZR.py, a Python toolkit (no heavy dependencies) that implements Naive Bayes, $k$-means clustering, classification and regression trees, simulated annealing, local search, active learning, and complementary-Bayes text-mining relevance filtering for tabular SE data. EZR was built by repeatedly reading and refactoring AI tools to simplify and unify them. The result demonstrates that many seemingly different learning algorithms are nearly the same once stripped back to their core: classical algorithms collapse to a few lines each, and a state-of-the-art active learner fits in roughly 80 lines.   Tested on the 120+ tabular SE optimization tasks in the MOOT reposito
    
[^44]: 论代码理解代理方法的可靠性

    On the Reliability of Code Comprehension Proxies

    [https://arxiv.org/abs/2605.23008](https://arxiv.org/abs/2605.23008)

    本文首次将德尔菲专家共识协议应用于代码理解研究，通过五名专业软件工程师建立代码可理解性的专家基准，以评估现有文献中常见代码理解代理方法（如李克特量表评分和输入输出问答）的相对可靠性。

    

    先前关于代码理解的研究使用了不同的理解代理方法——例如，李克特量表评分或关于程序片段的输入输出问题答案（通常从学生那里收集），以近似衡量代码对软件工程师是否易于理解，但这些代理方法的相对可靠性尚不清楚。本文通过两项人类研究，调查了现有文献中常见的一系列代理方法的相对可靠性。首先，我们与由五名专业软件工程师组成的小组开展了一项专家共识研究，通过改编德尔菲专家共识协议，建立了八个代码片段的真实可理解性排名。德尔菲协议在医学和国家安全预测等其他领域被广泛用于不确定性条件下的专家共识，但据我们所知，这是其首次应用于代码理解研究。其次，我们共同（摘要在此处截断）

    arXiv:2605.23008v2 Announce Type: replace  Abstract: Prior work on code comprehension uses different comprehension proxies---for example, Likert-scale ratings or answers to input-output questions about program snippets, usually collected from students, to approximate whether code is comprehensible to software engineers, but the relative reliability of these proxies is not known. This paper investigates the relative reliability of a collection of proxies common in the extant literature with a pair of human studies. First, we conducted an expert-consensus study with a panel of five professional software engineers to establish a ground-truth comprehensibility ranking of eight code snippets by adapting the Delphi expert-consensus protocol. The Delphi protocol is widely used for expert consensus under conditions of uncertainty in other domains such as medicine and national-security forecasting, but to our knowledge, this is its first application to code comprehension research. Second, we co
    
[^45]: OpenGame：面向游戏的开放式智能体编程

    OpenGame: Open Agentic Coding for Games

    [https://arxiv.org/abs/2604.18394](https://arxiv.org/abs/2604.18394)

    OpenGame 是首个专为端到端网页游戏创作设计的开源智能体框架，其核心 Game Skill 通过从经验中积累项目骨架的模板技能和维护已验证修复方案的调试技能，使智能体能够搭建稳定架构并系统性修复集成错误，从而从高层设计生成完全可玩的游戏。

    

    游戏开发处于创意设计与复杂软件工程的交汇点，需要协同编排游戏引擎、实时循环以及跨多个文件的紧耦合状态。尽管大型语言模型（LLM）和代码智能体现在已能轻松解决孤立的编程任务，但当被要求从高层设计生成一个完全可玩的游戏时，它们总是屡屡受挫，在跨文件不一致、场景连线断裂和逻辑不连贯等问题面前溃不成军。我们通过 OpenGame 弥合了这一差距，这是首个专为端到端网页游戏创作而设计的开源智能体框架。其核心在于 Game Skill——一种可复用、可演进的能力，由 Template Skill（模板技能）和 Debug Skill（调试技能）组成：前者能够从经验中不断积累项目骨架库，后者能够维护一份持续更新的已验证修复方案协议——二者共同使智能体能够搭建稳定的架构并系统性地修复集成错误。

    arXiv:2604.18394v2 Announce Type: replace  Abstract: Game development sits at the intersection of creative design and intricate software engineering, demanding the joint orchestration of game engines, real-time loops, and tightly coupled state across many files. While Large Language Models (LLMs) and code agents now solve isolated programming tasks with ease, they consistently stumble when asked to produce a fully playable game from a high-level design, collapsing under cross-file inconsistencies, broken scene wiring, and logical incoherence. We bridge this gap with OpenGame, the first open-source agentic framework explicitly designed for end-to-end web game creation. At its core lies Game Skill, a reusable, evolving capability composed of a Template Skill that grows a library of project skeletons from experience and a Debug Skill that maintains a living protocol of verified fixes - together enabling the agent to scaffold stable architectures and systematically repair integration error
    
[^46]: 从程序化技能到策略基因：迈向经验驱动的测试时演化

    From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution

    [https://arxiv.org/abs/2604.15097](https://arxiv.org/abs/2604.15097)

    本研究通过45个场景、4590次受控试验发现，紧凑的“策略基因”表示比面向文档的技能包更适合作为可复用经验的载体，在测试时控制与迭代演化中均表现更优，证明经验的表示方式本身是决定性因素。

    

    本贝塔版技术报告探讨了一个问题：可复用的经验应当如何表示，才能既作为有效的测试时控制手段，又作为迭代演化的基础。我们在45个科学代码求解场景中开展了4590次受控试验来研究这一问题。我们发现，面向文档的技能包所提供的控制并不稳定：其有效信号稀疏，而将一个紧凑的经验对象扩展为更完整的文档化包往往无济于事，甚至会降低整体平均表现。我们进一步证明，表示方式本身就是一阶关键因素：紧凑的基因表示能够取得最强的整体平均成绩，在显著的结构扰动下仍保持竞争力，并优于同等预算的技能片段，而重新附加面向文档的材料通常会削弱而非改善其表现。除一次性控制之外，我们还表明基因也是迭代式经验演化的更优载体。

    arXiv:2604.15097v3 Announce Type: replace-cross  Abstract: This beta technical report asks how reusable experience should be represented so that it can function as effective test-time control and as a substrate for iterative evolution. We study this question in 4.590 controlled trials across 45 scientific code-solving scenarios. We find that documentation-oriented Skill packages provide unstable control: their useful signal is sparse, and expanding a compact experience object into a fuller documentation package often fails to help and can degrade the overall average. We further show that representation itself is a first-order factor. A compact Gene representation yields the strongest overall average, remains competitive under substantial structural perturbations, and outperforms matched-budget Skill fragments, while reattaching documentation-oriented material usually weakens rather than improves it. Beyond one-shot control, we show that Gene is also a better carrier for iterative exper
    
[^47]: 自动代码修订中大语言模型置信度校准的细粒度方法

    Fine-grained Approaches for Confidence Calibration of LLMs in Automated Code Revision

    [https://arxiv.org/abs/2604.06723](https://arxiv.org/abs/2604.06723)

    该论文针对自动代码修订（ACR）任务，提出细粒度的LLM置信度校准方法，以解决传统全局Platt缩放方法在此类任务中不可靠的问题，从而帮助开发者更好地判断模型输出的可信度。

    

    在当今AI辅助软件工程的背景下，开发者越来越依赖能力强大但本质上并不完美的大语言模型（LLM）。这些模型产生错误输出的倾向会降低开发者的生产力。为此，一种经典的缓解方法是提供经过校准的置信度分数，在实例层面真实地反映其输出的正确可能性。这类信息使用户能够立即对输出做出接受与否的决策、放弃易出错的输出，并更好地使其期望与模型能力保持一致。由于经过后训练的LLM本身不会产生良好校准的置信度分数，研究人员开发了事后校准方法，其中对序列级置信度分数进行全局Platt缩放（Platt-scaling）的方法在许多生成式软件工程任务中被证明是有效的，但在自动代码修订（ACR）等任务中仍然不可靠或尚未被探索……

    arXiv:2604.06723v2 Announce Type: replace-cross  Abstract: In today's AI-assisted software engineering landscape, developers increasingly depend on LLMs that are highly capable, yet inherently imperfect. The tendency of these models to produce incorrect outputs can reduce developer productivity. To this end, a canonical mitigation method is to provide calibrated confidence scores that faithfully reflect their likelihood of correctness at the instance-level. Such information allows users to make immediate decisions regarding output acceptance, abstain error-prone outputs, and better align their expectations with the model's capabilities. Since post-trained LLMs do not inherently produce well-calibrated confidence scores, researchers have developed post-hoc calibration methods, with global Platt-scaling of sequence-level confidence scores proving effective in many generative software engineering tasks but remaining unreliable or unexplored for automated code revision (ACR) tasks such as 
    
[^48]: 动态量子电路的可扩展基准测试框架

    Scalable Benchmarking Framework for Dynamic Quantum Circuits

    [https://arxiv.org/abs/2604.03360](https://arxiv.org/abs/2604.03360)

    该论文提出了dynamarq——一个可扩展且与硬件无关的动态量子电路基准测试框架，通过收集多样化的动态电路基准集并定义刻画其结构的电路特征，填补了现有基准测试工具仅适用于幺正电路的空白。

    

    带有中途测量（MCMs）和前馈操作的动态量子电路在量子纠错和量子算法等多种应用中发挥着至关重要的作用。随着量子硬件的进步使得中途测量和前馈循环的实现成为可能，动态电路的使用日益普遍。由于现有的基准测试工具主要是为幺正电路设计的，无法简单地扩展到动态电路，因此迫切需要一个专门为动态电路设计、能够捕捉其独特属性的基准测试框架。我们提出了dynamarq，一个可扩展且与硬件无关的动态电路基准测试框架。我们收集了一组涵盖各种应用领域的动态电路基准测试集，并提出了一套广泛的电路特征来刻画这些动态电路的结构。我们在两台IBM量子处理器和Qiskit模拟器上运行了这些基准测试（摘要在此处截断）。

    arXiv:2604.03360v2 Announce Type: replace-cross  Abstract: Dynamic quantum circuits with mid-circuit measurements (MCMs) and feed-forward operations play a crucial role in various applications, such as quantum error correction and quantum algorithms. With advancements in quantum hardware enabling the implementation of MCM and feed-forward loops, the use of dynamic circuits has become increasingly prevalent. There is a significant need for a benchmarking framework specially designed for dynamic circuits to capture their unique properties, as current benchmarking tools are designed primarily for unitary circuits and cannot be trivially extended to dynamic circuits. We propose dynamarq, a scalable and hardware-agnostic benchmarking framework for dynamic circuits. We collect a set of dynamic circuit benchmarks spanning various applications and propose a broad set of circuit features to characterize the structure of these dynamic circuits. We run them on two IBM quantum processors and the Q
    
[^49]: 面向高级关系模型的超性质模型检测

    Model checking of hyperproperties for high-level relational models

    [https://arxiv.org/abs/2512.12024](https://arxiv.org/abs/2512.12024)

    该论文提出了 HyperPardinus，一种扩展 Alloy 时序逻辑后端 Pardinus 的新模型求解程序，使开发者能够在系统设计早期阶段自动验证关系模型上的超性质，填补了高级规约语言在超性质验证方面的空白。

    

    与安全性或并发性相关的许多属性必须被编码为所谓的超性质，即允许对系统的多条执行轨迹进行推理的时序属性。然而，尽管超性质模型检测最近取得了进展，目前仍然缺乏能够在系统设计早期阶段有效支持软件工程从业者验证此类属性的高级规约语言。Alloy 是一种轻量级形式化方法，拥有由自动化分析程序支持的高级规约语言，使其特别适合在开发早期阶段验证设计模型。然而，它本身并不支持超性质的验证。本工作提出了 HyperPardinus，这是一种新的模型求解程序，它扩展了 Pardinus——Alloy 语言的时序逻辑后端——以自动验证关系模型上的超性质。

    arXiv:2512.12024v3 Announce Type: replace  Abstract: Many properties related to security or concurrency must be encoded as so-called hyperproperties, temporal properties that allow reasoning about multiple traces of a system. However, despite recent advances on model checking hyperproperties, there is still a lack of higher-level specification languages that can effectively support software engineering practitioners in verifying properties of this class at early stages of system design.   Alloy is a lightweight formal method with a high-level specification language that is supported by automated analysis procedures, making it particularly well-suited for the verification of design models at early development stages. It does not natively support, however, the verification of hyperproperties.   This work proposes HyperPardinus, a new model finding procedure that extends Pardinus -- the temporal logic backend of the Alloy language -- to automatically verify hyperproperties over relational
    

