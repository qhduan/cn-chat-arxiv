# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TestPrism: Rethinking Test Evaluation Beyond a Single Reference](https://arxiv.org/abs/2610.12289) | 提出TestPrism基准，通过联合成功函数指标（要求测试同时通过所有有效实现并拒绝所有无效实现）超越单一参考评估方式，发现现有LLM编程智能体的真实测试质量仅为28%，并提出TestHelix方法加以改进。 |
| [^2] | [Cadence: Strategic Guidance for Coding Agents](https://arxiv.org/abs/2610.12269) | 提出动态监控框架Cadence，根据编程智能体的实时执行健康状况自适应地调度检查并提供分级指导（建议级与替换级），克服了现有静态触发机制响应滞后或开销过大的问题。 |
| [^3] | [Reliability Characterization for N-version Object Detection](https://arxiv.org/abs/2610.12149) | 提出了两个专门针对N版本目标检测的可靠性指标Cov_OD和Cer_OD，它们无需投票策略即可从单个检测结果中计算得出，弥补了传统mAP和准确率指标无法捕捉检测结果间多样性与一致性等可靠性因素的不足。 |
| [^4] | [Protecting CPU AI On Edge TEEs: WebAssembly's Promise and Practical Challenges](https://arxiv.org/abs/2610.12050) | 本工作提出在Arm TrustZone的OP-TEE中通过WAMR运行未经修改的WebAssembly编译AI模型，并通过加密二进制文件和将密钥存入设备熔丝来保护AI模型知识产权，仅引入22%的额外开销。 |
| [^5] | [Traceable World State: A Provenance-Aware State Representation and Deterministic Replay Framework for Robotic Systems](https://arxiv.org/abs/2610.12033) | 本文提出可追溯世界状态（TWS）框架，通过不可变快照更新、有序确定性重放和SHA-256哈希链，为机器人系统提供具备事实溯源、决策复现和防篡改审计能力的世界状态表示。 |
| [^6] | [Neural Network Verification for Deep Joint Source-Channel Coding](https://arxiv.org/abs/2610.11994) | 本文提出了首个针对深度联合信源信道编码解码器的界传播验证框架，通过对PReLU激活、转置卷积和瑞利衰落的处理，形式化地界定了无线信道噪声区域内的最坏情况重建误差。 |
| [^7] | [Can LLMs Fix It Without Code? Toward Automated Verification of No-Code Bug Fixes](https://arxiv.org/abs/2610.11963) | 本研究提出了一种基于执行的自动化流水线，通过在真实浏览器环境中让执行代理按照自然语言指令实施修复并用专用检查器验证缺陷是否依然存在，来评估大语言模型生成的无代码缺陷修复是否有效。 |
| [^8] | [Forms of LLM-Integrated Applications from LLM-Chats to Autonomous AI Agent System](https://arxiv.org/abs/2610.11899) | 该综述系统考察了chatbot、copilot、RAG、agent等LLM集成应用标签背后的真实架构内涵，归纳出七种反复出现的应用形式，并发现四家主流厂商的编程智能体均采用委派子智能体的“推理-行动”循环这一共同架构。 |
| [^9] | [Evaluating Exact Output and Checkpoint-State Prediction in Real Programs](https://arxiv.org/abs/2610.11889) | 该论文提出了一个基于371个真实Python和C++程序、共400个案例的基准，用于仅凭源代码和输入预测程序最终输出与循环内外的检查点状态，发现开启推理能力的模型显著优于关闭推理的设置，且更长的程序执行跟踪会大幅降低模型预测的准确率。 |
| [^10] | [NosRacer: Dynamic Detection of Race Conditions in On-Device Network Operating Systems](https://arxiv.org/abs/2610.11875) | 本文提出NosRacer，一个采用两阶段设计的动态分析框架，用于高效检测工业级设备内置网络操作系统（NOS）中由异步消息传递引起的竞态条件。 |
| [^11] | [Trajectory-Guided Fault Localization for Agent Skill Evolution](https://arxiv.org/abs/2610.11858) | 提出SkillMorph，通过在多次运行和任务的抽象轨迹中比较失败与成功证据来识别可疑动作，进而定位技能中需要编辑的位置并自动生成修订，实现基于轨迹引导故障定位的智能体技能演化。 |
| [^12] | [On the Risks of using LLM-Generated Tests for Regression Testing](https://arxiv.org/abs/2610.11835) | 本研究揭示了基于LLM生成的回归测试的风险：当被测代码本身存在缺陷时，生成的测试会固化错误行为，研究通过对145个真实拉取请求的实验分析量化了这一影响。 |
| [^13] | [ObliVul: Alert-Conditioned Safety Obligation Modeling and Bidirectional Counterfactual Validation for Code Vulnerability Detection](https://arxiv.org/abs/2610.11801) | 该论文提出ObliVul方法，通过以单个警报为条件建模安全义务并进行双向反事实验证，从静态分析产生的大量候选警报中识别出真正值得关注的漏洞警报。 |
| [^14] | [MAP4CS: A Multi-dimensional Data Pruning Framework for Efficient Code Retriever Fine-tuning](https://arxiv.org/abs/2610.11727) | 提出MAP4CS自适应数据剪枝框架，通过融合句法结构、语义多样性和分布表示三个维度，从大规模代码语料库中筛选出高质量核心子集，从而实现更高效、更高质量的代码检索器微调。 |
| [^15] | [Implementing the Spec Growth Engine: Preventing Spec-Code Divergence, and Growing the Spec with Agents](https://arxiv.org/abs/2610.11725) | 本文提出规格增长引擎的实现，用确定性引擎防止规格与代码偏差、用多个智能体按轮次扩展规格，并通过三个独立开关灵活控制人类判断的委派程度。 |
| [^16] | [One Skill Too Many: How Co-Installed Skills Conflict in Coding Agents](https://arxiv.org/abs/2610.11647) | 本文对编程智能体中共装相似技能之间的冲突进行了首个大规模实证研究，揭示了易冲突技能普遍存在，且这类冲突会在任务照常通过的情况下悄然剥夺已安装技能的核心功能，从而被现有基准测试所忽视。 |
| [^17] | [PolyCodeEval: Benchmarking Multilingual Code Generation from Functions to Repositories](https://arxiv.org/abs/2610.11618) | 该论文提出了PolyCodeEval——一个包含2,590个任务、覆盖五种编程语言且从函数级延伸到仓库级的多语言多粒度代码生成统一基准，并揭示了现有大模型与编码智能体仍难以正确生成跨粒度、跨语言的完整代码。 |
| [^18] | [Where Do the Tokens Go? Understanding and Reducing Costs in LLM Agents for Vulnerability Discovery](https://arxiv.org/abs/2610.11602) | 该研究通过分析 200 条 CyberGym 轨迹发现，LLM 漏洞发现智能体的 token 成本主要消耗在代码定位理解与漏洞推理触发设计两大瓶颈上（占 60.4%），而现有效率方法仅在 24.4% 的案例中能以更低成本保持成功，从而为降低智能体成本指明了方向。 |
| [^19] | [Runnable Commit Untangling for Coding Agents](https://arxiv.org/abs/2610.11593) | 本文提出 RucTangle，首个面向编码智能体的提交解缠方法，能在将大型纠缠补丁拆分为有序提交的同时保证每个提交后的代码均可运行，并弥补了现有研究未直接验证维护收益的不足。 |
| [^20] | [Chronos Enables Code Agents to Reason over Software Evolution](https://arxiv.org/abs/2610.11578) | Chronos 提出一个测试时框架，将历史拉取请求提炼为结构化经验卡片并通过类型化关系图谱连接，使基于 LLM 的代码智能体能够利用软件演化历史来生成补丁并在候选补丁间做出选择，从而提升代码修复效果。 |
| [^21] | [SWE-Journey: Towards More Realistic Evaluation of Coding Assistants through Long-Horizon, Multi-Turn Interaction](https://arxiv.org/abs/2610.11559) | 提出 SWE-Journey 基准，通过弱到强合成流水线自动构建长周期编码任务，并利用从真实交互数据挖掘的用户画像和用户模拟智能体重现多轮交互，从而实现对编码助手更贴近现实的评估。 |
| [^22] | [SoK: Are LLMs Reliable at Source Code Recovery? A Taxonomy and Empirical Evaluation](https://arxiv.org/abs/2610.11556) | 本文提出了首个专注于大语言模型辅助的二进制到源代码恢复的知识系统化研究，构建了以设计为中心的细粒度方法分类法，并通过七个关键指标在六个评估维度上开展系统性实证评估，填补了该领域方法碎片化与评估不统一的空白。 |
| [^23] | [SSCBench: Evaluating the Evidential Validity of Fault-Injection Tests for Tool-Using LLM Agents](https://arxiv.org/abs/2610.11514) | 本文提出SSCBench基准，系统研究了工具使用型LLM智能体故障注入测试的证据有效性问题，通过测量协议检验反驳性观测结果能否被智能体实际感知，揭示了故障采纳结论可能存在的偏差。 |
| [^24] | [Evaluating Local Language Model Agents for Reproducible Data Engineering: An Empirical Software Engineering Study of Mobility Workflows](https://arxiv.org/abs/2610.11482) | 该研究构建了一个包含十五个移动性工作流任务的基准测试，通过确定性检查器系统评估了十种本地部署的开源权重LLM智能体在生成正确且可复现数据工程制品方面的能力，并量化了闭环工作区条件等因素的影响。 |
| [^25] | [Closed-loop evaluation of LLM agents for embedded software development](https://arxiv.org/abs/2610.11447) | 该论文提出了一个包含五个嵌入式控制任务和四种反馈场景的基准测试，用于闭环评估LLM编码智能体在嵌入式软件开发中实现并自我验证设备行为的能力。 |
| [^26] | [Characterizing Overconfident Failure in LLM-Based Code Generation](https://arxiv.org/abs/2610.11300) | 本文在四个开源代码模型和三个执行基准上，揭示了代码大语言模型中错误程序常以与正确程序相当的 token 级置信度生成的过度自信问题，并系统评估了现有不确定性指标能否可靠预测执行正确性。 |
| [^27] | [Retromorphic Testing of Quantum Compiler Passes](https://arxiv.org/abs/2610.11195) | 针对量子编译器变换正确性难以验证的问题，本文系统分析了四大量子编程框架的单元测试现状，并提出了基于逆态测试的编译器变换自动化验证方法。 |
| [^28] | [Who Pays the Review Cost? Triage, Fairness, and Accountability in AI-authored Pull Requests](https://arxiv.org/abs/2610.11179) | 该研究通过对31个国家239名开发者的问卷调查发现，AI作者身份本身并非被拒信号，评审者是否投入评审精力取决于AI拉取请求是否以可问责贡献的形式出现，包括范围有限、理由基于项目、超越CI的验证以及贡献者的响应性。 |
| [^29] | [Skill Constellations: Tracing the Supply Chain of Agent Skills on GitHub](https://arxiv.org/abs/2610.11169) | 本文构建了首个基于git历史的智能体技能复制时间网络，覆盖GitHub上219万余次技能采用，揭示了少数源头仓库主导技能复制、GitHub星标无法识别这些源头、且副本几乎不随源头同步更新导致安全修复难以传播的关键问题。 |
| [^30] | [IRONPROOF: COBOL-to-Python Transpilation with SMT-Based Equivalence Checking](https://arxiv.org/abs/2610.11073) | IRONPROOF系统通过将COBOL和Python代码编码为Z3公式并利用SMT求解器进行等价性检查，为COBOL到Python的自动代码转换提供机器可验证的形式化等价性证明或反例，在2345个COBOL文件中成功证明了606个转换的等价性。 |
| [^31] | [Following Breadcrumbs in Code: What Accidentally Committed Ad-Hoc Logs Reveal about Developer Comprehension](https://arxiv.org/abs/2610.11041) | 该研究首次通过挖掘意外提交的代码和直播编程会话，构建了覆盖 Java、JavaScript 和 Python 的大型数据集，系统性地揭示了开发者利用临时日志来理解程序行为的使用模式。 |
| [^32] | [Probabilistic Sensing, Deterministic Authority: Admitting Model-Produced Observations into Sufficiency-Checked Governance Contracts](https://arxiv.org/abs/2610.10978) | 提出一种将模型输出的观测以带分数记录形式纳入确定性治理合约的框架：阈值接纳策略将分数映射为真/假/未知（未知即拒绝），感知改变裁决的概率受各感知字段错误接纳率与未知率之和（并集界）约束，并通过36,000次模型调用的注册研究验证了该界的有效性。 |
| [^33] | [Cross-Provider Review as a Runtime Contract for Coding Agents: A Controlled Pilot and Fault-Injection Study](https://arxiv.org/abs/2610.10961) | 提出将跨提供商代码审查规范化为一种运行时契约（涵盖独立资源池、有界执行、明确失败状态与持久证据等条款），并通过受控试点和故障注入研究验证了其有效发现缺陷的能力。 |
| [^34] | [When Flaws Cascade: Understanding Vulnerabilities and Exploitation Chains in JavaScript Engines](https://arxiv.org/abs/2610.10844) | 本文首次对JavaScript引擎漏洞进行了全面实证研究，构建了涵盖2017-2024年四个主流引擎的241个漏洞数据集，建立了症状与根本原因的分类体系，并系统分析了漏洞特征及潜在利用链策略。 |
| [^35] | [DITTO: A Context-aware Pickle-based Pre-Trained Model Scanner for Effective Security Audits](https://arxiv.org/abs/2610.10735) | 本文提出首个基于栈的上下文感知 Pickle 预训练模型扫描器 DITTO，通过忠实跟踪 Pickle 虚拟机状态转换并执行上下文感知语义分析来推断模型意图，同时构建了包含 959 个良性模型和 92 个恶意模型的 PickleBench 基准，有效弥合了现有扫描器在覆盖率与精确度之间的差距。 |
| [^36] | [Applying Security by Design at the Point of Execution: How Governed Security Requirements Affect the Security of AI-Generated Code](https://arxiv.org/abs/2610.10659) | 该研究证明，在代码生成执行时点通过MCP服务器向AI代码生成器提供来自受治理安全设计知识库的安全需求，可将通过全部安全测试的任务比例从44.1%提升至78.0%，并将BaxBench上功能正确且无漏洞利用的解决方案比例从65%提升至86%。 |
| [^37] | [Can LLMs Simulate Novice Programmers' Misconceptions?](https://arxiv.org/abs/2610.10656) | 本研究评估了13个大语言模型模拟新手编程误解的能力，发现前沿模型表现可靠，但小模型难以进行动态执行，代码调优模型在模拟误解时会回退到正确执行，且判断题格式的诊断任务显著易于开放式生成任务。 |
| [^38] | [SoK: Failure Modes in Common Criteria Product Evaluation - A Taxonomy and Design-for-Evaluability Guidance](https://arxiv.org/abs/2610.10644) | 本文从评估者的操作视角出发，对通用准则（CC）产品评估中反复出现的跨厂商失效模式进行了系统化梳理，构建了失效模式分类法，并提供了面向可评估性的设计指南。 |
| [^39] | [Visible Reasoning Is Not a Universal Optimizer: Persona- and Thinking-Dependent Effects in Analytics Code Generation](https://arxiv.org/abs/2610.10639) | 该论文通过跨 SQL 与 pandas 双语言、交叉角色设定与多种思考指令的受控执行基准实验发现，显式思维链推理并非普遍有效，其效果因角色表述、目标语言和思考格式而异，“用 SQL/Python 思考”这类匹配目标语言的指令并不能可靠带来提升。 |
| [^40] | [Has LLM Screening Performance Stalled in Software Engineering Systematic Reviews?](https://arxiv.org/abs/2610.10633) | 该研究通过基准测试评估了八个新大语言模型在软件工程系统综述文献筛选中的表现，发现新模型仅比旧模型略有提升（平均MCC从0.347升至0.365），表明LLM筛选性能进展缓慢，且不同研究间的差异仍大于模型间的差异。 |
| [^41] | [Ruleless Digital Twins: Toward Declarative Decision-Making Through Standardized Frameworks and Technologies](https://arxiv.org/abs/2610.10631) | 本文提出无规则数字孪生（RDTs）概念，通过标准化框架和技术，基于纯声明式的用户规范自动生成最优决策，以替代日益复杂难维护的传统基于规则的数字孪生决策模型。 |
| [^42] | [Agent4RE: A Self-Refining Multi-agent Framework for End-to-End Software Requirements Engineering and Benchmarking](https://arxiv.org/abs/2610.10628) | 提出了Agent4RE——一个具备双重迭代自精炼机制的多智能体需求工程框架，并构建了首个覆盖需求获取到生成全流程的端到端需求工程基准数据集RE-E2E。 |
| [^43] | [WorldBench: Evaluating LLMs on Three.js Voxel World Generation](https://arxiv.org/abs/2610.10622) | WorldBench通过让评判系统主动探索运行中的3D世界（控制时钟、环绕观察、派遣导航智能体取景）并将视觉观察与源代码相互交叉验证，解决了现有单一视角评判方法不可靠的问题，实现了对LLM生成的Three.js体素世界的可靠评估。 |
| [^44] | [TestJack: Should you trust the results in coding benchmarks? Agentic Coding Benchmarks Auditing via Evaluator Evolution](https://arxiv.org/abs/2610.10619) | 提出TestJack框架，通过评估器进化为每次试验动态生成针对性测试来审计智能体编码基准，揭示仅依赖固定单元测试可能高估LLM智能体的真实问题解决能力。 |
| [^45] | [MRCert: Towards Post-deployment Patch Robustness Certification for Adversarially Patched Samples via Type-specific Masking](https://arxiv.org/abs/2610.10617) | 提出首个基于掩码的认证恢复防御方法MRCert，通过对良性样本和对抗性补丁样本推断类型特定的必要属性，在保持高预测准确率的同时验证对抗性补丁样本标签的良性，实现部署后的补丁鲁棒性认证。 |
| [^46] | [PyCache Trap: The Inspection-Execution Gap in Agent Skill Scanners](https://arxiv.org/abs/2610.10612) | 该论文揭示了针对Agent技能扫描器的PyCache Trap攻击，利用Python字节码缓存与源代码之间的检查-执行鸿沟实现94-100%的攻击成功率，并提出执行感知验证（EAV）方法在类型化执行图中关联指令、脚本与运行时工件以检测此类隐藏威胁。 |
| [^47] | [Code Understanding is a Bottleneck for Coding Agents](https://arxiv.org/abs/2610.10610) | 该论文提出CABRA基准，通过调用图变换从零构建难度可精确控制、可扩展的代码理解任务，发现编程智能体是依靠工具调用来弥补其代码理解能力的不足，且工具调用次数比编辑代码行数更能预测智能体的表现。 |
| [^48] | [Beyond Type-checking: Towards Holistic Evaluation of Formal Specification Generation](https://arxiv.org/abs/2610.10604) | 该论文提出了一个从350个Lean任务构建的统一数据集和涵盖形式有效性、参考相似性与等价性、行为充分性的整体评估框架，以解决规范生成中仅靠类型检查无法验证生成规范是否忠实反映用户意图的问题。 |
| [^49] | [A Survey on LLM-Integrated Hardware Design Verification](https://arxiv.org/abs/2610.10580) | 本综述系统回顾了大语言模型在硬件功能验证中的应用，涵盖断言生成、测试平台生成、缺陷定位与形式化验证等多个方向，并指出LLM最有效的角色是作为嵌入验证流程中的语义推理、搜索和编排组件。 |
| [^50] | [ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding](https://arxiv.org/abs/2610.08662) | 提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。 |
| [^51] | [Understanding the Hierarchical Structure and Functional Landscape of the Model Context Protocol Ecosystem](https://arxiv.org/abs/2610.05319) | 该论文构建了迄今最大的MCP生态系统工具级地图MCPacific，通过LLM驱动的迭代流程建立了涵盖58,915项能力的层次化功能分类体系，解决了数十万MCP服务器因缺乏细粒度功能组织而导致智能体难以发现、比较和替代工具的问题。 |
| [^52] | [Governed Human-AI Prioritization Under Uncertainty: Adaptive Estimation and Dependency-Constrained Portfolio Selection](https://arxiv.org/abs/2609.10648) | 本文提出了一种在不确定性下可治理、可检查且可重新校准的人机协同优先级排序框架，通过五个定量算子（BVS、ERS、PVS、CCS、ODP）实现自适应估计与依赖约束下的组合选择，并用受控合成实验验证了该框架对参数扰动的鲁棒性。 |
| [^53] | [Beyond Component Testing: Validating Agentic AI Systems](https://arxiv.org/abs/2607.29405) | 该论文通过对262篇文献的系统性映射研究，提出涵盖行为、安全、时间、监管和多智能体五个维度的分类框架，刻画了智能体AI系统的验证问题，并揭示了现有研究方法在各维度上分布的密集与空白之处。 |
| [^54] | [Efficient and Scalable Provenance Tracking for LLM-Generated Code Snippets](https://arxiv.org/abs/2605.28510) | 该论文提出SourceTracker编码器与HybridSourceTracker两阶段混合流水线，先用向量搜索缩小候选范围、再用Winnowing指纹精确重排，从而实现对LLM生成代码在数十亿级训练语料上的高效可扩展溯源追踪。 |
| [^55] | [VISTA: An End-to-End Benchmark for Visual Spec-to-Web-App Coding Agents](https://arxiv.org/abs/2605.26144) | VISTA是一个端到端基准测试，评估编程智能体将多页面设计交付物转化为可运行的全栈Web和Android应用的能力，通过在运行的应用中定位并探测8,487个人工标注的交互组件来产生可追溯至各个设计需求的评分。 |
| [^56] | [PropGen: Automated Property Generation for Property-Based Testing of Mobile Apps](https://arxiv.org/abs/2604.13463) | 本文提出PropGen，利用大语言模型自动为移动应用生成属性，突破了人工编写属性的瓶颈，使基于属性的测试能够高效检测不崩溃但行为异常的功能性缺陷。 |
| [^57] | [ReCodeAgent: A Multi-agent Workflow for Language-Agnostic Translation and Validation of Large-Scale Repositories](https://arxiv.org/abs/2604.07341) | ReCodeAgent通过自主多智能体工作流，实现了仓库级代码翻译和验证的语言无关性，用户仅需指定源和目标编程语言即可自动处理整个仓库。 |
| [^58] | [Imperative Interference: Social Register Shapes Instruction Topology in Large Language Models](https://arxiv.org/abs/2603.25015) | 该研究发现大语言模型会按照社会语域惯例来处理系统提示指令——同一指令在英语中协作而在西班牙语中竞争，将祈使句改写为陈述句可降低81%的跨语言差异，说明模型把指令理解为社会行为而非技术规范。 |
| [^59] | [Arbiter: Detecting Interference in LLM Agent System Prompts](https://arxiv.org/abs/2603.08993) | 提出了Arbiter框架，通过结合形式化评估规则与多模型LLM排查来检测主流编码智能体系统提示词中的干扰模式，并发现提示词架构与故障类别强相关、多模型评估能揭示单模型分析无法发现的漏洞类型。 |
| [^60] | [PackMonitor: Enabling Zero Package Hallucinations Through Decoding-Time Monitoring](https://arxiv.org/abs/2602.20717) | 本文提出PackMonitor，首个通过在解码时持续监控并根据有限可枚举的权威包列表进行干预、从根本上彻底消除（而非仅仅降低）LLM包幻觉的方法。 |
| [^61] | [CAF\'E: Causal Black-Box Testing of Machine Unlearning](https://arxiv.org/abs/2509.16525) | 提出CAF'E框架，将机器遗忘测试构建为基于规范的测试，仅通过黑盒模型的预测输出对特征进行因果干预并传播到下游特征，从而有效检测特征影响在模型中的残留。 |
| [^62] | [Skill-Adaptive Imitation Learning for UI Test Reuse](https://arxiv.org/abs/2409.13311) | 该研究发现即使大语言模型具备高度准确的UI事件映射能力，仍不足以解决源应用与目标应用之间的实现差异，因此提出技能自适应模仿学习方法以提升UI测试复用的有效性。 |

# 详细

[^1]: TestPrism：超越单一参考的测试评估新思考

    TestPrism: Rethinking Test Evaluation Beyond a Single Reference

    [https://arxiv.org/abs/2610.12289](https://arxiv.org/abs/2610.12289)

    提出TestPrism基准，通过联合成功函数指标（要求测试同时通过所有有效实现并拒绝所有无效实现）超越单一参考评估方式，发现现有LLM编程智能体的真实测试质量仅为28%，并提出TestHelix方法加以改进。

    

    大语言模型（LLM）编程智能体已经在多种编程任务的测试生成方面取得了进展。然而，当前普遍的做法是仅针对单一参考解来评估测试，这忽略了其他同样有效的替代实现，并可能高估测试质量。我们提出了TestPrism，该基准包含来自17个来源的300个测试任务和3000个候选实现，有效解与无效解各占一半。其主要指标“联合成功函数”要求生成的测试在初始程序状态下失败、接受所有有效候选实现、并拒绝所有无效候选实现。在十四种基线编程智能体配置上，联合成功函数仅达到28.00%，而单参考成功率达到59.67%。我们的分析揭示了测试中存在遗漏行为、缺乏依据的断言以及错误的测试构造等问题。为了解决这些缺陷，我们提出了TestHelix，它结合了测试与修复对的异构合成。

    arXiv:2610.12289v1 Announce Type: new  Abstract: Large language model (LLM) coding agents have advanced test generation across diverse programming tasks. However, the common practice of evaluating tests against a single reference solution overlooks alternative valid implementations and can overstate test quality. We introduce TestPrism, comprising 300 test tasks from 17 sources and 3000 candidate implementations, evenly split between valid and invalid solutions. Its primary metric, Joint Success Function, requires the generated tests to fail on the initial program state, accept every valid candidate, and reject every invalid candidate. Across fourteen baseline coding agent configurations, Joint Success Function reaches only 28.00%, whereas single reference success reaches 59.67%. Our analysis reveals missed behaviors, unsupported assertions, and faulty test construction. To address these weaknesses, we introduce TestHelix, which combines heterogeneous synthesis of test and repair pairs
    
[^2]: Cadence：面向编程智能体的策略性引导

    Cadence: Strategic Guidance for Coding Agents

    [https://arxiv.org/abs/2610.12269](https://arxiv.org/abs/2610.12269)

    提出动态监控框架Cadence，根据编程智能体的实时执行健康状况自适应地调度检查并提供分级指导（建议级与替换级），克服了现有静态触发机制响应滞后或开销过大的问题。

    

    运行时监控器正被越来越多地用于提高基于大语言模型（LLM）的编程智能体的可靠性，其方式是检查执行轨迹并在检测到不当行为时提供纠正性指导。然而，它们的有效性仍然受限于静态的指导触发机制。现有的监控器要么依赖固定的检查间隔，导致在严重错误行为期间错过及时干预，同时在正常执行期间产生不必要的开销；要么依赖僵化的启发式规则，无法检测复杂的推理错误。为解决这些局限性，我们提出了Cadence，一个动态监控框架，它能够根据智能体的实时执行健康状况自适应地安排检查并提供指导。Cadence包含两个核心模块：双层干预模块和检查调度器。干预模块在正常执行和轻微失误时提供建议级（advisory-level）指导，而在出现严重问题时提供替换级（replacement-level）指导（原文摘要在此处被截断）。

    arXiv:2610.12269v1 Announce Type: new  Abstract: Runtime monitors are increasingly used to improve the reliability of LLM-based coding agents by inspecting execution trajectories and delivering corrective guidance upon detecting misbehavior. However, their effectiveness remains limited by static guidance triggering schemes. Existing monitors rely either on fixed inspection intervals, missing timely guidance during severe misbehaviors while incurring unnecessary overhead during healthy execution, or on rigid heuristic rules, failing to detect complex reasoning errors. To address these limitations, we propose Cadence, a dynamic monitoring framework that adaptively schedules inspections and delivers guidance according to the agent's real-time execution health. Cadence consists of two core modules: a two-tier intervention module and an inspection scheduler. The intervention module delivers advisory-level guidance for normal executions and minor lapses, while providing replacement-level gui
    
[^3]: N版本目标检测的可靠性表征

    Reliability Characterization for N-version Object Detection

    [https://arxiv.org/abs/2610.12149](https://arxiv.org/abs/2610.12149)

    提出了两个专门针对N版本目标检测的可靠性指标Cov_OD和Cer_OD，它们无需投票策略即可从单个检测结果中计算得出，弥补了传统mAP和准确率指标无法捕捉检测结果间多样性与一致性等可靠性因素的不足。

    

    N版本目标检测（OD）是一种通过使用多个模型或输入帧来使检测结果多样化，并通过聚合各个结果来减少检测错误的方法。多个检测结果之间的多样性和一致性是表征N版本OD系统各种可能配置可靠性的关键信息。然而，现有的性能指标（如mAP和准确率）无法捕捉这些因素，因为它们仅基于聚合后的最终结果来定义。为了克服这一局限性，我们提出了两个专门为N版本OD定义的可靠性指标，即目标检测错误覆盖率（Cov_OD）和目标检测准确预测确定性（Cer_OD），这两个指标可以直接从各个检测结果中计算得出，而无需依赖投票策略。我们通过对自动驾驶场景中车辆N版本OD的案例研究，实证展示了所提出指标的独特特性。

    arXiv:2610.12149v1 Announce Type: new  Abstract: N-version object detection (OD) is an approach to diversifying detection results using multiple models or input frames and reducing detection errors by aggregating individual results. Diversity and consistency across multiple detection results are critical information for characterizing the reliability of possible configurations of N-version OD systems. However, existing performance metrics such as mAP and Accuracy fail to capture these factors, as they are defined solely on the final outcome after aggregation. To overcome this limitation, we propose two reliability metrics particularly defined for N-version OD, namely coverage of errors in OD (Cov_OD) and certainty of accurate prediction in OD (Cer_OD), which can be computed from individual detection results without relying on voting strategies. We empirically demonstrate the unique features of the proposed metrics through a case study of N-version OD for a vehicle in an autonomous-driv
    
[^4]: 保护边缘可信执行环境中的CPU AI：WebAssembly的前景与实际挑战

    Protecting CPU AI On Edge TEEs: WebAssembly's Promise and Practical Challenges

    [https://arxiv.org/abs/2610.12050](https://arxiv.org/abs/2610.12050)

    本工作提出在Arm TrustZone的OP-TEE中通过WAMR运行未经修改的WebAssembly编译AI模型，并通过加密二进制文件和将密钥存入设备熔丝来保护AI模型知识产权，仅引入22%的额外开销。

    

    边缘硬件上的AI模型包含重要的知识产权（IP），但当攻击者获得root权限时便可能窃取这些知识产权。诸如Arm TrustZone之类的可信执行环境（TEE）可以防御这些操作系统（OS）级别的攻击。然而，TEE的使用具有挑战性。更准确地说，在TEE中运行未经修改的应用程序十分困难。这项工作提出了一种解决方案，允许在Arm TrustZone的OP-TEE中的WebAssembly微运行时（WAMR）上执行编译为WebAssembly的未经修改的AI模型。此外，这项工作提供了一个AI模型分发器，该分发器对WebAssembly二进制文件进行加密，并将加密密钥存储在设备的熔丝（fuse）中。这样，只有OP-TEE中的WAMR可信应用（TA）才能解密并执行该AI模型。对该解决方案的全面评估表明，与手动移植到OP-TEE的应用程序相比，它仅产生22%的额外开销，同时还……（摘要原文在此处被截断）

    arXiv:2610.12050v1 Announce Type: cross  Abstract: AI models on edge hardware contain important intellectual property (IP), but an adversary can steal it when they achieve root access. Trusted Execution Environments (TEE) like Arm TrustZone protect against these Operating System (OS) level attacks. However, they are challenging to use. More precisely, it is difficult to run unaltered applications inside a TEE. This work presents a solution that allows the execution of unaltered AI models, compiled to WebAssembly, on the WebAssembly Micro Runtime (WAMR) in OP-TEE for Arm TrustZone. Additionally, this work provides an AI model distributor that encrypts the WebAssembly binary and places the encryption key in one of the device's fuses. This way, only the WAMR Trusted Application (TA) in OP-TEE can decrypt and execute the AI model. A thorough evaluation of our solution shows that it incurs an additional overhead of 22% in comparison to an application manually ported to OP-TEE, while also fa
    
[^5]: 可追溯世界状态：一种面向机器人系统的溯源感知状态表示与确定性重放框架

    Traceable World State: A Provenance-Aware State Representation and Deterministic Replay Framework for Robotic Systems

    [https://arxiv.org/abs/2610.12033](https://arxiv.org/abs/2610.12033)

    本文提出可追溯世界状态（TWS）框架，通过不可变快照更新、有序确定性重放和SHA-256哈希链，为机器人系统提供具备事实溯源、决策复现和防篡改审计能力的世界状态表示。

    

    机器人系统在执行长时间任务时，必须维护一个由不同时间到达的观测所组成的世界状态，这些观测具有不同的置信度且可能被修订。传统的表示方法侧重于最新估计，这阻碍了事实溯源、决策复现或执行审计。我们提出了可追溯世界状态（TWS），这是一种中间件无关的语义表示和参考运行时，用于实现具备溯源感知能力的机器人世界状态。TWS快照捕获实体、关系、观测、置信度和修订元数据。经过验证的更新操作以不可变的方式转换快照，有序更新支持确定性重放，规范的SHA-256哈希链确保日志具备防篡改能力。我们通过模式符合性测试、完整状态生命周期、确定性重放和故障注入对TWS进行评估。该框架在Python 3.10-3.14上通过了38项测试，能够检测记录损坏、哈希链断裂、序列中

    arXiv:2610.12033v1 Announce Type: cross  Abstract: Robotic systems operating over extended tasks must maintain a world state assembled from observations arriving at different times, with varying confidence and potential revisions. Conventional representations emphasize latest estimates, hindering fact provenance, decision reproduction, or execution auditing. We present Traceable World State (TWS), a middleware-neutral semantic representation and reference runtime for provenance-aware robot world state. A TWS snapshot captures entities, relations, observations, confidence, and revision metadata. Validated update operations transform snapshots immutably, ordered updates support deterministic replay, and a canonical SHA-256 hash chain ensures tamper-evident logs. We evaluate TWS through schema conformance, complete state lifecycles, deterministic replay, and fault injection. Passing 38 tests across Python 3.10-3.14, the framework detects record corruptions, broken hash links, sequence dis
    
[^6]: 面向深度联合信源信道编码的神经网络验证

    Neural Network Verification for Deep Joint Source-Channel Coding

    [https://arxiv.org/abs/2610.11994](https://arxiv.org/abs/2610.11994)

    本文提出了首个针对深度联合信源信道编码解码器的界传播验证框架，通过对PReLU激活、转置卷积和瑞利衰落的处理，形式化地界定了无线信道噪声区域内的最坏情况重建误差。

    

    深度联合信源信道编码使用神经编码器-解码器通过无线信道端到端地传输数据，但在对抗性扰动和信道干扰下，重建质量可能急剧下降；目前尚无方法能对DeepJSCC的这种性能退化给出形式化的界。我们提出了首个用于验证DeepJSCC解码器的界传播框架，对给定无线信道噪声区域内的最坏情况重建误差进行界定。当前的深度神经网络验证器不支持DeepJSCC解码器的三个组件：参数化修正线性激活（PReLU）、转置卷积和瑞利衰落。我们扩展了DNN验证中最先进的线性松弛优化技术以支持PReLU，将转置卷积替换为其受限的先上采样后卷积形式，并将瑞利衰落表述为直接前置到解码器中的结构化扰动，从而可归约……

    arXiv:2610.11994v1 Announce Type: cross  Abstract: Deep joint source-channel coding (DeepJSCC) transmits data end-to-end over wireless channels using a neural encoder-decoder, but reconstruction quality can degrade sharply under adversarial perturbations and channel disturbances; no method formally bounds this degradation for DeepJSCC. We present the first bound-propagation framework for verifying DeepJSCC's decoder, bounding worst-case reconstruction error over a given wireless channel's noise region. Current deep neural network (DNN) verifiers do not support three DeepJSCC decoder components: parametric rectified linear activations (PReLU), transposed convolutions, and Rayleigh fading. We extend state-of-the-art techniques for optimization of linear relaxation in DNN verification for PReLU, replace the transposed convolution with its restricted upsample-then-convolution form, and formulate Rayleigh fading as a structural perturbation prepended directly into the decoder, thereby reduc
    
[^7]: 大语言模型能在不写代码的情况下修复问题吗？迈向无代码缺陷修复的自动化验证

    Can LLMs Fix It Without Code? Toward Automated Verification of No-Code Bug Fixes

    [https://arxiv.org/abs/2610.11963](https://arxiv.org/abs/2610.11963)

    本研究提出了一种基于执行的自动化流水线，通过在真实浏览器环境中让执行代理按照自然语言指令实施修复并用专用检查器验证缺陷是否依然存在，来评估大语言模型生成的无代码缺陷修复是否有效。

    

    无代码修复是通过引导用户更改某项设置、升级到问题已被修复的版本，或调整其工作流程来解决无效缺陷报告的方法。人工验证所提出的无代码修复是否能解决所报告的缺陷需要耗费开发人员大量时间。本研究提出了一种自动化的、基于执行的流水线，用于在大规模语言模型（LLM）在真实浏览器环境中生成无代码修复的能力评估。我们评估了先前研究基准中发布的12种配置所生成的322个无代码修复，这些修复涵盖被归类为错误配置、错误版本以及外部系统与依赖的缺陷报告。一个执行器代理按照自然语言指令应用每个修复，并由针对具体问题的检查器判断所报告的缺陷是否仍然存在。我们使用三个执行器重复了该流水线：两个计算机使用代理——OpenCUA-72B 和 Claude Sonnet 5，以及一个……

    arXiv:2610.11963v1 Announce Type: cross  Abstract: A no-code fix resolves an invalid bug report by directing the user to change a setting, update to a version where the problem is already fixed, or adjust their workflow. Manually verifying whether a proposed no-code fix resolves the reported bug takes considerable developer time. This study proposes an automated, execution-based pipeline for evaluating the capability of large language models (LLMs) to generate no-code fixes in a real browser environment. We evaluate 322 no-code fixes generated by the 12 configurations released with the benchmark of a previous study, covering bug reports categorized as Faulty Configuration, Wrong Version, or External System & Dependency. An executor agent applies each fix by following its natural-language instructions, and an issue-specific checker determines whether the reported bug persists. We repeat the pipeline with three executors: two Computer-Use Agents, OpenCUA-72B and Claude Sonnet 5, and one 
    
[^8]: 从LLM对话到自主AI智能体系统：LLM集成应用的形式

    Forms of LLM-Integrated Applications from LLM-Chats to Autonomous AI Agent System

    [https://arxiv.org/abs/2610.11899](https://arxiv.org/abs/2610.11899)

    该综述系统考察了chatbot、copilot、RAG、agent等LLM集成应用标签背后的真实架构内涵，归纳出七种反复出现的应用形式，并发现四家主流厂商的编程智能体均采用委派子智能体的“推理-行动”循环这一共同架构。

    

    arXiv:2610.11899v1 公告类型：cross 摘要：大语言模型（LLM）正日益作为组件被嵌入软件系统中，并以聊天机器人、副驾驶、检索增强生成、工作流、编程智能体和AI智能体等标签进行推广。这些标签究竟代表真正的架构形式，还是仅作为品牌营销手段，此前尚未得到系统性评估。在所调查的资料来源中，这些标签确实承载了架构内涵，这在厂商的使用中体现得最为明显：copilot（副驾驶）表示一种在用户逐步确认下操作宿主应用的路由器-工作器架构，而近期转向agent（智能体）这一标签则与AI规划的多步骤执行相吻合，用户只能看到执行结果。四家主要提供商的编程智能体共享同一种架构，即一种委派给子智能体的“推理-行动”循环。本综述描述了七种反复出现的形式——LLM对话、自定义智能体、检索增强生成（RAG）、AI增强工作流、副驾驶、编程智能体，以及……

    arXiv:2610.11899v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly embedded as components in software systems, marketed under labels such as chatbot, copilot, retrieval-augmented generation, workflow, coding agent and AI agent. Whether these labels denote genuine architectural forms or serve as branding has not been assessed systematically.   In the sources surveyed, labels do carry architectural content, most clearly in vendor usage: copilot denotes a router-worker architecture operating a host application under step-by-step user confirmation, while the more recent shift to the label agent coincides with AI-planned multi-step execution of which the user sees only the outcome. The coding agents of four major providers share one architecture, a reason-and-act loop delegating to subagents.   This survey describes seven recurring forms---LLM chats, custom agents, retrieval-augmented generation (RAG), AI-enhanced workflows, copilots, coding agents, and, in par
    
[^9]: 评估真实程序中的精确输出与检查点状态预测

    Evaluating Exact Output and Checkpoint-State Prediction in Real Programs

    [https://arxiv.org/abs/2610.11889](https://arxiv.org/abs/2610.11889)

    该论文提出了一个基于371个真实Python和C++程序、共400个案例的基准，用于仅凭源代码和输入预测程序最终输出与循环内外的检查点状态，发现开启推理能力的模型显著优于关闭推理的设置，且更长的程序执行跟踪会大幅降低模型预测的准确率。

    

    我们提出了一个仅凭源代码和输入来预测程序最终输出和检查点状态的基准测试。该基准扩展了CRUXEval风格的输出预测，包含配对的较短与较长跟踪输入，以及位于循环内部和循环之后的检查点。该基准包含来自371个Python和C++程序的400个案例，在四种模型家族的七种设置下进行评估，全程不使用工具或代码执行。在计划的11,200个预测中，有11,151个产生了可评分的响应。启用推理的设置在已完成的响应上比关闭推理的对应设置高出33.1到55.2个百分点。表现最强的设置在较短跟踪的最终输出上得分93.0%，在较长跟踪的最终输出上得分77.0%，在两个状态任务上分别得分65.5%和63.5%；即使将缺失的响应计为错误，这些分数依然成立。在2,397组源代码完全相同的匹配Python比较中，改用较长跟踪的输入会导致528个预测从正确变为错误，并出现147次结果反转。

    arXiv:2610.11889v1 Announce Type: cross  Abstract: We present a benchmark for predicting final output and checkpoint state from source and input alone. It extends CRUXEval-style output prediction with paired shorter- and longer-trace inputs and checkpoints inside and after a loop. The benchmark contains 400 cases from 371 Python and C++ programs, evaluated under seven settings from four model families without tools or code execution. Of 11,200 planned predictions, 11,151 produced gradable responses. Reasoning-enabled settings outperform their off counterparts by 33.1 to 55.2 percentage points on completed responses. The strongest setting scores 93.0% on shorter-trace final output, 77.0% on longer-trace final output, and 65.5% and 63.5% on the two state tasks; these scores also hold when missing responses count as wrong. Across 2,397 matched Python comparisons with identical source, changing to the longer-trace input yields 528 correct-to-wrong changes and 147 reversals. Source-clustere
    
[^10]: NosRacer：设备内置网络操作系统中竞态条件的动态检测

    NosRacer: Dynamic Detection of Race Conditions in On-Device Network Operating Systems

    [https://arxiv.org/abs/2610.11875](https://arxiv.org/abs/2610.11875)

    本文提出NosRacer，一个采用两阶段设计的动态分析框架，用于高效检测工业级设备内置网络操作系统（NOS）中由异步消息传递引起的竞态条件。

    

    商用设备内置网络操作系统（NOS）在生产路由器和交换机中运行复杂的控制平面，其中配置更新任务由多个松耦合组件通过异步消息传递执行。这类执行容易产生竞态条件：相同的有序输入命令在消息以不同顺序传递时可能产生不同的结果。竞态条件难以暴露，因为它们通常仅表现为细微的、延迟出现的故障。现有静态分析技术对大规模工业级NOS缺乏可扩展性和精确性，而动态分析技术在试图覆盖异步消息交织空间时会产生巨大的系统执行开销。为解决上述挑战，我们提出了NosRacer，一个面向工业级设备内置NOS的竞态条件检测动态分析框架。NosRacer采用两阶段设计。集中阶段……（摘要原文在此处被截断）

    arXiv:2610.11875v1 Announce Type: cross  Abstract: Commercial on-device network operating systems (NOSes) run complex control planes in production routers and switches, where configuration update tasks are executed by multiple loosely coupled components through asynchronous message passing. Such executions are prone to race conditions: the same ordered input commands may produce different outcomes when messages are delivered in different orders. The race conditions are difficult to expose because they often manifest only as subtle, delayed malfunctions. Existing static analysis techniques lack scalability and precision for large-scale industrial NOSes, while dynamic ones incur substantial system-execution cost when attempting to cover the space of asynchronous message interleavings.   To address the challenges above, we present NosRacer, a dynamic analysis framework for race condition detection in industrial-grade on-device NOSes. NosRacer uses a two-phase design. The concentration pha
    
[^11]: 面向智能体技能演化的轨迹引导故障定位

    Trajectory-Guided Fault Localization for Agent Skill Evolution

    [https://arxiv.org/abs/2610.11858](https://arxiv.org/abs/2610.11858)

    提出SkillMorph，通过在多次运行和任务的抽象轨迹中比较失败与成功证据来识别可疑动作，进而定位技能中需要编辑的位置并自动生成修订，实现基于轨迹引导故障定位的智能体技能演化。

    

    智能体技能为代码智能体提供可复用的指导，但不完整或不合适的指导可能会损害任务执行。为了减少技能精炼所需的人工投入，近期的方法使用大语言模型（LLM）根据执行反馈生成修订。然而，如何将这些修订建立在明确的行为证据之上仍然是一个挑战。为了解决这一空白，我们提出了SkillMorph，这是一种基于轨迹引导的智能体技能故障定位的技能演化方法。其核心思想是在生成修订之前，将执行证据与特定的技能内容关联起来。具体而言，SkillMorph在多次运行和多个任务的抽象轨迹中比较失败与成功的证据，并结合演化循环之间的变化来识别可疑动作。然后，它利用这些可疑动作来定位技能中的编辑位置，并生成相应的修订。在SWE-Skills-Bench和CannBot上的实验表明，演化后的技能…

    arXiv:2610.11858v1 Announce Type: cross  Abstract: Agent skills provide reusable guidance for code agents, but incomplete or unsuitable guidance can impair task execution. To reduce the manual effort of skill refinement, recent approaches use LLMs to generate revisions from execution feedback. However, grounding these revisions in explicit behavioral evidence remains challenging. To address this gap, we propose SkillMorph, a skill-evolution approach based on trajectory-guided fault localization in agent skills. Its core idea is to link execution evidence to specific skill contents before generating revisions. Specifically, SkillMorph compares failure and success evidence in abstracted trajectories across repeated runs and tasks, incorporating changes between evolution loops to identify suspicious actions. It then uses these suspicious actions to localize edit sites in the skills and generate corresponding revisions. Experiments on SWE-Skills-Bench and CannBot show that the skills evolv
    
[^12]: 关于使用大语言模型生成测试进行回归测试的风险研究

    On the Risks of using LLM-Generated Tests for Regression Testing

    [https://arxiv.org/abs/2610.11835](https://arxiv.org/abs/2610.11835)

    本研究揭示了基于LLM生成的回归测试的风险：当被测代码本身存在缺陷时，生成的测试会固化错误行为，研究通过对145个真实拉取请求的实验分析量化了这一影响。

    

    软件处于持续演进之中：开发人员不断添加新功能、修复缺陷并重构代码，其中任何变更都可能破坏现有功能。回归测试通过在测试用例中捕获预期行为来防范此类影响。基于大语言模型（LLM）的测试生成旨在通过直接从被测代码生成回归测试来自动化这一过程。当实现正确时，这种方法是有益的；但当代码存在缺陷时则会产生问题：生成的测试可能会编码并固化错误的行为。为了研究这一风险，我们将基于LLM的回归测试生成应用于合并到软件项目主分支的拉取请求，并研究生成的测试对项目后续演化的影响。我们区分了两类测试：揭示缺陷的测试（断言正确实现的行为）和固化缺陷的测试（断言缺陷行为）。在来自SciPy、Q……的145个拉取请求中（原文摘要在此处截断）。

    arXiv:2610.11835v1 Announce Type: new  Abstract: Software is under constant evolution: developers continuously add features, fix bugs, and refactor code, and any of these changes may break existing functionality. Regression testing guards against such effects by capturing expected behavior in test cases. LLM-based test generation aims to automate this process by generating regression tests directly from the code under test. This is beneficial when the implementation is correct, but problematic when the code contains faults: the generated tests may then encode and preserve incorrect behavior. To investigate this risk, we apply LLM-based regression test generation to pull requests merged into the main branch of software projects and study the impact of the generated tests on subsequent project evolution. We distinguish between fault-revealing tests, which assert correctly implemented behavior, and fault-enforcing tests, which assert faulty behavior. Across 145 pull requests from SciPy, Q
    
[^13]: ObliVul：面向代码漏洞检测的警报条件安全义务建模与双向反事实验证

    ObliVul: Alert-Conditioned Safety Obligation Modeling and Bidirectional Counterfactual Validation for Code Vulnerability Detection

    [https://arxiv.org/abs/2610.11801](https://arxiv.org/abs/2610.11801)

    该论文提出ObliVul方法，通过以单个警报为条件建模安全义务并进行双向反事实验证，从静态分析产生的大量候选警报中识别出真正值得关注的漏洞警报。

    

    在真实的软件开发中，漏洞检测的主要挑战往往不是发现可疑代码，而是从静态分析产生的大量候选警报中识别出真正值得关注的警报。现有的基于学习的方法主要在函数或行级别识别可疑模式，难以提取以单个警报为中心的完整程序证据。尽管大型语言模型可以从局部程序事实中推断出风险源、危险操作、保护条件和状态前提等语义信息，但这些信息无法可靠地与特定程序节点、依赖关系和传播路径对齐，因此不足以验证相应的安全义务是否真正影响当前警报。为解决这一问题，我们提出了ObliVul，一种警报条件下的安全义务建模与双向反事实验证方法。

    arXiv:2610.11801v1 Announce Type: new  Abstract: In real-world software development, the primary challenge in vulnerability detection is often not finding suspicious code, but identifying which alerts among the large number of candidate alerts produced by static analysis truly warrant attention. Existing learning-based methods mainly identify suspicious patterns at the function or line level, making it difficult to extract complete program evidence centered on an individual alert. Although large language models can infer risk sources, dangerous operations, protection conditions, and state preconditions from local program facts, such semantic information cannot be reliably aligned with specific program nodes, dependency relations, and propagation paths, and is therefore insufficient to verify whether the corresponding safety obligations truly affect the current alert. To address this problem, we propose ObliVul, an alert-conditioned safety obligation modeling and bidirectional counterfa
    
[^14]: MAP4CS：一个用于高效代码检索器微调的多维数据剪枝框架

    MAP4CS: A Multi-dimensional Data Pruning Framework for Efficient Code Retriever Fine-tuning

    [https://arxiv.org/abs/2610.11727](https://arxiv.org/abs/2610.11727)

    提出MAP4CS自适应数据剪枝框架，通过融合句法结构、语义多样性和分布表示三个维度，从大规模代码语料库中筛选出高质量核心子集，从而实现更高效、更高质量的代码检索器微调。

    

    检索增强生成（RAG）已成为软件工程领域中利用领域特定知识增强大语言模型（LLMs）能力的基石。然而，由于大规模代码语料库中固有的噪声和冗余，将检索器适配到不断演化的代码仓库仍然充满挑战。在完整语料库上进行标准微调不仅计算成本高昂，而且由于低质量样本导致的负迁移，往往会带来次优性能；相反，简单的随机采样又无法保证数据的代表性。为应对这些挑战，我们提出了MAP4CS（面向代码搜索的多维感知剪枝），一个自适应数据剪枝框架。MAP4CS通过整合句法结构、语义多样性和分布表示来识别一个小而高质量的核心子集，随后经过严格的基于规则的过滤流程。在两个大规模数据集上的大量实验证明了该方法的有效性。

    arXiv:2610.11727v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) has become a cornerstone in software engineering for enhancing Large Language Models (LLMs) with domain-specific knowledge. However, adapting retrievers to evolving code repositories remains challenging due to the noise and redundancy inherent in massive code corpora. Standard fine-tuning on the full corpus is computationally expensive and often leads to sub-optimal performance due to negative transfer from low-quality samples. Conversely, simple random sampling fails to guarantee data representativeness.   To address these challenges, we propose MAP4CS (Multi-dimensional Awareness Pruning for Code Search), an adaptive data pruning framework. MAP4CS identifies a small, high-quality core subset by integrating syntactic structure, semantic diversity, and distributional representation, followed by a rigorous rule-based filtering pipeline. Extensive experiments on two large-scale datasets demonstrate th
    
[^15]: 实现规格增长引擎：防止规格与代码的偏差，并用智能体扩展规格

    Implementing the Spec Growth Engine: Preventing Spec-Code Divergence, and Growing the Spec with Agents

    [https://arxiv.org/abs/2610.11725](https://arxiv.org/abs/2610.11725)

    本文提出规格增长引擎的实现，用确定性引擎防止规格与代码偏差、用多个智能体按轮次扩展规格，并通过三个独立开关灵活控制人类判断的委派程度。

    

    规格（Spec）增长引擎将 AI 辅助软件开发锚定在一张与代码相耦合的规格图上。本文描述了该引擎的实现，它服务于两项任务，并将二者作为两个相互分离的层。第一层防止规格-代码偏差：一个确定性引擎验证规格图，将其与代码的导入图进行比较，从记录的测试证据中为节点赋予已验证状态，并根据每项变更可能破坏的内容对其进行分类——全程无需调用任何模型。第二层借助智能体扩展规格：意图作者、规划者和编码者，每个角色由各自的模型扮演，以轮次方式扩展规格图，并在每轮结束后由一条确定性规则决定运行是否继续。人类判断被委派的程度由三个独立的开关控制——草稿门控、破坏性变更的委派以及运行模式——再加上两种奠定项目基础的方式，共提供十八种运行项目的方式。

    arXiv:2610.11725v1 Announce Type: new  Abstract: The Spec Growth Engine anchors AI-assisted software development in a graph of specifications that the code is coupled to. This paper describes its implementation, which serves two tasks and keeps them apart as two layers. The first layer prevents spec-code divergence: a deterministic engine validates the spec graph, compares it with the code's import graph, earns a node's verified status from recorded test evidence, and classifies every change by what it can break -- without calling a model. The second layer grows the spec with agents: an intent author, a planner and a coder, each played by its own model, extend the graph in rounds, and a deterministic rule decides after each round whether the run goes on. How much of the human's judgement is delegated is set by three independent switches -- a draft gate, a delegation for breaking changes, and the run mode -- which, with two ways of laying a project's floor, give eighteen ways to run a p
    
[^16]: 多一个技能成祸：编程智能体中共装技能如何产生冲突

    One Skill Too Many: How Co-Installed Skills Conflict in Coding Agents

    [https://arxiv.org/abs/2610.11647](https://arxiv.org/abs/2610.11647)

    本文对编程智能体中共装相似技能之间的冲突进行了首个大规模实证研究，揭示了易冲突技能普遍存在，且这类冲突会在任务照常通过的情况下悄然剥夺已安装技能的核心功能，从而被现有基准测试所忽视。

    

    编程智能体通过智能体技能进行扩展，技能是包含 SKILL.md 文件的目录，该文件告诉模型何时以及如何执行任务。由于技能来自独立的来源（团队、开发者、插件、复制的集合），一个已安装的技能可能与执行相同工作的相似技能被共同安装，而模型仅凭名称和描述在它们之间做出选择。在发生冲突时，已安装的技能会失去核心功能（例如禁止触碰 git 的限制），因为相似技能会替代其运行或改变其行为。此时任务仍然通过，因此仅检查任务完成情况的基准测试会漏掉这类问题。我们对此类冲突进行了首个实证研究。从 20,947 个仓库的快照中，我们挖掘出 822,109 个候选相似技能对，让大语言模型对其中 3,754 个分层样本进行判定，并在三个模型上运行了 312 个确认的技能对（共 6,368 次运行、169,294 次工具调用、542 个智能体小时）。我们报告了五项发现。（1）易冲突的技能很常见：将近……

    arXiv:2610.11647v1 Announce Type: cross  Abstract: Coding agents are extended with agent skills, directories whose SKILL.md tells the model when and how to perform a task. Because skills come from independent sources (teams, developers, plugins, copied collections), an installed skill can be co-installed with a similar skill doing the same job, and the model picks between them by name and description alone. In a conflict, the installed skill loses core functions (e.g., a ban on touching git) because the similar skill runs instead or changes what it does. The task still passes, so benchmarks that check only task completion miss such cases. We present the first empirical study of such conflicts. From snapshots of 20,947 repositories, we mine 822,109 candidate similar-skill pairs, have an LLM judge a stratified sample of 3,754, and run 312 confirmed pairs on three models (6,368 runs, 169,294 tool calls, 542 agent-hours). We report five findings. (1) Conflict-prone skills are common: nearl
    
[^17]: PolyCodeEval：从函数到仓库的多语言代码生成基准测试

    PolyCodeEval: Benchmarking Multilingual Code Generation from Functions to Repositories

    [https://arxiv.org/abs/2610.11618](https://arxiv.org/abs/2610.11618)

    该论文提出了PolyCodeEval——一个包含2,590个任务、覆盖五种编程语言且从函数级延伸到仓库级的多语言多粒度代码生成统一基准，并揭示了现有大模型与编码智能体仍难以正确生成跨粒度、跨语言的完整代码。

    

    随着大语言模型日益向仓库级软件工程方向发展，现有的代码生成基准在语言覆盖、任务粒度和评估协议方面仍然零散分割，阻碍了系统性比较。为填补这一空白，我们提出了PolyCodeEval，一个统一的多语言、多粒度代码生成基准。它包含2,590个代码生成任务，涵盖从函数级到仓库级的多个层级，这些任务源自五种编程语言的58个真实的、可执行的开源仓库。所有任务均在统一的基于执行的评估协议下进行评估，并针对各自的生成目标定制了集成流程。基于该基准，我们评估了前沿大语言模型、最先进的专业化方法以及通用编码智能体。我们的结果表明，现有方法仍然难以在不同粒度和不同语言下正确生成完整的代码片段。

    arXiv:2610.11618v1 Announce Type: new  Abstract: As large language models increasingly move toward repository-level software engineering, existing code-generation benchmarks remain fragmented across language coverage, task granularity, and evaluation protocols, impeding systematic comparison. To address this gap, we present PolyCodeEval, a unified multilingual and multi-granularity benchmark for code generation. It comprises 2,590 code generation tasks spanning functions to repositories, derived from 58 real, executable open-source repositories in five programming languages. All tasks are evaluated under a unified execution-based protocol with integration procedures tailored to their generation targets. Building on this benchmark, we evaluate frontier large language models, state-of-the-art specialized methods, and general coding agents. Our results show that existing approaches still struggle to correctly generate complete code fragments across granularities and languages. Specificall
    
[^18]: Token 都去哪儿了？理解并降低用于漏洞发现的 LLM 智能体成本

    Where Do the Tokens Go? Understanding and Reducing Costs in LLM Agents for Vulnerability Discovery

    [https://arxiv.org/abs/2610.11602](https://arxiv.org/abs/2610.11602)

    该研究通过分析 200 条 CyberGym 轨迹发现，LLM 漏洞发现智能体的 token 成本主要消耗在代码定位理解与漏洞推理触发设计两大瓶颈上（占 60.4%），而现有效率方法仅在 24.4% 的案例中能以更低成本保持成功，从而为降低智能体成本指明了方向。

    

    LLM 智能体在漏洞发现过程中可能消耗数百万个 token，却未能产出可用的概念验证（PoC）。是什么消耗了这些预算？为什么未能取得成果？我们通过对 200 条 CyberGym 轨迹进行多维度开放编码研究来诊断这些成本与失败，涵盖四个智能体（即 Codex、OpenCode、Cybench 和 EnIGMA），并在无辅助基线以及四种现有效率方法下进行测试。该研究揭示了三个关键发现：第一，不同智能体在成功率与成本上差异显著，更高的花费并不总能带来更好的结果；第二，代码定位与理解，以及漏洞推理与触发设计，共占 token 消耗的 60.4%，是失败运行中的两大主要瓶颈；第三，在匹配对比中仅有 24.4% 的案例能够在更低总成本下保持成功，不合适的信号和辅助开销限制了现有方法的收益。

    arXiv:2610.11602v1 Announce Type: cross  Abstract: LLM agents can spend millions of tokens during vulnerability discovery without producing a working proof of concept (PoC). What consumes that budget, and why does it fail to produce results? We diagnose these costs and failures through a multi-axis open-coding study of 200 CyberGym traces, spanning four agents (i.e., Codex, OpenCode, Cybench, and EnIGMA) under an unaided baseline and four existing efficiency methods. The study reveals three key findings. First, different agents vary substantially in success and cost, and higher spending does not consistently yield better outcomes. Second, code localization and understanding, together with vulnerability reasoning and trigger design, account for 60.4% of tokens and represent the two leading bottlenecks in failed runs. Third, only 24.4% of matched comparisons preserve success at lower total cost; unsuitable signals and auxiliary overhead limit the benefits of existing methods. Motivated b
    
[^19]: 面向编码智能体的可运行提交解缠

    Runnable Commit Untangling for Coding Agents

    [https://arxiv.org/abs/2610.11593](https://arxiv.org/abs/2610.11593)

    本文提出 RucTangle，首个面向编码智能体的提交解缠方法，能在将大型纠缠补丁拆分为有序提交的同时保证每个提交后的代码均可运行，并弥补了现有研究未直接验证维护收益的不足。

    

    编码智能体会生成庞大且纠缠混杂的补丁，其中混合了多种开发目的，使得代码难以审查和维护。提交解缠有望将这类大型补丁组织成解缠的、易于管理的提交。本文指出了现有提交解缠研究的两个重要局限。第一，现有研究没有考虑到解缠后的提交是有顺序的，且应保证代码在每个提交后仍然可以运行。在实践中，维护者不太可能接受导致代码无法运行的提交。第二，现有研究声称提交解缠有助于软件维护，但它们只是对解缠后的提交与开发者的原始提交进行句法层面的比较，并未直接展示所声称的维护收益。为弥补这些不足，本文做出了两项新颖的贡献：(1) RucTangle，首个能够在解缠提交的同时保证每个提交后代码可运行的智能体方法；(2)

    arXiv:2610.11593v1 Announce Type: cross  Abstract: Coding agents produce large, tangled patches that mix multiple development purposes, making the code hard to review and maintain. Commit untangling offers the promise of organizing such large patches into untangled, manageable commits. This paper emphasizes two important limitations in existing commit untangling studies. First, they do not consider that untangled commits are ordered and should leave the code runnable. In practice, maintainers are unlikely to accept commits that prevent the code from running. Second, existing studies claim that commit untangling helps software maintenance. However, they conduct syntactic comparisons between the untangled commits and developers' original commits without directly showing the claimed maintenance benefits. To address these gaps, this paper makes two novel contributions: (1) RucTangle, the first agentic method that untangles commits while keeping the code runnable after each commit; and (2) 
    
[^20]: Chronos 使代码智能体能够对软件演化进行推理

    Chronos Enables Code Agents to Reason over Software Evolution

    [https://arxiv.org/abs/2610.11578](https://arxiv.org/abs/2610.11578)

    Chronos 提出一个测试时框架，将历史拉取请求提炼为结构化经验卡片并通过类型化关系图谱连接，使基于 LLM 的代码智能体能够利用软件演化历史来生成补丁并在候选补丁间做出选择，从而提升代码修复效果。

    

    历史拉取请求记录了代码库当前状态背后的设计决策、兼容性约束和实现模式。与新任务相关的经验可能分布在多个相关变更中，而这些变更的描述侧重于不同的关注点。我们提出了 Chronos，一个测试时框架，使这种相互关联的历史可被基于大语言模型（LLM）的代码智能体利用。Chronos 将已合并的拉取请求提炼为结构化的经验卡片，并通过一个包含代码级、开发者意图和组织关系三类类型的图谱将它们连接起来。语义搜索用于识别入口卡片，加权的多跳扩展则检索相互关联的变更以供选择性阅读。同一份记忆同时指导候选生成和补丁选择：一个专注于补丁的变更智能体和一个验证策略智能体各自开发一个补丁，随后一个“演化管理者”参考历史记录在两者之间做出选择。在 SWE-Bench Verified 上，完整的工作流提升了（摘要至此被截断）……

    arXiv:2610.11578v1 Announce Type: cross  Abstract: Historical pull requests record the design decisions, compatibility constraints, and implementation patterns behind a codebase's current state. Experience relevant to a new task can span related changes whose descriptions emphasize different concerns. We introduce Chronos, a test-time framework that makes this connected history available to large language model (LLM)-based code agents. Chronos distills merged pull requests into structured experience cards and connects them through a typed graph of code-level, developer-intent, and organizational relations. Semantic search identifies entry cards, and weighted multi-hop expansion retrieves connected changes for selective reading. The same memory guides candidate generation and patch selection: a patch-focused change agent and a validation-strategy agent each develop a patch, and an evolution steward consults history to select between them. On SWE-Bench Verified, the full workflow improve
    
[^21]: SWE-Journey：通过长周期、多轮交互实现对编码助手更真实的评估

    SWE-Journey: Towards More Realistic Evaluation of Coding Assistants through Long-Horizon, Multi-Turn Interaction

    [https://arxiv.org/abs/2610.11559](https://arxiv.org/abs/2610.11559)

    提出 SWE-Journey 基准，通过弱到强合成流水线自动构建长周期编码任务，并利用从真实交互数据挖掘的用户画像和用户模拟智能体重现多轮交互，从而实现对编码助手更贴近现实的评估。

    

    诸如 Claude Code 和 Codex 等编码助手已成为大语言模型（LLM）智能体的主要应用，然而现有基准测试与真实使用场景仍相距甚远，尤其是在任务时长和交互长度方面。编码助手需要在不断演进的代码仓库中完成长链条的开发工作，同时通过多轮交互反复澄清需求并调整实现方案。为弥补这些差距，我们提出了 SWE-Journey，一个用于更真实评估编码助手的基准测试。为解决任务时长差距，我们提出了一种弱到强的合成流水线，可自动构建长周期编码任务。为解决交互差距，我们从真实交互数据中挖掘出四种具有代表性的用户画像，并构建了一个用户模拟智能体，以重现真实的代码辅助交互过程。平均而言，在与软件架构师角色的交互中，模型在所请求功能上的测试通过率超过 75%，但是……（摘要原文在此处截断）

    arXiv:2610.11559v1 Announce Type: cross  Abstract: Coding assistants such as Claude Code and Codex have become a major application of LLM agents, yet existing benchmarks remain far from real-world use, particularly in task horizon and interaction length. Code assistants require completing long chains of development work in continuously evolving repositories, while repeatedly clarifying requirements and adapting implementations through multi-turn interaction. To address these gaps, we introduce SWE-Journey, a benchmark for more realistic evaluation of coding assistants. To address the task-horizon gap, we propose a weak-to-strong synthesis pipeline that automatically constructs long-horizon coding tasks. To address the interaction gap, we mine four representative user personas from real interaction data and build a user-simulation agent to reproduce realistic code-assistance interactions. On average, models pass over 75% of tests for requested functionality with software architects, but
    
[^22]: SoK：大语言模型在源代码恢复方面可靠吗？一种分类法与实证评估

    SoK: Are LLMs Reliable at Source Code Recovery? A Taxonomy and Empirical Evaluation

    [https://arxiv.org/abs/2610.11556](https://arxiv.org/abs/2610.11556)

    本文提出了首个专注于大语言模型辅助的二进制到源代码恢复的知识系统化研究，构建了以设计为中心的细粒度方法分类法，并通过七个关键指标在六个评估维度上开展系统性实证评估，填补了该领域方法碎片化与评估不统一的空白。

    

    有效的源代码恢复对于恶意软件分析、漏洞评估和遗留系统维护等安全应用至关重要。大语言模型（LLM）正在重塑这一领域，将研究范式从基于规则的启发式方法，转变为从汇编代码或传统反编译器生成的伪C代码中进行概率性的、高保真度的语义源代码恢复。然而，尽管进展迅速，该领域仍存在众多方法碎片化以及评估不统一的问题，限制了客观比较。此外，现有工作对嵌入式和物联网架构的覆盖有限，对C/C++以外源语言的研究也不足。在本工作中，我们提出了首个专注于LLM辅助的二进制到源代码恢复的知识系统化研究。我们提供了一个以设计为中心的细粒度LLM辅助源代码恢复方法分类法，并使用七个关键指标在六个评估维度上进行了系统性评估。

    arXiv:2610.11556v1 Announce Type: cross  Abstract: Effective source recovery is critical to security applications such as malware analysis, vulnerability assessment, and legacy maintenance. Large Language Models (LLMs) are reshaping this field, shifting the paradigm away from rule-based heuristics to probabilistic and high fidelity semantic recovery of source code from assembly or classical decompiler-derived pseudo-C. However, despite rapid progress, the field suffers from fragmentation across numerous approaches as well as their non-unified evaluations, limiting objective comparisons. Further, existing works have limited coverage of embedded, IoT architectures and source languages beyond C/C++. In this work, we present the first Systematization of Knowledge (SoK) focused specifically on LLM-assisted binary-to-source recovery. We provide a granular design-centric taxonomy of LLM-assisted source recovery methods and systematic evaluations using seven key metrics along six evaluation di
    
[^23]: SSCBench：评估工具使用型LLM智能体故障注入测试的证据有效性

    SSCBench: Evaluating the Evidential Validity of Fault-Injection Tests for Tool-Using LLM Agents

    [https://arxiv.org/abs/2610.11514](https://arxiv.org/abs/2610.11514)

    本文提出SSCBench基准，系统研究了工具使用型LLM智能体故障注入测试的证据有效性问题，通过测量协议检验反驳性观测结果能否被智能体实际感知，揭示了故障采纳结论可能存在的偏差。

    

    故障注入越来越多地被用于评估工具使用型LLM智能体的可靠性。然而，当智能体自身决定在执行过程中哪些权威观测结果变得可见时，对于故障采纳结果应如何解释的研究还很有限。本文对智能体故障注入评估中的这一证据有效性问题进行了系统性研究。我们开发了一种测量协议，该协议规定了哪些观测结果可以驳斥注入的断言，判断这些观测结果能否在受影响的事实首次被使用之前变得可见，并记录被评估的执行是否实际实现了这一条件。我们构建了SSCBench作为该协议的一个实例，并在两个τ-bench环境中对四种故障算子和五种智能体配置共1,191次故障执行进行了评估。我们的实验表明，一个被采纳的故障案例和智能体配置可能在实质上产生显著差异（摘要原文在此处截断）。

    arXiv:2610.11514v1 Announce Type: new  Abstract: Fault injection is increasingly used to evaluate the reliability of tool-using LLM agents. However, there has been limited study of how fault-adoption results should be interpreted when the agent itself determines which authoritative observations become visible during execution. In this paper, we present a systematic study of this evidential validity problem in agent fault-injection evaluation. We develop a measurement protocol that specifies what observations can refute an injected assertion, determines whether they can become visible before the affected fact is first used, and records whether the evaluated execution actually realizes this condition. We construct SSCBench as an instantiation of the protocol and evaluate four fault operators and five agent configurations over 1,191 faulted executions in two $\tau$-bench environments. Our experiments show that an admitted fault case and agent configuration can realize substantially differ
    
[^24]: 评估本地语言模型智能体在可复现数据工程中的表现：一项针对移动性工作流的实证软件工程研究

    Evaluating Local Language Model Agents for Reproducible Data Engineering: An Empirical Software Engineering Study of Mobility Workflows

    [https://arxiv.org/abs/2610.11482](https://arxiv.org/abs/2610.11482)

    该研究构建了一个包含十五个移动性工作流任务的基准测试，通过确定性检查器系统评估了十种本地部署的开源权重LLM智能体在生成正确且可复现数据工程制品方面的能力，并量化了闭环工作区条件等因素的影响。

    

    背景：大语言模型（LLM）智能体正被越来越多地用作软件和数据工程助手，然而关于可本地部署的开源权重智能体的证据仍然有限。现有评估往往侧重于文本回复或孤立的代码生成，而非完整工程制品的有效性。目标：我们评估本地LLM智能体能否产出正确且可复现的数据工程制品，量化闭环工作区条件的影响，并考察模型规模、架构、量化、运行时、工具使用和失败模式方面的权衡。方法：我们构建了一个包含十五个移动性工作流任务的基准测试，涵盖数据发现、连接器、运输数据处理、语义增强、特征工程、验证、可视化和报告生成。确定性检查器用于评估生成的脚本、数据表、结构化文件、图形和报告。我们对十种本地配置进行了评估。

    arXiv:2610.11482v1 Announce Type: cross  Abstract: Context: Large language model (LLM) agents are increasingly used as software and data-engineering assistants, yet evidence about locally deployable open-weight agents remains limited. Existing evaluations often emphasize textual responses or isolated code generation rather than the validity of complete engineering artifacts.   Objectives: We evaluate whether local LLM agents can produce correct and reproducible data-engineering artifacts, quantify the effect of a closed-loop workspace condition, and examine trade-offs in model scale, architecture, quantization, runtime, tool use, and failure.   Methods: We introduce a benchmark of fifteen mobility-workflow tasks covering data discovery, connectors, transport-feed processing, semantic enrichment, feature engineering, validation, visualization, and reporting. Deterministic checkers assess generated scripts, tables, structured files, figures, and reports. Ten local configurations are eval
    
[^25]: 面向嵌入式软件开发的LLM智能体闭环评估

    Closed-loop evaluation of LLM agents for embedded software development

    [https://arxiv.org/abs/2610.11447](https://arxiv.org/abs/2610.11447)

    该论文提出了一个包含五个嵌入式控制任务和四种反馈场景的基准测试，用于闭环评估LLM编码智能体在嵌入式软件开发中实现并自我验证设备行为的能力。

    

    大语言模型（LLM）正越来越多地被部署为编码智能体，用于编辑文件、运行构建和测试、检查执行结果并迭代修复软件。嵌入式固件是一个具有挑战性的目标，因为其正确性取决于传感、时序和安全约束下的闭环行为，而不仅仅是静态源代码的质量。然而，针对嵌入式智能体的评估仍然有限，且往往侧重于一次性代码生成或离线正确性验证。我们提出了一个用于嵌入式编码智能体闭环评估的基准测试。每个任务提供一个纯文本的工程描述、受约束的工作空间以及可见的构建与运行时界面。智能体必须将需求转化为具体实现和自我验证步骤，然后不断迭代，直到实现所需的设备行为。该测试套件包含五个嵌入式控制任务和四种反馈场景：一次性生成、贴近现实的自我验证、CI风格的红/绿反馈……

    arXiv:2610.11447v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as coding agents that edit files, run builds and tests, inspect execution results, and repair software iteratively. Embedded firmware is a demanding target because correctness depends on closed-loop behavior under sensing, timing, and safety constraints, not only on static source quality. Yet embedded-agent evaluation remains limited and often emphasizes one-shot synthesis or offline correctness.   We present a benchmark for closed-loop evaluation of embedded coding agents. Each task provides a plain-text engineering description, constrained workspace, and visible build-and-runtime surface. The agent must translate requirements into implementation and self-verification steps, then iterate until the required device behavior is achieved. The suite contains five embedded-control tasks and four feedback scenarios: one-shot generation, realistic self-verification, CI-style red/green fee
    
[^26]: 刻画基于大语言模型代码生成中的过度自信失败

    Characterizing Overconfident Failure in LLM-Based Code Generation

    [https://arxiv.org/abs/2610.11300](https://arxiv.org/abs/2610.11300)

    本文在四个开源代码模型和三个执行基准上，揭示了代码大语言模型中错误程序常以与正确程序相当的 token 级置信度生成的过度自信问题，并系统评估了现有不确定性指标能否可靠预测执行正确性。

    

    大语言模型（LLMs）在自动化代码生成中的应用日益广泛，但生成的程序可能看起来在语法上合理，却在基于执行的正确性检查中失败。现有的验证方法（如测试和程序分析）仍然不可或缺，但往往不完整、成本高昂，或只能在生成之后才被应用。因此，模型自身产生的不确定性是一种天然的早期可靠性信号。本文研究了代码大语言模型中的过度自信困境，即错误的程序往往以与正确程序相当的 token 级置信度被生成。我们在四个开源代码模型和三个基于执行的基准上研究了这一困境。我们的分析首先调查现有不确定性指标能否可靠地作为代码生成中执行正确性的代理指标，然后在全局和局部 token 层面刻画过度自信现象，探讨错误的程序是否仍然存在于……（原文摘要在此处截断）

    arXiv:2610.11300v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used for automated code generation, but generated programs can appear syntactically plausible while still failing execution-based correctness checks. Existing validation methods, such as testing and program analysis, remain essential but are often incomplete, costly, or applied only after generation. Model-derived uncertainty is therefore a natural early reliability signal. This paper studies the dilemma of overconfidence in code LLMs where incorrect programs are often generated with token-level confidence comparable to correct programs. We study this dilemma across four open-source code models and three execution-based benchmarks. Our analysis begins by investigating whether existing uncertainty metrics provide reliable proxies for execution correctness in code generation. We then characterize overconfidence at both global and local token levels, asking whether incorrect programs remain in
    
[^27]: 量子编译器变换的逆态测试

    Retromorphic Testing of Quantum Compiler Passes

    [https://arxiv.org/abs/2610.11195](https://arxiv.org/abs/2610.11195)

    针对量子编译器变换正确性难以验证的问题，本文系统分析了四大量子编程框架的单元测试现状，并提出了基于逆态测试的编译器变换自动化验证方法。

    

    量子编译器在将高级量子程序转换为优化的、与硬件兼容的电路方面发挥着关键作用。然而，验证编译器变换的正确性仍然具有挑战性，因为确定大型、深度纠缠量子电路的预期输出在计算上是难以处理的。当编译器变换修改本已复杂的电路结构时，这一挑战进一步加剧，使得对变换后电路的人工验证变得不切实际。在这项工作中，我们对四个量子编程框架（PennyLane、Qiskit、Cirq 和 pytket）中量子编译器变换的单元测试进行了系统分析。我们的发现表明，现有验证主要依赖于程序内容和程序度量断言，且测试电路通常规模小、深度浅。基于这些观察，我们提出了一种基于逆态测试与……的量子编译器变换自动化验证测试方法。

    arXiv:2610.11195v1 Announce Type: cross  Abstract: Quantum compilers play a critical role in transforming high-level quantum programs into optimized, hardware-compatible circuits. However, verifying the correctness of compiler passes remains challenging, as determining the expected output of large, deeply entangled quantum circuits is computationally intractable. This challenge is further amplified when compiler passes modify already complex circuit structures, making manual validation of transformed circuits impractical.   In this work, we perform a systematic analysis of unit tests for quantum compiler passes in four quantum programming frameworks (PennyLane, Qiskit, Cirq, and pytket). Our findings indicate validation is dominated by program-content and program-metric assertions, and test circuits are generally small and shallow. Motivated by these observations, we introduce a testing methodology for automated validation of quantum compiler passes based on retromorphic testing and pr
    
[^28]: 谁来承担评审成本？AI创作拉取请求中的分诊、公平性与问责制

    Who Pays the Review Cost? Triage, Fairness, and Accountability in AI-authored Pull Requests

    [https://arxiv.org/abs/2610.11179](https://arxiv.org/abs/2610.11179)

    该研究通过对31个国家239名开发者的问卷调查发现，AI作者身份本身并非被拒信号，评审者是否投入评审精力取决于AI拉取请求是否以可问责贡献的形式出现，包括范围有限、理由基于项目、超越CI的验证以及贡献者的响应性。

    

    AI编码智能体正从本地代码辅助走向基于拉取请求的工作流，在此类工作流中，生成的贡献必须在现有项目规范下接受评审、解释和维护。尽管近期研究已开始刻画AI创作的拉取请求（AIPR），但关于评审者如何治理其进入评审的流程、AI作者身份如何重塑可信度与公平性，以及何种接纳机制能够保护评审的可持续性，人们仍知之甚少。本研究对来自31个国家、具有代码评审经验且对AIPR接触程度不同的239名从业者开展了混合方法问卷调查。在调查所引出的情景与自我报告中，AI作者身份并非一概拒绝的信号。相反，受访者表示评审工作量取决于AIPR是否以一项可问责的贡献的形式到来，即具备有限的范围、基于项目的理由、超越持续集成（CI）的验证、贡献者的响应能力以及（摘要原文在此处截断）。

    arXiv:2610.11179v1 Announce Type: new  Abstract: AI coding agents are moving from local code assistance into pull-based workflows, where generated contributions must be reviewed, explained, and maintained within existing project norms. Although recent work has begun to characterize AI-authored pull requests (AIPRs), less is known about how reviewers govern their entry into review, how AI authorship reshapes credibility and fairness, and what intake mechanisms protect review sustainability. We report a mixed-method questionnaire survey of 239 practitioners from 31 countries with code-review experience and varying exposure to AIPRs. In the scenarios and self-reports elicited by the survey, AI authorship was not a categorical rejection signal. Instead, respondents described review effort as conditional on whether an AIPR arrived as an accountable contribution, with bounded scope, project-grounded rationale, validation beyond Continuous Integration (CI), contributor responsiveness, and ide
    
[^29]: 技能星座：追踪GitHub上智能体技能的供应链

    Skill Constellations: Tracing the Supply Chain of Agent Skills on GitHub

    [https://arxiv.org/abs/2610.11169](https://arxiv.org/abs/2610.11169)

    本文构建了首个基于git历史的智能体技能复制时间网络，覆盖GitHub上219万余次技能采用，揭示了少数源头仓库主导技能复制、GitHub星标无法识别这些源头、且副本几乎不随源头同步更新导致安全修复难以传播的关键问题。

    

    智能体技能是指AI编码智能体（如Claude Code和Codex）以其用户权限运行的SKILL.md指令和脚本。开发者通过在仓库之间复制技能来共享它们，这使技能成为一个没有注册表、版本管理或来源追溯的软件供应链。因此，被复制技能的来源、安全修复的影响范围以及哪些仓库需要审查都是未知的。仅记录某一时间点哪些仓库持有某项技能的研究无法揭示谁从谁那里复制了它。我们贡献了首个带有时间戳的智能体技能复制网络，该网络基于GitSkills中每个SKILL.md的git历史构建，覆盖了GitHub上2,193,119次技能采用，并附带一个交互式查看器。研究发现：少数仓库是几乎所有技能复制的源头，而GitHub星标数量并不能识别出这些源头仓库；技能副本几乎从不随其源文件同步更新，因此源头处的安全修复很少能够传递到这些副本。

    arXiv:2610.11169v1 Announce Type: new  Abstract: Agent skills are SKILL.md instructions and scripts that AI coding agents such as Claude Code and Codex run with the permissions of their user. Developers share skills by copying them between repositories, which makes them a software supply chain without a registry, versions or provenance. The origin of a copied skill, the reach of a security fix and the repositories that warrant review are therefore unknown. Studies that record which repositories hold a skill at a single point in time cannot reveal who copied it from whom. We contribute the first dated copy network of agent skills, built from the git history of every SKILL.md in GitSkills and covering 2,193,119 skill adoptions across GitHub, together with an interactive viewer. A few repositories are the source of almost all copies, and GitHub stars do not identify them. Skill copies almost never change with their source, and a fix at the source therefore rarely reaches them. We fit a mo
    
[^30]: IRONPROOF：基于SMT等价性验证的COBOL到Python代码转换

    IRONPROOF: COBOL-to-Python Transpilation with SMT-Based Equivalence Checking

    [https://arxiv.org/abs/2610.11073](https://arxiv.org/abs/2610.11073)

    IRONPROOF系统通过将COBOL和Python代码编码为Z3公式并利用SMT求解器进行等价性检查，为COBOL到Python的自动代码转换提供机器可验证的形式化等价性证明或反例，在2345个COBOL文件中成功证明了606个转换的等价性。

    

    将COBOL翻译成现代语言可能会改变程序的计算结果，而大语言模型（LLM）的翻译无法保证等价性。我们提出了IRONPROOF，它将COBOL解析为中间表示，生成Python代码，并将两者编码为基于共享输入的Z3公式，最终输出机器可验证的等价性证书（UNSAT）或反例（SAT）。在2,345个COBOL文件（包括GnuCOBOL测试用例、NIST CCVS85、开源代码集合以及我们生成或编写的程序）中，782个进入验证路径：606个（77.5%）被证明等价，101个被部分验证，没有被反驳的案例，75个在我们的流水线中失败并保留在分母中。在独立编写的程序上，验证率为52.6%（291个中的153个），而在我们自己编写的程序上为92.3%。在153个独立证明中，37个仅覆盖了读取编码器未建模输入的程序的单次执行，而在所有606个证明中，仅有14个对已证明输出所量化的输入进行了处理。

    arXiv:2610.11073v1 Announce Type: new  Abstract: Translating COBOL to a modern language can change what a program computes, and LLM translations carry no guarantee of equivalence. We present IRONPROOF, which parses COBOL into an intermediate representation, generates Python, encodes both as Z3 formulas over shared inputs, and emits either a machine-checkable equivalence certificate (UNSAT) or a counterexample (SAT). On 2,345 COBOL files (GnuCOBOL tests, NIST CCVS85, open-source collections, and programs we generated or wrote), 782 enter the checking path: 606 (77.5%) are proved equivalent, 101 are partially verified, none is refuted, and 75 fail inside our pipeline and stay in the denominator. On independently authored programs the rate is 52.6% (153 of 291), against 92.3% on programs we wrote. Of the 153 independent proofs, 37 cover a single execution of a program that reads input the encoder does not model, and across all 606 proofs only 14 quantify over an input that a proved output
    
[^31]: 追踪代码中的面包屑：意外提交的临时日志揭示了开发者的代码理解行为

    Following Breadcrumbs in Code: What Accidentally Committed Ad-Hoc Logs Reveal about Developer Comprehension

    [https://arxiv.org/abs/2610.11041](https://arxiv.org/abs/2610.11041)

    该研究首次通过挖掘意外提交的代码和直播编程会话，构建了覆盖 Java、JavaScript 和 Python 的大型数据集，系统性地揭示了开发者利用临时日志来理解程序行为的使用模式。

    

    开发人员经常插入临时的打印或日志语句（称为**临时日志（ad-hoc logs）**），以便在运行时更好地理解程序行为，尤其是在面对意外问题或复杂控制流时。尽管这几乎是一种普遍的做法，但由于这类日志转瞬即逝，系统性研究一直很有限：它们通常只存在于本地环境中，并在代码提交前被删除，因此难以捕获。在这项工作中，我们通过挖掘开发人员无意中留下临时日志并在之后删除的意外提交，同时分析直播编程会话以观察这些日志在实际中的使用，来应对这一挑战。利用这些方法，我们构建了一个涵盖三种主要编程语言（Java、JavaScript 和 Python）的大型数据集，从而能够对日志记录实践进行大规模研究。我们的分析揭示了开发人员在何处以及如何使用这些日志的共性与语言特定模式。

    arXiv:2610.11041v1 Announce Type: new  Abstract: Developers frequently insert temporary print or log statements, known as **ad-hoc logs**, to better understand program behavior at runtime, particularly when facing unexpected issues or complex control flows. Despite being a nearly universal practice, systematic study has been limited because these logs are ephemeral: they usually remain only in local environments and are removed before code is committed, making them difficult to capture. In this work, we addressed this challenge by mining accidental commits where developers unintentionally left ad-hoc logs and later deleted them, and by analyzing live-streamed programming sessions to observe their use in practice.   Using these methods, we constructed a large dataset across three major programming languages (Java, JavaScript, and Python), enabling the large-scale investigation of logging practices.   Our analysis reveals both common and language-specific patterns in where and how develo
    
[^32]: 概率感知、确定性权威：将模型生成的观测纳入经过充分性检验的治理合约

    Probabilistic Sensing, Deterministic Authority: Admitting Model-Produced Observations into Sufficiency-Checked Governance Contracts

    [https://arxiv.org/abs/2610.10978](https://arxiv.org/abs/2610.10978)

    提出一种将模型输出的观测以带分数记录形式纳入确定性治理合约的框架：阈值接纳策略将分数映射为真/假/未知（未知即拒绝），感知改变裁决的概率受各感知字段错误接纳率与未知率之和（并集界）约束，并通过36,000次模型调用的注册研究验证了该界的有效性。

    

    当权威合约所需的某个字段仅存在于非结构化证据中时，模型可以对其进行感知。我们仅将模型的输出作为附带分数的观测记录予以接纳。一个接纳策略（其阈值在声明的假阳性上限下于留存数据集上拟合）将每个分数映射为“真”、“假”或“未知”，其中“未知”即拒绝。随后由一个确定性的、经过充分性检验的合约做出最终裁决。感知改变裁决的概率受限于合约各感知字段的错误接纳率与未知率之和，这是以合约为索引的并集界推理的一个实例。最小化该估计界是在多个充分合约之间进行选择的有效成本模型。在一项针对两个构建领域、两个传感器族（共36,000次模型调用）的注册研究中，没有任何实验单元推翻该界。由感知引发的“拒绝转允许”变化首次出现在该研究计划中：21,000个测试裁决中出现了13例。

    arXiv:2610.10978v1 Announce Type: cross  Abstract: When a field that an authority contract needs exists only in unstructured evidence, a model can sense it. We admit the model's output only as an observation record with a score. An admission policy, with thresholds fitted on a held-out split at a declared false-positive ceiling, maps each score to true, false or unknown. Unknown denies. A deterministic, sufficiency-checked contract decides. The probability that sensing changes the verdict is bounded by the sum, over the contract's sensed fields, of the admitted-wrong and unknown rates. This is an instantiation of union-bound reasoning, indexed by the contract. Minimising the estimated bound is a valid cost model for choosing among sufficient contracts. In a registered study on two constructed domains with two sensor families (36,000 model calls), no cell refuted the bound. Deny-to-allow changes from sensing appeared for the first time in this programme: 13 of 21,000 test verdicts, all 
    
[^33]: 跨提供商审查作为编码智能体的运行时契约：一项受控试点与故障注入研究

    Cross-Provider Review as a Runtime Contract for Coding Agents: A Controlled Pilot and Fault-Injection Study

    [https://arxiv.org/abs/2610.10961](https://arxiv.org/abs/2610.10961)

    提出将跨提供商代码审查规范化为一种运行时契约（涵盖独立资源池、有界执行、明确失败状态与持久证据等条款），并通过受控试点和故障注入研究验证了其有效发现缺陷的能力。

    

    编码智能体越来越多地共享同一工作站，同时分别使用不同的提供商和订阅额度。第二个智能体可以检查已完成的答案，但这次调用会消耗另一个资源池，且可能不会产生实质性的发现。我们描述了一种咨询式的跨提供商审查契约：独立的资源池、有界的执行、受限的审查者能力、完整的输入交付、可用的语义输出、明确的失败状态以及持久的每次尝试证据。在一项包含20对开发回合的受控、由智能体自行执行的试点研究中，其中8个回合产生了实质性的审查发现（95%精确区间为19.1%–63.9%）。对两个审查后端进行的边界条件扫描复现了此前发现的部分输入假成功问题：四个截断级别在历史上通过了，但在修复后失败了。该扫描还发现并修复了进程回收过程中出现的取消问题。在真实的CLI探测中，Claude没有写入工具；Codex尝试写入……（摘要原文在此处截断）

    arXiv:2610.10961v1 Announce Type: cross  Abstract: Coding agents increasingly share a workstation while drawing on separate providers and subscription allowances. A second agent can inspect a completed answer, but the call spends another pool and may provide no substantive finding. We describe an advisory cross-provider review contract: distinct resource pools, bounded execution, restricted reviewer capabilities, complete input delivery, usable semantic output, explicit failure states and durable per-attempt evidence. In a controlled, agent-authored pilot of 20 paired development turns, eight had a material reviewer finding (95% exact interval 19.1-63.9%). A boundary-condition scan across both reviewer backends reproduced a previously discovered false success on partial input: four truncation levels passed historically and failed after repair. The scan also found and repaired cancellation during process reaping. In real CLI probes, Claude had no writing tools; Codex attempted writes in
    
[^34]: 当缺陷级联时：理解JavaScript引擎中的漏洞与利用链

    When Flaws Cascade: Understanding Vulnerabilities and Exploitation Chains in JavaScript Engines

    [https://arxiv.org/abs/2610.10844](https://arxiv.org/abs/2610.10844)

    本文首次对JavaScript引擎漏洞进行了全面实证研究，构建了涵盖2017-2024年四个主流引擎的241个漏洞数据集，建立了症状与根本原因的分类体系，并系统分析了漏洞特征及潜在利用链策略。

    

    JavaScript引擎是现代网络浏览器的核心组件，支持动态和交互式Web应用程序的执行。然而，其复杂性和广泛应用使其成为攻击者利用漏洞的主要目标。虽然现有研究主要集中于检测JavaScript引擎的漏洞，但在系统性地理解这些漏洞的特征方面仍存在显著空白，包括其症状、根本原因和可利用性。本文通过首次对JavaScript引擎漏洞进行全面的实证研究来填补这一空白，调查其特征和潜在的利用策略。我们构建了一个数据集，涵盖2017年至2024年间四个主流JavaScript引擎中的241个漏洞。通过深入分析，我们首先建立了症状和根本原因的分类体系。在此基础上，我们研究了利用链的构建……

    arXiv:2610.10844v1 Announce Type: cross  Abstract: JavaScript engines are pivotal to modern web browsers, enabling the execution of dynamic and interactive web applications. However, their complexity and widespread adoption make them prime targets for attackers exploiting vulnerabilities. While existing research has focused on detecting vulnerabilities of JavaScript engines, a significant gap remains in systematically understanding the characteristics of these vulnerabilities, including their symptoms, root causes, and exploitability. This paper bridges this gap by presenting the first comprehensive empirical study on vulnerabilities in JavaScript engines, investigating their characteristics and potential exploitation strategies.   We construct a dataset comprising 241 vulnerabilities across four mainstream JavaScript engines from 2017 to 2024. Through in-depth analysis, we first develop taxonomies for symptoms and root causes. Building on this understanding, we investigate the exploit
    
[^35]: DITTO：一种上下文感知的基于 Pickle 的预训练模型扫描器，用于高效安全审计

    DITTO: A Context-aware Pickle-based Pre-Trained Model Scanner for Effective Security Audits

    [https://arxiv.org/abs/2610.10735](https://arxiv.org/abs/2610.10735)

    本文提出首个基于栈的上下文感知 Pickle 预训练模型扫描器 DITTO，通过忠实跟踪 Pickle 虚拟机状态转换并执行上下文感知语义分析来推断模型意图，同时构建了包含 959 个良性模型和 92 个恶意模型的 PickleBench 基准，有效弥合了现有扫描器在覆盖率与精确度之间的差距。

    

    预训练模型（PTMs）通常以序列化二进制文件的形式分发，但其重用往往使软件供应链面临反序列化攻击的风险。尽管更安全的序列化格式不断涌现，但不安全的 Pickle 格式仍然普遍存在：我们对超过 10,000 个热门 Hugging Face 仓库的分析表明，其中 9.3% 依赖 Pickle。虽然已有许多防御机制被提出，但最先进的模型扫描器存在覆盖率与精确度之间的差距，即遗漏安全敏感行为并产生过多误报。本文提出了 DITTO，这是首个基于栈的、上下文感知的 Pickle 预训练模型扫描器。DITTO 忠实地跟踪 Pickle 虚拟机的状态转换，并执行上下文感知的语义分析以推断模型意图。我们还提出了 PickleBench，这是一个包含 959 个良性模型和 92 个恶意真实世界模型的基准数据集，其中包括现有工具此前未能检测到的扩展注册表攻击。

    arXiv:2610.10735v1 Announce Type: cross  Abstract: Pre-trained models (PTMs) are widely distributed as serialized binaries, but their reuse often exposes software supply chains to deserialization attacks. Despite the emergence of safer serialization formats, the unsafe Pickle format remains prevalent: our analysis of over 10,000 popular Hugging Face repositories reveals that 9.3% rely on Pickle. While many defense mechanisms have been proposed, state-of-the-art model scanners suffer from a coverage-precision gap, missing security-sensitive behaviors and generating excessive false alerts. In this paper, we introduce DITTO, the first stack-based, context-aware scanner for Pickle-based PTMs. DITTO faithfully tracks Pickle virtual machine state transitions and performs context-aware semantic analysis to infer model intentions. We also present PickleBench, a benchmark of 959 benign and 92 malicious real-world models, including extension registry attacks previously missed by existing tools. 
    
[^36]: 在执行时点应用安全设计：受治理的安全需求如何影响AI生成代码的安全性

    Applying Security by Design at the Point of Execution: How Governed Security Requirements Affect the Security of AI-Generated Code

    [https://arxiv.org/abs/2610.10659](https://arxiv.org/abs/2610.10659)

    该研究证明，在代码生成执行时点通过MCP服务器向AI代码生成器提供来自受治理安全设计知识库的安全需求，可将通过全部安全测试的任务比例从44.1%提升至78.0%，并将BaxBench上功能正确且无漏洞利用的解决方案比例从65%提升至86%。

    

    安全设计理念要求在编写代码之前定义安全需求。而安全的代码基准测试通常测量的是相反的情况：智能体在不带需求的情况下接收任务，并用它未曾见过的安全测试进行评分。我们测量了当安全需求——从受治理的安全设计知识库中选取，并通过模型上下文协议（MCP）服务器在执行时点提供给代码生成器时——会带来什么变化。在使用语言模型评判器打分的DualGauge上，我们运行了59个Python任务：通过所有安全测试的任务比例从44.1%上升到78.0%，通过的安全测试比例从77.4%上升到93.0%。在运行功能测试和容器中真实漏洞利用的BaxBench上，我们运行了28个后端场景：在功能正确的解决方案中，没有成功漏洞利用的比例从65%上升到86%，与原作者的Oracle安全提醒方法效果相当。

    arXiv:2610.10659v1 Announce Type: new  Abstract: Security by design asks that security requirements are defined before code is written. Secure-code benchmarks typically measure the opposite situation: the agent receives the task without requirements and is scored with security tests it has not seen. We measured what changes when security requirements, selected from a governed security-by-design knowledge base (SbD-ToE) and delivered through a Model Context Protocol (MCP) server, are given to the generator at the point of execution. On DualGauge, which scores with a language-model judge, we ran 59 Python tasks: the share of tasks passing all security tests rose from 44.1% to 78.0% and the share of security tests passed from 77.4% to 93.0%. On BaxBench, which runs functional tests and real exploits in containers, we ran 28 backend scenarios: among functionally correct solutions, the share with no successful exploit rose from 65% to 86%, comparable to the authors' Oracle Security Reminder
    
[^37]: 大型语言模型能否模拟新手程序员的误解？

    Can LLMs Simulate Novice Programmers' Misconceptions?

    [https://arxiv.org/abs/2610.10656](https://arxiv.org/abs/2610.10656)

    本研究评估了13个大语言模型模拟新手编程误解的能力，发现前沿模型表现可靠，但小模型难以进行动态执行，代码调优模型在模拟误解时会回退到正确执行，且判断题格式的诊断任务显著易于开放式生成任务。

    

    我们在代码追踪问题上评估了13个大语言模型在生成、解决、模拟和诊断新手编程误解方面的能力。研究发现，前沿模型能够可靠地完成这些任务，而小型模型（参数量≤14B）在动态执行方面存在困难。值得注意的是，经过代码调优的模型在模拟误解时会失败，因为它们会回退到正确的执行方式。此外，误解诊断在判断题（对/错）格式下比在开放式生成任务中要容易得多。

    arXiv:2610.10656v1 Announce Type: new  Abstract: We evaluate 13 LLMs on generating, solving, simulating, and diagnosing novice programming misconceptions on code-tracing problems. While frontier models reliably perform these tasks, small models ($\le$14B) struggle with dynamic execution. Notably, code-tuned models fail during misconception simulation by reverting to correct execution. Finally, misconception diagnosis is significantly easier in True/False formats than in open-ended generation tasks.
    
[^38]: SoK（知识的系统化）：通用准则产品评估中的失效模式——失效分类法与可评估性设计指南

    SoK: Failure Modes in Common Criteria Product Evaluation - A Taxonomy and Design-for-Evaluability Guidance

    [https://arxiv.org/abs/2610.10644](https://arxiv.org/abs/2610.10644)

    本文从评估者的操作视角出发，对通用准则（CC）产品评估中反复出现的跨厂商失效模式进行了系统化梳理，构建了失效模式分类法，并提供了面向可评估性的设计指南。

    

    通用准则（Common Criteria，CC；ISO/IEC 15408）是评估 IT 产品安全性的主要国际框架，其证书是政府、国防和受监管行业采购的准入门槛。然而，评估过程经常停滞、失败，或产出的证书所承诺的安全保障无法在真实部署环境中经受考验。已有研究从两个方向切入这一问题：一是对激励错位与“安全表演”（security theatre）的经济学批判；二是近期对已认证产品中漏洞进行量化的数据驱动研究。相比之下，很少有工作从评估者的操作视角出发——即针对通用准则工作单元本身反复出现的、跨厂商的失效——对这一问题进行系统化梳理。本文对通用准则产品评估中的失效模式进行了知识系统化（SoK）。研究基于标准本身、通用评估方法学（ISO/IEC 18045）、公开的 NIAP 保护轮廓以及公开发表的……（原文摘要在此处截断）

    arXiv:2610.10644v1 Announce Type: cross  Abstract: The Common Criteria (CC; ISO/IEC 15408) is the principal international framework for evaluating the security of IT products, and its certificates gate procurement across government, defense, and regulated industry. Yet evaluations routinely stall, fail, or yield certificates whose assurances do not survive real-world deployment. Prior work has approached this from two directions: economic critiques of misaligned incentives and "security theatre"; and recent data-driven studies that quantify vulnerabilities in already-certified products. Comparatively little systematizes the problem from the evaluator's operational vantage - recurring, cross-vendor failures against the Common Criteria work units themselves. This paper presents a systematization of knowledge (SoK) of failure modes in Common Criteria product evaluation. Drawing on the standard, the Common Evaluation Methodology (ISO/IEC 18045), public NIAP Protection Profiles, and publish
    
[^39]: 可见推理并非万能优化器：分析式代码生成中依赖角色设定与思考方式的效应

    Visible Reasoning Is Not a Universal Optimizer: Persona- and Thinking-Dependent Effects in Analytics Code Generation

    [https://arxiv.org/abs/2610.10639](https://arxiv.org/abs/2610.10639)

    该论文通过跨 SQL 与 pandas 双语言、交叉角色设定与多种思考指令的受控执行基准实验发现，显式思维链推理并非普遍有效，其效果因角色表述、目标语言和思考格式而异，“用 SQL/Python 思考”这类匹配目标语言的指令并不能可靠带来提升。

    

    可见的思维链通常被视为一种普遍有用的推理指令，然而分析式代码生成同时涉及自然语言的歧义性、数据模式的对接、目标语言的约束以及模型自身的推理行为。由于同一个分析请求可以用两种不同的目标语言表达——SQL 与 Python（pandas）——这一设置为检验一个常见但未被充分验证的假设提供了天然的测试场景：即当可见推理的表示形式与所请求的目标语言相匹配时（如“用 SQL 思考”或“用 Python 思考”），推理会更有效。与“逐步思考”等通用指令一起，这类建议在受控的、基于实际执行结果的对比实验中仍未得到充分评估。我们研究了一个查询任务相互匹配的 SQL-pandas 基准，该基准交叉考察了角色设定措辞、目标语言、可见 CoT 格式、控制前缀、直接生成以及内部推理等多种配置。实验结果并不支持将可见推理视为普遍有效的优化手段这一假设。

    arXiv:2610.10639v1 Announce Type: cross  Abstract: Visible Chain-of-Thought (CoT) is often treated as a broadly useful reasoning instruction, yet analytics code generation combines natural-language ambiguity, schema grounding, target-language constraints, and model-specific inference behavior. Because the same analytics request can be expressed in two distinct target languages-SQL and Python (pandas)-this setting provides a natural test of a common but under-examined assumption: that visible reasoning is more effective when its representation matches the requested target, as in "think in SQL" or "think in Python." Together with generic instructions such as "think step-by-step," such recommendations remain insufficiently evaluated under controlled, execution-based comparisons. We study a query matched SQL-pandas benchmark that crosses persona phrasing, target language, visible-CoT format, control prefixes, direct generation, and internal-reasoning configurations. The results do not supp
    
[^40]: 大语言模型在软件工程系统综述中的筛选性能是否已陷入停滞？

    Has LLM Screening Performance Stalled in Software Engineering Systematic Reviews?

    [https://arxiv.org/abs/2610.10633](https://arxiv.org/abs/2610.10633)

    该研究通过基准测试评估了八个新大语言模型在软件工程系统综述文献筛选中的表现，发现新模型仅比旧模型略有提升（平均MCC从0.347升至0.365），表明LLM筛选性能进展缓慢，且不同研究间的差异仍大于模型间的差异。

    

    系统综述中的文献筛选步骤是人工且耗时的。先前的研究已经探索了利用大语言模型来自动化这一步骤，但由于大语言模型发展迅速，早期的性能结论可能已无法准确反映其当前的筛选能力。我们使用现有的软件工程系统综述筛选基准作为数据，并通过功效抽样构建了一个更小的新数据集，以便以更低的成本进行评估。基于这些数据，我们评估了八个新的大语言模型的筛选性能。此外，我们测试了不同的提示词，分析了大语言模型在筛选决策和筛选标准上的一致性，并检验了细化纳入与排除标准对筛选性能的影响。结果显示，八个新模型仅比七个旧模型略有提升：跨二级研究的平均MCC从0.347上升到0.365。二级研究之间的差异仍然大于大语言模型之间的差异。

    arXiv:2610.10633v1 Announce Type: cross  Abstract: Screening in systematic reviews (SRs) is manual and time-consuming. Prior work has explored large language models (LLMs) for automating this step, but LLMs are evolving rapidly, so earlier performance claims may no longer accurately reflect their screening performance. We used an existing software engineering SR screening benchmark (SESR-Eval) as our data. We also power-sampled a new, smaller dataset (SESR-Eval-Mini) that allows evaluation at lower costs. Using this data, we evaluated eight new LLMs for screening performance. Additionally, we tested different prompts, analyzed LLM agreement in screening decisions and criteria, and examined the effect of refining the inclusion and exclusion criteria on screening performance. The eight new LLMs performed marginally better than the seven old ones: avg. MCC across secondary studies rose from 0.347 to 0.365. Differences between secondary studies are still bigger than between LLMs. Computing
    
[^41]: 无规则数字孪生：通过标准化框架与技术实现声明式决策

    Ruleless Digital Twins: Toward Declarative Decision-Making Through Standardized Frameworks and Technologies

    [https://arxiv.org/abs/2610.10631](https://arxiv.org/abs/2610.10631)

    本文提出无规则数字孪生（RDTs）概念，通过标准化框架和技术，基于纯声明式的用户规范自动生成最优决策，以替代日益复杂难维护的传统基于规则的数字孪生决策模型。

    

    数字孪生（Digital Twins，DTs）可以被视为物理对象的数字对应物，或更广泛地说，是孪生目标（Twinning Targets，TTs）的数字映射。为了对孪生目标的属性实施变更和优化，数字孪生需要一个决策要素。传统上，许多领域（如家庭自动化）采用基于规则的决策模型，根据期望的孪生目标状态以命令式方式定义数字孪生的动作。随着用户需求的不断演变，基于规则的模型在开发和维护方面变得越来越复杂，尤其是在需要应对动态变化系统（如受天气或动态能源价格影响的系统）进行优化的场景中。我们提出了一种替代方案——无规则数字孪生（Ruleless Digital Twins，RDTs），它能够基于纯声明式的用户规范自动生成最优决策，类似于已成熟的无规则模型预测控制方法。其实现方式结合了语义知识（摘要在此处截断）

    arXiv:2610.10631v1 Announce Type: new  Abstract: Digital twins (DTs) can be thought of as digital counterparts of physical objects, or more generally, twinning targets (TTs). To enact changes on and optimize for properties of their TTs, DTs use an element of decision-making. Traditionally, many domains, such as home automation, utilize rule-based decision-making models to imperatively define DT actions based on desired TT conditions. With evolving user specifications, rule-based models have become increasingly complex to develop and maintain, particularly in scenarios involving optimizations under dynamically changing systems, such as those affected by weather or dynamic energy pricing. We present an alternative in the form of ruleless digital twins (RDTs) that automatically produce optimal decisions with respect to purely declarative user specifications, similarly to the well-established ruleless approach of model predictive control. They do so through a combination of a semantic know
    
[^42]: Agent4RE：一个用于端到端软件需求工程与基准测试的自精炼多智能体框架

    Agent4RE: A Self-Refining Multi-agent Framework for End-to-End Software Requirements Engineering and Benchmarking

    [https://arxiv.org/abs/2610.10628](https://arxiv.org/abs/2610.10628)

    提出了Agent4RE——一个具备双重迭代自精炼机制的多智能体需求工程框架，并构建了首个覆盖需求获取到生成全流程的端到端需求工程基准数据集RE-E2E。

    

    现有的基于大语言模型（LLM）的软件需求工程（RE）方法通常依赖于基础的提示策略或初级的智能体协作，未能充分发挥多智能体系统的潜力。同时，现有数据集主要关注孤立的子任务，如需求提取、分类和完整性检测，缺乏一个覆盖从需求获取到需求生成全过程的端到端需求工程基准。我们提出了Agent4RE——一个自精炼的多智能体需求工程系统，它协调多个专门化智能体并引入两个迭代改进循环。为支持评估，我们构建了RE-E2E——一个基于人工编写需求规格说明的真实世界数据集，实现了对需求工程工作流的端到端评估。在此基础上，我们进一步提出了两个增强版Agent4RE，分别引入自主自精炼机制或结构化人类反馈，并分析了它们的优势。

    arXiv:2610.10628v1 Announce Type: cross  Abstract: Existing LLM-based approaches for software Requirements Engineering (RE) typically rely on basic prompting strategies or rudimentary agent collaboration, under-utilizing the full potential of multi-agent systems. Meanwhile, available datasets focus on isolated subtasks, such as requirements extraction, classification, and completeness detection, leaving the absence of an end-to-end RE benchmark that spans from requirements elicitation to generation. We present Agent4RE - a self-refining multi-agent RE system that orchestrates specialized agents and incorporates two iterative improvement loops. To support evaluation, we construct RE-E2E - a real-world dataset built from human-written requirement specifications, enabling end-to-end assessment of RE workflows. Building on this foundation, we further propose two enhanced Agent4RE versions that incorporate either autonomous self-refinement or structured human feedback, and analyze their str
    
[^43]: WorldBench：评估大语言模型在Three.js体素世界生成上的能力

    WorldBench: Evaluating LLMs on Three.js Voxel World Generation

    [https://arxiv.org/abs/2610.10622](https://arxiv.org/abs/2610.10622)

    WorldBench通过让评判系统主动探索运行中的3D世界（控制时钟、环绕观察、派遣导航智能体取景）并将视觉观察与源代码相互交叉验证，解决了现有单一视角评判方法不可靠的问题，实现了对LLM生成的Three.js体素世界的可靠评估。

    

    arXiv:2610.10622v1 公告类型：cross 摘要：大语言模型现在已经能够以代码形式编写完整、可交互的3D世界，但对这些世界进行自动化评分却并不可靠。现有的评判方法只采用单一视角：要么由视觉-语言模型对少量渲染快照打分，要么由语言模型阅读源代码。在五个前沿模型生成的世界上，我们发现这两种视角在32%的必需项目上存在分歧，其中大部分是任何画面都无法展示的代码，而且固定视角会遗漏小型的特写内容。我们提出了WorldBench，一个针对开放式、由LLM生成的Three.js世界的基准与评判系统。从一个描述漂浮体素岛屿的提示词出发（包含十个生物群系、物理系统以及昼夜和四季循环），该评判系统会探索正在运行的世界，控制其时钟、绕轨道观察，并派遣一个导航智能体为每个生物群系取景，同时阅读代码来核实所见内容。两个信息通道都不会被单独信任：代码引用只有在源代码确实包含相应文本时才有效，而视觉声明则会……（原文在此处截断）

    arXiv:2610.10622v1 Announce Type: cross  Abstract: Large language models can now write complete, interactive 3D worlds as code, but grading those worlds automatically is unreliable. Existing judges take one view of the output: a vision-language model scores a few rendered snapshots, or a language model reads the source. On worlds written by five frontier models we find that the two views disagree on 32% of required items, mostly code that no frame shows, and that fixed views miss small close-up contents. We present WorldBench, a benchmark and judge for open-ended, LLM-generated Three.js worlds. From one prompt describing a floating voxel island with ten biomes, physics, and day/night and seasonal cycles, the judge explores the running world, controlling its clock, orbiting it, and sending a navigator agent to frame each biome, and reads the code for what it sees. Neither channel is trusted on its own: a code quote counts only if it is text the source contains, and visual claims are che
    
[^44]: TestJack：你应该相信编码基准测试的结果吗？通过评估器进化审计智能体编码基准

    TestJack: Should you trust the results in coding benchmarks? Agentic Coding Benchmarks Auditing via Evaluator Evolution

    [https://arxiv.org/abs/2610.10619](https://arxiv.org/abs/2610.10619)

    提出TestJack框架，通过评估器进化为每次试验动态生成针对性测试来审计智能体编码基准，揭示仅依赖固定单元测试可能高估LLM智能体的真实问题解决能力。

    

    大语言模型（LLM）智能体正在迅速重塑软件工程领域，与此同时新的代码基准测试也呈爆炸式增长。然而，几乎所有现有基准测试仍然依赖同一套沿用数十年的评判标准：如果一个解决方案通过了固定的单元测试集合，它就是正确的。这样的测试往往是不充分的：它们只检查任务所需的一部分内容，因此智能体可能对其进行奖励投机（reward hacking），或者在悄然遗漏所需行为的同时依然通过所有测试。结果是，更高的基准分数可能部分反映了对评估器的更好适应，而非更强的问题解决能力。现有工作专注于静态测试增强：它们在任何试验被观察之前就一次性强化每个任务的测试，因此忽视了真实试验实际上是如何失败的。我们提出了TestJack，一个超越固定测试来评估补丁的可扩展框架。对于每次试验，TestJack会针对补丁可能违反的提示要求生成测试，并仅保留通过的测试……（摘要在此截断）

    arXiv:2610.10619v1 Announce Type: cross  Abstract: Large language model (LLM) agents are rapidly reshaping software engineering, accompanied by an explosion of new code benchmarks. Yet nearly all existing benchmarks still rely on the same decades-old criterion: a solution is correct if it passes a fixed set of unit tests. Such tests are often insufficient: they check only part of what the task requires, so agents can reward hack them or silently miss required behavior while still passing every test. As a result, higher benchmark scores may partly reflect better adaptation to the evaluator rather than better problem solving. Existing works focus on static test augmentation: they strengthen each task's tests once, before any trial is seen, and thus overlook how real trials actually fail. We introduce TestJack, a scalable framework for evaluating patches beyond fixed tests. For each trial, TestJack generates tests targeting prompt requirements the patch may violate, retains only tests pas
    
[^45]: MRCert：基于类型特定掩码的对抗性补丁样本部署后补丁鲁棒性认证

    MRCert: Towards Post-deployment Patch Robustness Certification for Adversarially Patched Samples via Type-specific Masking

    [https://arxiv.org/abs/2610.10617](https://arxiv.org/abs/2610.10617)

    提出首个基于掩码的认证恢复防御方法MRCert，通过对良性样本和对抗性补丁样本推断类型特定的必要属性，在保持高预测准确率的同时验证对抗性补丁样本标签的良性，实现部署后的补丁鲁棒性认证。

    

    在部署后阶段，深度学习模型的输入可能被对抗性补丁攻击，也可能未被攻击。在补丁边界内对此类输入进行补丁鲁棒性认证可以验证其标签的良性，并且应保持较高的预测准确率。然而，现有的基于平滑和基于掩码的恢复防御方法无法同时实现这两点：前者大幅降低预测准确率，后者无法验证对抗性补丁输入所返回标签的良性。我们提出MRCert，这是首个基于掩码的认证恢复防御方法，证明了同时实现两者的可行性。与现有所有工作对两类输入（良性样本和对抗性补丁样本）应用统一认证条件不同，MRCert在部署后阶段为这两类输入推断深度学习模型的类型特定必要属性，并通过一种新颖的类型导向方法将它们形式化关联，以验证标签的良性。

    arXiv:2610.10617v1 Announce Type: cross  Abstract: In post-deployment time, inputs to deep learning models may or may not be adversarially patched. Patch robustness certification on such inputs within a patch bound can verify their label benignity and should retain high prediction accuracy. However, existing smoothing-based and masking-based recovery defenders cannot achieve both simultaneously: they degrade the prediction accuracy much and cannot verify the benignity of the returned label of an adversarially patched input, respectively. We propose MRCert, the first masking-based certified recovery defender that shows the feasibility of achieving both. Unlike all existing works to apply a common condition across both types of input (benign and adversarially patched samples) for certification, MRCert infers type-specific necessary properties of deep learning models for both types in post-deployment time and formally relates them to verify the label benignity through a novel type-oriente
    
[^46]: PyCache Trap：Agent技能扫描器中的检查-执行鸿沟

    PyCache Trap: The Inspection-Execution Gap in Agent Skill Scanners

    [https://arxiv.org/abs/2610.10612](https://arxiv.org/abs/2610.10612)

    该论文揭示了针对Agent技能扫描器的PyCache Trap攻击，利用Python字节码缓存与源代码之间的检查-执行鸿沟实现94-100%的攻击成功率，并提出执行感知验证（EAV）方法在类型化执行图中关联指令、脚本与运行时工件以检测此类隐藏威胁。

    

    Agent技能将指令与可执行资源相结合，使第三方包能够访问agent的运行时环境。现有的技能扫描器检查文档和可见的源代码，但Python可能会执行行为不同的捆绑字节码缓存。我们通过PyCache Trap研究这种检查与执行之间的差距，PyCache Trap将良性源代码与加载器接受的替换缓存配对，并将其连接到与任务相关的调用。扫描器引导的重写会改变调用的措辞同时保留缓存主体，从而将包准入与对隐藏行为的识别分离开来。在100个技能和七个扫描器上，PyCache Trap实现了94-100%的攻击成功率，且未对驻留在缓存中的行为产生语义识别。我们提出执行感知验证（EAV），在类型化执行图中连接被检查的指令、脚本、导入和运行时工件。EAV结合了基于事实的行为分析……

    arXiv:2610.10612v1 Announce Type: cross  Abstract: Agent skills combine instructions with executable resources, giving third-party packages access to an agent's runtime. Existing skill scanners inspect documentation and visible source, but Python may execute a bundled bytecode cache with different behavior. We study this gap between inspection and execution through PyCache Trap, which pairs benign source with a substituted cache accepted by the loader and connects it to a task-relevant invocation. Scanner-guided rewriting changes the invocation wording while preserving the cache body, separating package admission from recognition of the concealed behavior. Across 100 skills and seven scanners, PyCache Trap achieves 94-100% attack success, with no semantic recognition of the cache-resident behavior. We propose execution-aware validation (EAV) to connect inspected instructions, scripts, imports, and runtime artifacts in a typed execution graph. EAV combines grounded behavioral analysis w
    
[^47]: 代码理解是编程智能体的瓶颈

    Code Understanding is a Bottleneck for Coding Agents

    [https://arxiv.org/abs/2610.10610](https://arxiv.org/abs/2610.10610)

    该论文提出CABRA基准，通过调用图变换从零构建难度可精确控制、可扩展的代码理解任务，发现编程智能体是依靠工具调用来弥补其代码理解能力的不足，且工具调用次数比编辑代码行数更能预测智能体的表现。

    

    面向编程智能体的仓库级基准（如SWE-bench）通常假设编辑代码的行数可以预测任务难度，但此类数据集对代码类型和任务类型控制不足，使人难以确定究竟是哪些能力真正导致了智能体的错误。我们提出了CABRA：一个用于严格评估智能体的编程能力蓝图。CABRA从零开始将任务构建为调用图变换，并通过一个任务规模参数在四个维度上调节难度：函数遍历、搜索、运行时解析和指令遵循。我们在6,840个CABRA任务上运行了八个大语言模型和六个编程智能体，结果表明：1）随着任务规模增长，LLM的准确率下降，但智能体通过将工作卸载给工具（如grep）保持接近完美的表现；2）更大的CABRA任务会引发更多用于阅读和分析的工具调用，而在SWE-bench Verified上进行的一项独立研究表明，这些工具调用次数比编辑代码行数更能预测智能体的准确率，说明对智能体而言任务难度可（原文摘要在此处截断）。

    arXiv:2610.10610v1 Announce Type: cross  Abstract: Repository benchmarks (e.g., SWE-bench) for coding agents often assume that lines of code edited can predict task difficulty, but such datasets' poor control over code and task types makes it hard to know which abilities truly drive agent errors. We present CABRA: a Coding Ability Blueprint for Rigorous Agent evaluation. CABRA builds tasks from scratch as call graph transformations and scales difficulty via a task size parameter on four axes: function traversal, search, runtime resolution, and instruction following. We run eight LLMs and six coding agents on 6,840 CABRA tasks to show: 1) LLM accuracy falls as task size~grows, but agents stay near-perfect by offloading work to tools (e.g., grep); 2) Larger CABRA tasks elicit more tool calls for reading and analysis, while a separate study on SWE-bench Verified shows these tool call counts predict agents' accuracy better than lines of code edited, suggesting task difficulty for agents ca
    
[^48]: 超越类型检查：迈向形式化规范生成的整体评估

    Beyond Type-checking: Towards Holistic Evaluation of Formal Specification Generation

    [https://arxiv.org/abs/2610.10604](https://arxiv.org/abs/2610.10604)

    该论文提出了一个从350个Lean任务构建的统一数据集和涵盖形式有效性、参考相似性与等价性、行为充分性的整体评估框架，以解决规范生成中仅靠类型检查无法验证生成规范是否忠实反映用户意图的问题。

    

    在生成可验证代码时，自然语言需求通过大语言模型（LLM）和智能体工作流被映射为机器可检查的代码。该流程的一个关键组成部分是规范生成，它产生一个形式化契约，智能体可以据此证明实现的正确性。证明生成可以从定理证明器那里获得确定性的反馈，但规范生成缺乏一个明确的检查来判断生成的规范是否捕获了用户的意图。因此，一个通过验证的证明可能只是针对一个歪曲了预期行为的规范确立了正确性。我们朝着对规范生成进行整体评估迈出了一步，构建了一个统一的数据集，汇集了350个现有的Lean任务，其中189个来自VERINA，161个来自CLEVER，并提出了一个涵盖形式有效性、参考相似性与等价性以及行为充分性的评估框架。我们区分了对必需输入的接受与对有效输出的接受……

    arXiv:2610.10604v1 Announce Type: cross  Abstract: When generating verifiable code, natural language requirements are mapped to machine checked code using LLMs and agentic workflows. A crucial component of this pipeline is specification generation (SpecGen), which produces a formal contract against which an agent can prove implementation correctness. Proof generation can obtain deterministic feedback from a theorem prover, but SpecGen lacks a definitive check that a generated specification captures the user's intent. A checked proof can therefore establish correctness against a specification that misrepresents the intended behaviour. We take a step towards holistic SpecGen evaluation with a unified dataset assembled from $350$ existing Lean tasks, including $189$ from VERINA and $161$ from CLEVER, and a framework covering formal validity, reference similarity and equivalence, and behavioural adequacy. We distinguish acceptance of required inputs from acceptance of valid outputs and rej
    
[^49]: 大语言模型集成硬件设计验证综述

    A Survey on LLM-Integrated Hardware Design Verification

    [https://arxiv.org/abs/2610.10580](https://arxiv.org/abs/2610.10580)

    本综述系统回顾了大语言模型在硬件功能验证中的应用，涵盖断言生成、测试平台生成、缺陷定位与形式化验证等多个方向，并指出LLM最有效的角色是作为嵌入验证流程中的语义推理、搜索和编排组件。

    

    大语言模型（LLM）正日益被集成到硬件验证中，以实现规范解释、验证工件生成、调试、形式化推理和工具编排的自动化。本综述对LLM辅助的硬件功能验证进行了系统性回顾，涵盖SystemVerilog断言生成、激励与测试平台生成、缺陷定位与设计修复、模型检查与等价性检查、SAT/SMT优化以及新兴的智能体验证工作流。我们按照方法论、验证目标、工具交互、基准测试和评估标准对相关文献进行组织，并考察了推理时技术——包括提示工程、检索、结构化推理和智能体工作流——以及训练时的适配方法。在这些领域中，一个共同的模式逐渐显现：LLM作为嵌入其中的语义推理、搜索和编排组件时最为有效。

    arXiv:2610.10580v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly being integrated into hardware verification to automate specification interpretation, verification-artifact generation, debugging, formal reasoning, and tool orchestration. This survey provides a systematic review of LLM-assisted hardware functional verification across SystemVerilog assertion generation, stimulus and testbench generation, bug localization and design repair, model checking and equivalence checking, SAT/SMT optimization, and emerging agentic verification workflows. We organize the literature by methodology, verification objective, tool interaction, benchmark, and evaluation criterion, and examine both inference-time techniques--including prompting, retrieval, structured reasoning, and agentic workflows--and training-time adaptation. Across these areas, a common pattern emerges: LLMs are most effective as semantic reasoning, search, and orchestration components embedded within
    
[^50]: ParanoiaEval：智能体编程中不必要防御性工作的基准测试

    ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding

    [https://arxiv.org/abs/2610.08662](https://arxiv.org/abs/2610.08662)

    提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。

    

    随着编程智能体日益自主地承担真实世界的工作，判断其风险应对措施是否合理已变得尤为重要。现有工作从各自独立的角度评估相关的智能体行为，但缺乏一个统一这些行为的系统性框架。为弥合这一差距，我们提出了ParanoiaEval——首个用于统一评估编程智能体风险应对能力的基准。该基准以软件工程风险管理中成熟的“规避-转移-缓解-接受”（Avoidance-Transfer-Mitigation-Acceptance）框架为基础，将这4种基本风险应对措施操作化到编程智能体场景中，并包含200对证据受控的仓库级任务对，每对任务仅在定义应对措施的证据上存在差异。我们进一步引入了针对风险应对违规和证据响应性的专用指标，并采用经过人类校准的智能体裁判以实现可靠评估。在8个代表性模型上进行的大规模实验……

    arXiv:2610.08662v1 Announce Type: new  Abstract: As coding agents increasingly undertake real-world work autonomously, judging whether their risk treatments are warranted has become important. Existing work evaluates related agent behaviors from separate perspectives, but lacks a systematic framework for unifying these behaviors. To bridge this gap, we introduce ParanoiaEval, the first benchmark for unified evaluation of risk-treatment capabilities in coding agents. Grounded in the well-established Avoidance-Transfer-Mitigation-Acceptance framework in software engineering risk management, ParanoiaEval operationalizes its 4 fundamental treatments for coding-agent settings and contains 200 evidence-controlled repository-level task pairs, each differing only in treatment-defining evidence. We further introduce dedicated metrics for risk-treatment violations and evidence responsiveness, using a human-calibrated agentic judge for reliable evaluation. Large-scale experiments on 8 representat
    
[^51]: 理解模型上下文协议（MCP）生态系统的层次结构与功能图景

    Understanding the Hierarchical Structure and Functional Landscape of the Model Context Protocol Ecosystem

    [https://arxiv.org/abs/2610.05319](https://arxiv.org/abs/2610.05319)

    该论文构建了迄今最大的MCP生态系统工具级地图MCPacific，通过LLM驱动的迭代流程建立了涵盖58,915项能力的层次化功能分类体系，解决了数十万MCP服务器因缺乏细粒度功能组织而导致智能体难以发现、比较和替代工具的问题。

    

    AI智能体日益依赖通过模型上下文协议暴露的工具来完成用户任务。目前各市场列出了数十万个MCP服务器，但它们仅按粗粒度的、特定于市场的服务器类别进行组织，这使得智能体和用户难以识别适用于特定操作的工具、发现功能替代品并评估这些替代品之间的差异。我们提出了MCPacific，这是MCP生态系统中规模最大的工具级、跨市场地图。MCPacific收集了来自17个市场的368,754个MCP服务器条目，对应124,267个唯一服务器，从这些服务器中以静态方式提取了七种语言的1,328,233个工具规范，并将它们组织成一个包含58,915项能力的层次化功能分类体系。我们通过迭代的LLM驱动的“设计-测试-优化”流程构建该分类体系，并使用校准的嵌入路由将完整语料库映射到该体系上。我们的研究（原文摘要在此处截断）

    arXiv:2610.05319v2 Announce Type: replace  Abstract: AI agents increasingly rely on tools exposed through the Model Context Protocol (MCP) to complete user tasks. Hundreds of thousands of MCP servers are listed across marketplaces, yet they are organized only by coarse, marketplace-specific server categories. This makes it difficult for agents and users to identify tools for a given operation, find functional alternatives, and assess how those alternatives differ. We present MCPacific, the largest tool-level, cross-marketplace map of the MCP ecosystem. MCPacific collects 368,754 MCP server listings corresponding to 124,267 unique servers across 17 marketplaces, statically extracts 1,328,233 tool specifications from these servers in seven languages, and organizes them into a hierarchical functional taxonomy of 58,915 capabilities. We construct the taxonomy through an iterative LLM-driven design-test-refine process and map the full corpus to it using calibrated embedding routing. Our stu
    
[^52]: 不确定性下的受治理人机协同优先级排序：自适应估计与依赖约束下的组合选择

    Governed Human-AI Prioritization Under Uncertainty: Adaptive Estimation and Dependency-Constrained Portfolio Selection

    [https://arxiv.org/abs/2609.10648](https://arxiv.org/abs/2609.10648)

    本文提出了一种在不确定性下可治理、可检查且可重新校准的人机协同优先级排序框架，通过五个定量算子（BVS、ERS、PVS、CCS、ODP）实现自适应估计与依赖约束下的组合选择，并用受控合成实验验证了该框架对参数扰动的鲁棒性。

    

    AI原生软件工程日益将人类判断、历史类比、参数化估计和AI生成的预测融合到同一个优先级排序决策中。由此产生的核心问题不仅在于如何对候选工作进行排序，更在于如何以一种可检查、可重新校准的方式治理异构估计、不确定性、战略参数、依赖关系和有限容量。本文研究了D-POAF决策实践中使用的五个定量算子：业务价值评分（BVS）、工作量与风险评分（ERS）、优先级价值评分（PVS）、集体校准评分（CCS）和最优开发路径（ODP）。通过受控合成实验，在已知潜在变量和显式误差过程的条件下刻画了这些算子的行为。实验结果表明，适度的BVS权重扰动能够保持全局排序的稳定性（Spearman相关系数中位数为0.986），而更大范围的战略性变化则使前10%优先项的重叠度降低至0.788。可靠性加权的工作量聚合……

    arXiv:2609.10648v1 Announce Type: new  Abstract: AI-native software engineering increasingly combines human judgment, historical analogy, parametric estimation, and AI-generated forecasts inside the same prioritization decision. The resulting problem is not merely how to rank candidate work, but how to govern heterogeneous estimates, uncertainty, strategic parameters, dependencies, and limited capacity in a way that remains inspectable and recalibratable. We study five quantitative operators used in the D-POAF decision practice: Business Value Score (BVS), Effort and Risk Score (ERS), Prioritization Value Score (PVS), Collective Calibration Score (CCS), and Optimal Development Path (ODP). Controlled synthetic experiments characterize their behavior under known latent variables and explicit error processes. Moderate BVS-weight perturbations preserved global rankings (median Spearman 0.986), while broader strategic changes reduced top-10% overlap to 0.788. Reliability-weighted effort agg
    
[^53]: 超越组件测试：验证智能体AI系统

    Beyond Component Testing: Validating Agentic AI Systems

    [https://arxiv.org/abs/2607.29405](https://arxiv.org/abs/2607.29405)

    该论文通过对262篇文献的系统性映射研究，提出涵盖行为、安全、时间、监管和多智能体五个维度的分类框架，刻画了智能体AI系统的验证问题，并揭示了现有研究方法在各维度上分布的密集与空白之处。

    

    智能体AI系统通过结合规划、工具使用、记忆、交互与适应的多步骤轨迹来执行任务。这种行为使得验证实践超越了组件测试和一次性输入输出评估，因为可接受的系统行为现在取决于决策如何随时间推移以及在不断变化的环境条件下逐步展开。这项系统性映射研究综合了262篇论文，涵盖智能体评估、软件保证、信息物理系统、运行时监控和监管指导等领域，以刻画智能体系统的验证问题。该综述围绕一个五维分类体系组织，涵盖行为、安全、时间、监管和多智能体等方面的关注点，并利用该分类体系映射现有方法，识别方法与维度之间配对的密集与稀疏之处。由此得到的映射图显示，现有文献集中于行为评估（262篇论文中的105篇

    arXiv:2607.29405v2 Announce Type: replace  Abstract: Agentic AI systems act through multi-step trajectories that combine planning, tool use, memory, interaction, and adaptation. This behavior stretches validation practice beyond component testing and one-shot input-output evaluation, because acceptable system behavior now depends on how decisions unfold over time and under changing environmental conditions. This systematic mapping study synthesizes 262 papers spanning agent evaluation, software assurance, cyber-physical systems, runtime monitoring, and regulatory guidance in order to characterize the validation problem for agentic systems. The review is organized around a five-dimension taxonomy covering behavioral, safety, temporal, regulatory, and multi-agent concerns, and uses that taxonomy to map current approaches and identify dense and sparse pairings of approaches and dimensions. The resulting map shows that the literature concentrates on behavioral evaluation (105 of 262 papers
    
[^54]: 面向LLM生成代码片段的高效可扩展溯源追踪方法

    Efficient and Scalable Provenance Tracking for LLM-Generated Code Snippets

    [https://arxiv.org/abs/2605.28510](https://arxiv.org/abs/2605.28510)

    该论文提出SourceTracker编码器与HybridSourceTracker两阶段混合流水线，先用向量搜索缩小候选范围、再用Winnowing指纹精确重排，从而实现对LLM生成代码在数十亿级训练语料上的高效可扩展溯源追踪。

    

    大型语言模型（LLM）在代码补全与生成中的应用日益广泛，但它们可能会逐字复现训练样本而不进行作者归属标注，这引发了关于剽窃和许可证合规方面的法律与伦理问题。经典的基于指纹的剽窃检测方法（如Winnowing）仍然非常有效，但其检测过程需要将代码片段与整个训练集进行比较，且其线性时间搜索使得这些方法对于训练现代代码LLM所使用的数十亿级语料库而言并不实用。为了弥合这一差距，我们提出了SourceTracker——一个专为代码检索定制的3亿参数编码器，并构建了一个混合两阶段溯源追踪流水线HybridSourceTracker（HST）。HST首先通过向量搜索缩小候选代码片段的范围，然后使用Winnowing对精确指纹进行重排序。我们在……（训练和评估相关内容在原文中被截断）

    arXiv:2605.28510v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) for code completion and generation are increasingly used in software development, yet they may reproduce training examples verbatim and without authorship attribution, raising legal and ethical concerns around plagiarism and license compliance. Classical fingerprint-based plagiarism detectors, such as Winnowing, remain highly effective, yet the inspection requires comparing fragments of code to the entire training set, and their linear-time search makes them impractical for the billion-scale corpora used to train modern code LLMs. To bridge this gap, we introduce SourceTracker, a 300M-parameter encoder tailored for code retrieval, together with a hybrid two-stage provenance-tracking pipeline HybridSourceTracker (HST). HST first narrows down a small set of candidate snippets via vector search, then re-ranks those candidates using Winnowing on exact fingerprints. We train and evaluate our system on a 
    
[^55]: VISTA：面向视觉规格到网络应用编码智能体的端到端基准测试

    VISTA: An End-to-End Benchmark for Visual Spec-to-Web-App Coding Agents

    [https://arxiv.org/abs/2605.26144](https://arxiv.org/abs/2605.26144)

    VISTA是一个端到端基准测试，评估编程智能体将多页面设计交付物转化为可运行的全栈Web和Android应用的能力，通过在运行的应用中定位并探测8,487个人工标注的交互组件来产生可追溯至各个设计需求的评分。

    

    编程智能体现在能够根据设计原型构建应用程序，但看起来正确的界面并不等于功能正常的应用。现有基准测试要么在没有后端的情况下仅评估视觉重建效果，要么仅根据文本需求测试全栈功能，很少有将评分与设计所需的各个具体组件关联起来。我们提出了VISTA（视觉规格到应用），这是一个端到端基准测试，其中编程智能体将多页面设计交付物（Figma渲染图、结构和文本需求）转化为可运行的全栈Web和移动（Android）应用程序。VISTA在18个应用程序（126个网页和576个Android屏幕）中提供了8,487个人工标注的交互组件，并配备一个可执行评估器，该评估器在运行的应用程序中定位标注组件并探测其交互，从而产生可追溯到单个设计需求的定位、行为和联合评分。在14个部署的编程智能体系统中进行的评估显示……

    arXiv:2605.26144v4 Announce Type: replace-cross  Abstract: Coding agents can now build applications from design mockups, but a screen that looks right is not an application that works. Existing benchmarks either score visual reconstruction without a backend or test full-stack functionality from textual requirements, and few tie scores to the individual components a design requires. We introduce VISTA (VIsual Spec-To-App), an end-to-end benchmark in which coding agents turn multi-page design handoffs (Figma renders, structure, and textual requirements) into runnable full-stack Web and Mobile (Android) applications. VISTA provides 8,487 human-annotated interactive components across 18 applications (126 Web pages and 576 Android screens) and an executable evaluator that locates annotated components in the running application and probes their interactions, yielding localization, behavior, and joint scores traceable to individual design requirements. Across 14 deployed coding-agent systems,
    
[^56]: PropGen：面向移动应用基于属性测试的属性自动生成

    PropGen: Automated Property Generation for Property-Based Testing of Mobile Apps

    [https://arxiv.org/abs/2604.13463](https://arxiv.org/abs/2604.13463)

    本文提出PropGen，利用大语言模型自动为移动应用生成属性，突破了人工编写属性的瓶颈，使基于属性的测试能够高效检测不崩溃但行为异常的功能性缺陷。

    

    移动应用常常受到功能性缺陷的困扰，这类缺陷不会导致程序崩溃，而是在特定用户交互下表现为不正确的行为。由于这类缺陷通常缺乏显式的测试判定准则，传统的自动化测试技术难以检测到它们。基于属性的测试可以通过将预期行为形式化为属性，并在多样化的交互下对其进行检验，从而有效地暴露此类缺陷。然而，其实际应用受到人工编写属性的依赖所限制，而人工构建属性既困难又成本高昂。为了解决这一局限，本文探索利用大语言模型来自动化地为移动应用的基于属性测试构建属性。这在两个方面充满挑战：其一，难以系统地挖掘并执行多样化的应用功能；其二，难以从功能执行过程中推导出有效的属性。

    arXiv:2604.13463v2 Announce Type: replace  Abstract: Mobile apps often suffer from functional bugs that do not cause crashes but instead manifest as incorrect behaviors under specific user interactions. Such bugs are difficult to detect by conventional automatic testing techniques because they often lack explicit \textit{test oracles}. Property-based testing can effectively expose them by specifying intended behavior as properties and checking them under diverse interactions. However, its practical use is limited by the reliance on manually written properties, which are difficult and expensive to construct.   To address this limitation, this paper explores the use of large language models (LLMs) to automate property construction for property-based testing of mobile apps. This is challenging in two ways. \textit{First}, it is difficult to systematically uncover and execute diverse app functionalities. \textit{Second}, it is difficult to derive valid properties from functionality executi
    
[^57]: ReCodeAgent：一种用于大规模代码库语言无关翻译与验证的多智能体工作流

    ReCodeAgent: A Multi-agent Workflow for Language-Agnostic Translation and Validation of Large-Scale Repositories

    [https://arxiv.org/abs/2604.07341](https://arxiv.org/abs/2604.07341)

    ReCodeAgent通过自主多智能体工作流，实现了仓库级代码翻译和验证的语言无关性，用户仅需指定源和目标编程语言即可自动处理整个仓库。

    

    大多数仓库级代码翻译和验证技术仅在单一源-目标编程语言（PL）对上进行了评估，这是由于适应新PL对所需的复杂工程工作。编程智能体能够实现仓库级代码翻译和验证的语言无关性：它们可以跨多种PL合成代码，并自主使用针对每种PL分析的现有工具。然而，现有技术尚未提供一种完全自主的智能体方法，用于大规模程序的仓库级代码翻译和验证。本文提出了ReCodeAgent，一种自主多智能体方法，用于语言无关的仓库级代码翻译和验证。用户只需提供源PL中的项目并指定目标PL，ReCodeAgent即可自动翻译和验证整个仓库。ReCodeAgent是首个实现高翻译质量的技术。

    arXiv:2604.07341v3 Announce Type: replace-cross  Abstract: Most repository-level code translation and validation techniques have been evaluated on a single source-target programming language (PL) pair, owing to the complex engineering effort required to adapt new PL pairs. Programming agents can enable PL-agnosticism in repository-level code translation and validation: they can synthesize code across many PLs and autonomously use existing tools specific to each PL's analysis. However, state-of-the-art has yet to offer a fully autonomous agentic approach for repository-level code translation and validation of large-scale programs. This paper proposes ReCodeAgent, an autonomous multi-agent approach for language-agnostic repository-level code translation and validation. Users only need to provide the project in the source PL and specify the target PL for ReCodeAgent to automatically translate and validate the entire repository. ReCodeAgent is the first technique to achieve high translatio
    
[^58]: 祈使干扰：社会语域如何塑造大语言模型中的指令拓扑结构

    Imperative Interference: Social Register Shapes Instruction Topology in Large Language Models

    [https://arxiv.org/abs/2603.25015](https://arxiv.org/abs/2603.25015)

    该研究发现大语言模型会按照社会语域惯例来处理系统提示指令——同一指令在英语中协作而在西班牙语中竞争，将祈使句改写为陈述句可降低81%的跨语言差异，说明模型把指令理解为社会行为而非技术规范。

    

    在系统提示词中，相同语义内容的指令在英语里相互协作，在西班牙语里却相互竞争，呈现出截然相反的交互拓扑结构。我们通过跨四种语言、四个模型的指令级消融实验表明，这种拓扑反转由社会语域所介导：祈使语气在不同言语社区中承载着不同的强制力，而基于多语言数据训练的模型已经习得了这些惯例。将单个指令块改写为陈述句可使跨语言方差降低81%（p = 0.029，置换检验）。在十一个祈使句指令块中仅改写三个，即可将西班牙语的指令拓扑从竞争性转变为协作性，并对未改写的指令块产生溢出效应。这些发现表明，模型将指令处理为社会行为而非技术规范：“永远不要做X”是一种权威的行使，其效力依赖于语言；而“X：已禁用”则只是一个事实。

    arXiv:2603.25015v2 Announce Type: replace-cross  Abstract: System prompt instructions that cooperate in English compete in Spanish, with the same semantic content, but opposite interaction topology. We present instruction-level ablation experiments across four languages and four models showing that this topology inversion is mediated by social register: the imperative mood carries different obligatory force across speech communities, and models trained on multilingual data have learned these conventions. Declarative rewriting of a single instruction block reduces cross-linguistic variance by 81% (p = 0.029, permutation test). Rewriting three of eleven imperative blocks shifts Spanish instruction topology from competitive to cooperative, with spillover effects on unrewritten blocks. These findings suggest that models process instructions as social acts, not technical specifications: "NEVER do X" is an exercise of authority whose force is language-dependent, while "X: disabled" is a fact
    
[^59]: Arbiter：检测LLM智能体系统提示词中的干扰

    Arbiter: Detecting Interference in LLM Agent System Prompts

    [https://arxiv.org/abs/2603.08993](https://arxiv.org/abs/2603.08993)

    提出了Arbiter框架，通过结合形式化评估规则与多模型LLM排查来检测主流编码智能体系统提示词中的干扰模式，并发现提示词架构与故障类别强相关、多模型评估能揭示单模型分析无法发现的漏洞类型。

    

    基于LLM的编码智能体的系统提示词是支配智能体行为的软件工件，却缺乏应用于传统软件的测试基础设施。我们提出了Arbiter，一个将形式化评估规则与多模型LLM排查相结合的框架，用于检测系统提示词中的干扰模式。将该框架应用于三个主流编码智能体系统提示词：Claude Code（Anthropic）、Codex CLI（OpenAI）和Gemini CLI（Google），我们在无方向排查阶段识别出152项发现，并在对一家厂商的定向分析中识别出21个手动标注的干扰模式。我们表明，提示词架构（单体式、扁平式、模块化）与观察到的故障类别强相关，但与严重程度无关，且多模型评估能发现与单模型分析在类型上截然不同的漏洞类别。其中一项排查发现——Gemini CLI记忆系统中的结构性数据丢失——与已提交并修复的某个问题相一致。

    arXiv:2603.08993v2 Announce Type: replace-cross  Abstract: System prompts for LLM-based coding agents are software artifacts that govern agent behavior, yet lack the testing infrastructure applied to conventional software. We present Arbiter, a framework combining formal evaluation rules with multi-model LLM scouring to detect interference patterns in system prompts. Applied to three major coding agent system prompts: Claude Code (Anthropic), Codex CLI (OpenAI), and Gemini CLI (Google), we identify 152 findings across the undirected scouring phase and 21 hand-labeled interference patterns in directed analysis of one vendor. We show that prompt architecture (monolithic, flat, modular) strongly correlates with observed failure class but not with severity, and that multi-model evaluation discovers categorically different vulnerability classes than single-model analysis. One scourer finding was structural data loss in Gemini CLI's memory system was consistent with an issue filed and patche
    
[^60]: PackMonitor：通过解码时监控实现零包幻觉

    PackMonitor: Enabling Zero Package Hallucinations Through Decoding-Time Monitoring

    [https://arxiv.org/abs/2602.20717](https://arxiv.org/abs/2602.20717)

    本文提出PackMonitor，首个通过在解码时持续监控并根据有限可枚举的权威包列表进行干预、从根本上彻底消除（而非仅仅降低）LLM包幻觉的方法。

    

    arXiv:2602.20717v2 公告类型：替换 摘要：随着大型语言模型（LLMs）日益融入软件开发工作流程，其可信度已成为一个关键问题。然而，在依赖推荐场景中，LLMs的可靠性受到了普遍存在的包幻觉的破坏，即模型经常会推荐并不存在的幻觉包。近期研究已提出了一系列缓解该问题的方法。然而，现有方法通常只是降低幻觉率而非将其彻底消除，因而留下了持续的软件安全风险。在本工作中，我们论证了包幻觉在理论上是可预防的，其核心洞察在于：包的有效性可以通过有限且可枚举的权威包列表来判定。基于此，我们提出了PackMonitor，这是首个能够从根本上消除包幻觉的方法，它通过持续监控模型的解码过程并进行干预……

    arXiv:2602.20717v2 Announce Type: replace  Abstract: As Large Language Models (LLMs) are increasingly integrated into software development workflows, their trustworthiness has become a critical concern. However, in dependency recommendation scenarios, the reliability of LLMs is undermined by widespread package hallucinations, where models often recommend hallucinated packages. Recent studies have proposed a range of approaches to mitigate this issue. Nevertheless, existing approaches typically merely reduce hallucination rates rather than eliminate them, leaving persistent software security risks.   In this work, we argue that package hallucinations are theoretically preventable based on the key insight that package validity is decidable through finite and enumerable authoritative package lists. Building on this, we propose PackMonitor, the first approach capable of fundamentally eliminating package hallucinations by continuously monitoring the model's decoding process and intervening 
    
[^61]: CAF'E：机器遗忘的因果黑盒测试

    CAF\'E: Causal Black-Box Testing of Machine Unlearning

    [https://arxiv.org/abs/2509.16525](https://arxiv.org/abs/2509.16525)

    提出CAF'E框架，将机器遗忘测试构建为基于规范的测试，仅通过黑盒模型的预测输出对特征进行因果干预并传播到下游特征，从而有效检测特征影响在模型中的残留。

    

    机器学习模型越来越多地被部署为需要随需求变化而演进的软件组件。当特定的训练记录或特征不能再影响已部署的模型时，机器遗忘旨在不从头重新训练的情况下消除这种影响。由于遗忘通常是近似的，因此必须对其有效性进行测试。此类测试通常必须将模型视为黑盒，即无法访问其参数、训练历史或遗忘过程。特征带来了进一步的挑战：即使一个特征已从模型的输入中移除，其影响仍可能通过下游特征持续存在。许多现有的检查仅检验该特征的直接使用，因此可能会将一个实际上仍依赖该特征的模型误判为合格。我们将遗忘测试构建为基于规范的测试，并提出了CAF'E，它仅利用已部署模型的预测，对特征进行干预，并将变化传播到其下游特征。

    arXiv:2509.16525v2 Announce Type: replace-cross  Abstract: Machine learning models are increasingly deployed as software components that must evolve as requirements change. When specific training records or features must no longer influence a deployed model, machine unlearning aims to remove that influence without retraining from scratch. Because unlearning is often approximate, its effectiveness must be tested. Such tests must often treat the model as a black box, without access to its parameters, training history, or unlearning procedure. Features pose a further challenge: even after a feature is removed from a model's inputs, its influence can persist through downstream features. Many existing checks examine only the feature's direct use and can therefore certify a model that still depends on it. We frame unlearning testing as specification-based testing and present CAF\'E, which, using only a deployed model's predictions, intervenes on the feature, propagates the change to its down
    
[^62]: 面向UI测试复用的技能自适应模仿学习

    Skill-Adaptive Imitation Learning for UI Test Reuse

    [https://arxiv.org/abs/2409.13311](https://arxiv.org/abs/2409.13311)

    该研究发现即使大语言模型具备高度准确的UI事件映射能力，仍不足以解决源应用与目标应用之间的实现差异，因此提出技能自适应模仿学习方法以提升UI测试复用的有效性。

    

    arXiv:2409.13311v2 公告类型：替换。为了减轻手动编写用户界面（UI）测试用例的巨大成本，UI测试迁移旨在通过改写来自具有相似功能的源移动应用程序的测试用例，自动为目标移动应用程序生成测试用例。传统上，这一过程被视为一个顺序的UI事件映射问题，即根据文本描述将源应用中的事件映射到目标应用中的对应事件。以往的研究广泛专注于提高NLP模型的事件映射准确性。尽管具备出色NLP能力的大语言模型（LLM）的出现暗示了实现近乎完美事件映射的可能性，但我们的研究表明，即使LLM能够实现高度准确的事件映射，也不足以解决源应用与目标应用之间的实现差异问题，从而降低了LLM驱动的UI测试迁移解决方案的整体有效性。

    arXiv:2409.13311v2 Announce Type: replace  Abstract: To alleviate the substantial cost of manually crafting user interface (UI) test cases, UI test migration aims to automatically generate test cases for a target mobile application (app) by adapting those from a source app that shares similar functionalities. Traditionally, this process has been approached as a sequential UI-event-mapping problem, where events in the source app are mapped to those in the target one based on their textual descriptions. Prior research has extensively focused on enhancing the event-mapping accuracy of NLP models. Although the advent of large language models (LLMs) with impressive NLP capabilities suggests the potential for near-perfect event-mapping, our study demonstrates that even the highly accurate event-mapping of LLMs is insufficient to address the implementation discrepancies between the source and the target apps, reducing the overall effectiveness of LLM-driven solutions for UI test migration.   
    

