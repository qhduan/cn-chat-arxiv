# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Validating Autoformalized Specifications via Contrapuntal Languages](https://arxiv.org/abs/2610.30059) | 提出“对位语言”方法——使用一对彼此不存在保持结构翻译的任务特定形式语言来引入表示多样性，从而消除模型错误之间的相关性，使得多个LLM输出之间的一致性能够有效验证自动形式化规范是否忠实于用户意图，而无需人工审查等昂贵神谕。 |
| [^2] | [SWE-DIVE: Multi-Turn Coding Agent Evaluation with Dynamic User Simulation](https://arxiv.org/abs/2610.30047) | 该论文提出 SWE-DIVE 框架，通过引入可模拟多种行为模式的动态用户模拟器和对话验证器，在多轮交互环境中评估编程智能体，弥补了传统单轮、格式规范提示词基准测试与真实使用场景之间的差距。 |
| [^3] | [Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories](https://arxiv.org/abs/2610.30016) | 该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。 |
| [^4] | [HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements](https://arxiv.org/abs/2610.29996) | HiddenLens是首个专为检测、分类、质疑并解决软件需求中隐藏假设而设计的六层BERT-LLM框架，能够超越单纯的歧义标记，在早期发现并化解需求中的隐含假设，从而显著降低后期修复成本。 |
| [^5] | [FaultGene: LLM-Assisted Fault-Augmented Test Generation for Hardware-in-the-Loop Validation of Autonomous Driving Systems](https://arxiv.org/abs/2610.29969) | FaultGene利用大语言模型自动为自动驾驶系统的硬件在环验证生成可执行的故障增强测试场景，解决了传统手工准备故障场景导致的测试覆盖不足、可扩展性差和效率低下的问题。 |
| [^6] | [DynaCI: Generating GitHub Actions Workflows via Dynamic Analysis](https://arxiv.org/abs/2610.29890) | 提出DynaCI，通过动态分析利用开发者本地环境的运行时信息自动生成GitHub Actions CI工作流，解决了现有方法需要开发者手动编写工作流规范、且难以表达需求的局限。 |
| [^7] | [PAUSE: Search-Free Steering of Coding Agents Before Their First Source-Code Edit](https://arxiv.org/abs/2610.29870) | PAUSE通过在编码智能体第一次修改源代码前拦截其编辑请求、由独立规划器一次性生成修复计划，实现无需监控器、无需检查点搜索、无额外模型调用的单次干预式智能体引导。 |
| [^8] | [Controlling Variability in~Neuroimaging Analysis Pipelines with~Software Product Lines](https://arxiv.org/abs/2610.29821) | 本文倡导将软件产品线（SPL）原则引入神经影像分析流水线的设计与管理，以控制不同研究团队流水线之间的变异性，从而提升MRI研究的可复现性和一致性。 |
| [^9] | [ALIAS: Evaluating Performance and Memorization in LLM-Written TLA+ Specifications](https://arxiv.org/abs/2610.29767) | 该论文利用TLA+模型检查器作为验证神谕，发现LLM在标识符重命名任务上的正确率下降主要来自要求精确标识符名称的评分方式，若按行为语义评分，记忆化效应实际很小甚至消失。 |
| [^10] | [AssureCode: Structured Justification for Trustworthy Agentic Coding](https://arxiv.org/abs/2610.29758) | 该论文借鉴安全关键系统的保证工程技术，提出AssureCode方法论，通过结构化论证将自然语言提示、形式化规范、开发者意图假设与程序分析证据关联起来，从而提升AI智能体生成代码的可信度与可验证性。 |
| [^11] | [VeruSteer: Learning What to Repair Next in Verus](https://arxiv.org/abs/2610.29755) | VeruSteer 通过离线构建修复树并比较不同修复轨迹的验证结果，训练出一个结合逐步模仿与轨迹偏好排序的神经选择器，从而动态决定 Verus 证明修复中下一步应修复的错误类型，取代了传统的预设静态优先级策略。 |
| [^12] | [Bridging Foundational and Agentic Capabilities of LLMs](https://arxiv.org/abs/2610.29751) | 该论文提出一个操作层面的测量框架，通过定义“能力实现差距”来量化大语言模型基础能力向智能体能力的转化程度，发现基础能力基准只能部分预测智能体操作表现，画像相似的模型在实际执行中差异显著。 |
| [^13] | [Metamodel-Guided LLM Decoding for Consistent DSL Generation](https://arxiv.org/abs/2610.29749) | 提出MeMoDecode解码框架，在LLM生成DSL的过程中联合强制执行具体语法、抽象语法（元模型）和良构性约束，通过增量映射到部分模型并验证其可补全性来确保生成结果的一致性。 |
| [^14] | [Engineering Requirements for Generative AI Agents: The BARD Process Theory](https://arxiv.org/abs/2610.29741) | 该研究通过对36位从业者访谈的扎根理论分析，提出了面向生成式AI智能体的需求工程过程理论BARD，指出智能体需求并非预先固定，而是通过“界定范围—构建智能体—观察行为—修订需求”这一循环周期被持续发现。 |
| [^15] | [SAKURA: Serving Attention Key-Value State Under Rate Adaptation for Private Image Editing on Phones](https://arxiv.org/abs/2610.29733) | SAKURA提出了一种云边协同架构，通过速率自适应机制在带宽预算内向手机传输可复用的注意力KV状态，从而在保护隐私的前提下实现高效的端侧图像编辑。 |
| [^16] | [Benchmarking Coding Agents for System-Level Performance Optimization](https://arxiv.org/abs/2610.29695) | 本文提出了MicroPerf-Bench，首个用于评估编码智能体在微服务系统中进行端到端性能优化能力的基准，通过向真实开源系统注入基于实证的性能反模式来构建测试场景，填补了现有基准仅停留在函数或仓库层面的空白。 |
| [^17] | [Reference Manual of MEDiml](https://arxiv.org/abs/2610.29666) | MEDiml是一个开源且符合标准的影像组学平台，通过结合Python包和图形桌面应用程序，以互补的图形与编程接口打通特征提取、预测建模和结果解释的端到端流程，降低了临床研究人员使用影像组学的门槛。 |
| [^18] | [Who Verifies the Agent's Success Claim? Self-Report Leakage in GUI-Agent Completion Judging](https://arxiv.org/abs/2610.29609) | 论文揭示了GUI智能体完成度判定中的“自我报告泄漏”现象——智能体自身的成功声明会误导裁判的判定——并证明通过隔离自我报告、仅依据执行轨迹进行审计，可大幅提升危险误报的检出率（最高提升43.4个百分点）。 |
| [^19] | [On the Effectiveness of Snapshot Testing: An Empirical Study](https://arxiv.org/abs/2610.29549) | 本研究对340个GitHub仓库的大规模实证分析及60位从业者的调查表明，快照测试与传统的非快照测试高度互补，两者结合可将语句覆盖率中位数提升至86.7%，显著增强测试有效性。 |
| [^20] | [RegoMender: A Multi-agent Framework for Automated Rego Policy Repair](https://arxiv.org/abs/2610.29456) | 本文提出了首个Rego策略修复基准RegoBench（含1,159个真实缺陷）和反馈引导的多智能体框架RegoMender，通过共享黑板与原子工具包协作动态执行打补丁、重写或回滚，实现了82.9%的语法修复成功率和77.9%的语义修复成功率。 |
| [^21] | [CertMoE: Robustness Certification for Mixture-of-Experts LLMs via Routing-Aware Smoothing](https://arxiv.org/abs/2610.29395) | 该论文提出了CertMoE，首个针对混合专家大语言模型的路由感知鲁棒性认证框架，通过显式建模输入依赖的动态路由，解决了现有认证方法因假设固定计算图而失效的问题。 |
| [^22] | [Do Coding Agents Understand Why Systems Crash---and Why They Don't?](https://arxiv.org/abs/2610.29332) | 本文提出“源代码干预”方法和包含934对代码树的SysRB数据集，通过配对仅相差一次提交的源代码树来严格测试编码智能体的崩溃推理能力，发现智能体严重倾向于预测“不崩溃”，从而漏掉了大量真实的系统崩溃。 |
| [^23] | [Motivation and Contribution in Open Source Software: A Meta-Analytic Investigation](https://arxiv.org/abs/2610.29199) | 该元分析整合41项研究发现，八种动机前因均与持续贡献意愿正相关、其中七种与实际贡献努力正相关，并推翻了“内在动机比外在动机更能驱动实际贡献”的常见假设。 |
| [^24] | [CloudIR-Bench: A Benchmark for Root Cause Analysis and Mitigation of Cloud Incidents](https://arxiv.org/abs/2610.29185) | 本文提出了首个公开可用的基准测试 CloudIR-Bench，为基于事件报告的云事件根因分析与缓解措施预测提供数据集和评估框架，解决了以往依赖私有数据导致的可复现性和泛化性受限的问题。 |
| [^25] | [Turning Quantum Compiler Configuration Selection Upside Down: A Top-Down Paradigm](https://arxiv.org/abs/2610.29170) | 该论文提出了一种自顶向下的量子编译器配置选择新范式，利用量子编译器配置空间可控、可穷举搜索的特点，颠覆了传统“先选设备再调编译器”的自底向上方法，克服了初始设备决策对电路执行保真度的根本性限制。 |
| [^26] | [AI-Based End-to-End Automotive Design and Software Development for SDVs](https://arxiv.org/abs/2610.29160) | 本文提出了一种将大语言模型与模型驱动工程相结合的AI驱动工具链，能够将自然语言需求自动转化为软件定义汽车的系统架构、可执行软件和验证工件，并在HiL/SiL环境中通过自适应巡航控制和紧急制动功能验证了其符合汽车安全关键开发要求。 |
| [^27] | [VERDICT: Multi-Agent Parallel Modular Verification](https://arxiv.org/abs/2610.29147) | VERDICT是一个多智能体框架，通过基于假设的并行规范综合打破传统自底向上验证的依赖顺序，实现大规模程序模块化演绎验证的并行化。 |
| [^28] | [ORSMT: Unleashing the Potential of Satisfiability Modulo Theories for Semantic Verification and Repair in LLM-Based Optimization](https://arxiv.org/abs/2610.29145) | ORSMT 利用可满足性模理论（SMT）对大语言模型生成的优化产物（数学模型、求解器代码等）进行精确的语义关系验证，并通过产物间的重构实现自动修复，显著提升了模型与代码的语义匹配率。 |
| [^29] | [VeeReview: Measuring Build-Verify Pairs in Agent-Authored Pull Requests from a V-Model Perspective](https://arxiv.org/abs/2610.29143) | 本文提出基于V模型构建-验证对的VeeReview方法，通过仓库信号和语言模型裁判测量拉取请求中的验证证据，发现智能体创作的拉取请求在全部四个验证维度上的“仅构建”率均低于人类拉取请求，与普遍预期相反。 |
| [^30] | [Symmetry-Oriented Testing Framework for Variational Quantum Algorithms](https://arxiv.org/abs/2610.29053) | 提出了一种面向对称性的变分量子算法测试框架，以与目标哈密顿量对易的算符作为测试规范，解决了现有方法无法验证量子程序执行是否偏离预期问题、所求解是否物理合理的难题。 |
| [^31] | [VeriEvolve: Evaluator-First Evolution for Tool-Using Agents](https://arxiv.org/abs/2610.29005) | VeriEvolve提出“评估器优先”的进化框架，先从任务指令自动合成可执行的评估器和挑战状态作为可靠的适应度信号，再据此进化工具使用智能体，将MCPMark文件系统任务的外部验证成功率从6/30提升至18/30。 |
| [^32] | [Behavior Beyond Text: Evaluating Refusal Transfer Failure in LLM Agents](https://arxiv.org/abs/2610.28982) | 该论文揭示了大语言模型的文本拒绝安全行为无法可靠迁移到工具调用场景——在聊天中被拒绝的有害请求中有10%至52%会在工具启用时引发不安全的工具调用，并通过残差流中的拒绝方向对这一失效机制进行了分析。 |
| [^33] | [Back Up Before You Regret: Measuring Backup Awareness and Automating Backup in Coding Agents](https://arxiv.org/abs/2610.28934) | 该论文提出 RegretBench 基准，揭示编码智能体虽平均完成 86.1% 的任务、却仅在 15.2% 的“后悔时刻”前主动备份，并据此推出 BackUpper 插件，在每次修改状态的工具调用之前自动保存受影响的状态，从而实现备份自动化。 |
| [^34] | [SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges](https://arxiv.org/abs/2610.28886) | SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。 |
| [^35] | [Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources](https://arxiv.org/abs/2610.28834) | 提出班扎夫资源归因框架，通过受控因子实验因果地量化计算、信息、反馈等推理时资源对修复失败编码任务的边际贡献，发现任务相关规范最有价值、额外计算的价值随模型能力递减，而多数前沿模型却错误地将预算分配给收益甚微的反馈资源。 |
| [^36] | [How Much Does the Harness Matter, and for Whom? A Multi-Model Study of Context in LLM-Driven Vulnerability Discovery](https://arxiv.org/abs/2610.28802) | 该研究通过24种模型-运行器配置的多模型实验发现，测试框架对漏洞发现的贡献高度依赖于所用模型——工程化上下文（如CodeQL调用图）能将中小型模型的关键性加权漏洞发现产出提升20-86%，但对前沿模型仅提升2%，且没有任何单一配置能覆盖全部漏洞。 |
| [^37] | [Sign Here: Attributing a Source for a More Trustworthy Software Supply Chain](https://arxiv.org/abs/2610.28794) | 本文提出通过直接从源代码构建软件包来取代软件包注册中心，并利用“恶意软件包通常不标注源代码来源、而良性软件包会链接其代码仓库”这一假设，作为防范软件供应链恶意软件攻击的策略，从而消除整个攻击面。 |
| [^38] | [Multi-agent LLMs on the Cloud-Edge-IoT Continuum: A Systematic Mapping Study on Architectures, Deployment, and Evaluation](https://arxiv.org/abs/2610.28702) | 本文通过一项系统性映射研究，首次对多智能体大语言模型系统在云-边-物联网连续体上的架构、部署模式、应用领域与评估实践进行了系统分类与梳理，为该新兴交叉领域绘制了研究全景图。 |
| [^39] | [PCLA-Bridge: A Framework for Cross-Domain Deployment of CARLA Agents](https://arxiv.org/abs/2610.28696) | PCLA-Bridge是一个无需修改代码即可将CARLA排行榜自动驾驶智能体跨域部署到其他模拟引擎、真实数据集和物理平台的工具，它通过处理依赖隔离、张量封送、路由和合成时序，使集成者只需提供最小接口即可完成复用。 |
| [^40] | [Who Approved This? Coding Agents, Human Accounts, and Merge-Gate Evidence at a Fintech Startup](https://arxiv.org/abs/2610.28683) | 该论文通过分析一家金融科技初创公司十二周内786次合并请求的审计数据，揭示尽管设有非作者评审门禁，仍有27%的合并仅依赖编码智能体冒用技术负责人账号提交的批准，且平台记录无法证明任何合并的实际决策者，暴露了智能体时代合并门禁审计证据的失效。 |
| [^41] | [Runtime Semantic Discovery for Evolving Action Alphabets](https://arxiv.org/abs/2610.28682) | 本文提出一种基于执行轨迹中邻居上下文的轻量级运行时方法，通过定义k-邻居等价关系来识别演化系统中新动作是否与已知动作具有相同的操作角色，从而在不重建全局行为模型的情况下比较可观察字母表不同的系统。 |
| [^42] | [VerifierBench: Benchmarking the Generation of Executable Verifiers for LLM Agents](https://arxiv.org/abs/2610.28679) | 该论文提出了VerifierBench基准，用于评估大语言模型智能体生成可执行验证器的能力，发现现有模型生成的验证器虽然能可靠地拒绝无效解，但往往因过于严格而错误拒绝有效解。 |
| [^43] | [Benchmarking the Quantum Leap: Auditing LLM Evaluation for Quantum Code Generation](https://arxiv.org/abs/2610.28678) | 该研究系统性地基准测试了15个大型语言模型在量子代码生成任务上的表现，并首次揭示评估器的设计差异会极大扭曲评分结果——仅移除对未指定量的检查即可使模型pass@1从0%跃升至94.1%，说明量子代码生成评估亟需严格的审计与标准化。 |
| [^44] | [Compilation Is Not Correctness: Scan-by-Scan Evaluation of LLM-Generated PLC Programs](https://arxiv.org/abs/2610.28675) | 提出CycleBench基准测试，通过逐扫描周期比较LLM生成的PLC程序与参考程序的行为输出，揭示编译通过率与实际执行通过率之间存在11.92个百分点的差距，证明编译成功并不代表程序正确。 |
| [^45] | [Who Verifies the Builder? Toward Independent Assurance for Agent-Produced Software](https://arxiv.org/abs/2610.28667) | 当智能体既构建软件又自证正确性时，通过测试不再是独立证据，本文主张构建与验证在结构上分离，提出独立保障平面与“验证契约”框架，为智能体生成的软件提供独立的质量保证。 |
| [^46] | [From Natural Language Intent to Executable Control: Hierarchical Feedback-Guided LLM Planning for Multi-component Cyber-Physical Systems](https://arxiv.org/abs/2610.22441) | 提出了一种分层反馈框架，通过让LLM同时生成结构化任务计划与信号时序逻辑（STL）约束，并利用可靠算法识别二者间的矛盾以提供不同类型的反馈，从而引导LLM为多组件信息物理系统生成可执行的任务计划。 |
| [^47] | [WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents](https://arxiv.org/abs/2610.22039) | WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。 |
| [^48] | [When Agent Evaluation Becomes Unreliable: An Empirical Study of Oracle Failures and Configuration-Dependent Measurement in an LLM Agent Benchmark](https://arxiv.org/abs/2610.16067) | 本文提出一种面向LLM智能体评估工具的预言机完整性审计方法，发现了一个会在评分前改变环境状态的奖励函数缺陷，使基准测得的成功率从94.3%降至65.5%，并揭示了配置依赖的测量效应对评估结果的显著影响。 |
| [^49] | [TileCas: Cascading Kernel Development Across Programming Abstractions](https://arxiv.org/abs/2610.14740) | TileCas提出一个统一可插拔的接口，让编码智能体能够通过跨层诊断和抽象调度，在多个可编辑的编程抽象层次之间迁移和改进内核开发，从而兼顾高级DSL的搜索简便性与低级表示的细粒度控制。 |
| [^50] | [PatchRisk: Forecasting Future Vulnerability Exposure in Open-Source Dependency Networks](https://arxiv.org/abs/2609.13719) | 提出PatchRisk方法，基于发布时的包版本依赖图预测未来传递依赖是否会出现漏洞公告，实现软件供应链的主动风险预警，并应对标签泄露等评估陷阱。 |
| [^51] | [The Best Programming Language for Tokenmaxxing: An Investigation of Coding Agent Behavior Across Programming Languages](https://arxiv.org/abs/2607.22807) | 编程智能体的Token消耗因编程语言而显著不同且跨模型一致，其原因是智能体在不熟悉的语言中反复生成无法编译的代码并重复修改已通过的解决方案。 |
| [^52] | [The Hitchhiker's Guide to Program Analysis, Part III: Mostly Harmless LLMs](https://arxiv.org/abs/2606.15122) | 论文提出Evident系统，让LLM不再直接判断静态分析警告的真伪，而是通过识别相关代码并构建测试驱动（harness）来支持精确的针对性静态分析，从而可靠地过滤误报。 |
| [^53] | [Properties with Holes: Meta-Property-Based Testing of Data-Intensive Computing Frameworks](https://arxiv.org/abs/2606.11132) | 该论文提出Holey工具，引入“带洞的元性质”这一概念——固定工作负载的变换方式与输出需满足的关系、将表达式留空以待填充——从而系统化地为Spark、Flink等数据密集型计算框架生成强判定性质的测试，捕捉那些静默返回错误结果的缺陷。 |
| [^54] | [Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory](https://arxiv.org/abs/2606.06523) | 本文提出Lean4Agent，这是首个利用Lean4依赖类型形式化语言对智能体工作流的语义一致性进行形式化建模与验证、并能对执行轨迹揭示的运行时故障进行定位的框架。 |
| [^55] | [EvoRepair: Enhancing Vulnerability Repair Agents Through Experience-Based Self-Evolution](https://arxiv.org/abs/2605.30105) | EvoRepair是首个基于经验自进化的自动化漏洞修复智能体框架，通过循环的学习-修复过程实现修复经验的积累、精炼与跨漏洞复用，从而提升LLM的漏洞修复能力。 |
| [^56] | [MockingBird: Context-aware Repository-level Code Translation with Isolated Validation](https://arxiv.org/abs/2511.21878) | MockingBird 通过结合上下文感知的跨语言类型解析与基于模拟对象的隔离验证，无需翻译测试即可验证仓库级代码翻译的功能等价性，将 AlphaTrans 基准上 Java 到 Python 翻译的功能等价率从最先进的 25.14% 显著提升。 |
| [^57] | [TopoMap: A Feature-based Discriminator of the Topographical Regions in the Test Input Space](https://arxiv.org/abs/2509.03242) | TopoMap是一种模型无关的方法，通过基于输入特征对测试输入空间进行拓扑划分，帮助用户识别代表性不足、失败率高或模型更新后预测发生变化的区域，从而指导数据生成和标注工作的优先级排序。 |
| [^58] | [Stack Trace-Based Crash Deduplication with Transformer Adaptation](https://arxiv.org/abs/2508.19449) | 本文提出dedupT，一种基于Transformer的崩溃去重方法，通过整体建模堆栈跟踪并适配预训练语言模型，显著提升了重复崩溃排序和唯一崩溃检测的准确性，减少了人工分类负担。 |
| [^59] | [Generating Mitigations for Downstream Projects to Neutralize Upstream Library Vulnerability](https://arxiv.org/abs/2503.24273) | 提出LUMEN方法，在无法获取上游库源代码或补丁的情况下，为下游项目自动生成缓解措施，以中和上游第三方库的漏洞。 |

# 详细

[^1]: 基于对位语言验证自动形式化规范

    Validating Autoformalized Specifications via Contrapuntal Languages

    [https://arxiv.org/abs/2610.30059](https://arxiv.org/abs/2610.30059)

    提出“对位语言”方法——使用一对彼此不存在保持结构翻译的任务特定形式语言来引入表示多样性，从而消除模型错误之间的相关性，使得多个LLM输出之间的一致性能够有效验证自动形式化规范是否忠实于用户意图，而无需人工审查等昂贵神谕。

    

    形式化规范可以使AI生成的代码具备可机械验证性。由于用户意图通常以自然语言而非形式逻辑表达，因此必须首先从自然语言中推导出规范，例如通过基于大语言模型（LLM）的自动形式化。然而，自动形式化得到的规范本身并不可信，因此一个关键挑战是：在不依赖昂贵“神谕”（如人工审查）的前提下，验证这些规范是否准确捕捉了用户意图。在这种无需神谕的场景下，一种自然的策略是对多个模型输出进行投票，但由于模型错误往往是相互关联的，输出之间的一致可能反映的是共同的错误而非正确性。我们的关键洞察是：这些错误可以通过表示多样性来消除相关性。我们通过对位语言来实现这种多样性：对位语言是成对的任务特定形式语言，每种语言都被设计为能够表达目标规范，但两者之间不存在保持结构的翻译。LLM会将……（原文摘要截断）

    arXiv:2610.30059v1 Announce Type: new  Abstract: Formal specifications can make AI-generated code mechanically checkable. Because user intent is typically expressed in natural language rather than formal logic, specifications must first be derived from it, e.g., via LLM-based autoformalization. Yet autoformalized specifications are themselves untrusted, so a key challenge is validating whether they capture user intent without relying on a costly oracle such as human review. In this oracle-free setting, a natural strategy is to vote among model outputs, but because model errors are often correlated, agreement among outputs may reflect shared mistakes rather than correctness. Our key insight is that these errors can be decorrelated through representational diversity. We achieve this diversity with contrapuntal languages: pairs of task-specific formal languages, each designed to express the intended specification, yet admitting no structure-preserving translation between them. An LLM form
    
[^2]: SWE-DIVE：基于动态用户模拟的多轮编程智能体评估

    SWE-DIVE: Multi-Turn Coding Agent Evaluation with Dynamic User Simulation

    [https://arxiv.org/abs/2610.30047](https://arxiv.org/abs/2610.30047)

    该论文提出 SWE-DIVE 框架，通过引入可模拟多种行为模式的动态用户模拟器和对话验证器，在多轮交互环境中评估编程智能体，弥补了传统单轮、格式规范提示词基准测试与真实使用场景之间的差距。

    

    编程智能体通常使用一次性的、格式良好的提示词进行评估。诸如 SWE-bench 等基准测试使用现有的 GitHub issue 作为编程智能体的输入，并评估该智能体是否成功完成任务。然而，在现实世界的使用场景中，提示词往往不像现有的 GitHub issue 那样格式规范或结构清晰。此外，编程智能体会话并不局限于单个提示词；它通常涉及多轮交互，用户会在交互过程中迭代地添加智能体必须完成的新任务和需求。为了对这些能力进行基准测试，我们开发了 SWE-DIVE，这是一个动态的、交互式的验证与评估框架，用于在多轮环境中对编程智能体进行基准测试。为实现这一目标，我们引入了一个能够模拟不同行为模式并与编程智能体进行通信的用户模拟器。我们还引入了一个对话验证器，用于评估用户模拟器的输出以检（原文摘要在此处截断）

    arXiv:2610.30047v1 Announce Type: new  Abstract: Coding agents are typically evaluated using one-shot, well-formulated prompts. Benchmarks such as SWE-bench use existing GitHub issues as input for a coding agent and evaluate whether the agent successfully solves the task. However, in real-world use cases, prompts are often not as well-formulated or structured as existing GitHub issues. Additionally, a coding agent session is not limited to a single prompt; it typically involves a multi-turn interaction in which the user iteratively adds new tasks and requirements that the agent must fulfill.   To benchmark these abilities, we developed SWE-DIVE, a dynamic, interactive validation and evaluation framework for benchmarking coding agents in a multi-turn environment. To achieve this, we introduce a user simulator that can simulate different behavioral patterns and communicate with the coding agent. We also introduce a conversation validator that evaluates the user simulator's outputs to det
    
[^3]: Code Comprehensor：面向大型代码仓库的层次化文档与跨模态检索

    Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories

    [https://arxiv.org/abs/2610.30016](https://arxiv.org/abs/2610.30016)

    该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。

    

    理解大型、复杂且快速演进的代码库仍然是规模化软件工程的核心挑战，它阻碍了新人上手、代码维护、重构和影响分析。将这些信息提供给大语言模型（LLM）是另一个障碍，因为LLM不仅需要理解所需的代码方法，还需要理解其在整个仓库全局结构中相关联的方法。我们提出了Code Comprehensor，这是一个混合检索-推理框架，结合了：（i）代码依赖调用图，（ii）CodeScribe——利用代码感知的LLM在仓库调用图上生成层次化文档，（iii）CodeContrast——一个CLIP风格的双编码器模型，基于CodeScribe的输出进行训练以学习对齐的文本-代码嵌入。CodeContrast能够实现快速、语义精准的相关代码上下文邻域检索，这些邻域作为基于MCP的LLM编排器回答领域特定查询时的可靠依据。

    arXiv:2610.30016v1 Announce Type: cross  Abstract: Understanding large, complex, and rapidly evolving code bases remains a core challenge for software engineering at scale, hindering onboarding, maintenance, refactoring, and impact analysis. Providing this information to a Large-Language Model (LLM) is another hurdle, as the LLM must understand not only the required code method but also its associated methods in the grand scheme of the repository. We propose Code Comprehensor, a hybrid retrieval-reasoning framework combining (i) a code dependency callgraph, (ii) CodeScribe, which uses code-aware LLMs to generate hierarchical documentation across repository call graphs, and (iii) CodeContrast, a CLIP-style dual-encoder model trained on CodeScribe output to learn aligned text-code embeddings. CodeContrast enables fast, semantically precise retrieval of relevant code-context neighborhoods, which serve as grounded evidence for an MCP-based LLM orchestrator answering domain-specific queries
    
[^4]: HiddenLens：一个用于检测和解决软件需求中隐藏假设的多层大语言模型框架

    HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements

    [https://arxiv.org/abs/2610.29996](https://arxiv.org/abs/2610.29996)

    HiddenLens是首个专为检测、分类、质疑并解决软件需求中隐藏假设而设计的六层BERT-LLM框架，能够超越单纯的歧义标记，在早期发现并化解需求中的隐含假设，从而显著降低后期修复成本。

    

    自然语言形式的软件需求通常省略了关于性能、运行条件、用户行为和领域约束的假设，这些遗漏会影响架构设计、测试和部署。如果这些假设直到实现阶段才被发现，解决它们的代价将远高于在早期阶段处理。尽管歧义性和模糊性已得到广泛研究，但人们对嵌入在看似清晰合理的需求中的假设关注较少。我们提出了HiddenLens，一个新颖的六层BERT-LLM流水线，据我们所知，这是首个专为检测、分类、质疑和解决隐藏假设而设计的框架，而不仅仅是标记歧义。第一层使用BERT注意力熵来定位未解决的文本片段；第二层使用思维链提示的大语言模型推理来阐明每个假设及其失效风险；第三层使用微调后的BERT将假设分类为六类非功能性需求：性能、安全性、可扩展性等。

    arXiv:2610.29996v1 Announce Type: cross  Abstract: Natural-language software requirements often omit assumptions about performance, operating conditions, user behaviour, and domain constraints, affecting architecture, testing, and deployment. If undiscovered until implementation, resolving them is far more costly than addressing them early. Although ambiguity and vagueness are widely studied, less attention has been given to assumptions embedded in requirements that appear clear and reasonable. We propose HiddenLens, a novel six-layer BERT-LLM pipeline and, to our knowledge, the first framework engineered to detect, classify, question, and resolve hidden assumptions rather than merely flag ambiguity. Layer 1 uses BERT attention entropy to localise unresolved spans; Layer 2 uses CoT-prompted LLM reasoning to articulate each assumption and its failure risk; Layer 3 uses fine-tuned BERT to classify assumptions into six non-functional requirement categories: performance, security, scalabil
    
[^5]: FaultGene：基于大语言模型辅助的故障增强测试生成方法，用于自动驾驶系统的硬件在环验证

    FaultGene: LLM-Assisted Fault-Augmented Test Generation for Hardware-in-the-Loop Validation of Autonomous Driving Systems

    [https://arxiv.org/abs/2610.29969](https://arxiv.org/abs/2610.29969)

    FaultGene利用大语言模型自动为自动驾驶系统的硬件在环验证生成可执行的故障增强测试场景，解决了传统手工准备故障场景导致的测试覆盖不足、可扩展性差和效率低下的问题。

    

    对自动驾驶信息物理系统的可靠验证，不仅需要测试正常的驾驶行为，还需要测试由感知和嵌入式计算故障所引发的失效。实地道路测试能够提供真实的系统行为，但成本高昂且难以复现；仿真虽然支持大规模评估，却无法完全捕捉依赖硬件的行为特性及故障传播过程。硬件在环（HIL）验证通过在仿真环境中将驾驶软件栈运行于真实嵌入式硬件之上，弥合了这两者之间的差距。然而，为HIL测试准备故障增强场景目前仍主要依赖手工流程，这限制了测试覆盖率、可扩展性和效率。我们提出了FaultGene，一个由大语言模型辅助的框架，用于自动生成可在HIL故障韧性验证中执行的可执行故障增强测试。给定一个基准Scenic/CARLA场景和结构化的故障规范，FaultGene生成可执行的测试场景。

    arXiv:2610.29969v1 Announce Type: new  Abstract: Reliable validation of autonomous driving cyber-physical systems requires testing not only nominal driving behavior but also failures arising from faults in sensing and embedded computation. Physical road testing provides realistic system behavior but is costly and difficult to reproduce, while simulation enables scalable evaluation but cannot fully capture hardware-dependent behavior and fault propagation. Hardware-in-the-loop (HIL) validation bridges this gap by executing the driving stack on real embedded hardware within a simulated environment. However, preparing fault-augmented scenarios for HIL testing remains largely a manual process, limiting test coverage, scalability, and efficiency.   We present FaultGene, an LLM-assisted framework for automated generation of executable fault-augmented tests for HIL fault-resilience validation. Given a baseline Scenic/CARLA scenario and a structured fault specification, FaultGene generates a f
    
[^6]: DynaCI：通过动态分析生成GitHub Actions工作流

    DynaCI: Generating GitHub Actions Workflows via Dynamic Analysis

    [https://arxiv.org/abs/2610.29890](https://arxiv.org/abs/2610.29890)

    提出DynaCI，通过动态分析利用开发者本地环境的运行时信息自动生成GitHub Actions CI工作流，解决了现有方法需要开发者手动编写工作流规范、且难以表达需求的局限。

    

    持续集成（CI）是一项通过构建和测试自动验证代码变更的重要实践。然而，配置CI工作流仍然极具挑战性，因为现代软件通常具有复杂的架构、众多的依赖关系，并支持多样化的执行环境。此外，大多数CI平台还要求使用其特定的领域专用语言来编写工作流。因此，CI工作流往往容易出错且难以维护，尤其对于不熟悉CI语义的开发者而言更是如此。先前关于自动化CI工作流生成的工作要求开发者创建规范（如使用自然语言、部分工作流等）来描述其预期的工作流。然而在实践中，开发者可能难以清晰地表达需求，也可能不了解可用的CI功能。为了解决这些局限性，我们提出了DynaCI，这是一种新颖的方法，它利用开发者本地环境的运行时信息……

    arXiv:2610.29890v1 Announce Type: new  Abstract: Continuous Integration (CI) is an essential practice that automatically validates code changes through builds and tests. However, configuring CI workflows remains challenging, as modern software often features complex architectures, numerous dependencies, and support for diverse execution environments. Most CI platforms also require workflows to be written in their particular domain-specific languages. As a result, CI workflows are often error-prone and hard to maintain, especially for those unfamiliar with CI semantics. Prior work on automated CI workflow generation require developers to create specifications that describe their intended workflows using natural language, partial workflows, etc. However, in practice, developers may struggle to articulate requirements and may be unaware of available CI features. To address these limitations, we propose DynaCI, a novel approach that uses runtime information from developers' local environme
    
[^7]: PAUSE：在编码智能体首次源代码编辑之前的免搜索式引导

    PAUSE: Search-Free Steering of Coding Agents Before Their First Source-Code Edit

    [https://arxiv.org/abs/2610.29870](https://arxiv.org/abs/2610.29870)

    PAUSE通过在编码智能体第一次修改源代码前拦截其编辑请求、由独立规划器一次性生成修复计划，实现无需监控器、无需检查点搜索、无额外模型调用的单次干预式智能体引导。

    

    编码智能体可能在代码仓库探索和测试上投入大量精力，却仍然生成不成功的补丁。现有的在线引导方法要么周期性地审查智能体的执行轨迹，要么在监控器检测到行为异常时调用顾问；频繁的审查代价高昂，而且没有出现异常行为也并不代表智能体走在正确的道路上。我们提出PAUSE，一个通过单次规划干预来引导编码智能体的框架。智能体在完成探索、缺陷复现和测试之后，大约在轨迹进度的60%（中位数）处才会进行第一次源代码编辑。PAUSE在该编辑请求执行之前将其拦截，这既不需要学习得到的监控器，也不需要在检查点上进行搜索，更不需要额外的模型调用（免搜索）。一个独立的规划器读取任务描述和编辑前的对话记录，撰写一份修复计划；随后同一个智能体（作为执行器）带着该计划继续运行，并保留其原有的上下文和工作区。PAUSE应用o……（摘要原文在此处截断）

    arXiv:2610.29870v1 Announce Type: new  Abstract: Coding agents can invest substantial effort in repository exploration and testing yet still produce unsuccessful patches. Existing online steering methods review trajectories periodically or invoke an advisor when a monitor detects behavioral problems; frequent reviews are costly, and the absence of anomalous behavior does not mean that an agent is on the right track. We propose PAUSE, a framework for steering coding agents with a single planning intervention. Agents reach their first source-code edit at a median of 60% of trajectory progress, after exploration, bug reproduction, and testing. PAUSE intercepts this edit request before it executes, which requires no learned monitor, no search over checkpoints, and no extra model call (search-free). A separate Planner reads the task and the pre-edit transcript and writes one repair plan; the same agent, the Executor, resumes with this plan, keeping its context and workspace. PAUSE applies o
    
[^8]: 使用软件产品线控制神经影像分析流水线中的变异性

    Controlling Variability in~Neuroimaging Analysis Pipelines with~Software Product Lines

    [https://arxiv.org/abs/2610.29821](https://arxiv.org/abs/2610.29821)

    本文倡导将软件产品线（SPL）原则引入神经影像分析流水线的设计与管理，以控制不同研究团队流水线之间的变异性，从而提升MRI研究的可复现性和一致性。

    

    磁共振成像（MRI）是一种用于医学诊断的采集技术，尤其被广泛应用于结构和功能MRI（即fMRI）图像分析。众所周知，MRI数据复杂且充满噪声，因此需要构建分析流水线来提取和增强所获取图像中的相关信息。NARPS研究揭示了这样一个现象：当多个独立团队试图回答同一个关于大脑功能的研究问题时，他们所构建的不同流水线会产生显著的变异性。用不同实现方式完成同一任务是软件产品线（SPL）的典型应用场景，但神经影像领域的从业者可能对此并不熟悉。因此，从这些不同的流水线中归纳出一个软件产品线，本质上是一个典型的逆向工程问题，并伴随着其固有的困难。在本文中，我们倡导从概念上采纳软件产品线的原则，将其作为提升可复现性和一致性的一条有前景的方向。

    arXiv:2610.29821v1 Announce Type: new  Abstract: Magnetic resonance imaging (MRI) is an acquisition technique used for medical diagnosis. It is notably used for structural and functional MRI (or fMRI) image analysis. MRI data is known to be complex and noisy, so pipelines are created to extract and enhance relevant information in retrieved images. The NARPS study highlighted the significant variability that can emerge from different pipelines set up by independent teams while trying to answer the same research question about brain functioning. Having different implementations realizing the same task is a typical playground for software product lines (SPLs), which may be out of the sight of neuroimage practitioners. Thus, proposing an SPL out of these different pipelines falls back on a typical reverse-engineering problem with its usual difficulties. In this document, we advocate for the conceptual adoption of SPL principles as a promising direction to improve reproducibility and consis
    
[^9]: ALIAS：评估大语言模型编写的TLA+规约的性能与记忆化现象

    ALIAS: Evaluating Performance and Memorization in LLM-Written TLA+ Specifications

    [https://arxiv.org/abs/2610.29767](https://arxiv.org/abs/2610.29767)

    该论文利用TLA+模型检查器作为验证神谕，发现LLM在标识符重命名任务上的正确率下降主要来自要求精确标识符名称的评分方式，若按行为语义评分，记忆化效应实际很小甚至消失。

    

    记忆化（memorization）会夸大大语言模型（LLM）的基准测试分数，因为模型可以复现其在训练中见过的答案。研究人员通过“一致标识符重命名”（CIR）来检验这种现象，即给任务中的每个标识符起新名字，并将重命名任务上较低的得分视为模型记忆的证据。我们在时序动作逻辑语言（TLA+）上测试了这一方法，该语言的模型检查器可以作为一个“神谕”（oracle），确认重命名后的任务是否保留了原始任务的精确行为。在111个TLA+-Bench任务上，重命名使Claude Opus 4.8的正确率从28%降至18%，GPT-5从9%降至7%。这一下降大部分源于测试的评分方式，即要求提供大多数描述中从未给出的精确名称。若按行为进行评分，下降幅度可缩小至Opus 4.8约3个百分点，GPT-5则为0。在分别在互不相交的任务半区上微调的两个模型家族中，重命名消除了大约一半（摘要在此处被截断）

    arXiv:2610.29767v1 Announce Type: new  Abstract: Memorization inflates the benchmark scores of large language models (LLMs), since a model can reproduce answers it saw in training. Researchers test for it with consistent identifier renaming (CIR), which gives every identifier in a task a new name, and treat a lower score on the renamed task as evidence of recall. We test this method on the Temporal Logic of Actions Plus (TLA+) language, whose model checker acts as an oracle by confirming whether a renamed task preserves the exact behaviors of the original. Across 111 TLA+-Bench tasks, renaming lowers the correctness of Claude Opus 4.8 from 28% to 18% and of GPT-5 from 9% to 7%. Most of this decrease comes from how the test grades, which requires exact names that most descriptions never state. Grading by behavior shrinks the decrease to about 3 percentage points for Opus 4.8 and to 0 for GPT-5. In two model families fine-tuned on disjoint halves of the tasks, renaming removes about half
    
[^10]: AssureCode：面向可信智能体编程的结构化论证

    AssureCode: Structured Justification for Trustworthy Agentic Coding

    [https://arxiv.org/abs/2610.29758](https://arxiv.org/abs/2610.29758)

    该论文借鉴安全关键系统的保证工程技术，提出AssureCode方法论，通过结构化论证将自然语言提示、形式化规范、开发者意图假设与程序分析证据关联起来，从而提升AI智能体生成代码的可信度与可验证性。

    

    在软件开发中使用基于AI的工具正在迅速改变软件的生产方式，使得在有限的人工实现工作量下能够生成大量代码。然而，这给确保生成的代码制品符合开发者意图并经过适当验证带来了根本性挑战。在本工作中，我们提出使用结构化论证来驱动AI辅助的软件开发。受安全关键系统保证工程技术的启发，我们定义了一种论证理由语言，该语言以结构化且可验证的格式，显式关联非正式的自然语言提示、形式化规范、关于开发者意图的假设以及程序分析证据。随后，我们介绍了AssureCode，这是一种由工具支持的方法论，用于在智能体代码生成过程中创建此类论证理由。我们对所实现系统进行了实证分析……

    arXiv:2610.29758v1 Announce Type: new  Abstract: The use of AI-based tools in software development is rapidly changing how software is produced, enabling large volumes of code to be generated with limited human implementation effort. However, this raises fundamental challenges for ensuring that generated artifacts are aligned with developer intent and have been appropriately validated. In this work, we propose the use of structured argumentation in driving AI-assisted software development. Inspired by techniques from assurance engineering for safety-critical systems, we define a language of justification rationales that explicitly relate informal natural language prompts, formal specifications, assumptions about developer intentions, and program analysis evidence in a structured and verifiable format. We then describe AssureCode, a tool-supported methodology for creating such rationales as part of agentic code generation. We provide an empirical analysis of our implementation where we 
    
[^11]: VeruSteer：学习在 Verus 中接下来该修复什么

    VeruSteer: Learning What to Repair Next in Verus

    [https://arxiv.org/abs/2610.29755](https://arxiv.org/abs/2610.29755)

    VeruSteer 通过离线构建修复树并比较不同修复轨迹的验证结果，训练出一个结合逐步模仿与轨迹偏好排序的神经选择器，从而动态决定 Verus 证明修复中下一步应修复的错误类型，取代了传统的预设静态优先级策略。

    

    基于大语言模型（LLM）的验证系统通过根据 Verus 的诊断信息修改程序来修复 Rust 证明。当报告了多种错误类型时，诸如 AutoVerus 和 VeruSAGE 之类的流水线会使用预设的优先级来选择下一个修复目标。然而，每次修复都会改变程序及其后续出现的错误，因此相同的已报告错误类型在不同状态下可能需要不同的选择。我们提出了 VeruSteer，它从备选修复轨迹的结果中学习如何选择错误类型。该方法离线构建修复树，并通过验证是否成功、修复轮次数以及最终验证进展来比较源自同一初始程序的不同路径。这些比较用于监督一个选择器，该选择器将逐步模仿与轨迹偏好排序相结合。该选择器对程序、验证器反馈以及先前的选择应用自注意力机制，以对当前报告的错误类型进行排序。它在单个修复轨迹中取代了默认的优先级设置。

    arXiv:2610.29755v1 Announce Type: new  Abstract: LLM-based verification systems repair Rust proofs by revising programs in response to Verus diagnostics. When several error types are reported, pipelines such as AutoVerus and VeruSAGE select the next repair target using a preset priority. Yet each repair changes the program and the errors that follow, so the same reported types can require different choices at different states. We present VeruSteer, which learns to select error types from the outcomes of alternative repair trajectories. It constructs repair trees offline and compares paths from the same initial program by verification success, repair rounds, and final verification progress. These comparisons supervise a selector that combines step-wise imitation with trajectory preference ranking. The selector uses self-attention over the program, verifier feedback, and earlier selections to rank the currently reported types. It replaces the default priority within a single repair traje
    
[^12]: 桥接大语言模型的基础能力与智能体能力

    Bridging Foundational and Agentic Capabilities of LLMs

    [https://arxiv.org/abs/2610.29751](https://arxiv.org/abs/2610.29751)

    该论文提出一个操作层面的测量框架，通过定义“能力实现差距”来量化大语言模型基础能力向智能体能力的转化程度，发现基础能力基准只能部分预测智能体操作表现，画像相似的模型在实际执行中差异显著。

    

    大语言模型（LLM）越来越多地被部署为智能体，但其基础能力如何转化为智能体能力仍不清楚。端到端的评分无法判断失败究竟是由于基准能力低下还是部署环节薄弱所致，这限制了可操作的诊断。我们通过一个操作层面的测量框架来桥接基础能力与智能体能力。我们使用从能力基准得出的基础画像（foundational profile）和从交互轨迹得出的操作画像（operation profile）来刻画每个模型，并将“能力实现差距”定义为观测到的操作性能与基于基础能力预测值之间的偏差。在 τ²-Bench 上对十五个模型服务的实验表明，基础画像只能部分预测智能体操作表现。画像相近的模型在执行层面可能存在显著差异，且这些差距在不同任务子集之间保持一致。操作……

    arXiv:2610.29751v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as agents, yet it remains unclear how their foundational capabilities translate into agentic capabilities. End-to-end scores leave unresolved whether low benchmark capability or weak deployment drives a failure, limiting actionable diagnosis. We bridge foundational and agentic capabilities through an operation-level measurement framework. We represent each model with a foundational profile derived from capability benchmarks and an operation profile derived from interaction trajectories, and define capability realization gaps as deviations of observed operation performance from predictions based on foundational capabilities. Experiments on fifteen model services in $\tau^2$-Bench show that foundational profiles only partially predict agentic operations. Models with comparable profiles can differ substantially in execution, and these gaps remain consistent across task subsets. Operatio
    
[^13]: 元模型引导的大语言模型解码以实现一致的DSL生成

    Metamodel-Guided LLM Decoding for Consistent DSL Generation

    [https://arxiv.org/abs/2610.29749](https://arxiv.org/abs/2610.29749)

    提出MeMoDecode解码框架，在LLM生成DSL的过程中联合强制执行具体语法、抽象语法（元模型）和良构性约束，通过增量映射到部分模型并验证其可补全性来确保生成结果的一致性。

    

    arXiv:2610.29749v1 公告类型：新论文 摘要：大语言模型（LLM）正被越来越多地用于生成以领域特定语言（DSL）编写的软件工程制品。正确的DSL实例通常不仅要符合具体语法，还必须符合由元模型定义并辅以语义良构性约束的抽象语法。现有的约束解码方法通常只能保证符合具体语法，或只能保证符合基于生成文本所表达的有限语义约束集合，这使得元模型级别的约束一致性在生成过程中难以强制实施。我们提出了MeMoDecode：一个能够联合强制执行DSL的具体语法、抽象语法和良构性约束的解码框架。具体而言，每当LLM生成一个完整的元模型概念时，MeMoDecode会将当前输出增量地映射为一个部分模型，并验证其是否仍可被补全为一个一致的实例。我们将……（原文摘要在此处截断）

    arXiv:2610.29749v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to generate software engineering artifacts written in domain-specific languages (DSLs). Correct DSL instances usually must comply with not only the concrete syntax but also the abstract syntax defined by a metamodel augmented with semantic well-formedness constraints. Existing constrained decoding approaches typically guarantee compliance only with concrete syntax or with a limited set of semantic constraints expressed over the generated text, making metamodel-level constraint consistency difficult to enforce during generation. We present MeMoDecode: a decoding framework that enforces a DSL's concrete syntax, abstract syntax, and well-formedness constraints jointly. Specifically, whenever the LLM generates a complete metamodel concept, MeMoDecode incrementally maps the current output to a partial model and verifies whether it can still be completed into a consistent instance. We combine 
    
[^14]: 生成式AI智能体的需求工程：BARD过程理论

    Engineering Requirements for Generative AI Agents: The BARD Process Theory

    [https://arxiv.org/abs/2610.29741](https://arxiv.org/abs/2610.29741)

    该研究通过对36位从业者访谈的扎根理论分析，提出了面向生成式AI智能体的需求工程过程理论BARD，指出智能体需求并非预先固定，而是通过“界定范围—构建智能体—观察行为—修订需求”这一循环周期被持续发现。

    

    团队如何为围绕生成式人工智能（GenAI）智能体构建的系统开展需求工程，在实证上仍是未知领域。现有的需求工程（RE）研究主要针对广义的人工智能和机器学习系统，但尚未为智能体系统提供过程性论述——这类系统的非确定性行为无法在设计时被预知。为了解团队的实际做法，我们采用社会技术扎根理论（Socio-Technical Grounded Theory），对从业者进行了36次半结构化访谈，并将数据收集与分析交替进行，使新出现的发现驱动我们的理论抽样，并塑造后续的编码轮次。研究成果是行为智能体需求发现（BARD），即面向GenAI智能体的需求工程过程理论。BARD认为，需求并非预先固定，而是通过循环往复的四项活动周期被发现：团队界定需求范围、构建智能体、观察其行为，并据此修订他们所提出的需求。通过这个四阶段

    arXiv:2610.29741v1 Announce Type: new  Abstract: How teams engineer requirements for systems built around Generative AI (GenAI) Agents remains empirically uncharted. Existing Requirements Engineering (RE) research mainly addresses AI and ML-based systems broadly but provides no process account for agentic systems, whose non-deterministic behavior cannot be anticipated at design time. To understand how teams actually proceed, we applied Socio-Technical Grounded Theory across 36 semi-structured interviews with practitioners, interleaving data collection and analysis so that emerging findings drove our theoretical sampling and shaped successive rounds of coding. The result is Behavioral Agent Requirement Discovery (BARD), the process theory of RE for GenAI Agents. BARD holds that requirements are not fixed up front but discovered through a recurring cycle of four activities: teams scope the need, build the agent, observe its behavior, and revise what they ask for. Through this four-phase 
    
[^15]: SAKURA：面向手机端隐私图像编辑的速率自适应注意力键值状态服务

    SAKURA: Serving Attention Key-Value State Under Rate Adaptation for Private Image Editing on Phones

    [https://arxiv.org/abs/2610.29733](https://arxiv.org/abs/2610.29733)

    SAKURA提出了一种云边协同架构，通过速率自适应机制在带宽预算内向手机传输可复用的注意力KV状态，从而在保护隐私的前提下实现高效的端侧图像编辑。

    

    扩散Transformer被广泛应用于图像编辑服务中。为了同时兼顾效率和隐私，现有系统通常采用云-端协同的方式。将针对特定请求的推理保留在设备端进行，可以避免将个人图像和提示词发送到云端。这类工作负载通常使用通用模板，例如换脸和虚拟试穿，这使得KV状态可以被预先计算并在不同请求之间复用。然而，这种可复用的状态体积庞大，且随模板数量的增加而增长，因此将所有模板状态都存储在移动设备上并不现实。将它们存储在云端虽然可以解决容量问题，但缓存传输被置于推理路径上，甚至可能比设备端的计算还要慢，尤其是在移动网络波动的情况下。我们提出了SAKURA，这是一种云边协同架构，能够在带宽预算约束下向移动设备交付可复用的注意力状态。该系统首先利用细粒度的token重要性（摘要在此处截断）

    arXiv:2610.29733v1 Announce Type: new  Abstract: Diffusion transformers are widely used in image-editing services. To preserve both efficiency and privacy, existing systems commonly rely on cloud-device collaboration. Keeping request-specific inference on the device avoids sending personal images and prompts to the cloud. These workloads often use common templates, such as in face swapping and virtual try-on, enabling KV state to be precomputed and reused across requests. However, this reusable state is large and scales with the number of templates, making it impractical to store all of them on mobile devices. Storing them in the cloud addresses this capacity issue, but cache transfer is placed onto the inference path and can even be slower than on-device computation, especially under fluctuating mobile networks.   We introduce SAKURA, a cloud-edge architecture that delivers reusable attention state to mobile devices under a bandwidth budget. It first uses a fine-grained token-importan
    
[^16]: 面向系统级性能优化的编码智能体基准测试

    Benchmarking Coding Agents for System-Level Performance Optimization

    [https://arxiv.org/abs/2610.29695](https://arxiv.org/abs/2610.29695)

    本文提出了MicroPerf-Bench，首个用于评估编码智能体在微服务系统中进行端到端性能优化能力的基准，通过向真实开源系统注入基于实证的性能反模式来构建测试场景，填补了现有基准仅停留在函数或仓库层面的空白。

    

    编码智能体正日益成为自动化软件工程任务的重要范式。然而，现有的编码基准主要在函数或代码仓库层面对智能体进行评估，它们在更复杂的分布式软件系统中的能力在很大程度上尚未被探索。在实践中，许多现代应用程序采用微服务架构，其端到端性能产生于多个独立运行的服务之间的交互，这要求智能体能够超越局部代码变更进行推理。为了研究这一能力，我们提出了MicroPerf-Bench，一个用于评估编码智能体在微服务系统中进行端到端性能优化能力的基准。构建这样的基准具有挑战性，因为可复现的系统级性能修复非常稀少且难以归因。我们通过将基于实证的性能反模式注入到真实的开源系统中来解决这一挑战……

    arXiv:2610.29695v1 Announce Type: new  Abstract: Coding agents are becoming an increasingly important paradigm for automating software engineering tasks. However, existing coding benchmarks primarily evaluate agents at the function or repository level, leaving their capabilities in more complex distributed software systems largely unexplored. In practice, many modern applications adopt microservice architectures, where end-to-end performance emerges from interactions among multiple independently running services, requiring agents to reason beyond local code changes. To study this capability, we introduce MicroPerf-Bench, a benchmark for evaluating coding agents on end-to-end performance optimization in microservice systems. Constructing such a benchmark is challenging because reproducible system-level performance fixes are sparse and difficult to attribute. We address this challenge by injecting empirically grounded performance anti-patterns into real-world open-source systems while pr
    
[^17]: MEDiml参考手册

    Reference Manual of MEDiml

    [https://arxiv.org/abs/2610.29666](https://arxiv.org/abs/2610.29666)

    MEDiml是一个开源且符合标准的影像组学平台，通过结合Python包和图形桌面应用程序，以互补的图形与编程接口打通特征提取、预测建模和结果解释的端到端流程，降低了临床研究人员使用影像组学的门槛。

    

    影像组学是从医学图像中高通量提取定量特征（例如肿瘤形状、强度分布和纹理），以表征组织并发现人眼无法察觉的疾病模式。这些特征可以作为癌症诊断、预后和分类的候选影像生物标志物。然而，提取数千个特征使得针对特定临床任务识别最具信息量的特征变得困难，而构建端到端的工作流程又往往需要编程专业知识，这限制了临床研究人员的使用可及性。因此，我们提出了MEDiml，一个开源、符合标准的影像组学平台，通过互补的图形界面和编程接口将特征提取、预测建模和结果解释连接起来。MEDiml将Python包（MEDiml-pkg）与图形桌面应用程序（MEDiml-app）相结合，使研究人员能够开展实验……

    arXiv:2610.29666v1 Announce Type: new  Abstract: Radiomics is the high-throughput extraction of quantitative features (e.g., tumor shape, intensity distributions, and texture) from medical images to characterize tissues and uncover disease patterns invisible to the human eye. These features can serve as candidate imaging biomarkers for cancer diagnosis, prognosis, and classification. However, extracting thousands of features makes it difficult to identify the most informative ones for a given clinical task, while implementing an end-to-end workflow can require programming expertise that limits accessibility for clinical researchers. Hence, we propose MEDiml, an open-source, standards-compliant radiomics platform connecting feature extraction, predictive modeling, and interpretation through complementary graphical and programmatic interfaces. MEDiml combines a Python package (MEDiml-pkg) with a graphical desktop application (MEDiml-app) that enables researchers to conduct experiments wi
    
[^18]: 谁来验证智能体的成功声明？GUI智能体完成度判定中的自我报告泄漏

    Who Verifies the Agent's Success Claim? Self-Report Leakage in GUI-Agent Completion Judging

    [https://arxiv.org/abs/2610.29609](https://arxiv.org/abs/2610.29609)

    论文揭示了GUI智能体完成度判定中的“自我报告泄漏”现象——智能体自身的成功声明会误导裁判的判定——并证明通过隔离自我报告、仅依据执行轨迹进行审计，可大幅提升危险误报的检出率（最高提升43.4个百分点）。

    

    在自动化流水线中，GUI智能体充满信心的成功报告可能导致下游阶段继续推进，即使记录的执行轨迹并未显示智能体在运行中确实产生了所请求的结果。我们将这种情况称为“危险误报”：一种毫无保留的成功声明，而轨迹并不支持其作为运行过程中的因果性成果。我们研究“报告-轨迹审计”，即对照独立记录的轨迹来核查智能体的完成声明。一项受控的知晓/盲测干预揭示了“自我报告泄漏”现象：智能体自身的自我报告会改变完成度的判定结果。同一位裁判读取相同的任务和轨迹，唯一的区别是是否提供该自我报告。隐去自我报告后，仅基于轨迹的危险误报召回率在使用Qwen3.5-35B-A3B时提升了43.4个百分点，在使用DeepSeek-284B-A13B时提升了13.3个百分点，并且在留出的Batch 2上呈现相同趋势。这促使我们提出“报告隔离”机制。该报告记录智能体所声明的内容……

    arXiv:2610.29609v1 Announce Type: new  Abstract: In automation pipelines, a GUI agent's confident success report can let downstream stages proceed even when the recorded execution trace does not show that the agent produced the requested outcome. We call this case a dangerous misreport: an unhedged success claim that the trace does not support as causal accomplishment during the run. We study report--trace auditing, which checks an agent's completion claim against an independently recorded trace.   A controlled aware/blind intervention reveals self-report leakage: the agent's own self-report changes how completion is judged. The same judge reads the same task and trace; only that self-report is included or withheld. Withholding it improves trace-only dangerous-misreport recall by 43.4 percentage points with Qwen3.5-35B-A3B and 13.3 percentage points with DeepSeek-284B-A13B, with the same trend on held-out Batch 2. This motivates report isolation. The report records what the agent claim
    
[^19]: 快照测试有效性研究：一项实证研究

    On the Effectiveness of Snapshot Testing: An Empirical Study

    [https://arxiv.org/abs/2610.29549](https://arxiv.org/abs/2610.29549)

    本研究对340个GitHub仓库的大规模实证分析及60位从业者的调查表明，快照测试与传统的非快照测试高度互补，两者结合可将语句覆盖率中位数提升至86.7%，显著增强测试有效性。

    

    软件测试对于保证软件质量至关重要，然而测试复杂的前端用户界面往往耗时且困难。快照测试作为一种低成本的回归测试方法应运而生，它通过比较程序在修改前后的行为来检测意外变化。尽管快照测试的采用日益广泛，但关于其实际有效性的实证证据仍然稀缺。为填补这一空白，本研究对快照测试的有效性开展了大规模实证调查，涉及340个GitHub仓库以及对60位经验丰富从业者的问卷调查。我们的定量结果表明，将快照测试与传统的非快照测试相结合可显著提高代码覆盖率，语句覆盖率中位数达到86.7%。我们发现这两种方法高度互补，通常覆盖生产代码的不同区域。此外，包含快照测试的测试套件（摘要在此处截断）

    arXiv:2610.29549v1 Announce Type: new  Abstract: Software testing is essential for ensuring quality, yet testing complex front-end user interfaces is often time-consuming and difficult. Snapshot testing has emerged as a low-cost regression testing method that detects unexpected changes by comparing program behavior before and after modifications. Despite its growing adoption, empirical evidence regarding its actual effectiveness remains scarce. To fill this gap, this study presents a large-scale empirical investigation into the effectiveness of snapshot testing involving 340 GitHub repositories and a survey of 60 experienced practitioners.   Our quantitative results demonstrate that combining snapshot and traditional non-snapshot tests significantly enhances code coverage, achieving a median statement coverage of 86.7%. We find that the two approaches are highly complementary, typically exercising distinct regions of the production code. Furthermore, test suites incorporating snapshot 
    
[^20]: RegoMender：用于自动Rego策略修复的多智能体框架

    RegoMender: A Multi-agent Framework for Automated Rego Policy Repair

    [https://arxiv.org/abs/2610.29456](https://arxiv.org/abs/2610.29456)

    本文提出了首个Rego策略修复基准RegoBench（含1,159个真实缺陷）和反馈引导的多智能体框架RegoMender，通过共享黑板与原子工具包协作动态执行打补丁、重写或回滚，实现了82.9%的语法修复成功率和77.9%的语义修复成功率。

    

    策略即代码已成为云原生系统中的一种基础范式。然而，由于Rego策略严格的语法、声明式语义以及特定的验证要求，修复Rego策略仍然充满挑战。现有的自动程序修复（APR）方法主要针对主流编程语言设计，难以适应Rego独特的策略逻辑和修复工作流程。因此，Rego策略修复目前仍是一个依赖试错的手动过程。为解决这一问题，我们提出了RegoBench——首个自动化Rego策略基准，包含1,159个真实世界缺陷，并提出RegoMender，一个反馈引导的多智能体框架。RegoMender通过共享黑板机制和原子化工具包进行协作，动态执行打补丁、重写或回滚等操作，以应对复杂的修复场景。在RegoBench上的评估结果表明，RegoMender实现了82.9%的语法成功率和77.9%的基于测试的语义成功率。

    arXiv:2610.29456v1 Announce Type: new  Abstract: Policy-as-Code (PaC) has emerged as a foundational paradigm in cloud-native systems. However, repairing Rego policies remains challenging due to their strict syntax, declarative semantics, and specific verification requirements. Existing automated program repair (APR) methods are mainly designed for mainstream languages and struggle to accommodate the unique policy logic and repair workflows of Rego. Thus, Rego policy repair remains a manual process driven by trial and error. To address this, we introduce RegoBench, the first automated Rego policy benchmark comprising 1,159 real-world defects, and propose RegoMender, a feedback-guided multi-agent framework. Collaborating via a shared blackboard and an atomic toolkit, RegoMender dynamically executes patching, rewriting, or rolling back, to resolve complex repair scenarios. Evaluated on RegoBench, RegoMender achieves syntactic and test-based semantic success rates of 82.9\% and 77.9\%, res
    
[^21]: CertMoE：通过路由感知平滑实现的混合专家大语言模型鲁棒性认证

    CertMoE: Robustness Certification for Mixture-of-Experts LLMs via Routing-Aware Smoothing

    [https://arxiv.org/abs/2610.29395](https://arxiv.org/abs/2610.29395)

    该论文提出了CertMoE，首个针对混合专家大语言模型的路由感知鲁棒性认证框架，通过显式建模输入依赖的动态路由，解决了现有认证方法因假设固定计算图而失效的问题。

    

    可证明鲁棒性是可信机器学习的基石，旨在保证模型在对抗扰动下的预测。现有的认证框架受抽象解释或符号执行启发，通常假设固定的计算图，即输入扰动或变化只影响模型输出，而不影响底层执行路径。然而，这一假设在混合专家大语言模型中并不成立，因为其依赖输入的路由机制会产生离散且动态的执行路径。至关重要的是，这种动态计算将认证对象从单一固定函数变成了依赖输入的子函数的组合，导致认证保证不可靠。本文提出了CertMoE，这是首个针对混合专家大语言模型的路由感知鲁棒性认证框架，它在认证过程中显式地考虑了由输入引发的路由动态。

    arXiv:2610.29395v1 Announce Type: cross  Abstract: Provable robustness is a cornerstone of trustworthy machine learning, aiming to guarantee model predictions under adversarial perturbations. Existing certification frameworks, inspired by abstract interpretation or symbolic execution, typically assume a fixed computation graph, where input perturbations or variations affect only model outputs but not the underlying execution path. This assumption, however, is violated in Mixture-of-Experts (MoE) Large Language Models (LLMs), whose input-dependent routing induces discrete and dynamic execution paths. Crucially, this dynamic computation changes the certified object from a single fixed function to a composition of input-dependent sub-functions, leading to unreliable certification guarantees.   This paper presents \textsc{CertMoE}, the first routing-aware robustness certification framework for MoE LLMs, which explicitly accounts for input-induced routing dynamics in certification. Specific
    
[^22]: 编码智能体理解系统为什么崩溃——以及为什么不崩溃吗？

    Do Coding Agents Understand Why Systems Crash---and Why They Don't?

    [https://arxiv.org/abs/2610.29332](https://arxiv.org/abs/2610.29332)

    本文提出“源代码干预”方法和包含934对代码树的SysRB数据集，通过配对仅相差一次提交的源代码树来严格测试编码智能体的崩溃推理能力，发现智能体严重倾向于预测“不崩溃”，从而漏掉了大量真实的系统崩溃。

    

    编码智能体修复真实世界漏洞的能力日益增强，但它们在处理像Linux内核这样的大型系统代码库时仍然表现不佳。为了理解其中的原因，我们通过崩溃推理来测试智能体在这些代码库中的执行推理能力：即根据源代码预测给定输入是否会导致系统崩溃。然而，仅有正确的预测并不足以证明这种能力，因为预测可能仅仅来自于发现可疑代码。在本文中，我们提出了源代码干预方法：我们配对仅相差一次提交的源代码树，使得相同的输入只会使其中一棵代码树崩溃，并测试智能体是否更频繁地在崩溃的代码树上预测出崩溃。我们在SysRB——一个包含来自Linux和OSS-Fuzz项目的934对代码树的数据集——上对四个编码智能体进行了实验，得到了三个主要发现。首先，智能体倾向于预测不会崩溃：它们很少误报，但漏掉了许多真实的崩溃，即使是表现最好的智能体也漏掉了超过……

    arXiv:2610.29332v1 Announce Type: new  Abstract: Coding agents are increasingly capable of fixing real-world bugs, yet they still struggle with large systems codebases such as the Linux kernel. To understand why, we test how agents reason about execution in these codebases through crash reasoning: predicting from source code whether a given input crashes the system. A correct prediction alone, however, does not establish this ability, since it may come from merely spotting suspicious code. In this paper, we propose source interventions: we pair source trees that differ by a single commit, so that the same input crashes only one of them, and test whether agents predict crashes more often on the crashing tree. Experiments with four coding agents on SysRB, our dataset of 934 such pairs from Linux and OSS-Fuzz projects, yield three main findings. First, agents are biased toward predicting no crash: they rarely raise false alarms but miss many real crashes, and even the best agent misses ov
    
[^23]: 开源软件中的动机与贡献：一项元分析研究

    Motivation and Contribution in Open Source Software: A Meta-Analytic Investigation

    [https://arxiv.org/abs/2610.29199](https://arxiv.org/abs/2610.29199)

    该元分析整合41项研究发现，八种动机前因均与持续贡献意愿正相关、其中七种与实际贡献努力正相关，并推翻了“内在动机比外在动机更能驱动实际贡献”的常见假设。

    

    关于人们为何为开源软件（OSS）及相关开放协作平台做贡献的研究已持续二十余年，但其研究结论相互矛盾，且以往研究常将人们的实际贡献行为与其继续贡献的意愿视为可互换的概念。基于自我决定理论和动机挤出理论，我们对41项研究进行了元分析。主分析汇总了来自29个独立样本（N = 9,632）的178个相关系数，考察八种前因变量与贡献努力及持续意愿之间的关系。结果显示：七种前因变量与贡献努力呈正相关，全部八种前因变量与持续意愿呈正相关，且每种前因变量与意愿的相关性至少不弱于与努力的相关性。与一个普遍假设相反，内在动机与贡献努力的相关性并不比外在动机更强，但内在动机在意愿与努力之间的落差更大。声誉主要在具备声誉系统的平台上与贡献努力相关，且我们没有发现迹象表明……

    arXiv:2610.29199v1 Announce Type: cross  Abstract: Research on why people contribute to open source software (OSS) and related open collaboration platforms spans more than two decades, but its findings conflict, and studies often treat the work people contribute and their intention to keep contributing as interchangeable. Drawing on self-determination theory and motivation crowding theory, we meta-analyze 41 studies. The main analysis pools 178 correlations of eight antecedents with contribution effort and continuance intention from 29 independent samples (N = 9,632). Seven antecedents relate positively to effort, all eight relate positively to intention, and each relates at least as strongly to intention as to effort. Contrary to a common assumption, intrinsic motivation is not more strongly related to effort than extrinsic motivation, but its gap between intention and effort is larger. Reputation relates to effort mainly on platforms with a reputation system, and we find no sign that
    
[^24]: CloudIR-Bench：一个用于云事件根因分析与缓解的基准测试

    CloudIR-Bench: A Benchmark for Root Cause Analysis and Mitigation of Cloud Incidents

    [https://arxiv.org/abs/2610.29185](https://arxiv.org/abs/2610.29185)

    本文提出了首个公开可用的基准测试 CloudIR-Bench，为基于事件报告的云事件根因分析与缓解措施预测提供数据集和评估框架，解决了以往依赖私有数据导致的可复现性和泛化性受限的问题。

    

    根因分析（RCA）和缓解措施预测是事件管理的关键环节，能够识别故障的深层原因并确保复杂云系统的可靠性。随着事件数量和系统复杂性的不断上升，事件报告已成为完成这些任务的宝贵资源。最近的研究表明，使用大语言模型（LLM）从事件报告的文本描述中预测根因和缓解措施已取得了可喜的成果。尽管取得了这些进展，由于缺乏公开可用的数据集，进一步的进展受到阻碍；现有工作依赖于私有数据，这限制了透明度、可复现性和泛化能力。为填补这一空白，我们提出了 CloudIR-Bench，这是首个公开可用的基准测试，为基于事件报告的根因分析和缓解措施预测提供了数据集和评估框架。该数据集精选自五大领先云服务提供商的事件报告。

    arXiv:2610.29185v1 Announce Type: new  Abstract: Root cause analysis (RCA) and mitigation prediction are essential for incident management, enabling the identification of underlying causes of failures and ensuring the reliability of complex cloud-based systems. With rising incident volume and system complexity, incident reports have become a valuable resource for these tasks. Recent studies have shown promising results using Large Language Models (LLMs) to predict root causes and mitigation actions from textual descriptions in incident reports. Despite these advances, further progress is hindered by the lack of publicly available datasets; existing work relies on private data, which limits transparency, reproducibility, and generalization. To address this gap, we introduce CloudIR-Bench, the first publicly available benchmark providing a dataset and evaluation framework for incident report-based RCA and mitigation prediction. Curated from incident reports across five leading cloud-base
    
[^25]: 颠覆量子编译器配置选择：一种自顶向下的范式

    Turning Quantum Compiler Configuration Selection Upside Down: A Top-Down Paradigm

    [https://arxiv.org/abs/2610.29170](https://arxiv.org/abs/2610.29170)

    该论文提出了一种自顶向下的量子编译器配置选择新范式，利用量子编译器配置空间可控、可穷举搜索的特点，颠覆了传统“先选设备再调编译器”的自底向上方法，克服了初始设备决策对电路执行保真度的根本性限制。

    

    选择有效的量子编译器配置需要对硬件目标和编译器设置进行协调决策，其最终目标是提高电路执行保真度。现有SOTA方法通常将此问题分解为两个顺序阶段：首先选择一个设备，然后在该设备上通过基于强化学习的搜索来调整编译器选项。这种自底向上的策略可能从根本上受限于其初始设备决策，因为不匹配的后端可能会限制可达到的最佳保真度，无论后续编译器选项优化得多么有效。与经典编译器调优（其搜索空间往往大得令人望而却步）不同，量子编译器的配置空间仍然足够可控，可以支持穷举搜索。因此，与其反复从候选配置中进行搜索，不如从离线（数据中）推导出可复用的配置知识……（摘要被截断）

    arXiv:2610.29170v1 Announce Type: cross  Abstract: Selecting an effective quantum compiler configuration requires coordinated decisions over both hardware targets and compiler settings, with the ultimate goal of improving circuit execution fidelity. Existing SOTA approaches typically decompose this problem into two sequential stages: a device is chosen first, after which compiler options are tuned on that device through RL-based search. This bottom-up strategy can be fundamentally limited by its initial device decision, since a poorly matched backend may constrain the best fidelity achievable regardless of how effectively later compiler options are optimized. In contrast to classical compiler tuning, where the search space is often prohibitively large, the configuration space of quantum compilers remains manageable enough to support exhaustive search. Consequently, rather than repeatedly searching from candidate configurations, one can derive reusable configuration knowledge from offli
    
[^26]: 基于人工智能的软件定义汽车端到端设计与软件开发

    AI-Based End-to-End Automotive Design and Software Development for SDVs

    [https://arxiv.org/abs/2610.29160](https://arxiv.org/abs/2610.29160)

    本文提出了一种将大语言模型与模型驱动工程相结合的AI驱动工具链，能够将自然语言需求自动转化为软件定义汽车的系统架构、可执行软件和验证工件，并在HiL/SiL环境中通过自适应巡航控制和紧急制动功能验证了其符合汽车安全关键开发要求。

    

    本文探讨了人工智能（AI）在实现软件定义汽车（SDV）端到端汽车设计与软件开发方面的潜力。基于我们在中央车载服务器（CeCaS）项目及后续欧盟资助项目中的贡献，我们提出了一种AI驱动的工具链，将大语言模型（LLM）与模型驱动工程（MDE）相结合，以自动化方式将自然语言需求转化为系统架构、可执行软件和验证工件。该方法强调可追溯性、一致性，以及符合安全关键型汽车开发的要求。验证工作在支持自适应巡航控制和紧急制动的集中式车载计算平台上进行，并在硬件在环（HiL）和软件在环（SiL）环境中完成测试。我们讨论了当前的能力、实际局限性，以及朝着日益自动化方向发展的未来路径。

    arXiv:2610.29160v1 Announce Type: new  Abstract: This paper explores the potential of Artificial Intelligence (AI) to enable end-to-end automotive design and software development for Software-Defined Vehicles (SDVs). Based on our contributions within the Central Car Server (CeCaS) and follow-up EU-funded projects, we present an AI-driven toolchain integrating Large Language Models (LLMs) with Model-Driven Engineering (MDE) to automate the transformation of natural-language requirements into system architectures, executable software, and validation artifacts. The approach emphasizes traceability, consistency, and compliance with safety-critical automotive development requirements. Validation is performed on a centralized vehicle computing platform supporting adaptive cruise control and emergency braking, tested in Hardware-in-the-Loop (HiL) and Software-in-the-Loop (SiL) environments. We discuss current capabilities, practical limitations, and future directions toward increasingly autom
    
[^27]: VERDICT：多智能体并行模块化验证

    VERDICT: Multi-Agent Parallel Modular Verification

    [https://arxiv.org/abs/2610.29147](https://arxiv.org/abs/2610.29147)

    VERDICT是一个多智能体框架，通过基于假设的并行规范综合打破传统自底向上验证的依赖顺序，实现大规模程序模块化演绎验证的并行化。

    

    演绎验证通过模块化推理扩展到大型软件系统，其中形式化契约总结单个函数的行为，使其能够独立验证。其核心瓶颈在于规范综合：即构建模块化验证所需的契约。大语言模型（LLM）的最新进展使自动化规范综合达到了前所未有的规模。然而，现有方法在根本上仍是自底向上的，要求被调用函数必须先于其调用者完成验证。这种依赖关系限制了并行性，且缺乏一种有原则的机制来解决沿调用链传播的过程间不一致性问题。我们提出了VERDICT，一个用于并行化大规模程序模块化演绎验证的多智能体框架。VERDICT建立在两个关键原则之上。首先，基于假设的并行规范综合为每个函数分配一个智能体，并……

    arXiv:2610.29147v1 Announce Type: new  Abstract: Deductive verification scales to large software systems through modular reasoning, where formal contracts summarize the behavior of individual functions and enable their independent verification. Its central bottleneck is specification synthesis: constructing the contracts required for modular verification. Recent advances in large language models (LLMs) have enabled automated specification synthesis at unprecedented scale. However, existing approaches remain fundamentally bottom-up, requiring callees to be verified before their callers. This dependency limits parallelism and provides no principled mechanism for resolving interprocedural inconsistencies that propagate across call chains.   We present VERDICT, a multi-agent framework for parallelizing modular deductive verification of large-scale programs. VERDICT is built on two key principles. First, assumption-based parallel specification synthesis assigns an agent to each function and
    
[^28]: ORSMT：在基于大语言模型的优化中释放可满足性模理论在语义验证与修复方面的潜力

    ORSMT: Unleashing the Potential of Satisfiability Modulo Theories for Semantic Verification and Repair in LLM-Based Optimization

    [https://arxiv.org/abs/2610.29145](https://arxiv.org/abs/2610.29145)

    ORSMT 利用可满足性模理论（SMT）对大语言模型生成的优化产物（数学模型、求解器代码等）进行精确的语义关系验证，并通过产物间的重构实现自动修复，显著提升了模型与代码的语义匹配率。

    

    大语言模型正越来越多地实现从自然语言问题到优化建模的自动化，产出书面的数学模型、求解器代码以及返回的报告。跨这些产物验证同一个优化问题需要语义层面的比较：不相等的模型可能达到相同的目标值，等价的模型可能使用不同的辅助变量，而彼此一致的产物可能共同遗漏了源需求。ORSMT 提供了一个类型化的语义接口和精确的 SMT 关系查询。它对齐可观测的决策变量，投影辅助变量，证明或反驳可行域与目标之间的关系，将反驳定位到所涉及的行或定义域，并在可用时链接源文本。随后，它从另一个产物重构出目标产物，且每个被采纳的候选都必须通过所需的关系检查。在 3,034 个问题和六种生成器规模上，ORSMT 在全部 72 种基准与生成器组合中都提升了模型与代码的匹配率。

    arXiv:2610.29145v1 Announce Type: cross  Abstract: Large language models increasingly automate optimization modeling from natural language problems, producing written mathematics, solver code and a returned report. Verifying one optimization problem across these artifacts requires semantic comparison: unequal models can attain the same objective, equivalent models can use different auxiliary variables, and mutually consistent artifacts can omit the same source requirement. ORSMT provides a typed semantic interface and exact SMT relation queries. It aligns observable decisions, projects auxiliary variables, proves or refutes feasible region and objective relations, localizes refutations to implicated rows or domains and links source texts when available. It then reconstructs one artifact from another, and each adopted candidate passes the required relation checks. Across 3,034 problems and six generator sizes, ORSMT improves the model-code match rate in all 72 benchmark and generator co
    
[^29]: VeeReview：从V模型视角测量智能体创作拉取请求中的构建-验证对

    VeeReview: Measuring Build-Verify Pairs in Agent-Authored Pull Requests from a V-Model Perspective

    [https://arxiv.org/abs/2610.29143](https://arxiv.org/abs/2610.29143)

    本文提出基于V模型构建-验证对的VeeReview方法，通过仓库信号和语言模型裁判测量拉取请求中的验证证据，发现智能体创作的拉取请求在全部四个验证维度上的“仅构建”率均低于人类拉取请求，与普遍预期相反。

    

    背景：审查拉取请求需要针对所提议变更的验证证据，而编程智能体基准测试和拉取请求结果研究并未直接评估这种关系。目标：我们检验V模型的构建与验证对能否测量拉取请求中的验证证据，并支持智能体创作与人类创作变更之间的比较。方法：VeeReview使用仓库信号和语言模型裁判对四对构建-验证对进行分类，其中三对使用受确定性信号约束的模型判断，另一对则通过基于差异和仓库树的规则来判定“模块设计-单元测试”对。我们在由两名评估者标注的25个拉取请求参考集上评估了VeeReview，并将其应用于来自741个仓库的1,454对匹配的智能体与人类拉取请求。结果：VeeReview在所有四对构建-验证对上为智能体创作的拉取请求分配了更低的“仅构建”率，这与……（摘要在此处截断）

    arXiv:2610.29143v1 Announce Type: cross  Abstract: Context: Reviewing a pull request requires verification evidence that addresses the proposed change. Coding agent benchmarks and studies of pull request outcomes do not directly assess this relationship. Objective: We test whether the V-Model's build and verify pairs can measure verification evidence in pull requests and support comparisons between agent-authored and human-authored changes. Method: VeeReview classifies four build and verify pairs using repository signals and a language-model judge. Three pairs use model judgements constrained by deterministic signals, and a rule over the diff and repository tree decides Module Design-Unit Test. We evaluated VeeReview against a 25-pull request reference set labelled by two raters and applied it to 1,454 matched pairs of agent and human pull requests from 741 repositories. Results: VeeReview assigned lower build-only rates to agent-authored pull requests on all four pairs, contrary to th
    
[^30]: 面向对称性的变分量子算法测试框架

    Symmetry-Oriented Testing Framework for Variational Quantum Algorithms

    [https://arxiv.org/abs/2610.29053](https://arxiv.org/abs/2610.29053)

    提出了一种面向对称性的变分量子算法测试框架，以与目标哈密顿量对易的算符作为测试规范，解决了现有方法无法验证量子程序执行是否偏离预期问题、所求解是否物理合理的难题。

    

    摘要：测试量子软件极具挑战性，因为量子态无法被直接检查。变分量子算法（VQAs）是尤为重要的测试对象，因为它们是面向近期含噪量子硬件最有前景的方法之一。许多测试方法依赖于经典计算得到的结果，这与预期实现量子优势的场景相冲突。尽管先前的研究缓解了这一预言机问题，但核心困难依然存在：现有方法无法验证程序执行是否偏离了预期问题，也无法验证所获得的解在物理上是否合理。我们提出了一个面向对称性的变分量子算法测试框架，通过使用与目标哈密顿量对易的算符作为测试规范来填补这一空白。该框架提供了两种互补的诊断手段：一种基于对易子的健康检查，用于监控优化过程中每个执行步骤是否保持一致（摘要原文在此处截断）

    arXiv:2610.29053v1 Announce Type: new  Abstract: Testing quantum software is challenging because quantum states cannot be directly inspected. Variational quantum algorithms (VQAs) are a particularly important target as they are among the most promising approaches for near-term noisy quantum hardware. Many testing methods rely on classically computed results, which conflicts with the regime where quantum advantage is expected. Although prior studies mitigate this oracle problem, the core difficulty remains that existing approaches do not verify whether the execution deviates from the intended problem or whether the obtained solution is physically plausible.   We propose a symmetry-oriented testing framework for VQAs that addresses this gap by using operators commuting with the target Hamiltonian as test specifications. The framework provides two complementary diagnostics: a commutator-based health check that monitors whether each execution step during optimization remains consistent wit
    
[^31]: VeriEvolve：面向工具使用智能体的“评估器优先”进化框架

    VeriEvolve: Evaluator-First Evolution for Tool-Using Agents

    [https://arxiv.org/abs/2610.29005](https://arxiv.org/abs/2610.29005)

    VeriEvolve提出“评估器优先”的进化框架，先从任务指令自动合成可执行的评估器和挑战状态作为可靠的适应度信号，再据此进化工具使用智能体，将MCPMark文件系统任务的外部验证成功率从6/30提升至18/30。

    

    自我改进智能体面临的一个核心障碍是：对于现实中有状态的任务，缺乏可靠的适应度信号。我们提出了VeriEvolve，这是一个“评估器优先”的进化框架，它在优化智能体之前先构建这一缺失的评估层。VeriEvolve从任务指令出发，合成基于指令的挑战状态以及一个可执行的评估器来检验这些状态。在每轮工作流进化过程中，评估器被冻结并作为适应度信号；在两轮之间，VeriEvolve通过将强运行与弱运行的最终状态转换为一对反事实测试用例来修订评估器。在全部30个MCPMark文件系统任务上，VeriEvolve构建了30个包含141个挑战状态的评估器，在生成的测试套件上达到141/141的一致性，在执行智能体产生的保留状态上达到113/122的一致性。评估器引导的进化使外部验证的成功率从6/30提升到18/30。

    arXiv:2610.29005v1 Announce Type: cross  Abstract: A central obstacle to self-improving agents is the absence of a reliable fitness signal for realistic stateful tasks. We introduce \textsc{VeriEvolve}, an evaluator-first evolution framework that constructs this missing evaluation layer before optimizing the agent. From task instruction, VeriEvolve synthesizes instruction-grounded challenge states and an executable evaluator that checks them. During each round of workflow evolution, the evaluator is frozen and serves as the fitness signal. Between rounds, VeriEvolve revises the evaluator by converting the end states of strong and weak runs into a pair of counterfactual test cases.   Across all 30 MCPMark Filesystem tasks, VeriEvolve constructs 30 evaluators with 141 challenge states, achieving 141/141 agreement on the generated suites and 113/122 agreement on held-out states produced by executor agents. Evaluator-guided evolution increases externally verified success from 6/30 to 18/30
    
[^32]: 行为不止于文本：评估大语言模型智能体中的拒绝行为迁移失效

    Behavior Beyond Text: Evaluating Refusal Transfer Failure in LLM Agents

    [https://arxiv.org/abs/2610.28982](https://arxiv.org/abs/2610.28982)

    该论文揭示了大语言模型的文本拒绝安全行为无法可靠迁移到工具调用场景——在聊天中被拒绝的有害请求中有10%至52%会在工具启用时引发不安全的工具调用，并通过残差流中的拒绝方向对这一失效机制进行了分析。

    

    随着大语言模型越来越多地被部署为工具调用智能体，其安全评估仍然主要集中在它们在文本中拒绝有害请求的能力上。我们研究了文本中的拒绝行为是否会迁移到工具使用场景中。我们构建了一个涵盖四个领域、共2,304个提示词的基准测试，由匹配的三元组组成，这些三元组在纯文本、标准工具启用和对抗性框架的工具启用三种设置下呈现相同的有害或良性请求。随后，我们使用20个预定义的禁止操作来评估工具调用的安全性。通过测试五个不同规模和系列的模型，我们发现文本中被拒绝的请求经常导致不安全的工具调用。在标准工具启用的提示下，这种分歧发生在聊天中被拒绝的有害请求的10%至52%。为了从机制层面研究这一差距，我们在五个模型各自的残差流中识别出了一个与拒绝行为相关的方向。我们发现……

    arXiv:2610.28982v1 Announce Type: cross  Abstract: While large language models are increasingly deployed as tool-calling agents, their safety evaluations still focus primarily on their ability to refuse harmful requests in text. We investigate whether the refusal behavior in text transfers to tool use. We construct a benchmark of 2,304 prompts across four domains, consisting of matched triplets that present the same harmful or benign request in text-only, standard tool-enabled, and adversarially framed tool-enabled settings. We then assess tool-call safety using 20 predefined forbidden actions. Testing five models of varying sizes and families, we find that requests refused in text frequently result in unsafe tool calls. Under standard tool-enabled prompting, this divergence occurs for 10-52 percent of harmful requests refused in chat. To mechanistically investigate this gap, we identify a refusal-associated direction in the residual stream of each of the five models. We find that in t
    
[^33]: 先备份，免后悔：测量编码智能体的备份意识并实现备份自动化

    Back Up Before You Regret: Measuring Backup Awareness and Automating Backup in Coding Agents

    [https://arxiv.org/abs/2610.28934](https://arxiv.org/abs/2610.28934)

    该论文提出 RegretBench 基准，揭示编码智能体虽平均完成 86.1% 的任务、却仅在 15.2% 的“后悔时刻”前主动备份，并据此推出 BackUpper 插件，在每次修改状态的工具调用之前自动保存受影响的状态，从而实现备份自动化。

    

    编码智能体现在能够自主完成任务，并在此过程中修改工件、软件环境与配置等真实世界的状态。然而，它们表现出有限的备份意识——即识别某项操作将修改日后可能需要的状态并提前予以保存的能力。为了评估备份意识，我们提出了 RegretBench，一个包含 205 个可执行任务的基准，这些任务源自 X.com、GitHub 和 Stack Overflow 等平台上用户明确表示后悔未保存后续所需状态的“后悔时刻”。在六个前沿模型上，智能体平均完成 86.1% 的任务，但在这些时刻之前进行备份的比例仅为 15.2%，这表明任务完成能力并不意味着具备备份意识。由于难以预测哪些更改日后会被后悔，我们提出了 BackUpper——一个以测试框架插件形式存在的备份层，它检查每一次工具调用，并在修改之前保存受影响的状态。它结合了基于规则的……（原文摘要在此处被截断）

    arXiv:2610.28934v1 Announce Type: cross  Abstract: Coding agents now complete tasks autonomously, modifying real-world state such as artifacts, software environments, and configurations along the way. Yet they show limited backup awareness: recognizing that an action will modify state that may later be needed, and preserving it beforehand. To evaluate backup awareness, we introduce RegretBench, a benchmark of 205 executable tasks grounded in regret moments reported on platforms such as X.com, GitHub, and Stack Overflow, where users explicitly regretted not preserving state they later needed. Across six frontier models, agents complete 86.1% of tasks on average yet back up before these moments in only 15.2%, showing that task completion ability does not imply backup awareness. Because it is hard to predict which changes will be regretted, we propose BackUpper, a harness plug-in backup layer that checks every tool call and preserves affected state before modification. It combines rule-ba
    
[^34]: SANE：面向LLM评判器的自动化的评分器特定变异测试

    SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges

    [https://arxiv.org/abs/2610.28886](https://arxiv.org/abs/2610.28886)

    SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。

    

    LLM评判器（LLM judges）在基准评估、模型选择、数据筛选和模型训练中扮演着越来越重要的角色。只有当每个评判器能够对影响其评分标准的变动做出可靠响应时，其决策才有价值。现有的可靠性测试方法通过对输入施加具有已知预期评分影响的扰动来进行测试。主流的LLM评判器测试框架将人工编写的扰动配方与合成测试生成相结合，但将针对特定评分器的配方构思排除在自动化流程之外。这限制了它们只能使用固定的扰动方式和预先假设的评分器类型。我们提出了SANE，一个基于智能体（agent）的框架，它将随机评估器的行为需求编码为可复用的智能体指令。给定评分器规范、实现代码和代表性输入，该流水线首先识别评分器特定的能力，然后为每种能力自动推导扰动配方和预期的分数关系。SANE还定义了一个（摘要内容在此处被截断）

    arXiv:2610.28886v1 Announce Type: new  Abstract: LLM judges increasingly mediate benchmark evaluation, model selection, data curation, and training. Their decisions are useful only when each judge responds reliably to changes affecting its scoring criterion. Existing methods for reliability testing perturb inputs with known expected effects on judge scores. Prominent LLM-judge testing harnesses combine human-authored perturbation recipes with synthetic test generation, leaving scorer-specific recipe ideation outside the automated pipeline. This limits their use to fixed perturbations and pre-assumed scorer types. We introduce SANE, an agent-based framework that encodes behavioral requirements for stochastic evaluators as reusable agent instructions. Given a scorer specification, implementation, and representative inputs, the pipeline first identifies scorer-specific capabilities, then automatically derives perturbation recipes and expected score relations for each. SANE also defines a 
    
[^35]: 陷入困境的智能体应该把接下来的令牌花在哪里？推理时资源的因果归因

    Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources

    [https://arxiv.org/abs/2610.28834](https://arxiv.org/abs/2610.28834)

    提出班扎夫资源归因框架，通过受控因子实验因果地量化计算、信息、反馈等推理时资源对修复失败编码任务的边际贡献，发现任务相关规范最有价值、额外计算的价值随模型能力递减，而多数前沿模型却错误地将预算分配给收益甚微的反馈资源。

    

    推理时扩展通常被视为令牌预算问题，但一个失败的编码智能体可以寻求性质不同的资源：计算、信息或反馈。我们提出了班扎夫资源归因，这是一个受控的因子实验框架，在保持失败状态和修订协议固定的情况下，测量每种资源对修复的边际贡献及其交互作用。我们在MBPP+和HumanEval+的194个函数级失败以及SWE-bench Verified的126个仓库级失败上评估了六个模型。任务相关的规范始终能改善修复效果，而公开测试反馈仅带来小幅提升，检索到的代码几乎没有作用。额外计算的价值随模型能力的增强而下降，而信息仍然有用；资源通道大多表现出收益递减，尽管互补信息可以自我强化。然而，四个前沿模型中有三个选择了反馈……

    arXiv:2610.28834v1 Announce Type: cross  Abstract: Inference-time scaling is often treated as a token-budget problem, but a failed coding agent can seek qualitatively different resources: computation, information, or feedback. We introduce Banzhaf Resource Attribution (BRA), a controlled factorial framework that measures each resource's marginal contribution to repair and its interactions while holding the failed state and revision protocol fixed. We evaluate six models on 194 function-level failures from MBPP+ and HumanEval+ and 126 repository-level failures from SWE-bench Verified. Task-relevant specifications consistently improve repair, whereas public-test feedback provides small gains and retrieved code little to none. The value of additional computation declines with model capability, while information remains useful; resource channels largely exhibit diminishing returns, although complementary information can reinforce itself. Yet three of four frontier models select feedback in
    
[^36]: 测试框架有多重要，对谁重要？LLM驱动漏洞发现中上下文作用的多模型研究

    How Much Does the Harness Matter, and for Whom? A Multi-Model Study of Context in LLM-Driven Vulnerability Discovery

    [https://arxiv.org/abs/2610.28802](https://arxiv.org/abs/2610.28802)

    该研究通过24种模型-运行器配置的多模型实验发现，测试框架对漏洞发现的贡献高度依赖于所用模型——工程化上下文（如CodeQL调用图）能将中小型模型的关键性加权漏洞发现产出提升20-86%，但对前沿模型仅提升2%，且没有任何单一配置能覆盖全部漏洞。

    

    智能体测试框架对漏洞发现的影响有多大，其价值是否取决于模型？我们使用开源测试框架，在一个扫描时漏洞尚未公开的生产代码库上，研究了24种模型-运行器配置。该流水线结合了静态筛选、LLM调查、模型自验证、根因映射、独立源码审查以及隔离的概念验证确认。在265个已确认的安全根因中，规模最大的两次运行配置覆盖了60.4%，而14.3%的根因仅由单一模型发现，没有任何单一配置能够覆盖所有观测到的发现。在一项四模型上下文实验中，从最小提示词转向带有CodeQL衍生调用图的工程化提示词，使较小和中等规模的模型的关键性加权产出提升了20-86%，而前沿模型仅提升2%。该对比同时改变了提示词和上下文；将这两种效应分离后表明，工程化（上下文）……（原文摘要在此处截断）

    arXiv:2610.28802v1 Announce Type: cross  Abstract: How much does an agent harness affect vulnerability discovery, and does its value depend on the model? We study 24 model-runner configurations using an open-source harness on a production codebase whose vulnerabilities were non-public at scan time. The pipeline combines static screening, LLM investigation, model self-revalidation, root-cause mapping, independent source review, and isolated proof-of-concept confirmation.   Across 265 confirmed security root causes, the largest two-run configuration covers 60.4%, while 14.3% are found by exactly one model. No single configuration covers all observed findings. In a four-model context experiment, moving from a minimal prompt to an engineered prompt with a CodeQL-derived call graph raises criticality-weighted yield by 20-86% for smaller and mid-tier models, compared with 2% for the frontier model. This comparison changes both prompt and context. Separating those effects shows that the engin
    
[^37]: 在此签名：为更可信的软件供应链归因源头

    Sign Here: Attributing a Source for a More Trustworthy Software Supply Chain

    [https://arxiv.org/abs/2610.28794](https://arxiv.org/abs/2610.28794)

    本文提出通过直接从源代码构建软件包来取代软件包注册中心，并利用“恶意软件包通常不标注源代码来源、而良性软件包会链接其代码仓库”这一假设，作为防范软件供应链恶意软件攻击的策略，从而消除整个攻击面。

    

    软件供应链攻击持续上升，利用了帮助开源生态系统蓬勃发展的信任机制。近期诸如 Shai-Hulud 蠕虫和 Axios 入侵等事件表明，单个恶意软件包可能在一夜之间造成广泛而毁灭性的影响。这些备受瞩目的攻击都可以追溯到同一个罪魁祸首：软件包注册中心（package registries）。尽管从注册中心拉取预构建软件包是一种便利，但作为软件供应链中经常被利用的组件，它值得冒这个风险吗？在本文中，我们研究了通过直接从源代码构建软件包、将注册中心从开发者工作流程中移除，是否可以作为恶意软件防御策略——从而有效消除整个攻击面。我们的假设是：恶意软件包避免标注源代码来源以最小化其可见性，而良性软件包则寻求曝光度，因此会链接到其代码仓库。我们通过以下方式进行评估：

    arXiv:2610.28794v1 Announce Type: cross  Abstract: Software supply chain attacks continue to be on the rise, exploiting the trust that helps open source ecosystems thrive. Recent incidents like the Shai-Hulud worm and the Axios compromise demonstrate that a single malicious package can have wide reaching and devastating impacts overnight. These high profile attacks can be traced back to the same culprit: package registries. Although they are a useful convenience to pull prebuilt packages from, is such a commonly exploited component of the software supply chain worth the risk? In this paper, we investigate whether removing the registry from the developer workflow by building packages directly from source can serve as a malware prevention strategy - effectively removing an entire attack surface. Our hypothesis is that malicious packages avoid attributing a source to minimize visibility, while benign packages seek exposure and therefore link to their repositories. We evaluate this through
    
[^38]: 云-边-物联网连续体上的多智能体大语言模型：关于架构、部署与评估的系统性映射研究

    Multi-agent LLMs on the Cloud-Edge-IoT Continuum: A Systematic Mapping Study on Architectures, Deployment, and Evaluation

    [https://arxiv.org/abs/2610.28702](https://arxiv.org/abs/2610.28702)

    本文通过一项系统性映射研究，首次对多智能体大语言模型系统在云-边-物联网连续体上的架构、部署模式、应用领域与评估实践进行了系统分类与梳理，为该新兴交叉领域绘制了研究全景图。

    

    大语言模型（LLM）应用正越来越多地被构建为由协作组件组成的系统，而非单一的提示-响应流水线。物联网应用还带来了自身的约束，因为感知、推理、控制和执行分布在设备、边缘节点和云服务上，以满足延迟、隐私和带宽方面的要求。多智能体LLM系统如何在这一云-边-物联网连续体上进行组织和部署，此前尚未得到系统性梳理。因此，我们开展了一项系统性映射研究。从截至2026年3月检索到的831条记录中，我们筛选出45项主要研究，并按架构家族、部署模式、应用领域和评估实践对每项研究进行分类。我们还记录了每个系统的评估方式，以及其代码或数据是否公开。语料库中反复出现五个架构家族：专门的智能体团队部署在传感器旁运行，规划者将子任务委派给执行者，小型本地模型相互协作……

    arXiv:2610.28702v1 Announce Type: new  Abstract: LLM applications are increasingly built as systems of cooperating components rather than as single prompt-response pipelines. IoT applications add constraints of their own, because sensing, reasoning, control, and actuation are spread across devices, edge nodes, and cloud services to meet latency, privacy, and bandwidth requirements. How multi-agent LLM systems are organized and deployed across this cloud-edge-IoT continuum has not been mapped systematically. We therefore conducted a systematic mapping study. From 831 records retrieved up to March 2026, we selected 45 primary studies and classified each one by architecture family, deployment pattern, application domain, and evaluation practice. We also recorded how each system was evaluated and whether its code or data are public. Five architecture families recur in the corpus. Specialized agent teams run next to sensors, planners delegate subtasks to workers, and small local models coop
    
[^39]: PCLA-Bridge：一个用于CARLA智能体跨域部署的框架

    PCLA-Bridge: A Framework for Cross-Domain Deployment of CARLA Agents

    [https://arxiv.org/abs/2610.28696](https://arxiv.org/abs/2610.28696)

    PCLA-Bridge是一个无需修改代码即可将CARLA排行榜自动驾驶智能体跨域部署到其他模拟引擎、真实数据集和物理平台的工具，它通过处理依赖隔离、张量封送、路由和合成时序，使集成者只需提供最小接口即可完成复用。

    

    大多数高性能的端到端自动驾驶智能体（ADAs）都是为CARLA排行榜开发的，这导致它们与CARLA的API、传感器布局、坐标系和内部时钟紧密耦合。要在真实数据集、其他模拟器或物理平台上复用这些模型，需要进行大量的手动重新工程。我们提出了PCLA-Bridge，这是一个可部署的工具，能够在没有活跃CARLA服务器的情况下，运行未经修改的CARLA排行榜智能体，实现从任意传感器源到任意控制目标的转换。借助PCLA-Bridge处理依赖隔离、张量封送、路由和合成时序，集成者只需提供一个最小接口。我们在两种运行范式下对该工具进行了验证：一是在域内CARLA录制数据和真实世界nuScenes日志上进行离线回放，二是在三个不同的模拟引擎（MetaDrive、AWSIM和AlpaSim）上进行实时闭环控制。我们分析了每种平台所需的轻量级集成要求。

    arXiv:2610.28696v1 Announce Type: new  Abstract: Most high-performing end-to-end autonomous driving agents (ADAs) are developed for the CARLA Leaderboard, tightly coupling them to CARLA's APIs, sensor layouts, coordinate systems, and internal clock. Reusing these models on real datasets, alternative simulators, or physical platforms necessitates substantial manual re-engineering. We present PCLA-Bridge, a deployable tool that runs unmodified CARLA Leaderboard agents from any sensor source to any control destination without an active CARLA server. With PCLA-Bridge handling dependency isolation, tensor marshalling, routing, and synthetic timing, integrators only need to provide a minimal interface. We validate the tool across two operational paradigms: offline replay on in-domain CARLA recordings and real-world nuScenes logs, and live closed-loop control across three diverse simulation engines: MetaDrive, AWSIM, and AlpaSim. We analyze the lightweight integration requirements for each pl
    
[^40]: 谁批准的？一家金融科技初创公司中的编码智能体、人类账号与合并门禁证据

    Who Approved This? Coding Agents, Human Accounts, and Merge-Gate Evidence at a Fintech Startup

    [https://arxiv.org/abs/2610.28683](https://arxiv.org/abs/2610.28683)

    该论文通过分析一家金融科技初创公司十二周内786次合并请求的审计数据，揭示尽管设有非作者评审门禁，仍有27%的合并仅依赖编码智能体冒用技术负责人账号提交的批准，且平台记录无法证明任何合并的实际决策者，暴露了智能体时代合并门禁审计证据的失效。

    

    arXiv:2610.28683v1 公告类型：新。摘要：自10月3日起，每一个非机器人合并进入我们的主分支都必须通过一个非作者评审门禁，然而263次合并中有71次（27%）仅依赖于一个由编码智能体通过技术负责人的GitHub账号提交的批准，并且平台将其中每一次都记录为该负责人的普通批准。我们报告了在Bashi AI（一家正在准备SOC 2审计的金融科技初创公司）进行的为期十二周的智能体驱动开发：共786个已合并的拉取请求，其中657个通过显式标记或推断被关联到智能体，分析基于平台记录和保存下来的智能体会话转录。在启用门禁的阶段，负责人账号在同事的拉取请求上给出的每一次批准都是由智能体会话提交的，并且它发起了几乎所有的合并；记录无法显示其中任何一次是由谁决定的。我们为每一次合并重新计算了一个核心门禁属性：存在非作者批准且必需检查通过，两者都发生在合并之前，且都针对被合并的确切修订版本。

    arXiv:2610.28683v1 Announce Type: new  Abstract: From 3 October, every non-bot merge into our main branch passed a non-author review gate, yet 71 of those 263 merges (27%) rested only on an approval that a coding agent had submitted through the technical lead's GitHub account, and the platform recorded each one as an ordinary approval by the lead. We report twelve weeks of agent-driven development at Bashi AI, a fintech startup preparing for a SOC 2 audit: 786 merged pull requests, 657 of them associated with agents by explicit markers or by inference, analysed from platform records and saved agent-session transcripts. In the gated phase, an agent session submitted every approval the lead's account gave on a colleague's pull request, and issued nearly all of its merges; the records cannot show who decided any of them. We recompute, for every merge, a core gate property: a non-author approval and a passing required check, both before the merge and both on the exact revision merged. The 
    
[^41]: 面向演化动作字母表的运行时语义发现

    Runtime Semantic Discovery for Evolving Action Alphabets

    [https://arxiv.org/abs/2610.28682](https://arxiv.org/abs/2610.28682)

    本文提出一种基于执行轨迹中邻居上下文的轻量级运行时方法，通过定义k-邻居等价关系来识别演化系统中新动作是否与已知动作具有相同的操作角色，从而在不重建全局行为模型的情况下比较可观察字母表不同的系统。

    

    运行时验证通常假设可观察动作具有固定的字母表。这一假设在不断演化的软件系统中变得脆弱，因为更新、重新配置或架构变更可能会重命名、拆分或合并动作，同时保持系统的高层意图不变。在这些情况下，先前有效的监视器和规约可能与运行中的系统不再对齐。本文研究了运行时语义发现问题，即在不重建全局行为模型的情况下，识别新观察到的动作是否与先前已知的动作扮演相同的操作角色。为此，我们提出了一种局部的、可在运行时计算的动作相似性概念，该概念基于执行轨迹中动作出现时的邻居上下文。我们定义了一族k-邻居等价关系，并利用它们来比较可观察字母表不同的系统。我们的方法刻意保持轻量级：它并不……（原文摘要在此处截断）

    arXiv:2610.28682v1 Announce Type: new  Abstract: Runtime Verification typically assumes a fixed alphabet of observable actions. This assumption becomes fragile in evolving software systems, where updates, reconfigurations, or architectural changes may rename, split, or fuse actions while preserving the system's high-level intent. In these cases, previously valid monitors and specifications may become misaligned with the running system.   In this paper, we study runtime semantic discovery, namely the problem of identifying whether newly observed actions play the same operational role as previously known ones, without reconstructing a global behavioral model. To this end, we propose a local and runtime-computable notion of action similarity based on the neighbor contexts in which actions occur within execution traces. We define a family of k-neighbor equivalences and use them to compare systems whose observable alphabets differ.   Our approach is intentionally lightweight: it does not ai
    
[^42]: VerifierBench：面向大语言模型智能体的可执行验证器生成基准测试

    VerifierBench: Benchmarking the Generation of Executable Verifiers for LLM Agents

    [https://arxiv.org/abs/2610.28679](https://arxiv.org/abs/2610.28679)

    该论文提出了VerifierBench基准，用于评估大语言模型智能体生成可执行验证器的能力，发现现有模型生成的验证器虽然能可靠地拒绝无效解，但往往因过于严格而错误拒绝有效解。

    

    智能体需要可靠的方法来验证其输出，无论是在完成任务时，还是在过滤合成训练数据时。我们研究验证器生成问题：根据求解任务可用的指令和环境，智能体编写一个可执行的验证器，该验证器必须接受所有有效解并拒绝无效解。我们提出了VerifierBench，这是一个包含来自六个智能体基准的222个任务的基准测试，涵盖工具调用、基于状态和代码三类验证，配有4,993个隐藏测试用例，其中包括替代有效解和攻击样例；并通过六项检查的审计机制Verifier4Verifier（V4V）对其源验证器和测试套件进行筛查。在统一的智能体框架下对19个模型进行评估，归一化平衡准确率介于10.6至62.1之间，生成的验证器在拒绝无效解方面始终比接受有效解更为可靠。这种表面上的优势可能掩盖了过度严格的验证问题：在所分析的记录中，大多数错误拒绝…

    arXiv:2610.28679v1 Announce Type: new  Abstract: Agents need reliable ways to validate their outputs, both when completing tasks and when filtering synthetic training data. We study verifier generation: from the instruction and environment available for solving a task, an agent writes an executable verifier that must accept every valid solution and reject invalid ones. We introduce VerifierBench, a benchmark of 222 tasks from six agent benchmarks spanning tool-call, state-based, and code verification, with 4,993 hidden test cases that include alternative valid solutions and attacks; Verifier4Verifier (V4V), a six-check audit, screens its source verifiers and test suites. Across 19 models in a common agent harness, normalized balanced accuracy ranges from 10.6 to 62.1, with generated verifiers consistently rejecting invalid solutions more reliably than they accept valid ones. This apparent strength can mask overly restrictive verification: most false rejections in the analyzed records c
    
[^43]: 量子跃迁的基准测试：审计大语言模型量子代码生成的评估方法

    Benchmarking the Quantum Leap: Auditing LLM Evaluation for Quantum Code Generation

    [https://arxiv.org/abs/2610.28678](https://arxiv.org/abs/2610.28678)

    该研究系统性地基准测试了15个大型语言模型在量子代码生成任务上的表现，并首次揭示评估器的设计差异会极大扭曲评分结果——仅移除对未指定量的检查即可使模型pass@1从0%跃升至94.1%，说明量子代码生成评估亟需严格的审计与标准化。

    

    大型语言模型（LLM）能够生成量子程序，但由于各基准测试在任务构成、评分方式和失败处理上存在差异，可靠的评估仍然困难。我们在五个量子代码生成测试套件（共530个任务）上评估了15个托管的前沿模型、开放权重模型、量子专用模型和检索增强模型，并在我们新提出的QuantumBenchEval（QBE）基准（涵盖六个主题的96个任务）上评估了10个模型。我们报告了bootstrap置信区间、经过多重比较校正的配对比较，以及对输出有效性和评估器行为的审计。结果显示QiskitHumanEval-Standard已接近饱和（98.7% pass@1），而其Hard变体仍保留提升空间（63.8%）。经过多重比较校正后，主要测试套件中五项前二名模型的配对比较均不再显著。评估器的设计会显著改变分数：移除对未指定量的检查后，GPT-6 Astra在QBE上的pass@1从0%提升至94.1%（T1），从1.2%提升至92.9%

    arXiv:2610.28678v1 Announce Type: new  Abstract: Large language models (LLMs) can generate quantum programs, but reliable evaluation remains difficult because benchmarks differ in task composition, scoring, and failure handling. We evaluate 15 hosted frontier, open-weight, quantum-specialized, and retrieval-augmented models on five quantum code-generation suites (530 tasks), plus 10 models on our new QuantumBenchEval (QBE) benchmark (96 tasks across six topics). We report bootstrap confidence intervals, multiplicity-adjusted paired comparisons, and audits of output validity and evaluator behavior. QiskitHumanEval-Standard approaches saturation (98.7% pass@1), while its Hard variant retains headroom (63.8%). None of the five top-two comparisons across primary suites remains significant after multiple-comparison correction. Evaluator design substantially changes scores: removing checks for unspecified quantities raises GPT-6 Astra's QBE pass@1 from 0% to 94.1% (T1) and from 1.2% to 92.9%
    
[^44]: 编译不等于正确性：对LLM生成的PLC程序的逐扫描周期评估

    Compilation Is Not Correctness: Scan-by-Scan Evaluation of LLM-Generated PLC Programs

    [https://arxiv.org/abs/2610.28675](https://arxiv.org/abs/2610.28675)

    提出CycleBench基准测试，通过逐扫描周期比较LLM生成的PLC程序与参考程序的行为输出，揭示编译通过率与实际执行通过率之间存在11.92个百分点的差距，证明编译成功并不代表程序正确。

    

    可编程逻辑控制器（PLC）通过重复的扫描周期来自动化工业过程，每个扫描周期读取输入、执行控制逻辑并更新输出。工程师使用结构化文本（ST）编写这种控制逻辑，ST是IEC 61131-3标准中定义的一种高级文本语言。大型语言模型（LLM）现在可以根据自然语言需求生成ST代码，然而仅依赖编译成功或对特定属性、抽样扫描周期的检查，可能会使行为错误未被发现。我们提出了CycleBench，这是一个包含301个ST任务的基准测试，涵盖行为需求、公共接口、参考程序和可执行场景。其运行时环境在对齐的输入、初始化和逻辑时间条件下执行每个生成的程序及其参考程序，并在每个扫描周期后比较它们的输出。在八个LLM和基线生成方法的测试中，执行通过率比编译通过率低11.92个百分点；观察到的失败案例包括输出更新错误、算术……（摘要在此处被截断）

    arXiv:2610.28675v1 Announce Type: new  Abstract: Programmable logic controllers (PLCs) automate industrial processes through repeated scans, each reading inputs, executing control logic, and updating outputs. Engineers write this logic in Structured Text (ST), a high-level textual language defined in IEC 61131-3. Large language models (LLMs) can now generate ST from natural-language requirements, yet compilation success and checks on selected properties or sampled scans can leave behavioral errors undetected. We introduce CycleBench, a benchmark of 301 ST tasks with behavioral requirements, public interfaces, reference programs, and executable scenarios. Its runtime executes each generated program and its reference under aligned inputs, initialization, and logical time, comparing their outputs after every scan. Across eight LLMs and baseline generation methods, execution pass rate trails compilation pass rate by 11.92 percentage points; observed failures include output-update, arithmet
    
[^45]: 谁来验证构建者？迈向智能体生成软件的独立保障

    Who Verifies the Builder? Toward Independent Assurance for Agent-Produced Software

    [https://arxiv.org/abs/2610.28667](https://arxiv.org/abs/2610.28667)

    当智能体既构建软件又自证正确性时，通过测试不再是独立证据，本文主张构建与验证在结构上分离，提出独立保障平面与“验证契约”框架，为智能体生成的软件提供独立的质量保证。

    

    编程智能体现在已经能够实现规范说明、编写并运行测试、修复故障，并宣布结果可以发布。当同一个智能体系统既生产软件、又产出其正确性的证据时，通过的测试套件便不再是独立的证据：可见的测试会成为优化目标，代码和测试可能共享对规范的同一种误解，而构建者自己的测试也可能只是不完整。本文认为，智能体生成的软件需要一个在结构上独立于构建平面的保障平面。我们刻画了利益相关者意图与实际产出之间的四种差距（需求差距、一致性差距、评估差距和模型差距），并指出了验证能够弥合其中的哪些差距。我们提出了一个将构建与验证分离的框架，引入了“验证契约”来规定发布之前必须证明的内容，并以规范优先的阶段来组织验证工作……

    arXiv:2610.28667v1 Announce Type: new  Abstract: Coding agents can now implement a specification, write and run tests, repair failures, and declare the result ready for release. When the same agentic system produces both the software and the evidence of its correctness, a passing test suite is no longer independent evidence: visible tests become an optimization target, code and tests may share one misinterpretation of the specification, and the builder's own tests may simply be incomplete. This paper argues that agent-produced software requires an assurance plane that is structurally independent of the build plane. We characterize four gaps between stakeholder intent and production (requirement, conformance, evaluation, and model gaps) and identify which of them verification can close. We propose a framework that separates building from verifying, introduces a Verification Contract stating what must be demonstrated before release, and organizes verification in a specification-first pha
    
[^46]: 从自然语言意图到可执行控制：面向多组件信息物理系统的分层反馈引导LLM规划

    From Natural Language Intent to Executable Control: Hierarchical Feedback-Guided LLM Planning for Multi-component Cyber-Physical Systems

    [https://arxiv.org/abs/2610.22441](https://arxiv.org/abs/2610.22441)

    提出了一种分层反馈框架，通过让LLM同时生成结构化任务计划与信号时序逻辑（STL）约束，并利用可靠算法识别二者间的矛盾以提供不同类型的反馈，从而引导LLM为多组件信息物理系统生成可执行的任务计划。

    

    大语言模型（LLM）为将自然语言命令转换为信息物理系统（CPS）的任务计划提供了便捷的接口。然而，两个差距阻碍了它们在实际执行中的应用：生成的计划可能误解或遗漏用户的意图，而且即使逻辑上合理的计划，在组件动力学、交互或安全约束下也可能不可行。现有的基于LLM的规划方法通常缺乏关于生成计划为何失败的有信息量的反馈，对如何相应地修改计划几乎没有指导作用。为解决这些挑战，我们提出了一个分层反馈框架，该框架使用不同类型的反馈信息来引导LLM为CPS生成可执行的计划。我们提示LLM同时生成结构化的任务计划和相应的信号时序逻辑（STL）约束，并设计了一个可靠的算法来识别二者之间确定的矛盾，并为修改计划提供反馈……

    arXiv:2610.22441v1 Announce Type: new  Abstract: Large language models (LLMs) provide a convenient interface for translating natural language commands into task plans for cyber-physical systems (CPSs). However, two gaps hinder their use for real execution: generated plans may misinterpret or omit the user's intent, and even a logically plausible plan may be infeasible under component dynamics, interactions, or safety constraints. Existing LLM-based planning approaches often lack informative feedback on why a generated plan fails, providing little guidance for revising the plan accordingly. To address these challenges, we present a hierarchical feedback framework that uses different types of feedback information to guide LLMs in generating executable plans for CPSs. We prompt LLMs to generate both a structured task plan and corresponding signal temporal logic (STL) constraints, and design a sound algorithm that identifies definite contradictions between them and provides feedback for re
    
[^47]: WEAVE：将真实世界代码演进编织为依赖结构化任务，用于代码智能体的基准测试与训练

    WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents

    [https://arxiv.org/abs/2610.22039](https://arxiv.org/abs/2610.22039)

    WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。

    

    真实世界的软件工程很少由孤立的变更构成。功能开发、迁移和系统性修复往往跨越多个拉取请求（PR）展开，需要对整个代码库进行协调一致的修改。然而，大多数可执行的代码库级基准测试仍以单个 issue 或 PR 作为任务单元，而现有的多 PR 设置则依赖粗粒度的聚合和与结构无关的评估。这种错位掩盖了变更之间的依赖关系，限制了长程代码智能体的进展衡量和学习信号。我们提出了 WEAVE，一个将经过验证的 PR 历史转化为可执行的、依赖结构化任务的框架。WEAVE 利用代码库校准的类型化变更位点，将相关联的 PR 组织成有界 DAG（有向无环图），并将每个 DAG 实例化为统一的可执行任务。我们进一步引入了 Graph Credit，这是一种确定性的、基于测试的度量指标，用于衡量符合依赖关系的部分进展，并支持更密集的……

    arXiv:2610.22039v1 Announce Type: cross  Abstract: Real-world software engineering rarely consists of isolated changes. Features, migrations, and systematic fixes often unfold across multiple pull requests (PRs), requiring coordinated modifications across a repository. Yet most executable repository-level benchmarks retain a single issue or PR as the task unit, while existing multi-PR settings rely on coarse aggregation and structure-agnostic evaluation. This mismatch obscures dependencies among changes and limits progress measurement and learning signals for long-horizon code agents. We introduce WEAVE, a framework that transforms verified PR histories into executable, dependency-structured tasks. WEAVE leverages repository-calibrated typed change loci to organize related PRs into bounded DAGs and instantiate each as a unified executable task. We further introduce Graph Credit, a deterministic, test-grounded metric that measures dependency-consistent partial progress and supports dens
    
[^48]: 当智能体评估变得不可靠时：LLM智能体基准测试中的预言机故障与配置依赖性测量的实证研究

    When Agent Evaluation Becomes Unreliable: An Empirical Study of Oracle Failures and Configuration-Dependent Measurement in an LLM Agent Benchmark

    [https://arxiv.org/abs/2610.16067](https://arxiv.org/abs/2610.16067)

    本文提出一种面向LLM智能体评估工具的预言机完整性审计方法，发现了一个会在评分前改变环境状态的奖励函数缺陷，使基准测得的成功率从94.3%降至65.5%，并揭示了配置依赖的测量效应对评估结果的显著影响。

    

    智能体评估工具是运行测试套件、生成输入并通过测试预言机判定结果的软件系统。本文对这类软件进行审计，以发现分数表无法揭示的可靠性缺陷。我们提出了一种预言机完整性审计方法，包含五个状态完整性不变量和三个已知答案控制。我们通过在两个任务域和一个合成域中注入七类预言机缺陷来评估该方法。所有注入的缺陷类别均被检测到，且修复后的预言机没有误报。应用于一个智能体基准测试时，该审计揭示了一个奖励函数缺陷，该缺陷会在运行后评分之前改变环境状态。在4,320个回合中，基于重放的校正将数据库通过标签从4,194个变为2,883个，并将测得的成功率从94.3%降至65.5%。我们还研究了配置依赖的测量效应。在3,240个回合上的组件消融实验表明，智能体的策略文档（摘要在此处截断）

    arXiv:2610.16067v1 Announce Type: cross  Abstract: Agent evaluation harnesses are software systems that run test suites, generate inputs, and decide outcomes with test oracles. This paper audits such software for reliability defects that score tables cannot reveal.   We introduce an oracle-integrity auditing approach with five state-integrity invariants and three known-answer controls. We evaluate it by injecting seven oracle defect classes across two task domains and a synthetic domain. All injected defect classes are detected, and the repaired oracle has no false positives. Applied to an agent benchmark, the audit reveals a reward-function defect that changes the environment state before post-run scoring. Across 4,320 episodes, replay-based correction changes database-pass labels from 4,194 to 2,883 and reduces measured success from 94.3% to 65.5%.   We also study configuration-dependent measurement effects. A component ablation over 3,240 episodes shows that the agent's policy docum
    
[^49]: TileCas：跨编程抽象的级联内核开发

    TileCas: Cascading Kernel Development Across Programming Abstractions

    [https://arxiv.org/abs/2610.14740](https://arxiv.org/abs/2610.14740)

    TileCas提出一个统一可插拔的接口，让编码智能体能够通过跨层诊断和抽象调度，在多个可编辑的编程抽象层次之间迁移和改进内核开发，从而兼顾高级DSL的搜索简便性与低级表示的细粒度控制。

    

    编程抽象传统上是为人类设计的，但代码生成越来越多地由模型和自主智能体来完成。我们通过GPU和AI加速器内核生成来研究面向机器驱动编程的抽象。高级领域特定语言（DSL）通过将实现细节委托给编译器来简化搜索，而低级表示则以更大且更复杂的搜索空间为代价提供更细粒度的控制。本文提出了TileCas，一个统一的、可插拔的接口，使编码智能体能够在多个可编辑的抽象层次上开发内核。TileCas不局限于单一表示，而是允许智能体通过跨层诊断在不同层次间迁移和改进候选方案——跨层诊断将正确性和性能反馈映射为不同抽象层次上可操作的控制；抽象调度则在两者之间进行选择……（原文摘要在此处截断）

    arXiv:2610.14740v2 Announce Type: replace  Abstract: Programming abstractions have traditionally been designed for humans, but code generation is increasingly performed by models and autonomous agents. We investigate abstractions for machine-driven programming through GPU and AI-accelerator kernel generation. High-level domain-specific languages (DSLs) simplify search by delegating implementation details to compilers, while lower-level representations provide finer-grained control at the cost of a larger and more complex search space. In this paper, we present TileCas, a unified, pluggable interface that enables coding agents to develop kernels across multiple editable abstraction levels. Rather than committing to a single representation, TileCas allows an agent to transfer and refine candidates across levels through Cross-Level Diagnosis, which maps correctness and performance feedback to actionable controls at different abstraction levels; Abstraction Scheduling, which selects both t
    
[^50]: PatchRisk：预测开源依赖网络中的未来漏洞暴露

    PatchRisk: Forecasting Future Vulnerability Exposure in Open-Source Dependency Networks

    [https://arxiv.org/abs/2609.13719](https://arxiv.org/abs/2609.13719)

    提出PatchRisk方法，基于发布时的包版本依赖图预测未来传递依赖是否会出现漏洞公告，实现软件供应链的主动风险预警，并应对标签泄露等评估陷阱。

    

    开源软件生态系统是网络规模的依赖网络。一个下游包面临安全风险，可能并非因为它自身的源代码发生了变化，而是因为它的某个传递性依赖后来收到了漏洞公告。现有的漏洞检测工作通常关注代码当前是否存在漏洞，或者是否已经引入了已知的漏洞依赖。我们研究的是一个不同的问题：未来的传递性漏洞暴露。给定在发布时观察到的包版本依赖图，任务是预测在未来时间范围内是否有任何非根依赖会收到漏洞公告。这个问题对于网络智能和软件供应链安全非常重要，因为它支持在未来暴露显现之前进行主动的依赖风险分诊。此外，这个问题也很容易被错误评估：当前的漏洞依赖可能会泄露标签，而同一包的不同版本……（原文摘要在此处被截断）

    arXiv:2609.13719v1 Announce Type: cross  Abstract: Open-source software ecosystems are web-scale dependency networks. A downstream package can become exposed to security risk not because its own source code changes, but because one of its transitive dependencies later receives a vulnerability advisory. Existing vulnerability-detection work often focuses on whether code is currently vulnerable or whether a known vulnerable dependency is already present. We study a different problem: future transitive vulnerability exposure. Given a package-version dependency graph observed at release time, the task is to predict whether any non-root dependency will receive a vulnerability advisory within a future horizon. This problem is important for Web intelligence and software supply-chain security because it supports proactive dependency triage before future exposure is visible. It is also easy to evaluate incorrectly: current vulnerable dependencies can leak the label, and different versions of th
    
[^51]: Token消耗最大化（Tokenmaxxing）的最佳编程语言：编程智能体在不同编程语言下的行为研究

    The Best Programming Language for Tokenmaxxing: An Investigation of Coding Agent Behavior Across Programming Languages

    [https://arxiv.org/abs/2607.22807](https://arxiv.org/abs/2607.22807)

    编程智能体的Token消耗因编程语言而显著不同且跨模型一致，其原因是智能体在不熟悉的语言中反复生成无法编译的代码并重复修改已通过的解决方案。

    

    尽管编程智能体目前在多种编程语言中都已非常有效，本文首次表明其成本（以Token计）会因编程语言的不同而产生显著差异。我们在Python、Java、Rust和OCaml的编程问题上评估了五个最新模型，并严格控制问题难度，结果显示不同编程语言之间的Token消耗存在显著差异，且这一差异在各个模型中保持一致。为探究其原因，我们对智能体轨迹的结构和内容进行了分析。首先，我们重新执行每个中间解，将每条轨迹抽象为测试结果向量的序列，并对相继解之间的工作内容进行标注，这揭示了智能体在不熟悉的语言中会反复生成无法编译的解决方案，并且会反复修改已经通过测试的解决方案。其次，我们分析轨迹文本，发现智能体会在代码注释中规划解决方案，不信任所提供的测试用例，而更倾向于使用自行构造的输入……（原文此处被截断）

    arXiv:2607.22807v2 Announce Type: replace-cross  Abstract: Although coding agents are now very effective in a variety of programming languages, this paper first shows that the cost (in tokens) can very significantly by programming language. We evaluate five recent models on programming problems in Python, Java, Rust, and OCaml. We carefully control for problem difficulty, and show that there can be stark variation in token consumption that is consistent across models. To understand why, we analyze both the structure and content of agent trajectories. First, we re-execute every intermediate solution and abstract each trajectory as a sequence of test-outcome vectors, then label the work between successive solutions. This reveals agents repeatedly producing noncompiling solutions in unfamiliar languages and revising solutions that already pass. Second, we analyze trajectory text, finding that agents plan solutions in code comments, distrust the provided tests in favor of inputs they inven
    
[^52]: 程序分析漫游指南（三）：基本无害的大语言模型

    The Hitchhiker's Guide to Program Analysis, Part III: Mostly Harmless LLMs

    [https://arxiv.org/abs/2606.15122](https://arxiv.org/abs/2606.15122)

    论文提出Evident系统，让LLM不再直接判断静态分析警告的真伪，而是通过识别相关代码并构建测试驱动（harness）来支持精确的针对性静态分析，从而可靠地过滤误报。

    

    过滤掉一条静态分析警告意味着断言不存在任何可行的执行路径能够触发所报告的错误，而确认一个缺陷只需要找到一条能够触发的执行路径。近期的工作使用大语言模型（LLM）来过滤警告。然而，仅凭LLM生成的解释无法证明所报告的错误不可能发生，而错误地过滤掉一条警告会掩盖真实的缺陷。健全的静态分析可以提供这种保证，但可扩展的分析会遗漏我们想要过滤的那些警告，而足够精确的全程序分析又不切实际。幸运的是，只需分析程序的一小部分往往就足以证明一条警告是误报。因此，我们赋予LLM一个不同的角色：它们不再判断警告的有效性，而是识别相关代码并构建一个测试驱动（harness），为精确的、有针对性的静态分析提供所需的上下文。我们提出了Evident，它将LLM与静态分析器Fram...

    arXiv:2606.15122v2 Announce Type: replace  Abstract: Filtering out a static-analysis warning asserts that no feasible execution can trigger the reported error, whereas confirming a bug requires only one that does. Recent work uses LLMs to filter out warnings. However, LLM-generated explanations alone cannot establish that the reported error is impossible, and mistakenly filtering out a warning conceals a real bug. Sound static analysis can provide this guarantee, but scalable analyses leave behind the very warnings we seek to filter, whereas sufficiently precise whole-program analysis is impractical. Fortunately, analyzing a small part of the program is often enough to show that a warning is a false alarm. We therefore assign LLMs a different role: rather than judging the validity of warnings, they identify relevant code and construct a harness that supplies the context needed for precise, targeted static analysis.   We present Evident, which combines LLMs with the static analyzer Fram
    
[^53]: 带洞的性质：基于元性质的数据密集型计算框架测试

    Properties with Holes: Meta-Property-Based Testing of Data-Intensive Computing Frameworks

    [https://arxiv.org/abs/2606.11132](https://arxiv.org/abs/2606.11132)

    该论文提出Holey工具，引入“带洞的元性质”这一概念——固定工作负载的变换方式与输出需满足的关系、将表达式留空以待填充——从而系统化地为Spark、Flink等数据密集型计算框架生成强判定性质的测试，捕捉那些静默返回错误结果的缺陷。

    

    Apache Spark和Flink等数据密集型计算框架正不断通过新的数据类型、表达式和算子扩展其API。其最具破坏性的缺陷是静默地返回错误结果而非崩溃，因此仅能检测崩溃或异常的工具对此无能为力。基于性质的测试能够捕捉这类缺陷，但前提是有人编写带有强判定器的性质，而逐个API地编写性质无法跟上API的增长速度。我们观察到，许多这类性质都是少数教科书式原则的实例：等价表达式的同余性、关系代数等价性以及聚合分解。每条原则不仅对孤立算子成立，也对围绕该算子构建的任意工作负载成立。我们提出了Holey，它引入了元性质，即带洞的性质：元性质固定了工作负载如何被变换以及其输出必须满足何种关系，但将表达式留空待填……（摘要原文在此处截断）

    arXiv:2606.11132v2 Announce Type: replace  Abstract: Data-intensive computing frameworks such as Apache Spark and Flink keep expanding their APIs with new data types, expressions, and operators. Their most damaging bugs return silently wrong results rather than crashing, which tools that detect only crashes or exceptions cannot see. Property-based testing can catch such bugs, but only if someone writes a property with a strong oracle, and writing properties one API at a time cannot keep pace with this growth. We observe that many such properties are instances of a few textbook principles: congruence of equivalent expressions, relational-algebra equivalences, and aggregate decomposition. Each principle holds not only for an isolated operator but also for any workload built around it.   We present Holey, which introduces meta-properties: properties with holes. A meta-property fixes how a workload is transformed and which relation its outputs must satisfy, but leaves open the expressions 
    
[^54]: Lean4Agent：面向智能体工作流与轨迹的形式化建模与验证

    Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory

    [https://arxiv.org/abs/2606.06523](https://arxiv.org/abs/2606.06523)

    本文提出Lean4Agent，这是首个利用Lean4依赖类型形式化语言对智能体工作流的语义一致性进行形式化建模与验证、并能对执行轨迹揭示的运行时故障进行定位的框架。

    

    让大语言模型（LLM）能够执行可靠的多步骤工作流已成为人工智能领域的核心挑战。尽管LLM的智能体能力近来取得了进展，但大多数智能体系统仍然缺乏用于规范、验证和调试其工作流与执行轨迹的形式化方法。这一挑战与数学领域的一个长期问题相呼应：自然语言（NL）的歧义性推动了形式化语言（FL）的发展。受这一范式的启发，我们提出了Lean4Agent——据我们所知，这是首个使用Lean4（一种依赖类型形式化语言）来建模和验证智能体行为的框架。Lean4Agent推出了FormalAgentLib，这是一个可扩展的Lean4库，用于在显式假设下对智能体工作流的语义一致性进行形式化建模与验证，并能够对轨迹所揭示的运行时故障进行定位。基于FormalAgent

    arXiv:2606.06523v3 Announce Type: replace-cross  Abstract: Equipping Large Language Models (LLMs) to execute reliable multi-step workflows has become a central challenge in artificial intelligence. Despite recent advances in LLMs' agentic capabilities, most agent systems still lack formal methods for specifying, verifying, and debugging their workflow and execution trajectories. This challenge mirrors a long-standing problem in mathematics, where the ambiguity of natural languages (NLs) motivates the development of formal languages (FLs). Inspired by this paradigm, we propose **Lean4Agent**, to the best of our knowledge, the first framework that uses Lean4, a dependent-type FL to model and verify agent behavior. **Lean4Agent** launches **FormalAgentLib**, an extensible Lean4 library for formally modeling and verifying agent workflows' semantic consistency under explicit assumptions, and enabling localization of execution-time failures revealed by trajectories. Building on **FormalAgent
    
[^55]: EvoRepair：通过基于经验的自进化增强漏洞修复智能体

    EvoRepair: Enhancing Vulnerability Repair Agents Through Experience-Based Self-Evolution

    [https://arxiv.org/abs/2605.30105](https://arxiv.org/abs/2605.30105)

    EvoRepair是首个基于经验自进化的自动化漏洞修复智能体框架，通过循环的学习-修复过程实现修复经验的积累、精炼与跨漏洞复用，从而提升LLM的漏洞修复能力。

    

    大型语言模型（LLMs）在自动化漏洞修复（AVR）方面展现出巨大潜力，但仍面临若干局限性，包括缺乏漏洞内部的经验积累以及缺乏跨漏洞的经验复用。因此，LLMs在迭代修复过程中可能反复犯类似的错误，且未能充分利用历史漏洞中宝贵的修复知识。为解决这些挑战，我们提出了EvoRepair，这是首个基于经验的自进化AVR智能体框架，使LLMs能够在长周期的漏洞修复任务中积累、精炼并利用领域特定知识。EvoRepair遵循循环的“学习-修复”过程：检索相关的历史经验以指导修复，从修复轨迹中提取新经验，并通过质量感知评分更新经验库。我们在PAT上对EvoRepair与12个代表性漏洞修复基线进行了评估。

    arXiv:2605.30105v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have shown promise for automated vulnerability repair (AVR), but they still face several limitations, including the lack of intra-vulnerability experience accumulation and the lack of cross-vulnerability experience reuse. As a result, LLMs may repeatedly make similar mistakes during iterative repair and underutilize valuable repair knowledge from historical vulnerabilities. To address these challenges, we propose EvoRepair, the first experience-based self-evolving AVR agent framework that enables LLMs to accumulate, refine, and leverage domain-specific knowledge across long-horizon vulnerability repairs. EvoRepair follows a cyclic learn-and-repair process that retrieves relevant past experiences to guide repair, extracts new experiences from repair trajectories, and updates an experience bank using quality-aware scoring. We evaluate EvoRepair against 12 representative vulnerability repair baselines on PAT
    
[^56]: MockingBird：具有隔离验证的上下文感知仓库级代码翻译

    MockingBird: Context-aware Repository-level Code Translation with Isolated Validation

    [https://arxiv.org/abs/2511.21878](https://arxiv.org/abs/2511.21878)

    MockingBird 通过结合上下文感知的跨语言类型解析与基于模拟对象的隔离验证，无需翻译测试即可验证仓库级代码翻译的功能等价性，将 AlphaTrans 基准上 Java 到 Python 翻译的功能等价率从最先进的 25.14% 显著提升。

    

    尽管仓库级代码翻译已取得进展，但跨不同类型系统的自动类型解析以及翻译验证（即功能等价性检查）仍然充满挑战。本文提出了 MockingBird，将上下文感知的类型解析与基于模拟对象的隔离验证相结合。MockingBird 为每个源语言类型检索 API 文档和上下文代码信息，通过提示大语言模型（LLM）来解析精确的跨语言类型映射，从而减少了现有方法所需的人工投入。利用这一自动构建的类型映射，MockingBird 采用自定义的序列化/反序列化方法生成等价的目标语言模拟对象，在不翻译测试的情况下对每个翻译单元进行隔离验证，消除了测试过拟合的风险和测试翻译带来的挑战。在 AlphaTrans 基准的 10 个 Java 到 Python 任务上，MockingBird 将功能等价性从最先进的 25.14% 提升至……

    arXiv:2511.21878v3 Announce Type: replace  Abstract: Despite advances in repository-level code translation, automated type resolution across differing type systems and translation validation (functional-equivalence checking) remain challenging. This paper proposes MockingBird, combining context-aware type resolution with mock-based isolated validation. MockingBird retrieves API documentation and contextual code information per source-language type, prompting an LLM to resolve precise cross-language type mappings, reducing the manual effort of existing approaches. Using this automatically constructed type map, MockingBird employs custom serialization/deserialization to generate equivalent target-language mock objects, validating each translation unit in isolation without test translation, eliminating test-overfitting risks and translation challenges. On 10 Java-to-Python tasks from the AlphaTrans benchmark, MockingBird improves functional equivalence from the state-of-the-art 25.14% to 
    
[^57]: TopoMap：一种基于特征的测试输入空间拓扑区域判别器

    TopoMap: A Feature-based Discriminator of the Topographical Regions in the Test Input Space

    [https://arxiv.org/abs/2509.03242](https://arxiv.org/abs/2509.03242)

    TopoMap是一种模型无关的方法，通过基于输入特征对测试输入空间进行拓扑划分，帮助用户识别代表性不足、失败率高或模型更新后预测发生变化的区域，从而指导数据生成和标注工作的优先级排序。

    

    测试基于深度学习的系统是一项尚未解决的挑战。虽然找到导致深度学习模型出现错误行为的输入相对容易，但如何根据使这些输入对被测深度学习模型产生问题的特征来对它们进行分组，在很大程度上仍未被探索。在这项工作中，我们提出了TopoMap，这是一种模型无关的方法，仅依赖刻画输入的特征来构建输入空间的一种划分，其中每个区域代表一组特定的、可区分的输入特征。这样的划分为用户提供了一种可能性，即识别并分析那些代表性不足、失败率较高，或在模型更新后表现出预测变化的区域。随后，这些被识别的感兴趣区域可用于额外的数据生成，或帮助对已生成的、需要标注工作的输入进行优先级排序。为了根据输入所共有的特定特征对输入进行判别，我们首先应用降维……

    arXiv:2509.03242v3 Announce Type: replace  Abstract: Testing Deep Learning (DL)-based systems is an open challenge. Although it is relatively easy to find inputs that cause a DL model to misbehave, the grouping of such inputs by features that make them problematic for the DL model under test is largely unexplored. In this work, we propose TopoMap, a model-agnostic approach that relies solely on features that characterise inputs to construct a partitioning of the input space where each region represents a specific set of distinguishable input features. Such a partitioning gives users a possibility to identify and analyse areas that are under-represented, show higher failure rates, or exhibit prediction changes after a model update. The identified regions of interest can then be used for additional data generation or can help prioritise already generated inputs that require labelling efforts. To discriminate inputs according to the specific features they share, we first apply dimensional
    
[^58]: 基于堆栈跟踪的崩溃去重与Transformer适配

    Stack Trace-Based Crash Deduplication with Transformer Adaptation

    [https://arxiv.org/abs/2508.19449](https://arxiv.org/abs/2508.19449)

    本文提出dedupT，一种基于Transformer的崩溃去重方法，通过整体建模堆栈跟踪并适配预训练语言模型，显著提升了重复崩溃排序和唯一崩溃检测的准确性，减少了人工分类负担。

    

    arXiv:2508.19449v2 公告类型：替换交叉 摘要：自动化崩溃报告系统会生成大量重复报告，使问题跟踪系统不堪重负，并增加开发人员的工作负担。传统的基于堆栈跟踪的去重方法——依赖字符串相似性、基于规则的启发式方法或深度学习（DL）模型——通常无法捕捉堆栈跟踪中的上下文和结构关系。我们提出了dedupT，一种基于Transformer的方法，它将堆栈跟踪整体建模，而非视为孤立的帧。dedupT首先将预训练语言模型（PLM）适配到堆栈跟踪上，然后利用其嵌入来训练一个全连接网络（FCN），以有效排序重复崩溃。在真实世界数据集上的大量实验表明，dedupT在重复排序和唯一崩溃检测方面均优于现有的深度学习和传统方法（例如序列比对和信息检索技术），显著减少了人工分类工作。

    arXiv:2508.19449v2 Announce Type: replace-cross  Abstract: Automated crash reporting systems generate large volumes of duplicate reports, overwhelming issue-tracking systems and increasing developer workload. Traditional stack trace-based deduplication methods---relying on string similarity, rule-based heuristics, or deep learning (DL) models---often fail to capture the contextual and structural relationships within stack traces. We propose dedupT, a transformer-based approach that models stack traces holistically rather than as isolated frames. dedupT first adapts a pretrained language model (PLM) to stack traces, then uses its embeddings to train a fully-connected network (FCN) to rank duplicate crashes effectively. Extensive experiments on real-world datasets show that dedupT outperforms existing DL and traditional methods (e.g., sequence alignment and information retrieval techniques) in both duplicate ranking and unique crash detection, significantly reducing manual triage effort.
    
[^59]: 为下游项目生成缓解措施以消除上游库漏洞

    Generating Mitigations for Downstream Projects to Neutralize Upstream Library Vulnerability

    [https://arxiv.org/abs/2503.24273](https://arxiv.org/abs/2503.24273)

    提出LUMEN方法，在无法获取上游库源代码或补丁的情况下，为下游项目自动生成缓解措施，以中和上游第三方库的漏洞。

    

    第三方库在软件开发中至关重要，因为它们使开发者无需重新实现已有的功能。然而，这些库中的漏洞会对依赖它们的项目构成重大风险。当没有补丁版本可用，或下游项目存在严格的版本约束时，升级依赖项是不可行的。此外，当无法访问库的源代码时，修复漏洞也极具挑战性。最先进的自动漏洞修复和自动程序修复方法都无法解决这一问题。因此，在无法访问源代码或没有可用补丁的情况下缓解库漏洞，对于快速响应潜在的安全攻击至关重要。现有工具在通用性和功能安全方面存在挑战。在漏洞披露后，LUMEN 通过检索准备一个候选缓解策略池……

    arXiv:2503.24273v2 Announce Type: replace  Abstract: Third-party libraries are essential in software development as they prevent the need for developers to recreate existing functionalities. However, vulnerabilities within these libraries pose significant risks to dependent projects. Upgrading dependencies is infeasible when no patched version is available or when downstream projects have strict version constraints. Moreover, repairing the vulnerability proves challenging when the source code of the library is inaccessible. State-of-the-art automatic vulnerability repair and automatic program repair methods both fail to address this issue. Therefore, mitigating library vulnerabilities without access to source code or available patches is crucial for a swift response to potential security attacks. Existing tools encounter challenges concerning generalizability and functional security. Upon vulnerability disclosure, LUMEN prepares a pool of candidate mitigation strategies by retrieving e
    

