# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An analysis of the relationship of input metrics](https://arxiv.org/abs/2609.11824) | 本文提出了 k-alt-path 这一新输入度量，在提升测试灵敏度的同时减少冗余，并首次对软件测试中的常见输入度量进行了系统性比较分析，揭示了传统经验性比较策略的根本缺陷。 |
| [^2] | [Reproducibility in the Age of Agentic AI: Context Engineering at the Timescale of a Codebase](https://arxiv.org/abs/2609.11728) | 本文提出可复现研究实践本质上就是为AI编码智能体进行的上下文工程，智能体降低了维护测试、提交历史和决策记录等科研工件的成本并使收益即时化，但研究人员仍需承担验证这些工件及其科学判断的责任。 |
| [^3] | [Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents](https://arxiv.org/abs/2609.11677) | 该论文指出运行时框架进化中缺乏有原则的失败诊断是关键瓶颈——观察到的失败可能源于模型自身缺陷或框架系统性缺陷，直接针对个别失败优化会导致不必要的模型特定适应，因此提出Ecdysis方法以实现对LLM智能体运行时框架更高效、更有效且泛化能力更强的训练。 |
| [^4] | [PRISMA-LLM: An Empirical Reporting Framework for AI-Assisted Systematic Reviews](https://arxiv.org/abs/2609.11559) | 本文通过分析包含888篇论文的SciLitBench语料库，揭示了AI辅助系统综述中评估与报告的不一致问题，并提出PRISMA-LLM实证框架，将实现披露与后果敏感的评估及局限性报告分离，以规范这一领域。 |
| [^5] | [ChurnBench: A Drift-Aware Benchmark Demonstrating That Refresh Scheduling, Not Cache Age, Governs Staleness in Agentic AI](https://arxiv.org/abs/2609.11515) | ChurnBench是一个漂移感知的开源基准测试，通过时间线形式的企业数据织物和仅追加的真相账本，首次将“新鲜度错误”与“推理错误”区分开来，并证明智能体AI中的信息陈旧程度由刷新调度而非缓存时效决定。 |
| [^6] | [DeFiFlowBench: Benchmarking and Improving Safe Executability in Natural-Language DeFi Workflow Synthesis](https://arxiv.org/abs/2609.11504) | 提出DeFiFlowBench基准用于评估自然语言DeFi工作流合成的安全可执行性，并引入Koan-Safe方法，将静态安全得分从0.33提升至0.67，同时实现零不安全执行。 |
| [^7] | [Deep Learning-based Bug Triage System](https://arxiv.org/abs/2609.11420) | 本文提出一种基于预训练RoBERTa-base的自动化缺陷分诊系统，仅用五个训练周期即可达到0.90的缺陷识别准确率，展示了微调transformer模型在软件工程自动化中的高效性与高性能。 |
| [^8] | [Agent-Integrated Software: Interaction Contracts and Continuous Assurance](https://arxiv.org/abs/2609.11381) | 该论文提出智能体集成软件（AIS）软件模式与意图级交互抽象（IIA）任务语义，通过交互契约约束任务级交互与应用行为之间的对应关系，并以持续保障机制在依赖变化时维护声明，从而解决将智能体嵌入现有应用时的协调问题。 |
| [^9] | [CoSTAR: Data Synthesis-Driven Constraint-Aware COBOL Section Summarization for Legacy System Modernization](https://arxiv.org/abs/2609.11332) | 提出CoSTAR框架，通过执行验证的数据合成与约束感知模型训练相结合，解决COBOL节级代码摘要中的数据稀缺与迁移约束保持两大难题，助力COBOL遗留系统现代化。 |
| [^10] | [Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development](https://arxiv.org/abs/2609.11331) | 该论文将CPS开发中的模型不一致性问题创新性地转化为干预响应建模问题，结合Saltelli采样与多保真度蒙特卡洛估计训练代理模型，实现了对大范围不确定性空间中不一致性的系统性预测、分析与解释。 |
| [^11] | [Exploring the Role of Security Experience and ChatGPT Usage Strategies on Secure Software Engineering Education](https://arxiv.org/abs/2609.11303) | 该研究通过对26名网络安全硕士生的交互日志实证分析发现，ChatGPT使用的多样性（即采用的不同使用模式数量），而非个体使用模式或先前安全经验，与漏洞修复作业的成绩呈正相关。 |
| [^12] | [Can AI Remediate Backend Failures Safely? GuardedAct with Blast-Radius-Aware Sandboxing](https://arxiv.org/abs/2609.11264) | 提出了GuardedAct框架，通过爆炸半径感知的数字孪生沙箱模拟LLM生成的后端故障修复动作，并利用回滚置信度风险门控只自动执行低风险动作、将高风险动作交由人工审查，从而实现安全的AI自动故障修复。 |
| [^13] | [TripleBound: Triplet-Guided Heterogeneous Graph Learning for Microservice Decomposition](https://arxiv.org/abs/2609.11212) | TripleBound提出了一种混合框架，将从包结构、命名约定和代码位置解析出的弱监督三元组约束注入异构图神经网络的共享潜在空间，从而在统一的表示学习目标下联合整合结构依赖与语义相似性信号，实现单体应用到微服务的自动化分解。 |
| [^14] | [FST Pay: Deterministic Safety-Gated Architecture for Youth Digital Payments](https://arxiv.org/abs/2609.11195) | 提出FST Pay架构，通过在实时支付授权路径上强制实施严格的确定性安全门控，并将AI解释功能解耦至下游处理，从而在不引入概率性AI非确定性风险的前提下保障青少年数字支付安全。 |
| [^15] | [SemVerBench: Benchmarking LLM Comprehension of Version-Constraint Resolution Semantics](https://arxiv.org/abs/2609.11180) | 本文提出首个跨 npm、PEP 440 和 Cargo 三个生态系统的版本约束解析语义基准测试 SemVerBench，通过对 240 个机器可验证项目的评估，发现六个前沿大语言模型在版本约束语义处理上存在系统性盲点，如 Cargo 部分比较器进位规则使所有模型准确率降至约 60%。 |
| [^16] | [A Model-Centric DevOps Architecture for DEVS-Based Digital Twin Simulation Services](https://arxiv.org/abs/2609.11122) | 本文提出一种以模型为中心的DevOps架构，将基于DEVS的数字孪生仿真模型作为一等制品，通过声明式YAML定义、到multiPDEVS的形式化映射、CI/CD流水线中的结构与语义验证以及Kubernetes容器化微服务部署，实现了仿真模型的可追溯版本管理与持续交付。 |
| [^17] | [SaltBench: A Referee-Gated Protocol for Measuring Method Effects in Machine-Checked Software Work](https://arxiv.org/abs/2609.11076) | 提出了SaltBench，一个由机器裁判门控的基准测试协议，通过严格的隔离机制（经探针实际检验）、预注册预测和“停止即暂停”等规则，使机器裁判对编程智能体工作方式的影响变得可测量且不可事后叙述。 |
| [^18] | [Grounding Agent Memory: Environment-Probing Curation for Enterprise Agents](https://arxiv.org/abs/2609.11060) | 该论文提出“环境探测式记忆管理”方法，为智能体的异步记忆管理器赋予最小权限的只读环境工具以验证、限定和刷新候选记忆，无需重训模型即可将CLBench通过率从39%提升至73%。 |
| [^19] | [BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure](https://arxiv.org/abs/2609.11028) | 本文提出BenchShield，一个基于奖励相关事件有限生命周期形式化模型的检测工具层，通过静态与动态两种互补分析在基准测试基础设施内保障LLM智能体评估的奖励完整性，防范奖励劫持。 |
| [^20] | [RCL: A Retrieval-Confidence Layer for Detecting Insufficient Context in Enterprise Retrieval-Augmented Code Generation](https://arxiv.org/abs/2609.11023) | 该论文提出RCL（检索-置信度层），一个插入在检索与生成之间的轻量级模块，通过结合基于调用图的结构覆盖率分数与置信度信息，在生成开始前检测企业级RAG代码生成中检索上下文是否结构上充分。 |
| [^21] | [DeFiFusion: Combining Transaction Events with Smart Contracts to Detect Price Manipulation Attacks](https://arxiv.org/abs/2609.11008) | DeFiFusion是一个双模态检测框架，通过联合建模交易事件与智能合约执行语义来检测DeFi中的价格操纵攻击，克服了单纯基于交易或静态合约分析方法各自的根本局限。 |
| [^22] | [Engineering Reliable Commit Gates for Agentic AI: Cost-Aware Verification Portfolios under Common-Mode Data Failures](https://arxiv.org/abs/2609.10969) | 该论文发现验证器共享同一证据源时会继承上游故障（共享证据下62.9%的不安全提案被批准，而独立证据源仅为22.9%），据此提出VP-CONTROL成本感知验证组合控制器，仅利用部署可观测元数据选择验证方案，在锁定测试集上将不安全执行率降至1.9%并实现38.2%的自动化安全覆盖率。 |
| [^23] | [Decoupling Readiness from Release for Tail-Aware Scheduling of Agentic LLM Workflows](https://arxiv.org/abs/2609.10964) | 本文提出一种尾风险感知的智能体LLM工作流轮次释放调度方法，通过均值-CVaR目标联合决策释放时机与未完成工作量预算，将轮次就绪与释放解耦以降低尾延迟。 |
| [^24] | [What a Random Draw from the MCP Registry Contains, and What Tool-Use Benchmarks Contain Instead](https://arxiv.org/abs/2609.10962) | 该研究通过对MCP注册表进行可复现的随机概率抽样，首次揭示了真实服务器生态中近半数服务器根本无法启动、安全注释遗漏率高达58.8%，从而证明现行工具使用基准测试所依赖的人工精选样本会系统性高估生态系统的实际可用性与安全性。 |
| [^25] | [LLMVul: A Vulnerability-Labeled Dataset of LLM-Generated C/C++ Functions from Real Production Repositories](https://arxiv.org/abs/2609.10945) | 该论文提出了LLMVul数据集，通过从GitHub真实生产仓库中挖掘AI辅助开发活动，构建了包含来自226个仓库的21,430个LLM生成的C/C++函数并带有漏洞标注的数据集，填补了研究真实场景下LLM生成代码安全缺陷的空白。 |
| [^26] | [AspisAI: A Canonical, Machine-Interpretable Governance Framework for Automated Multi-Standard Compliance Monitoring](https://arxiv.org/abs/2609.10881) | 该论文提出 AspisAI 框架，将 ISO/IEC 27001、NIST CSF 2.0、Cyber Essentials、GDPR 等多个网络安全与隐私标准的要求转化为规范化的机器可解释控制模型，并通过基于条件的决策规则自动评估合规证据，实现可解释、可追溯的自动化多标准合规监控。 |
| [^27] | [A2ABreak: Systematic Security Analysis of the A2A Protocol](https://arxiv.org/abs/2609.10871) | 本文提出A2ABreak框架，首次对A2A协议进行系统性安全分析，通过LLM辅助从自然语言规范中提取经验证的有限状态机模型（37个状态、76个转换），并利用对抗性验证系统性发现协议层面的安全漏洞。 |
| [^28] | [DR-LabStack: Design and Implementation of a Clinician-Facing Web System for Diabetic Retinopathy Prediction](https://arxiv.org/abs/2609.10796) | DR-LabStack是一个基于React-Flask、面向临床医生的网络系统，通过统一的界面和后端适配机制集成了四种异构的糖尿病视网膜病变预训练预测模型，解决了不同模型在输入格式、预处理和输出语义上的差异问题。 |
| [^29] | [Beyond Static Guarantees: Measuring the Static-Pass Dynamic-Fail Gap in Security-Sensitive and LLM-Generated Python Code](https://arxiv.org/abs/2609.10762) | 该论文首次提出“静态通过-动态失败”（SPDF）现象，并设计了一个融合静态扫描、LLM驱动CWE推理与隔离容器内自主漏洞利用验证的三阶段智能体流水线，以量化静态分析通过但代码在运行时仍可被利用的安全评估盲区。 |
| [^30] | [Towards a Deterministic Math Solver for Clinical Language Models](https://arxiv.org/abs/2609.10728) | 本文提出让临床大语言模型不直接进行算术计算，而是生成针对性Python代码交由受限本地执行器作为确定性求解器运行，但在MedCalc-Bench Verified基准上的评估表明，在公式和标准变量均已提供的情况下，这种程序求解接口相比模型直接计算并无可靠优势。 |
| [^31] | [Governed Human-AI Prioritization Under Uncertainty: Adaptive Estimation and Dependency-Constrained Portfolio Selection](https://arxiv.org/abs/2609.10648) | 本文提出了一种在不确定性下可治理、可检查且可重新校准的人机协同优先级排序框架，通过五个定量算子（BVS、ERS、PVS、CCS、ODP）实现自适应估计与依赖约束下的组合选择，并用受控合成实验验证了该框架对参数扰动的鲁棒性。 |
| [^32] | [Numbat: Building and Verifying a Self-Contained Machine-Learning Stack](https://arxiv.org/abs/2609.10632) | Numbat是一个完全用Zig语言编写、零第三方运行时依赖的自包含机器学习技术栈，覆盖从张量计算到多GPU训练的完整功能，并通过稳定的C ABI接口和可执行的验收门来编码监管要求并实现验证。 |
| [^33] | [AI Safety: Not Optional, Not Later](https://arxiv.org/abs/2609.10630) | 论文提出一种“安全即设计”的多层次AI安全保证架构，通过结合模型级监督与系统级控制，并辅以独立验证、监控、证据基础设施和治理机制，全面保障AI系统安全。 |
| [^34] | [On the Relation between Code Quality and Machine Learning Performance: A Large-scale Empirical Study](https://arxiv.org/abs/2609.10610) | 本研究对265,363个Kaggle竞赛Python笔记本进行了大规模实证分析，以探究代码质量与机器学习性能之间的关系，并评估受欢迎程度和作者专业度等社交信号能否作为代码质量或模型性能的可靠指标。 |
| [^35] | [Generative AI for trustworthy systems - Towards a health check model](https://arxiv.org/abs/2609.10595) | 本文基于对十八位跨行业资深从业者的访谈研究，提出了一个包含系统层与组织层共八个维度的“可信自主性健康检查模型”，用于多维度刻画组织如何在生成式AI辅助的软件工程中建立信任。 |
| [^36] | [ReqEvolve: User-Oriented Software Self-Evolution through Automatic Requirement Interpretation](https://arxiv.org/abs/2609.10590) | ReqEvolve是一个运行时代码生成系统，通过融合自动需求工程与测试驱动开发，将用户的高层次自然语言请求直接转化为可执行功能，实现了无需开发人员介入的用户驱动软件自演化。 |
| [^37] | [Optimizing AI Inference Across the Deployment Stack](https://arxiv.org/abs/2609.10550) | 本文提出了一个统一的分析框架，将模型级技术、编译器转换和系统策略三个层面的推理优化方法纳入三层分类体系，并将部署表述为多目标优化问题，为跨部署栈的AI推理优化提供了系统性理论分析。 |
| [^38] | [When Passing Tests Hides Vulnerabilities: An Empirical Study of Silent Failures in Agentic Systems](https://arxiv.org/abs/2609.10548) | 该研究通过分析七个智能体框架的1,030条执行轨迹，首次系统性地识别并分类了LLM代码修复中“通过测试却仍存在漏洞”的静默失败，归纳出遗漏、引入和不足三大类问题。 |
| [^39] | [KG-Commit: A Dynamic Knowledge Graph for Online Just-in-Time Software Defect Prediction](https://arxiv.org/abs/2609.06272) | 本文提出KG-Commit动态知识图谱，通过增量维护代码库历史、文件内代码结构和提交语义，结合AST差异机制与完全基于CPU的轻量级图推理，实现在线即时软件缺陷预测，在11个Apache项目上取得最优综合性能。 |
| [^40] | [FaultLens: Learning Compact Behavioral Test Suites for Generated Operational Programs](https://arxiv.org/abs/2608.26746) | 本文提出FaultLens方法，通过结合故障驱动的贪婪选择和突变无关的多样性组件，学习紧凑的行为测试套件，以高效检测生成程序中的稀疏边界和交互故障。 |
| [^41] | [Causal Explanations of Process Monitor Predictions](https://arxiv.org/abs/2608.24672) | 本文提出了一种基于实际因果框架的新方法，通过定义捕获事件时间依赖性的因果模型，为过程监控预测生成局部解释，并量化事件对预测结果的影响。 |
| [^42] | [GitSkills: A Dataset of Agent Skills on GitHub](https://arxiv.org/abs/2608.10906) | 本文提出了GitSkills数据集，收录了GitHub上3,797,117个智能体技能文件，为首次实证研究开发者如何编写、复用和维护基于自然语言的智能体技能提供了数据基础。 |
| [^43] | [Rust Coreutils: Rebuilding Unix Foundations in a Modern Language](https://arxiv.org/abs/2608.07135) | 本文介绍了用Rust语言重新实现GNU coreutils的项目，该实现已可在大多数Linux发行版上作为即插即用的替代品，展示了现代编程语言如何为遗留关键软件带来可靠的现代化改造。 |
| [^44] | [Execution-First Synthetic Tool-Use Trace Generation for LLM Agents](https://arxiv.org/abs/2607.29175) | 提出执行优先框架SyntheticAgentTraceQA，通过先构建、执行并验证工具调用轨迹再合成用户任务，解决了传统查询优先数据合成无法保证工具交互有效性的问题。 |
| [^45] | [Assessing Language Models for Salient Class Identification](https://arxiv.org/abs/2606.21629) | 本研究构建了包含7,911个提交的新数据集ApacheJavaCM，并评估语言模型能否无需特征工程、图构建或训练，直接从代码提交中识别出显著类。 |
| [^46] | [Adaptive Proof Refinement with LLM-Guided Strategy Selection](https://arxiv.org/abs/2510.25103) | 提出了Adapt框架，利用LLM引导的决策器根据证明助手状态和错误证明的上下文动态选择最合适的精炼策略，突破了现有方法固定策略的局限，提升了自动定理证明的性能。 |
| [^47] | [GDPR-Relevant Privacy Concerns in Mobile Apps Research: A Systematic Literature Review](https://arxiv.org/abs/2411.19142) | 本文通过系统性文献综述，首次对移动应用领域GDPR相关隐私问题的现有研究进行了描述、分析和分类，填补了该领域缺乏二次研究的空白。 |

# 详细

[^1]: 输入度量之间关系的分析

    An analysis of the relationship of input metrics

    [https://arxiv.org/abs/2609.11824](https://arxiv.org/abs/2609.11824)

    本文提出了 k-alt-path 这一新输入度量，在提升测试灵敏度的同时减少冗余，并首次对软件测试中的常见输入度量进行了系统性比较分析，揭示了传统经验性比较策略的根本缺陷。

    

    输入度量根据测试套件中输入所具有的特征来评估测试的进展。早在20世纪50年代，先前的研究工作就建立了一些此类度量标准，但很少有工作尝试对它们进行比较。本文通过利用分区测试文献中为其他度量类别提出的现有方法来开展这项工作。在定义并回顾了常见的输入度量之后，我们首先通过一个简短的案例研究揭示了典型的经验性比较策略对于度量比较而言在根本上是不充分的。随后，我们通过定义并实现一种名为 k-alt-path 的新度量，展示了如何严格地改进标准度量，该新度量在提高灵敏度的同时减少了相对于 k-path 的冗余。接着，我们对其他常见输入度量进行了系统比较，并讨论了研究发现的启示。通过这些贡献，我们提出了能够为度量选择提供依据并形成策略的分区测试分析方法。

    arXiv:2609.11824v1 Announce Type: new  Abstract: Input metrics evaluate the progress of testing in terms of features of inputs present in a test suite. Previous works, as early as the 1950s, established a number of such metrics, but few endeavored to compare them. This paper does so by utilizing existing methods proposed for other metric classes in partition testing literature. After defining and reviewing common input metrics, we begin with a short case study revealing that typical empirical comparison strategies are fundamentally insufficient for comparing metrics. Then, we demonstrate how one rigorously improves a standard metric by defining and implementing $k$-alt-path, a new metric which reduces redundancy while improving sensitivity over $k$-path. Each of the other common input metrics are then systematically compared before discussing the implications of our findings. With these contributions, we bring forward partition testing analysis methods that justify and form a strategy 
    
[^2]: 智能体AI时代的可复现性：以代码库时间尺度进行的上下文工程

    Reproducibility in the Age of Agentic AI: Context Engineering at the Timescale of a Codebase

    [https://arxiv.org/abs/2609.11728](https://arxiv.org/abs/2609.11728)

    本文提出可复现研究实践本质上就是为AI编码智能体进行的上下文工程，智能体降低了维护测试、提交历史和决策记录等科研工件的成本并使收益即时化，但研究人员仍需承担验证这些工件及其科学判断的责任。

    

    可复现的研究实践本身就是为AI编码智能体进行的上下文工程。我认为，智能体降低了维护测试、提交历史、仓库结构、指令和决策记录的成本，同时使其收益立竿见影。研究人员仍然有责任验证这些工件以及它们所编码的科学判断。

    arXiv:2609.11728v1 Announce Type: new  Abstract: Reproducible research practices are context engineering for AI coding agents. I argue that agents lower the cost of maintaining tests, commit histories, repository structure, instructions, and decision records while making their benefits immediate. Researchers remain responsible for verifying these artifacts and the scientific judgments they encode.
    
[^3]: Ecdysis：面向大语言模型智能体的高效且有效的运行时框架训练

    Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents

    [https://arxiv.org/abs/2609.11677](https://arxiv.org/abs/2609.11677)

    该论文指出运行时框架进化中缺乏有原则的失败诊断是关键瓶颈——观察到的失败可能源于模型自身缺陷或框架系统性缺陷，直接针对个别失败优化会导致不必要的模型特定适应，因此提出Ecdysis方法以实现对LLM智能体运行时框架更高效、更有效且泛化能力更强的训练。

    

    自进化的运行时框架可以显著提升大语言模型（LLM）智能体的能力，并为优化智能体执行提供了一种有前景的范式。现有的框架进化方法通常依赖于迭代搜索，即基于任务实例的执行反馈反复评估和修改候选框架。虽然这种范式能够实现框架的持续优化，但由于需要反复执行智能体和修改代码，会产生巨大的时间开销，并且可能对观察到的任务和特定失败模式过拟合，导致对未见任务的泛化能力下降。我们发现缺乏有原则的失败诊断是框架进化的关键瓶颈：观察到的失败可能反映的是模型特定的缺陷或系统性的框架缺陷，直接针对个别失败进行优化可能导致不必要的模型特定适应。因此我们提出Ecdysis方法……（摘要原文在此处截断）

    arXiv:2609.11677v1 Announce Type: new  Abstract: Self-evolving runtime harnesses can substantially improve the capabilities of large language model (LLM) agents and provide a promising paradigm for optimizing agent execution. Existing harness evolution methods typically rely on iterative search, repeatedly evaluating and revising candidate harnesses based on execution feedback from task instances. While this paradigm enables continuous harness optimization, it incurs substantial time overhead due to repeated agent executions and code modifications, and may overfit to observed tasks and specific failure patterns, resulting in degraded generalization to unseen tasks. We identify the lack of principled failure diagnosis as a key bottleneck in harness evolution: an observed failure can reflect either model-specific deficiencies or systematic harness deficiencies, and directly optimizing against individual failures can lead to unnecessary model-specific accommodation. We therefore propose E
    
[^4]: PRISMA-LLM：一种AI辅助系统综述的实证报告框架

    PRISMA-LLM: An Empirical Reporting Framework for AI-Assisted Systematic Reviews

    [https://arxiv.org/abs/2609.11559](https://arxiv.org/abs/2609.11559)

    本文通过分析包含888篇论文的SciLitBench语料库，揭示了AI辅助系统综述中评估与报告的不一致问题，并提出PRISMA-LLM实证框架，将实现披露与后果敏感的评估及局限性报告分离，以规范这一领域。

    

    大语言模型和AI驱动的软件日益参与系统综述的决策过程，然而审计这些工作流程所需的信息报告却并不一致。我们分析了SciLitBench——一个包含888篇综述自动化论文和14,726条标注的语料库，以刻画方法、综述阶段使用、评估以及所报告局限性的变化。自动化已转向面向大语言模型和软件的工作流程，其中包括可能改变证据基础的阶段。自2023年以来，38.0%的软件/产品论文未报告任何评估，而大语言模型论文中这一比例为9.3%。报告覆盖率随大语言模型工作流程复杂性的增加而提高，但52%仅报告正面结果的大语言模型评估仍然存在未满足的可靠性或性能要求。基于这些模式，我们提出了PRISMA-LLM，这是一个基于实证的框架，将实现披露与后果敏感的评估和局限性报告分离开来。

    arXiv:2609.11559v1 Announce Type: new  Abstract: Large language models (LLMs) and AI-enabled software increasingly participate in systematic-review decisions, yet the information needed to audit these workflows is reported inconsistently. We analyze SciLitBench, a corpus of 888 review-automation papers with 14,726 annotations, to characterize changes in methods, review-stage use, evaluation and reported limitations. Automation has shifted toward LLM- and software-facing workflows, including stages that can alter the evidence base. Since 2023, 38.0% of software/product papers reported no evaluation, compared with 9.3% of LLM papers. Reporting coverage increased with LLM workflow complexity, yet 52% of positive-only LLM evaluations still reported an unmet reliability or performance requirement. From these patterns, we introduce PRISMA-LLM, an empirically grounded framework separating implementation disclosure from consequence-sensitive evaluation and limitation reporting.
    
[^5]: ChurnBench：一个漂移感知的基准测试，证明是刷新调度而非缓存时效决定了智能体AI中的信息陈旧程度

    ChurnBench: A Drift-Aware Benchmark Demonstrating That Refresh Scheduling, Not Cache Age, Governs Staleness in Agentic AI

    [https://arxiv.org/abs/2609.11515](https://arxiv.org/abs/2609.11515)

    ChurnBench是一个漂移感知的开源基准测试，通过时间线形式的企业数据织物和仅追加的真相账本，首次将“新鲜度错误”与“推理错误”区分开来，并证明智能体AI中的信息陈旧程度由刷新调度而非缓存时效决定。

    

    arXiv:2609.11515v1 公告类型：新论文 摘要：在生产环境中，智能体系统回答问题所依赖的数据分布在多个位置且不断变化：许可证被重新分配、用户被注销、价格发生变动、合同得到续签。现有的检索基准测试将数据冻结为静态快照，因此它们只能评估智能体是否找到了正确的段落，而无法判断智能体的答案是否仍然正确。我们提出了ChurnBench，这是一个开源基准测试，它将四源企业数据织物生成为时间线而非快照。每一项变更都被写入仅追加的真相账本中，黄金标准答案从该账本计算得出，而非从实时存储中获取。因此，一个在数据检索时正确但在评估时错误的答案会被检测出来并标记为“新鲜度错误”，与“推理错误”相区别；我们通过为每个报告的案例在两个时间戳上解析真相来验证这一机制。利用这一工具，我们发现当系统按计划刷新时，缓存时效并不……（原文在此处截断）

    arXiv:2609.11515v1 Announce Type: new  Abstract: In production, agentic systems answer questions over data that lives in several places and keeps changing: licenses are reassigned, users offboarded, prices changed, contracts renewed. Existing retrieval benchmarks freeze the data, so they cannot ask whether an agent's answer is still true, only whether it found the right passage. We present ChurnBench, an open-source benchmark that generates a four-source enterprise data fabric as a timeline rather than a snapshot. Every change is written to an append-only ground-truth ledger, and gold answers are computed from that ledger, never from the live stores. An answer that was correct when its data was retrieved but wrong when evaluated is therefore detected and labeled a freshness error, distinct from a reasoning error; we validate this by resolving ground truth at both timestamps for every case reported. Using the instrument, we find that when a system refreshes on a schedule, cache age does
    
[^6]: DeFiFlowBench：自然语言DeFi工作流合成中安全可执行性的基准测试与改进

    DeFiFlowBench: Benchmarking and Improving Safe Executability in Natural-Language DeFi Workflow Synthesis

    [https://arxiv.org/abs/2609.11504](https://arxiv.org/abs/2609.11504)

    提出DeFiFlowBench基准用于评估自然语言DeFi工作流合成的安全可执行性，并引入Koan-Safe方法，将静态安全得分从0.33提升至0.67，同时实现零不安全执行。

    

    结构上有效的DeFi工作流仍可能授权一笔代价高昂的交易。我们提出了DeFiFlowBench，这是一个包含207个团队编写的提示词的基准，用于自然语言DeFi工作流合成。它测量图覆盖率、配置完整性和声明的安全谓词，然后在本地EVM上测试支持的交易配置。在固定5%价格影响上限的条件下，直接提示、约束提示和少样本提示在每种配置下均产生14-19次不安全的保留执行。从报价导出的滑点界限并不能阻止订单本身的价格影响。我们提出了Koan-Safe，它结合了仅基于提示词的意图解析器、可替换的生成器以及带有默认安全参数的结构修复。在75个保留工作流提示上，其混合变体在静态安全代理指标上得分为0.67，而最佳基线仅为0.33。Koan-Safe在保存的基准输出上未记录到任何不安全执行。匹配候选消融实验产生（原文在此截断）

    arXiv:2609.11504v1 Announce Type: new  Abstract: A structurally valid DeFi workflow can still authorize a costly trade. We introduce DeFiFlowBench, a benchmark of 207 team-authored prompts for natural-language DeFi workflow synthesis. It measures graph coverage, configuration completeness, and declared safety predicates, then tests supported trade configurations on a local EVM. Direct, constrained, and few-shot prompting produce 14-19 unsafe held-out executions per configuration under a fixed 5% price-impact cap. A slippage bound derived from a quote does not prevent the price impact of the order itself. We propose Koan-Safe, which combines a prompt-only intent parser, a replaceable generator, and structural repair with default safety parameters. On 75 held-out workflow prompts, its hybrid variant scores 0.67 on the static safety proxy, compared with 0.33 for the best baseline. Koan-Safe records no unsafe executions on the saved benchmark outputs. A matched-candidate ablation produces 
    
[^7]: 基于深度学习的缺陷分诊系统

    Deep Learning-based Bug Triage System

    [https://arxiv.org/abs/2609.11420](https://arxiv.org/abs/2609.11420)

    本文提出一种基于预训练RoBERTa-base的自动化缺陷分诊系统，仅用五个训练周期即可达到0.90的缺陷识别准确率，展示了微调transformer模型在软件工程自动化中的高效性与高性能。

    

    有效的缺陷分诊通过准确分类和分配报告的软件缺陷，对于简化软件开发生命周期至关重要。本文提出了一种基于预训练RoBERTa-base transformer架构的自动化缺陷分诊系统。通过利用深度上下文表示，我们的方法能够高效地对传入的缺陷报告进行分类以优化分配。实验评估表明，所提出的系统仅用五个训练周期就实现了0.90的高缺陷识别准确率。这些发现凸显了微调transformer模型在实际软件工程自动化中的效率和高性能。

    arXiv:2609.11420v1 Announce Type: new  Abstract: Effective bug triage is crucial for streamlining the software development lifecycle by accurately categorizing and assigning reported software defects. In this paper, we propose an automated bug triage system built upon the pre-trained RoBERTa-base transformer architecture. By leveraging deep contextual representations, our approach efficiently classifies incoming bug reports to optimize assignment. Experimental evaluation demonstrates that the proposed system achieves a strong bug identification accuracy of 0.90 within just five training epochs. These findings highlight the efficiency and high performance of fine-tuned transformer models for practical software engineering automation.
    
[^8]: 智能体集成软件：交互契约与持续保障

    Agent-Integrated Software: Interaction Contracts and Continuous Assurance

    [https://arxiv.org/abs/2609.11381](https://arxiv.org/abs/2609.11381)

    该论文提出智能体集成软件（AIS）软件模式与意图级交互抽象（IIA）任务语义，通过交互契约约束任务级交互与应用行为之间的对应关系，并以持续保障机制在依赖变化时维护声明，从而解决将智能体嵌入现有应用时的协调问题。

    

    将智能体嵌入现有应用程序会产生一个持续的协调问题：在委托执行继续进行的同时，用户可以修改目标并操作共享对象。我们认为，可靠的集成需要在任务级交互与应用程序行为之间建立明确的对应关系。我们引入智能体集成软件（AIS）作为一种软件模式，它结合了传统核心、直接交互和内置智能体；并引入意图级交互抽象（IIA）作为任务语义，用户通过它来检查和控制委托的工作。一个开放迁移系统模型将AIS的执行与IIA的状态和事件相关联。交互契约通过任务绑定、角色特定的权限、控制转换和结果证据来约束这种关系；持续保障则在依赖关系发生变化时维护有范围限定的声明。一个紧凑的披露契约和条件命题说明了为什么……

    arXiv:2609.11381v1 Announce Type: new  Abstract: Embedding an intelligent agent in an existing application creates a persistent coordination problem: users can revise goals and manipulate shared objects while delegated execution continues. We argue that dependable integration requires an explicit correspondence between task-level interaction and application behavior. We introduce Agent-Integrated Software (AIS) as a software pattern combining a conventional core, direct interaction, and a built-in agent, and Intent-Level Interaction Abstraction (IIA) as the task semantics through which users inspect and control delegated work. An open transition-system model relates AIS execution to IIA states and events. Interaction contracts constrain this relation through task bindings, role-specific authority, control transitions, and outcome evidence; continuous assurance maintains scoped claims as their dependencies change. A compact disclosure contract and conditional propositions illustrate why
    
[^9]: CoSTAR：面向遗留系统现代化的数据合成驱动的约束感知COBOL节摘要生成

    CoSTAR: Data Synthesis-Driven Constraint-Aware COBOL Section Summarization for Legacy System Modernization

    [https://arxiv.org/abs/2609.11332](https://arxiv.org/abs/2609.11332)

    提出CoSTAR框架，通过执行验证的数据合成与约束感知模型训练相结合，解决COBOL节级代码摘要中的数据稀缺与迁移约束保持两大难题，助力COBOL遗留系统现代化。

    

    COBOL对政府、金融机构和大型企业仍然至关重要；然而，技术老化、专业知识萎缩以及文档缺失，使得基于COBOL的遗留系统现代化日益紧迫。在迁移之前，代码摘要生成是支持遗留系统理解的常见做法。然而，COBOL代码摘要生成，尤其是节级别的摘要生成，面临两个关键挑战：数据稀缺和迁移约束保持。为了解决这些挑战，我们提出了CoSTAR，一个将经过执行验证的数据合成与约束感知模型训练相结合的集成框架。CoSTAR重新利用通用编程任务，通过基于大语言模型（LLM）的生成方式来合成经过执行验证的COBOL代码-摘要数据，以克服数据稀缺问题。基于合成数据，CoSTAR用相关的数据声明和自然语言解释来增强目标节，并使用约束引导的（摘要在此处被截断）

    arXiv:2609.11332v1 Announce Type: new  Abstract: COBOL remains critical to governments, financial institutions, and large enterprises; yet, aging technologies, shrinking expertise, and missing documentation make modernization of COBOL-based legacy systems increasingly urgent. Before migration, code summarization is a common practice to support legacy system understanding. However, COBOL code summarization, especially on section-level, faces two key challenges: data scarcity and migration constraint preservation. To address these challenges, we propose CoSTAR, an integrated framework that combines execution-validated data synthesis with constraint-aware model training. CoSTAR repurposes general-purpose programming tasks to synthesize execution-validated COBOL code-summary data through LLM-based generation to overcome data scarcity. Based on the synthesized data, CoSTAR augments target sections with relevant data declarations and natural-language explanations, and uses constraint-guided 
    
[^10]: 网络物理系统开发中不确定性条件下的不一致性响应面估计

    Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development

    [https://arxiv.org/abs/2609.11331](https://arxiv.org/abs/2609.11331)

    该论文将CPS开发中的模型不一致性问题创新性地转化为干预响应建模问题，结合Saltelli采样与多保真度蒙特卡洛估计训练代理模型，实现了对大范围不确定性空间中不一致性的系统性预测、分析与解释。

    

    网络物理系统（CPS）通常通过多个相互关联的模型来表示。在开发过程中，CPS的一致性要求共享的模型元素在这些模型之间保持兼容。不确定性（例如由传感器噪声或模型抽象引起）会改变模型元素的可接受取值范围，并可能引入不一致性，即模型之间无法再被同时满足的情况。虽然现有方法可以判断给定不确定性配置下的一致性，但它们对系统性地探索、分析和解释大范围不确定性空间中的不一致性所提供的支持有限。我们通过将不一致性重新表述为干预响应建模问题来应对这一挑战。利用Saltelli采样和多保真度蒙特卡洛估计，我们生成干预-响应数据集，并训练一个代理模型，该模型能够直接根据传播的不确定性预测不一致性。

    arXiv:2609.11331v1 Announce Type: new  Abstract: Cyber-Physical Systems (CPS) are commonly represented through multiple interconnected models. During development, CPS consistency requires that shared model elements remain compatible across these models. Uncertainty, for example, due to sensor noise or model abstraction, changes the admissible values of model elements and can introduce inconsistencies, i.e., situations in which models can no longer be jointly satisfied. While existing approaches can determine consistency for a given uncertainty configuration, they provide limited support for systematically exploring, analyzing, and explaining inconsistency across large uncertainty spaces. We address this challenge by reformulating inconsistency as an intervention response modeling problem. Using Saltelli sampling and multi-fidelity Monte Carlo estimation, we generate intervention-response datasets and train a surrogate model that directly predicts inconsistency from the propagated uncer
    
[^11]: 探索安全经验与ChatGPT使用策略在安全软件工程教育中的作用

    Exploring the Role of Security Experience and ChatGPT Usage Strategies on Secure Software Engineering Education

    [https://arxiv.org/abs/2609.11303](https://arxiv.org/abs/2609.11303)

    该研究通过对26名网络安全硕士生的交互日志实证分析发现，ChatGPT使用的多样性（即采用的不同使用模式数量），而非个体使用模式或先前安全经验，与漏洞修复作业的成绩呈正相关。

    

    大语言模型（LLMs）的快速普及正在重塑软件工程教育，但其在安全软件工程教育中的作用仍鲜有探索。我们报告了一项探索性实证研究，考察了某兼职制网络安全理学硕士项目的26名研究生在漏洞修复作业中使用ChatGPT的情况。为了刻画ChatGPT的使用方式，我们采用结构化双重编码程序分析了学生的ChatGPT交互日志，并检验了使用模式以及先前的网络安全专业经验是否与作业成绩相关。结果表明，具有不同网络安全专业水平的学生使用了大体相似的ChatGPT策略。个体使用模式在不同成绩段上表现出描述性差异，但在对多重比较进行校正后，没有任何差异仍具有统计显著性。相反，ChatGPT使用的多样性，即所采用的不同使用模式的数量，与作业表现呈正相关。

    arXiv:2609.11303v1 Announce Type: new  Abstract: The rapid adoption of Large Language Models (LLMs) is reshaping software engineering education, but their role in secure software engineering education remains underexplored. We report an exploratory empirical study of how 26 graduate students in a part-time MSc Cybersecurity programme used ChatGPT during a vulnerability-fixing assignment. To characterise ChatGPT use, we analysed students' ChatGPT interaction logs using a structured double-coding procedure and examined whether usage patterns and prior cybersecurity expertise were associated with assignment performance. The results show that students with varying levels of cybersecurity expertise used broadly similar ChatGPT strategies. Individual usage patterns showed descriptive differences by grade, but none remained statistically significant after correcting for multiple comparisons. In contrast, diversity of ChatGPT usage, i.e., the number of distinct usage patterns adopted, was posi
    
[^12]: AI能否安全地修复后端故障？基于爆炸半径感知沙箱的GuardedAct

    Can AI Remediate Backend Failures Safely? GuardedAct with Blast-Radius-Aware Sandboxing

    [https://arxiv.org/abs/2609.11264](https://arxiv.org/abs/2609.11264)

    提出了GuardedAct框架，通过爆炸半径感知的数字孪生沙箱模拟LLM生成的后端故障修复动作，并利用回滚置信度风险门控只自动执行低风险动作、将高风险动作交由人工审查，从而实现安全的AI自动故障修复。

    

    大型语言模型（LLM）在为微服务故障生成修复动作方面展现出了令人瞩目的能力。然而，直接在生产环境中执行AI生成的修复动作存在引发连锁性附带损害的风险。我们提出了GuardedAct，这是一个沙箱优先的修复框架，它在LLM动作生成器与生产环境之间插入了一个爆炸半径感知的验证层。GuardedAct分为四个阶段运行：（1）接收诊断报告以及实时系统拓扑和近期遥测数据；（2）提示LLM生成按优先级排序的候选修复动作列表；（3）在轻量级数字孪生沙箱中模拟每个动作，估计其爆炸半径并分配风险标签；（4）执行回滚置信度门控，仅自动执行低风险动作，同时将高风险动作上报人工审查。我们在注入了五种故障场景的Deat（摘要在此处截断）……

    arXiv:2609.11264v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown promising capabilities in generating remediation actions for microservice failures. However, directly executing AI-generated repair actions in production risks cascading collateral damage. We propose GuardedAct, a sandbox-first remediation framework that interposes a blast-radius-aware verification layer between the LLM action generator and the production environment. GuardedAct operates in four phases: (1) ingesting a diagnosis report together with the live system topology and recent telemetry, (2) prompting an LLM to produce a ranked list of candidate remediation actions, (3) simulating each action in a lightweight digital-twin sandbox that estimates the blast radius and assigns a risk label, and (4) enforcing a rollback-confidence gate that auto-executes only low-risk actions while escalating high-risk ones for human review. We evaluate GuardedAct on five fault scenarios injected into the Deat
    
[^13]: TripleBound：面向微服务分解的三元组引导异构图学习

    TripleBound: Triplet-Guided Heterogeneous Graph Learning for Microservice Decomposition

    [https://arxiv.org/abs/2609.11212](https://arxiv.org/abs/2609.11212)

    TripleBound提出了一种混合框架，将从包结构、命名约定和代码位置解析出的弱监督三元组约束注入异构图神经网络的共享潜在空间，从而在统一的表示学习目标下联合整合结构依赖与语义相似性信号，实现单体应用到微服务的自动化分解。

    

    云计算和DevOps已使微服务成为构建可扩展、可维护软件系统的常见架构。然而，由于紧耦合和模糊的服务边界，将单体应用迁移到微服务仍然充满挑战。现有的分解方法通常仅依赖结构依赖或语义相似性信号之一，而很少在统一的表示学习目标中整合两者。本文提出了TripleBound，一个用于自动化单体应用到微服务分解的混合框架，该框架通过从基于包结构、命名约定和代码位置的解析器推断出的服务组所导出的弱监督三元组约束，来增强异构图神经网络。TripleBound将基于三元组的约束直接注入共享的结构潜在空间中，使两种信号能够在表示学习过程中被联合优化。结构依赖被捕获……

    arXiv:2609.11212v1 Announce Type: new  Abstract: Cloud computing and DevOps have made microservices a common architecture for scalable, maintainable software systems. However, migrating monoliths to microservices remains challenging due to tight coupling and unclear service boundaries. Existing decomposition approaches typically rely on either structural dependencies or semantic similarity signals, but rarely integrate both within a unified representation learning objective. This paper proposes TripleBound, a hybrid framework for automated monolith-to-microservices decomposition that augments a heterogeneous graph neural network with weakly supervised triplet constraints derived from parser-inferred service groups based on package structure, naming conventions, and code location. TripleBound injects triplet-based constraints directly into the shared structural latent space, enabling both signals to be jointly optimized during representation learning. Structural dependencies are capture
    
[^14]: FST Pay：面向青少年数字支付的确定性安全门控架构

    FST Pay: Deterministic Safety-Gated Architecture for Youth Digital Payments

    [https://arxiv.org/abs/2609.11195](https://arxiv.org/abs/2609.11195)

    提出FST Pay架构，通过在实时支付授权路径上强制实施严格的确定性安全门控，并将AI解释功能解耦至下游处理，从而在不引入概率性AI非确定性风险的前提下保障青少年数字支付安全。

    

    数字支付基础设施日益为青少年用户提供直接访问实时金融服务的机会。虽然早期接触有助于培养金融素养和促进数字包容性，但也使年轻用户面临冲动消费、社会工程诈骗、未经授权交易以及商户欺诈等严重风险。传统的应对措施依赖于概率性机器学习或僵化的静态控制。然而，让概率性或生成式人工智能（AI）模型直接影响实时支付授权，会引入非确定性、不可预测的边缘情况行为以及严重的审计漏洞。本文提出了FST Pay（青少年金融安全支付系统），作为一种架构和形式化规范。FST Pay建立在一个不可变的操作边界之上：在实时授权路径上实施严格的确定性安全门控，并将下游AI解释机制与之解耦。

    arXiv:2609.11195v1 Announce Type: new  Abstract: Digital payment infrastructures increasingly provide adolescent users with direct access to real-time financial services. While early access promotes financial literacy and digital inclusion, it exposes young users to severe risks of impulsive spending, social engineering frauds, unauthorized transactions, and merchant exploitation. Conventional countermeasures rely on probabilistic machine learning or rigid static controls. However, allowing probabilistic or generative artificial intelligence (AI) models to directly influence real-time payment authorization introduces non-determinism, unpredictable edge-case behavior, and critical audit vulnerabilities. This paper introduces Financial Safety for Teens Pay (FST Pay) as an architectural and formal specification. FST Pay is founded on an immutable operational boundary: strict deterministic safety gating on the real-time authorization path coupled with decoupled downstream AI explanation. T
    
[^15]: SemVerBench：基准测试大语言模型对版本约束解析语义的理解

    SemVerBench: Benchmarking LLM Comprehension of Version-Constraint Resolution Semantics

    [https://arxiv.org/abs/2609.11180](https://arxiv.org/abs/2609.11180)

    本文提出首个跨 npm、PEP 440 和 Cargo 三个生态系统的版本约束解析语义基准测试 SemVerBench，通过对 240 个机器可验证项目的评估，发现六个前沿大语言模型在版本约束语义处理上存在系统性盲点，如 Cargo 部分比较器进位规则使所有模型准确率降至约 60%。

    

    大语言模型（LLM）编程智能体经常需要判断某个版本是否满足诸如 ^1.2.3 或 >=2.0,<3 之类的约束，然而它们对版本约束语义的掌握程度从未被直接测量过。我们提出了 SemVerBench，这是首个针对大语言模型版本约束解析语义的基准测试，涵盖三个生态系统（npm、PEP 440、Cargo）：包含 240 个具有唯一答案的机器可验证项目，构建时保持作者中立，来自四个均衡的来源（每个生态系统的官方测试套件加上三个前沿大语言模型提议者），并由非循环的双重实现“神谕”进行标注。通过评估六个前沿模型，我们发现存在系统性的、可预测的按机制划分的盲点：部分比较器的进位规则（>1.2 意味着 >=1.3.0）使所有模型在 Cargo 上受困（准确率接近 60%）；尽管标准的 PEP 440 前缀匹配被普遍掌握，但在零填充/后发布版本的边缘情况上，GPT-5.1 表现崩溃（0/26），而 Claude 保持在 97-100%（在包含 67 个项目的神谕验证集上得到验证）。

    arXiv:2609.11180v1 Announce Type: cross  Abstract: Large language model (LLM) coding agents constantly decide whether a version satisfies a constraint such as ^1.2.3 or >=2.0,<3, yet their grasp of version-constraint semantics has never been measured directly. We introduce SemVerBench, the first benchmark of LLM version-constraint resolution semantics across three ecosystems (npm, PEP 440, Cargo): 240 machine-checkable items with unique answers, built author-neutrally from four balanced sources (each ecosystem's official test suite plus three frontier LLM proposers) and labeled by a non-circular two-implementation oracle. Evaluating six frontier models, we find systematic, predictable per-mechanism blind spots: a partial-comparator carry rule (>1.2 means >=1.3.0) traps every model on Cargo (near 60%), and although standard PEP 440 prefix matching is universal, on zero-pad/post-release corner cases GPT-5.1 collapses (0/26) while Claude stays at 97-100% (verified on a 67-item oracle-vali
    
[^16]: 面向基于DEVS的数字孪生仿真服务的以模型为中心的DevOps架构

    A Model-Centric DevOps Architecture for DEVS-Based Digital Twin Simulation Services

    [https://arxiv.org/abs/2609.11122](https://arxiv.org/abs/2609.11122)

    本文提出一种以模型为中心的DevOps架构，将基于DEVS的数字孪生仿真模型作为一等制品，通过声明式YAML定义、到multiPDEVS的形式化映射、CI/CD流水线中的结构与语义验证以及Kubernetes容器化微服务部署，实现了仿真模型的可追溯版本管理与持续交付。

    

    数字孪生仿真模型像软件一样不断演化和重新部署，然而基于DEVS的引擎虽然提供了健全的形式化基础，但在版本管理、自动化验证以及云原生环境中的持续交付方面支持甚少，导致大多数部署中的模型生命周期管理呈现临时性、缺乏规范的状态。本文提出了一种以模型为中心的DevOps架构，用于将基于DEVS的数字孪生仿真部署为托管服务。仿真模型被视为一等DevOps制品，采用声明式YAML语言定义，并具有到multiPDEVS的形式化映射，支持在CI/CD流水线中进行结构和语义验证，该流水线生成不可变的版本化制品，因此回滚到先前已验证的版本只需固定其标识符即可。该平台被分解为Kubernetes上的容器化微服务，并针对状态外置化和生命周期控制进行了引擎适配。在里加22路公交（案例研究中）……（摘要截断）

    arXiv:2609.11122v1 Announce Type: new  Abstract: Digital twin simulation models are evolved and redeployed like software, yet DEVS-based engines offer a sound formal basis with little support for versioning, automated validation, or continuous delivery in cloud-native environments, leaving model lifecycle management ad hoc in most deployments. This paper proposes a model-centric DevOps architecture for deploying DEVS-based digital twin simulations as managed services. Simulation models are treated as first-class DevOps artefacts defined in a declarative YAML language with a formal mapping to multiPDEVS, supporting structural and semantic validation in a CI/CD pipeline that produces immutable versioned artefacts, so that reverting to an earlier validated version reduces to pinning its identifier. The platform is decomposed into containerised microservices on Kubernetes, with engine adaptations for state externalisation and lifecycle control. An initial case study on the Riga Route 22 pu
    
[^17]: SaltBench：用于在机器校验软件工作中测量方法效应的裁判门控协议

    SaltBench: A Referee-Gated Protocol for Measuring Method Effects in Machine-Checked Software Work

    [https://arxiv.org/abs/2609.11076](https://arxiv.org/abs/2609.11076)

    提出了SaltBench，一个由机器裁判门控的基准测试协议，通过严格的隔离机制（经探针实际检验）、预注册预测和“停止即暂停”等规则，使机器裁判对编程智能体工作方式的影响变得可测量且不可事后叙述。

    

    SaltBench 是一个基准测试协议，旨在回答一个问题：机器裁判如何改变编程智能体的工作方式？机器裁判——即证明内核、程序验证器或保留的测试套件——决定智能体工作的价值，而智能体无法与其争辩。我们在此报告一种协议，使裁判的效应可被测量，且其结果无法在事后被主观叙述：每个结果都由智能体自身工具链之外裁决；智能体与网络、参考解决方案及测试框架本身被完全隔离，并在任何计分运行之前，通过试图突破隔离的探针来检验隔离墙，因此这种隔离是被观察到的而非被假设的；每次运行都由带有注册预测的日期化冻结授权；预算停止意味着暂停，而绝非失败。在本研究中，基准测试的对象是一个“席位”（seat），即标准测试框架中的一次智能体会话。我们测试了五个系统组件，全部由……

    arXiv:2609.11076v1 Announce Type: new  Abstract: SaltBench is a benchmark protocol for one question: How does a machine referee change the way a coding agent works? A machine referee --- a proof kernel, a program verifier, or a withheld test suite --- decides what an agent's work is worth, and the agent cannot argue with it. Here we report a protocol that makes the referee's effect measurable and whose answers cannot be narrated afterwards: every outcome is decided outside the agent's own toolchain; the agent is walled off from the network, the reference solutions and the harness itself, and the wall is tested by probes that try to breach it before any scored run, so the isolation is observed rather than assumed; every run is authorized by a dated freeze with its predictions registered; and a budget stop is a halt, never a failure. In this study, the subject of the benchmark is a ``seat'', meaning an agent session in its standard harness. We tested five systems components, all authored
    
[^18]: 智能体记忆的落地：面向企业智能体的环境探测式记忆管理

    Grounding Agent Memory: Environment-Probing Curation for Enterprise Agents

    [https://arxiv.org/abs/2609.11060](https://arxiv.org/abs/2609.11060)

    该论文提出“环境探测式记忆管理”方法，为智能体的异步记忆管理器赋予最小权限的只读环境工具以验证、限定和刷新候选记忆，无需重训模型即可将CLBench通过率从39%提升至73%。

    

    持久记忆正在进入面向生产的智能体平台，以帮助长时程智能体跨会话积累经验。然而，仅限于已完成轨迹的事后管理智能体可能会保留错误、过度概括局部证据或保留过时的知识。我们提出了环境探测式管理，这是一种兼容部署的扩展方案，为现有的异步记忆管理智能体提供最小权限、只读的世界工具，用于检查、限定范围和刷新候选记忆。该方法无需重新训练模型，且保持任务智能体、检索器、记忆表示和生产写入权限不变。在基于其SDK构建的类生产GitHub Copilot（GHCP）框架中，我们在CLBench数据库探索任务和90个改编的APEX管理咨询任务上比较了无状态执行、完全上下文学习、GHCP + Mem以及GHCP + Mem（带环境探测）四种方案。在CLBench上，环境探测将通过率从39%提升至73%……

    arXiv:2609.11060v1 Announce Type: cross  Abstract: Persistent memory is entering production-oriented agent platforms to help long-horizon agents accumulate experience across sessions. Yet a post-task curator agent restricted to completed trajectories can preserve errors, overgeneralize partial evidence, or retain stale knowledge. We introduce environment-probing curation, a deployment-compatible extension that gives an existing asynchronous curator agent least-privilege, read-only world tools to check, scope, and refresh candidate memories. It requires no model retraining and leaves the task agent, retriever, memory representation, and production write authority unchanged. In a production-like GitHub Copilot (GHCP) harness built on its SDK, we compare stateless execution, full in-context learning, GHCP + Mem, and GHCP + Mem (w/ Env Probing) on CLBench database exploration and 90 adapted APEX management-consulting tasks. On CLBench, probing raises pass rate from 39% to 73% and pass-disc
    
[^19]: BenchShield：面向LLM智能体评估基础设施中奖励完整性的形式化模型支撑的检测工具

    BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure

    [https://arxiv.org/abs/2609.11028](https://arxiv.org/abs/2609.11028)

    本文提出BenchShield，一个基于奖励相关事件有限生命周期形式化模型的检测工具层，通过静态与动态两种互补分析在基准测试基础设施内保障LLM智能体评估的奖励完整性，防范奖励劫持。

    

    语言模型智能体基准测试日益成为交互式评估基础设施。智能体观察状态、调用工具、修改工作区、提交产物，并从结果程序中接收奖励。这种交互性使评估容易受到奖励劫持的攻击：智能体通过利用与奖励相关的轨迹而非解决预期任务来提高其测得的分数。现有的防御措施主要依赖于针对特定任务的补丁、提示指令或事后检测器，无法提供可复用的证据来证明某次具体运行保持在预期的评估边界之内。本文提出了BenchShield，一个面向LLM智能体评估中奖励完整性的模型支撑的检测工具层。BenchShield将检测建立在评估的奖励相关事件的有限生命周期模型之上。在基准测试基础设施内，两种互补的分析在该模型上运行。一种静态的、阶段感知的污点分析……

    arXiv:2609.11028v1 Announce Type: cross  Abstract: LM-agent benchmarks increasingly function as interactive evaluation infrastructure. Agents observe state, call tools, modify workspaces,   submit artifacts, and receive rewards from outcome procedures. This interactivity makes evaluations vulnerable to reward hacking: an agent   improves its measured score by exploiting the reward-relevant trajectory instead of solving the intended task. Existing defenses rely largely   on task-specific patches, prompt instructions, or post-hoc detectors. They do not provide reusable evidence that a concrete run remained   within its intended evaluation boundary. This paper presents BenchShield, a model-backed instrumentation layer for reward integrity in   LLM-agent evaluation. BenchShield grounds detection in a finite lifecycle model of an evaluation's reward-relevant events. Within the   benchmark infrastructure, two complementary analyses operate over this model. A static, phase-aware taint analysi
    
[^20]: RCL：一种用于检测企业级检索增强代码生成中上下文不足的检索-置信度层

    RCL: A Retrieval-Confidence Layer for Detecting Insufficient Context in Enterprise Retrieval-Augmented Code Generation

    [https://arxiv.org/abs/2609.11023](https://arxiv.org/abs/2609.11023)

    该论文提出RCL（检索-置信度层），一个插入在检索与生成之间的轻量级模块，通过结合基于调用图的结构覆盖率分数与置信度信息，在生成开始前检测企业级RAG代码生成中检索上下文是否结构上充分。

    

    检索增强生成（RAG）在代码生成领域已在公开代码库上得到广泛研究，在这些场景下，模型的参数化知识通常能够弥补检索的不完美之处。然而这在企业代码库中会失效，因为私有API、内部框架和未成文的团队规范完全处于任何模型预训练分布之外。最近关于私有库代码生成的研究表明，即使是理想（完美）的检索也无法消除错误，而是将失败定位到API使用环节；此外，基于置信度门控的检索此前已在开放域问答中利用模型内部置信度进行了研究。但这些工作都没有解决在生成开始之前，检索结果本身在结构上是否足以支撑一个私有代码查询的问题。我们提出了RCL（检索-置信度层），一个插入在检索与生成之间的轻量级模块，它将基于调用图导出的结构覆盖率分数与（摘要内容未完整提供）

    arXiv:2609.11023v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) for code generation has been studied extensively on public repositories, where a model's parametric knowledge often compensates for imperfect retrieval. This breaks down in enterprise codebases, where private APIs, internal frameworks, and undocumented team conventions fall entirely outside any model's pretraining distribution. Recent work on private-library code generation shows that even oracle (perfect) retrieval does not eliminate errors, but locates failures downstream in API usage; separately, confidence-gated retrieval has been studied for open-domain question answering using model-internal confidence. Neither addresses whether retrieval itself was structurally sufficient for a private-code query before generation begins. We introduce RCL (Retrieval-Confidence Layer), a lightweight module inserted between retrieval and generation that combines a call-graph-derived structural coverage score with
    
[^21]: DeFiFusion：结合交易事件与智能合约检测价格操纵攻击

    DeFiFusion: Combining Transaction Events with Smart Contracts to Detect Price Manipulation Attacks

    [https://arxiv.org/abs/2609.11008](https://arxiv.org/abs/2609.11008)

    DeFiFusion是一个双模态检测框架，通过联合建模交易事件与智能合约执行语义来检测DeFi中的价格操纵攻击，克服了单纯基于交易或静态合约分析方法各自的根本局限。

    

    去中心化金融（DeFi）已成为一种快速增长的基于区块链的金融服务，其中市场交易动态与底层智能合约逻辑错综复杂地交织在一起。这种自主的相互作用虽然消除了中心化中介，但显著扩大了DeFi协议面对价格操纵攻击（PMAs）的脆弱面，此类攻击已经造成了灾难性的经济损失。尽管问题严重，现有的检测范式存在根本性局限：以交易为中心的方法缺乏对合约执行语义的感知，使其在合法市场波动下容易产生误报；而静态合约分析则忽略真实交易行为，经常报告在实际中无法被利用的漏洞。我们提出了DeFiFusion，这是一种双模态PMA检测框架，通过联合建模交易事件和智能合（约）……

    arXiv:2609.11008v1 Announce Type: cross  Abstract: Decentralized Finance (DeFi) has emerged as a rapidly growing blockchain-based financial service, where market transaction dynamics and underlying smart contract logic are intricately intertwined. This autonomous interplay, while eliminating centralized intermediaries, significantly expands the vulnerability surface of DeFi protocols to Price Manipulation Attacks (PMAs), which have already inflicted catastrophic financial losses. Despite their gravity, existing detection paradigms suffer from fundamental limitations. Transaction-centric methods lack awareness of contract execution semantics, making them prone to false positives under legitimate market volatility, while static contract analyses ignore real transaction behaviors and frequently report vulnerabilities that are infeasible to exploit in practice. We present DeFiFusion, a dual-modal PMA detection framework that closes this gap by jointly modeling transaction events and smart 
    
[^22]: 面向智能体AI的可靠提交门控工程化：共模数据故障下的成本感知验证组合

    Engineering Reliable Commit Gates for Agentic AI: Cost-Aware Verification Portfolios under Common-Mode Data Failures

    [https://arxiv.org/abs/2609.10969](https://arxiv.org/abs/2609.10969)

    该论文发现验证器共享同一证据源时会继承上游故障（共享证据下62.9%的不安全提案被批准，而独立证据源仅为22.9%），据此提出VP-CONTROL成本感知验证组合控制器，仅利用部署可观测元数据选择验证方案，在锁定测试集上将不安全执行率降至1.9%并实现38.2%的自动化安全覆盖率。

    

    智能体系统会执行改变状态的操作，但额外的验证器可能会继承相同的上游故障。我们提出了VP-CONTROL，这是一种面向成本感知提交门控的运行时保障设计和确定性基准测试。其48个任务模板在六种故障机制下生成了2,880个场景。通过固定调用次数的2x2实验，我们将验证器模型多样性与证据源多样性分离开来。在来自两个本地执行者家族的冻结提案上，基于共享证据的跨模型投票批准了62.9%的不安全提案，而使用独立证据源时仅为22.9%。证据源效应达40.9个百分点，而模型多样性效应仅为11.3个百分点。组合控制器仅使用部署可观测的元数据来选择验证方案。在锁定测试集上，以每任务标称5%为目标的近似聚类调整校准实现了1.9%的不安全执行率和38.2%的自动化安全覆盖率。在同等预算条件下，验证组合也优于固定的验证策略。

    arXiv:2609.10969v1 Announce Type: new  Abstract: Agentic systems commit state-changing actions, but additional verifiers can inherit the same upstream fault. We present VP-CONTROL, a runtime-assurance design and deterministic benchmark for cost-aware commit gates. Its 48 task templates yield 2,880 scenarios across six fault regimes. A fixed-call 2 x 2 experiment separates verifier-model diversity from evidence-source diversity. On frozen proposals from two local actor families, a cross-model vote over shared evidence approves 62.9% of unsafe proposals, versus 22.9% with an independent source. The source effect is 40.9 percentage points, compared with 11.3 for model diversity. A portfolio controller selects verification plans using only deployment-observable metadata. Approximate cluster-adjusted calibration at a nominal 5% per-task target yields 1.9% unsafe execution and 38.2% automated safe coverage on the locked test. Matched-budget portfolios also improve on fixed verification polic
    
[^23]: 解耦就绪与释放：面向智能体LLM工作流的尾延迟感知调度

    Decoupling Readiness from Release for Tail-Aware Scheduling of Agentic LLM Workflows

    [https://arxiv.org/abs/2609.10964](https://arxiv.org/abs/2609.10964)

    本文提出一种尾风险感知的智能体LLM工作流轮次释放调度方法，通过均值-CVaR目标联合决策释放时机与未完成工作量预算，将轮次就绪与释放解耦以降低尾延迟。

    

    智能体LLM工作流由模型轮次与工具交互交织而成的序列组成，因此其端到端完成时间不仅取决于推理速度，还取决于就绪轮次何时被释放。大多数运行时会在轮次就绪后立即将其释放。在资源竞争条件下，这种急切的释放策略会不断累积已释放但未完成的工作；而这些轮次一旦提交，就无法再被工作流级别的策略重新排序，从而增加尾延迟。我们提出了一种尾风险感知的轮次释放调度方法，该方法联合决定下一步应释放哪个就绪轮次，以及应维持多少已释放但未完成的工作。该方法使用均值-条件风险价值目标来捕捉未完成工作流不断演变的尾风险，在对就绪轮次进行优先级排序时纳入对轮次工作量的在线估计，并根据观察到的队列压力自适应调整已释放工作预算。我们使用真实的智能体执行对该方法进行了评估。

    arXiv:2609.10964v1 Announce Type: cross  Abstract: Agentic LLM workflows consist of sequences of model turns interleaved with tool interactions, so their end-to-end completion time depends not only on inference speed but also on when ready turns are released. Most runtimes release each turn immediately upon readiness. Under contention, this eager release policy can accumulate released but unfinished work; once submitted, those turns can no longer be reordered by the workflow-level policy, increasing tail latency. We present a tail-risk-aware turn release scheduling method that jointly decides which ready turn to release next and how much released but unfinished work to maintain. The method uses a mean--Conditional Value-at-Risk (CVaR) objective to capture the evolving tail risk of unfinished workflows, incorporates online estimates of turn work when prioritizing ready turns, and adapts the released work budget to observed queue pressure. We evaluate the method using real agent executio
    
[^24]: MCP注册表的随机抽样包含什么，以及工具使用基准测试实际包含了什么

    What a Random Draw from the MCP Registry Contains, and What Tool-Use Benchmarks Contain Instead

    [https://arxiv.org/abs/2609.10962](https://arxiv.org/abs/2609.10962)

    该研究通过对MCP注册表进行可复现的随机概率抽样，首次揭示了真实服务器生态中近半数服务器根本无法启动、安全注释遗漏率高达58.8%，从而证明现行工具使用基准测试所依赖的人工精选样本会系统性高估生态系统的实际可用性与安全性。

    

    对模型上下文协议（MCP）服务器生态系统的研究在抽取样本时，会以各种方式悄悄筛选出能够正常运行的服务器：参考集合、流行度榜单、人工精选框架，或是将服务器修复至能启动为止的流水线。我们报告了未经修复的概率样本实际包含的内容。从一次涵盖24,135个服务器的注册表普查中，我们使用公开的随机种子抽取了400个npm/stdio服务器，并对每个服务器进行了线上探测。只有48.8%的服务器完成了初始化握手，而以相同测量方法对人工精选框架测得的比例为66.7%；主要失败原因并非缺少凭证（13.3%），而是服务器根本无法启动（37.5%）。在195个能够运行的服务器中，硬性合规是完全的：在2,766个声明的工具中，致命的JSON Schema违规为零。可选的安全注释才是真正的差异所在：随机抽样的工具级遗漏率为58.8%，而精选框架为41.5%，说明人工筛选美化了这一数字……

    arXiv:2609.10962v1 Announce Type: new  Abstract: Studies of the Model Context Protocol (MCP) server ecosystem draw their samples in ways that quietly select for servers that work: reference sets, popularity lists, hand-curated frames, or pipelines that repair a server until it starts. We report what an unrepaired probability sample actually contains. From a 24,135-server registry census we draw 400 npm/stdio servers with a published seed and probe each one over the wire. Only 48.8% complete an initialize handshake, against 66.7% for a hand-curated frame measured with the same instrument, and the dominant failure is not missing credentials (13.3%) but servers that never start at all (37.5%). Among the 195 that do run, hard conformance is total: zero fatal JSON Schema violations across 2,766 advertised tools. Optional safety annotations are the real variance, and the tool-level omission rate on a random draw is 58.8% against 41.5% on the curated frame, so curation flatters this figure to
    
[^25]: LLMVul：一个来自真实生产仓库的LLM生成C/C++函数漏洞标注数据集

    LLMVul: A Vulnerability-Labeled Dataset of LLM-Generated C/C++ Functions from Real Production Repositories

    [https://arxiv.org/abs/2609.10945](https://arxiv.org/abs/2609.10945)

    该论文提出了LLMVul数据集，通过从GitHub真实生产仓库中挖掘AI辅助开发活动，构建了包含来自226个仓库的21,430个LLM生成的C/C++函数并带有漏洞标注的数据集，填补了研究真实场景下LLM生成代码安全缺陷的空白。

    

    大语言模型（LLM）越来越多地被用于生成和辅助软件开发，然而现有的漏洞数据集主要集中于人类编写的代码或受控的提示环境。这限制了研究真实软件项目中LLM生成代码的安全缺陷的能力。我们提出了LLMVul，这是一个从真实生产仓库中挖掘的LLM生成C/C++函数的漏洞标注数据集。我们从GitHub上挖掘了2022年11月13日至2026年9月3日这4年间的AI辅助开发活动，使用了提交元数据和AI相关署名证据等来源信号。经过过滤和去重后，LLMVul包含来自226个仓库的21,430个唯一的C/C++函数，以及仓库、提交、函数、来源和AI工具等元数据。我们使用互补的静态分析和基于模式的技术组合来建立漏洞标签。

    arXiv:2609.10945v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to generate and assist with software development, yet existing vulnerability datasets largely focus on human-written code or controlled prompting environments. This limits the ability to study security weaknesses in LLM-generated code as it appears in real-world software projects. We present LLMVul, a vulnerability-labeled dataset of LLM-generated C/C++ functions mined from real production repositories. We mine AI-assisted development activity from GitHub over a 4 year period, from November 13, 2022 to September 3, 2026, using provenance signals such as commit metadata and AI-related authorship evidence. After filtering and deduplication, LLMVul contains 21,430 unique C/C++ functions from 226 repositories, together with repository, commit, function, provenance, and AI-tool metadata. We establish vulnerability labels using an ensemble of complementary static-analysis and pattern-based tec
    
[^26]: AspisAI：一个用于自动化多标准合规监控的规范化、机器可解释的治理框架

    AspisAI: A Canonical, Machine-Interpretable Governance Framework for Automated Multi-Standard Compliance Monitoring

    [https://arxiv.org/abs/2609.10881](https://arxiv.org/abs/2609.10881)

    该论文提出 AspisAI 框架，将 ISO/IEC 27001、NIST CSF 2.0、Cyber Essentials、GDPR 等多个网络安全与隐私标准的要求转化为规范化的机器可解释控制模型，并通过基于条件的决策规则自动评估合规证据，实现可解释、可追溯的自动化多标准合规监控。

    

    在受监管和关键基础设施领域运营的组织必须同时满足多种异构的网络安全与隐私规范要求，包括但不限于 ISO/IEC 27001、NIST 网络安全框架 2.0、Cyber Essentials 以及 GDPR。在实践中，这些合规义务通常通过人工映射、基于电子表格的跟踪和定期审计来管理，这些方式维护成本高昂、在不同标准之间缺乏一致性，且可追溯性较弱。本文提出了 AspisAI，这是一个有界的、与标准无关的治理框架，它将多个框架中选定的需求转化为规范化的、机器可解释的控制模型，并基于条件化的决策规则评估所提交的证据，从而产生可解释、可追溯的合规判定。在包含 26 项代表性需求的有界范围内，该框架在受控仿真环境中针对五个治理（标准）进行了评估。（原文摘要在此处被截断）

    arXiv:2609.10881v1 Announce Type: cross  Abstract: Organisations operating in regulated and critical-infrastructure sectors must satisfy multiple, heterogeneous cybersecurity and privacy instruments simultaneously, including but not limited to ISO/IEC~27001, the NIST Cybersecurity Framework~2.0, Cyber Essentials, and the GDPR. In practice, these obligations are managed through manual mappings, spreadsheet-based tracking, and periodic audits that are costly to maintain, inconsistent across standards, and weak in traceability. This paper presents \emph{AspisAI}, a bounded, standard-agnostic governance framework that translates selected requirements from several frameworks into a canonical, machine-interpretable control model, and evaluates submitted evidence against condition-based decision rules to produce explainable, traceable compliance determinations. Within a bounded scope of 26 representative requirements, the framework is evaluated in a controlled simulation against five governan
    
[^27]: A2ABreak：A2A协议的系统性安全分析

    A2ABreak: Systematic Security Analysis of the A2A Protocol

    [https://arxiv.org/abs/2609.10871](https://arxiv.org/abs/2609.10871)

    本文提出A2ABreak框架，首次对A2A协议进行系统性安全分析，通过LLM辅助从自然语言规范中提取经验证的有限状态机模型（37个状态、76个转换），并利用对抗性验证系统性发现协议层面的安全漏洞。

    

    由Linux基金会管理的Agent2Agent（A2A）协议是一项开放标准，使自主AI智能体能够跨越组织边界相互发现、进行身份认证并委派任务。A2A旨在与用于工具集成的模型上下文协议（MCP）互补，正迅速成为多智能体生态系统的水平通信层。然而，该协议的安全性尚未得到系统性分析。本文提出了A2ABreak，这是对A2A协议首次严谨的系统性安全分析。我们引入了一种新颖的框架，利用大语言模型（LLM）辅助从自然语言规范中直接提取经验证的有限状态机，从929条形式化语句中构建了一个包含37个状态和76个转换的统一模型，随后对该模型进行系统性推理，在完全符合规范的条件下通过对抗性验证来发现协议层面的漏洞。

    arXiv:2609.10871v1 Announce Type: cross  Abstract: The Agent2Agent (A2A) protocol, now governed by the Linux Foundation, is an open standard that enables autonomous AI agents to discover, authenticate with, and delegate tasks to one another across organizational boundaries. Designed to complement the Model Context Protocol (MCP) for tool integration, A2A is rapidly emerging as the horizontal communication layer of the multi-agent ecosystem. Yet the protocol's security has received no systematic analysis.   This paper presents A2ABreak, the first rigorous systematic security analysis of the A2A protocol. We introduce a novel framework that utilizes an LLM-assisted extraction of a verified finite-state machine directly from the natural-language specification, producing a unified model of 37 states and 76 transitions from 929 formalized statements, and then systematically reasons over this model to discover protocol-level vulnerabilities through adversarial verification, under a full-comp
    
[^28]: DR-LabStack：面向临床医生的糖尿病视网膜病变预测网络系统的设计与实现

    DR-LabStack: Design and Implementation of a Clinician-Facing Web System for Diabetic Retinopathy Prediction

    [https://arxiv.org/abs/2609.10796](https://arxiv.org/abs/2609.10796)

    DR-LabStack是一个基于React-Flask、面向临床医生的网络系统，通过统一的界面和后端适配机制集成了四种异构的糖尿病视网膜病变预训练预测模型，解决了不同模型在输入格式、预处理和输出语义上的差异问题。

    

    预训练的糖尿病视网膜病变（DR）预测模型在输入字段、序列化格式、预处理要求和输出语义方面各不相同。因此，要通过统一的临床界面使用这些模型，需要用户界面与推理服务之间进行明确的协调。我们设计并实现了DR-LabStack，这是一个基于React-Flask的网络系统，集成了四个外部开发的预训练模型：RuleFit、剪枝RuleFit、精细化XGBoost和两级集成模型。系统通过一个共享表单获取有序的模型特征，渲染特定于模型的数值和分类控件，并构建位置输入向量。后端适配器加载异构的模型工件并应用集成模型附带的缩放器，同时统一的JSON响应支持二分类结果的展示以及方法和来源信息。2026年9月8日的功能评估使用了复制的应用程序文件和真实模型（摘要内容到此截断）。

    arXiv:2609.10796v1 Announce Type: new  Abstract: Pretrained diabetic retinopathy (DR) prediction models differ in their input fields, serialization formats, preprocessing requirements, and output semantics. Making these models accessible through a common clinical interface therefore requires explicit coordination between the user interface and the inference service. We designed and implemented DR-LabStack, a React-Flask web system integrating four externally developed pretrained models: RuleFit, Pruned RuleFit, Elaborative XGBoost, and Two-level Ensemble. A shared form retrieves ordered model features, renders model-specific numerical and categorical controls, and constructs a positional input vector. Backend adapters load heterogeneous artifacts and apply the ensemble's accompanying scaler, while a common JSON response supports binary classification display alongside method and source information. Functional evaluation on September 8, 2026 used copied application files and real model 
    
[^29]: 超越静态保证：度量安全敏感及LLM生成的Python代码中“静态通过-动态失败”差距

    Beyond Static Guarantees: Measuring the Static-Pass Dynamic-Fail Gap in Security-Sensitive and LLM-Generated Python Code

    [https://arxiv.org/abs/2609.10762](https://arxiv.org/abs/2609.10762)

    该论文首次提出“静态通过-动态失败”（SPDF）现象，并设计了一个融合静态扫描、LLM驱动CWE推理与隔离容器内自主漏洞利用验证的三阶段智能体流水线，以量化静态分析通过但代码在运行时仍可被利用的安全评估盲区。

    

    大语言模型（LLM）的进展推动了可扩展方法的需求，以评估生成代码和安全敏感软件的安全性。静态分析因其可扩展、可复现且成本低而被广泛用作安全把关手段，但它无法直接观察运行时的漏洞利用行为。依赖于对抗性输入、执行上下文或漏洞利用链的漏洞可能逃过静态检查，却在实际中仍可被利用，然而通过静态分析往往被视为代码安全的证据。本文提出了“静态通过-动态失败”（Static-Pass Dynamic-Fail, SPDF）现象，并构建了一个三阶段智能体流水线，结合静态扫描、LLM驱动的常见弱点枚举（CWE）推理，以及在隔离Docker容器中进行的自主漏洞利用验证。研究团队在SecurityEval、RedCode和CyberNative数据集上评估了1,355个Python样本。在复合Bandit-Semgrep门控下未产生任何发现的654个样本中……（摘要在此处截断）

    arXiv:2609.10762v1 Announce Type: cross  Abstract: Advances in large language models (LLMs) fuel the quest for scalable methods to assess the security of generated and security-sensitive software. Static analysis is widely adopted as a scalable, reproducible, and inexpensive security gate, but cannot directly observe runtime exploit behaviour. Vulnerabilities dependent on adversarial inputs, execution context, or exploit chaining may evade static checks while remaining exploitable in practice, yet passing static analysis is often treated as evidence of secure behaviour. This paper introduces the Static-Pass Dynamic-Fail (SPDF) phenomenon and a three-stage agentic pipeline combining static scanning, LLM-driven Common Weakness Enumeration (CWE) reasoning, and autonomous exploit verification in isolated Docker containers. We evaluate 1,355 Python samples from SecurityEval, RedCode, and CyberNative datasets. Of the 654 samples producing no findings under the composite Bandit-Semgrep gate, 
    
[^30]: 迈向临床语言模型的确定性数学求解器

    Towards a Deterministic Math Solver for Clinical Language Models

    [https://arxiv.org/abs/2609.10728](https://arxiv.org/abs/2609.10728)

    本文提出让临床大语言模型不直接进行算术计算，而是生成针对性Python代码交由受限本地执行器作为确定性求解器运行，但在MedCalc-Bench Verified基准上的评估表明，在公式和标准变量均已提供的情况下，这种程序求解接口相比模型直接计算并无可靠优势。

    

    大型语言模型在算术运算方面并不可靠，这对临床计算器而言是一个严重问题，因为单个数值错误就可能改变医疗建议。标准的应对方法是将每个计算器逐一硬编码为经过验证的函数。我们测试了一种替代方案：模型本身不进行计算，而是编写针对具体病例的Python代码，由一个受限的本地执行器作为确定性求解器运行，模型的任务则简化为决定如何使用该求解器。我们在依据现行临床指南对基准测试的公式进行审计、并标记出55个计算器中16个存在版本、用途或系数方面的疑虑之后，使用Qwen2.5-7B和Qwen2.5-32B-AWQ模型，在MedCalc-Bench Verified基准（1,100个病例、55个计算器）上将这种“程序-求解”接口与模型的直接算术运算以及手工编写的22个计算器库进行了对比评估。结果表明，在提供公式和标准变量、且两种方法都能读取完整病历的情况下，将计算任务交给求解器并不能带来可靠的优势。

    arXiv:2609.10728v1 Announce Type: cross  Abstract: Large language models are unreliable at arithmetic, which is a problem for clinical calculators where a single numerical error changes the recommendation. The standard response is to hardcode each calculator as a validated function, one at a time. We test an alternative: the model does not calculate. Instead, it writes case-specific Python that a restricted local executor runs as a deterministic solver, and the model's task reduces to deciding how to use it. We evaluate this Program-Solve interface on MedCalc-Bench Verified (1,100 cases, 55 calculators) against direct model arithmetic and a hand-written 22-calculator library, using Qwen2.5-7B and Qwen2.5-32B-AWQ, after auditing the benchmark's formulas against current clinical guidelines and flagging 16 of 55 with version, use or coefficient concerns. With formulas and gold variables supplied and both routes reading the whole note, handing off to the solver is not a reliable advantage 
    
[^31]: 不确定性下的受治理人机协同优先级排序：自适应估计与依赖约束下的组合选择

    Governed Human-AI Prioritization Under Uncertainty: Adaptive Estimation and Dependency-Constrained Portfolio Selection

    [https://arxiv.org/abs/2609.10648](https://arxiv.org/abs/2609.10648)

    本文提出了一种在不确定性下可治理、可检查且可重新校准的人机协同优先级排序框架，通过五个定量算子（BVS、ERS、PVS、CCS、ODP）实现自适应估计与依赖约束下的组合选择，并用受控合成实验验证了该框架对参数扰动的鲁棒性。

    

    AI原生软件工程日益将人类判断、历史类比、参数化估计和AI生成的预测融合到同一个优先级排序决策中。由此产生的核心问题不仅在于如何对候选工作进行排序，更在于如何以一种可检查、可重新校准的方式治理异构估计、不确定性、战略参数、依赖关系和有限容量。本文研究了D-POAF决策实践中使用的五个定量算子：业务价值评分（BVS）、工作量与风险评分（ERS）、优先级价值评分（PVS）、集体校准评分（CCS）和最优开发路径（ODP）。通过受控合成实验，在已知潜在变量和显式误差过程的条件下刻画了这些算子的行为。实验结果表明，适度的BVS权重扰动能够保持全局排序的稳定性（Spearman相关系数中位数为0.986），而更大范围的战略性变化则使前10%优先项的重叠度降低至0.788。可靠性加权的工作量聚合……

    arXiv:2609.10648v1 Announce Type: new  Abstract: AI-native software engineering increasingly combines human judgment, historical analogy, parametric estimation, and AI-generated forecasts inside the same prioritization decision. The resulting problem is not merely how to rank candidate work, but how to govern heterogeneous estimates, uncertainty, strategic parameters, dependencies, and limited capacity in a way that remains inspectable and recalibratable. We study five quantitative operators used in the D-POAF decision practice: Business Value Score (BVS), Effort and Risk Score (ERS), Prioritization Value Score (PVS), Collective Calibration Score (CCS), and Optimal Development Path (ODP). Controlled synthetic experiments characterize their behavior under known latent variables and explicit error processes. Moderate BVS-weight perturbations preserved global rankings (median Spearman 0.986), while broader strategic changes reduced top-10% overlap to 0.788. Reliability-weighted effort agg
    
[^32]: Numbat：构建并验证一个自包含的机器学习技术栈

    Numbat: Building and Verifying a Self-Contained Machine-Learning Stack

    [https://arxiv.org/abs/2609.10632](https://arxiv.org/abs/2609.10632)

    Numbat是一个完全用Zig语言编写、零第三方运行时依赖的自包含机器学习技术栈，覆盖从张量计算到多GPU训练的完整功能，并通过稳定的C ABI接口和可执行的验收门来编码监管要求并实现验证。

    

    机器学习系统几乎完全构建在少数几个由Python编排的大型框架之上，并继承了这些技术栈的工程成本：包含数百个版本相互耦合的软件包的环境、用于部署的独立导出工具链，以及研究所用语言与产品发布所用语言之间的割裂。我们报告了numbat的构建与验证工作——numbat是一个完全用一种通用语言编写、无任何第三方运行时依赖的机器学习技术栈。该技术栈涵盖张量计算、自动微分、神经网络模块、混合精度、多GPU训练、数据加载与监控；一个SDK通过稳定且采用增量式版本管理的C ABI（超过1400个入口点）将其暴露出来，并提供六种语言的绑定；其临床领域平面将监管要求编码为可执行的验收门，而非单纯的文档。验证这样一个技术栈是更困难的一半工作。

    arXiv:2609.10632v1 Announce Type: cross  Abstract: Machine-learning systems are built almost exclusively on a few large Python-orchestrated frameworks, and they inherit those stacks' engineering costs: environments of hundreds of version-coupled packages, separate export toolchains for deployment, and the split between the language research is written in and the language products ship in. We report on the construction and verification of numbat, a machine-learning stack written in one general-purpose language (Zig) with no third-party runtime dependencies. The stack spans tensor computation, automatic differentiation, neural-network modules, mixed precision, multi-GPU training, data loading and monitoring; an SDK exposes it behind a stable, additively versioned C ABI of over 1,400 entry points, with bindings for six languages; and its clinical domain planes encode regulatory requirements as executable acceptance gates rather than documentation. Verifying such a stack is the harder half
    
[^33]: AI安全：不是可选项，也不能再等

    AI Safety: Not Optional, Not Later

    [https://arxiv.org/abs/2609.10630](https://arxiv.org/abs/2609.10630)

    论文提出一种“安全即设计”的多层次AI安全保证架构，通过结合模型级监督与系统级控制，并辅以独立验证、监控、证据基础设施和治理机制，全面保障AI系统安全。

    

    各类事故表明，AI安全失效往往跨越多个层面同时发生。我们提出了一种“安全即设计”的保证架构，将模型层面的监督（如Scientist AI）与系统层面对脚手架和工具框架的控制相结合，并辅以独立验证、监控和证据基础设施，同时由治理机制提供问责制和证据互操作性支持。

    arXiv:2609.10630v1 Announce Type: new  Abstract: Incidents show that AI safety failures often arise across multiple layers. We present a safety-by-design assurance architecture combining model-level supervision, such as Scientist AI, with system-level controls over scaffolds and harnesses, independent verification, monitoring, and evidence infrastructure, supported by governance for accountability and evidence interoperability.
    
[^34]: 关于代码质量与机器学习性能之间关系的研究：一项大规模实证研究

    On the Relation between Code Quality and Machine Learning Performance: A Large-scale Empirical Study

    [https://arxiv.org/abs/2609.10610](https://arxiv.org/abs/2609.10610)

    本研究对265,363个Kaggle竞赛Python笔记本进行了大规模实证分析，以探究代码质量与机器学习性能之间的关系，并评估受欢迎程度和作者专业度等社交信号能否作为代码质量或模型性能的可靠指标。

    

    背景：计算笔记本是机器学习（ML）开发的标准环境。在机器学习社区中，模型性能往往是首要考虑的指标，而代码质量被视为次要问题。这种优先级排序依赖于一个在很大程度上未经检验的假设，即代码质量与机器学习性能无关。此外，从业者还会复用现有代码，这些代码可能来自通过社交信号（受欢迎程度、作者专业度）筛选出的笔记本，而这些信号作为质量替代指标的可靠性从未被评估过。目标：我们实证研究了笔记本中代码质量与机器学习性能之间的关系，并评估了受欢迎程度和作者专业度是否能作为代码质量或性能的指示指标。方法：我们对提交至Kaggle竞赛的265,363个Python笔记本进行了大规模实证研究。我们使用两个静态分析工具评估代码质量：Pylint，用于捕获一般性代码缺陷……

    arXiv:2609.10610v1 Announce Type: cross  Abstract: Context: Computational notebooks are the standard environment for machine learning (ML) development. Within the ML community, model performance is often the primary considered metric, and code quality is treated as a secondary concern. This prioritization relies on a largely untested assumption that code quality and ML performance are unrelated. Practitioners also reuse existing code that may come from notebooks selected through social signals (popularity, author expertise) whose reliability as quality proxies has never been assessed. Objective: We empirically investigated the relationship between code quality and ML performance in notebooks, and evaluated whether popularity and author expertise give indication on code quality or performance. Method: We conducted a large-scale empirical study of 265,363 Python notebooks submitted to Kaggle competitions. We assessed code quality with two static analysis tools: Pylint, capturing general 
    
[^35]: 生成式AI助力可信系统——迈向健康检查模型

    Generative AI for trustworthy systems - Towards a health check model

    [https://arxiv.org/abs/2609.10595](https://arxiv.org/abs/2609.10595)

    本文基于对十八位跨行业资深从业者的访谈研究，提出了一个包含系统层与组织层共八个维度的“可信自主性健康检查模型”，用于多维度刻画组织如何在生成式AI辅助的软件工程中建立信任。

    

    生成式AI在软件密集型系统中的应用正在快速推进，但目前的分析工具——主要是一维成熟度模型——将重要的配置性差异压缩到一个单一的渐进轴上。基于对来自电信、汽车、国防、航空、银行、能源、政府和企业软件服务等行业的十八位资深从业者的归纳式访谈研究，本文提出了可信自主性健康检查模型：一个结构化的、多维度的工具，用于刻画组织如何在生成式AI辅助的软件工程中建立信任。该模型将八个基于实证的维度组织为系统层（智能体权限范围、保障机制、数据可信性、架构隔离、可追溯性与可理解性）和组织层（治理、人类监督姿态、劳动力能力可持续性）。

    arXiv:2609.10595v1 Announce Type: new  Abstract: The adoption of generative AI in software-intensive systems is proceeding rapidly, but current analytical instruments - principally unidimensional maturity models - compress important configurational variation into a single progressive axis. Drawing on an inductive interview study of eighteen senior practitioners across telecommunications, automotive, defence, aviation, banking, energy, government, and enterprise software services, this paper presents the Trustworthy Autonomy Health Check Model: a structured, multidimensional instrument for characterizing how organizations establish trust in GenAI-assisted software engineering. The model organizes eight empirically grounded dimensions into a system layer (Scope of Agent Authority, Assurance Mechanisms, Data Trustworthiness, Architectural Containment, Traceability & Comprehensibility) and an organizational layer (Governance, Human Oversight Posture, Workforce Capability Sustainability), e
    
[^36]: ReqEvolve：通过自动需求解释实现面向用户的软件自演化

    ReqEvolve: User-Oriented Software Self-Evolution through Automatic Requirement Interpretation

    [https://arxiv.org/abs/2609.10590](https://arxiv.org/abs/2609.10590)

    ReqEvolve是一个运行时代码生成系统，通过融合自动需求工程与测试驱动开发，将用户的高层次自然语言请求直接转化为可执行功能，实现了无需开发人员介入的用户驱动软件自演化。

    

    软件自演化范式使系统能够在运行过程中自主地扩展和重新配置自身能力，以响应技术规范。然而，新功能的请求通常来自最终用户，且很少以技术术语表达。因此，开发人员必须先将用户需求转化为技术规范，系统才能进行演化，这使用户无法立即观察所请求功能的行为，从而延迟了该功能的早期验证。为了解决这一差距，我们提出了ReqEvolve，这是一个运行时代码生成系统，通过接受高层次的用户请求来实现用户驱动的自演化。该系统集成了自动需求工程（RE）和测试驱动开发（TDD），通过需求澄清、规范分解、测试生成和运行时集成，将这些请求转化为可执行的功能。我们对ReqEvolve进行了评估

    arXiv:2609.10590v1 Announce Type: new  Abstract: The paradigm of software self-evolution enables systems to autonomously extend and reconfigure their own capabilities during execution in response to technical specifications. Yet requests for new functionality often originate from end users and are rarely expressed in technical terms. As a result, developers must translate user needs into technical specifications before the system can evolve, delaying early validation of the requested functionality by preventing users from immediately observing the resulting behaviour. To address this gap, we present ReqEvolve, a runtime code generation system that enables user-driven self-evolution by accepting high-level user requests. The system integrates automatic requirements engineering (RE) and test-driven development (TDD) to transform these requests into executable functionality through clarification, specification decomposition, test generation, and runtime integration. We evaluate ReqEvolve 
    
[^37]: 跨部署栈的AI推理优化

    Optimizing AI Inference Across the Deployment Stack

    [https://arxiv.org/abs/2609.10550](https://arxiv.org/abs/2609.10550)

    本文提出了一个统一的分析框架，将模型级技术、编译器转换和系统策略三个层面的推理优化方法纳入三层分类体系，并将部署表述为多目标优化问题，为跨部署栈的AI推理优化提供了系统性理论分析。

    

    AI部署性能不仅由模型架构决定，还受到压缩、编译器转换和服务策略之间相互作用的影响。已发表的基准测试通常在不可比较的条件下报告延迟和吞吐量，限制了其在部署决策中的应用。本文对整个部署栈的推理优化进行了统一的分析处理。我们引入了一个三层分类体系，涵盖模型级技术（如量化、剪枝和蒸馏）、编译器转换（如图融合、布局优化和内核自动调优）以及系统策略（如动态批处理、准入控制和内存分层）。我们将部署问题表述为在准确性、延迟、吞吐量、内存占用和能耗等方面的约束多目标优化问题，并分析了具有帕累托单调性和尺度不变性的部署排序泛函。Roofline模型展示了如何……

    arXiv:2609.10550v1 Announce Type: cross  Abstract: AI deployment performance is shaped not by model architecture alone, but by interactions among compression, compiler transformations, and serving policies. Published benchmarks often report latency and throughput under incomparable conditions, limiting their use for deployment decisions. This paper presents a unified analytical treatment of inference optimization across the deployment stack. We introduce a three-layer taxonomy covering model-level techniques such as quantization, pruning, and distillation; compiler transformations such as graph fusion, layout optimization, and kernel autotuning; and system policies such as dynamic batching, admission control, and memory tiering. We formulate deployment as a constrained multi-objective optimization problem over accuracy, latency, throughput, memory footprint, and energy, and analyze a deployment-ranking functional with Pareto monotonicity and scale invariance. Roofline models show how m
    
[^38]: 当通过测试掩盖了漏洞：智能体系统中静默失败的实证研究

    When Passing Tests Hides Vulnerabilities: An Empirical Study of Silent Failures in Agentic Systems

    [https://arxiv.org/abs/2609.10548](https://arxiv.org/abs/2609.10548)

    该研究通过分析七个智能体框架的1,030条执行轨迹，首次系统性地识别并分类了LLM代码修复中“通过测试却仍存在漏洞”的静默失败，归纳出遗漏、引入和不足三大类问题。

    

    基于大语言模型（LLM）的自动化代码修复智能体近年来在研究和软件工程实践领域都受到了极大关注。然而，对于那些通过语法和功能验证、但仍然保留或引入了安全漏洞的补丁，人们给予的关注却十分有限。本研究旨在系统地识别和分类基于LLM的智能体代码修复中存在的此类“静默失败”。我们使用七个智能体框架配合GPT-4o-mini，在SecurityEval和CVEfixes两个安全导向的数据集上产生了1,030条有效执行轨迹，并据此开展了实证研究。经过三轮定性编码和人工验证，共确认了170个静默失败。关键结果包括：（i）识别出三大类静默失败：遗漏、引入和不足。其中遗漏占已确认失败的48.2%，引入占30.6%……

    arXiv:2609.10548v1 Announce Type: new  Abstract: LLM-based agents for automated code repair have received significant attention in recent years from both research and software engineering practice perspectives. However, limited attention has been paid to patches that pass syntactic and functional verification but still retain or introduce security vulnerabilities. The aim of this research is to systematically identify and categorize such silent failures in LLM-based agentic code repair.   We conducted an empirical study using 1,030 valid execution traces produced by seven agent frameworks with GPT-4o-mini across two security-focused datasets, SecurityEval and CVEfixes. Through three iterations of qualitative coding and manual verification, 170 confirmed silent failures were identified. The key results are: (i) Three main categories of silent failures were identified: Omission, Introduction, and Inadequacy. Omission accounts for 48.2% of the confirmed failures, Introduction for 30.6%, a
    
[^39]: KG-Commit：一种用于在线即时软件缺陷预测的动态知识图谱

    KG-Commit: A Dynamic Knowledge Graph for Online Just-in-Time Software Defect Prediction

    [https://arxiv.org/abs/2609.06272](https://arxiv.org/abs/2609.06272)

    本文提出KG-Commit动态知识图谱，通过增量维护代码库历史、文件内代码结构和提交语义，结合AST差异机制与完全基于CPU的轻量级图推理，实现在线即时软件缺陷预测，在11个Apache项目上取得最优综合性能。

    

    即时软件缺陷预测（JIT-SDP）旨在在风险提交到达时识别它们，并为开发者提供及时的反馈。这种低延迟需求导致大多数方法仅依赖提交级别的信息，而忽略了变更发生时更广泛的项目上下文。纳入这种上下文具有挑战性，因为它既需要对传入提交进行高效检索，又需要在代码库演变过程中进行持续维护。我们提出了KG-Commit，一个动态知识图谱，它随着项目的演变增量地维护代码库历史、文件内代码结构和提交语义。它还使用AST差异（AST-delta）机制来跟踪文件编辑之间的结构变化，并依赖于完全在CPU上运行的轻量级图推理。我们在11个Apache软件项目上与六个基线方法进行的评估表明，KG-Commit取得了最高的总体Macro-F1（0.704）、G-Mean（0.706）和AUC（0.809）。

    arXiv:2609.06272v2 Announce Type: replace  Abstract: Just-in-time software defect prediction (JIT-SDP) aims to identify risky commits as they arrive and provide developers with timely feedback. This need for low latency has led most approaches to rely on commit-level information and overlook the broader project context in which a change occurs. Incorporating this context is challenging because it requires both efficient retrieval for incoming commits and continual maintenance as the repository evolves. We introduce KG-Commit, a dynamic knowledge graph that incrementally maintains repository history, within-file code structure, and commit semantics as the project evolves. It also uses an AST-delta mechanism to track structural changes between file edits and relies on lightweight graph inference running entirely on CPU. Our evaluation on 11 Apache software projects against six baselines shows that KG-Commit achieves the highest aggregate Macro-F1 (0.704), G-Mean (0.706), and AUC (0.809) 
    
[^40]: FaultLens：为生成的操作程序学习紧凑的行为测试套件

    FaultLens: Learning Compact Behavioral Test Suites for Generated Operational Programs

    [https://arxiv.org/abs/2608.26746](https://arxiv.org/abs/2608.26746)

    本文提出FaultLens方法，通过结合故障驱动的贪婪选择和突变无关的多样性组件，学习紧凑的行为测试套件，以高效检测生成程序中的稀疏边界和交互故障。

    

    arXiv:2608.26746v1 公告类型：交叉 摘要：生成的操作程序通常通过少量手写示例或全面的回归测试套件进行验证。前者可能遗漏稀疏的边界和交互故障，而后者可能不必要地昂贵。我们引入了FaultLens，一种学习紧凑行为测试套件的方法，同时保持与执行证据的可审计联系。它执行一次丰富的探针域，将故障-探针杀死关系存储为稀疏结果缓存，并仅从早期程序生成中学习探针排序。一个故障驱动的贪婪组件利用已知的杀死结构，而一个与突变无关的多样性组件覆盖探针族、案例、模板和时间箱。它们的交替混合方法在新程序包含排序构建中不存在的故障机制时仍然有用。我们评估了四个环境中的二十个生成操作策略，十个执行种子，以及1,200个测量值。

    arXiv:2608.26746v1 Announce Type: cross  Abstract: Generated operational programs are often validated with either a few hand-written examples or exhaustive regression suites. The former can miss sparse boundary and interaction faults, while the latter can be unnecessarily expensive. We introduce FaultLens, a method for learning compact behavioral test suites while preserving an auditable connection to executed evidence. It executes a rich probe domain once, stores the fault-probe kill relation as a sparse outcome cache, and learns probe orderings only from earlier program generations. A fault-driven greedy component exploits known kill structure, while a mutation-independent diversity component covers probe families, cases, templates, and temporal bins. Their alternating hybrid remains useful when a new program contains a fault mechanism absent from ordering construction.   We evaluate twenty generated operational policies across four environments, ten execution seeds, 1,200 measured r
    
[^41]: 过程监控预测的因果解释

    Causal Explanations of Process Monitor Predictions

    [https://arxiv.org/abs/2608.24672](https://arxiv.org/abs/2608.24672)

    本文提出了一种基于实际因果框架的新方法，通过定义捕获事件时间依赖性的因果模型，为过程监控预测生成局部解释，并量化事件对预测结果的影响。

    

    过程挖掘被广泛用于诊断过程并识别性能和合规性问题。具体而言，预测性过程监控（PPM）技术使用AI模型来预测正在运行的过程实例的结果。尽管这些模型能够实现较高的预测性能，但其黑盒特性使得难以理解输出预测背后的潜在原因。在本文中，我们提出了一种基于实际因果框架的新方法，用于生成过程监控预测的局部（案例级）解释。我们定义了一个针对过程的因果模型，该模型捕获了轨迹中事件之间的时间依赖关系，从而使我们能够推理事件对预测结果的因果影响。我们的方法隐式地使用该模型来计算原因，并量化不同事件相对于预测结果的重要性。我们提出了一种实用的、模型无关的算法。

    arXiv:2608.24672v1 Announce Type: new  Abstract: Process mining is widely used to diagnose processes and identify performance and compliance issues. Specifically, Predictive Process Monitoring (PPM) techniques use AI models to predict outcomes of ongoing process instances. While these models can achieve high predictive performance, their black-box nature makes it difficult to understand the underlying reasons behind their output predictions. In this paper, we propose a novel approach for generating local (case-level) explanations of process monitor predictions based on the framework of actual causality. We define a causal model tailored to processes that captures temporal dependencies between events in a trace, thus allowing us to reason about causal influence of events on the predicted outcome. Our method uses this model implicitly to compute causes and quantify the importance of different events with respect to the predicted outcome. We present a practical, model-agnostic algorithm t
    
[^42]: GitSkills：GitHub上的智能体技能数据集

    GitSkills: A Dataset of Agent Skills on GitHub

    [https://arxiv.org/abs/2608.10906](https://arxiv.org/abs/2608.10906)

    本文提出了GitSkills数据集，收录了GitHub上3,797,117个智能体技能文件，为首次实证研究开发者如何编写、复用和维护基于自然语言的智能体技能提供了数据基础。

    

    智能体技能是一个包含SKILL.md文件的文件夹，其中含有面向语言模型智能体的指令，并可选择性地附带脚本和参考文件。当智能体判断某项任务与技能描述匹配时，便会加载该技能。Anthropic于2025年10月将该格式作为开放规范推出。九个月后，公开的GitHub仓库中已存在数百万个技能文件。技能不同于软件工程研究人员通常挖掘的软件制品：它们主要以自然语言编写，由模型在运行时以概率方式选择，且没有编译器或类型检查器来验证选择过程。技能也没有中央注册表或包管理器；开发者通过在仓库之间复制文件夹来复用技能。因此，开发者如何编写、复用和维护技能成为一个实证问题，而现有数据集尚未记录这一群体。我们提出了GitSkills，一个包含3,797,117个SKILL文件的数据集。

    arXiv:2608.10906v2 Announce Type: replace  Abstract: An agent skill is a folder containing a $\mathrm{SKILL.md}$ file with instructions for a language-model agent, optionally accompanied by scripts and reference files. The agent loads the skill when it judges that a task matches the skill description. Anthropic introduced the format in October 2025 as an open specification. Nine months later, public GitHub repositories hold millions of skill files. Skills are unlike the artifacts that software engineering researchers usually mine: they are written mainly in natural language, a model selects them probabilistically at run time, and no compiler or type checker verifies the selection. Skills also have no central registry or package manager; developers reuse them by copying folders between repositories. How developers write, reuse, and maintain skills is therefore an empirical question, and no existing dataset records this population. We present GitSkills, a dataset of 3,797,117 $\mathrm{SK
    
[^43]: Rust Coreutils：用现代语言重建Unix基础

    Rust Coreutils: Rebuilding Unix Foundations in a Modern Language

    [https://arxiv.org/abs/2608.07135](https://arxiv.org/abs/2608.07135)

    本文介绍了用Rust语言重新实现GNU coreutils的项目，该实现已可在大多数Linux发行版上作为即插即用的替代品，展示了现代编程语言如何为遗留关键软件带来可靠的现代化改造。

    

    GNU核心工具（coreutils）是现代UNIX系统中一个至关重要的软件包。它包含约100个基础命令——如ls、cp和cat——每天在数百万台计算机上运行。然而，GNU coreutils也是遗留软件，其C代码库可追溯到20世纪90年代初，并且可以说功能已经完备。如果考虑重新实现这个重要软件包，如何有效地做到这一点，以及为什么要这样做？本文回顾了Rust coreutils的开发历程，这是一个用Rust编程语言对GNU coreutils进行的现代化开源重实现，它已达到可即插即用替代GNU coreutils的地位，并与大多数Linux发行版兼容。通过比较Rust coreutils与其前身，我们为创建关键软件的可靠替代品提供了见解，并强调了现代编程特性如何能够吸引对遗留软件包的开发兴趣。

    arXiv:2608.07135v2 Announce Type: replace  Abstract: GNU core utilities (coreutils) is a crucial package in modern UNIX systems. It comprises around 100 fundamental commands---like ls, cp, and cat---which run every day on millions of computers. However, GNU coreutils is also legacy software, with its C codebase dating back to the early 1990s and arguably feature-complete. If one were to consider reimplementing this essential package, how would they do so effectively, and why? This paper recounts the development of Rust coreutils, a contemporary open source reimplementation of GNU coreutils in the Rust programming language, which has reached the status of a drop-in replacement for GNU coreutils, compatible with most Linux distributions. By comparing Rust coreutils with its ancestor, we offer insights into creating a reliable substitute for critical software and highlight how modern programming features can attract development interest in legacy packages.
    
[^44]: 面向大语言模型智能体的执行优先合成工具使用轨迹生成

    Execution-First Synthetic Tool-Use Trace Generation for LLM Agents

    [https://arxiv.org/abs/2607.29175](https://arxiv.org/abs/2607.29175)

    提出执行优先框架SyntheticAgentTraceQA，通过先构建、执行并验证工具调用轨迹再合成用户任务，解决了传统查询优先数据合成无法保证工具交互有效性的问题。

    

    智能体软件工程和工业系统越来越多地通过可执行工作流而非仅仅代码生成来运行：它们搜索工件、调用工具、检查结构化观察结果并查询数据库。训练这些智能体需要能够捕获有效工具交互和可执行工作流的监督数据。然而，传统的“查询优先”数据合成方法可能会失败，因为看似合理的用户请求可能并不对应有效的工具序列、兼容的参数或可用的数据。为了解决这一局限性，我们提出了SyntheticAgentTraceQA，一个用于为工具增强型智能体生成可扩展监督数据的执行优先框架。我们的框架首先构建高层工作流结构，通过依赖感知的分配将其映射到可用工具，在受控环境中执行并验证所得轨迹，然后才合成自然语言用户任务与教师信号。

    arXiv:2607.29175v2 Announce Type: replace  Abstract: Agentic software-engineering and industrial systems increasingly operate through executable workflows rather than code genera- tion alone: they search artifacts, invoke tools, inspect structured observations, and query databases. Training these agents requires supervision data that captures valid tool interactions and executable workflows. However, traditional query-first data synthesis can fail because plausible user requests may not correspond to valid tool sequences, compatible parameters, or available data. To address this limitation, we propose SyntheticAgentTraceQA, an execution- first framework for generating scalable supervision data for tool- augmented agents. Our framework first constructs high-level work- flow structures, maps them to available tools through dependency- aware assignment, executes and validates the resulting traces in con- trolled environments, and only then synthesizes natural-language user tasks, teacher-
    
[^45]: 评估语言模型在显著类识别中的能力

    Assessing Language Models for Salient Class Identification

    [https://arxiv.org/abs/2606.21629](https://arxiv.org/abs/2606.21629)

    本研究构建了包含7,911个提交的新数据集ApacheJavaCM，并评估语言模型能否无需特征工程、图构建或训练，直接从代码提交中识别出显著类。

    

    代码审查要求审查者理解代码变更的核心意图，而当一个提交修改多个类时，这一任务变得困难。在此类提交中，一个或多个主要被修改的类（称为显著类）可能会引发其他类的修改。准确识别显著类能够为审查者提供一个有效的切入点来浏览代码变更，并促进程序理解。现有最先进的方法依赖于复杂的程序分析流程，包括抽象语法树（AST）解析、类关系提取、手工特征工程或依赖图构建。为此，我们研究语言模型（LMs）能否在无需特征工程、图构建或训练的情况下，直接从提交中识别显著类。我们首先构建了一个新数据集ApacheJavaCM，该数据集源自ApacheCM数据集，包含7,911个提交和25,914个带标签的类。

    arXiv:2606.21629v2 Announce Type: replace  Abstract: Code review requires reviewers to understand the core intent of code changes, which becomes difficult when a commit modifies multiple classes. In such commits, one or more primarily modified classes, referred to as salient classes, may induce modifications in other classes. Accurate identification of salient classes offers reviewers an effective entry point to navigate code changes and facilitates program comprehension. Existing state-of-the-art approaches rely on complex program-analysis procedures, including Abstract Syntax Tree (AST) parsing, class relation extraction, handcrafted feature engineering, or dependency graph construction. To this end, we study whether language models (LMs) can identify salient classes directly from commits without feature engineering, graph construction, or training. We first construct a new dataset ApacheJavaCM, derived from the ApacheCM dataset, containing 7,911 commits and 25,914 labeled classes. O
    
[^46]: 基于大语言模型引导策略选择的自适应证明精炼

    Adaptive Proof Refinement with LLM-Guided Strategy Selection

    [https://arxiv.org/abs/2510.25103](https://arxiv.org/abs/2510.25103)

    提出了Adapt框架，利用LLM引导的决策器根据证明助手状态和错误证明的上下文动态选择最合适的精炼策略，突破了现有方法固定策略的局限，提升了自动定理证明的性能。

    

    基于定理证明的形式化验证能够对软件正确性进行富有表现力的规约和严格证明，但由于需要大量的人工投入和专业知识，难以规模化。尽管大语言模型（LLM）在证明生成方面展现出潜力，但它们在首次尝试时经常产生错误的证明，需要额外的策略进行迭代精炼。然而，现有方法采用固定的精炼策略，无法根据生成证明中的具体问题动态选择有效的策略，这限制了其性能。为克服这一局限，我们提出了Adapt，一种新颖的证明精炼框架，它利用LLM引导的决策器，根据证明助手的状态和错误证明的可用上下文，动态选择合适的精炼策略。我们在两个基准测试上将Adapt与四种现有方法进行了评估对比。

    arXiv:2510.25103v2 Announce Type: replace  Abstract: Formal verification via theorem proving enables the expressive specification and rigorous proof of software correctness, but it is difficult to scale due to the significant manual effort and expertise required. While Large Language Models (LLMs) show potential in proof generation, they frequently produce incorrect proofs on the first attempt and require additional strategies for iterative refinement. However, existing approaches employ fixed refinement strategies and cannot dynamically choose an effective strategy based on the particular issues in a generated proof, which limits their performance. To overcome this limitation, we introduce Adapt, a novel proof refinement framework that leverages an LLM-guided decision-maker to dynamically select a suitable refinement strategy according to the state of the proof assistant and available context of an incorrect proof. We evaluate Adapt on two benchmarks against four existing methods and 
    
[^47]: 移动应用研究中与GDPR相关的隐私问题：一项系统性文献综述

    GDPR-Relevant Privacy Concerns in Mobile Apps Research: A Systematic Literature Review

    [https://arxiv.org/abs/2411.19142](https://arxiv.org/abs/2411.19142)

    本文通过系统性文献综述，首次对移动应用领域GDPR相关隐私问题的现有研究进行了描述、分析和分类，填补了该领域缺乏二次研究的空白。

    

    《通用数据保护条例》（GDPR）被视为欧盟（EU）隐私和数据保护标准的基准。早在其2018年生效之前，软件工程（SE）文献中就已开展了大量研究，探讨GDPR隐私需求的获取、表示和验证。世界上任何地方部署的软件系统，只要处理欧盟居民的个人数据，就必须遵守GDPR。移动应用程序（apps）在这方面也不例外。随着移动应用的日益普及及其对个人数据需求的不断增长，隐私问题在软件工程界引起了更多关注。尽管关于移动应用中GDPR相关隐私问题的文献十分丰富，但目前尚无描述、分析和归类当前研究重点的二次研究。因此，研究空白和持续存在的挑战尚未得到解决……

    arXiv:2411.19142v4 Announce Type: replace  Abstract: The General Data Protection Regulation (GDPR) is considered as the benchmark in the European Union (EU) for privacy and data protection standards. Since before its entry into force in 2018, substantial research has been conducted in the software engineering (SE) literature investigating the elicitation, representation, and verification of GDPR privacy requirements. Software systems deployed anywhere in the world must comply with GDPR as long as they handle personal data of EU residents. Mobile applications (apps) are no different in that regard. With the growing pervasiveness of mobile apps and their increasing demand for personal data, privacy concerns have acquired further interest within the SE community. Despite the extensive literature on GDPR-relevant privacy concerns in mobile apps, there is no secondary study that describes, analyzes, and categorizes the current focus. Research gaps and persistent challenges are thus left unn
    

