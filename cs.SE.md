# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SoK: Assessing the Evolution of LLM Capability in Software Vulnerability Research](https://arxiv.org/abs/2611.26601) | 本文提出一个借鉴自动驾驶分级的自主性评级框架，系统化梳理了LLM在漏洞研究中的能力证据，发现2026年前智能体已能在无人工干预下发现、验证并修补真实软件漏洞，开源权重模型随后也达到了相当的自主性水平。 |
| [^2] | [Verify the Verifier: Specification-Grounded LLM Cross-Checking for Bugs That Escape UVM Scoreboards](https://arxiv.org/abs/2611.26588) | 提出首个基于规范、利用LLM交叉检查已通过回归测试追踪记录的框架"Verify the Verifier"，从564个注入缺陷中确认了41个逃过UVM记分板检查的缺陷。 |
| [^3] | [Backtracking Agent Applications Security Incidents](https://arxiv.org/abs/2611.26525) | 该论文提出了Origin回溯工具，通过弥合智能体轨迹日志与应用代码执行层之间的语义鸿沟，让分析人员仅需提交少量查询即可自动识别智能体应用安全事件的入侵入口点。 |
| [^4] | [Held While Waiting: What LLM Agent Sessions Reserve on Serverless and Agent Runtimes](https://arxiv.org/abs/2611.26480) | 论文通过实测揭示了 LLM 智能体在 AgentCore 等运行时上的资源浪费问题：智能体绝大部分时间在等待模型（每次等待中位数 2.7 秒，计算仅 5–40 毫秒），但“一会话一环境”的独占式预留加上会话结束后环境仍被多占用约 15 分钟，导致服务在算力耗尽之前就先耗尽环境资源，并针对两个成因提出了相应的改进方向。 |
| [^5] | [Anticipating Future Build Results with BuildPredictor](https://arxiv.org/abs/2611.26476) | 本文提出基于随机森林的BuildPredictor模型，利用源代码与构建代码的变更、代码复杂度变化以及历史构建结果等特征来预测CI构建结果，在项目内和整体设置中分别取得了0.82和0.86的中位数MCC，有望减少CI中因微小更改而浪费的时间和资源。 |
| [^6] | [OceanCoder: A Verification-Guided Large Language Model for Automated Simulation Modelling in Ocean Engineering](https://arxiv.org/abs/2611.26333) | OceanCoder 通过可扩展数据流水线和三层验证框架为 GRPO 提供分级奖励，构建了一个能自动生成满足工程要求仿真模型的海洋工程专用大语言模型。 |
| [^7] | [Static Analysis Guidance for Secure LLM Code Generation](https://arxiv.org/abs/2611.26312) | 该论文提出两种在代码生成过程中直接利用静态分析报警信号的新策略（代码注释回滚插入与用户消息引导），相比成本高昂的迭代修复循环，仅需基线0.9至1.9倍的输出token即可有效提升LLM生成代码的安全性。 |
| [^8] | [SWE-Recreation: Benchmarking Coding Agents for Software Recreation](https://arxiv.org/abs/2611.26311) | 该论文提出SWE-Recreation基准，要求AI智能体仅通过像素界面自主使用和探索54个真实应用（含1446项隐藏需求），然后从零重建这些软件，以评测智能体自主理解并复现陌生软件的能力，结果显示五个前沿系统最高仅达到0.530加权保真度、仅22.9%的应用被完全复现。 |
| [^9] | [SkillBridge: Cost-Aware Diagnosis and Repair of Cross-OS Compatibility Defects in Agent Skills](https://arxiv.org/abs/2611.26256) | SkillBridge 提出了一种基于契约的三阶段工作流（静态分析、动态验证与修复），以低成本方式诊断并修复 LLM 智能体技能在跨操作系统场景下的兼容性缺陷。 |
| [^10] | [SheetAlign: Grounded Interaction and Stage-Isolated Learning for Spreadsheet Agents](https://arxiv.org/abs/2611.26069) | SheetAlign是一个电子表格操作框架，通过阶段隔离策略学习、任务引导的API接地以及基于执行一致性的程序选择，将代码生成锚定在工作簿观察与可用操作上，从而在SpreadsheetBench上取得优异表现。 |
| [^11] | [Barriers to GenAI Integration During Large-Scale Agile Transformation: An Exploratory Case Study](https://arxiv.org/abs/2611.25967) | 本研究通过对一家大型北欧媒体组织的探索性案例研究，揭示了大范围敏捷转型中生成式AI整合所面临的团队层面（AI观念分歧、缺乏共享实践、实验能力有限）与组织层面（AI成熟度不均衡、学习机制薄弱、跨单元协调对齐困难）的双重障碍。 |
| [^12] | [Adopting Embedded AI Through an Innovation Team](https://arxiv.org/abs/2611.25939) | 通过对挪威一家大型金融机构的定性案例研究发现，拥有结构分离、团队内AI能力、明确授权下高自主性、资深专家和灵活工作方式的临时跨职能创新团队能够成功构建嵌入式AI解决方案，但其采纳落地依赖于与常设产品团队的协调，而路线图错位、优先级分歧和缺乏同地办公会阻碍这一协调。 |
| [^13] | [From Code Review Comments to Reuse in Production: An Industrial Study of Extracted Rules](https://arxiv.org/abs/2611.25918) | 通过追踪规则提取系统 Kleene 在生产环境中 65 天的实际使用，该工业研究发现“后续评论提及规则的次数”并非规则价值的可靠证据：规则归因既不保证源评审知识被忠实保留，也不保证评论真正提出了可执行的修改要求。 |
| [^14] | [Fast Computation of Lowest Common Ancestor in Version Control Commit Graph](https://arxiv.org/abs/2611.25912) | 该论文发现版本控制提交图中存在“支撑分支”这一额外结构，通过提取该结构进行预处理，可显著加速在提交图中查找最近公共祖先（合并基准）的算法。 |
| [^15] | [On the Reliability of CyberGym Evaluation: Task Material Consistency and Reference-Target Identity](https://arxiv.org/abs/2611.25790) | 该研究对CyberGym漏洞复现基准的全部1,507个任务进行了材料审计与PoC同一性评估，发现22.4%的任务存在材料不一致问题，揭示了该基准评估结果可靠性方面的缺陷。 |
| [^16] | [Characterizing and Detecting Unfaithful Work Reports by Coding Agents](https://arxiv.org/abs/2611.25760) | 该论文基于真实的开发者—智能体对话数据，系统性地刻画了编码智能体在真实开发工作中不忠实工作报告的表现形式及其带来的误导风险，并研究其检测方法。 |
| [^17] | [Auto-Alpha: A Framework for Autonomous Factor Research](https://arxiv.org/abs/2611.25732) | Auto-Alpha是一个通过测试驱动的实现循环将报告引导的初始化与自主探索研究相结合的多智能体因子研究框架，借助多重审查机制与经验库持续构建并优化因子库，其因子中位1日Rank IC达0.0301，周频Top-300组合年化收益达16.81%。 |
| [^18] | [Copilot or Culprit? How AI Is Changing Open-Source Bug Reporting and Maintenance](https://arxiv.org/abs/2611.25605) | 该研究首次对101个开源项目中超过12万份缺陷报告进行大规模分析，发现AI参与的缺陷报告正变得越来越完整可靠，其中AI用于复现和诊断与更高的报告有效性相关。 |
| [^19] | [Technology-Enabled Fuel Supply Chain Resilience: An Integrated Information System for Real-Time Monitoring and Decision Support](https://arxiv.org/abs/2611.25542) | 本研究将燃料物流信息系统重构迁移至SQL Server平台，使数据集成缺陷率从2.90%–3.46%降至0.14%–0.49%，并通过新增库存移动记录、自动警报和恢复控制功能，提升了燃料供应链的实时监控与决策支持能力。 |
| [^20] | [A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins](https://arxiv.org/abs/2611.25471) | 本研究提出一个包含静态有效性、模型构建、仿真收敛和行为等价性四个层级的可执行验证协议，用于评估LLM生成的HVAC数字孪生规范是否物理有效，并在八个LLM和十四种HVAC拓扑上进行了验证。 |
| [^21] | [Behavioral Containment: Source-to-Binary Function Matching for 1-Day Vulnerability Detection](https://arxiv.org/abs/2611.25394) | 该论文的核心创新是提出“行为包含”这一特性（即编译保留函数行为），并据此设计出SBIND方法，通过反编译二进制并检查行为包含而非直接比较代码，实现无需相同构建配置的源码到二进制函数匹配，从而检测已部署二进制中的1-day漏洞。 |
| [^22] | [Ontology of Software Maintenance](https://arxiv.org/abs/2611.25196) | 本文提出了基于五层本体架构构建的软件维护顶层领域本体SwMaintTDO，为软件维护领域提供了统一且可扩展的概念模型，以解决该领域长期存在的概念不一致和术语差异问题。 |
| [^23] | [Formally Verified Semidefinite Programming](https://arxiv.org/abs/2611.25174) | 本文的核心贡献是实现了首个经形式化验证的半定规划求解器Rust程序，其创新之处在于手工编写求解器代码后，借助AI工具Claude在Verus定理证明器中自动生成了全部正确性验证。 |
| [^24] | [Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models](https://arxiv.org/abs/2611.25170) | 该论文提出“选择之前”这一证据导向框架，区分前瞻性内容、前瞻性计算与因果引导，用以检验语言模型是否内部表征未被选择的轨迹，并通过预注册探针实验发现低秩探针未达标且存在排除法重构的混淆问题。 |
| [^25] | [BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection](https://arxiv.org/abs/2611.25088) | 该论文提出基于十个结构特征的代码复杂度指数（CCI）来有效预测编码基准的任务难度（R²=0.951），并据此构建了BuggyMLE框架，通过向可运行的代码库注入受控错误来规模化提升机器学习工程任务的难度。 |
| [^26] | [DVBench: End-to-End Benchmark for Real-World UVM Design Verification Agents](https://arxiv.org/abs/2611.25066) | 本文提出了首个面向真实世界UVM设计验证智能体的端到端基准测试DVBench，包含涵盖9个功能领域和5个难度级别的95个任务，要求智能体独立完成完整的验证工作流程，并通过编译、覆盖率和错误检测等多个维度综合评估。 |
| [^27] | [From Industrial Product Documentation to Automated Digital Product Passport Generation: An Experimental Study of Large Language Models](https://arxiv.org/abs/2611.24998) | 本文提出一种基于检索增强生成（RAG）与大语言模型的方法，从分散异构的工业产品文档中自动提取产品信息，并依据资产管理壳（AAS）标准自动生成符合欧盟要求的数字产品护照（DPP）。 |
| [^28] | [Show Your Work: Can Small Open Models Read SEC Filings with Evidence a Machine Can Check?](https://arxiv.org/abs/2611.24990) | 该论文提出Filing-Evidence基准，验证了小型开源模型在单张消费级GPU上能够以机器可核验的页面与逐字引用证据从SEC财报文件中提取数字，最佳模型准确率达84.5%，但每千个回答中仍有16至167个错误答案能骗过所有自动检查。 |
| [^29] | [Safe AI Evaluation: A Systems Perspective](https://arxiv.org/abs/2611.24987) | 本文从系统视角剖析了AI评估过程中特有的安全风险与控制挑战，并提出多样化监督者、独立验证器、静默监控和协调关停四种架构模式以支持安全的AI评估。 |
| [^30] | [Trace2ST: Recovering PLC Structured Text from I/O Traces with LLM Generation and Verification](https://arxiv.org/abs/2611.24979) | 本文提出“生成—回放—反馈”流水线，使大语言模型仅凭PLC的输入/输出轨迹记录即可恢复出可执行的结构化文本（ST）程序，并通过逐扫描周期回放比对完成程序验证。 |
| [^31] | [MetaphIR: Corpus-Scale Applicability Analysis for LLVM Peephole Optimizations](https://arxiv.org/abs/2611.23555) | 该论文提出MetaphIR，可在LLVM IR语料库规模上同时评估窥孔优化重写规则的模式匹配与前置条件满足情况，相比忽略前置条件的结构化搜索计数更精确，且避免了完整重编译流程高昂的工程成本与47分钟级的单次运行耗时。 |
| [^32] | [Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict](https://arxiv.org/abs/2611.21784) | 本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。 |
| [^33] | [Scientific Coding with Large Language Models: Evidence from Finite Element Analysis](https://arxiv.org/abs/2611.21176) | 该论文将FEM-Bench扩展至17个涉及非线性力学与结构动力学的有限元编程任务，对15个大语言模型的评估表明，单次生成的正确率有限，而结合代码执行与修订的智能体框架能显著提升性能（最高达96.5% Pass@3），揭示了LLM在自动化科学编程中的潜力与局限。 |
| [^34] | [What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation](https://arxiv.org/abs/2610.30491) | 该研究通过在GSM8K上构建包含损坏计算器输出的工具使用场景，审计了三个轨迹级行为检测器，发现最终正确性、可见重新验证与错误传播等标签会系统性不一致，并揭示了“静默工具输出覆盖”这一关键失败模式——即模型在不重新调用工具的情况下使用了与损坏输出不同的值。 |
| [^35] | [Automatically Building Machine-Checked Assurance Cases from C Codebases to Requirements](https://arxiv.org/abs/2609.40119) | 本文提出了LLM辅助框架CCV，能够从C代码库自动构建机器可验证的保证案例，通过接口协议建模允许的调用序列并协调需求引导分析与自底向上验证两个阶段，从而证明代码库满足预期需求。 |
| [^36] | [Distilling Reasoning Traces into Advisory Prompts for Software Engineering Tasks](https://arxiv.org/abs/2608.00437) | 受人类通过反思错误并内化规则来学习编程的启发，本文提出将LLM思考模式中避免错误的推理轨迹提炼为可内化的咨询式提示，使软件工程任务无需持续承担推理成本即可获得性能提升。 |
| [^37] | [Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents](https://arxiv.org/abs/2605.21347) | 该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。 |
| [^38] | [Model Compression vs. Adversarial Robustness: An Empirical Study on Language Models for Code](https://arxiv.org/abs/2508.03949) | 本文系统评估了剪枝、量化和知识蒸馏等常见压缩策略对三种主流代码语言模型在对抗攻击下鲁棒性的影响，为压缩模型的安全部署提供了实证依据。 |
| [^39] | [Scaling GR(1) Synthesis via a Compositional Framework for LTL Discrete Event Control](https://arxiv.org/abs/2506.16557) | 该论文提出了一种组合式的GR(1)控制器综合框架，利用被控系统的模块化结构和观测综合等价性来缓解状态爆炸问题，能够求解规模达到单体式方法1000倍的LTL离散事件控制综合问题。 |

# 详细

[^1]: SoK：评估大语言模型在软件漏洞研究中能力的演进

    SoK: Assessing the Evolution of LLM Capability in Software Vulnerability Research

    [https://arxiv.org/abs/2611.26601](https://arxiv.org/abs/2611.26601)

    本文提出一个借鉴自动驾驶分级的自主性评级框架，系统化梳理了LLM在漏洞研究中的能力证据，发现2026年前智能体已能在无人工干预下发现、验证并修补真实软件漏洞，开源权重模型随后也达到了相当的自主性水平。

    

    基于大语言模型（LLM）的智能体在发现、验证和修补真实软件漏洞方面的能力日益增强，厂商和研究人员报告称这已经在改变漏洞被发现和修复的方式。然而，这些说法散见于论文、厂商报告和披露信息中，鲜有可比性。我们引入了一个框架，借鉴自动驾驶自动化分级模型，在自主性等级尺度上对这些能力进行评级，并通过其正确性、可获得性以及证据强度对每项说法加以限定。我们利用该框架对已发表的关于LLM漏洞研究能力的证据进行了系统化梳理。我们发现，在2026年受限的前沿模型出现之前，智能体已经能够在无需人工干预的情况下发现、验证并修补真实世界软件中的漏洞，而开源权重模型随后在发现和验证方面也达到了相当的自主性水平。随后我们测量了……

    arXiv:2611.26601v1 Announce Type: cross  Abstract: LLM-based agents are becoming increasingly capable at discovering, validating and patching vulnerabilities in real software, and vendors and researchers report that this is already changing how vulnerabilities are found and fixed. These claims, however, are scattered across papers, vendor reports and disclosures and are rarely comparable. We introduce a framework that rates these capabilities on an autonomy scale modeled on the levels of driving automation, and qualifies each claim by its correctness, its availability and the strength of its evidence. We use this framework to systematize the published evidence on LLM capability in vulnerability research. We find that agents discovered, validated, and patched vulnerabilities in real-world software without human intervention before the restricted frontier models of 2026, and that open-weight models subsequently attained comparable autonomy for discovery and validation. We then measure th
    
[^2]: 验证验证器：基于规范的LLM交叉检查，捕获逃过UVM记分板的缺陷

    Verify the Verifier: Specification-Grounded LLM Cross-Checking for Bugs That Escape UVM Scoreboards

    [https://arxiv.org/abs/2611.26588](https://arxiv.org/abs/2611.26588)

    提出首个基于规范、利用LLM交叉检查已通过回归测试追踪记录的框架"Verify the Verifier"，从564个注入缺陷中确认了41个逃过UVM记分板检查的缺陷。

    

    通过的UVM回归测试仅表明设计与参考模型在记分板所检查的内容上一致，并不能表明其符合规范。那些通过所有测试却违反规范的缺陷，被称为“记分板逃逸”，即使面对成熟的测试平台也能存活。我们提出“Verify the Verifier”（验证验证器），据我们所知，这是首个利用基于大语言模型（LLM）的异常交叉检查器，将已通过回归测试的追踪记录与规范进行比对的框架。在注入到四个OpenTitan IP的564个缺陷中，有68个是候选的记分板逃逸缺陷：它们与原始设计不等价、被测试所覆盖，却能通过测试。该框架确认其中41个为记分板逃逸缺陷——相关断言在缺陷设计上失败，并在原始设计上得到验证通过。

    arXiv:2611.26588v1 Announce Type: cross  Abstract: A passing UVM regression shows the design agrees with the reference model on what the scoreboard checks, not that it meets the specification. Bugs that pass every test yet violate the specification, termed scoreboard escapes, survive mature testbenches. We present Verify the Verifier, to our knowledge the first framework whose LLM-based Anomaly Cross-Checker examines a passing regression's traces against the specification. Of 564 bugs inserted into four OpenTitan IPs, 68 are candidate scoreboard escapes: non-equivalent, exercised by the tests, yet passing. The framework confirmed 41 as scoreboard escapes with assertions failing on the buggy design and proven on the original.
    
[^3]: 回溯智能体应用的安全事件

    Backtracking Agent Applications Security Incidents

    [https://arxiv.org/abs/2611.26525](https://arxiv.org/abs/2611.26525)

    该论文提出了Origin回溯工具，通过弥合智能体轨迹日志与应用代码执行层之间的语义鸿沟，让分析人员仅需提交少量查询即可自动识别智能体应用安全事件的入侵入口点。

    

    智能体应用正在被越来越广泛地部署，然而针对智能体的特定攻击对它们构成了真实的威胁。当发生安全事件时，组织需要回溯以识别入侵入口点（CEP，compromise entry point）、修复损害并防止其再次发生。可用于回溯的主要工件是记录智能体功能的轨迹日志和源代码；而应用特定的函数调用日志往往相当有限。然而，轨迹日志处于智能体功能抽象层，而CEP候选则是应用代码执行层的函数调用。由于这种语义鸿沟，分析人员需要手动检查每个CEP候选的相关代码，以推断其对应的智能体功能。在本工作中，我们提出了Origin，这是一个回溯工具，允许分析人员提交少量查询来识别现有工具无法提供的CEP。我们的方法面临两个挑战：（1）定义……（摘要在此处截断）

    arXiv:2611.26525v1 Announce Type: cross  Abstract: Agent applications are increasingly deployed, yet agent-specific attacks pose real threats to them. Upon a security incident, organizations backtrack to identify the CEP (compromise entry point), remediate the damage, and prevent its recurrence. The main artifacts available for backtracking are trajectory logs, which record agent functionalities, and source code; app(lication)-specific function call logs are often fairly limited. However, trajectory logs are at the agent functionality abstraction layer, while the CEP candidates are function calls at the app code execution layer. Due to this semantic gap, the analyst needs to manually inspect the relevant code for each CEP candidate to infer its corresponding agent functionality. In this work, we present Origin, a backtracking tool for the analyst to submit a small number of queries to identify CEPs that existing tools cannot provide. Our approach poses two challenges: (1) define the in
    
[^4]: 等待期间被占用：LLM 智能体会话在无服务器与智能体运行时中预留的资源

    Held While Waiting: What LLM Agent Sessions Reserve on Serverless and Agent Runtimes

    [https://arxiv.org/abs/2611.26480](https://arxiv.org/abs/2611.26480)

    论文通过实测揭示了 LLM 智能体在 AgentCore 等运行时上的资源浪费问题：智能体绝大部分时间在等待模型（每次等待中位数 2.7 秒，计算仅 5–40 毫秒），但“一会话一环境”的独占式预留加上会话结束后环境仍被多占用约 15 分钟，导致服务在算力耗尽之前就先耗尽环境资源，并针对两个成因提出了相应的改进方向。

    

    LLM 智能体几乎所有时间都花在等待模型上。在 Amazon Bedrock AgentCore 上，我们的智能体每次等待模型回复的中位时间为 2.7 秒，而随后的实际计算时间估计仅为 5 到 40 毫秒。AgentCore 在等待期间不收取 CPU 费用，但它会为每个会话预留一个独立的环境——一台拥有独立内存的虚拟机，并且在会话关闭之前一直保持预留状态。因此，1,000 个会话会预留 1,000 个环境，而这些环境实际完成的工作量仅相当于 2 到 15 个处理器核心。服务在算力耗尽之前很早就会先耗尽环境资源。这种短缺有两个原因，且每个原因都有对应的解决办法。其一，环境在会话结束后仍保持预留状态：在我们运行的客服基准测试中，一个会话平均持续 33 秒，但除非客户端主动关闭，其环境会在此后继续被占用 15 分钟。其二，一个环境只服务一个会话，而该会话大部分时间都在等待：在我们的模拟……（原文摘要在此处截断）

    arXiv:2611.26480v1 Announce Type: cross  Abstract: LLM agents spend nearly all their time waiting for the model. On Amazon Bedrock AgentCore, our agents waited a median of 2.7 seconds for each model reply, then computed for an estimated 5 to 40 milliseconds. AgentCore does not charge for CPU during the wait. But it reserves a separate environment, a virtual machine with its own memory, for every conversation, and keeps it reserved until the conversation is closed. So 1,000 conversations reserve 1,000 environments to do the work of 2 to 15 processor cores. A service runs out of environments long before it runs out of processing power.   The shortage has two causes, and each has a fix. An environment stays reserved after its conversation ends: in the customer-service benchmark we ran, a conversation lasts 33 seconds on average, but its environment is held for 15 minutes more unless the client closes it. And an environment serves only one conversation, which is mostly waiting: in our simu
    
[^5]: 使用BuildPredictor预测未来构建结果

    Anticipating Future Build Results with BuildPredictor

    [https://arxiv.org/abs/2611.26476](https://arxiv.org/abs/2611.26476)

    本文提出基于随机森林的BuildPredictor模型，利用源代码与构建代码的变更、代码复杂度变化以及历史构建结果等特征来预测CI构建结果，在项目内和整体设置中分别取得了0.82和0.86的中位数MCC，有望减少CI中因微小更改而浪费的时间和资源。

    

    持续集成（CI）是一种流行的编程实践，用于将代码更改无缝集成到代码库中。使用CI，开发人员可以在每次提交后受益于自动化的代码组装和测试。然而，尽管频繁的集成和构建为开发人员提供了支持，但它也带来了高昂的代价，即对微小的更改也会浪费大量的时间和资源。在这项工作中，我们研究了通过使用机器学习技术预测构建结果来减少这些缺点的潜力。具体而言，我们提出了一个随机森林模型，该模型具有多种特征，例如源代码和构建代码的更改，以及代码复杂度的变化。我们观察到，我们的模型在项目内和整体设置中均表现出良好的性能，中位数MCC分别达到0.82和0.86。我们还发现，模型中最重要的特征与历史构建结果信息有关。

    arXiv:2611.26476v1 Announce Type: cross  Abstract: Continuous Integration (CI) is a popular programming practice for the seamless integration of code changes into the code base. Using CI, developers benefit from automated assembling and testing of the code after each commit. However, while the frequent integration and building supports developers, it also comes at the high costs of wasting time and resources for minimal changes.   In this work, we investigate the potential of reducing the drawbacks by anticipating the build outcome using machine learning techniques. In particular, we propose a random forest model with various features such as changes to the source and build code, and changes to the complexity of the code.   We observe that our model shows a good performance in an intra-project and holistic setting yielding a median MCC of 0.82 and 0.86, respectively. We also found that the most important features of our model concern the historical build result information. Additionall
    
[^6]: OceanCoder：面向海洋工程自动化仿真建模的验证引导大语言模型

    OceanCoder: A Verification-Guided Large Language Model for Automated Simulation Modelling in Ocean Engineering

    [https://arxiv.org/abs/2611.26333](https://arxiv.org/abs/2611.26333)

    OceanCoder 通过可扩展数据流水线和三层验证框架为 GRPO 提供分级奖励，构建了一个能自动生成满足工程要求仿真模型的海洋工程专用大语言模型。

    

    数值仿真在海洋工程中十分重要，但模型构建需要专家知识。大语言模型可以支持自动化建模，然而代码语法正确和能够成功执行并不代表生成的模型和结果满足工程要求。我们提出了 OceanCoder，一个基于 Qwen3-8B、面向海洋工程自动化仿真建模的大语言模型。为应对专业化的编程需求和训练数据稀缺的问题，我们开发了一个可扩展的数据构建流程，生成了涵盖 12 类工程任务的 12,000 条经过验证的监督微调（SFT）样本。一个三层验证框架用于检查模型输入是否符合工程参数与任务约束、验证所需仿真分析是否完成，并评估计算结果的工程合理性。其评估结果为群体相对策略优化（GRPO）提供分级奖励，引导……（原文在此处截断）

    arXiv:2611.26333v1 Announce Type: new  Abstract: Numerical simulation is important in ocean engineering, but model construction requires expert knowledge. Large language models support automated modelling, but code syntax and successful execution alone do not establish whether generated models and results meet engineering requirements. We present OceanCoder, a Qwen3-8B-based large language model for automated simulation modelling in ocean engineering. To address specialised programming requirements and scarce training data, we develop an extensible data construction pipeline producing 12,000 verified supervised fine-tuning (SFT) examples across 12 engineering task categories. A three-layer verification framework checks model inputs against engineering parameters and task constraints, verifies completion of required simulation analyses, and assesses the engineering plausibility of computed outputs. Its outcomes provide graded rewards for group relative policy optimisation (GRPO), guidin
    
[^7]: 静态分析指导的安全LLM代码生成

    Static Analysis Guidance for Secure LLM Code Generation

    [https://arxiv.org/abs/2611.26312](https://arxiv.org/abs/2611.26312)

    该论文提出两种在代码生成过程中直接利用静态分析报警信号的新策略（代码注释回滚插入与用户消息引导），相比成本高昂的迭代修复循环，仅需基线0.9至1.9倍的输出token即可有效提升LLM生成代码的安全性。

    

    静态分析器成本低廉且被广泛部署，这使得其报告成为安全LLM代码生成的候选反馈信号。现有方法将分析器置于包含中间步骤的迭代修复循环中，这不仅成本高昂，而且难以分离报告本身对模型的影响。我们提出了两种在生成过程中直接将分析发现提供给模型的策略：当分析器发出警报时，回滚生成过程，并将发现结果以代码注释的形式插入，或插入到用户消息中。我们在三个模型和三个Python基准测试上，将这两种策略与两种迭代方法以及不使用分析器的对照组进行比较，并使用每个基准测试的判定标准来衡量安全性。结果表明，生成过程中的指导成本低且可能有效，其输出token仅为基线的0.9至1.9倍，而迭代方法则高达24.1倍。在RiskyPy基准上使用DeepSeek-V4.1-flash时，Steering方法提升了（摘要在此处截断）

    arXiv:2611.26312v1 Announce Type: new  Abstract: Static analyzers are cheap and widely deployed, which makes their reports a candidate feedback signal for secure LLM code generation. Existing approaches place the analyzer inside an iterative repair loop with intermediate steps, which is costly and makes it hard to isolate the report's effect on the model. We introduce two strategies that give the finding to the model during generation. When the analyzer raises an alert, generation is rolled back, and the finding is inserted either as a code comment (Comment) or in the user message (Steering). We compare them with two iterative approaches and a control without the analyzer, across three models and three Python benchmarks, and we measure security using each benchmark's oracle. Guidance during generation is cheap and can be effective, at 0.9 to 1.9 times the baseline's output tokens against up to 24.1 times for iterative approaches. With DeepSeek-V4.1-flash on RiskyPy, Steering raises the
    
[^8]: SWE-Recreation：面向软件复现的编程智能体基准测试

    SWE-Recreation: Benchmarking Coding Agents for Software Recreation

    [https://arxiv.org/abs/2611.26311](https://arxiv.org/abs/2611.26311)

    该论文提出SWE-Recreation基准，要求AI智能体仅通过像素界面自主使用和探索54个真实应用（含1446项隐藏需求），然后从零重建这些软件，以评测智能体自主理解并复现陌生软件的能力，结果显示五个前沿系统最高仅达到0.530加权保真度、仅22.9%的应用被完全复现。

    

    智能体越来越多地操作软件，甚至编写软件，但它们几乎总是被告知要实现什么：给定一个目标、一个原型图或一个演示。我们提出这样一个问题：智能体能否通过使用一个陌生的产品来理解它，然后将其重建出来？智能体只能看到渲染后的像素，并通过鼠标指针和键盘进行操作，无法访问DOM、源代码、日志或后端。在这一界面下，它进行探索、生成复制品，并在评估之前检查自己的工作。我们提出了SWE-Recreation，这是一个包含54个真实应用程序（28个网站和26个Android应用）的基准测试，共有1,446项隐藏需求，描述多步骤、有状态的工作流，并明确排除了重建范围之外的行为。一个固定的评判者使用像素证据对工作流进行评分；另一个独立的事后评判者估计探索覆盖度。在54个被评估的应用中，五个前沿系统最高仅达到0.530的加权保真度，仅22.9%的应用能被完全复现。

    arXiv:2611.26311v1 Announce Type: new  Abstract: Agents increasingly operate software and even write it, yet they are almost always told what to achieve: given a goal, a mock-up, or a demonstration. We ask whether an agent can instead understand an unfamiliar product by using it, and then rebuild it. The agent sees only rendered pixels and acts through pointer and keyboard, without access to the DOM, source, logs, or back-end. Within this interface, it explores, generates a replica, and checks its own work before evaluation. We present SWE-Recreation, a benchmark of 54 real applications (28 websites and 26 Android apps) with 1,446 hidden requirements describing multistep, stateful workflows and explicit exclusions for behavior outside the reconstruction scope. A fixed judge grades workflows with pixel evidence; a separate post-hoc judge estimates exploration coverage. Across 54 evaluated applications, five frontier systems reach at most 0.530 weighted fidelity and 22.9% fully reproduce
    
[^9]: SkillBridge：智能体技能中跨操作系统兼容性缺陷的成本感知诊断与修复

    SkillBridge: Cost-Aware Diagnosis and Repair of Cross-OS Compatibility Defects in Agent Skills

    [https://arxiv.org/abs/2611.26256](https://arxiv.org/abs/2611.26256)

    SkillBridge 提出了一种基于契约的三阶段工作流（静态分析、动态验证与修复），以低成本方式诊断并修复 LLM 智能体技能在跨操作系统场景下的兼容性缺陷。

    

    LLM 智能体越来越多地通过技能来复用可执行能力，但在一个操作系统上正常工作的技能在另一个操作系统上可能会失败，这是因为操作系统相关的假设分散在指南文本、脚本、元数据、安装契约和本地入口点之中。通过在每个操作系统上运行每个技能来诊断此类失败的成本非常高昂：许多执行会消耗模型 token、运行器时间和审查精力，之后才能揭示失败是否真的与操作系统相关。我们提出了 SkillBridge，这是一个基于契约的三阶段工作流：静态分析、动态验证和修复。SkillBridge 首先筛选 3,270 个技能-目标操作系统用例（涵盖 Linux、macOS 和 Windows 上的 1,090 个技能），以寻找对操作系统敏感的契约证据。随后在选定用例上运行动态任务并进行根因审查，以确认失败是否由操作系统假设引起，并修复导致失败的契约字段，例如指南文本、软件包脚本、依赖项或平台相关配置。

    arXiv:2611.26256v1 Announce Type: cross  Abstract: LLM agents increasingly reuse executable capabilities through skills, but a skill that works on one OS may fail on another because OS assumptions are spread across guide text, scripts, metadata, install contracts, and local entrypoints. Diagnosing such failures by running every skill on every OS is expensive: many executions spend model tokens, runner time, and review effort before revealing whether the failure is truly OS-related. We present SkillBridge, a contract-based workflow with three stages: static analysis, dynamic validation, and repair. SkillBridge first screens 3,270 skill-target-OS cases, covering 1,090 skills across Linux, macOS, and Windows, to find OS-sensitive contract evidence. It then runs dynamic tasks and root-cause review on selected cases to confirm whether a failure is caused by an OS assumption, and repairs the contract field that caused the failure, such as guide text, package scripts, dependencies, or platfor
    
[^10]: SheetAlign：面向电子表格智能体的具接地交互与阶段隔离学习

    SheetAlign: Grounded Interaction and Stage-Isolated Learning for Spreadsheet Agents

    [https://arxiv.org/abs/2611.26069](https://arxiv.org/abs/2611.26069)

    SheetAlign是一个电子表格操作框架，通过阶段隔离策略学习、任务引导的API接地以及基于执行一致性的程序选择，将代码生成锚定在工作簿观察与可用操作上，从而在SpreadsheetBench上取得优异表现。

    

    电子表格被广泛用于数据分析和决策支持。在保护敏感数据和控制部署成本的前提下实现电子表格分析自动化，推动了本地部署的开源权重大语言模型（LLM）的应用。同时，复杂的电子表格任务既需要任务推理，也需要对工作簿的理解，而预定义的操作接口还带来了额外的工具选择与调用需求。我们提出了SheetAlign，一个具有阶段隔离策略学习的电子表格操作框架，围绕“在哪里操作、更改什么、以及如何实现转换”来组织。任务引导的检查和API接地将代码生成锚定在工作簿观察和可用操作上，而阶段隔离学习则利用编辑进度奖励来优化程序生成与审查。在推理阶段，执行一致性通过工作簿状态之间的一致性来选择程序。在SpreadsheetBench上...

    arXiv:2611.26069v1 Announce Type: new  Abstract: Spreadsheets are widely used for data analysis and supporting decision-making. Automating spreadsheet analysis while protecting sensitive data and controlling deployment costs motivates locally deployed open-weight large language models (LLMs). Meanwhile, complex spreadsheet tasks require both task reasoning and workbook understanding, while predefined operation interfaces impose additional tool-selection and invocation demands. We introduce SheetAlign, a spreadsheet manipulation framework with stage-isolated policy learning, organized around where to operate, what to change, and how to implement transformations. Task-guided inspection and API grounding anchor code generation in workbook observations and available operations, while stage-isolated learning optimizes program generation and review using edit-progress rewards. At inference, execution consistency selects programs through agreement among workbook states. On SpreadsheetBench, o
    
[^11]: 大规模敏捷转型中生成式AI整合的障碍：一项探索性案例研究

    Barriers to GenAI Integration During Large-Scale Agile Transformation: An Exploratory Case Study

    [https://arxiv.org/abs/2611.25967](https://arxiv.org/abs/2611.25967)

    本研究通过对一家大型北欧媒体组织的探索性案例研究，揭示了大范围敏捷转型中生成式AI整合所面临的团队层面（AI观念分歧、缺乏共享实践、实验能力有限）与组织层面（AI成熟度不均衡、学习机制薄弱、跨单元协调对齐困难）的双重障碍。

    

    正在进行大规模敏捷转型的组织正日益将生成式AI（GenAI）工具引入日常工作之中。尽管GenAI有望加速软件开发，但人们对组织如何将GenAI整合到现有的大规模敏捷工作方式中，以及它如何与自主性、一致性、学习和协调等方面的张力相互作用，仍知之甚少。我们通过对一家正在进行大规模敏捷转型的大型北欧媒体组织开展探索性案例研究，考察了GenAI整合所面临的障碍。研究结果揭示了团队层面的障碍，包括AI观念分歧、缺乏共享实践以及实验能力有限；以及组织层面的障碍，包括AI成熟度不均衡、组织学习机制薄弱以及跨单元协调与对齐方面的挑战。我们的研究结果表明，GenAI并不会创造新的组织挑战，而是与现有的组织问题交织在一起……

    arXiv:2611.25967v1 Announce Type: new  Abstract: Organizations undergoing large-scale agile transformations are increasingly introducing generative AI (GenAI) tools into everyday work. While GenAI promises to accelerate software development, less is known about how organizations integrate GenAI into existing large-scale agile ways of working and how it interacts with tensions related to autonomy, alignment, learning, and coordination. We examine barriers to GenAI integration through an exploratory case study of a large Nordic media organization undergoing a large-scale agile transformation. The findings reveal team-level barriers, including diverging AI beliefs, lack of shared practices, and limited experimentation capacity, and organizational-level barriers, including uneven AI maturity, weak organizational learning mechanisms, and cross-unit coordination and alignment challenges. Our findings suggest that GenAI does not create new organizational challenges but becomes intertwined wit
    
[^12]: 通过创新团队采纳嵌入式AI

    Adopting Embedded AI Through an Innovation Team

    [https://arxiv.org/abs/2611.25939](https://arxiv.org/abs/2611.25939)

    通过对挪威一家大型金融机构的定性案例研究发现，拥有结构分离、团队内AI能力、明确授权下高自主性、资深专家和灵活工作方式的临时跨职能创新团队能够成功构建嵌入式AI解决方案，但其采纳落地依赖于与常设产品团队的协调，而路线图错位、优先级分歧和缺乏同地办公会阻碍这一协调。

    

    软件组织正日益将AI能力嵌入到由现有团队拥有和维护的产品中，但这类开发工作应如何组织尚不明确。我们对挪威一家大型金融机构中一个临时性的跨职能创新团队开展了定性案例研究，研究基于对15名参与者的18次访谈、10天的办公室实地观察、对十次会议的观察以及内部文档分析。我们发现，结构上的分离、团队内部的AI能力、在明确授权范围内的高度自主性、资深专家的参与以及灵活的工作方式，使该团队能够探索并构建嵌入式AI解决方案。虽然结构分离促成了这项工作，但它也使采纳落地依赖于与最终拥有该解决方案的常设产品团队之间的协调。而路线图不一致、优先级差距以及缺乏同地办公，制约了这种协调。我们识别出七项……（原文摘要在此处截断）

    arXiv:2611.25939v1 Announce Type: new  Abstract: Software organizations are increasingly embedding AI capabilities into products that existing teams already own and maintain, but how this development work should be organized is not well understood. We conducted a qualitative case study of a temporary cross-functional innovation team in a large Norwegian financial organization, drawing on 18 interviews with 15 participants, 10 days of office observation, observations of ten meetings, and internal documents. We found that structural separation, AI competence inside the team, high autonomy within a clear mandate, senior expertise, and flexible ways of working enabled the team to explore and build embedded AI solutions. While structural separation enabled this work, it also made adoption dependent on coordination with the permanent product team that would own the solution. Misaligned roadmaps, prioritization gaps, and a lack of co-location constrained that coordination. We identify seven s
    
[^13]: 从代码评审评论到生产环境中的复用：对所提取规则的工业研究

    From Code Review Comments to Reuse in Production: An Industrial Study of Extracted Rules

    [https://arxiv.org/abs/2611.25918](https://arxiv.org/abs/2611.25918)

    通过追踪规则提取系统 Kleene 在生产环境中 65 天的实际使用，该工业研究发现“后续评论提及规则的次数”并非规则价值的可靠证据：规则归因既不保证源评审知识被忠实保留，也不保证评论真正提出了可执行的修改要求。

    

    许多组织特有的代码评审知识仍然埋藏在针对单个变更的评论之中，近期的一些系统通过将这些评论转化为供语言模型在后续变更中应用的规则来实现复用。团队随后必须决定保留哪些规则，而手头可用的证据是该系统后续评论提及每条规则的次数。通过对这样一个系统——Kleene——的规则在生产环境中 65 天的使用情况进行追踪，我们发现这种计数并不能反映某条规则的实际贡献。某条规则被后续评论归因引用，既不能说明其源内容被忠实保留，也不能说明该评论确实提出了修改要求。在后续评论作为理由提及的 12 条规则共 56 个条款中，只有 2 个条款在源评审讨论串中是明确存在的，38 个条款在相应段落之上增加了额外条件，16 个条款找不到可对应的来源段落；在提及这些规则的另外 79 条评论中，有 45 条并未要求进行任何修改，其中 27 条甚至在声称该规则适用的情况下也未要求修改。在 20 个构造出的代码变更上，……

    arXiv:2611.25918v1 Announce Type: new  Abstract: Much organization-specific code review knowledge remains embedded in comments on individual changes, and recent systems reuse it by turning those comments into rules a language model applies to later changes. Teams must then decide which rules to keep, and the evidence at hand is how often the system's later comments name each rule. Tracing the rules of one such system, Kleene, through 65 days of production use, we found that such a count does not show what a rule contributed. A rule attribution indicates neither whether the source content was preserved nor whether the comment requested a change. Of 56 clauses in 12 rules that later comments named as their reason, 2 were explicit in a source review thread, 38 added conditions to a corresponding passage, and 16 had no passage we could locate; 45 of 79 further comments naming those rules requested no change, 27 of them while stating that the rule applied. On 20 constructed code changes, a 
    
[^14]: 版本控制提交图中最近公共祖先的快速计算

    Fast Computation of Lowest Common Ancestor in Version Control Commit Graph

    [https://arxiv.org/abs/2611.25912](https://arxiv.org/abs/2611.25912)

    该论文发现版本控制提交图中存在“支撑分支”这一额外结构，通过提取该结构进行预处理，可显著加速在提交图中查找最近公共祖先（合并基准）的算法。

    

    当用户想要合并两个（或多个）开发分支时，例如将自己的开发内容合入公共分支，就会出现在提交图中查找最近公共祖先的问题。此时，版本控制系统必须确定合并基准——即各分支最新提交的共同祖先，以便展示分支之间的相互差异。该问题的复杂之处在于，提交图并不是一棵树，而是一个任意的无环图。此外，如果不存在共同祖先，查找过程可能需要遍历整个图。遍历一个拥有数百万个顶点的图需要数十秒，这种延迟对用户来说是显而易见的。我们证明了提交图具有额外的结构——支撑分支，并且可以利用这一结构来加速算法。通过提取支撑分支，可以对图进行预处理，从而加速最近公共祖先的查找。

    arXiv:2611.25912v1 Announce Type: cross  Abstract: The problem of finding the lowest common ancestor in the commit graph arises when a user wants to merge two (or more) development branches: for example, to add their own development into a common branch. At that moment, the version control system must determine the merge base - the common ancestor of the latest commits of the branches, in order to show the mutual difference between the branches. The problem is complicated by the fact that the commit graph is not a tree but an arbitrary acyclic graph. Moreover, if there is no common ancestor, searching for one may require traversing the entire graph. Traversing a graph with several million vertices takes tens of seconds, which becomes noticeable to users.   We show that the commit graph has additional structure - supporting branches - and that this can be used to speed up the algorithm. By extracting supporting branches, the graph can be preprocessed and the search for the lowest common
    
[^15]: 论CyberGym评估的可靠性：任务材料一致性与参照目标同一性

    On the Reliability of CyberGym Evaluation: Task Material Consistency and Reference-Target Identity

    [https://arxiv.org/abs/2611.25790](https://arxiv.org/abs/2611.25790)

    该研究对CyberGym漏洞复现基准的全部1,507个任务进行了材料审计与PoC同一性评估，发现22.4%的任务存在材料不一致问题，揭示了该基准评估结果可靠性方面的缺陷。

    

    arXiv:2611.25790v1 公告类型：交叉  摘要：CyberGym通过检验概念验证输入是否能使存在漏洞的程序版本崩溃、而不使修复后的版本崩溃，来评估漏洞复现能力。将这种差异性通过解读为成功的漏洞复现，需要任务材料保持一致，并需要证据表明该PoC确实触发了预期的缺陷。我们通过对全部1,507个CyberGym任务进行材料审计，以及对一次完整的Level-1 GLM-5.3运行所生成的最终PoC进行同一性评估，来检验上述要求。以每份参考崩溃报告所关联的缺陷为目标，在对分层盲审中识别出的两个标签进行修正后，审计为338个任务（占22.4%）标注了不一致性标签。其中最大的类别是覆盖率不一致（205个任务），以及描述与补丁共同指向另一个缺陷（110个任务）。我们采用执行检查、崩溃签名匹配以及大语言模型辅助的根因审查来评估PoC的同一性。在1,286个……（原文摘要在此处截断）

    arXiv:2611.25790v1 Announce Type: cross  Abstract: CyberGym evaluates vulnerability reproduction by checking whether a proof-of-concept input (PoC) crashes a vulnerable program version but not a fixed version. Interpreting this differential pass as successful reproduction requires consistent task materials and evidence that the PoC triggers the intended defect. We examine these requirements through a material audit of all 1,507 CyberGym tasks and an identity assessment of the final PoCs from one complete Level-1 GLM-5.3 run. Taking the defect associated with each reference crash report as the target, the audit assigns inconsistency labels to 338 tasks (22.4%) after correcting two labels identified by stratified blind review. The largest categories are coverage inconsistency (205 tasks) and descriptions and patches that jointly point to another defect (110 tasks). We assess PoC identity using an execution check, crash-signature matching, and LLM-assisted root-cause review. Of the 1,286 
    
[^16]: 编码智能体不忠实工作报告的表征与检测

    Characterizing and Detecting Unfaithful Work Reports by Coding Agents

    [https://arxiv.org/abs/2611.25760](https://arxiv.org/abs/2611.25760)

    该论文基于真实的开发者—智能体对话数据，系统性地刻画了编码智能体在真实开发工作中不忠实工作报告的表现形式及其带来的误导风险，并研究其检测方法。

    

    编码智能体越来越多地承担长周期的开发任务，自主规划并执行跨越文件、工具和会话的多步骤操作。由于开发者很少逐项跟进每一步操作，而接续工作的智能体又依赖于已完成工作的记录，智能体的工作报告往往成为对所更改和所建立内容的主要描述。当报告包含虚假或缺乏依据的声明时，会给接收者带来虚假的安全感：未完成的工作可能在未经核实的情况下被接受，对当前状态的错误假设可能传播到后续步骤和其他智能体，而采信声称成功的评估者可能奖励的是报告本身而非实际工作。然而，人们对于不忠实报告在真实开发工作中的具体表现，以及开发者和智能体在其出现后如何应对，仍知之甚少。基于 SWE-chat 中真实的开发者与智能体对话，我们对自我（原文摘要在此处截断，后续内容缺失）

    arXiv:2611.25760v1 Announce Type: cross  Abstract: Coding agents increasingly carry out long-horizon development tasks, autonomously planning and executing multi-step actions that span files, tools, and sessions. Because developers rarely follow each action, and agents resuming the work depend on records of what was done, the agent's work report often becomes the primary account of what was changed and established. When the report contains false or unsupported claims, it gives recipients false assurance: incomplete work may be accepted without verification, incorrect assumptions about the current state can propagate to later steps and agents, and evaluators that credit stated success may reward the report rather than the work. Yet little is known about what unfaithful reporting looks like in real development work, or how developers and agents handle it once it surfaces. Starting from real developer--agent conversations in SWE-chat, we conduct a systematic mixed-methods analysis of self
    
[^17]: Auto-Alpha：一个自主因子研究框架

    Auto-Alpha: A Framework for Autonomous Factor Research

    [https://arxiv.org/abs/2611.25732](https://arxiv.org/abs/2611.25732)

    Auto-Alpha是一个通过测试驱动的实现循环将报告引导的初始化与自主探索研究相结合的多智能体因子研究框架，借助多重审查机制与经验库持续构建并优化因子库，其因子中位1日Rank IC达0.0301，周频Top-300组合年化收益达16.81%。

    

    自主因子研究既需要生成经济学假设，也需要对其实施和检验。我们提出Auto-Alpha，一个通过测试驱动的实现循环将“报告引导的初始化”与“自主探索研究”相连接的框架。一个想法智能体提出经济机制与信号构造；一个研究智能体则结合回测反馈对其进行开发。通过语义审计、量化审查和人工准入，筛选出的因子进入一个持久化的因子库，同时一个经验库用于沉淀研究洞见。从120份研究报告出发，Auto-Alpha构建了包含32个因子的初始因子库。随后它持续生成新想法并改进现有候选因子，共产出420个实现，并额外保留62个因子。我们基于包含105个因子的固定库快照来评估因子质量与组合表现：1日Rank IC中位数为0.0301，周频Top-300组合年化收益达16.81%（扣除……后）。（注：摘要原文在此处截断）

    arXiv:2611.25732v1 Announce Type: cross  Abstract: Autonomous factor research requires generating economic hypotheses as well as implementing and testing them. We present Auto-Alpha, a framework that connects report-guided initialization to self-directed research through a test-driven implementation loop. An idea agent proposes economic mechanisms and signal constructions; a research agent develops them with backtest feedback. Semantic audit, quantitative review, and human admission select factors for a persistent library, while an Experience Library retains research insights. Starting from 120 research reports, Auto-Alpha builds an initial library of 32 factors. It then generates new ideas and refines existing candidates, producing 420 implementations and retaining 62 additional factors. We evaluate factor quality and portfolio performance using a fixed library snapshot of 105 factors. The median 1-day Rank IC is 0.0301, and a weekly Top-300 portfolio returns 16.81% annualized after s
    
[^18]: 副驾驶还是罪魁祸首？AI如何改变开源软件的缺陷报告与维护

    Copilot or Culprit? How AI Is Changing Open-Source Bug Reporting and Maintenance

    [https://arxiv.org/abs/2611.25605](https://arxiv.org/abs/2611.25605)

    该研究首次对101个开源项目中超过12万份缺陷报告进行大规模分析，发现AI参与的缺陷报告正变得越来越完整可靠，其中AI用于复现和诊断与更高的报告有效性相关。

    

    AI生成的缺陷报告曾因不可靠的陈述和错误的复现步骤给维护者带来负担而饱受批评。然而，随着大语言模型（LLM）能力的提升，一些维护者开始对AI在缺陷报告与维护中的应用持更加积极的看法。理解这一演变需要考察：AI参与的报告是否正变得更加可靠，AI的哪些用途与有效的发现相关联，以及维护者如何调查这些报告并据此采取行动。为此，我们开展了首个大规模研究，分析了2024年1月至2026年9月期间来自101个开源项目的120,918份缺陷报告，追踪AI从发现、报告到验证和解决全过程中的参与情况。我们的研究结果表明，AI参与的报告随时间推移变得更加完整，被确认为有效的报告比例不断上升。在缺陷报告环节，AI用于复现和诊断与更高的报告有效性相关，而用于撰写报告……（摘要原文在此处截断）

    arXiv:2611.25605v1 Announce Type: cross  Abstract: AI-generated bug reports have drawn criticism for unreliable claims and faulty reproducers that burden maintainers. However, as LLM capabilities improve, some maintainers have begun to express more positive views on AI use in bug reporting and maintenance. Understanding this evolution requires examining whether AI-involved reports are becoming more reliable, which uses of AI are associated with valid findings, and how maintainers investigate these reports and act on them. To this end, we conduct the first large-scale study of 120,918 bug reports from 101 open-source projects between January 2024 and September 2026, tracing AI involvement from discovery and reporting through validation and resolution. Our findings show that AI-involved reports become more complete over time, with a growing share of reports confirmed as valid. In bug reporting, AI for reproduction and diagnosis are associated with higher validity, whereas report writing 
    
[^19]: 技术赋能的燃料供应链韧性：一个用于实时监控与决策支持的集成信息系统

    Technology-Enabled Fuel Supply Chain Resilience: An Integrated Information System for Real-Time Monitoring and Decision Support

    [https://arxiv.org/abs/2611.25542](https://arxiv.org/abs/2611.25542)

    本研究将燃料物流信息系统重构迁移至SQL Server平台，使数据集成缺陷率从2.90%–3.46%降至0.14%–0.49%，并通过新增库存移动记录、自动警报和恢复控制功能，提升了燃料供应链的实时监控与决策支持能力。

    

    本研究评估并扩展了早期燃料物流研究中所描述的事务性信息系统，重点关注数据可靠性、监控与决策支持。由作者主导的重新评估审查了现有的Delphi和SQL代码，并在十二个评估场景中将重构的源数据基线系统（R0）与修订后的SQL Server系统（R1）进行对比。技术评估采用了合成交易数据和公开的美国能源数据，同时对组织流程和供应延迟进行了模拟。在受控的SQL Server测试中，R1改善了数据集成和报告准确性，并引入了库存移动记录、自动警报和恢复控制机制。在可比的数据源上，集成缺陷率从R0的2.90%至3.46%降至R1的0.14%至0.49%。在国家周度需求预测方面，所选方法并未显著优于上一年度均值。原文在流程与中断部分截断。

    arXiv:2611.25542v1 Announce Type: new  Abstract: This study evaluates and extends the transactional information system described in an earlier fuel logistics study, focusing on data reliability, monitoring, and decision support. An author-led re-evaluation examined available Delphi and SQL code and compared a reconstructed source-derived baseline (R0) with a revised SQL Server system (R1) across twelve evaluation scenarios. Synthetic transactions and public US energy data were used for technical assessment, while organizational processes and supply delays were simulated. R1 improved data integration and reporting accuracy in controlled SQL Server tests and introduced stock-movement records, automated alerts, and recovery controls. On comparable data feeds, integration defect rates decreased from 2.90 to 3.46 percent in R0 to 0.14 to 0.49 percent in R1. For national weekly demand forecasting, selected methods did not significantly outperform the previous-year mean. Process and disruptio
    
[^20]: 一种用于评估大语言模型生成的物理式暖通空调（HVAC）数字孪生拓扑规范的四层可执行验证协议

    A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins

    [https://arxiv.org/abs/2611.25471](https://arxiv.org/abs/2611.25471)

    本研究提出一个包含静态有效性、模型构建、仿真收敛和行为等价性四个层级的可执行验证协议，用于评估LLM生成的HVAC数字孪生规范是否物理有效，并在八个LLM和十四种HVAC拓扑上进行了验证。

    

    空调是制造业与建筑物中的主要能源消耗设备，其基于物理的数字孪生可以根据对设备、连接关系和尺寸参数的声明式规范自动生成。如今，大语言模型（LLM）能够根据自然语言描述撰写此类规范。然而，现有的规范评估方法仅停留在解析成功、运行完成或文本相似性层面，无法拒绝那些在仿真中收敛但物理上错误的模型。本研究提出了一种四层验证协议，用于检验LLM撰写的规范不仅能成功执行，还能产生物理上有效的数字孪生。每份规范需依次通过静态有效性检验、模型构建、仿真收敛，以及与隐藏参考模型的行为等价性验证（等价性带宽通过±20%的尺寸扰动进行校准）。该协议被应用于八个LLM和十四种HVAC拓扑结构，其中五种描述的是一个ex……（摘要内容被截断）

    arXiv:2611.25471v1 Announce Type: cross  Abstract: Physics-based digital twins of air conditioning, a principal energy consumer in manufacturing and buildings, can be generated automatically from a declarative specification of its equipment, connections, and sizing. Large language models (LLMs) now write such specifications from plain-language descriptions. Evaluations of these specifications stop at parse success, completed runs, or textual similarity, and cannot reject a model that converges in simulation but is physically wrong. In this study, a four-level protocol is proposed that tests whether an LLM-written specification not only executes but also yields a physically valid digital twin. Each specification is executed through static validity, model build, simulation convergence, and behavioral equivalence with a hidden reference model within bands calibrated by $\pm$20% sizing perturbations. The protocol was applied to eight LLMs and fourteen HVAC topologies, five describing an ex
    
[^21]: 行为包含：面向1-day漏洞检测的源码到二进制函数匹配

    Behavioral Containment: Source-to-Binary Function Matching for 1-Day Vulnerability Detection

    [https://arxiv.org/abs/2611.25394](https://arxiv.org/abs/2611.25394)

    该论文的核心创新是提出“行为包含”这一特性（即编译保留函数行为），并据此设计出SBIND方法，通过反编译二进制并检查行为包含而非直接比较代码，实现无需相同构建配置的源码到二进制函数匹配，从而检测已部署二进制中的1-day漏洞。

    

    摘要（arXiv:2611.25394v1，公告类型：新论文）：重用的开源代码中的漏洞往往会传播到已部署的二进制文件中。检测这类漏洞需要进行源码到二进制的函数匹配，即在二进制文件中定位受影响的源函数。若将以往的二进制到二进制方法应用于该任务，则需要使用与目标二进制文件相同的构建配置来编译源代码，而这种配置通常是不可获得的；对二进制文件进行反编译也无法使以往的源码到源码方法生效，因为反编译得到的代码与原始源代码存在很大差异。我们观察到，编译虽然会转换函数的代码，但在很大程度上保留了其行为。因此，实现某个源函数的二进制函数中包含了该源函数的行为，我们将这一特性称为“行为包含”。基于这一观察，我们提出了SBIND，它对二进制文件进行反编译，并检查行为包含而非比较代码。SBIND通过行为路径来表示每个函数……（原文摘要在此处被截断）

    arXiv:2611.25394v1 Announce Type: new  Abstract: Vulnerabilities in reused open-source code often propagate into deployed binaries. Detecting them requires source-to-binary function matching, which locates the affected source function in a binary. Applying prior binary-to-binary approaches to this task would require compiling the source code with the same build configuration as the target binary, which is usually unavailable. Decompiling the binary does not make prior source-to-source approaches work either, because decompiled code differs substantially from the original source code.   We observe that compilation transforms the code of a function but largely preserves its behavior. A binary function that implements a source function therefore contains its behavior, a property we call behavioral containment. Building on this, we present SBIND, which decompiles the binary and checks behavioral containment instead of comparing code. SBIND represents each function by behavioral paths, sequ
    
[^22]: 软件维护本体论

    Ontology of Software Maintenance

    [https://arxiv.org/abs/2611.25196](https://arxiv.org/abs/2611.25196)

    本文提出了基于五层本体架构构建的软件维护顶层领域本体SwMaintTDO，为软件维护领域提供了统一且可扩展的概念模型，以解决该领域长期存在的概念不一致和术语差异问题。

    

    软件维护通常是软件生命周期中漫长且成本高昂的阶段，涵盖缺陷修复、适应性调整、代码改进和功能增强等活动。概念上的不一致性与术语的差异性持续阻碍着研究人员与从业者之间的有效沟通。为弥补这一差距，本文提出了软件维护顶层领域本体，这是一个在五层本体架构中开发的顶层领域本体。该本体的构建采用了能力问题方法、对已文档化维护过程、制品和资源的本体分析，并与既有软件工程标准保持对齐。SwMaintTDO提供了一个统一且可扩展的概念模型，用于表示关键的维护过程、角色、资源、制品、目标目的，以及目标可修改实体和上下文实体从当前状态向更新状态的转变。

    arXiv:2611.25196v1 Announce Type: new  Abstract: Software maintenance is often a long and costly phase of the software lifecycle, encompassing activities such as defect correction, adaptation, code improvement, and functionality enhancement. Conceptual inconsistencies and terminological variation continue to hinder effective communication among researchers and practitioners. To address this gap, this paper presents the Software Maintenance Top-Domain Ontology (SwMaintTDO), a high-domain ontology developed within a five-tier ontological architecture. This ontology was constructed using competency questions, ontological analysis of documented maintenance processes, artifacts, and resources, and alignment with established software engineering standards. SwMaintTDO provides a unified and extensible conceptual model for representing key maintenance processes, roles, resources, artifacts, goal purposes, and the transition of target modifiable and context entities from current to updated main
    
[^23]: 形式化验证的半定规划

    Formally Verified Semidefinite Programming

    [https://arxiv.org/abs/2611.25174](https://arxiv.org/abs/2611.25174)

    本文的核心贡献是实现了首个经形式化验证的半定规划求解器Rust程序，其创新之处在于手工编写求解器代码后，借助AI工具Claude在Verus定理证明器中自动生成了全部正确性验证。

    

    我们提出了一个经形式化验证的半定规划求解器的Rust实现，这是一款一阶求解器，也是一段非同小可的数值软件。我们手工编写了该求解器，随后指导Claude在Verus（一个面向Rust程序的自动化定理证明器）中生成了完整的验证。我们讨论了求解器背后的算法、其在Rust中的实现，以及我们在Verus中证明其正确性的实用方法。

    arXiv:2611.25174v1 Announce Type: new  Abstract: We present a formally verified Rust implementation of a first-order semidefinite programming solver, a nontrivial piece of numerical software. We wrote the solver by hand and then directed Claude to generate the entire verification in Verus, an automated theorem prover for Rust programs. We discuss the algorithm underlying the solver, its implementation in Rust, and our pragmatic approach to proving it correct in Verus.
    
[^24]: 选择之前：研究语言模型中前瞻性表征的证据框架

    Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models

    [https://arxiv.org/abs/2611.25170](https://arxiv.org/abs/2611.25170)

    该论文提出“选择之前”这一证据导向框架，区分前瞻性内容、前瞻性计算与因果引导，用以检验语言模型是否内部表征未被选择的轨迹，并通过预注册探针实验发现低秩探针未达标且存在排除法重构的混淆问题。

    

    一个可观察的决策揭示了被选中的行动，但并不能确定内部表征了哪些备选方案。借鉴前瞻性研究与海马体前向扫掠的观测，我们提出了“选择之前”，一个面向证据的框架，用以区分面向未来的内容、前瞻性计算与因果引导。该框架组织了四组实验对比，并使其替代解释变得明确。其首个测试案例关注确定性符号环境中的未被选择轨迹。一项预注册的低秩探针在密封测试集上仅评估一次，未达到其预设标准（配对准确率为0.020，而基于文本与观测选择的基线为0.195；共293个有效配对）。探索性假设H1b揭示了“排除法重构”问题：识别出被选中的候选即可能确定未被选择的目标。在H1c中，匹配世界在保持候选键位级相同的同时改变未被选择的延续……

    arXiv:2611.25170v1 Announce Type: cross  Abstract: An observable decision reveals the selected action but does not determine which alternatives were internally represented. Drawing on prospection research and hippocampal forward-sweep observations, we develop Before the Choice, an evidence-oriented framework distinguishing future-directed content, prospective computation, and causal guidance. The framework organizes four experimental contrasts and makes their alternative explanations explicit. Its first test case concerns unchosen trajectories in deterministic symbolic environments. A preregistered low-rank probe evaluated once on a sealed TEST did not meet its criterion (paired accuracy 0.020 versus 0.195 for a text-and-observed-choice baseline; 293 admissible pairs). Exploratory H1b exposed a reconstruction-by-elimination problem: identifying the chosen candidate can determine the unchosen target. In H1c, matched worlds varied an unchosen continuation while keeping candidate keys bit
    
[^25]: BuggyMLE：通过错误注入扩展机器学习工程难度

    BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection

    [https://arxiv.org/abs/2611.25088](https://arxiv.org/abs/2611.25088)

    该论文提出基于十个结构特征的代码复杂度指数（CCI）来有效预测编码基准的任务难度（R²=0.951），并据此构建了BuggyMLE框架，通过向可运行的代码库注入受控错误来规模化提升机器学习工程任务的难度。

    

    让AI系统能够自主改进AI模型，需要具备执行机器学习工程（MLE）能力的智能体。在实践中，设计、训练和评估复杂模型需要修改多文件代码库，以复现结果、调试训练过程、整合数据集以及移植框架。然而，现有的MLE基准测试将智能体限制为仅能进行单文件编辑，且无法完整访问代码库或shell，而常见的代码库级软件工程基准测试已迅速饱和。为解决这一问题，我们提出了代码复杂度指数（CCI），这是一种基于十个结构特征的难度度量指标，涵盖代码库规模、依赖深度和补丁大小。CCI在八个公开编码基准测试中能够有力预测平均解决率（R²=0.951），并确定补丁大小和复杂度是最主要的预测因素。在CCI的指导下，我们开发了BuggyMLE，这是一个可扩展的框架，智能体在其中向正常工作的代码注入受控的变异错误（注：原文摘要至此截断）。

    arXiv:2611.25088v1 Announce Type: new  Abstract: Enabling AI systems to autonomously improve AI models requires agents capable of performing machine learning engineering (MLE). In practice, designing, training, and evaluating complex models demands modifying multi-file repositories to reproduce results, debug training runs, integrate datasets, and port frameworks. However, existing MLE benchmarks limit agents to single-file edits without full repository or shell access, and common repository-level software engineering benchmarks have rapidly saturated. To address this, we introduce the Code Complexity Index (CCI), a difficulty metric based on ten structural features spanning repository scale, dependency depth, and patch size. CCI strongly predicts average solve rates across eight public coding benchmarks ($R^2 = 0.951$), identifying patch size and complexity as top predictors. Guided by CCI, we develop BuggyMLE, a scalable framework where agents inject controlled mutations into working
    
[^26]: DVBench：面向真实世界UVM设计验证智能体的端到端基准测试

    DVBench: End-to-End Benchmark for Real-World UVM Design Verification Agents

    [https://arxiv.org/abs/2611.25066](https://arxiv.org/abs/2611.25066)

    本文提出了首个面向真实世界UVM设计验证智能体的端到端基准测试DVBench，包含涵盖9个功能领域和5个难度级别的95个任务，要求智能体独立完成完整的验证工作流程，并通过编译、覆盖率和错误检测等多个维度综合评估。

    

    设计验证（DV）仍然是硬件开发中的主要瓶颈，其中基于通用验证方法学（UVM）的大规模验证工作高度依赖人力。大语言模型（LLM）和智能体系统的最新进展为DV自动化提供了新的机遇。然而，现有的基准测试主要构建相互解耦的子任务，这与真实世界的工作流程存在偏差。在本文中，我们提出了DVBench，这是首个面向真实世界UVM设计验证智能体的端到端基准测试。DVBench包含95个任务，涵盖9个功能领域和5个难度级别，覆盖多样化的验证场景和设计规模。每个任务都要求智能体独立完成完整的验证工作流程。DVBench通过编译、覆盖率和错误检测来评估最终的验证平台，同时评估包括特性列表、测试计划和参考模型在内的中间产物。我们评估了

    arXiv:2611.25066v1 Announce Type: cross  Abstract: Design verification (DV) remains a major bottleneck in hardware development, with large-scale verification using the Universal Verification Methodology (UVM) being highly labor-intensive. Recent advances in large language models (LLMs) and agentic systems offer new opportunities for DV automation. However, existing benchmarks primarily construct mutually decoupled subtasks, which deviates from the real-world workflow. In this paper, we present DVBench, the first end-to-end benchmark for Real-World UVM design verification agents. DVBench comprises 95 tasks across 9 functional domains and 5 difficulty levels, covering diverse verification scenarios and design scales. Each task requires agents to independently complete the full verification workflow. DVBench evaluates final testbenches via compilation, coverage, and bug detection, while assessing intermediate artifacts including feature lists, test plans, and reference models. We evaluate
    
[^27]: 从工业产品文档到自动化数字产品护照生成：大语言模型的实验研究

    From Industrial Product Documentation to Automated Digital Product Passport Generation: An Experimental Study of Large Language Models

    [https://arxiv.org/abs/2611.24998](https://arxiv.org/abs/2611.24998)

    本文提出一种基于检索增强生成（RAG）与大语言模型的方法，从分散异构的工业产品文档中自动提取产品信息，并依据资产管理壳（AAS）标准自动生成符合欧盟要求的数字产品护照（DPP）。

    

    计划在欧盟（EU）销售特定产品的制造企业将被要求提供强制性的数字产品护照（DPP）。数字产品护照旨在循环经济中为产品的多个生命周期提供标准化的产品信息。所需的护照信息，如技术规格、材料成分和环境产品足迹，虽然存在于制造商的数据源中，但分散在不同的异构系统、文档和数据格式之中。本文研究了利用大语言模型（LLM）基于资产管理壳（AAS）自动生成数字产品护照的方法。该研究采用检索增强生成（RAG）方法，从工业文档中提取产品信息并生成填充完成的AAS实例。产品文档经过处理后被分割为文本块，随后这些文本块被存储在持久化的向量数据库中（注：原摘要在此处被截断）。

    arXiv:2611.24998v1 Announce Type: new  Abstract: Manufacturing companies that are planning to sell certain products in the European Union (EU) will be required to provide mandatory digital product passports (DPPs). The use of DPPs is to provide standardized product information throughout multiple product life cycles in a circular economy. Required passport information, such as technical specifications, material composition, and environmental product footprint, is available in manufacturers' data sources but is scattered across different heterogeneous systems, documents, and data formats. The given paper investigates using large language models (LLMs) to automate the generation of DPPs based on the asset administration shell (AAS). A retrieval-augmented generation (RAG) approach is used to extract product information from industrial documentation and generate populated AAS instances. Product documentation is processed and divided into text chunks. The text chunks are then stored in a pe
    
[^28]: 展示你的工作：小型开源模型能否以机器可验证的证据读取SEC文件？

    Show Your Work: Can Small Open Models Read SEC Filings with Evidence a Machine Can Check?

    [https://arxiv.org/abs/2611.24990](https://arxiv.org/abs/2611.24990)

    该论文提出Filing-Evidence基准，验证了小型开源模型在单张消费级GPU上能够以机器可核验的页面与逐字引用证据从SEC财报文件中提取数字，最佳模型准确率达84.5%，但每千个回答中仍有16至167个错误答案能骗过所有自动检查。

    

    arXiv:2611.24990v1 公告类型：新 摘要：语言模型给出的流畅回答并不等于经过验证的回答。我们探究了在单张消费级GPU上运行的小型开源模型，能否以机器可验证的证据——即所在页面和包含该数字的逐字引用——从SEC（美国证券交易委员会）文件中提取数字。我们提出了Filing-Evidence基准，包含针对45家美国公司90份近期10-K和10-Q文件的1000个问题，答案标准取自申报方的内联XBRL标签。我们在一张RTX 4090上评估了来自十家开发商的十个指令模型（7-21B参数），采用三种设置：普通模式、检查重试模式以及模型自身的推理模式。在普通设置下，带有可验证证据的准确率从17.1%到82.7%不等，最佳结果为84.5%（Qwen3.5-9B）。自动检查器能够拒绝许多错误答案，但每1000个回答中仍有16到167个对可回答的问题给出了错误数值，却通过了全部检查。在普通设置下，其中57%是量级或单位错误，30%属于……（原文摘要在此处截断）

    arXiv:2611.24990v1 Announce Type: new  Abstract: A fluent answer from a language model is not a verified one. We ask whether small open models that run on a single consumer GPU can extract numbers from SEC filings with evidence a machine can check: the page and a verbatim quote containing the number. We introduce Filing-Evidence, a benchmark of 1,000 questions over 90 recent 10-K and 10-Q filings of 45 US companies, with answer keys taken from the filers' inline XBRL tags. We evaluate ten instruct models from ten developers (7-21B parameters) on one RTX 4090 in three setups: plain, check-and-retry, and the model's own reasoning mode. Accuracy with checkable evidence ranges from 17.1% to 82.7% in the plain setup, and the best result is 84.5% (Qwen3.5-9B). An automatic checker rejects many wrong answers, but between 16 and 167 answers per 1,000 give a wrong value for an answerable question and still pass every check. In the plain setup, 57% of these are scale or unit errors and 30% take 
    
[^29]: 安全AI评估：一种系统视角

    Safe AI Evaluation: A Systems Perspective

    [https://arxiv.org/abs/2611.24987](https://arxiv.org/abs/2611.24987)

    本文从系统视角剖析了AI评估过程中特有的安全风险与控制挑战，并提出多样化监督者、独立验证器、静默监控和协调关停四种架构模式以支持安全的AI评估。

    

    AI评估可能涉及一个主动的、自适应的优化过程，其中AI智能体会重新规划、重试、使用工具或委托工作以实现目标。在此过程中可能出现未预料到的能力或不安全行为，而现有的控制措施可能不足以防止危害。此外，某些选定的防护措施也可能被有意放宽，以观察它们原本会抑制的行为，从而带来额外的风险。本文研究了评估过程中特有的安全考量及其对控制措施部署位置、观察和干预方式的影响，并提出了四种支持安全AI评估的架构模式，包括多样化监督者、独立验证器、静默监控和协调关停。

    arXiv:2611.24987v1 Announce Type: new  Abstract: AI evaluation can involve an active, adaptive optimisation process in which AI agents re-plan, retry, use tools or delegate work to achieve a goal. Unanticipated capabilities or unsafe behaviour may emerge, and existing controls may be inadequate to prevent harm. Selected safeguards may also be deliberately relaxed to observe behaviour they would otherwise suppress, creating additional risks. This article examines the distinctive safety considerations of evaluation and their implications for control placement, observation and intervention, and presents four architectural patterns to support safe AI evaluation, including diverse supervisors, an independent verifier, silent monitoring and coordinated shutdown.
    
[^30]: Trace2ST：利用大语言模型生成与验证，从输入/输出轨迹中恢复PLC结构化文本程序

    Trace2ST: Recovering PLC Structured Text from I/O Traces with LLM Generation and Verification

    [https://arxiv.org/abs/2611.24979](https://arxiv.org/abs/2611.24979)

    本文提出“生成—回放—反馈”流水线，使大语言模型仅凭PLC的输入/输出轨迹记录即可恢复出可执行的结构化文本（ST）程序，并通过逐扫描周期回放比对完成程序验证。

    

    可编程逻辑控制器（PLC）是驱动电力与流程工厂运行的工业计算机，其上运行的程序源代码往往已不复存在；然而，控制器的每一个输入和输出都会被持续记录。因此，本文探讨大语言模型（LLM）能否仅凭这类输入/输出记录（轨迹）恢复出以结构化文本（ST，即IEC 61131-3标准PLC语言）编写的可执行程序，以及同样重要的是，如何对恢复出的程序进行验证。为回答这一问题，我们构建了“生成—回放—反馈”流水线：首先，LLM根据轨迹的压缩摘要编写候选程序，全程不接触原始程序；随后，候选程序按单个控制周期（扫描）逐步执行并与记录进行比对，一旦出现不匹配，便将轨迹中对应的部分返回给LLM，最多进行三次修订。（原摘要在此处截断）

    arXiv:2611.24979v1 Announce Type: cross  Abstract: Programmable logic controllers (PLCs), the industrial computers that operate power and process plants, often run programs whose source code is no longer available; however, every input and output of the controller is logged continuously. This paper therefore asks whether a large language model (LLM) can recover an executable program in Structured Text (ST), the standard PLC language of IEC 61131-3, from such input/output records (traces) alone and, equally important, how the recovered program can be verified. To answer this question, we build a generate-replay-feedback pipeline. First, the LLM writes a candidate program from a compressed summary of the trace, without ever seeing the original program. The candidate is then executed one control cycle (scan) at a time and compared with the recording, and whenever a mismatch occurs the corresponding part of the trace is returned to the LLM, for at most three revisions. Because every result
    
[^31]: MetaphIR：面向LLVM窥孔优化的语料库规模适用性分析

    MetaphIR: Corpus-Scale Applicability Analysis for LLVM Peephole Optimizations

    [https://arxiv.org/abs/2611.23555](https://arxiv.org/abs/2611.23555)

    该论文提出MetaphIR，可在LLVM IR语料库规模上同时评估窥孔优化重写规则的模式匹配与前置条件满足情况，相比忽略前置条件的结构化搜索计数更精确，且避免了完整重编译流程高昂的工程成本与47分钟级的单次运行耗时。

    

    窥孔优化将较短的源指令序列替换为更高效的等价序列，通常受前置条件约束，例如要求常量操作数必须是2的幂。由于每新增一条重写规则都会增加维护负担和编译时间成本，编译器开发者在决定实现某条优化之前，需要了解真实程序同时匹配源模式并满足其前置条件的频率。现有工作流程存在不足：对包含480万个函数的中间表示（IR）语料库进行结构化搜索会忽略前置条件，因而导致计数偏高；而实现重写并重新编译整个语料库虽能得到精确计数，但需要先付出工程实现成本，且该语料库的按需重编译流水线每次运行的中位数耗时达47分钟。当需要泛化一条重写规则时，这种延迟尤其成为瓶颈，因为开发者往往需要迭代多个变体以权衡覆盖率……（原文摘要在此处截断）

    arXiv:2611.23555v1 Announce Type: cross  Abstract: Peephole optimizations replace short source instruction sequences with more efficient equivalents, typically guarded by preconditions such as requiring a constant operand to be a power of two. Since every new rewrite adds maintenance burden and compile-time cost, compiler developers need to know how often real-world programs both match the source pattern and satisfy its preconditions before committing to an implementation. Existing workflows fall short: structural search over an intermediate representation (IR) corpus of 4.8 million functions ignores preconditions and therefore overcounts, whereas implementing the rewrite and recompiling the corpus yields exact counts only after paying the engineering cost, with each run of the corpus's on-demand recompilation pipeline taking a median of 47 minutes. This delay is especially limiting when generalizing a rewrite, as developers often iterate over multiple variants to trade off coverage ag
    
[^32]: 重新审视智能体代码的存续：测量方法的选择如何决定结论

    Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict

    [https://arxiv.org/abs/2611.21784](https://arxiv.org/abs/2611.21784)

    本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。

    

    arXiv:2611.21784v1 公告类型：交叉

    arXiv:2611.21784v1 Announce Type: cross  Abstract: Studies of how long code from AI coding agents survives after it is merged reach conflicting verdicts: agent code is reported to be retained longer than human code, about as long, or modified sooner. Each study makes defensible measurement choices, but how much these choices decide the verdict is unknown. We revisit the survival of agent-attributed code in our earlier study: 854,139 lines that 2,802 merged pull requests (PRs) added to 215 repositories of the AIDev dataset. We vary nine measurement choices in five research questions and build a specification curve of 88 survival analyses, tested under three variance estimators, several observation windows, and permuted labels. For code from human-supervised agents, mostly GitHub Copilot, the choices change the size of the effect but not its direction. For code from the fully-autonomous agent Devin, they decide the verdict. Across repositories, 15 of 22 specifications report it modified 
    
[^33]: 基于大语言模型的科学编程：来自有限元分析的证据

    Scientific Coding with Large Language Models: Evidence from Finite Element Analysis

    [https://arxiv.org/abs/2611.21176](https://arxiv.org/abs/2611.21176)

    该论文将FEM-Bench扩展至17个涉及非线性力学与结构动力学的有限元编程任务，对15个大语言模型的评估表明，单次生成的正确率有限，而结合代码执行与修订的智能体框架能显著提升性能（最高达96.5% Pass@3），揭示了LLM在自动化科学编程中的潜力与局限。

    

    大语言模型（LLM）为自动化科学编程提供了一条潜在途径，但其实用性取决于能否持续生成数值正确的代码。我们将FEM-Bench扩展至17个涉及非线性和历史相关力学以及结构动力学的有限元任务，并在无执行反馈的条件下对15个模型的代码实现和测试生成能力进行了评估。表现最好的模型在单次实现试验中的成功率达到87.1%，但两个赛道的排名并不一致：最佳实现模型在联合测试中成功率为49.9%，而最佳测试生成模型达到82.6%。对于其中六个模型，我们比较了单次生成与两种允许代码执行和修订的智能体框架。智能体执行解决了一些在单次尝试中从未成功的任务，并提高了其他任务的稳定性，但收益因模型和框架而异，且也出现了性能回退的情况。Claude Opus 5结合SimpleAgent达到了96.5%的Pass@3。

    arXiv:2611.21176v1 Announce Type: cross  Abstract: Large language models (LLMs) offer a potential route to automating scientific programming, but their usefulness depends on consistently producing numerically correct code. We extend FEM-Bench to 17 finite element tasks involving nonlinear and history-dependent mechanics and structural dynamics, and evaluate 15 models on implementation and test generation without execution feedback. The best model succeeds in 87.1% of single-pass implementation trials, but rankings differ between tracks. The leading implementer achieves 49.9% joint test success, while the leading tester reaches 82.6%. For six models, we compare single-pass generation with two agentic harnesses that allow code execution and revision. Agentic execution solves some tasks that never succeeded in single-pass trials and improves consistency on others, but gains vary across models and harnesses, and regressions also occur. Claude Opus 5 with SimpleAgent reaches 96.5% Pass@3, t
    
[^34]: 轨迹所示与标签所言：审计智能体评估中的行为检测器

    What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation

    [https://arxiv.org/abs/2610.30491](https://arxiv.org/abs/2610.30491)

    该研究通过在GSM8K上构建包含损坏计算器输出的工具使用场景，审计了三个轨迹级行为检测器，发现最终正确性、可见重新验证与错误传播等标签会系统性不一致，并揭示了“静默工具输出覆盖”这一关键失败模式——即模型在不重新调用工具的情况下使用了与损坏输出不同的值。

    

    大语言模型智能体通常通过将执行轨迹简化为最终答案正确性、工具验证和错误传播等标签来进行评估。这些标签是由解析器、行为检测器和评估框架规则产生的度量结果，它们可能无法完整捕捉轨迹所显示的内容。我们在基于GSM8K构建的工具使用场景中研究这一问题，其中智能体会调用一个输出偶尔被损坏的计算器，这为工具返回了什么、模型后续使用了什么以及检测器报告了什么提供了明确的真值参照。我们在六个生产级模型上、每个模型30道题目，审计了三个轨迹级检测器。最终正确性、可见的重新验证和错误传播之间可能以系统性的方式出现不一致。一个关键的失败模式是静默的工具输出覆盖：计算器返回了一个被损坏的值，但模型后续在没有再次调用工具的情况下使用了另一个（通常是正确的）值，因此坏值停止了传播，而（摘要在此处被截断）

    arXiv:2610.30491v1 Announce Type: cross  Abstract: LLM agents are often evaluated by reducing execution traces to labels such as final-answer correctness, tool verification, and fault propagation. These labels are measurements produced by parsers, behavioral detectors, and harness rules, and they may not fully capture what the trace shows. We study this in a tool-use setting built on GSM8K, where agents call a calculator whose outputs are occasionally corrupted, giving clear ground truth for what the tool returned, what the model later used, and what the detectors reported. We audit three trace-level detectors across six production models and 30 problems per model. Final correctness, visible re-verification, and fault propagation can disagree in systematic ways. A key failure mode is silent tool-output override: the calculator returns a corrupted value, but the model later uses a different, often correct, value without calling the tool again, so the bad value stops propagating while a 
    
[^35]: 从C代码库到需求自动构建机器可验证的保证案例

    Automatically Building Machine-Checked Assurance Cases from C Codebases to Requirements

    [https://arxiv.org/abs/2609.40119](https://arxiv.org/abs/2609.40119)

    本文提出了LLM辅助框架CCV，能够从C代码库自动构建机器可验证的保证案例，通过接口协议建模允许的调用序列并协调需求引导分析与自底向上验证两个阶段，从而证明代码库满足预期需求。

    

    arXiv:2609.40119v1 公告类型： cross 摘要：大型语言模型（LLM）在自动化交互式定理证明方面已展现出前景，然而对真实世界C代码库的验证不仅仅是完成单个证明目标。该任务需要联合构建具有表达力的函数规范及其证明，并确保库接口即使在没有指定客户端的情况下，也能按照预期的调用序列进行组合。本文提出了CCV，一个LLM辅助的框架，用于构建机器可验证的保证案例：即支持“C代码库满足其预期需求”这一主张的结构化、可审计的工件。为了在开放库中建模预期的跨接口使用，CCV构建了一个接口协议，该协议公开允许的调用序列和资源假设以供审查，并在经过验证的契约和调用者义务的前提下提供条件性安全保证。CCV协调两个互补的阶段：（i）需求引导的分析与自底向上的构建

    arXiv:2609.40119v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown promise in automating interactive theorem proving, yet verification of real-world C codebases requires more than discharging individual proof goals. The task involves jointly constructing expressive function specifications and their proofs, and ensuring that library interfaces compose along intended call sequences even without a designated client. This paper presents CCV, an LLM-assisted framework for building machine-checked assurance cases: structured, auditable artifacts supporting the claim that a C codebase meets its intended requirements. To model intended cross-interface use in open libraries, CCV constructs an interface protocol that exposes permitted call sequences and resource assumptions for review, with a conditional safety guarantee under verified contracts and caller obligations. CCV coordinates two complementary phases: (i) requirement-guided analysis and bottom-up construction of 
    
[^36]: 将推理轨迹提炼为软件工程任务的咨询式提示

    Distilling Reasoning Traces into Advisory Prompts for Software Engineering Tasks

    [https://arxiv.org/abs/2608.00437](https://arxiv.org/abs/2608.00437)

    受人类通过反思错误并内化规则来学习编程的启发，本文提出将LLM思考模式中避免错误的推理轨迹提炼为可内化的咨询式提示，使软件工程任务无需持续承担推理成本即可获得性能提升。

    

    语言模型被广泛用于生成和处理代码（例如，预测行为、输出或可能的输入），本文探讨能否在不总是承担推理成本的情况下获得更好的性能。人类编程学习者通过以下方式学会避免错误：（a）首先识别错误，（b）反思导致这些错误的认知疏漏（本质上是“深入思考”错误），（c）从这些反思中推断出一般性规则或经验教训，（d）将这些教训内化为规则。在导师参与的辅导课中，这是一种非常常见的苏格拉底式师生互动。这类可内化的规则的例子可能包括“编码之前，重述需求以澄清它们”这样的要点。受这一过程的启发，本文描述了一种方法：首先识别（低资源）LLM 中“思考模式”避免错误的例子；然后，这些错误……（原文摘要截断于此）

    arXiv:2608.00437v2 Announce Type: replace-cross  Abstract: Language models are widely used for generating and otherwise processing code (e.g., predicting behavior, outputs, or likely inputs), and this paper asks whether better performance can be achieved without always incurring the cost of reasoning. Human students of programming learn to avoid mistakes by (a) first identifying them, (b) reflecting upon cognitive lapses that led to these mistakes (essentially, "thinking through" the errors), (c) inferring general rules or lessons from these reflections, and (d) internalizing these lessons into rules. In tutorial sessions with an instructor, this is a very common Socratic interaction with students. Examples of such internalizable rules might include the nugget "Before coding, restate the requirements to clarify them". Inspired by this process, this paper describes an approach where we first identify examples in which "thinking mode" in a (low-resource) LLM avoids errors; then, these er
    
[^37]: 洞察生成器：面向大语言模型智能体的系统性语料库级轨迹诊断

    Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents

    [https://arxiv.org/abs/2605.21347](https://arxiv.org/abs/2605.21347)

    该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。

    

    arXiv:2605.21347v4 公告类型：replace-cross 摘要：诊断大语言模型（LLM）智能体的失败在很大程度上仍然是手工完成的。从业者通常只检查一小部分执行轨迹，形成临时性的假设，然后不断迭代。这一过程会遗漏那些只有在轨迹群体层面才会显现的模式，也无法扩展到单条轨迹就包含数万 token 的生产级语料库。我们形式化了语料库级轨迹诊断问题：给定一个执行轨迹语料库，目标是在轨迹群体上刻画系统性的行为模式，生成有据可依的自然语言洞察，并且每条洞察都关联相应的支持性证据。我们提出了洞察生成器（Insights Generator, IG），这是一个多智能体系统，它通过在轨迹语料库上提出并检验假设来回答诊断问题，最终产出有证据支撑的洞察报告。我们从定性和客观两个维度评估了 IG，包括基于评分量表的报告评估，以及通过实施洞察所带来的下游性能提升……

    arXiv:2605.21347v4 Announce Type: replace-cross  Abstract: Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by impl
    
[^38]: 模型压缩与对抗鲁棒性：面向代码的语言模型实证研究

    Model Compression vs. Adversarial Robustness: An Empirical Study on Language Models for Code

    [https://arxiv.org/abs/2508.03949](https://arxiv.org/abs/2508.03949)

    本文系统评估了剪枝、量化和知识蒸馏等常见压缩策略对三种主流代码语言模型在对抗攻击下鲁棒性的影响，为压缩模型的安全部署提供了实证依据。

    

    基于Transformer的代码语言模型在各类软件分析任务中展现出卓越的性能，但其广泛应用受到高计算成本、缓慢的推理速度以及巨大的环境影响所阻碍。剪枝、量化和知识蒸馏等模型压缩技术在应对这些挑战方面日益受到关注。然而，这些压缩策略对代码语言模型在对抗场景下鲁棒性的影响仍鲜为人知。理解这些压缩模型在对抗攻击下的行为表现，对于其在真实世界应用中的安全有效部署至关重要。为填补这一知识空白，我们对常见压缩策略如何影响压缩模型的对抗鲁棒性进行了全面评估。我们评估了三种广泛使用的代码语言模型的压缩版本在三种（场景下的）鲁棒性……

    arXiv:2508.03949v3 Announce Type: replace  Abstract: Transformer-based language models for code have shown remarkable performance in various software analytics tasks, but their adoption is hindered by high computational costs, slow inference speeds, and substantial environmental impact. Model compression techniques such as pruning, quantization, and knowledge distillation have gained traction in addressing these challenges. However, the impact of these strategies on the robustness of compressed language models for code in adversarial scenarios remains poorly understood. Understanding how these compressed models behave under adversarial attacks is essential for their safe and effective deployment in real-world applications. To bridge this knowledge gap, we conduct a comprehensive evaluation of how common compression strategies affect the adversarial robustness of compressed models. We assess the robustness of compressed versions of three widely used language models for code across three
    
[^39]: 通过LTL离散事件控制的组合框架扩展GR(1)综合的可扩展性

    Scaling GR(1) Synthesis via a Compositional Framework for LTL Discrete Event Control

    [https://arxiv.org/abs/2506.16557](https://arxiv.org/abs/2506.16557)

    该论文提出了一种组合式的GR(1)控制器综合框架，利用被控系统的模块化结构和观测综合等价性来缓解状态爆炸问题，能够求解规模达到单体式方法1000倍的LTL离散事件控制综合问题。

    

    我们提出了一种组合式方法，用于综合具有线性时序逻辑（LTL）目标的离散事件系统控制器。我们利用被控对象（以一组标记转移系统（LTS）的形式给出）的模块化结构，来缓解单体式综合方法容易出现的状态爆炸问题。通过求解较弱的控制问题，为被控对象LTS的子集迭代地构建最大宽容安全控制器。利用观测综合等价性，通过抽象掉局部事件来减小被控对象受控子集的规模。综合的结果也是组合式的，即一组控制器，当它们并行运行时能确保实现LTL目标。我们在MTSA工具中针对LTL的一个富有表达力的子集GR(1)实现了该综合方法，并证明它所能计算的解的规模可达单体方法所能求解规模的1000倍。

    arXiv:2506.16557v3 Announce Type: replace  Abstract: We present a compositional approach to controller synthesis of discrete event system controllers with linear temporal logic (LTL) goals. We exploit the modular structure of the plant to be controlled, given as a set of labelled transition systems (LTS), to mitigate state explosion that monolithic approaches to synthesis are prone to. Maximally permissive safe controllers are iteratively built for subsets of the plant LTSs by solving weaker control problems. Observational synthesis equivalence is used to reduce the size of the controlled subset of the plant by abstracting away local events. The result of synthesis is also compositional, a set of controllers that when run in parallel ensure the LTL goal. We implement synthesis in the MTSA tool for an expressive subset of LTL, GR(1), and show it computes solutions to that can be up to 1000 times larger than those that the monolithic approach can solve.
    

