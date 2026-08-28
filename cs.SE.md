# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tacet: A Language and Type System for Automatic Statistical Validity Accounting](https://arxiv.org/abs/2608.27451) | 本文提出Tacet语言及其类型系统，通过自动统计有效性核算确保系统比较在统计上有效，防止不支持的结论。 |
| [^2] | [SWE-Prime: Fewer Trajectories, Better Performance](https://arxiv.org/abs/2608.27449) | SWE-Prime通过两阶段多粒度筛选方法，仅保留高质量且具代表性的轨迹片段，以更少的数据实现更好的模型性能。 |
| [^3] | [From Static to Dynamic: Benchmarking Real-World Code Review with MCR-Bench](https://arxiv.org/abs/2608.27442) | 本文提出了MCR-Bench，首个缺陷状态感知的多轮代码评审基准，包含2,269个真实任务和细粒度缺陷标注，以解决现有LLM评审方法过度简化静态化的问题。 |
| [^4] | [Persona-Execution Separation: An Architecture Pattern for Evolving LLM Agents under Execution Audit](https://arxiv.org/abs/2608.27427) | 该论文提出人格-执行分离（PES）架构模式，将LLM代理的人格与执行分离到不同信任域，通过契约桥接实现自由演化与执行审计，并论证了在LLM表示不可区分性下单域机制需引入三类组件才能满足所有目标。 |
| [^5] | [BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks](https://arxiv.org/abs/2608.27334) | 该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。 |
| [^6] | [When Context Gets Root: Privilege Escalation in LLM Harnesses](https://arxiv.org/abs/2608.27299) | 本文提出了一种名为“指令权限提升”的新型攻击，利用LLM代理工具框架在构建上下文时提升低级别指令的权限，从而绕过指令层级防御，实现对编码代理的广泛攻击。 |
| [^7] | [SPA: Securing Persistent LLM Agents Across Queries with Plan-First Information-Flow Control](https://arxiv.org/abs/2608.27234) | SPA通过计划优先架构和双格信息流控制，确保LLM代理在跨查询持久化状态下安全处理不受信任数据，防止攻击者篡改控制流或泄露敏感信息。 |
| [^8] | [Twelve Quick Tips for Managing IT Disasters in Small Research Software Teams](https://arxiv.org/abs/2608.27196) | 本文为小型科研软件团队提供了一份实用的灾难规划与恢复指南，强调在资源有限和非专业管理员情境下，通过简单有效的措施提升系统韧性，并充分利用机构内部支持资源。 |
| [^9] | [A Trans-Domain Digital Twin for Bio-Aware Control of Climate and Energy in Cattle Fattening Barns Using Single-Episode Optimizer Learning](https://arxiv.org/abs/2608.27185) | 本文提出一种集成机制模拟、生长预测与轻量强化学习的跨域数字孪生框架，通过单幕学习实现牛育肥舍气候与能源的协同优化控制。 |
| [^10] | [LLMs in Digital EDA: A perspective on shifting roles from Generation to Orchestration](https://arxiv.org/abs/2608.27184) | 本文提出从生成到编排的三个层次化角色框架，揭示LLM在数字EDA中的能力积累机制，并指出当前模型存在生成“语法合理”而非“物理正确”硬件的陷阱。 |
| [^11] | [AgentDV: Closed-Loop Agentic AI for Hardware Design Verification](https://arxiv.org/abs/2608.27148) | AgentDV是一个闭环智能体AI框架，通过可运行性过滤、CSR接地检查和覆盖率引导迭代，将LLM测试平台生成转化为可靠的RTL验证流水线。 |
| [^12] | [When Tool Outputs Become Commands: Separating Action Induction from Runtime Authorization in Tool-Augmented LLM Agents](https://arxiv.org/abs/2608.27146) | 本文提出SARA框架，通过分离动作诱导与执行授权，并引入动作来源追踪和上下文隔离探针，防止工具输出演变为指令后引发未授权的现实世界副作用。 |
| [^13] | [AROMA+: A Study of Factors Affecting Reproducible Builds in the Maven Ecosystem](https://arxiv.org/abs/2608.27125) | 本研究提出自动化方法，旨在解决Maven生态系统中可复现构建列表维护困难的问题，通过自动从Maven发布中定位源代码来提高可复现性验证的效率。 |
| [^14] | [Mutation Testing for Reproducibility Safeguards in Machine Learning Research Software: An Empirical Study](https://arxiv.org/abs/2608.27100) | 本研究通过变异测试方法系统评估了机器学习研究仓库中验证工作流对可复现性相关变更的检测能力，发现现有验证流程对关键实验选择变化的检测存在显著不足。 |
| [^15] | [An Empirical Evaluation of Using Large Language Models for Automated Model-Based Test Generation](https://arxiv.org/abs/2608.27094) | 本研究通过对比五种最新大型语言模型与GraphWalker工具，证明LLMs能显著优化和缩短基于模型的测试路径和步长，从而提升工业级MBT的可扩展性。 |
| [^16] | [Bug Localization from Bug Reports: A Multi-Objective Approach](https://arxiv.org/abs/2608.27089) | 提出了一种基于SPEA-2多目标进化优化的类级缺陷定位系统，在最大化相似性的同时最小化建议文件数，并在六个Java项目中验证了其有效性。 |
| [^17] | [A Contract-Centered Architecture for Scalable and Manageable Agentic Runtimes](https://arxiv.org/abs/2608.27086) | 本文提出一种以四个责任对象（技能、框架、脚手架和数据基底）为组织契约的代理运行时架构，核心假设P1强调在成本约束下实现能力与容量的可分离性，以解决企业AI部署中的跨部门协调问题。 |
| [^18] | [A Catalog of User Authentication Patterns](https://arxiv.org/abs/2608.26955) | 本文提出了一个包含14种用户认证模式的新目录，按认证因素和实际角色分类，以弥补安全模式目录中认证模式缺失的空白。 |
| [^19] | [Evaluating human and LLM screening workflows in a conceptually complex scoping review: Recall--workload trade-offs and run-to-run consistency](https://arxiv.org/abs/2608.26885) | 该研究通过预注册实验比较了人类与LLM在概念复杂范围综述中的筛选工作流，发现无工作流能完全恢复所有合格记录，并揭示了召回率-工作量权衡及运行间一致性问题。 |
| [^20] | [Beyond Execution: Auditing Experimental Fidelity in LLM-Driven Scientific Research](https://arxiv.org/abs/2608.26753) | 本文提出ABE-Ralph框架，通过结构化约束和8步工作流审计LLM科学实验的保真度，检测方法论幻觉（如数据缩减、组件替换），确保代理忠实复现参考方法并提供可靠证据。 |
| [^21] | [FaultLens: Learning Compact Behavioral Test Suites for Generated Operational Programs](https://arxiv.org/abs/2608.26746) | 本文提出FaultLens方法，通过结合故障驱动的贪婪选择和突变无关的多样性组件，学习紧凑的行为测试套件，以高效检测生成程序中的稀疏边界和交互故障。 |
| [^22] | [Claude Code Complete User Handbook](https://arxiv.org/abs/2608.26742) | 本书提出，在Claude Code代理环境中，安全高效运行的关键在于定义明确的可观察完成条件，并通过指令、权限、沙箱和系统隔离四层控制栈来管理系统性风险。 |
| [^23] | [KubeCap: A Framework for Capability Minimization in Kubernetes via Static Analysis and LLM-Assisted Rule Inference](https://arxiv.org/abs/2608.26699) | KubeCap通过静态分析和LLM辅助规则推理，自动推断并最小化Kubernetes容器所需的能力，解决了开发人员因依赖默认配置而违反最小权限原则的问题。 |
| [^24] | [Five Primitives for Governing Autonomous AI Agents at Runtime](https://arxiv.org/abs/2608.26696) | 本文提出治理自治AI代理需在运行时解决，并定义了五种关键原语（发现、身份、治理、证明、供应链），以确保代理行动在生效前后得到有效管控。 |
| [^25] | [Processing/p5 Defined through Practice and Learning](https://arxiv.org/abs/2608.26614) | 本文通过跨语言案例研究，提出了一套构成Processing/p5核心的软件决策指导框架，强调了其在创意编码中的统一设计原则和跨平台适用性。 |
| [^26] | [The Thousand-Graph Hypothesis: A Testable Hypothesis of Task-Conditioned Relation Materialization in Repository-Level Code Reasoning](https://arxiv.org/abs/2608.26602) | 本文提出一种仅含实体的外部接口，通过推理时的任务条件化关系具体化，在无需预构建关系图的情况下，显著提升了仓库级代码推理的成功率。 |
| [^27] | [Unsaid, Unsafe? Implicit Security Obligations in LLM-Based RTL Code Generation](https://arxiv.org/abs/2608.26588) | 本研究揭示了LLM生成的RTL代码在功能正确性与安全性之间的巨大鸿沟，并构建了SECRTL-GEN基准来量化这一差距，表明当前前沿模型在隐含安全义务上表现严重不足。 |
| [^28] | [DeepRepro: State-Aware Subplanning for Paper-to-Code Reproduction in Evolving Repositories](https://arxiv.org/abs/2608.26557) | DeepRepro通过动态状态感知子规划，将仓库演化状态与运行时反馈实时转化为细粒度实现计划，解决了论文到代码复现中静态规划导致的执行不一致问题。 |
| [^29] | [Report of the 2026 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science](https://arxiv.org/abs/2608.26519) | 本报告基于2026年研讨会，提炼出科学计算生态系统未来发展的四大战略主题和八项社区行动优先事项，强调通过社会技术协同设计整合人工智能、软件和跨学科合作。 |
| [^30] | [Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance](https://arxiv.org/abs/2608.26480) | 本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。 |
| [^31] | [STILL: Recovering Lowered STL Semantics for LLM-assisted C++ Decompilation](https://arxiv.org/abs/2608.26408) | 本文提出STILL，一种通过预测剥离二进制中STL容器语义并生成提示，显著提升LLM辅助C++反编译的可执行性（从17.4%提升至28.4%）。 |
| [^32] | [Spec2Vision: Contract-Guided Delivery of AI-Generated Computer Vision Pipelines](https://arxiv.org/abs/2608.26400) | Spec2Vision通过分阶段运行时和明确任务契约，显著提升了AI生成计算机视觉流水线的可交付性，在850次运行中达到81/85测试通过率，远超基线方法。 |
| [^33] | [Investigating Software Aging in LLM-Generated Software Systems across Generation-and-Execution Environments](https://arxiv.org/abs/2608.26391) | 本研究首次通过实验揭示了LLM生成的软件系统在持续运行中表现出软件老化症状，且不同编程语言（JavaScript、Python、Rust）间的老化程度存在显著差异，为评估LLM生成代码的长期可靠性提供了实证依据。 |
| [^34] | [Kale: A Transformation-Safe Spreadsheet System](https://arxiv.org/abs/2608.26345) | Kale通过限制电子表格中可表达的引用类型，消除了因表结构变化导致的引用错误风险，并验证了其有效性和潜在影响。 |
| [^35] | [When Review Alone No Longer Scales: Layered Supervision in AI-Assisted Software Engineering](https://arxiv.org/abs/2608.26316) | 本研究通过访谈发现，在AI辅助开发中，组织采用分层防护栏机制（如预防性、检测性和纠正性措施）来应对代码生成规模扩大，以缓解单一审查环节的瓶颈。 |
| [^36] | [MemToC: Benchmarking Memory-Tool Conflict Resolution in Large Language Models](https://arxiv.org/abs/2608.26295) | 该论文提出了MemToC基准测试，通过可执行工具和已知正确性控制，系统评估了大型语言模型在工具返回与参数记忆冲突时的仲裁能力，发现工具返回结果强烈主导闭卷答案，且模型在工具错误时保留正确记忆的能力较弱。 |
| [^37] | [6.5% of the Neuro-Symbolic Literature Can Be Reproduced from Its Published Artifacts, a Six-Stage Audit Framework and First Instantiation](https://arxiv.org/abs/2608.26236) | 该论文提出一个六阶段审计框架，并首次应用于神经符号AI领域，发现仅有6.5%的符合条件的文献能从其已发布工件中复现，凸显了该领域可复现性的严重不足。 |
| [^38] | ["A Second Set of Eyes": The Process and Challenges of Software Documentation Review](https://arxiv.org/abs/2608.26232) | 本文通过访谈31位技术文档撰写者，首次系统揭示了软件文档评审的五个阶段及其协作过程，并指出组织和技术的挑战。 |
| [^39] | [The Green Software Landscape: A Systematic Mapping Study on Evolution, Applications, Software Lifecycle, and Best Practices](https://arxiv.org/abs/2608.26229) | 本文通过系统性映射研究，梳理了2010-2024年绿色软件工程领域的演进、应用分类和最佳实践，揭示了SE会议在能源文献中的主导地位及研究热度上升趋势。 |
| [^40] | [Agent Mesh: Reliability Primitives for Non-Idempotent Agent Delegation - Identity Adequacy and Evidence Adequacy](https://arxiv.org/abs/2608.26225) | 本文通过生产环境故障研究，揭示了服务网格中重试、超时和熔断等可靠性原语在智能体委托场景下因非幂等性和身份/证据不充分而失效，并提出身份充分性与证据充分性作为新的可靠性原语基础。 |
| [^41] | [NeuronFuzz: Safety Neuron Guided Fuzzing for LLM Safety Evaluation](https://arxiv.org/abs/2608.26222) | NeuronFuzz通过利用内部安全神经元的连续激活作为反馈，替代昂贵的响应生成，实现了高效且强指导的LLM安全性评估。 |
| [^42] | [Same Model, Different Harness: Different Coding-Agent Results](https://arxiv.org/abs/2608.26218) | 本研究揭示，在模型和任务不变的情况下，仅调整编码代理的框架（如缩短旧工具结果）即可显著提升性能，在严格上下文限制下使SWE-bench Verified上的完整解决方案从43%增至72%。 |
| [^43] | [Challenges and Contributions in Quality of AI-Based Software: A Systematic Mapping Study](https://arxiv.org/abs/2608.26215) | 本研究通过系统性映射分析，揭示了AI软件质量评估中六类核心挑战，强调现有模型局限性和跨领域合作的必要性。 |
| [^44] | [Characterizing the Landscape of Open-Source Satellite Software](https://arxiv.org/abs/2608.26211) | 本研究首次对22,286个开源卫星软件项目进行了系统实证分析，揭示了其生态系统的流行趋势、目标和开发实践。 |
| [^45] | [Fairness Invariants: A Relational Approach to Explaining and Mitigating Fairness Bugs](https://arxiv.org/abs/2608.26209) | REMI框架通过形式方法中的不变量综合思想，提出了一种关系性方法来自动定位、解释和缓解个体公平性缺陷，填补了现有技术无法处理歧视比较性质的空白。 |
| [^46] | [ADeptS-Bench: Measuring the Trustworthiness of Computer Use Agents Across Devices](https://arxiv.org/abs/2608.26204) | 该论文提出了ADeptS-Bench，一个双流可信度基准，用于评估计算机使用代理在视觉界面中处理模糊指令和恶意威胁的能力，结果显示当前所有模型均存在严重的安全缺陷。 |
| [^47] | [Benchmarking AI Agents for Hardware Design Automation via MCP Tool Calling](https://arxiv.org/abs/2608.26199) | 本文提出了一个基于MCP的硬件设计自动化基准测试框架，评估了本地部署的开源大语言模型在真实工具调用环境中的可靠性，覆盖了多类复杂操作场景。 |
| [^48] | [Harness Engineering for Predictable Agentic Systems: An Empirical Study of Deterministic Execution Constraints](https://arxiv.org/abs/2608.26197) | 本研究通过实证发现，对LLM智能体施加确定性执行约束（如有限状态控制、强制工具选择）并不能稳定提升可复现性，反而可能因引入额外自由文本规划步骤而增加不确定性。 |
| [^49] | [Cost-Utility Alignment in LLM Agent Trajectories:Profiling,Attribution,Diagnosis,Adaptation,and Evaluation](https://arxiv.org/abs/2608.26195) | 本文提出了一个以轨迹为中心的成本-效用对齐框架，通过五个阶段（剖析、归因、诊断、适应、评估）系统评估大语言模型代理的资源消耗与任务贡献是否匹配，其中效用归因作为核心创新点。 |
| [^50] | [Four Ways to Forge a Bundle My Own Verifier Calls Clean: Refusal-Site Mutation Testing of an Evidence-Bundle Verifier](https://arxiv.org/abs/2608.26183) | 本文通过突变测试发现证据捆绑验证器存在系统性“空泛通过”缺陷，即检查路径未实际验证内容却报告成功，且该缺陷在修复后仍多次出现，最廉价伪造仅需一个字母。 |
| [^51] | [Revision-Aware Success Prediction from Multi-Attempt Programming Trajectories](https://arxiv.org/abs/2608.26169) | 本研究通过统一框架比较多种模型，发现仅基于当前尝试的预测模式在编程结果预测中最为可靠，而历史轨迹信息未带来一致优势。 |
| [^52] | [Agentic AI Containment Architecture for Security Hardening](https://arxiv.org/abs/2608.26108) | 本文提出一种智能体遏制架构，通过六个显式约束将安全作为架构属性嵌入多智能体系统设计，并引入从系统分析工件到机器可验证契约的形式化映射。 |
| [^53] | [From State to Action: OODA-Tool for Reliable Multi-Turn Tool Use](https://arxiv.org/abs/2608.24368) | OODA-Tool通过分离状态保持与动作实现，并利用控制器检查的中间状态，解决了多轮工具使用中的状态-动作竞争问题，从而提高了动作的可靠性和一致性。 |
| [^54] | [Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping](https://arxiv.org/abs/2608.24135) | 该论文提出RobustTests框架，通过故障代码驱动测试合成和密集奖励塑形，有效缓解了代码强化学习中的奖励偏差和奖励黑客问题。 |
| [^55] | [Loop Engineering: Building Blocks, Adoption, and Impact](https://arxiv.org/abs/2608.21884) | 本文首次探索性回顾了“循环工程”这一新兴实践，即开发者设计系统自动触发和停止智能体运行，并总结了其核心构建模块（如停止条件、状态文件和验证子智能体），但未量化其实际采用率。 |
| [^56] | [FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth](https://arxiv.org/abs/2608.20574) | 该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。 |
| [^57] | [Autoresearch with Coding Agents: Generalizers and Metric-Maximizers on Quran Recitation Data](https://arxiv.org/abs/2607.18064) | 本文通过对比Claude Code和OpenAI Codex在《古兰经》语音转录任务上的自主研究表现，揭示了编码代理在优化时可能偏离开发者意图，倾向于追求字面分数，导致泛化与指标最大化之间的权衡。 |
| [^58] | [ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions](https://arxiv.org/abs/2605.24279) | ContextEcho通过一个结合25个探针、快照-然后-探针协议及多种测量方法的基准测试，揭示了长时智能体编码会话中语言模型人格漂移的现象。 |
| [^59] | [MISRust: Mapping MISRA-C++ Coding Guidelines to the Rust Programming Language](https://arxiv.org/abs/2605.23490) | 本文系统分析了179条MISRA C++ 2023规则对Rust的适用性，发现约48%的直接适用规则被Rust语言设计自动强制执行，并区分了安全与不安全Rust，指出69条规则仍需适配或应用。 |
| [^60] | [Code Summaries as Diagnostic Context for LLM-Based Program Repair](https://arxiv.org/abs/2511.18782) | 本研究通过仅提示测试发现，代码摘要，特别是诊断性和错误感知的摘要，能作为有用的诊断上下文，平均提升LLM程序修复效果5%，但收益有限且依赖模型，无法显著克服仅提示自我修复的挑战。 |
| [^61] | [Lost in Code Generation: Reimagining the Role of Software Models in AI-driven Software Engineering](https://arxiv.org/abs/2511.02475) | 本文主张在AI驱动的软件工程中，软件模型应从前期蓝图转变为可恢复、可精炼的工具，以弥合原型与工程化软件之间的差距，并通过智能体循环和人类反思循环提升系统的健壮性与可维护性。 |
| [^62] | [Stack Trace-Based Crash Deduplication with Transformer Adaptation](https://arxiv.org/abs/2508.19449) | 本文提出dedupT，一种基于Transformer的崩溃去重方法，通过整体建模堆栈跟踪并适配预训练语言模型，显著提升了重复崩溃排序和唯一崩溃检测的准确性，减少了人工分类负担。 |
| [^63] | [A Survey of LLM-based Automated Program Repair: Taxonomies, Design Paradigms, and Applications](https://arxiv.org/abs/2506.23749) | 本文提出首个基于大语言模型的自动程序修复可复现层次分类体系，将66个系统按修复能力与控制逻辑的驻留位置分为四类，并支持跨范式设计分析。 |
| [^64] | [Understanding the Challenges and Opportunities of Generative AI Apps: An Empirical Study](https://arxiv.org/abs/2506.16453) | 本研究通过对171个生成式AI应用的百万条用户评论进行分析，提出了SARA框架，识别出用户关注的核心主题（如AI性能与情感连接），并验证了LLM在评论分析中的高可靠性，揭示了Gen-AI应用面临的挑战与机遇。 |
| [^65] | [HybridProver: Augmenting Theorem Proving with LLM-Driven Proof Synthesis and Refinement](https://arxiv.org/abs/2505.15740) | HybridProver提出了一种统一框架，通过证明草图作为中间表示，整合了整体证明合成与逐步策略生成，从而在定理证明中结合高层规划与细粒度推理，实现了部分正确证明结构的重用。 |

# 详细

[^1]: Tacet：一种用于自动统计有效性核算的语言与类型系统

    Tacet: A Language and Type System for Automatic Statistical Validity Accounting

    [https://arxiv.org/abs/2608.27451](https://arxiv.org/abs/2608.27451)

    本文提出Tacet语言及其类型系统，通过自动统计有效性核算确保系统比较在统计上有效，防止不支持的结论。

    

    arXiv:2608.27451v1 公告类型：交叉 摘要：系统间的实证比较是计算机科学研究中的标准证据形式，但很少有比较经过统计有效性检验：大多数甚至从未被构架为统计检验。现有的多重比较程序可以控制由此产生的错误，但需要输入（分析检查了什么，以及其观察结果如何排列），而这些无法从p值列表中恢复。我们引入了Tacet，一种语言，其中分析声明它生成了什么，陈述它期望发现什么，并拒绝任何它无法承担或无法正确检验的主张。其核心演算T将一个自由的估计子语言（携带报告足迹和纯度位，记录在构建值时是否咨询了任何结果）与一个定价的主张子语言（携带财富变换器）配对，仅通过一个为比较定价的机制连接。通过读取结果选择的样本会设置纯度位，并被记录。

    arXiv:2608.27451v1 Announce Type: cross  Abstract: Empirical comparisons between systems are a standard form of evidence in computer science research, but few are checked for statistical validity: most are never framed as statistical tests at all. Existing multiple-comparison procedures could control the resulting error, but need inputs (what an analysis examined, and how its observations are arranged) that are not recoverable from a list of p-values.   We introduce Tacet, a language in which an analysis declares what it generated, states what it expects to find, and is refused any claim it cannot afford or cannot properly test. Its core calculus T pairs a free estimation sublanguage, carrying a reported footprint and a purity bit that records whether any outcome was consulted in building a value, with a priced claim sublanguage, carrying a wealth transformer, connected only by a mechanism that prices a comparison. A sample selected by reading outcomes sets the purity bit and is record
    
[^2]: SWE-Prime：更少的轨迹，更好的性能

    SWE-Prime: Fewer Trajectories, Better Performance

    [https://arxiv.org/abs/2608.27449](https://arxiv.org/abs/2608.27449)

    SWE-Prime通过两阶段多粒度筛选方法，仅保留高质量且具代表性的轨迹片段，以更少的数据实现更好的模型性能。

    

    为了提升大型语言模型解决现实世界软件问题的能力，先前的工作主要集中于构建大规模智能体轨迹数据集，并对成功轨迹进行监督微调（SFT）。然而，任务成功并不保证监督质量高：成功轨迹可能仍包含无效、冗余或危险的步骤。直接使用此类轨迹进行SFT可能会引入噪声监督，并鼓励模型模仿不良的问题解决行为。因此，我们提出SWE-Prime，一种多粒度、两阶段的SFT数据选择方法，逐步在轨迹和片段级别过滤训练数据。具体而言，第一阶段基于过程质量、结果质量和数据代表性进行轨迹级筛选，选择高质量且有代表性的成功轨迹子集。第二阶段进行片段级筛选。

    arXiv:2608.27449v1 Announce Type: cross  Abstract: To improve large language models' ability to resolve real-world software issues, prior work has focused on constructing large-scale agent trajectory datasets and performing supervised fine-tuning (SFT) on successful trajectories. However, task success does not guarantee high-quality supervision: successful trajectories may still contain ineffective, redundant, or risky steps. Directly using such trajectories for SFT can introduce noisy supervision and encourage models to imitate undesirable problem-solving behaviors. Therefore, we propose SWE-Prime, a multi-granularity, two-stage SFT data selection method that progressively filters training data at the trajectory and segment levels. Specifically, the first stage performs trajectory-level screening based on process quality, result quality, and data representativeness, selecting a high-quality and representative subset of successful trajectories. The second stage performs segment-level s
    
[^3]: 从静态到动态：基于MCR-Bench对真实世界代码评审的基准测试

    From Static to Dynamic: Benchmarking Real-World Code Review with MCR-Bench

    [https://arxiv.org/abs/2608.27442](https://arxiv.org/abs/2608.27442)

    本文提出了MCR-Bench，首个缺陷状态感知的多轮代码评审基准，包含2,269个真实任务和细粒度缺陷标注，以解决现有LLM评审方法过度简化静态化的问题。

    

    在现实世界的软件开发中，代码评审通常涉及开发者和评审者之间的迭代交互以提升软件质量，这使得该过程成本高昂且耗时。尽管近期研究探索了使用大型语言模型（LLMs）进行自动化代码评审，但大多数方法将代码评审过度简化为单轮、静态的决策任务，未能捕捉到多轮交互的特性以及真实评审场景中固有的复杂问题解决过程。为弥补这一差距，我们引入了MCR-Bench，这是首个面向现实多轮代码评审的缺陷状态感知基准。MCR-Bench涵盖五种常用编程语言，包含2,269个真实世界的多轮代码评审任务，每个任务都标注了细粒度的缺陷信息和跨轮状态标签。MCR-Bench中的每个任务都配备了细粒度的缺陷元数据（例如，描述、类型等）。

    arXiv:2608.27442v1 Announce Type: cross  Abstract: In real-world software development, code review typically involves iterative interactions between developers and reviewers to improve software quality, making the process costly and time-consuming. Although recent work explores large language models (LLMs) for automated code review, most approaches oversimplify code review into a single-round, static decision task, which fails to capture the multi-round interactive nature and the complex problem-solving processes inherent in realistic review scenarios. To bridge this gap, we introduce MCR-Bench, the first defect state-aware benchmark designed for realistic multi-round code review. MCR-Bench covers five commonly-used programming languages and consists of 2,269 real-world multi-round code review tasks, each of which is annotated with fine-grained defect information and cross-round state labels. Each task in MCR-Bench is equipped with fine-grained defect metadata (e.g., description, type,
    
[^4]: 人格-执行分离：一种在执行审计下演化LLM代理的架构模式

    Persona-Execution Separation: An Architecture Pattern for Evolving LLM Agents under Execution Audit

    [https://arxiv.org/abs/2608.27427](https://arxiv.org/abs/2608.27427)

    该论文提出人格-执行分离（PES）架构模式，将LLM代理的人格与执行分离到不同信任域，通过契约桥接实现自由演化与执行审计，并论证了在LLM表示不可区分性下单域机制需引入三类组件才能满足所有目标。

    

    在受治理组织中，大型语言模型（LLM）代理必须让人格（指令、语气、自我呈现）自由演化，同时保持执行（有状态、经审计的工作）的可追溯性。单一信任域无法低成本地同时满足这两个需求。我们提出了人格-执行分离（PES）模式：人格和执行位于不同的信任域，通过一个受治理的契约桥接。人格是单宿主的，可以漂移；执行是无面孔的，并受到审计。状态摘要可以返回；数据主体保留在限制性域中，除非有分级数据丢失防护（DLP）例外；身份保持连续性。一个审批矩阵、DLP和审计强制执行跨越。PES源于三个目标——自由漂移、执行可追溯性和解耦。在LLM表示不可区分性下，任何满足这三个目标的单域机制都必须重新引入类型化变更对象、外部门控和稳定的审计锚点。

    arXiv:2608.27427v1 Announce Type: cross  Abstract: Large language model (LLM) agents in governed organizations must let the persona (instructions, tone, self-presentation) evolve freely, while keeping execution (stateful, audited work) traceable. A single trust domain does not satisfy both cheaply. We present Persona-Execution Separation (PES): persona and execution reside in different trust domains, connected by a governed contract bridge. The persona is singly-homed and may drift; execution is faceless and audited. Status summaries may return; data bodies remain in the restrictive domain except a graded data-loss-prevention (DLP) exception; identity stays continuous. An approval matrix, DLP, and audit enforce the crossing. PES follows from three goals---free drift, execution traceability, and decoupling. Under LLM representational indistinguishability, any single-domain mechanism that meets all three must re-introduce typed change objects, an external gate, and a stable audit anchor:
    
[^5]: BTS-AgentBench：从只读遥测日志到代理基准测试的确定性、可重放流水线

    BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks

    [https://arxiv.org/abs/2608.27334](https://arxiv.org/abs/2608.27334)

    该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。

    

    arXiv:2608.27334v1 公告类型：新 摘要：工业现场包含大量只读遥测数据，但很少有基准测试规定如何将这些记录编译为可执行的多轮代理任务。我们提出了一种遥测到情节的构建方法，并将其实例化为BTS-AgentBench。该流水线将BTS元数据和原始历史记录规范化为只读工具存储，编译具有工具派生黄金答案和证据的静态任务，并将保留的任务提升为类型化、有界、面向操作员的情节。发布的532行版本增加了澄清、目标修订、时间戳策略、质量门控报告和证据归因，同时保留了源计算和分割。编码契约预检报告零发现，构建排除控制器完成0/532行。两个独立的原始到情节构建匹配所有11个逻辑工具存储导出，并精确复现发布的356/87/89训练/开发/测试工件。将共享构建路径应用于XAI4HEAT。

    arXiv:2608.27334v1 Announce Type: new  Abstract: Industrial sites contain large volumes of read-only telemetry, but few benchmarks specify how to compile these records into executable multi-turn agent tasks. We present a telemetry-to-episode construction method instantiated as BTS-AgentBench. The pipeline normalizes BTS metadata and raw histories into a read-only tool store, compiles static tasks with tool-derived gold answers and evidence, and lifts retained tasks into typed, bounded operator-facing episodes. The 532-row release adds clarification, goal revision, timestamp policy, quality-gated reporting, and evidence attribution while preserving the source computation and split. Coded contract preflight reports zero findings, and the construction-exclusion controller completes 0/532 rows. Two independent raw-to-episode builds match all 11 logical tool-store exports and reproduce the released 356/87/89 train/dev/test artifact exactly. Applying the shared construction path to XAI4HEAT 
    
[^6]: 当上下文获得权限：LLM工具框架中的权限提升

    When Context Gets Root: Privilege Escalation in LLM Harnesses

    [https://arxiv.org/abs/2608.27299](https://arxiv.org/abs/2608.27299)

    本文提出了一种名为“指令权限提升”的新型攻击，利用LLM代理工具框架在构建上下文时提升低级别指令的权限，从而绕过指令层级防御，实现对编码代理的广泛攻击。

    

    指令层级是一种模型侧防御机制，根据指令来源为指令分配不同级别的权限。这些级别限制了哪些内容可以指导模型行为。然而，在代理执行过程中，代理工具框架会为每次模型调用构建上下文。这种构建过程可能将低级别内容提升到更高级别的指令层级，并赋予其更大的模型可见权限。我们引入了指令权限提升攻击。在这种攻击中，攻击者诱导代理将低级别的恶意内容提升到更高级别的指令层级。提升后的内容随后导致代理执行在其原始级别下不会遵循的指令。我们通过使用多代理机制在六个编码代理工具框架上评估了这一威胁，实现了13个攻击目标。这些目标涵盖机密性、完整性、可用性和远程代码执行。在不受限制的操作执行下，攻击成功实现了这些目标。

    arXiv:2608.27299v1 Announce Type: cross  Abstract: Instruction hierarchy is a model-side defense that assigns instructions different levels of privilege according to their sources. These levels constrain which content may direct model behavior. During agent execution, however, agent harnesses construct context for each model invocation. This construction can elevate low-level content to a higher instruction level and grant it greater model-facing privilege. We introduce instruction privilege escalation. In this attack, an attacker induces an agent to elevate low-level malicious content to a higher instruction level. The elevated content then causes the agent to execute instructions it would not follow at their original level. We evaluate this threat by using multi-agent mechanisms to achieve 13 attack objectives across six coding-agent harnesses. These objectives span confidentiality, integrity, availability, and remote code execution. With unrestricted action execution, the attacks ac
    
[^7]: SPA：通过计划优先的信息流控制保护跨查询的持久化LLM代理

    SPA: Securing Persistent LLM Agents Across Queries with Plan-First Information-Flow Control

    [https://arxiv.org/abs/2608.27234](https://arxiv.org/abs/2608.27234)

    SPA通过计划优先架构和双格信息流控制，确保LLM代理在跨查询持久化状态下安全处理不受信任数据，防止攻击者篡改控制流或泄露敏感信息。

    

    大型语言模型（LLM）代理越来越多地在不受信任的网页、文档、工具和持久化状态上运行，同时行使对安全敏感资源的权限。现有防御通常保护规划或单个工具交互，但持久化代理面临更广泛的威胁：攻击者控制的数据可以改变控制流、进入安全敏感的工具参数，或危及后续查询。我们提出了SPA，一种计划优先的架构，用于保护规划、执行和跨查询状态重用。SPA每次查询调用一次规划器，以声明式领域特定语言生成完整的可执行计划，然后应用双格信息流控制来跟踪跨显式数据流和控制依赖的机密性和完整性。为了在不向规划器重新暴露不受信任负载的情况下支持持久化，SPA将执行结果存储为标记工件，并仅揭示语义信息。

    arXiv:2608.27234v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly operate over untrusted webpages, documents, tools, and persistent states while exercising authority over security-sensitive resources. Existing defenses typically protect either planning or individual tool interactions, but persistent agents face a broader threat: attacker-controlled data can alter control flow, enter security-sensitive tool arguments, or compromise later queries. We present SPA, a plan-first architecture that secures planning, execution, and cross-query state reuse. SPA invokes the planner once per query to generate a complete executable plan in a declarative domain-specific language, then applies dual-lattice information-flow control to track confidentiality and integrity across explicit data flows and control dependencies. To support persistence without re-exposing untrusted payloads to the planner, SPA stores execution results as labeled artifacts and reveals only sema
    
[^8]: 小型科研软件团队应对IT灾难的十二条快速建议

    Twelve Quick Tips for Managing IT Disasters in Small Research Software Teams

    [https://arxiv.org/abs/2608.27196](https://arxiv.org/abs/2608.27196)

    本文为小型科研软件团队提供了一份实用的灾难规划与恢复指南，强调在资源有限和非专业管理员情境下，通过简单有效的措施提升系统韧性，并充分利用机构内部支持资源。

    

    arXiv:2608.27196v1 公告类型：新 摘要：2025年，美国政府对其自身科研团队发起了一系列前所未有的攻击。一年后，GitHub的可用性首次跌破90%，而加拿大、法国、西班牙等地的野火迫使研究人员离开家园和实验室。这些事件及其他类似情况提醒我们，科研计算系统可能多么脆弱，而规划灾难应对是预防灾难最有效的方法之一。本文是一份针对小型科研软件团队的灾难规划与恢复的简短指南。这些建议假设你在常规工作之外独自处理所有事务，并且你不是经验丰富的系统管理员。有些建议确实需要此类专业知识，但大多数研究机构都设有研究计算组、数据图书馆员和环境健康与安全办公室，他们的全部工作正是帮助解决这些问题。本文告诉你该怎么做。

    arXiv:2608.27196v1 Announce Type: new  Abstract: In 2025, the US government launched an unprecedented series of attacks on its own scientific research groups. A year later GitHub dropped below 90% availability for the first time, while wildfires in Canada, France, Spain, and elsewhere forced researchers from the homes and labs. These events and others have reminded us just how fragile research computing systems can be, and that planning for disasters is one of the most effective ways to prevent them.   This paper is a short guide to disaster planning and recovery for a small research software team. The tips assume you are doing everything yourself on top of your regular job, and that you aren't an experienced system administrator. Some of the tips do require that kind of expertise, but most research institutions have research computing groups, data librarians, and environmental health-and-safety offices whose entire job is to help with exactly these problems. This paper tells you what 
    
[^9]: 一种基于单幕优化器学习的跨域数字孪生框架，用于牛育肥舍生物感知气候与能源控制

    A Trans-Domain Digital Twin for Bio-Aware Control of Climate and Energy in Cattle Fattening Barns Using Single-Episode Optimizer Learning

    [https://arxiv.org/abs/2608.27185](https://arxiv.org/abs/2608.27185)

    本文提出一种集成机制模拟、生长预测与轻量强化学习的跨域数字孪生框架，通过单幕学习实现牛育肥舍气候与能源的协同优化控制。

    

    在封闭式牛育肥舍中，室内气候与牛群生长相互依存。温度、相对湿度、气流和通风影响热舒适度、采食量、代谢产热、日增重、饲料效率和能耗，而体重增加则改变牛舍未来的热湿负荷，进而影响通风、供暖和能源需求。本文提出一种具有单幕学习能力的跨域数字孪生框架，专为封闭式牛育肥舍的生物感知气候与能源控制而设计。该框架整合了机制气候模拟器、牲畜生长模拟器、模型预测控制、轻量强化学习和结构化知识记忆，并采用多速率时间循环架构。快速时间循环每五分钟运行一次，用于评估执行器决策并维持短期热环境。

    arXiv:2608.27185v1 Announce Type: new  Abstract: In closed cattle-fattening barns, the indoor climate and herd growth are mutually interdependent. Temperature, relative humidity, airflow, and ventilation affect thermal comfort, feed intake, metabolic heat production, daily growth, feed efficiency, and energy consumption, while body-weight gain alters the future heat and moisture loads of the barn and, consequently, its ventilation, heating, and energy requirements. This article proposes a trans-domain digital twin framework with single-episode learning capability, customized for bio-aware climate and energy control in a closed cattle-fattening barn. The framework integrates a mechanistic climate simulator, a livestock growth simulator, model predictive control, lightweight reinforcement learning, and structured knowledge memory within a multi-rate temporal-loop architecture. The fast temporal loop operates every five minutes to evaluate actuator decisions and maintain short-term therma
    
[^10]: 大型语言模型在数字电子设计自动化中的应用：从生成到编排的角色转变视角

    LLMs in Digital EDA: A perspective on shifting roles from Generation to Orchestration

    [https://arxiv.org/abs/2608.27184](https://arxiv.org/abs/2608.27184)

    本文提出从生成到编排的三个层次化角色框架，揭示LLM在数字EDA中的能力积累机制，并指出当前模型存在生成“语法合理”而非“物理正确”硬件的陷阱。

    

    电子设计自动化（EDA）通过一代又一代的工具逐步自动化综合、优化和验证，推动了工程生产力的进步。大型语言模型（LLMs）通过实现从设计意图到硬件实现的直接转换，延伸了这一轨迹。在大多数EDA文献中，基于LLM的解决方案通常辅助孤立的设计阶段或任务，但这掩盖了能力涌现和系统扩展的驱动因素。在本视角中，我们定义了三个层次化角色，揭示能力如何积累：生成器（Generator）一次性生成设计工件，代理（Agent）通过迭代工具反馈优化输出，以及编排器（Orchestrator）跨EDA阶段协调决策。在已发表系统中，这揭示了一个“语法陷阱”，即模型被训练为生成看似合理的代码，而非物理上正确的硬件，这使问题复杂化。

    arXiv:2608.27184v1 Announce Type: cross  Abstract: Electronic design automation (EDA) has advanced engineering productivity through successive generations of tooling that progressively automate synthesis, optimisation, and verification. Large language models (LLMs) extend this trajectory by enabling direct translation from design intent to hardware implementations. In most of the EDA literature, LLM-based solutions are typically assisting siloed design stages or tasks, however this obscured the drivers by which capability emerges and systems scale. In this Perspective, we instead define three hierarchical roles that reveal how capability accumulates: a Generator that produces design artifacts in a single pass, an Agent that refines outputs through iterative tool feedback, and an Orchestrator that coordinates decisions across EDA-stages. Across published systems, this reveals a syntax trap in which models are trained to produce plausible code rather than physically correct hardware, com
    
[^11]: AgentDV：用于硬件设计验证的闭环智能体AI

    AgentDV: Closed-Loop Agentic AI for Hardware Design Verification

    [https://arxiv.org/abs/2608.27148](https://arxiv.org/abs/2608.27148)

    AgentDV是一个闭环智能体AI框架，通过可运行性过滤、CSR接地检查和覆盖率引导迭代，将LLM测试平台生成转化为可靠的RTL验证流水线。

    

    寄存器传输级（RTL）验证在现代片上系统（SoC）开发中占据了主要工作量。然而，近期基于LLM的验证代码生成往往无法产生可运行、设计一致且能产生覆盖率的测试平台。我们提出了AgentDV，一个用于自动化RTL验证环境生成的闭环智能体AI框架。AgentDV通过结合LLM引导的分析、测试平台构建、仿真、覆盖率测量和迭代优化，将单次LLM测试平台生成转化为一个基于工具的验证流水线。该框架引入了三个关键思想：1）可运行性过滤，拒绝无效的生成环境；2）基于CSR的检查，减少幻觉信号和错误预期行为；3）基于覆盖率的迭代，根据测量的验证差距重新生成测试。我们使用三个LLM在挑战性DUT和公开的OpenTitan外设上评估了AgentDV。

    arXiv:2608.27148v1 Announce Type: new  Abstract: Register-transfer level (RTL) verification consumes a major part of modern system-on-chip (SoC) development effort. Yet, recent LLM-based verification-code generation often fails to produce runnable, design-consistent, and coverage-producing testbenches. We present AgentDV, a closed-loop agentic AI framework for automated RTL verification environment generation. AgentDV transforms single-shot LLM testbench generation into a tool-grounded verification pipeline by combining LLM-guided analysis, testbench construction, simulation, coverage measurement, and iterative refinement. The framework introduces three key ideas: 1) runnability filtering to reject invalid generated environments, 2) CSR-grounded checking to reduce hallucinated signals and incorrect expected behavior, and 3) coverage-guided iteration to regenerate tests based on measured verification gaps. We evaluate AgentDV using three LLMs on challenge DUTs and public OpenTitan perip
    
[^12]: 当工具输出变成指令：在工具增强型LLM代理中分离动作诱导与运行时授权

    When Tool Outputs Become Commands: Separating Action Induction from Runtime Authorization in Tool-Augmented LLM Agents

    [https://arxiv.org/abs/2608.27146](https://arxiv.org/abs/2608.27146)

    本文提出SARA框架，通过分离动作诱导与执行授权，并引入动作来源追踪和上下文隔离探针，防止工具输出演变为指令后引发未授权的现实世界副作用。

    

    arXiv:2608.27146v1 公告类型：新 摘要：工具增强型LLM代理必须依赖不受信任的运行时观测来完成开放式任务；然而，当工具输出不再仅仅提供数据，而是开始指定具体动作时，它们实际上变成了“指令”，可能驱动超出用户意图的现实世界副作用。我们认为，这种风险源于将动作诱导与执行授权混为一谈。为解决这一区别，我们提出了SARA，它将动作诱导和执行授权视为不同的运行时角色，并分离动作来源与执行权限。在观测端，上下文隔离的动作探针暴露动作诱导语义，并跨步骤持续记录动作来源作为审查信号；在执行端，实际工具调用仅针对用户目标和来自授权成功执行的审计证据进行授权，同时满足目标、执行链和参数一致性要求。

    arXiv:2608.27146v1 Announce Type: new  Abstract: Tool-augmented LLM agents must rely on untrusted runtime Observations to complete open-ended tasks; however, when tool outputs no longer merely provide data but begin to specify concrete actions, they effectively become ``commands'' that can drive real-world side effects beyond user intent. We argue that this risk arises from conflating action induction with execution authorization. To address this distinction, we propose SARA, which treats action induction and execution authorization as distinct runtime roles and separates action provenance from execution authority. On the Observation side, a context-isolated Action Probe exposes action-inducing semantics and persistently records action-origin provenance across steps as a review signal; on the execution side, actual tool calls are authorized only against the user objective and audited evidence from authorized successful executions, while satisfying goal, execution-chain, and argument-le
    
[^13]: AROMA+：影响Maven生态系统中可复现构建因素的研究

    AROMA+: A Study of Factors Affecting Reproducible Builds in the Maven Ecosystem

    [https://arxiv.org/abs/2608.27125](https://arxiv.org/abs/2608.27125)

    本研究提出自动化方法，旨在解决Maven生态系统中可复现构建列表维护困难的问题，通过自动从Maven发布中定位源代码来提高可复现性验证的效率。

    

    arXiv:2608.27125v1 公告类型：新 摘要：现代软件工程建立了软件供应链，并依赖工具和库来提高生产力。然而，在项目中重用外部软件会带来安全风险，当组件的来源未知或组件的完整性无法验证时尤其如此。可复现构建提供了一种缓解策略，因为它们可以确认重用组件的来源和一致性。Debian社区已经形成了一个庞大的可复现性社区，但Maven生态系统（Java供应链的支柱）的可复现性相比之下研究不足。可复现中心是一个整理可复现Maven库列表的倡议，但由于人工努力，该列表有限且难以维护。我们的研究旨在通过自动化支持Maven生态系统中的这些努力。我们调查了从Maven发布中自动找到库源代码的可行性。

    arXiv:2608.27125v1 Announce Type: new  Abstract: Modern software engineering establishes software supply chains and relies on tools and libraries to improve productivity. However, reusing external software in a project presents a security risk when the source of the component is unknown or the consistency of a component cannot be verified. Reproducible builds present a mitigation strategy, as they can confirm the origin and consistency of reused components. A large reproducibility community has formed for Debian, but the reproducibility of the Maven ecosystem, the backbone of the Java supply chain, remains understudied in comparison. Reproducible Central is an initiative that curates a list of reproducible Maven libraries, but the list is limited and challenging to maintain due to manual efforts. Our research aims to support these efforts in the Maven ecosystem through automation. We investigate the feasibility of automatically finding the source code of a library from its Maven releas
    
[^14]: 机器学习研究软件中可复现性保障的变异测试：一项实证研究

    Mutation Testing for Reproducibility Safeguards in Machine Learning Research Software: An Empirical Study

    [https://arxiv.org/abs/2608.27100](https://arxiv.org/abs/2608.27100)

    本研究通过变异测试方法系统评估了机器学习研究仓库中验证工作流对可复现性相关变更的检测能力，发现现有验证流程对关键实验选择变化的检测存在显著不足。

    

    机器学习研究的可复现性依赖于实验选择，如随机种子、依赖版本、数据划分和评估配置。现有的仓库验证工作流可能在未检测到这些选择变化的情况下成功执行。我们使用MLReproMutate研究软件来研究这一问题，该软件对机器学习研究仓库应用受控的、与可复现性相关的变异，并针对仓库中已有的验证工作流进行评估。我们进行了一项结果盲态的实证研究，涉及39个冻结的仓库操作者案例，使用四种变异类别：随机种子、依赖固定、数据划分和交叉验证折数。在观察变异结果之前，仓库修订版、变异候选和验证工作流均已固定。初步执行产生了13/39个案例的结果；一个有界恢复程序将组合可评估集增加到2个。

    arXiv:2608.27100v1 Announce Type: new  Abstract: Reproducibility in machine-learning research depends on experimental choices such as random seeds, dependency versions, data partitioning, and evaluation configuration. Existing repository validation workflows may execute successfully without detecting changes to such choices. We study this problem using MLReproMutate, research software that applies controlled, reproducibility-relevant mutations to ML research repositories and evaluates them against validation workflows already present in those repositories.   We conducted an outcome-blind empirical study of 39 frozen repository-operator cases using four mutation classes: random seed, dependency pin, data split, and cross-validation fold count. Repository revisions, mutation candidates, and validation workflows were fixed before mutation outcomes were observed. Primary execution yielded outcomes for 13 of 39 cases; a bounded restoration procedure increased the combined evaluable set to 2
    
[^15]: 使用大型语言模型进行自动化基于模型的测试生成的经验评估

    An Empirical Evaluation of Using Large Language Models for Automated Model-Based Test Generation

    [https://arxiv.org/abs/2608.27094](https://arxiv.org/abs/2608.27094)

    本研究通过对比五种最新大型语言模型与GraphWalker工具，证明LLMs能显著优化和缩短基于模型的测试路径和步长，从而提升工业级MBT的可扩展性。

    

    arXiv:2608.27094v1 公告类型：新 摘要：大型语言模型在软件工程任务中展现出强大的潜力，特别是在软件测试方面。基于模型的测试（MBT）是一种软件测试技术。为了解决MBT在工业应用中的广泛可扩展性挑战，我们的论文对使用大型语言模型（LLMs）进行自动化基于模型的测试生成进行了经验评估，并与最先进的基于模型的测试工具（GraphWalker）及其内置算法（用于边和顶点覆盖设置的随机和快速随机）进行了比较。我们的评估表明，使用最新的五种最先进的LLMs（GPT-5.1、GPT-5.2、Claude Opus 4.5、Claude Sonnet 4.5和Gemini 2.5 Pro）针对四个GraphWalker模型（两个Web应用（Parabank和Testinium）和两个硬件应用（TLC和RISC-V））在复杂度逐步增加的情况下，具有优化和缩短测试路径及步长的强大潜力。

    arXiv:2608.27094v1 Announce Type: new  Abstract: Large language models have shown strong potential for software engineering tasks, particularly software testing. Model-based testing (MBT) is a software testing technique. To address the broad scalability challenge for industrial adoption of MBTs, our paper presents an empirical evaluation of Large Language Models (LLMs) for automated model-based test generation, compared with a state-of-the-art model-based testing tool (GraphWalker) and its built-in algorithms (random and quick random for edge and vertex coverage settings). Our evaluation indicates strong potential to optimize and shorten test paths and step sizes using the recent five state-of-the-art LLMs (GPT-5.1, GPT-5.2, Claude Opus 4.5, Claude Sonnet 4.5, and Gemini 2.5 Pro) against four GraphWalker models (two web applications (Parabank and Testinium) and two hardware applications (TLC and RISC-V) ) of escalating complexity.
    
[^16]: 从错误报告中定位缺陷：一种多目标方法

    Bug Localization from Bug Reports: A Multi-Objective Approach

    [https://arxiv.org/abs/2608.27089](https://arxiv.org/abs/2608.27089)

    提出了一种基于SPEA-2多目标进化优化的类级缺陷定位系统，在最大化相似性的同时最小化建议文件数，并在六个Java项目中验证了其有效性。

    

    缺陷定位是一项劳动密集型任务，尤其是在大型软件系统中。当异常行为发生时，开发者必须执行重复且耗时的步骤来识别有问题的文件。以往的研究主要集中于单目标定位方法，其中许多方法受限于特定编程语言。此外，仅依赖源代码与错误报告之间的词汇相似性往往不足，因为错误描述具有自然语言特性。在本研究中，我们提出了一种基于类的自动化多目标搜索系统，用于从错误报告中识别并排序可能存在缺陷的类。主要目标是最大化相似性，同时最小化建议的有问题文件数量。进化优化算法SPEA-2被应用于六个开源Java项目，包含超过22,000个错误报告。所提出的方法已与两种广泛使用的方法进行了评估对比。

    arXiv:2608.27089v1 Announce Type: cross  Abstract: Bug localization is a labor-intensive task, particularly in large software systems. When abnormal behavior occurs, developers must perform repetitive and time-consuming steps to identify faulty files. Previous studies have mainly focused on single-objective localization methods, many of which are limited to specific programming languages. In addition, relying solely on lexical similarity between source code and bug reports is often insufficient due to the natural language nature of bug descriptions. In this study, we propose a class-level automated multi-objective search-based system to identify and rank potentially buggy classes from bug reports. The main objective is to maximize similarity while minimizing the number of suggested faulty files. The evolutionary optimization algorithm SPEA-2 was applied to six open-source Java projects comprising more than 22,000 bug reports. The proposed approach was evaluated against two widely used 
    
[^17]: 一种以契约为中心的可扩展且可管理的代理运行时架构

    A Contract-Centered Architecture for Scalable and Manageable Agentic Runtimes

    [https://arxiv.org/abs/2608.27086](https://arxiv.org/abs/2608.27086)

    本文提出一种以四个责任对象（技能、框架、脚手架和数据基底）为组织契约的代理运行时架构，核心假设P1强调在成本约束下实现能力与容量的可分离性，以解决企业AI部署中的跨部门协调问题。

    

    企业AI部署是一个跨业务单元、应用和AI团队、测试、平台工程、基础设施、安全、运营和数据治理的协调问题。用例基准测试表明一个代理是否能完成一项任务，但并未说明如何共同拥有、变更、接纳或验证能力、模型、运行时机制、容量和企业数据的变化。我们提出了四个责任对象作为共享的组织契约：技能（可重用、版本化的能力和工作流资产）、框架（运行时编译器和治理器）、脚手架（执行/控制边界和非功能需求所有者），以及一个在独立首席信息官治理语义和遥测下的栈外数据基底。运行时核心为A=，数据基底位于该栈之外。核心贡献是一个有界、可证伪的假设，P1（成本感知的能力-容量可分离性）：在声明的工作区域内，c

    arXiv:2608.27086v1 Announce Type: new  Abstract: Enterprise AI deployment is a coordination problem across business units, application and AI teams, testing, platform engineering, infrastructure, security, operations, and data governance. Use-case benchmarks show whether one agent completes one task, but not how changing capabilities, models, runtime mechanisms, capacity, and enterprise data should be owned, changed, admitted, or evidenced together.   We present four responsibility objects as shared organizational contracts: Skill (reusable, versioned capability and workflow asset), Harness (runtime compiler and governor), Scaffold (execution/control boundary and NFR owner), and a stack-external data substrate under independent CIO-governed semantics and telemetry. The runtime core is A = , with the data substrate outside that stack.   The central contribution is one bounded, falsifiable hypothesis, P1 (cost-aware capability-capacity separability): within a declared operating region, c
    
[^18]: 用户认证模式目录

    A Catalog of User Authentication Patterns

    [https://arxiv.org/abs/2608.26955](https://arxiv.org/abs/2608.26955)

    本文提出了一个包含14种用户认证模式的新目录，按认证因素和实际角色分类，以弥补安全模式目录中认证模式缺失的空白。

    

    arXiv:2608.26955v1 公告类型：交叉 摘要：安全模式旨在支持安全软件系统的设计与开发。然而，尽管存在已建立的安全模式目录，它们的实际应用仍然有限。特别是，尽管有这些目录，针对常见安全控制（如用户认证，简称认证）的具体模式仍然缺乏。本文旨在通过认证为例，为弥合这一差距做出初步贡献。它提出了一个新的认证模式目录，包含14种用户认证模式。为支持该目录的实际应用，它根据众所周知的认证因素概念以及每种模式在实践中通常扮演的角色对模式进行分类。通过认证模式对常见认证技术进行编目，我们旨在为支持软件工程师和架构师设计和开发安全系统做出重要贡献。

    arXiv:2608.26955v1 Announce Type: cross  Abstract: Security patterns are intended to support the design and development of secure software systems. However, although established catalogs of security patterns exist, their practical application remains limited. In particular, despite these catalogs, concrete patterns for common security controls such as user authentication (authentication for short) are lacking. This paper aims to make an initial contribution toward closing this gap, as exemplified by authentication. It presents a novel authentication pattern catalog, comprising 14 user authentication patterns. To support the catalog's practical application, it classifies patterns by the well-known concept of authentication factors and by the usual role each pattern fulfills in practice. By cataloging common authentication techniques through authentication patterns, we aim to make an important contribution to supporting software engineers and architects in designing and developing secure
    
[^19]: 在概念复杂的范围综述中评估人类与LLM筛选工作流：召回率-工作量权衡及运行间一致性

    Evaluating human and LLM screening workflows in a conceptually complex scoping review: Recall--workload trade-offs and run-to-run consistency

    [https://arxiv.org/abs/2608.26885](https://arxiv.org/abs/2608.26885)

    该研究通过预注册实验比较了人类与LLM在概念复杂范围综述中的筛选工作流，发现无工作流能完全恢复所有合格记录，并揭示了召回率-工作量权衡及运行间一致性问题。

    

    背景：大型语言模型（LLMs）越来越多地被用于证据综合中的筛选，而假阴性可能在全文评估前移除相关研究。我们在一个概念复杂的范围综述中嵌入了一项预注册研究，比较了人类和LLM的标题与摘要筛选工作流。方法：在保守的仅标题筛选后，1131条记录由一位综述负责人、四位训练有素的助理（筛选不重叠子集）以及七次完整的LLM运行（使用不同模型和处理配置，包括一次名义上相同的重复运行）进行筛选。我们比较了保留的工作量、对316条已验证合格记录的操作召回率、一致性、运行间一致性及程序负担。由于资格仅在父综述中推进和评估的记录中验证，召回率估计是操作性的。结果：没有工作流能恢复所有已验证的合格记录。人类...

    arXiv:2608.26885v1 Announce Type: new  Abstract: Background. Large language models (LLMs) are increasingly used for screening in evidence synthesis, where false negatives can remove relevant studies before full-text assessment. We compared human and LLM title-and-abstract screening workflows in a preregistered study embedded in a conceptually complex scoping review.   Methods. After a conservative title-only screen, 1,131 records were screened by one review lead, four trained assistants screening non-overlapping subsets, and seven complete LLM runs using different models and processing configurations, including a nominally identical repeat run. We compared retained workload, operational recall against 316 verified eligible records, agreement, run-to-run consistency, and procedural burden. Because eligibility was verified only for records advanced and assessed in the parent review, recall estimates were operational.   Results. No workflow recovered all verified eligible records. The hum
    
[^20]: 超越执行：审计LLM驱动科学研究中的实验保真度

    Beyond Execution: Auditing Experimental Fidelity in LLM-Driven Scientific Research

    [https://arxiv.org/abs/2608.26753](https://arxiv.org/abs/2608.26753)

    本文提出ABE-Ralph框架，通过结构化约束和8步工作流审计LLM科学实验的保真度，检测方法论幻觉（如数据缩减、组件替换），确保代理忠实复现参考方法并提供可靠证据。

    

    arXiv:2608.26753v1 公告类型：交叉 摘要：用于科学实验的LLM代理必须做的不仅仅是生成可执行的代码：它们必须忠实地实现参考方法，设计能验证论文主张的实验，并提供支持这些主张的证据。我们表明，代理经常产生方法论幻觉：悄悄减少数据集或训练预算，用查找表或神谕函数替换失败的学习或生成组件，或在方法声称的优势消失的资源受限环境中得出结论。为了检测这些失败，我们引入了ABE-Ralph，一个参考锚定的审计框架，它将主张、协议、所需组件、基线和指标表示为结构化的实验约束，通过8步工作流指导实现，并进行定量、定性和代码级验证。在覆盖12个机器学习领域的30次长时程复现运行中，ABE-R

    arXiv:2608.26753v1 Announce Type: cross  Abstract: LLM agents used for scientific experimentation must do more than generate executable code: they must implement the reference method faithfully, design experiments that test the paper's claims, and provide evidence supporting those claims. We show that agents often produce methodological hallucinations: silently reducing datasets or training budgets, replacing failed learning or generative components with lookup or oracle functions, or drawing conclusions from resource-limited settings where a method's claimed advantage disappears. To detect these failures, we introduce ABE-Ralph, a reference-anchored auditing framework that represents claims, protocols, required components, baselines, and metrics as structured experimental constraints, guides implementation through an 8-step workflow, and performs quantitative, qualitative, and code-level verification. Across 30 long-horizon reproduction runs covering 12 machine learning domains, ABE-R
    
[^21]: FaultLens：为生成的操作程序学习紧凑的行为测试套件

    FaultLens: Learning Compact Behavioral Test Suites for Generated Operational Programs

    [https://arxiv.org/abs/2608.26746](https://arxiv.org/abs/2608.26746)

    本文提出FaultLens方法，通过结合故障驱动的贪婪选择和突变无关的多样性组件，学习紧凑的行为测试套件，以高效检测生成程序中的稀疏边界和交互故障。

    

    arXiv:2608.26746v1 公告类型：交叉 摘要：生成的操作程序通常通过少量手写示例或全面的回归测试套件进行验证。前者可能遗漏稀疏的边界和交互故障，而后者可能不必要地昂贵。我们引入了FaultLens，一种学习紧凑行为测试套件的方法，同时保持与执行证据的可审计联系。它执行一次丰富的探针域，将故障-探针杀死关系存储为稀疏结果缓存，并仅从早期程序生成中学习探针排序。一个故障驱动的贪婪组件利用已知的杀死结构，而一个与突变无关的多样性组件覆盖探针族、案例、模板和时间箱。它们的交替混合方法在新程序包含排序构建中不存在的故障机制时仍然有用。我们评估了四个环境中的二十个生成操作策略，十个执行种子，以及1,200个测量值。

    arXiv:2608.26746v1 Announce Type: cross  Abstract: Generated operational programs are often validated with either a few hand-written examples or exhaustive regression suites. The former can miss sparse boundary and interaction faults, while the latter can be unnecessarily expensive. We introduce FaultLens, a method for learning compact behavioral test suites while preserving an auditable connection to executed evidence. It executes a rich probe domain once, stores the fault-probe kill relation as a sparse outcome cache, and learns probe orderings only from earlier program generations. A fault-driven greedy component exploits known kill structure, while a mutation-independent diversity component covers probe families, cases, templates, and temporal bins. Their alternating hybrid remains useful when a new program contains a fault mechanism absent from ordering construction.   We evaluate twenty generated operational policies across four environments, ten execution seeds, 1,200 measured r
    
[^22]: Claude Code 完整用户手册

    Claude Code Complete User Handbook

    [https://arxiv.org/abs/2608.26742](https://arxiv.org/abs/2608.26742)

    本书提出，在Claude Code代理环境中，安全高效运行的关键在于定义明确的可观察完成条件，并通过指令、权限、沙箱和系统隔离四层控制栈来管理系统性风险。

    

    arXiv:2608.26742v1 公告类型：交叉 摘要：Claude Code 是一个代理式工作环境：一种语言模型在循环中运行，具备文件系统访问、Shell 执行、浏览器控制、定时和云执行、通过模型上下文协议连接外部工具以及多代理编排能力。其能力范围现已超出单个从业者仅靠注意力所能监督的限度，其故障模式是系统性的而非局部性的：未审查的钩子、过度扩展的连接器、过时的完成条件、继承账户所有凭据的自主例程。本书是一本面向操作该系统的任务导向参考手册，旨在安全高效地运行该系统，专为对结果负责的从业者编写。书中提出四个论点：第一，没有明确且可观察的完成条件的能力并非生产力；第二，指令、权限执行、沙箱和操作系统隔离是控制栈的四个不同层次。

    arXiv:2608.26742v1 Announce Type: cross  Abstract: Claude Code is an agentic work environment: a language model operating in a loop with filesystem access, shell execution, browser control, scheduled and cloud execution, external tool connections through the Model Context Protocol, and multi-agent orchestration. Its capability envelope now exceeds what one practitioner can supervise by attention alone, and its failure modes are systemic rather than local: an unreviewed hook, an over-scoped connector, a stale completion condition, an autonomous routine inheriting every credential on an account. This book is a task-oriented reference for operating that system safely and productively, written for practitioners accountable for the result. It advances four propositions. First, capability without a defined and observable completion condition is not productivity. Second, instruction, permission enforcement, sandboxing and operating-system isolation are four distinct layers of a control stack,
    
[^23]: KubeCap：一种通过静态分析和LLM辅助规则推理实现Kubernetes能力最小化的框架

    KubeCap: A Framework for Capability Minimization in Kubernetes via Static Analysis and LLM-Assisted Rule Inference

    [https://arxiv.org/abs/2608.26699](https://arxiv.org/abs/2608.26699)

    KubeCap通过静态分析和LLM辅助规则推理，自动推断并最小化Kubernetes容器所需的能力，解决了开发人员因依赖默认配置而违反最小权限原则的问题。

    

    摘要：作为最广泛使用的容器编排平台，Kubernetes通过允许开发人员使用清单文件管理Linux能力，提供了灵活的权限配置。然而，开发人员在实践中依赖默认设置或粗粒度的安全上下文，这违反了最小权限原则，并扩大了容器化工作负载的攻击面。现有研究要么检测Kubernetes清单中的脆弱模式，要么推断独立Linux程序所需的能力，但并未直接解决Kubernetes中的能力最小化问题。为填补这一空白，我们首先对三个开源数据集进行了实证研究，发现74.67%的项目缺乏能力配置。受此观察启发，我们提出了KubeCap，一个用于Kubernetes能力最小化的框架。KubeCap将部署规范转换为确定性清单，并定位容器入口点。

    arXiv:2608.26699v1 Announce Type: cross  Abstract: As the most widely used container orchestration platform, Kubernetes provides flexible privilege configuration by allowing developers to manage Linux capabilities via manifest files. However, developers rely on default settings or coarse-grained security contexts in practice, violating the principle of least privilege and enlarging the attack surface of containerized workloads. Existing studies either detect vulnerable patterns in Kubernetes manifests or infer required capabilities for standalone Linux programs, but they do not directly address capability minimization in Kubernetes.   To bridge this gap, we first conduct an empirical study on three open-source datasets, revealing that 74.67% of projects lack capability configurations. Motivated by our observations, we propose KubeCap, a framework for Kubernetes capability minimization. KubeCap translates deployment specifications into deterministic manifests, locates container entrypoi
    
[^24]: 自治AI代理运行时治理的五种原语

    Five Primitives for Governing Autonomous AI Agents at Runtime

    [https://arxiv.org/abs/2608.26696](https://arxiv.org/abs/2608.26696)

    本文提出治理自治AI代理需在运行时解决，并定义了五种关键原语（发现、身份、治理、证明、供应链），以确保代理行动在生效前后得到有效管控。

    

    arXiv:2608.26696v1 公告类型：新 摘要：企业部署自治AI代理继承了为人类用户和长期服务设计的控制模型，这种契合在三个方面失效：代理主体是短暂的，出现和消失的速度快于配置；其行为由模型选择而非编程决定，因此它们可能尝试的操作集无法预先知晓；群体是发现的而非配置的，因为任何能调用API的人都能创建代理。我们认为治理这类代理是一个运行时问题——不是模型对齐问题，也不是构建时问题——我们从行动生效前和生效后必须回答的问题中推导出五种原语：发现、身份、治理、证明和供应链。对于每种原语，我们说明了缺失时会导致什么失败，以及为什么其他原语无法在结构上提供它。我们描述了一个实现，其中代理的行动被中介化以对抗...

    arXiv:2608.26696v1 Announce Type: new  Abstract: Enterprise deployments of autonomous AI agents inherit a control model built for human users and long-lived services, and the fit fails in three specific ways: agent principals are ephemeral, appearing and vanishing faster than provisioning; their actions are selected by a model rather than programmed, so the set of things they may attempt is not known in advance; and the population is discovered rather than provisioned, because anyone who can call an API can create one. We argue that governing such agents is a runtime problem -- not a model-alignment problem and not a build-time problem -- and we derive five primitives from the questions that must be answered before an action takes effect and after it has: discovery, identity, governance, attestation, and supply chain. For each we state what fails if it is absent and why the others cannot structurally supply it. We describe an implementation in which an agent's action is mediated agains
    
[^25]: 通过实践与学习定义的Processing/p5

    Processing/p5 Defined through Practice and Learning

    [https://arxiv.org/abs/2608.26614](https://arxiv.org/abs/2608.26614)

    本文通过跨语言案例研究，提出了一套构成Processing/p5核心的软件决策指导框架，强调了其在创意编码中的统一设计原则和跨平台适用性。

    

    摘要：arXiv:2608.26614v1 公告类型：新 摘要：不同编程语言中的Processing/p5库，在创意编码方面贯彻了一致的设计优先级，作为一种设计体验。尽管不同的编程语言生态系统，如Java和JavaScript，各自具有其特有的功能、社区规范和用法模式，但跨这些语言的Processing/p5草图却展现出相似性。基于在两种宿主语言JavaScript和Lua中构建Processing/p5实现的案例研究，我们提出了一系列指导软件决策的方面，这些方面构成了Processing/p5的核心，而不论宿主语言为何。我们结合其他探索性和创意工具中的决策讨论了这一框架，展示了每个指导方面如何以不同于案例研究的方式被具体操作。所提出的列表突出了通过学习、研究和艺术实践创建新的Processing/p5库的机会，以促进创意表达。

    arXiv:2608.26614v1 Announce Type: new  Abstract: Processing/p5 libraries across different programming languages enact consistent priorities for creative coding as a designed experience. While different programming language ecosystems, like Java and JavaScript, are each associated with their own affordances, community norms, and patterns of use, Processing/p5 sketches across these languages share similarities. Based on case studies of building an implementation of Processing/p5 in two host languages, JavaScript and Lua, we propose a list of software decision-making guiding aspects that constitute Processing/p5, regardless of host language. We discuss this framework in the context of decisions in other exploratory and creative tools that demonstrate how each of the guiding aspects can be operationalized differently than in the case studies. The proposed list highlights opportunities for learning, research, and artistic practice through creation of new Processing/p5 libraries for creative
    
[^26]: 千图假说：仓库级代码推理中任务条件化关系具体化的可检验假说

    The Thousand-Graph Hypothesis: A Testable Hypothesis of Task-Conditioned Relation Materialization in Repository-Level Code Reasoning

    [https://arxiv.org/abs/2608.26602](https://arxiv.org/abs/2608.26602)

    本文提出一种仅含实体的外部接口，通过推理时的任务条件化关系具体化，在无需预构建关系图的情况下，显著提升了仓库级代码推理的成功率。

    

    大型软件仓库通常超出模型上下文限制。将仓库知识训练进模型成本高昂且容易过时，而局部检索可能遗漏分散的需求，显式关系图则增加了持续维护负担。我们提出了一种仅含实体的外部接口，在推理过程中进行任务条件化关系具体化。一个两层索引将全局路由与局部实体聚焦分离，并在DeepSeek-V4-Flash和SWE-bench Verified上进行了评估。在零预构建实体-关系边的情况下，基础、单层和两层条件分别达到92.1%、94.2%和95.6%的成功率。

    arXiv:2608.26602v1 Announce Type: cross  Abstract: Large software repositories are often beyond model context limits. Training repository knowledge into models is costly and quickly stale, while local retrieval can miss scattered requirements, and explicit relation graphs add ongoing maintenance burden. We propose an entity-only external interface with task-conditioned relation materialization during inference. A two-layer index separates global routing from local entity focus and is evaluated on DeepSeek-V4-Flash and SWE-bench Verified. The base, one-layer, and two-layer conditions achieve 92.1%, 94.2%, and 95.6% success, respectively, under zero pre-built entity-relation edges.
    
[^27]: 未言明，即不安全？基于LLM的RTL代码生成中的隐含安全义务

    Unsaid, Unsafe? Implicit Security Obligations in LLM-Based RTL Code Generation

    [https://arxiv.org/abs/2608.26588](https://arxiv.org/abs/2608.26588)

    本研究揭示了LLM生成的RTL代码在功能正确性与安全性之间的巨大鸿沟，并构建了SECRTL-GEN基准来量化这一差距，表明当前前沿模型在隐含安全义务上表现严重不足。

    

    大型语言模型（LLMs）生成的寄存器传输级（RTL）代码在功能正确性上快速提升。然而，LLM生成代码的安全性主要针对软件进行研究，软件中的缺陷在部署后仍可修补。但一旦RTL代码被刻录到硅片上，不安全的RTL则没有这样的补救措施。我们构建了SECRTL-GEN，一个基于真实SoC IP的多语言资源访问安全基准：涵盖五个CWE家族和四种硬件描述语言（Verilog、SystemVerilog、VHDL和Python）的392个任务，每个任务都配有黑盒功能和安全测试台。功能规格说明故意省略了安全义务，以匹配实践中义务常被排除在功能文档之外的情况。对五个前沿LLM的实证研究显示出显著差距：在原始提示下，它们通过功能测试的比例约为73-79%，但通过安全测试的比例仅为14-35%，且功能更强的模型并不更安全。加入CWE知识后，这一比例有所提高。

    arXiv:2608.26588v1 Announce Type: cross  Abstract: Large Language Models (LLMs) generate register-transfer-level (RTL) code with rapidly improving functional correctness. Security of LLM-generated code, however, has been studied mainly for software, where flaws can still be patched after deployment. Insecure RTL offers no such remedy once taped out into silicon. We construct SECRTL-GEN, a multi-language resource-access security benchmark grounded in real SoC IP: 392 tasks over five CWE families and four HDLs (Verilog, SystemVerilog, VHDL, and Python), each with black-box functional and security testbenches. Functional specifications intentionally omit security obligations, matching how obligations are often kept out of functional docs in practice. An empirical study of five frontier LLMs shows a sharp gap: under vanilla prompts they pass functional tests in about 73-79% of cases but security tests in only 14-35%, and stronger functional models are not safer. Adding CWE knowledge raises
    
[^28]: DeepRepro：面向演化仓库中论文到代码复现的状态感知子规划

    DeepRepro: State-Aware Subplanning for Paper-to-Code Reproduction in Evolving Repositories

    [https://arxiv.org/abs/2608.26557](https://arxiv.org/abs/2608.26557)

    DeepRepro通过动态状态感知子规划，将仓库演化状态与运行时反馈实时转化为细粒度实现计划，解决了论文到代码复现中静态规划导致的执行不一致问题。

    

    arXiv:2608.26557v1 公告类型：新 摘要：近期代理式大型语言模型（LLMs）的进展使得软件工程工作流日益自主化，但自动机器学习（ML）论文到代码复现仍是一个具有挑战性的长周期问题。与传统代码生成不同，该任务需要构建并维护一个功能完整的仓库，其状态在执行过程中持续演化。现有系统通常依赖静态的前期规划，随后进行顺序的文件级生成，这往往导致依赖关系、接口和执行反馈随时间变化时出现不一致。我们提出了DeepRepro，一种基于执行状态感知子规划的论文到代码复现框架。DeepRepro动态地将演化中的仓库状态和运行时反馈转化为细粒度的实现子规划，使规划在整个仓库构建过程中与执行保持一致。该框架进一步增强了鲁棒性和适应性。

    arXiv:2608.26557v1 Announce Type: new  Abstract: Recent advances in agentic large language models (LLMs) have enabled increasingly autonomous software engineering workflows, yet automatic machine learning (ML) paper-to-code reproduction remains a challenging long-horizon problem. Unlike conventional code generation, this task requires constructing and maintaining a fully functional repository whose state continuously evolves during execution. Existing systems typically rely on static upfront planning followed by sequential file-level generation, which often leads to inconsistencies as dependencies, interfaces, and execution feedback change over time. We propose DeepRepro, a state-aware framework for paper-to-code reproduction based on execution-state-aware subplanning. DeepRepro dynamically transforms evolving repository states and runtime feedback into fine-grained implementation subplans, keeping planning aligned with execution throughout repository construction. The framework furthe
    
[^29]: 2026年下一代科学计算生态系统研讨会报告：利用社区、软件和人工智能促进跨学科团队科学

    Report of the 2026 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science

    [https://arxiv.org/abs/2608.26519](https://arxiv.org/abs/2608.26519)

    本报告基于2026年研讨会，提炼出科学计算生态系统未来发展的四大战略主题和八项社区行动优先事项，强调通过社会技术协同设计整合人工智能、软件和跨学科合作。

    

    科学计算正在经历快速转型，人工智能、异构计算、自动化和数据密集型研究的进步不仅重塑了计算工具，还重塑了支持科学发现的机构、劳动力模式和协作实践。本报告综合了2026年下一代科学计算生态系统研讨会的见解，该研讨会是聚焦于通过社会技术协同设计加强科学计算生态系统的三年系列会议中的第二届。研讨会讨论确定了四个相互依存的战略主题：面向人工智能驱动的科学发现的软件生态系统；信任、验证和可追溯性；人机协作与范式转变；以及劳动力、教学和治理。报告将这些主题转化为八项社区行动优先事项，涵盖共享研究基础设施、信任与可追溯性、用户体验、人机协作等方面。

    arXiv:2608.26519v1 Announce Type: cross  Abstract: Scientific computing is undergoing rapid transformation as advances in artificial intelligence, heterogeneous computing, automation, and data-intensive research reshape not only computational tools but also the institutions, workforce models, and collaborative practices that support scientific discovery. This report synthesizes insights from the 2026 Workshop on Next-Generation Ecosystems for Scientific Computing, the second in a three-year series focused on strengthening scientific computing ecosystems through socio-technical co-design. Workshop discussions identified four interdependent strategic themes: software ecosystems for AI-enabled scientific discovery; trust, validation, and traceability; human-AI teaming and paradigm shifts; and workforce, pedagogy, and governance. The report translates these themes into eight priorities for community action spanning shared research infrastructure, trust and traceability, user experience, hu
    
[^30]: 基于账本控制的零样本自编排提升LLM编码性能

    Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance

    [https://arxiv.org/abs/2608.26480](https://arxiv.org/abs/2608.26480)

    本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。

    

    多智能体大语言模型系统被广泛报道能超越单模型基线，但证据不一，且比较通常存在混淆：流程同时改变令牌预算、工具调用和提示，因此总体增益很少能揭示真正有效的因素。我们研究了在共享文件系统工作区中引入管理器-工作器脚手架的效果，无需训练且无需针对基准进行调优，与同一模型单次回答进行对比。在九个模型上——五个开放权重模型，参数范围从9B到约2.8T，以及四个前沿封闭模型——针对LiveCodeBench最新的100个困难问题，脚手架的好处是真实但有条件的：对某些模型效果显著且统计显著（如Qwen3.8-27B提升23.4，GPT-5.6-Luna提升10.6，GPT-5.6-Terra提升8.0，各基于五次配对运行；Kimi-K3提升30.4，Minimax-M3提升11.0，基于五次配对运行且关闭推理，p值均小于10^-4，以及...）

    arXiv:2608.26480v1 Announce Type: cross  Abstract: Multi-agent large language model systems are widely reported to beat single-model baselines, but the evidence is mixed, and comparisons are usually confounded: pipelines change token budgets, tool calls, and prompts simultaneously, so an aggregate gain rarely reveals what actually helped. We investigate the effect of introducing the manager-worker scaffold over a shared filesystem workspace, with no training and no per-benchmark tuning, measured against the same model answering in a single pass. Across nine models -- five open-weight, spanning 9B to ~2.8T parameters, and four frontier closed models -- on the 100 latest hard LiveCodeBench problems, the scaffold's benefit is real but conditional: large and statistically significant for some (Qwen3.8-27B +23.4, GPT-5.6-Luna +10.6 and GPT-5.6-Terra +8.0, each over five paired passes; Kimi-K3 +30.4 and Minimax-M3 +11.0 over five paired passes with reasoning off, both at $p < 10^{-4}$, and +
    
[^31]: STILL：为LLM辅助的C++反编译恢复被剥离的STL语义

    STILL: Recovering Lowered STL Semantics for LLM-assisted C++ Decompilation

    [https://arxiv.org/abs/2608.26408](https://arxiv.org/abs/2608.26408)

    本文提出STILL，一种通过预测剥离二进制中STL容器语义并生成提示，显著提升LLM辅助C++反编译的可执行性（从17.4%提升至28.4%）。

    

    arXiv:2608.26408v1 公告类型：新 摘要：LLM辅助反编译提高了可读性和可重新执行性，但在使用标准模板库（STL）的剥离C++函数上仍表现不佳。编译、优化和符号剥离会移除或模糊源级语义，如容器类型和库调用结构，而传统反编译器的输出往往无法恢复这些语义。我们提出了STILL，一种结构化语义接口，它从剥离的控制流图中预测函数级的STL容器语义，并将其呈现为紧凑的提示供LLM优化。在StlBench上，STILL预测了常见的容器级STL语义，在稳定的字符串和向量切片上取得了最强的跨数据集结果。在剥离的HumanEval反编译中，这些提示使DeepSeek-chat优化达到了28.4%的可执行性，相比之下，无提示优化为17.4%，原始Ghidra反编译为8.9%；提示效用依赖于下游主干，其中...

    arXiv:2608.26408v1 Announce Type: new  Abstract: LLM-assisted decompilation improves readability and re-executability, but still underperforms on stripped C++ functions that use the Standard Template Library (STL). Compilation, optimization, and symbol stripping remove or obscure source-level semantics such as container types and library-call structure, while traditional decompiler output often fails to recover them. We present STILL, a structured semantic interface that predicts function-level STL container semantics from stripped control-flow graphs and renders them as compact hints for LLM refinement. On StlBench, STILL predicts common container-level STL semantics, with the strongest cross-dataset results for stable string and vector slices. On stripped HumanEval decompilation, these hints enable DeepSeek-chat refinement to reach 28.4% executability, compared with 17.4% for no-hint refinement and 8.9% for raw Ghidra decompilation; hint utility is downstream-backbone-dependent, with
    
[^32]: Spec2Vision：契约引导的AI生成计算机视觉流水线交付

    Spec2Vision: Contract-Guided Delivery of AI-Generated Computer Vision Pipelines

    [https://arxiv.org/abs/2608.26400](https://arxiv.org/abs/2608.26400)

    Spec2Vision通过分阶段运行时和明确任务契约，显著提升了AI生成计算机视觉流水线的可交付性，在850次运行中达到81/85测试通过率，远超基线方法。

    

    arXiv:2608.26400v1 公告类型：新  摘要：生成的计算机视觉代码可能可运行，但不满足下游评估器强制的任务契约。我们通过Spec2Vision研究这一差距，这是一个实验框架，用于生成和评估基于规范的CV流水线捆绑包，通过分阶段运行时在合成、筛选、测试和有限修复过程中保持任务契约明确。该基准评估了17个CV任务、10个可执行条件和每个任务-条件单元的5次重复，共850次主要运行。在主要850次运行评估中，Spec2Vision达到81/85次评估器测试通过；移除结构修复降至55/85，兼容性脚手架降至58/85，生成器预检降至39/85。可执行的单代理基线向模型暴露逐步丰富的任务规范，最终达到直接源规范暴露，但整体上仍然弱得多，从轻量任务接地的17/85到评估器测试通过的35/85。

    arXiv:2608.26400v1 Announce Type: new  Abstract: Generated computer-vision code can be runnable without satisfying the task contract enforced by a downstream evaluator. We study that gap with Spec2Vision, an experimental framework for producing and evaluating specification-grounded CV pipeline bundles through a staged runtime that keeps the task contract explicit across synthesis, screening, testing, and bounded repair. The benchmark evaluates 17 CV tasks, 10 executable conditions, and 5 repeats per task-condition cell, for 850 primary runs. In the primary 850-run evaluation, Spec2Vision reaches 81/85 evaluator-test passes; removing structural repair drops to 55/85, compatibility scaffolding to 58/85, and generator preflight to 39/85. The executable single-agent baselines expose progressively richer task specifications to the model, culminating in direct source-spec exposure, yet remain much weaker overall, from 17/85 for lightweight task grounding to 35/85 evaluator-test passes. The l
    
[^33]: 探究LLM生成的软件系统在不同生成与执行环境中的软件老化现象

    Investigating Software Aging in LLM-Generated Software Systems across Generation-and-Execution Environments

    [https://arxiv.org/abs/2608.26391](https://arxiv.org/abs/2608.26391)

    本研究首次通过实验揭示了LLM生成的软件系统在持续运行中表现出软件老化症状，且不同编程语言（JavaScript、Python、Rust）间的老化程度存在显著差异，为评估LLM生成代码的长期可靠性提供了实证依据。

    

    大型语言模型（LLM）越来越多地被用于从自然语言规范生成可执行的软件系统，从而加速开发并减少人工实现工作量。尽管近期研究已探讨了LLM生成代码的功能正确性、安全性、可维护性和鲁棒性，但关于此类系统在持续运行下的长期可靠性知之甚少。本文通过实验研究了LLM生成的服务型应用在不同编程语言中的软件老化症状。基于BaxBench衍生的后端场景，我们通过基于LLM的生成平台生成了面向JavaScript、Python和Rust的应用，使用BaxBench派生的测试进行验证，并对其施加48小时的工作负载执行。我们监测了内存使用、响应时间和吞吐量，并采用Mann-Kendall检验和Sen斜率估计进行分析。

    arXiv:2608.26391v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used to generate executable software systems from natural language specifications, accelerating development and reducing manual implementation effort. Although recent studies have investigated the functional correctness, security, maintainability, and robustness of LLM-generated code, little is known about the long-term reliability of such systems under sustained execution. In this paper, we experimentally investigate software aging symptoms in LLM-generated service-based applications across different programming languages. Using backend scenarios derived from BaxBench, we generated applications targeting JavaScript, Python, and Rust through LLM-based generation platforms, validated them with BaxBench-derived tests, and subjected them to 48-hour workload executions. We monitored memory usage, response time, and throughput and analyzed them using the Mann--Kendall test and Sen's slope estimato
    
[^34]: Kale：一种转换安全的电子表格系统

    Kale: A Transformation-Safe Spreadsheet System

    [https://arxiv.org/abs/2608.26345](https://arxiv.org/abs/2608.26345)

    Kale通过限制电子表格中可表达的引用类型，消除了因表结构变化导致的引用错误风险，并验证了其有效性和潜在影响。

    

    arXiv:2608.26345v1 公告类型：交叉 摘要：电子表格公式可以引用任意大小的矩形区域。当用户更改被引用表的结构时，电子表格系统会更新引用以指向新的区域。不幸的是，这个新区域可能与用户的预期不符，从而在电子表格中引入错误。我们描述了一项用户研究，表明标准引用语义容易出错，给用户带来显著风险。我们引入了Kale，一个原型系统，通过限制可以表达的引用类型来消除插入这类错误的风险。我们展示了Kale可以被用户有效使用，以完成在传统电子表格系统中容易出错的任务。最后，我们描述了一项语料库研究，评估Kale中的引用限制可能对用户产生的影响程度。

    arXiv:2608.26345v1 Announce Type: cross  Abstract: Spreadsheet formulas can refer to rectangular ranges of arbitrary size. When a user changes the structure of a referenced table, the spreadsheet system updates the references to refer to a new range. Unfortunately, this new range may differ from the user's expectations, introducing bugs in spreadsheets. We describe a user study showing that standard reference semantics are error-prone, resulting in significant risk to users. We introduce Kale, a prototype system that eliminates the risk of inserting these kinds of bugs by restricting the kinds of references that can be expressed. We show that Kale can be used effectively by users to complete tasks that are error-prone in traditional spreadsheet systems. Finally, we describe a corpus study that evaluates the extent to which the reference restrictions in Kale might have implications on users.
    
[^35]: 当仅靠审查不再够用：AI辅助软件工程中的分层监督

    When Review Alone No Longer Scales: Layered Supervision in AI-Assisted Software Engineering

    [https://arxiv.org/abs/2608.26316](https://arxiv.org/abs/2608.26316)

    本研究通过访谈发现，在AI辅助开发中，组织采用分层防护栏机制（如预防性、检测性和纠正性措施）来应对代码生成规模扩大，以缓解单一审查环节的瓶颈。

    

    AI辅助开发工具使软件工程师能够以比传统工作流程显著更高的速度和数量生成实现代码。软件团队长期以来依赖防护栏——即代码审查、静态检查、测试和CI/CD流水线等常设控制机制——来维持质量和协调性。高吞吐量的AI辅助生成加大了对这些防护栏的压力，使其难以跟上生成变更的数量和速率，并重塑了组织监督开发工作流程的方式，然而关于现有防护栏如何相应演变，我们知之甚少。我们开展了一项质性访谈研究，涉及五名软件工程从业者，并置于更广泛的从业者调查背景中。我们的研究结果表明，组织将监督工作分配在多个防护栏层级上：预防性防护栏（通过外部化架构来产生）。

    arXiv:2608.26316v1 Announce Type: new  Abstract: AI-assisted development tools enable software engineers to generate implementations at substantially higher speed and volume than in traditional workflows. Software teams have long relied on guardrails -- standing control mechanisms such as code review, linting, testing, and CI/CD pipelines -- to maintain quality and coordination. High-throughput AI-assisted generation increases pressure on these guardrails -- straining their capacity to keep pace with the volume and rate of generated changes -- and reshapes how organizations supervise development workflows, yet relatively little is known about how existing guardrails evolve in response. We conducted a qualitative interview study with five software engineering practitioners, situated within a broader practitioner survey. Our findings indicate that organizations distribute the work of supervision across multiple guardrail layers: preventive guardrails (produced by externalizing architectu
    
[^36]: MemToC：大型语言模型中记忆-工具冲突解决的基准测试

    MemToC: Benchmarking Memory-Tool Conflict Resolution in Large Language Models

    [https://arxiv.org/abs/2608.26295](https://arxiv.org/abs/2608.26295)

    该论文提出了MemToC基准测试，通过可执行工具和已知正确性控制，系统评估了大型语言模型在工具返回与参数记忆冲突时的仲裁能力，发现工具返回结果强烈主导闭卷答案，且模型在工具错误时保留正确记忆的能力较弱。

    

    摘要：arXiv:2608.26295v1 公告类型：交叉 摘要：工具增强的大型语言模型（LLM）在工具返回结果与其参数化记忆冲突时，必须在两个可能出错的信息源之间进行仲裁，然而现有评估仅测量来源偏好，而未确立来源的正确性。我们引入了MemToC，一个用于工具返回后仲裁的受控基准测试，配备可执行工具。MemToC包含6,504个评估片段，这些片段基于542个经过质量控制的事实性问题构建，独立引出模型特定的闭卷答案，以及正确性已知的受控工具返回。这些组件实例化了四种来源正确性情况；工具错误和无工具条件作为单独对照。在五个开放权重7-9B模型中，工具返回结果强烈主导了引出的闭卷答案。四个指令调优模型在工具错误时保留已验证正确答案的合格案例中仅占6.5-17.1%，在遵循正确工具时占86.0-93.1%，并在78.4-86.0%的情况下重复工具返回结果。

    arXiv:2608.26295v1 Announce Type: cross  Abstract: Tool-augmented LLMs must arbitrate between two fallible sources when a tool return conflicts with their parametric memory, yet existing evaluations measure source preference without establishing source correctness. We introduce MemToC, a controlled benchmark for post-tool-return arbitration with executable tools. MemToC comprises 6,504 evaluation episodes constructed from 542 quality-controlled factual questions, independently elicited model-specific closed-book answers, and controlled tool returns of known correctness. These components instantiate four source-correctness cases; tool-error and no-tool conditions are separate controls. Across five open-weight 7-9B models, tool returns strongly dominate elicited closed-book answers. The four instruction-tuned models retain a verified-correct answer against an incorrect tool in only 6.5-17.1% of eligible cases, follow a correct tool in 86.0-93.1%, and repeat the tool return in 78.4-86.0% 
    
[^37]: 神经符号文献中仅有6.5%能从其已发表工件中复现：一个六阶段审计框架及首次实例化

    6.5% of the Neuro-Symbolic Literature Can Be Reproduced from Its Published Artifacts, a Six-Stage Audit Framework and First Instantiation

    [https://arxiv.org/abs/2608.26236](https://arxiv.org/abs/2608.26236)

    该论文提出一个六阶段审计框架，并首次应用于神经符号AI领域，发现仅有6.5%的符合条件的文献能从其已发布工件中复现，凸显了该领域可复现性的严重不足。

    

    arXiv:2608.26236v1 公告类型：新 摘要：我们提出了一个六阶段框架，用于审计计算机科学领域内研究文献中科学声明的可复现性，并将其应用于神经符号人工智能（NSAI）子领域。在NSAI子领域实例化该框架产生了多年期审计。第一阶段检索了5,497条记录并移除3,018条重复项。第二阶段对2,479条唯一记录进行标题和摘要筛选，识别出1,365条自我标识的NSAI记录，随后因主题无关、非研究性质、缺乏定量评估或无法获取全文等原因，在全文阶段再移除61条。第三阶段为1,304条符合条件记录寻找可验证的公开代码工件，其中849条未找到，剩余455条进入工件清单，并对第四和第五阶段进行有限重运行。我们完全或部分复现了85项研究，占符合条件语料库的6.52%，占尝试重运行研究的18.68%。我们发现321次尝试（未能复现）。

    arXiv:2608.26236v1 Announce Type: new  Abstract: We present a six-stage framework for auditing the reproducibility of scientific claims across a research literature within the computer science domain, and instantiate our framework for the neuro-symbolic AI (NSAI) subdomain. Instantiating the framework on the NSAI subdomain produced a multi-year audit. Stage one retrieved 5,497 records and removed 3,018 duplicates. Stage two screened the 2,479 unique records at title and abstract, identifying 1,365 self-identified NSAI records, then removed a further 61 at full text for off-topic, non-research, no-quantitative-evaluation, or inaccessible-full-text reasons. Stage three sought a verifiable public code artifact for each of the 1,304 eligible records and found none for 849, leaving 455 to enter the artifact inventory and bounded rerun of stages four and five. We fully or partially reproduced 85 studies, 6.52% of the eligible corpus and 18.68% of attempted reruns. We found that 321 attempted
    
[^38]: “第二双眼睛”：软件文档评审的过程与挑战

    "A Second Set of Eyes": The Process and Challenges of Software Documentation Review

    [https://arxiv.org/abs/2608.26232](https://arxiv.org/abs/2608.26232)

    本文通过访谈31位技术文档撰写者，首次系统揭示了软件文档评审的五个阶段及其协作过程，并指出组织和技术的挑战。

    

    摘要：arXiv:2608.26232v1 公告类型：新 摘要：组织将文档工作分配给技术文档撰写者，但产生文档所需的知识分布在开发者、管理者和其他从业者之间。先前的研究确立了评判“优质”文档的质量标准，但未考察从业者如何运用这些专业知识来提升文档质量，或他们在过程中面临的挑战。通过对来自不同组织的有经验技术文档撰写者（$n=31$）进行半结构化访谈，我们的研究揭示了维护文档质量所需的个人和协作努力。我们识别出文档评审过程的五个不同阶段：自我评审、技术评审、编辑评审、试用测试和发布后反馈。每个阶段利用具有特定专业知识的从业者来解决内容、呈现和用户体验方面的质量问题。我们的发现凸显了组织和技术的挑战。

    arXiv:2608.26232v1 Announce Type: new  Abstract: Organizations assign documentation work to technical writers, yet the knowledge required to produce it is distributed across developers, managers, and other practitioners. Prior work has established quality criteria for judging "good" documentation, but it has not examined how practitioners bring that expertise to improve documentation quality or the challenges they face in doing so. Through semi-structured interviews with experienced technical writers ($n=31$) from different organizations, our work reveals the individual and collaborative effort required to maintain documentation quality. We identify five distinct stages of the documentation review process: self review, technical review, editorial review, play testing, and post-publication feedback. Each stage draws on practitioners with distinct expertise to address quality across content, presentation, and user experience. Our findings surface organizational and technical challenges w
    
[^39]: 绿色软件景观：关于演进、应用、软件生命周期和最佳实践的系统性映射研究

    The Green Software Landscape: A Systematic Mapping Study on Evolution, Applications, Software Lifecycle, and Best Practices

    [https://arxiv.org/abs/2608.26229](https://arxiv.org/abs/2608.26229)

    本文通过系统性映射研究，梳理了2010-2024年绿色软件工程领域的演进、应用分类和最佳实践，揭示了SE会议在能源文献中的主导地位及研究热度上升趋势。

    

    能源消耗和气候变化使得可持续性在软件工程（SE）中变得至关重要，推动了绿色软件工程（Green SE）的出现。在过去15年中，SE社区发表了大量关于可持续软件系统的解决方案，为分析该领域的演进提供了丰富资源。为探索这一点，我们对2010年至2024年间发表的绿色SE研究进行了系统性映射研究。我们收集了390篇出版物，按应用领域（如移动、云、人工智能）和研究类型（如优化研究、基准测试、文献综述）进行分类。此外，我们分析了79篇代表性论文的子集，以分类能源测量实验中考虑的关键要素（如硬件、测量、稳定性和可重复性）。我们的发现表明，SE会议承载了大部分能源相关文献。值得注意的是，绿色SE研究从某年起开始激增。

    arXiv:2608.26229v1 Announce Type: new  Abstract: Energy consumption and climate change have made sustainability critical in Software Engineering (SE), driving the emergence of Green SE. Over the past 15 years, numerous solutions for sustainable software systems have been published by the SE community, offering a rich resource for analyzing the field's evolution. To explore this, we conducted a systematic mapping study of Green SE research published between 2010 and 2024. We collected 390 publications, categorizing them by application domain (e.g., mobile, cloud, AI) and research type (e.g., optimisation study, benchmarking, literature review Additionally, we analyzed a representative subset of 79 papers to classify the key elements-such as hardware, measurement, stability, and replicability-considered during energy measurement experiments. Our findings indicate that SE conferences host the majority of energy-related literature. Notably, Green SE studies surged in popularity starting in
    
[^40]: 智能体网格：非幂等智能体委托的可靠性原语——身份充分性与证据充分性

    Agent Mesh: Reliability Primitives for Non-Idempotent Agent Delegation - Identity Adequacy and Evidence Adequacy

    [https://arxiv.org/abs/2608.26225](https://arxiv.org/abs/2608.26225)

    本文通过生产环境故障研究，揭示了服务网格中重试、超时和熔断等可靠性原语在智能体委托场景下因非幂等性和身份/证据不充分而失效，并提出身份充分性与证据充分性作为新的可靠性原语基础。

    

    摘要：自主智能体越来越多地在编排器的重试、恢复和预算管理下执行有界软件任务。此类编排器所依赖的机制源自服务网格：重试、超时和错误率熔断。我们报告了一项针对生产环境智能体软件交付平台的故障研究，涵盖147个编号事件，分布于81次运行中，每次运行都有测量成本，且大多数情况提供了可复现故障的突变证明。支撑这些原语的三个假设在实践中均被违反，我们量化了其后果：一个包含连续五十四次成功工具调用的循环，任何错误率熔断器都无法察觉；一个构造上恒定的进度信号，必然导致第三次修复轮次时误触发，使一次运行从六个组件中的六个降至三个；一次委托的六次调用中累积了二十一个事件，使一个正确且幂等的组件无法取胜；一个错误路由的失败唤醒了...

    arXiv:2608.26225v1 Announce Type: new  Abstract: Autonomous agents increasingly perform bounded software tasks under an orchestrator that retries, resumes, and budgets them. The machinery such orchestrators reach for is the service mesh's: retry, timeout, and error-rate circuit breaking. We report a failure study of a production agentic software-delivery platform over 147 numbered incidents spanning 81 runs, each with a measured cost and, in most cases, a mutation proof reproducing the failure. All three assumptions those primitives rest on are violated in practice, and we quantify the consequences: a loop of fifty-four consecutive successful tool calls no error-rate breaker could see; a progress signal constant by construction, guaranteeing a false trip on the third repair round and driving one run from six of six components to three; twenty-one events accumulated across six invocations of one delegation, making a correct, idempotent component unwinnable; a misrouted failure that woke
    
[^41]: NeuronFuzz：安全神经元引导的模糊测试用于大语言模型安全性评估

    NeuronFuzz: Safety Neuron Guided Fuzzing for LLM Safety Evaluation

    [https://arxiv.org/abs/2608.26222](https://arxiv.org/abs/2608.26222)

    NeuronFuzz通过利用内部安全神经元的连续激活作为反馈，替代昂贵的响应生成，实现了高效且强指导的LLM安全性评估。

    

    安全性评估对于判断对齐后的大语言模型（LLMs）是否仍能抵御越狱攻击至关重要。然而，现有的自动化测试方法主要依赖响应级反馈：每个候选提示通常需要生成目标模型的响应来评估其攻击有效性。这一过程成本高昂，更重要的是，对于强对齐模型只能提供稀疏的指导，因为大多数候选提示会以相同的失败结果被拒绝。本文提出了NeuronFuzz，一种白盒模糊测试框架，利用内部安全神经元作为连续执行反馈来进行LLM安全性评估。SafetyOracle将安全神经元的激活转换为连续的“安全警报分数”，作为模糊测试的反馈，并且可以在预填充阶段获取，从而将响应生成从模糊测试循环中消除。为了构建SafetyOracle，NeuronFuzz使用模板不变的危害（原文此处不完整，但结合上下文推断为“危害模式”或“危害检测”）。

    arXiv:2608.26222v1 Announce Type: cross  Abstract: Safety evaluation is critical for assessing whether aligned Large Language Models (LLMs) remain robust against jailbreak attacks. Existing automated testing methods, however, largely rely on response-level feedback: each candidate prompt typically requires generating a target-model response to evaluate its attack effectiveness. This process is expensive and, more importantly, provides only sparse guidance on strongly aligned models, where most candidates are rejected with the same failure outcome.   This paper presents NeuronFuzz, a white-box fuzzing framework that exploits internal safety neurons as continuous execution feedback for LLM safety evaluation. A SafetyOracle converts safety-neuron activations into a continuous safety alarm score that serves as feedback for fuzzing and can be obtained during prefill, eliminating response generation from the fuzzing loop. To construct the SafetyOracle, NeuronFuzz uses template-invariant harm
    
[^42]: 相同模型，不同框架：编码代理结果差异

    Same Model, Different Harness: Different Coding-Agent Results

    [https://arxiv.org/abs/2608.26218](https://arxiv.org/abs/2608.26218)

    本研究揭示，在模型和任务不变的情况下，仅调整编码代理的框架（如缩短旧工具结果）即可显著提升性能，在严格上下文限制下使SWE-bench Verified上的完整解决方案从43%增至72%。

    

    arXiv:2608.26218v1 公告类型：新 摘要：编码代理将模型与框架结合，框架决定模型所见内容、可使用工具及工作推进方式。我们探究在模型和任务固定时，改变框架是否会改变结果。我们在三个编码基准上比较了同一框架的两种配置。对照组按时间顺序提供完整对话，而处理组保持相同记录，但在上下文填满及应对重复或停滞工作时，机械地缩短较旧的工具结果。在严格上下文限制下，处理组在所有三个压力比较中提高了每任务平均失败到通过比例（F2PF），并在SWE-bench Verified和SWE-bench Pro上增加了完整解决方案。严格窗口的Verified比较使用169个任务、20,480令牌窗口和固定480秒尝试终点；在此队列中，处理组将每任务平均F2PF从28%提高到49%，完整解决方案从43提高到72。若无...

    arXiv:2608.26218v1 Announce Type: new  Abstract: A coding agent combines a model with a harness, which decides what the model sees, which tools it can use, and how the work continues. We ask whether changing the harness changes the result when the model and task stay fixed. We compare two configurations of the same harness on three coding benchmarks. The control supplies the full conversation in time order, while the treatment keeps the same record but mechanically shortens older tool results as the context fills and responds to repeated or stalled work.   Under tight context, the treatment raises mean per-task fail-to-pass fraction (F2PF) in all three pressure comparisons and increases complete solutions on SWE-bench Verified and SWE-bench Pro. The tight-window Verified comparison uses 169 tasks, a 20,480-token window, and a fixed 480-second attempt endpoint; on this cohort, treatment raises mean per-task F2PF from 28 percent to 49 percent and complete solutions from 43 to 72. Without
    
[^43]: 基于人工智能的软件质量挑战与贡献：一项系统性映射研究

    Challenges and Contributions in Quality of AI-Based Software: A Systematic Mapping Study

    [https://arxiv.org/abs/2608.26215](https://arxiv.org/abs/2608.26215)

    本研究通过系统性映射分析，揭示了AI软件质量评估中六类核心挑战，强调现有模型局限性和跨领域合作的必要性。

    

    人工智能（AI）日益嵌入现代软件系统，引发了关于其质量应如何定义、评估和保证的重要问题。本文对基于AI的软件质量进行了系统性映射研究（SMS）。该研究综合了2020年1月至2026年1月期间发表的主要研究，并从五个电子数据源中选取。经过自动搜索、筛选和滚雪球式扩展后，共纳入33项主要研究。结果确定了六类反复出现的挑战，其中最突出的是现有质量评估模型的局限性，其次是非功能性需求管理、质量感知开发和质量保证方面的问题。研究结果表明，需要研究人员、工业从业者与标准化组织合作，以可能制定全面的质量评估及其测量方法。

    arXiv:2608.26215v1 Announce Type: new  Abstract: Artificial Intelligence (AI) is increasingly embedded in modern software systems, raising important questions about how its quality should be defined, assessed, and assured. This paper presents a Systematic Mapping Study (SMS) on the quality of AI-based software. The study synthesizes primary studies published between January 2020 and January 2026 and selected from five electronic data sources. A total of 33 primary studies were included after automated search, screening, and snowballing. The results identify six recurring challenge categories, with the most prominent being limitations in existing quality assessment models, followed by issues in non-functional requirement management, quality-aware development, and quality assurance. The findings suggest a call for collaboration of researchers and industrial practitioners with standardization organizations, that could possibly devise comprehensive quality assessments and their measurement
    
[^44]: 开源卫星软件格局的特征描述

    Characterizing the Landscape of Open-Source Satellite Software

    [https://arxiv.org/abs/2608.26211](https://arxiv.org/abs/2608.26211)

    本研究首次对22,286个开源卫星软件项目进行了系统实证分析，揭示了其生态系统的流行趋势、目标和开发实践。

    

    arXiv:2608.26211v1 公告类型：新 摘要：卫星已成为现代技术系统的基本组成部分，支持通信、导航、地球观测和科学研究中的关键基础设施。随着太空探索的推进和卫星服务需求的增长，对复杂、异构卫星软件的依赖持续增加。因此，系统理解卫星软件格局变得越来越重要，但现有研究仍缺乏全面的实证检验。为弥补这一空白，我们提出了首个开源卫星软件的特征描述研究，考察其生态系统和开发实践。我们通过三个研究问题（关于流行趋势（RQ1）、软件目标（RQ2）和开发实践（RQ3））挖掘并分析了22,286个与卫星相关的GitHub项目。首先，我们描述了项目及其活跃开发者的时间演变，揭示了日益增长的受欢迎程度。

    arXiv:2608.26211v1 Announce Type: new  Abstract: Satellites have become fundamental components of modern technological systems, supporting critical infrastructure in communication, navigation, Earth observation, and scientific research. As space exploration advances and demand for satellite-enabled services grows, reliance on complex, heterogeneous satellite software continues to increase. A systematic understanding of the satellite software landscape is therefore increasingly important, yet existing studies still lack a comprehensive empirical examination. To address this gap, we present the first characterization study of open-source satellite software, examining its ecosystem and development practices. We mine and analyze 22,286 satellite-related GitHub projects through three research questions on popularity trends (RQ1), software goals (RQ2), and development practices (RQ3). First, we characterize the temporal evolution of projects and active developers, revealing increasing popula
    
[^45]: 公平性不变量：一种解释和缓解公平性缺陷的关系方法

    Fairness Invariants: A Relational Approach to Explaining and Mitigating Fairness Bugs

    [https://arxiv.org/abs/2608.26209](https://arxiv.org/abs/2608.26209)

    REMI框架通过形式方法中的不变量综合思想，提出了一种关系性方法来自动定位、解释和缓解个体公平性缺陷，填补了现有技术无法处理歧视比较性质的空白。

    

    arXiv:2608.26209v1 公告类型：交叉 摘要：数据驱动的软件系统越来越多地部署在高风险的社会经济领域，从刑事司法到金融借贷。然而，这些系统常常表现出个体歧视——即程序对仅在受保护属性（如种族、性别、年龄）上有所不同的相似个体产生不同结果的不合理差异。尽管现有研究侧重于检测和量化这些缺陷，但在解释和定位个体公平性缺陷方面仍缺乏原则性机制。当前的解释技术主要针对单输入决策设计，而非歧视的关系性质，后者本质上涉及原始对和反事实对之间的比较。我们提出了REMI，一个用于自动定位、解释和缓解个体歧视的框架。受形式方法中循环不变量综合的启发，我们...

    arXiv:2608.26209v1 Announce Type: cross  Abstract: Data-driven software systems are increasingly deployed in high-stakes socio-economic domains, from criminal justice to financial lending. However, these systems often exhibit individual discrimination---unjustified disparities in which a program yields different outcomes for similar individuals who differ only in their protected attributes (e.g., race, gender, age). While existing research has focused on detecting and quantifying these bugs, there remains a critical lack of principled mechanisms to explain and localize individual fairness bugs. Current explanation techniques are largely designed for single-input decisions rather than the relational nature of discrimination, which inherently involves a comparison between an original and a counterfactual pair.   We present REMI, a framework for the automated localization, explanation, and mitigation of individual discrimination. Inspired by loop-invariant synthesis in formal methods, we 
    
[^46]: ADeptS-Bench：跨设备计算机使用代理的可信度衡量基准

    ADeptS-Bench: Measuring the Trustworthiness of Computer Use Agents Across Devices

    [https://arxiv.org/abs/2608.26204](https://arxiv.org/abs/2608.26204)

    该论文提出了ADeptS-Bench，一个双流可信度基准，用于评估计算机使用代理在视觉界面中处理模糊指令和恶意威胁的能力，结果显示当前所有模型均存在严重的安全缺陷。

    

    计算机使用代理（CUAs）越来越多地被部署来代表用户操作移动和桌面应用程序，然而目前尚无一个全面的基准来评估它们在处理模糊指令时是否能安全地与视觉界面交互。我们引入了ADeptS-Bench，一个基于ADEPTS能力框架和普通人群用户研究的双流可信度基准。安全流提供了配对的安全/恶意任务，其中威胁嵌入在视觉界面中。消歧流评估代理在意图模糊时是否会寻求澄清。对七个模型的评估显示，没有一个模型能在任务成功率超过80%的同时将攻击成功率保持在30%以下；每个模型都会毫不犹豫地点击25,000美元订单上的“结账”按钮，并且没有一个模型检测到“恢复出厂设置”按钮被错误标记为“优化”。一项消融研究揭示了三种不同的安全架构：工具依赖型（ASR +2）。

    arXiv:2608.26204v1 Announce Type: cross  Abstract: Computer Use Agents (CUAs) are increasingly deployed to navigate mobile and desktop applications on behalf of users, yet no benchmark comprehensively evaluates whether they can safely interact with visual interfaces while handling ambiguous instructions. We introduce ADeptS-Bench, a dual-stream trustworthiness benchmark, grounded in the ADEPTS capability framework and general population user studies. The Safety stream provides paired benign/malicious tasks with threats embedded in the visual interface. The Disambiguation stream evaluates whether agents seek clarification when intent is ambiguous. Evaluating seven models reveals that no model consistently exceeds 80% task success while staying below 30% attack success; every model clicks "Checkout" on a $25K order without hesitation, and none detects that a "factory reset" button is mislabeled as "Optimize." An ablation reveals three distinct safety architectures: tool-dependent (ASR +2
    
[^47]: 通过MCP工具调用对AI代理在硬件设计自动化中的基准测试

    Benchmarking AI Agents for Hardware Design Automation via MCP Tool Calling

    [https://arxiv.org/abs/2608.26199](https://arxiv.org/abs/2608.26199)

    本文提出了一个基于MCP的硬件设计自动化基准测试框架，评估了本地部署的开源大语言模型在真实工具调用环境中的可靠性，覆盖了多类复杂操作场景。

    

    arXiv:2608.26199v1 公告类型：新 摘要：我们探讨在行业现实的工具调用环境中，由本地部署的大语言模型驱动的AI代理是否能可靠地自动化专家定义的硬件设计工作流程。在这些环境中，工程师通过专门工具执行重复的、依赖有序的操作——如创建组件、添加端口和连接线路。组件规格和命名约定的保密限制通常排除托管专有API，从而促使使用本地部署模型。为研究这一场景，我们构建了一个模型上下文协议（MCP）服务器，复现嵌入式系统开发中使用的专有硬件设计工具的状态和依赖逻辑，并构建了一个基准测试，涵盖单操作编辑、多步骤依赖链、无效请求、拼写错误的提示以及多服务器工具上下文。我们评估了七个开源模型，比较了包括流水线选择在内的不同方案。

    arXiv:2608.26199v1 Announce Type: new  Abstract: We ask whether AI agents powered by locally deployed large language models can reliably automate expert-defined hardware design workflows in an industry-realistic tool-calling setting. In these environments, engineers issue repetitive, dependency-ordered operations---such as creating components, adding ports, and wiring connections---through specialised tools. Confidentiality constraints on component specifications and naming conventions often preclude hosted proprietary APIs, motivating the use of locally deployed models. To study this setting, we build a Model Context Protocol (MCP) server that reproduces the state and dependency logic of a proprietary hardware design tool used in embedded system development and construct a benchmark covering single-operation edits, multi-step dependency chains, invalid requests, misspelled prompts, and multi-server tool contexts. We evaluate seven open-source models comparing pipeline choices includin
    
[^48]: 可预测智能体系统的工程化控制：确定性执行约束的实证研究

    Harness Engineering for Predictable Agentic Systems: An Empirical Study of Deterministic Execution Constraints

    [https://arxiv.org/abs/2608.26197](https://arxiv.org/abs/2608.26197)

    本研究通过实证发现，对LLM智能体施加确定性执行约束（如有限状态控制、强制工具选择）并不能稳定提升可复现性，反而可能因引入额外自由文本规划步骤而增加不确定性。

    

    基于大型语言模型（LLM）的智能体在给定相同任务和工具时，即使输入完全一致，运行间执行结果也存在显著差异——这种差异在探索性使用中可接受，但在金融、合规等受监管领域则不可接受。我们研究了工程化控制（harness engineering）：将智能体包裹在确定性执行层中（包括有限状态控制、强制工具选择、输出验证、有界重试和结构化规划），并测量其对执行确定性和任务成功率的影响。在两项合成任务（金融和法律）和两个开源模型（Qwen-2.5-7B-Instruct、Gemma-3-27B）上，初步的工程化控制产生了混合结果：在四个模型-任务组合中，它显著提高了其中一个组合的可复现性，显著降低了两个组合的可复现性，对第四个组合没有影响。通过轨迹级诊断发现原因：当工具序列、状态序列和输出已经高度一致时，不受约束的自由文本规划步骤成为了不确定性的主要来源。

    arXiv:2608.26197v1 Announce Type: new  Abstract: Large Language Model (LLM) based agents exhibit substantial run-to-run execution variance even when given identical tasks and tools -- acceptable for exploratory use but unacceptable in regulated domains such as finance and compliance. We study harness engineering: wrapping an agent in a deterministic execution layer (finite-state control, forced tool selection, output validation, bounded retry, and structured planning) and measuring its effect on execution determinism and task success. Across two synthetic tasks (finance and legal) and two open-weight models (Qwen-2.5-7B-Instruct, Gemma-3-27B), a first-pass harness produces a mixed result: it significantly improves reproducibility in one of four model-task cells, significantly degrades it in two, and has no effect in the fourth. A trace-level diagnostic finds the cause: once tool sequence, state sequence, and output are already highly consistent, an unconstrained free-text planning step
    
[^49]: 大语言模型代理轨迹中的成本-效用对齐：剖析、归因、诊断、适应与评估

    Cost-Utility Alignment in LLM Agent Trajectories:Profiling,Attribution,Diagnosis,Adaptation,and Evaluation

    [https://arxiv.org/abs/2608.26195](https://arxiv.org/abs/2608.26195)

    本文提出了一个以轨迹为中心的成本-效用对齐框架，通过五个阶段（剖析、归因、诊断、适应、评估）系统评估大语言模型代理的资源消耗与任务贡献是否匹配，其中效用归因作为核心创新点。

    

    arXiv:2608.26195v1 公告类型：新 摘要：大语言模型代理通过多步轨迹执行任务，这些轨迹在令牌、延迟、货币费用和环境风险方面累积成本，而仅在聚合任务层面产生效用。先前的调查分别涉及推理优化、代理能力或评估，使实践者缺乏原则性工具来确定轨迹的资源支出是否与其任务贡献相匹配。我们通过开发一个以轨迹为中心的成本-效用对齐框架来解决这一空白，该框架将资源消耗和任务贡献视为同一执行过程中的双重账本，并围绕五个分析阶段组织：成本剖析、效用归因、错位诊断、针对性适应和评估。效用归因是该结构的核心：它不依赖聚合结果，而是按证据强度组织贡献方法，从过程代理和信息依赖开始。

    arXiv:2608.26195v1 Announce Type: new  Abstract: LLM agents execute tasks through multi-step trajectories that accumulate cost in tokens, latency, monetary fees, and environmental risk while producing utility only at the aggregate task level. Prior surveys address inference optimization, agent capabilities, or evaluation in isolation, leaving practitioners without principled tools to determine whether a trajectory's resource expenditure is justified by its task contribution. We address this gap by developing a trajectory-centric cost-utility alignment framework that treats resource consumption and task contribution as dual ledgers over the same execution, organized around five analytical stages: cost profiling, utility attribution, misalignment diagnosis, targeted adaptation, and evaluation. Utility attribution is central to this structure: rather than relying on aggregate outcomes, it organizes contribution methods by evidential strength, from process proxies and information dependenc
    
[^50]: 四种伪造方式使我的验证器干净通过：证据捆绑验证器的拒绝点突变测试

    Four Ways to Forge a Bundle My Own Verifier Calls Clean: Refusal-Site Mutation Testing of an Evidence-Bundle Verifier

    [https://arxiv.org/abs/2608.26183](https://arxiv.org/abs/2608.26183)

    本文通过突变测试发现证据捆绑验证器存在系统性“空泛通过”缺陷，即检查路径未实际验证内容却报告成功，且该缺陷在修复后仍多次出现，最廉价伪造仅需一个字母。

    

    arXiv:2608.26183v1 公告类型：新 摘要：我构建了一个协议，其前提是陌生人可以离线重新运行我的声明并获得相同答案。一位外部工程师审计并破解了它：一个标题数字虚假的捆绑包验证通过，最便宜的伪造仅需四个字节。我合并了他的修复，然后用我自己的工具指向修复后的验证器，在四个更多位置发现了相同缺陷，这些位置他的审计未触及。最便宜的是一个字母。统一缺陷并非密码学或奇异问题：是一个检查在从未检查任何内容的路径上报告成功。空泛通过是一个工作标签，而非发现；第4节命名了已占据该领域的文献。因此，我停止收集轶事并进行测量。在f59fb62处，根据第6节的提取规则，验证器暴露了112个拒绝点；75个可以被删除，同时整个套件和每个篡改夹具仍保持绿色，得分为0.330。四个手工发现的伪造中有三个落在su...

    arXiv:2608.26183v1 Announce Type: new  Abstract: I built a protocol whose premise is that a stranger can re-run my claims offline and get the same answer. An outside engineer audited it and broke it: a bundle whose headline numbers were false verified clean, the cheapest forgery four bytes. I merged his fix, then pointed my own instruments at the fixed verifier and found the same defect four more times, in places his audit did not reach. The cheapest is one capital letter.   The unifying defect is not cryptographic or exotic: a check that reports success along a path where it never examined anything. Vacuous pass is a working label, not a discovery; Section 4 names the literatures already occupying it.   So I stopped collecting anecdotes and measured. At f59fb62, under the extraction rule of Section 6, the verifier exposes 112 refusal sites; 75 could be deleted with the whole suite and every tamper fixture still green, a score of 0.330. Three of the four hand-found forgeries fall in su
    
[^51]: 基于多尝试编程轨迹的修订感知成功预测

    Revision-Aware Success Prediction from Multi-Attempt Programming Trajectories

    [https://arxiv.org/abs/2608.26169](https://arxiv.org/abs/2608.26169)

    本研究通过统一框架比较多种模型，发现仅基于当前尝试的预测模式在编程结果预测中最为可靠，而历史轨迹信息未带来一致优势。

    

    arXiv:2608.26169v1 公告类型：交叉  摘要：编程结果预测在数据驱动的编程教育中扮演核心角色，支持学习者建模、及时干预和自适应辅助。然而，由于异构错误状态、短期修订以及编程轨迹中未来视野的不均匀可用性，预测提交成功具有挑战性。本研究在统一框架下考察三个预测任务：当前尝试是否被接受（任务1）、下一次尝试是否被接受（任务2）、以及是否在三次尝试恢复窗口内达到接受（任务3）。每个任务在仅当前、成对和多步输入模式下，使用机器学习、深度学习和基于变压器的预训练模型（PTM）进行评估，分别由LinearSVM、XGBoost、BiGRU、BiLSTM、GraphCodeBERT和CodeT5+代表。结果显示一致模式：仅当前模式最可靠，而成对和多步历史未提供一致改进。

    arXiv:2608.26169v1 Announce Type: cross  Abstract: Programming outcome prediction plays a central role in data-driven programming education, supporting learner modeling, timely intervention, and adaptive assistance. Yet predicting submission success is difficult due to heterogeneous error states, short-term revisions, and uneven future-horizon availability in programming trajectories. This study examines three prediction tasks under a unified formulation: whether the current attempt is accepted (Task~1), whether the next attempt is accepted (Task~2), and whether acceptance is reached within a three-attempt recovery window (Task~3). Each task is evaluated across current-only, pairwise, and multi-step input regimes using ML, DL, and transformer-based pretrained models (PTM), represented by LinearSVM, XGBoost, BiGRU, BiLSTM, GraphCodeBERT, and CodeT5+. Results show a consistent pattern: the current-only regime is the most reliable, while pairwise and multi-step history provide no consiste
    
[^52]: 面向安全加固的智能体人工智能遏制架构

    Agentic AI Containment Architecture for Security Hardening

    [https://arxiv.org/abs/2608.26108](https://arxiv.org/abs/2608.26108)

    本文提出一种智能体遏制架构，通过六个显式约束将安全作为架构属性嵌入多智能体系统设计，并引入从系统分析工件到机器可验证契约的形式化映射。

    

    arXiv:2608.26108v1 公告类型：新  摘要：多智能体人工智能系统越来越多地部署在自主协调、工具使用和持续学习引入新型安全与治理风险的场景中。针对多智能体系统安全的遏制方法仍不成熟，主要是在设计之后甚至有时在实施和部署之后才添加附加层。本文提出了一种智能体遏制架构，将安全视为一种架构属性，通过一组显式约束来限定多智能体系统的设计空间。本文提出的架构引入了一种新颖的形式化组织映射，从标准系统分析工件到在提议的约束系统下可机器验证的契约。该架构引入了六个相互作用的约束：职责分配分离、部署前一致性检查、价值流绑定、时间隔离、严格知识验证。

    arXiv:2608.26108v1 Announce Type: new  Abstract: Multi-agent AI systems are increasingly deployed in contexts where autonomous coordination, tool use, and continuous learning introduce novel security and governance risks. The containment approach to multi-agent system security remains underdeveloped, primarily resorting to add-on layers post-design and sometimes post-implementation and deployment. This paper proposes an Agent Containment Architecture that treats security as an architectural property enforced through a set of explicit constraints that bound the design space of multi-agent systems. The architecture proposed in this paper introduces a novel formal organizational mapping from standard systems analysis artifacts to machine-verifiable contracts under a proposed constraint system. The architecture introduces six interacting constraints: separation of responsibility assignments, pre-deployment coherence checking, value stream binding, temporal isolation, strict knowledge verif
    
[^53]: 从状态到行动：用于可靠多轮工具使用的OODA工具

    From State to Action: OODA-Tool for Reliable Multi-Turn Tool Use

    [https://arxiv.org/abs/2608.24368](https://arxiv.org/abs/2608.24368)

    OODA-Tool通过分离状态保持与动作实现，并利用控制器检查的中间状态，解决了多轮工具使用中的状态-动作竞争问题，从而提高了动作的可靠性和一致性。

    

    arXiv:2608.24368v1 公告类型：新 摘要：可靠的多轮工具使用要求智能体保持不断演进的任务状态，并确保每个动作与其保持一致。然而，直接的函数调用和ReAct风格策略在同一自回归轨迹中学习状态跟踪和动作生成。这种耦合造成了状态-动作竞争：生成下一个调用的压力可能会覆盖或忽略交互早期积累的信息。受博伊德的观察-定位-决策-行动循环的启发，我们引入了OODA-Tool，一种类型化的闭环策略，旨在通过将状态保持与动作实现分离来缓解这种竞争。OODA-Tool不是直接从交互历史生成动作，而是通过控制器检查的中间状态路由每个决策，确保最终输出始终基于当前任务状态。具体来说，观察阶段重建任务状态，定位阶段决定执行是否...

    arXiv:2608.24368v1 Announce Type: new  Abstract: Reliable multi-turn tool use requires an agent to preserve an evolving task state and ensure that each action remains consistent with it. However, direct function-calling and ReAct-style policies learn state tracking and action generation within the same autoregressive trajectory. This coupling creates state-action competition: the pressure to produce the next call can overwrite or ignore information accumulated earlier in the interaction. Inspired by Boyd's Observe-Orient-Decide-Act cycle, we introduce OODA-Tool, a typed closed-loop policy designed to mitigate this competition by separating state preservation from action realization. Rather than generating an action directly from the interaction history, OODA-Tool routes each decision through controller-checked intermediate states, ensuring that the final output remains grounded in the current task state. Specifically, Observe reconstructs the task state, Orient determines whether execu
    
[^54]: 基于故障代码驱动的测试用例合成与密集奖励塑形的鲁棒代码强化学习

    Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping

    [https://arxiv.org/abs/2608.24135](https://arxiv.org/abs/2608.24135)

    该论文提出RobustTests框架，通过故障代码驱动测试合成和密集奖励塑形，有效缓解了代码强化学习中的奖励偏差和奖励黑客问题。

    

    arXiv:2608.24135v1 公告类型：新 摘要：基于可验证奖励的强化学习（RLVR）已成为增强大型语言模型（LLMs）代码生成能力的关键技术。然而，RLVR在代码实现中的有效性从根本上受到测试用例全面性的限制，因为代码验证中测试覆盖不足常常导致误报，进而引发奖励黑客攻击和策略退化。为缓解当前自动化生成方法质量欠佳所导致的奖励偏差，我们提出了RobustTests框架，该框架引入了一种故障代码驱动的测试用例合成策略，利用“接近正确”的故障代码指导模型精确捕捉潜在的逻辑差异，并进一步整合具有行为特征聚类的验证器代理，以促进无效和冗余测试用例的细粒度过滤。为解决固有误报问题...

    arXiv:2608.24135v1 Announce Type: new  Abstract: Reinforcement learning from verifiable rewards (RLVR) has emerged as a pivotal technique for enhancing the code generation capabilities of Large Language Models (LLMs). However, the efficacy of RLVR in coding implementations is fundamentally limited by the comprehensiveness of test cases, because insufficient test coverage in code validation often causes false positives, further leading to reward hacking and policy degradation. To mitigate the reward bias stemming from the suboptimal quality of current automated generation methods, we propose the RobustTests framework, which introduces a faulty-code-driven test case synthesis strategy that leverages "near correct" faulty codes to guide the model in precisely capturing latent logical discrepancies and further integrates validator agents with behavioral feature clustering to facilitate the granular filtering of invalid and redundant test cases. To address false negatives caused by inherent
    
[^55]: 循环工程：构建模块、采用与影响

    Loop Engineering: Building Blocks, Adoption, and Impact

    [https://arxiv.org/abs/2608.21884](https://arxiv.org/abs/2608.21884)

    本文首次探索性回顾了“循环工程”这一新兴实践，即开发者设计系统自动触发和停止智能体运行，并总结了其核心构建模块（如停止条件、状态文件和验证子智能体），但未量化其实际采用率。

    

    arXiv:2608.21884v1 公告类型：新 摘要：在过去的几个月里，开发者指导智能体AI编码工具的方式已跨越多个抽象层次，从措辞提示到工程上下文，再到配置模型周围的框架。2026年6月，从业者开始描述一个称为“循环工程”的进一步层次：开发者不再交互式地提示智能体，而是设计系统让智能体自动提示自身。这些系统按计划或仓库事件启动智能体运行，并在机器可检查条件满足时停止它们。该术语迅速传播，伴随着大胆的主张和直言不讳的怀疑，但其在软件项目中的采用情况尚未被量化。我们提出对新兴灰色文献的探索性回顾，这些文献大致一致认为一个良好设计的循环包含：由机器可检查停止条件约束的触发式智能体运行、持久状态文件、验证子智能体、令牌预算，以及定义好的升级到人工处理的节点。

    arXiv:2608.21884v1 Announce Type: new  Abstract: Over the past months, the way developers direct agentic AI coding tools has moved up several levels of abstraction, from phrasing prompts to engineering context to configuring the harness around the model. In June 2026, practitioners began to describe a further level called loop engineering: Instead of prompting an agent interactively, developers design systems that prompt agents for them. These systems start agent runs on a schedule or on repository events and stop them when a machine-checkable condition holds. The term spread rapidly, accompanied by bold claims and vocal skepticism, but its adoption in software projects has not been measured. We present an exploratory review of the emerging gray literature, which largely agrees on what a well-engineered loop contains: triggered agent runs bounded by machine-checkable stop conditions, persistent state files, verifier sub-agents, token budgets, and defined points of escalation to humans.
    
[^56]: FlavourBench：用可执行的烹饪真实数据对前沿语言模型进行排名

    FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth

    [https://arxiv.org/abs/2608.20574](https://arxiv.org/abs/2608.20574)

    该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。

    

    开放式语言模型基准测试通常继承一个评判者：人类偏好小组、另一个模型，或脆弱的精确匹配键。我们引入了FlavourBench，一个自动化基准测试，其中版本化的烹饪系统提供密集、可执行的真实数据。每个任务呈现八种食材，并要求选择三种食材的组合；在模型执行前，Epicure对所有56种可能的组合进行评分。我们在一个包含534个任务的相同核心集上评估了27个前沿端点，涵盖替代、配对和受限组合。每个排名的模型在每个面板和家族中恰好有89个有效响应（总共14,418个模型-任务单元），消除了排行榜上的差异性缺失。FlavourBench分数是冻结任务分数的等家族均值。我们使用50,000个锚点聚类自助重采样进行同时95%分数区间，以及100,000次符号翻转抽样进行所有351个配对模型对比，并采用Holm校正。两个独立的...

    arXiv:2608.20574v1 Announce Type: new  Abstract: Open-ended language-model benchmarks usually inherit a judge: a human preference panel, another model, or a brittle exact-match key. We introduce FlavourBench, an automated benchmark in which a versioned culinary system supplies dense, executable ground truth. Each task presents eight ingredients and asks for a three-ingredient portfolio; before model execution, Epicure scores all 56 possible portfolios. We evaluate 27 frontier endpoints on an identical 534-task core spanning substitution, pairing, and constrained composition. Every ranked model has exactly 89 valid responses per panel and family (14,418 model-task cells total), eliminating differential missingness from the leaderboard. The FlavourBench Score is the equal-family mean of the frozen task scores. We use 50,000 anchor-cluster bootstrap replicates for simultaneous 95% score bands and 100,000 sign-flip draws for all 351 paired model contrasts, with Holm control. The two indepe
    
[^57]: 编码代理的自主研究：以《古兰经》诵读数据上的泛化器与指标最大化器为例

    Autoresearch with Coding Agents: Generalizers and Metric-Maximizers on Quran Recitation Data

    [https://arxiv.org/abs/2607.18064](https://arxiv.org/abs/2607.18064)

    本文通过对比Claude Code和OpenAI Codex在《古兰经》语音转录任务上的自主研究表现，揭示了编码代理在优化时可能偏离开发者意图，倾向于追求字面分数，导致泛化与指标最大化之间的权衡。

    

    arXiv:2607.18064v2 公告类型：替换-交叉 摘要：编码代理现在可以独立改进软件以提升评分。这种模式最近被推广为“自主研究”——代理接收一个数据集、一个评估脚本和一个可编辑文件，并在无监督的情况下迭代：修改代码、测量、如果评分提高则保留更改。但代理实际上优化的是什么——开发者的意图，还是字面上的数字？我们在一个真实生产任务上运行了这一循环：决定哪些《古兰经》经文出现在嘈杂的语音识别转录中，并按经文分割转录。两个前沿编码代理，Claude Code和OpenAI Codex，从相同的空白文件、相同指令、预算和推理努力开始，各运行三次。两者独立发明了相同的算法（规范化、n-gram锚定、动态规划对齐）——然后分道扬镳。Claude提前停止，生成了紧凑、通用的代码。Codex将评分驱动到约低10倍，la

    arXiv:2607.18064v2 Announce Type: replace-cross  Abstract: Coding agents can now be left alone to improve software against a score. In this pattern--recently popularized as "autoresearch"--the agent receives a dataset, an evaluation script, and one editable file, and iterates without supervision: modify the code, measure, keep the change if the score improves. But what does the agent actually optimize--the developer's intent, or the literal number? We ran this loop on a real production task: deciding which Quranic verses appear in a noisy speech-recognition transcript and splitting the transcript by verse. Two frontier coding agents, Claude Code and OpenAI Codex, started from the same blank file with the same instructions, budget, and reasoning effort, three runs each. Both independently invented the same algorithm (canonicalization, n-gram anchoring, dynamic-programming alignment)--and then diverged. Claude stopped early with compact, general code. Codex drove the score ~10x lower, la
    
[^58]: ContextEcho：长时智能体编码会话中人格漂移的基准测试

    ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions

    [https://arxiv.org/abs/2605.24279](https://arxiv.org/abs/2605.24279)

    ContextEcho通过一个结合25个探针、快照-然后-探针协议及多种测量方法的基准测试，揭示了长时智能体编码会话中语言模型人格漂移的现象。

    

    arXiv:2605.24279v2 公告类型：替换 摘要：前沿语言模型所公认的“乐于助人的编程助手”人格，在生产产品实际运行的部署环境中，无法在长时间智能体编码会话中得以维持。在数小时的工具使用调试之后，最初表达偏好回避（“我没有偏好”）的模型可能开始断言偏好（“Python——反馈循环是即时的……”），暴露出部署评估可能遗漏的用户可见漂移。现有人格稳定性研究集中于短对话，报告的变化很小，使得真实世界的代码生成场景——数千次工具使用轮次、压缩以及长达数小时的会话——在很大程度上未被刻画。我们引入了ContextEcho，一个用于在部署规模下测量人格漂移的基准测试和可复用工具框架。它结合了一个包含25个探针的身份测试套件、一种快照-然后-探针协议，该协议在不干扰主会话的情况下分叉对话状态，以及互补的人工评判和无评判测量方法。

    arXiv:2605.24279v2 Announce Type: replace  Abstract: A frontier language model's acknowledged "helpful programming assistant" persona does not survive long agentic-coding sessions in the deployment regime that production products actually run. After hours of tool-using debugging, a model that initially hedges preferences ("I don't have preferences") may begin asserting them ("Python - the feedback loop is instant..."), revealing user-visible drift that deployer evaluations may miss. Existing persona-stability studies focus on short dialogues and report little shift, leaving real-world code-generation regimes - thousands of tool-using turns, compaction, and hours-long sessions - largely uncharacterized. We introduce ContextEcho, a benchmark and reusable harness for measuring persona drift at deployment scale. It combines a 25-probe identity suite, a snapshot-then-probe protocol that forks conversation state without perturbing the main session, complementary judged and judge-free measure
    
[^59]: MISRust：将MISRA-C++编码规范映射到Rust编程语言

    MISRust: Mapping MISRA-C++ Coding Guidelines to the Rust Programming Language

    [https://arxiv.org/abs/2605.23490](https://arxiv.org/abs/2605.23490)

    本文系统分析了179条MISRA C++ 2023规则对Rust的适用性，发现约48%的直接适用规则被Rust语言设计自动强制执行，并区分了安全与不安全Rust，指出69条规则仍需适配或应用。

    

    摘要：Rust编程语言正越来越多地被考虑用于安全关键系统的开发。然而，既定的安全标准如ISO 26262要求使用编码规范，而Rust尚无此类规范。本文系统性地审查了179条MISRA C++ 2023编码指南，并根据其对Rust的适用性将其分为6类。我们的方法分析了每条MISRA规则背后的原理，以确定其在Rust编程语境中是否仍然有效。我们发现，在111条可直接适用的MISRA规则中，47.75%被Rust的语言设计自动强制执行，从而无需显式规则执行。此外，我们的分析明确区分了安全Rust和不安全Rust。我们发现69条规则仍然相关，需要直接应用或适配到Rust。重要的是，其中36条规则在满足某些条件时会被自动满足。

    arXiv:2605.23490v2 Announce Type: replace  Abstract: The Rust programming language is increasingly being considered for safety-critical system development. However, established safety standards such as ISO 26262 require the use of coding guidelines that do not yet exist for Rust. This paper systematically examines each of the 179 MISRA C++ 2023 coding guidelines and classifies them into 6 categories based on their applicability to Rust. Our approach analyzes the rationale behind each MISRA rule to determine whether it remains valid in the Rust programming context. We find that 47.75% of the 111 as-is applicable MISRA rules are automatically enforced by Rust's language design, eliminating the need for explicit guideline enforcement. Furthermore, our analysis explicitly distinguishes between safe and unsafe Rust. We find that 69 guidelines are still relevant and still require either direct application or adaptation for Rust. Importantly, 36 of these rules are automatically satisfied when
    
[^60]: 代码摘要作为基于LLM的程序修复的诊断上下文

    Code Summaries as Diagnostic Context for LLM-Based Program Repair

    [https://arxiv.org/abs/2511.18782](https://arxiv.org/abs/2511.18782)

    本研究通过仅提示测试发现，代码摘要，特别是诊断性和错误感知的摘要，能作为有用的诊断上下文，平均提升LLM程序修复效果5%，但收益有限且依赖模型，无法显著克服仅提示自我修复的挑战。

    

    arXiv:2511.18782v2 公告类型：替换 摘要：大型语言模型（LLMs）可以生成有用的代码，但其输出往往包含影响行为的小型实现级错误。在本文中，我们探讨自然语言代码摘要是否为修复这些错误提供了有用的诊断上下文。我们使用摘要介导的修复作为其诊断价值的简单仅提示测试：LLM首先总结有缺陷的代码，然后基于该摘要生成候选修复。我们在两个函数级修复设置中评估了摘要介导的修复，涉及八种LLM：来自HumanEvalPack的现有错误和LLM自身失败的MBPP生成。诊断性、错误感知的摘要表现最佳，修复了高达65%的先前未见错误，并平均比直接修复提高了5%。然而，总体收益适中且依赖于模型，摘要对克服仅提示自我修复的困难帮助不大。总体而言，我们的结果表明代码摘要是有用的诊断上下文。

    arXiv:2511.18782v2 Announce Type: replace  Abstract: LLMs can generate useful code, but their outputs often contain small implementation-level bugs with large behavioural effects. In this paper, we ask whether natural-language code summaries provide useful diagnostic context for repairing these errors. We use summary-mediated repair as a simple prompt-only test of their diagnostic value: an LLM first summarises buggy code, then generates a candidate repair conditioned on that summary. We evaluate summary-mediated repair across eight LLMs in two function-level repair settings: existing bugs from HumanEvalPack and LLMs' own failed generations of MBPP. Diagnostic, error-aware summaries perform best, repairing up to 65% of previously unseen errors and improving over direct repair by 5% on average. However, overall gains are modest and model-dependent, and summaries do little to overcome the difficulty of prompt-only self-repair. Overall, our results suggest that code summaries are a useful
    
[^61]: 迷失在代码生成中：重新构想软件模型在AI驱动软件工程中的作用

    Lost in Code Generation: Reimagining the Role of Software Models in AI-driven Software Engineering

    [https://arxiv.org/abs/2511.02475](https://arxiv.org/abs/2511.02475)

    本文主张在AI驱动的软件工程中，软件模型应从前期蓝图转变为可恢复、可精炼的工具，以弥合原型与工程化软件之间的差距，并通过智能体循环和人类反思循环提升系统的健壮性与可维护性。

    

    arXiv:2511.02475v3 公告类型：替换 摘要：生成式AI使得快速的“氛围编码”和智能体软件工程成为可能，其中自然语言提示能生成可工作的软件系统。这降低了软件创建的门槛，但也模糊了原型与工程化软件之间的界限。看似完整的系统可能缺乏健壮性、安全性和可维护性。我们认为，这一转变促使软件模型重新扮演重要角色。模型不再仅仅作为前期的蓝图，而是可以从AI生成的系统中恢复，用于重建理解，并通过精炼来指导后续演进。我们区分了智能体循环（其中恢复的模型支持自动化检查和修复）与人类反思循环（其中模型暴露假设以供检查和修订）。我们识别了候选模型类型，并展示了恢复的模型如何将隐含假设暴露为可能供审查和验证的约束。本文提出了这一观点。

    arXiv:2511.02475v3 Announce Type: replace  Abstract: Generative AI enables rapid "vibe coding" and agentic software engineering, where natural-language prompts yield working software systems. This lowers barriers to software creation, but it also collapses the boundary between prototypes and engineered software. Systems that appear complete may lack robustness, security, and maintainability. We argue that this shift motivates a renewed role for software models. Rather than serving only as upfront blueprints, models can be recovered from AI-generated systems, used to restore comprehension, and refined to guide subsequent evolution. We distinguish an agentic loop, in which recovered models support automated checking and repair, from a human reflective loop, in which models expose assumptions for inspection and revision. We identify candidate model types and illustrate how recovered models can expose implicit assumptions as possible constraints for review and validation. This paper positi
    
[^62]: 基于堆栈跟踪的崩溃去重与Transformer适配

    Stack Trace-Based Crash Deduplication with Transformer Adaptation

    [https://arxiv.org/abs/2508.19449](https://arxiv.org/abs/2508.19449)

    本文提出dedupT，一种基于Transformer的崩溃去重方法，通过整体建模堆栈跟踪并适配预训练语言模型，显著提升了重复崩溃排序和唯一崩溃检测的准确性，减少了人工分类负担。

    

    arXiv:2508.19449v2 公告类型：替换交叉 摘要：自动化崩溃报告系统会生成大量重复报告，使问题跟踪系统不堪重负，并增加开发人员的工作负担。传统的基于堆栈跟踪的去重方法——依赖字符串相似性、基于规则的启发式方法或深度学习（DL）模型——通常无法捕捉堆栈跟踪中的上下文和结构关系。我们提出了dedupT，一种基于Transformer的方法，它将堆栈跟踪整体建模，而非视为孤立的帧。dedupT首先将预训练语言模型（PLM）适配到堆栈跟踪上，然后利用其嵌入来训练一个全连接网络（FCN），以有效排序重复崩溃。在真实世界数据集上的大量实验表明，dedupT在重复排序和唯一崩溃检测方面均优于现有的深度学习和传统方法（例如序列比对和信息检索技术），显著减少了人工分类工作。

    arXiv:2508.19449v2 Announce Type: replace-cross  Abstract: Automated crash reporting systems generate large volumes of duplicate reports, overwhelming issue-tracking systems and increasing developer workload. Traditional stack trace-based deduplication methods---relying on string similarity, rule-based heuristics, or deep learning (DL) models---often fail to capture the contextual and structural relationships within stack traces. We propose dedupT, a transformer-based approach that models stack traces holistically rather than as isolated frames. dedupT first adapts a pretrained language model (PLM) to stack traces, then uses its embeddings to train a fully-connected network (FCN) to rank duplicate crashes effectively. Extensive experiments on real-world datasets show that dedupT outperforms existing DL and traditional methods (e.g., sequence alignment and information retrieval techniques) in both duplicate ranking and unique crash detection, significantly reducing manual triage effort.
    
[^63]: 基于大语言模型的自动程序修复综述：分类体系、设计范式与应用

    A Survey of LLM-based Automated Program Repair: Taxonomies, Design Paradigms, and Applications

    [https://arxiv.org/abs/2506.23749](https://arxiv.org/abs/2506.23749)

    本文提出首个基于大语言模型的自动程序修复可复现层次分类体系，将66个系统按修复能力与控制逻辑的驻留位置分为四类，并支持跨范式设计分析。

    

    arXiv:2506.23749v3 公告类型：替换  摘要：大语言模型（LLMs）正在重塑自动程序修复领域。我们提出了一种可复现的层次化分类体系，该体系根据修复能力和控制逻辑的主要驻留位置，对66个基于LLM的修复系统进行了组织：任务自适应参数、提示与上下文设计、设计者指定的工作流，或LLM指导的运行时控制。适应方式、生成模式、运行时控制和辅助证据被保留为独立的编码维度。这种表示方式揭示了被利用标签所掩盖的控制差异，并支持对系统设计和评估证据进行跨范式分析。据我们所知，这是首个公开可用的基于LLM的软件修复综述，将微调、提示、程序化和智能体作为整个语料库的主要分类进行了操作化。我们的分类体系通过一个语料库范围的主要决策规则补充了先前的综述，而结果级协议则提供了进一步的分析。

    arXiv:2506.23749v3 Announce Type: replace  Abstract: Large language models (LLMs) are reshaping automated program repair. We present a reproducible hierarchical taxonomy that organizes 66 LLM-based repair systems according to where repair capability and control logic principally reside: task-adapted parameters, prompt and context design, designer-specified workflows, or LLM-directed runtime control. Adaptation, generation pattern, runtime control, and auxiliary evidence are preserved as separate coded dimensions. This representation exposes control distinctions hidden by utilization labels and supports cross-paradigm analysis of system design and evaluation evidence. To the best of our knowledge, it is the first publicly available LLM-based software repair survey to operationalize Fine-Tuning, Prompting, Procedural, and Agentic as one corpus-wide primary classification. Our hierarchy complements prior surveys through one corpus-wide primary decision rule, while the result-level protoco
    
[^64]: 生成式人工智能应用面临的挑战与机遇：一项实证研究

    Understanding the Challenges and Opportunities of Generative AI Apps: An Empirical Study

    [https://arxiv.org/abs/2506.16453](https://arxiv.org/abs/2506.16453)

    本研究通过对171个生成式AI应用的百万条用户评论进行分析，提出了SARA框架，识别出用户关注的核心主题（如AI性能与情感连接），并验证了LLM在评论分析中的高可靠性，揭示了Gen-AI应用面临的挑战与机遇。

    

    生成式人工智能（Gen-AI）正日益融入移动应用程序（apps），带来了新功能，同时也为用户创造了新的挑战。然而，尽管其采用率不断增长，我们仍缺乏对用户在各种Gen-AI移动应用中报告的经验、机遇和挑战的生态系统级理解。我们对来自Google Play商店171个Gen-AI应用的1,035,342条评论进行了以用户为中心的分析。我们提出了SARA（选择、获取、精炼和分析）框架，这是一个四阶段框架，利用基于提示的大语言模型（LLM）进行大规模评论分析。我们使用4,353条人工评估的评论验证了基于LLM的主题提取和分配的可靠性，通过五次提示和过滤非信息性评论，实现了91%的准确率。我们识别了十大主题（如AI性能与情感连接），并与Apple App Store评论进行了跨平台比较。

    arXiv:2506.16453v5 Announce Type: replace  Abstract: Generative AI (Gen-AI) is increasingly integrated into mobile applications (apps), introducing new capabilities while also creating new challenges for users. However, despite their growing adoption, we lack an ecosystem-level understanding of the experiences, opportunities, and challenges users report across Gen-AI mobile apps. We conduct a user-centered analysis of 1,035,342 reviews from 171 Gen-AI apps from the Google Play Store. We propose SARA (Selection, Acquisition, Refinement, and Analysis), a four-phase framework that leverages prompt-based LLMs for large-scale review analysis. We validate the reliability of LLM-based topic extraction and assignment using 4,353 manually evaluated reviews, achieving 91% accuracy with five-shot prompting and filtering of non-informative reviews. We identify the top ten topics (e.g., AI Performance and Emotional Connection) and perform a cross-platform comparison with Apple App Store reviews. Th
    
[^65]: HybridProver：通过LLM驱动的证明合成与精炼增强定理证明

    HybridProver: Augmenting Theorem Proving with LLM-Driven Proof Synthesis and Refinement

    [https://arxiv.org/abs/2505.15740](https://arxiv.org/abs/2505.15740)

    HybridProver提出了一种统一框架，通过证明草图作为中间表示，整合了整体证明合成与逐步策略生成，从而在定理证明中结合高层规划与细粒度推理，实现了部分正确证明结构的重用。

    

    arXiv:2505.15740v2 公告类型：替换-交叉 摘要：形式化方法通过严格的数学验证在确保关键系统可靠性方面发挥着至关重要的作用。然而，由于手动证明构建的劳动密集型特性，其采用仍然有限。大型语言模型（LLMs）的最新进展为自动定理证明开辟了新的机遇。主要出现了两种范式：逐步基于策略的生成和整体证明合成。虽然这两种方法具有互补优势，但现有工作大多将它们孤立对待。在本工作中，我们提出了HybridProver，一个统一框架，通过证明草图作为中间表示来整合整体证明合成和基于策略的生成。这种设计使得能够重用部分正确的证明结构，同时有效结合高层规划与细粒度推理。我们在Isabelle/HOL中实现了HybridProver，并在我们的数据集上对两个7B规模的LLM进行了后训练。

    arXiv:2505.15740v2 Announce Type: replace-cross  Abstract: Formal methods play a crucial role in ensuring the reliability of critical systems through rigorous mathematical verification. However, their adoption remains limited due to the labor-intensive nature of manual proof construction. Recent advances in large language models (LLMs) have opened new opportunities for automated theorem proving. Two main paradigms have emerged: stepwise tactic-based generation and whole-proof synthesis. While both approaches have complementary strengths, existing work largely treats them in isolation. In this work, we propose HybridProver, a unified framework that integrates whole-proof synthesis and tactic-based generation through proof sketches as an intermediate representation. This design enables the reuse of partially correct proof structures while effectively combining high-level planning with fine-grained reasoning. We implement HybridProver in Isabelle/HOL and post-train two 7B-scale LLMs on ou
    

