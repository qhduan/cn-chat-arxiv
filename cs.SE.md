# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cognitive Tours: Making Understanding a Deliverable in AI-Assisted Development](https://arxiv.org/abs/2611.08456) | 提出了“认知导览”工具，通过构建连接用户需求、设计决策、代码、测试与开放问题的策划路径，并借助AI辅助审计与人工决策，来缓解和修复AI辅助开发中的认知债务与意图债务。 |
| [^2] | [SoK: Post-Quantum Cryptography (PQC) Integration in Software Development: Approaches, Gaps, and Future Directions](https://arxiv.org/abs/2611.08442) | 本文系统梳理了软件开发中后量子密码学集成的研究现状，指出尽管现有工作奠定了重要的技术与组织基础，但各阶段（准备、实现、迁移、部署、维护及开发者实践）之间缺乏协调，生命周期支持仍是割裂的。 |
| [^3] | [Hard Issue, Weak Model, or Bad Run? Rethinking Failure Prediction for Coding Agents](https://arxiv.org/abs/2611.08285) | 该研究发现在 SWE-bench Verified 上，轨迹失败预测器的性能主要由问题难度和模型能力这两个与运行无关的因素决定，其从轨迹本身获得的信号在统计上几乎为零，从而质疑了此类预测器真正学到运行相关信息的假设。 |
| [^4] | [Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations](https://arxiv.org/abs/2611.08269) | 预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。 |
| [^5] | [QDiff: Differential Testing of Quantum Software Stacks](https://arxiv.org/abs/2611.08220) | 本论文提出了QDiff，一种针对量子软件栈的新型差分测试方法，通过保语义的源到源转换生成测试输入程序，以检测不同量子软件栈之间的错误与不一致性。 |
| [^6] | [RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection](https://arxiv.org/abs/2611.08162) | RouteAlign通过将任务需求（能力需求与难度）与模型性能画像对齐，并利用共享评分器和生成成本偏好进行选择，从而在异构大语言模型服务中实现更精准的服务选择。 |
| [^7] | [Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning](https://arxiv.org/abs/2611.08144) | 该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。 |
| [^8] | [Symmetry-Aware Surrogate-Assisted Testing of UAVs](https://arxiv.org/abs/2611.08012) | 提出一种对称性感知的代理辅助测试方法，通过将障碍物配置转换到统一规范坐标系，使基于单一参考任务训练的神经网络代理模型可复用于任意无人机飞行任务，从而以更少的仿真数据生成安全关键测试用例，临界测试用例命中率达42%，远超现有竞赛技术的约10%。 |
| [^9] | [Requirements Nobody Wrote Down: Emergence in a Public-Sector ML Pipeline](https://arxiv.org/abs/2611.07951) | 通过对巴西政府数字参与平台ML流水线为期15个月的案例研究，本文揭示了公共部门机器学习系统的需求并非预先明确获取，而是主要从实验过程中涌现，且大多数需求从未被正式记录。 |
| [^10] | [tok: a textual concrete syntax of BPMN and DMN for humans and AI agents](https://arxiv.org/abs/2611.07945) | 本文提出了tok这一BPMN和DMN的纯文本表示法，其令牌效率比XML高数倍、故障检测率更高，且语言模型能轻松学习并生成有效的tok模型。 |
| [^11] | [A Low Similarity Score Does Not Show Independence after Source-Conditioned Repository Rewriting](https://arxiv.org/abs/2611.07887) | 研究表明，大语言模型智能体能够低成本地将著作权留项目重写为通过全部测试且相似度很低的版本，而相同配置下多次运行的相似度波动很大，因此低相似度分数不能作为重写独立于原作的证据。 |
| [^12] | [Compositional Loop Invariant Inference for Multiphase Programs via Structured Neurosymbolic Reasoning](https://arxiv.org/abs/2611.07882) | 本文提出了CLOVER框架，通过将多阶段循环分解为符号化的分支序列、为各分支独立生成候选子不变式并进行针对性修复，实现组合式的循环不变式自动推断。 |
| [^13] | [Which Signals of Code Correctness Survive Distribution Shift? Hidden-State and Causal Predictors of Program and Verdict Correctness](https://arxiv.org/abs/2611.07876) | 本文系统比较了四类共六种免执行代码正确性预测器在基准与语言分布偏移下的稳健性，并将两种新颖的因果定位方法引入代码领域，同时评估其对程序正确性和模型自身判定可信度的预测能力。 |
| [^14] | [Adapt the Object to the Subject: Representative Study Material for Empirical Software Engineering](https://arxiv.org/abs/2611.07772) | 该论文首次关注实证软件工程中“研究对象与被试自身工作不匹配”这一被忽视的外部效度问题，并提出一种利用LLM在在线调查中根据参与者简短上下文描述即时生成适配语境的合成需求（可注入需求坏味作为受控刺激）的实验方法。 |
| [^15] | [Fast Unsupervised Anomaly Detection Methods on Software Logs - An Empirical Study](https://arxiv.org/abs/2611.07760) | 该研究通过对四种无监督异常检测模型、三种日志表示方式及多种预处理策略在七个真实数据集上进行大规模组合实证评估，为软件日志快速无监督异常检测的方案选择提供了实证依据。 |
| [^16] | [ReConMem: A Graph-Based Dual Memory System for Repository-Oriented Conversations](https://arxiv.org/abs/2611.07759) | 提出ReConMem，一种基于图的双记忆系统，将对话记忆与代码实体分别组织为相互关联的语义图和结构图，并通过结合自然语言与代码对齐的检索机制，实现面向仓库级开发对话的高效记忆召回。 |
| [^17] | [Uncovering Systematic Blind Spots in Graph Database Testing through Integration and Process-Level Analysis](https://arxiv.org/abs/2611.07754) | 该研究通过集成8种图数据库自动化测试方法并进行过程级实证分析，揭示了图数据库测试中存在大量由方言和模式匹配差异导致的误报缺陷报告等系统性盲点。 |
| [^18] | [Fix One, Miss Another: Leveraging Historical Bug Fixes to Uncover Residual Bugs in Database Systems](https://arxiv.org/abs/2611.07739) | 该论文提出Reaper方法，通过分析历史错误修复来推断触发条件与验证逻辑，从而系统性地发现数据库系统中同类处理器里残留的、与历史错误同源的未修复缺陷（同类错误）。 |
| [^19] | [Don't Prompt Your Agents. Equip Them: Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems](https://arxiv.org/abs/2611.07652) | 该论文提出Skillware开放框架，通过模块化、版本化的软件能力包（包含类型化JSON契约、确定性代码、任务指令、数据集和单元测试）取代脆弱的提示工程，将智能体的意图与执行正式分离，实现主权、可审计且确定性的能力执行。 |
| [^20] | [What the Middleman Delivers: Answer Behavior, Tool Permissions, and Billed Usage in LLM API Relays](https://arxiv.org/abs/2611.07650) | 本研究对20个提供GPT-6 Astra和Claude Opus 5的第三方LLM API中转服务进行系统性测试，发现约半数低价中转服务的回答行为与官方模型存在差异，并通过检查返回内容、工具权限与计费用量揭示了中转服务在交付质量上的不可靠性。 |
| [^21] | [AiXapply: Fast and Reliable Full-File Code Integration with Specialized Small Models for IDE Workflows](https://arxiv.org/abs/2611.07634) | 该论文将代码集成任务形式化为根据原始文件与更新片段生成完整合并文件的问题，构建了基于真实提交的多语言数据集，并通过监督微调与GRPO强化学习训练出专用4B模型aiXapply，以94.4%的准确率超越大模型基线，为IDE工作流提供快速可靠的全文件代码集成。 |
| [^22] | [Evolver: Turning Agent Exploration into Evolving Capabilities](https://arxiv.org/abs/2611.07595) | Evolver 是一个无需更新基础模型参数的经验驱动框架，它通过结构化的 Gene 资产将智能体探索产生的经验转化为持久可复用的能力，在四个基准测试中显著提升任务成功率并将执行 token 消耗降低约 24%。 |
| [^23] | [Route Before Response: Fail-Closed Routing for LLM-Enabled Customer-Service Software](https://arxiv.org/abs/2611.07534) | RouteBR提出“先路由后响应”的故障关闭式控制架构，将一次模型调用与确定性代码校验相结合，在真实金融机构的200个场景上实现97.5%最终路径精确匹配，并确保被拦截的主路由不会被低排名候选替代。 |
| [^24] | [The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks](https://arxiv.org/abs/2611.07461) | 提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。 |
| [^25] | [Req2Rust: A Requirements-Guided Agent Harness for Project-Level C++-to-Rust Migration](https://arxiv.org/abs/2611.07442) | Req2Rust是首个项目级基于大语言模型的C++到Rust迁移框架，通过先从源代码中推导出有证据支撑的迁移需求规范（MRS）来引导智能体完成迁移，在语义保真与符合Rust习惯的项目重构之间取得平衡。 |
| [^26] | [Rule-Guided and Trajectory-Aware Repair for LLM-Based HLS Generation](https://arxiv.org/abs/2611.07324) | MementoHLS框架通过可执行修复契约库（将跨任务HLS知识转化为证据驱动的修复动作）与证伪账本（将修复历史转化为轨迹证据以抑制无效路径）的双记忆引导机制，并借助同一候选评审者实现可靠的结果归因，显著提升了大语言模型从自然语言规范生成可综合HLS-C代码的端到端成功率。 |
| [^27] | [Take It From Here: Evaluating Coding Agents On Project Takeover](https://arxiv.org/abs/2611.07307) | 该论文提出了评估编程智能体项目接管能力的新基准，发现即使提供完整交互历史，智能体的平均完整交付率也仅为4.95%，揭示了当前智能体在从历史中恢复需求并完成项目交付方面的严重不足。 |
| [^28] | [Code, Commits, and Complexity in the GenAI Era: A Longitudinal Study of Team-Based Capstone Projects](https://arxiv.org/abs/2611.07297) | 该研究对2021-2025年间260个毕业设计团队代码仓库进行了十个学期的纵向分析，发现生成式AI工具出现后学生团队项目代码库规模增长约三倍，揭示了真实项目环境中学生开发行为与代码特征的显著变化。 |
| [^29] | [How Should Agents See Code? Rethinking Code Representations for Agentic Software Engineering](https://arxiv.org/abs/2611.07286) | 本文将基于地图的代码表示（抽象语法树、调用图、模块依赖、文档增强搜索及其混合形式）作为智能体软件工程的设计变量，系统评估了五种代码表示工作流如何影响智能体的检索行为、任务成功率、token消耗与延迟。 |
| [^30] | [SeVerus: Context-Aware, Gamified Benchmarking for Rust Verification](https://arxiv.org/abs/2611.07194) | 本文提出 SeVerus，一种上下文感知、游戏化的 Verus 基准测试生成技术，它利用已验证项目的原始证明依赖关系，通过逐步移除证明上下文将验证任务组织成从易到难的任务链，从而更真实地评估编码智能体自主发现并证明辅助引理的能力。 |
| [^31] | [Global Surveillance Under the Lens: Reconstructing Passive Dolev-Yao Intruders](https://arxiv.org/abs/2611.07170) | 本文提出一种自动化的规范指导方法，能够从PCAP捕获的网络流量中重构被动Dolev-Yao入侵者的执行过程，并依据AnBx形式化协议规范进行验证，同时解决了TCP流中对象重构、并发会话映射歧义和语义验证三大挑战。 |
| [^32] | [Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons?](https://arxiv.org/abs/2611.07166) | 提出长时程基准 Multi-step SWE，包含69个由5-15个依赖步骤组成的软件工程任务链，要求智能体在自身编写的代码上持续迭代开发，最强前沿模型的解决率仅18.5%，揭示了当前编程智能体在长时程自我延续开发中的严重不足。 |
| [^33] | [Automatic Assistance to Reduce Information Overload during Software Issue Report Understanding](https://arxiv.org/abs/2611.07109) | 本文提出了iSum工具，能够自动分类和总结软件问题报告中的信息类型，通过开发者视图为单个报告生成可视化摘要、通过经理视图展示项目内各报告的信息类型普遍程度与趋势，帮助开发者缓解问题报告带来的信息过载。 |
| [^34] | [Follow the Policy, Fail the Benchmark: Auditing and Revising Policy-Based Decision Benchmarks](https://arxiv.org/abs/2611.07042) | 论文审计了九个基于政策的决策基准，发现其中存在参考错误和评分缺陷，仅通过修正这些基准问题即可使模型接受率提升超过80个百分点，表明基准质量对模型评估结果有决定性影响。 |
| [^35] | [An LLM-Orchestrated Neuro-Symbolic Framework for Verifying Diagnostic Signal Reachability in Electronic Systems](https://arxiv.org/abs/2611.07040) | 该论文提出NetFlow-DSR，一个由大语言模型调度并结合符号验证的神经符号框架，用于自动化验证电子系统中诊断信号的可达性，克服了传统人工逐信号追踪效率低下以及LLM在结构连接性推理上不可靠的问题。 |
| [^36] | [Call-Level Revision Contracts for LLM-Authored Workflows](https://arxiv.org/abs/2611.06967) | 该论文为LLM编写的工作流提出了调用级修订契约，通过为每个调用赋予稳定的结构化标识以构成冻结前缀，并配合静态门控、运行时摘要和出现账本三重机制，解决了模型修复失败工作流时重复付费或接受已变更负载的难题。 |
| [^37] | [When Can Self-Play Fix a False Belief? Answer Keys, Pass Bands and Replacement Rules in an Execution-Grounded Loop](https://arxiv.org/abs/2611.06965) | 在以执行为依据的自我博弈循环中，执行自身代码得到的答案密钥只会复制并固化模型的错误信念，只有当答案密钥来自不含该错误的“教师”实现、且通过区间能够接纳纠正性任务时，自我博弈才能真正修复被植入的错误信念。 |
| [^38] | [Operationalizing Responsible AI in MLOps and ML Production: A Systematic Literature Review](https://arxiv.org/abs/2611.06961) | 本文通过系统性文献综述发现，负责任AI在MLOps和ML生产中正从抽象伦理原则转变为具体的工程机制，通过漂移监测、运行时检查、审计日志、治理门和策略即代码等技术手段落地实施。 |
| [^39] | [Recursively Testing and Vibe Debugging a Google AI Studio Generated Vibe-Coded Database Software Application](https://arxiv.org/abs/2611.06952) | 该论文提出了扩展的氛围编程定义与氛围调试概念，并给出一种可靠、完备且最优的算法方法论，证明了即使是使用非英语语言的非开发者，也能通过递归测试与氛围调试将Google AI Studio生成的软件应用迭代至无错误且功能满意的版本。 |
| [^40] | [KG-AST for Cheaper CI/CD: Budgeted Test Selection and AI Bug Review](https://arxiv.org/abs/2611.06950) | KG-AST利用git历史和代码图特征对提交进行风险评分，优先将测试与AI审查预算分配给最可能引入缺陷的提交，在11种语言的155个开源项目上显著优于随机选择和最大变更优先基线。 |
| [^41] | [When Does Combining Software-Effort Estimates Help? Evidence from Four Public Project Datasets](https://arxiv.org/abs/2611.06948) | 本文在四个公开软件工作量数据集上系统比较五种组合规则后发现，组合估计并非总是有益——当某一估计（尤其是人工估计）异常准确或各成员误差高度相关时，集成反而会稀释准确性或重复共同误差。 |
| [^42] | [How Python Audio Libraries Change Across Releases: Module Structure and Documentation](https://arxiv.org/abs/2611.06947) | 该研究首次将Python音频库的模块结构演化与发布文档进行联合挖掘分析，通过对49个软件包1,395个发布版本的导入图及20,864条语言模型标注的发布说明的大规模实证研究发现：顶级模块结构高度稳定（86.1%的连续发布中不变）、主版本号递增引发结构变化的频率是其他递增的5.2倍，但发布说明仅覆盖约8.9%的模块移除事件。 |
| [^43] | [Beyond the Agent Harness: Boundary Criteria for Software-Factory Orchestration - A Single-Operator Case Study](https://arxiv.org/abs/2611.06946) | 该论文的核心贡献是提出了一套基于“运行”操作性定义的边界判据，用以明确判定软件工厂编排中的机制与故障究竟归属于智能体框架、工厂框架、领域层还是基础设施/模型层。 |
| [^44] | [A Multi-Modal Robotic Framework for Physical GUI Interaction Testing](https://arxiv.org/abs/2611.06939) | 该论文提出了RFNIT，一个多模态、完全非侵入式的机器人框架，通过视觉感知模块、LLM推理代理和机器人执行器，实现了无需系统级权限的智能手机GUI物理交互端到端自动化测试。 |
| [^45] | [Replay Is a Family of Assertions: Fork-Safety in Event-Sourced Agent Runtimes](https://arxiv.org/abs/2611.06924) | 本文提出一个将事实与效果请求分离、使调度显式化的可执行 F# 契约，证明事件溯源式智能体运行时中的确定性重放、低成本分叉与谱系是三个相互独立的断言而非单一保证，并通过反例与大规模日志验证揭示了一般性结算既不终止也不合流。 |
| [^46] | [Towards Comprehensive Mobile Application Testing for User-centric Feature Coverage via Simulating Real-World Users with Personas](https://arxiv.org/abs/2611.06922) | 该论文提出了由3,012项用户中心功能数据集和自动化评估流水线构成的“以用户为中心功能覆盖”基准，并通过人物角色模拟真实用户，构建了比传统活动覆盖率更能反映用户真实需求的全面移动应用测试框架。 |
| [^47] | [TimeArch: A Multi-Agent Approach for Software Architecture Design](https://arxiv.org/abs/2611.06921) | TimeArch提出了一种多智能体方法，将软件架构设计结构化为受治理的生命周期，从而在GenAI编码工作流中显式填补需求与代码之间缺失的架构层。 |
| [^48] | [LIBRA: Programmatic Induction of Executable Tool Libraries](https://arxiv.org/abs/2611.05445) | LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。 |
| [^49] | [The Physical Cost of Context: Quality-Constrained Resource Accounting for Coding Agents](https://arxiv.org/abs/2610.34763) | 该论文提出了一种质量门控的资源核算协议，指出编码代理上下文策略的真实成本取决于调用次数和前缀缓存复用而非仅仅提示词长度，并证明预加载检索块（Eager）策略在保证代码质量的前提下可减少35.7%的模型调用和33.2%的GPU能耗。 |
| [^50] | [RetryBench: Separating Retry Permission from Recovery Outcomes in GPT Tool Use](https://arxiv.org/abs/2610.15884) | 该论文提出RetryBench基准，将GPT工具使用中“虚假重试许可”（即在契约不支持时重试超时工具调用）这一错误与实际恢复结果分离，发现即使在受保护条件下能成功恢复，模型也无法可靠地拒绝不受支持的重试。 |
| [^51] | [Executable but Unfaithful: Mock Object, Stub, and Assertion Generation by LLMs and Coding Agents](https://arxiv.org/abs/2610.14629) | 该论文提出了首个系统性基准MockBench（含165个来自真实Java与Python项目的任务），全面评估了大语言模型与编码智能体生成模拟对象、桩和模拟断言三类制品的能力，并发现其生成的代码虽可执行但往往与真实依赖行为不一致。 |
| [^52] | [Refactoring React Component Hierarchies to Eliminate Prop Drilling](https://arxiv.org/abs/2610.01376) | 本文提出一种静态分析方法并实现为命令行工具 ReactRefactor，可自动识别 React 代码库中的属性钻取问题，并利用基于 Context API 和组件组合的两种自动化重构策略将其消除，实证评估表明属性钻取在各规模的开源项目中普遍存在。 |
| [^53] | [SaltBench: A Referee-Gated Protocol for Measuring Method Effects in Machine-Checked Software Work](https://arxiv.org/abs/2609.11076) | 提出了SaltBench，一个由机器裁判门控的基准测试协议，通过严格的隔离机制（经探针实际检验）、预注册预测和“停止即暂停”等规则，使机器裁判对编程智能体工作方式的影响变得可测量且不可事后叙述。 |
| [^54] | [Vibe Coding in Software Development: A Multivocal Literature Review](https://arxiv.org/abs/2607.21652) | 这是首个在统一协议下对“氛围编程”进行的多声部文献综述，综合分析了266个学术与从业者来源（76个同行评审、190个灰色文献），围绕八个研究问题系统梳理了这一AI辅助开发方法的现有证据。 |
| [^55] | [Auditing Empirical Comparisons in Quantum Software](https://arxiv.org/abs/2607.00516) | 提出了CLAIMSTAB-QC审计框架，通过在计算结果前锁定审计设计、限定证据来源边界，将量子软件论文中的455条实证比较声明分类为“维持”、“未决”或“未保留”，从而检验比较结论是否有证据支撑。 |
| [^56] | [Large Language Models for Multi-Lingual Equivalent Mutant Detection: An Extended Empirical Study](https://arxiv.org/abs/2607.00511) | 本文首个针对大语言模型在多语言（Java与C）等价变异体检测上的全面实证研究，利用3,302个Java和1,088个C变异体对进行基准评估，验证了LLM克服传统纯代码分析和机器学习方法局限的潜力。 |
| [^57] | [Characterizing Robustness in Semantic Code Clone Detection](https://arxiv.org/abs/2606.25272) | 该论文提出系统化代码转换（SCT）框架，通过51个按代码层级、目标和编辑类别组织的转换算子，系统评估了14种语义代码克隆检测器对行为保持型变更的鲁棒性，发现25个高性能检测器与数据集组合中有21个在全算子组合下F1分数至少下降10个百分点。 |
| [^58] | [Are LLM Safety Judges Policy-Invariant? A Three-Principle Stress-Test](https://arxiv.org/abs/2605.06161) | 本文提出“策略不变性”这一安全评判者必须满足的基本属性，并将其操作化为三条可检验原则构成压力测试协议，揭示了现有LLM安全评判者的判定会同时受有意义规范变化和无意义措辞改写影响这一此前未被测量的失效模式。 |
| [^59] | [AI Slop and the Software Commons](https://arxiv.org/abs/2604.16754) | 本文论证 AI 生成的低质量代码正在引发软件开发的“公地悲剧”，并借鉴 Ostrom 的公共资源治理设计原则，为软件社区各类角色提出可持续治理垃圾内容的具体方案。 |

# 详细

[^1]: 认知导览：在AI辅助开发中让“理解”成为可交付成果

    Cognitive Tours: Making Understanding a Deliverable in AI-Assisted Development

    [https://arxiv.org/abs/2611.08456](https://arxiv.org/abs/2611.08456)

    提出了“认知导览”工具，通过构建连接用户需求、设计决策、代码、测试与开放问题的策划路径，并借助AI辅助审计与人工决策，来缓解和修复AI辅助开发中的认知债务与意图债务。

    

    生成式AI和智能体编程正在加速软件开发，同时也引发了对技术债务的担忧。然而，两种不太显眼的债务形式——认知债务和意图债务——正在成为影响软件健康的关键风险。认知债务是指团队共享理解的缺失或侵蚀，即没有人能够自信地解释系统如何运作，或预测某项变更会带来什么影响。意图债务是指缺乏清晰阐述的目标、约束和设计理由，而这些内容正是解释系统用途、指导系统演化的依据。本文提出了“认知导览”，这是一种旨在缓解和修复认知债务与意图债务的工具。认知导览是一个基于Web的原型，用于创建和遵循精心策划的路径，将用户需求、设计决策、代码、测试和开放性问题相互连接起来。AI辅助技能帮助审计现有的路标点和导览路径，发现潜在问题或过时内容，并推荐新增内容的候选方案；而最终由人来决定取舍。

    arXiv:2611.08456v1 Announce Type: new  Abstract: Generative AI and agentic coding are accelerating software development, leading to concerns about technical debt. But two less visible forms of debt, cognitive debt and intent debt, are emerging as key risks to software health. Cognitive debt is the absence or erosion of a team's shared understanding where no one can confidently explain how a system works or predict the impact of a change. Intent debt is the lack of clearly articulated goals, constraints, and rationale that explain what the system is for and guide how it should evolve. This paper presents Cognitive Tours, a tool designed to mitigate and repair cognitive and intent debt. Cognitive Tours is a web-based prototype for creating and following curated paths connecting user needs, design decisions, code, tests, and open questions. AI-assisted skills help audit existing waypoints and tours for possible problems or staleness and suggest candidates for new ones; humans decide what 
    
[^2]: SoK：后量子密码学（PQC）在软件开发中的集成：方法、差距与未来方向

    SoK: Post-Quantum Cryptography (PQC) Integration in Software Development: Approaches, Gaps, and Future Directions

    [https://arxiv.org/abs/2611.08442](https://arxiv.org/abs/2611.08442)

    本文系统梳理了软件开发中后量子密码学集成的研究现状，指出尽管现有工作奠定了重要的技术与组织基础，但各阶段（准备、实现、迁移、部署、维护及开发者实践）之间缺乏协调，生命周期支持仍是割裂的。

    

    标准化的后量子密码学（PQC）对于保护软件系统抵御未来的量子威胁至关重要，但仅有标准化算法并不能为其安全集成到软件中提供路径。本文对软件开发中的PQC集成进行了知识系统化（Systematisation of Knowledge, SoK）研究，综合梳理了现有在实现、迁移和开发者教育等方面的方法。我们依据NIST PQC标准与PQC迁移指南、软件开发生命周期覆盖范围、网络安全实践、部署成熟度以及测试与评估证据，对这些方法进行了分析。我们的分析表明，现有工作已奠定了重要的技术和组织基础，但在整个软件开发生命周期中所提供的支持是彼此割裂的，在准备、实现、迁移、部署、维护与开发者实践之间缺乏协调。评估证据同样……

    arXiv:2611.08442v1 Announce Type: new  Abstract: Standardised post-quantum cryptography (PQC) is critical for protecting software systems against future quantum threats, but standardised algorithms alone do not provide a pathway for their secure integration into software. This paper presents a Systematisation of Knowledge (SoK) of PQC integration in software development, synthesising existing approaches across implementation, migration, and developer education. We analyse these approaches against NIST PQC standards and PQC migration guidance, software development lifecycle coverage, cybersecurity practices, deployment maturity, and testing and evaluation evidence. Our analysis shows that existing work establishes important technical and organisational foundations but provides disconnected support across the software development lifecycle, with limited coordination between preparation, implementation, migration, deployment, maintenance, and developer practices. Evaluation evidence also 
    
[^3]: 难题、弱模型，还是糟糕的一次运行？重新思考编码智能体的失败预测

    Hard Issue, Weak Model, or Bad Run? Rethinking Failure Prediction for Coding Agents

    [https://arxiv.org/abs/2611.08285](https://arxiv.org/abs/2611.08285)

    该研究发现在 SWE-bench Verified 上，轨迹失败预测器的性能主要由问题难度和模型能力这两个与运行无关的因素决定，其从轨迹本身获得的信号在统计上几乎为零，从而质疑了此类预测器真正学到运行相关信息的假设。

    

    针对软件工程（SE）智能体的基于轨迹的失败预测器会读取一次运行的部分轨迹，以判断该运行是朝着正确还是错误的方向发展。如果运行出现问题，可以停止、重试该运行，或切换到另一个模型。我们探究这些预测器是否真正从运行本身学到了任何东西。在公开的 SWE-bench Verified 轨迹数据上（其中每个模型对每个问题只尝试一次），两个与运行无关的因素主导了预测器的性能。第一个是问题难度：一个从不查看运行内容、仅参考该问题对其他模型有多难的基准分数，达到了 0.92 的 AUROC，而轨迹预测器只能达到约 0.62。第二个是模型能力：在同一个问题内，仅知道是哪个模型在运行就比我们的预测器更能预测失败，而且这些预测器在很大程度上只是复现了模型的整体能力排名。除了难度和能力之外，预测器从轨迹中获得的信息在统计上等同于零……

    arXiv:2611.08285v1 Announce Type: new  Abstract: Trajectory-based failure predictors for software engineering (SE) agents read the partial trajectory of a run to flag whether it is going right or wrong. If it is going wrong, the run can be stopped, retried, or switched to another model. We ask whether these predictors learn anything about the run itself. On public SWE-bench Verified trajectories, where each model attempts each issue once, two run-independent factors dominate predictor performance. The first is issue difficulty: a reference score that never looks at the run, only at how hard the issue is for other models, reaches 0.92 AUROC, whereas trajectory predictors reach about 0.62. The second is model ability: within an issue, knowing which model ran predicts failure better than our predictors do, and the predictors largely reproduce the overall ability ranking of the models. Beyond difficulty and ability, what predictors gain from the trajectory is statistically equivalent to ze
    
[^4]: 被接受但实则错误：API迁移后LLM代码修复的静默失败

    Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations

    [https://arxiv.org/abs/2611.08269](https://arxiv.org/abs/2611.08269)

    预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。

    

    在运行时或API迁移之后，经LLM修复的程序可能顺利运行却返回错误结果，而基于典型测试输入的发布检查会将其放行发布。我们报告了一项预注册研究，包含73个任务，覆盖CPython、Node.js和SQLite中60个有文档记录的变更。每个候选修复都在真实运行时上针对未公开的契约执行。在整个修复空间中，基于典型输入的全断言门检查放行了0.410比例的错误修复。一个根据发布说明编写的边缘输入将该比例降至0.065，在15个未见过的集群上降至0.061。来自一个从未见过该程序的盲测GPT-5.6-luna编写者的三个输入达到了0.070。两个后续冻结阶段将候选菜单扩大至1491个片段，随后又添加了233个在典型输入上与遗留输出匹配的困难干扰项。边缘输入（0.058）和前沿模型编写者（0.058）通过了其预注册测试。然而，仍有0.168和0.153的困难错误程序通过了检查。

    arXiv:2611.08269v1 Announce Type: cross  Abstract: After a runtime or API migration, an LLM-repaired procedure can run cleanly yet return wrong results, and a release check on typical test inputs ships it. We report a preregistered study of 73 tasks covering 60 documented changes in CPython, Node.js and SQLite. Every candidate is executed on the real runtimes against withheld contracts. Over the complete repair space, the all-assertions gate on typical inputs released 0.410 of wrong repairs. One edge input written from the release note cut this to 0.065, and to 0.061 on 15 unseen clusters. Three inputs from a blind GPT-5.6-luna writer that never saw the program reached 0.070. Two further frozen stages enlarged the menu to 1491 fragments and then added 233 hard distractors that match the legacy output on typical inputs. The edge input (0.058) and the frontier-model writer (0.058) passed their preregistered tests. Yet 0.168 and 0.153 of the hard wrong programs still passed. Two local-mod
    
[^5]: QDiff：量子软件栈的差分测试

    QDiff: Differential Testing of Quantum Software Stacks

    [https://arxiv.org/abs/2611.08220](https://arxiv.org/abs/2611.08220)

    本论文提出了QDiff，一种针对量子软件栈的新型差分测试方法，通过保语义的源到源转换生成测试输入程序，以检测不同量子软件栈之间的错误与不一致性。

    

    在过去几年中，为应对量子计算硬件的快速发展，出现了多种量子软件栈。一个量子软件栈包括一种量子编程语言、一个将用高级语言编写的量子算法翻译成量子门指令的优化编译器、一个在经典设备上模拟这些指令的量子模拟器，以及一个基于量子电路向极其昂贵的量子硬件发送模拟信号的软件控制器。与传统的编译器和体系结构模拟器相比，由于测试结果的概率性质、缺乏明确的硬件规范以及量子编程的复杂性，量子软件栈难以进行测试。本工作设计了一种新颖的量子软件栈差分测试方法，命名为QDIFF，具有三项主要创新：（1）我们通过保语义的源到源转换生成待测试的输入程序，以探索（摘要在此处截断）

    arXiv:2611.08220v1 Announce Type: cross  Abstract: Over the past few years, several quantum software stacks (QSS) have been developed in response to rapid hardware advances in quantum computing. A QSS includes a quantum programming language, an optimizing compiler that translates a quantum algorithm written in a high-level language into quantum gate instructions, a quantum simulator that emulates these instructions on a classical device, and a software controller that sends analog signals to a very expensive quantum hardware based on quantum circuits. In comparison to traditional compilers and architecture simulators, QSSes are difficult to tests due to the probabilistic nature of results, the lack of clear hardware specifications, and quantum programming complexity.   This work devises a novel differential testing approach for QSSes, named QDIFF with three major innovations: (1) We generate input programs to be tested via semantics-preserving, source to source transformation to explor
    
[^6]: RouteAlign：将任务需求与模型画像对齐以实现大语言模型服务选择

    RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection

    [https://arxiv.org/abs/2611.08162](https://arxiv.org/abs/2611.08162)

    RouteAlign通过将任务需求（能力需求与难度）与模型性能画像对齐，并利用共享评分器和生成成本偏好进行选择，从而在异构大语言模型服务中实现更精准的服务选择。

    

    大语言模型服务选择需要能够反映当前任务需求的性能证据。聚合分数掩盖了能力与难度上的差异，而语义上相似的示例可能涉及不同的技能组合。我们提出了RouteAlign，它利用任务需求来同时构建模型性能度量与服务选择的结构。任务需求描述了能力需求和难度，从而能够与相应的模型性能度量实现对齐。历史需求决定了结果如何贡献于全局画像与局部能力估计；新到来的需求则决定了这些度量如何为当前决策提供依据。一个共享的评分器整合了这些证据，而生成成本偏好则控制在符合条件的服务中进行选择。在多样化任务和异构大语言模型服务上的评估表明，RouteAlign扩展了经验性的质量-成本权衡边界。

    arXiv:2611.08162v1 Announce Type: cross  Abstract: LLM service selection requires performance evidence that reflects the demands of the current task. Aggregate scores obscure differences in capability and difficulty, while semantically related examples can involve different combinations of skills. We present RouteAlign, which uses task requirements to structure both the measurement of model performance and the selection of services. Task requirements describe capability demand and difficulty, enabling alignment with corresponding model performance measurements. Historical requirements determine how outcomes contribute to global profiles and local capability estimates; incoming requirements determine how those measurements inform the current decision. A shared scorer integrates the evidence, and a generation-cost preference controls selection among eligible services. Evaluation across diverse tasks and heterogeneous LLM services shows that RouteAlign extends the empirical quality-cost f
    
[^7]: 强化学习中环境-训练器边界处的静默故障检测

    Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning

    [https://arxiv.org/abs/2611.08144](https://arxiv.org/abs/2611.08144)

    该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。

    

    机器人强化学习越来越多地在训练器之外运行环境：在另一个进程中、在另一台机器上、使用另一种编程语言。这种边界上的许多故障不会引发任何错误报告。我们将该边界建模为一种线协议，并探究是什么能够察觉到其中发生的故障。在 Pendulum、HalfCheetah、Hopper 和一个 7 关节机械臂上，我们向跨越线路传输的数据中注入了多达 23 种故障，并向训练器的日志中注入了 2 种故障，范围从单个值出错到全部值出错。除非每个值都是错误的，否则基于学习曲线阈值的检查最多只能发现十分之一的故障运行，与十个无故障种子对比的检查最多发现 21%。对所有 1,060 个最终策略的干净评估与曲线检查结果一致：在 728 个边界故障运行中，两种曲线检查均通过，其中 723 个的得分不比无故障种子低超过三个标准差，尽管几乎所有运行最终都收敛到了不同的权重，而且对于奖励类故障，记录的回报值也与正常情况不同。而对运行过程本身（而非其结果）的检查则发现了每一个故障。一个步进台阶……（摘要在此处截断）

    arXiv:2611.08144v1 Announce Type: cross  Abstract: Robot reinforcement learning increasingly runs environments outside the trainer: in another process, on another machine, in another language. Many faults at this boundary raise no error. We make the boundary a wire protocol and ask what notices a fault in it.   On Pendulum, HalfCheetah, Hopper and a 7-joint arm (Pusher), we injected up to 23 faults into what crosses the wire and 2 into the trainer's log, from one value to every value. Unless every value was wrong, a learning-curve threshold caught at most one run in ten, and a comparison with ten fault-free seeds at most 21%. A clean evaluation of all 1,060 final policies agreed with the curves: of the 728 boundary-fault runs both curve checks passed, 723 scored no more than three standard deviations below fault-free seeds, though nearly all ended on other weights and, for reward faults, logged other returns.   Checks of the run itself, not of its outcome, saw every fault. A step ledge
    
[^8]: 对称性感知的无人机代理辅助测试方法

    Symmetry-Aware Surrogate-Assisted Testing of UAVs

    [https://arxiv.org/abs/2611.08012](https://arxiv.org/abs/2611.08012)

    提出一种对称性感知的代理辅助测试方法，通过将障碍物配置转换到统一规范坐标系，使基于单一参考任务训练的神经网络代理模型可复用于任意无人机飞行任务，从而以更少的仿真数据生成安全关键测试用例，临界测试用例命中率达42%，远超现有竞赛技术的约10%。

    

    我们研究了一种面向无人机自动驾驶仪的对称性感知代理辅助测试方法。该方法通过将障碍物配置转换到共享的规范坐标系中，利用飞行任务之间的几何对称性。这使得仅在单一参考任务上训练的神经网络代理模型能够在任意任务中复用，从而减少了训练所需的高保真仿真数据量。该代理模型被训练用于预测无人机与周围障碍物之间的最小净空距离，并被用于引导测试生成走向安全关键场景。我们使用Aerialist无人机测试平台对该方法进行评估，并将其性能与2024至2026年无人机测试竞赛中的技术进行基准对比。结果表明，该方法在所评估任务中的平均临界测试用例命中率达到42%，而对比方法约为10%。

    arXiv:2611.08012v1 Announce Type: cross  Abstract: We investigate a symmetry-aware surrogate-assisted testing approach for UAV autopilots. The proposed approach exploits geometric symmetries between flight missions by trans- forming obstacle configurations into a shared canonical coordi- nate frame. This enables a neural-network surrogate trained on a single reference mission to be reused across arbitrary missions, reducing the amount of high-fidelity simulation data required for training. The surrogate model is trained to predict the minimum clearance distance between a UAV and surrounding obstacles and is used to guide test generation towards safety-critical scenarios. We evaluate this approach using the Aerialist UAV test bench and benchmark its performance against techniques from the 2024 to 2026 UAV Testing Competitions. Results show that the approach achieves an average critical-test-case hit rate of 42% across the evaluated missions, compared with approximately 10% for the const
    
[^9]: 无人记录的需求：公共部门机器学习流水线中的需求涌现

    Requirements Nobody Wrote Down: Emergence in a Public-Sector ML Pipeline

    [https://arxiv.org/abs/2611.07951](https://arxiv.org/abs/2611.07951)

    通过对巴西政府数字参与平台ML流水线为期15个月的案例研究，本文揭示了公共部门机器学习系统的需求并非预先明确获取，而是主要从实验过程中涌现，且大多数需求从未被正式记录。

    

    政府日益使用机器学习（ML）来处理超出其分析能力的数据量，然而对于ML实验走向实际运行过程中需求如何产生和演化，人们知之甚少。面向ML的需求工程通常假设需求可以预先获取；然而在公共部门，需求往往以模糊的形式到来，利益相关者使用不同的专业语言，而什么才算“好”的结果事先并不知晓。我们报告了一项为期15个月的案例研究，研究对象是巴西国家数字参与平台（该平台拥有超过60,000条公民提案）的主题分类流水线，通过对43个项目工件与六位项目参与者访谈进行三角验证。研究发现：制度性需求以宽泛目标的形式出现，虽获认可但缺乏资源支持，重建的95项需求和标准中仅有40%曾被正式化；该流水线的关键需求主要从实验中涌现。

    arXiv:2611.07951v1 Announce Type: new  Abstract: Governments increasingly use machine learning (ML) to handle data volumes beyond their analytical capacity, yet little is known about how requirements arise and evolve as an ML experiment moves toward operational use. Requirements Engineering (RE) for ML often assumes that requirements can be elicited upfront; in the public sector, however, demands arrive vague, stakeholders speak different professional languages, and what counts as a ``good'' result is unknown in advance. We report a 15-month case study of a thematic classification pipeline for Brazil's national digital participation platform, which holds over 60,000 citizen proposals, triangulating 43 project artifacts with interviews of six project participants. Institutional demands arrived as broad goals, endorsed but not sponsored, and only 40\% of the 95 reconstructed requirements and criteria were ever formalized. The pipeline's key requirements emerged largely from experiments, 
    
[^10]: tok：面向人类和AI智能体的BPMN与DMN文本具体语法

    tok: a textual concrete syntax of BPMN and DMN for humans and AI agents

    [https://arxiv.org/abs/2611.07945](https://arxiv.org/abs/2611.07945)

    本文提出了tok这一BPMN和DMN的纯文本表示法，其令牌效率比XML高数倍、故障检测率更高，且语言模型能轻松学习并生成有效的tok模型。

    

    tok是一种用于业务流程模型与表示法（BPMN）2.0.2流程模型和决策模型与表示法（DMN）1.5决策的纯文本表示法。一个引擎可以依据这两项标准对tok文档进行检查，将文档编译为带有自动生成图表的XML，并通过模型上下文协议为语言模型应用提供服务。该论文将这种表示法归类为BPMN和DMN元模型的另一种具体语法，并报告了一项研究。tok模型所消耗的令牌数量比37种工具导出的893个文件中最小的BPMN XML少2.2倍，比文件按原样导出时少7.7倍。该检查器能够识别1,411个注入故障中的98.2%，而带有bpmnlint的XML Schema只能识别85.8%。四个此前未接触过tok的语言模型仅凭一页参考文档，就能为来自两个数据集的87个流程描述中的85.7%至100%编写出有效的tok，并为每个描述编写出有效的BPMN XML，其中最强的模型编写的tok比标准XML少25%和19%的tok

    arXiv:2611.07945v1 Announce Type: cross  Abstract: tok is a plain-text notation for Business Process Model and Notation (BPMN) 2.0.2 process models and Decision Model and Notation (DMN) 1.5 decisions. One engine checks a tok document against both standards, compiles the document to XML with an automatic diagram, and serves language-model applications through the Model Context Protocol. The paper classifies the notation as an alternative concrete syntax of the BPMN and DMN metamodels and reports one study. A tok model takes 2.2 times fewer tokens than the smallest BPMN XML of 893 files exported by 37 tools and 7.7 times fewer than the files as exported. The checker signals 98.2% of 1,411 injected faults, against 85.8% for the XML Schema with bpmnlint. Four language models new to tok write valid tok from a one-page reference for 85.7 to 100% of 87 process descriptions from two datasets and valid BPMN XML for every description, and the strongest model writes tok with 25% and 19% fewer tok
    
[^11]: 低相似度分数无法证明源码条件化仓库重写后的独立性

    A Low Similarity Score Does Not Show Independence after Source-Conditioned Repository Rewriting

    [https://arxiv.org/abs/2611.07887](https://arxiv.org/abs/2611.07887)

    研究表明，大语言模型智能体能够低成本地将著作权留项目重写为通过全部测试且相似度很低的版本，而相同配置下多次运行的相似度波动很大，因此低相似度分数不能作为重写独立于原作的证据。

    

    项目维护者有时会引用较低的词元相似度分数，作为大语言模型辅助重写的著作权留（copyleft）项目独立于原作的证据。我们测试了这种分数在“源码条件化”重写场景下能够支持什么结论：在这种场景中，一个编程智能体将原始代码保留在其工作副本中，对仓库进行重写，并反复迭代直到通过固定的行为测试套件，期间没有任何人工对代码的修改。通过对六个著作权留 Python 项目的 69 次重写，我们发现：（1）纯智能体重写成本低廉且常常产生较低的分数。每次受控重写在修复后都能通过测试套件，初始重写通常在半小时内完成。在“从零重构”的提示词下，大多数重写的 JPlag 相似度低于 25%，且仍能通过测试套件。（2）单一的较低分数并不可靠。相同的项目—智能体—提示词配置会产生范围很大的相似度差异；逐次运行间的波动超过了项目和智能体因素的影响（摘要原文在此处截断）。

    arXiv:2611.07887v1 Announce Type: new  Abstract: Maintainers sometimes cite a low token-similarity score as evidence that an LLM-assisted rewrite of a copyleft project is independent of the original. We test what such a score can support under \emph{source-conditioned} rewriting: a coding agent keeps the original in its working copy, rewrites the repository, and iterates until a fixed behavioural harness passes, with no human edit to the code. On 69 rewrites of six copyleft Python projects we find that: \ding{182}~Agent-only rewriting is inexpensive and often yields low scores. Every controlled rewrite passes the harness after repair, usually within half an hour for the initial rewrite. Under a ground-up rearchitecture prompt, most rewrites fall below 25\% JPlag similarity and still pass the harness. \ding{183}~A single low score is an unreliable witness. Identical project--agent--prompt configurations produce wide spreads in similarity; run-to-run variation dominates project and agent
    
[^12]: 基于结构化神经符号推理的多阶段程序组合式循环不变式推断

    Compositional Loop Invariant Inference for Multiphase Programs via Structured Neurosymbolic Reasoning

    [https://arxiv.org/abs/2611.07882](https://arxiv.org/abs/2611.07882)

    本文提出了CLOVER框架，通过将多阶段循环分解为符号化的分支序列、为各分支独立生成候选子不变式并进行针对性修复，实现组合式的循环不变式自动推断。

    

    循环不变式对于演绎程序验证至关重要，然而如何为多阶段程序自动推断循环不变式仍然是一个开放性问题。多阶段程序的循环中包含条件分支，这些分支会导致具有不同行为的多个执行阶段。现有的基于大语言模型（LLM）的方法要么将循环作为一个整体进行推理，要么利用显式的分支信息跨分支联合生成原子断言，而未能维护完整的候选子不变式以支持独立生成和针对性修复。我们提出了CLOVER，一个用于多阶段不变式推断的结构化神经符号推理框架，它建立在两个相互关联的机制之上。第一，组合式推断：CLOVER静态地将循环分解为符号化的“循环分支序列”，即刻画不同执行场景的循环守卫条件与分支条件的合取，并引导LLM为每个场景独立生成候选子不变式，从而使得……

    arXiv:2611.07882v1 Announce Type: new  Abstract: Loop invariants are essential for deductive program verification, yet automatically inferring them for {multiphase programs}, whose loops contain conditional branches that induce multiple execution phases with distinct behaviors, remains an open problem. Existing LLM-based methods either reason about loops as a whole or use explicit branch information to generate atomic assertions jointly across branches, without maintaining complete candidate sub-invariants for independent generation and targeted repair. We present CLOVER, a structured neurosymbolic reasoning framework for multiphase invariant inference built on two interlinked mechanisms. First, compositional inference: CLOVER statically decomposes a loop into symbolic {loop branch sequences}, conjunctions of the loop guard and branch conditions that characterize distinct execution scenarios, and guides the LLM to generate candidate sub-invariants for each one independently, so that ea
    
[^13]: 哪些代码正确性信号能在分布偏移下依然有效？程序与判定正确性的隐状态及因果预测器

    Which Signals of Code Correctness Survive Distribution Shift? Hidden-State and Causal Predictors of Program and Verdict Correctness

    [https://arxiv.org/abs/2611.07876](https://arxiv.org/abs/2611.07876)

    本文系统比较了四类共六种免执行代码正确性预测器在基准与语言分布偏移下的稳健性，并将两种新颖的因果定位方法引入代码领域，同时评估其对程序正确性和模型自身判定可信度的预测能力。

    

    大语言模型越来越多地被用于评判代码，但在缺乏外部反馈的情况下，这类判断并不可靠，而通过执行测试或深度推理来验证这些判断的代价高昂。因此，免执行的正确性预测器利用模型输出概率、采样解或隐状态中的信号，来识别哪些判断值得进一步验证。然而，当这些预测器在跨项目、跨语言进行泛化时，可能会在分布偏移下性能退化。我们在 Qwen3-Coder-30B 和 gpt-oss-20b 上，比较了来自四个类别的六种免执行预测器，考察了 LiveCodeBench 与 BigCodeBench 之间的基准偏移，以及 Python、C++ 和 Java 之间的语言偏移。我们评估了两个目标：模型自身对程序的判定是否可信，以及程序本身是否正确。除已有方法外，我们还将两种新颖的因果定位方法适配到代码领域，这些方法已被证明更为稳健。

    arXiv:2611.07876v1 Announce Type: cross  Abstract: Large language models increasingly judge code, yet without external feedback such judgments are unreliable, and verifying them by executing tests or by extended reasoning is costly. Instead, execution-free correctness predictors use signals from the model's output probabilities, sampled solutions, or hidden states to identify judgments that warrant further verification. However, these predictors may degrade under distribution shifts when generalizing across projects and languages.   We compare six execution-free predictors from four families on Qwen3-Coder-30B and gpt-oss-20b under benchmark shift between LiveCodeBench and BigCodeBench and language shift across Python, C++ and Java. We evaluate two targets: whether the model's own verdict on a program can be trusted, and whether the program itself is correct. Besides established approaches, we adapt two novel causal localization methods to code, which have been shown to be more robust 
    
[^14]: 让客体适应主体：面向实证软件工程的代表性研究材料

    Adapt the Object to the Subject: Representative Study Material for Empirical Software Engineering

    [https://arxiv.org/abs/2611.07772](https://arxiv.org/abs/2611.07772)

    该论文首次关注实证软件工程中“研究对象与被试自身工作不匹配”这一被忽视的外部效度问题，并提出一种利用LLM在在线调查中根据参与者简短上下文描述即时生成适配语境的合成需求（可注入需求坏味作为受控刺激）的实验方法。

    

    外部效度是实证软件工程中的核心关注点，然而相关讨论大多聚焦于研究参与者是否具有代表性，而参与者所面对的研究对象——代码、模型或需求——在多大程度上能代表每位参与者自己的实际工作，却常常被忽视。从业者自身的工件（artifact）虽然是最理想的材料，但往往难以获取，因此研究人员通常从公开数据集中抽样或人工构造研究对象。这种主体与客体之间的不匹配是否会影响参与者对材料进行处理的能力，目前尚不清楚。我们朝着回答这一问题（针对需求领域）迈出了第一步。我们提出了一种方法：利用大语言模型（LLM）在在线调查过程中，根据参与者提供的简短上下文描述即时（just-in-time）生成适配上下文的合成需求，并可选择性地注入需求坏味（requirements smells）作为受控刺激。为验证该方法，我们进行了一项受控实验……

    arXiv:2611.07772v1 Announce Type: new  Abstract: External validity is a central concern in empirical software engineering, yet the debate has largely focused on whether study participants are representative. How well the study objects they reason about - code, models, or requirements - are representative of each participant's own work is often overlooked. Practitioners' own artifacts would be ideal, but are often out of reach, so researchers sample from public datasets or construct objects artificially. Whether this lack of subject-object fit affects participants' ability to act on the material remains unknown. We take a first step toward answering this question for requirements. We propose an approach that uses an LLM to generate context-adapted synthetic requirements just-in-time during an online survey from a short, participant-supplied context description, optionally injecting requirements smells as controlled stimuli. To validate the approach, we conducted a controlled experiment 
    
[^15]: 软件日志上的快速无监督异常检测方法——一项实证研究

    Fast Unsupervised Anomaly Detection Methods on Software Logs - An Empirical Study

    [https://arxiv.org/abs/2611.07760](https://arxiv.org/abs/2611.07760)

    该研究通过对四种无监督异常检测模型、三种日志表示方式及多种预处理策略在七个真实数据集上进行大规模组合实证评估，为软件日志快速无监督异常检测的方案选择提供了实证依据。

    

    对软件日志进行异常检测可以通过故障识别、根因分析和故障定位来保障系统可靠性。尽管有监督模型表现出色，但在实际场景中很少能获得带标签的训练数据，因此需要使用无监督模型。我们研究了数据预处理、异常检测方法和数据集所带来的影响。在预处理方面，我们考察了三种选择：使用无异常数据的训练集、使用随机选取（而非按时间顺序选取）的训练数据，以及使用自定义加载器从日志行中提取日志消息。在异常检测方面，我们使用了四种模型（KMeans、IsolationForest、RarityModel 和 Out-Of-Vocabulary 检测器）以及三种日志表示方式（词、三元字符组和解析后的事件）。我们在七个数据集（BGL、HDFS、Hadoop、Spirit、Thunderbird、Liberty 和 LO2Token）上评估了这些方法，并对所选因素的每种组合进行了运行测试（原文摘要在此处截断）。

    arXiv:2611.07760v1 Announce Type: new  Abstract: Anomaly detection on software logs ensures system reliability through failure identification, root cause analysis and fault localization. While supervised models show excellent performance, in a practical setting there rarely are labeled training data available. This calls for the use of unsupervised models. We study effects of data preprocessing, anomaly detection method, and the dataset. For preprocessing, we examine three options: The use of anomaly-free training data, using randomized as opposed to chronologically selected training data, and the use of a custom loader to extract log message from the log line. Regarding anomaly detection we use four models (KMeans, IsolationForest, RarityModel, Out-Of-Vocabulary Detector) and three representations (Words, trigrams, and parsed events). We evaluate these approaches on seven datasets: BGL, HDFS, Hadoop, Spirit, Thunderbird, Liberty, and LO2Token. We run each combination of the selected f
    
[^16]: ReConMem：一种面向仓库对话的基于图的双记忆系统

    ReConMem: A Graph-Based Dual Memory System for Repository-Oriented Conversations

    [https://arxiv.org/abs/2611.07759](https://arxiv.org/abs/2611.07759)

    提出ReConMem，一种基于图的双记忆系统，将对话记忆与代码实体分别组织为相互关联的语义图和结构图，并通过结合自然语言与代码对齐的检索机制，实现面向仓库级开发对话的高效记忆召回。

    

    基于大语言模型（LLM）的代码助手在迭代式仓库级开发中的应用日益增多，在此场景下，有效的辅助需要同时利用不断演变的对话历史和仓库上下文。然而，现有的记忆系统主要是为通用对话设计的，未能充分协调对话信息与其所指的结构化代码实体之间的关系。我们提出了ReConMem，一种面向仓库对话的基于图的双记忆系统。ReConMem将精炼的对话记忆和与对话相关的代码实体组织成相互关联的语义记忆图和结构记忆图，并在交互演变过程中持续维护它们之间的跨图关联。在记忆召回方面，ReConMem通过基于代码草稿的查询锚定将自然语言检索与代码对齐检索相结合，在图内和图间关系上对检索到的信息进行扩展，并对检索到的代码进行压缩……（摘要在此处截断）

    arXiv:2611.07759v1 Announce Type: new  Abstract: LLM-based code assistants are increasingly used in iterative repository-level development, where effective assistance requires jointly leveraging evolving conversational history and repository context. Existing memory systems, however, are primarily designed for general-purpose conversations and do not sufficiently coordinate conversational information with the structured code entities to which it refers. We propose ReConMem, a graph-based dual memory system for repository-oriented conversations. ReConMem organizes distilled conversational memories and conversation-relevant code entities into interconnected semantic and structural memory graphs, while maintaining their cross-graph associations as interactions evolve. For memory recall, ReConMem combines natural-language and code-aligned retrieval through code-draft-based query grounding, expands retrieved information over intra- and inter-graph relations, and compresses retrieved code ac
    
[^17]: 通过集成与过程级分析揭示图数据库测试中的系统性盲点

    Uncovering Systematic Blind Spots in Graph Database Testing through Integration and Process-Level Analysis

    [https://arxiv.org/abs/2611.07754](https://arxiv.org/abs/2611.07754)

    该研究通过集成8种图数据库自动化测试方法并进行过程级实证分析，揭示了图数据库测试中存在大量由方言和模式匹配差异导致的误报缺陷报告等系统性盲点。

    

    图数据库正在成为互联数据的基础设施，越来越多的自动化测试技术已在成熟的图数据库系统中发现了大量缺陷。每种技术检查正确性的不同方面，并暴露不同类型的缺陷。将它们结合起来是自然的下一步，但每种技术的评判标准在于它发现的真实缺陷数量，而非无用的报告或永远不会失败的测试，这决定了开发者能否负担起修复这些问题的成本。为了找出仍然存在的差距，我们集成了8种测试方法的测试能力，并对它们的行为进行了实证研究，在此过程中我们发现了两个盲点。第一，12小时的测试会产生数千份缺陷报告，远超一个成熟数据库可能包含的真实缺陷数量，其中大多数报告是由方言或模式匹配差异引起的误报，而其余报告中许多……（摘要在此处截断）

    arXiv:2611.07754v1 Announce Type: new  Abstract: Graph databases are becoming foundational infrastructure for connected data, and a growing body of automated testing techniques has uncovered many bugs across mature graph database systems. Each technique checks a different aspect of correctness and exposes a different class of defect. Combining them is the natural next step, but each technique is judged by the real bugs it finds, not by useless reports or tests that can never fail, and that is what decides whether a developer can afford to fix the problems. To identify the remaining gaps, we integrated test capabilities from 8 testing methods and conducted an empirical study of their behavior, during which we found two blind spots. First, 12 hours of testing produce thousands of bug reports, far more than the number of genuine defects a mature database is likely to contain. Most reports are false positives caused by dialect or pattern-matching differences, and many of the remaining repo
    
[^18]: 修复一个，漏掉另一个：利用历史错误修复来发现数据库系统中的残留错误

    Fix One, Miss Another: Leveraging Historical Bug Fixes to Uncover Residual Bugs in Database Systems

    [https://arxiv.org/abs/2611.07739](https://arxiv.org/abs/2611.07739)

    该论文提出Reaper方法，通过分析历史错误修复来推断触发条件与验证逻辑，从而系统性地发现数据库系统中同类处理器里残留的、与历史错误同源的未修复缺陷（同类错误）。

    

    数据库管理系统（DBMS）通过基于语法的调度处理器来实现SQL构造。尽管语法上同类的处理器实现的是不同的SQL构造，但它们往往共享一种通用的实现模式，包括参数提取、类型检查和权限验证。因此，针对历史错误所触发的那个处理器进行加固的错误修复，可能会使同类处理器中的相应逻辑仍然容易受到相同语义缺陷的攻击。我们将此类缺陷称为“同类错误”。尽管同类处理器之间存在结构上的相似性，现有的错误查找技术却很少利用历史修复来系统地发现这些相关缺陷。我们提出了Reaper，一种由历史错误修复引导、用于检测同类错误的新方法。Reaper首先分析一个错误修复，以推断出一个触发配置文件（trigger profile），它刻画了缺陷被暴露的条件；以及一个验证配置文件（validation profile），它捕捉……

    arXiv:2611.07739v1 Announce Type: new  Abstract: Database management systems (DBMSs) implement SQL constructs through grammar-driven dispatch handlers. Although handlers for syntactic siblings implement different SQL constructs, they often share a common implementation pattern, including argument extraction, type checking, and permission validation. Consequently, a bug fix that hardens the handler exercised by a historical bug may leave the corresponding logic in sibling handlers vulnerable to the same semantic flaw. We refer to such defects as sibling bugs. Despite the structural similarity among sibling handlers, existing bug-finding techniques rarely exploit historical fixes to systematically uncover these related defects.   We present Reaper, a novel approach for detecting sibling bugs guided by historical bug fixes. Reaper first analyzes a bug fix to infer a trigger profile, which characterizes the conditions under which the defect is exposed, and a validation profile, which captu
    
[^19]: 不要提示你的智能体，而是装备它们：Skillware——面向逻辑系统的主权、可审计且确定性能力执行的模块化标准

    Don't Prompt Your Agents. Equip Them: Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems

    [https://arxiv.org/abs/2611.07652](https://arxiv.org/abs/2611.07652)

    该论文提出Skillware开放框架，通过模块化、版本化的软件能力包（包含类型化JSON契约、确定性代码、任务指令、数据集和单元测试）取代脆弱的提示工程，将智能体的意图与执行正式分离，实现主权、可审计且确定性的能力执行。

    

    当前的自主逻辑系统依赖于一个脆弱的假设：语言模型应该动态编写代码、猜测API载荷，并通过提示工程来记忆操作约束。在实践中，这种方式会大量消耗上下文窗口、推高API费用，并产生在生产环境中不可预测地失效的脆弱智能体。在本工作中，我们提出了Skillware，一个开放的架构框架和能力注册表，它正式地将智能体“想要做什么”与“任务如何执行”分离开来。Skillware不是用松散的文本指令或被美其名曰“技能”的markdown文件来提示模型，而是为智能体配备模块化、版本化的软件能力包。就像视频游戏中的技能快捷键在固定世界规则下确定性地执行一样，每个能力包都结合了类型化的JSON输入契约、确定性的Python代码、任务指令、捆绑的数据集以及自动化单元测试。进程内适配器将……（原文摘要至此被截断）

    arXiv:2611.07652v1 Announce Type: new  Abstract: Current autonomous logical systems rely on a fragile assumption: that language models should dynamically write code, guess API payloads, and memorize operational constraints through prompt engineering. In practice, this burns massive context windows, inflates API bills, and produces brittle agents that fail unpredictably in production.   In this work, we present Skillware, an open architectural framework and capability registry that formally separates what an agent wants to do from how that task is executed. Instead of prompting models with loose text instructions or markdown files baptized as 'skills', Skillware equips them with modular, versioned software capability bundles. Much like skill hotkeys in video games that execute deterministically under fixed world rules, each bundle combines a typed JSON input contract, deterministic Python code, task directives, bundled datasets, and automated unit tests. In-process adapters translate th
    
[^20]: 中间商究竟交付了什么：LLM API中转服务中的回答行为、工具权限与计费用量

    What the Middleman Delivers: Answer Behavior, Tool Permissions, and Billed Usage in LLM API Relays

    [https://arxiv.org/abs/2611.07650](https://arxiv.org/abs/2611.07650)

    本研究对20个提供GPT-6 Astra和Claude Opus 5的第三方LLM API中转服务进行系统性测试，发现约半数低价中转服务的回答行为与官方模型存在差异，并通过检查返回内容、工具权限与计费用量揭示了中转服务在交付质量上的不可靠性。

    

    第三方应用程序接口（API）中转服务以通常更低的价格提供大型语言模型（LLM）的访问。用户依赖这些服务来保障模型质量、指令遵循能力以及准确的计费。然而，现有研究并未澄清中转服务能否同时满足这些期望。我们测试了20个提供GPT-6 Astra和Claude Opus 5的中转服务。首先，我们将知识类问题的回答与批量匹配的官方响应进行比较。在第一轮样本中，分析标记出7/10的低价中转服务和2/10的同价或更高价中转服务存在行为差异。在重新测试35对样本时，按照原始规则有七个服务被新增或撤销标记，而按照匹配的修正族规则则有六个。由于回答比较无法揭示所有交付问题，我们进一步检查了返回内容和工具权限。在针对固定的、非对抗性的代码与配置工作负载筛选的3,332条响应中，未确认存在注入行为，但有四个中转服务返回……

    arXiv:2611.07650v1 Announce Type: cross  Abstract: Third-party application programming interface (API) relays offer access to large language models (LLMs), often at lower prices. Users depend on these services for model quality, instruction compliance, and accurate billing. However, existing studies leave unclear whether relays meet these expectations together. We test 20 relays offering GPT-6 Astra and Claude Opus 5. First, we compare answers to knowledge questions with batch-matched official responses. In our first-round sample, the analysis flags 7/10 low-price and 2/10 same- or higher-price relays for behavioral differences. On retesting 35 pairs, seven gain or lose a flag under the original rules, and six under matched correction families. Because answer comparisons cannot reveal every delivery problem, we examine returned content and tool permissions. No injection is confirmed in 3,332 screened responses to a fixed, non-adversarial code and setup workload, yet four relays return 
    
[^21]: AiXapply：面向IDE工作流、基于专用小型模型的快速可靠全文件代码集成

    AiXapply: Fast and Reliable Full-File Code Integration with Specialized Small Models for IDE Workflows

    [https://arxiv.org/abs/2611.07634](https://arxiv.org/abs/2611.07634)

    该论文将代码集成任务形式化为根据原始文件与更新片段生成完整合并文件的问题，构建了基于真实提交的多语言数据集，并通过监督微调与GRPO强化学习训练出专用4B模型aiXapply，以94.4%的准确率超越大模型基线，为IDE工作流提供快速可靠的全文件代码集成。

    

    大型语言模型越来越多地通过提出小型代码编辑来辅助开发者，然而如何将部分更新片段可靠地集成到现有文件中并保留无关内容，目前的代码生成工作流仍缺乏良好支持。我们将代码集成任务形式化如下：给定一个原始文件和一个更新片段，系统必须生成完整的合并文件，同时保留所请求编辑之外的所有内容。我们通过结构化流水线从真实提交中构建了高质量的多语言数据集，该流水线将代码变更转换为与全文件真值配对的更新片段，并通过验证和过滤保证一致性。随后，我们使用监督微调和强化学习（GRPO）训练了一个专用的4B模型aiXapply。监督微调版本在我们的语言感知比较下达到94.4%的准确率，超越了强大的大模型基线（Deepseek-V3.2）。

    arXiv:2611.07634v1 Announce Type: new  Abstract: Large language models increasingly assist developers by proposing small code edits, yet reliably integrating a partial update snippet into an existing file while preserving unrelated content remains poorly supported by current code-generation workflows. We formalize the Code Integration (Apply) task as follows: given an original file and an update snippet, the system must produce the complete merged file while preserving all content outside the requested edit. We construct a high-quality multi-language dataset from real commits via a structured pipeline that converts code changes into update snippets paired with full-file ground truth and enforces consistency through verification and filtering. We then train a specialized 4B model, aiXapply, using supervised fine-tuning and reinforcement learning (GRPO). The supervised variant achieves 94.4% accuracy under our language-aware comparison, surpassing a strong large-model (Deepseek-V3.2) bas
    
[^22]: Evolver：将智能体探索转化为不断演进的能力

    Evolver: Turning Agent Exploration into Evolving Capabilities

    [https://arxiv.org/abs/2611.07595](https://arxiv.org/abs/2611.07595)

    Evolver 是一个无需更新基础模型参数的经验驱动框架，它通过结构化的 Gene 资产将智能体探索产生的经验转化为持久可复用的能力，在四个基准测试中显著提升任务成功率并将执行 token 消耗降低约 24%。

    

    智能体探索会产生有价值的程序性知识，然而这些经验大多仍局限于瞬态的执行轨迹中，未能被系统性地复用。我们提出 Evolver，一个经验驱动的框架，它能够在不更新基础模型参数的前提下，将智能体探索转化为持久且可复用的能力。Evolver 以结构化的 Gene 资产为核心，集成了经验获取、检索以及反馈引导的演化机制，从而在现有智能体框架中支持基于经验的条件化执行。我们使用 Claude Code 和 Codex 两种基础模型配置，以及 Terminal-Bench 2.1、SWE-bench Multilingual、SWE-rebench 和 GAIA 四个基准测试对 Evolver 进行了评估。在一个包含 320 组配对执行的诊断集上，Evolver 将任务成功率从 83.13% 提升至 87.50%（提高 4.38 个百分点），同时将记录的任务执行 token 消耗降低了 24.24%。此外，在 265 组成功……（摘要在此处截断）

    arXiv:2611.07595v1 Announce Type: cross  Abstract: Agent exploration produces valuable procedural knowledge, yet much of this experience remains confined to transient execution traces and is not systematically reused. We introduce Evolver, an experience-driven framework that transforms agent exploration into persistent, reusable capabilities without updating foundation model parameters. Built around structured Gene assets, Evolver integrates experience acquisition, retrieval, and feedback-guided evolution to support experience-conditioned execution across existing agent harnesses. We evaluate Evolver using Claude Code and Codex, two foundation model configurations, and four benchmarks: Terminal-Bench 2.1, SWE-bench Multilingual, SWE-rebench, and GAIA. On a selected diagnostic set of 320 paired executions, Evolver improves task success from 83.13% to 87.50% (+4.38 percentage points) while reducing recorded task-execution token consumption by 24.24%. Furthermore, among 265 pairs successf
    
[^23]: 先路由后响应：面向LLM赋能客服软件的故障关闭式路由

    Route Before Response: Fail-Closed Routing for LLM-Enabled Customer-Service Software

    [https://arxiv.org/abs/2611.07534](https://arxiv.org/abs/2611.07534)

    RouteBR提出“先路由后响应”的故障关闭式控制架构，将一次模型调用与确定性代码校验相结合，在真实金融机构的200个场景上实现97.5%最终路径精确匹配，并确保被拦截的主路由不会被低排名候选替代。

    

    金融机构中基于LLM的客服软件必须在响应之前选择下游服务。分类器和LLM路由器可以识别相关的标签或目的地，但一个组合式应用仍需验证分发权限与实体注册。我们将这一集成问题作为边界感知的响应前路由进行研究。我们提出了RouteBR，一种控制器：由一次模型调用提出主服务动作和注册对象，随后由确定性代码在分发之前检查标识符和版本化规则，或在指定情形下给出不分发的结果。排名较低的候选者不能替代被拦截的主路由。我们从一家真实金融机构反复出现的机密请求模式中构建了600个隐私保护场景；三名工程师共同审查了这些场景和标签。在200个案例的CONFIRM MAIN分割集上，RouteBR达到了97.5%的最终路径精确匹配。一个匹配的单次调用基线在给定……（摘要原文在此处被截断）

    arXiv:2611.07534v1 Announce Type: new  Abstract: LLM-enabled customer-service software in financial institutions must choose a downstream service before responding. Classifiers and LLM routers can identify a relevant label or destination, but a composed application must still verify dispatch permission and entity registration. We study this integration problem as boundary-aware pre-response routing. We present RouteBR, a controller in which one model call proposes a primary service action and registered object, then deterministic code checks identifiers and versioned rules before dispatch or a specified non-dispatch outcome. Lower-ranked candidates cannot replace a blocked primary route. We construct 600 privacy-preserving scenarios from recurring confidential-request patterns at a real financial institution; three engineers jointly reviewed the scenarios and labels. On the 200-case CONFIRM MAIN split, RouteBR reaches 97.5% final-path exact match. A matched one-call baseline given the 
    
[^24]: 通往真实世界的最后一英里：作为解决真实世界任务的可复用构建块的智能体原生原语

    The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks

    [https://arxiv.org/abs/2611.07461](https://arxiv.org/abs/2611.07461)

    提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。

    

    智能体如何端到端地处理真实世界任务？配备各自框架的前沿模型在真实世界基准测试中仍然表现不佳：在 Terminal-Bench 4.0 上，GPT-5.6 Sol 配合 Codex 仅解决 37.3% 的任务，Claude Opus 5 配合 Claude Code 解决 51.8%。这些任务在从获取资源到应用资源的步骤上失败，我们将其称为通往真实世界的最后一英里。现有方法只是将资源置于可及范围内，却把应用资源的能力留给调用者，因此每个框架都需要重建这种能力。我们提出了智能体原生原语——一种可复用的单元，它为资源配备一个常驻 LLM，该 LLM 能在上下文中应用该资源并与其他原语通信；同时提出 HEELO（围绕不断演化的智能体原生原语构建的框架工程方法，用于真实世界任务的最后一英里），这是一个从此类原语库为每个任务构建框架的框架工程方法。HEELO 将任务所需的原语组合成任务特定的框架……

    arXiv:2611.07461v1 Announce Type: cross  Abstract: How can an agent handle a real-world task end to end? Frontier models with their own harnesses still fall short on real-world benchmarks: on Terminal-Bench 4.0, GPT-5.6 Sol with Codex resolves 37.3% of the tasks and Claude Opus 5 with Claude Code 51.8%. These tasks fail at the step from reaching a resource to applying it, which we call the last mile to the real world. Existing approaches put the resource within reach but leave the competence to apply it with the caller, so each harness rebuilds that competence. We introduce agent-native primitives, reusable units that equip a resource with a resident LLM that applies it in context and communicates with other primitives, and HEELO (Harness Engineering around Evolving Agent-Native Primitives for the Last Mile to Real-WOrld Tasks), a harness engineering framework that builds each task's harness from a library of such primitives. HEELO composes the primitives a task needs into a task-speci
    
[^25]: Req2Rust：一种需求引导的智能体框架，用于项目级C++到Rust的迁移

    Req2Rust: A Requirements-Guided Agent Harness for Project-Level C++-to-Rust Migration

    [https://arxiv.org/abs/2611.07442](https://arxiv.org/abs/2611.07442)

    Req2Rust是首个项目级基于大语言模型的C++到Rust迁移框架，通过先从源代码中推导出有证据支撑的迁移需求规范（MRS）来引导智能体完成迁移，在语义保真与符合Rust习惯的项目重构之间取得平衡。

    

    内存安全问题使Rust成为对遗留C/C++系统进行现代化改造的理想目标，然而项目级的C++迁移必须在语义保真度与面向Rust的重构之间取得平衡。保留源代码结构的迁移能够追踪底层行为，但也可能将源语言的组织方式、不安全惯用法以及项目假设带入Rust中。直接基于智能体的翻译给予模型更多自由来构建Rust项目，但语义覆盖范围取决于模型能否识别出必须保留的源行为。这一问题在C++中尤为突出，因为C++的源行为往往隐藏在对象模型、析构函数驱动的生命周期协议以及其他难以定位的C++语义之中。本文提出了Req2Rust，这是首个项目级的基于大语言模型的C++到Rust迁移框架。Req2Rust首先推导出一个有证据支撑的迁移需求规范（MRS），用于记录需要保留的行为和约束条件。

    arXiv:2611.07442v1 Announce Type: new  Abstract: Memory-safety concerns make Rust an attractive target for modernizing legacy C/C++ systems, yet project-level C++ migration must balance semantic fidelity with Rust-oriented reconstruction. Source-structure-preserving migration can track low-level behavior, but it can also carry source-language organization, unsafe idioms, and project assumptions into Rust. Direct agent-based translation gives the model more freedom to construct a Rust project, but semantic coverage depends on the model identifying the source behaviors that must be preserved. This problem is especially acute for C++, where source behaviors are often hidden in object models, destructor-driven lifetime protocols, and other hard-to-localize C++ semantics. This paper presents Req2Rust, the first project-level LLM-based C++-to-Rust migration harness. Req2Rust first derives an evidence-grounded Migration Requirements Specification (MRS) that records the behaviors and constrain
    
[^26]: 基于大语言模型的HLS生成中的规则引导与轨迹感知修复方法

    Rule-Guided and Trajectory-Aware Repair for LLM-Based HLS Generation

    [https://arxiv.org/abs/2611.07324](https://arxiv.org/abs/2611.07324)

    MementoHLS框架通过可执行修复契约库（将跨任务HLS知识转化为证据驱动的修复动作）与证伪账本（将修复历史转化为轨迹证据以抑制无效路径）的双记忆引导机制，并借助同一候选评审者实现可靠的结果归因，显著提升了大语言模型从自然语言规范生成可综合HLS-C代码的端到端成功率。

    

    大语言模型为直接从自然语言规范生成功能正确且可综合的HLS-C代码提供了一条有前景的路径。然而，现有方法往往难以将多轮工具反馈和可复用的HLS知识转化为有效的修订，限制了端到端生成的成功率。我们提出了MementoHLS，这是一个围绕可执行且可证伪的双记忆引导修复构建的智能体式HLS生成框架。MementoHLS将可执行修复契约库（ERCL）与基于评审者的证伪账本（RFL）相结合：前者将跨任务的HLS知识转化为由证据触发的修复动作；后者将任务局部的修复历史转化为轨迹证据，以抑制无效的修复路径并保留已验证的进展。为确保可靠的结果归因，同一候选评审者将解析、编译、测试平台（TB）和综合（Synth）的结果绑定到相应的候选方案上。

    arXiv:2611.07324v1 Announce Type: cross  Abstract: Large language models (LLMs) offer a promising path for directly generating functionally correct and synthesizable HLS-C from natural-language specifications. However, existing approaches often struggle to turn multi-round tool feedback and reusable HLS knowledge into effective revisions, limiting end-to-end generation success. We propose MementoHLS, an agentic HLS generation framework built around executable and falsifiable dual-memory-guided repair. MementoHLS combines an Executable Repair Contract Library (ERCL), which converts cross-task HLS knowledge into evidence-gated repair actions, with a Reviewer-Grounded Falsification Ledger (RFL), which converts task-local repair history into trajectory evidence to suppress ineffective routes and preserve validated progress. To ensure reliable outcome attribution, a Same-Candidate Reviewer binds Parse, Compile, Testbench (TB), and Synthesis (Synth) outcomes to the corresponding candidate, w
    
[^27]: 从这里接手：评估编程智能体的项目接管能力

    Take It From Here: Evaluating Coding Agents On Project Takeover

    [https://arxiv.org/abs/2611.07307](https://arxiv.org/abs/2611.07307)

    该论文提出了评估编程智能体项目接管能力的新基准，发现即使提供完整交互历史，智能体的平均完整交付率也仅为4.95%，揭示了当前智能体在从历史中恢复需求并完成项目交付方面的严重不足。

    

    编程智能体越来越多地在长期项目中协同工作。在接管项目时，它们需要从交互历史中恢复当前适用的需求，并根据新请求对这些需求进行更新。然而，大多数编程基准测试不包含较长的交互历史，而一些考虑了历史的评估也未能覆盖从需求重构到项目交付的完整流程。因此，我们为智能体提供交互历史、当前请求以及请求前的项目环境，要求它们完成所请求的更改，并通过交付测试来评估其接管性能。在六种模型-框架配置以及37个可进行可执行验证的目标中，完整历史条件下的平均完整交付率仅为4.95%。随后，我们使用需求原子对更大规模的接管目标进行独立诊断，考察相关信息识别、请求前状态重构等方面……

    arXiv:2611.07307v1 Announce Type: new  Abstract: Coding agents increasingly collaborate on long-term projects. When taking over a project, they need to recover currently applicable requirements from the interaction history and update them in response to new requests. However, most coding benchmarks do not include long interaction histories, and some evaluations that consider history still do not cover the full process from requirement reconstruction to project delivery. We therefore provide agents with the interaction history, current request, and pre-request project environment, ask them to make the requested changes, and assess takeover performance through delivery tests. Across six model-harness configurations and 37 targets eligible for executable validation, the mean complete-delivery rate under Full History is only 4.95%. We then use requirement atoms to independently diagnose a larger set of takeover targets, examining relevant information identification, pre-request state recon
    
[^28]: 生成式AI时代的代码、提交与复杂度：基于团队毕业设计项目的纵向研究

    Code, Commits, and Complexity in the GenAI Era: A Longitudinal Study of Team-Based Capstone Projects

    [https://arxiv.org/abs/2611.07297](https://arxiv.org/abs/2611.07297)

    该研究对2021-2025年间260个毕业设计团队代码仓库进行了十个学期的纵向分析，发现生成式AI工具出现后学生团队项目代码库规模增长约三倍，揭示了真实项目环境中学生开发行为与代码特征的显著变化。

    

    生成式AI工具如今已广泛供学生使用，但关于团队项目代码库自其出现以来发生了怎样的变化，人们知之甚少。当项目式课程将代码仓库活动作为学习进展的证据时，这一点尤为重要。现有研究大多考察的是自我报告、受控任务或简短的个人作业，学生代码库在真实项目环境中如何演变仍是一个悬而未决的问题。我们对2021年至2025年间N=260个毕业设计团队代码仓库进行了为期十个学期的分析，涵盖了AI出现前后的不同群体。我们从三个维度进行考察：代码规模（代码行数）、代码特征（注释密度、错误处理密度和圈复杂度）以及开发行为（提交量、编辑大小和提交频率）。我们发现，学期末的代码库规模增长约三倍，从2021-2023年的约3.4千行代码增长到2024-2025年的约10千行代码。代码特征显示出明显的解耦……

    arXiv:2611.07297v1 Announce Type: cross  Abstract: Generative AI tools are now widely available to students, but little is known about how team-based project codebases have changed since they appeared. This matters when project courses use repository activity as evidence of progress and learning. Existing studies have largely examined self-reports, controlled tasks, or short individual assignments, leaving open questions around how student codebases evolve in authentic project settings. We present a ten-semester analysis of N=260 capstone team repositories from 2021 to 2025, spanning pre- and post-AI cohorts. We examine three dimensions: code size (lines of code), code characteristics (comment density, error-handling density, and cyclomatic complexity), and development behaviour (commit volume, edit size, and commit frequency). We find that end-of-semester codebases roughly triple, from about 3.4 KLOC in 2021--2023 to about 10 KLOC in 2024--2025. Code characteristics show a clear decou
    
[^29]: 智能体应如何“看”代码？重新思考面向智能体软件工程的代码表示

    How Should Agents See Code? Rethinking Code Representations for Agentic Software Engineering

    [https://arxiv.org/abs/2611.07286](https://arxiv.org/abs/2611.07286)

    本文将基于地图的代码表示（抽象语法树、调用图、模块依赖、文档增强搜索及其混合形式）作为智能体软件工程的设计变量，系统评估了五种代码表示工作流如何影响智能体的检索行为、任务成功率、token消耗与延迟。

    

    大型语言模型智能体正越来越多地被用于仓库级软件工程任务。然而，大多数智能体仍通过原始文件、词法搜索和shell命令来访问代码仓库，主要将代码视为纯文本。本文将基于“地图”式代码表示的工作流作为智能体软件工程中的一个设计变量进行研究，考察不同的仓库视图如何改变智能体的检索行为、任务成功率、token消耗和延迟。本文研究了五种代码表示工作流：抽象语法树、调用图、模块依赖、文档增强搜索，以及将它们结合起来的混合工作流。每种工作流都为智能体提供了一张可在任务执行期间查询的仓库地图。我们在四个仓库级任务上，使用搭载Haiku 4.5的Claude Code与原始工具基线进行对比评估：利用LinuxFLBench进行故障定位、利用SWE-bench Pro进行仓库级代码生成与修复、利用SWT-bench进行测试生成（原文此处截断）。

    arXiv:2611.07286v1 Announce Type: cross  Abstract: LLM agents are increasingly used for repository-level software engineering tasks. However, most agents still access repositories through raw files, lexical search, and shell commands, treating code primarily as text. This paper investigates map-based code-representation workflows as a design variable in agentic software engineering, examining how different repository views change agents' retrieval behavior, task success, token use, and latency.   In this paper, we study five code-representation workflows: abstract syntax trees, call graphs, module dependencies, documentation-enriched search, and a hybrid workflow that combines them. Each workflow exposes a repository map that the agent can query during a task. We evaluate Claude Code with Haiku 4.5 against a raw-tool baseline on four repository-level tasks: fault localization with LinuxFLBench, repository-level code generation and repair with SWE-bench Pro, test generation with SWT-ben
    
[^30]: SeVerus：面向 Rust 程序验证的上下文感知、游戏化基准测试

    SeVerus: Context-Aware, Gamified Benchmarking for Rust Verification

    [https://arxiv.org/abs/2611.07194](https://arxiv.org/abs/2611.07194)

    本文提出 SeVerus，一种上下文感知、游戏化的 Verus 基准测试生成技术，它利用已验证项目的原始证明依赖关系，通过逐步移除证明上下文将验证任务组织成从易到难的任务链，从而更真实地评估编码智能体自主发现并证明辅助引理的能力。

    

    编码智能体在 Verus（一种 Rust 程序验证器）的现有基准测试上取得了很高的成功率。这些基准测试通常直接提供完成证明所需的辅助引理。然而在实践中，这类引理必须被发现并加以证明。为了刻画这一挑战，我们提出了 SEVERUS，这是一种上下文感知的 Verus 基准测试生成技术，它以游戏化的方式将验证任务组织成任务链，每条任务链又被划分为不同难度级别。利用已验证项目中原始的证明依赖关系，SEVERUS 为每个验证目标构建一条任务链，并通过逐步移除证明上下文来创建各个级别。这些级别从最简单的（只需补全单个缺失引理）到最困难的（需要在没有任何辅助引理的情况下证明目标）。一个独立的检查器会拒绝对可执行代码的修改、被削弱的规范以及不健全的证明构造。一种带噪声的二分搜索方法用于估计……（原文在此处截断）

    arXiv:2611.07194v1 Announce Type: cross  Abstract: Coding agents achieve high success rates on existing benchmarks for Verus, a verifier for Rust programs. These benchmarks typically provide the auxiliary lemmas needed to complete a proof. In practice, however, such lemmas must be discovered and proved. To capture this challenge, we present SEVERUS, a context-aware technique to generate Verus benchmarks that organizes verification tasks into quests, each divided into levelsin a gamified manner. Using the original proof dependencies in verified projects, SEVERUS constructs one quest per verification target and progressively removes proof context to create individual levels. These levels range from the easiest, requiring completing only a single missing lemma, to the hardest, requiring proving the target without any supporting lemmas. An independent checker rejects changes to executable code, weakened specifications, and unsound proof constructs. A noisy binary search estimates the highe
    
[^31]: 镜头下的全球监控：重构被动Dolev-Yao入侵者

    Global Surveillance Under the Lens: Reconstructing Passive Dolev-Yao Intruders

    [https://arxiv.org/abs/2611.07170](https://arxiv.org/abs/2611.07170)

    本文提出一种自动化的规范指导方法，能够从PCAP捕获的网络流量中重构被动Dolev-Yao入侵者的执行过程，并依据AnBx形式化协议规范进行验证，同时解决了TCP流中对象重构、并发会话映射歧义和语义验证三大挑战。

    

    全球监控系统记录大量网络流量用于回溯分析，但仅凭数据包捕获难以揭示协议层面的语义。本文提出了一种自动化的、规范指导下的方法论，用于从捕获的网络流量中重构被动Dolev-Yao入侵者的执行过程，并依据形式化协议规范对重构的执行进行验证。该框架以PCAP捕获文件、AnBx规范及相关部署元数据作为输入；从TCP流中提取Java序列化对象，重构协议会话以及信道与步骤之间的映射关系，并生成一个被动入侵者来评估AnBx编译器产生的Dolev-Yao推导检查。本文解决了三个重构挑战：TCP流中包含多个序列化对象和对象引用的问题、交织连接与并发协议会话之间映射的歧义问题，以及语义验证问题。

    arXiv:2611.07170v1 Announce Type: cross  Abstract: Global surveillance systems record large volumes of network traffic for retrospective analysis, but packet captures alone reveal little about protocol-level semantics. This paper presents an automated, specification-guided methodology for reconstructing passive Dolev-Yao intruder executions from captured network traffic and validating the reconstructed executions against a formal protocol specification. The framework takes a PCAP capture, an AnBx specification, and the associated deployment metadata; it extracts Java-serialised objects from TCP streams, reconstructs protocol sessions and channel-to-step assignments, and generates a passive intruder that evaluates the Dolev-Yao derived checks produced by the AnBx compiler. Three reconstruction challenges are addressed: multiple serialised objects and object references within a TCP stream, ambiguity in mapping interleaved connections to concurrent protocol sessions, and semantic validati
    
[^32]: 多步软件工程：编程智能体能否在长时程中持续基于自己编写的代码进行构建？

    Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons?

    [https://arxiv.org/abs/2611.07166](https://arxiv.org/abs/2611.07166)

    提出长时程基准 Multi-step SWE，包含69个由5-15个依赖步骤组成的软件工程任务链，要求智能体在自身编写的代码上持续迭代开发，最强前沿模型的解决率仅18.5%，揭示了当前编程智能体在长时程自我延续开发中的严重不足。

    

    开发者很少只给编程智能体提出单次请求：他们通常要求智能体先构建一个组件，然后对其进行扩展、升级和优化，而每次操作都是基于智能体自己编写的代码。然而，大多数基准测试一次只评估单个任务。我们提出了 Multi-step SWE，这是一个包含69个软件工程任务的长时程基准，每个任务都是在单一代码库上由5-15个相互依赖的步骤组成的链条。每个步骤都是一个完整的多小时工程请求，它从智能体在前几步生成的代码开始，因此早期的设计决策和错误会延续下去。这些任务涵盖四大类工程工作和九种编程语言，既包括从零开始的构建，也包括现有的开源代码库，并且每一步都基于一个既有的真实标准，该标准同时定义了指令本身及其验证器。在九个前沿模型中，表现最强的模型解决率仅为18.5%，69个任务中有38个没有任何模型能够解决。我们发布了 Mult

    arXiv:2611.07166v1 Announce Type: cross  Abstract: Developers rarely give a coding agent a single request: they ask it to build a component and then to extend, upgrade, and optimize it, each time on code the agent itself wrote. Most benchmarks, however, score one task at a time. We introduce Multi-step SWE, a long-horizon benchmark of 69 software engineering tasks, each a chain of 5-15 dependent steps on a single codebase. Every step is a complete, multi-hour engineering request that starts from the code the agent produced in the preceding steps, so early design decisions and errors carry forward. The tasks span four families of engineering work and nine languages, include both from-scratch builds and existing open-source codebases, and ground every step in an existing source of truth that defines both the instruction and its verifier. Across nine frontier models, the strongest model reaches a solve rate of only 18.5%, and 38 of the 69 tasks are not solved by any model. We release Mult
    
[^33]: 用于减少软件问题报告理解过程中信息过载的自动辅助工具

    Automatic Assistance to Reduce Information Overload during Software Issue Report Understanding

    [https://arxiv.org/abs/2611.07109](https://arxiv.org/abs/2611.07109)

    本文提出了iSum工具，能够自动分类和总结软件问题报告中的信息类型，通过开发者视图为单个报告生成可视化摘要、通过经理视图展示项目内各报告的信息类型普遍程度与趋势，帮助开发者缓解问题报告带来的信息过载。

    

    现代问题跟踪系统用于记录问题并跟踪其从产生到关闭的全过程。包含利益相关者多样讨论的问题报告往往篇幅庞大且冗长，使其难以快速分析。我们对开发者的访谈表明，他们在处理问题报告中的多样化信息时面临困难，并期望有自动化技术来分类和总结信息。我们设计了一个名为 iSum（问题摘要器）的工具，可对问题报告中讨论的信息类型进行自动分类和总结。该摘要器提供两种视图：1. 开发者视图：以单个问题报告为输入，输出报告中信息的可视化摘要。2. 经理视图：以给定软件项目下所有问题报告的集合为输入，展示不同信息类型在各报告中的普遍程度和变化趋势。我们通过三项研究对 iSum 进行了评估。1. 在一项针对（原文此处截断）……的实证研究中

    arXiv:2611.07109v1 Announce Type: new  Abstract: Modern issue tracking systems are used to log an issue and track it from its inception to closure. Issue reports with diverse discussions from stakeholders can be large and verbose, making them challenging to analyze quickly. Our developer interviews revealed that they struggled with the diverse information in issue reports and demanded automated techniques to classify and summarize information. We designed a tool, iSum (issue summarizer), that helps with the automatic classification and summarization of information types discussed in issue reports. The summarizer offers two views: 1. Developer View takes an issue report as input and outputs visualized summaries of information in the report. 2. Manager View takes a suite of all issue reports under a given software project as inputs and shows the prevalence and trends of different information types across the reports. We evaluated iSum through three studies. 1. In an empirical study over 
    
[^34]: 遵循政策却未通过基准测试：审计与修订基于政策的决策基准

    Follow the Policy, Fail the Benchmark: Auditing and Revising Policy-Based Decision Benchmarks

    [https://arxiv.org/abs/2611.07042](https://arxiv.org/abs/2611.07042)

    论文审计了九个基于政策的决策基准，发现其中存在参考错误和评分缺陷，仅通过修正这些基准问题即可使模型接受率提升超过80个百分点，表明基准质量对模型评估结果有决定性影响。

    

    基于政策的决策基准用于评估系统是否将书面规则应用于财务和运营案例。然而，当基准测试错误地陈述了所需决策时，遵循规则的回答可能被标记为错误。我们通过来源追溯、程序执行和受控检查，对九个公开的基于政策的决策基准进行了审计。我们发现了无依据的目标值、在重放中依然存在或作为已解决步骤传播的参考错误，以及拒绝可接受答案或给不完整决策加分的评估器。为衡量评分错误的影响，我们在不更改模型回答的前提下重新评分。在96个LegalBench SARA数值案例中，仅纠正评分器就使三个模型的接受率各提高了超过80个百分点，而无需改变参考答案或模型回答。在RuleArena税务任务中，恢复所需的零下限改变了300个参考答案中的27个，并缩小了观察到的模型差距。在一个明确了所有权和资格条件的受控联邦豁免任务中……

    arXiv:2611.07042v1 Announce Type: new  Abstract: Policy-based decision benchmarks evaluate whether systems apply written rules to financial and operational cases. Yet a rule-following response can be marked wrong when the benchmark misstates the required decision. We audit nine public policy-based decision benchmarks through source tracing, program execution, and controlled checks. We find unsupported targets, reference errors that survive replay or propagate as solved steps, and evaluators that reject acceptable answers or credit incomplete decisions. To measure the scoring impact, we rescore unchanged model responses. On the 96 LegalBench SARA numeric cases, correcting the scorer raises acceptance by more than 80 percentage points for each of three models, without changing references or responses. In RuleArena Tax, restoring a required zero floor changes 27 of 300 references and narrows an observed model gap. A controlled federal-exemption task with explicit ownership and eligibility
    
[^35]: 一种由大语言模型调度的神经符号框架，用于验证电子系统中的诊断信号可达性

    An LLM-Orchestrated Neuro-Symbolic Framework for Verifying Diagnostic Signal Reachability in Electronic Systems

    [https://arxiv.org/abs/2611.07040](https://arxiv.org/abs/2611.07040)

    该论文提出NetFlow-DSR，一个由大语言模型调度并结合符号验证的神经符号框架，用于自动化验证电子系统中诊断信号的可达性，克服了传统人工逐信号追踪效率低下以及LLM在结构连接性推理上不可靠的问题。

    

    诊断信号可达性（DSR）描述的是板级健康信号能否被管理控制器观测到，这些控制器包括基板管理控制器（BMC）、复杂可编程逻辑器件（CPLD）以及其他智能控制器。验证DSR对于数据中心机群的现场可诊断性至关重要，因为同一设计会被大规模部署，然而这项工作目前仍主要依赖人工，因为每个信号都需要在原理图页面、重命名的网络、串联组件和监控设备之间手工追踪。大语言模型（LLM）虽然很有吸引力，但在结构连接性推理方面并不可靠，过度宣称可能会悄悄掩盖罕见的覆盖缺口。我们提出了NetFlow-DSR，这是一个神经符号框架，其中LLM负责调度DSR流程，并主要根据器件名称族和库引脚名称来确定器件角色，仅对存在歧义的器件查阅规格说明书。引脚方向性则从设计库中读取。

    arXiv:2611.07040v1 Announce Type: new  Abstract: Diagnostic Signal Reachability (DSR) captures whether board-level health signals are observable by management controllers, including baseboard management controllers (BMCs), complex programmable logic devices (CPLDs), and other smart controllers. Verifying DSR is essential for field diagnosability in datacenter fleets, where one design is deployed at massive scale, yet it remains largely manual, since each signal is traced by hand across schematic sheets, renamed nets, series components, and monitoring devices. Large language models (LLMs) are attractive but unreliable at structural connectivity reasoning, where over-claiming can silently hide a rare coverage gap. We present NetFlow-DSR, a neuro-symbolic framework in which an LLM orchestrates the DSR pipeline and resolves device roles mainly from part-name families and library pin names, consulting specifications only for ambiguous parts. Pin directionality is read from the design librar
    
[^36]: 面向LLM编写工作流的调用级修订契约

    Call-Level Revision Contracts for LLM-Authored Workflows

    [https://arxiv.org/abs/2611.06967](https://arxiv.org/abs/2611.06967)

    该论文为LLM编写的工作流提出了调用级修订契约，通过为每个调用赋予稳定的结构化标识以构成冻结前缀，并配合静态门控、运行时摘要和出现账本三重机制，解决了模型修复失败工作流时重复付费或接受已变更负载的难题。

    

    持久化执行引擎通过重放已记录的历史来运行，且不会比较活动负载。当模型修复一个在中途失败的工作流时，它要么必须为已经成功的调用再次付费，要么接受一个负载已被更改的记录结果。我们为编排代码由模型编写的工作流定义了一种修订契约。每个能力调用都携带一个由其调用位置、扇出位置和出现次数构成的结构化标识，且该标识在保持调用位置不变的改写下保持稳定，因此已提交的标识构成一个冻结前缀。冻结单元是调用表达式本身，而周围的代码保持可写。静态门控会在容器启动之前拒绝任何会改变已提交调用的改写；运行时摘要会在已提交标识处拦截被更改的负载；出现账本则用于标记重复付费的负载。在三个假设条件下，门控所接受的候选项可重放已记录的历史……（摘要在此处截断）

    arXiv:2611.06967v1 Announce Type: new  Abstract: Durable execution engines replay a recorded history and do not compare activity payloads. When a model repairs a workflow that failed partway through, it must either pay again for calls that succeeded or accept a recorded result whose payload changed. We define a revision contract for workflows whose orchestration code is written by a model. Each capability call carries a structural identity built from its callsite, its fanout position, and its occurrence count, and the identity stays stable under rewrites that leave the call in place, so committed identities form a frozen prefix. The frozen unit is the call expression, and the surrounding code stays writable. A static gate rejects a rewrite that would change a committed call before a container starts. A runtime digest stops a changed payload at a committed identity, and an occurrence ledger names a repeated paid payload. Under three assumptions, a candidate the gate accepts replays ever
    
[^37]: 自博弈何时能修复错误信念？执行接地循环中的答案密钥、通过区间与替换规则

    When Can Self-Play Fix a False Belief? Answer Keys, Pass Bands and Replacement Rules in an Execution-Grounded Loop

    [https://arxiv.org/abs/2611.06965](https://arxiv.org/abs/2611.06965)

    在以执行为依据的自我博弈循环中，执行自身代码得到的答案密钥只会复制并固化模型的错误信念，只有当答案密钥来自不含该错误的“教师”实现、且通过区间能够接纳纠正性任务时，自我博弈才能真正修复被植入的错误信念。

    

    在自我博弈循环中，模型编写任务、推导答案密钥，并基于自身生成的匹配解进行训练；当答案密钥来自执行代码时，这类循环常被称为“接地”（grounded）的。我们要问的是：这样的循环能消除哪些错误信念？我们在输出预测任务上使用 Qwen3-4B-Base 运行专家迭代式自我博弈，改变答案密钥的来源，并追踪关于 Python 语义的自然信念，以及通过微调植入的两种对任务规格的误读（把“至多 n”误读为 < n；把“第 k 小”误读为从 0 开始索引）。执行生成器自身代码所得到的答案密钥，在所有受审计任务中都编码了“至多”这一误读，基于它进行训练始终未能将正确率提升到 0.03 以上。而来自未植入该错误的模型（即一位没有此错误的“教师”）所给实现导出的答案密钥，在四次运行中经三轮训练后将其提升至 0.23–0.58。但答案密钥正确并不足够：将任务保持在 8 次尝试中解决 1–7 次的通过区间，几乎不接纳任何具有纠正作用的任务……（原文摘要在此处截断）

    arXiv:2611.06965v1 Announce Type: new  Abstract: Self-play loops in which a model writes tasks, derives an answer key and trains on its own matching solutions are often called grounded when the key comes from executing code. We ask which false beliefs such a loop removes. We run expert-iteration self-play with Qwen3-4B-Base on output-prediction tasks, vary the source of the answer key, and track natural beliefs about Python semantics and two misreadings of task specifications planted by fine-tuning ("at most n" read as < n; "the k-th smallest" read as 0-indexed). Executing the generator's own code gave a key that encoded the "at most" misreading on every audited task, and training on it never raised the truth rate above 0.03. A key from an implementation by the unplanted model, a teacher without the error, raised it to 0.23-0.58 after three rounds in four runs. A correct key is not sufficient: the pass band, which keeps tasks solved 1-7 times of 8, admits almost no corrective task whil
    
[^38]: 在MLOps与机器学习生产中落地负责任人工智能：一项系统性文献综述

    Operationalizing Responsible AI in MLOps and ML Production: A Systematic Literature Review

    [https://arxiv.org/abs/2611.06961](https://arxiv.org/abs/2611.06961)

    本文通过系统性文献综述发现，负责任AI在MLOps和ML生产中正从抽象伦理原则转变为具体的工程机制，通过漂移监测、运行时检查、审计日志、治理门和策略即代码等技术手段落地实施。

    

    启用人工智能（AI）的系统越来越多地部署于实际运行环境中，其中机器学习（ML）模型需要与不断变化的数据、用户、基础设施和组织流程进行交互。尽管负责任AI（Responsible AI）原则已被广泛讨论，但这些原则如何在机器学习运维（MLOps）和ML生产工作流中被转化为具体机制，目前仍知之甚少。本文报告了一项针对在ML生产中落地负责任AI相关研究的系统性文献综述。该综述综合了监测与运行时保障、审计与合规、ML生命周期集成以及可追溯性/来源记录/文档化等方面的证据。在最终的研究集合中，文献显示出从抽象伦理原则向工程机制转变的趋势，例如漂移监测、运行时检查、审计日志、来源记录、证据包、治理门、策略即代码等机制。

    arXiv:2611.06961v1 Announce Type: new  Abstract: Artificial intelligence (AI)-enabled systems are increasingly deployed in operational environments where machine learning (ML) models interact with changing data, users, infrastructure, and organizational processes. While Responsible AI principles are widely discussed, less is known about how these principles are translated into concrete mechanisms within Machine Learning Operations (MLOps) and ML production workflows. This paper reports a systematic literature review of studies that operationalize Responsible AI in ML production. The review synthesizes evidence across monitoring and runtime assurance, audit and compliance, ML lifecycle integration, and traceability/provenance/documentation. Across the final study set, the literature shows a shift from abstract ethical principles toward engineering mechanisms such as drift monitoring, runtime checks, audit logs, provenance records, evidence bundles, governance gates, policy-as-code, mode
    
[^39]: 对Google AI Studio生成的氛围编码数据库软件应用进行递归测试与氛围调试

    Recursively Testing and Vibe Debugging a Google AI Studio Generated Vibe-Coded Database Software Application

    [https://arxiv.org/abs/2611.06952](https://arxiv.org/abs/2611.06952)

    该论文提出了扩展的氛围编程定义与氛围调试概念，并给出一种可靠、完备且最优的算法方法论，证明了即使是使用非英语语言的非开发者，也能通过递归测试与氛围调试将Google AI Studio生成的软件应用迭代至无错误且功能满意的版本。

    

    这篇研究论文提出了氛围编程的更广泛定义，将原有定义扩展至非开发者以及英语以外的自然语言，并引入了氛围调试的概念，以及一种可靠、完备且最优的算法方法论，用于测试和氛围调试氛围编码的软件应用。该方法论通过应用于一个由Google AI Studio Pro生成的中等复杂度的真实世界氛围编码数据库应用来加以说明。我们证明了即使是非开发者也能获得最终无错误且功能令人满意的应用版本，即使使用英语以外的语言也是如此。我们还证明了，尽管Google AI Studio Pro拥有用于语言分析、软件开发和自动化测试的复杂内置引擎，它仍然会犯很多错误，因此对其生成的代码进行测试和氛围调试是必不可少的。

    arXiv:2611.06952v1 Announce Type: cross  Abstract: This research paper proposes a broader definition of vibe coding, extending the original one with non-developers and natural languages other than English, and introduces the notion of vibe debugging, as well as a sound, complete, and optimal algorithmic methodology for testing and vibe debugging vibe-coded software applications. This methodology is illustrated by applying it to a vibe-coded real-world database application of medium complexity generated with Google AI Studio Pro. We prove that even non-developers can obtain a final bug-free and functional satisfactory application version, even in languages other than English. We also prove that, although Google AI Studio Pro has sophisticated built-in engines for both language analysis, software development, and automated testing, it is still making lot of mistakes, so testing and vibe debugging of its generated code is a must.
    
[^40]: 面向更经济CI/CD的KG-AST：预算化测试选择与AI缺陷审查

    KG-AST for Cheaper CI/CD: Budgeted Test Selection and AI Bug Review

    [https://arxiv.org/abs/2611.06950](https://arxiv.org/abs/2611.06950)

    KG-AST利用git历史和代码图特征对提交进行风险评分，优先将测试与AI审查预算分配给最可能引入缺陷的提交，在11种语言的155个开源项目上显著优于随机选择和最大变更优先基线。

    

    持续集成与AI辅助审查在每个提交上花费相同的预算，尽管只有少数提交会引入缺陷。我们研究了KG-AST，它在运行重型测试或AI审查之前，基于git历史和代码图特征为每个提交进行风险评分，从而使预算优先分配给风险最高的提交，并且只向AI审查者提供变更周围的图切片。在11种语言的155个开源项目上，一个在其他项目上训练的单一跨语言模型对提交进行排序，使得对风险最高的20%提交进行重型测试即可覆盖49-64%的引入缺陷的提交（因语言而异），相比之下随机选择仅覆盖20%，优先测试最大变更的策略覆盖40-55%。每种语言都比最大变更基线高出3.8至13.2个百分点。在20个Java项目中，20%预算下的增益为15.1个百分点（64.6%对49.5%，95%置信区间为+11.6至+18.4），且模型在20个项目中的19个项目上获胜。大约十二个训练项目就足够了。

    arXiv:2611.06950v1 Announce Type: cross  Abstract: Continuous integration and AI-assisted review spend the same budget on every commit, although few commits introduce bugs. We study KG-AST, which scores each commit risk from git history and code-graph features before heavy tests or AI review run, so the budget goes to the riskiest commits first, and which hands the AI reviewer only a graph slice around the change. On 155 open-source projects in 11 languages, a single cross-language model trained on other projects ranks commits so that heavy testing of the riskiest 20% covers 49-64% of bug-introducing commits depending on the language, against 20% for random selection and 40-55% for testing the biggest change first.   Every language beats the biggest-change baseline, by 3.8 to 13.2 points. In 20 Java projects the gain at a 20% budget is 15.1 points (64.6% vs 49.5%, 95% interval +11.6 to +18.4), and the model wins in 19 of the 20 projects. About twelve training projects suffice; richer f
    
[^41]: 何时组合软件工作量估计才有帮助？来自四个公开项目数据集的证据

    When Does Combining Software-Effort Estimates Help? Evidence from Four Public Project Datasets

    [https://arxiv.org/abs/2611.06948](https://arxiv.org/abs/2611.06948)

    本文在四个公开软件工作量数据集上系统比较五种组合规则后发现，组合估计并非总是有益——当某一估计（尤其是人工估计）异常准确或各成员误差高度相关时，集成反而会稀释准确性或重复共同误差。

    

    组合估计可以降低单个预测的误差，但集成也可能稀释一个异常准确的估计器，或重复其成员共同犯下的错误。我们在四个历史软件工作量数据集（Kitchenham：145个项目、Desharnais：77个、China：499个、Maxwell：63个）上研究这些相互竞争的效应，每个数据集采用100次重复的70/30训练与测试划分。三种传统模型（参数回归、类比估计和随机森林）相互组合，并在Kitchenham数据集中与已记录的人工估计相结合。五种组合规则由折外校准误差驱动：等权重加权、逆MSE加权、非负单纯形组合、带符号的Bates-Granger加权，以及带协方差收缩的Bates-Granger加权。所有预处理均在训练折内学习。模型误差相关性介于0.54至0.97之间。在Kitchenham数据集中，人工估计占主导地位：等权重组合使平均测试log-MAE从0.239上升到0.4……

    arXiv:2611.06948v1 Announce Type: new  Abstract: Combining estimates can reduce individual prediction errors, but an ensemble may also dilute an unusually accurate estimator or repeat errors shared by its members. We study these competing effects on four archival software-effort datasets (Kitchenham: 145 projects, Desharnais: 77, China: 499, Maxwell: 63) with 100 repeated 70/30 training and test partitions per dataset. Three conventional models (parametric regression, analogy, and random forest) are combined with each other and, in Kitchenham, with a recorded human estimate. Out-of-fold calibration errors drive five combination rules: equal weighting, inverse-MSE weighting, a nonnegative simplex combination, signed Bates-Granger weighting, and Bates-Granger with covariance shrinkage. All preprocessing is learned within training folds. Model-error correlations range from 0.54 to 0.97. In Kitchenham, the human estimate dominates: equal weighting raises mean test log-MAE from 0.239 to 0.4
    
[^42]: Python音频库在版本发布间的变化：模块结构与文档

    How Python Audio Libraries Change Across Releases: Module Structure and Documentation

    [https://arxiv.org/abs/2611.06947](https://arxiv.org/abs/2611.06947)

    该研究首次将Python音频库的模块结构演化与发布文档进行联合挖掘分析，通过对49个软件包1,395个发布版本的导入图及20,864条语言模型标注的发布说明的大规模实证研究发现：顶级模块结构高度稳定（86.1%的连续发布中不变）、主版本号递增引发结构变化的频率是其他递增的5.2倍，但发布说明仅覆盖约8.9%的模块移除事件。

    

    arXiv:2611.06947v1 公告类型：新论文 摘要：升级Python音频库的开发者需要了解其导入的代码发生了哪些变化，以及发布说明中是否对此有所记载。针对一般软件的发布说明和破坏性变更已有挖掘研究，但音频库的模块结构及其文档尚未被共同研究。我们挖掘了56个用于机器学习工作流的音频仓库：涵盖49个软件包在1,395个发布版本中的导入依赖图，并将其与20,864条发布说明和变更日志条目相关联，这些条目由三个开源权重语言模型进行标注，并由两名人类编码员进行校验。我们发现：(1) 在86.1%的连续发布版本对中，软件包的顶级模块集合保持不变，而模块文件集合在61.4%的连续发布版本对中保持不变；(2) 主版本号递增导致顶级模块发生变化的频率是其他类型递增的5.2倍，主版本号递增与0.x次版本号递增合计占所观察到变化的48.7%；(3) 当某个发布版本移除模块时，其发布说明中仅提及了8.9%的（被移除模块，摘要在此处截断）。

    arXiv:2611.06947v1 Announce Type: new  Abstract: Developers who upgrade a Python audio library need to know what changed in the code they import and whether the release notes say so. Release notes and breaking changes have been mined for general software, but the module structure of audio libraries and its documentation have not been studied together. We mine 56 audio repositories used in machine learning workflows: the import graph of 49 packages at 1,395 releases, joined to 20,864 release-note and changelog entries, labeled by three open-weight language models and checked against two human coders. We find that (1) the set of a package's top-level modules is unchanged in 86.1% of consecutive release pairs, while the set of module files is unchanged in 61.4%. (2) A major increment changes the top level 5.2 times as often as another increment, and major and 0.x minor increments together account for 48.7% of the observed changes. (3) When a release removes modules, its notes name 8.9% of
    
[^43]: 超越智能体框架：软件工厂编排的边界判据——一项单操作员案例研究

    Beyond the Agent Harness: Boundary Criteria for Software-Factory Orchestration - A Single-Operator Case Study

    [https://arxiv.org/abs/2611.06946](https://arxiv.org/abs/2611.06946)

    该论文的核心贡献是提出了一套基于“运行”操作性定义的边界判据，用以明确判定软件工厂编排中的机制与故障究竟归属于智能体框架、工厂框架、领域层还是基础设施/模型层。

    

    近年来出现的若干系统在编码智能体框架之上增加了一个层次，将众多智能体运行组织为持续交付。这一层次的存在本身并非我们的贡献；所缺失的，是一条用于判定某个机制或故障应归属于哪一层的规则。我们基于对“运行”的操作性定义提出了边界判据：局限于单次运行的机制属于智能体框架；协调多次运行、将状态从一次运行传递至后续运行、或依据运行结果进行更新的机制属于工厂框架；必须随应用领域而改变的机制属于领域层。另有两项残余测试用于归派基础设施与模型方面的原因。对于故障，我们将缺陷起源之处、哪个控制环节未能阻止它、以及它最终在哪里被修复编码为三个相互独立的字段，并且在将故障归因于环境之前要求有明确的正面证据。我们在来自单一操作员的已记录故障数据上开发了这些判据。

    arXiv:2611.06946v1 Announce Type: new  Abstract: Several recent systems add a layer above the coding-agent harness that organises many agent runs into continuous delivery. The existence of such a layer is not our contribution. What is missing is a rule for deciding which layer a mechanism or a failure belongs to. We propose boundary criteria based on an operational definition of a run. A mechanism confined to one run belongs to the agent harness. One that coordinates runs, hands state from one run to a later one, or is updated from run outcomes belongs to a factory harness. One that must change with the application domain belongs to the domain layer. Two residual tests assign infrastructure and model causes. For failures, we code where the defect originated, which control failed to stop it, and where it was fixed as three separate fields, and we require positive evidence before attributing a failure to the environment. We developed the criteria on documented failures from a single oper
    
[^44]: 一种用于物理GUI交互测试的多模态机器人框架

    A Multi-Modal Robotic Framework for Physical GUI Interaction Testing

    [https://arxiv.org/abs/2611.06939](https://arxiv.org/abs/2611.06939)

    该论文提出了RFNIT，一个多模态、完全非侵入式的机器人框架，通过视觉感知模块、LLM推理代理和机器人执行器，实现了无需系统级权限的智能手机GUI物理交互端到端自动化测试。

    

    当需要重现真实的用户交互，并且在不依赖插桩或ADB级控制等侵入性机制的情况下验证GUI行为时，移动应用程序的自动化测试仍然具有挑战性。现有方法（包括DLD和R-DLD）能够检测由活动重启触发的数据丢失，但仍然依赖于系统级访问权限，这限制了它们在真实设备上的适用性。本博士研究提出了RFNIT，一个多模态、完全非侵入式的机器人框架，用于智能手机上的物理GUI交互测试。RFNIT通过外部感知来感知设备状态并执行物理交互，从而支持端到端自动化，在实现类人操作的同时保持可重复性和执行精度。该框架集成了基于视觉的感知模块、LLM驱动的推理代理以及能够执行触摸、滑动和打字手势的机器人执行器。

    arXiv:2611.06939v1 Announce Type: new  Abstract: Automated testing of mobile applications remains challenging when realistic user interactions must be reproduced and GUI behavior verified without relying on invasive mechanisms such as instrumentation or ADB-level control. Existing approaches, including DLD and R-DLD, detect data loss triggered by activity restarts but still depend on system-level access, which limits their applicability on real devices.   This doctoral research proposes RFNIT, a multi-modal, fully non-invasive robotic framework for physical GUI interaction testing on smartphones. RFNIT supports end-to-end automation by perceiving the device state through external sensing and executing physical interactions, enabling human-like actions while preserving reproducibility and execution accuracy.   The framework integrates a vision-based perception module, an LLM-driven reasoning agent, and a robotic actuator capable of performing touch, swipe, and typing gestures. Multimoda
    
[^45]: 重放是一族断言：事件溯源式智能体运行时中的分叉安全性

    Replay Is a Family of Assertions: Fork-Safety in Event-Sourced Agent Runtimes

    [https://arxiv.org/abs/2611.06924](https://arxiv.org/abs/2611.06924)

    本文提出一个将事实与效果请求分离、使调度显式化的可执行 F# 契约，证明事件溯源式智能体运行时中的确定性重放、低成本分叉与谱系是三个相互独立的断言而非单一保证，并通过反例与大规模日志验证揭示了一般性结算既不终止也不合流。

    

    事件溯源式智能体运行时通常声称具备确定性重放、低成本分叉和谱系追踪能力，但这些实际上是一族断言，而非单一保证。固定的日志折叠可能是确定性的，而响应式结算仍然对调度顺序敏感；记录下来的预言机结果可以实现零调用重放，而原始交互却已经改变了成本或世界状态；前缀副本无法撤销被丢弃后缀中的动作。我们提出了一个可执行的 F# 契约，它将事实与效果请求分离，使调度显式化，并相对于投影、严格执行重放、外部延续或声明的反事实世界观察来评估截断。保留的反例表明，一般性结算既不具终止性也不具合流性，且互不相交的声明写入并不能防止读/触发干扰。在 5,540 个公开运行时日志的实验中，全部 317,296 个投影截断均满足所述观察，而 36,251 个截断……

    arXiv:2611.06924v1 Announce Type: cross  Abstract: Event-sourced agent runtimes commonly claim deterministic replay, cheap forking, and lineage. These are families of assertions, not one guarantee. A fixed-log fold may be deterministic while reactive settlement remains schedule-sensitive; a recorded oracle result can enable zero-call replay while the original interaction still changed cost or world state; and a prefix copy cannot undo an action in a discarded suffix.   We present an executable F# contract that separates facts from effect requests, makes scheduling explicit, and assesses a cut relative to projection, strict execution replay, external continuation, or a declared counterfactual-world observation. Preserved counterexamples show that general settlement is neither terminating nor confluent and that disjoint declared writes do not prevent read/trigger interference. Against 5,540 public runtime logs, all 317,296 projection cuts satisfy the stated observation, while 36,251 cuts
    
[^46]: 通过人物角色模拟真实用户实现以用户为中心功能覆盖的全面移动应用测试

    Towards Comprehensive Mobile Application Testing for User-centric Feature Coverage via Simulating Real-World Users with Personas

    [https://arxiv.org/abs/2611.06922](https://arxiv.org/abs/2611.06922)

    该论文提出了由3,012项用户中心功能数据集和自动化评估流水线构成的“以用户为中心功能覆盖”基准，并通过人物角色模拟真实用户，构建了比传统活动覆盖率更能反映用户真实需求的全面移动应用测试框架。

    

    大语言模型智能体日益被用于移动应用的黑盒测试，其最终目标是全面覆盖用户所依赖的功能。然而，现有研究大多聚焦于“活动覆盖率”，即衡量测试会话中访问的不同应用界面所占的比例，但这并不能直接衡量智能体是否真正测试了用户关心的功能。为解决这一问题，我们聚焦于：(1) 如何自动衡量一个测试框架是否覆盖了用户关心的功能；(2) 如何设计一个能够实现更全面功能覆盖的框架。针对问题(1)，我们提出了“以用户为中心的功能覆盖”基准，该基准由两个部分组成：一个基于真实用户评论和应用描述手工构建的、包含3,012项用户中心目标功能的数据集，以及一个自动化评估流水线，该流水线将智能体生成的测试用例与目标功能转换到共享的语义空间……

    arXiv:2611.06922v1 Announce Type: new  Abstract: LLM agents are increasingly used for black-box mobile application testing, where the ultimate goal is the comprehensive coverage of the features users rely on. However, most existing studies focus on Activity Coverage, which measures the proportion of distinct application screens visited during a test session but does not directly measure whether agents actually test the features users care about. To address this, we focus on (1) how to automatically measure whether a testing framework covers the features users care about, and (2) how to design a framework that achieves more comprehensive feature coverage. To answer (1), we introduce the User-Centric Feature Coverage benchmark, which consists of two components: a manually crafted dataset of 3,012 user-centric target features based on real reviews and app descriptions, and an automated evaluation pipeline that translates agent-generated test cases and target features into a shared semanti
    
[^47]: TimeArch：一种用于软件架构设计的多智能体方法

    TimeArch: A Multi-Agent Approach for Software Architecture Design

    [https://arxiv.org/abs/2611.06921](https://arxiv.org/abs/2611.06921)

    TimeArch提出了一种多智能体方法，将软件架构设计结构化为受治理的生命周期，从而在GenAI编码工作流中显式填补需求与代码之间缺失的架构层。

    

    生成式AI正在让软件实现变得更快速，但不一定带来更好的架构。当前大多数GenAI编码工作流直接从提示词跳到代码，而本应连接需求、质量属性、利益相关者、设计决策与实现之间的架构推理仍然缺乏充分支持。架构风格、系统分解、权衡分析、质量属性决策以及利益相关者关注点往往被隐式处理，或在AI支持的工作流之外进行处理。这在需求与代码之间造成了一个缺失的架构层，限制了GenAI赋能软件开发中的可追溯性、验证与治理。本文提出了TimeArch，这是一种使该架构层显式化的多智能体方法。TimeArch将软件架构设计结构化为一个受治理的生命周期，而非一次性生成任务。它将基于智能体的推理与检索增强……（原文摘要到此截断）

    arXiv:2611.06921v1 Announce Type: new  Abstract: Generative AI is making software implementation faster, but not necessarily better architected. Most current GenAI coding workflows move directly from prompts to code, while the architectural reasoning that should connect requirements, quality attributes, stakeholders, design decisions, and implementation remains weakly supported. Architectural styles, system decomposition, tradeoff analysis, quality-attribute decisions, and stakeholder concerns are often left implicit or handled outside the AI-supported workflow. This creates a missing architectural layer between requirements and code, limiting traceability, validation, and governance in GenAI-enabled software development.   This paper presents TimeArch, a multi-agent approach that makes this architectural layer explicit. TimeArch structures software architecture design as a governed lifecycle rather than a one-shot generation task. It combines agent-based reasoning with retrieval-augme
    
[^48]: LIBRA：可执行工具库的程序化归纳

    LIBRA: Programmatic Induction of Executable Tool Libraries

    [https://arxiv.org/abs/2611.05445](https://arxiv.org/abs/2611.05445)

    LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。

    

    大型语言模型智能体越来越多地从庞大的工具目录中检索工具，然而大多数检索器仅根据请求相似度对每个工具进行独立排序。对于多工具任务，这种方式只能召回与目标匹配的工具，而遗漏了工作流程所需的先决工具。我们提出 LIBRA——一个通过运行时组装进行工具库归纳的框架，它将扁平的工具目录转化为可执行的工具库。大语言模型为每个工具提出契约，包括候选调用、可复用输出与状态变更以及成功条件；LIBRA 随后在沙盒环境中测试这些组合，并保留在运行时得到验证支持的工具间关系。该工具库在用户查询之前构建完成，无需依赖关系标注、工具使用轨迹或任务特定训练，同时支持一次性检索和智能体式检索。在 AppWorld、BFCL 和 SciToolEval 三个基准上，相比相同的稠密检索器，LIBRA 将一次性检索的 Recall 和 NDCG 平均提升了 7.9 和 3.7 个百分点。

    arXiv:2611.05445v1 Announce Type: cross  Abstract: Large language model agents increasingly retrieve tools from large catalogs, yet most retrievers rank each tool independently by request similarity. For multi tool tasks, this recovers the goal matching tool while omitting prerequisite tools required by the workflow.We introduce LIBRA, a framework for Library Induction by Runtime Assembly that turns a flat tool catalog into an executable tool library. An LLM proposes per tool contracts of candidate invocations, reusable outputs and state changes, and success conditions; LIBRA then tests these compositions in a sandbox and retains the relations supported at runtime. Constructed before user queries, the library requires no dependency annotations, tool use trajectories, or task specific training, and supports both one-shot and agentic retrieval. On AppWorld, BFCL, and SciToolEval, LIBRA improves one-shot Recall and NDCG over the same dense retriever by 7.9 and 3.7 percentage points on ave
    
[^49]: 上下文的物理代价：面向编码代理的质量约束资源核算

    The Physical Cost of Context: Quality-Constrained Resource Accounting for Coding Agents

    [https://arxiv.org/abs/2610.34763](https://arxiv.org/abs/2610.34763)

    该论文提出了一种质量门控的资源核算协议，指出编码代理上下文策略的真实成本取决于调用次数和前缀缓存复用而非仅仅提示词长度，并证明预加载检索块（Eager）策略在保证代码质量的前提下可减少35.7%的模型调用和33.2%的GPU能耗。

    

    编码代理在每次模型调用时都会重新发送不断增长的上下文，而上下文策略通常仅以节省的提示词token数量来评判。但实际执行的工作量还取决于调用次数和前缀复用：启用缓存的运行时只会预填充最长已缓存前缀之后的token。因此，一个更短的提示词如果重写了已缓存的历史，反而可能代价更高。我们提出的质量门控协议将实际执行的预填充量拆分为“逻辑提示词减去复用前缀”，并测量KV缓存驻留、时间和能耗。在基于SWE-rebench V2的450个定位案例中，预加载十个检索块的策略（Eager）以及在此基础上增加锚定式历史压缩的策略（Compact）均通过了五点非劣效性质量门槛。在另一项900单元的实验中，与按需搜索（Grep）相比，Eager将模型调用次数减少了35.7%，GPU能耗代理指标减少了33.2%；而Compact未带来进一步节省。启用前缀缓存后，Eager的提示词总量基本持平（+8.3%；其95%置信区间包含零），但其未缓存预填充量降低了（原文在此处截断）

    arXiv:2610.34763v1 Announce Type: cross  Abstract: Coding agents resend a growing context on every model call, and context policies are usually judged by the prompt tokens they save. Executed work also depends on call count and prefix reuse: a caching runtime prefills only the tokens after the longest cached prefix. A shorter prompt can therefore cost more if it rewrites cached history. Our quality-gated protocol splits executed prefill into logical prompt minus reused prefix and measures KV residency, time, and energy. On 450 localization cases derived from SWE-rebench V2, preloading ten retrieved blocks (Eager) and adding anchored history compaction (Compact) both pass a five-point non-inferiority gate. In a separate 900-cell study, Eager cuts model calls by 35.7% and the GPU-energy proxy by 33.2% against on-demand search (Grep); Compact saves nothing further. Enabling prefix caching leaves Eager's prompt volume similar (+8.3%; its 95% CI spans zero) but cuts its uncached prefill by 
    
[^50]: RetryBench：在GPT工具使用中将重试许可与恢复结果相分离

    RetryBench: Separating Retry Permission from Recovery Outcomes in GPT Tool Use

    [https://arxiv.org/abs/2610.15884](https://arxiv.org/abs/2610.15884)

    该论文提出RetryBench基准，将GPT工具使用中“虚假重试许可”（即在契约不支持时重试超时工具调用）这一错误与实际恢复结果分离，发现即使在受保护条件下能成功恢复，模型也无法可靠地拒绝不受支持的重试。

    

    虚假重试许可（FRP）是指智能体在可见契约不支持的情况下重试超时的工具调用。RetryBench将这一错误与实际实现的恢复分离开来。成功的受保护恢复并不意味着能够可靠地拒绝不受支持的重试。在一项720次响应的密钥对照实验中，有效密钥通常能实现安全推进，但被提供的已更改密钥在Sol（1/80）、Terra（55/80）和Luna（60/80）中引发了不受支持的重试。随后的一项240次响应对照实验移除了候选密钥并要求模型自行提供密钥，Luna仍进行了8/80次不受支持的调度；Sol和Terra在各自80次响应中均未出现此类情况。因此，接受显示的错误密钥无法解释所有观察到的错误，尽管更改的输出要求和条件使得纯粹的显示效应比较无法进行。在一项六终点研究中，移除弱契约标签增加了Sol和Terra在编码动作条件下的重试次数。另一项独立的1,536次响应的规范动作……（原文摘要在此处截断）

    arXiv:2610.15884v1 Announce Type: new  Abstract: False retry permission (FRP) occurs when an agent retries a timed-out tool call without support from the visible contract. RetryBench separates this error from realized recovery. Successful protected recovery does not imply reliable rejection of unsupported retries. In a 720-response key control, valid keys usually enabled safe progress, but offered changed keys elicited unsupported retries in Sol (1/80), Terra (55/80), and Luna (60/80). A later 240-response control removed the candidate and required models to supply a key. Luna still made 8/80 unsupported dispatches; Sol and Terra made none in 80 responses each. Accepting a displayed bad key therefore cannot explain all observed errors, although changed output requirements and conditions preclude a pure display-effect comparison. Removing the weak-contract label increased retries in Sol and Terra's coded-action conditions in a six-endpoint study. A separate 1,536-response canonical-acti
    
[^51]: 可执行但不可信：基于大语言模型与编码智能体的模拟对象、桩与断言生成

    Executable but Unfaithful: Mock Object, Stub, and Assertion Generation by LLMs and Coding Agents

    [https://arxiv.org/abs/2610.14629](https://arxiv.org/abs/2610.14629)

    该论文提出了首个系统性基准MockBench（含165个来自真实Java与Python项目的任务），全面评估了大语言模型与编码智能体生成模拟对象、桩和模拟断言三类制品的能力，并发现其生成的代码虽可执行但往往与真实依赖行为不一致。

    

    模拟是单元测试中的一项基础实践，它使开发者能够通过用模拟替代品替换真实依赖来隔离被测代码。自动化生成模拟对象、桩和模拟断言一直是一个长期的研究目标。然而，捕获-重放方法依赖于记录真实依赖的运行时行为；诸如StubCoder之类的演化方法虽然克服了这一要求，却将生成限制在预定义的符号池中。大语言模型（LLM）和编码智能体提供了一种替代方案：它们可以直接从源代码生成模拟制品，而无需记录依赖的执行轨迹或固定的符号池。然而，目前尚无系统性研究对它们在全部三类制品上的能力进行评估。为填补这一空白，我们构建了MockBench——一个包含来自真实Java和Python项目的165个模拟制品生成任务的基准数据集，并配有经过验证的参考（摘要在此处被截断）。

    arXiv:2610.14629v2 Announce Type: replace  Abstract: Mocking is a foundational practice in unit testing, enabling developers to isolate the code under test by replacing real dependencies with simulated alternatives. Automating the generation of mock objects, stubs, and mock assertions has been a long-standing research goal. However, capture-and-replay approaches rely on recording the runtime behavior of actual dependencies. Evolutionary approaches such as StubCoder overcome this requirement but constrain generation to predefined symbol pools. Large language models (LLMs) and coding agents offer an alternative by generating mocking artifacts from source code without requiring recorded dependency executions or fixed symbol pools. Yet, no systematic study has evaluated their capabilities across all three artifact types.   To address this gap, we construct MockBench, a benchmark comprising 165 mock artifact generation tasks from real-world Java and Python projects, with validated reference
    
[^52]: 重构 React 组件层次结构以消除属性钻取

    Refactoring React Component Hierarchies to Eliminate Prop Drilling

    [https://arxiv.org/abs/2610.01376](https://arxiv.org/abs/2610.01376)

    本文提出一种静态分析方法并实现为命令行工具 ReactRefactor，可自动识别 React 代码库中的属性钻取问题，并利用基于 Context API 和组件组合的两种自动化重构策略将其消除，实证评估表明属性钻取在各规模的开源项目中普遍存在。

    

    在 React 前端开发中，属性钻取是指在组件层次结构的多个层级之间通过组件属性逐层传递数据的做法。尽管 React 官方文档不推荐这种做法，且近期研究将其归类为一种代码坏味道，但人们对其普遍性、复杂性以及自动化重构的潜力仍知之甚少。在本工作中，我们提出了一种静态分析方法，用于识别 React 代码库中的属性钻取实例，并采用基于 Context API 和组件组合的两种自动化重构策略来消除它们。所提出的方法被实现为一个名为 ReactRefactor 的 Node.js 命令行工具，并在一个开源 React 应用程序基准数据集上进行了实证评估。实证评估的主要结果表明：(a) 属性钻取在各基准项目中频繁出现，且与项目规模无关；(b) 其复杂度普遍处于低到中等水平

    arXiv:2610.01376v1 Announce Type: new  Abstract: In React front-end development, Prop Drilling is the practice of propagating data through component properties across multiple levels of the component hierarchy. Despite being discouraged by React documentation and characterized as a code smell by recent research, little is known about its prevalence, complexity, and potential for automated refactoring. In this work, we propose a static analysis method for identifying Prop Drilling instances in a React codebase and eliminating them using two automated refactoring strategies based on Context API and Component Composition. The proposed method is implemented as a Node.js command line tool, ReactRefactor, and empirically evaluated on a benchmark dataset of open-source React applications. The main findings of the empirical evaluation indicate (a) the frequent occurrence of prop drillings across benchmark projects, irrespective of project size; (b) their generally low to moderate complexity in
    
[^53]: SaltBench：用于在机器校验软件工作中测量方法效应的裁判门控协议

    SaltBench: A Referee-Gated Protocol for Measuring Method Effects in Machine-Checked Software Work

    [https://arxiv.org/abs/2609.11076](https://arxiv.org/abs/2609.11076)

    提出了SaltBench，一个由机器裁判门控的基准测试协议，通过严格的隔离机制（经探针实际检验）、预注册预测和“停止即暂停”等规则，使机器裁判对编程智能体工作方式的影响变得可测量且不可事后叙述。

    

    SaltBench 是一个基准测试协议，旨在回答一个问题：机器裁判如何改变编程智能体的工作方式？机器裁判——即证明内核、程序验证器或保留的测试套件——决定智能体工作的价值，而智能体无法与其争辩。我们在此报告一种协议，使裁判的效应可被测量，且其结果无法在事后被主观叙述：每个结果都由智能体自身工具链之外裁决；智能体与网络、参考解决方案及测试框架本身被完全隔离，并在任何计分运行之前，通过试图突破隔离的探针来检验隔离墙，因此这种隔离是被观察到的而非被假设的；每次运行都由带有注册预测的日期化冻结授权；预算停止意味着暂停，而绝非失败。在本研究中，基准测试的对象是一个“席位”（seat），即标准测试框架中的一次智能体会话。我们测试了五个系统组件，全部由……

    arXiv:2609.11076v1 Announce Type: new  Abstract: SaltBench is a benchmark protocol for one question: How does a machine referee change the way a coding agent works? A machine referee --- a proof kernel, a program verifier, or a withheld test suite --- decides what an agent's work is worth, and the agent cannot argue with it. Here we report a protocol that makes the referee's effect measurable and whose answers cannot be narrated afterwards: every outcome is decided outside the agent's own toolchain; the agent is walled off from the network, the reference solutions and the harness itself, and the wall is tested by probes that try to breach it before any scored run, so the isolation is observed rather than assumed; every run is authorized by a dated freeze with its predictions registered; and a budget stop is a halt, never a failure. In this study, the subject of the benchmark is a ``seat'', meaning an agent session in its standard harness. We tested five systems components, all authored
    
[^54]: 软件开发中的氛围编程：一项多声部文献综述

    Vibe Coding in Software Development: A Multivocal Literature Review

    [https://arxiv.org/abs/2607.21652](https://arxiv.org/abs/2607.21652)

    这是首个在统一协议下对“氛围编程”进行的多声部文献综述，综合分析了266个学术与从业者来源（76个同行评审、190个灰色文献），围绕八个研究问题系统梳理了这一AI辅助开发方法的现有证据。

    

    氛围编程是一种AI辅助的软件开发方法，开发人员以自然语言表达意图，由大型语言模型生成代码。据我们所知，目前尚无综述在统一协议下对学术文献与从业者来源进行综合。我们对2022年1月至2026年9月9日的出版物开展了多声部文献综述，文献通过2025年10月和2026年9月的检索确定。在评估来源质量与可信度后，我们围绕八个研究问题分析了266个来源（76个同行评审文献、190个灰色文献）：其中146个为直接相关，93个为相邻相关，27个为背景资料。存在性发现与结果性发现分别被赋予报告支持标签和证据强度标签。第二作者进行了十二次核对；其中部分核对在调和前的一致性较低，但最终直接性判定规则的一致性较高，而结果字段的一致性为中等，这些字段均经过完整核对。文献来源描述了一种生成-评估……（原文摘要在此处截断）

    arXiv:2607.21652v2 Announce Type: replace  Abstract: Vibe coding is an AI-assisted software development approach in which developers express intent in natural language and large language models generate code. To our knowledge, no review has synthesized academic and practitioner sources under a common protocol. We conducted a multivocal literature review of publications from January 2022 to 9 September 2026, identified through searches in October 2025 and September 2026. After assessing source quality and credibility, we analyzed 266 sources (76 peer-reviewed, 190 grey) across eight research questions: 146 were direct, 93 adjacent, and 27 background. Presence and outcome findings receive reporting-support and evidence-strength labels, respectively. The second author conducted twelve checks; pre-reconciliation agreement was low in several, but substantial for the final directness rule and moderate for the outcome fields, which were checked in full. Sources describe a generation-evaluatio
    
[^55]: 量子软件实证比较的审计

    Auditing Empirical Comparisons in Quantum Software

    [https://arxiv.org/abs/2607.00516](https://arxiv.org/abs/2607.00516)

    提出了CLAIMSTAB-QC审计框架，通过在计算结果前锁定审计设计、限定证据来源边界，将量子软件论文中的455条实证比较声明分类为“维持”、“未决”或“未保留”，从而检验比较结论是否有证据支撑。

    

    量子软件实证论文常常声称某个编译器能生成更小的电路、某个优化器更可靠、或某个拟设能获得更好的解质量。这类比较依赖于基准测试范围、电路构建、编译、采样、后端、噪声模型、优化器选择和资源预算等因素。现有的测试、基准测试和可复现性方法有助于评估程序和实验，但并不能直接审计所报告的比较结论是否得到了论文与工件中可用证据的支持。我们提出了 CLAIMSTAB-QC，这是一个以来源为边界的框架，它记录基线、度量指标、关系类型和可采纳的证据；在计算结果之前锁定审计设计；并给出有明确范围的关系性结论或明确的证据边界。在锁定的审计范围内，严格的标量方向性比较会被分类为“维持”、“未决”或“未保留”。我们考察了455条比较性声明……（原文摘要在此处截断）

    arXiv:2607.00516v2 Announce Type: replace  Abstract: Empirical quantum-software papers often claim that one compiler produces smaller circuits, one optimizer is more reliable, or one ansatz achieves better solution quality. Such comparisons depend on benchmark scope, circuit construction, compilation, sampling, backends, noise models, optimizer choices, and resource budgets. Existing testing, benchmarking, and reproducibility methods help assess programs and experiments but do not directly audit whether the reported comparison is supported by the evidence available in the paper and artifact. We present CLAIMSTAB-QC, a source-bounded framework that records baselines, metric, relation, and admissible evidence; locks the audit design before outcomes are computed; and reports a scoped relation outcome or an explicit evidence boundary. Strict scalar-directional comparisons are classified as Sustained, Unresolved, or Not preserved within the locked audit scope. We examine 455 comparative cla
    
[^56]: 面向多语言等价变异体检测的大语言模型：一项扩展实证研究

    Large Language Models for Multi-Lingual Equivalent Mutant Detection: An Extended Empirical Study

    [https://arxiv.org/abs/2607.00511](https://arxiv.org/abs/2607.00511)

    本文首个针对大语言模型在多语言（Java与C）等价变异体检测上的全面实证研究，利用3,302个Java和1,088个C变异体对进行基准评估，验证了LLM克服传统纯代码分析和机器学习方法局限的潜力。

    

    变异测试是一种保障软件质量的强大技术。然而，等价变异体的存在会带来不必要的成本和偏差，限制了其实际效果。尽管学界已提出众多等价变异体检测（EMD）方法，但它们往往面临各自独特的挑战：纯代码分析方法可能受限于对特定编译器基础设施的依赖，而现有机器学习方法则受制于训练数据稀缺以及对未见变异体泛化能力有限的问题。大语言模型（LLMs）近期通过更好地捕捉程序语义，在各类代码相关任务中展现出卓越性能。然而，它们在EMD方面的潜力在很大程度上仍未被探索，尤其是在多语言场景下。本文开展了首个关于LLM用于EMD的全面实证研究，使用3,302个Java变异体对和1,088个C变异体对，与最先进的（原文在此处截断）……

    arXiv:2607.00511v2 Announce Type: replace  Abstract: Mutation testing is a powerful technique for ensuring software quality. However, the presence of equivalent mutants introduces unnecessary costs and biases, limiting its practical effectiveness. Although numerous equivalent mutant detection (EMD) methods have been proposed, they often face distinct challenges: pure-code analysis methods can be limited by their reliance on specific compiler infrastructures, while existing machine-learning approaches remain constrained by scarce training data and limited generalization to unseen mutants. Large language models (LLMs) have recently demonstrated remarkable performance across diverse code-related tasks by better capturing program semantics. Yet their potential for EMD remains largely unexplored, particularly in the multi-lingual context. This paper presents the first comprehensive empirical study on LLMs for EMD, using 3,302 Java and 1,088 C mutant pairs to benchmark against state-of-the-a
    
[^57]: 语义代码克隆检测中的鲁棒性表征

    Characterizing Robustness in Semantic Code Clone Detection

    [https://arxiv.org/abs/2606.25272](https://arxiv.org/abs/2606.25272)

    该论文提出系统化代码转换（SCT）框架，通过51个按代码层级、目标和编辑类别组织的转换算子，系统评估了14种语义代码克隆检测器对行为保持型变更的鲁棒性，发现25个高性能检测器与数据集组合中有21个在全算子组合下F1分数至少下降10个百分点。

    

    语义代码克隆检测旨在识别不同实现之间共享的功能，因此对行为保持型变更的鲁棒性是其有效性的核心。先前的研究已揭示了此类变更下的失败案例，这促使我们系统地研究检测器敏感性如何随评估所用的重写方式而变化。我们提出了系统化代码转换框架，该框架按代码层级、目标和编辑类别组织了51个转换算子，并在BigCloneBench、Google Code Jam和OJClone数据集上评估了14种基于token、树、图和提示的检测器。通过比较转换分组、重复应用以及替代构造方式，我们将整体性能变化与特定的重写条件联系起来。在初始F1分数超过90%的25个token、树和图检测器与数据集组合中，有21个在所有算子的组合作用下损失了至少10个百分点。这些损失随转换类型的不同而变化。

    arXiv:2606.25272v2 Announce Type: replace  Abstract: Semantic code clone detection seeks to recognize shared functionality across different implementations, making robustness to behavior-preserving changes central to its effectiveness. Prior studies have exposed failures under such changes, motivating a systematic examination of how sensitivity varies with the rewrites used for evaluation. We present Systematic Code Transformation (SCT), a framework organizing 51 operators by code level, target, and edit category, and evaluate 14 token, tree, graph, and prompt-based detectors on BigCloneBench, Google Code Jam, and OJClone. By comparing transformation groups, repeated application, and alternative constructions, we connect overall performance changes with specific rewriting conditions. Of 25 token, tree, and graph detector and dataset combinations starting above 90% F1, 21 lose at least 10 percentage points under the composition of all operators. These losses vary with transformation typ
    
[^58]: LLM安全评判者是否具有策略不变性？一项三原则压力测试

    Are LLM Safety Judges Policy-Invariant? A Three-Principle Stress-Test

    [https://arxiv.org/abs/2605.06161](https://arxiv.org/abs/2605.06161)

    本文提出“策略不变性”这一安全评判者必须满足的基本属性，并将其操作化为三条可检验原则构成压力测试协议，揭示了现有LLM安全评判者的判定会同时受有意义规范变化和无意义措辞改写影响这一此前未被测量的失效模式。

    

    LLM-as-a-Judge（以大语言模型作为评判者）流水线已成为智能体安全评估的事实标准，然而现有基准测试将评判者的判定结果视为真实标准的代理，却从未检验这些判定究竟取决于智能体的行为，还是仅仅取决于评估策略恰好采用的措辞方式。我们认为，任何可信的安全评判者都必须满足一个我们称之为“策略不变性”的基本属性，并将其操作化为三条可检验的原则：在经认证的等价改写下评分标准的语义不变性；在有意进行的由严格到宽松的阈值偏移下评分标准的阈值不变性；以及歧义感知校准，使得判定结果的不稳定性集中于真正存在歧义的案例。我们将这些原则实例化为一套压力测试协议，使用四个智能体级评判者在来自ASSEBench和R-Judge的轨迹上进行测试，揭示了一种此前未被测量过的失效模式：当今的评判者对有意义的规范变化与无意义的结构性改写……（摘要原文在此处截断）

    arXiv:2605.06161v2 Announce Type: replace  Abstract: LLM-as-a-Judge pipelines have become the de facto evaluator for agent safety, yet existing benchmarks treat their verdicts as ground-truth proxies without checking whether the verdicts depend on the agent's behavior or merely on how the evaluation policy happens to be worded. We argue that any trustworthy safety judge must satisfy a basic property we call policy invariance, and we operationalize it as three testable principles: rubric-semantics invariance under certified-equivalent rewrites, rubric-threshold invariance under intentional strict-to-lenient shifts, and ambiguity-aware calibration so that verdict instability concentrates on genuinely ambiguous cases. Instantiating these principles as a stress-test protocol with four agent-class judges on trajectories drawn from ASSEBench and R-Judge, we surface a previously unmeasured failure mode: today's judges respond to meaningful normative shifts and to meaningless structural rewrit
    
[^59]: AI 垃圾内容与软件公共资源

    AI Slop and the Software Commons

    [https://arxiv.org/abs/2604.16754](https://arxiv.org/abs/2604.16754)

    本文论证 AI 生成的低质量代码正在引发软件开发的“公地悲剧”，并借鉴 Ostrom 的公共资源治理设计原则，为软件社区各类角色提出可持续治理垃圾内容的具体方案。

    

    我们认为，软件开发中的 AI 垃圾内容正在制造一场“公地悲剧”。垃圾内容生成成本低廉，但审查代价高昂，其成本被转嫁到共享资源上，例如审查者的精力——早在 AI 助手普及之前，这种资源就已十分有限。Ostrom 记录了社区如何通过制定并执行自身规则来避免此类悲剧。我们将 Ostrom 关于可持续公共资源制度的设计原则转化为面向贡献者、审查者、工具开发者、团队负责人、组织领导者和教育工作者的具体建议。

    arXiv:2604.16754v4 Announce Type: replace  Abstract: We argue that AI slop in software development is creating a tragedy of the commons. Slop is cheap to generate but expensive to review, and its costs fall on shared resources such as reviewer capacity, which was limited even before AI assistants became widespread. Ostrom documented how communities avoid such tragedies by setting and enforcing their own rules. We translate Ostrom's design principles for sustainable commons institutions into concrete proposals for contributors, reviewers, tool developers, team leads, organizational leaders, and educators.
    

