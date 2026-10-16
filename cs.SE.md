# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FDE-Gym: The Last Mile of AI Delivery](https://arxiv.org/abs/2610.22350) | 提出了FDE-Gym基准，包含来自18个行业20个客户场景的95个端到端交付任务，用于评估智能体在真实客户交付中厘清模糊需求、与利益相关方互动并在类生产系统中交付可用解决方案的能力。 |
| [^2] | [Are We Testing the Wrong Quantum Bugs? A Study of Fault Representation Gaps in Quantum Software Testing](https://arxiv.org/abs/2610.22089) | 该研究通过审查量子变异测试工具和人工整理的缺陷基准，并收集两年内410个真实缺陷，揭示量子软件测试评估中存在显著的缺陷表示差距——现有变异算子与基准仅覆盖极少数缺陷族，无法充分代表真实量子软件中的缺陷。 |
| [^3] | [WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents](https://arxiv.org/abs/2610.22039) | WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。 |
| [^4] | [Stateful Testing for Deep Learning Compilers](https://arxiv.org/abs/2610.22003) | 提出面向深度学习编译器的有状态测试方法 Rondo，通过将模型与调用序列配对、利用编译行为引导的 MCMC 采样进行序列变异，并结合三路差分预言机，检测由编译器保留状态引发且在单独执行时无法复现的缺陷。 |
| [^5] | [Benchmarking and Enhancing Design-Aware Code Review in the Linux Kernel](https://arxiv.org/abs/2610.21925) | 本文提出了DesignLore知识增强方法和DesignReviewBench基准，通过从历史审查讨论中提取、组织并复用设计决策知识，系统性评估并增强了Linux内核中的设计感知代码审查。 |
| [^6] | [CodeHealth Predicts the Token Cost of Agentic Code Review: A Pre-Registered Study of 465 Agent-Authored Pull Requests](https://arxiv.org/abs/2610.21914) | 这项预注册研究通过对465个智能体创建的PR进行分层抽样分析，首次证实代码健康度（CodeHealth）能够预测智能体代码审查的Token成本——健康度每提高1分，成本降低3.5%至6.1%，且该效应独立于PR规模。 |
| [^7] | [Realization Divergence in Multi-Provider Quantum Computing: A Pre-Registered Study of What Survives Measurement](https://arxiv.org/abs/2610.21714) | 该研究提出“实现分歧”概念来量化同一量子电路在不同供应商软件栈中的实现差异，并通过覆盖三家供应商、四台设备的预注册硬件研究发现，嵌入半径比电路规模更能预测双量子比特开销，且电路规模与执行分歧的排序在不同设备上可能完全相反。 |
| [^8] | [Independent Data-Integrity Verification for LLM-Driven Functional Coverage Closure in Hardware Design](https://arxiv.org/abs/2610.21570) | 该论文提出一个带有独立数据完整性验证的LLM覆盖率收敛智能体，通过从仿真报告读取覆盖率并使用独立记分板检查数据正确性，仅用一个与设计无关的智能体就在三种不同RTL设计上将覆盖率提升至100%，并额外发现了一个时钟域跨越错误。 |
| [^9] | [Poison Valleys: Diagnosing Path-Dependent Failure in LLM Code Repair](https://arxiv.org/abs/2610.21514) | 该论文提出“毒谷”诊断框架，通过向LLM代码修复的假设与计划中注入看似合理但错误的软件机制，首次量化揭示了上游错误概念的位置如何路径依赖性地决定下游代码失败——将错误概念从假设移至计划阶段会使失败率增加约29%至32%。 |
| [^10] | [The Role of Agentic Instruction Categories for Code Generation: A Study of Five Python Libraries](https://arxiv.org/abs/2610.21421) | 该研究利用两个大语言模型和五个Python库系统评估了五类代理指令的全部31种组合，发现指令数量本身并不能决定代理文件生成的token消耗，且不同模型和库之间的表现差异显著。 |
| [^11] | [Evaluating and Improving Static Application Security Testing Tools for Rust](https://arxiv.org/abs/2610.21373) | 本文通过真实漏洞对主流Rust静态安全测试工具进行实证评估，系统揭示其弱点及根本原因，并提出了基于大语言模型辅助的自动化测试方法TRUST来改进Rust安全测试工具。 |
| [^12] | [Agentic Dependency Transformations: Lessons from an Exploratory Case Study of Flask](https://arxiv.org/abs/2610.21372) | 本文通过对Flask框架的探索性案例研究，首次提出“依赖转换”这一新兴概念——即AI编程智能体改变软件仓库处理外部依赖的方式，总结出五点初步经验教训，并将其确立为一个具有五部分研究议程的新兴研究方向。 |
| [^13] | [Combining LLM and Dynamic Analysis for Universal Testing of Static Application Security Testing Tools](https://arxiv.org/abs/2610.21309) | 提出LLM4SAST，首个结合大语言模型与动态反馈来检测静态应用安全测试工具缺陷的通用框架，通过蜕变关系引导的多语言代码变体生成实现自动化测试。 |
| [^14] | [ToolBottleneckBench: Measuring Tool-Agent Performance under Controlled Pressures and Fixed Harness Budgets](https://arxiv.org/abs/2610.21098) | 该论文提出ToolBottleneckBench基准，在固定框架预算下通过单独变化工具目录、可见菜单、认证工作流或返回记录编码等压力因素来测量工具智能体性能，发现可见工具菜单是唯一对模型造成大幅惩罚的压力且其根源是完成上限处的截断，而大多数失败实际终止于框架限制而非模型能力不足。 |
| [^15] | [Skip the Compressor: Coding Agents Fix More Issues When They Pick Whole Files](https://arxiv.org/abs/2610.21073) | 在相同token预算下，与其用压缩器删减文本或检索器挑选代码块，不如将预算用于选择需要原样完整呈现的文件（如CQ2-CARE），这样编码代理能解决更多软件工程问题。 |
| [^16] | [Where the SRAM Goes: Per-Symbol Attribution of Static Memory in FreeRTOS-C and Embassy-Rust IoT Firmware on a 64 KB RISC-V Microcontroller](https://arxiv.org/abs/2610.21064) | 提出一种逐符号的静态 SRAM 归因方法，在 64 KB RISC-V 微控制器上将 FreeRTOS-C 与 Embassy-Rust 物联网固件的静态内存精确分解为调度器堆、网络缓冲池、任务栈和链接器保留区域等类别，并与总消耗量完全对账验证。 |
| [^17] | [Robotic-supported Data Loss Detection in Android Applications](https://arxiv.org/abs/2610.20807) | 该论文提出利用机器人自动化技术以非侵入方式检测安卓应用中的数据丢失问题，从而高效应对多样化硬件与操作系统配置下的应用测试挑战。 |
| [^18] | [Causal Syntax Circuits in CodeLMs: Budget and Probe Decide What Looks Localised](https://arxiv.org/abs/2610.20683) | 该论文通过均值消融、激活修补、贪心电路提取与激活引导的统一流程，系统研究了代码语言模型默认偏好Python的内部机制，发现该偏好随模型规模增长并在指令微调后仍存在，但其局域化表现高度依赖于组件预算与探针的选择。 |
| [^19] | [HPCAgent-Bench: A Framework for Sound Benchmarking of Code-Optimizing Agents, Harnesses, and Skills](https://arxiv.org/abs/2610.20673) | 该论文提出了HPCAgent-Bench——一个包含689个内核、覆盖三个赛道的代码优化基准框架，通过统一的内核规范、外部防作弊裁判以及“干预效力”等新指标，实现对代码优化智能体、执行框架与技能各自贡献的可靠隔离评估。 |
| [^20] | [Developer Experiences with Cursor AI-native IDE](https://arxiv.org/abs/2610.20618) | 本研究通过分析Cursor官方论坛两年多来的14,714条帖子，大规模纵向考察了开发者使用AI原生IDE的真实体验、采用摩擦及工作流演变。 |
| [^21] | [The Code of Theseus: Can Agents Rebuild Software from Recovered Design Documents?](https://arxiv.org/abs/2610.20068) | 提出一个从紧凑仓库表示中恢复相互对齐的意图、需求与架构文档的框架，并发布覆盖 10,764 个多语言仓库的工件数据集，用于研究智能体能否凭借恢复的设计文档重建软件。 |
| [^22] | [SWE-Nexus: Automated Composition of Software Engineering Tasks from Test Execution](https://arxiv.org/abs/2610.20038) | SWE-Nexus 是一个完全自动化的流水线，通过追踪测试执行中生产函数的足迹、验证屏蔽效果并确定性组合兼容作用域，自动构建出涵盖 105 个仓库、共 600 个任务的仓库级软件工程实现基准。 |
| [^23] | [ReTerm: Recovery Training for Self-Improving Terminal Agents](https://arxiv.org/abs/2610.18484) | 提出ReTerm终端智能体，通过恢复示范数据集ReTerm-10K进行SFT冷启动，并利用从自身失败状态分叉出多次恢复尝试的在线强化学习算法ReTerm-GRPO，让智能体从自身恢复经验中自我改进，在Terminal-Bench 2.0上取得34.8%的Pass@1成绩。 |
| [^24] | [NosRacer: Dynamic Detection of Race Conditions in On-Device Network Operating Systems](https://arxiv.org/abs/2610.11875) | 本文提出NosRacer，一个采用两阶段设计的动态分析框架，用于高效检测工业级设备内置网络操作系统（NOS）中由异步消息传递引起的竞态条件。 |
| [^25] | [ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding](https://arxiv.org/abs/2610.08662) | 提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。 |
| [^26] | [Testing and Verification of Quantum Compilers through Assurance Contracts and Evidence](https://arxiv.org/abs/2610.00255) | 该论文是一项关于量子编译器测试与验证的批判性综合综述，系统比较了经验证变换、等价性检查、差分与蜕变测试、基于属性的测试等多种方法，并通过失败案例和工作示例说明如何借助保证契约与证据确保量子编译器在其交付接口上的正确性。 |
| [^27] | [From Migration to Calibration: Preserving Agent Capabilities across Models, Jurisdictions, and Scale](https://arxiv.org/abs/2609.35149) | 提出将智能体校准构建为标准优先的适配框架，通过定义基础能力、技术环境与用户情境标准，系统诊断差距并应用修订，从而在模型更换、跨管辖区部署和规模化过程中保持智能体能力不退化。 |
| [^28] | [SWE-PolyVision: Benchmarking Cross-Image Abductive Reasoning for Repository-Level Software Engineering](https://arxiv.org/abs/2609.29754) | 提出 SWE-PolyVision 基准，通过 92 个包含多图像输入的真实软件工程任务，首次系统评估智能体将跨图像视觉证据与代码证据整合并进行经验证的仓库级修复的能力，发现视觉访问的效果因模型和任务而异。 |
| [^29] | [A Model-Centric DevOps Architecture for DEVS-Based Digital Twin Simulation Services](https://arxiv.org/abs/2609.11122) | 本文提出一种以模型为中心的DevOps架构，将基于DEVS的数字孪生仿真模型作为一等制品，通过声明式YAML定义、到multiPDEVS的形式化映射、CI/CD流水线中的结构与语义验证以及Kubernetes容器化微服务部署，实现了仿真模型的可追溯版本管理与持续交付。 |
| [^30] | [Praxist: From Experimental Artifacts to Solution Lineages](https://arxiv.org/abs/2608.25955) | Praxist通过谱系中心方法将实验人工制品转化为类型化证据图，分离局部构建与证据综合，使研发代理能继承已验证机制，避免重复学习并提升长期研发效率。 |
| [^31] | [SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts](https://arxiv.org/abs/2608.25202) | 我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。 |
| [^32] | [Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels](https://arxiv.org/abs/2608.02712) | 该论文提出“调试而非重新生成”的范式转变，设计了一个领域专用调试智能体，用于修复那些能够编译运行但未通过数值验证的“差一点成功”加速器内核，从而将原本被丢弃的失败候选转化为可复用的知识。 |
| [^33] | [Identifier-Free Code Embedding Models for Scalable Search](https://arxiv.org/abs/2605.05251) | 该论文形式化了源代码与去除符号的反编译代码之间的双向函数关联问题，并通过对比学习微调Qwen3-Embedding模型，在所有基线测试中大幅超越现有模型。 |
| [^34] | [Isolating Recurring Backend-Dependent Abnormal Patterns on NISQ Quantum Devices](https://arxiv.org/abs/2604.17519) | 提出了QRisk框架，能够从量子硬件执行中隔离出重复出现的、依赖特定后端的异常门序列，从而指导门排序重写，弥合编译器噪声模型与真实设备行为之间的差距。 |
| [^35] | [ORACLE-SWE: Quantifying the Contribution of Critical Information Signals on SWE Agents](https://arxiv.org/abs/2604.07789) | 该论文提出ORACLE-SWE，首次系统量化了复现测试、回归测试、编辑位置、执行上下文和API用法等关键中间信息信号对软件工程代理任务成功的贡献，并测量了代理发现的信号与理想神谕信号之间尚存的改进空间。 |
| [^36] | [SEVerA: Verified Synthesis of Self-Evolving Agents](https://arxiv.org/abs/2603.25111) | 提出了SEVerA框架，通过形式化守护生成模型（FGGM）将自演化智能体代码生成形式化为受约束的学习问题，从而为自演化LLM智能体提供安全性和正确性的形式化保证。 |
| [^37] | [LLMON: An LLM-native Markup Language to Leverage Structure and Semantics at the LLM Interface](https://arxiv.org/abs/2603.22519) | 提出了一种LLM原生标记语言LLMON，能够在LLM接口层面自然地传达文本的结构与语义元数据，从而提升模型的准确性、安全性以及抵抗提示注入攻击的能力。 |
| [^38] | [Benchmarking Code LLM-as-a-Judge via Constrained Perturbation and Calibration](https://arxiv.org/abs/2512.20159) | 该论文提出通过约束扰动与校准技术构建代码LLM-as-a-Judge评估基准，克服了现有基准中粗粒度二元标签无法区分缺陷严重程度以及细粒度人工评分主观不可靠的局限性。 |
| [^39] | [RAG4Tickets: AI-Powered Ticket Resolution via Retrieval-Augmented Generation on JIRA and GitHub Data](https://arxiv.org/abs/2510.08667) | 该论文提出RAG4Tickets框架，通过将Sentence-Transformers语义嵌入、FAISS向量搜索与大语言模型相结合，利用JIRA和GitHub的历史工单及PR数据，为软件工单提供有依据且可解释的解决建议。 |
| [^40] | [Multi-LLM Orchestration for High-Quality Code Generation: Exploiting Complementary Model Strengths](https://arxiv.org/abs/2510.01379) | 提出PerfOrch多智能体系统，通过Memory模块记录各LLM在不同开发阶段、编程语言和算法类别上的实测性能排名，从而在代码生成、调试和优化的每个环节动态选用最合适的模型，利用模型间互补优势实现比单一模型更高的代码正确性和执行性能。 |
| [^41] | [Enhancing Knowledge Density via Hierarchical Context Pruning for Retrieval-Augmented Repository-Level Code Completion](https://arxiv.org/abs/2508.10068) | 提出SaraCoder框架，在固定上下文预算下通过分层上下文剪枝提升检索上下文的知识密度，避免冗余代码浪费token并释放跨文件依赖与相似代码片段的协同作用，从而提升仓库级代码补全性能。 |
| [^42] | [An Empirical Analysis of EOS Blockchain: Architecture, Contract, and Security](https://arxiv.org/abs/2505.15051) | 本研究首次从系统架构、去中心化、性能、智能合约和行为安全五个维度对EOS区块链进行了全面的实证分析，揭示了账户活跃度低、超级节点选举参与有限等关键问题。 |

# 详细

[^1]: FDE-Gym：AI交付的最后一公里

    FDE-Gym: The Last Mile of AI Delivery

    [https://arxiv.org/abs/2610.22350](https://arxiv.org/abs/2610.22350)

    提出了FDE-Gym基准，包含来自18个行业20个客户场景的95个端到端交付任务，用于评估智能体在真实客户交付中厘清模糊需求、与利益相关方互动并在类生产系统中交付可用解决方案的能力。

    

    现有的智能体基准测试通常在由问题报告、规格说明或任务工单所定义的明确任务上评估智能体。真实的客户交付则不同：前沿部署工程师必须厘清模糊的需求，在运行中的系统中工作，并发掘未被明确表述的约束条件。我们提出了FDE-Gym，这是一个包含95个端到端交付任务的基准，这些任务来自横跨18个行业的20个客户场景，每一个都由行业专家根据真实交付项目改编而成。在配备剧本化利益相关方和可修改客户系统的模拟客户环境中，智能体与利益相关方互动，并在类生产系统上进行操作以交付可用的解决方案。FDE-Gym通过交付对客户自身成功指标的改善程度来进行评分，归一化到0（系统未改变）到1（参考解决方案）之间，并对利益相关方互动（消息、试运行和等待时间）施加扣分。一个在线平台允许人类……

    arXiv:2610.22350v1 Announce Type: new  Abstract: Existing agent benchmarks typically evaluate agents on well-specified tasks defined by issue reports, specifications, or task tickets. Real-world customer delivery is different: forward-deployed engineers (FDEs) must clarify ambiguous requirements, work in live systems, and uncover unstated constraints. We introduce FDE-Gym, a benchmark of 95 end-to-end delivery tasks from 20 customer scenarios across 18 sectors, each adapted by industry experts from real-world delivery projects. In simulated customer environments with scripted stakeholders and modifiable customer systems, agents interact with stakeholders and operate on production-like systems to deliver working solutions. FDE-Gym scores a delivery by how much it improves the customer's own success metric, normalized between 0 (unchanged system) and 1 (reference solution), with penalties for stakeholder interactions (messages, trial runs, and waiting time). An online platform lets human
    
[^2]: 我们是否在测试错误的量子缺陷？量子软件测试中缺陷表示差距的研究

    Are We Testing the Wrong Quantum Bugs? A Study of Fault Representation Gaps in Quantum Software Testing

    [https://arxiv.org/abs/2610.22089](https://arxiv.org/abs/2610.22089)

    该研究通过审查量子变异测试工具和人工整理的缺陷基准，并收集两年内410个真实缺陷，揭示量子软件测试评估中存在显著的缺陷表示差距——现有变异算子与基准仅覆盖极少数缺陷族，无法充分代表真实量子软件中的缺陷。

    

    量子软件依赖编译器、中间表示和执行框架，这些组件会在程序执行前对其进行转换。这些组件中的缺陷可能会改变原本正确的量子程序的行为。然而，量子测试技术通常使用变异生成的缺陷和人工整理的缺陷基准来评估。我们研究了这些工件所表示的缺陷能在多大程度上反映现代量子软件中实际遇到的缺陷。我们审查了四个量子变异测试工具和三个人工整理的缺陷基准，并在两年内从十个量子软件仓库中收集了410个已验证的缺陷。我们将来自这三个来源的缺陷统一归类到一个包含56个缺陷族的分类体系中，并比较了它们在变异算子、人工整理基准和近期真实世界缺陷中的覆盖情况。变异测试工具支持20个缺陷族，人工整理的基准又增加了15个。然而，410个真实世界缺陷中仅有56个（13.7%）属于……（原文摘要在此处截断）

    arXiv:2610.22089v1 Announce Type: new  Abstract: Quantum software relies on compilers, intermediate representations, and execution frameworks that transform programs before execution. Faults in these components can change the behavior of otherwise correct quantum programs. Yet quantum testing techniques are often evaluated using mutation-generated faults and curated bug benchmarks. We investigate how well the faults represented by these artifacts reflect those encountered in modern quantum software. We audit four quantum mutation-testing tools and three curated bug benchmarks, and collect 410 verified bugs from ten quantum software repositories over two years. We classify faults from all three sources into a common taxonomy of 56 bug families and compare their representation across mutation operators, curated benchmarks, and recent real-world bugs. The mutation tools support 20 bug families, and curated benchmarks add 15 more. Yet only 56 of the 410 real-world bugs (13.7%) belong to fa
    
[^3]: WEAVE：将真实世界代码演进编织为依赖结构化任务，用于代码智能体的基准测试与训练

    WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents

    [https://arxiv.org/abs/2610.22039](https://arxiv.org/abs/2610.22039)

    WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。

    

    真实世界的软件工程很少由孤立的变更构成。功能开发、迁移和系统性修复往往跨越多个拉取请求（PR）展开，需要对整个代码库进行协调一致的修改。然而，大多数可执行的代码库级基准测试仍以单个 issue 或 PR 作为任务单元，而现有的多 PR 设置则依赖粗粒度的聚合和与结构无关的评估。这种错位掩盖了变更之间的依赖关系，限制了长程代码智能体的进展衡量和学习信号。我们提出了 WEAVE，一个将经过验证的 PR 历史转化为可执行的、依赖结构化任务的框架。WEAVE 利用代码库校准的类型化变更位点，将相关联的 PR 组织成有界 DAG（有向无环图），并将每个 DAG 实例化为统一的可执行任务。我们进一步引入了 Graph Credit，这是一种确定性的、基于测试的度量指标，用于衡量符合依赖关系的部分进展，并支持更密集的……

    arXiv:2610.22039v1 Announce Type: cross  Abstract: Real-world software engineering rarely consists of isolated changes. Features, migrations, and systematic fixes often unfold across multiple pull requests (PRs), requiring coordinated modifications across a repository. Yet most executable repository-level benchmarks retain a single issue or PR as the task unit, while existing multi-PR settings rely on coarse aggregation and structure-agnostic evaluation. This mismatch obscures dependencies among changes and limits progress measurement and learning signals for long-horizon code agents. We introduce WEAVE, a framework that transforms verified PR histories into executable, dependency-structured tasks. WEAVE leverages repository-calibrated typed change loci to organize related PRs into bounded DAGs and instantiate each as a unified executable task. We further introduce Graph Credit, a deterministic, test-grounded metric that measures dependency-consistent partial progress and supports dens
    
[^4]: 面向深度学习编译器的有状态测试

    Stateful Testing for Deep Learning Compilers

    [https://arxiv.org/abs/2610.22003](https://arxiv.org/abs/2610.22003)

    提出面向深度学习编译器的有状态测试方法 Rondo，通过将模型与调用序列配对、利用编译行为引导的 MCMC 采样进行序列变异，并结合三路差分预言机，检测由编译器保留状态引发且在单独执行时无法复现的缺陷。

    

    现代深度学习编译器采用惰性编译方式，并在多次调用之间缓存编译产物。保留的编译器状态可能导致后续调用产生错误结果，即使该调用在单独执行时是正确的。对 PyTorch、JAX 和 TensorFlow 中 26 个公开报告问题的分析表明，检测此类缺陷需要先建立编译器状态、再暴露故障的调用序列。我们提出了 Rondo，一种有状态测试方法，它将每个测试用例表示为一个模型与一个调用序列的组合。Rondo 对调用序列进行变异，并使用由编译行为引导的马尔可夫链蒙特卡洛采样器，以偏向较少被观察到的行为。其三路差分测试预言机比较三种执行方式：不编译直接执行、重置编译器状态后的编译执行，以及保留编译器状态时的编译执行。它通过在保留状态与重置状态两种条件下重放相同调用（同时保持输入和程序状态不变）来确认有状态缺陷，然后……（摘要在此处截断）

    arXiv:2610.22003v1 Announce Type: cross  Abstract: Modern deep learning compilers compile models lazily and cache artifacts across invocations. Retained compiler state can make a later invocation incorrect even when it is correct in isolation. An analysis of 26 publicly reported issues in PyTorch, JAX, and TensorFlow shows that detecting such bugs requires sequences that first establish compiler state and then expose a failure. We present Rondo, a stateful testing method that represents each test case as a model paired with an invocation sequence. Rondo mutates invocation sequences and uses a Markov chain Monte Carlo sampler guided by compilation behavior to favor rarely observed behaviors. Its three-way differential oracle compares execution without compilation, compiled execution with compiler state reset, and compiled execution with state retained. It confirms stateful bugs by replaying the same invocation with state retained and reset while preserving inputs and program state, then
    
[^5]: 基准测试与增强Linux内核中的设计感知代码审查

    Benchmarking and Enhancing Design-Aware Code Review in the Linux Kernel

    [https://arxiv.org/abs/2610.21925](https://arxiv.org/abs/2610.21925)

    本文提出了DesignLore知识增强方法和DesignReviewBench基准，通过从历史审查讨论中提取、组织并复用设计决策知识，系统性评估并增强了Linux内核中的设计感知代码审查。

    

    代码审查往往需要超越局部正确性的推理。在成熟的软件系统中，审查者必须识别所提议变更引入的设计权衡，比较备选解决方案，并推理其与项目特定约束的兼容性。我们将这一问题称为“设计感知代码审查”。然而，现有的自动化审查方法主要关注缺陷和正确性，而历史审查讨论中积累的设计知识仍然难以复用。为了应对这些挑战，我们提出了DesignLore，这是一种知识增强方法，它从历史审查讨论中提取设计决策，将其组织成自适应的设计分类体系，并复用相关的历史知识来指导代码审查。我们进一步构建了DesignReviewBench，这是一个源自真实补丁审查历史的基准，用于系统地评估设计感知代码审查。我们的评估表明……

    arXiv:2610.21925v1 Announce Type: new  Abstract: Code review often requires reasoning beyond local correctness. In mature software systems, reviewers must recognize design trade-offs introduced by a proposed change, compare alternative solutions, and reason about their compatibility with project-specific constraints. We refer to this problem as design-aware code review. However, existing automated review approaches primarily focus on defects and correctness, while the design knowledge accumulated in historical review discussions remains difficult to reuse.   To address these challenges, we present DesignLore, a knowledge-augmented approach that extracts design decisions from historical review discussions, organizes them into an adaptive design taxonomy, and reuses relevant historical knowledge to guide code review. We further construct DesignReviewBench, a benchmark derived from real patch-review histories for systematically evaluating design-aware code review. Our evaluation shows tha
    
[^6]: 代码健康度可预测智能体代码审查的Token成本：一项针对465个智能体创建的Pull Request的预注册研究

    CodeHealth Predicts the Token Cost of Agentic Code Review: A Pre-Registered Study of 465 Agent-Authored Pull Requests

    [https://arxiv.org/abs/2610.21914](https://arxiv.org/abs/2610.21914)

    这项预注册研究通过对465个智能体创建的PR进行分层抽样分析，首次证实代码健康度（CodeHealth）能够预测智能体代码审查的Token成本——健康度每提高1分，成本降低3.5%至6.1%，且该效应独立于PR规模。

    

    背景：各组织日益将代码审查工作委托给按Token计费的多智能体大语言模型（LLM）流水线。低代码质量会增加人工维护成本，但其对智能体审查成本的影响尚未被测量。目标：我们检验基线CodeHealth（代码健康度）能否在Pull Request（PR）规模之外预测智能体审查与修复的Token成本。方法：在一项预注册的观察性研究中，我们从AIDev的246个代码仓库中抽样了465个由智能体创建的PR，在仓库数量上限约束下按CodeHealth和PR规模进行联合分层。一条固定的五智能体审查-修复流水线使用三种不同的LLM处理了每个PR。针对每个LLM，我们拟合了以CodeHealth和对数PR规模为自变量、以对数Token数为因变量的混合效应模型，并加入仓库随机截距。结果：对于所有三种LLM，CodeHealth的系数均为负值，且95%置信区间不包含零；CodeHealth每提高1分，Token成本降低3.5%至6.1%。CodeHealth（原文摘要在此处截断）

    arXiv:2610.21914v1 Announce Type: cross  Abstract: Context: Organizations increasingly delegate code review to multi-agent large language model (LLM) pipelines billed per token. Low code quality raises human maintenance effort, but its effect on agent review cost has not been measured.   Objective: We test whether baseline CodeHealth predicts the token cost of agentic review and fix beyond pull request (PR) footprint.   Method: In a pre-registered observational study, we sampled 465 agent-authored PRs from 246 repositories in AIDev, jointly stratified by CodeHealth and footprint under repository caps. A frozen five-agent review-and-fix pipeline processed every PR with three different LLMs. Per LLM, we fitted a mixed model of log tokens on CodeHealth and log PR footprint, with a repository random intercept.   Results: For all three LLMs the CodeHealth coefficient was negative, with 95% CI excluding zero; each CodeHealth point was associated with 3.5% to 6.1% lower token cost. CodeHealth
    
[^7]: 多供应商量子计算中的实现分歧：一项关于测量结果幸存内容的预注册研究

    Realization Divergence in Multi-Provider Quantum Computing: A Pre-Registered Study of What Survives Measurement

    [https://arxiv.org/abs/2610.21714](https://arxiv.org/abs/2610.21714)

    该研究提出“实现分歧”概念来量化同一量子电路在不同供应商软件栈中的实现差异，并通过覆盖三家供应商、四台设备的预注册硬件研究发现，嵌入半径比电路规模更能预测双量子比特开销，且电路规模与执行分歧的排序在不同设备上可能完全相反。

    

    量子程序在执行前需要经过编译器、转译器和供应商服务的处理。然而，所编写的程序与实际运行内容之间的差距却很少被测量。我们将这一差距称为“实现分歧”：即同一源电路在不同量子软件栈中被实现和运行时所表现出的可观测差异。我们报告了一项覆盖三家供应商、四台设备的硬件研究，将预注册的预测与明确标识的探索性分析及修正相结合。在一项采用指定放置策略的编译实验中，嵌入半径与双量子比特开销的 Spearman 相关系数达到 0.805，而电路规模仅为 0.492。在固定的最低索引放置策略下，一个八量子比特的 Bernstein-Vazirani 电路在 IQM Emerald 上表现出的执行分歧大于十二量子比特电路，而在 IQM Garnet 上则出现相反的排序。两项比较均在两次重复实验中得到复现，且置信区间互不重叠。

    arXiv:2610.21714v1 Announce Type: new  Abstract: Quantum programs pass through compilers, transpilers and vendor services before execution. However, the distance between what was written and what runs is rarely measured. We call this distance realization divergence: the observable difference in how an identical source circuit is realized and behaves across quantum software stacks. We report a hardware study across three vendors and four devices, combining pre-registered predictions with explicitly identified exploratory analyses and amendments. In a compilation experiment with assigned placement policies, embedding radius correlates with two-qubit overhead at Spearman 0.805, against 0.492 for circuit size. Under a fixed lowest-index policy, an eight-qubit Bernstein-Vazirani circuit exhibits greater execution divergence than a twelve-qubit circuit on IQM Emerald. The opposite ordering occurs on IQM Garnet. Both comparisons replicate across two repetitions with disjoint confidence interv
    
[^8]: 面向硬件设计中LLM驱动功能覆盖率收敛的独立数据完整性验证

    Independent Data-Integrity Verification for LLM-Driven Functional Coverage Closure in Hardware Design

    [https://arxiv.org/abs/2610.21570](https://arxiv.org/abs/2610.21570)

    该论文提出一个带有独立数据完整性验证的LLM覆盖率收敛智能体，通过从仿真报告读取覆盖率并使用独立记分板检查数据正确性，仅用一个与设计无关的智能体就在三种不同RTL设计上将覆盖率提升至100%，并额外发现了一个时钟域跨越错误。

    

    LLM智能体能够编写定向测试来闭合功能覆盖率，但覆盖率只能表明某个状态被到达，并不能表明通往该状态的路径上的数据是正确的。这是验证领域早已广为人知的观点，我们并不声称其为新发现。我们描述了一个将该观点应用于LLM循环的小型覆盖率收敛智能体：覆盖率从仿真自身导出的报告（cocotb-coverage YAML）中读取，而绝不采信模型的自述；数据完整性则由独立于模型的记分板进行检查。在三个架构上各不相同的RTL设计（异步FIFO、UART发送器和AHB2APB桥）上，该智能体仅使用一个与设计无关的智能体，在每次记录的运行中均能在8次迭代的预算内将覆盖率从91.7%、88.9%和67.4%提升至100%。连接到FIFO基础测试平台的响应式记分板监视器还在作者自己编写的激励代码中发现了一个时钟域跨越（CDC）错误。该监视器尚未（摘要在此处被截断）

    arXiv:2610.21570v1 Announce Type: cross  Abstract: LLM agents can write directed tests that close functional coverage, but coverage only shows that a state was reached, not that the data on the way there was correct. This is a long-known point in verification and we do not claim it as new. We describe a small coverage-closure agent that applies it to an LLM loop: coverage is read from the report that the simulation itself exports (cocotb-coverage YAML) and never taken from the model's account, and data integrity is checked by scoreboards that are separate from the model. On three architecturally distinct RTL designs (an asynchronous FIFO, a UART transmitter and an AHB2APB bridge) the agent raised coverage from 91.7%, 88.9% and 67.4% to 100% within its 8-iteration budget in every logged run, using one design-agnostic agent. A reactive scoreboard monitor attached to the FIFO's base testbench also found a clock-domain-crossing bug in the author's own stimulus code. The monitor is not yet 
    
[^9]: 毒谷：诊断LLM代码修复中的路径依赖性失败

    Poison Valleys: Diagnosing Path-Dependent Failure in LLM Code Repair

    [https://arxiv.org/abs/2610.21514](https://arxiv.org/abs/2610.21514)

    该论文提出“毒谷”诊断框架，通过向LLM代码修复的假设与计划中注入看似合理但错误的软件机制，首次量化揭示了上游错误概念的位置如何路径依赖性地决定下游代码失败——将错误概念从假设移至计划阶段会使失败率增加约29%至32%。

    

    当代码修复智能体失败时，最终测试只能揭示补丁是错误的，但无法说明是哪一个早期的承诺导致了失败，或哪个软件义务未能满足。Poison Valleys 在两个合成 Python 案例的 1,844 次模型请求中追踪了这一失败路径：PersistQueue 为基于磁盘的队列添加原子批量认领功能，Dijkstra 则为最短路径搜索添加源敏感过滤功能。我们定义了看似合理但不正确的软件机制，将每种错误概念分别置于模型的假设、实现计划或两者之中，并使用 Oracle 和有针对性的诊断坐标来评估生成的代码。在匹配实验组合后，将错误概念从假设阶段移至计划阶段使 PersistQueue 的失败率增加了 32.2%，Dijkstra 的失败率增加了 28.9%。不同的错误概念产生了独特的、依赖模型的具体义务违反模式，尽管其中若干错误可能收敛于相同的行为。这建立了从上游错误概念到代码失败之间的量化联系。

    arXiv:2610.21514v1 Announce Type: cross  Abstract: When code-repair agents fail, final tests reveal that a patch is wrong but not which earlier commitment caused it or which software obligation failed. Poison Valleys traces this path in 1,844 model requests on two synthetic Python cases: PersistQueue adds atomic batch claiming to a disk-backed queue, while Dijkstra adds source-sensitive filtering to shortest-path search. We define plausible but incorrect software mechanisms, place each in a model's hypothesis, implementation plan, or both, and evaluate the generated code with an Oracle and targeted diagnostic coordinates. Moving the misconception from hypothesis to plan increased failure by 32.2% in PersistQueue and 28.9% in Dijkstra after matching the experimental mix. Different misconceptions produced distinct, model-dependent patterns of violated obligations, although several could converge on the same behavior. This creates a measured connection from upstream misconception to code 
    
[^10]: 代理式指令类别在代码生成中的作用：五个人工智能Python库的研究

    The Role of Agentic Instruction Categories for Code Generation: A Study of Five Python Libraries

    [https://arxiv.org/abs/2610.21421](https://arxiv.org/abs/2610.21421)

    该研究利用两个大语言模型和五个Python库系统评估了五类代理指令的全部31种组合，发现指令数量本身并不能决定代理文件生成的token消耗，且不同模型和库之间的表现差异显著。

    

    代理式编码系统日益依赖仓库级的代理文件（如AGENTS.md）来传递项目知识和开发者意图。然而，目前尚不清楚哪些类别的代理指令能够实现高效的代理文件生成和准确的代码生成。我们使用两个大语言模型（LLM）——Gemma 4和GPT-5.4，以及五个人类编写的Python库，评估了五类指令：仓库概述、函数描述、实现细节、测试和重构步骤，涵盖了所有31种非空组合。我们的研究围绕代理文件生成和代码生成提出了两个研究问题（包含五个子问题）。结果表明，仅凭指令数量并不能决定代理文件生成的token使用量，且不同模型和库之间存在显著差异。尽管无论采用哪种指令组合，大多数AI生成的库都能成功编译……

    arXiv:2610.21421v1 Announce Type: new  Abstract: Agentic coding systems increasingly rely on repository-level agent files, such as AGENTS.md, to communicate project knowledge and developer intent. However, it remains unclear which categories of agent instructions enable efficient agent-file generation and accurate code generation. We evaluate five instruction categories: Repository Overview, Function Description, Implementation Detail, Testing, and Reconstruction Steps, across all 31 non-empty combinations using two Large Language Models (LLMs), Gemma 4 and GPT-5.4, and five human-written Python libraries. Our study addresses two research questions, comprising five sub-questions, on agent file generation and code generation. Results show that the number of instructions alone does not determine token usage for agent-file generation, with substantial variation across models and libraries. Although most AI-generated libraries compiled successfully regardless of the instruction combination
    
[^11]: 评估与改进面向Rust的静态应用程序安全测试工具

    Evaluating and Improving Static Application Security Testing Tools for Rust

    [https://arxiv.org/abs/2610.21373](https://arxiv.org/abs/2610.21373)

    本文通过真实漏洞对主流Rust静态安全测试工具进行实证评估，系统揭示其弱点及根本原因，并提出了基于大语言模型辅助的自动化测试方法TRUST来改进Rust安全测试工具。

    

    Rust凭借其对内存和并发安全性的强大保证，近年来已成为一种流行的系统编程语言，许多类型的系统软件都已用Rust重新实现。然而，由于不安全（unsafe）代码的存在以及对Rust复杂机制的误解，Rust程序仍然可能遭受常见的内存和并发漏洞。为了缓解此类漏洞，开发者通常依赖静态应用程序安全测试（SAST）工具。然而，现有Rust SAST工具在真实世界漏洞上的有效性仍未得到充分探索。在本文中，我们首先使用真实Rust应用程序中发现的安全漏洞，对主流Rust SAST工具进行了实证评估，系统地评估了它们的有效性，揭示了常见的弱点，并分析了这些弱点背后的根本原因。基于该研究，我们提出了TRUST，一种由大语言模型（LLM）辅助的自动化测试方法。

    arXiv:2610.21373v1 Announce Type: cross  Abstract: Rust has become a popular systems programming language in recent years because of its strong guarantees on memory and concurrency safety. Many types of system software have been re-implemented in Rust. However, Rust programs can still suffer from common memory and concurrency vulnerabilities due to unsafe code and misunderstandings of its complex mechanisms. To mitigate such vulnerabilities, developers often rely on static application security testing (SAST) tools. However, the effectiveness of existing Rust SAST tools on real-world vulnerabilities remains underexplored. In this paper, we first conduct an empirical evaluation of mainstream Rust SAST tools using security vulnerabilities identified in real-world Rust applications. We systematically assess their effectiveness, uncover common weaknesses, and analyze the root causes behind them. Motivated by the study, we propose TRUST, a large language model (LLM)-assisted automated testin
    
[^12]: 智能体驱动的依赖转换：来自Flask探索性案例研究的经验教训

    Agentic Dependency Transformations: Lessons from an Exploratory Case Study of Flask

    [https://arxiv.org/abs/2610.21372](https://arxiv.org/abs/2610.21372)

    本文通过对Flask框架的探索性案例研究，首次提出“依赖转换”这一新兴概念——即AI编程智能体改变软件仓库处理外部依赖的方式，总结出五点初步经验教训，并将其确立为一个具有五部分研究议程的新兴研究方向。

    

    现代软件高度依赖第三方库，维护者需要在复用带来的好处与依赖相关的维护成本和风险之间进行权衡。我们将“依赖转换”定义为一种新兴方法，即由AI编程智能体改变仓库处理外部依赖的方式。本地重新实现是一种转换策略，它将所需功能重新实现为本地代码；而依赖转换的范围更广，可能包含其他策略。在这篇新兴成果论文中，我们报告了一项针对Flask的探索性案例研究，以探索依赖转换策略。我们设计了实验，展示可用上下文如何塑造智能体对依赖的发现以及对依赖的迭代式转换。该案例总结出五点初步经验教训。这些经验教训以及五部分组成的研究议程将依赖转换确立为一个新兴的研究方向，并促使人们提出显式的“转换契约”这一概念。

    arXiv:2610.21372v1 Announce Type: new  Abstract: Modern software relies heavily on third-party libraries, requiring maintainers to balance the benefits of reuse against dependency-related maintenance and risk. We define dependency transformation as an emerging approach in which an AI coding agent alters how a repository handles external dependencies. Local reimplementation is one transformation strategy that recreates required functionality as local code; dependency transformation is broader and may include other strategies. In this emerging-results paper, we report an exploratory case study of Flask to explore dependency transformation strategies. We design experiments that show how available context shapes an agent's dependency discovery and iterative transformation of dependencies. The case yields five preliminary lessons. These lessons and the five-part research agenda position dependency transformation as an emerging research direction and motivate explicit transformation contract
    
[^13]: 结合大语言模型与动态分析实现静态应用程序安全测试工具的通用测试

    Combining LLM and Dynamic Analysis for Universal Testing of Static Application Security Testing Tools

    [https://arxiv.org/abs/2610.21309](https://arxiv.org/abs/2610.21309)

    提出LLM4SAST，首个结合大语言模型与动态反馈来检测静态应用安全测试工具缺陷的通用框架，通过蜕变关系引导的多语言代码变体生成实现自动化测试。

    

    静态应用程序安全测试（SAST）工具有助于检测潜在的软件漏洞，但它们自身也存在缺陷，损害了其可靠性和可用性。自动化测试可以解决这些问题，然而现有技术存在程序变换多样性有限、且缺乏验证所生成变体的机制等问题，导致缺陷检测能力下降且人工成本高昂。我们提出了LLM4SAST，这是首个利用大语言模型和动态反馈来检测SAST工具缺陷的通用框架。我们的核心洞察是利用大语言模型来理解并变换多种编程语言的代码。我们设计了新颖的蜕变关系及相应变换来指导变体生成，并使用大语言模型生成满足这些关系的多语言变体。为了减少错误变体并识别独特缺陷，我们提出了两种启发式方法：动态反馈驱动的选择和基于程序差分的方法。

    arXiv:2610.21309v1 Announce Type: cross  Abstract: Static application security testing (SAST) tools help detect potential software vulnerabilities, but they themselves contain faults undermining reliability and usability. Automated testing can address these issues, yet current techniques suffer from limited program transformation diversity and lack mechanisms to verify generated variants, leading to reduced fault detection and high manual effort. We present LLM4SAST, the first universal framework detecting faults in SAST tools using large language models and dynamic feedback. Our key insight is leveraging LLMs to understand and transform code in diverse programming languages. We design novel metamorphic relations with transformations to guide variant generation, using an LLM to produce multi-language variants that satisfy these relations. To reduce erroneous variants and identify unique faults, we propose two heuristics: dynamic-feedback-driven selection and program-differencing-based 
    
[^14]: ToolBottleneckBench：在受控压力与固定运行框架预算下测量工具智能体性能

    ToolBottleneckBench: Measuring Tool-Agent Performance under Controlled Pressures and Fixed Harness Budgets

    [https://arxiv.org/abs/2610.21098](https://arxiv.org/abs/2610.21098)

    该论文提出ToolBottleneckBench基准，在固定框架预算下通过单独变化工具目录、可见菜单、认证工作流或返回记录编码等压力因素来测量工具智能体性能，发现可见工具菜单是唯一对模型造成大幅惩罚的压力且其根源是完成上限处的截断，而大多数失败实际终止于框架限制而非模型能力不足。

    

    工具智能体的成功得分同时反映了其模型能力和运行它的框架（harness）的预算。我们提出ToolBottleneckBench，这是一个在受控压力与固定框架预算下测量工具智能体性能的基准测试。合成的企业任务保持不变，而每次仅变化一种压力因素：已注册的工具目录、可见的工具菜单、经过认证的工作流程，或返回记录的编码方式。一个精确的评分器会记录每个失败回合中第一个未满足的义务，以及（若存在）终止该回合的框架限制。在21种模型与框架配置中，可见工具菜单是三种作用于模型的压力因素中唯一造成大幅惩罚的因素，且该惩罚与完成上限处的截断相关。将上限提高十六倍后，在截断频发的配置中大部分惩罚被消除。大多数失败终止于某个框架限制，且失败记录对指定压力的识别可靠性并不比（原文在此处截断）……

    arXiv:2610.21098v1 Announce Type: cross  Abstract: The success score of a tool agent reflects both its model and the budgets of the harness that runs it. We present ToolBottleneckBench, a benchmark that measures tool-agent performance under controlled pressures and fixed harness budgets. Synthetic enterprise tasks stay fixed while one pressure varies: the registered tool catalog, the visible tool menu, the certified workflow, or the encoding of returned records. An exact grader records the first unmet obligation of each failed episode and the harness limit, if any, that ended it. Across 21 model and harness configurations, the visible menu is the only one of the three pressures that act on the model with a large penalty, and this penalty tracks truncation at the completion cap. Raising the cap sixteenfold removes most of the penalty in the truncation-heavy configurations. Most failures end at a harness limit, and the failure record identifies the assigned pressure no more reliably than
    
[^15]: 跳过压缩器：编码代理在选择完整文件时能修复更多问题

    Skip the Compressor: Coding Agents Fix More Issues When They Pick Whole Files

    [https://arxiv.org/abs/2610.21073](https://arxiv.org/abs/2610.21073)

    在相同token预算下，与其用压缩器删减文本或检索器挑选代码块，不如将预算用于选择需要原样完整呈现的文件（如CQ2-CARE），这样编码代理能解决更多软件工程问题。

    

    编码代理所处理的代码库可能远超其上下文窗口的容量。提示压缩器和检索器为这一问题提供了解决方案，但修补一个问题需要同时满足两点：必须修改的文件，以及这些文件中原样保留的代码，因为模型必须复现它所编辑的代码行。检索器按照代码与问题的相似度而非是否必须修改来对代码排序，而压缩器则通过删除文本来适应预算限制。我们证明，在相同的token预算下，当预算用于选择哪些文件以原文完整呈现时（正如我们简单的选择器CQ2-CARE所做的那样），代理能解决更多问题，其效果优于将预算用于修剪token或检索代码块。我们在300个最新的SWE-rebench问题上将每个上下文都调整到相同的单问题预算，并使用基准测试的测试用例对每个补丁进行评分。使用GPT-5.6和Claude Opus 5时，CQ2-CARE分别解决了43.7%和46.7%的问题，比三种压缩器中最好的一个高出6.3个百分点。

    arXiv:2610.21073v1 Announce Type: new  Abstract: A coding agent works on repositories that can be far larger than its context window can hold. Prompt compressors and retrievers offer solutions to this problem, but patching an issue needs two things at once: the files that must change, and their code exactly as written, since the model must reproduce the lines it edits. Retrievers rank code by its similarity to the issue rather than by whether it must change, and compressors delete text to fit the budget. We show that at an equal token budget, an agent resolves more issues when the budget goes to choosing which files to show verbatim, as our simple selector CQ2-CARE does, than when it goes to pruning tokens or retrieving chunks. We fit every context to the same per-issue budget on 300 recent SWE-rebench issues and grade each patch with the benchmark's tests. With GPT-5.6 and Claude Opus 5, CQ2-CARE resolves 43.7% and 46.7% of the issues, 6.3 points more than the best of three compressor
    
[^16]: SRAM 的去向：64 KB RISC-V 微控制器上 FreeRTOS-C 与 Embassy-Rust 物联网固件静态内存的逐符号归因分析

    Where the SRAM Goes: Per-Symbol Attribution of Static Memory in FreeRTOS-C and Embassy-Rust IoT Firmware on a 64 KB RISC-V Microcontroller

    [https://arxiv.org/abs/2610.21064](https://arxiv.org/abs/2610.21064)

    提出一种逐符号的静态 SRAM 归因方法，在 64 KB RISC-V 微控制器上将 FreeRTOS-C 与 Embassy-Rust 物联网固件的静态内存精确分解为调度器堆、网络缓冲池、任务栈和链接器保留区域等类别，并与总消耗量完全对账验证。

    

    为内存受限的微控制器选择固件栈时，通常以聚合的二进制大小作为依据，它只能报告一次构建消耗多少 SRAM，却无法说明这些内存去了哪里、为何被消耗。我们用一种逐符号归因方法填补这一空白，该方法将静态 RAM 分解为架构性类别——调度器堆、网络缓冲区与内存池、任务栈以及链接器保留区域——通过将有确定大小的 ELF 符号限定到 .data、.bss 和 .stack 段范围来实现。该方法通过对所有十六次构建的各类别总和与基于 size 的总量进行精确对账而得到验证。我们将其应用于八个累积的固件里程碑，从裸任务脚手架到完整的基于以太网的 MQTT 客户端，分别在 FreeRTOS-C 与 Embassy-Rust 中实现两次，目标平台为 CH32V203RBT6——一款具有 64 KB SRAM 及片上以太网 MAC 和 PHY 的 RISC-V 微控制器，并在每个里程碑上通过硬件验证了功能对等性。该分解显示（原文在此处截断）……

    arXiv:2610.21064v1 Announce Type: cross  Abstract: Selecting a firmware stack for a memory-constrained microcontroller is usually guided by aggregate binary sizes, which report how much SRAM a build consumes but not where it goes or why. We address that gap with a per-symbol attribution method that decomposes static RAM into architectural categories - scheduler heap, network buffers and pools, task stacks, and linker-reserved regions - by bounding sized ELF symbols to the .data, .bss and .stack section ranges. The method is validated by reconciling every category sum exactly against the size-based totals for all sixteen builds. We apply it to eight cumulative firmware milestones, from a bare task scaffold to a full MQTT-over-Ethernet client, implemented twice, in FreeRTOS-C and in Embassy-Rust, on the CH32V203RBT6, a RISC-V microcontroller with 64 KB SRAM and an on-chip Ethernet MAC and PHY, with functional parity verified on hardware at every milestone. The decomposition shows that bo
    
[^17]: 机器人支持的安卓应用数据丢失检测

    Robotic-supported Data Loss Detection in Android Applications

    [https://arxiv.org/abs/2610.20807](https://arxiv.org/abs/2610.20807)

    该论文提出利用机器人自动化技术以非侵入方式检测安卓应用中的数据丢失问题，从而高效应对多样化硬件与操作系统配置下的应用测试挑战。

    

    智能手机因其多样化的应用已成为现代生活中不可或缺的一部分。然而，为这些设备开发应用程序面临重大挑战，主要源于两个因素：(i) 多样化的硬件和操作系统版本需要在多种配置下进行测试；(ii) 以非侵入方式（类似终端用户交互）测试智能手机的必要性，这种方式通常由人工执行，成本高、枯燥乏味且容易出错。在此背景下，机器人自动化已成为应对智能手机系统和版本多样性的有效解决方案，并能以非侵入的方式执行某些测试任务。此外，自动化机器人便于进行快速、准确且可重复的测试，从而使开发人员能够在各种配置下有效地验证其应用程序。这使得验证过程的侵入性更低，并带来显著的……（原文摘要截断）

    arXiv:2610.20807v1 Announce Type: new  Abstract: Smartphones have become integral to modern life due to their diverse applications. However, the development of applications for these devices faces significant challenges, primarily due to two factors: (i) the diverse hardware and operating system versions that require testing across multiple configurations; and (ii) the imperative activity of testing the smartphone in a non-invasive way (similar to an end-user interaction), which is expensive, tedious and error-prone as it is usually carried out manually. In this context, robotic automation has emerged as an effective solution for addressing the diversity of smartphone systems and versions, and in performing some testing tasks non-invasively. In addition, automated robots facilitate quick, accurate, and repetitive testing, thus enabling developers to validate their applications across various configurations effectively. This results in a less invasive verification and in a significant r
    
[^18]: 代码语言模型中的因果语法电路：预算与探针决定了什么看起来是局域化的

    Causal Syntax Circuits in CodeLMs: Budget and Probe Decide What Looks Localised

    [https://arxiv.org/abs/2610.20683](https://arxiv.org/abs/2610.20683)

    该论文通过均值消融、激活修补、贪心电路提取与激活引导的统一流程，系统研究了代码语言模型默认偏好Python的内部机制，发现该偏好随模型规模增长并在指令微调后仍存在，但其局域化表现高度依赖于组件预算与探针的选择。

    

    当提示中没有说明使用哪种语言时，代码模型会默认编写Python。我们针对这一习惯提出了一个机制性问题：它是否由模型内部一小群可找到的组件所产生？我们在55个模型-语言设置上运行了一条统一的实验流程——均值消融、激活修补、贪心电路提取、随机对照显著性检验以及激活引导——涵盖十个开源代码模型（五个基础版和五个指令微调版）、三种目标语言（Rust、C#、PHP）、两套提示集和两种组件预算。我们报告了五项发现。第一，这种偏好很大，随模型规模增长，并且在指令微调后依然存在：每个指令微调版本上的偏好都大于其对应的基础版本，平均高出12%。第二，其测量大小主要取决于探针的属性；用XLCoST平行函数替换我们对齐的提示后，在相同模型上的同一测量值从约1.6降至约0.01。第三，当……（原文在此处截断）

    arXiv:2610.20683v1 Announce Type: new  Abstract: Code models write Python when the prompt does not say which language to use. We ask a mechanistic question about that habit: is it produced by a small, findable group of components inside the model? We run one pipeline---mean ablation, activation patching, greedy circuit extraction, a random-control significance test, and activation steering---over 55 model-language settings spanning ten open code models (five base and five instruction-tuned), three target languages (Rust, C#, PHP), two prompt suites, and two component budgets. We report five findings. First, the preference is large, grows with model size, and survives instruction tuning: it is larger on every instruction-tuned checkpoint than on its base version, by 12% on average. Second, its measured size is mostly a property of the probe; replacing our aligned prompts with XLCoST parallel functions takes the same measurement on the same models from about 1.6 to about 0.01. Third, whe
    
[^19]: HPCAgent-Bench：一个用于代码优化智能体、执行框架与技能的可靠基准测试框架

    HPCAgent-Bench: A Framework for Sound Benchmarking of Code-Optimizing Agents, Harnesses, and Skills

    [https://arxiv.org/abs/2610.20673](https://arxiv.org/abs/2610.20673)

    该论文提出了HPCAgent-Bench——一个包含689个内核、覆盖三个赛道的代码优化基准框架，通过统一的内核规范、外部防作弊裁判以及“干预效力”等新指标，实现对代码优化智能体、执行框架与技能各自贡献的可靠隔离评估。

    

    代码智能体正被越来越多地用于优化代码，但现有的基准测试通常固定单一的加速器、编程模型或执行框架，因此无法隔离各组件各自的贡献，也无法衡量新增的技能和工具是否物有所值。我们提出了HPCAgent-Bench，其中包含689个内核，分为三个赛道：循环级优化挑战、从高性能计算（HPC）代码中提取的内核，以及机器学习模型。该框架用NumPy对每个内核进行一次性定义，推导出C和Fortran参考实现，通过统一的C ABI或Python接口接受提交，并可在单节点或多节点环境下的CPU或GPU上运行。一个外部裁判会在隐藏的模糊测试输入上检查正确性并对每次提交进行计时，从而封堵智能体利用测试系统的途径。我们引入了两个用于报告结果的指标：\textit{干预效力}通过配对运行在加速比、成功率和成本等方面衡量某项干预（例如添加工具）的效果，以及\textit{加权……（摘要在此处截断）

    arXiv:2610.20673v1 Announce Type: new  Abstract: Coding agents are increasingly used to optimize code, but existing benchmarks fix one accelerator, programming model or harness, so they cannot isolate what each component contributes or show whether added skills and tools repay their cost. We present HPCAgent-Bench with 689 kernels in three tracks: loop-level optimization challenges, kernels extracted from HPC codes, and machine-learning models. The framework specifies each kernel once in NumPy, derives C and Fortran references, accepts submissions through a common C ABI or Python interface, and runs them on CPUs or GPUs, on one or multiple nodes. An external judge checks correctness on hidden fuzzed inputs and times each submission, closing paths agents use to exploit the testing system. We introduce two metrics for reporting results: \textit{intervention efficacy} measures an intervention, such as adding a tool, through paired runs on speedup, success rate and cost, and \textit{weight
    
[^20]: 开发者使用Cursor AI原生集成开发环境的体验

    Developer Experiences with Cursor AI-native IDE

    [https://arxiv.org/abs/2610.20618](https://arxiv.org/abs/2610.20618)

    本研究通过分析Cursor官方论坛两年多来的14,714条帖子，大规模纵向考察了开发者使用AI原生IDE的真实体验、采用摩擦及工作流演变。

    

    AI驱动的集成开发环境（IDE）日益被视为软件开发的实用工具，将AI辅助融入编辑、重构和调试等日常活动中。然而，开发者如何评估AI IDE作为工作环境、哪些摩擦因素影响了他们的采用、以及在日常使用中哪些体验方面最为重要，这些问题仍未得到充分探索。此前关于采用和使用的证据主要来自受控任务、小型访谈样本和短时间观察窗口，这些方法对于纵向的、社区驱动的体验以及日常实践中出现的工作流漂移所提供的可见性有限。为了弥合这一差距，我们通过Cursor的官方公共论坛对其开展了大规模纵向研究。Cursor是一款知名的AI原生IDE，用户在该论坛上报告错误、请求功能并讨论日常使用。我们的分析涵盖了两年多时间里收集的14,714条论坛帖子。

    arXiv:2610.20618v1 Announce Type: new  Abstract: AI-powered integrated development environments (IDEs) are increasingly positioned as practical tools for software development, integrating AI assistance into everyday activities such as editing, refactoring, and debugging. However, how developers evaluate AI IDEs as working environments, what frictions shape their adoption, and which aspects of the experience matter most in everyday use remain underexplored. Prior evidence on adoption and use is mainly controlled tasks, small interview samples, and short observation windows, which offer limited visibility into longitudinal, community-driven experience and the workflow drift that emerges in everyday practice. To bridge this gap, we conduct a large-scale longitudinal study of Cursor, a prominent AI-native IDE, through its official public forum, where users report bugs, request features, and discuss everyday use. Our analysis covers 14,714 forum posts collected over more than two years. We 
    
[^21]: 忒修斯之代码：智能体能否依据恢复的设计文档重建软件？

    The Code of Theseus: Can Agents Rebuild Software from Recovered Design Documents?

    [https://arxiv.org/abs/2610.20068](https://arxiv.org/abs/2610.20068)

    提出一个从紧凑仓库表示中恢复相互对齐的意图、需求与架构文档的框架，并发布覆盖 10,764 个多语言仓库的工件数据集，用于研究智能体能否凭借恢复的设计文档重建软件。

    

    LLM 智能体越来越多地在实现系统之前先进行设计，然而用于训练和评估它们的数据却集中在实现与测试上：需求与设计工件常常缺失、过时或属于私有。相比之下，源代码却十分丰富。以往的工作通过将源代码直接输入 LLM 或让智能体探索源代码来恢复架构文档，但这会耗尽上下文窗口、难以扩展到大量仓库，且只能产出单个工件而非相互对齐的一组工件。我们提出了一个框架，能够从紧凑的仓库表示、静态依赖图和关键上下文文件中恢复相互对齐的利益相关者意图、需求与架构，并发布了来自 10,764 个 Java、Python、C/C++ 和 C# 仓库的工件作为数据集。LLM 评审认为三个生成器产出的工件清晰且内部一致，但在正确性和恰当的抽象层级方面可靠性较低。为检验这些工件是否（摘要原文在此处截断）……

    arXiv:2610.20068v1 Announce Type: new  Abstract: LLM agents increasingly design a system before implementing it, yet the data to train and evaluate them centres on implementation and testing: requirements and design artefacts are often missing, outdated or private. Source code, in contrast, is abundant. Prior work recovers architecture documentation by feeding LLMs the source code or letting agents explore it, which saturates the context, does not scale to many repositories, and yields a single artefact rather than aligned ones.   We present a framework that recovers aligned stakeholder intent, requirements and architecture from a compact repository representation, a static dependency graph and key contextual files, and release the artefacts of 10,764 Java, Python, C/C++ and C# repositories as a dataset. An LLM judge rates the artefacts of three generators as clear and internally consistent, but less reliably correct and at the right level of abstraction.   To test whether the artefact
    
[^22]: SWE-Nexus：从测试执行中自动构建软件工程任务

    SWE-Nexus: Automated Composition of Software Engineering Tasks from Test Execution

    [https://arxiv.org/abs/2610.20038](https://arxiv.org/abs/2610.20038)

    SWE-Nexus 是一个完全自动化的流水线，通过追踪测试执行中生产函数的足迹、验证屏蔽效果并确定性组合兼容作用域，自动构建出涵盖 105 个仓库、共 600 个任务的仓库级软件工程实现基准。

    

    历史代码变更提供了便利的任务边界，但无法识别所选测试究竟执行了哪些生产函数，也无法支持超越单次变更的已验证行为组合。我们提出了 SWE-Nexus，这是一个完全自动化的流水线，能够从测试执行中构建仓库级的实现任务。四个构建智能体负责选择测试、为每个仓库适配追踪与评估协议，并生成行为规范。对于每个选定的测试文件，追踪会记录生产函数体的观测执行足迹。仅当在原始仓库中所有预期评分点均通过、且屏蔽被追踪的函数体会导致至少一个失败或错误时，该作用域才会被保留。兼容的作用域以确定性的方式进行组合，共享函数仅计算一次，并且每个组成测试文件都会在完整的组合屏蔽下重新验证。该基准测试包含来自六个（领域）的 105 个仓库的 600 个任务……（摘要原文在此处截断）

    arXiv:2610.20038v1 Announce Type: new  Abstract: Historical code changes provide convenient task boundaries, but do not identify which production functions selected tests exercise or support composing validated behaviors beyond a single change. We present SWE-Nexus, a fully automated pipeline that constructs repository-level implementation tasks from test execution. Four construction agents select tests, adapt tracing and evaluation protocols to each repository, and generate behavioral specifications. For each selected test file, tracing records an observed execution footprint of production function bodies. A scope is retained only if all expected scoring points pass in the original repository and masking the traced bodies causes at least one failure or error. Compatible scopes are composed deterministically, with shared functions counted once and every constituent test file revalidated under the complete combined mask. The benchmark contains 600 tasks from 105 repositories across six 
    
[^23]: ReTerm：面向自我改进终端智能体的恢复训练

    ReTerm: Recovery Training for Self-Improving Terminal Agents

    [https://arxiv.org/abs/2610.18484](https://arxiv.org/abs/2610.18484)

    提出ReTerm终端智能体，通过恢复示范数据集ReTerm-10K进行SFT冷启动，并利用从自身失败状态分叉出多次恢复尝试的在线强化学习算法ReTerm-GRPO，让智能体从自身恢复经验中自我改进，在Terminal-Bench 2.0上取得34.8%的Pass@1成绩。

    

    终端智能体在复杂编程任务中展现出强大的能力。然而，它们在长时间运行过程中往往难以从错误中恢复（例如命令错误、有缺陷的实现），最终导致任务失败。为此，我们提出了ReTerm，一种通过学习如何从自身失败的终端任务中进行恢复而训练的自我改进终端智能体。为支持其训练，我们构建了一个定制化的数据流水线，并生成了ReTerm-10K，这是一个高质量数据集，为SFT冷启动提供了有效的恢复示范。此外，我们提出了ReTerm-GRPO，一种在线强化学习算法，它从智能体自身的失败状态分叉出多个独立的自我恢复尝试。通过学习这些尝试是否成功完成任务，智能体通过自身的恢复经验实现自我改进。基于Qwen3.5-9B构建的ReTerm-9B在Terminal-Bench 2.0上实现了34.8%的Pass@1，较基础模型有所提升。

    arXiv:2610.18484v1 Announce Type: new  Abstract: Terminal agents have shown strong abilities in complex coding tasks. However, they often struggle to recover from errors during long-running execution (e.g., command errors, faulty implementations), eventually leading to task failures. To this end, we present ReTerm, a self-improving terminal agent trained by learning how to recover from its own failed terminal tasks. To support its training, we construct a customized data pipeline and produce ReTerm-10K, a high-quality dataset that provides effective recovery demonstrations for SFT cold start. Furthermore, we propose ReTerm-GRPO, an online RL algorithm that forks from the agent's own failure states to generate multiple independent self-recovery attempts. By learning from whether these attempts successfully complete the task, the agent self-improves through its own recovery experience. Built on Qwen3.5-9B, our ReTerm-9B achieves 34.8% Pass@1 on Terminal-Bench 2.0, improving over the base
    
[^24]: NosRacer：设备内置网络操作系统中竞态条件的动态检测

    NosRacer: Dynamic Detection of Race Conditions in On-Device Network Operating Systems

    [https://arxiv.org/abs/2610.11875](https://arxiv.org/abs/2610.11875)

    本文提出NosRacer，一个采用两阶段设计的动态分析框架，用于高效检测工业级设备内置网络操作系统（NOS）中由异步消息传递引起的竞态条件。

    

    商用设备内置网络操作系统（NOS）在生产路由器和交换机中运行复杂的控制平面，其中配置更新任务由多个松耦合组件通过异步消息传递执行。这类执行容易产生竞态条件：相同的有序输入命令在消息以不同顺序传递时可能产生不同的结果。竞态条件难以暴露，因为它们通常仅表现为细微的、延迟出现的故障。现有静态分析技术对大规模工业级NOS缺乏可扩展性和精确性，而动态分析技术在试图覆盖异步消息交织空间时会产生巨大的系统执行开销。为解决上述挑战，我们提出了NosRacer，一个面向工业级设备内置NOS的竞态条件检测动态分析框架。NosRacer采用两阶段设计。集中阶段……（摘要原文在此处被截断）

    arXiv:2610.11875v1 Announce Type: cross  Abstract: Commercial on-device network operating systems (NOSes) run complex control planes in production routers and switches, where configuration update tasks are executed by multiple loosely coupled components through asynchronous message passing. Such executions are prone to race conditions: the same ordered input commands may produce different outcomes when messages are delivered in different orders. The race conditions are difficult to expose because they often manifest only as subtle, delayed malfunctions. Existing static analysis techniques lack scalability and precision for large-scale industrial NOSes, while dynamic ones incur substantial system-execution cost when attempting to cover the space of asynchronous message interleavings.   To address the challenges above, we present NosRacer, a dynamic analysis framework for race condition detection in industrial-grade on-device NOSes. NosRacer uses a two-phase design. The concentration pha
    
[^25]: ParanoiaEval：智能体编程中不必要防御性工作的基准测试

    ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding

    [https://arxiv.org/abs/2610.08662](https://arxiv.org/abs/2610.08662)

    提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。

    

    随着编程智能体日益自主地承担真实世界的工作，判断其风险应对措施是否合理已变得尤为重要。现有工作从各自独立的角度评估相关的智能体行为，但缺乏一个统一这些行为的系统性框架。为弥合这一差距，我们提出了ParanoiaEval——首个用于统一评估编程智能体风险应对能力的基准。该基准以软件工程风险管理中成熟的“规避-转移-缓解-接受”（Avoidance-Transfer-Mitigation-Acceptance）框架为基础，将这4种基本风险应对措施操作化到编程智能体场景中，并包含200对证据受控的仓库级任务对，每对任务仅在定义应对措施的证据上存在差异。我们进一步引入了针对风险应对违规和证据响应性的专用指标，并采用经过人类校准的智能体裁判以实现可靠评估。在8个代表性模型上进行的大规模实验……

    arXiv:2610.08662v1 Announce Type: new  Abstract: As coding agents increasingly undertake real-world work autonomously, judging whether their risk treatments are warranted has become important. Existing work evaluates related agent behaviors from separate perspectives, but lacks a systematic framework for unifying these behaviors. To bridge this gap, we introduce ParanoiaEval, the first benchmark for unified evaluation of risk-treatment capabilities in coding agents. Grounded in the well-established Avoidance-Transfer-Mitigation-Acceptance framework in software engineering risk management, ParanoiaEval operationalizes its 4 fundamental treatments for coding-agent settings and contains 200 evidence-controlled repository-level task pairs, each differing only in treatment-defining evidence. We further introduce dedicated metrics for risk-treatment violations and evidence responsiveness, using a human-calibrated agentic judge for reliable evaluation. Large-scale experiments on 8 representat
    
[^26]: 基于保证契约与证据的量子编译器测试与验证

    Testing and Verification of Quantum Compilers through Assurance Contracts and Evidence

    [https://arxiv.org/abs/2610.00255](https://arxiv.org/abs/2610.00255)

    该论文是一项关于量子编译器测试与验证的批判性综合综述，系统比较了经验证变换、等价性检查、差分与蜕变测试、基于属性的测试等多种方法，并通过失败案例和工作示例说明如何借助保证契约与证据确保量子编译器在其交付接口上的正确性。

    

    量子编译器的保证要求语义关系与预期用途相匹配，并需要能够在交付接口处暴露违规行为的观测。这项批判性综合综述比较了经验证的变换、等价性检查、差分测试与蜕变测试、基于属性的测试，以及跨修订版本的保证证据。通过一个源文献目录和聚焦的提取矩阵，该综述区分了形式化保证、报告的实证结果、分析性推论以及所提出的实践方法。所记录的覆盖范围更新和源层面的选择决策使该综述在其声明的范围内可被审计。四个有记录的失败案例将条件适用性、参数关联、终止性与相位约定与方法选择联系起来。一个涉及布局元数据、参数绑定和辅助量子比特的工作示例演示了为什么在重构解释下验证电路可能会使交付……（摘要在此处截断）

    arXiv:2610.00255v1 Announce Type: new  Abstract: Quantum compiler assurance requires a semantic relation that matches the intended use and observations that can expose violations at the delivered interface. This critical integrative survey compares verified transformations, equivalence checking, differential and metamorphic testing, property-based testing, and evidence for assurance across revisions. A source catalogue and focused extraction matrix distinguish formal guarantees, reported empirical findings, analytical deductions, and proposed practice. A recorded coverage update and source-level selection decisions make the review auditable within its declared scope. Four documented failure cases connect conditional applicability, parameter association, termination, and phase conventions to method selection. A worked example involving layout metadata, parameter bindings, and ancillary qubits demonstrates why validating a circuit under a reconstructed interpretation can leave the delive
    
[^27]: 从迁移到校准：跨模型、司法管辖区与规模保持智能体能力

    From Migration to Calibration: Preserving Agent Capabilities across Models, Jurisdictions, and Scale

    [https://arxiv.org/abs/2609.35149](https://arxiv.org/abs/2609.35149)

    提出将智能体校准构建为标准优先的适配框架，通过定义基础能力、技术环境与用户情境标准，系统诊断差距并应用修订，从而在模型更换、跨管辖区部署和规模化过程中保持智能体能力不退化。

    

    部署、迁移或扩展智能体可能会改变其模型、运行框架、基础设施、应用场景及目标用户。我们将智能体校准表述为标准优先的适配方法：定义基础能力、技术环境和用户情境标准；诊断差距；生成并应用修订；并在固定预算内重新检查相同标准。这些标准族在信息层、运行框架层和用户接受层之间相互作用。源行为仅具诊断价值，并非完美的参考或能力上限：模型替换可能将正确答案变为错误，或将错误变为正确。资格认定要求通过所有强制性已知测试、真实的端到端部署路径、硬性谓词以及声明的任务/用户最低标准；总体收益不能抹去硬性失败。修订可能更改工具或运行框架，添加示范和任务描述，或使用经验证的目标原生轨迹来训练策略

    arXiv:2609.35149v2 Announce Type: replace  Abstract: Deploying, migrating, or scaling an agent can change its model, harness, infrastructure, application, and intended users. We formulate agent calibration as standards-first adaptation: define basic-capability, technical-environment, and user-context standards; diagnose gaps; generate and apply revisions; and recheck the same standards within fixed budgets. These standard families interact across information, harness, and user-acceptance layers. Source behavior is diagnostic, not a perfect reference or capability ceiling: model replacement can turn correct answers into errors or errors into correct answers. Qualification requires all mandatory known tests, actual end-to-end deployment paths, hard predicates, and declared task/user minimums to pass; aggregate gains cannot erase hard failures. Revisions may change tools or harnesses, add demonstrations and task descriptions, or use validated target-native trajectories to train a policy, 
    
[^28]: SWE-PolyVision：面向仓库级软件工程的跨图像溯因推理基准测试

    SWE-PolyVision: Benchmarking Cross-Image Abductive Reasoning for Repository-Level Software Engineering

    [https://arxiv.org/abs/2609.29754](https://arxiv.org/abs/2609.29754)

    提出 SWE-PolyVision 基准，通过 92 个包含多图像输入的真实软件工程任务，首次系统评估智能体将跨图像视觉证据与代码证据整合并进行经验证的仓库级修复的能力，发现视觉访问的效果因模型和任务而异。

    

    当前多模态软件工程基准测试仅将图像作为额外的上下文呈现，但并未检验智能体能否将分布在多张图像中的证据整合到经过验证的仓库级修复中。我们提出 SWE-PolyVision，这是一个可执行的基准测试，包含来自 36 个开源组织的 92 个真实任务，其中 48 个为公开任务，44 个为私有保留任务。该发布版本包含 402 张静态图像和 6 个视频，每个任务至少有两个视觉输入。每个任务将固定的修复前仓库与隔离的验证器配对，并在三种访问模式（纯文本、原生视觉和工具介导视觉）中受支持的条件进行评估。在十一个代码模型上的实验表明，视觉访问会改变哪些任务能够被解决，但其效果同时取决于模型和任务。两个与推理轨迹关联的原生视觉案例展示了互补的视觉与文本线索如何引导实现源代码定位且经过验证的修复；受控干预实验表明，这种……（原文摘要在此处截断）

    arXiv:2609.29754v1 Announce Type: new  Abstract: Current multimodal software-engineering benchmarks expose images as additional context, but do not test whether an agent can integrate evidence distributed across images into a verified repository-level repair. We present SWE-PolyVision, an executable benchmark of 92 real tasks from 36 open-source organizations, with 48 public tasks and 44 private holdouts. The release contains 402 static images and 6 videos, with at least two visual inputs per task. Each task pairs a fixed pre-fix repository with an isolated verifier and is evaluated under the supported conditions among three access modes: Text-only, Native Vision, and Tool-mediated Vision. Across eleven coding models, visual access changes which tasks are solved, but effects depend on both model and task. Two trace-linked Native Vision cases illustrate how complementary visual and textual clues can lead to source-localized, verified repairs; controlled interventions show that this conv
    
[^29]: 面向基于DEVS的数字孪生仿真服务的以模型为中心的DevOps架构

    A Model-Centric DevOps Architecture for DEVS-Based Digital Twin Simulation Services

    [https://arxiv.org/abs/2609.11122](https://arxiv.org/abs/2609.11122)

    本文提出一种以模型为中心的DevOps架构，将基于DEVS的数字孪生仿真模型作为一等制品，通过声明式YAML定义、到multiPDEVS的形式化映射、CI/CD流水线中的结构与语义验证以及Kubernetes容器化微服务部署，实现了仿真模型的可追溯版本管理与持续交付。

    

    数字孪生仿真模型像软件一样不断演化和重新部署，然而基于DEVS的引擎虽然提供了健全的形式化基础，但在版本管理、自动化验证以及云原生环境中的持续交付方面支持甚少，导致大多数部署中的模型生命周期管理呈现临时性、缺乏规范的状态。本文提出了一种以模型为中心的DevOps架构，用于将基于DEVS的数字孪生仿真部署为托管服务。仿真模型被视为一等DevOps制品，采用声明式YAML语言定义，并具有到multiPDEVS的形式化映射，支持在CI/CD流水线中进行结构和语义验证，该流水线生成不可变的版本化制品，因此回滚到先前已验证的版本只需固定其标识符即可。该平台被分解为Kubernetes上的容器化微服务，并针对状态外置化和生命周期控制进行了引擎适配。在里加22路公交（案例研究中）……（摘要截断）

    arXiv:2609.11122v1 Announce Type: new  Abstract: Digital twin simulation models are evolved and redeployed like software, yet DEVS-based engines offer a sound formal basis with little support for versioning, automated validation, or continuous delivery in cloud-native environments, leaving model lifecycle management ad hoc in most deployments. This paper proposes a model-centric DevOps architecture for deploying DEVS-based digital twin simulations as managed services. Simulation models are treated as first-class DevOps artefacts defined in a declarative YAML language with a formal mapping to multiPDEVS, supporting structural and semantic validation in a CI/CD pipeline that produces immutable versioned artefacts, so that reverting to an earlier validated version reduces to pinning its identifier. The platform is decomposed into containerised microservices on Kubernetes, with engine adaptations for state externalisation and lifecycle control. An initial case study on the Riga Route 22 pu
    
[^30]: Praxist：从实验性人工制品到解决方案谱系

    Praxist: From Experimental Artifacts to Solution Lineages

    [https://arxiv.org/abs/2608.25955](https://arxiv.org/abs/2608.25955)

    Praxist通过谱系中心方法将实验人工制品转化为类型化证据图，分离局部构建与证据综合，使研发代理能继承已验证机制，避免重复学习并提升长期研发效率。

    

    arXiv:2608.25955v1 公告类型：交叉 摘要：自主研发代理现在能够在自动评估下编写、运行和改进可执行人工制品——但很大程度上仍作为实验室工具：在精选基准上展示，其改进难以追溯到具体原因，且成本远超持续工程实践所能承受的范围。这种局限性是结构性的。大多数系统将每次尝试视为几乎自包含的过程，因此日志、记忆和搜索树记录了发生的事件，却未确立哪个设计元素产生了改进、其证据是否经受住了验证，或它如何与其他元素重组。因此，长期活动不断重复学习相同的教训。我们引入了Praxist，一个以谱系为中心的世代系统，将可复现的人工制品和评估结果转化为一个类型化的证据图，包含发现、车道结构前沿和议程。将局部人工制品构建与队列级证据综合分离，使后续尝试能够继承经过验证的机制。

    arXiv:2608.25955v1 Announce Type: cross  Abstract: Autonomous R\&D agents now write, run, and improve executable artifacts under automated evaluation---but largely as laboratory instruments: shown on curated benchmarks, with gains that are hard to trace to a cause and costs well above what sustained engineering practice absorbs. The limitation is structural. Most systems treat each attempt as nearly self-contained, so logs, memories, and search trees record what happened without establishing which design element produced an improvement, whether its evidence survived validation, or how it recombines with others. Long campaigns therefore keep re-learning the same lessons. We introduce Praxist, a lineage-centered generational system that converts reproducible artifacts and evaluator outcomes into a typed evidence graph of findings, lane-structured frontiers, and agendas. Separating local artifact construction from cohort-level evidence synthesis lets later attempts inherit validated mecha
    
[^31]: SPECMINE：一个大规模的规范驱动开发工件语料库

    SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts

    [https://arxiv.org/abs/2608.25202](https://arxiv.org/abs/2608.25202)

    我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。

    

    arXiv:2608.25202v1 公告类型：新 摘要：规范驱动开发（SDD）是一种快速兴起的新实践，其中由开发者编写、或（更常见地）由AI工具起草再由开发者整理的、结构化自然语言规范，驱动AI编码代理的实现。自2025年以来，一波工具（如GitHub Spec Kit [3]、OpenSpec [4]、AWS Kiro [5]以及数十种其他工具）已经出现，但这些工具产生的工件从未被大规模研究过。我们提出了SPECMINE，一个通过两次普查捕捉公共GitHub仓库中SDD的语料库：一次广泛普查覆盖了大多数工具的spec.md/specs.md文件（涵盖73,030个仓库中的470,795个文件，归属于17个命名工具），以及一次针对Kiro独特的需求/设计/任务布局的普查（涵盖12,910个仓库中的98,574个文件）。每个规范都附有完整的仓库元数据、完整的提交历史以及解析后的文档结构。规范如何转化为代码本身就是一个开放问题，因此对于...

    arXiv:2608.25202v1 Announce Type: new  Abstract: Spec-Driven Development (SDD) is a fast-emerging practice in which a structured natural-language specification, written by a developer, or (more often) drafted by an AI tool and then curated by the developer, drives an AI coding agent's implementation. A wave of tooling (GitHub Spec Kit [3], OpenSpec [4], AWS Kiro [5], and dozens of others) has appeared since 2025, yet the artifacts these tools produce have never been studied at scale. We present SPECMINE, a corpus that captures SDD in public GitHub repositories through two censuses: a broad census of spec.md/specs.md files covering most tools (470,795 files across 73,030 repositories, attributed to 17 named tools), and a Kiro census of its distinct requirements/design/tasks layout (98,574 files across 12,910 repositories). Each spec is enriched with full repository metadata, complete commit history, and parsed document structure. How a spec becomes code is itself an open question, so fo
    
[^32]: 不要重新生成，而要调试：一个用于修复“差一点成功”加速器内核的领域专用智能体

    Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels

    [https://arxiv.org/abs/2608.02712](https://arxiv.org/abs/2608.02712)

    该论文提出“调试而非重新生成”的范式转变，设计了一个领域专用调试智能体，用于修复那些能够编译运行但未通过数值验证的“差一点成功”加速器内核，从而将原本被丢弃的失败候选转化为可复用的知识。

    

    arXiv:2608.02712v2 公告类型: replace-cross 摘要：面向GPU和NPU等硬件加速器的内核生成已成为大语言模型（LLM）的试金石，最先进的系统通过将LLM与智能体强化学习和进化搜索相结合的流水线来提升正确性。此类流水线会生成、编译并执行大量候选内核，却丢弃其中的大部分，错失了将失败提炼为可复用知识的机会。许多被丢弃的候选者是“差一点成功”的算子——它们能够编译和运行，但无法通过数值验证；每一个都蕴含着真实的领域知识，以及在LLM推理、交叉编译和硬件执行方面的可观投入。我们主张一次范式转变：与其重新生成，不如调试。调试比从头生成受到的约束多得多：搜索空间小，反馈密集。我们提出一个领域专用的调试智能体，以解决自动化内核……

    arXiv:2608.02712v2 Announce Type: replace-cross  Abstract: Kernel generation for hardware accelerators such as GPUs and NPUs has become a proving ground for large language models (LLMs), and state-of-the-art systems raise correctness through pipelines that couple LLMs with agentic reinforcement learning and evolutionary search. Such pipelines generate, compile, and execute large numbers of candidate kernels, discarding most of them and forgoing the opportunity to distill failures into reusable knowledge. Many discarded candidates are near-miss operators that compile and run but fail numerical validation; each embodies genuine domain knowledge and a nontrivial investment in LLM inference, cross-compilation, and hardware execution. We argue for a paradigm shift: rather than regenerate, debug. Debugging is far more constrained than generating from scratch: the search space is small and feedback is dense. We present a domain-specific debug agent that addresses three core challenges in auto
    
[^33]: 面向可扩展搜索的无标识符代码嵌入模型

    Identifier-Free Code Embedding Models for Scalable Search

    [https://arxiv.org/abs/2605.05251](https://arxiv.org/abs/2605.05251)

    该论文形式化了源代码与去除符号的反编译代码之间的双向函数关联问题，并通过对比学习微调Qwen3-Embedding模型，在所有基线测试中大幅超越现有模型。

    

    函数关联对二进制逆向工程师来说是一项非常有用的技术。目前已有可在大规模场景下执行关联的搜索工具，但它们并未充分利用AI赋能搜索所提供的全部能力。先前的工作已探索了为某些逆向工程代码表示之间的关联开发嵌入模型，但这些工作并未涵盖在标准预处理要求下源代码与去除符号的反编译代码之间的双向关联。为了填补这一空白，我们对这一函数关联问题进行了形式化定义，并评估了嵌入模型在这两种代码表示之间实现双向关联的能力。为提升模型在该任务上的性能，我们使用对比学习对Qwen3-Embedding模型进行了微调。我们发现，我们的新模型在所有函数关联基线测试中均以显著优势超越其他模型，并能泛化至常数算法关联等场景（原文在此处截断）。

    arXiv:2605.05251v2 Announce Type: replace-cross  Abstract: Function association is a useful process for binary reverse engineers. Search tools exist to perform association at scale, but they do not utilize the full range of capabilities that AI-enabled search provides. Prior work has explored the development of embedding models for association between certain reverse engineering code representations, but that work does not cover bidirectional association between source code and decompiled, stripped code with standard preprocessing requirements. To bridge this gap, we formalize this function association problem and evaluate the extent to which embedding models can bidirectionally associate between these two representations. To improve model performance at this task, we fine-tune a Qwen3-Embedding model with contrastive learning. We find that our new model outperforms other models on all function association baselines by a substantial margin and generalizes to a constant-algorithm associ
    
[^34]: 在NISQ量子设备上隔离重复出现的后端相关异常模式

    Isolating Recurring Backend-Dependent Abnormal Patterns on NISQ Quantum Devices

    [https://arxiv.org/abs/2604.17519](https://arxiv.org/abs/2604.17519)

    提出了QRisk框架，能够从量子硬件执行中隔离出重复出现的、依赖特定后端的异常门序列，从而指导门排序重写，弥合编译器噪声模型与真实设备行为之间的差距。

    

    量子硬件仍然极易受噪声影响，这使得编译成为在现有设备上获得可靠执行的关键步骤。当今的量子编译器使用基于校准数据得到的门、量子比特和耦合边开销来近似硬件噪声，并利用这些开销指导量子比特映射、路由和调度。然而，这种抽象是不完整的：编译器预测硬件噪声完全相同的语义等价电路，在真实设备上仍可能表现出截然不同的行为。执行上下文可能是造成这一差距的原因之一：在特定量子比特邻域上执行的短有序门序列可能与硬件模型失配相关联。识别这些依赖于后端的异常模式，可以指导在观察到这些模式的电路族和布局中进行门排序重写。为了找出这些模式，我们提出了QRisk——一个从量子硬件执行中分离出重复异常门序列的框架。

    arXiv:2604.17519v3 Announce Type: replace  Abstract: Quantum hardware remains highly susceptible to noise, making compilation a critical step for obtaining reliable execution on existing devices. Today's quantum compilers approximate hardware noise using calibration-derived costs over gates, qubits, and coupling edges, and use the costs to guide qubit mapping, routing, and scheduling. However, this abstraction is incomplete: semantically equivalent circuits with identical compiler-predicted hardware noise can still behave very differently on real devices. Execution context can contribute to this gap: short ordered gate sequences on specific qubit neighborhoods can be associated with hardware-model mismatch. Identifying these backend-dependent abnormal patterns can guide gate-ordering rewrites in the circuit family and layout where they were observed.   To find these patterns, we propose QRisk, a framework that isolates recurring abnormal gate sequences from quantum hardware executions.
    
[^35]: ORACLE-SWE：量化关键信息信号对软件工程代理的贡献

    ORACLE-SWE: Quantifying the Contribution of Critical Information Signals on SWE Agents

    [https://arxiv.org/abs/2604.07789](https://arxiv.org/abs/2604.07789)

    该论文提出ORACLE-SWE，首次系统量化了复现测试、回归测试、编辑位置、执行上下文和API用法等关键中间信息信号对软件工程代理任务成功的贡献，并测量了代理发现的信号与理想神谕信号之间尚存的改进空间。

    

    语言模型（LM）代理的最新进展日益聚焦于仓库级软件工程（SWE）任务，由此产生了多种代理式工作流、训练方法和失败分析技术。有趣的是，这些方法并非直接生成或评估编辑补丁，而是首先检查并改进在代理探索过程中收集到的中间信息信号，包括复现测试、回归测试、编辑位置、执行上下文和API用法。尽管不同的研究考察了这些信号的不同子集，但目前尚无工作系统地比较它们在提升最终任务成功率方面的相对有效性，也没有测量每个信号中还剩余多大的改进空间。为填补这一空白，本工作量化了“神谕”信号与代理提取的信号对最终任务成功的贡献，并测量了当前代理所发现的信号与其对应神谕信号之间的差距。

    arXiv:2604.07789v3 Announce Type: replace-cross  Abstract: Recent advances in Language Model (LM) agents have increasingly focused on repository-level software engineering (SWE) tasks, leading to various agentic workflows, training methods and failure analysis. Interestingly, rather than directly generating or evaluating edit patches, these approaches first examine and improve intermediate information signals collected during agent exploration, including Reproduction Test, Regression Test, Edit Location, Execution Context, and API Usage. Although different studies examine different subsets of these signals, no work has systematically compared their relative effectiveness in improving final task success or measured the remaining space for improvement in each signal. To address this gap, this work quantifies the contributions of both oracle and agent-extracted signals to final task success and measures the gap between current agent-found signals and their oracle counterparts. We hope our
    
[^36]: SEVerA：自演化智能体的经验证合成

    SEVerA: Verified Synthesis of Self-Evolving Agents

    [https://arxiv.org/abs/2603.25111](https://arxiv.org/abs/2603.25111)

    提出了SEVerA框架，通过形式化守护生成模型（FGGM）将自演化智能体代码生成形式化为受约束的学习问题，从而为自演化LLM智能体提供安全性和正确性的形式化保证。

    

    近期的研究进展表明，自演化的大语言模型（LLM）智能体在程序修复和科学发现等任务上展现出了有效性。在这一范式中，规划器LLM会合成一个调用参数化模型（包括LLM）的智能体程序，然后针对每个任务对这些模型进行调优以提升性能。然而，现有的自演化智能体框架并未提供安全性或正确性的形式化保证。由于此类程序通常在未见过的输入上自主执行，这种保证的缺失引发了可靠性和安全性方面的担忧。我们将智能体代码生成问题形式化为一个受约束的学习问题，将硬性的形式化规范与捕捉任务效用的软目标相结合。我们提出了形式化守护生成模型（FGGM），它允许规划器LLM使用一阶逻辑为每次生成模型调用指定形式化的输出契约。每次FGGM调用都会将底层模型封装在一个带有经验证组件的拒绝采样器中……

    arXiv:2603.25111v3 Announce Type: replace  Abstract: Recent advances have shown the effectiveness of self-evolving LLM agents on tasks such as program repair and scientific discovery. In this paradigm, a planner LLM synthesizes an agent program that invokes parametric models, including LLMs, which are then tuned per task to improve performance. However, existing self-evolving agent frameworks provide no formal guarantees of safety or correctness. Because such programs are often executed autonomously on unseen inputs, this lack of guarantees raises reliability and security concerns. We formulate agentic code generation as a constrained learning problem, combining hard formal specifications with soft objectives capturing task utility. We introduce Formally Guarded Generative Models (FGGM), which allow the planner LLM to specify a formal output contract for each generative model call using first-order logic. Each FGGM call wraps the underlying model in a rejection sampler with a verified 
    
[^37]: LLMON：一种在LLM接口层面利用结构与语义的LLM原生标记语言

    LLMON: An LLM-native Markup Language to Leverage Structure and Semantics at the LLM Interface

    [https://arxiv.org/abs/2603.22519](https://arxiv.org/abs/2603.22519)

    提出了一种LLM原生标记语言LLMON，能够在LLM接口层面自然地传达文本的结构与语义元数据，从而提升模型的准确性、安全性以及抵抗提示注入攻击的能力。

    

    文本大语言模型（LLM）提供了一个简单且为大家熟悉的接口：输入和输出都是一串文本。然而，传递给LLM的信息通常具有更丰富的结构和语义，而这些信息在字符串中并未得到体现。例如，大多数提示词既包含指令（“将这篇论文总结为一段话”），也包含数据（需要总结的论文），但在传递给模型时二者通常不加区分。这可能导致模型混淆以及安全风险，例如提示注入攻击。本工作通过引入一种LLM原生标记语言LLMON（LLM Object Notation，发音为"Lemon"）来解决这一缺陷，该语言能够以自然的方式将文本的结构和语义元数据传达给LLM。这些信息随后可用于模型训练、模型提示和推理实现，从而在模型准确性、安全性和保密性方面带来提升。

    arXiv:2603.22519v3 Announce Type: replace  Abstract: Textual Large Language Models (LLMs) provide a simple and familiar interface: a string of text is used for both input and output. However, the information conveyed to an LLM often has a richer structure and semantics, which is not conveyed in a string. For example, most prompts contain both instructions ("Summarize this paper into a paragraph") and data (the paper to summarize), but these are usually not distinguished when passed to the model. This can lead to model confusion and security risks, such as prompt injection attacks.   This work addresses this shortcoming by introducing an LLM-native mark-up language, LLMON (LLM Object Notation, pronounced "Lemon"), that enables the structure and semantic metadata of the text to be communicated in a natural way to an LLM. This information can then be used during model training, model prompting, and inference implementation, leading to improvements in model accuracy, safety, and security. 
    
[^38]: 通过约束扰动与校准对代码大语言模型评审进行基准测试

    Benchmarking Code LLM-as-a-Judge via Constrained Perturbation and Calibration

    [https://arxiv.org/abs/2512.20159](https://arxiv.org/abs/2512.20159)

    该论文提出通过约束扰动与校准技术构建代码LLM-as-a-Judge评估基准，克服了现有基准中粗粒度二元标签无法区分缺陷严重程度以及细粒度人工评分主观不可靠的局限性。

    

    LLM辅助开发所生成的代码可能包含API幻觉等缺陷，这促使研究人员开发代码评估指标，以便在部署前对此类代码进行评估。由于传统的基于规则的指标仅根据与参考程序的相似性对代码打分，而基于执行的指标需要昂贵的人工测试准备，研究人员开发了用于行为与非行为代码评估的LLM-as-a-judge指标，并整理了多个基准来验证其有效性。然而，这些基准存在关键局限性：一些基准采用通过/未通过单元测试的粗粒度二元标签，无法区分根本性缺陷与轻微缺陷；另一些基准则使用带有手动标注分数作为真实标签的细粒度标准，但此类标准往往定义模糊，导致人工评分存在主观性和不可靠性。此外，它们通常直接提示LLM……

    arXiv:2512.20159v2 Announce Type: replace-cross  Abstract: Code from LLM-assisted development may contain defects like API hallucinations, motivating the development of code evaluation metrics to assess such code before deployment. As conventional rule-based metrics merely score code based on similarity to reference programs, and execution-based metrics require expensive manual preparation for tests, researchers have developed LLM-as-a-judge metrics for both behavioral and nonbehavioral code evaluation, and curated multiple benchmarks to validate their effectiveness. However, such benchmarks suffer from critical limitations: Some feature coarse-grained binary labels of passing/failing unit tests, unable to distinguish between fundamental and minor flaws; others utilize fine-grained criteria with manually annotated scores as ground-truth, but such criteria are often vaguely defined, introducing subjectivity and unreliability in human scores. Furthermore, they often directly prompt LLMs 
    
[^39]: RAG4Tickets：基于检索增强生成的JIRA与GitHub数据驱动的AI工单解决方案

    RAG4Tickets: AI-Powered Ticket Resolution via Retrieval-Augmented Generation on JIRA and GitHub Data

    [https://arxiv.org/abs/2510.08667](https://arxiv.org/abs/2510.08667)

    该论文提出RAG4Tickets框架，通过将Sentence-Transformers语义嵌入、FAISS向量搜索与大语言模型相结合，利用JIRA和GitHub的历史工单及PR数据，为软件工单提供有依据且可解释的解决建议。

    

    现代软件团队经常由于知识碎片化地分散在JIRA工单、开发者讨论和GitHub拉取请求（PR）中，而在解决重复出现或相关问题时遭遇延迟。为了应对这一挑战，我们提出了一种检索增强生成（RAG）框架，该框架将Sentence-Transformers的语义嵌入与基于FAISS的向量搜索相结合，以提供上下文感知的工单解决建议。该方法对历史JIRA工单、用户评论以及相关联的PR元数据进行嵌入，从而检索语义相似的历史案例，随后由大型语言模型（LLM）将这些案例综合为有依据且可解释的解决建议。该框架的贡献包括：连接JIRA和GitHub数据的统一流水线、针对异构软件工件的嵌入与FAISS索引策略，以及由检索证据引导的解决建议生成模块。使用精确率……（摘要原文在此处被截断）

    arXiv:2510.08667v3 Announce Type: replace-cross  Abstract: Modern software teams frequently encounter delays in resolving recurring or related issues due to fragmented knowledge scattered across JIRA tickets, developer discussions, and GitHub pull requests (PRs). To address this challenge, we propose a Retrieval-Augmented Generation (RAG) framework that integrates Sentence-Transformers for semantic embeddings with FAISS-based vector search to deliver context-aware ticket resolution recommendations. The approach embeds historical JIRA tickets, user comments, and linked PR metadata to retrieve semantically similar past cases, which are then synthesized by a Large Language Model (LLM) into grounded and explainable resolution suggestions. The framework contributes a unified pipeline linking JIRA and GitHub data, an embedding and FAISS indexing strategy for heterogeneous software artifacts, and a resolution generation module guided by retrieved evidence. Experimental evaluation using precis
    
[^40]: 面向高质量代码生成的多LLM编排：利用模型互补优势

    Multi-LLM Orchestration for High-Quality Code Generation: Exploiting Complementary Model Strengths

    [https://arxiv.org/abs/2510.01379](https://arxiv.org/abs/2510.01379)

    提出PerfOrch多智能体系统，通过Memory模块记录各LLM在不同开发阶段、编程语言和算法类别上的实测性能排名，从而在代码生成、调试和优化的每个环节动态选用最合适的模型，利用模型间互补优势实现比单一模型更高的代码正确性和执行性能。

    

    大语言模型（LLM）是自动化代码生成的核心，然而大多数方法在从编写代码到修复和优化的每个阶段都只使用单一模型。LLM具有互补的优势：它们在不同编程语言、算法类别和开发阶段上的失败并不完全重叠。例如，在Python中，代码生成准确率最高的模型在代码修复方面却排名最后。因此，使用一个模型生成代码、另一个模型修复代码、第三个模型优化代码，可以比全程使用单一模型获得更高的正确性和更好的执行性能。我们提出了PerfOrch，这是一个多LLM多智能体系统，包含四个分别负责分类、生成、调试和优化的智能体。生成、调试和优化智能体通过一个Memory（记忆）模块来选择模型，该模块根据候选LLM在每个阶段、每种编程语言和每个算法类别上的实测性能对其进行排名。我们……

    arXiv:2510.01379v3 Announce Type: replace  Abstract: Large Language Models (LLMs) are central to automated code generation, yet most approaches use one model at every stage, from writing code to repairing and optimizing it. LLMs have complementary strengths: their failures do not fully overlap across programming languages, algorithmic categories, and development stages. For example, in Python, the model with the highest code-generation accuracy ranks last at code repair. Using one model to generate code, another to repair it, and a third to refine it can therefore achieve higher correctness and better execution performance than a single model throughout. We propose PerfOrch, a multi-LLM multi-agent system with four agents for categorization, generation, debugging, and refinement. The generation, debugging, and refinement agents select models through a Memory module that ranks candidate LLMs by their measured performance for each stage, programming language, and algorithmic category. We
    
[^41]: 通过分层上下文剪枝提升知识密度以实现检索增强的仓库级代码补全

    Enhancing Knowledge Density via Hierarchical Context Pruning for Retrieval-Augmented Repository-Level Code Completion

    [https://arxiv.org/abs/2508.10068](https://arxiv.org/abs/2508.10068)

    提出SaraCoder框架，在固定上下文预算下通过分层上下文剪枝提升检索上下文的知识密度，避免冗余代码浪费token并释放跨文件依赖与相似代码片段的协同作用，从而提升仓库级代码补全性能。

    

    arXiv:2508.10068v3 公告类型：replace-cross。摘要：检索增强生成通过检索外部代码片段、API文档或类型定义来扩展模型的上下文感知能力，从而推动了仓库级代码补全的发展。然而，在上下文窗口有限的token预算下，现有的检索增强方法往往用高度相似甚至冗余的代码填满预算，这不仅浪费了token，还抑制了关键的跨文件依赖，阻碍了语义相似的参考片段与跨文件依赖定义之间的协同交互。因此，性能的进一步提升受到检索上下文中知识密度低的限制，而非模型能力的限制。我们提出了SaraCoder，一个在固定上下文预算下通过分层上下文剪枝来提升知识密度的检索增强框架。其中的分层特征优化模块分四个阶段对超完备的候选池进行剪枝……（摘要原文在此处截断）

    arXiv:2508.10068v3 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) has advanced repository-level code completion by retrieving external code snippets, API documentation, or type definitions to expand the model's contextual awareness. However, under the finite token budget of the context window, existing retrieval-augmented methods often saturate the budget with highly similar or even redundant code, which not only wastes tokens but also suppresses critical cross-file dependencies, preventing synergistic interaction between semantically similar reference snippets and cross-file dependency definitions. Therefore, further gains are limited by low knowledge density in the retrieved context rather than by model capacity. We propose SaraCoder, a retrieval-augmented framework that enhances knowledge density via hierarchical context pruning under a fixed context budget. A Hierarchical Feature Optimization module prunes an over-complete candidate pool in four stages
    
[^42]: EOS区块链的实证分析：架构、合约与安全

    An Empirical Analysis of EOS Blockchain: Architecture, Contract, and Security

    [https://arxiv.org/abs/2505.15051](https://arxiv.org/abs/2505.15051)

    本研究首次从系统架构、去中心化、性能、智能合约和行为安全五个维度对EOS区块链进行了全面的实证分析，揭示了账户活跃度低、超级节点选举参与有限等关键问题。

    

    随着区块链技术的快速发展，各类区块链系统正展现出蓬勃的活力与潜力。作为区块链3.0的代表，EOS区块链一直被视为以太坊的有力竞争者。然而，与比特币和以太坊相比，针对EOS的学术研究和深入分析仍然十分匮乏。为填补这一空白，本研究从五个关键维度对EOS区块链进行了全面调查：系统架构、去中心化、性能、智能合约以及行为安全。架构分析聚焦于EOS系统的六个核心组件，详细阐述了它们的功能和运行流程。基于XBlock数据共享平台数据的去中心化与性能评估揭示了若干关键问题：账户活跃度低、超级节点选举过程参与有限、区块生产者集合变化极小等。

    arXiv:2505.15051v2 Announce Type: replace-cross  Abstract: With the rapid development of blockchain technology, various blockchain systems are exhibiting vitality and potential. As a representative of Blockchain 3.0, the EOS blockchain has been regarded as a strong competitor to Ethereum. Nevertheless, compared with Bitcoin and Ethereum, academic research and in-depth analyses of EOS remain scarce. To address this gap, this study conducts a comprehensive investigation of the EOS blockchain from five key dimensions: system architecture, decentralization, performance, smart contracts, and behavioral security. The architectural analysis focuses on six core components of the EOS system, detailing their functionalities and operational workflows. The decentralization and performance evaluations, based on data from the XBlock data-sharing platform, reveal several critical issues: low account activity, limited participation in the supernode election process, minimal variation in the set of blo
    

