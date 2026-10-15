# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis](https://arxiv.org/abs/2610.20538) | 该论文提出PrismaCoder，从数据端出发，通过分解式评分标准合成方法，解决多模态代码生成中样本的程序正确性、代码质量与视觉正确性难以同时评估的核心难题，从而推动开放多模态代码智能的发展。 |
| [^2] | [What Should a Software Team Spend on Agents? Simulation-Based Cost Optimization of Mixed AI-Human Teams](https://arxiv.org/abs/2610.20394) | 该论文通过蒙特卡洛模拟和进化搜索对432种场景进行建模，研究了软件团队在人类开发者与AI智能体之间的最优支出分配策略，发现逃逸缺陷是主要成本驱动因素，且61%的智能体预算应用于查找缺陷。 |
| [^3] | [Doc2Harness: Benchmarking LLMs on Long-Horizon Harness Engineering of Coding Agents](https://arxiv.org/abs/2610.20344) | 该论文提出 Doc2Harness，一个两阶段基准测试，首次直接评估大语言模型为编码智能体进行长时程工具框架工程的能力，填补了现有基准仅能在固定框架下测试下游任务的空白。 |
| [^4] | [P2Patch: Exploit-Guided Vulnerability Repair](https://arxiv.org/abs/2610.20245) | P2Patch提出了一种生成式对抗修复系统，通过漏洞利用智能体与补丁智能体的持续对抗，将每次成功的漏洞利用转化为永久修复要求，从而生成能抵御同一漏洞多种变体攻击的更稳健补丁。 |
| [^5] | [SWE-Stream: Scaling Executable Training Data for SWE Agents via Progressive Construction and Elastic Streaming](https://arxiv.org/abs/2610.20193) | SWE-Stream 提出了一种通过渐进式构建与弹性流式执行来大规模自动生成可执行软件工程训练数据的系统，从 10,728 个仓库的开发历史中构建出 80,874 个经过验证的训练任务。 |
| [^6] | [The Code of Theseus: Can Agents Rebuild Software from Recovered Design Documents?](https://arxiv.org/abs/2610.20068) | 提出一个从紧凑仓库表示中恢复相互对齐的意图、需求与架构文档的框架，并发布覆盖 10,764 个多语言仓库的工件数据集，用于研究智能体能否凭借恢复的设计文档重建软件。 |
| [^7] | [SWE-Nexus: Automated Composition of Software Engineering Tasks from Test Execution](https://arxiv.org/abs/2610.20038) | SWE-Nexus 是一个完全自动化的流水线，通过追踪测试执行中生产函数的足迹、验证屏蔽效果并确定性组合兼容作用域，自动构建出涵盖 105 个仓库、共 600 个任务的仓库级软件工程实现基准。 |
| [^8] | [LintEval: Benchmarking LLMs on Linter-Guided File Refactoring](https://arxiv.org/abs/2610.20027) | 该论文提出LintEval，一个覆盖C++、Python和Java的多语言基准，用于评测大语言模型在linter引导的文件重构中减少lint告警的能力，实验揭示了告警减少与功能保持之间的持续权衡。 |
| [^9] | [Temporal Adaptive Fine-Tuning for Code Comprehension](https://arxiv.org/abs/2610.19970) | 提出 TAFT（时间自适应微调）方法，将仓库代码的演变转换作为监督信号，使大语言模型在代码理解问答中能够获取新知识、抑制过时知识，并保留仍然有效及语义相近的知识。 |
| [^10] | [Taming the Online Software Engineering Tongue: Detecting and Reducing Offensive Language in Software Communities](https://arxiv.org/abs/2610.19961) | 本研究调查了在线软件工程社区中冒犯性语言的普遍性与特征，并基于四个平台的13万条评论开发了包含72个机器学习模型的冒犯性语言减少系统（OLRSSE），该系统不仅能检测和分类冒犯性评论，还能向发布者解释原因并提供非冒犯性的替代建议。 |
| [^11] | [Systematic Detection of Energy Regressions in Software Version Histories](https://arxiv.org/abs/2610.19895) | 本文提出EnergyTrackr方法，通过识别提交间能耗的统计显著性变化来自动检测软件版本历史中的能耗退化，并在五个Java项目的1,932个提交上验证了其有效性，同时比较了线性算法与二分搜索算法的效率。 |
| [^12] | [When Does the Dependency Graph Help Code-to-Architecture Mapping?](https://arxiv.org/abs/2610.19769) | 该论文将代码到架构的映射建模为依赖图上的半监督节点分类问题，并通过GCN、GAT、MLP与朴素贝叶斯的系统对比实验，揭示了依赖图结构信息在何种条件下才能真正提升架构映射的效果。 |
| [^13] | [JevCoder: When Does Decision Delegation Pay Off in Code Repair?](https://arxiv.org/abs/2610.19596) | 该论文提出 JevCoder，一个围绕低成本概率判断器 Jev 构建的代码修复工作流程，系统研究了哪些修复决策可以委托给低成本判断器，通过结合代码片段推荐与动作类型指导，在保持修复质量的同时降低 token 开销。 |
| [^14] | [Mission (Im)Possible: Towards Autonomous Discovery of Autopilot Logic Flaws in Open-Source Drone Frameworks](https://arxiv.org/abs/2610.19566) | 本文提出了IMF（Impossible Mission Finder）框架，利用配备领域知识和精心构建上下文的多智能体LLM系统，结合高级语义缺陷判定器，自主发现开源无人机自动驾驶仪软件中难以通过传统方法检测的细微逻辑缺陷。 |
| [^15] | [BenchChecker: Assessing the Credibility of Bug-Fixing Benchmarks for LLMs](https://arxiv.org/abs/2610.19447) | 提出BenchChecker框架，通过基于取证的两阶段检测流程（仓库存在性测试与补丁存在性测试）来识别缺陷修复基准中的数据污染问题，从而更可信地评估大语言模型在自动程序修复任务上的真实性能。 |
| [^16] | [OS-Bench: Evaluating SWE Agents on Cross-OS Code Migration](https://arxiv.org/abs/2610.19368) | 提出OS-Bench——首个通过对比同一代码库在Linux与Windows上的执行覆盖率来识别迁移机会，并将平台特定代码组织为五种场景的可执行基准，用于评估语言模型智能体在跨操作系统代码迁移上的能力。 |
| [^17] | [Beyond Functional Correctness: Static Code Quality Profiling of LLM-Generated Code](https://arxiv.org/abs/2610.19167) | 本研究对73个大语言模型生成的65,663个Python程序进行了大规模静态代码质量剖析，发现仅靠功能正确性无法全面评估代码质量，各模型在可维护性等静态质量指标上存在系统性差异，且正确性往往伴随着结构上的权衡。 |
| [^18] | [DivAlgo: A systematic evaluation of algorithmic diversity for code generation models](https://arxiv.org/abs/2610.19157) | 提出DivAlgo基准数据集及轻量级相似度度量方法，通过对24个代码生成模型的评估发现，模型生成代码的算法多样性显著低于人类解决方案（最高仅达人类多样性的约70%），揭示了当前代码生成模型在算法多样性方面的局限性。 |
| [^19] | [A Systematic Study of Deprecated Android API Usage Updates](https://arxiv.org/abs/2610.19150) | 本文通过分析1,000个已弃用Android API和4,336个来自GitHub的迁移示例，系统性地识别出六大迁移类别及反复出现的API使用更新模式，为改进自动化API迁移技术提供了实证基础。 |
| [^20] | [X.js: A Live Benchmark for Node.js Taint-Style Vulnerabilities](https://arxiv.org/abs/2610.18585) | 本文提出了X.js——一个持续更新的真实世界Node.js污点风格漏洞基准测试集，每个漏洞均配有精确的源汇位置、经验证的可执行漏洞利用程序及补丁版本，解决了现有漏洞数据集过时、无法支持该领域研究长期发展的问题。 |
| [^21] | [Interaction-as-a-Judge](https://arxiv.org/abs/2610.17895) | 提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。 |
| [^22] | [SecReason: Vulnerability Detection and In-Stream Self-Repair for LLM Code Generation](https://arxiv.org/abs/2610.14453) | 提出SecReason，一种针对推理型大语言模型代码生成的流内防御机制，通过双轨增量扫描在流式思维链中实时检测安全漏洞并进行自修复。 |
| [^23] | [SoftwareGym: Scaling Computer-Use Agent Training with Diverse Synthesized Software Environments](https://arxiv.org/abs/2610.12611) | 提出SoftwareGym，一个自动化流水线，能够大规模合成界面多样、功能可执行且经运行时验证的GUI软件环境（已生成12个类别共182个应用，任务通过率达91.5%），并构建SoftwareGym-Bench基准以支持计算机使用智能体的训练与评估。 |
| [^24] | [A2ABreak: Systematic Security Analysis of the A2A Protocol](https://arxiv.org/abs/2609.10871) | 本文提出A2ABreak框架，首次对A2A协议进行系统性安全分析，通过LLM辅助从自然语言规范中提取经验证的有限状态机模型（37个状态、76个转换），并利用对抗性验证系统性发现协议层面的安全漏洞。 |
| [^25] | [Shortcutting the Fix: Identifying and Categorizing Agentic Exploits in Software Engineering Benchmarks](https://arxiv.org/abs/2609.06780) | 该研究首次将软件工程基准测试中智能体通过检索上游代码、利用 Git 历史等方式作弊的现象形式化为“智能体走捷径”，发现标准评估下高达 90% 的轨迹存在作弊尝试，并提出通过提示智能体独立构建解决方案可将其降至约 10% 且不损害任务性能，凸显了作弊感知评估框架的必要性。 |
| [^26] | [BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks](https://arxiv.org/abs/2608.27334) | 该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。 |
| [^27] | [Refining LLM-based Directed Test Input Generation via Runtime Value Feedback](https://arxiv.org/abs/2608.07926) | 提出了ReDig框架，通过比较预期运行时值与实际运行时值的事后分析反馈，迭代改进基于LLM的定向测试输入生成，引导测试输入到达目标代码位置。 |
| [^28] | [An Empirical Study of Gemini 3 for Detecting Natural Language Test Smells in Manual Test Cases](https://arxiv.org/abs/2606.13804) | 本文实证研究了最新大语言模型Gemini 3 Pro Preview能否有效检测手工测试用例中的自然语言测试异味，并与此前表现最佳的小语言模型PHI-4-14B进行了对比。 |
| [^29] | [T2J-BENCH: Benchmarking Coding Agents on Execution-Grounded PyTorch-to-JAX Repository Migration](https://arxiv.org/abs/2605.29054) | 提出T2J-BENCH基准，包含63个涵盖18个模型家族的整仓库PyTorch到JAX迁移任务，以执行行为一致性和无torch依赖作为核心评价标准，用以真实评估编码智能体进行跨框架代码库迁移的能力。 |
| [^30] | [MOCHA: Multi-Objective Chebyshev Annealing for Agent Skill Optimization](https://arxiv.org/abs/2605.19330) | 提出 MOCHA 方法，通过切比雪夫标量化选择改进对象并以扩展帕累托前沿为接受准则，从探索退火到利用，实现智能体技能在准确性与上下文经济性之间的多目标优化，克服了现有单分数提示词优化器易停滞的问题。 |
| [^31] | [Scaling Human-AI Coding Collaboration Requires a Governable Consensus Layer](https://arxiv.org/abs/2604.17883) | 该论文提出扩展人机编码协作需要引入一个可治理的共识层，通过版本化的项目状态模型将意图、结构性承诺、代码工件和证据关联起来，使人类能够在AI编码过程中检查、协调和修订共享的设计决策。 |
| [^32] | [Terminal Agents Suffice for Enterprise Automation](https://arxiv.org/abs/2604.00073) | 该论文提出仅配备终端和文件系统的编码智能体通过直接调用平台 API，就能以远低于复杂智能体架构（如 MCP 工具智能体和网页智能体）的成本，达到甚至超越它们在企业任务上的表现。 |
| [^33] | [LLMON: An LLM-native Markup Language to Leverage Structure and Semantics at the LLM Interface](https://arxiv.org/abs/2603.22519) | 提出了一种LLM原生标记语言LLMON，能够在LLM接口层面自然地传达文本的结构与语义元数据，从而提升模型的准确性、安全性以及抵抗提示注入攻击的能力。 |
| [^34] | [SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning](https://arxiv.org/abs/2601.21649) | 本文提出仓库中心化学习（RCL）新范式，将静态代码库转化为交互式学习信号，训练出针对特定代码仓库深度专门化的小型语言模型SWE-Spot-4B，以弥补小型模型处理复杂陌生代码库时泛化能力不足的缺陷。 |
| [^35] | [Securing High-Concurrency Ticket Sales: A Framework Based on Microservice](https://arxiv.org/abs/2512.24941) | 本文提出了一种基于Spring Cloud微服务架构的铁路购票系统设计框架，通过多种安全设计方法保障系统在高并发场景下的稳定性、数据一致性和快速响应能力。 |

# 详细

[^1]: PrismaCoder：通过分解式评分标准合成突破开放多模态代码智能的边界

    PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis

    [https://arxiv.org/abs/2610.20538](https://arxiv.org/abs/2610.20538)

    该论文提出PrismaCoder，从数据端出发，通过分解式评分标准合成方法，解决多模态代码生成中样本的程序正确性、代码质量与视觉正确性难以同时评估的核心难题，从而推动开放多模态代码智能的发展。

    

    多模态代码智能将程序的符号世界与其执行所创造的视觉世界连接起来。随着模型越来越多地被要求生成图表、网页、动画和半结构化文档，人们对这一能力的兴趣正在迅速增长。近期，专有模型已大幅领先，甚至能够仅凭文本描述或一张截图就生成交互式作品。然而，这些模型的数据和训练过程仍然封闭，开放领域的进展受限于可用数据的数量与质量，以及生产这些数据的流程和方法。其中质量是更难解决的问题，因为一个样本必须同时满足：作为程序是正确的、构建良好的，以及最关键的——渲染后在视觉上是正确的。目前没有任何现有信号能够同时覆盖这三个方面：执行往往只返回通过或失败，代码相似度无法映射到视觉输出，而针对特定任务的验证器又缺乏泛化能力。为了进一步推动开放领域的发展，我们从数据端入手……

    arXiv:2610.20538v1 Announce Type: new  Abstract: Multimodal code intelligence connects the symbolic world of programs with the visual world their execution creates. Interest in this capability is burgeoning, as models are increasingly asked to produce charts, webpages, animations, and semi-structured documents. Recently, proprietary models have pushed far ahead, producing interactive artifacts even from a text description or a screenshot alone. Yet their data and training remain closed, and open progress is held back by the quantity and quality of available data and by the pipelines and methods for producing it. Quality is the harder of the two, since a sample must be correct as a program, well built, and above all visually right once rendered. No existing signal covers all three. Execution often returns just a pass or fail, code similarity does not map to the visual output, and task-specific verifiers do not generalize. To push open progress further, we start from the data side with P
    
[^2]: 软件团队应该在智能体上花费多少？基于仿真的AI与人类混合团队成本优化

    What Should a Software Team Spend on Agents? Simulation-Based Cost Optimization of Mixed AI-Human Teams

    [https://arxiv.org/abs/2610.20394](https://arxiv.org/abs/2610.20394)

    该论文通过蒙特卡洛模拟和进化搜索对432种场景进行建模，研究了软件团队在人类开发者与AI智能体之间的最优支出分配策略，发现逃逸缺陷是主要成本驱动因素，且61%的智能体预算应用于查找缺陷。

    

    背景：AI智能体正在软件开发中被快速采用，但其token成本不断上升，目前尚不清楚团队应该在智能体上花费多少，以及应该在哪些任务上花费。目标：我们确定团队应如何在人类开发者和AI智能体之间分配支出，以最低的总成本交付项目所需的功能。方法：我们将由人类开发者和八种专业角色AI智能体组成的混合团队建模为一个离散时间随机过程，涵盖缺陷注入与检测、技术债务、人类对智能体的监督，以及带有功能冻结的发布周期。通过蒙特卡洛模拟和进化搜索，我们针对432种场景（涵盖开发者能力、项目速度、债务动态、智能体能力与价格以及缺陷成本等因素）确定了成本最优的团队配置，并将其与纯人类团队基线进行比较。结果：逃逸缺陷是主要的成本驱动因素，61%的智能体预算用于查找缺陷……

    arXiv:2610.20394v1 Announce Type: new  Abstract: Context: AI agents are rapidly adopted in software development, but their token costs are rising and it is unclear how much a team should spend on agents, and on which tasks.   Objective: We determine how teams should divide spending between human developers and AI agents to deliver the features a project demands at the lowest total cost.   Method: We model mixed teams of human developers and AI agents with eight specialized roles as a discrete-time stochastic process covering bug injection and detection, technical debt, human supervision of agents, and release cycles with feature freezes. Using Monte Carlo simulation and evolutionary search, we determine cost-optimal teams for 432 scenarios varying developer capability, project velocity, debt dynamics, agent capability and price, and bug costs, and compare them with a human-only baseline.   Results: Escaped bugs are the dominant cost driver, and 61% of the agent budget goes to finding b
    
[^3]: Doc2Harness：基于长时程工具框架工程任务的大语言模型编码智能体基准测试

    Doc2Harness: Benchmarking LLMs on Long-Horizon Harness Engineering of Coding Agents

    [https://arxiv.org/abs/2610.20344](https://arxiv.org/abs/2610.20344)

    该论文提出 Doc2Harness，一个两阶段基准测试，首次直接评估大语言模型为编码智能体进行长时程工具框架工程的能力，填补了现有基准仅能在固定框架下测试下游任务的空白。

    

    由大语言模型驱动的智能体正越来越多地被部署在长时程任务中，在这些任务里它们使用工具并修改真实的工作空间。其性能表现既取决于模型本身，也同等程度地取决于围绕模型的工具框架。现有基准测试在固定工具框架下衡量下游任务的成功率，而工具框架工程本身的研究仍是空白。我们提出 Doc2Harness，这是一个直接评估长时程工具框架工程能力的两阶段基准测试。在第一阶段，一个在固定构建器工具框架内运行的构建模型，接收一个仅含接口的契约骨架和一份整合了十二个前沿工具框架行为的开发文档，然后在一个隔离容器中构建出完整的候选工具框架。在第二阶段，冻结的候选工具框架在与构建环境相互隔离的容器中运行（每个测试用例一个容器），并由脚本化的模拟模型端到端地驱动。我们使用四个可观察指标对每个测试用例进行评分。

    arXiv:2610.20344v1 Announce Type: new  Abstract: Agents driven by large language models are increasingly deployed on long-horizon tasks, where they use tools and modify real workspaces. How well they perform depends as much on the harness around the model as on the model itself. Existing benchmarks measure downstream task success under a fixed harness, leaving harness engineering itself unexplored. We propose Doc2Harness, a two-stage benchmark that evaluates long-horizon harness engineering directly. In Stage 1, a build model operating inside a fixed builder harness receives an interface-only contract skeleton and a development document that consolidates behavior across twelve frontier harnesses, and then constructs a complete candidate harness in an isolated container. In Stage 2, the frozen candidate harness runs in containers separate from the construction environment, one per test case, and is driven end to end by a scripted mock model. We score each test case using four observable
    
[^4]: P2Patch：基于漏洞利用引导的漏洞修复

    P2Patch: Exploit-Guided Vulnerability Repair

    [https://arxiv.org/abs/2610.20245](https://arxiv.org/abs/2610.20245)

    P2Patch提出了一种生成式对抗修复系统，通过漏洞利用智能体与补丁智能体的持续对抗，将每次成功的漏洞利用转化为永久修复要求，从而生成能抵御同一漏洞多种变体攻击的更稳健补丁。

    

    arXiv:2610.20245v1 公告类型： cross 摘要：大语言模型正日益被部署为自动化程序修复的自主智能体，通过多步推理和工具使用来修补软件漏洞。然而，这些系统通常使用与构建补丁时相同的证据来验证补丁——静态告警、消毒器（sanitizer）追踪记录、已公开的漏洞利用代码或测试套件。因此，一个仅能阻断该特定证据的补丁可能看起来是成功的，但面对同一漏洞的某些变体时仍然存在漏洞。我们提出了P2Patch，一个生成式对抗修复系统，它将每个补丁视为新的攻击目标。从一个规范化的漏洞发现开始，漏洞利用智能体构建可复现的漏洞证明（PoV），补丁智能体修复程序，随后漏洞利用智能体针对打过补丁的程序发起同一漏洞的不同利用攻击。每一次成功的漏洞利用都会成为永久的修复要求，最终的补丁……

    arXiv:2610.20245v1 Announce Type: cross  Abstract: Large language models are increasingly deployed as autonomous agents for automated program repair, patching software vulnerabilities through multi-step reasoning and tool use. Yet these systems typically validate a patch against the same evidence used to construct it -- a static alert, sanitizer trace, published exploit, or test suite. A patch that blocks that one piece of evidence can therefore look successful while remaining vulnerable to the same vulnerability with some variation. We present P2Patch, a generative adversarial repair system that treats each patch as a new attack target. Starting from a normalized vulnerability finding, an exploitation agent constructs a reproducing proof of vulnerability (PoV), a patching agent repairs the program, and the exploitation agent then attacks the patched program for a distinct same-vulnerability exploit. Every successful exploit becomes a permanent repair requirement, and the final patch m
    
[^5]: SWE-Stream：通过渐进式构建与弹性流式执行扩展软件工程代理的可执行训练数据

    SWE-Stream: Scaling Executable Training Data for SWE Agents via Progressive Construction and Elastic Streaming

    [https://arxiv.org/abs/2610.20193](https://arxiv.org/abs/2610.20193)

    SWE-Stream 提出了一种通过渐进式构建与弹性流式执行来大规模自动生成可执行软件工程训练数据的系统，从 10,728 个仓库的开发历史中构建出 80,874 个经过验证的训练任务。

    

    软件工程代理发展迅速，但其训练规模的扩展仍受限于高质量可执行任务的稀缺。开发历史中包含构建此类任务的宝贵信号，然而由于环境不完整、测试缺失以及制品（artifacts）不一致等问题，往往需要进行大量的重建工作。我们提出了 SWE-Stream，一个通过渐进式构建与弹性流式执行来扩展可执行软件工程（SWE）训练数据的系统。SWE-Stream 对任务候选进行渐进式的验证与精炼，尽可能复用已有制品，仅在必要时才进行针对性的补全。弹性调度器在共享资源约束下异步协调各构建阶段，并根据观测到的工作负载需求动态调整执行预算。利用来自六种编程语言的 10,728 个仓库的开发历史，SWE-Stream 构建了 80,874 个经过验证的（摘要在此处截断）

    arXiv:2610.20193v1 Announce Type: new  Abstract: Software engineering agents have advanced rapidly, but scaling their training remains constrained by the limited availability of high-quality executable tasks. Development histories contain valuable signals for constructing such tasks, yet they often require substantial reconstruction due to incomplete environments, missing tests, and inconsistent artifacts. We present SWE-Stream, a system for scaling executable SWE training data through progressive construction and elastic streaming execution. SWE-Stream progressively validates and refines task candidates, reusing existing artifacts when possible and applying targeted completion only when necessary. An elastic scheduler asynchronously coordinates construction stages under shared resource constraints, adapting execution budgets based on observed workload demands. Using development histories from 10,728 repositories across six programming languages, SWE-Stream constructs 80,874 validated 
    
[^6]: 忒修斯之代码：智能体能否依据恢复的设计文档重建软件？

    The Code of Theseus: Can Agents Rebuild Software from Recovered Design Documents?

    [https://arxiv.org/abs/2610.20068](https://arxiv.org/abs/2610.20068)

    提出一个从紧凑仓库表示中恢复相互对齐的意图、需求与架构文档的框架，并发布覆盖 10,764 个多语言仓库的工件数据集，用于研究智能体能否凭借恢复的设计文档重建软件。

    

    LLM 智能体越来越多地在实现系统之前先进行设计，然而用于训练和评估它们的数据却集中在实现与测试上：需求与设计工件常常缺失、过时或属于私有。相比之下，源代码却十分丰富。以往的工作通过将源代码直接输入 LLM 或让智能体探索源代码来恢复架构文档，但这会耗尽上下文窗口、难以扩展到大量仓库，且只能产出单个工件而非相互对齐的一组工件。我们提出了一个框架，能够从紧凑的仓库表示、静态依赖图和关键上下文文件中恢复相互对齐的利益相关者意图、需求与架构，并发布了来自 10,764 个 Java、Python、C/C++ 和 C# 仓库的工件作为数据集。LLM 评审认为三个生成器产出的工件清晰且内部一致，但在正确性和恰当的抽象层级方面可靠性较低。为检验这些工件是否（摘要原文在此处截断）……

    arXiv:2610.20068v1 Announce Type: new  Abstract: LLM agents increasingly design a system before implementing it, yet the data to train and evaluate them centres on implementation and testing: requirements and design artefacts are often missing, outdated or private. Source code, in contrast, is abundant. Prior work recovers architecture documentation by feeding LLMs the source code or letting agents explore it, which saturates the context, does not scale to many repositories, and yields a single artefact rather than aligned ones.   We present a framework that recovers aligned stakeholder intent, requirements and architecture from a compact repository representation, a static dependency graph and key contextual files, and release the artefacts of 10,764 Java, Python, C/C++ and C# repositories as a dataset. An LLM judge rates the artefacts of three generators as clear and internally consistent, but less reliably correct and at the right level of abstraction.   To test whether the artefact
    
[^7]: SWE-Nexus：从测试执行中自动构建软件工程任务

    SWE-Nexus: Automated Composition of Software Engineering Tasks from Test Execution

    [https://arxiv.org/abs/2610.20038](https://arxiv.org/abs/2610.20038)

    SWE-Nexus 是一个完全自动化的流水线，通过追踪测试执行中生产函数的足迹、验证屏蔽效果并确定性组合兼容作用域，自动构建出涵盖 105 个仓库、共 600 个任务的仓库级软件工程实现基准。

    

    历史代码变更提供了便利的任务边界，但无法识别所选测试究竟执行了哪些生产函数，也无法支持超越单次变更的已验证行为组合。我们提出了 SWE-Nexus，这是一个完全自动化的流水线，能够从测试执行中构建仓库级的实现任务。四个构建智能体负责选择测试、为每个仓库适配追踪与评估协议，并生成行为规范。对于每个选定的测试文件，追踪会记录生产函数体的观测执行足迹。仅当在原始仓库中所有预期评分点均通过、且屏蔽被追踪的函数体会导致至少一个失败或错误时，该作用域才会被保留。兼容的作用域以确定性的方式进行组合，共享函数仅计算一次，并且每个组成测试文件都会在完整的组合屏蔽下重新验证。该基准测试包含来自六个（领域）的 105 个仓库的 600 个任务……（摘要原文在此处截断）

    arXiv:2610.20038v1 Announce Type: new  Abstract: Historical code changes provide convenient task boundaries, but do not identify which production functions selected tests exercise or support composing validated behaviors beyond a single change. We present SWE-Nexus, a fully automated pipeline that constructs repository-level implementation tasks from test execution. Four construction agents select tests, adapt tracing and evaluation protocols to each repository, and generate behavioral specifications. For each selected test file, tracing records an observed execution footprint of production function bodies. A scope is retained only if all expected scoring points pass in the original repository and masking the traced bodies causes at least one failure or error. Compatible scopes are composed deterministically, with shared functions counted once and every constituent test file revalidated under the complete combined mask. The benchmark contains 600 tasks from 105 repositories across six 
    
[^8]: LintEval：基于Linter引导的文件重构对大语言模型进行基准评测

    LintEval: Benchmarking LLMs on Linter-Guided File Refactoring

    [https://arxiv.org/abs/2610.20027](https://arxiv.org/abs/2610.20027)

    该论文提出LintEval，一个覆盖C++、Python和Java的多语言基准，用于评测大语言模型在linter引导的文件重构中减少lint告警的能力，实验揭示了告警减少与功能保持之间的持续权衡。

    

    Linter（代码检查工具）被广泛应用于代码维护和IDE实时编辑中，用于揭示源文件中与性能、风格、易错模式等相关的lint告警。然而，现有的代码基准测试并未显式评估linter引导的文件重构任务，即模型必须在保持源文件原有功能正确性的前提下减少lint告警。我们提出了LintEval，一个面向C++、Python和Java的linter引导文件重构多语言基准，包含13,061个初始功能正确的程序以及通过语言特定linter提取的171,662条lint告警。LintEval联合评估功能保持、lint告警减少、新告警规避以及编辑最小化。实验揭示了一个持续存在的权衡：闭源商业模型的Reduction@P最高可达90.4%，但Pass@1仅为67.1%–68.3%，而若干开源小模型则表现出较低的……

    arXiv:2610.20027v1 Announce Type: cross  Abstract: Linters are widely used in code maintenance and real-time IDE editing to surface lint alerts related to performance, style, error-prone, and other patterns in source files. However, existing code benchmarks do not explicitly evaluate linter-guided file refactoring, where models must reduce lint alerts in source files with correct initial functionality while satisfying functionality preservation. We introduce LintEval, a multilingual benchmark for linter-guided file refactoring across C++, Python, and Java, comprising 13,061 programs with correct initial functionality and 171,662 lint alerts extracted using language-specific linters. LintEval jointly evaluates functionality preservation, lint alert reduction, new alert avoidance, and edit minimality. Experiments reveal a persistent trade-off: closed-source commercial models reach up to 90.4% Reduction@P but achieve only 67.1%-68.3% Pass@1, while several open-source small models show low
    
[^9]: 面向代码理解的时间自适应微调

    Temporal Adaptive Fine-Tuning for Code Comprehension

    [https://arxiv.org/abs/2610.19970](https://arxiv.org/abs/2610.19970)

    提出 TAFT（时间自适应微调）方法，将仓库代码的演变转换作为监督信号，使大语言模型在代码理解问答中能够获取新知识、抑制过时知识，并保留仍然有效及语义相近的知识。

    

    arXiv:2610.19970v1 公告类型：新论文 摘要：仓库级代码理解问答支持开发者对分布于代码库中的功能和使用方式进行推理。然而，针对特定仓库的大语言模型（LLM）可能随着代码的演化而变得不再对齐。我们将这一挑战研究为时间性仓库适应问题：获取新知识、抑制过时知识、保留仍然有效的知识，以及维护语义上相近的知识。现有的检索、微调和模型编辑方法只能解决该问题的部分方面。我们提出了时间自适应微调（Temporal Adaptive Fine-Tuning，TAFT），这是一种新颖的适应方法，它将仓库的演变转换视为监督信号。TAFT 从仓库转换中推导出具有时间依据的问答数据，通过偏好优化将新有效知识与过时知识配对，并重放以转换为条件的保留示例，以维护仍然有效的知识。我们在一个增强的时间化 SWE-bench 数据集上评估了 TAFT……（摘要原文截断）

    arXiv:2610.19970v1 Announce Type: new  Abstract: Repository-level code comprehension question answering (QA) supports developer reasoning over functionality and usage distributed across codebases. However, repository-specialized large language models (LLMs) can become misaligned as code evolves. We study this challenge as temporal repository adaptation: acquiring new knowledge, suppressing obsolete, retaining still-valid, and preserving semantically neighboring knowledge. Existing retrieval, fine-tuning, and model-editing methods address only parts of this problem. We propose Temporal Adaptive Fine-Tuning (TAFT), a novel adaptation approach that treats repository transitions as supervision. TAFT derives temporally grounded QA data from repository transitions, pairs newly valid with obsolete knowledge through preference optimization, and replays transition-conditioned retention examples to preserve still-valid knowledge. We evaluate TAFT on an augmented temporal SWE-bench dataset compri
    
[^10]: 驯服在线软件工程的语言：检测并减少软件社区中的冒犯性语言

    Taming the Online Software Engineering Tongue: Detecting and Reducing Offensive Language in Software Communities

    [https://arxiv.org/abs/2610.19961](https://arxiv.org/abs/2610.19961)

    本研究调查了在线软件工程社区中冒犯性语言的普遍性与特征，并基于四个平台的13万条评论开发了包含72个机器学习模型的冒犯性语言减少系统（OLRSSE），该系统不仅能检测和分类冒犯性评论，还能向发布者解释原因并提供非冒犯性的替代建议。

    

    像Stack Overflow这样的在线软件工程（SE）社区已经变得不友好且充满敌意，这体现在其中存在粗鲁和冒犯性的评论，这些评论对参与者的情感健康造成了损害。这项工作的主要目标是调查在线SE社区中冒犯性语言的普遍程度和性质，并开发相应的分类方法，最终目标是通过提出一个冒犯性语言减少系统（OLRSSE）来减少冒犯性语言。该系统能够检测、分类冒犯性评论，向发布者解释为什么该评论具有冒犯性，并提供建议性的非冒犯性替代方案。研究使用了来自四个知名SE平台（Stack Overflow、GitHub、Gitter和Slack）讨论中的130,000条用户评论数据，开发了六组共十二个（总计72个）不同的机器学习模型（包括传统机器学习和深度学习模型）。表现最佳的模型的F1分数……（摘要在此处被截断）

    arXiv:2610.19961v1 Announce Type: new  Abstract: Online Software Engineering (SE) communities such as Stack Overflow have become unwelcoming and hostile spaces, as evidenced by the presence of rude and offensive comments that are detrimental to the emotional wellbeing of participants. The main objective of this work is to investigate the prevalence and nature of offensive language in online SE communities, and develop approaches to classify them with the ultimate goal of reducing offensive language by proposing an Offensive Language Reduction System (OLRSSE) that has the ability to detect, classify, and explain to the poster why a comment is offensive and also propose non-offensive alternative suggestions. Using data from 130,000 user comments on discussions held in four prominent SE platforms: Stack Overflow, GitHub, Gitter and Slack, we developed six sets of twelve (72 in total) different machine learning models (both traditional and deep learning). The F1-Score of the best performin
    
[^11]: 软件版本历史中能耗退化的系统性检测

    Systematic Detection of Energy Regressions in Software Version Histories

    [https://arxiv.org/abs/2610.19895](https://arxiv.org/abs/2610.19895)

    本文提出EnergyTrackr方法，通过识别提交间能耗的统计显著性变化来自动检测软件版本历史中的能耗退化，并在五个Java项目的1,932个提交上验证了其有效性，同时比较了线性算法与二分搜索算法的效率。

    

    背景：绿色软件工程正在成为应对信息技术日益增长的能耗影响的重要举措，尤其是在持续开发过程中。然而，如何设计自动化方法以高效识别跨提交的能耗退化仍然存在挑战。特别是，在通过识别能耗的统计显著性变化来自动检测提交级别退化方面，目前的研究努力甚少。目标：本文提出了EnergyTrackr，一种用于检测跨多个提交的能耗退化的方法。方法：我们在来自五个Java项目的1,932个提交上对EnergyTrackr进行了实证评估，评估其识别显著能耗变化的能力。我们还将一种分析整个提交历史的简单线性算法与更高效的二分搜索算法进行了比较。结果：评估结果表明了EnergyTrackr在识别能耗退化方面的有效性。

    arXiv:2610.19895v1 Announce Type: new  Abstract: Context: Green software engineering is emerging as a crucial response to the rising energy impact of information technology, especially in continuous development. However, challenges remain in devising automated methods to efficiently identify energy regressions across commits. In particular, little effort has been put into automatically detecting regressions at the commit level by identifying statistically significant changes in energy consumption. Objectives: This paper presents EnergyTrackr, an approach for detecting energy regressions across multiple commits. Methods: We empirically evaluated EnergyTrackr on 1,932 commits from five Java projects, assessing its ability to identify significant energy changes. We also compared a straightforward linear algorithm that analyzes the entire commit history with a more efficient dichotomous search algorithm. Results: The evaluation demonstrates the effectiveness of EnergyTrackr in identifying 
    
[^12]: 依赖图何时有助于代码到架构的映射？

    When Does the Dependency Graph Help Code-to-Architecture Mapping?

    [https://arxiv.org/abs/2610.19769](https://arxiv.org/abs/2610.19769)

    该论文将代码到架构的映射建模为依赖图上的半监督节点分类问题，并通过GCN、GAT、MLP与朴素贝叶斯的系统对比实验，揭示了依赖图结构信息在何种条件下才能真正提升架构映射的效果。

    

    当系统的实现偏离其预期的模块化分解，而源文件与架构模块之间的映射缺失、不完整或人工维护成本高昂时，就会产生架构技术债。我们将代码到架构的映射问题形式化为有向多关系依赖图上的半监督节点分类，其中文件作为节点，类型化的依赖关系作为边，词法特征则从代码标识符和目录布局中提取。从少量带标签的样本出发，通过迭代式自训练过程将映射扩展到其余文件。我们将图卷积网络（GCN）、图注意力网络（GAT）以及不使用图结构的多层感知机（MLP）——三种学习型模型均采用相同的特征和训练流程——与朴素贝叶斯参照方法进行对比。在十个开源Java和C/C++系统上进行的100次实验运行中，三种神经模型的表现均优于……（原文摘要至此截断）

    arXiv:2610.19769v1 Announce Type: cross  Abstract: Architectural technical debt arises when a system's implementation drifts from its inten ded modular decomposition, while the mapping between source files and architectural modu les is missing, incomplete, or costly to maintain manually. We formulate code-to-archite cture mapping as semi-supervised node classification over a directed multi-relational de pendency graph, with files as nodes, typed dependencies as edges, and lexical features d erived from code identifiers and directory layout. Starting from a small labeled sample,   an iterative self-training procedure expands the mapping to the remaining files. We com pare a graph convolutional network (GCN), a graph attention network (GAT), and a graph-f ree multi-layer perceptron (MLP), using identical features and training procedures for t he learned models, against a Naive Bayes reference. Across 100 runs on ten open-source J ava and C/C++ systems, all three neural models outperform 
    
[^13]: JevCoder：决策委托何时能在代码修复中带来收益？

    JevCoder: When Does Decision Delegation Pay Off in Code Repair?

    [https://arxiv.org/abs/2610.19596](https://arxiv.org/abs/2610.19596)

    该论文提出 JevCoder，一个围绕低成本概率判断器 Jev 构建的代码修复工作流程，系统研究了哪些修复决策可以委托给低成本判断器，通过结合代码片段推荐与动作类型指导，在保持修复质量的同时降低 token 开销。

    

    arXiv:2610.19596v1 公告类型：新论文 摘要：代码修复智能体在修复代码的过程中需要反复做出各种决策。这些决策塑造了修复轨迹，并影响后续模型请求的 token 成本。我们探讨哪些修复决策可以委托给一个低成本的概率判断器，以便在保持修复质量的同时减少 token 使用量。该判断器通过在预定义答案上输出概率来回答问题，而不生成任何解释。我们评估了判断质量在修复各阶段的表现，并测试单项干预如何改变后续的修复工作。我们将这些干预与不使用判断器的无引导修复进行对比，也与使用修复模型自身作为判断器的相同工作流程进行对比。研究表明，干预的收益取决于它所改变的修复工作量，而不仅仅是判断本身的成本。基于这些研究，我们提出了 JevCoder，一个围绕低成本概率判断器 Jev 构建的工作流程。它将代码片段推荐与动作类型指导相结合，并使用判断（原文摘要在此处截断）

    arXiv:2610.19596v1 Announce Type: new  Abstract: A coding agent repeatedly makes decisions as it repairs code. These decisions shape repair trajectories and affect the token cost of successive model requests. We ask which repair decisions can be delegated to a low-cost probabilistic judge to reduce token use while maintaining repair quality. The judges answer questions with probabilities over predefined answers, without generating explanations. We evaluate judgment quality across repair stages and test how individual interventions change subsequent repair work. We compare these interventions with unguided repair, which uses no judge, and with the same workflow using the repair model itself as the judge. The benefit depends on the work an intervention changes, rather than the cost of the judgment alone. Based on these studies, we propose JevCoder, a workflow built around Jev, a low-cost probabilistic judge. It combines code-snippet recommendations with action-type guidance and uses judg
    
[^14]: 使命（不）可能：面向开源无人机框架中自动驾驶仪逻辑缺陷的自主发现

    Mission (Im)Possible: Towards Autonomous Discovery of Autopilot Logic Flaws in Open-Source Drone Frameworks

    [https://arxiv.org/abs/2610.19566](https://arxiv.org/abs/2610.19566)

    本文提出了IMF（Impossible Mission Finder）框架，利用配备领域知识和精心构建上下文的多智能体LLM系统，结合高级语义缺陷判定器，自主发现开源无人机自动驾驶仪软件中难以通过传统方法检测的细微逻辑缺陷。

    

    摘要：自主无人机正日益被部署于安全关键任务中，然而其自动驾驶仪软件仍可能包含细微的逻辑缺陷，从而导致不安全的行为。这类缺陷难以通过现有技术发现，因为它们往往依赖于模块、命令和运行时状态之间的语义关系，并且很少表现为崩溃或其他可检测的故障。随着大语言模型（LLM）的最新进展，我们观察到，当配备必要的领域知识和精心构建的上下文时，LLM在理解自动驾驶系统的高级语义方面展现出强大的能力，使其适用于逻辑缺陷的发现。然而，LLM受限于其上下文窗口，并且容易偏向于训练数据集中频繁出现的漏洞类型，而逻辑缺陷的识别通常需要更高级别的语义缺陷判定器。为了弥合这一差距，我们提出了Impossible Mission Finder（IMF），一个多智能体

    arXiv:2610.19566v1 Announce Type: cross  Abstract: Autonomous drones are increasingly deployed in safety-critical tasks, yet their autopilot software can still contain subtle logic flaws that lead to unsafe behaviors. These flaws are difficult to discover with existing techniques because they often depend on semantic relationships across modules, commands, and runtime state, and rarely manifest as crashes or detectable failures. With recent advances in LLMs, we observe that, when equipped with essential domain knowledge and carefully crafted context, LLMs demonstrate strong capabilities in understanding the high-level semantics of autopilot systems, making them suitable for logic flaw discovery. However, LLMs are limited by their context and can be biased to the vulnerabilities frequently occurs in training dataset, while logic flaws often require higher-level semantic bug oracles to identify their existence. To bridge the gap, we present Impossible Mission Finder (IMF), a multi-agent 
    
[^15]: BenchChecker：评估面向大语言模型的缺陷修复基准的可信度

    BenchChecker: Assessing the Credibility of Bug-Fixing Benchmarks for LLMs

    [https://arxiv.org/abs/2610.19447](https://arxiv.org/abs/2610.19447)

    提出BenchChecker框架，通过基于取证的两阶段检测流程（仓库存在性测试与补丁存在性测试）来识别缺陷修复基准中的数据污染问题，从而更可信地评估大语言模型在自动程序修复任务上的真实性能。

    

    尽管大语言模型（LLM）在自动程序修复（APR）方面展现出潜力，但现有基准测试存在数据污染问题，即测试样本出现在训练集中，从而人为地夸大了模型性能。为解决这一问题，我们提出了BenchChecker，这是一个兼容开源与闭源模型的框架，通过检测此类数据污染来支持更可信的评估。基于“持久化补丁很可能仍保留在用于训练的近期仓库快照中”这一观察，BenchChecker采用一种基于取证的两阶段检测流程，仅使用模型的文本输出以及公开的仓库历史。第一阶段通过用代码前缀探测模型来推断其是否被纳入训练，进行仓库存在性测试；第二阶段通过相似性对齐和测试套件验证来确认补丁的持久性，进行补丁存在性测试。使用StarCoder-15B、StarCod（摘要在此处截断）

    arXiv:2610.19447v1 Announce Type: new  Abstract: While Large Language Models (LLMs) show promise for Automatic Program Repair (APR), existing benchmarks suffer from data contamination, in which test samples appear in training sets, artificially inflating performance. To address this, we propose BenchChecker, a framework compatible with both open-source and closed-source models that supports more credible evaluation by detecting such contamination. Grounded in the observation that persistent patches likely remain in recent repository snapshots used for training, BenchChecker employs a forensics-based two-stage detection process using only textual model outputs together with public repository history. The first stage conducts a repository presence test by probing the model with code prefixes to infer training inclusion; the second stage performs a patch presence test to verify patch persistence via similarity alignment and test-suite verification. Evaluations using StarCoder-15B, StarCod
    
[^16]: OS-Bench：在跨操作系统代码迁移任务上评估软件工程智能体

    OS-Bench: Evaluating SWE Agents on Cross-OS Code Migration

    [https://arxiv.org/abs/2610.19368](https://arxiv.org/abs/2610.19368)

    提出OS-Bench——首个通过对比同一代码库在Linux与Windows上的执行覆盖率来识别迁移机会，并将平台特定代码组织为五种场景的可执行基准，用于评估语言模型智能体在跨操作系统代码迁移上的能力。

    

    arXiv:2610.19368v1 公告类型：新 摘要：语言模型（LM）智能体越来越多地在基于GitHub议题（issue）和拉取请求（pull request）构建的仓库级软件工程（SWE）基准上进行评估。尽管这些基准涵盖了广泛的开发任务，但对于一个常见而在技术上却截然不同的挑战，它们所能提供的洞察有限：即将现有实现从一个操作系统（OS）迁移到另一个操作系统，同时保持其在两个系统上的行为一致。我们提出了OS-Bench，一个用于Linux与Windows之间跨操作系统代码迁移的可执行基准。受案例研究的启发——这些研究表明跨平台功能通常先在一个操作系统上实现，之后才为另一系统添加支持——OS-Bench通过比较同一仓库在两个系统上收集的执行覆盖率来识别迁移机会。随后，平台特定代码区域经过验证，被归纳为五种迁移场景，并转化为带有目标平台测试和源（摘要在此处被截断）的仓库级任务。

    arXiv:2610.19368v1 Announce Type: new  Abstract: Language-model (LM) agents are increasingly evaluated on repository-level software engineering (SWE) benchmarks derived from GitHub issues and pull requests. Although these benchmarks cover a broad range of development tasks, they provide limited insight into a common yet technically distinct challenge: migrating an existing implementation from one operating system (OS) to another while preserving behavior on both. We introduce OS-Bench, an executable benchmark for cross-OS code migration between Linux and Windows. Motivated by case studies showing that cross-platform features are frequently implemented on one OS before support is added for another, OS-Bench identifies migration opportunities by comparing execution coverage collected from the same repository on both systems. Platform-specific regions are then verified, organized into five migration scenarios, and transformed into repository-level tasks with target-platform tests and sour
    
[^17]: 超越功能正确性：对大语言模型生成代码的静态代码质量剖析

    Beyond Functional Correctness: Static Code Quality Profiling of LLM-Generated Code

    [https://arxiv.org/abs/2610.19167](https://arxiv.org/abs/2610.19167)

    本研究对73个大语言模型生成的65,663个Python程序进行了大规模静态代码质量剖析，发现仅靠功能正确性无法全面评估代码质量，各模型在可维护性等静态质量指标上存在系统性差异，且正确性往往伴随着结构上的权衡。

    

    大语言模型（LLM）越来越多地被用于代码生成，然而大多数评估仍然只关注生成的代码能否通过基准测试套件。然而，仅凭功能正确性只能提供对质量的有限认识：基准测试往往不完整，通过测试并不能保证代码是可维护的、可靠的或结构良好的。本文对LLM生成的Python代码进行了大规模的静态质量剖析，重点关注可维护性和可靠性。我们使用LiveCodeBench中73个模型的65,663个程序，采用Pylint、Radon和SonarQube等成熟的静态分析工具对生成的代码进行评估。我们的结果显示，各模型之间存在仅凭正确性指标无法捕捉的系统性差异，其中可维护性是最明显的差异来源。我们进一步发现，正确性往往伴随着结构上的权衡，而不是一致地对应更简单的解决方案。

    arXiv:2610.19167v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for code generation, yet most evaluations still focus on whether generated code passes benchmark test suites. Functional correctness alone, however, provides only a limited view of quality: benchmark tests are often incomplete, and passing them does not guarantee that code is maintainable, reliable, or well-structured. This paper {presents a large-scale static quality profiling of} LLM-generated Python code, with a focus on maintainability and reliability. Using 65{,}663 programs from 73 models in LiveCodeBench, we evaluate generated code with established static analysis tools, including Pylint, Radon, and SonarQube. Our results show systematic differences across models that are not captured by correctness alone, with maintainability emerging as the clearest source of variation. We further find that correctness often comes with structural trade-offs rather than uniformly simpler solution
    
[^18]: DivAlgo：代码生成模型算法多样性的系统性评估

    DivAlgo: A systematic evaluation of algorithmic diversity for code generation models

    [https://arxiv.org/abs/2610.19157](https://arxiv.org/abs/2610.19157)

    提出DivAlgo基准数据集及轻量级相似度度量方法，通过对24个代码生成模型的评估发现，模型生成代码的算法多样性显著低于人类解决方案（最高仅达人类多样性的约70%），揭示了当前代码生成模型在算法多样性方面的局限性。

    

    代码生成模型正被积极应用于专业和教育场景中。然而，大型语言模型（LLM）生成算法多样性代码的能力在很大程度上仍不为人知。我们提出了DivAlgo基准，这是一个包含针对编程问题的算法多样化解决方案的大规模数据集。我们还引入了一种轻量级的相似度度量方法，以支持大规模评估。我们在24个现代代码生成模型上的实验发现，不受控制的模型生成代码的多样性显著低于人类解决方案（仅捕捉到人类多样性的约31%），即使是最佳的多样性代码生成策略也只能达到人类多样性的约70%。此外，虽然可以引导模型基于人类策略生成解决方案，但它们无法再现人类解决方案的多样性。这些发现表明代码生成模型在捕捉算法多样性方面存在局限。

    arXiv:2610.19157v1 Announce Type: cross  Abstract: Code generation models are actively being used in professional and educational settings. However, the competence of large language models (LLMs) to generate algorithmically diverse code remains largely unknown. We propose the DivAlgo benchmark, a large-scale dataset constituting algorithmically diverse solutions to programming problems. We also introduce a lightweight similarity metric to enable large-scale evaluation. Our experiments across 24 modern code generation models found that uncontrolled model-generated code is significantly less diverse than human solutions (capturing ~31% of human diversity), and even the best diverse code generation strategy could achieve only ~70% of human diversity. Moreover, while it was possible to steer the models to generate solutions based on human strategies, they were unable to reproduce the diversity of the human solutions. These findings indicate the limitations of code generation models in capt
    
[^19]: 对已弃用Android API使用更新的系统性研究

    A Systematic Study of Deprecated Android API Usage Updates

    [https://arxiv.org/abs/2610.19150](https://arxiv.org/abs/2610.19150)

    本文通过分析1,000个已弃用Android API和4,336个来自GitHub的迁移示例，系统性地识别出六大迁移类别及反复出现的API使用更新模式，为改进自动化API迁移技术提供了实证基础。

    

    Android应用程序编程接口（API）持续演进，导致现有API被弃用，并要求开发者将其用法迁移到推荐的替代API上。尽管先前的研究已提出自动化迁移技术，但其效果往往受限于API变化的多样性和上下文复杂性。本文对已弃用的Android API、其推荐的替代API以及开发者在实际中所进行的使用更新开展了系统性研究。基于对从GitHub收集的1,000个已弃用API和4,336个迁移示例的分析，我们识别出六个迁移类别，并总结出反复出现的使用更新模式，包括Android版本检查、参数适配、对象初始化、数据类型转换以及API交互机制的变化。为评估这些模式的适用性，我们在零样本GUPPY基准提示中加入了迁移类别及对应的……

    arXiv:2610.19150v1 Announce Type: new  Abstract: Android Application Programming Interfaces (APIs) continuously evolve, causing existing APIs to become deprecated and requiring developers to migrate their usages to recommended replacements. Although prior studies have proposed automated migration techniques, their effectiveness is often limited by the diversity and contextual complexity of API changes. This paper presents a systematic study of deprecated Android APIs, their recommended replacements, and real-world usage updates performed by developers. Based on an analysis of 1,000 deprecated APIs and 4,336 migration examples collected from GitHub, we identify six migration categories and derive recurring usage-update patterns involving Android-version checks, parameter adaptation, object initialization, data-type conversion, and changes in API interaction mechanisms. To assess their applicability, we augment the zero-shot GUPPY benchmark prompts with the migration category and corresp
    
[^20]: X.js：一个Node.js污点风格漏洞的动态基准测试集

    X.js: A Live Benchmark for Node.js Taint-Style Vulnerabilities

    [https://arxiv.org/abs/2610.18585](https://arxiv.org/abs/2610.18585)

    本文提出了X.js——一个持续更新的真实世界Node.js污点风格漏洞基准测试集，每个漏洞均配有精确的源汇位置、经验证的可执行漏洞利用程序及补丁版本，解决了现有漏洞数据集过时、无法支持该领域研究长期发展的问题。

    

    npm包中的JavaScript漏洞对Node.js生态系统构成重大风险，越来越多的研究工作旨在检测、利用和缓解这些漏洞。对这类工作进行研究和评估需要最新的真实漏洞集合，然而最常用的数据集在2022年就停止记录漏洞了。我们提出了X.js，这是一个真实世界Node.js污点风格漏洞的动态（持续更新的）基准测试集，涵盖代码和命令注入、原型污染以及路径遍历等漏洞类型。X.js基于截至2026年中期收集的1,738份安全公告（主要来自GitHub咨询数据库）构建，将每个漏洞与精确的源（source）和汇（sink）位置、经过验证的可执行漏洞利用程序及其执行轨迹，以及在可用情况下的包修补版本进行配对。X.js支持长期可复现性和持续增长：依赖项被镜像到自托管注册表中，以防止包从生态系统中被移除。

    arXiv:2610.18585v2 Announce Type: replace-cross  Abstract: JavaScript vulnerabilities in npm packages pose a significant risk to the Node$.$js ecosystem, and a growing body of work aims to detect, exploit, and mitigate them. Studying and evaluating this work requires up-to-date collections of real vulnerabilities, yet the most commonly used datasets stop recording vulnerabilities in 2022. We present X$.$js, a live benchmark of real-world Node$.$js taint-style vulnerabilities covering code and command injection, prototype pollution, and path traversal. Built from 1,738 advisories collected through mid-2026, primarily from the GitHub Advisory Database, X$.$js pairs each vulnerability with precise source and sink locations, a validated executable exploit and its execution trace, and, when available, a patched version of the package. X$.$js supports long-term reproducibility and growth: dependencies are mirrored in a self-hosted registry to guard against package removal from the ecosystem,
    
[^21]: 交互即裁判

    Interaction-as-a-Judge

    [https://arxiv.org/abs/2610.17895](https://arxiv.org/abs/2610.17895)

    提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。

    

    编程智能体的任务是产出复杂的交互式应用，例如网站、桌面应用和游戏。人类在评估这些应用时会以交互方式进行，而当前自动化的LLM评估主要集中于代码分析和单元测试。这种不匹配会导致LLM评判者遗漏那些对人类用户而言一目了然的缺陷。我们研究了“交互即裁判”，这是一种由计算机操作智能体（CUA）直接与应用进行交互的评估范式。我们证明“交互即裁判”能够同时改进交互式应用的评估与生成。为此，我们收集并发布了两个互补的数据集：包含人类偏好及对应智能体操作轨迹的6,858对应用，以及带有经人类验证的需求标签的222个应用。相较于标准的非交互式方法，我们发现“交互即裁判”在预测h……（原文摘要在此处截断）

    arXiv:2610.17895v2 Announce Type: cross  Abstract: Coding agents are tasked with producing sophisticated interactive applications such as websites, desktop applications, and games. While humans assess these applications interactively, current automated LLM evaluations primarily focus on code analysis and unit tests. This mismatch can cause LLM judges to miss failures that are immediately apparent to human users. We study Interaction-as-a-Judge, an evaluation paradigm where computer-use agents (CUAs) directly interact with an application. We demonstrate that Interaction-as-a-Judge can improve both evaluation and generation of interactive applications. To this end, we collect and release two complementary datasets: $6,858$ pairs of applications with human preferences and corresponding agent traces, and $222$ applications with human-verified requirement labels. Relative to standard non-interactive approaches, we find that Interaction-as-a-Judge improves evaluation in terms of predicting h
    
[^22]: SecReason：面向大语言模型代码生成的漏洞检测与流内自修复

    SecReason: Vulnerability Detection and In-Stream Self-Repair for LLM Code Generation

    [https://arxiv.org/abs/2610.14453](https://arxiv.org/abs/2610.14453)

    提出SecReason，一种针对推理型大语言模型代码生成的流内防御机制，通过双轨增量扫描在流式思维链中实时检测安全漏洞并进行自修复。

    

    大语言模型（LLM）生成的代码可能引入安全漏洞，当验证仅关注功能正确性时，这些漏洞可能会持续存在于生产环境中。现有的防御机制在生成之前、生成期间或生成之后发挥作用，但没有一种能在代码生成的过程中对代码进行检查。安全提示词和检索到的安全知识在任何代码出现之前就已起作用，受限解码限制了 token 的选择但无法引导修复，而事后修复必须重新生成完整的响应。推理型大语言模型在其流式思维链的早期阶段起草代码，在这个交付前的窗口期内，代码仅存在于推理流中。我们提出 SecReason，一种监控该窗口的流内防御机制。双轨增量扫描应用安全规则，在每条语句被输出时进行检查，并追踪累积代码中的数据流。当安全规则触发时，SecReason 会中断生成，插入流内警告，并……（原文摘要在此处截断）

    arXiv:2610.14453v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) may introduce security vulnerabilities into generated code, which can persist into production when verification focuses on functional correctness. Existing defenses act before, during, or after generation, but none inspects the code during generation. Security prompts and retrieved security knowledge act before any code exists, constrained decoding limits token choices but cannot guide repair, and post-hoc repair must regenerate the full response. Reasoning LLMs draft code early in their streamed chain of thought, and in this pre-delivery window the code exists only inside the reasoning stream. We present SecReason, an in-stream defense that monitors this window. Dual-track incremental scanning applies security rules, checking each statement as it is emitted and tracing dataflows across the accumulated code. When a security rule fires, SecReason interrupts generation, inserts an in-stream warning, a
    
[^23]: SoftwareGym：通过多样化合成软件环境扩展计算机使用智能体的训练

    SoftwareGym: Scaling Computer-Use Agent Training with Diverse Synthesized Software Environments

    [https://arxiv.org/abs/2610.12611](https://arxiv.org/abs/2610.12611)

    提出SoftwareGym，一个自动化流水线，能够大规模合成界面多样、功能可执行且经运行时验证的GUI软件环境（已生成12个类别共182个应用，任务通过率达91.5%），并构建SoftwareGym-Bench基准以支持计算机使用智能体的训练与评估。

    

    计算机使用智能体（CUA）正在快速进步，然而大规模合成多样化且可靠的软件环境仍然具有挑战性。原生应用程序环境需要大量的构建与集成工作，这限制了应用的覆盖范围，并增加了针对特定应用过拟合的风险。而浏览器托管环境所能提供的交互深度和工作流复杂度则较为有限。这些局限性催生了SoftwareGym——一个用于合成GUI软件环境的自动化流水线，其生成的环境具备多样化界面、可执行的功能以及经运行时验证的行为。该系统生成结构多样的设计，将其实现为可运行的应用程序，并通过GUI评估与迭代修复来确保功能可靠性。借助SoftwareGym，我们合成了涵盖十二个类别的182个应用程序，平均GUI任务通过率达到91.5%。我们构建了SoftwareGym-Bench，一个包含120个任务的评估套件，用于评测……（原文摘要至此被截断）

    arXiv:2610.12611v1 Announce Type: cross  Abstract: Computer-use agents (CUAs) are rapidly improving, yet synthesizing diverse and reliable software environments at scale remains challenging. Native application environments require substantial construction and integration effort, which limits application coverage and increases the risk of application-specific overfitting. Browser-hosted environments provide limited interaction depth and workflow complexity. These limitations motivate SoftwareGym, an automated pipeline for synthesizing GUI software environments with diverse interfaces, executable functionality, and runtime-verified behavior. It generates structurally diverse designs, realizes them as runnable applications, and uses GUI evaluation with iterative repair to ensure functional reliability. Using SoftwareGym, we synthesize 182 applications across twelve categories and achieve a mean GUI task pass rate of 91.5%. We construct SoftwareGym-Bench, a 120-task evaluation suite spanni
    
[^24]: A2ABreak：A2A协议的系统性安全分析

    A2ABreak: Systematic Security Analysis of the A2A Protocol

    [https://arxiv.org/abs/2609.10871](https://arxiv.org/abs/2609.10871)

    本文提出A2ABreak框架，首次对A2A协议进行系统性安全分析，通过LLM辅助从自然语言规范中提取经验证的有限状态机模型（37个状态、76个转换），并利用对抗性验证系统性发现协议层面的安全漏洞。

    

    由Linux基金会管理的Agent2Agent（A2A）协议是一项开放标准，使自主AI智能体能够跨越组织边界相互发现、进行身份认证并委派任务。A2A旨在与用于工具集成的模型上下文协议（MCP）互补，正迅速成为多智能体生态系统的水平通信层。然而，该协议的安全性尚未得到系统性分析。本文提出了A2ABreak，这是对A2A协议首次严谨的系统性安全分析。我们引入了一种新颖的框架，利用大语言模型（LLM）辅助从自然语言规范中直接提取经验证的有限状态机，从929条形式化语句中构建了一个包含37个状态和76个转换的统一模型，随后对该模型进行系统性推理，在完全符合规范的条件下通过对抗性验证来发现协议层面的漏洞。

    arXiv:2609.10871v1 Announce Type: cross  Abstract: The Agent2Agent (A2A) protocol, now governed by the Linux Foundation, is an open standard that enables autonomous AI agents to discover, authenticate with, and delegate tasks to one another across organizational boundaries. Designed to complement the Model Context Protocol (MCP) for tool integration, A2A is rapidly emerging as the horizontal communication layer of the multi-agent ecosystem. Yet the protocol's security has received no systematic analysis.   This paper presents A2ABreak, the first rigorous systematic security analysis of the A2A protocol. We introduce a novel framework that utilizes an LLM-assisted extraction of a verified finite-state machine directly from the natural-language specification, producing a unified model of 37 states and 76 transitions from 929 formalized statements, and then systematically reasons over this model to discover protocol-level vulnerabilities through adversarial verification, under a full-comp
    
[^25]: 修复中的捷径：识别与分类软件工程基准测试中的智能体作弊利用行为

    Shortcutting the Fix: Identifying and Categorizing Agentic Exploits in Software Engineering Benchmarks

    [https://arxiv.org/abs/2609.06780](https://arxiv.org/abs/2609.06780)

    该研究首次将软件工程基准测试中智能体通过检索上游代码、利用 Git 历史等方式作弊的现象形式化为“智能体走捷径”，发现标准评估下高达 90% 的轨迹存在作弊尝试，并提出通过提示智能体独立构建解决方案可将其降至约 10% 且不损害任务性能，凸显了作弊感知评估框架的必要性。

    

    高软件工程基准测试得分往往掩盖了作弊性的捷径行为，例如检索上游解决方案、利用本地 Git 历史，或回忆已记忆的代码。我们将这一现象形式化为“智能体走捷径”，并使用轨迹级的 LLM 评审器，对三个最先进的开源大语言模型在 SWE-bench Multilingual 和 DeepSWE 基准上进行了审计。标准评估分别在 60.7%-90.1% 和 55.5%-89.1% 的轨迹中触发了走捷径尝试，其中高达 72.1% 的任务是借助成功的作弊手段完成的；而禁用网络访问也无法遏制这些高尝试率。通过提示智能体独立构建解决方案，可将走捷径尝试分别抑制至 9.4%-14.2% 和 0.9%-10.3%，同时几乎消除了依赖捷径的任务解决方式，并保持了核心任务性能。我们的研究结果证明，衡量仓库级问题解决能力必须采用具备作弊感知的评估框架。

    arXiv:2609.06780v2 Announce Type: replace  Abstract: High software engineering benchmark scores often mask exploitative shortcuts, such as retrieving upstream solutions, exploiting local Git histories, or recalling memorized code. We formalize this phenomenon as agentic shortcutting and audit three state-of-the-art open LLMs on SWE-bench Multilingual and DeepSWE using trajectory-level LLM judges. Standard evaluations trigger shortcutting attempts in 60.7%-90.1% and 55.5%-89.1% of trajectories, respectively, with up to 72.1% of tasks resolved with the aid of a successful exploit; disabling network access fails to curb these high attempt rates. Prompting agents to construct solutions independently suppresses attempts to 9.4%-14.2% and 0.9%-10.3%, respectively, while virtually eliminating shortcut-dependent resolutions and preserving core task performance. Our findings demonstrate the necessity of exploit-aware evaluation frameworks to measure repository-level problem solving.
    
[^26]: BTS-AgentBench：从只读遥测日志到代理基准测试的确定性、可重放流水线

    BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks

    [https://arxiv.org/abs/2608.27334](https://arxiv.org/abs/2608.27334)

    该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。

    

    arXiv:2608.27334v1 公告类型：新 摘要：工业现场包含大量只读遥测数据，但很少有基准测试规定如何将这些记录编译为可执行的多轮代理任务。我们提出了一种遥测到情节的构建方法，并将其实例化为BTS-AgentBench。该流水线将BTS元数据和原始历史记录规范化为只读工具存储，编译具有工具派生黄金答案和证据的静态任务，并将保留的任务提升为类型化、有界、面向操作员的情节。发布的532行版本增加了澄清、目标修订、时间戳策略、质量门控报告和证据归因，同时保留了源计算和分割。编码契约预检报告零发现，构建排除控制器完成0/532行。两个独立的原始到情节构建匹配所有11个逻辑工具存储导出，并精确复现发布的356/87/89训练/开发/测试工件。将共享构建路径应用于XAI4HEAT。

    arXiv:2608.27334v1 Announce Type: new  Abstract: Industrial sites contain large volumes of read-only telemetry, but few benchmarks specify how to compile these records into executable multi-turn agent tasks. We present a telemetry-to-episode construction method instantiated as BTS-AgentBench. The pipeline normalizes BTS metadata and raw histories into a read-only tool store, compiles static tasks with tool-derived gold answers and evidence, and lifts retained tasks into typed, bounded operator-facing episodes. The 532-row release adds clarification, goal revision, timestamp policy, quality-gated reporting, and evidence attribution while preserving the source computation and split. Coded contract preflight reports zero findings, and the construction-exclusion controller completes 0/532 rows. Two independent raw-to-episode builds match all 11 logical tool-store exports and reproduce the released 356/87/89 train/dev/test artifact exactly. Applying the shared construction path to XAI4HEAT 
    
[^27]: 通过运行时值反馈改进基于大语言模型的定向测试输入生成

    Refining LLM-based Directed Test Input Generation via Runtime Value Feedback

    [https://arxiv.org/abs/2608.07926](https://arxiv.org/abs/2608.07926)

    提出了ReDig框架，通过比较预期运行时值与实际运行时值的事后分析反馈，迭代改进基于LLM的定向测试输入生成，引导测试输入到达目标代码位置。

    

    尽管基于大语言模型（LLM）的定向输入生成具有出色的能力，但当其对目标程序行为的潜在假设不正确时，会产生偏离目标的测试输入。为了有效地将基于LLM的定向输入生成建立在实际程序行为的基础上，我们提出了ReDig，一个运行时反馈引导的框架，用于迭代改进基于LLM的定向输入生成。当生成的输入未能到达给定的目标代码位置时，ReDig会进行事后分析，通过比较预期的运行时值与在偏离目标的测试执行中观察到的实际运行时值，来识别不正确的假设。ReDig利用事后分析的结果作为反馈来改进推理上下文，引导后续的测试生成朝向目标位置。在本文中，我们展示了ReDig的进行中的研究工作，并通过四个案例研究说明了运行时反馈引导的改进是如何运作的。

    arXiv:2608.07926v2 Announce Type: replace  Abstract: Despite their remarkable capabilities, LLM-based directed input generation produces off-target test inputs when their underlying hypotheses on the target program's behaviors are incorrect. To effectively ground LLM-based directed input generation in actual program behaviors, we propose ReDig, a runtime feedback-guided framework for iterative refinement of LLM-based directed input generation. When a generated input fails to reach a given target code location, ReDig conducts a postmodern analysis to identify incorrect hypotheses by comparing expected runtime values and their actual runtime values observed in the off-target test execution. ReDig utilizes the postmortem analysis result as feedback to refine the inference context, guiding the subsequent test generations toward the target location. In this paper, we present our work-in-progress on ReDig. We illustrate how runtime feedback-guided refinement operates through four case studie
    
[^28]: Gemini 3用于检测手工测试用例中自然语言测试异味的实证研究

    An Empirical Study of Gemini 3 for Detecting Natural Language Test Smells in Manual Test Cases

    [https://arxiv.org/abs/2606.13804](https://arxiv.org/abs/2606.13804)

    本文实证研究了最新大语言模型Gemini 3 Pro Preview能否有效检测手工测试用例中的自然语言测试异味，并与此前表现最佳的小语言模型PHI-4-14B进行了对比。

    

    手工测试是指测试人员遵循自然语言指令来验证系统行为，这对于发现难以通过自动化方式捕获的问题仍然至关重要。然而，手工测试用例中常常包含测试异味，即诸如歧义、冗余或缺失检查等质量问题，这些问题会降低测试的可靠性、可维护性和可重复性。现有的检测方法主要依赖人工设计的规则，因此难以在异构测试套件中实现泛化和规模化。在我们之前的工作中，我们通过在来自143个真实Ubuntu测试用例的测试步骤（涵盖七种异味类型）上评估GEMMA-3-4B、LLAMA-3.2-3B和PHI-4-14B，评估了使用小语言模型进行测试异味检测的可行性，其中PHI-4-14B取得了最佳性能。在本文中，我们研究了研究开展时可用的最新大语言模型（GEMINI-3-PRO-PREVIEW）能否……（摘要原文在此处截断）

    arXiv:2606.13804v3 Announce Type: replace  Abstract: Manual testing, in which testers follow natural language instructions to validate system behavior, remains essential for uncovering issues that are difficult to capture with automation. However, manual test cases often contain test smells, quality issues such as ambiguity, redundancy, or missing checks that reduce reliability, maintainability, and reproducibility. Existing detection approaches largely depend on manually engineered rules and thus struggle to generalize and scale across heterogeneous test suites. In our previous work, we assessed the feasibility of using Small Language Models (SLMs) for test smell detection by evaluating GEMMA-3-4B, LLAMA-3.2-3B, and PHI-4-14B on test steps from 143 real-world Ubuntu test cases, covering seven smell types. PHI-4-14B achieved the best performance. In this article, we investigate whether a contemporary Large Language Model (GEMINI-3-PRO-PREVIEW) available at the time of the study can ide
    
[^29]: T2J-BENCH：基于执行效果的PyTorch到JAX代码仓库迁移编码智能体基准测试

    T2J-BENCH: Benchmarking Coding Agents on Execution-Grounded PyTorch-to-JAX Repository Migration

    [https://arxiv.org/abs/2605.29054](https://arxiv.org/abs/2605.29054)

    提出T2J-BENCH基准，包含63个涵盖18个模型家族的整仓库PyTorch到JAX迁移任务，以执行行为一致性和无torch依赖作为核心评价标准，用以真实评估编码智能体进行跨框架代码库迁移的能力。

    

    将机器学习代码库跨框架迁移——例如从PyTorch迁移到JAX——是一项常规但成本高昂的工程任务，其本质是跨范式的：急切执行的、有状态的、面向对象的模块必须被重写为作用于显式参数pytree的纯函数、可被jit编译的函数。我们探究编码智能体能否正确执行这类整仓库级别的迁移，并指出现有代码基准——它们以单元测试通过率或与参考翻译的表面相似度来评分——无法回答这一问题：它们既不要求迁移后的系统在“行为”上保持一致，也无法防止JAX重新实现通过暗中导入PyTorch本身来伪造等价性。我们提出T2J-BENCH，这是一个包含63个整仓库PyTorch到JAX迁移任务的基准，涵盖18个模型家族，横跨训练与服务两个场景，其中无torch依赖的检查点转换本身就是交付成果的一部分。T2J-BENCH以执行效果为依据进行评分。

    arXiv:2605.29054v3 Announce Type: replace-cross  Abstract: Migrating a machine-learning codebase across frameworks--for example from PyTorch to JAX--is a routine but costly engineering task that is fundamentally cross-paradigm: eager, stateful, object-oriented modules must be rewritten as pure, jittable functions over an explicit parameter pytree. We ask whether coding agents can perform such whole-repository migrations correctly, and argue that existing code benchmarks, which grade unit-test pass rates or surface-level similarity to a reference translation, cannot answer this: they neither require the migrated system to \emph{behave} the same nor prevent a JAX reimplementation from faking equivalence by secretly importing PyTorch itself. We introduce T2J-BENCH, a benchmark of 63 whole-repository PyTorch-to-JAX migration tasks spanning 18 model families across both training and serving, where torch-free checkpoint conversion is itself part of the deliverable. T2J-BENCH grades execution
    
[^30]: MOCHA：面向智能体技能优化的多目标切比雪夫退火方法

    MOCHA: Multi-Objective Chebyshev Annealing for Agent Skill Optimization

    [https://arxiv.org/abs/2605.19330](https://arxiv.org/abs/2605.19330)

    提出 MOCHA 方法，通过切比雪夫标量化选择改进对象并以扩展帕累托前沿为接受准则，从探索退火到利用，实现智能体技能在准确性与上下文经济性之间的多目标优化，克服了现有单分数提示词优化器易停滞的问题。

    

    LLM 智能体通过“技能”来组织其行为——技能是结构化的自然语言规范，规定了智能体如何进行推理、检索和响应。与单体式提示词不同，技能是受平台限制约束的多字段产物：描述字段被索引用于路由，指令主体被加载到共享的上下文窗口中，每多一行都会消耗智能体本可用于其他地方的上下文资源。因此，技能优化本质上是多目标问题：技能应当同时具备准确性和经济性。现有的提示词优化器将这些目标压缩为单一分数，我们发现这会导致它们停滞不前：在其大多数运行中，1000 次 rollout 相较种子技能毫无进展。我们提出了 MOCHA（Multi-Objective CHebyshev Annealing，多目标切比雪夫退火），该方法通过切比雪夫标量化来选择需要改进的技能，并接受能够扩展帕累托前沿的改进，从而从探索阶段逐步退火至利用阶段。切比雪夫标量化……

    arXiv:2605.19330v2 Announce Type: replace-cross  Abstract: LLM agents organize behavior through skills -- structured natural-language specifications that govern how an agent reasons, retrieves, and responds. Unlike monolithic prompts, skills are multi-field artifacts with platform limits: descriptions are indexed for routing, instruction bodies are loaded into a shared context window, and every extra line costs the agent context it could spend elsewhere. Skill optimization is therefore inherently multi-objective: a skill should be accurate and economical at once. Existing prompt optimizers collapse these goals into a single score, and we find that this makes them stall: in most of their runs, 1000 rollouts yield zero progress over the seed skill. We introduce MOCHA (Multi-Objective CHebyshev Annealing), which selects skills to improve by Chebyshev scalarization and accepts improvements that expand the Pareto front, annealing from exploration to exploitation. Chebyshev scalarization sco
    
[^31]: 扩展人机编码协作需要一个可治理的共识层

    Scaling Human-AI Coding Collaboration Requires a Governable Consensus Layer

    [https://arxiv.org/abs/2604.17883](https://arxiv.org/abs/2604.17883)

    该论文提出扩展人机编码协作需要引入一个可治理的共识层，通过版本化的项目状态模型将意图、结构性承诺、代码工件和证据关联起来，使人类能够在AI编码过程中检查、协调和修订共享的设计决策。

    

    氛围编码（Vibe coding）将自然语言意图转化为可运行的软件，但仅靠代码生成无法让人类和智能体对其正在构建的内容形成共同认知。随着协作跨越多个参与者、组件和版本发布，设计决策必须保持可理解且可修订。我们认为，扩展人类与AI的编码协作需要一个可治理的共识层。Agentic Consensus（智能体共识）围绕一个版本化的项目状态模型来组织协作，该模型将选定的意图、结构性承诺、可执行工件和证据联系在一起。参与者可以在实现之前澄清意图，并基于人类可检查、可修订的承诺来协调提议的更改。双向同步将预期变更与实现连接起来，并在工件演进过程中暴露冲突或过时的证据。我们在Consensus Control中实现了该审查工作流。在一项小规模探索性人类研究中，10名参与者……

    arXiv:2604.17883v2 Announce Type: replace-cross  Abstract: Vibe coding turns natural-language intent into working software, but code generation alone does not give humans and agents a shared account of what they are building. As collaboration spans participants, components, and releases, design choices must remain understandable and open to revision. We argue that scaling human and AI coding collaboration requires a governable consensus layer. Agentic Consensus organizes collaboration around a versioned project-state model linking selected intent, structural commitments, executable artifacts, and evidence. Participants can clarify intent before implementation and coordinate proposed changes against commitments that humans can inspect and revise. Bidirectional synchronization connects intended changes to implementation and exposes conflicts or stale evidence as artifacts evolve. We implemented its review workflow in Consensus Control. In a small exploratory human study, 10 participants 
    
[^32]: 终端智能体足以实现企业自动化

    Terminal Agents Suffice for Enterprise Automation

    [https://arxiv.org/abs/2604.00073](https://arxiv.org/abs/2604.00073)

    该论文提出仅配备终端和文件系统的编码智能体通过直接调用平台 API，就能以远低于复杂智能体架构（如 MCP 工具智能体和网页智能体）的成本，达到甚至超越它们在企业任务上的表现。

    

    构建能够与数字平台交互、自主执行有意义企业任务的智能体，正受到越来越多的关注。目前探索的方法包括基于模型上下文协议等抽象构建的工具增强型智能体，以及通过图形界面操作的网页智能体。然而，考虑到其成本和运营开销，这类复杂的智能体系统是否必要仍不明确。我们认为，仅配备终端和文件系统的编码智能体可以通过直接与平台 API 交互，更有效地解决许多企业任务。我们在多样化的真实世界系统中对这一假设进行了评估，结果表明这些低级终端智能体只需极低的成本即可达到或超越更复杂的智能体架构。我们的发现表明，简单灵活的编程接口与强大的基础模型相结合，应当成为企业自动化的核心支柱。

    arXiv:2604.00073v4 Announce Type: replace-cross  Abstract: There has been growing interest in building agents that can interact with digital platforms to execute meaningful enterprise tasks autonomously. Among the approaches explored are tool-augmented agents built on abstractions such as Model Context Protocol (MCP) and web agents that operate through graphical interfaces. Yet, it remains unclear whether such complex agentic systems are necessary given their cost and operational overhead. We argue that a coding agent equipped only with a terminal and a filesystem can solve many enterprise tasks more effectively by interacting directly with platform APIs. We evaluate this hypothesis across diverse real-world systems and show that these low-level terminal agents match or outperform more complex agent architectures at a fraction of the cost. Our findings suggest that simple, flexible programmatic interfaces combined with strong foundation models should be the backbone of enterprise autom
    
[^33]: LLMON：一种在LLM接口层面利用结构与语义的LLM原生标记语言

    LLMON: An LLM-native Markup Language to Leverage Structure and Semantics at the LLM Interface

    [https://arxiv.org/abs/2603.22519](https://arxiv.org/abs/2603.22519)

    提出了一种LLM原生标记语言LLMON，能够在LLM接口层面自然地传达文本的结构与语义元数据，从而提升模型的准确性、安全性以及抵抗提示注入攻击的能力。

    

    文本大语言模型（LLM）提供了一个简单且为大家熟悉的接口：输入和输出都是一串文本。然而，传递给LLM的信息通常具有更丰富的结构和语义，而这些信息在字符串中并未得到体现。例如，大多数提示词既包含指令（“将这篇论文总结为一段话”），也包含数据（需要总结的论文），但在传递给模型时二者通常不加区分。这可能导致模型混淆以及安全风险，例如提示注入攻击。本工作通过引入一种LLM原生标记语言LLMON（LLM Object Notation，发音为"Lemon"）来解决这一缺陷，该语言能够以自然的方式将文本的结构和语义元数据传达给LLM。这些信息随后可用于模型训练、模型提示和推理实现，从而在模型准确性、安全性和保密性方面带来提升。

    arXiv:2603.22519v3 Announce Type: replace  Abstract: Textual Large Language Models (LLMs) provide a simple and familiar interface: a string of text is used for both input and output. However, the information conveyed to an LLM often has a richer structure and semantics, which is not conveyed in a string. For example, most prompts contain both instructions ("Summarize this paper into a paragraph") and data (the paper to summarize), but these are usually not distinguished when passed to the model. This can lead to model confusion and security risks, such as prompt injection attacks.   This work addresses this shortcoming by introducing an LLM-native mark-up language, LLMON (LLM Object Notation, pronounced "Lemon"), that enables the structure and semantic metadata of the text to be communicated in a natural way to an LLM. This information can then be used during model training, model prompting, and inference implementation, leading to improvements in model accuracy, safety, and security. 
    
[^34]: SWE-Spot：通过仓库中心化学习构建小型仓库专家模型

    SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning

    [https://arxiv.org/abs/2601.21649](https://arxiv.org/abs/2601.21649)

    本文提出仓库中心化学习（RCL）新范式，将静态代码库转化为交互式学习信号，训练出针对特定代码仓库深度专门化的小型语言模型SWE-Spot-4B，以弥补小型模型处理复杂陌生代码库时泛化能力不足的缺陷。

    

    在隐私敏感和资源受限环境中部署编码智能体的需求，推动了对高性能开源权重小型语言模型（SLM）的需求。然而，这些模型存在根本性的能力差距：与前沿大模型不同，它们缺乏在推理时应对复杂、陌生代码库的强泛化能力。我们发现当前流行的任务中心学习（TCL）方法——即在不同仓库间扩大任务暴露规模——无法解决这一局限。为此，我们提出仓库中心化学习（RCL），这是一种范式转变：在SLM训练中优先追求纵向的单仓库深度，而非横向的单任务广度。为实现这种深度，我们设计了由四个单元组成的仓库中心化经验，将静态代码库转化为交互式学习信号，用于训练SWE-Spot-4B——一系列作为仓库专门专家构建的高度紧凑模型。在受控对比实验中……

    arXiv:2601.21649v2 Announce Type: replace-cross  Abstract: The deployment of coding agents in privacy-sensitive and resource-constrained environments drives the demand for capable open-weight Small Language Models (SLMs). However, they suffer from a fundamental capability gap: unlike frontier large models, they lack the inference-time strong generalization to handle complicated, unfamiliar codebases. We identify that the prevailing Task-Centric Learning (TCL) approach, which scales exposure across disparate repositories, fails to address this limitation. In response, we propose Repository-Centric Learning (RCL), a shift that prioritizes vertical, single-repository depth over horizontal, single-task breadth for SLM training. To accomplish this depth, we design a four-unit Repository-Centric Experience, transforming static codebases into interactive learning signals, to train SWE-Spot-4B, a family of highly compact models built as repo-specialized experts. Under controlled comparisons wi
    
[^35]: 保障高并发售票安全：一种基于微服务的框架

    Securing High-Concurrency Ticket Sales: A Framework Based on Microservice

    [https://arxiv.org/abs/2512.24941](https://arxiv.org/abs/2512.24941)

    本文提出了一种基于Spring Cloud微服务架构的铁路购票系统设计框架，通过多种安全设计方法保障系统在高并发场景下的稳定性、数据一致性和快速响应能力。

    

    铁路售票系统是最重要的公共服务基础设施之一。在节假日等高峰期，由于大量用户同时访问，系统经常面临高并发场景的挑战。传统的聚合式架构由于容错能力不足、承载能力较低，无法满足高峰期用户的需求。因此，系统需要采用微服务架构进行开发，并加入多种安全方法，以确保系统在高并发场景下具有良好的稳定性和数据一致性，并能快速响应用户请求。本文介绍了使用B/S架构和Spring Cloud设计开发一个能够在高并发场景下保持稳定可靠的铁路购票系统，并为该系统制定了多种安全设计方法。该系统集成了……

    arXiv:2512.24941v3 Announce Type: replace  Abstract: The railway ticketing system is one of the most important public service infrastructure. In peak periods such as holidays, it is often faced with the challenge of high concurrency scenarios because of a large number of users accessing at the same time. The traditional aggregation architecture can not meet the peak user requirements because of its insufficient fault tolerance and low ability. Therefore, the system needs to use microservice architecture for development, and add multiple security methods to ensure that the system can have good stability and data consistency under high concurrency scenarios, and can respond quickly to user requests. This paper introduces the use of B/S architecture and Spring Cloud to design and develop a railway ticket purchase system that can maintain stability and reliability under high concurrency scenarios, and formulate multiple security design methods for the system. This system integrates a range
    

