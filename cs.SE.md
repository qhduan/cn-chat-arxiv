# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FastCI: Efficient GPU-Intensive CI for LLM Training Frameworks](https://arxiv.org/abs/2610.01967) | FastCI通过利用运行时证据选择并剪枝测试、优先调度高风险测试以及优化测试工作负载，显著提升了大语言模型训练框架中GPU密集型持续集成的效率。 |
| [^2] | [Detecting Inconsistencies in Model Specifications with LLM-as-Verifier Reasoning](https://arxiv.org/abs/2610.01847) | VeriSpec是首个通过审计规范文本本身、以LLM作为验证器来直接检测模型规范内部不一致性的方法，无需将自然语言规范形式化或依赖行为测试。 |
| [^3] | [Continuous Process-Level Evaluation for Evolving Enterprise AI Agent Skills](https://arxiv.org/abs/2610.01833) | 该论文提出了一种结合结果级与过程级检查的持续评估框架，能够检测出最终输出评估所遗漏的过程级行为漂移——在通过全部最终数值检查的试验中，仍有92.6%存在其他评估器发现的偏差。 |
| [^4] | [CONTRA: Discovering and Qualifying Behavior-Changing Questions for Selective Clarification in LLM Code Generation](https://arxiv.org/abs/2610.01769) | CONTRA是一种无需训练的方法，通过广泛发现候选澄清问题，并结合语义评估与基于执行的验证来筛选出真正会改变代码行为的关键问题，从而让LLM代码生成智能体进行选择性澄清提问，在防止因假设错位导致行为偏差的同时避免不必要的打扰。 |
| [^5] | [Code Detectors Have a Half-Life: Obsolescence and Metric Illusions in LLM-Generated Code Detection](https://arxiv.org/abs/2610.01664) | 论文提出“检测器半衰期”概念，揭示代码检测器会随生成模型演进而过时，且准确率指标可能掩盖严重的预测偏差，而通用LLM评判器在检测LLM生成代码方面表现更为可靠。 |
| [^6] | [Architectural Degradation: How to Measure and to Remediate](https://arxiv.org/abs/2610.01611) | 该论文通过结合LLM辅助流程的多声部文献综述，系统梳理了284项研究中架构退化的度量、评估与修复方法，并构建了跨维度关联的分类体系。 |
| [^7] | [MCRI: A Four-Dimensional Framework for Analyzing and Evaluating Agent Skills](https://arxiv.org/abs/2610.01506) | 该论文提出四维MCRI框架及基于大语言模型的评估方法MCRI-Eval，实现了对智能体技能的系统性分析与评估，并在下游任务排序一致性和top-1技能选择上超越了现有最强基线。 |
| [^8] | [When Does a Second Model Help? Cross-Model Review in LLM Verification](https://arxiv.org/abs/2610.01471) | 在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。 |
| [^9] | [Refactoring React Component Hierarchies to Eliminate Prop Drilling](https://arxiv.org/abs/2610.01376) | 本文提出一种静态分析方法并实现为命令行工具 ReactRefactor，可自动识别 React 代码库中的属性钻取问题，并利用基于 Context API 和组件组合的两种自动化重构策略将其消除，实证评估表明属性钻取在各规模的开源项目中普遍存在。 |
| [^10] | [A Design Theory for AI-Assisted Software Development Derived from Christopher Alexander's Theory of Form](https://arxiv.org/abs/2610.01372) | 该论文基于克里斯托弗·亚历山大的形式理论，提出了一套AI辅助软件开发的设计理论，将LLM建模为“非本土的乡土建造者”，并通过显式表示问题框架与组织传统来推导和识别生成代码中的“不适配”之处。 |
| [^11] | [Trustworthy Data- and ML-Ops for Intelligent Transportation Systems and Logistics](https://arxiv.org/abs/2610.01282) | 本文对面向智能交通系统与物流领域的可信数据运维与机器学习运维进行了全面综述，系统阐述了其必要性、关键组成部分、可用工具与案例研究，并着重强调了AI可信性在该领域中的关键作用。 |
| [^12] | [CompProv Produces Machine Readable Graphs Encoding Microscopic Algebraic Provenance for Reproducible Computation](https://arxiv.org/abs/2610.01203) | CompProv 是一个基于 Java 的面向审计的溯源框架，通过高精度数值包装对象在单个代数运算的原子级捕获血缘关系，生成可序列化的计算溯源图谱（CPG），从而保障计算结果的可复现性与可审计性。 |
| [^13] | [Safety Must Survive Self-Improvement: Why Failures Persist and How Agents Recover](https://arxiv.org/abs/2610.01073) | 该研究通过受控实验发现，在递归自我改进过程中，当环境依赖发生变化时，历史分数会使不安全程序持续留存（48个历史中有22个未恢复），而刷新分数虽能恢复原有测试的正确性但仍有残余失败，揭示了智能体安全维护需要防止失败持续并建立恢复机制。 |
| [^14] | [Groundability, Not Scale Alone: When Weak Reviewers Can Audit Strong Coding Agents](https://arxiv.org/abs/2610.01023) | 该研究表明，只要为审查者提供恰当结构化并固定的证据格式（尤其是官方执行证据），即便名义上更弱的审查者也能可靠地审计强编码智能体生成的补丁，且审查者模型规模并非质量的稳定预测因素。 |
| [^15] | [ABSENTIA: Detecting Broken Access Control Vulnerabilities in Web Applications](https://arxiv.org/abs/2610.00977) | ABSENTIA是一个安全脚手架框架，通过引导LLM智能体构建路由与后端代码的映射图，并逐路由应用不变量证伪技术，系统性地检测Web应用中的失效访问控制漏洞。 |
| [^16] | [Finding the Right Fit: Model-Harness Interactions across Agent Tasks](https://arxiv.org/abs/2610.00917) | 该研究通过评估66种模型与执行框架的组合发现，模型排名会随框架和任务基准的变化而逆转，官方框架和更高成本都无法保证更优表现，因此构建智能体系统时必须针对具体任务寻找模型与框架的最佳搭配。 |
| [^17] | [ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization](https://arxiv.org/abs/2610.00906) | 提出ActiveSaddler，首次将智能体框架优化中的训练课程选择形式化为自动化课程学习问题，通过非平稳多臂老虎机建模将反复失败抽象为可复用的失败模式臂，并根据潜在学习进展自适应分配优化目标，实现训练课程与框架的协同演化。 |
| [^18] | [Understanding Issues, Causes and Solutions in Open-Source LLM-based Multi-Agent Systems](https://arxiv.org/abs/2610.00905) | 该论文通过对21个开源LLM多智能体系统项目中944个相关议题的实证分析，系统揭示了实践者在开发和部署此类系统时面临的主要问题、深层成因及潜在解决方案。 |
| [^19] | [Cross-Benchmark Transfer from RL on Agentic Coding Tasks](https://arxiv.org/abs/2610.00890) | 仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。 |
| [^20] | [FORALL-LEAN-AGENT for Auditable Reasoning in Formal Mathematics and Software Verification](https://arxiv.org/abs/2610.00885) | 该论文提出FORALL-LEAN-AGENT框架，通过隔离工作区、陈述比对、公理审计和独立证明检查等机制实现Lean形式化证明的可审计验证，在VeriSoftBench上将GPT-5.6 Sol的成功率从93提升至100并降低了成本。 |
| [^21] | [Correctness, Convergence, and AI-Generated Code Detection: A Longitudinal Study of Student and Large Language Model Code in Introductory Programming](https://arxiv.org/abs/2610.00863) | 该纵向研究通过分析2021至2025年间近三万份学生提交与九万条大语言模型生成的解，发现LLM生成的解通常正确且在实现上高度收敛，学生代码与生成参考解的匹配率随时间上升，但由于简单题目本身就只有少数自然解法，这种匹配能否作为检测学生使用AI写代码的有效依据仍存疑。 |
| [^22] | [ASAD: Adaptive Software Agents for Debugging](https://arxiv.org/abs/2610.00629) | ASAD 提出了一种自适应多智能体调试系统，根据 bug 的性质和复杂度动态配置智能体数量、专业角色及协作策略，克服了传统固定架构框架“一刀切”的局限性。 |
| [^23] | [Understanding and Mitigating Library-Related Issues in LLM-Generated Code](https://arxiv.org/abs/2610.00622) | 本文通过探索性研究揭示84%的LLM生成代码文件存在库相关错误，并据此提出一种代理式方法来理解和缓解这些库使用问题。 |
| [^24] | [Extending LLM-based support for software engineers with ADHD](https://arxiv.org/abs/2610.00555) | 本文提出了Tether 2.0，一个基于大语言模型的助手，通过结构化交互模式、活动感知上下文和持久记忆，在规划、编码、调试和评审的完整工作流中为患有ADHD的软件工程师提供针对性支持。 |
| [^25] | [Code That Works, Environments That Don't: Measuring Environment Reproducibility in AI-Generated Software](https://arxiv.org/abs/2610.00425) | 提出环境规约智能体协议及“声明式—运行时—必要且充分”三层依赖框架，系统评估揭示当前编码智能体虽能生成功能正确的代码，却普遍无法准确指定软件环境依赖。 |
| [^26] | [Rules to Tools: Executable Checks for LLM Agents in Scientific Computing](https://arxiv.org/abs/2610.00313) | 该论文提出Rules to Tools（R2T）方法，将书面形式的科学计算要求转化为可由大语言模型智能体直接调用的预先准备好的可执行检查，实验证明这种可执行检查能显著提升科学编码智能体修复程序的成功率（从26/30提升至29/30）。 |
| [^27] | [Testing and Verification of Quantum Compilers through Assurance Contracts and Evidence](https://arxiv.org/abs/2610.00255) | 该论文是一项关于量子编译器测试与验证的批判性综合综述，系统比较了经验证变换、等价性检查、差分与蜕变测试、基于属性的测试等多种方法，并通过失败案例和工作示例说明如何借助保证契约与证据确保量子编译器在其交付接口上的正确性。 |
| [^28] | [Localizing Post-Wire Semantic Changes in MCP Agent Frameworks](https://arxiv.org/abs/2610.00182) | 提出一种差分测试方法，通过追踪固定的工具结果经过 MCP 智能体框架的公共接口来定位语义变化，在四个 Python 集成的 18 个测试用例中发现了 13 处涉及结构化值、错误声明和丰富内容的分歧。 |
| [^29] | [Characterizing and Codifying Malware Sophistication](https://arxiv.org/abs/2610.00098) | 本文首次以质量视角系统化定义了恶意软件复杂度，通过重新解读ISO/IEC 25010软件质量标准并筛选出可通过静态二进制分析度量的相关特性，为源代码不可用时一致评估恶意软件复杂度的框架奠定了基础。 |
| [^30] | [Guarded Commits: Transactional Human Approvals for LLM Workflows](https://arxiv.org/abs/2610.00037) | 该论文提出“受保护提交”设计，将人工审批作为LLM工作流状态的组成部分，在不可逆外部操作执行前，通过账本决议记录与凭证受限的提交适配器，强制校验每条风险路径均已通过审批关口并核实证据、策略版本与执行路径。 |
| [^31] | [Software Project Management with LLM-Based Automation: Coordination, Validation, and Governance in Practice](https://arxiv.org/abs/2610.00027) | 本研究通过探索性案例研究发现，基于大语言模型的自动化并未引入新的正式管理实践，而是影响了软件项目经理在规划、估算、协调、监控和治理等活动中的工作方式与学习需求。 |
| [^32] | [From Verification Failures to Reusable Guidance for Coding Agents](https://arxiv.org/abs/2609.39022) | 该论文提出将专家对验证失败的诊断转化为编码智能体可复用的指导，结合K框架的可执行语言语义与一套用于构建规范、修复证明和审计充分性的工具包，在HumanEval上实现164/164的全通过率，并通过对照实验证明审计能识别出证明通过但存在缺陷的软件包。 |
| [^33] | [Zero2Repo: Can Coding Agents Build Repositories from Scratch?](https://arxiv.org/abs/2609.38269) | 提出 Zero2Repo 基准，通过语言无关的自动化流水线将真实的开源项目转化为“从零构建完整代码仓库”的任务，并以可执行验收测试和对抗性验证来严格评估编程智能体的从零构建能力。 |
| [^34] | [Neuro-Symbolic Indirect-Call Analysis under Opaque Pointers](https://arxiv.org/abs/2609.33547) | 提出Facet，这是首个在不透明指针的LLVM IR上重建间接调用分发关系的分析，通过识别调用加载函数指针的结构体字段，并独立恢复通过初始化器、存储和聚合拷贝赋给该字段的函数，再按字段标识将二者关联。 |
| [^35] | [What Does a Skill Actually Do? Estimands and Evaluation Validity for Tool and Skill Use in LLM Agents: A Critical Review](https://arxiv.org/abs/2609.33153) | 本文对大语言模型智能体中工具与技能使用的评估研究进行批判性综述，提出以处理对比、目标人群、结果、预算约束、汇总度量和识别假设六个维度刻画评估设计的分析框架，并证明常见的同任务配对运行设计在以触发为条件时无法识别技能调用的因果效应。 |
| [^36] | [The Next Challenge for Agentic Cybersecurity: A Realistic, Contamination-Free Reverse Engineering Benchmark](https://arxiv.org/abs/2608.11469) | 本文提出了SRE-Bench，这是首个现实且无污染的逆向工程基准，由专家从零构建，确保实例在训练数据中不可见，以真实评估AI智能体的逆向工程能力。 |
| [^37] | [Evaluating Neural Decompilation of Dart AOT Binaries: Fine-Tuning, Metric Validity, Specification Leakage, and Reliability](https://arxiv.org/abs/2607.06125) | 该论文对Dart AOT二进制文件的神经反编译进行了基于执行的系统性评估，揭示了微调适配器存在功能性回归、传统静态指标与真实功能正确性关联较弱，以及模型严重依赖语义名称等规范泄漏线索，从而质疑了当前神经反编译评估方法的可靠性。 |
| [^38] | [Source-Free Detection and Impact Analysis of Compiler Optimization Problems in Mobile Applications](https://arxiv.org/abs/2606.23512) | 提出了无需源码的OptDetect框架，可直接从应用二进制文件中检测原生库的编译器优化问题，大规模分析发现30.5%的原生库使用低优化级别，影响了91.7%的主流移动应用。 |
| [^39] | [SWE-chat: Coding Agent Interactions From Real Users in the Wild](https://arxiv.org/abs/2604.20779) | SWE-chat 是首个从开源开发者真实场景中持续收集的大规模编程智能体会话数据集，揭示出使用模式呈双峰分布——41% 的会话中智能体编写几乎全部代码（“氛围编程”），25% 则完全由人工编写，且智能体在自然环境中效率依然低下。 |
| [^40] | [Emergence-as-Code as a Foundation for Reliable Self-Governance](https://arxiv.org/abs/2602.05458) | 该论文提出“涌现即代码”（EmaC）框架，将局部适应对系统可靠性的系统级影响纳入可执行、证据一致的复合SLO评估，通过持续追踪旅程义务与版本化运行时假设，为系统的可靠自主治理奠定基础，并在PetClinic实验中实现了100%准确的变化检测且无误报。 |
| [^41] | [Bridging the Sim-to-Real Gap with multipanda_ros2: A Real-Time ROS2 Framework for Multimanual Systems](https://arxiv.org/abs/2602.02269) | 该论文提出了开源 ROS2 框架 multipanda_ros2，通过单进程控制多台 Franka 机器人、维持 1kHz 实时控制频率、实现不超过 2 毫秒的控制器切换延迟，并集成带定量评估指标的高保真 MuJoCo 仿真来弥合仿真到现实的差距。 |
| [^42] | [HarnessAgent: Scaling Automatic Fuzzing Harness Construction with Tool-Augmented LLM Pipelines](https://arxiv.org/abs/2512.03420) | HarnessAgent是一个工具增强的智能体框架，通过动态获取规范、依赖和使用示例等丰富上下文信息，克服了现有方法上下文不足及LLM钻验证指标空子的问题，实现了在大型多样化项目上完全自动化、可扩展的功能性模糊测试驱动程序构建。 |
| [^43] | [Efficient Solvers for SLOPE in R, Python, Julia, and C++](https://arxiv.org/abs/2511.02430) | 该论文提出了 R、Python、Julia 和 C++ 中高效求解 SLOPE 问题的软件包套件，采用高效的混合坐标下降算法支持多种损失函数和数据结构，并在速度上超越了现有的 SLOPE 实现。 |
| [^44] | [Incentives and Outcomes in Bug Bounties](https://arxiv.org/abs/2509.16655) | 该研究利用谷歌漏洞奖励计划2024年7月奖励金额最高上调200%这一自然实验，实证发现提高赏金显著增加了高价值漏洞的提交数量，漏洞研究人员的劳动供给弹性很高，且奖励提升既重新激发了资深研究人员也吸引了新人参与。 |
| [^45] | [On the Illusion of Success: An Empirical Study of Job Reruns and Silent Failures in Industrial CI](https://arxiv.org/abs/2509.14347) | 本文首次通过重跑成功作业的实践，对工业持续集成中的“静默失败”（即作业被标记为成功却未完成全部或部分任务）进行了实证研究，分析了81个工业项目中的142,387个作业，揭示了这种制造成功假象并可能让缺陷逃逸至生产环境的失败现象。 |
| [^46] | [Are AI Coders Snitches? An Empirical Study of Pretraining Data Detection on Code Large Language Models](https://arxiv.org/abs/2507.17389) | 该论文对七种最先进的训练数据检测方法在源代码数据上的有效性进行了全面的实证研究，评估了它们在八个代码大语言模型上的表现，填补了训练数据检测方法在代码领域研究的空白。 |

# 详细

[^1]: FastCI：面向大语言模型训练框架的高效GPU密集型持续集成

    FastCI: Efficient GPU-Intensive CI for LLM Training Frameworks

    [https://arxiv.org/abs/2610.01967](https://arxiv.org/abs/2610.01967)

    FastCI通过利用运行时证据选择并剪枝测试、优先调度高风险测试以及优化测试工作负载，显著提升了大语言模型训练框架中GPU密集型持续集成的效率。

    

    随着大语言模型（LLM）在规模和复杂性上的不断增长，其训练框架也在快速演进。因此，持续集成（CI）对于维护这些框架的质量和稳定性至关重要。然而，与传统软件不同，LLM训练框架的持续集成依赖于GPU密集型测试，这些测试通常涉及完整的模型训练或评估，这使得持续集成本身成为快速开发过程中的新瓶颈。本文提出了FastCI，一个提升LLM训练框架持续集成效率的框架。FastCI利用运行时证据来选择受影响的测试，并剪枝那些在等价上下文中执行已变更代码的测试。随后，FastCI优先调度高风险测试以更早地暴露潜在故障，并在每个测试预期验证范围之外的维度上优化测试工作负载。该框架在我们的LLM训练框架的持续集成工作负载上进行了评估。

    arXiv:2610.01967v1 Announce Type: cross  Abstract: As large language models (LLMs) keep growing in size and complexity, their training frameworks evolve at a rapid pace as well. Therefore, continuous integration (CI) is critical for maintaining the quality and stability of these frameworks. However, unlike traditional software, CI for LLM training frameworks relies on GPU-intensive tests, which usually involve complete model training or evaluation. This leads CI itself to become a new bottleneck for fast-paced development. In this paper, we introduce FastCI, a framework that improves the efficiency of CI for LLM training frameworks. FastCI leverages runtime evidence to select affected tests and prune tests that execute changed code in equivalent contexts. Then FastCI prioritizes high-risk tests to expose potential failures earlier, and optimizes test workloads along dimensions outside the intended validation scope of each test. Evaluated on the CI workload of our LLM training framework
    
[^2]: 基于LLM作为验证器推理的模型规范不一致性检测

    Detecting Inconsistencies in Model Specifications with LLM-as-Verifier Reasoning

    [https://arxiv.org/abs/2610.01847](https://arxiv.org/abs/2610.01847)

    VeriSpec是首个通过审计规范文本本身、以LLM作为验证器来直接检测模型规范内部不一致性的方法，无需将自然语言规范形式化或依赖行为测试。

    

    模型规范定义了大语言模型（LLM）应当如何表现，指导着对齐训练、推理时的行为以及评估。然而，这些规范本身可能存在缺陷：两个各自合理的原则在应用于同一情形时可能规定互不兼容的行为，导致没有任何回复能够同时满足两者。检测这类不一致性极具挑战性：将自然语言规范形式化可能会丢失细微的语义区别，而基于行为的测试又无法可靠地区分规范缺陷与模型行为差异。我们提出了VeriSpec，这是首个通过审计规范文本本身来直接检测模型规范中不一致性的方法。我们的关键洞察在于，在保留规范自然语言形式的同时，使用LLM作为验证器。VeriSpec提取结构化的、情境感知的规则，并构建主题引导的图来聚类行为上相关……

    arXiv:2610.01847v1 Announce Type: cross  Abstract: Model specifications define how large language models (LLMs) should behave, guiding alignment training, inference-time behavior, and evaluation. Yet these specifications may themselves contain defects: two individually reasonable principles may prescribe incompatible behavior when applied to the same situation, leaving no response that satisfies both. Detecting such inconsistencies is challenging. Formalizing natural-language specifications risks losing subtle distinctions, while behavior-based testing cannot reliably distinguish specification defects from differences in model behavior. We introduce VeriSpec, the first approach to directly detect inconsistencies in model specifications by auditing the specification text itself. Our key insight is to preserve the specification in natural language while using an LLM as a verifier. VeriSpec extracts structured, context-aware rules, constructs a topic-guided graph to cluster behaviorally r
    
[^3]: 面向持续演化的企业AI智能体技能的过程级持续评估

    Continuous Process-Level Evaluation for Evolving Enterprise AI Agent Skills

    [https://arxiv.org/abs/2610.01833](https://arxiv.org/abs/2610.01833)

    该论文提出了一种结合结果级与过程级检查的持续评估框架，能够检测出最终输出评估所遗漏的过程级行为漂移——在通过全部最终数值检查的试验中，仍有92.6%存在其他评估器发现的偏差。

    

    随着工具API、模型和规范的不断变更，企业AI智能体技能也在持续演化，然而仅评估最终输出可能会遗漏过程层面的行为漂移。我们提出了一个结合结果级与过程级检查的持续评估框架，并将其应用于企业价值感知弹性系统中商业价值判定技能的收入与生产力两个变体。该框架独立计算每次运行的真值，生成可复用的模板测试，并通过程序化检查和范围受限的LLM评判器来评估工具选择、参数、执行顺序以及数据库完整性。我们在两种技能、两种规范变体、两种智能体运行框架和三种模型上共评估了240次试验。在通过所有适用的最终数值检查的175次试验中，有162次（92.6%；Wilson 95%置信区间：87.7-95.6%）被其他评估器检测到存在偏差。在更广泛的七项检查的最终状态定义下，15

    arXiv:2610.01833v1 Announce Type: new  Abstract: Enterprise AI agent skills evolve as tool APIs, models, and specifications change, yet final-output evaluation can miss process-level behavioral drift. We present a continuous evaluation framework combining outcome-level and process-level checks, applied to Revenue and Productivity variants of a Business Value Determination skill in an enterprise Value Aware Resiliency system. The framework independently computes per-run ground truth, materializes reusable template tests, and evaluates tool selection, arguments, execution order, and database integrity through programmatic checks and a narrowly scoped LLM judge. We evaluate 240 trials across two skills, two specification variants, two agent harnesses, and three models. Of 175 trials passing all applicable final numerical checks, 162 (92.6 percent; Wilson 95 percent CI: 87.7-95.6 percent) contained another evaluator-detected deviation. Under a broader seven-check final-state definition, 15
    
[^4]: CONTRA：发现并评估可改变程序行为的问题，用于大语言模型代码生成中的选择性澄清

    CONTRA: Discovering and Qualifying Behavior-Changing Questions for Selective Clarification in LLM Code Generation

    [https://arxiv.org/abs/2610.01769](https://arxiv.org/abs/2610.01769)

    CONTRA是一种无需训练的方法，通过广泛发现候选澄清问题，并结合语义评估与基于执行的验证来筛选出真正会改变代码行为的关键问题，从而让LLM代码生成智能体进行选择性澄清提问，在防止因假设错位导致行为偏差的同时避免不必要的打扰。

    

    编码智能体可能生成看似正确、但实际实现了用户从未预期的行为的代码。当智能体通过自身假设默默地去解决欠明确的需求时，就会产生这种不匹配。随着后续开发建立在这些假设之上，纠正由此产生的行为的成本会越来越高。尽早进行澄清提问有助于防止此类不匹配，但不必要的问题会打断开发者并拖慢开发进度。现有方法难以在识别关键澄清问题的同时避免提出不必要的问题。因此，我们提出了CONTRA，这是一种无需训练的方法，它将广泛的问题发现与基于语义和基于执行的问题评估相结合。CONTRA首先生成候选问题，并过滤掉与所需行为无关、或已被需求内容所解决的问题。对于剩下的每个问题，它会基于两个合理的答案分别生成程序并进行检查……

    arXiv:2610.01769v1 Announce Type: new  Abstract: Coding agents can generate code that appears correct but implements behavior the user never intended. This mismatch can arise when an agent silently resolves underspecified requirements through its own assumptions. As subsequent development builds on these assumptions, correcting the resulting behavior can become increasingly costly. Early clarification can help prevent such mismatches, but unnecessary questions can interrupt developers and slow down development. Existing methods struggle to identify key clarification questions while avoiding unnecessary ones. Therefore, we propose CONTRA, a training-free method that combines broad question discovery with semantic and execution-based question qualification. CONTRA first generates candidate questions and filters out those unrelated to required behavior or already resolved by the requirement. For each remaining question, it generates programs conditioned on two plausible answers and checks
    
[^5]: 代码检测器存在半衰期：LLM生成代码检测中的过时性与指标假象

    Code Detectors Have a Half-Life: Obsolescence and Metric Illusions in LLM-Generated Code Detection

    [https://arxiv.org/abs/2610.01664](https://arxiv.org/abs/2610.01664)

    论文提出“检测器半衰期”概念，揭示代码检测器会随生成模型演进而过时，且准确率指标可能掩盖严重的预测偏差，而通用LLM评判器在检测LLM生成代码方面表现更为可靠。

    

    随着代码生成模型的不断演进，代码检测器可能会过时：在某一代模型上验证有效的检测器可能无法迁移到下一代模型。我们将这种有限的有效期称为“检测器半衰期”。我们在C++、Java和Python三种语言上，评估了八个通用LLM评判器和三个专用检测器在人类编写代码与七个生成器所产生代码上的表现。结果揭示了两个问题：首先，不同生成器和提示策略下的性能差异显著，表明某些检测器依赖于特定生成器的模式，而非代码来源的一般性证据；其次，准确率可能掩盖严重的预测偏差——DetectCodeGPT和GPT-Sniffer在所有生成器上达到了0.50的准确率，但F1分数仅为0.00，因为它们几乎将所有样本都判定为AI生成。相比之下，通用LLM评判器取得了更强的准确率和F1分数。我们的结果表明，通用LLM在该任务上具有前景。

    arXiv:2610.01664v1 Announce Type: new  Abstract: Code detectors can become obsolete as code-generating models evolve: a detector validated on one generation of models may not transfer to the next. We call this limited useful life a detector half-life. We evaluate eight general-purpose LLM judges and three dedicated detectors on human-written code and code produced by seven generators across C++, Java, and Python. Our results reveal two problems. First, performance varies considerably across generators and prompting strategies, suggesting that some detectors rely on generator-specific patterns rather than general evidence of code provenance. Second, accuracy can conceal severe prediction bias. DetectCodeGPT and GPT-Sniffer achieved an accuracy of 0.50 but an F1 score of 0.00 across all generators because they classified almost every sample as AI-generated. However, general-purpose LLM judges achieved stronger accuracy and F1 scores. Our results show that general-purpose LLMs are promisi
    
[^6]: 架构退化：如何度量与修复

    Architectural Degradation: How to Measure and to Remediate

    [https://arxiv.org/abs/2610.01611](https://arxiv.org/abs/2610.01611)

    该论文通过结合LLM辅助流程的多声部文献综述，系统梳理了284项研究中架构退化的度量、评估与修复方法，并构建了跨维度关联的分类体系。

    

    背景。架构退化会损害软件的可维护性、可演化性和质量。然而，现有研究在度量方法、指标、工具和修复策略方面仍较为零散，限制了我们对这些要素在退化生命周期中如何相互关联的理解。目标。我们通过考察研究者如何度量架构退化、哪些指标和工具支持其评估，以及现有方法如何进行修复，来整合架构退化领域的研究现状。方法。我们对284项同行评审研究和灰色文献研究进行了多声部文献综述。我们采用本地运行的LLM辅助流程，结合检索增强生成、多模型验证和人工裁定，来支持文献筛选、数据提取和分类。随后我们分析了所得的分类体系及其跨维度关系。结果与结论。我们识别出277种度量方法（摘要在此处被截断）

    arXiv:2610.01611v1 Announce Type: new  Abstract: Context. Architectural degradation undermines software maintainability, evolvability, and quality. However, existing research remains fragmented across measurement approaches, metrics, tools, and remediation strategies, limiting our understanding of how these elements relate across the degradation lifecycle. Aim. We consolidate the state of the art on architectural degradation by examining how researchers measure it, which metrics and tools support its assessment, and how existing approaches address remediation. Method. We conducted a Multivocal Literature Review of 284 peer-reviewed and grey-literature studies. We supported screening, data extraction, and classification with a locally executed LLM-assisted pipeline combining Retrieval-Augmented Generation, multi-model validation, and human adjudication. We then analyzed the resulting taxonomies and their cross-dimensional relationships. Results and Conclusions. We identified 277 measure
    
[^7]: MCRI：一个用于分析和评估智能体技能的四维框架

    MCRI: A Four-Dimensional Framework for Analyzing and Evaluating Agent Skills

    [https://arxiv.org/abs/2610.01506](https://arxiv.org/abs/2610.01506)

    该论文提出四维MCRI框架及基于大语言模型的评估方法MCRI-Eval，实现了对智能体技能的系统性分析与评估，并在下游任务排序一致性和top-1技能选择上超越了现有最强基线。

    

    随着智能体从单一工具系统演进为模块化的复合架构，技能正成为能力开发与分发的重要机制。然而，学术界目前缺乏一个用于系统性分析和评估技能的结构化框架。基于信息增益和行为约束，我们提出了四维MCRI框架，并将其操作化为MCRI-Eval——一种基于大语言模型的评估方法。我们使用来自OpenClaw技能中心的63,812个公开技能对MCRI-Eval进行评估，在BigCodeBench、BFCL-Fundamental和Mind2Web三个基准上共进行了58,275次技能条件下的模型执行。结果表明，MCRI-Eval的分数与社区流行度信号呈正相关，并且在所评估的方法中实现了最高的下游任务排序一致性。MCRI-Eval还在所有三个基准测试中提升了top-1技能选择的效果：与每个基准上最强的基线相比，所选技能……

    arXiv:2610.01506v1 Announce Type: new  Abstract: As agents evolve from single-tool systems into modular, composite architectures, skills are becoming an important mechanism for capability development and distribution. However, the academic community lacks a structured framework for systematically analyzing and evaluating skills. Drawing on information gain and behavioral constraint, we propose the four-dimensional MCRI Framework and operationalize it as MCRI-Eval, a large language model-based evaluation method. We evaluate MCRI-Eval using 63,812 public skills from the OpenClaw skill Hub, with 58,275 skill-conditioned model executions across BigCodeBench, BFCL-Fundamental, and Mind2Web. MCRI-Eval scores are positively associated with community popularity signals and achieve the highest downstream ranking agreement among the evaluated methods. MCRI-Eval also improves top-1 skill selection across all three benchmarks: compared with the strongest baseline on each benchmark, the skills sele
    
[^8]: 第二个模型何时有帮助？大语言模型验证中的跨模型审查

    When Does a Second Model Help? Cross-Model Review in LLM Verification

    [https://arxiv.org/abs/2610.01471](https://arxiv.org/abs/2610.01471)

    在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。

    

    大语言模型如今能够生成代码、文档和分析内容，并且越来越多地被用于审查这类输出。本文探究的问题是：由不同的模型进行第二次审查在何时才有帮助？在作者早期预印本研究（在单一模型内改变上下文、重复次数和角色结构）的基础上，我们通过一项受控实验来检验模型独立性：实验包含30个工件（其中埋设150个错误）、10种审查条件，以及由来自两个开发者的三个审查模型执行的900次审查会话。在该实验中，(1) 顶级跨模型审查者在F1分数上与同模型在全新会话中的审查（CCR）无显著差异，但这并不等同于二者等价；(2) 两者发现的错误部分不同（Jaccard相似度为41.2%）；(3) 在两次审查调用的设定下，一次CCR加一次跨模型审查所匹配到的埋设错误多于两次CCR审查（56.7% vs. 42.7%；经Holm校正的p=.006），但并不显著多于两次顶级跨模型审查。

    arXiv:2610.01471v1 Announce Type: cross  Abstract: Large language models now generate code, documentation, and analyses, and are increasingly used to review such output. We ask when a second review by a different model helps. Building on the author's earlier preprints, which varied context, repetition, and role structure within one model, we test model independence in a controlled experiment: 30 artifacts with 150 planted errors, 10 review conditions, and 900 review sessions with three reviewer models from two developers. In this experiment, (1) a top-tier cross-model reviewer is not significantly different in F1 from same-model review in a fresh session (CCR), which does not establish equivalence; (2) the two find partly different errors (Jaccard 41.2%); and (3) at two review calls, one CCR plus one cross-model review matches more planted errors than two CCR reviews (56.7% vs. 42.7%; Holm-adjusted p=.006), but not significantly more than two reviews by the top-tier cross-model reviewe
    
[^9]: 重构 React 组件层次结构以消除属性钻取

    Refactoring React Component Hierarchies to Eliminate Prop Drilling

    [https://arxiv.org/abs/2610.01376](https://arxiv.org/abs/2610.01376)

    本文提出一种静态分析方法并实现为命令行工具 ReactRefactor，可自动识别 React 代码库中的属性钻取问题，并利用基于 Context API 和组件组合的两种自动化重构策略将其消除，实证评估表明属性钻取在各规模的开源项目中普遍存在。

    

    在 React 前端开发中，属性钻取是指在组件层次结构的多个层级之间通过组件属性逐层传递数据的做法。尽管 React 官方文档不推荐这种做法，且近期研究将其归类为一种代码坏味道，但人们对其普遍性、复杂性以及自动化重构的潜力仍知之甚少。在本工作中，我们提出了一种静态分析方法，用于识别 React 代码库中的属性钻取实例，并采用基于 Context API 和组件组合的两种自动化重构策略来消除它们。所提出的方法被实现为一个名为 ReactRefactor 的 Node.js 命令行工具，并在一个开源 React 应用程序基准数据集上进行了实证评估。实证评估的主要结果表明：(a) 属性钻取在各基准项目中频繁出现，且与项目规模无关；(b) 其复杂度普遍处于低到中等水平

    arXiv:2610.01376v1 Announce Type: new  Abstract: In React front-end development, Prop Drilling is the practice of propagating data through component properties across multiple levels of the component hierarchy. Despite being discouraged by React documentation and characterized as a code smell by recent research, little is known about its prevalence, complexity, and potential for automated refactoring. In this work, we propose a static analysis method for identifying Prop Drilling instances in a React codebase and eliminating them using two automated refactoring strategies based on Context API and Component Composition. The proposed method is implemented as a Node.js command line tool, ReactRefactor, and empirically evaluated on a benchmark dataset of open-source React applications. The main findings of the empirical evaluation indicate (a) the frequent occurrence of prop drillings across benchmark projects, irrespective of project size; (b) their generally low to moderate complexity in
    
[^10]: 源自克里斯托弗·亚历山大形式理论的AI辅助软件开发设计理论

    A Design Theory for AI-Assisted Software Development Derived from Christopher Alexander's Theory of Form

    [https://arxiv.org/abs/2610.01372](https://arxiv.org/abs/2610.01372)

    该论文基于克里斯托弗·亚历山大的形式理论，提出了一套AI辅助软件开发的设计理论，将LLM建模为“非本土的乡土建造者”，并通过显式表示问题框架与组织传统来推导和识别生成代码中的“不适配”之处。

    

    大语言模型（LLM）生成的代码不能被假定满足指定的需求。评审、测试和静态分析仍然适用，但一个充分的保障体系（harness）需要其中的哪些手段、以及它们各自扮演什么角色，仍是开放问题。我们提出一个源自克里斯托弗·亚历山大形式理论的设计理论，以及应用该理论的方法论。在亚历山大的论述中，形式与其情境之间的契合只能以否定的方式被感知，即通过已识别出的“不适配”（misfits）的缺失。我们将组织的传统显性化，并从中以及从问题的分类中推导出不适配之处。该理论将LLM建模为一个“非本土的乡土建造者”，它在众多代码库上接受训练却不属于任何一个，其输出倾向于漂移向主流惯例而非本地传统。我们构建了四个机制：问题的显式表示（Jackson的问题框架）与传统的显式表示（一种四形式模式语言）；确定性……（摘要在此处被截断）

    arXiv:2610.01372v1 Announce Type: new  Abstract: Code generated by large language models (LLMs) cannot be assumed to meet specified requirements. Reviews, testing, and static analysis still apply, but which of them a sufficient harness needs, and in what role, is open. We propose a design theory derived from Christopher Alexander's theory of form, and a methodology for applying it. In Alexander's account, fit between a form and its context can be perceived only negatively, through the absence of identified misfits. We make the organization's tradition explicit and derive the misfits from it and from the problem's classification. The theory models the LLM as a non-native vernacular builder, trained on many codebases but native to none, whose output tends to drift toward mainstream conventions rather than the local tradition. We engineer four pieces of machinery: explicit representations of the problem (Jackson's problem frames) and of the tradition (a four-form pattern language); determ
    
[^11]: 面向智能交通系统与物流的可信数据运维与机器学习运维

    Trustworthy Data- and ML-Ops for Intelligent Transportation Systems and Logistics

    [https://arxiv.org/abs/2610.01282](https://arxiv.org/abs/2610.01282)

    本文对面向智能交通系统与物流领域的可信数据运维与机器学习运维进行了全面综述，系统阐述了其必要性、关键组成部分、可用工具与案例研究，并着重强调了AI可信性在该领域中的关键作用。

    

    arXiv:2610.01282v1 公告类型：新论文 摘要：智能交通系统与物流（ITS&L）的快速发展已成为现代社会经济的基石，这在很大程度上依赖于数据、人工智能（AI），特别是机器学习（ML）的融合集成。本文对智能交通系统与物流领域的可信数据运维与机器学习运维进行了全面综述，强调了它们在提升交通与物流服务的效率、可靠性和决策精度方面的重要性。我们首先识别了现有文献中的研究空白，为本文的贡献提供了清晰的背景。随后，我们深入探讨了DataOps和MLOps的复杂性，讨论了其必要性、关键组成部分、可用工具、实践经验以及与ITS&L相关的案例研究。此外，我们还探讨了AI应用中可信性这一关键问题，审视了旨在增强（摘要在此处被截断）……的相关方法和工具。

    arXiv:2610.01282v1 Announce Type: new  Abstract: The rapid evolution of Intelligent Transportation Systems and Logistics (ITS\&L) has become a cornerstone of the modern social economy, relying heavily on the integration of Data, Artificial Intelligence (AI), and, more specifically, Machine Learning (ML). This paper provides a comprehensive review of Trustworthy Data and Machine Learning Operations (DataOps and MLOps) in the ITS\&L domain, underscoring their importance in improving efficiency, reliability, and decision-making precision within transportation and logistics services. We begin by identifying gaps in current literature, offering clear context for our contribution. Subsequently, we explore the complexities of DataOps and MLOps, discussing their necessity, key components, available tools, practical insights, and case studies relevant to ITS\&L. Additionally, we address the critical issue of Trustworthiness in AI applications, examining methods and tools designed to strengthen 
    
[^12]: CompProv 生成机器可读图谱以编码微观代数溯源，实现可复现计算

    CompProv Produces Machine Readable Graphs Encoding Microscopic Algebraic Provenance for Reproducible Computation

    [https://arxiv.org/abs/2610.01203](https://arxiv.org/abs/2610.01203)

    CompProv 是一个基于 Java 的面向审计的溯源框架，通过高精度数值包装对象在单个代数运算的原子级捕获血缘关系，生成可序列化的计算溯源图谱（CPG），从而保障计算结果的可复现性与可审计性。

    

    科学研究与金融建模中计算结果的可靠性日益依赖于可验证的可追溯性，而不仅仅是对所报告输出的信任。现有的溯源系统以文件、数据集或流水线阶段为粒度运行，未能记录连接算法输入与输出的内部代数变换序列；一个舍入误差、一次未记录的替换或一个缺失的中间值都可能传播到最终结果且无可恢复的痕迹。本工作提出 CompProv，这是一个基于 Java、面向审计的溯源框架，它通过将数值封装在高精度包装对象中，在单个代数运算的原子级层面捕获数据血缘，生成可序列化的计算溯源图谱（Calculation Provenance Graph, CPG），该图谱作为自包含的持久化制品保存，而非被丢弃的副产品。该框架通过三个异构案例进行评估。

    arXiv:2610.01203v1 Announce Type: new  Abstract: The reliability of computational results in scientific research and financial modeling increasingly depends on verifiable traceability, not merely on trust in a reported output. Existing provenance systems operate at the granularity of files, datasets, or pipeline stages, leaving the internal sequence of algebraic transformations connecting an algorithm's inputs to its outputs unrecorded; a rounding error, an undocumented substitution, or a missing intermediate value can propagate to a final result with no recoverable trace. This work presents CompProv, a Java-based, audit-oriented provenance framework that captures lineage at the atomic level of individual algebraic operations by encapsulating numerical values in high-precision wrapper objects, producing a serializable Calculation Provenance Graph (CPG) that persists as a self-contained artifact rather than a discarded byproduct. The framework is evaluated through three heterogeneous ca
    
[^13]: 安全必须在自我改进中存续：失败为何持续存在以及智能体如何恢复

    Safety Must Survive Self-Improvement: Why Failures Persist and How Agents Recover

    [https://arxiv.org/abs/2610.01073](https://arxiv.org/abs/2610.01073)

    该研究通过受控实验发现，在递归自我改进过程中，当环境依赖发生变化时，历史分数会使不安全程序持续留存（48个历史中有22个未恢复），而刷新分数虽能恢复原有测试的正确性但仍有残余失败，揭示了智能体安全维护需要防止失败持续并建立恢复机制。

    

    递归自我改进（RSI）允许智能体将有用的变更跨代传递。在这些代际中维护安全性既涉及防止不安全行为的持续存在，也涉及在失败发生时实现恢复。我们通过一个由有状态授权任务构成的受控测试平台来研究这些挑战，其中固定的LLM编辑器优化可执行的智能体组件，并由独立的轨迹记录其效果。成对干预实验将“哪些修订通过验证”、“哪些程序继续运行”以及“编辑器接下来修订哪个程序”区分开来。当一个新的授权依赖使之前测试过的优化失效后，尽管每个受影响的存档中都存在正确的替代方案，历史分数仍在48个框架历史中的22个里保留了相同的不安全程序。刷新分数恢复了原始测试套件上的正确性，但在独立构造的测试中仍存在残余失败。在未发生变化的条件下失败同样会持续存在（摘要在此处被截断）。

    arXiv:2610.01073v1 Announce Type: new  Abstract: Recursive self-improvement (RSI) allows agents to carry useful changes across generations. Maintaining safety across these generations involves both preventing unsafe behavior from persisting and enabling recovery when failures occur. We study these challenges through a controlled testbed of stateful authorization tasks, where fixed LLM editors optimize executable agent components and independent traces record their effects. Paired interventions separate which revisions pass validation, which program continues running, and which program the editor revises next. After a new authorization dependency invalidates previously tested optimizations, historical scores preserve the same unsafe programs in 22 of 48 framework histories despite a correct alternative in every affected archive. Refreshing scores restores correctness on the original suite, with residual failures on independently composed tests. Failures also persist under an unchanged c
    
[^14]: 有据可依，而非仅凭规模：弱审查者何时能审计强编码智能体

    Groundability, Not Scale Alone: When Weak Reviewers Can Audit Strong Coding Agents

    [https://arxiv.org/abs/2610.01023](https://arxiv.org/abs/2610.01023)

    该研究表明，只要为审查者提供恰当结构化并固定的证据格式（尤其是官方执行证据），即便名义上更弱的审查者也能可靠地审计强编码智能体生成的补丁，且审查者模型规模并非质量的稳定预测因素。

    

    编码智能体可能返回看似合理却遗漏了所需行为的补丁。这类失败难以审查，因为冗长的执行轨迹和充满自信的总结往往会掩盖被遗漏之处。我们研究的问题是：名义上更弱的审查者在何时能够可靠地判断一个补丁是否真正解决了对应的问题。我们分析了来自三个智能体的411条带执行结果标注的轨迹，以及101个受控案例。在154条GPT-5.4轨迹上，结构化但未经核实的信息会同时提高缺陷捕获率与过度拒绝率。随后，我们引入官方执行证据作为上限诊断基准。在为每位审查者从两种格式中选定并固定一种之后，六位审查者中有五位在122条留出轨迹上同时改善了这两项指标，其中两位对所有轨迹均做出了正确分类。审查者的规模并非质量的稳定预测因素。鉴于部署环境中无法获得官方测试，我们还评估了一种固定的级联方案，该方案利用补丁导致的静态错误，以及在未打补丁的代码仓库上首次运行即失败的生成测试。在121个评分……（原文摘要在此处截断）

    arXiv:2610.01023v1 Announce Type: cross  Abstract: Coding agents can return plausible patches that omit required behavior. These failures are hard to review because long traces and confident summaries often hide what was missed. We ask when a nominally weaker reviewer can reliably decide whether a patch solves its issue. We study 411 execution-labeled traces from three agents and 101 controlled cases. On 154 GPT-5.4 traces, structured but unchecked evidence raises both defect catch and over-rejection. We then provide official execution evidence as an upper-bound diagnostic. After choosing and freezing one of two formats per reviewer, five of six reviewers improve both rates on 122 held-out traces; two classify every trace correctly. Reviewer size is not a consistent predictor of quality. Because official tests are unavailable in deployment, we also evaluate a frozen cascade with patch-caused static errors and generated tests that first fail on the unpatched repository. On 121 scored he
    
[^15]: ABSENTIA：检测Web应用程序中的失效访问控制漏洞

    ABSENTIA: Detecting Broken Access Control Vulnerabilities in Web Applications

    [https://arxiv.org/abs/2610.00977](https://arxiv.org/abs/2610.00977)

    ABSENTIA是一个安全脚手架框架，通过引导LLM智能体构建路由与后端代码的映射图，并逐路由应用不变量证伪技术，系统性地检测Web应用中的失效访问控制漏洞。

    

    失效访问控制，即授权机制的失效，是最普遍的Web安全风险之一。与注入不同（注入是不可信输入流向危险操作的数据流），授权是一种关系：谁可以对什么执行操作，而非数据如何流动。每个应用程序都自行决定这种关系，因此预先编写的规则无法适用于下一个应用。LLM智能体可以从代码中推断出这种关系，但由于缺乏系统性地覆盖应用程序并确定检查优先级的方法，其搜索仍然是无方向的，导致访问控制缺陷无法被检测到。我们提出了ABSENTIA，这是一个安全脚手架，它将通用LLM智能体转变为针对Web应用程序后端的系统性漏洞检测器，由维护代码的开发人员和安全工程师作为审计来运行。在其指导下，智能体构建一个图，将应用程序的路由映射到其背后的代码。然后ABSENTIA逐个路由地开展工作，应用不变量证伪方法：……

    arXiv:2610.00977v1 Announce Type: cross  Abstract: Broken access control, the failure of authorization, is one of the most prevalent web security risks. Unlike injection, a flow of untrusted input into a dangerous operation, authorization is a relation: who may act on what, not how data moves. Each application decides that relation for itself, so no rule written in advance carries to the next. An LLM agent can infer it from the code, but with no systematic way to cover the application and prioritize what to inspect, its search stays undirected and access-control flaws go undetected.   We present ABSENTIA, a security scaffolding that turns general LLM agents into systematic vulnerability detectors for the backend of web applications, run as an audit by the developers and security engineers who maintain the code. Under its direction, the agents build a graph that maps the application's routes to the code behind them. ABSENTIA then works route by route, applying invariant falsification: i
    
[^16]: 寻找合适的搭配：智能体任务中的模型与执行框架交互

    Finding the Right Fit: Model-Harness Interactions across Agent Tasks

    [https://arxiv.org/abs/2610.00917](https://arxiv.org/abs/2610.00917)

    该研究通过评估66种模型与执行框架的组合发现，模型排名会随框架和任务基准的变化而逆转，官方框架和更高成本都无法保证更优表现，因此构建智能体系统时必须针对具体任务寻找模型与框架的最佳搭配。

    

    选择一个智能体系统意味着同时选择一个语言模型和它赖以行动的执行框架。我们探究一个强大的模型、框架或组合在环境改变时是否依然强大。我们评估了66种配置：四个可配置框架（OpenHands、DeepSeek Harness、PI 和 openJiuwen）与五个模型在 TUA-Bench、ALE-CLI 和 Terminal-Bench 4 上的组合，外加原生的 Codex-GPT 和 Claude Code-Claude 搭配。结果显示，模型排名会在不同框架之间逆转。在 Terminal-Bench 4 上，Claude 在 OpenHands 中领先 GPT 7.94 分，但在 PI 中却落后 GPT 30.16 分。对于五个模型中的四个，最佳框架会随基准测试而改变，但也有一些稳定的组合：openJiuwen 在所有三个基准上都使 Kimi 获得最高分数，领先幅度为 5.61 至 11.11 分。模型自家的官方框架并不可靠地是其最佳选择，更高的成本也不一定能换来更高的分数。在 Terminal-Bench 4 上，GPT 在 PI 下的得分高于在 DSH 下，同时成本更低。

    arXiv:2610.00917v1 Announce Type: new  Abstract: Choosing an agent system means choosing both a language model and the harness through which it acts. We ask whether a strong model, harness, or pairing stays strong when the setting changes. We evaluate 66 configurations: four configurable harnesses (OpenHands, DeepSeek Harness, PI, and openJiuwen) paired with five models on TUA-Bench, ALE-CLI, and Terminal-Bench 4, plus the native Codex-GPT and Claude Code-Claude pairings. Model rankings reverse across harnesses. On Terminal-Bench 4, Claude leads GPT by 7.94 points in OpenHands but trails it by 30.16 points in PI. For four of the five models, the best harness changes from one benchmark to another, yet some pairings hold: openJiuwen gives Kimi its highest score on all three benchmarks, by 5.61 to 11.11 points. A model's own vendor harness is not reliably its best, and higher cost does not reliably buy a higher score. On Terminal-Bench 4, GPT scores higher under PI than under DSH at less 
    
[^17]: ActiveSaddler：面向智能体框架优化的自动化课程学习

    ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization

    [https://arxiv.org/abs/2610.00906](https://arxiv.org/abs/2610.00906)

    提出ActiveSaddler，首次将智能体框架优化中的训练课程选择形式化为自动化课程学习问题，通过非平稳多臂老虎机建模将反复失败抽象为可复用的失败模式臂，并根据潜在学习进展自适应分配优化目标，实现训练课程与框架的协同演化。

    

    自动化框架优化能够通过根据执行反馈迭代更新大语言模型智能体的提示词、工具接口和控制逻辑，从而显著提升其表现。然而，现有方法主要优化框架如何被更新，却在很大程度上固定了生成驱动这些更新的反馈的训练场景。随着框架不断演进，对进一步优化最有用的场景可能会发生变化，这表明训练课程本身应当与框架一同自适应调整。我们将框架优化中这一缺失的维度形式化为一个自动化课程学习问题，并提出了ActiveSaddler。ActiveSaddler将不断演化的课程建模为具有动态实例化优化目标的非平稳多臂老虎机问题。它将反复出现的失败抽象为可复用的失败模式臂，估计继续针对每种模式进行优化所能带来的潜在学习进展，并自适应地平衡对已知弱点的重访与……

    arXiv:2610.00906v1 Announce Type: new  Abstract: Automated harness optimization can substantially improve LLM agents by iteratively updating their prompts, tool interfaces, and control logic from execution feedback. However, existing methods primarily optimize how the harness is updated while largely fixing which training scenarios generate the feedback that drives those updates. As the harness evolves, the scenarios most useful for further optimization can change, suggesting that the training curriculum itself should adapt alongside the harness. We formulate this missing dimension of harness optimization as an automated curriculum learning problem and introduce ActiveSaddler. ActiveSaddler models the evolving curriculum as a non-stationary bandit with dynamically instantiated optimization targets. It abstracts recurring failures into reusable failure-pattern arms, estimates the potential learning progress from further targeting each pattern, and adaptively balances revisiting known we
    
[^18]: 理解基于大语言模型的开源多智能体系统中的问题、成因与解决方案

    Understanding Issues, Causes and Solutions in Open-Source LLM-based Multi-Agent Systems

    [https://arxiv.org/abs/2610.00905](https://arxiv.org/abs/2610.00905)

    该论文通过对21个开源LLM多智能体系统项目中944个相关议题的实证分析，系统揭示了实践者在开发和部署此类系统时面临的主要问题、深层成因及潜在解决方案。

    

    随着基于大语言模型的多智能体系统的不断发展，越来越多的开源项目开始采用多智能体架构作为其核心功能的基础。尽管针对多智能体系统的研究与实践已引起广泛关注，但探讨开源LLM多智能体系统实践者所面临挑战、这些挑战的成因以及潜在解决方案的研究仍然有限。为填补这一空白，我们开展了一项实证研究，以了解实践者在开发和使用开源LLM多智能体系统时遇到的问题、这些问题的可能成因以及潜在的解决方案。我们从21个开源LLM多智能体系统项目中收集了22,848个已关闭的议题，并采用自动化与人工相结合的混合过滤方法，将数据集缩减至944个与LLM多智能体系统相关的议题。随后，我们对这些议题进行分析，以了解实践者经常遇到的问题及其深层成因。

    arXiv:2610.00905v1 Announce Type: cross  Abstract: With the advancement of LLM-based multi-agent systems (MAS), an increasing number of opensource projects are adopting multi-agent architectures as the foundation of their core functionality. Although research and practice on MAS have attracted considerable attention, limited studies have explored the challenges faced by practitioners of open-source LLM-based MAS, the causes of these challenges, and potential solutions. To address this gap,we conducted an empirical study to understand the issues that practitioners encounter when developing and using open-source LLM-based MAS, the possible causes of these issues, and potential solutions. We collected 22,848 closed issues from 21 open-source LLM-basedMASand applied a mixed automated and manual filtering approach to reduce the dataset to 944 issues related to LLM-based MAS.We then analyzed these issues to understand the frequent issues encountered by practitioners, their underlying causes,
    
[^19]: 基于智能体编码任务强化学习的跨基准迁移

    Cross-Benchmark Transfer from RL on Agentic Coding Tasks

    [https://arxiv.org/abs/2610.00890](https://arxiv.org/abs/2610.00890)

    仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。

    

    编码智能体常常在“最后一公里”失败：它们完成了功能的大部分却遗漏某项需求、只测试其实现已经能处理的情况、破坏了本应保持不变的行为，或基于未经检验的假设进行验证。我们探究在专家构建的智能体编码任务上进行强化学习（RL）能否弥合这一差距，以及智能体所学内容能否迁移到训练分布之外。我们仅使用强化学习对 Kimi K2.7 Code（一个 1 万亿参数、激活参数 32B 的开源混合专家模型）在 1,700 个任务上进行后训练：其中 1,000 个仓库任务由隐藏的 fail-to-pass 测试和保护既有行为的 pass-to-pass 测试评分，700 个终端任务由专家编写的隐藏验证器评分。奖励为目标检查的通过比例，若任何 pass-to-pass 测试失败则奖励降为零。在秩为 32 的 LoRA 适配器上进行一个 epoch 的 GSPO 训练，在我们评估的六个外部基准上的 pass@1 均有提升，跨越三个……（原文摘要在此处截断）

    arXiv:2610.00890v1 Announce Type: cross  Abstract: Coding agents often fail in the last mile: they build most of a feature but drop a requirement, test only the cases their implementation already handles, break behavior that was supposed to stay intact, or validate against an unchecked assumption. We ask whether reinforcement learning (RL) on expert-built agentic coding tasks closes this gap, and whether what the agent learns transfers beyond the training distribution. We post-train Kimi K2.7 Code, a 1T-parameter (32B active) open-weight mixture-of-experts model, with RL alone on 1,700 tasks: 1,000 repository tasks graded by hidden fail-to-pass tests and by pass-to-pass tests of existing behavior, and 700 terminal tasks graded by expert-written hidden verifiers. The reward is the fraction of target checks passed and drops to zero if any pass-to-pass test fails. One epoch of GSPO on a rank-32 LoRA adapter improves pass@1 on each of the six external benchmarks we evaluated, across three 
    
[^20]: FORALL-LEAN-AGENT：面向形式化数学与软件验证中可审计推理的框架

    FORALL-LEAN-AGENT for Auditable Reasoning in Formal Mathematics and Software Verification

    [https://arxiv.org/abs/2610.00885](https://arxiv.org/abs/2610.00885)

    该论文提出FORALL-LEAN-AGENT框架，通过隔离工作区、陈述比对、公理审计和独立证明检查等机制实现Lean形式化证明的可审计验证，在VeriSoftBench上将GPT-5.6 Sol的成功率从93提升至100并降低了成本。

    

    编码智能体日益自动化Lean证明的开发，但仅凭编译成功并不能确立候选证明在可接受的假设下证明了预期的陈述。我们提出FORALL-LEAN-AGENT，这是一个与前端无关的框架，用于形式化数学和软件验证中的可审计推理。该框架结合了隔离工作区、Lean工具，以及包含陈述比对、公理审计和独立证明检查（在支持的情况下）的全新审查机制。验证证据与审查者的决定均绑定到同一候选工件，使接受过程可追溯。我们在VeriSoftBench、PutnamBench以及Lean Eval软件验证赛道中的两个问题上对该框架进行了评估。在100个任务的VeriSoftBench子集上，与FORALL-LEAN-AGENT集成使GPT-5.6 Sol在低工作量下的基准规则成功率从93提升至100，同时将成本从69美元降至62美元。PutnamBench评估接受了全部672个（摘要在此处被截断）

    arXiv:2610.00885v1 Announce Type: cross  Abstract: Coding agents increasingly automate Lean proof development, but successful compilation alone does not establish that a candidate proves the intended statement under acceptable assumptions. We present FORALL-LEAN-AGENT, a frontend-agnostic framework for auditable reasoning in formal mathematics and software verification. The framework combines isolated workspaces, Lean tools, and fresh review with statement comparison, axiom audits, and independent proof checking where supported. Verification evidence and reviewer decisions are bound to the same candidate artifact, making acceptance traceable. We evaluate the framework on VeriSoftBench, PutnamBench, and both problems in the Lean Eval softwareverification track. On the 100-task VeriSoftBench subset, integration with FORALLLEAN-AGENT raises benchmark-rule success from 93 to 100 for GPT-5.6 Sol at low effort while reducing cost from $69 to $62. The PutnamBench evaluation accepts all 672 pr
    
[^21]: 正确性、收敛性与AI生成代码检测：对入门编程课程中学生代码与大语言模型代码的纵向研究

    Correctness, Convergence, and AI-Generated Code Detection: A Longitudinal Study of Student and Large Language Model Code in Introductory Programming

    [https://arxiv.org/abs/2610.00863](https://arxiv.org/abs/2610.00863)

    该纵向研究通过分析2021至2025年间近三万份学生提交与九万条大语言模型生成的解，发现LLM生成的解通常正确且在实现上高度收敛，学生代码与生成参考解的匹配率随时间上升，但由于简单题目本身就只有少数自然解法，这种匹配能否作为检测学生使用AI写代码的有效依据仍存疑。

    

    大语言模型能够为编程作业生成看似合理的解决方案，这使得通过将学生代码与一个由生成解构成的参考库进行匹配来检测AI的使用变得很有吸引力。然而，当一道作业题只存在少数几种自然的实现方式时，相似的代码也可能自然出现，这就使得匹配结果究竟意味着什么仍是一个悬而未决的问题。我们使用来自2021年、2023年和2025年十个Python实验的29,970份学生提交，以及由三个前沿大语言模型回顾性生成的90,000次求解尝试，研究了生成参考匹配方法。我们使用隐藏的教师测试用例验证生成的解决方案，在排除起始代码后使用MOSS进行代码比较，并检查所选函数的精确抽象语法树（AST）形式。这些模型通常能生成正确的解决方案，并且在大多数作业上收敛于相似的实现。学生提交在较晚的……

    arXiv:2610.00863v1 Announce Type: cross  Abstract: Large language models can generate plausible solutions to programming assignments, making it tempting to detect their use by matching student code against a reference bank of generated solutions. Yet similar code can also arise when an assignment admits only a few natural implementations, which leaves open what a match actually shows.   We investigate generated-reference matching using 29,970 student submissions from ten Python labs offered in 2021, 2023, and 2025, together with 90,000 solution attempts generated retrospectively by three frontier LLMs. We validate the generated solutions using hidden instructor tests, compare code with MOSS after excluding the starter code, and examine the exact abstract syntax tree (AST) forms of selected functions.   The models usually produced correct solutions and, across most assignments, converged on similar implementations. Student submissions matched the generated references more often in later
    
[^22]: ASAD：用于调试的自适应软件智能体

    ASAD: Adaptive Software Agents for Debugging

    [https://arxiv.org/abs/2610.00629](https://arxiv.org/abs/2610.00629)

    ASAD 提出了一种自适应多智能体调试系统，根据 bug 的性质和复杂度动态配置智能体数量、专业角色及协作策略，克服了传统固定架构框架“一刀切”的局限性。

    

    将大语言模型（LLM）集成到多智能体系统中，已在自动化调试领域展现出巨大潜力。然而，几乎所有现有框架都依赖于僵化的预定义架构：智能体的数量、角色及其交互模式在任何 bug 分析开始之前就已经固定。这种“一刀切”的方法与软件缺陷的异质性本质从根本上不匹配——简单的 bug 会因不必要的协调而浪费资源，而复杂的 bug 则因专业知识不足或匹配不当而表现不佳。本文提出了 ASAD，一个用于调试的自适应智能体系统，它能够根据每个 bug 的性质和复杂度来配置其团队。ASAD 通过分析有缺陷的代码来启动调试过程，并动态确定需要部署的智能体数量、它们应承担的专业角色以及它们应遵循的协作策略。一个中央协调器负责编排整个……

    arXiv:2610.00629v1 Announce Type: new  Abstract: The integration of Large Language Models (LLMs) into multi-agent systems has shown great potential for automated debugging. Yet nearly all current frameworks rely on rigid, predefined architectures: the number of agents, their roles, and their interaction patterns are fixed before any analysis of the bug occurs. This one-size-fits-all approach is fundamentally mismatched to the heterogeneous nature of software defects. Simple bugs waste resources on unnecessary coordination, while complex ones suffer from insufficient or poorly aligned expertise. This paper introduces ASAD, an adaptive agentic system for debugging that configures its team according to the nature and complexity of each bug. ASAD initiates the debugging process by analyzing the faulty code and dynamically determines the number of agents to deploy, the specialized roles they should have, and the collaboration strategy they should follow. A central coordinator orchestrates t
    
[^23]: 理解并缓解大语言模型生成代码中的库相关问题

    Understanding and Mitigating Library-Related Issues in LLM-Generated Code

    [https://arxiv.org/abs/2610.00622](https://arxiv.org/abs/2610.00622)

    本文通过探索性研究揭示84%的LLM生成代码文件存在库相关错误，并据此提出一种代理式方法来理解和缓解这些库使用问题。

    

    软件从业者越来越依赖大语言模型（LLM）来生成集成外部库的代码。然而，LLM经常产生不正确的库使用方式，例如无效的导入、过时的API调用和幻觉依赖，导致编译或运行时失败，降低了AI辅助软件开发的可靠性。在本文中，我们提出了一种代理式方法来缓解LLM生成代码中的库相关错误。更具体地说，我们首先开展了一项探索性研究，以刻画LLM产生的库相关问题的特征。我们对100个LLM生成的代码文件的分析表明，84%的生成文件包含至少一个库相关错误，其中反复出现的模式包括错误的导入路径、缺失的导入、幻觉库、已弃用的库使用以及未使用的导入。基于这些发现，我们设计了一种代理式方法，该方法集成了任务分析、文档（摘要在此处被截断）……

    arXiv:2610.00622v1 Announce Type: new  Abstract: Software practitioners increasingly rely on Large Language Models (LLMs) to generate code that integrates external libraries. However, LLMs often produce incorrect library usage, such as invalid imports, outdated API calls, and hallucinated dependencies, leading to compilation or runtime failures that reduce the reliability of AI-assisted software development. In this paper, we propose an agentic approach to mitigate libraryrelated errors in LLM-generated code. More specifically, we first conduct an exploratory study to characterize the library-related issues produced by LLMs. Our analysis of 100 LLM-generated code files reveals that 84% of generated files contain at least one library-related error, with recurring patterns including incorrect import paths, missing imports, hallucinated libraries, deprecated library usage, and unused imports. Based on these findings, we design an agentic approach that integrates task analysis, documentati
    
[^24]: 扩展基于大语言模型对患有ADHD的软件工程师的支持

    Extending LLM-based support for software engineers with ADHD

    [https://arxiv.org/abs/2610.00555](https://arxiv.org/abs/2610.00555)

    本文提出了Tether 2.0，一个基于大语言模型的助手，通过结构化交互模式、活动感知上下文和持久记忆，在规划、编码、调试和评审的完整工作流中为患有ADHD的软件工程师提供针对性支持。

    

    软件工程工作流程通常未能满足患有注意缺陷多动障碍（ADHD）的开发者的需求，尽管其在任务启动、持续注意力和任务完成方面存在已知的挑战。与此同时，大语言模型（LLM）正日益被集成到编程工具中，但现有系统并未考虑神经多样性，也不支持跨开发任务的结构化推进。在本工作中，我们提出了Tether 2.0，一个基于大语言模型的助手，旨在通过面向工作流的交互，在规划、编码、调试和评审等环节为患有ADHD的软件工程师提供支持。该工具被命名为Tether 2.0，因为它建立在原始Tether系统的开源代码和基础之上。我们的方法结合了结构化交互模式、活动感知上下文和持久记忆，以支持任务的推进和连续性。我们通过专家反馈来评估该系统。

    arXiv:2610.00555v1 Announce Type: new  Abstract: Software engineering workflows are often not designed to accommodate the needs of developers with Attention Deficit Hyperactivity Disorder (ADHD), despite known challenges related to task initiation, sustained attention, and completion. At the same time, large language models (LLMs) are increasingly integrated into programming tools, but existing systems do not account for neurodiversity or support structured progression across development tasks. In this work, we present Tether 2.0, an LLM based assistant designed to support software engineers with ADHD through workflow oriented interaction across planning, coding, debugging, and review. The tool was named Tether 2.0 because it builds upon the open source code and foundations of the original Tether system. Our approach combines structured interaction modes, activity aware context, and persistent memory to support task progression and continuity. We evaluate the system through expert feed
    
[^25]: 代码能运行，环境却失灵：衡量AI生成软件中的环境可复现性

    Code That Works, Environments That Don't: Measuring Environment Reproducibility in AI-Generated Software

    [https://arxiv.org/abs/2610.00425](https://arxiv.org/abs/2610.00425)

    提出环境规约智能体协议及“声明式—运行时—必要且充分”三层依赖框架，系统评估揭示当前编码智能体虽能生成功能正确的代码，却普遍无法准确指定软件环境依赖。

    

    代码生成已成为大语言模型的一项核心能力，编码智能体现在能够根据自然语言提示生成功能正确的软件项目。然而，仅凭功能正确性并不能涵盖生成质量的一个关键维度：环境规约，即准确识别运行所生成代码所需的依赖项，同样至关重要。我们开发了一种用于环境规约的智能体协议，并引入了一个三层框架，由声明式依赖、运行时安装依赖以及必要且充分的依赖构成，以系统性地评估编码智能体的环境规约能力。利用该协议，我们评估了编码智能体在多大程度上系统性地错误指定软件环境依赖，以及这种错误指定在三个智能体、四种编程语言和五十个编程任务之间的变化情况。我们的结果表明，当前的编码智能体……（原文在此处截断）

    arXiv:2610.00425v1 Announce Type: cross  Abstract: Code generation has emerged as a central capability of large language models, with coding agents now able to produce functionally correct software projects from natural language prompts. However, functional correctness alone does not capture a critical dimension of generation quality: environment specification, defined as the accurate identification of the dependencies required to execute generated code, is equally critical. We develop an agent protocol for environment specification and introduce a three-layer framework comprising declared, runtime-installed, and necessary-and-sufficient dependencies to systematically assess coding agents for environment specification. Using this protocol, we evaluate the extent to which coding agents systematically misspecify software environment dependencies and how this misspecification varies across three agents, four languages, and fifty programming tasks. Our results show that current coding agen
    
[^26]: 规则到工具：面向科学计算中大语言模型智能体的可执行检查

    Rules to Tools: Executable Checks for LLM Agents in Scientific Computing

    [https://arxiv.org/abs/2610.00313](https://arxiv.org/abs/2610.00313)

    该论文提出Rules to Tools（R2T）方法，将书面形式的科学计算要求转化为可由大语言模型智能体直接调用的预先准备好的可执行检查，实验证明这种可执行检查能显著提升科学编码智能体修复程序的成功率（从26/30提升至29/30）。

    

    科学编码智能体以书面形式接收方程、边界条件和输出要求，随后必须评估其所修改的程序。Rules to Tools（R2T）提供了针对公开科学要求预先准备好的可执行检查。匹配的SciCode修复组共享相同的书面检查、起始程序、模型和预算；其中工具组额外获得一个可调用的实现。在两个任务ID队列中，使用文本描述时的完全修复率为26/30，而使用准备好的可执行检查时为29/30。其中三个任务ID偏向工具组，一个偏向文本组，十一个任务ID打平。八个ID队列的得分为13/16对15/16，两组差异的任务聚类bootstrap 95%置信区间为[-12.5, 43.75]个百分点。更大规模的共享定义SciCode队列中两组均为13/24，结果持平。五个在开发阶段暴露过的任务（使用替代起始程序）的得分为3/10对7/10。工具组在任务17、77和11上占优；初始检查标记了任务17的违规，并报告任务77无违规。

    arXiv:2610.00313v1 Announce Type: new  Abstract: Scientific coding agents receive equations, boundary conditions, and output requirements in writing, then must assess the programs they revise. Rules to Tools (R2T) supplies prepared executable checks of public scientific requirements. Matched SciCode repair groups share written checks, starting programs, model, and budgets; the tool group receives a callable implementation. Across two task-ID cohorts, complete repair is 26/30 with text and 29/30 with the prepared checks. Three task IDs favor tools, one favors text, and eleven tie. The eight-ID cohort scores 13/16 versus 15/16, with a task-cluster bootstrap 95% interval of [-12.5, 43.75] percentage points for the difference. The larger shared-definition SciCode cohort ties at 13/24 per group. Five development-exposed tasks with alternate starting programs score 3/10 versus 7/10. The tool group favors tasks 17, 77, and 11; initial checks flag task 17 and report no violation for tasks 77 a
    
[^27]: 基于保证契约与证据的量子编译器测试与验证

    Testing and Verification of Quantum Compilers through Assurance Contracts and Evidence

    [https://arxiv.org/abs/2610.00255](https://arxiv.org/abs/2610.00255)

    该论文是一项关于量子编译器测试与验证的批判性综合综述，系统比较了经验证变换、等价性检查、差分与蜕变测试、基于属性的测试等多种方法，并通过失败案例和工作示例说明如何借助保证契约与证据确保量子编译器在其交付接口上的正确性。

    

    量子编译器的保证要求语义关系与预期用途相匹配，并需要能够在交付接口处暴露违规行为的观测。这项批判性综合综述比较了经验证的变换、等价性检查、差分测试与蜕变测试、基于属性的测试，以及跨修订版本的保证证据。通过一个源文献目录和聚焦的提取矩阵，该综述区分了形式化保证、报告的实证结果、分析性推论以及所提出的实践方法。所记录的覆盖范围更新和源层面的选择决策使该综述在其声明的范围内可被审计。四个有记录的失败案例将条件适用性、参数关联、终止性与相位约定与方法选择联系起来。一个涉及布局元数据、参数绑定和辅助量子比特的工作示例演示了为什么在重构解释下验证电路可能会使交付……（摘要在此处截断）

    arXiv:2610.00255v1 Announce Type: new  Abstract: Quantum compiler assurance requires a semantic relation that matches the intended use and observations that can expose violations at the delivered interface. This critical integrative survey compares verified transformations, equivalence checking, differential and metamorphic testing, property-based testing, and evidence for assurance across revisions. A source catalogue and focused extraction matrix distinguish formal guarantees, reported empirical findings, analytical deductions, and proposed practice. A recorded coverage update and source-level selection decisions make the review auditable within its declared scope. Four documented failure cases connect conditional applicability, parameter association, termination, and phase conventions to method selection. A worked example involving layout metadata, parameter bindings, and ancillary qubits demonstrates why validating a circuit under a reconstructed interpretation can leave the delive
    
[^28]: 定位 MCP 智能体框架中网络传输后的语义变化

    Localizing Post-Wire Semantic Changes in MCP Agent Frameworks

    [https://arxiv.org/abs/2610.00182](https://arxiv.org/abs/2610.00182)

    提出一种差分测试方法，通过追踪固定的工具结果经过 MCP 智能体框架的公共接口来定位语义变化，在四个 Python 集成的 18 个测试用例中发现了 13 处涉及结构化值、错误声明和丰富内容的分歧。

    

    有效的模型上下文协议（MCP）消息并不能保证智能体框架会保留下游软件所需的语义区分。我们提出了一种差分测试方法，该方法追踪一个固定的工具结果经过每个框架的公共接口，并检查明确的消费者需求。该方法应用于四个版本固定的 Python 集成中的 18 个专门设计的测试用例，识别出 13 处独特的测试用例与任务之间的分歧，涉及结构化值、声明的错误和丰富内容。Google ADK 在其观察到的路径中满足所有主要契约；其他集成则表现出特定于接口的变化或执行失败。某一变化是否重要取决于消费者：将缺失的可选字段视为等同于 null 可以解释 OpenAI 大多数严格的丰富内容失败情况。在一项探索性重放实验中，解析 JSON 文本可以恢复更多结构化值，但也会返回错误的值以及源数据中本不存在的字段的值。

    arXiv:2610.00182v1 Announce Type: new  Abstract: Valid Model Context Protocol (MCP) messages do not guarantee that an agent framework preserves the distinctions downstream software needs. We present a differential testing method that follows a fixed tool result through each framework's public interfaces and checks explicit consumer requirements. Applied to 18 designed fixtures in four pinned Python integrations, the method identifies 13 unique fixture-task divergences involving structured values, declared errors, and rich content. Google ADK satisfies every primary contract in its observed path; the other integrations show interface-specific changes or an execution failure. Whether a change matters depends on the consumer: treating absent optional fields as equivalent to null explains most of OpenAI's strict rich-content failures. In an exploratory replay, parsing JSON text recovers more structured values but also returns incorrect values and values for fields absent at the source. Doc
    
[^29]: 恶意软件复杂度的表征与规范化

    Characterizing and Codifying Malware Sophistication

    [https://arxiv.org/abs/2610.00098](https://arxiv.org/abs/2610.00098)

    本文首次以质量视角系统化定义了恶意软件复杂度，通过重新解读ISO/IEC 25010软件质量标准并筛选出可通过静态二进制分析度量的相关特性，为源代码不可用时一致评估恶意软件复杂度的框架奠定了基础。

    

    “复杂”一词被广泛用于描述恶意软件，但在学术文献中却缺乏一致的定义。尽管现有的软件质量与复杂度指标能为理解恶意软件结构提供一定见解，但它们未能捕捉到决定真实世界威胁潜力的更广泛的对抗性与操作性特征。本文对使用静态二进制分析评估恶意软件质量的现有方法进行了系统化梳理。我们通过以质量为中心的视角定义恶意软件复杂度，重新解读了ISO/IEC 25010软件质量标准中的部分特性，包括可靠性、可维护性、灵活性和安全性，并评估了这些特性对恶意软件二进制文件的适用性。我们识别出哪些特性既与恶意软件相关又可通过静态分析进行度量，为未来在源代码不可用情况下一致评估恶意软件复杂度的框架奠定了基础。

    arXiv:2610.00098v1 Announce Type: cross  Abstract: 'Sophisticated' is widely used to describe malware, yet it lacks a consistent definition within academic literature. While existing software quality and complexity metrics offer some insight into malware structure, they do not capture the broader adversarial and operational traits that contribute to real-world threat potential. This paper presents a systematization of existing approaches for assessing malware quality using static binary analysis. We define malware sophistication through a quality-focused lens by reinterpreting select characteristics from the ISO/IEC 25010 software quality standard, including reliability, maintainability, flexibility, and security, and evaluating their applicability to malware binaries. We identify which characteristics are both relevant to malware and measurable through static analysis, forming the basis for future frameworks that consistently assess malware sophistication when source code is unavailab
    
[^30]: 受保护的提交：面向大语言模型工作流的事务式人工审批

    Guarded Commits: Transactional Human Approvals for LLM Workflows

    [https://arxiv.org/abs/2610.00037](https://arxiv.org/abs/2610.00037)

    该论文提出“受保护提交”设计，将人工审批作为LLM工作流状态的组成部分，在不可逆外部操作执行前，通过账本决议记录与凭证受限的提交适配器，强制校验每条风险路径均已通过审批关口并核实证据、策略版本与执行路径。

    

    大语言模型（LLM）工作流在执行不可逆的外部操作前，通常需要人工审批。大多数系统将这种审批置于工作流之外，仅表现为一次界面点击或一条审计记录。因此，工作流缺乏提交时检查机制，无法确保每条风险路径都经过了审批关口；其日志也可能无法保存被审查的证据，或保存复用先前决策所需的条件。我们提出一种“受保护提交”设计，将人工审批纳入工作流状态的一部分。在执行外部操作之前，决策源会将一条决议记录及其引用的证据追加到账本中；随后，一个凭证受限的提交适配器会将该记录与制品、策略版本及实际执行的路径进行核对。我们的证据仅限于轨迹重建：在合成无环工作流计划上进行的验证器测试能够接受无缺陷的计划，并拒绝每一种被注入缺陷的计划——缺失审批关口、关口类型错误或路径覆盖不完整。在三个公……（摘要原文在此处截断）

    arXiv:2610.00037v1 Announce Type: cross  Abstract: LLM workflows often require human approval before an irreversible external action. Most systems keep that approval outside the workflow, as an interface click or an audit entry. The workflow therefore lacks a commit-time check that every risky path reached an approval gate. Its logs may not preserve the reviewed evidence or the conditions for reusing an earlier decision. We present a guarded-commit design that makes human approval part of workflow state. Before the external action runs, a decision source appends a resolution record and its referenced evidence to a ledger. A credential-confined commit adapter then checks that record against the artifact, policy version, and executed path. Our evidence is limited to trace reconstruction. A validator test on synthetic acyclic workflow plans accepts unfaulted plans and rejects plans with each injected fault: a missing gate, the wrong gate type, or incomplete path coverage. Across three pub
    
[^31]: 基于大语言模型自动化的软件项目管理：实践中的协调、验证与治理

    Software Project Management with LLM-Based Automation: Coordination, Validation, and Governance in Practice

    [https://arxiv.org/abs/2610.00027](https://arxiv.org/abs/2610.00027)

    本研究通过探索性案例研究发现，基于大语言模型的自动化并未引入新的正式管理实践，而是影响了软件项目经理在规划、估算、协调、监控和治理等活动中的工作方式与学习需求。

    

    随着软件工程的不断发展，开发环境日益集成了对测试、分析和代码生成等任务的自动化支持，这要求项目管理能够同时协调人类工作与自动化工作流程。在本文中，我们研究了软件项目经理如何感知与大语言模型（LLM）自动化相关的工作变化及学习需求。鉴于软件工程研究此前大多聚焦于任务层面和以开发者为中心的LLM应用而存在研究空白，我们采用探索性案例研究方法，捕捉实践中关于自动化的管理者视角。基于在一家大型多项目软件组织中工作的软件项目经理的经验，我们的分析表明，基于LLM的自动化影响的是规划、估算、协调、监控和治理等活动，而非引入新的正式管理实践。参与者……（原文摘要在此处截断）

    arXiv:2610.00027v1 Announce Type: new  Abstract: As software engineering has evolved, development environments have increasingly integrated automated support for tasks such as testing, analysis, and code generation, requiring project management to coordinate both human work and automated workflows. In this paper, we investigate how software project managers perceive changes in their work and learning demands associated with LLM-based automation. Motivated by a gap in software engineering research, which has largely focused on task-level and developer-centered uses of LLMs, we adopted an exploratory case study approach to capture managerial perspectives on automation in practice. Based on the experience of software project managers working in a large, multi-project software organization, our analysis indicates that LLM-based automation influences planning, estimation, coordination, monitoring, and governance activities rather than introducing new formal management practices. Participant
    
[^32]: 从验证失败到编码智能体的可复用指导

    From Verification Failures to Reusable Guidance for Coding Agents

    [https://arxiv.org/abs/2609.39022](https://arxiv.org/abs/2609.39022)

    该论文提出将专家对验证失败的诊断转化为编码智能体可复用的指导，结合K框架的可执行语言语义与一套用于构建规范、修复证明和审计充分性的工具包，在HumanEval上实现164/164的全通过率，并通过对照实验证明审计能识别出证明通过但存在缺陷的软件包。

    

    编码智能体需要确认程序满足规范，并且该规范确实刻画了所要求的行为。我们研究如何将专家对验证失败的诊断转化为这项工作中可复用的指导。我们的方法将K框架中的可执行语言定义与一套用于构建规范、修复证明以及审计其充分性的流程工具包相结合。在HumanEval（一个包含164个Python编程任务的基准测试）上进行的人工指导开发活动中，借助该语义定义和工具包，以两次针对性修复后最终AI审计的Pass判定为衡量标准，达到了164/164的成功率。为了检验审计能否发现成功证明所遗留的未决问题，我们构建了12对经作者审查的“干净”与“缺陷”软件包。每个软件包都通过了其K证明，而已完成的审计识别出了所有缺陷，并接受了所有干净的软件包。随后，我们使用KleverBench来测试规范……（摘要原文在此处截断）

    arXiv:2609.39022v1 Announce Type: cross  Abstract: Coding agents need to establish that a program satisfies a specification and that the specification captures the requested behavior. We study how expert diagnosis of verification failures can become reusable guidance for this work. Our approach combines executable language definitions in the K framework with a kit of procedures for constructing specifications, repairing proofs, and auditing their adequacy. A human-guided development campaign on HumanEval, a benchmark of 164 Python programming tasks, achieves a 164/164 success rate with the semantics and the kit, measured by final AI audit Pass verdicts after two targeted repairs. To examine whether auditing detects problems that successful proofs leave unresolved, we construct 12 author-reviewed pairs of clean and defective packages. Every package passes its K proofs, and completed audits identify all defects and accept all clean packages. We then use KleverBench to test specification 
    
[^33]: Zero2Repo：编程智能体能否从零开始构建代码仓库？

    Zero2Repo: Can Coding Agents Build Repositories from Scratch?

    [https://arxiv.org/abs/2609.38269](https://arxiv.org/abs/2609.38269)

    提出 Zero2Repo 基准，通过语言无关的自动化流水线将真实的开源项目转化为“从零构建完整代码仓库”的任务，并以可执行验收测试和对抗性验证来严格评估编程智能体的从零构建能力。

    

    编程智能体越来越多地被要求从零构建软件，而非修补现有软件，然而针对从零构建代码仓库的基准测试大多局限于单一语言，并且依赖人工整理的任务。我们提出了 Zero2Repo，这是一个基准测试：智能体接收一份产品需求文档、一个接口契约和一个空的工作区，必须在该项目原生生态系统中交付一个完整的代码仓库。任务由一个与语言无关的任务生成流水线产出，该流水线将真实的、版本锁定的开源项目转换为行为规范、可复现的环境和隐藏的验收测试。每个任务都通过执行来验证：源自上游项目的参考实现必须通过测试，且对抗性验证必须证明这些测试能够拒绝错误的实现。评估在生产级编程智能体上进行，智能体运行于隔离容器中，在明确提交之前无法获得验收测试。

    arXiv:2609.38269v1 Announce Type: cross  Abstract: Coding agents are increasingly asked to build software rather than patch it, yet benchmarks for from-scratch repository construction are mostly limited to a single language and depend on manually curated tasks. We introduce Zero2Repo, a benchmark in which an agent receives a product requirements document, an interface contract, and an empty workspace, and must deliver a complete repository in the project's native ecosystem. Tasks are produced by a language-agnostic authoring pipeline that converts real, version-pinned open-source projects into behavioral specifications, reproducible environments, and hidden acceptance tests. Each task is validated by execution: a reference implementation derived from the upstream project must pass, and adversarial validation must show that the tests reject incorrect implementations. Evaluation runs production coding agents in isolated containers, withholds the acceptance tests until an explicit submiss
    
[^34]: 不透明指针下的神经符号间接调用分析

    Neuro-Symbolic Indirect-Call Analysis under Opaque Pointers

    [https://arxiv.org/abs/2609.33547](https://arxiv.org/abs/2609.33547)

    提出Facet，这是首个在不透明指针的LLVM IR上重建间接调用分发关系的分析，通过识别调用加载函数指针的结构体字段，并独立恢复通过初始化器、存储和聚合拷贝赋给该字段的函数，再按字段标识将二者关联。

    

    解析间接调用是构建C语言调用图的核心。诸如MLTA这类可扩展的基于类型的分析，利用LLVM IR中的类型信息将间接调用与赋给相应结构体字段的函数关联起来。然而，单一的被指类型往往无法准确表征指针所寻址的内存，并且LLVM 17移除了被指类型，转而采用不透明指针。因此，字段敏感的分析失去了匹配键。恢复被擦除的类型虽然能够找回匹配键，但仍然遗漏了类型所编码的关系：即程序将哪些函数赋给该字段。我们提出了Facet，据我们所知，这是第一个在不透明IR之上重建这种分发关系的分析。Facet识别间接调用从哪个结构体字段加载其函数指针，并分别通过初始化器、存储操作和聚合拷贝来恢复赋给该字段的函数，然后通过字段标识将两者连接起来。

    arXiv:2609.33547v2 Announce Type: replace  Abstract: Resolving indirect calls is central to call-graph construction for C. Scalable type-based analyses such as MLTA use type information in LLVM IR to associate indirect calls with functions assigned to the corresponding structure fields. However, a single pointee type often misrepresents the memory a pointer addresses, and LLVM 17 removed pointee types in favor of opaque pointers. Therefore, field-sensitive analyses lose their matching key. Recovering the erased types restores the matching key but still misses the relation that the type encoded: which functions the program assigns to the field. We present Facet, to our knowledge the first analysis that reconstructs this dispatch relation over opaque IR. Facet identifies the structure field from which an indirect call loads its function pointer. It separately recovers the functions assigned to that field through initializers, stores, and aggregate copies. It then joins the two by field i
    
[^35]: 技能究竟能做什么？大语言模型智能体中工具与技能使用的估计对象与评估效度：一项批判性综述

    What Does a Skill Actually Do? Estimands and Evaluation Validity for Tool and Skill Use in LLM Agents: A Critical Review

    [https://arxiv.org/abs/2609.33153](https://arxiv.org/abs/2609.33153)

    本文对大语言模型智能体中工具与技能使用的评估研究进行批判性综述，提出以处理对比、目标人群、结果、预算约束、汇总度量和识别假设六个维度刻画评估设计的分析框架，并证明常见的同任务配对运行设计在以触发为条件时无法识别技能调用的因果效应。

    

    arXiv:2609.33153v2 公告类型：replace-cross 摘要：大语言模型智能体中工具与可复用技能所带来的报告改进，实际指向的是各不相同的比较对象。本批判性叙述性综述考察了这些评估究竟在估计什么，以及其研究设计能够支持哪些结论。本综述梳理了一百篇被引文献的角色，并从三十五项研究中详细提取了重点评估设计；同时对另外三十五篇已发表或已录用的研究进行了针对性阅读，以扩展在工具创建、记忆、交互式基准、可靠性与风险等方面的覆盖范围。这些设计按照处理对比、目标人群、结果指标、预算约束、汇总度量以及识别假设加以刻画。解析分解与反例表明：当评估以处理组运行中的触发为条件时，在同一任务上配对运行本身并不能识别出调用效应。配对增益与回归计数描述了在耦合条件下……

    arXiv:2609.33153v2 Announce Type: replace-cross  Abstract: Reported improvements from tools and reusable skills in large language model agents refer to different comparisons. This critical narrative review examines what these evaluations estimate and which conclusions their designs support. The review checks the roles of one hundred cited papers and extracts focal evaluation designs in detail from thirty-five studies. Targeted readings of thirty-five additional published or accepted studies broaden coverage of tool creation, memory, interactive benchmarks, reliability, and risk. Designs are characterized by treatment contrast, target population, outcome, budget constraint, summary measure, and identification assumptions. Analytic decompositions and counterexamples show that pairing runs on the same task does not itself identify an invocation effect when evaluation conditions on a trigger within the treated run. Paired gain and regression counts describe discordance under the coupling p
    
[^36]: 智能体网络安全的下一个挑战：一个现实且无污染的逆向工程基准测试

    The Next Challenge for Agentic Cybersecurity: A Realistic, Contamination-Free Reverse Engineering Benchmark

    [https://arxiv.org/abs/2608.11469](https://arxiv.org/abs/2608.11469)

    本文提出了SRE-Bench，这是首个现实且无污染的逆向工程基准，由专家从零构建，确保实例在训练数据中不可见，以真实评估AI智能体的逆向工程能力。

    

    arXiv:2608.11469v1 公告类型：交叉 摘要：当源代码可供分析时，AI智能体在网络安全能力方面正在迅速提升，然而对网络安全最具影响力的软件，包括恶意软件、固件和专有应用，通常仅以二进制形式提供。分析此类软件需要逆向工程（RE）：在分析有效执行前恢复程序语义。然而，评估智能体逆向工程面临根本性挑战：基准实例必须作为源代码在LLM训练数据中不可见，以防止模型通过识别它们而非真正分析来走捷径，同时还要匹配真实软件的规模和反分析保护。不幸的是，现有基准无法同时满足这些要求。为此，我们引入了SRE-Bench，这是首个现实且无污染的逆向工程基准，由逆向工程专家从零构建，耗时超过5000小时。

    arXiv:2608.11469v1 Announce Type: cross  Abstract: AI agents are rapidly improving in cybersecurity capabilities when the source code is available for analysis, yet much of the software most consequential to cybersecurity, including malware, firmware, and proprietary applications, is available only as binaries. Analyzing such software requires reverse engineering(RE): recovering program semantics before the analysis can be meaningfully performed. However, evaluating agentic RE poses a fundamental challenge: benchmark instances must be unseen as source code in the LLMs' training data to prevent models from taking shortcuts by recognizing them rather than really analyzing them, while also matching the scale and anti-analysis protections of real software. Unfortunately, however, existing benchmarks do not jointly satisfy these requirements. To this end, we introduce SRE-Bench, the first realistic, contamination-free RE benchmark. Built entirely from scratch by RE experts with over 5,000 h
    
[^37]: 评估Dart AOT二进制文件的神经反编译：微调、指标有效性、规范泄漏与可靠性

    Evaluating Neural Decompilation of Dart AOT Binaries: Fine-Tuning, Metric Validity, Specification Leakage, and Reliability

    [https://arxiv.org/abs/2607.06125](https://arxiv.org/abs/2607.06125)

    该论文对Dart AOT二进制文件的神经反编译进行了基于执行的系统性评估，揭示了微调适配器存在功能性回归、传统静态指标与真实功能正确性关联较弱，以及模型严重依赖语义名称等规范泄漏线索，从而质疑了当前神经反编译评估方法的可靠性。

    

    我们针对Dart提前编译（AOT）二进制文件的神经反编译提出了一种基于执行的评估方法，并审查了其评分实际衡量的内容。在六组存档的适配器与基线比较中，对k=1、5、10时的pass@k进行配对检验，并在18个评估端点上进行Holm校正后，发现两个Qwen3-8B适配器在每个k值下均存在功能性回归，其余四组比较结果不确定。在141个经参考认证且契约有效的任务上，三个独立训练的图前缀系统对同一批候选结果进行评分：最佳CodeBLEU与pass@10仅呈中等关联（ρ = .218–.246），compile@10关联较弱（ρ = .072–.082），且仅有21.0–23.3%的可编译候选真正通过测试。一项配对单种子干预实验在保留类型、参数数量和指令内容的同时移除语义名称及相关线索，使任务覆盖率从154个任务中的42个降至7个；而匹配的图结构扰动则未显示出可检测的性能退化。

    arXiv:2607.06125v2 Announce Type: replace-cross  Abstract: We present an execution-based evaluation of neural decompilation for Dart ahead-of-time binaries and an audit of what its scores measure. Across six archived adapter-baseline comparisons, paired tests of pass@k at k = 1, 5, and 10, with Holm adjustment over 18 endpoints, identify functional regressions in both Qwen3-8B adapters at every k. The other four comparisons are inconclusive.   On 141 reference-certified, contract-valid tasks, three independently trained graph-prefix systems score the same candidates. Best CodeBLEU has modest association with pass@10 ($\rho$ = .218-.246), compile@10 has weak association ($\rho$ = .072-.082), and only 21.0-23.3% of compiling candidates pass.   A paired single-seed intervention that removes semantic names and related cues, while retaining types, arity, and instruction content, reduces coverage from 42/154 to 7/154 tasks. Matched graph perturbations show no detectable degradation under the
    
[^38]: 移动应用中编译器优化问题的无源码检测与影响分析

    Source-Free Detection and Impact Analysis of Compiler Optimization Problems in Mobile Applications

    [https://arxiv.org/abs/2606.23512](https://arxiv.org/abs/2606.23512)

    提出了无需源码的OptDetect框架，可直接从应用二进制文件中检测原生库的编译器优化问题，大规模分析发现30.5%的原生库使用低优化级别，影响了91.7%的主流移动应用。

    

    移动应用经常遭受掉帧、过热和过度耗电的困扰。当开发者优化算法和调试代码时，一个关键瓶颈往往被忽视：以低优化级别（O0/O1而非O2/O3）编译的原生库。由于这些库在执行时不会产生功能性错误，由此导致的性能下降在生产环境中发布的应用里依然处于隐蔽状态。我们提出了OptDetect，一个无需源码、可直接从应用二进制文件中检测编译器优化问题的框架。OptDetect通过二进制反汇编、代码块级别分类和加权分数聚合来处理混合优化级别的情况，在受控数据集上达到93.0%的准确率，在真实世界数据集上达到81.9%的准确率。将OptDetect应用于来自830个谷歌Play高排名应用的21,972个原生库，我们发现30.5%的库使用了低优化级别，影响了91.7%的应用。

    arXiv:2606.23512v4 Announce Type: replace  Abstract: Mobile apps frequently suffer from frame drops, overheating, and excessive power consumption. While developers optimize algorithms and debug code, a critical bottleneck often goes unnoticed: native libraries compiled with low optimization levels (O0/O1 instead of O2/O3). Because these libraries execute without functional errors, the resulting performance degradation remains hidden in production apps.   We present \textsc{OptDetect}, a source-free framework that detects compiler optimization problems directly from app binaries. \textsc{OptDetect} handles mixed optimization levels through binary disassembly, chunk-level classification, and weighted score aggregation, achieving 93.0\% accuracy on controlled datasets and 81.9\% on real-world datasets. Applying \textsc{OptDetect} to 21,972 native libraries from 830 top-ranked Google Play apps, we find that 30.5\% of libraries use low optimization levels, affecting 91.7\% of apps.   Throug
    
[^39]: SWE-chat：来自真实用户野外使用的编程智能体交互数据

    SWE-chat: Coding Agent Interactions From Real Users in the Wild

    [https://arxiv.org/abs/2604.20779](https://arxiv.org/abs/2604.20779)

    SWE-chat 是首个从开源开发者真实场景中持续收集的大规模编程智能体会话数据集，揭示出使用模式呈双峰分布——41% 的会话中智能体编写几乎全部代码（“氛围编程”），25% 则完全由人工编写，且智能体在自然环境中效率依然低下。

    

    AI 编程智能体正在被大规模采用，然而我们缺乏关于人们实际如何使用它们以及其产出在实践中有多大用处的实证证据。我们提出了 SWE-chat，这是首个从开源开发者真实使用场景中收集的大规模编程智能体会话数据集。该数据集目前包含近 18,000 个会话，涵盖超过 229,000 条用户提示和 200 万次智能体工具调用。SWE-chat 是一个“活”数据集，我们的收集流程会自动且持续地从公开仓库中发现并处理会话。利用 SWE-chat，我们对真实世界中编程智能体的使用情况和失败模式提供了初步的实证刻画。我们发现编程模式呈双峰分布：在 41% 的会话中，智能体编写了几乎所有提交的代码（即“氛围编程”），而在 25% 的会话中，人类自行编写全部代码。尽管智能体能力快速提升，但在自然使用环境中其效率仍然低下。

    arXiv:2604.20779v2 Announce Type: replace  Abstract: AI coding agents are being adopted at scale, yet we lack empirical evidence on how people actually use them and how much of their output is useful in practice. We present SWE-chat, the first large-scale dataset of real coding agent sessions collected from open-source developers in the wild. The dataset currently contains almost 18,000 sessions, comprising more than 229,000 user prompts and 2 million agent tool calls. SWE-chat is a living dataset; our collection pipeline automatically and continually discovers and processes sessions from public repositories. Leveraging SWE-chat, we provide an initial empirical characterization of real-world coding agent usage and failure modes. We find that coding patterns are bimodal: in 41% of sessions, agents author virtually all committed code ("vibe coding"), while in 25%, humans write all code themselves. Despite rapidly improving capabilities, coding agents remain inefficient in natural setting
    
[^40]: 涌现即代码：可靠自主治理的基础

    Emergence-as-Code as a Foundation for Reliable Self-Governance

    [https://arxiv.org/abs/2602.05458](https://arxiv.org/abs/2602.05458)

    该论文提出“涌现即代码”（EmaC）框架，将局部适应对系统可靠性的系统级影响纳入可执行、证据一致的复合SLO评估，通过持续追踪旅程义务与版本化运行时假设，为系统的可靠自主治理奠定基础，并在PetClinic实验中实现了100%准确的变化检测且无误报。

    

    局部适应可以在保持组件健康的同时，改变系统是否满足其可靠性要求。系统层面的后果取决于受影响的交互及其在用户旅程中所扮演的角色。涌现即代码使这种关系成为可执行的、证据一致的复合SLO（CompositeSLO）评估的一部分。已声明的旅程义务持续存在，而发现机制则维护关于其运行时实现方式的假设。通过推断得到的算子到角色的绑定会选取计算中所使用的度量指标；其证据状态决定了该绑定是否能够支持数值评估。每个结果都保留了义务和版本化的假设，作为治理决策的依据。一项基于PetClinic的概念验证在全部20个随机处理中均准确恢复了算子-状态/边缘绑定的具体变化，且在20个对照组中没有出现任何误判。EmaC与人工动态复合方法在全部40个条件下均产生了互不重叠但语义一致的结果。

    arXiv:2602.05458v3 Announce Type: replace  Abstract: Local adaptations can preserve component health while changing whether a system meets its reliability requirements. The system-level consequence depends on the interaction affected and its role in the user journey. Emergence-as-Code (EmaC) makes this relationship part of an executable, evidence-reconciled CompositeSLO assessment. A declared journey obligation persists while discovery maintains a hypothesis of its runtime realization. An inferred operator-to-role binding selects the measurements used in the calculation; its evidential status determines whether the binding supports a numerical assessment. Each result retains the obligation and versioned hypothesis as a basis for governance decisions. A PetClinic proof of concept recovered the exact operator-state/edge-binding change in all 20 randomized treatments, with no false change in 20 controls. EmaC and a manual dynamic composite matched disjoint semantic outcomes in all 40 cond
    
[^41]: 基于 multipanda_ros2 弥合仿真到现实的差距：面向多机械臂系统的实时 ROS2 框架

    Bridging the Sim-to-Real Gap with multipanda_ros2: A Real-Time ROS2 Framework for Multimanual Systems

    [https://arxiv.org/abs/2602.02269](https://arxiv.org/abs/2602.02269)

    该论文提出了开源 ROS2 框架 multipanda_ros2，通过单进程控制多台 Franka 机器人、维持 1kHz 实时控制频率、实现不超过 2 毫秒的控制器切换延迟，并集成带定量评估指标的高保真 MuJoCo 仿真来弥合仿真到现实的差距。

    

    我们提出了 multipanda_ros2，这是一种用于 Franka Robotics 机器人多机器人控制的新型开源 ROS2 架构。该框架利用 ros2_control，为从单个进程控制任意数量的机器人提供了原生 ROS2 接口。我们的核心贡献解决了实时力矩控制中的关键挑战，包括交互控制和机器人-环境建模。本工作的核心焦点是维持 1kHz 的控制频率，这既是实时控制的必要条件，也是安全标准要求的最低频率。此外，我们引入了一种控制单元（controllet）-特征设计模式，能够实现不超过 2 毫秒的控制器切换延迟，从而促进可复现的基准测试和复杂的多机器人交互场景。为了弥合仿真到现实的差距，我们集成了高保真的 MuJoCo 仿真，并针对运动学精度和动态一致性（力矩、力等）提供定量评估指标

    arXiv:2602.02269v2 Announce Type: replace-cross  Abstract: We present $multipanda\_ros2$, a novel open-source ROS2 architecture for multi-robot control of Franka Robotics robots. Leveraging ros2 control, this framework provides native ROS2 interfaces for controlling any number of robots from a single process. Our core contributions address key challenges in real-time torque control, including interaction control and robot-environment modeling. A central focus of this work is sustaining a 1kHz control frequency, a necessity for real-time control and a minimum frequency required by safety standards. Moreover, we introduce a controllet-feature design pattern that enables controller-switching delays of $\le 2$ ms, facilitating reproducible benchmarking and complex multi-robot interaction scenarios. To bridge the simulation-to-reality (sim2real) gap, we integrate a high-fidelity MuJoCo simulation with quantitative metrics for both kinematic accuracy and dynamic consistency (torques, forces,
    
[^42]: HarnessAgent：利用工具增强的LLM流水线实现可扩展的自动化模糊测试驱动程序构建

    HarnessAgent: Scaling Automatic Fuzzing Harness Construction with Tool-Augmented LLM Pipelines

    [https://arxiv.org/abs/2512.03420](https://arxiv.org/abs/2512.03420)

    HarnessAgent是一个工具增强的智能体框架，通过动态获取规范、依赖和使用示例等丰富上下文信息，克服了现有方法上下文不足及LLM钻验证指标空子的问题，实现了在大型多样化项目上完全自动化、可扩展的功能性模糊测试驱动程序构建。

    

    基于大语言模型（LLM）的技术在为程序模糊测试生成测试驱动程序方面已取得显著进展。然而，将其大规模应用于任意函数（尤其是内部函数）仍然具有挑战性，因为这需要复杂的上下文信息，例如规范说明、依赖关系和使用示例。现有最先进的方法严重依赖静态或不完整的上下文提供方式，导致无法生成功能性的测试驱动程序。此外，LLM往往会钻验证指标的空子，生成看似合理却在逻辑上毫无用处的代码。因此，在大型且多样化的项目中进行测试驱动程序生成，在可靠编译、健壮的代码检索和全面验证等方面仍然面临挑战。为解决这些问题，我们提出了HarnessAgent——一个工具增强的智能体框架，可在数百个项目中实现完全自动化、可扩展的测试驱动程序构建……

    arXiv:2512.03420v4 Announce Type: replace-cross  Abstract: Large language model (LLM)-based techniques have achieved notable progress in generating harnesses for program fuzzing. However, applying them to arbitrary functions (especially internal functions) \textit{at scale} remains challenging due to the requirement of sophisticated contextual information, such as specification, dependencies, and usage examples. State-of-the-art methods heavily rely on static or incomplete context provisioning, causing failure of generating functional harnesses. Furthermore, LLMs tend to exploit harness validation metrics, producing plausible yet logically useless code. % Therefore, harness generation across large and diverse projects continues to face challenges in reliable compilation, robust code retrieval, and comprehensive validation.   To address these challenges, we present HarnessAgent, a tool-augmented agentic framework that achieves fully automated, scalable harness construction over hundreds
    
[^43]: R、Python、Julia 和 C++ 中的高效 SLOPE 求解器

    Efficient Solvers for SLOPE in R, Python, Julia, and C++

    [https://arxiv.org/abs/2511.02430](https://arxiv.org/abs/2511.02430)

    该论文提出了 R、Python、Julia 和 C++ 中高效求解 SLOPE 问题的软件包套件，采用高效的混合坐标下降算法支持多种损失函数和数据结构，并在速度上超越了现有的 SLOPE 实现。

    

    我们提出了一套在 R、Python、Julia 和 C++ 中的软件包，用于高效求解排序 L1 正则化估计问题。这些软件包采用高效的混合坐标下降算法，可以拟合广义线性模型（GLM），并支持多种损失函数，包括高斯损失、二项损失、泊松损失和多项逻辑回归损失。我们的实现设计追求快速、节省内存且灵活。这些软件包支持多种数据结构（稠密矩阵、稀疏矩阵和内存外矩阵），能够高效地拟合完整的 SLOPE 路径，并处理 SLOPE 模型的交叉验证，包括松弛 SLOPE（relaxed SLOPE）。我们展示了如何使用这些软件包的示例，并通过基准测试证明了这些软件包在真实数据和模拟数据上的性能，结果表明我们的软件包在速度方面优于现有的 SLOPE 实现。

    arXiv:2511.02430v4 Announce Type: replace-cross  Abstract: We present a suite of packages in R, Python, Julia, and C++ that efficiently solve the Sorted L-One Penalized Estimation (SLOPE) problem. The packages feature a highly efficient hybrid coordinate descent algorithm that fits generalized linear models (GLMs) and supports a variety of loss functions, including Gaussian, binomial, Poisson, and multinomial logistic regression. Our implementation is designed to be fast, memory-efficient, and flexible. The packages support a variety of data structures (dense, sparse, and out-of-memory matrices) and are designed to efficiently fit the full SLOPE path as well as handle cross-validation of SLOPE models, including the relaxed SLOPE. We present examples of how to use the packages and benchmarks that demonstrate the performance of the packages on both real and simulated data and show that our packages outperform existing implementations of SLOPE in terms of speed.
    
[^44]: 漏洞赏金计划中的激励与成果

    Incentives and Outcomes in Bug Bounties

    [https://arxiv.org/abs/2509.16655](https://arxiv.org/abs/2509.16655)

    该研究利用谷歌漏洞奖励计划2024年7月奖励金额最高上调200%这一自然实验，实证发现提高赏金显著增加了高价值漏洞的提交数量，漏洞研究人员的劳动供给弹性很高，且奖励提升既重新激发了资深研究人员也吸引了新人参与。

    

    过去十年间，漏洞赏金计划为科技公司的安全做出了重大贡献，但人们对其奖励激励在产生有效成果方面所起的作用知之甚少。我们分析了谷歌漏洞奖励计划（VRP）中的激励与成果，该计划是全球最大的漏洞赏金计划之一。我们研究了所收到漏洞的质量和数量对报酬变化的响应性，重点关注谷歌于2024年7月公布的奖励金额调整，其中最高影响等级的奖励金额提高了最多200%。我们的实证结果表明，奖励上调后收到的高价值漏洞数量有所增加，且此类漏洞的劳动供给表现出很高的正弹性。我们进一步将这一增长的来源分解为资深研究人员与新研究人员，发现奖励上调既重新引导了资深研究人员的注意力，又吸引了新研究人员参与。

    arXiv:2509.16655v2 Announce Type: replace  Abstract: Bug bounty programs have contributed significantly to security in technology firms in the last decade, but little is known about the role of reward incentives in producing useful outcomes. We analyze incentives and outcomes in Google's Vulnerability Rewards Program (VRP), one of the world's largest bug bounty programs. We analyze the responsiveness of the quality and quantity of bugs received to changes in payments, focusing on a change in Google's reward amounts posted in July, 2024, in which reward amounts increased by up to 200% for the highest impact tier. Our empirical results show an increase in the volume of high-value bugs received after the reward increase, as well as a high positive observed elasticity of labor supply for such bugs. We further break down the sources of this increase between veteran researchers and new researchers, showing that the reward increase both redirected the attention of veteran researchers and attr
    
[^45]: 论成功之幻象：工业持续集成中作业重跑与静默失败的实证研究

    On the Illusion of Success: An Empirical Study of Job Reruns and Silent Failures in Industrial CI

    [https://arxiv.org/abs/2509.14347](https://arxiv.org/abs/2509.14347)

    本文首次通过重跑成功作业的实践，对工业持续集成中的“静默失败”（即作业被标记为成功却未完成全部或部分任务）进行了实证研究，分析了81个工业项目中的142,387个作业，揭示了这种制造成功假象并可能让缺陷逃逸至生产环境的失败现象。

    

    构建结果的可靠性是有效持续集成（CI）的基石。然而在实践中，开发人员经常受到代码或CI基础设施中非确定性问题的困扰，这些问题削弱了对构建结果的信任。当面对这类意外结果时，开发人员通常会反复重跑作业以期获得真正的成功，但众所周知，这种做法会增加CI成本并降低生产力。虽然近期的研究主要集中在间歇性作业失败上，但此前尚无研究调查“静默失败”——即构建作业被标记为成功，却未能完成全部或部分任务的情况。这类静默失败往往不被察觉，制造出成功的假象，并带来诸如缺陷逃逸到生产环境等有害后果。本文通过重跑成功作业的实践，首次对静默失败进行了实证研究。对81个工业项目中142,387个作业的分析表明，11

    arXiv:2509.14347v2 Announce Type: replace  Abstract: Reliability of build outcomes is a cornerstone of effective Continuous Integration (CI). Yet in practice, developers often struggle with non-deterministic issues in the code or CI infrastructure, which undermine trust in build results. When faced with such unexpected outcomes, developers often repeatedly rerun jobs hoping for true success, but this practice is known to increase CI costs and reduce productivity. While recent studies have focused on intermittent job failures, no prior work has investigated silent failures, where build jobs are marked as successful but fail to complete all or part of their tasks. Such silent failures often go unnoticed, creating an illusion of success with detrimental consequences such as bugs escaping into production. This paper presents the first empirical study of silent failures through the practice of rerunning successful jobs. An analysis of 142,387 jobs across 81 industrial projects shows that 11
    
[^46]: AI 编码者是泄密者吗？关于代码大语言模型预训练数据检测的实证研究

    Are AI Coders Snitches? An Empirical Study of Pretraining Data Detection on Code Large Language Models

    [https://arxiv.org/abs/2507.17389](https://arxiv.org/abs/2507.17389)

    该论文对七种最先进的训练数据检测方法在源代码数据上的有效性进行了全面的实证研究，评估了它们在八个代码大语言模型上的表现，填补了训练数据检测方法在代码领域研究的空白。

    

    代码大语言模型（CodeLLMs）的最新进展使其成为现代软件工程中不可或缺的工具。然而，这些模型有时会产生包含专有或敏感代码片段的输出，这引发了对训练数据潜在不合规使用的担忧，并对隐私和知识产权构成风险。为了确保 CodeLLMs 的负责任和合规部署，训练数据检测（TDD）已成为一项关键任务。尽管近期的 TDD 方法在自然语言场景中已展现出良好的前景，但它们在代码数据上的有效性在很大程度上仍未被充分探索。鉴于代码具有结构化的语法以及与自然语言截然不同的相似性判断标准，这一研究空白尤为重要。为了解决这一问题，我们对七种最先进的 TDD 方法在源代码数据上进行了全面的实证研究，评估了它们在八个 CodeLLMs 上的表现。为支持这一评估，我们……

    arXiv:2507.17389v2 Announce Type: replace-cross  Abstract: Recent advances in code large language models (CodeLLMs) have made them indispensable tools in modern software engineering. However, these models occasionally produce outputs that contain proprietary or sensitive code snippets, raising concerns about potential non-compliant use of training data, and posing risks to privacy and intellectual property. To ensure responsible and compliant deployment of CodeLLMs, training data detection (TDD) has become a critical task. While recent TDD methods have shown promise in natural language settings, their effectiveness on code data remains largely underexplored. This gap is particularly important given code's structured syntax and distinct similarity criteria compared to natural language. To address this, we conduct a comprehensive empirical study of seven state-of-the-art TDD methods on source code data, evaluating their performance across eight CodeLLMs. To support this evaluation, we in
    

