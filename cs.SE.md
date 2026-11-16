# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From AI-Enabled Remote Patient Monitoring Toward Agentic Robotics: Embodiment, Governance, and Care Integration](https://arxiv.org/abs/2611.18484) | 本文通过结构化范围综述指出，AI远程患者监测正迈向具身智能体机器人护理，但成熟的智能体护理尚未实现，未来需要以具身化、人类监督、治理和安全可互操作的护理架构为基础，实现从监测到行动的转变。 |
| [^2] | [Evaluating Feedback-Driven Black-Box Processes via Stochastic Modeling and Simulation](https://arxiv.org/abs/2611.18477) | 本文提出了一个基于马尔可夫链和仿真的随机建模框架，用于统一评估反馈驱动的黑盒过程，并通过新颖的模糊测试马尔可夫链模型加以实例化，从而将实证实践与理论联系起来。 |
| [^3] | [HerAgent: A Block-Level Checkpointing Harness for Coding-Agent Environment Setup](https://arxiv.org/abs/2611.18443) | HerAgent提出“语义化搭建块”这一块级进度控制抽象，将执行、验证、检查点保存、修复与回滚统一起来，在四个环境搭建数据集上全面超越六个先前智能体，成功率最高提升33个百分点且搭建成本降低高达94.5%。 |
| [^4] | [Between Obligations and Evidence: A Participatory Study of Translating the EU AI Act into Technical Requirements](https://arxiv.org/abs/2611.18439) | 该研究通过32名跨学科专家参与的参与式研讨会，揭示了将欧盟《人工智能法案》义务转化为技术要求时在可测量性、责任归属与人工监督实践、以及合规证据在动态运行条件下的有效性方面反复出现的张力。 |
| [^5] | [PrivacyPR: A Dataset of Real-World Privacy Fixes for Benchmarking Privacy-Aware Code Review](https://arxiv.org/abs/2611.18293) | 该论文提出了PrivacyPR数据集，包含4,131个带LLM生成的隐私相关性和GDPR原则标注的真实GitHub拉取请求，为评估编码智能体能否在代码审查中发现法律要求的隐私缺陷提供了首个大规模基准。 |
| [^6] | [PerfArena: Ranking LLMs by Code Performance](https://arxiv.org/abs/2611.18030) | PERFARENA是一个开源、可复现的评测竞技场，通过测量LLM生成代码在多种语言和竞赛中的延迟、峰值内存与能耗等性能指标来对模型排名，弥补了现有基准仅关注正确性、语言覆盖窄且忽略能耗的不足。 |
| [^7] | [Soteria: Formally Verified Planning with Runtime Enforcement for Safe LLM Agents](https://arxiv.org/abs/2611.17880) | Soteria 通过在工具调用前对分层计划进行形式化验证并在运行时强制执行规范，使 LLM 智能体在不牺牲任务效用的情况下保证安全性，并在不安全动作被阻止时借助已验证计划提供有原则的恢复指导。 |
| [^8] | [DesignSpecs: A DESIGN.md Dataset of Visual Design Specifications for AI Agents](https://arxiv.org/abs/2611.17713) | 该论文提出了首个大规模DESIGN.md视觉设计规范数据集DesignSpecs，包含来自15,794个公开GitHub仓库、经严格筛选的36,906个设计规范文件，为AI智能体在生成和修改用户界面时提供视觉设计规范的数据基础。 |
| [^9] | [OCTAVE: A Capability-Resolved Benchmark for End-to-End HTML5 Game Generation](https://arxiv.org/abs/2611.17681) | 该论文提出OCTAVE基准，通过八个能力维度、180个任务和1,800条评估需求对端到端HTML5游戏生成进行需求级能力诊断，并揭示当前最优模型组合的完整性与丰富度仅约33%。 |
| [^10] | [Meta Build Systems as Logic Programs](https://arxiv.org/abs/2611.17672) | 该论文提出将构建配置表达为关系推理，并介绍了 buildlog 这一元构建系统，其分层逻辑核心将构建推导与环境探测隔离，拒绝工具链和软件包选择中的歧义，并通过溯源解释推导结果。 |
| [^11] | [SZZBench: Controlled Git Histories with Explicit Oracles for Diagnosing SZZ Implementations](https://arxiv.org/abs/2611.17669) | 论文提出SZZBench，一个通过声明式场景生成带有显式预言的受控Git仓库、并借助26个朴素基线诊断SZZ实现成败原因的评测工具，目前包含17个类别共279个场景及四个SZZ工具的适配器。 |
| [^12] | [Choosing Quantized Code LLMs by File Size and Energy per Solved Task](https://arxiv.org/abs/2611.17605) | 本研究在 MacBook Air M2 上以“每解决任务的能耗”评测 Qwen2.5-Coder 的 24 个量化版本，给出按文件大小索引、附证据标签的模型选择指南，并揭示 3 GB 档位的 7B-q2_K 相比 2 GB 档位的 3B-q4_K_M 无显著性能提升却耗能翻倍。 |
| [^13] | [Beyond Source Code: Evaluating Natural Language Specifications for LLM-Based C-to-Rust Translation](https://arxiv.org/abs/2611.17581) | 该论文首次系统评估了自然语言规范作为输入在LLM驱动的C到Rust翻译中的作用，发现仅用源代码翻译虽测试通过率最高（98.6%）但往往照搬C的实现模式，而引入自然语言规范有望生成更地道、更少依赖unsafe结构的Rust代码。 |
| [^14] | [GraphScent: Calibrating Directional Relevance Signals for Bounded-Context Code Localization](https://arxiv.org/abs/2611.17527) | 该研究发现冻结语言模型对外围作用域和上游调用者的零温度相关性评分能可靠区分真正的结构近邻与词汇干扰项，而其他关系类型的评分接近随机水平，据此校准方向性相关性信号以实现单张消费级GPU上的无状态有界上下文代码定位。 |
| [^15] | [Interpreting Coding Agents Through Their Behavioral Trajectories](https://arxiv.org/abs/2611.17440) | 该论文提出一种基于规则的解析器，将编码智能体的原始执行轨迹映射为高级行为，并通过对14万余条轨迹的行为画像分析，揭示解决率相近的智能体在探索、修改和验证代码的方式上存在显著差异，且这些行为画像本身足以在不同仓库与语言间识别出对应的智能体技术栈。 |
| [^16] | [Time-Consistent Evaluation of Hierarchical Code Graphs for SWE Agents](https://arxiv.org/abs/2611.17388) | 该论文构建了一个零泄漏的时间一致性评估基准，在DragonFly和React的历史截止快照上，以四种提示粒度对比评估基线、扁平AST图与分层依赖图ServiceExplorer三种上下文增强方案对SWE智能体解决100个后续拉取请求的效果。 |
| [^17] | [ProdPilot: A Layered Model Context Protocol System for Automated Production Readiness and Deployment of Vibe-Coded Projects](https://arxiv.org/abs/2611.17381) | ProdPilot是一个本地MCP服务器，其核心创新在于将50条生产就绪规则中的88%实现为固定模板或基于项目信息提取的确定性修复，而非依赖不可控的模型自由生成，从而对氛围编码项目实现可验证的自动化审计、修复、门控与部署。 |
| [^18] | [Everything is an Operational View: UI as a Projection over Context in Agentic Systems](https://arxiv.org/abs/2611.17355) | 本文提出AFS-UI框架，将用户界面重新概念化为智能体文件系统之上的上下文投影，使界面成为对共享资源的动态可检查视图，从而通过统一操作界面同时支持人类与智能体的交互。 |
| [^19] | [TSBugs Explorer: Exploring and Benchmarking TypeScript Bugs](https://arxiv.org/abs/2611.14940) | 本文提出了TSBugs Explorer，一个基于633个真实TypeScript缺陷数据集构建的交互式可视化分析平台，支持缺陷探索、跨仓库故障特征比较以及面向研究的基准子集构建。 |
| [^20] | [Caprese: Change Impact Recommendation Using History and Runtime Analysis](https://arxiv.org/abs/2611.14937) | Caprese是一个面向Node.js的开源函数级变更影响推荐框架，它结合历史共变更挖掘与运行时依赖分析，生成可能受代码变更影响的函数排序列表。 |
| [^21] | [Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models](https://arxiv.org/abs/2611.13223) | 本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。 |
| [^22] | [PRISM: Preference-based Reward and Uncertainty-Integrated Synthesis Minimization for RTL PPA Optimization](https://arxiv.org/abs/2611.11051) | PRISM是一个预算感知的多目标RTL优化框架，它通过基于偏好的奖励模型与不确定性引导的采集策略，在大幅减少昂贵商业综合工具调用次数的同时，有效优化设计的功耗、性能和面积（PPA）。 |
| [^23] | [SoK: Post-Quantum Cryptography (PQC) Integration in Software Development: Approaches, Gaps, and Future Directions](https://arxiv.org/abs/2611.08442) | 本文系统梳理了软件开发中后量子密码学集成的研究现状，指出尽管现有工作奠定了重要的技术与组织基础，但各阶段（准备、实现、迁移、部署、维护及开发者实践）之间缺乏协调，生命周期支持仍是割裂的。 |
| [^24] | [Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations](https://arxiv.org/abs/2611.08269) | 预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。 |
| [^25] | [Operationalizing Governed Agentic Workflows in Regulated Finance: A Customer Zero Case Study using Zafin AIOS](https://arxiv.org/abs/2610.33253) | 本文提出Zafin AIOS这一端到端编排与控制平台，并通过“零号客户”内部生产案例研究，验证了治理型智能体工作流在受监管金融行业中从试点走向生产落地的可行性与有效性。 |
| [^26] | [CAMGen: A Core Change-Aware Multi-View Framework for Commit Message Generation](https://arxiv.org/abs/2610.31552) | CAMGen通过识别代表主要变更意图的核心实体，并借助仓库级语义聚类构建功能语义视图，将多代码实体协作实现的功能上下文融入提交信息生成，从而生成更能准确解释变更及其影响的提交信息。 |
| [^27] | [Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3](https://arxiv.org/abs/2610.30499) | 该论文通过受控实验首次系统考察前沿大语言模型能否自主构建并迭代改进自己的执行框架，让模型在ARC-AGI-3交互式基准上独立编写、使用并两次修订完整框架代码库，再于二十个从未参与改进的留出游戏上检验其泛化能力。 |
| [^28] | [Cross-Benchmark Transfer from RL on Agentic Coding Tasks](https://arxiv.org/abs/2610.00890) | 仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。 |
| [^29] | [Reproducibility in the Age of Agentic AI: Context Engineering at the Timescale of a Codebase](https://arxiv.org/abs/2609.11728) | 本文提出可复现研究实践本质上就是为AI编码智能体进行的上下文工程，智能体降低了维护测试、提交历史和决策记录等科研工件的成本并使收益即时化，但研究人员仍需承担验证这些工件及其科学判断的责任。 |
| [^30] | [From Reading Code to Reading Spec: A Verified Layer for LLM-Driven Codebase Maintenance](https://arxiv.org/abs/2609.06383) | 论文提出PROOF系统，将代码库拓扑抽象为分层自然语言规约，并通过从规约重构源代码来证明语义等价，使LLM能够借助规约间接、可靠地执行代码库维护并防止语义漂移。 |
| [^31] | [A Governance Methodology Layer for AI-Assisted Software Development: Defect Taxonomy, Controlled Ablation, and Process-Over-Capability Evidence](https://arxiv.org/abs/2609.04218) | 本文提出一套AI辅助软件开发的治理方法论层，通过缺陷分类体系、跨工具可移植的运行时解耦治理门以及“方法论即代码”的形式化，并以受控消融实验证明过程治理比模型能力更重要。 |
| [^32] | [Drive the Thoughts: Runtime Monitoring of VLA Reasoning-Trajectory Consistency](https://arxiv.org/abs/2608.29583) | 本文提出DriveAlignBench数据集，通过运行时监控VLA模型思维链推理与生成轨迹之间的一致性，来检测自动驾驶系统中潜在的不安全行为。 |
| [^33] | [A Jagged Frontier: Evaluating Robustness of Code Agents to Semantics-Preserving Transformations](https://arxiv.org/abs/2608.18389) | 本文通过引入随机语义保持变换采样器，系统评估了代码代理在代码库语义等价改写下的鲁棒性，发现其修复性能会显著下降，揭示了现有代理的脆弱性。 |
| [^34] | [REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming](https://arxiv.org/abs/2607.07738) | 本文提出REFORGE——一个带溯源追踪的评测流水线，通过八门置信度漏斗将编译器优化下的二进制-源码对齐不确定性量化分级，从而解决了LLM逆向工程能力公平评估中函数级真值可靠性不足这一核心障碍。 |
| [^35] | [Agent Security Meets Regulatory Reality -- A Practitioner Systematization of Autonomous-Agent Threats and Controls in Regulated Financial Systems](https://arxiv.org/abs/2606.29142) | 该论文基于生产环境证据，将六类自主智能体威胁映射到美欧金融监管的具体控制义务，提炼出五个运行时控制不变量及其形式化模型与可审计执行机制，弥合了智能体安全研究与金融监管合规实践之间的鸿沟。 |
| [^36] | [Hybrid Mechanical and Agentic Specification Inference](https://arxiv.org/abs/2605.10005) | 该论文将机械式最弱前置条件（WP）分析与LLM智能体驱动的规约推断相结合，集成到 Move Prover 中自动为 Move 智能合约推导规约，在 Aptos 框架的无规约公开函数上实现了超过 90% 的任务完成率。 |
| [^37] | [From Conversation to Evaluation: Benchmarking LLMs on Development Knowledge via SimpleDevQA](https://arxiv.org/abs/2512.08867) | 本文通过分析真实用户与大语言模型的对话并设计三阶段转换流水线，构建了源自真实对话的多语言开发知识问答基准 SimpleDevQA，弥补了现有基准知识范围有限且缺乏真实用户提问的不足。 |
| [^38] | [A GNN-Based Approach for Code Smell Detection and Refactoring Opportunity Identification](https://arxiv.org/abs/2511.12069) | 该论文提出了一种基于图神经网络的深度学习方法，通过在类级别和方法级别构建代码图来检测代码坏味并识别重构机会，从而克服了传统方法依赖人工规则、机器学习方法训练数据受限以及大语言模型计算成本过高的局限性。 |
| [^39] | [BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution](https://arxiv.org/abs/2510.08697) | 该论文提出了 BigCodeArena，一个基于实时执行环境的开放式代码生成人类评估平台，通过让人类与 LLM 生成代码的执行过程和结果进行交互，从而揭示出更可靠的人类偏好。 |
| [^40] | [Training Language Model Agents to Find Vulnerabilities with CTF-Dojo](https://arxiv.org/abs/2508.18370) | 提出了 CTF-Dojo——首个面向 LLM 训练的大规模可验证执行环境，包含 658 个容器化 CTF 挑战，并配套自动化流水线 CTF-Forge，可数分钟内将公开工件转化为即用型执行环境，仅需 486 条高质量轨迹即可训练出擅长漏洞发现的智能体。 |
| [^41] | [Identifying and Mitigating API Misuse in Large Language Models](https://arxiv.org/abs/2503.22821) | 该论文对三种主流大语言模型生成代码中的API误用模式进行了大规模研究，通过人工标注识别出四类常见误用类型，并针对幻觉和意图错位等问题提出了缓解方案。 |
| [^42] | [A Taxonomy of Real Faults in Hybrid Quantum-Classical Architectures](https://arxiv.org/abs/2502.08739) | 本文通过实证分析GitHub上超过5000个已关闭issue，提出并验证了首个混合量子-经典（NISQ）架构中真实故障的分类体系，并发布了包含133个标注真实故障的数据集。 |
| [^43] | [BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](https://arxiv.org/abs/2406.15877) | BigCodeBench是一个新的代码生成基准测试，通过要求模型调用多样化函数库并准确理解复杂指令，来评估大语言模型解决实际任务的能力。 |
| [^44] | [Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2401.00788) | 该论文提出了Astraios套件——包含28个采用7种微调方法、4种规模（最大160亿参数）指令微调的代码大语言模型，系统研究发现全参数微调总体性能最佳，LoRA通常提供最有利的成本-性能权衡，且PEFT方法的效果随模型规模显著变化。 |

# 详细

[^1]: 从AI赋能的远程患者监测迈向智能体机器人：具身化、治理与护理整合

    From AI-Enabled Remote Patient Monitoring Toward Agentic Robotics: Embodiment, Governance, and Care Integration

    [https://arxiv.org/abs/2611.18484](https://arxiv.org/abs/2611.18484)

    本文通过结构化范围综述指出，AI远程患者监测正迈向具身智能体机器人护理，但成熟的智能体护理尚未实现，未来需要以具身化、人类监督、治理和安全可互操作的护理架构为基础，实现从监测到行动的转变。

    

    AI赋能的远程患者监测正在从被动的数据收集演进为具有能动性的AI系统，这类系统能够解读患者状态、协调护理，并在家庭和临床环境中发起行动。这项结构化的范围综述发现，成熟的智能体护理尚未实现：大多数系统仍由人类主导，而更强的决策-行动耦合和具身AI在很大程度上仍局限于原型与试点阶段。医疗机器人通过将AI生成的建议和决策转化为语音、动作、提示、物理辅助、升级上报和护理协调，使从监测到行动的转变尤为直观，这也使具身化、临床整合、人类监督和治理成为基本的设计要求。进一步的进展需要安全且可互操作的护理架构，使智能组件能够在经过临床验证且可问责的工作流程中运行。我们提出了一种……（原文摘要在此处截断）

    arXiv:2611.18484v1 Announce Type: cross  Abstract: AI-enabled remote patient monitoring is evolving beyond passive data collection toward agentic AI systems that interpret patient states, coordinate care, and initiate action across domestic and clinical settings. This structured scoping review finds that mature agentic care has not yet emerged: Most systems remain human led, while stronger decision-action coupling and embodied AI remain largely confined to prototypes and pilots. Healthcare robotics makes the transition from monitoring to action especially visible by translating AI-generated recommendations and decisions into speech, movement, prompting, physical assistance, escalation, and care coordination, making embodiment, clinical integration, human oversight, and governance fundamental design requirements. Further progress requires safe and interoperable care architectures in which intelligent components operate through clinically validated and accountable workflows. We propose a
    
[^2]: 基于随机建模与仿真评估反馈驱动的黑盒过程

    Evaluating Feedback-Driven Black-Box Processes via Stochastic Modeling and Simulation

    [https://arxiv.org/abs/2611.18477](https://arxiv.org/abs/2611.18477)

    本文提出了一个基于马尔可夫链和仿真的随机建模框架，用于统一评估反馈驱动的黑盒过程，并通过新颖的模糊测试马尔可夫链模型加以实例化，从而将实证实践与理论联系起来。

    

    我们研究软件工程与机器学习领域中反馈驱动的黑盒过程。这些过程本质上是顺序进行的：策略生成输入、观察系统的输出/反馈，并利用反馈来指导后续输入的设计。虽然许多策略是在对底层系统做出结构假设的前提下开发的，但许多现实世界的系统在缺乏此类知识的情况下运行，因此需要黑盒策略，而这类策略目前缺乏统一的评估框架。我们引入了一个随机框架，利用马尔可夫链和仿真来建模与评估此类策略，从而将实证实践与理论联系起来。我们通过一种新颖的模糊测试马尔可夫链模型，将该框架实例化于基于变异的模糊测试（一种广泛用于通过顺序输入变异发现软件缺陷的技术）中，该模型的转移由变异策略控制，其反馈由程序执行给出。我们开发了一个严格的……（摘要原文在此处截断）

    arXiv:2611.18477v1 Announce Type: cross  Abstract: We study feedback-driven black-box processes across software engineering and machine learning. These processes are inherently sequential: policies generate inputs, observe outputs/feedback from systems, and use the feedback to guide the design of subsequent inputs. While many policies are developed under structural assumptions on the underlying systems, many real-world systems operate without such knowledge, necessitating black-box policies that lack a unified evaluation framework. We introduce a stochastic framework that models and evaluates such policies using Markov chains and simulation, thereby connecting empirical practice with theory. We instantiate the framework in mutation-based fuzzing (widely used for discovering software bugs via sequential input mutation), by a novel Markov chain model of fuzzing whose transitions are governed by the mutation strategy and whose feedback is given by program execution. We develop a rigorous 
    
[^3]: HerAgent：面向编程智能体环境搭建的块级检查点框架

    HerAgent: A Block-Level Checkpointing Harness for Coding-Agent Environment Setup

    [https://arxiv.org/abs/2611.18443](https://arxiv.org/abs/2611.18443)

    HerAgent提出“语义化搭建块”这一块级进度控制抽象，将执行、验证、检查点保存、修复与回滚统一起来，在四个环境搭建数据集上全面超越六个先前智能体，成功率最高提升33个百分点且搭建成本降低高达94.5%。

    

    现代编程智能体在执行测试、调试和补丁验证等软件工程任务之前，需要先建立可执行的软件环境。现有的环境搭建智能体通常以单命令或整个脚本的粒度来控制进度，这使得恢复操作要么过于局部而无法清除污染状态，要么过于粗糙而无法保留有用的进展。我们提出了HerAgent（Harness EnviRonment Setup Agent），这是一个面向编程智能体环境搭建的块级框架。HerAgent的主要贡献是语义化搭建块：这是一种统一的进度控制抽象，用于执行、验证、检查点保存、修复和回滚。在四个环境搭建数据集上与六个先前智能体进行对比评估，HerAgent在所有对比中均提升了搭建成功率，在Installamatic-Bench（从52.0%提升至85.0%）和SetupBench（从46.3%提升至70.4%）上取得了最大幅度的提升。它还将搭建成本降低高达94.5%，表明块级……

    arXiv:2611.18443v1 Announce Type: new  Abstract: Modern coding agents rely on establishing executable software environments before they can perform software engineering tasks such as testing, debugging, and patch validation. Existing setup agents typically control progress at either single-command or whole-script granularity, making recovery either too local to remove polluted state or too coarse to preserve useful progress. We present HerAgent (Harness EnviRonment Setup Agent), a block-level harness for coding-agent environment setup. HerAgent's main contribution is semantic setup blocks: a single progress-control abstraction used for execution, validation, checkpointing, repair, and rollback. Evaluated against six prior agents on four environment-setup datasets, HerAgent improves setup success in every comparison, achieving the largest gains on Installamatic-Bench (52.0% to 85.0%) and SetupBench (46.3% to 70.4%). It also reduces setup costs by up to 94.5%, showing that block-level pr
    
[^4]: 在义务与证据之间：将欧盟《人工智能法案》转化为技术要求的参与式研究

    Between Obligations and Evidence: A Participatory Study of Translating the EU AI Act into Technical Requirements

    [https://arxiv.org/abs/2611.18439](https://arxiv.org/abs/2611.18439)

    该研究通过32名跨学科专家参与的参与式研讨会，揭示了将欧盟《人工智能法案》义务转化为技术要求时在可测量性、责任归属与人工监督实践、以及合规证据在动态运行条件下的有效性方面反复出现的张力。

    

    欧盟《人工智能法案》对高风险人工智能系统提出了技术和组织方面的要求，但应用这些要求涉及情境化的选择——即应该测量什么，以及什么证据能够支持合规性声明。我们通过一场由来自不同学科的32名研究人员、政策制定者和从业者参与的参与式研讨会来研究这些挑战，并采用医疗健康、招聘和公共交通领域的现实场景，考察监管义务如何被解读、转化为技术要求，并在实践中得到评估。通过对研讨会数据的主题分析，我们识别出反复出现的张力：监管义务如何转化为可测量的系统属性、责任与人工监督在实践中如何行使，以及合规证据的有效性如何受到运行条件变化和系统用途改变的影响。例如，参与者强调了公平……（原文摘要在此处截断）

    arXiv:2611.18439v1 Announce Type: cross  Abstract: The EU AI Act imposes technical and organizational requirements for high-risk AI systems, but applying these requirements involves contextual choices about what to measure and what evidence supports a compliance claim. We investigate these challenges through a participatory workshop involving 32 researchers, policymakers, and practitioners from different disciplines, using realistic scenarios in healthcare, recruitment, and public transport to examine how regulatory obligations are interpreted, translated into technical requirements, and evaluated in practice. Through a thematic analysis of workshop data, we identify recurring tensions between how regulatory obligations are translated into measurable system properties, how responsibility and human oversight are exercised in practice, and how the validity of compliance evidence is affected by changing operational conditions and system uses. For example, participants highlighted how fair
    
[^5]: PrivacyPR：一个用于隐私感知代码审查基准测试的真实世界隐私修复数据集

    PrivacyPR: A Dataset of Real-World Privacy Fixes for Benchmarking Privacy-Aware Code Review

    [https://arxiv.org/abs/2611.18293](https://arxiv.org/abs/2611.18293)

    该论文提出了PrivacyPR数据集，包含4,131个带LLM生成的隐私相关性和GDPR原则标注的真实GitHub拉取请求，为评估编码智能体能否在代码审查中发现法律要求的隐私缺陷提供了首个大规模基准。

    

    编码智能体越来越多地参与代码审查，但它们能否发现GDPR等法律要求软件避免的隐私缺陷，这在很大程度上仍未得到检验，部分原因在于很少有数据集将真实的隐私缺陷与其被接受的修复方案及审查上下文联系起来。我们提出了PrivacyPR，这是一个包含来自1,469个仓库的4,131个已合并GitHub拉取请求（PR）的数据集，创建时间为2018年6月至2024年1月。每个PR都包含其提交记录、代码差异、审查讨论，以及由大语言模型生成的隐私相关性、修复类型和GDPR原则标签。PrivacyPR提供了1,169个高置信度隐私修复和1,643个非修复PR，还包括Verified（一个经人工核验的子集，包含250个隐私修复和250个非修复PR）以及Small（一个100+100的评估子集）。在Verified子集上，两名标注者均认可了LLM所做GDPR原则标注中的72%。除了支持对隐私缺陷如何产生和修复的实证研究之外，PrivacyPR还提供了一个基准测试平台。

    arXiv:2611.18293v1 Announce Type: cross  Abstract: Coding agents increasingly review code, yet whether they catch the privacy defects that laws such as the GDPR require software to avoid remains largely untested, partly because few datasets connect real privacy defects to their accepted fixes and review context. We present PrivacyPR, a dataset of 4,131 merged GitHub pull requests (PRs) from 1,469 repositories, created between June 2018 and January 2024. Each PR includes its commits, diffs, review discussion, and LLM-generated labels for privacy relevance, fix type, and GDPR principles. PrivacyPR provides 1,169 high-confidence privacy fixes and 1,643 non-fix PRs, together with Verified, a manually checked split of 250 privacy fixes and 250 non-fix PRs, and Small, a 100 + 100 subset for evaluation. On Verified, both annotators endorsed 72% of the LLM's GDPR-principle assignments. Beyond supporting empirical study of how privacy defects arise and are fixed, PrivacyPR provides a benchmark 
    
[^6]: PerfArena：按代码性能对大语言模型进行排名

    PerfArena: Ranking LLMs by Code Performance

    [https://arxiv.org/abs/2611.18030](https://arxiv.org/abs/2611.18030)

    PERFARENA是一个开源、可复现的评测竞技场，通过测量LLM生成代码在多种语言和竞赛中的延迟、峰值内存与能耗等性能指标来对模型排名，弥补了现有基准仅关注正确性、语言覆盖窄且忽略能耗的不足。

    

    当前业界已依赖大语言模型（LLM）编写大量生产代码。尽管这些代码大部分情况下是正确的，但并非总是如此，而且仅有正确性并不能决定代码是否适合被反复和广泛使用，这还取决于代码在时间、内存和能耗上的开销。然而，大多数代码生成基准只单独评估正确性，而大多数面向性能的基准覆盖的语言较少，且很少测量能耗。为填补这一空白，我们提出了PERFARENA，一个开源、可复现的竞技场，它根据LLM所生成代码的性能在多种语言和竞赛中对模型进行排名。PERFARENA在类Unix主机上执行每个LLM的解答，并报告正确性以及延迟、峰值内存和能耗。在PERFARENA中，正确性判定器、性能判定器、性能分析器和代码生成工作流均是可插拔的，从而能够集成新的评判后端。

    arXiv:2611.18030v1 Announce Type: cross  Abstract: Industry now relies on LLMs to write a large share of production code. Although that code is mostly, but not always, correct, correctness alone does not determine whether it is fit for repeated and extensive use, which depends on what the code costs in time, memory and energy. Most code-generation benchmarks nonetheless evaluate correctness in isolation, and most performance-oriented benchmarks cover few languages and rarely measure energy consumption. To address this gap, we present PERFARENA, an open-source, reproducible arena that ranks LLMs by the performance of their generated code across multiple languages and competitions. PERFARENA executes the solutions of each LLM on a Unix-like host and reports correctness together with latency, peak memory, and energy consumption. Correctness oracles, performance oracles, profilers, and code-generation workflows are pluggable in PERFARENA, enabling the integration of new judging backends an
    
[^7]: Soteria：面向安全 LLM 智能体的形式化验证规划与运行时强制执行

    Soteria: Formally Verified Planning with Runtime Enforcement for Safe LLM Agents

    [https://arxiv.org/abs/2611.17880](https://arxiv.org/abs/2611.17880)

    Soteria 通过在工具调用前对分层计划进行形式化验证并在运行时强制执行规范，使 LLM 智能体在不牺牲任务效用的情况下保证安全性，并在不安全动作被阻止时借助已验证计划提供有原则的恢复指导。

    

    执行多步骤工具调用的 LLM 智能体必须同时满足两个目标：完成用户的任务（效用）和遵守领域策略与正确性约束（安全）。这两个目标之间存在张力——阻止一个不安全的动作虽然可以防止违规，但可能使智能体陷入困境且缺乏有原则的恢复途径。现有方法往往牺牲其中一个目标来换取另一个。我们提出 Soteria，一个通过经过验证的分层规划和规范的运行时强制执行来调和安全与效用的框架。在执行之前，智能体生成一个结构化计划，并在任何工具调用之前针对规范对其进行形式化验证。在执行期间，经过验证的计划发挥双重作用：它确保智能体采取符合规范的轨迹，并在不安全动作被阻止时提供恢复指导。在涵盖多种工具使用任务的多个基准测试中……（原文在此处截断）

    arXiv:2611.17880v1 Announce Type: new  Abstract: LLM agents that execute multi-step tool calls must satisfy two objectives simultaneously: completing the user's task (utility) and conforming to domain policies and correctness constraints (safety). These objectives are in tension -- blocking an unsafe action prevents a violation but can leave the agent stranded with no principled way to recover. Existing approaches sacrifice one objective for the other. We introduce Soteria, a framework that reconciles safety and utility through verified hierarchical planning and runtime enforcement of specifications. Before execution, the agent generates a structured plan that is formally verified against the specifications prior to any tool invocation. During execution, the verified plan serves dual roles: it ensures that the agent takes specification-conformant trajectories and provides guidance when unsafe actions are blocked. Across multiple benchmarks covering a diverse range of tool-use tasks, So
    
[^8]: DesignSpecs：面向AI智能体的视觉设计规范DESIGN.md数据集

    DesignSpecs: A DESIGN.md Dataset of Visual Design Specifications for AI Agents

    [https://arxiv.org/abs/2611.17713](https://arxiv.org/abs/2611.17713)

    该论文提出了首个大规模DESIGN.md视觉设计规范数据集DesignSpecs，包含来自15,794个公开GitHub仓库、经严格筛选的36,906个设计规范文件，为AI智能体在生成和修改用户界面时提供视觉设计规范的数据基础。

    

    尽管AI智能体越来越多地生成和修改用户界面，但它们很少依赖针对特定项目的视觉设计规范。DESIGN.md格式起源于谷歌的AI设计工具Stitch，并于2026年4月开源，它通过设计令牌（如命名颜色、文本样式）和Markdown文本来描述视觉识别，从而填补了这一空白。据我们所知，目前尚不存在此类设计规范文件的大规模研究数据集。我们提出了DesignSpecs，这是一个包含来自15,794个公开、非分叉GitHub仓库的36,906个DESIGN.md文件的数据集，代表17,240个唯一的文件内容。我们于2026年10月收集了613,207个名为DESIGN.md的文件。由于相同的文件名也被用于表示建筑规划和其他设计文档，我们通过YAML前置元数据、设计系统名称、至少一个可识别的设计令牌以及官方linter零错误等条件对这些文件进行了筛选。我们的JSON Lines数据集提供……（摘要原文在此处被截断）

    arXiv:2611.17713v1 Announce Type: new  Abstract: While AI agents increasingly generate and modify user interfaces, they rarely rely on a visual design specification for a certain project. The DESIGN.md format, which originated from Google's AI design tool Stitch and which was open-sourced in April 2026, addresses this gap by describing a visual identity through design tokens (such as named colors, text styles) and Markdown prose. To our knowledge, no large-scale research dataset of such design specification files exists. We present DesignSpecs, a dataset of 36,906 DESIGN.md files from 15,794 public, non-fork GitHub repositories, representing 17,240 unique file contents. We collected 613,207 files named DESIGN.md in October 2026. Because the same file name also denotes architectural plans and other design documents, we filtered those with YAML front matter, a design-system name, at least one recognized design token, and zero errors under the official linter. Our JSON Lines dataset provi
    
[^9]: OCTAVE：一个面向端到端HTML5游戏生成的能力分辨基准

    OCTAVE: A Capability-Resolved Benchmark for End-to-End HTML5 Game Generation

    [https://arxiv.org/abs/2611.17681](https://arxiv.org/abs/2611.17681)

    该论文提出OCTAVE基准，通过八个能力维度、180个任务和1,800条评估需求对端到端HTML5游戏生成进行需求级能力诊断，并揭示当前最优模型组合的完整性与丰富度仅约33%。

    

    完整的游戏生成要求智能体在漫长的开发过程中协调规则、交互、状态与呈现。单一的聚合分数会掩盖哪些能力已被实现、哪些需求仍未被满足。我们提出OCTAVE，一个以八个能力维度组织的完整HTML5游戏生成基准，包含180个任务和1,800条评估需求，这些需求来源于真实玩家请求、商业机制重建以及人类专家设计。该基准涵盖PC网页、移动网页、2D和3D场景。评估需求区分“完整性”——评估公开指定的核心实现——与“丰富度”——评估隐藏的、更深层的系统关系。基于静态源代码的判定可提供跨能力和任务条件的需求级诊断。对11种模型-框架组合的评估结果显示，完整性为33.2%，丰富度为32.5%；19.9%的生成记录属于静态门控失败或缺失。

    arXiv:2611.17681v1 Announce Type: new  Abstract: Complete game generation requires agents to coordinate rules, interactions, states, and presentation over a long development process. A single aggregate score obscures which capabilities are implemented and which requirements remain unmet. We present OCTAVE, a benchmark for complete HTML5 game generation organized around eight Capability Dimensions, with 180 tasks and 1,800 Evaluation Requirements drawn from real player requests, commercial-mechanics reconstruction, and human expert design. It covers PC Web, Mobile Web, 2D, and 3D. Requirements distinguish Completeness, assessing publicly specified core implementation, from Richness, assessing hidden, deeper system relationships. Static source-code judgments provide requirement-level diagnosis across capabilities and task conditions. Evaluation of 11 model-harness combinations yields 33.2% Completeness and 32.5% Richness; 19.9% of generation records are static gate failures or missing ar
    
[^10]: 作为逻辑程序的元构建系统

    Meta Build Systems as Logic Programs

    [https://arxiv.org/abs/2611.17672](https://arxiv.org/abs/2611.17672)

    该论文提出将构建配置表达为关系推理，并介绍了 buildlog 这一元构建系统，其分层逻辑核心将构建推导与环境探测隔离，拒绝工具链和软件包选择中的歧义，并通过溯源解释推导结果。

    

    诸如 CMake、Meson 和 Bazel 等元构建系统从项目描述、用户选择和机器事实中推导出命令图。命令式脚本、搜索路径和特殊情况使得常规决策之间交互不佳：搜索顺序可以在没有明确项目选择的情况下选择兼容的软件包，打包人员必须修补作者的编译标志偏好，而交叉构建则需要注释或嵌套构建树来支持构建期间运行的工具。我们的核心洞察是将配置表达为关系推理：目标文件源于源代码，编译标志和库源于依赖关系，语言标准源于传递性需求。我们介绍了 buildlog，一个元构建系统，其分层的逻辑核心将构建推导与非纯的环境探测隔离开来。它拒绝工具链或软件包选择中的歧义，并通过溯源来解释推导出的事实。作者约束和用户偏好使用不同的、经过静态检查的输入关系……

    arXiv:2611.17672v1 Announce Type: cross  Abstract: Meta build systems such as CMake, Meson and Bazel derive command graphs from project descriptions, user choices and machine facts. Imperative scripts, search paths and special cases make routine decisions interact poorly: search order can choose compatible packages without an explicit project choice, packagers must patch author flag preferences, and cross-builds need annotations or nested build trees for tools run during the build.   Our central insight is to express configuration as relational inference: objects follow from sources, flags and libraries from dependencies, and language standards from transitive requirements. We introduce buildlog, a meta build system whose stratified logical core isolates build derivation from impure environmental probes. It rejects ambiguity in toolchain or package selection and explains derived facts through provenance. Author constraints and user preferences use distinct, statically checked input rel
    
[^11]: SZZBench：具有显式预言的受控Git历史记录，用于诊断SZZ实现

    SZZBench: Controlled Git Histories with Explicit Oracles for Diagnosing SZZ Implementations

    [https://arxiv.org/abs/2611.17669](https://arxiv.org/abs/2611.17669)

    论文提出SZZBench，一个通过声明式场景生成带有显式预言的受控Git仓库、并借助26个朴素基线诊断SZZ实现成败原因的评测工具，目前包含17个类别共279个场景及四个SZZ工具的适配器。

    

    SZZ算法将修复缺陷的提交回溯到引入该缺陷的提交，其标注结果是缺陷预测数据集以及众多软件仓库挖掘研究的基础。SZZ实现大多使用从真实项目中挖掘出的预言进行评估，这类预言虽然贴近现实，但每个案例都混合了多种决定实现成败的历史因素，而聚合得分无法揭示其中的原因。我们提出了SZZBench这一能够揭示原因的工具：它根据带有显式预言的声明式场景生成普通的Git仓库，在声明的评估策略下对任意SZZ实现的预测结果进行评分，并将每个结果与26个朴素基线（例如总是回答第二个提交）进行对比。SZZBench还会依据朴素基线的结果来校验新场景（例如层级）的声明。目前，SZZBench提供了17个类别共279个场景，以及四个SZZ工具的适配器。运行……

    arXiv:2611.17669v1 Announce Type: new  Abstract: The SZZ algorithm traces a bug-fixing commit back to the commits that introduced the bug. Its labels underlie defect-prediction data sets and many mining software repositories studies. SZZ implementations are mostly evaluated against oracles mined from real projects. Such oracles are realistic, but each case mixes the history factors that make an implementation succeed or fail, and an aggregate score cannot show why. We present SZZBench, a tool that shows why: it generates ordinary Git repositories from declarative scenarios with explicit oracles, scores the predictions of any SZZ implementation under declared evaluation policies, and reads each result against 26 naive baselines, such as always answering the second commit. SZZBench also checks the declarations of new scenarios, such as the tier, against the results of the naive baselines. Currently, SZZBench ships 279 scenarios in 17 categories and adapters for four SZZ tools. Running th
    
[^12]: 按文件大小与每解决任务能耗选择量化代码大语言模型

    Choosing Quantized Code LLMs by File Size and Energy per Solved Task

    [https://arxiv.org/abs/2611.17605](https://arxiv.org/abs/2611.17605)

    本研究在 MacBook Air M2 上以“每解决任务的能耗”评测 Qwen2.5-Coder 的 24 个量化版本，给出按文件大小索引、附证据标签的模型选择指南，并揭示 3 GB 档位的 7B-q2_K 相比 2 GB 档位的 3B-q4_K_M 无显著性能提升却耗能翻倍。

    

    在笔记本电脑上运行代码助手的开发者只需下载一个模型文件，本地推理便把每次请求的能耗都转移到这台机器上。一条常见的经验法则是：低精度的大模型往往胜过高精度的小模型。我们在一台无风扇的 MacBook Air M2 上运行了四种规模的 24 个 Qwen2.5-Coder GGUF 文件，并在 542 个 EvalPlus 任务上以“每解决一个任务所消耗的能量”为代价指标对每个文件进行评估。最终产出一份以 Ollama 所列文件大小为索引的模型选择指南，且每个推荐都附有证据标签。在 1 GB 和 2 GB 档位，指南推荐 1.5B 和 3B 模型的 q4_K_M 文件，二者在任何核算方式下都位于正确性与能耗的前沿。在所列的 3 GB 档位，“模型越大越好”的规则会选择 7B-q2_K——这是唯一实测能装下的 7B 文件——但在此处它并不划算：与 2 GB 档位的选择 3B-q4_K_M 相比，该文件在解题数上没有显著提升，且 90% 置信区间将 3B-q4_K_M 的任何劣势限制在 2.4 分以内，而 7B-q2_K 每解决一个任务所需的能耗却是 2.03 倍。

    arXiv:2611.17605v1 Announce Type: cross  Abstract: A developer running a code assistant on a laptop downloads one model file, and local inference moves the energy of every request onto that machine. A common rule says a larger model at lower precision tends to beat a smaller one at higher precision. We ran 24 GGUF files of Qwen2.5-Coder at four sizes on a fanless MacBook Air M2 and priced each in energy per solved task on 542 EvalPlus tasks. The result is a selection guide indexed by Ollama's listed file size, with an evidence label on each pick. At 1 and 2 GB it picks the q4_K_M files of the 1.5B and 3B models, both on the correctness and energy frontier under every accounting. At 3 GB as listed, the larger-model rule picks 7B-q2_K, the only measured 7B file that fits, and there it does not pay. That file brings no significant gain in solves over the 2 GB pick, 3B-q4_K_M, and the 90% CI limits any 3B-q4_K_M deficit to 2.4 points. Yet 7B-q2_K needs 2.03 times the energy per solved task
    
[^13]: 超越源代码：评估自然语言规范在基于大语言模型的C到Rust翻译中的作用

    Beyond Source Code: Evaluating Natural Language Specifications for LLM-Based C-to-Rust Translation

    [https://arxiv.org/abs/2611.17581](https://arxiv.org/abs/2611.17581)

    该论文首次系统评估了自然语言规范作为输入在LLM驱动的C到Rust翻译中的作用，发现仅用源代码翻译虽测试通过率最高（98.6%）但往往照搬C的实现模式，而引入自然语言规范有望生成更地道、更少依赖unsafe结构的Rust代码。

    

    大语言模型（LLM）在C到Rust的翻译任务中已展现出可喜的成果，但现有方法主要依赖源代码作为输入，而将程序行为的自然语言规范作为替代或补充表示的研究尚不充分。我们假设直接进行代码翻译会促使LLM保留C语言特有的实现选择，从而生成功能上正确但冗长、依赖不安全（unsafe）结构、且不够地道的Rust代码。为验证这一假设，我们比较了三种翻译设置：仅代码、仅规范、以及代码与规范相结合。我们使用IBM CodeNet数据集中的300个C程序，从功能正确性、可维护性、复杂度、地道性、不安全结构的使用以及成本等方面评估了GPT-5.2和GPT-4o-mini。结果显示，仅代码翻译取得了最高的测试通过率（98.6%），但其生成的代码往往沿用C语言的实现模式。相比之下……（原文摘要在此处截断）

    arXiv:2611.17581v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promising results for C-to-Rust translation, yet existing approaches primarily rely on source code as input, leaving natural-language specifications of program behavior underexplored as an alternative or complementary representation. We hypothesize that direct code translation encourages LLMs to preserve C-specific implementation choices, producing Rust that may be functionally correct but unnecessarily verbose, reliant on unsafe constructs, and less idiomatic. We investigate this hypothesis by comparing three translation settings: code-only, specification-only, and code-and-specification. Using 300 C programs from the IBM CodeNet dataset, we evaluate GPT-5.2 and GPT-4o-mini on functional correctness, maintainability, complexity, idiomaticity, unsafe construct usage, and cost. Code-only translation achieves the highest test pass rate (98.6%) but often mirrors C's implementation patterns. In contras
    
[^14]: GraphScent：校准方向性相关性信号以实现有界上下文代码定位

    GraphScent: Calibrating Directional Relevance Signals for Bounded-Context Code Localization

    [https://arxiv.org/abs/2611.17527](https://arxiv.org/abs/2611.17527)

    该研究发现冻结语言模型对外围作用域和上游调用者的零温度相关性评分能可靠区分真正的结构近邻与词汇干扰项，而其他关系类型的评分接近随机水平，据此校准方向性相关性信号以实现单张消费级GPU上的无状态有界上下文代码定位。

    

    在单张消费级GPU上本地运行开放权重大语言模型，需要用简短的无状态代码检查来取代不断膨胀的多轮探索历史。然而，无状态的代码仓库定位取决于一个基础问题：当一个冻结的语言模型从紧凑视图中评估一个未经修改的代码实体时，哪些结构关系能使它的相关性判断对附近的修复目标具有参考价值？通过在86个代码仓库的211个真实Python问题上研究Gemma 4模型，我们发现了一种系统性的方向性相关性不对称现象：在零温度设置下，模型对外围作用域和上游调用者给出的相关性评分能够稳定地将真正的结构近邻与困难的词汇干扰项区分开来，而对下游被调用者、同层方法、历史共变伙伴以及稠密节点嵌入的评分则接近随机水平。零泄漏的掩码作用域探测实验以及控制词汇相似度与代码片段长度的回归分析（摘要在此处被截断）……

    arXiv:2611.17527v1 Announce Type: cross  Abstract: Running open-weight language models locally on a single consumer GPU requires replacing expanding multi-turn exploration histories with short, stateless code inspections. Stateless repository localization, however, hinges on a foundational question: when a frozen language model evaluates an unedited code entity from a compact view, which structural relationships make its relevance judgment informative about a nearby repair target? Studying Gemma 4 models across 211 real-world Python issues in 86 repositories, we uncover a systematic directional relevance asymmetry: zero-temperature relevance scores on enclosing scopes and upstream callers consistently separate true structural neighbors from hard lexical distractors, whereas scores on downstream callees, peer methods, historical co-change partners, and dense node embeddings remain near chance. Zero-leakage masked-scope probes and regressions controlling for lexical similarity and snippe
    
[^15]: 通过行为轨迹解读编码智能体

    Interpreting Coding Agents Through Their Behavioral Trajectories

    [https://arxiv.org/abs/2611.17440](https://arxiv.org/abs/2611.17440)

    该论文提出一种基于规则的解析器，将编码智能体的原始执行轨迹映射为高级行为，并通过对14万余条轨迹的行为画像分析，揭示解决率相近的智能体在探索、修改和验证代码的方式上存在显著差异，且这些行为画像本身足以在不同仓库与语言间识别出对应的智能体技术栈。

    

    标准的编码智能体基准测试主要评估二元化的任务成功与否，将执行过程视为黑箱，掩盖了产生这些结果的行为。我们分析了来自125种智能体配置、涵盖117个不同模型的大型语料库，共143,209条执行轨迹。为了能在异构的智能体接口之间进行比较，我们开发了一个基于规则的解析器，将原始执行痕迹映射为仓库搜索、文件编辑、测试执行等高级动作，并针对500个事件以六模型共识参考对其预测进行了审计。我们发现，解决率相近的智能体在探索、修改和验证代码的方式上仍可能存在显著差异。此外，在不使用任务结果、token数量或工具名称作为特征的情况下，汇总的行为画像能够在不相交的仓库集合和不同编程语言中检索出对应的智能体技术栈，尽管任务构成尚未被完全……（摘要在此处截断）

    arXiv:2611.17440v1 Announce Type: cross  Abstract: Standard coding-agent benchmarks primarily evaluate binary task success, treating execution as a black box and obscuring the behaviors that produce those outcomes. We analyze a large corpus of 143,209 execution trajectories from 125 agent configurations spanning 117 unique models. To enable comparisons across heterogeneous agent interfaces, we develop a rule-based parser that maps raw execution traces to high-level actions such as repository search, file editing, and test execution, and audit its predictions against a six-model consensus reference on 500 events. We find that agents with similar solve rates can nevertheless differ substantially in how they explore, modify and verify code. Moreover, without using task outcomes, token counts, or tool names as features, aggregate behavioral profiles retrieve the corresponding agent stack across disjoint repository sets and programming languages, although task composition is not fully contr
    
[^16]: 面向软件工程智能体的分层代码图时间一致性评估

    Time-Consistent Evaluation of Hierarchical Code Graphs for SWE Agents

    [https://arxiv.org/abs/2611.17388](https://arxiv.org/abs/2611.17388)

    该论文构建了一个零泄漏的时间一致性评估基准，在DragonFly和React的历史截止快照上，以四种提示粒度对比评估基线、扁平AST图与分层依赖图ServiceExplorer三种上下文增强方案对SWE智能体解决100个后续拉取请求的效果。

    

    评估具备仓库感知能力的软件工程（SWE）智能体需要在既有仓库知识与未来代码演进之间进行严格分离，然而标准基准测试常常受到时间污染和提示词层面解题方案泄漏的困扰。基于Sun（2026）提出的时间一致性基准协议，我们在C++项目DragonFly和JavaScript/TypeScript项目React上强制实施严格的零泄漏隔离，将仓库快照与知识图谱锁定在历史截止提交T₀处，并在最简、简洁、上下文相关和引导式四种提示粒度下评估100个T₀之后的拉取请求。在该零泄漏协议下，我们对三种上下文增强方案进行了匹配评估：（1）未增强的基线；（2）扁平的语法级Tree-sitter AST图；（3）ServiceExplorer——一种确定性的两级分层依赖图，能够捕获一级系统/包容器及二级内容（原文摘要在此处截断）。

    arXiv:2611.17388v1 Announce Type: cross  Abstract: Evaluating repository-aware software engineering (SWE) agents requires strict separation between pre-existing repository knowledge and future code evolution, yet standard benchmarks frequently suffer from temporal contamination and prompt-level solution leakage. Building on the time-consistent benchmark protocol of Sun (2026), we enforce strict zero-leakage isolation across DragonFly in C++ and React in JavaScript/TypeScript, locking repository snapshots and knowledge graphs at historical cutoff commits $T_0$ and evaluating 100 post-$T_0$ pull requests across four prompt granularities: Minimal, Concise, Contextual, and Guided.   Under this zero-leakage protocol, we conduct a matched evaluation of three context-augmentation regimes: (1) an unaugmented Baseline, (2) a flat syntax-level Tree-sitter AST graph, and (3) ServiceExplorer, a deterministic two-level hierarchical dependency graph capturing Level-1 system/package containers and Le
    
[^17]: ProdPilot：一个面向氛围编码项目自动化生产就绪检查与部署的分层模型上下文协议系统

    ProdPilot: A Layered Model Context Protocol System for Automated Production Readiness and Deployment of Vibe-Coded Projects

    [https://arxiv.org/abs/2611.17381](https://arxiv.org/abs/2611.17381)

    ProdPilot是一个本地MCP服务器，其核心创新在于将50条生产就绪规则中的88%实现为固定模板或基于项目信息提取的确定性修复，而非依赖不可控的模型自由生成，从而对氛围编码项目实现可验证的自动化审计、修复、门控与部署。

    

    AI辅助开发环境能够将自然语言请求转化为可运行的应用程序，然而这些项目在首次部署时常常因运维层面的问题而失败：源代码文件中包含凭据、缺失安全中间件、没有容器定义、没有健康检查端点以及没有交付流水线。对此类缺陷的自动修复通常依赖于自由格式的模型生成，其结果在不同运行之间变化不定，且很少经过独立验证。本文提出了ProdPilot，一个本地模型上下文协议服务器，它依据50条生产就绪规则对Node.js和React项目进行审计，通过有界循环修复违规项，对结果进行门控检查并完成部署。每条规则根据其修复所需的内容进行分类：28条修复采用固定模板，16条通过从项目中提取的值进行填充，仅有6条由IDE代理的模型编写，因此88.0%的规则集不需要模型编写的内容，并在499个真实仓库上……（原文摘要至此被截断）

    arXiv:2611.17381v1 Announce Type: cross  Abstract: AI-assisted development environments turn natural-language requests into working applications, yet the projects they produce often fail on their first deployment for operational reasons: credentials in source files, missing security middleware, no container definition, no health endpoint and no delivery pipeline. Automated repair of such defects usually relies on free-form model generation, which varies between runs and is seldom checked independently. This paper presents ProdPilot, a local Model Context Protocol server that audits Node.js and React projects against 50 production readiness rules, repairs violations through a bounded loop, gates the result and deploys it. Each rule is classified by what its fix requires: 28 fixes are fixed templates, 16 are filled with values extracted from the project and only 6 are written by the IDE agent's model, so 88.0\% of the ruleset needs no model-written content, and across 499 real repositori
    
[^18]: 一切皆是操作视图：智能体系统中作为上下文投影的用户界面

    Everything is an Operational View: UI as a Projection over Context in Agentic Systems

    [https://arxiv.org/abs/2611.17355](https://arxiv.org/abs/2611.17355)

    本文提出AFS-UI框架，将用户界面重新概念化为智能体文件系统之上的上下文投影，使界面成为对共享资源的动态可检查视图，从而通过统一操作界面同时支持人类与智能体的交互。

    

    生成式人工智能的兴起催生了能够根据用户意图动态构建界面的生成式用户界面，但现有方法依赖于请求-响应管道，这种管道割裂了上下文并限制了透明度与控制能力。本文提出AFS-UI，将用户界面重新概念化为基于智能体文件系统抽象构建的上下文投影，该抽象在统一、持久的工作空间中组织数据、能力和交互状态。特别地，AFS-UI解决了当前系统中的关键局限性，包括异构资源的可发现性不足、操作语义不兼容、投影与上下文相互纠缠、数据源组合与替换机制僵化以及可演化性受限等问题。在该框架下，界面成为共享资源之上动态、可检查的视图，通过统一操作界面同时支持人类与智能体的交互。

    arXiv:2611.17355v1 Announce Type: cross  Abstract: The emergence of generative AI has enabled Generative User Interfaces (GenUI) that dynamically construct interfaces from user intent, yet existing approaches rely on request-response pipelines that fragment context and limit transparency and control. This paper proposes AFS-UI, which reconceptualises UI as projections over context built on an agentic file-system abstraction that organizes data, capabilities, and interaction state within a unified, persistent workspace. In particular, AFS-UI addresses key limitations in current systems, including heterogeneous resource discoverability, incompatible operation semantics, tangling projection and context, rigid source composition and substitution, and limited evolvability. Within this framework, interfaces become dynamic, inspectable views over shared resources, supporting both human and agent interaction through a common operational surface.
    
[^19]: TSBugs Explorer：TypeScript缺陷的探索与基准测试平台

    TSBugs Explorer: Exploring and Benchmarking TypeScript Bugs

    [https://arxiv.org/abs/2611.14940](https://arxiv.org/abs/2611.14940)

    本文提出了TSBugs Explorer，一个基于633个真实TypeScript缺陷数据集构建的交互式可视化分析平台，支持缺陷探索、跨仓库故障特征比较以及面向研究的基准子集构建。

    

    精选的真实世界软件缺陷数据集为软件测试、调试和自动化程序修复等领域的实证研究提供了支持。然而，探索异构的缺陷集合、比较不同项目间的故障特征以及构建面向特定研究的子集，往往需要人工检查和自定义脚本。我们提出了TSBugs Explorer，这是一个交互式可视化分析平台，构建于我们先前发表的、包含来自16个开源TypeScript仓库的633个缺陷并划分为11个故障类别的数据集之上。该平台通过协同联动的可视化、交互式过滤和详细的缺陷检查，整合了缺陷记录、仓库特征、依赖信息以及分类体系指导。它支持故障分布的多种聚合与归一化方式、成对的仓库比较，以及对所观察到的故障特征差异的分析。集成的基准构建器允许研究者（摘要在此处被截断）

    arXiv:2611.14940v2 Announce Type: replace  Abstract: Curated datasets of real-world software bugs support empirical research in software testing, debugging, and automated program repair. However, exploring heterogeneous bug collections, comparing fault characteristics across projects, and constructing research-specific subsets often require manual inspection and custom scripts. We present TSBugs Explorer, an interactive visual analytics platform built upon our previously-published dataset of 633 bugs from 16 open-source TypeScript repositories, classified into 11 fault categories. The platform integrates bug records, repository characteristics, dependency information, and taxonomy guidance through coordinated visualizations, interactive filtering, and detailed bug inspection. It supports alternative aggregation and normalization of fault distributions, pairwise repository comparisons, and analysis of differences in observed fault profiles. An integrated benchmark builder allows researc
    
[^20]: Caprese：基于历史与运行时分析的变更影响推荐

    Caprese: Change Impact Recommendation Using History and Runtime Analysis

    [https://arxiv.org/abs/2611.14937](https://arxiv.org/abs/2611.14937)

    Caprese是一个面向Node.js的开源函数级变更影响推荐框架，它结合历史共变更挖掘与运行时依赖分析，生成可能受代码变更影响的函数排序列表。

    

    变更影响推荐用于识别可能受源代码变更影响的程序实体。历史共变更和运行时依赖提供了互补的证据，尤其是在JavaScript中，动态和异步行为使依赖分析变得复杂。我们提出了Caprese，一个用于Node.js函数级变更影响推荐的开源框架。给定一个变更集，它会生成一个按优先级排序的、包含可能受影响函数的影响集。它支持三种推荐方式：通过置信度引导地挖掘反复出现的、与变更集相关的不同长度的共变更模式，实现基于仓库的推荐；通过执行时衍生的依赖关系（涉及调用、回调、事件、定时器和共享状态），实现动态推荐；以及整合两种分析的混合推荐。其模块化实现支持可复用的分析。在十个开源应用和100个选定变更集上的评估表明，平均而言……

    arXiv:2611.14937v2 Announce Type: replace  Abstract: Change impact recommendation identifies program entities potentially affected by source code changes. Historical co-changes and runtime dependencies provide complementary evidence, particularly in JavaScript, where dynamic and asynchronous behaviour complicates dependency analysis. We present Caprese, an open-source framework for function-level change impact recommendation in Node.js. Given a change set, it produces a ranked impact set of potentially affected functions. It supports repository-based recommendation through confidence-guided mining of recurring, change-set-relevant co-change patterns of varying lengths; dynamic recommendation through execution-derived dependencies involving calls, callbacks, events, timers, and shared state; and hybrid recommendation integrating both analyses. Its modular implementation supports reusable analyses. Evaluation on ten open-source applications and 100 selected change sets shows that, on ave
    
[^21]: 删除敏感性是模型的一种属性：对七个开放推理模型的三条件测量

    Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

    [https://arxiv.org/abs/2611.13223](https://arxiv.org/abs/2611.13223)

    本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。

    

    推理模型在多轮工具会话的每一轮都会写下一条推理轨迹，而后续轮次会读取这段历史。在94k条ShareGPT对话中，73%的用户交互跨越多轮。上游提供商将每条推理轨迹绑定到其发出者，因此跨发出者转发的请求到达时其历史已被删除，而这种剥离操作带来的任务成本此前从未被测量。据我们所知，本文首次对多轮工具调用中完整推理历史删除进行了直接、盲测、多种子的测量，涵盖三种渲染条件、七个开放权重推理模型、三个随机种子、794个任务以及官方检查器。相同的删除操作导致的性能变化跨度为-4.4到+51.3分，且两端均不包含零；对于唯一一个从0%到100%删除比例重新测试的模型，其读数变化最多仅1.5分。删除敏感性是模型层面的属性。该读数跟随模型本身并保持稳定（原文在此处截断）。

    arXiv:2611.13223v1 Announce Type: cross  Abstract: Reasoning models write a reasoning trace at every turn of a multi-turn tool session, and later turns read this history. Across 94k ShareGPT conversations, 73 percent of user interactions span multiple turns. Upstream providers bind each trace to its issuer, so a request forwarded across issuers arrives with its history deleted, and the task cost of this stripping remains unmeasured. To the best of our knowledge, this paper presents the first direct, blind, multi-seed measurement of full reasoning-history deletion in multi-turn tool calling, through three rendering conditions, seven open-weight reasoning models, three seeds, 794 tasks, and the official checker. The same deletion spans -4.4 to +51.3 points with both ends excluding zero, and for the one model re-dosed from 0 to 100 percent deletion, the reading varies by at most 1.5 points. Deletion sensitivity is a model-level property. The reading follows the model itself and stays cons
    
[^22]: PRISM：面向RTL PPA优化的基于偏好奖励与不确定性集成的综合最小化框架

    PRISM: Preference-based Reward and Uncertainty-Integrated Synthesis Minimization for RTL PPA Optimization

    [https://arxiv.org/abs/2611.11051](https://arxiv.org/abs/2611.11051)

    PRISM是一个预算感知的多目标RTL优化框架，它通过基于偏好的奖励模型与不确定性引导的采集策略，在大幅减少昂贵商业综合工具调用次数的同时，有效优化设计的功耗、性能和面积（PPA）。

    

    本文提出了一种预算感知的多目标框架（称为PRISM），用于RTL（寄存器传输级）优化，在大幅减少昂贵的商业综合工具调用的同时，提升功耗、性能和面积（PPA）指标。该方法从一个可综合的基线设计和功能测试平台出发，迭代执行以下流程：LLM驱动的代码重写生成、快速的编译/仿真等价性过滤、低成本的代理特征提取，以及选择性的商业综合工具评估。为了优先筛选出值得进行高质量综合的候选设计，我们利用从商业综合工具结果中获得的成对排序监督信号，训练了一个基于偏好的奖励模型，并将集成不确定性与多样性感知的采集规则相结合，以避免早期偏差和冗余评估。该方法的一个关键优势是，PRISM在整个运行过程中持续维护一个由非支配设计组成的最佳权衡集合G：G保留了最佳的……

    arXiv:2611.11051v2 Announce Type: replace-cross  Abstract: This paper introduces a budget-aware, multi-objective framework (called PRISM) for RTL optimization that improves power, performance, and area (PPA) while sharply limiting expensive commercial synthesis tool calls. Starting from a synthesizable baseline and functional testbench, the method iterates through LLM-driven rewrite generation, fast compile/simulation equivalence filtering, low-cost proxy feature extraction, and selective commercial synthesis tool (Synopsys Design Compiler (DC)) evaluation. To prioritize which candidates deserve high-quality synthesis, we train a preference-based reward model using pairwise ranking supervision derived from commercial synthesis tool outcomes, and combine ensemble uncertainty with a diversity-aware acquisition rule to avoid early bias and redundant evaluations. A key advantage is that PRISM maintains a best-tradeoff set G of non-dominated designs throughout the run: G preserves the best 
    
[^23]: SoK：后量子密码学（PQC）在软件开发中的集成：方法、差距与未来方向

    SoK: Post-Quantum Cryptography (PQC) Integration in Software Development: Approaches, Gaps, and Future Directions

    [https://arxiv.org/abs/2611.08442](https://arxiv.org/abs/2611.08442)

    本文系统梳理了软件开发中后量子密码学集成的研究现状，指出尽管现有工作奠定了重要的技术与组织基础，但各阶段（准备、实现、迁移、部署、维护及开发者实践）之间缺乏协调，生命周期支持仍是割裂的。

    

    标准化的后量子密码学（PQC）对于保护软件系统抵御未来的量子威胁至关重要，但仅有标准化算法并不能为其安全集成到软件中提供路径。本文对软件开发中的PQC集成进行了知识系统化（Systematisation of Knowledge, SoK）研究，综合梳理了现有在实现、迁移和开发者教育等方面的方法。我们依据NIST PQC标准与PQC迁移指南、软件开发生命周期覆盖范围、网络安全实践、部署成熟度以及测试与评估证据，对这些方法进行了分析。我们的分析表明，现有工作已奠定了重要的技术和组织基础，但在整个软件开发生命周期中所提供的支持是彼此割裂的，在准备、实现、迁移、部署、维护与开发者实践之间缺乏协调。评估证据同样……

    arXiv:2611.08442v1 Announce Type: new  Abstract: Standardised post-quantum cryptography (PQC) is critical for protecting software systems against future quantum threats, but standardised algorithms alone do not provide a pathway for their secure integration into software. This paper presents a Systematisation of Knowledge (SoK) of PQC integration in software development, synthesising existing approaches across implementation, migration, and developer education. We analyse these approaches against NIST PQC standards and PQC migration guidance, software development lifecycle coverage, cybersecurity practices, deployment maturity, and testing and evaluation evidence. Our analysis shows that existing work establishes important technical and organisational foundations but provides disconnected support across the software development lifecycle, with limited coordination between preparation, implementation, migration, deployment, maintenance, and developer practices. Evaluation evidence also 
    
[^24]: 被接受但实则错误：API迁移后LLM代码修复的静默失败

    Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations

    [https://arxiv.org/abs/2611.08269](https://arxiv.org/abs/2611.08269)

    预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。

    

    在运行时或API迁移之后，经LLM修复的程序可能顺利运行却返回错误结果，而基于典型测试输入的发布检查会将其放行发布。我们报告了一项预注册研究，包含73个任务，覆盖CPython、Node.js和SQLite中60个有文档记录的变更。每个候选修复都在真实运行时上针对未公开的契约执行。在整个修复空间中，基于典型输入的全断言门检查放行了0.410比例的错误修复。一个根据发布说明编写的边缘输入将该比例降至0.065，在15个未见过的集群上降至0.061。来自一个从未见过该程序的盲测GPT-5.6-luna编写者的三个输入达到了0.070。两个后续冻结阶段将候选菜单扩大至1491个片段，随后又添加了233个在典型输入上与遗留输出匹配的困难干扰项。边缘输入（0.058）和前沿模型编写者（0.058）通过了其预注册测试。然而，仍有0.168和0.153的困难错误程序通过了检查。

    arXiv:2611.08269v1 Announce Type: cross  Abstract: After a runtime or API migration, an LLM-repaired procedure can run cleanly yet return wrong results, and a release check on typical test inputs ships it. We report a preregistered study of 73 tasks covering 60 documented changes in CPython, Node.js and SQLite. Every candidate is executed on the real runtimes against withheld contracts. Over the complete repair space, the all-assertions gate on typical inputs released 0.410 of wrong repairs. One edge input written from the release note cut this to 0.065, and to 0.061 on 15 unseen clusters. Three inputs from a blind GPT-5.6-luna writer that never saw the program reached 0.070. Two further frozen stages enlarged the menu to 1491 fragments and then added 233 hard distractors that match the legacy output on typical inputs. The edge input (0.058) and the frontier-model writer (0.058) passed their preregistered tests. Yet 0.168 and 0.153 of the hard wrong programs still passed. Two local-mod
    
[^25]: 在受监管金融领域落地治理型智能体工作流：基于 Zafin AIOS 的“零号客户”案例研究

    Operationalizing Governed Agentic Workflows in Regulated Finance: A Customer Zero Case Study using Zafin AIOS

    [https://arxiv.org/abs/2610.33253](https://arxiv.org/abs/2610.33253)

    本文提出Zafin AIOS这一端到端编排与控制平台，并通过“零号客户”内部生产案例研究，验证了治理型智能体工作流在受监管金融行业中从试点走向生产落地的可行性与有效性。

    

    金融和医疗等受监管行业面临着关键的治理挑战，这些挑战阻碍了AI辅助软件开发的采用，也阻碍了从实验性试点向生产环境工作的过渡。本文介绍了Zafin AIOS，一个专为在金融等受监管行业中部署AI而设计的端到端编排平台和控制平面。我们提供了架构原则，以及AI辅助工作负载与人工主导工作负载之间的实证比较。为了对该框架进行实证验证，我们对Zafin的内部企业工作流程进行了案例研究分析，我们将其称为“零号客户”。本文为受监管行业中的AI辅助软件工程贡献了一个生产案例研究。我们在两个不同的生产代码库上评估了使用Zafin AIOS的结果，一个用于定价执行平台，另一个用于客户交易谈判平台。我们的贡献是一个……

    arXiv:2610.33253v1 Announce Type: cross  Abstract: Regulated industries such as finance and healthcare face critical governance challenges that hamper the adoption of AI-assisted software development and hinder the transition from experimental pilots to production work. This paper presents Zafin AIOS, an end-to-end orchestration platform and control plane designed to deploy AI within regulated industries such as finance. We provide the architectural principles and an empirical comparison between AI-assisted and human-led workloads. To validate this framework empirically, we conduct a case study analysis of Zafin's internal enterprise workflows, which we refer to as Customer Zero. This paper contributes a production case study of AI-assisted software engineering in regulated industries. We evaluate the results of using Zafin AIOS with two different production codebases, one for the pricing execution platform and the other for the client deal negotiation platform. Our contribution is a c
    
[^26]: CAMGen：一种核心变更感知的多视图提交信息生成框架

    CAMGen: A Core Change-Aware Multi-View Framework for Commit Message Generation

    [https://arxiv.org/abs/2610.31552](https://arxiv.org/abs/2610.31552)

    CAMGen通过识别代表主要变更意图的核心实体，并借助仓库级语义聚类构建功能语义视图，将多代码实体协作实现的功能上下文融入提交信息生成，从而生成更能准确解释变更及其影响的提交信息。

    

    提交信息帮助开发者理解代码变更并支持软件维护。近期的方法利用仓库上下文来改进基于大语言模型（LLM）的提交信息生成（CMG）。然而，现有的CMG方法对于理解局部编辑如何影响由多个代码实体共同实现的功能方面提供的支持有限。这种功能层面的理解对于生成能够解释主要变更及其影响的提交信息至关重要。为了解决这一问题，我们提出了CAMGen，一个核心变更感知的多视图提交信息生成框架，它将功能上下文融入到生成过程中。CAMGen首先识别代表主要变更意图的核心实体；然后通过仓库范围的语义聚类形成候选功能单元，并利用包含核心实体的单元构建功能语义视图。该视图为解释变更提供了语义上下文……

    arXiv:2610.31552v1 Announce Type: new  Abstract: Commit messages help developers understand code changes and support software maintenance. Recent approaches use repository context to improve commit message generation (CMG) with large language models (LLMs). However, existing CMG methods provide limited support for understanding how local edits affect functionality jointly implemented by multiple code entities. This functional understanding is essential for generating commit messages that explain the main change and its impact. To address this problem, we propose CAMGen, a core change-aware multi-view commit message generation framework that incorporates functional context into generation. CAMGen identifies core entities that represent the primary change intent. It then forms candidate functional units through repository-wide semantic clustering and uses those containing the core entities to construct a Functional Semantic View. This view supplies semantic context for interpreting the a
    
[^27]: 大语言模型能否改进自身的执行框架？基于ARC-AGI-3的对照研究

    Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3

    [https://arxiv.org/abs/2610.30499](https://arxiv.org/abs/2610.30499)

    该论文通过受控实验首次系统考察前沿大语言模型能否自主构建并迭代改进自己的执行框架，让模型在ARC-AGI-3交互式基准上独立编写、使用并两次修订完整框架代码库，再于二十个从未参与改进的留出游戏上检验其泛化能力。

    

    通用智能系统能够高效地习得新技能，而无需人类为每个新任务重新设计其软件脚手架。然而，语言模型很少单独行动——执行框架是一种符号程序，它代表模型编排各类调用，负责处理感知、记忆、工具使用、异常恢复和推理分配等事务。ARC-AGI-3是一个交互式基准测试，要求智能体探索陌生环境、在行动过程中自主弄清目标并持续学习。在该基准上，由人类引导的系统已经在公开游戏中报告了满分区分数，说明有效的设计是确实存在的。本文提出的问题是：前沿模型能否自行构建并持续改进这样一个执行框架？通过一个固定的编码引导流程，每个模型先构建一个完整的任务执行框架，通过该框架在五个开发游戏上执行任务，并对整个代码库进行两次修订。随后，每个版本都在二十个从未参与过适应过程的留出游戏上进行评估。

    arXiv:2610.30499v1 Announce Type: cross  Abstract: A generally intelligent system acquires new skills efficiently without requiring humans to redesign its software scaffold for every new task. Yet language models rarely act alone. A harness is a symbolic program that orchestrates calls to the model, handling perception, memory, tool use, recovery, and inference allocation on the model's behalf. ARC-AGI-3 is an interactive benchmark in which agents must explore unfamiliar environments, figure out their goals on the fly, and keep learning as they go. On it, human-steered systems already report ceiling scores across the public games, so we know effective designs exist. We ask whether frontier models can build and improve such a harness on their own. Through a fixed coding bootstrap, each model constructs a complete task harness, acts through that harness on five development games, and revises the entire codebase twice. Every version is evaluated on twenty games that never influence adapta
    
[^28]: 基于智能体编码任务强化学习的跨基准迁移

    Cross-Benchmark Transfer from RL on Agentic Coding Tasks

    [https://arxiv.org/abs/2610.00890](https://arxiv.org/abs/2610.00890)

    仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。

    

    编码智能体常常在“最后一公里”失败：它们完成了功能的大部分却遗漏某项需求、只测试其实现已经能处理的情况、破坏了本应保持不变的行为，或基于未经检验的假设进行验证。我们探究在专家构建的智能体编码任务上进行强化学习（RL）能否弥合这一差距，以及智能体所学内容能否迁移到训练分布之外。我们仅使用强化学习对 Kimi K2.7 Code（一个 1 万亿参数、激活参数 32B 的开源混合专家模型）在 1,700 个任务上进行后训练：其中 1,000 个仓库任务由隐藏的 fail-to-pass 测试和保护既有行为的 pass-to-pass 测试评分，700 个终端任务由专家编写的隐藏验证器评分。奖励为目标检查的通过比例，若任何 pass-to-pass 测试失败则奖励降为零。在秩为 32 的 LoRA 适配器上进行一个 epoch 的 GSPO 训练，在我们评估的六个外部基准上的 pass@1 均有提升，跨越三个……（原文摘要在此处截断）

    arXiv:2610.00890v1 Announce Type: cross  Abstract: Coding agents often fail in the last mile: they build most of a feature but drop a requirement, test only the cases their implementation already handles, break behavior that was supposed to stay intact, or validate against an unchecked assumption. We ask whether reinforcement learning (RL) on expert-built agentic coding tasks closes this gap, and whether what the agent learns transfers beyond the training distribution. We post-train Kimi K2.7 Code, a 1T-parameter (32B active) open-weight mixture-of-experts model, with RL alone on 1,700 tasks: 1,000 repository tasks graded by hidden fail-to-pass tests and by pass-to-pass tests of existing behavior, and 700 terminal tasks graded by expert-written hidden verifiers. The reward is the fraction of target checks passed and drops to zero if any pass-to-pass test fails. One epoch of GSPO on a rank-32 LoRA adapter improves pass@1 on each of the six external benchmarks we evaluated, across three 
    
[^29]: 智能体AI时代的可复现性：以代码库时间尺度进行的上下文工程

    Reproducibility in the Age of Agentic AI: Context Engineering at the Timescale of a Codebase

    [https://arxiv.org/abs/2609.11728](https://arxiv.org/abs/2609.11728)

    本文提出可复现研究实践本质上就是为AI编码智能体进行的上下文工程，智能体降低了维护测试、提交历史和决策记录等科研工件的成本并使收益即时化，但研究人员仍需承担验证这些工件及其科学判断的责任。

    

    可复现的研究实践本身就是为AI编码智能体进行的上下文工程。我认为，智能体降低了维护测试、提交历史、仓库结构、指令和决策记录的成本，同时使其收益立竿见影。研究人员仍然有责任验证这些工件以及它们所编码的科学判断。

    arXiv:2609.11728v1 Announce Type: new  Abstract: Reproducible research practices are context engineering for AI coding agents. I argue that agents lower the cost of maintaining tests, commit histories, repository structure, instructions, and decision records while making their benefits immediate. Researchers remain responsible for verifying these artifacts and the scientific judgments they encode.
    
[^30]: 从读代码到读规约：面向LLM驱动的代码库维护的验证层

    From Reading Code to Reading Spec: A Verified Layer for LLM-Driven Codebase Maintenance

    [https://arxiv.org/abs/2609.06383](https://arxiv.org/abs/2609.06383)

    论文提出PROOF系统，将代码库拓扑抽象为分层自然语言规约，并通过从规约重构源代码来证明语义等价，使LLM能够借助规约间接、可靠地执行代码库维护并防止语义漂移。

    

    LLM生成代码的快速增长加剧了软件的复杂性，并加重了工程师的维护负担。虽然LLM提供了一种潜在的自动化替代方案，但这种结构性复杂性阻碍了它们直接管理代码库的能力。我们提出了原始功能可证明表示（PROOF），它通过结构化规约间接地管理代码库。为了严格地通过这些规约实现代码库的全生命周期管理，PROOF将代码库拓扑抽象为分层的自然语言表示。为了建立绝对的信任，该系统通过仅从规约重构源代码来证明语义等价性。这一经过验证的基础驱动维护请求，在执行代码修改的同时同步更新自身，以防止语义漂移。在真实世界代码库上的实验证实了这些规约的有效性。

    arXiv:2609.06383v2 Announce Type: replace-cross  Abstract: The rapid growth of LLM-generated code increases software complexity and the maintenance burden on engineers. While LLMs offer a potential automated alternative, this structural complexity hinders their ability to manage codebases directly. We introduce the Provable Representation Of Original Functionality (PROOF), which manages codebases indirectly via structured specifications. To enable full-lifecycle codebase management strictly through these specifications, PROOF abstracts codebase topology into a hierarchical natural-language representation. To establish absolute trust, the system proves semantic equivalence by reconstructing source code exclusively from this specification. This verified foundation drives maintenance requests, executing code modifications while synchronously updating itself to prevent semantic drift. Experiments on real-world repositories confirm the effectiveness of these specifications.
    
[^31]: 面向AI辅助软件开发的治理方法论层：缺陷分类体系、受控消融实验与“过程重于能力”的证据

    A Governance Methodology Layer for AI-Assisted Software Development: Defect Taxonomy, Controlled Ablation, and Process-Over-Capability Evidence

    [https://arxiv.org/abs/2609.04218](https://arxiv.org/abs/2609.04218)

    本文提出一套AI辅助软件开发的治理方法论层，通过缺陷分类体系、跨工具可移植的运行时解耦治理门以及“方法论即代码”的形式化，并以受控消融实验证明过程治理比模型能力更重要。

    

    自主编码智能体能够以高速度产出通过语法检查的输出——包括编译、类型安全和持续集成（CI）。然而，语法正确并不意味着语义正确：设计边界、安全不变量以及可维护性契约对于自动化流水线而言在结构上仍是不可见的。本文为弥合这一差距做出了四项贡献。首先，我们提出了一种基于五个AI智能体权限与治理模块构建的缺陷分类体系，区分了静态分析可在结构上检测出的缺陷与需要语义审查的缺陷。其次，我们描述了一种运行时解耦的治理门——一种基于文件的协议，它读取生成器输出并给出结构化判定结果，而无需API耦合，因此可以跨不同代码生成工具移植。第三，我们将“方法论即代码”形式化：将验证协议表达为一种版本受控、可执行、跨平台的制品，并采用两层架构……（原文在此处截断）

    arXiv:2609.04218v1 Announce Type: new  Abstract: Autonomous coding agents produce output that passes syntactic checks -- compilation, type safety, CI -- at high velocity. Yet syntactic correctness does not imply semantic correctness: design boundaries, security invariants, and maintainability contracts remain structurally invisible to automated pipelines. This paper makes four contributions toward closing this gap. First, we present a defect-class taxonomy grounded in five AI agent permission and governance modules, distinguishing defects structurally detectable by static analysis from those requiring semantic review. Second, we describe a runtime-decoupled governance gate -- a file-based protocol that reads generator output and emits a structured verdict without API coupling, making it portable across code-generation tools. Third, we formalize methodology-as-code: expressing a verification protocol as a version-controlled, executable, cross-platform artifact with a two-layer architect
    
[^32]: 驱动思维：VLA推理-轨迹一致性的运行时监控

    Drive the Thoughts: Runtime Monitoring of VLA Reasoning-Trajectory Consistency

    [https://arxiv.org/abs/2608.29583](https://arxiv.org/abs/2608.29583)

    本文提出DriveAlignBench数据集，通过运行时监控VLA模型思维链推理与生成轨迹之间的一致性，来检测自动驾驶系统中潜在的不安全行为。

    

    自动驾驶汽车（AV）在复杂环境中运行，一旦发生故障后果严重。用于感知与规划的先进机器学习模型是克服至少部分复杂性的关键，但其黑盒特性使验证与确认（V&V）变得困难。近期将视觉-语言-动作（VLA）模型集成到自动驾驶汽车中带来了独特机遇：除了生成轨迹之外，这些模型还会产生显式的思维链（CoT）来解释其底层推理依据。该思维链为交叉检验模型输出、检测可能暴露不安全或非预期行为的不一致性提供了丰富的规范。本文评估了来自近期开放驾驶VLA模型的思维链能否支持此类监控。我们构建了DriveAlignBench，这是一个源自NVIDIA Alpamayo 1.5自动驾驶VLA的专用数据集，包含150对思维链-轨迹对，并对其进行了人工标注，涵盖可靠性、轨迹一致性以及……

    arXiv:2608.29583v1 Announce Type: new  Abstract: Autonomous vehicles (AVs) operate in complex environments where failures are consequential. Sophisticated machine learning models for perception and planning are key to overcoming at least part of that complexity, but their black-box nature complicates validation and verification (V&V). The recent integration of Vision-Language-Action (VLA) models into AVs introduces a unique opportunity: besides generating trajectories, these models produce an explicit Chain-of-Thought (CoT) explaining their underlying rationale. This CoT provides a rich specification to cross-check model outputs and detect inconsistencies that may expose unsafe or unintended behavior. This paper assesses whether CoTs from a recent open driving VLA can support such monitoring. We curate DriveAlignBench, a specialized dataset from NVIDIA's Alpamayo 1.5 VLA for AVs containing 150 CoT-trajectory pairs, which we manually annotate for reliability, trajectory consistency, and
    
[^33]: 一个崎岖的前沿：评估代码代理对语义保持变换的鲁棒性

    A Jagged Frontier: Evaluating Robustness of Code Agents to Semantics-Preserving Transformations

    [https://arxiv.org/abs/2608.18389](https://arxiv.org/abs/2608.18389)

    本文通过引入随机语义保持变换采样器，系统评估了代码代理在代码库语义等价改写下的鲁棒性，发现其修复性能会显著下降，揭示了现有代理的脆弱性。

    

    arXiv:2608.18389v1 公告类型：新 摘要：AI代码代理正日益被部署来解决真实软件问题，然而它们在表面代码变化下的可靠性仍鲜为人知。我们评估了当周围代码库被重写为语义等价形式时，修复仓库级问题的代码代理是否仍保持可靠。我们引入了一个随机变体采样器，应用常见的语义保持变换（SPTs）——涵盖控制流重写、死代码注入和标识符重命名——以生成扰动变体。我们评估了两种代理框架（mini-SWE agent和OpenCode），每种框架由四个前沿模型之一（Claude Opus 4.5、Kimi K2.5、MiniMax M2.5和Qwen 3.6-27B）支持，在来自SWE-bench Verified和SWE-bench Pro的实例上进行测试。对于每个实例，代理在未扰动和扰动变体上多次运行，产生配对解决率估计，以隔离扰动效应与内在变化。

    arXiv:2608.18389v1 Announce Type: new  Abstract: AI code agents are increasingly deployed to resolve real software issues, yet their reliability under superficial code variations remains poorly understood. We evaluate whether coding agents that repair repository-level issues remain reliable when the surrounding codebase is rewritten into a semantically equivalent form. We introduce a random variant sampler that applies common semantics-preserving transformations (SPTs) - spanning control-flow rewrites, dead-code injection, and identifier renaming - to produce perturbed variants. We evaluate two agentic scaffolds (mini-SWE agent and OpenCode) each backed by one of four frontier models (Claude Opus 4.5, Kimi K2.5, MiniMax M2.5, and Qwen 3.6-27B) across instances drawn from SWE-bench Verified and SWE-bench Pro. For each instance, the agent is run multiple times on the unperturbed and perturbed variants, yielding paired resolve-rate estimates that isolate the perturbation effect from intri
    
[^34]: REFORGE：一种用于评测大语言模型在反编译二进制函数命名中逆向工程能力的方法

    REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming

    [https://arxiv.org/abs/2607.07738](https://arxiv.org/abs/2607.07738)

    本文提出REFORGE——一个带溯源追踪的评测流水线，通过八门置信度漏斗将编译器优化下的二进制-源码对齐不确定性量化分级，从而解决了LLM逆向工程能力公平评估中函数级真值可靠性不足这一核心障碍。

    

    大语言模型（LLM）正越来越多地被应用于逆向工程任务，最近的威胁情报报告显示它们已在实际的进攻性安全工作流程中运行。然而，关于其能力的宣称已超出了我们衡量该能力的水准。现有的LLM辅助二进制分析基准将函数级真值的构建视为已解决的前处理步骤，并在报告准确率时未披露有多少函数可被可靠地评估。我们认为，公平评估的主要障碍并非模型能力，而是编译器优化下二进制代码与源码对齐的可靠性。本文提出了Reforge，这是一个带有溯源追踪的流水线，它通过编译、DWARF与语法提取、对齐以及反编译等步骤，从C源代码构建函数级真值，并将对齐不确定性操作化为一个八门置信度漏斗，分为三个层级……

    arXiv:2607.07738v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly applied to reverse-engineering tasks, and recent threat-intelligence reporting shows them operating inside live offensive-security workflows. Claims about their capability, however, outpace our ability to measure it. Existing benchmarks for LLM-assisted binary analysis treat the construction of function-level ground truth as a solved pre-processing step and report accuracy without disclosing how many functions were reliably evaluable. We argue that the principal obstacle to fair evaluation is not model capability but the reliability of binary-to-source alignment under compiler optimization. This paper presents Reforge, a provenance-tracked pipeline that constructs function-level ground truth from C source through compilation, DWARF and syntactic extraction, alignment, and decompilation, and that operationalizes alignment uncertainty as an eight-gate confidence funnel with three-tier
    
[^35]: 智能体安全与监管现实的交汇——受监管金融系统中自主智能体威胁与控制的从业者系统化梳理

    Agent Security Meets Regulatory Reality -- A Practitioner Systematization of Autonomous-Agent Threats and Controls in Regulated Financial Systems

    [https://arxiv.org/abs/2606.29142](https://arxiv.org/abs/2606.29142)

    该论文基于生产环境证据，将六类自主智能体威胁映射到美欧金融监管的具体控制义务，提炼出五个运行时控制不变量及其形式化模型与可审计执行机制，弥合了智能体安全研究与金融监管合规实践之间的鸿沟。

    

    大型语言模型智能体正在进入受监管的金融系统，然而针对其攻击面的研究仍主要停留在实验室阶段，面向从业者的指导既未经同行评审，也未与正式的威胁模型建立联系。基于生产环境的实证证据，我们将六类智能体威胁——即提示注入、身份与授权、行为可审计性、工具滥用、数据驻留以及边界策略执行——映射到美国和欧盟金融监管所规定的具体控制义务（ECOA与Regulation B、欧盟AI法案、GDPR第22条，以及FINRA的2026年智能体指导意见），并阐明了法律责任如何提升这些威胁在运营层面的重要性。我们将这些监管义务提炼为五个运行时控制不变量，并给出在可审计的架构边界上强制执行这些不变量的参考算法。我们构建了一个形式化模型，将每个不变量表达为基于案例绑定事件链的谓词，并证明了其可审计性……

    arXiv:2606.29142v2 Announce Type: replace-cross  Abstract: Large language model agents are entering regulated financial systems, yet research on their attack surface remains predominantly laboratory-based, and practitioner guidance is neither peer-reviewed nor connected to a formal threat model. Using production evidence, we map six agentic threat categories, namely prompt injection, identity and authorization, action auditability, tool abuse, data residency, and boundary policy enforcement, to the specific control obligations imposed by US and EU financial regulation (ECOA and Regulation B, EU AI Act, GDPR Article 22, and FINRA's 2026 agent guidance), showing how legal accountability increases their operational significance. We distill these obligations into five runtime control invariants with reference algorithms that enforce them at auditable architectural boundaries. We develop a formal model expressing each invariant as a predicate over a case-bound event chain and prove the audi
    
[^36]: 机械式与智能体式混合的规约推断

    Hybrid Mechanical and Agentic Specification Inference

    [https://arxiv.org/abs/2605.10005](https://arxiv.org/abs/2605.10005)

    该论文将机械式最弱前置条件（WP）分析与LLM智能体驱动的规约推断相结合，集成到 Move Prover 中自动为 Move 智能合约推导规约，在 Aptos 框架的无规约公开函数上实现了超过 90% 的任务完成率。

    

    形式化验证的采用长期以来一直受到人工编写规约成本的阻碍。为了证明高层属性，人们还必须为辅助函数编写契约，而其中许多是机械性的且处于低层次。规约推断可以从程序中自动推导出此类契约。我们针对智能合约语言 Move 提出了一种将机械式推断与智能体式推断相结合的混合方法，并将其集成到其验证器 Move Prover 中。一个大语言模型（LLM）智能体通过一个调用 Move Prover 的工具来驱动推断与验证，并通过最弱前置条件（WP）分析从代码中推导规约。我们在 26 个目标上评估了该方法，其中 24 个提取自 Aptos 框架中没有规约的公开 Move 函数，测试使用了同一模型家族的前沿模型与中端模型（Sol 5.6 和 Terra 5.6）。无论是否使用 WP 分析，智能体都完成了超过 90% 的运行任务，因此智能体式推断……

    arXiv:2605.10005v3 Announce Type: replace-cross  Abstract: Adoption of formal verification has long been hindered by the cost of manual specification. To prove high-level properties, one must also write contracts for supporting functions, many of which are mechanical and low-level. Specification inference automatically derives such contracts from the program. We present a hybrid of mechanical and agentic inference for Move, a smart contract language, integrated into its verifier, the Move Prover. A large language model (LLM) agent drives inference and verification through a tool that calls the Move Prover and derives specifications from code by weakest-precondition (WP) analysis. We evaluate our approach on 26 targets, 24 of them extracted from public Move functions of the Aptos framework that have no specifications, under a frontier model and a mid-tier model of the same family (Sol 5.6 and Terra 5.6). With and without WP, the agent completes over 90% of its runs, so agentic inference
    
[^37]: 从对话到评测：基于 SimpleDevQA 对大语言模型的开发知识进行基准测试

    From Conversation to Evaluation: Benchmarking LLMs on Development Knowledge via SimpleDevQA

    [https://arxiv.org/abs/2512.08867](https://arxiv.org/abs/2512.08867)

    本文通过分析真实用户与大语言模型的对话并设计三阶段转换流水线，构建了源自真实对话的多语言开发知识问答基准 SimpleDevQA，弥补了现有基准知识范围有限且缺乏真实用户提问的不足。

    

    开发知识问答（Dev Knowledge QA）任务旨在为软件开发过程中寻求知识的提问提供准确的自然语言回答。为了探究开发知识问答在 AI 辅助软件开发中的重要性及其被探索研究的程度，我们对来自 WildChat 的真实用户与大语言模型对话进行了初步分析。我们的研究结果表明，开发知识问答在真实软件开发场景中扮演着重要角色，而这些原始对话无法被直接用于构建开发知识问答基准。现有的开发知识问答基准在开发知识范围上较为有限，且往往并非基于真实用户提问构建。为弥补这一空白，我们设计了一个三阶段流水线，将真实世界的对话转化为简单的开发知识问答对。通过该流水线，我们构建了 SimpleDevQA——一个多语言的开发知识问答基准。

    arXiv:2512.08867v3 Announce Type: replace  Abstract: The Development Knowledge Question Answering (Dev Knowledge QA) task aims to provide accurate natural language answers to knowledge-seeking questions during software development. To investigate the importance of Dev Knowledge QA in AI-assisted software development and the extent to which it has been explored, we conduct a preliminary analysis of real user-LLM dialogues from WildChat. Our findings indicate that Dev Knowledge QA plays a significant role in real-world software development scenarios, and these raw dialogues cannot be directly used to construct a Dev Knowledge QA benchmark. Existing Dev Knowledge QA benchmarks are limited in development knowledge scope and often not built from real user queries. To bridge this gap, we design a three-phase pipeline that transforms real-world dialogue into simple development knowledge-seeking QA pairs. Through this pipeline, we introduce SimpleDevQA, a multilingual Dev Knowledge QA benchmar
    
[^38]: 一种基于图神经网络（GNN）的代码坏味检测与重构机会识别方法

    A GNN-Based Approach for Code Smell Detection and Refactoring Opportunity Identification

    [https://arxiv.org/abs/2511.12069](https://arxiv.org/abs/2511.12069)

    该论文提出了一种基于图神经网络的深度学习方法，通过在类级别和方法级别构建代码图来检测代码坏味并识别重构机会，从而克服了传统方法依赖人工规则、机器学习方法训练数据受限以及大语言模型计算成本过高的局限性。

    

    代码坏味（Code Smells）指示了潜在的设计或实现问题，可能会对软件的可维护性产生负面影响。在过去几十年中，已有大量针对代码坏味重构的方法被提出，包括基于软件度量、预定义规则、机器学习/深度学习（ML/DL）以及近期基于大语言模型（LLM）的方法。然而，传统的基于度量和规则的方法严重依赖人工定义的标准，而机器学习/深度学习方法往往受限于训练数据的可获得性和模型架构的设计。尽管大语言模型提供了先进的代码理解能力，但它们通常会产生高昂的计算和时间成本，尤其是在应用于大规模软件系统时。在本研究中，我们提出了一种基于图的深度学习方法用于代码坏味重构。具体而言，我们在类级别和方法级别构建代码图，并采用图……（原文摘要在此处截断）

    arXiv:2511.12069v5 Announce Type: replace  Abstract: Code smells indicate latent design or implementation problems that may negatively affect software maintainability. Over the past decades, numerous approaches have been proposed for code smell refactoring, including those based on software metrics, predefined rules, machine learning/deep learning (ML/DL), and, more recently, large language models (LLMs). However, traditional metric- and rule-based approaches rely heavily on manually defined criteria, while ML/DL approaches are often constrained by the availability of training data and the design of model architectures. Although LLMs provide advanced code understanding capabilities, they typically incur substantial computational and time costs, particularly when applied to large-scale software systems. In this study, we propose a graph-based deep learning approach for code smell refactoring. Specifically, we construct code graphs at both the class and method levels and employ graph- an
    
[^39]: BigCodeArena：通过代码执行揭示代码生成中更可靠的人类偏好

    BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution

    [https://arxiv.org/abs/2510.08697](https://arxiv.org/abs/2510.08697)

    该论文提出了 BigCodeArena，一个基于实时执行环境的开放式代码生成人类评估平台，通过让人类与 LLM 生成代码的执行过程和结果进行交互，从而揭示出更可靠的人类偏好。

    

    众包模型评估平台（如 Chatbot Arena）能够从人类视角实时评估模型响应的质量。在编程领域，人工检查大语言模型（LLM）生成内容的质量极具挑战性，因为这需要理解大段原始代码并仔细模拟代码执行。为此，我们推出了 BigCodeArena，一个开放的代码生成人类评估平台，其背后由全面且实时的执行环境支持。BigCodeArena 构建于 Chatbot Arena 之上，能够执行 LLM 生成的代码，并允许人类与执行过程和执行结果进行交互。我们在 10 个广泛使用的 LLM 上收集了超过 14,000 个以代码为中心的原始对话会话，涵盖 10 种语言和 8 类执行环境。在这些对话中，我们识别出超过 4,700 个包含成对人类偏好的多轮样本。

    arXiv:2510.08697v3 Announce Type: replace-cross  Abstract: Crowdsourced model evaluation platforms, such as Chatbot Arena, enable real-time evaluation from human perspectives to assess the quality of model responses. In the coding domain, manually examining the quality of LLM-generated content is extremely challenging, as it requires understanding long chunks of raw code and deliberately simulating code execution. To this end, we introduce BigCodeArena, an open human evaluation platform for code generation backed by a comprehensive and on-the-fly execution environment. Built on top of Chatbot Arena, BigCodeArena enables the execution of LLM-generated code and allows humans to interact with the execution process and outcomes. We collected over 14,000 raw code-centric conversation sessions across 10 widely used LLMs, spanning 10 languages and 8 types of execution environments. Among these conversations, we identified more than 4,700 multi-turn samples with pairwise human preferences. Fur
    
[^40]: 使用 CTF-Dojo 训练语言模型智能体发现漏洞

    Training Language Model Agents to Find Vulnerabilities with CTF-Dojo

    [https://arxiv.org/abs/2508.18370](https://arxiv.org/abs/2508.18370)

    提出了 CTF-Dojo——首个面向 LLM 训练的大规模可验证执行环境，包含 658 个容器化 CTF 挑战，并配套自动化流水线 CTF-Forge，可数分钟内将公开工件转化为即用型执行环境，仅需 486 条高质量轨迹即可训练出擅长漏洞发现的智能体。

    

    大型语言模型（LLM）在可执行运行时环境中训练时展现出了卓越的能力，尤其通过可验证的反馈循环在软件工程任务中表现突出。然而，可扩展且可泛化的基于执行的环境仍然稀缺，限制了训练更强机器学习智能体的进展。我们提出了 CTF-Dojo，这是首个专为训练具有可验证反馈的 LLM 而设计的大规模可执行运行时环境，包含 658 个功能完整的夺旗（CTF）式挑战，并以 Docker 容器化方式确保可重现性。为了在没有人工干预的情况下实现快速扩展，我们开发了 CTF-Forge，这是一个自动化流水线，能够在几分钟内将公开可用的工件转换为可直接使用的执行环境，省去了传统上需要数周专家配置的过程。我们仅使用来自 CTF-D（原文在此处截断）的 486 条高质量、经过执行验证的轨迹对基于 LLM 的智能体进行训练……

    arXiv:2508.18370v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-D
    
[^41]: 识别与缓解大型语言模型中的API误用

    Identifying and Mitigating API Misuse in Large Language Models

    [https://arxiv.org/abs/2503.22821](https://arxiv.org/abs/2503.22821)

    该论文对三种主流大语言模型生成代码中的API误用模式进行了大规模研究，通过人工标注识别出四类常见误用类型，并针对幻觉和意图错位等问题提出了缓解方案。

    

    大型语言模型（LLM）生成代码中的API误用正成为软件开发中一个严重且日益加剧的挑战——尽管LLM展现出令人印象深刻的代码生成能力，但其与复杂库API的交互往往容易出错，可能导致软件故障和安全漏洞。在本文中，我们针对LLM生成代码中的API误用模式开展了一项大规模研究，通过分析Python和Java两种语言中的方法选择和参数使用情况，使用了三个具有代表性的LLM：StarCoder-7B、Qwen2.5-Coder-7B和GitHub Copilot。基于对3,209个方法级误用和3,492个参数级误用的大量人工标注，我们在已有API误用分类法的基础上进行扩展和细化，识别并归纳出四种反复出现的误用类型。对这三个LLM的评估揭示了API使用中持续存在的挑战，尤其是幻觉和意图错位问题。为解决这些问题，我们提出了Dr…（原文摘要在此处截断）

    arXiv:2503.22821v3 Announce Type: replace  Abstract: API misuse in code generated by large language models (LLMs) presents a serious and growing challenge in software development, as although LLMs demonstrate impressive code generation capabilities, their interactions with complex library APIs are often error-prone and can lead to software failures and vulnerabilities. In this paper, we conduct a large-scale study of API misuse patterns in LLM-generated code by analyzing both method selection and parameter usage across Python and Java, using three representative LLMs: StarCoder-7B, Qwen2.5-Coder-7B, and GitHub Copilot. Based on extensive manual annotation of 3,209 method-level and 3,492 parameter-level misuses, we identify and categorize four recurring misuse types by building on and refining prior API misuse taxonomies. Our evaluation of the three LLMs reveals persistent challenges in API usage, particularly hallucination and intent misalignment. To address these issues, we propose Dr
    
[^42]: 混合量子-经典架构中真实故障的分类体系

    A Taxonomy of Real Faults in Hybrid Quantum-Classical Architectures

    [https://arxiv.org/abs/2502.08739](https://arxiv.org/abs/2502.08739)

    本文通过实证分析GitHub上超过5000个已关闭issue，提出并验证了首个混合量子-经典（NISQ）架构中真实故障的分类体系，并发布了包含133个标注真实故障的数据集。

    

    随着混合量子-经典架构（Hybrid Quantum-Classical architectures）的日益普及，特别是含噪声中等规模量子（NISQ）架构的广泛应用，需要针对其特定故障量身定制的质量保证方法。在本研究中，我们提出了混合量子-经典架构中故障的分类体系，并附带一个包含已识别类别中真实故障的数据集。为实现这一目标，我们对开源代码库中已修复的故障进行了实证分析。我们分析了GitHub上超过5000个已关闭的issue，并根据严格定义的纳入标准预先筛选出529个。我们最终选取了133个故障，并围绕故障的症状和成因对其进行了标注。两位作者对每个故障的分类和标签进行了交叉验证。最终，我们构建了混合量子-经典架构中真实故障的分类体系。随后，我们通过对十一位开发者的访谈验证了该分类体系的有效性。

    arXiv:2502.08739v4 Announce Type: replace  Abstract: With the popularity of Hybrid Quantum-Classical architectures, particularly noisy intermediate-scale quantum (NISQ) architectures, comes the need for quality assurance methods tailored to their specific faults. In this study, we propose a taxonomy of faults in Hybrid Quantum-Classical architectures accompanied by a dataset of real faults in the identified categories. To achieve this, we empirically analysed open-source repositories for fixed faults. We analysed over 5000 closed issues on GitHub and pre-selected 529 of them based on rigorously defined inclusion criteria. We selected 133 faults that we labelled around symptoms and the origin of the faults. We cross-validated the classification and labels assigned to every fault between two of the authors. As a result, we introduced a taxonomy of real faults in Hybrid Quantum-Classical architectures. Subsequently, we validated the taxonomy through interviews conducted with eleven develo
    
[^43]: BigCodeBench：利用多样化函数调用与复杂指令对代码生成进行基准测试

    BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions

    [https://arxiv.org/abs/2406.15877](https://arxiv.org/abs/2406.15877)

    BigCodeBench是一个新的代码生成基准测试，通过要求模型调用多样化函数库并准确理解复杂指令，来评估大语言模型解决实际任务的能力。

    

    大语言模型（LLM）通过Python代码的最新进展极大地赋能了任务自动化，其任务范围涵盖从软件工程开发到通用推理。虽然现有基准测试表明LLM能够像人类开发者一样使用程序解决问题，但其大多数评估仅限于简短且独立的算法任务或单一函数调用。解决具有挑战性和实用性的任务，需要具备将多样化函数调用作为工具的能力，以高效实现数据分析和网页开发等功能。此外，使用多个工具解决任务还需要通过准确理解复杂指令来进行组合推理。同时满足这两个特性对LLM而言是一项巨大挑战。为了评估LLM通过程序解决具有挑战性和实用性任务的能力，我们推出了BigCodeBench这一基准测试……

    arXiv:2406.15877v5 Announce Type: replace-cross  Abstract: Task automation has been greatly empowered by the recent advances in Large Language Models (LLMs) via Python code, where the tasks ranging from software engineering development to general-purpose reasoning. While current benchmarks have shown that LLMs can solve tasks using programs like human developers, the majority of their evaluations are limited to short and self-contained algorithmic tasks or standalone function calls. Solving challenging and practical tasks requires the capability of utilizing diverse function calls as tools to efficiently implement functionalities like data analysis and web development. In addition, using multiple tools to solve a task needs compositional reasoning by accurately understanding complex instructions. Fulfilling both of these characteristics can pose a great challenge for LLMs.To assess how well LLMs can solve challenging and practical tasks via programs, we introduce BigCodeBench, a benchm
    
[^44]: Astraios：参数高效的指令微调代码大语言模型

    Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models

    [https://arxiv.org/abs/2401.00788](https://arxiv.org/abs/2401.00788)

    该论文提出了Astraios套件——包含28个采用7种微调方法、4种规模（最大160亿参数）指令微调的代码大语言模型，系统研究发现全参数微调总体性能最佳，LoRA通常提供最有利的成本-性能权衡，且PEFT方法的效果随模型规模显著变化。

    

    大语言模型（LLM）全参数微调（FFT）的高昂成本催生了一系列参数高效微调（PEFT）方法。然而，目前尚不清楚哪些方法在不同模型规模下能够提供最佳的成本-性能权衡。我们提出了Astraios，这是一个包含28个经过指令微调的OctoCoder模型的套件，采用了7种微调方法和4种模型规模，最大参数量达160亿。通过对涵盖代码理解和代码生成任务的5个任务和8个不同数据集的调查研究，我们发现FFT通常在所有模型规模下都能带来最佳的下游性能，而PEFT方法的有效性则随模型规模的不同而有显著差异。LoRA通常在成本和性能之间提供最有利的权衡。进一步研究这些方法对模型鲁棒性和代码安全性的影响后发现，较大的模型往往表现出较低的鲁棒性。

    arXiv:2401.00788v2 Announce Type: replace-cross  Abstract: The high cost of full-parameter fine-tuning (FFT) of Large Language Models (LLMs) has led to a series of parameter-efficient fine-tuning (PEFT) methods. However, it remains unclear which methods provide the best cost-performance trade-off at different model scales. We introduce Astraios, a suite of 28 instruction-tuned OctoCoder models using 7 tuning methods and 4 model sizes up to 16 billion parameters. Through investigations across 5 tasks and 8 different datasets encompassing both code comprehension and code generation tasks, we find that FFT generally leads to the best downstream performance across all scales, and PEFT methods differ significantly in their efficacy based on the model scale. LoRA usually offers the most favorable trade-off between cost and performance. Further investigation into the effects of these methods on both model robustness and code security reveals that larger models tend to demonstrate reduced robu
    

