# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Higher-Scoring Coding Models Did Not Get More Work Done](https://arxiv.org/abs/2611.23216) | 该研究利用美国出口管制导致Claude Fable 5被停用19天的自然实验，发现在编码基准测试中得分更高的模型（Fable 5）相比得分较低的模型（Opus 4.8）并未给用户带来更高的实际工作产出，揭示了基准分数与真实生产力之间的脱节。 |
| [^2] | [Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU](https://arxiv.org/abs/2611.23205) | 该论文在笔记本电脑集成GPU上用66个样例通过SFT结合LoRA/QLoRA微调Gemma 4边缘模型（E2B和E4B）用于MATLAB信号处理代码生成，使通过率显著提升（如E4B从32/66升至47/66），且发现微调主要修正DSP惯例错误而非信号构建错误，运行时代价仅为17-19%的生成速度损失。 |
| [^3] | [Identifying and Repairing Energy Smells with Agent Skills](https://arxiv.org/abs/2611.23203) | 本文提出 DR. GREENS 智能体技能，基于 12 种能耗坏味道与 65 种根因的分类体系及 230 个实测修复案例，在开发者编码环境中检测并修复 Python 代码的能耗坏味道，并通过静态分析器与独立验证器确保修复保持行为不变且确实节能。 |
| [^4] | [stable-envs-v1: Fast and Reproducible Environments with Controllable Variations](https://arxiv.org/abs/2611.23151) | stable-envs是一个开源的GPU原生渲染库，通过直接在GPU上渲染图像观测并支持可控的视觉变化，实现了高达1.33倍的端到端训练加速且不损失性能，同时支持可复现的大规模评估。 |
| [^5] | [Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance](https://arxiv.org/abs/2611.23142) | 该论文提出并实际部署了一个以"TriAgent"分诊智能体为核心的多智能体系统，通过持续挖掘客服工单和代码-文档不一致等显性与隐性信号，自动检测并修复客户文档与技术实现之间的漂移。 |
| [^6] | [BugGen 2.0: Rethinking the Roles of LLMs for Scalable RTL Bug Synthesis](https://arxiv.org/abs/2611.23091) | BugGen 2.0重新划分了LLM与确定性算法的分工，由算法负责RTL缺陷的结构分析、目标选择与验证，LLM专注于局部语义改写，从而实现高效可靠的大规模RTL缺陷数据集自动生成。 |
| [^7] | [RagHarnessDebugger: Jev-Guided Agentic RAG in Contract Inference Repair Auditing](https://arxiv.org/abs/2611.22991) | 提出了RagHarnessDebugger框架，通过Jev引导的测试工具记录并审计智能体RAG在契约推理修复过程中的全部决策，从而暴露出被准确率等聚合指标所掩盖的失败模式。 |
| [^8] | [Think Before You Code: A Benchmark for Testing LLMs on Software Engineering Deliberation](https://arxiv.org/abs/2611.22905) | 该研究提出了TBYC基准框架，通过将GitHub议题讨论转化为结构化的工程思考产物并与模型生成的“思考”进行对比评估，来衡量代码大语言模型能否像人类工程师一样审慎地思考软件工程问题。 |
| [^9] | [Stopping Is Not the Limit: Understanding and Eliciting Agent Capability in Long-Horizon Vulnerability Exploitation](https://arxiv.org/abs/2611.22888) | 该论文发现漏洞利用智能体在首次停止后继续执行仍能带来额外收益，并提出上下文压缩方法 SCoRe 以激发智能体在长程漏洞利用任务中的潜在能力。 |
| [^10] | [When Passing Checks Do Not Establish Completion: Evidence Interpretation in Coding Agents](https://arxiv.org/abs/2611.22835) | 该研究通过因子实验与118次修复事件分析证明：编程智能体中检查通过并不等同于任务完成，正确报告的执行事实未必能导出所请求的有限结论，部分回复还存在误报事实或超出授权范围的问题。 |
| [^11] | [Python Code Generation for Isabelle/HOL](https://arxiv.org/abs/2611.22613) | 本文为 Isabelle 代码生成器开发了 Python 后端，使经过形式化验证的算法可以直接在 Python 生态中使用，或作为参考实现对现有 Python 库进行差分测试。 |
| [^12] | [ReBoT-MiCS: Reliability-Aware Bounding Time for Enhancing Timing Behavior of Embedded Mixed-Criticality Systems](https://arxiv.org/abs/2611.22588) | 该论文提出ReBoT-MiCS方法，通过可靠性感知的低关键性WCET约束时间确定机制，在联合权衡QoS、模式切换与可靠性约束的前提下，增强嵌入式混合关键性系统的时序行为。 |
| [^13] | [Coordinating integration of multidisciplinary design analysis and optimisation codes with design structure matrices](https://arxiv.org/abs/2611.22522) | 本文将核聚变反应堆的集成设计问题纳入多学科设计分析与优化（MDAO）框架，引入设计结构矩阵（DSM）来可视化和协调集成分析与优化代码的耦合架构，从而提升计算效率。 |
| [^14] | [Missing Security-Fix Backporting for GitHub Actions Workflows](https://arxiv.org/abs/2611.22519) | 本文首次大规模实证研究了 GitHub Actions 工作流安全修复在发布分支上回移植缺失的问题，发现近一半（224,392/474,490）已审计的发布分支仍存在已在默认分支修复的已知漏洞。 |
| [^15] | [VoxelSBFL: Voxel-Level Spectrum-Based Failure Prediction for Vision-Language-Action Models](https://arxiv.org/abs/2611.22453) | 该论文提出VoxelSBFL，创新性地将软件工程中的频谱故障定位（SBFL）技术黑盒迁移到机器人测试领域，以工作空间体素取代程序元素作为分析单元，利用TCP轨迹的体素覆盖信息与执行结果构建可疑度地图，从而在无需模型内部信息的情况下实现体素级的失败预测。 |
| [^16] | [Adaptive Budget Allocation for Smart-Contract Fuzzing: Why Online Feedback Fails to Improve Vulnerability Triage](https://arxiv.org/abs/2611.22430) | 该论文发现智能合约模糊测试中的在线反馈策略均无法超越均匀预算分配，并提出一种基于离线概率曲线预测首次报告概率、按每次执行预测增益分配预算的分配器来改进漏洞分类效果。 |
| [^17] | [BehaVerify: A Formal Verification Tool for Behavior Trees](https://arxiv.org/abs/2611.22403) | BehaVerify是一个面向行为树及神经符号行为树的形式化验证工具箱，通过带形式语法与语义的DSL、时序逻辑规范以及高效的nuXmv编码实现快速验证，并能生成可集成于PyTrees和ROS的Python/Haskell实现。 |
| [^18] | [RyBench: Runtime Verification Benchmarking Framework over Publish/Subscribe Networks](https://arxiv.org/abs/2611.22401) | 本文提出了RyBench——一个基于发布/订阅的基准测试框架，它作为合成事件发布者替代被监控系统，并通过将事件生成、序列化和传输解耦为可灵活组合的独立组件，为运行时验证工具的吞吐量与判决延迟评估提供了统一的标准化测试平台。 |
| [^19] | [Formal Analysis of a Fault-Tolerant Extension of the ROSACE Flight Controller](https://arxiv.org/abs/2611.22394) | 本文利用COMPASTA工具为ROSACE飞行控制器基准扩展了基于三模冗余的FDIR容错组件，并通过形式化建模与故障注入对其运行时故障抵御能力进行分析与验证。 |
| [^20] | [Agent Trajectories as Evidence for Software Quality Assurance](https://arxiv.org/abs/2611.22323) | 该论文提出将智能体轨迹与规格说明及AI生成产物结合分析，作为针对性软件质量保证的证据，以优先审查智能体影响重大的决策、暴露隐式假设并持续改进开发实践。 |
| [^21] | [When the World Is the Test Suite: Agentic Intelligence for the Physical World](https://arxiv.org/abs/2611.22291) | 本文提出将物理世界视为程序的测试套件，系统考察了面向物理世界的智能体智能这一新兴领域，围绕智能体如何为物理世界编写程序、如何测试这些程序以及如何从测试结果中学习三个核心问题展开论述。 |
| [^22] | [OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions](https://arxiv.org/abs/2611.22262) | 该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。 |
| [^23] | [Readie: Low-Latency Code Sandboxes for AI Agents](https://arxiv.org/abs/2611.22171) | 提出Readie系统，通过从历史执行中学习一组小型、快速初始化的启动快照集合，使其共同覆盖大部分AI代理生成的代码，从而解决现有代码沙箱启动缓慢的问题。 |
| [^24] | [What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super](https://arxiv.org/abs/2611.22161) | 通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。 |
| [^25] | [Operating an Agent-Driven Dataset Factory: Discovery, Construction, and Public Release](https://arxiv.org/abs/2611.21957) | 该论文部署了一个由语言模型智能体驱动、确定性软件支撑的自动化“数据集工厂”，在81.53小时内以约2.74美元的单次发布成本持续产出14个预测数据集，并通过审查暴露了构建规则与数据总体声明不一致的治理问题。 |
| [^26] | [EmbedGym: Letting Coding Agents Develop and Debug Embedded Firmware Without Physical Hardware](https://arxiv.org/abs/2611.21845) | EmbedGym是一个交互式虚拟硬件环境，通过指令级MCU仿真器结合41个外设模型、模拟电路仿真器及虚拟WiFi/BLE端点，让编码代理无需物理硬件即可开发和调试嵌入式固件。 |
| [^27] | [Can Coding Agents Write and Transpile Unit Tests](https://arxiv.org/abs/2611.21797) | 本研究对三个编程智能体进行了基准测试，发现其编写的单元测试在覆盖率上远超人类测试套件，但测试质量随项目规模增大而急剧下降，且跨语言测试翻译仍不一致且不可靠。 |
| [^28] | [Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict](https://arxiv.org/abs/2611.21784) | 本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。 |
| [^29] | [A Survey of Modeling Languages for Automotive Digital Twins](https://arxiv.org/abs/2611.21636) | 本文通过对德国汽车行业专家的问卷调查，识别了汽车数字孪生的期望用途与能力、汽车开发各阶段相关的建模技术以及非正式模型的作用，弥合了学术文献与工业实践之间的差距。 |
| [^30] | [We Fixed the Scorer First, and Cluster Correction Killed Our Own Planted-Defect Result in AI Code Review](https://arxiv.org/abs/2611.21635) | 本文的核心贡献在于：修正评分器后，通过精确置换检验与Holm多重校正证明原先看似显著的AI代码审查植入缺陷检测结果并不成立，并指出提升检测统计功效需要增加测试案例数（12-25个）而非增加重复次数。 |
| [^31] | [Trusting Trust in AI Software Engineering Agents: The Why, What, and How?](https://arxiv.org/abs/2611.21634) | 该论文的核心贡献是提出了一个结构化信任分析框架，将AI软件工程代理的信任分解为“为什么信任（信任场景）、信任什么（信任配置）、信任多少”三个分析性问题，以克服将信任视为单一可测量属性的局限性。 |
| [^32] | [Measuring Human-AI Alignment as an Early Indicator for Effective Collaboration in Hybrid SE Teams](https://arxiv.org/abs/2611.21633) | 本文提出开发一套软件工程专用的人机对齐静态测量指标体系，借鉴软件架构、组织心理学和人机交互原理，并利用声明式业务流程监控语言构建静态评估工具，作为评估混合人机团队有效协作的早期指标。 |
| [^33] | [Fragmented Markers, Mixed Results: A Systematic Review of AI Coding Assistants and Developer Productivity](https://arxiv.org/abs/2611.21632) | 该论文通过一项分阶段的系统性文献综述，指出现有AI编程助手对开发者生产力影响研究的核心问题在于生产力衡量指标碎片化、缺乏可比性，并基于SPACE框架将分析范围扩展至以人为本的生产力相关成果。 |
| [^34] | [A Model-Driven Automation Pipeline for Building Operation](https://arxiv.org/abs/2611.21631) | 该论文提出了一个基于模型驱动工程的楼宇控制系统自动化流水线，利用两种领域特定建模语言从平台无关模型自动生成并部署可执行制品，将设计、编程与自动调试整合为四步工作流，以减少人工投入和配置错误。 |
| [^35] | [Reusing Battle-tested Code: A Systematic Literature Review of Automated Software Transplantation](https://arxiv.org/abs/2611.21630) | 本文通过系统性文献综述评估了自动化软件移植领域的最新进展，识别出27种现有方法，发现尽管代码适配已高度自动化，但器官定位与测试套件增强等人工前置步骤仍是阻碍该技术走向主流工程实践的主要瓶颈。 |
| [^36] | [Usage Patterns of Programming Languages and Tools: A Systematic Literature Review](https://arxiv.org/abs/2611.21627) | 本系统性文献综述分析了2015-2024年间的37项研究，发现Java是工业界最流行的编程语言、Travis CI和GitHub Actions是最常用的CI工具，但针对工业界流行语言和工具（尤其是IDE使用）的研究仍然匮乏。 |
| [^37] | [X-LocVul: An Evidence-Propagation Audit of Cascaded C/C++ Vulnerability Detection, Localization, and Explanation](https://arxiv.org/abs/2611.21626) | X-LocVul对C/C++漏洞检测-定位-解释级联流水线进行了以测量为先的严格审计，揭示出级联系统在各阶段间传播的证据往往薄弱且不可靠：真实CWE监督与随机对照无统计学差异，且超过90%的生成语句无法精确匹配源代码，凸显了此类流水线在证据传播上的可信度问题。 |
| [^38] | [Evaluating AST-Derived Knowledge Graph Retrieval for AIOps Incident Diagnosis in Apache Kafka](https://arxiv.org/abs/2611.21625) | 本文在Apache Kafka事件诊断任务上系统评估了基于AST派生知识图谱的检索方法，发现仅向量检索已是强基线（Recall@20达0.956）且图增强方法难以显著超越，而分阶段图扩展可将重排序开销减少约52%。 |
| [^39] | [Trustworthy AI Teammates for Software Engineering: A Research Agenda for Lifecycle Context, Explainability, and Agentic Collaboration](https://arxiv.org/abs/2611.21624) | 该论文提出AI辅助软件工程的下一阶段应从被动式编程助手转向可信AI队友，并围绕生命周期感知、有原则的可解释性和智能体协作三大要求构建人机协同演进的研究议程。 |
| [^40] | [The Software Engineering Factory: Engineering Accountability and Control in the Age of Abundant Code](https://arxiv.org/abs/2611.21623) | 该论文提出“软件工程工厂”框架，利用以常规代码实现的确定性协议引擎将组织控制嵌入代码生产流程之中，从而在AI智能体大规模生成代码的时代实现可问责的代码验收，同时保留人类对重大决策的最终权威。 |
| [^41] | [On Improving the Adoption, Usability, and Retention of Static Application Security Testing (SAST) Tools](https://arxiv.org/abs/2611.21621) | 本论文通过对89项相关研究的文献综述识别出开发人员在使用SAST工具时的主要痛点（如误报等可用性障碍），并提出了提升静态应用安全测试工具的采用率、可用性与持续使用的方法。 |
| [^42] | [From AIOps Pipelines to Agentic Service Operations: A Survey of Capability, Authority, and Evidence](https://arxiv.org/abs/2611.21620) | 该综述提出以能力、运维权限和评估证据三个相互关联的视角来审视从传统AIOps流水线到智能体化服务运营的转变，并依据实际评估行为而非预期自主性对22个语言模型驱动的系统进行了审计。 |
| [^43] | [Stochastic Modeling of the Mean Time Between Software Failures: A review](https://arxiv.org/abs/2611.21619) | 本文综述并比较了基于非齐次泊松过程与传染机制的软件故障检测随机模型中故障间平均时间（MTBF）的推导方法，评估其在敏捷开发方法下的适用性，其中传染模型的故障率同时依赖于时间与已检测到的故障数量。 |
| [^44] | [The Defense Rests: A Courtroom-Inspired Multi-Agent LLM Framework for Automated Safety Case Generation](https://arxiv.org/abs/2611.21618) | 本文提出了一种受法庭审判启发的多智能体大语言模型框架，通过模拟检察官、辩护律师、法官和陪审团等角色协作，实现任务关键系统安全论证的自动化生成。 |
| [^45] | [A Chain-of-Thought Prompt Engineering Framework for Automated Technical Specification Generation in Enterprise Integration: A Real-World Case Study](https://arxiv.org/abs/2611.21617) | 该论文提出了一种基于思维链的提示工程框架，通过两阶段生成架构与六阶段提示流水线相结合，利用GitHub Copilot在企业集成项目中自动生成技术规范文档，并在将250多个企业集成迁移至Azure的真实工业场景中验证了其有效性。 |
| [^46] | [Revision-Consistent Code Context for Agents That Edit Their Environment](https://arxiv.org/abs/2611.21616) | 本文提出了一种代理可见的完整修订发布契约，用于解决编辑型代理在同时查询和修改代码库时出现的“混合修订代码上下文”问题，确保单次查询返回的所有视图（源代码、符号、图关系）都来自同一代码库版本。 |
| [^47] | [Wrong yet Confident: Automated Detection of Silent Failures in Location-Dependent Software](https://arxiv.org/abs/2611.21615) | 该论文提出一种名为“往返偏移”的自动化测试预言机，无需人工标注即可在部署规模上检测地理编码器的静默失败，并在真实食品援助部署数据中发现10.0%的坐标对所有地理编码平台均不可见。 |
| [^48] | [Does Your Agent Need to Know Its Budget? A Placebo-Controlled Experiment on LLM Coding Agents](https://arxiv.org/abs/2611.21614) | 该研究通过包含 5,835 次运行的安慰剂对照析因实验发现，对 LLM 编码智能体强制执行 token 预算会以任务解决率下降 20 个百分点为代价使 token 消耗约降低十二倍，而告知智能体预算的存在不仅毫无帮助，甚至可能轻微损害性能。 |
| [^49] | [Exploring Test Smells Across Programming Languages: A Systematic Mapping Study](https://arxiv.org/abs/2611.21613) | 该研究通过系统性映射方法分析了截至2025年8月的117篇论文，首次从编程语言视角系统审视测试坏味，填补了测试坏味与特定编程语言及测试框架关系研究的空白。 |
| [^50] | [A Systematic Survey of Decompilation: Mapping Classical and Machine-Learning Approaches](https://arxiv.org/abs/2611.21612) | 本文对1973至2026年的405项反编译研究进行了系统性综述，将经典方法、ML训练方法和ML提示方法映射到八阶段的流水线中心分类法中，并分析了116项基于机器学习的研究在技术设计与评估证据上的差异。 |
| [^51] | [Spec-Driven PBT: Deriving Semantic Invariants from Code Specifications for Agentic Property-based Testing](https://arxiv.org/abs/2611.15920) | 该论文提出规格驱动的属性测试方法，利用智能体编程中已有的自然语言规格说明（而非仅从代码实现）来推导语义不变量，从而更有效地测试AI生成的代码。 |
| [^52] | [OverServe: Resource Amplification Attacks on SaaS Agents via Tool--Prompt Co-Design](https://arxiv.org/abs/2611.14808) | 提出了OverServe多智能体攻击框架，通过协同设计MCP状态工具与完成确认提示词，诱导SaaS智能体在业务任务完成后仍反复检查未完成条件，从而放大对服务提供商资源的消耗。 |
| [^53] | [National and State-Level Approaches to Automated Vehicle Certification: A Comparative Framework for State Operational Governance](https://arxiv.org/abs/2611.08681) | 该论文构建了一个结构化比较框架，分析美国十个州对SAE L4-L5级自动驾驶系统的认证与治理方法，并结合欧盟、中国、日本和澳大利亚的国际实践，为北卡罗来纳州的自动驾驶监管战略提供决策参考。 |
| [^54] | [Persistent Recursive Worlds Enable Autonomous Software Evolution](https://arxiv.org/abs/2608.10450) | 本文提出EvoX Genesis框架，通过将软件表示为持久递归世界，使项目持久化而代理有限寿命，实现了无需人工干预的自主软件演化，并成功构建了大型编译器。 |
| [^55] | [A Grey Literature Review of Graduate Students' Research Practices and Skills in the LLM Era](https://arxiv.org/abs/2607.29519) | 该研究通过对Reddit讨论的灰色文献综述，识别出研究生使用LLM开展研究的七类实践和九类所需技能，为研究生培养项目教授、指导和评估LLM使用能力提供了依据。 |
| [^56] | [IH-Benchmark: A Conflict-Centered Benchmark for Instruction-Hierarchy Robustness in Large Language Models](https://arxiv.org/abs/2607.25987) | 提出了IH-Benchmark，一个以冲突为中心的基准测试，通过七个约束类别和统一的二元评估协议，系统评估大语言模型在直接系统-用户冲突及工具中介用户-工具冲突下的指令层级鲁棒性。 |
| [^57] | [Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?](https://arxiv.org/abs/2607.23624) | 该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。 |
| [^58] | [AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation](https://arxiv.org/abs/2605.12925) | 本文提出AgentLens过程级评估框架并发布AgentLens-Bench数据集，发现10.7%通过测试的智能体轨迹存在“幸运通过”问题（如回归循环、盲目重试、缺失验证等），证明仅以测试是否通过来评估软件工程智能体是不充分的。 |
| [^59] | [EmbeWebAgent: Embedding Web Agents into Custom Enterprise UIs](https://arxiv.org/abs/2602.14865) | EmbeWebAgent通过轻量级前端钩子和可复用的后端工作流，将网络智能体直接嵌入企业现有UI中，实现与技术栈无关的混合粒度操作，并借助MCP工具完成真实界面中的多步骤行为。 |

# 详细

[^1]: 更高得分的编码模型并没有完成更多工作

    Higher-Scoring Coding Models Did Not Get More Work Done

    [https://arxiv.org/abs/2611.23216](https://arxiv.org/abs/2611.23216)

    该研究利用美国出口管制导致Claude Fable 5被停用19天的自然实验，发现在编码基准测试中得分更高的模型（Fable 5）相比得分较低的模型（Opus 4.8）并未给用户带来更高的实际工作产出，揭示了基准分数与真实生产力之间的脱节。

    

    新的编码模型以编码基准测试的分数提升作为宣传卖点。然而，目前尚无研究检验在这些基准测试中得分更高的模型是否真的能完成更多实际工作。2026年6月12日，一项美国出口管制指令禁止访问Claude Fable 5达19天，迫使用户切换到其他模型。在此期间，大多数前Fable 5用户切换到了Opus 4.8（该模型在SWE-Bench Pro上比Fable 5低11分）。我们将这次停用视为一个自然实验：人员和项目保持不变，只有模型发生了变化，因此我们可以分离出Fable 5对用户生产力的影响。我们跟踪了297名Fable 5用户在停用前、停用期间和停用后的GitHub拉取请求（pull request），并以8,908名始终使用Opus 4.8的开发者作为对照组。此外，我们还在2026年停用前后的五次模型升级中重复了这一比较（涉及1,208名开发者和15,442个拉取请求）。

    arXiv:2611.23216v1 Announce Type: cross  Abstract: New coding models advertise themselves on score improvements of coding benchmarks. Yet no study has tested if a model that scores higher on these benchmarks gets more real work done. On June 12, 2026, a U.S. export control directive banned access to Claude Fable 5 for 19 days, which forced users to switch to other models. During this time, most former Fable 5 users switched to Opus 4.8 (11 points lower on SWE-Bench Pro compared to Fable 5). We treat this suspension as a natural experiment, where the people and projects stayed the same and only the model changed, so we could isolate the effect of Fable 5 on user productivity. We tracked GitHub pull requests of 297 Fable 5 users before, during, and after the suspension. We used 8,908 developers who stayed on Opus 4.8 all the time as the control group. We also repeated the comparison across five model upgrades before and after the suspension in 2026 (1,208 developers, 15,442 pull requests
    
[^2]: 在笔记本电脑集成GPU上使用SFT与QLoRA微调Gemma 4边缘模型以生成MATLAB信号处理代码

    Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU

    [https://arxiv.org/abs/2611.23205](https://arxiv.org/abs/2611.23205)

    该论文在笔记本电脑集成GPU上用66个样例通过SFT结合LoRA/QLoRA微调Gemma 4边缘模型（E2B和E4B）用于MATLAB信号处理代码生成，使通过率显著提升（如E4B从32/66升至47/66），且发现微调主要修正DSP惯例错误而非信号构建错误，运行时代价仅为17-19%的生成速度损失。

    

    能在笔记本电脑上运行的小型语言模型对于离线编写信号处理代码很有吸引力，但它们在数字信号处理（DSP）惯例方面会犯系统性错误。我们使用66个经过检查的样例（涵盖11个信号处理MATLAB子任务）对Gemma 4边缘模型E2B和E4B进行监督微调（SFT），在笔记本电脑的集成GPU（iGPU）上采用低秩适应（LoRA）和量化低秩适应（QLoRA），并通过OpenVINO部署带有适配器的8位整数（int8）模型。在由确定性检查器评分的保留测试变体上，E2B的通过率从18/66提升至37/66（LoRA）和37/66（QLoRA），E4B的通过率从32/66提升至47/66（QLoRA）。通过阅读全部159个失败程序发现，适配器消除了大多数惯例性错误（如滞后的符号、单位和数组形状），但很少修复构建信号和噪声方面的错误。在运行时应用适配器会使生成速度降低17-19%。

    arXiv:2611.23205v1 Announce Type: cross  Abstract: Small language models that run on a laptop are attractive for writing signal-processing code offline, but they make systematic errors on digital signal processing (DSP) conventions. We apply supervised fine-tuning (SFT) to the Gemma 4 edge models E2B and E4B with 66 checked examples covering 11 signal-processing MATLAB subtasks, using low-rank adaptation (LoRA) and quantized LoRA (QLoRA) on the integrated GPU (iGPU) of a laptop, and deploy the 8-bit integer (int8) models with the adapters through OpenVINO. On held-out variants scored by deterministic checkers, the pass rate rises from 18/66 to 37/66 (LoRA) and 37/66 (QLoRA) for E2B and from 32/66 to 47/66 for E4B (QLoRA). Reading all 159 failing programs shows that the adapters remove most errors of convention, such as the sign of a lag, units and array shapes, but few errors in building signals and noise. Applied at run time, the adapter costs 17-19% of the generation speed.
    
[^3]: 利用智能体技能识别与修复能耗坏味道

    Identifying and Repairing Energy Smells with Agent Skills

    [https://arxiv.org/abs/2611.23203](https://arxiv.org/abs/2611.23203)

    本文提出 DR. GREENS 智能体技能，基于 12 种能耗坏味道与 65 种根因的分类体系及 230 个实测修复案例，在开发者编码环境中检测并修复 Python 代码的能耗坏味道，并通过静态分析器与独立验证器确保修复保持行为不变且确实节能。

    

    能耗坏味道是指浪费能源的低效代码实现。一旦代码进入生产环境，修复这些坏味道的成本更高、风险更大，因此能效检查应当“左移”，即提前到开发生命周期中编写代码的阶段。现有的能耗坏味道工具依赖于固定的静态规则，且不测量节能效果；而基于大语言模型的优化器大多以执行时间为目标，很少将其修改建立在具体的坏味道定义之上。为解决这两方面的局限，我们提出了 DR. GREENS，一种在开发者编码环境中检测并修复 Python 代码能耗坏味道的智能体技能。该技能以我们提出的包含 12 种能耗坏味道和 65 种根因的分类体系为基础，并辅以来自 230 个实测案例的修复指南。该技能还封装了一个静态分析器和一个独立验证器，用以确认每次尝试的修复既保持了原有行为又实现了节能。我们评估了……（摘要原文在此处截断）

    arXiv:2611.23203v1 Announce Type: new  Abstract: Energy smells are inefficient implementations that waste energy. Once code reaches production, these smells become costlier and riskier to fix, so energy efficiency checks should shift left, i.e., move earlier in the development lifecycle, to where code is written. Existing energy smell tools rely on fixed static rules and do not measure energy savings, whereas LLM-based optimizers mostly target execution time and rarely ground their changes in concrete smell definitions. To address both limitations, we present DR. GREENS, an agent skill that detects and repairs energy smells in Python code within the developer's coding environment. The skill grounds the coding agent in our taxonomy of 12 energy smells and 65 root causes, augmented with repair guidance drawn from 230 measured cases. The skill also encapsulates a static analyzer and an independent verifier to confirm behavior preservation and energy saving for each attempted repair. We ev
    
[^4]: stable-envs-v1：具有可控变化的快速且可复现的环境

    stable-envs-v1: Fast and Reproducible Environments with Controllable Variations

    [https://arxiv.org/abs/2611.23151](https://arxiv.org/abs/2611.23151)

    stable-envs是一个开源的GPU原生渲染库，通过直接在GPU上渲染图像观测并支持可控的视觉变化，实现了高达1.33倍的端到端训练加速且不损失性能，同时支持可复现的大规模评估。

    

    训练世界模型依赖于处理大量高维数据，这些数据通常需要被存储、读入内存，并从内存移动到计算单元。原则上，GPU原生渲染避免了将大型图像张量从内存移动到计算核心的过程，从而实现更好的GPU利用率以及训练加速。按需渲染还允许通过即时应用视觉变化来控制感知外观，从而在训练中实现更高的多样性。为此，我们引入了stable-envs，一个用于快速GPU原生渲染的开源库，可在多个模拟环境中应用可控变化。通过直接在GPU上渲染图像观测，stable-envs实现了高达1.33倍的端到端训练加速，且不损失性能。除了更快的训练运行之外，stable-envs还通过批处理物理实现了可复现的大规模评估……

    arXiv:2611.23151v1 Announce Type: new  Abstract: Training World Models (WMs) relies on processing large amounts of high-dimensional data, which typically needs to be stored, read into memory, and moved from memory to compute units. In principle, GPU-native rendering avoids this movement of large image tensors from memory to compute cores, allowing for better GPU utilization, and training speed-ups. Rendering on-demand also allows to control perceptual appearance through the application of visual variations on-the-fly, allowing for higher diversity in training. To this end, we introduce stable-envs, an open-source library for fast, GPU-native rendering allowing to apply controllable variations on multiple simulation environments. By rendering image observations directly on GPU, stable-envs allows end-to-end training speed-ups of up to 1.33x, with no loss of performance. Besides faster training runs, stable-envs also enables reproducible, large-scale evaluations through batched physics w
    
[^5]: 注意差距：一个用于文档维护的已部署多智能体系统

    Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance

    [https://arxiv.org/abs/2611.23142](https://arxiv.org/abs/2611.23142)

    该论文提出并实际部署了一个以"TriAgent"分诊智能体为核心的多智能体系统，通过持续挖掘客服工单和代码-文档不一致等显性与隐性信号，自动检测并修复客户文档与技术实现之间的漂移。

    

    “AI革命”使科技行业的各团队能够以前所未有的速度交付软件功能。然而，当代码以比以往更快的速度演进时，面向客户的文档却常常难以跟上步伐。随着用户越来越依赖以文档作为主要交互接口的基于大语言模型（LLM）的智能体，这种文档差距的影响日益加剧：过时的信息会被传播并被当作权威信息对待。用于识别文档滞后于代码的信号其实早已大量存在，既有显性信号（例如，已解决的客服工单能够揭示文档缺失之处），也有隐性信号（例如，可通过自动化循环检测出的代码与文档之间的不一致）。我们部署了一个多智能体系统，其核心是我们称之为"TriAgent"的分诊智能体，该系统持续提取这些信号并修复文档漂移。在43天的影子模式运行中，我们的系统对499个工单进行了分诊，并发现了92个文档差距。

    arXiv:2611.23142v1 Announce Type: cross  Abstract: The "AI revolution" has enabled teams across the technology industry to deliver software features at unprecedented velocity. While code evolves faster than ever, customer-facing documentation often struggles to stay up to speed. As users increasingly rely on LLM-based agents that consume documentation as their primary interface, the impact of this documentation gap worsens: stale information propagates and is treated as authoritative. Signals for identifying where documentation has fallen behind already exist in abundance, both explicit (e.g., resolved support tickets revealing what documentation lacks) and implicit (e.g., code-documentation discrepancies detectable from automated loops). We deploy a multi-agent system, built around a triage agent we call "TriAgent," that continuously extracts these signals and repairs documentation drift. Over 43 days of shadow-mode operation, our system triaged 499 tickets and surfaced 92 gaps in the
    
[^6]: BugGen 2.0：重新思考大语言模型在可扩展RTL缺陷合成中的角色

    BugGen 2.0: Rethinking the Roles of LLMs for Scalable RTL Bug Synthesis

    [https://arxiv.org/abs/2611.23091](https://arxiv.org/abs/2611.23091)

    BugGen 2.0重新划分了LLM与确定性算法的分工，由算法负责RTL缺陷的结构分析、目标选择与验证，LLM专注于局部语义改写，从而实现高效可靠的大规模RTL缺陷数据集自动生成。

    

    AI辅助的RTL（寄存器传输级）定位与调试需要具有已知缺陷位置和可观测故障的真实缺陷数据集，而测试平台评级则使用注入的缺陷来衡量检测能力。大规模创建此类数据集仍然是EDA领域的一项重大挑战。BugGen 1.0证明了大语言模型（LLM）能够生成具有上下文感知能力的RTL变异，但它同时依赖模型来选择变异位置和生成编辑内容，这增加了推理成本，并可能导致无效的选择。BugGen 2.0改变了这种分工方式：确定性算法负责结构分析、目标选择和验证，而LLM则专注于局部语义改写。在四个OpenTitan IP模块和FabScalar上结合三种LLM的实验中，BugGen 2.0达到了所有请求的配额，生成了可编译且导致至少两次测试失败的缺陷场景，并且16种配置的变异类型中有14-16种得到体现。（原文摘要在此处截断）

    arXiv:2611.23091v1 Announce Type: cross  Abstract: AI-assisted RTL localization and debugging need ground-truth bug datasets with known locations and observable failures, while testbench grading uses injected bugs to measure detection capability. Creating such datasets at scale remains a major EDA challenge. BugGen 1.0 shows that Large Language Models (LLMs) can generate context-aware RTL mutations. Still, it relies on the model for both choosing where to mutate and generating the edit, which increases inference costs and can lead to invalid choices. BugGen 2.0 changes this division of labor. Deterministic algorithms handle structural analysis, target selection, and validation, while the LLM focuses on localized semantic rewriting. Across four OpenTitan IP blocks and FabScalar with three LLMs, BugGen 2.0 reaches every requested quota of compilable bug scenarios that cause at least two test failures, with 14-16 of 16 configured mutation types represented. In completed same-model compari
    
[^7]: RagHarnessDebugger：契约推理修复审计中基于Jev引导的智能体RAG框架

    RagHarnessDebugger: Jev-Guided Agentic RAG in Contract Inference Repair Auditing

    [https://arxiv.org/abs/2611.22991](https://arxiv.org/abs/2611.22991)

    提出了RagHarnessDebugger框架，通过Jev引导的测试工具记录并审计智能体RAG在契约推理修复过程中的全部决策，从而暴露出被准确率等聚合指标所掩盖的失败模式。

    

    使用检索增强生成（RAG）的智能体系统必须通过收集更多证据、测试替代策略，并决定是否、何时以及需要何种修复来自行修正其答案。然而，当准确率等指标可能具有误导性时，我们如何知道系统的修复是否真正有效？我们提出了RagHarnessDebugger，这是一个在Jev引导的测试框架中记录契约推理的每一个修复决策，并对这些记录进行审计的框架，从而能够对RAG修复过程中采取的所有决策进行完整调试。RagHarnessDebugger检查每条轨迹是否完整、每个验证是否适用于其所统计的候选对象，以及接受规则能够覆盖哪些答案。我们在ContractNLI保密协议任务上使用两个阅读器模型以及一组独立合同构成的复现集对我们的方法进行了测试。我们的调试器审计层暴露了隐藏在聚合指标之下的失败模式，以及策略接受成本方面的…

    arXiv:2611.22991v1 Announce Type: new  Abstract: Agentic systems using retrieval-augmented generation (RAG) must repair their own answers by gathering more evidence, testing alternative policies and deciding if, when, and what repairs are needed. Yet how do we know if a system's repairs are working when metrics like accuracy can be misleading? We present RagHarnessDebugger, a framework that records every repair decision in a Jev-guided harness for contract inference and audits the record to facilitate complete debugging of all decisions taken during a RAG repair journey. RagHarnessDebugger checks that each trace is intact, that each validation applies to the candidate it is counted for, and which answers the acceptance rule can reach. We tested our approach on ContractNLI non-disclosure agreements with two reader models and a replication set of separate contracts. Our debugger's audit layer exposed failure patterns that were hidden under aggregate metrics, policies acceptance cost of r
    
[^8]: 先思考再编码：一个测试大语言模型软件工程审慎思考能力的基准

    Think Before You Code: A Benchmark for Testing LLMs on Software Engineering Deliberation

    [https://arxiv.org/abs/2611.22905](https://arxiv.org/abs/2611.22905)

    该研究提出了TBYC基准框架，通过将GitHub议题讨论转化为结构化的工程思考产物并与模型生成的“思考”进行对比评估，来衡量代码大语言模型能否像人类工程师一样审慎地思考软件工程问题。

    

    Think Before You Code（TBYC）是一个基准测试流水线，用于研究以代码为中心的语言模型能否以类似人类工程审慎思考的方式对软件问题进行推理。该基准将GitHub议题讨论整理为结构化记录，提取人类讨论中的产物，如问题、解决方案、约束条件、理由和权衡，然后利用检索到的代码上下文和少样本示例，提示模型生成针对议题的“思考”。它将模型响应转换回相同的产物空间，并使用互补的指标（包括产物类型检测、元数据匹配、标签重叠度和语义摘要相似度）将其与人工标注进行对比评估。除了评分之外，该研究还包括可视化、排行榜和基线工具，使得能够跨仓库和议题比较不同的模型、提示策略和提取方法。Think Before You Code 是一个端到端的评估框架。

    arXiv:2611.22905v1 Announce Type: new  Abstract: Think Before You Code (TBYC) is a benchmark pipeline for studying whether code-focused language models can reason about software issues in ways that resemble human engineering deliberation. The benchmark curates GitHub issue discussions into structured records, extracts human discussion artifacts such as problems, solutions, constraints, rationales, and trade-offs, and then prompts models to generate issue-level "thoughts" using retrieved code context and few-shot examples. It converts model responses back into the same artifact space and evaluates them against human annotations using complementary metrics for artifact type detection, metadata matching, tag overlap, and semantic summary similarity. Beyond scoring, this study includes visualization, leaderboard, and baseline tools that make it possible to compare models, prompting strategies, and extraction methods across repositories and issues. Think Before You Code is an end-to-end eva
    
[^9]: 停止并非极限：理解并激发智能体在长程漏洞利用中的能力

    Stopping Is Not the Limit: Understanding and Eliciting Agent Capability in Long-Horizon Vulnerability Exploitation

    [https://arxiv.org/abs/2611.22888](https://arxiv.org/abs/2611.22888)

    该论文发现漏洞利用智能体在首次停止后继续执行仍能带来额外收益，并提出上下文压缩方法 SCoRe 以激发智能体在长程漏洞利用任务中的潜在能力。

    

    漏洞利用测试的是语言模型智能体能否基于已知软件缺陷开发出可用的攻击利用代码。顺序式测试时扩展可以提升漏洞利用性能，但更大的预算并不保证持续的探索：智能体常常在预算尚有剩余时就停止。然而，当执行在停止之后继续进行时会发生什么，这一问题仍未得到充分研究。我们在 ExploitBench 上使用 DeepSeek-V4-Flash-0731 和 MiMo-V2.6-Pro 对这一空白展开了研究。我们发现，在首次停止响应之后仍可获得额外的经验证的收益，但收益随执行时间的延长而递减。我们识别出与低效续跑相关的两种模式：同一上下文中停止响应比例的不断增加，以及更早停止与摘要中保留的关于未实现能力的不可能性判断相关。为检验上下文管理能否激发更多能力，我们提出了 SCoRe，它在停止爆发点压缩上下文并标记这些 ju…（原文摘要在此处截断）

    arXiv:2611.22888v1 Announce Type: cross  Abstract: Vulnerability exploitation tests whether language model agents can develop working exploits from known software flaws. Sequential test-time scaling can improve exploitation performance, but larger budgets do not ensure sustained exploration: agents often stop with budget remaining. Yet what happens when execution continues beyond stopping remains underexplored. We investigate this gap on ExploitBench using DeepSeek-V4-Flash-0731 and MiMo-V2.6-Pro. We find additional verified gains after the first stop response, but gains diminish over longer execution. We identify two patterns associated with less productive continuation: an increasing proportion of stop responses within the same context and earlier stopping associated with summaries retaining impossibility judgments about unachieved capabilities. To test whether context management can elicit further capability, we propose SCoRe, which compacts context at stop bursts and marks these ju
    
[^10]: 当通过检查并不能确立任务完成时：编程智能体中的证据解释

    When Passing Checks Do Not Establish Completion: Evidence Interpretation in Coding Agents

    [https://arxiv.org/abs/2611.22835](https://arxiv.org/abs/2611.22835)

    该研究通过因子实验与118次修复事件分析证明：编程智能体中检查通过并不等同于任务完成，正确报告的执行事实未必能导出所请求的有限结论，部分回复还存在误报事实或超出授权范围的问题。

    

    编程智能体中的验证程序必须保留执行事实、解释其范围并履行检查义务。我们通过合格的仓库检查、受控的证据消费者和可执行的工作流来研究这些职责。一项前瞻性的十三任务因子研究发现，正确报告的事实未必能得出所请求的有限结论。相反，在一项留出的回执供应研究中，所有122个交付的标签都是正确的，但有十六个回复误报了事实，另有十一个回复超出了授权范围。生产修复实验提供了成功的对照组以及有限支持的保留失败案例。在31个任务上的118次修复事件中，源码干预将启动可达性与修复区分开来，并将见证生成与所声明的操作区分开来。生命周期和资源控制界定了通过验证所能确立的范围。生成说明的比较显示出局部修复差异，但可能陈述错误的解释。

    arXiv:2611.22835v1 Announce Type: new  Abstract: Verification procedures in coding agents must preserve execution facts, interpret their scope and discharge checking obligations. We study these responsibilities through qualified repository checks, controlled evidence consumers and executable workflows. A prospective thirteen-task factorial study finds that correct reported facts need not yield the requested finite verdict. Conversely, all 122 delivered labels in a held-out receipt-supply study are correct, while sixteen responses misreport facts and eleven others exceed authorized scope. Production repair supplies successful controls and finite-supported preservation failures. Across 118 repair episodes on 31 tasks, source interventions separate startup reachability from repair and witness production from the claimed operation. Lifetime and resource controls delimit what passing validation establishes. Generated-note comparisons yield local repair differences but can state false explan
    
[^11]: Isabelle/HOL 的 Python 代码生成

    Python Code Generation for Isabelle/HOL

    [https://arxiv.org/abs/2611.22613](https://arxiv.org/abs/2611.22613)

    本文为 Isabelle 代码生成器开发了 Python 后端，使经过形式化验证的算法可以直接在 Python 生态中使用，或作为参考实现对现有 Python 库进行差分测试。

    

    Isabelle 定理证明器为数学形式化和程序推理提供了环境。为了使经过验证的实现具有实际可用性，其代码生成器将逻辑中可执行片段表达的函数翻译成常规编程语言。其目标语言传统上是函数式语言（Standard ML、OCaml、Haskell 和 Scala），最近新增了 Go。而 Python 作为在科学计算、数据科学和教育领域日益占据主导地位的语言，却一直未被支持。在本工作中，我们通过为 Isabelle 的代码生成器开发一个 Python 后端来填补这一空白。该生成器使经过验证的算法能够在 Python 应用中使用，或作为经过验证的参考实现，用于对现有 Python 库进行差分测试，而无需离开 Python 生态系统。我们通过覆盖 HOL 库大部分内容的回归测试套件进行测试，对该后端进行了验证。（摘要在此处截断）

    arXiv:2611.22613v1 Announce Type: cross  Abstract: The Isabelle theorem prover provides an environment for formalising mathematics and reasoning about programs. To make verified implementations practically usable, its code generator translates functions expressed in the executable fragment of its logic into conventional programming languages. Its targets have traditionally been functional languages (Standard ML, OCaml, Haskell, and Scala), recently joined by Go. Python, which is becoming the dominant language of scientific computing, data science, and education, has remained out of reach. In this work, we close this gap with a Python backend for Isabelle's code generator. This generator enables verified algorithms to be used in Python applications, or to serve as verified references against which established Python libraries can be differentially tested, without leaving the Python ecosystem. We validate the backend on regression suites covering large parts of the HOL library and by tes
    
[^12]: ReBoT-MiCS：面向嵌入式混合关键性系统时序行为增强的可靠性感知约束时间

    ReBoT-MiCS: Reliability-Aware Bounding Time for Enhancing Timing Behavior of Embedded Mixed-Criticality Systems

    [https://arxiv.org/abs/2611.22588](https://arxiv.org/abs/2611.22588)

    该论文提出ReBoT-MiCS方法，通过可靠性感知的低关键性WCET约束时间确定机制，在联合权衡QoS、模式切换与可靠性约束的前提下，增强嵌入式混合关键性系统的时序行为。

    

    arXiv:2611.22588v1 公告类型：cross 摘要：在混合关键性系统中，为同一任务考虑多个最坏情况执行时间（WCET），对应于不同的系统运行模式，是MC系统设计中的一个根本性挑战。为较低关键性系统运行模式确定合适的WCET本质上是困难的。过于紧凑的低WCET值可能触发频繁的模式切换，通过丢弃或降级低关键性任务并削减其功能来降低服务质量（QoS）；然而，更保守的取值则会降低资源效率。此外，低WCET的选择会影响所采用的瞬态故障模型下的暴露假设，而该模型被用于可靠性感知的调度分析。尽管已有一些研究考虑了低WCET的确定，以便在处理器利用率、模式切换概率和QoS之间进行合理权衡，但可靠性约束通常通过故障到（后文截断）……

    arXiv:2611.22588v1 Announce Type: cross  Abstract: In Mixed-Criticality (MC) systems, considering multiple Worst-Case Execution Times (WCETs) for the same task, corresponding to different system operation modes, presents a fundamental challenge in MC system design. Identifying suitable WCETs for lower (LO) criticality system operational modes is inherently difficult. Tight low WCET values may trigger frequent mode switches that degrade Quality-of-Service (QoS) by dropping/degrading the lower-criticality tasks and reducing their functionality. However, more conservative values reduce resource efficiency. Moreover, low WCET selection affects exposure assumptions under the adopted transient-fault model, which is used in reliability-aware scheduling analysis. Although some studies have considered low WCET determination to make a reasonable trade-off between processor utilization, mode switching probability and QoS, reliability constraints are typically addressed separately through fault-to
    
[^13]: 使用设计结构矩阵协调多学科设计分析与优化代码的集成

    Coordinating integration of multidisciplinary design analysis and optimisation codes with design structure matrices

    [https://arxiv.org/abs/2611.22522](https://arxiv.org/abs/2611.22522)

    本文将核聚变反应堆的集成设计问题纳入多学科设计分析与优化（MDAO）框架，引入设计结构矩阵（DSM）来可视化和协调集成分析与优化代码的耦合架构，从而提升计算效率。

    

    用于设计核聚变反应堆的集成分析与优化代码需要变得日益大规模、多学科且复杂，以便在设计流程中更早、更彻底地对设计需求进行计算验证。尽管集成模型的耦合方式对计算效率有重大影响，但耦合结构通常隐藏在设计框架的实现之中。由于这一问题并非聚变领域所独有，我们将集成设计问题构建为更广泛的多学科设计分析与优化（MDAO）领域的应用。MDAO领域最初是为航空航天等复杂系统设计问题而开发的，该领域使用（扩展的）设计结构矩阵（DSM）来展示集成分析与优化代码的架构。本工作的目的是介绍这些DSM方法，并演示它们如何（应用于聚变设计问题）。

    arXiv:2611.22522v1 Announce Type: cross  Abstract: The integrated analysis and optimisation codes used to design nuclear fusion reactors need to become increasingly large scale, multidisciplinary and complex to computationally verify design requirements more thoroughly and earlier in the design process. Though the manner in which the integrated models are coupled can have significant implications for computational efficiency, the structure of couplings typically remains hidden in the implementation of a design framework. As this problem is not unique to fusion, we frame integrated design problems as applications of the broader field of Multidisciplinary Design Analysis and Optimisation (MDAO). Originally developed for complex system design problems in e.g. aerospace, the MDAO field uses (extended) Design Structure Matrices (DSMs) to display the architecture of integrated analysis and optimisation codes. The aim of this work is to introduce these DSM methods and demonstrate how they can
    
[^14]: GitHub Actions 工作流中安全修复回移植的缺失

    Missing Security-Fix Backporting for GitHub Actions Workflows

    [https://arxiv.org/abs/2611.22519](https://arxiv.org/abs/2611.22519)

    本文首次大规模实证研究了 GitHub Actions 工作流安全修复在发布分支上回移植缺失的问题，发现近一半（224,392/474,490）已审计的发布分支仍存在已在默认分支修复的已知漏洞。

    

    GitHub 通过 GitHub Actions 原生支持工作流自动化，GitHub Actions 目前是该平台上占主导地位的 CI/CD 工具。然而，保持这些工作流的安全性仍然是一项持续的负担：维护者通常只在默认分支上修补漏洞，而下游用户持续用于构建的发布分支和维护分支上却依然存在相同缺陷。将已知的修复迁移到这些分叉分支上，即所谓的“回移植”，却很少被执行。本文通过首次大规模实证研究量化了这一差距。我们从 GitHub Actions 工作流历史的公开语料库中挖掘了 1,903,191 个修改工作流的提交，使用静态扫描器 zizmor 对每个提交的前后状态进行扫描，在 24,238 个项目中识别出 69,865 个安全修复，并审计每个发布分支以检查修复是否被回移植。这一差距普遍存在：在审计的 474,490 个发布分支中，有 224,392 个仍然存在漏洞，而刻意的回移植行为非常罕见。

    arXiv:2611.22519v1 Announce Type: new  Abstract: GitHub natively supports workflow automation through GitHub Actions, now the dominant CI/CD tool on the platform. Yet keeping these workflows secure remains an ongoing burden: maintainers often patch a vulnerability on the default branch but leave the same flaw on the release and maintenance branches that downstream consumers continue to build from. Carrying a known fix onto these divergent branches, called backporting, is rarely done, and this paper presents the first large-scale empirical study quantifying the resulting gap. From a public corpus of GitHub Actions workflow histories, we mine 1,903,191 workflow-modifying commits, scan each before and after with the static scanner zizmor to identify 69,865 security fixes across 24,238 projects, and audit every release branch to check whether each fix was carried over. The gap is pervasive: 224,392 of 474,490 audited release branches remain vulnerable, and deliberate backporting is rare an
    
[^15]: VoxelSBFL：面向视觉-语言-动作模型的体素级基于频谱的失败预测方法

    VoxelSBFL: Voxel-Level Spectrum-Based Failure Prediction for Vision-Language-Action Models

    [https://arxiv.org/abs/2611.22453](https://arxiv.org/abs/2611.22453)

    该论文提出VoxelSBFL，创新性地将软件工程中的频谱故障定位（SBFL）技术黑盒迁移到机器人测试领域，以工作空间体素取代程序元素作为分析单元，利用TCP轨迹的体素覆盖信息与执行结果构建可疑度地图，从而在无需模型内部信息的情况下实现体素级的失败预测。

    

    视觉-语言-动作模型集成了视觉感知、语言理解和动作生成能力来控制机器人。对这类系统进行测试会产生轨迹和任务结果，但现有方法对于如何利用这些记录的轨迹来评估后续执行或指导策略修复提供的指导十分有限。我们推测，对于给定的策略和任务，穿越某些工作空间区域与更高的失败可能性相关。为了验证这一猜想，我们提出了VoxelSBFL，这是基于频谱的故障定位技术的一种黑盒适配方案，其中工作空间体素取代程序元素作为分析单元。记录的工具中心点（TCP）轨迹提供二值化的体素覆盖信息，再结合执行结果，利用成熟的SBFL公式构建可疑度地图。这些地图能够识别与失败相关的区域，并为未见过的轨迹进行评分，而无需访问模型内部信息。

    arXiv:2611.22453v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models integrate visual perception, language understanding, and action generation to control robots. Testing these systems produces trajectories and task outcomes, but existing approaches provide limited guidance on using these recorded trajectories to assess subsequent executions or inform policy repair. We conjecture that, for a given policy and task, traversing certain workspace regions is associated with a higher likelihood of failure. To investigate this conjecture, we propose VoxelSBFL, a black-box adaptation of Spectrum-Based Fault Localization (SBFL) in which workspace voxels replace program elements as units of analysis. Recorded tool-center-point (TCP) trajectories provide binary voxel coverage, which is combined with execution outcomes to construct suspiciousness maps using established SBFL formulas. These maps identify failure-associated regions and score unseen trajectories without requiring in
    
[^16]: 智能合约模糊测试的自适应预算分配：为什么在线反馈无法改进漏洞分类

    Adaptive Budget Allocation for Smart-Contract Fuzzing: Why Online Feedback Fails to Improve Vulnerability Triage

    [https://arxiv.org/abs/2611.22430](https://arxiv.org/abs/2611.22430)

    该论文发现智能合约模糊测试中的在线反馈策略均无法超越均匀预算分配，并提出一种基于离线概率曲线预测首次报告概率、按每次执行预测增益分配预算的分配器来改进漏洞分类效果。

    

    智能合约模糊测试活动共享有限的预算，而首个正确类别漏洞报告的执行成本在各合约间差异很大。通过对143个带标注合约和350个注入漏洞合约进行轨迹重放，我们检验在线反馈是否能够改进漏洞分类。在每合约平均100次执行的预算下，均匀分配分别达到各自预知最优值的72%和70%。全部八种在线策略在总体分类表现上都不及均匀分配，在漏洞注入语料库上重复测试的七种策略同样如此。五种在线信号对即将出现的报告的排序AUC仅为0.32-0.51。合成预言机实验表明，足够精确的检测成本估计比可检测性排序能够弥补更多的性能差距。这启发了一种离线概率曲线分配器，它预测各预算水平下的首次报告概率，并按每次执行的预测增益来分配预算。在每语料库使用四个新鲜种子的冻结评估中，检测预算AUC得到提升。

    arXiv:2611.22430v1 Announce Type: new  Abstract: Smart-contract fuzzing campaigns share a finite budget and vary widely in the execution cost of a first correct-class vulnerability report. Using trace replay on 143 annotated and 350 bug-injected contracts, we test whether online feedback improves triage. At a mean budget of 100 executions per contract, uniform allocation reaches 72% and 70% of the respective clairvoyant optima. All eight online policies trail uniform on aggregate triage, as do the seven repeated on the bug-injected corpus. Five online signals yield ranking AUCs of 0.32-0.51 for imminent reports. Synthetic-oracle experiments show that sufficiently accurate detection-cost estimates recover more of the gap than detectability rankings. This motivates an offline probability-curve allocator that predicts first-report probabilities across budgets and allocates by predicted gain per execution. In a frozen evaluation on four fresh seeds per corpus, detection-budget AUC improves
    
[^17]: BehaVerify：一种面向行为树的形式化验证工具

    BehaVerify: A Formal Verification Tool for Behavior Trees

    [https://arxiv.org/abs/2611.22403](https://arxiv.org/abs/2611.22403)

    BehaVerify是一个面向行为树及神经符号行为树的形式化验证工具箱，通过带形式语法与语义的DSL、时序逻辑规范以及高效的nuXmv编码实现快速验证，并能生成可集成于PyTrees和ROS的Python/Haskell实现。

    

    本文提出了BehaVerify，一个面向行为树（BT）及神经符号行为树的验证工具箱。行为树是一种用于规划与排序行为的图形化语言，类似于扩展状态机，并且在安全关键领域的应用日益增多。BehaVerify使用户能够使用一种具有形式化语法和语义的领域特定语言（DSL）来描述行为树。该语言还支持线性时序逻辑（LTL）和计算树逻辑（CTL）规范。BehaVerify采用一种被称为“快进”的高效nuXmv编码方法，快速验证规范在行为树上是否成立。此外，BehaVerify还能生成行为树的Python和Haskell实现。通过Python，BehaVerify已与PyTrees和机器人操作系统（ROS）集成，并已部署到多个自主机器人系统中，包括用于验证运行时监控器。尽管先前的工作已经发展了底层理论，本文……（摘要在此处被截断）

    arXiv:2611.22403v1 Announce Type: new  Abstract: This paper presents BehaVerify, a verification toolbox for behavior trees (BTs) and Neuro-Symbolic BTs. BTs are a graphical language for planning and sequencing behaviors - similar to extended state machines - and are increasingly used in safety-critical domains. BehaVerify enables users to specify a BT using a Domain Specific Language (DSL) with formal syntax and semantics. This language further includes Linear Temporal Logic and Computation Tree Logic specifications. BehaVerify uses an efficient encoding into nuXmv, called fastforwarding, to quickly verify if a specification holds over the BT. Additionally, BehaVerify can generate implementations of the BT in Python and Haskell. Through Python, BehaVerify has been integrated with PyTrees and Robot Operating System (ROS) and has been deployed to several autonomous robotic systems, including for verifying runtime monitors. While prior works have developed the underlying theory, this manu
    
[^18]: RyBench：基于发布/订阅网络的运行时验证基准测试框架

    RyBench: Runtime Verification Benchmarking Framework over Publish/Subscribe Networks

    [https://arxiv.org/abs/2611.22401](https://arxiv.org/abs/2611.22401)

    本文提出了RyBench——一个基于发布/订阅的基准测试框架，它作为合成事件发布者替代被监控系统，并通过将事件生成、序列化和传输解耦为可灵活组合的独立组件，为运行时验证工具的吞吐量与判决延迟评估提供了统一的标准化测试平台。

    

    arXiv:2611.22401v1 公告类型：交叉 摘要：以分布式、基于节点的形式构建的自治系统通过发布/订阅中间件进行通信。运行时验证（RV）工具用于依据形式化规范对这些系统进行监控，并通过发布/订阅通道接收被监控的事件。除了正确性之外，RV工具的实用价值还取决于其监控吞吐量能力和判决延迟，因此需要一个系统化的基准测试设置来评估这些工具的运行效率。本文提出了RyBench，一个基于发布/订阅的基准测试框架，它充当合成事件发布者来替代被监控的系统，从而提供统一的基准测试环境。RyBench将基准测试生成、序列化和传输分离为独立的组件，使不同的事件生成策略、数据格式和传输层能够灵活组合。我们还使用Open Simulatio（原文在此处截断）

    arXiv:2611.22401v1 Announce Type: cross  Abstract: Autonomous systems built as distributed, node-based systems communicate via publish/subscribe middleware. Runtime Verification (RV) tools are used to monitor those systems against formal specifications, receiving monitored events through the pub/sub channel. An RV tool's practical value depends on its monitoring throughput capacity and verdict latency, besides its correctness, so a systematic benchmarking setup is needed to evaluate how efficiently these tools run. This paper presents RyBench, a pub/sub-based benchmarking framework that acts as a synthetic event publisher in place of a monitored system, providing a common benchmarking setup. RyBench separates benchmark generation, serialization, and transport into distinct components, enabling the combination of different event-generation strategies, data formats, and transport layers. We also evaluate the framework's throughput capacity across these configurations using Open Simulatio
    
[^19]: ROSACE飞行控制器容错扩展的形式化分析

    Formal Analysis of a Fault-Tolerant Extension of the ROSACE Flight Controller

    [https://arxiv.org/abs/2611.22394](https://arxiv.org/abs/2611.22394)

    本文利用COMPASTA工具为ROSACE飞行控制器基准扩展了基于三模冗余的FDIR容错组件，并通过形式化建模与故障注入对其运行时故障抵御能力进行分析与验证。

    

    自主航空电子控制器即使在运行时出现故障的情况下，也必须满足严格的实时功能与性能要求。在本文中，我们分析了ROSACE——一个用于飞行控制系统的开源控制工程基准测试，重点关注其对传感器噪声和传感器故障等运行时故障的抵御能力。我们通过一个基于三模冗余的FDIR（故障检测、识别与恢复）组件对该基准测试进行了扩展。我们的方法建立在COMPASTA之上，这是一个基于欧洲航天局（ESA）TASTE工具开发的模型驱动设计工具。COMPASTA支持对系统设计进行故障建模与故障注入。此外，它实现了基于仿真的验证与确认工作流程，能够依据用户自定义的覆盖标准，自动生成并执行一组测试用例。利用COMPASTA，我们构建了一组形式化模型，包括带故障的模型和带FDIR的模型，并对它们进行形式化分析。

    arXiv:2611.22394v1 Announce Type: new  Abstract: Autonomous avionics controllers must satisfy stringent real-time functional and performance require- ments even in the presence of run-time faults. In this paper, we analyze ROSACE, an open-source control engineering benchmark for flight control systems, focusing on its run-time resilience against faults such as sensor noise and sensor failures. We extend the benchmark with an FDIR (Fault Detection, Identification and Recovery) component based on Triple Modular Redundancy.   Our approach builds upon COMPASTA, a tool for Model-Based Design developed on top of the TASTE tool by ESA. COMPASTA enables the modeling and injection of faults into a system's design. Moreover, it implements a simulation-based verification and validation workflow, automating the generation and execution of a set of test cases, based on a user-defined notion of coverage. Using COMPASTA, we construct a set of formal models, with faults and with FDIR, and we formally 
    
[^20]: 智能体轨迹作为软件质量保证的证据

    Agent Trajectories as Evidence for Software Quality Assurance

    [https://arxiv.org/abs/2611.22323](https://arxiv.org/abs/2611.22323)

    该论文提出将智能体轨迹与规格说明及AI生成产物结合分析，作为针对性软件质量保证的证据，以优先审查智能体影响重大的决策、暴露隐式假设并持续改进开发实践。

    

    自主软件智能体在极少人工监督的情况下，越来越多地承担软件的规划、实现与测试工作。它们的决策可能引入最终代码和通过的测试无法充分暴露的隐式假设。我们提出将可观察的智能体轨迹，连同软件与智能体规格说明以及AI生成的产物一同分析，作为针对性质量保证的证据。其目标是优先审查那些影响重大的决策，调查它们对软件问题可能产生的作用，并为改进规格说明和开发实践提供依据。工业界的经验激发了这一想法；现有的执行记录和初步分析方法使其具备可行性。然而，仅凭轨迹并不能证明决策是合理的、结果是可接受的，或纠正性修改是有效的。我们探讨这种证据如何支持评审、诊断和持续的质量改进，以及何时采用更简单的质量保证方法……

    arXiv:2611.22323v1 Announce Type: cross  Abstract: Autonomous software agents increasingly plan, implement, and test software with limited human oversight. Their decisions may introduce implicit assumptions that final code and passing tests do not adequately expose. We propose analyzing observable agent trajectories alongside software and agent specifications and AI generated artifacts as evidence for targeted quality assurance. The aim is to prioritize consequential decisions for scrutiny, investigate their possible contribution to software problems, and inform improvements to specifications and development practices. Industrial experience motivates the idea; available execution records and preliminary analysis methods make it plausible. However, trajectories alone do not establish that decisions are justified, outcomes acceptable, or corrective changes effective. We ask how this evidence can support review, diagnosis, and continuous quality improvement, and when simpler QA approaches
    
[^21]: 当世界成为测试套件：面向物理世界的智能体智能

    When the World Is the Test Suite: Agentic Intelligence for the Physical World

    [https://arxiv.org/abs/2611.22291](https://arxiv.org/abs/2611.22291)

    本文提出将物理世界视为程序的测试套件，系统考察了面向物理世界的智能体智能这一新兴领域，围绕智能体如何为物理世界编写程序、如何测试这些程序以及如何从测试结果中学习三个核心问题展开论述。

    

    为修复软件而构建的编程智能体，如今开始为机器人、实验室和工业控制器编写、测试和修复程序。与软件工程不同，这些代码并不附带现成的测试套件。程序是否有效，必须通过在真实机器人上运行、在物理模拟器中执行、或通过它所辅助训练的策略来验证，而这类验证证据获取成本高昂，且本身也可能是错误的。随着这些智能体承担越来越多的物理任务，程序的评判方式与其编写方式变得同等重要。本文通过将物理世界视为程序的测试套件，考察了这一新兴领域——面向物理世界的智能体智能。我们围绕三个问题组织讨论：智能体如何为物理世界编写程序、这些程序如何被测试、以及智能体如何从测试结果中学习。我们首先追溯了面向机器人的语言模型与软件工程智能体如何走向融合，并探讨哪些软件测试实践可以……

    arXiv:2611.22291v1 Announce Type: cross  Abstract: Coding agents built to repair software now write, test, and repair programs for robots, laboratories, and industrial controllers. Unlike in software engineering, no test suite comes with the code. Whether a program works must be established by running it on a robot, in a physics simulator, or through the policies it helps to train, and such evidence is costly to obtain and can itself be wrong. As these agents take on more physical tasks, how their programs are judged matters as much as how they are written. This paper examines this emerging area, agentic intelligence for the physical world, by treating the world as the program's test suite. We organize the discussion around three questions: how agents write programs for the physical world, how these programs are tested, and how agents learn from the results. We first trace how language models for robots and software engineering agents came together, ask which software testing practices
    
[^22]: OfficePrism：面向企业业务职能单域操作的智能体基准测试

    OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions

    [https://arxiv.org/abs/2611.22262](https://arxiv.org/abs/2611.22262)

    该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。

    

    宽泛的职场能力评分可能将单一业务职能内的胜任能力与跨职能、跨系统协调的额外成本混为一谈，导致失败难以定位。OfficePrism 通过 420 个涵盖七大业务职能的单域企业任务，在统一的、由基准定义的 L1–L3 任务结构层级下评估这一前提。每个任务都在一个全新的环境中运行，该环境由一个共享的合成组织与一个特定职能领域及类型化状态访问组合而成。我们的流水线利用有据可依的记录、业务规则和工具模式，联合构建中文工单、隐藏的可执行参考见证以及预期响应，随后进行自动过滤和人工筛选。同一任务集通过完整模式工具包和搜索-分发工具包两种方式进行评估。在 124 种模型-框架-工具包配置中，每个案例的自然语言与数据库联合通过率介于 22.6% 至 76.4% 之间。

    arXiv:2611.22262v1 Announce Type: cross  Abstract: Broad workplace scores can mix competence within one business function with the extra cost of coordinating across functions and systems, making failures hard to localize. OfficePrism evaluates this prerequisite through 420 single-domain enterprise tasks spanning seven business functions under shared, benchmark-defined L1--L3 task-structure levels. Each task runs in a fresh environment combining a shared synthetic organization with one function-specific domain and yped state access. Our pipeline uses grounded records, business rules, and tool schemas to jointly construct a Chinese-language ticket, a hidden executable reference witness, and an expected response, followed by automated filtering and human curation. The same task set is evaluated through full-schema and search-and-dispatch packages. Across 124 model--harness--package configurations, the per-case joint natural-language-and-database pass rate ranges from 22.6% to 76.4%, with 
    
[^23]: Readie：面向AI代理的低延迟代码沙箱

    Readie: Low-Latency Code Sandboxes for AI Agents

    [https://arxiv.org/abs/2611.22171](https://arxiv.org/abs/2611.22171)

    提出Readie系统，通过从历史执行中学习一组小型、快速初始化的启动快照集合，使其共同覆盖大部分AI代理生成的代码，从而解决现有代码沙箱启动缓慢的问题。

    

    从量化研究到个人助理，AI代理通过迭代式地生成并执行代码来完成任务。为了将代理框架与代码执行解耦，许多代理将执行工作卸载到称为代码沙箱的专用系统中。然而，现有的代码沙箱存在启动缓慢的问题。无论它们依赖自定义快照还是单一的大型镜像，根本原因都是相同的：很难构建一个紧凑且预热的快照，既能在多样化的代理生成代码之间共享，又能覆盖其所需的几乎所有工件（例如软件包、数据集）。在这项工作中，我们提出了用于低延迟代码沙箱的学习型启动快照。我们的洞察是：虽然单个快照很难同时做到紧凑并覆盖大多数工作负载，但一组快照可以同时实现这两点。每个快照都很小、初始化迅速，并覆盖一部分代理生成的代码；这些快照组合在一起则能覆盖其中的大部分。我们从历史执行数据中学习这组快照……

    arXiv:2611.22171v1 Announce Type: new  Abstract: From quantitative research to personal assistance, AI agents act by iteratively generating and executing code. To decouple the agent harness from code execution, many agents offload execution to dedicated systems called code sandboxes. However, existing code sandboxes suffer from slow startup. Whether they rely on custom snapshots or a single fat image, the root cause is the same: it is hard to build a compact, pre-warmed snapshot that can be shared across diverse agent-generated code while covering nearly all the artifacts (e.g., packages, datasets) it requires.   In this work, we propose learned startup snapshots for low-latency code sandboxes. Our insight is that while it is hard for a single snapshot to be both compact and cover most workloads, a set of snapshots can achieve both. Each snapshot is small, initializes quickly, and covers a fraction of agent-generated code; together, the set covers most of it. We learn this set from a l
    
[^24]: 专家构建任务上的强化学习教会了 Nemotron 3.5 Super 什么

    What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super

    [https://arxiv.org/abs/2611.22161](https://arxiv.org/abs/2611.22161)

    通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。

    

    我们对 NVIDIA 的 Nemotron 3.5 Super（总参数量 120B，激活参数 12B）进行强化学习（RL）后训练，使用五个专家构建的任务数据集，每次训练一个：真实世界软件工程、复杂约束、长程工具使用、专业图表理解和企业文档推理。每次运行都在 Tinker 上使用 CISPO 训练一个全新的 rank-32 LoRA 适配器，不设监督学习阶段，并在相同的任务、测试框架和设置下将每个适配器与基础模型进行比较。在十二个主要基准测试上（其中八个由其他团队构建，四个由 Surge AI 构建），提升幅度从 IFBench 上的 +4.5 分到 GDP_pdf 上的 +13.8 分不等（后者从 10.4% 提升至 24.2%）；SWE-bench Pro V2 从 86.9% 提升至 92.1%，Terminal-Bench 2.1 从 63.9% 提升至 70.3%，MultiChallenge 从 58.3% 提升至 66.6%，AutomationBench 从 14.8% 提升至 27.5%，Chartography 从 39.4% 提升至 50.6%。文档适配器在没有 API 工具的情况下训练，也在 AutomationBench 上获得了 +8.0 分的提升。

    arXiv:2611.22161v1 Announce Type: cross  Abstract: We post-train NVIDIA's Nemotron 3.5 Super (120B total, 12B active parameters) with reinforcement learning (RL) on five expert-built task datasets, one at a time: real-world software engineering, complex constraints, long-horizon tool use, professional chart understanding and enterprise document reasoning. Each run trains a fresh rank-32 LoRA adapter with CISPO on Tinker, with no supervised stage, and each adapter is compared with the base model on the same tasks, harness and settings. On twelve headline benchmarks, eight built by other groups and four by Surge AI, the gains range from +4.5 points on IFBench to +13.8 on GDP_pdf, which rises from 10.4% to 24.2%; SWE-bench Pro V2 rises from 86.9% to 92.1%, Terminal-Bench 2.1 from 63.9% to 70.3%, MultiChallenge from 58.3% to 66.6%, AutomationBench from 14.8% to 27.5% and Chartography from 39.4% to 50.6%. The document adapter, trained without API tools, also gains +8.0 points on AutomationB
    
[^25]: 运营一个智能体驱动的数据集工厂：发现、构建与公开发布

    Operating an Agent-Driven Dataset Factory: Discovery, Construction, and Public Release

    [https://arxiv.org/abs/2611.21957](https://arxiv.org/abs/2611.21957)

    该论文部署了一个由语言模型智能体驱动、确定性软件支撑的自动化“数据集工厂”，在81.53小时内以约2.74美元的单次发布成本持续产出14个预测数据集，并通过审查暴露了构建规则与数据总体声明不一致的治理问题。

    

    将公开数据转化为预测数据集，需要把数据源发现、任务定义、构建和分发等环节连接起来。我们部署了 Dataset Factory（数据集工厂），在常设发布政策下持续执行这些步骤。有界的语言模型任务负责解读数据源、编写构建代码并审查任务含义；确定性软件则负责管理状态、工作进程、重试和发布。在81.53小时内，该工厂发布了14个数据集，估计的经常性运营支出约为每次发布2.74美元。该成本核算包含了失败的尝试、推理、工作进程和托管费用。独立的智能体在全部12个被评估的发布版本上完成了预测实验。尽管如此，数据源审查仍然发现了一个任务，其构建过滤器与其总体声明相矛盾。原始审查者接受了后续记录中被排除的条目，却没有将其与已发布的选择规则进行核对。这一经验记录了一次

    arXiv:2611.21957v1 Announce Type: new  Abstract: Turning public data into prediction datasets requires connecting source discovery, task definition, construction, and distribution. We deployed Dataset Factory to perform these steps continuously under a standing release policy. Bounded language-model tasks interpret sources, write construction code, and review task meaning; deterministic software manages state, workers, retries, and publication. Over 81.53 hours, the factory published 14 datasets at an estimated recurring operating expenditure of about \$2.74 per publication. The accounting includes failed attempts, inference, workers, and hosting. Separate agents completed prediction experiments on all twelve evaluated releases. Source review nevertheless identified a task whose construction filters contradicted its population declaration. The original reviewer accepted later-record exclusions without reconciling them with the published selection rule. This experience documents an oper
    
[^26]: EmbedGym：让编码代理无需物理硬件即可开发和调试嵌入式固件

    EmbedGym: Letting Coding Agents Develop and Debug Embedded Firmware Without Physical Hardware

    [https://arxiv.org/abs/2611.21845](https://arxiv.org/abs/2611.21845)

    EmbedGym是一个交互式虚拟硬件环境，通过指令级MCU仿真器结合41个外设模型、模拟电路仿真器及虚拟WiFi/BLE端点，让编码代理无需物理硬件即可开发和调试嵌入式固件。

    

    现代编码代理在通用软件基准测试中表现出色，因为它们可以在闭环中运行代码和单元测试。然而，嵌入式固件开发打破了这一循环：一个能够编译的程序在真实硬件上仍可能因引脚极性、总线协议、时序或网络问题而失败，而这些问题对于处于文本终端的代理来说是无法观察到的。现有的嵌入式编码代理要么跳过硬件执行，要么使用具有数秒延迟的仅支持GPIO的模拟器，要么在静态环境中测试物理板卡。在本文中，我们提出了EmbedGym，这是一个交互式虚拟硬件环境，使编码代理能够在没有物理硬件的情况下开发和调试嵌入式固件。EmbedGym使用指令级微控制器（MCU）仿真器来执行与烧录到物理板上相同的二进制文件，并将其连接到41个外设模型、一个模拟电路仿真器以及虚拟WiFi/低功耗蓝牙（BLE）端点。

    arXiv:2611.21845v1 Announce Type: cross  Abstract: Modern coding agents perform well on general software benchmarks because they can run code and unit tests in a closed loop. Embedded firmware development, however, breaks this loop: a program that compiles can still fail on real hardware due to pin polarity, bus protocol, timing, or networking bugs that an agent at a text terminal cannot observe. Existing embedded coding agents either skip hardware execution, drive GPIO-only simulators with multi-second delays, or test on physical boards in a static environment. In this paper, we present EmbedGym, an interactive virtual hardware environment that lets coding agents develop and debug embedded firmware without physical hardware. EmbedGym uses instruction-level microcontroller unit (MCU) emulators to execute the same binary that would be flashed onto a physical board, connects them to 41 peripheral models, an analog circuit simulator, and virtual WiFi/Bluetooth Low Energy (BLE) endpoints, 
    
[^27]: 编程智能体能否编写和转译单元测试

    Can Coding Agents Write and Transpile Unit Tests

    [https://arxiv.org/abs/2611.21797](https://arxiv.org/abs/2611.21797)

    本研究对三个编程智能体进行了基准测试，发现其编写的单元测试在覆盖率上远超人类测试套件，但测试质量随项目规模增大而急剧下降，且跨语言测试翻译仍不一致且不可靠。

    

    基于大语言模型（LLM）构建的编程助手已成为现代软件开发的重要组成部分，人们越来越信任它们端到端地执行工作流程和改造现代化代码。这种信任很大程度上建立在测试通过的基础上，而这些测试往往是由编程智能体自己编写的。在这项工作中，我们检验了这些智能体编写可靠测试以及在编程语言之间转换测试的能力。我们对三个流行的编程智能体进行了基准测试，每个都使用原生前沿模型和高推理强度进行评估。通过广泛的指标验证，我们发现编程智能体在传统覆盖率指标上远超人类编写的测试套件。然而，这一优势并不均衡：智能体的测试质量随项目规模增大而急剧下降，且其优势很大程度上源于编写了更多的测试数量。我们还表明，智能体测试翻译虽然在总体上看起来表现强劲，但仍然不一致且不可靠。

    arXiv:2611.21797v1 Announce Type: cross  Abstract: Coding assistants built on Large Language Models (LLMs) have become a staple of modern software development, increasingly trusted to execute workflows and modernize code end-to-end. Much of this trust rests on passing tests, which are often written by the coding agents themselves. In this work, we examine the ability of these agents to both write reliable tests and translate them between programming languages. We benchmark three popular coding agents, each evaluated with a native frontier model and high reasoning effort. Verifying across a broad suite of metrics, we find that coding agents far exceed human written suites on traditional measures of coverage. This advantage, however, is uneven: agent test quality degrades sharply with project size, and much of the advantage lies in a greater number of written tests. We also show that agentic test translation---while seemingly strong in aggregate---remains inconsistent and unreliable.
    
[^28]: 重新审视智能体代码的存续：测量方法的选择如何决定结论

    Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict

    [https://arxiv.org/abs/2611.21784](https://arxiv.org/abs/2611.21784)

    本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。

    

    arXiv:2611.21784v1 公告类型：交叉

    arXiv:2611.21784v1 Announce Type: cross  Abstract: Studies of how long code from AI coding agents survives after it is merged reach conflicting verdicts: agent code is reported to be retained longer than human code, about as long, or modified sooner. Each study makes defensible measurement choices, but how much these choices decide the verdict is unknown. We revisit the survival of agent-attributed code in our earlier study: 854,139 lines that 2,802 merged pull requests (PRs) added to 215 repositories of the AIDev dataset. We vary nine measurement choices in five research questions and build a specification curve of 88 survival analyses, tested under three variance estimators, several observation windows, and permuted labels. For code from human-supervised agents, mostly GitHub Copilot, the choices change the size of the effect but not its direction. For code from the fully-autonomous agent Devin, they decide the verdict. Across repositories, 15 of 22 specifications report it modified 
    
[^29]: 汽车数字孪生建模语言综述

    A Survey of Modeling Languages for Automotive Digital Twins

    [https://arxiv.org/abs/2611.21636](https://arxiv.org/abs/2611.21636)

    本文通过对德国汽车行业专家的问卷调查，识别了汽车数字孪生的期望用途与能力、汽车开发各阶段相关的建模技术以及非正式模型的作用，弥合了学术文献与工业实践之间的差距。

    

    汽车行业对数字孪生及合适建模技术的需求正在迅速增长。然而，汽车领域对数字孪生尚缺乏统一的理解，也没有建立起用于创建汽车数字孪生的建模技术。近期关于数字孪生的研究主要集中于对汽车领域或一般性数字孪生文献的分析，因而忽视了汽车从业者的行业视角。为了弥合科学文献与工业视角之间的差距，我们在德国汽车行业的专家中开展了一项问卷调查，以识别：（i）数字孪生所期望的用途与能力，（ii）在汽车开发各阶段中与数字孪生工程化和运行相关的建模技术，以及（iii）非正式模型在汽车开发过程中所扮演的角色。为此，我们联系了189位软件定义车辆（Software-Defined Vehicle）社区的成员……

    arXiv:2611.21636v1 Announce Type: new  Abstract: The demand for digital twins and suitable modeling techniques in the automotive industry is increasing rapidly. Yet, there is no common understanding of digital twins in automotive, nor are there modeling techniques established to create automotive digital twins. Recent studies on digital twins focus on the analysis of the literature on digital twins for automotive or in general and, thus, neglect the industrial perspective of automotive practitioners. To mitigate this gap between scientific literature and the industrial perspective, we conducted a questionnaire survey among experts in the German automotive industry to identify (i) the desired purposes for and capabilities of digital twins, (ii) the modeling techniques related to engineering and operating digital twins across the phases of automotive development, and (iii) the role informal models play during automotive development. To this end, we contacted 189 members of the Software-D
    
[^30]: 我们先修正了评分器，随后聚类校正推翻了我们在AI代码审查中植入缺陷检测的自家结果

    We Fixed the Scorer First, and Cluster Correction Killed Our Own Planted-Defect Result in AI Code Review

    [https://arxiv.org/abs/2611.21635](https://arxiv.org/abs/2611.21635)

    本文的核心贡献在于：修正评分器后，通过精确置换检验与Holm多重校正证明原先看似显著的AI代码审查植入缺陷检测结果并不成立，并指出提升检测统计功效需要增加测试案例数（12-25个）而非增加重复次数。

    

    仅凭AI代码审查管线能捕获多少问题来评判它，会掩盖另一个维度：它漏掉了什么。更强的筛选会“免费”提升精确率，而其代价永远不会体现在报告中。我们在相同的八个案例上运行四个实验分支、使用同一个评分器，探究植入缺陷检测率能否在不提高误报率的情况下上升。三次重复嵌套于八个案例之中，因此p值按每个对比分别报告且报告两次：在基于案例的精确置换检验（2^8种排列）下，原本在独立性假设下“可发表”的两组对比（Fisher p=0.017与0.019）分别升至0.0625和0.109，而经Holm校正后没有任何对比仍具显著性。这是“在本设计下无法观测到”，而非“没有效应”：符号检验需要在同一方向上出现六个不一致案例，其在原假设下的实际I类错误率为0.0078，而要达到80%统计功效需要单案例胜率接近0.80；弥补这一差距需要12至25个案例，而非更多重复次数。在同一轮运行中，第二次检查永远不会移除第一次检查所发现的问题。

    arXiv:2611.21635v1 Announce Type: new  Abstract: Judging an AI code-review pipeline only by how much it catches hides an axis: what it dropped. Stronger selection raises precision for free; the cost never reaches the report. We ran four arms on the same eight cases, one scorer, asking whether planted-defect detection can rise without raising the claim error rate. Three repetitions are nested in eight cases, so p is reported per contrast and twice: under an exact permutation test over cases (2^8 arrangements) the two contrasts publishable under independence (Fisher p=0.017, 0.019) move to 0.0625 and 0.109, and after Holm correction no contrast separates. This is "not observable with this design," not "no effect": the sign test needs six discordant cases in one direction, its realized type-I rate under the null is 0.0078, and 80% power needs a per-case win probability near 0.80; closing it takes 12-25 cases, not repetitions. Within a run, a second pass never removes what the first found 
    
[^31]: 信任AI软件工程代理中的信任：为什么、是什么以及如何？

    Trusting Trust in AI Software Engineering Agents: The Why, What, and How?

    [https://arxiv.org/abs/2611.21634](https://arxiv.org/abs/2611.21634)

    该论文的核心贡献是提出了一个结构化信任分析框架，将AI软件工程代理的信任分解为“为什么信任（信任场景）、信任什么（信任配置）、信任多少”三个分析性问题，以克服将信任视为单一可测量属性的局限性。

    

    AI软件工程代理正日益融入开发工作流中，从自动补全助手到自主修复代理，再到多代理软件开发流水线。信任之所以重要，是因为这些代理能够生成、修改并传递工件，而开发者、其他代理或组织随后可能在不确定性和风险之下依赖这些工件。然而，对此类系统的信任通常被视为AI系统的一个单一可测量属性。这对于软件工程领域而言是不够的，因为在软件工程中，信任取决于谁依赖该代理、这种依赖关系如何构建，以及在该部署中必须满足哪些属性。我们提出了一个用于推理AI软件工程代理信任问题的结构化框架。受软件系统中信任的递归性质启发，我们将信任分解为三个分析性问题：为什么信任（Why trust）？——识别信任场景；信任什么（What trust）？——识别信任配置；以及信任多少（How much）？

    arXiv:2611.21634v1 Announce Type: new  Abstract: AI software engineering agents are increasingly integrated into development workflows, from autocomplete assistants to autonomous repair agents and multi-agent software development pipelines. Trust matters because these agents can produce, revise, and pass on artifacts that developers, other agents, or organisations may later rely on under uncertainty and risk. Yet trust in such systems is often treated as a single measurable property of an AI system. This is insufficient for software engineering, where trust depends on who relies on the agent, how that reliance is structured, and which properties must hold in that deployment. We propose a structural framework for reasoning about trust in AI software engineering agents. Inspired by the recursive nature of trust in software systems, we decompose trust into three analytical questions: Why trust?, identifying the trust scenario; What trust?, identifying the trust configuration; and How much
    
[^32]: 将人机对齐作为混合软件工程团队有效协作的早期指标进行测量

    Measuring Human-AI Alignment as an Early Indicator for Effective Collaboration in Hybrid SE Teams

    [https://arxiv.org/abs/2611.21633](https://arxiv.org/abs/2611.21633)

    本文提出开发一套软件工程专用的人机对齐静态测量指标体系，借鉴软件架构、组织心理学和人机交互原理，并利用声明式业务流程监控语言构建静态评估工具，作为评估混合人机团队有效协作的早期指标。

    

    随着AI智能体日益融入软件工程（SE）流程，确保这些混合团队的有效协作成为一项关键挑战。现有指标侧重于观察动态运行时性能，但缺乏针对早期人机对齐——即AI的行为和粒度与人类期望匹配的程度——的系统性静态测量。本立场论文提出了一个研究议程，旨在借鉴软件架构、组织心理学和人机交互的原理，开发一套SE特定的对齐指标体系。此外，本文概述了一种利用业务流程监控（BPM）概念和技术的验证方法，并引入了利用声明式BPM语言的静态评估工具，以确保智能体遵守特定的项目约束。

    arXiv:2611.21633v1 Announce Type: cross  Abstract: As AI agents increasingly integrate into software engineering (SE) processes, ensuring effective collaboration in these hybrid teams becomes a critical challenge. While existing metrics focus on observing dynamic run-time performance, there is a lack of systematic, static measurements for early human-AI alignment -- the degree to which an AI's actions and granularity map to human expectations. This position paper proposes a research agenda to develop an SE-specific alignment metric suite drawing principles from software architecture, organizational psychology, and human-computer interaction. Furthermore, it outlines a validation methodology using concepts and techniques from business process monitoring (BPM) and introduces static assessment tools leveraging declarative BPM languages to ensure agents adhere to specific project constraints.
    
[^33]: 碎片化的指标，喜忧参半的结果：AI编程助手与开发者生产力的系统性综述

    Fragmented Markers, Mixed Results: A Systematic Review of AI Coding Assistants and Developer Productivity

    [https://arxiv.org/abs/2611.21632](https://arxiv.org/abs/2611.21632)

    该论文通过一项分阶段的系统性文献综述，指出现有AI编程助手对开发者生产力影响研究的核心问题在于生产力衡量指标碎片化、缺乏可比性，并基于SPACE框架将分析范围扩展至以人为本的生产力相关成果。

    

    AI编程助手正日益深度融入软件工程实践，然而现有文献仍未形成对其如何影响软件开发者生产力的累积性、易于解读的整体图景。核心问题并非研究的匮乏，而是缺乏可比性：先前的研究通过异质的衡量指标、研究情境和报告风格来操作化生产力概念，导致跨研究得出结论十分困难。为解决这一问题，我们开展了一项分阶段的系统性文献综述。在第一阶段，我们综合了明确考察开发者生产力的研究；在第二阶段，我们以SPACE框架作为有理论依据的视角，将研究范围扩展至与生产力相关的开发者成果，捕捉显性生产力研究常常忽略的以人为本的证据。在两个阶段中，我们分析了生产力是如何被操作化的、讨论了哪些混淆因素，以及已报告的效果呈现怎样的分布。

    arXiv:2611.21632v1 Announce Type: cross  Abstract: AI coding assistants are increasingly embedded in software engineering practice, yet the literature still offers no cumulative, easily interpretable picture of their effects on software developer productivity. The central problem is not a lack of studies, but a lack of comparability: prior work operationalizes productivity through heterogeneous markers, settings, and reporting styles, making cross-study conclusions difficult. To address this, we conduct a staged systematic literature review. In the first stage, we synthesize studies that explicitly examine developer productivity. In the second, we extend the scope to productivity-related developer outcomes using SPACE as a theory-informed lens, capturing human-centered evidence that explicit productivity studies often omit. Across both stages, we analyze how productivity is operationalized, which confounding factors are discussed, and how reported effects are distributed.   We identify
    
[^34]: 面向建筑运营的模型驱动自动化流水线

    A Model-Driven Automation Pipeline for Building Operation

    [https://arxiv.org/abs/2611.21631](https://arxiv.org/abs/2611.21631)

    该论文提出了一个基于模型驱动工程的楼宇控制系统自动化流水线，利用两种领域特定建模语言从平台无关模型自动生成并部署可执行制品，将设计、编程与自动调试整合为四步工作流，以减少人工投入和配置错误。

    

    在建筑、工程、施工与运营（AECO）领域，楼宇控制系统的工程化过程仍然复杂、依赖人工且容易出错，工具割裂和手动调试导致设计与建成现实之间出现偏差，形成性能差距。我们认为，一个覆盖设计、编程和运营环节的完全集成、自动化且由工具支持的工作流是可行的，它能够在减少人工投入和配置错误的同时提升效率。基于模型驱动工程，我们提出了一个用于楼宇控制系统建模、编程和自动调试的自动化流程流水线。该流水线使用两种支持工具的领域特定建模语言，从平台无关模型中综合并部署可执行制品，以定义明确的四步工作流取代割裂的多方交接过程，从而提高工程全生命周期内的一致性。我们的评估报告显示……（原文摘要在此处截断）

    arXiv:2611.21631v1 Announce Type: new  Abstract: Building Control Systems (BCS) engineering in the Architecture, Engineering, Construction & Operations (AECO) domain remains a complex, manual, and error-prone process, with fractured tools and manual commissioning causing design and built reality to diverge into a performance gap. We argue that a fully integrated and automated, tool-supported workflow spanning design, programming, and operation is feasible and can improve efficiency while reducing manual effort and configuration errors. Building on model-driven engineering, we present an automated process pipeline for modeling, programming, and auto-commissioning BCS. It uses two domain-specific modeling languages with tooling support to synthesize and deploy executable artifacts from platform-independent models, replacing a fractured, multi-stakeholder handoff process with a defined four-step workflow and improving consistency across the engineering lifecycle. Our evaluation reports no
    
[^35]: 重用久经考验的代码：自动化软件移植的系统性文献综述

    Reusing Battle-tested Code: A Systematic Literature Review of Automated Software Transplantation

    [https://arxiv.org/abs/2611.21630](https://arxiv.org/abs/2611.21630)

    本文通过系统性文献综述评估了自动化软件移植领域的最新进展，识别出27种现有方法，发现尽管代码适配已高度自动化，但器官定位与测试套件增强等人工前置步骤仍是阻碍该技术走向主流工程实践的主要瓶颈。

    

    软件重用仍然是软件工程中的一个根本性挑战，常常陷于外部库的安全风险与生成式AI的可靠性问题之间。自动化软件移植提供了一种引人注目的替代方案，它能够将久经考验的逻辑（即已在生产环境中经受住严苛考验、可能经过多年广泛实际使用验证的代码）直接提取并集成到新的宿主系统中。本文提出了一项系统性文献综述，以评估该技术的最新研究现状，并识别出阻碍其成为主流工程实践的剩余障碍。我们共识别出27种不同的自动化软件移植方法。我们的分析表明，尽管代码适配环节已高度自动化，但在器官定位和测试套件增强等人工前置步骤中仍存在显著瓶颈。最后，我们提出了一个研究议程。

    arXiv:2611.21630v1 Announce Type: new  Abstract: Software reuse remains a fundamental challenge in software engineering, often caught between the security risks of external libraries and the reliability concerns of generative AI. Automated software transplantation offers a compelling alternative by enabling the direct extraction and integration, into new host systems, of battle-tested logic (code that has already survived the rigors of production, possibly over many years of extensive in-production use). This paper presents a systematic literature review to evaluate the current state-of-the-art and identify the remaining barriers preventing this technology from becoming a mainstream engineering practice. We identified 27 distinct automated software transplantation approaches. Our analysis reveals that while the adaptation of code is highly automated, significant bottlenecks remain in the manual prerequisite steps of organ location and test suite augmentation. We conclude with a researc
    
[^36]: 编程语言与工具的使用模式：一项系统性文献综述

    Usage Patterns of Programming Languages and Tools: A Systematic Literature Review

    [https://arxiv.org/abs/2611.21627](https://arxiv.org/abs/2611.21627)

    本系统性文献综述分析了2015-2024年间的37项研究，发现Java是工业界最流行的编程语言、Travis CI和GitHub Actions是最常用的CI工具，但针对工业界流行语言和工具（尤其是IDE使用）的研究仍然匮乏。

    

    技术以及编程语言和工具的流行程度正在快速演变。因此，持续评估工业界最流行的语言和工具非常重要，这将促进学术界与软件产业之间的进一步合作。在本研究中，我们采用系统性文献综述方法，调查了2015年至2024年间发表的37项研究，这些研究探讨了工业界所使用的编程语言、IDE（集成开发环境）和CI（持续集成）工具。我们观察到，Java最常被列为最流行的编程语言，而针对特定IDE使用情况的研究则较为有限。此外，Travis CI和GitHub Actions是被认定为工业界主流CI工具中最常见的两种CI工具。总体而言，证据表明，目前缺乏针对工业界流行语言和工具的研究，这在很大程度上是由于可能的研究方法存在诸多局限，例如无法访问闭源项目以及抽样受限等。

    arXiv:2611.21627v1 Announce Type: new  Abstract: Technology and the popularity of languages and tools are rapidly evolving. Thus, it is important to constantly assess the most popular languages and tools used in industry, furthering the collaboration between research and the software industry. In this study, we use a systematic literature review and investigate 37 studies published between 2015 -- 2024, which explored the programming languages, IDEs and CI tools used in industry. We observed that Java is most often listed among the most popular languages, and there are limited studies that investigate specific IDE usage. Also, Travis CI and GitHub Actions are the most common CI tools identified as being top CI tools used in industry. Overall, evidence shows that there is a lack of research focused on popular languages and tools used in industry, largely as there are many limitations with the possible methodology approaches. For example, lack of access to closed-source projects and samp
    
[^37]: X-LocVul：级联式C/C++漏洞检测、定位与解释的证据传播审计

    X-LocVul: An Evidence-Propagation Audit of Cascaded C/C++ Vulnerability Detection, Localization, and Explanation

    [https://arxiv.org/abs/2611.21626](https://arxiv.org/abs/2611.21626)

    X-LocVul对C/C++漏洞检测-定位-解释级联流水线进行了以测量为先的严格审计，揭示出级联系统在各阶段间传播的证据往往薄弱且不可靠：真实CWE监督与随机对照无统计学差异，且超过90%的生成语句无法精确匹配源代码，凸显了此类流水线在证据传播上的可信度问题。

    

    面向开发者的漏洞分析应当将函数级警告与源代码位置及有依据的解释相关联。级联系统即使在各组件看似合理的情况下，也可能传播薄弱的证据。我们提出了X-LocVul，这是一项以测量为先的审计，针对结合了UniXCoder、CodeT5和Qwen2.5-Coder的模块化C/C++检测-定位-解释流水线。我们检验在各阶段之间传递的证据究竟是有用的、仅停留在源代码层面的，还是具有误导性的。在重复数据与时间泄露控制条件下，UniXCoder的点估计值高于CodeBERT，但真实CWE监督与打乱CWE的对照组在PR-AUC、MCC和漏洞类别F1指标上在统计学上无法区分。原始生成的语句中有90.83%的情况无法与源代码精确匹配。模糊投影达到83.79%的Top-1准确率，比语义投影高出1.59个百分点，尽管其相对于原始排序的优势尚未得到证实。在一项随机的、条件标注的（实验中……摘要在此处截断）。

    arXiv:2611.21626v1 Announce Type: new  Abstract: Developer-facing vulnerability analysis should link a function-level warning to a source location and a supported explanation. Cascaded systems can propagate weak evidence despite plausible components. We present X-LocVul, a measurement-first audit of a modular C/C++ detector-localizer-explainer pipeline combining UniXCoder, CodeT5, and Qwen2.5-Coder. We test whether evidence passed across stages is useful, merely source-resident, or misleading.   Under duplication and temporal-leakage controls, UniXCoder has higher point estimates than CodeBERT, but true-CWE supervision is statistically indistinguishable from a shuffled-CWE control on PR-AUC, MCC, and vulnerable-class F1. Raw generated statements fail exact source matching in 90.83% of cases. Fuzzy projection reaches 83.79% Top-1 accuracy and exceeds semantic projection by 1.59 percentage points, although its advantage over raw ranking is not established. In a randomized, condition-labe
    
[^38]: 评估基于抽象语法树（AST）的知识图谱检索在Apache Kafka AIOps事件诊断中的应用

    Evaluating AST-Derived Knowledge Graph Retrieval for AIOps Incident Diagnosis in Apache Kafka

    [https://arxiv.org/abs/2611.21625](https://arxiv.org/abs/2611.21625)

    本文在Apache Kafka事件诊断任务上系统评估了基于AST派生知识图谱的检索方法，发现仅向量检索已是强基线（Recall@20达0.956）且图增强方法难以显著超越，而分阶段图扩展可将重排序开销减少约52%。

    

    事件诊断需要从大型代码库中检索与源代码相关联的运维证据。我们使用从抽象语法树（AST）派生的所有权图谱，基于1,600条与源代码关联的日志记录以及315个经审计、涵盖103个独特目标的事件式查询基准，评估了面向Apache Kafka的图增强检索。仅向量检索是一个强基线，Recall@20达到0.956，而无论是静态还是置信度引导的分阶段AST扩展均未能显著提升其Recall@20。作为外部对照，我们评估了HippoRAG（一个已发表的基于文本派生图谱的检索系统）与其自身的ColBERTv2检索骨干：相比仅使用ColBERTv2（Recall@20为0.803），HippoRAG仅达到0.308；事后进行的文档集成敏感性分析将Recall@20提升至0.648，但仍未消除差距。相对于静态扩展，分阶段方法将交叉编码器的配对工作负载和实测重排序时间减少了约52%。（注：原文摘要末尾被截断）

    arXiv:2611.21625v1 Announce Type: cross  Abstract: Incident diagnosis requires retrieving source-linked operational evidence from large codebases. We evaluate graph-augmented retrieval for Apache Kafka using an abstract-syntax-tree (AST)-derived ownership graph over 1,600 source-linked log records and an audited benchmark of 315 incident-style queries covering 103 unique targets. Vector-only retrieval is a strong baseline at Recall@20 of 0.956, and neither static nor confidence-guided staged AST expansion significantly improves Recall@20 over it. As an external control we evaluate HippoRAG, a published text-derived graph-retrieval system, against its own ColBERTv2 retrieval backbone. Against ColBERTv2-only (Recall@20 0.803), HippoRAG attained 0.308; a post-hoc document-ensemble sensitivity raised Recall@20 to 0.648 but did not eliminate the deficit. Relative to static expansion, staging reduces cross-encoder pair workload and measured reranking time by about 52%. A manually evaluated 1
    
[^39]: 面向软件工程的可信AI队友：关于生命周期上下文、可解释性与智能体协作的研究议程

    Trustworthy AI Teammates for Software Engineering: A Research Agenda for Lifecycle Context, Explainability, and Agentic Collaboration

    [https://arxiv.org/abs/2611.21624](https://arxiv.org/abs/2611.21624)

    该论文提出AI辅助软件工程的下一阶段应从被动式编程助手转向可信AI队友，并围绕生命周期感知、有原则的可解释性和智能体协作三大要求构建人机协同演进的研究议程。

    

    大语言模型正在通过支持代码生成、测试、调试、文档编写和程序理解，迅速改变软件工程（SE）的实践方式。然而，当前大多数AI编程助手仍然是被动式工具：它们只响应局部提示、依赖有限的上下文，且其建议背后的理据往往难以被开发者检验。本文认为，AI辅助软件工程的下一阶段需要实现从编程助手到可信AI队友的转变。我们提出了一个关于软件工程中人机协同演进的研究议程，围绕三项要求展开：第一，AI系统必须具备生命周期感知能力，能够跨越需求及其他部署制品进行推理，而非局限于孤立的文件或提示；第二，它们必须提供有原则的解释，以支持开发者判断的方式揭示假设、证据、权衡与风险；第三，它们必须支持智能体协作（agentic collaboration）……

    arXiv:2611.21624v1 Announce Type: new  Abstract: Large Language Models are rapidly changing software engineering (SE) practice by supporting code generation, testing, debugging, documentation, and program comprehension. Yet most current AI coding assistants remain reactive tools: they respond to local prompts, rely on limited context, and provide suggestions whose rationale is often difficult for developers to inspect. This paper argues that the next phase of AI-assisted SE requires a shift from coding assistants to trustworthy AI teammates. We present a research agenda for human-AI co-evolution in SE, organized around three requirements. First, AI systems must become lifecycle-aware, reasoning across requirements and other deployment artifacts rather than isolated files or prompts. Second, they must provide principled explanations that expose assumptions, evidence, trade-offs, and risks in ways that support developer judgment. Third, they must enable agentic collaboration, in which sp
    
[^40]: 软件工程工厂：代码泛滥时代的工程问责与控制

    The Software Engineering Factory: Engineering Accountability and Control in the Age of Abundant Code

    [https://arxiv.org/abs/2611.21623](https://arxiv.org/abs/2611.21623)

    该论文提出“软件工程工厂”框架，利用以常规代码实现的确定性协议引擎将组织控制嵌入代码生产流程之中，从而在AI智能体大规模生成代码的时代实现可问责的代码验收，同时保留人类对重大决策的最终权威。

    

    AI 智能体现在能够以任何人类组织都无法逐行审查的规模来规划、编写、测试、审查和修改代码。因此，最终的代码审查不能再作为掌控代码变更的首要组织控制手段。软件工程工厂（SE Factory）将控制转移到产生代码的过程之中。工厂协议定义了工作如何流转、工作推进之前必须存在哪些证据，以及谁对结果负责。一个用常规代码而非提示词实现的确定性协议引擎，通过控制阶段转换、要求提供证据，并在工作推进之前在需要时调用门控和人工升级机制来执行这些协议。软件工程工厂保留了人类对重大决策的权威，使可问责的验收成为可能，而无需人类监督每一个智能体动作。软件工程工厂并非“黑暗工厂”。人类仍然通过定义范围和粒度来保持控制。

    arXiv:2611.21623v1 Announce Type: new  Abstract: AI agents can now plan, write, test, review, and revise code at a volume no human organization can inspect line by line. Final code review therefore cannot remain the primary organizational control for owning code changes.   The Software Engineering (SE) Factory moves control into the process that produces the code. The Factory protocols define how work moves, what evidence must exist before it can proceed, and who owns the result. A deterministic protocol engine, implemented in conventional code rather than prompts, enforces those protocols by controlling stage transitions, requiring evidence, and invoking gates and human escalation, where needed, before work can advance. The SE Factory preserves human authority over consequential decisions and makes accountable acceptance possible without requiring humans to supervise every agent action.   The SE Factory is not a dark factory. Humans remain in control by defining the scope and granular
    
[^41]: 关于提升静态应用安全测试（SAST）工具的采用率、可用性与持续使用的研究

    On Improving the Adoption, Usability, and Retention of Static Application Security Testing (SAST) Tools

    [https://arxiv.org/abs/2611.21621](https://arxiv.org/abs/2611.21621)

    本论文通过对89项相关研究的文献综述识别出开发人员在使用SAST工具时的主要痛点（如误报等可用性障碍），并提出了提升静态应用安全测试工具的采用率、可用性与持续使用的方法。

    

    随着互联网将世界上越来越多的部分连接起来，软件漏洞带来了日益增长的安全风险。由于软件开发人员负责构建和维护安全的应用程序，他们依赖能够在开发早期帮助识别漏洞的工具和技术。静态应用安全测试（SAST）工具通过分析源代码，在软件部署之前识别潜在的安全弱点。尽管SAST可以帮助开发人员主动应对漏洞，但这些工具也带来了可用性和采用方面的挑战。本论文研究了这些挑战，并探索了改善SAST的采用、可用性和持续使用的方法。首先，对89项与SAST实施和持续使用相关的研究工作进行了文献综述，识别出开发人员反复出现的关注点，包括误报以及其他可能降低对SAST结果信任度的可用性障碍。其次，本论文提出了一个流程……（摘要原文在此处截断）

    arXiv:2611.21621v1 Announce Type: new  Abstract: As the Internet connects more of the world, software vulnerabilities create growing security risks. Because software developers are responsible for building and maintaining secure applications, they rely on tools and techniques that help identify vulnerabilities early in development. Static Application Security Testing (SAST) tools analyze source code to identify potential security weaknesses before software is deployed. Although SAST can help developers address vulnerabilities proactively, these tools also present usability and adoption challenges. This thesis examines those challenges and explores ways to improve the adoption, usability, and continued use of SAST. First, a literature review of 89 research works related to SAST implementation and continued usage identifies recurring developer concerns, including false positives and other usability barriers that can reduce trust in SAST results. Second, the thesis presents a process for 
    
[^42]: 从AIOps流水线到智能体化服务运营：能力、权限与证据的综述

    From AIOps Pipelines to Agentic Service Operations: A Survey of Capability, Authority, and Evidence

    [https://arxiv.org/abs/2611.21620](https://arxiv.org/abs/2611.21620)

    该综述提出以能力、运维权限和评估证据三个相互关联的视角来审视从传统AIOps流水线到智能体化服务运营的转变，并依据实际评估行为而非预期自主性对22个语言模型驱动的系统进行了审计。

    

    现代服务系统集成了快速变化的依赖关系、异构的遥测数据，以及其副作用可能放大故障的运维操作。传统的AIOps研究已经产生了有效的检测器、定位器和诊断模型，但现有综述通常按照任务、遥测模态或学习技术来组织这些方法。这种分类方式只描述了方法能够预测什么，却没有说明它可以做出哪些决策，以及何种证据能够支撑其权限。本文从三个相互关联的视角——能力、运维权限和评估证据——综述了从AIOps流水线向智能体化服务运营的转变。我们首先将SOA、微服务、Kubernetes、服务网格和无服务器平台与故障生命周期联系起来，然后回顾了传统的AIOps方法，并根据22个基于语言模型的系统的实际评估行为（而非其预期的自主性）对它们进行了审计。审计结果表明，大多数系统仍然……（原文摘要在此处截断）

    arXiv:2611.21620v1 Announce Type: new  Abstract: Modern service systems combine rapidly changing dependencies, heterogeneous telemetry, and operational actions whose side effects may amplify an incident. Conventional AIOps research has produced effective detectors, localizers, and diagnostic models, yet surveys typically organize them by task, telemetry modality, or learning technique. Such taxonomies describe what a method predicts, but not which decisions it may make or which evidence justifies that authority. We survey the transition from AIOps pipelines to agentic service operations through three coupled lenses: capability, operational authority, and evaluation evidence. We first relate SOA, microservices, Kubernetes, service meshes, and serverless platforms to the incident lifecycle. We then review conventional AIOps methods and audit 22 language-model-enabled systems according to their evaluated behavior rather than their intended autonomy. The audit shows that most systems remai
    
[^43]: 软件故障间平均时间的随机建模：综述

    Stochastic Modeling of the Mean Time Between Software Failures: A review

    [https://arxiv.org/abs/2611.21619](https://arxiv.org/abs/2611.21619)

    本文综述并比较了基于非齐次泊松过程与传染机制的软件故障检测随机模型中故障间平均时间（MTBF）的推导方法，评估其在敏捷开发方法下的适用性，其中传染模型的故障率同时依赖于时间与已检测到的故障数量。

    

    本文对软件故障检测随机过程的若干模型进行了综述。我们重点关注如何从这些模型中推导出故障间平均时间（MTBF），并给出了一项比较研究。所考虑的模型包括一些广为人知的基于非齐次泊松过程的模型，以及近期提出的一种基于传染机制的模型。我们的主要目标是评估这些模型在应用于采用敏捷方法论开发的现代软件项目时的性能。近年来发展起来的软件工程实践（如TDD、Scrum等）提出了开发、测试与调试之间的迭代流程，这给软件可靠性建模带来了特殊的挑战。与非齐次泊松过程不同，传染模型的故障率不仅取决于时间，还取决于先前已检测到的故障数量。所有被考虑的模型均为非齐次……（原文此处被截断）的特例。

    arXiv:2611.21619v1 Announce Type: new  Abstract: A review of several models for the software failure detection stochastic process is per- formed. We focus on how the Mean Time Between Failures is obtained from those models and present a comparative study. The models considered involve some of the well known based on non- homogeneous Poisson processes and a recently proposed model based on contagion. Our main target is to evaluate the models performance when they are applied to modern software projects developed under Agile methodologies. Recently developed Software Engineering practices (TDD, Scrum, etc.), propose an iterative procedure be- tween development, testing and debugging, introducing this way, particular challenges to Software Reliability modeling. As opposed to the non-homogeneous Poisson processes, the failure rate of the contagion model depends not just on time but also on the number of previously detected failures. All the considered models are particular cases of non- ho
    
[^44]: 辩方举证完毕：一个受法庭启发的多智能体大语言模型框架，用于自动化安全论证生成

    The Defense Rests: A Courtroom-Inspired Multi-Agent LLM Framework for Automated Safety Case Generation

    [https://arxiv.org/abs/2611.21618](https://arxiv.org/abs/2611.21618)

    本文提出了一种受法庭审判启发的多智能体大语言模型框架，通过模拟检察官、辩护律师、法官和陪审团等角色协作，实现任务关键系统安全论证的自动化生成。

    

    安全论证（safety cases）被广泛用于证明任务关键系统（如自动驾驶车辆和无人机系统）在部署后能够安全运行。然而，手动构建安全论证过程复杂、耗时且劳动强度大，因此需要提出自动化方法来支持其创建。受安全论证构建的法律属性启发，我们通过法庭审判的类比将安全论证创建问题形式化，其中各利益相关方协作地提出、审查、批评、评审、改进和验证安全主张及其支持证据。在这一类比中，安全分析师扮演检察官的角色，负责识别危害和安全问题；保证论证开发人员则扮演辩护律师的角色，负责构建和证实安全论证；编排器（orchestrator）扮演法官的角色，负责协调各利益相关方之间的互动；评审委员会则扮演陪审团的角色，负责评估……

    arXiv:2611.21618v1 Announce Type: new  Abstract: Safety cases are widely used to demonstrate that mission-critical systems, such as autonomous vehicles and unmanned aerial systems, can operate safely after deployment. However, manually constructing safety cases is complex, time-consuming, and labor-intensive, which calls for the proposal of automated approaches to support their creation. Inspired by the legal nature of safety case construction, we formalize the safety case creation problem through a courtroom analogy, where stakeholders collaboratively propose, examine, criticize, review, improve and validate safety claims and their supporting evidence. In this analogy, the safety analyst acts as a prosecutor by identifying hazards and safety concerns, while the assurance case developer serves as a defense attorney by constructing and substantiating safety arguments. An orchestrator acts as a judge by coordinating stakeholder interactions, and a review board serves as a jury by assessi
    
[^45]: 面向企业集成中技术规范自动生成的思维链提示工程框架：一项真实案例研究

    A Chain-of-Thought Prompt Engineering Framework for Automated Technical Specification Generation in Enterprise Integration: A Real-World Case Study

    [https://arxiv.org/abs/2611.21617](https://arxiv.org/abs/2611.21617)

    该论文提出了一种基于思维链的提示工程框架，通过两阶段生成架构与六阶段提示流水线相结合，利用GitHub Copilot在企业集成项目中自动生成技术规范文档，并在将250多个企业集成迁移至Azure的真实工业场景中验证了其有效性。

    

    技术规范文档将功能需求转化为可实施的设计，是受治理的企业应用集成（EAI）项目中的必备产物，但其制作过程仍然是一个手动的、依赖专业知识的瓶颈。大语言模型（LLM）在代码生成方面已被广泛研究，但长篇的、受模板约束的工程文档生成仍相对缺乏探索。本文报告了一项工业案例研究，该研究是在澳大利亚某大型批发分销商将250多个企业集成迁移至Microsoft Azure集成服务的过程中开展的，提出了一个使用GitHub Copilot自动生成技术规范文档的思维链提示工程框架。该框架结合了两阶段生成架构，将模型锚定于功能规范、组织标准和已批准的模板；并采用一个包含六个阶段的提示流水线……

    arXiv:2611.21617v1 Announce Type: new  Abstract: Technical Specification Documents (TSDs) translate functional requirements into implementable designs and are mandatory artifacts in governed Enterprise Application Integration (EAI) programs, yet their production remains a manual, expertise-intensive bottleneck. Large Language Models (LLMs) have been studied extensively for code generation, but long-form, template-governed engineering documents remain comparatively unexplored. This paper reports an industrial case study, conducted during a migration of more than 250 enterprise integrations to Microsoft Azure Integration Services at a large Australian wholesale distributor, that introduces a chain-of-thought prompt-engineering framework for automated TSD generation using GitHub Copilot. The framework combines a two-phase generation architecture that grounds the model in the functional specification, organizational standards, and the approved template; a six-stage prompting pipeline with 
    
[^46]: 面向编辑自身环境的代理的版本一致代码上下文

    Revision-Consistent Code Context for Agents That Edit Their Environment

    [https://arxiv.org/abs/2611.21616](https://arxiv.org/abs/2611.21616)

    本文提出了一种代理可见的完整修订发布契约，用于解决编辑型代理在同时查询和修改代码库时出现的“混合修订代码上下文”问题，确保单次查询返回的所有视图（源代码、符号、图关系）都来自同一代码库版本。

    

    编辑型代理在查询代码库的同时也在修改代码库。源代码内容、符号、图关系和检索状态可能由不同的存储系统维护，并在不同的时间点刷新。因此，在更新过程中，一次成功的查询可能会组合那些从未属于同一代码库状态的事实。我们将这种情况称为“混合修订代码上下文”。本文针对复合式代码智能查询提出了一种代理可见的完整修订发布契约。在查询可见的发布点之前，服务必须等待、报告不可用，或提供完整的旧修订版本；在该发布点之后，则提供完整的新修订版本。该契约要求在同一次响应中返回的每个视图都共享相同的版本标识和发布边界。我们在Codetrip中实例化了该方法，其简要操作将源代码、符号和图信息联接在一起。在冻结的Gin v1.12.0上的受控比较中，实时源组件……

    arXiv:2611.21616v1 Announce Type: new  Abstract: Editing agents query repositories while simultaneously changing them. Source content, symbols, graph relations, and retrieval state may be maintained by different stores and refreshed at different times. During an update, a successful query can therefore combine facts that never belonged to the same repository state. We call this condition mixed-revision code context.   This paper proposes an Agent-visible complete-revision publication contract for composite code-intelligence queries. Before a query-visible publication point, a service must wait, report unavailability, or serve a complete old revision; after that point, it serves the complete new revision. The contract requires a shared version identity and publication boundary across every view returned in one response.   We instantiate the method in Codetrip, whose brief operation joins source, symbol, and graph facts. In a controlled comparison on frozen Gin v1.12.0, live-source compo
    
[^47]: 错误却自信：位置相关软件中静默失败的自动检测

    Wrong yet Confident: Automated Detection of Silent Failures in Location-Dependent Software

    [https://arxiv.org/abs/2611.21615](https://arxiv.org/abs/2611.21615)

    该论文提出一种名为“往返偏移”的自动化测试预言机，无需人工标注即可在部署规模上检测地理编码器的静默失败，并在真实食品援助部署数据中发现10.0%的坐标对所有地理编码平台均不可见。

    

    我们报告一篇经验论文，将“往返偏移”——一种针对位置相关软件的自动化测试预言机——应用于部署规模，涵盖波多黎各巴亚蒙一次USDA COVID-19食品援助部署中的32,019笔GPS配送交易，对应24,501个唯一坐标。该预言机通过测量输入坐标与经过“先逆后正”地理编码后恢复的坐标之间的偏差，来检验每个地理编码器的隐式空间精度声明，而无需人工标注正确性。我们在这一经过运营验证的语料库上评估了一个社区贡献的和两个商业的地理编码器。该经验揭示了两种不同的失败类别，我们以各自的分母分别报告它们，因为它们不是可比较的量。首先，24,501个唯一坐标中有2,441个（10.0%）对所有三个平台都不可见：一种已被承认且无解决方案的失败……

    arXiv:2611.21615v1 Announce Type: new  Abstract: We report an experience paper applying round-trip offset---an automated testing oracle for location-dependent software---at deployment scale across 32,019 GPS delivery transactions, corresponding to 24,501 unique coordinates, from a single USDA COVID-19 food assistance deployment in Bayam\'on, Puerto Rico. The oracle tests each geocoder's implicit spatial accuracy claim by measuring the deviation between an input coordinate and the coordinate recovered after reverse-then-forward geocoding, without requiring manual labeling of correctness. We evaluated one community-contributed and two commercial geocoders against this operationally validated corpus. The experience surfaces two distinct failure classes, which we report with their respective denominators because they are not comparable quantities. First, 2,441 of the 24,501 unique coordinates (10.0%) are invisible to all three platforms: an acknowledged failure with no resolution, and a pr
    
[^48]: 你的智能体需要知道它的预算吗？一项针对大语言模型编码智能体的安慰剂对照实验

    Does Your Agent Need to Know Its Budget? A Placebo-Controlled Experiment on LLM Coding Agents

    [https://arxiv.org/abs/2611.21614](https://arxiv.org/abs/2611.21614)

    该研究通过包含 5,835 次运行的安慰剂对照析因实验发现，对 LLM 编码智能体强制执行 token 预算会以任务解决率下降 20 个百分点为代价使 token 消耗约降低十二倍，而告知智能体预算的存在不仅毫无帮助，甚至可能轻微损害性能。

    

    各组织在部署自主编码智能体时会施加 token 预算限制，然而预算本身以及告知智能体预算所带来的因果效应尚未被测量。我们在一个冻结的、带时间戳的协议下开展了一项安慰剂对照的析因实验，在两个可执行的程序修复基准上（共 5,835 次运行、两个开源模型、三档上限水平），将预算强制执行与预算告知进行交叉组合。强制执行一个经过校准的上限使任务解决率降低了 20.0 个百分点，同时使每个已解决任务的 token 消耗大约降低了十二倍。告知智能体被强制执行的预算从未带来帮助：安慰剂对照下的风险差在中等上限时为 -2.8 个百分点（95% 置信区间 [-6.2, +0.7]），在严格上限时为 -4.2 个百分点（[-8.1, -0.2]，名义显著、未经多重校正），且所有提示词措辞和逐轮反馈变体的效应都远低于 10 个百分点这一关注的效应量。在施加上限的条件下，观察到的运行间结果方差约为无上限水平的 1.5 倍，尽管其中部分……（原文摘要在此处被截断）

    arXiv:2611.21614v1 Announce Type: new  Abstract: Organizations deploying autonomous coding agents impose token budgets, yet the causal effects of a budget, and of telling the agent about it, are unmeasured. We conduct a placebo-controlled factorial experiment under a frozen, timestamped protocol, crossing budget enforcement with budget disclosure on two executable program-repair benchmarks (5,835 runs, two open models, three cap levels). Enforcing a calibrated cap reduced task resolution by 20.0 percentage points while reducing tokens per resolved task roughly twelvefold. Disclosing the enforced budget never helped: the placebo-controlled risk difference was -2.8 pp (95% CI [-6.2, +0.7]) at the medium cap and -4.2 pp ([-8.1, -0.2], nominally significant, unadjusted) at the tight cap, with every prompt wording and per-turn feedback variant far below the 10-point effect size of interest. Observed run-to-run outcome variance was roughly 1.5x the uncapped level under caps, though part of t
    
[^49]: 探索跨编程语言的测试坏味：一项系统性映射研究

    Exploring Test Smells Across Programming Languages: A Systematic Mapping Study

    [https://arxiv.org/abs/2611.21613](https://arxiv.org/abs/2611.21613)

    该研究通过系统性映射方法分析了截至2025年8月的117篇论文，首次从编程语言视角系统审视测试坏味，填补了测试坏味与特定编程语言及测试框架关系研究的空白。

    

    测试对于确保软件开发中的代码质量至关重要。然而，不良的实现实践可能会损害测试代码的可维护性和演化，导致额外的成本和工作量。这些实践被称为测试坏味（test smells），已成为众多研究的焦点，这些研究扩展了已知测试坏味的目录。尽管取得了这些进展，但从编程语言的角度审视测试坏味却很少受到关注，特别是它们与特定语言或测试框架之间的关系。因此，研究人员和从业者提出了各种方法和工具来检测和重构不同语言中的相同测试坏味。在这项工作中，我们通过从编程语言的角度分析文献中的测试坏味来填补这一空白。我们开展了一项系统性映射研究，收集了截至2025年8月发表的论文数据，共审查了117篇论文。我们识别了……

    arXiv:2611.21613v1 Announce Type: new  Abstract: Tests are essential for ensuring code quality in software development. However, poor implementation practices can compromise the maintainability and evolution of test code, leading to additional costs and effort. These practices, known as test smells, have been the focus of numerous studies that expanded the catalog of known test smells. Despite this progress, little attention has been given to examining test smells from the perspective of programming languages, particularly their relationship with specific languages or test frameworks. As a result, researchers and practitioners have proposed various approaches and tools to detect and refactor the same test smells across different languages. In this work, we address this gap by analyzing test smells in the literature from a programming language perspective. We conducted a systematic mapping study, gathering data from papers published up to August 2025, examining 117 papers. We identified
    
[^50]: 反编译的系统性综述：经典方法与机器学习方法的映射

    A Systematic Survey of Decompilation: Mapping Classical and Machine-Learning Approaches

    [https://arxiv.org/abs/2611.21612](https://arxiv.org/abs/2611.21612)

    本文对1973至2026年的405项反编译研究进行了系统性综述，将经典方法、ML训练方法和ML提示方法映射到八阶段的流水线中心分类法中，并分析了116项基于机器学习的研究在技术设计与评估证据上的差异。

    

    背景：反编译器从编译后的二进制文件中重构源代码级代码，当源代码不可用时，分析人员依赖它们完成分析工作。近期许多研究将机器学习（ML），包括大型语言模型，应用于该任务的部分环节。然而仅从原始文献来看，这些方法在经典反编译流水线中所处的位置并不清晰，其结论所获支持的程度也不明确。目标：我们将反编译研究映射到一个以流水线为中心的分类法中，并回答两个研究问题：经典方法、基于ML训练的方法和基于ML提示的方法分别处于什么位置（RQ1），以及基于ML的研究在技术设计和评估证据方面存在怎样的差异（RQ2）。方法：我们通过数据库搜索和引用追踪，调研了1973年至2026年间的405项反编译研究，将其中394项主要研究划分为八个流水线阶段和两个互补类别。对于提出或改进反编译方法的116项基于ML的研究，我们……

    arXiv:2611.21612v1 Announce Type: new  Abstract: Context: Decompilers reconstruct source-level code from compiled binaries, and analysts rely on them whenever source code is unavailable. Many recent studies apply machine learning (ML), including large language models, to parts of this task. From the primary literature alone, it is unclear where these methods sit in the classical decompilation pipeline and how well their claims are supported. Objectives: We map decompilation research onto a pipeline-centered taxonomy and ask two questions: where classical, ML-trained, and ML-prompted approaches are positioned (RQ1), and how ML-based studies vary in technical design and evaluation evidence (RQ2). Methods: We surveyed 405 decompilation studies from 1973 to 2026, collected through database search and citation tracing. We classified 394 primary studies into eight pipeline stages and two complementary categories. For the 116 ML-based studies that propose or modify decompilation methods, we e
    
[^51]: 规格驱动的属性测试（Spec-Driven PBT）：从代码规格说明中推导语义不变量以实现智能体式基于属性的测试

    Spec-Driven PBT: Deriving Semantic Invariants from Code Specifications for Agentic Property-based Testing

    [https://arxiv.org/abs/2611.15920](https://arxiv.org/abs/2611.15920)

    该论文提出规格驱动的属性测试方法，利用智能体编程中已有的自然语言规格说明（而非仅从代码实现）来推导语义不变量，从而更有效地测试AI生成的代码。

    

    软件越来越多地由AI智能体编写，即智能体根据自然语言规格说明生成代码实现，这种做法被称为智能体编程。如今代码的产生速度已超过人工审查的速度，并且已被证明会表现出其特有的失败模式。基于属性的测试是一种历史悠久的测试技术，它通过大量自动生成的输入来检验程序的不变属性。最近，这一技术被重新审视，由LLM智能体来编写测试属性，从而消除了长期以来阻碍其推广的高成本不变量推导工作。然而，现有方法仅从代码实现本身推导这些属性（即代码驱动的PBT）。与此同时，详尽的规格说明（如产品需求文档）已成为智能体编程流程中的常规产物，但其在测试所生成代码中的作用仍鲜有探索。我们研究了规格说明作为测试属性来源的价值……

    arXiv:2611.15920v1 Announce Type: cross  Abstract: Software is increasingly written by AI agents that produce an implementation from a natural language specification, a practice known as agentic coding. Code is now produced faster than it can be reviewed, and has been shown to exhibit its own characteristic failure patterns. Property-based testing (PBT), a long-standing technique in which invariant properties of a program are checked against many generated inputs, has recently been revisited with LLM agents authoring the properties, thereby removing the high-effort derivation of invariants that has long hindered its adoption. Existing methods, however, derive these properties from the implementation alone (Code-driven PBT). At the same time, a thorough specification such as a product requirements document has become routine within agentic coding, yet its role in testing the resulting code remains underexplored. We study the value of the specification as a source of properties for testi
    
[^52]: OverServe：通过工具—提示词协同设计对SaaS智能体实施的资源放大攻击

    OverServe: Resource Amplification Attacks on SaaS Agents via Tool--Prompt Co-Design

    [https://arxiv.org/abs/2611.14808](https://arxiv.org/abs/2611.14808)

    提出了OverServe多智能体攻击框架，通过协同设计MCP状态工具与完成确认提示词，诱导SaaS智能体在业务任务完成后仍反复检查未完成条件，从而放大对服务提供商资源的消耗。

    

    大型语言模型（LLM）的进步使软件即服务提供商能够借助智能体自动化更长的业务工作流。在基于订阅的SaaS部署中，模型调用和工具调用会消耗提供商的资源，而客户无需为此额外付费。由于智能体自主决定何时停止执行，恶意租户可以向智能体发送提示词，利用其完成判定机制来延长执行时间并浪费提供商的资源。在本文中，我们提出了OverServe，一个用于构建此类针对SaaS智能体攻击的多智能体框架。OverServe为攻击者协同设计模型上下文协议（MCP）状态工具与完成确认提示词。每次工具调用都会返回准确的状态并正常终止，但智能体可能会为了满足提示词的要求而反复检查尚未完成的条件。这使得即使在原始业务任务完成之后，SaaS提供商的资源仍会被持续消耗。

    arXiv:2611.14808v1 Announce Type: cross  Abstract: Advances in large language models (LLMs) are enabling software-as-a-service (SaaS) providers to automate longer business workflows with agents. In subscription-based SaaS deployments, model and tool calls consume provider resources without additional customer payments. Because the agent autonomously decides when to stop, a malicious tenant can send a prompt to the agent, exploiting its completion decisions to extend execution and waste provider resources. In this paper, we present OverServe, a multi-agent framework for constructing such attacks against SaaS agents. OverServe co-designs Model Context Protocol (MCP) status tools and completion-confirmation prompts for the attacker. Each tool call returns accurate status and terminates normally, yet the agent may repeatedly check an unfinished condition to satisfy the prompt. This consumes the SaaS provider's resources even after the original business task is completed. OverServe construc
    
[^53]: 国家与州层面的自动驾驶车辆认证方法：州级运行治理的比较框架

    National and State-Level Approaches to Automated Vehicle Certification: A Comparative Framework for State Operational Governance

    [https://arxiv.org/abs/2611.08681](https://arxiv.org/abs/2611.08681)

    该论文构建了一个结构化比较框架，分析美国十个州对SAE L4-L5级自动驾驶系统的认证与治理方法，并结合欧盟、中国、日本和澳大利亚的国际实践，为北卡罗来纳州的自动驾驶监管战略提供决策参考。

    

    美国自动驾驶车辆（AV）的部署正在不断扩展，但缺乏统一的联邦认证框架，导致各州在授权和监督自动驾驶车辆测试与部署方面存在显著差异。本文考察了美国各州如何监管美国汽车工程师学会（SAE）L4-L5级自动驾驶系统（ADS），识别了常见的认证方法，将其与国际实践进行比较，并评估了这些做法对北卡罗来纳州未来监管战略的启示。研究审查了公开可获得的州法律、行政法规、机构指南以及联邦政策文件。为涵盖不同监管方法的范围，选取了十个州，采用涵盖运行范围、治理与监督、联邦互动以及实施成果的结构化框架进行分析。同时，还审查了欧盟、中国、日本和澳大利亚的认证实践以进行比较分析。

    arXiv:2611.08681v1 Announce Type: cross  Abstract: Automated vehicle (AV) deployment in the United States is expanding without a unified federal certification framework, leading to substantial variation in how states authorize and oversee AV testing and deployment. This paper examines how U.S. states regulate Society of Automotive Engineers (SAE) Level 4-5 automated driving systems (ADS), identifies common certification approaches, compares them with international practices, and evaluates the implications for North Carolina's future regulatory strategy. Publicly available state statutes, administrative regulations, agency guidance, and federal policy documents were reviewed. Ten states, selected to capture the range of regulatory approaches, were analyzed using a structured framework covering operational scope, governance and oversight, federal interaction, and implementation outcomes. Certification practices in the European Union, China, Japan, and Australia were reviewed for comparat
    
[^54]: 持久递归世界实现软件自主演化

    Persistent Recursive Worlds Enable Autonomous Software Evolution

    [https://arxiv.org/abs/2608.10450](https://arxiv.org/abs/2608.10450)

    本文提出EvoX Genesis框架，通过将软件表示为持久递归世界，使项目持久化而代理有限寿命，实现了无需人工干预的自主软件演化，并成功构建了大型编译器。

    

    复杂软件系统的发展时间尺度超过了任何单个编码代理的寿命。大多数代理软件系统通过持久会话、记忆、管理器或共享上下文来保持连续性。我们引入了EvoX Genesis（以下简称Genesis），它反而使软件项目持久化，同时允许本地代理保持有限寿命。Genesis将软件表示为持久递归世界：每个本地世界由已接受的版本和仓库路径定位，有限寿命的代理提出局部更改，递归委托在路径间移动工作，只有被接受的后果推进持久版本历史。我们评估了这种组织在形成、延续和再开发中的表现。从一个没有编译器实现的仓库开始，Genesis使用DeepSeek V4 Flash构建了一个基于Rust的C编译器，约25万行跟踪代码；运行持续超过120小时，arch...

    arXiv:2608.10450v2 Announce Type: replace-cross  Abstract: Complex software systems develop over timescales that exceed the lifespan of any individual coding agent. Most agentic software systems preserve continuity through persistent sessions, memories, managers or shared context. We introduce EvoX Genesis (hereafter, Genesis), which instead makes the software project persistent while allowing local agents to remain finite-lived. Genesis represents software as a persistent recursive world: each local world is situated by an accepted version and a repository path, finite-lived agents propose local changes, recursive delegation moves work across paths, and only accepted consequences advance the persistent version history. We evaluate this organization across formation, continuation and redevelopment. Starting from a repository with no compiler implementation, Genesis used DeepSeek V4 Flash to build a Rust-based C compiler with about 250k tracked lines; the run lasted over 120 hours, arch
    
[^55]: LLM时代研究生研究实践与技能的灰色文献综述

    A Grey Literature Review of Graduate Students' Research Practices and Skills in the LLM Era

    [https://arxiv.org/abs/2607.29519](https://arxiv.org/abs/2607.29519)

    该研究通过对Reddit讨论的灰色文献综述，识别出研究生使用LLM开展研究的七类实践和九类所需技能，为研究生培养项目教授、指导和评估LLM使用能力提供了依据。

    

    背景：计算领域的研究生如今在整个研究过程中使用大型语言模型（LLM），涵盖文献综述、编码、数据分析和写作等环节。然而，对于熟练使用这些工具需要哪些技能，目前缺乏共识性理解，导致研究生培养项目难以对其开展教学、指导或评估，且未经核实的输出可能会渗透进科学记录中。目的：我们调查研究生在其研究中采用了哪些基于LLM的实践，以及他们需要哪些技能才能从这些实践中受益。方法：我们对Reddit讨论进行了灰色文献综述。我们从五个面向研究的子版块中检索了1,383个帖子，依据明确的纳入与排除标准进行筛选，并对纳入的54个帖子开展主题分析，由两位作者独立地对每个摘录进行编码。结果：我们识别出七类实践和九类技能。

    arXiv:2607.29519v2 Announce Type: replace  Abstract: Context: Graduate students in computing now use Large Language Models (LLMs) throughout their research, from literature reviews and coding to data analysis and writing. Yet there is little shared understanding of which skills competent use of these tools requires, so graduate programs struggle to teach, mentor, or assess it, and unverified outputs can propagate into the scientific record.   Objective: We investigate which LLM-based practices graduate students employ in their research and which skills they need to benefit from these practices.   Method: We conducted a grey literature review of Reddit discussions. We retrieved 1,383 posts from five research-oriented subreddits, screened them against explicit inclusion and exclusion criteria, and analyzed the 54 included posts with thematic analysis, in which two authors independently coded each excerpt.   Results: We identified seven categories of practices and nine categories of skill
    
[^56]: IH-Benchmark：一个以冲突为中心的大型语言模型指令层级鲁棒性基准测试

    IH-Benchmark: A Conflict-Centered Benchmark for Instruction-Hierarchy Robustness in Large Language Models

    [https://arxiv.org/abs/2607.25987](https://arxiv.org/abs/2607.25987)

    提出了IH-Benchmark，一个以冲突为中心的基准测试，通过七个约束类别和统一的二元评估协议，系统评估大语言模型在直接系统-用户冲突及工具中介用户-工具冲突下的指令层级鲁棒性。

    

    当语言模型收到来自不同优先级层级的冲突指令时，它实际上会遵循哪一条？这个问题是大语言模型可靠部署的核心。现有基准测试只能部分回答这一问题，通常仅关注单一的层级组件，或者基于工具调用覆盖有限公共数据集进行改编。我们提出了IH-Benchmark，这是一个以冲突为中心的指令层级鲁棒性基准测试，涵盖直接的系统≻用户（S≻U）冲突和工具中介的用户≻工具（U≻T）冲突。我们根据所评估的高优先级要求将测试组织为七个约束类别。每个测试均采用统一的二元通过/失败评估协议，该协议结合了谓词领域特定语言（DSL）与基于评分标准的LLM评判者。我们报告类别平衡的合规率，并单独评估非冲突对照组。我们对32个模型配置的评估……

    arXiv:2607.25987v3 Announce Type: replace-cross  Abstract: When a language model receives conflicting instructions from different priority levels, which one does it actually follow? This question lies at the heart of reliable LLM deployment. Existing benchmarks answer this only partially, often focusing on a single hierarchy component or adapting public datasets with limited tool-use coverage. We present IH-Benchmark, a conflict-centered benchmark for instruction-hierarchy robustness across direct System $\succ$ User (S $\succ$ U) conflicts and tool-mediated User $\succ$ Tool (U $\succ$ T) conflicts. We organize tests into seven constraint categories according to the higher-priority requirements under evaluation. Each test is evaluated with a uniform binary pass/fail protocol combining a predicate domain-specific language (DSL) with rubric-based LLM judges. We report category-balanced compliance rates and evaluate non-conflict controls separately. Our evaluation of 32 model configurati
    
[^57]: 在智能体软件开发中使用第三方API路由器的权衡在哪里？

    Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?

    [https://arxiv.org/abs/2607.23624](https://arxiv.org/abs/2607.23624)

    该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。

    

    第三方API路由器已成为一种常见的中间层，统一了对日益多样化的LLM提供商的访问。在编码智能体工作流中，高自主性操作被广泛采用，因为它减少了交互开销。因此，位于智能体与上游提供商之间的第三方API路由器不可避免地占据了可信路径。它可以检查和修改每一个请求与响应，然而目前没有任何机制来验证提供商的输出与智能体最终执行的仓库级操作之间的一致性。因此，客户端权限机制在实践中可能变得无效。这种控制缺口是否会对软件开发任务产生真实的、难以检测的影响，此前尚未得到实证测量。在本文中，我们对编码智能体中的路由器端注入进行了实证研究，考察了四个隐蔽程度递增的干预级别：响应替换（L1）、响应追加……

    arXiv:2607.23624v3 Announce Type: replace-cross  Abstract: Third-party API routers have become a common layer that unifies access across increasingly diverse LLM providers. In coding-agent workflows, high-autonomy operation is widely adopted because it reduces interaction overhead. As a result, a third-party API router, which sits between the agent and the upstream provider, inevitably occupies the trusted path. It can inspect and modify every request and response, yet no mechanism verifies alignment between the provider's output and the repository-level actions ultimately executed by the agent. Consequently, client-side permission mechanisms may become ineffective in practice. Whether this control gap produces real, hard-to-detect effects on software development tasks remains empirically unmeasured. In this paper, we conduct an empirical study of router-side injection in coding agents, examining four intervention levels of increasing subtlety: Response Substitution (L1), Response Appe
    
[^58]: AgentLens：揭示软件工程智能体评估中的“幸运通过”问题

    AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation

    [https://arxiv.org/abs/2605.12925](https://arxiv.org/abs/2605.12925)

    本文提出AgentLens过程级评估框架并发布AgentLens-Bench数据集，发现10.7%通过测试的智能体轨迹存在“幸运通过”问题（如回归循环、盲目重试、缺失验证等），证明仅以测试是否通过来评估软件工程智能体是不充分的。

    

    软件工程（SWE）智能体的评估主要受一个二元信号主导：最终补丁是否通过测试。这种仅看结果的观点将有原则的解决方案与混乱的试错过程视为等同。我们证明这种等同性在经验上是错误的。我们在60个SWE-bench Verified任务上评估了来自八个模型后端的2,614条OpenHands轨迹。其中47个任务拥有足够多的通过轨迹来构建任务级的过程参考，从而得到一个包含1,815条轨迹的评估子集。在该子集的通过轨迹中，有10.7%表现出我们称之为“幸运通过”的行为：回归循环、盲目重试、缺失验证，或者探索、实现与验证在时间顺序上的混乱。我们提出了AgentLens，一个用于SWE智能体轨迹过程级评估的框架，并发布了AgentLens-Bench数据集，其中包含1,815条标注了质量分数、浪费信号和分歧信息的轨迹。

    arXiv:2605.12925v5 Announce Type: replace-cross  Abstract: Evaluation of software engineering (SWE) agents is dominated by a binary signal: whether the final patch passes the tests. This outcome-only view treats a principled solution and a chaotic trial-and-error process as equivalent. We show that this equivalence is empirically false. We evaluate 2,614 OpenHands trajectories from eight model backends on 60 SWE-bench Verified tasks. Of these, 47 have enough passing trajectories to construct task-level process references, yielding a 1,815-trajectory evaluation subset. Among passing trajectories in this subset, 10.7% exhibit behavior we call a Lucky Pass: regression cycles, blind retries, missing verification, or temporally disordered exploration, implementation, and verification. We introduce AgentLens, a framework for process-level assessment of SWE-agent trajectories, and release AgentLens-Bench, a dataset of 1,815 trajectories annotated with quality scores, waste signals, divergence
    
[^59]: EmbeWebAgent：将网络智能体嵌入自定义企业用户界面

    EmbeWebAgent: Embedding Web Agents into Custom Enterprise UIs

    [https://arxiv.org/abs/2602.14865](https://arxiv.org/abs/2602.14865)

    EmbeWebAgent通过轻量级前端钩子和可复用的后端工作流，将网络智能体直接嵌入企业现有UI中，实现与技术栈无关的混合粒度操作，并借助MCP工具完成真实界面中的多步骤行为。

    

    大多数网络智能体在人类界面层面运行，只能观察截图或原始DOM树，而不具备应用级访问权限，这限制了其鲁棒性和动作表达能力。然而，在企业环境中，可以显式地控制前端和后端。我们提出EmbeWebAgent，这是一个将智能体直接嵌入现有UI的框架，它使用轻量级前端钩子（精心设计的基于ARIA和URL的观察，以及通过WebSocket暴露的每页函数注册表）和可复用的后端工作流来执行推理并采取行动。EmbeWebAgent与技术栈无关（例如支持React或Angular），支持从GUI原语到更高层级复合操作的混合粒度动作，并通过MCP工具协调导航、操作和领域特定的分析。我们的演示展示了最小的改造工作量以及在真实UI环境中的多步骤行为。演示视频：https://youtu.be/Cy06Ljee1JQ

    arXiv:2602.14865v2 Announce Type: replace  Abstract: Most web agents operate at the human interface level, observing screenshots or raw DOM trees without application-level access, which limits robustness and action expressiveness. In enterprise settings, however, explicit control of both the frontend and backend is available. We present EmbeWebAgent, a framework for embedding agents directly into existing UIs using lightweight frontend hooks (curated ARIA and URL-based observations, and a per-page function registry exposed via a WebSocket) and a reusable backend workflow that performs reasoning and takes actions. EmbeWebAgent is stack-agnostic (e.g., React or Angular), supports mixed-granularity actions ranging from GUI primitives to higher-level composites, and orchestrates navigation, manipulation, and domain-specific analytics via MCP tools. Our demo shows minimal retrofitting effort and multi-step behaviors grounded in a live UI setting. Demo Video: https://youtu.be/Cy06Ljee1JQ
    

