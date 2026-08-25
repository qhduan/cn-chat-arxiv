# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SWE Refactor Bench: Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration?](https://arxiv.org/abs/2608.23564) | 本文提出了SWE重构基准，通过三阶段评估协议解决现有基准无法检测迁移是否真正发生的问题，从而衡量编码代理在长时程全仓库栈迁移中的能力。 |
| [^2] | [Prime Agent: A Self-Improving RLM Harness](https://arxiv.org/abs/2608.23552) | Prime Agent是一个开源工具框架，通过持久化REPL和递归子代理机制，将长期评估和编码代理工作流标准化，从而防止工具故障干扰模型，最大化模型潜力。 |
| [^3] | [An Interactive Agent for Requirement-Driven Candidate Sourcing](https://arxiv.org/abs/2608.23501) | 该论文提出了一种交互式需求驱动的候选人搜寻代理和配套基准，将模糊的人员搜索视为需求工程任务，通过引导、验证和确认流程来生成合理的候选名单。 |
| [^4] | [Right-Sizing LLM-Agent Decomposition in VAT Determination: A Pilot Controlled Sweep](https://arxiv.org/abs/2608.23395) | 该论文通过受控扫描实验发现，在增值税确定任务中，中等数量的智能体分解配置（而非极端窄或宽）表现最优，平衡了提示预算与智能体数量效应。 |
| [^5] | [Formalizing and Automating Fine-Grained Move Refactorings Across Methods](https://arxiv.org/abs/2608.23377) | 该论文形式化了五种跨方法的语句级移动重构变体，并通过迭代细化提出了额外前置条件，实现了细粒度表达式移动的自动化，在十个项目上验证了其适用性和可编译性。 |
| [^6] | [DPIAgent: Divide, Protocol, Isolate for Agentic Reproduction Test Generation](https://arxiv.org/abs/2608.23341) | 本文提出DPIAgent框架，通过将复现测试生成任务分解为缺陷探索和测试生成两个单目标阶段，并采用协议交接和工具隔离，有效解决了智能体在复合目标下的目标漂移问题。 |
| [^7] | [From Natural Language Policies to Executable Obligations: A Verification Harness for Dependable In-Car LLM Agents](https://arxiv.org/abs/2608.23282) | 本文提出AgentGuardUtil，通过运行时政策编译器将自然语言政策转换为可执行义务，并用确定性门控和验证循环确保车载LLM智能体可靠遵守操作规范。 |
| [^8] | [An Empirical Study of the TianoCore Community](https://arxiv.org/abs/2608.23280) | 本研究通过调查和访谈揭示了TianoCore社区固件开发中的安全与维护差距，并提出通过改进安全实践、采用内存安全技术和自动化来强化UEFI固件。 |
| [^9] | [TianoForge: An Automated Bug Triage Approach for the TianoCore UEFI Firmware Development Community](https://arxiv.org/abs/2608.23259) | TianoForge通过结合GPT大型语言模型和检索增强生成技术，自动化了UEFI固件开发中的缺陷分类流程，显著提高了缺陷处理效率。 |
| [^10] | [LLMCrater: Lifecycle-Aware FAIR Metadata Generation using Large Language Models](https://arxiv.org/abs/2608.23158) | 该论文提出LLMCrater框架，利用大型语言模型和阶段特定的RO-Crate配置文件，在研究生命周期的四个阶段自动生成并逐步丰富FAIR元数据，以解决现有方法仅在发表时生成元数据而错过上下文信息的问题。 |
| [^11] | [An AI-Assisted Migration Framework for Transforming Legacy Scientific Applications into Reusable Cloud-Based Workflows](https://arxiv.org/abs/2608.23146) | 该论文提出一种结合RM-ODP、LLM和DSM的AI辅助框架，能系统地将遗留科学应用迁移为可复用的云原生工作流，解决了现有方法仅针对单个工件且缺乏系统性的问题。 |
| [^12] | [From Metrics to Improvement: A Lifecycle-Aware LLM Feedback Framework for Research Software Quality](https://arxiv.org/abs/2608.23118) | 该论文提出了一种生命周期感知框架，结合定量质量评估与LLM驱动的代码精炼，将研究软件质量指标转化为可操作的改进建议。 |
| [^13] | [A Multi-Viewpoint Modeling Framework for Digital Twin Integration and Reuse with LLM-Assisted Compatibility Analysis](https://arxiv.org/abs/2608.23115) | 本文提出一个基于RM-ODP的多视角建模框架，结合大语言模型辅助的兼容性分析，以支持数字孪生中异构模型的系统化集成与复用，并实现早期可行性评估。 |
| [^14] | [ARGUS: MCP-Grounded Root Cause Analysis for Kubernetes Incidents](https://arxiv.org/abs/2608.23084) | ARGUS通过标准化MCP服务器连接商业LLM到实时Kubernetes可观测数据，在Slack中提供结构化诊断摘要，实现了跨组织可复用的自动化根因分析。 |
| [^15] | [AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces](https://arxiv.org/abs/2608.23041) | AutoSaddler通过将框架优化视为离线学习问题，利用失败轨迹诊断和代码式补丁生成，自动迭代改进代理框架，在多个基准上显著提升性能。 |
| [^16] | [PatchWrite: One Line, Not One Section -- Compile-Gated, Validity-Preserving Editing for AI-Drafted Manuscripts](https://arxiv.org/abs/2608.23001) | PatchWrite通过编译门控和证据锁机制，仅允许局部编辑而非整节重写，从而在修复手稿缺陷时严格保留无关内容，显著提升编辑的有效性和安全性。 |
| [^17] | [Fairness Hazard Analysis for Socio-Technical Processes: A Multiple-Case Study in Bias-sensitive Organisational Settings](https://arxiv.org/abs/2608.22978) | 本文提出并验证了公平性危害分析方法论，该方法能在需求工程阶段系统识别和缓解社会技术过程中的公平性风险，以支持主动的公平性设计。 |
| [^18] | [Execution-Anchored Hallucination Calibration Reranking for Verilog Code Generation](https://arxiv.org/abs/2608.22938) | 本文提出了一种基于执行锚定的幻觉校准重排序方法，结合执行信号和推理信号，有效解决了Verilog代码生成中候选选择时的测试平台质量低和推理幻觉问题。 |
| [^19] | [Concepts for Securing Agentic AI Coding and the Terok Environment](https://arxiv.org/abs/2608.22930) | 本文评估了代理式AI编码的IT安全风险，提出了在不损害其优势的前提下缓解风险的概念，并概述了Terok环境中的实现方案。 |
| [^20] | [Evaluating Inference-Time Defenses Against Package Hallucination in LLM-Generated Code](https://arxiv.org/abs/2608.22652) | 本文发现现有评估方法高估了LLM代码生成中的包幻觉率，并提出RAG等推理时防御方法能有效降低幻觉，在多数模型和语言组合中表现最佳。 |
| [^21] | [Do Not Copy/Paste: Soft Barriers for Copying in AI-Assisted Programming](https://arxiv.org/abs/2608.22638) | 本文提出“软性障碍”机制，通过Unicode扰动等技术手段，在AI编程助手中管理模型代码向软件的交接，以平衡生成速度与代码审查、安全等需求。 |
| [^22] | [Benchmarking the Titans: A Multi-Dimensional Empirical Evaluation of LLM Code Generation Quality in the .NET Ecosystem](https://arxiv.org/abs/2608.22529) | 本文提出了一个多维度C#代码生成评估框架，对比GPT、Gemini、Claude和Grok四个LLM在85个任务上的表现，弥补了现有基准仅关注Python和单一Pass@k指标的不足。 |
| [^23] | [KONTOGRAPH: Verified Point-in-Time Feature Consistency and Amortised Explanation for Real-Time Anti-Money Laundering under a 200 ms Decision Budget](https://arxiv.org/abs/2608.22389) | 本文提出KONTOGRAPH，一个在200毫秒严格预算下运行的实时反洗钱流水线，通过时间图网络和节点内存显著提升检测性能，并满足欧盟即时支付法规要求。 |
| [^24] | [Learning Spectral Representations of Code through Latent Graph Learning for Generalizable Cross-Language Code Clone Detection](https://arxiv.org/abs/2608.22383) | 本文提出SPECTRA-Siam，一种通过潜在图学习生成跨语言可比较的谱表示，从而提升代码克隆检测泛化能力的新方法。 |
| [^25] | [Mitigating Error Propagation in Chain-of-Thought: A Tree-of-Thought Framework for Smart Contract Repair](https://arxiv.org/abs/2608.22345) | 提出一种结合文档解析、静态分析和思维树推理的智能合约修复框架，通过并行探索修复路径和错误消除，显著提高了修复成功率。 |
| [^26] | [Learning from the Test: Self-Referential Differential Testing for Deep RL Agents](https://arxiv.org/abs/2608.22284) | 我们提出Delta框架，通过两阶段方法自动识别DRL代理的安全关键性和最优性缺陷，解决最优性测试的预言机问题。 |
| [^27] | [CodeMechanic: Bug-Property-Guided Program Mitigation](https://arxiv.org/abs/2608.22275) | CodeMechanic通过从崩溃中重建内存安全属性并插入本地失败停止防护，生成受限缓解，避免了LLM修复代理因弱验证导致的不可靠修复。 |
| [^28] | [D-Diff: An Interactive Environment for Adjusting Commit Boundaries Based on an Editable 3-way Diff](https://arxiv.org/abs/2608.22207) | D-Diff通过将两个连续提交的差异整合到一个可编辑的三方差异视图中，显著降低了调整提交边界的认知负担。 |
| [^29] | [Disagree to Explore, Agree to Commit: Routing-Guided Test-Time Scaling for Software Agents](https://arxiv.org/abs/2608.22191) | 提出Risa方法，利用原生MoE路由信号在测试时引导软件代理的探索与收敛，无需外部评判或测试执行，显著提升仓库级任务修复效率。 |
| [^30] | [Towards Actionable Visualization: Ten Years Later, What Generative AI Changes and What It Cannot](https://arxiv.org/abs/2608.22151) | 本文回顾十年前软件可视化研究，指出生成式AI通过降低按需生成成本，将重点从生产人工制品转向人类感知与指导，并发现AI使被忽视领域（如理由）变得可行，但工具可持续性问题可能先恶化后改善。 |
| [^31] | [Constraint-Driven Modeling Enabling Dual Model Checking and Simulation for Discrete Event Systems](https://arxiv.org/abs/2608.22095) | 本文提出Constraint-DEVS方法，通过扩展DEVS-Suite框架，实现有界并行DEVS模型的双重模型检查与仿真，支持非确定性和性能验证，为离散事件系统提供独特的开发与验证框架。 |
| [^32] | [W-RAG: Source-Aware Retrieval for Enterprise Document Generation from Heterogeneous Knowledge Bases](https://arxiv.org/abs/2608.22081) | W-RAG通过本体引导检索和每个知识库内的局部排序，解决了企业文档生成中异构知识库全局排序导致的不平衡上下文问题，从而生成更完整的草稿。 |
| [^33] | [LLM-Enhanced Commit Message Generation via Issue Information: An Exploratory Study](https://arxiv.org/abs/2608.22004) | 本文提出ISAC框架，通过结合代码差异和问题信息作为LLM输入，系统证明了在提交消息生成中纳入问题信息能持续提升性能，尤其对CIDEr指标改善最大。 |
| [^34] | [How Reliable Are NVD CWE Labels? A Large-Scale Semantic Audit with Seclometry](https://arxiv.org/abs/2608.21977) | 本文通过构建基于seclometry的CWEAgent审计工具，首次大规模测量了NVD中CWE标签的可靠性，发现其准确率有限，揭示了现有标签体系存在的系统性质量问题。 |
| [^35] | [Repo2Skill-Evo: Repository Skills Go Stale in Silence](https://arxiv.org/abs/2608.21964) | 本文提出Repo2Skill-Evo框架，研究智能体在软件仓库版本更新时维护和更新外部化技能的能力，揭示技能过时但无显式信号的问题。 |
| [^36] | [Ontology-supported Design Parameter Management for Change Impact Analysis](https://arxiv.org/abs/2608.21949) | 本文提出了一种基于本体的设计参数管理方法，通过需求追溯和专家知识库实现变更影响分析，适用于软硬件设计，并能提高异构系统间的知识传递效率。 |
| [^37] | [Ontology-based Requirements Transformation](https://arxiv.org/abs/2608.21945) | 本文提出一种基于本体的方法论和系统，利用OWL语言和模型支持，将任务剖面需求进行供应链感知转换，以提高工程效率、促进知识集成与重用。 |
| [^38] | [Loop Engineering: Building Blocks, Adoption, and Impact](https://arxiv.org/abs/2608.21884) | 本文首次探索性回顾了“循环工程”这一新兴实践，即开发者设计系统自动触发和停止智能体运行，并总结了其核心构建模块（如停止条件、状态文件和验证子智能体），但未量化其实际采用率。 |
| [^39] | [Architecture as Capability Equalizer for Coding Agents](https://arxiv.org/abs/2608.21747) | 本论文通过对照实验发现，架构规范格式对编码代理生成代码质量的影响依赖于模型能力，对较弱模型使用代码邻近格式可显著缩小与强模型的能力差距。 |
| [^40] | [XRFix: Exploring Performance Bug Repair of Extended Reality Applications with Large Language Models](https://arxiv.org/abs/2608.21718) | 本文提出了XRFix框架，利用大型语言模型自动检测并修复扩展现实应用中的性能缺陷，解决了缺乏真实数据集、检测工具和修复工具的三大挑战。 |
| [^41] | [ExploreAI: Agentic Exploration Knowledge Bases for Reproducible Observable-Regression Testing of Black-Box VR and 3D Applications](https://arxiv.org/abs/2608.21628) | ExploreAI利用大语言模型驱动智能体框架，通过语义指导的探索知识库，实现了黑盒VR和3D应用可复现且高效的回归测试。 |
| [^42] | [Large Language Models for Requirements Engineering: A Cross-Task Empirical Evaluation](https://arxiv.org/abs/2608.21531) | 本研究通过受控实验和工业案例研究，跨五项任务评估了大型语言模型在需求工程中的性能，填补了跨任务评估和复现性的空白。 |
| [^43] | [Neuro-Formal Verification: Agentic Language-Agnostic Formal Program Reasoning](https://arxiv.org/abs/2608.21516) | 本文提出神经形式验证（NFV），通过AI智能体将主流语言代码翻译为形式化验证语言，实现一键式经验性准确验证，无需形式化方法专业知识。 |
| [^44] | [BeTaL-GBI: Admission-Aware Benchmark Tuning and Full-Stack Verification of Geometric Belief Interfaces](https://arxiv.org/abs/2608.21503) | 本文提出 BeTaL-GBI，一种通过 LLM 参与的基准调优方法，在验证几何信念接口时分离格式准入与条件评估，以提升架构验证的可信度和可审计性。 |
| [^45] | [Composable Building Blocks for Resilient Asynchronous Code](https://arxiv.org/abs/2608.21489) | 本文提出了一种统一的、可组合的高阶组合子框架，通过嵌套表达式实现异步代码的弹性策略（如超时、重试、限流等），无需修改业务逻辑，并兼容JavaScript的Promise和异步迭代两种异步形态。 |
| [^46] | [SLICE: Specification-Level Isolation of Contract Enforcement](https://arxiv.org/abs/2608.21483) | SLICE是一种新型代码生成框架，通过分离输入契约和功能需求的生成阶段，确保生成的代码同时满足输入条件和计算要求，解决了契约执行中的完整性和严格性平衡问题。 |
| [^47] | [From Subjective Judgments to Auditable Standards:Protocol-Guided AI Auditing of Website Redundancy](https://arxiv.org/abs/2608.21476) | 本文提出CORA协议引导的网站冗余审计方法，通过分离测量重复负荷、正常使用税和恢复储备，实现可审计的AI审计流程，并证明其优于传统标量基线。 |
| [^48] | [Composable Trust Infrastructure for Manufacturing Knowledge Graphs: Cross-System Provenance, Temporal Reasoning, and Decision Traceability](https://arxiv.org/abs/2608.21418) | 本文提出一种可组合信任基础设施，通过共享标识符整合四种信任能力，产生单独能力无法提供的涌现信任属性，并实验验证了每种能力对组合查询的关键性。 |
| [^49] | [On the Time and Frequency Domain Representations of Signals for CPS Specification](https://arxiv.org/abs/2608.21167) | 本文提出了一种新的规约语言S2TL，利用时频表示来增强CPS需求规约的表达能力，特别适用于描述信号形状和动态行为。 |
| [^50] | [Grounding AI Agents in Contracts: An Empirical Evaluation of Spec-Driven Test Generation](https://arxiv.org/abs/2608.17177) | 提出规格驱动测试生成方法，通过让LLM代理先显式推理和记录代码合同（前置/后置条件及未定义行为）作为认知脚手架，显著提升了生产环境中的缺陷检测率和分支覆盖率。 |
| [^51] | [Who Will Become the Next Senior? How Generative AI Erodes the Development Pathway in Software Engineering](https://arxiv.org/abs/2607.17067) | 本研究通过访谈揭示GenAI通过“吸收”模式将初级工作转给高级-AI流程，导致初级工程师失去成长所需的挑战，并因集体正常化而固化这一损失。 |
| [^52] | [Falsification-Based Verification of LLM-Generated Optimization Models: Sound Test Batteries and Their Detection Limits](https://arxiv.org/abs/2607.16646) | 本文提出了一种基于证伪的验证方法，通过对偶性和灵敏度分析生成健全的求解器测试组，以零误报率检测LLM生成的优化模型中的错误，并明确了其检测极限。 |
| [^53] | [Exploring the Potential of Program Flowcharts on Code Generation Using Multimodal LLMs](https://arxiv.org/abs/2607.09146) | 本研究首次系统探索了将程序流程图作为视觉输入与问题描述结合，可显著提升多模态大语言模型（如GPT-4o）的代码生成性能。 |
| [^54] | [SABER: Benchmarking Operational Safety of LLM Coding Agents in Stateful Project Workspaces](https://arxiv.org/abs/2606.01317) | SABER提出了一个环境感知的操作安全性基准，通过状态化项目工作区中的最终状态评估编码代理的安全表现，并发现当前模型在现实环境中存在超过54%的高安全违规率。 |
| [^55] | [LLM-based Low-Level Integration Test Generation for Java](https://arxiv.org/abs/2605.26851) | IntTestGen是一种基于LLM的Java低层集成测试生成方法，通过挖掘依赖模式并应用多级约束修复，有效解决了LLM缺乏项目知识和违反约束的问题。 |
| [^56] | [Programming by Chat: A Large-Scale Behavioral Analysis of 11,579 Real-World AI-Assisted IDE Sessions](https://arxiv.org/abs/2604.00436) | 这项研究首次大规模分析了IDE原生环境中11,579个真实世界AI辅助编程会话，揭示了对话式编程作为渐进式规格说明的三个关键转变。 |
| [^57] | [CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537) | CRANE-LLM通过向大语言模型提供从笔记本内核提取的运行时信息，在单元格执行前实现崩溃预测与诊断，性能提升7-10个百分点。 |
| [^58] | [Vibe Coding on Trial: Operating Characteristics of Unanimous LLM Juries](https://arxiv.org/abs/2602.18492) | 本研究提出并评估了一种基于大语言模型陪审团的一致通过机制，用于自动审查文本到SQL任务中的候选代码，在安全优先场景下有效平衡了接受准确性与人工干预成本。 |
| [^59] | [Reading Between the Code Lines: On the Use of Self-Admitted Technical Debt for Security Analysis](https://arxiv.org/abs/2602.03470) | 本研究首次系统性地探讨了自我承认的技术债务（SATD）中编码的安全信息如何为静态分析工具（SATs）提供互补性安全洞察，从而增强漏洞检测的覆盖率和准确性。 |
| [^60] | [Dynamic Cogeneration of Bug Reproduction Test in Agentic Program Repair](https://arxiv.org/abs/2601.19066) | 本文提出并评估了在智能体程序修复中动态协同生成缺陷复现测试与修复补丁的策略，并开发了考虑测试变更的补丁选择器，以提升开发者对AI生成补丁的信心。 |
| [^61] | [An Empirical Study of Java Code Improvements Based on Stack Overflow Answer Edits](https://arxiv.org/abs/2511.05813) | 本研究通过分析Stack Overflow上Java答案的编辑历史，利用改进的代码克隆搜索工具，将这些编辑应用于开源项目中的代码改进，揭示了众包知识对提升代码质量的潜力。 |
| [^62] | [LLM-Driven Cost-Effective Requirements Change Impact Analysis](https://arxiv.org/abs/2511.00262) | 提出了一种基于大语言模型的成本效益型需求变更影响分析方法ProReFiCIA，在工业数据集上达到85.7%召回率，显著减少人工错误和成本。 |
| [^63] | [NaturalEdit: Code Modification through Direct Interaction with Adaptive Natural Language Representation](https://arxiv.org/abs/2510.04494) | NaturalEdit通过引入自适应、多面代码摘要和交互式映射机制，使代码修改过程更直观，降低了认知负担，并实现了自然语言表示与源代码之间的紧密连接。 |
| [^64] | [Entity Representation Learning Through Onsite-Offsite Graph for Pinterest Ads](https://arxiv.org/abs/2508.02609) | 本文提出了一种基于用户现场和场外活动的大规模异构图构建方法，并引入了带锚点的TransR模型（TransRA）来高效集成图嵌入，从而提升Pinterest广告排序模型的性能。 |
| [^65] | [LSem2Vec: A Simple yet Effective Two-Stage Approach for Source Code Embedding](https://arxiv.org/abs/2409.14644) | LSem2Vec通过两阶段方法（LLM提取语义+句子嵌入生成向量）实现无需监督训练或微调的源代码嵌入，有效处理错误信息并提升性能。 |

# 详细

[^1]: SWE重构基准：编码代理能否完成长时程、全仓库的栈迁移？

    SWE Refactor Bench: Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration?

    [https://arxiv.org/abs/2608.23564](https://arxiv.org/abs/2608.23564)

    本文提出了SWE重构基准，通过三阶段评估协议解决现有基准无法检测迁移是否真正发生的问题，从而衡量编码代理在长时程全仓库栈迁移中的能力。

    

    现代软件系统在数十年的开发过程中积累了技术债务，这使得迁移成本高昂且大部分依赖人工。随着编码代理在修复缺陷方面能力日益增强，它们能否自主执行此类迁移？现有基准无法回答这个问题，因为它们仅评估行为正确性，而不评估迁移是否真正发生。这导致了一种简单的作弊手段：代理复制原始实现以使测试通过。我们称之为“盲区”。为解决这一问题，我们引入了SWE重构基准，该基准包含20个全仓库迁移任务，涵盖4种类型的技术债务。一个三阶段评估协议同时衡量迁移完整性和行为正确性：（1）迁移审计验证迁移是否发生。（2）行为测试使用固定测试套件衡量正确性。（3）代理验证使用6个独立的编码代理生成针对性测试。

    arXiv:2608.23564v1 Announce Type: cross  Abstract: Modern software systems accumulate technical debt over decades of development, which makes migration expensive and largely manual. As coding agents become increasingly capable at bug fixing, can they autonomously perform such migrations? Existing benchmarks cannot answer this question because they evaluate only behavioural correctness, not whether the migration actually occurred. This leads an easy hack: agents copy the original implementation to make tests pass. We call this Blindness. To address this problem, we introduce SWE Refactor Bench, a benchmark comprising 20 whole-repository migrations, covering 4 kinds of technical debt. A three-stage evaluation protocol measures both migration completeness and behavioural correctness. (1) Migration Audit verifies that the migration occurred. (2) Behavioural Tests measure correctness with a fixed test suite. (3) Agentic Verification uses 6 independent coding agents to generate targeted test
    
[^2]: Prime Agent：一个自我改进的递归语言模型工具框架

    Prime Agent: A Self-Improving RLM Harness

    [https://arxiv.org/abs/2608.23552](https://arxiv.org/abs/2608.23552)

    Prime Agent是一个开源工具框架，通过持久化REPL和递归子代理机制，将长期评估和编码代理工作流标准化，从而防止工具故障干扰模型，最大化模型潜力。

    

    arXiv:2608.23552v1 公告类型：新 摘要：语言模型是顺序处理器，但长期代理任务需要超越模型权重和活动上下文的外部信息与计算。Prime Agent是一个用于长期评估和编码代理工作流的开源工具框架。一个持久化的IPython REPL遵循递归语言模型抽象，实现程序化上下文处理和测试时计算，而持续工具框架跨轨迹保留历史、记忆、技能、提示和子代理规范。递归子代理通过直接的代理间通信进行协调，代理视图允许人类检查和管控守护进程支持的会话。Prime Agent标准化了执行、恢复、验证和资源核算，同时将策略构建留给模型。这种低摩擦、表达力强的膜防止工具框架故障变成模型故障，并将测量推向模型真正的最大潜在上限。

    arXiv:2608.23552v1 Announce Type: new  Abstract: Language models are sequential processors, but long-horizon agency requires external information and computation beyond model weights and active context. Prime Agent is an open-source harness for long-horizon evaluation and coding-agent workflows. A persistent IPython REPL follows the Recursive Language Model abstraction for programmatic context processing and test-time compute, while Continual Harness preserves histories, memories, skills, prompts, and subagent specifications across trajectories. Recursive subagents coordinate through direct agent-to-agent communication, and the Agents View lets humans inspect and manage daemon-backed sessions. Prime Agent standardizes execution, recovery, verification, and resource accounting while leaving strategy construction to the model. This low-friction, expressive membrane prevents harness failures from becoming model failures and pushes measurement toward the model's true maximal underlying cap
    
[^3]: 一个用于需求驱动型候选人搜寻的交互式代理

    An Interactive Agent for Requirement-Driven Candidate Sourcing

    [https://arxiv.org/abs/2608.23501](https://arxiv.org/abs/2608.23501)

    该论文提出了一种交互式需求驱动的候选人搜寻代理和配套基准，将模糊的人员搜索视为需求工程任务，通过引导、验证和确认流程来生成合理的候选名单。

    

    arXiv:2608.23501v1 公告类型：新 摘要：从自然语言描述（如“从机器学习转向生物技术研究角色的工程师”）中寻找人员，越来越多地委托给LLM代理，并被框架化为信息检索。我们认为这本质上是一项需求工程任务：这样的请求是一个欠约束的需求，带有隐含约束、许多有效答案，且没有接受标准，因此有用的答案需要在搜索之前进行需求引导、验证和确认。我们提出了\sys{}，据我们所知，这是第一个交互式、需求驱动的候选人搜寻代理（它通过有限引导、工作流模板、两阶段提交协议和双向终止保护，将模糊的人员请求引导、验证、检索并确认成一个合理的候选名单），以及\bench{}，一个运行需求生命周期（基于标准锚定的验证、多模型证据基础的真相构建）的基准测试。

    arXiv:2608.23501v1 Announce Type: new  Abstract: Finding people from a natural-language description (``ML engineers transitioning to research roles in biotech'') is increasingly delegated to LLM agents and framed as information retrieval. We argue that it is fundamentally a requirements engineering task: such a request is an under-determined requirement with implicit constraints, many valid answers, and no acceptance criterion, so useful answers require eliciting, validating, and verifying the requirement before search can matter. We present \sys{}, to our knowledge the first interactive, requirements-driven candidate-sourcing agent (it elicits, validates, retrieves, and verifies a vague people-request into a justified slate through bounded elicitation, workflow templates, a two-stage commit protocol, and bidirectional termination guards) and \bench{}, a benchmark that runs the requirements lifecycle (criteria-anchored validation, multi-model evidence-grounded oracle construction, and 
    
[^4]: 增值税确定中LLM智能体分解的规模优化：一项受控扫描试点研究

    Right-Sizing LLM-Agent Decomposition in VAT Determination: A Pilot Controlled Sweep

    [https://arxiv.org/abs/2608.23395](https://arxiv.org/abs/2608.23395)

    该论文通过受控扫描实验发现，在增值税确定任务中，中等数量的智能体分解配置（而非极端窄或宽）表现最优，平衡了提示预算与智能体数量效应。

    

    arXiv:2608.23395v1 公告类型：交叉 摘要：近期LLM智能体系统在设计上存在冲突：要么将工作分解到多个狭窄智能体，要么使用一个强大的工具使用智能体。本试点研究在具有反向收费的跨境增值税确定场景下探讨了这一选择，其中每个案例都有标准答案标签，每个中间决策都可独立评分。我们保持活动表面不变（子任务、工具、输入/输出模式、验证检查、协调器、基础模型和合并策略），仅在不同编排配置中改变子任务分配给工作者的方式，从单一宽泛工作者到五个狭窄工作者，并与S0（一个经过调优的无协调器单一智能体）对比，使用确定性规则引擎作为标准答案。该程序涵盖4，400次运行：一个40案例、五次重复的主扫描、匹配令牌臂（区分提示预算与智能体数量效应）以及三个故障注入臂，所有均根据预注册的证伪标准进行评判。两个中间配置表现领先。

    arXiv:2608.23395v1 Announce Type: cross  Abstract: Recent LLM-agent systems make conflicting design bets: decompose work across many narrow agents, or use one strong tool-using agent. This pilot studies that choice on bounded cross-border VAT determination with reverse charge, where every case has an oracle label and each intermediate decision is independently scoreable. We hold the activity surface fixed (subtasks, tools, I/O schemas, validation checks, orchestrator, base model, and merge policy) and vary only the assignment of subtasks to workers across four orchestrated configurations, from one wide worker to five narrow ones, against S0, a tuned no-orchestrator single agent, with a deterministic rule engine as oracle. The program spans 4,400 runs: a 40-case, five-repeat main sweep, matched-token arms separating prompt-budget from agent-count effects, and three failure-injection arms, all judged against pre-registered falsification criteria. The two intermediate configurations lead 
    
[^5]: 形式化与自动化跨方法的细粒度移动重构

    Formalizing and Automating Fine-Grained Move Refactorings Across Methods

    [https://arxiv.org/abs/2608.23377](https://arxiv.org/abs/2608.23377)

    该论文形式化了五种跨方法的语句级移动重构变体，并通过迭代细化提出了额外前置条件，实现了细粒度表达式移动的自动化，在十个项目上验证了其适用性和可编译性。

    

    arXiv:2608.23377v1 公告类型：新 摘要：开发者使用自动化的移动重构来改善源代码的模块化结构和职责分配。现代集成开发环境（IDE）中，类级和方法级的移动重构已实现自动化，但调整方法边界的语句级和表达式级移动在很大程度上仍未自动化。我们将移动语句重构的五个变体形式化为前置条件和步骤，这些步骤基于四个基本条件，涵盖数据可达性、执行次数、副作用以及编译所需的语法约束，其中除副作用条件外，其余均可静态检查。结合现有技术，这还可实现表达式和部分表达式的更细粒度移动。我们进一步针对一个真实项目迭代细化形式化过程，推导出二十个额外的前置条件和步骤，以处理实践中Java语法的多样性。我们在十个项目上评估了适用性和可编译性，并be

    arXiv:2608.23377v1 Announce Type: new  Abstract: Developers use automated Move refactorings to improve the modular structure of source code and the assignment of responsibilities. Class- and method-level Move refactorings are automated in modern IDEs, but statement- and expression-level moves that adjust method boundaries remain largely unautomated. We formalize five variants of Move Statement refactoring as preconditions and steps grounded in four basic conditions covering data reachability, execution count, side effects, and syntactic constraints required for compilation, of which all but the side-effect condition are checked statically. Combined with existing techniques, this also yields finer-grained moves of expressions and partial expressions. We further refine the formalization iteratively against a real project, deriving twenty additional preconditions and steps that handle Java syntactic diversity in practice. We evaluate applicability and compilability on ten projects, and be
    
[^6]: DPIAgent：分而治之、协议交接、隔离操作——面向智能体复现测试生成

    DPIAgent: Divide, Protocol, Isolate for Agentic Reproduction Test Generation

    [https://arxiv.org/abs/2608.23341](https://arxiv.org/abs/2608.23341)

    本文提出DPIAgent框架，通过将复现测试生成任务分解为缺陷探索和测试生成两个单目标阶段，并采用协议交接和工具隔离，有效解决了智能体在复合目标下的目标漂移问题。

    

    复现测试生成，即产生一个先失败后通过的测试以捕获所报告的错误，是自动化软件工程中的关键步骤。现有的智能体方法将此视为一个整体循环，尽管该任务本质上包含两个性质不同的子任务：诊断根本原因和编写失败到通过的测试。如果没有明确分离，智能体将面临一个复合目标，且中间目标不明确，导致目标漂移。我们提出DPIAgent，一种基于三个原则——分而治之、协议交接、隔离操作（DPI）的结构化智能体框架，以缓解复合目标模糊性和目标漂移：它将任务划分为缺陷探索和测试生成的单一目标阶段；强制执行一个交接协议，记录诊断和测试计划，防止上下文丢失；并通过为每个阶段定制工具集来隔离其操作空间，防止无关工具误导智能体。

    arXiv:2608.23341v1 Announce Type: new  Abstract: Reproduction test generation, producing a failing-then-passing test that captures a reported bug, is a critical step in automated software engineering. Existing agentic methods treat this as a monolithic loop, despite the task inherently comprising two subtasks of distinct nature: diagnosing the root cause and writing a fail-to-pass test. Without explicit separation, the agent faces a compound objective with underspecified intermediate goals, leading to goal drift. We propose DPIAgent, a structured agentic framework built on three principles, Divide, Protocol, Isolate (DPI), that mitigates compound-objective ambiguity and goal drift: it Divides the task into single-objective phases of defect exploration and test generation; enforces a handoff Protocol that records the diagnosis and test plan, preventing context loss; and Isolates each phase's action space by tailoring the toolset to its task, preventing irrelevant tools from misleading e
    
[^7]: 从自然语言政策到可执行义务：为可靠的车载大语言模型智能体设计的验证框架

    From Natural Language Policies to Executable Obligations: A Verification Harness for Dependable In-Car LLM Agents

    [https://arxiv.org/abs/2608.23282](https://arxiv.org/abs/2608.23282)

    本文提出AgentGuardUtil，通过运行时政策编译器将自然语言政策转换为可执行义务，并用确定性门控和验证循环确保车载LLM智能体可靠遵守操作规范。

    

    摘要：部署在车辆中的大语言模型（LLM）智能体必须在每一轮交互中满足书面操作政策：一个幻觉标识符、遗漏的强制副作用或过早的完成声明都可能导致任务失败。我们提出了AgentGuardUtil，这是我们对CAR-bench Track~1的参赛方案，它将AI规划器（LLM）视为一个在基于实证的验证与修订循环中易出错提议者。其核心新颖之处在于一个运行时政策编译器：随每次对话提供的自然语言政策，每次按政策编译成类型化的机器可检查规则，其中一部分规则获得可执行形式。一个确定性义务引擎根据实时工具结果和草稿本身的模拟写入后状态来解释这些规则，发出带有计算参数的确切补救调用，而不是自然语言提醒。围绕该引擎，25个确定性门控（标识符来源、模式与枚举有效性、先收集后行动、混淆检测等）提供额外保障。

    arXiv:2608.23282v1 Announce Type: new  Abstract: Large Language Models (LLMs) agents deployed in vehicles must satisfy a written operating policy on every turn: a single hallucinated identifier, omitted mandatory side-effect, or premature completion claim fails the task. We present AgentGuardUtil, our entry to CAR-bench Track~1, which treats the AI planer (LLM) as a fallible proposer inside a grounded verify-and-revise loop. Its core novelty is a runtime policy compiler: the natural-language policy shipped with each conversation is compiled, once per policy, into typed machine-checkable rules, a subset of which receive an executable form. A deterministic obligation engine interprets these rules against live tool results and the simulated post-write state of the draft itself, emitting the exact remedial calls with computed arguments rather than natural-language reminders. Around this engine, 25 deterministic gates (identifier provenance, schema and enum validity, gather-before-act, conf
    
[^8]: TianoCore社区实证研究

    An Empirical Study of the TianoCore Community

    [https://arxiv.org/abs/2608.23280](https://arxiv.org/abs/2608.23280)

    本研究通过调查和访谈揭示了TianoCore社区固件开发中的安全与维护差距，并提出通过改进安全实践、采用内存安全技术和自动化来强化UEFI固件。

    

    arXiv:2608.23280v1 公告类型：新 摘要：我们调查了TianoCore社区中各利益相关者所采用的软件安全和维护实践，并识别出改进固件开发工作流程的机会。我们开展了一项调查和有限的访谈研究，参与者包括独立固件供应商、原始设备制造商、安全专家、固件开发者和学术研究人员。该开源开发社区维护着UEFI固件核心的参考实现。我们强调了TianoCore生态系统内当前固件开发状态的重要差距，并识别出关键领域，在这些领域中，改进的安全实践、更广泛采用内存安全技术以及增加手动流程的自动化可以加强UEFI固件的维护和安全性。

    arXiv:2608.23280v1 Announce Type: new  Abstract: We investigate the software security and maintenance practices adopted by stakeholders in the TianoCore community and identify opportunities to improve firmware development workflows. We conduct a survey and a limited interview study with participants representing independent firmware vendors, original equipment manufacturers, security experts, firmware developers, and academic researchers. This open-source development community maintains a reference implementation for the core of the UEFI firmware. We highlight important gaps in the current state of firmware development within the TianoCore ecosystem and identify key areas in which improved security practices, greater adoption of memory-safe technologies, and increased automation of manual processes could strengthen the maintenance and security of the UEFI firmware.
    
[^9]: TianoForge：面向TianoCore UEFI固件开发社区的自动化缺陷分类方法

    TianoForge: An Automated Bug Triage Approach for the TianoCore UEFI Firmware Development Community

    [https://arxiv.org/abs/2608.23259](https://arxiv.org/abs/2608.23259)

    TianoForge通过结合GPT大型语言模型和检索增强生成技术，自动化了UEFI固件开发中的缺陷分类流程，显著提高了缺陷处理效率。

    

    arXiv:2608.23259v1 公告类型：新公告 摘要：我们提出了一种在TianoCore开源UEFI固件开发生态系统中进行缺陷分类的新方法。这种集成方法称为TianoForge，它部署了人工智能的最新技术，特别是机器学习，以实现自动化缺陷分类。这包括无效缺陷报告检测、重复缺陷报告检测、缺陷报告优先级排序和缺陷报告分配。我们使用各种生成式预训练变换器（GPT）大型语言模型（LLM），并结合或不结合检索增强生成（RAG）来自动化这些任务。鉴于缺陷分类在软件维护中的关键作用，以及TianoCore社区（尤其是其主要项目EDK II）中大量未分类问题，我们预计这将对TianoCore软件维护流程（主要是缺陷分类和解决）的效率产生显著影响。我们的实验研究表明，TianoForge将平均缺陷分类时间减少了[未指定数值]。

    arXiv:2608.23259v1 Announce Type: new  Abstract: We propose a novel approach to bug triage in the TianoCore open-source UEFI firmware development ecosystem. This integrated approach, called TianoForge, deploys the state of the art in artificial intelligence, specifically machine learning, to enable automated bug triage. This includes invalid bug report detection, duplicate bug report detection, bug report prioritization, and bug report assignment. We use various Generative Pretrained Transformer (GPT) Large Language Models (LLMs) with and without Retrieval Augmented Generation (RAG) to automate these tasks. Given the crucial role of bug triage in software maintenance and the huge number of untriaged issues in the TianoCore community, in particular, their primary project, EDK II, we expect a significant impact on the efficiency of TianoCore software maintenance processes, primarily bug triage and resolution. Our experimental study shows that TianoForge reduces the average bug triage tim
    
[^10]: LLMCrater：使用大型语言模型的生命周期感知FAIR元数据生成

    LLMCrater: Lifecycle-Aware FAIR Metadata Generation using Large Language Models

    [https://arxiv.org/abs/2608.23158](https://arxiv.org/abs/2608.23158)

    该论文提出LLMCrater框架，利用大型语言模型和阶段特定的RO-Crate配置文件，在研究生命周期的四个阶段自动生成并逐步丰富FAIR元数据，以解决现有方法仅在发表时生成元数据而错过上下文信息的问题。

    

    arXiv:2608.23158v1 公告类型：新 摘要：FAIR（可查找、可访问、可互操作和可重用）元数据对于科研资产的发现、互操作性和重用至关重要。然而，创建和维护FAIR元数据在很大程度上仍依赖人工操作，这使得针对研究生命周期中产生的异构研究工件的过程耗时费力。现有方法主要在发表时生成元数据，错过了在信息可用时捕获上下文信息的机会。为解决这一局限性，我们提出了\emph{LLMCrater}，一种生命周期感知的元数据生成框架，它结合了大型语言模型（LLMs）与阶段特定的RO-Crate元数据配置文件。该框架在研究生命周期的四个阶段（设计、开发、部署、执行与溯源）中逐步丰富元数据，同时保持与RO-Crate 1.1和EOSC元数据建议的兼容性。它自动提取元数据

    arXiv:2608.23158v1 Announce Type: new  Abstract: FAIR (Findable, Accessible, Interoperable, and Reusable) metadata is essential for the discovery, interoperability, and reuse of scientific research assets. However, creating and maintaining FAIR metadata remains largely manual, making the process time-consuming for heterogeneous research artifacts generated throughout the research lifecycle. Existing approaches primarily generate metadata at publication time, missing opportunities to capture contextual information as it becomes available. To address this limitation, we present \emph{LLMCrater}, a lifecycle-aware metadata generation framework that combines Large Language Models (LLMs) with stage-specific RO-Crate metadata profiles. The framework progressively enriches metadata across four research lifecycle stages (Design, Development, Deployment, and Execution \& Provenance) while remaining compatible with RO-Crate~1.1 and EOSC metadata recommendations. It automatically extracts metadat
    
[^11]: 一种将遗留科学应用转化为可复用云工作流的AI辅助迁移框架

    An AI-Assisted Migration Framework for Transforming Legacy Scientific Applications into Reusable Cloud-Based Workflows

    [https://arxiv.org/abs/2608.23146](https://arxiv.org/abs/2608.23146)

    该论文提出一种结合RM-ODP、LLM和DSM的AI辅助框架，能系统地将遗留科学应用迁移为可复用的云原生工作流，解决了现有方法仅针对单个工件且缺乏系统性的问题。

    

    遗留科学应用仍然是宝贵的研究资产，但通常与特定项目的执行环境紧密耦合，限制了它们在现代科学工作流系统和云原生虚拟研究环境（VREs）中的复用、可重现性和部署。现有迁移方法主要针对单个工件（如笔记本或容器），在系统性地将异构遗留应用转化为可复用云原生工作流方面支持有限。本文提出了一种AI辅助迁移框架，结合了开放分布式处理参考模型（RM-ODP）引导的架构分析、大型语言模型（LLMs）和设计结构矩阵（DSM）分析。该框架首先使用RM-ODP引导LLM从异构遗留应用中识别可复用的工作流组件、其接口和执行依赖。由此产生的工作流（此处未完整显示，但原文后续内容涉及工作流生成和验证）被用于构建可复用的云工作流。

    arXiv:2608.23146v1 Announce Type: new  Abstract: Legacy scientific applications remain valuable research assets but are often tightly coupled to project-specific execution environments, limiting their reuse, reproducibility, and deployment within modern scientific workflow systems and cloud-native Virtual Research Environments (VREs). Existing migration approaches primarily target individual artifacts, such as notebooks or containers, and provide limited support for systematically transforming heterogeneous legacy applications into reusable cloud-native workflows. This paper presents an AI-assisted migration framework that combines the Reference Model of Open Distributed Processing (RM-ODP)-guided architectural analysis, Large Language Models (LLMs), and Design Structure Matrix (DSM) analysis. The framework first uses RM-ODP to guide an LLM in identifying reusable workflow components, their interfaces, and execution dependencies from heterogeneous legacy applications. The resulting wor
    
[^12]: 从度量到改进：一种生命周期感知的LLM反馈框架，用于研究软件质量提升

    From Metrics to Improvement: A Lifecycle-Aware LLM Feedback Framework for Research Software Quality

    [https://arxiv.org/abs/2608.23118](https://arxiv.org/abs/2608.23118)

    该论文提出了一种生命周期感知框架，结合定量质量评估与LLM驱动的代码精炼，将研究软件质量指标转化为可操作的改进建议。

    

    研究软件在科学工作流中日益重要，但通常由软件工程专业知识有限的研究人员开发。这可能导致可维护性、可复现性、重用性和可持续性方面的质量问题。现有的静态分析工具能够识别这些问题，但其输出往往需要专家解读，且在将质量评估转化为可操作的改进方面支持有限。为弥补这一不足，我们提出了一种生命周期感知框架，将定量软件质量评估与基于大语言模型（LLM）的代码精炼相结合。该框架包含两个阶段：首先，根据既定软件质量标准与从业者需求构建生命周期感知的质量模型，该模型定义了五个质量维度和25个候选指标，其中14个通过现有分析工具和自定义测量实现操作化。

    arXiv:2608.23118v1 Announce Type: new  Abstract: Research software is increasingly central to scientific workflows, yet it is often developed by researchers with limited software engineering expertise. This can lead to quality issues that hinder maintainability, reproducibility, reuse, and sustainability. Existing static analysis tools can identify such issues, but their outputs often require expert interpretation and provide limited support for translating quality assessments into actionable improvements. To address this gap, we propose a lifecycle-aware framework that integrates quantitative software quality assessment with Large Language Model (LLM)-based code refinement. The framework comprises two stages. First, a lifecycle-aware Quality Model is developed from established software quality standards and practitioner requirements. The model defines five quality dimensions and 25 candidate metrics, of which 14 are operationalized using existing analysis tools and custom measurements
    
[^13]: 面向数字孪生集成与复用的多视角建模框架及基于大语言模型的兼容性分析

    A Multi-Viewpoint Modeling Framework for Digital Twin Integration and Reuse with LLM-Assisted Compatibility Analysis

    [https://arxiv.org/abs/2608.23115](https://arxiv.org/abs/2608.23115)

    本文提出一个基于RM-ODP的多视角建模框架，结合大语言模型辅助的兼容性分析，以支持数字孪生中异构模型的系统化集成与复用，并实现早期可行性评估。

    

    数字孪生（DT）生态系统整合异构计算模型，以在动态、目标特定的需求下表示复杂系统。系统性复用现有高质量模型和数据集对于可扩展的DT开发至关重要，但受到语义意图、数据结构、行为接口和执行环境异构性的制约。因此，集成成为一个跨模型、跨视角的一致性问题，难以在设计选择之间进行预测、量化和比较。现有标准和集成平台分别处理这些问题，对结构化、目标感知的兼容性评估以及在新DT目标下复用模型时的早期可行性分析支持有限。本文引入一个基于开放分布式处理参考模型（RM-ODP）的多视角集成建模框架。该框架将集成相关的结构组织起来。

    arXiv:2608.23115v1 Announce Type: new  Abstract: Digital Twin (DT) ecosystems integrate heterogeneous computational models to represent complex systems under evolving, purpose-specific objectives. Systematic reuse of existing high-quality models and datasets is essential for scalable DT development, yet is constrained by heterogeneity in semantic intent, data structures, behavioral interfaces, and execution environments. As a result, integration becomes a cross-model, cross-view consistency problem that is hard to predict, quantify, and compare across design choices. Existing standards and integration platforms address these concerns separately, offering limited support for structured, purpose-aware compatibility assessment and early feasibility analysis when models are reused under new DT objectives. This paper introduces a multi-viewpoint integration modeling framework grounded in the Reference Model of Open Distributed Processing (RM-ODP). The framework structures integration-releva
    
[^14]: ARGUS：基于MCP的Kubernetes事件根因分析系统

    ARGUS: MCP-Grounded Root Cause Analysis for Kubernetes Incidents

    [https://arxiv.org/abs/2608.23084](https://arxiv.org/abs/2608.23084)

    ARGUS通过标准化MCP服务器连接商业LLM到实时Kubernetes可观测数据，在Slack中提供结构化诊断摘要，实现了跨组织可复用的自动化根因分析。

    

    arXiv:2608.23084v1 公告类型：新公告  摘要：Kubernetes事件分类需要关联来自指标、日志、容器状态和消息系统等多个监控工具的信号，这种碎片化的工作流程会减慢诊断速度并导致告警疲劳。大型语言模型（LLMs）在自动化根因分析（RCA）方面显示出潜力，但现有系统依赖于自定义的、特定于系统的数据访问层，无法跨组织复用。我们提出了ARGUS，一个基于MCP的RCA助手，它通过标准化的MCP服务器（覆盖Kubernetes状态、Prometheus指标、Loki日志和NATS消息）将商业LLM连接到实时Kubernetes可观测性数据，并在值班工程师已使用的Slack事件频道中提供结构化的诊断摘要。我们使用三种互补方法对ARGUS进行了初步评估：在十个Kubernetes事件场景中进行受控故障注入、基于规则的评分以及...（摘要截断）

    arXiv:2608.23084v1 Announce Type: new  Abstract: Kubernetes incident triage requires correlating signals from metrics, logs, container state, and messaging systems across multiple monitoring tools, a fragmented workflow that slows diagnosis and contributes to alert fatigue. Large language models (LLMs) have shown promise for automated root cause analysis (RCA), but existing systems rely on custom, system-specific data access layers that cannot be reused across organisations. We present ARGUS, an MCP-grounded RCA assistant that connects a commercial LLM to live Kubernetes observability data through standardised MCP servers covering Kubernetes state, Prometheus metrics, Loki logs, and NATS messaging, and delivers structured diagnostic summaries inside the Slack incident channel where on-call engineers already work. We conduct a preliminary evaluation of ARGUS using three complementary methods: controlled fault injection across ten Kubernetes incident scenarios, rubric-based scoring of th
    
[^15]: AutoSaddler：基于代理执行轨迹的持久更新自动框架优化

    AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces

    [https://arxiv.org/abs/2608.23041](https://arxiv.org/abs/2608.23041)

    AutoSaddler通过将框架优化视为离线学习问题，利用失败轨迹诊断和代码式补丁生成，自动迭代改进代理框架，在多个基准上显著提升性能。

    

    arXiv:2608.23041v1 公告类型：新  摘要：大型语言模型代理在长期任务中仍不可靠，微小的局部失败可能在长时间交互中累积并导致整体任务失败。尽管外部框架能显著提升鲁棒性，但框架设计仍是一个手动且昂贵的过程，需要在大量提示、工具配置和控制逻辑的搜索空间中进行探索。我们提出AutoSaddler，一种自动框架优化框架，将框架改进形式化为离线学习问题，并利用小批量中的失败信号迭代更新框架。AutoSaddler结合了失败轨迹诊断、将框架视为代码的结构化补丁生成，以及基于验证的更新选择。在GAIA2、SWE-Bench Pro和Terminal-Bench 2.0上的实验表明，AutoSaddler显著提升了代理在对应基础框架上的性能，分别实现了9.0、9.6和10.0个百分点的增益。

    arXiv:2608.23041v1 Announce Type: new  Abstract: LLM agents remain unreliable on long-horizon tasks, where small local failures can compound over extended interactions and lead to overall task failure. Although external harnesses can substantially improve robustness, harness design remains a manual and expensive process that requires searching over a large space of prompts, tool configurations, and control logic. We propose AutoSaddler, an automatic harness optimization framework that formulates harness improvement as an offline learning problem and iteratively updates the harness using failure signals from mini-batches. AutoSaddler combines failure-trace diagnosis, structured patch generation that treats the harness as code, and validation-based update selection. Experiments on GAIA2, SWE-Bench Pro, and Terminal-Bench 2.0 show that AutoSaddler substantially improves agent performance over the corresponding base harnesses, achieving gains of 9.0, 9.6, and 10.0 percentage points, respec
    
[^16]: PatchWrite：一行而非整节——面向AI草稿的编译门控、有效性保持编辑

    PatchWrite: One Line, Not One Section -- Compile-Gated, Validity-Preserving Editing for AI-Drafted Manuscripts

    [https://arxiv.org/abs/2608.23001](https://arxiv.org/abs/2608.23001)

    PatchWrite通过编译门控和证据锁机制，仅允许局部编辑而非整节重写，从而在修复手稿缺陷时严格保留无关内容，显著提升编辑的有效性和安全性。

    

    自动手稿流水线通常为了修复局部缺陷而重新生成整个章节，这会导致无关的指标和引用发生变化，即使最终生成的PDF仍能正常构建。PatchWrite则限制了候选编辑如何成为已提交的手稿状态：它重用有界的EDIT N M编辑和回滚，但通过致命日志检查加强了编译接受条件，并添加了证据锁，要求每个引用的键和实验数值令牌都必须由参考注册表或实验日志验证。未通过任一检查的候选将被拒绝，并保留之前的HEAD状态。在24篇手稿×8种故障的Oracle压力测试（768个任务，平均分为编译破坏和仅内容故障）中，整槽重写每次都会改变无关的“12层”行（0/192保留；数值Jaccard指数为0.6667），而PatchWrite在192/192的情况下保留了该行。移除编译门控后接受率降至0，而重新...

    arXiv:2608.23001v1 Announce Type: new  Abstract: Automated manuscript pipelines often regenerate an entire section to repair a local defect, allowing unrelated metrics and citations to change even when the resulting PDF still builds. PatchWrite instead constrains how candidate edits become committed manuscript states: it reuses bounded EDIT N M editing and rollback, but tightens compilation acceptance with fatal-log checks and adds evidence locks that require every cited key and experimental numeric token to be attested by a reference registry or experimental log. Candidates that fail either check are rejected and the previous HEAD is retained. On a 24-manuscript x 8-fault oracle stress test (768 jobs, evenly split between compile-breaking and content-only faults), whole-slot rewriting mutated an unrelated "12-layer" line in every case (0/192 preserved; numeric Jaccard 0.6667), whereas PatchWrite preserved it in 192/192 cases. Removing the compile gate reduced acceptance to 0, while re
    
[^17]: 社会技术过程中的公平性危害分析：偏见敏感组织环境中的多案例研究

    Fairness Hazard Analysis for Socio-Technical Processes: A Multiple-Case Study in Bias-sensitive Organisational Settings

    [https://arxiv.org/abs/2608.22978](https://arxiv.org/abs/2608.22978)

    本文提出并验证了公平性危害分析方法论，该方法能在需求工程阶段系统识别和缓解社会技术过程中的公平性风险，以支持主动的公平性设计。

    

    摘要：公平性日益被视为社会技术过程中的一类首要需求，在这些过程中，人类参与者、软件系统和人工智能技术之间的交互可能导致决策工作流中的不公平结果。如果不加以解决，公平性危害可能会积累并强化系统性偏见，这凸显了主动设计公平性的必要性。尽管对公平性感知系统的兴趣日益增长，但系统性地识别社会技术过程中的公平性危害并推导出需求级缓解措施的方法仍然有限。为了在需求工程（RE）期间支持公平性设计，公平性危害分析（FHA）被引入作为一种系统识别、分析和缓解公平性危害的方法论。FHA首先通过两个焦点小组进行概念验证评估，然后，一项涉及两个组织的定性多案例研究检验了其适用性。

    arXiv:2608.22978v1 Announce Type: new  Abstract: Fairness is increasingly recognised as a first-class requirement in socio-technical processes, where interactions among human actors, software systems, and AI technologies may lead to unfair outcomes in decision-making workflows. If left unaddressed, fairness hazards may accumulate and reinforce systemic bias, highlighting the need to engineer fairness proactively. Despite growing interest in fairness-aware systems, systematic methods for identifying fairness hazards in socio-technical processes and deriving requirements-level mitigations remain limited. To support fairness-by-design during requirements engineering (RE), Fairness Hazard Analysis (FHA) is introduced as a methodology for systematically identifying, analysing, and mitigating fairness hazards. FHA is first assessed through a proof-of-concept validation conducted via two focus groups. Then, a qualitative multiple-case study involving two organisations examines its applicabili
    
[^18]: 基于执行锚定的幻觉校准重排序用于Verilog代码生成

    Execution-Anchored Hallucination Calibration Reranking for Verilog Code Generation

    [https://arxiv.org/abs/2608.22938](https://arxiv.org/abs/2608.22938)

    本文提出了一种基于执行锚定的幻觉校准重排序方法，结合执行信号和推理信号，有效解决了Verilog代码生成中候选选择时的测试平台质量低和推理幻觉问题。

    

    大型语言模型（LLMs）在代码生成方面展示了显著的能力，但在处理Verilog等低资源硬件描述语言时，其性能显著下降。虽然多候选采样提高了生成正确解决方案的可能性，但自动选择最优候选仍然是一个开放的挑战。通过对九个模型和两个基准的系统性实证研究，我们识别出两个关键限制：（1）现有的基于执行的重排序方法，依赖于测试平台的通过/失败结果，由于生成的测试平台质量较低，表现出较差的领域迁移性；（2）LLM作为评判者存在推理幻觉，对执行等效的代码产生不一致的判断。这些发现揭示了两种具有正交错误的信号类型：执行信号（确定性但测试平台覆盖有限）和推理信号（语义丰富但存在幻觉）。

    arXiv:2608.22938v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet their performance degrades significantly on low-resource Hardware Description Languages such as Verilog. While multi-candidate sampling improves the likelihood of generating correct solutions, au-tomatically selecting the optimal candidate remains an open challenge. Through a systematic empirical study across nine models and two benchmarks, we identify two critical limitations:(1) existing execution-based reranking methods, which rely on testbench pass/fail outcomes, exhibit poor domain transferability due to low-quality generated testbenches; and (2) LLM-as-a-Judge suffers from reasoning hallucination, producing incon-sistent judgments for execution-equivalent code. These findings reveal two signal types with orthogonal errors: execution signals(deterministic but testbench coverage limited)and reasoning signals (semantically rich but hallucina
    
[^19]: 保护代理式AI编码及Terok环境的概念

    Concepts for Securing Agentic AI Coding and the Terok Environment

    [https://arxiv.org/abs/2608.22930](https://arxiv.org/abs/2608.22930)

    本文评估了代理式AI编码的IT安全风险，提出了在不损害其优势的前提下缓解风险的概念，并概述了Terok环境中的实现方案。

    

    代理式AI是软件开发中一种令人着迷的新工具，相较于“传统”AI辅助编码，它迈出了一大步，而后者在更早前也取得了显著突破。通过LLM的AI支持是一个年轻且快速发展的领域。“传统”（非代理式）形式在2025年初（约18个月前）变得实用且高效，代理式形式则在2025年秋季（约9个月前）随之而来。除了其诸多好处和潜力外，它也给IT安全带来了一些根本性风险。代理式方法在加剧其他风险的同时，引入了非常严重的风险。尽管有探索这一迷人新工具的动机，我们不应忽视这些风险，而应积极应对。我们提出了（I）对IT安全风险的评估，（II）在不损害其益处的前提下缓解这些风险的概念，以及（III）我们概念实现情况的概述。在这个高度动态的领域中。

    arXiv:2608.22930v1 Announce Type: new  Abstract: Agentic AI is a fascinating new tool for software development. It is a huge step forward compared to "conventional" AI assisted coding, which in turn was a considerable breakthrough earlier. AI support through LLMs is a young and very fast-moving field. The "conventional" (non-agentic) flavor became useful and productive in early 2025 (around 18 months ago) and the agentic flavor followed in fall 2025 (approximately 9 months ago). Besides all its benefits and potential, it also carries some fundamental risks for IT security. And the agentic approach added very severe risks while making others much more dangerous.   With all the motivation to explore this fascinating new tool we should not ignore the risks but actively address them. We present (I) an assessment of the IT security risks, (II) a concept for mitigating them without breaking its benefits, and (III) an overview about an implementation of our concept. In this very dynamic field
    
[^20]: 评估针对大语言模型生成代码中包幻觉的推理时防御方法

    Evaluating Inference-Time Defenses Against Package Hallucination in LLM-Generated Code

    [https://arxiv.org/abs/2608.22652](https://arxiv.org/abs/2608.22652)

    本文发现现有评估方法高估了LLM代码生成中的包幻觉率，并提出RAG等推理时防御方法能有效降低幻觉，在多数模型和语言组合中表现最佳。

    

    arXiv:2608.22652v1 公告类型：交叉 摘要：大语言模型（LLMs）越来越多地用于代码生成，然而它们经常幻觉出不存在的软件包，从而为软件供应链创造了可利用的入口点。我们针对此问题做出了四项贡献。首先，我们表明先前的评估方法通过在某些语言中将标准库模块误分类为幻觉，系统地夸大了幻觉率。对于Python，这种高估达到了9.4个百分点。其次，我们评估了七种用于缓解包幻觉的推理时防御方法，包括五种引导式解码策略（Greedy、Contrastive、DoLa、Nudging和Active Layer-Contrastive Decoding）、一种迭代自优化方法（Self-Refine）以及一种基于检索增强生成（RAG）的防御方法。在跨越五个模型家族和四种编程语言（Python、JavaScript、Ruby、Rust）的八个模型中，RAG在32个模型-语言组合中的18个中降低了包幻觉率（PHR）。

    arXiv:2608.22652v1 Announce Type: cross  Abstract: LLMs are increasingly used for code generation, yet they frequently hallucinate non-existent software packages, creating exploitable entry points into the software supply chain. We make four contributions to this problem. First, we show that prior evaluation methodologies systematically inflate hallucination rates by misclassifying standard-library modules as hallucinations in some languages. For Python, the overestimation reaches 9.4 percentage points. Second, we evaluate seven inference-time defenses for mitigating package hallucinations, including five guided decoding strategies (Greedy, Contrastive, DoLa, Nudging, and Active Layer-Contrastive Decoding), an iterative self-refinement approach (Self-Refine), and a Retrieval-Augmented Generation (RAG)-based defense.. Across eight models spanning five families and four programming languages (Python, JavaScript, Ruby, Rust), RAG reduces the package hallucination rate (PHR) in 18 of 32 mo
    
[^21]: 不要复制/粘贴：AI辅助编程中复制的软性障碍

    Do Not Copy/Paste: Soft Barriers for Copying in AI-Assisted Programming

    [https://arxiv.org/abs/2608.22638](https://arxiv.org/abs/2608.22638)

    本文提出“软性障碍”机制，通过Unicode扰动等技术手段，在AI编程助手中管理模型代码向软件的交接，以平衡生成速度与代码审查、安全等需求。

    

    arXiv:2608.22638v1 公告类型：交叉 摘要：从聊天窗口复制一个函数到编辑器中不到一秒钟。对于AI编码工具的许多用途而言，这种速度是重点；但在编程教育、代码审查和安全敏感开发等场景中，这也可能成为问题。本文将复制粘贴框定为“AI代码交接问题”：模型生成的文本从对话上下文转移到可执行或已提交软件的那一刻，是一个设计边界，而当前工具大多未对此进行管理。我们认为，AI编码助手不仅应通过其生成的代码来评估，还应通过其如何调节该代码向软件工件的转移来评估。我们提出“软性障碍”作为一类交接感知机制。软性障碍保留了对AI辅助的访问，同时使未经检查的转移不再那么无摩擦。作为初步技术探索，我们通过保留可行性的Unicode输出扰动来实例化这一想法。

    arXiv:2608.22638v1 Announce Type: cross  Abstract: Copying a function from a chat window into an editor takes less than a second. For many uses of AI coding tools, that speed is the point; in settings such as programming education, code review, and security-sensitive development, it can also be the problem. This paper frames copy-paste as an \emph{AI code handoff problem}: the moment model-generated text crosses from a conversational context into executable or committed software is a design boundary that current tools leave largely unmanaged. We argue that AI coding assistants should not only be evaluated by the code they generate, but also by how they mediate the transfer of that code into software artifacts. We propose \emph{soft barriers} as one class of handoff-aware mechanisms. Soft barriers preserve access to AI assistance while making unexamined transfer less frictionless. As an initial technical probe, we instantiate this idea using Unicode output perturbations that preserve vi
    
[^22]: 基准测试巨头：.NET生态系统中LLM代码生成质量的多维度实证评估

    Benchmarking the Titans: A Multi-Dimensional Empirical Evaluation of LLM Code Generation Quality in the .NET Ecosystem

    [https://arxiv.org/abs/2608.22529](https://arxiv.org/abs/2608.22529)

    本文提出了一个多维度C#代码生成评估框架，对比GPT、Gemini、Claude和Grok四个LLM在85个任务上的表现，弥补了现有基准仅关注Python和单一Pass@k指标的不足。

    

    评估大型语言模型（LLM）代码生成质量，不仅需要考察生成的代码是否正确，还需评估其可维护性、效率和风格合理性，这些品质对软件工程实践者直接相关。现有基准测试将评估简化为单一的Pass@k指标，这掩盖了功能正确性与结构质量之间的关键权衡。另一个局限是几乎只关注Python，导致C#和.NET等企业相关生态系统缺乏专门评估。本文提出了一种自动化的多维度C#代码生成评估框架，并将其应用于四个最先进的LLM：GPT、Gemini、Claude和Grok。我们在源自HumanEval的85个算法任务上进行了受控实验，共生成并评估340个解决方案，其中每个解决方案从三个独立维度进行评估。

    arXiv:2608.22529v1 Announce Type: new  Abstract: Evaluating Large Language Model (LLM) code generation quality requires examining not just whether the generated code is correct, but whether it is maintainable, efficient, and stylistically sound, all of which are qualities of direct importance to software engineering practitioners. Existing benchmarks reduce evaluation to a single Pass@k metric, which obscures critical trade-offs between functional correctness and structural quality. A further limitation is the near-exclusive focus on Python, leaving enterprise-relevant ecosystems such as C# and .NET without dedicated evaluation. This paper presents an automated, multi-dimensional evaluation framework for C# code generation, applying it to four state-of-the-art LLMs: GPT, Gemini, Claude, and Grok. We conduct a controlled experiment across 85 algorithmic tasks derived from HumanEval, generating and evaluating 340 solutions in total, in which each solution is assessed across three indepen
    
[^23]: KONTOGRAPH：在200毫秒决策预算下实现实时反洗钱的点时间特征一致性与摊销解释验证

    KONTOGRAPH: Verified Point-in-Time Feature Consistency and Amortised Explanation for Real-Time Anti-Money Laundering under a 200 ms Decision Budget

    [https://arxiv.org/abs/2608.22389](https://arxiv.org/abs/2608.22389)

    本文提出KONTOGRAPH，一个在200毫秒严格预算下运行的实时反洗钱流水线，通过时间图网络和节点内存显著提升检测性能，并满足欧盟即时支付法规要求。

    

    摘要：欧盟条例(EU) 2024/886要求欧洲支付服务提供商在十秒内全天候结算欧元信用转账。这消除了传统反洗钱(AML)分析依赖的隔夜批处理窗口以及使资金追回成为可能的结算延迟，迫使检测、解释和决策必须在个位数秒的时间窗口内完成。我们提出了KONTOGRAPH，一个为SEPA即时支付通道设计的端到端反洗钱流水线，在自设的200毫秒第99百分位预算下运行，并报告了一项基于1,562,860笔模拟支付（包含注入的洗钱模式和故意不完整的标签）的实证研究。除系统本身外，三个发现值得关注。首先，带有节点内存的时间图网络在PR-AUC上比梯度提升表格基线从0.0053提升到0.1717，配对日分块自举差异为+0.166，95%置信区间[0.105, 0.241]；仅节点内存一项就使性能提升了一倍以上。

    arXiv:2608.22389v1 Announce Type: cross  Abstract: Regulation (EU) 2024/886 obliges European payment service providers to settle euro credit transfers in under ten seconds, around the clock. This removes both the overnight batch window in which anti-money-laundering (AML) analytics traditionally ran and the settlement delay that made recovery possible, forcing detection, explanation and decision inside a single-digit-second envelope. We present KONTOGRAPH, an end-to-end AML pipeline for the SEPA Instant rail built under a self-imposed 200 ms 99th-percentile budget, and report an empirical study on 1,562,860 simulated payments with injected typologies and deliberately incomplete labels. Three findings are of interest beyond the system itself. First, a temporal graph network with per-node memory improves PR-AUC over a gradient-boosted tabular baseline from 0.0053 to 0.1717, a paired day-blocked bootstrap difference of +0.166 with 95% CI [0.105, 0.241]; per-node memory alone more than dou
    
[^24]: 通过潜在图学习学习代码的谱表示以实现跨语言代码克隆检测的泛化

    Learning Spectral Representations of Code through Latent Graph Learning for Generalizable Cross-Language Code Clone Detection

    [https://arxiv.org/abs/2608.22383](https://arxiv.org/abs/2608.22383)

    本文提出SPECTRA-Siam，一种通过潜在图学习生成跨语言可比较的谱表示，从而提升代码克隆检测泛化能力的新方法。

    

    arXiv:2608.22383v1 公告类型：新 摘要：当前的代码克隆检测（CCD）方法依赖于固定的、特定于语言的图表示，如抽象语法树（AST）或程序依赖图（PDG）。由于功能相同的代码片段可能产生截然不同的结构，这些刚性图会产生非判别性的谱，其性能接近随机水平。为了解决这个问题，我们提出了SPECTRA-Siam，一种孪生潜在图学习网络，通过优化下游CCD性能，学习一个潜在空间，使得图的谱作为代码功能的判别性签名。给定片段的AST和数据依赖关系，SPECTRA-Siam通过软槽分配和多头注意力诱导一个固定大小的加权潜在图，并从其归一化拉普拉斯算子中提取多尺度谱表示。将所有片段映射到这个共享空间，可以在不同编程语言间产生可比较的谱。在BigCloneBench、At上的实验...

    arXiv:2608.22383v1 Announce Type: new  Abstract: Current code clone detection (CCD) methods rely on fixed, language-specific graph representations like abstract syntax trees (ASTs) or program dependency graphs (PDGs). Because functionally identical code fragments can yield wildly different structures, these rigid graphs produce non-discriminative spectra that perform close to chance. To address this, we propose SPECTRA-Siam, a Siamese latent graph learning network that learns a latent space such that the graph's spectrum serves as a discriminative signature of code functionality by optimizing downstream CCD performance. Given a fragment's AST and data-dependencies, SPECTRA-Siam induces a fixed-size weighted latent graph through soft slot assignment and multi-head attention, and extracts a multi-scale spectral representation from its normalized Laplacian. Mapping all fragments into this shared space yields comparable spectra across programming languages. Experiments on BigCloneBench, At
    
[^25]: 缓解思维链中的错误传播：用于智能合约修复的思维树框架

    Mitigating Error Propagation in Chain-of-Thought: A Tree-of-Thought Framework for Smart Contract Repair

    [https://arxiv.org/abs/2608.22345](https://arxiv.org/abs/2608.22345)

    提出一种结合文档解析、静态分析和思维树推理的智能合约修复框架，通过并行探索修复路径和错误消除，显著提高了修复成功率。

    

    智能合约支撑着去中心化金融（DeFi）和NFT等区块链应用。然而，一旦部署，它们便无法修改。即便是微小的缺陷也可能导致重大的财务损失。当前基于人工智能的修复方法依赖线性推理，这会导致错误的累积和不可靠的补丁。我们的方法结合了文档解析、静态分析和思维树推理。我们首先将审计报告转换为结构化数据，然后使用Slither定位确切的漏洞代码。我们的三步框架同时探索多条修复路径，评估选项，并消除不良选择。最后，我们通过编译和人工检查验证补丁。我们在Code4Rena的50个真实漏洞上测试了我们的方法。我们的方法实现了62%的单次成功率和84%的前三成功率，分别比ContractTinker高出12和6个百分点。我们还提高了完全有效补丁的比例。

    arXiv:2608.22345v1 Announce Type: cross  Abstract: Smart contracts power blockchain applications such as DeFi and NFTs. However, once deployed, they cannot be modified. Even minor bugs can result in significant financial losses. Current AI-based repair methods rely on linear reasoning, which leads to the accumulation of errors and unreliable patches. Our method combines document parsing, static analysis, and Tree of Thoughts reasoning. We first convert audit reports into structured data. Then we use Slither to locate the exact vulnerable code. Our three-step framework explores multiple repair paths simultaneously, evaluates options, and eliminates poor choices. Finally, we verify patches through compilation and manual checks. We test our method on 50 real vulnerabilities from Code4Rena. Our method achieves a 62% single success rate and an 84% top-3 success rate, outperforming ContractTinker by 12 and 6 percentage points, respectively. We also increase the proportion of fully effective 
    
[^26]: 从测试中学习：深度强化学习代理的自引用差分测试

    Learning from the Test: Self-Referential Differential Testing for Deep RL Agents

    [https://arxiv.org/abs/2608.22284](https://arxiv.org/abs/2608.22284)

    我们提出Delta框架，通过两阶段方法自动识别DRL代理的安全关键性和最优性缺陷，解决最优性测试的预言机问题。

    

    arXiv:2608.22284v1 公告类型：交叉 摘要：深度强化学习（DRL）在复杂决策问题中取得了显著成功。随着DRL系统越来越多地部署在现实应用中，确保其质量和可靠性至关重要。当前工作主要集中于检测安全关键性故障，往往忽视策略最优性，这可能导致效率降低、用户不信任和经济损失。这种忽视，加上最优性固有的“测试预言机问题”，在全面评估DRL系统方面留下了显著空白。为解决这一空白，我们提出了Delta（DRL代理的差分测试），一个新颖且全面的框架，能自动识别DRL代理中的安全关键性和最优性缺陷。Delta采用两阶段方法：（1）安全测试，在此阶段，被测代理（AUT）在收集其决策策略数据的同时评估灾难性故障，以及（2）最优性测试，利用这些数据通过自引用差分测试检测次优行为。

    arXiv:2608.22284v1 Announce Type: cross  Abstract: Deep Reinforcement Learning (DRL) has achieved significant success in complex decision-making problems. As DRL systems are increasingly deployed in real-world applications, ensuring their quality and reliability is paramount. Current works primarily focus on detecting safety-critical failures, often neglecting policy optimality, which can lead to reduced efficiency, user distrust, and economic losses. This oversight, compounded by the inherent "testing oracle problem" for optimality, leaves a significant gap in comprehensively evaluating DRL systems. To address this gap, we propose Delta (Differential Testing for DRL Agents), a novel and comprehensive framework that automatically identifies both safety-critical and optimality bugs in DRL agents. Delta employs a two-phase approach: (1) Safety Testing, where the Agent Under Test (AUT) is evaluated for catastrophic failures while collecting data from its decision-making policy, and (2) Op
    
[^27]: CodeMechanic：基于缺陷属性引导的程序缓解

    CodeMechanic: Bug-Property-Guided Program Mitigation

    [https://arxiv.org/abs/2608.22275](https://arxiv.org/abs/2608.22275)

    CodeMechanic通过从崩溃中重建内存安全属性并插入本地失败停止防护，生成受限缓解，避免了LLM修复代理因弱验证导致的不可靠修复。

    

    自动化测试发现漏洞的速度快于开发者调查和修复的速度，这导致已知的内存破坏在可被利用的期间内仍然存在。端到端的LLM修复代理可以缩短这一间隔，但它们合成开放式的代码更改，通常仅通过重放概念验证（PoC）来验证。这种弱验证机制会接受那些通过改变无关行为来掩盖观察到的崩溃的补丁，从而使得非预期的部署变得危险。我们提出了CodeMechanic，一个基于缺陷属性引导的系统，用于生成针对空间内存破坏的受限缓解措施。CodeMechanic不要求LLM生成永久修复，而是从崩溃中重建被违反的内存安全属性，验证解引用的指针及其缓冲区范围，并在危险访问之前插入一个本地失败停止防护。当边界检查失败时，防护会终止执行。由此产生的缓解措施...

    arXiv:2608.22275v1 Announce Type: new  Abstract: Automated testing discovers vulnerabilities faster than developers can investigate and repair them, leaving an interval in which known memory corruptions remain exploitable. End- to-end LLM repair agents can shorten this interval, but they synthesize open-ended code changes and commonly validate them only by replaying a proof of concept (PoC). This weak oracle accepts patches that silence the observed crash by changing unrelated behavior, making unintended deployment risky.   We present CodeMechanic, a bug-property-guided system for generating constrained mit- igations for spatial memory corruption. Instead of asking an LLM to generate a permanent repair, CodeMechanic reconstructs the violated memory-safety property from the crash, validates the dereferenced pointer and its buffer range, and inserts a local fail-stop guard before the dangerous access. The guard terminates execution when the boundary check fails. The resulting mitigation 
    
[^28]: D-Diff：一种基于可编辑三方差异的交互式提交边界调整环境

    D-Diff: An Interactive Environment for Adjusting Commit Boundaries Based on an Editable 3-way Diff

    [https://arxiv.org/abs/2608.22207](https://arxiv.org/abs/2608.22207)

    D-Diff通过将两个连续提交的差异整合到一个可编辑的三方差异视图中，显著降低了调整提交边界的认知负担。

    

    在版本控制中，建议每次提交只包含与一个任务相关的更改。为了遵循这一建议，开发者可能需要调整提交边界，即比较和修改两个连续提交之间的差异。现有工具要么一次只显示单个差异，迫使开发者在比较差异时依赖记忆，要么同时显示三个文件而不展示两个连续的差异，迫使开发者自行推断；这些都增加了他们的认知负担并妨碍了调整。作为支持这一过程的第一步，我们提出了D-Diff，一种针对两个涉及同一文件的连续提交的交互式差异调整环境。基于三方差异显示，D-Diff将两个差异整合到一个紧凑的视图中，并提供修改它们的方式。一项有8名参与者的受试者内用户研究表明，D-Diff显著优于现有方法。

    arXiv:2608.22207v1 Announce Type: new  Abstract: In version control, it is recommended that each commit include only changes related to one task. To follow this recommendation, developers may need to adjust commit boundaries, that is, to compare and modify the diffs between two consecutive commits. Existing tools either display only a single diff at a time, forcing developers to rely on their memory when comparing diffs, or display three files simultaneously without showing the two consecutive diffs, forcing developers to infer them; both increase their cognitive load and hamper the adjustment. As a first step toward supporting this process, we propose D-Diff, an interactive diff adjustment environment for two consecutive commits that each involve the same single file. Based on a 3-way diff display, D-Diff integrates the two diffs into a single compact view, which also provides a way to modify them. A within-subjects user study with 8 participants showed that D-Diff significantly outpe
    
[^29]: 分歧探索，一致提交：路由引导的软件代理测试时扩展

    Disagree to Explore, Agree to Commit: Routing-Guided Test-Time Scaling for Software Agents

    [https://arxiv.org/abs/2608.22191](https://arxiv.org/abs/2608.22191)

    提出Risa方法，利用原生MoE路由信号在测试时引导软件代理的探索与收敛，无需外部评判或测试执行，显著提升仓库级任务修复效率。

    

    软件工程代理通过长序列、随机性的工具使用轨迹解决仓库级任务，多次尝试往往能发现单次运行遗漏的修复。测试时扩展面临挑战，因为补丁缺乏标准答案形式，而共享前缀的兄弟动作之间存在相关性。我们研究了原生MoE路由器轨迹是否能在无外部评判或选择时测试执行的情况下，引导代理的决策和选择。分析表明，路由提供了稳健的行为角色信号；令牌级读取和决策匹配的比较集将其转化为有效控制。因此，我们引入了Risa（路由引导的导向与仲裁）：在轨迹内部，路由鼓励多样化探索并在补丁提交阶段实现受控收敛；在跨独立采样的轨迹之间，信息丰富补丁位置的一致性选择最终候选。我们在SWE-bench Verified上使用开放权重稀疏模型进行了评估。

    arXiv:2608.22191v1 Announce Type: new  Abstract: Software-engineering agents solve repository-level tasks through long, stochastic tool-use trajectories, and repeated attempts often find fixes missed by one run. Test-time scaling is difficult because patches lack canonical answer forms, while sibling actions from a shared prefix are correlated. We study whether native MoE router traces can guide steering and selection without an external judge or selection-time test execution. Our analysis shows that routing provides a robust behavioral role signal; token-granular readouts and decision-matched comparison sets turn it into effective control. We therefore introduce Risa (Routing-Informed Steering and Arbitration): within trajectories, routing encourages diverse exploration and controlled convergence during patch commitment; across separately sampled trajectories, agreement at informative patch positions selects a final candidate. We evaluate on SWE-bench Verified using open-weight sparse
    
[^30]: 走向可操作的视觉化：十年后，生成式AI改变了什么以及它无法改变什么

    Towards Actionable Visualization: Ten Years Later, What Generative AI Changes and What It Cannot

    [https://arxiv.org/abs/2608.22151](https://arxiv.org/abs/2608.22151)

    本文回顾十年前软件可视化研究，指出生成式AI通过降低按需生成成本，将重点从生产人工制品转向人类感知与指导，并发现AI使被忽视领域（如理由）变得可行，但工具可持续性问题可能先恶化后改善。

    

    arXiv:2608.22151v1 公告类型：新 摘要：十年前，我们调查了软件可视化研究，假设其采用挑战在于将开发者的需求与技术相匹配。生成式AI可能通过将按需生成可视化的成本推向零，使这一假设过时。我们认为，这反映了软件工程中已经发生的更广泛反转：随着AI使人工制品的生产贬值，它提升了人类感知和指导这些制品的工作。回顾我们2016年的研究，我们发现我们标记为被忽视的领域，如理由（rationale），正是AI现在使其易于处理的领域，而当时我们诊断出的工具可持续性问题，生成式AI在帮助解决之前可能会先加剧它。一个发现完全转变：通过沉浸式环境交付的研究比例增长了约十五倍，尽管它仍是该领域的小部分。我们认为，未来研究应聚焦于帮助……

    arXiv:2608.22151v1 Announce Type: new  Abstract: Ten years ago, we surveyed software visualization research under the assumption that the challenge for adoption was matching developers' needs with techniques. Generative AI might have made that assumption obsolete by driving the cost of producing a visualization on demand toward zero. We argue that this mirrors a broader inversion already underway in software engineering: as AI devalues the production of artifacts, it elevates the human work of perceiving and directing them. Looking back at our 2016 research, we found that domains we flagged as neglected, such as rationale, are exactly the ones AI now makes tractable, and a tool-sustainability problem we diagnosed then is one generative AI may worsen before it helps solve. One finding shifted outright: the share of studies delivered through immersive environments grew roughly fifteen-fold, though it remains a small minority of the field. We argue that future research should focus on hel
    
[^31]: 约束驱动建模实现离散事件系统的双模型检查与仿真

    Constraint-Driven Modeling Enabling Dual Model Checking and Simulation for Discrete Event Systems

    [https://arxiv.org/abs/2608.22095](https://arxiv.org/abs/2608.22095)

    本文提出Constraint-DEVS方法，通过扩展DEVS-Suite框架，实现有界并行DEVS模型的双重模型检查与仿真，支持非确定性和性能验证，为离散事件系统提供独特的开发与验证框架。

    

    验证与确认（V&V）是评估动态模型满足其预期目的的需求和规范的关键方法。并行离散事件系统规范（PDEVS）是一种基于系统理论的建模方法，用于创建模块化、层次化的组件式仿真模型。在本文中，我们引入了Constraint-DEVS，一种创建有界并行DEVS模型的方法，这些模型除了支持仿真外，还适用于模型检查。我们扩展了DEVS-Suite框架，以创建Constraint-DEVS规范，然后可以通过提出的状态探索协议与并行DEVS抽象模拟器协议进行模型检查。这些能力，以及对非确定性、复杂数据传输和性能相关属性检查的支持，使Constraint-DEVS及其配套的DEVS-Suite成为一个独特的框架，用于动态系统的开发、验证和确认。

    arXiv:2608.22095v1 Announce Type: new  Abstract: Verification and validation (V&V) are crucial methods for evaluating the requirements and specifications of dynamical models that fulfill their intended purposes. Parallel Discrete EVent System Specification (PDEVS) is a system-theoretic modeling approach for creating modular, hierarchical component-based simulation models. In this paper, we introduce Constraint-DEVS, a method for creating bounded Parallel DEVS models that lend themselves, in addition to simulation, to model checking. We extend the DEVS-Suite framework to create Constraint-DEVS specifications which can then be model checked using a proposed state exploration protocol with the Parallel DEVS abstract simulator protocol. These capabilities, along with the support for non-determinism, complex data transfer, and performance-related property checking, make Constraint-DEVS and its accompanying DEVS-Suite a unique framework for the development, verification, and validation of di
    
[^32]: W-RAG：面向异构知识库的企业文档生成的源感知检索

    W-RAG: Source-Aware Retrieval for Enterprise Document Generation from Heterogeneous Knowledge Bases

    [https://arxiv.org/abs/2608.22081](https://arxiv.org/abs/2608.22081)

    W-RAG通过本体引导检索和每个知识库内的局部排序，解决了企业文档生成中异构知识库全局排序导致的不平衡上下文问题，从而生成更完整的草稿。

    

    检索增强生成（RAG）使大型语言模型能够在生成过程中融入外部知识，从而提升事实依据和领域适应性。然而，现有的RAG流程假设从多个知识库中检索到的证据可以通过单一相似度函数进行全局排序。虽然这种假设适用于开放域检索，但在企业文档生成中却失效了，因为异构知识库（如政策、法规、技术文档和部门指南）各自扮演不同的角色，且必须在生成的文档中共同呈现。因此，全局排序常常会产生由部分来源主导的不平衡上下文，导致企业草稿不完整。为解决这一局限性，我们提出了W-RAG，一种源感知检索框架，该框架执行本体引导的检索、在每个知识库内进行局部排序，以及源级加权。

    arXiv:2608.22081v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) enables large language models to incorporate external knowledge during generation, improving factual grounding and domain adaptability. However, existing RAG pipelines assume that evidence retrieved from multiple repositories can be ranked globally using a single similarity function. While suitable for open-domain retrieval, this assumption breaks down in enterprise document generation, where heterogeneous knowledge bases (such as policies, regulations, technical documentation, and departmental guidelines) serve distinct roles and must be jointly represented in the generated document. As a result, global ranking often produces unbalanced context dominated by a subset of sources, leading to incomplete enterprise drafts. To address this limitation, we propose W-RAG, a source-aware retrieval framework that performs ontology-guided retrieval, local ranking within each knowledge base, and source-level we
    
[^33]: 基于问题信息的LLM增强提交消息生成：一项探索性研究

    LLM-Enhanced Commit Message Generation via Issue Information: An Exploratory Study

    [https://arxiv.org/abs/2608.22004](https://arxiv.org/abs/2608.22004)

    本文提出ISAC框架，通过结合代码差异和问题信息作为LLM输入，系统证明了在提交消息生成中纳入问题信息能持续提升性能，尤其对CIDEr指标改善最大。

    

    arXiv:2608.22004v1 公告类型：新 摘要：提交消息帮助开发者理解代码变更、支持协作并改善长期维护。然而，仅将问题信息作为基于LLM的提交消息生成（CMG）的外部上下文尚未被系统研究。我们提出了一种结合代码差异与问题信息作为LLM输入的ISsue-Augmented框架（ISAC）用于提交消息生成。为支持评估，我们构建了ApacheCM-Issue，这是一个基于ApacheCM构建的提交-问题对齐数据集，通过链接GitHub和Apache Jira中的问题与提交。使用来自Scala、Java和C++项目的样本，我们评估了两种代表性LLM（GPT-5.5和DeepSeek-V4-Flash）在不同推理配置下的四种输入配置。结果表明，在所有评估的模型配置和指标中，纳入问题信息持续改善了基于LLM的CMG，其中CIDEr的提升最大。纳入类似历史信息也带来了显著收益。

    arXiv:2608.22004v1 Announce Type: new  Abstract: Commit messages help developers understand code changes, support collaboration, and improve long-term maintenance. However, the use of issue information alone as the external context for LLM-based CMG has not been systematically studied. We propose an ISsue-Augmented framework for Commit message generation (ISAC) by combining code diffs with issue information as LLM input. To support the evaluation, we construct ApacheCM-Issue, a commit-issue aligned dataset built upon ApacheCM by linking commits with issues from GitHub and Apache Jira. Using samples from Scala, Java, and C++ projects, we evaluate four input configurations using two representative LLMs, GPT-5.5 and DeepSeek-V4-Flash in different reasoning configurations. The results show that incorporating issue information consistently improves LLM-based CMG across all evaluated model configurations and metrics, with the largest gains observed for CIDEr. Incorporating a similar historic
    
[^34]: NVD CWE标签的可靠性如何？基于Seclometry的大规模语义审计

    How Reliable Are NVD CWE Labels? A Large-Scale Semantic Audit with Seclometry

    [https://arxiv.org/abs/2608.21977](https://arxiv.org/abs/2608.21977)

    本文通过构建基于seclometry的CWEAgent审计工具，首次大规模测量了NVD中CWE标签的可靠性，发现其准确率有限，揭示了现有标签体系存在的系统性质量问题。

    

    arXiv:2608.21977v1 公告类型：交叉 摘要：国家漏洞数据库（NVD）中的CWE标签被广泛视为漏洞搜索、扫描器评估、基准构建、基于学习的安全工具以及漏洞优先级排序的基准真相。然而，尽管人们对NVD的丰富性积压问题日益担忧，并有关于标签不准确、模糊或缺失的轶事报告，其可靠性尚未在规模上得到系统测量。本文提出了一种基于代码语义的大规模NVD CWE标签质量测量方法。我们构建了CWEAgent，一种基于seclometry（一种结构化漏洞语义表示，捕获漏洞代码的根本原因、触发条件、违反的安全属性、利用机制和影响）的验证审计工具。在手动策划的100个开源CVE基准上，CWEAgent实现了85%的top-1准确率和92%的模糊感知准确率。将CWEAgent应用于15,556个开源CVE中。

    arXiv:2608.21977v1 Announce Type: cross  Abstract: CWE labels in the National Vulnerability Database (NVD) are widely treated as ground truth for vulnerability search, scanner evaluation, benchmark construction, learning-based security tools, and vulnerability prioritization. Yet their reliability has not been systematically measured at scale, despite growing concerns about NVD's enrichment backlog and anecdotal reports of inaccurate, ambiguous, or missing labels. This paper presents a large-scale, code-semantics-grounded measurement of CWE labeling quality in NVD. We build CWEAgent, a validated auditing instrument based on seclometry, a structured representation of vulnerability semantics that captures the root cause, trigger condition, violated security property, exploit mechanism, and impact of vulnerable code. On a manually curated benchmark of 100 open-source CVEs, CWEAgent achieves 85% top-1 accuracy and 92% ambiguity-aware accuracy. Applying CWEAgent to 15,556 open-source CVEs d
    
[^35]: Repo2Skill-Evo：仓库技能在静默中过时

    Repo2Skill-Evo: Repository Skills Go Stale in Silence

    [https://arxiv.org/abs/2608.21964](https://arxiv.org/abs/2608.21964)

    本文提出Repo2Skill-Evo框架，研究智能体在软件仓库版本更新时维护和更新外部化技能的能力，揭示技能过时但无显式信号的问题。

    

    arXiv:2608.21964v1 公告类型：新公告 摘要：大型语言模型（LLM）智能体越来越多地操作于不断演化的软件仓库中，其成功取决于仓库特定的程序性知识：调用哪些API、运行哪些脚本，以及当前版本期望的惯例。智能体技能将这些知识外部化为可重用单元，先前研究表明它们能提升智能体性能。尚不清楚的是，这种提升是否持久。使技能有用的版本特异性同样使其脆弱：发布后，技能可能在没有明确信号的情况下变得过时，同时继续提供过时的指导。将知识外部化为技能因此可能使其衰减变得不可见。我们研究智能体是否能保持这种外部化知识的时效性。Repo2Skill-Evo将每次版本过渡视为技能维护任务：给定V1技能集和官方V1到V2补丁，智能体必须更新过时的技能。

    arXiv:2608.21964v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly operate over evolving software repositories, where success depends on repository-specific procedural knowledge: which APIs to call, which scripts to run, and which conventions the current release expects. Agent skills externalize this knowledge into reusable units, and prior work shows that they can improve agent performance. What remains unclear is whether that improvement is durable. The same version specificity that makes a skill useful also makes it fragile: after a release, it may become stale without raising any explicit signal, while continuing to provide obsolete guidance. Externalizing knowledge into a skill can therefore make its decay invisible.   We study whether agents can keep this externalized knowledge current. Repo2Skill-Evo casts each release transition as a skill-maintenance task: given a V1 skill set and the official V1-to-V2 patch, an agent must update obsolete skill con
    
[^36]: 基于本体的设计参数管理用于变更影响分析

    Ontology-supported Design Parameter Management for Change Impact Analysis

    [https://arxiv.org/abs/2608.21949](https://arxiv.org/abs/2608.21949)

    本文提出了一种基于本体的设计参数管理方法，通过需求追溯和专家知识库实现变更影响分析，适用于软硬件设计，并能提高异构系统间的知识传递效率。

    

    本文提出了一种基于本体的工程设计参数管理方法。该方法旨在通过需求追溯性和设计参数的专业知识来实现变更影响分析。该方法适用于软件和硬件设计。其主要活动和特性通过以下方式获得：(1) 应用基于本体的通用系统建模过程提案进行模型集成，(2) 利用知识库捕获专家知识，(3) 构建语义任务剖面感知设计平台。使用OWL表示信息，底层数据模型可以改善复杂工程项目中常见的异构系统间的知识传递，同时减少对此类模型进行推理的工作量。通过两个说明性用例的演示和实操描述作为补充。

    arXiv:2608.21949v1 Announce Type: new  Abstract: This paper presents an ontology-supported approach to the management of design parameters in engineering. This approach aims specifically at enabling Change Impact Analysis through Requirements Traceability and acquainted expert knowledge of design parameters. The approach is suitable for both software and hardware designs. The activities and features are mainly obtained by (1) the application of an ontology-based universal system modeling procedure proposal for model integration, (2) the utilization of a knowledge base for capturing expert knowledge and (3) a semantic Mission Profile Aware Design platform. OWL is used to represent information and the underlying data model can improve knowledge transfer among heterogeneous systems which are common in complex engineering projects. At the same time, effort to perform reasoning on such models can be reduced. A demonstration and hands-on description of two illustrative use cases complements 
    
[^37]: 基于本体的需求转换

    Ontology-based Requirements Transformation

    [https://arxiv.org/abs/2608.21945](https://arxiv.org/abs/2608.21945)

    本文提出一种基于本体的方法论和系统，利用OWL语言和模型支持，将任务剖面需求进行供应链感知转换，以提高工程效率、促进知识集成与重用。

    

    本文提出了一种基于本体的方法，用于将所谓的任务剖面（MPs）给出的功能性和环境负载需求进行供应链感知转换。该方法旨在通过支持转换过程并实现转换与现有基于模型的系统工程（MBSE）流程的更好集成，来提高工程过程的效率。我们提出了一种方法论和一个支持系统，该系统辅助转换过程，而后者的特性通过构建和处理模型来实现。一致使用标准化语言OWL来表达模型表示，进一步促进了异构系统之间的知识集成和转移。此外，这有利于跨项目的知识重用，从而可能降低总体成本。而且，该系统能够从任务剖面中剥离无关信息，从而改进过程。

    arXiv:2608.21945v1 Announce Type: new  Abstract: This paper presents an ontology-based approach to the supply chain-aware transformation of functional and environmental load requirements given by so-called Mission Profiles (MPs). The approach aims at improving the efficiency of the engineering process through supporting the transformation process and enabling a better integration of the transformation into existing Model-based Systems Engineering (MBSE) processes. We propose a methodology and a supporting system which aids in the transformation process while the latter feature is obtained by constructing and working on models. Consequent utilization of the standardized language OWL to express model representations further enables better knowledge integration and transfer among hetero-geneous systems. In addition to that, this favors knowledge reuse across projects which can reduce overall costs. Moreover, the system enables stripping off irrelevant information from MPs, thus improving 
    
[^38]: 循环工程：构建模块、采用与影响

    Loop Engineering: Building Blocks, Adoption, and Impact

    [https://arxiv.org/abs/2608.21884](https://arxiv.org/abs/2608.21884)

    本文首次探索性回顾了“循环工程”这一新兴实践，即开发者设计系统自动触发和停止智能体运行，并总结了其核心构建模块（如停止条件、状态文件和验证子智能体），但未量化其实际采用率。

    

    arXiv:2608.21884v1 公告类型：新 摘要：在过去的几个月里，开发者指导智能体AI编码工具的方式已跨越多个抽象层次，从措辞提示到工程上下文，再到配置模型周围的框架。2026年6月，从业者开始描述一个称为“循环工程”的进一步层次：开发者不再交互式地提示智能体，而是设计系统让智能体自动提示自身。这些系统按计划或仓库事件启动智能体运行，并在机器可检查条件满足时停止它们。该术语迅速传播，伴随着大胆的主张和直言不讳的怀疑，但其在软件项目中的采用情况尚未被量化。我们提出对新兴灰色文献的探索性回顾，这些文献大致一致认为一个良好设计的循环包含：由机器可检查停止条件约束的触发式智能体运行、持久状态文件、验证子智能体、令牌预算，以及定义好的升级到人工处理的节点。

    arXiv:2608.21884v1 Announce Type: new  Abstract: Over the past months, the way developers direct agentic AI coding tools has moved up several levels of abstraction, from phrasing prompts to engineering context to configuring the harness around the model. In June 2026, practitioners began to describe a further level called loop engineering: Instead of prompting an agent interactively, developers design systems that prompt agents for them. These systems start agent runs on a schedule or on repository events and stop them when a machine-checkable condition holds. The term spread rapidly, accompanied by bold claims and vocal skepticism, but its adoption in software projects has not been measured. We present an exploratory review of the emerging gray literature, which largely agrees on what a well-engineered loop contains: triggered agent runs bounded by machine-checkable stop conditions, persistent state files, verifier sub-agents, token budgets, and defined points of escalation to humans.
    
[^39]: 架构作为编码代理的能力均衡器

    Architecture as Capability Equalizer for Coding Agents

    [https://arxiv.org/abs/2608.21747](https://arxiv.org/abs/2608.21747)

    本论文通过对照实验发现，架构规范格式对编码代理生成代码质量的影响依赖于模型能力，对较弱模型使用代码邻近格式可显著缩小与强模型的能力差距。

    

    基于LLM的编码代理能从高层描述生成完整软件系统，然而关于架构规范格式如何影响生成代码质量，以及这种影响是否取决于模型能力，目前知之甚少。我们进行了一项对照实验，比较了五种信息等价规范格式（非正式散文、带约束和ADR的Mermaid图、OpenAPI、C4/Structurizr DSL、以及带ArchUnit风格规则的TypeScript接口契约），涉及来自三个供应商家族（Anthropic Claude、OpenAI GPT、Google Gemini）的六种模型。在90次多轮代理试验中，规范格式显示出强烈的格式×模型交互效应。在最强大的模型（Sonnet 4.6、GPT-5）上，格式影响甚微（质量差异0.17-0.92）。在较弱模型上，格式产生0.83-2.42分的差异，而接近代码的格式（OpenAPI、TypeScript契约）弥补了大部分能力差距。

    arXiv:2608.21747v1 Announce Type: cross  Abstract: LLM-based coding agents generate complete software systems from high-level descriptions, yet little is known about how the format of architecture specifications affects the quality of generated code or whether this effect depends on model capability. We present a controlled experiment comparing five informationally equivalent specification formats (informal prose, Mermaid diagrams with constraints and ADRs, OpenAPI, C4/Structurizr DSL, and TypeScript interface contracts with ArchUnit-style rules) across six models from three vendor families (Anthropic Claude, OpenAI GPT, Google Gemini). Across 90 multi-turn agent trials, specification format shows a strong format x model interaction. On the strongest models (Sonnet 4.6, GPT-5), format barely matters (quality spread 0.17-0.92). On weaker models, format produces spreads of 0.83-2.42 points, with code-proximate formats (OpenAPI, TypeScript contracts) recovering most of the capability gap.
    
[^40]: XRFix：利用大型语言模型探索扩展现实应用的性能缺陷修复

    XRFix: Exploring Performance Bug Repair of Extended Reality Applications with Large Language Models

    [https://arxiv.org/abs/2608.21718](https://arxiv.org/abs/2608.21718)

    本文提出了XRFix框架，利用大型语言模型自动检测并修复扩展现实应用中的性能缺陷，解决了缺乏真实数据集、检测工具和修复工具的三大挑战。

    

    作为一种新兴技术，扩展现实为用户提供了与虚拟和物理环境交互的沉浸式体验。与传统软件不同，XR应用的执行涉及更多计算复杂的操作，例如三维场景渲染、实时动画和过程模拟。在XR应用的软件开发过程中，低效的编码实践可能导致各种性能缺陷，降低用户体验，甚至引发晕动症。因此，迫切需要开发一个自动化的程序修复框架，用于修复复杂XR程序中的性能缺陷。然而，由于若干技术挑战，实现这一目标并非易事：（1）缺乏真实的XR代码库和缺陷数据集，（2）没有准确的缺陷检测工具，（3）没有针对XR性能缺陷设计的有效修复工具。为应对这些挑战，我们提出了一种新颖的大型语言模型。

    arXiv:2608.21718v1 Announce Type: new  Abstract: As an emerging technology, Extended Reality provides end-users with an immersive experience of interacting with virtual and physical environments. Unlike traditional software, the execution of XR applications involves more computationally complex operations, such as 3D scene rendering, real-time animation, and process simulations. Inefficient coding practices during the software development of XR applications may cause various performance bugs, degrading user experience and even causing motion sickness. Thus, it is an urgent need to develop an automated program repair framework for fixing performance bugs in complex XR programs. However, it is non-trivial to achieve this goal due to several technical challenges: (1) a lack of a real-world XR codebase and bug dataset, (2) no accurate bug detection tool, and (3) no effective bug-fixing tool designed for XR performance bugs. To tackle these challenges, we present a novel large language mode
    
[^41]: ExploreAI：用于黑盒VR和3D应用可复现可观察回归测试的智能探索知识库

    ExploreAI: Agentic Exploration Knowledge Bases for Reproducible Observable-Regression Testing of Black-Box VR and 3D Applications

    [https://arxiv.org/abs/2608.21628](https://arxiv.org/abs/2608.21628)

    ExploreAI利用大语言模型驱动智能体框架，通过语义指导的探索知识库，实现了黑盒VR和3D应用可复现且高效的回归测试。

    

    黑盒VR和3D应用难以进行回归测试，因为可观察的故障取决于测试者移动的位置、可见的对象以及捕获的视图。手动探索性测试可以找到此类故障，但其证据难以重现；系统性扫描是可重现的，但缺乏语义指导，并将探索预算花费在低价值视点上。我们观察到，大语言模型（LLM）可以做出人类测试者在探索过程中所做的高层决策：解释任务、选择要检查的对象、分组相关对象、记录所见内容，并决定何时缺失证据应触发另一次尝试。基于这一观察，我们提出了ExploreAI，一个由LLM驱动的智能体框架，将重复的感知、导航、多视角捕获执行和日志记录卸载到专用模块，同时利用LLM进行规划、证据记录、捕获策略决策以及后续操作。

    arXiv:2608.21628v1 Announce Type: new  Abstract: Black-box VR and 3D applications are difficult to regression test because observable failures depend on where a tester moves, what objects are visible, and which views are captured. Manual exploratory testing can find such failures, but its evidence is time-consuming to reproduce; systematic sweeps are reproducible, but they lack semantic guidance and spend exploration budget on low-value viewpoints. We observe that an LLM can make the high-level decisions a human tester makes during exploration: interpreting a task, choosing which objects to inspect, grouping related objects, recording what it saw, and deciding when missing evidence should trigger another attempt. Based on this observation, we present ExploreAI, an LLM-driven agentic framework that offloads repeated perception, navigation, multi-view capture execution, and logging to specialized modules while using the LLM for planning, evidence recording, capture-policy decisions, and 
    
[^42]: 大型语言模型在需求工程中的应用：跨任务实证评估

    Large Language Models for Requirements Engineering: A Cross-Task Empirical Evaluation

    [https://arxiv.org/abs/2608.21531](https://arxiv.org/abs/2608.21531)

    本研究通过受控实验和工业案例研究，跨五项任务评估了大型语言模型在需求工程中的性能，填补了跨任务评估和复现性的空白。

    

    arXiv:2608.21531v1 公告类型：新 摘要：需求相关信息分散在异构工件中，如用户反馈、开发者讨论和软件仓库，使得提取可操作的需求知识既劳动密集又难以扩展。大型语言模型（LLMs）可以支持许多需求工程（RE）活动，从分类和可追溯性识别到规范和解释生成，但现有证据在任务、工件类型和评估设置方面分散，且研究很少提供跨任务评估或复现包。我们提出了两项互补的实证研究，评估LLMs在五项与RE相关的活动中的表现。第一项是针对五个轻量级开源LLMs的受控实验，用于反馈驱动的需求分类和规范生成。第二项是探索性工业案例研究，涉及两个前沿LLMs用于可追溯性链接识别。

    arXiv:2608.21531v1 Announce Type: new  Abstract: Requirements-related information is scattered across heterogeneous artefacts such as user feedback, developer discussions, and software repositories, making the extraction of actionable requirements knowledge labour-intensive and hard to scale. Large Language Models (LLMs) can support many Requirements Engineering (RE) activities, from classification and traceability identification to specification and explanation generation, but existing evidence is fragmented across tasks, artefact types, and evaluation settings, and studies rarely offer cross-task evaluations or replication packages. We present two complementary empirical studies evaluating LLMs across five RE-related activities. The first is a controlled experiment on five lightweight open-source LLMs for feedback-driven requirements classification and specification generation. The second is an exploratory industrial case study on two frontier LLMs for traceability link identificatio
    
[^43]: 神经形式验证：智能体驱动的语言无关形式化程序推理

    Neuro-Formal Verification: Agentic Language-Agnostic Formal Program Reasoning

    [https://arxiv.org/abs/2608.21516](https://arxiv.org/abs/2608.21516)

    本文提出神经形式验证（NFV），通过AI智能体将主流语言代码翻译为形式化验证语言，实现一键式经验性准确验证，无需形式化方法专业知识。

    

    arXiv:2608.21516v1 公告类型：新 摘要：形式化验证为软件提供了最强有力的保证，而支持验证的语言已使其自动化成为现实。然而，这些好处惠及不到大多数主流开发者，他们使用的语言大多缺乏验证支持。此外，指定属性和建模环境需要形式化方法方面的专业知识。因此，证明仅限于少数著名的工件，而实际交付的生产代码仅通过审查和测试来证明。我们引入了神经形式验证（NFV），它利用这种自动化惠及主流编程语言的开发者：一个AI编码智能体负责翻译，一个成熟的验证器负责判定，以主流语言提出的问题通过按钮式操作得到回答，其准确性基于经验而非严格可靠性，并附有机检证明。在Python编程问题的正确与错误解决方案数据集上的结果令人鼓舞：NFV返回一个Dafny证明。

    arXiv:2608.21516v1 Announce Type: new  Abstract: Formal verification offers the strongest assurance available for software, and verification-aware languages have made its automation real. Yet the benefits reach few mainstream developers, most of whose languages have no verification support. Besides, specifying properties and modeling the environment require expertise in formal methods. Proof is therefore reserved for a few celebrated artifacts, while the production code that ships is attested only through review and testing.   We introduce neuro-formal verification (NFV), which harnesses that automation for developers of mainstream programming languages: an AI coding agent translates, an established verifier decides, and a question posed in a mainstream language is answered push-button, at empirical accuracy rather than soundness, with a machine-checked proof. Results on a dataset of correct and incorrect solutions to Python programming problems are encouraging: NFV returns a Dafny pro
    
[^44]: BeTaL-GBI：准入感知的基准调优与几何信念接口的全栈验证

    BeTaL-GBI: Admission-Aware Benchmark Tuning and Full-Stack Verification of Geometric Belief Interfaces

    [https://arxiv.org/abs/2608.21503](https://arxiv.org/abs/2608.21503)

    本文提出 BeTaL-GBI，一种通过 LLM 参与的基准调优方法，在验证几何信念接口时分离格式准入与条件评估，以提升架构验证的可信度和可审计性。

    

    验证基底在暴露其自身主张中的错误时，比仅暴露模型输出更具可信度。GBI-DCSE v3 推翻了一项架构主张：报告的 Fisher 值 epsilon ~ 0.066 仅在切片 [epsilon, 3, 4, 5] 上满足 kappa^2 <= 10^4 的预算，而完整盒 [epsilon, 20]^4 需要 epsilon ~ 0.326472。此勘误突显了企业验证架构能否在保持主张可审计的同时，隔离接口故障、任务能力、策略准入性和控制完整性。BoundaryBench v0.1 建立了基线：Qwen3-4B-Instruct-2507 完成了 768 次冻结执行，但 0% 通过合约（369 次解析失败，399 次验证失败），限制了下游选择性指标。本伴随研究评估了三个连续改进。首先，BeTaL-GBI v0.2 应用了带 LLM 参与的基准调优，覆盖 2,218,750,380 个网格点，将格式准入与条件评估分离。

    arXiv:2608.21503v1 Announce Type: new  Abstract: A verification substrate is more credible when exposing errors in its own claims, not just model outputs. GBI-DCSE v3 falsified an architectural claim: the reported Fisher value epsilon ~ 0.066 satisfies the kappa^2 <= 10^4 budget only on the slice [epsilon, 3, 4, 5], while the full box [epsilon, 20]^4 requires epsilon ~ 0.326472. This erratum highlights whether an enterprise verification architecture can isolate interface failure, task competence, policy admissibility, and control integrity while keeping claims auditable.   BoundaryBench v0.1 established the baseline: Qwen3-4B-Instruct-2507 completed 768 frozen executions, but 0% cleared the contract (369 failed parsing, 399 failed validation), limiting downstream selectivity metrics. This companion study evaluates three successive improvements.   First, BeTaL-GBI v0.2 applies Benchmark Tuning with an LLM-in-the-loop over 2,218,750,380 grid points, separating format admission from condi
    
[^45]: 可组合的构建模块，用于弹性异步代码

    Composable Building Blocks for Resilient Asynchronous Code

    [https://arxiv.org/abs/2608.21489](https://arxiv.org/abs/2608.21489)

    本文提出了一种统一的、可组合的高阶组合子框架，通过嵌套表达式实现异步代码的弹性策略（如超时、重试、限流等），无需修改业务逻辑，并兼容JavaScript的Promise和异步迭代两种异步形态。

    

    arXiv:2608.21489v1 公告类型：新 摘要：对网络服务、数据库或语言模型的异步调用必须应对瞬时错误、缓慢或缺失的响应、限流和原子性违规。我们展示了高阶组合子如何统一解决此类问题，包括超时、重试、速率限制、缓存、可重入锁和取消。每个组合子将一个异步函数映射到另一个相同类型的函数，因此它们共享统一的“形状”，并通过嵌套组合成一个表达式，该表达式实现程序的整个弹性和并发策略，而业务逻辑保持不变。该设计同时覆盖JavaScript的两种原生异步形状——返回Promise的函数和返回异步可迭代对象的函数，并使用统一的关注点词汇。现有解决方案散布于生态系统中，但形式各异的库难以组合。我们提供了案例研究，其中使用组合子通过添加缺失的弹性功能来强化实际包。

    arXiv:2608.21489v1 Announce Type: new  Abstract: Asynchronous calls to a network service, database, or language model must cope with transient errors, slow or missing responses, throttling, and atomicity violations. We show how higher-order combinators solve such problems uniformly, including timeouts, retries, rate limiting, caching, reentrant locking, and cancellation. Every combinator maps an async function to another of the same type, so they share a uniform \emph{shape} and compose by nesting into one expression that implements a program's whole resilience and concurrency policy, leaving its business logic untouched. The same design spans both of JavaScript's native async shapes, promise-returning and async-iterable-returning functions, with one vocabulary of concerns. Solutions exist across the ecosystem but are scattered over differently shaped libraries that are hard to combine. We present case studies where the combinators are used to harden real packages by adding missing res
    
[^46]: SLICE：契约执行的规范级隔离

    SLICE: Specification-Level Isolation of Contract Enforcement

    [https://arxiv.org/abs/2608.21483](https://arxiv.org/abs/2608.21483)

    SLICE是一种新型代码生成框架，通过分离输入契约和功能需求的生成阶段，确保生成的代码同时满足输入条件和计算要求，解决了契约执行中的完整性和严格性平衡问题。

    

    arXiv:2608.21483v1 公告类型：新公告。摘要：编程问题通常同时指定函数应执行的计算以及其输入必须满足的条件。大型语言模型被广泛用于从这些问题规范生成代码，生成的函数必须实现所需计算，同时执行指定的输入条件。这些指定的输入条件共同构成一个输入契约。执行此契约很困难：不完整的执行会接受本应拒绝的输入，而过度严格的执行会拒绝本应接受的输入。现有的代码生成方法未能提供一个生成过程，能够同时识别输入契约和功能需求，并生成满足两者的代码。因此，我们引入了SLICE，一个生成框架，它识别这两种需求，并通过分离的生成阶段来处理它们。SLICE包含三个阶段：（i）Gra

    arXiv:2608.21483v1 Announce Type: new  Abstract: Programming problems commonly specify both the computation a function should perform and the conditions that its inputs must satisfy. Large language models are widely used to generate code from these problem specifications, and the generated function must implement the required computation while enforcing the stated input conditions. The stated input conditions collectively form an input contract. Enforcing this contract is difficult: incomplete enforcement accepts inputs that should be rejected, whereas overly restrictive enforcement rejects inputs that should be accepted. Existing code generation methods do not provide a generation process that identifies both the input contract and the functional requirements and generates code that satisfies them jointly. We therefore introduce SLICE, a generation framework that identifies both requirements and addresses them through separate generation stages. SLICE consists of three stages: (i) Gra
    
[^47]: 从主观判断到可审计标准：协议引导的网站冗余AI审计

    From Subjective Judgments to Auditable Standards:Protocol-Guided AI Auditing of Website Redundancy

    [https://arxiv.org/abs/2608.21476](https://arxiv.org/abs/2608.21476)

    本文提出CORA协议引导的网站冗余审计方法，通过分离测量重复负荷、正常使用税和恢复储备，实现可审计的AI审计流程，并证明其优于传统标量基线。

    

    网站冗余没有单一固定的含义。相同的重复元素可能在一个任务中造成干扰，而在另一个任务中提供备份。我们引入了CORA（反事实、可观察冗余审计），它分别测量重复负荷、正常使用税和故障域恢复储备。每次运行保留截图、稳定元素身份和任务轨迹。一个版本化的视觉语言模型提出注释。类型化验证和发布检查随后确定校准维度是否可以报告；失败或格式错误的输出保留在固定分母中。在一个透明的机械测试平台上，因子化的CORA表示将储备与正常使用税分开，并比标量负荷基线更准确地预测扰动后的成功率。模型研究随后表明，为什么可重复性不足：两个小型局部视觉语言模型产生了重复输出，但两个仪器都未满足所有相关标准。

    arXiv:2608.21476v1 Announce Type: new  Abstract: Website redundancy does not have a single fixed meaning. The same repeated element may distract during one task and provide backup during another. We introduce CORA (Counterfactual, Observable Redundancy Audit), which measures repetition load, normal-use tax, and failure-domain recovery reserve separately. Each run retains screenshots, stable element identities, and task traces. A versioned vision-language model proposes the annotations. Typed validation and release checks then determine whether a calibrated dimension can be reported; failed or malformed outputs stay in the fixed denominator. On a transparent mechanistic testbed, the factorized CORA representation separated reserve from normal-use tax and predicted perturbed success more accurately than scalar-load baselines. The model studies then showed why repeatability is not enough: two small local vision-language models produced recurring outputs, but neither instrument met all rel
    
[^48]: 面向制造知识图谱的可组合信任基础设施：跨系统溯源、时间推理与决策可追踪性

    Composable Trust Infrastructure for Manufacturing Knowledge Graphs: Cross-System Provenance, Temporal Reasoning, and Decision Traceability

    [https://arxiv.org/abs/2608.21418](https://arxiv.org/abs/2608.21418)

    本文提出一种可组合信任基础设施，通过共享标识符整合四种信任能力，产生单独能力无法提供的涌现信任属性，并实验验证了每种能力对组合查询的关键性。

    

    arXiv:2608.21418v1 公告类型：新 摘要：整合异构工业系统数据的制造知识图谱面临信任赤字：消费者无法确定查询数据是否有效、在做出决策时数据是否有效、数据源自何处，或数据如何被处理。我们认为，四种信任能力——SHACL验证、PROV-O溯源、领域感知的双时态版本管理以及图原生决策对象——通过共享关联标识符进行组合，产生单独任何能力都无法提供的涌现信任属性。我们提出一种可组合的信任基础设施，将这四种能力整合到统一的RDF架构中。这些能力通过共享实体URI、摄取活动标识符和时间关联键进行组合，支持跨越所有四个维度的复合查询。一项实验性消融研究确认，移除任何单一能力会导致六个组合查询中的恰好三个失败。

    arXiv:2608.21418v1 Announce Type: new  Abstract: Manufacturing knowledge graphs that integrate data from heterogeneous industrial systems face a trust deficit: consumers cannot determine whether queried data is valid, whether it was valid when a decision was made, where it originated, or how it was acted upon. We argue that four trust capabilities -- SHACL validation, PROV-O provenance, domain-aware bi-temporal versioning, and graph-native decision objects -- compose through shared correlation identifiers to produce emergent trust properties that no single capability delivers alone. We present a composable trust infrastructure that integrates these four capabilities into a unified RDF architecture. Capabilities compose through shared entity URIs, ingestion activity identifiers, and temporal correlation keys, enabling compound queries spanning all four dimensions. An experimental ablation confirms that removing any single capability causes exactly three of six composition queries to fai
    
[^49]: 面向CPS规约的信号时域与频域表示研究

    On the Time and Frequency Domain Representations of Signals for CPS Specification

    [https://arxiv.org/abs/2608.21167](https://arxiv.org/abs/2608.21167)

    本文提出了一种新的规约语言S2TL，利用时频表示来增强CPS需求规约的表达能力，特别适用于描述信号形状和动态行为。

    

    规约语言对于网络物理系统（CPS）的验证与确认至关重要。大多数最先进的规约语言使用信号的时域表示，这并不总是适合描述信号形状和动态行为。相反，控制和机器人等领域使用频域表示来表征这些行为。时频表示结合了两个域的能力。我们研究了使用时频表示来规定CPS需求的可能性。我们分析了现有的CPS需求分类法，以识别哪些需求类别可以从时频表示中受益。我们得出了使用时频表示的规约语言的设计要求，并提出了信号-频谱时间逻辑（S2TL），这是一种能够对频率区间和频率分量之间的关系进行断言的规约语言。我们进行了操作化实现。

    arXiv:2608.21167v1 Announce Type: new  Abstract: Specification languages are instrumental to the Verification \& Validation of Cyber-Physical Systems (CPSs). Most state-of-the-art specification languages use the time-domain representation of signals, which is not always suitable for describing signal shapes and dynamic behaviours. Instead, fields like control and robotics use the frequency-domain representation to characterise these behaviours. Time-frequency representations combine the capabilities of both domains. We investigate the use of time-frequency representations to specify CPS requirements. We analyse existing taxonomies of CPS requirements to identify which requirement classes can benefit from time-frequency representations. We derive the desiderata for a specification language that uses time-frequency representations and propose Signal-Spectrum Temporal Logic (S2TL), a language enabling assertions over frequency intervals and relations between frequency components. We opera
    
[^50]: 将AI代理锚定在合同中：规格驱动测试生成的实证评估

    Grounding AI Agents in Contracts: An Empirical Evaluation of Spec-Driven Test Generation

    [https://arxiv.org/abs/2608.17177](https://arxiv.org/abs/2608.17177)

    提出规格驱动测试生成方法，通过让LLM代理先显式推理和记录代码合同（前置/后置条件及未定义行为）作为认知脚手架，显著提升了生产环境中的缺陷检测率和分支覆盖率。

    

    arXiv:2608.17177v1 公告类型：新 摘要：基于LLM的代理越来越多地用于编码任务，在这些任务中，它们已超越许多经典方法，并扩展到仓库级任务，如测试生成。然而，当直接提示生成测试时，这些代理可能无法推理代码及其底层合同，从而遗漏影响测试质量的边缘案例和行为边界。为解决这一限制，我们提出了规格驱动测试生成方法，即指示代理首先推理并显式记录代码的前置条件、后置条件和未定义行为。这种中间半形式化规格作为认知脚手架，指导后续的测试生成。我们在Google生产缺陷上的评估显示，与直接提示相比，规格驱动代理在缺陷检测率上提高了9.8个百分点（p = 0.0352），在分支覆盖率上提高了2.5个百分点（p = 0.0034）。

    arXiv:2608.17177v1 Announce Type: new  Abstract: LLM-based agents are increasingly used for coding tasks, where they have outperformed many classical approaches and scaled to repository-level tasks, such as test generation. However, when directly prompted to generate tests, these agents can fail to reason about the code and its underlying contracts, thereby missing edge cases and behavioral boundaries that affect test quality. To address this limitation, we propose Spec-Driven Test Generation, where we instruct an agent to first reason about -- and explicitly document -- code pre-conditions, post-conditions, and undefined behaviors. This intermediate semi-formal specification acts as a cognitive scaffold to guide subsequent test generation. Our evaluation on production bugs from Google shows that the spec-driven agent can deliver a 9.8 percentage points ($p = 0.0352$) improvement in bug detection rate and a 2.5 percentage point ($p = 0.0034$) improvement in branch coverage, compared to
    
[^51]: 谁将成为下一位高级工程师？生成式人工智能如何侵蚀软件工程中的职业发展路径

    Who Will Become the Next Senior? How Generative AI Erodes the Development Pathway in Software Engineering

    [https://arxiv.org/abs/2607.17067](https://arxiv.org/abs/2607.17067)

    本研究通过访谈揭示GenAI通过“吸收”模式将初级工作转给高级-AI流程，导致初级工程师失去成长所需的挑战，并因集体正常化而固化这一损失。

    

    摘要：arXiv:2607.17067v2 公告类型：替换-交叉 摘要：生成式人工智能（GenAI）正在重塑软件工程领域，引发了对初级工程师成长为高级工程师的发展路径如何被侵蚀的担忧。尽管宏观统计显示初级职位招聘减少，且控制性研究证明了AI对个人任务绩效的影响，但GenAI在真实组织和教育环境中如何重塑早期职业发展的机制尚未得到深入探讨。通过对韩国处于进入软件工程门槛的初级工程师和高级软件工程师进行14次半结构化访谈，并采用反思性主题分析，我们揭示了一个基础模式——“吸收”——即GenAI将入门级工作重新导向为高级-AI工作流程，并产生三个后果：（1）初级工程师失去了曾经培养专业技能的“生产性挣扎”；（2）这种损失通过集体正常化在结构上得到复制；

    arXiv:2607.17067v2 Announce Type: replace-cross  Abstract: Generative AI (GenAI) is reshaping software engineering, raising concerns about how the development pathway through which juniors become seniors is being eroded. While macro statistics show a decline in junior hiring and controlled studies demonstrate the effects of AI on individual task performance, the mechanisms through which GenAI reshapes early-career development in real organizational and educational contexts have not been thoroughly examined. Through 14 semi-structured interviews with juniors at the threshold of entering software engineering and senior software engineers in South Korea, analyzed using Reflexive Thematic Analysis, we reveal a foundational pattern of Absorption -- GenAI redirects entry-level work into senior-AI workflows -- and three consequences: (1) juniors losing the productive struggle through which expertise once developed; (2) the structural reproduction of this loss through collective normalization 
    
[^52]: 基于证伪的LLM生成优化模型验证：健全测试组及其检测极限

    Falsification-Based Verification of LLM-Generated Optimization Models: Sound Test Batteries and Their Detection Limits

    [https://arxiv.org/abs/2607.16646](https://arxiv.org/abs/2607.16646)

    本文提出了一种基于证伪的验证方法，通过对偶性和灵敏度分析生成健全的求解器测试组，以零误报率检测LLM生成的优化模型中的错误，并明确了其检测极限。

    

    大型语言模型现在能将自然语言描述的决策问题转化为可直接求解的优化模型，但它们的失败往往是无声的。生成的模型常常能运行，却编码了错误的问题，而标准评估方法是将最优值与标记答案进行比较，这些答案在实际部署中并不可得。如何在没有任何参考的情况下认证这样的模型，是本文要解决的问题。我们开发了基于证伪的验证方法。问题描述中的每个数值量在文本中都有其明确角色，如容量、需求或单位成本，任何正确的模型都必须按照所述角色对这些量的变化做出响应。通过对偶性和灵敏度分析，我们推导出一组基于求解器的测试，这些测试单独都是健全的，因此违反测试即证明模型有缺陷，且误报率在设计上为零。我们刻画了这类测试无法检测的错误类型。

    arXiv:2607.16646v2 Announce Type: replace-cross  Abstract: Large language models now translate natural-language descriptions of decision problems into solver-ready optimization models, and they fail silently. A generated model often runs and still encodes the wrong problem, while standard evaluation compares optimal values against labeled answers that deployment does not provide. How to certify such a model without any reference is the question this paper addresses. We develop falsification-based verification. Every numeric quantity in a problem description plays a role that the text itself states, such as a capacity, a requirement, or a unit cost, and any correct model must respond to changes in these quantities as the stated roles dictate. From duality and sensitivity analysis we derive a battery of solver-based tests that are individually sound, so a violation certifies a faulty model and the false-positive rate is zero by design. We characterize the errors that no test of this kind
    
[^53]: 探索多模态大语言模型在代码生成中程序流程图的潜力

    Exploring the Potential of Program Flowcharts on Code Generation Using Multimodal LLMs

    [https://arxiv.org/abs/2607.09146](https://arxiv.org/abs/2607.09146)

    本研究首次系统探索了将程序流程图作为视觉输入与问题描述结合，可显著提升多模态大语言模型（如GPT-4o）的代码生成性能。

    

    摘要：近年来，大型语言模型（LLMs）取得了显著进展，催生了能够处理图像和音频等多种输入的多模态LLMs。先前研究表明，向多模态LLMs提供文本和视觉信息的组合可以提升自动代码生成能力。在软件开发中，流程图等图表被广泛用于促进代码理解等任务。虽然现有研究探讨了视觉输入对LLMs的影响以及软件图表的使用，但提供流程图对多模态LLM性能的潜在影响仍未得到充分探索。在本研究中，我们从AtCoder问题的示例解决方案代码生成了流程图，并将这些视觉辅助信息与问题描述一起提供给GPT-4o进行代码生成。我们的研究结果表明，将流程图与问题描述结合使用能带来性能提升。

    arXiv:2607.09146v2 Announce Type: replace  Abstract: In recent years, Large Language Models (LLMs) have made significant strides, leading to the emergence of multimodal LLMs capable of processing diverse inputs such as images and audio. Previous research indicates that the supply of multimodal LLMs with combined textual and visual information improves the automatic code generation capabilities. In software development, diagrams such as flowcharts are widely employed to facilitate tasks like code comprehension. While existing studies investigated the impact of visual inputs on LLMs and the usage of software diagrams, the potential influence of providing flowcharts on multimodal LLM performance remains underexplored. In this study, we generated flowcharts from example solution code for AtCoder problems and provided these visual aids alongside problem statements to GPT-4o for code generation. Our findings demonstrate that integrating flowcharts with problem statements yields performance i
    
[^54]: SABER：对状态化项目工作区中大语言模型编码代理的操作安全性进行基准测试

    SABER: Benchmarking Operational Safety of LLM Coding Agents in Stateful Project Workspaces

    [https://arxiv.org/abs/2606.01317](https://arxiv.org/abs/2606.01317)

    SABER提出了一个环境感知的操作安全性基准，通过状态化项目工作区中的最终状态评估编码代理的安全表现，并发现当前模型在现实环境中存在超过54%的高安全违规率。

    

    大型语言模型越来越多地被部署为编码代理，其安全性从单个响应转移到动作序列。然而，现有基准主要评估模型是否拒绝不安全提示，而忽略了对状态化工作区的影响。我们提出了SABER，一个环境感知的操作安全性基准，它将模型置于真实的代理风格项目中，并根据一系列动作后的最终环境状态评估安全性。除了二元的违规报告外，SABER还按原因对违规进行分类，从而能够分析模型特定的安全概况。我们的评估显示，即使性能最好的模型也有超过54%的有害安全违规率（HSR），这表明当前的校准对于现实项目环境仍然不足。SABER进一步揭示了不同模型之间的不同安全概况。我们的基准可在https://github.com/sssr-lab/s公开获取。

    arXiv:2606.01317v2 Announce Type: replace  Abstract: Large language models are increasingly deployed as coding agents, shifting safety from individual responses to action sequences. Existing benchmarks, however, primarily assess whether models refuse unsafe prompts, leaving impacts on stateful workspaces largely unexamined. We present SABER, a benchmark for environment-aware operational safety that places models in realistic agent-style projects and evaluates safety from the final environment state after a sequence of actions. Beyond binary safety-violation reports, SABER categorizes violations by cause, enabling analysis of model-specific safety profiles. Our evaluations show that even the best-performing model has more than a 54% harmful safety-violation rate (HSR), suggesting that current alignment remains insufficient for realistic project environments. SABER further reveals distinct safety profiles across models. Our benchmark is publicly available at https://github.com/sssr-lab/s
    
[^55]: 基于大型语言模型的Java低层集成测试生成

    LLM-based Low-Level Integration Test Generation for Java

    [https://arxiv.org/abs/2605.26851](https://arxiv.org/abs/2605.26851)

    IntTestGen是一种基于LLM的Java低层集成测试生成方法，通过挖掘依赖模式并应用多级约束修复，有效解决了LLM缺乏项目知识和违反约束的问题。

    

    arXiv:2605.26851v2 公告类型：替换 摘要：大型语言模型（LLMs）在自动化测试生成方面显示出潜力，但大多数方法针对的是具有模拟依赖关系的单元测试。低层集成测试则使用类及其真实的项目内依赖关系来执行测试，从而暴露涉及对象构造、API调用序列和组件交互的故障。生成此类测试具有挑战性，因为LLMs可能缺乏项目特定知识（不知道）或违反提供的约束（不遵循）。我们提出了IntTestGen，一种基于LLM的方法，结合了上下文增强生成与约束强制修复。它从项目代码中挖掘依赖使用模式以指导测试生成，然后在修复过程中应用符号、协议和迭代级别的约束，使用ClassIndex、马尔可夫类型状态模型和经验记忆。我们在Defe上评估了IntTestGen与最先进的基于LLM的基线PANTA和基于搜索的基线EvoSuite。

    arXiv:2605.26851v2 Announce Type: replace  Abstract: Large language models (LLMs) show promise for automated test generation, but most approaches target unit tests with mocked dependencies. Low-level integration testing instead exercises a class with its real, in-project dependencies, exposing faults involving object construction, API call sequences, and component interactions. Generating such tests is challenging because LLMs may lack project-specific knowledge (not knowing) or violate provided constraints (not following).   We present IntTestGen, an LLM-based approach that combines context-enriched generation with constraint-enforced fixing. It mines dependency usage patterns from project code to guide test generation, then applies symbol-, protocol-, and iteration-level constraints during repair using a ClassIndex, a Markov typestate model, and experience memory.   We evaluate IntTestGen against the state-of-the-art LLM-based baseline PANTA and search-based baseline EvoSuite on Defe
    
[^56]: 通过聊天编程：对11,579个真实世界AI辅助IDE会话的大规模行为分析

    Programming by Chat: A Large-Scale Behavioral Analysis of 11,579 Real-World AI-Assisted IDE Sessions

    [https://arxiv.org/abs/2604.00436](https://arxiv.org/abs/2604.00436)

    这项研究首次大规模分析了IDE原生环境中11,579个真实世界AI辅助编程会话，揭示了对话式编程作为渐进式规格说明的三个关键转变。

    

    arXiv:2604.00436v2 公告类型：替换 摘要：集成在IDE中的AI编程助手，能够在开发者的工作代码库中以对话方式运行，并访问项目上下文和多文件编辑功能，正迅速改变软件开发方式。然而，对这一转变的实证研究仍然有限：现有研究大多依赖于小规模、受控环境，或分析通用聊天机器人，而非感知代码库的IDE工作流程。我们提出了据我们所知首个针对IDE原生环境中真实世界对话式编程的大规模研究，分析了来自11,579个聊天会话的74,998条开发者消息，涵盖1,300个代码库和899名使用Cursor和GitHub Copilot的开发者。这些聊天作为常规开发的一部分被提交到公共代码库，捕获了真实场景中的行为。我们的发现揭示了编程工作组织方式的三个转变：对话式编程以渐进式规格说明运作，开发者...

    arXiv:2604.00436v2 Announce Type: replace  Abstract: IDE-integrated AI coding assistants, which operate conversationally within developers' working codebases with access to project context and multi-file editing, are rapidly reshaping software development. However, empirical investigation of this shift remains limited: existing studies largely rely on small-scale, controlled settings or analyze general-purpose chatbots rather than codebase-aware IDE workflows. We present, to the best of our knowledge, the first large-scale study of real-world conversational programming in IDE-native settings, analyzing 74,998 developer messages from 11,579 chat sessions across 1,300 repositories and 899 developers using Cursor and GitHub Copilot. These chats were committed to public repositories as part of routine development, capturing in-the-wild behavior. Our findings reveal three shifts in how programming work is organized: conversational programming operates as progressive specification, with deve
    
[^57]: CRANE-LLM：运行时增强的大语言模型用于机器学习笔记本中的崩溃预测与诊断

    CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks

    [https://arxiv.org/abs/2602.18537](https://arxiv.org/abs/2602.18537)

    CRANE-LLM通过向大语言模型提供从笔记本内核提取的运行时信息，在单元格执行前实现崩溃预测与诊断，性能提升7-10个百分点。

    

    Jupyter笔记本已成为早期机器学习（ML）开发的热门工具，支持交互式和迭代式实验。然而，ML笔记本容易出现各种错误，其中崩溃最具破坏性。尽管这些崩溃在实际中非常重要，但针对ML笔记本的崩溃预测与诊断研究仍十分匮乏。我们提出了CRANE-LLM，一种运行时增强的源代码分析方法，它为大语言模型（LLMs）提供从笔记本内核中提取的结构化运行时信息，结合源代码，在目标单元格执行前预测并诊断其崩溃。我们在JunoBench上评估了CRANE-LLM，该基准包含111个Kaggle ML笔记本，涵盖多种ML库和崩溃类型。在三种最先进的大语言模型（Gemini、Qwen和GPT-5）上，我们的结果表明，运行时信息显著提升了崩溃预测与诊断性能，提高了7-10个百分点。

    arXiv:2602.18537v2 Announce Type: replace  Abstract: Jupyter notebooks have become popular for early machine learning (ML) development, enabling interactive and iterative experimentation. However, ML notebooks are prone to bugs, among which crashes are the most disruptive. Despite their practical importance, crash prediction and diagnosis in ML notebooks remain largely unexplored. We present CRANE-LLM, a runtime-augmented source code analysis approach that provides large language models (LLMs) with structured runtime information extracted from the notebook kernel, together with source code, to predict and diagnose crashes in a target cell before executing it. We evaluate CRANE-LLM on JunoBench, a benchmark of 111 Kaggle ML notebooks containing crashes across multiple ML libraries and crash types. Across three state-of-the-art LLMs (Gemini, Qwen, and GPT-5), our results show that runtime information significantly improves crash prediction and diagnosis performance by 7-10 percentage poi
    
[^58]: 审判中的“氛围编程”：一致通过的大语言模型陪审团的操作特性

    Vibe Coding on Trial: Operating Characteristics of Unanimous LLM Juries

    [https://arxiv.org/abs/2602.18492](https://arxiv.org/abs/2602.18492)

    本研究提出并评估了一种基于大语言模型陪审团的一致通过机制，用于自动审查文本到SQL任务中的候选代码，在安全优先场景下有效平衡了接受准确性与人工干预成本。

    

    大语言模型（LLMs）在编程方面已经足够优秀，开发者可以用自然语言描述意图，让工具生成初版代码，这种工作流程日益集成到GitHub Copilot、Cursor和Replit等工具中。目前缺少的是一种可靠的方法，用来判断哪些模型生成的查询可以安全接受，而不必事事都交给人工审核。我们研究应用LLM陪审团来执行这一审查步骤。我们首先在82个MySQL文本到SQL任务上对15个开源模型进行了基准测试，采用基于执行验证的协议，以清晰确定哪些模型表现强劲。然后，从六个最佳模型中构建了规模为1到6的一致通过委员会，这些委员会查看提示、数据库模式和候选SQL，并且仅当所有成员都认为正确时才接受该查询。这一规则符合安全优先的部署场景，其中错误接受比错误拒绝代价更高。我们测量了真正率、假正率和Youden J指数，并进一步分析了……

    arXiv:2602.18492v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are now good enough at coding that developers can describe intent in plain language and let the tool produce the first code draft, a workflow increasingly built into tools like GitHub Copilot, Cursor, and Replit. What is missing is a reliable way to tell which model written queries are safe to accept without sending everything to a human. We study the application of an LLM jury to run this review step. We first benchmark 15 open models on 82 MySQL text to SQL tasks using an execution grounded protocol to get a clean baseline of which models are strong. From the six best models we build unanimous committees of sizes 1 through 6 that see the prompt, schema, and candidate SQL and accept it only when every member says it is correct. This rule matches safety first deployments where false accepts are more costly than false rejects. We measure true positive rate, false positive rate and Youden J and we als
    
[^59]: 代码行间的解读：论自我承认的技术债务在安全分析中的应用

    Reading Between the Code Lines: On the Use of Self-Admitted Technical Debt for Security Analysis

    [https://arxiv.org/abs/2602.03470](https://arxiv.org/abs/2602.03470)

    本研究首次系统性地探讨了自我承认的技术债务（SATD）中编码的安全信息如何为静态分析工具（SATs）提供互补性安全洞察，从而增强漏洞检测的覆盖率和准确性。

    

    arXiv:2602.03470v2 公告类型：交叉替换 摘要：静态分析工具（SATs）是安全工程活动的核心，因为它们能够在无需执行代码的情况下早期识别代码弱点。然而，其有效性常常受到高误报率和漏洞类别覆盖不全的限制。同时，开发人员经常在软件工件（如代码注释）中记录与安全相关的捷径和妥协，作为自我承认的技术债务（SATD）。尽管先前的研究已认识到SATD是安全信息的丰富来源，但在SAT辅助的安全分析中是否以及如何利用它仍不清楚。目标：本研究探讨SATD中编码的安全相关信息是否以及如何为SATs提供补充性的安全洞察。方法：我们采用混合方法，包括（i）使用三个SATs分析一个手动策展的、带有SATD注释的漏洞数据集，以及（ii）（此处原文未完整，但根据上下文推断为后续分析步骤）。

    arXiv:2602.03470v2 Announce Type: replace-cross  Abstract: Static Analysis Tools (SATs) are central to security engineering activities, as they enable early identification of code weaknesses without requiring execution. However, their effectiveness is often limited by high false-positive rates and incomplete coverage of vulnerability classes. At the same time, developers frequently document security-related shortcuts and compromises as Self-Admitted Technical Debt (SATD) in software artifacts, such as code comments. While prior work has recognized SATD as a rich source of security information, it remains unclear whether -and in what ways- it is utilized during SAT-aided security analysis. OBJECTIVE: This work explores whether and how the security-related information encoded in SATD provides complementary security insights to SATs. METHOD: We followed a mixed-methods approach comprising (i) the analysis of a manually curated, SATD-annotated vulnerability dataset using three SATs and (ii
    
[^60]: 智能体程序修复中缺陷复现测试的动态协同生成

    Dynamic Cogeneration of Bug Reproduction Test in Agentic Program Repair

    [https://arxiv.org/abs/2601.19066](https://arxiv.org/abs/2601.19066)

    本文提出并评估了在智能体程序修复中动态协同生成缺陷复现测试与修复补丁的策略，并开发了考虑测试变更的补丁选择器，以提升开发者对AI生成补丁的信心。

    

    缺陷复现测试（BRTs）已被许多自动程序修复（APR）系统使用，主要用于验证修复和辅助修复生成。在实践中，当开发者提交补丁时，他们通常会在修复的同时实现BRT。我们在部署智能体APR中的经验表明，开发者希望AI生成的补丁中包含BRT以增强其信心。然而，典型的APR系统倾向于分别生成BRT和修复，并专注于在最终补丁中仅生成修复。在本文中，我们研究了智能体APR在协同生成背景下的应用，其中APR智能体被指示在同一补丁中同时生成修复和BRT。我们评估了在Google的120个人类报告的错误上不同协同生成策略的有效性，并通过其对APR智能体行为的影响来表征不同协同生成策略。我们开发并评估了考虑测试变更的补丁选择器，以选择补丁。

    arXiv:2601.19066v3 Announce Type: replace-cross  Abstract: Bug Reproduction Tests (BRTs) have been used in many Automated Program Repair (APR) systems, primarily for validating fixes and aiding fix generation. In practice, when developers submit a patch, they often implement the BRT alongside the fix. Our experience deploying agentic APR reveals that developers desire a BRT within AI-generated patches to increase their confidence. However, canonical APR systems tend to generate BRTs and fixes separately, and focus on producing only the fix in the final patch. In this paper, we study agentic APR in the context of cogeneration, where the APR agent is instructed to generate both a fix and a BRT in the same patch. We evaluate the effectiveness of different cogeneration strategies on 120 human-reported bugs at Google and characterize different cogeneration strategies by their influence on APR agent behavior. We develop and evaluate patch selectors that account for test change to select patc
    
[^61]: 基于Stack Overflow答案编辑的Java代码改进实证研究

    An Empirical Study of Java Code Improvements Based on Stack Overflow Answer Edits

    [https://arxiv.org/abs/2511.05813](https://arxiv.org/abs/2511.05813)

    本研究通过分析Stack Overflow上Java答案的编辑历史，利用改进的代码克隆搜索工具，将这些编辑应用于开源项目中的代码改进，揭示了众包知识对提升代码质量的潜力。

    

    arXiv:2511.05813v2 公告类型：替换 摘要：次优代码在软件系统中普遍存在。开发人员常因技术知识缺口、经验不足、时间压力、管理决策或个人因素而编写低质量代码。一旦集成，这些次优代码的积累会导致显著的维护成本和技术债务。开发人员经常咨询外部知识库，如API文档和像Stack Overflow（SO）这样的问答网站，以辅助编程任务。SO的众包协作性质创建了庞大的编程知识库，其社区策划内容不断演变，新答案发布或现有答案被编辑。在本文中，我们呈现了一项关于SO Java答案编辑及其在开源项目代码改进中应用的实证研究。我们使用修改后的代码克隆搜索工具来分析带版本历史的SO代码片段，并将其应用于开源项目。

    arXiv:2511.05813v2 Announce Type: replace  Abstract: Suboptimal code is prevalent in software systems. Developers often write low-quality code due to factors like technical knowledge gaps, insufficient experience, time pressure, management decisions, or personal factors. Once integrated, the accumulation of this suboptimal code leads to significant maintenance costs and technical debt.   Developers frequently consult external knowledge bases, such as API documentation and Q&A websites like Stack Overflow (SO), to aid their programming tasks. SO's crowdsourced, collaborative nature has created a vast repository of programming knowledge. Its community-curated content is constantly evolving, with new answers posted or existing ones edited.   In this paper, we present an empirical study of SO Java answer edits and their application to improving code in open-source projects. We use a modified code clone search tool to analyze SO code snippets with version history and apply it to open-source
    
[^62]: 基于大语言模型的成本效益型需求变更影响分析

    LLM-Driven Cost-Effective Requirements Change Impact Analysis

    [https://arxiv.org/abs/2511.00262](https://arxiv.org/abs/2511.00262)

    提出了一种基于大语言模型的成本效益型需求变更影响分析方法ProReFiCIA，在工业数据集上达到85.7%召回率，显著减少人工错误和成本。

    

    需求在软件开发生命周期中本质上容易发生变化。在需求工程师有限的预算内，手动识别这些变更对其他需求的影响容易出错且耗费精力，尤其是在受监管的领域。这可能导致被忽略的受影响需求，如果管理不当，会在下游任务中引发严重问题。受大型语言模型（LLM）在多个领域日益增长的潜力启发，我们提出了ProReFiCIA，一种由LLM驱动的方法，用于在变更发生时自动识别受影响的需求。我们使用多种LLM和针对此任务定制的提示变体对ProReFiCIA进行了广泛评估。使用最佳的LLM-提示组合，ProReFiCIA在未见过的工业数据集上实现了85.7%的召回率，证明了其在识别受影响需求方面的有效性。此外，应用Pr的成本也较低。

    arXiv:2511.00262v5 Announce Type: replace  Abstract: Requirements are inherently subject to change throughout the software development lifecycle. Within the limited budget available to requirements engineers, manually identifying the impact of such changes on other requirements is error-prone and effort-intensive, especially in regulated domains. This can lead to overlooked impacted requirements, which, if not properly managed, can cause serious issues in downstream tasks. Inspired by the growing potential of large language models (LLMs) across diverse domains, we propose ProReFiCIA, an LLM-driven approach to automatically identify impacted requirements when changes occur. We conduct an extensive evaluation of ProReFiCIA using several LLMs and prompt variants tailored to this task. Using the best LLM-prompt combination, ProReFiCIA achieves 85.7% recall on an unseen industrial dataset, demonstrating its effectiveness in identifying impacted requirements. Further, the cost of applying Pr
    
[^63]: NaturalEdit：通过自适应自然语言表示的直接交互进行代码修改

    NaturalEdit: Code Modification through Direct Interaction with Adaptive Natural Language Representation

    [https://arxiv.org/abs/2510.04494](https://arxiv.org/abs/2510.04494)

    NaturalEdit通过引入自适应、多面代码摘要和交互式映射机制，使代码修改过程更直观，降低了认知负担，并实现了自然语言表示与源代码之间的紧密连接。

    

    代码修改要求开发者理解代码、规划更改、表达意图并验证结果，这使得任务在认知上非常繁重。虽然自然语言代码摘要为这一过程提供了一种有前景的外部表示，但现有方法仍存在局限性。基于探索性数据分析的系统局限于狭窄领域，而通用系统则强制使用固定的自然语言表示，并假设开发者能够直接将模糊意图转化为精确的文本编辑。我们提出了NaturalEdit，它将代码摘要视为与源代码紧密相连的交互式表示。基于符号的认知维度理论，NaturalEdit引入了三个关键特性：（1）具有灵活抽象梯度的自适应、多面代码摘要；（2）摘要与代码之间的交互式映射机制，确保紧密且结构稳定的映射接近性；（3）在...

    arXiv:2510.04494v3 Announce Type: replace-cross  Abstract: Code modification requires developers to comprehend code, plan changes, articulate intent, and validate outcomes, making it cognitively demanding. While natural language (NL) code summaries offer a promising external representation of this process, existing approaches remain limited. Systems grounded in exploratory data analysis are restricted to narrow domains, while general-purpose systems enforce fixed NL representations and assume that developers can directly translate vague intent into precise textual edits. We present NaturalEdit, which treats code summaries as interactive representations tightly linked to source code. Grounded in the Cognitive Dimensions of Notations, NaturalEdit introduces three key features: (1) adaptive, multi-faceted code summaries with a flexible Abstraction Gradient; (2) interactive mapping mechanisms between summaries and code that ensure tight, structurally stable Closeness of Mapping; and (3) in
    
[^64]: 基于现场-场外图进行实体表示学习以应用于Pinterest广告

    Entity Representation Learning Through Onsite-Offsite Graph for Pinterest Ads

    [https://arxiv.org/abs/2508.02609](https://arxiv.org/abs/2508.02609)

    本文提出了一种基于用户现场和场外活动的大规模异构图构建方法，并引入了带锚点的TransR模型（TransRA）来高效集成图嵌入，从而提升Pinterest广告排序模型的性能。

    

    arXiv:2508.02609v3 公告类型：交叉替换 摘要：图神经网络（GNN）已广泛应用于工业推荐系统，如GraphSage、TwHIM、LiGNN等模型所示。在这些工作中，图是基于用户在平台上的活动构建的，并开发了各种图模型来有效学习节点嵌入。除了用户的现场活动外，他们的场外转化对于广告模型捕捉购物兴趣至关重要。为了更好地利用场外转化数据并探索现场与场外活动之间的联系，我们基于用户的现场广告互动和选择加入的场外转化活动构建了一个大规模异构图。此外，我们引入了TransRA（带锚点的TransR），一种新颖的知识图谱嵌入（KGE）模型，以更高效地将图嵌入集成到广告排序模型中。然而，我们的广告排序模型...

    arXiv:2508.02609v3 Announce Type: replace-cross  Abstract: Graph Neural Networks (GNN) have been extensively applied to industry recommendation systems, as seen in models like GraphSage\cite{GraphSage}, TwHIM\cite{TwHIM}, LiGNN\cite{LiGNN} etc. In these works, graphs were constructed based on users' activities on the platforms, and various graph models were developed to effectively learn node embeddings. In addition to users' onsite activities, their offsite conversions are crucial for Ads models to capture their shopping interest. To better leverage offsite conversion data and explore the connection between onsite and offsite activities, we constructed a large-scale heterogeneous graph based on users' onsite ad interactions and opt-in offsite conversion activities. Furthermore, we introduced TransRA (TransR\cite{TransR} with Anchors), a novel Knowledge Graph Embedding (KGE) model, to more efficiently integrate graph embeddings into Ads ranking models. However, our Ads ranking models i
    
[^65]: LSem2Vec：一种简单而有效的两阶段源代码嵌入方法

    LSem2Vec: A Simple yet Effective Two-Stage Approach for Source Code Embedding

    [https://arxiv.org/abs/2409.14644](https://arxiv.org/abs/2409.14644)

    LSem2Vec通过两阶段方法（LLM提取语义+句子嵌入生成向量）实现无需监督训练或微调的源代码嵌入，有效处理错误信息并提升性能。

    

    摘要：大型语言模型（LLMs）的出现显著推进了软件工程中的人工智能，源代码嵌入在诸如源代码克隆检测和源代码聚类等任务中扮演着关键角色。然而，现有的源代码嵌入方法，包括基于LLMs的方法，通常依赖昂贵的监督训练或微调来进行领域适应。本文提出了LSem2Vec（LLM提取的代码语义到向量嵌入），一种简单而有效的两阶段方法，通过结合大型语言模型和句子嵌入模型来嵌入源代码。具体来说，LSem2Vec利用LLM提取源代码的语义，然后使用句子嵌入模型生成表示向量。与之前的方法相比，LSem2Vec消除了对任务特定训练或微调的需求，并有效解决了源代码中常见的错误信息问题。

    arXiv:2409.14644v4 Announce Type: replace-cross  Abstract: The advent of large language models (LLMs) has significantly advanced artificial intelligence in software engineering, with source code embeddings playing a crucial role in tasks such as source code clone detection and source code clustering. However, existing methods for source code embedding, including those based on LLMs, often rely on costly supervised training or fine-tuning for domain adaptation. This paper proposes LSem2Vec (LLM-extracted code Semantics to Vector embedding), a simple yet effective two-stage approach to embedding source code by combining large language and sentence embedding models. Specifically, LSem2Vec leverages an LLM to extract the semantics of source code, and then uses a sentence embedding model to generate representation vectors. Compared with previous approaches, LSem2Vec eliminates the need for task-specific training or fine-tuning and effectively addresses erroneous information commonly found i
    

