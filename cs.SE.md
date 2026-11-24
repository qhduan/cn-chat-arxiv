# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Mixed-Methods Study of Personal Software Creation](https://arxiv.org/abs/2611.31217) | 本研究通过分析 6,168 条网络与 Reddit 帖子并跟踪八名用户使用 AI 编程智能体构建个人软件的过程，首次大规模揭示了人们创作个人软件的行为、动机及其面临的机遇与挑战。 |
| [^2] | [Carried-Over and Agent-Native Threats in the Supply Chain of LLM Agent Skills](https://arxiv.org/abs/2611.30932) | 该论文首次在注册表规模上对 LLM 智能体技能供应链中由非故意漏洞引发的攻击面进行实证研究，对 ClawHub 全部 48,682 个技能开展静态分析、人工审查（3,428 个标记案例）与可达性测试，覆盖依赖项和第一方代码两类遗留风险，以及安装指令和凭证泄露两类智能体原生风险。 |
| [^3] | [Useful but Unwieldy: How Software Engineers Perceive the Design of Chain-of-Thought Explanations in Coding Agents](https://arxiv.org/abs/2611.30894) | 本研究通过对36名软件工程师的调查与访谈发现，编码代理展示的思维链消息是软件工程师不可或缺的可解释性工具，但其设计仍显得笨拙、不易使用。 |
| [^4] | [Interrupt-Aware Static Detection of Deadlocks in Rust Operating Systems](https://arxiv.org/abs/2611.30713) | 该论文提出了一种结合中断感知静态分析与诊断引导的智能体审计的新方法，利用轻量级源码标注和基于MIR的流敏感分析来检测Rust操作系统中由中断引发的死锁。 |
| [^5] | [ReproShift: Cross-Version Adaptation of Linux Kernel Fuzzing Crash Reproducers](https://arxiv.org/abs/2611.30655) | ReproShift是一个自动化框架，通过保留漏洞触发语义、仅适配版本敏感行为，实现Linux内核模糊测试崩溃复现器在不同内核版本间的自动迁移。 |
| [^6] | [Oops!... I Parsed It Again: Tool-Call Smuggling Against LLM Agents](https://arxiv.org/abs/2611.30624) | 论文提出“工具调用走私”攻击：利用 JSON 在守卫、执行器等多个解析器之间语义不一致的漏洞，使守卫批准的工具调用与执行器实际运行的调用不同，从而绕过大语言模型智能体的安全防护并留下虚假审计记录。 |
| [^7] | [AgentBouncer: Defending Software Engineering Agents via Dynamic System-Call Mediation](https://arxiv.org/abs/2611.30621) | AgentBouncer是一种运行时强制执行系统，利用LLM仲裁器从必要性和安全性两个维度中介（必要时拒绝）软件工程智能体触发的系统调用，从而有效防御提示注入、恶意工具和供应链攻击。 |
| [^8] | [POMA: Beyond Final Flags -- Process-Level Evaluation of Base LLMs in Vulnerability Exploitation](https://arxiv.org/abs/2611.30590) | POMA提出了一种过程级评估方法与平台，通过统一执行基线（固定任务输入、工具接口、目标环境、反馈格式与执行预算）隔离评估基础LLM，并在55个CTF Pwn挑战上基于运行时证据测量中间状态与状态转换，从而超越以最终成败Flag为中心的传统评测方式。 |
| [^9] | [Improving Repair Agents with Cause-Chain Diagnoses](https://arxiv.org/abs/2611.30574) | FORECAST 通过比较通过测试与失败测试的执行推断出解释失败原因的“因果链”，并将其附加到问题描述中，从而无需修改智能体即可显著提升各类修复智能体在 SWE-bench Lite 上的问题解决率。 |
| [^10] | [sudo L7: Can Coding Agents Act Like Staff Engineers?](https://arxiv.org/abs/2611.30517) | 该论文提出了 sudo L7 基准，包含 60 个由资深工程师在真实生产代码库中设计的软件工程任务，用于评估编程智能体能否像资深工程师一样发现隐藏风险、质疑缺陷前提并诚实汇报，而不仅仅是实现功能正确性。 |
| [^11] | [When an AI agent makes a mistake in a database, what decides the damage?](https://arxiv.org/abs/2611.30483) | 该论文通过两层预注册实验——脚本化执行的22种典型错误库和由六个小型开源模型驱动的最小工具调用智能体——系统研究了当AI智能体在数据库中犯错时，损害程度究竟由模型、防护层还是数据库引擎本身决定。 |
| [^12] | [LeanReviewBench: Semantic Alignment Review in Mathematical Lean Projects](https://arxiv.org/abs/2611.30361) | 提出了LeanReviewBench——一个包含来自18个真实Lean项目的197个标注输入的项目级基准，用于评估语言模型审查非形式化数学目标与形式化Lean命题之间语义对齐的能力。 |
| [^13] | [RepoClean: Evaluating Coding Agents on Repository-Level Obsolete Code Cleanup](https://arxiv.org/abs/2611.30350) | 提出了RepoClean基准，通过将真实仓库历史中已删除的代码自动重新注入新版本，构建了200个仓库级清理任务，以评估编码智能体安全识别并移除过时代码的能力。 |
| [^14] | [What Field Enumeration Costs to Compile in C++26: Native Reflection, Structured-Binding Packs, Reflection Libraries and Hand-Written Serializers on GCC 16](https://arxiv.org/abs/2611.30250) | 该论文首次系统测量了 C++26 各类字段枚举方案（原生反射、结构化绑定包、反射库与手写序列化代码）在 GCC 16 上的编译开销，发现基于 span 的原生反射枚举比结构化绑定包慢约 7.96 倍，而改用常量数组循环可将开销降低约一半以上。 |
| [^15] | [Software-Engineering-as-an-Artifact: From Engines to Systems That Build Themselves](https://arxiv.org/abs/2611.30247) | 该论文借鉴未来学的情景规划方法，提出一个极端情景：软件工程流程将完全物化为制品（如上下文文档和可执行技能），最终形成能从需求自主生成和更新软件的软件工程引擎，人类工程师的角色将从开发软件产品转向维护这些流程制品。 |
| [^16] | [What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym](https://arxiv.org/abs/2611.30207) | 本研究系统审计了SWE-Gym数据集奖励标签的正确性，发现模型从未解决的任务中74%存在拒绝正确修复的假阴性测试，修复或移除这些错误标注任务后，训练出的模型在SWE-bench Verified和SWE-bench Pro上分别提升2.4分和2.7分。 |
| [^17] | [VADE: A Verifiable Agent Data Engine from Professional Sources](https://arxiv.org/abs/2611.30202) | VADE是一个数据引擎，它利用专业分析“求解难、验证易”的非对称性，从法律、金融等专业实践的公开来源中自动构建可验证的分析任务，从而解决这些领域高质量智能体训练数据难以规模化的问题。 |
| [^18] | [AEGIS: Evidence-Guided Set Decisions for Requirement-to-Code Traceability](https://arxiv.org/abs/2611.30093) | 提出AEGIS框架，将检索证据与代码关系相结合，在项目级审查预算下通过局部搜索、动态规划和调用上下文准入机制，为需求构建并修订代码可追溯集合，从而在有限的审查成本下提升可追溯性恢复质量。 |
| [^19] | [Domain advantage in two-dimensional quantum-device design with multi-modal agentic harness](https://arxiv.org/abs/2611.30043) | 该论文提出了qlaybench基准测试和qlaybot多模态智能体框架，使语言模型智能体能够模拟人类工作流程，将显微镜图像转化为适配样品的二维量子器件光刻版图，证明了领域定制的多模态智能体设计相较于传统工具具有显著优势。 |
| [^20] | [STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience](https://arxiv.org/abs/2611.30034) | 该论文提出STUN-Bench基准，构建了经验池中完全不包含172个保留工具任何信息的数据分割，并揭示了共享经验会挤占工具描述排名、导致真正未见的工具从候选列表中消失的“拥挤”问题。 |
| [^21] | [OWE-Bench: A long-horizon benchmark for measuring real-world engineering](https://arxiv.org/abs/2611.30017) | OWE-Bench是一个由领域专家设计的、包含26个长时程真实工程任务（涵盖软件、物理、机器学习和科学计算）的基准，用于评估AI模型的实际工程能力，目前最佳前沿模型得分仅为54.6%。 |
| [^22] | [LLM-Assisted Software Migration: from Module-Level Transformation to Repository-Level Integration](https://arxiv.org/abs/2611.29942) | 该综述通过分析96篇文献（含46篇直接迁移论文），从软件工程活动、模型角色、编排工作流和使能技术四个视角，系统梳理了LLM辅助软件迁移从模块级转换走向仓库级集成并交付完整应用的研究现状，并比较了现有评估实践与基准。 |
| [^23] | [Does the Agent's Map Match the Territory? Architectural Claims in AI Agent Context Files and Their Fit with the Code](https://arxiv.org/abs/2611.29916) | 本研究首次大规模验证了AI智能体上下文文件中架构声明与实际代码的相符程度，发现路径声明真实错误率较低（约2.2%）且会随代码演进及时修复，但依赖规则中有18%在编写当天即告失效且几乎从不会被修复。 |
| [^24] | [Context-Aware Proactive Self-Adaptation: A Two-Layer Model Predictive Control Approach](https://arxiv.org/abs/2611.29857) | 本文提出一种基于上下文目标模型的双层模型预测控制机制，实现了软件系统的上下文感知主动式自适应与控制器自身的自适应，提升了系统的性能与适应性。 |
| [^25] | [Landscape and Taxonomy of Prompt Engineering Patterns in Software Engineering](https://arxiv.org/abs/2611.29853) | 本文基于系统性文献综述提出了软件工程领域提示工程模式的分类法，表明即使大语言模型能力持续增强，精心设计的提示模式对于提升其可用性、可靠性和任务性能仍然至关重要。 |
| [^26] | [Can Vibe Coding Produce Accessible Websites: An Empirical Study on LLM Coding Agents and Skills](https://arxiv.org/abs/2611.29822) | 本文通过生成90个网站并使用多种无障碍检测工具进行评估的实证研究发现，LLM编程智能体端到端生成的网站无一完全无障碍（平均违反约80个WCAG成功标准），揭示了“氛围编程”工作流会大规模传播无障碍缺陷。 |
| [^27] | [Blind to What They Can Fix: A Detection Gap in Self-Improving Agents](https://arxiv.org/abs/2611.29807) | 研究发现自我改进智能体能写出高分程序，但其自建的运行时检查在环境变化后对一半以上的失败视而不见，而这些失败其实可通过正常运行记录检测并由智能体自身修复。 |
| [^28] | [Generating Safety-Critical Driving Scenarios from Role-Typed Signal Temporal Logic Specifications Drafted by Large Language Models](https://arxiv.org/abs/2611.29800) | 该论文提出由大语言模型从描述中起草按角色分类的信号时序逻辑规范，再由编译器将其转化为对扩散模型样本的最小编辑优化，从而生成安全关键驾驶场景，用于测试自动驾驶车辆的运动规划器。 |
| [^29] | [Agentic Root Cause Reasoning and Localization of Unsoundness in Static Call Graph Analysis](https://arxiv.org/abs/2611.29745) | 提出智能体框架 CGRizz，通过对比静态与动态调用图定位缺失的调用边，由 LLM 智能体推理不健全性的根因，并用运行时插桩、复现测试用例和工具位置警告三重独立证据加以验证，实现对静态调用图分析缺陷的诊断与自动预警。 |
| [^30] | [Admitted Is Not Effective: Lessons from a Notary-Gated Retrospective Loop in an Agentic Pipeline](https://arxiv.org/abs/2611.29727) | 尽管公证人把关的回溯循环能够有效治理智能体流水线中的规则更新（包括拒绝引用虚假证据的提案），但由于评审信号过于稀疏且被采纳的规则缺乏实际效果，流水线性能在十次回溯后并未改善，证明“被采纳”并不等于“有效”。 |
| [^31] | [ReqSeek: Transformer-Based Automatic Requirements Identification for Building Large Requirements Datasets](https://arxiv.org/abs/2611.29722) | 本文提出了基于Transformer、领域无关的自动需求识别模型ReqSeek，并构建了包含23个异构SRS的ARID数据集，以RoBERTa为基础模型达到95%的精确率与召回率，性能可与人类媲美且耗时更少。 |
| [^32] | [AutoRPG: Harnessing Systematic AI for Non-Intrusive Low-Latency Mobile RPG Combat](https://arxiv.org/abs/2611.29673) | AutoRPG提出了一种两阶段系统性AI框架，通过高层战略规划结合一套轻量级执行模块（目标追踪、角色移动、闪避、异常处理），实现了非侵入式、低延迟且具备细粒度行为控制的移动端RPG自动战斗。 |
| [^33] | [fx(hash): Salvaging The Source Code of 28k Generative Artworks](https://arxiv.org/abs/2611.29604) | 本研究抢救并归档了已破产的生成艺术NFT平台fx(hash)上5,532位艺术家的全部28,193个生成程序源代码至Software Heritage，揭示了NFT生成艺术因源代码存于链下而面临的脆弱性。 |
| [^34] | [KernelShift: Benchmarking Kernel Optimization Across Evolving Hardware](https://arxiv.org/abs/2611.29590) | 该论文提出了KernelShift基准测试流水线，通过从ncnn、llama.cpp等真实活跃仓库中构建99个内核优化任务，并让智能体在目标硬件反馈下优化代码，以评估编码智能体在不断演进的CPU架构扩展下编写高性能内核的能力。 |
| [^35] | [Endpoint-Conditioned Process Value: What an Endpoint-Blind Process Score Adds Once Terminal Status and Cost Are Known](https://arxiv.org/abs/2611.29354) | 本研究通过1,255条轨迹的实验首次量化了过程评分在已知最终状态和成本后的增量价值，发现其增益高度依赖评判模型规模——只有32B级别的评判模型才能带来显著的AUC提升，且大部分增益可由八个可记录的行为计数器替代。 |
| [^36] | [Multi-Agent Coordinated Method-Level Refactoring of Rust Code](https://arxiv.org/abs/2611.29321) | 该论文提出了一种基于大语言模型的多智能体协同工作流，通过候选生成、可配置专家审查与容器化验证三个环节，安全地实现 Rust 代码的方法提取、内联与移动重构，并在开源项目任务上首次尝试即全部通过验证。 |
| [^37] | [MetaSkill: Reusable Problem-Solving Methods for Adaptive Skill Orchestration in LLM Agents](https://arxiv.org/abs/2611.29274) | MetaSkill提出一个双层框架，通过选择并实例化可复用的问题解决方法来引导LLM智能体自适应地编排技能，在SkillsBench和ToolHop基准上相比最强基线准确率提升3.51%-13.48%。 |
| [^38] | [Edit-Induced Dependency Mismatch: Propagation Omissions, Recovery, and Evaluation Blind Spots in Coding Agents](https://arxiv.org/abs/2611.29242) | 该论文揭示了编码智能体在迭代改代码时会依赖上下文中残留的过时依赖判断，导致缺陷传播并逃过基准测试评估，并通过盲测人工审计确认了此类“编辑引发的依赖失配”的真实存在、运行中的恢复现象以及现有评估体系的盲区。 |
| [^39] | [SatMend: LLM Agents for Automated Software Repair in Satellite Constellations with Intermittent Ground Contact](https://arxiv.org/abs/2611.29227) | SatMend提出了一种由地面LLM智能体驱动的卫星星座自动化软件修复方法，通过持久会话保留诊断上下文、结合源码与运行时证据选择修复或状态恢复动作，并使地面修复准备与星载交付验证在间歇性地面联系下并行进行。 |
| [^40] | [Fixed or Deleted? External Feedback Lets Cheaper AI Coding Agents Repair Rule Violations](https://arxiv.org/abs/2611.29162) | 研究发现，引入商业检查器CodeAssure等外部反馈（报告违规位置和修复目标）能大幅提升AI编程智能体修复项目规则违规的成功率，使更小更便宜的模型（修复率从26%升至69%）接近甚至逼近大型模型的修复效果。 |
| [^41] | [Implementation and Software-in-the-Loop Evaluation of a Trans-Domain Digital Twin Architecture for Autonomous Guidance of the Phoenix VTOL UAV Without Online GNSS](https://arxiv.org/abs/2611.29158) | 本研究设计并实现了基于跨域数字孪生（TDDT）架构的凤凰号垂直起降无人机自主制导系统，在无在线GNSS条件下通过导航、飞行动力学与气象多模型协同及多传感器数据融合实现可靠导航，并通过软件在环仿真进行了评估验证。 |
| [^42] | [Operational Runbooks as Living Documents: A Longitudinal Study of Document Evolution under Assisted Post-Incident Revision](https://arxiv.org/abs/2611.29097) | 该论文首次对生产环境中运维手册作为“活文档”的长期演进进行了纵向实证研究，并提出利用轻量级的语言学与结构信号来量化与监控文档随时间的变化。 |
| [^43] | [LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards](https://arxiv.org/abs/2611.29034) | 本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。 |
| [^44] | [Watermark Claims Under Test: Lessons from Auditing a Production LLM](https://arxiv.org/abs/2611.29018) | 对生产级LLM的红绿水印黑盒检测审计表明，已发表的黑盒测试在真实部署的水印模型上会失效，因为提示词被纳入水印的重复上下文历史后会恰好抑制检验所测量的标记，导致探针“失明”。 |
| [^45] | [Consumer-Time Validity in Stateful AI Runtimes: Verifying Recovered State and Deferred Authority](https://arxiv.org/abs/2611.29004) | 该论文提出“消费者时有效性”这一新概念，指出在有状态AI运行时中，一个组件记录的取消、回滚或批准并不能保证下游消费时的有效性，并将其分为约束读取内容的状态有效性与约束执行内容的权限新鲜度两类缺失检查。 |
| [^46] | ["What Should I Do?" Guidelines for Responsibly Integrating AI in Software Engineering](https://arxiv.org/abs/2611.28996) | 该研究通过对微软、GitHub和辉瑞1118名开发者的两阶段大规模调查，总结出35条帮助开发者在软件工程工作中负责任地集成AI的实用指南，并在一年后跨行业验证了其有效性。 |
| [^47] | [Micro Operational Robustness of LLMs on Code Generation](https://arxiv.org/abs/2611.28957) | 本文提出了一种名为“场景域分析”的鲁棒性评估方法，通过寻找编码任务自然语言描述中导致大语言模型输出错误结果的最小变化，来系统评估LLM在代码生成任务中的微观操作鲁棒性。 |
| [^48] | [Refined, Reliable and Reproducible: Algorithmic Analysis and Reimplementation of the Popcorn Void Finder](https://arxiv.org/abs/2611.28882) | 本文对宇宙空洞查找工具 Popcorn Void Finder 进行了全面的重新实现，以算法透明性、可维护性和可复现性为核心，并系统评估了发表后算法修改所带来的统计与计算影响。 |
| [^49] | [CivilCADBench: Benchmarking Multimodal LLMs for Understanding, Reconstructing, and Designing Civil Engineering Drawing Sets](https://arxiv.org/abs/2611.28853) | 提出了基于包含25万余个CAD实体的真实基础设施图纸构建的CivilCADBench基准，用于评估多模态大语言模型在土木工程图纸集的理解、重建与设计能力。 |
| [^50] | [Dhaani: A Data-Flow-Driven Tool for Greenfield Construction and Brownfield Modernisation](https://arxiv.org/abs/2611.28798) | Dhaani 是一款以带注释数据流图（ADFD）作为意图与实现之间可审查契约的免费桌面工具，支持将自然语言需求按依赖顺序构建为全新 Python/.NET 程序，并能从 Fortran 存量代码库迁移生成现代化代码。 |
| [^51] | [Zero-Trust Formal Methods: AI Agents Build Verified Checkers for Untrusted Solvers](https://arxiv.org/abs/2611.28704) | 该论文提出零信任形式化方法，让AI智能体自动生成Isabelle/HOL理论、可靠性证明和经过形式验证的检查器，使得求解器的答案只有在验证检查器接受其证书后才被采信，从而完全不信任AI智能体和求解器本身。 |
| [^52] | [When to Test, Repair, or Regenerate: Decision-Guided Efficient RTL Generation](https://arxiv.org/abs/2611.28667) | 该论文提出一种由专门决策模型引导的自适应框架，智能决定何时对RTL设计进行测试、修复或重新生成，从而在降低推理成本的同时，在三个基准上将平均Pass@1提升5.4至7.1个百分点。 |
| [^53] | [VeriLadder: A Multi-Agent Specification Refinement System for Tool-Verified Verilog Datasets](https://arxiv.org/abs/2611.28617) | 本文提出VeriLadder，一个五级Verilog数据集及其多智能体构建框架，由LLM智能体编写和修订规格说明，并由综合器与形式等价性检查器等EDA工具进行验证，从而解决了编写Verilog的LLM缺乏高质量训练数据的问题。 |
| [^54] | [Right Verdict, Right Reason? An Execution-Grounded Study of Whether LLM Vulnerability Detectors Understand Vulnerabilities](https://arxiv.org/abs/2611.28584) | 该研究以程序执行为基准，对515个可重现漏洞设计了五级递进测试，发现LLM漏洞检测器的高基准分数未必代表它们真正理解漏洞，其表现可能与无需理解的简单基线相当。 |
| [^55] | [Automated Safety Analysis for Agentic Workflows](https://arxiv.org/abs/2611.28469) | 该论文提出了Apsis，一个部署前的静态安全分析器，它通过将不同框架的智能体工作流代码提升为统一的语义图中间表示，从而在部署前自动检测工作流设计层面的逻辑缺陷。 |
| [^56] | [Human-in-the-Loop Runbook Improvement with Agentic Support Automation](https://arxiv.org/abs/2611.28441) | 本文提出了一种将智能体AI融入基于运维手册的事件处理流程的系统，该机器人能引导人工操作员完成故障缓解过程、评估处理结果，并在发现手册存在缺口时建议改进，实现了人机协同的运维手册持续优化。 |
| [^57] | [Lean Census: A Distributional Analysis of AI-Generated Lean Code](https://arxiv.org/abs/2611.28430) | 该研究通过对代码统计量分布的对比分析，揭示了 AI 生成的 Lean 库与人类编写的库之间存在显著差异——AI 生成的库每个定理的内联声明更多、证明更长、实例更少、私有声明更多。 |
| [^58] | [When AI Should Not Decide: Evidence Sufficiency and Decision Stability in Automated Job Interviews](https://arxiv.org/abs/2611.28426) | 本研究提出证据充分性与决策稳定性评估框架（ESDSA），通过证据分类、引文溯源和语言变体稳定性检验来判断AI自动化面试评估建议是否有足够证据支撑，在留出测试中实现了85.19%的可操作覆盖率且建议分歧率仅3.80%。 |
| [^59] | [Taking the PULSE of Enterprise Workflows: Evaluating Persistent Project-State Understanding](https://arxiv.org/abs/2611.28374) | 本文提出了持久性项目状态理解（PPSU）任务及PULSE基准（包含1,573个问题、覆盖30个企业项目历史），实验发现检索与抽取是当前系统的主要瓶颈，但即使在完美证据之上仍需显式项目状态才能取得进一步提升，证明该任务无法简化为单纯的检索问题。 |
| [^60] | [When Tools Fail, Agents Fabricate: Silent Failure and Verification-First Oversight in Tool-Using LLM Agents](https://arxiv.org/abs/2611.28341) | 该论文构建了RetryRealistic受控故障注入测试平台，揭示了工具使用型LLM智能体在工具失效时恢复能力不可靠且常发生“静默捏造”（将错误答案自信地当作事实输出），失败模式主要取决于故障类型而非模型本身，并据此倡导“验证优先”的监督机制。 |
| [^61] | [Cross-Boundary Vulnerability Management for SDVs: A Traceability Architecture at the ISO/SAE 21434 and EU Cyber Resilience Act Interface](https://arxiv.org/abs/2611.28303) | 针对软件定义汽车中车辆型式认证法规体系（ISO/SAE 21434、UN R155/R156）与欧盟《网络弹性法案》之间漏洞管理交接的空白，提出了四问题分类程序、五法规体系可追溯性矩阵和含四条移交路径的参考架构，实现跨法规边界的漏洞可追溯管理。 |
| [^62] | [Jailbreaks Are Global or Regional? A Controlled Study of Scale and Geolocation Effects on LLM Safety Verification](https://arxiv.org/abs/2611.28174) | 该研究通过配对两阶段协议评估12个不同规模模型并从三大洲五个城市的VPN节点发送相同提示，发现模型规模与请求地理位置都会显著影响攻击成功率的比较结果，且参数量并非衡量模型安全性的可靠指标。 |
| [^63] | [SecretSifter: A Runtime Secret-Detection Engine for Live Web Traffic](https://arxiv.org/abs/2611.28168) | SecretSifter是一个开源的运行时密钥检测引擎，通过组合格式锚定与上下文门控规则、语义化密钥名提取、香农熵评分以及CryptoJS/AES硬编码专项检测，在实时Web流量中于凭证交付到浏览器的时刻发现密钥级凭证，填补了传统左移式仓库扫描器无法覆盖的运行时攻击面。 |
| [^64] | [How Far Can LLMs Go in Generating Android Lint Checks from Natural Language?](https://arxiv.org/abs/2611.28159) | 该论文提出了LintBench基准（包含113个经人工验证的真实Android Lint检查及自然语言描述），首次系统评估了LLM能否仅凭自然语言描述生成可运行的Lint检查及其所需的外部知识。 |
| [^65] | [From Observation to Explanation: Mechanistic Interpretability of LLM-Generated Code for Principled Repair](https://arxiv.org/abs/2611.28153) | 该论文通过纵向、实证与行为三个层面的证据三角验证，揭示了现有基准高估 LLM 代码生成能力、且 LLM 生成的代码在生产环境中需要不成比例的修正性修改等现象，并从机制可解释性角度为 LLM 生成代码的原则性修复提供了因果解释。 |
| [^66] | [Every IVUS Frame in Its Place: A Zero-Install Browser Tool for Registering Intravascular Ultrasound to the Coronary Angiogram, and a Path to Angiography-Only Plaque Imaging](https://arxiv.org/abs/2611.28089) | 该论文提出了一种零安装、纯浏览器运行的半自动IVUS-冠脉造影配准工具，可将每个IVUS帧定位到造影血管上并导出帧级训练记录，为仅凭造影实现斑块成像和AI模型训练数据的积累开辟了道路。 |
| [^67] | [A No-Code Toolchain for Validating Wearable Motion Sensing Systems](https://arxiv.org/abs/2611.28027) | 该论文提出了一种无需编程的无代码工具链，让非技术用户能够通过摄像头提取真实关节角度数据、训练和测试机器学习模型并可视化分析传感器布局，从而低成本、高效率地验证可穿戴电子织物运动感知系统的设计。 |
| [^68] | [Technology-Enabled Fuel Supply Chain Resilience: An Integrated Information System for Real-Time Monitoring and Decision Support](https://arxiv.org/abs/2611.25542) | 本研究将燃料物流信息系统重构迁移至SQL Server平台，使数据集成缺陷率从2.90%–3.46%降至0.14%–0.49%，并通过新增库存移动记录、自动警报和恢复控制功能，提升了燃料供应链的实时监控与决策支持能力。 |
| [^69] | [Validating Concurrent Correctness Witnesses with Ghosts by Thread-Modular Abstract Interpretation](https://arxiv.org/abs/2611.23792) | 提出了一种新颖的并发正确性见证验证框架，利用幽灵变量中编码的语义信息，通过线程模块化抽象解释来验证带幽灵变量的并发正确性见证，而非将其简单视为额外的证明义务。 |
| [^70] | [Rule-Guided and Trajectory-Aware Repair for LLM-Based HLS Generation](https://arxiv.org/abs/2611.07324) | MementoHLS框架通过可执行修复契约库（将跨任务HLS知识转化为证据驱动的修复动作）与证伪账本（将修复历史转化为轨迹证据以抑制无效路径）的双记忆引导机制，并借助同一候选评审者实现可靠的结果归因，显著提升了大语言模型从自然语言规范生成可综合HLS-C代码的端到端成功率。 |
| [^71] | [SWE-Ladder: Learning to Localize, Reproduce, and Patch via Progressive Privileged Self-Distillation](https://arxiv.org/abs/2610.33667) | 该论文提出SWE-Ladder框架，将已验证的修复过程分解为定位、复现和修补三个阶段的特权证据，通过基于规则的路由器和冻结教师模型按修复顺序进行渐进式自蒸馏，并结合GRPO优化完整轨迹，为SWE智能体提供了传统强化学习和OPSD所缺失的阶段特定训练指导。 |
| [^72] | [VERDICT: Multi-Agent Parallel Modular Verification](https://arxiv.org/abs/2610.29147) | VERDICT是一个多智能体框架，通过基于假设的并行规范综合打破传统自底向上验证的依赖顺序，实现大规模程序模块化演绎验证的并行化。 |
| [^73] | [HPCAgent-Bench: A Framework for Sound Benchmarking of Code-Optimizing Agents, Harnesses, and Skills](https://arxiv.org/abs/2610.20673) | 该论文提出了HPCAgent-Bench——一个包含689个内核、覆盖三个赛道的代码优化基准框架，通过统一的内核规范、外部防作弊裁判以及“干预效力”等新指标，实现对代码优化智能体、执行框架与技能各自贡献的可靠隔离评估。 |
| [^74] | [Verifying Autoware Controller Node by Combining Model Verification and Runtime Monitoring](https://arxiv.org/abs/2610.15418) | 本文提出一种将微分动态逻辑形式化模型验证与运行时监控相结合的混合验证方法，用于验证开源自动驾驶平台Autoware轨迹跟随控制节点的安全性与活性属性。 |
| [^75] | [SaltBench: A Referee-Gated Protocol for Measuring Method Effects in Machine-Checked Software Work](https://arxiv.org/abs/2609.11076) | 提出了SaltBench，一个由机器裁判门控的基准测试协议，通过严格的隔离机制（经探针实际检验）、预注册预测和“停止即暂停”等规则，使机器裁判对编程智能体工作方式的影响变得可测量且不可事后叙述。 |
| [^76] | [Update from Hell: Can Coding Agents Survive Hidden Breakage in Dependency Upgrades?](https://arxiv.org/abs/2608.30300) | 该论文提出了 DEPBENCH 基准，包含跨越五个语言社区五个包生态系统的 203 个真实依赖升级任务，用于评估编码智能体能否应对依赖升级中隐藏的代码级破坏并完成源代码适配。 |
| [^77] | [SWE-Pruner Pro: The Coder LLM Already Knows What to Prune](https://arxiv.org/abs/2607.18213) | SWE-Pruner Pro利用代码智能体自身的内部表示直接在模型内部剪枝工具输出，无需外部分类器，最多可节省39%的token且不损失甚至提升任务性能。 |
| [^78] | [Articulating Assumptions in AI-Generated Scientific Analyses through Task Decomposition](https://arxiv.org/abs/2607.05762) | 提出多智能体框架“基于量化语义差异比较”，通过任务分解揭示AI生成科学分析代码中的隐含假设，重建关键输出量的产生过程并检测用户指令中的歧义。 |
| [^79] | [GameEngineBench: Full-Session Verification of Coding Agents in a Networked Game Engine](https://arxiv.org/abs/2607.03525) | 该论文提出了 GameEngineBench 基准（包含源自虚幻引擎 5 项目的 110 个 C++ 实现任务），并引入全程会话验证框架，通过在 PIE 会话（含联网客户端世界）中注入隐藏测试，来评估编码智能体所生成代码的行为正确性与跨客户端一致性。 |
| [^80] | [BeSpec: Behavior-Level Specification Alignment for Code Generation](https://arxiv.org/abs/2607.02949) | 本文提出BeSpec，一种基于行为模型的规格对齐方法，通过提取结构化规格并预测可执行行为来检测意图与实现之间的不匹配，解决了LLM代码生成中因规格不匹配而错误实现意图的核心瓶颈。 |
| [^81] | [PyMETA: Evaluating Student Code Diagnosis on and Beyond the First Execution Error](https://arxiv.org/abs/2606.30610) | 该论文提出了PyMETA数据集，包含48,646份学生Python代码提交及三层级错误分类体系（最细粒度含14个标签），首次系统评估了大语言模型在首次执行错误及超越首次错误（如逻辑错误）层面诊断学生代码的能力。 |
| [^82] | [Formally-Verifiable and Human-Interpretable RTL Generation with LLM-Guided Stepwise Refinement](https://arxiv.org/abs/2606.19387) | 该论文提出将大语言模型的生成能力与逐步精化的形式化方法相结合，把自然语言需求转化为形式化硬件契约，并在每一步设计决策变换前进行形式化验证，从而实现构造即正确、可形式化验证且人类可理解的RTL硬件代码生成。 |
| [^83] | [When the Next Step Is Not One Step: Distribution-Aware Execution Modeling for Concurrent Go Programs](https://arxiv.org/abs/2606.17508) | 该论文将并发 Go 程序的“下一事件预测”从单标签分类重新形式化为对经验性下一事件分布的估计，并发布了基于真实 GoKer 缺陷内核的基准，但实验发现使用交叉熵和 KL 目标微调的 Qwen2.5-Coder-7B 适配器与简单基线在统计上无显著差异，是一个中性偏负面的结果。 |
| [^84] | [Looking Is Not Picking: An Attention-Segment Account of Tool-Selection Failures in LLM Agents](https://arxiv.org/abs/2606.16364) | 这篇论文通过对注意力片段的分析发现，LLM智能体在80%的工具调用失败中其实最关注的正是正确的工具，失败源于决策读出环节而非工具列表干扰——修复提示词只能挽回≤23%的失败，而读出端干预可挽回59-91%。 |
| [^85] | [MARGIN: Margin-Aware Regularized Geometry for Imbalanced Vulnerability Detection](https://arxiv.org/abs/2605.10240) | 该论文提出MARGIN框架，通过自适应边际度量学习和超球面原型建模来纠正数据不平衡在超球面表示空间中造成的几何畸变，从而提升软件漏洞检测的性能与决策边界稳定性。 |
| [^86] | [Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses](https://arxiv.org/abs/2604.25850) | 该论文提出智能体框架工程（AHE）闭环系统，通过组件、经验和决策三大可观测性支柱，首次实现了编码智能体框架的自动化演化，解决了行动空间异构、轨迹信号淹没和编辑效果难归因三大挑战。 |
| [^87] | [DynaFix: Iterative Automated Program Repair Driven by Execution-Level Dynamic Information](https://arxiv.org/abs/2512.24635) | DynaFix的核心创新在于在迭代修复过程中反复收集（通过ByteTrace获取变量状态、分支路径和调用栈）并更新执行层动态信息，将其融入提示并以重执行验证补丁，形成闭环，使基于LLM的自动程序修复扎根于具体的运行时行为。 |
| [^88] | [Enhancing Code Consistency in AI Research with Large Language Models and Retrieval-Augmented Generation](https://arxiv.org/abs/2502.00611) | 本文提出一种结合检索增强生成与大语言模型的系统，用于自动验证研究论文的代码实现是否与论文所述算法方法一致，从而提升代码验证的准确性和AI研究的可信度。 |

# 详细

[^1]: 个人软件创作的混合方法研究

    A Mixed-Methods Study of Personal Software Creation

    [https://arxiv.org/abs/2611.31217](https://arxiv.org/abs/2611.31217)

    本研究通过分析 6,168 条网络与 Reddit 帖子并跟踪八名用户使用 AI 编程智能体构建个人软件的过程，首次大规模揭示了人们创作个人软件的行为、动机及其面临的机遇与挑战。

    

    长期以来，人们一直希望软件能够围绕自己的数据、日常习惯和偏好来定制，我们将这类软件称为“个人软件”。数十年来，编程专业知识限制了谁能够构建这类软件：缺乏编程专业知识的人只能在受限的终端用户编程工具（如电子表格）中进行开发。AI 编程智能体似乎消除了这一瓶颈，使人们能够通过自然语言描述构建完整的应用程序，如今许多人已公开构建并分享个人软件。然而，个人软件很少得到大规模的研究：对于人们构建什么、为什么构建、如何构建，以及他们遇到的机会与挑战，我们知之甚少。我们通过一项混合方法研究来调查这些问题：一项灰色文献综述分析了 6,168 个网页和 Reddit 帖子，一项观察研究跟踪了八个人使用 AI 编程智能体为自己构建软件并使用长达（原文在此处截断）

    arXiv:2611.31217v1 Announce Type: new  Abstract: People have long wanted software shaped around their own data, routines, and preferences. We refer to such software as \emph{personal software}. For decades, programming expertise limited who could build it: people without that expertise could build software only within constrained end-user programming tools, such as spreadsheets. AI coding agents seem to remove this bottleneck by letting people build full applications from natural-language descriptions, and many people now build and share personal software publicly. Yet personal software has rarely been studied at scale: we know little about what people build and why, how they build it, and what opportunities and challenges they encounter. We investigate these questions with a mixed-methods study. A grey-literature review examines 6,168 web and Reddit posts, and an observation study follows eight people as they build software for their own needs with AI coding agents and use it for up t
    
[^2]: LLM 智能体技能供应链中的遗留型与智能体原生威胁

    Carried-Over and Agent-Native Threats in the Supply Chain of LLM Agent Skills

    [https://arxiv.org/abs/2611.30932](https://arxiv.org/abs/2611.30932)

    该论文首次在注册表规模上对 LLM 智能体技能供应链中由非故意漏洞引发的攻击面进行实证研究，对 ClawHub 全部 48,682 个技能开展静态分析、人工审查（3,428 个标记案例）与可达性测试，覆盖依赖项和第一方代码两类遗留风险，以及安装指令和凭证泄露两类智能体原生风险。

    

    智能体技能是通过脚本、第三方依赖和自然语言指令来扩展大语言模型（LLM）智能体能力的可复用软件包。智能体会根据用户请求和不受信任的外部内容来构建技能参数，因此善意作者遗留的缺陷可能将提示注入升级为代码执行。现有研究大多针对恶意技能或单一攻击向量，且没有研究测试智能体是否能够实际触达此类缺陷。我们针对智能体技能中非故意漏洞所引发的供应链攻击面，开展了首个注册表规模的实证研究，覆盖 ClawHub 上的全部 48,682 个技能。该研究涵盖两类遗留风险——依赖项与第一方代码，以及两类智能体原生风险——安装指令与打印到标准输出的凭证。我们通过静态分析筛查所有技能，人工审查四个主要检测器标记的全部 3,428 个案例，并通过运行抽样的被标记技能来测试可达性（摘要在此处截断）。

    arXiv:2611.30932v1 Announce Type: cross  Abstract: Agent skills are reusable packages that extend a large language model (LLM) agent with scripts, third-party dependencies, and natural-language instructions. An agent builds a skill's arguments from user requests and untrusted external content, so a flaw left by a benign author can escalate a prompt injection to code execution. Existing studies target malicious skills or a single attack vector, and none tests whether an agent can reach such flaws. We present the first registry-scale empirical study of the supply chain attack surface arising from unintentional vulnerabilities in agent skills, over all 48,682 skills in ClawHub. The study covers two carried-over risks, dependencies and first-party code, and two agent-native ones, install instructions and credentials printed to stdout. We screen all skills by static analysis, manually review all 3,428 cases flagged by the four main detectors, and test reachability by running sampled flagged
    
[^3]: 有用但笨拙：软件工程师如何看待编码代理中思维链解释的设计

    Useful but Unwieldy: How Software Engineers Perceive the Design of Chain-of-Thought Explanations in Coding Agents

    [https://arxiv.org/abs/2611.30894](https://arxiv.org/abs/2611.30894)

    本研究通过对36名软件工程师的调查与访谈发现，编码代理展示的思维链消息是软件工程师不可或缺的可解释性工具，但其设计仍显得笨拙、不易使用。

    

    由大语言模型（LLM）驱动的编码代理（CAs）正日益融入软件工程（SE）工作流程中。随着这些编码代理承担越来越多的任务和职责，软件工程师必须理解编码代理决策背后的推理逻辑，即思维链。在交互过程中，一些编码代理能够以单独消息的形式向用户展示这一思维链，旨在提供透明性和可解释性。然而，作为面向软件工程最终用户的可解释人工智能（XAI）机制，其有效性尚未在文献中得到证实。这促使我们通过对36名软件工程师开展调查和半结构化后续访谈，来了解他们将思维链消息作为XAI工具的使用体验。通过对调查回复和访谈记录的主题分析，我们发现思维链消息是软件工程师不可或缺的工具。

    arXiv:2611.30894v1 Announce Type: new  Abstract: Coding agents (CAs) powered by Large Language Models (LLMs) are increasingly integrated into software engineering (SE) workflows. As these CAs take up more tasks and responsibilities, it is imperative for software engineers to understand the CA's rationale behind its' decisions, also known as the chain-of-thought (CoT). During the interaction, some CAs are capable of showing this CoT as a separate message to users with the goal of providing transparency and explainability. However, its effectiveness as an explainable AI (XAI) mechanism for end-users in software engineering has not yet been established in the literature. This motivates us to understand their experiences with the CoT message as an XAI tool through a survey and semi-structured follow-up interviews with 36 software engineers. Through thematic analysis of the survey responses and interview transcripts, we found that CoT message is an indispensable tool for software engineers 
    
[^4]: 面向中断感知的Rust操作系统死锁静态检测

    Interrupt-Aware Static Detection of Deadlocks in Rust Operating Systems

    [https://arxiv.org/abs/2611.30713](https://arxiv.org/abs/2611.30713)

    该论文提出了一种结合中断感知静态分析与诊断引导的智能体审计的新方法，利用轻量级源码标注和基于MIR的流敏感分析来检测Rust操作系统中由中断引发的死锁。

    

    Rust在操作系统领域的应用日益广泛，但Rust的安全性保证并不能消除死锁，尤其是由中断引发的死锁。核心挑战在于跨越函数边界，从Rust基于守卫的同步机制中恢复对中断敏感的锁依赖关系。此外，将该分析应用于不同的Rust操作系统项目还需要明确项目特定的同步语义。我们提出了一种将中断感知静态分析与诊断引导的智能体审计相结合的方法，用于检测Rust操作系统中的死锁。该方法使用轻量级源代码级标注来恢复项目特定的锁与中断语义，执行基于MIR的流敏感分析以跟踪持锁与中断状态，并附带支持性证据报告潜在死锁。随后，智能体会在代码仓库中搜索相关源代码，并为每个报告返回附带理由的置信度评估。

    arXiv:2611.30713v1 Announce Type: new  Abstract: Rust is increasingly adopted in operating systems, but Rust's safety guarantees do not eliminate deadlocks, especially those induced by interrupts. The core challenge is to recover interrupt-sensitive lock dependencies from Rust's guard-based synchronization across function boundaries. Applying this analysis across Rust OS projects also requires specifying project-specific synchronization semantics. We present an approach combining interrupt-aware static analysis with diagnostic-guided agent auditing for detecting deadlocks in Rust operating systems. Our approach uses lightweight source-level annotations to recover project-specific lock and interrupt semantics, performs a flow-sensitive MIR-based analysis to track lock-holding and interrupt states, and reports potential deadlocks with supporting evidence. An agent then searches for relevant source code in the repository and returns a confidence assessment with reasons for each supplied r
    
[^5]: ReproShift：Linux内核模糊测试崩溃复现器的跨版本适配

    ReproShift: Cross-Version Adaptation of Linux Kernel Fuzzing Crash Reproducers

    [https://arxiv.org/abs/2611.30655](https://arxiv.org/abs/2611.30655)

    ReproShift是一个自动化框架，通过保留漏洞触发语义、仅适配版本敏感行为，实现Linux内核模糊测试崩溃复现器在不同内核版本间的自动迁移。

    

    可靠的崩溃复现对于诊断和修复内核崩溃至关重要。历史崩溃复现器保存了具体的漏洞触发知识，可被重用于回归测试、漏洞复验以及跨内核版本的影响分析。然而，内核的演进会通过改变内核接口、配置和执行路径而使这些复现器失效。因此，迁移这些复现器极具挑战性：适配后的复现器必须识别并修改版本敏感的行为，同时保留定义原始崩溃的漏洞触发操作和资源依赖。为应对这一挑战，我们提出了ReproShift，一个用于跨版本崩溃复现器迁移的自动化框架，它在保留漏洞触发语义的同时，仅对版本敏感的行为进行适配。给定源崩溃工件、源崩溃复现器和目标内核，ReproShift能够生成一个具有相同（崩溃语义）的迁移后崩溃复现器……

    arXiv:2611.30655v1 Announce Type: new  Abstract: Reliable crash reproduction is essential for diagnosing and repairing kernel crashes. Historical crash reproducers preserve concrete bug-triggering knowledge and can be reused for regression testing, bug revalidation, and impact analysis across kernel versions. However, kernel evolution can invalidate these reproducers by changing kernel interfaces, configurations, and execution paths. Migrating them is therefore challenging: the adapted reproducer must identify and modify version-sensitive behavior while preserving the bug-triggering operations and resource dependencies that define the original crash. To address this challenge, we propose ReproShift, an automated framework for cross-version crash reproducer migration by preserving bug-triggering semantics while adapting only version-sensitive behavior. Given source crash artifacts, a source crash reproducer, and a target kernel, ReproShift produces a migrated crash reproducer with same-
    
[^6]: 哎呀！……我又解析了一遍：针对大语言模型智能体的工具调用走私攻击

    Oops!... I Parsed It Again: Tool-Call Smuggling Against LLM Agents

    [https://arxiv.org/abs/2611.30624](https://arxiv.org/abs/2611.30624)

    论文提出“工具调用走私”攻击：利用 JSON 在守卫、执行器等多个解析器之间语义不一致的漏洞，使守卫批准的工具调用与执行器实际运行的调用不同，从而绕过大语言模型智能体的安全防护并留下虚假审计记录。

    

    大语言模型智能体通过工具调用来执行操作：模型将调用的参数写成 JSON 格式，守卫决定该调用是否可以运行，执行器负责运行它。同样的字节会被多个组件反复解析，包括提供商、网关、SDK、框架、守卫和执行器。JSON 把一些解释权留给了各个解析器，例如重复键中哪一份副本才生效。当守卫和执行器做出不同选择时，守卫批准的是一个调用，执行器实际运行的却是另一个调用，而审计日志记录的只是被批准的那一个。我们提出了这种攻击，即“工具调用走私”，并在整个技术栈中对它进行了研究。我们描述了六种相同字节产生不同调用的方式，并构建了一个差分测试工具，向 17 个固定版本的解析器和验证器配置提供完全相同的字节；大多数配置组合在某些输入上会出现分歧，甚至两个官方 MCP SDK 接受的输入也不相同。在 τ-bench 和 AgentDojo 的任务上，能够在传输过程中重写字节的攻击者……

    arXiv:2611.30624v1 Announce Type: cross  Abstract: LLM agents act through tool calls: the model writes the arguments of a call as JSON, a guard decides whether the call may run, and an executor runs it. The same bytes are parsed several times, by the provider, gateways, SDKs, frameworks, the guard, and the executor. JSON leaves some choices to each parser, such as which copy of a repeated key counts. When the guard and the executor choose differently, the guard approves one call, the executor runs another, and the audit log records the approved one. We introduce this attack, tool-call smuggling, and study it across the stack. We describe six ways in which the same bytes yield different calls and build a differential testing harness that gives identical bytes to 17 pinned parser and validator configurations; most pairs disagree on some inputs, and even two official MCP SDKs accept different inputs. On tasks from $\tau$-bench and AgentDojo, an attacker who can rewrite the bytes in transi
    
[^7]: AgentBouncer：通过动态系统调用中介防御软件工程智能体

    AgentBouncer: Defending Software Engineering Agents via Dynamic System-Call Mediation

    [https://arxiv.org/abs/2611.30621](https://arxiv.org/abs/2611.30621)

    AgentBouncer是一种运行时强制执行系统，利用LLM仲裁器从必要性和安全性两个维度中介（必要时拒绝）软件工程智能体触发的系统调用，从而有效防御提示注入、恶意工具和供应链攻击。

    

    软件工程（SE）智能体可以通过与开发环境交互来完成多样化的工程任务，这些交互借助访问文件、运行命令和下载资源的工具来实现。然而，正是这种使SE智能体有用的能力，也使它们容易通过提示注入、恶意工具或供应链攻击而执行非预期的恶意操作。现有的分析智能体轨迹的防御方法可能无法检测到此类攻击，因为它们仅检查初始提示和工具调用序列，而不检查在环境中实际执行的操作。本文提出了AgentBouncer，这是一种新颖的运行时强制执行系统，通过中介（并在必要时拒绝）智能体通过工具调用触发的系统调用，来防止SE智能体执行恶意操作。AgentBouncer的核心是一个LLM仲裁器，它从两个维度判断系统调用：必要性和对当前任务的安全性。

    arXiv:2611.30621v1 Announce Type: cross  Abstract: Software engineering (SE) agents can complete diverse engineering tasks by interacting with the development environment via tools that access files, run commands, and download resources. Unfortunately, the same ability that makes SE agents useful also makes them prone to performing unintended malicious actions through prompt injections, malicious tools, or supply-chain attacks. Existing defenses that analyze the agent trajectory may fail to detect such attacks, as they inspect only the initial prompt and the sequence of tool calls, but not the actual actions that are executed in the environment. This paper presents AgentBouncer, a novel runtime enforcement system that prevents SE agents from performing malicious actions by mediating, and if necessary denying, system calls that agents trigger via tool calls. The core of AgentBouncer is an LLM arbiter that judges system calls along two axes, necessity and safety for the current task. It 
    
[^8]: POMA：超越最终Flag——漏洞利用任务中基础大语言模型的过程级评估

    POMA: Beyond Final Flags -- Process-Level Evaluation of Base LLMs in Vulnerability Exploitation

    [https://arxiv.org/abs/2611.30590](https://arxiv.org/abs/2611.30590)

    POMA提出了一种过程级评估方法与平台，通过统一执行基线（固定任务输入、工具接口、目标环境、反馈格式与执行预算）隔离评估基础LLM，并在55个CTF Pwn挑战上基于运行时证据测量中间状态与状态转换，从而超越以最终成败Flag为中心的传统评测方式。

    

    随着大语言模型（LLM）从文本生成演进为自主软件工程执行，自动化漏洞可利用性验证已成为软件质量保障与安全验证领域的重要前沿。然而，现有安全基准大多以二元的通过/失败结果或粗糙的阶段里程碑来评判模型。这种以结果为中心的评分方式掩盖了长程执行过程中细粒度的失败动态，并将基础LLM的固有能力与外部智能体脚手架的作用混为一谈。我们提出了POMA（Pwn-Oriented Model Assessment，面向Pwn的模型评估），这是一种过程级评估方法与平台，它在统一的执行基线下对基础LLM进行隔离评估——固定任务输入、工具接口、目标环境、反馈格式和执行预算——同时从运行时证据中测量中间状态与状态转换。该方法在55个CTF Pwn挑战题上对四个前沿基础LLM进行了评估……

    arXiv:2611.30590v1 Announce Type: cross  Abstract: As large language models (LLMs) evolve from text generation to autonomous software-engineering execution, automated vulnerability exploitability verification has emerged as a vital frontier in software quality assurance and security validation. Existing security benchmarks, however, largely judge models by binary pass/fail outcomes or coarse stage milestones. Such outcome-centric scoring obscures fine-grained failure dynamics across long-horizon execution and conflates the intrinsic capability of a base LLM with external agent scaffolding. We present POMA (Pwn-Oriented Model Assessment), a process-level evaluation methodology and platform that isolates base LLMs under a unified execution baseline -- fixing task inputs, tool interfaces, target environments, feedback formats, and execution budgets -- while measuring intermediate states and transitions from runtime evidence. Evaluating four frontier base LLMs across 55 CTF Pwn challenges 
    
[^9]: 基于因果链诊断改进修复智能体

    Improving Repair Agents with Cause-Chain Diagnoses

    [https://arxiv.org/abs/2611.30574](https://arxiv.org/abs/2611.30574)

    FORECAST 通过比较通过测试与失败测试的执行推断出解释失败原因的“因果链”，并将其附加到问题描述中，从而无需修改智能体即可显著提升各类修复智能体在 SWE-bench Lite 上的问题解决率。

    

    修复智能体正在解决越来越多的现实世界问题。它们从与阅读缺陷报告的开发者相同的信息出发，需要在真正修复之前先弄清楚程序是如何失败的。我们提出了 FORECAST，它为修复智能体提供故障诊断。通过比较通过测试与失败测试的执行，FORECAST 推断出一条“因果链”——一段简短的执行特征序列（例如条件判断或数据流），用以解释失败运行是如何产生分歧的。FORECAST 将这条因果链附加到智能体接收的问题描述中，因此它可以与任何基于问题描述构建提示词的智能体协同工作，而无需修改智能体本身。我们在 SWE-bench Lite 的全部 300 个问题上，使用五个修复智能体和两个模型（商业模型 Claude Haiku 4.5 与开源权重模型 Qwen3.5）对 FORECAST 进行了评估。借助因果链诊断，每个智能体解决的问题数量都显著增加。在所有智能体上，已解决问题的比例从 48.5%（摘要在此处截断，应为提升至某个更高的百分比）。

    arXiv:2611.30574v1 Announce Type: new  Abstract: Repair agents resolve a growing share of real-world issues. Starting with the same information as a developer reading a bug report, they have to find out how the program fails before actually fixing it.   We present FORECAST, which provides repair agents with a diagnosis of the failure. From the executions of passing and failing tests, FORECAST infers a cause chain, a short sequence of execution features, such as conditions or data flow, that explains how the failing runs diverge. FORECAST appends this cause chain to the issue an agent receives. It thus works with any agent that builds its prompts from the issue, without modifying the agent.   We evaluate FORECAST on all 300 issues of SWE-bench Lite with five repair agents and two models, the commercial Claude Haiku 4.5 and the open-weight Qwen3.5. With the cause-chain diagnosis, every agent resolves significantly more issues. Over all agents, the share of resolved issues rises from 48.5
    
[^10]: sudo L7：编程智能体能像资深工程师一样工作吗？

    sudo L7: Can Coding Agents Act Like Staff Engineers?

    [https://arxiv.org/abs/2611.30517](https://arxiv.org/abs/2611.30517)

    该论文提出了 sudo L7 基准，包含 60 个由资深工程师在真实生产代码库中设计的软件工程任务，用于评估编程智能体能否像资深工程师一样发现隐藏风险、质疑缺陷前提并诚实汇报，而不仅仅是实现功能正确性。

    

    编程智能体现在能够将一个明确指定的任务单转化为通过测试的代码补丁。而经验丰富的工程师还会注意到请求中从未提及的风险、质疑有缺陷的前提、复用已有的成果，并坦率地说明自己验证了什么、没有验证什么。我们提出了 sudo L7，这是一个由资深工程师编写的 60 个软件工程任务组成的基准测试，这些任务大多来自私有生产代码库，且每个任务都围绕一个前沿模型曾经失败的场景构建。每个任务将一个真实的、规范不完整的请求与一份隐藏的、由专家撰写的原子化评估标准（共计 1,287 条）配对，这些标准被标记为八个行为维度，涵盖从功能正确性到真实汇报等方面。一个智能体评估器会阅读智能体的交互记录、检查并运行其代码，并将测试输出作为证据。一次尝试只有在满足两个最高严重级别的所有标准时才算通过，而这些门槛标准中有一半以上考察的是功能正确性之外的能力。

    arXiv:2611.30517v1 Announce Type: cross  Abstract: Coding agents can now turn a well-specified ticket into a patch that passes its tests. Experienced engineers also notice risks the request never mentioned, question flawed premises, reuse what already exists and say plainly what they did and did not check. We present sudo L7, a benchmark of 60 software-engineering tasks written by staff-level engineers, mostly in private production codebases and each built around a place where a frontier model failed. A task pairs a realistic, underspecified request with a hidden, expert-authored rubric of atomic criteria (1,287 in all), tagged with eight behavioral dimensions from functional correctness to truthful reporting. An agentic grader reads the agent's transcript, inspects and runs its code, and treats test output as evidence. An attempt passes only if it meets every criterion of the two highest severities, and more than half of these gating criteria test something other than functional corre
    
[^11]: 当AI智能体在数据库中犯错时，什么决定了损害程度？

    When an AI agent makes a mistake in a database, what decides the damage?

    [https://arxiv.org/abs/2611.30483](https://arxiv.org/abs/2611.30483)

    该论文通过两层预注册实验——脚本化执行的22种典型错误库和由六个小型开源模型驱动的最小工具调用智能体——系统研究了当AI智能体在数据库中犯错时，损害程度究竟由模型、防护层还是数据库引擎本身决定。

    

    AI智能体现在会对生产数据库执行语句，随后发生的几起事故删除了生产数据，这些数据只能通过备份或提供商的帮助才得以恢复。我们探究当这样的智能体犯错时，什么决定了损害程度：是模型本身、数据库前面的防护层，还是数据库引擎本身。该研究包含两个预注册的实验层。在A层中，一个包含22种典型错误的固定错误库由脚本执行（不涉及模型），在五种权限配置和两种恢复配置下应用于四个关系型引擎，随后进行引擎原生的恢复演练（792个实验单元；874次演练运行，包括10分钟的重复运行）。在B层中，一个由六个小型开源权重模型（4B至12B参数，外加一个约3B激活参数的35B混合专家模型，均本地部署）驱动的最小化工具调用智能体，在九个引擎上执行包含危险隐患的任务：PostgreSQL、MySQL、SQLite、ClickHouse、Apache Cassandra、Mon…（原文摘要在此处被截断）

    arXiv:2611.30483v1 Announce Type: new  Abstract: AI agents now run statements against production databases, and several of the incidents that followed deleted production data, which came back only through backups or the provider's help. We ask what decides the damage when such an agent makes a mistake: the model, a guard layer in front of the database, or the database engine itself. The study has two preregistered layers. In Layer A, a frozen library of 22 canonical mistakes is applied by script, with no model involved, to four relational engines under five permission arms and two recovery arms, followed by an engine-native recovery drill (792 cells; 874 drill runs including 10-minute repeats). In Layer B, a minimal tool-calling agent driven by six small open-weight models (4B to 12B parameters, plus one 35B mixture-of-experts model with about 3B active, served locally) works on tasks that contain hazards across nine engines: PostgreSQL, MySQL, SQLite, ClickHouse, Apache Cassandra, Mon
    
[^12]: LeanReviewBench：数学Lean项目中的语义对齐审查

    LeanReviewBench: Semantic Alignment Review in Mathematical Lean Projects

    [https://arxiv.org/abs/2611.30361](https://arxiv.org/abs/2611.30361)

    提出了LeanReviewBench——一个包含来自18个真实Lean项目的197个标注输入的项目级基准，用于评估语言模型审查非形式化数学目标与形式化Lean命题之间语义对齐的能力。

    

    Lean能够验证一个证明是否确立了其形式化命题，但该命题是否保留了非形式化数学源的内容则需要单独的审查。在项目规模的形式化工作中，这种对应关系可能依赖于分布在众多声明中的定义与约束。我们提出了LeanReviewBench，一个用于项目级语义对齐审查的基准，其中包含从18个真实Lean项目中收集的197个带版本信息的输入，共有87个对齐参考判断和110个未对齐参考判断。每个输入将一个非形式化数学目标绑定到选定的形式化声明以及提取出的依赖上下文。该数据集的构建结合了开发历史发现、独立目标绑定、在统一Lean环境下的重构，以及Astra标注并对不确定情况进行人工审查。该数据集涵盖多样化的数学领域，并包含69个历史配对。我们使用Dire（原文在此处截断）评估了六个语言模型……

    arXiv:2611.30361v1 Announce Type: new  Abstract: Lean verifies that a proof establishes its formal proposition, but whether that proposition preserves the mathematical content of an informal source requires separate review. In project-scale formalizations, this correspondence can depend on definitions and constraints distributed across many declarations. We introduce LeanReviewBench, a benchmark for project-level semantic alignment review containing 197 versioned inputs collected from 18 real Lean projects, with 87 aligned and 110 misaligned reference judgments. Each input binds an informal mathematical target to selected formal declarations and an extracted dependency context. Construction combines development-history discovery, independent target binding, reconstruction under a common Lean environment, and Astra annotation with human review of uncertain cases. The collection spans diverse mathematical domains and includes 69 historical pairs. We evaluate six language models with Dire
    
[^13]: RepoClean：评估编码智能体在仓库级过时代码清理上的能力

    RepoClean: Evaluating Coding Agents on Repository-Level Obsolete Code Cleanup

    [https://arxiv.org/abs/2611.30350](https://arxiv.org/abs/2611.30350)

    提出了RepoClean基准，通过将真实仓库历史中已删除的代码自动重新注入新版本，构建了200个仓库级清理任务，以评估编码智能体安全识别并移除过时代码的能力。

    

    大型语言模型（LLM）智能体已展现出强大的仓库级编程能力。然而，随着软件的不断演进，仓库代码可能变得冗余或过时，从而增加维护成本，并可能引入功能性问题。因此，智能体必须能够安全地识别并移除不必要的代码，但自主的仓库级代码清理能力尚未得到充分的独立评估。为评估这一能力，我们提出了RepoClean——一个包含来自34个热门开源Python仓库的200个清理任务的基准测试。RepoClean通过将真实仓库历史中已删除的代码自动重新引入较新版本的方式构建，同时保留当前的实现，并保留那些先前通过的测试因此失败的情形。为了在不开发新功能的前提下恢复现有功能，智能体必须利用仓库接口和跨文件依赖关系来判断哪些历史代码应当被移除。

    arXiv:2611.30350v1 Announce Type: cross  Abstract: Large language model (LLM) agents have demonstrated strong repository-level programming capabilities. However, as software evolves, repository code may become redundant or obsolete, increasing maintenance costs and potentially introducing functional issues. Agents must therefore identify and remove unnecessary code safely, yet autonomous repository-level code cleanup remains insufficiently evaluated in isolation. To evaluate this capability, we introduce RepoClean, a benchmark of 200 cleanup tasks from 34 popular open-source Python repositories. RepoClean is constructed by automatically reintroducing deleted code from real repository histories into newer versions, preserving current implementations and retaining instances where previously passing tests consequently fail. To restore existing functionality without new feature development, agents must use repository interfaces and cross-file dependencies to determine which historical code
    
[^14]: C++26 中字段枚举的编译成本：GCC 16 上的原生反射、结构化绑定包、反射库与手写序列化器

    What Field Enumeration Costs to Compile in C++26: Native Reflection, Structured-Binding Packs, Reflection Libraries and Hand-Written Serializers on GCC 16

    [https://arxiv.org/abs/2611.30250](https://arxiv.org/abs/2611.30250)

    该论文首次系统测量了 C++26 各类字段枚举方案（原生反射、结构化绑定包、反射库与手写序列化代码）在 GCC 16 上的编译开销，发现基于 span 的原生反射枚举比结构化绑定包慢约 7.96 倍，而改用常量数组循环可将开销降低约一半以上。

    

    C++26 增加了枚举结构体字段的标准方法，其中包括静态反射（P2996，附带 P3394 注解）和结构化绑定包（P1061），此外还有手写代码、X 宏以及目前同时提供 C++26 之前路径和 C++26 路径的各类库（Boost.PFR、reflect-cpp、Glaze）。我们在 GCC 16.2.0 上，在输出字节完全相同的结构体序列化器中对每种方法进行了计时，测量分别在前端和 -O2 阶段进行，采用随机交错轮次，并通过代码级检查验证每一个单因素对比。在前端，以 P2996 自身多个示例中所用形式呈现的内联原生枚举——即对 define_static_array 的 std::span 进行展开的展开语句——对于具有 16 个标量字段的结构体，其单次成本大约与 Boost.PFR 相当，是 P1061 包的 7.96 倍，而 P1061 包本身与手写代码一样廉价。对常量数组执行相同循环的成本仅为 0.47 倍：span 形式约一半的成本来自 span 本身。在异构类型图上以及……（原文在此处被截断）

    arXiv:2611.30250v1 Announce Type: cross  Abstract: C++26 adds standard ways to enumerate a struct's fields, among them static reflection (P2996, with P3394 annotations) and structured-binding packs (P1061), alongside hand-written code, X-macros and libraries (Boost.PFR, reflect-cpp, Glaze) that now ship pre-C++26 and C++26 paths. We time each in a struct serializer with byte-identical output on GCC 16.2.0, at the front end and at -O2, in randomized interleaved rounds with code-level checks of every single-factor contrast. At the front end, inline native enumeration in the form used in several of P2996's own examples, an expansion statement over the std::span of define_static_array, costs about as much per struct of 16 scalar fields as Boost.PFR and 7.96x a P1061 pack, itself about as cheap as hand-written code. The same loop over a constant array costs 0.47x as much: about half of the span form's cost comes with the span. The ratio to P1061 shrinks on a heterogeneous type graph and at 
    
[^15]: 作为制品的软件工程：从引擎到能够自我构建的系统

    Software-Engineering-as-an-Artifact: From Engines to Systems That Build Themselves

    [https://arxiv.org/abs/2611.30247](https://arxiv.org/abs/2611.30247)

    该论文借鉴未来学的情景规划方法，提出一个极端情景：软件工程流程将完全物化为制品（如上下文文档和可执行技能），最终形成能从需求自主生成和更新软件的软件工程引擎，人类工程师的角色将从开发软件产品转向维护这些流程制品。

    

    随着生成式AI和基于大语言模型（LLM）的智能体能力不断增强，软件工程（SE）正在迅速演变。然而，作为一个研究社区，我们面临着用这些新工具去解决一个不断变化的软件工程流程可能不再存在的问题的风险。我们探讨了AI可能如何变革——部分已经在变革——软件工程流程及其产出的系统。借鉴未来学研究中的情景规划方法，我们从观察到的趋势外推到一个刻意设定的极端情景：在该情景中，软件工程流程完全物化为制品（如上下文文档和可执行技能）。这样的流程最终将催生软件工程引擎，能够直接根据需求和用户请求自主生成并更新系统。届时，人类工程师的工作将随之从产品转向流程：他们将维护这些引擎的制品，而由引擎来生产和演进软件。我们通过情景验证访谈来探究这一情景。

    arXiv:2611.30247v1 Announce Type: new  Abstract: Software engineering (SE) is evolving rapidly as generative AI and LLM-based agents grow more capable. Yet, as a community, we risk using these new tools to solve problems that a changing SE process may no longer pose. We consider how AI may transform, and in part already is transforming, both the SE process and the systems it produces. Drawing on scenario planning methods from futures studies, we extrapolate from observed trends to a deliberately extreme scenario in which the SE process fully materializes in artifacts (like context documentation and executable skills). Such a process would culminate in SE engines that autonomously produce and update systems directly from requirements and user requests. The work of human engineers would then shift from the product to the process: they would maintain the artifacts of these engines, while the engines produce and evolve the software. We probe this scenario in scenario-validation interviews 
    
[^16]: 什么才是训练编码智能体的好数据？基于SWE-Gym的系统性研究

    What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym

    [https://arxiv.org/abs/2611.30207](https://arxiv.org/abs/2611.30207)

    本研究系统审计了SWE-Gym数据集奖励标签的正确性，发现模型从未解决的任务中74%存在拒绝正确修复的假阴性测试，修复或移除这些错误标注任务后，训练出的模型在SWE-bench Verified和SWE-bench Pro上分别提升2.4分和2.7分。

    

    强化学习（RL）通过软件任务来训练编码智能体，其奖励由隐藏测试决定，然而现有任务集合仅针对参考修复方案来验证这些测试。我们探究什么使一个任务成为好的训练数据，并重点关注其奖励标签的正确性。我们使用两个大语言模型系列对SWE-Gym进行了审计，且只有当执行能够复现疑似错误时才将其计为错误。我们发现“未被解决”往往意味着“被错误标注”：在一个9B基础模型从未解决的任务中，74%的任务存在会拒绝正确修复方案的测试（即假阴性），而在其偶尔能解决的任务中这一比例仅为20%。我们修复或移除了这类任务，审计后保留的任务能够训练出更好的策略。在相同规模和匹配难度下，用这些任务训练的模型在SWE-bench Verified和SWE-bench Pro的保留任务上分别比用原始任务训练的模型高出2.4分和2.7分，而简单地删除从未被解决的任务则无法带来这样的提升。

    arXiv:2611.30207v1 Announce Type: cross  Abstract: Reinforcement learning (RL) trains coding agents on software tasks whose hidden tests decide the reward, yet task collections check these tests only against the reference fix. We ask what makes a task good training data, focusing on the correctness of its reward labels. We audit SWE-Gym with two LLM families, and a suspected error counts only if execution reproduces it. We find that unsolved often means mislabelled: 74% of the tasks that a 9B base model never solves have a test that rejects correct fixes (a false negative), versus 20% of those it sometimes solves. We repair or remove such tasks, and the audited tasks that remain train better policies. At equal size and matched difficulty, models trained on them score 2.4 and 2.7 points higher on held-out tasks from SWE-bench Verified and SWE-bench Pro than models trained on the original tasks, whereas simply dropping the never-solved tasks shows no such gain. We further find that wrong
    
[^17]: VADE：一个源自专业来源的可验证智能体数据引擎

    VADE: A Verifiable Agent Data Engine from Professional Sources

    [https://arxiv.org/abs/2611.30202](https://arxiv.org/abs/2611.30202)

    VADE是一个数据引擎，它利用专业分析“求解难、验证易”的非对称性，从法律、金融等专业实践的公开来源中自动构建可验证的分析任务，从而解决这些领域高质量智能体训练数据难以规模化的问题。

    

    语言智能体已经展示了其实用价值，尤其是在软件工程领域，那里丰富的公开代码和可执行测试支持了大规模的训练与评估。然而，将这一成功扩展到其他专业领域（如法律和金融）并非易事：许多智能体任务涉及开放式的分析，其结果难以验证，因而也难以用作可扩展的监督信号。因此，这些领域中的高质量任务仍主要由专家手工编写，导致扩展成本高昂，限制了其用于训练的价值。为了解决这一瓶颈，我们提出了VADE——一个能够从专业实践中使用的公开来源自动构建可验证分析任务的数据引擎。VADE利用了一个简单的非对称性：专业分析之所以难以求解，是因为相关证据必须先被找到并加以解读；而一旦证据被固定，验证就变得容易得多。在任务合成过程中

    arXiv:2611.30202v1 Announce Type: new  Abstract: Language agents have shown their practical value, particularly in software engineering, where abundant public code and executable tests support training and evaluation at scale. Extending this success to other professional domains, such as law and finance, is not straightforward: many agent tasks involve open-ended analysis, making outcomes hard to verify and, in turn, hard to use as scalable supervision. As a result, high-quality tasks in these domains are still largely expert-authored, making them expensive to scale and limiting their use for training. To address this bottleneck, we introduce VADE, a data engine that automatically constructs verifiable analysis tasks from public sources used in professional practice. VADE exploits a simple asymmetry: professional analyses are difficult to solve because the relevant evidence must first be found and interpreted, but much easier to verify once that evidence is fixed. During task synthesis
    
[^18]: AEGIS：面向需求-代码可追溯性的证据引导集合决策

    AEGIS: Evidence-Guided Set Decisions for Requirement-to-Code Traceability

    [https://arxiv.org/abs/2611.30093](https://arxiv.org/abs/2611.30093)

    提出AEGIS框架，将检索证据与代码关系相结合，在项目级审查预算下通过局部搜索、动态规划和调用上下文准入机制，为需求构建并修订代码可追溯集合，从而在有限的审查成本下提升可追溯性恢复质量。

    

    需求到代码的可追溯性恢复旨在识别实现某个需求的代码工件，但人工审查成本限制了开发者能够检查的候选数量。代码关系可以揭示文本检索遗漏的候选，但也可能引入与需求无关的代码。我们提出AEGIS，它在项目级审查预算下结合检索证据和代码关系来构建和修订可追溯集合。分组检索和语义检索提供候选。集合目标函数将需求相关性同结构兼容性相结合。局部搜索构建不同规模的集合，动态规划在预算内为每个需求选择一个集合。随后，调用上下文准入利用涉及受保护成员的调用中参数和返回值的证据来比较替换项，同时约束检索分数的损失。在13个项目上，AEGIS在……上取得了最高的项目宏平均（摘要原文在此处截断）。

    arXiv:2611.30093v1 Announce Type: new  Abstract: Requirement-to-code traceability recovery identifies code artifacts that implement a requirement, but manual review costs limit how many candidates developers can inspect. Code relations can reveal candidates missed by textual retrieval, yet can also introduce code unrelated to the requirement. We propose AEGIS, which combines retrieval evidence and code relations to construct and revise trace sets under a project-wide review budget. Grouped retrieval and semantic retrieval supply candidates. A set objective combines relevance to the requirement with structural compatibility. Local search constructs sets of different sizes, and dynamic programming selects one set per requirement within the budget. Call-context admission then compares replacements using evidence from arguments and return values in calls involving protected members, while bounding retrieval-score loss. Across 13 projects, AEGIS achieves the highest project-macro average on
    
[^19]: 基于多模态智能体框架的二维量子器件设计中的领域优势

    Domain advantage in two-dimensional quantum-device design with multi-modal agentic harness

    [https://arxiv.org/abs/2611.30043](https://arxiv.org/abs/2611.30043)

    该论文提出了qlaybench基准测试和qlaybot多模态智能体框架，使语言模型智能体能够模拟人类工作流程，将显微镜图像转化为适配样品的二维量子器件光刻版图，证明了领域定制的多模态智能体设计相较于传统工具具有显著优势。

    

    高质量的二维量子器件通常由机械剥离的晶体加工而成，由于晶体的不规则形状和位置，需要针对每个样品定制光刻版图。而这些版图的绘制在很大程度上仍依赖人工。我们提出了qlaybench，一个测试语言模型智能体能否将显微镜图像和实验需求转化为符合样品的器件版图的基准测试。配备传统工具的智能体在给定材料轮廓时通常能够构建器件，但难以从图像中自行识别这些轮廓。qlaybot是一个仿照人类工作流程设计的多模态智能体框架，将图像解读与对齐图像、绘制器件特征以及检查所得版图的工具相连接。我们借鉴实际设计经验中的检查方法来评估每个版图，检验其是否符合样品并正确连接了预期部件。通过投入更多计算资源，……（摘要原文在此处截断）

    arXiv:2611.30043v1 Announce Type: cross  Abstract: High-quality two-dimensional quantum devices are commonly fabricated from mechanically exfoliated crystals, whose irregular shapes and placement require a lithographic layout tailored to each specimen. Drawing these layouts remains largely manual. We introduce qlaybench, a benchmark that tests whether language-model agents can turn microscope images and experimental requirements into device layouts that fit the specimen. Agents with conventional tools often construct devices when material outlines are supplied, but struggle to identify those outlines from images. Modelled on the human workflow, qlaybot, a multimodal agentic harness, connects image interpretation to the tools used to align images, draw device features and inspect the resulting layout. We evaluate each layout with checks drawn from practical design experience, asking whether it fits the specimen and connects its intended components. With greater computational effort, an 
    
[^20]: STUN-Bench：用于评估共享工具经验的真正未见工具基准

    STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience

    [https://arxiv.org/abs/2611.30034](https://arxiv.org/abs/2611.30034)

    该论文提出STUN-Bench基准，构建了经验池中完全不包含172个保留工具任何信息的数据分割，并揭示了共享经验会挤占工具描述排名、导致真正未见的工具从候选列表中消失的“拥挤”问题。

    

    工具路由器为每个请求检索少量候选工具，并让语言模型进行选择。跨组织共享经验的方法会将关于每个工具的历史请求添加到路由器检索的内容中。StableToolBench 将其中某些测试组称为对“未见”工具的测试，然而检索器和经验共享方法从中学习的 ToolBench 指令池，却包含了这两个单工具测试组中的每一个测试查询，并标记了其中所有的黄金工具。我们发布了 STUN-Bench（StableToolBench, Truly UNseen），这是一个新的数据分割，其经验池中不存在任何标记或命名172个保留工具的条目。这里的“未见”指的是在共享经验和检索器训练数据中缺失，而非在预训练数据中缺失。在该分割数据集上，我们发现了“拥挤”现象：从五个模拟客户端汇集的经验被添加到工具描述索引后，其排名超过了没有任何客户端使用过的工具的描述，导致这些工具经常从候选列表中被淘汰。在 Jina-v2 检索器下……

    arXiv:2611.30034v1 Announce Type: cross  Abstract: A tool router retrieves a few candidate tools per request and lets a language model choose. Methods that share experience across organizations add past requests about each tool to what the router retrieves. StableToolBench calls some of its test groups tests on "unseen" tools, yet the ToolBench instruction pool, which retrievers and experience-sharing methods learn from, contains every test query and labels every gold tool of the two single-tool groups among them. We release STUN-Bench (StableToolBench, Truly UNseen), a split whose experience pool has no item that labels or names 172 held-out tools. Unseen here means absent from shared experience and retriever training data, not from pretraining. On the split we find crowding: experience pooled from five simulated clients, added to an index of tool descriptions, outranks the descriptions of tools no client has used, which then often drop out of the candidate list. Under the Jina-v2 ret
    
[^21]: OWE-Bench：一个用于衡量真实世界工程能力的长时程基准

    OWE-Bench: A long-horizon benchmark for measuring real-world engineering

    [https://arxiv.org/abs/2611.30017](https://arxiv.org/abs/2611.30017)

    OWE-Bench是一个由领域专家设计的、包含26个长时程真实工程任务（涵盖软件、物理、机器学习和科学计算）的基准，用于评估AI模型的实际工程能力，目前最佳前沿模型得分仅为54.6%。

    

    我们提出了OWE-Bench（OpenWorldEngineer-Bench），这是一个包含26个长时程工程任务的基准，这些任务由软件工程、物理工程、机器学习和科学计算领域的专家撰写。任务范围从优化内核、设计飞机，到识别引发地震的断层。每个任务都需要大量的领域知识和工程工作，模型平均每个任务消耗数亿个token。解决方案根据速度、成本和准确性等实用指标进行评分，既包括智能体可以测试的给定用例，也包括工程师从任务描述中可以预期的隐藏用例。目前表现最好的前沿模型Claude Fable 5.2得分为54.6%，平均每个任务消耗2.11亿个token。我们强调了两个发现：首先，当模型获得更多时间时，它们会使用更多的测试时计算并获得更高的分数；其次，模型在……方面的扩展方式各不相同（原文在此截断）。

    arXiv:2611.30017v1 Announce Type: new  Abstract: We present OWE-Bench (OpenWorldEngineer-Bench), a benchmark of 26 long-horizon engineering tasks written by domain experts in software engineering, physical engineering, machine learning, and scientific computing. Tasks range from optimizing kernels and designing an aircraft, to identifying the faults responsible for an earthquake. Each task requires significant domain knowledge and engineering work, with models averaging hundreds of millions of tokens per task. Solutions are scored on practical measures such as speed, cost and accuracy, both on provided cases the agent could test against and on hidden cases that an engineer would expect from the task description. The best-performing frontier model so far, Claude Fable 5.2, scores 54.6% and averages 211 million tokens per task. We highlight two findings. First, when models are given more time, they use more test-time compute and achieve higher scores. Models also scale differently with t
    
[^22]: LLM辅助的软件迁移：从模块级转换到仓库级集成

    LLM-Assisted Software Migration: from Module-Level Transformation to Repository-Level Integration

    [https://arxiv.org/abs/2611.29942](https://arxiv.org/abs/2611.29942)

    该综述通过分析96篇文献（含46篇直接迁移论文），从软件工程活动、模型角色、编排工作流和使能技术四个视角，系统梳理了LLM辅助软件迁移从模块级转换走向仓库级集成并交付完整应用的研究现状，并比较了现有评估实践与基准。

    

    软件迁移是在保持原有行为的前提下，将现有实现适配到新的语言、版本、API、框架或平台。大型语言模型（LLM）在单个函数和模块上取得了成功，但这种成功并不能直接延伸到整个代码仓库，尤其是可部署的应用程序。本综述从模块级转换到仓库级集成、直至完整应用程序交付的全过程，对LLM辅助的软件迁移进行了回顾。我们从四个视角分析了96篇文献（其中包括46篇直接涉及迁移的论文）：（1）所支持的软件工程活动；（2）模型作为编辑器或编排者的角色；（3）编排工作流（无智能体、单智能体或多智能体）；（4）使能技术，包括知识工程、项目上下文、环境反馈和模型适配。随后，我们比较了相关的评估实践与基准测试。尽管46项研究中有30项的目标集中于……

    arXiv:2611.29942v1 Announce Type: cross  Abstract: Software migration adapts an existing implementation to a new language, version, API, framework, or platform while preserving its behavior. Large language models (LLMs) succeed on individual functions and modules, but this success does not readily carry over to entire repositories, particularly deployable applications. This survey reviews LLM-assisted migration from module-level transformation to repository-level integration, culminating in the delivery of complete applications. We analyze 96 sources---including 46 direct migration papers---across four perspectives: (1) supported software engineering activities, (2) model roles as editors or orchestrators, (3) orchestration workflows (agentless, single-agent, or multi-agent), and (4) enabling techniques including knowledge engineering, project context, environment feedback, and model adaptation. We then compare evaluation practices and benchmarks. Although 30 of the 46 studies target r
    
[^23]: 智能体的地图与实际疆域是否相符？AI智能体上下文文件中的架构声明及其与代码的契合度

    Does the Agent's Map Match the Territory? Architectural Claims in AI Agent Context Files and Their Fit with the Code

    [https://arxiv.org/abs/2611.29916](https://arxiv.org/abs/2611.29916)

    本研究首次大规模验证了AI智能体上下文文件中架构声明与实际代码的相符程度，发现路径声明真实错误率较低（约2.2%）且会随代码演进及时修复，但依赖规则中有18%在编写当天即告失效且几乎从不会被修复。

    

    AI编码智能体在每次会话开始时都没有关于系统如何构建的记忆。开发者使用上下文文件（如CLAUDE.md和AGENTS.md）来填补这一空白。由于智能体会将这些文件的内容视为真实可信，因此这些文件是否与代码相符就显得十分重要。我们研究了两种可以对照代码仓库进行验证的架构声明：路径声明（指明文件夹和文件）和依赖规则（说明什么可以导入什么）。我们对照代码及其历史记录，验证了来自3,346个开源项目中3,602个上下文文件的41,090条路径声明和184条依赖规则。约5%的路径声明指向我们无法找到的路径，但我们估计其中只有2.2%是真正的错误，这一比例与README文件（3.5%）相近。当代码位置发生变化时，61%变得过时的声明会被修复，且大多在同一次提交中完成。依赖规则的表现则截然不同：184条规则中有33条（18%）在编写当天就被确认已失效，通常是被较旧的代码所违反，而且几乎所有失效的规则都一直保持着失效状态。

    arXiv:2611.29916v1 Announce Type: new  Abstract: AI coding agents start every session without memory of how a system is built. Developers fill this gap with context files, such as CLAUDE.md and AGENTS.md. Agents treat these files as true, so it matters whether they match the code. We study two kinds of architectural claims that can be checked against a repository: path claims, which name folders and files, and dependency rules, which say what may import what. We check 41,090 path claims and 184 dependency rules from 3,602 context files in 3,346 open-source projects against the code and its history. About 5% of path claims point to paths we cannot find, but we estimate that only 2.2% are real errors, similar to READMEs (3.5%). When code moves, 61% of claims that became stale are fixed, most in the same commit. Dependency rules behave differently: 33 of 184 (18%) were confirmed broken on the day they were written, usually by older code, and nearly all stayed broken. Context files are goo
    
[^24]: 上下文感知的主动式自适应：一种双层模型预测控制方法

    Context-Aware Proactive Self-Adaptation: A Two-Layer Model Predictive Control Approach

    [https://arxiv.org/abs/2611.29857](https://arxiv.org/abs/2611.29857)

    本文提出一种基于上下文目标模型的双层模型预测控制机制，实现了软件系统的上下文感知主动式自适应与控制器自身的自适应，提升了系统的性能与适应性。

    

    在自适应软件系统中，上下文的作用至关重要，尤其是对于主动式自适应而言。然而，当前研究并未充分探索上下文的影响，例如对需求优先级的影响。为了弥补这一空白，我们引入了一种新颖的上下文目标模型来捕捉这些因素及其对系统的影响。基于该模型，我们提出了一种双层控制机制，采用上下文感知的模型预测控制，以实现软件系统的主动式自适应以及控制器自身的自适应。通过上下文预测和更精确的系统模型，我们的方法利用模型预测控制来实现及时且高效的系统自适应，从而提升性能和适应性。同时，我们通过需求自适应来更新上下文目标模型，进而更新控制器的目标函数和约束条件。我们在……上进行了实验评估（摘要在此处截断）

    arXiv:2611.29857v1 Announce Type: new  Abstract: In self-adaptive software systems, the role of context is paramount, especially for proactive self-adaptation. Current research, however, does not fully explore context's impact, for example on priorities of the requirements. To address this gap, we introduce a novel contextual goal model to capture these factors and their influence on the system. Using this, we propose a two-layer control mechanism with a context-aware model predictive control to achieve proactive adaptation for the software system and adaptation for the controller itself. By contextual prediction and a more accurate system model, our approach utilizes model predictive control to facilitate timely and efficient system adaptations, improving both performance and adaptability. Meanwhile, we perform requirement adaptation to update the contextual goal model, which in turn updates the objective function and constraints of the controller. Our experimental evaluations across 
    
[^25]: 软件工程中提示工程模式的全景与分类法

    Landscape and Taxonomy of Prompt Engineering Patterns in Software Engineering

    [https://arxiv.org/abs/2611.29853](https://arxiv.org/abs/2611.29853)

    本文基于系统性文献综述提出了软件工程领域提示工程模式的分类法，表明即使大语言模型能力持续增强，精心设计的提示模式对于提升其可用性、可靠性和任务性能仍然至关重要。

    

    大语言模型（LLM）的进步增强了其处理模糊用户指令的能力。然而，有效的提示模式对于可用性和可理解性仍然至关重要。本文提出了一个面向软件工程的提示工程模式分类法。该分类法基于2023年初进行的一项系统性文献综述，当时大语言模型在上下文长度和推理能力方面仍存在显著局限。我们的研究探讨了提升大语言模型可用性和可靠性的技术，强调了精心设计的提示在优化任务性能方面持续的重要性。我们的发现突出了提示模式在最大化大语言模型潜力方面的关键作用，即使其能力仍在不断演进。

    arXiv:2611.29853v1 Announce Type: cross  Abstract: Advancements in large language models (LLMs) have enhanced their ability to handle ambiguous user instructions. However, effective prompt patterns remain crucial for usability and comprehension. This article presents a taxonomy of prompt engineering patterns for software engineering. It is based on a systematic literature review that was conducted in early 2023, when LLMs still faced significant limitations in context length and inference capabilities. Our study explores techniques that enhance the usability and reliability of LLMs, emphasizing the ongoing importance of well-designed prompts in optimizing task performance. Our findings highlight the critical role of prompt patterns in maximizing LLM's potential, even as their capabilities continue to evolve.
    
[^26]: 氛围编程能否产出无障碍网站：关于LLM编程智能体与技能的实证研究

    Can Vibe Coding Produce Accessible Websites: An Empirical Study on LLM Coding Agents and Skills

    [https://arxiv.org/abs/2611.29822](https://arxiv.org/abs/2611.29822)

    本文通过生成90个网站并使用多种无障碍检测工具进行评估的实证研究发现，LLM编程智能体端到端生成的网站无一完全无障碍（平均违反约80个WCAG成功标准），揭示了“氛围编程”工作流会大规模传播无障碍缺陷。

    

    大型语言模型（LLM）编程智能体日益使开发者和非专业人士能够通过自然语言指令创建网站，而对生成的源代码只需极少检查。尽管这种“氛围编程”工作流降低了软件创建的门槛，但它也可能大规模传播无障碍缺陷。我们研究了由当代编程智能体端到端生成的网站的无障碍性，以及智能体级的无障碍干预能否改善其输出。我们使用三种编程智能体、四个语言模型和多个推理强度级别的组合，为三个项目生成了90个网站。我们使用axe-core、IBM Equal Access Accessibility Checker和QualWeb评估了完整的交互旅程，并汇总报告以观察违规情况的普遍性及所违反的WCAG成功标准（SC）。90个基线网站中没有一个完全无障碍，平均违反80.43个

    arXiv:2611.29822v1 Announce Type: cross  Abstract: Large language model (LLM) coding agents increasingly allow developers and non-professionals to create websites through natural-language instructions, with limited inspection of the generated source code. Although this ``vibe coding'' workflow lowers barriers to software creation, it may also propagate accessibility defects at scale. We investigate the accessibility of websites generated end-to-end by contemporary coding agents and whether agent-level accessibility interventions improve their output. We generated 90 websites for three projects using combinations of three coding agents, four language models, and multiple reasoning-effort levels. We evaluated complete interaction journeys using axe-core, IBM Equal Access Accessibility Checker, and QualWeb, and converged reports to observe violation prevalence and violated WCAG Success Criterion (SC). None of the 90 baseline websites was fully accessible, with an average of 80.43 violated
    
[^27]: 对可修复的失败视而不见：自我改进智能体中的检测缺口

    Blind to What They Can Fix: A Detection Gap in Self-Improving Agents

    [https://arxiv.org/abs/2611.29807](https://arxiv.org/abs/2611.29807)

    研究发现自我改进智能体能写出高分程序，但其自建的运行时检查在环境变化后对一半以上的失败视而不见，而这些失败其实可通过正常运行记录检测并由智能体自身修复。

    

    AI智能体能够编写无需在每一步都调用模型的程序来执行任务，从而使任务运行快速且逻辑透明可审查。智能体通过运行程序、分析失败原因并修改代码来改进这些程序，这是一种递归自我改进（RSI）的形式。我们探究这一过程是否也能产生可信的运行时检查。在五个任务领域中，使用多达七个编写代码的模型，智能体能够将其程序提升至高分。然而当环境发生变化时，在四个领域中，它们被要求编写的检查机制对一半甚至更多的由此产生的失败保持沉默。这些失败实际上是可以被检测和修复的：基于正常运行记录构建的监视器在三个领域中捕获了其中大部分失败，而且一旦被提醒，同样的智能体通常能够修复这些问题。这些检查机制没有审查那些能够揭示失败的记录关系，而当最强的智能体被告知其检查存在遗漏时，它编写的检查便能捕获大部分失败。

    arXiv:2611.29807v1 Announce Type: new  Abstract: AI agents can write programs that carry out a task without calling a model at every step, so the task runs fast and its logic stays open to inspection. Agents improve these programs by running them, studying their failures, and revising the code, a form of recursive self-improvement (RSI). We ask whether this process also yields runtime checks that can be trusted. In five task domains, with up to seven author models, agents bring their programs to high scores. Yet when the environment changes, the checks they were told to write stay silent on half or more of the resulting failures in four domains. These failures can be detected and repaired: a monitor built from records of normal runs catches most of them in three domains, and the same agents usually repair them once called. The checks do not examine the recorded relations that reveal the failures, and the strongest agent writes checks that catch most of them once told that its checks ar
    
[^28]: 基于大语言模型起草的角色类型化信号时序逻辑规范生成安全关键驾驶场景

    Generating Safety-Critical Driving Scenarios from Role-Typed Signal Temporal Logic Specifications Drafted by Large Language Models

    [https://arxiv.org/abs/2611.29800](https://arxiv.org/abs/2611.29800)

    该论文提出由大语言模型从描述中起草按角色分类的信号时序逻辑规范，再由编译器将其转化为对扩散模型样本的最小编辑优化，从而生成安全关键驾驶场景，用于测试自动驾驶车辆的运动规划器。

    

    在记录的真实驾驶数据中，安全关键场景十分罕见，因此必须通过生成的方式来测试和训练自动驾驶车辆的运动规划器。在引导扩散模型中，每种场景类型都需要各自独立的加权目标函数，且与有效性检查相互分离。若将攻击者与被测车辆一起优化，被测车辆可能消除攻击或将攻击引向自身。我们将每种场景表述为带有按角色分类子句的信号时序逻辑规范。每个子句都指明它所改变的主体和它所读取的主体，且被测车辆仅在攻击发生后才开始读取攻击者的信息。大语言模型根据文字描述起草该规范，并由类型检查器进行验证。编译器推导出相应的优化过程：通过对扩散模型的样本进行满足规范的最小改动来编辑样本，每个子句仅移动其所属主体，攻击者在被测车辆作出响应之前先针对未响应的被测车辆进行求解……

    arXiv:2611.29800v1 Announce Type: cross  Abstract: Safety-critical scenarios are rare in recorded driving and must be generated to test and train motion planners of automated vehicles. In guided diffusion models, each scenario type needs its own weighted objective, separate from the validity check. Optimized together with the attacker, the vehicle under test can cancel the attack or draw it toward itself. We state each scenario as a signal temporal logic specification with clauses typed by role. Every clause names the agent it may change and the agents it may read, and the vehicle under test reads the attacker only from the attack onset. A large language model drafts the specification from a description, and a type checker accepts it. A compiler derives the optimization: it edits a diffusion model's samples by the smallest change that satisfies the specification, each clause moves only its owner, the attacker is solved against the unreacted vehicle under test before its response, and t
    
[^29]: 静态调用图分析中不健全性的智能体式根因推理与定位

    Agentic Root Cause Reasoning and Localization of Unsoundness in Static Call Graph Analysis

    [https://arxiv.org/abs/2611.29745](https://arxiv.org/abs/2611.29745)

    提出智能体框架 CGRizz，通过对比静态与动态调用图定位缺失的调用边，由 LLM 智能体推理不健全性的根因，并用运行时插桩、复现测试用例和工具位置警告三重独立证据加以验证，实现对静态调用图分析缺陷的诊断与自动预警。

    

    静态调用图工具无法健全地建模所有动态语言特性，却很少揭示其结果在何时、何地、为何不健全。我们提出了 CGRizz，一个为开发者诊断不健全性并向用户发出警告的智能体式框架。在训练阶段，CGRizz 通过比较静态调用图与动态调用图来识别边界边，即运行时观察到但静态图中缺失的调用。对于每一条这样的边，一个 LLM 智能体会提出根因假设，包括未被建模的程序模式以及相应的工具代码位置。CGRizz 使用三种相互独立的证据来验证每个假设：检测该模式的运行时插桩、重现缺失边的测试用例，以及在涉事工具位置插入的警告。经过验证的报告帮助开发者诊断工具的局限性，而这些警告则构成一个插桩构建版本，无需动态分析即可在新程序中标记潜在的不健全性。我们在三个基准上评估了 CGRizz……

    arXiv:2611.29745v1 Announce Type: new  Abstract: Static call graph tools cannot soundly model every dynamic language feature, yet rarely reveal when, where, or why their results are unsound. We present CGRizz, an agentic framework that diagnoses unsoundness for developers and warns users about it. During training, CGRizz compares static and dynamic call graphs to identify boundary edges: calls observed at runtime but missing from the static graph. For each edge, an LLM agent proposes a root cause, including the unmodeled program pattern and responsible tool location. CGRizz validates each proposal using three independent forms of evidence: runtime instrumentation detecting the pattern, a test case reproducing the missing edge, and a warning inserted at the implicated tool location. Validated reports help developers diagnose limitations, while the warnings form an instrumented build that flags potential unsoundness in new programs without dynamic analysis. We evaluate CGRizz on three be
    
[^30]: 被采纳不等于有效：智能体流水线中公证人把关回溯循环的经验教训

    Admitted Is Not Effective: Lessons from a Notary-Gated Retrospective Loop in an Agentic Pipeline

    [https://arxiv.org/abs/2611.29727](https://arxiv.org/abs/2611.29727)

    尽管公证人把关的回溯循环能够有效治理智能体流水线中的规则更新（包括拒绝引用虚假证据的提案），但由于评审信号过于稀疏且被采纳的规则缺乏实际效果，流水线性能在十次回溯后并未改善，证明“被采纳”并不等于“有效”。

    

    具有冻结提示词的智能体软件开发流水线会在不同拉取请求中重复同样的错误，而对指令文件不加治理的追加内容会无限增长，且通常并不能提高任务成功率。我们为一个四阶段大语言模型（LLM）流水线提出了一个回溯循环：一个“回溯智能体”每十个PR对学习到的规则提出修改建议，一个“公证人”则负责采纳、合并或拒绝这些建议，并将每个决定记录在溯源账本中。在一项涵盖两个Python仓库共880个PR的预注册研究中，该循环完整地端到端运行，公证人采纳了89个提案中的55个，并拒绝了8个引用不存在证据的提案。然而，在十次回溯之后，留出的评审评论和隐藏测试的解决率均未改善。通过对循环每一步的追踪，我们发现稀疏的评审信号（每PR仅0.45条评论）使得可供学习的内容寥寥无几，而且基于其所述证据被采纳的规则并未表现出可测量的改进。

    arXiv:2611.29727v1 Announce Type: cross  Abstract: Agentic software development pipelines with frozen prompts can repeat the same mistakes across pull requests (PRs), yet ungoverned additions to instruction files grow without bound and do not generally improve task success. We present a retrospective loop for a four-stage large language model (LLM) pipeline in which a Retrospective agent proposes edits to learned rules every ten PRs and a Notary admits, merges, or rejects them while recording each decision in a provenance ledger. In a pre-registered study of 880 PRs on two Python repositories, the loop ran end to end and the Notary admitted 55 of 89 proposals, rejecting 8 that cited nonexistent evidence.   However, held-out review comments and hidden-test resolution did not improve after ten retrospectives. Tracing each step of the loop, we found that sparse review signal (0.45 comments/PR) left little to learn from and that rules admitted on their stated evidence did not show measurab
    
[^31]: ReqSeek：基于Transformer的自动需求识别方法，用于构建大规模需求数据集

    ReqSeek: Transformer-Based Automatic Requirements Identification for Building Large Requirements Datasets

    [https://arxiv.org/abs/2611.29722](https://arxiv.org/abs/2611.29722)

    本文提出了基于Transformer、领域无关的自动需求识别模型ReqSeek，并构建了包含23个异构SRS的ARID数据集，以RoBERTa为基础模型达到95%的精确率与召回率，性能可与人类媲美且耗时更少。

    

    大规模、多样化且高质量的数据集对于推动需求工程领域的人工智能发展至关重要，然而从软件需求规格说明（SRS）中自动识别需求仍是一项挑战。现有的自动需求识别（ARI）方法在异构SRS文档上对未见文档的泛化能力有限。本工作提出了ReqSeek，一个基于Transformer、与领域无关的自动需求识别模型，用于识别SRS文档中符合ISO 29148风格的需求，以支持数据集构建。我们策划了自动需求识别数据集（ARID），包含来自23个异构SRS的2,396个训练实例，并微调了四种Transformer架构。RoBERTa被选为ReqSeek的基础模型，在留出评估集上达到了95%的精确率、召回率和F1值。探索性的人机对比基准测试表明，该模型的性能与人类相当，同时所需时间显著更少（摘要在此处截断）。

    arXiv:2611.29722v1 Announce Type: cross  Abstract: Large, diverse, and high-quality datasets are significant for advancing AI in requirements engineering, yet automated identification of requirements from Software Requirement Specifications (SRS) remains a challenge. Automatic Requirements Identification (ARI) methods demonstrate limited generalization to unseen documents across heterogeneous SRSs. This work presents ReqSeek, a transformer-based, domain-agnostic ARI model for recognizing ISO 29148-style requirements in SRS documents, supporting dataset construction. We curated the Automatic Requirements Identification Dataset (ARID), comprising 2,396 training instances from 23 heterogeneous SRSs, and fine-tuned four transformer architectures. RoBERTa was selected as the base model for ReqSeek and achieved 95% precision, recall, F1, and F0.89 on the held-out evaluation set. Exploratory human-model benchmarking showed performance comparable to humans while requiring substantially less pr
    
[^32]: AutoRPG：利用系统性AI实现非侵入式低延迟的移动端RPG战斗自动化

    AutoRPG: Harnessing Systematic AI for Non-Intrusive Low-Latency Mobile RPG Combat

    [https://arxiv.org/abs/2611.29673](https://arxiv.org/abs/2611.29673)

    AutoRPG提出了一种两阶段系统性AI框架，通过高层战略规划结合一套轻量级执行模块（目标追踪、角色移动、闪避、异常处理），实现了非侵入式、低延迟且具备细粒度行为控制的移动端RPG自动战斗。

    

    移动端RPG游戏中的自动战斗在游戏质量保障中扮演着关键角色，要求实现与人类玩家操作高度相似的实时行为控制。然而，现有方法无法满足这些要求，原因要么是缺乏细粒度的行为控制能力，要么是推理延迟过高。为应对这些挑战，我们结合用户游戏体验设计了AutoRPG，这是一个用于移动端RPG战斗自动化的系统性AI框架。在高层次上，AutoRPG将战斗过程分解为两个阶段：首先是制定整体战斗策略的高层战略规划阶段，随后是由一套精心设计的轻量级模块驱动的系统性执行阶段，这些模块涵盖目标追踪、角色移动、闪避操控和异常状态处理。在《原神》、《鸣潮》和《明日方舟：终末地》三款热门移动端RPG上的实验表明，

    arXiv:2611.29673v1 Announce Type: new  Abstract: Automated combat in mobile RPGs plays a critical role in game quality assurance, requiring real-time behavior control that closely resembles human gameplay. However, existing approaches fail to satisfy these requirements, either because they lack fine-grained behavioral control or because they incur high inference latency. To address these challenges, we incorporate user gameplay experience to design AutoRPG, a systematic AI framework for automating mobile RPG combat. At a high level, AutoRPG decomposes the combat process into two stages: a high-level strategic planning phase that formulates the overall combat strategy, followed by a systematic execution phase powered by a suite of carefully designed lightweight modules encompassing target tracking, character movement, dodge maneuvering, and abnormal-state handling. Experiments on three popular mobile RPGs, i.e., Genshin Impact, Wuthering Waves, and Arknights: Endfield, demonstrate that 
    
[^33]: fx(hash)：抢救2.8万件生成艺术作品的源代码

    fx(hash): Salvaging The Source Code of 28k Generative Artworks

    [https://arxiv.org/abs/2611.29604](https://arxiv.org/abs/2611.29604)

    本研究抢救并归档了已破产的生成艺术NFT平台fx(hash)上5,532位艺术家的全部28,193个生成程序源代码至Software Heritage，揭示了NFT生成艺术因源代码存于链下而面临的脆弱性。

    

    fx(hash)是最具激进包容性的生成艺术NFT平台，于2026年8月17日破产并关闭。其2,705,350件艺术品的所有权在Tezos和以太坊区块链上不可篡改，但生成这些艺术品的源代码存储于链下且岌岌可危。我们抢救了5,532位国际艺术家贡献的全部28,193个生成程序，并将其归档于Software Heritage。本文解释了NFT生成艺术为何脆弱，以及我们为抢救这些软件文化地标所采取的具体步骤。

    arXiv:2611.29604v1 Announce Type: new  Abstract: fx(hash), the most radically inclusive generative art NFT platform, went bankrupt and shut down on August 17, 2026. The ownership of its 2,705,350 artworks is immutable on the Tezos and Ethereum blockchains, but the source code that generates them is off-chain and in peril. We salvaged all 28,193 generative programs contributed by 5,532 international artists and archived them in Software Heritage. Here we explain why NFT generative art is fragile and the steps we took to salvage these software cultural landmarks.
    
[^34]: KernelShift：面向不断演进硬件的内核优化基准测试

    KernelShift: Benchmarking Kernel Optimization Across Evolving Hardware

    [https://arxiv.org/abs/2611.29590](https://arxiv.org/abs/2611.29590)

    该论文提出了KernelShift基准测试流水线，通过从ncnn、llama.cpp等真实活跃仓库中构建99个内核优化任务，并让智能体在目标硬件反馈下优化代码，以评估编码智能体在不断演进的CPU架构扩展下编写高性能内核的能力。

    

    高效的AI计算依赖于随时间不断演进的硬件，这改变了实现高性能代码所需的策略。对于CPU内核而言，一个关键的演进因素是可用的架构扩展集合，这些扩展提供了专用的硬件指令，从而改变了优化内核所采用的技术。我们研究了编码智能体如何有效地利用这些新的、可能不熟悉的扩展来编写快速代码。我们提出了KernelShift，这是一个用于评估编码智能体在不断演进的CPU架构上进行内核优化能力的基准测试流水线和一组环境。其智能体辅助、人工验证的流水线可将代码仓库转换为基准测试实例，其环境让智能体能够根据目标硬件的反馈来优化代码。利用该流水线，我们从真实且处于活跃开发状态的仓库（ncnn、llama.cpp、Arm SIMD Loops和KleidiAI）中构建了99个内核，包括由此衍生的推理工作负载。

    arXiv:2611.29590v1 Announce Type: cross  Abstract: Efficient AI computation depends on hardware that evolves over time, shifting the strategies needed to implement performant code. For CPU kernels, a key evolving factor is the set of available architectural extensions, which expose specialized hardware instructions that change the techniques used to optimize kernels. We study how effectively coding agents use these new, potentially unfamiliar extensions to write fast code. We introduce KernelShift, a benchmarking pipeline and a set of environments for evaluating coding agents in kernel optimization across evolving CPU architectures. Its agent-assisted, human-verified pipeline turns code repositories into benchmark instances, and its environments let agents optimize code with feedback from the target hardware. Using this pipeline, we construct 99 kernels from real-world, actively developed repositories (ncnn, llama.cpp, Arm SIMD Loops, and KleidiAI), including inference workloads derive
    
[^35]: 端点条件化过程价值：当已知最终状态与成本后，盲端点过程评分还能增加什么

    Endpoint-Conditioned Process Value: What an Endpoint-Blind Process Score Adds Once Terminal Status and Cost Are Known

    [https://arxiv.org/abs/2611.29354](https://arxiv.org/abs/2611.29354)

    本研究通过1,255条轨迹的实验首次量化了过程评分在已知最终状态和成本后的增量价值，发现其增益高度依赖评判模型规模——只有32B级别的评判模型才能带来显著的AUC提升，且大部分增益可由八个可记录的行为计数器替代。

    

    轨迹度量承诺提供超越最终答案准确性的信息，但尚无研究测量过当评估已知结果和成本后，过程评分还能增加什么。我们构建了这一比较：一个需要双方都无法看到产物的密封后果实验、一个由最终状态和成本构成的操作基线，以及将评判模型规模与证据访问作为实验因素。在7个Python代码库中158个测试生成任务的1,255条轨迹上，答案是取决于评判模型的。四个4B到14B的评判模型（先推理后评分）所增加的信息与零无法区分；而在相同轨迹上，一个Qwen3 32B评判模型增加了+0.050 AUC（95%置信区间[+0.020, +0.081]），并通过了重新拟合的bootstrap检验（[+0.013, +0.095]）。它所增加的大部分信息可以从八个行为计数器中恢复，因此测试框架应当记录这些信息（在此之上评判模型仅增加+0.020）。评判模型所看到的内容会进一步改变结果：披露最终判定结论……

    arXiv:2611.29354v1 Announce Type: new  Abstract: Trajectory metrics promise information beyond final-answer accuracy, but no one has measured what a process score adds once an evaluation knows the outcome and the cost. We build that comparison: a sealed consequence needing an artifact neither agent nor judge ever sees, an operational baseline of terminal status and cost, and judge scale and evidence access as experimental factors. On 1,255 trajectories over 158 test-generation tasks in 7 Python repositories, the answer is gated on the judge. Four judges from 4B to 14B, one reasoning before scoring, add nothing an interval separates from zero; a Qwen3 32B judge on identical trajectories adds +0.050 AUC (95% CI [+0.020, +0.081]) and survives a refitting bootstrap ([+0.013, +0.095]). Most of what it adds is recoverable from eight behavioural counters, so a harness should log them (the judge adds +0.020 beyond them). What the judge is shown moves it further: disclosing the terminal verdict
    
[^36]: 多智能体协同的 Rust 代码方法级重构

    Multi-Agent Coordinated Method-Level Refactoring of Rust Code

    [https://arxiv.org/abs/2611.29321](https://arxiv.org/abs/2611.29321)

    该论文提出了一种基于大语言模型的多智能体协同工作流，通过候选生成、可配置专家审查与容器化验证三个环节，安全地实现 Rust 代码的方法提取、内联与移动重构，并在开源项目任务上首次尝试即全部通过验证。

    

    随着 Rust 代码库的演进，开发者会重构代码以减少重复、简化复杂方法，并将相关功能组织在一起。开发者可以手动进行这些更改，或使用 IDE 辅助完成受支持的操作，然后通过编译器诊断和测试来检查 resulting 代码。Rust 中的方法级重构必须在考虑所有权、借用和生命周期的同时保持可观察行为不变。我们提出了一种基于大语言模型（LLM）的工作流，用于在 Rust 中提取、内联和移动方法。该工作流结合了候选生成、可配置的专家审查以及基于容器的验证。在来自三个开源 Rust 项目的六个任务上，三种端到端配置在首次尝试中即通过了所有任务检查，分别使用了 6、13 和 31 次模型调用。为了隔离审查环节的效果，我们随后给四种配置提供了相同的六个参考补丁和六个故意改变行为的变体。（注：原始摘要在此处被截断）

    arXiv:2611.29321v1 Announce Type: new  Abstract: As Rust codebases evolve, developers refactor code to reduce duplication, simplify complex methods, and keep related functionality together. Developers can make these changes manually or use IDE assists for supported operations, then check the resulting code with compiler diagnostics and tests. Method-level refactoring in Rust must preserve observable behavior while accounting for ownership, borrowing, and lifetimes. We present a large language model (LLM)-based workflow for extracting, inlining, and moving methods in Rust. The workflow combines candidate generation, configurable specialist review,and container-based validation. On six tasks from three open-source Rust projects, three end-to-end configurations passed all task checks on their first attempt, using 6, 13, and 31 model calls. To isolate the effect of review, we then gave four configurations the same six reference patches and six deliberately behavior-changing variants. A des
    
[^37]: MetaSkill：面向LLM智能体自适应技能编排的可复用问题解决方法

    MetaSkill: Reusable Problem-Solving Methods for Adaptive Skill Orchestration in LLM Agents

    [https://arxiv.org/abs/2611.29274](https://arxiv.org/abs/2611.29274)

    MetaSkill提出一个双层框架，通过选择并实例化可复用的问题解决方法来引导LLM智能体自适应地编排技能，在SkillsBench和ToolHop基准上相比最强基线准确率提升3.51%-13.48%。

    

    可复用技能提供了专门化的操作流程，但大语言模型（LLM）智能体仍需自行决定如何组织和调整这些技能的使用。我们提出了MetaSkill，一个跨任务、跨领域复用问题解决方法的双层框架。方法层从26个精选方法中进行选择，并将抽象角色绑定到任务对象上。执行层利用实例化的方法来识别中间目标、协调原生技能，并引导修复或方法重选。在89个SkillsBench任务和995个ToolHop示例上，MetaSkill在全部六种配置中均优于所评估的基线方法，相比最强基线准确率提升3.51%-13.48%。消融实验表明，显式的方法引导编排优于直接方法注入，且方法重选能带来进一步提升。

    arXiv:2611.29274v1 Announce Type: new  Abstract: Reusable skills provide specialized procedures, but large language model (LLM) agents must still determine how to organize and adapt their use. We introduce MetaSkill, a two-layer framework for reusing problem-solving methods across tasks and domains. The method layer selects from 26 curated methods and binds abstract roles to task objects. The execution layer uses the instantiated method to identify intermediate objectives, coordinate native skills, and guide repair or method reselection. On 89 SkillsBench tasks and 995 ToolHop examples, MetaSkill outperforms the evaluated baselines in all six configurations, with accuracy gains of 3.51-13.48% over the strongest baseline. Ablations support explicit method-guided orchestration beyond direct method injection and further gains from method reselection.
    
[^38]: 编辑引发的依赖失配：编码智能体中的传播遗漏、恢复与评估盲区

    Edit-Induced Dependency Mismatch: Propagation Omissions, Recovery, and Evaluation Blind Spots in Coding Agents

    [https://arxiv.org/abs/2611.29242](https://arxiv.org/abs/2611.29242)

    该论文揭示了编码智能体在迭代改代码时会依赖上下文中残留的过时依赖判断，导致缺陷传播并逃过基准测试评估，并通过盲测人工审计确认了此类“编辑引发的依赖失配”的真实存在、运行中的恢复现象以及现有评估体系的盲区。

    

    在迭代代码修改过程中，编码智能体可能在后续推理中继续依赖过时的依赖判断，而这些判断的文本仍固定保留在上下文中，从而导致后续推理出错，并使消费者缺陷未被原始任务评估发现，进而暴露了所测基准中的盲区。我们研究了在基准任务上收集的真实智能体执行轨迹：包括 deepseek-v4-flash 在 DeepSWE 和 SWE-bench Verified 上的运行，并以 Qwen3.8-Flash 在 FeatureBench 上的案例作为补充。聚焦于编辑引发的消费者义务，对 DeepSWE 运行结果的全语料盲测人工审计识别出了终端失配，其中包括任务范围仍存争议的诊断。在补充执行检查之后，所有保留的 DeepSWE 终端事件均获得了执行层面的支持。审计还表明，在同一运行中，轨迹内的恢复可以与终端遗漏并存。过程分析将……（原文摘要至此截断）

    arXiv:2611.29242v1 Announce Type: new  Abstract: During iterative code modification, coding agents may continue to rely in subsequent reasoning on outdated dependency judgments whose text remains fixed in the context, leading to errors in subsequent reasoning and leaving consumer defects undetected by the original task evaluations, thereby exposing blind spots in the tested benchmarks. We study actual agent execution trajectories collected on benchmark tasks: deepseek-v4-flash runs on DeepSWE and SWE-bench Verified, supplemented by Qwen3.8-Flash cases on FeatureBench. Focusing on edit-induced consumer obligations, a full-corpus blinded human audit of the DeepSWE runs identifies terminal mismatches, including diagnoses whose task scope remains disputed. Following supplementary execution checks, all retained DeepSWE terminal events have execution support. The audit also shows that recovery within a trajectory can coexist with terminal omissions in the same run. The process analysis links
    
[^39]: SatMend：面向间歇性地面联系卫星星座的基于LLM智能体的自动化软件修复

    SatMend: LLM Agents for Automated Software Repair in Satellite Constellations with Intermittent Ground Contact

    [https://arxiv.org/abs/2611.29227](https://arxiv.org/abs/2611.29227)

    SatMend提出了一种由地面LLM智能体驱动的卫星星座自动化软件修复方法，通过持久会话保留诊断上下文、结合源码与运行时证据选择修复或状态恢复动作，并使地面修复准备与星载交付验证在间歇性地面联系下并行进行。

    

    卫星星座依赖星载软件来维持跨协作航天器提供的服务。维护这些软件需要处理相互关联的组件版本和不同的故障原因，同时还要通过间歇性的地面联系来交付修复。卫星星座中的自动化软件修复目前仍未得到充分探索，尤其是在间歇性地面联系条件下，如何跨相关版本协调修复并进行经过验证的星载恢复。为填补这一空白，我们提出了SatMend，一种由地面大语言模型（LLM）智能体驱动的软件修复方法。SatMend将相关版本组织为持久会话以保留诊断上下文，并结合源代码与运行时证据来选择代码修复或状态恢复动作。它对每个版本分别进行验证，并在持续修复的过程中提交已就绪的候选版本，使地面准备工作能够与星载交付和验证并行进行。实验……

    arXiv:2611.29227v1 Announce Type: new  Abstract: Satellite constellations rely on onboard software to sustain services across cooperating spacecraft. Maintaining this software requires handling related component versions and different failure causes while delivering repairs through intermittent ground contact. Automated software repair in satellite constellations remains underexplored, particularly the coordination of repairs across related versions with verified onboard recovery under intermittent ground contact. To address this gap, we propose SatMend, a software repair method driven by ground-based large language model (LLM) agents. SatMend organizes related versions into persistent sessions to retain diagnostic context and combines source and runtime evidence to select code repair or state recovery actions. It validates each version separately and submits ready candidates during continued repair, allowing ground preparation to overlap with onboard delivery and verification. Experim
    
[^40]: 修复还是删除？外部反馈让更便宜的AI编程智能体修复规则违规

    Fixed or Deleted? External Feedback Lets Cheaper AI Coding Agents Repair Rule Violations

    [https://arxiv.org/abs/2611.29162](https://arxiv.org/abs/2611.29162)

    研究发现，引入商业检查器CodeAssure等外部反馈（报告违规位置和修复目标）能大幅提升AI编程智能体修复项目规则违规的成功率，使更小更便宜的模型（修复率从26%升至69%）接近甚至逼近大型模型的修复效果。

    

    摘要：许多缺陷违反了代码本身无法反映的项目规则，例如分层约束或必需的测试。我们研究了当外部检查器报告这些违规及其位置和修复目标时，LLM编程智能体是否能更好地修复此类违规。由AI智能体驱动的三个模型（小型、中型和大型模型）处理了十个开源项目中25个包含31个植入缺陷的拉取请求，外加5个正确的对照组，分别在无外部反馈和结合商业检查器CodeAssure（CA）检测结果的条件下各运行五次。我们还将结果与GitHub Copilot代码审查进行比较，并将CodeAssure的检测能力与九个开箱即用的免费静态分析工具（包括Semgrep和SonarQube Community）进行对比。只有当修复后的代码能够编译并保留原有功能时，该次运行才被计为成功修复。实验结果显示，CA的检测结果使最小模型的修复率从26%提升至69%，中型模型从46%提升至90%，大型模型从45%提升至93%。

    arXiv:2611.29162v1 Announce Type: cross  Abstract: Many defects violate project rules that the code itself does not reveal, such as layering constraints or required tests. We study whether LLM coding agents repair such violations better when an external checker reports them, with their location and a repair goal. Three models, a small, medium and large model, driven by an AI agent, worked on 25 pull requests with 31 planted defects in ten open-source projects, plus 5 correct controls, five times alone and five times with findings from CodeAssure (CA), a commercial checker. We also compare with GitHub Copilot code review, and compare CodeAssure's detection with nine free static-analysis tools run out of the box, including Semgrep and SonarQube Community. Counting a run as repaired only when the repaired code also compiles and keeps its feature, CA findings raised the repair rate from 26% to 69% for the smallest model, from 46% to 90% for the medium model and from 45% to 93% for the larg
    
[^41]: 无在线GNSS条件下凤凰号垂直起降无人机跨域数字孪生自主制导架构的实现与软件在环评估

    Implementation and Software-in-the-Loop Evaluation of a Trans-Domain Digital Twin Architecture for Autonomous Guidance of the Phoenix VTOL UAV Without Online GNSS

    [https://arxiv.org/abs/2611.29158](https://arxiv.org/abs/2611.29158)

    本研究设计并实现了基于跨域数字孪生（TDDT）架构的凤凰号垂直起降无人机自主制导系统，在无在线GNSS条件下通过导航、飞行动力学与气象多模型协同及多传感器数据融合实现可靠导航，并通过软件在环仿真进行了评估验证。

    

    在应急救援、基础设施监测和货物运输中的自主无人机制导需要可靠的导航。然而，在封闭空间、由自然障碍物和建筑物造成的盲区，以及受电子战中无线电干扰和信号欺骗影响的环境里，基于卫星的定位可能变得不可用或不可靠。本研究设计、实现并评估了一个基于跨域数字孪生（TDDT）的系统，用于凤凰号垂直起降无人机的自主制导。独立的导航模型、飞行动力学模型和气象模型在状态估计、预测、决策制定、指令执行和反馈的循环中协同工作。来自惯性、磁力、压力和空速传感器的数据，结合环境信息，构成了估计与预测的基础。此外，共享状态、时间同步、信息来源的保存以及……

    arXiv:2611.29158v1 Announce Type: cross  Abstract: Autonomous UAV guidance in emergency response, infrastructure monitoring, and cargo transportation requires reliable navigation. However, satellite-based positioning may become unavailable or unreliable in enclosed spaces, blind spots caused by natural obstacles and structures, and environments affected by radio interference and signal spoofing during electronic warfare. This study designs, implements, and evaluates a system based on a Trans-Domain Digital Twin (TDDT) for autonomous guidance of the Phoenix VTOL UAV. Independent navigation, flight dynamics, and meteorological models are coordinated within a cycle of state estimation, prediction, decision-making, command execution, and feedback. Data from inertial, magnetic, pressure, and airspeed sensors, together with environmental information, form the basis for estimation and prediction. Furthermore, a shared state, temporal synchronization, preservation of information provenance, an
    
[^42]: 作为“活文档”的运维手册：辅助式事后修订下文档演进的纵向研究

    Operational Runbooks as Living Documents: A Longitudinal Study of Document Evolution under Assisted Post-Incident Revision

    [https://arxiv.org/abs/2611.29097](https://arxiv.org/abs/2611.29097)

    该论文首次对生产环境中运维手册作为“活文档”的长期演进进行了纵向实证研究，并提出利用轻量级的语言学与结构信号来量化与监控文档随时间的变化。

    

    运维手册在运维工作流中日益发挥着“活文档”的作用：它们由人员维护、用于事件支持，并随着组织知识的变化而持续修订。然而，目前鲜有研究了解这类文档集合在生产环境中如何随时间演进，以及哪些可解释的信号对于监控文档变化是有用的。我们分析了17周内版本控制的运维手册快照，这些快照产生于机器辅助审查与人工监督之下的事后修订过程。我们基于先前关于程序性语言、技术沟通和文档组织的研究，采用轻量级的语言学与结构信号来量化文档演进。相关性分析显示出混合的关联模式，包括正相关与负相关关系，其效应量从可忽略到中等不等，并在不同语料库之间存在差异。时间维度的摘要分析进一步区分了……

    arXiv:2611.29097v1 Announce Type: new  Abstract: Operational runbooks increasingly function as living documents within operational workflows: they are maintained by people, used in incident support, and continuously revised as organizational knowledge changes. Yet little is known about how such document collections evolve over time in production settings, or which interpretable signals are useful for monitoring document change. We analyze 17 weeks of version-controlled runbook snapshots produced during post-incident revision with machine-assisted review and human oversight. We quantify document evolution using lightweight linguistic and structural signals grounded in prior work on procedural language, technical communication, and document organization. Correlation analysis shows mixed patterns of association, including both positive and negative relationships, with effect sizes ranging from negligible to moderate and varying across corpora. Temporal summaries further distinguish compar
    
[^43]: LITMUS——基于文献的用例映射与风险卡综合测试平台

    LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards

    [https://arxiv.org/abs/2611.29034](https://arxiv.org/abs/2611.29034)

    本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。

    

    大语言模型（LLM）和人工智能（AI）在现实世界应用中的快速普及，加剧了对系统化、面向安全的部署框架的需求。现有方法通常优先围绕LLM和AI架构本身来构建数据集和风险卡，却很少考虑驱动模型选择的特定用例。这种疏漏是有问题的，因为任何实际的AI部署都始于详细的用例描述，而该描述正是评估上下文风险的基础。在本文中，我们提出了一个结构化流水线，给定一个用例，它能够自动生成定制的数据集以及配套的风险卡。该流程确保所识别的风险与用例紧密耦合，并提供来自同行评审文献的可追溯证据，以证实每一个风险与用例之间的关系。先前的声明验证方法（FActSco...

    arXiv:2611.29034v1 Announce Type: new  Abstract: The proliferation of large language models (LLMs) and Artificial Intelligence (AI) in real-world applications has heightened the need for systematic, safety-oriented deployment frameworks. Existing approaches typically prioritize the construction of datasets and risk cards around the LLM and AI architecture itself, yet they provide scant consideration for the specific use-case that drives model selection. This omission is problematic because any practical AI deployment begins with a detailed use-case description that informs the assessment of contextual hazards. In this paper, we introduce a structured pipeline that, given a use-case, automatically produces a tailored dataset and an accompanying risk card. The process ensures that identified risks are tightly coupled to the use-case, and it offers traceable evidence from peer-reviewed literature to substantiate each risk to use-case relationship. Prior claim-verification methods (FActSco
    
[^44]: 水印声明接受检验：审计生产级大语言模型的经验教训

    Watermark Claims Under Test: Lessons from Auditing a Production LLM

    [https://arxiv.org/abs/2611.29018](https://arxiv.org/abs/2611.29018)

    对生产级LLM的红绿水印黑盒检测审计表明，已发表的黑盒测试在真实部署的水印模型上会失效，因为提示词被纳入水印的重复上下文历史后会恰好抑制检验所测量的标记，导致探针“失明”。

    

    我们能否在不知道LLM密钥的情况下判断其是否带有水印？已发表的黑盒测试表明，精心设计的提示词可以揭示水印的统计足迹。本文考察了当这些测试从受控的参考实现转移到水印参数未公开的已部署模型时会发生什么。我们对Claude上的红绿水印存在性检验进行了审计。在十种初始配置中，该已发表的检验在供应商声明已加水印的模型上从未产生拒绝结果，尽管在可比样本量下它能检测到本地参考实现。受控实验揭示了一种失效模式：将提示词纳入水印的重复上下文历史中，恰好会在检验所测量的位置抑制水印标记。水印仍然处于激活状态，但探针变得“失明”。修复探针的尝试揭示了更多的混淆因素：构造的上下文可能触发模型原生的数字转单词偏好转换……（摘要截断）

    arXiv:2611.29018v1 Announce Type: cross  Abstract: Can we tell whether an LLM is watermarked without knowing its secret key? Published black-box tests suggest that carefully designed prompts can reveal the statistical footprint of a watermark. This paper examines what happens when those tests move from controlled reference implementations to a deployed model whose watermarking parameters are undisclosed. We present an audit of the Red-Green watermark presence test on Claude. Across ten initial configurations, the published test never rejects on a model the provider states is watermarked, despite detecting a local reference at comparable sample sizes. Controlled experiments expose a failure mode: including the prompt in the watermark's repeated-context history suppresses marking exactly where the test measures it. The watermark remains active, but the probe becomes blind. Attempts to repair the probe reveal further confounds. Constructed contexts can trigger native digit-to-word prefere
    
[^45]: 有状态AI运行时中的消费者时有效性：验证恢复状态与延迟授权

    Consumer-Time Validity in Stateful AI Runtimes: Verifying Recovered State and Deferred Authority

    [https://arxiv.org/abs/2611.29004](https://arxiv.org/abs/2611.29004)

    该论文提出“消费者时有效性”这一新概念，指出在有状态AI运行时中，一个组件记录的取消、回滚或批准并不能保证下游消费时的有效性，并将其分为约束读取内容的状态有效性与约束执行内容的权限新鲜度两类缺失检查。

    

    有状态的LLM服务引擎和智能体运行时会在组件与会话之间复用执行状态和延迟权限。取消、回滚或批准由一个组件记录，而被另一个组件依赖，但记录本身并不能保证后续使用的有效性：下游消费者可能读取状态或调度一个其身份、生产状态、分配或目标修订版本此后已发生变化的动作。我们将这种缺失的检查称为“消费者时有效性”，并将其区分为两种形式。状态有效性约束消费者所读取的内容：在配备LMCache的vLLM上，被中断的KV缓存加载会留下已索引但未写入的块，而仅在主机端进行的计数器回退会导致GPU位置失去同步，因此在10/10次原生故障运行中，第二个租户生成了模式合法但指向错误租户的工具调用；该调用被记录但未被执行。权限新鲜度约束消费者所执行的内容：在多会话批准运行时上，一个未被撤销的延迟授权覆盖……

    arXiv:2611.29004v1 Announce Type: cross  Abstract: Stateful LLM serving engines and agent runtimes reuse execution state and deferred permissions across components and sessions. A cancellation, rollback, or approval is recorded by one component and relied on by another, and recording it does not make the later use valid: a downstream consumer may read state or dispatch an action whose identity, production status, allocation, or target revision has since changed. We call the missing check consumer-time validity and separate two forms. State validity governs what a consumer reads. On vLLM with LMCache, an interrupted KV load leaves unwritten blocks indexed, and a host-only counter rewind desynchronizes the GPU position, so a second tenant generates a schema-valid tool call naming the wrong tenant in 10/10 native-fault runs; the call is recorded and not executed. Authority freshness governs what a consumer executes: on a multi-session approval runtime, an unrevoked deferred grant overwrot
    
[^46]: 《“我该怎么做？”在软件工程中负责任地集成人工智能的指南》

    "What Should I Do?" Guidelines for Responsibly Integrating AI in Software Engineering

    [https://arxiv.org/abs/2611.28996](https://arxiv.org/abs/2611.28996)

    该研究通过对微软、GitHub和辉瑞1118名开发者的两阶段大规模调查，总结出35条帮助开发者在软件工程工作中负责任地集成AI的实用指南，并在一年后跨行业验证了其有效性。

    

    关于开发者将人工智能（AI）集成到软件工程（SE）工作中的“正确”方式，目前存在诸多猜测，但现有证据稀少且相互矛盾。然而，开发者需要对其（无论是否借助AI帮助）发布的产品负责，因此他们必须以负责任的方式集成AI。为填补这一空白，我们对微软、GitHub和辉瑞的1118名开发者开展了两阶段的大规模研究。在第一阶段，我们调查了微软的860名开发者，了解他们在日常工作中在哪些环节寻求AI支持、在哪些环节抗拒AI支持，并根据其反馈总结出35条面向开发者的指南：其中29条涵盖软件工程工作的五大类别，另有6条为跨领域通用指南。一年后，随着AI能力与开发者经验的共同演进，我们在第二阶段对来自科技领域（微软、GitHub）和生物医药领域（辉瑞）的258名开发者对这些指南进行了验证。研究中的开发者认为这些指南表述清晰（原文在此处截断）。

    arXiv:2611.28996v1 Announce Type: new  Abstract: Speculations abound about the "right" ways for developers to integrate AI into their software engineering (SE) work, but the evidence that exists is sparse and contradictory. Yet, developers are held responsible for the products they ship with (or without) AI's help, so they need to integrate AI in responsible ways. To address this gap, we conducted a two-phase, large-scale investigation with 1,118 developers at Microsoft, GitHub, and Pfizer. In Phase 1, we surveyed 860 developers at Microsoft on where they sought and resisted AI support in their daily work, and from their accounts, synthesized 35 developer-facing guidelines: 29 across five categories of SE work and 6 cross-cutting ones. A year later, as AI capabilities and developers' experiences co-evolved, we validated them in Phase 2 with 258 developers in technology (Microsoft, GitHub) and biopharmaceutical (Pfizer) domains. Developers in our study found the guidelines clearly worde
    
[^47]: 大语言模型在代码生成中的微观操作鲁棒性

    Micro Operational Robustness of LLMs on Code Generation

    [https://arxiv.org/abs/2611.28957](https://arxiv.org/abs/2611.28957)

    本文提出了一种名为“场景域分析”的鲁棒性评估方法，通过寻找编码任务自然语言描述中导致大语言模型输出错误结果的最小变化，来系统评估LLM在代码生成任务中的微观操作鲁棒性。

    

    如今在软件开发中，使用大语言模型（LLM）生成程序代码已成为常见做法，因此评估LLM在这一用途中的鲁棒性十分必要。本文特别关注LLM对编码任务描述变化的敏感程度。然而，现有的鲁棒性评估技术并不适用于代码生成，因为自然语言（NL）描述的输入数据空间是离散且稀疏的。为解决这一问题，我们提出了一种称为“场景域分析”的鲁棒性评估方法，旨在找到编码任务的自然语言描述中会导致LLM产生错误输出的预期最小变化。我们从理论上正式证明了该方法的理论性质，并开展了大量实验来评估四种最先进的大语言模型（Gemini-pro、Codex、Llama2和Falcon 7B）的鲁棒性。

    arXiv:2611.28957v1 Announce Type: cross  Abstract: It is now common practice in software development for large language models (LLMs) to be used to generate program code. It is desirable to evaluate the robustness of LLMs for this usage. This paper is concerned in particular with how sensitive LLMs are to variations in descriptions of the coding tasks. However, existing techniques for evaluating this robustness are unsuitable for code generation because the input data space of natural language (NL) descriptions is discrete and sparse. To address this problem, we propose a robustness evaluation method called scenario domain analysis, which aims to find the expected minimal change in the NL descriptions of coding tasks that would cause the LLMs to produce incorrect outputs. We have formally proved the theoretical properties of the method and also conducted extensive experiments to evaluate the robustness of four state-of-the-art LLMs: Gemini-pro, Codex, Llamma2 and Falcon 7B, and have fo
    
[^48]: 精炼、可靠且可复现：Popcorn Void Finder 的算法分析与重新实现

    Refined, Reliable and Reproducible: Algorithmic Analysis and Reimplementation of the Popcorn Void Finder

    [https://arxiv.org/abs/2611.28882](https://arxiv.org/abs/2611.28882)

    本文对宇宙空洞查找工具 Popcorn Void Finder 进行了全面的重新实现，以算法透明性、可维护性和可复现性为核心，并系统评估了发表后算法修改所带来的统计与计算影响。

    

    宇宙空洞是宇宙中巨大的欠密集区域，蕴含着关于宇宙大尺度结构和演化的宝贵信息。通过将球形空洞扩展为相互重叠的球体结构，Popcorn Void Finder 展示了对宇宙空洞群体更优的统计表征能力。然而，其作为研究原型的出身逐渐阻碍了软件的可维护性和可扩展性。此外，论文发表后的算法修改尚未得到系统性评估，其统计和计算方面的后果也未得到充分表征。在本工作中，我们对 Popcorn Void Finder 进行了全面的重新实现，优先考虑算法透明性、可维护性和可复现性。我们记录了完整的空洞识别流程，重新审视了原始的算法设计选择，并评估了其统计和计算方面的后果。不同实现之间的比较……（摘要此处截断）

    arXiv:2611.28882v1 Announce Type: cross  Abstract: Cosmic voids are vast underdense regions of the Universe that encode valuable information about its large-scale structure and evolution. By extending spherical voids into overlapping sphere structures, the Popcorn Void Finder demonstrated an improved statistical characterization of cosmic void populations. However, its origin as a research prototype progressively hindered its maintainability and extensibility. Moreover, post-publication algorithmic modifications have not been systematically evaluated, leaving their statistical and computational consequences insufficiently characterized. In this work, we present a comprehensive reimplementation of the Popcorn Void Finder, prioritizing algorithmic transparency, maintainability, and reproducibility. We document the complete identification procedure, revisit the original algorithmic choices, and evaluate their statistical and computational consequences. Comparisons between implementations 
    
[^49]: CivilCADBench：面向土木工程图纸集理解、重建与设计的多模态大语言模型基准测试

    CivilCADBench: Benchmarking Multimodal LLMs for Understanding, Reconstructing, and Designing Civil Engineering Drawing Sets

    [https://arxiv.org/abs/2611.28853](https://arxiv.org/abs/2611.28853)

    提出了基于包含25万余个CAD实体的真实基础设施图纸构建的CivilCADBench基准，用于评估多模态大语言模型在土木工程图纸集的理解、重建与设计能力。

    

    民用基础设施是通过工程图纸集进行设计、建造和维护的，然而图纸的编制与修订仍然是一项劳动密集型工作，亟需自动化手段。多模态大语言模型（MLLMs）是很有前景的候选方案，但要对其进行评估，需要一个立足于真实工程实践的基准。我们提出了CivilCADBench，一个用于理解、重建和设计土木工程图纸集的基准，它基于真实的基础设施图纸和源CAD文件构建，其中包含超过25万个CAD实体。与参数化3D CAD模型不同，这些图纸内容密集、分层组织且包含多张图纸；由于打印件是有损投影、图层信息不会以像素形式保留，一个文件可能打印正确，但其内部组织结构却是错误的。CivilCADBench包含三个难度递增的任务族：理解任务通过2,183道经专家审核的问题进行评估，这些问题涵盖从单张图纸到整套图纸；而重建任务……（摘要原文在此处截断）

    arXiv:2611.28853v1 Announce Type: cross  Abstract: Civil infrastructure is designed, built, and maintained through engineering drawing sets, yet preparing and revising them remains labour-intensive and calls for automation. Multimodal large language models (MLLMs) are promising candidates, but assessing them requires a benchmark grounded in real engineering practice. We introduce CivilCADBench, a benchmark for understanding, reconstructing, and designing civil engineering drawing sets, built from real infrastructure drawings and source CAD files containing over 250,000 CAD entities. Unlike parametric 3D CAD models, these drawings are dense, layered, and multi-sheet; and since prints are lossy projections with no pixel trace of layers, a file can print correctly yet remain wrongly organised. CivilCADBench comprises three task families of increasing difficulty: understanding tasks are evaluated with 2,183 expert-reviewed questions ranging from single sheets to whole sets, while reconstru
    
[^50]: Dhaani：一个面向全新程序构建与存量系统现代化的数据流驱动工具

    Dhaani: A Data-Flow-Driven Tool for Greenfield Construction and Brownfield Modernisation

    [https://arxiv.org/abs/2611.28798](https://arxiv.org/abs/2611.28798)

    Dhaani 是一款以带注释数据流图（ADFD）作为意图与实现之间可审查契约的免费桌面工具，支持将自然语言需求按依赖顺序构建为全新 Python/.NET 程序，并能从 Fortran 存量代码库迁移生成现代化代码。

    

    arXiv:2611.28798v1 公告类型：new 摘要：大语言模型能够快速生成看似合理的代码，但端到端的提示方式使开发人员几乎无法控制任务分解、接口设计，以及那些必须在后续修订中得以保留的假设。我们提出了 Dhaani，这是一款用于全新（greenfield）程序构建与存量（brownfield）代码库现代化的免费桌面应用程序。该工具将带注释的数据流图（ADFD）作为意图与实现之间可供审查的契约。在全新构建工作流 Build 中，用户将自然语言需求转换为类型化图，审查过程规范及前置/后置条件，批准生成的模块，并按依赖顺序完成 Python 或 .NET（C#）的开发。在存量迁移工作流 Migrate 中，工具通过源代码推断从 Fortran 代码库中恢复出 ADFD，在有界依赖组内执行目标代码生成，并展示源代码与目标代码的对齐关系、生成的代码、错误信息、耗时、token 用量以及成本。该工具保持代码与（摘要在此处截断）

    arXiv:2611.28798v1 Announce Type: new  Abstract: Large language models can generate plausible code quickly, but an end-to-end prompt leaves developers with little control over decomposition, interfaces, or the assumptions that must survive later revisions. We present Dhaani, a free desktop application for greenfield program construction and brownfield repository modernisation. It makes an annotated data-flow diagram (ADFD) the inspectable contract between intent and implementation. In the greenfield workflow, Build, a user turns natural-language requirements into a typed graph, reviews process specifications and pre/post-conditions, ratifies generated modules, and proceeds in dependency order for Python or .NET (C#). In the brownfield workflow, Migrate uses source inference to recover an ADFD from a Fortran repository, performs target generation in bounded dependency groups, and exposes source-target alignment, generated code, errors, time, token use, and cost. The tool keeps code and 
    
[^51]: 零信任形式化方法：AI智能体为不可信求解器构建经验证的检查器

    Zero-Trust Formal Methods: AI Agents Build Verified Checkers for Untrusted Solvers

    [https://arxiv.org/abs/2611.28704](https://arxiv.org/abs/2611.28704)

    该论文提出零信任形式化方法，让AI智能体自动生成Isabelle/HOL理论、可靠性证明和经过形式验证的检查器，使得求解器的答案只有在验证检查器接受其证书后才被采信，从而完全不信任AI智能体和求解器本身。

    

    AI编程智能体如今生成代码的速度已超过人类审查的速度。安全分析依赖于求解器的答案，例如“不可满足”（即不存在攻击路径）或“最优”。与满足模型不同，这些答案无法通过直接查看来检查。形式化验证将信任从代码转移到了规约上，但验证器仍然将大型SAT、SMT和答案集求解器以及向求解器查询的转换过程置于其可信计算基之中，而这些组件曾出现过可靠性漏洞。与信任其验证器和求解器的基于智能体的流水线不同，我们的农场默认不信任任何组件：既不信任可能犯错或为达成目标而投机取巧的AI智能体，也不信任求解器。只有当经验证的检查器接受其证书时，求解器的答案才算有效。每一个理论、每一份证明、每一行代码都由AI生成。对于每个组件，智能体首先创建一个包含规约、可靠性证明和参考检查器的Isabelle/HOL理论。

    arXiv:2611.28704v1 Announce Type: new  Abstract: AI coding agents now generate code faster than humans can review it. Security analyses rely on solver answers such as "unsatisfiable" (for example, no attack path exists) or "optimal." Unlike satisfying models, they cannot be checked by looking at them. Formal verification moves trust from code to specifications, but verifiers still place large SAT, SMT, and answer-set solvers, and the translation into solver queries, in their trusted computing base, and these have had soundness bugs. Unlike agent-based pipelines that trust their verifier and solver, our farm trusts no component by default: neither the AI agents, which might make mistakes or game their objectives, nor the solvers. A solver answer counts only if a verified checker accepts its certificate. Every theory, proof, and line of code is generated by AI. For each component, agents first create an Isabelle/HOL theory with a specification, a soundness proof, and a reference checker 
    
[^52]: 何时测试、修复还是重新生成：决策引导的高效RTL生成

    When to Test, Repair, or Regenerate: Decision-Guided Efficient RTL Generation

    [https://arxiv.org/abs/2611.28667](https://arxiv.org/abs/2611.28667)

    该论文提出一种由专门决策模型引导的自适应框架，智能决定何时对RTL设计进行测试、修复或重新生成，从而在降低推理成本的同时，在三个基准上将平均Pass@1提升5.4至7.1个百分点。

    

    大型语言模型（LLM）能够根据自然语言规范生成寄存器传输级（RTL）设计。为了提高功能正确性，许多RTL智能体使用仿真反馈来指导优化。然而，在没有参考测试平台的情况下，为每个初始实现都生成测试平台，即使在RTL本身正确时也会产生不必要的推理成本。因此，我们引入了一个自适应框架，利用专门的决策模型来统筹测试、优化和候选选择。该框架接受高评分的初始实现，并仅为其余实现生成测试平台。当需要进一步搜索时，初始评分用于指导修复与重新生成之间的资源分配，而缺陷排序和仿真反馈则用于指导修复过程。在最终选择阶段，该框架优先考虑通过测试的候选方案，并依据决策模型的评分对其进行排序。在三个基准测试中，该框架结合DeepSeek使平均Pass@1提升了5.4至7.1个百分点。

    arXiv:2611.28667v1 Announce Type: cross  Abstract: Large language models (LLMs) can generate register-transfer-level (RTL) designs from natural-language specifications. To improve functional correctness, many RTL agents use simulation feedback to guide refinement. However, without a reference testbench, generating one for every initial implementation incurs inference cost even when the RTL is correct. We therefore introduce an adaptive harness that uses a specialized decision model to orchestrate testing, refinement, and candidate selection. It accepts high-scoring initial implementations and generates testbenches for the rest. When further search is needed, the initial score guides repair--regeneration allocation, while defect rankings and simulation feedback guide repair. For final selection, the harness prioritizes test-passing candidates and ranks them using decision-model scores. Across three benchmarks, the harness with DeepSeek improves mean Pass@1 by 5.4--7.1 percentage points 
    
[^53]: VeriLadder：一个用于工具验证Verilog数据集的多智能体规格细化系统

    VeriLadder: A Multi-Agent Specification Refinement System for Tool-Verified Verilog Datasets

    [https://arxiv.org/abs/2611.28617](https://arxiv.org/abs/2611.28617)

    本文提出VeriLadder，一个五级Verilog数据集及其多智能体构建框架，由LLM智能体编写和修订规格说明，并由综合器与形式等价性检查器等EDA工具进行验证，从而解决了编写Verilog的LLM缺乏高质量训练数据的问题。

    

    训练数据是人工智能领域的瓶颈，对于编写Verilog代码的大型语言模型（LLM）尤为如此，因为寄存器传输级（RTL）代码很少公开发表，且现有数据集大多只检查代码的语法正确性，而非指令本身的正确性。本文提出了VeriLadder——一个五级Verilog数据集，以及构建该数据集的多智能体框架。在该框架中，LLM智能体负责编写和修订规格说明，而EDA工具（综合器和形式等价性检查器）决定哪些内容被接受。其分级遵循工程师的检查流程。第一级对来自两个大型Verilog语料库（RTLCoder和PyraNet）的设计进行标注，包括综合结果、失败原因、关键警告和综合后指标。RTLCoder中28%的设计无法通过综合，这些设计被保留作为负样本。第二级依据四项规格质量指标对每条指令进行评分，低分预示着重建会失败。第三级将每条指令重写为精细化的规格说明并接受它。

    arXiv:2611.28617v1 Announce Type: new  Abstract: Training data is a bottleneck in AI, especially for large language models (LLMs) that write Verilog, because register-transfer-level (RTL) code is rarely published and existing datasets mostly check code for syntax, not instruction correctness. This paper presents VeriLadder, a five-level Verilog dataset, and the multi-agent framework that builds it, in which LLM agents write and revise specifications, and EDA tools (a synthesizer and a formal equivalence checker) decide what is accepted. Its levels follow an engineer's checks. Level 1 labels designs from two large Verilog corpora, namely RTLCoder and PyraNet, with synthesis results, failure causes, critical warnings and post-synthesis metrics. 28% of RTLCoder designs fail synthesis and are kept as negatives. Level 2 grades each instruction on four specification qualities, and low grades predict failed rebuilds. Level 3 rewrites each instruction into a refined specification and accepts i
    
[^54]: 判决正确，理由也正确吗？一项基于程序执行的研究：LLM漏洞检测器是否真正理解漏洞

    Right Verdict, Right Reason? An Execution-Grounded Study of Whether LLM Vulnerability Detectors Understand Vulnerabilities

    [https://arxiv.org/abs/2611.28584](https://arxiv.org/abs/2611.28584)

    该研究以程序执行为基准，对515个可重现漏洞设计了五级递进测试，发现LLM漏洞检测器的高基准分数未必代表它们真正理解漏洞，其表现可能与无需理解的简单基线相当。

    

    大语言模型（LLM）越来越多地被用于漏洞检测，其基准测试分数常被解读为模型理解所报告漏洞的证据。我们通过程序执行来检验这一解读。针对来自BigVul和PrimeVul的515个C和C++函数（这些函数的漏洞我们均已用概念验证成功重现），我们确定了故障发生的代码行以及出错运行所依赖的语句。这些事实构成了一组由五个测试组成的递进阶梯：从区分漏洞函数与其修复版本，到定位故障并对故障移除作出反应，再到评判经执行验证的代码编辑以及预测具体运行的结果。每个测试都配有一个无需理解能力的基线（例如选取包含风险调用的代码行的规则），并且我们在195个发布于模型训练截止日期之后的漏洞上重复了该研究。研究得出四个发现。首先，区分一个漏洞……（原文摘要在此处截断）

    arXiv:2611.28584v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to detect vulnerabilities, and their benchmark scores are often read as evidence that they understand the vulnerabilities they report. We test this reading against program execution. For 515 C and C++ functions from BigVul and PrimeVul whose vulnerability we reproduce with a proof of concept, we identify the line at which the fault occurs and the statements on which the faulting run depended. These facts ground a ladder of five tests, from telling a vulnerable function from its fixed version, through locating the fault and reacting to its removal, to judging execution-confirmed edits and predicting the outcome of a concrete run. Each test is paired with a baseline that needs no understanding, such as a rule that picks lines with risky calls, and we repeat the study on 195 vulnerabilities published after the training cutoffs of the models. Four findings emerge. First, telling a vulnerab
    
[^55]: 智能体工作流的自动化安全分析

    Automated Safety Analysis for Agentic Workflows

    [https://arxiv.org/abs/2611.28469](https://arxiv.org/abs/2611.28469)

    该论文提出了Apsis，一个部署前的静态安全分析器，它通过将不同框架的智能体工作流代码提升为统一的语义图中间表示，从而在部署前自动检测工作流设计层面的逻辑缺陷。

    

    LLM应用正日益以智能体工作流的形式构建：这类程序将模型调用、外部工具和控制逻辑相结合，能够代表用户自主执行操作。由于这些工作流会执行真实世界中的动作，工作流连接方式中的逻辑缺陷可能在部署后的系统中造成严重危害。与提示注入或格式错误的API调用不同，这类工作流缺陷源于工作流设计本身的缺陷，然而目前尚无工具能够在部署前静态地验证跨框架的工作流逻辑。我们提出了Apsis，一个能够在部署前检测工作流缺陷的静态安全分析器。为了实现跨框架运行，Apsis将框架特定的代码提升为一种统一的中间表示（IR）。该IR将工作流建模为由步骤、数据依赖和共享状态构成的语义图，并为每个组件标注行为属性，例如某个输入是否为必需项，或共享通道如何合并并发的写入。

    arXiv:2611.28469v1 Announce Type: cross  Abstract: LLM applications are increasingly built as agentic workflows: programs that combine model calls, external tools, and control logic to act autonomously on a user's behalf. Because these workflows execute real-world actions, logical flaws in how a workflow is wired can cause severe harm in deployed systems. Unlike prompt injections or malformed API calls, these workflow bugs stem from defects in the workflow's design, yet no existing tool can statically validate workflow logic across frameworks prior to deployment. We present Apsis, a static safety analyzer that detects workflow bugs before deployment. To operate across frameworks, Apsis lifts framework-specific code into a common intermediate representation (IR). This IR models a workflow as a semantic graph of steps, data dependencies, and shared state, while annotating each component with behavioral properties such as whether an input is mandatory or how a shared channel merges concur
    
[^56]: 基于智能体支持自动化的人机协同运维手册改进

    Human-in-the-Loop Runbook Improvement with Agentic Support Automation

    [https://arxiv.org/abs/2611.28441](https://arxiv.org/abs/2611.28441)

    本文提出了一种将智能体AI融入基于运维手册的事件处理流程的系统，该机器人能引导人工操作员完成故障缓解过程、评估处理结果，并在发现手册存在缺口时建议改进，实现了人机协同的运维手册持续优化。

    

    运维支持是生产软件服务的重要组成部分。支持请求往往具有突发性，且形式多样，例如客户升级投诉或计划外的服务中断。各组织的工程师们已经成功实现了自动化来帮助简化支持流程，但许多解决方案仍依赖于人工操作员。提升操作员响应时间的一个有效策略是维护最新状态的运维手册，这是一类详细列举面对已识别事故时的分诊步骤和修复流程的文档。本文对一种将智能体AI融入基于运维手册的事故解决模式的运维系统进行了研究。该机器人利用运维手册语料库中的信息以及一些特定于系统的工具，引导人工操作员完成缓解流程，然后评估处理结果，并在发现手册存在缺口和不足之处时建议对运维手册进行修改完善。

    arXiv:2611.28441v1 Announce Type: new  Abstract: Operational support is an important component of production software services. Support requests are often emergent and can come in many forms such as customer escalations or unplanned service interruption. Engineers across organizations have been successful in implementing automation to help streamline support processes but many solutions remain in the hands of human operators. A successful strategy to improve operator response times is to maintain up-to-date runbooks which are documents enumerating triage steps and remediation procedures in the face of identified incidents. This paper is an investigation into an operational system which incorporates Agentic AI into a runbook based incident resolution pattern. The bot uses information in the runbook corpus along with some system-specific tooling to guide the manual operator through the mitigation process then evaluate the outcome and suggest edits to the runbook where it found gaps and i
    
[^57]: Lean Census：AI 生成的 Lean 代码的分布分析

    Lean Census: A Distributional Analysis of AI-Generated Lean Code

    [https://arxiv.org/abs/2611.28430](https://arxiv.org/abs/2611.28430)

    该研究通过对代码统计量分布的对比分析，揭示了 AI 生成的 Lean 库与人类编写的库之间存在显著差异——AI 生成的库每个定理的内联声明更多、证明更长、实例更少、私有声明更多。

    

    我们通过代码简单统计量的分布，将两个人类编写的和四个 AI 生成的 Lean 库与 Mathlib 进行比较。对于每个库，我们测量了证明中的内联声明、证明长度、每个声明的使用频率，以及诸如实例和私有声明等设计选择，并将每个库与 Mathlib 自身各领域之间的差异进行对照分析。与人类编写的库相比，AI 生成的库中每个定理的内联声明更多、证明更长、实例更少、私有声明更多。我们将该分析的代码以 lean-census 包的形式发布于 https://github.com/project-numina/lean-census。

    arXiv:2611.28430v1 Announce Type: cross  Abstract: We compare two human-written and four AI-generated Lean libraries with Mathlib through the distributions of simple statistics of their code. For each library we measure inline claims in proofs, proof length, how often each declaration is used, and design choices such as instances and private declarations, and we read each against the variation among Mathlib's own areas. The AI-generated libraries have more inline claims per theorem, longer proofs, fewer instances, and more private declarations than the human-written ones. We release the code for the analysis as a package, lean-census, at https://github.com/project-numina/lean-census.
    
[^58]: 何时AI不应做决策：自动化求职面试中的证据充分性与决策稳定性

    When AI Should Not Decide: Evidence Sufficiency and Decision Stability in Automated Job Interviews

    [https://arxiv.org/abs/2611.28426](https://arxiv.org/abs/2611.28426)

    本研究提出证据充分性与决策稳定性评估框架（ESDSA），通过证据分类、引文溯源和语言变体稳定性检验来判断AI自动化面试评估建议是否有足够证据支撑，在留出测试中实现了85.19%的可操作覆盖率且建议分歧率仅3.80%。

    

    人工智能越来越多地被用于评估求职候选人，然而一致的面试评分并不能确立自动化建议是否得到了充分证据的支持。本研究提出了证据充分性与决策稳定性评估（ESDSA）框架，该框架结合了标准层面的证据分类、确定性的引文可追溯性，以及在实质等同的语言变体上进行的稳定性检验。我们使用来自软件工程、客户支持和运营领域60个评估场景族的540份合成面试回答，评估了四种方法（仅评分、证据感知、基于可追溯性和集成式ESDSA v2）。在留出测试集（216份回答）中，集成式ESDSA实现了85.19%的可操作覆盖率，且相对于AI辅助的临时参考，可操作建议之间的分歧率仅为3.80%。证据感知评估实现了92.59%的覆盖率……

    arXiv:2611.28426v1 Announce Type: cross  Abstract: Artificial intelligence is increasingly used to assess employment candidates, yet consistent interview scoring does not establish whether automated recommendations are supported by sufficient evidence. This study introduces the Evidence Sufficiency and Decision Stability Assessment (ESDSA), which combines criterion-level evidence classification, deterministic quotation traceability and stability checks across linguistic variants assessed as substantively equivalent. We evaluated four approaches (scoring-only, evidence-aware, traceability-based and integrated ESDSA v2) using 540 synthetic interview responses from 60 evaluated scenario families across Software Engineering, Customer Support and Operations. In held-out testing (216 responses), integrated ESDSA achieved 85.19% actionable coverage and 3.80% disagreement among actionable recommendations against AI-assisted provisional references. Evidence-aware evaluation achieved 92.59% cove
    
[^59]: 把脉企业工作流：评估持久性项目状态理解

    Taking the PULSE of Enterprise Workflows: Evaluating Persistent Project-State Understanding

    [https://arxiv.org/abs/2611.28374](https://arxiv.org/abs/2611.28374)

    本文提出了持久性项目状态理解（PPSU）任务及PULSE基准（包含1,573个问题、覆盖30个企业项目历史），实验发现检索与抽取是当前系统的主要瓶颈，但即使在完美证据之上仍需显式项目状态才能取得进一步提升，证明该任务无法简化为单纯的检索问题。

    

    企业智能体必须对分布在异构工件中、随时间不断演变、且对不同协作者可见程度不均的工作保持最新理解。我们将这一挑战形式化为持久性项目状态理解，并引入PULSE，一个包含1,573个问题、覆盖30个企业项目历史记录的基准。PULSE从带源链接的记录中构建确定性的、按时间索引的项目状态以及有文档记录的协作者视图，并评估三类问题：未闭合事项、项目演变和视图差距。在检索智能体、图增强与记忆增强检索、长上下文以及oracle条件下，我们发现当前系统仍有巨大的提升空间。检索和抽取占据了差距的最大份额，但在金标准证据之上再提供显式项目状态能带来进一步的显著收益，这表明PPSU不能被简化为单纯的检索问题。进一步的分析显示

    arXiv:2611.28374v1 Announce Type: cross  Abstract: Enterprise agents must maintain an up-to-date understanding of work distributed across heterogeneous artifacts, evolving over time, and unevenly visible to different collaborators. We formulate this challenge as persistent project-state understanding (PPSU) and introduce PULSE, a benchmark of 1,573 questions over 30 enterprise project histories. PULSE constructs deterministic, time-indexed project state and documented collaborator views from source-linked records, and evaluates three families of questions: open loops, project evolution, and view gaps. Across retrieval agents, graph- and memory-augmented retrieval, long-context, and oracle conditions, we find substantial headroom for current systems. Retrieval and extraction account for the largest share of the gap, but providing explicit project state on top of gold evidence yields further substantial gains, showing that PPSU cannot be reduced to retrieval alone. Further analysis shows
    
[^60]: 当工具失效时，智能体便捏造结果：工具使用型LLM智能体中的静默失败与“验证优先”监督机制

    When Tools Fail, Agents Fabricate: Silent Failure and Verification-First Oversight in Tool-Using LLM Agents

    [https://arxiv.org/abs/2611.28341](https://arxiv.org/abs/2611.28341)

    该论文构建了RetryRealistic受控故障注入测试平台，揭示了工具使用型LLM智能体在工具失效时恢复能力不可靠且常发生“静默捏造”（将错误答案自信地当作事实输出），失败模式主要取决于故障类型而非模型本身，并据此倡导“验证优先”的监督机制。

    

    使用工具的LLM智能体在工具调用失败时必须能够恢复，例如重试、切换工具或修改计划。我们借助RetryRealistic研究了当工具失效时智能体如何失败——这是一个沙箱化但贴近真实的受控故障注入测试平台。在该平台中，智能体在具有真实权限执行机制的真实操作系统文件系统、SQLite数据库、确定性本地HTTP服务器和受约束的shell上运行，涵盖8种工具、9种具有受控持续性的故障类型以及30个多步骤任务。通过对四个智能体共2,100次故障注入运行进行与真实基准的对比评估，我们发现恢复能力并不可靠（正确率仅为19-68%），且失败常常是静默的：6-26%的运行以“自信的捏造”告终，即错误答案被当作事实进行断言。这种失败模式更多由故障类型而非模型本身决定：返回部分数据的故障会诱发捏造行为（高达38%），而硬性拒绝则会促使智能体诚实地报告失败。其背后机制是我们所称的“静默摄取”：智能体接受……（原文摘要在此处截断）

    arXiv:2611.28341v1 Announce Type: new  Abstract: Tool-using LLM agents must recover when a tool call fails, by retrying, switching tools, or revising their plan. We study how agents fail when tools fail, using RetryRealistic, a sandboxed yet realistic testbed for controlled fault injection in which agents operate on a real OS filesystem with genuine permission enforcement, SQLite databases, a deterministic local HTTP server, and a constrained shell, across 8 tools, 9 fault types with controlled persistence, and 30 multi-step tasks. Evaluating four agents over 2,100 faulted runs against ground truth, we find that recovery is unreliable (19-68% correct) and frequently silent: 6-26% of runs end in a confident fabrication, a wrong answer asserted as fact. The failure mode is governed more by the fault type than by the model: faults that return partial data drive fabrication (up to 38%), while hard denials drive honest failure. The mechanism is what we call silent ingestion: the agent accep
    
[^61]: 软件定义汽车的跨边界漏洞管理：ISO/SAE 21434与欧盟《网络弹性法案》接口处的可追溯性架构

    Cross-Boundary Vulnerability Management for SDVs: A Traceability Architecture at the ISO/SAE 21434 and EU Cyber Resilience Act Interface

    [https://arxiv.org/abs/2611.28303](https://arxiv.org/abs/2611.28303)

    针对软件定义汽车中车辆型式认证法规体系（ISO/SAE 21434、UN R155/R156）与欧盟《网络弹性法案》之间漏洞管理交接的空白，提出了四问题分类程序、五法规体系可追溯性矩阵和含四条移交路径的参考架构，实现跨法规边界的漏洞可追溯管理。

    

    软件定义汽车依赖于处于不同欧盟法规体系之下的产品。通过型式认证的车辆及其配套的系统、组件和独立技术单元受欧盟法规(EU) 2019/2144管辖，该法规使UN R155成为强制性要求；法规(EU) 2018/858对R156作出了同样规定；ISO/SAE 21434则支持R155所要求的管理体系。而充电器、配套应用程序、售后改装设备及许多后端系统可能转而属于《网络弹性法案》（CRA）的管辖范围，该法案第14条规定的报告义务自2026年9月11日起已开始适用。然而，双方均未定义在一方发现的漏洞如何传递到另一方的风险处理流程中。本文的贡献包括：一个四问题分类程序、一个覆盖五类法规体系的可追溯性矩阵、一个包含四条移交路径的参考架构，以及三项边界发现。基于一个冻结的公开来源语料库，对75个已分类产品的分析得出47个外围产品、11个核心产品、1个其他产品和16个边界产品……（摘要原文在此处截断）

    arXiv:2611.28303v1 Announce Type: cross  Abstract: A software-defined vehicle depends on products that sit under different EU regimes. The type-approved vehicle and the systems, components and separate technical units made for it fall under Regulation (EU) 2019/2144, which makes UN R155 mandatory; Regulation (EU) 2018/858 does so for R156; and ISO/SAE 21434 supports the management system R155 requires. Chargers, companion apps, aftermarket devices and many backends may fall under the Cyber Resilience Act (CRA) instead, whose reporting duty in Article 14 has applied since 11 September 2026. Neither side defines how a vulnerability found on one reaches the other's risk process. We contribute a four-question classification procedure, a five-regime traceability matrix, a reference architecture with four handoff paths and three boundary findings. On 75 classified products from a frozen public-source corpus, the procedure yields 47 Periphery, 11 Core, 1 Other and 16 Boundary products, three 
    
[^62]: 越狱攻击是全球性还是区域性的？关于模型规模与地理位置对大语言模型安全验证影响的受控研究

    Jailbreaks Are Global or Regional? A Controlled Study of Scale and Geolocation Effects on LLM Safety Verification

    [https://arxiv.org/abs/2611.28174](https://arxiv.org/abs/2611.28174)

    该研究通过配对两阶段协议评估12个不同规模模型并从三大洲五个城市的VPN节点发送相同提示，发现模型规模与请求地理位置都会显著影响攻击成功率的比较结果，且参数量并非衡量模型安全性的可靠指标。

    

    安全测试被用于比较语言模型并决定其可部署的场景。其主要指标——攻击成功率（ASR）——只有在除模型本身之外其他条件均保持不变时，才能在模型之间进行有效比较。我们测试了两个通常被默认忽略的因素：模型规模以及请求看似来自的地理位置。我们采用配对两阶段协议（先发送普通有害提示，再将相同请求嵌入已发表的越狱模板中）评估了参数量从0.5B到671B的十二个模型。我们还通过横跨三大洲五个城市的VPN出口节点发送相同的提示。在Llama和Qwen模型家族内部，第二阶段的ASR随模型规模急剧下降（Qwen2.5：0.5B为70.8%，7B为51.2%，72B为12.6%）。越狱模板包装在对数几率尺度上使每个模型产生相近幅度的偏移（1.7至2.9）；小模型的偏移略大，但在考虑评审器的误报后，这一差距几乎消失。参数量仅是安全性的一个薄弱代理指标：

    arXiv:2611.28174v1 Announce Type: new  Abstract: Safety tests are used to compare language models and to decide where they can be deployed. Their main number, the attack success rate (ASR), can only be compared across models if nothing but the model changes. We test two factors that are usually assumed away: the size of the model and the place a request seems to come from. We evaluate twelve models from 0.5B to 671B parameters with a paired two-phase protocol (plain harmful prompts, then the same requests inside published jailbreak templates). We also send identical prompts through VPN exit nodes in five cities on three continents. Inside the Llama and Qwen families, Phase~2 ASR falls steeply with size (Qwen2.5: 70.8\% at 0.5B, 51.2\% at 7B, 12.6\% at 72B). Wrapping shifts every model by a similar amount on the log-odds scale (1.7 to 2.9); small models shift slightly more, but that gap nearly vanishes when we allow for judge false positives. Parameter count is a weak proxy for safety: 
    
[^63]: SecretSifter：面向实时Web流量的运行时密钥检测引擎

    SecretSifter: A Runtime Secret-Detection Engine for Live Web Traffic

    [https://arxiv.org/abs/2611.28168](https://arxiv.org/abs/2611.28168)

    SecretSifter是一个开源的运行时密钥检测引擎，通过组合格式锚定与上下文门控规则、语义化密钥名提取、香农熵评分以及CryptoJS/AES硬编码专项检测，在实时Web流量中于凭证交付到浏览器的时刻发现密钥级凭证，填补了传统左移式仓库扫描器无法覆盖的运行时攻击面。

    

    嵌入在客户端交付的Web工件（如压缩后的JavaScript包、webpack代码块、服务端渲染的状态对象以及JSON或XML API响应）中的凭证，构成了一种仓库时（左移）密钥扫描器无法观测到的运行时攻击面，因为这些凭证是在构建时或运行时进入所交付工件的，此时所有源代码门禁检查均已通过。本文提出了SecretSifter，一个开源的运行时密钥检测引擎，它检查实时Web流量，并在密钥级凭证被交付到浏览器的时刻进行报告。该引擎结合了格式锚定与上下文门控的检测规则、语义化密钥名称提取、香农熵评分，以及对硬编码CryptoJS/AES密钥材料的专项检测，覆盖完整的交付面。SecretSifter基于同一个共享检测核心提供三个版本：Burp Suite扩展、Chrome扩展以及独立的Windows和mac（OS应用）。

    arXiv:2611.28168v1 Announce Type: new  Abstract: Credentials embedded in client-delivered web artifacts, such as minified JavaScript bundles, webpack chunks, server-side-rendered state objects, and JSON or XML API responses, form a runtime attack surface that repository-time (shift-left) secret scanners do not observe, because the credential enters the served artifact at build time or at runtime, after every source-code gate has passed. This paper presents SecretSifter, an open-source runtime secret-detection engine that inspects live web traffic and reports secret-grade credentials at the point they are delivered to the browser. The engine combines format-anchored and context-gated detection rules, semantic key-name extraction, Shannon-entropy scoring, and dedicated detection of hardcoded CryptoJS/AES material across the full served surface. SecretSifter ships as three editions over one shared detection core: a Burp Suite extension, a Chrome extension, and a standalone Windows and mac
    
[^64]: 大语言模型能从自然语言生成Android Lint检查到多远的程度？

    How Far Can LLMs Go in Generating Android Lint Checks from Natural Language?

    [https://arxiv.org/abs/2611.28159](https://arxiv.org/abs/2611.28159)

    该论文提出了LintBench基准（包含113个经人工验证的真实Android Lint检查及自然语言描述），首次系统评估了LLM能否仅凭自然语言描述生成可运行的Lint检查及其所需的外部知识。

    

    静态分析工具（如Android Lint、PMD和CodeQL）允许开发人员为项目特定的规则添加自定义检查。然而，编写一个检查是一项要求很高的编程任务：一条规则可以用几句话来表述，但要将其转化为一个可运行且经过测试的检查，需要对分析框架及其问题检测逻辑有深入的了解。最近的研究使用大语言模型（LLM）来生成此类检查，但每个系统都需要开发人员所拥有的自然语言描述之外的额外信息：例如一个错误修复补丁、一条CVE记录，或一个开发人员编写的用于对照修复的测试套件。此前尚无研究衡量LLM能否仅凭描述就生成可运行的检查，或者它需要哪些外部知识才能做到这一点。我们提出了LintBench，这是一个经过人工验证的基准，包含113个真实世界的Android Lint检查，每个检查都配有自然语言描述及其原始测试套件作为保留的验证标准。我们评估了四个LLM，包括propr（原文摘要在此处截断）

    arXiv:2611.28159v1 Announce Type: cross  Abstract: Static analysis tools such as Android Lint, PMD, and CodeQL let developers add custom checks for project-specific rules. However, authoring a check is a demanding programming task: a rule can be stated in a few sentences, yet turning it into a working, tested check requires detailed knowledge of the analysis framework and its issue-detection logic. Recent work generates such checks with large language models (LLMs), but each system needs more than the natural language description a developer always has: a bug-fix patch, a CVE record, or a developer-written test suite to repair against. No prior work measures whether an LLM can turn a description into a working check, or what external knowledge it needs to do so. We introduce LintBench, a human-validated benchmark of 113 real-world Android Lint checks, each paired with a natural language description and its original test suite as a held-out oracle. We evaluate four LLMs, including propr
    
[^65]: 从观察到解释：面向原则性修复的大语言模型生成代码的机制可解释性

    From Observation to Explanation: Mechanistic Interpretability of LLM-Generated Code for Principled Repair

    [https://arxiv.org/abs/2611.28153](https://arxiv.org/abs/2611.28153)

    该论文通过纵向、实证与行为三个层面的证据三角验证，揭示了现有基准高估 LLM 代码生成能力、且 LLM 生成的代码在生产环境中需要不成比例的修正性修改等现象，并从机制可解释性角度为 LLM 生成代码的原则性修复提供了因果解释。

    

    大语言模型（LLM）已经彻底改变了软件开发，但我们对它们如何生成代码的理解仍然严重不足。我们通过在三个层面上进行证据三角验证来弥补这一差距。纵向来看，在语义修改定义下，自主智能体生成的代码与人类代码在统计上无法区分（HR = 1.019，p = 0.490），但在生产环境中却吸引了不成比例的修正性修改（49.1% 对比人类基线的 37.9%）。这一发现需要因果层面的解释。实证上，现有基准高估了 LLM 的能力：在合成任务上达到 84% 至 89% 正确率的模型，在真实世界的类级别代码上仅能达到 24% 至 35%。行为上，检测系统揭示了令人困惑的不对称性：GPT-3.5 远比现代模型更容易被检测到（AUC-ROC = 0.96，相差 27 个百分点），尽管现代模型生成的代码结构彼此相似；然而没有任何单一特征能够……（原文截断）

    arXiv:2611.28153v1 Announce Type: cross  Abstract: Large language models (LLMs) have transformed software development, yet our understanding of how they generate code remains critically underdeveloped. We address this gap by triangulating evidence across three layers. Longitudinally, autonomous agent-attributed code is statistically indistinguishable from human code under a semantic modification definition (HR = 1.019, p = 0.490), yet attracts disproportionately corrective modifications in production (49.1% vs 37.9% human baseline). This finding demands a causal explanation. Empirically, existing benchmarks overestimate LLM capabilities: models that achieve 84 to 89% correctness on synthetic tasks attain only 24 to 35% on real-world class-level code. Behaviourally, detection systems reveal puzzling asymmetries: GPT-3.5 is far more detectable than modern models (AUC-ROC = 0.96, a 27-point gap), even though modern models produce mutually similar code structures; yet no single feature rel
    
[^66]: 每个IVUS帧各就其位：一种将血管内超声配准到冠状动脉造影的零安装浏览器工具，以及迈向仅凭造影进行斑块成像的路径

    Every IVUS Frame in Its Place: A Zero-Install Browser Tool for Registering Intravascular Ultrasound to the Coronary Angiogram, and a Path to Angiography-Only Plaque Imaging

    [https://arxiv.org/abs/2611.28089](https://arxiv.org/abs/2611.28089)

    该论文提出了一种零安装、纯浏览器运行的半自动IVUS-冠脉造影配准工具，可将每个IVUS帧定位到造影血管上并导出帧级训练记录，为仅凭造影实现斑块成像和AI模型训练数据的积累开辟了道路。

    

    血管内超声（IVUS）能够显示冠状动脉造影无法显示的斑块成分、钙化和支架贴壁情况，但将每个IVUS横截面与造影图像上的位置相关联，目前仍主要依靠操作者头脑中的判断。商用的共同配准系统绑定于厂商工作站和采集时的设置，因此存档的造影图像和回撤图像无法进行配准，用于训练AI模型的配对数据依然稀缺。我们提出了一种完全在网页浏览器中运行的半自动配准工具，无需安装、无需上传。临床医生选择一个造影剂充盈的参考帧（借助自动造影剂和心脏时相分析辅助），选择或描记血管中心线（借助Frangi血管性滤波与最小路径方法），并标记回撤的起止点和可见的解剖标志；随后通过单调分段线性映射将每个IVUS帧定位到血管上，每个病例可导出为帧级训练记录。原文在此处被截断。

    arXiv:2611.28089v1 Announce Type: cross  Abstract: Intravascular ultrasound (IVUS) shows plaque composition, calcium and stent apposition that the coronary angiogram cannot, but relating each IVUS cross-section to its position on the angiogram is still largely done in the operator's head. Commercial co-registration is tied to vendor consoles and to set-up at acquisition, so archived angiograms and pullbacks cannot be co-registered, and paired data for training AI models remain scarce. We present a semi-automatic registration tool that runs entirely in a web browser, with no installation and no upload. The clinician picks a contrast-filled reference frame (aided by automatic contrast and cardiac-phase analysis), selects or traces the vessel centerline (aided by Frangi vesselness and minimal paths), and pins the pullback ends and visible landmarks; a monotone piecewise-linear map then places every IVUS frame on the vessel, and each case exports as a frame-level training record. The origi
    
[^67]: 一种用于验证可穿戴运动感知系统的无代码工具链

    A No-Code Toolchain for Validating Wearable Motion Sensing Systems

    [https://arxiv.org/abs/2611.28027](https://arxiv.org/abs/2611.28027)

    该论文提出了一种无需编程的无代码工具链，让非技术用户能够通过摄像头提取真实关节角度数据、训练和测试机器学习模型并可视化分析传感器布局，从而低成本、高效率地验证可穿戴电子织物运动感知系统的设计。

    

    为人体运动感知设计电子织物（e-textiles）通常需要传感器数据处理方面的工程专业知识，且往往依赖先进的机器学习方法。此外，验证设计——收集传感器测量数据和真实运动数据、训练和测试机器学习模型、以及迭代调整传感器布局——可能既费力又耗时。为解决这一问题，我们提出了一种无代码工具链，使非技术用户能够高效地验证系统设计。该工具提供三项关键功能：(1) 从网络摄像头录像中提取关节角度的真实测量值；(2) 使用收集的传感器数据训练和测试机器学习模型；(3) 可视化和分析运动感知性能及传感器布局选择。我们的工具只需简单、低成本的设置，并能在几分钟内生成测试报告。我们进行了一项用户研究以验证该工具链的可用性。

    arXiv:2611.28027v1 Announce Type: cross  Abstract: Designing e-textiles for human motion sensing often demands engineering expertise for sensor data processing, typically relying on advanced machine learning methods. In addition, validating designs---collecting sensor measurements and ground-truth motion data, training and testing machine learning models, and iterating on sensor layouts---can be labour- and time-intensive. To address this, we present a no-code toolchain that enables non-technical users to validate system designs efficiently. The tool provides three key capabilities: (1) deriving ground-truth joint-angle measurements from webcam recordings; (2) training and testing machine learning models using collected sensor data; and (3) visualising and analysing motion sensing performance and sensor layout choices. Our tool requires a simple, low-cost setup and produces test reports within minutes. A user study was conducted to validate the usability of this toolchain. This toolcha
    
[^68]: 技术赋能的燃料供应链韧性：一个用于实时监控与决策支持的集成信息系统

    Technology-Enabled Fuel Supply Chain Resilience: An Integrated Information System for Real-Time Monitoring and Decision Support

    [https://arxiv.org/abs/2611.25542](https://arxiv.org/abs/2611.25542)

    本研究将燃料物流信息系统重构迁移至SQL Server平台，使数据集成缺陷率从2.90%–3.46%降至0.14%–0.49%，并通过新增库存移动记录、自动警报和恢复控制功能，提升了燃料供应链的实时监控与决策支持能力。

    

    本研究评估并扩展了早期燃料物流研究中所描述的事务性信息系统，重点关注数据可靠性、监控与决策支持。由作者主导的重新评估审查了现有的Delphi和SQL代码，并在十二个评估场景中将重构的源数据基线系统（R0）与修订后的SQL Server系统（R1）进行对比。技术评估采用了合成交易数据和公开的美国能源数据，同时对组织流程和供应延迟进行了模拟。在受控的SQL Server测试中，R1改善了数据集成和报告准确性，并引入了库存移动记录、自动警报和恢复控制机制。在可比的数据源上，集成缺陷率从R0的2.90%至3.46%降至R1的0.14%至0.49%。在国家周度需求预测方面，所选方法并未显著优于上一年度均值。原文在流程与中断部分截断。

    arXiv:2611.25542v1 Announce Type: new  Abstract: This study evaluates and extends the transactional information system described in an earlier fuel logistics study, focusing on data reliability, monitoring, and decision support. An author-led re-evaluation examined available Delphi and SQL code and compared a reconstructed source-derived baseline (R0) with a revised SQL Server system (R1) across twelve evaluation scenarios. Synthetic transactions and public US energy data were used for technical assessment, while organizational processes and supply delays were simulated. R1 improved data integration and reporting accuracy in controlled SQL Server tests and introduced stock-movement records, automated alerts, and recovery controls. On comparable data feeds, integration defect rates decreased from 2.90 to 3.46 percent in R0 to 0.14 to 0.49 percent in R1. For national weekly demand forecasting, selected methods did not significantly outperform the previous-year mean. Process and disruptio
    
[^69]: 通过线程模块化抽象解释验证带幽灵变量的并发正确性见证

    Validating Concurrent Correctness Witnesses with Ghosts by Thread-Modular Abstract Interpretation

    [https://arxiv.org/abs/2611.23792](https://arxiv.org/abs/2611.23792)

    提出了一种新颖的并发正确性见证验证框架，利用幽灵变量中编码的语义信息，通过线程模块化抽象解释来验证带幽灵变量的并发正确性见证，而非将其简单视为额外的证明义务。

    

    并发程序的编写和推理是出了名的困难，即使是基于健全理论基础的工具，也可能由于实现缺陷而产生错误的结果。正确性见证被提出作为一种通过支持独立验证来提高软件验证器判定结果可信度的方法。最近，通过引入幽灵变量以允许编码更具表达力的不变式，正确性见证已被扩展到并发场景。尽管存在多种生成带幽灵变量见证的方法，但相应的验证技术仍然缺乏；验证器并没有利用幽灵变量中编码的信息来指导验证过程，而只是将幽灵插桩和关于幽灵变量的不变式视为额外的证明义务。我们提出了一种新颖的并发正确性见证验证框架，该框架利用幽灵变量中编码的语义信息……

    arXiv:2611.23792v1 Announce Type: cross  Abstract: Concurrent programs are notoriously difficult to write and to reason about, and even tools based on sound theoretical foundations can produce incorrect results due to implementation bugs. Correctness witnesses have been proposed as a way to increase the confidence in the verdicts of software verifiers by enabling independent validation. Recently, correctness witnesses have been extended to the concurrent setting, by introducing ghost variables to allow encoding more expressive invariants. While different approaches to producing witnesses with ghosts exist, validation techniques are still lacking; instead of leveraging the information encoded in the ghosts to guide the validation process, validators merely treat ghost instrumentation and invariants over ghost variables as additional proof obligations. We propose a novel framework for the validation of concurrent correctness witnesses that exploits the semantic information encoded in gho
    
[^70]: 基于大语言模型的HLS生成中的规则引导与轨迹感知修复方法

    Rule-Guided and Trajectory-Aware Repair for LLM-Based HLS Generation

    [https://arxiv.org/abs/2611.07324](https://arxiv.org/abs/2611.07324)

    MementoHLS框架通过可执行修复契约库（将跨任务HLS知识转化为证据驱动的修复动作）与证伪账本（将修复历史转化为轨迹证据以抑制无效路径）的双记忆引导机制，并借助同一候选评审者实现可靠的结果归因，显著提升了大语言模型从自然语言规范生成可综合HLS-C代码的端到端成功率。

    

    大语言模型为直接从自然语言规范生成功能正确且可综合的HLS-C代码提供了一条有前景的路径。然而，现有方法往往难以将多轮工具反馈和可复用的HLS知识转化为有效的修订，限制了端到端生成的成功率。我们提出了MementoHLS，这是一个围绕可执行且可证伪的双记忆引导修复构建的智能体式HLS生成框架。MementoHLS将可执行修复契约库（ERCL）与基于评审者的证伪账本（RFL）相结合：前者将跨任务的HLS知识转化为由证据触发的修复动作；后者将任务局部的修复历史转化为轨迹证据，以抑制无效的修复路径并保留已验证的进展。为确保可靠的结果归因，同一候选评审者将解析、编译、测试平台（TB）和综合（Synth）的结果绑定到相应的候选方案上。

    arXiv:2611.07324v1 Announce Type: cross  Abstract: Large language models (LLMs) offer a promising path for directly generating functionally correct and synthesizable HLS-C from natural-language specifications. However, existing approaches often struggle to turn multi-round tool feedback and reusable HLS knowledge into effective revisions, limiting end-to-end generation success. We propose MementoHLS, an agentic HLS generation framework built around executable and falsifiable dual-memory-guided repair. MementoHLS combines an Executable Repair Contract Library (ERCL), which converts cross-task HLS knowledge into evidence-gated repair actions, with a Reviewer-Grounded Falsification Ledger (RFL), which converts task-local repair history into trajectory evidence to suppress ineffective routes and preserve validated progress. To ensure reliable outcome attribution, a Same-Candidate Reviewer binds Parse, Compile, Testbench (TB), and Synthesis (Synth) outcomes to the corresponding candidate, w
    
[^71]: SWE-Ladder：通过渐进式特权自蒸馏学习定位、复现与修补

    SWE-Ladder: Learning to Localize, Reproduce, and Patch via Progressive Privileged Self-Distillation

    [https://arxiv.org/abs/2610.33667](https://arxiv.org/abs/2610.33667)

    该论文提出SWE-Ladder框架，将已验证的修复过程分解为定位、复现和修补三个阶段的特权证据，通过基于规则的路由器和冻结教师模型按修复顺序进行渐进式自蒸馏，并结合GRPO优化完整轨迹，为SWE智能体提供了传统强化学习和OPSD所缺失的阶段特定训练指导。

    

    解决仓库级问题通常需要软件工程（SWE）智能体定位故障代码、复现失败并进行代码修补。强化学习通过最终补丁的单一测试结果来训练此类智能体，而策略内自蒸馏（OPSD）则从拥有特权证据的教师模型中添加token级监督，在每一步都暴露完整的参考补丁。然而，两者都无法为定位或复现阶段提供针对性的指导。我们提出SWE-Ladder，将每个训练实例的已验证修复转化为特定阶段的证据：用于定位的变更文件和行号、用于复现的fail-to-pass测试，以及用于修补的代码差异。基于规则的路由器根据工具调用将端到端rollout的每一步分配到相应阶段，冻结的教师模型只接收与该阶段匹配的证据，各阶段按修复顺序依次进行蒸馏，同时由GRPO优化完整轨迹。使用Qw…（原文摘要截断）

    arXiv:2610.33667v1 Announce Type: new  Abstract: Resolving a repository-level issue typically requires a software-engineering (SWE) agent to localize the faulty code, reproduce the failure, and patch the code. Reinforcement learning trains such agents from a single test result on the final patch, while on-policy self-distillation (OPSD) adds token-level supervision from a teacher with privileged evidence, exposing the full reference patch at every step. Neither provides stage-specific guidance for localization or reproduction. We propose SWE-Ladder, which turns the verified repair of each training instance into stage-specific evidence: the changed files and lines for localization, the fail-to-pass tests for reproduction, and the diff for patching. A rule-based router assigns each step of an end-to-end rollout to a stage based on its tool call, a frozen teacher sees only the matching evidence, and the stages are distilled in repair order while GRPO optimizes the full trajectory. With Qw
    
[^72]: VERDICT：多智能体并行模块化验证

    VERDICT: Multi-Agent Parallel Modular Verification

    [https://arxiv.org/abs/2610.29147](https://arxiv.org/abs/2610.29147)

    VERDICT是一个多智能体框架，通过基于假设的并行规范综合打破传统自底向上验证的依赖顺序，实现大规模程序模块化演绎验证的并行化。

    

    演绎验证通过模块化推理扩展到大型软件系统，其中形式化契约总结单个函数的行为，使其能够独立验证。其核心瓶颈在于规范综合：即构建模块化验证所需的契约。大语言模型（LLM）的最新进展使自动化规范综合达到了前所未有的规模。然而，现有方法在根本上仍是自底向上的，要求被调用函数必须先于其调用者完成验证。这种依赖关系限制了并行性，且缺乏一种有原则的机制来解决沿调用链传播的过程间不一致性问题。我们提出了VERDICT，一个用于并行化大规模程序模块化演绎验证的多智能体框架。VERDICT建立在两个关键原则之上。首先，基于假设的并行规范综合为每个函数分配一个智能体，并……

    arXiv:2610.29147v1 Announce Type: new  Abstract: Deductive verification scales to large software systems through modular reasoning, where formal contracts summarize the behavior of individual functions and enable their independent verification. Its central bottleneck is specification synthesis: constructing the contracts required for modular verification. Recent advances in large language models (LLMs) have enabled automated specification synthesis at unprecedented scale. However, existing approaches remain fundamentally bottom-up, requiring callees to be verified before their callers. This dependency limits parallelism and provides no principled mechanism for resolving interprocedural inconsistencies that propagate across call chains.   We present VERDICT, a multi-agent framework for parallelizing modular deductive verification of large-scale programs. VERDICT is built on two key principles. First, assumption-based parallel specification synthesis assigns an agent to each function and
    
[^73]: HPCAgent-Bench：一个用于代码优化智能体、执行框架与技能的可靠基准测试框架

    HPCAgent-Bench: A Framework for Sound Benchmarking of Code-Optimizing Agents, Harnesses, and Skills

    [https://arxiv.org/abs/2610.20673](https://arxiv.org/abs/2610.20673)

    该论文提出了HPCAgent-Bench——一个包含689个内核、覆盖三个赛道的代码优化基准框架，通过统一的内核规范、外部防作弊裁判以及“干预效力”等新指标，实现对代码优化智能体、执行框架与技能各自贡献的可靠隔离评估。

    

    代码智能体正被越来越多地用于优化代码，但现有的基准测试通常固定单一的加速器、编程模型或执行框架，因此无法隔离各组件各自的贡献，也无法衡量新增的技能和工具是否物有所值。我们提出了HPCAgent-Bench，其中包含689个内核，分为三个赛道：循环级优化挑战、从高性能计算（HPC）代码中提取的内核，以及机器学习模型。该框架用NumPy对每个内核进行一次性定义，推导出C和Fortran参考实现，通过统一的C ABI或Python接口接受提交，并可在单节点或多节点环境下的CPU或GPU上运行。一个外部裁判会在隐藏的模糊测试输入上检查正确性并对每次提交进行计时，从而封堵智能体利用测试系统的途径。我们引入了两个用于报告结果的指标：\textit{干预效力}通过配对运行在加速比、成功率和成本等方面衡量某项干预（例如添加工具）的效果，以及\textit{加权……（摘要在此处截断）

    arXiv:2610.20673v1 Announce Type: new  Abstract: Coding agents are increasingly used to optimize code, but existing benchmarks fix one accelerator, programming model or harness, so they cannot isolate what each component contributes or show whether added skills and tools repay their cost. We present HPCAgent-Bench with 689 kernels in three tracks: loop-level optimization challenges, kernels extracted from HPC codes, and machine-learning models. The framework specifies each kernel once in NumPy, derives C and Fortran references, accepts submissions through a common C ABI or Python interface, and runs them on CPUs or GPUs, on one or multiple nodes. An external judge checks correctness on hidden fuzzed inputs and times each submission, closing paths agents use to exploit the testing system. We introduce two metrics for reporting results: \textit{intervention efficacy} measures an intervention, such as adding a tool, through paired runs on speedup, success rate and cost, and \textit{weight
    
[^74]: 通过结合模型验证与运行时监控来验证Autoware控制器节点

    Verifying Autoware Controller Node by Combining Model Verification and Runtime Monitoring

    [https://arxiv.org/abs/2610.15418](https://arxiv.org/abs/2610.15418)

    本文提出一种将微分动态逻辑形式化模型验证与运行时监控相结合的混合验证方法，用于验证开源自动驾驶平台Autoware轨迹跟随控制节点的安全性与活性属性。

    

    由于运行空间庞大且控制逻辑复杂，自动驾驶系统的形式化建模与验证仍然充满挑战。我们提出了一种混合验证方法，将形式化模型验证与运行时监控相结合，应用于开源自动驾驶平台Autoware的轨迹跟随控制节点。我们将该控制器形式化为微分动态逻辑中的混合系统，并引入一个捕捉核心控制动力学的抽象模型，随后使用KeYmaera X证明助手证明了该模型的安全性和活性属性。作为补充，我们在基于Autoware工具链的数值仿真运行中监控确保模型与Autoware实现之间兼容性的约束条件，量化了实现行为在多大程度上被已验证模型所涵盖。实验结果证实了……

    arXiv:2610.15418v2 Announce Type: replace  Abstract: Formal modeling and verification of autonomous driving systems remain challenging due to the vast operational space and the complexity of the control logic. We propose a hybrid verification approach that combines formal model verification and runtime monitoring for the trajectory-following control node of Autoware, an open-source autonomous driving platform. We formalize the controller as a hybrid system in differential dynamic logic (dL), and introduce an abstract model that captures the essential control dynamics. We then prove safety and liveness properties of the model using the KeYmaera X proof assistant. Complementarily, we monitor the constraints that assure the compatibility between the model and the Autoware implementation across numerical simulation runs using the Autoware toolchain, quantifying to what extent the implementation's behavior is subsumed by the verified model. The experimental results confirmed that (i) monito
    
[^75]: SaltBench：用于在机器校验软件工作中测量方法效应的裁判门控协议

    SaltBench: A Referee-Gated Protocol for Measuring Method Effects in Machine-Checked Software Work

    [https://arxiv.org/abs/2609.11076](https://arxiv.org/abs/2609.11076)

    提出了SaltBench，一个由机器裁判门控的基准测试协议，通过严格的隔离机制（经探针实际检验）、预注册预测和“停止即暂停”等规则，使机器裁判对编程智能体工作方式的影响变得可测量且不可事后叙述。

    

    SaltBench 是一个基准测试协议，旨在回答一个问题：机器裁判如何改变编程智能体的工作方式？机器裁判——即证明内核、程序验证器或保留的测试套件——决定智能体工作的价值，而智能体无法与其争辩。我们在此报告一种协议，使裁判的效应可被测量，且其结果无法在事后被主观叙述：每个结果都由智能体自身工具链之外裁决；智能体与网络、参考解决方案及测试框架本身被完全隔离，并在任何计分运行之前，通过试图突破隔离的探针来检验隔离墙，因此这种隔离是被观察到的而非被假设的；每次运行都由带有注册预测的日期化冻结授权；预算停止意味着暂停，而绝非失败。在本研究中，基准测试的对象是一个“席位”（seat），即标准测试框架中的一次智能体会话。我们测试了五个系统组件，全部由……

    arXiv:2609.11076v1 Announce Type: new  Abstract: SaltBench is a benchmark protocol for one question: How does a machine referee change the way a coding agent works? A machine referee --- a proof kernel, a program verifier, or a withheld test suite --- decides what an agent's work is worth, and the agent cannot argue with it. Here we report a protocol that makes the referee's effect measurable and whose answers cannot be narrated afterwards: every outcome is decided outside the agent's own toolchain; the agent is walled off from the network, the reference solutions and the harness itself, and the wall is tested by probes that try to breach it before any scored run, so the isolation is observed rather than assumed; every run is authorized by a dated freeze with its predictions registered; and a budget stop is a halt, never a failure. In this study, the subject of the benchmark is a ``seat'', meaning an agent session in its standard harness. We tested five systems components, all authored
    
[^76]: 来自地狱的更新：编码智能体能否在依赖升级的隐性破坏中存活下来？

    Update from Hell: Can Coding Agents Survive Hidden Breakage in Dependency Upgrades?

    [https://arxiv.org/abs/2608.30300](https://arxiv.org/abs/2608.30300)

    该论文提出了 DEPBENCH 基准，包含跨越五个语言社区五个包生态系统的 203 个真实依赖升级任务，用于评估编码智能体能否应对依赖升级中隐藏的代码级破坏并完成源代码适配。

    

    现代软件系统严重依赖第三方依赖项，但升级这些依赖项仍然是一项成本高昂的维护活动。依赖升级并不总是能保留现有代码所依赖的函数签名、类型系统、API 或运行时语义。因此，开发人员通常需要执行源代码适配来应对依赖项引发的变化。然而，此类代码级别的更改往往不会明确传达给项目维护人员，这对软件可靠性构成了重大挑战。与此同时，编码智能体作为一种新型软件开发工具出现，并因其自动化能力而被开发人员日益广泛地采用。在本文中，我们介绍了 DEPBENCH，这是一个由 203 个真实世界依赖升级任务组成的基准测试，涵盖五个语言社区的五个包生态系统，每个任务都涉及需要源代码适配的隐藏代码级更改。

    arXiv:2608.30300v1 Announce Type: new  Abstract: Modern software systems rely heavily on third-party dependencies, but upgrading those dependencies remains a costly maintenance activity. Dependency upgrades do not always preserve the function signatures, type systems, APIs, or runtime semantics assumed by existing code. Consequently, developers often need to perform source code adaptations to accommodate dependency-induced changes. However, such code-level changes are often not explicitly communicated to project maintainers, posing a significant challenge to software reliability.   Meanwhile, coding agents have emerged as a new form of software development tool and are increasingly adopted by developers due to their automation capabilities. In this paper, we introduce DEPBENCH, a benchmark consisting of 203 real-world dependency-upgrade tasks across five package ecosystems spanning five language communities, each involving hidden code-level changes that require source code adaptation. 
    
[^77]: SWE-Pruner Pro：代码大模型自身早已知道该剪枝什么

    SWE-Pruner Pro: The Coder LLM Already Knows What to Prune

    [https://arxiv.org/abs/2607.18213](https://arxiv.org/abs/2607.18213)

    SWE-Pruner Pro利用代码智能体自身的内部表示直接在模型内部剪枝工具输出，无需外部分类器，最多可节省39%的token且不损失甚至提升任务性能。

    

    对编程智能体进行长上下文剪枝一直是实现高效上下文管理的一项关键技术。现有的上下文剪枝方法（如SWE-Pruner）通过附加一个独立的代码分类器来实现这一目标，而我们发现，智能体自身在读取工具输出时，其内部表示中已经编码了代码上下文的相关性信息。基于这一发现，我们提出了SWE-Pruner Pro，它直接在智能体内部对工具输出进行剪枝。具体而言，一个小型头网络将智能体自身的内部表示转化为每一行的“保留或剪枝”标签，并结合与每个工具输出行数相关联的长度感知嵌入。在两个开源权重骨干模型和四个多轮基准测试中，SWE-Pruner Pro在保持任务质量的同时，最多可节省39%的提示与补全token，且推理开销可控。值得注意的是，在MiMo-V2-Flash上，SWE-Pruner Pro还将SWE-Bench Verified的解决率额外提升了3.8%，并提升了长上下文Ool（摘要原文在此处截断）

    arXiv:2607.18213v2 Announce Type: replace  Abstract: Pruning long context for coding agents has been a vital technology for efficient context management. While existing context pruning methods such as SWE-Pruner realize this by attaching a separate code classifier, we find the agent itself encodes internal representations indicating the relevance of code context when reading tool output. Based on this finding, we propose SWE-Pruner Pro, which prunes tool outputs directly inside the agent. Concretely, a small head turns the agent's own internal representations into a keep-or-prune label for each line, with a length-aware embedding keyed to each tool output's line count. Across two open-weight backbones and four multi-turn benchmarks, SWE-Pruner Pro saves up to 39% of prompt and completion tokens while preserving task quality, with bounded inference overhead. Notably, on MiMo-V2-Flash SWE-Pruner Pro additionally raises the SWE-Bench Verified resolve rate by +3.8% and the long-context Ool
    
[^78]: 通过任务分解阐明AI生成科学分析中的假设

    Articulating Assumptions in AI-Generated Scientific Analyses through Task Decomposition

    [https://arxiv.org/abs/2607.05762](https://arxiv.org/abs/2607.05762)

    提出多智能体框架“基于量化语义差异比较”，通过任务分解揭示AI生成科学分析代码中的隐含假设，重建关键输出量的产生过程并检测用户指令中的歧义。

    

    由大语言模型生成的分析代码所产生的科学结果必须易于理解且可复现。然而，不确定性可能在流程的不同阶段产生，既存在于原始的自然语言规范中，也存在于生成的代码实现中。因此，即使是可执行的代码，也可能无法清晰地说明正在计算哪些量，或者哪些假设决定了最终结果。为了应对这一挑战，我们提出了“基于量化语义差异比较”，这是一个用于分析和比较由大语言模型生成的科学程序的多智能体框架。该框架将代码生成、执行、追踪和验证分配给不同的智能体，从而能够重建关键输出量是如何产生的，并识别预期分析与实际实现代码之间的差异。我们还引入了一个模块，用于检查初始用户指令中的歧义并提出替代性的重写建议。

    arXiv:2607.05762v2 Announce Type: replace  Abstract: Scientific results produced by LLM generated analysis code must be understandable and reproducible. However, uncertainty can arise at different stages of the process, both in the original natural language specification and in the generated implementation. As a result, even executable code may not provide a clear understanding of which quantities are being computed or which assumptions determine the final results. To address this challenge, we introduce quantity grounded semantic differencing, a multi-agent framework for analyzing and comparing scientific programs generated by LLMs. The framework assigns code generation, execution, tracing, and validation to separate agents, allowing it to reconstruct how key output quantities are produced and to identify differences between the intended analysis and the implemented code. We also introduce a module that inspects ambiguities in the initial user instruction and suggests alternative rewr
    
[^79]: GameEngineBench：在网络化游戏引擎中对编码智能体进行全程会话验证

    GameEngineBench: Full-Session Verification of Coding Agents in a Networked Game Engine

    [https://arxiv.org/abs/2607.03525](https://arxiv.org/abs/2607.03525)

    该论文提出了 GameEngineBench 基准（包含源自虚幻引擎 5 项目的 110 个 C++ 实现任务），并引入全程会话验证框架，通过在 PIE 会话（含联网客户端世界）中注入隐藏测试，来评估编码智能体所生成代码的行为正确性与跨客户端一致性。

    

    编码智能体在游戏开发中面临新的挑战，它们必须生成不仅语法正确，而且与引擎管理的生命周期、状态和网络行为保持一致的代码。然而，现有的游戏开发基准测试未能将针对引擎运行时状态的断言与跨客户端一致性相结合，因而无法评估生成的代码在多大程度上符合预期的行为规范。为了解决这些局限性，我们提出了 GameEngineBench，这是一个包含 110 个 C++ 实现任务的基准测试，这些任务源自 9 个现有的虚幻引擎 5（Unreal Engine 5）项目。每个任务要求智能体在既定的游戏玩法架构中实现缺失的行为。我们还引入了全程会话验证，这是一个测试框架，它在智能体完成任务后注入隐藏测试，并在编辑器内运行会话中评估最终实现，其中 71 个任务包含联网的客户端世界。测试……

    arXiv:2607.03525v3 Announce Type: replace-cross  Abstract: Coding agents face new challenges in game development, where they must produce code that is not only syntactically correct but also consistent with engine-managed lifecycle, state, and networking behavior. Existing game-development benchmarks, however, do not combine assertions over engine-based runtime states with cross-client consistency, failing to evaluate how well generated code adheres to the desired behavioral specifications. To address these limitations, we present GameEngineBench, a benchmark of 110 C++ implementation tasks derived from 9 existing Unreal Engine 5 projects. Each task requires an agent to implement missing behavior within an established gameplay architecture. We also introduce Full-Session Verification, a testing framework that injects hidden tests after the agent finishes and evaluates the resulting implementation inside Play-in-Editor (PIE) sessions, with networked client worlds in 71 tasks. The tests 
    
[^80]: BeSpec：面向代码生成的行为级规格对齐

    BeSpec: Behavior-Level Specification Alignment for Code Generation

    [https://arxiv.org/abs/2607.02949](https://arxiv.org/abs/2607.02949)

    本文提出BeSpec，一种基于行为模型的规格对齐方法，通过提取结构化规格并预测可执行行为来检测意图与实现之间的不匹配，解决了LLM代码生成中因规格不匹配而错误实现意图的核心瓶颈。

    

    大语言模型（LLM）已经推动了从自然语言描述（即意图）生成代码的进展。现有大多数方法利用执行反馈来改进代码，同时保持底层规格说明不变，隐含地假设LLM正确理解了意图。然而，模糊或不完整的描述可能导致即使是能力很强的模型也“正确地”实现了错误的意图，这使得规格不匹配成为核心瓶颈。本文提出了BeSpec，一种基于行为模型的规格对齐方法。BeSpec将意图视为预期程序行为的部分证据，从中提取结构化的规格说明，并预测可执行的行为，以刻画正确程序应满足的属性。这种方式避免了需要精确的测试预言（test oracles），因为预测精确测试预言的难度可能与求解问题本身相当。BeSpec在探针输入上执行候选程序并比较其观测到的行为……

    arXiv:2607.02949v2 Announce Type: replace  Abstract: Large language models (LLMs) have advanced code generation from natural language descriptions of desired behavior (intent). Most existing methods refine code using execution feedback while leaving the underlying specification unchanged, implicitly assuming that the LLM correctly understands the intent. Yet ambiguous or underspecified descriptions can lead even capable models to correctly implement the wrong intent, making specification mismatch a central bottleneck.   This paper presents BeSpec, a behavioral model based approach to specification alignment. Treating intent as partial evidence of intended program behavior, BeSpec extracts a structured specification and predicts executable behaviors that capture properties a correct program should satisfy. These avoid requiring exact test oracles, which can be as hard to predict as the solution itself. BeSpec executes candidate programs on probe inputs and compares their observed behavi
    
[^81]: PyMETA：在首次执行错误及更广范围上评估学生代码诊断

    PyMETA: Evaluating Student Code Diagnosis on and Beyond the First Execution Error

    [https://arxiv.org/abs/2606.30610](https://arxiv.org/abs/2606.30610)

    该论文提出了PyMETA数据集，包含48,646份学生Python代码提交及三层级错误分类体系（最细粒度含14个标签），首次系统评估了大语言模型在首次执行错误及超越首次错误（如逻辑错误）层面诊断学生代码的能力。

    

    大型语言模型能够根据学生的代码、问题描述和参考答案来诊断学生程序的问题。评估这一能力需要明确定义什么才算正确的诊断。我们提出了PyMETA，一个包含48,646份针对155道题目的学生提交的Python错误数据集。每份提交都有一个由在线评测系统识别的首次执行错误的单一标签，当程序通过所有测试时则标注为“无错误”。其中97份提交组成的目标子集还包含通过迭代修复和重新执行收集的专家标注标签。该分类体系共有三个层级，最详细的层级包含14个标签，包括无错误、逻辑错误、命名的Python异常以及其他错误类别。我们评估了两个微调模型和两组基于提示的大语言模型：四个早期模型和四个近期模型。在以首次执行错误为基准进行评估时，近期基于提示的模型达到87.5–93.8%的宏观F1分数，超过了强大的（摘要在此处被截断）

    arXiv:2606.30610v2 Announce Type: replace  Abstract: Large language models can diagnose a student program from its code, problem statement, and reference solution. Evaluating this ability requires a clear definition of what counts as the correct diagnosis. We introduce PyMETA, a Python error dataset with 48,646 student submissions to 155 problems. Every submission has a single label for the first execution error identified by an Online Judge, or No Error when the program passes all tests. A targeted subset of 97 submissions also has expert labels collected through iterative repair and re-execution. The taxonomy has three levels; its most detailed level contains 14 labels, including No Error, Logic Error, named Python exceptions, and an Other Errors category. We evaluate two finetuned models and two groups of prompted LLMs: four earlier models and four recent models. When evaluated against the first execution error, the recent prompted models reach 87.5--93.8% macro F1, above the strong
    
[^82]: 基于LLM引导的逐步精化实现可形式化验证且人类可理解的RTL生成

    Formally-Verifiable and Human-Interpretable RTL Generation with LLM-Guided Stepwise Refinement

    [https://arxiv.org/abs/2606.19387](https://arxiv.org/abs/2606.19387)

    该论文提出将大语言模型的生成能力与逐步精化的形式化方法相结合，把自然语言需求转化为形式化硬件契约，并在每一步设计决策变换前进行形式化验证，从而实现构造即正确、可形式化验证且人类可理解的RTL硬件代码生成。

    

    大语言模型（LLM）正在加速寄存器传输级（RTL）的开发，但确保生成设计的正确性仍然是一个重大挑战。现有的LLM驱动的工作流程严重依赖基于仿真的验证，这只能提供有限的正确性保证，且通常对生成的RTL背后的设计决策缺乏洞察。我们提出了一个LLM引导的、构造即正确的RTL生成框架，该框架将LLM的生成能力与逐步精化的数学严谨性相结合。该框架将自然语言需求转化为形式化硬件契约，并逐步将契约精化为实现。在每个精化步骤中，由LLM提出设计决策，而精化引擎仅在形式化验证其正确性之后才应用相应的转换。最终得到的程序针对契约进行形式化验证（原文摘要至此截断）。

    arXiv:2606.19387v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) are accelerating register-transfer level (RTL) development, but ensuring the correctness of generated designs remains a major challenge. Existing LLM-driven workflows rely heavily on simulation-based verification, which provides limited correctness guarantees, and often offer little insight into the design decisions behind the generated RTL. We present a framework for LLM-guided, correct-by-construction RTL generation that combines the generative capabilities of LLMs with the mathematical rigor of stepwise refinement. The framework translates natural-language requirements into a formal hardware contract and progressively refines the contract into an implementation. At each refinement step, the LLM proposes a design decision, while the refinement engine applies the corresponding transformation only after formally verifying its correctness. The resulting program is formally verified against the contra
    
[^83]: 当下一步不止一步：面向并发 Go 程序的分布感知执行建模

    When the Next Step Is Not One Step: Distribution-Aware Execution Modeling for Concurrent Go Programs

    [https://arxiv.org/abs/2606.17508](https://arxiv.org/abs/2606.17508)

    该论文将并发 Go 程序的“下一事件预测”从单标签分类重新形式化为对经验性下一事件分布的估计，并发布了基于真实 GoKer 缺陷内核的基准，但实验发现使用交叉熵和 KL 目标微调的 Qwen2.5-Coder-7B 适配器与简单基线在统计上无显著差异，是一个中性偏负面的结果。

    

    将并发程序的下一个调度器事件预测视为单标签分类是一个病态问题：从同一轨迹前缀出发的重复运行会产生不同但同样有效的下一事件。我们将并发 Go 程序的下一事件预测形式化为估计从重复执行中获得的经验性下一事件分布，并发布了一个基准，其中包含来自手写与生成程序的 945 个训练样本，以及在来自 CockroachDB、Kubernetes、gRPC 和 etcd 等系统的 54 个真实世界 GoKer 缺陷内核上的 798 个留存预测。我们在交叉熵目标和 KL 散度（对齐经验分布）目标下使用低秩适配器微调 Qwen2.5-Coder-7B，并将其与零样本模型和简单基线进行比较。结果是一个中性偏负面的结果：微调后的适配器（重新评估：交叉熵 36.3%，KL 33.6%；最初报告为 36.2% 和 35.8%）在统计上与总是预测最常见事件的平凡基线难以区分。

    arXiv:2606.17508v2 Announce Type: replace  Abstract: Predicting the next scheduler event of a concurrent program is ill-posed as single-label classification: repeated runs from the same trace prefix produce different, equally valid next events. We formulate next-event prediction for concurrent Go programs as estimating an empirical next-event distribution obtained from repeated executions, and release a benchmark with 945 training examples from hand-written and generated programs and 798 held-out predictions on 54 real-world GoKer bug kernels from systems such as CockroachDB, Kubernetes, gRPC and etcd. We fine-tune Qwen2.5-Coder-7B with low-rank adapters under a cross-entropy objective and a KL-to-empirical-distribution objective and compare them with zero-shot models and trivial baselines. The outcome is a neutral-to-negative result. The fine-tuned adapters (re-evaluated: 36.3% cross-entropy, 33.6% KL; originally reported 36.2% and 35.8%) are statistically indistinguishable from alway
    
[^84]: 看见并不等于选中：LLM智能体工具选择失败的注意力-片段解释

    Looking Is Not Picking: An Attention-Segment Account of Tool-Selection Failures in LLM Agents

    [https://arxiv.org/abs/2606.16364](https://arxiv.org/abs/2606.16364)

    这篇论文通过对注意力片段的分析发现，LLM智能体在80%的工具调用失败中其实最关注的正是正确的工具，失败源于决策读出环节而非工具列表干扰——修复提示词只能挽回≤23%的失败，而读出端干预可挽回59-91%。

    

    大语言模型（LLM）智能体会错误地调用工具，一个自然的猜测是：模型在拥挤的工具列表中没能“看到”正确的工具。我们通过一个被同期研究所忽视的视角——模型对带有标签的工具定义片段的注意力——证明了事实恰恰相反。在真实的BFCL失败案例上，通过逐候选的注意力argmax分析，模型在80%的情况下最关注的正是正确的工具（相比之下随机水平为21%），而正确工具成为注意力最低片段的情况仅占10%：模型看着正确的工具，却仍然选错了。这直接反驳了直觉上的“工具列表拥挤/迷失在中间”解释：失败发生在决策读出环节，而非工具列表本身，我们通过三种方式将失败定位于此。(1) 输入 vs. 读出：修复提示词（重新排序或复制正确工具）只能恢复不超过23%的失败案例，而读出端的干预可恢复59-91%。(2) 表征不变性：两种指向正确工具但作用于不同表征的干预——一种加性注意力logit偏置……（摘要原文在此处截断）

    arXiv:2606.16364v3 Announce Type: replace  Abstract: LLM agents mis-call tools, and the natural guess is that the model failed to see the right tool in a crowded harness. We show the opposite through a lens concurrent work sets aside -- the model's attention to labeled tool-definition segments. On real BFCL failures, by per-candidate attention argmax the model attends most to the correct tool 80% of the time (vs. 21% chance), and the gold is the under-attended segment on only 10%: it looks at the right tool and still picks wrong. This directly refutes the intuitive "crowded-harness / lost-in-the-middle" explanation: the failure is at the decision readout, not the harness, and we pin it there three ways. (1) Input vs. readout: repairing the prompt (reordering or duplicating the gold tool) recovers <=23% of failures, while readout-side interventions recover 59-91%. (2) Representation-invariance: two gold-pointed interventions in different representations -- an additive attention-logit bi
    
[^85]: MARGIN：面向不平衡漏洞检测的边际感知正则化几何

    MARGIN: Margin-Aware Regularized Geometry for Imbalanced Vulnerability Detection

    [https://arxiv.org/abs/2605.10240](https://arxiv.org/abs/2605.10240)

    该论文提出MARGIN框架，通过自适应边际度量学习和超球面原型建模来纠正数据不平衡在超球面表示空间中造成的几何畸变，从而提升软件漏洞检测的性能与决策边界稳定性。

    

    软件漏洞检测对于确保软件的安全性和可靠性至关重要。尽管深度学习近来取得了进展，但现实世界的漏洞数据集面临两个严峻的挑战：频率不平衡和难度不平衡。我们从嵌入几何的角度重新解释了这些挑战，观察到此类不平衡会在超球面表示空间中诱发几何畸变。为解决这一问题，我们提出了MARGIN——一个基于度量的框架，通过自适应边际度量学习和超球面原型建模来学习具有判别力的漏洞表示。MARGIN根据由von Mises-Fisher集中度估计的分布结构动态调整几何正则化，将嵌入分布的概率质量与其对应的Voronoi单元对齐，从而减少几何畸变并获得更稳定的决策边界。

    arXiv:2605.10240v4 Announce Type: replace-cross  Abstract: Software vulnerability detection is critical for ensuring software security and reliability. Despite recent advances in deep learning, real-world vulnerability datasets suffer from two severe challenges: frequency imbalance and difficulty imbalance. We reinterpret these challenges from an embedding geometry perspective, observing that such imbalances induce geometric distortions in hyperspherical representation space. To address this issue, we propose MARGIN, a metric-based framework that learns discriminative vulnerability representations through adaptive margin metric learning and hyperspherical prototype modeling. MARGIN dynamically adjusts geometric regularization according to the distribution structure estimated by the von Mises-Fisher concentration, aligning the probability mass of embedding distributions with their corresponding Voronoi cells, thereby reducing geometric distortion and yielding more stable decision bounda
    
[^86]: 智能体框架工程：基于可观测性的编码智能体框架自动演化

    Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses

    [https://arxiv.org/abs/2604.25850](https://arxiv.org/abs/2604.25850)

    该论文提出智能体框架工程（AHE）闭环系统，通过组件、经验和决策三大可观测性支柱，首次实现了编码智能体框架的自动化演化，解决了行动空间异构、轨迹信号淹没和编辑效果难归因三大挑战。

    

    框架目前已成为编码智能体性能的核心，它调节着模型与工具及执行环境之间的交互方式。然而，框架工程仍然是一门依赖人工的手艺，因为将其自动化面临诸多挑战：跨可编辑组件的异构行动空间、淹没可操作信号的海量轨迹，以及效果难以归因的编辑。我们提出了智能体框架工程，这是一个闭环系统，通过三大相互匹配的可观测性支柱来应对这些挑战：(1) 组件可观测性为每个可编辑的框架组件提供文件级表示，使行动空间显式且可回滚；(2) 经验可观测性将数百万原始轨迹标记提炼为分层、可下钻的证据语料库，供演化智能体实际消费使用；(3) 决策可观测性为每次编辑配备一个自主声明的预测，并在下一轮任务级结果中对其进行验证。

    arXiv:2604.25850v5 Announce Type: replace  Abstract: Harnesses are now central to coding-agent performance, mediating how models interact with tools and execution environments. Yet harness engineering remains a manual craft, because automating it faces a heterogeneous action space across editable components, voluminous trajectories that bury actionable signal, and edits whose effect is hard to attribute. We introduce Agentic Harness Engineering (AHE), a closed loop that addresses these challenges through three matched observability pillars: (1) component observability gives every editable harness component a file-level representation so the action space is explicit and revertible; (2) experience observability distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and (3) decision observability pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes. Together
    
[^87]: DynaFix：由执行层动态信息驱动的迭代式自动程序修复

    DynaFix: Iterative Automated Program Repair Driven by Execution-Level Dynamic Information

    [https://arxiv.org/abs/2512.24635](https://arxiv.org/abs/2512.24635)

    DynaFix的核心创新在于在迭代修复过程中反复收集（通过ByteTrace获取变量状态、分支路径和调用栈）并更新执行层动态信息，将其融入提示并以重执行验证补丁，形成闭环，使基于LLM的自动程序修复扎根于具体的运行时行为。

    

    arXiv:2512.24635v3 通告类型：replace-cross 摘要：自动程序修复（APR）旨在为有缺陷的程序生成正确的补丁，但近年来基于大语言模型（LLM）的修复系统对表面代码变化较为敏感，并且在迭代修复过程中往往依赖粗糙的测试结果。我们提出DynaFix，一个基于LLM的APR框架，它将执行层动态信息集成到闭环修复工作流中。DynaFix的显著特点不在于使用执行层信息本身，而在于在整个迭代修复过程中反复更新并利用这些信息。在每一轮修复中，DynaFix使用ByteTrace收集变量状态、分支路径和调用栈，将这些运行时信号融入结构化提示中，通过重新执行来验证生成的补丁，并将由此获得的更新后的追踪信息反馈到后续的修复迭代中。这一过程使补丁生成立足于具体的执行行为，而非……（原文截断）

    arXiv:2512.24635v3 Announce Type: replace-cross  Abstract: Automated Program Repair (APR) seeks to generate correct patches for buggy programs, but recent Large Language Model (LLM)-based repair systems remain sensitive to superficial code changes and often rely on coarse test outcomes during iterative repair. We propose DynaFix, an LLM-based APR framework that integrates execution-level dynamic information into a closed-loop repair workflow. The distinguishing feature of DynaFix lies not in using execution-level information per se, but in repeatedly updating and exploiting such information throughout the iterative repair process. In each repair round, DynaFix uses ByteTrace to collect variable states, branch paths, and call stacks, incorporates these runtime signals into structured prompts, validates generated patches through re-execution, and feeds the resulting updated traces into subsequent repair iterations. This process grounds patch generation in concrete execution behavior rath
    
[^88]: 使用大语言模型与检索增强生成提升人工智能研究中的代码一致性

    Enhancing Code Consistency in AI Research with Large Language Models and Retrieval-Augmented Generation

    [https://arxiv.org/abs/2502.00611](https://arxiv.org/abs/2502.00611)

    本文提出一种结合检索增强生成与大语言模型的系统，用于自动验证研究论文的代码实现是否与论文所述算法方法一致，从而提升代码验证的准确性和AI研究的可信度。

    

    确保代码准确反映研究论文中描述的算法和方法，对于维持人工智能研究的可信度并培养信任至关重要。本文提出了一种新颖的系统，旨在根据相应研究论文中概述的算法和方法来验证代码实现。我们的系统采用检索增强生成技术从研究论文和代码库中提取相关细节，然后使用大语言模型进行结构化比较。这种方法提高了代码实现验证的准确性和全面性，同时有助于提升人工智能研究的透明度、可解释性和可复现性。通过自动化验证过程，我们的系统减少了人工工作量，增强了研究的可信度，并最终推动了代码验证技术的发展。

    arXiv:2502.00611v2 Announce Type: replace-cross  Abstract: Ensuring that code accurately reflects the algorithms and methods described in research papers is critical for maintaining credibility and fostering trust in AI research. This paper presents a novel system designed to verify code implementations against the algorithms and methodologies outlined in corresponding research papers. Our system employs Retrieval-Augmented Generation to extract relevant details from both the research papers and code bases, followed by a structured comparison using Large Language Models. This approach improves the accuracy and comprehensiveness of code implementation verification while contributing to the transparency, explainability, and reproducibility of AI research. By automating the verification process, our system reduces manual effort, enhances research credibility, and ultimately advances the state of the art in code verification.
    

