# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AI with Authority, from Application to Silicon](https://arxiv.org/abs/2608.21356) | 本文展示了生成式AI如何通过机器验证成为不可腐蚀的裁判，使得单人在五周内指挥AI代理从应用代码到RISC-V硅片流片，全程无需人工审查或编写RTL，提出了盐方法这一新纪律。 |
| [^2] | [Natural-Language Workflows Are Not Software Yet: Artifact-Driven Compilation for Reliable Agent Execution](https://arxiv.org/abs/2608.21341) | 本文提出Artic编译器，通过将自然语言工作流转换为显式声明数据依赖和控制流的工件驱动形式，以提升智能体执行的可靠性。 |
| [^3] | [AI-to-AI Code Reviews of GitHub Pull Requests](https://arxiv.org/abs/2608.21311) | 本文构建了首个大规模AI对AI代码审查数据集，揭示了跨产品AI审查在代理生成的PR中占比虽小但增长迅速，形成AI编写与AI审查的闭环生态。 |
| [^4] | [Beyond Fault Localization: A Trajectory-Level Study of LLM Agents for Microservice Root Cause Analysis](https://arxiv.org/abs/2608.21310) | 本文提出一种轨迹级评估框架，揭示了大语言模型代理在微服务根因分析中即使定位正确，也可能因未能重建故障传播路径而导致诊断质量不足。 |
| [^5] | [Human-AI Collaboration in Requirements Engineering: Evidence of the Negative Effect of LLMs on Requirements Inspection](https://arxiv.org/abs/2608.21298) | 本研究通过受控实验发现，大语言模型支持在需求审查中可能对气味识别和严重性分类的有效性产生负面影响，尽管可能缩短审查时间。 |
| [^6] | [The Substitution Escrow Threshold: When "Compatible With" Becomes Safe Enough to Buy](https://arxiv.org/abs/2608.21221) | 本文提出了一个五条件框架（替代托管阈值），用于区分兼容性声明是真正降低机构风险还是仅减少集成成本，并应用于五个基础设施案例以提供实用诊断。 |
| [^7] | [Specification Portability Across LLM Development Agents: Cross-Agent Compatibility in Specification-Driven Software Migration](https://arxiv.org/abs/2608.21208) | 该研究通过Oracle到PostgreSQL迁移实验发现，规格大小不能预测实现质量，而跨代理规格迁移可显著提升实现效果。 |
| [^8] | [On the Time and Frequency Domain Representations of Signals for CPS Specification](https://arxiv.org/abs/2608.21167) | 本文提出了一种新的规约语言S2TL，利用时频表示来增强CPS需求规约的表达能力，特别适用于描述信号形状和动态行为。 |
| [^9] | [Large Language Models at the Intersection of Software Engineering and Software Security:An Evidence-Centered Structured Survey and Research Agenda](https://arxiv.org/abs/2608.21107) | 本综述提出了一个以证据为中心的结构化框架，整合软件工程与软件安全中的LLM研究，并引入保障框架来区分功能正确性、安全性、可靠性、溯源和代理权限，强调执行反馈和仓库访问对提升性能的重要性。 |
| [^10] | [Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems](https://arxiv.org/abs/2608.21095) | 本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。 |
| [^11] | [PromptResponse: Optimizing Prompts for LLM Coding Tasks](https://arxiv.org/abs/2608.21074) | 本文通过对照实验发现，使用一致的格式（如JSON）优化提示词可提升编码任务的生成效率和稳定性，而基于LLM的提示词调整反而显著降低了任务性能。 |
| [^12] | [Spike-Killer: Evidence-Gated LLM Assistance for Safe Performance Diagnosis on a Real Windows Workstation](https://arxiv.org/abs/2608.21069) | 本文提出了一种人工批准、证据门控的LLM辅助诊断工作流程，通过事务化操作和快照保留，在真实Windows工作站上安全高效地解决了帧时间问题。 |
| [^13] | [Scalable Distributed Simulation-Based Testing for Automated Driving Systems](https://arxiv.org/abs/2608.20904) | 本文提出了一种基于Kubernetes和Argo Workflows的DevOps驱动框架，实现了CARLA仿真测试的自动化构建、部署和分布式执行，显著提升了自动驾驶系统大规模场景测试的可扩展性和效率。 |
| [^14] | [Generation of Web Apps with Agentic IDEs: An Empirical Assessment](https://arxiv.org/abs/2608.20903) | 本研究实证评估了三种智能体IDE在生成全栈Web应用时的性能，发现它们在处理常见模式时表现成熟，但在生成复杂分布式架构时错误频出，表明它们不能替代开发者，而是将开发者角色转变为通过自然语言编排智能体。 |
| [^15] | [Beyond the Traceback: Using LLMs for Adaptive Explanations of Programming Errors](https://arxiv.org/abs/2608.20896) | 本研究发现，大型语言模型生成的错误解释虽然显著提升了用户的主观感受，但并未改善实际的调试客观表现，且解释风格需根据程序员技能水平进行自适应调整。 |
| [^16] | [BC-Bench: Evaluating Agentic Engineering in a Domain-Specific Language for ERP](https://arxiv.org/abs/2608.20851) | BC-Bench是一个针对ERP领域特定语言AL的基准测试，通过101个真实任务评估智能体工程能力，涵盖代码生成、测试生成和多模态问题处理，弥补了通用基准在ERP场景中的不足。 |
| [^17] | [An Extensive Empirical Study on Code Translation Technique](https://arxiv.org/abs/2608.20776) | 本研究通过大规模实证比较发现，LLM和基于LLM的代码翻译方法在方法级正确性上优于传统学习方法，但相似性指标不能可靠预测功能正确性，且翻译方向对性能有显著影响。 |
| [^18] | [Temporal Validity on Real Software Histories: Eliminating Stale-Fact Errors in Code-Assistant Memory over GitHub Fixes](https://arxiv.org/abs/2608.20685) | 本文验证了MemStrata在真实软件历史中通过确定性过时记忆消除RAG的时间盲点，显著提升答案准确率（0.91对比0.57-0.59），并减少过时事实错误。 |
| [^19] | [The Software Supply Chain as a Market for Lemons: A Multivocal Review of Trust Signal Collapse](https://arxiv.org/abs/2608.20678) | 本研究通过多声部综述揭示，开源依赖采纳中的廉价信任信号在对抗性操纵、不可区分的游戏化和AI驱动膨胀的三重压力下系统性崩塌，生态系统应对仍以建议为主。 |
| [^20] | [DreamBench-SWE: A Multi-Session Memory-Hygiene Benchmark for Software Agents](https://arxiv.org/abs/2608.20664) | 该论文提出了DreamBench-SWE基准，用于评估软件代理在多会话中的记忆卫生，并通过实验表明外部记忆机制的性能差异显著，但未达到等效性证据。 |
| [^21] | [Toward Understanding Operating System Defects](https://arxiv.org/abs/2608.20643) | 这项研究首次对1500个来自Android、Linux和HarmonyOS操作系统的缺陷进行大规模分析，揭示了缺陷分布的关键特征，为操作系统缺陷检测和调试方法的设计提供了基础性指导。 |
| [^22] | [Behavior Specification-Guided Program Synthesis for Binary Deobfuscation](https://arxiv.org/abs/2608.20628) | 本文提出一种从结构变换转向行为驱动合成的二进制去混淆新范式，通过行为规范指导程序合成，以克服现有反编译方法在保持运行时行为和代码质量方面的局限性。 |
| [^23] | [Applying Anthropic Primitives at Large Enterprises: Harness Paradigm for Knowledge Work](https://arxiv.org/abs/2608.20622) | 本文提出在企业知识工作中采用“驾驭范式”（harness paradigm）作为第三种方案，以同时避免定制代码维护成本高和集中治理范围有限的问题，并基于近期研究证明其在任务层面优于复杂架构。 |
| [^24] | [Testing and Evaluation of Agentic AI Systems In Military Command and Control](https://arxiv.org/abs/2608.20597) | 本文通过审查240项测试与评估实践，发现智能体AI系统的特性削弱了传统测试方法中的八项关键假设，导致现有测试结果无法有效支持军事指挥控制系统的安全性承诺。 |
| [^25] | [FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth](https://arxiv.org/abs/2608.20574) | 该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。 |
| [^26] | [AutoMOOSE: Use Case and Logical Views of Agentic Phase-Field Simulation Software](https://arxiv.org/abs/2608.20571) | AutoMOOSE通过六代理流水线和物理证伪/自动修复分离机制，将自然语言请求转化为可检查且可重用的MOOSE相场模拟，适用于多物理场材料设计。 |
| [^27] | [Making Deployments Safe at Meta: Health Checks for Continuous Change-Safety](https://arxiv.org/abs/2608.20513) | 本文介绍了Meta的Service Health Checker系统，通过模板化健康检查和自动回滚机制，平衡大规模持续部署的发布速度与可靠性，并解决了规模化中的操作挑战。 |
| [^28] | [Terminal Agents: A Survey of AI Agents in Command-Line Environments](https://arxiv.org/abs/2608.20485) | 本文首次以终端介导执行为核心视角，系统梳理了命令行环境中AI代理的架构、能力与评估，提出七维终端能力框架，并指出当前评估过度关注最终结果而忽视过程质量与恢复机制。 |
| [^29] | [Vibe Coding: Practice, Performance, Productivity, and Risk -A State-of-the-Art Review](https://arxiv.org/abs/2608.20446) | 本文首次系统综述了“氛围编程”的实证证据，揭示了其任务级能力不均（代码生成可靠但故障检测弱）和生产力证据矛盾（实验显示+26%但随机试验显示-19%）的关键现状。 |
| [^30] | [SDAD: Spec-Driven Agentic Development for the AI-Native SDLC](https://arxiv.org/abs/2608.20341) | 本文提出SDAD框架，将规格驱动开发与智能体技术结合，实现从意图到代码的自主交付，并对比了传统敏捷与未来智能体驱动的SDLC范式转变。 |
| [^31] | [Stable Within, Unidentified Across: Semantic Identification of Benchmark Effects and Rankings](https://arxiv.org/abs/2608.19269) | 该论文通过局部非蕴含实例表明，基准效应和排名的识别依赖于评估者控制的语义家族，不同语义选择可能导致截然不同的结论。 |
| [^32] | [LongRCA Bench: Diagnosing Responsible Roles and Root Causes in Long-Horizon Agent Failures](https://arxiv.org/abs/2608.15242) | 本文提出LongRCA Bench，一个包含1,140条长时程失败轨迹的基准，用于诊断智能体失败中的责任角色和最早根本原因步骤，并引入无需训练的RCTA方法，以提升诊断精度。 |
| [^33] | [Skillware: A Software Ontology and Engineering Lifecycle for Persistent Behavioral Artifacts](https://arxiv.org/abs/2607.18970) | 本文提出了技能软件（Skillware）这一软件抽象，将软件工程原则应用于AI代理中的持久行为工件，通过定义技能工件、技能软件单元和代理主机执行关系来规范其生命周期。 |
| [^34] | [RSE of a Quantum Transport Code and its Effects](https://arxiv.org/abs/2605.21334) | 本文通过两年RSE实践，展示了持续集成和基准测试如何揭示量子输运代码中的关键缺陷，包括未初始化内存、越界写入及数学建模错误，并指出Fortran科学代码中危险缺陷的普遍性。 |
| [^35] | [MalSkills: Detecting Malicious Skills in the Agentic Supply Chain via Neuro-symbolic Reasoning](https://arxiv.org/abs/2603.27204) | 本文提出了MalSkills，一个神经符号框架，通过结合符号解析和LLM辅助语义分析，从异构工件中提取安全敏感操作并构建技能依赖图，从而有效检测代理供应链中的恶意技能。 |
| [^36] | [Does My README File Need To Be Updated? Exploring LLM-Based README Maintenance](https://arxiv.org/abs/2603.00489) | 该论文提出了一种基于大语言模型的框架，用于自动推荐开源项目中README文件的精准更新，在人机协同流程中判断更新需求、定位修改位置并解释触发原因，在25,511个拉取请求上实现了一半恢复率和28%的用户端准确率。 |
| [^37] | [Reward Engineering for Software Tasks: A Survey of Reinforcement Learning Approaches](https://arxiv.org/abs/2601.19100) | 本文首次系统综述了软件任务中强化学习的奖励工程，按奖励来源、粒度和聚合方式分类，并为实践提供了指导。 |
| [^38] | [Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks](https://arxiv.org/abs/2512.03262) | 该论文提出SUSVIBES基准，评估了12种编码智能体在真实任务中的安全性，发现所有智能体生成代码的安全率极低（最高仅11.8%），且简单安全提示无法有效改善。 |
| [^39] | [Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries](https://arxiv.org/abs/2509.22202) | 本研究首次系统分析了开发者查询变化如何触发大型语言模型生成代码中的库幻觉，揭示了不同提示条件下的系统性风险模式。 |
| [^40] | [Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation](https://arxiv.org/abs/2505.16222) | 本研究首次系统性地揭示了大语言模型在代码评估中对表面差异（如变量名、注释和格式）存在偏见，并通过多种语言和模型实证证明了这些偏见会影响评估的公平性。 |

# 详细

[^1]: 拥有权威的AI：从应用到硅片

    AI with Authority, from Application to Silicon

    [https://arxiv.org/abs/2608.21356](https://arxiv.org/abs/2608.21356)

    本文展示了生成式AI如何通过机器验证成为不可腐蚀的裁判，使得单人在五周内指挥AI代理从应用代码到RISC-V硅片流片，全程无需人工审查或编写RTL，提出了盐方法这一新纪律。

    

    arXiv:2608.21356v1 公告类型：交叉 摘要：六十年来，机器验证一直是主要的成本开销，仅适用于特殊工件。在此我们报告，生成式AI逆转了这一关系：在AI速度下，机器验证不仅经济实惠，而且对生产力至关重要——它是不可腐蚀的裁判，让一个人能安全地大规模指挥自主机器工作。在五周内，一位使用消费级AI订阅的研究者，指挥一小队AI代理从应用代码出发，通过经过验证的编译器和执行器，最终在社区硅穿梭片上流片出一款RISC-V处理器；没有一项证明经过人工审查，也没有任何RTL代码由人类编写。这一工作纪律——盐方法——依赖于一个无法通过任何幻觉证明的证明核心：数学声明以核心检查的工件形式在代理之间传递，而人类的注意力则保留在陈述、设计和裁决上。验证被逐环陈述，从

    arXiv:2608.21356v1 Announce Type: cross  Abstract: For sixty years, machine verification has been a major cost overhead, affordable only for exceptional artifacts. Here we report that generative AI inverts this relationship: at AI speed, machine verification is not only economical but essential to productivity --- it is the incorruptible referee that lets one person safely direct autonomous machine work at scale. In five weeks, one researcher on consumer AI subscriptions directed a small fleet of AI agents from application code, through a verified compiler and executive, to a RISC-V processor taped out on a community silicon shuttle; no proof passed through human review, and no RTL was written by a human. The working discipline --- the Salt method --- rests on a proof kernel no hallucinated proof can pass: mathematical claims travel between agents as kernel-checked artifacts, and human attention is reserved for statements, designs, and rulings. Verification is stated link by link, from
    
[^2]: 自然语言工作流尚非软件：面向可靠智能体执行的工件驱动编译

    Natural-Language Workflows Are Not Software Yet: Artifact-Driven Compilation for Reliable Agent Execution

    [https://arxiv.org/abs/2608.21341](https://arxiv.org/abs/2608.21341)

    本文提出Artic编译器，通过将自然语言工作流转换为显式声明数据依赖和控制流的工件驱动形式，以提升智能体执行的可靠性。

    

    arXiv:2608.21341v1 公告类型：新 摘要：自然语言工作流为智能体提供了一种类似软件的接口：领域专家可以编写可复用的流程，智能体可以将其作为指令执行。然而，这一承诺尚不可靠。工作流描述往往隐含数据依赖，导致执行者必须推断某一步骤应使用哪些先前结果；智能体在上下文压力下也可能无法遵循长流程或分支指令。我们提出Artic，一种工件驱动的工作流编译器，将自然语言工作流转换为工件驱动的工作流，其中每个步骤声明其读取和写入的工件，约束条件门控生成的工件，显式控制转移路由执行。这种表示暴露了智能体执行所承担的强制负担，使编译器能够识别依赖过多状态或包含复杂控制逻辑的步骤，并通过约束优化对其进行细化。为验证LLM辅助...

    arXiv:2608.21341v1 Announce Type: new  Abstract: Natural-language workflows offer a software-like interface for agents: domain experts can write reusable procedures, and agents can execute them as instructions. This promise is not yet reliable. Workflow descriptions often leave data dependencies implicit, so the executor must infer which prior results a step should use; agents can also fail to follow long or branching instructions under context pressure. We propose Artic, an artifact-driven workflow compiler that transforms a natural-language workflow into an artifact-driven workflow in which each step declares the artifacts it reads and writes, constraints gate produced artifacts, and explicit control transfers route execution. This representation exposes the enforcement burden placed on agent execution, allowing the compiler to identify steps that depend on too much state or contain difficult control logic and refine them through constrained optimization. To validate the LLM-assisted
    
[^3]: AI对AI的GitHub拉取请求代码审查

    AI-to-AI Code Reviews of GitHub Pull Requests

    [https://arxiv.org/abs/2608.21311](https://arxiv.org/abs/2608.21311)

    本文构建了首个大规模AI对AI代码审查数据集，揭示了跨产品AI审查在代理生成的PR中占比虽小但增长迅速，形成AI编写与AI审查的闭环生态。

    

    arXiv:2608.21311v1 公告类型：新 摘要：AI编码代理正日益融入软件开发工作流程，在拉取请求（PR）过程的两端运作：AI编写代理创建或修改PR，而AI审查代理对其进行评估。这形成了一个闭环，其中一个AI编码代理审查归因于另一个代理的贡献。我们通过将AI归因的PR与来自CodAGE（一个编码代理生成的GitHub事件的公共数据集）中的AI归因审查事件关联起来，构建了一个大规模的AI对AI代码审查数据集。我们的数据集包含248,641个独特的AI归因PR，这些PR至少收到一次AI归因审查。其中，45,269个接受了跨产品审查，208,145个接受了同产品审查；4,773个PR同时接受了两种审查。跨产品AI对AI审查在已识别的代理作者PR中约占1.6%，但绝对数量可观，且其数量从2025年第一季度到第三季度增加了两个数量级以上。

    arXiv:2608.21311v1 Announce Type: new  Abstract: AI coding agents are increasingly integrated into software development workflows, operating on both sides of the pull-request (PR) process: AI authoring agents create or modify PRs, while AI reviewers evaluate them. This creates a closed loop in which one AI coding agent reviews a contribution attributed to another. We construct a large-scale dataset of AI-to-AI code review by linking AI-attributed PRs with AI-attributed review events from CodAGE, a public dataset of coding-agent-generated GitHub events. Our dataset contains 248,641 unique AI-attributed PRs that received at least one AI-attributed review. Of these, 45,269 received cross-product review and 208,145 received same-product review; 4,773 PRs received both. Cross-product AI-to-AI review occurred in approximately 1.6% of identified agent-authored PRs but was substantial in absolute terms, and its volume increased by more than two orders of magnitude from 2025-Q1 to 2025-Q3. Revi
    
[^4]: 超越故障定位：面向微服务根因分析的大语言模型代理轨迹级研究

    Beyond Fault Localization: A Trajectory-Level Study of LLM Agents for Microservice Root Cause Analysis

    [https://arxiv.org/abs/2608.21310](https://arxiv.org/abs/2608.21310)

    本文提出一种轨迹级评估框架，揭示了大语言模型代理在微服务根因分析中即使定位正确，也可能因未能重建故障传播路径而导致诊断质量不足。

    

    现有对微服务自动根因分析（RCA）的评估主要依据端点正确性来评判诊断性能：即方法是否定位了负责的服务。这一标准便于比较，但无法揭示诊断的证据基础，或连接故障源到观察症状的故障传播路径，而这些正是值班站点可靠性工程师判断是否需要采取行动所必需的。因此，我们将RCA视为一个可观测的诊断过程。我们的轨迹级框架根据人工整理的服务级故障传播路径来评估代理执行。应用于一个公开的微服务RCA基准，它分析了3500条诊断轨迹，刻画了代理在何处进行调查以及如何使用检索到的遥测数据。我们发现答案正确性与诊断质量之间存在脱节：一个代理可能定位了故障源，却未能重建其传播路径。

    arXiv:2608.21310v1 Announce Type: new  Abstract: Existing evaluations of automated root cause analysis (RCA) for microservices assess diagnostic performance mainly by endpoint correctness: whether a method localizes the responsible service. This criterion enables comparison but does not reveal the evidentiary basis of a diagnosis or the fault-propagation route connecting the source to observed symptoms, both of which an on-call site reliability engineer needs to judge whether action is warranted. We therefore treat RCA as an observable diagnostic process. Our trajectory-level framework evaluates agent executions against manually curated service-level fault-propagation paths. Applied to a public microservice RCA benchmark, it analyzes 3,500 diagnostic trajectories, characterizing where agents investigate and how they use retrieved telemetry. We find a disconnect between answer correctness and diagnostic quality: an agent may localize the fault source yet fail to reconstruct its propagat
    
[^5]: 人机协作在需求工程中的研究：大语言模型对需求审查负面影响的证据

    Human-AI Collaboration in Requirements Engineering: Evidence of the Negative Effect of LLMs on Requirements Inspection

    [https://arxiv.org/abs/2608.21298](https://arxiv.org/abs/2608.21298)

    本研究通过受控实验发现，大语言模型支持在需求审查中可能对气味识别和严重性分类的有效性产生负面影响，尽管可能缩短审查时间。

    

    背景：需求审查（RI）是在软件生命周期早期检测需求工件潜在缺陷的成熟实践。近年来，大语言模型（LLMs）的进步激发了人们对其支持需求工程（RE）任务的兴趣。然而，关于LLMs作为协作助手在人工执行的需求审查中影响的实证证据仍然匮乏。目的：我们旨在调查LLM支持对人工需求审查的影响，考虑在气味识别和严重性分类（即有害与无害）方面的审查有效性，以及审查时长。方法：我们进行了一项受控交叉设计实验，涉及34名参与者，他们在有和无LLM支持的情况下审查文本规格说明，识别和分类需求气味，同时记录审查时间。我们使用一种贝叶斯回归模型分析数据。

    arXiv:2608.21298v1 Announce Type: new  Abstract: Background. Requirements inspection (RI) is a well-established practice for detecting potential defects in requirements artifacts early in the software lifecycle. Recent advances in large language models (LLMs) have stimulated interest in their potential to support requirements engineering (RE) tasks. However, empirical evidence on the effects of LLMs when used as collaborative assistants in human-performed RI remains scarce. Aims. We aim to investigate the impact of LLM support on human-performed RI, considering inspection effectiveness in terms of smell identification and severity classification (i.e., nocuous vs innocuous), as well as inspection duration. Method. We conducted a controlled crossover design experiment with 34 participants, who inspected textual specifications with and without LLM support, identifying and classifying requirements smells while recording inspection time. We analyzed the data using one Bayesian regression m
    
[^6]: 替代托管阈值：当“兼容”变得足够安全以购买时

    The Substitution Escrow Threshold: When "Compatible With" Becomes Safe Enough to Buy

    [https://arxiv.org/abs/2608.21221](https://arxiv.org/abs/2608.21221)

    本文提出了一个五条件框架（替代托管阈值），用于区分兼容性声明是真正降低机构风险还是仅减少集成成本，并应用于五个基础设施案例以提供实用诊断。

    

    arXiv:2608.21221v1 公告类型：新 摘要：企业基础设施买家通常评估兼容性声明——“S3兼容”、“PostgreSQL兼容”、“OpenAI兼容”——作为未来替代选项的代理指标。然而，大多数兼容性声明并未托管其暗示的替代路径。本文引入了替代托管阈值，这是一个五条件框架，用于确定兼容性声明何时真正降低机构风险，而非仅仅减少首次集成成本。五个条件——边界闭合、可执行一致性、保管独立性、状态和操作可逆性以及扩展隔离——应用于五个基础设施案例（OCI、Kubernetes、OpenTelemetry、S3、PostgreSQL），这些案例填充了五个不同的结果单元。该框架为企业架构师、平台工程师、采购团队以及评估依赖兼容性的基础设施决策的投资者提供了可操作的诊断工具，并识别出……

    arXiv:2608.21221v1 Announce Type: new  Abstract: Enterprise infrastructure buyers routinely evaluate compatibility claims--"S3-compatible," "PostgreSQL-compatible," "OpenAI compatible"--as proxies for future substitution options. Yet most compatibility claims do not escrow the substitution path they imply. This paper introduces the Substitution Escrow Threshold, a five-condition framework that determines when a compatibility claim genuinely reduces institutional risk versus merely reducing first-integration cost. The five conditions--boundary closure, executable conformance, custody independence, state and operations reversibility, and extension quarantine--are applied to five infrastructure cases (OCI, Kubernetes, OpenTelemetry, S3, PostgreSQL) that populate five distinct outcome cells. The framework produces actionable diagnostics for enterprise architects, platform engineers, procurement teams, and investors evaluating compatibility-dependent infrastructure decisions, and identifies
    
[^7]: 跨LLM开发代理的规格可移植性：规格驱动软件迁移中的跨代理兼容性

    Specification Portability Across LLM Development Agents: Cross-Agent Compatibility in Specification-Driven Software Migration

    [https://arxiv.org/abs/2608.21208](https://arxiv.org/abs/2608.21208)

    该研究通过Oracle到PostgreSQL迁移实验发现，规格大小不能预测实现质量，而跨代理规格迁移可显著提升实现效果。

    

    本文以Oracle到PostgreSQL迁移作为受控软件转换任务，研究了跨代理规格可移植性。研究包含两个实验阶段。首先，在1,006个PL/SQL文件上评估了规格优先迁移流程，其中623个成功重新生成，380个生成的脚本在PostgreSQL 16中成功执行。其次，使用Amazon Kiro、Google Gemini和GitHub Copilot在包含1,802个Oracle脚本及对应PostgreSQL实现的数据集上进行了跨代理实验，初始单代理评估中还包括Claude Code和Cursor。使用Token F1、精确匹配、SQL语法有效性、AST精确匹配、AST平均相似度和即时可运行性来评估原生和外部规格。结果表明，规格大小本身不能预测实现质量，且跨代理迁移可能产生显著改进。

    arXiv:2608.21208v1 Announce Type: cross  Abstract: This paper investigates cross-agent specification portability using Oracle-to-PostgreSQL migration as a controlled software transformation task. The study combines two experimental stages. First, a specification-first migration pipeline was evaluated on 1,006 PL/SQL files, of which 623 were successfully regenerated and 380 generated scripts executed successfully in PostgreSQL 16. Second, cross-agent experiments were conducted on a dataset of 1,802 Oracle scripts with corresponding PostgreSQL implementations using Amazon Kiro, Google Gemini, and GitHub Copilot, with Claude Code and Cursor included in the initial single-agent evaluation. Native and foreign specifications were assessed using Token F1, exact match, SQL syntax validity, AST exact match, AST mean similarity, and immediate runnability. The results show that specification size alone does not predict implementation quality and that cross-agent transfer can produce substantial a
    
[^8]: 面向CPS规约的信号时域与频域表示研究

    On the Time and Frequency Domain Representations of Signals for CPS Specification

    [https://arxiv.org/abs/2608.21167](https://arxiv.org/abs/2608.21167)

    本文提出了一种新的规约语言S2TL，利用时频表示来增强CPS需求规约的表达能力，特别适用于描述信号形状和动态行为。

    

    规约语言对于网络物理系统（CPS）的验证与确认至关重要。大多数最先进的规约语言使用信号的时域表示，这并不总是适合描述信号形状和动态行为。相反，控制和机器人等领域使用频域表示来表征这些行为。时频表示结合了两个域的能力。我们研究了使用时频表示来规定CPS需求的可能性。我们分析了现有的CPS需求分类法，以识别哪些需求类别可以从时频表示中受益。我们得出了使用时频表示的规约语言的设计要求，并提出了信号-频谱时间逻辑（S2TL），这是一种能够对频率区间和频率分量之间的关系进行断言的规约语言。我们进行了操作化实现。

    arXiv:2608.21167v1 Announce Type: new  Abstract: Specification languages are instrumental to the Verification \& Validation of Cyber-Physical Systems (CPSs). Most state-of-the-art specification languages use the time-domain representation of signals, which is not always suitable for describing signal shapes and dynamic behaviours. Instead, fields like control and robotics use the frequency-domain representation to characterise these behaviours. Time-frequency representations combine the capabilities of both domains. We investigate the use of time-frequency representations to specify CPS requirements. We analyse existing taxonomies of CPS requirements to identify which requirement classes can benefit from time-frequency representations. We derive the desiderata for a specification language that uses time-frequency representations and propose Signal-Spectrum Temporal Logic (S2TL), a language enabling assertions over frequency intervals and relations between frequency components. We opera
    
[^9]: 大型语言模型在软件工程与软件安全交叉领域：以证据为中心的结构化综述与研究议程

    Large Language Models at the Intersection of Software Engineering and Software Security:An Evidence-Centered Structured Survey and Research Agenda

    [https://arxiv.org/abs/2608.21107](https://arxiv.org/abs/2608.21107)

    本综述提出了一个以证据为中心的结构化框架，整合软件工程与软件安全中的LLM研究，并引入保障框架来区分功能正确性、安全性、可靠性、溯源和代理权限，强调执行反馈和仓库访问对提升性能的重要性。

    

    arXiv:2608.21107v1 公告类型：新 摘要：大型语言模型（LLM）正从代码补全迈向仓库级代理，这些代理能检索上下文、编辑文件、执行工具，并参与安全敏感的工作流程。然而，关于这些系统的证据仍然分散在两类评估中：一类是以功能任务完成为中心的软件工程评估，另一类是以漏洞检测、安全生成或利用导向验证为中心的软件安全评估。本项以证据为中心的结构化综述，综合了截至2026年5月31日可获得的代表性工作，涵盖软件工程任务、软件安全任务、适应机制、工件粒度以及评估设计。除了任务分类法外，我们还引入了一个保障框架，该框架区分功能正确性、安全性、操作可靠性、证据溯源和代理权限。综述表明，执行反馈和仓库访问能显著提升性能。

    arXiv:2608.21107v1 Announce Type: new  Abstract: Large Language Models (LLMs) are moving from code completion toward repository-scale agents that retrieve context, edit files, execute tools, and participate in security-sensitive workflows. The evidence for these systems, however, remains divided between software engineering evaluations centered on functional task completion and software security evaluations centered on vulnerability detection, secure generation, or exploit-oriented validation. This evidence-centered structured survey synthesizes representative work available through May 31, 2026 across software engineering tasks, software security tasks, adaptation mechanisms, artifact granularity, and evaluation design. In addition to a task taxonomy, we introduce an assurance framework that separates functional correctness, security, operational reliability, evidence provenance, and agent authority. The review shows that execution feedback and repository access can substantially impr
    
[^10]: 可信RAG：用于检测生成式AI系统中错误信息与知识投毒的评估代理

    Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems

    [https://arxiv.org/abs/2608.21095](https://arxiv.org/abs/2608.21095)

    本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。

    

    检索增强生成（RAG）将大型语言模型（LLM）的输出锚定在外部知识上，但RAG系统通常信任其检索到的任何内容，从而造成“安全-可靠性差距”：高语义相关性并不保证事实真实性。攻击者利用这一漏洞进行知识投毒，插入恶意文档以引发定向错误信息。我们提出了一种评估代理，这是一种中间件，结合了自然语言推理（NLI）事实核查、具有相关性加权聚合的五信号投毒检测器，以及一个信任指数 T = 0.4 F + 0.35 C + 0.25 (1 - P )，并针对高污染情境采用非线性阻尼器。在TruthfulQA上使用Llama 3.3 70B时，该代理达到了91%的准确率和100%的精确率，对指令注入的召回率为100%，而就地编辑（如实体替换）仍难以检测。在三个LLM上，信任指数保持判别性，接收者操作特征（ROC）曲线表现良好。

    arXiv:2608.21095v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) outputs in external knowledge, but RAG systems usually trust whatever they retrieve, creating a Security-Reliability Gap: high semantic relevance does not guarantee factual truth. Adversaries exploit this through knowledge poisoning, inserting malicious documents to cause targeted misinformation. We propose an Evaluation Agent, middleware that combines Natural Language Inference (NLI) factual verification, a five-signal poison detector with relevance-weighted aggregation, and a Trust Index T = 0.4 F + 0.35 C + 0.25 (1 - P ) with a non-linear dampener for high-contamination contexts. On TruthfulQA with Llama 3.3 70B, the agent reaches 91% accuracy and 100% precision, with 100% recall on instruction injection, while in-place edits, such as entity swaps, remain hard to detect. Across three LLMs the Trust Index stays discriminative, with a Receiver Operating Characteri
    
[^11]: 提示响应：优化大型语言模型编码任务的提示词

    PromptResponse: Optimizing Prompts for LLM Coding Tasks

    [https://arxiv.org/abs/2608.21074](https://arxiv.org/abs/2608.21074)

    本文通过对照实验发现，使用一致的格式（如JSON）优化提示词可提升编码任务的生成效率和稳定性，而基于LLM的提示词调整反而显著降低了任务性能。

    

    大型语言模型（LLMs）在研究工作流程和软件开发管道中的应用日益增多，但其输出对输入提示词的变化仍然敏感。本文介绍了“提示响应”（PromptResponse），一项对照研究，探讨了编码任务提示词的格式和基于LLM的调整如何影响生成代码的性能、效率和稳定性。我们使用了HumanEval数据集的五个语义相同但语法不同的变体——基线、JSON、Markdown、YAML以及一个LLM调整版本——让GPT-4o在8200次执行中解决其编码问题。结果表明，一致的格式（尤其是JSON）提高了生成效率和语法稳定性，并在任务性能上有小幅提升。相反，LLM调整的提示词导致任务性能显著下降，且没有明显的稳定性收益。

    arXiv:2608.21074v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in research workflows and software development pipelines, yet their output remains sensitive to input prompt variations. This paper presents $\unicode{x00AB}$PromptResponse$\unicode{x00BB}$, a controlled study examining how formatting and LLM-based tuning of coding task prompts affect the resulting code's performance, efficiency, and stability. Using five semantically identical yet syntactically distinct variants of the HumanEval dataset$\unicode{x2014}$baseline, JSON, Markdown, YAML, and an LLM-tuned version$\unicode{x2014}$we had GPT-4o solve its coding problems over 8200$\unicode{x00A0}$executions. Our results show that consistent formatting$\unicode{x2014}$especially JSON$\unicode{x2014}$improves generation efficiency and syntactic stability, with minor gains in task performance. Conversely, the LLM-tuned prompts resulted in significantly degraded task performance without significa
    
[^12]: 尖峰杀手：证据门控的LLM辅助安全性能诊断在真实Windows工作站上的应用

    Spike-Killer: Evidence-Gated LLM Assistance for Safe Performance Diagnosis on a Real Windows Workstation

    [https://arxiv.org/abs/2608.21069](https://arxiv.org/abs/2608.21069)

    本文提出了一种人工批准、证据门控的LLM辅助诊断工作流程，通过事务化操作和快照保留，在真实Windows工作站上安全高效地解决了帧时间问题。

    

    arXiv:2608.21069v1 公告类型：新 摘要：LLM辅助代理可以综合系统证据、提出配置更改并自动化诊断任务，但其灵活性使得不精确的操作或侵入性收集器成为运营风险。我们提出了尖峰杀手，一个用于在真实Windows工作站上诊断帧时间问题的人工批准工作流程。该工作流程将每个操作视为一个证据门控事务：记录精确的目标状态、分类风险、保留快照、验证后置条件，并将失败的测量作为一等证据保留。这篇经验论文报告了一项同日完成的关于《反恐精英2》作为高要求目标应用的研究。证据包包含保留的状态快照、探索性微基准、十次同状态重复性探针、实时遥测、修复的过宽注册表操作、不兼容的演示捕获尝试、无效的本地回放以及系统级追踪回放。

    arXiv:2608.21069v1 Announce Type: new  Abstract: LLM-assisted agents can synthesize system evidence, propose configuration changes, and automate diagnostic tasks, but their flexibility makes an imprecise action or an intrusive collector an operational risk. We present Spike-Killer, a human-approved workflow for diagnosing frame-time complaints on one real Windows workstation. The workflow treats each action as an evidence-gated transaction: it records the exact target state, classifies risk, preserves a snapshot, verifies a postcondition, and retains failed measurements as first-class evidence.   This experience paper reports a completed same-day study with Counter-Strike 2 as a demanding target application. The evidence bundle contains preserved state snapshots, exploratory microbenchmarks, a ten-run same-state repeatability probe, live telemetry, a repaired over-broad registry action, incompatible presentation-capture attempts, an invalid local replay, and a system-level tracing repl
    
[^13]: 面向自动驾驶系统的可扩展分布式仿真测试

    Scalable Distributed Simulation-Based Testing for Automated Driving Systems

    [https://arxiv.org/abs/2608.20904](https://arxiv.org/abs/2608.20904)

    本文提出了一种基于Kubernetes和Argo Workflows的DevOps驱动框架，实现了CARLA仿真测试的自动化构建、部署和分布式执行，显著提升了自动驾驶系统大规模场景测试的可扩展性和效率。

    

    基于虚拟场景的测试是验证自动驾驶系统（ADS）和智能交通系统（ITS）的关键手段。然而，执行可能涉及数千个场景的大规模测试套件仍然劳动密集且难以扩展。本文提出了一个端到端、DevOps驱动的框架，该框架在轻量级Kubernetes集群上自动化了基于CARLA的ADS场景测试的构建、部署和分布式执行。ROS 2应用被打包为从仓库规范生成的标准Kubernetes Helm图表，而整个仿真环境则通过动态Helmfile清单以声明方式组合。本文描述了如何在Argo Workflows中实现分布式测试工作流，以配置环境、从可配置源聚合和批处理OpenSCENARIO测试用例、在集群节点间并行执行场景，并收集日志和资源指标。

    arXiv:2608.20904v1 Announce Type: cross  Abstract: Virtual scenario-based testing is a key enabler for validating automated driving systems (ADS) and intelligent transport systems (ITS). However, executing large-scale test suites involving possibly thousands of scenarios remains labor-intensive and difficult to scale. This paper presents an end-to-end, DevOps-driven framework that automates build, deployment, and distributed execution of CARLA-based scenario tests of an ADS on a lightweight Kubernetes cluster. ROS 2 applications are packaged as standardized Kubernetes Helm charts generated from repository specifications, while entire simulation environments are composed declaratively via dynamic Helmfile manifests. The paper describes how a distributed testing workflow can be implemented in Argo Workflows to provision environments, aggregate and batch OpenSCENARIO test cases from configurable sources, execute scenarios in parallel across cluster nodes, and collect logs and resource met
    
[^14]: 智能体集成开发环境生成Web应用：一项实证评估

    Generation of Web Apps with Agentic IDEs: An Empirical Assessment

    [https://arxiv.org/abs/2608.20903](https://arxiv.org/abs/2608.20903)

    本研究实证评估了三种智能体IDE在生成全栈Web应用时的性能，发现它们在处理常见模式时表现成熟，但在生成复杂分布式架构时错误频出，表明它们不能替代开发者，而是将开发者角色转变为通过自然语言编排智能体。

    

    arXiv:2608.20903v1 公告类型：新 摘要：智能体集成开发环境是软件工程领域最重要的创新之一，旨在通过基于LLM的智能体加速应用开发，这些智能体可在开发过程中协助开发者。然而，它们在涉及生成完整应用的端到端开发任务中的评估仍然有限。为填补这一空白，我们提出了一项严格的比较分析，对三种流行的智能体集成开发环境（Copilot、Cursor和Windsurf）在从头生成五个全栈Web应用方面的性能进行了评估。结果显示，在生成成熟模式（如CRUD操作和认证功能）方面表现出高成熟度。相比之下，生成较少见的分布式架构（如任务队列架构）时产生的错误显著增多。总体而言，结果表明智能体集成开发环境无法取代开发者，但将其角色转向通过自然语言编排基于LLM的智能体来构建软件。

    arXiv:2608.20903v1 Announce Type: new  Abstract: Agentic IDEs are among the most significant innovations in software engineering, aiming to accelerate application development through LLM-based agents that can assist developers during development. However, their evaluation in end-to-end development tasks involving the generation of complete applications remains limited. To fill this gap, we propose a rigorous comparative analysis of three popular agentic IDEs (Copilot, Cursor, and Windsurf) in the generation of five full-stack Web applications from scratch.   Results show high maturity in the generation of established patterns, such as CRUD operations and authentication features. In contrast, the generation of less common distributed architectures, such as a task queue architecture, produces significantly more errors. Overall, results show that Agentic IDEs cannot replace developers but shift their role toward building software by orchestrating LLM-based agents through natural-language 
    
[^15]: 超越堆栈追踪：利用大型语言模型对编程错误进行自适应解释

    Beyond the Traceback: Using LLMs for Adaptive Explanations of Programming Errors

    [https://arxiv.org/abs/2608.20896](https://arxiv.org/abs/2608.20896)

    本研究发现，大型语言模型生成的错误解释虽然显著提升了用户的主观感受，但并未改善实际的调试客观表现，且解释风格需根据程序员技能水平进行自适应调整。

    

    arXiv:2608.20896v1 公告类型：新 摘要：编程错误信息对于软件开发至关重要，但新手程序员往往难以解读这些信息。虽然大型语言模型（LLMs）可以将这些错误重写为更清晰的解释，但尚不清楚提高可读性是否能客观改善调试性能，或者解释风格应如何与程序员的技能水平相匹配。我们进行了一项多阶段众包研究（N=103），评估针对技能定制的、由LLM生成的Python错误信息。通过自定义能力评估，我们按技能水平对参与者进行分类，并测试了标准解释器消息与两种LLM生成的风格：务实型（面向行动）和条件型（支架式解释）。我们测量了客观调试指标（修复率、尝试次数、修复时间）和主观感知（可读性、认知负荷、语气）。结果显示，虽然LLM重写的消息显著改善了主观评价，但客观调试性能并未相应提升。

    arXiv:2608.20896v1 Announce Type: new  Abstract: Programming error messages are critical for software development, yet they remain difficult for novice programmers to interpret. While Large Language Models (LLMs) can rewrite these errors into clearer explanations, it remains unclear whether increased readability improves objective debugging performance or how explanation styles should align with programmer skill. We present a multi-stage crowdsourced study N=103 evaluating skill-targeted, LLM-generated Python error messages. Using a custom proficiency assessment, we categorized participants by skill level and tested standard interpreter messages against two LLM-generated styles: pragmatic (action-oriented) and contingent (scaffolded explanations). We measured both objective debugging metrics (fix rate, attempts, time-to-fix) and subjective perceptions (readability, cognitive load, tone). Our results show that while LLM-rewritten messages significantly improved subjective evaluations, w
    
[^16]: BC-Bench：在ERP领域特定语言中评估智能体工程能力

    BC-Bench: Evaluating Agentic Engineering in a Domain-Specific Language for ERP

    [https://arxiv.org/abs/2608.20851](https://arxiv.org/abs/2608.20851)

    BC-Bench是一个针对ERP领域特定语言AL的基准测试，通过101个真实任务评估智能体工程能力，涵盖代码生成、测试生成和多模态问题处理，弥补了通用基准在ERP场景中的不足。

    

    摘要：智能体工程系统在通用基准测试中表现出色，但它们在企业资源规划（ERP）领域特定语言（DSL）中的有效性仍未得到充分探索。我们引入了BC-Bench，一个旨在评估智能体工程在AL（微软Dynamics 365 Business Central的DSL）中真实世界任务表现的基准测试。BC-Bench包含从两个微软自有生产代码库中提取的101个手工整理的任务，反映了真实的ERP开发工作流程。通过改编SWE-Bench方法论，我们解决了AL生态系统的独特约束——包括有限的公共资源和复杂的环境配置。除了生成功能性代码外，BC-Bench还评估测试生成，并支持多模态问题陈述，其中视觉上下文通常存在。我们评估了多个前沿模型在两个智能体框架上的表现，利用多次运行指标来处理非确定性。

    arXiv:2608.20851v1 Announce Type: cross  Abstract: Agentic engineering systems have shown strong performance on general-purpose benchmarks, yet their effectiveness in enterprise resource planning (ERP) domain-specific languages (DSLs) remains underexplored. We introduce BC-Bench, a benchmark designed to evaluate agentic engineering on real-world tasks in AL, the DSL for Microsoft Dynamics 365 Business Central. BC-Bench comprises 101 manually curated tasks extracted from two Microsoft-owned production repositories, reflecting authentic ERP development workflows. Adapting the SWE-Bench methodology, we address the unique constraints of the AL ecosystem---including limited public resources and complex environment provisioning. Beyond generating functional code, BC-Bench evaluates test generation and supports multimodal problem statements where visual context is commonly present. We evaluate multiple frontier models across two agent harnesses, utilizing multi-run metrics to account for nond
    
[^17]: 代码翻译技术的广泛实证研究

    An Extensive Empirical Study on Code Translation Technique

    [https://arxiv.org/abs/2608.20776](https://arxiv.org/abs/2608.20776)

    本研究通过大规模实证比较发现，LLM和基于LLM的代码翻译方法在方法级正确性上优于传统学习方法，但相似性指标不能可靠预测功能正确性，且翻译方向对性能有显著影响。

    

    arXiv:2608.20776v1 公告类型：新 摘要：自动化代码翻译对软件演化日益重要，然而基于学习的方法和基于大型语言模型（LLM）的方法之间的相对优势和局限性仍未被充分理解。为弥补这一空白，我们开展了一项大规模实证研究，比较了跨方法论范式和翻译粒度的代表性代码翻译技术。我们评估了基于学习的方法、基于LLM的方法以及通用LLM在多语言方法级和类级基准上的表现，这些基准涉及多种编程语言。我们的分析考虑了可执行正确性、代码相似性、翻译方向、翻译粒度和失败模式。结果表明，LLM和基于LLM的方法在方法级正确性上通常优于基于学习的方法，尽管仅凭相似性指标并不能可靠反映功能正确性。翻译方向显著影响性能。

    arXiv:2608.20776v1 Announce Type: new  Abstract: Automated code translation is increasingly important for software evolution, yet the relative strengths and limitations of learning-based and large language model (LLM)-based techniques remain insufficiently understood. To address this gap, we conduct a large-scale empirical study comparing representative code translation techniques across methodological paradigms and translation granularities. We evaluate learning-based methods, LLM-based methods, and general-purpose LLMs on multilingual method-level and class-level benchmarks involving multiple programming languages. Our analysis considers executable correctness, code similarity, translation direction, translation granularity, and failure patterns. The results show that LLMs and LLM-based methods generally outperform learning-based methods in method-level correctness, although similarity metrics alone do not reliably reflect functional correctness. Translation direction substantially a
    
[^18]: 真实软件历史中的时间有效性：消除GitHub修复中代码助手记忆的过时事实错误

    Temporal Validity on Real Software Histories: Eliminating Stale-Fact Errors in Code-Assistant Memory over GitHub Fixes

    [https://arxiv.org/abs/2608.20685](https://arxiv.org/abs/2608.20685)

    本文验证了MemStrata在真实软件历史中通过确定性过时记忆消除RAG的时间盲点，显著提升答案准确率（0.91对比0.57-0.59），并减少过时事实错误。

    

    检索增强生成（RAG）缺乏时间模型：当编码会话中事实发生变化——函数被重命名、端点移动、依赖项升级——RAG会检索到新旧值，且相似度几乎相同，无法判断哪个是当前的，因此会提供已过时的值。论文1在合成单值基准上表明，确定性（主体、关系、对象）的过时记忆消除可以解决此失败。本文在真实软件历史上进行端到端验证。从707个真实GitHub问题（SWE-bench Lite + Verified）中提取130个干净的原子状态转换，即修复将一个可识别值从修复前形式变为修复后形式，并将每个标记去除（过时和当前语句仅值不同）。在此数据集上，MemStrata达到0.91的答案准确率，而RAG为0.57-0.59；并且，结构性结果表明，当被迫回答时，RAG在36-3%的情况下提供过时值。

    arXiv:2608.20685v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) has no model of time: when a fact changes across a coding session - a function is renamed, an endpoint moves, a dependency is bumped - RAG retrieves both the old and new value with near-identical similarity and cannot tell which is current, so it serves the superseded value. Paper 1 showed, on synthetic single-value benchmarks, that a deterministic (subject, relation, object) supersession memory eliminates this failure. Here we validate it end-to-end on real software history. From 707 real GitHub issues (SWE-bench Lite + Verified) we extract 130 clean atomic state transitions, a fix that changes one identifiable value from a pre-fix to a post-fix form, and render each marker-free (the stale and current statements differ only in the value). On this set, MemStrata reaches 0.91 answer accuracy versus RAG's 0.57-0.59; and, the structural result, when forced to answer RAG serves the superseded value 36-3
    
[^19]: 软件供应链作为柠檬市场：信任信号崩塌的多声部综述

    The Software Supply Chain as a Market for Lemons: A Multivocal Review of Trust Signal Collapse

    [https://arxiv.org/abs/2608.20678](https://arxiv.org/abs/2608.20678)

    本研究通过多声部综述揭示，开源依赖采纳中的廉价信任信号在对抗性操纵、不可区分的游戏化和AI驱动膨胀的三重压力下系统性崩塌，生态系统应对仍以建议为主。

    

    arXiv:2608.20678v1 公告类型：交叉  摘要：评估开源依赖项的从业者依赖于廉价的信任信号，例如星标数、下载量和贡献者活动，作为直接代码检查的替代品，假设这些信号反映了真正的可信度。先前的研究记录了单个信号的游戏化操纵，但所有依赖采纳信号的整体崩塌景观以及生态系统的响应仍未探索。本研究的目标是通过对252个Google搜索来源和870个Reddit帖子的多声部综述，帮助软件从业者理解依赖采纳信任信号（如下载量和贡献者活动）的可靠性。在对语料库进行编码后，我们发现廉价信任信号在三种同时作用的力量下崩塌：对抗性操纵、与合法行为难以区分的游戏化技术，以及非对抗性AI驱动的膨胀。记录在案的响应更多是建议。

    arXiv:2608.20678v1 Announce Type: cross  Abstract: Practitioners evaluating open-source dependencies rely on cheap trust signals, e.g., stars, download counts, and contributor activity, as substitutes for direct code inspection, assuming those signals reflect genuine trustworthiness. Prior work has documented individual signal gaming, but the landscape of collapses across all dependency-adoption signals, as well as the ecosystem's response, remains unexplored. The goal of this study is to aid software practitioners in understanding the reliability of dependency adoption trust signals, such as download counts and contributor activity, by conducting a multivocal review of 252 Google Search sources and 870 Reddit threads. After coding the corpora, we find that cheap trust signals collapse under three simultaneous forces: adversarial manipulation, gaming techniques indistinguishable from legitimate behavior, and non-adversarial AI-driven inflation. The documented responses are more advice 
    
[^20]: DreamBench-SWE：面向软件代理的多会话记忆卫生基准

    DreamBench-SWE: A Multi-Session Memory-Hygiene Benchmark for Software Agents

    [https://arxiv.org/abs/2608.20664](https://arxiv.org/abs/2608.20664)

    该论文提出了DreamBench-SWE基准，用于评估软件代理在多会话中的记忆卫生，并通过实验表明外部记忆机制的性能差异显著，但未达到等效性证据。

    

    arXiv:2608.20664v1 公告类型：新 摘要：DreamBench-SWE是一个用于软件代理记忆卫生的多会话基准，其中后续软件任务依赖于早期会话中无法推断的证据，并通过可执行的隐藏预言机进行评分。我们报告了原始缩放版v2折叠，以及一个单独预注册的v2.1后续审计，该审计在该研究之后设计，但在后续结果检查之前冻结。后续运行完成了四个条件下的360/360个工作单元和720/720个S3单元。在原始折叠中，主要DF-hybrid--B5对比为无效（95/180对89/180；聚类p=.518，Holm p=1），这不是等效性的证据，且C9/C10保留了B0-头空间限制。在后续中，无外部记忆达到21/180次通过（率0.1167），确定性逐字事件记忆82/180次（率0.4556），类型加原始参考探针83/180次（率0.4611），以及一个固定的托管Mem0字面存储配置97/180次（率0.5389）。注册的六槽Fa...

    arXiv:2608.20664v1 Announce Type: new  Abstract: DreamBench-SWE is a multi-session benchmark for software-agent memory hygiene in which later software tasks depend on non-inferable evidence from earlier sessions and are scored by executable hidden oracles. We report the original scaled v2 fold and a separately preregistered v2.1 successor audit designed after that study but frozen before successor outcome inspection. The successor run completed 360/360 work units and 720/720 S3 cells across four conditions. In the original fold, the primary DF-hybrid--B5 contrast was null (95/180 versus 89/180; clustered p=.518, Holm p=1), not evidence of equivalence, and C9/C10 retained B0-headroom limitations. In the successor, no external memory achieved 21/180 passes (rate 0.1167), deterministic verbatim event memory 82/180 (rate 0.4556), the typed-plus-raw reference probe 83/180 (rate 0.4611), and one pinned hosted Mem0 literal-storage configuration 97/180 (rate 0.5389). The registered six-slot Fa
    
[^21]: 理解操作系统缺陷的研究

    Toward Understanding Operating System Defects

    [https://arxiv.org/abs/2608.20643](https://arxiv.org/abs/2608.20643)

    这项研究首次对1500个来自Android、Linux和HarmonyOS操作系统的缺陷进行大规模分析，揭示了缺陷分布的关键特征，为操作系统缺陷检测和调试方法的设计提供了基础性指导。

    

    摘要：arXiv:2608.20643v1 公告类型：新 摘要：操作系统（OS）是所有其他软件系统的基础，因此操作系统中的缺陷可能导致严重后果，如系统崩溃和数据损坏，影响数十亿用户。这种广泛影响凸显了确保操作系统质量的必要性和重要性。理解操作系统缺陷的特征是这一质量保证任务的基本步骤，因为它有助于设计有效的缺陷检测和调试方法。在这项工作中，我们对来自三个不同且具有代表性的操作系统（Android、Linux 和 HarmonyOS）的 1,500 个缺陷进行了大规模研究，涵盖移动和桌面环境。据我们所知，这是该领域规模最大的研究。通过分析操作系统缺陷在多个分类维度上的分布，包括缺陷发生的操作系统层、它们影响的功能、它们如何被触发、它们的严重性等。

    arXiv:2608.20643v1 Announce Type: new  Abstract: Operating systems (OS) serve as the foundation for all other software systems, and thus defects in OSes can lead to severe conquences, such as system crashes and data corruption, affecting billions of users. This broad impact underscores the necessity and importance of ensuring OS quality. Understanding the characteristics of OS defects is a fundamental step in this quality assurance task, as it facilitates the design of effective defect detection and debugging approaches. In this work, we conduct a large-scale study of 1,500 defects from three distinct and representative operating systems (Android, Linux, and HarmonyOS) spanning both mobile and desktop environments. To the best of our knowledge, this is the largest study of its kind in this domain. By analyzing the distribution of OS defects across multiple classification dimensions, including the OS layer where defects occur, the functions they affect, how they are triggered, their sev
    
[^22]: 行为规范引导的二进制去混淆程序合成

    Behavior Specification-Guided Program Synthesis for Binary Deobfuscation

    [https://arxiv.org/abs/2608.20628](https://arxiv.org/abs/2608.20628)

    本文提出一种从结构变换转向行为驱动合成的二进制去混淆新范式，通过行为规范指导程序合成，以克服现有反编译方法在保持运行时行为和代码质量方面的局限性。

    

    去混淆对于逆向工程和安全分析至关重要，因为它能恢复混淆代码的可读性和可分析性。然而，现有研究主要关注源代码级去混淆，而二进制级去混淆尽管在源代码不可用时具有实际重要性，却仍未得到充分探索。现有的二进制去混淆方法通常将二进制代码反编译为伪代码，然后应用结构变换。然而，由于编译过程会丢弃高级语义（如精确的类型信息和源代码级结构），这种基于反编译的范式往往产生低质量的代码，并且对恢复的代码是否保持原始程序的运行时行为提供的保证有限。为了解决这些限制，我们提出了一种从结构变换到行为驱动合成的范式转变。我们的核心见解是，尽管混淆扭曲了程序的结构，但我们可以通过行为规范来指导合成过程，从而生成既保持原始程序行为又具有可读性的代码。

    arXiv:2608.20628v1 Announce Type: new  Abstract: Deobfuscation is critical to reverse engineering and security analysis because it restores the readability and analyzability of obfuscated code. However, existing research primarily focuses on source-code deobfuscation, while binary-level deobfuscation remains largely underexplored despite its practical importance when source code is unavailable. Existing binary deobfuscation methods typically decompile binaries into pseudocode and then apply structural transformations. However, because compilation discards high-level semantics such as precise type information and source-level structures, this decompilation-based paradigm often produces low-quality code and provides limited assurance that the recovered code preserves the runtime behavior of the original program. To address these limitations, we propose a paradigm shift from structural transformation to behavior-driven synthesis. Our core insight is that although obfuscation distorts a pr
    
[^23]: 在大型企业中应用Anthropic原语：知识工作的驾驭范式

    Applying Anthropic Primitives at Large Enterprises: Harness Paradigm for Knowledge Work

    [https://arxiv.org/abs/2608.20622](https://arxiv.org/abs/2608.20622)

    本文提出在企业知识工作中采用“驾驭范式”（harness paradigm）作为第三种方案，以同时避免定制代码维护成本高和集中治理范围有限的问题，并基于近期研究证明其在任务层面优于复杂架构。

    

    arXiv:2608.20622v1 公告类型：新 摘要：前沿模型已经大幅降低了编写自定义代码的成本：专家在自己领域遇到的一个小众问题，现在只需一个下午就能解决。但审查和维护这些代码的成本并未下降。每个解决方案都彼此偏离；理解一个方案意味着从头阅读其代码库。大型企业转而构建集中治理的方案：最差是现成的产品，最好是为每个用例定制的图编排框架，或作为编排器的低代码平台。这些方案每次都是定制的，且范围有限。企业没有考虑第三种选择，它能同时摆脱这两种限制：驾驭范式。最近的研究将编码代理驾驭视为企业基础设施而非编码工具，得出了三个结论：驾驭范式在任务层面足够胜任，并且在企业工作中优于更复杂的架构（arXiv:2604.00073, arXiv:2604.13107）；驾驭选择占...

    arXiv:2608.20622v1 Announce Type: new  Abstract: Frontier models have collapsed the cost of writing custom code: a niche problem a specialist sees in their own domain now costs an afternoon. The cost of reviewing and maintaining that code hasn't collapsed. Each solution drifts from the next; understanding one means reading its codebase from scratch. Large enterprises build something centrally governed instead: at worst an off-the-shelf product, at best a graph-orchestration framework wired bespoke per use case, or a low-code platform used as the orchestrator. These are custom every time and limited in scope. Enterprises don't weigh a third option that escapes both constraints: the harness paradigm.   Recent work treats the coding-agent harness as enterprise infrastructure rather than a coding tool, converging on three findings: harnesses suffice at the task level and outperform more elaborate architectures on enterprise work (arXiv:2604.00073, arXiv:2604.13107); harness choice accounts
    
[^24]: 军事指挥控制中智能体AI系统的测试与评估

    Testing and Evaluation of Agentic AI Systems In Military Command and Control

    [https://arxiv.org/abs/2608.20597](https://arxiv.org/abs/2608.20597)

    本文通过审查240项测试与评估实践，发现智能体AI系统的特性削弱了传统测试方法中的八项关键假设，导致现有测试结果无法有效支持军事指挥控制系统的安全性承诺。

    

    arXiv:2608.20597v1 公告类型：交叉 摘要：智能体AI系统正在被采购用于军事指挥控制（C2），并附有严格测试和人类监督的公开承诺。这些承诺能否兑现取决于其支持性保证案例，该案例需要三个要素：规定可接受性条件的声明、与这些声明相关的证据，以及连接两者的论证。通过对240项记录的测试与评估（T&E）实践进行结构化审查，涵盖八个评估维度和三个生命周期阶段，我们识别出成熟方法对其测试对象所做的八项假设，这些假设分为四类：系统可指定性、稳定性、可组合性和可监督性。智能体特性削弱了所有八项假设。这种削弱影响了连接证据与声明的论证，而非声明或证据本身。因此，测试结果可能满足流程要求，但并不能证明其有效性。

    arXiv:2608.20597v1 Announce Type: cross  Abstract: Agentic AI systems are being procured for military command and control (C2) under public commitments to rigorous testing and human oversight. Whether such commitments can be discharged depends on their supporting assurance case, which requires three elements: claims specifying the conditions for acceptability, evidence bearing on those claims, and an argument connecting the two. Through a structured review of 240 documented Testing and Evaluation (T&E) practices, spanning eight evaluation dimensions and three lifecycle stages, we identify eight assumptions that established methods make about their test article, grouped into four clusters: system specifiability, stability, composability, and supervisability. Agentic properties weaken all eight assumptions. This erosion affects the argument connecting evidence to claims, not the claims or evidence themselves. As a result, test results may satisfy process requirements, but they do not war
    
[^25]: FlavourBench：用可执行的烹饪真实数据对前沿语言模型进行排名

    FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth

    [https://arxiv.org/abs/2608.20574](https://arxiv.org/abs/2608.20574)

    该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。

    

    开放式语言模型基准测试通常继承一个评判者：人类偏好小组、另一个模型，或脆弱的精确匹配键。我们引入了FlavourBench，一个自动化基准测试，其中版本化的烹饪系统提供密集、可执行的真实数据。每个任务呈现八种食材，并要求选择三种食材的组合；在模型执行前，Epicure对所有56种可能的组合进行评分。我们在一个包含534个任务的相同核心集上评估了27个前沿端点，涵盖替代、配对和受限组合。每个排名的模型在每个面板和家族中恰好有89个有效响应（总共14,418个模型-任务单元），消除了排行榜上的差异性缺失。FlavourBench分数是冻结任务分数的等家族均值。我们使用50,000个锚点聚类自助重采样进行同时95%分数区间，以及100,000次符号翻转抽样进行所有351个配对模型对比，并采用Holm校正。两个独立的...

    arXiv:2608.20574v1 Announce Type: new  Abstract: Open-ended language-model benchmarks usually inherit a judge: a human preference panel, another model, or a brittle exact-match key. We introduce FlavourBench, an automated benchmark in which a versioned culinary system supplies dense, executable ground truth. Each task presents eight ingredients and asks for a three-ingredient portfolio; before model execution, Epicure scores all 56 possible portfolios. We evaluate 27 frontier endpoints on an identical 534-task core spanning substitution, pairing, and constrained composition. Every ranked model has exactly 89 valid responses per panel and family (14,418 model-task cells total), eliminating differential missingness from the leaderboard. The FlavourBench Score is the equal-family mean of the frozen task scores. We use 50,000 anchor-cluster bootstrap replicates for simultaneous 95% score bands and 100,000 sign-flip draws for all 351 paired model contrasts, with Holm control. The two indepe
    
[^26]: AutoMOOSE：代理式相场模拟软件的用例与逻辑视图

    AutoMOOSE: Use Case and Logical Views of Agentic Phase-Field Simulation Software

    [https://arxiv.org/abs/2608.20571](https://arxiv.org/abs/2608.20571)

    AutoMOOSE通过六代理流水线和物理证伪/自动修复分离机制，将自然语言请求转化为可检查且可重用的MOOSE相场模拟，适用于多物理场材料设计。

    

    arXiv:2608.20571v1 公告类型：交叉 摘要：AutoMOOSE是一个代理式软件框架，能将自然语言请求转换为已执行、筛选并解释的MOOSE相场模拟。在此，我们将AutoMOOSE部署为代理式软件，以补充我们先前专注于代理工具开发的工作。我们通过1+5架构视图模型中的用例视图和逻辑视图来描述我们的软件框架和架构，涵盖其用户角色、组件结构、六代理流水线、物理插件层、模型上下文协议接口以及筛选/证伪/恢复循环。我们的架构将物理证伪与自动修复分离，因此修正后的模拟保持可检查性，并且必须在接受前重新准入。我们专注于AutoMOOSE框架的软件设计、可扩展性、互操作性和重用性，以广泛应用于多物理场材料设计问题。

    arXiv:2608.20571v1 Announce Type: cross  Abstract: AutoMOOSE is an agentic software framework that converts a natural-language request into an executed, screened, and interpreted MOOSE phase-field simulation. Here, we deploy AutoMOOSE as a agentic software, complementing our prior work which focused on development of the agentic tool. We describe our software framework and architecture through Use Case and logical views of the 1+5 architectural-views model, covering its user roles, component structure, six-agent pipeline, physics plugin layer, Model Context Protocol interface, and screening/falsification/recovery loop. Our architecture separates physical falsification from automatic repair, so corrected simulations remain inspectable and must be re-admitted before acceptance. We focus on software design, extensibility, interoperability, and reuse of the AutoMoose framework for broad utilization in multiphysics materials design problems.
    
[^27]: Meta的安全部署实践：持续变更安全的健康检查机制

    Making Deployments Safe at Meta: Health Checks for Continuous Change-Safety

    [https://arxiv.org/abs/2608.20513](https://arxiv.org/abs/2608.20513)

    本文介绍了Meta的Service Health Checker系统，通过模板化健康检查和自动回滚机制，平衡大规模持续部署的发布速度与可靠性，并解决了规模化中的操作挑战。

    

    大规模生产系统的持续部署在发布速度和可靠性之间产生了张力。每一次变更都可能引发可靠性事故，而每一次延迟则意味着错失机会。本文描述了Meta用于在数千个异构服务中缓解这一张力的部署时健康检查基础设施。我们总结了这一基于预防的分布式系统服务——Service Health Checker的架构，解释了检查作者如何组合模板化指标查询、阈值和工作流谓词，并讨论了该系统如何与分层和分阶段发布集成，以便回归触发自动回滚。随后，我们描述了在规模化过程中出现的操作问题，如噪声、警报疲劳、漂移和未覆盖的回归，以及我们为应对这些问题而部署的度量、工具和改进默认值的方案。最后，我们总结了从中获得的经验教训。

    arXiv:2608.20513v1 Announce Type: cross  Abstract: Continuous deployment to large scale production systems creates a tension between release velocity and reliability. Every change is a potential reliability incident, yet every delay is a missed opportunity. This paper describes the deployment time health check infrastructure that Meta uses to mediate this tension across thousands of heterogeneous services. We summarize the architecture of this prevention based distributed system's service called Service Health Checker, explain how check authors compose templated metric queries, thresholds, and workflow predicates; and discuss how the system is integrated with tiered and phased rollouts so that regressions trigger automatic rollback. We then describe the operational problems that emerged at scale, such as noise, alert fatigue, drift, and uncovered regressions, and the program of measurement, tooling, and improved defaults we deployed to address them. We close with lessons learned from y
    
[^28]: 终端代理：命令行环境中AI代理的综述

    Terminal Agents: A Survey of AI Agents in Command-Line Environments

    [https://arxiv.org/abs/2608.20485](https://arxiv.org/abs/2608.20485)

    本文首次以终端介导执行为核心视角，系统梳理了命令行环境中AI代理的架构、能力与评估，提出七维终端能力框架，并指出当前评估过度关注最终结果而忽视过程质量与恢复机制。

    

    大型语言模型代理日益通过终端进行操作，然而现有综述将终端介导的行为分散在软件工程、工具使用和计算机使用研究中。我们将终端代理视为其主导进展性动作-观察循环由终端命令执行、文本反馈和有状态环境交互介导的系统。以终端介导执行为组织视角，本综述确立了工作负载级别的边界，并通过七维终端能力概况连接系统架构、能力获取和评估。我们的综合表明，实际行为由模型、接口、框架、运行时和环境共同塑造。可执行轨迹将学习基于行动后果、验证和恢复，而主流评估强调最终结果，并暴露过程质量、恢复和治理方面的不均衡性。

    arXiv:2608.20485v1 Announce Type: new  Abstract: Large language model agents increasingly act through terminals, yet existing surveys disperse terminal-mediated behavior across software engineering, tool use, and computer-use research. We regard terminal agents as systems whose dominant progress-bearing action--observation loop is mediated by terminal command execution, textual feedback, and stateful environment interaction. Using terminal-mediated execution as an organizing lens, this survey establishes workload-level boundaries and connects system architecture, competence acquisition, and evaluation through a seven-dimensional terminal competence profile. Our synthesis shows that realized behavior is jointly shaped by the model, interface, harness, runtime, and environment. Executable trajectories ground learning in action consequences, verification, and recovery, whereas prevailing evaluations emphasize final outcomes and expose process quality, recovery, and governance unevenly. Bo
    
[^29]: 氛围编程：实践、性能、生产力与风险——一项最新综述

    Vibe Coding: Practice, Performance, Productivity, and Risk -A State-of-the-Art Review

    [https://arxiv.org/abs/2608.20446](https://arxiv.org/abs/2608.20446)

    本文首次系统综述了“氛围编程”的实证证据，揭示了其任务级能力不均（代码生成可靠但故障检测弱）和生产力证据矛盾（实验显示+26%但随机试验显示-19%）的关键现状。

    

    摘要：arXiv:2608.20446v1 公告类型：新 摘要：氛围编程——一种AI辅助软件开发方式，开发者用自然语言描述意图，并通过运行而非阅读生成的代码来验证结果——由Andrej Karpathy于2025年2月命名，并在十七个月内产生了首批实证证据。这项最新综述汇集了跨学科语料库中的证据，涵盖软件工程、人机交互、劳动经济学、安全研究、治理和教育领域。我们调查了模型格局、工具生态系统以及按任务类型划分的性能记录，发现早期基准已饱和，但任务级能力不均：可靠的代码生成伴随着薄弱的故障检测和难以审计的文档。生产力记录起初相互矛盾：同行评审的实地实验报告每周任务增加26%，独立随机试验测得19%的放缓，而团队层面的[记录]

    arXiv:2608.20446v1 Announce Type: new  Abstract: Vibe coding - AI-assisted software development in which the developer describes intent in natural language and validates results by running rather than reading the generated code - was named by Andrej Karpathy in February 2025 and produced its first body of empirical evidence within seventeen months. This state-of-the-art review assembles that evidence across a cross-disciplinary corpus spanning software engineering, human-computer interaction, labour economics, security research, governance, and education. We survey the model landscape, the tool ecosystem, and the performance record by task type, finding the early benchmarks saturated but task-level capability uneven: reliable code generation alongside weak fault detection and hard-to-audit documentation. The productivity record is at first contradictory: peer-reviewed field experiments report +26% more tasks per week, independent randomised trials measure a 19% slowdown, and team-level
    
[^30]: SDAD：面向AI原生软件开发生命周期的规格驱动智能体开发

    SDAD: Spec-Driven Agentic Development for the AI-Native SDLC

    [https://arxiv.org/abs/2608.20341](https://arxiv.org/abs/2608.20341)

    本文提出SDAD框架，将规格驱动开发与智能体技术结合，实现从意图到代码的自主交付，并对比了传统敏捷与未来智能体驱动的SDLC范式转变。

    

    arXiv:2608.20341v1 公告类型：新公告  摘要：由大型语言模型支持的前沿编码智能体，其上下文窗口从数十万到数百万个令牌不等，正在重构软件开发生命周期（SDLC）。丰富的上下文处理和多步推理现在允许在单个工作流中摄入大量的功能需求文档（FRD）和仓库上下文，使得规格质量成为自主交付的执行燃料。本报告将规格驱动智能体开发（SDAD）形式化为纪律性前期形式化与高速实现之间的综合：意图捕获、机器可读规格、智能体合成，以及在人工签署下的独立多智能体验证。我们重新审视了瀑布模型与敏捷模型之间的历史摇摆，将AI代码引入为第四种生产范式，并比较了人类敏捷（约2020年）与智能体SDAD（约2026年）在工件、节奏、责任和安全方面的差异。

    arXiv:2608.20341v1 Announce Type: new  Abstract: Frontier coding agents backed by large language models with context windows from hundreds of thousands to millions of tokens are restructuring the Software Development Life Cycle (SDLC). Rich context handling and multi-step reasoning now allow substantial Functional Requirement Documents (FRDs) and repository context to be ingested in a single workflow, making specification quality the execution fuel for autonomous delivery. This report formalises Spec-Driven Agentic Development (SDAD) as a synthesis of disciplined up-front formalisation and high-velocity implementation: intent capture, machine-readable specification, agentic synthesis, and independent multi-agent verification under human sign-off. We revisit the historical pendulum between Waterfall and Agile, introduce AI-code as a fourth production paradigm, and compare Human-Agile (circa 2020) with Agentic-SDAD (circa 2026) across artefacts, cadence, accountability, and security post
    
[^31]: 内部稳定，外部不确定：基准效应与排名的语义识别

    Stable Within, Unidentified Across: Semantic Identification of Benchmark Effects and Rankings

    [https://arxiv.org/abs/2608.19269](https://arxiv.org/abs/2608.19269)

    该论文通过局部非蕴含实例表明，基准效应和排名的识别依赖于评估者控制的语义家族，不同语义选择可能导致截然不同的结论。

    

    评估结论取决于评估者控制的语义：法律引用、可评分性和聚合方式。我们称一个由工件定义的端点Q(s)在声明的家族内不变时，为评估语义可识别。一个冻结的217行分析在其受限的合同家族内显得稳定。在TraceElephant中，F_asym产生了从-40.76到-43.03个百分点的精确任务分离采用效应，而F_cc中两个同时冻结的对比可比处理恰好产生零效应，且区间为[0, 0]。因此，它们的并集F_audit无法识别T，尽管F_cc识别了T=0。一项因子审计将全部分歧归因于单侧终端删除，非终端贡献为零。排名在对称处理间仍依赖于规格。这是一个局部非蕴含见证和可复用的家族索引审计，而非普遍性估计或详尽语义分析。

    arXiv:2608.19269v1 Announce Type: new  Abstract: Evaluation conclusions depend on evaluator-controlled semantics: legal references, scoreability, and aggregation. We call an artifact-defined endpoint Q(s) evaluation-semantically identified when it is invariant over a declared family. A frozen 217-row analysis appears stable within its restricted contract family. In TraceElephant, F_asym yields precise task-disjoint adoption effects from -40.76 to -43.03 percentage points, whereas both jointly frozen contrast-comparable treatments in F_cc yield exactly zero with [0, 0] intervals. Their union F_audit therefore does not identify T, even though F_cc identifies T = 0. A factorial audit attributes 100 percent of the disagreement to one-sided terminal deletion, with zero nonterminal contribution. Rankings remain specification-dependent across the symmetric treatments. This is a localized non-implication witness and reusable family-indexed audit, not a prevalence estimate, exhaustive semantic 
    
[^32]: LongRCA Bench：诊断长时程智能体失败中的责任角色与根本原因

    LongRCA Bench: Diagnosing Responsible Roles and Root Causes in Long-Horizon Agent Failures

    [https://arxiv.org/abs/2608.15242](https://arxiv.org/abs/2608.15242)

    本文提出LongRCA Bench，一个包含1,140条长时程失败轨迹的基准，用于诊断智能体失败中的责任角色和最早根本原因步骤，并引入无需训练的RCTA方法，以提升诊断精度。

    

    arXiv:2608.15242v1 公告类型：新 摘要：当长时程智能体执行失败时，结果级评估仅显示失败结果，但未揭示决定性错误在轨迹中的何处引入。开发者必须检查整个执行过程，以识别责任角色并定位最早的决定性根本原因步骤。现有的失败归因基准主要集中于较短的轨迹，导致对跨数百个记录步骤的诊断仍未充分探索。我们引入了LongRCA Bench，包含五个领域中1,140条失败轨迹，且未注入人为错误。该基准为责任角色和最早决定性根本原因步骤提供了独立评分的人工标签。轨迹中位数包含145个步骤，最强基线仅达到13.2%的精确根本原因步骤准确率。我们进一步提出了根因轨迹归因（RCTA），一种无需训练的方法，从段摘要中检索候选错误步骤，并将其追溯到可用的早期交接点。

    arXiv:2608.15242v1 Announce Type: new  Abstract: When a long-horizon agent execution fails, outcome-level evaluation reveals the unsuccessful result but not where the decisive error entered the trajectory. Developers must then inspect the full execution to identify the responsible role and localize the earliest decisive root-cause step. Existing failure-attribution benchmarks largely focus on shorter traces, leaving diagnosis across hundreds of recorded steps underexplored. We introduce LongRCA Bench, comprising 1,140 failed trajectories across five domains without injected errors. It provides independently scored human labels for the responsible role and earliest decisive root-cause step. The median trajectory contains 145 steps, and the strongest baseline reaches only 13.2% exact root-step accuracy. We further present Root-Cause Trajectory Attribution (RCTA), a training-free method that retrieves candidate error steps from segment summaries and traces them to available earlier handof
    
[^33]: 技能软件：一种用于持久行为工件的软件本体论与工程生命周期

    Skillware: A Software Ontology and Engineering Lifecycle for Persistent Behavioral Artifacts

    [https://arxiv.org/abs/2607.18970](https://arxiv.org/abs/2607.18970)

    本文提出了技能软件（Skillware）这一软件抽象，将软件工程原则应用于AI代理中的持久行为工件，通过定义技能工件、技能软件单元和代理主机执行关系来规范其生命周期。

    

    arXiv:2607.18970v3 公告类型：替换交叉 摘要：代理技能已成为独立AI代理系统中的持久行为工件。它们结合了自然语言任务规范与元数据，以及可选的引用、脚本、资产、钩子、包清单、测试和配套接口。现有研究解释了技能如何被指定、执行、维护和演化，但缺乏一种本体论来将这些工件定义为独立的软件对象。本文引入了技能软件（Skillware）作为一种软件抽象，将软件工程扩展到代理系统中的持久行为工件。技能工件（Skill Artifact）指定可重用的任务行为；技能软件单元（Skillware Unit）通过独立的身份和生命周期将该工件作为软件进行管理。兼容的代理主机（Agent Host）激活该单元以进行运行时解释。三个必要条件操作化类别成员资格：行为优先性、独立软件身份和代理主机执行关系。

    arXiv:2607.18970v3 Announce Type: replace-cross  Abstract: Agent Skills have become persistent behavioral artifacts across independent AI agent systems. They combine natural-language task specifications with metadata and optional references, scripts, assets, hooks, package manifests, tests, and companion interfaces. Existing studies explain how Skills are specified, executed, maintained, and evolved, but lack an ontology that defines these artifacts as independent software objects. This paper introduces Skillware as the software abstraction that extends software engineering to persistent Behavioral Artifacts in agent systems. A Skill Artifact specifies reusable task behavior; a Skillware Unit manages that artifact as software through an independent identity and lifecycle. A compatible Agent Host activates the unit for runtime interpretation. Three necessary conditions operationalize category membership: behavioral primacy, independent software identity, and an Agent Host execution rela
    
[^34]: 量子输运代码的RSE实践及其影响

    RSE of a Quantum Transport Code and its Effects

    [https://arxiv.org/abs/2605.21334](https://arxiv.org/abs/2605.21334)

    本文通过两年RSE实践，展示了持续集成和基准测试如何揭示量子输运代码中的关键缺陷，包括未初始化内存、越界写入及数学建模错误，并指出Fortran科学代码中危险缺陷的普遍性。

    

    本文介绍了我们过去两年在libNEGF（一款量子输运代码）上的研究软件工程（RSE）经验。我们描述了代码质量保证的实用方法——包括持续集成、自动化测试和编译器警告修正——以及通过持续基准测试进行的性能工程。我们系统性地应用这些实践，揭示了关键缺陷：未初始化内存读取、越界写入，以及值得注意的是，边界条件处理中一个被误解的数学模型。我们还记录了持续基准测试如何暴露由HPC系统配置变化引起的性能回退。我们的发现提供了数据点，表明一类危险的缺陷——等同于C/C++中的未定义行为和Fortran中的处理器依赖行为——在Fortran科学代码中与其他地方一样普遍存在。尽管libNEGF是用Fortran实现的，但大多数建议具有普适性。

    arXiv:2605.21334v2 Announce Type: replace  Abstract: This paper presents our research software engineering (RSE) experiences over two years with libNEGF, a quantum transport code. We describe practical approaches to code quality assurance--including continuous integration, automated testing, and compiler warning correction--and performance engineering through continuous benchmarking. Our systematic application of these practices revealed critical defects: uninitialized memory reads, out-of-bounds writes, and notably, a misunderstood mathematical model in our boundary condition handling. We also document how continuous benchmarking exposed performance regressions caused by HPC system configuration changes. Our findings provide data points suggesting that a dangerous class of defects--equivalent to undefined behavior in C/C++ and processor-dependent behavior in Fortran--is as prevalent in Fortran scientific codes as elsewhere. While libNEGF is implemented in Fortran, most recommendations
    
[^35]: MalSkills：通过神经符号推理检测代理供应链中的恶意技能

    MalSkills: Detecting Malicious Skills in the Agentic Supply Chain via Neuro-symbolic Reasoning

    [https://arxiv.org/abs/2603.27204](https://arxiv.org/abs/2603.27204)

    本文提出了MalSkills，一个神经符号框架，通过结合符号解析和LLM辅助语义分析，从异构工件中提取安全敏感操作并构建技能依赖图，从而有效检测代理供应链中的恶意技能。

    

    arXiv:2603.27204v2 公告类型：替换交叉 摘要：技能通过将提示、代码和配置打包成可复用模块，越来越多地被用于扩展LLM代理。随着公共注册表和市场的扩展，它们形成了一个新兴的代理供应链，但也引入了恶意技能的新攻击面。检测恶意技能具有挑战性，因为相关证据通常分布在异构工件中，并且必须在上下文中进行推理。现有的静态、基于LLM和动态方法各自只捕捉了这一问题的一部分，使其不足以进行稳健的现实世界检测。在本文中，我们提出了MalSkills，一个用于恶意技能检测的神经符号框架。MalSkills首先通过符号解析和LLM辅助语义分析的组合，从异构工件中提取安全敏感操作。然后，它构建技能依赖图，连接工件、操作、操作数和跨这些元素的值流。

    arXiv:2603.27204v2 Announce Type: replace-cross  Abstract: Skills are increasingly used to extend LLM agents by packaging prompts, code, and configurations into reusable modules. As public registries and marketplaces expand, they form an emerging agentic supply chain, but also introduce a new attack surface for malicious skills. Detecting malicious skills is challenging because relevant evidence is often distributed across heterogeneous artifacts and must be reasoned in context. Existing static, LLM-based, and dynamic approaches each capture only part of this problem, making them insufficient for robust real-world detection. In this paper, we present MalSkills, a neuro-symbolic framework for malicious skills detection. MalSkills first extracts security-sensitive operations from heterogeneous artifacts through a combination of symbolic parsing and LLM-assisted semantic analysis. It then constructs the skill dependency graph that links artifacts, operations, operands, and value flows acr
    
[^36]: 我的README文件需要更新吗？探索基于大语言模型的README维护

    Does My README File Need To Be Updated? Exploring LLM-Based README Maintenance

    [https://arxiv.org/abs/2603.00489](https://arxiv.org/abs/2603.00489)

    该论文提出了一种基于大语言模型的框架，用于自动推荐开源项目中README文件的精准更新，在人机协同流程中判断更新需求、定位修改位置并解释触发原因，在25,511个拉取请求上实现了一半恢复率和28%的用户端准确率。

    

    arXiv:2603.00489v2 公告类型：替换 摘要：README文件对于理解开源软件和引导贡献者加入至关重要，但它们经常变得过时。我们将精准的文档更新推荐作为一个任务提出，并展示了一个由大语言模型驱动的框架，用于人机协同工作流程。给定一个拉取请求，该框架判断是否需要更新README，确定应在何处进行更改，并解释触发事件。我们在714个热门仓库的25,511个拉取请求上评估了该框架。其最佳配置能够恢复历史上有README更新伴随的拉取请求的一半，并在观察到此类更新的普遍性下实现了28%的用户端准确率。定性失败分析进一步识别了改进机会。我们还对20个抽样仓库进行了回顾性研究，并与一个大型开源项目的开发者进行了案例研究。手动标注...

    arXiv:2603.00489v2 Announce Type: replace  Abstract: README files are critical for understanding and onboarding contributors to open-source software, yet they frequently become outdated. We formulate surgical documentation update recommendation as a task and present a Large Language Model-driven framework for use in a human-in-the-loop workflow. Given a pull request, the framework determines whether a README update is needed, identifies where changes should be made, and explains the triggering events. We evaluate the framework on 25,511 pull requests from 714 popular repositories. Its best configuration recovers half of the pull requests historically accompanied by README updates and achieves 28% user-facing accuracy under the observed prevalence of such updates. A qualitative failure analysis further identifies opportunities for improvement. We also conduct a retrospective study of 20 sampled repositories and a case study with a developer from a large open-source project. Manual annot
    
[^37]: 软件任务中的奖励工程：强化学习方法综述

    Reward Engineering for Software Tasks: A Survey of Reinforcement Learning Approaches

    [https://arxiv.org/abs/2601.19100](https://arxiv.org/abs/2601.19100)

    本文首次系统综述了软件任务中强化学习的奖励工程，按奖励来源、粒度和聚合方式分类，并为实践提供了指导。

    

    arXiv:2601.19100v2 公告类型：替换 摘要：强化学习越来越多地用于以代码为中心的软件工程任务，包括代码生成、理解、修复、测试和优化，尤其是在大型语言模型和自主代理兴起的背景下。这些场景中的核心挑战是奖励设计。与具有明确标量目标的标准强化学习领域不同，软件任务涉及相互竞争的目标，如正确性、安全性、效率和可读性，这些难以用单一奖励来捕捉。因此，用于软件工程的强化学习系统依赖异构信号，包括编译结果、单元测试、覆盖率指标、检索分数和学习到的偏好。然而，这些工作在不同任务和社区中仍然分散。本综述首次对软件任务中强化学习的奖励工程进行了系统性回顾。我们按奖励来源、粒度和聚合方式对先前工作进行组织。然后，我们将研究发现提炼为实用指南。

    arXiv:2601.19100v2 Announce Type: replace  Abstract: Reinforcement learning is increasingly used for code-centric software engineering tasks, including code generation, understanding, repair, testing, and optimization, especially with the rise of large language models and autonomous agents. A core challenge in these settings is reward design. Unlike standard RL domains with clear scalar objectives, software tasks involve competing goals such as correctness, security, efficiency, and readability, which are difficult to capture with a single reward. As a result, RL-for-SE systems rely on heterogeneous signals, including compilation results, unit tests, coverage metrics, retrieval scores, and learned preferences. Yet this work remains scattered across tasks and communities. This survey provides the first systematic review of reward engineering for RL in software tasks. We organize prior work by reward source, granularity, and aggregation. We then distill the findings into practical guidan
    
[^38]: 氛围编程安全吗？真实世界任务中智能体生成代码漏洞的基准评估

    Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks

    [https://arxiv.org/abs/2512.03262](https://arxiv.org/abs/2512.03262)

    该论文提出SUSVIBES基准，评估了12种编码智能体在真实任务中的安全性，发现所有智能体生成代码的安全率极低（最高仅11.8%），且简单安全提示无法有效改善。

    

    arXiv:2512.03262v3 公告类型：交叉替换 摘要：氛围编程是一种新的软件开发范式，在这种范式中，人类工程师提示大型语言模型（LLM）智能体在极少监督下完成复杂的编码任务。尽管氛围编程日益被采用，但生成的代码在生产环境中部署真的安全吗？为了探究这一问题，我们提出了SUSVIBES基准，该基准包含来自真实世界开源项目的186个功能请求软件工程任务，针对这些任务，人类程序员提交了存在漏洞的实现。我们在该基准上评估了12种广泛使用的编码智能体设置，并采用了前沿模型。令人不安的是，所有智能体在软件安全方面表现不佳。尽管来自SWE-Agent与Claude 4 Sonnet的解决方案中57%在功能上正确，但只有11.8%是安全的。进一步实验表明，初步安全策略，例如在功能请求中添加漏洞提示，无法缓解这些问题。

    arXiv:2512.03262v3 Announce Type: replace-cross  Abstract: Vibe coding is a new software development paradigm in which human engineers prompt a large language model (LLM) agent to complete complex coding tasks with little supervision. Although vibe coding is increasingly adopted, is the generated code really safe to deploy in production? To investigate this question, we propose SUSVIBES, a benchmark consisting of 186 feature-request software engineering tasks from real-world open-source projects, for which, human programmers committed vulnerable implementations. We evaluate 12 widely used coding agentic settings with frontier models on the benchmark. Disturbingly, all agents perform poorly in terms of software security. Although 57% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 11.8% are secure. Further experiments demonstrate that preliminary security strategies, such as augmenting the feature request with vulnerability hints, cannot mitigate thes
    
[^39]: 大型语言模型生成代码中的库幻觉：基于开发者查询的风险分析

    Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries

    [https://arxiv.org/abs/2509.22202](https://arxiv.org/abs/2509.22202)

    本研究首次系统分析了开发者查询变化如何触发大型语言模型生成代码中的库幻觉，揭示了不同提示条件下的系统性风险模式。

    

    arXiv:2509.22202v4 公告类型：替换交叉 摘要：大型语言模型（LLMs）在代码生成中现已扮演核心角色，但它们仍会出现幻觉，经常虚构不存在的库。此类库幻觉不仅仅是良性错误：它们可能误导开发者、破坏构建，并使系统面临供应链威胁，如“slopsquatting”（一种恶意软件包抢占攻击）。尽管对这些风险的认识日益增强，但对于库幻觉在现实使用条件下如何表现的理解仍然有限。为填补这一空白，我们首次系统性地研究了用户级提示变化如何影响LLM生成代码中的库幻觉。在七个不同的LLM中，我们分析了库名称幻觉（无效导入）和库成员幻觉（来自有效库的无效调用），考察了现实开发者语言和受控用户错误（包括拼写错误和虚构库或成员）的影响。我们的发现揭示了系统性漏洞。

    arXiv:2509.22202v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) now play a central role in code generation, yet they continue to hallucinate, frequently inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite growing awareness of these risks, there is limited understanding of how library hallucinations manifest under realistic usage conditions. To fill this gap, we present the first systematic study of how user-level prompt variations influence library hallucinations in LLM-generated code. Across seven diverse LLMs, we analyse library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries), examining the effects of realistic developer language and controlled user mistakes, including misspellings and fabricated libraries or members. Our findings expose systemic v
    
[^40]: 不要以貌取“码”：探索大语言模型在代码评估中的偏见

    Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation

    [https://arxiv.org/abs/2505.16222](https://arxiv.org/abs/2505.16222)

    本研究首次系统性地揭示了大语言模型在代码评估中对表面差异（如变量名、注释和格式）存在偏见，并通过多种语言和模型实证证明了这些偏见会影响评估的公平性。

    

    arXiv:2505.16222v2 公告类型：替换 摘要：随着大语言模型（LLMs）作为评估者的使用日益增长，其应用已扩展到代码评估任务，即在不依赖参考实现的情况下评估生成代码的正确性。虽然这提供了可扩展性和灵活性，但也引发了一个关键且未解决的问题：LLM法官能否公平且稳健地评估具有表面差异的语义等价代码？功能正确的代码通常表现出差异——例如变量名、注释或格式的不同——这些差异不应影响其正确性。然而，LLM法官能否可靠地处理这些差异仍不清楚。我们首次全面研究了这一问题，定义了代码评估中六种潜在偏见类型，并揭示了它们对LLM法官的系统性影响。在五种编程语言和多种LLM中，我们实证表明，所有测试的LLM法官都容易受到这些偏见的影响。

    arXiv:2505.16222v2 Announce Type: replace  Abstract: With the growing use of large language models(LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. While this offers scalability and flexibility, it also raises a critical, unresolved question: Can LLM judges fairly and robustly evaluate semantically equivalent code with superficial variations? Functionally correct code often exhibits variations-such as differences in variable names, comments, or formatting-that should not influence its correctness. Yet, whether LLM judges can reliably handle these variations remains unclear. We present the first comprehensive study of this issue, defining six types of potential bias in code evaluation and revealing their systematic impact on LLM judges. Across five programming languages and multiple LLMs, we empirically demonstrate that all tested LLM judges are susceptible to b
    

